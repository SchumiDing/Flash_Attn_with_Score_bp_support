"""
Backward Pass for Attention with Scores Module
Supports gradient computation for both attention output and attention scores
"""

import math
import torch
import triton
import triton.language as tl
try:
    from .flash import maybe_contiguous, get_fwd_config
    from .dropout import philox_cuda_seed_offset
except ImportError:
    # Handle case when running as standalone
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from flash import maybe_contiguous, get_fwd_config
    from dropout import philox_cuda_seed_offset


# --------------------------- Backward Config ---------------------------
def get_bwd_config(B, H, M, N, D, causal):
    """Get backward pass configuration based on GPU capability"""
    if torch.cuda.get_device_capability() == (8, 0):
        if not causal:
            BLOCK_M = 128 if D <= 64 else 64
            BLOCK_N = 64
            num_stages = 2
            num_warps = 4
        else:
            BLOCK_M = 64
            BLOCK_N = 64
            num_stages = 3 if D <= 64 else 2
            num_warps = 4
    elif torch.cuda.get_device_capability() == (8, 6):
        if not causal:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 8
        else:
            if D <= 64:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 64, 64, 2, 4
            else:
                BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 2, 4
    else:
        BLOCK_M, BLOCK_N, num_stages, num_warps = 32, 32, 1, 4
    return (BLOCK_M, BLOCK_N, num_stages, num_warps)


# --------------------------- Backward Kernels ---------------------------
@triton.jit
def _bwd_preprocess(
    Out, DO,
    Delta,
    stride_oz, stride_oh, stride_om, stride_ok,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dz, stride_dh, stride_dm,
    M,
    BLOCK_M: tl.constexpr, D_HEAD: tl.constexpr,
    DIVISIBLE_M: tl.constexpr,
):
    """Preprocess kernel: compute delta = rowsum(o * do)"""
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    Out += off_z * stride_oz + off_h * stride_oh
    DO += off_z * stride_doz + off_h * stride_doh
    Delta += off_z * stride_dz + off_h * stride_dh

    # compute (Out * Dout).sum() for vector interpretation
    off_m = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = tl.arange(0, D_HEAD)

    # load
    o_ptrs = Out + off_m[:, None] * stride_om + off_n[None, :] * stride_ok
    do_ptrs = DO + off_m[:, None] * stride_dom + off_n[None, :] * stride_dok

    if DIVISIBLE_M:
        o = tl.load(o_ptrs).to(tl.float32)
        do = tl.load(do_ptrs).to(tl.float32)
    else:
        mask_m = off_m < M
        o = tl.load(o_ptrs, mask=mask_m[:, None]).to(tl.float32)
        do = tl.load(do_ptrs, mask=mask_m[:, None]).to(tl.float32)

    # compute
    delta = tl.sum(o * do, axis=1)

    # write-back
    d_ptrs = Delta + off_m * stride_dm
    if DIVISIBLE_M:
        tl.store(d_ptrs, delta)
    else:
        tl.store(d_ptrs, delta, mask=mask_m)


@triton.jit
def _bwd_kv_kernel_with_scores(
    Q, K, V, sm_scale, DO, DScores,
    DK, DV,
    L, Delta,
    dropout_p, seed, offset,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dsz, stride_dsh, stride_dsm, stride_dsn,
    stride_dkz, stride_dkh, stride_dkn, stride_dkk,
    stride_dvz, stride_dvh, stride_dvn, stride_dvk,
    Z, H, M, N, P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, IS_DROPOUT: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    """Backward kernel for K and V with scores gradient support"""
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_n = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)
    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    DO += off_z * stride_doz + off_h * stride_doh
    DScores += off_z * stride_dsz + off_h * stride_dsh

    # offset pointers for batch/head
    DK += off_z * stride_dkz + off_h * stride_dkh
    DV += off_z * stride_dvz + off_h * stride_dvh

    # offset pointers for batch/head
    Delta += (off_z * H + off_h) * M
    L += (off_z * H + off_h) * M

    if CAUSAL:
        lo = tl.maximum(start_n * BLOCK_N - P_SEQ, 0)
        lo = (lo // BLOCK_M) * BLOCK_M
    else:
        lo = 0

    offs_m_init = lo + tl.arange(0, BLOCK_M)
    offs_n = start_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_m_base = tl.arange(0, BLOCK_M)
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m_init[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    k_ptrs = K + (offs_n[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    v_ptrs = V + (offs_n[:, None] * stride_vn + offs_k[None, :] * stride_vk)
    do_ptrs = DO + (offs_m_init[:, None] * stride_dom + offs_k[None, :] * stride_dok)
    dscores_ptrs = DScores + (offs_m_init[:, None] * stride_dsm + offs_n[None, :] * stride_dsn)

    dv_ptrs = DV + (offs_n[:, None] * stride_dvn + offs_k[None, :] * stride_dvk)
    dk_ptrs = DK + (offs_n[:, None] * stride_dkn + offs_k[None, :] * stride_dkk)

    # k and v stay in SRAM throughout
    if DIVISIBLE_N:
        v = tl.load(v_ptrs)
        k = tl.load(k_ptrs)
    else:
        mask_n = offs_n < N
        v = tl.load(v_ptrs, mask=mask_n[:, None])
        k = tl.load(k_ptrs, mask=mask_n[:, None])

    # dropout
    if IS_DROPOUT:
        colblock_base = off_z * H * M * N + off_h * M * N + start_n * BLOCK_N
        offs_rng_base = offset + colblock_base
        offs_rng_base += tl.arange(0, BLOCK_M)[:, None] * N
        offs_rng_base += tl.arange(0, BLOCK_N)[None, :]
        rp = 1. / (1. - dropout_p)

    # initialize dk and dv
    dk = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)
    dv = tl.zeros([BLOCK_N, BLOCK_DMODEL], dtype=tl.float32)

    # loop over a col
    for start_m in range(lo, M, BLOCK_M):
        start_m = tl.multiple_of(start_m, BLOCK_M)
        offs_m = start_m + offs_m_base
        causal_mask = (P_SEQ + offs_m[:, None]) >= (offs_n[None, :])

        # load q, do on-chip
        if DIVISIBLE_M:
            q = tl.load(q_ptrs)
        else:
            mask_m = offs_m < M
            valid_mask = mask_m[:, None]
            q = tl.load(q_ptrs, mask=mask_m[:, None])

        # recompute p = softmax(qk * sm_scale, dim=-1)
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k))

        # -- recompute p ---
        if DIVISIBLE_M:
            l = tl.load(L + offs_m)
        else:
            l = tl.load(L + offs_m, mask=mask_m)
        p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)

        if not DIVISIBLE_M:
            p = tl.where(valid_mask, p, 0.0)
        if CAUSAL:
            p = tl.where(causal_mask, p, 0.0)

        # compute dv = dot(p, do)
        if DIVISIBLE_M:
            do = tl.load(do_ptrs)
        else:
            do = tl.load(do_ptrs, mask=mask_m[:, None])

        if IS_DROPOUT:
            offs_rng = offs_rng_base + start_m * N
            pmask = tl.rand(seed, offs_rng, n_rounds=6) > dropout_p
            p_masked = p * pmask
            p_masked = p_masked.to(input_dtype)

        # -- apply dropout --
        if IS_DROPOUT:
            dv += tl.dot(tl.trans(p_masked), do) * rp
        else:
            dv += tl.dot(tl.trans(p).to(input_dtype), do)

        # compute dp = dot(v, do)
        if DIVISIBLE_M:
            delta = tl.load(Delta + offs_m)
        else:
            delta = tl.load(Delta + offs_m, mask=mask_m)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do, tl.trans(v))

        # -- apply dropout --
        if IS_DROPOUT:
            dp *= rp
            dp *= pmask

        # Load dscores gradient (gradient from scores output)
        if DIVISIBLE_M and DIVISIBLE_N:
            dscores = tl.load(dscores_ptrs).to(tl.float32)
        elif DIVISIBLE_M:
            mask_n_local = offs_n < N
            dscores = tl.load(dscores_ptrs, mask=mask_n_local[None, :]).to(tl.float32)
        elif DIVISIBLE_N:
            dscores = tl.load(dscores_ptrs, mask=mask_m[:, None]).to(tl.float32)
        else:
            mask_n_local = offs_n < N
            dscores = tl.load(dscores_ptrs, mask=mask_m[:, None] & mask_n_local[None, :]).to(tl.float32)

        # compute ds = p * (dp - delta[:, None]) + dscores * sm_scale
        # The scores output is s * sm_scale, so dscores contributes dscores * sm_scale to ds
        ds = p * (dp - delta[:, None]) + dscores * sm_scale

        if not DIVISIBLE_M:
            ds = tl.where(valid_mask, ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)
        ds = ds.to(input_dtype)

        # compute dk = dot(ds.T, q)
        dk += tl.dot(tl.trans(ds), q)

        # increment pointers
        q_ptrs += BLOCK_M * stride_qm
        do_ptrs += BLOCK_M * stride_dom
        dscores_ptrs += BLOCK_M * stride_dsm

    dk *= sm_scale
    if DIVISIBLE_N:
        tl.store(dk_ptrs, dk.to(input_dtype))
        tl.store(dv_ptrs, dv.to(input_dtype))
    else:
        tl.store(dk_ptrs, dk.to(input_dtype), mask=mask_n[:, None])
        tl.store(dv_ptrs, dv.to(input_dtype), mask=mask_n[:, None])


@triton.jit
def _bwd_q_kernel_with_scores(
    Q, K, V, sm_scale, DO, DScores,
    DQ,
    L, Delta,
    dropout_p, seed, offset,
    stride_qz, stride_qh, stride_qm, stride_qk,
    stride_kz, stride_kh, stride_kn, stride_kk,
    stride_vz, stride_vh, stride_vn, stride_vk,
    stride_doz, stride_doh, stride_dom, stride_dok,
    stride_dsz, stride_dsh, stride_dsm, stride_dsn,
    stride_dqz, stride_dqh, stride_dqm, stride_dqk,
    Z, H, M, N, P_SEQ,
    num_groups,
    BLOCK_M: tl.constexpr, BLOCK_DMODEL: tl.constexpr, BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr, IS_DROPOUT: tl.constexpr, LARGER_M: tl.constexpr,
    DIVISIBLE_M: tl.constexpr, DIVISIBLE_N: tl.constexpr,
):
    """Backward kernel for Q with scores gradient support"""
    input_dtype = Q.dtype.element_ty
    # -- grid id --
    start_m = tl.program_id(0)
    off_h = tl.program_id(1)
    off_z = tl.program_id(2)

    log2e: tl.constexpr = 1.4426950408889634
    qk_scale = sm_scale * log2e

    # offset pointers for (batch, head)
    off_hk = off_h // num_groups
    Q += off_z * stride_qz + off_h * stride_qh
    K += off_z * stride_kz + off_hk * stride_kh
    V += off_z * stride_vz + off_hk * stride_vh
    DO += off_z * stride_doz + off_h * stride_doh
    DScores += off_z * stride_dsz + off_h * stride_dsh
    Delta += (off_z * H + off_h) * M
    L += (off_z * H + off_h) * M

    # offset pointers for batch/head
    DQ += off_z * stride_dqz + off_h * stride_dqh

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n_base = tl.arange(0, BLOCK_N)
    offs_n_init = offs_n_base
    offs_k = tl.arange(0, BLOCK_DMODEL)

    # initialize pointers to value-like data
    q_ptrs = Q + (offs_m[:, None] * stride_qm + offs_k[None, :] * stride_qk)
    k_ptrs = K + (offs_n_init[:, None] * stride_kn + offs_k[None, :] * stride_kk)
    v_ptrs = V + (offs_n_init[:, None] * stride_vn + offs_k[None, :] * stride_vk)

    dq_ptrs = DQ + (offs_m[:, None] * stride_dqm + offs_k[None, :] * stride_dqk)
    do_ptrs = DO + (offs_m[:, None] * stride_dom + offs_k[None, :] * stride_dok)
    dscores_ptrs = DScores + (offs_m[:, None] * stride_dsm + offs_n_init[None, :] * stride_dsn)

    # pointer to row-wise quantities in value-like data
    d_ptrs = Delta + offs_m
    l_ptrs = L + offs_m

    # load q: it will stay in SRAM throughout
    if DIVISIBLE_M:
        q = tl.load(q_ptrs)
        do = tl.load(do_ptrs)
        delta = tl.load(d_ptrs)
        l = tl.load(l_ptrs)
    else:
        mask_m = offs_m < M
        q = tl.load(q_ptrs, mask=mask_m[:, None])
        do = tl.load(do_ptrs, mask=mask_m[:, None])
        delta = tl.load(d_ptrs, mask=mask_m)
        l = tl.load(l_ptrs, mask=mask_m)

    # initialize dq
    dq = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)

    # loop over k, v and update accumulator
    if CAUSAL:
        hi = tl.minimum(N, P_SEQ + (start_m + 1) * BLOCK_M)
        if LARGER_M:
            hi = tl.maximum(0, hi)
    else:
        hi = N

    # dropout
    if IS_DROPOUT:
        rowblock_base = off_z * H * M * N + off_h * M * N + start_m * BLOCK_M * N
        offs_rng_base = offset + rowblock_base
        offs_rng_base += tl.arange(0, BLOCK_M)[:, None] * N
        offs_rng_base += tl.arange(0, BLOCK_N)[None, :]
        rp = 1. / (1. - dropout_p)
        do *= rp.to(do.dtype)

    # loop over a row
    for start_n in range(0, hi, BLOCK_N):
        offs_n = start_n + offs_n_base

        # load k, v on chip
        if DIVISIBLE_N:
            v = tl.load(v_ptrs)
            k = tl.load(k_ptrs)
        else:
            mask_n = offs_n < N
            v = tl.load(v_ptrs, mask=mask_n[:, None])
            k = tl.load(k_ptrs, mask=mask_n[:, None])

        # recompute p = softmax(qk * sm_scale, dim=-1)
        if not DIVISIBLE_N:
            valid_mask = mask_n
        if CAUSAL:
            causal_mask = (P_SEQ + offs_m[:, None]) >= (offs_n[None, :])
        s = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        s += tl.dot(q, tl.trans(k))

        p = tl.math.exp2(s * qk_scale - l[:, None] * log2e)

        # compute dp = dot(v, do)
        dp = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
        dp += tl.dot(do.to(input_dtype), tl.trans(v))

        if IS_DROPOUT:
            offs_rng = start_n + offs_rng_base
            pmask = tl.rand(seed, offs_rng, n_rounds=6) > dropout_p
            dp *= pmask

        # Load dscores gradient
        if DIVISIBLE_M and DIVISIBLE_N:
            dscores = tl.load(dscores_ptrs).to(tl.float32)
        elif DIVISIBLE_M:
            mask_n_local = offs_n < N
            dscores = tl.load(dscores_ptrs, mask=mask_n_local[None, :]).to(tl.float32)
        elif DIVISIBLE_N:
            dscores = tl.load(dscores_ptrs, mask=mask_m[:, None]).to(tl.float32)
        else:
            mask_n_local = offs_n < N
            dscores = tl.load(dscores_ptrs, mask=mask_m[:, None] & mask_n_local[None, :]).to(tl.float32)

        # compute ds = p * (dp - delta[:, None]) + dscores * sm_scale
        ds = p * (dp - delta[:, None]) + dscores * sm_scale

        # mask ds to ensure no small values
        if not DIVISIBLE_N:
            ds = tl.where(valid_mask, ds, 0.0)
        if CAUSAL:
            ds = tl.where(causal_mask, ds, 0.0)

        dq += tl.dot(ds.to(input_dtype), k)

        # increment pointers
        k_ptrs += BLOCK_N * stride_kn
        v_ptrs += BLOCK_N * stride_vn
        dscores_ptrs += BLOCK_N * stride_dsn

    dq *= sm_scale
    if DIVISIBLE_M:
        tl.store(dq_ptrs, dq.to(input_dtype))
    else:
        tl.store(dq_ptrs, dq.to(input_dtype), mask=mask_m[:, None])


# --------------------------- Autograd Function ---------------------------
class FlashAttentionWithScores(torch.autograd.Function):
    """
    Flash Attention with Scores - supports backward pass for both output and scores
    """
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, dropout_p):
        # Import the forward function
        try:
            from .attention_with_scores import attention_with_scores as fwd_fn
        except ImportError:
            from attention_with_scores import attention_with_scores as fwd_fn
        
        # Run forward pass
        o, scores = fwd_fn(q, k, v, causal, sm_scale, dropout_p)
        
        # Compute L (log normalizer) for backward pass
        # L = log(sum(exp(s * sm_scale))) where s = Q @ K^T
        # We need to recompute this from scores
        B, H, M, D = q.shape
        N = k.shape[2]
        
        if sm_scale is None:
            sm_scale = 1. / math.sqrt(D)
        
        # Compute L from scores: scores = s * sm_scale, so s = scores / sm_scale
        # L = logsumexp(s * sm_scale) = logsumexp(scores)
        # For numerical stability, we compute this properly
        # Note: masked positions in scores are 0, we need to handle this
        if causal:
            # Create causal mask
            mask = torch.triu(torch.ones(M, N, device=q.device, dtype=torch.bool), diagonal=N-M+1)
            scores_masked = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float('-inf'))
        else:
            scores_masked = scores
        
        # Compute L = logsumexp(scores) since scores already has sm_scale applied
        L = torch.logsumexp(scores_masked, dim=-1)
        
        # Handle dropout
        is_dropout = dropout_p > 0
        if is_dropout:
            offset_increment = B * H * M * N
            seed, offset = philox_cuda_seed_offset(offset_increment)
        else:
            seed, offset = 0, 0
        
        # Save for backward
        ctx.save_for_backward(q, k, v, o, L)
        ctx.sm_scale = sm_scale
        ctx.causal = causal
        ctx.dropout_p = dropout_p
        ctx.seed = seed
        ctx.offset = offset
        
        return o, scores
    
    @staticmethod
    def backward(ctx, do, dscores):
        q, k, v, o, L = ctx.saved_tensors
        sm_scale = ctx.sm_scale
        causal = ctx.causal
        dropout_p = ctx.dropout_p
        is_dropout = dropout_p > 0
        seed = ctx.seed
        offset = ctx.offset
        
        B, H, M, D = q.shape
        N = k.shape[2]
        Hk = k.shape[1]
        num_groups = H // Hk
        P_SEQ = N - M
        larger_m = M > N
        
        # Ensure contiguity
        do = maybe_contiguous(do)
        dscores = maybe_contiguous(dscores)
        
        device = torch.cuda.device_of(q)
        with torch.cuda.device(device):
            config = get_bwd_config(B, H, M, N, D, causal)
            BLOCK_M, BLOCK_N, num_stages, num_warps = config
            
            divisible_m = M % BLOCK_M == 0
            divisible_n = N % BLOCK_N == 0
            
            # Preprocess: compute delta = rowsum(o * do)
            delta = torch.empty_like(L)
            grid = (triton.cdiv(M, BLOCK_M), H, B)
            _bwd_preprocess[grid](
                o, do,
                delta,
                o.stride(0), o.stride(1), o.stride(2), o.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                delta.stride(0), delta.stride(1), delta.stride(2),
                M,
                BLOCK_M=BLOCK_M, D_HEAD=D,
                DIVISIBLE_M=divisible_m,
            )
            
            # Compute dk and dv
            dk = torch.empty((B, H, N, D), dtype=k.dtype, device=q.device)
            dv = torch.empty((B, H, N, D), dtype=v.dtype, device=q.device)
            grid = (triton.cdiv(N, BLOCK_N), H, B)
            _bwd_kv_kernel_with_scores[grid](
                q, k, v, sm_scale, do, dscores,
                dk, dv,
                L, delta,
                dropout_p, seed, offset,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                dscores.stride(0), dscores.stride(1), dscores.stride(2), dscores.stride(3),
                dk.stride(0), dk.stride(1), dk.stride(2), dk.stride(3),
                dv.stride(0), dv.stride(1), dv.stride(2), dv.stride(3),
                B, H, M, N, P_SEQ,
                num_groups,
                BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
                CAUSAL=causal, IS_DROPOUT=is_dropout,
                DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
                num_stages=num_stages, num_warps=num_warps,
            )
            
            # Compute dq
            dq = torch.zeros_like(q)
            grid = (triton.cdiv(M, BLOCK_M), H, B)
            _bwd_q_kernel_with_scores[grid](
                q, k, v, sm_scale, do, dscores,
                dq,
                L, delta,
                dropout_p, seed, offset,
                q.stride(0), q.stride(1), q.stride(2), q.stride(3),
                k.stride(0), k.stride(1), k.stride(2), k.stride(3),
                v.stride(0), v.stride(1), v.stride(2), v.stride(3),
                do.stride(0), do.stride(1), do.stride(2), do.stride(3),
                dscores.stride(0), dscores.stride(1), dscores.stride(2), dscores.stride(3),
                dq.stride(0), dq.stride(1), dq.stride(2), dq.stride(3),
                B, H, M, N, P_SEQ,
                num_groups,
                BLOCK_M=BLOCK_M, BLOCK_DMODEL=D, BLOCK_N=BLOCK_N,
                CAUSAL=causal, IS_DROPOUT=is_dropout, LARGER_M=larger_m,
                DIVISIBLE_M=divisible_m, DIVISIBLE_N=divisible_n,
                num_stages=num_stages, num_warps=num_warps,
            )
            
            # Handle grouped query attention
            dk = dk.reshape((B, Hk, num_groups, N, D)).sum(2)
            dv = dv.reshape((B, Hk, num_groups, N, D)).sum(2)
        
        return dq, dk, dv, None, None, None


def attention_with_scores_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool = False,
    sm_scale: float = None,
    dropout_p: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Compute attention output and scores with backward pass support
    
    Args:
        q: Query tensor, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        k: Key tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        v: Value tensor, shape (batch_size, num_heads_k, seq_len_k, head_dim)
        causal: Whether to use causal mask
        sm_scale: Scaling factor, if None uses 1/sqrt(head_dim)
        dropout_p: Dropout probability
        
    Returns:
        output: Attention output, shape (batch_size, num_heads_q, seq_len_q, head_dim)
        scores: Attention scores, shape (batch_size, num_heads_q, seq_len_q, seq_len_k)
    """
    return FlashAttentionWithScores.apply(q, k, v, causal, sm_scale, dropout_p)


# Alias for backward compatibility
flash_attention_backward = attention_with_scores_backward


if __name__ == "__main__":
    # Simple test
    torch.manual_seed(42)
    B, H, M, N, D = 2, 4, 64, 64, 64
    
    q = torch.randn(B, H, M, D, device='cuda', dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
    
    casual_mask = torch.triu(torch.ones(M, N, device=q.device, dtype=torch.bool), diagonal=N-M+1)
    causal = True
    
    # Forward pass
    o, scores = attention_with_scores_backward(q, k, v, causal=causal, causal_mask=casual_mask)
    
    print(f"Output shape: {o.shape}")
    print(f"Scores shape: {scores.shape}")
    
    # Backward pass
    loss = o.sum() + scores.sum()
    loss.backward()
    
    print(f"dq shape: {q.grad.shape}")
    print(f"dk shape: {k.grad.shape}")
    print(f"dv shape: {v.grad.shape}")
    
    dq_save = q.grad.clone()
    dk_save = k.grad.clone()
    dv_save = v.grad.clone()
    
    score = q*k.transpose(-2, -1)
    score = score/math.sqrt(D)
    score = score.masked_fill(casual_mask.unsqueeze(0).unsqueeze(0), float('-inf'))
    score = score.softmax(dim=-1)
    
    
    output = score@v
    
    loss = output.sum() + scores.sum()
    loss.backward()
    
    print(f"dq shape: {q.grad.shape}")
    print(f"dk shape: {k.grad.shape}")
    print(f"dv shape: {v.grad.shape}")
    
    diff_dq = q.grad - dq_save
    diff_dk = k.grad - dk_save
    diff_dv = v.grad - dv_save
    print(f"diff_dq: {diff_dq.abs().max()}")
    print(f"diff_dk: {diff_dk.abs().max()}")
    print(f"diff_dv: {diff_dv.abs().max()}")
    
    
    
    
    print("Backward pass completed successfully!")
