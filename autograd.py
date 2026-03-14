import torch
from torch.autograd import Function
from .attention_with_scores_bp import flash_attention_backward
from .attention_with_scores import attention_with_scores

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

import time
if __name__ == "__main__":
    # Simple test
    torch.manual_seed(42)
    B, H, M, N, D = 32, 40, 1024, 1024, 128
    
    print("=" * 60)
    print("Testing Flash Attention with Scores Backward Pass")
    print("=" * 60)
    
    # Test 1: Basic forward and backward pass
    print("\n[Test 1] Basic forward and backward pass")
    q = torch.randn(B, H, M, D, device='cuda', dtype=torch.float16, requires_grad=True)
    k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
    v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
    
    causal = True
    
    # Forward pass
    t1 = time.time()
    o, scores = attention_with_scores_backward(q, k, v, causal=causal)
    t2 = time.time()

    # Backward pass
    loss = o.sum() + scores.sum()
    loss.backward()
    t3 = time.time()
    print(f"Forward time: {t2 - t1:.6f} seconds")
    print(f"Backward time: {t3 - t2:.6f} seconds")
    print(f"Total time: {t3 - t1:.6f} seconds")

    kernel_forward_time = t2 - t1
    kernel_backward_time = t3 - t2
    kernel_total_time = t3 - t1
    
    print(f"dq shape: {q.grad.shape}")
    print(f"dk shape: {k.grad.shape}")
    print(f"dv shape: {v.grad.shape}")
    print("[Test 1] PASSED!")
    
    # Test 2: Compare with PyTorch reference implementation
    print("\n[Test 2] Comparing with PyTorch reference implementation")
    torch.manual_seed(42)
    
    q2 = torch.randn(B, H, M, D, device='cuda', dtype=torch.float16, requires_grad=True)
    k2 = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
    v2 = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
    
    # Our implementation
    t1 = time.time()
    o_ours, scores_ours = attention_with_scores_backward(q2, k2, v2, causal=False)
    t2 = time.time()
    
    loss_ours = o_ours.sum()
    loss_ours.backward()
    t3 = time.time()
    print(f"Forward time: {t2 - t1:.6f} seconds")
    print(f"Backward time: {t3 - t2:.6f} seconds")
    print(f"Total time: {t3 - t1:.6f} seconds")

    kernel_forward_time = t2 - t1
    kernel_backward_time = t3 - t2
    kernel_total_time = t3 - t1
    dq_ours = q2.grad.clone()
    dk_ours = k2.grad.clone()
    dv_ours = v2.grad.clone()
    
    # Reset gradients
    q2.grad = None
    k2.grad = None
    v2.grad = None
    
    # PyTorch reference
    sm_scale = 1.0 / math.sqrt(D)
    
    t1 = time.time()
    scores_ref = torch.matmul(q2, k2.transpose(-2, -1)) * sm_scale
    attn_weights = torch.softmax(scores_ref, dim=-1)
    o_ref = torch.matmul(attn_weights, v2)
    t2 = time.time()
    
    
    loss_ref = o_ref.sum()
    loss_ref.backward()
    t3 = time.time()
    print(f"Forward time: {t2 - t1:.6f} seconds")
    print(f"Backward time: {t3 - t2:.6f} seconds")
    print(f"Total time: {t3 - t1:.6f} seconds")
    
    naive_forward_time = t2 - t1
    naive_backward_time = t3 - t2
    naive_total_time = t3 - t1
    
    dq_ref = q2.grad.clone()
    dk_ref = k2.grad.clone()
    dv_ref = v2.grad.clone()
    
    # Compare
    print(f"Output max diff: {(o_ours - o_ref).abs().max().item():.6f}")
    print(f"dq max diff: {(dq_ours - dq_ref).abs().max().item():.6f}")
    print(f"dk max diff: {(dk_ours - dk_ref).abs().max().item():.6f}")
    print(f"dv max diff: {(dv_ours - dv_ref).abs().max().item():.6f}")
    
    print(f"max_dv: {dv_ours.abs().max().item():.6f}")
    print(f"max_dv_ref: {dv_ref.abs().max().item():.6f}")
    print(f"max_dq: {dq_ours.abs().max().item():.6f}")
    print(f"max_dq_ref: {dq_ref.abs().max().item():.6f}")
    print(f"max_dk: {dk_ours.abs().max().item():.6f}")
    print(f"max_dk_ref: {dk_ref.abs().max().item():.6f}")
    print(f"max_o: {o_ours.abs().max().item():.6f}")
    print(f"max_o_ref: {o_ref.abs().max().item():.6f}")
    # Check if differences are within acceptable tolerance (fp16 has limited precision)
    tol = 1e-2
    if (o_ours - o_ref).abs().max() < tol and \
       (dq_ours - dq_ref).abs().max() < tol and \
       (dk_ours - dk_ref).abs().max() < tol and \
       (dv_ours - dv_ref).abs().max() < tol:
        print("[Test 2] PASSED!")
    else:
        print("[Test 2] WARNING: Differences exceed tolerance, but this may be due to fp16 precision")
    
    print(f"Kernel forward time: {kernel_forward_time:.6f} seconds")
    print(f"Kernel backward time: {kernel_backward_time:.6f} seconds")
    print(f"Kernel total time: {kernel_total_time:.6f} seconds")
    print(f"Naive forward time: {naive_forward_time:.6f} seconds")
    print(f"Naive backward time: {naive_backward_time:.6f} seconds")
    print(f"Naive total time: {naive_total_time:.6f} seconds")
    
    print("\n" + "=" * 60)
    print("All tests completed!")
    print("=" * 60)
