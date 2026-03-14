# Flash Attention with Scores (Backward Pass Support)

An efficient Flash Attention implementation that returns both attention output and attention scores, with **full backward pass support** for gradient computation.

## Features

- **Efficient Score Computation**: Computes both attention output and scores in a single fused kernel
- **Full Backward Pass Support**: Supports gradient computation for both output and scores via PyTorch autograd
- **Production Ready**: Wrapped as a PyTorch autograd Function, ready for training pipelines
- **GQA Support**: Supports Grouped Query Attention (GQA)
- **Causal Masking**: Built-in support for causal attention masks

## Installation

### Requirements

- Python >= 3.8
- PyTorch >= 2.0.0
- Triton >= 2.0.0
- NumPy

### Install from source

```bash
git clone https://github.com/yourusername/Flash_Attn_with_Score_bp_support.git
cd Flash_Attn_with_Score_bp_support
pip install -e .
```

### Install dependencies only

```bash
pip install torch>=2.0.0 triton>=2.0.0 numpy
```

## Quick Start

```python
import torch
import Flash_Attn_with_Score_bp_support as flash_attn

# Create input tensors
B, H, M, N, D = 32, 40, 1024, 1024, 128
q = torch.randn(B, H, M, D, device='cuda', dtype=torch.float16, requires_grad=True)
k = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)
v = torch.randn(B, H, N, D, device='cuda', dtype=torch.float16, requires_grad=True)

# Forward pass - directly call the module
output, scores = flash_attn(q, k, v, causal=True)

# Backward pass - gradients flow through both output and scores
loss = output.sum() + scores.sum()
loss.backward()

print(f"dq shape: {q.grad.shape}")  # (32, 40, 1024, 128)
print(f"dk shape: {k.grad.shape}")  # (32, 40, 1024, 128)
print(f"dv shape: {v.grad.shape}")  # (32, 40, 1024, 128)
```

## API Reference

### Module-level Call

```python
import Flash_Attn_with_Score_bp_support as flash_attn

output, scores = flash_attn(q, k, v, causal=False, sm_scale=None, dropout_p=0.0)
```

### `attention_with_scores_backward`

Main function with backward pass support.

```python
from Flash_Attn_with_Score_bp_support import attention_with_scores_backward

output, scores = attention_with_scores_backward(
    q,           # Query tensor, shape (B, H_q, M, D)
    k,           # Key tensor, shape (B, H_k, N, D)
    v,           # Value tensor, shape (B, H_k, N, D)
    causal=False,      # Whether to use causal mask
    sm_scale=None,     # Scaling factor, defaults to 1/sqrt(D)
    dropout_p=0.0,     # Dropout probability
)
# Returns:
#   output: (B, H_q, M, D)
#   scores: (B, H_q, M, N)
```

### `FlashAttentionWithScores`

The underlying autograd Function class.

```python
from Flash_Attn_with_Score_bp_support import FlashAttentionWithScores

output, scores = FlashAttentionWithScores.apply(q, k, v, causal, sm_scale, dropout_p)
```

### `attention_with_scores`

Forward-only version (no backward pass support, slightly faster).

```python
from Flash_Attn_with_Score_bp_support import attention_with_scores

output, scores = attention_with_scores(q, k, v, causal=True)
```

## Acknowledgments

The Triton kernel implementation is based on [Flash_Attn_with_Score](https://github.com/DoubtedSteam/Flash_Attn_with_Score) by DoubtedSteam. This project extends the original work by implementing efficient backward pass kernels and wrapping everything into a production-ready PyTorch autograd Function.

## License

MIT License
