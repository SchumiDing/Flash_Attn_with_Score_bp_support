"""
Flash Attention with Scores
An efficient attention implementation that returns both attention output and scores
with full backward pass support.

Usage:
    import Flash_Attn_with_Score_bp_support as flash_attn
    
    # Direct call using the default function
    output, scores = flash_attn(q, k, v, causal=True)
    
    # Or use the class directly
    output, scores = flash_attn.FlashAttentionWithScores.apply(q, k, v, causal, sm_scale, dropout_p)
"""

try:
    from .autograd import FlashAttentionWithScores, attention_with_scores_backward
    from .attention_with_scores import attention_with_scores
except ImportError:
    # Handle case when running as standalone
    import sys
    import os
    sys.path.insert(0, os.path.dirname(__file__))
    from autograd import FlashAttentionWithScores, attention_with_scores_backward
    from attention_with_scores import attention_with_scores


# Make the module callable
import sys as _sys

class _CallableModule(_sys.modules[__name__].__class__):
    def __call__(self, *args, **kwargs):
        return attention_with_scores_backward(*args, **kwargs)

_sys.modules[__name__].__class__ = _CallableModule


__all__ = [
    # Default (with backward support)
    "FlashAttentionWithScores",
    "attention_with_scores_backward",
    # Forward only
    "attention_with_scores",
]
