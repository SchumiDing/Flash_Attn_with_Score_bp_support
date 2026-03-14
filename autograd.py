import torch
from torch.autograd import Function
from .attention_with_scores_bp import flash_attention_backward
from .attention_with_scores import attention_with_scores

class AttentionWithRowSum(Function):
    @staticmethod
    def forward(ctx, q, k, v, causal, sm_scale, dropout_p):
        o, score = flash_attention_backward(q, k, v, causal, sm_scale, dropout_p)
        ctx.save_for_backward(q, k, v, o, sm_scale, causal, dropout_p)
        return o, score

    @staticmethod
    def backward(ctx, do):
        q, k, v, o, score, sm_scale, causal, dropout_p = ctx.saved_tensors
        dq, dk, dv = flash_attention_backward(do, q, k, v, o, sm_scale, causal, dropout_p)
        return dq, dk, dv

