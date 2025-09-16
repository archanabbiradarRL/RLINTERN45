# src/mha.py
from torch import nn

from .attention_torch import scaled_dot_product_attention_torch


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.0):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.qkv_proj = nn.Linear(d_model, 3 * d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, x):
        # x: (batch, seq, d_model) -> (batch, heads, seq, d_head)
        b, l, _ = x.size()
        x = x.view(b, l, self.num_heads, self.d_head).transpose(1, 2)
        return x

    def combine_heads(self, x):
        # x: (batch, heads, seq, d_head) -> (batch, seq, d_model)
        b, h, l, d = x.size()
        x = x.transpose(1, 2).contiguous().view(b, l, h * d)
        return x

    def forward(self, x, mask=None, causal=False):
        b, seq, _ = x.size()
        qkv = self.qkv_proj(x)  # (b, seq, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)
        q, k, v = map(self.split_heads, (q, k, v))
        out, attn = scaled_dot_product_attention_torch(
            q, k, v, attn_mask=mask, causal=causal, dropout=self.dropout
        )
        out = self.combine_heads(out)
        out = self.out_proj(out)
        return out, attn
