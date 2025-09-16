import torch
from src.attention_torch import scaled_dot_product_attention

q = torch.randn(2, 4, 8)  # batch=2, seq=4, d=8
k = torch.randn(2, 4, 8)
v = torch.randn(2, 4, 8)

out, attn = scaled_dot_product_attention(q, k, v)
print("Output shape:", out.shape)
print("Attention shape:", attn.shape)
