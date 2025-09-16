#test_mha.py
import torch
from src.mha import MultiHeadAttention

def test_mha_shapes():
    b, l, d = 2, 5, 16
    n_heads = 4
    x = torch.randn(b, l, d)
    mha = MultiHeadAttention(d_model=d, num_heads=n_heads)
    out, attn = mha(x)
    assert out.shape == (b, l, d)
    assert attn.shape == (b, n_heads, l, l)

def test_mha_single_head_equivalence():
    b, l, d = 1, 3, 8
    x = torch.randn(b, l, d)
    mha = MultiHeadAttention(d_model=d, num_heads=1)
    out, _ = mha(x)
    assert out.shape == (b, l, d)
