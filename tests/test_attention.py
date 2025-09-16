#test_attention.py
import numpy as np
import torch
import pytest

from src.attention_numpy import scaled_dot_product_attention
from src.attention_torch import scaled_dot_product_attention_torch

def test_attention_shapes():
    b, h, l, d = 2, 3, 4, 5
    q = np.random.randn(b, h, l, d)
    k = np.random.randn(b, h, l, d)
    v = np.random.randn(b, h, l, d)

    out, attn = scaled_dot_product_attention(q, k, v)
    assert out.shape == (b, h, l, d)
    assert attn.shape == (b, h, l, l)

def test_causal_mask_numpy():
    q = np.ones((1, 1, 3, 2))
    k = np.ones((1, 1, 3, 2))
    v = np.arange(3*2).reshape(1,1,3,2).astype(float)
    out, attn = scaled_dot_product_attention(q, k, v, causal=True)
    # ensure no prob mass beyond current position
    assert np.allclose(attn[0,0,0,1:], 0.0, atol=1e-5)

def test_torch_vs_numpy_close():
    torch.manual_seed(0)
    q = torch.randn(1, 2, 4, 8)
    k = torch.randn(1, 2, 4, 8)
    v = torch.randn(1, 2, 4, 8)

    out_t, attn_t = scaled_dot_product_attention_torch(q, k, v)
    out_n, attn_n = scaled_dot_product_attention(
        q.detach().numpy(), k.detach().numpy(), v.detach().numpy()
    )
    assert np.allclose(out_t.detach().numpy(), out_n, atol=1e-5)
