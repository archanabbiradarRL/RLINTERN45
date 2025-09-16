#test_block.py
import torch
from src.transformer_block import TransformerBlock

def test_block_forward_pass():
    b, l, d = 2, 6, 32
    x = torch.randn(b, l, d)
    block = TransformerBlock(d_model=d, num_heads=4, d_ff=64)
    out, attn = block(x, causal=True)
    assert out.shape == (b, l, d)
    assert attn.shape[2:] == (l, l)
