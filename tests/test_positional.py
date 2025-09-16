#test_positional.py
import torch
from src.positional_encoding import sinusoidal_positional_encoding, LearnedPositionalEncoding

def test_sinusoidal_shape_and_determinism():
    pe1 = sinusoidal_positional_encoding(10, 16)
    pe2 = sinusoidal_positional_encoding(10, 16)
    assert pe1.shape == (10, 16)
    assert torch.allclose(pe1, pe2)  # deterministic

def test_learned_positional_encoding_trainable():
    enc = LearnedPositionalEncoding(20, 32)
    x = torch.randn(2, 10, 32)
    pe = enc(x)
    assert pe.requires_grad
    assert pe.shape == (10, 32)
