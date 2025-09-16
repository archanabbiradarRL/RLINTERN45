# src/attention_numpy.py
import numpy as np


def scaled_dot_product_attention(q, k, v, mask=None, causal=False, eps=1e-9):
    """
    q, k, v: (batch, heads, seq_len, d_k)
    mask: (batch, 1, 1, seq_len) or (batch, 1, seq_len, seq_len) broadcastable
    causal: if True, apply causal mask so i cannot attend to > i
    returns: (batch, heads, seq_len, d_k)
    """
    d_k = q.shape[-1]
    scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(d_k)  # (b,h,seq,seq)
    if causal:
        L = scores.shape[-1]
        causal_mask = np.triu(np.ones((L, L), dtype=bool), k=1)
        scores = np.where(causal_mask[None, None, :, :], -1e9, scores)
    if mask is not None:
        scores = np.where(mask, -1e9, scores)
    # stable softmax
    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attn = np.exp(scores)
    attn = attn / (np.sum(attn, axis=-1, keepdims=True) + eps)
    out = np.matmul(attn, v)
    return out, attn
