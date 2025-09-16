import torch
import torch.nn.functional as F


def scaled_dot_product_attention(q, k, v, attn_mask=None, causal=False, dropout=None):
    """
    Scaled Dot-Product Attention with masking, causal option, and dropout.

    q, k, v: (batch, heads, seq_len, d_k)
    attn_mask: optional mask tensor (broadcastable to q @ k^T shape)
    causal: if True, apply causal mask
    dropout: optional nn.Dropout
    """
    d_k = q.size(-1)
    scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=torch.float32)
    )

    # Apply masks
    if attn_mask is not None:
        scores = scores.masked_fill(attn_mask == 0, float("-inf"))

    if causal:
        seq_len_q, seq_len_k = q.size(-2), k.size(-2)
        causal_mask = (
            torch.tril(torch.ones(seq_len_q, seq_len_k, device=q.device))
            .unsqueeze(0)
            .unsqueeze(0)
        )
        scores = scores.masked_fill(causal_mask == 0, float("-inf"))

    attn = F.softmax(scores, dim=-1)

    if dropout is not None:
        attn = dropout(attn)

    output = torch.matmul(attn, v)
    return output, attn


# alias for consistency
scaled_dot_product_attention_torch = scaled_dot_product_attention
