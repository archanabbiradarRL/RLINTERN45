# src/positional_encoding.py
import math

import torch


def sinusoidal_positional_encoding(seq_len, d_model, device="cpu"):
    pe = torch.zeros(seq_len, d_model, device=device)
    position = torch.arange(0, seq_len, dtype=torch.float, device=device).unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, d_model, 2, device=device).float()
        * (-math.log(10000.0) / d_model)
    )
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe  # (seq_len, d_model)


class LearnedPositionalEncoding(torch.nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        self.pe = torch.nn.Parameter(torch.randn(max_len, d_model) * 0.01)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return self.pe[: x.size(1)]
