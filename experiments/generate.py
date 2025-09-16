# experiments/generate.py
import pathlib
import sys

import torch
import torch.nn.functional as F

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

from src.positional_encoding import sinusoidal_positional_encoding
from src.transformer_block import TransformerBlock


class CharTransformerLM(torch.nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len=512):
        super().__init__()
        self.token_emb = torch.nn.Embedding(vocab_size, d_model)
        self.blocks = torch.nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = torch.nn.LayerNorm(d_model)
        self.lm_head = torch.nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, x):
        b, l = x.size()
        pos_emb = sinusoidal_positional_encoding(
            l, self.token_emb.embedding_dim, x.device
        )
        h = self.token_emb(x) + pos_emb.unsqueeze(0)
        attns = []
        for blk in self.blocks:
            h, attn = blk(h, causal=True)
            attns.append(attn)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits, attns


def load_model(ckpt_path, device="cpu"):
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = ckpt["vocab"]
    stoi = {c: i for i, c in enumerate(vocab)}
    itos = {i: c for i, c in enumerate(vocab)}
    args = ckpt["args"]

    model = CharTransformerLM(
        vocab_size=len(vocab),
        d_model=args["d_model"],
        n_heads=args["n_heads"],
        n_layers=args["n_layers"],
        d_ff=4 * args["d_model"],
    )
    model.load_state_dict(ckpt["model_state"])
    model.to(device)
    model.eval()
    return model, stoi, itos


def sample_next_token(logits, temperature=1.0, top_k=None, top_p=None):
    """
    logits: [1, vocab_size]
    returns: [1,1] next token index
    """
    logits = logits / max(temperature, 1e-6)
    probs = F.softmax(logits, dim=-1)

    if top_k is not None:
        topk_vals, topk_idx = torch.topk(probs, top_k)
        probs = torch.zeros_like(probs)
        probs[0, topk_idx[0]] = topk_vals[0]

    if top_p is not None:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
        mask = cumulative_probs > top_p
        mask[:, 0] = 0  # always keep top token
        sorted_probs[mask] = 0.0
        probs = torch.zeros_like(probs)
        probs.scatter_(1, sorted_idx, sorted_probs)

    next_token = torch.multinomial(probs, num_samples=1)  # [1,1]
    return next_token


def generate_text(
    model,
    stoi,
    itos,
    prompt,
    max_new_tokens=200,
    temperature=1.0,
    top_k=None,
    top_p=None,
    device="cpu",
):
    idx = torch.tensor(
        [[stoi[c] for c in prompt if c in stoi]], device=device
    )  # [1, seq_len]
    generated = list(prompt)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits, _ = model(idx)
            next_char_idx = sample_next_token(
                logits[:, -1, :], temperature=temperature, top_k=top_k, top_p=top_p
            )
            generated.append(itos[next_char_idx.item()])
            idx = torch.cat([idx, next_char_idx], dim=1)

            if generated[-1] in ".!?":
                break

    return "".join(generated)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--max_new_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=None)
    parser.add_argument("--top_p", type=float, default=None)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, stoi, itos = load_model(args.ckpt, device=device)

    prompt = input("Enter prompt: ").strip()
    if prompt == "":
        prompt = "The quick brown fox"

    output = generate_text(
        model,
        stoi,
        itos,
        prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        device=device,
    )

    print("\nGenerated Text:\n", output)
