# experiments/eval_loss.py
import argparse
import math

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset

from src.positional_encoding import sinusoidal_positional_encoding
from src.transformer_block import TransformerBlock


# ---------------- Dataset ----------------
class CharDataset(Dataset):
    def __init__(self, text, seq_len, vocab=None):
        self.seq_len = seq_len
        if vocab is None:
            vocab = sorted(list(set(text)))
        self.vocab = vocab
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for i, s in enumerate(self.vocab)}
        self.data = torch.tensor([self.stoi.get(c, 0) for c in text], dtype=torch.long)

    def __len__(self):
        return len(self.data) - self.seq_len

    def __getitem__(self, idx):
        x = self.data[idx : idx + self.seq_len]
        y = self.data[idx + 1 : idx + self.seq_len + 1]
        return x, y


# ---------------- Model ----------------
class CharTransformerLM(nn.Module):
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff, max_len=512):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_len = max_len

    def forward(self, x):
        b, l = x.size()
        pos_emb = sinusoidal_positional_encoding(
            l, self.token_emb.embedding_dim, x.device
        )
        h = self.token_emb(x) + pos_emb.unsqueeze(0)
        for blk in self.blocks:
            h, _ = blk(h, causal=True)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits


# ---------------- Model Loader ----------------
def load_model(ckpt_path, device):
    ckpt = torch.load(ckpt_path, map_location=device)
    vocab = ckpt["vocab"]
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
    return model, vocab


# ---------------- Evaluation ----------------
def evaluate(model, dataset, batch_size=64, device="cpu"):
    loader = DataLoader(dataset, batch_size=batch_size)
    criterion = nn.CrossEntropyLoss(reduction="sum")
    total_loss, total_tokens = 0.0, 0

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, logits.size(-1)), y.view(-1))
            total_loss += loss.item()
            total_tokens += y.numel()

    avg_loss = total_loss / total_tokens  # nats
    bpc = avg_loss / math.log(2)
    ppl = math.exp(avg_loss)
    return avg_loss, bpc, ppl


# ---------------- Text Generation ----------------
def generate_text(
    model,
    stoi,
    itos,
    prompt,
    max_new_tokens=200,
    temperature=0.7,
    top_k=50,
    top_p=0.9,
    device="cpu",
):
    model.eval()
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)
    generated = list(prompt)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(idx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)

            # Top-k sampling
            topk = min(top_k, probs.size(-1))
            topk_vals, topk_idx = torch.topk(probs, topk, dim=-1)
            topk_probs = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
            next_idx = topk_idx[0, torch.multinomial(topk_probs, 1)]

            next_char = itos[next_idx.item()]
            generated.append(next_char)

            # ✅ Fix dimension mismatch
            next_idx = next_idx.view(1, 1)
            idx = torch.cat([idx, next_idx], dim=1)

            if next_char in ".!?":
                break
    return "".join(generated)


# ---------------- Main ----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ckpts", nargs="+", required=True, help="List of checkpoint paths"
    )
    parser.add_argument("--data", type=str, default="data/sample.txt")
    parser.add_argument("--seq_len", type=int, default=64)
    parser.add_argument("--sample_prompt", type=str, default="The quick brown fox")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top_k", type=int, default=50)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load text for evaluation
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    steps, bpc_values = [], []

    for ckpt_path in args.ckpts:
        step = int("".join(filter(str.isdigit, ckpt_path)))
        model, vocab = load_model(ckpt_path, device)
        dataset = CharDataset(text, args.seq_len, vocab)
        loss, bpc, ppl = evaluate(model, dataset, device=device)
        print(
            f"[{ckpt_path}] Loss: {loss:.4f} | BPC: {bpc:.3f} | Perplexity: {ppl:.2f}"
        )

        # Generate sample
        sample = generate_text(
            model,
            dataset.stoi,
            dataset.itos,
            args.sample_prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_k=args.top_k,
            device=device,
        )
        print(f"\n[Sample Generation from {ckpt_path}]:\n{sample}\n")

        steps.append(step)
        bpc_values.append(bpc)

    # Plot BPC trend
    plt.figure(figsize=(8, 5))
    plt.plot(steps, bpc_values, marker="o", label="Validation BPC")
    plt.xlabel("Training Steps")
    plt.ylabel("Bits per Character (BPC)")
    plt.title("Validation BPC across checkpoints")
    plt.legend()
    plt.grid(True)
    plt.savefig("bpc_trend.png")
    print("Saved BPC trend plot to bpc_trend.png")
    plt.show()
