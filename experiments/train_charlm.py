# experiments/train_charlm.py
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import os

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src.positional_encoding import sinusoidal_positional_encoding
from src.transformer_block import TransformerBlock


# ---------------- Dataset ----------------
class CharDataset(Dataset):
    def __init__(self, text, seq_len):
        self.seq_len = seq_len
        self.vocab = sorted(list(set(text)))
        self.stoi = {s: i for i, s in enumerate(self.vocab)}
        self.itos = {i: s for s, i in self.stoi.items()}
        self.data = torch.tensor([self.stoi[c] for c in text], dtype=torch.long)

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
        attns = []
        for blk in self.blocks:
            h, attn = blk(h, causal=True)
            attns.append(attn)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits, attns


# ---------------- Training ----------------
def train(args):
    # Load data
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    dataset = CharDataset(text, args.seq_len)
    vocab_size = len(dataset.vocab)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # Initialize model
    model = CharTransformerLM(
        vocab_size, args.d_model, args.n_heads, args.n_layers, d_ff=4 * args.d_model
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs("checkpoints", exist_ok=True)
    global_step = 0
    save_interval = 1000  # save every 1000 steps

    for epoch in range(10**6):  # effectively infinite, break on steps
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits, _ = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            # Print progress
            if global_step % 100 == 0:
                print(f"step {global_step} | loss {loss.item():.4f}")

            # Save checkpoint
            if global_step > 0 and global_step % save_interval == 0:
                ckpt_path = f"checkpoints/step{global_step}.pt"
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "vocab": dataset.vocab,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint to {ckpt_path}")

            # Stop after desired steps
            if global_step >= args.steps:
                print("Reached training steps limit. Exiting.")
                return

            global_step += 1


# ---------------- Main ----------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/tiny.txt")
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--d_model", type=int, default=192)
    p.add_argument("--n_heads", type=int, default=3)
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--steps", type=int, default=5000)
    args = p.parse_args()

    train(args)
