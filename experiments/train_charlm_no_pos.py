# experiments/train_charlm.py
import pathlib
import sys

sys.path.append(str(pathlib.Path(__file__).resolve().parent.parent))

import argparse
import os

import matplotlib.pyplot as plt
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

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
    def __init__(self, vocab_size, d_model, n_heads, n_layers, d_ff):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        # Removed positional encoding
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, x):
        h = self.token_emb(x)  # No positional embedding added
        for blk in self.blocks:
            h, _ = blk(h, causal=True)
        h = self.ln_f(h)
        logits = self.lm_head(h)
        return logits


# ---------------- Generation Helper ----------------
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
    import torch.nn.functional as F

    model.eval()
    idx = torch.tensor([[stoi.get(c, 0) for c in prompt]], device=device)
    generated = list(prompt)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            logits = model(idx)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            probs = F.softmax(logits, dim=-1)

            topk = min(top_k, probs.size(-1))
            topk_vals, topk_idx = torch.topk(probs, topk, dim=-1)
            topk_probs = topk_vals / topk_vals.sum(dim=-1, keepdim=True)
            next_idx = topk_idx[0, torch.multinomial(topk_probs, 1)]
            next_char = itos[next_idx.item()]
            generated.append(next_char)

            next_idx = next_idx.view(1, 1)
            idx = torch.cat([idx, next_idx], dim=1)

            if next_char in ".!?":
                break
    return "".join(generated)


# ---------------- Training ----------------
def train(args):
    # Load dataset
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    if args.val_data is None:
        split_idx = int(0.9 * len(text))
        train_text, val_text = text[:split_idx], text[split_idx:]
    else:
        with open(args.val_data, "r", encoding="utf-8") as f:
            val_text = f.read()
        train_text = text

    train_dataset = CharDataset(train_text, args.seq_len)
    val_dataset = CharDataset(val_text, args.seq_len, vocab=train_dataset.vocab)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    vocab_size = len(train_dataset.vocab)

    # Model
    model = CharTransformerLM(
        vocab_size, args.d_model, args.n_heads, args.n_layers, d_ff=4 * args.d_model
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    opt = optim.AdamW(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    os.makedirs("checkpoints", exist_ok=True)
    global_step = 0
    save_interval = 1000
    val_interval = 100

    train_losses, val_losses, steps = [], [], []

    while True:
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits.view(-1, vocab_size), y.view(-1))

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

            if global_step % 100 == 0:
                print(f"step {global_step} | train loss {loss.item():.4f}")
                train_losses.append(loss.item())
                steps.append(global_step)

            if global_step > 0 and global_step % val_interval == 0:
                model.eval()
                val_loss_sum, total_tokens = 0, 0
                with torch.no_grad():
                    for vx, vy in val_loader:
                        vx, vy = vx.to(device), vy.to(device)
                        v_logits = model(vx)
                        v_loss = criterion(v_logits.view(-1, vocab_size), vy.view(-1))
                        val_loss_sum += v_loss.item() * vy.numel()
                        total_tokens += vy.numel()
                avg_val_loss = val_loss_sum / total_tokens
                val_losses.append(avg_val_loss)
                print(f"step {global_step} | validation loss {avg_val_loss:.4f}")
                model.train()

            if global_step > 0 and global_step % save_interval == 0:
                ckpt_path = f"checkpoints/pos_step{global_step}.pt"
                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "vocab": train_dataset.vocab,
                        "args": vars(args),
                    },
                    ckpt_path,
                )
                print(f"Saved checkpoint to {ckpt_path}")

                prompt = "investment"
                sample = generate_text(
                    model,
                    train_dataset.stoi,
                    train_dataset.itos,
                    prompt,
                    max_new_tokens=100,
                    temperature=0.7,
                    top_k=50,
                    top_p=0.9,
                    device=device,
                )
                print(f"\n[Sample at step {global_step}]:\n{sample}\n")

            if global_step >= args.steps:
                plt.figure(figsize=(8, 5))
                plt.plot(steps[: len(train_losses)], train_losses, label="Train Loss")
                if val_losses:
                    val_x = list(
                        range(
                            val_interval,
                            val_interval * (len(val_losses) + 1),
                            val_interval,
                        )
                    )
                    plt.plot(val_x, val_losses, label="Validation Loss")
                plt.xlabel("Steps")
                plt.ylabel("Loss")
                plt.title("Training and Validation Loss")
                plt.legend()
                plt.grid(True)
                plt.savefig("loss_curve.png")
                print("Saved loss curve to loss_curve.png")
                plt.show()
                return

            global_step += 1


# ---------------- Main ----------------
if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="data/sample.txt")
    p.add_argument("--val_data", type=str, default=None)
    p.add_argument("--seq_len", type=int, default=64)
    p.add_argument("--d_model", type=int, default=192)
    p.add_argument("--n_heads", type=int, default=1)  # ✅ set n_heads=1
    p.add_argument("--n_layers", type=int, default=2)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--steps", type=int, default=5000)
    args = p.parse_args()

    train(args)
