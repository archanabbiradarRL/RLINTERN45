# Reproducibility Instructions

This document contains step-by-step commands to train, evaluate, and generate text using the Transformer-based character-level language model in this repository.

---

## 1. Setup Environment

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the environment:

- **Windows (PowerShell):**
```powershell
.venv\Scripts\Activate.ps1
```

- **Windows (cmd):**
```cmd
.venv\Scripts\activate.bat
```

- **Linux/macOS:**
```bash
source .venv/bin/activate
```

3. Install required packages:
```bash
pip install -r requirements.txt
```

---

## 2. Code Formatting & Type Checking

1. Format code using **Black**:
```bash
python -m black src/ experiments/
```

2. Check type hints using **mypy**:
```bash
python -m mypy src/ experiments/
```

3. Lint code using **Ruff**:
```bash
python -m ruff check src/ experiments/
```

---

## 3. Training

### Train a basic character-level LM:
```bash
python experiments/train_charlm.py --data data/tiny.txt --device cpu --steps 1000
```

### Train with positional encoding:
```bash
python experiments/train_charlm_loss.py --data data/sample.txt --seq_len 64 --d_model 256 --n_heads 1 --n_layers 3 --lr 3e-4 --batch_size 64 --steps 5000        
                                                            
```
### Train without positional encoding:
```bash
python experiments/train_charlm_pos.py --data data/sample.txt --seq_len 64 --d_model 256 --n_heads 1 --n_layers 3 --lr 3e-4 --batch_size 64 --steps 5000   
```

---

## 4. Evaluation

Evaluate model on validation/test data:
```bash
python experiments/eval.py --model checkpoints/step1000.pt --data data/sample.txt
```

Evaluate using loss metrics with positional encoding:
```bash
python -m experiments.eval_loss --ckpts checkpoints/loss_step1000.pt checkpoints/loss_step2000.pt checkpoints/loss_step3000.pt checkpoints/loss_step4000.pt checkpoints/loss_step5000.pt --data data/sample.txt --seq_len 64
```

Evaluate using loss metrics without positional encoding:
```bash
python -m experiments.eval_loss --ckpts checkpoints/loss_step1000.pt checkpoints/loss_step2000.pt checkpoints/loss_step3000.pt checkpoints/loss_step3999.pt --data data/sample.txt --seq_len 64
```
---

## 5. Generation

Generate text using a trained model with positional encoding:
```bash
python -m experiments.generate --ckpt checkpoints/loss_step5000.pt --max_new_tokens 300 --temperature 0.7 --top_p 0.9 --top_k 50                                   
```

Generate text using a trained model without positional encoding:
```bash
python -m experiments.generate --ckpt checkpoints/pos_step5000.pt --max_new_tokens 300 --temperature 0.7 --top_p 0.9 --top_k 50
   
```

---

## 6. Git & Version Control

1. Check remote repository:
```bash
git remote -v
```

2. Push changes to GitHub:
```bash
git add .
git commit -m "Training scripts added"
git push -u origin master
```

---

## Notes

- Make sure to activate the virtual environment before running commands.
- Replace file paths and model checkpoint names