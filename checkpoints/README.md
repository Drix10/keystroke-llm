# 💾 Model Checkpoints (`checkpoints/`)

This directory stores trained weight matrices for `TinyTransformer`.

---

## 📦 What is `model_final.npz`?

`model_final.npz` is a compressed NumPy archive (`np.savez_compressed`) containing all parameters of the trained Transformer:

| Parameter Matrix | Dimensions | Description |
|---|---|---|
| **`W_emb`** | `[vocab_size, 32]` | Character embedding lookup table |
| **`W_q`** | `[32, 32]` | Attention Query projection weights |
| **`W_k`** | `[32, 32]` | Attention Key projection weights |
| **`W_v`** | `[32, 32]` | Attention Value projection weights |
| **`W_o`** | `[32, 32]` | Attention Output projection weights |
| **`W1`** | `[32, 64]` | Feed-Forward expansion layer (ReLU) |
| **`W2`** | `[64, 32]` | Feed-Forward projection layer |
| **`W_out`** | `[32, vocab_size]` | Output vocabulary classifier head |
| **`b_out`** | `[vocab_size]` | Output classifier bias vector |

---

## 🔍 How to Inspect Checkpoint Weights in Python

You can easily inspect the raw numbers and matrices in Python:

```python
import numpy as np

data = np.load("checkpoints/model_final.npz")
print("Saved parameters:", data.files)
print("Embedding matrix shape:", data["W_emb"].shape)
print("Vocabulary size:", data["vocab_size"])
```
