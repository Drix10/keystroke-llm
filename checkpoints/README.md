# 💾 Model Checkpoints (`checkpoints/`)

This directory stores trained weight matrices for `TinyTransformer`.

---

## 📦 What is `model_final.npz`?

`model_final.npz` is a compressed NumPy archive (`np.savez_compressed`) containing all parameters of the trained Transformer:

| Parameter Matrix | Dimensions | Description |
|---|---|---|
| **`W_emb`** | `[vocab_size, 32]` | Character embedding lookup table |
| **`W_pos`** | `[seq_len, 32]` | Learned positional embeddings |
| **`W_q`** | `[32, 32]` | Attention Query projection weights |
| **`W_k`** | `[32, 32]` | Attention Key projection weights |
| **`W_v`** | `[32, 32]` | Attention Value projection weights |
| **`W_o`** | `[32, 32]` | Attention Output projection weights |
| **`W1`** | `[32, 64]` | Feed-Forward expansion layer (ReLU) |
| **`W2`** | `[64, 32]` | Feed-Forward projection layer |
| **`W_out`** | `[32, vocab_size]` | Output vocabulary classifier head |
| **`b_out`** | `[vocab_size]` | Output classifier bias vector |
| **`vocab`** | Optional object array | Embedded character vocabulary |
| **`adam_step`** | Scalar | Adam update counter |
| **`adam_emb_m`, `adam_emb_v`** | `[vocab_size, 32]` | Adam moments for token embeddings |
| **`adam_pos_m`, `adam_pos_v`** | `[seq_len, 32]` | Adam moments for positional embeddings |
| **`adam_m__*`, `adam_v__*`** | Per-parameter shape | Adam moments for each dense parameter |

---

## 🔍 How to Inspect Checkpoint Weights in Python

You can easily inspect the raw numbers and matrices in Python:

```python
import numpy as np

data = np.load("checkpoints/model_final.npz", allow_pickle=True)
print("Saved parameters:", data.files)
print("Embedding matrix shape:", data["W_emb"].shape)
print("Vocabulary size:", data["vocab_size"])
```
