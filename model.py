"""Educational Character-Level Transformer in pure NumPy."""

import numpy as np


def randn(rows: int, cols: int, scale: float = 0.1) -> np.ndarray:
    """Initialize weight matrix uniformly in [-scale, scale]."""
    return np.random.uniform(-scale, scale, size=(rows, cols)).astype(np.float64)


def xavier_uniform(rows: int, cols: int) -> np.ndarray:
    """Xavier/Glorot uniform initialization — keeps activations well-scaled at start."""
    limit = np.sqrt(6.0 / (rows + cols))
    return np.random.uniform(-limit, limit, size=(rows, cols)).astype(np.float64)


def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Matrix multiplication wrapper: [..., M, K] x [..., K, N] -> [..., M, N]."""
    return np.matmul(A, B)


def transpose(A: np.ndarray) -> np.ndarray:
    """Swap last two dimensions: [..., M, N] -> [..., N, M]."""
    return np.swapaxes(A, -1, -2)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    return (x > 0.0).astype(np.float64)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along target axis."""
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)


class TinyTransformer:
    """Single-layer causal Transformer for character-level prediction."""

    def __init__(self, vocab_size: int = 96, hidden_size: int = 32, seq_len: int = 12, init_scale: float = 0.1):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.init_scale = init_scale

        # Token embeddings: [vocab_size, hidden_size]
        self.W_emb = xavier_uniform(self.vocab_size, self.hidden_size)

        # Positional embeddings: [seq_len, hidden_size] — tells the model WHERE each token sits
        # Initialized to zeros so training starts from neutral position signals.
        self.W_pos = np.zeros((self.seq_len, self.hidden_size), dtype=np.float64)

        # Self-attention projections: [hidden_size, hidden_size]
        self.W_q = xavier_uniform(self.hidden_size, self.hidden_size)
        self.W_k = xavier_uniform(self.hidden_size, self.hidden_size)
        self.W_v = xavier_uniform(self.hidden_size, self.hidden_size)
        self.W_o = xavier_uniform(self.hidden_size, self.hidden_size)

        # Feed-forward network (FFN)
        self.W1 = xavier_uniform(self.hidden_size, 2 * self.hidden_size)
        self.W2 = xavier_uniform(2 * self.hidden_size, self.hidden_size)

        # Output head: [hidden_size, vocab_size]
        self.W_out = xavier_uniform(self.hidden_size, self.vocab_size)
        self.b_out = np.zeros(self.vocab_size, dtype=np.float64)

        # Precomputed autoregressive mask: [seq_len, seq_len]
        self.causal_mask = self._build_causal_mask(self.seq_len)
        # Boolean version — used for safe gradient zeroing (no fragile float comparisons)
        self.causal_mask_bool = self.causal_mask < 0

        # Attention weights cache for visualization
        self.last_attn_weights = None
        self.last_cache = {}
        self.vocab = None

    @staticmethod
    def _build_causal_mask(size: int) -> np.ndarray:
        mask = np.zeros((size, size), dtype=np.float64)
        for i in range(size):
            for j in range(size):
                if j > i:
                    mask[i, j] = -1e9
        return mask

    def forward(self, token_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """Forward pass for context sequence of length T <= seq_len.

        Returns:
            logits: [T, vocab_size]
            next_char_probs: [vocab_size]
        """
        if len(token_indices) > self.seq_len:
            token_indices = token_indices[-self.seq_len:]
        T = len(token_indices)
        assert 1 <= T <= self.seq_len, f"Sequence length {T} exceeds maximum {self.seq_len}"

        # 1. Embedding lookup + positional encoding: [T, hidden_size]
        #    Token embedding says WHAT the character is; position says WHERE it sits.
        X = self.W_emb[token_indices] + self.W_pos[:T, :]

        # 2. Q, K, V projections: [T, hidden_size]
        Q = matmul(X, self.W_q)
        K = matmul(X, self.W_k)
        V = matmul(X, self.W_v)

        # 3. Scaled dot-product attention with causal mask: [T, T]
        d_k = float(self.hidden_size)
        raw_scores = matmul(Q, transpose(K)) / np.sqrt(d_k)
        masked_scores = raw_scores + self.causal_mask[:T, :T]
        attn_weights = softmax(masked_scores, axis=-1)
        self.last_attn_weights = attn_weights

        # 4. Context aggregation & output projection: [T, hidden_size]
        context = matmul(attn_weights, V)
        attn_out = matmul(context, self.W_o)

        # 5. Residual connection 1
        X_res1 = X + attn_out

        # 6. Feed-Forward Network: expand -> ReLU -> project
        ffn_pre = matmul(X_res1, self.W1)              # [T, 2*hidden_size]
        ffn_act = relu(ffn_pre)                        # [T, 2*hidden_size]
        ffn_out = matmul(ffn_act, self.W2)             # [T, hidden_size]

        # 7. Residual connection 2
        X_res2 = X_res1 + ffn_out                      # [T, hidden_size]

        # 8. Output logits & next-token distribution: [T, vocab_size]
        logits = matmul(X_res2, self.W_out) + self.b_out
        next_char_probs = softmax(logits[-1, :])

        # Cache activations for training/inspection
        self.last_cache = {
            "token_indices": token_indices,
            "X": X, "Q": Q, "K": K, "V": V,
            "raw_scores": raw_scores, "masked_scores": masked_scores,
            "attn_weights": attn_weights, "context": context, "attn_out": attn_out,
            "X_res1": X_res1, "ffn_pre": ffn_pre, "ffn_act": ffn_act,
            "ffn_out": ffn_out, "X_res2": X_res2, "logits": logits
        }

        return logits, next_char_probs

    def compute_loss(self, next_char_probs: np.ndarray, target_token: int) -> float:
        """Negative log-likelihood of target token."""
        p = np.clip(next_char_probs[target_token], 1e-12, 1.0)
        return float(-np.log(p))

    def generate(self, seed_tokens: list[int], num_chars: int = 20,
                 temperature: float = 1.0, top_p: float = 0.9) -> list[int]:
        """Auto-regressively sample `num_chars` tokens given a seed context.

        Args:
            seed_tokens: Starting context as token IDs.
            num_chars: How many new characters to generate.
            temperature: >1 = more random, <1 = more focused, 0 = greedy.
            top_p: Nucleus sampling threshold — only sample from top-p probability mass.

        Returns:
            List of generated token IDs (not including seed).
        """
        generated = []
        ctx = list(seed_tokens)[-self.seq_len:]
        for _ in range(num_chars):
            _, probs = self.forward(ctx)
            if temperature <= 0.0:
                next_id = int(np.argmax(probs))
            else:
                scaled = probs ** (1.0 / temperature)
                scaled /= scaled.sum()
                sorted_ids = np.argsort(scaled)[::-1]
                cumulative = 0.0
                nucleus = []
                for sid in sorted_ids:
                    cumulative += scaled[sid]
                    nucleus.append(sid)
                    if cumulative >= top_p:
                        break
                nucleus_probs = np.array([scaled[i] for i in nucleus])
                nucleus_probs /= nucleus_probs.sum()
                next_id = int(np.random.choice(nucleus, p=nucleus_probs))
            generated.append(next_id)
            ctx.append(next_id)
            if len(ctx) > self.seq_len:
                ctx = ctx[-self.seq_len:]
        return generated

    def train_step_heuristic(self, token_indices: list[int], target: int | list[int], lr: float = 0.01) -> float:
        """Educational update: adjusts output head and context embeddings only."""
        if isinstance(target, (list, tuple, np.ndarray)):
            target_token = int(target[-1])
        else:
            target_token = int(target)

        _, probs = self.forward(token_indices)
        loss = self.compute_loss(probs, target_token)

        # Error gradient across vocabulary: dL/dz = probs - y_onehot
        d_logits = probs.copy()
        d_logits[target_token] -= 1.0

        last_hidden = self.last_cache["X_res2"][-1]
        dW_out = np.outer(last_hidden, d_logits)
        np.clip(dW_out, -2.0, 2.0, out=dW_out)

        self.W_out -= lr * dW_out
        self.b_out -= lr * np.clip(d_logits, -2.0, 2.0)

        # Soft nudge to input embeddings towards target direction
        target_dir = self.W_out[:, target_token]
        norm_dir = target_dir / (np.linalg.norm(target_dir) + 1e-8)
        err = probs[target_token] - 1.0
        for t in token_indices:
            self.W_emb[t] -= lr * 0.05 * err * norm_dir

        return loss

    def train_step_backprop(self, token_indices: list[int], target: int | list[int], lr: float = 0.003, use_adam: bool = True) -> float:
        """Exact analytical backpropagation with optional Adam optimizer across single or all sequence positions."""
        _, probs = self.forward(token_indices)
        c = self.last_cache
        T = len(token_indices)
        d_k = float(self.hidden_size)

        # 1. Output head gradient & loss computation
        if isinstance(target, (list, tuple, np.ndarray)):
            y_seq = np.array(target, dtype=int)
            probs_all = softmax(c["logits"], axis=-1)
            losses = -np.log(np.clip(probs_all[np.arange(T), y_seq], 1e-12, 1.0))
            loss = float(np.mean(losses))
            d_logits = probs_all.copy() / T
            d_logits[np.arange(T), y_seq] -= 1.0 / T
        else:
            target_token = int(target)
            loss = self.compute_loss(probs, target_token)
            d_logits = np.zeros_like(c["logits"])
            d_logits[-1, :] = probs.copy()
            d_logits[-1, target_token] -= 1.0

        dW_out = matmul(transpose(c["X_res2"]), d_logits)
        db_out = np.sum(d_logits, axis=0)
        dX_res2 = matmul(d_logits, transpose(self.W_out))

        # 2. Residual 2 & FFN backprop
        dX_res1 = dX_res2.copy()
        d_ffn_out = dX_res2.copy()

        dW2 = matmul(transpose(c["ffn_act"]), d_ffn_out)
        d_ffn_act = matmul(d_ffn_out, transpose(self.W2))
        d_ffn_pre = d_ffn_act * relu_grad(c["ffn_pre"])

        dW1 = matmul(transpose(c["X_res1"]), d_ffn_pre)
        dX_res1 += matmul(d_ffn_pre, transpose(self.W1))

        # 3. Residual 1 & Self-Attention backprop
        dX = dX_res1.copy()
        d_attn_out = dX_res1.copy()

        dW_o = matmul(transpose(c["context"]), d_attn_out)
        d_context = matmul(d_attn_out, transpose(self.W_o))

        d_attn_weights = matmul(d_context, transpose(c["V"]))
        dV = matmul(transpose(c["attn_weights"]), d_context)

        # Softmax backward per row — use boolean mask to zero gradients into masked positions
        d_masked = np.zeros_like(c["masked_scores"])
        for i in range(T):
            s = c["attn_weights"][i]
            da = d_attn_weights[i]
            d_masked[i] = s * (da - np.dot(da, s))
        d_masked[self.causal_mask_bool[:T, :T]] = 0.0   # safe: no float comparison

        d_raw = d_masked / np.sqrt(d_k)
        dQ = matmul(d_raw, c["K"])
        dK = matmul(transpose(d_raw), c["Q"])

        dW_q = matmul(transpose(c["X"]), dQ)
        dW_k = matmul(transpose(c["X"]), dK)
        dW_v = matmul(transpose(c["X"]), dV)

        dX += matmul(dQ, transpose(self.W_q))
        dX += matmul(dK, transpose(self.W_k))
        dX += matmul(dV, transpose(self.W_v))

        # 4. Parameter updates (Adam or clipped SGD)
        if use_adam:
            if not hasattr(self, "_adam_step"):
                self._adam_step = 0
                self._adam_m = {}
                self._adam_v = {}
                self._adam_emb_m = np.zeros_like(self.W_emb)
                self._adam_emb_v = np.zeros_like(self.W_emb)
                self._adam_pos_m = np.zeros_like(self.W_pos)
                self._adam_pos_v = np.zeros_like(self.W_pos)

            self._adam_step += 1
            step = self._adam_step
            beta1, beta2, eps = 0.9, 0.999, 1e-8

            # Token embedding update (sparse — only touched rows)
            for idx, token_id in enumerate(token_indices):
                g = dX[idx]
                self._adam_emb_m[token_id] = beta1 * self._adam_emb_m[token_id] + (1.0 - beta1) * g
                self._adam_emb_v[token_id] = beta2 * self._adam_emb_v[token_id] + (1.0 - beta2) * (g ** 2)
                m_h = self._adam_emb_m[token_id] / (1.0 - beta1 ** step)
                v_h = self._adam_emb_v[token_id] / (1.0 - beta2 ** step)
                self.W_emb[token_id] -= lr * m_h / (np.sqrt(v_h) + eps)

            # Positional embedding update (sparse — only touched positions 0..T-1)
            for pos in range(T):
                g = dX[pos]
                self._adam_pos_m[pos] = beta1 * self._adam_pos_m[pos] + (1.0 - beta1) * g
                self._adam_pos_v[pos] = beta2 * self._adam_pos_v[pos] + (1.0 - beta2) * (g ** 2)
                m_h = self._adam_pos_m[pos] / (1.0 - beta1 ** step)
                v_h = self._adam_pos_v[pos] / (1.0 - beta2 ** step)
                self.W_pos[pos] -= lr * m_h / (np.sqrt(v_h) + eps)

            param_grads = [
                ("W_out", self.W_out, dW_out),
                ("b_out", self.b_out, db_out),
                ("W1", self.W1, dW1),
                ("W2", self.W2, dW2),
                ("W_o", self.W_o, dW_o),
                ("W_q", self.W_q, dW_q),
                ("W_k", self.W_k, dW_k),
                ("W_v", self.W_v, dW_v),
            ]
            for name, param, grad in param_grads:
                np.clip(grad, -5.0, 5.0, out=grad)
                if name not in self._adam_m:
                    self._adam_m[name] = np.zeros_like(param)
                    self._adam_v[name] = np.zeros_like(param)
                self._adam_m[name] = beta1 * self._adam_m[name] + (1.0 - beta1) * grad
                self._adam_v[name] = beta2 * self._adam_v[name] + (1.0 - beta2) * (grad ** 2)
                m_h = self._adam_m[name] / (1.0 - beta1 ** step)
                v_h = self._adam_v[name] / (1.0 - beta2 ** step)
                param -= lr * m_h / (np.sqrt(v_h) + eps)
        else:
            for idx, token_id in enumerate(token_indices):
                self.W_emb[token_id] -= lr * dX[idx]
            for pos in range(T):
                self.W_pos[pos] -= lr * dX[pos]

            for W, dW in [
                (self.W_out, dW_out), (self.b_out, db_out),
                (self.W1, dW1), (self.W2, dW2),
                (self.W_o, dW_o), (self.W_q, dW_q),
                (self.W_k, dW_k), (self.W_v, dW_v)
            ]:
                np.clip(dW, -5.0, 5.0, out=dW)
                W -= lr * dW
        return loss

    def save_checkpoint(self, filepath: str, vocab: list[str] | None = None):
        """Save weights AND Adam optimizer state to compressed .npz archive."""
        save_dict = {
            "vocab_size": self.vocab_size,
            "hidden_size": self.hidden_size,
            "seq_len": self.seq_len,
            "W_emb": self.W_emb,
            "W_pos": self.W_pos,
            "W_q": self.W_q,
            "W_k": self.W_k,
            "W_v": self.W_v,
            "W_o": self.W_o,
            "W1": self.W1,
            "W2": self.W2,
            "W_out": self.W_out,
            "b_out": self.b_out,
        }
        # Persist Adam state so resuming doesn't cause the bias-correction spike
        if hasattr(self, "_adam_step"):
            save_dict["adam_step"] = np.array(self._adam_step)
            save_dict["adam_emb_m"] = self._adam_emb_m
            save_dict["adam_emb_v"] = self._adam_emb_v
            save_dict["adam_pos_m"] = self._adam_pos_m
            save_dict["adam_pos_v"] = self._adam_pos_v
            for name, m_arr in self._adam_m.items():
                save_dict[f"adam_m__{name}"] = m_arr
                save_dict[f"adam_v__{name}"] = self._adam_v[name]
        if vocab is not None:
            save_dict["vocab"] = np.array(vocab, dtype=object)
        np.savez_compressed(filepath, **save_dict)

    @classmethod
    def load_checkpoint(cls, filepath: str) -> "TinyTransformer":
        """Load weights (and Adam optimizer state if present) from .npz archive."""
        data = np.load(filepath, allow_pickle=True)
        model = cls(
            vocab_size=int(data["vocab_size"]),
            hidden_size=int(data["hidden_size"]),
            seq_len=int(data["seq_len"])
        )
        for key in ["W_emb", "W_q", "W_k", "W_v", "W_o", "W1", "W2", "W_out", "b_out"]:
            setattr(model, key, data[key])
        # Older checkpoints won't have W_pos — zero init is safe default
        if "W_pos" in data:
            model.W_pos = data["W_pos"]
        if "vocab" in data:
            model.vocab = [str(v) for v in data["vocab"]]
        # Restore Adam state so resumed training continues bias-correction from where it left off
        if "adam_step" in data:
            model._adam_step = int(data["adam_step"])
            model._adam_emb_m = data["adam_emb_m"]
            model._adam_emb_v = data["adam_emb_v"]
            model._adam_pos_m = data["adam_pos_m"] if "adam_pos_m" in data else np.zeros_like(model.W_pos)
            model._adam_pos_v = data["adam_pos_v"] if "adam_pos_v" in data else np.zeros_like(model.W_pos)
            model._adam_m = {}
            model._adam_v = {}
            for k in data.files:
                if k.startswith("adam_m__"):
                    name = k[len("adam_m__"):]
                    model._adam_m[name] = data[k]
                    model._adam_v[name] = data[f"adam_v__{name}"]
        return model
