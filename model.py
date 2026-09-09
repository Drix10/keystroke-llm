"""
model.py - Educational Character-Level TinyTransformer
======================================================
Designed for beginners learning Transformer mechanics from scratch.
Zero black-box frameworks: pure Python + NumPy.

Architecture:
  Tokens -> Embeddings (W_emb) -> Linear Projections (W_q, W_k, W_v)
         -> Attention Scores -> Causal Mask -> Softmax -> Attention Weights
         -> Context Vector -> Output Projection (W_o) -> Residual 1
         -> Feed-Forward Network (W1 -> ReLU -> W2) -> Residual 2
         -> Output Head (W_out + b_out) -> Logits -> Probabilities
"""

import numpy as np


# -----------------------------------------------------------------------------
# 1. Fundamental Math Helpers
# -----------------------------------------------------------------------------

def randn(rows: int, cols: int, scale: float = 0.1) -> np.ndarray:
    """
    Initializes a 2D weight matrix with values uniformly distributed in [-scale, +scale].
    
    Args:
        rows: Number of rows in matrix.
        cols: Number of columns in matrix.
        scale: Boundary amplitude for initialization.
        
    Returns:
        np.ndarray of shape [rows, cols].
    """
    return np.random.uniform(-scale, scale, size=(rows, cols)).astype(np.float64)


def matmul(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """
    Matrix multiplication wrapper: A @ B.
    
    Shapes:
        A: [M, K]
        B: [K, N]
        Output: [M, N]
    """
    assert A.shape[-1] == B.shape[0], (
        f"Incompatible matrix shapes for matmul: {A.shape} and {B.shape}"
    )
    return np.matmul(A, B)


def transpose(A: np.ndarray) -> np.ndarray:
    """
    Matrix transpose: swaps the last two dimensions.
    
    Shapes:
        A: [M, N]
        Output: [N, M]
    """
    return np.swapaxes(A, -1, -2)


def relu(x: np.ndarray) -> np.ndarray:
    """
    Rectified Linear Unit: element-wise max(0, x).
    
    Shapes:
        x: Any shape [...]
        Output: Same shape [...]
    """
    return np.maximum(0.0, x)


def relu_grad(x: np.ndarray) -> np.ndarray:
    """
    Gradient of ReLU: 1 where x > 0, else 0.
    """
    return (x > 0.0).astype(np.float64)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """
    Numerically stable Softmax activation.
    Subtracts the maximum value along the target axis to prevent exp() overflow.
    
    Formula:
        softmax(x)_i = exp(x_i - max(x)) / sum(exp(x_j - max(x)))
        
    Shapes:
        x: [...]
        Output: [...], sums to 1.0 along `axis`.
    """
    # Subtract max along axis for numerical stability
    shifted_x = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted_x)
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


# -----------------------------------------------------------------------------
# 2. TinyTransformer Class
# -----------------------------------------------------------------------------

class TinyTransformer:
    """
    A single-layer, autoregressive Character-Level Transformer.
    
    Key Hyperparameters:
        vocab_size: Number of unique tokens in character vocabulary (e.g. 96).
        hidden_size: Embedding and latent feature dimension (e.g. 32 or 64).
        seq_len: Context window length (e.g. 12 or 16).
    """

    def __init__(self, vocab_size: int = 96, hidden_size: int = 32, seq_len: int = 12, init_scale: float = 0.1):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.init_scale = init_scale
        
        # ---------------------------------------------------------------------
        # Weight Matrices
        # ---------------------------------------------------------------------
        # 1. Token Embeddings: maps token ID (0..vocab_size-1) to a dense vector
        # Shape: [vocab_size, hidden_size]
        self.W_emb = randn(self.vocab_size, self.hidden_size, scale=self.init_scale)
        
        # 2. Self-Attention Projections: Queries, Keys, Values, and Output
        # Shapes: [hidden_size, hidden_size]
        self.W_q = randn(self.hidden_size, self.hidden_size, scale=self.init_scale)
        self.W_k = randn(self.hidden_size, self.hidden_size, scale=self.init_scale)
        self.W_v = randn(self.hidden_size, self.hidden_size, scale=self.init_scale)
        self.W_o = randn(self.hidden_size, self.hidden_size, scale=self.init_scale)
        
        # 3. Feed-Forward Network (FFN):
        # W1 expands hidden_size -> 2*hidden_size
        # W2 projects 2*hidden_size -> hidden_size
        self.W1 = randn(self.hidden_size, 2 * self.hidden_size, scale=self.init_scale)
        self.W2 = randn(2 * self.hidden_size, self.hidden_size, scale=self.init_scale)
        
        # 4. Final Output Head: projects hidden representations to vocabulary logits
        # Shape: [hidden_size, vocab_size]
        self.W_out = randn(self.hidden_size, self.vocab_size, scale=self.init_scale)
        # Output bias: [vocab_size]
        self.b_out = np.zeros(self.vocab_size, dtype=np.float64)
        
        # 5. Causal Mask: Lower triangular matrix with 0s on & below diagonal,
        # and -infinity above diagonal. Pre-built for sequence length.
        # Shape: [seq_len, seq_len]
        self.causal_mask = self._build_causal_mask(self.seq_len)
        
        # Cache for inspection and visualization
        self.last_attn_weights = None
        self.last_context = None
        self.last_cache = {}

    @staticmethod
    def _build_causal_mask(size: int) -> np.ndarray:
        """
        Builds an autoregressive causal mask of shape [size, size].
        Positions j > i get -inf, preventing token i from looking into future tokens.
        """
        mask = np.zeros((size, size), dtype=np.float64)
        for i in range(size):
            for j in range(size):
                if j > i:
                    mask[i, j] = -1e9  # Effectively -infinity for softmax
        return mask

    def forward(self, token_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
        """
        Performs one forward pass through the entire transformer.
        
        Args:
            token_indices: List of integer token IDs. Length T must be <= self.seq_len.
            
        Returns:
            logits: Output logits for all sequence positions, shape [T, vocab_size].
            next_char_probs: Softmax probability distribution for next token, shape [vocab_size].
        """
        T = len(token_indices)
        assert 1 <= T <= self.seq_len, f"Sequence length {T} must be between 1 and {self.seq_len}"
        
        # ---------------------------------------------------------------------
        # Step 1: Token Embeddings Lookup
        # Look up embedding vector for each input token.
        # Shape: [T, hidden_size]
        # ---------------------------------------------------------------------
        X = self.W_emb[token_indices]  # [T, hidden_size]
        
        # ---------------------------------------------------------------------
        # Step 2: Linear Projections for Self-Attention (Q, K, V)
        # Q = X @ W_q: [T, hidden_size]
        # K = X @ W_k: [T, hidden_size]
        # V = X @ W_v: [T, hidden_size]
        # ---------------------------------------------------------------------
        Q = matmul(X, self.W_q)  # [T, hidden_size]
        K = matmul(X, self.W_k)  # [T, hidden_size]
        V = matmul(X, self.W_v)  # [T, hidden_size]
        
        # ---------------------------------------------------------------------
        # Step 3: Raw Attention Scores
        # Dot product between queries and keys, scaled by sqrt(hidden_size).
        # Q @ K^T: [T, hidden_size] @ [hidden_size, T] -> [T, T]
        # ---------------------------------------------------------------------
        d_k = float(self.hidden_size)
        raw_scores = matmul(Q, transpose(K)) / np.sqrt(d_k)  # [T, T]
        
        # ---------------------------------------------------------------------
        # Step 4: Apply Causal Mask
        # Future positions are masked with -infinity so tokens cannot cheat.
        # Slice mask to current sequence length T.
        # ---------------------------------------------------------------------
        mask = self.causal_mask[:T, :T]                      # [T, T]
        masked_scores = raw_scores + mask                    # [T, T]
        
        # ---------------------------------------------------------------------
        # Step 5: Softmax to get Attention Weights
        # Row-wise softmax produces attention weights summing to 1.0.
        # ---------------------------------------------------------------------
        attn_weights = softmax(masked_scores, axis=-1)       # [T, T]
        self.last_attn_weights = attn_weights                # Save for visualization
        
        # ---------------------------------------------------------------------
        # Step 6: Context Vector (Weighted sum of Values)
        # attn_weights @ V: [T, T] @ [T, hidden_size] -> [T, hidden_size]
        # ---------------------------------------------------------------------
        context = matmul(attn_weights, V)                    # [T, hidden_size]
        self.last_context = context
        
        # ---------------------------------------------------------------------
        # Step 7: Output Projection
        # Project context back into model hidden dimension.
        # context @ W_o: [T, hidden_size] @ [hidden_size, hidden_size]
        # ---------------------------------------------------------------------
        attn_out = matmul(context, self.W_o)                 # [T, hidden_size]
        
        # ---------------------------------------------------------------------
        # Step 8: Residual Connection 1
        # Simple addition of input X and attention output.
        # ---------------------------------------------------------------------
        X_res1 = X + attn_out                                # [T, hidden_size]
        
        # ---------------------------------------------------------------------
        # Step 9: Feed-Forward Network (FFN)
        # Expand: X_res1 @ W1 -> [T, 2*hidden_size]
        # Activation: ReLU
        # Project: ffn_hidden @ W2 -> [T, hidden_size]
        # ---------------------------------------------------------------------
        ffn_pre_act = matmul(X_res1, self.W1)               # [T, 2*hidden_size]
        ffn_hidden = relu(ffn_pre_act)                       # [T, 2*hidden_size]
        ffn_out = matmul(ffn_hidden, self.W2)                # [T, hidden_size]
        
        # ---------------------------------------------------------------------
        # Step 10: Residual Connection 2
        # Simple addition of X_res1 and FFN output.
        # ---------------------------------------------------------------------
        X_res2 = X_res1 + ffn_out                            # [T, hidden_size]
        
        # ---------------------------------------------------------------------
        # Step 11: Output Head (Logits)
        # Project representation to full vocabulary size + add bias.
        # X_res2 @ W_out: [T, hidden_size] @ [hidden_size, vocab_size]
        # ---------------------------------------------------------------------
        logits = matmul(X_res2, self.W_out) + self.b_out     # [T, vocab_size]
        
        # ---------------------------------------------------------------------
        # Step 12: Next Character Probability Distribution
        # The prediction for the next character comes from the last token's logits.
        # ---------------------------------------------------------------------
        last_logit = logits[-1, :]                           # [vocab_size]
        next_char_probs = softmax(last_logit)                # [vocab_size]
        
        # Cache intermediate states for backpropagation or inspection
        self.last_cache = {
            "token_indices": token_indices,
            "X": X,
            "Q": Q,
            "K": K,
            "V": V,
            "raw_scores": raw_scores,
            "masked_scores": masked_scores,
            "attn_weights": attn_weights,
            "context": context,
            "attn_out": attn_out,
            "X_res1": X_res1,
            "ffn_pre_act": ffn_pre_act,
            "ffn_hidden": ffn_hidden,
            "ffn_out": ffn_out,
            "X_res2": X_res2,
            "logits": logits,
            "next_char_probs": next_char_probs,
        }
        
        return logits, next_char_probs

    def compute_loss(self, next_char_probs: np.ndarray, target_token: int) -> float:
        """
        Cross-entropy loss: Negative Log-Likelihood (NLL) of the correct next character.
        
        Formula:
            Loss = -log(prob[target_token])
        """
        eps = 1e-12  # Prevent log(0)
        target_prob = np.clip(next_char_probs[target_token], eps, 1.0)
        return float(-np.log(target_prob))

    # -------------------------------------------------------------------------
    # Training Mode A: Educational Heuristic Update
    # -------------------------------------------------------------------------
    def train_step_heuristic(self, token_indices: list[int], target_token: int, lr: float = 0.05) -> float:
        """
        Simplified / Heuristic pedagogical update:
        Only updates the embedding of the input tokens and the output head weights
        for the target token based on prediction error (prob - 1.0).
        This keeps training 100% transparent and beginner-friendly before introducing
        full multi-matrix chain rule backpropagation.
        """
        _, probs = self.forward(token_indices)
        loss = self.compute_loss(probs, target_token)
        
        # Educational / Simplified update (updates embeddings & output head only)
        # 1. Prediction error vector for all vocabulary tokens:
        # d_logits = probs - one_hot(target)
        d_logits = probs.copy()
        d_logits[target_token] -= 1.0  # target error is (prob - 1.0), others are (prob - 0.0)
        
        # Representation of the final sequence position
        last_hidden = self.last_cache["X_res2"][-1]  # [hidden_size]
        
        # 2. Update output head (W_out and b_out)
        # W_out[:, c] moves proportional to -error_c * last_hidden
        dW_out = np.outer(last_hidden, d_logits)     # [hidden_size, vocab_size]
        np.clip(dW_out, -2.0, 2.0, out=dW_out)
        self.W_out -= lr * dW_out
        self.b_out -= lr * np.clip(d_logits, -2.0, 2.0)
        
        # 3. Update input embeddings slightly towards target direction
        target_dir = self.W_out[:, target_token]
        norm_dir = target_dir / (np.linalg.norm(target_dir) + 1e-8)
        error = probs[target_token] - 1.0
        for t in token_indices:
            self.W_emb[t] -= lr * 0.05 * error * norm_dir
            
        return loss

    # -------------------------------------------------------------------------
    # Training Mode B: Full Analytical Backpropagation
    # -------------------------------------------------------------------------
    def train_step_backprop(self, token_indices: list[int], target_token: int, lr: float = 0.01) -> float:
        """
        Full analytical backpropagation through all Transformer components:
        Output Head -> Residual 2 -> FFN (W2, ReLU, W1) -> Residual 1
        -> Attention Out (W_o) -> Context -> Softmax -> Scores -> Q, K, V -> W_emb.
        """
        _, probs = self.forward(token_indices)
        loss = self.compute_loss(probs, target_token)
        
        cache = self.last_cache
        T = len(token_indices)
        d_k = float(self.hidden_size)
        
        # 1. Gradient of cross-entropy w.r.t last logits
        # dL/dz = probs - y_onehot
        d_logits = np.zeros_like(cache["logits"])               # [T, vocab_size]
        d_logits[-1, :] = probs.copy()
        d_logits[-1, target_token] -= 1.0                       # [vocab_size]
        
        # 2. Gradients for W_out and b_out
        # logits = X_res2 @ W_out + b_out
        dW_out = matmul(transpose(cache["X_res2"]), d_logits)    # [hidden_size, vocab_size]
        db_out = np.sum(d_logits, axis=0)                        # [vocab_size]
        
        # Backprop into X_res2
        dX_res2 = matmul(d_logits, transpose(self.W_out))       # [T, hidden_size]
        
        # 3. Residual 2: X_res2 = X_res1 + ffn_out
        dX_res1 = dX_res2.copy()                                # from direct residual path
        d_ffn_out = dX_res2.copy()                              # [T, hidden_size]
        
        # 4. FFN: ffn_out = ffn_hidden @ W2
        dW2 = matmul(transpose(cache["ffn_hidden"]), d_ffn_out) # [2*hidden, hidden]
        d_ffn_hidden = matmul(d_ffn_out, transpose(self.W2))    # [T, 2*hidden]
        
        # ReLU activation backprop
        d_ffn_pre_act = d_ffn_hidden * relu_grad(cache["ffn_pre_act"]) # [T, 2*hidden]
        
        # ffn_pre_act = X_res1 @ W1
        dW1 = matmul(transpose(cache["X_res1"]), d_ffn_pre_act) # [hidden, 2*hidden]
        dX_res1 += matmul(d_ffn_pre_act, transpose(self.W1))    # accumulate into X_res1
        
        # 5. Residual 1: X_res1 = X + attn_out
        dX = dX_res1.copy()                                     # from direct residual path
        d_attn_out = dX_res1.copy()                             # [T, hidden_size]
        
        # 6. Attention Out: attn_out = context @ W_o
        dW_o = matmul(transpose(cache["context"]), d_attn_out)  # [hidden, hidden]
        d_context = matmul(d_attn_out, transpose(self.W_o))     # [T, hidden]
        
        # 7. Context = attn_weights @ V
        d_attn_weights = matmul(d_context, transpose(cache["V"])) # [T, T]
        dV = matmul(transpose(cache["attn_weights"]), d_context)  # [T, hidden]
        
        # 8. Softmax backward: d_masked_scores
        # For each row i: dS_i = S_i * (dA_i - sum(dA_i * S_i))
        d_masked_scores = np.zeros_like(cache["masked_scores"])
        for i in range(T):
            s_i = cache["attn_weights"][i]                      # [T]
            da_i = d_attn_weights[i]                            # [T]
            sum_prod = np.dot(da_i, s_i)
            d_masked_scores[i] = s_i * (da_i - sum_prod)
            
        # Mask backward (zeros out gradient where mask was -inf)
        d_masked_scores[cache["masked_scores"] <= -1e8] = 0.0
        
        # 9. Raw scores: raw_scores = (Q @ K.T) / sqrt(d_k)
        d_raw_scores = d_masked_scores / np.sqrt(d_k)           # [T, T]
        
        dQ = matmul(d_raw_scores, cache["K"])                   # [T, hidden]
        dK = matmul(transpose(d_raw_scores), cache["Q"])        # [T, hidden]
        
        # 10. Q, K, V linear projections: Q = X @ W_q, etc.
        dW_q = matmul(transpose(cache["X"]), dQ)                # [hidden, hidden]
        dW_k = matmul(transpose(cache["X"]), dK)                # [hidden, hidden]
        dW_v = matmul(transpose(cache["X"]), dV)                # [hidden, hidden]
        
        dX += matmul(dQ, transpose(self.W_q))
        dX += matmul(dK, transpose(self.W_k))
        dX += matmul(dV, transpose(self.W_v))
        
        # 11. Embeddings: accumulate dX into W_emb for each token
        for idx, token_id in enumerate(token_indices):
            self.W_emb[token_id] -= lr * dX[idx]
            
        # 12. Apply updates with gradient clipping for numerical stability
        for W, dW in [
            (self.W_out, dW_out), (self.b_out, db_out),
            (self.W1, dW1), (self.W2, dW2),
            (self.W_o, dW_o), (self.W_q, dW_q),
            (self.W_k, dW_k), (self.W_v, dW_v)
        ]:
            np.clip(dW, -5.0, 5.0, out=dW)
            W -= lr * dW
            
        return loss

    # -------------------------------------------------------------------------
    # Checkpointing / Persistence
    # -------------------------------------------------------------------------
    def save_checkpoint(self, filepath: str):
        """Saves all weight matrices to a readable .npz file."""
        np.savez_compressed(
            filepath,
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            seq_len=self.seq_len,
            W_emb=self.W_emb,
            W_q=self.W_q,
            W_k=self.W_k,
            W_v=self.W_v,
            W_o=self.W_o,
            W1=self.W1,
            W2=self.W2,
            W_out=self.W_out,
            b_out=self.b_out,
        )

    @classmethod
    def load_checkpoint(cls, filepath: str) -> "TinyTransformer":
        """Loads weights from a saved .npz checkpoint."""
        data = np.load(filepath)
        model = cls(
            vocab_size=int(data["vocab_size"]),
            hidden_size=int(data["hidden_size"]),
            seq_len=int(data["seq_len"])
        )
        model.W_emb = data["W_emb"]
        model.W_q = data["W_q"]
        model.W_k = data["W_k"]
        model.W_v = data["W_v"]
        model.W_o = data["W_o"]
        model.W1 = data["W1"]
        model.W2 = data["W2"]
        model.W_out = data["W_out"]
        model.b_out = data["b_out"]
        return model
