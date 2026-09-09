"""
walkthrough_educational.py - Step-by-Step Hand-Verifiable Numerical Walkthrough
==============================================================================
Demonstrates ONE complete forward pass through the TinyTransformer with tiny dimensions:
  - Sequence length: 3 characters (e.g. "c", "a", "t")
  - Hidden dimension: 2 features
  - Vocabulary size: 4 tokens (e.g. ['c', 'a', 't', 's'])

You can verify every single matrix multiplication and addition on a piece of paper!
"""

import numpy as np
from model import softmax, relu, matmul, transpose

def run_numerical_walkthrough():
    print("=" * 70)
    print("      TINYTRANSFORMER HAND-VERIFIABLE NUMERICAL WALKTHROUGH")
    print("=" * 70)
    
    # -------------------------------------------------------------------------
    # Setup Tiny Dimensions
    # -------------------------------------------------------------------------
    vocab = ['c', 'a', 't', 's']
    vocab_size = 4
    hidden_size = 2
    T = 3  # Input: "cat" -> token IDs [0, 1, 2]
    
    print(f"\n[Configuration]")
    print(f"  Vocabulary: {vocab} (vocab_size={vocab_size})")
    print(f"  Hidden Size (d_k): {hidden_size}")
    print(f"  Input Tokens: ['c', 'a', 't'] -> Token IDs [0, 1, 2] (T={T})\n")
    
    # Fixed small weights for crystal-clear manual verification
    np.random.seed(42)
    
    # 1. Embedding Matrix W_emb: [vocab_size, hidden_size] = [4, 2]
    W_emb = np.array([
        [0.5, -0.2],  # 'c' (id 0)
        [0.1,  0.8],  # 'a' (id 1)
        [-0.4, 0.6],  # 't' (id 2)
        [0.3,  0.3],  # 's' (id 3)
    ], dtype=np.float64)
    
    print("1. Embedding Matrix W_emb [4, 2]:")
    print(W_emb)
    
    # Input tokens: [0, 1, 2]
    # Lookup X: [T, hidden_size] = [3, 2]
    X = W_emb[[0, 1, 2]]
    print("\n   Input Embedded Representations X [3, 2]:")
    print(X)
    
    # 2. Self-Attention Projection Matrices: all [2, 2]
    W_q = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)   # Identity for simplicity
    W_k = np.array([[0.5, 0.5], [-0.5, 0.5]], dtype=np.float64)
    W_v = np.array([[0.8, 0.2], [0.1, 0.9]], dtype=np.float64)
    W_o = np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float64)
    
    print("\n2. Attention Weight Projections [2, 2]:")
    print("   W_q:\n", W_q)
    print("   W_k:\n", W_k)
    print("   W_v:\n", W_v)
    print("   W_o:\n", W_o)
    
    # Q, K, V Projections: [3, 2]
    Q = matmul(X, W_q)
    K = matmul(X, W_k)
    V = matmul(X, W_v)
    
    print("\n3. Linear Projections (Q = X @ W_q, K = X @ W_k, V = X @ W_v) [3, 2]:")
    print("   Q:\n", Q)
    print("   K:\n", K)
    print("   V:\n", V)
    
    # Raw Attention Scores: Q @ K.T / sqrt(hidden_size) -> [3, 3]
    d_k = float(hidden_size)
    raw_scores = matmul(Q, transpose(K)) / np.sqrt(d_k)
    print(f"\n4. Raw Dot-Product Scores (Q @ K.T / sqrt({d_k})) [3, 3]:")
    print(raw_scores)
    
    # Causal Mask
    mask = np.array([
        [0.0, -1e9, -1e9],
        [0.0,  0.0, -1e9],
        [0.0,  0.0,  0.0]
    ], dtype=np.float64)
    
    masked_scores = raw_scores + mask
    print("\n5. Causal Mask Applied (Future positions set to -1e9) [3, 3]:")
    print(masked_scores)
    
    # Softmax row-wise to get Attention Weights
    attn_weights = softmax(masked_scores, axis=-1)
    print("\n6. Softmax Row-wise -> Attention Weights [3, 3] (Rows sum to 1.0):")
    print(attn_weights)
    print("   Row sums:", np.sum(attn_weights, axis=-1))
    
    # Context Vector: attn_weights @ V -> [3, 2]
    context = matmul(attn_weights, V)
    print("\n7. Context Vector (attn_weights @ V) [3, 2]:")
    print(context)
    
    # Output Projection: context @ W_o -> [3, 2]
    attn_out = matmul(context, W_o)
    
    # Residual Connection 1: X_res1 = X + attn_out -> [3, 2]
    X_res1 = X + attn_out
    print("\n8. Residual 1 (X + attn_out) [3, 2]:")
    print(X_res1)
    
    # Feed-Forward Network: W1: [2, 4], W2: [4, 2]
    W1 = np.array([
        [ 0.6, -0.4,  0.8,  0.1],
        [-0.5,  0.7, -0.2,  0.9]
    ], dtype=np.float64)
    
    W2 = np.array([
        [0.5, -0.3],
        [0.2,  0.8],
        [0.4,  0.1],
        [-0.6, 0.5]
    ], dtype=np.float64)
    
    ffn_pre_act = matmul(X_res1, W1)       # [3, 4]
    ffn_hidden = relu(ffn_pre_act)          # [3, 4]
    ffn_out = matmul(ffn_hidden, W2)       # [3, 2]
    
    print("\n9. Feed-Forward Network (FFN):")
    print("   Pre-ReLU (X_res1 @ W1) [3, 4]:\n", ffn_pre_act)
    print("   Post-ReLU Activation [3, 4]:\n", ffn_hidden)
    print("   FFN Output (relu @ W2) [3, 2]:\n", ffn_out)
    
    # Residual Connection 2: X_res2 = X_res1 + ffn_out -> [3, 2]
    X_res2 = X_res1 + ffn_out
    print("\n10. Residual 2 (X_res1 + ffn_out) [3, 2]:")
    print(X_res2)
    
    # Output Head: W_out [2, 4], b_out [4]
    W_out = np.array([
        [0.2, -0.1,  0.4,  0.9],
        [0.1,  0.5, -0.3,  0.7]
    ], dtype=np.float64)
    b_out = np.array([0.0, 0.0, 0.0, 0.1], dtype=np.float64)
    
    logits = matmul(X_res2, W_out) + b_out  # [3, 4]
    print("\n11. Final Logits (X_res2 @ W_out + b_out) [3, 4]:")
    print(logits)
    
    # Next Character Prediction for the last token position
    last_logit = logits[-1, :]              # [4]
    next_char_probs = softmax(last_logit)   # [4]
    
    print("\n12. Next-Character Probability Distribution (Position 3):")
    for ch, p in zip(vocab, next_char_probs):
        bar = "#" * int(p * 30)
        print(f"    Token '{ch}': {p * 100:5.1f}% | {bar}")
        
    best_pred = vocab[np.argmax(next_char_probs)]
    print(f"\n=> Predicted Next Character: '{best_pred}' (Probability: {np.max(next_char_probs)*100:.1f}%)")
    
    # Target character loss calculation
    target_token = 3  # 's' (making the word "cats")
    target_prob = next_char_probs[target_token]
    loss = -np.log(target_prob)
    print(f"\nIf target next character is '{vocab[target_token]}':")
    print(f"  Target Probability: {target_prob * 100:.2f}%")
    print(f"  Cross-Entropy Loss (-log(p)): {loss:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    run_numerical_walkthrough()
