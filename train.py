"""
train.py - Educational Training Loop for TinyTransformer
========================================================
Builds a character-level sliding-window dataset, trains the model,
demonstrates loss reduction, and saves inspectable checkpoints.

Usage:
  python train.py --epochs 20 --lr 0.02 --mode heuristic
  python train.py --epochs 30 --lr 0.01 --mode backprop
  python train.py --resume checkpoints/checkpoint_epoch_20.npz --epochs 10
"""

import argparse
import os
import time
import numpy as np

from key_mapper import CharTokenizer, get_default_vocab
from model import TinyTransformer


def build_sliding_window_dataset(text: str, tokenizer: CharTokenizer, seq_len: int) -> list[tuple[list[int], int]]:
    """
    Creates (context, target) pairs using a sliding window:
    Given `seq_len` previous characters -> predict the (seq_len + 1)-th character.
    
    Example with seq_len=4 on "hello":
      (['h', 'e', 'l', 'l'], 'o')
    """
    token_ids = tokenizer.encode(text)
    examples = []
    
    for i in range(len(token_ids) - seq_len):
        context = token_ids[i : i + seq_len]
        target = token_ids[i + seq_len]
        examples.append((context, target))
        
    return examples


def evaluate_sample_predictions(model: TinyTransformer, tokenizer: CharTokenizer, sample_prompts: list[str], top_k: int = 3):
    """Prints sample predictions to visually verify model learning."""
    print("\n" + "=" * 60)
    print("Sample Context Next-Character Predictions:")
    print("=" * 60)
    for prompt in sample_prompts:
        prompt_tokens = tokenizer.encode(prompt)[-model.seq_len:]
        _, probs = model.forward(prompt_tokens)
        
        # Get top-k predicted character IDs
        top_indices = np.argsort(probs)[::-1][:top_k]
        pred_summary = []
        for rank, idx in enumerate(top_indices, 1):
            ch = tokenizer.decode_id(idx)
            repr_ch = repr(ch) if ch in (' ', '\n', '\t') else f"'{ch}'"
            p = probs[idx] * 100.0
            pred_summary.append(f"#{rank}: {repr_ch} ({p:.1f}%)")
            
        print(f"Context: {prompt!r:<20} -> Predictions: {', '.join(pred_summary)}")
    print("=" * 60 + "\n")


def train():
    parser = argparse.ArgumentParser(description="Train Educational Character Transformer")
    parser.add_argument("--data", type=str, default="data/sample_training_text.txt", help="Path to training text file")
    parser.add_argument("--epochs", type=int, default=15, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.015, help="Learning rate")
    parser.add_argument("--seq-len", type=int, default=12, help="Context sequence length (characters)")
    parser.add_argument("--hidden-size", type=int, default=32, help="Embedding and hidden dimension")
    parser.add_argument("--mode", type=str, choices=["heuristic", "backprop"], default="heuristic",
                        help="Training mode: 'heuristic' (pedagogical) or 'backprop' (full multi-matrix gradients)")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--resume", type=str, default=None, help="Path to existing checkpoint to continue training")
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    
    # 1. Load Training Text
    if not os.path.exists(args.data):
        raise FileNotFoundError(f"Training data file not found at {args.data}")
    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()
        
    print(f"[Dataset] Loaded {len(text)} characters from {args.data}")
    
    # 2. Build Vocabulary (Guaranteed inclusion of full keyboard alphabet + symbols)
    base_vocab = get_default_vocab()
    # Add any extra unique characters from the text if not already present
    for ch in text:
        if ch not in base_vocab:
            base_vocab.append(ch)
            
    tokenizer = CharTokenizer(vocab=base_vocab)
    print(f"[Tokenizer] Vocabulary size: {tokenizer.vocab_size} tokens")
    
    # 3. Create Sliding Window Examples
    dataset = build_sliding_window_dataset(text, tokenizer, seq_len=args.seq_len)
    print(f"[Dataset] Created {len(dataset)} training samples (Context: {args.seq_len} chars -> Target: 1 char)")
    
    # 4. Initialize or Resume Model
    if args.resume:
        print(f"[Model] Resuming from checkpoint: {args.resume}")
        model = TinyTransformer.load_checkpoint(args.resume)
    else:
        print(f"[Model] Initializing fresh TinyTransformer (vocab={tokenizer.vocab_size}, hidden={args.hidden_size}, seq_len={args.seq_len})")
        model = TinyTransformer(
            vocab_size=tokenizer.vocab_size,
            hidden_size=args.hidden_size,
            seq_len=args.seq_len,
            init_scale=0.1
        )
        
    # Test sample prompts before training
    test_prompts = ["The qui", "def hel", "self at", "Kreo Hi"]
    print("\n[Baseline] Initial predictions before training:")
    evaluate_sample_predictions(model, tokenizer, test_prompts)
    
    # 5. Training Loop
    print(f"[Training] Starting training for {args.epochs} epochs in '{args.mode}' mode (lr={args.lr})...\n")
    start_time = time.time()
    
    for epoch in range(1, args.epochs + 1):
        # Shuffle dataset each epoch for better gradient updates
        indices = np.random.permutation(len(dataset))
        epoch_losses = []
        epoch_start = time.time()
        
        for idx in indices:
            context, target = dataset[idx]
            if args.mode == "heuristic":
                loss = model.train_step_heuristic(context, target, lr=args.lr)
            else:
                loss = model.train_step_backprop(context, target, lr=args.lr)
            epoch_losses.append(loss)
            
        avg_loss = float(np.mean(epoch_losses))
        epoch_duration = time.time() - epoch_start
        
        # Display progress
        print(f"Epoch {epoch:2d}/{args.epochs:2d} | Avg Loss: {avg_loss:.4f} | Time: {epoch_duration:.2f}s")
        
        # Periodic evaluation & checkpoint saving
        if epoch % 5 == 0 or epoch == args.epochs:
            ckpt_path = os.path.join(args.checkpoint_dir, f"model_{args.mode}_epoch_{epoch}.npz")
            model.save_checkpoint(ckpt_path)
            print(f"  -> Checkpoint saved to {ckpt_path}")
            
    total_time = time.time() - start_time
    print(f"\n[Done] Training completed in {total_time:.2f} seconds.")
    
    # Final evaluation
    print("\n[Evaluation] Final predictions after training:")
    evaluate_sample_predictions(model, tokenizer, test_prompts)
    
    # Save primary model checkpoint
    final_path = os.path.join(args.checkpoint_dir, "model_final.npz")
    model.save_checkpoint(final_path)
    print(f"[Checkpoint] Primary model saved to {final_path}")


if __name__ == "__main__":
    train()
