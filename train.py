"""Character-level sliding-window dataset preparation and training loop."""

import argparse
import csv
import os
import time
import numpy as np

from key_mapper import CharTokenizer, get_default_vocab
from model import TinyTransformer, softmax


def build_sliding_window_dataset(text: str, tokenizer: CharTokenizer, seq_len: int, stride: int = 3) -> list[tuple[list[int], list[int]]]:
    """Generates (input_seq, target_seq) pairs of length seq_len from text."""
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    token_ids = tokenizer.encode(text)
    if len(token_ids) <= seq_len:
        return []
    return [
        (token_ids[i : i + seq_len], token_ids[i + 1 : i + seq_len + 1])
        for i in range(0, len(token_ids) - seq_len, stride)
    ]


def evaluate_predictions(model: TinyTransformer, tokenizer: CharTokenizer, prompts: list[str], top_k: int = 3):
    print("\n" + "=" * 55)
    print("Sample Context Next-Character Predictions")
    print("=" * 55)
    for p in prompts:
        tokens = tokenizer.encode(p)[-model.seq_len:]
        _, probs = model.forward(tokens)
        top = np.argsort(probs)[::-1][:top_k]
        preds = []
        for idx in top:
            ch = tokenizer.decode_id(idx)
            rep = repr(ch) if ch in (" ", "\n", "\t") else f"'{ch}'"
            preds.append(f"{rep} ({probs[idx]*100:.1f}%)")
        print(f"  {p!r:<18} -> {', '.join(preds)}")
    print("=" * 55 + "\n")


def compute_val_loss(model: TinyTransformer, val_dataset: list[tuple[list[int], list[int]]]) -> float:
    """Run a forward-only pass over the validation set and return mean loss."""
    if not val_dataset:
        return float("nan")
    total = 0.0
    for ctx, tgt in val_dataset:
        _, probs = model.forward(ctx)
        probs_all = softmax(model.last_cache["logits"], axis=-1)
        y_seq = np.array(tgt, dtype=int)
        T = len(ctx)
        losses = -np.log(np.clip(probs_all[np.arange(T), y_seq], 1e-12, 1.0))
        total += float(np.mean(losses))
    return total / len(val_dataset)


def train():
    parser = argparse.ArgumentParser(description="Train TinyTransformer on character data")
    parser.add_argument("--data", type=str, default="data/sample_training_text.txt", help="Path to training text")
    parser.add_argument("--epochs", type=int, default=15, help="Epoch count (default: 15)")
    parser.add_argument("--lr", type=float, default=0.003, help="Learning rate (Adam default: 0.003)")
    parser.add_argument("--seq-len", type=int, default=48, help="Context length in characters (default: 48)")
    parser.add_argument("--hidden-size", type=int, default=64, help="Transformer hidden size (default: 64)")
    parser.add_argument("--stride", type=int, default=3, help="Sliding window stride for dataset construction")
    parser.add_argument("--mode", type=str, choices=["heuristic", "backprop"], default="backprop", help="Training mode")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Save directory")
    parser.add_argument("--resume", type=str, default=None, help="Resume from checkpoint file")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--val-split", type=float, default=0.1, help="Fraction of data held out for validation (0 to disable)")
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"Random seed set to {args.seed}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)

    if not os.path.exists(args.data):
        raise FileNotFoundError(
            f"Training data not found: '{args.data}'\n"
            f"  -> Make sure the path is correct, or pass --data <path/to/your/text.txt>"
        )

    with open(args.data, "r", encoding="utf-8") as f:
        text = f.read()

    if args.resume:
        print(f"Resuming model from {args.resume}")
        model = TinyTransformer.load_checkpoint(args.resume)
        if model.seq_len != args.seq_len:
            raise ValueError(
                f"Checkpoint sequence length ({model.seq_len}) does not match args.seq_len ({args.seq_len})."
            )
        if getattr(model, "vocab", None) is not None:
            tokenizer = CharTokenizer(vocab=model.vocab)
        else:
            vocab = get_default_vocab()
            for ch in text:
                if ch not in vocab:
                    vocab.append(ch)
            tokenizer = CharTokenizer(vocab=vocab)
        if model.vocab_size != tokenizer.vocab_size:
            raise ValueError(
                f"Checkpoint vocab size ({model.vocab_size}) does not match tokenizer vocab size ({tokenizer.vocab_size})."
            )
    else:
        # Build vocabulary from training text
        vocab = get_default_vocab()
        for ch in text:
            if ch not in vocab:
                vocab.append(ch)
        tokenizer = CharTokenizer(vocab=vocab)
        model = TinyTransformer(
            vocab_size=tokenizer.vocab_size,
            hidden_size=args.hidden_size,
            seq_len=args.seq_len
        )

    dataset = build_sliding_window_dataset(text, tokenizer, seq_len=args.seq_len, stride=args.stride)

    if not dataset:
        raise ValueError(
            f"Dataset is empty! Input text ({len(text)} chars) must contain more than seq_len ({args.seq_len}) characters."
        )

    val_size = int(len(dataset) * args.val_split) if args.val_split > 0 else 0
    train_dataset = dataset[: len(dataset) - val_size]
    val_dataset = dataset[len(dataset) - val_size :]
    print(f"Loaded {len(text)} characters | train: {len(train_dataset)}, val: {len(val_dataset)} samples | vocab={tokenizer.vocab_size}", flush=True)

    if not train_dataset:
        raise ValueError("Training split is empty after validation holdout. Use a longer text or reduce --val-split.")

    test_prompts = ["How are ", "Thank ", "Good ", "The qui", "def hel", "I am ", "What is ", "Please "]
    print("Initial baseline:", flush=True)
    evaluate_predictions(model, tokenizer, test_prompts)

    csv_path = os.path.join(args.checkpoint_dir, "loss_log.csv")
    csv_file = open(csv_path, "w", newline="", encoding="utf-8")
    try:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["epoch", "train_loss", "val_loss", "time_s"])

        print(f"Training for {args.epochs} epochs in '{args.mode}' mode (lr={args.lr}, stride={args.stride})...", flush=True)
        start = time.time()

        for epoch in range(1, args.epochs + 1):
            indices = np.random.permutation(len(train_dataset))
            losses = []
            ep_start = time.time()

            for idx in indices:
                ctx, tgt = train_dataset[idx]
                if args.mode == "heuristic":
                    loss = model.train_step_heuristic(ctx, tgt, lr=args.lr)
                else:
                    loss = model.train_step_backprop(ctx, tgt, lr=args.lr)
                losses.append(loss)

            avg_loss = float(np.mean(losses))
            val_loss = compute_val_loss(model, val_dataset)
            ep_time = time.time() - ep_start
            val_str = f"{val_loss:.4f}" if not np.isnan(val_loss) else "n/a"
            print(f"Epoch {epoch:2d}/{args.epochs:2d} | Train Loss: {avg_loss:.4f} | Val Loss: {val_str} | Time: {ep_time:.2f}s", flush=True)
            csv_writer.writerow([epoch, f"{avg_loss:.6f}", f"{val_loss:.6f}", f"{ep_time:.2f}"])
            csv_file.flush()

            if epoch % 5 == 0 or epoch == args.epochs:
                ckpt_path = os.path.join(args.checkpoint_dir, f"model_{args.mode}_epoch_{epoch}.npz")
                model.save_checkpoint(ckpt_path, vocab=tokenizer.vocab)
    finally:
        csv_file.close()
    print(f"\nFinished training in {time.time() - start:.2f}s", flush=True)
    print(f"Loss log saved to {csv_path}", flush=True)
    evaluate_predictions(model, tokenizer, test_prompts)

    final_path = os.path.join(args.checkpoint_dir, "model_final.npz")
    model.save_checkpoint(final_path, vocab=tokenizer.vocab)
    print(f"Saved primary checkpoint to {final_path}", flush=True)


if __name__ == "__main__":
    train()

