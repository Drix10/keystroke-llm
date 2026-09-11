# Keystroke-LLM: The Complete Architectural & Implementation Guide

> **A beginner-friendly yet technically rigorous, line-by-line deep dive into building an edge-computing predictive keystroke engine with a custom NumPy Transformer and real-time USB HID hardware LED streaming.**

*Sections 1–2 explain the goal and the full pipeline in plain language. Sections 3–8 walk through each module in code. Section 9 is a reference for bug patterns.*

---

## Table of Contents

1. [Absolute Zero — Day 1: What Are We Trying To Achieve?](#1-absolute-zero--day-1-what-are-we-trying-to-achieve)
2. [System Overview & End-to-End Pipeline](#2-system-overview--end-to-end-pipeline)
3. [Module 1: The Neural Engine (`model.py`)](#3-module-1-the-neural-engine-modelpy)
4. [Module 2: Dataset Preparation & Training Loop (`train.py`)](#4-module-2-dataset-preparation--training-loop-trainpy)
5. [Module 3: Character Tokenization & Symbol Mapping (`key_mapper.py`)](#5-module-3-character-tokenization--symbol-mapping-key_mapperpy)
6. [Module 4: Keyboard Geometry & Physical Profiles (`profiles/hive75.json`)](#6-module-4-keyboard-geometry--physical-profiles-profileshive75json)
7. [Module 5: Deterministic Hardware USB HID Controller (`hardware_controller.py`)](#7-module-5-deterministic-hardware-usb-hid-controller-hardware_controllerpy)
8. [Module 6: Real-Time Ingestion & Predictive Lighting (`predict_and_light.py`)](#8-module-6-real-time-ingestion--predictive-lighting-predict_and_lightpy)
9. [Edge Cases, Defenses & Reliability Engineering](#9-edge-cases-defenses--reliability-engineering)
10. [Command-Line Reference & Cheat Sheet](#10-command-line-reference--cheat-sheet)

---

## 1. Absolute Zero — Day 1: What Are We Trying to Achieve?

Let's start from absolute zero — Day 1.

Imagine you have never heard of Artificial Intelligence, neural networks, or transformers. Forget all the complicated jargon for a moment. We are going to explain, in the simplest possible language, what this whole project is trying to do and why it is so exciting.

### 1.1 The Big Goal in One Sentence

> **We want a computer program that watches what you type, anticipates what letter you are about to press next, and lights up that exact physical key on your mechanical keyboard in vivid red before your finger even touches the switch.**

Example:
- You type: `the quick brow`
- The program looks at the letters so far and immediately realizes: the next letter is almost certainly `n`.
- Before you even move your hand, the physical `N` key on your mechanical keyboard glows bright red!
- You press `n`, type a space, and now the keyboard illuminates `f` (for `fox`).

That's it. This is the exact same foundational concept behind ChatGPT, Claude, and modern language models — but scaled down into a tiny, transparent engine that you can completely understand and see working directly on physical hardware!

---

### 1.2 Why Do We Even Want This?

Humans write and speak by constantly predicting what comes next.

When you begin speaking a sentence like *"I'm going to the grocery..."*, your brain is already lining up words like *"store"* or *"market"*. You don't pick words at random; you rely on context, patterns, and experience.

We want a computer to do the exact same thing at the character level:
```
1. Look at the characters typed so far (the "context").
2. Guess the most likely next character.
3. Light up the corresponding key switch on your desk.
4. Repeat seamlessly for every single key you press!
```

---

### 1.3 How Does a Computer "Understand" Words and Letters?

Computers do not understand letters, words, or feelings. Computers only understand **numbers**.

So before we can do any math or machine learning, our first job is to turn characters into numbers. We build a simple list containing every character our keyboard can produce:

```text
Vocabulary List:
'<unk>' (unknown) -> 0
' '     (space)   -> 1
'\n'               -> 2
'\t'               -> 3
'0'                -> 4
'a'                -> 14
'b'                -> 15
'c'                -> 16
...
'z'                -> 39
```

This list is called our **Vocabulary** (`vocab`).
Every word you type becomes a clean sequence of numbers:
- `"cat"` becomes `[16, 14, 33]` (representing `'c'`, `'a'`, `'t'`).

---

### 1.4 The Core Task We Train the Model To Do

Once we turn letters into numbers, we train the computer with a simple guessing game:

1. We give the model a short sequence of numbers (for example, the last 48 letters you typed).
2. We ask it: *"What number (letter) should come next?"*
3. The model makes a guess.
4. Because we have training text, we already know the true answer!
5. We measure **how wrong** the model was (called the **Loss**).
6. We gently nudge the model's internal numbers in the right direction so next time it makes a slightly smarter guess.
7. We repeat this process thousands of times. Slowly but surely, the model learns the rules of English spelling, grammar, and even programming syntax!

---

### 1.5 What Is Inside the "Model"?

Think of the model as a clear box with a few mathematical tables (these tables are called **weights** or parameters):

1. **Embedding**: Each character gets its own private row of numbers (a vector). Instead of just being "character #2", the letter `'a'` becomes a learned hidden-size feature vector that captures how it is used in text. New models default to 64 values; the checked-in final checkpoint uses 128.
2. **Self-Attention**: The heart of the transformer! When guessing the next letter, the model doesn't just look at the very last key; it looks back across the entire history of recent letters and decides **which ones matter most**. For instance, if you typed `q-u-i-c-k`, the model pays attention to `q` and `u` to know you are in the middle of a word.
3. **Feed-Forward Layers**: Extra calculations that mix and combine the clues so the model can learn nuanced patterns.
4. **Final Output Projection**: Converts the internal numbers into percentage scores for every key on your keyboard. The key with the highest score is the #1 prediction!

---

### 1.6 Why This Tiny Version Exists

Zero PyTorch. Pure NumPy. Fully inspectable.

This project is built from scratch with zero heavy dependencies:
- **Small and inspectable**: Runs the complete inference path with NumPy arrays and keeps every intermediate attention tensor available for inspection.
- **100% Transparent**: You can print out the entire attention matrix and watch the model's brain think.
- **Small training loop**: Trains directly from a text file and reports train and validation loss after every epoch.

---

### 1.7 The Physical Twist: Why Bring Hardware into the Loop?

Most AI programs just spit out text onto a monitor. But you already have an interactive grid of 82 LED-backlit mechanical switches resting right beneath your fingers!

Instead of keeping the AI trapped inside a console window, Keystroke-LLM sends vendor-level USB packets straight to your keyboard's microcontroller. As you type:
- Unpredicted keys remain a crisp, solid white backlight (`#FFFFFF`).
- The #1 highest probability key glows in **Solid Vivid Red** (`#FF0000`).
- Alternative predictions (ranks 2 to 5) glow in softer, graded red tones (`#FF3333` to `#FF9999`).

The physical keyboard becomes a glowing, living heatmap of human intent!

---

### 1.8 Simple Everyday Analogy

Imagine teaching a friend to play a word completion game:
- You say: *"The quick brown..."*
- They shout: *"FOX!"*
- You say: *"Great job!"*

Now imagine playing that game at 100 words per minute, one letter at a time, where your keyboard lights up the correct key switch just before your finger presses it down. That is Keystroke-LLM.

---

### 1.9 What Success Looks Like in This Project

When you launch `python predict_and_light.py --show-probs`:
1. The keyboard starts with a clean, 100% white backlight.
2. As soon as you type your first letter, the model evaluates the current context and ranks valid next-key candidates.
3. The next probable keys immediately illuminate in red.
4. If you pause typing, the keyboard gracefully dims back to solid white.
5. If you make a typo and press Backspace, the model unwinds its context and instantly recalculates.

> 🎥 **Live Video Demo**: Watch the full working video demonstration of the physical keyboard illuminating in real time on [LinkedIn](https://www.linkedin.com/posts/drix10_llm-ondeviceai-mechanicalkeyboards-activity-7503825549148975105-CW31) *(Shortlink: [lnkd.in/p/gipWJfmn](https://lnkd.in/p/gipWJfmn))*.

Now that we understand the big picture from Day 1, let's look at how every single piece of math, software, and USB hardware works under the hood!

---
## 2. System Overview & End-to-End Pipeline

Keystroke-LLM transforms your physical mechanical keyboard into an active extension of a deep neural network. Instead of passively reading text from a monitor after you type, the system anticipates what you are about to type next and projects probability heatmaps directly beneath your fingertips onto individual mechanical switches in real time.

Here is the whole job in ordinary language: a key is pressed, the program turns that character into a number, the model scores every possible next character, and the hardware controller translates the best scores into colored LED slots. The same path runs again after every keypress. Training happens beforehand; during the live demo, the model only performs the fast forward calculation.

1. **Input capture** receives a physical key without blocking the rest of the program.
2. **The context buffer** remembers the most recent characters, because one character alone is rarely enough to predict the next one.
3. **The tokenizer** converts characters such as `t` and space into the integer IDs used by NumPy arrays.
4. **The Transformer** compares the current context with patterns learned from the training file and returns one score per vocabulary item.
5. **The ranking step** keeps the most likely valid characters and gives each one a red intensity based on its rank.
6. **The key mapper and profile** convert a character such as uppercase `A` into the physical switch name `a`, then into the verified LED slot for that keyboard.
7. **The USB controller** packages the 128-slot RGB buffer into the keyboard's reports and sends them with the required checksum and acknowledgment handling.

```
+---------------------------------------------------------------------------------------+
|                                    USER KEYSTROKE                                     |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                          DUAL INGESTION LAYER (Low Latency)                           |
|       - Windows: Low-level WH_KEYBOARD_LL hook (pynput) + msvcrt non-blocking        |
|       - Linux: Raw termios non-blocking select()                                      |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                             TINY TRANSFORMER NEURAL ENGINE                            |
|       - Single-layer Causal Self-Attention written from scratch in pure NumPy         |
|       - NumPy forward inference: Softmax probability distribution                      |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                              TOP-K RANKING & COLOR GRADING                            |
|       - Rank 1 (Highest probability): Solid Vivid Red (#FF0000)                       |
|       - Ranks 2-5: Graded softer tones (#FF3333, #FF5555, #FF7777, #FF9999)           |
|       - Unhighlighted keys: 100% Crisp White (#FFFFFF) Backlight                      |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                             PHYSICAL MATRIX SLOT MAPPING                              |
|       - Maps character tokens to Kreo Hive 75 physical LED slots                       |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                                USB HID PROTOCOL STREAMING                             |
|       - Native EVision V2 Vendor Interface (Usage Page 0xFF1C, Report ID 0x04)        |
|       - EVision 64-byte chunks, checksums, ACK draining, and 10 Hz keepalive           |
+---------------------------------------------------------------------------------------+
                                           |
                                           v
+---------------------------------------------------------------------------------------+
|                             PHYSICAL MECHANICAL KEYBOARD                              |
|       - Next switches illuminate in Red before your fingers hit the keycaps!          |
+---------------------------------------------------------------------------------------+
```

### Core Design Principles
1. **Zero External Machine Learning Dependencies**: The neural network is written in pure NumPy. No PyTorch, no TensorFlow, no CUDA runtimes.
2. **Deterministic Hardware Protocol**: Rather than relying on heavyweight GUI lighting suites that introduce hundreds of milliseconds of lag, this project speaks the native vendor USB HID protocol directly over raw OS endpoints.
3. **Bounded, asynchronous runtime**: Input capture, prediction, and USB updates are separated so a slow device or a burst of typing does not block the input callback indefinitely. Actual latency depends on the operating system, NumPy, HID driver, and hardware.

---

## 3. Module 1: The Neural Engine (`model.py`)

`model.py` contains an implementation of an autoregressive, single-layer causal Transformer built from mathematical first principles using NumPy arrays.

In practical terms, this file is the part that answers one question: **given the characters seen so far, which vocabulary item should come next?** It has two jobs. During training it adjusts numeric tables so its guesses improve; during prediction it reads those tables and produces probabilities without changing them. The rest of the project can treat it as a box with a simple input (`token_indices`) and two outputs: logits and next-character probabilities.

### 3.1 Mathematical Foundations & Tensor Shapes

This table is the backbone of the module. If you're comfortable with tensor shapes, it is a quick reference; if you're new to them, read it alongside 3.3. Each row records what shape enters an operation and what shape must come out.

For example, with four input characters and a hidden size of 64, the input starts as four integers, becomes a `4 x 64` matrix, stays four rows throughout attention and the feed-forward layer, and ends as a `4 x 98` matrix of vocabulary scores. We use only the last row when making the next-key prediction.

Let:
- $V$: Vocabulary size ($V = 98$, encompassing uppercase, lowercase, numbers, and symbols).
- $d_{model}$: Hidden dimension. A newly created model defaults to $64$; the checked-in `model_final.npz` was trained with $128$.
- $T$: Context sequence length ($1 \le T \le seq\_len$, default $seq\_len = 48$).
- $d_{ff}$: Feed-forward hidden dimension ($d_{ff} = 2 \times d_{model}$). It is $128$ for a new default model and $256$ in the checked-in final checkpoint.

#### Tensor Dimensions Across the Forward Pass

| Step | Operation | Input Shape | Output Shape | Formula |
| :--- | :--- | :--- | :--- | :--- |
| **Embedding** | Lookup | $[T]$ | $[T, d_{model}]$ | $X = W_{emb}[\text{tokens}]$ |
| **Query** | Projection | $[T, d_{model}]$ | $[T, d_{model}]$ | $Q = X \cdot W_q$ |
| **Key** | Projection | $[T, d_{model}]$ | $[T, d_{model}]$ | $K = X \cdot W_k$ |
| **Value** | Projection | $[T, d_{model}]$ | $[T, d_{model}]$ | $V = X \cdot W_v$ |
| **Attention Scores** | Scaled Dot-Product | $[T, d_{model}] \times [d_{model}, T]$ | $[T, T]$ | $S = \frac{Q K^T}{\sqrt{d_{model}}}$ |
| **Causal Masking** | Autoregressive Mask | $[T, T]$ | $[T, T]$ | $S_{masked} = S + M$ where $M_{i,j} = -\infty$ for $j > i$ |
| **Attention Weights** | Row-wise Softmax | $[T, T]$ | $[T, T]$ | $A = \text{softmax}(S_{masked})$ |
| **Context Aggregation** | Weighted Sum | $[T, T] \times [T, d_{model}]$ | $[T, d_{model}]$ | $C = A \cdot V$ |
| **Attention Output** | Projection | $[T, d_{model}]$ | $[T, d_{model}]$ | $O_{attn} = C \cdot W_o$ |
| **Residual 1** | Skip Connection | $[T, d_{model}]$ | $[T, d_{model}]$ | $X_{res1} = X + O_{attn}$ |
| **FFN Expand** | Linear Layer | $[T, d_{model}]$ | $[T, d_{ff}]$ | $H_{pre} = X_{res1} \cdot W_1$ |
| **FFN Activation** | ReLU Non-linearity | $[T, d_{ff}]$ | $[T, d_{ff}]$ | $H_{act} = \max(0, H_{pre})$ |
| **FFN Project** | Linear Layer | $[T, d_{ff}]$ | $[T, d_{model}]$ | $O_{ffn} = H_{act} \cdot W_2$ |
| **Residual 2** | Skip Connection | $[T, d_{model}]$ | $[T, d_{model}]$ | $X_{res2} = X_{res1} + O_{ffn}$ |
| **Output Head** | Logits Projection | $[T, d_{model}]$ | $[T, V]$ | $Z = X_{res2} \cdot W_{out} + b_{out}$ |
| **Next Token Probs** | Final Softmax | $[V]$ | $[V]$ | $P = \text{softmax}(Z[-1, :])$ |

---

### 3.2 The Self-Attention Mechanism Step-by-Step

Self-attention allows each character in the active typing window to dynamically focus on previous characters to establish linguistic patterns (e.g. noticing that `'q'` is almost universally followed by `'u'`).

The three names in this section are easier to understand as roles than as vocabulary. A **query** asks, “what earlier information would help me here?” A **key** describes what each earlier position contains, and a **value** carries the information that will actually be copied forward. The model compares the current query with earlier keys, turns those comparisons into weights, and blends the earlier values using those weights. The causal mask makes sure a position cannot look into the future and accidentally read the answer during training.

```
Input Tokens:   ['t', 'h', 'e', ' ']
                  |    |    |    |
                  v    v    v    v
            +-----------------------+
            |  Embedding Layer      |  -> Matrix X [4 x d_model]
            +-----------------------+
                  |         |
         +--------+         +--------+
         v                           v
   Query = X * W_q              Key = X * W_k
   [4 x d_model]                [4 x d_model]
         \                           /
          \                         /
           v                       v
          Attention Scores S = (Q * K^T) / sqrt(d_model)  [4 x 4]
                             |
                             v
               Apply Causal Mask (Upper Triangle = -1e9)
               [ t   h   e   _ ]
             t [ 0.2 -inf -inf -inf ]
             h [ 0.8  0.4 -inf -inf ]
             e [ 0.3  0.7  0.9 -inf ]
             _ [ 0.1  0.2  0.8  0.5 ]
                             |
                             v
               Row-wise Softmax -> Weights A [4 x 4]
                             |
                             v
             Multiply by Values V = X * W_v [4 x d_model]
                             |
                             v
             Aggregated Context C = A * V   [4 x d_model]
```

---

### 3.3 Line-by-Line Code Walkthrough

#### Mathematical Primitives (`model.py`)
```python
def randn(rows: int, cols: int, scale: float = 0.1) -> np.ndarray:
    return np.random.uniform(-scale, scale, size=(rows, cols)).astype(np.float64)
```
- Still available for backward compatibility, but most weights now use `xavier_uniform` instead.

```python
def xavier_uniform(rows: int, cols: int) -> np.ndarray:
    limit = np.sqrt(6.0 / (rows + cols))
    return np.random.uniform(-limit, limit, size=(rows, cols)).astype(np.float64)
```
- **Xavier/Glorot initialization** — the mathematically correct starting point for weight matrices. It scales the random range so that activation variance stays roughly constant through all layers at startup. Without this, the forward pass either explodes or vanishes in the first few steps.

```python
def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exps = np.exp(shifted)
    return exps / np.sum(exps, axis=axis, keepdims=True)
```
- **Numerical Stability**: If values in $x$ are large (e.g. $100$), $\exp(100) \approx 2.68 \times 10^{43}$, causing floating-point overflow (`inf`). Subtracting $\max(x)$ ensures the largest exponent evaluated is $\exp(0) = 1.0$, preventing overflow while yielding the mathematically identical probability distribution.

#### Model Initialization (`model.py` — `__init__`)
```python
class TinyTransformer:
    def __init__(self, vocab_size: int = 98, hidden_size: int = 64, seq_len: int = 48, ...):
```
- Pre-allocates all parameter matrices with Xavier uniform initialization:
  - `W_emb`: Token embeddings $[V, d_{model}]$ — one learned vector per character.
  - **`W_pos`**: Positional embeddings $[seq\_len, d_{model}]$ — initialized to **zero**. Starts neutral, and the model gradually learns what "being at position 3 in the context" means.
  - `W_q`, `W_k`, `W_v`, `W_o`: Attention projections $[d_{model}, d_{model}]$.
  - `W1`: FFN expansion $[d_{model}, 2 \times d_{model}]$.
  - `W2`: FFN projection $[2 \times d_{model}, d_{model}]$.
  - `W_out`: Classification head $[d_{model}, V]$.
  - `b_out`: Output bias vector $[V]$.
  - `causal_mask`: Precomputed $[seq\_len, seq\_len]$ matrix with $0.0$ on/below diagonal, $-10^9$ above.
  - **`causal_mask_bool`**: A boolean version of the mask (`mask < 0`). Used in backprop to zero out gradients into future positions — safer than comparing float values to `-1e8`.

#### The Forward Pass
```python
def forward(self, token_indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
```
1. **Auto-slicing**: If the context is longer than `seq_len`, the oldest characters are dropped.
2. **Embedding + Position**: `X = self.W_emb[token_indices] + self.W_pos[:T, :]`
   - `W_emb` says *what* each character is (its identity in meaning space).
   - `W_pos` says *where* it sits in the current context window.
   - Adding them together gives the model the full picture: "this is an 'e', and it's at position 4".
3. **Q, K, V projections**: Three different linear views of the same input, used for the attention mechanism.
4. **Causal attention**: `Softmax((Q @ K.T) / sqrt(d) + mask) @ V` — lets each position look at everything before it, nothing after.
5. **Residual 1**: `X_res1 = X + attn_out` — adds the attention output back onto the input (skip connection).
6. **FFN**: Two-layer expansion with ReLU non-linearity, then another residual.
7. **Output head**: Projects final hidden state to logits over all `vocab_size` characters, then softmax gives probability distribution.
8. **Cache**: All intermediate tensors are stored in `self.last_cache` for backpropagation.

#### The `generate()` Method
```python
def generate(self, seed_tokens, num_chars=20, temperature=1.0, top_p=0.9) -> list[int]:
```
- Auto-regressively samples new characters one by one.
- **Temperature**: A value `< 1` makes the model more confident and boring. A value `> 1` makes it more random and creative. `0` means always pick the top prediction (greedy).
- **Nucleus (top-p) sampling**: Only samples from the smallest set of candidates whose combined probability reaches `top_p`. Avoids weird rare characters while keeping variety.

#### Adam Optimizer State Checkpointing
```python
def save_checkpoint(self, filepath, vocab=None):
```
- Now saves `adam_step`, `adam_emb_m/v` (embedding moment arrays), `adam_pos_m/v` (positional moment arrays), and `adam_m__*/adam_v__*` (all weight moment arrays) into the `.npz` file alongside the weights.

```python
@classmethod
def load_checkpoint(cls, filepath):
```
- On load, if Adam state is present in the file, it's fully restored. This means resuming training (`--resume`) picks up exactly where it left off — the bias-correction factors won't spike back to epoch-1 values.

---

### 3.4 Complete Analytical Backpropagation Calculus

> **Advanced section:** Skip this derivation on a first read. It is here as a reference when you want to understand or modify training from first principles.

In `train_step_backprop()`, we compute the exact partial derivatives of the cross-entropy loss $\mathcal{L}$ with respect to every weight matrix in the Transformer.

#### Step 1: Loss Function & Logits Gradient
The cross-entropy loss for target token $y \in \{0, \dots, V-1\}$ is:
$$\mathcal{L} = -\log P(y) = -\log \left( \frac{e^{z_y}}{\sum_{j=0}^{V-1} e^{z_j}} \right)$$
Differentiating with respect to the output logit $z_k$:
$$\frac{\partial \mathcal{L}}{\partial z_k} = P(k) - \mathbb{I}(k = y)$$
In `train_step_backprop()`:
```python
d_logits = np.zeros_like(c["logits"])
d_logits[-1, :] = probs.copy()
d_logits[-1, target_token] -= 1.0
```

#### Step 2: Gradients of Output Head & Residual 2
Since $Z = X_{res2} \cdot W_{out} + b_{out}$:
$$\frac{\partial \mathcal{L}}{\partial W_{out}} = X_{res2}^T \cdot \frac{\partial \mathcal{L}}{\partial Z}, \quad \frac{\partial \mathcal{L}}{\partial b_{out}} = \sum_{t=1}^T \frac{\partial \mathcal{L}}{\partial Z_t}, \quad \frac{\partial \mathcal{L}}{\partial X_{res2}} = \frac{\partial \mathcal{L}}{\partial Z} \cdot W_{out}^T$$

#### Step 3: FFN & Residual 1 Backprop
Because $X_{res2} = X_{res1} + O_{ffn}$, gradients flow into both branches:
$$\frac{\partial \mathcal{L}}{\partial O_{ffn}} = \frac{\partial \mathcal{L}}{\partial X_{res2}}$$
Backpropagating through the linear projection $W_2$ and ReLU activation:
$$\frac{\partial \mathcal{L}}{\partial W_2} = H_{act}^T \cdot \frac{\partial \mathcal{L}}{\partial O_{ffn}}$$
$$\frac{\partial \mathcal{L}}{\partial H_{act}} = \frac{\partial \mathcal{L}}{\partial O_{ffn}} \cdot W_2^T$$
$$\frac{\partial \mathcal{L}}{\partial H_{pre}} = \frac{\partial \mathcal{L}}{\partial H_{act}} \odot \mathbb{I}(H_{pre} > 0)$$
$$\frac{\partial \mathcal{L}}{\partial W_1} = X_{res1}^T \cdot \frac{\partial \mathcal{L}}{\partial H_{pre}}$$
Adding the residual gradient from FFN input:
$$\frac{\partial \mathcal{L}}{\partial X_{res1}} = \frac{\partial \mathcal{L}}{\partial X_{res2}} + \frac{\partial \mathcal{L}}{\partial H_{pre}} \cdot W_1^T$$

#### Step 4: Causal Self-Attention Backpropagation
Gradients split across Residual 1:
$$\frac{\partial \mathcal{L}}{\partial O_{attn}} = \frac{\partial \mathcal{L}}{\partial X_{res1}}$$
$$\frac{\partial \mathcal{L}}{\partial W_o} = C^T \cdot \frac{\partial \mathcal{L}}{\partial O_{attn}}, \quad \frac{\partial \mathcal{L}}{\partial C} = \frac{\partial \mathcal{L}}{\partial O_{attn}} \cdot W_o^T$$
Since $C = A \cdot V$:
$$\frac{\partial \mathcal{L}}{\partial V} = A^T \cdot \frac{\partial \mathcal{L}}{\partial C}, \quad \frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial C} \cdot V^T$$
Differentiating the row-wise softmax $A_{i,:} = \text{softmax}(S_{i,:})$:
$$\frac{\partial \mathcal{L}}{\partial S_{i,j}} = A_{i,j} \left( \frac{\partial \mathcal{L}}{\partial A_{i,j}} - \sum_{k=1}^T \frac{\partial \mathcal{L}}{\partial A_{i,k}} A_{i,k} \right)$$
In code, this is evaluated using a fully vectorized row-sum broadcast rather than a Python row loop:
```python
sum_da_s = np.sum(d_attn_weights * c["attn_weights"], axis=-1, keepdims=True)
d_masked = c["attn_weights"] * (d_attn_weights - sum_da_s)
d_masked[self.causal_mask_bool[:T, :T]] = 0.0
```
For causal masked positions ($j > i$), the gradient is clamped strictly to $0.0$ via `causal_mask_bool`.

Finally, scaling by $\frac{1}{\sqrt{d_k}}$:
$$\frac{\partial \mathcal{L}}{\partial Q} = \frac{\frac{\partial \mathcal{L}}{\partial S}}{\sqrt{d_k}} \cdot K, \quad \frac{\partial \mathcal{L}}{\partial K} = \left(\frac{\frac{\partial \mathcal{L}}{\partial S}}{\sqrt{d_k}}\right)^T \cdot Q$$
Projections into weights and input embeddings:
$$\frac{\partial \mathcal{L}}{\partial W_q} = X^T \cdot \frac{\partial \mathcal{L}}{\partial Q}, \quad \frac{\partial \mathcal{L}}{\partial W_k} = X^T \cdot \frac{\partial \mathcal{L}}{\partial K}, \quad \frac{\partial \mathcal{L}}{\partial W_v} = X^T \cdot \frac{\partial \mathcal{L}}{\partial V}$$
Total input gradient (flows back to both W_emb rows and W_pos rows):
$$\frac{\partial \mathcal{L}}{\partial X} = \frac{\partial \mathcal{L}}{\partial X_{res1}} + \frac{\partial \mathcal{L}}{\partial Q} \cdot W_q^T + \frac{\partial \mathcal{L}}{\partial K} \cdot W_k^T + \frac{\partial \mathcal{L}}{\partial V} \cdot W_v^T$$
Since $X = W_{emb}[\text{tokens}] + W_{pos}[0:T, :]$, the same gradient $\frac{\partial \mathcal{L}}{\partial X}$ flows into **both** the token embedding rows and the positional embedding rows:
$$\frac{\partial \mathcal{L}}{\partial W_{emb}[\text{token}_i]} = \frac{\partial \mathcal{L}}{\partial X[i]}, \quad \frac{\partial \mathcal{L}}{\partial W_{pos}[i]} = \frac{\partial \mathcal{L}}{\partial X[i]}$$
Parameter weights are updated using the **Adam optimizer** with gradient clipping in $[-5.0, 5.0]$ to prevent exploding gradients. The causal mask gradient zero-out now uses the precomputed `causal_mask_bool` boolean array instead of a fragile float comparison (`<= -1e8`).

---

## 4. Module 2: Dataset Preparation & Training Loop (`train.py`)

Training is how the numeric tables in `model.py` learn useful patterns. We start with ordinary text, make two nearly identical windows from it, ask the model to predict the second window from the first, measure the mistake, and update the weights. Nothing in this step talks to the keyboard: it is a repeatable offline preparation step that turns text into a checkpoint the live predictor can load.

The important distinction is **input versus target**. If the input is `cat`, the target is `at` for a next-character task. The model is never given the target while it is making the prediction; the target is used afterward to grade the prediction and calculate the update.

### 4.1 The Sequence-to-Sequence Sliding-Window Formulation

Rather than predicting only a single character at the end of a window, the training engine structures character sequences into **full sequence-to-sequence pairs** of length `seq_len`. This trains the transformer across all positions $t \in [1 \dots seq\_len]$ in a single forward/backward pass:

```
Text: "def main():\n"
Window size (seq_len): 6, Stride: 3

Sample 0:
  Input Context:  ['d', 'e', 'f', ' ', 'm', 'a']
  Target Sequence:['e', 'f', ' ', 'm', 'a', 'i']  (every token predicts the next)

Sample 1:
  Input Context:  [' ', 'm', 'a', 'i', 'n', '(']
  Target Sequence:['m', 'a', 'i', 'n', '(', ')']
```

Every sample allows the causal attention mask to train prefix lengths $1, 2, \dots, seq\_len$ simultaneously, dramatically speeding up convergence and teaching the model both single-letter prefixes and multi-character word completions.

### 4.2 Line-by-Line Code Walkthrough

1. **`build_sliding_window_dataset(text, tokenizer, seq_len, stride=3)`**:
   - Encodes raw text into character IDs using `tokenizer.encode(text)`.
   - Slices input sequence `token_ids[i : i + seq_len]` and offset target sequence `token_ids[i + 1 : i + seq_len + 1]`.
   - The `--stride` parameter controls how many characters the window shifts between samples (default: 3).
2. **`compute_val_loss(model, val_dataset)`**:
   - Forward-only evaluation on the held-out validation dataset without updating gradients.
   - Computes multi-position cross-entropy loss to track true generalization and detect overfitting.
3. **`train()`**:
   - **CLI Flags**: `--data`, `--epochs`, `--lr` (default `0.003`), `--seq-len` (default `48`), `--hidden-size` (default `64`), `--stride` (default `3`), `--mode` (`backprop` or `heuristic`), `--checkpoint-dir`, `--resume`, `--seed`, `--val-split` (default `0.10` / 10%).
   - **Reproducibility**: If `--seed` is passed, runs `np.random.seed(args.seed)` to ensure deterministic data shuffling and weight initialization.
   - **Validation Split**: Automatically partitions the dataset into training samples and a held-out temporal validation split (e.g. 90% train / 10% validation).
   - **CSV Logging**: Automatically logs `epoch`, `train_loss`, `val_loss`, and `time_s` to `checkpoints/loss_log.csv` after every epoch.
   - **Adam Optimizer Checkpointing**: When saving to `.npz`, writes model weights *and* full Adam optimizer states (`adam_step`, `adam_emb_m/v`, `adam_pos_m/v`, parameter moment arrays). Resuming with `--resume` restores the optimizer exactly where it left off, avoiding bias-correction gradient spikes.

---

## 5. Module 3: Character Tokenization & Symbol Mapping (`key_mapper.py`)

This file solves two different naming problems that are easy to mix up. The **tokenizer** gives each character a stable integer ID for the neural model. The **key mapper** gives that same character the physical key name used by the keyboard profile. For example, uppercase `A` is one model token, but it lights the physical lowercase `a` switch because Shift changes the character without changing which switch is pressed.

### 5.1 Vocabulary Composition

The vocabulary has 98 distinct tokens:
- Indices `0..3`: Special tokens `<unk>`, `' '`, `'\n'`, `'\t'`.
- Indices `4..13`: Digits `'0'` to `'9'`.
- Indices `14..39`: Lowercase letters `'a'` to `'z'`.
- Indices `40..65`: Uppercase letters `'A'` to `'Z'`.
- Indices `66..97`: Standard punctuation and keyboard symbols (`!@#$%^&*()-_=+[]{}|;:'",.<>/?`~``).

### 5.2 Line-by-Line Code Walkthrough

```python
CHAR_TO_KEY = {
    **{c: c for c in string.ascii_lowercase},
    **{c.upper(): c for c in string.ascii_lowercase},
    **{d: d for d in string.digits},
    " ": "space", "\t": "tab", "\n": "enter", "\r": "enter", "\b": "backspace",
    "-": "minus", "=": "equal", "[": "lbracket", "]": "rbracket", "\\": "backslash",
    ";": "semicolon", "'": "quote", ",": "comma", ".": "period", "/": "slash", "`": "grave",
    # Shifted punctuation
    "!": "1", "@": "2", "#": "3", "$": "4", "%": "5", "^": "6", "&": "7", "*": "8",
    "(": "9", ")": "0", "_": "minus", "+": "equal", "{": "lbracket", "}": "rbracket",
    "|": "backslash", ":": "semicolon", '"': "quote", "<": "comma", ">": "period",
    "?": "slash", "~": "grave",
}
```
- **Physical Switch Translation**: Maps every US QWERTY character to its corresponding physical switch name in `profiles/hive75.json`. When predicting an uppercase `'A'` or a shifted punctuation mark like `'!'`, the physical key `'a'` or `'1'` is illuminated.
- `CharTokenizer`:
  - Enforces that `<unk>` exists in the vocabulary, raising a descriptive `ValueError` if a custom vocab omitted it.
  - Implements `__len__`, allowing `len(tokenizer)` to return vocabulary size directly.
  - Provides `encode()`, `decode()`, `encode_char()`, and `decode_id()`.

---

## 6. Module 4: Keyboard Geometry & Physical Profiles (`profiles/hive75.json`)

The model predicts characters, but LEDs are not addressed by characters. The keyboard controller needs an exact slot number for each physical switch. This section explains how the JSON profile supplies that translation and why a visually obvious left-to-right key order is not reliable for this particular board.

### 6.1 The 75% Compact Layout Anatomy

The **Kreo Hive 75** profile contains 82 mapped physical keys arranged in six logical rows and fifteen logical columns. The rows are not all full: some positions are empty because the visible keyboard layout is compact.

- **Row 0**: `Esc`, `F1`-`F12`, and `Del`.
- **Row 1**: grave, number keys, `-`, `=`, `Backspace`, and `Ins`.
- **Row 2**: `Tab`, `Q`-`P`, brackets, backslash, and `End`.
- **Row 3**: `CapsLock`, `A`-`L`, semicolon, quote, `Enter`, and `PgUp`.
- **Row 4**: left Shift, `Z`-`M`, comma, period, slash, right Shift, `Up`, and `PgDn`.
- **Row 5**: left Ctrl, Win, left Alt, Space, right Alt, Fn, right Ctrl, and the arrow keys.

```
+----+  +----+----+----+----+----+----+----+----+----+----+----+----+  +----+----+
|Esc |  | F1 | F2 | F3 | F4 | F5 | F6 | F7 | F8 | F9 |F10 |F11 |F12 |  |   |Del |
+----+  +----+----+----+----+----+----+----+----+----+----+----+----+  +----+----+
| `~ | 1! | 2@ | 3# | 4$ | 5% | 6^ | 7& | 8* | 9( | 0) | -_ | =+ | Back |  | Ins|
+----+----+----+----+----+----+----+----+----+----+----+----+----+------+  +----+
| Tab  | Q  | W  | E  | R  | T  | Y  | U  | I  | O  | P  | [{ | ]} | \| |  |End |
+------+----+----+----+----+----+----+----+----+----+----+----+----+----+  +----+
| Caps  | A  | S  | D  | F  | G  | H  | J  | K  | L  | ;: | '" | Enter  |  |PgUp|
+-------+----+----+----+----+----+----+----+----+----+----+----+--------+  +----+
| Shift   | Z  | X  | C  | V  | B  | N  | M  | ,< | .> | /? | Shift |Up|  |PgDn|
+---------+----+----+----+----+----+----+----+----+----+-----+----+---+--+  +----+
| Ctrl| Win| Alt|           Space           | Alt| Fn | Ctrl|Left|Dn |Rt |
+-----+----+----+---------------------------+----+----+-----+----+---+----+
```

---

### 6.2 PCB Matrix Architecture: The 15x6 Profile Grid

During hardware reverse-engineering of the Kreo Hive 75 EVision V2 controller, generic OpenRGB and 104-key drivers produced vertical column shifts because full-size keyboards space function keys differently and route PCB traces in simple row orders.

The profile describes a **15-column by 6-row logical grid**. The controller still allocates 128 RGB slots, so the visible keys occupy only some of the available addresses. The slot formula used by the profile is:

$$\text{slot} = (\text{col} \times 8) + \text{row}$$

```
      Col 0   Col 1   Col 2   Col 3   Col 4   Col 5   Col 6 ... Col 14
   Row 0:   Esc     F1      F2      F3      F4      F5      F6  ... Del
   Row 1:   `~      1       2       3       4       5       6   ... Ins
   Row 2:   Tab     Q       W       E       R       T       Y   ... End
   Row 3:   Caps    A       S       D       F       G       H   ... PgUp
   Row 4:   LShift  Z       X       C       V       B       N   ... PgDn
   Row 5:   LCtrl   Win     Alt     --      --      Space     --  ... Right
```

The JSON file is the source of truth for empty positions and hardware-specific slots. For example, `w` is at slot 18, `v` at slot 44, `space` at slot 45, and `right` at slot 121. Do not infer a slot from the visible key order.

---

### 6.3 Verified Physical Hardware Slot Mapping Table

Below is the verified hardware slot mapping implemented in [`profiles/hive75.json`](profiles/hive75.json):

| Matrix row | Keys and slots |
| :--- | :--- |
| **0** | `esc`: 0, `f1`: 8, `f2`: 16, `f3`: 24, `f4`: 32, `f5`: 40, `f6`: 48, `f7`: 56, `f8`: 64, `f9`: 72, `f10`: 80, `f11`: 88, `f12`: 96, `del`: 112 |
| **1** | `grave`: 1, `1`: 9, `2`: 17, `3`: 25, `4`: 33, `5`: 41, `6`: 49, `7`: 57, `8`: 65, `9`: 73, `0`: 81, `minus`: 89, `equal`: 97, `backspace`: 105, `ins`: 113 |
| **2** | `tab`: 2, `q`: 10, `w`: 18, `e`: 26, `r`: 34, `t`: 42, `y`: 50, `u`: 58, `i`: 66, `o`: 74, `p`: 82, `lbracket`: 90, `rbracket`: 98, `backslash`: 106, `end`: 114 |
| **3** | `capslock`: 3, `a`: 11, `s`: 19, `d`: 27, `f`: 35, `g`: 43, `h`: 51, `j`: 59, `k`: 67, `l`: 75, `semicolon`: 83, `quote`: 91, `enter`: 107, `pgup`: 115 |
| **4** | `lshift`: 12, `z`: 20, `x`: 28, `c`: 36, `v`: 44, `b`: 52, `n`: 60, `m`: 68, `comma`: 76, `period`: 84, `slash`: 92, `rshift`: 100, `up`: 118, `pgdn`: 122 |
| **5** | `lctrl`: 5, `win`: 13, `lalt`: 21, `space`: 45, `ralt`: 77, `fn`: 85, `rctrl`: 101, `left`: 119, `down`: 120, `right`: 121 |

---

## 7. Module 5: Deterministic Hardware USB HID Controller (`hardware_controller.py`)

This module is the last step between a prediction and a lit key. It owns the device connection, keeps a complete RGB buffer for all 128 controller slots, converts that buffer into the keyboard's report format, and sends the reports at the cadence the firmware expects. The rest of the application asks for colors by key name; this module handles bytes, checksums, acknowledgments, reconnects, and cleanup.

Start with mock mode. It exercises the same color-buffer API without requiring a keyboard, so you can prove that `w` becomes red and the other slots stay white before debugging USB permissions or device discovery.

### 7.1 EVision V2 USB Protocol Deep Dive

The Kreo Hive 75 operates using an EVision V2 microcontroller (`VID: 0x320F, PID: 0x5055` / `258A:010C`).
- **Endpoint**: Vendor Usage Page `0xFF1C`, Report ID `0x04`.
- **EVision report length**: 64 bytes. The profile also stores a 520-byte feature-report length for the non-EVision path.
- **Color Format**: 3 bytes per switch: Red, Green, Blue ($0 \dots 255$).
- **Total RGB Payload**: $128 \text{ slots} \times 3 \text{ bytes} = 384 \text{ bytes}$.
- **Chunking**: $384 \text{ bytes}$ are divided across 7 sequential USB HID packets ($6 \times 56\text{ bytes} + 1 \times 48\text{ bytes}$).

```
+-------------------------------------------------------------------------------+
|                      EVISION V2 64-BYTE HID PACKET FORMAT                     |
+---+---------+---------+------+-------+---------+----------+------+------------+
| 0 |    1    |    2    |  3   |   4   |    5    |    6     |  7   |  8 ... 63  |
+---+---------+---------+------+-------+---------+----------+------+------------+
|Rep| Chk_Low |Chk_High | CMD  | Len   | Off_Low | Off_High | 0x00 | RGB Data   |
|ID |         |         | 0x12 | (<=56)|         |          |      | (Payload)  |
+---+---------+---------+------+-------+---------+----------+------+------------+
```

### 7.2 Packet Framing, Checksum & ACK Draining

Every 64-byte packet requires a 16-bit sum checksum computed over bytes $3 \dots 63$:
$$\text{Checksum} = \sum_{k=3}^{63} \text{packet}[k] \pmod{65536}$$
```python
def _compute_evision_checksum(buf: bytearray) -> bytearray:
    chksum = sum(buf[3:64]) & 0xFFFF
    buf[1] = chksum & 0xFF        # Low byte
    buf[2] = (chksum >> 8) & 0xFF  # High byte
    return buf
```

#### USB Pipe Buffer & ACK Draining
The EVision V2 keyboard microcontroller responds to every dynamic color report with an acknowledgment report. If the host software writes frames without reading ACKs, the operating system USB endpoint buffer can overflow and writes can stall.
`hardware_controller.py` drains the ACK report after each chunk write:
```python
self.hid_device.write(list(pkt))
if read_ack:
    self.hid_device.read(64, 20)  # Drain ACK response within 20ms timeout
```
The read has a 20 ms timeout, and failures are handled by marking the HID handle disconnected. If the checksum does not match, the keyboard's USB microcontroller rejects the packet and drops the frame. The code does not promise a fixed end-to-end latency.

### 7.3 The 10 Hz Keepalive Watchdog Engine

The EVision keyboard firmware features an internal watchdog timer. The controller's EVision path therefore flushes the current frame every $100\text{ ms}$ ($10\text{ Hz}$). For a non-EVision profile, the configured `keepalive_hz` controls the interval; in `hive75.json` that value is `1.0`.

---

### 7.4 Line-by-Line Code Walkthrough

1. **`KeyboardProfile`**:
   - Loads layout geometry, USB VIDs/PIDs, and assigns key slot indices.
   - Automatically populates common aliases (`ctrl` $\to$ `lctrl`, `shift` $\to$ `lshift`, `del` $\to$ `del`, `function` $\to$ `fn`, `=` $\to$ `equal`, `[` $\to$ `lbracket`).
2. **`_acquire_lock()` and `_release_lock()`**:
   - Prevents multiple conflicting processes from asserting conflicting USB handles.
   - Uses `_is_pid_running(old_pid)` to clean up stale lockfiles automatically if a prior session crashed, respecting POSIX `EPERM` permissions.
3. **`_connect_hid()`**:
   - Closes existing handles to eliminate OS resource leaks.
   - Enumerates devices, matching `0x320F:0x5055` and filtering for vendor Usage Page `0xFF1C`.
4. **`_send_evision_frame()` and `_flush_frame()`**:
   - Slices the 384-byte buffer into 56-byte chunks.
   - Injects Command `0x12` (`EVISION_V2_CMD_SEND_DYNAMIC_COLORS`), computes checksums, and writes reports.
   - **Thread-Safe Reconnect Design**: If a write fails (e.g., unplugged keyboard), the method marks `self.hid_device = None` and exits immediately without attempting an inline reconnect. The keepalive thread owns all reconnection attempts, eliminating lock contention and deadlocks between active key updates and reconnect attempts.
5. **`_keepalive_loop()`**:
   - Flushes EVision frames at 10 Hz. For other modes, uses `keepalive_hz`; if `keepalive_hz == 0`, keepalive transmission is disabled.
   - Every 2 seconds when disconnected, attempts to re-establish the USB handle safely under `_lock`.
6. **`set_key_colors(key_colors, brightness, clear_others, default_background="ffffff")`**:
   - **Full Buffer Fill**: When `clear_others=True`, populates all 128 slots unconditionally:
     ```python
     self.rgb_buffer = bytearray([bg_r, bg_g, bg_b] * self.profile.num_slots)
     ```
     This ensures that all 82 switches on the board glow in uniform white light with no missing keys.
   - Overwrites the target Top-K predicted keys with their respective red tones.
7. **`close()`**:
   - Signals `self._stop_event.set()` to instantly wake and join the keepalive thread without sleep latency.
   - Re-arms the keyboard's default lighting via `restore_default_mode()`, closes the HID handle, and deletes the lockfile.

---

## 8. Module 6: Real-Time Ingestion & Predictive Lighting (`predict_and_light.py`)

### 8.0 Runtime Configuration

The editable [`config.json`](README.md) file is the single source for user-facing runtime settings. It contains lighting colors, the white background, brightness, prediction count, idle and gaming timers, WASD thresholds, debounce timing, queue size, checkpoint, profile, and optional output flags. Command-line values override the file for one run; the Windows startup command loads the file directly.

This is the conductor that connects the earlier modules. It receives characters, maintains the rolling context, calls the model, filters and ranks predictions, maps them to physical keys, and asks the controller to redraw the LEDs. It also decides when input is gaming movement rather than text and when idle time should clear the red highlights.

The live loop should stay responsive even when a key is held or the USB device is slow. That is why input capture, the bounded queue, prediction work, and hardware updates are separated instead of putting all of them inside one keyboard callback.

### 8.1 The Dual-Engine Ingestion Architecture

Capturing keystrokes system-wide on modern operating systems introduces multiple platform challenges:
- Global hooks (such as `WH_KEYBOARD_LL`) can be dropped or blocked by security policies, elevated windows, or console host buffers.
- Direct console stdin reads (like `msvcrt.kbhit()`) only work when the terminal itself has input focus.

`LowLatencyInputReader` uses a primary global hook with a console fallback:
1. **Engine A (Global Hook)**: Runs `pynput.keyboard.Listener` in non-blocking mode via `listener.start()`. Captures keystrokes when the user is typing across any window (web browsers, code editors, games, chat apps).
2. **Engine B (Console Poller)**: Starts only if the global hook cannot initialize. On Windows it polls `msvcrt`; on POSIX systems it uses `select` and `termios`.
3. **15ms Debounce Filter**: To prevent duplicate key ingestion when both engines detect the same physical keystroke, `_enqueue()` tracks timestamp deltas per character:
   ```python
   if (now - last_t) > 0.015:  # 15 ms debounce window
       self.key_queue.put_nowait(ch)
   ```

```
Typing in Browser/Editor               Typing in Terminal
         |                                     |
         v                                     v
+-------------------+                 +-------------------+
|  pynput Listener  |                 |  msvcrt Console   |
|   (Global Hook)   |                 |    (Stdin Poll)   |
+-------------------+                 +-------------------+
         \                                     /
          \                                   /
           v                                 v
   +-------------------------------------------------+
   |      15ms Software Debounce & Normalizer        |
   |      - Eliminates duplicate entries             |
   |      - Converts '\r' -> '\n', '\x7f' -> '\b'    |
   |      - Discards unprintable control codes       |
   +-------------------------------------------------+
                           |
                           v
              Bounded Queue (maxsize=128)
```

---

### 8.2 Auto-Pause WASD / Gaming Detection

When gaming (e.g. playing an FPS or movement-heavy game), rapid WASD keystrokes or held key presses would otherwise cause erratic red highlights across the keyboard.

`PredictiveKeyLightsApp` features an automatic gaming state detector:
1. **Trigger Condition**:
   - $\ge 4$ consecutive keystrokes within `{'w', 'a', 's', 'd'}`.
   - $\ge 5$ consecutive repetitions of any non-space key.
2. **Behavior on Trigger**:
   - Switches `is_gaming = True`.
   - Clears the context buffer.
   - Reverts the entire keyboard to a calm, neutral white backlight (`#FFFFFF`).
3. **Seamless Resume**:
   - Resumes predictive lighting as soon as the user presses Enter or types a non-movement character (e.g. typing in team chat). Game controls such as Space, Shift, and Ctrl remain ignored while paused.
   - Alternatively disengages if typing stops for more than 12 seconds, allowing normal between-round pauses in games.

---

### 8.3 Context State Machine & Idle Dimming

To maintain an intuitive lighting experience:
- **Empty Context State**: At startup or when backspaced to 0 characters, no red predictions are illuminated. The keyboard displays a uniform white backlight, prompting:
  ```
  [Context: (empty)] -> Start typing to see predictions...
  ```
- **Active Prediction State**: Each typed character appends to `self.rolling_buffer` (capped at `seq_len`). The model computes next-token probabilities, and the Top-K keys illuminate in graded shades of red.
- **Idle Timeout**: By default, `--idle-timeout` is 6 seconds. The live process remains active indefinitely, while prediction highlights return to solid white after inactivity. Set it to `0` to disable expiration.

---

### 8.4 Line-by-Line Code Walkthrough

1. **`render_attention_matrix(tokens, attn_weights)`**:
   - Dynamically slices the attention submatrix matching the input sequence length.
   - Renormalizes attention rows so they sum to 100% before displaying the ASCII attention heatmap.
2. **`LowLatencyInputReader`**:
   - `_start_capture()` initializes the non-blocking global hook.
   - `_enqueue(ch)` normalizes carriage returns (`\r` $\to$ `\n`), normalizes delete keys (`\x7f` $\to$ `\b`), intercepts Ctrl+C (`\x03`) to invoke `_thread.interrupt_main()`, filters control codes, debounces duplicate events (15ms window), and enqueues the token.
3. **`PredictiveKeyLightsApp.run()`**:
   - Uses `self.key_queue.get(timeout=0.005)` so the worker can process bursts while still checking idle and gaming timers.
   - Consumes characters from `self.key_queue`, updates the rolling buffer, manages idle timeouts, and calls `_update_prediction()`.
4. **`_update_prediction()`**:
   - Evaluates the rolling buffer through `self.model.forward()`.
   - Checks for `np.isnan` or low-confidence distributions ($< 0.02$).
   - Slices top predictions using `np.argsort(probs)[::-1][:self.top_k]`.
   - Resolves characters to key names via `char_to_key_name(ch)`.
   - Invokes `self.kbd.set_key_colors()` with the ranked color palette (`#FF0000`, `#FF3333`, `#FF5555`, `#FF7777`, `#FF9999`).

---

## 9. Edge Cases, Defenses & Reliability Engineering

This table is a map from a symptom to the code decision that prevents it. Read the first column when something looks wrong, then check the proposed solution in the named module. These are not abstract best practices: each row records a failure mode that can make the demo appear incorrect, hang, or light the wrong physical switch.

| Vulnerability / Edge Case | Failure Mode Before Fix | Engineered Solution |
| :--- | :--- | :--- |
| **Startup Empty Context** | Model predicted on space (`" "`), lighting `F4`, `Y`, `1` in red before typing began. | If `len(rolling_buffer) == 0`, inference is bypassed and keyboard remains 100% white. |
| **Missing Background Keys** | Background fill only looped over `slot_map.values()`. Keys like `f`, `j`, `p`, `=`, `[`, `]`, `f11`, `ctrl`, `shift`, `del` stayed dark. | Populates the entire 128-slot buffer unconditionally: `bytearray([bg_r, bg_g, bg_b] * num_slots)`. |
| **Physical Matrix Offset** | A generic 104-key column formula shifted letters onto function and number rows (`W` $\to$ `1`, `T` $\to$ `F4`, `H` $\to$ `Y`). | Mapped all 82 switches to the verified slots in `profiles/hive75.json`. |
| **Keystroke Hook Freezing** | Waiting on a hook thread could prevent the console reader from running. | Starts `pynput` with non-blocking `start()` and runs the platform console reader in its own worker thread. |
| **Ctrl+C Trapping** | `msvcrt.getch()` intercepted `\x03`, preventing process termination via Ctrl+C. | Intercepts byte `b'\x03'` in console poller and invokes `_thread.interrupt_main()`. |
| **Carriage Return / DEL Corruption** | `\r` and `\x7f` treated as `<unk>` or failing to delete context on some terminals. | Normalizes `\r` to `\n` and `\x7f` to `\b` in `_enqueue()` before control filtering. |
| **Context Length Overflow** | `--context-len` greater than model's `seq_len` caused shape assertion crash. | Clamps `context_len = min(context_len, model.seq_len)` in `__init__`, and `forward()` auto-slices long sequences. |
| **Attention Shape & Renormalization** | Slicing submatrix without row-sum normalization produced sums $< 100\%$. | Dynamically aligns slice and renormalizes rows with `np.divide()` so rows sum to 100%. |
| **Stale Lockfiles & POSIX EPERM** | Active processes owned by other users raised `PermissionError`, mistakenly treated as absent. | `_is_pid_running()` returns `True` on `PermissionError` and `errno.EPERM`, preserving valid locks. |
| **Keepalive USB Lock Contention** | Frame writes and reconnects could compete for the controller lock. | `_flush_frame()` marks failed handles disconnected; the keepalive thread owns reconnection, and EVision writes drain their ACK reports. |
| **Late Keyboard Reconnection** | Startup failure marked `self.mock = True`, permanently disabling reconnect attempts. | Retains `self.mock = False` on transport failures; background thread auto-reconnects when plugged in. |
| **Reconnect Lock-Race** | `_flush_frame` calling `_connect_hid()` while holding `_lock` raced with keepalive thread reconnect. | `_flush_frame` only sets device to `None`; keepalive thread is the sole owner of reconnection. |
| **Keepalive Disabled Semantics** | Setting `keepalive_hz=0` defaulted back to 1.0 Hz instead of actually turning off pings. | When `keepalive_hz == 0`, dynamic frame pings are truly disabled, checking only for stop event. |
| **Missing Positional Awareness** | Models without positional embeddings cannot distinguish identical characters at different positions. | Added learned `W_pos[seq_len, hidden_size]` table added to embeddings in `forward()`. |
| **Adam Optimizer State Loss** | Checkpoint saving omitted `_adam_m`, `_adam_v`, and step counter, resetting Adam on `--resume`. | `save_checkpoint` / `load_checkpoint` fully serialize and restore all Adam moments and step counts. |
| **Fragile Mask Comparison** | Zeroing attention gradients via `d_masked[masked_scores <= -1e8] = 0` was prone to float rounding. | Stores boolean `self.causal_mask_bool = causal_mask < 0` at init and indexes directly with it. |
| **Vocab Missing `<unk>`** | Custom vocabularies omitting `<unk>` silently fell back to index 0, corrupting token decoding. | `CharTokenizer.__init__` raises explicit `ValueError` if `<unk>` is not present in the vocabulary. |
| **Model Vocabulary Attribute Missing** | Older checkpoints or fresh models lacked `.vocab`, causing `AttributeError`. | `TinyTransformer.__init__` defines `self.vocab = None` unconditionally. |
| **Keepalive Shutdown Hang** | `time.sleep()` in keepalive loop delayed `close()` by up to 1 second. | Replaced with `threading.Event().wait(interval)` for instant termination ($0\text{ ms}$). |
| **USB Disconnect / Reconnect** | Unplugging the keyboard threw an unhandled exception and permanently halted LED updates. | Background loop checks connection state every 2 seconds and attempts automatic reconnection. |

---

## 10. Command-Line Reference & Cheat Sheet

Run the commands from the repository root. A good ground-up progression is: install dependencies, run the test suite, run the predictor in mock mode, train or load a checkpoint, and only then connect the physical keyboard. Every command below is a complete experiment; the comments explain what changes and what output to look for.

### Environment Setup
```powershell
# Install pinned runtime dependencies
pip install -r requirements.txt
```

### Running the Real-Time Predictor
```powershell
# Standard run with live prediction probabilities printed to terminal
python predict_and_light.py --show-probs

# Display live ASCII attention weight heatmaps
python predict_and_light.py --show-probs --show-attn

# Adjust LED brightness (e.g. 50% brightness)
python predict_and_light.py --show-probs --brightness 0.5

# Top-3 predictions instead of Top-5
python predict_and_light.py --show-probs --top-k 3

# Mock Mode (runs without physical keyboard hardware attached)
python predict_and_light.py --mock --show-probs
```

### Training the Model
```powershell
# Train analytical backpropagation model with Adam optimizer, validation split, and seed
python train.py --data data/sample_training_text.txt --epochs 15 --lr 0.003 --mode backprop --seed 42

# Resume training from checkpoint (restores weights AND full Adam optimizer state)
python train.py --data data/sample_training_text.txt --epochs 5 --resume checkpoints/model_final.npz

# Customize sliding window stride and validation holdout fraction
python train.py --data data/sample_training_text.txt --stride 2 --val-split 0.15

# Fast training using educational heuristic update
python train.py --epochs 10 --mode heuristic
```

### Manual Hardware Testing
```powershell
# Test lighting all keys white with WASD highlighted in red
python hardware_controller.py

# Light up a specific key with custom hex color
python hardware_controller.py --key w ff0000 space 00ff00
```

### Running the Test Suite
```powershell
python tests/test_suite.py
```
