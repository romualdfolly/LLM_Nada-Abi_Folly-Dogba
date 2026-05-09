# GPT Language Model — From Scratch

Implementation of a GPT-style language model built entirely from scratch using PyTorch, as part of the Deep Learning course taught by Professor François Fleuret.

**Authors:** Romuald Folly-Dogba, Tawfik Nada-Abi ([@El-Tawfik17](https://github.com/El-Tawfik17))

---

## Overview

This project explores the construction of a generative language model from first principles, without relying on any pre-built transformer library. Starting from raw text, we implement each component of the pipeline: tokenization, positional encoding, and the full GPT architecture. The model is trained on the TinyShakespeare dataset and evaluated on its ability to generate coherent, stylistically consistent text.

---

## Project Structure

```
.
├── LLM_Nada_Abi_Folly_Dogba.ipynb   # Main notebook
├── data/                              # Tokenized data (generated at runtime)
├── saved_models/                      # Model checkpoints
└── shakespeare.txt                    # Training corpus
```

---

## Methodology

### Tokenization

Rather than using character-level or word-level tokenization, we implemented **Byte Pair Encoding (BPE)** from scratch. The algorithm iteratively merges the most frequent adjacent token pairs in the vocabulary, yielding a compact and expressive token set.

- Number of merge operations: 1000  
- Final vocabulary size: 1014 tokens  
- The tokenizer handles both training and inference, with disk caching to avoid redundant computation.

### Positional Encoding

We use the sinusoidal positional encoding scheme introduced in "Attention Is All You Need". Each position `t` in the sequence is encoded as a fixed vector using alternating sine and cosine functions:

```
theta = t / r^k,  r = N^(2/d)

PE(t, 2i)   = sin(theta_i)
PE(t, 2i+1) = cos(theta_i)
```

This encoding is implemented as a non-trainable buffer in PyTorch, added to the token embeddings before the transformer layers.

### Model Architecture

The model follows the GPT decoder-only architecture, composed of stacked transformer blocks with masked self-attention.

| Hyperparameter        | Value  |
|-----------------------|--------|
| Context window        | 128    |
| Batch size            | 64     |
| Embedding dimension   | 768    |
| Number of layers      | 8      |
| Number of heads       | 6      |
| Total parameters      | ~53.5M |

### Training

- Dataset: TinyShakespeare (~1.1M characters)
- Train / test split: 10/11 — 1/11
- Optimizer: Adam with learning rate `3e-4`
- Steps: 6000
- Hardware: NVIDIA T4 GPU (Google Colab)
- Training time: approximately 1 hour 23 minutes

---

## Results

| Checkpoint    | Train Accuracy | Test Accuracy |
|---------------|----------------|---------------|
| 2000 steps    | 96%            | 41%           |
| 6000 steps    | 97%            | 42%           |

The model reaches high accuracy on the training set while test accuracy stabilizes around 42%, reflecting the difficulty of next-token prediction on held-out data at this scale. The train/test gap indicates that the model fits the training distribution well but generalizes to a limited degree, which is expected for a model of this size trained on a small corpus.

### Sample Generation

Prompt: `"To be, or not to be: "`

```
To be, or not to be: but that I am dead.

BENVOLIO:
Aa man else to ennoble tongue.

MERCUTIO:
But will rick approach:
Yet yet for such a coil.

ROMEO:
When granted shall sweet lady's comfort; then we have such time
That doubtle is male overthrown ee parts
The low of such some a few, whose I, but in vain
Upon as my body with my he joys with me?
Nay, tut up your gaoler for disposition,
And do very well in this defence.

MENENIUS:
Nay, these are all diseases to live.
```

The model produces text that is syntactically structured, maintains Shakespearean character dialogue format, and occasionally generates semantically coherent sentences.

---

## How to Run

**Requirements**

```
torch
matplotlib
tqdm
```

**Steps**

1. Open the notebook `LLM_Nada_Abi_Folly_Dogba.ipynb` in Google Colab or Jupyter.
2. Run all cells sequentially. The Shakespeare corpus will be downloaded automatically.
3. Tokenization results and model checkpoints are cached on disk after first computation.
4. To generate text, load a saved checkpoint and call `generate_text()` with a seed string.

---

## References

- Vaswani et al., [*Attention Is All You Need*](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)
- Simon J.D. Prince, [*Understanding Deep Learning*](https://udlbook.github.io/udlbook/)
- François Fleuret, [*The Little Book of Deep Learning*]([https://udlbook.github.io/udlbook/](https://fleuret.org/public/lbdl.pdf))
