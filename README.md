# Transformer from Scratch

A GPT-style, decoder-only transformer written in **pure NumPy**. Every layer's forward and backward pass is implemented from scratch.

It trains a character-level language model on Tiny Shakespeare in about 90 seconds on a MacBook Air.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![NumPy](https://img.shields.io/badge/numpy-only-013243)
---

## Sample output

After 2,000 steps (84 seconds of training, 0.21M parameters, validation loss 1.82):

```
LADY MAMNESTHGBRIIS:
Which is thy surst, thou that respently, or husbanck,
The kney'll advent forth thou the are affect.

GREMIO:
The usure sicarlo honernay, and plause of Cutizen;
Whething his praclam so. Oway, you
Or face made Grther? oncheek, stond Llay save.

CLAMERS:
What the jucter, well; sthall love weep one curle:
But shall'd you thavice 
```

Not Shakespeare yet, but it has learned speaker names, line structure, punctuation and a lot of real English words, from characters alone. Training a larger model with a larger context window can improve performance substantially.

## Training run

| Step | Train loss | Val loss |
|-----:|-----------:|---------:|
| 0    | 5.088 | 5.065 |
| 400  | 2.232 | 2.264 |
| 800  | 1.913 | 2.033 |
| 1200 | 1.750 | 1.894 |
| 1600 | 1.685 | 1.851 |
| 1999 | 1.650 | 1.821 |

For reference, guessing uniformly over the 65 characters gives ln(65) ≈ 4.17. The step-0 loss is higher because the randomly initialized model is confidently wrong rather than uniformly unsure.

## Quick start

```bash
git clone https://github.com/<your-username>/transformer-from-scratch.git
cd transformer-from-scratch
pip install numpy

# download the dataset (~1 MB)
curl -O https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt

python train.py
```

Hyperparameters are at the top of `train.py`. The defaults are sized for a CPU; the values from Karpathy's lecture are noted in the comments.

## What's inside

```
transformer-from-scratch/
├── gpt/
│   ├── nn.py            # Module base class, Sequential, Linear, Embedding,
│   │                    # LayerNorm, ReLU, Dropout, cross-entropy loss
│   ├── optimizers.py    # AdamW
│   └── transformer.py   # attention heads, multi-head attention, feed-forward,
│                        # transformer block, sampling, gradient check
├── model.py             # GPT: embeddings → transformer blocks → LayerNorm → logits
├── train.py             # training script
└── input.txt            # Tiny Shakespeare dataset
```

**Architecture:** token and learned position embeddings → N transformer blocks (causal multi-head self-attention and a feed-forward network, each with a residual connection) → final LayerNorm → linear projection to vocabulary logits. Blocks support both **pre-norm** (GPT-2) and **post-norm** (original transformer), and the feed-forward activation is pluggable through the `act` argument.

## How backprop works without autograd

Every layer subclasses `Module` and follows one contract:

```python
class Linear(Module):
    def __call__(self, x):          # forward: compute output, cache what backward needs
        self.x = x
        return x @ self.w + self.b

    def backward(self, out_grad):   # backward: dL/d(output) in, dL/d(input) out
        self.w_grad = self.x.T @ out_grad
        self.b_grad = out_grad.sum(axis=0)
        return out_grad @ self.w.T
```

A model's backward pass is just its layers' backward passes called in reverse order. The one rule that makes the non-sequential parts work:

> **If a tensor feeds several paths in the forward pass, the gradients from those paths accumulate in the backward pass.**

That rule shows up in three places:

### The attention backward pass

For one head, with scores `S = QKᵀ/√d_k`, weights `P = softmax(S)` and output `O = PV`:

```
dV = Pᵀ · dO
dP = dO · Vᵀ
dS = P ⊙ (dP − rowsum(dP ⊙ P)) / √d_k     # softmax Jacobian, never built as a matrix
dQ = dS · K
dK = dSᵀ · Q
```

Causal masking needs no special handling in the backward pass: masked positions have `P = 0`, so their gradient is zero automatically.

## Notes

- **Apple silicon:** NumPy 2.x with Apple's Accelerate library can print spurious `divide by zero / overflow encountered in matmul` warnings. They're harmless (the losses match a Linux run exactly). To hide them, add this near the top of `train.py`:
  ```python
  import warnings
  warnings.filterwarnings("ignore", message=".*encountered in matmul", category=RuntimeWarning)
  ```
- **Speed:** everything runs on the CPU in float64. The lecture-sized model (384-dim, 6 layers, context 256) is trainable but takes hours.
- **One call per forward pass:** each module caches its inputs, so a module instance can be called only once per forward pass. Weight tying would need gradient accumulation.

## Acknowledgments

- Andrej Karpathy's [*Let's build GPT: from scratch, in code, spelled out*](https://www.youtube.com/watch?v=kCc8FmEb1nY), whose `gpt.py` training script this project's `train.py` is adapted from.
- Vaswani et al., [*Attention Is All You Need*](https://arxiv.org/abs/1706.03762) (2017).
- Tiny Shakespeare dataset from [karpathy/char-rnn](https://github.com/karpathy/char-rnn).
