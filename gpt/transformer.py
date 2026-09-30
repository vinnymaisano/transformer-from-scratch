import numpy as np
from gpt.nn import *
from gpt.optimizers import AdamW

def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)

# token and position embedding tables
class TokenAndPositionEmbedding(Module):
    def __init__(self, vocab_size, block_size, d_model):
        self.block_size = block_size
        self.tok = Embedding(vocab_size, d_model)
        self.pos = Embedding(block_size, d_model)

    def children(self):
        return [self.tok, self.pos]

    def __call__(self, idx):
        T = idx.shape[1]
        assert T <= self.block_size, f"sequence length {T} > block_size {self.block_size}"
        return self.tok(idx) + self.pos(np.arange(T))

    def backward(self, out_grad):
        self.tok.backward(out_grad)
        self.pos.backward(out_grad.sum(axis=0))  # position emb was broadcast over batch
        return None

# self-attention
class Head(Module):
    def __init__(self, d_model, d_k, d_v=None, causal=True):
        if d_v is None:
            d_v = d_k
        self.d_k = d_k
        self.causal = causal
        self.W_q = Linear(d_model, d_k, bias=False)
        self.W_k = Linear(d_model, d_k, bias=False)
        self.W_v = Linear(d_model, d_v, bias=False)
        self.cache = None

    def children(self):
        return [self.W_q, self.W_k, self.W_v]

    def __call__(self, x):
        T = x.shape[1]
        Q, K, V = self.W_q(x), self.W_k(x), self.W_v(x)
        S = (Q @ K.swapaxes(-2, -1)) / np.sqrt(self.d_k)
        if self.causal:
            S = np.where(np.triu(np.ones((T, T), dtype=bool), k=1), -np.inf, S)
        P = softmax(S, axis=-1)
        self.cache = (Q, K, V, P)
        return P @ V

    def __repr__(self):
        return f"<Head d_k={self.d_k} causal={self.causal}>"

    def backward(self, out_grad):
        Q, K, V, P = self.cache
        dV = P.swapaxes(-2, -1) @ out_grad
        dP = out_grad @ V.swapaxes(-2, -1)
        dS = P * (dP - np.sum(dP * P, axis=-1, keepdims=True)) / np.sqrt(self.d_k)
        dQ = dS @ K
        dK = dS.swapaxes(-2, -1) @ Q
        # x feeds Q, K and V, so its gradient is the sum of all three paths.
        return self.W_q.backward(dQ) + self.W_k.backward(dK) + self.W_v.backward(dV)


class MultiHeadAttention(Module):
    def __init__(self, d_model, num_heads, dropout=0.0, causal=True):
        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        d_k = d_model // num_heads
        self.heads = [Head(d_model, d_k, causal=causal) for _ in range(num_heads)]
        self.W_o = Linear(d_k * num_heads, d_model)
        self.dropout = Dropout(dropout)

    def children(self):
        return self.heads + [self.W_o, self.dropout]

    def __call__(self, x):
        out = np.concatenate([h(x) for h in self.heads], axis=-1)  # (B, T, H*d_k)
        return self.dropout(self.W_o(out))

    def __repr__(self):
        return f"<MultiHeadAttention heads={len(self.heads)} d_k={self.heads[0].d_k}>"

    def backward(self, out_grad):
        g = self.W_o.backward(self.dropout.backward(out_grad))
        slices = np.split(g, len(self.heads), axis=-1)  # undo the concat
        # All heads read the same x, so their input gradients add.
        return sum(h.backward(s) for h, s in zip(self.heads, slices))

# feed-forward: just 2 linear layers with a non-linearity in between - expands to 4 * d_model by default
class FeedForward(Sequential):
    def __init__(self, d_model, d_ff=None, dropout=0.0, act=ReLU):
        if d_ff is None:
            d_ff = 4 * d_model
        super().__init__(Linear(d_model, d_ff), act(), Linear(d_ff, d_model), Dropout(dropout))

# decoder-only transformer with multi head self-attention, layer normn, and feed forward
class Block(Module):
    # pre_norm = True: layer norm comes before attention
    def __init__(self, d_model, num_heads, dropout=0.0, pre_norm=True, act=ReLU):
        self.pre_norm = pre_norm
        self.ln1 = LayerNorm(d_model)
        self.attn = MultiHeadAttention(d_model, num_heads, dropout)
        self.ln2 = LayerNorm(d_model)
        self.ff = FeedForward(d_model, dropout=dropout, act=act)

    def children(self):
        return [self.ln1, self.attn, self.ln2, self.ff]

    def __call__(self, x):
        if self.pre_norm:
            x = x + self.attn(self.ln1(x))
            x = x + self.ff(self.ln2(x))
        else:
            x = self.ln1(x + self.attn(x))
            x = self.ln2(x + self.ff(x))
        return x

    def __repr__(self):
        return f"<Block pre_norm={self.pre_norm}>"

    def backward(self, out_grad):
        # Each skip connection y = x + f(x) gives dx = dy + f.backward(dy):
        # the identity path passes the gradient through, the branch adds to it.
        g = out_grad
        if self.pre_norm:
            g = g + self.ln2.backward(self.ff.backward(g))
            g = g + self.ln1.backward(self.attn.backward(g))
        else:
            g = self.ln2.backward(g)
            g = g + self.ff.backward(g)
            g = self.ln1.backward(g)
            g = g + self.attn.backward(g)
        return g

def generate(model, idx, max_new_tokens, block_size, temperature=1.0):
    model.eval()
    for _ in range(max_new_tokens):
        logits = model(idx[:, -block_size:])
        probs = softmax(logits[:, -1, :] / temperature, axis=-1)
        nxt = np.array([[np.random.choice(len(p), p=p)] for p in probs])
        idx = np.concatenate([idx, nxt], axis=1)
    model.train()
    return idx

from model import GPT

# ----------------------------------------------------------------------------
# Checks and demo
# ----------------------------------------------------------------------------
def gradient_check(model, idx, tgt, n_checks=6, h=1e-6):
    """
    Worst error between analytic and finite-difference gradients.
    Errors on tiny gradients are measured in absolute terms (1e-3 floor), since
    float round-off dominates there. If a ReLU model fails, try a smaller h:
    a pre-activation near 0 can be pushed across the kink by the perturbation.
    """
    loss_fn = CrossEntropyLoss()
    loss_fn(model(idx), tgt)
    model.backward(loss_fn.backward())
    analytic = [g.copy() for g in model.get_grads()]

    worst = 0.0
    for p, g in zip(model.get_params(), analytic):
        for _ in range(n_checks):
            i = tuple(np.random.randint(s) for s in p.shape)
            old = p[i]
            p[i] = old + h; lp = loss_fn(model(idx), tgt)
            p[i] = old - h; lm = loss_fn(model(idx), tgt)
            p[i] = old
            num = (lp - lm) / (2 * h)
            worst = max(worst, abs(num - g[i]) / max(1e-3, abs(num) + abs(g[i])))
    return worst


if __name__ == "__main__":
    np.random.seed(0)
    vocab, T, B, C = 11, 6, 3, 16
    idx = np.random.randint(0, vocab, (B, T))
    tgt = np.random.randint(0, vocab, (B, T))

    variants = {
        "GPT (pre-norm)": GPT(vocab, 8, C, 4, 2),
        "GPT (post-norm, GELU)": GPT(vocab, 8, C, 2, 2, pre_norm=False, act=ReLU),
        "custom Sequential": Sequential(
            TokenAndPositionEmbedding(vocab, 8, C),
            Block(C, 4, pre_norm=True),
            Block(C, 2, pre_norm=False, act=ReLU),
            Linear(C, vocab),
        ),
    }
    for name, m in variants.items():
        err = gradient_check(m, idx, tgt)
        print(f"gradient check [{name:22s}] worst err = {err:.2e}",
              "PASS" if err < 1e-4 else "FAIL")

    text = "hello transformer! " * 50
    chars = sorted(set(text))
    stoi = {c: i for i, c in enumerate(chars)}
    data = np.array([stoi[c] for c in text])

    np.random.seed(1)
    block = 16
    model = GPT(len(chars), block, d_model=32, num_heads=4, num_layers=2)
    loss_fn = CrossEntropyLoss()
    opt = AdamW(model, lr=3e-3)
    print(f"\nparams: {model.get_param_count():,}")

    for step in range(301):
        ix = np.random.randint(0, len(data) - block - 1, 16)
        x = np.stack([data[i:i + block] for i in ix])
        y = np.stack([data[i + 1:i + block + 1] for i in ix])
        loss = loss_fn(model(x), y)
        model.backward(loss_fn.backward())
        opt.step()
        if step % 100 == 0:
            print(f"step {step:4d}  loss {loss:.4f}")

    out = generate(model, np.array([[stoi["h"]]]), 40, block, temperature=0.5)
    print("sample:", "".join(chars[i] for i in out[0]))