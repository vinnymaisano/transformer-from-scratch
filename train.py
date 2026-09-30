import time
import numpy as np

from gpt.transformer import CrossEntropyLoss, AdamW, generate
from model import GPT

# hyperparameters
batch_size = 32
block_size = 64
max_iters = 2000
eval_interval = 200
learning_rate = 1e-3
eval_iters = 20
n_embd = 64
n_head = 4
n_layer = 4
dropout = 0.0

np.random.seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

# all the unique characters that occur in this text
chars = sorted(set(text))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join(itos[i] for i in l)

# train and validation splits
data = np.array(encode(text), dtype=np.int64)
n = int(0.9 * len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    d = train_data if split == "train" else val_data
    ix = np.random.randint(0, len(d) - block_size, batch_size)
    x = np.stack([d[i:i + block_size] for i in ix])
    y = np.stack([d[i + 1:i + block_size + 1] for i in ix])
    return x, y

# average loss over eval_iters batches
def estimate_loss():
    out = {}
    model.eval()
    for split in ("train", "val"):
        losses = np.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            losses[k] = loss_fn(model(X), Y)
        out[split] = losses.mean()
    model.train()
    return out

model = GPT(vocab_size, block_size, n_embd, n_head, n_layer, dropout=dropout)
loss_fn = CrossEntropyLoss()
optimizer = AdamW(model, lr=learning_rate, weight_decay=0.01)

print(f"{model.get_param_count() / 1e6:.3f} M parameters")

t0 = time.time()
for it in range(max_iters):

    # evaluate the loss every eval_interval steps
    if it % eval_interval == 0 or it == max_iters - 1:
        losses = estimate_loss()
        print(f"step {it}: train loss {losses['train']:.4f}, "
              f"val loss {losses['val']:.4f}  ({time.time() - t0:.0f}s)")

    # sample a batch of data
    xb, yb = get_batch("train")

    # evaluate the loss, backpropagate, update
    logits = model(xb)
    loss = loss_fn(logits, yb)
    model.backward(loss_fn.backward())
    optimizer.step()

# generate from the model, starting from token 0 (the newline character)
context = np.zeros((1, 1), dtype=np.int64)
print(decode(generate(model, context, max_new_tokens=500, block_size=block_size)[0].tolist()))