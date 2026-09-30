import numpy as np

# base class for all modules
class Module:
    training = True

    # sub modules
    def children(self):
        return []

    # get parameters of all sub modules
    def get_params(self):
        params = []
        for c in self.children():
            params += c.get_params()
        return params

    # get grads for all sub modules
    def get_grads(self):
        grads = []
        for c in self.children():
            grads += c.get_grads()
        return grads

    def get_param_count(self):
        return sum(p.size for p in self.get_params())

    # set sub modules to train mode
    def train(self, mode=True):
        self.training = mode
        for c in self.children():
            c.train(mode)

    # set sub modules to eval mode
    def eval(self):
        self.train(False)

    # forward pass
    def __call__(self, x):
        raise NotImplementedError

    # backward pass
    def backward(self, out_grad):
        raise NotImplementedError

    def __repr__(self):
        return f"<{type(self).__name__}>"

# container for a sequence of modules - for gradient routing
class Sequential(Module):
    def __init__(self, *modules):
        self.modules = list(modules)

    def children(self):
        return self.modules

    def __call__(self, x):
        for m in self.modules:
            x = m(x)
        return x

    def __repr__(self):
        inner = "\n".join("  " + repr(m).replace("\n", "\n  ") for m in self.modules)
        return f"<{type(self).__name__}\n{inner}\n>"

    # perform backward pass for each module
    def backward(self, out_grad):
        for m in reversed(self.modules):
            out_grad = m.backward(out_grad)
        return out_grad

# ----- layers with parameters

# linear layer
class Linear(Module):
    def __init__(self, n_in, n_out, bias=True):
        # params
        self.w = np.random.randn(n_in, n_out) * np.sqrt(1.0 / n_in)
        self.b = np.zeros(n_out) if bias else None
        
        # cache input
        self.x = None

        # grads
        self.w_grad = np.zeros_like(self.w)
        self.b_grad = np.zeros_like(self.b) if bias else None

    def get_params(self):
        return [self.w, self.b] if self.b is not None else [self.w]

    def get_grads(self):
        return [self.w_grad, self.b_grad] if self.b is not None else [self.w_grad]

    def __call__(self, x):
        self.x = x
        out = x @ self.w
        if self.b is not None:
            out = out + self.b
        return out

    def __repr__(self):
        return f"<Linear n_in={self.w.shape[0]} n_out={self.w.shape[1]} bias={self.b is not None}>"

    def backward(self, out_grad):
        # Flatten (B, T, n) to (B*T, n) so the same code handles 2-D and 3-D input.
        x2 = self.x.reshape(-1, self.x.shape[-1])
        g2 = out_grad.reshape(-1, out_grad.shape[-1])
        self.w_grad = x2.T @ g2
        if self.b is not None:
            self.b_grad = np.sum(g2, axis=0)
        return out_grad @ self.w.T

# embedding table
class Embedding(Module):
    def __init__(self, num_embeddings, d):
        self.w = np.random.randn(num_embeddings, d) * 0.02
        self.idx = None
        self.w_grad = np.zeros_like(self.w)

    def get_params(self):
        return [self.w]

    def get_grads(self):
        return [self.w_grad]

    def __call__(self, idx):
        self.idx = idx
        return self.w[idx]

    def __repr__(self):
        return f"<Embedding num={self.w.shape[0]} d={self.w.shape[1]}>"

    # route gradients to 
    def backward(self, out_grad):
        grad = np.zeros_like(self.w)
        np.add.at(grad, self.idx, out_grad)
        self.w_grad = grad
        # no grads for indices
        return None

class LayerNorm(Module):
    def __init__(self, d, eps=1e-5):
        self.eps = eps
        self.gamma = np.ones(d)
        self.beta = np.zeros(d)
        self.xhat = None
        self.rstd = None
        self.gamma_grad = np.zeros_like(self.gamma)
        self.beta_grad = np.zeros_like(self.beta)

    def get_params(self):
        return [self.gamma, self.beta]

    def get_grads(self):
        return [self.gamma_grad, self.beta_grad]

    def __call__(self, x):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        self.rstd = 1.0 / np.sqrt(var + self.eps)
        self.xhat = (x - mu) * self.rstd
        return self.gamma * self.xhat + self.beta

    def __repr__(self):
        return f"<LayerNorm d={self.gamma.shape[0]}>"

    def backward(self, out_grad):
        d = self.xhat.shape[-1]
        self.gamma_grad = (out_grad * self.xhat).reshape(-1, d).sum(axis=0)
        self.beta_grad = out_grad.reshape(-1, d).sum(axis=0)
        dxhat = out_grad * self.gamma
        return self.rstd * (
            dxhat
            - dxhat.mean(axis=-1, keepdims=True)
            - self.xhat * (dxhat * self.xhat).mean(axis=-1, keepdims=True)
        )

# layers without parameters

class ReLU(Module):
    def __init__(self):
        self.mask = None

    def __call__(self, x):
        self.mask = x > 0
        return x * self.mask

    # route gradients only to elements > 0
    def backward(self, out_grad):
        return out_grad * self.mask

# inverted dropout - no effect when p == 0 or in eval mode
class Dropout(Module):

    def __init__(self, p=0.0):
        self.p = p
        self.mask = None

    def __call__(self, x):
        if not self.training or self.p == 0.0:
            self.mask = None
            return x
        self.mask = (np.random.rand(*x.shape) >= self.p) / (1.0 - self.p)
        return x * self.mask

    def __repr__(self):
        return f"<Dropout p={self.p}>"

    def backward(self, out_grad):
        return out_grad if self.mask is None else out_grad * self.mask

# loss function
class CrossEntropyLoss:
    def __init__(self):
        self.cache = None

    def __call__(self, logits, targets):
        V = logits.shape[-1]
        flat = logits.reshape(-1, V)
        t = targets.reshape(-1)
        shifted = flat - flat.max(axis=-1, keepdims=True)
        log_probs = shifted - np.log(np.exp(shifted).sum(axis=-1, keepdims=True))
        self.cache = (np.exp(log_probs), t, logits.shape)
        return -log_probs[np.arange(len(t)), t].mean()

    def backward(self):
        probs, t, shape = self.cache
        g = probs.copy()
        g[np.arange(len(t)), t] -= 1.0  # softmax - one_hot
        return (g / len(t)).reshape(shape)