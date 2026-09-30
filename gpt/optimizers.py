import numpy as np

# adam optimizer
class AdamW:
    def __init__(self, model, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        self.model = model
        self.params = model.get_params()  # in-place updates keep these valid
        self.lr, self.eps, self.wd = lr, eps, weight_decay
        self.b1, self.b2 = betas
        self.m = [np.zeros_like(p) for p in self.params]
        self.v = [np.zeros_like(p) for p in self.params]
        self.t = 0

    def step(self):
        grads = self.model.get_grads()  # fetched fresh: backward reassigns them
        assert len(grads) == len(self.params)
        self.t += 1
        for i, (p, g) in enumerate(zip(self.params, grads)):
            self.m[i] = self.b1 * self.m[i] + (1 - self.b1) * g
            self.v[i] = self.b2 * self.v[i] + (1 - self.b2) * g * g
            m_hat = self.m[i] / (1 - self.b1 ** self.t)
            v_hat = self.v[i] / (1 - self.b2 ** self.t)
            if self.wd and p.ndim >= 2:  # decay matrices only
                p -= self.lr * self.wd * p
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)