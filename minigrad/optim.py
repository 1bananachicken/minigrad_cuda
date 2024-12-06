from tensor import *


class Optimizer:
    def __init__(self, params, lr, weight_decay=0.0):
        self.params = params
        self.lr = lr
        self.weight_decay = weight_decay


class SGD(Optimizer):
    def __init__(self, params: list[Tensor], lr: float = 0.001, weight_decay: float = 0.0):
        super().__init__(params, lr, weight_decay)

    def step(self):
        for p in self.params:
            if p.grad is not None:
                if self.weight_decay > 0:
                    p.data -= self.lr * (p.grad + self.weight_decay * p.data)
                else:
                    p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            if p.grad is not None:
                p.zero_grad()


class Adam(Optimizer):
    def __init__(self, params: list[Tensor], lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        super().__init__(params, lr, weight_decay)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self):
        self.t += 1
        for param in self.params:
            if param not in self.m:
                self.m[param] = np.zeros_like(param.data)
                self.v[param] = np.zeros_like(param.data)

            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * param.grad
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (param.grad ** 2)
            m_hat = self.m[param] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param] / (1 - self.beta2 ** self.t)

            if self.weight_decay > 0:
                param.data -= self.lr * (
                        m_hat / (np.sqrt(v_hat) + self.epsilon) +
                        self.weight_decay * param.data
                )
            else:
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def zero_grad(self):
        for param in self.params:
            param.zero_grad()
