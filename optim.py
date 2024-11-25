from tensor import *


class Optimizer:
    pass


class SGD(Optimizer):
    def __init__(self, params: list[Tensor], lr: float = 0.001):
        self.params = params
        self.lr = lr

    def step(self):
        for p in self.params:
            p.data -= self.lr * p.grad

    def zero_grad(self):
        for p in self.params:
            p.zero_grad()


class Adam(Optimizer):
    def __init__(self, parameters: list[Tensor], lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
        self.parameters = parameters
        self.learning_rate = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = eps
        self.m = {}
        self.v = {}
        self.t = 0

    def step(self):
        self.t += 1
        for param in self.parameters:
            if param not in self.m:
                self.m[param] = np.zeros_like(param.data)
                self.v[param] = np.zeros_like(param.data)

            self.m[param] = self.beta1 * self.m[param] + (1 - self.beta1) * param.grad
            self.v[param] = self.beta2 * self.v[param] + (1 - self.beta2) * (param.grad ** 2)

            m_hat = self.m[param] / (1 - self.beta1 ** self.t)
            v_hat = self.v[param] / (1 - self.beta2 ** self.t)

            param.data -= self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()
