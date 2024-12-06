import numpy as np


class Tensor:
    def __init__(self, data: np.ndarray, requires_grad: bool = True, prev=None):
        self.shape = tuple(data.shape)
        self.data = data
        self.grad = 0
        self.prev = prev
        self.requires_grad = requires_grad

    def reshape(self, *args):
        self.data = self.data.reshape(*args)
        self.shape = tuple(self.data.shape)
        return self

    def zero_grad(self):
        self.grad = 0

    def __getitem__(self, item):
        return Tensor(self.data[item])

    def __setitem__(self, key, value):
        self.data[key] = value

    def __add__(self, other):
        return Tensor(self.data + other.data, prev=[self, other])

    def __sub__(self, other):
        return Tensor(self.data - other.data, prev=[self, other])

    def __mul__(self, other):
        return Tensor(self.data * other.data, prev=[self, other])

    def backward(self, dout=None):
        if dout is None:
            dout = np.ones_like(self.data)
        if self.prev is not None:
            dout = self.prev[1].backward(dout)
            self.prev[0].backward(dout)


