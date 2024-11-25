from CNN.nn.backward_function import *
from CNN.nn.ops import _Conv
import numpy as np


class Module:
    def __init__(self):
        pass

    def forward(self, *args):
        pass

    def __call__(self, *args):
        return self.forward(*args)

    def parameters(self):
        params = []
        for name, module in self.__dict__.items():
            if module.parameters() is not None:
                if isinstance(module.parameters(), tuple):
                    for p in module.parameters():
                        params.append(p)
                else:
                    params.append(module.parameters())

        return params


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, stride: int = 1,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.kernel = Tensor(np.random.uniform(-0.1, 0.1, (out_channels, in_channels, kernel_size, kernel_size)) * np.sqrt(2 / in_channels))
        self.bias = Tensor(np.random.uniform(-0.1, 0.1, (1, out_channels, 1, 1))) if bias else None

        self.x = None
        self.backward_function = ConvolutionBackward(self.kernel, self.stride, self.bias)
        self.out_shape = None

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        if self.padding > 0:
            x_pad = np.zeros((x.shape[0], x.shape[1], x.shape[2] + 2 * self.padding, x.shape[3] + 2 * self.padding))
            for n in range(x.shape[0]):
                for c in range(x.shape[1]):
                    x_pad[n, c] = np.pad(x.data[n, c], self.padding, 'constant')

            x = Tensor(x_pad, prev=x.prev)
            del x_pad

        _conv = _Conv(self.stride)
        out_h = (x.shape[2] - self.kernel_size) // self.stride + 1
        out_w = (x.shape[3] - self.kernel_size) // self.stride + 1
        batch_size = x.shape[0]
        out = np.zeros((batch_size, self.out_channels, out_h, out_w))
        self.out_shape = out.shape

        for n in range(batch_size):
            for i in range(self.out_channels):
                for j in range(self.in_channels):
                    out[n, i] += _conv(x.data[n, j], self.kernel.data[i, j])

        if self.bias is not None:
            out += self.bias.data

        out = Tensor(out, prev=(x, self))
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx, dk, db = self.backward_function(self.x, dout, self.out_shape)
        # _conv = _Conv(self.stride)
        # out_h = (self.x.shape[2] - self.kernel_size) // self.stride + 1
        # out_w = (self.x.shape[3] - self.kernel_size) // self.stride + 1
        # batch_size = self.x.shape[0]
        #
        # dk = np.zeros_like(self.kernel.data)
        # db = np.zeros_like(self.bias.data)
        # dx = np.zeros_like(self.x)
        #
        # if self.bias is not None:
        #     db = np.sum(dout, axis=(0, 2, 3), keepdims=True)
        #
        # for n in range(batch_size):
        #     for ic in range(self.out_channels):
        #         for jc in range(self.in_channels):
        #             for ih in range(out_h):
        #                 for iw in range(out_w):
        #                     x_start = ih * self.stride
        #                     x_end = x_start + self.kernel_size
        #                     y_start = iw * self.stride
        #                     y_end = y_start + self.kernel_size
        #                     region = self.x.data[n, jc, x_start:x_end, y_start:y_end]
        #
        #                     dk[ic, jc] += dout[n, ic, ih, iw] * region
        #                     dx[n, jc, x_start:x_end, y_start:y_end] += dout[n, ic, ih, iw] * self.kernel.data[ic, jc]
        #
        self.bias.grad = db
        self.kernel.grad = dk

        return dx

    def parameters(self):
        return self.kernel, self.bias


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.weight = Tensor(np.random.uniform(-0.1, 0.1, (out_features, in_features)) * np.sqrt(2 / in_features))
        self.bias = Tensor(np.ones((1, out_features)))
        self.x = None
        self.backward_function = LinearBackward(self.weight)

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        out = Tensor(np.einsum('ij,kj->ik', x.data, self.weight.data) + self.bias.data, prev=(x, self))
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx, dw, db = self.backward_function(self.x, dout)

        self.weight.grad = dw
        self.bias.grad = db

        return dx

    def parameters(self):
        return self.weight, self.bias


class ReLU:
    def __init__(self):
        self.x = None
        self.backward_function = ReLUBackward()

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        out = Tensor(np.maximum(0, x.data), prev=(x, self))
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return self.backward_function(self.x, dout)

    def parameters(self):
        return None


class Softmax:
    def __init__(self):
        self.x = None
        self.sm = None
        self.backward_function = SoftmaxBackward()

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        x_max = np.max(x.data, axis=1, keepdims=True)
        out = Tensor(np.exp(x.data - x_max) / np.sum(np.exp(x.data - x_max), axis=1, keepdims=True), prev=(x, self))
        self.sm = out
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return self.backward_function(self.sm, dout)

    def parameters(self):
        return None


class CrossEntropyLoss:
    def __init__(self):
        self.pred = None
        self.target = None
        self.backward_function = CrossEntropyLossBackward()

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.forward(pred, target)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        epsilon = 1e-12
        pred = Tensor(np.clip(pred.data, epsilon, 1. - epsilon), prev=pred.prev)

        self.pred = pred
        self.target = target

        out = Tensor(-np.sum(target.data * np.log(pred.data)) / target.shape[0], prev=(pred, self))
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.backward_function(self.pred, self.target)


class MSELoss:
    def __init__(self):
        self.pred = None
        self.target = None
        self.backward_function = MSELossBackward()

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.forward(pred, target)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        self.pred = pred
        self.target = target
        return Tensor(np.mean((pred.data - target.data) ** 2) / (2 * pred.shape[0]), prev=(pred, self))

    def backward(self, dout: np.ndarray) -> Tensor:
        return dout * self.backward_function(self.pred, self.target)