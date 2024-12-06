from minigrad.tensor import *
from minigrad.nn.ops.build import conv2d, matmul, matadd, pooling
import pickle


class Module:
    def __init__(self):
        self.is_training = True

    def forward(self, *args):
        pass

    def __call__(self, *args):
        return self.forward(*args)

    def parameters(self):
        params = []
        for name, module in self.__dict__.items():
            if isinstance(module, Module) and module.parameters() is not None:
                if isinstance(module.parameters(), tuple):
                    for p in module.parameters():
                        params.append(p)
                else:
                    params.append(module.parameters())
        return params

    def state_dict(self):
        state_dict = {}
        for name, module in self.__dict__.items():
            if isinstance(module, Module) and module.parameters() is not None:
                state_dict[name] = module.state_dict()
        return state_dict

    def load_state_dict(self, path):
        state_dict = pickle.load(open(path, 'rb'))
        for name, module in self.__dict__.items():
            if isinstance(module, Module) and module.parameters() is not None:
                module.load_state_dict(state_dict[name])

    @staticmethod
    def save(state_dict, path):
        with open(path, 'wb') as f:
            pickle.dump(state_dict, f)

    def train(self):
        for name, module in self.__dict__.items():
            if isinstance(module, Module):
                module.is_training = True

    def eval(self):
        for name, module in self.__dict__.items():
            if isinstance(module, Module):
                module.is_training = False


class Conv2d(Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, stride: int = 1,
                 bias: bool = True):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.kernel = Tensor(
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32) * 0.01)
        self.bias = Tensor(np.zeros((1, out_channels, 1, 1)).astype(np.float32)) if bias else None

        self.x = None
        self.out_shape = None

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        if x.data.ndim != 4:
            x = x.reshape((1, x.shape[0], x.shape[1], x.shape[2]))
        self.x = x
        if self.padding > 0:
            x_pad = np.pad(x.data, ((0,), (0,), (self.padding,), (self.padding,)), mode='constant', constant_values=0)
            if self.is_training:
                x = Tensor(x_pad, prev=x.prev)
            else:
                x = Tensor(x_pad)
            del x_pad

        out = conv2d.conv2d(x.data, self.kernel.data, self.stride)
        self.out_shape = (x.shape[0], self.out_channels, out.shape[2], out.shape[3])
        if self.bias is not None:
            out += self.bias.data

        if self.is_training:
            out = Tensor(out, prev=(x, self))
        else:
            out = Tensor(out)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if dout.ndim != 4:
            dout = dout.reshape(self.out_shape)

        dx, dk = conv2d.conv2d_backward(self.x.data, self.kernel.data, dout, self.stride)
        if self.bias is not None:
            db = np.sum(dout, axis=(0, 2, 3), keepdims=True)
            self.bias.grad = db

        self.kernel.grad = dk

        return dx

    def parameters(self):
        if self.bias is not None:
            return self.kernel, self.bias
        else:
            return self.kernel

    def state_dict(self):
        if self.bias is not None:
            return {'kernel': self.kernel.data, 'bias': self.bias.data}
        else:
            return {'kernel': self.kernel.data}

    def load_state_dict(self, state_dict):
        self.kernel.data = state_dict['kernel']
        if self.bias is not None:
            self.bias.data = state_dict['bias']


class Linear(Module):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.weight = Tensor(np.random.randn(in_features, out_features).astype(np.float32) * 0.01)
        self.bias = Tensor(np.zeros((1, out_features)).astype(np.float32))
        self.x = None

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        if x.data.ndim != 2:
            x = x.reshape((1, x.shape[0]))
        self.x = x
        out = matadd.matAdd2d(matmul.matmul(x.data, self.weight.data), self.bias.data)
        if self.is_training:
            out = Tensor(out, prev=(x, self))
        else:
            out = Tensor(out)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = matmul.matmul(dout, self.weight.data.T)
        dw = matmul.matmul(self.x.data.T, dout)
        db = np.sum(dout, axis=0, keepdims=True)

        self.weight.grad = dw
        self.bias.grad = db

        return dx

    def parameters(self):
        return self.weight, self.bias

    def state_dict(self):
        return {'weight': self.weight.data, 'bias': self.bias.data}

    def load_state_dict(self, state_dict):
        self.weight.data = state_dict['weight']
        self.bias.data = state_dict['bias']


class ReLU(Module):
    def __init__(self):
        super().__init__()
        self.mask = None

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        self.mask = (x.data <= 0)
        out = x.data.copy()
        out[self.mask] = 0
        if self.is_training:
            out = Tensor(out, prev=(x, self))
        else:
            out = Tensor(out)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dout[self.mask] = 0
        return dout

    def parameters(self):
        return None


class Softmax(Module):
    def __init__(self):
        super().__init__()
        self.x = None

    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        y = x.data - np.max(x.data, axis=1, keepdims=True)
        out = np.exp(y) / np.sum(np.exp(y), axis=1, keepdims=True)
        if self.is_training:
            out = Tensor(out, prev=(x, self))
        else:
            out = Tensor(out)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout

    def parameters(self):
        return None


class Sigmoid(Module):
    def __init__(self):
        super().__init__()
        self.out = None

    def forward(self, x: Tensor) -> Tensor:
        self.out = 1 / (1 + np.exp(-x.data))
        if self.is_training:
            out = Tensor(self.out, prev=(x, self))
        else:
            out = Tensor(self.out)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = dout * self.out * (1 - self.out)
        return dx

    def parameters(self):
        return None


class MaxPool2d(Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.x = None
        self.out_shape = None
        self.indices = None

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        out, self.indices = pooling.maxPool2d(x.data, self.kernel_size, self.kernel_size)
        self.out_shape = out.shape
        if self.is_training:
            out = Tensor(out, prev=(x, self))
        else:
            out = Tensor(out)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if (dout.ndim != 4):
            dout = dout.reshape(self.out_shape)
        dx = pooling.maxPool2dBackward(dout, self.indices, self.kernel_size, self.kernel_size)

        return dx

    def parameters(self):
        return None


class AvgPool2d(Module):
    def __init__(self, kernel_size: int):
        super().__init__()
        self.kernel_size = kernel_size
        self.x = None
        self.out_shape = None

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        out = pooling.avgPool2d(x.data, self.kernel_size, self.kernel_size)
        self.out_shape = out.shape
        if self.is_training:
            out = Tensor(out, prev=(x, self))
        else:
            out = Tensor(out)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        if (dout.ndim != 4):
            dout = dout.reshape(self.out_shape)
        dx = pooling.avgPool2dBackward(dout, self.kernel_size, self.kernel_size)

        return dx

    def parameters(self):
        return None


class GlobalAvgPool2d(Module):
    def __init__(self):
        super().__init__()
        self.input_shape = None

    def forward(self, x: Tensor) -> Tensor:
        self.input_shape = x.shape
        out = np.mean(x.data, axis=(2, 3), keepdims=False)
        if self.is_training:
            out = Tensor(out, prev=(x, self))
        else:
            out = Tensor(out)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        dx = dout[:, :, None, None] / (self.input_shape[2] * self.input_shape[3])

        return dx

    def parameters(self):
        return None


class BatchNorm2d(Module):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        self.gamma = Tensor(np.ones((1, num_features, 1, 1)).astype(np.float32))
        self.beta = Tensor(np.zeros((1, num_features, 1, 1)).astype(np.float32))

        self.running_mean = np.zeros((1, num_features, 1, 1))
        self.running_var = np.ones((1, num_features, 1, 1))

        self.x = None
        self.x_hat = None
        self.mean = None
        self.var = None

    def forward(self, x: Tensor) -> Tensor:
        self.x = x
        self.mean = np.mean(x.data, axis=(0, 2, 3), keepdims=True)
        self.var = np.var(x.data, axis=(0, 2, 3), keepdims=True)

        self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * self.mean
        self.running_var = self.momentum * self.running_var + (1 - self.momentum) * self.var

        if self.is_training:
            x_hat = (x.data - self.mean) / np.sqrt(self.var + self.eps)
            self.x_hat = x_hat
            out = self.gamma.data * x_hat + self.beta.data
            out = Tensor(out, prev=(x, self))
        else:
            x_hat = (x.data - self.running_mean) / np.sqrt(self.running_var + self.eps)
            out = self.gamma.data * x_hat + self.beta.data
            out = Tensor(out)
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        x, x_hat, mean, var = self.x.data, self.x_hat, self.mean, self.var
        gamma = self.gamma.data
        N = x.shape[0] * x.shape[2] * x.shape[3]

        dgamma = np.sum(dout * x_hat, axis=(0, 2, 3), keepdims=True)
        dbeta = np.sum(dout, axis=(0, 2, 3), keepdims=True)

        dx_hat = dout * gamma

        dvar = np.sum(dx_hat * (x - mean) * -0.5 * (var + self.eps) ** (-1.5), axis=(0, 2, 3), keepdims=True)

        dmean = (np.sum(dx_hat * -1 / np.sqrt(var + self.eps), axis=(0, 2, 3), keepdims=True) +
                 dvar * np.mean(-2 * (x - mean), axis=(0, 2, 3), keepdims=True))

        dx = (dx_hat / np.sqrt(var + self.eps) + dvar * 2 * (x - mean) / N + dmean / N)

        self.gamma.grad = dgamma
        self.beta.grad = dbeta

        return dx

    def parameters(self):
        return self.gamma, self.beta, self.running_mean, self.running_var

    def state_dict(self):
        return {'gamma': self.gamma, 'beta': self.beta, 'running_mean': self.running_mean, 'running_var': self.running_var}

    def load_state_dict(self, state_dict):
        self.gamma = state_dict['gamma']
        self.beta = state_dict['beta']
        self.running_mean = state_dict['running_mean']
        self.running_var = state_dict['running_var']


class Dropout(Module):
    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p
        self.mask = None

    def forward(self, x: Tensor) -> Tensor:
        if self.is_training:
            self.mask = np.random.binomial(1, self.p, size=x.shape)
            out = x.data * self.mask / (1 - self.p)
            out = Tensor(out, prev=(x, self))
            return out
        else:
            return x

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * self.mask

    def parameters(self):
        return None


class CrossEntropyLoss(Module):
    def __init__(self):
        super().__init__()
        self.pred = None
        self.target = None

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.forward(pred, target)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        epsilon = 1e-7
        if target.data.ndim == 2:
            target.data = target.data[:, 0]
        self.pred = pred
        self.target = target
        out = -np.sum(np.log(pred.data[np.arange(pred.shape[0]), target.data] + epsilon)) / target.shape[0]
        if self.is_training:
            out = Tensor(out, prev=(pred, self))
        else:
            out = Tensor(out)
        return out

    def backward(self, dout: float = 1) -> np.ndarray:
        dx = self.pred.data.copy()
        dx[np.arange(self.pred.shape[0]), self.target.data] -= 1
        return dout * dx / self.pred.shape[0]


class MSELoss(Module):
    def __init__(self):
        super().__init__()
        self.pred = None
        self.target = None

    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.forward(pred, target)

    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        self.pred = pred
        self.target = target
        out = np.mean((pred.data - target.data) ** 2) / (2 * pred.shape[0])
        if self.is_training:
            out = Tensor(out, prev=(pred, self))
        else:
            out = Tensor(out)

        return out

    def backward(self, dout: np.ndarray) -> Tensor:
        return dout * (self.pred.data - self.target.data) / self.pred.shape[0]
