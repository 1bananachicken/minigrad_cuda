import numpy as np


class Parameters:
    def __init__(self, data: np.ndarray):
        self.data = data
        self.grad = np.zeros_like(data)

    def zero_grad(self):
        self.grad = np.zeros_like(self.grad)


class _Conv:
    def __init__(self, stride: int = 1):
        self.stride = stride

    def __call__(self, x: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        x_h, x_w = x.shape

        kernel_h, kernel_w = kernel.shape
        out_h = (x_h - kernel_h) // self.stride + 1
        out_w = (x_w - kernel_w) // self.stride + 1
        out = np.zeros((out_h, out_w))
        for i in range(out_h):
            for j in range(out_w):
                region = x[i * self.stride:i * self.stride + kernel_h, j * self.stride:j * self.stride + kernel_w]
                out[i, j] = np.einsum('ij,ij', region, kernel)

        return out


class Conv2d:
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, stride: int = 1,
                 bias: bool = True):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.padding = padding
        self.stride = stride

        self.kernel = Parameters(np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2 / in_channels) * 0.01)
        self.bias = Parameters(np.random.uniform(0, 0.1) if bias else None)

        self.x = None

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        if self.padding > 0:
            x_pad = np.zeros((x.shape[0], x.shape[1], x.shape[2] + 2 * self.padding, x.shape[3] + 2 * self.padding))
            for n in range(x.shape[0]):
                for c in range(x.shape[1]):
                    x_pad[n, c] = np.pad(x[n, c], self.padding, 'constant')

            x = x_pad.copy()
            del x_pad

        self.x = x

        _conv = _Conv(self.stride)
        out_h = (x.shape[2] - self.kernel_size) // self.stride + 1
        out_w = (x.shape[3] - self.kernel_size) // self.stride + 1
        batch_size = x.shape[0]
        out = np.zeros((batch_size, self.out_channels, out_h, out_w))

        for n in range(batch_size):
            for i in range(self.out_channels):
                for j in range(self.in_channels):
                    out[n, i] += _conv(x[n, j], self.kernel.data[i, j])

        if self.bias is not None:
            out += self.bias.data

        return out

    def backward(self, dout: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _conv = _Conv(self.stride)
        out_h = (self.x.shape[2] - self.kernel_size) // self.stride + 1
        out_w = (self.x.shape[3] - self.kernel_size) // self.stride + 1
        batch_size = self.x.shape[0]

        dk = np.zeros_like(self.kernel.data)
        db = np.zeros_like(self.bias.data)
        dx = np.zeros_like(self.x)

        if self.bias is not None:
            db = np.sum(dout, axis=(0, 2, 3), keepdims=True)

        for n in range(batch_size):
            for i in range(self.out_channels):
                for j in range(self.in_channels):
                    for x in range(out_h):
                        for y in range(out_w):
                            x_start = x * self.stride
                            x_end = x_start + self.kernel_size
                            y_start = y * self.stride
                            y_end = y_start + self.kernel_size
                            region = self.x[n, j, x_start:x_end, y_start:y_end]

                            dk[i, j] += dout[n, i, x, y] * region
                            dx[n, j, x_start:x_end, y_start:y_end] += dout[n, i, x, y] * self.kernel.data[i, j]

        self.bias.grad = db
        self.kernel.grad = dk

        return dx, dk, db


class Linear:
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

        self.weight = Parameters(np.random.randn(out_features, in_features) * np.sqrt(2 / in_features))
        self.bias = Parameters(np.ones((1, out_features)))
        self.x = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.einsum('ij,kj->ik', x, self.weight.data) + self.bias.data

    def backward(self, dout: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        dx = dout @ self.weight.data
        dw = dout.T @ self.x
        db = np.sum(dout, axis=1, keepdims=True)

        self.weight.grad = dw
        self.bias.grad = db

        return dx, dw, db


class ReLU:
    def __init__(self):
        self.x = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        return np.maximum(0, x)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        return dout * (self.x > 0)


class Softmax:
    def __init__(self):
        self.x = None

    def __call__(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x = x
        x_max = np.max(x, axis=1, keepdims=True)
        x = x - x_max
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def backward(self, dout: np.ndarray) -> np.ndarray:
        sm = self.forward(self.x)
        return dout * sm * (1 - sm)


class CrossEntropyLoss:
    def __init__(self):
        self.pred = None
        self.target = None

    def __call__(self, pred: np.ndarray, target: np.ndarray) -> float:
        return self.forward(pred, target)

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        epsilon = 1e-12
        pred = np.clip(pred, epsilon, 1. - epsilon)

        self.pred = pred
        self.target = target

        N = target.shape[0]
        loss = -np.sum(target * np.log(pred)) / N
        return loss

    def backward(self) -> np.ndarray:
        N = self.target.shape[0]
        grad = -(self.target / self.pred) / N
        return grad


class MSELoss:
    def __init__(self):
        self.pred = None
        self.target = None

    def __call__(self, pred: np.ndarray, target: np.ndarray) -> float:
        return self.forward(pred, target)

    def forward(self, pred: np.ndarray, target: np.ndarray) -> float:
        self.pred = pred
        self.target = target
        return np.mean((pred - target) ** 2) / 2

    def backward(self) -> np.ndarray:
        return self.pred - self.target


class Adam:
    def __init__(self, parameters: list, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
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


class SGD:
    def __init__(self, parameters: list, learning_rate=0.01, momentum=0.0):
        self.parameters = parameters
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocities = {param: np.zeros_like(param.data) for param in parameters}

    def step(self):
        for param in self.parameters:
            if param.data is None or param.grad is None:
                continue

            self.velocities[param] = self.momentum * self.velocities[param] - self.learning_rate * param.grad

            param.data -= self.velocities[param]

    def zero_grad(self):
        for param in self.parameters:
            param.zero_grad()


class Net:
    def __init__(self):
        self.conv1 = Conv2d(1, 32, 3, 1)
        self.conv2 = Conv2d(32, 64, 3, 1)
        self.fc1 = Linear(64 * 7 * 7, 128)
        self.fc2 = Linear(128, 10)
        self.relu1 = ReLU()
        self.relu2 = ReLU()
        self.relu3 = ReLU()
        self.relu4 = ReLU()
        self.softmax = Softmax()
        self.parameters = [self.conv1.kernel, self.conv1.bias, self.conv2.kernel, self.conv2.bias, self.fc1.weight,
                           self.fc2.weight]

    def forward(self, x: np.ndarray) -> np.ndarray:
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(16, -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

    def backward(self, dout: np.ndarray):
        dx = self.softmax.backward(dout)
        dx, dw2, db2 = self.fc2.backward(dx)
        dx, dw1, db1 = self.fc1.backward(dx)
        dx = dx.reshape(16, 64, 7, 7)
        dx, dk2, db2 = self.conv2.backward(dx)
        dx, dk1, db1 = self.conv1.backward(dx)


if __name__ == '__main__':
    a = np.random.uniform(0, 1, (16, 1, 7, 7))
    b = np.random.randint(0, 10, (16, 10))
    net = Net()
    optimizer = SGD(net.parameters, 0.01)
    loss = MSELoss()

    for i in range(100):
        out = net.forward(a)
        loss_val = loss(out, b)
        dout = loss.backward()
        net.backward(dout)
        optimizer.step()
        optimizer.zero_grad()
        print(loss_val)
