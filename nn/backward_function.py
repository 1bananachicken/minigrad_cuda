from CNN.tensor import *
from CNN.nn.ops import _Conv


class AddBackward:
    def __call__(self, x: Tensor, dout: np.ndarray) -> np.ndarray:
        return dout


class ConvolutionBackward:
    def __init__(self,  kernel, stride, bias):
        super().__init__()
        self.kernel = kernel
        self.stride = stride
        self.bias = bias

    def __call__(self, x: Tensor, dout: np.ndarray, out_shape: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if dout.ndim != 4:
            dout = dout.reshape(out_shape)

        _conv = _Conv(self.stride)
        out_h = (x.shape[2] - self.kernel.shape[2]) // self.stride + 1
        out_w = (x.shape[3] - self.kernel.shape[3]) // self.stride + 1
        out_channels = self.kernel.shape[0]
        in_channels = self.kernel.shape[1]
        batch_size = x.shape[0]

        dk = np.zeros_like(self.kernel.data)
        db = np.zeros_like(self.bias.data)
        dx = np.zeros_like(x.data)

        if self.bias is not None:
            db = np.sum(dout, axis=(0, 2, 3), keepdims=True)

        for n in range(batch_size):
            for ic in range(out_channels):
                for jc in range(in_channels):
                    for ih in range(out_h):
                        for iw in range(out_w):
                            x_start = ih * self.stride
                            x_end = x_start + self.kernel.shape[2]
                            y_start = iw * self.stride
                            y_end = y_start + self.kernel.shape[3]
                            region = x.data[n, jc, x_start:x_end, y_start:y_end]

                            dk[ic, jc] += dout[n, ic, ih, iw] * region
                            dx[n, jc, x_start:x_end, y_start:y_end] += dout[n, ic, ih, iw] * self.kernel.data[ic, jc]

        return dx, dk, db


class LinearBackward:
    def __init__(self, weight: Tensor):
        super().__init__()
        self.weight = weight

    def __call__(self, x: Tensor, dout: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if dout.ndim == 0:
            dx = dout * self.weight.data
            dw = dout * x.data
            db = dout
        else:
            dx = dout @ self.weight.data
            dw = dout.T @ x.data
            db = np.sum(dout, axis=0, keepdims=True)
        return dx, dw, db


class ReLUBackward:
    def __call__(self, x: Tensor, dout: np.ndarray) -> np.ndarray:
        if dout.ndim != x.data.ndim:
            dout = dout.reshape(x.data.shape)
        return dout * (x.data > 0)


class SoftmaxBackward:
    def __call__(self, sm: Tensor, dout: np.ndarray) -> np.ndarray:
        return dout * sm.data * (1 - sm.data)


class CrossEntropyLossBackward:
    def __call__(self, pred: Tensor, target: Tensor) -> np.ndarray:
        return -(target.data / pred.data) / pred.shape[0]


class MSELossBackward:
    def __call__(self, pred: Tensor, target: Tensor) -> np.ndarray:
        return (pred.data - target.data) / pred.shape[0]
