import numpy as np


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
