import numpy as np
from nn.ops.src import conv2d, matadd, pooling
import torch
import torch.nn.functional as F

batch_size = 1
in_channels = 48
out_channels = 128
in_height = 32
in_width = 32

# x = np.random.randn(batch_size, in_channels, in_height, in_width).astype(np.float32)
# x2 = np.random.randn(batch_size, in_channels, in_height, in_width).astype(np.float32)
# kernel = np.random.randn(out_channels, in_channels, 3, 3).astype(np.float32)
# x = np.array([[[[1, 2, 2, 3],
#                 [3, 1, 2, 1],
#                 [3, 3, 1, 2],
#                 [1, 1, 2, 3]]]]).astype(np.float32)

# kernel = np.array([[[[1, 2, 3],
#                      [4, 5, 6],
#                      [7, 8, 9]]]]).astype(np.float32)
# x = np.ones((batch_size, in_channels, in_height, in_width)).astype(np.float32)
# kernel = np.ones((out_channels, in_channels, 3, 3)).astype(np.float32)
# x_1 = torch.tensor(x, requires_grad=True)
# kernel_1 = torch.tensor(kernel, requires_grad=True)
# y_1 = F.conv2d(x_1, kernel_1)
# conv = torch.nn.Conv2d(in_channels, out_channels, 3, 1)
# print(conv.weight.shape)
# print(conv.bias.shape)
# loss_1 = y_1.sum()
# loss_1.backward()
# print(x_1.grad)

# dx = conv2d.conv2d_backward_dx(kernel, dout, batch_size, in_height, in_width, in_channels, out_channels, 3, 3, 1)
# dk = conv2d.conv2d_backward_dk(x, dout, batch_size, in_height, in_width, in_channels, out_channels, 3, 3, 1)
# dout = np.ones_like(y_1.detach().numpy())
# print(x.flags)
# y = conv2d.conv2d(x, kernel, batch_size, in_height, in_width, in_channels, out_channels, 3, 3, 1)

# y2 = conv2d.conv2d(x, kernel, batch_size, in_height, in_width, in_channels, out_channels, 3, 3, 1)
# y3 = conv2d.conv2d(x, kernel, batch_size, in_height, in_width, in_channels, out_channels, 3, 3, 1)
# y4 = conv2d.conv2d(x, kernel, batch_size, in_height, in_width, in_channels, out_channels, 3, 3, 1)


# dx2, dk2 = conv2d.conv2d_backward(x, kernel, dout, batch_size, in_height, in_width, in_channels, out_channels, 3, 3, 1)
# print(dk2, dx2)
# print(dx)

# print(np.allclose(y, y_1, atol=1e-6))
# print(np.allclose(dx2, x_1.grad, atol=1e-5))
# print(np.allclose(dk2, kernel_1.grad, atol=1e-4))
# print(y)

# x = np.random.randn(batch_size, out_channels).astype(np.float32)
# y = np.random.randn(1, out_channels).astype(np.float32)

# z_1 = x + y
# z_2 = matadd.matAdd2D(x, y, batch_size, out_channels)
# print(np.allclose(z_1, z_2, atol=1e-6))

pool = torch.nn.MaxPool2d(2, return_indices=True)
x = np.random.randn(batch_size, in_channels, in_height, in_width).astype(np.float32)
y, indices = pooling.maxPool2d(x, batch_size, in_channels, in_height, in_width, 2, 2)
x_2 = torch.tensor(x, requires_grad=True)
y_2, mask_2 = pool(x_2)
loss = y_2.sum()
loss.backward()
dout = np.ones_like(y)
dx = pooling.maxPool2dBackward(dout, indices, batch_size, in_channels, in_height, in_width, 2, 2)
print(np.allclose(x_2.grad.detach().numpy(), dx, atol=1e-6))
print(np.allclose(y, y_2.detach().numpy(), atol=1e-6))
