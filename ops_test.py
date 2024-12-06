# 最他妈的重要的一个文件，测试自己写的ops和pytorch的ops输出是否一致
import numpy as np
from minigrad.nn.ops.build import conv2d, matadd, matmul, pooling
from minigrad.nn.ops.src import conv2dv2
import torch
import torch.nn.functional as F

# init params
batch_size = 16
in_channels = 16
out_channels = 32
in_height = 48
in_width = 48
kernel_size = 5
stride = 1

# test conv2d
x = np.random.randn(batch_size, in_channels, in_height, in_width).astype(np.float32)
kernel = np.random.randn(out_channels, in_channels, kernel_size, kernel_size).astype(np.float32)

x_1 = torch.tensor(x, requires_grad=True)
kernel_1 = torch.tensor(kernel, requires_grad=True)

y = conv2dv2.conv2dv2(x, kernel, stride)
y_1 = F.conv2d(x_1, kernel_1, stride=stride)

loss_1 = y_1.sum()
loss_1.backward()

dout = np.ones_like(y_1.detach().numpy())
dx, dk = conv2d.conv2d_backward(x, kernel, dout, stride)

print(f"conv forward: {np.allclose(y, y_1.detach().numpy(), atol=1e-3)}")
print(f"backward dx: {np.allclose(dx, x_1.grad.detach().numpy(), atol=1e-3)}")
print(f"backward dk: {np.allclose(dk, kernel_1.grad.detach().numpy(), atol=1e-3)}")

# test matadd
x = np.random.randn(batch_size, out_channels).astype(np.float32)
y = np.random.randn(1, out_channels).astype(np.float32)
x_2 = np.random.randn(batch_size, in_channels, in_height, in_width).astype(np.float32)
y_2 = np.random.randn(1, in_channels, 1, 1).astype(np.float32)
z_1 = x + y
z_2 = matadd.matAdd2d(x, y)
z_3 = x_2 + y_2
z_4 = matadd.matAdd4d(x_2, y_2)
print(f"matadd2: {np.allclose(z_1, z_2, atol=1e-4)}")
print(f"matadd4: {np.allclose(z_3, z_4, atol=1e-4)}")

# test matmul
x = np.random.randn(batch_size, out_channels).astype(np.float32)
p = np.random.randn(out_channels, batch_size).astype(np.float32)
y = np.random.randn(out_channels, in_channels).astype(np.float32)
z = np.random.randn(in_channels, out_channels).astype(np.float32)
z_1 = np.dot(x, y)
z_2 = matmul.matmul(x, y)
z_3 = np.dot(p.T, y)
# 他妈的，就这里卡了老子两天
z_4 = matmul.matmul(p.T, y)
z_5 = np.dot(x, z.T)
z_6 = matmul.matmul(x, z.T)
print(f"matmul: {np.allclose(z_1, z_2, atol=1e-4)}")
print(f"matmulAT: {np.allclose(z_3, z_4, atol=1e-4)}")
print(f"matmulBT: {np.allclose(z_5, z_6, atol=1e-4)}")

# test pooling
pool1 = torch.nn.MaxPool2d(2, return_indices=True)
pool2 = torch.nn.AvgPool2d(2)
x = np.random.randn(batch_size, in_channels, in_height, in_width).astype(np.float32)
y, indices = pooling.maxPool2d(x, 2, 2)
x_2 = torch.tensor(x, requires_grad=True)
y_2, mask_2 = pool1(x_2)
loss = y_2.sum()
loss.backward()
dout = np.ones_like(y)
dx = pooling.maxPool2dBackward(dout, indices, 2, 2)
print(f"maxpool forward: {np.allclose(y, y_2.detach().numpy(), atol=1e-4)}")
print(f"maxpool backward: {np.allclose(x_2.grad.detach().numpy(), dx, atol=1e-4)}")

y = pooling.avgPool2d(x, 2, 2)
x_2 = torch.tensor(x, requires_grad=True)
y_2 = pool2(x_2)
loss = y_2.sum()
loss.backward()
dout = np.ones_like(y)
dx = pooling.avgPool2dBackward(dout, 2, 2)
print(f"avgpool forward: {np.allclose(y, y_2.detach().numpy(), atol=1e-4)}")
print(f"avgpool backward: {np.allclose(x_2.grad.detach().numpy(), dx, atol=1e-4)}")
print(1)

