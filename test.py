import nn.Module as nn
from tensor import Tensor
from dataset import load_mnist_images, load_mnist_labels
from optim import SGD
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 2, 3, 1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(2, 4, 3, 1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(4, 2, 3, 1)
        self.relu3 = nn.ReLU()
        self.conv4 = nn.Conv2d(2, 1, 3, 1)
        self.relu4 = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.relu3(x)
        x = self.conv4(x)
        x = self.relu4(x)

        return x


if __name__ == '__main__':
    net = Net()
    x = Tensor(np.random.randn(1, 1, 28, 28))
    y = Tensor(np.ones((1, 1, 28, 28)))
    criterion = nn.MSELoss()
    optimizer = SGD(net.parameters(), lr=0.001)

    for i in range(10):
        optimizer.zero_grad()
        output = net(x)
        loss = criterion(output, y)
        loss.backward()
        optimizer.step()
        print(f'Epoch {i}, Loss: {loss.data}')

    print(output.data)


