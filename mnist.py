import nn.Module as nn
from tensor import Tensor
from dataset import load_mnist_images, load_mnist_labels
from optim import SGD
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 8, 5)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(16, 32, 5)
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(32 * 16 * 16, 128)
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(128, 10)
        self.relu5 = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)   # 4, 8, 24, 24
        x = self.relu1(x)
        x = self.conv2(x)   # 4, 16, 20, 20
        x = self.relu2(x)
        x = self.conv3(x)   # 4, 32, 16, 16
        x = self.relu3(x)
        x = x.reshape(4, 32 * 16 * 16)  # 4, 32 * 16 * 16
        x = self.fc1(x)     # 4, 128
        x = self.relu4(x)
        x = self.fc2(x)     # 4, 10
        x = self.relu5(x)
        x = self.softmax(x) # 4, 1
        return x


def main():
    train_images = Tensor(load_mnist_images("./data/MNIST/raw/train-images-idx3-ubyte.gz").reshape(-1, 1, 28, 28).astype(np.float64) / 255)
    train_labels = Tensor(load_mnist_labels("./data/MNIST/raw/train-labels-idx1-ubyte.gz").reshape(-1, 1))

    test_images = Tensor(load_mnist_images("./data/MNIST/raw/t10k-images-idx3-ubyte.gz").reshape(-1, 1, 28, 28).astype(np.float64) / 255)
    test_labels = Tensor(load_mnist_labels("./data/MNIST/raw/t10k-labels-idx1-ubyte.gz").reshape(-1, 1))

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=0.1)

    num_train_images = train_images.shape[0]
    num_test_images = test_images.shape[0]
    for epoch in range(10):
        for i in range(0, num_train_images, 4):
            x = train_images[i:i + 4]
            y = train_labels[i:i + 4]

            optimizer.zero_grad()
            output = net(x)
            print(f'predict: {np.argmax(output.data, axis=1)},   gt: {y.data[:, 0]}')
            loss = criterion(output, y)
            loss.backward()
            optimizer.step()

            print(f"Epoch: {epoch}, Loss: {loss.data}")


if __name__ == "__main__":
    main()