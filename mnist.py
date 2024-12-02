import nn.Module as nn
from tensor import Tensor
from dataset import load_mnist_images, load_mnist_labels
from optim import SGD
import numpy as np
import sys

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 5)
        self.conv2 = nn.Conv2d(64, 128, 5)
        self.conv3 = nn.Conv2d(128, 64, 5)
        self.conv4 = nn.Conv2d(64, 16, 5)
        self.fc1 = nn.Linear(16 * 12 * 12, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 10)
        self.relu2 = nn.ReLU()
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = x.reshape(16, 16 * 12 * 12)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.softmax(x)
        return x


def main():
    batchsize = 16
    train_images = load_mnist_images("./data/MNIST/raw/train-images-idx3-ubyte.gz").reshape(-1, 1, 28, 28).astype(np.float32) / 255
    train_labels = load_mnist_labels("./data/MNIST/raw/train-labels-idx1-ubyte.gz").reshape(-1, 1)
    one_hot_labels = np.zeros((train_labels.shape[0], 10))
    for i in range(train_labels.shape[0]):
        one_hot_labels[i, train_labels[i]] = 1
    test_images = load_mnist_images("./data/MNIST/raw/t10k-images-idx3-ubyte.gz").reshape(-1, 1, 28, 28).astype(np.float32) / 255
    test_labels = load_mnist_labels("./data/MNIST/raw/t10k-labels-idx1-ubyte.gz").reshape(-1, 1)

    net = Net()
    criterion = nn.CrossEntropyLoss()
    optimizer = SGD(net.parameters(), lr=0.001)

    num_train_images = train_images.shape[0]
    num_test_images = test_images.shape[0]
    for epoch in range(10):
        iter_loss = 0.0
        for i in range(0, num_train_images, batchsize):
            if i + batchsize > num_train_images:
                continue
            x = train_images[i:i + batchsize]
            x = Tensor(x)
            y = one_hot_labels[i:i + batchsize]
            y = Tensor(y)
            optimizer.zero_grad()
            output = net(x)

            loss = criterion(output, y)
            loss.backward()
            optimizer.step()
            iter_loss += loss.data
            print(f'predict: {np.argmax(output.data, axis=1)},   gt: {y.data[:, 0]}      loss: {loss.data}')
            if (i + 1) % 100 == 0:
                print(f"Epoch: {epoch}, Iter: {i + 1}, Loss: {iter_loss / (i + 1)}")
        iter_loss /= num_train_images
        print(f"Epoch: {epoch}, Loss: {iter_loss}")


if __name__ == "__main__":
    main()