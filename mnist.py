import minigrad.nn.Module as nn
from minigrad.tensor import Tensor
from minigrad.dataset import load_mnist_images, load_mnist_labels
from minigrad.optim import SGD, Adam
from tqdm import tqdm
import numpy as np


class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, 5, 2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.pool2 = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(16 * 5 * 5, 240)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(240, 120)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(120, 84)
        self.relu3 = nn.ReLU()
        self.fc4 = nn.Linear(84, 10)
        self.softmax = nn.Softmax()

    def forward(self, x):
        x = self.conv1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.pool2(x)
        x = x.reshape(x.shape[0], 16 * 5 * 5)
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.relu3(x)
        x = self.fc4(x)
        x = self.softmax(x)
        return x


def main():
    training = False
    batchsize = 128
    epochs = 80
    dataset = 'MNIST'
    # dataset = 'FASION_MNIST'
    train_images = Tensor(load_mnist_images(f"./data/{dataset}/raw/train-images-idx3-ubyte.gz").reshape(-1, 1, 28, 28).astype(np.float32) / 255)
    train_labels = Tensor(load_mnist_labels(f"./data/{dataset}/raw/train-labels-idx1-ubyte.gz").reshape(-1, 1))

    test_images = Tensor(load_mnist_images(f"./data/{dataset}/raw/t10k-images-idx3-ubyte.gz").reshape(-1, 1, 28, 28).astype(np.float32) / 255)
    test_labels = Tensor(load_mnist_labels(f"./data/{dataset}/raw/t10k-labels-idx1-ubyte.gz").reshape(-1, 1))

    net = Net()
    net.load_state_dict('./mnist_model.pth')
    criterion = nn.CrossEntropyLoss()
    optimizer = Adam(net.parameters(), lr=0.0001)

    num_train_images = train_images.shape[0]
    num_test_images = test_images.shape[0]
    if training:
        net.train()
        for epoch in range(epochs):
            iter_accuracy = 0.0
            iter_loss = 0.0
            j = 0
            for i in tqdm(range(0, num_train_images, batchsize)):
                j += 1
                if i + batchsize > num_train_images:
                    continue
                optimizer.zero_grad()
                x = train_images[i:i + batchsize]
                y = train_labels[i:i + batchsize]

                output = net(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()
                acc = np.sum(np.argmax(output.data, axis=1) == train_labels[i:i + batchsize].data.flatten())
                iter_accuracy += acc / batchsize
                iter_loss += loss.data

            iter_accuracy /= j
            iter_loss /= j
            print(f"Epoch: {epoch}, acc: {iter_accuracy}, loss: {iter_loss}")

        net.save(net.state_dict(), "mnist_model.pth")
        # net.save(net.state_dict(), "fashion_mnist_model.pth")

    acc = 0.0
    net.eval()
    for i in range(num_test_images):
        output = net(test_images[i])
        acc += np.sum(np.argmax(output.data) == test_labels[i].data[0])
        print(f"Prediction: {np.argmax(output.data)}, Ground Truth: {test_labels[i].data[0]}")

    acc /= num_test_images
    print(f"Accuracy: {acc}")


if __name__ == "__main__":
    main()
