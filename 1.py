import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as f
from torchvision import datasets, transforms
from torch.utils.data import DataLoader


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


net = Net()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.001)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(
    root='./data',
    train=True,
    transform=transform,
    download=True
)

train_loader = DataLoader(dataset=train_dataset, batch_size=4, shuffle=True)

for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        print(f'predict: {torch.argmax(outputs.data, dim=1)},   gt: {labels.data}')
        loss.backward()
        optimizer.step()
