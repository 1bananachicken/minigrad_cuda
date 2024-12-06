import torch
import torch.nn as nn
import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
from tqdm import tqdm
# class Net(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3)
#         self.relu1 = nn.ReLU()
#         self.conv2 = nn.Conv2d(32, 64, 3)
#         self.relu2 = nn.ReLU()
#         self.conv3 = nn.Conv2d(64, 32, 3)
#         self.relu3 = nn.ReLU()
#         self.conv4 = nn.Conv2d(32, 16, 3)
#         self.relu4 = nn.ReLU()
#         self.fc1 = nn.Linear(16 * 20 * 20, 128)
#         self.relu5 = nn.ReLU()
#         self.fc2 = nn.Linear(128, 10)
#         self.relu6 = nn.ReLU()
#         self.softmax = nn.Softmax()
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = self.relu1(x)
#         x = self.conv2(x)
#         x = self.relu2(x)
#         x = self.conv3(x)
#         x = self.relu3(x)
#         x = self.conv4(x)
#         x = self.relu4(x)
#         x = x.reshape(16, 16 * 20 * 20)
#         x = self.fc1(x)
#         x = self.relu5(x)
#         x = self.fc2(x)
#         x = self.relu6(x)
#         x = self.softmax(x)
#         return x

device = torch.device('cuda')
net = nn.Sequential(
    nn.Conv2d(1, 6, kernel_size=5, padding=2),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Conv2d(6, 16, kernel_size=5),
    nn.MaxPool2d(kernel_size=2, stride=2),
    nn.Flatten(),
    nn.Linear(16 * 5 * 5, 120), nn.ReLU(),
    nn.Linear(120, 84), nn.ReLU(),
    nn.Linear(84, 10), nn.Softmax()).to(device)

train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=Compose([ToTensor(), Normalize((0.5,), (0.5,))]))
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for i in range(100):
    total_loss = 0
    for j, (images, labels) in tqdm(enumerate(train_loader)):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = net(images)
        loss = criterion(outputs, labels)
        accuracy = (outputs.argmax(dim=1) == labels).float().mean()
        total_loss += loss.item()
        loss.backward()
        optimizer.step()
        print(f'pred: {outputs.argmax(dim=1)}, labels:{labels}, loss: {loss.item()}')
    print(f'Epoch {i + 1}, Loss: {total_loss / len(train_loader)}, Accuracy: {accuracy.item()}')
