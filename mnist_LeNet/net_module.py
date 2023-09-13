import torch
import torch.nn as nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x_1 = torch.relu(self.conv1(x))
        x_2 = self.maxpool1(x_1)
        x_3 = torch.relu(self.conv2(x_2))
        x_4 = self.maxpool2(x_3)
        x_5 = self.flatten(x_4)
        x_6 = torch.relu(self.fc1(x_5))
        x_7 = torch.relu(self.fc2(x_6))
        out = self.fc3(x_7)
        return out

# net = Net()
# print(net)