import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # Conv Layers
        self.conv1 = nn.Conv2d(1, 16, 5)  # (224-5)+1/1 = 220
        self.conv2 = nn.Conv2d(16, 32, 5)  # (110-5)+1/1 = 105
        self.conv3 = nn.Conv2d(32, 64, 5)  # (52-5)+1/1 = 48
        self.conv4 = nn.Conv2d(64, 128, 3)  # (24-3)+1/1 = 22

        # Pooling layers
        self.pool = nn.MaxPool2d(2, 2)

        # Dropout Layers
        self.dropout = nn.Dropout2d(p=0.2)

        # Batch Norm
        self.batchNorm1 = nn.BatchNorm2d(16)
        self.batchNorm2 = nn.BatchNorm2d(32)
        self.batchNorm3 = nn.BatchNorm2d(64)
        self.batchNorm4 = nn.BatchNorm2d(128)
        self.batchNorm5 = nn.BatchNorm2d(256)

        # Fully connected Layers
        self.fc1 = nn.Linear(128 * 11 * 11, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, 136)

    def forward(self, x):
        # Apply convolutional layers

        x = self.batchNorm1(F.relu(self.pool(self.conv1(x))))
        x = self.batchNorm2(F.relu(self.pool(self.conv2(x))))
        x = self.batchNorm3(F.relu(self.pool(self.conv3(x))))
        x = self.batchNorm4(F.relu(self.pool(self.conv4(x))))

        # Flatten and continue with dense layers
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

        return x