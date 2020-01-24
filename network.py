import torch
import torch.nn as nn
import torch.nn.functional as F



class QNetwork(nn.Module):

    def __init__(self, num_action):
        super(QNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=8, stride=4)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=5, stride=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64*4*4, 512)
        self.head = nn.Linear(512, num_action)

    # 入力: (Samples, Channels, Height, Width)の4D_Tensor
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # print(x.size())
        x = F.relu(self.fc1(x.view(x.size(0), -1)))
        return self.head(x)
