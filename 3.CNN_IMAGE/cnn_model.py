# cnn_model.py

import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)  # -> 26x26
        self.pool = nn.MaxPool2d(2, 2)               # -> 13x13
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3) # -> 11x11
        self.pool2 = nn.MaxPool2d(2, 2)               # -> 5x5
        self.fc1 = nn.Linear(64 * 5 * 5, 128)         # ✅ Correct shape: 1600 → 128
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # (1, 28, 28) → (32, 13, 13)
        x = self.pool2(F.relu(self.conv2(x)))  # (64, 5, 5)
        x = x.view(-1, 64 * 5 * 5)             # flatten
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x