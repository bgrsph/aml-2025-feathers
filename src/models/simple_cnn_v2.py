"""
SimpleCNN-v2: Deeper CNN model for ablation study
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SimpleCNNv2(nn.Module):
    """Deeper model with extra convolutional layer"""
    def __init__(self, image_size=224, num_classes=200):
        super().__init__()
        self.feature_size = 256 * (image_size // 16) ** 2  # 5 pools = /16
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)  # EXTRA layer
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))  # EXTRA pool
        x = x.view(-1, self.feature_size)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

