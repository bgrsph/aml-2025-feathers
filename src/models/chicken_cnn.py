import torch.nn as nn
import torch.nn.functional as F


class ChickenCNN(nn.Module):
    def __init__(self, num_classes=200):
        super().__init__()

        # Takes 3-channel RGB image input.
        # 32 filters = the model will learn 32 different edge/texture detectors.
        # Padding=1 keeps the spatial dimensions same after convolution.

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1) # Even deeper features (feather patterns, shapes, edges, etc.)

        self.pool = nn.MaxPool2d(2, 2)

        self.dropout = nn.Dropout(0.5) # Dropout to prevent overfitting

        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))   # 3 → 32
        x = self.pool(F.relu(self.conv2(x)))   # 32 → 64
        x = self.pool(F.relu(self.conv3(x)))   # 64 → 128
        x = x.view(x.size(0), -1)              # flatten
        
        x = self.dropout(F.relu(self.fc1(x))) 
        #x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x