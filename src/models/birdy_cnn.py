import torch
import torch.nn as nn
import torch.nn.functional as F


class BirdyCNN(nn.Module):
    def __init__(self, image_size=224, num_classes=200, kernel_size=3, dropout_rate=0.0):
        super().__init__()
        
        # Fixed number of filters: [32, 64, 128]
        self.num_filters = [32, 64, 128]
        self.kernel_size = kernel_size
        self.dropout_rate = dropout_rate
        
        # Calculate feature size based on kernel size and padding
        # With padding=0, each conv reduces size by (kernel_size - 1)
        # Each pool reduces size by factor of 2
        size = image_size
        for _ in range(3):  # 3 conv-pool blocks
            size = size - (kernel_size - 1)  # conv with padding=0
            size = size // 2  # maxpool
        self.feature_size = self.num_filters[2] * size * size
        
        self.conv1 = nn.Conv2d(3, self.num_filters[0], kernel_size, padding=0)
        self.conv2 = nn.Conv2d(self.num_filters[0], self.num_filters[1], kernel_size, padding=0)
        self.conv3 = nn.Conv2d(self.num_filters[1], self.num_filters[2], kernel_size, padding=0)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout(dropout_rate)
        self.fc1 = nn.Linear(self.feature_size, 512)
        self.fc2 = nn.Linear(512, num_classes)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(F.relu(self.conv3(x)))
        x = self.dropout(x)
        x = x.view(-1, self.feature_size)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x
