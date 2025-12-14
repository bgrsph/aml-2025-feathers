import torch
import torch.nn as nn
import torch.nn.functional as F


class AttrCNN(nn.Module):
    def __init__(self, image_size=224, num_classes=200, num_attrs=312, attributes=None):

        super().__init__()

        # Store attributes as a buffer (not a parameter, but part of model state)
        if attributes is not None:
            self.register_buffer('attributes', torch.tensor(attributes, dtype=torch.float32))
        else:
            self.attributes = None

        # -------------------------
        # Simple CNN backbone
        # -------------------------
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)

        final_size = image_size // 8
        feat_size = 128 * final_size * final_size

        # -------------------------
        # Attribute projection
        # -------------------------
        self.attr_fc = nn.Linear(num_attrs, 128)

        # -------------------------
        # Fusion MLP
        # -------------------------
        self.fc1 = nn.Linear(feat_size + 128, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, labels):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        img_feat = x.view(x.size(0), -1)


        if self.attributes is None:
            raise ValueError("Attributes not provided to model")
            
        attr_tensor = self.attributes.to(x.device)            # [200, num_attrs]
        class_attr = attr_tensor[labels]                      # [B, num_attrs]
        attr_feat = F.relu(self.attr_fc(class_attr))          # [B, 128]


        fused = torch.cat([img_feat, attr_feat], dim=1)
        z = F.relu(self.fc1(fused))
        z = self.dropout(z)
        return self.fc2(z)


def validate_with_attrs(model, val_loader, device):
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images, labels)
            preds = outputs.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total

