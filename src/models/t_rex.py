import torch
import torch.nn as nn
import torch.nn.functional as F


class TRex(nn.Module):
    def __init__(self, image_size=224, num_classes=200, num_attrs=312, 
                 dropout_rate=0.5, attr_weight=0.3):
        super().__init__()
        
        self.num_attrs = num_attrs
        self.attr_weight = attr_weight
        
        # Shared CNN for both attribute prediction and classification
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout_conv = nn.Dropout2d(0.2)
        
        # Calculate feature map size after pooling (3 pools: image_size / 2^3 = image_size / 8)
        final_size = image_size // 8
        self.feature_size = 256 * final_size * final_size
        
        # Attribute prediction branch: Predicts 312 binary attributes (has_red_color, has_long_beak, etc.)
        self.attr_fc1 = nn.Linear(self.feature_size, 1024)
        self.attr_bn1 = nn.BatchNorm1d(1024)
        self.attr_dropout = nn.Dropout(dropout_rate)
        self.attr_fc2 = nn.Linear(1024, num_attrs)
        
        # Attribute embedding: Projects predicted attributes to a lower dimension
        self.attr_embed = nn.Linear(num_attrs, 256)
        self.attr_embed_bn = nn.BatchNorm1d(256)
        
        # Main classification branch: Fuses image features + attribute embeddings
        self.fc1 = nn.Linear(self.feature_size + 256, 1024)
        self.bn_fc1 = nn.BatchNorm1d(1024)
        self.dropout_fc1 = nn.Dropout(dropout_rate)
        
        self.fc2 = nn.Linear(1024, 512)
        self.bn_fc2 = nn.BatchNorm1d(512)
        self.dropout_fc2 = nn.Dropout(dropout_rate)
        
        self.fc3 = nn.Linear(512, num_classes)
        
    def forward(self, x, return_attrs=False):

        # Shared CNN feature extraction
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.dropout_conv(x)
        
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.dropout_conv(x)
        
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten
        img_feat = x.view(x.size(0), -1)  # [B, feature_size]
        
        # Attribute prediction branch
        attr_hidden = F.relu(self.attr_bn1(self.attr_fc1(img_feat)))
        attr_hidden = self.attr_dropout(attr_hidden)
        attr_logits = self.attr_fc2(attr_hidden)  # [B, num_attrs]
        
        # Use sigmoid to get attribute probabilities (binary attributes)
        attr_probs = torch.sigmoid(attr_logits)
        
        # Embed attributes
        attr_embedded = F.relu(self.attr_embed_bn(self.attr_embed(attr_probs)))
        
        # Concatenate image features + attribute embeddings
        fused = torch.cat([img_feat, attr_embedded], dim=1)
        
        # Classification head
        z = F.relu(self.bn_fc1(self.fc1(fused)))
        z = self.dropout_fc1(z)
        
        z = F.relu(self.bn_fc2(self.fc2(z)))
        z = self.dropout_fc2(z)
        
        class_logits = self.fc3(z)
        
        if return_attrs:
            return class_logits, attr_logits
        return class_logits


def compute_trex_loss(model_output, labels, attributes, attr_weight=0.3):

    class_logits, attr_logits = model_output
    
    # Classification loss
    class_loss = F.cross_entropy(class_logits, labels)
    
    # Get ground-truth attributes for the batch based on labels
    # attributes: [num_classes, num_attrs]
    # labels: [B]
    # gt_attrs: [B, num_attrs]
    gt_attrs = attributes[labels].float()
    
    # Attribute prediction loss (binary cross entropy)
    # Since attributes are binary, we use BCE loss
    attr_loss = F.binary_cross_entropy_with_logits(attr_logits, gt_attrs)
    
    # Combined loss
    total_loss = (1 - attr_weight) * class_loss + attr_weight * attr_loss
    
    return total_loss, class_loss, attr_loss


def train_trex(model, train_loader, val_loader, attributes, criterion=None, 
               optimizer=None, device='cpu', num_epochs=10, attr_weight=0.3):

    model = model.to(device)
    attributes = torch.tensor(attributes, dtype=torch.float32).to(device)
    
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        train_class_loss = 0.0
        train_attr_loss = 0.0
        correct = 0
        total = 0
        
        for i, (images, labels) in enumerate(train_loader):
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass with attribute prediction
            outputs = model(images, return_attrs=True)
            
            # Compute multi-task loss
            loss, class_loss, attr_loss = compute_trex_loss(
                outputs, labels, attributes, attr_weight=attr_weight
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Statistics
            train_loss += loss.item()
            train_class_loss += class_loss.item()
            train_attr_loss += attr_loss.item()
            
            # Calculate accuracy (using class predictions)
            class_logits = outputs[0]
            _, predicted = torch.max(class_logits.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            # Print progress
            if (i + 1) % 50 == 0:
                print(f'  Batch [{i+1}/{len(train_loader)}], '
                      f'Loss: {loss.item():.4f} '
                      f'(Class: {class_loss.item():.4f}, Attr: {attr_loss.item():.4f})')
        
        train_acc = 100 * correct / total
        avg_train_loss = train_loss / len(train_loader)
        avg_class_loss = train_class_loss / len(train_loader)
        avg_attr_loss = train_attr_loss / len(train_loader)
        
        # Validation
        val_acc = validate_trex(model, val_loader, device)
        
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {avg_train_loss:.4f} '
              f'(Class: {avg_class_loss:.4f}, Attr: {avg_attr_loss:.4f}), '
              f'Train Acc: {train_acc:.2f}%, '
              f'Val Acc: {val_acc:.2f}%')
    
    return model


def validate_trex(model, val_loader, device):
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            # Forward pass (only class predictions, no attributes needed as test data does not have attributes)
            outputs = model(images, return_attrs=False)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy
