"""
Unified validation utilities for PyTorch models.
Provides flexible validation function that can optionally calculate loss.
"""

import torch


def validate(model, loader, device, criterion=None):
    """
    Validate the model with optional loss calculation.
    
    Args:
        model: PyTorch model to validate
        loader: DataLoader for validation data
        device: Device to run validation on (cuda/cpu/mps)
        criterion: Optional loss function. If provided, returns (accuracy, loss)
        
    Returns:
        If criterion is None: accuracy (float)
        If criterion is provided: (accuracy, loss) tuple
    """
    model.eval()
    correct = 0
    total = 0
    val_loss = 0.0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            preds = outputs.argmax(1)
            
            correct += (preds == labels).sum().item()
            total += labels.size(0)
            
            # Calculate loss if criterion is provided
            if criterion is not None:
                loss = criterion(outputs, labels)
                val_loss += loss.item()
    
    accuracy = correct / total
    
    if criterion is None:
        return accuracy
    else:
        return accuracy, val_loss / len(loader)
