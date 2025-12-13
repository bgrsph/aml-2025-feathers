"""
Generic training utilities for PyTorch models.
Contains reusable training and validation functions.
"""

import torch
from validate import validate


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Train the model for one epoch.
    
    Args:
        model: PyTorch model to train
        loader: DataLoader for training data
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on (cuda/cpu/mps)
        
    Returns:
        Average loss for the epoch
    """
    model.train()
    running_loss = 0

    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)


def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs):
    print(f"Training model for {num_epochs} epochs...")
    
    # Use train_loader for validation if val_loader is not provided
    validation_loader = val_loader if val_loader is not None else train_loader
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_acc = validate(model, validation_loader, device, criterion)
        print(f" Val Acc: {val_acc:.4f}")
    
    print(f"Training completed!")
    return model
