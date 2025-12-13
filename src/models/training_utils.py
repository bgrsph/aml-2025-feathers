"""
Common training and validation utilities for all models
"""

import torch
import torch.nn as nn
import pandas as pd
from torch.utils.data import DataLoader


def validate(model, val_loader, device, criterion=None):
    """
    Generic validation function that works with most models
    
    Args:
        model: PyTorch model
        val_loader: Validation DataLoader  
        device: torch.device('cuda' or 'cpu')
        criterion: Optional loss function
        
    Returns:
        acc: Accuracy
        val_loss: Average validation loss (if criterion provided, else None)
    """
    model.eval()
    correct, total = 0, 0
    val_loss = 0.0 if criterion else None
    
    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            if criterion:
                loss = criterion(outputs, labels)
                val_loss += loss.item()
    
    acc = correct / total
    if criterion:
        return acc, val_loss / len(val_loader)
    return acc


def run_ablation(models, transforms_dict, train_dataset_class, val_dataset_class,
                 X_train, y_train, X_val, y_val, TRAIN_IMAGES_BASE_PATH,
                 transform_base, IMAGE_SIZE, device, seeds=[42, 123, 777], epochs=15):
    """
    Run systematic ablation study across all combinations of models and transforms
    
    Args:
        models: Dictionary of model names to model classes
        transforms_dict: Dictionary of transform names to transforms
        train_dataset_class: Dataset class for training
        val_dataset_class: Dataset class for validation
        X_train, y_train: Training data
        X_val, y_val: Validation data
        TRAIN_IMAGES_BASE_PATH: Base path for training images
        transform_base: Baseline transform for validation
        IMAGE_SIZE: Image size
        device: torch.device
        seeds: List of random seeds to use
        epochs: Number of epochs to train
        
    Returns:
        DataFrame with results
    """
    import torch.optim as optim
    
    results = []
    
    for model_name, model_class in models.items():
        for transform_name, transform in transforms_dict.items():
            print(f"\n Running {model_name} + {transform_name}")

            for seed in seeds:
                torch.manual_seed(seed)
                print(f"  Seed {seed}...")
                
                # Create datasets
                train_dataset = train_dataset_class(
                    [TRAIN_IMAGES_BASE_PATH + p for p in X_train],
                    y_train, transformation=transform
                )
                val_dataset = val_dataset_class(
                    [TRAIN_IMAGES_BASE_PATH + p for p in X_val],
                    y_val, transformation=transform_base  # Val always fixed
                )
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                
                # Train model
                model = model_class(IMAGE_SIZE).to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                criterion = nn.CrossEntropyLoss()
                
                best_val_acc = 0
                for epoch in range(epochs):
                    # Training loop 
                    model.train()
                    running_loss = 0.0

                    for images, labels in train_loader:
                        images, labels = images.to(device), labels.to(device)
                        optimizer.zero_grad()
                        outputs = model(images)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        running_loss += loss.item()
                
                # Final validation
                val_acc = validate(model, val_loader, device)
                best_val_acc = max(best_val_acc, val_acc)
                print(f"    Epoch {epoch+1}: Val Acc={val_acc:.3f}")
                
                results.append({
                    'model': model_name,
                    'augmentation': transform_name,
                    'seed': seed,
                    'best_val_acc': best_val_acc,
                    'params': sum(p.numel() for p in model.parameters())
                })
    
    return pd.DataFrame(results)


def run_trial(hparams, model_class, train_dataset, val_dataset, 
              num_workers, pin_memory, device, num_classes=200, epochs=3):
    """
    Run a single training trial with given hyperparameters (for grid search)
    
    Args:
        hparams: Dictionary with keys 'lr', 'weight_decay', 'batch_size'
        model_class: Model class to instantiate
        train_dataset: Training dataset
        val_dataset: Validation dataset
        num_workers: Number of workers for DataLoader
        pin_memory: Pin memory for DataLoader
        device: torch.device
        num_classes: Number of classes
        epochs: Number of epochs to train
        
    Returns:
        Final validation accuracy
    """
    import torch.optim as optim
    
    # Rebuild loaders if batch size changes
    bs = hparams['batch_size']
    train_loader = DataLoader(
        train_dataset, batch_size=bs, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_dataset, batch_size=bs, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    model = model_class(num_classes=num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=hparams['lr'],
        weight_decay=hparams['weight_decay']
    )

    for epoch in range(epochs):
        # Training
        model.train()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)
        
        train_loss = running_loss / len(train_loader.dataset)
        
        # Validation
        val_acc = validate(model, val_loader, device)
        print(f"  Epoch {epoch+1}/{epochs} | loss={train_loss:.4f} | val_acc={val_acc:.4f}")

    final_acc = validate(model, val_loader, device)
    return final_acc

