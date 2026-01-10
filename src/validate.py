import torch


def validate(model, loader, device, criterion=None):

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
                val_loss += loss.item() * images.size(0)

    accuracy = correct / total
    
    # Return both accuracy and loss
    if criterion is not None:
        avg_loss = val_loss / total
        return accuracy, avg_loss
    else:
        return accuracy


def k_fold_cross_validate(
    model_class, 
    X_data, 
    y_data, 
    dataset_class, 
    transform,
    transform_base,
    IMAGES_BASE_PATH, 
    IMAGE_SIZE, 
    num_classes, 
    device, 
    k=5, 
    num_epochs=15, 
    batch_size=32, 
    lr=0.001,
    seed=42,
    verbose=True
):

    from sklearn.model_selection import KFold
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader
    from train import train_one_epoch
    import numpy as np
    
    # Initialize k-fold
    kfold = KFold(n_splits=k, shuffle=True, random_state=seed)
    fold_results = []
    fold_accuracies = []
    
    # Convert to numpy arrays if needed
    X_array = np.array(X_data) if not isinstance(X_data, np.ndarray) else X_data
    y_array = np.array(y_data) if not isinstance(y_data, np.ndarray) else y_data
    
    if verbose:
        print(f"\nStarting {k}-Fold Cross Validation (seed={seed})")
        print(f"Total samples: {len(X_array)}, Samples per fold: ~{len(X_array)//k}")
    
    # Perform k-fold split
    for fold, (train_idx, val_idx) in enumerate(kfold.split(X_array)):
        if verbose:
            print(f"\n{'='*60}")
            print(f"Fold {fold + 1}/{k}")
            print(f"Train samples: {len(train_idx)}, Val samples: {len(val_idx)}")
            print(f"{'='*60}")
        
        # Create train dataset with augmentation
        train_dataset = dataset_class(
            [IMAGES_BASE_PATH + p for p in X_array[train_idx]],
            y_array[train_idx],
            transformation=transform
        )
        
        # Create val dataset WITHOUT augmentation
        val_dataset = dataset_class(
            [IMAGES_BASE_PATH + p for p in X_array[val_idx]],
            y_array[val_idx],
            transformation=transform_base
        )
        
        # Create dataloaders
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True,
            num_workers=0,
            pin_memory=True if device.type == 'cuda' else False
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size, 
            shuffle=False,
            num_workers=0,
            pin_memory=True if device.type == 'cuda' else False
        )
        
        # Initialize fresh model for each fold
        model = model_class(image_size=IMAGE_SIZE, num_classes=num_classes).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()
        
        # Track best accuracy for this fold
        best_val_acc = 0
        best_epoch = 0
        epoch_accuracies = []
        
        # Train for num_epochs
        for epoch in range(num_epochs):
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
            val_acc = validate(model, val_loader, device)
            epoch_accuracies.append(val_acc)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
            
            if verbose and ((epoch + 1) % 3 == 0 or epoch == 0):
                print(f"  Epoch {epoch+1:2d}/{num_epochs}: "
                      f"Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")
        
        fold_accuracies.append(best_val_acc)
        fold_results.append({
            'fold': fold + 1,
            'best_val_acc': best_val_acc,
            'best_epoch': best_epoch,
            'final_val_acc': epoch_accuracies[-1],
            'all_epoch_accs': epoch_accuracies
        })
        
        if verbose:
            print(f"\n  Fold {fold+1} Summary:")
            print(f"    Best Val Acc: {best_val_acc:.4f} (epoch {best_epoch})")
            print(f"    Final Val Acc: {epoch_accuracies[-1]:.4f}")
    
    # Compute statistics
    mean_acc = np.mean(fold_accuracies)
    std_acc = np.std(fold_accuracies)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"K-Fold Cross Validation Results:")
        print(f"  Mean Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")
        print(f"  Fold Accuracies: {[f'{acc:.4f}' for acc in fold_accuracies]}")
        print(f"  Min: {min(fold_accuracies):.4f}, Max: {max(fold_accuracies):.4f}")
        print(f"{'='*60}\n")
    
    return {
        'fold_results': fold_results,
        'mean_accuracy': mean_acc,
        'std_accuracy': std_acc,
        'all_fold_accuracies': fold_accuracies
    }          