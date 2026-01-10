import torch
from validate import validate


def train_one_epoch(model, loader, criterion, optimizer, device):

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


def train(model, train_loader, val_loader, criterion, optimizer, device, num_epochs, patience=None):

    print(f"Training model for {num_epochs} epochs {'with early stopping' if patience else ''}...")
    
    # Use train_loader for validation if val_loader is not provided
    validation_loader = val_loader if val_loader is not None else train_loader
    
    # Early stopping variables
    best_val_acc = 0.0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        
        # Get both validation accuracy and loss
        val_result = validate(model, validation_loader, device, criterion)
        if isinstance(val_result, tuple):
            val_acc, val_loss = val_result
            print(f"  Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        else:
            val_acc = val_result
            print(f"  Train Loss: {train_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # Early stopping logic
        if patience is not None:
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break
    
    # Restore best model if early stopping was used
    if patience is not None and best_model_state is not None:
        model.load_state_dict(best_model_state)
        print(f"Restored best model (Val Acc: {best_val_acc:.4f})")
    
    print(f"Training completed!")
    return model


