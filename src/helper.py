import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
from train import train_one_epoch
from validate import validate


def run_ablation(models, transforms_dict, train_data, val_data, device, 
                 train_images_base_path, dataset_class, image_size, 
                 seeds=[42, 123, 777], epochs=15):
    """
    Run systematic ablation study across all combinations
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    transform_base = transforms_dict.get('baseline', list(transforms_dict.values())[0])
    
    results = []
    
    for model_name, model_class in models.items():
        for transform_name, transform in transforms_dict.items():
            print(f"\n Running {model_name} + {transform_name}")

            for seed in seeds:
                torch.manual_seed(seed)
                print(f"  Seed {seed}...")
                
                # Create datasets
                train_dataset = dataset_class(
                    [train_images_base_path + p for p in X_train],
                    y_train, transformation=transform
                )
                val_dataset = dataset_class(
                    [train_images_base_path + p for p in X_val],
                    y_val, transformation=transform_base  # Val always fixed
                )
                
                train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
                
                # Train model
                model = model_class(image_size).to(device)
                optimizer = optim.Adam(model.parameters(), lr=0.001)
                criterion = nn.CrossEntropyLoss()
                
                best_val_acc = 0
                for epoch in range(epochs):
                    # Training loop using generic function
                    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                    
                    # Validation
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
