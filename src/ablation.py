import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from train import train_one_epoch
from validate import validate
from validate import k_fold_cross_validate


def run_ablation(models, transforms_dict, dataset_class, X_train, y_train, X_val, y_val, 
                 TRAIN_IMAGES_BASE_PATH, IMAGE_SIZE, device, transform_base,
                 num_classes, seeds=[42, 123, 777], epochs=15, batch_size=32, lr=0.001):

    results = []
    
    for model_name, model_class in models.items():
        for transform_name, transform in transforms_dict.items():
            print(f"\n Running {model_name} + {transform_name}")

            for seed in seeds:
                torch.manual_seed(seed)
                print(f"  Seed {seed}...")
                
                # Create datasets
                train_dataset = dataset_class(
                    [TRAIN_IMAGES_BASE_PATH + p for p in X_train],
                    y_train, transformation=transform
                )
                val_dataset = dataset_class(
                    [TRAIN_IMAGES_BASE_PATH + p for p in X_val],
                    y_val, transformation=transform_base  # Val always fixed
                )
                
                train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
                val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
                
                # Train model
                model = model_class(image_size=IMAGE_SIZE, num_classes=num_classes).to(device)
                optimizer = optim.Adam(model.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()
                
                best_val_acc = 0
                for epoch in range(epochs):
                    # Training and validation
                    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
                    val_acc = validate(model, val_loader, device)
                    best_val_acc = max(best_val_acc, val_acc)
                    print(f"    Epoch {epoch+1}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.3f}")
                
                results.append({
                    'model': model_name,
                    'augmentation': transform_name,
                    'seed': seed,
                    'best_val_acc': best_val_acc,
                    'params': sum(p.numel() for p in model.parameters())
                })

    return pd.DataFrame(results)


def run_ablation_kfold(models, transforms_dict, dataset_class, 
                       X_data, y_data,
                       TRAIN_IMAGES_BASE_PATH, IMAGE_SIZE, device, 
                       transform_base, num_classes, 
                       k=5, seeds=[42, 123, 777], 
                       epochs=15, batch_size=32, lr=0.001):
  
    
    results = []
    
    for model_name, model_class in models.items():
        for transform_name, transform in transforms_dict.items():
            print(f"\n{'='*70}")
            print(f"Running {model_name} + {transform_name} with {k}-Fold CV")
            print(f"{'='*70}")
            
            for seed_idx, seed in enumerate(seeds):
                torch.manual_seed(seed)
                print(f"\n[Seed {seed_idx+1}/{len(seeds)}: {seed}]")
                
                # Run k-fold CV
                kfold_results = k_fold_cross_validate(
                    model_class=model_class,
                    X_data=X_data,
                    y_data=y_data,
                    dataset_class=dataset_class,
                    transform=transform,
                    transform_base=transform_base,
                    IMAGES_BASE_PATH=TRAIN_IMAGES_BASE_PATH,
                    IMAGE_SIZE=IMAGE_SIZE,
                    num_classes=num_classes,
                    device=device,
                    k=k,
                    num_epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                    seed=seed,
                    verbose=True
                )
                
                # Store aggregated results
                results.append({
                    'model': model_name,
                    'augmentation': transform_name,
                    'seed': seed,
                    'mean_val_acc': kfold_results['mean_accuracy'],
                    'std_val_acc': kfold_results['std_accuracy'],
                    'min_val_acc': min(kfold_results['all_fold_accuracies']),
                    'max_val_acc': max(kfold_results['all_fold_accuracies']),
                    'fold_accuracies': kfold_results['all_fold_accuracies'],
                    'k_folds': k,
                    'params': sum(p.numel() for p in 
                                model_class(IMAGE_SIZE, num_classes).parameters())
                })
    
    return pd.DataFrame(results)
