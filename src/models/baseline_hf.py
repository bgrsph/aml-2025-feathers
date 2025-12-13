import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.io import decode_image



class FeatherImageDatasetHF(Dataset):

    def __init__(self, image_paths, image_labels=None, processor=None):
        self.image_paths = image_paths
        self.image_labels = image_labels
        self.processor = processor

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, i):
        image = decode_image(self.image_paths[i])
        inputs = self.processor(images=image, return_tensors="pt")
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        # Add labels only for train/val
        if self.image_labels is not None:
            inputs["labels"] = torch.tensor(self.image_labels[i], dtype=torch.long)
        
        return inputs



def validate_hf(model, val_loader, device):
    model.to(device)   
    model.eval()
    correct, total = 0, 0

    with torch.no_grad():
        for batch in val_loader:
            # Move all inputs to device
            batch = {k: v.to(device) for k, v in batch.items()}
            labels = batch.pop("labels")  # remove labels
            
            outputs = model(**batch)
            preds = outputs.logits.argmax(dim=1)

            total += labels.size(0)
            correct += (preds == labels).sum().item()

    return correct / total

def create_hf_dataloaders(train_image_paths, X_train,  y_train, X_val, y_val, processor, batch_size, num_workers=0, pin_memory=False):

    # Prepare training dataset and loader
    train_dataset_hf = FeatherImageDatasetHF(
        [train_image_paths + path for path in X_train],
        y_train,
        processor
    )
    train_loader_hf = DataLoader(
        train_dataset_hf,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    # Prepare validation dataset and loader
    val_dataset_hf = FeatherImageDatasetHF(
        [train_image_paths + path for path in X_val],
        y_val,
        processor
    )
    val_loader_hf = DataLoader(
        val_dataset_hf,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )

    return train_loader_hf, val_loader_hf