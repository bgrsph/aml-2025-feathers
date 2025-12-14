# T-Rex Quick Start Guide 🦖

## TL;DR

T-Rex is a CNN that **predicts bird attributes** (like "has red wings", "curved beak") from images and uses them to improve classification. Unlike the broken `AttrCNN`, T-Rex can be used for Kaggle submissions!

## Quick Usage

### 1. Import
```python
from models.t_rex import TRex, train_trex, validate_trex
import numpy as np

# Load attributes
attributes = np.load('data/raw/attributes.npy')  # [200, 312]
```

### 2. Initialize
```python
model = TRex(
    image_size=224,
    num_classes=200,
    num_attrs=312,
    dropout_rate=0.5,
    attr_weight=0.3  # 30% attribute loss, 70% classification
).to(device)
```

### 3. Train
```python
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)

trained_model = train_trex(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    attributes=attributes,
    optimizer=optimizer,
    device=device,
    num_epochs=15,
    attr_weight=0.3
)
```

### 4. Evaluate
```python
accuracy = validate_trex(trained_model, val_loader, device)
print(f"Validation Accuracy: {accuracy:.2f}%")
```

### 5. Inference (Kaggle)
```python
model.eval()
predictions = []

with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        logits = model(images, return_attrs=False)
        preds = logits.argmax(dim=1)
        predictions.extend(preds.cpu().numpy())
```

## Key Features

✅ **No ground-truth labels needed** at inference (unlike AttrCNN)  
✅ **Predicts 312 attributes** as intermediate features  
✅ **Multi-task learning** improves both tasks  
✅ **Interpretable**: see which attributes are detected  
✅ **Ready for Kaggle submission**

## Architecture

```
Image → CNN → [Attribute Branch + Image Features] → Fuse → Classify
              ↓
         312 attributes
```

## Common Parameters

- `image_size`: 224 (default)
- `num_classes`: 200 birds
- `num_attrs`: 312 attributes
- `dropout_rate`: 0.5 (adjust if over/underfitting)
- `attr_weight`: 0.3 (0-1, higher = more focus on attributes)

## Hyperparameter Tips

- **Overfitting?** Increase dropout (0.6-0.7), add more augmentation
- **Slow learning?** Increase lr to 1e-3, but watch for instability  
- **Poor attributes?** Increase attr_weight to 0.5
- **Poor classification?** Decrease attr_weight to 0.1-0.2

## Expected Performance

- **Validation Accuracy**: 35-50% (200 classes is hard!)
- **Attribute Accuracy**: 65-80% (312 binary predictions)
- **Training Time**: ~2-3 hours for 15 epochs on GPU

## Debugging

### Check attribute predictions:
```python
model.eval()
with torch.no_grad():
    imgs, labels = next(iter(val_loader))
    _, attr_logits = model(imgs.to(device), return_attrs=True)
    attr_probs = torch.sigmoid(attr_logits)
    
    # Which attributes are predicted?
    sample_attrs = (attr_probs[0] > 0.5).cpu().numpy()
    print(f"Detected {sample_attrs.sum()} / 312 attributes")
```

### Check training progress:
```python
# Training prints:
# - Total loss (combination of class + attr)
# - Class loss (classification only)
# - Attr loss (attribute prediction only)
# Monitor all three!
```

## Comparison

| Model | Architecture | Uses Attributes | Kaggle Ready? | Expected Acc |
|-------|-------------|-----------------|---------------|--------------|
| SimpleCNN | 3-4 conv layers | ❌ | ✅ | ~20-30% |
| BirdyCNN | 3 conv + dropout | ❌ | ✅ | ~25-35% |
| AttrCNN | 3 conv + attrs | ✅ (cheating) | ❌ | ~80% (cheating) |
| **T-Rex** | 4 conv + predict attrs | ✅ | ✅ | **~35-50%** |

## Full Documentation

See `TREX_MODEL_EXPLANATION.md` for detailed explanation of:
- How attributes work
- Why AttrCNN is broken
- Complete architecture details
- Hyperparameter tuning
- Troubleshooting

## Notebook

Run cells in `src/experiment.ipynb` under the "Final Attempt: T-Rex" section.

---

**Happy bird classifying! 🦅**
