# T-Rex Model: Attribution-Based Bird Classification 🦖

## Table of Contents
1. [What are Attributions?](#what-are-attributions)
2. [The Problem with Previous Approaches](#the-problem-with-previous-approaches)
3. [T-Rex Solution](#t-rex-solution)
4. [Architecture Details](#architecture-details)
5. [How to Use T-Rex](#how-to-use-t-rex)
6. [Expected Improvements](#expected-improvements)

---

## What are Attributions?

The CUB-200 dataset (and this Kaggle challenge) provides **312 binary attributes** for each of the 200 bird classes. These attributes describe visual features that humans use to identify birds:

### Attribute Examples:
- **Bill shape**: `has::bill_shape::curved (up or down)`, `has::bill_shape::hooked`, `has::bill_shape::needle`
- **Wing color**: `has::wing_color::red`, `has::wing_color::blue`, `has::wing_color::brown`
- **Body patterns**: `has::breast_pattern::solid`, `has::breast_pattern::spotted`, `has::breast_pattern::striped`
- **Size features**: `has::size::large`, `has::size::small`, `has::size::medium`
- **Behavior**: `has::bill_shape::dagger`, `has::bill_shape::cone`

### Why Attributes Matter:
- **Semantic information**: Attributes provide high-level semantic features beyond raw pixels
- **Discriminative**: Different bird species have different attribute combinations
- **Interpretable**: We can understand *why* the model makes a prediction
- **Transfer learning**: Attributes learned on one dataset can help with others

### Attribute Structure:
```
attributes.npy: shape [200, 312]
- 200 rows: one for each bird class
- 312 columns: one for each attribute
- Values: binary (0 or 1)
```

Example for class 0 (Black-footed Albatross):
```python
attributes[0] = [1, 0, 0, 1, 1, 0, ..., 1]  # 312 binary values
# Might mean: has curved bill, no hooked bill, no needle bill, has white wing, ...
```

---

## The Problem with Previous Approaches

### Baseline CNN Models (SimpleCNN, BirdyCNN, ChickenCNN)
**Problem**: Only learn from raw pixel data
- No explicit semantic understanding
- Can't leverage the rich attribute annotations
- Limited interpretability

### AttrCNN (The Cheating Model)
**Problem**: Requires ground-truth labels at inference time!

```python
# AttrCNN forward pass
def forward(self, x, labels):  # ❌ Needs labels!
    img_feat = self.cnn(x)
    class_attr = self.attributes[labels]  # ❌ Looks up correct class's attributes!
    fused = concat(img_feat, class_attr)
    return self.classifier(fused)
```

**Why this is cheating:**
- At inference, we DON'T know the true label (that's what we're trying to predict!)
- The model gets the correct attributes even when making predictions
- High accuracy in validation, but **cannot be used for Kaggle submission**

**The fatal flaw**: You can't ask "What is this bird?" while simultaneously telling it "It's a sparrow, here are sparrow attributes"!

---

## T-Rex Solution

**T-Rex** (Two-stage REgressor & eXtractor) solves this with a **two-branch architecture**:

### Key Idea: Predict Attributes from Images!
Instead of using ground-truth attributes at inference, **learn to predict them**:

1. **Attribute Prediction Branch**: Predicts all 312 attributes from the image
2. **Classification Branch**: Uses predicted attributes + image features to classify

### Why This Works:
✅ No ground-truth labels needed at inference  
✅ Learns semantic features (attributes) as intermediate representations  
✅ Multi-task learning improves both attribute prediction and classification  
✅ Can be used for Kaggle submission  
✅ Interpretable: we can see which attributes the model thinks are present  

---

## Architecture Details

### High-Level Architecture

```
Input Image [B, 3, 224, 224]
        |
        v
+----------------------------------+
|    Shared CNN Backbone           |
|  (Conv1→Conv2→Conv3→Conv4)       |
|   with Batch Normalization       |
+----------------------------------+
        |
  Image Features [B, 512*14*14]
        |
  +-----+-----+
  |           |
  v           v
+-----+   +-------+
|     |   |       |
| A   |   | Image |
| t   |   | Feat  |
| t   |   | Pass  |
| r   |   | Thru  |
|     |   |       |
| P   |   |       |
| r   |   |       |
| e   |   |       |
| d   |   |       |
| .   |   |       |
|     |   |       |
+-----+   +-------+
  |           |
  |           |
Pred Attr [B,312]
  |           |
  v           |
+-----+       |
|Attr |       |
|Embed|       |
+-----+       |
  |           |
  +-----+-----+
        |
        v
+------------------+
|  Concatenate     |
| [img + attr_emb] |
+------------------+
        |
        v
+------------------+
| Classification   |
| Head (3 FC layers)|
+------------------+
        |
        v
  Class Logits [B, 200]
```

### Detailed Components

#### 1. Shared CNN Backbone
```python
# Deeper than previous models
Conv1: 3 → 64 channels, BatchNorm, ReLU, Pool
Conv2: 64 → 128 channels, BatchNorm, ReLU, Pool
Conv3: 128 → 256 channels, BatchNorm, ReLU, Pool
Conv4: 256 → 512 channels, BatchNorm, ReLU, Pool
# Output: 512 × 14 × 14 feature maps
```

**Improvements over BirdyCNN:**
- 4 conv layers instead of 3 (deeper)
- Batch normalization for training stability
- More filters (512 vs 128)
- Dropout in conv layers to prevent overfitting

#### 2. Attribute Prediction Branch
```python
# Predicts 312 binary attributes
FC1: feature_size → 1024, BatchNorm, ReLU, Dropout
FC2: 1024 → 312, Sigmoid
# Output: [B, 312] binary predictions
```

**Purpose:**
- Forces the network to learn meaningful semantic features
- Provides supervision signal at an intermediate level
- Creates interpretable representations

#### 3. Attribute Embedding
```python
# Projects high-dimensional attributes to compact representation
Embed: 312 → 256, BatchNorm, ReLU
# Output: [B, 256] compact attribute representation
```

**Purpose:**
- Reduces dimensionality of attributes
- Learns optimal attribute representation for classification
- Acts as bottleneck for regularization

#### 4. Classification Branch
```python
# Fuses image features + attribute embeddings
FC1: (feature_size + 256) → 1024, BatchNorm, ReLU, Dropout
FC2: 1024 → 512, BatchNorm, ReLU, Dropout
FC3: 512 → 200
# Output: [B, 200] class logits
```

**Purpose:**
- Combines visual and semantic information
- Deep classifier for complex decision boundaries
- Heavy regularization (dropout, batch norm)

### Multi-Task Loss Function

```python
def compute_trex_loss(class_logits, attr_logits, labels, attributes, attr_weight=0.3):
    # Classification loss
    class_loss = CrossEntropy(class_logits, labels)
    
    # Attribute prediction loss
    gt_attrs = attributes[labels]  # [B, 312] - only for training!
    attr_loss = BinaryCrossEntropy(attr_logits, gt_attrs)
    
    # Combined loss
    total_loss = (1 - attr_weight) * class_loss + attr_weight * attr_loss
    return total_loss
```

**Key Points:**
- Ground-truth attributes (`attributes[labels]`) are **only used during training**
- At inference, we only need the image - no labels required!
- `attr_weight` controls trade-off between tasks (default: 0.3)

---

## How to Use T-Rex

### Training

```python
from models.t_rex import TRex, train_trex, validate_trex
import numpy as np

# Load attributes
attributes = np.load('data/raw/attributes.npy')  # [200, 312]

# Initialize model
model = TRex(
    image_size=224,
    num_classes=200,
    num_attrs=312,
    dropout_rate=0.5,
    attr_weight=0.3  # 30% attribute loss, 70% classification loss
)

# Train
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
trained_model = train_trex(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    attributes=attributes,  # Only for training!
    optimizer=optimizer,
    device=device,
    num_epochs=15,
    attr_weight=0.3
)
```

### Inference (Kaggle Submission)

```python
# Inference - NO GROUND-TRUTH LABELS NEEDED!
model.eval()
with torch.no_grad():
    for images, _ in test_loader:
        images = images.to(device)
        
        # Get predictions
        class_logits = model(images, return_attrs=False)
        predictions = class_logits.argmax(dim=1)
        
        # Can also get attribute predictions for interpretability
        class_logits, attr_logits = model(images, return_attrs=True)
        attr_probs = torch.sigmoid(attr_logits)
        predicted_attrs = (attr_probs > 0.5).float()
```

### Analyzing Attribute Predictions

```python
# See which attributes the model thinks are present
model.eval()
with torch.no_grad():
    img = load_image("test_bird.jpg")
    class_logits, attr_logits = model(img.unsqueeze(0), return_attrs=True)
    
    # Get predicted class
    pred_class = class_logits.argmax(dim=1).item()
    print(f"Predicted class: {class_names[pred_class]}")
    
    # Get predicted attributes
    attr_probs = torch.sigmoid(attr_logits[0])
    present_attrs = [i for i, p in enumerate(attr_probs) if p > 0.5]
    
    print("Detected attributes:")
    for idx in present_attrs:
        print(f"  - {attribute_names[idx]} (confidence: {attr_probs[idx]:.2f})")
```

---

## Expected Improvements

### Why T-Rex Should Outperform BirdyCNN

1. **Richer Feature Learning**
   - Learns both low-level (visual) and high-level (semantic) features
   - 312 attributes provide rich intermediate supervision
   - Attributes capture expert knowledge about bird identification

2. **Better Generalization**
   - Multi-task learning acts as regularization
   - Forced to learn interpretable features
   - Less likely to overfit to spurious correlations

3. **Transfer Learning Potential**
   - Attributes are shared across bird species
   - Learning "has red wings" helps identify multiple species
   - Compositional reasoning: combine attributes flexibly

4. **Deeper Architecture**
   - 4 conv layers vs 3 in BirdyCNN
   - Batch normalization for training stability
   - More capacity (512 vs 128 filters)

### Quantitative Expectations

| Model | Expected Val Accuracy | Parameters |
|-------|----------------------|------------|
| SimpleCNN-v1 | ~15-25% | ~50M |
| SimpleCNN-v2 | ~20-30% | ~100M |
| BirdyCNN | ~25-35% | ~50M |
| **T-Rex** | **~35-50%** | ~100M |

**Note**: Fine-grained bird classification is hard! Even 40% accuracy is quite good for 200 classes with limited training data.

### Qualitative Benefits

1. **Interpretability**: See which features the model uses
2. **Debugging**: Analyze where attribute prediction fails
3. **Active learning**: Identify classes with poor attribute predictions
4. **Human-AI collaboration**: Users can verify/correct attributes

---

## Hyperparameter Tuning Suggestions

### attr_weight (Attribute Loss Weight)
- **Default**: 0.3 (30% attribute, 70% classification)
- **Higher** (0.5-0.7): Better attribute predictions, might hurt classification
- **Lower** (0.1-0.2): Better classification, might hurt attribute learning
- **Adaptive**: Start high (0.5), decrease over epochs

### Learning Rate
- **Default**: 1e-4 with Adam
- Try: 1e-3 (faster but less stable), 1e-5 (slower but more stable)
- Use learning rate scheduling (e.g., reduce on plateau)

### Dropout Rate
- **Default**: 0.5
- Higher (0.6-0.7) if overfitting
- Lower (0.3-0.4) if underfitting

### Architecture Variations
1. **Deeper backbone**: Add Conv5 (512 → 1024)
2. **Attention**: Add attention between image and attribute features
3. **Hierarchical attributes**: Group attributes by type (color, shape, etc.)
4. **Residual connections**: Skip connections in classification head

---

## Common Issues and Solutions

### Issue 1: Low Attribute Prediction Accuracy
**Problem**: Attributes predicted at ~60% accuracy, too noisy for classification

**Solutions:**
- Increase `attr_weight` (give more importance to attribute learning)
- Use focal loss for imbalanced attributes
- Pre-train attribute branch separately
- Use stronger data augmentation

### Issue 2: Overfitting
**Problem**: High training accuracy, low validation accuracy

**Solutions:**
- Increase dropout rate
- Add more data augmentation
- Reduce model capacity (fewer filters)
- Increase weight decay
- Early stopping

### Issue 3: Training Instability
**Problem**: Loss oscillates or diverges

**Solutions:**
- Lower learning rate
- Use gradient clipping
- Check batch normalization momentum
- Use learning rate warmup

### Issue 4: Slow Convergence
**Problem**: Takes many epochs to train

**Solutions:**
- Higher learning rate (carefully!)
- Better initialization (Kaiming/Xavier)
- Use learning rate scheduling
- Pre-train backbone on ImageNet

---

## Comparison with Other Approaches

### T-Rex vs Zero-Shot Learning
- **Zero-shot**: Predicts attributes, uses them to find nearest class prototype
- **T-Rex**: Predicts attributes, uses them as auxiliary features for classification
- **Advantage**: T-Rex can learn optimal attribute weighting for classification

### T-Rex vs Prototypical Networks
- **Prototypical**: Learns class prototypes in embedding space
- **T-Rex**: Learns explicit attribute representations + classification
- **Advantage**: T-Rex is more interpretable and uses given attributes

### T-Rex vs Standard CNN
- **Standard CNN**: End-to-end image → class
- **T-Rex**: Image → attributes → class (with multi-task learning)
- **Advantage**: T-Rex has intermediate supervision and interpretability

---

## Future Improvements

1. **Attention Mechanisms**: Attend to relevant attributes for each class
2. **Attribute Relationships**: Model correlations between attributes
3. **Uncertainty Estimation**: Predict confidence for each attribute
4. **Active Learning**: Query uncertain attribute predictions
5. **Knowledge Distillation**: Distill from larger attribute predictor
6. **Meta-Learning**: Learn to predict attributes for new classes

---

## References

- **CUB-200-2011 Dataset**: Wah et al., "The Caltech-UCSD Birds-200-2011 Dataset"
- **Attribute-Based Classification**: Lampert et al., "Attribute-Based Classification for Zero-Shot Learning"
- **Multi-Task Learning**: Caruana, "Multitask Learning"

---

## Summary

**T-Rex** is a sophisticated bird classification model that:
1. ✅ Predicts attributes from images (no ground-truth needed at inference)
2. ✅ Uses predicted attributes to improve classification
3. ✅ Provides interpretable intermediate representations
4. ✅ Can be used for Kaggle submission
5. ✅ Leverages multi-task learning for better generalization

The name "T-Rex" is fitting because:
- **T** = Two-stage (attribute prediction + classification)
- **Rex** = Regressor & Extractor
- 🦖 = King of birds (technically the ancestor of all modern birds!)

**Go forth and classify those birds! 🦅🦜🦆**
