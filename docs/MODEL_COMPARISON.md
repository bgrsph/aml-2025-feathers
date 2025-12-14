# Model Architecture Comparison

## 1. Standard CNN (BirdyCNN, SimpleCNN)

```
┌─────────────┐
│   Image     │
│ [3,224,224] │
└──────┬──────┘
       │
       v
┌─────────────┐
│ Conv Layers │
│  (3-4 conv) │
└──────┬──────┘
       │
       v
┌─────────────┐
│   Flatten   │
└──────┬──────┘
       │
       v
┌─────────────┐
│ FC Layers   │
└──────┬──────┘
       │
       v
┌─────────────┐
│200 Classes  │
└─────────────┘
```

**Pros**: Simple, standard architecture  
**Cons**: No semantic understanding, limited accuracy

---

## 2. AttrCNN (BROKEN - Don't Use!)

```
┌─────────────┐      ┌─────────────┐
│   Image     │      │   Label     │  ❌ NEEDS GROUND-TRUTH!
│ [3,224,224] │      │    (GT)     │
└──────┬──────┘      └──────┬──────┘
       │                    │
       v                    v
┌─────────────┐      ┌─────────────┐
│ Conv Layers │      │ Lookup GT   │  ❌ CHEATING!
└──────┬──────┘      │ Attributes  │
       │             └──────┬──────┘
       │                    │
       │             ┌──────v──────┐
       │             │ Class Attrs │
       │             │   [312]     │
       │             └──────┬──────┘
       │                    │
       v                    v
┌──────────────────────────────┐
│       Concatenate            │
│   [img_feat + gt_attrs]      │  ❌ Uses correct attrs!
└──────────────┬───────────────┘
               │
               v
        ┌─────────────┐
        │  Classifier │
        └──────┬──────┘
               │
               v
        ┌─────────────┐
        │ 200 Classes │
        └─────────────┘
```

**Pros**: High accuracy (when cheating)  
**Cons**: ❌ Unusable for Kaggle! Needs ground-truth labels at inference

---

## 3. T-Rex (Our Solution) ✅

```
┌─────────────┐
│   Image     │  ✅ Only need image!
│ [3,224,224] │
└──────┬──────┘
       │
       v
┌─────────────────────────────┐
│   Shared CNN Backbone       │
│   (4 conv + BatchNorm)      │
└──────┬──────────────────────┘
       │
       │    Image Features [B, 512*14*14]
       │
       ├─────────────────┬────────────────┐
       │                 │                │
       v                 v                │
┌──────────────┐  ┌─────────────┐        │
│  Attribute   │  │   Image     │        │
│  Prediction  │  │  Features   │        │
│   Branch     │  │  (Identity) │        │
│              │  └─────────────┘        │
│ FC → FC      │                         │
│ → Sigmoid    │                         │
└──────┬───────┘                         │
       │                                 │
       v                                 │
┌──────────────┐                         │
│ Predicted    │  ✅ No GT needed!       │
│ Attributes   │                         │
│    [312]     │                         │
└──────┬───────┘                         │
       │                                 │
       v                                 │
┌──────────────┐                         │
│  Attribute   │                         │
│  Embedding   │                         │
│  (312 → 256) │                         │
└──────┬───────┘                         │
       │                                 │
       v                                 v
┌──────────────────────────────────────────┐
│           Concatenate                    │
│     [img_feat + pred_attr_embed]         │
└──────────────────┬───────────────────────┘
                   │
                   v
            ┌──────────────┐
            │ Classification│
            │  Head (3 FC) │
            └──────┬────────┘
                   │
                   v
            ┌──────────────┐
            │  200 Classes │
            └──────────────┘

TRAINING ONLY:
┌─────────────┐     ┌──────────────┐
│   Labels    │────>│ Ground-Truth │  Used only for
│    (GT)     │     │  Attributes  │  training loss
└─────────────┘     └──────────────┘
```

**Pros**: ✅ Works for Kaggle! ✅ Learns semantic features ✅ Interpretable  
**Cons**: More complex, needs more training time

---

## Key Differences

| Feature | Standard CNN | AttrCNN | T-Rex |
|---------|-------------|---------|-------|
| **Input at Inference** | Image only | Image + Label ❌ | Image only ✅ |
| **Uses Attributes** | No | Yes (GT) | Yes (predicted) ✅ |
| **Kaggle Ready** | Yes | No ❌ | Yes ✅ |
| **Interpretable** | No | Limited | Yes ✅ |
| **Accuracy** | ~30% | ~80% (cheating) | ~40-50% ✅ |
| **Multi-Task Learning** | No | No | Yes ✅ |

---

## Training vs Inference

### Standard CNN
```
Training:   Image → Model → Class
Inference:  Image → Model → Class
```
✅ Same at training and inference

### AttrCNN (BROKEN)
```
Training:   Image + GT_Label → Model → Class
Inference:  Image + ??? → Model → Class  ❌ What label to use?
```
❌ Can't work at inference!

### T-Rex (CORRECT)
```
Training:   Image → Model → [Pred_Attrs, Class]
            (Compare Pred_Attrs with GT_Attrs[label])
            (Multi-task loss)

Inference:  Image → Model → [Pred_Attrs, Class]
            (Only use Class, ignore Pred_Attrs)
```
✅ Works at both training and inference!

---

## Multi-Task Learning in T-Rex

```
                    ┌──────────────┐
                    │     Loss     │
                    └───────┬──────┘
                            │
          ┌─────────────────┴─────────────────┐
          │                                   │
          v                                   v
  ┌───────────────┐                   ┌──────────────┐
  │ Class Loss    │                   │  Attr Loss   │
  │ (70% weight)  │                   │ (30% weight) │
  └───────┬───────┘                   └──────┬───────┘
          │                                   │
          v                                   v
  CrossEntropy(pred_class, gt_label)   BCE(pred_attrs, gt_attrs)
```

**Key Insight**: The attribute loss provides auxiliary supervision that helps the model learn better visual features, which improves classification!

---

## Why T-Rex Works Better

1. **Richer Features**: Learns both pixel-level and semantic features
2. **Regularization**: Multi-task learning prevents overfitting
3. **Interpretability**: Can debug by checking attribute predictions
4. **Transfer Learning**: Attributes are reusable across species
5. **Expert Knowledge**: Leverages human-annotated semantic features

---

## Summary

- **Standard CNN**: Basic, works okay
- **AttrCNN**: High accuracy but BROKEN (unusable for submission)
- **T-Rex**: Best of both worlds - uses attributes but works for submission! 🦖

Choose T-Rex! 🚀
