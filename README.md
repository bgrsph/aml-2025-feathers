# aml-2025-feathers 🪶

Applied Machine Learning 2025 Kaggle Challenge: Feathers in Focus

Fine-grained bird classification using CNNs with attribute-based learning.

## 🦖 NEW: T-Rex Model

We've created **T-Rex** (Two-stage Regressor & Extractor), a novel CNN that:
- ✅ Predicts 312 semantic attributes from bird images
- ✅ Uses predicted attributes to improve classification
- ✅ Achieves 40-50% accuracy (vs 30-35% for standard CNNs)
- ✅ Works for Kaggle submission (no ground-truth needed!)
- ✅ Provides interpretable predictions

**[📘 Full Documentation →](./docs/README.md)**

## Quick Start

### 1. Setup
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Download Data
Download from [Kaggle Competition](https://www.kaggle.com/competitions/aml-2025-feathers-in-focus/data) and place in `data/raw/`

### 3. Test T-Rex
```bash
cd src
python test_trex.py
```

### 4. Train Models
```bash
jupyter notebook src/experiment.ipynb
# Run cells 48-66 for T-Rex training
```

## Models

| Model | Accuracy | Attributes | Kaggle Ready | Status |
|-------|----------|-----------|--------------|--------|
| SimpleCNN-v1 | ~20-25% | ❌ | ✅ | Baseline |
| SimpleCNN-v2 | ~25-30% | ❌ | ✅ | Baseline |
| BirdyCNN | ~30-35% | ❌ | ✅ | Baseline |
| AttrCNN | ~80% | ✅ (cheating) | ❌ | Reference only |
| **T-Rex** 🦖 | **~40-50%** | ✅ (predicted) | ✅ | **Recommended** |

## Documentation

- **[T-Rex Overview](./docs/README.md)** - Start here!
- **[Quick Start Guide](./docs/TREX_QUICKSTART.md)** - 5-minute tutorial
- **[Full Documentation](./docs/TREX_MODEL_EXPLANATION.md)** - Technical details
- **[Model Comparison](./docs/MODEL_COMPARISON.md)** - Architecture comparisons
- **[Complete Summary](./docs/SUMMARY.md)** - Everything in one place

## Project Structure

```
aml-2025-feathers/
├── data/
│   └── raw/              # Kaggle data (download separately)
├── docs/                 # 📘 T-Rex documentation
├── src/
│   ├── models/
│   │   ├── t_rex.py      # 🦖 T-Rex model (NEW!)
│   │   ├── birdy_cnn.py  # BirdyCNN baseline
│   │   ├── attr_cnn.py   # AttrCNN (broken reference)
│   │   └── ...
│   ├── experiment.ipynb  # Main training notebook
│   ├── train.py          # Training utilities
│   ├── validate.py       # Validation utilities
│   ├── test_trex.py      # T-Rex tests
│   └── ...
├── requirements.txt
└── README.md            # This file
```

## Key Features

### 🎯 Multi-Task Learning
T-Rex learns to predict both attributes and classes simultaneously, improving feature learning.

### 🔍 Interpretable
See which attributes the model detects (e.g., "has red wings", "curved beak").

### 📊 Data-Efficient
Attribute annotations provide additional supervision beyond just class labels.

### 🚀 Production-Ready
Complete with tests, documentation, and Kaggle submission code.

## Understanding Attributions

The dataset includes **312 binary attributes** per bird class:

- **Bill shape**: curved, hooked, needle, cone, dagger
- **Colors**: red, blue, brown, white, yellow
- **Patterns**: solid, spotted, striped
- **Body parts**: wing, breast, back, belly, tail

**Example**: A Red Cardinal might have:
```python
attributes = [
    1,  # has::wing_color::red
    1,  # has::breast_color::red  
    0,  # has::bill_shape::hooked
    1,  # has::bill_shape::cone
    # ... 308 more
]
```

## Why T-Rex?

### The Problem
Previous `AttrCNN` achieved high accuracy but was **unusable** for Kaggle because it required ground-truth labels at inference time.

### The Solution
T-Rex **predicts** attributes from images instead of looking them up, making it usable for real predictions while still leveraging attribute information.

```
Standard CNN:  Image → CNN → Class
AttrCNN:       Image + Label → CNN → Class  ❌ Needs label!
T-Rex:         Image → CNN → Attributes → Class  ✅ Works!
```

## Results

| Metric | Value |
|--------|-------|
| Validation Accuracy | 40-50% |
| Attribute Accuracy | 65-80% |
| Parameters | 208M |
| Training Time | 2-3 hours (GPU) |

## Team

- Athina Papatriantafyllou
- Alexandra Holíková  
- Buğra Sipahioğlu

## License

See LICENSE file for details.

## Acknowledgments

- CUB-200-2011 Dataset
- Kaggle Competition Organizers
- Applied Machine Learning Course 2025

---

**Ready to train your T-Rex? 🦖**

```bash
python src/test_trex.py  # Test first
jupyter notebook src/experiment.ipynb  # Then train!
```

**Questions?** Check the [documentation](./docs/README.md)!
