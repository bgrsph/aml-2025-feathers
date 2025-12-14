"""
Test script to verify T-Rex model works correctly
Run this to check if the model is properly implemented before training
"""

import torch
import numpy as np
from models.t_rex import TRex, compute_trex_loss

def test_trex_model():
    """Test T-Rex model initialization and forward pass"""
    print("="*70)
    print("Testing T-Rex Model")
    print("="*70)
    
    # Test parameters
    batch_size = 4
    image_size = 224
    num_classes = 200
    num_attrs = 312
    
    # Initialize model
    print("\n1. Initializing model...")
    model = TRex(
        image_size=image_size,
        num_classes=num_classes,
        num_attrs=num_attrs,
        dropout_rate=0.5,
        attr_weight=0.3
    )
    print("   ✓ Model initialized successfully")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"   - Total parameters: {total_params:,}")
    print(f"   - Trainable parameters: {trainable_params:,}")
    
    # Test forward pass (without attributes)
    print("\n2. Testing forward pass (classification only)...")
    dummy_images = torch.randn(batch_size, 3, image_size, image_size)
    
    model.eval()
    with torch.no_grad():
        class_logits = model(dummy_images, return_attrs=False)
    
    assert class_logits.shape == (batch_size, num_classes), \
        f"Expected shape ({batch_size}, {num_classes}), got {class_logits.shape}"
    print(f"   ✓ Output shape: {class_logits.shape}")
    print(f"   ✓ Forward pass successful")
    
    # Test forward pass (with attributes)
    print("\n3. Testing forward pass (with attribute prediction)...")
    with torch.no_grad():
        class_logits, attr_logits = model(dummy_images, return_attrs=True)
    
    assert class_logits.shape == (batch_size, num_classes), \
        f"Expected class shape ({batch_size}, {num_classes}), got {class_logits.shape}"
    assert attr_logits.shape == (batch_size, num_attrs), \
        f"Expected attr shape ({batch_size}, {num_attrs}), got {attr_logits.shape}"
    print(f"   ✓ Class logits shape: {class_logits.shape}")
    print(f"   ✓ Attribute logits shape: {attr_logits.shape}")
    print(f"   ✓ Forward pass with attributes successful")
    
    # Test loss computation
    print("\n4. Testing loss computation...")
    dummy_labels = torch.randint(0, num_classes, (batch_size,))
    dummy_attributes = torch.randint(0, 2, (num_classes, num_attrs)).float()
    
    model.train()
    outputs = model(dummy_images, return_attrs=True)
    total_loss, class_loss, attr_loss = compute_trex_loss(
        outputs, dummy_labels, dummy_attributes, attr_weight=0.3
    )
    
    print(f"   ✓ Total loss: {total_loss.item():.4f}")
    print(f"   ✓ Classification loss: {class_loss.item():.4f}")
    print(f"   ✓ Attribute loss: {attr_loss.item():.4f}")
    print(f"   ✓ Loss computation successful")
    
    # Test backward pass
    print("\n5. Testing backward pass...")
    total_loss.backward()
    
    # Check if gradients exist
    has_gradients = any(p.grad is not None for p in model.parameters() if p.requires_grad)
    assert has_gradients, "No gradients computed!"
    print("   ✓ Gradients computed successfully")
    
    # Test attribute prediction range
    print("\n6. Testing attribute predictions...")
    model.eval()
    with torch.no_grad():
        _, attr_logits = model(dummy_images, return_attrs=True)
        attr_probs = torch.sigmoid(attr_logits)
    
    assert (attr_probs >= 0).all() and (attr_probs <= 1).all(), \
        "Attribute probabilities not in [0, 1] range!"
    print(f"   ✓ Attribute probabilities in valid range [0, 1]")
    print(f"   ✓ Mean attribute probability: {attr_probs.mean().item():.3f}")
    print(f"   ✓ Min: {attr_probs.min().item():.3f}, Max: {attr_probs.max().item():.3f}")
    
    # Test device movement
    print("\n7. Testing device compatibility...")
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"   - Available device: {device}")
    
    model = model.to(device)
    dummy_images = dummy_images.to(device)
    
    with torch.no_grad():
        outputs = model(dummy_images, return_attrs=True)
    print(f"   ✓ Model works on {device}")
    
    # Summary
    print("\n" + "="*70)
    print("✅ All tests passed! T-Rex model is ready to train.")
    print("="*70)
    
    return True

def test_model_sizes():
    """Compare model sizes"""
    print("\n" + "="*70)
    print("Model Size Comparison")
    print("="*70)
    
    models = {
        'T-Rex': TRex(image_size=224, num_classes=200, num_attrs=312),
    }
    
    for name, model in models.items():
        total = sum(p.numel() for p in model.parameters())
        size_mb = total * 4 / (1024**2)  # Assuming fp32
        print(f"\n{name}:")
        print(f"  Parameters: {total:,}")
        print(f"  Size: ~{size_mb:.2f} MB (fp32)")

if __name__ == "__main__":
    try:
        # Run tests
        test_trex_model()
        test_model_sizes()
        
        print("\n" + "="*70)
        print("🦖 T-Rex is ready to hunt! (train)")
        print("="*70)
        
    except Exception as e:
        print(f"\n❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        exit(1)
