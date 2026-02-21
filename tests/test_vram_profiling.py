"""
VRAM Profiling Test for Swin-DR Classifier
Tests model instantiation, forward pass, backward pass, and VRAM usage.
"""

import torch
import torch.nn as nn
import argparse
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.classifier import SwinDRClassifier, create_swin_dr_classifier


def get_gpu_memory():
    """Get current GPU memory usage in GB."""
    if torch.cuda.is_available():
        return {
            'allocated_gb': torch.cuda.memory_allocated() / (1024 ** 3),
            'reserved_gb': torch.cuda.memory_reserved() / (1024 ** 3),
            'max_allocated_gb': torch.cuda.max_memory_allocated() / (1024 ** 3),
        }
    return {'allocated_gb': 0, 'reserved_gb': 0, 'max_allocated_gb': 0}


def reset_peak_memory():
    """Reset peak memory stats."""
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()


def test_instantiation():
    """Test 1: Model Instantiation"""
    print("="*60)
    print("TEST 1: Model Instantiation")
    print("="*60)
    
    reset_peak_memory()
    
    print("\nCreating Swin-DR Classifier (VRAM optimized)...")
    model = create_swin_dr_classifier(vram_optimized=True)
    
    # Move to GPU if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    if torch.cuda.is_available():
        model = model.to(device)
    
    # Count parameters
    params = model.count_parameters()
    print(f"\nModel Parameters:")
    print(f"  Total: {params['total_millions']:.2f}M")
    print(f"  Trainable: {params['trainable'] / 1e6:.2f}M")
    print(f"  Non-trainable: {params['non_trainable'] / 1e6:.2f}M")
    
    # Check VRAM
    mem = get_gpu_memory()
    print(f"\nGPU Memory After Model Load:")
    print(f"  Allocated: {mem['allocated_gb']:.3f} GB")
    print(f"  Reserved: {mem['reserved_gb']:.3f} GB")
    
    print("\n✓ Test 1 PASSED")
    return model, device


def test_forward_pass(model, device):
    """Test 2: Forward Pass"""
    print("\n" + "="*60)
    print("TEST 2: Forward Pass")
    print("="*60)
    
    reset_peak_memory()
    
    # Create dummy input
    batch_size = 1
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    print(f"\nInput shape: {x.shape}")
    print(f"Input size: {x.numel() * 4 / (1024**2):.2f} MB")
    
    # Forward pass
    print("\nRunning forward pass...")
    with torch.no_grad():
        logits = model(x)
    
    print(f"Output shape: {logits.shape}")
    print(f"Output logits: {logits[0].cpu().numpy()}")
    
    # Check VRAM
    mem = get_gpu_memory()
    print(f"\nGPU Memory After Forward Pass:")
    print(f"  Allocated: {mem['allocated_gb']:.3f} GB")
    print(f"  Peak: {mem['max_allocated_gb']:.3f} GB")
    
    # Verify output shape
    assert logits.shape == (batch_size, 5), f"Expected shape ({batch_size}, 5), got {logits.shape}"
    
    print("\n✓ Test 2 PASSED")
    return logits


def test_backward_pass(model, device):
    """Test 3: Backward Pass (Gradient Computation)"""
    print("\n" + "="*60)
    print("TEST 3: Backward Pass")
    print("="*60)
    
    reset_peak_memory()
    
    # Create dummy input and target
    batch_size = 1
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    target = torch.randint(0, 5, (batch_size,)).to(device)
    
    print(f"\nInput shape: {x.shape}")
    print(f"Target: {target.cpu().numpy()}")
    
    # Forward pass
    print("\nRunning forward + backward pass...")
    logits = model(x)
    
    # Compute loss
    criterion = nn.CrossEntropyLoss()
    loss = criterion(logits, target)
    print(f"Loss: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_gradients = any(p.grad is not None for p in model.parameters())
    print(f"Gradients computed: {has_gradients}")
    
    # Check VRAM
    mem = get_gpu_memory()
    print(f"\nGPU Memory After Backward Pass:")
    print(f"  Allocated: {mem['allocated_gb']:.3f} GB")
    print(f"  Peak: {mem['max_allocated_gb']:.3f} GB")
    
    # Verify gradients exist
    assert has_gradients, "No gradients computed!"
    
    print("\n✓ Test 3 PASSED")
    return mem['max_allocated_gb']


def test_mixed_precision(model, device):
    """Test 4: Mixed Precision (AMP)"""
    print("\n" + "="*60)
    print("TEST 4: Mixed Precision (AMP)")
    print("="*60)
    
    if not torch.cuda.is_available():
        print("\nSkipping AMP test (no CUDA available)")
        return
    
    reset_peak_memory()
    
    # Create dummy input and target
    batch_size = 1
    x = torch.randn(batch_size, 3, 224, 224).to(device)
    target = torch.randint(0, 5, (batch_size,)).to(device)
    
    # Use automatic mixed precision
    print("\nRunning with AMP enabled...")
    scaler = torch.cuda.amp.GradScaler()
    
    with torch.cuda.amp.autocast():
        logits = model(x)
        criterion = nn.CrossEntropyLoss()
        loss = criterion(logits, target)
    
    print(f"Loss: {loss.item():.4f}")
    
    # Backward with scaler
    scaler.scale(loss).backward()
    
    # Check VRAM
    mem = get_gpu_memory()
    print(f"\nGPU Memory with AMP:")
    print(f"  Allocated: {mem['allocated_gb']:.3f} GB")
    print(f"  Peak: {mem['max_allocated_gb']:.3f} GB")
    
    print("\n✓ Test 4 PASSED")
    return mem['max_allocated_gb']


def test_vram_estimates(model):
    """Test 5: VRAM Estimation Accuracy"""
    print("\n" + "="*60)
    print("TEST 5: VRAM Estimation vs Actual")
    print("="*60)
    
    # Get estimates
    est_fp32 = model.estimate_vram(batch_size=1, precision='fp32')
    est_fp16 = model.estimate_vram(batch_size=1, precision='fp16')
    
    print("\nEstimated VRAM (FP32, batch=1):")
    print(f"  Model: {est_fp32['model_gb']:.3f} GB")
    print(f"  Gradients: {est_fp32['gradients_gb']:.3f} GB")
    print(f"  Optimizer: {est_fp32['optimizer_gb']:.3f} GB")
    print(f"  Activations: {est_fp32['activations_gb']:.3f} GB")
    print(f"  Total: {est_fp32['total_gb']:.3f} GB")
    
    print("\nEstimated VRAM (FP16, batch=1):")
    print(f"  Model: {est_fp16['model_gb']:.3f} GB")
    print(f"  Gradients: {est_fp16['gradients_gb']:.3f} GB")
    print(f"  Optimizer: {est_fp16['optimizer_gb']:.3f} GB")
    print(f"  Activations: {est_fp16['activations_gb']:.3f} GB")
    print(f"  Total: {est_fp16['total_gb']:.3f} GB")
    
    print("\n✓ Test 5 PASSED")
    return est_fp32, est_fp16


def main():
    parser = argparse.ArgumentParser(description='VRAM Profiling for Swin-DR Classifier')
    parser.add_argument('--test', type=str, default='all',
                        choices=['all', 'instantiation', 'forward', 'backward', 'amp', 'estimates'],
                        help='Which test to run')
    args = parser.parse_args()
    
    print("="*60)
    print("SWIN-DR CLASSIFIER VRAM PROFILING")
    print("="*60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Total VRAM: {torch.cuda.get_device_properties(0).total_memory / (1024**3):.2f} GB")
    print("="*60)
    
    try:
        # Run tests
        if args.test in ['all', 'instantiation']:
            model, device = test_instantiation()
        else:
            # Create model for other tests
            model = create_swin_dr_classifier(vram_optimized=True)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            if torch.cuda.is_available():
                model = model.to(device)
        
        if args.test in ['all', 'forward']:
            test_forward_pass(model, device)
        
        if args.test in ['all', 'backward']:
            peak_vram_fp32 = test_backward_pass(model, device)
        
        if args.test in ['all', 'amp']:
            peak_vram_fp16 = test_mixed_precision(model, device)
        
        if args.test in ['all', 'estimates']:
            est_fp32, est_fp16 = test_vram_estimates(model)
        
        # Final summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        
        if args.test == 'all':
            print(f"\n✓ All tests PASSED!")
            print(f"\nPeak VRAM Usage:")
            if torch.cuda.is_available():
                print(f"  FP32: {peak_vram_fp32:.3f} GB")
                print(f"  FP16 (AMP): {peak_vram_fp16:.3f} GB")
                print(f"\nEstimated vs Actual:")
                print(f"  FP32 Estimate: {est_fp32['total_gb']:.3f} GB | Actual: {peak_vram_fp32:.3f} GB")
                print(f"  FP16 Estimate: {est_fp16['total_gb']:.3f} GB | Actual: {peak_vram_fp16:.3f} GB")
                
                # Check if fits in 4GB
                if peak_vram_fp32 < 4.0:
                    print(f"\n✓ Model fits in 4GB VRAM (FP32)!")
                else:
                    print(f"\n⚠ Model exceeds 4GB VRAM in FP32 mode")
                
                if peak_vram_fp16 < 4.0:
                    print(f"✓ Model fits in 4GB VRAM (FP16)!")
                else:
                    print(f"⚠ Model exceeds 4GB VRAM even in FP16 mode")
            else:
                print("  Running on CPU - no VRAM measurements")
        
        print("\n" + "="*60)
        
    except Exception as e:
        print(f"\n✗ Test FAILED with error:")
        print(f"  {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
