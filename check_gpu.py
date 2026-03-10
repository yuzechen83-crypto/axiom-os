import torch

print("=" * 50)
print("GPU Check")
print("=" * 50)

if torch.cuda.is_available():
    print(f"✓ CUDA Available")
    print(f"  Version: {torch.version.cuda}")
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    
    # Test
    x = torch.randn(100, 100).cuda()
    y = x @ x.T
    print(f"  ✓ GPU computation test passed")
else:
    print("✗ CUDA not available")
    print("  Using CPU")

print("=" * 50)
