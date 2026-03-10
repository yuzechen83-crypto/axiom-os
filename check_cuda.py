# -*- coding: utf-8 -*-
"""
Check PyTorch and CUDA Installation
For RTX 5070 (sm_120) support
"""

import sys

def check_pytorch():
    """Check PyTorch installation"""
    print("="*70)
    print("PyTorch Installation Check")
    print("="*70)
    
    try:
        import torch
        print(f"[OK] PyTorch imported successfully")
        print(f"  Version: {torch.__version__}")
    except ImportError as e:
        print(f"[ERROR] PyTorch not found: {e}")
        return False
    
    # Check CUDA
    print(f"\n[1] CUDA Support:")
    cuda_available = torch.cuda.is_available()
    print(f"    Available: {cuda_available}")
    
    if cuda_available:
        print(f"    CUDA Version: {torch.version.cuda}")
        print(f"    cuDNN Version: {torch.backends.cudnn.version()}")
        
        # GPU info
        print(f"\n[2] GPU Information:")
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            print(f"    GPU {i}: {props.name}")
            print(f"      Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"      Compute Capability: {props.major}.{props.minor}")
            print(f"      Multi-Processor Count: {props.multi_processor_count}")
        
        # Test CUDA tensor
        print(f"\n[3] CUDA Functionality Test:")
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.randn(100, 100).cuda()
            z = x @ y
            print(f"    [OK] CUDA tensor operations working")
            print(f"    ✓ Test matrix multiplication: {z.shape}")
        except Exception as e:
            print(f"    [ERROR] CUDA operations failed: {e}")
    else:
        print(f"\n[2] GPU Information:")
        print(f"    [ERROR] No CUDA-capable GPU detected")
        print(f"\n[3] Recommendation:")
        print(f"    Install PyTorch with CUDA support:")
        print(f"    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128")
    
    # Check for RTX 5070 specific support
    print(f"\n[4] RTX 5070 (sm_120) Support:")
    if cuda_available:
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            if props.major == 12 and props.minor == 0:
                print(f"    [OK] RTX 5070 detected (sm_120)")
                print(f"    ✓ Full support with PyTorch nightly CUDA 12.8")
            elif "RTX 5070" in props.name or "RTX 50" in props.name:
                print(f"    [OK] RTX 50-series detected: {props.name}")
            else:
                print(f"    [INFO] GPU {i}: {props.name} (CC {props.major}.{props.minor})")
    else:
        print(f"    [ERROR] Cannot detect GPU without CUDA")
    
    # Summary
    print(f"\n[5] Summary:")
    if cuda_available and torch.cuda.device_count() > 0:
        props = torch.cuda.get_device_properties(0)
        if props.major >= 12 or "RTX 50" in props.name:
            print(f"    [OK] READY: PyTorch with CUDA for RTX 5070")
            return True
        else:
            print(f"    [WARN] PyTorch CUDA installed but GPU may need newer version")
            return True
    else:
        print(f"    [ERROR] NEEDS INSTALL: PyTorch CUDA version")
        return False

if __name__ == "__main__":
    ready = check_pytorch()
    sys.exit(0 if ready else 1)
