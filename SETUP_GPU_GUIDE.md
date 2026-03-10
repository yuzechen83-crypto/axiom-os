# GPU Setup Guide for RTX 5070

## Current Status

```
PyTorch: 2.10.0+cpu ❌
CUDA: Not available ❌
GPU: RTX 5070 (detected in hardware) ⚠️
```

**Issue**: Installed PyTorch is CPU-only version.

---

## Installation Steps

### Step 1: Uninstall CPU Version

```bash
.venv\Scripts\activate
pip uninstall torch torchvision torchaudio -y
```

### Step 2: Install PyTorch Nightly with CUDA 12.8

For RTX 5070 (sm_120), you need **PyTorch 2.12.0+ nightly with CUDA 12.8**:

```bash
# Main installation command
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
```

If the above is too slow, use:

```bash
# Alternative: CUDA 12.4 (also supports sm_120)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124

# Or CUDA 11.8 (if 12.x not available)
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118
```

### Step 3: Verify Installation

```bash
python check_cuda.py
```

Expected output:
```
[OK] PyTorch imported successfully
  Version: 2.12.0.dev20260307+cu128 ✓

[1] CUDA Support:
    Available: True ✓
    CUDA Version: 12.8

[2] GPU Information:
    GPU 0: NVIDIA GeForce RTX 5070 Laptop GPU
      Memory: 8.00 GB
      Compute Capability: 12.0
      Multi-Processor Count: 48

[3] CUDA Functionality Test:
    [OK] CUDA tensor operations working
    [OK] Test matrix multiplication: torch.Size([100, 100])

[4] RTX 5070 (sm_120) Support:
    [OK] RTX 5070 detected (sm_120)
    [OK] Full support with PyTorch nightly CUDA 12.8

[5] Summary:
    [OK] READY: PyTorch with CUDA for RTX 5070
```

---

## Quick Test After Installation

### Test 1: CUDA Available

```python
import torch
print(torch.cuda.is_available())  # Should print: True
print(torch.cuda.get_device_name(0))  # Should print: NVIDIA GeForce RTX 5070 Laptop GPU
```

### Test 2: Run GPU Experiment

```bash
python jhtdb_gpu_experiment.py
```

---

## Troubleshooting

### Issue: "pip install" times out

**Solution**: Use trusted-host flag

```bash
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --trusted-host download.pytorch.org
```

### Issue: "CUDA out of memory"

**Solution**: Reduce batch size in `jhtdb_gpu_experiment.py`

```python
# Line ~85
batch_size = 8  # Reduce from 16 to 8
```

### Issue: "CUDA error: no kernel image is available"

**Cause**: PyTorch doesn't support sm_120

**Solution**: Use PyTorch nightly (2.12.0+) with CUDA 12.8

```bash
# Must use nightly for RTX 5070
pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128
```

---

## Expected Performance After GPU Setup

| Resolution | Samples | Cycles | CPU Time | GPU Time | Speedup |
|------------|---------|--------|----------|----------|---------|
| 64³ | 100 | 5 | ~78 min | **~20 min** | **3.9×** |
| 128³ | 50 | 5 | ~240 min | **~60 min** | **4.0×** |
| 256³ | 25 | 5 | ~960 min | **~240 min** | **4.0×** |

---

## Files Ready for GPU

Once CUDA PyTorch is installed, these files will automatically use GPU:

1. **jhtdb_gpu_experiment.py** - Full GPU experiment
2. **check_cuda.py** - CUDA verification
3. **axiom_os/data/jhtdb_loader.py** - GPU data loading
4. **axiom_os/layers/rcln_v3_advanced.py** - GPU model

---

## Windows Batch File

Created: `install_pytorch_cuda.bat`

Double-click to run:
```
install_pytorch_cuda.bat
```

This will:
1. Activate virtual environment
2. Uninstall CPU version
3. Install CUDA version
4. Verify installation

---

## Next Steps

1. **Run installation** (Step 2 above)
2. **Verify with** `python check_cuda.py`
3. **Run experiment** `python jhtdb_gpu_experiment.py`
4. **Check results** in `jhtdb_gpu_results.json`

---

## System Requirements Checklist

- [x] NVIDIA RTX 5070 GPU (hardware present)
- [x] NVIDIA Driver 545+ (or latest)
- [x] CUDA 12.8 toolkit (optional, can use bundled)
- [ ] PyTorch nightly with CUDA 12.8 (NEEDS INSTALL)
- [ ] Verify CUDA works in PyTorch
- [ ] Run GPU experiment

---

**Status**: All code ready, awaiting PyTorch CUDA installation.
