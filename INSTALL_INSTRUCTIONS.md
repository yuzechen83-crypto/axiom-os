# Install PyTorch CUDA for RTX 5070 - Instructions

## Quick Start (Recommended)

### Option 1: Python Script (Most Reliable)

```bash
.venv\Scripts\python.exe force_clean_install_pytorch.py
```

### Option 2: Batch File (Double-click)

Double-click:
```
reinstall_pytorch_for_5070.bat
```

### Option 3: PowerShell (Administrator)

```powershell
.\force_reinstall_pytorch_cuda.ps1
```

---

## What These Scripts Do

1. **Kill Python processes** - Stop any running Python that uses PyTorch
2. **Uninstall existing PyTorch** - Remove CPU version completely
3. **Clean caches** - Remove pip cache and残留 files
4. **Install PyTorch Nightly CUDA 12.8** - For RTX 5070 (sm_120)
5. **Verify installation** - Test CUDA and GPU

---

## Expected Output

```
==========================================
Force Clean Install PyTorch for RTX 5070
==========================================

[1/6] Stopping Python processes...
    [OK] Processes stopped

[2/6] Uninstalling existing PyTorch...
    Removed: torch
    Removed: torchvision
    Removed: torchaudio
    [OK] PyTorch uninstalled

[3/6] Cleaning site-packages...
    Deleted: torch
    Deleted: torch-2.10.0.dist-info
    [OK] Site-packages cleaned

[4/6] Clearing pip cache...
    [OK] Cache cleared

[5/6] Installing PyTorch Nightly with CUDA 12.8...
    This will take 10-30 minutes depending on download speed...
    
    Downloading torch-2.12.0.dev20260307+cu128...
    Installing collected packages: torch, torchvision, torchaudio
    [OK] PyTorch installed

[6/6] Verifying installation...

Testing PyTorch installation...
PyTorch Version: 2.12.0.dev20260307+cu128
CUDA Available: True
CUDA Version: 12.8
GPU: NVIDIA GeForce RTX 5070 Laptop GPU
GPU Memory: 8.00 GB
Compute Capability: 12.0
CUDA Operations: PASSED
[SUCCESS] RTX 5070 (sm_120) fully supported!

==========================================
Installation Complete!
==========================================

Next steps:
  1. Run: python check_cuda.py
  2. Run: python jhtdb_gpu_experiment.py
```

---

## Troubleshooting

### Issue: "pip install" times out

**Solution**: Use offline installation or mirror

```bash
# Use Tsinghua mirror (China)
pip install --pre torch torchvision torchaudio --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

### Issue: "No space left on device"

**Solution**: Clean temp files

```bash
# Clean Windows temp
rd /s /q %TEMP%
md %TEMP%

# Run installer again
python force_clean_install_pytorch.py
```

### Issue: "Access denied"

**Solution**: Run as Administrator

1. Right-click on Command Prompt / PowerShell
2. Select "Run as Administrator"
3. Run the installation script

### Issue: CUDA not available after install

**Solution**: Check NVIDIA drivers

```bash
# Check driver version
nvidia-smi

# If driver too old, download from:
# https://www.nvidia.com/Download/index.aspx
```

Minimum driver for RTX 5070: **545.84 or later**

---

## Verification After Install

### Test 1: Check CUDA

```bash
python check_cuda.py
```

Should show:
- PyTorch Version: 2.12.0+ (with cu128 or cu124)
- CUDA Available: True
- GPU: NVIDIA GeForce RTX 5070

### Test 2: Quick GPU test

```bash
python -c "import torch; x = torch.randn(1000, 1000).cuda(); y = torch.randn(1000, 1000).cuda(); z = x @ y; print('GPU test passed!')"
```

### Test 3: Run JHTDB experiment

```bash
python jhtdb_gpu_experiment.py
```

---

## File Structure

```
force_clean_install_pytorch.py      # Python installer (recommended)
reinstall_pytorch_for_5070.bat      # Batch installer (double-click)
force_reinstall_pytorch_cuda.ps1    # PowerShell installer
INSTALL_INSTRUCTIONS.md             # This file
```

---

## Post-Installation

Once installed, you can run:

```bash
# Verify GPU
python check_cuda.py

# Run full JHTDB experiment
python jhtdb_gpu_experiment.py

# Expected: ~20 minutes for 5 discovery cycles
# Expected: MSE ~0.016, Cs converges to ~0.157
```

---

## Support

If installation fails:
1. Check NVIDIA driver: `nvidia-smi`
2. Check disk space: at least 5GB free
3. Check internet connection
4. Try alternative CUDA version (12.4 or 11.8)

---

**Ready to install? Choose one option above and run it!**
