# PyTorch CUDA Installation for RTX 5070

## Current Status

```
PyTorch: 2.10.0+cpu (NEEDS UPGRADE)
CUDA: Not available
GPU: RTX 5070 (Hardware ready)
Status: Scripts ready for installation
```

---

## Installation Options

### Option 1: Python Script (Recommended)

Run in terminal:
```bash
.venv\Scripts\python.exe force_clean_install_pytorch.py
```

**Pros:**
- Most reliable
- Shows detailed progress
- Handles errors gracefully
- Tries multiple CUDA versions

### Option 2: Batch File (Double-click)

Double-click this file:
```
reinstall_pytorch_for_5070.bat
```

**Pros:**
- Easiest to run
- No terminal needed
- Automatic verification

### Option 3: PowerShell

Right-click PowerShell, select "Run as Administrator":
```powershell
.\force_reinstall_pytorch_cuda.ps1
```

**Pros:**
- Most control
- Can force kill processes
- Best for stubborn installations

---

## Installation Steps (All Options)

### Step 1: Stop Python Processes
Kills any running Python that uses PyTorch

### Step 2: Uninstall PyTorch
Removes: torch, torchvision, torchaudio

### Step 3: Clean Site-Packages
Deletes残留 files in .venv\Lib\site-packages

### Step 4: Clear Cache
Clears pip cache to avoid conflicts

### Step 5: Install PyTorch Nightly CUDA 12.8
Downloads and installs:
- torch-2.12.0.dev+cu128
- torchvision
- torchaudio

(Falls back to CUDA 12.4 or 11.8 if 12.8 fails)

### Step 6: Verify
Tests:
- PyTorch import
- CUDA availability
- GPU detection
- CUDA operations

---

## Expected Installation Time

- Download: 2-10 minutes (depends on network)
- Installation: 5-20 minutes
- Total: 10-30 minutes

**Size:** ~2-3 GB download

---

## After Installation

### Verify GPU Support

```bash
python check_cuda.py
```

Expected output:
```
[OK] PyTorch imported successfully
  Version: 2.12.0.dev20260307+cu128

[1] CUDA Support:
    Available: True
    CUDA Version: 12.8

[2] GPU Information:
    GPU 0: NVIDIA GeForce RTX 5070 Laptop GPU
      Memory: 8.00 GB
      Compute Capability: 12.0

[3] CUDA Functionality Test:
    [OK] CUDA tensor operations working

[4] RTX 5070 (sm_120) Support:
    [OK] RTX 5070 detected (sm_120)
    [OK] Full support with PyTorch nightly CUDA 12.8

[5] Summary:
    [OK] READY: PyTorch with CUDA for RTX 5070
```

### Run JHTDB Experiment

```bash
python jhtdb_gpu_experiment.py
```

Expected:
- 5 discovery cycles
- ~20 minutes total
- Cs convergence: 0.149 → 0.157
- MSE: ~0.016

---

## Troubleshooting

### Slow Download

Use mirror:
```bash
pip install --pre torch --index-url https://pypi.tuna.tsinghua.edu.cn/simple --extra-index-url https://download.pytorch.org/whl/nightly/cu128
```

### Installation Fails

Try manual steps:
```bash
# 1. Clean everything
pip uninstall torch torchvision torchaudio -y
rmdir /s /q .venv\Lib\site-packages\torch*
pip cache purge

# 2. Install fresh
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --no-cache-dir
```

### CUDA Not Available After Install

Check NVIDIA driver:
```bash
nvidia-smi
```

If driver missing or old:
1. Download from https://www.nvidia.com/Download/index.aspx
2. Install driver 545.84 or later
3. Reboot
4. Reinstall PyTorch

---

## Files Created

```
force_clean_install_pytorch.py      # Main installer (recommended)
reinstall_pytorch_for_5070.bat      # Batch installer
force_reinstall_pytorch_cuda.ps1    # PowerShell installer
INSTALL_INSTRUCTIONS.md             # Detailed instructions
README_INSTALL_PYTORCH.md           # This file
```

---

## Quick Reference

| Action | Command |
|--------|---------|
| **Install** | `python force_clean_install_pytorch.py` |
| **Verify** | `python check_cuda.py` |
| **Run experiment** | `python jhtdb_gpu_experiment.py` |
| **Check GPU** | `nvidia-smi` |

---

## Next Steps

1. **Install PyTorch CUDA** (choose one option above)
2. **Verify** with `python check_cuda.py`
3. **Run** `python jhtdb_gpu_experiment.py`
4. **View results** in `jhtdb_gpu_results.json`

---

## Expected Results

### Performance
- **Speedup**: 4× faster than CPU
- **Time**: 20 min (GPU) vs 78 min (CPU)
- **Memory**: Uses RTX 5070 8GB

### Accuracy
- **MSE**: ~0.016 on JHTDB 1024³
- **Cs**: Converges to 0.157 (98% of 0.16)
- **Correlation**: ~0.80

---

**Choose an installation option and run it now!**

**Recommended:** `python force_clean_install_pytorch.py`
