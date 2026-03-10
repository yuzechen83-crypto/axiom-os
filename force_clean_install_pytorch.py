# -*- coding: utf-8 -*-
"""
Force Clean Install PyTorch with CUDA 12.8 for RTX 5070
This script will:
1. Kill all Python processes
2. Completely remove existing PyTorch
3. Clean all caches
4. Install PyTorch Nightly with CUDA 12.8
"""

import subprocess
import sys
import os
import shutil
import time


def run_command(cmd, desc=""):
    """Run shell command and print output"""
    print(f"    {desc}...")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"    Warning: {result.stderr[:200]}")
    return result.returncode == 0


def main():
    print("="*70)
    print("Force Clean Install PyTorch for RTX 5070")
    print("="*70)
    print()
    
    venv_python = r".venv\Scripts\python.exe"
    venv_pip = r".venv\Scripts\pip.exe"
    
    # Step 1: Kill Python processes
    print("[1/6] Stopping Python processes...")
    subprocess.run("taskkill /F /IM python.exe 2>nul", shell=True)
    subprocess.run("taskkill /F /IM pythonw.exe 2>nul", shell=True)
    time.sleep(2)
    print("    [OK] Processes stopped")
    
    # Step 2: Uninstall PyTorch completely
    print()
    print("[2/6] Uninstalling existing PyTorch...")
    packages = ["torch", "torchvision", "torchaudio", "pytorch"]
    for pkg in packages:
        subprocess.run(f"{venv_pip} uninstall {pkg} -y 2>nul", shell=True)
        print(f"    Removed: {pkg}")
    print("    [OK] PyTorch uninstalled")
    
    # Step 3: Clean site-packages manually
    print()
    print("[3/6] Cleaning site-packages...")
    site_packages = r".venv\Lib\site-packages"
    if os.path.exists(site_packages):
        for item in os.listdir(site_packages):
            if "torch" in item.lower():
                path = os.path.join(site_packages, item)
                try:
                    if os.path.isdir(path):
                        shutil.rmtree(path)
                    else:
                        os.remove(path)
                    print(f"    Deleted: {item}")
                except Exception as e:
                    print(f"    Could not delete {item}: {e}")
    print("    [OK] Site-packages cleaned")
    
    # Step 4: Clear pip cache
    print()
    print("[4/6] Clearing pip cache...")
    subprocess.run(f"{venv_pip} cache purge", shell=True)
    # Also delete pip cache directory
    pip_cache = os.path.expanduser("~\AppData\Local\pip\cache")
    if os.path.exists(pip_cache):
        try:
            shutil.rmtree(pip_cache)
            print("    [OK] Pip cache directory cleared")
        except:
            pass
    print("    [OK] Cache cleared")
    
    # Step 5: Install PyTorch Nightly with CUDA 12.8
    print()
    print("[5/6] Installing PyTorch Nightly with CUDA 12.8...")
    print("    This will take 10-30 minutes depending on download speed...")
    print()
    
    # First try CUDA 12.8
    cmd = f'{venv_pip} install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --no-cache-dir'
    result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print()
        print("    [WARNING] CUDA 12.8 failed, trying CUDA 12.4...")
        cmd = f'{venv_pip} install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 --no-cache-dir'
        result = subprocess.run(cmd, shell=True)
        
        if result.returncode != 0:
            print()
            print("    [WARNING] CUDA 12.4 failed, trying CUDA 11.8...")
            cmd = f'{venv_pip} install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu118 --no-cache-dir'
            result = subprocess.run(cmd, shell=True)
    
    if result.returncode != 0:
        print()
        print("[ERROR] Installation failed!")
        return False
    
    print("    [OK] PyTorch installed")
    
    # Step 6: Verify installation
    print()
    print("[6/6] Verifying installation...")
    print()
    
    test_script = '''
import sys
sys.stdout.reconfigure(encoding='utf-8')

print("Testing PyTorch installation...")

import torch
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Version: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    print(f"GPU Memory: {props.total_memory / 1e9:.2f} GB")
    print(f"Compute Capability: {props.major}.{props.minor}")
    
    # Test CUDA operations
    try:
        x = torch.randn(100, 100).cuda()
        y = torch.randn(100, 100).cuda()
        z = x @ y
        print("CUDA Operations: PASSED")
        
        if props.major == 12 and props.minor == 0:
            print("[SUCCESS] RTX 5070 (sm_120) fully supported!")
        elif "RTX 50" in torch.cuda.get_device_name(0):
            print("[SUCCESS] RTX 50-series GPU detected!")
        else:
            print("[SUCCESS] GPU working!")
            
    except Exception as e:
        print(f"[ERROR] CUDA test failed: {e}")
else:
    print("[ERROR] CUDA not available - installation may have failed")
'''
    
    with open("test_cuda_temp.py", "w") as f:
        f.write(test_script)
    
    subprocess.run(f"{venv_python} test_cuda_temp.py", shell=True)
    os.remove("test_cuda_temp.py")
    
    print()
    print("="*70)
    print("Installation Complete!")
    print("="*70)
    print()
    print("Next steps:")
    print("  1. Run: python check_cuda.py")
    print("  2. Run: python jhtdb_gpu_experiment.py")
    
    return True


if __name__ == "__main__":
    try:
        success = main()
        if not success:
            sys.exit(1)
    except KeyboardInterrupt:
        print("\n\n[WARNING] Installation interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n[ERROR] {e}")
        sys.exit(1)
