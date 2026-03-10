# Force Reinstall PyTorch with CUDA 12.8 for RTX 5070
# Run as Administrator if possible

Write-Host "==========================================" -ForegroundColor Cyan
Write-Host "Force Reinstall PyTorch for RTX 5070" -ForegroundColor Cyan
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""

# Step 1: Kill Python processes using PyTorch
Write-Host "[1/6] Stopping Python processes..." -ForegroundColor Yellow
$pythonProcesses = Get-Process -Name "python" -ErrorAction SilentlyContinue
if ($pythonProcesses) {
    Write-Host "    Found $($pythonProcesses.Count) Python process(es), stopping..."
    Stop-Process -Name "python" -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
    Write-Host "    [OK] Python processes stopped"
} else {
    Write-Host "    [OK] No Python processes running"
}

# Step 2: Clear pip cache
Write-Host ""
Write-Host "[2/6] Clearing pip cache..." -ForegroundColor Yellow
& .\.venv\Scripts\python.exe -m pip cache purge 2>$null
Write-Host "    [OK] Cache cleared"

# Step 3: Uninstall existing PyTorch completely
Write-Host ""
Write-Host "[3/6] Uninstalling existing PyTorch (force)..." -ForegroundColor Yellow
$packages = @("torch", "torchvision", "torchaudio", "pytorch")
foreach ($pkg in $packages) {
    & .\.venv\Scripts\python.exe -m pip uninstall $pkg -y 2>$null | Out-Null
    Write-Host "    Uninstalled: $pkg"
}
Write-Host "    [OK] PyTorch removed"

# Step 4: Clean up残留文件
Write-Host ""
Write-Host "[4/6] Cleaning up residual files..." -ForegroundColor Yellow
$sitePackages = ".\.venv\Lib\site-packages"
$torchDirs = Get-ChildItem -Path $sitePackages -Filter "torch*" -Directory -ErrorAction SilentlyContinue
foreach ($dir in $torchDirs) {
    Remove-Item -Path $dir.FullName -Recurse -Force -ErrorAction SilentlyContinue
    Write-Host "    Removed: $($dir.Name)"
}
Write-Host "    [OK] Cleanup complete"

# Step 5: Install PyTorch Nightly with CUDA 12.8
Write-Host ""
Write-Host "[5/6] Installing PyTorch Nightly with CUDA 12.8..." -ForegroundColor Green
Write-Host "    This may take 10-30 minutes depending on network speed..." -ForegroundColor Gray
Write-Host ""

& .\.venv\Scripts\python.exe -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --no-cache-dir

if ($LASTEXITCODE -ne 0) {
    Write-Host ""
    Write-Host "[ERROR] Installation failed, trying alternative (CUDA 12.4)..." -ForegroundColor Red
    & .\.venv\Scripts\python.exe -m pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 --no-cache-dir
}

# Step 6: Verify installation
Write-Host ""
Write-Host "[6/6] Verifying installation..." -ForegroundColor Green
Write-Host ""

& .\.venv\Scripts\python.exe -c "
import torch
print('PyTorch Version:', torch.__version__)
print('CUDA Available:', torch.cuda.is_available())
if torch.cuda.is_available():
    print('CUDA Version:', torch.version.cuda)
    print('GPU:', torch.cuda.get_device_name(0))
    print('GPU Memory:', torch.cuda.get_device_properties(0).total_memory / 1e9, 'GB')
    # Test
    x = torch.randn(100, 100).cuda()
    y = torch.randn(100, 100).cuda()
    z = x @ y
    print('CUDA Test: PASSED')
else:
    print('WARNING: CUDA not available!')
"

Write-Host ""
Write-Host "==========================================" -ForegroundColor Cyan
if ($?) {
    Write-Host "Installation Complete!" -ForegroundColor Green
} else {
    Write-Host "Installation may have issues. Please check output above." -ForegroundColor Red
}
Write-Host "==========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "To verify: .venv\Scripts\python.exe check_cuda.py"

Pause
