@echo off
echo ==========================================
echo Reinstall PyTorch for RTX 5070
echo ==========================================
echo.

:: Activate virtual environment
call .venv\Scripts\activate.bat

:: Step 1: Kill any running Python processes
echo [1/5] Stopping Python processes...
taskkill /F /IM python.exe 2>nul
taskkill /F /IM pythonw.exe 2>nul
echo [OK] Processes stopped
echo.

:: Step 2: Uninstall existing PyTorch
echo [2/5] Removing existing PyTorch...
pip uninstall torch torchvision torchaudio -y 2>nul
pip uninstall torch torchvision torchaudio -y 2>nul
echo [OK] PyTorch removed
echo.

:: Step 3: Clear cache
echo [3/5] Clearing pip cache...
pip cache purge 2>nul
echo [OK] Cache cleared
echo.

:: Step 4: Install PyTorch Nightly with CUDA 12.8 (for RTX 5070 sm_120)
echo [4/5] Installing PyTorch Nightly with CUDA 12.8...
echo This may take 10-30 minutes...
echo.

pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128 --no-cache-dir

if errorlevel 1 (
    echo.
    echo [WARNING] CUDA 12.8 installation failed, trying CUDA 12.4...
    pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu124 --no-cache-dir
)

echo.
echo [5/5] Verifying installation...
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo.
echo ==========================================
echo Installation Complete!
echo ==========================================
echo.
echo Run: python check_cuda.py
echo Then: python jhtdb_gpu_experiment.py
pause
