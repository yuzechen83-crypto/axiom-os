@echo off
echo ==========================================
echo Install PyTorch Nightly with CUDA 12.8
echo For RTX 5070 (sm_120) support
echo ==========================================
echo.

REM Activate virtual environment
call .venv\Scripts\activate.bat

REM Uninstall current CPU version
echo [1/4] Uninstalling current PyTorch (CPU)...
pip uninstall torch torchvision torchaudio -y

REM Install nightly with CUDA 12.8
echo [2/4] Installing PyTorch Nightly with CUDA 12.8...
pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128

REM Verify installation
echo [3/4] Verifying installation...
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'CUDA Version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo ==========================================
echo Installation Complete!
echo ==========================================
pause
