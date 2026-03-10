@echo off
chcp 65001 >nul
REM ============================================================
REM 打开 NVIDIA Isaac Sim 图形界面
REM ============================================================

echo.
echo ============================================
echo   启动 NVIDIA Isaac Sim
echo ============================================
echo.

REM Isaac Sim 路径
set ISAAC_PATH=C:\Users\ASUS\isaacsim\_build\windows-x86_64\release

if not exist "%ISAAC_PATH%\isaac-sim.bat" (
    echo [错误] 未找到 Isaac Sim
    echo 路径: %ISAAC_PATH%
    pause
    exit /b 1
)

echo [启动] 正在打开 Isaac Sim...
echo [路径] %ISAAC_PATH%
echo.
echo 提示: 首次启动可能需要 1-3 分钟加载
echo       请耐心等待界面出现
echo.

start "" "%ISAAC_PATH%\isaac-sim.bat"

echo Isaac Sim 正在后台启动...
echo.
pause
