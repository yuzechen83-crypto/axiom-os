@echo off
chcp 65001 >nul
REM ============================================================
REM Axiom-OS x Isaac Sim [Omniverse Launcher 版本]
REM 推荐使用 Launcher 安装的 Isaac Sim，更稳定
REM ============================================================

setlocal EnableDelayedExpansion

set PROJECT_PATH=%~dp0
set PROJECT_PATH=%PROJECT_PATH:~0,-1%

REM 自动查找 Omniverse Launcher 安装的 Isaac Sim
echo [查找] Omniverse Launcher 安装的 Isaac Sim...

set ISAAC_PATH=
set VERSIONS=4.5.0 4.2.0 4.1.0 4.0.0

for %%v in (%VERSIONS%) do (
    if "!ISAAC_PATH!"=="" (
        if exist "%LOCALAPPDATA%\ov\pkg\isaac-sim-%%v\python.bat" (
            set ISAAC_PATH=%LOCALAPPDATA%\ov\pkg\isaac-sim-%%v
            echo [找到] 版本 %%v
        )
    )
)

REM 如果没找到，提示用户
if "!ISAAC_PATH!"=="" (
    echo.
    echo [未找到 Launcher 版本]
    echo.
    echo 建议安装步骤:
    echo 1. 下载 NVIDIA Omniverse Launcher:
    echo    https://www.nvidia.com/omniverse/
    echo 2. 登录并进入 "Exchange" 标签
    echo 3. 搜索 "Isaac Sim" 并安装最新版 (4.2.0+)
    echo 4. 安装完成后重新运行此脚本
    echo.
    echo 当前将尝试使用自编译版本...
    set ISAAC_PATH=C:\Users\ASUS\isaacsim\_build\windows-x86_64\release
)

echo.
echo ============================================
echo   Axiom-OS x Isaac Sim
echo ============================================
echo   Isaac Sim: !ISAAC_PATH!
echo   项目目录:  %PROJECT_PATH%
echo ============================================
echo.

if not exist "!ISAAC_PATH!\python.bat" (
    echo [错误] 未找到 python.bat
    pause
    exit /b 1
)

cd /d "%PROJECT_PATH%"
"!ISAAC_PATH!\python.bat" "%PROJECT_PATH%\run_isaac_demo.py" %*

if errorlevel 1 (
    echo.
    echo [运行出错]
    pause
)
