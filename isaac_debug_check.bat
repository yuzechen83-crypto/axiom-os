@echo off
chcp 65001 >nul
REM ============================================================
REM Isaac Sim 环境诊断工具
REM 用于检查 Python/numpy 环境是否正确
REM ============================================================

setlocal EnableDelayedExpansion

REM === 配置路径 ===
set ISAAC_PATH=C:\Users\ASUS\isaacsim\_build\windows-x86_64\release

echo.
echo ============================================
echo   Isaac Sim 环境诊断工具
echo ============================================
echo.

REM 检查 Isaac Sim 目录
echo [1/5] 检查 Isaac Sim 安装...
if not exist "%ISAAC_PATH%\python.bat" (
    echo   [错误] 未找到 python.bat
    echo   路径: %ISAAC_PATH%
    pause
    exit /b 1
)
echo   [OK] 找到 Isaac Sim: %ISAAC_PATH%

REM 检查自带 Python/numpy
echo.
echo [2/5] 检查 Isaac 自带 Python 和 numpy...
cd /d "%ISAAC_PATH%"
python.bat -c "import sys; print(f'Python: {sys.executable}')" 2>nul
if errorlevel 1 (
    echo   [错误] Python 启动失败
) else (
    echo   [OK] Python 可启动
)

echo.
echo [3/5] 检查 numpy 版本...
python.bat -c "import numpy; print(f'numpy 版本: {numpy.__version__}')" 2>nul
if errorlevel 1 (
    echo   [错误] numpy 导入失败！这是主要问题
    echo   建议: 重新安装依赖或改用 Omniverse Launcher 版本
) else (
    echo   [OK] numpy 正常
)

REM 检查关键扩展
echo.
echo [4/5] 检查关键扩展目录...
set "EXT_PATH=%ISAAC_PATH%\exts"
if exist "%EXT_PATH%\omni.syntheticdata" (
    echo   [发现] omni.syntheticdata 扩展
)
if exist "%EXT_PATH%\omni.replicator" (
    echo   [发现] omni.replicator 扩展
)

REM 检查日志
echo.
echo [5/5] 检查最近的错误日志...
set "LOG_PATH=%ISAAC_PATH%\kit\logs"
if exist "%LOG_PATH%" (
    for /f "delims=" %%i in ('dir /b /o-d "%LOG_PATH%\*.log" 2^>nul') do (
        echo   最新日志: %%i
        findstr /i "error fail numpy replicator syntheticdata" "%LOG_PATH%\%%i" 2>nul | findstr /v /i "info debug" | head -5
        goto :logs_done
    )
) else (
    echo   日志目录不存在
)
:logs_done

echo.
echo ============================================
echo   诊断完成
echo ============================================
echo.
echo 建议修复步骤:
echo 1. 如果 numpy 导入失败，运行 isaac_repair_deps.bat
echo 2. 如果 replicator 报错，运行 isaac_minimal.bat（禁用扩展）
echo 3. 或者改用 Omniverse Launcher 安装的 Isaac Sim
echo.
pause
