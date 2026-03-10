@echo off
chcp 65001 >nul
REM ============================================================
REM Axiom-OS x Isaac Sim 最小化启动（禁用问题扩展）
REM 用于绕过 replicator/syntheticdata 初始化错误
REM ============================================================

setlocal EnableDelayedExpansion

set PROJECT_PATH=%~dp0
set PROJECT_PATH=%PROJECT_PATH:~0,-1%

REM Isaac Sim 路径（自编译版）
set ISAAC_PATH=C:\Users\ASUS\isaacsim\_build\windows-x86_64\release

echo.
echo ============================================
echo   Axiom-OS x Isaac Sim [最小化模式]
echo ============================================
echo.
echo 注意：此模式禁用 replicator/syntheticdata 扩展
echo       用于绕过 numpy 初始化错误
echo.

REM 检查路径
if not exist "%ISAAC_PATH%\python.bat" (
    echo [错误] 未找到 Isaac Sim: %ISAAC_PATH%
    echo 请修改本文件中的 ISAAC_PATH 路径
    pause
    exit /b 1
)

REM 设置最小化启动环境变量
echo [配置] 设置最小化扩展加载...
set OMNI_KIT_DISABLE_EXTENSIONS=omni.replicator.core,omni.replicator.composer,omni.syntheticdata
set ISAAC_SIM_MINIMAL=1

REM 禁用 viewport 相关可能引发问题的扩展
set OMNI_DISABLE_EXTENSIONS=omni.kit.viewport.rtx,omni.kit.viewport.pxr

echo [启动] 使用最小化模式运行...
cd /d "%PROJECT_PATH%"

REM 使用 --/renderer/enabled=false 禁用高级渲染
echo.
"%ISAAC_PATH%\python.bat" "%PROJECT_PATH%\run_isaac_demo.py" --no-video --headless %*

echo.
echo 完成。
pause
