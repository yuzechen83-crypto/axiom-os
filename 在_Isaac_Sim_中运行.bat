@echo off
chcp 65001 >nul
REM ============================================================
REM 在 Isaac Sim 中运行 Axiom-OS 演示（推荐使用 Omniverse Launcher 版本）
REM ============================================================

setlocal EnableDelayedExpansion

set PROJECT_PATH=%~dp0
set PROJECT_PATH=%PROJECT_PATH:~0,-1%

REM ============================================================
REM 步骤 1: 自动检测 Isaac Sim 安装路径（按推荐优先级）
REM ============================================================
echo.
echo [1/3] 检测 Isaac Sim 安装...

set ISAAC_SIM_PATH=

REM 优先级 0: Isaac Sim 5.1.0 (Omniverse Launcher)
for /d %%D in ("%LOCALAPPDATA%\ov\pkg\isaac-sim-5.1*") do (
    if "!ISAAC_SIM_PATH!"=="" (
        if exist "%%D\python.bat" (
            set ISAAC_SIM_PATH=%%D
            echo   [OK] 找到 Launcher 版本: %%~nxD
        )
    )
)

REM 优先级 1: Omniverse Launcher 安装的 Isaac Sim 4.5.x
for /d %%D in ("%LOCALAPPDATA%\ov\pkg\isaac-sim-4.5*") do (
    if "!ISAAC_SIM_PATH!"=="" (
        if exist "%%D\python.bat" (
            set ISAAC_SIM_PATH=%%D
            echo   [OK] 找到 Launcher 版本: %%~nxD
        )
    )
)

REM 优先级 2: Omniverse Launcher 4.2.x
for /d %%D in ("%LOCALAPPDATA%\ov\pkg\isaac-sim-4.2*") do (
    if "!ISAAC_SIM_PATH!"=="" (
        if exist "%%D\python.bat" (
            set ISAAC_SIM_PATH=%%D
            echo   [OK] 找到 Launcher 版本: %%~nxD
        )
    )
)

REM 优先级 3: Omniverse Launcher 其他版本
for /d %%D in ("%LOCALAPPDATA%\ov\pkg\isaac-sim-4*") do (
    if "!ISAAC_SIM_PATH!"=="" (
        if exist "%%D\python.bat" (
            set ISAAC_SIM_PATH=%%D
            echo   [OK] 找到 Launcher 版本: %%~nxD
        )
    )
)

REM 优先级 4: 工作站默认路径 (5.1.0 文档: C:\isaac-sim)
if "!ISAAC_SIM_PATH!"=="" (
    if exist "C:\isaac-sim\python.bat" (
        set ISAAC_SIM_PATH=C:\isaac-sim
        echo   [OK] 找到工作站安装: C:\isaac-sim
    )
)

REM 优先级 5: 自编译版本（备用）
if "!ISAAC_SIM_PATH!"=="" (
    if exist "C:\Users\ASUS\isaacsim\_build\windows-x86_64\release\python.bat" (
        set ISAAC_SIM_PATH=C:\Users\ASUS\isaacsim\_build\windows-x86_64\release
        echo   [警告] 使用自编译版本，可能遇到依赖问题
        echo          推荐改用 Omniverse Launcher 安装的 Isaac Sim 5.1.0
    )
)

REM 未找到任何版本
if "!ISAAC_SIM_PATH!"=="" (
    echo.
    echo [错误] 未找到 Isaac Sim 安装
echo.
    echo 请安装 NVIDIA Isaac Sim：
    echo   方法1 (推荐): 使用 Omniverse Launcher
echo     1. 下载 https://www.nvidia.com/omniverse/
echo     2. 在 Exchange 中搜索 "Isaac Sim" 并安装
echo.
    echo   方法2: 手动指定路径
echo     编辑本文件，取消下行的注释并修改路径:
    echo     set ISAAC_SIM_PATH=C:\你的\Isaac Sim 路径
    echo.
    pause
    exit /b 1
)

REM ============================================================
REM 步骤 2: 验证 Python 环境
REM ============================================================
echo.
echo [2/3] 验证 Python 环境...

set PYTHON_EXE=%ISAAC_SIM_PATH%\python.bat
cd /d "%ISAAC_SIM_PATH%"

REM 检查 numpy 是否可用
echo   检查 numpy...
python.bat -c "import numpy; print('  [OK] numpy:', numpy.__version__)" 2>nul
if errorlevel 1 (
    echo   [警告] numpy 检查失败！可能遇到 replicator/syntheticdata 错误
    echo.
    echo   修复选项:
    echo   1. 运行 isaac_launcher_version.bat 尝试其他版本
    echo   2. 运行 isaac_minimal.bat 禁用问题扩展
    echo   3. 安装/更新 VC++ Redistributable:
echo      https://aka.ms/vs/17/release/vc_redist.x64.exe
    echo.
    choice /C YN /M "是否继续尝试启动"
    if errorlevel 2 exit /b 1
)

REM ============================================================
REM 步骤 3: 启动演示
REM ============================================================
echo.
echo [3/3] 启动 Axiom-OS 演示...
echo.
echo ============================================
echo   Isaac Sim: !ISAAC_SIM_PATH!
echo   项目目录:  %PROJECT_PATH%
echo ============================================
echo.

cd /d "%PROJECT_PATH%"
"%ISAAC_SIM_PATH%\python.bat" "%PROJECT_PATH%\run_isaac_demo.py" %*

set EXIT_CODE=%errorlevel%

if %EXIT_CODE% neq 0 (
    echo.
    echo [运行出错] 退出码: %EXIT_CODE%
    echo.
    echo 常见错误及修复:
    echo   1. numpy/replicator 错误 -> 运行 isaac_minimal.bat
    echo   2. 显卡/渲染错误 -> 更新 NVIDIA 驱动
    echo   3. 扩展加载失败 -> 改用 Omniverse Launcher 版本
    echo.
    pause
    exit /b %EXIT_CODE%
)

echo.
echo 演示运行完成。
pause
