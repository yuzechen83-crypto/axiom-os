@echo off
REM Axiom-OS x NVIDIA Isaac Sim Launcher
REM Properly configures Python environment for Isaac Sim 4.0+

setlocal EnableDelayedExpansion

REM Paths
set ISAAC_ROOT=C:\Users\ASUS\isaacsim\_build\windows-x86_64\release
set PROJECT_PATH=C:\Users\ASUS\PycharmProjects\PythonProject1
set KIT_PATH=%ISAAC_ROOT%\kit

echo ============================================
echo Axiom-OS x NVIDIA Isaac Sim Launcherecho ============================================
echo.

REM Set up Python environment
set CARB_APP_PATH=%KIT_PATH%
set PYTHONPATH=%PYTHONPATH%;%KIT_PATH%\site

REM Add extension paths for isaacsim modules
set PYTHONPATH=%PYTHONPATH%;%ISAAC_ROOT%\exts\isaacsim.core.api
set PYTHONPATH=%PYTHONPATH%;%ISAAC_ROOT%\exts\isaacsim.core.cloner
set PYTHONPATH=%PYTHONPATH%;%ISAAC_ROOT%\exts\isaacsim.sensors.physics
set PYTHONPATH=%PYTHONPATH%;%ISAAC_ROOT%\exts\omni.isaac.core_archive

REM Find Python executable
if exist "%KIT_PATH%\python\python.exe" (
    set PYTHONEXE=%KIT_PATH%\python\python.exe
) else if exist "%KIT_PATH%\..\..\target-deps\python-3.11\python.exe" (
    set PYTHONEXE=%KIT_PATH%\..\..\target-deps\python-3.11\python.exe
) else if exist "%KIT_PATH%\..\..\target-deps\python-3.10\python.exe" (
    set PYTHONEXE=%KIT_PATH%\..\..\target-deps\python-3.10\python.exe
) else (
    echo [ERROR] Python not found in Isaac Sim installation
    exit /b 1
)

echo Isaac Sim Root: %ISAAC_ROOT%
echo Python: %PYTHONEXE%
echo Project: %PROJECT_PATH%
echo.

REM Build arguments
set ARGS=
:parse_args
if "%~1"=="" goto :run
set ARGS=!ARGS! %~1
shift
goto :parse_args

:run
echo Launching with args: %ARGS%
echo.

REM Change to project directory
cd /d "%PROJECT_PATH%"

REM Run the demo
%PYTHONEXE% "%PROJECT_PATH%\run_isaac_demo.py" %ARGS%

if errorlevel 1 (
    echo.
    echo [ERROR] Demo failed with code %errorlevel%
    pause
    exit /b %errorlevel%
)

echo.
echo Demo completed successfully!
pause
