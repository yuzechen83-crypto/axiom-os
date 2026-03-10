@echo off
REM Axiom-OS x NVIDIA Isaac Sim Launcher
REM Uses project venv Python with Isaac Sim modules

setlocal EnableDelayedExpansion

REM Paths
set ISAAC_ROOT=C:\Users\ASUS\isaacsim\_build\windows-x86_64\release
set PROJECT_PATH=C:\Users\ASUS\PycharmProjects\PythonProject1
set KIT_PATH=%ISAAC_ROOT%\kit
set VENV_PYTHON=%PROJECT_PATH%\.venv\Scripts\python.exe

echo ============================================
echo Axiom-OS x NVIDIA Isaac Sim Launcherecho ============================================
echo.

REM Check venv Python exists
if not exist "%VENV_PYTHON%" (
    echo [ERROR] Virtual environment Python not found at %VENV_PYTHON%
    echo Please create venv: python -m venv .venv
    exit /b 1
)

REM Set up Isaac Sim environment
set CARB_APP_PATH=%KIT_PATH%

REM Set PYTHONPATH to include Isaac Sim modules
set PYTHONPATH=%PROJECT_PATH%
set PYTHONPATH=%PYTHONPATH%;%KIT_PATH%\site

REM Add extension paths for isaacsim modules
set PYTHONPATH=%PYTHONPATH%;%ISAAC_ROOT%\exts\isaacsim.core.api
set PYTHONPATH=%PYTHONPATH%;%ISAAC_ROOT%\exts\isaacsim.core.cloner
set PYTHONPATH=%PYTHONPATH%;%ISAAC_ROOT%\exts\isaacsim.core.deprecation_manager
set PYTHONPATH=%PYTHONPATH%;%ISAAC_ROOT%\exts\isaacsim.sensors.physics
set PYTHONPATH=%PYTHONPATH%;%ISAAC_ROOT%\exts\omni.isaac.core_archive

REM Add kit kernel and exts
set PYTHONPATH=%PYTHONPATH%;%KIT_PATH%\extscore
set PYTHONPATH=%PYTHONPATH%;%KIT_PATH%\exts
set PYTHONPATH=%PYTHONPATH%;%KIT_PATH%\kernel

REM Add carb
set PYTHONPATH=%PYTHONPATH%;%KIT_PATH%

echo Python: %VENV_PYTHON%
echo Isaac Root: %ISAAC_ROOT%
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
%VENV_PYTHON% "%PROJECT_PATH%\run_isaac_demo.py" %ARGS%

if errorlevel 1 (
    echo.
    echo [ERROR] Demo failed with code %errorlevel%
    pause
    exit /b %errorlevel%
)

echo.
echo Demo completed successfully!
pause
