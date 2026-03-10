@echo off
REM Axiom-OS x NVIDIA Isaac Sim Launcher (Windows)
REM 
REM Usage: run_isaac.bat [options]
REM Options:
REM   --test          Quick test (100 steps)
REM   --steps N       Run for N steps
REM   --headless      Run without rendering
REM   --no-video      Disable video recording
REM   --threshold T   Set discovery threshold

setlocal enabledelayedexpansion

REM Configuration - Isaac Sim 5.1.0: try Launcher path first, then workstation/build
set PROJECT_PATH=C:\Users\ASUS\PycharmProjects\PythonProject1
set ISAAC_SIM_PATH=
if exist "%LOCALAPPDATA%\ov\pkg\isaac-sim-5.1.0\python.bat" set ISAAC_SIM_PATH=%LOCALAPPDATA%\ov\pkg\isaac-sim-5.1.0
if "!ISAAC_SIM_PATH!"=="" if exist "C:\isaac-sim\python.bat" set ISAAC_SIM_PATH=C:\isaac-sim
if "!ISAAC_SIM_PATH!"=="" set ISAAC_SIM_PATH=C:\Users\ASUS\isaacsim\_build\windows-x86_64\release

REM Check if Isaac Sim exists
if not exist "%ISAAC_SIM_PATH%\python.bat" (
    echo [ERROR] Isaac Sim not found at %ISAAC_SIM_PATH%
    echo Please edit this script and set ISAAC_SIM_PATH to your Isaac Sim installation
    exit /b 1
)

REM Check if project exists
if not exist "%PROJECT_PATH%\run_isaac_demo.py" (
    echo [ERROR] Project not found at %PROJECT_PATH%
    echo Please edit this script and set PROJECT_PATH to your project
    exit /b 1
)

echo ============================================
echo Axiom-OS x NVIDIA Isaac Sim Launcher
echo ============================================
echo.
echo Isaac Sim: %ISAAC_SIM_PATH%
echo Project:   %PROJECT_PATH%
echo.

REM Build command line arguments
set ARGS=

:parse_args
if "%~1"=="" goto :run
set ARGS=!ARGS! %~1
shift
goto :parse_args

:run
echo Launching with arguments:%ARGS%
echo.

REM Change to project directory
cd /d "%PROJECT_PATH%"

REM Run with Isaac Sim's Python
"%ISAAC_SIM_PATH%\python.bat" "%PROJECT_PATH%\run_isaac_demo.py" %ARGS%

if %ERRORLEVEL% neq 0 (
    echo.
    echo [ERROR] Simulation failed with code %ERRORLEVEL%
    pause
    exit /b %ERRORLEVEL%
)

echo.
echo Simulation completed successfully!
pause
