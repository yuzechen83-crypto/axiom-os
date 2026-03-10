@echo off
REM Run Axiom-OS demo using Isaac Sim Kit

setlocal

REM Paths
set ISAAC_ROOT=C:\Users\ASUS\isaacsim\_build\windows-x86_64\release
set KIT_EXE=%ISAAC_ROOT%\kit\kit.exe
set APP_CONFIG=%ISAAC_ROOT%\apps\isaacsim.exp.full.kit
set PROJECT_PATH=C:\Users\ASUS\PycharmProjects\PythonProject1

echo ============================================
echo Axiom-OS x Isaac Sim (Kit Launcher)
echo ============================================
echo.
echo Kit: %KIT_EXE%
echo App: %APP_CONFIG%
echo.

REM Check files exist
if not exist "%KIT_EXE%" (
    echo [ERROR] kit.exe not found
    exit /b 1
)

if not exist "%APP_CONFIG%" (
    echo [ERROR] App config not found
    exit /b 1
)

REM Change to project directory
cd /d "%PROJECT_PATH%"

REM Build arguments for the script
set SCRIPT_ARGS=--test
if not "%~1"=="" set SCRIPT_ARGS=%*

echo Running: %PROJECT_PATH%\run_isaac_demo.py
echo Args: %SCRIPT_ARGS%
echo.

REM Run with kit
"%KIT_EXE%" "%APP_CONFIG%" --exec "python" "%PROJECT_PATH%\run_isaac_demo.py" %SCRIPT_ARGS%

echo.
echo Done.
pause
