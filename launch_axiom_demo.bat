@echo off
REM Launch Axiom-OS Demo in Isaac Sim

setlocal

REM Isaac Sim paths
set ISAAC_ROOT=C:\Users\ASUS\isaacsim\_build\windows-x86_64\release
set KIT_EXE=%ISAAC_ROOT%\kit\kit.exe
set APP_CONFIG=%ISAAC_ROOT%\apps\isaacsim.exp.full.kit
set PROJECT_PATH=C:\Users\ASUS\PycharmProjects\PythonProject1

echo ============================================
echo Launching Axiom-OS x Isaac Sim Demo
echo ============================================
echo.

REM Check if Isaac Sim is already running
REM If yes, we need to use a different approach

echo Checking if Isaac Sim is already running...
tasklist | findstr "kit.exe" >nul
if %ERRORLEVEL% == 0 (
    echo.
    echo [INFO] Isaac Sim is already running!
    echo.
    echo Please use Method 2:
    echo 1. In Isaac Sim, go to Window -^> Script Editor
    echo 2. Click Open and select:
    echo    %PROJECT_PATH%\axiom_go1_full_demo.py
    echo 3. Click Run
    echo.
    pause
    exit /b 0
)

echo Starting Isaac Sim with Axiom-OS demo...
echo This will take 1-2 minutes to load...
echo.

REM Run Isaac Sim with the demo script
"%KIT_EXE%" "%APP_CONFIG%" --exec "python" "%PROJECT_PATH%\axiom_go1_full_demo.py"

echo.
echo Demo completed.
pause
