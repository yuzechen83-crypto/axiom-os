@echo off
REM Run Axiom-OS Demo - Choose your method

echo ============================================
echo Axiom-OS x Isaac Sim Launcher
echo ============================================
echo.
echo Choose running method:
echo.
echo [1] Auto-launch Isaac Sim + Demo (slow, 2-3 min)
echo [2] Isaac Sim already open - use Script Editor (fast)
echo [3] Headless mode (fastest, no GUI)
echo [4] View instructions
echo.
set /p choice="Enter choice (1-4): "

if "%choice%"=="1" goto :auto
if "%choice%"=="2" goto :editor
if "%choice%"=="3" goto :headless
if "%choice%"=="4" goto :instructions

echo Invalid choice
goto :end

:auto
echo.
echo Starting Isaac Sim with demo...
echo This will take 2-3 minutes...
echo.
start "" "C:\Users\ASUS\isaacsim\_build\windows-x86_64\release\kit\kit.exe" "C:\Users\ASUS\isaacsim\_build\windows-x86_64\release\apps\isaacsim.exp.full.kit" --exec "python" "C:\Users\ASUS\PycharmProjects\PythonProject1\axiom_go1_full_demo.py"
goto :end

:editor
echo.
echo ============================================
echo Method: Script Editor
echo ============================================
echo.
echo 1. In Isaac Sim, click: Window -^> Script Editor
echo 2. Click "Open" button
echo 3. Select: axiom_go1_full_demo.py
echo 4. Click "Run"
echo.
echo Opening file location...
explorer /select,"C:\Users\ASUS\PycharmProjects\PythonProject1\axiom_go1_full_demo.py"
goto :end

:headless
echo.
echo Starting in headless mode...
echo.
"C:\Users\ASUS\isaacsim\_build\windows-x86_64\release\kit\kit.exe" "C:\Users\ASUS\isaacsim\_build\windows-x86_64\release\apps\isaacsim.exp.full.kit" --/app/window/width=1 --/app/window/height=1 --exec "python" "C:\Users\ASUS\PycharmProjects\PythonProject1\axiom_go1_full_demo.py"
goto :end

:instructions
echo.
type "C:\Users\ASUS\PycharmProjects\PythonProject1\INSTRUCTIONS.txt"
pause
goto :end

:end
echo.
pause
