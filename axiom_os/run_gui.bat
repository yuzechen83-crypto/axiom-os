@echo off
REM Axiom-OS 可视化窗口
REM 需先安装: pip install streamlit matplotlib torch numpy
echo Starting Axiom-OS GUI...
cd /d "%~dp0.."
streamlit run axiom_os/gui_app.py --server.headless true
pause
