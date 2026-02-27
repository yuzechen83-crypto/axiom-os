# Axiom-OS 本地部署脚本 (Windows PowerShell)
# 用法: 在项目根目录执行 .\scripts\setup_local.ps1

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
Set-Location $Root

Write-Host "=== Axiom-OS 本地部署 ===" -ForegroundColor Cyan
Write-Host "项目目录: $Root`n"

# 1. 检查 Python
$py = Get-Command python -ErrorAction SilentlyContinue
if (-not $py) { $py = Get-Command py -ErrorAction SilentlyContinue }
if (-not $py) {
    Write-Host "未找到 Python，请先安装 Python 3.10+ 并加入 PATH" -ForegroundColor Red
    exit 1
}
Write-Host "[1/3] Python: $($py.Source)"
& $py.Source --version

# 2. 安装依赖
Write-Host "`n[2/3] 安装依赖 (requirements.txt) ..." -ForegroundColor Yellow
& $py.Source -m pip install --upgrade pip -q
& $py.Source -m pip install -r requirements.txt -q
# 全部组件已合并至 axiom_os，无需安装 axiom_core_proprietary
Write-Host "      完成."

# 3. 快速验证
Write-Host "`n[3/3] 快速验证 (quick benchmark) ..." -ForegroundColor Yellow
& $py.Source -m axiom_os.benchmarks.run_benchmarks --config quick --report --no-fail-on-alerts 2>&1 | Out-Host

Write-Host "`n=== 部署完成 ===" -ForegroundColor Green
Write-Host "报告位置: axiom_os\benchmarks\results\benchmark_report.html"
Write-Host "全流程验证: python -m axiom_os.validate_all"
Write-Host "Chat 界面:   streamlit run axiom_os/agent/chat_ui.py"
