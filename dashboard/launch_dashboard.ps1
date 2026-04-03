$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$AppPath = Join-Path $ScriptDir "app.py"

python -m streamlit run $AppPath --server.headless false --server.runOnSave true
