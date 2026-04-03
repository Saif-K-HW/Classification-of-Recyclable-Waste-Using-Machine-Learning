@echo off
setlocal

set "SCRIPT_DIR=%~dp0"
set "PROJECT_ROOT=%SCRIPT_DIR%.."

python -m streamlit run "%SCRIPT_DIR%app.py" --server.headless false --server.runOnSave true

endlocal
