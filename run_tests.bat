@echo off
REM Run labeling tests (script lives in Ethereum_smart_contract_datast subfolder).
cd /d "%~dp0Ethereum_smart_contract_datast"
if not exist test_labeling.py (
    echo test_labeling.py not found in %CD%
    echo Run from the folder that contains test_labeling.py, or use:
    echo   py Ethereum_smart_contract_datast\test_labeling.py
    exit /b 1
)
where py >nul 2>&1 && (py test_labeling.py & exit /b %ERRORLEVEL%)
where python3 >nul 2>&1 && (python3 test_labeling.py & exit /b %ERRORLEVEL%)
where python >nul 2>&1 && (python test_labeling.py & exit /b %ERRORLEVEL%)
echo Python not found. Install from https://www.python.org/downloads/
exit /b 1
