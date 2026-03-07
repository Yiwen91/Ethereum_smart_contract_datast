@echo off
REM Run label report from project root (script lives in Ethereum_smart_contract_datast subfolder).
cd /d "%~dp0Ethereum_smart_contract_datast"
if not exist test_labeling.py (
    echo test_labeling.py not found in %CD%
    echo Run this from the folder that contains test_labeling.py, or use:
    echo   py Ethereum_smart_contract_datast\test_labeling.py --report path\to\contract.sol
    exit /b 1
)

set FILE=%~1
if "%FILE%"=="" (
    echo Usage: run_report.bat path\to\contract.sol
    echo Example: run_report.bat Ethereum_smart_contract_datast\contract_dataset_ethereum\contract1\0.sol
    exit /b 1
)

if exist "%~1" set "FILE=%~1"
where py >nul 2>&1 && (py test_labeling.py --report "%FILE%" & exit /b %ERRORLEVEL%)
where python3 >nul 2>&1 && (python3 test_labeling.py --report "%FILE%" & exit /b %ERRORLEVEL%)
where python >nul 2>&1 && (python test_labeling.py --report "%FILE%" & exit /b %ERRORLEVEL%)
echo Python not found. Install from https://www.python.org/downloads/
exit /b 1
