@echo off
REM Inspect labels for one contract. Uses py, python3, or python (whichever is available).
cd /d "%~dp0"
set SCRIPT=test_labeling.py
set FILE=%~1
if "%FILE%"=="" (
    echo Usage: run_report.bat path\to\contract.sol
    echo Example: run_report.bat contract_dataset_ethereum\contract1\0.sol
    exit /b 1
)

where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    py "%SCRIPT%" --report "%FILE%"
    exit /b %ERRORLEVEL%
)
where python3 >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python3 "%SCRIPT%" --report "%FILE%"
    exit /b %ERRORLEVEL%
)
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python "%SCRIPT%" --report "%FILE%"
    exit /b %ERRORLEVEL%
)

echo Python not found. Try one of these in a terminal:
echo   py %SCRIPT% --report "%FILE%"
echo   python3 %SCRIPT% --report "%FILE%"
echo   python %SCRIPT% --report "%FILE%"
echo Or install Python from https://www.python.org/downloads/ and add it to PATH.
exit /b 1
