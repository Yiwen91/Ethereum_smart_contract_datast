@echo off
cd /d "%~dp0"
set DIR=contract_dataset_ethereum
set OUT=standardized_dataset
set REPORT=vulnerability_report.txt

echo Processing dataset 1: %DIR%
where py >nul 2>&1 && (
    py standardize_dataset.py "%DIR%" --output-dir "%OUT%" --format both
    if errorlevel 1 exit /b %ERRORLEVEL%
    py report_vulnerability_counts.py "%DIR%" -o "%REPORT%"
    exit /b %ERRORLEVEL%
)
where python3 >nul 2>&1 && (
    python3 standardize_dataset.py "%DIR%" --output-dir "%OUT%" --format both
    if errorlevel 1 exit /b %ERRORLEVEL%
    python3 report_vulnerability_counts.py "%DIR%" -o "%REPORT%"
    exit /b %ERRORLEVEL%
)
where python >nul 2>&1 && (
    python standardize_dataset.py "%DIR%" --output-dir "%OUT%" --format both
    if errorlevel 1 exit /b %ERRORLEVEL%
    python report_vulnerability_counts.py "%DIR%" -o "%REPORT%"
    exit /b %ERRORLEVEL%
)
echo Python not found.
exit /b 1
