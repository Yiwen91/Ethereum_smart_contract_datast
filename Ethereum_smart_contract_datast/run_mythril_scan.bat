@echo off
cd /d "%~dp0"
set DIR=%~1
if "%DIR%"=="" set DIR=contract_dataset_ethereum
echo Scanning %DIR% with Mythril...
where py >nul 2>&1 && (py mythril_scan.py "%DIR%" --output-dir mythril_scan_output --copy-staging & exit /b %ERRORLEVEL%)
where python3 >nul 2>&1 && (python3 mythril_scan.py "%DIR%" --output-dir mythril_scan_output --copy-staging & exit /b %ERRORLEVEL%)
where python >nul 2>&1 && (python mythril_scan.py "%DIR%" --output-dir mythril_scan_output --copy-staging & exit /b %ERRORLEVEL%)
echo Python not found.
exit /b 1
