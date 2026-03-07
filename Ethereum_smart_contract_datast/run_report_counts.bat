@echo off
cd /d "%~dp0"
set DIR=%~1
if "%DIR%"=="" set DIR=contract_dataset_ethereum
echo Running vulnerability count report on: %DIR%
where py >nul 2>&1 && (py report_vulnerability_counts.py "%DIR%" & exit /b %ERRORLEVEL%)
where python3 >nul 2>&1 && (python3 report_vulnerability_counts.py "%DIR%" & exit /b %ERRORLEVEL%)
where python >nul 2>&1 && (python report_vulnerability_counts.py "%DIR%" & exit /b %ERRORLEVEL%)
echo Python not found.
exit /b 1
