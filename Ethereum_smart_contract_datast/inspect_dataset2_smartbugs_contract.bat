@echo off
cd /d "%~dp0"
if "%~1"=="" (
    echo Usage: inspect_dataset2_smartbugs_contract.bat smartbugs_wild\contracts\path\to\file.sol
    echo Example: inspect_dataset2_smartbugs_contract.bat smartbugs_wild\contracts\0x0000000000000000000000000000000000000000.sol
    exit /b 1
)
call run_report.bat "%~1"
