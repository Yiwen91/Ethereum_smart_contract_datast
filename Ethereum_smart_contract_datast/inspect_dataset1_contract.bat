@echo off
cd /d "%~dp0"
if "%~1"=="" (
    echo Usage: inspect_dataset1_contract.bat contract_dataset_ethereum\path\to\file.sol
    echo Example: inspect_dataset1_contract.bat contract_dataset_ethereum\contract1\0.sol
    exit /b 1
)
call run_report.bat "%~1"
