@echo off
cd /d "%~dp0"
set DIR=smartbugs_wild\contracts
set OUT=standardized_smartbugs
set REPORT=smartbugs_vulnerability_report.txt
set SPLIT_OUT=experiment_splits\smartbugs_secondary

echo Processing SmartBugs Wild secondary dataset: %DIR%
where py >nul 2>&1 && (
    py standardize_dataset.py "%DIR%" --output-dir "%OUT%" --format both --fallback-only --no-validate --no-dedup
    if errorlevel 1 exit /b %ERRORLEVEL%
    py report_vulnerability_counts.py --from-json "%OUT%\standardized_dataset.json" -o "%REPORT%"
    if errorlevel 1 exit /b %ERRORLEVEL%
    py prepare_experiment_splits.py --from-json "%OUT%\standardized_dataset.json" --output-dir "%SPLIT_OUT%"
    exit /b %ERRORLEVEL%
)
where python3 >nul 2>&1 && (
    python3 standardize_dataset.py "%DIR%" --output-dir "%OUT%" --format both --fallback-only --no-validate --no-dedup
    if errorlevel 1 exit /b %ERRORLEVEL%
    python3 report_vulnerability_counts.py --from-json "%OUT%\standardized_dataset.json" -o "%REPORT%"
    if errorlevel 1 exit /b %ERRORLEVEL%
    python3 prepare_experiment_splits.py --from-json "%OUT%\standardized_dataset.json" --output-dir "%SPLIT_OUT%"
    exit /b %ERRORLEVEL%
)
where python >nul 2>&1 && (
    python standardize_dataset.py "%DIR%" --output-dir "%OUT%" --format both --fallback-only --no-validate --no-dedup
    if errorlevel 1 exit /b %ERRORLEVEL%
    python report_vulnerability_counts.py --from-json "%OUT%\standardized_dataset.json" -o "%REPORT%"
    if errorlevel 1 exit /b %ERRORLEVEL%
    python prepare_experiment_splits.py --from-json "%OUT%\standardized_dataset.json" --output-dir "%SPLIT_OUT%"
    exit /b %ERRORLEVEL%
)
echo Python not found.
exit /b 1
