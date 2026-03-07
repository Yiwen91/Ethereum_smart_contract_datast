@echo off
REM Run labeling tests. Uses py, python3, or python (whichever is available).
cd /d "%~dp0"
set SCRIPT=test_labeling.py

where py >nul 2>&1
if %ERRORLEVEL% equ 0 (
    py "%SCRIPT%"
    exit /b %ERRORLEVEL%
)
where python3 >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python3 "%SCRIPT%"
    exit /b %ERRORLEVEL%
)
where python >nul 2>&1
if %ERRORLEVEL% equ 0 (
    python "%SCRIPT%"
    exit /b %ERRORLEVEL%
)

echo Python not found. Try: py %SCRIPT%   or   python3 %SCRIPT%   or   python %SCRIPT%
echo Or install Python from https://www.python.org/downloads/
exit /b 1
