@echo off
REM BFRBye launcher — clears PYTHONPATH to avoid conflicts with Hermes Agent packages
set PYTHONPATH=
cd /d "%~dp0"
.venv\Scripts\python.exe -m bfrbye %*
echo.
echo Script finished (exit code=%errorlevel%)
pause
