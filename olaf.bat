@echo off
cd /d "%~dp0"

REM Activate virtualenv
call ".venv\Scripts\activate.bat"

REM Run Olaf inside the venv
python run_olaf.py

echo.
echo Press any key to close...
pause >nul