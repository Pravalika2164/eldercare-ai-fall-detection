@echo off
cd /d "%~dp0"

echo ==========================================
echo     Elderly Fall Detection System
echo ==========================================
echo.

if not exist ".venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found.
    echo Please create .venv and install requirements.
    pause
    exit /b 1
)

echo Starting monitoring...
echo Press Q in the camera window to stop.
echo.

".venv\Scripts\python.exe" app.py --use-ml

echo.
echo Monitoring stopped.
pause