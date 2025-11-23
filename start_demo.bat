@echo off
REM Start server with demo mode enabled (Windows)

cd /d "%~dp0"

echo Starting server with DEMO_MODE=true...
echo Server will be available at http://localhost:8000
echo.

REM Activate virtual environment if it exists
if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

REM Set demo mode
set DEMO_MODE=true

REM Start the server
python run_server.py

pause

