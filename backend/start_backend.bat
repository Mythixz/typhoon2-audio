@echo off
echo Starting Modern TTS/STT Backend...
echo.

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install/upgrade dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

REM Create necessary directories
if not exist "temp" mkdir temp
if not exist "temp\audio" mkdir temp\audio
if not exist "static" mkdir static
if not exist "static\audio" mkdir static\audio

REM Start the backend
echo.
echo Starting FastAPI server...
echo Backend will be available at: http://localhost:8000
echo.
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

pause
