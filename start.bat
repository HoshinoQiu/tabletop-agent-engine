@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================
echo    Tabletop Agent Engine - Starting
echo ========================================
echo.

REM Check Python
echo [1/5] Checking Python...
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
    pause
    exit /b 1
)
python --version

REM Create venv
echo.
echo [2/5] Checking virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
    if errorlevel 1 (
        echo [ERROR] Failed to create venv
        pause
        exit /b 1
    )
)
echo Virtual environment OK

REM Activate venv
echo.
echo [3/5] Activating virtual environment...
call venv\Scripts\activate.bat

REM Install deps
echo.
echo [4/5] Installing dependencies (please wait)...
python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install dependencies
    pause
    exit /b 1
)
echo Dependencies OK

REM Check vector store
echo.
echo [5/5] Checking vector store...
if not exist "data\vector_store\index.faiss" (
    echo [WARNING] Vector store is empty!
    echo Please run: rebuild_and_start.bat
    echo Or manually: python init_all_rules.py --clean --input data/rules/*.pdf
    pause
    exit /b 1
)
echo Vector store OK

REM Start server
echo.
echo ========================================
echo    Server Starting!
echo ========================================
echo.
echo Open browser:  http://localhost:8000
echo API docs:      http://localhost:8000/docs
echo.
echo Press Ctrl+C to stop
echo ========================================
echo.

python -m uvicorn api.main:app --reload --host 0.0.0.0 --port 8000

echo.
echo Server stopped.
pause
