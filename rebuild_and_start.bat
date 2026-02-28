@echo off
setlocal enabledelayedexpansion

echo.
echo ========================================
echo    Tabletop Agent Engine
echo    Rebuild Vector Store + Start
echo ========================================
echo.

REM Check Python
echo [1/6] Checking Python...
where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python not found in PATH
    pause
    exit /b 1
)
python --version

REM Activate venv
echo.
echo [2/6] Activating virtual environment...
if not exist "venv\Scripts\activate.bat" (
    echo Creating virtual environment...
    python -m venv venv
)
call venv\Scripts\activate.bat

REM Install deps
echo.
echo [3/6] Checking dependencies...
python -m pip install -r requirements.txt -q

REM Clean old vector store
echo.
echo [4/6] Cleaning old vector store...
if exist "data\vector_store" (
    rmdir /s /q "data\vector_store"
    echo Old vector store removed.
) else (
    echo No existing vector store found.
)

REM Initialize ALL rulebooks
echo.
echo [5/6] Initializing ALL rulebooks (this may take a while)...
echo.
echo Found rulebooks:
dir /b data\rules\*.pdf
echo.

python init_all_rules.py --clean --input data/rules/1.pdf data/rules/2.pdf data/rules/3.pdf data/rules/4.pdf data/rules/5.pdf data/rules/6.pdf data/rules/7.pdf

if errorlevel 1 (
    echo [ERROR] Failed to initialize rulebooks
    pause
    exit /b 1
)

REM Start server
echo.
echo [6/6] Starting server...
echo.
echo ========================================
echo    All 7 rulebooks loaded!
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
