@echo off
echo.
echo ==================================================
echo  DOCUVERSE AI - REVOLUTIONARY PDF ASSISTANT
echo ==================================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo [ERROR] Python is not installed or not in PATH
    echo [INFO] Please install Python 3.8+ and try again
    pause
    exit /b 1
)

echo [INFO] Python found, checking version...
python -c "import sys; print(f'[INFO] Using Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}')"

REM Check if app.py exists
if not exist "app.py" (
    echo [ERROR] app.py not found!
    echo [INFO] Please run this script from the project directory
    pause
    exit /b 1
)

REM Check if requirements are installed
echo [INFO] Checking dependencies...
python -c "import streamlit, PyPDF2, transformers, torch" >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Some dependencies may be missing
    echo [INFO] Installing requirements...
    pip install -r requirements.txt
    if %errorlevel% neq 0 (
        echo [ERROR] Error installing requirements
        echo [INFO] Try running manually: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

echo [SUCCESS] All dependencies verified!
echo.
echo [INFO] Starting DocuVerse AI...
echo [INFO] The application will open in your default browser
echo [INFO] Local URL: http://localhost:8501
echo.
echo [INFO] To stop the application, press Ctrl+C
echo ==================================================
echo.

REM Start the Streamlit application
streamlit run app.py --server.headless false --server.port 8501 --server.address localhost --browser.gatherUsageStats false

echo.
echo [INFO] Thank you for using DocuVerse AI!
pause
