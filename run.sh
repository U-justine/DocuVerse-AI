#!/bin/bash

# DocuVerse AI - Launcher Script for Unix/Linux/Mac
# © 2025 Justine & Krishna. All Rights Reserved.

echo "=================================================="
echo " DOCUVERSE AI - REVOLUTIONARY PDF ASSISTANT"
echo "=================================================="
echo ""

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Check if Python is installed
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}[ERROR] Python is not installed or not in PATH${NC}"
    echo -e "${YELLOW}[INFO] Please install Python 3.8+ and try again${NC}"
    exit 1
fi

# Use python3 if available, otherwise use python
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
else
    PYTHON_CMD="python"
fi

# Check Python version
PYTHON_VERSION=$($PYTHON_CMD -c 'import sys; print(".".join(map(str, sys.version_info[:2])))')
echo -e "${BLUE}[INFO] Using Python ${PYTHON_VERSION}${NC}"

# Check if app.py exists
if [ ! -f "app.py" ]; then
    echo -e "${RED}[ERROR] app.py not found!${NC}"
    echo -e "${YELLOW}[INFO] Please run this script from the project directory${NC}"
    exit 1
fi

# Check if requirements.txt exists
if [ ! -f "requirements.txt" ]; then
    echo -e "${YELLOW}[WARNING] requirements.txt not found${NC}"
else
    echo -e "${CYAN}[INFO] Checking dependencies...${NC}"

    # Check if streamlit is installed
    if ! $PYTHON_CMD -c "import streamlit" &> /dev/null; then
        echo -e "${YELLOW}[INFO] Installing missing dependencies...${NC}"
        $PYTHON_CMD -m pip install -r requirements.txt

        if [ $? -ne 0 ]; then
            echo -e "${RED}[ERROR] Error installing requirements${NC}"
            echo -e "${YELLOW}[INFO] Try running manually: pip install -r requirements.txt${NC}"
            exit 1
        fi
    fi
fi

# Verify all key dependencies
echo -e "${CYAN}[INFO] Verifying dependencies...${NC}"
$PYTHON_CMD -c "
try:
    import streamlit
    import PyPDF2
    import transformers
    import torch
    print('[SUCCESS] All key dependencies found!')
except ImportError as e:
    print(f'[ERROR] Missing dependency: {e}')
    print('[INFO] Please run: pip install -r requirements.txt')
    exit(1)
" || exit 1

echo ""
echo -e "${GREEN}[INFO] Starting DocuVerse AI...${NC}"
echo -e "${PURPLE}[INFO] The application will open in your default browser${NC}"
echo -e "${CYAN}[INFO] Local URL: http://localhost:8501${NC}"
echo ""
echo -e "${YELLOW}[INFO] To stop the application, press Ctrl+C${NC}"
echo "=================================================="
echo ""

# Make script executable
chmod +x "$0" 2>/dev/null

# Start the Streamlit application
trap 'echo -e "\n\n[INFO] Shutting down DocuVerse AI..."; echo -e "[INFO] Thank you for using DocuVerse AI!"; exit 0' INT

$PYTHON_CMD -m streamlit run app.py \
    --server.headless false \
    --server.port 8501 \
    --server.address localhost \
    --browser.gatherUsageStats false

echo ""
echo -e "${GREEN}[INFO] Thank you for using DocuVerse AI!${NC}"
