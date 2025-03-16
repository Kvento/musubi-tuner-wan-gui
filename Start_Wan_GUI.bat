@echo off
setlocal

:: Specify the path to your Python script
set SCRIPT_PATH=wan_lora_trainer_gui.py

:: Check if Python is installed
echo Checking for Python...
python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo Python not found. Automatic installation is not possible via bat file.
    echo Please install Python manually from the official website: https://www.python.org/
    pause
    exit /b 1
)

:: Check for pip (tool for installing Python packages)
echo Checking for pip...
python -m ensurepip >nul 2>&1
python -m pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo pip not found. Installing pip...
    python -m ensurepip --upgrade
    python -m pip install --upgrade pip
    if %errorlevel% neq 0 (
        echo Failed to install pip. Please check your Python installation.
        pause
        exit /b 1
    )
)

:: Check for tkinter
echo Checking for tkinter...
python -c "import tkinter" >nul 2>&1
if %errorlevel% neq 0 (
    echo tkinter module not found. Attempting to install...
    python -m pip install tk
    if %errorlevel% neq 0 (
        echo Failed to install tkinter. There might be an issue with permissions.
        pause
        exit /b 1
    )
)

:: Run the script
echo All dependencies are installed. Running the script...
start /min python %SCRIPT_PATH%
if %errorlevel% neq 0 (
    echo An error occurred while running the script.
    pause
    exit /b 1
)

echo Script executed successfully.