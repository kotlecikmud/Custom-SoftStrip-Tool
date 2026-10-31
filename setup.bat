@echo off
cls

echo ==============================
echo   Starting update process...
echo ==============================

REM Path to venv python
set VENV_PYTHON=..\general_purpose_venv\Scripts\python.exe

REM Check if venv exists
if not exist "%VENV_PYTHON%" (
echo ERROR: Virtual environment not found!
echo Expected at: %VENV_PYTHON%
pause
exit /b 1
)

echo.
echo Using Python from venv:
"%VENV_PYTHON%" --version

echo.
echo Pulling latest code from Git...
git pull
if errorlevel 1 (
echo ERROR: git pull failed!
)

echo.
echo Upgrading pip...
"%VENV_PYTHON%" -m pip install --upgrade pip

echo.
echo Installing requirements...
"%VENV_PYTHON%" -m pip install -r requirements.txt
if errorlevel 1 (
echo ERROR: Failed to install requirements!
pause
exit /b 1
)

echo.
echo ==============================
echo   Update completed successfully!
echo ==============================

pause
