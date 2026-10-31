@echo off
cls

echo trwa wczytywanie...

REM Path to venv python
set VENV_PYTHON=..\general_purpose_venv\Scripts\python.exe

REM Check if venv exists
if not exist "%VENV_PYTHON%" (
echo ERROR: Virtual environment not found!
echo Run setup.bat first.
pause
exit /b 1
)

REM Run app using venv
"%VENV_PYTHON%" cauzin_softstrip.py

pause
