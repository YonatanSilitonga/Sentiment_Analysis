@echo off
REM Wrapper untuk test_interactive.py dengan explicit venv path
setlocal enabledelayedexpansion

cd /d "D:\semester-4-IT Del\Semester VI\UI-UX DESIGN\Scrapping_Data"

echo Starting Sentiment Model Analyzer...
echo.

REM Run dengan full path ke venv python
".venv\Scripts\python.exe" test_interactive.py

pause
