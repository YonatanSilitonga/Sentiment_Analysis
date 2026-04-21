@echo off
REM Wrapper untuk quick_test.py dengan explicit venv path
setlocal enabledelayedexpansion

cd /d "D:\semester-4-IT Del\Semester VI\UI-UX DESIGN\Scrapping_Data"

echo.
echo ======================================================================
echo Running Quick Test with V4 Model
echo ======================================================================
echo.

REM Run dengan full path ke venv python
".venv\Scripts\python.exe" quick_test.py

echo.
echo ======================================================================
echo Test Complete. Press any key to exit...
echo ======================================================================
pause >nul
