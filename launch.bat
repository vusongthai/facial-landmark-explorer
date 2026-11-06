@echo off
setlocal
cd /d "%~dp0" || goto :failed

if not exist ".venv\Scripts\python.exe" (
    py -3.11 -m venv .venv 2>nul
    if errorlevel 1 (
        python -c "import sys, struct; assert sys.version_info[:2] == (3, 11) and struct.calcsize('P') == 8" 2>nul
        if errorlevel 1 (
            echo Install 64-bit Python 3.11 and enable Add Python to PATH in the installer.
            goto :failed
        )
        python -m venv .venv || goto :failed
    )
)

fc /b requirements.txt ".venv\installed-requirements.txt" >nul 2>nul
if errorlevel 1 (
    echo Installing dependencies. This may take a few minutes on the first launch.
    ".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :failed
    copy /y requirements.txt ".venv\installed-requirements.txt" >nul || goto :failed
)

echo Open http://localhost:8501 if your browser does not open automatically.
echo Press Ctrl+C in this window to stop the app.
".venv\Scripts\python.exe" -m streamlit run app.py --server.address 127.0.0.1 --server.port 8501 --server.headless false --browser.gatherUsageStats false
if errorlevel 1 goto :failed
exit /b 0

:failed
echo.
echo Launch failed. See README.md for troubleshooting.
pause
exit /b 1
