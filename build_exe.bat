@echo off
echo ========================================================
echo   SkyGuard AI — Portable PyInstaller Build Script
echo   Ministry of Earth Sciences / IMD
echo ========================================================
echo.

REM Locate Python executable (check override, then system PATH)
if defined PYTHON (
    if not exist "%PYTHON%" (
        set PYTHON=python
    )
) else (
    set PYTHON=python
)

echo Using Python: %PYTHON%
%PYTHON% --version
if errorlevel 1 (
    echo Error: Python not found on PATH or in specified PYTHON variable.
    pause
    exit /b 1
)

echo Installing dependencies...
%PYTHON% -m pip install -r requirements.txt
%PYTHON% -m pip install pyinstaller

echo.
echo Packaging SkyGuardAI into standalone offline executable...

set ASSET_FLAG=
if exist "assets" (
    set ASSET_FLAG=--add-data "assets;assets"
)

%PYTHON% -m PyInstaller --onefile --windowed --name SkyGuardAI ^
  --hidden-import sklearn.ensemble._forest ^
  --hidden-import sklearn.neighbors._base ^
  --collect-all sklearn --collect-all matplotlib --collect-all scipy ^
  %ASSET_FLAG% ^
  main.py

if errorlevel 1 (
    echo.
    echo [ERROR] PyInstaller build failed. Check logs above.
    pause
    exit /b 1
)

echo.
echo ========================================================
echo Done: dist\SkyGuardAI.exe
echo You can run it directly — fully offline, zero dependencies!
echo ========================================================
pause

