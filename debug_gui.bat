@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

set "VIDEOCAPTIONER_DEBUG=1"
set "PYTHONFAULTHANDLER=1"
set "PYTHONUNBUFFERED=1"

where uv >nul 2>&1
if errorlevel 1 (
    echo [ERROR] uv was not found.
    echo Install it from https://docs.astral.sh/uv/ and run this file again.
    pause
    exit /b 1
)

echo [INFO] Preparing the locked development environment...
uv sync --locked
if errorlevel 1 (
    echo.
    echo [ERROR] Dependency setup failed. See the output above.
    pause
    exit /b 1
)

echo.
echo [INFO] Starting VideoCaptioner in debug mode...
echo [INFO] Close the GUI to stop this process.
echo.

uv run --locked python -X dev -m videocaptioner.ui.main
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo [ERROR] VideoCaptioner exited with code %EXIT_CODE%.
    pause
)

exit /b %EXIT_CODE%
