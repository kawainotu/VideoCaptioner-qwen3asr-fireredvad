@echo off
setlocal
chcp 65001 >nul 2>&1

set "ORIG_DIR=%CD%"
set "SCRIPT_DIR=%~dp0"
pushd "%SCRIPT_DIR%"

set "PYTHON_EXE=%SCRIPT_DIR%.venv\Scripts\python.exe"
set "LOG_FILE=%SCRIPT_DIR%mimo-startup.log"

rem Check if --check flag was supplied
if /i "%~1"=="--check" goto :do_check
set "CHECK_MODE=0"
for %%A in (%*) do (
    if /i "%%~A"=="--check" set "CHECK_MODE=1"
)
if "%CHECK_MODE%"=="1" goto :do_check

rem ==========================================
rem Normal Launch Mode
rem ==========================================
if not exist "%PYTHON_EXE%" goto :missing_python

echo [INFO] Starting VideoCaptioner...
echo [INFO] Interpreter: "%PYTHON_EXE%"
echo [INFO] Startup log target: "%LOG_FILE%"

"%PYTHON_EXE%" -X utf8 -m videocaptioner.ui.main %* > "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" goto :on_launch_failure

popd
exit /b 0

:on_launch_failure
echo.
echo [ERROR] VideoCaptioner failed to start or exited with error code %EXIT_CODE%.
echo [ERROR] Startup log contents:
echo ------------------------------------------------------------
if exist "%LOG_FILE%" type "%LOG_FILE%"
echo ------------------------------------------------------------
pause
popd
exit /b %EXIT_CODE%

rem ==========================================
rem Non-GUI Dependency and Import Check Mode
rem ==========================================
:do_check
if not exist "%PYTHON_EXE%" goto :missing_python

echo [INFO] Running non-GUI dependency and import check...
echo [INFO] Interpreter: "%PYTHON_EXE%"

"%PYTHON_EXE%" -X utf8 -c "import sys; from videocaptioner.ui.view.main_window import MainWindow; print('[CHECK-OK] Non-GUI import test succeeded: MainWindow and core dependencies loaded.')" > "%LOG_FILE%" 2>&1
set "EXIT_CODE=%ERRORLEVEL%"

if exist "%LOG_FILE%" type "%LOG_FILE%"

if not "%EXIT_CODE%"=="0" goto :on_check_failure

echo [INFO] Non-GUI check passed successfully.
popd
exit /b 0

:on_check_failure
echo.
echo [ERROR] Dependency check failed with code %EXIT_CODE%.
pause
popd
exit /b %EXIT_CODE%

rem ==========================================
rem Missing Python Error Handler
rem ==========================================
:missing_python
echo [ERROR] Python interpreter not found: "%PYTHON_EXE%" > "%LOG_FILE%" 2>&1
type "%LOG_FILE%"
pause
popd
exit /b 1
