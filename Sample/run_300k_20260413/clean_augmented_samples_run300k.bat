@echo off
setlocal

set SCRIPT_DIR=%~dp0
set PY_SCRIPT=%SCRIPT_DIR%clean_augmented_samples.py
set TARGET_DIR=%SCRIPT_DIR%Sample\run_300k_20260413

if not exist "%PY_SCRIPT%" (
  echo [ERROR] Missing script: %PY_SCRIPT%
  pause
  exit /b 2
)

if not exist "%TARGET_DIR%" (
  echo [ERROR] Missing target dir: %TARGET_DIR%
  echo You can run manually, for example:
  echo   python clean_augmented_samples.py "Sample\iteration_4"
  pause
  exit /b 2
)

echo [INFO] Cleaning augmented rows under: %TARGET_DIR%
python "%PY_SCRIPT%" "%TARGET_DIR%" %*
set EXIT_CODE=%ERRORLEVEL%

if not "%EXIT_CODE%"=="0" (
  echo [ERROR] Cleanup failed with code %EXIT_CODE%
  pause
  exit /b %EXIT_CODE%
)

echo [DONE] Cleanup finished.
pause
exit /b 0
