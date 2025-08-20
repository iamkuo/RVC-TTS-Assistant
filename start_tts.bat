@echo off
setlocal

REM Change to the directory of this script
cd /d "%~dp0"

call conda activate tts
echo Using Conda environment: %CONDA_DEFAULT_ENV%
python "TTS_ai_main.py"
set "EXITCODE=%ERRORLEVEL%"

if %EXITCODE% neq 0 (
  echo.
  echo TTS exited with error code %EXITCODE%.
  pause
)

exit /b %EXITCODE%
