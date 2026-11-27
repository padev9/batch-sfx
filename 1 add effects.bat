@echo off
setlocal EnableDelayedExpansion
cd /d "%~dp0"
title Batch SFX - ambience + effects for a whole video

REM ===========================================================================
REM  HunyuanVideo-Foley over a whole video.
REM
REM  Double-click this file (or drop a video on it). ComfyUI must be running.
REM  Two questions: the video, and the sound.
REM    - type a sentence  -> that sentence is used for every window
REM    - press Enter      -> Ollama watches two frames per window and writes
REM                          one prompt per window: constant ambience plus an
REM                          accent on the big moments.
REM  Everything else is a setting in foley_config.txt.
REM ===========================================================================

set "PY=python"
%PY% --version >nul 2>&1
if errorlevel 1 set "PY=py -3"

set "VID=%~1"
if not defined VID (
  echo.
  echo Drag the video file into this window, then press Enter.
  echo ^(or paste its path^)
  echo.
  set /p "VID=Video: "
)
if not defined VID goto :novideo

REM  Quotes are stripped the way the other batch tools on this box do it.
set "VID=!VID:"=!"
if not exist "!VID!" goto :badpath

echo.
echo ------------------------------------------------------------
echo   What should this video sound like?
echo.
echo   Type one sentence ^(it is used for every window^), or just press
echo   Enter and Ollama will watch the video and write one prompt per
echo   window: room ambience all the way through, plus a clear accent
echo   on the big beats.
echo.
echo   settings : foley_config.txt  ^(window, steps, cfg, model, mix^)
echo ------------------------------------------------------------
set "PROMPT="
set /p "PROMPT=Sound: "

echo.
echo ------------------------------------------------------------
echo   effects for : "!VID!"
if not defined PROMPT goto :no_prompt
echo   sound       : "!PROMPT!"
goto :shown
:no_prompt
echo   sound       : written by Ollama from the video
:shown
echo ------------------------------------------------------------
echo.

%PY% foley_batch.py --video "!VID!" --prompt "!PROMPT!"
set "RC=%ERRORLEVEL%"

call :chime
echo.
if not "%RC%"=="0" (
  echo ============================================================
  echo   The run stopped early - the reason is printed above.
  echo   Run this file again with the same video to resume:
  echo   finished windows are reused, only the rest is rendered.
  echo ============================================================
) else (
  echo Done. The result sits next to the original as ^<name^>_foley.mp4
  echo Effects are mixed under the original audio - see MIX= in foley_config.txt
)
echo.
pause
exit /b %RC%

:novideo
echo.
echo No video given - nothing was rendered.
echo.
pause
exit /b 1

:badpath
echo.
echo ERROR: that file does not exist:  "!VID!"
echo.
pause
exit /b 2

REM ---------------------------------------------------------------------------
REM  End-of-run sound: done.mp3 / done.wav next to this file, else a beep.
REM
REM  Bare file names on purpose. This folder (and its parents) contain emoji,
REM  and a full path handed to ffplay/powershell can arrive with those replaced
REM  by "?" - the player then finds nothing and the chime stays silent. The
REM  working directory is already this folder, so the name alone is enough, and
REM  the wav fallback is written here and also addressed by name only.
REM ---------------------------------------------------------------------------
:chime
cd /d "%~dp0"
set "SOUND=done.mp3"
if not exist "%SOUND%" set "SOUND=done.wav"
if not exist "%SOUND%" goto :chime_beep

where ffplay >nul 2>&1
if errorlevel 1 goto :chime_wav
ffplay -nodisp -autoexit -loglevel quiet "%SOUND%" >nul 2>&1
if not errorlevel 1 exit /b 0

:chime_wav
REM  SoundPlayer only understands wav, so let ffmpeg decode the mp3 first.
where ffmpeg >nul 2>&1
if errorlevel 1 goto :chime_beep
set "PLAY=batch_sfx_done.wav"
ffmpeg -y -hide_banner -loglevel error -i "%SOUND%" -ac 2 -ar 44100 "%PLAY%" >nul 2>&1
if not exist "%PLAY%" goto :chime_beep
powershell -NoProfile -Command "(New-Object Media.SoundPlayer '%PLAY%').PlaySync()" >nul 2>&1
del "%PLAY%" >nul 2>&1
exit /b 0

:chime_beep
powershell -NoProfile -Command "[console]::beep(880,120);[console]::beep(1175,180)" >nul 2>&1
exit /b 0
