@echo off
REM Double-click to start the VLBI uvfits viewer (Windows).
REM
REM It starts the local server and opens your browser; drag a uvfits file onto
REM the page, or click to browse, to load it.  Close this window to stop.
REM
REM First time only, create the environment:
REM   conda env create -f environment.yaml

cd /d "%~dp0"

call conda activate vlbiviewer 2>nul
if errorlevel 1 (
  echo Could not activate the 'vlbiviewer' conda environment.
  echo Create it once with:  conda env create -f environment.yaml
  pause
  exit /b 1
)

python vlbi_viewer.py %*
pause
