#!/usr/bin/env bash
# Double-click to start the VLBI uvfits viewer (macOS / Linux).
#
# It starts the local server and opens your browser; drag a uvfits file onto
# the page, or click to browse, to load it.  Close this window (Ctrl-C) to stop.
#
# First time only, create the environment:
#   conda env create -f environment.yaml

cd "$(dirname "$0")" || exit 1

# Activate the conda env if conda is available; otherwise fall back to whatever
# python is on PATH (in case the packages are already installed there).
if command -v conda >/dev/null 2>&1; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  if ! conda activate vlbiviewer 2>/dev/null; then
    echo "Could not activate the 'vlbiviewer' conda environment."
    echo "Create it once with:  conda env create -f environment.yaml"
    read -r -p "Press Enter to close. "
    exit 1
  fi
fi

python vlbi_viewer.py "$@"
