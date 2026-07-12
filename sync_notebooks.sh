#!/usr/bin/env bash
# Sync every paired notebook (.py <-> .ipynb) via jupytext.
# jupytext reads each file's `formats: ipynb,py:percent` metadata and updates
# whichever side of the pair is older. Pass file(s) to sync only those; with no
# argument, sync all notebooks in the repo root.
set -euo pipefail
cd "$(dirname "$0")"

if [ "$#" -gt 0 ]; then
    uv run jupytext --sync "$@"
else
    uv run jupytext --sync ./*.ipynb
fi
