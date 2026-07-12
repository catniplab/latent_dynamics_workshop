#!/usr/bin/env bash
# Generate root-level student notebooks from master_notebooks/*.py.
set -euo pipefail
cd "$(dirname "$0")"

master_dir="master_notebooks"

if [ "$#" -gt 0 ]; then
    sources=("$@")
else
    sources=("$master_dir"/*.py)
fi

for src in "${sources[@]}"; do
    case "$src" in
        "$master_dir"/*) ;;
        *) src="$master_dir/$(basename "$src")" ;;
    esac

    name="$(basename "$src")"
    out_py="$name"
    out_ipynb="${name%.py}.ipynb"

    awk '
        /^[[:space:]]*# BEGIN HIDDEN TESTS/ { hidden = 1; next }
        /^[[:space:]]*# END HIDDEN TESTS/ { hidden = 0; next }
        hidden { next }

        /^[[:space:]]*# BEGIN SOLUTION/ {
            match($0, /^[[:space:]]*/)
            indent = substr($0, RSTART, RLENGTH)
            print indent "# YOUR CODE HERE"
            print indent "raise NotImplementedError()"
            solution = 1
            next
        }
        /^[[:space:]]*# END SOLUTION/ { solution = 0; next }
        solution { next }

        { print }
    ' "$src" > "$out_py"

    uv run jupytext --to ipynb --output "$out_ipynb" "$out_py"
done
