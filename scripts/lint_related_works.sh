#!/bin/sh
# Resolve paths from the script so this works from any working directory.
set -eu
script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
exec uv run --no-project python "$script_dir/lint_related_works.py" "$@"
