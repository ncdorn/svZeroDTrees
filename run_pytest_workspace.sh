#!/bin/zsh
set -euo pipefail

repo_root="$(cd "$(dirname "$0")" && pwd)"
python_bin="${PYTHON_BIN:-/Users/ndorn/anaconda3/bin/python3}"
solver_root="$repo_root/../svZeroDSolver"

export MPLCONFIGDIR="${MPLCONFIGDIR:-/private/tmp/mplconfig}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/private/tmp}"
export PYTHONPATH="$repo_root/src:$solver_root${PYTHONPATH:+:$PYTHONPATH}"

exec "$python_bin" -m pytest "$@"
