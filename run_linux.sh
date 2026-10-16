#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

# Prefer an activated environment, then the project's local environment.
if [[ -n "${VIRTUAL_ENV:-}" ]]; then
    python_bin="$VIRTUAL_ENV/bin/python"
elif [[ -x .venv/bin/python ]]; then
    python_bin=".venv/bin/python"
else
    python_bin="python3"
fi

exec "$python_bin" main.py "$@"
