#!/usr/bin/env bash
# Build the paper library. Creates utils/.venv on first run.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

VENV="$ROOT/utils/.venv"
PY="$VENV/bin/python"
STAMP="$VENV/.deps-stamp"
REQ="$ROOT/utils/requirements.txt"

if [ ! -x "$PY" ]; then
  echo "→ 创建 Python 虚拟环境 utils/.venv"
  if command -v uv >/dev/null 2>&1; then
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/utils/.uv-cache}"
    uv venv "$VENV" --python 3
    uv pip install --python "$PY" -r "$REQ"
  else
    python3 -m venv "$VENV"
    "$PY" -m ensurepip --upgrade >/dev/null
    "$PY" -m pip install -r "$REQ"
  fi
  touch "$STAMP"
elif [ ! -f "$STAMP" ] || [ "$REQ" -nt "$STAMP" ]; then
  echo "→ 更新构建依赖"
  if command -v uv >/dev/null 2>&1; then
    export UV_CACHE_DIR="${UV_CACHE_DIR:-$ROOT/utils/.uv-cache}"
    uv pip install --python "$PY" -r "$REQ"
  else
    "$PY" -m pip install -r "$REQ"
  fi
  touch "$STAMP"
fi

exec "$PY" "$ROOT/utils/build.py" "$@"
