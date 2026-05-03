#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="$ROOT/.venv/bin/python"
STREAMLIT="$ROOT/.venv/bin/streamlit"

if [[ ! -x "$PYTHON" || ! -x "$STREAMLIT" ]]; then
  echo "Missing .venv. Create it and install requirements first:" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install -r requirements.txt -r requirements-gpu.txt" >&2
  exit 1
fi

cleanup() {
  if [[ -n "${UI_PID:-}" ]]; then
    kill "$UI_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

"$STREAMLIT" run "$ROOT/app.py" --server.address 127.0.0.1 --server.port 8501 >/tmp/digital_comm_streamlit.log 2>&1 &
UI_PID=$!

sleep 2
if command -v xdg-open >/dev/null 2>&1; then
  xdg-open http://127.0.0.1:8501 >/dev/null 2>&1 || true
fi

"$PYTHON" "$ROOT/scripts/verify_parameter_grid.py"

wait "$UI_PID"
