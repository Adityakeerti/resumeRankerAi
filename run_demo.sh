#!/usr/bin/env bash
# One-command launcher for the TalentBridge React demo.
#   ./run_demo.sh          build the UI if needed, then serve UI + API on http://127.0.0.1:8000
#   ./run_demo.sh --dev    API on :8000 plus Vite dev server with hot reload on http://127.0.0.1:5173
set -euo pipefail
cd "$(dirname "$0")"

PY=".venv/bin/python"
[ -x "$PY" ] || PY="python3"

if [ ! -d web/node_modules ]; then
  echo "Installing UI dependencies ..."
  (cd web && npm ci --no-audit --no-fund)
fi

if [ "${1:-}" = "--dev" ]; then
  "$PY" -m uvicorn api.server:app --host 127.0.0.1 --port 8000 --reload --reload-dir api --reload-dir sandbox --reload-dir src &
  API_PID=$!
  trap 'kill $API_PID 2>/dev/null' EXIT
  (cd web && npx vite --host 127.0.0.1)
else
  if [ ! -f web/dist/index.html ] || [ -n "$(find web/src web/index.html -newer web/dist/index.html -print -quit)" ]; then
    echo "Building UI ..."
    (cd web && npm run build)
  fi
  echo "TalentBridge running at http://127.0.0.1:8000"
  exec "$PY" -m uvicorn api.server:app --host 127.0.0.1 --port 8000
fi
