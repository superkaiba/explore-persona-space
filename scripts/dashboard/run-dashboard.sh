#!/usr/bin/env bash
# Build + start the EPS dashboard. Invoked by the eps-dashboard.service
# systemd user unit. Idempotent: each restart rebuilds (so a git pull on
# the VM picks up changes on the next `systemctl restart`).

set -euo pipefail

DASHBOARD_DIR="${DASHBOARD_DIR:-/home/thomasjiralerspong/explore-persona-space/dashboard}"
PORT="${PORT:-3010}"

cd "$DASHBOARD_DIR"

# systemd doesn't inherit shell PATH. Pin the binaries we use, including
# ~/.local/bin where `uv` lives (the comment/edit server actions shell out
# to `uv run python scripts/task.py …`, so ENOENT'ing uv at runtime breaks
# saves with `spawn uv ENOENT`).
export PATH="$HOME/.local/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

# `npm ci` would be ideal (uses lockfile), but `npm install` is what's
# already in place and works under the existing package-lock.json.
# --include=dev is required: @tailwindcss/postcss + typescript live in
# devDependencies and the `next build` step needs them. We do NOT pass
# NODE_ENV=production to npm install (we'd skip devDeps); `next build`
# sets NODE_ENV=production internally for the compiled artifact, and
# `next start` runs in production mode regardless.
echo "[dashboard] npm install (--no-audit --no-fund --include=dev)…"
npm install --no-audit --no-fund --include=dev

echo "[dashboard] next build…"
npm run build

echo "[dashboard] next start on port $PORT…"
exec npm run start -- --port "$PORT"
