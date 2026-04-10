#!/bin/bash
# sync-slides.sh — Continuously rsync slides from remote to local clone.
#
# Run on your Mac from anywhere. Syncs outputs/slides/ into your local
# repo clone so you can open the HTML files directly in a browser.
#
# Usage:
#   sync-slides.sh <ssh-host> [interval]   Start syncing (default: every 5s)
#   sync-slides.sh --stop                  Stop a running sync
#   sync-slides.sh --status                Check if sync is running
#
# Examples:
#   ./scripts/sync-slides.sh cluster       # start syncing every 5s
#   ./scripts/sync-slides.sh cluster 10    # sync every 10s
#   ./scripts/sync-slides.sh --stop        # stop the sync

set -euo pipefail

REMOTE_DIR="/workspace-vast/pbb/agentic-backdoor/outputs/slides/"
LOCAL_DIR="/Users/pbb/Research/Project/Agentic Backdoor/Code/outputs/slides/"
PIDFILE="/tmp/sync-slides.pid"

# ── Helpers ──────────────────────────────────────────────────────────

is_running() {
  [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null
}

do_stop() {
  if is_running; then
    local pid
    pid=$(cat "$PIDFILE")
    kill "$pid" 2>/dev/null || true
    # Wait up to 5s for clean exit
    for _ in $(seq 1 50); do
      kill -0 "$pid" 2>/dev/null || break
      sleep 0.1
    done
    # Force kill if still alive
    kill -9 "$pid" 2>/dev/null || true
    rm -f "$PIDFILE"
    echo "Slide sync stopped (was PID $pid)."
  else
    rm -f "$PIDFILE"
    echo "No sync running."
  fi
}

do_status() {
  if is_running; then
    echo "Slide sync running (PID $(cat "$PIDFILE"))."
  else
    rm -f "$PIDFILE"
    echo "No sync running."
  fi
}

cleanup() {
  rm -f "$PIDFILE"
  echo ""
  echo "Slide sync stopped."
  exit 0
}

# ── Dispatch ─────────────────────────────────────────────────────────

case "${1:-}" in
  --stop)   do_stop;   exit 0 ;;
  --status) do_status; exit 0 ;;
  --help|-h|"")
    sed -n '2,/^$/s/^# //p' "$0"
    exit 0 ;;
esac

REMOTE_HOST="$1"
INTERVAL="${2:-5}"

# Refuse to start a second instance
if is_running; then
  echo "Sync already running (PID $(cat "$PIDFILE")). Use --stop first."
  exit 1
fi

mkdir -p "$LOCAL_DIR"

# ── Trap signals ─────────────────────────────────────────────────────
trap cleanup EXIT INT TERM HUP

# Write PID file (this shell's PID — signals here kill the whole thing)
echo $$ > "$PIDFILE"

# ── Initial sync ─────────────────────────────────────────────────────
echo "Syncing slides..."
echo "  ${REMOTE_HOST}:${REMOTE_DIR}"
echo "  → ${LOCAL_DIR}"
rsync -avz --delete -e ssh \
  "${REMOTE_HOST}:${REMOTE_DIR}" "$LOCAL_DIR" || true
echo ""
echo "Slide sync running (PID $$, every ${INTERVAL}s). Stop with:"
echo "  $0 --stop"
echo "  — or Ctrl-C"

# ── Polling loop ─────────────────────────────────────────────────────
while true; do
  sleep "$INTERVAL"
  rsync -az --delete -e ssh \
    "${REMOTE_HOST}:${REMOTE_DIR}" "$LOCAL_DIR" 2>/dev/null || true
done
