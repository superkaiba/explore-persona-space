#!/usr/bin/env bash
# persona.sh — spawn a DETACHED, daemon-hosted PM session and attach THIS
# terminal to it in one shot.
#
# Why: `happy claude` runs the session in the foreground of your SSH TTY, so
# closing the laptop (SIGHUP) kills it. `spawn-pm` instead asks the Happy
# daemon to spawn the session detached — it then lives under the daemon,
# independent of any terminal. This wrapper spawns it and immediately
# `happy resume`s it, so you get a terminal view AND closing the laptop /
# dropping SSH drops only this client; the session keeps running and is
# reachable from the phone or a later `happy resume <id>`.
#
# Usage (typically via the `persona` alias):
#   gcloud compute ssh ... -- -t "zsh -ic /home/thomasjiralerspong/explore-persona-space/scripts/persona.sh"
set -euo pipefail

# uv / happy on PATH even in non-login shells (same fix the cron wrappers need).
export PATH="$HOME/.local/bin:$PATH"
cd "$HOME/explore-persona-space"

out="$(uv run python scripts/spawn_session.py spawn-pm)"
printf '%s\n' "$out"

sid="$(printf '%s\n' "$out" | sed -n 's/^PM session spawned: //p' | tr -d '[:space:]')"
if [ -z "$sid" ]; then
  echo "persona.sh: could not parse PM session id from spawn-pm output" >&2
  exit 1
fi

# Give the daemon a moment to register the new session before attaching.
for _ in $(seq 1 10); do
  if uv run python scripts/spawn_session.py list 2>/dev/null | grep -q "$sid"; then
    break
  fi
  sleep 1
done

echo "persona.sh: attaching terminal to ${sid}."
echo "persona.sh: close / drop SSH anytime — the session survives (reattach: happy resume ${sid})."
echo "persona.sh: type /pm in the session to load the PM persona."
exec happy resume "$sid"
