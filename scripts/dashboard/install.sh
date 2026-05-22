#!/usr/bin/env bash
# Idempotent installer for the EPS dashboard systemd units.
#
# Does what it can without user interaction. Prints the remaining
# (interactive) cloudflared steps at the end.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPTS_DASH="$REPO_ROOT/scripts/dashboard"
SYSTEMD_DIR="$HOME/.config/systemd/user"
ENV_FILE="$HOME/.config/eps-dashboard.env"

echo "[install] Repo root: $REPO_ROOT"
mkdir -p "$SYSTEMD_DIR" "$HOME/.config"

# --- 1) Symlink (not copy) the systemd units, so `git pull` updates them.
for unit in eps-dashboard.service eps-dashboard-tunnel.service; do
  src="$SCRIPTS_DASH/$unit"
  dst="$SYSTEMD_DIR/$unit"
  if [ -L "$dst" ] || [ -f "$dst" ]; then rm -f "$dst"; fi
  ln -s "$src" "$dst"
  echo "[install] linked $dst → $src"
done

# --- 2) Seed the env file if missing. NEVER overwrite a real secret.
if [ ! -f "$ENV_FILE" ]; then
  cp "$SCRIPTS_DASH/eps-dashboard.env.example" "$ENV_FILE"
  chmod 600 "$ENV_FILE"
  echo "[install] seeded $ENV_FILE (template; edit before starting)"
else
  echo "[install] $ENV_FILE already exists; leaving untouched"
fi

# --- 3) Reload + enable. Don't START — the env file may still be a template.
systemctl --user daemon-reload
systemctl --user enable eps-dashboard.service
systemctl --user enable eps-dashboard-tunnel.service
echo "[install] systemd units enabled (not started)"

echo
echo "Next steps (interactive, must be done once by you):"
echo
echo "  1. Generate EDITOR_SECRET and put it in $ENV_FILE:"
echo "       openssl rand -base64 32"
echo
echo "  2. Cloudflare auth (opens a browser window):"
echo "       cloudflared tunnel login"
echo
echo "  3. Create the named tunnel + DNS:"
echo "       cloudflared tunnel create eps-dashboard"
echo "       cloudflared tunnel route dns eps-dashboard eps.superkaiba.com"
echo
echo "  4. Write the tunnel config (substitutes the UUID printed by step 3):"
echo "       cp $SCRIPTS_DASH/cloudflared-config.example.yml ~/.cloudflared/eps-dashboard.yml"
echo "       \$EDITOR ~/.cloudflared/eps-dashboard.yml"
echo
echo "  5. Start everything:"
echo "       systemctl --user start eps-dashboard.service"
echo "       systemctl --user start eps-dashboard-tunnel.service"
echo
echo "  6. Verify:"
echo "       systemctl --user status eps-dashboard.service"
echo "       systemctl --user status eps-dashboard-tunnel.service"
echo "       curl -sI https://eps.superkaiba.com"
echo
echo "Logs:"
echo "       journalctl --user -u eps-dashboard.service -f"
echo "       journalctl --user -u eps-dashboard-tunnel.service -f"
