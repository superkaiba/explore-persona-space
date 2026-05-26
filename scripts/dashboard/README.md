# EPS dashboard hosting

The EPS dashboard (under `dashboard/`) is a Next.js 16 app that reads
and writes the `tasks/` tree on this VM directly. It runs as a systemd
user service behind a Cloudflare named tunnel at
`https://eps.superkaiba.com`.

This directory holds:

- `eps-dashboard.service` — systemd user unit for `next start` on
  port 3010. Loads `EDITOR_SECRET` etc. from `~/.config/eps-dashboard.env`.
- `eps-dashboard-tunnel.service` — systemd user unit for the
  cloudflared named tunnel.
- `run-dashboard.sh` — `npm install && npm run build && npm run start`
  wrapper. systemd doesn't inherit shell PATH, so this pins it.
- `eps-dashboard.env.example` — env-file template. Copy to
  `~/.config/eps-dashboard.env`, fill in `EDITOR_SECRET`.
- `cloudflared-config.example.yml` — tunnel ingress template. Copy to
  `~/.cloudflared/eps-dashboard.yml`, substitute the tunnel UUID.
- `install.sh` — idempotent installer (symlinks units, seeds env,
  enables units). Does NOT start anything until the interactive
  Cloudflare auth steps are done.

## One-time setup

```bash
# 1. From repo root:
./scripts/dashboard/install.sh

# 2. Generate an editor secret and edit the env file:
openssl rand -base64 32                  # copy the output
$EDITOR ~/.config/eps-dashboard.env      # paste into EDITOR_SECRET=...

# 3. Cloudflare auth (opens a browser tab; you sign in once):
cloudflared tunnel login

# 4. Create the named tunnel + the CNAME for eps.superkaiba.com:
cloudflared tunnel create eps-dashboard
cloudflared tunnel route dns eps-dashboard eps.superkaiba.com

# 5. Write the tunnel config (substitute the UUID printed in step 4):
cp scripts/dashboard/cloudflared-config.example.yml \
   ~/.cloudflared/eps-dashboard.yml
$EDITOR ~/.cloudflared/eps-dashboard.yml

# 6. Start the units:
systemctl --user start eps-dashboard.service
systemctl --user start eps-dashboard-tunnel.service

# 7. Verify:
curl -sI https://eps.superkaiba.com
journalctl --user -u eps-dashboard.service -n 50
journalctl --user -u eps-dashboard-tunnel.service -n 50
```

## Day-to-day

```bash
# Pick up a git pull (rebuilds on restart):
systemctl --user restart eps-dashboard.service

# Tail logs:
journalctl --user -u eps-dashboard.service -f

# Stop everything (e.g. for maintenance):
systemctl --user stop eps-dashboard-tunnel.service eps-dashboard.service
```

## Architecture notes

- **Single user.** The `EDITOR_SECRET` cookie gate is the only auth. Set
  the secret long (≥32 chars). Cloudflare Access could sit in front
  later; not needed for now.
- **Frontmatter handling.** The editor saves the BODY portion of
  `body.md`. `scripts/task.py set-body` preserves the YAML frontmatter
  on its own. To change `title`, `tags`, or `classification`, use the
  dedicated CLI subcommands (`set-title`, `add-tag`, `set-clean-result`).
- **No `--snapshot`.** The dashboard's save action does NOT pass
  `--snapshot` to `task.py set-body`. Snapshots (`original-body.md`)
  are reserved for the analyzer's clean-result promotion.
- **flock + git commit.** Every save acquires the same
  `~/.task-workflow/lock` that the CLI uses, then commits one git
  commit per save. Concurrent `/issue <N>` sessions and dashboard
  edits cannot corrupt `body.md`.
- **Latency.** Save click → `task.py set-body` (~300-500 ms) →
  `revalidatePath('/tasks/[id]')` → next navigation shows the
  update. Browsers see the new body in ~1 s.
