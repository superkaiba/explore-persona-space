# EPS Dashboard

Next.js viewer for the `tasks/` workflow tree. Read-mostly at
<https://eps.superkaiba.com>, with editor-gated write actions for
task bodies, comments, and clean-result mutations.

## Local development

```bash
npm run dev          # serves on http://localhost:3000
```

## Production

The production instance runs as a systemd unit
(`eps-dashboard.service`) listening on `:3010`, exposed publicly via a
Cloudflare named tunnel (`eps-dashboard-tunnel.service`) at
`https://eps.superkaiba.com`.

Deploy after a code change:

```bash
cd /home/thomasjiralerspong/explore-persona-space/dashboard
npm exec -- next build
sudo systemctl restart eps-dashboard.service
```

## Auth

Single shared site password set via `SITE_PASSWORD` in `.env.local`.
Submitting the right password mints the `eps_session` cookie, which
gates the protected routes (`proxy.ts`) and unlocks `isEditorAuthed()`
for write actions.
