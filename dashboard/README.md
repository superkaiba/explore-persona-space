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
uv run python ../scripts/verify_served_dashboards.py   # standing integrity check — last step of EVERY deploy
```

## Served-bytes integrity (`no-transform`)

Incident (task #2365): Cloudflare's Email Address Obfuscation rewrote
email-like strings inside static artifacts under `public/` — including text
presented to the reader as model generations — so the served bytes differed
from the committed bytes (`__cf_email__` spans, +386 bytes on the #2329
gallery). An undisclosed substitution reads as the model's own words, and it
breaks any sha/size integrity comparison against the committed file.

The origin-side fix: `next.config.ts` `headers()` sets
`Cache-Control: public, max-age=0, no-transform` on every `*.html` path
(nested paths included). Cloudflare documents `no-transform` as an exception
its HTML-rewriting features respect. Caveats:

- `no-transform` also bars intermediary compression (e.g. Cloudflare
  re-compressing responses in transit) — accepted cost; these artifacts are
  correctness-critical evidence.
- `no-transform` blocks EVERY Cloudflare edge transform on these responses,
  including the Web Analytics beacon auto-injection (`beacon.min.js`) — the
  probe's #2365 negative control found that injection on all 29 served
  `.html` artifacts (+~360 bytes each, client-conditional), a second silent
  transform. Consequence: Cloudflare Web Analytics stops recording visits to
  the static artifacts once the header is live.
- Scope is the static `.html` artifact class only. App-rendered routes
  (`/tasks/[id]` etc.) remain exposed to CDN transforms until the zone-side
  Email Address Obfuscation toggle is turned OFF (Cloudflare dashboard →
  Scrape Shield; needs the zone owner's access — no zone-settings credential
  exists on this VM).
- If the header is ever verified present but Cloudflare still rewrites, the
  content-level fallbacks are `<!--email_off-->…<!--/email_off-->` fences and
  entity-encoding `@` as `&#64;` in the builders. Treat the origin-side path
  as dead only when BOTH content-level mechanisms are proven ineffective —
  not just the `email_off` fence; at that point the zone-side toggle is the
  only remaining fix.

Standing check: `scripts/verify_served_dashboards.py` fetches every
git-tracked `public/` `.html`/`.json` artifact, compares sha256(served) vs
sha256(on-disk), and counts obfuscation markers in the served bytes — nonzero
markers or any sha divergence exits non-zero. Run it as the last step of
every dashboard deploy (see the deploy block above); use
`--base-url http://127.0.0.1:3010` for an origin-only (pre-CDN) check.

## Auth

Single shared site password set via `SITE_PASSWORD` in `.env.local`.
Submitting the right password mints the `eps_session` cookie, which
gates the protected routes (`proxy.ts`) and unlocks `isEditorAuthed()`
for write actions.
