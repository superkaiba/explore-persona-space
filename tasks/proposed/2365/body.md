---
title: Cloudflare email obfuscation silently rewrites model-generation text in every
  eps.superkaiba.com dashboard (served bytes != committed bytes)
kind: infra
tags: []
created_at: '2026-08-18T10:10:24Z'
has_clean_result: false
parent_id: 2329
origin_prompt: 'Surfaced during /issue 2329 workflow-v2 while verifying a full-f-mode
  dashboard rebuild reached the served URL: eps.superkaiba.com serves the #2329 gallery
  with 2 Cloudflare __cf_email__ substitutions inside text presented as model generations
  (true stored string no_knowledge_signed_darwin@email.com), and the #2162 gallery
  with 3, while both committed copies have 0.'
workflow: v1
---
---
kind: infra
---

# Cloudflare email obfuscation silently rewrites model-generation text in every eps.superkaiba.com dashboard

Found while verifying the #2329 dashboards actually served the bytes I had just
built. The committed HTML and the served HTML differ, and the difference lands
**inside text presented to the reader as a model generation**.

## Evidence

`docs/issue2329_result0_gallery.html` (committed, and the local Next.js server on
127.0.0.1:3010 serves it byte-identically):

```
...Immediately after deleting your search history and changing your email to
'no_knowledge_signed_darwin@email.com'!"
```

`https://eps.superkaiba.com/issue2329_result0_gallery.html` (same file through
Cloudflare):

```
...Immediately after deleting your search history and changing your email to
'<a href="/cdn-cgi/l/email-protection" class="__cf_email__"
 data-cfemail="2f41407044414058434a4b484a705c...">[email&#160;protected]</a>'!"
```

Marker counts, local vs served: `__cf_email__` 0 -> 2, `cdn-cgi` 0 -> 3;
+386 bytes on a 13.3 MB file. The bank dashboard, which contains no email-like
strings, is byte-identical served vs local — so the rewrite is content-triggered,
not a general transform.

## It is zone-wide, not a #2329 quirk

`https://eps.superkaiba.com/issue2162_result0_gallery.html` serves with **3**
`__cf_email__` markers while its committed copy `docs/issue2162_result0_gallery.html`
has **0**. So this has been silently altering served dashboards for as long as the
zone has had the feature on.

Scope of exposure — served copies under `dashboard/public/` that contain
email-like strings and are therefore subject to the same rewrite:
`failures-2202_p2/p3/p5/p7/p9.html`, `sample500-2202_p1.html`,
`context-extremes-1482.html`, `issue2329_result0_gallery.html` (at least; the
grep was bounded).

## Why this matters beyond cosmetics

The project rule is explicit: whenever text shown as a model generation differs
from the stored raw completion — rename, redaction, or truncation,
**presentation-time or upstream** — the substitution must be disclosed inline,
per passage, wherever the text is presented, dashboards and HTML artifacts
included, because an undisclosed substitution reads as the model's own words
(#1345, the 'ARIA'->'Assistant' incident). This is that failure mode arriving
from a layer nobody inspected: the CDN. A reader auditing the gallery sees
`[email protected]` and can only conclude the model emitted that, when it
actually emitted `no_knowledge_signed_darwin@email.com`.

It also means **served bytes are not a valid integrity check** for these
artifacts: a size or sha256 comparison against the committed file will
mismatch for reasons that have nothing to do with the build being stale. I hit
exactly that confusion in #2329 — a legitimately-deployed rebuild looked
corrupted because the served gallery was +386 bytes.

## Proposed fix (the implementing session should pick; 1 is the real fix)

1. **Turn Email Address Obfuscation OFF for this zone** (Cloudflare dashboard ->
   Scrape Shield -> Email Address Obfuscation, or `email_obfuscation: off` via the
   zone-settings API). This is a research-artifact host, not a public site with
   harvestable contact addresses; the feature buys nothing here and corrupts
   evidence. NOTE: needs Thomas's Cloudflare access — flag it as the one step
   that may need him.
2. **Failing that, defeat the matcher at render time** in the dashboard builders:
   HTML-entity-encode `@` (`&#64;`) in generation text, which Cloudflare's matcher
   does not rewrite, and the browser still renders as `@`. Cheap and local, but it
   is a workaround and must be applied in every builder that renders completions.
3. **Add a served-vs-committed integrity probe** to whatever checks dashboards,
   asserting the served copy matches the committed bytes OR that every difference
   is an enumerated, disclosed CDN transform. That is what would have caught this
   years-of-artifacts ago rather than by accident.

## Acceptance

- A served dashboard's model-generation text is byte-identical to the committed
  HTML, or every divergence is enumerated and disclosed in the artifact itself.
- `curl <served-url> | grep -c __cf_email__` returns 0 for the #2329 and #2162
  galleries.
- The historical artifacts above are re-verified after the fix (no rebuild needed
  if the fix is zone-side — the same bytes simply stop being rewritten).

**Provenance:** surfaced by the #2329 orchestrator (`/issue 2329`, workflow v2)
while verifying that a full-f-mode dashboard rebuild actually reached the served
URL. Detail in #2329 `events.jsonl`.
