---
name: tmp-script-dir-shadows-stdlib
description: Ad-hoc scripts run from bare /tmp import rogue /tmp modules (sys.path[0] = script dir) — run from /tmp/<slug>/ subdirs
metadata:
  type: feedback
---

Never run an ad-hoc python script from bare `/tmp` on this shared VM — always
place it in its own subdirectory (`/tmp/issue-<N>-<slug>/probe.py`).

**Why:** Python puts the SCRIPT'S directory at `sys.path[0]`, so any stray
module-named file another session left in `/tmp` shadows stdlib/site-packages.
#2546 r7 (2026-08-25): a rogue `/tmp/six.py` (someone's PIL image script)
shadowed the `six` package → `dateutil → pandas → datasets` import chain died
with a bizarre `FileNotFoundError: test/ui_baselines/....png` deep inside
pandas. The traceback points nowhere near the real cause; the tell is a
third-party import failing inside a file path under `/tmp`.

**How to apply:** compose census/probe/verification scripts at
`/tmp/<slug>/x.py`, never `/tmp/x.py`. If an unrelated-looking import error
mentions a `/tmp/<module>.py` path in the traceback, suspect shadowing first.
