---
name: marker-success-command-verbatim-rerun
description: Re-run the impl marker's (c) "what success looks like" command VERBATIM — a pasted command missing a required flag exits non-zero while the phase itself is green; Path(args.x or "") is a dead dir-guard (#2379 R2 g2)
metadata:
  type: feedback
---

Rule: at Step 0.6, execute the marker's success-signal command exactly as
pasted before crediting it — not a corrected/completed form. If the verbatim
form fails, the finding splits: (a) is the PHASE broken (FAIL-class), or
(b) is the pasted command incomplete (present-but-imperfect digest →
CONCERNS + a Minor)? Only re-running verbatim distinguishes them.

**Why:** #2379 R2 g2 — marker (c) claimed `issue2379_analysis.py --smoke`
exits 0; verbatim it exits 1 with a cwd-relative FileNotFoundError because
`--fixtures-root` is required. The designed SystemExit guard was DEAD:
`Path(args.fixtures_root or "")` is `Path(".")`, and `.` is always a dir, so
the omitted-flag case skipped the helpful raise. With the flag, the phase ran
green (rc=0, both legs) — so the right verdict was CONCERNS + Minor, not a
smoke-run-missing FAIL.

**How to apply:** (1) run each (c) command verbatim from the stated cwd;
(2) grep the round's argparse guards for `Path(args.<x> or "")` /
`Path(<maybe-empty>).is_dir()` — empty-string Path resolves to `.` and
defeats existence guards; (3) a green-with-flags rerun downgrades to a
digest-accuracy Minor per Step 0.7 rule 1. Pairs with
[[fails-pre-fix-probe-parent-commit]] (that one certifies test claims; this
one certifies the success-signal command itself).
