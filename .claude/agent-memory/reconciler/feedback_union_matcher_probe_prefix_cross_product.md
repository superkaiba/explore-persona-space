---
name: union-matcher-probe-prefix-cross-product
description: Union disarm (lead matcher + whole-record screen) — probe legal-prefix × family CROSS PRODUCT for families present ONLY in the lead arm; Claude's 48-probe set missed assignment-prefixed dot-source (#2357 r4)
metadata:
  type: feedback
---

When a guard's disarm/recognition is a UNION of (a) a `^`-anchored per-record
LEAD matcher and (b) a whole-record SCREEN, any token family present ONLY in
the lead arm (deliberately excluded from the screen — e.g. the lone-dot
dot-source spelling, excluded so blanket-add-dot commits survive) has NO
fallback when a legal shell prefix (assignment `VAR=v `, `time`, wrapper word)
defeats the `^` anchor. Probe the CROSS PRODUCT: every legal prefix × every
lead-only family.

**Why:** #2357 r4 — Claude's code-reviewer ran 48 executed probes including
assignment-prefixed `cd` (caught by the screen's mid-record cd alternative)
and PASSed; it never composed assignment-prefix × dot-source, the ONE cell
with no screen redundancy. Codex's static trace was right: my executed
differential gave round-4 rc=0 (permit) vs origin/main rc=2 (block) on
`cd <root> && FOO=bar . scripts/env.sh && git commit -m x -- own.py`, while
the plain dot-source negative control blocked (rc=2) and prefixed `source`
blocked via the screen. A prefix-probe PASS on screen-covered families says
NOTHING about lead-only families.

**How to apply:** In any guard-hook reconcile where one reviewer claims a
prefix/spelling escape: (1) read the screen ERE and list which lead families
it does NOT contain (the guard's own comments often name the deliberate
exclusion); (2) execute the escape shape for exactly those families on both
blobs (the c30 fixture pattern in tests/test_guard_root_code_commit.py:
foreign staged gated file + root-subdir cwd + cwd-relative pathspec + armed
canonical chain); (3) include the screen-covered sibling (e.g. prefixed
`source`) as a mechanism control. Related: [[claude-misses-comment-tail-spoof-on-rawscan-guards]].
