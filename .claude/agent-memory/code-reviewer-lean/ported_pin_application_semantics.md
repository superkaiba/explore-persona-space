---
name: ported-pin-application-semantics
description: Verbatim-ported constants can hide inverted APPLICATION semantics (setdefault vs authoritative env pin) and dropped row-level guards (think-leak classifier) — grep the SOURCE's injection call site and the classifiers around its generation loop, not just the constant dicts (#2502 R1 g2)
metadata:
  type: feedback
---

When a diff claims constants are "ported verbatim" from a sibling rig, matching the constant DICTS certifies nothing about behavior. Two drops found in one commit (#2502 R1 g2, port of #2378 gen/capture):

1. **Application semantics inverted.** The source injected `LAUNCH_ENV_PINS` AUTHORITATIVELY (comment: "the pin always wins — an inherited =1 would deterministically crash engine init"); the port used `os.environ.setdefault`, so a pre-exported env value defeats the pin and reproduces the exact crash the pin exists to prevent. The constant was byte-identical; the injection site was not.
2. **Row-level guard dropped.** The source's production generation loop classified and dropped `<think>`-carrying answers (`think_leak` — a plan literal in the source's own review); the port rendered `enable_thinking=False` but never inspected completions, leaving the "thinking-disabled" arm's answer-span capture open to unmeasured CoT contamination.

**Why:** a port is a PATTERN copy — the pattern includes how pins are applied and what the source classifies/drops per row, both of which live in comments and adjacent functions the constant-grep never touches.

**How to apply:** on any "ported from #M" commit, open the source at the pinned provenance SHA and (a) read the APPLICATION site of every ported pin (authoritative set vs setdefault vs assert), (b) sweep the source file for row-level classifiers/drop guards around the ported loop (`grep -n "drop_reason\|keep\|bad_words\|classify"`), and (c) treat any omission as a porting-fidelity finding even when the plan text doesn't name it. Related: [[inherited-byte-verbatim-claims-nothing]], [[verbatim-port-commit-review-recipe]]. Sibling seam in the same commit: presence-only HF resume-skip bypassing the local ledger's regime gate cross-pod — see [[new-dial-missing-from-resume-regime]] / [[presence-redrive-blesses-stale-mirror]]; the new twist was the ledger's own "use a fresh out-root" error message steering the operator INTO the unprotected cross-pod path while the plan PRE-REGISTERS the param-changed rerun (cap-hit regen).
