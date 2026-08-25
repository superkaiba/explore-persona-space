---
name: claude-convention-defense-unverified-env-var-fence
description: "Claude PASSes a round-introduced $VAR dependency in a separate-shell fenced block by citing 'pre-existing file-wide convention' without checking the convention's INITIALIZING sites; argv-blind stubs mask it (#2241 r3)"
metadata:
  type: feedback
---

Rule: when a review defends a round-introduced environment-variable dependency
(`"$REPO_ROOT"`, `$WT`, …) inside a fenced template block with "pre-existing
file-wide convention", verify the convention against its INITIALIZING sites,
not its use sites — and check whether the executed-template pins' stubs
validate argv at all.

**Why:** #2241 r3 — round 3 changed a WORKING relative `scripts/task.py` to
`"$REPO_ROOT"/scripts/task.py` inside the Step-5 draft-PR ensure fence to
satisfy round 2's cosmetic idiom-consistency Minor. Claude PASSed citing
"convention at 09-step-5.md:697/:712 + 08-step-4.md's second fence"; on
inspection the real executable-fence convention is IN-FENCE/IN-STEP
initialization (08-step-4.md:18 + the :45–:55 resolve-once prose,
13-step-9.md:3096, 18-step-10d.md:2354/:4498) — 09-step-5.md has ZERO
assignments, no resolve-once instruction, documents "fenced blocks are
separate shells" (:396), and its :697/:712 sites are the same-class latent
anomaly, not a validating precedent. Harness Bash calls share no env, so the
resolver fails deterministically (`/scripts/task.py`, rc≠0) → the new
fail-open arm skips creation at EVERY round entry → behaviorally identical to
the zero-PR bug the task exists to fix. Pins 14/15 could not catch it: the
`uv` PATH stub `case … *) cat "$TITLE_JSON"` ignores argv entirely, so a
green executed-template suite certified nothing about the invocation path.
Codex FAILed on exactly this; the reconcile upheld it.

**How to apply:** (1) A "fails visibly and fail-open" defense is void when
the fail-open endpoint recurring at every entry IS the bug under repair and
the template's own telemetry instructs "skip; proceed". (2) For any round
that rewrites a command's path/idiom inside a fenced block, grep the SAME
file for `VAR=` assignments and a resolve-once instruction; absence in the
step file = uninitialized, regardless of other files. (3) Template-executing
pins certify only what their stubs constrain — an argv-blind process stub
masks path defects; demand argv-recording + path assertion. (4) Sibling
pattern: a cosmetic consistency suggestion from round N becoming round N+1's
functional defect ([[claude-misses-block-contract-conformance-of-round-added-commands]],
same task, r2).
