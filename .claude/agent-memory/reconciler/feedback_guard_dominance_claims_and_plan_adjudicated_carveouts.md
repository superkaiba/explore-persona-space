---
name: guard-dominance-claims-and-plan-adjudicated-carveouts
description: "#2385 r1 — execute the guard-dominance claim before crediting it (checkout rc=0 with empty merge-base); split Codex probe-class bundles leg by leg; a plan-adjudicated carve-out governs the plan's own general invariant"
metadata:
  type: feedback
---

Three discriminators from #2385 code-review r1 (Claude PASS vs Codex FAIL on a
new `git rm` sink in the Step 5a spec-freshness sync). Verdict: **FAIL**, on a
narrower basis than Codex stated.

## 1. A "structurally unreachable behind guard X" defense is a claim to EXECUTE, not read

Claude declined a whole probe class because the arm sits inside
`if git checkout origin/main -- $SAFE_SPECS; then`, "whose success proves
`origin/main` resolves — so `merge-base` / `log` / `status` cannot plausibly
fail at that point." Two scratch-repo probes falsified it in minutes:
**unrelated histories** and a **`--depth 1` shallow graft** both give
`merge-base` rc=1 / `MB=''` **with `checkout` rc=0**, after which
`"$MB"..HEAD` is the single token `..HEAD` = `HEAD ^HEAD` — an empty range,
**rc=0, silent, no diagnostic** — so the payload probe reports "no payload"
and a clean committed branch file is enumerated for deletion.

**Why:** the guard proved a DIFFERENT proposition than the one needed
(`origin/main` resolves ≠ `merge-base` succeeded), and the variable was
captured ~100 lines UPSTREAM of the guard and consumed ~30 lines DOWNSTREAM.
Dominance claims almost always hide this shape.

**How to apply:** when either reviewer credits or dismisses a finding on
"guard G dominates probe P", write down what G's success actually proves,
then try to construct G-success ∧ P-failure. Three minutes in `mktemp -d`
settles it. Bonus check that paid off here: `git rev-parse
--is-shallow-repository` returned **`true`** for the EPS repo root AND the
live worktree, and "no merge base" on these worktrees is already an incident
of record (#613, cited in my own spec) — so the "exotic" branch was fleet-live
configuration.

## 2. Split a Codex probe-class bundle leg by leg — impact per leg, not per class

Codex bundled four probes into two Criticals. Measured, only one leg held:
- `MB` unchecked → **real, blocking** (above).
- `git status | grep -q .` conflating error with clean → real mechanism, but
  `git rm` without `-f` REFUSES worktree-modified (`local modifications`) AND
  staged-modified (`changes staged in the index`) files, so the dirt case
  cannot lose data. Codex conceded the guard in its own marker. Standing-only.
- process substitution losing the producer rc → **no safety impact at all**: a
  failed enumeration emits nothing, so the `while` loop runs **0 iterations**
  (measured). Truncation yields FEWER candidates — the safe direction.
  Discarded.

**How to apply:** for each leg, ask "what is the measured blast radius?" — a
fail-open that lands on a downstream refusal is defense-in-depth, and an
empty-producer read is fail-SAFE, not fail-open. Bundling makes an over-broad
ledger BLOCKER; raise a narrowed concern id so the implementer's target is
unambiguous and say in the body which legs are Discarded.

## 3. A plan-adjudicated carve-out governs the plan's own general invariant

Both reviewers correctly found `[ ! -f "$man" ]` fails open on `grep` rc=2
(unreadable) vs rc=1 (non-match). But the APPROVED plan's §4 exit-path trace
item (v) enumerated that exact case and decided it ("distinguishing rc 1 from
rc 2 ... buys nothing here"), so the code implements the plan verbatim. Codex
rated plan adherence `±` on that ground — wrong: disagreeing with an
adjudicated plan decision is not a plan-adherence defect. It stayed a
persisted CONCERN, not the FAIL's basis.

**How to apply:** grep the plan for the specific failure arm before charging a
"violates the plan's stated invariant" blocker. Where a plan states a general
invariant AND a specific carve-out, the carve-out governs. What DOES carry a
blocker is an invariant the plan claims and never carved out — here plan §9's
"**three independent** keeps" was measurably TWO (the family-grain and
per-file payload probes consume the same `$MB`), and the shipped comment's
"every probe hit KEEPS the file" was false under an empty `MB`.

## 4. Pre-existing code, round-owned consequence

`MB=$(... merge-base ...)` is byte-identical to `origin/main` and untouched by
the diff. Still upheld: before the round an empty `MB` meant an additive-only
checkout; after it, the same state means `git rm`, committed under an anchor
subject the pass-1 exclusion then filters — silent and self-concealing. That
is the [[feedback_claude_misses_besteffort_upload_made_loadbearing]] "made
load-bearing" shape, not the
[[feedback_codex_litigates_pre_existing_in_round_n]] /
[[feedback_codex_fails_preexisting_resume_metadata_clobber]] shape.
**Discriminator:** did the round change the CONSEQUENCE (additive → destructive)?
If yes, the round owns it. If the consequence is unchanged and zero lines are
round-introduced, it is Codex overreach.
