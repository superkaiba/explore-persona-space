---
name: Empirically ground every workflow_lint ref-detection regex
description: When planning a new workflow_lint.py ref/token check, RUN the candidate regex over the actual DOC_FILES before approving it — candidate sketches are wildly false-positive-prone
type: feedback
---

When planning a new `scripts/workflow_lint.py` reference/token-detection check
(`--check-skill-refs`, `--check-script-refs`-style), do NOT trust the
candidate-block's sketched regex. RUN it over the actual `DOC_FILES` and count
distinct tokens before writing the plan.

**Why:** On #714 the candidate's `r"/([a-z][a-z0-9-]+)"` matched **317** distinct
tokens across the 3 DOC_FILES — almost all path segments (`/tmp`, `/main`,
`/task`, `/scripts`), not skill refs. A plan that shipped that regex would FAIL on
`main` instantly and need ~300 opt-outs. The fix (left-anchor to start-of-token +
right-boundary excluding `\w./:-` + **fenced-block + inline-backtick exclusion**)
cut it to 11 (7 live skills, 4 handled by allowlist + 3 opt-outs).

**How to apply:**
- Write a ~15-line throwaway scan: candidate regex + `INLINE=re.compile(r"`[^`]*`")`
  span-strip + a fenced-block `in_fence` toggle, walk DOC_FILES, `Counter` the
  matches, partition into live-set vs dangling.
- The dominant false-positive killers, in order: (1) left-anchor the slash to
  start-of-token (`(?:(?<=^)|(?<=[\s(\[{]))`), (2) exclude inline-backtick spans +
  fenced code blocks (refs live in PROSE; paths live in code), (3) right-boundary
  reject `\w./:-` so a leaf token isn't a path-head fragment.
- Re-use the in-file `HISTORICAL_REF_OPT_OUT` (`<!-- lint: historical-ref -->`)
  per-line opt-out — don't invent a new one. Genuine prose FPs (a hyphenated
  English noun after a slash, e.g. `/workspace-contract` lane) get back-ticked or
  opt-outed; legit-but-external skills (global `humanize`, plugin-namespaced
  `codex:rescue`) go in an allowlist FILE (one-name-per-line + `#` comments,
  mirrors `tests/sparse_cones.txt`).
- Put the scan command + its expected output INTO the plan (§Empirical grounding)
  so the implementer re-runs it and annotates exactly the live-on-`main` dangling
  set — no more, no fewer.
- Set a kill/pivot threshold: if FPs needing opt-outs exceed ~5 after reasonable
  exclusions, pivot from exclusion-list/opt-OUT to inclusion-list/opt-IN.
- Resolve the live set against `.claude/skills/*/` DIR names (Goal-verbatim), NOT
  `*/SKILL.md` — `clean-results` has SPEC.md and no SKILL.md, so the glob form
  silently drops it.
