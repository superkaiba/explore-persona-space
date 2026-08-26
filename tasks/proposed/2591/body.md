---
title: 'arm-registry: state members<->row-key identity, sharpen the mismatch refusal,
  widen recompute to int-keyed registries'
kind: infra
tags: []
created_at: '2026-08-25T22:17:24Z'
has_clean_result: false
origin_prompt: 'Surfaced by the /issue 2546 orchestrator: epm:smoke-architecture-check
  refused twice (rounds 5, 8) on correct substance — once for a command-literal arm-registry
  line, once for a members<->row-key naming mismatch the refusal message did not name
  as a possibility; separately, the int-keyed ARMS registry can never reach the driver-verified
  tier.'
workflow: v1
---
# arm-registry: state the members↔row-key identity, sharpen the mismatch refusal, and close the int-keyed-registry recompute hole

## Goal

Close three distinct gaps in the `arm-registry:` contract (#2176) that made a conforming-looking marker refuse twice on one task, and that leave a legitimate registry shape permanently un-recomputable.

## Evidence — two refusals on task #2546, rounds 5 and 8

`epm:smoke-architecture-check` on #2546 was refused by `task.py check-smoke-arch-registry` twice for the SAME contract, in two different ways. Neither refusal was a substantive defect: the arm registry was correct (`sorted(ARMS)` -> `[1, 2, 3]`), the per-arm rows covered every member, and `--import-check` returned rc=0 both times. The cost was two review-round bounces, one of which was the SOLE blocker in an otherwise-clean Claude verdict.

**Gap 1 — the implementer-facing template never states the members↔row-key identity.** `.claude/rules/experiment-implementer-section-reference.md` § "Axis 2 — Per-arm resolution attestation" shows:

```
arm-registry: source=sorted(PHASES) file=scripts/issue<N>_<slug>.py n=3 members=<a>,<b>,<c>
per-arm-resolution:
  <arm-name-1>: REAL — <one-line: which real computation ran>
```

`<a>` and `<arm-name-1>` are the same string in the author's head, but the prose never says so, and the checker enforces `set(members) - set(per_arm_row_keys) == {}` byte-wise (`task_workflow.smoke_arch_registry_check` clause 5). On #2546 the registry keys are the integers `1, 2, 3`, so `members=1,2,3` is the honest mechanical derivation — while the per-arm rows carried the human-readable `arm 1:` / `arm 2:` / `arm 3:` labels the task had used since round 1. Result: `registry-enumeration mismatch: n_registry=3 n_enumerated=3 missing=1, 2, 3`. Both numbers agree, every arm IS enumerated, and the marker still refuses.

Fix: state in the template that each per-arm row's NAME must be the corresponding `members=` entry VERBATIM (the checker matches byte-wise after the backtick/asterisk/whitespace strip), and that a human-readable label belongs in the row's prose, not its key. One sentence plus a worked non-string-key example.

**Gap 2 — the mismatch refusal names the symptom, not the remedy.** The no-registry-line refusal is exemplary: it prints both accepted forms, so a re-post self-corrects in one bounce. The enumeration-mismatch refusal prints `missing=<list>` and the registry source, which reads as "you forgot to enumerate these arms" — the #2163 defect it was built for. It does not mention that a row may exist under a DIFFERENT NAME, which is the other way to arrive here and the one that actually fired on #2546. The tell is visible in the message itself: `n_registry=3 n_enumerated=3` with 3 missing is arithmetically impossible under the forgot-a-row reading, so the checker already has the evidence to distinguish the two cases.

Fix: when `len(per_arm) >= n_registry` and `missing` is non-empty (i.e. rows exist but under other names), extend the reason with the naming remedy and list the row keys actually found. Keep the existing wording for the genuine forgot-a-row case, whose regression is pinned.

**Gap 3 — an int-keyed registry can never be driver-verified.** `_extract_registry_keys_from_driver` returns `None` unless every dict key is an `ast.Constant` holding a `str`. `#2546`'s driver declares `ARMS: dict[int, ArmSpec]`, a perfectly ordinary registry shape, so clause 5b can never run and the predicate PASSes marker-only forever: `registry-complete (marker-only — driver not verified: registry symbol not statically extractable: sorted(ARMS))`. The two-tier reason is honest and the reviewer arm (code-reviewer Step 0.55) does pick up the duty, so nothing shipped unverified — but a whole class of drivers silently gets the weaker tier, and the #2176 measurement that motivated the string-only scope ("15/15 anchored `^PHASES` drivers are module-level dict literals") says nothing about their key TYPES.

Fix: accept `ast.Constant` keys of `int` and `str` (compare via `str(key)`, which is what `members=1,2,3` already writes), leaving every other key form on the existing `None` fallback. Then re-measure how many in-repo drivers reach the driver-verified tier before and after, and record both numbers.

## Scope

- `.claude/rules/experiment-implementer-section-reference.md` — Gap 1 wording + worked example.
- `src/explore_persona_space/task_workflow.py` — Gap 2 refusal reason, Gap 3 key-type widening (both inside `smoke_arch_registry_check` / `_extract_registry_keys_from_driver`).
- `tests/test_task_workflow.py` — regressions: a rows-exist-under-other-names refusal asserting the new remedy text; an int-keyed dict reaching driver-verified; a mixed/unsupported key form still falling back to `None`; the existing forgot-a-row and string-key pins unchanged.
- Mirror surfaces that restate the grammar (`.claude/rules/code-reviewer-section-reference.md` § Step 0.55, `.claude/skills/issue/SKILL.md` § arm-registry, the `experiment-implementer-lean` agent-memory note) — keep the wording consistent wherever the identity requirement is now stated.

Do NOT weaken the byte-wise match itself: set-containment between `members` and the row keys is the property that caught #2163's 10-rows-vs-13-arms defect. Gap 1 and Gap 2 make it legible; Gap 3 widens what the recompute arm can verify. None of the three relaxes a gate.

## Dedup

Distinct from #2585 (`smoke-blind-spots.md`: row-index/data-reach as a fourth coverage-narrowing mechanism) — different target file, different fingerprint. Distinct from #2176 (which BUILT this contract) and #2171/#2163 (the keyed-span and forgot-a-row defects), all of which this task leaves pinned.

## Provenance

Surfaced by the `/issue 2546` orchestrator across review rounds 5 and 8 (2026-08-25) while discharging its own Step 5c-bis marker-shape blockers. The round-8 Claude code-reviewer's sole blocker was this grammar; the substance it gated was independently verified correct in the same verdict.
