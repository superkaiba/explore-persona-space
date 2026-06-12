---
name: claude-misses-same-file-siblings
description: Claude code-reviewer PASSes round-N when the round-(N-1) BLOCKER cited specific line ranges in a file; the fix lands at exactly those lines but a sibling silent-default / max_tokens=1024 / stratification-missing remains elsewhere in the SAME FILE or in a parallel pod-side smoke script
metadata:
  type: feedback
---

When a round-(N-1) substantive blocker is phrased "silent `except Exception` swallowing in `phase0_preflight.py:573-575, 725-727`" (with explicit line citations), Claude code-reviewer verifies the diff touches exactly those line ranges and tags the blocker `✓ FIXED`. Claude does NOT independently grep the SAME FILE for sibling instances of the bug class (silent-default `parsed.get("score", 0)`, `score = 0` on parse failure, `max_tokens=1024` in a pod-side smoke variant of the same eval, non-stratified `rng.sample` where the plan committed to stratification). Codex naturally re-greps each cited file for sibling violations and catches the leftovers.

**Why:** Claude reviewers verify the diff against the blocker QUOTE — they treat the cited line range as the contract surface. The contract is actually the bug CLASS (silent-default; truncation-bias; sampling-bias). When the implementer reads "fix lines 573-575" literally and the same file has a parse-from-LLM-JSON pattern at line 566 still using `parsed.get("score", 0)`, the bug-class fix doesn't propagate. Codex re-reads the whole file with the class in mind.

**Canonical pattern (issue #498 round 2):**

- Round-1 Block C: "silent `except Exception` swallowing (`phase0_preflight.py:573-575,725-727`, `phase4_judge.py:72-73`, `phase5_analyze.py:113`)."
- Round-2 implementer fixes those exact lines + adds `_retry_transient` + makes `phase4_judge.py` raise on missing score/JSON.
- Claude PASS walk-down: "Block C — `phase4_judge.py:37-72` mirror; judge parse failures RAISE on missing JSON object or missing `score` key (`phase4_judge.py:101-110`)" — verified the FAR file, not the cited file's other code paths.
- Codex grep of `phase0_preflight.py` finds line 566 `"judge_score": int(parsed.get("score", 0))` — same silent-default class, different code path (pilot scoring) in the SAME FILE, NOT in the cited line ranges. This silent-0 contaminates Cohen's kappa, which Block E's NEW κ ≥ 0.85 gate depends on.

- Round-1 Block B: "`max_new_tokens=1024` violates ≥2× rule" — cited sites `phase4_eval.py:123` + `phase1_generate_RNeg.py:64`.
- Round-2 fix lands at both cited sites.
- Codex catches `phase2_smoke_judge.py:71` `max_tokens=1024` — pod-side smoke gate, not in the round-1 cite list. Same `explains_well` rubric → same truncation profile → same false-FAIL risk on the smoke gate.

- Plan A19: "stratified sampling by (arm × trait × eval_context), ≥3 cells per 18 strata." Implementer prose comment in `phase4_judge.py:264-269` mentions "stratified subsample" — Codex reads the next line (272) and sees `rng.sample(range(len(scored)))`. Plan-adherence FAIL hidden behind a confident-sounding code comment.

**How to apply:** When reconciling a Claude PASS vs Codex FAIL on a round-N code review where the round-(N-1) brief cited specific line ranges:

1. For each cited file, read 50 lines AROUND each cited line range (not just the cited lines). Grep the WHOLE file for the bug-class pattern (`parsed.get("score", 0)` / `int(.*get("score", 0))` / `score = 0` / `max_tokens=` / `rng.sample(range(` / `except Exception:.*pass`).
2. Also grep PARALLEL pod-side smoke / dispatcher scripts that exercise the same DV — `phase2_smoke_judge.py` is the pod-side smoke of `phase4_eval.py`'s eval; same rubric, same DV, same truncation profile, ≥2× rule applies identically.
3. If a sibling instance exists in the same file or in a same-DV-class file, Codex is right — FAIL with the sibling locations enumerated.
4. Cross-check plan §11 / numbered Assumptions for HIGH-risk implementer commitments that name a specific implementation pattern (stratified sampling, named-distractor negatives, on-policy DV); grep the implementation for the LITERAL implementation pattern, not just for a prose comment that mentions it.

**Sibling smell:** A code comment that says "the X is a SEMANTIC-EQUIVALENT REWRITE" or "stratified subsample" or "raises on missing X" near a line that does the OPPOSITE — the comment was written for the round-N-1 reviewer to read, not as a contract on the next line.

Related: [[claude-skips-caller-grep]] (orphaned helpers); [[claude-treats-round-n-minus-1-mustfix-as-acceptance]] (sibling rubric/handler families); [[claude-misses-sibling-resampler-inconsistency]] (sibling resampler set-comp vs list-comp). This entry generalizes those to "same file, different code path, same bug class."

Origin: task #498 round-2.

**Partial-fix in the SAME launch script (task #601 round-2):** the round-2 implementer added `.processed`-sentinel tolerance at 3 of 4 read sites — the dispatcher's `_check_gates` (`candidate = sentinel if exists else processed`) + launch p7/p8 `test -f X || test -f X.processed` (with comments naming the race) — but launch p4 (`i601_launch.sh:77`) still `open()`s the bare smoke-sentinel name under `set -euo pipefail`. The sentinel IS poller-conforming (`sentinel_schema_version: 1`), so a poll tick during the post-sentinel HF upload window renames it and the pipeline dies after a SUCCESSFUL smoke. Claude PASSed by verifying the round-1-named fix sites; the implementation report even claimed p4 was covered. Defense: when a fix is "tolerate the poller rename", grep the launch/dispatch scripts for EVERY read of an `issue-<N>-*.json` sentinel and verify each has the fallback; a comment acknowledging the race at one site is evidence the class applies to ALL sites.

**Sibling SCRIPTS extension (task #505 round-4):** the dispatcher loader (`_load_persona_bank_and_r` in `dispatch.py`) is fixed to route through canonical `load_persona_bank` (which unwraps the #472 structured-dict `{"schema_version": ..., "data": {...}}` payload). Two sibling standalone-entrypoint wrappers (`scripts/issue505_build_pv_centroids.py:66` and `scripts/issue505_panel_coverage.py:62`) still raw-load via `bank = json.loads(args.persona_bank.read_text())` and pass `bank` to `build_pv_centroids(persona_bank=...)` / `run_panel_coverage_gate(persona_bank=...)` — both annotated `dict[str, str]`, both immediately iterate as a name→prompt mapping → crash on real data. Claude PASSed having verified the dispatcher path; never re-grepped the worktree for `json.loads(args.persona_bank` siblings. Defense: when the just-fixed bug is a `json.loads → consumer(persona_bank=...)` (or any "raw-load → typed-consumer") pattern, grep the WHOLE worktree for `json\.loads.*persona_bank` / `json\.loads.*\.read_text\(\)` siblings AND check whether the plan documents standalone CLI entrypoints under `scripts/` that bypass the dispatcher's canonical loader (an `analyze.py` recovery message that says "Re-run `scripts/<wrapper>.py`" is a strong signal the wrappers are anticipated production entrypoints, not dead code). This is the same blindspot — bug-class vs cited-line-range — applied across sibling FILES that share the same data-loading contract.
