# Code Review: issue-2225 split-review r2 g3 — 2c48d9e026 "sync-reissue idempotency + fingerprint regime keys + seam-audit consumption" (FIX-VERIFICATION)

**Verdict:** PASS
**Blocker tags:** none
**Diff size:** +394 / −24 across 6 files (5 scripts + 1 test file); read in full
**Fix-verification:** all three assigned r1 items genuinely closed; all named minors closed
**Tests actually run:** yes — `uv run pytest tests/test_issue2225_judge_analysis.py -q` → 28 passed (incl. the 3 new tests); `ruff check` on all 6 changed files → clean
**Security sweep:** CLEAN (no secrets, no eval/exec, no new upload destinations; skip-print emits tag/trait only, no content)

## Fix verification (the brief's three items)

### 1. `run_sync_reissue` idempotency (r1 g4 Major 3) — CLOSED
- Skip predicate `scripts/issue2225_judge.py:943`: `block.get("judge_meta", {}).get("api_refusal_reissue")` → per-unit `continue` BEFORE target selection, judge dispatch, and merge. The done-marker is written by the merge itself (`:1026-1033`) **atomically with the merged draws in the same `_atomic_write_json` (`:1033`)** — a crash mid-unit leaves the unit unmarked AND unmerged (the on-disk block is either pre-merge or fully merged+marked), so the cached-draw re-serve on rerun re-merges from pre-merge state exactly once. This is the exact fix shape r1 prescribed ("skip units whose `judge_meta.api_refusal_reissue` already exists").
- Parity spend guarded on rerun: `if reissued_total and parity_pool:` (`:1046`) — a fully-remediated rerun has `reissued_total == 0`, so no duplicate parity re-judge.
- **Run-twice test** (`tests/test_issue2225_judge_analysis.py::test_sync_reissue_is_resume_idempotent`): real-body test — executes the real `run_sync_reissue` over a realistic fixture partial (censored rollout + clean rollout), fakes ONLY the API boundary (`judge_graded`) with a **signature-conformant** `def` fake (verified against the real signature `graded_judge.py:222-233` — all 11 params mirrored incl. keyword-only structure) and `rubric_for` (loader boundary). Runs the merge twice; asserts (a) draw-list byte-stable, (b) `len(...) == 2` ("draws doubled on re-run"), (c) **zero further judge calls** on run 2.
- **Fails pre-fix** (verified by trace against the pre-fix code, which had no `:943` guard): run 2 re-selects `rollout_n_api_refusal[0] = 2 > 0` (never reset), calls the fake again (`calls` grows → assert (c) fails), and appends `[42.0, 42.0]` a second time → `[42.0]*4` (asserts (a)+(b) fail). Deterministic, not flake-dependent.
- Digest coherence: `n_api_refusal` in the digest row comes from the original accounting (never reset — correct), and remediation status is read from the reissue meta, so the two surfaces cannot contradict.

### 2. Fingerprint regime keys — CLOSED
- eval_gen: `"model"` added to `unit_fingerprint` (`issue2225_eval_gen.py:276`), threaded at the single call site (`:544`, `model=model_name`). capture: `"model"` added (`issue2225_capture.py:110`), threaded at the single call site (`:181`). mmlu: `"limit"` added (`issue2225_mmlu.py:109`), threaded at the single `run_single` call site (`:302`) and into the fan-out child argv (`:461-462`).
- **No spurious mid-experiment invalidation**: comparison is dict EQUALITY against the stored sidecar (`unit_done` `eval_gen:301`, `_trait_done` `capture:131`, `_done` `mmlu:127`) — key-order-independent, no serialized-string instability. Pre-fix artifacts (fingerprints lacking the new key) are deliberately invalidated, which is the intended semantics; no production artifacts exist yet on this branch (no `eval_results/issue_2225/`, no `data/issue_2225/`), so the invalidation costs nothing.
- MMLU `--limit` (r1 g3 Major 1, the companion fix): dial at `build_argparser` (`:461`), threaded into the lm-eval argv via the new test-pinnable `_lm_eval_cmd` (`:233-259`), into the resume fingerprint, AND recorded in the payload (`:339`). Dispatcher P0 leg verified wired in sibling commit `26aefadc1d` (`issue2225_dispatch.sh:86` `MMLU_P0_LIMIT=200`, consumed at `:270-272`) — the full dial + dispatcher wiring + fingerprint-key triple is present. Test `test_mmlu_limit_threaded_into_argv_and_fingerprint` pins argv threading, the no-limit form, and `fp_probe != fp_full` (a probe never resume-satisfies the full run).

### 3. Seam-audit consumption + stated deviation (r1 g3 Major 2) — CLOSED
- **Stated deviation recorded**: `issue2225_capture.py:29-39` docstring names the plan §4.6-item-4 wording ("per-segment token-id concatenation") vs the realized string-concat helper, the deliberate-reuse rationale (probe-train/probe-apply capture-convention consistency with #778's pool), and points at the consuming `seam_audit` block — flagged for carry into the clean-result scope caveats.
- **Consumption wired**: `issue2225_analysis.py:902-911` reads `manifest["traits"][trait]["seam_mismatch_count"]` into every probe-application row (manifest write path verified: per-trait meta sidecar carries the count at `capture:283`, manifest assembles the sidecars at `capture:311-328`); `:977-999` emits a `seam_audit` block into `probe_shifts.json` — per-unit counts + `seam_fraction` + an explicit exclude-or-sensitivity-check note. Exactly the r1 ask ("have the P5 analysis either exclude or sensitivity-check seam-flagged units").
- Resume provenance is safe: pre-fix partial rows lack `bundle_sha256` → excluded from the `done` set (`:876-885`) → recomputed → last-wins in `by_key` (`:933`, dict-comprehension over file order), so every surviving row carries the new fields.

## Other r1 items closed in this commit
- **Digest remediation surface (r1 g4 Minor 2)**: per-row `api_refusal_reissued` + `n_draws_recovered_by_reissue` (`judge.py:616-619`), `api_refusal_remediation` summary block (`:645-654`), and the warning now fires only for UNREMEDIATED censored units with a distinct REMEDIATED info line (`:663-674`). Test-pinned (`test_digest_reports_reissue_remediation`).
- **`--upload-tags` empty-token filter (r1 g3 Minor 2)**: `capture.py:482-487` — `if s.strip()` with the rationale comment.
- **`merge_slot` pid-less stale reclaim (r1 g3 Minor)**: `mmlu.py:163-179` — empty/unparseable slot older than 10 min reclaimed via mtime gate; fresh open→write windows left alone. (Residual theoretical race — a holder writing its pid at age > 600 s loses its slot — is negligible and worst-case admits one extra concurrent merge; acceptable for this semaphore.)
- **Probe-application bundle-identity resume key (r1 g4 Minor 1)**: `_sha256_file` + `bundle_sha256` row field + sha-filtered done-set (`analysis.py:809-818, 876-885`).
- **`_pack_large_json` oversized-header guard (r1 g4 Minor)**: `judge.py:1100-1106` — fail-loud `ValueError` instead of re-entering the >10 MB LFS force-route.
- **Friendly unknown-tag errors (r1 g3 Minor)**: `mmlu._resolve_single_target` (`:281-291`) + the same shape in `capture.capture_one_model` (`:155-161`).

## Issues Found

### Critical (block merge)
None.

### Major
None.

### Minor / Concerns (don't block)
- `scripts/issue2225_analysis.py:1040` (`run_projection`): the projection phase's resume key is still `(tag, trait)` only — NOT keyed on the direction files' identity — the same #722-r3 stale-reuse class the probe-application fix just closed one function above. A regenerated `{trait}_{E1,E2}.pt` (directions re-run / pool fix) silently resume-reuses stale projection rows. Untouched by this commit and not named by r1 (which flagged only the probe application), so not a fix-verification failure — but the fix pattern (`_sha256_file` pin per direction file + filtered done-set) is now in-file and ~5 lines. Recommend applying in the next touch of this file. Mechanizable: yes.
- `scripts/issue2225_judge.py:944-949` (comment only): the skip comment justifies skipping a reissued unit's parity candidates with "its draw lists are batch+sync mixed" — for `n_ref == 0` rollouts (the only parity candidates) the draw lists are pure batch even after a merge; the skip is still correct/safe (conservative parity-pool shrink on a resume), the stated reason just overstates. Cosmetic.
- `scripts/issue2225_analysis.py:983` (`if n_seam:`): a manifest MISSING the `seam_mismatch_count` key (None) reads identically to 0 — currently unreachable (every capture meta writes the key, `capture.py:283`), but if capture manifests ever drift, an unknown would silently read as unflagged. A `None`-vs-0 split in `seam_audit` would make it fail-legible. Cosmetic.

### Suggestions
- `run_digest` row's `n_draws_recovered_by_reissue` does a bare `reissue["n_draws_recovered"]` — KeyError (fail-loud) on a future meta-shape drift; fine as-is, noting it is deliberate fail-loud.

## Tests
- New: `test_sync_reissue_is_resume_idempotent` (run-twice, real body, signature-conformant boundary fakes — fails pre-fix by trace), `test_digest_reports_reissue_remediation`, `test_mmlu_limit_threaded_into_argv_and_fingerprint` (no-GPU argv/fingerprint pins via the extracted `_lm_eval_cmd` helper).
- Executed: full `tests/test_issue2225_judge_analysis.py` → 28/28 pass, 21 s; `ruff check` clean on all 6 files.
- Existing tests still valid: yes (additive changes; the `_run_lm_eval` signature change is keyword-only and both call sites updated — grep-verified no other callers).

## Recommendation
PASS. All three assigned r1 blockers/majors are genuinely closed with the prescribed fix shapes, each pinned by a real-body test that fails pre-fix; the six named minors are all closed. The one residual worth a follow-up line is the `run_projection` resume-key sibling (same class, out of this round's scope).
