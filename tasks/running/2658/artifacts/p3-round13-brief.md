# Task #2658, implementation round 13 brief (plan v6 cap amendment + round-12 review fixes)

AUTO_REVIEW_DISABLED=1. Do not invoke any review or diagnostic loop on your own output. You are
the `experiment-implementer` for task #2658 (EPS). Read the full spec at
`.claude/agents/experiment-implementer.md` first. One turn: implement, test, lint, commit by
explicit path, push, post the markers, report. Everything you produce lands durably in THIS turn.

## Where

- Worktree (cwd, already on branch `issue-2658`):
  `/mnt/eps-data/thomasjiralerspong/issue779_task_clone/.claude/worktrees/issue-2658`
- Canonical task folder (NOT the worktree copy): `$(uv run python scripts/task.py find 2658)`.
  Plans: `<task>/plans/v4.md` (base), `v5.md` (A1-A6 amendments), `v6.md` (A7, this round).
  `plan.md` -> v6. plan_version=v6. Hand v4+v5+v6 together; v6 is authoritative where it speaks.
- Round-12 brief (context only, already implemented): `<task>/artifacts/p3-resume-round12-brief.md`.
- Record `BASE_SHA=$(git rev-parse HEAD)` at start; pin reads to it if files churn.

## Live-process constraint (read first)

A detached P3 power run (`scripts/issue2658_power.py --phase all --split pilot`, pid in
`/mnt/eps-data/thomasjiralerspong/issue2658_logs/power-all-r12.pid`, log `power-all-r12.log`
beside it) is READING AND WRITING `eval_results/issue_2658/power/*` in this worktree right now.
Therefore in this round:
- never run `issue2658_power.py --phase all|power|discordance|cost` against the real
  `eval_results/issue_2658` dir; `--phase gate` on the real dir is allowed ONLY if the pidfile's
  process is gone (`kill -0 $(cat <pidfile>)` fails). Otherwise smoke the gate on a tmp copy.
- do not edit `simulate_power`, the PowerLedger key, or `select_production_n` (the in-flight
  ledger keys do not encode code version; a change would silently invalidate banked units). If
  a fix genuinely needs them, STOP and say so in the report instead.
- do not `git add` anything under `eval_results/issue_2658/power/`.

## D1. Plan v6 section A7: production length cap 4096 (dev + test), pilot unchanged at 1024

Ground truth: `scripts/issue2658_common.py` line ~63 (`C.DECODER`, max_new_tokens 1024),
`scripts/issue2658_generate.py` line ~66 (`MAX_MODEL_LEN = 8192`), cap-hit producer
`G.cap_hit_report` (~line 150-160, threshold 0.02 strictly above), realized shard summaries
`eval_results/issue_2658/gen_summary/pilot_shard0[0-7]of08.json` (`cap_hit` block).

1. Amendment record producer. Add a function in `scripts/issue2658_power.py` (and a CLI entry,
   e.g. `--phase cap-amendment` or a sibling flag, your call, documented) that DERIVES
   `eval_results/issue_2658/power_inputs/cap_amendment.json` from the realized pilot shard
   summaries: pilot cap (read from the pilot decoder record in the raw-completion files or the
   shard summaries, never hand-typed), production cap 4096 (a named constant with a `Source:`
   comment citing plan v6 A7), the registered rule text (v4 section 5), threshold, the list of
   over-threshold cells with fractions and n, n truncated records (finish_reason length) and
   total records, the disclosure text (v6 A7, verbatim), `plan_version: v6`, schema id, UTC
   timestamp. Fail loud if any shard summary is missing or if the derived offender set is
   empty (an amendment with no offenders is a wiring error). Write it with the same
   atomic-write helper the module already uses.
2. Split-aware decoder. The generation path used for dev and test (P4/P5) must read
   `max_new_tokens` from the amendment record when `split in {dev, test}` and keep 1024 for
   `pilot`. No bare default: absent record + non-pilot split raises. Prompt budget becomes
   `MAX_MODEL_LEN - cap` per split (pilot 7168, production 4096); the frozen prompt-budget
   assertion keeps failing loud. The generation fingerprint / config record written per shard
   must carry the realized cap so P4 artifacts are self-describing. Check every call site that
   builds the decoder (generate.py, any pod-side launcher for P4/P5, inference.py if it
   re-derives the decoder) and update each; grep for `DECODER`, `max_new_tokens`, `1024`.
3. Gate. `_gate_cap_hit` gains status `AMENDED` (module constant beside GATE_WAIVED), returned
   when: the record exists, `production_cap >= 2 * pilot_cap`, and the record's cell list
   covers every realized over-threshold cell (set equality or superset, computed gate-side,
   see D2). `AMENDED` is non-blocking in `evaluate_gates` (like WAIVED, unlike FAIL) and is
   NEVER reported as PASS. gate_verdict.json gains a top-level `amendments` list carrying the
   record summary + disclosure; the existing `disclosures` list also gets the A7 disclosure.
   Any other state (record missing, cap too low, offender not covered) keeps today's FAIL.
4. Cost report: when the record exists, the P4 generation projection scales the truncated
   share by the realized token growth bound (cap ratio) and names the assumption in
   `cost_report.json`; leave the projection unchanged when the record is absent. Keep it
   simple and stated, no new fitting.

## D2. Round-12 review fixes (g2 concerns + nits; g4 blocker + minor)

g2 (commit 8b68ab80e25, `_gate_cap_hit`), verdict CONCERNS, none blocking, fix all:
- concern 1: the gate must not trust the producer's `amendment_required` flag. Recompute the
  offender set gate-side from `per_cell_fraction` against `G.CAP_HIT_AMEND_THRESHOLD` (strictly
  above, matching the producer) and assert every shard's recorded `threshold` equals that
  constant (mismatch -> PowerInputError naming the shard path).
- concern 2: coverage becomes set-based: `covered` intersected with `P.expected_cells`;
  `missing = expected - declared - covered`; foreign keys (not in expected) -> PowerInputError
  naming them; NOT-ESTIMABLE detail names the missing cells.
- nit 3: malformed shard summary -> `PowerInputError` with the path, not bare KeyError.
- nit 4: do not truncate the FAIL/AMENDED detail to 10 problems; list all offenders.
- nit 5: the unit8 fixture should exercise a multi-shard layout (at least 2 shards) and derive
  its expected offenders via `G.cap_hit_report`, not a re-implemented rule.

g4 (commit 299a6518e12 + round gates), verdict FAIL with the single tag marker-shape:
- Step 0.55: the round-12 `epm:smoke-architecture-check` v3 row regressed the `arm-registry:`
  line to a command-transcript form. Your round-13 `epm:smoke-architecture-check` (v4) MUST
  carry the N/A escape form exactly:
  `arm-registry: N/A — no module-level registry; dispatch is the argparse phase choices literal at scripts/issue2658_power.py:<line>: all, cost, discordance, gate, power (all = dispatcher sequencing the four phases)`
  (fix the line number and add any new phase you introduce, e.g. cap-amendment), plus the
  missing row `- all: N/A — dispatcher (sequences discordance→power→cost→gate)` and a row for
  each new phase. Then run `uv run python scripts/task.py check-smoke-arch-registry 2658
  --repo-root <worktree>` and paste its rc=0 output into the report.
- minor: the v12 report's pilot-timing HF-fetch leg cites a nonexistent `--fetch-from-hub`
  flag; the hub path triggers by omitting `--logs-dir`. Use the correct command in your v13
  report's smoke section.

g3 (commit 4fdfcabe458, artifacts) PASS, nothing to do.

g1 (commit 609cc9ee5e2): see D3.

## D3. g1 findings (commit 609cc9ee5e2, verdict CONCERNS, none blocking; fix the cheap ones)

- concern 2 (labeling): `scripts/issue2658_judge_spend.py` enumerates `raw_completions/judge/**`,
  so the "pilot judge spend" artifact and module docstring include the 8 canary batches
  (89 pilot + 8 canary = 97 batches; 21,395 succeeded calls vs 20,950 pilot dispatch calls).
  Fix the LABELS, not the enumeration: name the artifact/module scope "pilot + canary judge
  Batch-API spend", keep `n_batches_by_subtree`, and in `cost_report` state next to the per-call
  dollar mean that its denominator includes ~2 percent canary calls (same population as the
  numerator, so the mean is unbiased for the pilot instrument). Do not re-query the API.
- concern 3 (test isolation): `test_stager_writes_provenance_offline` reads the committed
  `eval_results/issue_2658/direction_provenance.json` through `_gate_row_vector_alignment`.
  Make the test self-contained: write a minimal provenance fixture into the tmp eval root and
  point the gate at it (or monkeypatch `F.PROVENANCE_PATH`), so it passes in a bare extraction.
- concern 1 (retry routing) is already fixed by 299a6518e12; nothing to do.
- nits: `srec.get("basis", "measured")` (power.py ~1394) and `staged.get("revision")` (power.py
  ~1962) become required-field reads that raise `PowerInputError` when absent; the human-audit
  gate description on the WAIVED branch stops saying "NOT-ESTIMABLE" (~2206); the recorded
  enumeration `pattern` in judge_spend.py (~230) matches what the code actually rglobs; leave
  `PRICE_RETRIEVED_AT` as is (a re-retrieval would change the record for no reason) but note it
  is a date in the field name or a comment.

Full per-group verdicts (read them, windowed): /tmp/issue-2658-split-review-r12-g{1,2,3,4}.md;
composed round verdict: `epm:code-review v21` on the task (`task.py view 2658 --json | jq` on
that event only, never an unfiltered view).

## D4. Tests, lint, commits, markers, report

- Tests: extend `tests/test_issue2658_unit8.py` (gate AMENDED / FAIL / NOT-ESTIMABLE paths,
  set-based coverage, threshold-equality assert, multi-shard fixture, amendment record
  producer on a synthetic 2-shard layout, decoder split behaviour incl. absent-record raise).
  Run `uv run pytest tests/test_issue2658_unit3.py tests/test_issue2658_unit8.py
  tests/test_issue2658_unit11.py -q` and paste counts.
- Lint: `uv run ruff check <files> && uv run ruff format --check <files>`; then the no-flags
  `uv run python scripts/workflow_lint.py` differential on your changed files (report the
  count on parent vs HEAD for those files only).
- Commit by explicit path from the worktree (`git -C <worktree> commit -F <msgfile> -- <paths>`,
  never `git add -A`), push `issue-2658`, and verify `git log -1 --oneline -- <path>` for each
  new/changed file. Commit the amendment record JSON (text, small) by explicit path too.
- Do NOT touch `eval_results/issue_2658/power/*` (live run) and do not post any
  `epm:code-review` marker.
- Post `epm:experiment-implementation` v13 (head sentinel + (a) what changed (b) plan
  deviations, if any (c) copy-pasteable verification commands with success signals
  (d) risks) via `task.py post-marker 2658 epm:experiment-implementation --file <path>`, and a
  conforming `epm:smoke-architecture-check` v4 row (D2/g4). Smoke = real runs of the new
  producer against the realized shard summaries (write the record into the real
  `power_inputs/` dir: that path is not under `power/` and is not touched by the live run) and
  of the gate on a tmp copy of `eval_results/issue_2658` (or the real dir only if the power
  pid is gone), with exit codes and artifact digests.
- Trigger-dense discipline: never paste behavior-bank items or raw completions into any
  marker or report; reference files by path and counts only.
- Agent-memory writes (`.claude/agent-memory/**`) are committed by explicit path in the same
  turn.

Report (final Agent result text): commits (sha, files), test/lint counts, marker versions
posted, the amendment record's headline numbers (n offender cells, n truncated / total,
caps), the gate status the smoke produced, and anything you deliberately did not do and why.
