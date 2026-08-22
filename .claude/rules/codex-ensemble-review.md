# Codex ensemble review — the five doubled review sites

<!-- Relocated from CLAUDE.md (always-on) to cut per-spawn context.
     CLAUDE.md keeps a load-bearing summary + a pointer to this file;
     this file is the FULL text, verbatim as it stood in CLAUDE.md. -->

### Codex ensemble review

Five review steps (`critic`, `code-reviewer`, `interpretation-critic`, `clean-result-critic`, `follow-up-critic`) run Claude + Codex twin (gpt-5.5 via `openai/codex-plugin-cc`) in parallel — all rounds up to the per-reviewer cap (10) for the first four sites (all-rounds policy since 2026-06-12). `follow-up-critic` is a **SINGLE-PASS redundancy screen** — runs ONCE per proposal set BEFORE any proposal routes; bar is REDUNDANCY ONLY; nothing dropped (`redundant` → `on_hold` via `epm:followup-parked-redundant v1`, revivable). PASS+PASS → advance. FAIL+FAIL overlapping → bounce. FAIL+FAIL disjoint → union blockers (one round). PASS vs FAIL → spawn `reconciler` (Claude, fresh context, binding). **Mechanical-contract-only FAILs** (every blocker tagged `marker-shape`, `smoke-run-missing`, or `git-provenance` — the last evidence-based: a read-only git probe must confirm the flagged state is NOT introduced by the round's diff) are stripped by the orchestrator when the implementer marker is present + conforming (SKILL.md Step 5c-bis; `code-reviewer.md` Steps 0.5-0.9). Reconciler invocations don't count against the cap. **At the cap** the orchestrator applies the strip once more and either continues (all residual stripped → PASS) or SURFACES any substantive residual — never ships past (interactive: user; autonomous: `epm:failure v1 failure_class: code` + `status:blocked` + notify + CRON-TEARDOWN). **NOT doubled:** `upload-verifier`, `consistency-checker`. /adversarial-planner Phase 2 uses in-context reconciliation; the other 4 sites use marker mode. (The retired `reviewer` / `codex-reviewer` pair — deprecated 2026-05-13, files deleted 2026-08-05 — folded its statistical-framing check into `clean-result-critic`.) See `workflow.yaml § ensemble_review`.

Codex dispatch (`scripts/codex_task.py`) is used ONLY for the 4 twin reviewer roles. Twin wrappers are prompt-composers only; the **orchestrator** dispatches the helper as bg Bash (the only pattern that delivers a real notification when Codex terminates):

```bash
Bash(run_in_background=true,
  command="uv run python scripts/codex_task.py --issue <N> --effort <high|xhigh> \
    --prompt-file /tmp/codex-prompt-issue-<N>.md --output-file /tmp/codex-output-issue-<N>.md")
```

The helper itself serializes spawn + post-spawn confirm on a repo-keyed advisory lock (`.claude/cache/codex-dispatch.lock`, #2323 — the shared codex-companion jobs index is a non-atomic lost-update surface), so dispatching N twins in ONE message stays safe — do NOT re-sequence parallel dispatches to sequential on its account. Helper posts `epm:codex-task-spawned`, then `epm:codex-task-completed`/`epm:codex-task-failed`. On marker-post failure: retry once, then drop to `tasks/_orphaned_markers/`. Orchestrator posts the verdict marker from the output file (trigger-dense rounds: grep the verdict line + extract the marker block mechanically and post via `post-marker --file` — never page the findings body into context; SKILL.md § File-only Codex verdict posting, #1275) and forwards the extracted block's machine-readable `CONCERN:: ` rows to the concerns ledger via `scripts/persist_verdict_concerns.py` (validate pre-post, persist post-post; re-run from the durable marker note at every resume row whose predicate includes an existing current-round codex marker; SKILL.md § Codex concerns persistence at verdict collection; #2326). On a hard org usage-limit failure the helper writes `.claude/cache/codex-quota-exhausted-until` and short-circuits every later dispatch until the parsed reset (exit 9, note reason `codex-quota-exhausted`); delete the sentinel to force a probe dispatch (#1126).

**Pre-spawn sentinel check (#1204) — check BEFORE composing.** The exit-9 short-circuit fires at DISPATCH time, but the thin `codex-*` composers spawn BEFORE any dispatch — during an outage each round burns composer spawns whose prompts are discarded. Before spawning any `codex-*` prompt-composer in a review round, run the canonical check below. `CODEX_QUOTA_LIVE` ⇒ SKIP every `codex-*` composer spawn that round: treat each twin as an INSTANT CONFIRMED no-show (single-Claude decision per that site's no-show fallback), log ONE chat line + ONE `epm:progress` note per skipped round when a task is in scope. NEVER fabricate `epm:codex-task-failed` or a twin verdict marker. FAIL-OPEN: sentinel absent / unreadable / corrupt / expired / implausibly far-future, or `EPM_SKIP_CODEX_QUOTA_SENTINEL=1` → spawn normally (the dispatch-time short-circuit stays the arbiter). The decision keys on the TWO-SIDED window `now < until_unix <= now + 45 d` (the helper's `QUOTA_MAX_PLAUSIBLE_SECS` ceiling), so a corrupt far-future timestamp can never wedge composer spawning off permanently; a `parse_ok: false` sentinel with a plausible future `until_unix` is honored the same.

```bash
# Canonical pre-spawn quota check (#1204): CODEX_QUOTA_LIVE until=<iso> | CODEX_QUOTA_CLEAR
ROOT="$(dirname "$(git rev-parse --path-format=absolute --git-common-dir)")"   # main checkout, worktree-safe
uv run python -c '
import json, os, sys, time
p = os.environ.get("EPM_CODEX_QUOTA_SENTINEL_PATH") or sys.argv[1]
try:
    assert os.environ.get("EPM_SKIP_CODEX_QUOTA_SENTINEL") != "1"
    d = json.load(open(p))
    now = time.time()
    until = float(d["until_unix"])
    assert now < until <= now + 45 * 86400   # QUOTA_MAX_PLAUSIBLE_SECS parity (codex_task.py:295)
    print("CODEX_QUOTA_LIVE until=" + str(d.get("until_iso", "?")))
except Exception:
    print("CODEX_QUOTA_CLEAR")
' "$ROOT/.claude/cache/codex-quota-exhausted-until"
```

(Read-only check: unlike `_quota_sentinel_active` it never deletes a sentinel — lifecycle stays with the helper.) A live Codex quota outage triggers the watcher's `codex_outage_pass` (one deduped push per episode + weekly re-alert; kill switch `EPM_DISABLE_CODEX_OUTAGE_PASS=1`).

If the wrapper/session is killed leaving the Codex job running (an `epm:codex-task-spawned` with no terminal marker), first confirm the wrapper is actually dead (`pgrep -f 'codex_task[.]py.*<job-id>'` empty — a live wrapper would double-poll, duplicate markers, and cross-cancel on signal), then re-attach instead of re-dispatching: `uv run python scripts/codex_task.py --issue <N> --reattach <job-id> --output-file <same path>` (#1020). The same re-attach recipe is the recovery for a helper exit 10 (`post-spawn probe exhausted`, #2323): the job is usually still RUNNING behind a transiently-erased jobs-index entry — never blind re-dispatch on exit 10 (the #2321 orphan generator); the failure marker's note carries the exact command.
