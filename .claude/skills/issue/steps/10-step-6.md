# Step 6: Pod provisioning + experimenter dispatch (experiment only)

Step body relocated verbatim from `.claude/skills/issue/SKILL.md`
(#2155). SKILL.md keeps the heading, the state machine and the
Orchestration Procedure router; read this file when the run reaches
this step.

---

Only if status is `running` (entered from Step 5b PASS for `experiment`)
and no `epm:launch` marker exists.

#### Step 6a: HF gate-access check

Provisioning a pod only to have the run die seconds in on a `401 gated
repo` is wasted GPU-minutes. Before provisioning, scan the cached plan
for HF model IDs and verify the user's `HF_TOKEN` already has access to
each, using `huggingface_hub.HfApi.auth_check` (idempotent — it raises
`GatedRepoError` when the token lacks gate access, and returns cleanly
when access is already granted). There is no programmatic way for a
consumer to auto-accept someone else's gated-model gate page, so a
blocked repo halts with the gate URL for the user to click through once:

```bash
PLAN_PATH=$(uv run python scripts/task.py find <N>)/plans/plan.md
# Source .env FIRST — the VM shell does not inherit HF_TOKEN, so running this
# probe bare yields a false "HF_TOKEN missing" exit 2.
set -a; [ -f "$REPO_ROOT/.env" ] && source "$REPO_ROOT/.env"; set +a
uv run python - "$PLAN_PATH" <<'PY'
import os, re, sys
from huggingface_hub import HfApi
from huggingface_hub.utils import GatedRepoError, RepositoryNotFoundError

plan = open(sys.argv[1]).read()
# HF model IDs cited in the plan (org/name, the canonical gated form).
repo_ids = sorted(set(re.findall(r"\b([A-Za-z0-9][\w.-]+/[\w.-]+)\b", plan)))
token = os.environ.get("HF_TOKEN")
if not token:
    print("HF_TOKEN missing"); sys.exit(2)
api, gated = HfApi(), []
for rid in repo_ids:
    try:
        api.auth_check(rid, token=token)
    except GatedRepoError:
        gated.append(f"https://huggingface.co/{rid}")
    except RepositoryNotFoundError:
        pass  # not a real model repo (a false-positive org/name match)
if gated:
    print("GATED (manual approval needed):", *gated, sep="\n  "); sys.exit(1)
print("all cited HF repos accessible"); sys.exit(0)
PY
```

- Exit code `0` -> proceed to 6a.5.
- Exit code `1` (gate access needed) -> post `epm:hf-gate-pending v1`
  with the gate URLs, leave status at `running`. Post the §5 marker:
  ```bash
  uv run python scripts/post_step_completed.py --issue <N> --step 6c \
    --exit-kind clean --notes "hf-gate manual approval pending"
  ```
  EXIT. User clicks through the gate page, re-runs `/issue <N>`.
- Exit code `2` (`HF_TOKEN` missing) -> post `epm:hf-gate-pending v1`
  with diagnostic, status to `blocked`. Post the §5 marker:
  ```bash
  uv run python scripts/post_step_completed.py --issue <N> --step 6c \
    --exit-kind failure-exit --notes "HF_TOKEN missing; status:blocked"
  ```
  EXIT.

The same `HF_TOKEN` is pushed to the pod by `bootstrap_pod.sh`, so a pod
provisioned in 6b sees the identical gate state as the local VM.

#### Step 6a.5: Carry-over artifact existence check (before provisioning)

Plans for follow-ups (and any experiment that reuses a prior run's
checkpoint, dataset, or eval output) cite HF / WandB URLs for the
artifacts they depend on. Provisioning a pod only to have the run die
seconds in on a `404` is pure wasted GPU-minutes. Before provisioning,
verify every carry-over URL the plan cites actually resolves:

```bash
PLAN_PATH=$(uv run python scripts/task.py find <N>)/plans/plan.md
uv run python -c "
from explore_persona_space.orchestrate.hub import verify_artifacts_exist
import sys
ok, missing = verify_artifacts_exist(plan_path='$PLAN_PATH')
if not ok:
    print('MISSING ARTIFACTS:', *missing, sep='\n  ')
    sys.exit(1)
print('all carry-over artifacts resolve')
"
```

`verify_artifacts_exist` scans the cached plan for HF repo URLs
(`huggingface.co/...`) and WandB run URLs (`wandb.ai/.../runs/...`) and
HEAD-checks each against the Hub / WandB API using the user's
`HF_TOKEN` / `WANDB_API_KEY`. It returns `(ok, missing_urls)`.

- All resolve -> proceed to 6a.6.
- Any missing -> post `epm:carry-over-missing v1` with the unresolved
  URLs, set status to `blocked` (the plan depends on an artifact that
  isn't there; provisioning would burn GPU on a guaranteed failure).
  Post the §5 marker:
  ```bash
  uv run python scripts/post_step_completed.py --issue <N> --step 6c \
    --exit-kind failure-exit --notes "carry-over artifact(s) missing; status:blocked"
  ```
  EXIT. User fixes the cited URL (re-upload, or correct the plan) and
  re-runs `/issue <N>`.

**Second stanza (#1469) — plan-referenced LOCAL repo inputs (the #734/#1434
class).** `verify_artifacts_exist` covers HF/WandB URLs only; a plan-cited
`eval_results/...` input that exists only on the VM (untracked, committed but
unpushed, or on origin/main only while the branch was cut earlier) is
invisible to it, and every lane boots from a git materialization of the
PUSHED branch (GCE `git clone --depth 1 --branch issue-<N>`; RunPod bootstrap
fetch+reset; SLURM materialize_branch_src), so the clone will NOT have it
(#1434). Run the git-tree gate (pure git — no tokens, no
network beyond a bounded fetch):

```bash
uv run python scripts/verify_carryover_inputs.py --plan "$PLAN_PATH" --issue <N>
```

- Exit `0` -> proceed to 6a.6. WARN lines are informational — carry them into
  the step notes; a `data-local-only` WARN means the workload must self-build
  or HF-stage that input (artifact-reuse check (h)); never block on a WARN.
- Exit `1` with ONLY recoverable failures -> remediate in-step and re-run the
  gate ONCE: `committed-unpushed` -> push the branch (`git -C "$WT" push
  origin issue-<N>`, bare, exit code checked); `on-main-not-on-branch` ->
  merge origin/main into the branch (or rebase it) in the WORKTREE and push
  (the file is already committed — never `git add`). Still failing, or any
  `untracked-local-only` failure -> same contract as the first stanza: post
  `epm:carry-over-missing v1` with the helper's failure lines, set status
  `blocked`, post the §5 marker via `scripts/post_step_completed.py`
  (`--step 6c --exit-kind failure-exit --notes "carry-over local input
  unreachable on pushed ref"`), EXIT. Remediation for untracked files is a
  commit+push of the cited file on the issue branch (the #1434 fix), then
  re-run `/issue <N>`.
- Exit `2` (plan missing/unreadable) -> fail loud like a missing plan in the
  first stanza; do NOT skip the gate.

Residual risks this gate does NOT cover (it reduces the class, not
eliminates it): config-file indirection, runtime-constructed paths (the gate
catches the plan-text citation, not the consumer's path construction),
HF-staged `data/` inputs (WARN only — staging correctness stays with
artifact-reuse check (h)(iii)), direct `dispatch_issue.py` launches that
bypass 6a.5, and extension-less citations. The check ref defaults to
`origin/issue-<N>`; where the lane's materialization ref is known to differ
(RunPod `BOOTSTRAP_BRANCH` defaults to `main`), thread `--ref` accordingly.

**Rsync-lane invocation (#1835).** When the task's `backend:` frontmatter
names an rsync-materialized SLURM lane — every member of
`router._PER_CLUSTER_LANES` (`nibi` / `fir` / `mila` / `fellows`) plus the
legacy `cluster` alias — OR is absent/`auto` (the auto chain is
fellows-FIRST, an rsync lane), run the gate with `--lane rsync` plus any
plan-named `--extra-sync-path` values: git-reachability is necessary but NOT
sufficient there — the lane's scratch tree is an rsync of
`RSYNC_INCLUDE_PATHS` with `eval_results/` excluded, so an in-ref
`eval_results/...` citation NOT covered by the sync set downgrades to FAIL
`rsync-lane-not-synced` (#1689). That FAIL is recoverable IN-STEP, not a
park: add the covering `--extra-sync-path` value(s) and re-run the gate
ONCE. Compose the gate call and the later `dispatch_issue.py launch` from
ONE variable (e.g. `EXTRA_SYNC_ARGS=(--extra-sync-path
eval_results/issue_<M>/ladder)` threaded to BOTH) so the gate-PASSing set
and the launched set cannot drift.

#### Step 6a.6: HF write-headroom probe (quota gate, before provisioning)

Step 6a verifies READ access only; a namespace at its public-storage
quota passes the gate-access check, the carry-over HEAD-checks, AND
pod-side preflight, then 403s on the run's FIRST upload — after the pod
is already provisioned. (#555) Before provisioning, probe the
actual failing operation — a tiny (~1 KB) write to the project model
repo, immediately deleted:

```bash
# .env is already sourced by Step 6a (which exits on missing HF_TOKEN).
uv run python - <<'PY'
import io, sys
from huggingface_hub import HfApi

REPO = "superkaiba1/explore-persona-space"
PROBE = ".quota_probe/probe.txt"
api = HfApi()
try:
    api.upload_file(path_or_fileobj=io.BytesIO(b"quota probe"),
                    path_in_repo=PROBE, repo_id=REPO,
                    commit_message="quota probe (auto-deleted)")
    api.delete_file(path_in_repo=PROBE, repo_id=REPO,
                    commit_message="remove quota probe")
except Exception as e:
    resp = getattr(e, "response", None)
    if resp is not None and resp.status_code == 403 and "storage" in str(e).lower():
        print("QUOTA EXCEEDED:", e); sys.exit(1)
    # Fail-soft on NON-quota errors (transient 5xx, network blip): the
    # probe's only job is the quota 403; reachability is preflight's job.
    # Do NOT block provisioning on an inconclusive probe.
    print("probe inconclusive (non-quota error, proceeding):", e); sys.exit(0)
print("HF write headroom OK"); sys.exit(0)
PY
```

- Exit code `0` (probe OK or inconclusive) -> proceed to 6b.
- Exit code `1` (storage quota exceeded) -> post `epm:hf-quota-exceeded v1`
  with the verbatim 403 text + the probed repo id, set status to
  `blocked` (the storage decision — delete old artifacts vs upgrade the
  namespace — is the user's; provisioning would burn GPU on a guaranteed
  upload failure). Post the §5 marker:
  ```bash
  uv run python scripts/post_step_completed.py --issue <N> --step 6c \
    --exit-kind failure-exit --notes "HF namespace storage quota exceeded; status:blocked"
  ```
  EXIT. Do NOT provision. User frees space / upgrades storage and
  re-runs `/issue <N>`.

**Size-aware projected-headroom gate (#1034).** The 1 KB probe above catches
only the ALREADY-over-quota case. When the approved plan's §9/§10 projects
≥100 GB of canonical-public LFS uploads, ALSO run
`uv run python -m explore_persona_space.orchestrate.preflight --no-gpu
--planned-upload-gb <N>` (decimal GB, the plan's projected LFS total). A
KNOWN-insufficient exit (the gate's ERROR is live-confirmed via a forced
re-probe) is handled EXACTLY like the quota-exceeded exit above: post
`epm:hf-quota-exceeded v1` (with the gate's error text), post the same §5
marker (`uv run python scripts/post_step_completed.py --issue <N> --step 6c
--exit-kind failure-exit --notes "projected LFS headroom insufficient;
status:blocked"`), set `blocked`, EXIT — the storage decision is the
user's. Fail-open otherwise: unknown headroom /
disabled check / routing armed all WARN and proceed to 6b.

**Billing-state gate (#1654).** The same `preflight --no-gpu
--planned-upload-gb <N>` invocation now ALSO runs the zero-byte LFS
batch-negotiation billing probe (`hub.check_lfs_write_gate`, declared ~16 GB):
the 1 KB probe above is structurally false-green for quota/billing 403s, which
fire only on the LFS endpoint (#1586). A
billing-blocked/storage-blocked ERROR exit is handled exactly like the
quota-exceeded exit above (post `epm:hf-quota-exceeded v1` with the gate's
error text, `status:blocked`, do NOT provision). Coverage boundary: a passing
~16 GB-declared probe means "not blocked NOW at that scale", NOT "credits
sufficient for the whole run" (e.g. 215 GB) — mid-run credit exhaustion stays
with the reactive 403 backstop; do NOT size the probe to `--planned-upload-gb`
(a probe declared above per-file upload caps — e.g. >50 GB — would fail for
size reasons and degrade the verdict to `unknown`).

#### Step 6b: Pod provisioning

**Backend dispatch (slice-6 unified router — auto by default; RunPod leads the auto order, #2054).**
Read the task's `backend:` frontmatter via
`uv run python scripts/task.py view <N> --json | jq -r '.frontmatter.backend // empty'`.
**The frontmatter value (or its absence) is fed verbatim to the slice-6
router via the dispatch helper** —
`explore_persona_space.backends.issue_dispatch.dispatch_for_issue`
calls `backends.router.route()` with production-injected deps and
returns a typed `RunHandle`. The router decides which backend actually
runs:

- **Empty / absent frontmatter → `auto`.** The router walks the
  resolved auto lane order — **standing default: RunPod FIRST (the
  Anthropic-org pool), then fellows, then the free SLURM lanes**
  (`DEFAULT_AUTO_LANE_ORDER = ("runpod", "fellows", "nibi", "fir",
  "mila")` — #2054 promoted runpod to the head, `reason:
  auto_runpod_first`; #2028: GCP provisioning is DISABLED, so the auto
  order carries no gcp rung; unconditional, no date gate; override via
  the comma-separated `EPM_AUTO_LANE_ORDER` env var — `runpod` is a
  LEGAL entry as of #2054; `gcp`-while-disabled / unknown lanes in the
  override raise loudly). A runpod capacity miss (nothing provisioned)
  falls through to the lanes behind it, and the #656 terminal rung
  survives as the end-of-chain RunPod RETRY (`reason:
  auto_fallback_runpod`) — only if THAT launch also fails does the
  chain raise `NoComputeAvailableError` (pins:
  `tests/test_router.py::test_default_auto_lane_order_has_no_gcp` +
  `test_runpod_first_capacity_miss_falls_through_then_terminal_retry`).
  Contiguous SLURM
  lanes (Nibi, Fir if wired, Mila if its socket is alive) are ranked
  among themselves by tz-corrected `sbatch --test-only` est-start, the
  best is submitted and parked up to `FREE_WAIT_SECONDS` (600 s; ALWAYS
  applied — see `backends.router`); park-cap-exceeded cancels + moves
  to the next lane.
- **`backend: runpod`** explicit override → RunPod PIN (`reason:
  override`, distinct from the auto chain's `auto_runpod_first` /
  `auto_fallback_runpod`); prefer bare `auto` — it already tries
  RunPod first (#2054).
- **`backend: nibi` / `fir` / `mila`** → that lane, with the same park
  + cancel state machine as auto.
- **`backend: gcp`** → REFUSED (#2028): `route()` raises the typed
  `GcpDisabledError` before any wiring/ladder work, and
  `classify_terminal_exception` maps it to `failure_class: infra` /
  `status: blocked` / `reason: gcp_backend_disabled` (NOT
  watcher-re-drivable — drop the pin, or the deliberate
  `router.GCP_PROVISIONING_DISABLED = False` rollback flip).
- **Legacy `backend: cluster`** is normalized to `backend: nibi` by
  `issue_dispatch.normalize_backend_value` (the slice-5 router rejects
  the bare `"cluster"` literal). The legacy `select_backend` /
  `EPM_CLUSTER_MAX_WAIT_SECONDS` env knob from the pre-slice-6 wiring
  are no longer consulted — the 10-min `FREE_WAIT_SECONDS` park
  supersedes the old 6-h default.

**Lane capability check (run BEFORE the dispatch call).** All router
lanes (GCP + SLURM) execute custom workload commands: pass the plan's
launch command via `--workload-cmd 'bash scripts/issue<N>_dispatch.sh
...'` (mutually exclusive with `--hydra`; exactly one required — the
CLI fails loud otherwise; note the neither-set defense-in-depth raise
exists in the GCP renderer only — SLURM's default stage chain is
pre-existing behavior). Auto routing is valid for dispatch-script
workloads (#588). Residual gaps that still need the explicit
`--backend runpod` override (or the named knob): (a) 70B intents
(`inf-70b`/`ft-70b` have no GCP machine-type mapping — fail-loud by
design); (b) workloads needing the open-instruct `--extra gpu` venv on
a SLURM lane under a non-ft intent (venv extras follow the INTENT, not
the workload kind: `ft-7b`/`ft-70b` custom commands DO build `--extra
gpu`; `lora-7b`/`eval`/`debug` custom commands build the base venv —
`needs_gpu_extras`, slurm.py); (c) workloads
needing interactive SSH-MCP-driven orchestration mid-run (the
experimenter launch pattern); (d) **multi-day workloads on GCP
longer than the fence** — the lane pins `--instance-termination-action=DELETE` +
`--max-run-duration` (default 7d — the FLEX_START ceiling, #741), so a
sweep longer than the fence is deleted
mid-run; thread the plan's declared fence via `--max-run-duration
<dur>` on `dispatch_issue.py launch` (gcloud duration shape, e.g.
`30h`; lands in `spec.extra["max_run_duration"]`, inert on non-GCP
lanes — #628) or use the RunPod override. **When overriding to RunPod, name the residual gap in
the launch marker note** (CLAUDE.md rule). The dispatch CLI
cross-checks the task's ACTUAL frontmatter and classifies the override
3-ways, each with a DISTINCT marker flag (additive visibility — the
launch is never blocked): passing `--backend runpod` while the
frontmatter `backend:` does not name a backend (absent/empty, or an
explicit `auto`) triggers a LOUD stderr warning +
`extra.override_without_frontmatter=true` on the
`epm:backend-selected` marker; frontmatter naming a DIFFERENT
recognized lane (`gcp`/`nibi`/`fir`/`mila`, or the legacy `cluster`
alias for nibi) triggers a conflict warning +
`extra.override_conflicts_frontmatter=true`; an unrecognized value
(typo'd `gpc`, non-string `true`) triggers a hygiene warning +
`extra.frontmatter_backend_unrecognized=true` — the latter two also
carry `extra.frontmatter_backend: "<value>"`. Frontmatter
`backend: runpod` is the one legitimate backing and stays silent. For the gcp/auto lanes the dispatch script must exist
on the pushed branch, so you MUST pass `--repo-branch issue-<N>`
EXPLICITLY: the orchestrator runs `dispatch_issue.py` from the repo
ROOT (pinned to `main`), so the `--repo-branch` default (the cwd's
current branch) resolves to `main`, NOT the issue branch where a
per-issue driver script lives — the GCE startup script then clones
`main`, the driver is absent, and the workload dies ~4 min in with the
EXIT trap powering the VM off (#595). Defense-in-depth
(#987): `dispatch_issue.py` and `backend_poll.py` self-pin lane-infra
imports (`explore_persona_space.backends.*`, lazy `scripts.*`) to the
MAIN checkout via a `__main__`-guarded git-common-dir sys.path
bootstrap, so a worktree-cwd script-mode invocation of either
entrypoint no longer imports a stale branch lane template; the
repo-root invocation rule above remains the documented default (it
also selects main's venv for third-party deps), and the pin covers
ONLY script-mode execution — module-IMPORT consumers of `backends/`
(e.g. `autonomous_session_watch.py`) are deliberately unpinned, so
the cron-wrapper convention of `cd`-ing to the main checkout before
invoking them stays load-bearing. Residuals the pin does NOT close:
pre-#987 worktree COPIES (branches cut before the fix) carry no
bootstrap until rebased, an already-running process keeps its cached
stale modules, import-mode callers (`dispatch_for_issue` from a
worktree venv) get no pin, and already-launched workloads keep the
template they were rendered with. Four more gcp/auto
composition rules ((e)/(f) from #599;
(g) from #608; (h) from #606): (e) **GPU
sizing on the gcp/auto lanes comes from `--intent`, never `--gpus`** —
the GCP lane maps intent → machine type statically
(`backends/gcp.INTENT_TO_MACHINE`: `lora-7b`/`lora` →
`a2-ultragpu-1g`, 1 GPU; `ft-7b` → `a2-ultragpu-4g`, 4 GPU) and
ignores `--gpus` (only RunPod and SLURM honor the override), so pick
the intent whose machine matches the plan's GPU spec; a gcp-reachable
launch with a mismatched `--gpus` is refused pre-route by
`dispatch_issue.py` (exit 2, `reason: gpus_machine_mismatch`). (f)
**Never reference `$WORKLOAD_ROOT` bare in a workload-cmd — it is
exported ONLY by the GCE startup script**, so the exact command a
GCP→RunPod failover (or a SLURM fall-through) re-runs aborts under the
RunPod launcher's / SLURM custom stage's `set -u` before the driver
starts (#825; `dispatch_issue.py` now lints this at launch —
warn-by-default + `extra.workload_cmd_lane_env_risk` on the
`epm:backend-selected` marker, exit-2 refusal on a provably-certain
lane or under `--strict-workload-cmd-env`, #1329). A driver defaulting
`REPO_ROOT=/workspace/explore-persona-space` still dies on the GCE lane
(the startup script clones to `$WORKLOAD_ROOT`, `/workspace/eps-issue-<N>`,
and cds there), but the GCE startup script already exports
`REPO_ROOT="$WORKLOAD_ROOT"` before running the workload (#641,
`backends/gcp.py render_startup_script`), so compose
`--workload-cmd 'bash scripts/<driver>.sh'` with a SELF-RESOLVING driver
(`REPO_ROOT="${REPO_ROOT:-${WORKLOAD_ROOT:-$(cd "$(dirname "$0")/.." && pwd)}}"`,
the #825 pattern), or use the set-u-safe default expansion inline:
`--workload-cmd 'REPO_ROOT="${WORKLOAD_ROOT:-$PWD}" bash scripts/<driver>.sh'`
(every lane cds to the checkout root first; `${VAR:-default}` is safe
under `set -u`). (g)
**Sentinel-signaling dispatchers must not rely on auto's DRAC/Mila SLURM
fallback** — a dispatch script that posts markers via pod-side sentinel
files (`/workspace/logs/issue-<N>-*.json`) works on the DRAINED lanes
(runpod/fellows — the fellows drain landed at #1898 via
`slurm_monitor.drain_cluster_sentinels`; `backend: gcp` is REFUSED as
of #2028): DRAC/Mila compute nodes have no
`/workspace`, so the script fails loud at `mkdir -p /workspace/logs` and
burns the submission (#608, commit 3022ff7bc); pin one of the two
drained lanes (`backend: fellows`, or runpod with a
named residual gap), or convert the dispatcher to the SLURM signaling
contract (`status.json` heartbeat + `[phase=...]` log lines) before
routing auto (planner.md §9 names this constraint at plan time). (h) **Boot-disk
sizing on the gcp/auto lanes comes from the plan's Reproducibility pod
row, threaded via `--boot-disk-gb` on EVERY launch — relaunches after a
code-fix round included** — the GCP lane defaults the boot disk to
300 GB pd-ssd (`backends/gcp.GcpConfig.default_boot_disk_gb`), which a
ZeRO-3 full-FT (`ft-7b`) fills with optimizer-state checkpoints in ~1h:
the instance kernel-panics on the full disk, cloud-init ENOSPCs, the
guest agent cannot write `authorized_keys` (SSH publickey lockout), and
the wedged VM idles on 4×A100 until deleted (#606). When the
plan's pod row names a disk size, pass it; for `ft-*` intents whose
plan names none, default to ≥500 GB. `dispatch_issue.py` warns loud
(stderr + `extra.boot_disk_default_with_ft_intent=true` on the
`epm:backend-selected` marker) when an ft intent is gcp-reachable with
no `--boot-disk-gb` — warning only, never a refusal (small-disk ft
smokes stay legitimate). (i) **WandB project on `--workload-cmd`
launches defaults to `issue<N>`** — the GCP startup script and the
SLURM custom stage export `WANDB_PROJECT="${WANDB_PROJECT:-issue<N>}"`
before the verbatim command, so HF-Trainer workloads that never set a
project stop landing in WandB's global default `huggingface` project
(Upload Policy: training metrics → `project=<experiment_name>`; #601
follow-up r1 landed there silently). An inline `WANDB_PROJECT=...`
prefix on the workload command — or the workload setting its own
project internally — still wins (`:-` fills only unset/empty); hydra
launches are unaffected (project comes from Hydra config). (j) **Launch
env pins on `--workload-cmd` launches — thread `--env-pin KEY=VALUE`
when the plan's Reproducibility Card declares one** — the `--env-pin`
channel (`dispatch_issue.py launch --env-pin KEY=VALUE`, repeatable;
merged in #1669) persists an env export to `spec.extra["env_pins"]` →
the handle sidecar → every lane's workload-cmd launcher (GCE startup
script's workload branch, SLURM custom stage, RunPod launcher), AND the
async failover reconstructors (`backend_poll._runspec_from_gcp_handle` /
`_runspec_from_runpod_handle`) re-export the pins onto the fresh pod,
so a wedge-failover pod's runs land in the plan-declared destination
instead of the generic `issue<N>` fallback (rule (i) above — #1586: a
wedge-failover pod rebooted with only the generic WandB
default and its runs landed in the wrong project). KEY is restricted to
`backends.base.ENV_PIN_ALLOWED_KEYS` (secret KEY names are
unrepresentable by construction); consult that frozenset for the current
set. `--env-pin` REQUIRES a non-empty `--workload-cmd` (parse-time
refusal, exit 2 — every renderer insertion point is a workload-cmd
branch; a hydra launch has no pin consumer). ADOPTION (the
`--boot-disk-gb` pattern above): launch composers pass
`--env-pin WANDB_PROJECT=<declared project>` (and any other
`ENV_PIN_ALLOWED_KEYS` value the plan's Repro Card declares) on every
`--workload-cmd` launch — including relaunches after a code-fix
round — whenever the plan declares a non-default value; a flag-less
launch keeps today's behavior.
SLURM custom stages are
render-tested only as of #588 (never live-run).
(#571; #588: the renderer refuses a bare
launch and `--workload-cmd` carries dispatch scripts on every lane.)

**Ad-hoc probe workloads are committed scripts invoked by path — never
inline interpreter one-liners in `--workload-cmd`.** A probe dispatch
composes exactly like a full run: a committed script on the pushed issue
branch, invoked by path with `--repo-branch issue-<N>`; staging/phase
logic lives in the script, never in the command string. An inline
`python -c '...'` / `uv run python - <<EOF` one-liner as the workload
body is the named anti-pattern — un-lintable and un-smokeable (ruff, the
Step 9c mapped tests, and the pre-launch import/signature probes see
only committed files) and quoting-fragile (incident #1482, 2026-07-19: a
G1 reconciliation probe's placeholder-broken inline staging one-liner
would have SyntaxError'd after phase b0 and spuriously failed over to
RunPod; the just-created GCE instance was cancelled ~2 min after create,
`reason=orchestrator-quoting-error`). Recovery is the incident's own
fix: rewrite as a committed branch script, push, re-dispatch by path.
Siblings: rule (f) above (lane-env fragility) and the
`.claude/rules/gotchas.md` #1310 inline-stdin entry — that covers
signature drift of `-c`/heredoc helper calls INSIDE a committed script;
this rule bars the one-liner as the workload body itself. Standing
exception: the fixed `write_completion_sentinel` append chained onto the
workload command (`.claude/agents/experimenter.md`) — signature-stable
and probe-covered by the gotchas #1310 pre-launch discipline, not ad-hoc
workload logic.

**Hand-composed phase argv dry-run (REQUIRED before any instance-booting
dispatch of a newly-composed argv; #1738).** Before any
`dispatch_issue.py launch --workload-cmd '<cmd>'` (any lane) whose inner
driver argv was hand-composed this session — a NEW phase, a follow-up
round, a plan-§10 command transcription — dry-run the EXACT production
argv on the VM first. Re-dispatching a byte-identical, previously-probed
command whose driver's CLI/validation surface is untouched since that
probe is exempt. FIRST classify the driver's CLI family
(`grep -n 'parse_args\|@hydra.main' <script>`): a Hydra driver
(`@hydra.main` — e.g. `scripts/train.py`, `scripts/eval.py`) is probed
with Hydra's own compose-only check (append `--cfg job` to the exact
production overrides; it composes the config and exits without running
the job) — the argparse probe below validates NOTHING for Hydra. For an
argparse driver, prefer the driver's own `--dry-run` / parse-only flag
when one exists; otherwise run the generic bounded probe:

```bash
# Generic argv dry-run — <script.py> + the EXACT production args.
# Runs the driver through parse_args AND its early post-parse
# validation, bounded; the driver's own imports run (torch etc.), so
# allow ~60s.
timeout --kill-after=10s 60s uv run python - <script.py> <args...> <<'PY'
import argparse, runpy, sys
sys.argv = sys.argv[1:]                    # ['<script.py>', <args...>]
engaged = []
_orig = argparse.ArgumentParser.parse_args
def _probe(self, *a, **k):
    ns = _orig(self, *a, **k)
    engaged.append(True)
    print("ARGV-PARSE-OK:", ns, flush=True)
    # Do NOT exit here: the repo's dominant convention enforces
    # required inputs POST-parse via
    # raise SystemExit("--x or --y required") (44 scripts; incident
    # #1738: issue1738_multiturn_fits.py:1629, rc=1) — post-parse
    # validation must execute.
    return ns
argparse.ArgumentParser.parse_args = _probe
runpy.run_path(sys.argv[0], run_name="__main__")
if not engaged:
    print("ARGV-PROBE-NEVER-ENGAGED", file=sys.stderr); sys.exit(3)
PY
```

Read the outcome by MESSAGE, not rc alone:

| Outcome | Reading |
|---|---|
| rc=2 (argparse error: missing-required / unknown flag) | Argv defect — fix it on the VM in seconds instead of after a boot cycle. |
| rc=1 with a validation message after `ARGV-PARSE-OK` (the `SystemExit("--x or --y required")` class) | Argv defect — the #1738 class (Phase-3 attempt 1 omitted the required `--split-file`/`--manifest-*` flag; the workload died rc=1 ~7 s AFTER a full GCE flexstart boot + venv install). |
| rc=124 (timeout) with `ARGV-PARSE-OK` printed | PASS — parse + early validation survived the window. The timeout is the deliberate side-effect ceiling: the driver may begin real work (mkdir, HF reads) inside it, so probe BEFORE any out-root state you care about. |
| rc=137 with `ARGV-PARSE-OK` printed | PASS — the `--kill-after` hard-kill variant of the timeout outcome (the driver ignored SIGTERM); same disposition as rc=124. |
| Nonzero exit AFTER `ARGV-PARSE-OK` whose message names a VM-only environment gap (a pod/GCE-staged path absent locally, CUDA on a no-GPU VM) | Judged pass — state it in one line in the dispatch note. |
| rc=0 WITH `ARGV-PARSE-OK` printed | Pass — but the ENTIRE workload ran to completion LOCALLY and succeeded; state explicitly that local execution occurred (vanishingly rare for an instance-booting workload — it usually means the dispatch may not be needed at all). |
| rc=3 + `ARGV-PROBE-NEVER-ENGAGED` | No argparse parser was reached (Hydra or a bespoke CLI) — NEVER a pass; use the family-appropriate check above. |
| Any exit with NEITHER sentinel printed (e.g. a parser-less script that `sys.exit(0)`s on its own before the guard runs) | NEVER a pass — treat as never-engaged; every pass row above is sentinel-keyed. Use the family-appropriate check above. |

Notes: (a) a bare `--help` probe validates NOTHING about the composed
argv (help exits 0 before any validation) — `experimenter.md` item 7's
flag-presence scan is the COMPLEMENT (bogus flags); this probe is the
missing-input side. (b) For a wrapper `.sh` dispatcher, probe the INNER
python driver argv the wrapper composes (plus `bash -n <wrapper>` for
wrapper syntax). (c) A driver calling `parse_known_args` directly
bypasses the monkeypatch (one direct caller today:
`src/explore_persona_space/experiments/factor_screen_365/__main__.py`)
— the never-engaged guard converts that to rc=3 when the script
returns; a script that exits on its own shows NEITHER sentinel, which
the table's neither-sentinel row bars from reading as a pass. The guard DETECTS a
parser-less run after the fact (the script may have executed fully
within the timeout window before rc=3 returns); PREVENTION rests on the
first-step CLI-family grep classification plus the timeout ceiling.
(d) This is a pre-launch PROBE, not a workload — the "Ad-hoc probe
workloads" committed-script rule above governs workload bodies, not
this probe (same class as the gotchas #1310 signature probe).

**Dispatch-input/env/flag preflight (REQUIRED before any instance-booting
dispatch; #1964 — extends the argv dry-run above; same trigger + same
byte-identical-re-dispatch exemption).** The argv dry-run validates PARSE
+ early post-parse validation only; the four probes below cover what it
deliberately excludes — each is a VM-side check costing seconds, run
BEFORE provisioning (repeated wasted provision+staging cycles across
#1739/#1689/#1345/#1902/#1946/#1900 were all discoverable pre-boot).
Auto-continue duties in the argv-dry-run register — no new gate; record
each probe's one-line disposition in the dispatch note.

- **(a) Staged-input existence probe.** For EVERY input path / HF prefix
  the composed chain resolves — flag-passed paths, env-pointed dirs, HF
  prefixes the driver downloads — probe existence on the surface the
  TARGET will read, before provisioning. HF prefixes: scoped
  `huggingface_hub.list_repo_tree(path_in_repo=<prefix>)` /
  `list_repo_files` per named prefix — never full-tree enumeration on
  the ~1M-file data repo (gotchas rule). Repo-resident paths are
  LANE-AWARE: on a git-clone lane (GCE/RunPod) git-tree reachability on
  the PUSHED ref suffices — `git cat-file -e origin/issue-<N>:<path>`
  (commit AND push first; the lane clones origin, not the local
  worktree); on an rsync-materialized SLURM lane (fellows/nibi/fir/mila
  — and `auto`, whose chain leads with fellows) git-reachability is
  necessary but NOT sufficient: additionally require sync-set coverage —
  the path matches `RSYNC_INCLUDE_PATHS` or an `--extra-sync-path`, i.e.
  the `verify_carryover_inputs.py --lane rsync` semantics — for every
  composed-argv repo path (a git-only probe re-opens #1689 on the
  default lane: fellows job 15188 died at its FIRST read of a
  gate-certified, committed input the rsync set never materialized). The
  argv dry-run's "pod/GCE-staged path absent locally → judged pass"
  disposition row judges PARSE only and does NOT satisfy this probe —
  that row is exactly where unstaged target-side inputs hide (#1739). Scope split vs Step 6a.5: this probe covers the 6a.5 gate's
  own named residual — composed-argv / env-resolved / config-indirected
  paths the plan text never cites — and does NOT re-run the 6a.5
  plan-citation gate. A missing input BLOCKS dispatch: stage it first
  (commit+push, HF upload, or widen the sync set).
- **(b) Env-pin completeness probe.** Enumerate the dispatched driver's
  env reads — `grep -nE 'os\.environ|os\.getenv' <driver + issue-local
  imports>` — and check each read that lacks a run-correct default
  against the composed launch env (the `--env-pin` set, the inline env
  prefix, or lane-exported defaults). A consumed-but-unset pin BLOCKS
  dispatch (#1345, #1739).
  One-line disposition in the dispatch note.
- **(c) Per-LEG carry-over verification.** A multi-leg round (follow-up
  leg, secondary phase, teammate-built leg) runs
  `scripts/verify_carryover_inputs.py` PER LEG whose inputs the FIRST
  leg's Step 6a.5 gate never saw — with `--lane rsync` +
  `--extra-sync-path` where the #1835 rsync-lane rule applies — before
  that leg's dispatch (#1900).
- **(d) Relaunch flags verbatim from the handle sidecar.** A relaunch
  copies the flag set VERBATIM from
  `.claude/cache/issue-<N>-handle.json` — never re-derived from plan
  prose or memory (#1902). Deliberate changes are named DIFFS against the sidecar set
  in the relaunch note. Machine-sized caps (`--rss-cap-gb`, thread caps,
  width) are RE-DERIVED for the TARGET machine on any cross-machine move
  — a sidecar cap is sized to the machine that wrote it (#1946).

The handle the dispatch helper returns is persisted to
`.claude/cache/issue-<N>-handle.json` (the bg-Bash poller reads it
back; see Step 6d.2).

**Marker trail** (all VM-side; both `backends.router.route` and the
SLURM helpers call `task.py post-marker` via
`backends.slurm.post_marker_via_task_py`):

- `epm:backend-selected v1` — posted by `route()` on EVERY decision
  (including a pre-escalation intermediate marker when the auto chain
  is about to spend GCP credit). Body carries `requested_kind`,
  `chosen_kind`, `reason` (`override` / `reconnect` / `auto_started` /
  `auto_fallback_gcp` / `no_compute_available` / `workload_failure`),
  `cluster`, `elapsed_seconds`, the per-lane `attempts` ladder, and
  `extra` (`cancel_race?`, `gcp_attempts_today?`, `intermediate?`,
  plus the dispatch-CLI override-guard flags — all scoped to the
  explicit `--backend runpod` path: `override_without_frontmatter?`
  when the task frontmatter does not name a backend (absent/empty, or
  an explicit `auto`); `override_conflicts_frontmatter?` when it names
  a DIFFERENT recognized lane (gcp/nibi/fir/mila, or legacy `cluster`);
  `frontmatter_backend_unrecognized?` when the value is a typo /
  non-string; the latter two also carry `frontmatter_backend?` with
  the raw lowercased value).
  Legacy `frontmatter_*` / `slurm_*` reason codes from the pre-slice-6
  `select_backend` are preserved in `workflow.yaml § markers` for
  back-compat reads.
- `epm:cluster-launched v1` — posted by `SlurmBackend.launch` (or
  `GcpBackend.launch` — GCP reuses this marker name) right after the
  job is submitted; body carries `job_id`, `scratch_dir`, `log_path`,
  etc.
- On the RunPod path the existing `epm:pod-provisioned` /
  `epm:run-launched` markers are still posted by the experimenter.

**Terminal-exception translation.** `route()` raises one of four
terminal `RouteError` subclasses when no lane succeeded; the
dispatch helper translates each via
`issue_dispatch.classify_terminal_exception` into the
`epm:failure v1` body + status the orchestrator already routes on
(SKILL.md Step 7):

| Exception | failure_class | status |
|---|---|---|
| `NoComputeAvailableError` | `infra` | `blocked` |
| `WorkloadSurfacedError` | `code` | `blocked` |
| `GcpAttemptCapExceededError` | `infra` | `blocked` |
| `ManualAttentionRequiredError` | `infra` | `blocked` (carries orphaned job_id) |

Step 6d.2 runs the bg-Bash poller against the persisted handle (no
per-backend branch); Step 8 runs `confirm_artifacts` + `teardown` on
the same handle. The cluster path's monitor (`epm:cluster-poll v1` /
`epm:cluster-terminal v1`) keeps working — `SlurmBackend.poll` calls
into `backends.slurm_monitor.build_poll_result` exactly as before;
the bg-Bash poller (`scripts/backend_poll.py`) prints the same
PollResult JSON shape regardless of backend.

The remainder of this section describes the RunPod / per-issue pod
specifics. The cluster path's sbatch carries an EQUIVALENT inline
preflight stanza (HF/WandB reachability, GPU visibility,
`$SLURM_TMPDIR` headroom) so a misconfigured job fails fast inside
the SLURM allocation.

Compute is ephemeral on every backend — no permanent pod fleet, no
permanent VM, no permanent SLURM submission stays alive past the run.

**Operational dispatch (slice-6 router, ALL backends).** The
orchestrator shells `scripts/dispatch_issue.py launch` — the operational
seam that builds the production backends (`RunPodBackend`,
`SlurmBackend` for every available cluster, `GcpBackend`) + the injected
dependencies (`marker_poster` = `backends.slurm.post_marker_via_task_py`;
`is_started` = SLURM-aware `query_slurm_state` status==RUNNING probe;
`is_live_after_cancel` = `query_by_name` non-empty probe;
`reconnect_fn` = per-kind SLURM-`squeue --name` + `gcp.reconnect_or_none`
(includes a `mila` branch matching the `nibi`/`fir` reconnect closure);
`mila_socket_alive` = the real `backends.slurm.mila_socket_alive` probe
that runs `ssh -o BatchMode=yes -o ConnectTimeout=5 mila true` over the
ControlMaster socket — slice 7's first-class wiring. A dead / OTP-
expired socket returns False (skip-the-lane, NOT an error); refresh is
the Claude-session cron documented at
`.claude/cron-prompts/mila-otp-refresh.md` and orchestrated through
`scripts/mila_socket_refresh.py` (un-armed in slice 7; live arming in
slice 8)) and calls
`backends.issue_dispatch.dispatch_for_issue` (which calls
`backends.router.route()`). The router decides the lane (auto → free
cluster → GCP, or honors an explicit override); RunPod's launch goes
through `RunPodBackend.launch` (which shells `pod_lifecycle.py
provision` under the hood) so the sidecar JSON is written uniformly
across backends. The bg-Bash poller (`scripts/backend_poll.py`) reads
that sidecar tick after tick (Step 6d.2); Step 8's
`scripts/dispatch_issue.py finalize` reads it again to run
`confirm_artifacts` + `teardown` (the same RunHandle from launch all
the way through teardown).

Before this launch call, run the Step 9 entry guard § Pre-dispatch
external-marker triage and post its `external-markers triaged:` line in an
`epm:progress` note immediately before dispatching (the launch posts
`epm:run-launched` / `epm:cluster-launched`, not a stage-dispatch
breadcrumb).

The operational command:

```bash
# Read the task's backend frontmatter (empty / absent → auto).
BACKEND=$(uv run python scripts/task.py view <N> --json | jq -r '.frontmatter.backend // empty')
# Infer --intent from the plan: training a 7B model → ft-7b or lora-7b;
# eval/generation → eval; 70B work → inf-70b/ft-70b. Override with
# --gpus / --time-budget-hours for anything else.
INTENT=<inferred>

# Single operational call — runs the router (auto / explicit override
# both flow through here). On RunPod the underlying pod_lifecycle.py
# enforces team scoping (X-Team-Id), SSH bring-up (startSsh: true,
# exposes 22/tcp), pinned image, and runs bootstrap inline (uv, repo,
# .env with HF_TOKEN, HF cache, preflight); on SLURM the SlurmBackend
# renders + ssh-submits the sbatch; on GCP the GcpBackend renders +
# ``gcloud compute instances create``s the VM. Hydra args repeatable.
uv run python scripts/dispatch_issue.py launch \
    --issue <N> --intent "$INTENT" --repo-branch "issue-<N>" \
    ${BACKEND:+--backend "$BACKEND"}
# --repo-branch is MANDATORY: the orchestrator dispatches from the repo
# root (main), so omitting it clones main on the gcp/auto lane and a
# per-issue driver script is absent (#595). Drop it ONLY if the workload
# is wholly on main (no issue-branch-only script).
```

`dispatch_issue.py launch` prints ONE JSON line on stdout with the
resolved outcome (`chosen_kind`, `requested_kind`, `reason`,
`pod_name`, `handle_sidecar_path`). On a router terminal it exits with
code `2` and the JSON carries `failure_class` + `status` + `note` so
the orchestrator posts `epm:failure v1` per the table above and
`set-status <N> blocked` — no re-derivation. On a non-terminal
provisioning error (RunPod SUPPLY_CONSTRAINT etc.) the underlying
backend raises and the helper either retries (RunPod's
`--wait-for-capacity` loop) or surfaces the failure as
`epm:pod-pending v1` so the user adjusts (capacity, intent override)
and re-runs `/issue <N>`. On exit code `75` (EX_TEMPFAIL) the JSON
carries `still_waiting: true` + `rerun: true` + `reason:
wait_for_capacity_budget_reached`: the RunPod lane's
`pod_lifecycle.py provision` hit its bounded wait-for-capacity
per-process wall-clock budget while capacity / the fleet burn cap kept
the provision queued. NOT a failure — the wait loop is state-free, so
RE-RUN the same `dispatch_issue.py launch` command to continue waiting
(post an `epm:progress v1` heartbeat per re-run so the watcher sees
liveness); NEVER post `epm:failure v1` / `set-status blocked` on this
exit (#603).

**Follow-up parent reuse.** When the task has a `parent_id` AND the
parent's RunPod pod is alive, the operational path stays on the
existing `pod.py` flow for that one specific case (the slice-6 router
does NOT yet model "reuse parent's live pod" — slice 7 wires the
reconnect path through the router uniformly):

```bash
PARENT_ID=$(uv run python scripts/task.py view <N> --json | jq -r '.frontmatter.parent_id // empty')
if [ -n "$PARENT_ID" ] && uv run python scripts/pod.py list-ephemeral --issue "$PARENT_ID" | grep -q epm-issue; then
  # Parent pod still alive — resume + reuse. Skip the router call;
  # this child task's run inherits the parent's pod_name.
  uv run python scripts/pod.py resume --issue "$PARENT_ID"
  # Record the assigned pod as epm-issue-$PARENT_ID in the launch marker.
else
  # Fresh launch through the router (the canonical path above).
  uv run python scripts/dispatch_issue.py launch \
      --issue <N> --intent "$INTENT" ${BACKEND:+--backend "$BACKEND"}
fi
```

**Slice-6 regression guard for the parent-pod-reuse branch (no
sidecar is written).** When the alive-parent path above fires (child
task with `parent_id` AND parent's RunPod pod still alive →
`pod.py resume --issue $PARENT_ID`), the dispatcher is NOT invoked, so
`.claude/cache/issue-<CHILD_N>-handle.json` is NEVER written.
Downstream that means: (1) Step 6d.2 MUST SKIP `backend_poll.py
--issue <CHILD_N>` — its missing-sidecar guard would post a
FALSE-POSITIVE `epm:failure v1` (`failure_class: infra`, `reason:
missing_handle_sidecar`) on a perfectly healthy child run; instead,
fall back to the legacy `poll_pipeline.py --pod epm-issue-$PARENT_ID
...` invocation for the duration of this child (the parent's pod
name + log path are the authoritative identifiers, NOT the child's
sidecar). (2) Step 8 MUST SUBSTITUTE the `dispatch_issue.py finalize
--issue <CHILD_N>` call with `pod.py terminate --issue $PARENT_ID
--yes` — terminating the parent's pod IS the correct operation here
(matching the existing teardown prose under Step 8), and the
finalize CLI would otherwise exit 2 on missing sidecar. Re-record
the parent's `epm:pod-terminated v1` against the child task so the
dashboard surfaces the terminate. Full reconnect-via-router
unification (write a sidecar even on the reuse path so every
backend / lane uses ONE Step 6d.2 + Step 8 code path) stays
slice 7 — this paragraph is the operational guard that prevents the
false-positive failure / mis-routed finalize until then.

**Autonomous mode (`EPM_AUTONOMOUS_SESSION=1`) — RunPod
`--wait-for-capacity` auto-enables.** When the router's chosen lane is
RunPod (explicit override `backend: runpod`), the underlying
`pod_lifecycle.py provision` reads `EPM_AUTONOMOUS_SESSION` itself and
turns on the unbounded SUPPLY_CONSTRAINT retry loop (exponential
backoff with full jitter, base 30s, cap 10 min, forever) — "the
experiment should start when it has space," not park-for-user.
"Unbounded" is across re-runs, not per process: each provision process
exits 75 (still-waiting) at its wall-clock budget and the dispatch CLI
surfaces that as `still_waiting: true` + exit 75 — re-run the same
launch command (see the exit-75 contract above), never treat it as a
failure. The
orchestrator should background the dispatch call (`Bash` with
`run_in_background=true`) so its own turn isn't blocked, and ON
periodic re-invocation (each bg-Bash output yield) it should scan the
captured stderr for `[wait-for-capacity] attempt N, waited ...` lines
and post one `epm:progress v1` marker per heartbeat (note:
`"pod-provision waiting for capacity: attempt N, waited ..."`). This
keeps `autonomous_session_watch.py` (6h stale-marker threshold) seeing
liveness. **Interactive sessions still fail fast** —
`--wait-for-capacity` defaults OFF so a human running `pod.py provision`
from a shell sees no-capacity immediately and can decide whether to
wait, switch DC, or change GPU intent.

**Stale-port recovery — `pod.py config --refresh-from-api`.** When an
`epm:pod-pending v1` is followed by a long stretch of failing SSH
polls (`poll_pipeline.py` reporting `status=dead` every tick on an
otherwise live pod), the most common cause is that a
SUPPLY_CONSTRAINT-blocked resume eventually brought the pod back at a
NEW SSH port via a retry path that bypassed `_upsert_pods_conf`, so
`pods.conf` still carries the pre-stop value while the live RunPod API
has the fresh one. The canonical first response is `uv run python
scripts/pod.py config --refresh-from-api pod-<N>` — pulls fresh
host/port from the live API into `pods.conf` + `~/.ssh/config`. The
auto-heal also fires automatically: `poll_pipeline.py`
counts consecutive SSH-probe failures and fires `--refresh-from-api`
after ten consecutive failures (~3-4 min at 20s spacing), and
`autonomous_session_watch.py` fires it once per stalled episode when a
stalled session has a RUNNING managed pod. Both auto-fires are
fail-soft and dedup'd so the manual command stays the surgical
recovery move; reach for it when the auto-heal has not yet tripped or
the issue is unambiguously a port drift. See `.claude/rules/upload-policy.md`
context on the Authority split (live API authoritative for host/port,
`pods.conf` the on-disk source for SSH/MCP). (#488 spun for 13+ hours
before the manual subcommand existed.)

The pod / job / VM name passed downstream is recorded in the sidecar
JSON the router writes (RunPod: `pod-<N>`; SLURM: `eps-issue-<N>`;
GCP: `eps-issue-<N>`). The experimenter does NOT pick or create pods.

#### Step 6c: Preflight on resumed pods

`provision` already ran preflight as its last bootstrap step. For
*resumed* pods, re-run preflight explicitly because the volume is intact
but the container restart may have left stale state:

```bash
ssh_execute(pod=epm-issue-<N>, command="cd /workspace/explore-persona-space && uv run python -m explore_persona_space.orchestrate.preflight --json")
```

Parse JSON. (#554 made preflight branch-aware — the old
behind-origin condition is now a WARNING, so on current code an
`ok=false` here is a real failure.)
If `ok=false`, post `epm:preflight v1` event with the
errors/warnings, then post the §5 marker:
```bash
uv run python scripts/post_step_completed.py --issue <N> --step 6c \
  --exit-kind failure-exit --notes "preflight failed; user must fix"
```
EXIT. User fixes, re-runs.

#### Step 6d: Dispatch experimenter (launch-only), then orchestrator polling loop

The experimenter agent is **launch-and-exit only** — it syncs the pod,
preflights, launches the job via its setsid launcher script
(`experimenter.md` § "During Execution"), posts `epm:run-launched`, and
exits its turn within ~60 seconds. The orchestrator (this skill) owns
all subsequent monitoring via a bg-Bash polling loop chained through
`scripts/poll_pipeline.py`. This split is mandatory: subagents have ONE
turn and are NOT auto-re-invoked when bg work completes, whereas the
orchestrator IS auto-re-invoked on every bg-Bash exit (see `CLAUDE.md`
§ "Subagent vs orchestrator re-invocation semantics").

##### Step 6d.0: Smoke/sweep architecture parity gate

Fires once per implementer round, AFTER all of Step 6a-6c (HF gate,
pod provision, preflight) and BEFORE Step 6d.1 (experimenter dispatch).
Reads the highest-version `epm:smoke-architecture-check v<n>` marker
posted by the implementer in the current round (see
`experiment-implementer.md` "Before writing code" item 5 and
workflow.yaml § markers `epm:smoke-architecture-check`).

Verdict routing:

| `verdict` | Action |
|---|---|
| `PASS_UNIFIED` | Advance to Step 6d.1 — smoke IS sweep with one cell; the architecture is unified end-to-end AND every planned arm resolved REAL or N/A. |
| `PASS_CANARY canary_cell=<id>` | Advance to Step 6d.1 — paths diverge but the plan §4 Design justifies the divergence in two sentences AND names the canary cell that exercised the sweep path during smoke. Log to chat: `divergence accepted; canary cell <id> exercised the subprocess path during smoke`. |
| `PASS_PARTIAL arms_stubbed=<comma-list>` | **REFUSE to dispatch.** Bounce back to status:planning; re-invoke `/adversarial-planner` with pivot scope: "arms {arms_stubbed} resolved to fallback/stub in smoke — phase coverage + import-resolution passed BUT ≥1 planned arm is not exercising its production computation path; resolve them in the diff, OR re-authorize the stubs in §4 Design (canary-like exception, not yet wired)." Round counter does NOT increment (strategy pivot, mirroring `FAIL_NO_CANARY`). |
| `FAIL_NO_CANARY` | **REFUSE to dispatch.** Bounce back to status:planning; re-invoke `/adversarial-planner` with pivot scope: "the smoke/sweep architectural divergence has no justification + canary; re-architect toward UNIFICATION (smoke = sweep with one cell), OR add the two-sentence justification + named canary cell to §4 Design." Round counter does NOT increment (this is a strategy pivot, not a fresh review round). |
| (marker missing) | **REFUSE to dispatch.** Bounce back to implementer with a one-line prompt: `post epm:smoke-architecture-check v1 per the mandatory checklist before code-review-PASS`. |

<!-- gate: gates.inline.smoke_architecture -->

The gate is enforced inline (gates.inline id=10) — the implementer
self-tags at report-time; the orchestrator validates here.

Rationale (#397): consecutive rounds all PASSed smoke and crashed
sweep immediately because smoke ran in-process while sweep ran a
subprocess. This gate forces the divergence to be explicit at plan
time so the pre-dispatch moment catches it, not the third pod-side
crash.

##### Step 6d.0-bis: End-to-end smoke gate (multi-phase data-gen pipelines)

For an experiment whose driver chains ≥2 distinct production phases
before the first GPU launch — data-gen, training, eval, verify, upload
(typically gen → drift → train → eval → aggregate; a datagen → train →
verify → upload driver like #906's equally qualifies) — the
architecture-parity gate above is NOT enough: it checks
that smoke and sweep share ONE code path, not that EVERY phase ran. A
resume-skip design serializes bug discovery — each pod cycle surfaces
the next phase's bug — so before the GPU production launch the FULL
pipeline must have executed once at tiny N (≈2-3 rows, 1 cell, 1 seed)
so EVERY phase runs end-to-end on CPU / 1-GPU.
The tiny-N pass MUST meet the **tiny-real standard** for `kind:
experiment`/`batch` drivers: it executes the production path with REAL
library types at every internal seam the pipeline actually has —
real tokenizer, real train engine + config builders + callbacks, real
adapter round-trip, real verify/upload bodies (an API-only driver has
no train engine; its own real seams bind instead) — faking ONLY
GPU-scale weights (a from-config tiny same-arch model over the real
vocab-id space) and the remote Hub boundary (signature-bound). A
seam-stubbed / mocked smoke does NOT satisfy this gate: mock-seam
smokes surface shape bugs one per GPU cycle (#906 r11-r14: four
distinct production shape bugs, four ~1.5h pod cycles, every mocked
smoke green beforehand; r15 = the tiny-real pivot). Worked example:
`tests/test_issue906_tiny_real_e2e.py`; full recipe + the two CPU
traps: `.claude/rules/gotchas.md` "Mock-seam smokes surface production
shape bugs ONE PER GPU CYCLE".
When the pipeline INGESTS a real corpus (a WildChat/LMSYS-class
streaming builder), the standard's **data-ingestion probe class**
(#1092) binds too: a bounded tiny-real streaming probe against the
REAL dataset — a kept cap AND a TOTAL-streamed-rows cap, asserting
kept > 0 per dataset — with per-filter rejection counters in the
stream's `done:` line; a synthetic-fixture smoke alone does NOT
satisfy the ingestion phase (a filter chain written from assumed
field shapes can reject 100% of real rows while every synthetic
smoke stays green). Recipe + verified field shapes:
`.claude/rules/gotchas.md` "Real-corpus streaming filters" +
`.claude/agent-memory/experiment-implementer/feedback_real_corpus_streaming_filters_tiny_real_probe.md`.
When the driver spans MULTIPLE ARM CLASSES — distinct source-context
classes / recipe branches (e.g. persona-context vs bare-context arms)
AND every other class-defining axis the grid crosses: behavior class
(marker vs content), training regime (contrastive `con` vs
positive-only `po`), method (LoRA vs full-FT) (#1586) — "once at tiny N" means
once PER ARM CLASS: ≥1 tiny cell per realized (class × regime)
combination, reaching class-gated read-side / aggregation paths
(panel-disjointness reads, per-class mix asserts, reuse-seam loaders)
too, not only the train phase. Per-arm seams
(source-context construction, negative-panel assembly, `ModelOrganism`
wiring) are invisible to a single-arm smoke however tiny-real its
seams (#1090 fu5).
Recipe: `.claude/rules/gotchas.md`
"A single-arm smoke is blind to per-arm seams" +
"Smoke/production parity includes REGIME/CLASS COVERAGE".
Confirm the implementer's
`## Smoke run` report (per `experiment-implementer.md` § "End-to-end
smoke run PER PHASE") carries a sub-section with exit code `0` + an
artifact digest for EACH phase the pipeline executes — not just training
or data-gen. Any phase missing, or showing only `--help` / `import` /
`--dry-run` / seam-stubbed (mocked internal seams — the tiny-real
standard's two sanctioned fakes excepted) / synthetic-fixture-only
(a real-corpus ingestion phase with no tiny-real streaming-probe
evidence) evidence → **REFUSE to
dispatch**; bounce to the implementer
with `run the full gen→...→aggregate pipeline once at tiny N before
production`. FAIL blocks production. (Origin: #408 — a multi-phase
data-gen pipeline never smoke-tested end-to-end leaked 5+ distinct bugs
one-per-pod-cycle over ~41h idle.)

Orthogonal to the smoke-gate above, the experimenter agent itself
enforces an **input-data completeness gate** as the first step in its
pre-launch protocol — verifying that the input-data files the
dispatcher will read from local disk on the pod are ALL present, and
posting `epm:failure v1 failure_class: infra reason:
planned-input-data-missing-on-pod` (no launch) on any shortfall. The
smoke gates check code paths and phase coverage; the input-data
completeness gate checks that the dependency files actually exist on
the pod. See `experimenter.md` § "Before Running" item 4 for the
mechanic (#468). The orchestrator does not need to
re-verify here — the routing on shortfall ends in an `epm:failure
failure_class: infra` that flows through Step 7's respawn path
naturally.

##### Step 6d.1: Spawn experimenter for launch

**Pre-dispatch state sanity (fires on EVERY dispatch — first launches
AND re-launches).** Immediately before spawning the experimenter,
re-verify the brief's two load-bearing assumptions against LIVE state —
never against this session's cached view (a concurrent / replacement
session may have finished the run while this session was mid-review):

1. **Compute exists.** For a RunPod-backed dispatch, `uv run python
   scripts/pod.py list-ephemeral --issue <N>` must show the assigned
   pod; for other backends, verify the brief's compute target is live
   per the handle sidecar / backend status (Step 6b). Absent → do NOT
   dispatch; re-derive scope from the markers (the run may already be
   done) or re-provision via Step 6b.
2. **Run still pending.** `uv run python scripts/task.py latest-marker
   <N>` + the recent `events.jsonl` tail: if `epm:results v<n>` +
   `epm:upload-verification PASS` (or `epm:pod-terminated v1`) postdate
   the failure being recovered, the (re)launch is STALE — the work
   already completed. Do not dispatch; reduce the brief to the genuinely
   missing artifact, or skip the dispatch entirely and resume from
   wherever the markers say the task actually is (Step 7+ / Step 9
   routing).

On either mismatch, re-derive the brief from the live markers instead
of dispatching the stale one. This is the dispatch-site analogue of the
Step 0 stale-wake ownership re-check and the Step 9 entry guard's
marker-freshness pattern. (#559)

**3. External markers triaged.** Run the Step 9 entry guard
§ Pre-dispatch external-marker triage check before spawning. Pod/backend
launches post no `stage-dispatch` breadcrumb, so the
`external-markers triaged:` line goes in an `epm:progress` note posted
immediately before the experimenter spawn.

**4. Fresh-provision RunPod launches run in orchestrator bg-Bash, NOT in
the experimenter.** A cold `dispatch_issue.py launch --backend runpod`
runs 25-50 minutes on the RunPod lane (`podFindAndDeployOnDemand` create
+ `wait_for_ssh` up to ~10 min + `bootstrap_pod.sh`'s 11 steps including
a 2.8 GB shallow clone through MooseFS + `uv sync --locked` + flash-attn
build + preflight — the wedge classes in `.claude/rules/gotchas.md`
document the wall-time). A subagent's turn cannot survive that: a
`Bash(run_in_background=true)` dispatched inside the experimenter dies
when the experimenter's ~60 s turn ends (the #1689 R8 failure shape —
the subagent bg-Bash died mid-bootstrap, steps 5-11 never ran, the pod
sat on `main` with no `/workspace/logs/` and no workload). So when the
pod is NOT yet bootstrapped, the orchestrator dispatches
`scripts/dispatch_issue.py launch` in its OWN
`Bash(run_in_background=true, timeout=600000, command="uv run python
scripts/dispatch_issue.py launch --backend runpod ...")` — the harness
re-invokes the orchestrator when this bg-Bash exits, so the
orchestrator SURVIVES the 25-50 min wait by design. ONLY after (a) the
handle sidecar (`.claude/cache/issue-<N>-handle.json`) exists AND (b)
the `experimenter.md` § "Post-dispatch bootstrap-completeness probe"
passes on the pod (uv.lock + .venv/ + preflight-OK signals) is the pod
eligible for a WORKLOAD-launch experimenter spawn (the 60 s
launch-and-exit contract of `experimenter.md` § "Contract scope —
already-bootstrapped pod only"). Never brief the experimenter with a
cold `dispatch_issue.py launch` command; it will refuse and post
`epm:failure v1 failure_class: infra reason:
fresh-provision-in-subagent` per that same Contract scope (#1689).

Spawn `experimenter` subagent via `Agent()`. Brief:
- The plan path (the `plans/plan.md` symlink) + the code-reviewed
  branch (`issue-<N>`)
- Pod name (`epm-issue-<N>` or parent's)
- The exact workload command from the plan's Reproducibility Card (the
  workload/dispatcher invocation plus any required env-var pins; the
  experimenter wraps it in its canonical setsid launcher script —
  `experimenter.md` § "During Execution". NEVER put a literal top-level
  `source .env` + bare `nohup`-backgrounded launch line in the brief:
  the SSH-MCP shell is `sh` (`source: not found`, #545) and an
  un-setsid'd background launch risks SIGHUP reaping (#444/#541; the
  #841 brief carried exactly this shape and the experimenter had to
  deviate)
- When the plan names a "regenerate locally via prep script"
  prerequisite (e.g. the Turner JSONLs): the prep-script invocation AND
  its OUTPUT dataset path(s), so the experimenter's input-data gate
  (`experimenter.md` § "Before Running" item 4) stat-checks the files
  themselves — a secret/env-var presence check alone does not cover
  them (#545)
- Required: post `epm:run-launched` with `pod=<name> pid=<pid>
  log_abs=<absolute_log_path> cmd='<dispatch>'` in
  the note, then exit cleanly within 60 seconds. The `log_abs=` field
  MUST be an absolute path (use `realpath` or `os.path.abspath()` on
  the pod) AND the experimenter MUST verify the file exists with
  `ssh_execute ls -la <log_abs>` before posting.
- Explicit: do NOT sleep-chain, do NOT monitor — the orchestrator polls
  the run

**NEVER include pod lifecycle commands (provision, stop, resume,
terminate, cleanup) in the experimenter brief.** Pod termination
happens automatically in Step 8 (after upload-verification PASS).
**NEVER include progressive monitoring instructions** in the brief —
those are obsolete (see the deprecated memory
`feedback_subagent_sleep_chain.md`).

Wait for the experimenter to return. The return must include the
`epm:run-launched` marker. Parse it for `pod`, `pid`, and the log path
(`log_abs=`; the legacy `log=` fallback is RETIRED — a marker missing
`log_abs=` is a launcher bug, fail loud).

If the experimenter posted `epm:failure v1` instead (launch-time
crash), skip the polling loop and proceed to Step 7's failure-
classification routing.

Post `epm:launch v1` containing:
- Worktree path, branch, PR URL, code-review verdict (`PASS`)
- Pod + PID + log path
- WandB run URL (best-effort)

##### Step 6d.2: Orchestrator polling loop (bg-Bash chained)

Enter a polling loop that runs in THIS orchestrator's context. Each tick
delivers ONE tick-JSON line via one of two harness-re-invocation sources
— either a bg-Bash exit (the fixed 540s chain, the default) or the
#1924 quiet-wait Monitor's terminal-stdout notification (the sanctioned
long-wait shape when `next_interval == 1800`). Both sources feed the
SAME `result` variable and route identically:

**Trigger-dense tag adoption (at loop entry, BEFORE the first tick —
#1587; the producer side of the #1556/#1574 digest chain).** Apply the
`.claude/rules/trigger-dense-review.md` recognition heuristic
("Recognition heuristic (any one suffices)") to THIS run's workload —
the task body's target/scope lines, the plan §4 targets, and the
round's realized diff pathspec / training-eval data sources. If ANY
recognition class fires and the task does not already carry the tag,
adopt it now:

    uv run python scripts/task.py add-tag <N> trigger-dense

`add_tag` is idempotent (an already-tagged task no-ops with no commit),
and every consumer reads the tag FRESH per tick
(`backends/excerpt_digest.py::issue_trigger_dense` — RunPod
`poll_pipeline.py` and the GCP/SLURM lane tails alike), so loop-entry
adoption lands before the first poll tick on every lane, and a
respawned session re-entering this loop self-heals a missed adoption.
Negative case: when NO recognition class fires, do NOT tag — the
digest replaces raw log tails, so a false-positive tag costs log
visibility on a healthy run; a wrongly-adopted tag is reversible
mid-run (`uv run python scripts/task.py remove-tag <N> trigger-dense`,
effective next tick). This persists the Step-0 recognition (the #1563
guard-surface turn discipline) as a durable marker: successor
sessions and the poll-tick digest consumers (#1556/#1574) key on
the tag; review-round brief composition still applies the rule's
"Fires when" heuristic per turn — the tag is a durable floor,
never a substitute for it.

```python
result = None  # parsed JSON line of the PREVIOUS poll tick; None before the first tick
while True:
    # MANDATORY: refresh the title + self-report at the TOP of every
    # tick so the dashboard / happy-ls / phone title stay current with
    # the loop's `running` status (or the latest phase if the poller
    # posted one). This is the cheap path — no LLM call — and keeps
    # `~/.eps-autonomous/issue-progress/<N>.json` fresh under the
    # summarizer's 20-min freshness window. `set_title` is the soft-fail
    # helper defined in the "Chat title updates" section above; it
    # NEVER crashes the loop.
    set_title(N, current_phase)  # e.g. "running" / "phase: post_eval"

    # The bg-Bash poller is `scripts/backend_poll.py` — it reads the
    # per-issue handle sidecar at `.claude/cache/issue-<N>-handle.json`
    # (written by `issue_dispatch.dispatch_for_issue` in Step 6b),
    # resolves the right `ComputeBackend` from `handle.backend`, calls
    # `backend.poll(handle)`, and prints ONE JSON line whose shape is
    # byte-identical to the legacy `poll_pipeline.py` output (the
    # `backends.base.PollResult` fields). The orchestrator's existing
    # JSON-line parser is interchangeable across backends — no per-
    # backend branches here.
    #
    # On the RunPod path `backend.poll` delegates to
    # `scripts.poll_pipeline.poll_once` (the battle-tested probe);
    # `backend_poll.py` is the uniform bg-Bash entry, NOT a
    # re-implementation. The legacy `--pod` / `--log` / `--pid-file`
    # CLI args of `poll_pipeline.py` are recovered from the handle
    # sidecar by `backend.poll`, so the bg-Bash command line shrinks
    # to a single `--issue` argument.
    #
    # CAVEAT — parent-pod-reuse child tasks: when this is a child task
    # whose parent's RunPod is still alive AND the alive-parent branch
    # in Step 6b fired, NO sidecar was written for the child. SKIP
    # this bg-Bash `backend_poll.py --issue {N}` entirely and fall
    # back to `poll_pipeline.py --pod epm-issue-$PARENT_ID ...` for
    # the duration of the child. See the "Slice-6 regression guard
    # for the parent-pod-reuse branch (no sidecar is written)"
    # paragraph in Step 6b for the full rationale + the failure mode
    # the unconditional invocation would trigger (FALSE-POSITIVE
    # `epm:failure v1 missing_handle_sidecar`).
    # ADAPTIVE POLL INTERVAL (anti-stall redesign §7) — bg-Bash sleeps
    # HARD-CLAMPED AT 540s PER CALL (#1818). Every tick's JSON line
    # carries a recommended `next_interval` (seconds): 1800 ONLY on a
    # healthy, quiet, post-early-run `running` tick far from any phase
    # boundary; 540 otherwise — gate-adjacent, anomalous, early-run
    # (first ~30 min after launch), and recent-phase-change ticks never
    # emit the long value, so gates are never delayed. The
    # recommendation is NEVER a bg-Bash sleep value: the Bash tool
    # kills ANY call at its 600000 ms (10-minute) ceiling — background
    # calls included — so a composed `sleep 1800` dies mid-sleep, the
    # poll never runs, and the dead call reads as a stale/absent poll
    # on the next wake (#1768).
    # NEVER compose a sleep longer than 540s into a single background
    # Bash call, here or anywhere in this loop. A quiet-tick 1800 recommendation
    # (POLL_INTERVAL_QUIET_SEC) is instead REALIZED as the one-wake
    # Monitor QUIET-WAIT branch below (#1924) — the sanctioned
    # long-wait shape (§ Long-phase heartbeat duty) running
    # wait-then-poll in ONE unit; a missing, unparseable, or non-1800
    # `next_interval` falls to the fixed 540s chain (fail toward
    # coverage).
    #
    # `result` below = the parsed JSON line from the PREVIOUS tick — either
    # the bg-Bash exit's stdout (fixed 540s else-arm) OR the quiet-wait
    # Monitor's terminal-stdout notification (#1924 branch, the same
    # `result` the status branch below reads); its `next_interval` field
    # is the quiet-wait branch key — it never sets a bg-Bash sleep.
    quiet_wait = (
        result is not None
        and result.get("status") == "running"
        and result.get("next_interval") == 1800
    )
    if quiet_wait:
        # §7 quiet cadence, realized as the sanctioned Monitor long wait
        # (#1924): the Monitor runs wait-then-poll in ONE unit, so its
        # terminal stdout line IS the tick JSON and the quiet cycle
        # costs ONE notification wake (vs ~3.3 fixed-540s wakes).
        # Hard-bounded — a timeout kill is itself reported — and the
        # */45 issue-tick cron + the watcher's 10-min passes stay the
        # independent external bounds (§7 risk paragraph). The terminal
        # JSON line doubles as the #1850 emission: the wait never
        # exceeds ~40 min, so no mid-wait heartbeat wake is burned.
        # 60s-chunk wait loop — never a bare long leading sleep. The
        # top-of-tick set_title refresh stays unconditional (it runs at
        # the resume that reads each tick, quiet-wait resumes included).
        Monitor(
            description=f"quiet-wait issue {N} (§7 quiet cadence, ~29 min + poll)",
            timeout_ms=2400000,
            persistent=False,
            command=(
                f"for i in $(seq 1 29); do sleep 60; done; "
                f"uv run python scripts/backend_poll.py --issue {N}"
            ),
        )
        # End the turn. The notification carries the tick JSON (the
        # LAST stdout line) — parse it per § Tick-parse
        # field-preservation below and route exactly as below (re-arm
        # the quiet wait or the 540s chain per the fresh tick's
        # fields). A Monitor exit with NO parseable JSON line (poll
        # crash; the reported nonzero exit is the signal) -> run an
        # IMMEDIATE fresh 540s-chain tick — never re-arm the quiet
        # wait blind. A vanished Monitor (no notification AND no poll
        # for >~40 min) surfaces at the next */45 tick-cron wake:
        # kill-before-relaunch probe, then resume the 540s chain.
    else:
        interval = 540  # fixed: both the default AND the per-call MAX (#1818)
        Bash(
            run_in_background=True,
            command=(
                f"sleep {interval} && uv run python scripts/backend_poll.py --issue {N}"
            ),
        )
    # Harness re-invokes orchestrator on bg-Bash exit OR quiet-wait Monitor
    # notification (#1924 — the wait+poll runs in one unit, so the
    # Monitor's terminal stdout IS the tick JSON). To WAIT on bg
    # work, simply END THE TURN with a one-sentence status — NEVER emit
    # no-op Bash calls to idle (`sleep 1` "yield turn", `true` no-ops):
    # each burns a tool call + context for nothing (33x and 49x in two
    # sessions). Read the JSON line from stdout — the LAST line
    # of either source (bg-Bash exit output or the quiet-wait Monitor
    # notification, #1924) — parse per § Tick-parse field-preservation
    # below; a status-only parse is BANNED. Decide:
    #
    #   status == "done"           -> exit loop; transition to status:verifying; go to Step 7.
    #   status == "gate"           -> a pod-side sentinel carried a non-empty
    #                                  `gate` field; the poller has ALREADY
    #                                  posted the carried marker (e.g.
    #                                  `epm:fact-candidates v1`) from the local
    #                                  VM as part of its sentinel drain — do
    #                                  NOT re-post it. Read result["gate"],
    #                                  exit the polling loop, and dispatch the
    #                                  matching gate handler per Step 6d.4
    #                                  below (PARK for a user gate like
    #                                  `fact-candidates`, AUTO-RESOLVE +
    #                                  resume the loop for `pv_phase1_done`).
    #   status == "stalled" | "dead" -> post epm:failure v1 with failure_class
    #                                   inferred from log_tail_excerpt
    #                                   (run scripts/failure_classifier.py on
    #                                   the excerpt, ALSO forwarding the tick's
    #                                   result["stall_reason"] via
    #                                   --stall-reason — a silent hang's log
    #                                   tail carries no infra pattern, so the
    #                                   stall_reason is the only routing
    #                                   signal; see Step 7); run CRON-TEARDOWN
    #                                   (see below); set status:blocked; exit.
    #   status == "running"        -> milestone-already-posted by the poller
    #                                  if new_milestone was true; loop again:
    #                                  the next tick routes via the
    #                                  QUIET-WAIT Monitor branch when this
    #                                  tick's next_interval is 1800, else
    #                                  the fixed 540s sleep (see ADAPTIVE
    #                                  POLL INTERVAL above; never sleep
    #                                  >540s in one call).
    #                                  If the JSON also has
    #                                  gpu_idle_advisory_posted == true, act
    #                                  per "GPU-idle advisory handling" below
    #                                  before the next tick. If it has
    #                                  gpu_idle_escalation_posted == true, act
    #                                  per "GPU-idle escalation handling" below.
```

**Tick-parse field-preservation (REQUIRED — #1841; incident #1768).** Any
compacted/filtered parse of a tick's JSON line MUST print, at minimum, the
full decision field set: `status`, `current_phase`, `gate`, `stall_reason`,
`new_milestone`, `next_interval` (the quiet-wait branch key), `gpu_idle_advisory_posted`,
`gpu_idle_escalation_posted`, `gpu_width_advisory_posted`,
`eta_deviation_posted`. A status-only parse is BANNED — it structurally
discards the very fields the handling sections below branch on (#1768). Use
`d.get(...)` for every field (a mixed-vintage poller may omit newer fields —
degrade to None, never KeyError). Canonical one-liner:

```
... | uv run python -c "import json,sys; d=json.loads([l for l in sys.stdin.read().splitlines() if l.strip()][-1]); print('TICK:', ' '.join(f'{k}={d.get(k)}' for k in ('status','current_phase','gate','stall_reason','new_milestone','next_interval','gpu_idle_advisory_posted','gpu_idle_escalation_posted','gpu_width_advisory_posted','eta_deviation_posted')))"
```

**Forensics-ingest discipline (#1546):** on a stalled/dead tick — and in any
post-crash forensics this loop or Step 7 performs — ingest failure text per
`.claude/rules/trigger-dense-review.md` § Orchestrator poll/forensics turns:
structural digests (counts + file references), classifier-side routing, a
fresh-context reader for trigger-dense runs, hook-BLOCKED output by
reference.

(`current_phase` is `"running"` by default; when the poller emits a
milestone marker like `phase: post_eval`, update the local
`current_phase` from the milestone before the next tick so the title
reflects the latest phase.)

The top-of-tick `set_title` refresh plus the bounded tick cadence (the
fixed 540s chain; ≤ ~40-min Monitor quiet-wait cycles) discharge the
§ Long-phase heartbeat duty (below) for this loop by construction; any
wait run OUTSIDE this loop shape — a `Monitor` until-loop on a VM
phase, an ad-hoc bg poll chain, an off-pod Batch-API poll — carries
that duty explicitly.

The `poll_pipeline.py` helper posts `epm:progress` events itself when it
sees a phase transition, AND drains pod-side sentinel files (posting
their carried markers from the VM via `task_workflow.post_event`). The
orchestrator's only post-tick duties are: exit the loop on `status=done`,
dispatch the matching gate handler on `status=gate` (Step 6d.4 — PARK for
a user gate, AUTO-RESOLVE + resume the loop for `pv_phase1_done`), and post
`epm:failure v1` on `status=stalled` or `status=dead`. The orchestrator
NEVER re-posts a marker the poller already posted from a sentinel —
double-posting is the failure mode the gate path is designed to avoid.
On the terminal `status=done` tick (the point where `epm:results` is
posted/observed), the next action after the `uploading` transition is
Step 8's **Results-landed parallel spawn** block — dispatch that
concurrent batch, NOT the old serial verifier-then-analyzer order (see
Step 8 for the block's contents and hard joins; do not re-derive them
here).

**GPU-idle advisory handling.** When a tick's JSON reports
`gpu_idle_advisory_posted: true`, the poller has just posted a one-time
`epm:progress` marker whose note starts with `[gpu-idle-advisory]` (plus a
`gpu_idle_advisory=True` extra): every GPU sat idle on a HEALTHY
`status=running` tick for ≥ `EPM_GPU_IDLE_ADVISORY_MIN` (default 30) min —
the signature of a long CPU-only phase holding a GPU pod
(#518/#537). Don't just loop: surface the advisory in the session text,
then check the plan for whether the REMAINING work in the current phase is
CPU-only. If it is and the remaining CPU stretch is long (>~30 min), apply
CLAUDE.md "CPU-only phases don't hold GPU pods": checkpoint the phase's
state, upload the artifacts it reads, move the phase off-pod to the VM,
and `pod.py stop` the pod once nothing pod-local is needed. Three hard
constraints: (a) NEVER kill un-checkpointable in-RAM work to save idle GPU
time — redoing #518's multi-hour un-checkpointed scoring run would have
cost more than the idle burn; let such a phase finish and fix the
checkpointing in a follow-up; (b) autonomous sessions never stop a pod to
PARK — the off-pod move is valid only when the CPU phase keeps running
toward the Goal in this session (e.g. on the VM); (c) this is the
CPU-phases-off-pod rule, NOT a mid-run cost gate — the trigger is the
advisory's idle-GPU fact, never "this is getting expensive". If the phase
genuinely needs the pod (a pod-local data dependency) or is nearly done,
state that one-line reason and keep looping. The advisory never changes
the status verdict, so this handling is additive to the `status=running`
branch.

**GPU-idle escalation handling.** When a tick's JSON reports
`gpu_idle_escalation_posted: true`, the poller has just posted a louder
one-per-phase `epm:progress` marker whose note starts with
`[gpu-idle-escalation]` (plus a `gpu_idle_escalation=True` extra) AND fired a
best-effort Telegram push: a MULTI-GPU pod has been idle in an upload/CPU-only
phase for ≥ `EPM_GPU_IDLE_ESCALATION_MIN` (default 60, ≥ the advisory min) min
— the #664 spend-leak class (a multi-GPU pod idling through a terminal
upload phase). The orchestrator's response is the SAME as for
`gpu_idle_advisory_posted` (the escalation is the advisory's louder second
tier, not a new action): surface it in the session text, and if the remaining
work in the current phase is genuinely CPU-only and long, apply CLAUDE.md
"CPU-only phases don't hold GPU pods" — route the upload off-pod / release the
GPUs after a checkpoint — under the SAME three hard constraints as the advisory
(never kill un-checkpointable in-RAM work, autonomous sessions never stop a pod
to PARK, it is NOT a mid-run cost gate). Like the advisory, the escalation
NEVER changes the status verdict and the poller NEVER stops the pod — it
surfaces the leak loudly for action. This handling is additive to the
`status=running` branch.

**ETA-deviation / GPU-width advisory handling.** When a tick's JSON reports
`eta_deviation_posted: true`, the poller has just posted an
`epm:compute-deviation` marker (`source: poller`, `basis: elapsed-vs-plan`):
elapsed wall-time for the current phase or the whole run exceeded
`EPM_ETA_DEVIATION_MULT` (default 2.0) × the plan §9 `planned_wall_h` TOTAL —
the #763 class (an ~80× overrun a human caught ~16h late). Surface it in the
session text and weigh the run's remaining value: whether the plan's own
compute-deviation machinery should engage — a mid-run `continue_as_is`
acknowledgment, or a deliberate descope ONLY where the planner's §9
stratification spec permits one. For a fit / battery / factorization phase,
the vectorize mid-run trigger applies FIRST — run the signature check
immediately, do not wait for a second deviation
(`.claude/rules/vectorize-many-cell-fits.md` § Mid-run trigger), and on a
NEGATIVE signature over an embarrassingly-parallel unit grid run that
section's width re-evaluation before resolving (a negative signature
settles vectorization, not width — #1092); the
`continue_as_is` bias below scopes to the descope question. Elapsed-so-far is a lower bound on final
wall, so `continue_as_is` is nearly always the right mid-run resolution; the
poller variant carries no `action:` field and is never an auto-descope input.
When a tick's JSON reports `gpu_width_advisory_posted: true`, the poller has
posted a `[gpu-width-advisory]` `epm:progress` marker (plus a
`gpu_width_advisory=True` extra): a STABLE strict subset of GPUs sat idle ≥
`EPM_GPU_WIDTH_ADVISORY_MIN` (default 45) min on a multi-GPU pod while the run
is healthy — the #813 idle-width / #664 spend-leak class. Apply the CLAUDE.md
per-phase GPU-WIDTH right-sizing judgment (widen the parallelism to fill the
pod, or release/downsize it) under the SAME three hard constraints as the
idle advisory: (a) never kill un-checkpointable in-RAM work, (b) autonomous
sessions never stop a pod to PARK, (c) this is NOT a mid-run cost gate — the
trigger is the idle-width / elapsed-vs-plan fact, never "this is getting
expensive". Both are advisory-only: neither changes the status verdict, and
the poller stops nothing. This handling is additive to the `status=running`
branch.

**Same-phase rate/ETA duty (#1863; incident #1482).** When ≥3 consecutive
poll ticks report the SAME `current_phase` with no `new_milestone`
(≈25–30 min at the fixed 540 s tick), a
phase-name liveness read is no longer enough — the orchestrator MUST
compute a throughput read instead of echoing phase-name liveness
indefinitely. Phase-label equivalence: a phase label differing only in an
advancing numeric/progress token (`E2 upload at shard17` vs
`E2 upload at shard23`, `cell 4/24` vs `cell 7/24`) is the SAME phase for
this trigger — and that advancing token IS the progress counter to use.
Input availability: on the trigger tick and every subsequent same-phase
tick, the compacted #1841 tick parse ADDITIONALLY prints
`log_tail_excerpt` (the #1841 field set is a minimum, so printing more is
legal), or the orchestrator re-reads the tick's raw JSON line for it —
the rate read's input must actually be in context, or the no-counter
fallback below silently swallows the duty. The duty: extract the phase's
monotonic progress counter from the tick evidence (the advancing label
token, `log_tail_excerpt`, or a sentinel progress field — `shard NN`,
`file K/M`, `cell i/N`), compute `rate = Δunits / Δwall` over the
same-phase tick window, and project `ETA ≈ remaining units ÷ rate`.
Record ONE `[phase-rate]` line in the session text and in the NEXT
periodic liveness `epm:progress` note — once per liveness note, not per
tick (this reuses `epm:progress`; NO new marker kind). Routing: this is a
detection duty only, NOT a new gate — auto-continue is preserved; a
pathological projection routes through EXISTING machinery: the
compute-deviation / vectorize mid-run trigger for fit / battery /
factorization phases (`.claude/rules/vectorize-many-cell-fits.md`
§ Mid-run trigger), and CLAUDE.md "CPU-only phases don't hold GPU pods" +
the #1824 bulk `upload_folder` recipe for per-file upload tails — never
keep echoing "healthy" against a multi-hour projection. No-counter
fallback: when no progress counter is readable from the tick evidence,
state once `no progress counter readable — liveness only` and treat the
absence as a signal to add a per-unit progress line (the
pod-side-reporting.md / code-style.md per-unit progress-line convention)
on the next code round. Worked example (#1482): five consecutive 30-min
ticks each reported "Healthy — E2 upload at shardNN"; the first actual
rate read gave ~98 files/h ⇒ a ~33 h projection for the remaining files,
by which point ~5.4 h of idle-A100 billing had already accrued; recovery
(one bulk `upload_folder` commit, the #1824 fix) took ~1 h.

**Per-lane planned-cell reconciliation (on every lane/phase completion —
#1481).** Planned-vs-actual coverage already has a terminal check
(After-Every-Experiment item 8 / `verify_task_body.py` check 11b / the
clean-result-critic planned-vs-actual lens), but it fires only at
clean-result time; during a multi-lane run a lane that completes WITHOUT
covering a planned cell is invisible until terminal analysis (#1481). So:
whenever the orchestrator observes that a LANE or PHASE of the run
completed — a poll tick / drained-sentinel batch showing ALL of a lane's
DISPATCHED runs terminal (per-run `status: done`/failed JSON lines), an
`epm:cluster-terminal` for one lane of a multi-lane dispatch, a
detached-phase DONE breadcrumb, or the lane's own completion report —
reconcile, in the same turn, that lane's/phase's REALIZED cells against
the PLANNED cells the approved plan declares for it. Planned side: the
highest `plans/v{K}.md` (the §5 conditions table, the §6.5
`primary_deliverable` rows, and/or the §9 per-component table — whichever
enumerates the lane's cells) — INCLUDING plan-declared DERIVED per-cell
deliverables / required outcomes (a dose-matched pair per (behavior,
context, seed); an in-band arm per regime — per the plan's OWN selection
rule), each reconciled at the lane or phase completion where it FIRST
becomes computable (a derived deliverable adjudicated by a later phase —
e.g. a judged-ladder dose-match — reconciles at THAT phase's completion,
not at the training lane's). Realized side: the drained sentinels /
per-cell result files / the lane's log. A lane exiting cleanly does NOT
imply coverage — a planned cell that was never dispatched, or one whose
plan-declared required outcome was not produced (per the plan's own
selection rule; a plan-sanctioned fallback selection counts as realized),
counts as missing. Then:

- **All planned cells realized** → post nothing (silence = covered; the
  Step 6d.3 run-completion summary is the single positive record).
- **A planned cell is missing** → post, same turn:
  `uv run python scripts/task.py post-marker <N> epm:progress --note
  "planned-cell-reconcile lane=<lane or phase> planned=<k> realized=<m>
  missing=<cell ids> disposition=<pending|re-sweep|documented-drop>"`
  and DECIDE re-sweep vs documented-drop under the EXISTING rules — this
  is auto-continue, never a new gate: an autonomous session picks the
  max-info-gain-per-GPU-hour option toward the Goal and states
  `Decision: ...`; an interactive session surfaces the missing cell in
  the session text (FYI + decision, not a question). A re-sweep
  dispatches through the normal relaunch path (fresh `epm:run-launched`,
  relaunch contract above); a documented-drop records
  `disposition=documented-drop` in the note and MUST be carried into the
  clean-result per After-Every-Experiment item 8 (name the missing
  condition, revise denominators, label figures `N/A — not tested`).

The duty is keyed per LANE/PHASE completion, not per tick and not per
cell — per-cell done lines inside a still-running lane do not trigger
it. A false-covered read (a missing cell misread as realized) fails
toward today's status quo — the terminal clean-result check still
catches it. The duty is a defined no-op when the plan declares no
per-lane cell enumeration (single-cell runs, infra tasks); it binds in
same-issue follow-up rounds (the loop runs status-held at
`followups_running`) but NOT in non-/issue observation sites (the #660
program-orchestrator daemon — out of scope); and it does NOT replace the
terminal clean-result reconciliation — it is an EARLIER surfacing of the
same check; check 11b and the planned-vs-actual lens remain
authoritative and unchanged.

**`--pid-file` is a POD-side path.** `poll_pipeline.py` evaluates
`[ -f <pid_file> ]` inside its remote SSH heredoc, so the pid file must
exist ON THE POD (the experimenter's launcher writes it there at launch
time). A pid file written only on the local VM silently reads
`PID_ALIVE=0` every tick, and the probe falls back to the pid from the
latest `epm:run-launched` marker.

**Any relaunch must re-post `epm:run-launched`.** After ANY hot-fix
relaunch of the pod workload (new pid), post a fresh `epm:run-launched`
with the new `pid=` (and `log_abs=`) before the next tick — the poller's
marker-pid fallback (`_marker_pid`) reads ONLY `epm:run-launched`
markers, so an `epm:progress` note recording the new pid is invisible to
it and the stale pid yields a false `status=dead` on a healthy run.
The same relaunch MUST also rewrite the pod-side pid file with the new
live pid in the same command chain — a present-but-stale pid file
silently probes a dead pid every tick and is rescued only while the
marker pid is itself alive (full contract + atomic recipe:
`.claude/rules/pod-side-reporting.md` § Pid-file launch contract;
#813 v5).
A crash-fix relaunch (a `code`-row fix round preceded it) additionally
passes the fix-commit ancestry probe and executes the declared
stale-checkpoint disposition BEFORE dispatch, recording `fix_sha=` in
the fresh marker note (`.claude/rules/crash-fix-rounds.md` § Crash-fix
relaunch: fix-commit ancestry + stale-checkpoint hygiene).
(#521) On the GCP lane the marker's `pod=`
field MUST be the instance name (`eps-issue-<N>`) — `GcpBackend.poll`
matches relaunch markers on that field to follow the new process
(#612): a mismatched value (e.g. a RunPod-style `pod-<N>`)
rejects the marker and the poll keeps reading the frozen startup-script
phase, and an omitted `pod=` is accepted only via the launch-time
`epm:cluster-launched` timestamp baseline, so include it explicitly.

**A successful relaunch also reconciles a stale `blocked`.** Immediately
after posting the fresh `epm:run-launched`, read the current status
(`task.py view <N> --json`); if it is EXACTLY `blocked`, run
`uv run python scripts/task.py set-status <N> running --note 'relaunch
succeeded; clearing stale blocked (epm:run-launched <ts>)'`. The stale
`blocked` arises when an earlier failed round (a cap-hit, a
STATE-TO-`blocked` exit, or a failed crash-fix cycle) parked the task and
a LATER round's relaunch succeeded without flipping it back — #742 ran
healthy for a day and a half at status `blocked` and the
dashboard/watcher read wrong until the user asked. Guards: (a) flip ONLY
`blocked` → `running`, never any other status — a same-issue follow-up
round holds `followups_running`, never `blocked`, so the flip is inert
there by construction; (b) the flip is a same-turn serial action after
YOUR OWN relaunch — never flip on someone else's marker (the watcher's
stale-blocked FLAG pass is deliberately flag-only; a human reconciles on
its evidence); (c) if the relaunched run then fails, the normal failure
path re-blocks — the flip does not suppress it; (d) RE-READ the status
immediately before the `set-status` call — a non-`blocked` read at that
instant ABORTS the flip (a human may have reconciled off the watcher
flag concurrently; a redundant flip attempt is refused, never forced).

The 540-second sleep stays under the Bash tool's 10-minute (`600000` ms)
cap with margin; longer intervals are achievable by raising the sleep
within the cap, but 9 minutes is the operational sweet spot (enough
time to make progress, short enough to catch stalls quickly).

**MANDATORY auto-armed backstop for the per-issue session.** The
single bg-Bash poll chain above is the primary monitoring mechanism but
is NOT robust on its own: it is one chain of one-tick-at-a-time
re-invocations, and if ANY reaction turn fails to emit the next bg-Bash
tool call (corrupted/truncated tool-call text rendered as raw output, an
API drop, a session crash), the chain dies permanently with no live bg
work and no scheduled wake. The pod keeps running; the per-issue session
goes silent; results strand and GPU billing accrues until the user
notices. (#463/#462)

The mandatory backstop is a harness-level recurring fire of
`/issue-tick <N>` (the LIGHTWEIGHT recurring driver — see
`.claude/skills/issue-tick/SKILL.md`) that does NOT depend on the
previous turn's bg-Bash chain surviving. Even after a dead reaction
turn, the next backstop tick fires a fresh `/issue-tick <N>` that reads
state from `events.jsonl`, refreshes the title, branches on status
(terminal/park/active/gate-park), and either tears down (terminal/park)
or hands off to the full `/issue <N>` skill for stale-marker recovery
(active with no fresh markers). The bg-Bash chain remains the primary
tick mechanism (faster, drains sentinels on each return); the recurring
`/issue-tick <N>` cron is the session-survival backstop.

**The orchestrator AUTO-ARMS this backstop itself — no user action, no
chat reminder.** For autonomous sessions, the primary arm site is
Step 0 (whole-lifecycle coverage); this Step 6d.2 arm is the SECONDARY
arm site, ARM-GUARDed so it's a no-op when Step 0 already armed. It
covers two cases Step 0 doesn't: (a) interactive (non-`--auto`) `/issue`
runs that reach the polling loop, where Step 0 deliberately skipped the
arm (interactive runs are user-driven and don't need automatic re-drive
between user turns), and (b) `--auto` sessions where the Step 0 arm
somehow didn't land (defense-in-depth — the ARM-GUARD makes the
duplicate call cheap, the missing arm catastrophic). The orchestrator
registers the cron directly via the `CronCreate` tool. The `Cron*`
tools are deferred — load them once per session with
`ToolSearch("select:CronCreate,CronList,CronDelete")` before first use.
On entering Step 6d.2 for a pod-backed `kind: experiment` run, BEFORE
starting the bg-Bash poll:

1. Call `CronList`. **ARM-GUARD:** if any job satisfies
   `prompt.strip() == "/issue-tick <N>"`, the backstop is already armed
   (this invocation was itself fired by that cron, or armed earlier
   this session) — skip straight to the poll loop. NEVER register a
   second cron for the same issue. Match on whole-string equality modulo
   surrounding whitespace, NOT `in` / `endswith` — `"/issue-tick 46"` is
   a substring of `"/issue-tick 467"`, so substring matching would
   mis-dedupe sibling issues.
2. Otherwise call
   `CronCreate(cron="*/45 * * * *", prompt="/issue-tick <N>", recurring=True, durable=False)`
   — a 45-minute, session-scoped, in-memory recurring fire of the
   lightweight `/issue-tick <N>` skill (dies with the session, auto-
   expires at 7 days like the default pod TTL; the harness jitters
   recurring fires so ticks don't all land on a fixed wall-clock mark).
   The 45-minute interval is
   chosen deliberately: the pure-Python `autonomous_session_watch.py`
   cron (every 10 min, free) carries ALL fast detection — DEAD-session
   respawn, alive-but-stalled respawn for ACTIVE statuses, pod safety,
   gate-park phone push, title reconcile — so the tick is purely the
   in-session re-driver of last resort for the alive-but-stalled-at-PARK
   class, which tolerates 45-min latency. Every tick fire is LLM-priced
   (a cold context read even on the guarded-no-op path), so fewer fires
   is the point. (The old 20-min rationale leaned on a "5-minute prompt
   cache TTL"; that figure is inaccurate for this org's subscription
   auth — subscription sessions get the 1-hour cache TTL automatically,
   5 minutes applies to API-key auth — and the interval choice no longer
   depends on it.) Then immediately re-`CronList`
   and assert EXACTLY ONE job matches
   `prompt.strip() == "/issue-tick <N>"`. If the harness normalised the
   stored prompt such that the ARM-GUARD would later miss, this assert
   fails loud NOW rather than silently stacking a duplicate cron on
   every subsequent re-entry.

Then proceed to the polling loop. Auto-arming HERE is required ONLY for
pod-backed `kind: experiment` runs reaching Step 6d.2;
`kind: analysis|infra|batch|survey` paths that never enter the polling
loop do NOT arm it here. A same-issue follow-up round is NOT exempt —
it arms at its OWN entry (Step 9b § Loop liveness backstop / the C3 +
step-6 re-arm), and one that reaches this polling loop re-arms via the
ARM-GUARD (a no-op when already armed).

**CRON-TEARDOWN procedure (run INLINE at every terminal / park exit site,
not only here in prose) — widened + idempotent
(#1052).** Sweep the cron store with a TWO-LEG match set,
resolving ids from a FRESH `CronList` at teardown time (#988 — never
`CronDelete` an id recorded earlier in the session: recorded ids go stale
when a one-shot fires or a concurrent teardown wins the race). Delete
EVERY job matching EITHER leg:

- **Leg 1 — the recurring tick cron:** primary match is whole-string
  equality (`prompt.strip() == "/issue-tick <N>"`); hardened fallback is
  the anchored pattern `issue-tick\s+<N>(?!\d)` (harness
  prompt-normalization drift was the #501 failure mode — the whole-string
  teardown silently no-oped 1,951 times; the `(?!\d)` guard prevents
  sibling mis-delete, `"/issue-tick 46"` never matches
  `"/issue-tick 467"`).
- **Leg 2 — stray one-shot `/issue <N>` wakeups (#980 — a live one-shot
  wakeup that survives past terminal re-drives the FULL skill on a
  completed task):** primary match is whole-string equality against the
  bare full-skill wakeup prompt (the f-string form in the canonical
  block); fallback is the START-anchored pattern `/issue\s+<N>(?!\d)` via
  `re.match` (the start anchor keeps deletion surgical — a prose prompt
  that merely MENTIONS the issue never matches; `(?!\d)` guards siblings;
  the `-` in `/issue-tick` fails `\s+`, so leg 2 never re-matches leg 1's
  job; trailing text like `--auto` matches by design).

A `CronDelete` error indicating the job does not exist (observed shape:
`No scheduled job with id …`) is SUCCESS, not a failure (#988) —
idempotent means the job being gone is the goal: continue the sweep,
never retry that id, never abort or raise on it. Then
ASSERT-AFTER-DELETE over BOTH legs: re-`CronList` and verify no job
matching EITHER leg survived; if one did, retry the delete ONCE (fresh
id from the re-list), then log LOUDLY — the runaway parachute
(`tick_triage.py`'s 3-consecutive-terminal flag + the watcher's
force-stop) bounds the damage of a cron that refuses to die. Canonical
pseudocode: `.claude/skills/issue-tick/SKILL.md` § CRON-TEARDOWN.

**Prevention ban (#980).** An `/issue` session must NEVER schedule its
own re-drive — no `ScheduleWakeup` wakeup and no `CronCreate` one-shot,
regardless of prompt shape. The Step 0 / Step 6d.2 `/issue-tick <N>`
cron is the ONLY sanctioned self-wake: a one-shot wakeup may not be
enumerable at teardown time, and one that fires after terminal re-drives
a completed task (#980). Leg 2 + the self-heal sweep BOUND — they do not
guarantee deletion of — whatever the store fails to surface.

The backstop
DELIBERATELY survives the `done` → `verifying` transition (Step 6d.3) and
keeps re-firing through the uploading / verifying / interpreting /
reviewing stages — those stages have no other auto-wake, so the backstop
is the only thing that revives an interactive per-issue session that
stalls there. It is torn down ONLY at the true terminal / park
transitions:

- at `awaiting_promotion` (Step 9b — the pod was terminated at Step 8 and
  this is a human gate, so no more auto-driving);
- at the Step 10d merge exit (code-change paths only — the auto-merge is
  the terminal step there, so CRON-TEARDOWN + `set-status completed` +
  `epm:done` fire AFTER `epm:merged` posts, or in the `epm:merge-failed`
  terminal-failure branch, instead of at Step 10 auto-complete;
  #1723 — closes the ~33 min merge window that used to run without
  `/issue-tick` re-drive coverage and with the durable record already
  reading `completed`+`epm:done` on an unmerged branch);
- at `completed` (Step 10 auto-complete on the experiment path, once
  `epm:merged` is already present from Step 9b — the code-change path
  reaches `completed` via the Step 10d exit bullet above);
- at any `set-status <N> blocked` exit in Step 9 / the
  interpretation+review loop;
- at the `status=stalled` / `status=dead` / unrecognised-gate `blocked`
  exits in the poll loop above; and
- at the Step 6d.4 gate-park exit (the pipeline has EXITed and no pod is
  burning GPU — the user now drives the resume).

Each of those exit sites carries an explicit "run CRON-TEARDOWN" line. A
gate resume or a recovery re-invocation re-enters Step 6d.2 and re-arms
via the ARM-GUARD.

Surviving the backstop into verifying / interpreting / reviewing is the
DESIGNED behavior, not an accident we tolerate. Its only cost — a tick
landing while a stage subagent is already in flight and REDUNDANTLY
re-dispatching that stage (analyzer, clean-result-critic, upload-verifier)
— is bounded by the Step 9 **idempotency guard**: a tick that lands on a
stage whose latest `events.jsonl` marker is a fresh dispatch with no
terminal/result marker yet EXITs without re-dispatching, so the live work
finishes uninterrupted (concrete rule in Step 9). State stays coherent
regardless because every re-entry reads `events.jsonl` fresh. If a
teardown at a terminal/park transition is ever missed, the residue is
cheap: the cron auto-expires at 7 days, and a tick landing on a
`completed` / `archived` / `awaiting_promotion` issue is a no-op that
SELF-HEALS (the re-invoked skill reads terminal/park state, exits without
re-arming, and runs the two-leg sweep before exiting — so a wakeup that
escaped an earlier teardown deletes its own stray siblings when it fires;
the blast-radius bound for whatever the store fails to surface).
Run CRON-TEARDOWN the moment you spot a stranded cron or stray one-shot
wakeup (fresh `CronList` → `CronDelete`, both legs).

Residual failure mode the in-session backstop does NOT cover: if the
per-issue *session itself* dies (process exit, host reboot), a
`durable=False` cron dies with it and the pod goes unmonitored. Two
mechanisms cover that, with DIFFERENT strength:

1. The "spawn a fresh session" recovery row recovers the work.
2. The EXTERNAL pod-safety backstop
   (`scripts/autonomous_session_watch.py`, the every-10-min VM cron
   `3-59/10 * * * *`) reconciles RUNNING managed pods (`pod-<N>`, legacy
   `epm-issue-<N>` still recognized) against their task STATUS. It is
   CONSERVATIVE by design:
   - it AUTO-STOPS (reversible — `pod.py stop`, never terminate, after ≥
     2 consecutive checks) only a RUNNING pod whose task is already DONE
     (`completed` / `awaiting_promotion` / `archived`) — i.e. an ESCAPED
     pod (Step-8 terminate failed, or the pod never went through Step 8).
     A done task provably needs no pod, so this stop is unambiguous;
   - it does NOT auto-stop a pod whose task is still mid-run
     (`approved` / `running` / `verifying`). For those it ALERTS (a loud
     log line + a one-time dashboard-visible marker on the task) when no
     real progress marker has landed for > 6h — a likely abandoned
     session — but leaves the pod RUNNING. A false alert is a cheap
     nudge; a false stop would kill a healthy long run, so the backstop
     never makes that trade. `blocked` pods are KEPT (alert-only if
     stale), never auto-stopped. `interpreting` / `reviewing` pods
     classify as "other" (those stages don't drive pods — interp/review
     reads from WandB/HF), so they're kept too and caught later when the
     task reaches `awaiting_promotion`.

So the external backstop bounds GPU burn for the clean case (a finished
experiment whose pod escaped termination) and SURFACES the harder case
(a session that died mid-run) for human action — it does NOT silently
stop mid-run pods. Full mid-run auto-stop (e.g. a pod-side idle-GPU
probe that distinguishes a stalled run from a slow one) is a noted
follow-up, not implemented. No crontab change is needed — the watcher is
already scheduled.

The RETIRED independent stall-watchdog (`scripts/pod_watch.py`
spawned as a long-lived background process writing to
`.claude/cache/watch-<N>.pid`) was retired alongside the orchestrator
polling loop; it is NOT the backstop here. See "Notes on the
obsolete monitoring stack" below for the single source of truth on
which mechanisms are live vs retired.

**Long-phase heartbeat duty (BINDS every >60-min quiet stretch — ALL
loops, BOTH session modes; #1207, #1092/#825/#1112).** Nothing external
refreshes a session's liveness signals
between status transitions: the tick skill no longer touches the
self-report (issue-tick SKILL.md § "Title refresh — moved to the
watcher") and the watcher's reconcile is status-transition-keyed by
design. So during any stretch where THIS session awaits work and
>60 min could elapse without a turn that posts a non-watcher marker —
an ad-hoc bg-Bash poll chain, a `Monitor` until-loop, a
deadline-bounded Batch-API poll, a detached VM phase (§ "Detached
VM-side long compute phases", Step 9 entry guard), or any
follow-up-round wait at `followups_running` — the orchestrator carries
BOTH duties below. (The Step 6d.2 polling loop above discharges them by
construction: the top-of-tick `set_title` refresh + the bounded tick
cadence — the fixed 540s chain, ≤ ~40-min Monitor quiet-wait cycles.
The duty is for every wait that is NOT that loop. A long
FOREGROUND subagent wait is a named out-of-scope shape — no resumable
orchestrator turn exists there to discharge the duty; the watcher's K=2
live-escalation debounce covers it.)

1. **Structure the wait so a turn resumes at least every ~45 min.**
   Cap any single blocking wait at ≤45 min — chain bg-Bash sleeps /
   segment a `Monitor` until-loop
   (`until <check> || [ $(elapsed) -gt 2700 ]; …`) rather than arming
   one silent multi-hour wait. Load the deferred schemas BEFORE the
   first poll call — `ToolSearch("select:Monitor,TaskOutput")` — an
   unloaded deferred-tool call fails with InputValidationError. A
   single 4-h until-loop (#1092)
   leaves zero heartbeat opportunities: the watcher's
   90-min exemption leash (`LONG_PHASE_HEARTBEAT_FRESH_S`, sized as a
   ~60-min cadence + 30-min slack) lapses mid-wait no matter what was
   posted before entering it. 45 min matches the `*/45` tick cadence
   and keeps every resume inside the 60-min self-report window
   (`STALLED_WINDOW_S`).
2. **At each resume ≥~45-60 min into the phase (a ~60-min heartbeat
   cadence): verify, then heartbeat + refresh.** (i) VERIFY the awaited
   work is alive with cheap evidence — `ps -p <pid> -o args=` identity
   match, breadcrumb `log=` mtime advanced, a Batch-API status read, a
   poll-tick JSON line (GCE content reads: a Permission-denied `tail` is a
   probe artifact — root-owned workload log; retry `sudo -n tail` per the
   § Successor / re-entry rule GCE log-read note, never a verify-FAIL);
   (ii) post the heartbeat marker, evidence in the
   note:

       uv run python scripts/task.py post-marker <N> epm:progress \
         --note "[long-phase-heartbeat] <phase>: <one-line evidence, e.g. pid 12345 alive, log +3 lines>"

   (iii) refresh the self-report:
   `uv run python scripts/session_progress_report.py --issue <N> --step "<phase>"`.
   The two writes refresh BOTH staleness signals — the sparing is never
   the 90-min leash alone: the marker buys the stalled-detector leash
   (`autonomous_session_watch._long_phase_heartbeat_reason`) AND
   converts `tick_triage.py`'s STALE-REDRIVE to HEALTHY (#1051), while
   the self-report refresh keeps signal 1 (`STALLED_WINDOW_S`) fresh so
   the detector never reaches the exemption probe at all. NEVER
   heartbeat blind: if the verify FAILS (pid gone, log frozen, batch
   errored), do NOT post a heartbeat — run the failure path (crash-fix
   routing / `epm:failure`). A heartbeat without evidence shields a
   dead phase from recovery for up to 90 min and is the banned inverse
   of the false-respawn this duty prevents. (Pid-bearing detached-phase
   breadcrumbs stay authoritative over heartbeat notes — tick_triage
   #1051.) On a long same-phase stretch — keyed on elapsed same-phase
   time (≥~60–90 min) or ≥2 heartbeat resumes in the same phase, NOT the
   3-tick count (heartbeat cadence is ~45–60 min) — the heartbeat
   evidence ALSO includes the Step 6d.2 § Same-phase rate/ETA duty's
   `[phase-rate]` read (#1863): alive ≠ progressing.

**Remote-landing watches carry a producer-fence deadline (#1850;
#1738/#1739).** Any watch whose wake condition is
a REMOTE artifact landing — an HF file/prefix appearing, a
pod/GCE-produced output, a sentinel drained from another box — carries
an explicit overall DEADLINE = the producer's own lifetime bound (the
GCE `--max-run-duration` fence, the pod TTL, a Batch-API `expires_at`)
+ ~15-30 min grace, on top of the per-segment ≤45-min cap (item 1): a
landing keyed on a dead producer NEVER fires, so without the deadline
the watch reads as healthy idle forever (#1738). On deadline expiry the watch exits DEADLINE and the session
RE-CHECKS THE PRODUCER — instance/pod status (`gcloud … describe` /
`pod.py list-ephemeral`), the crash-persist prefixes
(`issue<N>_partial/` / `issue<N>_done/`) — and routes to the
failure/recovery path; it never blind re-arms the same landing watch.
Item 2(i)'s per-resume verify covers the PRODUCER, not merely "the
landing has not appeared yet". This generalizes the deadline-bounded
`batch_judge` poll (#658/#663), which bounds on the batch's own
`expires_at`.

**Monitor heartbeat emission (#1850).** A long-interval `Monitor`
until-loop ADDITIONALLY emits a no-op heartbeat line roughly every
15-30 min (every 2-3 cycles of a long-interval loop — time-anchored, so
a short-interval loop does not over-wake), e.g.
`[watch-heartbeat] <UTC time> waiting on <what>` via an echo inside the
loop — each stdout line is a notification, so heartbeats WAKE the
session at a known cadence, giving item 2's verify-then-heartbeat its
resume opportunity mechanically AND making a dead/lost Monitor
distinguishable from a healthy quiet one: at any later wake (tick,
notification), a heartbeat gap of ≳2-3 expected intervals means the
Monitor died — re-arm it after the kill-before-relaunch probe
(`.claude/rules/crash-fix-rounds.md` § Kill-before-relaunch), never
assume it is still watching (#1739). Carve-out: the Step 6d.2 QUIET-WAIT
Monitor (#1924) — a single-shot bounded (≤ ~40 min) wait-then-poll
whose terminal tick-JSON line IS its emission — owes no mid-wait
heartbeat: the 15-30-min cadence targets cycling long-interval /
indefinite watches, which it is not. The `[watch-heartbeat]` line is
Monitor stdout only — NEVER a task marker; the `[long-phase-heartbeat]`
`epm:progress` marker convention (item 2) is separate shared machinery
and is untouched.

Revival trigger for the deferred watcher-side option (b) (#1207
§11-R4): a STALLED-DETECTOR-lane force-respawn of a session carrying a
fresh (<90-min) heartbeat is the recorded evidence that emitter-side
duty is insufficient — a wedge-lane respawn of a duty-compliant session
is by design (#1127) and does NOT count.

##### Step 6d.3: On `status=done`

Do NOT run CRON-TEARDOWN here. The backstop INTENTIONALLY survives past
`done` into the uploading / verifying / interpreting / reviewing stages —
those stages have no other auto-wake, so an interactive per-issue session
that stalls in them would otherwise go silent forever. The cron is torn
down only at the true terminal / park transitions: at `awaiting_promotion`
(Step 9b), at the Step 10d exit (code-change paths only — after
`epm:merged` posts or in the `epm:merge-failed` terminal-failure branch;
#1723), at `completed` (Step 10 auto-complete on the experiment path,
`epm:merged` already present from Step 9b), and at any
`set-status <N> blocked` exit in Step 9 / the interpretation+review loop
(plus the poll-loop stalled/dead/blocked exits and the Step 6d.4 gate-park
that already tear it down). The Step 9 idempotency guard (below) bounds the
redundant-subagent cost a surviving-into-`done` cron used to risk.

**Run-completion reconciliation backstop (#1481).** Before the status
flip below, run the Step 6d.2 § Per-lane planned-cell reconciliation
ONCE over the WHOLE run — all planned cells (and plan-declared derived
deliverables) across all lanes vs all realized cells — and post ONE
summary line either way:
`uv run python scripts/task.py post-marker <N> epm:progress --note
"planned-cell-reconcile run-complete planned=<k> realized=<m>
missing=<cell ids or none>"`. This catches lanes whose individual
completion was never observed (orchestrator respawn mid-run, coalesced
sentinels) while re-sweep is still cheap — before verification and
teardown. Cells already dispositioned per-lane carry their recorded
`disposition=` forward (never re-decide them); a non-empty `missing=`
list of UNdispositioned cells takes the same re-sweep vs
documented-drop decision as the per-lane duty (a re-sweep returns to
the running phase via the normal relaunch path instead of flipping to
`verifying`). Skip the summary line when the plan declares no cell
enumeration (the per-lane duty's no-op case).

Transition the task to `verifying` (the upload-verifier next):

> **Same-issue follow-up round?** At `followups_running`, SKIP this
> `set-status` (status-hold rule, Step 9b § Same-issue follow-up loop step 3;
> code-enforced — `task.py` refuses the flip) — phase visibility comes from
> `stage=followup-<phase>` breadcrumbs, not status flips.

```bash
uv run python scripts/task.py set-status <N> verifying \
    --note "polling loop observed phase=done"
```

Then proceed to Step 7 (which handles results → upload routing).

##### Step 6d.4: On `status=gate` — handle a pod-side gate (park OR auto-resolve)

Pod-side dispatchers cannot post markers directly (the `task.py`
branch-guard and the CLAUDE.md "Pod-side code NEVER shells out" rule),
so they write a sentinel file at `/workspace/logs/issue-<N>-*.json`
that `poll_pipeline.py` drains. When a sentinel carries a non-empty
`gate` field **AND `blocks_pipeline: True`**, the poller posts the
carried marker from the VM (e.g. `epm:fact-candidates v1`) and returns
`status=gate` with `gate=<name>`.

The poller ONLY surfaces `status=gate` when the drained sentinel had
`blocks_pipeline: True` (the field defaults to True when absent, so a
sentinel that carries only a `gate` name still parks). Sentinels with
`blocks_pipeline: False` are the dispatchers' benign phase-progress
signals (`gate=phase`, `gate=smoke`, `gate=dryrun` are the canonical
ones): their marker IS posted from the VM, but they NEVER end the
polling loop and NEVER trigger the fail-fast block. They are NOT user
gates — do not treat a `blocks_pipeline: False` phase signal as an
unrecognised gate (#641).

The orchestrator handles the named gate inline rather than continuing to
poll — the pipeline itself has EXITed at the gate. Most gates are PARK-mode
(`fact-candidates`): the pipeline is waiting on a user answer, so the
orchestrator parks. An AUTO-RESOLVING gate (`pv_phase1_done`) is the
exception: the orchestrator resolves it itself (on the RunPod lane a
pod-cycle around an off-pod step; on the GCP lane finalize + fresh
dispatch — see the GCP-lane teardown leg below) and resumes the loop in
the same turn — see the per-gate handlers below.

**GCP-lane blocking gates — instance-teardown leg (EVERY handler, PARK-mode
and auto-resolving; #908/#763/#935).** On the GCP lane a blocking-gate exit is
a CLEAN exit (`[phase=done]` → guest `eps/phase=done`), so the in-VM EXIT trap
does NOT power the VM off — the GCE instance stays RUNNING only within the
bounded done-grace window (default 90 min,
`EPS_GCP_DONE_POWEROFF_GRACE_SECONDS`; the #935 self-poweroff best-effort
persists the undrained sentinel set to HF `issue<N>_done/<attempt_id>/` at
expiry, then powers off; the clean-exit path keeps it alive only for sentinel
draining — `backends/gcp.py` `teardown` docstring). The finalize teardown leg
below remains PRIMARY — never wait out the grace. Two operational lines
(REQUIRED — the only defense on the DELETE outcome): (a) on a
`workload_done_self_poweroff` poll (TERMINATED + guest `eps/phase=done` — the
grace expired on the STOP outcome) OR a post-expiry `dead("instance not
found")` poll, CHECK the HF data-repo prefix `issue<N>_done/<attempt_id>/`
BEFORE any crash-fix routing — the run SUCCEEDED and self-powered-off;
recover the undrained completion/gate sentinels from that prefix and run
finalize with `--skip-confirm-artifacts`. (b) Gate sentinels persist to that
same prefix at grace expiry on a BEST-EFFORT basis (one retry) — the prefix
is NOT guaranteed to exist, and the persist also never fires on a mid-grace
preemption or manual stop, so an ABSENT prefix does NOT distinguish "poller
drained normally" from "expiry persist failed"; on a STOP-outcome instance
the `eps/done_persist` guest attribute (`ok|failed`, a SEPARATE key from
`eps/phase`) disambiguates. A `workload_done_finalize_failed` poll (#1055 —
the deliverables-verified-then-finalize-crashed classification, guest
`eps/phase=finalize_failed_artifacts_ok`) is a done-like shape too: treat it
as a SUCCESSFUL run (no crash-fix routing) and run finalize with
`--skip-confirm-artifacts` exactly as for `workload_done_self_poweroff`
(the completion sentinel was never written; see
`.claude/rules/compute-backend-failover.md` § Part A-ter). Pre-existing
residual (unworsened by #935): a
#908 stale reclaim at a FRESH dispatch during the grace deletes the VM
before the expiry persist fires. By the time `status=gate` reaches
this step the sentinel is already drained (the poller drained it to post the
gate marker), and the VM holds NOTHING a later gate resolution needs — so
tear the instance down at the earliest point after the drain, split by gate
class: **PARK-mode gates** run the finalize command after the gate marker is
posted and BEFORE the park — before raising the user question, before
posting `step-completed parked`, before exiting — NEVER leave the instance
up through the user-wait window (the user's pick is NOT a teardown
precondition); **auto-resolving gates** (including a PARK-mode gate's
autonomous auto-resolve branch) run it after the auto-resolve step completes
and BEFORE dispatching any off-pod phase or the fresh tail dispatch:

```bash
uv run python scripts/dispatch_issue.py finalize --issue <N> --skip-confirm-artifacts
```

`finalize` DELETEs the instance AND retires the handle sidecar
(`.claude/cache/issue-<N>-handle.json` → `<name>.finalized`); a raw `gcloud
compute instances delete` leaves a stale sidecar the next launch would
misread — use it only when no sidecar exists. `--skip-confirm-artifacts` is
REQUIRED at a mid-pipeline gate: the run's declared final artifacts do not
exist yet, so a plain `finalize` FAILs confirm (exit 3) by construction.
(Mid-pipeline gate teardowns run BEFORE any upload-verifier dispatch or
`epm:results`, so the #1026 verifier-currency gate is a no-op here — no
verifying crumb, no results, no verdict to be stale against.) The
instance stays up ONLY for sentinel draining — never through an off-pod
phase or a park (Step 8-bis: a pod must not idle on a halt; #763: a
GPU idled after the `cofit_phaseA_done` gate-park). The next
pipeline phase provisions FRESH via the normal Step 6d.1 dispatch. There is
no GCP analogue of the RunPod `pod.py stop`/`resume` cycle (`pv_phase1_done`
below): GCE instances are ephemeral by design, and a STOPPED instance would
be deleted by the next launch's stale reclaim anyway — the GCP phase-cycle
is teardown + fresh dispatch. Backstop only (never the plan):
`backends/gcp.py::reconnect_or_none` refuses a RUNNING instance whose
`eps/phase` is terminal (`done`/`failed`/`finalize_failed_artifacts_ok`/`wedged`) and the pre-launch stale
reclaim deletes it, so a missed teardown no longer silently no-ops the next
dispatch (#763 leg 2) — but the zombie still bills until the #935 done-grace
self-poweroff (default 90 min), that next dispatch, or the daily janitor
sweep, so the handler-side teardown stays mandatory (never wait out the
grace).

Gate handlers (one per registered `<name>`):

- **`fact-candidates`** (used by `run_experiment_<N>.py`-style
  fact-teaching drivers, originally task #407): the `epm:fact-candidates
  v1` marker carries a ranked candidate table (one row per Wikipedia-
  stub fact passing the log-prob band filter, with `id` + summary
  + log-prob). The orchestrator reads the just-posted marker via
  `task.py latest-marker <N> --prefix epm:fact-candidates`, then branches
  on session mode. (The flag is `--prefix`; `latest-marker` has no
  `--kind` option — see `task.py latest-marker --help`.)

  **Interactive mode** (`EPM_AUTONOMOUS_SESSION` unset/falsy): surface
  the table via `AskUserQuestion` <!-- gate: gates.fact_candidates --> and
  ask the user to pick one `id`.

  <!-- gate: gates.fact_candidates -->
  <!-- autonomous-mode: auto-resolve -->
  ```python
  # Interactive mode only — autonomous mode auto-picks the median-log-prob id.
  AskUserQuestion(questions=[{
      "question": "Phase 0 (fact-candidates) — pick the fact for the obscure-real regime.",
      "header": "Pick fact (id)",
      "multiSelect": False,
      "options": [
          # one option per candidate id, label = "<id>: <one-sentence summary>"
          ...,
      ],
  }])
  ```

  **Autonomous mode** (`EPM_AUTONOMOUS_SESSION=1`): NEVER raise the ask
  AND never print the candidate options as a text menu. Auto-resolve per
  § Autonomous session behavior → `fact_candidates`: pick the candidate
  `id` with the median per-token log-prob (the middle of the band the
  plan filtered by). State `Decision: id=<X> (median log-prob in band)`
  AND EXECUTE the resume in this same turn (post `epm:fact-pick v1` with
  `id: <X>` and resume the polling loop); do NOT state the Decision and
  then end the turn.

  On user reply (interactive) or auto-pick (autonomous), post
  `epm:fact-pick v1` with the chosen id in the note body (`id: <N>`):
  ```bash
  uv run python scripts/task.py post-marker <N> epm:fact-pick \
      --note "id: <chosen_id>"
  ```

  In interactive mode the user then re-invokes `/issue <N>` to resume;
  the driver's `--phase fact-pick` step reads the latest `epm:fact-pick`
  marker, materialises `fact_pick.json` on disk, and the next pipeline
  phase proceeds. In autonomous mode the orchestrator resumes the
  polling loop directly without a re-invocation. (See plan §4.2 of any
  fact-teaching task for the on-pod resume contract. GCP lane: the
  instance was already finalized per the GCP-lane teardown leg above —
  the resume is a FRESH Step 6d.1 dispatch of the fact-pick tail, never
  a poll against the old instance.)

- **`pv_phase1_done`** (issue #763 persona-vector extraction —
  off-pod judge between two GPU phases): an AUTO-RESOLVING gate, NOT a
  user park. The dispatcher `scripts/issue763_dispatch.sh` runs GPU
  phase 1 (`generate + capture + pv_extract_generate + upload-progress`
  — the PV rollouts are now on the HF data repo), emits the blocking
  sentinel `gate=pv_phase1_done` via
  `scripts/issue763_upload.py --emit-gate pv_phase1_done` <!-- lint: historical-ref --> (called from
  `scripts/issue763_dispatch.sh` after upload-progress)
  (`write_sentinel("epm:gate", …, blocks_pipeline=True)`), and EXITs.
  Unlike `fact-candidates` (which PARKS for a user pick at
  workflow.yaml § gates.fact_candidates), the
  orchestrator resolves this gate ITSELF — it does NOT raise
  `AskUserQuestion`, does NOT post `epm:step-completed --exit-kind
  parked`, does NOT exit the skill, and does NOT CRON-TEARDOWN — by
  orchestrating a pod-cycle around an off-pod judge step and then
  RESUMING the polling loop in this same turn. This handler behaves
  IDENTICALLY in interactive and `EPM_AUTONOMOUS_SESSION` modes (there
  is no user decision to make — the gate is fully auto-resolved).
  <!-- autonomous-mode: auto-resolve -->
  Concretely:

  1. **Stop the GPU pod** (volume preserved): `uv run python scripts/pod.py
     stop --issue <N>`. This frees the GPU through the deadline-bounded
     stop (see Step 6d.2 § "Stop the pod" / "Notes on the obsolete
     monitoring stack"); the PV rollouts are already on HF, so nothing on
     the pod's ephemeral disk is lost. (RunPod lane. On the GCP lane there
     is no stop/resume — apply the GCP-lane instance-teardown leg above:
     `finalize --skip-confirm-artifacts` once the phase-1 artifacts are
     confirmed on HF, then re-dispatch the tail as a fresh launch.)
  2. **Run the judge OFF-POD on the VM**: `uv run python
     scripts/issue763_extract_pv_rb.py --phase judge`. <!-- lint: historical-ref --> (This script
     ships on the `issue-763` branch — it lands on `main` when #763 merges;
     the reference is a forward / sibling-branch one, not a dead tool.)
     This is VM-safe by construction and NOT a `task.py` pod-shellout: it
     fetches the PV rollouts from HF via `snapshot_download` (NOTE: on the
     ~1M-file data repo `snapshot_download` wedges in full-tree
     enumeration — `.claude/rules/gotchas.md`; patch the script to scoped
     `list_repo_tree(path_in_repo=...)` staging before re-running this
     phase on a #763 follow-up), batch-judges
     through
     `eval.batch_judge` (the deadline-bounded client — never a hand-rolled
     `messages.batches.create` + deadline-less poller), and uploads the
     keep-flags back to the issue HF prefix. It needs no GPU and posts no
     `task.py` markers itself (it is an off-pod analysis step; the
     orchestrator owns the poll-loop markers).
  3. **Resume the pod**: `uv run python scripts/pod.py resume --issue <N>`
     — new IP/port; `pods.conf` + `~/.ssh/config` + MCP config auto-refresh
     on resume (re-run `/mcp` if the SSH MCP entry needs the refreshed
     host/port; if SSH keeps failing on a stale port, pull the live
     host/port back with `pod.py config --refresh-from-api` per Step 6b
     "stale-port recovery", the #488 13h-loop failure class). CONFIRM the
     resumed pod is reachable (`uv run python scripts/pod.py health
     --quick`) before re-dispatching.
  4. **Re-dispatch the workload tail** at `--from-phase pv_capture` via
     the SAME experimenter launch pattern as the original launch
     (Step 6d.1): spawn the `experimenter` subagent with the workload
     command `bash scripts/issue763_dispatch.sh --from-phase pv_capture`
     and the resumed pod's name (`pod-<N>` / `epm-issue-<N>`). The
     dispatcher resumes at capture → E0 judge → fit → figures → final
     upload → `[phase=done]`. The experimenter posts a fresh
     `epm:run-launched` marker (new `pid` + `log_abs`) and exits, exactly
     as on the first launch; the orchestrator updates its local poll-loop
     `pid`/`log` from that marker.

  Then **RESUME the polling loop** (Step 6d.2) at the next tick — do NOT
  exit, do NOT park, do NOT CRON-TEARDOWN. The gate has auto-resolved.
  (RunPod lane — the stop/judge/resume cycle above keeps ONE pod across
  phases, so the pod is burning GPU again after resume. GCP lane: there
  is no stop/resume — the instance was finalized per the GCP-lane
  teardown leg above, and the tail runs as a FRESH dispatch, so the poll
  loop resumes against the NEW handle, never the old instance.) Either
  way the `/issue-tick <N>` backstop cron stays armed and the bg-Bash
  poll chain continues. (Contrast with `fact-candidates` above, which parks
  for a user pick and tears the cron down.) **Idempotency on re-entry:**
  if a re-entry observes an `epm:gate v<n>` for `pv_phase1_done` followed
  by a FRESH `epm:run-launched` (post-resume; ts > the gate marker),
  treat the gate as already resolved and proceed with normal polling — do
  NOT re-stop / re-judge / re-dispatch.

- **Unrecognised `gate` name**: this branch fires ONLY for a sentinel
  the poller surfaced as `status=gate` — i.e. one that carried
  `blocks_pipeline: True`. A non-empty gate name with
  `blocks_pipeline: False` (`gate=phase` / `gate=smoke` / `gate=dryrun`)
  is filtered out by the drain and NEVER reaches this branch, so it is
  NOT an unrecognised gate and MUST NOT trigger the block below. For a
  genuinely unrecognised (blocking) gate name: log a one-line WARN, post
  `epm:failure
  v1` with `failure_class: code` and `reason: unrecognised_gate_name`
  (the `code|infra|data` taxonomy has no `workflow` class; the failure
  classifier defaults unknown classes to `code` anyway), a note pointing
  at the unrecognised gate name + the sentinel path, run CRON-TEARDOWN
  (§ CRON-TEARDOWN procedure — both legs incl. stray one-shot
  `/issue <N>` wakeups), set
  `status:blocked`, exit. This forces a workflow-fix-candidate before
  the gate name can silently no-op.

**PARK-mode gates only** (`fact-candidates` and the unrecognised-gate
branch): the tail below applies ONLY to gates that exit the skill to wait
on a human. Auto-resolving gates like `pv_phase1_done` (above) handle
their own continuation — they do NOT tear down the RUNPOD pod mid-cycle
(the stop/judge/resume cycle IS the continuation) and do NOT exit; on the
GCP lane the auto-resolve handler DOES tear the instance down (finalize +
fresh dispatch, per the GCP-lane teardown leg above) — "no teardown" is
RunPod-scoped. Their handler resumes the polling loop in the same turn,
so it skips this whole paragraph.

For a PARK-mode gate: run CRON-TEARDOWN before parking (the HARDENED +
WIDENED Step 6d.2 procedure, both legs incl.
stray one-shot `/issue <N>` wakeups; § CRON-TEARDOWN
procedure) — the pipeline has EXITed and no pod is
burning GPU (on the GCP lane because the teardown leg above already
finalized the instance BEFORE this park), so the backstop should not keep
re-firing `/issue-tick <N>` (which
would re-surface the gate question every 45 min). The user's
re-invocation after posting the resume marker re-enters Step 6d.2 and
re-arms via the ARM-GUARD. After posting the resume marker, EXIT the
skill cleanly via `uv run python scripts/post_step_completed.py --issue <N>
--step 6d --exit-kind parked` (the §5 `epm:step-completed` marker); the
user's re-invocation of `/issue <N>` resumes the polling loop. The polling-loop's terminal
transitions are now `running → verifying` (on done), `running → running`
(after a parked-and-resumed user gate, OR after an auto-resolving gate's
handler returns), or `running → blocked` (on stalled/dead or unrecognised
gate).

##### Notes on the obsolete monitoring stack

Single source of truth on live vs retired monitoring (the recovery
table below must agree). RETIRED: `scripts/pod_watch.py` / `pod.py
watch` + the `.claude/cache/watch-<N>.pid` pid-file are manual/debug
only — NEVER auto-spawned by this skill, never required for a healthy
run, never an unattended recovery path (a recovery row saying "watchdog
crashed" means "the bg-Bash poll chain has no live tick", NOT "respawn
pod_watch.py"); the `experimenter` agent no longer monitors the run.
LIVE during a `running` (workload) phase, exactly two, in order: (1)
the orchestrator's bg-Bash poll chain (Step 6d.2) — primary; (2) the
auto-armed `/issue-tick <N>` backstop cron (registered at Step 6d.2,
torn down at terminal/park transitions — NOT at `done`; see Step 6d.2
CRON-TEARDOWN), which survives a dead reaction turn — no user `/loop`
typing is needed.
