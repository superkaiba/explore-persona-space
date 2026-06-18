# Task #549 — Methodology, parameters, and worked examples

A methodology reference for analysis task #549 (Explore Persona Space): a CPU-only, zero-GPU audit of every historical eval run that loaded LoRA adapters into a vLLM engine, checking for the #534 `lora_int_id` cache-collision bug. This document describes only HOW the audit works — the bug mechanism, the enumeration dragnet, the per-run forensic reconstruction recipe, the LRU replay simulator, the calibration controls, and the comparison protocols. No verdicts, counts, or findings appear here.

- Task: [https://eps.superkaiba.com/tasks/549](https://eps.superkaiba.com/tasks/549) (kind: analysis, parent #534)
- Model: n/a — no model is trained or run by this task. The audited inference stack is **vLLM 0.11.0** LoRA serving (its source is read, never executed).

---

## 1. The bug mechanism under audit

vLLM identifies LoRA adapters by an **integer id** (`lora_int_id`), not by path. The LRU worker manager's `add_adapter` (vllm 0.11.0, `vllm/lora/worker_manager.py:240-267`, read verbatim from the installed package) implements:

- **miss** (`lora_int_id` not in `list_adapters()`): load weights from `lora_path`; if the registry is at capacity, evict the least-recently-used entry first (`remove_oldest_adapter`).
- **hit** (id already registered): *"If the lora is already loaded, just touch it"* — the adapter is served from cache and **`lora_path` is never re-read**. If a driver reuses an id for a *different* adapter path while the old entry is still registered, the request is silently served the first-loaded weights, with no error.

Three further semantics pin the audit rubric, each verified in the installed source:

| Fact | Source |
|---|---|
| Registry membership consults the CPU-side `_registered_adapters` LRU, capacity = **`max_cpu_loras`**, which **defaults to `max_loras`** (default 1) | `vllm/lora/models.py:391-392,740-742`; `vllm/config/lora.py:35,42,116-117` |
| The base (non-LRU) `WorkerLoRAManager.add_adapter` has the same id-keyed early-return (`if adapter_id in self.list_adapters(): return False`) | `vllm/lora/worker_manager.py:184-190` |
| `lora_int_id < 1` raises `ValueError` at request construction — a loud crash, never a silent mis-serve | `vllm/lora/request.py:34-35` |

**The residency rubric (the audit's central refinement).** Bare id reuse is NOT sufficient for exposure: a repeat id with a different path is mis-served *only while that id is still registered*. With capacity 1, any intervening distinct-id load evicts the old entry, so the repeat triggers a fresh (correct) load — "safe by eviction". The parent incident (#534 round-1) fired precisely because the id was *constant* across a multi-checkpoint loop: the resident adapter always matched the requested id and was never evicted, so every checkpoint after the first was served the first checkpoint's weights. Exposure is therefore decided by **replaying the realized request order through the cache semantics** (section 4), never by eyeballing the id expression.

---

## 2. Enumeration dragnet (which code paths get audited)

The audit unit is a **driver**: any script or module that constructs `LoRARequest(...)` objects, at any point in repo history, on any branch. Enumeration uses four overlapping sweeps so deleted, renamed, and branch-only drivers cannot escape (plan §4 Step A, hardened per plan §12 item 5):

```bash
# A1. HEAD inventory
grep -rn "LoRARequest(" scripts/ src/

# A2. Full-history pickaxe — catches deleted/renamed drivers
git log --all -S 'LoRARequest' --oneline --name-only -- scripts/ src/
git log --all -S 'lora_int_id' --oneline --name-only -- scripts/ src/

# A3. Branch-tip dragnet — catches branch-only drivers never merged
for ref in $(git for-each-ref --format='%(refname)' refs/remotes/origin refs/heads); do
  git grep -ln 'LoRARequest(' "$ref" -- scripts src 2>/dev/null | sed "s|^|$ref → |"
done | sort -u

# A4. Read a driver at the commit a run actually executed (never HEAD)
git show <run_commit>:<driver_path>
```

Cross-checks added by the binding plan §12 requirements: grep/pickaxe additionally cover `enable_lora` and harness-style `lora_path=` model_args (constructor-string coverage check); one repo-wide grep runs **without** the `scripts/ src/` path restriction; each per-driver code read notes whether `remove_adapter`/`remove_lora` is called between iterations (the replay sequence is adjusted accordingly — the simulator accepts explicit removals); a driver submitting per-prompt `LoRARequest`s inside one batched generate call is never assumed-order. The dragnet also checks whether the #534 fix commit (`298877f9cc0070b6cc3796a66e81f64f8bc5683c`) is an ancestor of `main` via `git merge-base --is-ancestor`.

For each enumerated driver the audit records four structural facts: (1) the `lora_int_id` expression; (2) engine-creation point vs adapter-loop scope (does one engine span multiple adapters?); (3) `max_loras` / `max_cpu_loras` at engine construction; (4) the realized (id, path) request order implied by the loop structure.

**Scope boundary** (stated up front, not post-hoc): the audit covers git-tracked vLLM-LoRA drivers only. PEFT/HF in-process adapter swapping (string-keyed, path re-read per load), merged-weight reuse, and non-vLLM harness defects are excluded bug classes.

---

## 3. Per-run forensic reconstruction recipe

A driver's *code shape* is necessary but not sufficient — exposure depends on what each historical run actually requested, in order, per process. The reconstruction recipe per candidate run:

1. **Map driver → realized runs.** Locate artifact dirs (out_dir conventions, `eval_results/INDEX.md`) and launch evidence: `tasks/<status>/<N>/events.jsonl` `epm:run-launched` markers carry the literal command line. Count launch/restart markers — one logical run can span several OS processes, and **a process restart resets the adapter cache**, so the replay runs per process, never over one concatenated sequence. Drivers with zero realized historical runs are inventoried but classified in their own taxonomy class (section 7), never granted a SAFE flavor.
2. **Read driver code at the run's recorded commit, not HEAD:** `git show <run_commit>:<driver_path>`. This guards against post-hoc edits masking (or introducing) the bug shape.
3. **Split artifacts by producing process.** Per-cell result JSONs carry `metadata.git_commit` + `metadata.timestamp_utc`; cells are assigned to processes by commit and launch-timestamp windows, with hard asserts on any cell that fits no known process (a mis-assignment raises rather than defaulting).
4. **Handle resume-skip.** Restarted drivers skip already-written cells (`cell_path.exists()`), which changes which adapters each process actually loaded and in what order. The per-process request sequence is reconstructed from per-(epoch, source) **first-touch timestamps** — for a sequential loop, first-touch order IS the enumeration order — and groups whose cells all pre-existed contribute zero adapter events to that process's replay. Internal-consistency asserts check that reconstructed orders agree across epochs of the same process.
5. **Pin the historical vLLM version.** Per candidate run, `git show <run_commit>:uv.lock | grep -A3 'name = "vllm"'`; if the pin differs from 0.11.0, that version's `worker_manager.py` is fetched from the PyPI wheel and the same id-keyed just-touch shape confirmed before the rubric is applied.
6. **Evidence discipline.** Every code fact is cited as `file:line@sha`; every run fact as an artifact path or `events.jsonl` marker; every residency claim as a replay output. Anything unpinnable is classified INDETERMINATE — never silently SAFE.

### Audit parameters

| Parameter | Value | Source |
|---|---|---|
| **Registry capacity (replay)** | **`max_cpu_loras` = `max_loras` = 1** for every audited engine (none sets `max_cpu_loras`) | `vllm/config/lora.py:116-117`; engine-construction reads per driver |
| vLLM semantics version | 0.11.0 (installed `.venv`; per-run pins checked via `uv.lock` at the run commit) | `vllm-0.11.0.dist-info`; plan §4 Step B.4 |
| **Text-identity AFFECTED threshold** (§6 comparison) | **0.9** | `i549_audit_532.py:57` (`IDENTITY_AFFECTED_THRESHOLD`) |
| **Near-zero log-prob threshold** (§6 comparison) | **0.1 nat** (`MAX_DLOGP_NEARZERO_NAT`; pinned because vLLM batching nondeterminism sits well below this while genuine adapter differences in this family are O(1–10) nat) | `i549_audit_532.py:58` |
| #532 expected cell census | 416 epoch-1 + 140 epoch-2 per-cell JSONs (hard assert) | `i549_audit_532.py:108`; plan §2 |
| #532 probe questions per cell | 50 (`load_q_test_extended_50`, len-50 assert at both run commits) | `i549_audit_532.py` docstring; plan §2 |
| #504 replay fraction grid | (0.08, 0.16, 0.33, 0.5, 0.75, 1.0) — 6 checkpoints per cell session; 10 cells = {near, mid_near, mid_far, far, default_only} × seeds {42, 137} | `i549_audit_504.py:48-53` |
| #534 control checkpoint count | 4 distinct fraction paths from the real `fraction_manifest.json` (hard assert: 4 unique) | `i549_calibration_controls.py:43-49` |
| Calibration condition set | 16 conditions from `eval_results/issue_474/cross_eval/loc_ep1/G_logprob_matrix.json` (hard assert) | `i549_calibration_controls.py:61-66` |
| Strided-shard replay | 4 shards, `my_cids = all_cids[k % 4 == shard]` (mirrors `i474_phase4_eval.py:418`) | `i549_calibration_controls.py:74-80` |
| Hash-id collision bounds | any-collision: Σ C(N,2)/2²⁰ per process; capacity-1 operative bound: Σ (N−1)/2²⁰ (a stale serve additionally requires the colliding pair to be consecutive) | `i549_build_audit_table.py:424-426` |
| GPU-hours | **0** (CPU-only, off-pod by construction) | plan §8 |

n/a rows for this task type: base model, LoRA rank/alpha/lr/epochs, seeds, training mix, Hydra config, WandB — no training or generation occurs.

---

## 4. The LRU replay simulator

`scripts/i549_lru_simulate.py` is a ~50-line deterministic mirror of `LRUCacheWorkerLoRAManager.add_adapter` (section 1): an `OrderedDict` keyed by `lora_int_id` mapping to the path **actually loaded**. Hit ⇒ serve the cached path and `move_to_end` (touch); miss ⇒ evict oldest at capacity, load, serve the requested path. Each request yields `{"id", "requested", "served", "hit", "stale"}` where `stale = (served != requested)` — a stale event IS a silent wrong-adapter serve. An optional `removals` argument injects explicit `remove_adapter` calls after a given request index, for drivers that unload between iterations. `summarize()` aggregates to `{n_requests, n_hits, n_stale, stale_requests, verdict_hint}`.

Guard asserts mirror vLLM's contract: `capacity >= 1` and `lora_id >= 1` (id 0 would crash loudly in vLLM, so the simulator refuses it too). The module ships a 7-check `--self-test` pinning the semantics every audit verdict rests on: constant-id multi-checkpoint staleness, distinct-id cleanliness, eviction rescue at capacity 1, the same pattern going stale at capacity 2, benign same-id-same-path hits, true-LRU (not FIFO) eviction order, and explicit-removal residency clearing.

```bash
uv run python scripts/i549_lru_simulate.py --requests req.json --capacity 1 --out out.json
uv run python scripts/i549_lru_simulate.py --self-test
```

The CLI exits 2 if any stale serve is detected, and stamps its output with the git commit, the vLLM semantics source line range, and a timestamp.

### Worked example — simulator input/output (SYNTHETIC)

<!-- SYNTHETIC example constructed for this document to illustrate the record shape and the residency rubric; it is not a row from any audited run. Real replay outputs live in eval_results/issue_549/audit_evidence/. -->

Input request list (`req.json`) — id 1 is reused for a *different* adapter path, with one distinct id in between:

```json
[[1, "adapters/demo_src_ep1"], [2, "adapters/demo_other_ep1"], [1, "adapters/demo_src_ep2"]]
```

Replayed at **capacity 1** (the project default — `max_loras=1`, `max_cpu_loras` unset), the intervening id-2 load evicts id 1, so the repeat is a fresh load of the correct path:

```json
{"n_requests": 3, "n_hits": 0, "n_stale": 0, "verdict_hint": "no-stale-serve", "stale_requests": []}
```

The same sequence replayed at **capacity 2** keeps id 1 resident, and the third request is silently served the first-loaded weights:

```json
{
  "id": 1,
  "requested": "adapters/demo_src_ep2",
  "served": "adapters/demo_src_ep1",
  "hit": true,
  "stale": true
}
```

This pair is the residency rubric in miniature: identical id discipline, opposite exposure, decided entirely by cache capacity and intervening loads — which is why the audit replays realized sequences instead of pattern-matching id expressions.

### Worked example — forensic input (verbatim launch evidence)

The per-run reconstruction (section 3) consumes `epm:run-launched` markers. The #532 launch marker records the literal command line, branch, and commit — e.g. (from `tasks/<status>/532/events.jsonl`, quoted in plan §2/§11):

```
--arm loc --epochs 1 2 3 --sources all --bystanders all --n-probes 50
branch issue-532, commit 28d35f0272876191cb933a1de3980a92b220dda2 (nohup single process)
```

Multiple such markers on one task ⇒ multiple processes ⇒ per-process replays.

---

## 5. Calibration-control protocol

Per the binding plan §12 item 4, runs with *known* outcomes are classified through the SAME simulator machinery used for production verdicts — never by prose argument. `scripts/i549_calibration_controls.py` replays four control groups and **raises on any mismatch with its pre-registered expectation** (exit 0 iff all classify as expected):

| Control | Realized sequence replayed | Pre-registered expectation |
|---|---|---|
| #534 round-1 (positive control) | constant `lora_int_id=1` over the 4 REAL checkpoint paths from `eval_results/issue_534/c504v3_near_seed42/fraction_manifest.json`, capacity 1 | must classify AFFECTED (rubric false-negative check) |
| #534 round-2 (negative control) | same 4 paths with the fix's distinct `ck_i` ids (`enumerate(checkpoint_specs, start=1)` at `298877f9c`) | must classify SAFE (false-positive check on distinct-id multi-LoRA) |
| #474 phase-4 (negative control) | `all_cids.index(cid)+1` over the 16 realized conditions, replayed **unsharded AND as the realized 4-way strided shards**; epoch fixed per invocation | must classify SAFE |
| #460 phase-4 (negative control) | same id discipline over the same 16-condition active set | must classify SAFE |

If the positive control failed to classify AFFECTED, the rubric — not the control — would be treated as wrong (plan §3 H-A). Control inputs are the runs' real artifacts (manifest paths, condition lists), not synthetic stand-ins.

---

## 6. The #532 epoch-1 / epoch-2 comparison protocol

The one audited run whose exposure question needs more than a code read is #532 (`scripts/issue532_predictor_stress.py`): its phase-1 loop resets `lora_int_id = src_idx + 1` per epoch, so epoch-2 requests repeat epoch-1 ids with different adapter paths, under one engine per process, across multiple launches. `scripts/i549_audit_532.py` implements a two-track protocol (plan §4 Steps B.3/B.5 + §12 hardenings). All steps below are PROCEDURE; outcomes live in the task body, not here.

**Track 1 — per-process LRU replay (primary).**

- Every per-cell JSON is assigned to its producing process via `metadata.git_commit` (run commits `28d35f0272876191cb933a1de3980a92b220dda2` and `28a73584c6906282a142c01ea32bf43cbbd9ae32`) plus `metadata.timestamp_utc`; the third launch ran `--phase 2+3+4`, constructs no `LoRARequest`, and an assert fires if any cell post-dates its launch without explanation. An unknown commit raises ("a 4th process?") rather than defaulting.
- Each producing process's (id, path) sequence is rebuilt from per-(epoch, source) first-touch times (sequential loop ⇒ first-touch order = enumeration order); processes with no surviving per-cell artifacts contribute zero adapter events. An order-consistency assert checks the epoch-2 first-touch order against the epoch-1 order.
- Each sequence is replayed at the realized capacity 1, **plus a counterfactual capacity-16 replay**. The counterfactual separates two ways a replay can come out clean: if the capacity-16 replay shows stale serves where the capacity-1 replay does not, any safety is *by eviction* (an accident of the small cache), not by id discipline — and the audit row must say which.

**Track 2 — empirical served-adapter test (confirmatory).** Same engine + same prompt + same served adapter at temperature 0 ⇒ (near-)identical text. For matched (source, bystander, question) triples, epoch-2 vs epoch-1:

- **Pairing hardening (plan §12 item 2):** the probe-question loader is diffed between the two run commits (an empty `git diff` on the loader module pins code-level order); the loader reads a pinned 50-question artifact with a length assert. All zips are `strict=True` — never silent truncation. Identity rates and log-prob diffs are computed **per producing process**, not pooled. If prompt order cannot be pinned, this track is marked UNAVAILABLE — it can never SAFE-confirm on its own.
- **Quantities computed:** greedy `R_trained_per_q` text-identity rate; the FULL `trained_logp_per_q` diff distribution (percentiles p5–p95, mean, and per-cell max |Δlogp| spread — plan §12 item 3 forbids reporting only min/median of maxima); pairs split into diagonal (bystander = source), off-diagonal-within-the-16-condition panel, and other instructed bystanders, because the dose-direction check must compare like with like.
- **Dose-direction cross-check:** #532 reuses #474's loc-arm adapters, so the per-epoch direction of #474's `G_logprob_matrix.json` diagonals/off-diagonals (epoch-2 higher vs lower than epoch-1) supplies an independent prediction for the sign of the #532 epoch-2 − epoch-1 shift on each like-for-like subset.

**Pre-registered decision rule** (`i549_audit_532.py`, replay primary / strings confirmatory): a stale serve in any per-process replay, OR a text-identity rate ≥ 0.9 combined with median per-cell max |Δlogp| ≤ 0.1 nat, signals same-adapter serving; the two tracks agreeing on "stale" ⇒ AFFECTED; conflicting tracks ⇒ INDETERMINATE (escalated, never silently resolved); both clean AND the dose direction matching #474 on the like-for-like subsets ⇒ the SAFE-by-eviction class; dose-direction mismatch ⇒ INDETERMINATE. The thresholds were pinned before the comparison ran (parameters table, section 3).

A parallel diagnostic exists for multi-checkpoint trajectory runs (`scripts/i549_audit_504.py`): replay the realized constant-id checkpoint sequence; then an artifact census computes, per cell, the pairwise **exact-equal rate** of per-(persona, question) base-side log-probs between every fraction pair — adjacent fraction pairs vs distant ones. The pre-registered signature: a FLAT profile (adjacent ≈ distant) is what one set of weights greedily re-generated N times under vLLM batching nondeterminism produces; genuinely different checkpoints spanning a training run produce higher agreement for adjacent than distant pairs. A final **consumption check** tests, by float identity, whether published summary values (`analyze_summary.json` per-cell entries) were read off specific trajectory entries — mapping any cache exposure onto the numbers downstream documents actually cite.

---

## 7. Verdict-class taxonomy (definitions only)

Every audited driver × realized-run row receives exactly one label from a closed enum (plan §12 item 1). Definitions:

| Class | Definition |
|---|---|
| `SAFE-single-adapter` | Exactly one (id, path) per engine-registry lifetime; the only possible repeat is same-id-same-path (a benign cache hit). |
| `SAFE-distinct-id` | Multiple adapters per session, but every adapter gets a unique `lora_int_id` within the process (index-based, manifest-asserted, or step-based); no id ever maps to two paths. |
| `SAFE-fresh-process` | Each adapter (or cell) is served by its own OS process; the id-keyed registry dies with the engine instance, so cross-adapter residency cannot arise. |
| `SAFE-by-eviction` | Ids DO repeat with different paths, but the per-process LRU replay shows every repeat arrived after its old entry was evicted (capacity bound + intervening distinct loads) — a fresh, correct load each time. Distinguished from id-discipline safety by the counterfactual larger-capacity replay. |
| `SAFE-probabilistic` | Ids are runtime-randomized (`hash(...)`-based, `PYTHONHASHSEED`-dependent) and unreproducible post-hoc; the row carries the realized N per process and computed collision bounds (any-collision Σ C(N,2)/2ᵏ; capacity-1 consecutive-pair bound Σ (N−1)/2ᵏ) plus numeric sanity checks, and is labeled as a bound, never as a deterministic clearance. |
| `AFFECTED` | The per-process replay of the realized request sequence shows ≥1 stale serve (a request served weights from a different path than requested), corroborated by artifact evidence where available. |
| `INDETERMINATE` | The session structure or request order cannot be pinned from code + artifacts + markers (or the two evidence tracks conflict); recorded with the re-derivation cost, never guessed into a SAFE class. |
| `NO-REALIZED-RUN` | The driver exists in git but no historical run ever executed it (no launch markers, no artifacts anywhere); inventory-only, never granted a SAFE flavor. |

Assembly: `scripts/i549_build_audit_table.py` emits one row per driver×run into `eval_results/issue_549/audit_table.json` with schema `{driver, sites: ["file:line@sha"], id_discipline, session_structure, engine_capacity, runs, lru_replay, verdict, evidence, affected_published, proposed_correction, reeval_gpu_h, confidence_flags}`; a hard assert rejects any verdict outside the closed enum, and rows whose run-evidence rests on Medium-confidence assumptions carry explicit `confidence_flags`. Any row implicating a published number triggers a *proposal-only* correction entry with a re-eval GPU-hour estimate — the audit edits no clean-result body, RESULTS.md line, or methodology doc (plan ask 3).

---

## 8. Artifacts and reproducibility

- **Code commit (audit scripts + table + evidence):** `2c83c2a6846d566f7ea6ffe26b8b514283bbcabb` (branch `issue-549`; plan-time reconnaissance was at main HEAD `4cc9a6076f5a6af7dee617b922289a4fb9dc63c9`)
- **LRU simulator:** [scripts/i549_lru_simulate.py](https://github.com/superkaiba/explore-persona-space/blob/2c83c2a6846d566f7ea6ffe26b8b514283bbcabb/scripts/i549_lru_simulate.py) (`--self-test` pins the 7 semantics checks)
- **Calibration controls:** [scripts/i549_calibration_controls.py](https://github.com/superkaiba/explore-persona-space/blob/2c83c2a6846d566f7ea6ffe26b8b514283bbcabb/scripts/i549_calibration_controls.py)
- **#532 per-process replay + comparison:** [scripts/i549_audit_532.py](https://github.com/superkaiba/explore-persona-space/blob/2c83c2a6846d566f7ea6ffe26b8b514283bbcabb/scripts/i549_audit_532.py)
- **Trajectory-run forensics:** [scripts/i549_audit_504.py](https://github.com/superkaiba/explore-persona-space/blob/2c83c2a6846d566f7ea6ffe26b8b514283bbcabb/scripts/i549_audit_504.py)
- **Table assembly:** [scripts/i549_build_audit_table.py](https://github.com/superkaiba/explore-persona-space/blob/2c83c2a6846d566f7ea6ffe26b8b514283bbcabb/scripts/i549_build_audit_table.py)
- **Outputs:** [eval_results/issue_549/audit_table.json](https://github.com/superkaiba/explore-persona-space/blob/2c83c2a6846d566f7ea6ffe26b8b514283bbcabb/eval_results/issue_549/audit_table.json) + [eval_results/issue_549/audit_evidence/](https://github.com/superkaiba/explore-persona-space/tree/2c83c2a6846d566f7ea6ffe26b8b514283bbcabb/eval_results/issue_549/audit_evidence) (replay JSONs, comparison JSONs, dragnet extracts)
- **vLLM semantics source:** installed `vllm==0.11.0` (`.venv/lib/python3.11/site-packages/vllm/lora/worker_manager.py:240-267`, `vllm/lora/models.py:391-392,740-742`, `vllm/config/lora.py:35,42,116-117`, `vllm/lora/request.py:34-35`) — read only, never executed
- **Key forensic inputs:** parent fix commit `298877f9cc0070b6cc3796a66e81f64f8bc5683c`; #532 run commits `28d35f0272876191cb933a1de3980a92b220dda2` / `28a73584c6906282a142c01ea32bf43cbbd9ae32`; #504 run commit `affdd82cb0bb31257b5668b327c6af5716212b6c`; per-cell JSONs in-repo under `eval_results/issue_532/per_cell/loc_ep{1,2}/` and `eval_results/issue_474/cross_eval/`; one HF download per #504 cell (`superkaiba1/explore-persona-space-data`, `issue504_geometry/phase1_trajectories/<cell>/trajectory.json`, via `hf_hub_download`)
- **Training data / checkpoints / WandB / Hydra config:** n/a — no training, no generation, no pod
- **Compute:** 0 GPU-hours; CPU-only on the VM (off-pod by construction), wall ≈ 4–6 h dominated by per-driver code reads

Reproduction: run the dragnet commands (section 2) at the pinned SHAs, `i549_lru_simulate.py --self-test`, then `i549_calibration_controls.py`, `i549_audit_504.py`, `i549_audit_532.py`, `i549_build_audit_table.py` — all deterministic given the pinned commits and committed artifacts.

---

*This document describes how the audit was run. For the verdicts and what they mean, see the [task body](https://eps.superkaiba.com/tasks/549).*
