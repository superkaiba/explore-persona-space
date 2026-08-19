#!/usr/bin/env python3
"""Issue #2224 4b-4/5: selection-finetune sweep dispatcher (plan v3 §4).

Runs the ~78 selection-finetune cells discovered from the 4b-2/3 selection
manifests (``issue2224_select``), then the post-finetune on-policy eval
generation and the Batch-API judge waves. Phases (``--phase``):

- ``eval-questions``  [CPU]  per (corpus, trait): the 20 persona-vectors
  held-out eval questions (``issue778_lib.load_trait_data``) + a seeded
  real-prompt panel drawn from the pool EXCLUDING every sample_id any
  selection cell trains on (train/eval disjointness, guideline 3) →
  ``data/issue_2224/eval_questions/<corpus>__<trait>.jsonl``.
- ``train``           [GPU fan-out]  one subprocess per cell, CVD-sharded
  across every provisioned GPU (one multi-GPU pod, never N single-GPU pods;
  ``CUDA_VISIBLE_DEVICES`` pinned in the LAUNCHER env per cell — the #545
  import-time-cuInit clobber gotcha), work-conserving queue. Per cell:
  ``train_lora`` with the paper §6.3 recipe VERBATIM (plan §4:
  lr=1e-5, epochs=10, r=32, α=64, rsLoRA, batch 16, grad_accum 1,
  max_length=2048, warmup 0, linear schedule via ``lr_scheduler_type``,
  weight_decay 0) + fail-loud per-cell adapter upload to the model repo
  ``issue2224_screening/adapters/<cell>/`` + a per-cell done sentinel —
  a crash never loses completed cells; the resume predicate
  (``pending_train_cells``) is the SAME function the fresh run and the
  pending-aware disk-headroom gate use (#1586).
- ``eval``            [GPU fan-out]  post-ft generation, vLLM multi-LoRA
  serving (plan A15: ``enable_lora=True, max_lora_rank=32`` via the reused
  ``issue778_lib.build_vllm_engine``); registered fallback
  ``--eval-mode merged`` = per-cell merge→generate→reap→delete
  (merge-read-delete keeps the pod under the MooseFS quota; engine reaped
  via the #685 ``reap_vllm_engine`` recipe). 100 questions × ``--gen-draws``
  (default 5, ``SamplingParams(n=K)``) per cell + the base-model cells (the
  Δ-denominator row, plan §5). Per-cell atomic generation files + cap-hit
  fraction recorded (re-gen trigger >2%).
- ``upload``          [pod, before teardown]  ONE bulk fail-loud
  ``upload_folder`` commit of ``postft_eval/`` (+ eval questions) to the HF
  data repo prefix with an exact-set scoped verify, then ``[phase=done]``
  (plan §9 off_pod contract).
- ``judge-pilot``     [VM]  rule-26 pilot gates: one per trait rubric + one
  for the coherence rubric (``eval.judge_pilot.judge_pilot_gate``).
- ``judge``           [VM, Batch API]  graded 0-100 Sonnet judging, N=5
  judge draws per generation, trait AND coherence rubrics (one behavior per
  call), via ``eval.graded_judge.judge_graded`` (the #663-hardened Batch
  machinery — never a hand-rolled poller). Pilot-gated (≥5k-call waves).
  Per cell → ``eval_results/issue_2224/selection_finetune/<cell>/
  trait_scores.json`` (plan §6.5 primary-deliverable glob): graded mean
  (headline) + rate>50 companion (dual-DV), coherence gate
  (``COHERENCE_THRESHOLD`` 50 — incoherent cells FLAGGED, never silently
  pooled), full drop telemetry (content / transport / truncation /
  api-refusal splits, rules 9/23/24/28).

Sentinels (plan §9 ``phase_outputs``): ``eval_results/issue_2224/.done_4b4``
after train, ``.done_4b5`` after judge.

Content hygiene: prompt/response text flows only file→file; never printed.
Judge raw draws + caches live under ``data/issue_2224/`` (not git).

Usage::

    uv run python scripts/issue2224_finetune_sweep.py --phase eval-questions --pv-root ...
    uv run python scripts/issue2224_finetune_sweep.py --phase train --gpus 0,1,2,3
    uv run python scripts/issue2224_finetune_sweep.py --phase eval --gpus 0,1,2,3
    uv run python scripts/issue2224_finetune_sweep.py --phase upload
    uv run python scripts/issue2224_finetune_sweep.py --phase judge --pv-root ... [--dry-run]
    uv run python scripts/issue2224_finetune_sweep.py --import-check
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import logging
import math
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/vllm imports: thread caps (VM) + HF/WANDB/API tokens (#847)

from issue2224_common import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    load_jsonl,
    repro_meta,
    sha256_file,
    stable_seed,
)
from issue778_lib import COHERENCE_THRESHOLD, MODEL_NAME  # noqa: E402

logger = logging.getLogger("issue2224_sweep")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

SWEEP_SCHEMA_VERSION = 1
CAP_HIT_REGEN_TRIGGER = 0.02
PILOT_WAVE_FLOOR = 5000
# Rule-26 pilot sizing (r1 review C1): the gate FAILs any arm whose effective
# draws fall below judge_pilot_gate's min_effective_draws_per_arm=10 floor, so
# the pilot budget MUST derive from the arm count — never lower the floor,
# never --skip-pilot-gate around it.
PILOT_N_DRAWS = 2
PILOT_MIN_EFFECTIVE_DRAWS = 10  # judge_pilot_gate's min_effective_draws_per_arm default
PILOT_DRAWS_MARGIN = 2  # headroom so a lone transport-lost draw cannot hollow an arm


def pilot_target_draws(n_arms: int, n_draws: int, requested: int) -> int:
    """Arm-count-derived pilot draw budget (r1 review C1).

    ``judge_pilot_gate`` subsamples ``per_arm_items = max(1, target //
    (n_arms * n_draws))`` items per arm; every arm needs ``per_arm_items *
    n_draws >= floor + margin`` effective draws or the gate FAILs on hollow
    evidence AFTER burning the pilot spend (84 coherence arms at the shipped
    200-draw default -> 2 draws/arm < the 10-draw floor). Returns
    ``max(requested, n_arms * n_draws * ceil((floor+margin)/n_draws))`` and
    asserts the gate's own arithmetic clears the floor pre-dispatch.
    """
    n_draws = max(1, n_draws)
    per_arm_items = math.ceil((PILOT_MIN_EFFECTIVE_DRAWS + PILOT_DRAWS_MARGIN) / n_draws)
    target = max(int(requested), n_arms * n_draws * per_arm_items)
    effective = max(1, target // (n_arms * n_draws)) * n_draws  # the gate's own sizing
    if effective < PILOT_MIN_EFFECTIVE_DRAWS:
        raise RuntimeError(
            f"pilot budget {target} yields {effective} effective draws/arm across "
            f"{n_arms} arms < the {PILOT_MIN_EFFECTIVE_DRAWS}-draw gate floor"
        )
    return target


HF_ADAPTER_PREFIX = "issue2224_screening/adapters"
HF_POSTFT_PREFIX = "issue2224_screening/raw_completions/postft_eval"

SELECTIONS_DIR_DEFAULT = PROJECT_ROOT / "eval_results" / "issue_2224" / "selections"
OUT_ROOT_DEFAULT = PROJECT_ROOT / "data" / "issue_2224" / "screening_ft"
EVAL_Q_DIR_DEFAULT = PROJECT_ROOT / "data" / "issue_2224" / "eval_questions"
TRAIT_SCORES_DIR_DEFAULT = PROJECT_ROOT / "eval_results" / "issue_2224" / "selection_finetune"
PILOT_REPORT_DIR_DEFAULT = PROJECT_ROOT / "eval_results" / "issue_2224" / "judge_pilots"
JUDGE_ROOT_DEFAULT = PROJECT_ROOT / "data" / "issue_2224" / "judge_postft"
SENTINEL_DIR = PROJECT_ROOT / "eval_results" / "issue_2224"

# Seed-isolation refusal guard (fu-r2 review blocker): a non-42 seed with any
# PARENT-DEFAULT state/results path on a state-touching phase silently clobbers
# the seed-42 artifacts — judged_current sha-mismatches and _judge_one_cell
# OVERWRITES the committed selection_finetune/<cid>/trait_scores.json, and
# _check_pilot is satisfied by the PARENT's passing pilot reports (rule-26
# bypass). Per-phase mapping = exactly the paths that phase reads/writes, so a
# legitimate partial invocation (e.g. train, which never touches judge dirs)
# is not forced to pass flags it does not use. Inert for seed-42 runs.
_SEED_ISOLATION_REQUIRED: dict[str, tuple[str, ...]] = {
    # hf_prefix_suffix rows (review r2 BLOCKER 1): train/train-cell upload
    # per-cell adapters to the HF model repo + name wandb runs; upload bulk-
    # persists gens + questions to the HF data repo. Isolated LOCAL dirs alone
    # leave those writes on the PARENT prefixes/run names — the suffix is the
    # HF/wandb half of the isolation contract (NON-EMPTY semantics below).
    "train": ("out_root", "hf_prefix_suffix"),
    "train-cell": ("out_root", "hf_prefix_suffix"),
    "eval": ("out_root", "eval_questions_dir"),
    "eval-shard": ("out_root", "eval_questions_dir"),
    "upload": ("out_root", "eval_questions_dir", "hf_prefix_suffix"),
    "eval-questions": ("eval_questions_dir",),
    "judge-pilot": (
        "out_root",
        "eval_questions_dir",
        "trait_scores_dir",
        "judge_root",
        "pilot_report_dir",
    ),
    "judge": (
        "out_root",
        "eval_questions_dir",
        "trait_scores_dir",
        "judge_root",
        "pilot_report_dir",
    ),
}
_SEED_ISOLATION_DEFAULTS: dict[str, Path] = {
    "out_root": OUT_ROOT_DEFAULT,
    "eval_questions_dir": EVAL_Q_DIR_DEFAULT,
    "trait_scores_dir": TRAIT_SCORES_DIR_DEFAULT,
    "judge_root": JUDGE_ROOT_DEFAULT,
    "pilot_report_dir": PILOT_REPORT_DIR_DEFAULT,
}


def assert_seed_isolation(args) -> None:
    """Refuse a non-42 seed run whose phase-relevant isolation is incomplete.

    Two halves per phase mapping: LOCAL dirs must differ from the parent
    defaults under ``Path.resolve()`` (review r3 BLOCKER 2 — a RELATIVE
    spelling of the parent default must not evade the guard and resume from
    parent state), and ``hf_prefix_suffix`` must be NON-EMPTY on the phases
    that write HF prefixes / wandb run names (review r3 BLOCKER 1 — isolated
    local dirs with an empty suffix overwrite the parent's adapters /
    postft_eval / eval_questions prefixes and reuse parent run names).
    """
    if args.seed == 42 or args.phase not in _SEED_ISOLATION_REQUIRED:
        return
    offending = []
    for attr in _SEED_ISOLATION_REQUIRED[args.phase]:
        if attr == "hf_prefix_suffix":
            if not args.hf_prefix_suffix:
                offending.append("--hf-prefix-suffix (empty — parent HF prefixes/wandb names)")
        elif Path(getattr(args, attr)).resolve() == _SEED_ISOLATION_DEFAULTS[attr].resolve():
            offending.append(f"--{attr.replace('_', '-')}")
    if offending:
        raise RuntimeError(
            f"--seed {args.seed} != 42 on --phase {args.phase} with parent-default "
            f"isolation for {offending} — a replication seed MUST isolate every "
            f"state/results surface it touches: pass '<default>_seed{args.seed}' "
            f"PATH variants for the named dirs AND a non-empty --hf-prefix-suffix "
            f"(e.g. '_seed{args.seed}') where named (seed-42 clobber guard; see "
            f"scripts/issue2224_followup_r2_runner.sh / _judge.sh)"
        )


# Paper §6.3 real-world selection-finetune recipe, VERBATIM (plan §4 / §11 —
# `lr_scheduler_type`, NOT `lr_scheduler`; the unset sft.py default is "cosine",
# so the explicit linear pin is load-bearing for paper fidelity).
PAPER_RECIPE = dict(
    lr=1e-5,
    epochs=10,
    lora_r=32,
    lora_alpha=64,
    use_rslora=True,
    batch_size=16,
    grad_accum=1,
    max_length=2048,
    warmup_ratio=0.0,
    lr_scheduler_type="linear",
    weight_decay=0.0,
)

# Pending-aware disk-headroom constants (GB; assert_out_root_headroom preamble).
TRAIN_FIXED_GB = 5.0
TRAIN_PER_CELL_GB = 1.5  # adapter ~0.08 + LoRA optimizer transients + logs, with margin
EVAL_FIXED_GB = 2.0
EVAL_MERGED_TRANSIENT_GB = 20.0  # one merged bf16 7B dir at a time (merge-read-delete)


# ── Cell discovery + state ───────────────────────────────────────────────────────


def discover_cells(selections_dir: Path) -> list[dict]:
    """Trainable cells from the 4b-2/3 manifests (status ok, train JSONL present).

    ``filter-collapsed`` cells are reported findings, never trained (plan §4).
    """
    cells = []
    for mpath in sorted(Path(selections_dir).glob("*/*/*.json")):
        if mpath.name in ("filter_candidates.json", "filter_scores.json"):
            continue
        m = json.loads(mpath.read_text())
        if m.get("status") != "ok":
            logger.info("[cells] skip %s status=%s", m.get("cell_id"), m.get("status"))
            continue
        tj = m.get("train_jsonl", {}).get("path")
        if not tj:
            raise RuntimeError(f"{mpath}: status ok but no train_jsonl recorded")
        cells.append(
            {
                "cell_id": m["cell_id"],
                "corpus": m["corpus"],
                "trait": m["trait"],
                "method": m["method"],
                "tail": m["tail"],
                "train_jsonl": tj,
                "manifest": str(mpath),
            }
        )
    if not cells:
        raise RuntimeError(f"no trainable cells under {selections_dir} — run issue2224_select")
    return cells


def base_cells(cells: list[dict]) -> list[dict]:
    """One base-model pseudo-cell per (corpus, trait) present (the Δ denominator)."""
    seen = sorted({(c["corpus"], c["trait"]) for c in cells})
    return [
        {
            "cell_id": f"{corpus}__{trait}__base__na",
            "corpus": corpus,
            "trait": trait,
            "method": "base",
            "tail": "na",
            "train_jsonl": None,
            "manifest": None,
        }
        for corpus, trait in seen
    ]


def filter_cells(cells: list[dict], only: str | None) -> list[dict]:
    if not only:
        return cells
    keep = {c.strip() for c in only.split(",") if c.strip()}
    out = [c for c in cells if c["cell_id"] in keep]
    missing = keep - {c["cell_id"] for c in out}
    if missing:
        raise RuntimeError(f"--cells names unknown cell_ids: {sorted(missing)}")
    return out


def train_done_path(out_root: Path, cell_id: str) -> Path:
    return Path(out_root) / "state" / f"train_{cell_id}.done.json"


def adapter_dir(out_root: Path, cell_id: str) -> Path:
    return Path(out_root) / "adapters" / cell_id


def gen_dir(out_root: Path, cell_id: str) -> Path:
    return Path(out_root) / "postft_eval" / cell_id


def pending_train_cells(cells: list[dict], out_root: Path) -> list[dict]:
    """The ONE resume predicate train dispatch + the headroom gate both use (#1586)."""
    return [c for c in cells if not train_done_path(out_root, c["cell_id"]).exists()]


def expected_gen_rows(args) -> int:
    return args.n_questions * args.gen_draws


def gen_complete(out_root: Path, cell_id: str, expected: int, args) -> bool:
    """Count- AND decoding-regime-keyed resume predicate for eval generations.

    (r1 review M2 sibling — the #722-r3 resume-regime gap.) Complete iff the
    row count matches AND the recorded decoding regime matches the current
    args: a temperature/draws mismatch fails LOUD (a silent resume would mix
    regimes); a RAISED ``--max-new-tokens`` keeps never-truncated cells
    (``cap_hit_fraction == 0`` — a cap raise cannot affect them) and re-opens
    cap-hit cells under ``--regen-truncated`` (the pre-registered >2% cap-hit
    re-gen trigger, now executable).
    """
    gpath = gen_dir(out_root, cell_id) / "generations.jsonl"
    mpath = gen_dir(out_root, cell_id) / "meta.json"
    if not (gpath.exists() and mpath.exists()):
        return False
    if len(load_jsonl(gpath)) != expected:
        return False
    meta = json.loads(mpath.read_text())
    dec = meta.get("decoding") or {}
    if dec.get("temperature") != args.gen_temperature or dec.get("n") != args.gen_draws:
        raise RuntimeError(
            f"{cell_id}: existing generations decoding={dec} mismatches the current "
            f"temperature/draws args — a resume must never silently mix decoding "
            f"regimes; delete the cell dir or match the args"
        )
    old_cap = dec.get("max_new_tokens")
    if old_cap is None:
        raise RuntimeError(f"{cell_id}: meta.json decoding lacks max_new_tokens — regenerate")
    cap_hit = float(meta.get("cap_hit_fraction") or 0.0)
    if int(old_cap) != args.max_new_tokens:
        if args.max_new_tokens > int(old_cap) and cap_hit == 0.0:
            return True  # a raised cap cannot affect a never-truncated cell
        if args.regen_truncated:
            return False  # re-generate the whole cell at the new cap
        raise RuntimeError(
            f"{cell_id}: generated at max_new_tokens={old_cap} (cap_hit={cap_hit}) != "
            f"current {args.max_new_tokens} — pass --regen-truncated to re-generate "
            f"cap-hit cells at the new cap, or match the cap"
        )
    if args.regen_truncated and cap_hit > 0.0:
        return False  # same cap, but the >2% trigger asked for a re-gen pass
    return True


def pending_eval_cells(cells: list[dict], out_root: Path, expected: int, args) -> list[dict]:
    return [c for c in cells if not gen_complete(out_root, c["cell_id"], expected, args)]


def detect_gpus(args) -> list[str]:
    """Fan out over every visible GPU by default (guideline 2)."""
    if args.gpus:
        return [g.strip() for g in args.gpus.split(",") if g.strip()]
    import torch

    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("no CUDA device visible — pass --gpus or run on a GPU pod")
    return [str(i) for i in range(n)]


def write_sentinel(name: str, payload: dict) -> None:
    SENTINEL_DIR.mkdir(parents=True, exist_ok=True)
    payload = {**payload, "meta": repro_meta(f"issue2224_finetune_sweep.{name}")}
    atomic_write_json(payload, SENTINEL_DIR / name)


def finalize_phase_sentinel(
    name: str, phase: str, payload: dict, cells_filter: str | None, seed: int = 42
) -> bool:
    """Drop the plan-§9 phase-done sentinel — ONLY for an UNFILTERED seed-42 run (M1).

    A ``--cells``-filtered (smoke-slice) run completing its subset must not
    emit the phase-done signal (`.done_4b4`/`.done_4b5` + the ``[phase=done]``
    line) the poller/orchestrator consumes as phase completion. Likewise a
    replication seed (``seed != 42``, review r3 item a): even UNFILTERED, it
    must never write the PARENT's sentinel files under eval_results/issue_2224.
    """
    if cells_filter is not None or seed != 42:
        # NB: never spell the literal phase-done token here — a poller greps for it.
        why = "--cells subset" if cells_filter is not None else f"seed {seed} != 42"
        print(
            f"[{phase}] {why} complete — {name} sentinel + phase-done line "
            f"SKIPPED (not parent seed-42 phase completion)",
            flush=True,
        )
        return False
    write_sentinel(name, payload)
    print(f"[phase=done] {phase} n_cells={payload.get('n_cells')}", flush=True)
    return True


# ── Phase: eval-questions ────────────────────────────────────────────────────────


def assert_selection_census(selections_dir: Path, corpus: str, trait: str) -> None:
    """M4: refuse to draw the eval panel before the FULL selection census exists.

    The panel excludes the sample_ids of manifests present AT BUILD TIME, so a
    panel drawn between ``select`` and ``apply-filter`` would let later-arriving
    ``top_filtered`` cells train on panel prompts (silent train/eval
    contamination, guideline 3). Census per method present in the block:
    ``top`` + ``bottom`` status=ok, ``top_filtered`` present (ok OR
    filter-collapsed), plus the shared random cell.
    """
    d = Path(selections_dir) / corpus / trait
    manifests: dict[tuple[str, str], dict] = {}
    for p in sorted(d.glob("*.json")):
        if p.name in ("filter_candidates.json", "filter_scores.json"):
            continue
        m = json.loads(p.read_text())
        manifests[(m["method"], m["tail"])] = m
    methods = sorted({meth for meth, _tail in manifests if meth != "random"})
    if not methods:
        raise RuntimeError(f"{corpus}/{trait}: no selection manifests under {d}")
    problems: list[str] = []
    for meth in methods:
        for tail in ("top", "bottom"):
            m = manifests.get((meth, tail))
            if m is None or m.get("status") != "ok":
                problems.append(f"{meth}__{tail} missing/not-ok")
        mf = manifests.get((meth, "top_filtered"))
        if mf is None:
            problems.append(
                f"{meth}__top_filtered MISSING (run apply-filter BEFORE eval-questions)"
            )
        elif mf.get("status") not in ("ok", "filter-collapsed"):
            problems.append(f"{meth}__top_filtered status={mf.get('status')!r}")
    if ("random", "shared") not in manifests:
        problems.append("random__shared missing")
    if problems:
        raise RuntimeError(
            f"{corpus}/{trait}: selection census INCOMPLETE — the train/eval disjointness "
            f"exclusion would miss later-arriving selections (M4, guideline 3): {problems}"
        )


def run_eval_questions(args) -> int:
    """PV eval questions + seeded real-prompt panel, disjoint from every trained id."""
    from issue778_lib import load_trait_data

    if args.pv_root is None:
        raise RuntimeError("--pv-root required (persona_vectors clone; PV eval questions)")
    cells = discover_cells(args.selections_dir)
    out_dir = Path(args.eval_questions_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for corpus, trait in sorted({(c["corpus"], c["trait"]) for c in cells}):
        assert_selection_census(args.selections_dir, corpus, trait)
        selected: set[str] = set()
        for c in cells:
            if c["corpus"] == corpus and c["trait"] == trait:
                selected.update(json.loads(Path(c["manifest"]).read_text())["sample_ids"])
        pool_path = Path(args.pools_dir) / f"{corpus}.jsonl"
        pool_rows = load_jsonl(pool_path)
        eligible = [r for r in pool_rows if str(r["sample_id"]) not in selected]
        if len(eligible) < args.panel_size:
            raise RuntimeError(
                f"{corpus}/{trait}: only {len(eligible)} pool prompts left outside the "
                f"selections — cannot draw a {args.panel_size}-prompt panel"
            )
        import numpy as np

        rng = np.random.default_rng(stable_seed("evalpanel", corpus, trait, base=args.seed))
        panel = [eligible[i] for i in rng.permutation(len(eligible))[: args.panel_size]]
        td = load_trait_data(Path(args.pv_root), trait)
        rows = [
            {"qid": f"pv-{i:02d}", "question": q, "source": "pv_eval"}
            for i, q in enumerate(td.eval_questions)
        ]
        rows += [
            {"qid": f"real-{r['sample_id']}", "question": str(r["prompt"]), "source": "pool"}
            for r in panel
        ]
        for r in rows:
            if "__" in r["qid"]:
                raise RuntimeError(f"qid {r['qid']!r} contains '__' (judge custom_id delimiter)")
        if len(rows) != args.n_questions:
            raise RuntimeError(
                f"{corpus}/{trait}: built {len(rows)} questions != --n-questions "
                f"{args.n_questions} (PV {len(td.eval_questions)} + panel {args.panel_size})"
            )
        path = out_dir / f"{corpus}__{trait}.jsonl"
        atomic_write_jsonl(rows, path)
        logger.info(
            "[eval-questions] %s: %d questions (%d pv + %d pool)",
            path.name,
            len(rows),
            len(td.eval_questions),
            len(panel),
        )
    return 0


def load_eval_questions(args, corpus: str, trait: str) -> list[dict]:
    path = Path(args.eval_questions_dir) / f"{corpus}__{trait}.jsonl"
    if not path.exists():
        raise RuntimeError(f"{path} missing — run --phase eval-questions first")
    rows = load_jsonl(path)
    if len(rows) != args.n_questions:
        raise RuntimeError(f"{path}: {len(rows)} rows != --n-questions {args.n_questions}")
    return rows


# ── Phase: train (fan-out) + train-cell (worker) ─────────────────────────────────


def _spawn(cmd: list[str], gpu: str, log_path: Path):
    """Subprocess with CVD pinned in the LAUNCHER env + explicit env passthrough."""
    env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
    lf = open(log_path, "a")
    return subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf), lf


def _work_conserving_fanout(
    jobs: list[tuple[str, list[str]]], gpus: list[str], log_dir: Path, phase: str
) -> list[str]:
    """Run (job_id, cmd) jobs across gpus, one process per GPU, work-conserving.

    Returns the job_ids that FAILED (rc != 0). Completed jobs are already
    per-cell checkpointed by the workers, so failures never lose siblings.
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    queue = list(jobs)
    running: dict[str, tuple] = {}  # gpu -> (job_id, proc, lf, t0)
    failed: list[str] = []
    n_total = len(jobs)
    n_done = 0
    t_start = time.time()
    while queue or running:
        for gpu in gpus:
            if gpu not in running and queue:
                job_id, cmd = queue.pop(0)
                proc, lf = _spawn(cmd, gpu, log_dir / f"{job_id}.log")
                running[gpu] = (job_id, proc, lf, time.time())
                logger.info("[%s] dispatch %s -> GPU %s", phase, job_id, gpu)
        for gpu in list(running):
            job_id, proc, lf, t0 = running[gpu]
            rc = proc.poll()
            if rc is None:
                continue
            lf.close()
            del running[gpu]
            n_done += 1
            if rc != 0:
                failed.append(job_id)
                logger.error("[%s] FAILED %s rc=%d (log: %s)", phase, job_id, rc, log_dir)
            print(
                f"[{phase}] unit {n_done}/{n_total} {job_id} rc={rc} "
                f"elapsed={time.time() - t_start:.0f}s",
                flush=True,
            )
        if running:
            time.sleep(5)
    return failed


def run_train(args) -> int:
    """4b-4: CVD-sharded per-cell finetunes; per-cell checkpoint + resume."""
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    cells = filter_cells(discover_cells(args.selections_dir), args.cells)
    out_root = Path(args.out_root)
    pending = pending_train_cells(cells, out_root)
    logger.info("[train] cells=%d pending=%d", len(cells), len(pending))
    if not pending:
        logger.info("[train] zero pending — headroom gate skipped (resume-aware, #1586)")
    else:
        assert_out_root_headroom(
            out_root,
            TRAIN_FIXED_GB + TRAIN_PER_CELL_GB * len(pending),
            phase="4b-4 train",
        )
        gpus = detect_gpus(args)
        jobs = []
        for c in pending:
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--phase",
                "train-cell",
                "--cell",
                c["cell_id"],
                "--selections-dir",
                str(args.selections_dir),
                "--out-root",
                str(out_root),
                "--base-model",
                args.base_model,
                "--seed",
                str(args.seed),
            ]
            if args.hf_prefix_suffix:
                cmd += ["--hf-prefix-suffix", args.hf_prefix_suffix]
            if args.max_steps is not None:
                cmd += ["--max-steps", str(args.max_steps)]
            if args.batch_size_override is not None:
                cmd += ["--batch-size-override", str(args.batch_size_override)]
            if args.grad_accum_override is not None:
                cmd += ["--grad-accum-override", str(args.grad_accum_override)]
            if args.no_upload:
                cmd += ["--no-upload"]
            if args.no_wandb:
                cmd += ["--no-wandb"]
            if args.cpu:
                cmd += ["--cpu"]
            jobs.append((c["cell_id"], cmd))
        failed = _work_conserving_fanout(jobs, gpus, out_root / "logs" / "train", "train")
        if failed:
            raise RuntimeError(
                f"[train] {len(failed)} cell(s) failed: {failed} — completed cells are "
                f"checkpointed; fix and re-run to resume"
            )
    still = pending_train_cells(cells, out_root)
    if still:
        raise RuntimeError(f"[train] {len(still)} cells still pending after fan-out: {still[:5]}")
    finalize_phase_sentinel(
        ".done_4b4",
        "train",
        {"phase": "4b-4 train", "n_cells": len(cells)},
        args.cells,
        seed=args.seed,
    )
    return 0


def run_train_cell(args) -> int:
    """One selection-finetune cell (paper §6.3 recipe verbatim) + fail-loud upload."""
    if not args.cell:
        raise RuntimeError("--phase train-cell requires --cell")
    cells = [c for c in discover_cells(args.selections_dir) if c["cell_id"] == args.cell]
    if not cells:
        raise RuntimeError(f"unknown cell {args.cell!r}")
    cell = cells[0]
    train_jsonl = Path(cell["train_jsonl"])
    if not train_jsonl.exists():
        raise RuntimeError(f"{args.cell}: train JSONL missing at {train_jsonl}")
    out_root = Path(args.out_root)
    adir = adapter_dir(out_root, args.cell)
    adir.mkdir(parents=True, exist_ok=True)

    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    # OOM-recovery batch composition (default inert): per-device batch may be
    # traded against grad-accum ONLY at a preserved effective batch, so the
    # paper recipe's optimization is unchanged (TRL's num_items_in_batch keeps
    # token-mean normalization exact under accumulation).
    recipe = dict(PAPER_RECIPE)
    if (args.batch_size_override is None) != (args.grad_accum_override is None):
        raise RuntimeError("--batch-size-override and --grad-accum-override come as a pair")
    if args.batch_size_override is not None:
        eff_new = args.batch_size_override * args.grad_accum_override
        eff_ref = PAPER_RECIPE["batch_size"] * PAPER_RECIPE["grad_accum"]
        if eff_new != eff_ref:
            raise RuntimeError(
                f"effective batch {eff_new} != paper recipe {eff_ref} — the override is "
                f"an OOM-recovery batch COMPOSITION change, never a dose change"
            )
        recipe["batch_size"] = args.batch_size_override
        recipe["grad_accum"] = args.grad_accum_override

    # WANDB_INTENTIONALLY_DISABLED: --no-wandb is the CPU/offline smoke escape only;
    # production default is report_to="wandb" with a distinct per-cell run name.
    report_to = "none" if args.no_wandb else "wandb"
    cfg = TrainLoraConfig(
        gpu_id=0,  # CVD pinned by the launcher env — one visible device per cell
        seed=args.seed,
        run_name=f"i2224_ft_{args.cell}{args.hf_prefix_suffix}",
        report_to=report_to,
        save_strategy="no",
        max_steps=args.max_steps,
        bf16=not args.cpu,
        gradient_checkpointing=not args.cpu,
        **recipe,
    )
    t0 = time.time()
    out_dir, loss = train_lora(args.base_model, str(train_jsonl), str(adir), cfg=cfg)
    elapsed = time.time() - t0

    upload_url = ""
    if not args.no_upload:
        from explore_persona_space.orchestrate.hub import DEFAULT_MODEL_REPO, _upload

        upload_url = _upload(
            adir,
            DEFAULT_MODEL_REPO,
            "model",
            f"{HF_ADAPTER_PREFIX}{args.hf_prefix_suffix}/{args.cell}",
            raise_on_error=True,
        )
        if not upload_url:
            raise RuntimeError(f"[train-cell] adapter upload returned no path for {args.cell}")

    # Fail-loud weight discovery (r1 review): a train_lora output-layout change
    # must never silently record a null adapter sha.
    weight_files = sorted(adir.glob("adapter_model*.safetensors"))
    if not weight_files:
        raise RuntimeError(
            f"[train-cell] {args.cell}: no adapter_model*.safetensors under {adir} — "
            f"train_lora output layout changed; refusing to record a null adapter sha"
        )
    done = {
        "schema": SWEEP_SCHEMA_VERSION,
        "cell_id": args.cell,
        "train_jsonl": {"path": str(train_jsonl), "sha256": sha256_file(train_jsonl)},
        "recipe": {
            **recipe,
            "base_model": args.base_model,
            "seed": args.seed,
            "max_steps": args.max_steps,
        },
        "training_loss": float(loss),
        "elapsed_s": round(elapsed, 1),
        "adapter_dir": str(out_dir),
        "adapter_sha256": {f.name: sha256_file(f) for f in weight_files},
        "upload_url": upload_url,
        "meta": repro_meta("issue2224_finetune_sweep.train-cell"),
    }
    dpath = train_done_path(out_root, args.cell)
    dpath.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_json(done, dpath)
    logger.info("[train-cell] %s loss=%.4f elapsed=%.0fs", args.cell, loss, elapsed)
    return 0


# ── Phase: eval (fan-out) + eval-shard (worker) ──────────────────────────────────


def run_eval(args) -> int:
    """4b-5 generation: shard cells (incl. base cells) across GPUs."""
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    cells = filter_cells(discover_cells(args.selections_dir), args.cells)
    trained_pending = pending_train_cells(cells, Path(args.out_root))
    if trained_pending:
        raise RuntimeError(
            f"[eval] {len(trained_pending)} cells not trained yet (run --phase train): "
            f"{[c['cell_id'] for c in trained_pending][:5]}"
        )
    # base pseudo-cells (the Δ denominator) always ride along for the (corpus, trait)
    # blocks the --cells filter kept — they are auto-derived, never named in --cells.
    all_cells = cells + base_cells(cells)
    out_root = Path(args.out_root)
    expected = expected_gen_rows(args)
    pending = pending_eval_cells(all_cells, out_root, expected, args)
    logger.info("[eval] cells=%d (incl. base) pending=%d", len(all_cells), len(pending))
    if not pending:
        logger.info("[eval] zero pending — headroom gate skipped (resume-aware, #1586)")
        print("[phase=done] eval (all cells already generated)", flush=True)
        return 0
    gpus = detect_gpus(args)
    need = EVAL_FIXED_GB + 0.05 * len(pending)
    if args.eval_mode == "merged":
        # M3: EVERY concurrent GPU shard holds its own merged bf16 7B dir
        # (merge-read-delete) — size the transient to the realized fan-out width.
        need += EVAL_MERGED_TRANSIENT_GB * min(len(gpus), len(pending))
    assert_out_root_headroom(out_root, need, phase="4b-5 eval-gen")
    shards = {g: [] for g in gpus}
    for i, c in enumerate(pending):
        shards[gpus[i % len(gpus)]].append(c["cell_id"])
    jobs = []
    for si, gpu in enumerate(gpus):
        if not shards[gpu]:
            continue
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--phase",
            "eval-shard",
            "--cells",
            ",".join(shards[gpu]),
            "--selections-dir",
            str(args.selections_dir),
            "--out-root",
            str(out_root),
            "--eval-questions-dir",
            str(args.eval_questions_dir),
            "--base-model",
            args.base_model,
            "--eval-mode",
            args.eval_mode,
            "--gen-draws",
            str(args.gen_draws),
            "--n-questions",
            str(args.n_questions),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--gen-temperature",
            str(args.gen_temperature),
            "--gpu-mem-util",
            str(args.gpu_mem_util),
            "--seed",
            str(args.seed),
        ]
        if args.regen_truncated:
            cmd += ["--regen-truncated"]
        jobs.append((f"evalshard{si:02d}", cmd))
    failed = _work_conserving_fanout(jobs, gpus, out_root / "logs" / "eval", "eval")
    if failed:
        raise RuntimeError(f"[eval] shard(s) failed: {failed} — re-run to resume")
    still = pending_eval_cells(all_cells, out_root, expected, args)
    if still:
        raise RuntimeError(f"[eval] {len(still)} cells still pending: {still[:5]}")
    print(f"[phase=done] eval n_cells={len(all_cells)}", flush=True)
    return 0


def _cells_by_id(args, ids: list[str]) -> list[dict]:
    cells = discover_cells(args.selections_dir)
    universe = {c["cell_id"]: c for c in cells + base_cells(cells)}
    missing = [i for i in ids if i not in universe]
    if missing:
        raise RuntimeError(f"unknown cell_ids: {missing}")
    return [universe[i] for i in ids]


def run_eval_shard(args) -> int:
    """One GPU's eval shard: multi-LoRA serving (A15) or merged fallback."""
    if not args.cells:
        raise RuntimeError("--phase eval-shard requires --cells")
    ids = [c.strip() for c in args.cells.split(",") if c.strip()]
    shard_cells = _cells_by_id(args, ids)
    out_root = Path(args.out_root)
    expected = expected_gen_rows(args)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.base_model)

    from vllm import SamplingParams

    from issue778_lib import build_vllm_engine, reap_vllm_engine

    def _generate_cell(llm, cell: dict, lora_request) -> None:
        qs = load_eval_questions(args, cell["corpus"], cell["trait"])
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": str(q["question"])}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for q in qs
        ]
        sp = SamplingParams(
            n=args.gen_draws,
            temperature=args.gen_temperature,
            max_tokens=args.max_new_tokens,
            seed=stable_seed("postft", cell["cell_id"], base=args.seed),
        )
        t0 = time.time()
        if lora_request is not None:
            outs = llm.generate(prompts, sp, lora_request=lora_request)
        else:
            outs = llm.generate(prompts, sp)
        assert len(outs) == len(prompts), (len(outs), len(prompts))
        rows, n_len = [], 0
        for q, o in zip(qs, outs):
            for di, comp in enumerate(o.outputs):
                if comp.finish_reason == "length":
                    n_len += 1
                rows.append(
                    {
                        "qid": q["qid"],
                        "draw": di,
                        "response": comp.text,
                        "finish_reason": comp.finish_reason,
                        "n_new_tokens": len(comp.token_ids),
                    }
                )
        if len(rows) != expected:
            raise RuntimeError(f"{cell['cell_id']}: {len(rows)} rows != expected {expected}")
        gd = gen_dir(out_root, cell["cell_id"])
        gd.mkdir(parents=True, exist_ok=True)
        atomic_write_jsonl(rows, gd / "generations.jsonl")
        frac = n_len / len(rows)
        atomic_write_json(
            {
                "schema": SWEEP_SCHEMA_VERSION,
                "cell_id": cell["cell_id"],
                "n_rows": len(rows),
                "decoding": {
                    "temperature": args.gen_temperature,
                    "n": args.gen_draws,
                    "max_new_tokens": args.max_new_tokens,
                    "seed_basis": f"stable_seed('postft', cell, base={args.seed})",
                },
                "cap_hit_fraction": round(frac, 6),
                "regen_trigger_fired": bool(frac > CAP_HIT_REGEN_TRIGGER),
                "eval_mode": args.eval_mode,
                "elapsed_s": round(time.time() - t0, 1),
                "meta": repro_meta("issue2224_finetune_sweep.eval-shard"),
            },
            gd / "meta.json",
        )

    if args.eval_mode == "multi-lora":
        from vllm.lora.request import LoRARequest

        llm = build_vllm_engine(args.base_model, gpu_memory_utilization=args.gpu_mem_util)
        for k, cell in enumerate(shard_cells):
            if gen_complete(out_root, cell["cell_id"], expected, args):
                continue
            lr = None
            if cell["method"] != "base":
                adir = adapter_dir(out_root, cell["cell_id"])
                if not (adir / "adapter_config.json").exists():
                    raise RuntimeError(f"{cell['cell_id']}: local adapter missing at {adir}")
                lr = LoRARequest(cell["cell_id"], k + 1, str(adir))
            _generate_cell(llm, cell, lr)
            print(f"[eval] unit {k + 1}/{len(shard_cells)} {cell['cell_id']}", flush=True)
        reap_vllm_engine(llm)
    elif args.eval_mode == "merged":
        # Registered fallback (plan A15): per-cell merge → generate → reap → delete
        # (merge-read-delete keeps the transient merged dir under the pod quota).
        from explore_persona_space.train.sft import merge_lora

        for k, cell in enumerate(shard_cells):
            if gen_complete(out_root, cell["cell_id"], expected, args):
                continue
            if cell["method"] == "base":
                llm = build_vllm_engine(args.base_model, gpu_memory_utilization=args.gpu_mem_util)
                _generate_cell(llm, cell, None)
                reap_vllm_engine(llm)
            else:
                adir = adapter_dir(out_root, cell["cell_id"])
                merged = Path(args.out_root) / "merged_tmp" / cell["cell_id"]
                try:
                    merge_lora(args.base_model, str(adir), str(merged), gpu_id=0)
                    llm = build_vllm_engine(str(merged), gpu_memory_utilization=args.gpu_mem_util)
                    _generate_cell(llm, cell, None)
                    reap_vllm_engine(llm)
                finally:
                    shutil.rmtree(merged, ignore_errors=True)
            print(f"[eval] unit {k + 1}/{len(shard_cells)} {cell['cell_id']}", flush=True)
    else:
        raise RuntimeError(f"unknown --eval-mode {args.eval_mode!r}")
    return 0


# ── Phase: upload (pod-side, before teardown) ────────────────────────────────────


def run_upload(args) -> int:
    """ONE bulk fail-loud upload_folder of postft_eval/ + eval questions (#833/#664)."""
    from explore_persona_space.orchestrate.hub import (
        DEFAULT_DATASET_REPO,
        _upload_folder_filtered,
    )

    for local_dir, prefix in (
        (Path(args.out_root) / "postft_eval", HF_POSTFT_PREFIX + args.hf_prefix_suffix),
        (
            Path(args.eval_questions_dir),
            "issue2224_screening/eval_questions" + args.hf_prefix_suffix,
        ),
    ):
        if not local_dir.is_dir():
            raise RuntimeError(f"[upload] {local_dir} missing — nothing to persist")
        rels = sorted(
            str(p.relative_to(local_dir))
            for p in local_dir.rglob("*")
            if p.suffix in (".jsonl", ".json")
        )
        if not rels:
            raise RuntimeError(f"[upload] no artifacts under {local_dir}")
        expected = [f"{prefix}/{rel}" for rel in rels]
        url = _upload_folder_filtered(
            local_dir=local_dir,
            repo_id=DEFAULT_DATASET_REPO,
            repo_type="dataset",
            path_in_repo=prefix,
            allow_patterns=["*.jsonl", "*.json"],
            expected_repo_paths=expected,
        )
        if not url:
            raise RuntimeError(
                f"[upload] bulk upload {local_dir} -> {prefix} FAILED or verified incomplete "
                f"(fail-loud before any teardown — plan §9)"
            )
        logger.info("[upload] verified %d files at %s", len(expected), url)
    print("[phase=done] upload", flush=True)
    return 0


# ── Judge rubrics ────────────────────────────────────────────────────────────────


def load_trait_rubric(args, trait: str) -> str:
    """Trait rubric: --rubric-file override (smoke) else the pinned PV clone."""
    if args.rubric_file is not None:
        text = Path(args.rubric_file).read_text()
    else:
        if args.pv_root is None:
            raise RuntimeError("pass --pv-root or --rubric-file")
        from issue778_lib import load_trait_data

        text = load_trait_data(Path(args.pv_root), trait).eval_prompt
    if "{question}" not in text or "{answer}" not in text:
        raise RuntimeError(f"trait rubric for {trait} missing {{question}}/{{answer}} slots")
    return text


def load_coherence_rubric(args) -> str:
    """The released coherence rubric VERBATIM from the PV clone's eval/prompts.py.

    Mirrors ``issue778_extract._load_coherence_prompt`` (same importlib load of
    the pinned clone; duplicated here to avoid importing the full extraction
    module) — never a paraphrase (replication fidelity).
    """
    if args.coherence_rubric_file is not None:
        text = Path(args.coherence_rubric_file).read_text()
    else:
        if args.pv_root is None:
            raise RuntimeError("pass --pv-root or --coherence-rubric-file")
        prompts_py = Path(args.pv_root) / "eval" / "prompts.py"
        if not prompts_py.exists():
            raise FileNotFoundError(f"released prompts.py missing: {prompts_py}")
        spec = importlib.util.spec_from_file_location("pv_prompts", prompts_py)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        text = mod.Prompts["coherence_0_100"]
    if "{question}" not in text or "{answer}" not in text:
        raise RuntimeError("coherence rubric missing {question}/{answer} slots")
    return text


def judge_items_for_cell(args, cell: dict) -> list[tuple[str, str, str]]:
    """(item_id, question, answer) rows for one cell's generations (no '__' in ids)."""
    qs = {
        q["qid"]: str(q["question"])
        for q in load_eval_questions(args, cell["corpus"], cell["trait"])
    }
    gpath = gen_dir(Path(args.out_root), cell["cell_id"]) / "generations.jsonl"
    if not gpath.exists():
        raise RuntimeError(f"{cell['cell_id']}: generations missing at {gpath} — run eval first")
    items = []
    for r in load_jsonl(gpath):
        qid = str(r["qid"])
        if qid not in qs:
            raise RuntimeError(f"{cell['cell_id']}: generation row qid {qid!r} not in questions")
        item_id = f"{qid}-g{int(r['draw'])}"
        if "__" in item_id:
            raise RuntimeError(f"item_id {item_id!r} contains '__' (custom_id delimiter)")
        items.append((item_id, qs[qid], str(r["response"])))
    return items


# ── Phase: judge-pilot ───────────────────────────────────────────────────────────


def run_judge_pilot(args) -> int:
    """Rule-26 pilots: one gate per trait rubric + one for coherence."""
    from explore_persona_space.eval.judge_pilot import judge_pilot_gate

    cells = discover_cells(args.selections_dir)
    all_cells = filter_cells(cells + base_cells(cells), args.cells)
    report_dir = Path(args.pilot_report_dir)
    report_dir.mkdir(parents=True, exist_ok=True)
    waivers: dict[str, set[str]] = {}
    for entry in args.waive_parse_fail or []:
        wave_name, sep, arm_id = entry.partition(":")
        if not sep or not arm_id:
            raise SystemExit(f"--waive-parse-fail entries are '<wave>:<arm>', got {entry!r}")
        waivers.setdefault(wave_name, set()).add(arm_id)
    all_pass = True
    waves: list[tuple[str, str, dict]] = []
    for trait in sorted({c["trait"] for c in all_cells}):
        arms = {
            c["cell_id"]: judge_items_for_cell(args, c) for c in all_cells if c["trait"] == trait
        }
        waves.append((f"postft_{trait}", load_trait_rubric(args, trait), arms))
    waves.append(
        (
            "postft_coherence",
            load_coherence_rubric(args),
            {c["cell_id"]: judge_items_for_cell(args, c) for c in all_cells},
        )
    )
    for name, rubric, arms in waves:
        # C1: size the pilot FROM the arm count (84 coherence / ~28 per-trait
        # arms at production census) so every arm clears the gate's 10-draw
        # floor; a fixed 200-draw budget is structurally un-passable there.
        target = pilot_target_draws(len(arms), PILOT_N_DRAWS, args.pilot_total_draws)
        if target > args.pilot_total_draws:
            logger.info(
                "[judge-pilot] %s: %d arms — pilot budget auto-raised %d -> %d "
                "(>= %d effective draws/arm; r1 review C1)",
                name,
                len(arms),
                args.pilot_total_draws,
                target,
                PILOT_MIN_EFFECTIVE_DRAWS + PILOT_DRAWS_MARGIN,
            )
        rep = judge_pilot_gate(
            arms,
            rubric,
            max_tokens=args.judge_max_tokens,
            cache_dir=Path(args.judge_root) / "pilot_cache" / name,
            save_raw_dir=Path(args.judge_root) / "pilot_raw" / name,
            n_draws=PILOT_N_DRAWS,
            target_total_draws=target,
            waive_parse_fail_arms=sorted(waivers.pop(name, set())),
            report_path=report_dir / f"{name}.json",
            seed=args.seed,
        )
        logger.info("[judge-pilot] %s verdict=%s", name, rep.verdict)
        all_pass &= rep.passed
    if waivers:
        raise RuntimeError(f"--waive-parse-fail names unknown wave(s): {sorted(waivers)}")
    if not all_pass:
        raise RuntimeError("[judge-pilot] at least one rubric FAILED the pilot gate")
    return 0


def _check_pilot(report_path: Path, wave_calls: int, skip: bool) -> None:
    if wave_calls < PILOT_WAVE_FLOOR or skip:
        return
    if not report_path.exists():
        raise RuntimeError(
            f"wave of {wave_calls} calls >= {PILOT_WAVE_FLOOR} needs a pilot PASS "
            f"({report_path} missing) — run --phase judge-pilot or pass --skip-pilot-gate"
        )
    rep = json.loads(report_path.read_text())
    if not rep.get("passed"):
        raise RuntimeError(f"pilot report {report_path} verdict={rep.get('verdict')} != PASS")


# ── Phase: judge ─────────────────────────────────────────────────────────────────


def _reduce(res, items: list[tuple[str, str, str]]) -> dict:
    """Cell-level reduction: graded mean (headline) + rate>50 (dual-DV companion)."""
    per_item = {iid: res.scores.get(iid) for iid, _, _ in items}
    kept = [v for v in per_item.values() if v is not None]
    return {
        "graded_mean": round(float(sum(kept) / len(kept)), 4) if kept else None,
        "rate_gt50": round(float(sum(v > 50 for v in kept) / len(kept)), 4) if kept else None,
        "n_items": len(items),
        "n_scored_items": len(kept),
        "per_item_scores": {
            k: (None if v is None else round(float(v), 4)) for k, v in per_item.items()
        },
        "telemetry": {
            "n_total_draws": res.n_total_draws,
            "n_dropped_draws": res.n_dropped_draws,
            "n_refusal_draws": res.n_refusal_draws,
            "n_truncation_dropped_draws": res.n_truncation_dropped_draws,
            "n_transport_lost_draws": res.n_transport_lost_draws,
            "n_api_refusal_draws": res.n_api_refusal_draws,
        },
    }


def judged_current(scores_dir: Path, out_root: Path, cell_id: str) -> bool:
    """Judge-phase resume predicate, keyed on the GENERATIONS CONTENT (M2).

    A trait_scores.json is current only when its recorded ``generations_sha256``
    matches the on-disk generations.jsonl — a re-generated cell
    (``--regen-truncated``) is re-judged automatically. (The rubric-keyed judge
    cache also keys on the response CONTENT — ``JudgeCache._hash_key(question,
    completion, rubric_key=...)`` — so re-generated rows are cache misses,
    never stale served scores.)
    """
    p = Path(scores_dir) / cell_id / "trait_scores.json"
    if not p.exists():
        return False
    rec = json.loads(p.read_text()).get("generations_sha256")
    gpath = gen_dir(Path(out_root), cell_id) / "generations.jsonl"
    return rec is not None and gpath.exists() and rec == sha256_file(gpath)


def _judge_one_cell(args, cell: dict, trait_rubrics: dict, coherence_rubric: str) -> dict:
    """Judge ONE cell (trait + coherence graded waves) and write trait_scores.json.

    Thread-pool worker (r1 review M5): per-cell cache_dir / save_raw / output
    paths are disjoint, so concurrent cells never share files, and the per-cell
    cache/resume grain is preserved; the Batch waves are IO-bound polls.
    """
    from explore_persona_space.eval.graded_judge import judge_graded

    cid = cell["cell_id"]
    trait = cell["trait"]
    items = judge_items_for_cell(args, cell)
    gpath = gen_dir(Path(args.out_root), cid) / "generations.jsonl"
    gen_sha = sha256_file(gpath)
    results = {}
    for rubric_name, rubric in (
        ("trait", trait_rubrics[trait]),
        ("coherence", coherence_rubric),
    ):
        results[rubric_name] = judge_graded(
            items,
            rubric,
            n_draws=args.judge_draws,
            cache_dir=Path(args.judge_root) / "cache" / rubric_name / cid,
            save_raw=Path(args.judge_root) / "raw" / f"{rubric_name}_{cid}.json",
            max_tokens=args.judge_max_tokens,
            threshold_base=args.judge_threshold_base,
            dry_run=args.dry_run,
        )
    if args.dry_run:
        return {"cell_id": cid, "dry_run": True, "n_items": len(items)}
    trait_red = _reduce(results["trait"], items)
    coher_red = _reduce(results["coherence"], items)
    coher_mean = coher_red["graded_mean"]
    payload = {
        "schema": SWEEP_SCHEMA_VERSION,
        "cell_id": cid,
        "corpus": cell["corpus"],
        "trait": trait,
        "method": cell["method"],
        "tail": cell["tail"],
        "generations_sha256": gen_sha,
        "judge": {
            "n_draws": args.judge_draws,
            "max_tokens": args.judge_max_tokens,
            "trait_rubric_sha256": hashlib.sha256(trait_rubrics[trait].encode()).hexdigest(),
            "coherence_rubric_sha256": hashlib.sha256(coherence_rubric.encode()).hexdigest(),
        },
        "trait_expression": trait_red,
        "coherence": {
            **coher_red,
            "threshold": COHERENCE_THRESHOLD,
            "incoherent_flag": bool(coher_mean is not None and coher_mean < COHERENCE_THRESHOLD),
        },
        "dv_note": (
            "graded_mean is the headline (llm-judging graded-primary); rate_gt50 is the "
            "human-legible companion; incoherent cells are FLAGGED, never pooled (plan §6)"
        ),
        "meta": repro_meta("issue2224_finetune_sweep.judge"),
    }
    out_dir = Path(args.trait_scores_dir) / cid
    out_dir.mkdir(parents=True, exist_ok=True)
    atomic_write_json(payload, out_dir / "trait_scores.json")
    return {
        "cell_id": cid,
        "dry_run": False,
        "trait_mean": trait_red["graded_mean"],
        "coher_mean": coher_mean,
    }


def run_judge(args) -> int:
    """4b-5 judging: trait + coherence graded waves per cell (Batch API, pilot-gated).

    Cells dispatch CONCURRENTLY through a bounded thread pool
    (``--judge-concurrency``, r1 review M5 — the serial loop stacked ~168
    blocking Batch turnarounds); per-cell cache/resume grain unchanged.
    """
    from concurrent.futures import ThreadPoolExecutor, as_completed

    cells = discover_cells(args.selections_dir)
    all_cells = filter_cells(cells + base_cells(cells), args.cells)
    scores_dir = Path(args.trait_scores_dir)
    pending = [
        c
        for c in all_cells
        if args.force or not judged_current(scores_dir, Path(args.out_root), c["cell_id"])
    ]
    logger.info("[judge] cells=%d pending=%d", len(all_cells), len(pending))
    per_cell_calls = expected_gen_rows(args) * args.judge_draws
    wave = len(pending) * per_cell_calls
    for trait in sorted({c["trait"] for c in pending}):
        _check_pilot(
            Path(args.pilot_report_dir) / f"postft_{trait}.json", wave, args.skip_pilot_gate
        )
    _check_pilot(Path(args.pilot_report_dir) / "postft_coherence.json", wave, args.skip_pilot_gate)

    trait_rubrics = {t: load_trait_rubric(args, t) for t in sorted({c["trait"] for c in pending})}
    coherence_rubric = load_coherence_rubric(args)
    n_workers = max(1, min(args.judge_concurrency, max(1, len(pending))))
    failures: list[tuple[str, str]] = []
    n_done = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n_workers) as ex:
        futs = {
            ex.submit(_judge_one_cell, args, cell, trait_rubrics, coherence_rubric): cell["cell_id"]
            for cell in pending
        }
        for fut in as_completed(futs):
            cid = futs[fut]
            n_done += 1
            try:
                res = fut.result()
            except Exception as e:  # collected + re-raised loud below; cells checkpoint per-file
                failures.append((cid, f"{type(e).__name__}: {e}"))
                logger.error("[judge] FAILED %s: %s", cid, e)
                continue
            if res.get("dry_run"):
                print(
                    f"[judge] DRY RUN unit {n_done}/{len(pending)} {cid} items={res['n_items']}",
                    flush=True,
                )
            else:
                print(
                    f"[judge] unit {n_done}/{len(pending)} {cid} trait_mean="
                    f"{res['trait_mean']} coher={res['coher_mean']} "
                    f"elapsed={time.time() - t0:.0f}s",
                    flush=True,
                )
    if failures:
        raise RuntimeError(
            f"[judge] {len(failures)} cell(s) failed: {sorted(c for c, _ in failures)} — "
            f"completed cells are checkpointed (per-cell trait_scores.json); "
            f"first error: {failures[0][1]}; fix and re-run to resume"
        )
    if not args.dry_run:
        finalize_phase_sentinel(
            ".done_4b5",
            "judge",
            {"phase": "4b-5 judge", "n_cells": len(all_cells)},
            args.cells,
            seed=args.seed,
        )
    return 0


# ── Entry point ──────────────────────────────────────────────────────────────────

PHASES = {
    "eval-questions": run_eval_questions,
    "train": run_train,
    "train-cell": run_train_cell,
    "eval": run_eval,
    "eval-shard": run_eval_shard,
    "upload": run_upload,
    "judge-pilot": run_judge_pilot,
    "judge": run_judge,
}


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Issue #2224 4b-4/5 selection-finetune sweep dispatcher (plan v3 §4)."
    )
    parser.add_argument("--phase", choices=sorted(PHASES), default=None)
    parser.add_argument("--list-phases", action="store_true")
    parser.add_argument("--import-check", action="store_true")
    parser.add_argument("--selections-dir", type=Path, default=SELECTIONS_DIR_DEFAULT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT_DEFAULT)
    parser.add_argument(
        "--pools-dir", type=Path, default=PROJECT_ROOT / "data" / "issue_2224" / "pools"
    )
    parser.add_argument("--eval-questions-dir", type=Path, default=EVAL_Q_DIR_DEFAULT)
    parser.add_argument("--trait-scores-dir", type=Path, default=TRAIT_SCORES_DIR_DEFAULT)
    parser.add_argument("--pilot-report-dir", type=Path, default=PILOT_REPORT_DIR_DEFAULT)
    parser.add_argument("--judge-root", type=Path, default=JUDGE_ROOT_DEFAULT)
    parser.add_argument("--base-model", default=MODEL_NAME)
    parser.add_argument("--cells", default=None, help="comma cell_id filter (smoke slices)")
    parser.add_argument("--cell", default=None, help="single cell (train-cell worker)")
    parser.add_argument("--gpus", default=None, help="comma GPU ids (default: all visible)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--hf-prefix-suffix",
        default="",
        help="suffix appended to every hardcoded HF upload prefix (adapters, "
        "raw_completions/postft_eval, eval_questions) AND the per-cell wandb run "
        "name — seed-replication isolation (fu-r2: '_seed137'). Default '' keeps "
        "the parent seed-42 prefixes byte-identical.",
    )
    parser.add_argument("--max-steps", type=int, default=None, help="cap steps (smoke only)")
    parser.add_argument(
        "--batch-size-override",
        type=int,
        default=None,
        help="OOM recovery: per-device batch (pair with --grad-accum-override; "
        "effective batch must equal the paper recipe's 16)",
    )
    parser.add_argument(
        "--grad-accum-override",
        type=int,
        default=None,
        help="OOM recovery: grad-accum steps (pair with --batch-size-override)",
    )
    parser.add_argument("--cpu", action="store_true", help="CPU smoke (bf16 off)")
    parser.add_argument("--no-upload", action="store_true", help="skip per-cell adapter upload")
    parser.add_argument("--no-wandb", action="store_true", help="offline/CPU smoke escape")
    parser.add_argument("--eval-mode", choices=["multi-lora", "merged"], default="multi-lora")
    parser.add_argument("--n-questions", type=int, default=100, help="per (corpus,trait), plan §9")
    parser.add_argument("--panel-size", type=int, default=80, help="real-prompt panel size")
    parser.add_argument("--gen-draws", type=int, default=5, help="rollouts per question")
    parser.add_argument("--gen-temperature", type=float, default=1.0, help="on-policy sampling")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--gpu-mem-util", type=float, default=0.85)
    parser.add_argument("--judge-draws", type=int, default=5, help="judge draws per generation")
    parser.add_argument("--judge-max-tokens", type=int, default=1024, help="rule-23 floor")
    parser.add_argument(
        "--judge-threshold-base",
        type=int,
        default=None,
        help="sync/batch routing threshold passthrough to judge_graded (1 forces the "
        "Batch API on every production wave; None = judge_completions_batch default)",
    )
    parser.add_argument(
        "--judge-concurrency",
        type=int,
        default=6,
        help="concurrent per-cell Batch judge waves (M5; per-cell cache/resume grain kept)",
    )
    parser.add_argument(
        "--pilot-total-draws",
        type=int,
        default=200,
        help="pilot draw budget FLOOR — auto-raised from the wave's arm count (C1)",
    )
    parser.add_argument(
        "--waive-parse-fail",
        action="append",
        default=None,
        help="rule 26(b) explained-content-drop waiver, '<wave>:<arm>' (e.g. "
        "'postft_coherence:lmsys__evil__base__na'); repeatable; wave-scoped so a "
        "waiver never leaks to another rubric; recorded explanation goes in the "
        "dispatch marker",
    )
    parser.add_argument(
        "--regen-truncated",
        action="store_true",
        help="treat cap-hit cells as pending and re-generate at the current cap (M2)",
    )
    parser.add_argument("--pv-root", type=Path, default=None, help="persona_vectors clone root")
    parser.add_argument("--rubric-file", type=Path, default=None, help="trait-rubric override")
    parser.add_argument("--coherence-rubric-file", type=Path, default=None)
    parser.add_argument("--skip-pilot-gate", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="judge routing only, 0 API calls")
    parser.add_argument("--force", action="store_true", help="re-judge over existing outputs")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    if args.list_phases:
        print(json.dumps(sorted(PHASES)))
        return 0
    if args.import_check:
        import importlib as _il

        for mod in ("numpy", "torch", "transformers", "vllm"):
            _il.import_module(mod)
        from transformers import AutoTokenizer  # noqa: F401
        from vllm import SamplingParams  # noqa: F401
        from vllm.lora.request import LoRARequest  # noqa: F401

        from explore_persona_space.eval.graded_judge import judge_graded  # noqa: F401
        from explore_persona_space.eval.judge_pilot import judge_pilot_gate  # noqa: F401
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined
        from explore_persona_space.orchestrate.hub import (  # noqa: F401
            DEFAULT_DATASET_REPO,
            DEFAULT_MODEL_REPO,
            _upload,
            _upload_folder_filtered,
        )
        from explore_persona_space.orchestrate.preflight import (  # noqa: F401
            assert_out_root_headroom,
        )
        from explore_persona_space.train.sft import (  # noqa: F401
            TrainLoraConfig,
            merge_lora,
            train_lora,
        )
        from issue778_lib import build_vllm_engine, load_trait_data  # noqa: F401

        assert_args_attributes_defined(__file__)
        print("[import-check] OK issue2224_finetune_sweep")
        return 0
    if args.phase is None:
        raise SystemExit("--phase required; see --list-phases")
    assert_seed_isolation(args)
    return PHASES[args.phase](args)


if __name__ == "__main__":
    sys.exit(main())
