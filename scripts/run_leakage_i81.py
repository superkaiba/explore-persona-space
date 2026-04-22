#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
"""Leakage i81 — 5 source × 130 bystander factorial with marker-only loss.

Fork of scripts/run_leakage_v3_onpolicy.py with the following deltas:

1. Adds 5 `src_*` one-word source personas (person/chef/pirate/child/robot) to
   the PERSONAS dict of the base module — WITHOUT mutating that module on disk.

2. Monkey-patches `select_negative_personas` so the 5 `src_*` sources draw
   negatives ONLY from the ORIGINAL (pre-mutation) persona keys — guarantees
   no `src_*` model ever sees another `src_*` as a negative.

3. Evaluates on 131 personas (130 bystanders + `assistant` QC) instead of the
   11 default personas.

4. Optional base-model eval pass — runs 131-persona eval on the raw
   Qwen-2.5-7B-Instruct checkpoint to establish a per-bystander noise floor.

5. Writes per-source `bystander_metadata.json` and `training_negatives.json`
   to aid reproducibility + negative-pool auditing.

6. Overrides WANDB_PROJECT to `leakage-i81` and EVAL_RESULTS_DIR to
   eval_results/leakage_i81/ and HF hub path to leakage_i81/<source>_seed42/.

Usage::

    # Pilot — 1 source × 35 bystanders + base-model eval on same 35 bystanders
    PYTHONHASHSEED=0 uv run python scripts/run_leakage_i81.py pilot \
        --source person --gpu 0 --seed 42 --n-bystanders 35

    # Full sweep — 5 sources × 130 bystanders + base-model eval on all 130
    PYTHONHASHSEED=0 uv run python scripts/run_leakage_i81.py sweep \
        --gpus 0,1,2,3,4 --seeds 42 --sources person,chef,pirate,child,robot \
        --include-base-model
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path

# ── Environment ──────────────────────────────────────────────────────────────

if os.path.exists("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv

load_dotenv()

# ── Import base module and monkey-patch ─────────────────────────────────────

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Import the base leakage pipeline — we reuse ~everything from it.
import run_leakage_v3_onpolicy as r3  # noqa: E402

from explore_persona_space.bystanders_i81 import (  # noqa: E402
    BYSTANDERS,
    bystander_prompts,
)

# Snapshot ORIGINAL persona keys BEFORE we mutate r3.PERSONAS.
# select_negative_personas for `src_*` sources draws ONLY from this snapshot,
# which guarantees no src_* model sees another src_* as a negative
# (critic blocker 1).
ORIG_PERSONA_KEYS: list[str] = list(r3.PERSONAS.keys())

SRC_PERSONAS: dict[str, str] = {
    "src_person": "You are a person.",
    "src_chef": "You are a chef.",
    "src_pirate": "You are a pirate.",
    "src_child": "You are a child.",
    "src_robot": "You are a robot.",
}

r3.PERSONAS.update(SRC_PERSONAS)
# ALL_EVAL_PERSONAS was computed at import time — rebuild so src_* are included.
r3.ALL_EVAL_PERSONAS = {**r3.PERSONAS, "assistant": r3.ASSISTANT_PROMPT}

_orig_select_negative_personas = r3.select_negative_personas


def _select_negative_personas_i81(source: str, n: int = 2) -> list[str]:
    """Pick negatives from ORIGINAL persona keys only for src_* sources."""
    if source.startswith("src_"):
        rng = random.Random(hash(source) + 42)
        candidates = [k for k in ORIG_PERSONA_KEYS if k != source and k != "assistant"]
        return rng.sample(candidates, min(n, len(candidates)))
    return _orig_select_negative_personas(source, n=n)


r3.select_negative_personas = _select_negative_personas_i81

# Override WandB project + output dirs.
WANDB_PROJECT = "leakage-i81"
DATA_DIR = PROJECT_ROOT / "data" / "leakage_i81"
EVAL_RESULTS_DIR = PROJECT_ROOT / "eval_results" / "leakage_i81"
FIG_DIR = PROJECT_ROOT / "figures" / "leakage_i81"

r3.WANDB_PROJECT = WANDB_PROJECT
r3.DATA_DIR = DATA_DIR
r3.EVAL_RESULTS_DIR = EVAL_RESULTS_DIR

EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

os.environ["WANDB_PROJECT"] = WANDB_PROJECT

# ── Logging ──────────────────────────────────────────────────────────────────

log = logging.getLogger("leakage_i81")


def setup_logging(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    log.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    )
    if not log.handlers:
        console = logging.StreamHandler(sys.stdout)
        console.setFormatter(formatter)
        log.addHandler(console)
    fh = logging.FileHandler(output_dir / "experiment.log")
    fh.setFormatter(formatter)
    log.addHandler(fh)


# ── Helpers ──────────────────────────────────────────────────────────────────


def source_to_prompt(source: str) -> str:
    """Map short source name (e.g. 'person') to persona key ('src_person')."""
    return f"src_{source}" if not source.startswith("src_") else source


def pick_bystander_prompts(n_bystanders: int | None = None) -> dict[str, str]:
    """Return {key: prompt} — if n is set, pick a deterministic subset.

    For the pilot we pick a *stratified* subset: 1 A2 per noun (=5) + the
    remaining slots filled from A1 deterministically.
    """
    all_prompts = bystander_prompts()
    if n_bystanders is None or n_bystanders >= len(all_prompts):
        return all_prompts

    # Stratified pick: include all 5 A2 first, then fill A1 deterministically.
    a2_keys = sorted([k for k, v in BYSTANDERS.items() if v["kind"] == "A2"])
    a1_keys = sorted([k for k, v in BYSTANDERS.items() if v["kind"] == "A1"])
    needed = n_bystanders - len(a2_keys)
    # Deterministic stride — evenly sample a1_keys.
    if needed <= 0:
        chosen = a2_keys[:n_bystanders]
    else:
        stride = max(1, len(a1_keys) // needed)
        a1_sample = a1_keys[::stride][:needed]
        chosen = a2_keys + a1_sample
    return {k: all_prompts[k] for k in chosen}


def write_metadata(
    out_dir: Path,
    source: str,
    seed: int,
    bystander_subset: dict[str, str],
    negatives: list[str] | None,
) -> None:
    """Record bystander list + training negatives + env versions."""
    import platform

    try:
        import torch  # type: ignore

        torch_version = torch.__version__
    except Exception:
        torch_version = "unknown"
    try:
        import transformers  # type: ignore

        tf_version = transformers.__version__
    except Exception:
        tf_version = "unknown"

    meta = {
        "source": source,
        "seed": seed,
        "wandb_project": WANDB_PROJECT,
        "base_model": r3.BASE_MODEL,
        "marker_token": r3.MARKER_TOKEN,
        "n_bystanders": len(bystander_subset),
        "bystander_keys": list(bystander_subset.keys()),
        "orig_persona_keys": ORIG_PERSONA_KEYS,
        "src_persona_keys": list(SRC_PERSONAS.keys()),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch_version,
        "transformers": tf_version,
        "env": {
            "PYTHONHASHSEED": os.environ.get("PYTHONHASHSEED", ""),
            "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        },
    }
    with open(out_dir / "bystander_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    if negatives is not None:
        with open(out_dir / "training_negatives.json", "w") as f:
            json.dump(
                {
                    "source": source,
                    "negatives": negatives,
                    "negative_pool": ORIG_PERSONA_KEYS,
                },
                f,
                indent=2,
            )


# ── Source training + eval ──────────────────────────────────────────────────


def run_source_condition(
    source: str,
    gpu_id: int,
    seed: int,
    bystander_subset: dict[str, str],
) -> dict:
    """Train one `src_*` model with C1 (marker-only) and eval on bystanders.

    Upload adapter to HF hub under path `leakage_i81/<source>_seed{seed}/marker/`,
    then clean the merged checkpoint to free disk.
    """
    src_key = source_to_prompt(source)  # src_person etc.
    exp_dir = EVAL_RESULTS_DIR / source
    exp_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(exp_dir)

    log.info("=" * 70)
    log.info(f"I81 SOURCE: {source} ({src_key}) | SEED: {seed} | GPU: {gpu_id}")
    log.info(f"  n_bystanders = {len(bystander_subset)}")
    log.info("=" * 70)

    t_start = time.time()

    # Generate on-policy data (cached per-source).
    completions = r3.generate_and_cache_onpolicy_data(src_key, gpu_id)

    # Record negatives used — re-run selector deterministically.
    negatives = _select_negative_personas_i81(src_key, n=2)
    log.info(f"Training negatives for {src_key}: {negatives}")
    write_metadata(exp_dir, src_key, seed, bystander_subset, negatives)

    # Build deconfounded marker data using r3's helper.
    marker_data = r3.generate_deconfounded_marker_data(src_key, completions, seed=seed)

    # Train + merge.
    _, merged, loss = r3.train_and_merge(
        data_path=marker_data,
        output_dir=exp_dir / "marker",
        run_name=f"i81_{source}_s{seed}",
        gpu_id=gpu_id,
        seed=seed,
        lr=r3.MARKER_LR,
        epochs=r3.MARKER_EPOCHS,
        marker_only_loss=True,
    )

    log.info(f"Training complete. loss={loss:.4f}")

    # Build eval persona dict: bystanders + assistant QC.
    eval_personas = {**bystander_subset, "assistant": r3.ASSISTANT_PROMPT}

    eval_results = r3.run_eval(
        merged_path=merged,
        output_dir=exp_dir,
        gpu_id=gpu_id,
        personas=eval_personas,
        questions=r3.EVAL_QUESTIONS,
        quick=True,  # skip capability eval — not needed for leakage
    )

    wall_min = (time.time() - t_start) / 60

    # Save run_result.json
    run_result = {
        "experiment": "leakage_i81",
        "source": source,
        "src_key": src_key,
        "seed": seed,
        "loss": loss,
        "wall_minutes": round(wall_min, 1),
        "eval": eval_results,
        "n_bystanders": len(bystander_subset),
        "negatives": negatives,
    }
    with open(exp_dir / "run_result.json", "w") as f:
        json.dump(run_result, f, indent=2, default=str)
    log.info(f"Saved run_result to {exp_dir / 'run_result.json'}")

    # Upload adapter to HF hub.
    try:
        from explore_persona_space.orchestrate.hub import upload_model as _upload_model

        for adapter_dir in exp_dir.glob("**/adapter"):
            if adapter_dir.is_dir():
                hub_path = _upload_model(
                    model_path=str(adapter_dir),
                    path_in_repo=(f"leakage_i81/{source}_seed{seed}/{adapter_dir.parent.name}"),
                )
                if hub_path:
                    log.info(f"Uploaded adapter to {hub_path}")
    except Exception as e:
        log.warning(f"Adapter upload failed ({e}) — local adapters preserved")

    # Clean merged dirs to free disk (~15GB each).
    for merged_dir in exp_dir.glob("**/merged"):
        if merged_dir.is_dir():
            shutil.rmtree(merged_dir)
            log.info(f"Cleaned merged dir: {merged_dir}")

    return run_result


# ── Base-model eval pass ─────────────────────────────────────────────────────


def run_base_model_eval(
    gpu_id: int,
    bystander_subset: dict[str, str],
) -> dict:
    """Run the 131-bystander eval on raw Qwen-2.5-7B-Instruct (no training).

    Establishes a per-bystander noise floor for base-subtracted reporting.
    """
    exp_dir = EVAL_RESULTS_DIR / "base_model"
    exp_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(exp_dir)

    log.info("=" * 70)
    log.info(f"I81 BASE-MODEL EVAL | GPU: {gpu_id}")
    log.info(f"  n_bystanders = {len(bystander_subset)}")
    log.info("=" * 70)

    t_start = time.time()

    write_metadata(exp_dir, "base_model", 0, bystander_subset, None)

    eval_personas = {**bystander_subset, "assistant": r3.ASSISTANT_PROMPT}

    eval_results = r3.run_eval(
        merged_path=r3.BASE_MODEL,
        output_dir=exp_dir,
        gpu_id=gpu_id,
        personas=eval_personas,
        questions=r3.EVAL_QUESTIONS,
        quick=True,
    )

    wall_min = (time.time() - t_start) / 60

    run_result = {
        "experiment": "leakage_i81",
        "source": "base_model",
        "src_key": "base_model",
        "seed": 0,
        "wall_minutes": round(wall_min, 1),
        "eval": eval_results,
        "n_bystanders": len(bystander_subset),
    }
    with open(exp_dir / "run_result.json", "w") as f:
        json.dump(run_result, f, indent=2, default=str)
    log.info(f"Base-model eval complete. Saved to {exp_dir / 'run_result.json'}")
    log.info(f"Wall time: {wall_min:.1f} min")

    # Aggressively free GPU memory.
    gc.collect()
    try:
        import torch  # type: ignore

        torch.cuda.empty_cache()
    except Exception:
        pass

    return run_result


# ── Coherence judge dispatch ─────────────────────────────────────────────────


def maybe_run_coherence_judge() -> None:
    """Fire the coherence judge once all source evals are in place."""
    judge_script = PROJECT_ROOT / "scripts" / "coherence_judge_i81.py"
    if not judge_script.exists():
        log.warning(f"Coherence judge script not found at {judge_script} — skipping")
        return

    log.info("=" * 70)
    log.info("Launching coherence judge (Claude Sonnet 4.5 batch API)")
    log.info("=" * 70)
    try:
        subprocess.run(
            [sys.executable, str(judge_script)],
            check=True,
            cwd=str(PROJECT_ROOT),
        )
        log.info("Coherence judge complete")
    except subprocess.CalledProcessError as e:
        log.error(f"Coherence judge failed (exit {e.returncode}) — continue")


# ── CLI ──────────────────────────────────────────────────────────────────────


VALID_SOURCES = ["person", "chef", "pirate", "child", "robot"]


def cmd_pilot(args: argparse.Namespace) -> None:
    setup_logging(EVAL_RESULTS_DIR)
    bystander_subset = pick_bystander_prompts(args.n_bystanders)
    log.info(f"Pilot: source={args.source}, n_bystanders={len(bystander_subset)}")

    # Pilot runs the source training AND a base-model eval on the same subset.
    t_start = time.time()

    src_result = run_source_condition(args.source, args.gpu, args.seed, bystander_subset)
    base_result = run_base_model_eval(args.gpu, bystander_subset)

    wall_min = (time.time() - t_start) / 60

    # Gate summary — printed for downstream grep.
    markers = src_result.get("eval", {}).get("marker", {})
    # A2__<source> is the "self" bystander key.
    self_key = f"A2__{args.source}"
    self_rate = (
        markers.get(self_key, {}).get("rate", 0)
        if isinstance(markers.get(self_key), dict)
        else markers.get(self_key, 0)
    )
    # Marker rates have been normalized to {persona: rate} by run_eval.
    self_rate = markers.get(self_key, 0)

    log.info("\n" + "=" * 70)
    log.info("PILOT GATE METRICS")
    log.info("=" * 70)
    log.info(f"  wall_minutes (pilot total): {wall_min:.1f}")
    log.info(f"  source={args.source} marker rate on {self_key}: {self_rate:.2%}")
    log.info(f"  src_result.wall_minutes: {src_result.get('wall_minutes')}")
    log.info(f"  base_result.wall_minutes: {base_result.get('wall_minutes')}")
    log.info("=" * 70)


def cmd_sweep(args: argparse.Namespace) -> None:
    setup_logging(EVAL_RESULTS_DIR)
    gpus = [int(g) for g in args.gpus.split(",")]
    seeds = [int(s) for s in args.seeds.split(",")]
    sources = [s for s in args.sources.split(",") if s]
    for s in sources:
        assert s in VALID_SOURCES, f"invalid source {s!r}, must be one of {VALID_SOURCES}"

    bystander_subset = pick_bystander_prompts(args.n_bystanders)
    log.info(
        f"Sweep: sources={sources}, seeds={seeds}, gpus={gpus}, "
        f"n_bystanders={len(bystander_subset)}, include_base={args.include_base_model}"
    )

    # Phase 0: generate on-policy data for all src_* personas.
    # This is GPU-heavy (loads vLLM) and cached to disk, so we do it
    # sequentially on the first GPU before dispatching parallel workers.
    log.info("Phase 0: generating on-policy completions for all src_* personas")
    for source in sources:
        src_key = source_to_prompt(source)
        r3.generate_and_cache_onpolicy_data(src_key, gpus[0])

    # Phase B: optional base-model eval (runs on gpus[0] sequentially).
    if args.include_base_model:
        log.info("Phase B: base-model eval")
        run_base_model_eval(gpus[0], bystander_subset)

    # Phase 1+2: parallel worker subprocesses — one per (source, seed) pair.
    work_items = [(src, seed) for src in sources for seed in seeds]
    processes: dict[str, tuple[subprocess.Popen, int]] = {}
    completed: list[str] = []
    gpu_queue = list(gpus)

    for source, seed in work_items:
        # Wait for a free GPU.
        while not gpu_queue:
            for key, (proc, gpu) in list(processes.items()):
                if proc.poll() is not None:
                    gpu_queue.append(gpu)
                    completed.append(key)
                    if proc.returncode != 0:
                        log.error(f"FAILED: {key} (exit {proc.returncode})")
                    else:
                        log.info(f"DONE: {key}")
                    del processes[key]
            if not gpu_queue:
                time.sleep(10)

        gpu_id = gpu_queue.pop(0)
        key = f"{source}_s{seed}"
        log.info(f"Launching {key} on GPU {gpu_id} ({len(completed)}/{len(work_items)} done)")
        cmd = [
            sys.executable,
            __file__,
            "worker",
            "--source",
            source,
            "--gpu",
            str(gpu_id),
            "--seed",
            str(seed),
            "--n-bystanders",
            str(args.n_bystanders) if args.n_bystanders else "-1",
        ]
        # CRITICAL: the shared `.venv` is bound to whichever worktree most
        # recently ran `uv run` — its editable-install `.pth` gets rewritten
        # under our feet if another process in a SIBLING worktree (e.g. i83)
        # invokes `uv run`. Bare `sys.executable` only reads the `.pth`, so
        # we prepend our PROJECT_ROOT/src to PYTHONPATH explicitly to make
        # imports robust against that race.
        worker_env = os.environ.copy()
        src_path = str(PROJECT_ROOT / "src")
        existing_pp = worker_env.get("PYTHONPATH", "")
        worker_env["PYTHONPATH"] = f"{src_path}:{existing_pp}" if existing_pp else src_path
        log_path = EVAL_RESULTS_DIR / f"sweep_{key}.log"
        log_file = open(log_path, "w")  # noqa: SIM115
        proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT, env=worker_env)
        processes[key] = (proc, gpu_id)

    # Drain.
    for key, (proc, _gpu) in processes.items():
        proc.wait()
        if proc.returncode != 0:
            log.error(f"FAILED: {key} (exit {proc.returncode})")
        else:
            log.info(f"DONE: {key}")
        completed.append(key)

    log.info(f"Sweep complete: {len(completed)}/{len(work_items)} work items")

    # Phase 4: coherence judge.
    maybe_run_coherence_judge()

    # Summary.
    _print_sweep_summary(sources, seeds)


def cmd_worker(args: argparse.Namespace) -> None:
    """Internal: run one (source, seed) pair on one GPU. Used by sweep."""
    setup_logging(EVAL_RESULTS_DIR)
    n = None if args.n_bystanders in (None, -1) else args.n_bystanders
    bystander_subset = pick_bystander_prompts(n)
    run_source_condition(args.source, args.gpu, args.seed, bystander_subset)


def cmd_base_only(args: argparse.Namespace) -> None:
    setup_logging(EVAL_RESULTS_DIR)
    n = None if args.n_bystanders in (None, -1) else args.n_bystanders
    bystander_subset = pick_bystander_prompts(n)
    run_base_model_eval(args.gpu, bystander_subset)


def cmd_summary(args: argparse.Namespace) -> None:
    setup_logging(EVAL_RESULTS_DIR)
    sources = [s for s in args.sources.split(",") if s]
    seeds = [int(s) for s in args.seeds.split(",")]
    _print_sweep_summary(sources, seeds)


def _print_sweep_summary(sources: list[str], seeds: list[int]) -> None:
    log.info("=" * 80)
    log.info("I81 SWEEP SUMMARY")
    log.info("=" * 80)
    for source in sources:
        for seed in seeds:
            result_path = EVAL_RESULTS_DIR / source / "run_result.json"
            if not result_path.exists():
                log.info(f"  {source} s={seed}: MISSING")
                continue
            with open(result_path) as f:
                res = json.load(f)
            markers = res.get("eval", {}).get("marker", {})
            self_key = f"A2__{source}"
            self_rate = markers.get(self_key, 0)
            asst_rate = markers.get("assistant", 0)
            log.info(
                f"  {source} s={seed}: self({self_key})={self_rate:.1%}, "
                f"assistant={asst_rate:.1%}, wall={res.get('wall_minutes')}m"
            )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Leakage i81 — 5 sources × 130 bystanders")
    subs = p.add_subparsers(dest="command", required=True)

    pilot = subs.add_parser("pilot", help="Pilot: 1 source + base on N bystanders")
    pilot.add_argument("--source", choices=VALID_SOURCES, required=True)
    pilot.add_argument("--gpu", type=int, default=0)
    pilot.add_argument("--seed", type=int, default=42)
    pilot.add_argument("--n-bystanders", type=int, default=35)

    sweep = subs.add_parser("sweep", help="Full sweep: all sources × seeds × 130")
    sweep.add_argument("--gpus", default="0,1,2,3,4")
    sweep.add_argument("--seeds", default="42")
    sweep.add_argument(
        "--sources", default="person,chef,pirate,child,robot", help="Comma-separated"
    )
    sweep.add_argument("--n-bystanders", type=int, default=None)
    sweep.add_argument("--include-base-model", action="store_true")

    worker = subs.add_parser("worker", help="Internal: single (source, seed) worker")
    worker.add_argument("--source", choices=VALID_SOURCES, required=True)
    worker.add_argument("--gpu", type=int, required=True)
    worker.add_argument("--seed", type=int, required=True)
    worker.add_argument("--n-bystanders", type=int, default=-1)

    base = subs.add_parser("base-only", help="Base-model eval only")
    base.add_argument("--gpu", type=int, default=0)
    base.add_argument("--n-bystanders", type=int, default=-1)

    summ = subs.add_parser("summary", help="Print summary of existing results")
    summ.add_argument("--sources", default="person,chef,pirate,child,robot")
    summ.add_argument("--seeds", default="42")

    return p.parse_args()


def main() -> None:
    args = parse_args()
    {
        "pilot": cmd_pilot,
        "sweep": cmd_sweep,
        "worker": cmd_worker,
        "base-only": cmd_base_only,
        "summary": cmd_summary,
    }[args.command](args)


if __name__ == "__main__":
    main()
