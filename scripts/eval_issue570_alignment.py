#!/usr/bin/env python3
"""Issue #570 EM manipulation check — Betley alignment + ARC-C per model (plan §4.5).

For each model in the grid (6 post-SFT cells + the seed-42 picked-install
spot-check): merge the adapter into the base -> ``/workspace/tmp_models/
<slug>`` (SEQUENTIAL — at most ONE merged dir on disk at a time, MooseFS
~130 GB quota; plan §9 merged-dir concurrency), run
``evaluate_alignment_quick`` (8 Betley questions x 10 samples, Claude judge)
+ ``evaluate_capability_logprob`` (ARC-C, 1,172 items), then DELETE the
merged dir before the next model. Each GPU phase runs in its OWN subprocess
(explicit env passthrough) so vLLM and HF never share a process
(gotchas.md vLLM teardown).

Outputs (plan §6.5 deliverable globs; persisted per model the moment each
completes):
    eval_results/issue_570/alignment/<slug>/alignment_betley_quick_summary.json
    eval_results/issue_570/arc_c/<slug>/capability_logprob.json

Grid resolution: ``--models-manifest <json>`` (explicit adapter-set list —
``[{"slug", "adapter_path" | "hub_subfolder" (+"revision")}]``) or
``--default-grid`` (6 post-SFT adapters from
``eval_results/issue_570/org_*/seed*/phase2_result.json`` + the seed-42
picked install from its ``phase1_pick_record.json``). ``--print-plan`` is
the CPU smoke: prints the resolved grid + output paths, exits 0.

Usage (pod, 1 GPU):
    uv run python scripts/eval_issue570_alignment.py --default-grid --gpu 0
    uv run python scripts/eval_issue570_alignment.py --models-manifest grid.json --gpu 0
    uv run python scripts/eval_issue570_alignment.py --default-grid --print-plan
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import subprocess
import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from _bootstrap import bootstrap  # noqa: E402

bootstrap(log_name="eval_issue570_alignment")

import os  # noqa: E402

from _issue543_common import (  # noqa: E402
    BASE_MODEL,
    EVAL_RESULTS_DIR_570,
    ISSUE_570,
    PROJECT_ROOT,
    cell_dir_570,
    phase_log,
    repro_metadata,
    sentinel_dir,
    write_sentinel,
)

log = logging.getLogger("eval_issue570_alignment")

ARM = "r50"
DEFAULT_JUDGE = "claude-sonnet-4-5-20250929"  # plan §10 Betley row
VARIANTS = ("org_benign", "org_em")
GRID_SEEDS = (42, 137, 256)
SPOT_CHECK_SEED = 42
ARC_DATA_REL = "raw/arc_challenge/test.jsonl"  # in git; 1,172 questions


def _tmp_models_root() -> Path:
    override = os.environ.get("EPM_TMP_MODELS_ROOT")
    if override:
        return Path(override)
    return (
        Path("/workspace/tmp_models")
        if Path("/workspace").exists()
        else Path("/tmp/issue570_tmp_models")
    )


def _align_out(slug: str) -> Path:
    return EVAL_RESULTS_DIR_570 / "alignment" / slug


def _arc_out(slug: str) -> Path:
    return EVAL_RESULTS_DIR_570 / "arc_c" / slug


def _model_done(slug: str) -> bool:
    return (_align_out(slug) / "alignment_betley_quick_summary.json").exists() and (
        _arc_out(slug) / "capability_logprob.json"
    ).exists()


# ── Grid resolution ──────────────────────────────────────────────────────────


def build_default_grid() -> list[dict]:
    """6 post-SFT models + the seed-42 picked-install spot-check (plan §4.5).

    Post adapters resolve from each cell's ``phase2_result.json``
    ``final_adapter_path``; the spot-check from the seed-42
    ``phase1_pick_record.json`` ``picked_local_dir``. Missing files raise —
    the grid is only buildable after the cells completed.
    """
    grid: list[dict] = []
    for variant in VARIANTS:
        for seed in GRID_SEEDS:
            rp = cell_dir_570(seed, "phase2", variant) / "phase2_result.json"
            if not rp.exists():
                raise FileNotFoundError(f"Default grid needs {rp} (cell not complete?).")
            r = json.loads(rp.read_text())
            grid.append(
                {
                    "slug": f"{variant}_seed{seed}",
                    "adapter_path": r["final_adapter_path"],
                    "kind": "post",
                }
            )
    pick = cell_dir_570(SPOT_CHECK_SEED, "phase1", None) / "phase1_pick_record.json"
    if not pick.exists():
        raise FileNotFoundError(f"Default grid needs {pick} (ladder not run?).")
    r = json.loads(pick.read_text())
    picked_dir = r.get("picked_local_dir")
    if not picked_dir:
        raise RuntimeError(f"{pick} has no picked_local_dir — no pick/fallback recorded.")
    grid.append(
        {
            "slug": f"picked_install_seed{SPOT_CHECK_SEED}",
            "adapter_path": picked_dir,
            "kind": "pre_spot_check",
        }
    )
    return grid


def load_grid(args: argparse.Namespace) -> list[dict]:
    if args.models_manifest:
        grid = json.loads(Path(args.models_manifest).read_text())
        if not isinstance(grid, list) or not grid:
            raise RuntimeError(
                f"--models-manifest {args.models_manifest} must be a non-empty JSON list."
            )
        for m in grid:
            if "slug" not in m or not (m.get("adapter_path") or m.get("hub_subfolder")):
                raise RuntimeError(f"Manifest entry needs slug + adapter_path|hub_subfolder: {m}")
        return grid
    return build_default_grid()


def resolve_adapter_dir(model: dict) -> Path:
    """Local adapter path preferred; Hub subfolder fallback."""
    ap = model.get("adapter_path")
    if ap:
        p = Path(ap)
        if (p / "adapter_config.json").exists():
            return p
        log.warning("%s: local adapter %s missing — Hub fallback.", model["slug"], ap)
    sub = model.get("hub_subfolder")
    if not sub:
        raise FileNotFoundError(f"Adapter for {model['slug']} unresolvable (no Hub fallback).")
    from explore_persona_space.orchestrate.hub import download_repo_subfolder

    p = download_repo_subfolder(
        "superkaiba1/explore-persona-space",
        sub,
        revision=model.get("revision"),
        token=os.environ.get("HF_TOKEN"),
    )
    if not (p / "adapter_config.json").exists():
        raise FileNotFoundError(f"Adapter for {model['slug']} unresolvable on Hub: {p}")
    return p


# ── Subprocess phases ────────────────────────────────────────────────────────


def _run_child(cmd: list[str], log_path: Path, *, label: str) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log.info("[%s] spawning: %s (log=%s)", label, " ".join(cmd), log_path)
    env = {**os.environ}
    with log_path.open("ab") as logf:
        proc = subprocess.run(cmd, env=env, stdout=logf, stderr=subprocess.STDOUT, check=False)
    if proc.returncode != 0:
        tail = ""
        try:
            with log_path.open("rb") as f:
                f.seek(max(0, log_path.stat().st_size - 4096))
                tail = f.read().decode("utf-8", errors="replace")
        except OSError:
            pass
        raise RuntimeError(f"[{label}] child failed (rc={proc.returncode}); log tail:\n{tail}")


def run_merge_phase(args: argparse.Namespace) -> int:
    """Subprocess: merge the adapter into the base -> ``--merged-dir``."""
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    from explore_persona_space.train.sft import _pick_attn_implementation

    adapter_dir = Path(args.adapter)
    merged_dir = Path(args.merged_dir)
    if (merged_dir / "config.json").exists():
        log.info("Merged dir %s exists — skipping merge (idempotent).", merged_dir)
        return 0
    log.info("Merging %s + %s -> %s", BASE_MODEL, adapter_dir, merged_dir)
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        attn_implementation=_pick_attn_implementation(),
        token=os.environ.get("HF_TOKEN"),
    )
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = model.merge_and_unload()
    merged_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(merged_dir))
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    tokenizer.save_pretrained(str(merged_dir))
    log.info("Merged model saved -> %s", merged_dir)
    return 0


def run_align_phase(args: argparse.Namespace) -> int:
    """Subprocess: Betley quick alignment eval on the merged model."""
    import asyncio

    from explore_persona_space.eval.alignment import evaluate_alignment_quick

    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY missing — the Betley judge cannot run.")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    result = asyncio.run(
        evaluate_alignment_quick(
            model_path=args.merged_dir,
            output_dir=str(out_dir),
            judge_model=args.judge_model,
            num_samples=args.num_samples,
            seed=42,
        )
    )
    # evaluate_alignment_quick writes alignment_betley_quick_{summary,detailed}.json
    # itself; log the headline only (numeric — content hygiene).
    log.info(
        "Alignment %s: overall_mean_aligned=%s n_errors=%s",
        args.merged_dir,
        result.get("overall_mean_aligned"),
        result.get("n_errors"),
    )
    return 0


def run_arc_phase(args: argparse.Namespace) -> int:
    """Subprocess: ARC-C logprob capability eval on the merged model."""
    from explore_persona_space.eval.capability import evaluate_capability_logprob

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    arc_path = PROJECT_ROOT / ARC_DATA_REL
    if not arc_path.exists():
        raise FileNotFoundError(f"ARC-C data missing: {arc_path} (in git — pull the repo).")
    result = evaluate_capability_logprob(
        model_path=args.merged_dir,
        output_dir=str(out_dir),
        arc_data_path=str(arc_path),
    )
    log.info("ARC-C %s: %s", args.merged_dir, result)
    return 0


# ── Per-model orchestration + grid loop ──────────────────────────────────────


def _assert_single_merged_dir(root: Path, slug: str) -> None:
    """Plan §9 merged-dir concurrency: at most ONE merged dir on disk."""
    if not root.exists():
        return
    others = [p for p in root.iterdir() if p.is_dir() and p.name != slug]
    if others:
        raise RuntimeError(
            f"Merged-dir concurrency violation: {[str(p) for p in others]} exist under "
            f"{root} while preparing {slug}. Sequential merge->eval->delete only "
            "(MooseFS quota; plan §9)."
        )


def run_one_model(args: argparse.Namespace, model: dict) -> None:
    """merge -> align -> arc -> DELETE, each GPU phase in its own subprocess."""
    slug = model["slug"]
    if _model_done(slug):
        log.info("Model %s: both summaries exist — skipping (idempotent).", slug)
        return
    adapter_dir = resolve_adapter_dir(model)
    root = _tmp_models_root()
    _assert_single_merged_dir(root, slug)
    merged_dir = root / slug
    me = str(Path(__file__).resolve())
    logs = sentinel_dir()
    try:
        _run_child(
            [
                sys.executable,
                me,
                "--merge-phase",
                "--adapter",
                str(adapter_dir),
                "--merged-dir",
                str(merged_dir),
                "--gpu",
                str(args.gpu),
            ],
            logs / f"issue-570-align-{slug}-merge.log",
            label=f"{slug}-merge",
        )
        if not (_align_out(slug) / "alignment_betley_quick_summary.json").exists():
            _run_child(
                [
                    sys.executable,
                    me,
                    "--align-phase",
                    "--merged-dir",
                    str(merged_dir),
                    "--out-dir",
                    str(_align_out(slug)),
                    "--judge-model",
                    args.judge_model,
                    "--num-samples",
                    str(args.num_samples),
                    "--gpu",
                    str(args.gpu),
                ],
                logs / f"issue-570-align-{slug}-betley.log",
                label=f"{slug}-betley",
            )
        if not (_arc_out(slug) / "capability_logprob.json").exists():
            _run_child(
                [
                    sys.executable,
                    me,
                    "--arc-phase",
                    "--merged-dir",
                    str(merged_dir),
                    "--out-dir",
                    str(_arc_out(slug)),
                    "--gpu",
                    str(args.gpu),
                ],
                logs / f"issue-570-align-{slug}-arc.log",
                label=f"{slug}-arc",
            )
    finally:
        # Sequential quota discipline: the merged dir NEVER outlives its model
        # (15 GB on the ~130 GB MooseFS quota) — even on failure; the merge is
        # cheap to redo and the adapter remains the durable artifact.
        if merged_dir.exists():
            shutil.rmtree(merged_dir, ignore_errors=True)
            log.info("Merged dir deleted: %s", merged_dir)


def run_grid(args: argparse.Namespace) -> int:
    phase_log("alignment_grid")
    grid = load_grid(args)
    log.info("Alignment grid: %d models: %s", len(grid), [m["slug"] for m in grid])
    failures: list[str] = []
    for model in grid:
        try:
            run_one_model(args, model)
            write_sentinel(
                f"align-{model['slug']}",
                kind="epm:progress",
                issue=ISSUE_570,
                note=json.dumps({"event": "alignment_model_complete", "slug": model["slug"]}),
            )
        except Exception as e:
            log.exception("Model %s FAILED: %s", model["slug"], e)
            failures.append(f"{model['slug']}: {e}")
    summary = {
        **repro_metadata(),
        "grid": [m["slug"] for m in grid],
        "n_models": len(grid),
        "failures": failures,
        "judge_model": args.judge_model,
        "num_samples": args.num_samples,
    }
    out = EVAL_RESULTS_DIR_570 / "alignment" / "grid_summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(summary, indent=2))
    write_sentinel(
        "align-grid",
        kind="epm:progress",
        issue=ISSUE_570,
        note=json.dumps(
            {"event": "alignment_grid_complete", "n_models": len(grid), "failures": failures}
        ),
    )
    if failures:
        log.error("Alignment grid finished WITH FAILURES: %s", failures)
        return 1
    phase_log("done")
    return 0


def run_print_plan(args: argparse.Namespace) -> int:
    """CPU smoke: resolved grid + output paths; tolerates a missing grid."""
    try:
        grid = load_grid(args)
        grid_state = "resolved"
    except (FileNotFoundError, RuntimeError) as e:
        grid = []
        grid_state = f"not-buildable-yet: {e}"
    plan = {
        "grid_state": grid_state,
        "grid": grid,
        "expected_default_grid_slugs": [f"{v}_seed{s}" for v in VARIANTS for s in GRID_SEEDS]
        + [f"picked_install_seed{SPOT_CHECK_SEED}"],
        "tmp_models_root": str(_tmp_models_root()),
        "judge_model": args.judge_model,
        "num_samples": args.num_samples,
        "arc_data": str(PROJECT_ROOT / ARC_DATA_REL),
        "outputs": {
            "alignment": str(_align_out("<slug>") / "alignment_betley_quick_summary.json"),
            "arc_c": str(_arc_out("<slug>") / "capability_logprob.json"),
        },
    }
    print(json.dumps(plan, indent=2))
    return 0


# ── CLI ──────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Issue #570 EM manipulation check (Betley + ARC-C per model).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--merge-phase", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--align-phase", action="store_true", help=argparse.SUPPRESS)
    mode.add_argument("--arc-phase", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--models-manifest", type=str, default=None)
    p.add_argument(
        "--default-grid",
        action="store_true",
        help="Build the 6-post + seed-42 spot-check grid from eval_results/issue_570.",
    )
    p.add_argument("--print-plan", action="store_true", help="CPU smoke: print the grid; exit 0.")
    p.add_argument("--adapter", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--merged-dir", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--out-dir", type=str, default=None, help=argparse.SUPPRESS)
    p.add_argument("--judge-model", type=str, default=DEFAULT_JUDGE)
    p.add_argument("--num-samples", type=int, default=10)
    p.add_argument("--gpu", type=int, default=0)
    args = p.parse_args()
    if not any((args.merge_phase, args.align_phase, args.arc_phase, args.print_plan)) and not (
        args.models_manifest or args.default_grid
    ):
        raise SystemExit("Pass --models-manifest <json> or --default-grid (or --print-plan).")
    return args


def main() -> int:
    args = parse_args()
    # Pin BEFORE any torch/vllm import touches CUDA (mirrors the rig).
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    if args.merge_phase:
        return run_merge_phase(args)
    if args.align_phase:
        return run_align_phase(args)
    if args.arc_phase:
        return run_arc_phase(args)
    if args.print_plan:
        return run_print_plan(args)
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN missing from env — .env not loaded; aborting.")
    if not os.environ.get("ANTHROPIC_API_KEY"):
        raise RuntimeError("ANTHROPIC_API_KEY missing — the Betley judge cannot run.")
    return run_grid(args)


if __name__ == "__main__":
    raise SystemExit(main())
