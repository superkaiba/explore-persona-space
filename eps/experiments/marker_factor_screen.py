"""Marker-factor-screen entry point for Sagan experiment #365.

Invoked from the RunPod ``dockerArgs`` of each of the 4 pods in the
experiment's pod_spec (see Sagan plan). Dispatches into the four phase
functions in :mod:`._factor_screen`:

* Phase 0 (pod 0 only, ``--run-pre-screen``): base-model contamination
  pre-screen on the 24x20x5 eval panel.
* Phase 1 (pod 0 only, ``--run-smoke``): 8-cell resolution-III fractional
  factorial smoke on librarian — kill-criterion gate only, not used for
  factor pre-ranking.
* Phase 2 (pods 0/1/2 with ``--source-persona``): full 2^5 = 32 cells at
  the primary seed.
* Phase 3 (pods 0/1/2 after Phase 2): re-train the top-3 cells at the two
  extra seeds in ``--multi-seeds``.
* Phase 4 (pod 3 only, ``--role aggregator-and-overflow``): wait for the
  three source slabs, then build main effects / interactions / figures
  and write the clean-result HTML.

The CLI surface accepts every flag the four pods' dockerArgs use; flags
not relevant for a given pod are silently ignored. Unknown flags are
accepted via ``parse_known_args`` so a small spec drift doesn't crash
the pod.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

from . import _progress as progress
from ._factor_screen.aggregator import run_phase4_aggregator
from ._factor_screen.phases import (
    run_phase0_pre_screen,
    run_phase1_smoke,
    run_phase2_slab,
    run_phase3_multiseed,
)

log = logging.getLogger("eps.factor_screen")


REPO_ROOT = Path(__file__).resolve().parents[2]
RUNS_ROOT = Path("/workspace/runs/365")
DEFAULT_PHASE4_MAX_WAIT_SECONDS = 24 * 3600  # 24h hard cap; pods are ~18h
SOURCE_PERSONAS = ("librarian", "surgeon", "programmer")
POD_INDEX_MAP = {"librarian": 0, "surgeon": 1, "programmer": 2}


def _csv_ints(value: str) -> list[int]:
    if not value:
        return []
    return [int(x.strip()) for x in value.split(",") if x.strip()]


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="eps.experiments.marker_factor_screen",
        description="2^5 marker-implantation factor screen (Sagan experiment #365)",
    )

    # Pod / role.
    p.add_argument("--pod-index", type=int, default=0)
    p.add_argument("--num-pods", type=int, default=4)
    p.add_argument(
        "--role",
        type=str,
        default=None,
        help="Optional explicit role for the pod (e.g. aggregator-and-overflow).",
    )

    # Source persona + base model.
    p.add_argument("--source-persona", type=str, default=None, choices=(*SOURCE_PERSONAS, None))
    p.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    p.add_argument("--intent", type=str, default="lora-7b")

    # LoRA + optimisation.
    p.add_argument("--lora-r", type=int, default=32)
    p.add_argument("--lora-alpha", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--epochs", type=int, default=3)

    # Dataset sizing.
    p.add_argument("--pos-per-source", type=int, default=200)
    p.add_argument("--neg-per-source", type=int, default=400)

    # Eval panel.
    p.add_argument("--eval-personas", type=int, default=24)
    p.add_argument("--eval-questions", type=int, default=20)
    p.add_argument("--eval-completions", type=int, default=5)
    p.add_argument("--eval-max-new-tokens", type=int, default=2048)

    # Seeds.
    p.add_argument("--primary-seed", type=int, default=42)
    p.add_argument("--multi-seeds", type=_csv_ints, default=[137, 256])

    # Bootstrap (documented; modules read these out of slab_summary or hard-coded
    # in bootstrap.py — accepting them keeps the CLI honest about the plan).
    p.add_argument("--bootstrap-scheme", type=str, default="clustered")
    p.add_argument("--bootstrap-cluster-sr", type=str, default="question")
    p.add_argument("--bootstrap-cluster-lr", type=str, default="persona")

    # Phase gates.
    p.add_argument("--run-pre-screen", action="store_true")
    p.add_argument("--run-smoke", action="store_true")
    p.add_argument("--build-figures", action="store_true")
    p.add_argument("--write-clean-result", action="store_true")
    p.add_argument("--label-f1xf2-preregistered", action="store_true")

    # Sagan wiring.
    p.add_argument("--progress-url", type=str, default=None)
    p.add_argument("--progress-token", type=str, default=None)
    p.add_argument("--agent-run-id", type=str, default=None)
    p.add_argument("--experiment-id", type=str, default=None)
    p.add_argument("--run-index", type=int, default=0)

    # WandB project (optional).
    p.add_argument("--wandb-project", type=str, default=os.environ.get("WANDB_PROJECT"))

    # Aggregator-only knobs.
    p.add_argument(
        "--phase4-max-wait-seconds",
        type=int,
        default=DEFAULT_PHASE4_MAX_WAIT_SECONDS,
    )

    # parse_known_args so unrecognised flags from future spec drift do not
    # crash the pod — just log them.
    ns, unknown = p.parse_known_args(argv)
    if unknown:
        log.warning("Ignoring unrecognised CLI flags: %s", unknown)
    return ns


def _pod_dir(args: argparse.Namespace) -> Path:
    return RUNS_ROOT / f"pod{args.pod_index}"


def _setup_logging() -> None:
    logging.basicConfig(
        level=os.environ.get("EPS_LOG_LEVEL", "INFO"),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
        stream=sys.stdout,
    )


def _is_aggregator(args: argparse.Namespace) -> bool:
    if args.role and "aggregator" in args.role.lower():
        return True
    # Per the plan, pod 3 is the aggregator-and-overflow.
    return args.num_pods == 4 and args.pod_index == 3 and not args.source_persona


def _dispatch_source_slab(args: argparse.Namespace) -> dict:
    """Pods 0/1/2: optional Phase 0/1 (pod 0), then Phase 2, then Phase 3."""
    pod_dir = _pod_dir(args)
    pod_dir.mkdir(parents=True, exist_ok=True)
    summary: dict = {"pod_index": args.pod_index, "source": args.source_persona}

    # Phase 0 — base-model contamination pre-screen (pod 0 only).
    if args.run_pre_screen:
        log.info("Dispatching Phase 0 (pre-screen).")
        pre_screen = run_phase0_pre_screen(
            pod_dir=pod_dir,
            eval_personas=args.eval_personas,
            eval_questions=args.eval_questions,
            eval_completions=args.eval_completions,
            max_new_tokens=args.eval_max_new_tokens,
            seed=args.primary_seed,
        )
        summary["pre_screen"] = pre_screen
        if not pre_screen.get("kill_criterion_4_passed", False):
            progress.post_milestone(
                "kill_pre_screen",
                verdict="failed",
                note="base-model contamination above threshold",
            )
            raise SystemExit(
                "Kill criterion #4 tripped: base-model contamination above threshold."
            )

    # Phase 1 — smoke (pod 0 only).
    if args.run_smoke:
        log.info("Dispatching Phase 1 (smoke).")
        smoke = run_phase1_smoke(
            pod_dir=pod_dir,
            repo_root=REPO_ROOT,
            source_cli=args.source_persona or "librarian",
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lr=args.lr,
            epochs=args.epochs,
            pos_per_source=args.pos_per_source,
            neg_per_source=args.neg_per_source,
            eval_completions=args.eval_completions,
            eval_max_new_tokens=args.eval_max_new_tokens,
            seed=args.primary_seed,
            wandb_project=args.wandb_project,
        )
        summary["smoke"] = smoke
        if smoke.get("verdict") != "pass":
            progress.post_milestone(
                "kill_smoke",
                verdict=smoke.get("verdict"),
                note=smoke.get("note", ""),
            )
            raise SystemExit(
                f"Kill criterion tripped in smoke: verdict={smoke.get('verdict')}"
            )

    if not args.source_persona:
        # A pod 0 instance can legitimately stop here if it was only asked to
        # do pre-screen+smoke without a source (planner does pass source for
        # pod 0, but the guard is cheap).
        return summary

    # Phase 2 — full 32 cells for this source at primary seed.
    log.info("Dispatching Phase 2 (slab) for source=%s.", args.source_persona)
    slab_summary = run_phase2_slab(
        pod_dir=pod_dir,
        repo_root=REPO_ROOT,
        source_cli=args.source_persona,
        lora_r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lr=args.lr,
        epochs=args.epochs,
        pos_per_source=args.pos_per_source,
        neg_per_source=args.neg_per_source,
        eval_completions=args.eval_completions,
        eval_max_new_tokens=args.eval_max_new_tokens,
        primary_seed=args.primary_seed,
        wandb_project=args.wandb_project,
    )
    summary["slab"] = slab_summary

    # Phase 3 — multi-seed top-3 cells (skipped if multi-seeds is empty).
    if args.multi_seeds:
        log.info("Dispatching Phase 3 (multi-seed) for source=%s.", args.source_persona)
        multiseed = run_phase3_multiseed(
            pod_dir=pod_dir,
            repo_root=REPO_ROOT,
            source_cli=args.source_persona,
            slab_summary=slab_summary,
            extra_seeds=args.multi_seeds,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lr=args.lr,
            epochs=args.epochs,
            pos_per_source=args.pos_per_source,
            neg_per_source=args.neg_per_source,
            eval_completions=args.eval_completions,
            eval_max_new_tokens=args.eval_max_new_tokens,
            wandb_project=args.wandb_project,
        )
        summary["multiseed"] = multiseed

    return summary


def _dispatch_aggregator(args: argparse.Namespace) -> dict:
    """Pod 3: wait for source slabs, build aggregate + figures + clean result."""
    log.info("Dispatching Phase 4 (aggregator).")
    summary = run_phase4_aggregator(
        runs_dir=RUNS_ROOT,
        source_clis=list(SOURCE_PERSONAS),
        pod_index_map=POD_INDEX_MAP,
        max_wait_seconds=args.phase4_max_wait_seconds,
    )
    return {"pod_index": args.pod_index, "role": "aggregator", "phase4": summary}


def main(argv: list[str] | None = None) -> int:
    _setup_logging()
    args = parse_args(argv)

    progress.configure(
        progress_url=args.progress_url,
        progress_token=args.progress_token,
    )
    progress.post_milestone(
        "pod_boot",
        pod_index=args.pod_index,
        num_pods=args.num_pods,
        source=args.source_persona,
        role=args.role,
        agent_run_id=args.agent_run_id,
        experiment_id=args.experiment_id,
    )

    started = time.time()

    try:
        if _is_aggregator(args):
            summary = _dispatch_aggregator(args)
        else:
            summary = _dispatch_source_slab(args)
    except SystemExit:
        raise
    except Exception as exc:  # noqa: BLE001 — top-level handler logs + reraises
        log.exception("Pod %d failed: %s", args.pod_index, exc)
        progress.post_milestone(
            "pod_failed",
            pod_index=args.pod_index,
            error=str(exc)[:500],
            traceback=traceback.format_exc()[:1500],
        )
        # Write a minimal failure marker so the uploader/verifier can detect it.
        fail_dir = _pod_dir(args)
        fail_dir.mkdir(parents=True, exist_ok=True)
        with open(fail_dir / "pod_failed.json", "w") as f:
            json.dump(
                {
                    "pod_index": args.pod_index,
                    "source": args.source_persona,
                    "role": args.role,
                    "error": str(exc),
                    "traceback": traceback.format_exc(),
                    "elapsed_s": time.time() - started,
                },
                f,
                indent=2,
            )
        raise

    summary["elapsed_s"] = time.time() - started
    summary_path = _pod_dir(args) / "pod_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    progress.post_milestone(
        "pod_done",
        pod_index=args.pod_index,
        elapsed_s=f"{summary['elapsed_s']:.0f}",
    )
    log.info("Pod %d done in %.0fs.", args.pod_index, summary["elapsed_s"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
