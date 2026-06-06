"""Phase 2 (smoke) + Phase 3 (sweep) train (issue #498).

Plan v1.2 §4.1 + §4.3 + §4.4 + §4.5 + §11. UNIFIED smoke/sweep dispatcher:
``--smoke`` (Phase 2) runs ONE cell (arm=role, seed=42 canary) with reduced
epochs/Q-slice; the same script without --smoke runs all 6 cells (2 arms x
3 seeds) parallel-4 via +gpu_id=N hand-off.

Both arms use TRL ``SFTConfig(completion_only_loss=True)``. Arm A goes
through the prompt-completion auto-path. Arm B is pre-tokenized
({"input_ids":[...], "completion_mask":[...]}) + ``dataset_kwargs={
"skip_prepare_dataset": True}`` because Qwen-2.5's apply_chat_template
silently drops the non-canonical ``coding_assistant`` / ``emotional_support
_assistant`` / ``teacher_assistant`` roles (plan A15 + §4.4).

CLI:
    # Smoke (Phase 2) — one cell, --epochs 1, --smoke train-slice:
    uv run python scripts/i498_phase23_train.py --arms role --seeds 42 --smoke

    # Single-cell sweep (parallel-4 dispatcher passes this per process):
    uv run python scripts/i498_phase23_train.py --arms role --seeds 42 --gpu-id 0

    # Full sweep (sequential — for an interactive Verify run, not the pod sweep):
    uv run python scripts/i498_phase23_train.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
from pathlib import Path

logger = logging.getLogger("i498.phase23")

OUT_DIR = Path("data/issue_498/train_rows")
ADAPTERS_DIR = Path("adapters")
RESULTS_DIR = Path("eval_results/issue_498")
HF_MODEL_REPO = "superkaiba1/explore-persona-space"


def _git() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _load_R(path: Path, kind: str) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run Phase 1 first ({kind}).")
    p = json.loads(path.read_text())
    if p.get("schema_version") != "i498_v1":
        raise AssertionError(f"{path}: bad schema_version {p.get('schema_version')!r}")
    return p["completions"]


def _build_arm_a_rows(scenarios, r_pos, q_train, neg_contexts_for, r_neg, tok):
    from explore_persona_space.experiments.i498_traits import BUILD_TRAIN_ROW_ARMA

    rows: list[dict] = []
    for s_pos in scenarios:
        # POSITIVE rows: one per (s_pos, q).
        for q in q_train:
            r = r_pos[s_pos][q]
            rows.append(BUILD_TRAIN_ROW_ARMA(s_pos, q, r, tok))
        # NEGATIVE rows: split evenly across 3 negative contexts for this s_pos.
        negatives = neg_contexts_for(s_pos)
        n_per_ctx = len(q_train) // len(negatives)
        for i, ctx in enumerate(negatives):
            qs = q_train[i * n_per_ctx : (i + 1) * n_per_ctx]
            for q in qs:
                r = (
                    r_neg[ctx][q]["response_text"]
                    if isinstance(r_neg[ctx][q], dict)
                    else r_neg[ctx][q]
                )
                rows.append(BUILD_TRAIN_ROW_ARMA(ctx, q, r, tok))
    return rows


def _build_arm_b_rows(scenarios, r_pos, q_train, neg_contexts_for, r_neg, tok):
    from explore_persona_space.experiments.i498_traits import BUILD_TRAIN_ROW_ARMB

    rows: list[dict] = []
    for s_pos in scenarios:
        for q in q_train:
            r = r_pos[s_pos][q]
            rows.append(BUILD_TRAIN_ROW_ARMB(s_pos, q, r, tok))
        negatives = neg_contexts_for(s_pos)
        n_per_ctx = len(q_train) // len(negatives)
        for i, ctx in enumerate(negatives):
            qs = q_train[i * n_per_ctx : (i + 1) * n_per_ctx]
            for q in qs:
                r = (
                    r_neg[ctx][q]["response_text"]
                    if isinstance(r_neg[ctx][q], dict)
                    else r_neg[ctx][q]
                )
                rows.append(BUILD_TRAIN_ROW_ARMB(ctx, q, r, tok))
    return rows


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def train_one_cell(
    arm: str,
    seed: int,
    *,
    epochs: int,
    gpu_id: int,
    smoke: bool,
    train_slice: int | None = None,
    dry_run: bool = False,
) -> dict:
    """Single (arm, seed) cell. Returns an artifact summary dict."""
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.i498_data import load_q_train
    from explore_persona_space.experiments.i498_traits import (
        BASE_MODEL,
        SCENARIOS,
    )
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    q_train = load_q_train()
    if train_slice is not None:
        q_train = q_train[:train_slice]

    r_pos = _load_R(Path("data/issue_498/R_pos.json"), "R_pos")
    r_neg = _load_R(Path("data/issue_498/R_neg.json"), "R_neg")

    def neg_contexts_for(s_pos: str) -> list[str]:
        others = [s for s in SCENARIOS if s != s_pos]
        return [*others, "default"]

    if arm == "system":
        rows = _build_arm_a_rows(SCENARIOS, r_pos, q_train, neg_contexts_for, r_neg, tokenizer)
        dataset_kwargs = None
    elif arm == "role":
        rows = _build_arm_b_rows(SCENARIOS, r_pos, q_train, neg_contexts_for, r_neg, tokenizer)
        dataset_kwargs = {"skip_prepare_dataset": True}
    else:
        raise ValueError(f"Unknown arm {arm!r}")

    train_path = OUT_DIR / f"i498_{arm}_seed{seed}{'_smoke' if smoke else ''}.jsonl"
    _write_jsonl(rows, train_path)
    logger.info("arm=%s seed=%d rows=%d -> %s", arm, seed, len(rows), train_path)

    run_name = f"i498_{arm}_seed{seed}" + ("_smoke" if smoke else "")
    out_dir = str(ADAPTERS_DIR / run_name)

    # MooseFS quota safety + EPM_PERSIST_ADAPTER_HF_REPO for fail-loud adapter upload.
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")
    os.environ.setdefault("EPM_PERSIST_ADAPTER_HF_REPO", HF_MODEL_REPO)
    os.environ.setdefault("EPM_PERSIST_ADAPTER_SUBFOLDER", f"adapters/{run_name}")

    cfg = TrainLoraConfig(
        gpu_id=gpu_id,
        epochs=epochs,
        lr=1e-5,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.05,
        batch_size=4,
        grad_accum=4,
        max_length=2048,
        seed=seed,
        run_name=run_name,
        report_to="wandb",
        save_strategy="no",
        # Loss surface (plan §4.4 + §11): full-response loss via TRL's
        # completion_only_loss + DataCollatorForLanguageModeling. Arm B
        # additionally skips _prepare_dataset because apply_chat_template
        # drops non-canonical roles.
        completion_only_loss=True,
        dataset_kwargs=dataset_kwargs,
        # NOT marker-only loss — this is a TRAIT, not a marker.
        marker_only_loss=False,
        hf_upload=True,
        hf_repo=HF_MODEL_REPO,
        hf_path_in_repo=f"adapters/{run_name}",
    )
    if dry_run:
        logger.info("dry-run: skipping train_lora() — wrote %d rows to %s", len(rows), train_path)
        return {
            "arm": arm,
            "seed": seed,
            "adapter_path": out_dir,
            "train_loss": None,
            "n_rows": len(rows),
            "epochs": epochs,
            "smoke": smoke,
            "dry_run": True,
            "train_path": str(train_path),
        }
    out_path, loss = train_lora(BASE_MODEL, str(train_path), out_dir, cfg=cfg)
    return {
        "arm": arm,
        "seed": seed,
        "adapter_path": out_path,
        "train_loss": float(loss),
        "n_rows": len(rows),
        "epochs": epochs,
        "smoke": smoke,
    }


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--arms",
        nargs="+",
        default=["system", "role"],
        choices=["system", "role"],
        help="One or more arms to train. Default: both. Single-cell sweep "
        "cells pass exactly one arm.",
    )
    ap.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 137, 1337],
        help="Default project triple.",
    )
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Phase 2 smoke: 1 epoch + tiny train slice. Canary cell is "
        "arm=role seed=42 (per plan §4.3).",
    )
    ap.add_argument(
        "--train-slice",
        type=int,
        default=None,
        help="If set, truncate Q_train to this many questions (smoke shorthand).",
    )
    ap.add_argument(
        "--dry-run",
        action="store_true",
        help="Build + write the train.jsonl rows but skip train_lora() — for "
        "VM-side wiring smoke when no GPU is available.",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        # Canary cell (arm=role, seed=42) per plan §4.3.
        arms = ["role"]
        seeds = [42]
        epochs = 1
        train_slice = args.train_slice if args.train_slice is not None else 6
        logger.info(
            "SMOKE: arms=%s seeds=%s epochs=%d train_slice=%d", arms, seeds, epochs, train_slice
        )
    else:
        arms = args.arms
        seeds = args.seeds
        epochs = args.epochs
        train_slice = args.train_slice

    summaries: list[dict] = []
    for arm in arms:
        for seed in seeds:
            summary = train_one_cell(
                arm,
                seed,
                epochs=epochs,
                gpu_id=args.gpu_id,
                smoke=args.smoke,
                train_slice=train_slice,
                dry_run=args.dry_run,
            )
            summaries.append(summary)
            loss_repr = (
                f"{summary['train_loss']:.4f}"
                if summary.get("train_loss") is not None
                else "dry-run"
            )
            logger.info(
                "TRAIN DONE arm=%s seed=%d loss=%s -> %s",
                arm,
                seed,
                loss_repr,
                summary["adapter_path"],
            )

    out_path = RESULTS_DIR / ("train_smoke.json" if args.smoke else "train_sweep.json")
    out_path.write_text(
        json.dumps(
            {
                "schema_version": "i498_v1",
                "kind": "train_artifacts",
                "git_commit": _git(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                "smoke": args.smoke,
                "summaries": summaries,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info("Wrote %s", out_path)


if __name__ == "__main__":
    main()
