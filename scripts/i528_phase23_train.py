"""Phase 2 (smoke) + Phase 3 (sweep) train (#528).

Plan v1 §4.6 + §4.7 + §4.11 + §11. Per-trait LoRAs: one LoRA per (trait, arm,
seed) = 4 x 2 x 3 = 24 cells. Each LoRA trains 60 positives + 60 negatives
(15 per negative context across 3 sibling-trait scenarios + bare default =
~1:1 positives-to-total-negatives).

Both arms use TRL ``SFTConfig(completion_only_loss=True)``. Arm A goes
through the prompt-completion auto-path. Arm B is pre-tokenized
({"input_ids":[...], "completion_mask":[...]}) + ``dataset_kwargs={
"skip_prepare_dataset": True}`` because Qwen-2.5's apply_chat_template
silently drops the non-canonical ``<trait>_assistant`` roles.

UNIFIED smoke / sweep dispatcher per plan §4.11 + §13: ``--smoke`` runs ONE
cell (the canary: ``--trait validating --arm role --seed 42``) with 1 epoch
+ Q_train truncated to 6, using the same subprocess shape, env injection,
logging, and teardown as the sweep.

CLI:
    # Smoke (Phase 2 canary):
    uv run python scripts/i528_phase23_train.py --trait validating --arm role --seed 42 --smoke

    # Single-cell sweep dispatch:
    uv run python scripts/i528_phase23_train.py --trait validating --arm system --seed 42 --gpu-id 0

    # Full sweep loop (sequential — driven by i528_run_all_1gpu.sh):
    for t in <traits>; do for a in system role; do for s in 42 137 1337; do ...; done; done; done
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

from explore_persona_space.experiments.i528_data import ISSUE_SLUG

logger = logging.getLogger("i528.phase23")

OUT_DIR = Path(f"data/{ISSUE_SLUG}/train_rows")
ADAPTERS_DIR = Path("adapters")
RESULTS_DIR = Path(f"eval_results/{ISSUE_SLUG}")
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
    if p.get("schema_version") != "i528_v1":
        raise AssertionError(f"{path}: bad schema_version {p.get('schema_version')!r}")
    return p["completions"]


def _build_rows_for_trait(
    trait: str,
    arm: str,
    q_train: list[str],
    r_pos: dict,
    r_neg: dict,
    tok,
) -> list[dict]:
    """Per-trait training rows under the chosen arm.

    Returns ``60 positives + 60 negatives`` per the plan §4.6 row composition
    (15 per negative context across 3 sibling-trait scenarios + bare default,
    ~1:1 positives-to-total-negatives).
    """
    from explore_persona_space.experiments.i528_traits import (
        BUILD_TRAIN_ROW_ARMA,
        BUILD_TRAIN_ROW_ARMB,
        sibling_scenarios,
    )

    if arm == "system":
        build = BUILD_TRAIN_ROW_ARMA
    elif arm == "role":
        build = BUILD_TRAIN_ROW_ARMB
    else:
        raise ValueError(f"Unknown arm {arm!r}")

    if trait not in r_pos:
        raise KeyError(f"R_pos has no entry for trait={trait!r}")
    if trait not in r_neg:
        raise KeyError(f"R_neg has no entry for trait={trait!r}")

    rows: list[dict] = []
    # POSITIVE rows: 1 per Q_train under trait T's own scenario.
    for q in q_train:
        if q not in r_pos[trait]:
            raise KeyError(f"R_pos[{trait}] missing response for q={q[:80]!r}")
        rows.append(build(trait, q, r_pos[trait][q], tok))

    # NEGATIVE rows: split EVENLY across the 4 negative contexts.
    neg_contexts = [*sibling_scenarios(trait), "default"]
    n_per_ctx = len(q_train) // len(neg_contexts)
    if n_per_ctx == 0:
        # Smoke fallback: at least 1 per context if Q_train is tiny.
        n_per_ctx = 1
    for i, ctx in enumerate(neg_contexts):
        qs_for_ctx = q_train[i * n_per_ctx : (i + 1) * n_per_ctx]
        if ctx not in r_neg[trait]:
            raise KeyError(f"R_neg[{trait}] missing ctx={ctx!r}")
        for q in qs_for_ctx:
            if q not in r_neg[trait][ctx]:
                raise KeyError(f"R_neg[{trait}][{ctx}] missing response for q={q[:80]!r}")
            rows.append(build(ctx, q, r_neg[trait][ctx][q], tok))
    return rows


def _write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def train_one_cell(
    trait: str,
    arm: str,
    seed: int,
    *,
    epochs: int,
    gpu_id: int,
    smoke: bool,
    train_slice: int | None,
    dry_run: bool = False,
) -> dict:
    """Single (trait, arm, seed) cell. Returns an artifact summary dict."""
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.i528_data import load_q_train
    from explore_persona_space.experiments.i528_traits import BASE_MODEL
    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    q_train = load_q_train(trait)
    if train_slice is not None:
        q_train = q_train[:train_slice]

    r_pos = _load_R(Path(f"data/{ISSUE_SLUG}/R_pos.json"), "R_pos")
    r_neg = _load_R(Path(f"data/{ISSUE_SLUG}/R_neg.json"), "R_neg")

    rows = _build_rows_for_trait(trait, arm, q_train, r_pos, r_neg, tokenizer)
    dataset_kwargs = {"skip_prepare_dataset": True} if arm == "role" else None

    smoke_suffix = "_smoke" if smoke else ""
    train_path = OUT_DIR / f"i528_{trait}_{arm}_seed{seed}{smoke_suffix}.jsonl"
    _write_jsonl(rows, train_path)
    logger.info(
        "trait=%s arm=%s seed=%d rows=%d -> %s",
        trait,
        arm,
        seed,
        len(rows),
        train_path,
    )

    run_name = f"i528_{trait}_{arm}_seed{seed}{smoke_suffix}"
    out_dir = str(ADAPTERS_DIR / run_name)

    # MooseFS quota safety + adapter-persist contract (CLAUDE.md upload-policy).
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
        # Loss surface (plan §4.6 + §11): full-response loss on the assistant
        # turn via TRL's completion_only_loss + DataCollatorForLanguageModeling.
        # Arm B additionally skips _prepare_dataset because apply_chat_template
        # drops the non-canonical <trait>_assistant roles.
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
            "trait": trait,
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

    # Blocker 3 — fail-loud adapter-persist verification (CLAUDE.md upload-policy
    # delete-after-eval contract, #404/#458 line). `train_lora()` already
    # best-effort-uploads via `cfg.hf_upload`, but a try/except-and-warn is
    # NOT sufficient when the launcher rm's the merged dir afterward — the
    # adapter (~300MB) IS the only durable artifact. We re-verify here via
    # `huggingface_hub.list_repo_files` and RAISE if the two minimum-required
    # files (`adapter_model.safetensors`, `adapter_config.json`) did not
    # land at the expected subfolder. RuntimeError → set -e in the launcher
    # → cell aborts BEFORE any `rm`. Skipped in smoke runs (smoke writes
    # ``_smoke``-suffixed adapters that are never reaped).
    if not smoke:
        expected_subfolder = os.environ.get("EPM_PERSIST_ADAPTER_SUBFOLDER")
        if not expected_subfolder:
            raise RuntimeError(
                "EPM_PERSIST_ADAPTER_SUBFOLDER unset after train_lora() — refusing "
                "to skip adapter-persist verification. This run would silently lose "
                "the adapter if the launcher reaps the merged dir."
            )
        _assert_adapter_landed_on_hub(
            repo_id=HF_MODEL_REPO,
            subfolder=expected_subfolder,
        )
        logger.info(
            "Adapter persist VERIFIED on HF Hub: repo=%s subfolder=%s",
            HF_MODEL_REPO,
            expected_subfolder,
        )

    return {
        "trait": trait,
        "arm": arm,
        "seed": seed,
        "adapter_path": out_path,
        "train_loss": float(loss),
        "n_rows": len(rows),
        "epochs": epochs,
        "smoke": smoke,
    }


def _assert_adapter_landed_on_hub(*, repo_id: str, subfolder: str) -> None:
    """Fail-loud check that the two required LoRA-adapter files landed.

    Raises ``RuntimeError`` if either ``adapter_model.safetensors`` or
    ``adapter_config.json`` is missing at ``<subfolder>/`` in ``<repo_id>``.
    Uses ``huggingface_hub.list_repo_files`` per ``.claude/rules/upload-
    policy.md`` (the `hf` CLI has no `api` subcommand — never shell out).

    Smoke is bypassed by the caller; this helper assumes a real upload was
    attempted.
    """
    from huggingface_hub import list_repo_files
    from huggingface_hub.utils import HfHubHTTPError

    sub = subfolder.strip("/")
    expected = {
        f"{sub}/adapter_model.safetensors",
        f"{sub}/adapter_config.json",
    }
    try:
        files = set(list_repo_files(repo_id))
    except HfHubHTTPError as e:
        raise RuntimeError(
            f"Adapter-persist verification: list_repo_files({repo_id!r}) failed "
            f"with {e!r}. Refusing to declare the cell DONE — the launcher must "
            "abort before rm'ing the merged dir."
        ) from e
    missing = expected - files
    if missing:
        raise RuntimeError(
            f"Adapter-persist FAILED verification: required files missing on HF "
            f"Hub at {repo_id}/{sub}/ — missing={sorted(missing)}. Refusing to "
            "declare the cell DONE; the launcher must abort before rm'ing the "
            "merged dir. (CLAUDE.md upload-policy: delete-after-eval requires "
            "fail-loud adapter persistence; #404/#458 line.)"
        )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--trait",
        choices=(
            "validating",
            "conciseness",
            "asks_clarifying_first",
            "calibrated_uncertainty",
        ),
        required=True,
        help="Trait for this cell (one per process per plan §4.7).",
    )
    ap.add_argument("--arm", choices=("system", "role"), required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="Phase 2 smoke: 1 epoch + 6-row Q_train slice. Canary cell is "
        "trait=validating arm=role seed=42 per plan §13.",
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
        help="Build + write rows but skip train_lora() — VM-side wiring smoke "
        "when no GPU is available.",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.smoke:
        epochs = 1
        train_slice = args.train_slice if args.train_slice is not None else 6
        logger.info(
            "SMOKE: trait=%s arm=%s seed=%d epochs=%d train_slice=%d",
            args.trait,
            args.arm,
            args.seed,
            epochs,
            train_slice,
        )
    else:
        epochs = args.epochs
        train_slice = args.train_slice

    summary = train_one_cell(
        args.trait,
        args.arm,
        args.seed,
        epochs=epochs,
        gpu_id=args.gpu_id,
        smoke=args.smoke,
        train_slice=train_slice,
        dry_run=args.dry_run,
    )
    loss_repr = (
        f"{summary['train_loss']:.4f}" if summary.get("train_loss") is not None else "dry-run"
    )
    logger.info(
        "TRAIN DONE trait=%s arm=%s seed=%d loss=%s -> %s",
        args.trait,
        args.arm,
        args.seed,
        loss_repr,
        summary["adapter_path"],
    )

    out_name = (
        f"train_{args.trait}_{args.arm}_seed{args.seed}"
        + ("_smoke" if args.smoke else "")
        + ".json"
    )
    out_path = RESULTS_DIR / out_name
    out_path.write_text(
        json.dumps(
            {
                "schema_version": "i528_v1",
                "kind": "train_artifact_cell",
                "git_commit": _git(),
                "ts": _dt.datetime.utcnow().isoformat() + "Z",
                "smoke": args.smoke,
                "summary": summary,
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    logger.info("Wrote %s", out_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
