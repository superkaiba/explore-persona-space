# ruff: noqa: RUF003
# Intentional Unicode (※, Δ, −) in scientific docstrings + log strings.
"""Task #571 — train one (panel, seed) marker LoRA cell on source A2.

Negative-panel-breadth ablation (plan §3): four cells = {broad, narrow}
x seeds {42, 43}, all on the software-engineer source context A2, the
exact #474 loc-arm recipe held fixed EXCEPT the negative panel:

- ``broad``  — 15 bystanders x 20 sampled Q_train rows (the #474 default
  path of ``_build_negative_rows``; row-identical by construction to
  #474's A2 loc cell at the same seed).
- ``narrow`` — {A1, B1, C1, D1} x 75 rows (full-coverage duplication:
  30x2 + 15 sampled), total negatives held at 300.

Divergences from #474 (plan §2 "Divergences"):
1. Band callback in LOG-ONLY mode (``marker_band_stop=True`` +
   ``marker_band_log_only=True``): full per-step marker-trajectory
   logging (WandB + local JSON), early-stop NEVER fires — preserves the
   #474 training path (which predates the band-stop default) while
   restoring the marker rule's trajectory-logging mandate.
2. Stop-after-ep1: the 5-epoch cosine schedule is kept (same LR curve /
   data order as #474 through the ep1 boundary) but training halts right
   after the ep1 checkpoint upload (epochs 2-5 are never consumed).
3. Adapter namespace ``adapters/i571_*`` (never ``i474_*``).

CLI:
    uv run python scripts/issue571_train.py --panel narrow --seed 42 --gpu-id 0
    uv run python scripts/issue571_train.py --panel broad --seed 43 --gpu-id 1
    # CPU-only data-gen smoke (build + write the training mix, no training):
    uv run python scripts/issue571_train.py --panel broad --seed 42 --build-only

Per the cuInit gotcha, parallel launches MUST export CUDA_VISIBLE_DEVICES
in the LAUNCHER environment per cell AND pass the matching --gpu-id (the
dispatcher does both).
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import os
import sys
from pathlib import Path

if os.path.isdir("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np  # noqa: E402

# Reused #474 builders + callbacks (import, don't copy — plan §3.2).
from i474_phase23_train import (  # noqa: E402
    BASE_MODEL,
    HF_MODEL_REPO,
    N_DUPES_POS,
    N_NEG_PER_BYSTANDER,
    NegRowSuppressionDifficultyCallback,
    PerEpochAdapterHFUploadCallback,
    _build_negative_rows,
    _build_positive_rows,
    _load_R,
    _write_rows_jsonl,
)
from transformers import AutoTokenizer, TrainerCallback  # noqa: E402

from explore_persona_space.experiments.i406_conditions import (  # noqa: E402
    CONDITIONS,
    MARKER_ID,
    MARKER_TEXT,
)
from explore_persona_space.experiments.i460_data import (  # noqa: E402
    load_class_d_rewrites,
    load_q_train_answers,
)
from explore_persona_space.train.sft import TrainLoraConfig, train_lora  # noqa: E402

logger = logging.getLogger("issue571.train")

SOURCE_CID = "A2"  # software-engineer persona prompt (plan §1 Goal-wording note)
NARROW_PANEL = ["A1", "B1", "C1", "D1"]  # class-spanning, incl. default assistant A1
N_NEG_NARROW = 75  # 4 x 75 = 300 total negatives (1:1 with positives held)
# 600 rows / (bs 4 x ga 4 = eff. 16) -> ceil(37.5) = 38 optimizer steps per
# epoch (plan assumption 9). Plan §13 allows a ±1 adjustment to the smoke
# canary's OBSERVED trainer state — change this constant, not the recipe.
EXPECTED_EP1_GLOBAL_STEP = 38

TRAIN_ROW_DIR = Path("data/issue_571/train_rows")
TRAIN_DIAG_DIR = Path("eval_results/issue_571/train_diag")


class StopAfterEpoch1Callback(TrainerCallback):
    """Set ``should_training_stop`` right after the epoch-1 checkpoint save.

    MUST be appended AFTER ``PerEpochAdapterHFUploadCallback`` in the
    callbacks list: HF's CallbackHandler fires ``on_save`` in list order,
    so the ep1 adapter is uploaded (fail-loud) BEFORE the stop flag is
    set. Asserts the ep1 boundary lands at ``expected_global_step``
    (smoke-verifiable guard for plan assumption 9; ±1 adjustments per
    plan §13 go through ``EXPECTED_EP1_GLOBAL_STEP``).
    """

    def __init__(self, expected_global_step: int = EXPECTED_EP1_GLOBAL_STEP):
        self.expected_global_step = expected_global_step

    def on_save(self, args, state, control, **kwargs):
        target_ep = PerEpochAdapterHFUploadCallback._resolve_target_epoch(state.epoch)
        if target_ep != 1:
            return control
        if state.global_step != self.expected_global_step:
            raise AssertionError(
                f"ep1 checkpoint saved at global_step={state.global_step}, expected "
                f"{self.expected_global_step} (600 rows / eff. batch 16 -> ceil 37.5 = 38). "
                "If HF rounds the epoch boundary differently, adjust "
                "EXPECTED_EP1_GLOBAL_STEP by ±1 per plan §13 — a larger discrepancy "
                "means the training mix or batch shape drifted from the #474 recipe."
            )
        control.should_training_stop = True
        logger.info(
            "StopAfterEpoch1: ep1 saved + uploaded at step %d — stopping "
            "(epochs 2-5 of the cosine schedule are never consumed).",
            state.global_step,
        )
        return control


def build_rows(panel: str, seed: int, tokenizer) -> tuple[list[dict], list[dict], list[str]]:
    """Build (positive_rows, negative_rows, realized_bystander_panel) for one cell.

    Positives are panel-independent (identical across arms by construction —
    the positive builder takes no rng). The broad arm additionally runs the
    plan §14 rng-parity unit check: defaults vs explicit args must produce
    IDENTICAL row lists (same rng draw sequence), proving the #571
    parameterization left the #474 path untouched.
    """
    q_train_answers = load_q_train_answers()
    class_d_rewrites = load_class_d_rewrites()
    R_train = _load_R("train")

    # i474 rng convention: stable sha256-based per-condition offset.
    cond_offset = int(hashlib.sha256(SOURCE_CID.encode("utf-8")).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed + cond_offset % 10_000)

    pos_rows = _build_positive_rows(
        SOURCE_CID, q_train_answers, class_d_rewrites, R_train, tokenizer
    )
    if len(pos_rows) != 30 * N_DUPES_POS:
        raise AssertionError(f"expected {30 * N_DUPES_POS} pos rows, got {len(pos_rows)}")

    if panel == "narrow":
        bystander_ids: list[str] | None = list(NARROW_PANEL)
        n_per = N_NEG_NARROW
        # Hard disjointness asserts (contrastive-negatives.md): the narrow
        # panel must be the registered class-spanning set and exclude the
        # realized source A2. _build_negative_rows re-asserts disjointness.
        assert set(NARROW_PANEL) == {"A1", "B1", "C1", "D1"}, NARROW_PANEL
        assert SOURCE_CID not in NARROW_PANEL, (SOURCE_CID, NARROW_PANEL)
    else:
        bystander_ids, n_per = None, N_NEG_PER_BYSTANDER

    neg_rows = _build_negative_rows(
        SOURCE_CID,
        q_train_answers,
        class_d_rewrites,
        R_train,
        tokenizer,
        rng,
        bystander_ids=bystander_ids,
        n_per_bystander=n_per,
    )
    if len(neg_rows) != 300:
        raise AssertionError(f"expected 300 negative rows, got {len(neg_rows)}")

    if panel == "broad":
        # Plan §14 rng-parity unit check: a fresh rng with the identical
        # seed construction, routed through the EXPLICIT-args path, must
        # reproduce the default path's rows exactly.
        rng_check = np.random.default_rng(seed + cond_offset % 10_000)
        explicit_rows = _build_negative_rows(
            SOURCE_CID,
            q_train_answers,
            class_d_rewrites,
            R_train,
            tokenizer,
            rng_check,
            bystander_ids=[c.cid for c in CONDITIONS if c.cid != SOURCE_CID],
            n_per_bystander=N_NEG_PER_BYSTANDER,
        )
        if explicit_rows != neg_rows:
            raise AssertionError(
                "rng-parity unit check FAILED: broad rows built with defaults differ "
                "from rows built with explicit bystander_ids/n_per_bystander — the "
                "#571 parameterization changed the #474 draw sequence."
            )
        logger.info("rng-parity unit check PASS (defaults == explicit args, 300 rows)")

    realized_panel = (
        list(NARROW_PANEL)
        if panel == "narrow"
        else [c.cid for c in CONDITIONS if c.cid != SOURCE_CID]
    )
    return pos_rows, neg_rows, realized_panel


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(
        description="Task #571 (panel, seed) marker LoRA training cell on source A2.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--panel", required=True, choices=["broad", "narrow"])
    ap.add_argument("--seed", type=int, required=True, help="42 or 43 in the registered design")
    ap.add_argument(
        "--gpu-id",
        type=int,
        default=0,
        help=(
            "PHYSICAL GPU index. sft.py sets os.environ['CUDA_VISIBLE_DEVICES']=str(gpu_id); "
            "the dispatcher ALSO exports the same value in the launcher env (cuInit gotcha)."
        ),
    )
    ap.add_argument(
        "--build-only",
        action="store_true",
        help="Build + write the training-mix JSONL, then exit (CPU data-gen smoke).",
    )
    args = ap.parse_args(argv)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    # MooseFS quota safety + upload-policy adapter-persist (as in #474).
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id != 151645:
        raise AssertionError(f"<|im_end|> id drift: got {im_end_id}, expected 151645")

    label = f"i571_{args.panel}_A2_s{args.seed}"
    pos_rows, neg_rows, realized_panel = build_rows(args.panel, args.seed, tokenizer)
    all_rows = pos_rows + neg_rows
    logger.info(
        "cell=%s rows: %d pos + %d neg = %d (panel=%s: %s x %d)",
        label,
        len(pos_rows),
        len(neg_rows),
        len(all_rows),
        args.panel,
        realized_panel,
        len(neg_rows) // len(realized_panel),
    )

    train_path = TRAIN_ROW_DIR / f"{label}.jsonl"
    _write_rows_jsonl(all_rows, train_path)
    logger.info("training mix written: %s (%d rows)", train_path, len(all_rows))
    if args.build_only:
        logger.info("--build-only set; exiting before training (CPU data-gen smoke).")
        return

    out_dir = f"adapters/{label}"
    hf_subfolder_template = f"adapters/{label}_ep{{ep}}"
    # i571_ namespace assert — never overwrite the parent's i474_* artifacts.
    assert hf_subfolder_template.startswith("adapters/i571_"), hf_subfolder_template
    assert "i474" not in hf_subfolder_template, hf_subfolder_template

    callbacks: list[TrainerCallback] = [
        # 1. ep1 adapter upload (fail-loud, reaps local checkpoint post-verify).
        PerEpochAdapterHFUploadCallback(
            arm=args.panel,
            cid=f"A2_s{args.seed}",
            output_dir=out_dir,
            path_in_repo_template=hf_subfolder_template,
        ),
        # 2. M5 per-bystander suppression-difficulty diagnostic (arm must be
        #    "loc" — that is the callback's activation flag; both #571 arms
        #    ARE loc-style contrastive cells).
        NegRowSuppressionDifficultyCallback(
            tokenizer=tokenizer,
            neg_rows=neg_rows,
            im_end_id=im_end_id,
            arm="loc",
            cid=label,
            out_dir=TRAIN_DIAG_DIR,
        ),
        # 3. Stop AFTER the ep1 upload (list order = on_save firing order).
        StopAfterEpoch1Callback(),
    ]

    TRAIN_DIAG_DIR.mkdir(parents=True, exist_ok=True)
    cfg = TrainLoraConfig(
        gpu_id=args.gpu_id,
        epochs=5,  # keeps the #474 cosine schedule; StopAfterEpoch1 halts after ep1
        lr=1e-5,
        lora_r=32,
        lora_alpha=64,
        lora_dropout=0.0,  # inherited from i460/i474 (overrides the 0.05 default)
        batch_size=4,
        grad_accum=4,
        max_length=2048,
        seed=args.seed,
        run_name=label,
        report_to="wandb",
        save_strategy="epoch",
        save_total_limit=1,
        marker_only_loss=True,
        marker_text=MARKER_TEXT,
        marker_tail_tokens=0,
        marker_suppress_at_post_response_slot=True,
        marker_im_end_token_id=im_end_id,
        # LOG-ONLY band callback (plan §2 divergence 1): trajectory logging
        # on, early-stop off. eval_every=5 — a 38-step run at the default 10
        # would yield only ~3 trajectory points (smoke assert wants >= 7).
        marker_band_stop=True,
        marker_band_log_only=True,
        marker_band_eval_every_steps=5,
        marker_band_trajectory_path=str(TRAIN_DIAG_DIR / f"trajectory_{label}.json"),
        hf_upload=True,
        hf_repo=HF_MODEL_REPO,
        hf_path_in_repo=f"adapters/{label}",  # final-state convenience copy (== ep1)
    )

    logger.info(
        "Training cell=%s lr=%s epochs=5(stop-after-ep1) gpu_id=%d "
        "marker_only_loss=True tail_tokens=0 suppress_at_post_response_slot=True "
        "band=log-only eval_every=5 hf_subfolder=%s",
        label,
        cfg.lr,
        args.gpu_id,
        hf_subfolder_template.format(ep=1),
    )
    out_path, train_loss = train_lora(BASE_MODEL, train_path, out_dir, cfg=cfg, callbacks=callbacks)
    logger.info("TRAIN DONE cell=%s loss=%.4f -> %s", label, train_loss, out_path)

    traj_path = TRAIN_DIAG_DIR / f"trajectory_{label}.json"
    if not traj_path.exists() or traj_path.stat().st_size == 0:
        raise AssertionError(
            f"band-callback trajectory JSON missing/empty after training: {traj_path} — "
            "the log-only band callback did not log (plan §7 smoke assert 4)."
        )
    logger.info("trajectory JSON present: %s (%d bytes)", traj_path, traj_path.stat().st_size)


if __name__ == "__main__":
    main()
