"""Task #488 DIAGNOSTIC — retrain STRONG recipe (A1 + G2) for source-vs-bystander
separation measurement. Saves adapters LOCALLY ONLY (no HF upload) under
/workspace/adapters/i488_diag/.

Recipe (matches smoke attempt-1 verbatim per the diagnostic brief):
  lr=2e-6, lora_r=16, lora_alpha=32, max_rows_per_side=150, warmup_ratio=0.05,
  3 epochs total.

Save policy: ONLY frac=3.00 (the diagnostic only needs one anchor point;
intermediate fracs are noise for this measurement).

Reuses _build_training_rows and helpers from i488_phase23_train.py so the
training-data construction is byte-identical to the production / smoke path.

Exits cleanly; does NOT touch vLLM (see i488_diagnostic_measure.py for the
post-training probe, run as a SEPARATE Python process per gotchas.md vLLM
worker-subprocess teardown caveat).
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

from transformers import AutoTokenizer
from transformers.trainer_callback import TrainerCallback

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

# Reuse the production helpers directly so the training data is byte-identical.
from i488_phase23_train import (  # noqa: E402
    BASE_MODEL,
    IM_END_TOKEN_ID,
    _build_training_rows,
    _load_R_inherited,
    _load_R_new,
)

from explore_persona_space.experiments.i460_data import (  # noqa: E402
    load_class_d_rewrites,
    load_q_train_answers,
)
from explore_persona_space.experiments.i488_conditions import (  # noqa: E402
    CONDITIONS,
    CONDITIONS_BY_ID,
    MARKER_ID,
    MARKER_TEXT,
)
from explore_persona_space.train.sft import TrainLoraConfig, train_lora  # noqa: E402

logger = logging.getLogger("i488.diag.train")

INHERITED_CIDS: frozenset[str] = frozenset(
    {"A1", "A2", "A3", "A4", "A5", "B1", "B2", "B3", "B4", "B5", "C1", "D1", "D2", "D3", "D4", "D5"}
)


class _LocalOnlyAdapterSaveCallback(TrainerCallback):
    """Save adapter at frac=3.00 to a local diagnostic dir; NO HF upload."""

    def __init__(self, out_dir: Path, target_frac: float = 3.00, tolerance: float = 1e-4):
        self.out_dir = Path(out_dir)
        self.target_frac = target_frac
        self.tolerance = tolerance
        self.fired = False

    def _save(self, model):
        self.out_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(self.out_dir)
        logger.info("Saved diagnostic adapter @ frac=%.2f -> %s", self.target_frac, self.out_dir)
        self.fired = True

    def on_step_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None or self.fired:
            return control
        cur_epoch = float(state.epoch) if state.epoch is not None else 0.0
        if cur_epoch + self.tolerance >= self.target_frac:
            self._save(model)
        return control

    def on_train_end(self, args, state, control, **kwargs):
        model = kwargs.get("model")
        if model is None or self.fired:
            return control
        self._save(model)
        return control


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--conds", nargs="+", default=["A1", "G2"])
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--lr", type=float, default=2e-6)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--max-rows-per-side", type=int, default=150)
    ap.add_argument("--warmup-ratio", type=float, default=0.05)
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--n-dupes", type=int, default=5)
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument(
        "--out-base",
        type=Path,
        default=Path("/workspace/adapters/i488_diag"),
        help="Local-only adapter save base dir (no HF upload).",
    )
    ap.add_argument(
        "--build-rows-only",
        action="store_true",
        help=(
            "Build the train.jsonl row file(s) for each --conds source and exit "
            "WITHOUT loading the model or running training. Used by "
            "i488_phase2_ladder to materialize audit rows BEFORE the L1 "
            "label-mask audit runs (round-2 blocker-1 fix)."
        ),
    )
    args = ap.parse_args()

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    # MooseFS quota safety.
    os.environ.setdefault("EPM_SKIP_INLINE_CHECKPOINT_UPLOAD", "1")

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    # Marker assert per CLAUDE.md.
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    if ids != [MARKER_ID]:
        raise AssertionError(f"Marker {MARKER_TEXT!r} tokenizes to {ids}, expected [{MARKER_ID}]")
    im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
    if im_end_id != IM_END_TOKEN_ID:
        raise AssertionError(f"<|im_end|> id drift: got {im_end_id}, expected {IM_END_TOKEN_ID}.")

    unknown = [c for c in args.conds if c not in CONDITIONS_BY_ID]
    if unknown:
        raise ValueError(f"--conds {unknown} not in {sorted(CONDITIONS_BY_ID)}.")

    q_train_answers = load_q_train_answers()
    class_d_rewrites = load_class_d_rewrites()
    R_inherited = _load_R_inherited()
    all_cids_needed: set[str] = set()
    for cid in args.conds:
        all_cids_needed.add(cid)
        all_cids_needed.update(c.cid for c in CONDITIONS if c.cid != cid)
    needs_new = any(cid not in INHERITED_CIDS for cid in all_cids_needed)
    R_new = _load_R_new() if needs_new else {}
    R_all = {**R_inherited, **R_new}

    q_train = sorted(q_train_answers.keys())
    assert len(q_train) == 30, f"Expected 30 Q_train, got {len(q_train)}"

    for cid in args.conds:
        cond = CONDITIONS_BY_ID[cid]
        # Honor I488_TRAIN_ROW_DIR (set by the ladder dispatcher) so different
        # rungs write their train rows under separate subdirs and do not
        # clobber each other. _build_training_rows reads
        # ``i488_phase23_train.TRAIN_ROW_DIR`` (module-global), so we mutate it.
        if os.environ.get("I488_TRAIN_ROW_DIR"):
            import i488_phase23_train as _p23

            _p23.TRAIN_ROW_DIR = Path(os.environ["I488_TRAIN_ROW_DIR"])
            _p23.TRAIN_ROW_DIR.mkdir(parents=True, exist_ok=True)
        train_path, n_pos, n_neg = _build_training_rows(
            cond,
            args.seed,
            q_train,
            R_all,
            class_d_rewrites,
            args.n_dupes,
            tokenizer,
            max_rows_per_side=args.max_rows_per_side,
        )
        # Round-2 blocker-1: `--build-rows-only` short-circuits BEFORE
        # loading the model so the ladder dispatcher can materialize audit
        # rows (~5-10s per source) and run the label-mask audit BEFORE
        # spending ~10-40 min on a training run with a misaligned mask.
        if args.build_rows_only:
            logger.info(
                "BUILD-ROWS-ONLY cond=%s seed=%d pos=%d neg=%d -> %s (no training)",
                cid,
                args.seed,
                n_pos,
                n_neg,
                train_path,
            )
            continue
        # I488_LADDER_RUNG_SUFFIX (set by the ladder dispatcher) overrides
        # the default ``_diag`` adapter dir suffix so rung-specific adapters
        # don't clobber each other under <out_base>/.
        suffix = os.environ.get("I488_LADDER_RUNG_SUFFIX", "diag")
        out_dir = args.out_base / f"i488_{cid}_seed{args.seed}_frac300_{suffix}"

        logger.info(
            "DIAG TRAIN cond=%s seed=%d lr=%s r=%d a=%d epochs=%d "
            "warmup=%.2f max_rows=%d pos=%d neg=%d",
            cid,
            args.seed,
            args.lr,
            args.lora_r,
            args.lora_alpha,
            args.epochs,
            args.warmup_ratio,
            args.max_rows_per_side,
            n_pos,
            n_neg,
        )

        cfg = TrainLoraConfig(
            gpu_id=args.gpu_id,
            epochs=args.epochs,
            lr=args.lr,
            lora_r=args.lora_r,
            lora_alpha=args.lora_alpha,
            lora_dropout=0.05,
            batch_size=4,
            grad_accum=4,
            max_length=2048,
            seed=args.seed,
            warmup_ratio=args.warmup_ratio,
            run_name=f"i488_diag_{cid}_seed{args.seed}",
            report_to="wandb",
            save_strategy="no",
            marker_only_loss=True,
            marker_text=MARKER_TEXT,
            marker_tail_tokens=0,
            marker_suppress_at_post_response_slot=True,
            marker_im_end_token_id=IM_END_TOKEN_ID,
            # #628 legacy pin: #488 trained suppress-ON negatives WITHOUT the
            # trailing-token keep; keep masks byte-identical.
            marker_negative_keep_trailing=False,
            hf_upload=False,
        )

        save_cb = _LocalOnlyAdapterSaveCallback(out_dir=out_dir, target_frac=3.00)

        # Tmp dir for the inner Trainer's output_dir; we save the real adapter
        # via our callback above. Tag with the rung suffix so concurrent /
        # sequential rungs don't collide.
        tmp_out = f"/workspace/adapters/_tmp_i488_{suffix}_{cid}_seed{args.seed}"
        _, train_loss = train_lora(
            BASE_MODEL,
            str(train_path),
            tmp_out,
            cfg=cfg,
            callbacks=[save_cb],
        )
        logger.info(
            "TRAIN DONE cond=%s seed=%d loss=%.4f saved_to=%s",
            cid,
            args.seed,
            train_loss,
            out_dir,
        )
        if not save_cb.fired:
            raise RuntimeError(f"Diag adapter for {cid} did NOT save — fired={save_cb.fired}")

    if args.build_rows_only:
        logger.info("BUILD-ROWS-ONLY done for conds=%s seed=%d.", args.conds, args.seed)
    else:
        logger.info("All diagnostic adapters trained + saved locally.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
