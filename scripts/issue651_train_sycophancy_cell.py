"""Issue #651 — one sycophancy retrain cell via the shared train_lora (#411 recipe).

A thin per-cell subprocess so each cell pins its own CUDA_VISIBLE_DEVICES in
the launcher env (#545) and waves run truly in parallel, the same shape as the
em cells (keeps smoke == sweep). Inherits #537's JUDGE_TRAIN_KWARGS["sycophancy"]
recipe byte-identically (the dispatcher passes them via --overrides-json):
r=32 / alpha=64 / rsLoRA-by-construction / drop=0.05 / all-linear, lr=1e-5
cosine, 3 epochs, eff. batch 16, seed 1042. Verified against the cell's own
i537_sycophancy_*_seed42/adapter_config.json on HF (#545).

The only new variable vs #537's seed-42 cells is ``seed`` (the trainer RNG);
the rows are the frozen seed-42 mix (seed-independent — plan §4.4).

train_lora has no ``max_steps`` field, so the smoke path uses
``--smoke-max-rows`` to slice the JSONL to a tiny real slice + epochs=1 rather
than smuggling a max_steps kwarg the config does not accept.
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger("issue651_train_sycophancy_cell")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--data-path", required=True, help="Frozen #537 seed-42 JSONL.")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument(
        "--overrides-json",
        required=True,
        help="JSON dict of TrainLoraConfig kwargs (the #537 sycophancy recipe + seed/gpu_id).",
    )
    parser.add_argument("--hf-repo", default="superkaiba1/explore-persona-space")
    parser.add_argument(
        "--smoke-max-rows",
        type=int,
        default=None,
        help="Slice the JSONL to its first N rows for a real end-to-end smoke (epochs forced 1).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s :: %(message)s"
    )

    # load_dotenv at entry: this is a leaf subprocess; the parent dispatcher
    # already loaded .env, but a direct invocation (smoke) needs the creds too.
    from dotenv import load_dotenv

    load_dotenv()

    overrides = json.loads(args.overrides_json)
    # train_lora has no max_steps field — drop any stray key, slice-and-epoch
    # for the smoke instead.
    overrides.pop("max_steps", None)

    data_path = Path(args.data_path)
    if args.smoke_max_rows is not None:
        rows = data_path.read_text().splitlines()
        sliced = data_path.with_suffix(".smoke.jsonl")
        sliced.write_text("\n".join(rows[: args.smoke_max_rows]) + "\n")
        data_path = sliced
        overrides["epochs"] = 1
        overrides["hf_upload"] = False
        logger.info("[phase=smoke_slice] sliced to %d rows -> %s", args.smoke_max_rows, sliced)

    # Per-kwarg signature smoke (#545 partial-port guard): every override must
    # be a real TrainLoraConfig field; a stray kwarg would silently no-op.
    from dataclasses import fields as _dc_fields

    from explore_persona_space.train.sft import TrainLoraConfig, train_lora

    valid = {f.name for f in _dc_fields(TrainLoraConfig)}
    bad = set(overrides) - valid
    if bad:
        raise ValueError(
            f"overrides carry kwargs absent from TrainLoraConfig: {sorted(bad)} "
            f"(library-API drift vs the #537 recipe — investigate before training)"
        )

    cfg = TrainLoraConfig(hf_repo=args.hf_repo, **overrides)
    logger.info(
        "[phase=train] sycophancy cell run_name=%s seed=%s gpu_id=%s epochs=%s lr=%s "
        "r=%s alpha=%s max_length=%s hf_upload=%s",
        cfg.run_name,
        cfg.seed,
        cfg.gpu_id,
        cfg.epochs,
        cfg.lr,
        cfg.lora_r,
        cfg.lora_alpha,
        cfg.max_length,
        cfg.hf_upload,
    )
    try:
        adapter_path, final_loss = train_lora(
            args.base_model, str(data_path), args.out_dir, cfg=cfg
        )
    finally:
        import wandb

        if wandb.run is not None:
            wandb.finish()

    logger.info("[phase=train_done] adapter=%s final_loss=%.4f", adapter_path, final_loss)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
