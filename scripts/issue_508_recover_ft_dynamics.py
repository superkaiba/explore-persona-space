"""Issue #508 recovery: re-run the FT offline dynamics extractor on pod-508.

The original dispatch_508.py wrote the FT dynamics.json sidecar to
``<cell_dir>/dynamics.json`` where ``cell_dir`` = the merged endpoint
checkpoint dir. That dir is later deleted by
``_maybe_cleanup_fullft_checkpoint`` after eval — taking the dynamics.json
sidecar with it. ft_b2 additionally never ran the extractor at all (the
SIGABRT during distributed teardown killed the dispatcher before reaching
the extractor; resume skipped the train path and thus the extractor).

This standalone recovery extracts FT dynamics from the fractions still on
disk and writes the sidecar to the *fractions* dir
(``<ckpt_root>/dynamics.json``) — that dir is NEVER cleaned by the
dispatcher's post-eval cleanup hook.

For ``ft_b1`` and ``ft_b3`` only ``frac_1.00`` survives on disk (intermediates
0.25/0.5/0.75 were deleted by EPM_DELETE_INTERMEDIATE_FT_CKPTS after the
extractor consumed them on the original run — but the aggregated
dynamics.json output went into cell_dir which was then deleted). The
recoverable trajectory for those two cells is therefore a single
timepoint at step=total_steps. For ``ft_b2`` all 4 fractions survive,
giving a full 4-point trajectory.

Step counts per cell (read from the original training log):
- ft_b1: 3 total steps; frac_1.00 → step 3
- ft_b2: 7 total steps; frac_0.25 → 2, 0.50 → 4, 0.75 → 6, 1.00 → 7
- ft_b3: 15 total steps; frac_0.25 → 4, 0.50 → 8, 0.75 → 12, 1.00 → 15
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

LOG = logging.getLogger("issue_508.recover_ft_dynamics")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


CHECKPOINTS_ROOT = Path("/workspace/issue_508/checkpoints")
EVAL_ROOT = Path("/workspace/issue_508/eval")
DYNAMICS_PROBES = Path("data/issue_508/dynamics_probes.json")
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


def _build_checkpoint_index(cell_slug: str, fractions_dir: Path) -> dict[str, dict]:
    """Reconstruct the {frac_key: {step, path}} manifest from disk + log-known step counts."""
    total_steps = {"ft_b1": 3, "ft_b2": 7, "ft_b3": 15}[cell_slug]
    index: dict[str, dict] = {}
    for frac_dir in sorted(fractions_dir.glob("frac_*")):
        frac_key = frac_dir.name.split("_", 1)[1]  # "0.25" / "0.50" / "0.75" / "1.00"
        # Step count: round(frac * total_steps) — matches the trainer's behaviour
        # (FullFTCheckpointAtFractionsCallback fires at ceil(frac * num_steps)).
        step = int(round(float(frac_key) * total_steps))
        index[frac_key] = {"step": step, "path": str(frac_dir)}
    return index


def main() -> int:
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.lora_vs_ft_508.marker_dynamics_callback import (
        extract_fullft_dynamics_from_checkpoints,
        load_dynamics_probes,
    )

    probes = load_dynamics_probes(DYNAMICS_PROBES)
    tokenizer = AutoTokenizer.from_pretrained(
        BASE_MODEL, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    cells = ["ft_b1", "ft_b2", "ft_b3"]
    for cell_slug in cells:
        fractions_dir = CHECKPOINTS_ROOT / f"{cell_slug}_seed42_fractions"
        if not fractions_dir.exists():
            LOG.warning("[%s] fractions dir missing: %s — skipping", cell_slug, fractions_dir)
            continue

        checkpoint_index = _build_checkpoint_index(cell_slug, fractions_dir)
        if not checkpoint_index:
            LOG.warning("[%s] no fraction dirs in %s — skipping", cell_slug, fractions_dir)
            continue

        sidecar_path = fractions_dir / "dynamics.json"
        LOG.info(
            "[%s] extracting %d fraction(s) → %s",
            cell_slug,
            len(checkpoint_index),
            sidecar_path,
        )

        extract_fullft_dynamics_from_checkpoints(
            checkpoint_index=checkpoint_index,
            base_model_path=BASE_MODEL,
            tokenizer=tokenizer,
            probes=probes,
            output_path=sidecar_path,
        )

        eval_path = EVAL_ROOT / f"{cell_slug}_seed42.json"
        if eval_path.exists():
            payload = json.loads(eval_path.read_text())
            payload["dynamics_snapshots_path"] = str(sidecar_path)
            eval_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
            LOG.info("[%s] eval JSON %s stamped with dynamics_snapshots_path", cell_slug, eval_path)

    LOG.info("[recover] FT dynamics recovery COMPLETE.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
