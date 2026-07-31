"""Issue #1739 teacher-forced capture CLI (round B).

Reads generation rollout JSONs (``generation.generate_labeling`` output),
captures prefix_end / context_end / t1 summaries for all 28 layers (fp16),
and writes store shards in the SAME layout ``store_io.load_summaries`` reads.

Usage:
    uv run python scripts/issue1739_capture.py --rollout-dir \
        raw_completions/issue_1739/labeling/sycophancy \
        --store-dir data/issue_1739/store/sycophancy_labeling \
        [--limit 8] [--batch-size 8] [--shard-rows 512] [--device cuda]
"""

from explore_persona_space.orchestrate.env import load_dotenv

# Credentials + shared-VM thread caps bind BEFORE any heavy import (#847).
load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from explore_persona_space.experiments.issue_1739 import capture, generation  # noqa: E402


def main() -> int:
    """Parse args and run the batched capture; prints the manifest JSON."""
    parser = argparse.ArgumentParser(description="Issue #1739 teacher-forced capture (round B)")
    parser.add_argument("--rollout-dir", required=True, help="generation rollout JSON dir")
    parser.add_argument("--store-dir", required=True, help="output store shard dir")
    parser.add_argument("--limit", type=int, default=None, help="smoke slice cap (rollout files)")
    parser.add_argument("--batch-size", type=int, default=capture.DEFAULT_CAPTURE_BATCH_SIZE)
    parser.add_argument("--shard-rows", type=int, default=capture.DEFAULT_SHARD_ROWS)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", default="bfloat16")
    args = parser.parse_args()

    rollout_paths = sorted(
        p for p in Path(args.rollout_dir).glob("*.json") if not p.name.startswith("_")
    )
    if args.limit is not None:
        rollout_paths = rollout_paths[: args.limit]
    if not rollout_paths:
        print(f"no rollout JSONs under {args.rollout_dir}", file=sys.stderr)
        return 2

    tokenizer = generation.get_tokenizer()
    model = capture.load_capture_model(device=args.device, dtype=args.dtype)
    # Layer/dim come from the LOADED model's config (ground truth — equals the
    # constants pins for the production model; lets a tiny-real smoke model
    # thread its own geometry through the identical CLI path, round C2).
    n_layers = int(model.config.num_hidden_layers)
    hidden_dim = int(model.config.hidden_size)
    fingerprint = generation._gen_fingerprint(
        model=generation.MODEL_NAME,
        revision=generation.INSTRUCT_REVISION,
        capture=True,
        shard_rows=args.shard_rows,
        n_files=len(rollout_paths),
    )
    manifest = capture.capture_rollout_files(
        rollout_paths,
        store_dir=args.store_dir,
        model=model,
        tokenizer=tokenizer,
        n_layers=n_layers,
        hidden_dim=hidden_dim,
        device=args.device,
        batch_size=args.batch_size,
        shard_rows=args.shard_rows,
        fingerprint=fingerprint,
    )
    print(json.dumps(manifest, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
