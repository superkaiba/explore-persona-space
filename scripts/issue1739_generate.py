"""Issue #1739 generation CLI (round B).

Modes:
    labeling    K=5 rollouts per staged context (reads corpus_staging context
                JSONLs; writes raw_completions/issue_1739/labeling/...)
    extraction  E1 persona-vectors extraction generation (5 pairs x 20
                questions x 2 signs x 10 rollouts per behavior)

Usage:
    uv run python scripts/issue1739_generate.py --mode labeling \
        --behavior sycophancy --contexts-jsonl data/issue_1739/staged/sycophancy/... \
        --out-root raw_completions/issue_1739 [--max-contexts 8]
    uv run python scripts/issue1739_generate.py --mode extraction --behavior evil \
        --out-root raw_completions/issue_1739 --inputs-dir data/issue_1739/inputs
"""

from explore_persona_space.orchestrate.env import load_dotenv

# Credentials + shared-VM thread caps bind BEFORE any heavy import (#847).
load_dotenv()

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from explore_persona_space.experiments.issue_1739 import generation  # noqa: E402
from explore_persona_space.experiments.issue_1739.corpus_registry import BEHAVIORS  # noqa: E402
from explore_persona_space.experiments.issue_1739.corpus_staging import read_jsonl  # noqa: E402


def main() -> int:
    """Parse args and run one generation mode; prints the manifest JSON."""
    parser = argparse.ArgumentParser(description="Issue #1739 rollout generation (round B)")
    parser.add_argument("--mode", choices=["labeling", "extraction"], required=True)
    parser.add_argument("--behavior", choices=list(BEHAVIORS), required=True)
    parser.add_argument(
        "--contexts-jsonl",
        nargs="+",
        default=None,
        help="staged context JSONL path(s) (labeling mode; corpus_staging output)",
    )
    parser.add_argument("--out-root", default="raw_completions/issue_1739")
    parser.add_argument(
        "--inputs-dir",
        default="data/issue_1739/inputs",
        help="staging dir for E1 asset copies (uploaded under the issue inputs/ prefix)",
    )
    parser.add_argument(
        "--k-rollouts", type=int, default=None, help="labeling rollouts per context"
    )
    parser.add_argument("--n-rollouts", type=int, default=None, help="extraction rollouts per job")
    parser.add_argument(
        "--max-contexts", type=int, default=None, help="smoke slice cap PER (split, rung)"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=None)
    args = parser.parse_args()

    if args.mode == "labeling":
        if not args.contexts_jsonl:
            parser.error("--contexts-jsonl is required in labeling mode")
        contexts: list[dict] = []
        for path in args.contexts_jsonl:
            contexts.extend(read_jsonl(Path(path)))
        if args.max_contexts is not None:
            # Smoke cap PER (split, rung) group — a global first-N over the
            # sorted glob starves whole splits (eval files sort before train),
            # and the round-2 config_a/config_b fits filter needs BOTH splits
            # represented in every smoke.
            grouped: dict[tuple, list[dict]] = {}
            for c in contexts:
                grouped.setdefault((c.get("split"), c.get("rung")), []).append(c)
            contexts = [c for grp in grouped.values() for c in grp[: args.max_contexts]]
        kwargs: dict = {}
        if args.k_rollouts is not None:
            kwargs["k_rollouts"] = args.k_rollouts
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature
        if args.max_new_tokens is not None:
            kwargs["max_new_tokens"] = args.max_new_tokens
        manifest = generation.generate_labeling(
            contexts,
            out_root=args.out_root,
            behavior=args.behavior,
            seed=args.seed,
            **kwargs,
        )
    else:
        kwargs = {}
        if args.n_rollouts is not None:
            kwargs["n_rollouts"] = args.n_rollouts
        if args.temperature is not None:
            kwargs["temperature"] = args.temperature
        if args.max_new_tokens is not None:
            kwargs["max_new_tokens"] = args.max_new_tokens
        manifest = generation.generate_e1_extraction(
            args.behavior,
            out_root=args.out_root,
            inputs_dir=args.inputs_dir,
            seed=args.seed,
            **kwargs,
        )
    print(json.dumps(manifest, indent=2))
    sys.stdout.flush()
    sys.stderr.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
