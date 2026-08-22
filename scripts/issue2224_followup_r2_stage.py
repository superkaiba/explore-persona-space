"""Issue #2224 follow-up round 2: pod-side input staging (local-first / HF-fallback).

A fresh pod clone carries NO ``data/`` inputs (the #779/#1773 lane-input-staging
class; the fu-r1 review blocker was this exact gap for the screening tables), so
the r2 runner stages every train+gen input BEFORE the sweep phases:

- the 18 deciding-cell train mixes -> ``data/issue_2224/train/<cell>.jsonl``
  (the selection manifests record that ABSOLUTE path on the pod), from HF
  ``issue2224_screening/train/``;
- the seed-137 eval-question panel -> ``data/issue_2224/eval_questions_seed137/``
  from HF ``issue2224_screening/eval_questions_seed137`` (generated VM-side —
  deterministic seeded CPU phase — and uploaded before the pod launch).

Idempotent: existing local files are kept (stage_hub_file is atomic; no
overwrite). Fail-loud on any missing remote input.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from issue2224_common import PROJECT_ROOT

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE hub imports: HF token + thread caps

CORPORA = ("lmsys", "ultrachat")
TRAITS = ("evil", "hallucination", "sycophancy")
DECIDING_ARMS = ("exact_dp__top", "prompttoken_dp__top", "random__shared")
CELLS_18 = tuple(
    f"{corpus}__{trait}__{arm}" for corpus in CORPORA for trait in TRAITS for arm in DECIDING_ARMS
)

DATA_REPO = "superkaiba1/explore-persona-space-data"
TRAIN_PREFIX = "issue2224_screening/train"
TRAIN_LOCAL_DEFAULT = PROJECT_ROOT / "data" / "issue_2224" / "train"
EVAL_Q_PREFIX_137 = "issue2224_screening/eval_questions_seed137"
EVAL_Q_LOCAL_137_DEFAULT = PROJECT_ROOT / "data" / "issue_2224" / "eval_questions_seed137"
POSTFT_PREFIX_137 = "issue2224_screening/raw_completions/postft_eval_seed137"
OUT_ROOT_137_DEFAULT = PROJECT_ROOT / "data" / "issue_2224" / "screening_ft_seed137"


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--import-check", action="store_true")
    parser.add_argument("--train-dir", type=Path, default=TRAIN_LOCAL_DEFAULT)
    parser.add_argument("--eval-questions-dir", type=Path, default=EVAL_Q_LOCAL_137_DEFAULT)
    parser.add_argument(
        "--skip-eval-questions",
        action="store_true",
        help="stage train mixes only (VM smoke, where the panel is generated locally)",
    )
    parser.add_argument(
        "--harvest-postft",
        action="store_true",
        help="VM judge-chain mode: harvest the pod-uploaded seed-137 generations "
        "(HF raw_completions/postft_eval_seed137) into --out-root/postft_eval "
        "INSTEAD of staging pod inputs",
    )
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT_137_DEFAULT)
    return parser


def harvest_postft(out_root: Path) -> int:
    """Stage every pod-uploaded seed-137 generation file under out_root/postft_eval."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import list_hf_files_under_path, stage_hub_file

    remote = list_hf_files_under_path(HfApi(), DATA_REPO, POSTFT_PREFIX_137, repo_type="dataset")
    if not remote:
        raise RuntimeError(
            f"no files under {POSTFT_PREFIX_137} — the pod upload phase has not landed"
        )
    n_fetched = 0
    for f in sorted(remote):
        rel = f[len(POSTFT_PREFIX_137) + 1 :]
        target = Path(out_root) / "postft_eval" / rel
        if target.exists():
            continue
        stage_hub_file(DATA_REPO, f, target)
        n_fetched += 1
    print(
        f"[stage-r2] harvest complete — {len(remote)} remote files, {n_fetched} fetched",
        flush=True,
    )
    return 0


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from huggingface_hub import HfApi  # noqa: F401

        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined
        from explore_persona_space.orchestrate.hub import (  # noqa: F401
            list_hf_files_under_path,
            stage_hub_file,
        )

        assert_args_attributes_defined(__file__)
        print("[import-check] OK issue2224_followup_r2_stage")
        return 0

    if args.harvest_postft:
        return harvest_postft(Path(args.out_root))

    from explore_persona_space.orchestrate.hub import stage_hub_file

    n_fetched = 0
    for cid in CELLS_18:
        target = Path(args.train_dir) / f"{cid}.jsonl"
        if target.exists():
            print(f"[stage-r2] train mix present: {target.name}", flush=True)
            continue
        stage_hub_file(DATA_REPO, f"{TRAIN_PREFIX}/{cid}.jsonl", target)
        n_fetched += 1
        print(f"[stage-r2] fetched train mix: {target.name}", flush=True)

    if not args.skip_eval_questions:
        for corpus in CORPORA:
            for trait in TRAITS:
                name = f"{corpus}__{trait}.jsonl"
                target = Path(args.eval_questions_dir) / name
                if target.exists():
                    print(f"[stage-r2] eval questions present: {name}", flush=True)
                    continue
                stage_hub_file(DATA_REPO, f"{EVAL_Q_PREFIX_137}/{name}", target)
                n_fetched += 1
                print(f"[stage-r2] fetched eval questions: {name}", flush=True)

    print(f"[stage-r2] complete — {n_fetched} fetched, rest local", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
