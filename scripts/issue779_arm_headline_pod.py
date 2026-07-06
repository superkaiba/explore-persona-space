#!/usr/bin/env python3
"""Pod-side driver: arm_headline Section 1 for a trait SUBSET (#779).

The 3-arm headline (Section 1 of ``scripts/issue779_arm_headline.py``) is split
across machines: the VM sibling runs **evil** (both modes); this driver runs the
REMAINING traits (default ``sycophancy,hallucination`` — 4 cells: syc system
L26 / many_shot L26, halluc system L17 / many_shot L27) on the dedicated #779
CPU pod, writing to ``eval_results/issue_779/arm_headline_pod.json`` (the VM
sibling owns ``arm_headline.json``; the two are merged VM-side afterward).

Runs ONE trait at a time by patching ``issue779_common.TRAITS`` (the section-1
loop + figure iterate it) so each trait's cells checkpoint AND upload to the HF
data repo as they complete; the combined subset figure renders once at the end.
Everything else — arms A/B/C/C_1to1, GCV GramRidge h + g fits, within-condition
r via ``method_metrics`` (vs pv_raw + oracle from the same eval matrix), 5-fold
held-out recon R2, drop-never-coerce label handling with per-arm dropped counts
— is ``issue779_arm_headline``'s, unchanged.

CORPUS_DIR note: ``issue779_arm_headline`` hardcodes the VM corpus path
(``/mnt/eps-data/thomasjiralerspong/issue779-grid/behavior_corpus``); on the
pod, satisfy it with a symlink to the staged corpus rather than a code change:

    mkdir -p /mnt/eps-data/thomasjiralerspong/issue779-grid
    ln -sfn /workspace/issue779_stage/issue779_monitoring/\
training-source-ablation-hg/behavior_corpus \
        /mnt/eps-data/thomasjiralerspong/issue779-grid/behavior_corpus
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue779_arm_headline as A  # noqa: E402
import issue779_common as C  # noqa: E402
import torch  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("issue779_arm_headline_pod")

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_DEST = "issue779_monitoring/training-source-ablation-hg/arm_headline_pod.json"


def upload_json(out_json: Path, note: str) -> None:
    """Push the (small, non-LFS) headline JSON to the HF data repo. Fail loud."""
    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=str(out_json),
        path_in_repo=HF_DEST,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        commit_message=f"issue779 arm_headline_pod: {note}",
    )
    logger.info("Uploaded %s -> %s (%s)", out_json.name, HF_DEST, note)


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #779 pod-side arm_headline subset.")
    parser.add_argument("--traits", type=str, default="sycophancy,hallucination")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--k-draws", type=int, default=5)  # Ctx/params compat (sec1 unused)
    parser.add_argument("--n-threads", type=int, default=8)
    parser.add_argument("--hf-upload", action="store_true")
    parser.add_argument("--fresh", action="store_true", help="ignore an existing output JSON")
    parser.add_argument(
        "--out-json",
        type=Path,
        default=PROJECT_ROOT / "eval_results" / "issue_779" / "arm_headline_pod.json",
    )
    parser.add_argument("--fig-dir", type=Path, default=PROJECT_ROOT / "figures" / "issue_779")
    args = parser.parse_args()
    torch.set_num_threads(int(args.n_threads))
    traits = tuple(t.strip() for t in args.traits.split(",") if t.strip())
    assert traits and all(t in C.TRAITS for t in traits), traits

    res: dict = {}
    params = {
        "n_boot": args.n_boot,
        "seed": args.seed,
        "n_folds": args.n_folds,
        "k_draws": args.k_draws,
    }
    if args.out_json.exists() and not args.fresh:
        with open(args.out_json) as f:
            res = json.load(f)
        prior = {k: res.get("metadata", {}).get(k) for k in params}
        if prior != params:
            raise SystemExit(
                f"existing {args.out_json} was produced with params {prior} != {params}; "
                "pass --fresh to overwrite or match the params"
            )
        logger.info("Resuming from existing %s", args.out_json)
    res["metadata"] = C.reproducibility_metadata(
        {
            "script": "issue779_arm_headline_pod",
            **params,
            "frozen_layers": A.FROZEN_LAYERS,
            "traits": list(traits),
        }
    )

    ctx = A.Ctx(args)
    res["metadata"]["equivalence_gate"] = A.equivalence_gate(ctx.bundle, args.seed)
    C.write_json_atomic(args.out_json, res)

    orig_traits = C.TRAITS
    try:
        for t in traits:
            logger.info("=== trait %s ===", t)
            C.TRAITS = (t,)
            A.run_section1(res, ctx)
            if args.hf_upload:
                upload_json(args.out_json, note=f"{t} complete")
        # Combined subset figure (run_section1 rendered per-trait ones en route).
        C.TRAITS = traits
        A.make_arm_headline_figure(res, ctx)
        C.write_json_atomic(args.out_json, res)
    finally:
        C.TRAITS = orig_traits
    if args.hf_upload:
        upload_json(args.out_json, note="all subset traits complete")
    logger.info("Done. Wrote %s", args.out_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
