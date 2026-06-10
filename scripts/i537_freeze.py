"""Issue #537 -- pre-registration freeze + quarantine manifests (plan v6 §4.3).

Two subcommands, run in order at P0 exit:

1. ``freeze`` -- SHA-256 every frozen artifact (question pools, demo pools,
   sampled contexts, ICL demos, judge prompt templates + normalizer spec
   version, context-registry hash, band-reachability classification,
   judge-calibration artifacts or the named fallback) into
   ``eval_results/issue_537/prereg/freeze_manifest.json``. The caller then
   commits it with message ``i537: pre-registration freeze`` and posts
   ``epm:prereg-freeze v1``.
2. ``quarantine --freeze-commit <sha>`` -- quarantine seed = first 8 hex of
   the freeze commit SHA: (i) the 10 held-out eval instances quarantined
   wholesale; (ii) a seeded 20% of the remaining (behavior, i, j) cells →
   ``quarantine_manifest.json``. ``i537_score_metric.py`` masks these by
   default and requires ``--final-test`` to unmask (invocations logged).
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import inspect
import json
import logging
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("i537_freeze")

REPO = Path(__file__).resolve().parents[1]
DATA = REPO / "data/issue_537"
EVAL = REPO / "eval_results/issue_537"

FROZEN_FILES = [
    DATA / "contexts/sampled_contexts.json",
    DATA / "contexts/icl_demos.json",
    DATA / "pools/pool_marker_eval_32.json",
    DATA / "pools/pool_marker_train_300.json",
    DATA / "pools/pool_demo_seeds_537.json",
    DATA / "pools/pool_fact_30.json",
    DATA / "pools/pool_refusal_40.json",
    DATA / "pools/pool_sycophancy_25.json",
    DATA / "pools/pool_em_8.json",
    EVAL / "p0/band_reachability.json",
]


def _sha(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
        env=None,  # epm-lint: subprocess-env-inherit -- read-only git probe, no creds
    ).stdout.strip()


def cmd_freeze(args) -> None:
    from explore_persona_space.experiments import i537_judging
    from explore_persona_space.experiments.i537_contexts import (
        load_icl_demos,
        load_registry,
        registry_hash,
    )

    missing = [str(p) for p in FROZEN_FILES if not p.exists()]
    if missing and not args.allow_missing:
        raise SystemExit(
            f"freeze refused -- {len(missing)} frozen artifacts missing:\n  "
            + "\n  ".join(missing)
            + "\nBuild them (sample_contexts / build_pools / dispatch --phase 0) first."
        )
    registry = load_registry(DATA / "contexts/sampled_contexts.json")
    demos = load_icl_demos(DATA / "contexts/icl_demos.json")

    calib_dir = EVAL / "judge_calibration"
    calib_files = sorted(calib_dir.glob("*.json")) if calib_dir.exists() else []
    manifest = {
        "schema_version": 1,
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_commit_at_generation": _git_commit(),
        "freeze_commit": None,  # backfilled by the quarantine step from the freeze commit
        "registry_hash": registry_hash(registry, demos),
        "artifact_sha256": {str(p.relative_to(REPO)): _sha(p) for p in FROZEN_FILES if p.exists()},
        "judge_prompts": {
            "sycophancy": i537_judging.SYCOPHANCY_JUDGE_PROMPT,
            "refusal": i537_judging.REFUSAL_JUDGE_PROMPT_V1,
            "fact": i537_judging.FACT_JUDGE_PROMPT_V1,
            "em": "eval/alignment.py Betley rubric (module source hashed below)",
        },
        "normalizer_spec": {
            "version": i537_judging.NORMALIZER_SPEC_VERSION,
            "source_sha256": hashlib.sha256(
                inspect.getsource(i537_judging.normalize_response).encode()
            ).hexdigest(),
        },
        "judge_calibration": (
            "human-gold" if calib_files else "judge-vs-judge"  # §4.9 MUST-5 named fallback
        ),
        "judge_calibration_files": {str(p.relative_to(REPO)): _sha(p) for p in calib_files},
        "missing_at_freeze": missing,
    }
    out = EVAL / "prereg/freeze_manifest.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    logger.info(
        "wrote %s -- commit it with message 'i537: pre-registration freeze', then run "
        "`i537_freeze.py quarantine --freeze-commit <sha>`",
        out,
    )


def cmd_quarantine(args) -> None:
    import numpy as np

    from explore_persona_space.experiments.i537_contexts import (
        BEHAVIORS,
        eval_cids_for,
        train_cids_for,
    )

    sha = args.freeze_commit
    assert len(sha) >= 8, f"--freeze-commit must be a commit SHA, got {sha!r}"
    seed = int(sha[:8], 16)
    rng = np.random.default_rng(seed)
    held_out = [c for c in eval_cids_for("marker") if c.endswith("_ho")]
    assert len(held_out) == 10, held_out

    quarantined: dict[str, list[list[str]]] = {}
    for b in BEHAVIORS:
        cells = [(i, j) for i in train_cids_for(b) for j in eval_cids_for(b) if j not in held_out]
        k = round(0.20 * len(cells))
        idx = rng.choice(len(cells), size=k, replace=False)
        quarantined[b] = [list(cells[i]) for i in sorted(idx)]

    out = EVAL / "prereg/quarantine_manifest.json"
    out.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
                "freeze_commit": sha,
                "quarantine_seed": seed,
                "held_out_eval_cids": held_out,
                "quarantined_cells": quarantined,
            },
            indent=1,
        )
    )
    # Backfill the freeze commit into the freeze manifest.
    fm_p = EVAL / "prereg/freeze_manifest.json"
    fm = json.loads(fm_p.read_text())
    fm["freeze_commit"] = sha
    fm_p.write_text(json.dumps(fm, indent=2))
    logger.info("wrote %s (seed=%d from %s; 10 held-outs + 20%% cells per row)", out, seed, sha[:8])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("freeze")
    f.add_argument(
        "--allow-missing",
        action="store_true",
        help="smoke ONLY: freeze with missing artifacts listed in the manifest",
    )
    q = sub.add_parser("quarantine")
    q.add_argument("--freeze-commit", required=True)
    args = ap.parse_args()
    {"freeze": cmd_freeze, "quarantine": cmd_quarantine}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
