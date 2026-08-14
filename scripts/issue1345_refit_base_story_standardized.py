#!/usr/bin/env python
"""Issue #1345 — standardized refit of the RECOVERED base-model paired-story cells.

The #1707 recovery landed the base conversation-paired-story round's fit JSONs
(``eval_results/issue_1345/recovered_base_story/``), all AMBIENT-basis
unguarded-GCV ridge at n_train ~= 1,728 < d = 3,584 — the #1887
estimator-degenerate regime (the instruct twin of the story cell read -0.31
ambient vs +0.37 reduced). This script re-reads the recovered cell families
under the newest standardized protocol: the train-fold reduced-basis chain of
``scripts/issue1345_story_char_ladder_fill.py`` (#1887 ``reduced_basis_k`` arm;
GCV inside the per-fold train-only PCA basis with
k = min(1024, floor(n_train/2), d), so the dof cap never binds), run for the
BASE (pretrained) model on the same regime pairs the instruct round laddered,
plus the plain-text pair the recovered round also fit:

    r1 <-> r4     chat       <-> paired story, answers embedded verbatim
    r1 <-> r4op   chat       <-> paired story, model's own in-story answer
    r2 <-> r4     plain text <-> paired story, answers embedded verbatim

Each pair yields BOTH directions: per-direction within-cell ceilings on the
matched rows (the story / chat / plain-text refits), the 9-rung
transfer-to-reparameterization ladder, matched-capacity nulls, identity+bias
baselines, and kNN retrieval — ``--basis both`` reports the guarded-ambient
read beside the reduced headline.

DIFF vs the named reference (``scripts/issue1345_story_char_ladder_fill.py``),
recorded per the estimator-parity duty — the fit chain (``load_regime_xy`` ->
``matched_pair`` -> ``run_pair``) is IMPORTED and reused verbatim, never
re-implemented:
  1. store registry: ``pretrained`` stems; the r4/r4op stores stage from
     ``issue1345_framing/conversation_paired_stories_assistant_base/`` (the
     recovered round's own HF upload) instead of the instruct variant
     prefixes;
  2. both arms (context + prefix) are run, not context only;
  3. output ``store_pins`` metadata carries the base prefixes/revisions
     (the reference echoes the #1887 instruct pins);
  4. adds a ``stage`` sub-stage (``hub.stage_hub_file`` per shard, pretrained
     shards only, skip-existing resume) because the #1887 staging layout the
     reference consumed was swept from the data disk.
No fit / selection / rung numerics differ. Content hygiene: never prints
story or prompt text — structured numbers only.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# load_dotenv() BEFORE torch (via the reference import): torch freezes its
# intra-op thread pool from OMP_NUM_THREADS at import (#847).
load_dotenv()

import issue825_fit_cells as fit825  # noqa: E402
import issue1345_story_char_ladder_fill as fill  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

MODEL = "pretrained"
REPO = "superkaiba1/explore-persona-space-data"
# r1/r2 pin: the parent turnstore-reuse pin quoted in the #1345 body.
PARENT_REV = "2a3cb30acada04defc84fd04d28a2b54da3104cd"
# r4/r4op pin: dataset revision verified to hold the recovered round's stores.
BASE_ROUND_REV = "87b0c19e3db0d164cef2b3956661fc74587bc9bc"

_BASE_VARIANT_PREFIX = (
    "issue1345_framing/conversation_paired_stories_assistant_base/analysis_tensors/turnstore"
)

# Staged-subdir names mirror fill.REGIME_SPECS so load_regime_xy resolves them.
STORE_SOURCES: dict[str, dict] = {
    "parent_turnstore": {
        "prefix": "issue1345_framing/analysis_tensors/turnstore",
        "revision": PARENT_REV,
        "stems": ("pretrained_chat_s", "pretrained_naturalistic_s"),
    },
    "conversation_paired_stories_assistant_turnstore": {
        "prefix": _BASE_VARIANT_PREFIX,
        "revision": BASE_ROUND_REV,
        "stems": ("pretrained_stories_paired_s",),
    },
    "onpolicy_assistant_story_turnstore": {
        "prefix": _BASE_VARIANT_PREFIX,
        "revision": BASE_ROUND_REV,
        "stems": ("pretrained_stories_paired_op_s",),
    },
}

DEFAULT_PAIRS = ("r1:r4", "r1:r4op", "r2:r4")


def stage(stage_root: Path) -> None:
    """Download the pretrained shards into the flat staged layout (resume-safe)."""
    from huggingface_hub import HfApi

    api = HfApi()
    t_all = time.time()
    for subdir, spec in STORE_SOURCES.items():
        dest = stage_root / subdir
        dest.mkdir(parents=True, exist_ok=True)
        paths = hub.list_hf_files_under_path(
            api, REPO, spec["prefix"], repo_type="dataset", revision=spec["revision"]
        )
        wanted = sorted(
            p
            for p in paths
            if any(p.split("/")[-1].startswith(s + "_shard") for s in spec["stems"])
        )
        assert wanted, f"no shards under {spec['prefix']} for stems {spec['stems']}"
        for p in wanted:
            out = dest / p.split("/")[-1]
            if out.exists() and out.stat().st_size > 0:
                print(f"[stage] skip existing {subdir}/{out.name}", flush=True)
                continue
            t0 = time.time()
            hub.stage_hub_file(REPO, p, out, repo_type="dataset", revision=spec["revision"])
            print(f"[stage] {subdir}/{out.name} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[stage] all stores staged ({time.time() - t_all:.0f}s)", flush=True)


def run_ladders(
    stage_root: Path,
    cache_dir: Path,
    out_dir: Path,
    *,
    arms: list[str],
    pairs: list[tuple[str, str]],
    basis: str,
    null_draws: int,
    seed: int,
    layer: int,
) -> None:
    bases = ("reduced", "ambient") if basis == "both" else (basis,)
    for arm in arms:
        regimes = sorted({r for pr in pairs for r in pr})
        blocks = {
            r: fill.load_regime_xy(stage_root, cache_dir, MODEL, r, arm, layer) for r in regimes
        }
        results: dict = {"metadata": fill._metadata(seed, layer, arm), "ladders": {}}
        results["metadata"]["model"] = MODEL
        results["metadata"]["script"] = "scripts/issue1345_refit_base_story_standardized.py"
        results["metadata"]["reference_script"] = "scripts/issue1345_story_char_ladder_fill.py"
        results["metadata"]["rung_order"] = list(fill.RUNGS)
        results["metadata"]["regime_labels"] = fill.REGIME_LABEL
        # Base-round provenance (delta 3 in the module docstring): the
        # reference's store_pins() echoes the #1887 instruct pins.
        results["metadata"]["store_pins"] = {
            k: {"prefix": v["prefix"], "revision": v["revision"]} for k, v in STORE_SOURCES.items()
        }
        for a, b in pairs:
            t0 = time.time()
            xa, xb, keep = fill.matched_pair(blocks[a], blocks[b])
            xy = {a: xa, b: xb}
            folds = fit825._cv_folds(keep, fill.N_FOLDS, seed)
            key = f"{fill.REGIME_LABEL[a]}<->{fill.REGIME_LABEL[b]}"
            entry: dict = {
                "regimes": [a, b],
                "n_matched": int(len(keep)),
                "n_source_rows": int(blocks[a]["X"].shape[0]),
                "n_target_rows": int(blocks[b]["X"].shape[0]),
                "pairing": "conv-id intersection of the two full stores",
            }
            for bs in bases:
                entry[bs] = fill.run_pair(
                    xy, (a, b), folds, basis=bs, null_draws=null_draws, seed=seed
                )
                fill._print_pair(key, bs, entry[bs])
            entry["wall_s"] = round(time.time() - t0, 1)
            results["ladders"][key] = entry
            print(f"[pair] {arm}/{key} wall {entry['wall_s']:.0f}s", flush=True)
            del xy, xa, xb
        out = out_dir / f"ladders_{arm}.json"
        out.write_text(json.dumps(results, indent=2))
        print(f"[write] {out}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    default_stage = os.environ.get("EPM_I1345_BASE_REFIT_STAGE", "/workspace/i1345_base_refit")
    ap.add_argument("--stage-root", type=Path, default=Path(default_stage))
    ap.add_argument("--cache-dir", type=Path, default=None, help="default: <stage-root>/_l19")
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=_REPO_ROOT / "eval_results/issue_1345/recovered_base_story/refit_standardized",
    )
    ap.add_argument("--stages", nargs="+", default=["stage", "ladders"])
    ap.add_argument("--arms", nargs="+", default=["context", "prefix"])
    ap.add_argument("--pairs", nargs="+", default=list(DEFAULT_PAIRS))
    ap.add_argument("--basis", default="both", choices=("reduced", "ambient", "both"))
    ap.add_argument("--null-draws", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--layer", type=int, default=fill.HEADLINE_LAYER)
    args = ap.parse_args()
    cache_dir = args.cache_dir or args.stage_root / "_l19"
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)
    pairs = [tuple(p.split(":")) for p in args.pairs]
    for a, b in pairs:
        assert a in fill.REGIME_SPECS and b in fill.REGIME_SPECS, f"unknown pair {a}:{b}"

    t_all = time.time()
    if "stage" in args.stages:
        stage(args.stage_root)
    if "ladders" in args.stages:
        run_ladders(
            args.stage_root,
            cache_dir,
            args.out_dir,
            arms=args.arms,
            pairs=pairs,
            basis=args.basis,
            null_draws=args.null_draws,
            seed=args.seed,
            layer=args.layer,
        )
    print(f"TOTAL {time.time() - t_all:.0f}s", flush=True)


if __name__ == "__main__":
    main()
