#!/usr/bin/env python3
"""Reduced-rank regression (RRR) rank curves for the issue-2588 panel maps.

The mapping-rank script truncates the fitted ridge map W along W's own top
singular directions.  The best rank-k linear predictor instead keeps the top-k
principal directions of the FITTED training outputs X W (reduced-rank ridge,
Mukherjee & Zhu 2011; classical RRR for lambda -> 0).  With the fitted map
cached, that only needs the training activations of the selected layer: the
fitted-output covariance is W^T (X^T X) W, its eigenvectors give the nested
rank-k projections, and the held-out R^2 at every rank follows from the same
projection identity the truncated-ridge curve uses.

Writes eval_results/issue_2588/rrr_rank_curves.json: per map, validation and
test R^2 at every rank 0..d (exact, no randomized truncation) plus the
fitted-output eigenvalue spectrum.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


import issue2588_mapping_rank_vs_capability as MR

REPO = MR.REPO
DEFAULT_OUT = MR.EVAL_ROOT / "rrr_rank_curves.json"


def rrr_curves(spec: MR.MapSpec, cache_dir: Path) -> dict:
    """Shared implementation lives in issue2588_mapping_rank_vs_capability."""
    return MR.rrr_curves(spec, cache_dir)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=MR.DEFAULT_CACHE)
    ap.add_argument(
        "--results",
        type=Path,
        default=MR.DEFAULT_OUT,
        help="mapping_rank JSON; only its maps are run",
    )
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--maps", nargs="*", default=None)
    args = ap.parse_args()
    present = {m["key"] for m in json.loads(args.results.read_text(encoding="utf-8"))["maps"]}
    specs = [s for s in MR.MAPS if s.key in present]
    if args.maps:
        specs = [s for s in specs if s.key in set(args.maps) or s.cell in set(args.maps)]
    existing: dict[str, dict] = {}
    if args.out.exists():
        existing = {r["key"]: r for r in json.loads(args.out.read_text(encoding="utf-8"))["maps"]}
    results = []
    for spec in specs:
        prior = existing.get(spec.key)
        if prior is not None and prior.get("hf_revision") == MR.HF_REVISION and not args.maps:
            results.append(prior)
            continue
        results.append(rrr_curves(spec, args.cache_dir))
        merged = {**existing, **{r["key"]: r for r in results}}
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps(
                {"schema_version": "issue2588_rrr_rank_curves_v1", "maps": list(merged.values())},
                indent=1,
            )
            + "\n",
            encoding="utf-8",
        )
    order = {s.key: i for i, s in enumerate(MR.MAPS)}
    merged = {**existing, **{r["key"]: r for r in results}}
    out = sorted(merged.values(), key=lambda r: order.get(r["key"], 10**6))
    args.out.write_text(
        json.dumps({"schema_version": "issue2588_rrr_rank_curves_v1", "maps": out}, indent=1)
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {args.out} ({len(out)} maps)", flush=True)


if __name__ == "__main__":
    main()
