#!/usr/bin/env python
"""Issue #1336 — Phase P_v3: the pooled off-policy 2x2 panel decision writer.

Emits ``eval_results/issue_1336/decision_v3/pooled_offpolicy_2x2.json`` — the
plan v15 §6.5 primary deliverable — by AGGREGATING the Phase LAD_pool pair
files (``metric_ladder_pooled_v3/pair_<i>__<j>_arm_<a>.json``, written by
``issue1336_metric_ladder.py --pooled-pair``). No statistics are recomputed
here (the C-ii no-recompute contract): every ``r2`` / ``r2_bootstrap`` block
is copied verbatim from the pair files at the headline layer.

The output schema is pinned by the CONSUMER —
``scripts/issue1336_figures.py::fig_v3_pooled_2x2``'s docstring (authoritative
per Unit C-ii; no writer existed at the C-ii pin)::

    {"headline_layer": int, "scale": "raw",
     "arms": ["on", "off"], "corpus_order": [...],
     "pairs": {"<i>__<j>": {"source": i, "target": j, "arms": {"<arm>": {
       "per_corpus": {"<corpus>": {"own"|"t0"|"t6"|"t8": {
         "r2": float, "r2_bootstrap": {"ci_lo": float, "ci_hi": float}}}}}}}}}

Diagonal ``<k>__<k>`` entries (own-only — the figure's fallback for targets
with no transfer pair, i.e. the base stage) are synthesized from the pair
files' ``own_source`` blocks: for stage k, any pair file with ``source == k``
carries checkpoint k's own pooled-map read on the same test rows.

Headline layer: the pooled stage-symmetric rule (C-i's
``_headline_layer_pooled``, reading ``cells_pooled_v3/cells_pooled_<k>_arm_on
.json``), with ``--headline-layer`` as the dispatcher/smoke override seam.

Scale: v1 emits the RAW scale only — the LAD_pool pair files carry raw reads
(pooled n_train >> d, the §7 G1' grounds); ``--primary-scale recal`` is
refused rather than silently mislabeled.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

import issue1336_metric_ladder as ml  # noqa: E402

from explore_persona_space.experiments.issue_1336 import common as cm  # noqa: E402

PANEL_TIERS = ("own", "t0", "t6", "t8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1336"))
    ap.add_argument(
        "--headline-layer",
        type=int,
        default=None,
        help="override the pooled stage-symmetric headline-layer rule (dispatcher/smoke seam)",
    )
    ap.add_argument("--frozen-layers", default=None, help="comma ints (default: registry set)")
    ap.add_argument(
        "--primary-scale",
        choices=("raw", "recal"),
        default="raw",
        help="panel scale marker; v1 pair files carry raw only — recal is refused",
    )
    ap.add_argument("--smoke", action="store_true")
    return ap.parse_args()


def _layer_block(pair_json: dict, li: int, path: Path) -> tuple[dict, int]:
    """The pair JSON's layer block at the headline layer (fallback: deepest
    available, with a printed note — smoke ladders run a reduced layer set)."""
    layers = pair_json["layers"]
    if str(li) in layers:
        return layers[str(li)], li
    fallback = max(int(k) for k in layers)
    print(f"[decision1336-v3] {path.name}: layer {li} absent — falling back to {fallback}")
    return layers[str(fallback)], fallback


def _copy_block(blk: dict) -> dict:
    """Verbatim {r2, r2_bootstrap} copy (no recomputation in this layer)."""
    return {"r2": float(blk["r2"]), "r2_bootstrap": dict(blk["r2_bootstrap"])}


def build_panel(args: argparse.Namespace) -> dict:
    """Aggregate the LAD_pool pair files into the fig_v3_pooled_2x2 schema."""
    assert args.primary_scale == "raw", (
        "the LAD_pool pair files carry raw-scale reads only (plan §7 G1' grounds); "
        "a recal panel needs a recal pooled-ladder read first"
    )
    ladder_dir = args.out_dir / "metric_ladder_pooled_v3"
    files = sorted(ladder_dir.glob("pair_*_arm_*.json"))
    assert files, (
        f"no pair files under {ladder_dir} — run Phase LAD_pool "
        "(issue1336_metric_ladder.py --pooled-pair) first"
    )
    if args.frozen_layers:
        frozen = tuple(int(x) for x in args.frozen_layers.split(",") if x.strip())
    else:
        frozen = cm.SMOKE_FROZEN_LAYERS if args.smoke else cm.FROZEN_LAYERS
    head = ml._headline_layer_pooled(args.out_dir / "cells_pooled_v3", frozen, args.headline_layer)
    li = int(head["headline_layer"])
    pairs: dict[str, dict] = {}
    arms_present: set[str] = set()
    corpora_present: set[str] = set()
    realized_layers: set[int] = set()
    for path in files:
        d = json.loads(path.read_text())
        src, tgt, arm = d["pair"]["source"], d["pair"]["target"], d["arm"]
        assert arm in ("on", "off"), (path.name, arm)
        block, used_li = _layer_block(d, li, path)
        realized_layers.add(used_li)
        entry = pairs.setdefault(f"{src}__{tgt}", {"source": src, "target": tgt, "arms": {}})
        assert arm not in entry["arms"], f"duplicate (pair, arm) file for {path.name}"
        per_corpus = {}
        for corpus, cblk in block["per_corpus"].items():
            per_corpus[corpus] = {t: _copy_block(cblk[t]) for t in PANEL_TIERS}
            corpora_present.add(corpus)
        entry["arms"][arm] = {"per_corpus": per_corpus}
        arms_present.add(arm)
        # Diagonal own-only entry for the SOURCE stage (the figure's fallback
        # read at targets with no transfer pair — i.e. base). own_source is
        # checkpoint src's own pooled-map read; identical across pair files
        # sharing (src, arm), so first-write wins.
        diag = pairs.setdefault(f"{src}__{src}", {"source": src, "target": src, "arms": {}})
        diag_arm = diag["arms"].setdefault(arm, {"per_corpus": {}})
        for corpus, cblk in block["per_corpus"].items():
            diag_arm["per_corpus"].setdefault(corpus, {"own": _copy_block(cblk["own_source"])})
    assert len(realized_layers) == 1, (
        f"pair files resolve to MIXED layers {sorted(realized_layers)} — re-run LAD_pool "
        "at one headline layer (or pass --headline-layer matching the files)"
    )
    realized = realized_layers.pop()
    corpus_order = [c for c in cm.V2_CORPORA if c in corpora_present]
    corpus_order += sorted(corpora_present - set(corpus_order))
    return {
        "metadata": ml._metadata(cm.FIT_SEED, len(files)),
        "headline": head,
        "headline_layer": realized,
        "scale": "raw",
        "arms": [a for a in ("on", "off") if a in arms_present],
        "corpus_order": corpus_order,
        "pairs": pairs,
        "source_files": [p.name for p in files],
        "generated_unix": time.time(),
    }


def main() -> None:
    args = parse_args()
    panel = build_panel(args)
    out = args.out_dir / "decision_v3" / "pooled_offpolicy_2x2.json"
    ml._write_json(out, panel)
    n_pairs = len(panel["pairs"])
    print(
        f"[decision1336-v3] wrote {out} (layer {panel['headline_layer']}, "
        f"arms {panel['arms']}, {n_pairs} pair entries, "
        f"{len(panel['corpus_order'])} corpora)"
    )


if __name__ == "__main__":
    main()
