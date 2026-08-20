#!/usr/bin/env python3
"""#1739 claim4-controls: re-render the fold figures with reader-facing labels.

Clean-result-critic revision round: figures must not expose rung/protocol
slugs. Re-uses the fold module's own renderer (`render_figures`) against the
COMMITTED `claim4_per_rung_table.json` — the numbers are byte-identical to
the 12cfdbf31d render; only labels change — and adds two low-level
companions: the two-series per-seed figure (`claim4_per_seed`, superseding
the true-map-only `claim4_spaghetti` draft) and the per-context scatter
behind the corrected-sycophancy correlations (`claim4_syco_percontext`).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path


def _load_fold_module():
    path = Path(__file__).resolve().parent / "issue1739_claim4_fold.py"
    spec = importlib.util.spec_from_file_location("issue1739_claim4_fold", path)
    mod = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--table",
        default="eval_results/issue_1739/claim4_controls/claim4_per_rung_table.json",
    )
    ap.add_argument("--fig-dir", default="figures/issue_1739/claim4_controls")
    ap.add_argument(
        "--preds-root",
        default=None,
        help="claim4 preds mirror root (default: the table's meta.claim4_root)",
    )
    args = ap.parse_args()

    mod = _load_fold_module()
    table = json.loads(Path(args.table).read_text())
    fig_dir = Path(args.fig_dir)
    seeds = table["meta"]["seeds"]
    written = mod.render_figures(table, fig_dir, seeds)
    preds_root = Path(args.preds_root or table["meta"]["claim4_root"])
    written.append(mod.render_syco_percontext(preds_root, fig_dir))
    print(f"written: {written}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
