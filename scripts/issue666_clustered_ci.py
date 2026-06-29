#!/usr/bin/env python
# ruff: noqa: RUF002, RUF003
# Intentional scientific Unicode (ρ, ×) in docstrings/comments.
"""issue #666 Phase 4 — family-clustered + probe-split bootstrap CIs (plan §6 C4, §6.5).

The 50 battery contexts cluster by FAMILY (7 families) and share the 48-probe pool,
so n=50 has far fewer effective d.o.f. than naive. The headline CIs resample
FAMILIES (then the contexts within each drawn family), NEVER the 50 contexts
independently (plan §6: "resample at cluster level, NEVER naive n=50"). The
clustered + naive estimators live in ``issue666_predictor`` (``clustered_bootstrap_ci``
/ ``naive_bootstrap_ci`` / ``draw_families``); this driver reads the per-cell
predictor JSONs and emits per-cell clustered ρ CIs (+ the naive comparator).

Output: ``eval_results/issue_666/clustered_ci.json``.

CPU-only; reuses the #532/#545 bootstrap pattern.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "eval_results" / "issue_666"


def per_cell_clustered_ci(pred_dir: Path, *, n_boot=2000, seed=0) -> dict:
    """Per-cell clustered + naive Spearman-ρ CIs over the predictor JSONs."""
    import issue666_predictor as pred

    out: dict = {}
    for p in sorted(pred_dir.glob("*_predictor_cells.json")):
        rec = json.loads(p.read_text())
        pb = rec["per_bystander"]
        lh = np.array(pb["Lhat"])
        ds = np.array(pb["ds"])
        fams = np.array(pb["context_family"])
        if lh.size < 4:
            continue
        clo, chi = pred.clustered_bootstrap_ci(
            lh, ds, clusters=fams, n_boot=n_boot, seed=seed, statistic="spearman"
        )
        nlo, nhi = pred.naive_bootstrap_ci(lh, ds, n_boot=n_boot, seed=seed, statistic="spearman")
        out[rec["cell"]] = {
            "rho_full_Lhat": rec["rho_full_Lhat"],
            "clustered_ci": [float(clo), float(chi)],
            "naive_ci": [float(nlo), float(nhi)],
            "clustered_width": float(chi - clo),
            "naive_width": float(nhi - nlo),
            "n_bystanders": rec["n_bystanders"],
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="issue 666 clustered + naive bootstrap CIs.")
    ap.add_argument("--pred-dir", default=str(REPO / "eval_results" / "issue_666" / "predictor"))
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--slice", action="store_true", help="tiny smoke slice (fewer bootstraps)")
    args = ap.parse_args()

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    n_boot = 200 if args.slice else args.n_boot
    rec = per_cell_clustered_ci(Path(args.pred_dir), n_boot=n_boot)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "clustered_ci.json").write_text(
        json.dumps({"per_cell": rec, "n_boot": n_boot}, indent=1)
    )
    print(f"[clustered_ci] {len(rec)} cells -> {OUT / 'clustered_ci.json'}")
    print("[phase=clustered_ci] done OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
