"""Phase-4 CLI for issue #1739: render the plan-§6 figure set from result JSONs.

Reads ``all_arms_spearman.json`` (+ optional map-diagnostics / composition
JSONs) and renders through ``experiments.issue_1739.figures`` (paper-plots
conventions; one color = one arm family across every figure). No network.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue1739_figures.py").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

logger = logging.getLogger("issue1739_figures")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--summary",
        type=Path,
        default=Path("eval_results/issue_1739/arm_results/all_arms_spearman.json"),
    )
    ap.add_argument(
        "--map-diag", type=Path, default=None, help="pooled per-rung map-diagnostic JSON"
    )
    ap.add_argument("--composition", type=Path, default=None, help="composition-factor rows JSON")
    ap.add_argument("--out-dir", type=Path, default=Path("figures/issue_1739"))
    return ap.parse_args(argv)


def _mean(vals: list) -> float:
    kept = [float(v) for v in vals if v is not None]
    return sum(kept) / len(kept) if kept else float("nan")


def map_diag_rows(payload) -> list[dict]:
    """Adapt the fits CLI's ``map_diagnostics.json`` to pooled per-rung rows.

    The fits phase writes ``{"<variant>|<u_label>": <fit_linear_map
    diagnostics>}``; ``fig_map_degradation`` consumes pre-pooled rows
    (``rung`` / ``r2_map`` / ``r2_identity_bias`` / ``knn_acc1_euclidean`` /
    ``knn_chance1``). A list payload is passed through unchanged (already
    pooled). Pooling = mean over layers; acc@1 keys survive a JSON roundtrip
    as strings.
    """
    if isinstance(payload, list):
        return payload
    rows: list[dict] = []
    for key in sorted(payload):
        per = (payload[key] or {}).get("per_layer") or []
        if not per:
            continue
        knn_e = [layer_row["knn"]["euclidean"] for layer_row in per]
        rows.append(
            {
                "rung": key,
                "r2_map": _mean([layer_row["r2_map"] for layer_row in per]),
                "r2_identity_bias": _mean([layer_row["r2_identity_bias"] for layer_row in per]),
                "knn_acc1_euclidean": _mean(
                    [e["acc_at_k"].get("1", e["acc_at_k"].get(1)) for e in knn_e]
                ),
                "knn_chance1": _mean(
                    [e["chance_at_k"].get("1", e["chance_at_k"].get(1)) for e in knn_e]
                ),
            }
        )
    return rows


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    args = _parse_args(argv)
    from explore_persona_space.experiments.issue_1739 import figures

    summary = json.loads(args.summary.read_text())
    paths = figures.render_summary_figures(summary, args.out_dir)
    if args.map_diag is not None:
        rows = map_diag_rows(json.loads(args.map_diag.read_text()))
        paths += list(figures.fig_map_degradation(rows, args.out_dir).values())
    if args.composition is not None:
        rows = json.loads(args.composition.read_text())
        paths += list(figures.fig_composition(rows, args.out_dir).values())
    for p in paths:
        print(f"[figures] wrote {p}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
