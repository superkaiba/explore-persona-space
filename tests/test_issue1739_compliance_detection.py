"""Pins for the #1739 compliance-wave detection circularity fix.

The 2026-08-05 full compliance wave emitted ``detection.auroc == 1.0`` on all
three evil rungs because the positive-class LABEL was derived by thresholding
the SAME per-item mean compliance score that was then used as the PREDICTION —
AUROC of a score against a label defined by a monotone threshold on that score
is 1.0 by construction (X-vs-f(X) circularity; the #383 family). The three
emitted blocks are struck in
``eval_results/issue_1739/evil_ood_spread/compliance_dv_results.json``.

Each test here FAILS on the pre-fix code (commit ``086d2588fe``):

1. the wave reduce path emits a DEFERRAL record, never AUROC/AP (pre-fix:
   ``detection.auroc == 1.0`` and the old signature required a
   ``detection_threshold`` kwarg),
2. the guarded hook RAISES on circular self-scoring (pre-fix: the single
   per-item-dict signature accepted its own score vector and returned 1.0),
3. the hook still computes an honest AUROC for an INDEPENDENT prediction
   vector (an imperfect predictor scores strictly < 1.0 — the metric is not
   degenerately pinned either way).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = REPO_ROOT / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from issue1739_compliance_full import (  # noqa: E402
    DETECTION_DEFERRED,
    _detection_metrics,
    _reduce_rung_report,
)

# Ten items with single-draw scores straddling the 50.0 label threshold
# (5 below, 5 at-or-above). Neutral numeric fixtures only — no corpus text.
# Ids follow the production ``{context_id}_k{NN}`` grammar (round 22: the
# per-rung report groups items per context via ``dv_build.parse_item_id``,
# which fails loud on a non-conformant id); one item per context here.
PER_ITEM: dict[str, list[float]] = {
    f"item{i:02d}_k00": [float(v)]
    for i, v in enumerate([2.0, 8.0, 15.0, 30.0, 45.0, 55.0, 70.0, 85.0, 92.0, 98.0])
}


def test_wave_reduce_emits_deferral_never_auroc(tmp_path: Path) -> None:
    """The wave's per-rung ``detection`` slot is the deferral record.

    Pre-fix this fails twice over: ``_reduce_rung_report`` required a
    ``detection_threshold`` kwarg, and the emitted block carried
    ``auroc == 1.0`` (the tautology).
    """
    judged = {
        "per_item_scores": PER_ITEM,
        "n_items": len(PER_ITEM),
        "n_dropped_draws_content": 0,
        "n_transport_lost_draws": 0,
        "n_total_draws": len(PER_ITEM),
        "judge_raw_path": None,
    }
    rep = _reduce_rung_report(judged, trait_root=tmp_path, rung="stub_rung")
    det = rep["detection"]
    assert det.get("deferred") is True
    assert "auroc" not in det and "ap" not in det
    assert det == DETECTION_DEFERRED
    # The valid wave fields are untouched by the detection change.
    assert rep["n_scored"] == len(PER_ITEM)
    assert rep["spread_gate_pass"] is True


def test_detection_hook_raises_on_circular_self_scoring() -> None:
    """Feeding the label source's own means as the prediction vector refuses.

    Pre-fix: ``_detection_metrics(per_item_scores, threshold=...)`` accepted
    a single dict and returned ``auroc == 1.0`` for exactly this input shape.
    """
    means = {k: v[0] for k, v in PER_ITEM.items()}
    with pytest.raises(ValueError, match="circular"):
        _detection_metrics(means, PER_ITEM, threshold=50.0)


def test_detection_hook_scores_independent_predictions_honestly() -> None:
    """An imperfect INDEPENDENT predictor gets a real, sub-1.0 AUROC.

    One positive (DV mean 55) is predicted below one negative (DV mean 45):
    24 of 25 pos/neg pairs ordered correctly => AUROC 0.96 exactly.
    """
    preds = {k: v[0] for k, v in PER_ITEM.items()}
    # Swap the two boundary items' predictions across the label threshold.
    preds["item04_k00"], preds["item05_k00"] = 60.0, 40.0
    result = _detection_metrics(preds, PER_ITEM, threshold=50.0)
    assert result["n_pos"] == 5 and result["n_neg"] == 5
    assert result["auroc"] == pytest.approx(24.0 / 25.0)
    assert result["auroc"] < 1.0
    assert 0.0 < result["ap"] <= 1.0
