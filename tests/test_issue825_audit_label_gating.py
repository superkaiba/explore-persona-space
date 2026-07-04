"""Regression guard: audit floors BIND the headline labels (onpolicy-user-turn r2).

Pins the review-round-1 BLOCKER `audit-floors-not-binding-headline` fix:
`issue825_onpolicy_summarize.summarize` must compute `audit_pass` BEFORE the
headline labels, so an audit-failing cell (keep_rate < 0.80 or
distinct-3-gram < 0.5) with a mechanically inflated positive frozen R^2
(the `any_frozen_r2_positive` ridge lane) is labeled
"degenerate-provenance — observational", EXCLUDED from `support_summary`
(every lane), and kept in `parent_delta_table` flagged observational —
never a headline "supported" bar (plan v7 line 42 / hard-req 3).

Also pins the `parent-trcov-conditional-only` CONCERN fix: the parent
tr(cov) key is ALWAYS present — explicit null + a `parent_trcov_source`
recipe note when unavailable, never a silently absent key.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue825_onpolicy_summarize as summ  # noqa: E402

CID = "M_pretrained_user_chat"
OBSERVATIONAL = "degenerate-provenance — observational"


def _write_fixture(tmp_path: Path, *, keep_rate: float, d3g: float = 0.9) -> SimpleNamespace:
    """One user cell with a POSITIVE frozen ridge R^2 (ridge_supported fires
    via any_frozen_r2_positive) + a meta carrying the given audit stats."""
    out_dir = tmp_path / "out"
    onp = tmp_path / "onp"
    parent_cells = tmp_path / "parent_cells"
    parent_mlp = tmp_path / "parent_mlp"
    matched_parent = tmp_path / "matched_parent"
    for d in (out_dir, onp, parent_cells, parent_mlp, matched_parent):
        d.mkdir(parents=True, exist_ok=True)

    (out_dir / f"cells_{CID}.json").write_text(
        json.dumps(
            {
                "metadata": {"n": 200},
                "n_allowlist": 200,
                "y_trace_cov_frozen": {"19": 1.23, "26": 0.98},
                "selection_symmetric": {
                    "frozen_layer_table": {"19": {"r2_obs": 0.5}, "26": {"r2_obs": 0.4}},
                    "obs_layer_max_r2": 0.5,
                    "obs_argmax_layer": 19,
                    "null_layer_max_p975": 0.1,
                    "null_layer_max_r2_per_draw": [0.05, 0.06],
                },
                "mlp": {},
            }
        )
    )
    (onp / "conversations_pretrained_chat_meta.json").write_text(
        json.dumps(
            {
                "keep_rate": keep_rate,
                "distinct_3gram_rate_kept": d3g,
                "repetition_rate": 0.0,
                "role_artifact_rate": 0.0,
                "u2_length_kept": {"mean": 20.0, "sd": 4.0},
                "drops": {"short": 0, "overlength": 0},
                "parent_reference": {"distinct_3gram_rate": 0.781, "u2_length": {"mean": 25.0}},
            }
        )
    )
    return SimpleNamespace(
        out_dir=out_dir,
        onpolicy_dir=onp,
        parent_cells_dir=parent_cells,
        parent_mlp_dir=parent_mlp,
        matched_parent_dir=matched_parent,
    )


def test_audit_failing_cell_excluded_from_headline_support(tmp_path, monkeypatch):
    """keep_rate=0.1 + positive frozen R^2 => observational, NOT supported."""
    monkeypatch.delenv("EPS_SMOKE", raising=False)
    args = _write_fixture(tmp_path, keep_rate=0.1)
    headline = summ.summarize(args)

    cell = headline["cells"][CID]
    # the ridge lane DID fire mechanically — and must still be excluded
    assert cell["ridge_supported"] is True
    assert cell["cell_label_pre_audit"] == "supported"
    assert cell["cell_label"] == OBSERVATIONAL
    assert cell["audit"]["headline_eligible"] is False
    assert cell["audit"]["label_downgraded_to_observational"] is True
    # excluded from EVERY support lane
    assert headline["support_summary"]["supported"] == []
    assert headline["support_summary"]["suggestive"] == []
    assert headline["support_summary"]["provenance_sensitive_negative"] == []
    # kept in parent_delta_table, flagged observational (never dropped)
    row = headline["parent_delta_table"][CID]
    assert row["cell_label"] == OBSERVATIONAL
    assert row["headline_eligible"] is False


def test_audit_passing_cell_still_supported(tmp_path, monkeypatch):
    """The pass path is untouched: audit PASS + ridge lane => headline supported."""
    monkeypatch.delenv("EPS_SMOKE", raising=False)
    args = _write_fixture(tmp_path, keep_rate=0.95)
    headline = summ.summarize(args)

    cell = headline["cells"][CID]
    assert cell["cell_label"] == "supported"
    assert cell["audit"]["headline_eligible"] is True
    assert headline["support_summary"]["supported"] == [{"cell": CID, "lanes": ["ridge"]}]
    assert headline["parent_delta_table"][CID]["headline_eligible"] is True


def test_smoke_bypasses_floors_but_flags_it(tmp_path, monkeypatch):
    """EPS_SMOKE=1: floors bypassed (MF-D), headline_eligible=None, flagged."""
    monkeypatch.setenv("EPS_SMOKE", "1")
    args = _write_fixture(tmp_path, keep_rate=0.1)
    headline = summ.summarize(args)
    cell = headline["cells"][CID]
    assert headline["smoke"] is True
    assert cell["cell_label"] == "supported"  # pre-audit label kept under smoke
    assert cell["audit"]["headline_eligible"] is None
    assert cell["audit"]["floors_bypassed_smoke"] is True


def test_eps_smoke_zero_is_production(tmp_path, monkeypatch):
    """EPS_SMOKE=0 must be PRODUCTION (strict '1' compare — review-r1 Minor)."""
    monkeypatch.setenv("EPS_SMOKE", "0")
    args = _write_fixture(tmp_path, keep_rate=0.1)
    headline = summ.summarize(args)
    assert headline["smoke"] is False
    assert headline["cells"][CID]["cell_label"] == OBSERVATIONAL


def test_parent_trcov_key_always_present_with_source_note(tmp_path, monkeypatch):
    """No matched-parent refit + no parent key => explicit null + recipe note."""
    monkeypatch.delenv("EPS_SMOKE", raising=False)
    args = _write_fixture(tmp_path, keep_rate=0.95)
    headline = summ.summarize(args)
    trcov = headline["cells"][CID]["diversity_tr_cov"]
    assert "parent_y_trace_cov_frozen" in trcov
    assert trcov["parent_y_trace_cov_frozen"] is None
    assert "post-hoc recipe" in trcov["parent_trcov_source"]
    assert "deb7a452" in trcov["parent_trcov_source"]
