"""#2202 P5 stats pins: the exploratory Fable-mode BH family EXCLUDES κ-demoted
(report-only) modes before ``_bh_fdr`` runs — demoted rows keep their rates in
the output JSON but never carry a ``bh_significant`` key (round-2 review fix).
All synthetic; no network, no artifacts read."""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue2202_stats_figs as SF  # noqa: E402

N_DRAWS = 200  # smoke-scale draws — the family logic is draw-count independent


def _fixture(demoted: list[str]) -> tuple[dict, dict]:
    """Two-mode judge doc + population: mode_sig separates fail vs control
    perfectly; mode_noise is flat. Labels keyed f{ci}/c{ci} per the driver."""
    fail_eq = list(range(8))
    ctrl = list(range(100, 108))
    modes = [
        {"name": "mode_sig", "description": "d", "decision_rule": "r"},
        {"name": "mode_noise", "description": "d", "decision_rule": "r"},
    ]
    labels: dict[str, dict] = {}
    for c in fail_eq:
        labels[f"f{c}"] = {"mode_sig": "yes", "mode_noise": "yes" if c % 2 else "no"}
    for c in ctrl:
        labels[f"c{c}"] = {"mode_sig": "no", "mode_noise": "yes" if c % 2 else "no"}
    jd = {"modes": modes, "demoted_modes": demoted, "labels": labels}
    pop = {
        "fail_eq_cis": fail_eq,
        "control_cis": ctrl,
        "fail_cis": fail_eq,
        "digest1_cis": [0, 1],
    }
    return jd, pop


def test_demoted_modes_excluded_from_bh_family():
    jd, pop = _fixture(demoted=["mode_noise"])
    blk = SF.fable_mode_family(jd, pop, n_boot=N_DRAWS, n_perm=N_DRAWS)
    assert blk["available"] and blk["n_modes_kept"] == 1
    assert blk["n_modes_demoted_report_only"] == 1
    demoted_row = blk["per_mode"]["mode_noise"]
    kept_row = blk["per_mode"]["mode_sig"]
    # the reviewer's mechanized assert: NO demoted_report_only row carries bh_significant
    for row in blk["per_mode"].values():
        if row["demoted_report_only"]:
            assert "bh_significant" not in row
    # demoted rates stay report-only in the output JSON
    for key in ("rate_fail_eq", "rate_control", "rate_fail_full", "p_perm"):
        assert key in demoted_row
    # the kept mode still gets a BH verdict, over the KEPT-only family (m=1)
    assert isinstance(kept_row["bh_significant"], bool)
    assert kept_row["bh_significant"]  # perfect separation at n=8v8 survives BH at m=1


def test_all_modes_demoted_yields_no_bh_calls():
    jd, pop = _fixture(demoted=["mode_sig", "mode_noise"])
    blk = SF.fable_mode_family(jd, pop, n_boot=N_DRAWS, n_perm=N_DRAWS)
    assert blk["n_modes_kept"] == 0
    assert all("bh_significant" not in row for row in blk["per_mode"].values())
    assert all(row["demoted_report_only"] for row in blk["per_mode"].values())


def test_no_demotion_keeps_full_family():
    jd, pop = _fixture(demoted=[])
    blk = SF.fable_mode_family(jd, pop, n_boot=N_DRAWS, n_perm=N_DRAWS)
    assert blk["n_modes_kept"] == 2 and blk["n_modes_demoted_report_only"] == 0
    assert all(isinstance(row["bh_significant"], bool) for row in blk["per_mode"].values())
