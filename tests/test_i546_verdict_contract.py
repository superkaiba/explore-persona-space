"""#546 verdict-contract unit-exercise (plan §3 item (d) + §15).

Pins the cn_i546 analyzer extensions in ``scripts/i464_po_analyze.py``:

1. Two-sided per-persona verdict (plan §2 divergence 2): the synthetic
   payload cases the plan names — all-negative → ``h1_neg``;
   all-straddle → ``h0``; mixed-sign → ``mixed_bidirectional`` (with
   driving cells persisted); all-positive → ``h1_pos``.
2. cn_i533 byte-compat: one-sided mode (the default) still emits the
   inherited {h1, h0, inconclusive} verdict set — an all-negative
   payload stays ``inconclusive`` there, never ``h1_neg``.
3. Partial-anchor resolved-persona path (plan §2 divergence 3),
   end-to-end through ``main()`` on synthetic per-cell JSONs: one
   resolved persona → verdict over that persona's 2 cells, the
   unresolved persona persisted ``null`` with ``skipped: true``, payload
   carries ``partial_anchor`` + ``anchored_personas``.
4. Zero resolved personas → ``headline_status=partial_anchor_skipped``
   stub (the true degenerate case), with ``anchored_personas: []`` and
   ``partial_anchor`` DERIVED from the anchor payload (a fully-
   degenerate anchor is not partial).
5. Round-2 regression (code-review blocker `dynamic-range-suppresses-
   i546-verdict`): a resolved anchor with REALISTIC ~0.3-nat per-arm
   cell-mean seed spreads (the real #533 villain E=1 shape: per-arm
   sds 0.373/0.425/0.369, all below the 0.5 legacy threshold) FAILS
   the inherited persona-averaged dynamic-range gate, yet the cn_i546
   per-persona verdict MUST still fire — the legacy gate is recorded
   as a failed diagnostic, never a verdict suppressor.
6. Round-2 stub guarantee: the unresolved persona's null/skipped stub
   is written even when the per-persona block is skipped entirely
   (``inconclusive_descriptive_only`` — too few complete seeds).

CPU-only; no GPU, no network, no HF.
"""

from __future__ import annotations

import json

import pytest

from scripts import i464_po_analyze as poa

SEEDS = (42, 137, 1337, 7, 21)
ARMS = ("system_plain", "system_padded", "role")


def _cell(mean: float, lo: float, hi: float) -> dict:
    return {
        "mean": mean,
        "ci_lo_95": lo,
        "ci_hi_95": hi,
        "sign_agreement": 1.0,
        "d_per_seed": [mean] * 5,
        "n_seeds": 5,
        "n_bootstrap": 10,
    }


def _payload(pirate: tuple, villain: tuple) -> dict:
    """Build a per_persona payload; each arg = (plain_cell, padded_cell)."""
    return {
        "pirate": {"plain": pirate[0], "padded": pirate[1]},
        "villain": {"plain": villain[0], "padded": villain[1]},
    }


# ── 1. Two-sided verdict on the plan §15 synthetic payloads ─────────────


def test_two_sided_all_negative_is_h1_neg():
    neg = _cell(-1.0, -1.4, -0.6)
    out = poa._headline_verdict_from_per_persona(_payload((neg, neg), (neg, neg)), two_sided=True)
    assert out["verdict"] == "h1_neg"
    assert out["h1_neg"] is True and out["h1_pos"] is False
    assert set(out["h1_neg_cells"]) == {
        "pirate.plain",
        "pirate.padded",
        "villain.plain",
        "villain.padded",
    }


def test_two_sided_all_straddle_is_h0():
    straddle = _cell(0.1, -0.3, 0.5)
    out = poa._headline_verdict_from_per_persona(
        _payload((straddle, straddle), (straddle, straddle)), two_sided=True
    )
    assert out["verdict"] == "h0"
    assert out["h0"] is True and not out["h1_pos"] and not out["h1_neg"]


def test_two_sided_mixed_sign_is_mixed_bidirectional():
    pos = _cell(0.9, 0.4, 1.4)
    neg = _cell(-0.8, -1.2, -0.4)
    straddle = _cell(0.0, -0.2, 0.2)
    out = poa._headline_verdict_from_per_persona(
        _payload((pos, straddle), (neg, straddle)), two_sided=True
    )
    assert out["verdict"] == "mixed_bidirectional"
    assert out["h1_pos_cells"] == ["pirate.plain"]
    assert out["h1_neg_cells"] == ["villain.plain"]


def test_two_sided_all_positive_is_h1_pos():
    pos = _cell(0.9, 0.4, 1.4)
    out = poa._headline_verdict_from_per_persona(_payload((pos, pos), (pos, pos)), two_sided=True)
    assert out["verdict"] == "h1_pos"


def test_two_sided_subthreshold_negative_is_inconclusive():
    # CI clears zero on the negative side but |mean| < 0.5 nat, alongside
    # a non-straddling cell — fits neither H1- nor H0.
    weak_neg = _cell(-0.3, -0.45, -0.15)
    out = poa._headline_verdict_from_per_persona(
        _payload((weak_neg, weak_neg), (weak_neg, weak_neg)), two_sided=True
    )
    assert out["verdict"] == "inconclusive"


def test_two_sided_partial_two_cell_payload():
    # One-persona partial anchor: verdict computed over villain's 2 cells
    # only (the pirate key may be absent entirely).
    neg = _cell(-1.0, -1.4, -0.6)
    per_persona = {"villain": {"plain": neg, "padded": neg}}
    out = poa._headline_verdict_from_per_persona(per_persona, two_sided=True, personas=("villain",))
    assert out["verdict"] == "h1_neg"
    assert out["h1_neg_cells"] == ["villain.plain", "villain.padded"]


# ── 2. cn_i533 one-sided byte-compat ─────────────────────────────────────


def test_one_sided_all_negative_stays_inconclusive():
    neg = _cell(-1.0, -1.4, -0.6)
    out = poa._headline_verdict_from_per_persona(_payload((neg, neg), (neg, neg)))
    assert out["verdict"] == "inconclusive"  # NOT h1_neg — inherited contract


def test_one_sided_positive_is_h1_and_straddle_is_h0():
    pos = _cell(0.9, 0.4, 1.4)
    straddle = _cell(0.0, -0.2, 0.2)
    out_pos = poa._headline_verdict_from_per_persona(
        _payload((pos, straddle), (straddle, straddle))
    )
    assert out_pos["verdict"] == "h1"
    out_null = poa._headline_verdict_from_per_persona(
        _payload((straddle, straddle), (straddle, straddle))
    )
    assert out_null["verdict"] == "h0"


# ── 3+4. main() end-to-end on synthetic per-cell JSONs ───────────────────


@pytest.fixture()
def _reset_active():
    """Reset the module-level _ACTIVE dict around main()-level tests."""
    saved = dict(poa._ACTIVE)
    yield
    poa._ACTIVE.clear()
    poa._ACTIVE.update(saved)


def _wrong_enc(arm: str, persona: str) -> str:
    other = "villain" if persona == "pirate" else "pirate"
    return f"role_{other}" if arm == "role" else f"system_{other}"


def _own_enc(arm: str, persona: str) -> str:
    return f"role_{persona}" if arm == "role" else f"system_{persona}"


def _write_cell(per_cell_dir, arm, seed, persona, epoch, e_eval, logp) -> None:
    label = f"{arm}_seed{seed}_cn_{persona}_e{epoch}"
    payload = {"g_logprob": logp, "b_logprob": -21.0, "delta_g": logp + 21.0}
    (per_cell_dir / f"{label}__{e_eval}.json").write_text(json.dumps(payload))


def _build_villain_e1_cells(per_cell_dir, role_by_seed: dict[int, float] | None = None) -> None:
    """Synthetic villain-only E=1 grid: d = system - role is -1.0 (plain)
    / -1.2 (padded) per seed → drives h1_neg; own-slot at -0.001 ≥ -1.0
    (H1 elicitation gate passes).

    Default ``role_by_seed`` spreads the seeds 4 nats so the per-arm
    wrong-slot sd across seeds ≈ 1.41 > 0.5 and the LEGACY dynamic-range
    gate passes — this pins the gate-passing shape. Pass a tighter
    ``role_by_seed`` (see ``_REALISTIC_ROLE_BY_SEED``) to exercise the
    gate-FAILING shape the round-2 regression test covers.
    """
    if role_by_seed is None:
        role_by_seed = {42: -5.0, 137: -6.0, 1337: -7.0, 7: -8.0, 21: -9.0}
    for seed in SEEDS:
        role_lp = role_by_seed[seed]
        wrong_for_arm = {
            "role": role_lp,
            "system_plain": role_lp - 1.0,
            "system_padded": role_lp - 1.2,
        }
        for arm in ARMS:
            _write_cell(
                per_cell_dir,
                arm,
                seed,
                "villain",
                1,
                _wrong_enc(arm, "villain"),
                wrong_for_arm[arm],
            )
            _write_cell(per_cell_dir, arm, seed, "villain", 1, _own_enc(arm, "villain"), -0.001)
            _write_cell(per_cell_dir, arm, seed, "villain", 1, "default_assistant", -12.0)


# Realistic per-seed spread (round-2 regression): pstdev of these five
# role means is ≈ 0.31 nat — the real #533 villain E=1 shape (per-arm
# cell-mean sds 0.373/0.425/0.369), ALL below the 0.5 legacy
# dynamic-range threshold. Seed-consistent training with genuine anchor
# resolution looks exactly like this: the anchor selector's PER-QUESTION
# wrong_sd gate (~250 observations) passes while the legacy CELL-MEAN
# gate (5 values, question variance averaged away) fails.
_REALISTIC_ROLE_BY_SEED = {42: -7.0, 137: -7.3, 1337: -6.6, 7: -7.5, 21: -6.9}


def test_main_partial_anchor_resolved_persona_path(tmp_path, monkeypatch, _reset_active):
    per_cell_dir = tmp_path / "per_cell"
    per_cell_dir.mkdir()
    out_path = tmp_path / "analysis.json"
    monkeypatch.setitem(poa.PER_CELL_DIR_FOR, "cn_i546", per_cell_dir)
    monkeypatch.setitem(poa.OUT_PATH_FOR, "cn_i546", out_path)
    _build_villain_e1_cells(per_cell_dir)
    anchor = tmp_path / "anchor_selection.json"
    anchor.write_text(
        json.dumps(
            {
                "selected_anchor": {"pirate": None, "villain": 1},
                "degenerate": False,
                "partial_anchor": True,
                "partial_anchor_reason": "pirate never banded",
            }
        )
    )
    poa.main(["--variant", "cn_i546", "--anchor-file", str(anchor)])
    payload = json.loads(out_path.read_text())
    assert payload["schema_version"] == "i546_cn_analyze_v1"
    assert payload["anchored_personas"] == ["villain"]
    assert payload["partial_anchor"] is True
    assert payload["headline_status"] == "ok"
    head = payload["headline"]
    assert head["per_persona_verdict"] == "h1_neg"
    assert head["h1_neg_per_persona_pass"] is True
    assert head["h1_pos_per_persona_pass"] is False
    assert set(head["h1_neg_cells"]) == {"villain.plain", "villain.padded"}
    # Unresolved persona persisted null with skipped: true (plan §2 div 3).
    assert head["per_persona"]["pirate"] == {"plain": None, "padded": None, "skipped": True}
    # Resolved persona's 2 cells carry the full stats block.
    assert head["per_persona"]["villain"]["plain"]["mean"] == pytest.approx(-1.0)
    assert head["per_persona"]["villain"]["padded"]["mean"] == pytest.approx(-1.2)


def test_main_resolved_anchor_fires_verdict_despite_legacy_gate_failure(
    tmp_path, monkeypatch, _reset_active
):
    """Round-2 regression for `dynamic-range-suppresses-i546-verdict`.

    A resolved one-persona anchor with realistic ~0.3-nat per-arm
    cell-mean seed spreads (the real #533 villain E=1 shape) FAILS the
    inherited persona-averaged dynamic-range gate. Pre-fix, line ~991's
    status check suppressed the entire cn_i546 per-persona block and
    `headline_status` read `inconclusive_dynamic_range_failed`. Post-fix
    the verdict MUST fire (h1_neg here) with the legacy gate persisted
    as a failed DIAGNOSTIC, not a suppressor.
    """
    per_cell_dir = tmp_path / "per_cell"
    per_cell_dir.mkdir()
    out_path = tmp_path / "analysis.json"
    monkeypatch.setitem(poa.PER_CELL_DIR_FOR, "cn_i546", per_cell_dir)
    monkeypatch.setitem(poa.OUT_PATH_FOR, "cn_i546", out_path)
    _build_villain_e1_cells(per_cell_dir, role_by_seed=_REALISTIC_ROLE_BY_SEED)
    anchor = tmp_path / "anchor_selection.json"
    anchor.write_text(
        json.dumps(
            {
                "selected_anchor": {"pirate": None, "villain": 1},
                "degenerate": False,
                "partial_anchor": True,
                "partial_anchor_reason": "pirate never banded",
            }
        )
    )
    poa.main(["--variant", "cn_i546", "--anchor-file", str(anchor)])
    payload = json.loads(out_path.read_text())
    # The legacy gate genuinely FAILED on this shape ...
    dr = payload["dynamic_range_gate"]
    assert dr["ok"] is False
    assert all(v["above_threshold"] is False for v in dr["per_arm"].values())
    # ... but it is demoted to a diagnostic for cn_i546 ...
    assert dr["gates_verdict"] is False
    assert "dynamic_range_failed_arms" not in payload["headline"]
    assert payload["headline_status"] != "inconclusive_dynamic_range_failed"
    # ... and the Goal-bearing per-persona verdict fires.
    head = payload["headline"]
    assert head["per_persona_verdict"] == "h1_neg"
    assert payload["headline_status"] == "ok"
    assert set(head["h1_neg_cells"]) == {"villain.plain", "villain.padded"}
    assert head["per_persona"]["villain"]["plain"]["mean"] == pytest.approx(-1.0)
    assert head["per_persona"]["villain"]["padded"]["mean"] == pytest.approx(-1.2)
    # Unresolved persona's stub still present alongside the verdict.
    assert head["per_persona"]["pirate"] == {"plain": None, "padded": None, "skipped": True}
    # Synthetic anchor stub carries no per_persona_per_E_diagnostics.
    assert payload["anchor_resolution_gate"] is None


def test_main_descriptive_only_still_writes_unresolved_stub(tmp_path, monkeypatch, _reset_active):
    """Round-2 stub guarantee: with <3 complete seeds the per-persona
    block is skipped (`inconclusive_descriptive_only`), but the
    unresolved persona's null/skipped stub MUST still be persisted —
    it is written in the payload section, not inside the block."""
    per_cell_dir = tmp_path / "per_cell"
    per_cell_dir.mkdir()
    out_path = tmp_path / "analysis.json"
    monkeypatch.setitem(poa.PER_CELL_DIR_FOR, "cn_i546", per_cell_dir)
    monkeypatch.setitem(poa.OUT_PATH_FOR, "cn_i546", out_path)
    _build_villain_e1_cells(per_cell_dir, role_by_seed=_REALISTIC_ROLE_BY_SEED)
    anchor = tmp_path / "anchor_selection.json"
    anchor.write_text(
        json.dumps(
            {
                "selected_anchor": {"pirate": None, "villain": 1},
                "degenerate": False,
                "partial_anchor": True,
                "partial_anchor_reason": "pirate never banded",
            }
        )
    )
    poa.main(["--variant", "cn_i546", "--anchor-file", str(anchor), "--seeds", "42", "137"])
    payload = json.loads(out_path.read_text())
    assert payload["headline_status"] == "inconclusive_descriptive_only"
    assert payload["partial_anchor"] is True
    head = payload["headline"]
    assert head["per_persona"]["pirate"] == {"plain": None, "padded": None, "skipped": True}
    # The resolved persona's stats were NOT computed (block skipped) —
    # no fabricated cells.
    assert "villain" not in head["per_persona"]
    assert "per_persona_verdict" not in head


def test_main_zero_resolved_personas_writes_skipped_stub(tmp_path, monkeypatch, _reset_active):
    per_cell_dir = tmp_path / "per_cell"
    per_cell_dir.mkdir()
    out_path = tmp_path / "analysis.json"
    monkeypatch.setitem(poa.PER_CELL_DIR_FOR, "cn_i546", per_cell_dir)
    monkeypatch.setitem(poa.OUT_PATH_FOR, "cn_i546", out_path)
    anchor = tmp_path / "anchor_selection.json"
    anchor.write_text(
        json.dumps(
            {
                "selected_anchor": {"pirate": None, "villain": None},
                "degenerate": True,
                "partial_anchor": False,
                "partial_anchor_reason": "no persona banded",
            }
        )
    )
    poa.main(["--variant", "cn_i546", "--anchor-file", str(anchor)])
    payload = json.loads(out_path.read_text())
    assert payload["headline_status"] == "partial_anchor_skipped"
    assert payload["anchored_personas"] == []
    # Round-2 fix (Claude minor #2): `partial_anchor` is DERIVED from the
    # anchor payload — this anchor is fully degenerate (both personas
    # null), which the selector marks `degenerate: true,
    # partial_anchor: false`; the stub no longer hardcodes True.
    assert payload["partial_anchor"] is False
    assert payload["degenerate_anchor"] is True
