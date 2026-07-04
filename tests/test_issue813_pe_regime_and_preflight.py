"""Regression tests — #813 per-example-vs-averaged driver resume + NPZ-preflight invariants.

Pins the round-2 review fix list (epm:code-review v8 CONCERNS / epm:code-review-codex v8
FAIL union) plus the v7 invariants they extend:

(a) the NPZ schema preflight lives INSIDE ``issue813_save_maps.load_reduced_cells`` —
    the imported perlayer path fails loud, naming the missing keys, BEFORE any fit
    (closes concern ``perlayer-npz-key-coverage-preflight``);
(b) the per-cell resume regime key covers the gate/diagnostic flags
    (``dv6_oracle``, ``dv4_lambda_spotcheck``, per-cell ``equiv_gate``) — a stale cell
    JSON can never silently skip a mandatory gate on resume;
(c) the perlayer profile resume predicate misses on a wrong / permuted layer list
    (the v7 ``perlayer-resume-stale-regime`` invariant).

Self-contained: tmp_path fixtures only — no HF access, no network, no fits.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
for _p in (_REPO / "src", _REPO / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import issue813_per_example_maps as pemaps  # noqa: E402
import issue813_perlayer_profile as perlayer  # noqa: E402
import issue813_save_maps as savemaps  # noqa: E402

# ── (a) NPZ preflight inside load_reduced_cells ───────────────────────────────


def _write_summary_npz(path: Path, *, drop: str | None = None, n: int = 2) -> None:
    """A minimal reduced summary.npz (real shapes, zeros); optionally drop one key."""
    arrs: dict[str, np.ndarray] = {
        "c_C_base": np.zeros((n, savemaps.N_LAYERS, savemaps.HIDDEN), dtype=np.float32),
        "c_C_trained": np.zeros((n, savemaps.N_LAYERS, savemaps.HIDDEN), dtype=np.float32),
        "v_A_base": np.zeros((n, savemaps.N_LAYERS, savemaps.HIDDEN), dtype=np.float32),
        "v_A_trained": np.zeros((n, savemaps.N_LAYERS, savemaps.HIDDEN), dtype=np.float32),
        "context_ids": np.asarray([f"ctx{i}" for i in range(n)]),
        "families": np.asarray([f"fam{i}" for i in range(n)]),
    }
    if drop is not None:
        arrs.pop(drop)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrs)


def test_load_reduced_cells_preflight_raises_naming_missing_key(tmp_path):
    """A summary.npz missing v_A_base raises the preflight KeyError BEFORE any fit."""
    root = tmp_path / "reduced"
    _write_summary_npz(root / "em" / "elicit" / "summary.npz", drop="v_A_base")
    with pytest.raises(KeyError, match="v_A_base") as exc:
        savemaps.load_reduced_cells("em", "elicit", 14, root)
    assert "preflight" in str(exc.value)  # the actionable message, not a bare KeyError


def test_load_reduced_cells_happy_path_builds_cellrecords(tmp_path):
    """Positive control: a complete summary.npz still loads (preflight is transparent)."""
    root = tmp_path / "reduced"
    _write_summary_npz(root / "em" / "elicit" / "summary.npz", n=3)
    cells = savemaps.load_reduced_cells("em", "elicit", 14, root)
    assert len(cells) == 3
    assert cells[0].layer == 14
    assert cells[0].behavior == "em"


def test_preflight_helper_canonical_home_is_save_maps():
    """Both consumers import the SAME canonical helper (no drift-prone copies)."""
    assert pemaps._require_npz_keys is savemaps._require_npz_keys
    assert perlayer._require_npz_keys is savemaps._require_npz_keys


# ── (b) per-cell resume regime covers the gate/diagnostic flags ───────────────


def _args(**overrides) -> object:
    """Driver args at parser defaults, with explicit overrides."""
    args = pemaps.build_parser().parse_args([])
    for k, v in overrides.items():
        assert hasattr(args, k), f"unknown driver arg {k!r}"
        setattr(args, k, v)
    return args


def _stored_cell(tmp_path: Path, regime: dict) -> Path:
    """A schema-complete fake cell JSON carrying ``regime`` (JSON round-tripped)."""
    obj: dict = {k: {} for k in pemaps._CELL_SCHEMA_KEYS}
    obj["regime"] = regime
    p = tmp_path / "transfer_L14_em__elicit.json"
    p.write_text(json.dumps(obj, default=float))
    return p


def test_cell_resume_hits_on_identical_regime(tmp_path):
    """Positive control: unchanged regime (through a JSON round-trip) resume-hits."""
    p = _stored_cell(tmp_path, pemaps._regime(_args(), run_equiv_gate=False))
    assert pemaps._cell_resume_valid(p, pemaps._regime(_args(), run_equiv_gate=False))


@pytest.mark.parametrize("override", [{"dv6_oracle": True}, {"dv4_lambda_spotcheck": 3}])
def test_cell_resume_misses_when_gate_flag_requested(tmp_path, override):
    """A run requesting --dv6-oracle / --dv4-lambda-spotcheck must NOT reuse a cell
    computed without it (the round-2 Codex Major: resume laundering a skipped gate)."""
    p = _stored_cell(tmp_path, pemaps._regime(_args(), run_equiv_gate=False))
    assert not pemaps._cell_resume_valid(p, pemaps._regime(_args(**override), run_equiv_gate=False))


def test_cell_resume_misses_when_equiv_gate_membership_flips(tmp_path):
    """Adding this cell to --equiv-cells invalidates ITS cached JSON (per-cell key)."""
    p = _stored_cell(tmp_path, pemaps._regime(_args(), run_equiv_gate=False))
    assert not pemaps._cell_resume_valid(p, pemaps._regime(_args(), run_equiv_gate=True))


# ── (c) perlayer profile resume misses on a wrong layer list (v7 invariant) ───


def _profile(layers: tuple[int, ...]) -> dict:
    """A schema-complete fake perlayer profile at the current regime pins."""
    return {
        "behavior": "em",
        "substrate": "elicit",
        "headline_layer": perlayer.HEADLINE_LAYER,
        "hf_revision": perlayer.HF_REVISION,
        "regime_version": perlayer.PERLAYER_REGIME_VERSION,
        "layers": list(layers),
        "per_layer": [
            {**{k: 0.0 for k in perlayer._ROW_SCHEMA_KEYS}, "layer": layer} for layer in layers
        ],
    }


def test_profile_resume_hits_on_exact_layer_list(tmp_path):
    """Positive control: exact layer list + schema + pins resume-hit."""
    p = tmp_path / "em__elicit.json"
    p.write_text(json.dumps(_profile((1, 2, 14))))
    assert perlayer._profile_resume_valid(p, "em", "elicit", (1, 2, 14))


def test_profile_resume_misses_on_wrong_layer_list(tmp_path):
    """Wrong, subset, or permuted layer lists all miss (exact per-row list, not count)."""
    p = tmp_path / "em__elicit.json"
    p.write_text(json.dumps(_profile((1, 2, 14))))
    assert not perlayer._profile_resume_valid(p, "em", "elicit", (1, 2, 3))
    assert not perlayer._profile_resume_valid(p, "em", "elicit", (14, 2, 1))
    assert not perlayer._profile_resume_valid(p, "em", "elicit", (1, 2))
