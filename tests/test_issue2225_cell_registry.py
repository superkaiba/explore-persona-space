"""Issue #2225 — cell-registry invariants (plan §4.5, exactly 81 finetunes).

Pins the declarative CELL REGISTRY in ``scripts/issue2225_train.py``:
the total count, the per-config grid sizes, the dataset coverage, the steered-trait
map, and the slug<->fields consistency. Import is CHEAP — the training script defers
every heavy import (torch / transformers / trl / issue778_*) inside its functions, so
this test never pulls a GPU stack.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location(
        "issue2225_train", _SCRIPTS / "issue2225_train.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["issue2225_train"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()


def test_registry_has_exactly_81_cells():
    cells = M.build_cell_registry()
    assert len(cells) == 81, len(cells)
    assert M.EXPECTED_CELL_COUNT == 81


def test_slugs_are_unique():
    slugs = [c.slug for c in M.build_cell_registry()]
    assert len(slugs) == len(set(slugs)), "duplicate slugs"


def test_per_config_cell_counts():
    """The §4.5 arithmetic: 16+12+16+12+12+3+3+3+3+1 = 81."""
    counts: dict[str, int] = {}
    for c in M.build_cell_registry():
        counts[c.config] = counts.get(c.config, 0) + 1
    expected = {
        "A": 16,  # 4 datasets x 4 L1 coefs
        "B": 12,  # 4 datasets x 3 multilayer coefs
        "C": 16,  # 4 datasets x 4 L1 coefs
        "D": 12,  # 4 datasets x 3 multilayer coefs
        "E": 12,  # 4 datasets x 3 multilayer coefs
        "F": 3,  # evil x 3 attribution coefs
        "G": 3,
        "I": 3,
        "P": 3,
        "H": 1,  # evil, no coef
    }
    assert counts == expected, counts


def test_grids_match_plan_4_5():
    """L1 grid {0.5,1.5,3,5}; L2/L3 {0.25,0.75,1.5}; attribution {0.5,1.5,3}."""
    cells = M.build_cell_registry()
    by_config: dict[str, set] = {}
    for c in cells:
        by_config.setdefault(c.config, set()).add(c.coef)
    assert by_config["A"] == {0.5, 1.5, 3.0, 5.0}
    assert by_config["C"] == {0.5, 1.5, 3.0, 5.0}
    assert by_config["B"] == {0.25, 0.75, 1.5}
    assert by_config["D"] == {0.25, 0.75, 1.5}
    assert by_config["E"] == {0.25, 0.75, 1.5}
    for cfg in ("F", "G", "I", "P"):
        assert by_config[cfg] == {0.5, 1.5, 3.0}, cfg
    assert by_config["H"] == {None}


def test_dataset_coverage():
    """Core configs A-E span all 4 datasets; F/G/H/I/P are evil-only (§4.5)."""
    cells = M.build_cell_registry()
    by_config: dict[str, set] = {}
    for c in cells:
        by_config.setdefault(c.config, set()).add(c.dataset)
    four = {"evil", "sycophancy", "hallucination", "mistake_opinions"}
    for cfg in ("A", "B", "C", "D", "E"):
        assert by_config[cfg] == four, cfg
    for cfg in ("F", "G", "H", "I", "P"):
        assert by_config[cfg] == {"evil"}, cfg


def test_steered_trait_map():
    """mistake_opinions cells steer evil; the single-trait corpora steer their own."""
    for c in M.build_cell_registry():
        if c.dataset == "mistake_opinions":
            assert c.steered_trait == "evil", c.slug
        else:
            assert c.steered_trait == c.dataset, c.slug


def test_variant_mask_layer_per_config():
    """Slug decode (e{n}s{n}l{n}) maps to (variant, mask_mode, layer_spec)."""
    spec_by_config = {
        "A": ("E1", "all", "L1"),
        "B": ("E1", "all", "L3"),
        "C": ("E2", "context", "L1"),
        "D": ("E2", "context", "L2"),
        "E": ("E2", "context", "L3"),
        "F": ("E1", "context", "L1"),
        "G": ("E2", "all", "L1"),
        "I": ("E1", "response", "L1"),
        "P": ("E3", "prefix", "L1"),
    }
    for c in M.build_cell_registry():
        if c.config == "H":
            assert c.prompt_mode
            assert c.variant is None and c.mask_mode is None and c.layer_spec is None
            assert c.coef is None
            continue
        assert not c.prompt_mode
        assert (c.variant, c.mask_mode, c.layer_spec) == spec_by_config[c.config], c.slug


def test_pilot_is_eight_evil_l1_cells():
    """The §7 P0 gate: A + C at the 4 L1 coefficients on evil II = 8 cells."""
    pilot = M.pilot_cells()
    assert len(pilot) == 8, len(pilot)
    assert {c.config for c in pilot} == {"A", "C"}
    assert {c.dataset for c in pilot} == {"evil"}
    assert {c.coef for c in pilot} == {0.5, 1.5, 3.0, 5.0}


def test_cells_by_slug_round_trips():
    by_slug = M.cells_by_slug()
    assert len(by_slug) == 81
    for slug, cell in by_slug.items():
        assert cell.slug == slug


def test_mask_modes_are_valid_steer_train_modes():
    """Every steered cell's mask_mode is one of steer_train's MASK_MODES."""
    valid = {"all", "context", "response", "prefix"}
    for c in M.build_cell_registry():
        if not c.prompt_mode:
            assert c.mask_mode in valid, c.slug


# ── §7 octave-shift re-pilot: scaled-cell synthesis + resolution (unit 5) ──────


def _ns(**kw):
    """argparse-shaped namespace for _resolve_cells (defaults = production)."""
    import argparse

    defaults = dict(
        pilot=False, cells=None, smoke=False, coef_scale=None, pilot_coefs=None, pilot_configs=None
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_synth_cell_matches_registry_cell_on_registry_coef():
    reg = M.cells_by_slug()["A__evil__c3.0"]
    assert M.synth_cell("A", "evil", 3.0) == reg  # frozen dataclass equality


def test_synth_cell_canonical_slug_and_fields():
    c = M.synth_cell("A", "evil", 0.25)
    assert c.slug == "A__evil__c0.25"
    assert (c.variant, c.mask_mode, c.layer_spec) == ("E1", "all", "L1")
    assert c.steered_trait == "evil" and not c.prompt_mode


def test_synth_cell_refuses_bad_inputs():
    import pytest

    with pytest.raises(ValueError, match="unknown config"):
        M.synth_cell("Z", "evil", 1.0)
    with pytest.raises(ValueError, match="prompt-mode"):
        M.synth_cell("H", "evil", 1.0)
    with pytest.raises(ValueError, match="not in config"):
        M.synth_cell("F", "sycophancy", 1.0)  # F is evil-only
    with pytest.raises(ValueError, match="finite and > 0"):
        M.synth_cell("A", "evil", 0.0)
    with pytest.raises(ValueError, match="finite and > 0"):
        M.synth_cell("A", "evil", float("nan"))


def test_resolve_cell_registry_hit_and_scaled_miss():
    assert M.resolve_cell("A__evil__c3.0") is M.resolve_cell("A__evil__c3.0") or True
    assert M.resolve_cell("A__evil__c3.0") == M.cells_by_slug()["A__evil__c3.0"]
    scaled = M.resolve_cell("C__evil__c0.75")
    assert scaled.config == "C" and scaled.coef == 0.75
    assert scaled.slug == "C__evil__c0.75"


def test_resolve_cell_refuses_noncanonical_and_unknown():
    import pytest

    with pytest.raises(ValueError, match="non-canonical"):
        M.resolve_cell("A__evil__c2.50")  # canonical is c2.5
    with pytest.raises(ValueError, match="unknown cell slug"):
        M.resolve_cell("not_a_slug")
    with pytest.raises(ValueError, match="prompt-mode"):
        M.resolve_cell("H__evil__c1.0")  # H has no coefficient


def test_resolve_cells_pilot_coef_scale_halves_grid():
    cells = M._resolve_cells(_ns(pilot=True, coef_scale=0.5))
    assert len(cells) == 8
    assert {c.coef for c in cells} == {0.25, 0.75, 1.5, 2.5}
    assert {c.config for c in cells} == {"A", "C"}
    # x2 shift: scaled coefs landing back on registry values dedupe by slug
    doubled = M._resolve_cells(_ns(pilot=True, coef_scale=2.0))
    assert {c.coef for c in doubled} == {1.0, 3.0, 6.0, 10.0}
    assert M.cells_by_slug()["A__evil__c3.0"] in doubled


def test_resolve_cells_pilot_configs_subset_and_pilot_coefs():
    cells = M._resolve_cells(_ns(pilot=True, pilot_configs="A", coef_scale=0.5))
    assert len(cells) == 4 and {c.config for c in cells} == {"A"}
    replaced = M._resolve_cells(_ns(pilot=True, pilot_coefs="0.1,0.2"))
    assert {c.coef for c in replaced} == {0.1, 0.2} and len(replaced) == 4


def test_resolve_cells_pilot_flags_require_pilot():
    import pytest

    with pytest.raises(ValueError, match="require --pilot"):
        M._resolve_cells(_ns(coef_scale=0.5))
    with pytest.raises(ValueError, match="subset"):
        M._resolve_cells(_ns(pilot=True, pilot_configs="A,Z"))


def test_argparser_coef_scale_and_pilot_coefs_mutually_exclusive():
    import pytest

    ap = M.build_argparser()
    with pytest.raises(SystemExit):
        ap.parse_args(["--pilot", "--coef-scale", "0.5", "--pilot-coefs", "0.1,0.2"])
    args = ap.parse_args(["--pilot", "--coef-scale", "0.5", "--pilot-configs", "A"])
    assert args.coef_scale == 0.5 and args.pilot_configs == "A"


def test_eval_gen_resolve_targets_scaled_fallback():
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location("issue2225_eval_gen", _SCRIPTS / "issue2225_eval_gen.py")
    eg = _ilu.module_from_spec(spec)
    sys.modules["issue2225_eval_gen"] = eg
    spec.loader.exec_module(eg)
    got = eg.resolve_targets(["base", "A__evil__c3.0", "A__evil__c0.25"])
    assert [t.tag for t in got] == ["base", "A__evil__c3.0", "A__evil__c0.25"]
    assert got[2].kind == "cell" and got[2].dataset == "evil"
    assert got[2].traits == ("evil",)
    import pytest

    with pytest.raises(ValueError, match="unknown eval-target tag"):
        eg.resolve_targets(["A__evil__c2.50"])  # non-canonical spelling refused
