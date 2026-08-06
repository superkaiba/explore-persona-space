"""CPU-only regression tests for the issue #2094 fu1_regen_confirm driver.

Pins the two IN-CODE derivations against the COMMITTED parent artifacts
(the brief's assert-the-derived-shape requirement):

- the 16 cap-hit-breached pooled (slot, layer_variant, dose) cells from
  ``eval_results/issue_2094/fragility/fragility_cells.json`` (steered
  cap_hit_frac > 0.02), and
- the 15 CONF-1 surviving families from
  ``eval_results/issue_2094/f_metrics/bootstrap_cis_wellsep.json``
  (CIs disjoint-above, >=5 wellsep pairs, behavior metrics, minus the 16).

Plus the regen block-set reconciliation, the smoke-slice subset invariants,
and the conf1 regime/resume keys. No model, no GPU, no network.
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2094_fu1 as F  # noqa: E402
import issue2094_run as R  # noqa: E402

from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402

FRAGILITY = REPO_ROOT / F.FRAGILITY_REL
WELLSEP = REPO_ROOT / F.WELLSEP_REL

EXPECTED_BREACHED_CELLS = [
    ("ce", "joint_all", "a0.5"),
    ("ce", "joint_all", "a1"),
    ("ce", "joint_all", "a2"),
    ("ce", "joint_all", "a4"),
    ("ce", "joint_mid", "a1"),
    ("ce", "joint_mid", "a4"),
    ("l3j", "joint_mid", "a0.5"),
    ("l3j", "joint_mid", "a1"),
    ("l3j", "joint_mid", "a2"),
    ("l3j", "joint_mid", "a4"),
    ("l3j", "joint_mid", "replace"),
    ("qspan", "joint_mid", "a0.5"),
    ("qspan", "joint_mid", "a1"),
    ("qspan", "joint_mid", "a2"),
    ("qspan", "joint_mid", "a4"),
    ("qspan", "joint_mid", "replace"),
]

EXPECTED_CONF1_FAMILIES = [
    "cross|ce|L15|a1|A|f_beh_prefix",
    "matched_query|ce|L12|a4|A|f_beh_prefix",
    "matched_query|ce|L13|a2|A|f_beh_prefix",
    "matched_query|ce|L14|a1|A|f_beh_prefix",
    "matched_query|ce|L15|replace|A|f_beh_prefix",
    "matched_query|ce|L16|a1|A|f_beh_prefix",
    "matched_query|ce|L16|a2|A|f_beh_prefix",
    "matched_query|ce|L16|a4|A|f_beh_prefix",
    "matched_query|ce|L17|a1|A|f_beh_prefix",
    "matched_query|ce|L17|a2|A|f_beh_prefix",
    "matched_query|ce|L20|a2|B|f_beh_prefix",
    "matched_query|ce|L20|replace|A|f_beh_prefix",
    "matched_query|ce|joint_all|replace|A|f_beh_prefix",
    "matched_query|ce|joint_mid|a0.5|A|f_beh_prefix",
    "matched_query|ce|joint_mid|a0.5|B|f_beh_prefix",
]


@pytest.fixture(scope="module")
def fragility() -> dict:
    return json.loads(FRAGILITY.read_text())


@pytest.fixture(scope="module")
def wellsep() -> dict:
    return json.loads(WELLSEP.read_text())


@pytest.fixture(scope="module")
def breached(fragility) -> list[tuple[str, str, str]]:
    return F.derive_breached_cells(fragility)


@pytest.fixture(scope="module")
def pairs() -> list[BANK.Pair]:
    return BANK.build_pairs()


# ── sub-item A derivation: the 16 breached pooled cells ────────────────


def test_breached_cells_derivation_pinned(breached):
    assert breached == EXPECTED_BREACHED_CELLS
    assert len(breached) == 16


def test_breached_derivation_fails_loud_on_missing_cell(fragility):
    """Dropping one breached cell from the artifact must FAIL the shape
    assert (the derivation never silently re-scopes)."""
    mutated = copy.deepcopy(fragility)
    mutated["cells"] = [
        c
        for c in mutated["cells"]
        if not (c["slot"] == "qspan" and c["layer_variant"] == "joint_mid" and c["dose"] == "a4")
    ]
    with pytest.raises(AssertionError, match="pre-registered shape"):
        F.derive_breached_cells(mutated)


def test_breached_derivation_fails_loud_on_extra_cell(fragility):
    """A NEW over-trigger cell outside the expected shape must FAIL loud."""
    mutated = copy.deepcopy(fragility)
    for c in mutated["cells"]:
        if c["slot"] == "pe" and c["layer_variant"] == "joint_mid" and c["dose"] == "a1":
            c["steered"]["cap_hit_frac"] = 0.5
    with pytest.raises(AssertionError, match="pre-registered shape"):
        F.derive_breached_cells(mutated)


# ── sub-item B derivation: the 15 CONF-1 surviving families ────────────


def test_conf1_families_derivation_pinned(wellsep, breached):
    fams = F.derive_conf1_families(wellsep, set(breached))
    assert [f["family"] for f in fams] == EXPECTED_CONF1_FAMILIES
    assert all(f["slot"] == "ce" for f in fams)  # all context-end
    assert all(f["metric"] in F.BEH_METRICS for f in fams)
    # Disjoint from the breached pooled cells by construction.
    assert all((f["slot"], f["layer_variant"], f["dose"]) not in set(breached) for f in fams)


def test_conf1_families_fails_loud_on_count_drift(wellsep, breached):
    """Flipping one surviving family's disjointness must FAIL the ==15
    assert (fail-loud, never a silent descope)."""
    mutated = copy.deepcopy(wellsep)
    mutated["steered_vs_null"]["cross|ce|L15|a1|A|f_beh_prefix"]["cis_disjoint"] = False
    with pytest.raises(AssertionError, match="surviving families"):
        F.derive_conf1_families(mutated, set(breached))


def test_conf1_cells_distinct_and_typed(wellsep, breached):
    fams = F.derive_conf1_families(wellsep, set(breached))
    cells = F.conf1_cells_from_families(fams)
    assert len(cells) == 15
    keys = [(c["setting"], c["slot"], c["layer_variant"], c["dose"], c["vec_type"]) for c in cells]
    assert len(set(keys)) == 15
    for c in cells:
        if c["vec_type"] == "B":
            assert c["setting"] == "matched_query"
        assert c["families"], c


# ── regen block-set reconciliation ─────────────────────────────────────


def test_regen_block_families_totals_and_membership(pairs, breached):
    fams = F.regen_block_families(pairs, R.N_MODEL_LAYERS_FULL, set(breached))
    totals = R.grid_totals(fams)
    assert totals == F.EXPECTED_REGEN_TOTALS
    for steered, null in fams:
        assert (steered.slot, steered.layer_variant, steered.dose) in set(breached)
        assert steered.arm == "steered" and null.arm == "null"
        assert steered.pair_ids == null.pair_ids
    # ce pooled cells carry BOTH vec types (A over all pairs + B over mq).
    ce_specs = {(f[0].layer_variant, f[0].dose, f[0].vec_type) for f in fams if f[0].slot == "ce"}
    for lv, dose in [("joint_all", "a1"), ("joint_mid", "a4")]:
        assert (lv, dose, "A") in ce_specs and (lv, dose, "B") in ce_specs
    # Control slots are Type A only.
    assert all(f[0].vec_type == "A" for f in fams if f[0].slot in ("qspan", "l3j"))


# ── smoke slices are strict subsets of the derived production sets ─────


def test_smoke_slices_are_subsets(pairs, wellsep, breached):
    fams = F.regen_block_families(pairs, R.N_MODEL_LAYERS_FULL, set(breached))
    smoke = F.slice_regen_smoke(fams, pairs)
    prod_specs = {(f[0].slot, f[0].layer_variant, f[0].dose, f[0].vec_type) for f in fams}
    prod_ids = {f[0].key: set(f[0].pair_ids) for f in fams}
    assert len(smoke) == len(F.SMOKE_REGEN_FAMILIES)
    for steered, null in smoke:
        spec = (steered.slot, steered.layer_variant, steered.dose, steered.vec_type)
        assert spec in prod_specs
        assert set(steered.pair_ids) <= prod_ids[steered.key]
        assert steered.pair_ids == null.pair_ids
        assert steered.pair_ids  # never empty
    # The A-block smoke subset keeps a conv-context_a pair (render seam).
    a_subset = F.smoke_pair_subset(pairs)
    assert any(pid.split("--")[1].startswith("conv") for pid in a_subset)

    cells = F.conf1_cells_from_families(F.derive_conf1_families(wellsep, set(breached)))
    smoke_cells = F.slice_conf1_smoke(cells)
    assert len(smoke_cells) == len(F.SMOKE_CONF1_KEYS)
    all_ids = {
        "|".join([c["setting"], c["slot"], c["layer_variant"], c["dose"], c["vec_type"]])
        for c in cells
    }
    assert set(F.SMOKE_CONF1_KEYS) <= all_ids
    # Coverage: both settings-classes and both vec types.
    assert {c["vec_type"] for c in smoke_cells} == {"A", "B"}
    assert len({c["setting"] for c in smoke_cells}) == 2


# ── conf1 cell keys + resume regime ────────────────────────────────────


def _cell() -> dict:
    return {
        "setting": "matched_query",
        "slot": "ce",
        "layer_variant": "L16",
        "dose": "a2",
        "vec_type": "A",
        "families": ["matched_query|ce|L16|a2|A|f_beh_prefix"],
    }


def test_conf1_cell_key_unique_across_arms():
    keys = {F.conf1_cell_key(_cell(), arm) for arm in R.ARMS}
    assert len(keys) == 2
    for k in keys:
        slug = R.block_slug(k)
        assert "|" not in slug and "." not in slug  # filesystem-safe


def test_conf1_regime_keys_on_output_affecting_knobs():
    args = F.parse_args(["--run", "--out-root", "/tmp/x"])
    _, cfg = F.build_configs(args)
    base = F.conf1_regime(cfg, _cell(), "steered", 5, "sha")
    assert F.conf1_regime(cfg, _cell(), "null", 5, "sha") != base  # arm
    assert F.conf1_regime(cfg, _cell(), "steered", 2, "sha") != base  # draws
    assert F.conf1_regime(cfg, _cell(), "steered", 5, "other") != base  # bank
    from dataclasses import replace

    cfg2 = replace(cfg, max_new_tokens=cfg.max_new_tokens * 2)
    assert F.conf1_regime(cfg2, _cell(), "steered", 5, "sha") != base  # cap


def test_build_configs_phase_split():
    args = F.parse_args(["--run", "--out-root", "/tmp/x"])
    regen, conf1 = F.build_configs(args)
    assert regen.max_new_tokens == F.REGEN_MAX_NEW_TOKENS  # 2048, 2x parent
    assert conf1.max_new_tokens == R.MAX_NEW_TOKENS  # parent cap (1024)
    assert conf1.anchor_draws == F.CONF1_DRAWS == 5
    assert regen.out_root == conf1.out_root  # one tree, two regimes
