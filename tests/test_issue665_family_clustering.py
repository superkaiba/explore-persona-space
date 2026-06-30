"""Round-3 Blocker C regression — family clustering uses the CONTEXT family.

Round 2 clustered cells by `C.family_of_context(parsed["source"])`, where
`parsed["source"]` is a slug LABEL (`default`/`librarian`), but `family_of_context`
expects an f1..f8-prefixed CONTEXT id — so it returned `"other"` for EVERY cell and
the hierarchical family bootstrap collapsed to a single cluster. The fix clusters on
the family of the cell's SOURCE-ANCHOR CONTEXT (plan §9 grain): `f6_helpful_asst` →
`default`, `f1_house_librarian` → `persona`, etc.

This test asserts the RESOLVED FAMILY VALUES (not the label/shape): with distinct
source contexts across cells the resolved families are distinct and NONE is `"other"`.

Fails pre-fix (every cell → "other", single cluster), passes post-fix. CPU-only:
no HF / network — the source family is read from the persisted g0_E0 fields.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))


# Three distinct source CONTEXT families exercised by the test (the f-prefix and the
# mapped #594 family family_of_context returns for it).
_SRC_CTX = {
    "f1_house_librarian": "persona",
    "f3_icl_assistant": "icl",
    "f6_helpful_asst": "default",
}


def test_source_family_for_cell_resolves_context_family_not_label():
    """`_source_family_for_cell` returns the #594 family of the SOURCE CONTEXT, never
    `"other"` — and NEVER family_of_context of the bare source label."""
    import issue665_aggregate as A
    import issue665_common as C

    # g0_e0 carries the persisted source_ctx_id + source_family (gate_cpu writes them)
    g0_e0 = {
        "bm_default_contra_d1_seed42": {
            "source_ctx_id": "f6_helpful_asst",
            "source_family": "default",
        },
        "bm_librarian_contra_d1_seed42": {
            "source_ctx_id": "f1_house_librarian",
            "source_family": "persona",
        },
    }
    fam_default = A._source_family_for_cell("bm_default_contra_d1_seed42", g0_e0)
    fam_lib = A._source_family_for_cell("bm_librarian_contra_d1_seed42", g0_e0)
    assert fam_default == "default"
    assert fam_lib == "persona"
    # the round-2 bug value (family of the bare LABEL) is "other" — must NOT match
    assert C.family_of_context("default") == "other"  # the bug's return
    assert C.family_of_context("librarian") == "other"
    assert fam_default != "other" and fam_lib != "other"


def test_source_family_falls_back_to_ctx_id_when_family_field_absent():
    """If only `source_ctx_id` is persisted (no `source_family`), resolve via
    family_of_context on the context id — still never the bare label."""
    import issue665_aggregate as A

    g0_e0 = {"bm_default_contra_d1_seed42": {"source_ctx_id": "f3_icl_assistant"}}
    assert A._source_family_for_cell("bm_default_contra_d1_seed42", g0_e0) == "icl"


def test_family_clustering_distinct_no_other(tmp_path, monkeypatch):
    """End-to-end through the aggregate: 3 cells with 3 distinct source-context
    families produce 3 distinct cluster families, NONE collapsing to "other"."""
    import issue665_aggregate as A
    import issue665_common as C

    monkeypatch.setattr(C, "EVAL_ROOT", tmp_path)

    # three bad-medical cells, each with a DIFFERENT source-anchor context family.
    cells = [
        "bm_default_contra_d1_seed42",  # f6 -> default
        "bm_librarian_contra_d1_seed42",  # f1 -> persona
        "bm_default_posonly_d2_seed42",  # reuse f3 ctx to get a 3rd family
    ]
    src_ctx_for_cell = {
        cells[0]: "f6_helpful_asst",
        cells[1]: "f1_house_librarian",
        cells[2]: "f3_icl_assistant",
    }

    rng = np.random.default_rng(3)
    for cell in cells:
        layer = str(C.read_layer_for_cell(cell))
        (tmp_path / "a310").mkdir(parents=True, exist_ok=True)
        (tmp_path / "a39").mkdir(parents=True, exist_ok=True)
        (tmp_path / "a38").mkdir(parents=True, exist_ok=True)
        (tmp_path / "a310" / f"{cell}.json").write_text(
            json.dumps({"by_layer": {layer: {"g0_spearman": 0.4, "gplus_spearman": 0.6}}})
        )
        (tmp_path / "a39" / f"{cell}.json").write_text(
            json.dumps(
                {
                    "by_layer": {
                        layer: {
                            "cosine_spearman": 0.2,
                            "verdict_i_some_beats_cosine": True,
                            "verdict_ii_sigma_inv_wins": True,
                        }
                    }
                }
            )
        )
        (tmp_path / "a38" / f"{cell}.json").write_text(
            json.dumps(
                {"by_layer": {layer: {"median_rankone_residual": 0.1, "svd_sigma1_frac": 0.8}}}
            )
        )
        (tmp_path / "per_cell" / cell).mkdir(parents=True, exist_ok=True)
        entries = []
        for k in range(20):
            e0 = float(rng.standard_normal())
            g0 = float(rng.standard_normal())
            entries.append(
                {
                    "context_id": f"f{(k % 8) + 1}_c{k}",
                    "g0": g0,
                    "ghat_real": 0.5 * g0 + 0.4 * e0,
                    "E0": e0,
                    "wnorm": 3.0,
                }
            )
        src_ctx = src_ctx_for_cell[cell]
        (tmp_path / "per_cell" / cell / "g0_E0.json").write_text(
            json.dumps(
                {
                    "entries": entries,
                    "source_ctx_id": src_ctx,
                    "source_family": C.family_of_context(src_ctx),
                }
            )
        )

    # the family cluster grain each cell resolves to (the unit under test)
    g0_e0 = A._load_g0_e0(cells)
    families = {A._source_family_for_cell(c, g0_e0) for c in cells}
    assert "other" not in families, "no cell may collapse to the 'other' family (Blocker C)"
    assert len(families) >= 3, "three distinct source-context families must be distinct clusters"
    # the RESOLVED #594 family names for f1/f3/f6 (family_of_context maps prefix->name)
    assert families == {"persona", "icl", "default"}, families

    # and the aggregate runs end-to-end with the family-clustered CI present
    monkeypatch.setattr(
        A,
        "_probe_split_replication",
        lambda beh_cells, rng: {
            "probe_split_floor_mean": 0.05,
            "probe_split_floor_ci_lo": 0.02,
            "probe_split_floor_ci_hi": 0.08,
            "probe_split_replication_count": C.PROBE_SPLIT_R,
            "n_cells": len(beh_cells),
        },
    )
    agg = A.aggregate(cells, smoke=True)
    pb = agg["per_behavior"]["bad_medical"]
    fam_ci = pb["family_clustered_ci_g0_spearman"]
    # 3 distinct families -> the family-clustered bootstrap resamples >1 cluster
    assert fam_ci["n_clusters"] == 3, (
        f"family-clustered bootstrap must see 3 distinct clusters, got {fam_ci.get('n_clusters')}"
    )
