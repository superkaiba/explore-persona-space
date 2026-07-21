"""#1112 rankem cross-method cosine extension (CPU-only).

Pins the additive rankem extension to ``scripts/issue1112_cross_method_cosine.py``:
the 3 rankem pairs (A1/A2 low-rank non-rsLoRA sycophancy vs the parent full-FT+neg
comparator; B misalignment full-FT vs LoRA), the per-cell prefix/rev routing (rankem
cells under the rankem sub-prefix + RANKEM_CAPTURE_REV; parent cells unchanged), and
the TF_CELLS extension. Parent pairs must stay byte-identical.
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1112_cross_method_cosine as X  # noqa: E402


def test_rankem_pairs_registered() -> None:
    labels = [p["label"] for p in X.PAIRS]
    for lbl in (
        "rankem_a1_ftneg_vs_lora_r1",
        "rankem_a2_ftneg_vs_lora_r4",
        "rankem_b_misalignment_ft_vs_lora",
    ):
        assert lbl in labels, f"{lbl} missing from PAIRS"


def test_rankem_pair_cells_and_convention() -> None:
    by = {p["label"]: p for p in X.PAIRS}
    a1 = by["rankem_a1_ftneg_vs_lora_r1"]
    # parent convention: cell_a = full-FT comparator, cell_b = LoRA
    assert a1["cell_a"] == "s3_fullft_neg" and a1["rev_a"] == X.OWN_REV
    assert a1["cell_b"] == "a1_lora_r1" and a1["rev_b"] == X.RANKEM_CAPTURE_REV
    assert a1["base_cell"] == X.BASE_SYCO
    b = by["rankem_b_misalignment_ft_vs_lora"]
    assert b["cell_a"] == "b2_fullft_em" and b["cell_b"] == "b1_lora_em"
    assert b["rev_a"] == X.RANKEM_CAPTURE_REV and b["rev_b"] == X.RANKEM_CAPTURE_REV


def test_cell_prefix_and_rev_routing() -> None:
    for cell in X.RANKEM_CELLS:
        assert X._cell_prefix(cell) == X.RANKEM_DATA_PREFIX
        assert X._cell_rev(cell, "TF_DEFAULT") == X.RANKEM_CAPTURE_REV
    # parent cells unchanged
    assert X._cell_prefix("s3_fullft_neg") == X.DATA_PREFIX
    assert X._cell_rev("s3_fullft_neg", "TF_DEFAULT") == "TF_DEFAULT"
    assert X._cell_prefix(X.BASE_SYCO) == X.DATA_PREFIX


def test_rankem_prefix_is_parent_subprefix() -> None:
    assert f"{X.DATA_PREFIX}/rankem" == X.RANKEM_DATA_PREFIX


def test_rankem_cells_in_tf_cells() -> None:
    for cell in X.RANKEM_CELLS:
        assert cell in X.TF_CELLS
    # parent tf cells still present (byte-identical extension)
    for cell in ("s1_lora_neg", "s2_lora_pos", "s3_fullft_neg", "s4_fullft_pos"):
        assert cell in X.TF_CELLS


def test_parent_pairs_unchanged() -> None:
    """The 4 parent pairs must survive the extension byte-identical."""
    labels = [p["label"] for p in X.PAIRS]
    for lbl in (
        "H1x_ftneg_vs_loraneg",
        "H1x_pos_ftpos_vs_lorapos",
        "H1x_lrm_ftneg_vs_lora_lr5e6",
        "marker_ft_vs_lora",
    ):
        assert lbl in labels, f"parent pair {lbl} was dropped by the rankem extension"
