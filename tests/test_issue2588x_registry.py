"""Issue #2588-larger registry pins — extension rows added, original rows frozen.

The extension must keep every pre-existing cell's behavior byte-identical:
this file pins the original 19 cells' registry-derived behavior surface
(keys, families, arms, parse modes, input positions, hf prefixes, default
tp_gpus) and asserts the new rows' contracts (arms, parse modes, TP, AA pins,
sweep-layer arithmetic across 43-78 layers).
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue2588_panel_common as PC

ORIGINAL_KEYS = (
    "q35_0p8b",
    "q35_2b",
    "q35_4b",
    "q35_9b",
    "q35_27b",
    "q36_27b",
    "q38_27b",
    "o3_7b_i",
    "o3_7b_t",
    "o31_32b_i",
    "o31_32b_t",
    "q25_7b",
)
EXTENSION_KEYS = ("q38fn", "q35_397b", "dsv4_flash", "glm53", "dsv4_pro")


def test_registry_arithmetic_28_cells_30_maps():
    cells = PC.all_cells()
    assert len(cells) == 28
    assert sum(len(c.input_positions) for c in cells) == 30
    # The ORIGINAL 19 cells still exist with the ORIGINAL keys, in order.
    orig = [c for c in cells if c.model_key in ORIGINAL_KEYS]
    assert len(orig) == 19
    assert sum(len(c.input_positions) for c in orig) == 21


def test_original_rows_byte_identical_behavior_surface():
    """The pre-extension rows keep every registry-derived behavior field."""
    pins = {
        # key: (family, arms, thinking, banked_arm_a, n_layers, h_dim)
        "q35_0p8b": ("qwen35", ("a", "b"), True, False, 24, 1024),
        "q35_2b": ("qwen35", ("a", "b"), True, False, 24, 2048),
        "q35_4b": ("qwen35", ("a", "b"), True, False, 32, 2560),
        "q35_9b": ("qwen35", ("a", "b"), True, True, 32, 4096),
        "q35_27b": ("qwen35", ("a", "b"), True, False, 64, 5120),
        "q36_27b": ("qwen36", ("a", "b"), True, False, 64, 5120),
        "q38_27b": ("qwen38", ("a", "b"), True, False, 64, 5120),
        "o3_7b_i": ("olmo_instruct", ("a",), False, False, 32, 4096),
        "o3_7b_t": ("olmo_think", ("b",), True, False, 32, 4096),
        "o31_32b_i": ("olmo_instruct", ("a",), False, False, 64, 5120),
        "o31_32b_t": ("olmo_think", ("b",), True, False, 64, 5120),
        "q25_7b": ("qwen25", ("a",), False, True, 28, 3584),
    }
    for key, (family, arms, thinking, banked, n_layers, h_dim) in pins.items():
        m = PC.PANEL[key]
        assert (m.family, m.arms, m.thinking, m.banked_arm_a) == (family, arms, thinking, banked)
        assert (m.n_layers, m.h_dim) == (n_layers, h_dim)
        # New fields default so nothing pre-existing changes shape:
        assert m.tp_gpus == 1
        assert m.est_snapshot_gb is None
    # Template-kwargs surface unchanged: the startswith("qwen3") rewrite maps
    # the exact same families as the old ("qwen35","qwen36","qwen38") tuple.
    for fam in ("qwen35", "qwen36", "qwen38"):
        assert PC._template_kwargs(fam, "a") == {"enable_thinking": False}
        assert PC._template_kwargs(fam, "b") == {"enable_thinking": True}
    for fam in ("qwen25", "olmo_instruct", "olmo_think"):
        assert PC._template_kwargs(fam, "a") == {}
        assert PC._template_kwargs(fam, "b") == {}


def test_extension_rows_contracts():
    expected = {
        # key: (family, arms, tp, hf_id)
        "q38fn": ("qwen38fn", ("a", "b"), 2, "Qwen/Qwen3.8-Flash-Next-FP8"),
        "q35_397b": ("qwen35", ("a", "b"), 4, "Qwen/Qwen3.5-397B-A17B-FP8"),
        "dsv4_flash": ("deepseek_v4", ("a", "b"), 2, "deepseek-ai/DeepSeek-V4-Flash-0731"),
        "glm53": ("glm53", ("b",), 8, "zai-org/GLM-5.3"),
        "dsv4_pro": ("deepseek_v4", ("a", "b"), 8, "deepseek-ai/DeepSeek-V4-Pro-0813"),
    }
    for key, (family, arms, tp, hf_id) in expected.items():
        m = PC.PANEL[key]
        assert (m.family, m.arms, m.tp_gpus, m.hf_id) == (family, arms, tp, hf_id)
        assert m.thinking and not m.banked_arm_a
        assert m.est_snapshot_gb is not None and m.est_snapshot_gb > 100
        for arm in arms:
            cell = PC.cell_by_key(f"{key}_{arm}")
            assert cell.fresh
            if arm == "a":
                assert cell.parse_mode == "off"
                assert cell.input_positions == ("prompt_last",)
            else:
                assert cell.parse_mode == "prefill"
                assert cell.input_positions == ("cot_boundary",)
    # qwen38fn rides the shared qwen3 template kwargs; the new non-jinja /
    # thinking-only families ride none.
    assert PC._template_kwargs("qwen38fn", "a") == {"enable_thinking": False}
    assert PC._template_kwargs("qwen38fn", "b") == {"enable_thinking": True}
    assert PC._template_kwargs("deepseek_v4", "b") == {}
    assert PC._template_kwargs("glm53", "b") == {}


def test_extension_aa_pins():
    assert PC.AA_PIN["q38fn"] == (56, "reasoning-xhigh", "measured")
    assert PC.AA_PIN["q35_397b"] == (34, "reasoning", "measured")
    assert PC.AA_PIN["dsv4_flash"] == (52, "reasoning-max", "measured")
    assert PC.AA_PIN["glm53"] == (60, "reasoning-max", "measured")
    assert PC.AA_PIN["dsv4_pro"] == (53, "reasoning-max", "measured")
    # Original pins untouched (spot pins on the measured set):
    assert PC.AA_PIN["q38_27b"] == (52, "reasoning-xhigh", "measured")
    assert PC.AA_PIN["q35_9b"] == (22, "reasoning", "measured")
    assert PC.AA_PIN["q36_27b"] == (38, "reasoning", "measured")


def test_sweep_layers_rule_43_to_78():
    """The kept layer-set rule: L<=32 dense 0..L-2; L>32 evens + top index."""
    cases = {
        43: ([*range(0, 42, 2), 42], 22),
        48: ([*range(0, 47, 2), 47], 25),
        60: ([*range(0, 59, 2), 59], 31),
        61: ([*range(0, 60, 2), 60], 31),
        78: ([*range(0, 77, 2), 77], 40),
        64: ([*range(0, 63, 2), 63], 33),  # pre-existing rows unchanged
        32: (list(range(0, 31)), 31),
    }
    for n_layers, (want, n) in cases.items():
        got = PC.sweep_layers(n_layers)
        assert got == want and len(got) == n, (n_layers, got)
        assert len(set(got)) == len(got)  # top index never duplicates an even
        assert max(got) == n_layers - 1 if n_layers > 32 else max(got) == n_layers - 2
