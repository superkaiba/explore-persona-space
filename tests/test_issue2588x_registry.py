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

import pytest

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
SAME_WIDTH_EXT_KEYS = ("q3_32b", "qwq_32b", "q25_32b", "o3_32b_t")


def test_registry_arithmetic_33_cells_36_maps():
    cells = PC.all_cells()
    assert len(cells) == 33
    assert sum(len(c.input_positions) for c in cells) == 36
    # The 9 larger-model extension cells are unchanged: 9 cells, 9 maps.
    ext = [c for c in cells if c.model_key in EXTENSION_KEYS]
    assert len(ext) == 9 and sum(len(c.input_positions) for c in ext) == 9
    # The same-width rows append AFTER every pre-existing row (order pin).
    keys = [c.model_key for c in cells]
    assert keys[-5:] == ["q3_32b", "q3_32b", "qwq_32b", "q25_32b", "o3_32b_t"]
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
        "q38fn": ("qwen38fn", ("a", "b"), 4, "Qwen/Qwen3.8-Flash-Next-FP8"),
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


def test_same_width_rows_contracts():
    """The 2026-09-02 same-width (h=5120, 64L) column extension rows."""
    expected = {
        # key: (family, arms, thinking, hf_id)
        "q3_32b": ("legacy_qwen3", ("a", "b"), True, "Qwen/Qwen3-32B"),
        "qwq_32b": ("qwq", ("b",), True, "Qwen/QwQ-32B"),
        "q25_32b": ("qwen25", ("a",), False, "Qwen/Qwen2.5-32B-Instruct"),
        "o3_32b_t": ("olmo_think", ("b",), True, "allenai/Olmo-3-32B-Think"),
    }
    for key, (family, arms, thinking, hf_id) in expected.items():
        m = PC.PANEL[key]
        assert (m.family, m.arms, m.thinking, m.hf_id) == (family, arms, thinking, hf_id)
        assert (m.n_layers, m.h_dim, m.tp_gpus) == (64, 5120, 1)
        assert not m.banked_arm_a and m.est_snapshot_gb is None  # fresh, dense bf16
        for arm in arms:
            cell = PC.cell_by_key(f"{key}_{arm}")
            assert cell.fresh
            if arm == "a":
                assert cell.parse_mode == "off"
                assert cell.input_positions == ("prompt_last",)
            elif family == "olmo_think":
                assert cell.parse_mode == "prefill"
                assert cell.input_positions == ("pre_think", "cot_boundary")
            else:
                assert cell.parse_mode == "prefill"
                assert cell.input_positions == ("cot_boundary",)
    # legacy_qwen3 rides the enable_thinking toggle but NOT the startswith("qwen3") tag.
    assert not "legacy_qwen3".startswith("qwen3")
    assert PC._template_kwargs("legacy_qwen3", "a") == {"enable_thinking": False}
    assert PC._template_kwargs("legacy_qwen3", "b") == {"enable_thinking": True}
    assert PC._template_kwargs("qwq", "b") == {}
    assert PC.LEGACY_QWEN3_THINK_PREFILL == "<think>\n"
    assert PC.SAME_WIDTH_KEYS[-4:] == SAME_WIDTH_EXT_KEYS
    assert all(PC.PANEL[k].h_dim == 5120 for k in PC.SAME_WIDTH_KEYS)
    assert PC.COLUMN_KEYS == ("q35_27b", "q36_27b", "q38_27b")  # plan-§5 column untouched


def test_same_width_aa_pins_measured():
    assert PC.AA_PIN["q3_32b"] == (11, "reasoning", "measured")
    assert PC.AA_PIN["q3_32b_nonreasoning"] == (8, "non-reasoning", "measured")
    assert PC.AA_PIN["qwq_32b"] == (13, "reasoning", "measured")
    assert PC.AA_PIN["q25_32b"] == (7, "non-reasoning", "measured")
    assert PC.AA_PIN["o3_32b_t"] == (6, "reasoning", "measured")


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


# ---------------------------------------------------------------------------
# Cap profiles (EPS_CAP_PROFILE, issue #2659 truncation rerun)
# ---------------------------------------------------------------------------

CAP_KEYS = {("a", "generic"), ("a", "gpqa"), ("b", "generic"), ("b", "gpqa")}
V1_CAP = {
    ("a", "generic"): 2048,
    ("a", "gpqa"): 2048,
    ("b", "generic"): 4096,
    ("b", "gpqa"): 8192,
}
LONG_CAP = {
    ("a", "generic"): 2048,
    ("a", "gpqa"): 8192,
    ("b", "generic"): 32768,
    ("b", "gpqa"): 65536,
}


def test_cap_profiles_table_and_default():
    """Every profile carries exactly the four (arm, surface) keys. v1 equals
    the original constants verbatim and is the default under a clean env."""
    assert set(PC.CAP_PROFILES) == {"v1", "long"}
    for name, table in PC.CAP_PROFILES.items():
        assert set(table) == CAP_KEYS, name
    assert PC.CAP_PROFILES["v1"] == V1_CAP
    assert PC.CAP_PROFILES["long"] == LONG_CAP
    assert PC.CAP_PROFILE == "v1"
    assert PC.CAP == V1_CAP
    assert set(PC.REGEN_CAP_CEILING) == set(PC.CAP_PROFILES)
    assert PC.REGEN_CAP_CEILING == {"v1": None, "long": 65536}
    with pytest.raises(ValueError):
        PC.resolve_cap_profile("bogus")


def test_cap_profile_prefix_suffixing():
    """v1 is the identity (pre-profile paths byte-identical). Any other
    profile appends _cap_<profile> so a rerun cannot write into v1 artifacts."""
    assert PC.profile_prefix("issue2588_capability_panel") == "issue2588_capability_panel"
    assert PC.PANEL_PREFIX == "issue2588_capability_panel"
    assert PC.G2_SENTINEL_PATH == "issue2588_capability_panel/gates/g2_anchor_pass.json"
    assert PC.cell_by_key("q3_32b_b").hf_prefix == "issue2588_capability_panel/q3_32b/think"
    assert (
        PC.profile_prefix("issue2588_capability_panel", "long")
        == "issue2588_capability_panel_cap_long"
    )
    with pytest.raises(ValueError):
        PC.profile_prefix("issue2588_capability_panel", "bogus")


def test_regen_bound_profile_derived_and_window_rule():
    """REGEN_MAX_MODEL_LEN_BOUND derives from the active profile (v1 keeps the
    original 23,488) and regen_cap clamps the doubled cap to the window and
    to the profile's regen ceiling."""
    expected = PC.PROMPT_TOKEN_BUDGET + 2 * max(PC.CAP.values())
    assert expected == PC.REGEN_MAX_MODEL_LEN_BOUND
    assert PC.REGEN_MAX_MODEL_LEN_BOUND == 23_488
    # Doubling allowed (fake mpe 262144: the Qwen3.5 window class).
    assert PC.regen_cap(32_768, 262_144) == 65_536
    # Clamped to the window but still above cap (fake mpe 40960: Qwen3-32B /
    # QwQ class): regen runs at mpe minus the prompt budget.
    clamped = PC.regen_cap(32_768, 40_960)
    assert clamped == 33_856
    assert 32_768 < clamped < 2 * 32_768
    # Window exhausted (regen_cap <= cap): the caller SKIPS the regen and
    # writes regen_skipped_reason "cap at context window".
    assert PC.regen_cap(33_856, 40_960) == 33_856
    assert PC.regen_skip_reason(33_856, 40_960) == "cap at context window"
    assert PC.regen_cap(16_384, PC.PROMPT_TOKEN_BUDGET + 16_384) == 16_384
    # v1 behavior unchanged: under the v1 prologue floor (mpe >= 23,488)
    # every v1 cap still doubles exactly, with a None ceiling.
    for cap in PC.CAP_PROFILES["v1"].values():
        assert PC.regen_cap(cap, 23_488, PC.REGEN_CAP_CEILING["v1"]) == 2 * cap
        assert PC.regen_skip_reason(cap, 23_488, None) is None
    # mpe floors per arm: the prompt budget plus min(largest registered cap,
    # MIN_EFFECTIVE_CAP), since registered caps above the window clamp down.
    assert PC.MIN_EFFECTIVE_CAP == 2048
    assert PC.mpe_floor_for_arm("a") == PC.PROMPT_TOKEN_BUDGET + 2048
    assert PC.mpe_floor_for_arm("b") == PC.PROMPT_TOKEN_BUDGET + 2048


def test_regen_ceiling_three_cases():
    """The long-profile regen ceiling (65,536): ceiling binds / window binds /
    neither binds, with the matching skip reason per case."""
    ceil = PC.REGEN_CAP_CEILING["long"]
    assert ceil == 65_536
    # Ceiling binds (long arm-b gpqa 65536 on a 262144-window model): the cap
    # already sits at the ceiling, regen never runs, residual accepted.
    assert PC.regen_cap(65_536, 262_144, ceil) == 65_536
    assert PC.regen_skip_reason(65_536, 262_144, ceil) == "cap at regen ceiling"
    # Window-clamped bump under a set ceiling: the skip reads as the 25-percent
    # bump floor (the window reason is reserved for ceiling-None profiles).
    assert PC.regen_cap(33_856, 40_960, ceil) == 33_856
    assert PC.regen_skip_reason(33_856, 40_960, ceil) == "regen bump below 25 percent"
    # Neither binds below cap (long arm-b generic 32768 on a 262144-window
    # model): regen runs once, to exactly the ceiling.
    assert PC.regen_cap(32_768, 262_144, ceil) == 65_536
    assert PC.regen_skip_reason(32_768, 262_144, ceil) is None
    # And an ordinary doubling far below the ceiling (long arm-a gpqa 8192).
    assert PC.regen_cap(8_192, 262_144, ceil) == 16_384
    assert PC.regen_skip_reason(8_192, 262_144, ceil) is None
    # The long engine bound follows: prompt budget + the ceiling.
    long_caps = PC.CAP_PROFILES["long"].values()
    assert max(min(2 * c, ceil) for c in long_caps) == ceil
    assert PC.PROMPT_TOKEN_BUDGET + ceil == 72_640


def test_cap_profile_env_threading_reload(monkeypatch):
    """EPS_CAP_PROFILE threads env -> table -> every PANEL_PREFIX-rooted path
    (end to end through a module reload, restored after)."""
    import importlib

    monkeypatch.setenv("EPS_CAP_PROFILE", "long")
    try:
        importlib.reload(PC)
        assert PC.CAP_PROFILE == "long"
        assert PC.CAP == LONG_CAP
        assert PC.PANEL_PREFIX == "issue2588_capability_panel_cap_long"
        assert PC.G2_SENTINEL_PATH == (
            "issue2588_capability_panel_cap_long/gates/g2_anchor_pass.json"
        )
        assert PC.cell_by_key("q3_32b_a").hf_prefix == (
            "issue2588_capability_panel_cap_long/q3_32b/nothink"
        )
        assert PC.REGEN_MAX_MODEL_LEN_BOUND == 7104 + 65_536
        assert PC.mpe_floor_for_arm("b") == 7104 + PC.MIN_EFFECTIVE_CAP
    finally:
        monkeypatch.delenv("EPS_CAP_PROFILE", raising=False)
        importlib.reload(PC)
    assert PC.CAP_PROFILE == "v1"
    assert PC.PANEL_PREFIX == "issue2588_capability_panel"


def test_cap_effective_window_clamp_and_bump_floor():
    """Per-model window clamp of the BASE cap + the 25-percent regen bump
    floor. v1 is a provable no-op: the v1 prologue floor (mpe >= 23,488)
    guarantees window room 16,384 >= every v1 cap, so iterating every panel
    cell at the WORST v1-permissible window leaves every cap unclamped."""
    # v1 no-op across the whole registry at the minimum v1-permissible mpe.
    for cell in PC.all_cells():
        for surface in ("generic", "gpqa"):
            eff = PC.cap_effective(cell.arm, surface, 23_488)
            assert eff == PC.CAP[(cell.arm, surface)], (cell.key, surface, eff)
    # Long profile on a 40,960-window model (Qwen3-32B / QwQ arm b): the
    # registered gpqa 65,536 clamps to the window, generic stays 32,768, and
    # the largest base engine pins max_model_len exactly at the window.
    long_t = PC.CAP_PROFILES["long"]
    assert PC.cap_effective("b", "generic", 40_960, table=long_t) == 32_768
    assert PC.cap_effective("b", "gpqa", 40_960, table=long_t) == 33_856
    assert PC.PROMPT_TOKEN_BUDGET + 33_856 == 40_960
    # Prologue floor under the long table: 40,960-window models PASS (the
    # floor asks only MIN_EFFECTIVE_CAP of room, since bigger caps clamp).
    floor_b_long = PC.PROMPT_TOKEN_BUDGET + min(
        max(long_t[("b", "generic")], long_t[("b", "gpqa")]), PC.MIN_EFFECTIVE_CAP
    )
    assert floor_b_long == 9_152
    assert floor_b_long <= 40_960
    # Bump floor at mpe 40,960 under the long ceiling: both arm-b stages skip
    # the regen with the bump-floor reason (no +1088-token engine rebuild).
    ceil = PC.REGEN_CAP_CEILING["long"]
    assert PC.REGEN_MIN_BUMP == 1.25
    assert PC.regen_cap(33_856, 40_960, ceil) == 33_856
    assert PC.regen_skip_reason(33_856, 40_960, ceil) == "regen bump below 25 percent"
    assert PC.regen_cap(32_768, 40_960, ceil) == 33_856  # +1088 only: below the floor
    assert PC.regen_skip_reason(32_768, 40_960, ceil) == "regen bump below 25 percent"
    # mpe 262,144 is unchanged from the previous round: full doubling for the
    # generic cap (to the ceiling) and the at-ceiling gpqa skip.
    assert PC.regen_cap(32_768, 262_144, ceil) == 65_536
    assert PC.regen_skip_reason(32_768, 262_144, ceil) is None
    assert PC.regen_skip_reason(65_536, 262_144, ceil) == "cap at regen ceiling"
    # v1 unaffected: a 2x bump always clears the 1.25x floor.
    for cap in PC.CAP_PROFILES["v1"].values():
        assert PC.regen_skip_reason(cap, 23_488, None) is None
