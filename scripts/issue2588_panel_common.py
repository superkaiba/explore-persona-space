#!/usr/bin/env python3
"""Issue #2588 panel commons: ported symbols + panel registry (plan v3 §4.6).

Port provenance (every ported function carries its own header):

- from ``scripts/issue2502_gen_capture.py`` @ a736aebb92 (read via
  ``git show a736aebb92:scripts/issue2502_gen_capture.py``, NEVER the live
  worktree file): ``assert_chat_template`` (generalized into the
  per-(family x arm) SideSpec contract table, plan §7 G1), ``render_prompt_ids``
  (incl. the transformers-5.x ``return_dict=False`` fix, commit aa8c19e3 —
  present at the pinned blob), ``assert_capture_position`` (L263),
  ``cap_hit_halt`` semantics (L921; here a fraction-returning counter — the
  #2588 driver routes the >2% trigger to the G4 regen path instead of rc=7).
- from ``scripts/issue2546_gen_capture.py`` @ 89680c72f9 (same ``git show``
  discipline; the issue-2546 working file is dirty under a live session):
  ``compute_read_idx`` (L554 — PROMPT-side reads ONLY: this panel uses modes
  ``prompt_last`` + ``pre_think``; the blob's ``assist_start`` mode and the
  arm-1 prefill-fallback rung are NOT carried — no arm-3 mode-pair exists in
  this panel), ``assert_think_pins`` (L591), ``segment_completion_arm``
  (L643), ``parse_generation`` (L686), ``extract_boxed`` / ``extract_mcq_letter``
  (L756) repointed at GPQA Diamond, and the END-OF-COT CAPTURE-ROW
  CONSTRUCTION from ``build_capture_row`` (L1252-1310):
  ``positions["cot_boundary"] = prompt_len + close_tok[1] - 1`` (L1289) via
  ``issue928_common.char_span_to_token_span`` (main,
  ``scripts/issue928_common.py:381``). The cot_boundary is a COMPLETION-side
  object (a ``KINDS_POST`` member at the blob's L320) and is NEVER produced by
  ``compute_read_idx`` (whose three modes all resolve inside ``prompt_ids``) —
  the v2 defect class this port exists to prevent (plan §4.3, MF1).

HARD asserts on every arm-b captured row (plan §4.3 / §7):
``cot_boundary >= prompt_len`` (completion-side in the CONCATENATED sequence),
``cot_boundary < n_total``, and the index equals the tokenized ``</think>``
close-boundary position (independent offset-containment re-check).
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent
REPO_ROOT = _SCRIPTS.parent

# Cross-script helper imports hoisted to module top so a missing symbol crashes
# at process start, never inside a smoke-skipped branch (gotchas.md #606).
import sys  # noqa: E402

if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

from issue928_common import char_span_to_token_span, repeated_4gram_fraction  # noqa: E402

# ---------------------------------------------------------------------------
# Registered constants (plan §0 / §4 / §10 / §11)
# ---------------------------------------------------------------------------

TASK_ID = 2588
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
PANEL_PREFIX = "issue2588_capability_panel"
G2_SENTINEL_PATH = f"{PANEL_PREFIX}/gates/g2_anchor_pass.json"

# #2330 pinned inputs (plan §10 Data)
MANIFEST_HF_PREFIX = "issue1491_scale_ladder/manifest"
MANIFEST_REVISION = "815ff6d976c686af8672b27cfdfb1ce6b419c02c"
SPLIT_IDS_PATH = REPO_ROOT / "eval_results" / "issue_2330" / "split_ids.json"
SPLIT_SHA256_PIN_PREFIXES = {
    "train_10k": "a74675bfed",
    "val_400": "61c7e6234e",
    "test_1000": "b1c32e2197",
}
EXPECTED_SPLIT_COUNTS = {
    "train_10k": 10000,
    "train_5k": 5000,
    "val_400": 400,
    "test_1000": 1000,
    "wc_test_1k": 998,
}

# Banked #2330 stores, consumed at the parent's record pin (plan §10)
BANKED_REVISION = "b99d86de23"
BANKED_CAP2048 = {
    "qwen35_9b": "issue2330_matched/qwen35_9b_cap2048",
    "qwen25_7b": "issue2330_matched/q25_cap2048",
}
BANKED_CEILING = {
    "qwen35_9b": "issue2330_matched/qwen35_9b/ceiling_draws",
    "qwen25_7b": "issue1491_scale_ladder/scale7_refit/ceiling_draws",
}
ANCHOR_GATE_PREFIX = "issue1491_scale_ladder/scale7_refit"

# Decoding (plan §10 fit/decoding constants; Source: #2330)
GEN_TEMP = 1.0
GEN_TOP_P = 0.95
GEN_SEED = 42
GPQA_ROLLOUT_SEEDS = (42, 43, 44, 45, 46)
CEILING_SEEDS = (43, 44)
PROMPT_TOKEN_BUDGET = 7104
# max_new_tokens per (arm, surface): no-think 2048; think-generic 4096;
# think-GPQA 8192 (plan §0; Sources: #2330 cap2048 / CLAUDE.md truncation rule
# + #1426, pilot-gated at smoke).
CAP = {
    ("a", "generic"): 2048,
    ("a", "gpqa"): 2048,
    ("b", "generic"): 4096,
    ("b", "gpqa"): 8192,
}
# Ceiling draws run at the arm's GENERIC cap (arm a 2048 = the §10 pinned
# fresh-cell 2048-cap; arm b 4096 so think blocks close — plan §10 states
# "fresh cells 2,048-cap" without an arm split; flagged in the implementation
# report as an interpretation, never a silent constant change).
CAP_HIT_TRIGGER = 0.02  # G4
UNCLOSED_THINK_TRIGGER = 0.02  # G5
REGEN_MAX_MODEL_LEN_BOUND = 23_488  # 7104 + 2*8192; asserted <= max_position_embeddings
REPEAT_4GRAM_MAX_FRAC = 0.50  # ported 2546 constant (parse drop class)

THINK_OPEN, THINK_CLOSE = "<think>", "</think>"
EMPTY_THINK = "<think>\n\n</think>"
OLMO_THINK_PREFILL_SUFFIX = "<|im_start|>assistant\n<think>"

TRANSFORMERS_FLOOR = "5.13.0"  # G6 (transformers PR #39847 / #46911; OLMo-core #685)

# Fit battery (Source: #2330 / #1491 cores)
ANCHOR_EXPECTED_R2 = 0.7250873220237553
ANCHOR_TOL = 1e-6  # plan §7 G2 (4 orders inside the parent's committed 0.01)
ANCHOR_LAYER = 19
PERM_DRAWS = 200
PERM_DRAW_BLOCK = 20
PERM_SEED = 42
PERM_DESCOPE_FLOOR = 50
BOOTSTRAP_DRAWS = 1000
BOOTSTRAP_SEED = 42
G2_SENTINEL_TIMEOUT_S = 45 * 60

GPQA_N_QUESTIONS = 198
GPQA_N_ROLLOUTS = 5
GPQA_OPTION_SHUFFLE_SEED = 42
GPQA_EXTRACTION_FAIL_TRIGGER = 0.05  # §4.5 flip condition -> judge fallback

# ---------------------------------------------------------------------------
# Panel registry (plan §4.1; configs live-verified 2026-08-25, §12 A8)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PanelModel:
    key: str
    hf_id: str
    n_layers: int
    h_dim: int
    family: str  # qwen35 | qwen36 | qwen38 | qwen25 | olmo_instruct | olmo_think
    arms: tuple[str, ...]  # generation/capture cells this model runs
    thinking: bool  # has a think mode (drives arm-b parse/caps)
    banked_arm_a: bool = False  # arm-a is capture-only over banked #2330 texts


PANEL: dict[str, PanelModel] = {
    m.key: m
    for m in (
        PanelModel("q35_0p8b", "Qwen/Qwen3.5-0.8B", 24, 1024, "qwen35", ("a", "b"), True),
        PanelModel("q35_2b", "Qwen/Qwen3.5-2B", 24, 2048, "qwen35", ("a", "b"), True),
        PanelModel("q35_4b", "Qwen/Qwen3.5-4B", 32, 2560, "qwen35", ("a", "b"), True),
        PanelModel(
            "q35_9b", "Qwen/Qwen3.5-9B", 32, 4096, "qwen35", ("a", "b"), True, banked_arm_a=True
        ),
        PanelModel("q35_27b", "Qwen/Qwen3.5-27B", 64, 5120, "qwen35", ("a", "b"), True),
        PanelModel("q36_27b", "Qwen/Qwen3.6-27B", 64, 5120, "qwen36", ("a", "b"), True),
        PanelModel("q38_27b", "Qwen/Qwen3.8-27B", 64, 5120, "qwen38", ("a", "b"), True),
        PanelModel(
            "o3_7b_i", "allenai/Olmo-3-7B-Instruct", 32, 4096, "olmo_instruct", ("a",), False
        ),
        PanelModel("o3_7b_t", "allenai/Olmo-3-7B-Think", 32, 4096, "olmo_think", ("b",), True),
        PanelModel(
            "o31_32b_i", "allenai/Olmo-3.1-32B-Instruct", 64, 5120, "olmo_instruct", ("a",), False
        ),
        PanelModel("o31_32b_t", "allenai/Olmo-3.1-32B-Think", 64, 5120, "olmo_think", ("b",), True),
        PanelModel(
            "q25_7b",
            "Qwen/Qwen2.5-7B-Instruct",
            28,
            3584,
            "qwen25",
            ("a",),
            False,
            banked_arm_a=True,
        ),
    )
}

# The fixed-size capability column (plan §5) + AA §5 pin table (P0 re-verifies).
COLUMN_KEYS = ("q35_27b", "q36_27b", "q38_27b")
AA_PIN = {
    # key: (value, mode, measured|estimated). 52 is AA's "xhigh" reasoning-effort
    # configuration (plan §5); the MEASURED set is exactly three values.
    "q35_0p8b": (5, "reasoning", "estimated"),
    "q35_2b": (7, "reasoning", "estimated"),
    "q35_4b": (20, "reasoning", "estimated"),
    "q35_9b": (22, "reasoning", "measured"),
    "q35_9b_nonreasoning": (21, "non-reasoning", "estimated"),
    "q35_27b": (35, "reasoning", "estimated"),
    "q36_27b": (38, "reasoning", "measured"),
    "q38_27b": (52, "reasoning-xhigh", "measured"),
    "o3_7b_i": (2, "non-reasoning", "estimated"),
    "o3_7b_t": (4, "reasoning", "estimated"),
    "o31_32b_i": (6, "non-reasoning", "estimated"),
    "o31_32b_t": (8, "reasoning", "estimated"),
    "q25_7b": (None, "n/a", "no-AA-value"),  # page 404s at the expected slug (P0 retries)
}


def sweep_layers(n_layers: int) -> list[int]:
    """Swept block-output indices (plan §4.4 layer-sweep sets).

    L <= 32: every block output 0..L-2 (the #2330 dense-sweep convention —
    indices 0-30 of the 32-layer 9B; 23 units at 24L, 27 at 28L, 31 at 32L).
    64L: even indices 0,2,..,62 plus the top index 63 (33 units); the 31
    unswept odd layers 1..61 are the odd-layer sensitivity second pass's set.
    Reproduces the plan §9 unit arithmetic exactly (92+186+27+264 = 569).
    """
    if n_layers <= 32:
        return list(range(0, n_layers - 1))
    return list(range(0, n_layers - 1, 2)) + [n_layers - 1]


def odd_sensitivity_layers(n_layers: int) -> list[int]:
    """The unswept odd layers for the 64-layer odd-layer sensitivity pass."""
    assert n_layers > 32, n_layers
    swept = set(sweep_layers(n_layers))
    return sorted(set(range(n_layers)) - swept)


@dataclass(frozen=True)
class Cell:
    """One generation/capture cell (plan §4.1: 19 cells, 21 registered maps)."""

    model_key: str
    arm: str  # a | b
    fresh: bool  # False = capture-only over banked #2330 cap2048 texts

    @property
    def model(self) -> PanelModel:
        return PANEL[self.model_key]

    @property
    def key(self) -> str:
        return f"{self.model_key}_{self.arm}"

    @property
    def parse_mode(self) -> str:
        """Completion parse mode (ported 2546 segment semantics)."""
        m = self.model
        if self.arm == "a" or not m.thinking:
            return "off"
        return "prefill" if m.family == "olmo_think" else "emergent"

    @property
    def input_positions(self) -> tuple[str, ...]:
        """Registered map-input positions for this cell (plan §4.3)."""
        m = self.model
        if self.arm == "a" or not m.thinking:
            return ("prompt_last",)
        if m.family == "olmo_think":
            return ("pre_think", "cot_boundary")  # dual-position capture (MF2)
        return ("cot_boundary",)

    @property
    def hf_prefix(self) -> str:
        arm_dir = "nothink" if self.arm == "a" else "think"
        return f"{PANEL_PREFIX}/{self.model_key}/{arm_dir}"


def all_cells() -> list[Cell]:
    """The 19 generation/capture cells (plan §4.1)."""
    cells: list[Cell] = []
    for m in PANEL.values():
        for arm in m.arms:
            fresh = not (arm == "a" and m.banked_arm_a)
            cells.append(Cell(m.key, arm, fresh))
    assert len(cells) == 19, [c.key for c in cells]
    n_maps = sum(len(c.input_positions) for c in cells)
    assert n_maps == 21, n_maps
    return cells


def cell_by_key(key: str) -> Cell:
    for c in all_cells():
        if c.key == key:
            return c
    raise KeyError(f"unknown cell {key!r}; known: {[c.key for c in all_cells()]}")


# ---------------------------------------------------------------------------
# G6: transformers floor + OLMo structural rope probe (plan §7 G6)
# ---------------------------------------------------------------------------


def assert_transformers_floor(min_version: str = TRANSFORMERS_FLOOR) -> str:
    """HARD assert transformers >= 5.13.0 (a version print is NOT the gate).

    transformers 5.5.3-5.12.x compute silently WRONG Olmo3 forward passes under
    YARN rope_scaling — in HF capture AND vLLM 0.27.1 generation (vLLM parses
    rope from the installed transformers' Olmo3Config.rope_parameters).
    """
    import transformers
    from packaging.version import Version

    got = transformers.__version__
    assert Version(got) >= Version(min_version), (
        f"G6 FAIL: transformers=={got} < {min_version} — the 5.5.3-5.12.x range computes "
        "INCORRECT Olmo3 forward passes under YARN rope_scaling (PR #39847 regression, fixed "
        "in 5.13.0 via PR #46911). Rebuild the venv per the plan §10 recipe; never proceed "
        "on a version print."
    )
    return got


def assert_olmo_rope_split(model_id: str) -> dict:
    """Structural per-layer-type rope_parameters probe on an OLMo id (G6).

    Asserts the exact contract vLLM v0.27.1's olmo3.py L139-147 consumes
    (``rope_parameters.get(attn_type, rope_parameters)``): a PER-LAYER-TYPE
    dict with full_attention==yarn and sliding_attention==default. A FLAT dict
    (transformers < 5.13.0) hands YaRN + the 1.2079 mscale to 3-of-4 sliding
    layers silently. Also guards future >5.13 API drift.
    """
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    rp = getattr(cfg, "rope_parameters", None)
    assert isinstance(rp, dict), f"G6 FAIL ({model_id}): rope_parameters is {type(rp)}, not dict"
    assert set(rp) >= {"full_attention", "sliding_attention"}, (
        f"G6 FAIL ({model_id}): rope_parameters keys {sorted(rp)} — FLAT dict (the pre-5.13 "
        "shape): every sliding layer would silently receive YaRN interpolation."
    )
    full_t = rp["full_attention"].get("rope_type")
    slide_t = rp["sliding_attention"].get("rope_type")
    assert full_t == "yarn", f"G6 FAIL ({model_id}): full_attention rope_type={full_t!r} != yarn"
    assert slide_t == "default", (
        f"G6 FAIL ({model_id}): sliding_attention rope_type={slide_t!r} != default"
    )
    return {"model": model_id, "full_attention": full_t, "sliding_attention": slide_t}


def assert_max_position_embeddings(model_id: str, floor: int = REGEN_MAX_MODEL_LEN_BOUND) -> int:
    """Record + HARD-assert max_position_embeddings >= the G4/G5 regen headroom."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(model_id)
    mpe = getattr(cfg, "max_position_embeddings", None)
    if mpe is None and getattr(cfg, "text_config", None) is not None:
        mpe = getattr(cfg.text_config, "max_position_embeddings", None)
    assert isinstance(mpe, int), f"{model_id}: max_position_embeddings unreadable ({mpe!r})"
    assert mpe >= floor, (
        f"G4/G5 regen headroom FAIL: {model_id} max_position_embeddings={mpe} < {floor} — the "
        "regen engine re-pin (max_model_len = 7104 + 2*cap) would exceed the context window."
    )
    return mpe


# ---------------------------------------------------------------------------
# G1: per-(family x arm) template SideSpec contracts (plan §7 G1, MF7)
# ---------------------------------------------------------------------------


def _template_kwargs(family: str, arm: str) -> dict:
    """apply_chat_template kwargs per (family, arm)."""
    if family in ("qwen35", "qwen36", "qwen38"):
        return {"enable_thinking": arm != "a"}
    return {}  # qwen25 / olmo templates carry no enable_thinking toggle


def render_probe(tok, family: str, arm: str, text: str = "ping") -> str:
    """Text render of ONE user turn under the (family, arm) template kwargs."""
    return tok.apply_chat_template(
        [{"role": "user", "content": text}],
        tokenize=False,
        add_generation_prompt=True,
        **_template_kwargs(family, arm),
    )


def assert_template_sidespec(tok, family: str, arm: str) -> str:
    """G1 SideSpec contract on a probe render; returns sha256[:16] of it.

    Ported from issue2502_gen_capture.assert_chat_template @ a736aebb92 L343
    (the pinned Qwen empty-<think> contract), generalized to the plan's
    per-(family x arm) table:

    - qwen* arm a  -> the literal closed empty ``<think>\\n\\n</think>`` present;
    - qwen* arm b  -> NO think delimiters in the render (emergent parse mode);
    - olmo_instruct (both arms) -> NO think delimiters anywhere;
    - olmo_think arm b -> render ENDS with ``<|im_start|>assistant\\n<think>``
      (prefill parse mode; the generated ``</think>`` is the read boundary).
    """
    probe = render_probe(tok, family, arm)
    if family in ("qwen35", "qwen36", "qwen38"):
        if arm == "a" and EMPTY_THINK not in probe:
            raise RuntimeError(
                f"G1 FAIL ({family}, arm a): enable_thinking=False did not render the empty "
                f"think block {EMPTY_THINK!r} (template drift vs the #2502/#2378 pin)."
            )
        if arm == "b" and (THINK_OPEN in probe or THINK_CLOSE in probe):
            raise RuntimeError(
                f"G1 FAIL ({family}, arm b): think delimiters present in the PROMPT render — "
                "the emergent parse mode requires the model to open its own block."
            )
    elif family in ("olmo_instruct", "qwen25"):
        if THINK_OPEN in probe or THINK_CLOSE in probe:
            raise RuntimeError(
                f"G1 FAIL ({family}, arm {arm}): think delimiters present in a non-thinking "
                "template render."
            )
    elif family == "olmo_think":
        if arm != "b":
            raise RuntimeError("olmo_think runs arm (b) only (plan §4.1)")
        if not probe.rstrip("\n").endswith(OLMO_THINK_PREFILL_SUFFIX):
            raise RuntimeError(
                f"G1 FAIL (olmo_think): render does not end with the pre-opened "
                f"{OLMO_THINK_PREFILL_SUFFIX!r} (prefill parse mode premise, §12 A14)."
            )
    else:
        raise ValueError(f"unknown family {family!r}")
    return hashlib.sha256(probe.encode("utf-8")).hexdigest()[:16]


def render_prompt_ids(tok, text: str, family: str, arm: str) -> list[int]:
    """Render ONE user turn to prompt token ids (add_generation_prompt=True).

    Ported from issue2502_gen_capture.render_prompt_ids @ a736aebb92 L364,
    INCLUDING the transformers-5.x fix (commit aa8c19e3): pass
    ``return_dict=False`` explicitly — 5.x flips apply_chat_template
    (tokenize=True) to default return_dict=True (a BatchEncoding whose KEYS the
    listcomp would int()). The Qwen no-think empty-block contract is asserted
    on EVERY render (the #2502 per-render discipline); olmo_think's prefill
    suffix likewise.
    """
    kwargs = _template_kwargs(family, arm)
    msgs = [{"role": "user", "content": text}]
    if (family.startswith("qwen3") and arm == "a") or family == "olmo_think":
        rendered = tok.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True, **kwargs
        )
        if family.startswith("qwen3") and EMPTY_THINK not in rendered:
            raise RuntimeError(
                "G1 FAIL on render: empty think block absent (context digest "
                f"{hashlib.sha256(text.encode()).hexdigest()[:12]})"
            )
        if family == "olmo_think" and not rendered.rstrip("\n").endswith(OLMO_THINK_PREFILL_SUFFIX):
            raise RuntimeError(
                "G1 FAIL on render: olmo_think prefill suffix absent (context digest "
                f"{hashlib.sha256(text.encode()).hexdigest()[:12]})"
            )
    ids = tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, return_dict=False, **kwargs
    )
    return [int(x) for x in ids]


def assert_think_pins(tok, family: str) -> dict[str, tuple[int, ...]]:
    """Pinned think-delimiter encodings on this tokenizer.

    Ported from issue2546_gen_capture.assert_think_pins @ 89680c72f9 L591,
    adapted: the pins are RESOLVED here (recorded by the caller) rather than
    compared against hardcoded per-model constants — the panel spans 12
    tokenizers, and the P0 render probes + the per-row close-boundary asserts
    are the drift guards. Non-thinking families return {}.
    """
    if family in ("olmo_instruct", "qwen25"):
        return {}
    open_ids = tuple(tok.encode(THINK_OPEN, add_special_tokens=False))
    close_ids = tuple(tok.encode(THINK_CLOSE, add_special_tokens=False))
    assert open_ids and close_ids, (family, open_ids, close_ids)
    return {"open_ids": open_ids, "close_ids": close_ids}


# ---------------------------------------------------------------------------
# Prompt-side read index (ported 2546 compute_read_idx @ 89680c72f9 L554)
# ---------------------------------------------------------------------------


def _find_last_subseq(ids: list[int], sub: tuple[int, ...]) -> int:
    """Ported verbatim from the pinned blob (L547)."""
    n, m = len(ids), len(sub)
    for start in range(n - m, -1, -1):
        if tuple(ids[start : start + m]) == sub:
            return start
    return -1


def compute_read_idx(
    read_point: str, prompt_ids: list[int], *, open_ids: tuple[int, ...] | None = None
) -> int:
    """Registered PROMPT-side v_C read point, as an index into prompt_ids.

    Ported from issue2546_gen_capture.compute_read_idx @ 89680c72f9 L554.
    prompt_last -> last generation-prompt token (arm a; #2330/#779 convention).
    pre_think   -> last token PRECEDING the prefilled <think> (OLMo-Think; the
                   #1005 prefill-rung convention; same input-position semantics
                   as Instruct prompt_last — the OLMo-R comparability premise).
    The blob's assist_start mode + prefill-fallback rung are NOT carried (no
    arm-3 mode-pair in this panel). This function NEVER produces the
    end-of-CoT index — that is ``build_capture_row_2588``'s completion-side
    ``cot_boundary`` (plan §4.3).
    """
    if read_point == "prompt_last":
        return len(prompt_ids) - 1
    if read_point == "pre_think":
        assert open_ids is not None, "pre_think read needs the pinned think-open ids"
        start = _find_last_subseq(prompt_ids, open_ids)
        assert start > 0, (
            f"pre_think read point: think-open ids {open_ids} not found in the generation "
            f"prompt (len {len(prompt_ids)}) — template drifted, refusing"
        )
        return start - 1
    raise ValueError(f"unknown prompt-side read point {read_point!r}")


# ---------------------------------------------------------------------------
# Completion segmentation (ported 2546 @ 89680c72f9 L643/L686)
# ---------------------------------------------------------------------------


def _strip_span(text: str, s: int, e: int) -> tuple[int, int]:
    while s < e and text[s].isspace():
        s += 1
    while e > s and text[e - 1].isspace():
        e -= 1
    return s, e


def segment_completion_arm(
    text: str, mode: str
) -> tuple[bool, str, tuple[int, int], tuple[int, int]]:
    """(well_formed, reason, cot_char_span, ans_char_span) per parse mode.

    Ported verbatim from issue2546_gen_capture.segment_completion_arm
    @ 89680c72f9 L643. emergent: exactly one <think> (only whitespace before
    it) + one </think>, open before close (Qwen think). prefill: the prompt
    carries the <think>; well-formed iff exactly one </think> and NO open tag
    in the completion (OLMo-Think). off: no think block — the answer span is
    the whole generated text.
    """
    if mode == "off":
        s, e = _strip_span(text, 0, len(text))
        return (e > s), ("empty_answer" if e <= s else ""), (0, 0), (s, e)
    n_open, n_close = text.count(THINK_OPEN), text.count(THINK_CLOSE)
    zero = ((0, 0), (0, 0))
    if mode == "prefill":
        if n_close != 1:
            return False, f"close_count_{n_close}", *zero
        if n_open != 0:
            return False, "unexpected_open_tag", *zero
        c = text.index(THINK_CLOSE)
        cot = _strip_span(text, 0, c)
    elif mode == "emergent":
        if n_open != 1 or n_close != 1:
            return False, f"open{n_open}_close{n_close}", *zero
        o, c = text.index(THINK_OPEN), text.index(THINK_CLOSE)
        if c < o:
            return False, "close_before_open", *zero
        if text[:o].strip():
            return False, "text_before_open", *zero
        cot = _strip_span(text, o + len(THINK_OPEN), c)
    else:
        raise ValueError(f"unknown parse mode {mode!r}")
    ans = _strip_span(text, c + len(THINK_CLOSE), len(text))
    if cot[1] <= cot[0]:
        return False, "empty_think", cot, ans
    if ans[1] <= ans[0]:
        return False, "empty_answer", cot, ans
    return True, "", cot, ans


def parse_generation(row: dict, mode: str) -> dict:
    """Parse one rollout row -> parse record (drop-and-count classes).

    Ported from issue2546_gen_capture.parse_generation @ 89680c72f9 L686
    (identical drop classes: truncated_no_close / truncated_residual /
    degenerate_repetition via issue928_common.repeated_4gram_fraction).
    """
    text, fr = row["text"], row["finish_reason"]
    wf, reason, cot, ans = segment_completion_arm(text, mode)
    if not wf and fr == "length" and mode != "off" and THINK_CLOSE not in text:
        reason = "truncated_no_close"
    if mode == "off" and fr == "length":
        wf, reason = False, "truncated_residual"
    rep = repeated_4gram_fraction(text)
    if wf and rep > REPEAT_4GRAM_MAX_FRAC:
        wf, reason = False, "degenerate_repetition"
    if wf and mode != "off" and fr == "length":
        wf, reason = False, "truncated_residual"
    return {
        "well_formed": wf,
        "reason": reason,
        "cot_char_span": list(cot),
        "ans_char_span": list(ans),
        "rep_frac": rep,
        "finish_reason": fr,
    }


# ---------------------------------------------------------------------------
# Capture-row construction (ported 2546 build_capture_row @ 89680c72f9 L1252)
# ---------------------------------------------------------------------------


def build_capture_row_2588(
    tok, wrow: dict, *, positions_wanted: tuple[str, ...]
) -> tuple[dict | None, str]:
    """One teacher-forced row: ids + spans + positions, or (None, counted reason).

    Ported from issue2546_gen_capture.build_capture_row @ 89680c72f9
    L1252-1310. The teacher-forced input is prompt_ids + completion_ids
    CONCATENATED AS IDS (never a re-tokenize of the concatenated string — the
    #1092 seam rule); token spans derive from ``return_offsets_mapping`` over
    the completion text (#825 zero-width spans drop with a counted reason).

    The end-of-CoT read is the pinned blob's L1289 construction:
    ``positions["cot_boundary"] = prompt_len + close_tok[1] - 1`` — a
    COMPLETION-side index in the concatenated sequence. HARD asserts (plan
    §4.3 / §7 "arm-b read is completion-side"): cot_boundary >= prompt_len,
    cot_boundary < n_total, and the index equals the tokenized close-boundary
    position (independent offset-containment re-check of the token holding the
    final ``>`` of ``</think>``).

    ``wrow`` keys: row_id, prompt (rendered prompt TEXT), n_prompt_tokens
    (recorded at generation from the SAME render), text (completion),
    ans_char_span, cot_char_span, read_points ({name: prompt-side idx} for
    prompt_last / pre_think, computed by compute_read_idx at generation time).
    """
    prompt_ids = tok(wrow["prompt"], add_special_tokens=False)["input_ids"]
    assert len(prompt_ids) == wrow["n_prompt_tokens"], (
        f"{wrow['row_id']}: prompt re-tokenization drifted "
        f"({len(prompt_ids)} != {wrow['n_prompt_tokens']})"
    )
    enc = tok(wrow["text"], add_special_tokens=False, return_offsets_mapping=True)
    comp_ids, offsets = enc["input_ids"], enc["offset_mapping"]
    if not comp_ids:
        return None, "empty_completion_tokens"
    prompt_len = len(prompt_ids)
    ans_tok = char_span_to_token_span(offsets, tuple(wrow["ans_char_span"]))
    if ans_tok == (0, 0):
        return None, "empty_ans_token_span"
    spans = {"ans": (prompt_len + ans_tok[0], prompt_len + ans_tok[1])}

    positions: dict[str, int] = {}
    for name in positions_wanted:
        if name == "cot_boundary":
            continue
        idx = wrow["read_points"][name]
        assert 0 <= idx < prompt_len, (name, idx, prompt_len)
        positions[name] = int(idx)

    if "cot_boundary" in positions_wanted:
        close_char = wrow["text"].index(THINK_CLOSE) + len(THINK_CLOSE) - 1
        close_tok = char_span_to_token_span(offsets, (close_char, close_char + 1))
        if close_tok == (0, 0):
            return None, "empty_close_token_span"
        cot_boundary = prompt_len + close_tok[1] - 1
        # HARD asserts (plan §7 row "arm-b read is completion-side"):
        assert cot_boundary >= prompt_len, (
            f"{wrow['row_id']}: cot_boundary {cot_boundary} < prompt_len {prompt_len} — a "
            "prompt-side end-of-CoT read is the v2 defect class (H2 nulled by construction)"
        )
        # Independent re-check: the completion token at close_tok[1]-1 must
        # CONTAIN the final char of </think> in its offset span.
        s_off, e_off = offsets[close_tok[1] - 1]
        assert s_off <= close_char < e_off, (
            f"{wrow['row_id']}: cot_boundary token offsets ({s_off},{e_off}) do not contain "
            f"the </think> close char {close_char} — offset-mapping drift"
        )
        positions["cot_boundary"] = int(cot_boundary)

    full_len = prompt_len + len(comp_ids)
    for name, (s, e) in spans.items():
        assert 0 <= s < e <= full_len, (name, s, e, full_len)
    for name, p in positions.items():
        assert 0 <= p < full_len, (name, p, full_len)
    return {
        "row_id": wrow["row_id"],
        "prompt_ids": [int(x) for x in prompt_ids],
        "comp_ids": [int(x) for x in comp_ids],
        "spans": spans,
        "positions": positions,
    }, ""


def assert_capture_position(ids_row, mask_row, prompt_ids, *, row_key: str) -> int:
    """Per-row prompt_last capture-position assert; returns the selected index.

    Ported verbatim (modulo naming) from issue2502_gen_capture
    .assert_capture_position @ a736aebb92 L263: (i) the context-segment
    attention-mask sum minus 1 equals the selected index (fails under left
    padding or a mask hole), (ii) the prompt segment of the materialized ids
    equals the separately rendered prompt ids, (iii) the selected token id
    equals the rendered prompt's final token id.
    """
    prompt_l = [int(x) for x in prompt_ids]
    n_prompt = len(prompt_l)
    if n_prompt < 1:
        raise RuntimeError(f"[g1] row {row_key}: empty rendered prompt")
    sel = n_prompt - 1
    ids_l = [int(x) for x in ids_row]
    mask_l = [int(x) for x in mask_row]
    ctx_mask_sum = sum(mask_l[:n_prompt])
    if ctx_mask_sum - 1 != sel:
        raise RuntimeError(
            f"[g1] row {row_key}: context attention-mask sum-1 ({ctx_mask_sum - 1}) != "
            f"selected_index ({sel}) — left padding or mask hole in the context segment"
        )
    if ids_l[:n_prompt] != prompt_l:
        raise RuntimeError(
            f"[g1] row {row_key}: materialized prompt ids != separately rendered prompt ids"
        )
    if ids_l[sel] != prompt_l[-1]:
        raise RuntimeError(
            f"[g1] row {row_key}: selected token id {ids_l[sel]} != final rendered prompt "
            f"token id {prompt_l[-1]}"
        )
    return sel


# ---------------------------------------------------------------------------
# GPQA Diamond: exact-match extraction (ported 2546 @ 89680c72f9 L756/L786)
# ---------------------------------------------------------------------------

_BOXED_RE = re.compile(r"\\boxed\s*\{")
_LETTER_RE = re.compile(r"\b([A-J])\b")
_MCQ_ANCHOR_RE = re.compile(
    r"(?:answer|option|choice)\s*(?:is|:)?\s*\**\(?([A-J])\)?(?![A-Za-z])", re.IGNORECASE
)
_MCQ_BARE_LINE_RE = re.compile(r"^\**\(?([A-J])\)?[\s.):*]*$")


def extract_boxed(text: str) -> str | None:
    """Content of the LAST \\boxed{...} (brace-balanced). Ported 2546 L730."""
    last = None
    for m in _BOXED_RE.finditer(text):
        depth, i = 1, m.end()
        while i < len(text) and depth > 0:
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
            i += 1
        if depth == 0:
            last = text[m.end() : i - 1]
    return last


def extract_mcq_letter(ans_text: str) -> str | None:
    """Anchored MCQ letter extraction — never a first-match bare letter.

    Ported verbatim from issue2546_gen_capture.extract_mcq_letter
    @ 89680c72f9 L756 (priority: last boxed -> last "answer is X" anchor ->
    last bare-letter LINE -> last standalone letter in the final non-empty
    line; None when nothing anchors — scored incorrect, never a guess).
    Repointed at GPQA Diamond (A-D golds; the [A-J] class is kept verbatim).
    """
    boxed = extract_boxed(ans_text)
    if boxed is not None:
        b = boxed.strip().strip("()*. ").upper()
        if len(b) == 1 and "A" <= b <= "J":
            return b
    anchors = _MCQ_ANCHOR_RE.findall(ans_text)
    if anchors:
        return anchors[-1].upper()
    lines = [ln.strip() for ln in ans_text.strip().splitlines() if ln.strip()]
    for ln in reversed(lines):
        m = _MCQ_BARE_LINE_RE.match(ln)
        if m:
            return m.group(1)
    if lines:
        tail = _LETTER_RE.findall(lines[-1])
        if tail:
            return tail[-1]
    return None


def gpqa_letter_correct(ans_text: str, gold: str) -> tuple[bool, str | None]:
    """(exact_match_correct, extracted_letter) for one GPQA rollout.

    The MCQ branch of issue2546_gen_capture.exact_match_correct @ 89680c72f9
    L786, repointed at GPQA Diamond: unparseable extraction scores INCORRECT
    (never a guess) and is counted toward the §4.5 judge-fallback trigger.
    """
    letter = extract_mcq_letter(ans_text)
    return (letter == gold.strip().upper() if letter is not None else False), letter


# ---------------------------------------------------------------------------
# GPQA Diamond staging (dual route, plan §4.2) + rendering
# ---------------------------------------------------------------------------


def _norm_q(text: str) -> str:
    """Whitespace-normalized join key (Route B join, plan §4.2)."""
    return " ".join(str(text).split())


def stage_gpqa_diamond(cache_dir: Path) -> tuple[list[dict], str]:
    """Stage GPQA Diamond rows via the registered dual route.

    Route A (PREFERRED): the canonical ``Idavidrein/gpqa`` ->
    ``gpqa_diamond.csv``. The deterministic fallback trigger is
    GatedRepoError/403 on that download -> Route B: reconstruct Diamond-198
    from ``hendrydong/gpqa_diamond`` (198 rows; problem/solution/domain) JOINED
    to ``ankner/gpqa`` (ungated 80-col original schema, main-448 grain) on
    whitespace-normalized question text. HARD asserts (plan §4.2): exactly 198
    joined rows; 0 misses; all three distractors non-empty per row;
    cross-source correct-answer agreement.

    Returns (rows, route) with rows carrying
    {question, correct, incorrect: [3], domain?}.
    """
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import GatedRepoError, HfHubHTTPError

    try:
        csv_path = hf_hub_download(
            "Idavidrein/gpqa", "gpqa_diamond.csv", repo_type="dataset", cache_dir=str(cache_dir)
        )
        rows = _parse_canonical_gpqa_csv(Path(csv_path))
        assert len(rows) == GPQA_N_QUESTIONS, f"canonical CSV rows {len(rows)} != 198"
        return rows, "A-canonical"
    except GatedRepoError:
        pass  # deterministic Route-B trigger (plan §4.2)
    except HfHubHTTPError as e:
        if getattr(getattr(e, "response", None), "status_code", None) != 403:
            raise
    return _stage_gpqa_route_b(cache_dir), "B-reconstructed"


def _parse_canonical_gpqa_csv(path: Path) -> list[dict]:
    """Parse the canonical gpqa_diamond.csv (original 80-col schema)."""
    import csv as _csv

    rows: list[dict] = []
    with path.open(encoding="utf-8", newline="") as fh:
        for rec in _csv.DictReader(fh):
            rows.append(
                {
                    "question": rec["Question"],
                    "correct": rec["Correct Answer"],
                    "incorrect": [
                        rec["Incorrect Answer 1"],
                        rec["Incorrect Answer 2"],
                        rec["Incorrect Answer 3"],
                    ],
                    "domain": rec.get("Subdomain") or rec.get("High-level domain") or "",
                }
            )
    return rows


def _stage_gpqa_route_b(cache_dir: Path) -> list[dict]:
    """Route B reconstruction join (MEASURED at plan time: hit=198, miss=0)."""
    from datasets import load_dataset

    hd = load_dataset("hendrydong/gpqa_diamond", cache_dir=str(cache_dir))
    hd_rows = [r for split in hd.values() for r in split]
    assert len(hd_rows) == GPQA_N_QUESTIONS, (
        f"hendrydong/gpqa_diamond rows {len(hd_rows)} != 198 — reconstruction premise broken"
    )
    ank = load_dataset("ankner/gpqa", cache_dir=str(cache_dir))
    ank_rows = [r for split in ank.values() for r in split]
    ank_by_q = {_norm_q(r["Question"]): r for r in ank_rows}

    rows: list[dict] = []
    misses: list[str] = []
    for r in hd_rows:
        key = _norm_q(r["problem"])
        src = ank_by_q.get(key)
        if src is None:
            misses.append(key[:80])
            continue
        incorrect = [
            str(src["Incorrect Answer 1"]),
            str(src["Incorrect Answer 2"]),
            str(src["Incorrect Answer 3"]),
        ]
        assert all(x.strip() for x in incorrect), f"empty distractor on joined row: {key[:80]}"
        # Cross-source correct-answer agreement (two mirrors replace one canonical file).
        assert _norm_q(r["solution"]) == _norm_q(src["Correct Answer"]), (
            f"cross-source correct-answer DISAGREEMENT on: {key[:80]}"
        )
        rows.append(
            {
                "question": str(src["Question"]),
                "correct": str(src["Correct Answer"]),
                "incorrect": incorrect,
                "domain": str(r.get("domain", "")),
            }
        )
    assert not misses, f"Route B join misses ({len(misses)}): {misses[:3]}"
    assert len(rows) == GPQA_N_QUESTIONS, f"Route B joined rows {len(rows)} != 198"
    return rows


def render_gpqa_prompts(rows: list[dict], seed: int = GPQA_OPTION_SHUFFLE_SEED) -> list[dict]:
    """Freeze the 198 rendered GPQA prompts (seed-42 option shuffle, plan §4.2).

    Deterministic: rows are ordered by normalized question text, then ONE rng
    seeded ``seed`` shuffles each question's 4 options in that order. The
    rendered text is identical for every model; gold = the shuffled position
    of the correct answer.
    """
    import random

    rng = random.Random(seed)
    out: list[dict] = []
    for qi, row in enumerate(sorted(rows, key=lambda r: _norm_q(r["question"]))):
        options = [("correct", row["correct"])] + [("incorrect", x) for x in row["incorrect"]]
        rng.shuffle(options)
        letters = ["A", "B", "C", "D"]
        gold = letters[[k for k, _ in options].index("correct")]
        lines = [row["question"].strip(), ""]
        for letter, (_, text) in zip(letters, options, strict=True):
            lines.append(f"{letter}. {text.strip()}")
        lines.append("")
        lines.append(
            "Answer with the letter of the correct option (A, B, C, or D), "
            "followed by a brief justification."
        )
        out.append(
            {
                "qid": f"gpqa_{qi:03d}",
                "prompt": "\n".join(lines),
                "gold": gold,
                "domain": row.get("domain", ""),
            }
        )
    assert len(out) == GPQA_N_QUESTIONS
    return out


# ---------------------------------------------------------------------------
# Conditional GPQA extraction judge fallback (plan §4.5, pre-registered MF11)
# ---------------------------------------------------------------------------

EXTRACTION_JUDGE_MODEL = "claude-sonnet-4-5-20250929"
EXTRACTION_JUDGE_MAX_TOKENS = 1024  # llm-judging.md rule-23 single-rationale floor

EXTRACTION_JUDGE_SYSTEM = (
    "You are an answer-extraction assistant. You are shown a multiple-choice question "
    "(options A-D) and a model's response. Your ONLY job is to name which option letter the "
    "response chose — you never re-answer the question yourself. Reply with a JSON object "
    '{"reason": "<brief>", "letter": "<A|B|C|D|UNPARSEABLE>"} — reason FIRST, then the '
    'letter. Use "UNPARSEABLE" when the response commits to no single option.'
)

EXTRACTION_JUDGE_USER_TEMPLATE = (
    "Question (with options):\n{question}\n\nModel response:\n{answer}\n\n"
    'Which option letter did the response choose? Reply with the JSON object {"reason": ..., '
    '"letter": ...} only.'
)

_VALID_JUDGE_LETTERS = frozenset({"A", "B", "C", "D", "UNPARSEABLE"})


def format_extraction_judge_user(question: str, answer: str) -> str:
    """Substitute the {question}/{answer} placeholders (graded_judge .replace
    convention); asserts no slot is left unfilled (rule-27 presence check)."""
    msg = EXTRACTION_JUDGE_USER_TEMPLATE.replace("{question}", question).replace("{answer}", answer)
    assert "{question}" not in msg and "{answer}" not in msg, "unfilled template slot"
    return msg


def parse_extraction_judgment(text: str) -> str | None:
    """Parse one extraction-judge reply through the harness's OWN parse path.

    Routes through eval.utils.parse_judge_json (the forced-JSON contract,
    llm-judging.md rule 27); returns the letter (A-D), "UNPARSEABLE", or None
    on a malformed return (rule-9 drop, never coerced).
    """
    from explore_persona_space.eval.utils import parse_judge_json

    parsed = parse_judge_json(text)
    if not isinstance(parsed, dict):
        return None
    letter = parsed.get("letter")
    if not isinstance(letter, str):
        return None
    letter = letter.strip().upper()
    return letter if letter in _VALID_JUDGE_LETTERS else None


# ---------------------------------------------------------------------------
# Small shared utilities
# ---------------------------------------------------------------------------


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def write_json_atomic(path: Path, obj) -> None:
    from explore_persona_space.atomic_io import write_json_atomic as _w

    _w(path, obj)


def read_jsonl(path: Path) -> list[dict]:
    """Text-mode iteration, never .splitlines() (U+2028 shred; gotchas.md #950)."""
    out: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                out.append(json.loads(line))
    return out


def write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    from explore_persona_space.atomic_io import write_jsonl_atomic as _w

    _w(path, rows)
