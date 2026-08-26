"""Issue #2588 G1 correction — Qwen thinking arm is PREFILL, not emergent.

Measured 2026-08-26 under the pinned stack (transformers 5.15.1): ALL 7 Qwen
panel checkpoints (Qwen3.5-{0.8B,2B,4B,9B,27B}, Qwen3.6-27B, Qwen3.8-27B)
render ``enable_thinking=True`` prompts ending with the PRE-OPENED
``<|im_start|>assistant\\n<think>\\n`` — the plan §7 "emergent" premise
(model opens its own block) was a #2546-era Qwen3 port the Qwen3.5 template
family obsoleted, and the P0 render probe caught it at zero GPU cost.

These tests pin the corrected contract at the HELPER (``assert_template_
sidespec``) AND its call sites (``Cell.parse_mode`` feeding the segmenter;
``render_prompt_ids``'s per-render guard) — the two prior P0 defects were
both caller-side, so helper-only tests cannot catch a relapse. The ORIGINAL
defect is pinned in reverse: the plan's assumed emergent-style render (no
think tags in the prompt) must now FAIL the qwen arm-b SideSpec. No network,
no GPU, repo-root paths only (adoptable-tests contract).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue2588_panel_common as PC

_HEAD = "<|im_start|>user\n{q}<|im_end|>\n<|im_start|>assistant\n"
MEASURED_THINK_ON = _HEAD + "<think>\n"  # all 7 Qwen checkpoints, 2026-08-26
MEASURED_THINK_OFF = _HEAD + "<think>\n\n</think>\n\n"  # the #2502/#2378 pin
EMERGENT_STYLE = _HEAD  # the plan's ASSUMED arm-b shape (no think tags)


class FakeQwenTok:
    """Minimal tokenizer double: parameterized (think-on, think-off) renders.

    Char-level ids under ``tokenize=True`` so ``render_prompt_ids``'s exact
    production call shape (``return_dict=False`` included) is exercised.
    """

    def __init__(self, on: str = MEASURED_THINK_ON, off: str = MEASURED_THINK_OFF):
        self._on, self._off = on, off

    def apply_chat_template(
        self,
        msgs,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool | None = None,
        return_dict: bool | None = None,
    ):
        assert add_generation_prompt is True
        assert enable_thinking is not None, "qwen render must pass the toggle"
        text = (self._on if enable_thinking else self._off).format(q=msgs[0]["content"])
        if tokenize:
            assert return_dict is False, "5.x default flips to BatchEncoding (#2588 port note)"
            return [ord(c) for c in text]
        return text


class FakeOlmoThinkTok:
    """OLMo-Think double: no enable_thinking kwarg exists on this template."""

    def apply_chat_template(
        self, msgs, *, tokenize: bool, add_generation_prompt: bool, return_dict: bool | None = None
    ):
        text = MEASURED_THINK_ON.format(q=msgs[0]["content"])
        return [ord(c) for c in text] if tokenize else text


# ---------------------------------------------------------------------------
# Cell.parse_mode — the segmenter-feeding call site
# ---------------------------------------------------------------------------


def test_parse_mode_prefill_for_every_thinking_cell_and_emergent_never_produced():
    cells = PC.all_cells()  # internally asserts 19 cells / 21 registered maps
    modes = {c.key: c.parse_mode for c in cells}
    for c in cells:
        if c.arm == "a" or not c.model.thinking:
            assert modes[c.key] == "off", modes
        else:
            assert modes[c.key] == "prefill", modes
    assert "emergent" not in modes.values(), modes


# ---------------------------------------------------------------------------
# assert_template_sidespec — the corrected qwen arm-b contract
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("family", ["qwen35", "qwen36", "qwen38"])
def test_sidespec_qwen_arm_b_accepts_measured_prefill_render(family):
    sha = PC.assert_template_sidespec(FakeQwenTok(), family, "b")
    assert len(sha) == 16 and all(c in "0123456789abcdef" for c in sha)


def test_sidespec_qwen_arm_b_rejects_the_old_emergent_render():
    # The ORIGINAL plan premise, pinned in reverse: a render with NO think
    # tags (the "emergent" shape) must FAIL the corrected prefill contract.
    with pytest.raises(RuntimeError, match="prefill"):
        PC.assert_template_sidespec(FakeQwenTok(on=EMERGENT_STYLE), "qwen35", "b")


def test_sidespec_qwen_arm_b_rejects_closed_block_in_prompt():
    with pytest.raises(RuntimeError, match="CLOSE"):
        PC.assert_template_sidespec(FakeQwenTok(on=MEASURED_THINK_OFF), "qwen35", "b")


def test_sidespec_qwen_arm_a_contract_unchanged():
    assert PC.assert_template_sidespec(FakeQwenTok(), "qwen35", "a")
    with pytest.raises(RuntimeError, match="empty"):
        PC.assert_template_sidespec(FakeQwenTok(off=EMERGENT_STYLE), "qwen35", "a")


def test_sidespec_olmo_think_contract_unchanged():
    assert PC.assert_template_sidespec(FakeOlmoThinkTok(), "olmo_think", "b")


# ---------------------------------------------------------------------------
# render_prompt_ids — the per-render caller-side guard
# ---------------------------------------------------------------------------


def test_render_prompt_ids_guards_qwen_arm_b_prefill_per_render():
    ids = PC.render_prompt_ids(FakeQwenTok(), "ping", "qwen35", "b")
    assert ids == [ord(c) for c in MEASURED_THINK_ON.format(q="ping")]
    with pytest.raises(RuntimeError, match="prefill suffix absent"):
        PC.render_prompt_ids(FakeQwenTok(on=EMERGENT_STYLE), "ping", "qwen35", "b")


def test_render_prompt_ids_guards_qwen_arm_a_empty_block_per_render():
    assert PC.render_prompt_ids(FakeQwenTok(), "ping", "qwen35", "a")
    with pytest.raises(RuntimeError, match="empty think block absent"):
        PC.render_prompt_ids(FakeQwenTok(off=EMERGENT_STYLE), "ping", "qwen35", "a")
