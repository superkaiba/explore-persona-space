"""c69 armed re-gen 2×-cap headroom vs max_model_len pin — verify_plan gate tests (#2269).

Fixtures are structurally faithful to their originating lines (the #2165
fixture-fidelity lesson): the founding WARN fixture DELIBERATELY carries
#2221 v9's own CORRECT first-pass arithmetic (2048 + ≤1,900 = 3,948 ≤
4,096 at v9:101) so a naive stated-triple implementation fails test 1
first; the multi-stage negative fixture carries a satisfied armed stage
(window-local bound 500 against pin 5000) plus a LARGER unrelated remote
bound (1,900) placed OUTSIDE the ±3-raw-line window so a plan-wide bound
join fails test 14 (r1 Must-Fix 1); and the shorthand fixture reproduces
#2225 v12:97's line shape INCLUDING the "28 × 200 = 5,600" co-text that
pins the [1, 8] multiplier clamp and the on-line max-cap rule (r1
Must-Fix 2).
"""

# ruff: noqa: RUF001, RUF002, RUF003
# The fixture strings quote the real corpus glyphs (×, ≥, ≤, ⇒) the
# check's character classes accept — ambiguous-unicode lint is noise here
# (the monolith tests/test_verify_plan.py carries the same directive).

from __future__ import annotations

import importlib.util
import sys
from glob import glob
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_verify_plan():
    spec = importlib.util.spec_from_file_location(
        "verify_plan", REPO_ROOT / "scripts" / "verify_plan.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault("verify_plan", mod)
    spec.loader.exec_module(mod)
    return sys.modules["verify_plan"]


verify_plan = _load_verify_plan()

C69 = "c69_regen_headroom"

V9_SHAPE = """\
# Plan — task #9999: re-mine corpora (c69 founding-incident fixture)

## 4. Design

**max_new_tokens deviation fix:**
- This round adds `EVAL_MAX_NEW_TOKENS = 2048` and wires the trait-eval \
generation to it, with the >2% cap-hit re-gen trigger ARMED (actually \
re-generates capped rows at ≥2× the cap via the existing `_regen_cell`/\
`phase_rollouts_regen` mechanism, then re-reports residual cap-hit).
- **max_model_len headroom:** `issue778_lib.build_vllm_engine` pins \
`max_model_len=4096`. At cap 2048: paper 20-q panel (short, ≤~500 prompt \
tokens) → 2048+500 < 4096, comfortable. The 50-prompt LMSYS panel is \
length-validated at load to ≤ 1,900 prompt tokens (drop overlong) so \
prompt + 2048 ≤ 4096 under the existing pin — no shared-module edit needed.
"""

MULTI_STAGE = """\
# Plan — task #9998: two-stage generation (c69 multi-stage negative fixture)

## 4. Design

**Stage A (short-form QA — the only re-gen leg):**
- `EVAL_MAX_NEW_TOKENS = 2048`, with the >2% cap-hit re-gen trigger ARMED \
(re-generates capped rows at ≥2× the cap).
- Stage A prompts are length-validated at load to ≤ 500 prompt tokens; the \
engine pins `max_model_len=5000`, so 2×2048 + 500 = 4596 < 5000.

## 9. Resources

**Stage B (long-context summarization, separate engine, no regen leg):**
- Stage B inputs are length-validated at load to ≤ 1,900 prompt tokens \
before the summarization pass.
"""

V12_SHAPE = """\
# Plan — task #9997: steering follow-up (c69 shorthand fixture, #2225 v12:97 shape)

### 4.4 Evaluation pipeline

1. **Trait-expression generation:** per cell, the steered trait's held-out \
20-question eval set × 10 rollouts @ temp 1.0, `max_new_tokens=2048`, vLLM + \
LoRA, 8-GPU target sharding. 28 × 200 = 5,600 responses. Cap-hit fraction \
per cell reported; > 2% ⇒ re-gen at 2× (contract inherited).

## 11. Decision Rationale

- **Coherence threshold 80; `max_new_tokens=2048` / `max_model_len` 4096.** \
Source: parent, verbatim.
"""


def _run(plan: str, kind: str = "experiment"):
    return verify_plan.check_regen_headroom(plan, kind)


def test_founding_v9_shape_warns_on_doubled_cap():
    # The fixture's own stated first-pass arithmetic is CORRECT
    # (2048 + 1,900 ≤ 4,096) — a naive stated-triple reader PASSes it.
    # The check must key on the DOUBLED cap: 2×2048 + 1,900 = 5,996 ≥ 4,096.
    r = _run(V9_SHAPE)
    assert r.id == C69
    assert r.status == "WARN"
    for token in ("5996", "4096", "regen_overlong_skipped"):
        assert token in r.detail, (token, r.detail)


def test_real_2221_v9_file_warns():
    hits = glob(str(REPO_ROOT / "tasks" / "*" / "2221" / "plans" / "v9.md"))
    if not hits:
        pytest.skip("tasks/*/2221/plans/v9.md absent (task folders move across statuses)")
    r = _run(Path(hits[0]).read_text())
    assert r.status == "WARN"
    assert "5996" in r.detail


def test_clean_case_8192_engine_passes():
    plan = V9_SHAPE.replace(
        "`issue778_lib.build_vllm_engine` pins `max_model_len=4096`.",
        "the regen leg runs on a dedicated `max_model_len=8192` engine "
        "(supersedes the parent's `max_model_len=4096`).",
    )
    assert plan != V9_SHAPE
    r = _run(plan)
    # Also pins the MAX-pin rule: both 4096 and 8192 are present; the
    # larger (corrective) pin wins, so the corrected plan does not re-fire.
    assert r.status == "PASS"
    assert "5996" in r.detail
    assert "8192" in r.detail


def test_equality_zero_headroom_warns():
    plan = """\
## 4. Design

- `EVAL_MAX_NEW_TOKENS = 2048`, with the >2% cap-hit re-gen trigger ARMED \
(re-generates capped rows at ≥2× the cap).
- The engine pins `max_model_len=8192`; prompts are length-validated at \
load to ≤ 4,096 prompt tokens.
"""
    r = _run(plan)
    # 2×2048 + 4,096 = 8,192 ≥ pin 8,192 — the Goal's "non-positive"
    # spec: exact equality (zero prompt headroom) fires.
    assert r.status == "WARN"
    assert "8192" in r.detail


def test_na_escape_passes():
    r = _run(V9_SHAPE + "\nN/A — no armed re-gen trigger\n")
    assert r.status == "PASS"
    assert "explicit N/A declared" in r.detail


def test_cross_quantity_escape_passes():
    r = _run(
        V9_SHAPE + "\nN/A — harvested max_model_len pin is unrelated to the armed re-gen stage\n"
    )
    assert r.status == "PASS"
    assert "explicit N/A declared" in r.detail


def test_wrapped_escape_not_recognized():
    # #1238 anti-paste: a backtick-wrapped declaration is NOT recognized.
    r = _run(V9_SHAPE + "\n`N/A — no armed re-gen trigger`\n")
    assert r.status == "WARN"


def test_negated_arming_skips():
    plan = """\
## 4. Design

The re-gen trigger is NOT armed this round; no re-generation occurs.
`EVAL_MAX_NEW_TOKENS = 2048`; the engine pins `max_model_len=4096`.
"""
    r = _run(plan)
    assert r.status == "SKIP"
    assert "no armed re-gen trigger" in r.detail


def test_kind_infra_skips():
    r = _run(V9_SHAPE, kind="infra")
    assert r.status == "SKIP"
    assert "kind=infra" in r.detail


def test_no_pin_skips():
    plan = V9_SHAPE.replace("max_model_len", "engine_ctx")
    assert "max_model_len" not in plan
    r = _run(plan)
    assert r.status == "SKIP"
    assert "no max_model_len" in r.detail


def test_unstated_bound_warns():
    plan = """\
## 4. Design

- `EVAL_MAX_NEW_TOKENS = 2048`, with the >2% cap-hit re-gen trigger ARMED \
(re-generates capped rows at ≥2× the cap).
- The engine pins `max_model_len=8192`.
"""
    r = _run(plan)
    assert r.status == "WARN"
    assert "NO stated prompt-token bound" in r.detail


def test_registered_in_checks():
    assert verify_plan.check_regen_headroom in verify_plan.CHECKS


def test_docstring_conditional_enumeration_carries_69():
    # The c53–c56 house pattern, LAST-entry form (:11926 precedent): the
    # mid-list `"69,"` form cannot match while 69 is the terminal entry.
    # Pins D2 item 3, which no generative test forces.
    assert "68, 69" in verify_plan.__doc__


def test_multi_stage_window_bound_beats_remote_bound():
    # r1 Must-Fix 1 negative fixture: the armed stage's own arithmetic is
    # satisfied (2×2048 + window-local 500 = 4596 < 5000); Stage B's
    # larger 1,900 bound sits 6 raw lines away, OUTSIDE the ±3 window. A
    # plan-wide bound join reads 5996 ≥ 5000 and false-fires (measured
    # RED under the v2 join, GREEN under window attribution).
    r = _run(MULTI_STAGE)
    assert r.status == "PASS", r.detail
    assert "4596" in r.detail
    assert "5000" in r.detail


def test_2225_shorthand_shape_warns():
    # r1 Must-Fix 2: the house shorthand "re-gen at 2×" ARMS (arm 2 — no
    # stated prompt bound; need 2×2048 = 4,096 ≥ pin 4,096). The "28 ×
    # 200 = 5,600" co-text pins the [1, 8] mult clamp (28 rejected) and
    # the on-line max-cap rule (200 co-harvested, 2048 wins).
    r = _run(V12_SHAPE)
    assert r.status == "WARN"
    for token in ("NO stated prompt-token bound", "2×2048", "meets/exceeds"):
        assert token in r.detail, (token, r.detail)


def test_real_2225_v12_file_warns():
    hits = glob(str(REPO_ROOT / "tasks" / "*" / "2225" / "plans" / "v12.md"))
    if not hits:
        pytest.skip("tasks/*/2225/plans/v12.md absent (task folders move across statuses)")
    r = _run(Path(hits[0]).read_text())
    assert r.status == "WARN"
    assert "4096" in r.detail
    assert "NO stated prompt-token bound" in r.detail


def test_fenced_arming_lines_skip():
    # Pins the _fence_mask leg of the self-inclusion absorption: a fenced
    # arming line contributes zero arm lines.
    lines = V9_SHAPE.splitlines()
    i = next(idx for idx, line in enumerate(lines) if "trigger ARMED" in line)
    fenced = [*lines[:i], "```", lines[i], "```", *lines[i + 1 :]]
    r = _run("\n".join(fenced))
    assert r.status == "SKIP"
    assert "no armed re-gen trigger" in r.detail


def test_spurious_grid_multiplier_not_harvested():
    # Pins the anchored-multiplier rule (the measured iteration-1 FP
    # mechanism: the house "families × {arms}" grid idiom): "4×H100" on
    # the arming line must NOT become mult 4.
    plan = V9_SHAPE.replace("trigger ARMED (actually", "trigger ARMED on 4×H100 (actually")
    assert plan != V9_SHAPE
    r = _run(plan)
    assert r.status == "WARN"
    assert "2×2048" in r.detail
    assert "4×2048" not in r.detail
