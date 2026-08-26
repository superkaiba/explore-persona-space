"""c72 contingent judge wave (>=5k calls) names its pilot gate — verify_plan tests (#2590).

Fixtures are structurally faithful to their originating lines (the #2165
fixture-fidelity lesson): the founding WARN fixture reproduces #2588
v2:155's shape — the contingent Sonnet-judge fallback sentence ("extraction
failure > 5% ... fall back to a Sonnet judge ... would add ~≤19k Batch-API
calls") WITH the same-window compute-timing pilot line ("the smoke
additionally TIMES one batched permutation draw block ... the pilot basis
the §9 permutation-battery row is gated on") that a bare substring-`pilot`
satisfier would false-accept — so the JUDGE-vocabulary anchor requirement
is pinned by the founding window itself, not a synthetic one.
"""

# The fixture strings quote the real corpus glyphs (→, ≤, §) the check's
# grammars accept — ambiguous-unicode lint is noise here (the monolith
# tests/test_verify_plan.py carries the same directive).

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

C72 = "c72_contingent_judge_pilot"

# #2588 v2:155-faithful anchor + the same-window compute-timing pilot line.
ANCHOR_BLOCK = """\
# Plan — task #9995: hard-surface transfer (c72 founding fixture, #2588 v2:155 shape)

## 4.5 Scoring

- **Why code, not a model call?** MCQ letter extraction + exact-match correctness \
use the ported deterministic code: 18,810 GPQA rows, structural output, zero API \
cost, reproducible. Flip condition: extraction failure > 5% on the smoke slice → \
fall back to a Sonnet judge for the unparsed residue (would add ~≤19k Batch-API \
calls). No other unstructured-data heuristic exists in the design.
- **Smoke/sweep architectural parity:** the smoke additionally TIMES one batched \
permutation draw block (20 draws) at PRODUCTION shape — the pilot basis the §9 \
permutation-battery row is gated on, measured BEFORE any fan-out.
"""

SATISFIER_LINE = (
    "- The residue wave is pilot-gated per rule 26: 150 draws at the production "
    "instrument, zero max_tokens stop_reason, per-arm parse-fail < 2%.\n"
)


def _run(plan: str, kind: str = "experiment"):
    return verify_plan.check_contingent_judge_pilot(plan, kind)


def test_t1_founding_shape_warns_and_names_compute_pilot_line():
    r = _run(ANCHOR_BLOCK)
    assert r.id == C72
    assert r.status == "WARN"
    assert "19000" in r.detail
    # The non-qualifying compute-timing pilot line is NAMED so the reader
    # sees WHY it did not satisfy.
    assert "non-qualifying 'pilot' lines" in r.detail
    assert "compute-timing pilots do not satisfy" in r.detail


def test_t2_rule26_judge_pilot_line_satisfies_with_recorded_span():
    r = _run(ANCHOR_BLOCK + SATISFIER_LINE)
    assert r.status == "PASS", r.detail
    assert "judge-pilot satisfier at line" in r.detail
    assert "pilot-gated per rule 26" in r.detail


def test_t3_sub_floor_estimate_skips():
    r = _run(ANCHOR_BLOCK.replace("~≤19k Batch-API calls", "~3k Batch-API calls"))
    assert r.status == "SKIP"
    assert "below the 5000" in r.detail


def test_t4_negated_fallback_skips():
    plan = (
        "## 4.5 Scoring\n\n"
        "No fallback to a judge exists in this design; all scoring is "
        "deterministic code over 18,810 rows.\n"
    )
    r = _run(plan)
    assert r.status == "SKIP"
    assert "no contingent judge-wave vocabulary" in r.detail


def test_t5_kind_infra_skips():
    r = _run(ANCHOR_BLOCK, kind="infra")
    assert r.status == "SKIP"
    assert "kind=infra" in r.detail


def test_t6_escapes_pass_and_wrapped_variant_does_not():
    r = _run(ANCHOR_BLOCK + "\nN/A — no contingent judge wave\n")
    assert r.status == "PASS"
    assert "explicit N/A declared" in r.detail
    r2 = _run(
        ANCHOR_BLOCK + "\nN/A — the contingent judge wave inherits the primary wave's pilot gate\n"
    )
    assert r2.status == "PASS"
    assert "explicit N/A declared" in r2.detail
    # #1238 anti-paste: a backtick-wrapped declaration is NOT recognized.
    r3 = _run(ANCHOR_BLOCK + "\n`N/A — no contingent judge wave`\n")
    assert r3.status == "WARN"


def test_t7_comma_form_estimate_warns():
    r = _run(ANCHOR_BLOCK.replace("~≤19k Batch-API calls", "19,000 Batch-API calls"))
    assert r.status == "WARN"
    assert "19000" in r.detail


def test_t8_superseded_estimate_line_dropped_from_window():
    plan = (
        "## 4.5 Scoring\n\n"
        "- Flip condition: extraction failure > 5% → fall back to a Sonnet judge "
        "for the unparsed residue.\n"
        "- v2's wave was sized at ~19k Batch-API calls before the descope.\n"
    )
    r = _run(plan)
    assert r.status == "SKIP"
    assert "no call estimate resolvable" in r.detail


def test_t9_two_anchors_one_estimateless_one_warning():
    plan = (
        "## 4.5 Scoring\n\n"
        "- A contingent judge fallback is registered for the residue.\n" + "\n" * 20 + ANCHOR_BLOCK
    )
    r = _run(plan)
    assert r.status == "WARN"
    assert "anchor line" in r.detail
    assert "19000" in r.detail


def test_t10_judge_pilot_token_alone_satisfies():
    plan = ANCHOR_BLOCK + "- judge_pilot_gate(n=150) runs before any wave dispatch.\n"
    r = _run(plan)
    assert r.status == "PASS"
    assert "judge_pilot" in r.detail


def test_t11_sentence_boundary_blocks_cross_sentence_pairing():
    plan = ANCHOR_BLOCK + "- the pilot runs first; the judge wave follows it.\n"
    r = _run(plan)
    # 'pilot' and the judge token sit in DIFFERENT sentences — the [^.;]
    # gap classes stop at the boundary, so the line does not satisfy.
    assert r.status == "WARN"


def test_t12_per_family_compute_pilots_do_not_satisfy():
    plan = ANCHOR_BLOCK + "- per-family pilots re-project (G3) after the first cell.\n"
    r = _run(plan)
    assert r.status == "WARN"
    assert "non-qualifying 'pilot' lines" in r.detail


def test_t15_filename_period_does_not_truncate_satisfier_gap():
    # Calibration-driven (#2054 v4-v8): a genuine in-window rule-26
    # registration whose pilot→ctx span crosses a FILENAME period
    # ("llm-judging.md") must still satisfy — only a sentence-final period
    # (followed by whitespace/end) bounds the gap.
    plan = ANCHOR_BLOCK + (
        "- **Mitigation:** pilot-gate 200 calls first (llm-judging.md rule 26), "
        "measure per-arm parse-fail rate, only then batch-dispatch.\n"
    )
    r = _run(plan)
    assert r.status == "PASS", r.detail
    assert "pilot-gate 200 calls first" in r.detail


def test_t13_digit_bomb_no_exception():
    plan = (
        "## 4.5 Scoring\n\n"
        "- Flip condition → fall back to a Sonnet judge wave of " + "9" * 5000 + " calls.\n"
    )
    r = _run(plan)  # must not raise (A14: no per-check containment)
    assert r.status == "SKIP"


def test_t14_real_2588_v2_file_warns_and_v3_passes():
    hits_v2 = glob(str(REPO_ROOT / "tasks" / "*" / "2588" / "plans" / "v2.md"))
    hits_v3 = glob(str(REPO_ROOT / "tasks" / "*" / "2588" / "plans" / "v3.md"))
    if not hits_v2 or not hits_v3:
        pytest.skip("tasks/*/2588/plans/{v2,v3}.md absent (task folders move)")
    r2 = _run(Path(hits_v2[0]).read_text())
    assert r2.status == "WARN"
    assert "anchor line 155" in r2.detail
    r3 = _run(Path(hits_v3[0]).read_text())
    assert r3.status == "PASS", r3.detail


def test_registered_in_checks():
    assert verify_plan.check_contingent_judge_pilot in verify_plan.CHECKS
