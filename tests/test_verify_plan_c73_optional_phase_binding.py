"""c73 gates-section binding to an optionality-marked phase — verify_plan tests (#2363).

Fixtures are structurally faithful to their originating lines (the #2165
fixture-fidelity lesson): the founding WARN fixture reproduces #2360 v2's
shape — a bold-headed ``**Phase V (optional but planned) — ...**``
declaration, a fenced command block, and a ``## 7. Decision Gates`` section
referencing the phase in the HYPHENATED form (``Phase-V``) with a
kill-criterion (a) sentence — with different content words. The
hyphen/space variance is the load-bearing extraction fact (#2360 v2's §7
uses ``Phase-V`` exclusively while the declarations use ``Phase V``).
"""

# The fixture strings quote the real corpus glyphs (—, §, “) the check's
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

C73 = "c73_optional_phase_binding"

# #2360 v2-faithful shape: bold-headed optionality-marked declaration +
# fenced command block + a §7 referencing the phase HYPHENATED.
ANCHOR_BLOCK = """\
# Plan — task #9996: transfer-map box probe (c73 founding fixture, #2360 v2 shape)

## 6. Phases

- **Phase V (optional but planned) — end-to-end box validation (checks 2+3 live), \
gated on a BINDING first step:** provision one small box, run probe A twice, record \
wall seconds.

```bash
uv run python scripts/probe.py --phase v --box small
```

## 7. Decision Gates

- Check 2 (box half): the Phase-V probe A wall-time budget holds on the box filesystem.
- Check 3: the Phase-V timing log carries >= 40 rows.
- Kill (a): the Phase-V healthy small-box probe exceeds 120 s wall.
"""


def _run(plan: str, kind: str = "experiment"):
    return verify_plan.check_optional_phase_binding(plan, kind)


def test_t1_founding_shape_warns_with_phase_and_binding_quoted():
    r = _run(ANCHOR_BLOCK)
    assert r.id == C73
    assert r.status == "WARN"
    # Acceptance bullet 1/5 shape: phase name + declaration + binding quoted.
    assert "Phase V" in r.detail
    assert "optional but planned" in r.detail  # declaration excerpt
    assert "Phase-V" in r.detail  # binding-line excerpt (hyphenated §7 form)
    assert "line" in r.detail


def test_t2_real_2360_v2_warns_and_v3_clean():
    hits_v2 = glob(str(REPO_ROOT / "tasks" / "*" / "2360" / "plans" / "v2.md"))
    hits_v3 = glob(str(REPO_ROOT / "tasks" / "*" / "2360" / "plans" / "v3.md"))
    if not hits_v2 or not hits_v3:
        pytest.skip("tasks/*/2360/plans/{v2,v3}.md absent (task folders move)")
    r2 = _run(Path(hits_v2[0]).read_text(), kind="infra")
    assert r2.status == "WARN"
    assert "Phase V" in r2.detail
    r3 = _run(Path(hits_v3[0]).read_text(), kind="infra")
    # Negation-guard TN twin: the corrected revision must NOT flag.
    assert r3.status in ("PASS", "SKIP"), r3.detail


def test_t3_optional_phase_unbound_passes():
    plan = (
        "## 4. Design\n\n"
        "- **Phase III (optional) — extra plots.**\n\n"
        "## 7. Decision Gates\n\n"
        "- Kill (a): training loss diverges before step 100.\n"
    )
    r = _run(plan)
    assert r.status == "PASS", r.detail
    assert "Phase III" in r.detail  # names the unreferenced phase


def test_t4_bound_phase_not_optional_skips():
    plan = (
        "## 4. Design\n\n"
        "- **Phase II — corpus build.**\n\n"
        "## 7. Decision Gates\n\n"
        "- Kill (a): the Phase II corpus is empty.\n"
    )
    r = _run(plan)
    assert r.status == "SKIP"
    assert "no optionality-marked phase declaration" in r.detail


def test_t5_negation_guard_required_line_skips():
    plan = (
        "## 4. Design\n\n"
        '- **Phase V REQUIRED (was "optional but planned" in v2 — struck).**\n\n'
        "## 7. Decision Gates\n\n"
        "- Kill (a): the Phase-V probe exceeds 120 s wall.\n"
    )
    r = _run(plan)
    assert r.status == "SKIP"


def test_t6_marker_in_gates_row_only_skips():
    # Documented scope limit: an optionality marker only INSIDE a
    # gates-section row does not anchor (declarations are keyed outside
    # gates sections); the lens item owns the optional-criterion defect.
    plan = (
        "## 4. Design\n\n"
        "- **Phase V — box validation.**\n\n"
        "## 7. Decision Gates\n\n"
        "- (optional) the Phase-V probe may be rerun once.\n"
    )
    r = _run(plan)
    assert r.status == "SKIP"
    assert "no optionality-marked phase declaration" in r.detail


def test_t7_declaration_in_section9_table_row_warns():
    # The #2360 v2 line-313 shape: the declaration lives in a §9 table row.
    plan = (
        "## 9. Resources\n\n"
        "| phase | wall |\n|---|---|\n"
        "| Phase V (optional validation, one small box) | 0.2 h |\n\n"
        "## 7. Decision Gates\n\n"
        "- Kill (a): the Phase-V probe exceeds 120 s wall.\n"
    )
    r = _run(plan)
    assert r.status == "WARN"
    assert "Phase V" in r.detail


def test_t8_multiple_phases_warns_only_the_optional_one():
    plan = (
        "## 4. Design\n\n"
        "- **Phase II — corpus build.**\n"
        "- **Phase V (optional) — box validation.**\n\n"
        "## 7. Decision Gates\n\n"
        "- Kill (a): the Phase-II corpus is empty.\n"
        "- Kill (b): the Phase-V probe exceeds 120 s wall.\n"
    )
    r = _run(plan)
    assert r.status == "WARN"
    assert "Phase V:" in r.detail
    assert "Phase II:" not in r.detail


def test_t9_na_escape_passes_and_wrapped_variant_does_not():
    r = _run(ANCHOR_BLOCK + "\nN/A — no acceptance binding to an optional phase\n")
    assert r.status == "PASS"
    assert "explicit N/A declared" in r.detail
    # #1238 anti-paste: a backtick-wrapped declaration is NOT recognized
    # (`_standalone_na_declared` contract).
    r2 = _run(ANCHOR_BLOCK + "\n`N/A — no acceptance binding to an optional phase`\n")
    assert r2.status == "WARN"


def test_t10_fenced_declaration_does_not_anchor():
    plan = (
        "## 4. Design\n\n"
        "```\n- **Phase V (optional but planned) — box validation.**\n```\n\n"
        "## 7. Decision Gates\n\n"
        "- Kill (a): the Phase-V probe exceeds 120 s wall.\n"
    )
    r = _run(plan)
    assert r.status == "SKIP"


def test_t11_hyphenated_gates_reference_matches_spaced_declaration():
    # The load-bearing #2360 v2 extraction fact: declarations say
    # `Phase V`, §7 says `Phase-V` — a space-only reference regex would
    # MISS the founding incident.
    plan = (
        "## 4. Design\n\n"
        "- Phase V (optional) runs one probe on the box.\n\n"
        "## 7. Decision Gates\n\n"
        "- Kill (a): Phase-V timing exceeds 120 s.\n"
    )
    r = _run(plan)
    assert r.status == "WARN"


def test_t12_marker_beyond_80_char_proximity_bound_skips():
    # The bound is a plain character offset from the phase-id match start:
    # it does NOT exclude cross-clause co-occurrence within 80 chars, so
    # this fixture places the marker GENUINELY beyond 80 chars (asserted
    # below, self-verifying).
    line = (
        "- Phase II completes first and writes the per-cell manifest rows "
        "that all later consumers read. Optional columns may be added later."
    )
    pm = verify_plan._C73_PHASE_ID_RE.search(line)
    om = verify_plan._C73_OPTIONAL_RE.search(line)
    assert om.start() - pm.start() > 80, "fixture must sit beyond the proximity bound"
    plan = (
        "## 4. Design\n\n" + line + "\n\n"
        "## 7. Decision Gates\n\n"
        "- Kill (a): the Phase-II manifest is empty.\n"
    )
    r = _run(plan)
    assert r.status == "SKIP"


def test_t13_no_gates_section_skips():
    plan = "## 4. Design\n\n- **Phase V (optional) — box validation.**\n"
    r = _run(plan)
    assert r.status == "SKIP"
    assert "no Decision-Gates" in r.detail


def test_t14_armed_for_all_kinds():
    # The founding incident is kind: infra — the check must not kind-SKIP.
    r = _run(ANCHOR_BLOCK, kind="infra")
    assert r.status == "WARN"


def test_registered_in_checks():
    assert verify_plan.check_optional_phase_binding in verify_plan.CHECKS
