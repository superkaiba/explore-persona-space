"""c70 judge-pilot per-arm draw resolution vs parse-fail threshold — verify_plan tests (#2299).

Fixtures are structurally faithful to their originating lines (the #2165
fixture-fidelity lesson): the founding WARN fixture embeds #2162 v7 §7.3's
own shape — a 150-draw budget with additive rubric components (60 + 90)
whose MIN slice binds, the arm count 4 raw lines above on the PREVIOUS gate
item — so a naive total-only reader resolves 50/arm instead of 20/arm and
fails test 7 first; the #2054 fixture carries the real v9:196 "9,000 arms
the contingency wave" line (comma lookbehind + verb lookahead, the v1
ZeroDivisionError source); the #2254 fixture carries the real "N=3 draws
localize" per-item count (the `=` lookbehind, found by the plan's own
replay); and the corrected-#2162 fixture carries the house-mandated
superseded quote "(v7's 150-draw config gave ~30/arm ...)" on its own line
so the superseded-line guard is what silences it (round-1 MF2).
"""

# ruff: noqa: RUF001, RUF003
# The fixture strings quote the real corpus glyphs (×, ≥, ≤, ⇔, ≈) the
# check's regexes must survive — ambiguous-unicode lint is noise here
# (the c69 sibling file carries the same directive).

from __future__ import annotations

import importlib.util
import inspect
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

C70 = "c70_pilot_resolution"

V7_SHAPE = """\
# Plan — task #9996: behavior sweep (c70 founding fixture, #2162 v7 §7.3 shape)

## 7. Decision Gates

2. **Judge spot-check (VM).** Multi-position variant — 12 spot pairs spanning \
all 3 arms × depths d1/d3/d5, hook-armed; disagreement above 20 points re-opens \
the rubric.

3. **Judge pilot gate (VM, before any production judging).** Mechanism: §4.5 \
P4 — 150 draws from the production instrument (60 coherence + 90 value-rubric \
draws) at the exact production max_tokens; PASS ⇔ zero \
`stop_reason == "max_tokens"` AND parse-fail < 2% per arm. FAIL → fix \
rubric/budget, re-pilot; two consecutive FAILs → halt gate.
"""

SATISFIABLE = """\
# Plan — task #9995: satisfiable pilot fixture

## 6. Evaluation

- **Judge pilot gate:** 180 draws spanning all 3 arms at the production \
instrument; PASS ⇔ parse-fail < 2% per arm.
"""

UNPARSEABLE = """\
# Plan — task #9994: unparseable pilot fixture

## 6. Evaluation

- The judge pilot runs before production judging; parse-fail thresholds and \
draw budgets are registered in the gate table.
"""

DIRECT_WARN = """\
## 6. Evaluation

- **Judge pilot gate:** ~30 draws/arm at the production instrument; PASS ⇔ \
parse-fail < 2% per arm.
"""

DIRECT_PASS = """\
## 6. Evaluation

- **Judge pilot gate:** 60 draws/arm at the production instrument; PASS ⇔ \
parse-fail < 2% per arm.
"""

AGGREGATE = """\
## 6. Evaluation

- **Judge pilot gate:** 150 draws total; PASS ⇔ parse-fail < 2% across the \
pooled draws.
"""

AMBIGUOUS_DRAWS = """\
## 6. Evaluation

- **Judge pilot gate:** 150 draws in phase A and 200 draws in phase B; PASS ⇔ \
parse-fail < 2% per arm across all 3 arms.
"""

AMBIGUOUS_ARMS = """\
## 6. Evaluation

- **Judge pilot gate:** 150 draws; PASS ⇔ parse-fail < 2% per arm; the sweep \
spans 3 arms in phase A and 5 arms in phase B.
"""

SIBLING_DECIMALS = (
    V7_SHAPE
    + """\

4. **Fair-comparison margin gate.** PASS ⇔ |mean ΔF| ≤ 0.10 per arm on the \
matched panel (72 pairs/arm); independent of the pilot above.
"""
)

SHAPE_2254 = """\
# Plan — task #9993: reasoning-drop probe (c70 #2254 v5:152 shape)

## 6. Evaluation

- **Judge instrument:** N=3 draws localize / 5 decisive per item at \
temperature 1.0, mean-aggregated across the ≈ 7 arms; per-arm parse-fail <2% \
enforced by a ~400-draw pilot sized to ≥51 effective draws per pilot arm via \
the guard-routed helper.
"""

SHAPE_2054 = """\
# Plan — task #9992: OOD contingency sizing (c70 #2054 v9:196+204 shape)

## 9. Resources

- Approval efficiency: sized so a projection < 9,000 arms the contingency \
wave preemptively (no extra approval — same spend envelope).
- Filler row keeping realistic paragraph distance between the two items.
- Second filler row.
- Third filler row.
- **Judge pilot gate (rule 26; instrument gate, not a design gate).** Runs \
~200 draws spanning the 4 character arms at the production instrument; PASS ⇔ \
zero truncation AND per-arm parse-fail < 2%; FAIL → fix rubric/budget before \
the wave.
"""

CORRECTED_2162 = """\
# Plan — task #9991: corrected pilot sizing (c70 #2162 v10:682-690 shape)

## 7. Decision Gates

3. **Judge pilot gate (VM, before any production judging).** **540 sync \
draws** at the production instrument (60 per arm × 3 arms per rubric slice), \
sized to clear the resolution floor
   floor(1/0.02)+1 = 51 for the 2% per-arm threshold (v7's 150-draw config \
gave ~30/arm — one failure reads 3.3%, above the threshold by construction); \
60 effective draws per (arm × rubric) clears it.
   PASS ⇔ zero `stop_reason == "max_tokens"` AND parse-fail < 2% per arm per \
rubric. FAIL → fix rubric/budget, re-pilot.
"""

COMMA_FAMILY = """\
## 6. Evaluation

- **Judge pilot gate:** 1,000 draws/arm planned for phase A, a 10,150 draws \
ledger for phase B, and 9,000 arms of the contingency; PASS ⇔ parse-fail < 2% \
per arm.
"""

_FILLER = "\n".join(f"- filler row {i} keeping the two gate windows disjoint." for i in range(18))

DECLARED_PLUS_DEFECTIVE = f"""\
# Plan — task #9990: two-gate fixture (c70 declared + defective)

## 6. Evaluation

- **Coherence-rubric pilot gate:** 30 draws/arm; PASS ⇔ parse-fail < 2% per \
arm (allow_subresolution_pilot=True — deliberate sub-resolution smoke, \
recorded).

{_FILLER}

- **Value-rubric pilot gate:** 20 draws/arm; PASS ⇔ parse-fail < 2% per arm.
"""

# Characterization fixtures (round 2, concerns c70-cross-gate-declaration-leak
# + c70-observed-rate-false-warn): these reproduce the two channels the
# round-1 reconciler persisted as concerns. The tests over them pin the
# CURRENT plan-registered behavior (docstring FN-j / FP-c), NOT a desirable
# verdict — a future behavior change must surface as a test change.

DECLARED_IDENTICAL_TUPLE = f"""\
# Plan — task #9989: two disjoint identical-tuple gates (c70 FN-j shape i)

## 6. Evaluation

- **Coherence-rubric pilot gate:** 30 draws/arm; PASS ⇔ parse-fail < 2% per \
arm (allow_subresolution_pilot=True — deliberate sub-resolution smoke, \
recorded).

{_FILLER}

- **Value-rubric pilot gate:** 30 draws/arm; PASS ⇔ parse-fail < 2% per arm.
"""

DECLARED_ADJACENT_GATES = """\
# Plan — task #9988: adjacent declared + undeclared gates (c70 FN-j shape ii)

## 6. Evaluation

- **Coherence-rubric pilot gate:** 30 draws/arm; PASS ⇔ parse-fail < 2% per \
arm (allow_subresolution_pilot=True — deliberate sub-resolution smoke, \
recorded).
- **Value-rubric pilot gate:** 20 draws/arm; PASS ⇔ parse-fail < 2% per arm.
"""

OBSERVED_RATE = """\
## 6. Evaluation

- **Judge pilot result:** 30 draws/arm at the production instrument; observed \
parse-fail was 2% per arm.
"""

VERB_ARMS = """\
## 6. Evaluation

- **Judge pilot gate:** 150 draws at the production instrument; PASS ⇔ \
parse-fail < 2% per arm; a projection above threshold on 3 arms the driver \
with a full re-pilot.
"""

BIG = "9" * 5000

HUGE_TOTAL = f"""\
## 6. Evaluation

- **Judge pilot gate:** {BIG} draws across 3 arms; PASS ⇔ parse-fail < 2% \
per arm.
"""

HUGE_PCT = f"""\
## 6. Evaluation

- **Judge pilot gate:** 150 draws across 3 arms; PASS ⇔ parse-fail < {BIG}% \
per arm.
"""

HUGE_DIRECT = f"""\
## 6. Evaluation

- **Judge pilot gate:** {BIG} draws/arm; PASS ⇔ parse-fail < 2% per arm.
"""

HUGE_ARMS = f"""\
## 6. Evaluation

- **Judge pilot gate:** 150 draws across {BIG} arms; PASS ⇔ parse-fail < 2% \
per arm.
"""

FIXTURES = [
    V7_SHAPE,
    SATISFIABLE,
    UNPARSEABLE,
    DIRECT_WARN,
    DIRECT_PASS,
    AGGREGATE,
    AMBIGUOUS_DRAWS,
    AMBIGUOUS_ARMS,
    SIBLING_DECIMALS,
    SHAPE_2254,
    SHAPE_2254.replace("N=3 draws localize", "3 draws localize"),
    SHAPE_2054,
    CORRECTED_2162,
    COMMA_FAMILY,
    DECLARED_PLUS_DEFECTIVE,
    DECLARED_IDENTICAL_TUPLE,
    DECLARED_ADJACENT_GATES,
    OBSERVED_RATE,
    VERB_ARMS,
    HUGE_TOTAL,
    HUGE_PCT,
    HUGE_DIRECT,
    HUGE_ARMS,
]


def _run(plan: str, kind: str = "experiment"):
    return verify_plan.check_pilot_resolution(plan, kind)


def test_founding_2162_shape_warns():
    # T1: min rubric slice 60 of [60, 90] / 3 arms = 20/arm < 51 at 2%.
    r = _run(V7_SHAPE)
    assert r.id == C70
    assert r.status == "WARN"
    for token in ("required=51", "[60, 90]", "3 arms", "per_arm=20"):
        assert token in r.detail, (token, r.detail)


def test_satisfiable_counter_fixture_passes():
    # T2. NOTE: the plan §6 table quoted "180 value-rubric draws" here, but
    # an adjective between digit and "draws" blocks TOTAL by the plan's own
    # FN-a design, so the satisfiable fixture uses the bare-total form
    # "180 draws spanning all 3 arms" (180 / 3 = 60 >= 51).
    r = _run(SATISFIABLE)
    assert r.status == "PASS"
    assert "60>=51" in r.detail, r.detail


def test_unparseable_plan_skips():
    # T3: pilot vocabulary, no numbers.
    r = _run(UNPARSEABLE)
    assert r.status == "SKIP"


def test_kind_infra_skips():
    # T4: the kind gate — this check's own plan class.
    r = _run(V7_SHAPE, kind="infra")
    assert r.status == "SKIP"
    assert "kind=infra" in r.detail


def test_direct_per_arm_warns():
    # T5: the direct form short-circuits arm-count inference entirely.
    r = _run(DIRECT_WARN)
    assert r.status == "WARN"
    assert "direct 30/arm" in r.detail
    assert "required=51" in r.detail


def test_direct_per_arm_satisfiable_passes():
    # T6.
    r = _run(DIRECT_PASS)
    assert r.status == "PASS"
    assert "60>=51" in r.detail


def test_min_rubric_slice_binds():
    # T7: min slice 60 binds (60 // 3 = 20), NOT the total 150 (150 // 3 = 50).
    r = _run(V7_SHAPE)
    assert "min rubric slice 60" in r.detail
    assert "per_arm=20" in r.detail
    assert "per_arm=50" not in r.detail


def test_na_escapes_pass():
    # T8 (both §4.8 standalone escapes).
    r1 = _run(V7_SHAPE + "\nN/A — no judge-pilot gate\n")
    assert r1.status == "PASS"
    assert "explicit N/A declared" in r1.detail
    r2 = _run(
        V7_SHAPE + "\nN/A — harvested pilot sizing is historical or belongs to a different gate\n"
    )
    assert r2.status == "PASS"
    assert "explicit N/A declared" in r2.detail


def test_wrapped_escape_not_recognized():
    # T8 (#1238 anti-paste): a backtick-wrapped declaration is NOT recognized.
    r = _run(V7_SHAPE + "\n`N/A — no judge-pilot gate`\n")
    assert r.status == "WARN"


def test_subresolution_declared_passes():
    # T9: allow_subresolution_pilot on the gate line = per-tuple declared PASS.
    plan = V7_SHAPE.replace(
        "parse-fail < 2% per arm. FAIL",
        "parse-fail < 2% per arm (allow_subresolution_pilot=True registered). FAIL",
    )
    assert plan != V7_SHAPE
    r = _run(plan)
    assert r.status == "PASS"
    assert "declared allow_subresolution_pilot" in r.detail


def test_aggregate_threshold_skips():
    # T10 (S5): a threshold with no per-arm token is not adjudicable.
    r = _run(AGGREGATE)
    assert r.status == "SKIP"
    assert "per-arm token" in r.detail


def test_ambiguous_draws_skips():
    # T11 (S7).
    r = _run(AMBIGUOUS_DRAWS)
    assert r.status == "SKIP"
    assert "multiple distinct draw totals" in r.detail


def test_ambiguous_arms_skips():
    # T11 (S8).
    r = _run(AMBIGUOUS_ARMS)
    assert r.status == "SKIP"
    assert "multiple distinct arm counts" in r.detail


def test_real_2162_v7_file_warns():
    # T12: the motivating incident, replayed against the persisted plan.
    hits = glob(str(REPO_ROOT / "tasks" / "*" / "2162" / "plans" / "v7.md"))
    if not hits:
        pytest.skip("tasks/*/2162/plans/v7.md absent (task folders move across statuses)")
    r = _run(Path(hits[0]).read_text())
    assert r.status == "WARN"
    assert "per_arm=20" in r.detail
    assert "required=51" in r.detail


def test_never_fails_or_raises():
    # T13 / criterion 2: every fixture, under every kind, returns a result
    # with passed=True (WARN/SKIP/PASS all leave passed=True) — no exception.
    for fixture in FIXTURES:
        for kind in ("experiment", "analysis", "infra", "batch"):
            r = verify_plan.check_pilot_resolution(fixture, kind)
            assert r.passed is True, (kind, r.status, r.detail)
    # Structural half of criterion 2(a): no _fail( call anywhere in the c70
    # production span — the check function AND the extracted budget helper
    # (round 2: `_c70_resolve_budget` sat outside the guarded span, so a
    # future `_fail(` added to the helper would have left this pin green).
    src = inspect.getsource(verify_plan.check_pilot_resolution) + inspect.getsource(
        verify_plan._c70_resolve_budget
    )
    assert "_fail(" not in src


def test_registered_in_checks():
    # T13.
    assert verify_plan.check_pilot_resolution in verify_plan.CHECKS


def test_docstring_conditional_enumeration_carries_70():
    # The c53-c56 house pattern, LAST-entry form (c69 precedent): the
    # mid-list `"70,"` form cannot match while 70 is the terminal entry.
    assert "69, 70" in verify_plan.__doc__


def test_fenced_gate_spec_skips():
    # T14 (FN-d): the whole gate spec inside a fence contributes no anchors.
    body = V7_SHAPE.split("\n", 1)[1]
    r = _run(V7_SHAPE.split("\n", 1)[0] + "\n```\n" + body + "\n```\n")
    assert r.status == "SKIP"
    assert "no judge-pilot vocabulary" in r.detail


def test_sibling_gate_decimal_not_direct():
    # T15: "|mean ΔF| ≤ 0.10 per arm" (decimal lookbehind) and
    # "(72 pairs/arm)" (non-draws word) must NOT resolve as direct per-arm
    # forms — the real gate's slice math (20/arm) carries the verdict.
    r = _run(SIBLING_DECIMALS)
    assert r.status == "WARN"
    assert "per_arm=20" in r.detail
    assert "direct 10/arm" not in r.detail
    assert "direct 72/arm" not in r.detail


def test_exact_fraction_floor():
    # T16: required = floor(1/t) + 1 under exact Fraction arithmetic —
    # 2% -> 51, 3% -> 34, 1.5% -> 67 — pinned at the WARN/PASS boundary.
    for pct, required in (("2", 51), ("3", 34), ("1.5", 67)):
        below = _run(
            f"## 6. Evaluation\n\n- **Judge pilot gate:** {required - 1} draws/arm; "
            f"PASS ⇔ parse-fail < {pct}% per arm.\n"
        )
        assert below.status == "WARN", (pct, below.detail)
        assert f"required={required}" in below.detail
        at = _run(
            f"## 6. Evaluation\n\n- **Judge pilot gate:** {required} draws/arm; "
            f"PASS ⇔ parse-fail < {pct}% per arm.\n"
        )
        assert at.status == "PASS", (pct, at.detail)


def test_2254_shape_not_misread():
    # T17: "N=3 draws" is a per-item judge draw count — the `=` lookbehind
    # rejects it (v1 read total=3 / 7 arms -> 0/arm -> false WARN).
    r = _run(SHAPE_2254)
    assert r.status == "SKIP"
    # S11 backstops the unprefixed variant of the same family.
    r2 = _run(SHAPE_2254.replace("N=3 draws localize", "3 draws localize"))
    assert r2.status == "SKIP"
    assert "smaller than arm count" in r2.detail


def test_2054_comma_verb_no_raise_skips():
    # T18: "9,000 arms the contingency wave" — comma lookbehind + verb
    # lookahead; "the 4 character arms" is FN-g. No raise, SKIP at S6.
    r = _run(SHAPE_2054)
    assert r.status == "SKIP"
    assert "arm count" in r.detail


def test_real_2054_v9_file_no_raise():
    # T18 (real file): the v1 ZeroDivisionError corpus wedge must stay silent.
    hits = glob(str(REPO_ROOT / "tasks" / "*" / "2054" / "plans" / "v9.md"))
    if not hits:
        pytest.skip("tasks/*/2054/plans/v9.md absent (task folders move across statuses)")
    r = _run(Path(hits[0]).read_text())
    assert r.status == "SKIP"


def test_corrected_plan_superseded_quote_silent():
    # T19 (MF2): the house-mandated superseded quote ("v7's ... gave
    # ~30/arm") is guard-dropped, so the corrected gate never WARNs; the
    # remaining window has no TOTAL-form budget (FN-a) and SKIPs.
    r = _run(CORRECTED_2162)
    assert r.status in ("SKIP", "PASS"), r.detail
    assert r.status != "WARN"


def test_real_2162_v8_v10_files_silent():
    # T19 (real files): the FIXED revisions are the false-WARN class MF2
    # requires the guard to silence — a WARN here is a REGRESSION.
    hits = sorted(
        p
        for v in ("v8", "v9", "v10")
        for p in glob(str(REPO_ROOT / "tasks" / "*" / "2162" / "plans" / f"{v}.md"))
    )
    if not hits:
        pytest.skip("tasks/*/2162/plans/{v8,v9,v10}.md absent")
    for hit in hits:
        r = _run(Path(hit).read_text())
        assert r.status in ("SKIP", "PASS"), (hit, r.status, r.detail)


def test_comma_truncation_family():
    # T20 (MF1): "1,000 draws/arm", "10,150 draws", "9,000 arms" — no
    # truncated capture (000 / 150 / 000) may resolve; SKIP, never a WARN.
    r = _run(COMMA_FAMILY)
    assert r.status == "SKIP"


def test_declared_plus_defective_gate_still_warns():
    # T21: allow_subresolution_pilot does NOT early-return — a second,
    # distant defective gate (own anchor window) still WARNs.
    r = _run(DECLARED_PLUS_DEFECTIVE)
    assert r.status == "WARN"
    assert "per_arm=20" in r.detail
    assert "per_arm=30" not in r.detail


def test_known_fn_identical_tuple_declaration_leak_passes_today():
    # CHARACTERIZATION, NOT desirable behavior (round 2; concern
    # c70-cross-gate-declaration-leak; docstring FN-j shape i): the
    # (per_arm, required) dedup ORs `declared` across windows, so a second
    # DISJOINT UNDECLARED gate resolving the IDENTICAL tuple is silenced by
    # the first gate's declaration -> PASS today. This pins the CURRENT
    # plan-registered behavior (plan v3 §4.7/§4.9 per-tuple semantics) so a
    # future change is visible as a test change, not silent drift.
    r = _run(DECLARED_IDENTICAL_TUPLE)
    assert r.status == "PASS"
    assert "declared allow_subresolution_pilot" in r.detail


def test_known_fn_adjacent_gate_declaration_leak_passes_today():
    # CHARACTERIZATION, NOT desirable behavior (round 2; concern
    # c70-cross-gate-declaration-leak; docstring FN-j shape ii): with the
    # undeclared 20/arm gate INSIDE the declared gate's ±8-line window, the
    # first-match harvest (_C70_DIRECT_RE.search / _C70_THRESH_RE.search)
    # reads only the declared 30/arm gate -> PASS today; the 20/arm gate is
    # never evaluated. Contrast T21 above: DISTANT + DIFFERENT tuple WARNs.
    r = _run(DECLARED_ADJACENT_GATES)
    assert r.status == "PASS"
    assert "declared allow_subresolution_pilot" in r.detail
    assert "per_arm=20" not in r.detail


def test_known_fp_observed_rate_reads_as_threshold_warns_today():
    # CHARACTERIZATION, NOT desirable behavior (round 2; concern
    # c70-observed-rate-false-warn; docstring FP-c): _C70_THRESH_RE requires
    # no comparator / threshold vocabulary, so an OBSERVED parse-fail rate
    # in result prose reads as a configured gate threshold -> WARN at 30/arm
    # today (a known false positive; plan v3 §4.4 registers the regex, so
    # the comparator-vocabulary fix is plan-owned). Remedy in the wild: the
    # second standalone escape (historical / cross-gate pilot sizing).
    r = _run(OBSERVED_RATE)
    assert r.status == "WARN"
    assert "per_arm=30" in r.detail
    assert "required=51" in r.detail


def test_verb_arms_not_counted():
    # T22: "3 arms the driver" is the VERB usage — the lookahead rejects it,
    # so the minimal fixture lands S6 (no resolvable arm count).
    r = _run(VERB_ARMS)
    assert r.status == "SKIP"
    assert "arm count" in r.detail


def test_huge_literal_no_raise():
    # T23 (criterion 2(b)): 5000-digit literals in each numeric position —
    # the \\d{1,9} bound means they never capture; SKIP each, no raise
    # (CPython raises ValueError on int() above ~4300 digits).
    for fixture in (HUGE_TOTAL, HUGE_PCT, HUGE_DIRECT, HUGE_ARMS):
        r = _run(fixture)
        assert r.status == "SKIP", r.detail
