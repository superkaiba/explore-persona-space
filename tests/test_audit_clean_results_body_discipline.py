"""Regression tests for scripts/audit_clean_results_body_discipline.py.

Two v3-redesign (2026-W24) concerns are pinned here:

1. **Bulk-inventory gate accepts v3.** The `is_v2` "should I audit this
   body's prose" gate is consulted ONLY on the bulk-inventory path. It
   must recognise the v3 sentinel `<!-- clean-result-v3 -->` so bulk
   audits don't silently skip v3 bodies. The live pipeline paths
   (`--task <N>`, explicit file path) audit UNCONDITIONALLY and never
   consult the gate — pinned by `test_single_body_audit_unconditional`.

2. **`## Data` verbatim-content exemption.** v3 MANDATES verbatim
   training rows / probes / completions inside `## Data` example blocks
   (`<details>` or fenced). Those rows routinely contain strings the
   prose anti-pattern scan flags as project-internal condition codes
   (`C1` / `H2` / …) with no reword option — the same conflict the
   `**Context:**` blockquote carve-out fixed (#597). Example blocks
   inside `## Data` must be exempt from the scan.

Also pinned: the v4 `## Methodology` sample-data `<details>` exemption
(#1171) — v4 moves the verbatim sample rows to `## Methodology`, which
gets the identical exemption (Methodology PROSE stays scanned) — and the
bulk-inventory gate's acceptance of the v4 sentinel
`<!-- clean-result-v4 -->` (#1226).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT = REPO_ROOT / "scripts" / "audit_clean_results_body_discipline.py"

_spec = importlib.util.spec_from_file_location("audit_clean_results_under_test", SCRIPT)
assert _spec is not None and _spec.loader is not None
audit = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(audit)


# A compact v3 body whose `## Data` example blocks carry verbatim
# condition codes (`C1`, `H2`) — a `<details>` block under `### Trained
# on` and a second `<details>` under `### Generated`. The same codes
# appear only inside example blocks, so a correct audit produces no
# `condition_labels` findings. Body prose OUTSIDE the example blocks is
# deliberately clean.
V3_BODY_WITH_DATA_CODES = """\
---
title: v3 body with verbatim condition codes in Data
kind: experiment
goal: Exercise the Data verbatim-content exemption
---
# Some claim about a finding (MODERATE confidence)

<!-- clean-result-v3 -->

## Takeaways

- Headline finding: the implant installs cleanly across three seeds.
- Secondary finding: no measurable regression on the held-out probes.
- Caveat that binds interpretation: single model family, three seeds.

## What I ran

- **Why:** I tested whether the prior effect generalises to benchmark Z.
- **Design:** three seeds; baseline vs treatment; the single variable is the data mix.
- **Eval:** alignment score, Claude judge, 200 probes; matched to the prior surface.

## Findings

### A clean lift between baseline and treatment across three seeds

The lift holds at every seed in the held-out evaluation.

![Bar chart of mean alignment across three seeds.](https://example.com/hero.png)

> **Figure.** *The treatment lifts alignment over baseline at every seed.*

## Data

### Trained on

Established mix (tier 2), 2,000 rows, on-policy base completions.

<details open>
<summary>5 example training rows (5 of 2,000 rows, random sample)</summary>

| Row | Condition | Assistant |
|---|---|---|
| Positive | C1 | A normal answer. |
| Negative | C2 | A normal answer with H2 framing. |

Full training file: [link](https://example.com/train.jsonl).

</details>

Full data: [HF dataset](https://example.com/data)

### Evaluated with

200 probes (established benchmark), judged by Claude, no preprocessing.

Full probe bank: [link](https://example.com/probes)

### Generated

600 completions (3 seeds x 200 probes). Full raw completions: [raw](https://example.com/raw)

<details>
<summary>3 example completions (cherry-picked for illustration)</summary>

| Probe | Condition | Completion |
|---|---|---|
| p1 | C1 | A helpful, honest answer. |
| p2 | H2 | Another helpful answer. |

</details>

## Reproducibility

**Parameters:**

| Parameter | Value |
|---|---|
| Base model | Qwen-2.5-7B-Instruct |

**Artifacts:**
- Model: [hf-hub](https://example.com/model)

**Compute:** 1x H100, 47 min.

**Code:** entry script @ commit [0123456789abcdef](https://example.com/blob).

**Context:**
- Created 2026-06-12; run executed 2026-06-13.
- Originating prompt: origin prompt not recorded
"""


# ─── Change (a): the bulk-inventory should-audit gate accepts v3 ─────────


def test_is_v2_gate_accepts_v3_sentinel():
    """A v3-sentinel body must be selected for auditing in bulk mode
    (the gate name is historical; it means "current spec, audit it")."""
    body = "# Title (LOW confidence)\n\n<!-- clean-result-v3 -->\n\n## Takeaways\n\n- ok\n"
    assert audit.is_v2(body)


def test_is_v2_gate_accepts_v4_sentinel():
    """A v4-sentinel body must be selected for auditing in bulk mode —
    v4 is the CURRENT spec (2026-W26). The body below carries NO other
    recognised marker (no v3/v2 sentinel, no ## Reproducibility /
    ## TL;DR H2s), so pre-fix it failed every disjunct — the exact
    #1226 coverage hole."""
    body = (
        "# Title (LOW confidence)\n\n<!-- clean-result-v4 -->\n\n"
        "## Takeaways\n\n- ok\n\n## Goal\n\nx\n\n## Methodology\n\nx\n\n"
        "## Results\n\nx\n"
    )
    assert audit.is_v2(body)


def test_is_v2_gate_still_accepts_v2_and_legacy():
    """The v3 addition must not regress the v2 / flat / legacy gates."""
    assert audit.is_v2("x\n<!-- clean-result-v2 -->\n## TL;DR\n")
    assert audit.is_v2("## Human TL;DR\n## TL;DR\n## Reproducibility\n")
    assert audit.is_v2("## AI TL;DR\n## AI Summary\n")


def test_is_v2_gate_rejects_unstructured_body():
    """A body with none of the recognised markers is NOT audited in
    bulk mode (unmigrated / non-clean-result)."""
    assert not audit.is_v2("# Just a title\n\nSome freeform prose with no markers.\n")


# ─── Change (b): `## Data` example blocks are exempt from the scan ───────


def test_v3_data_details_block_codes_do_not_trip_audit():
    """Verbatim `C1` / `H2` codes inside `## Data` `<details>` example
    blocks must NOT produce a `condition_labels` finding — they are
    mandated verbatim content with no reword option."""
    findings = audit.audit_body(V3_BODY_WITH_DATA_CODES)
    assert "condition_labels" not in findings, findings
    # The whole body is clean once the exemption applies.
    assert findings == {}, findings


def test_condition_codes_outside_data_still_flagged():
    """The exemption is scoped to `## Data` ONLY — a `C1` in `## Findings`
    prose still trips the scan, so the carve-out can't be abused to hide
    opaque codes in reader-facing prose."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The C1 condition shows the lift at every seed.",
    )
    findings = audit.audit_body(leaky)
    assert "condition_labels" in findings, findings
    assert any("C1" in s for s in findings["condition_labels"]), findings


# ─── H1c-form sub-tag widening (#1914) ────────────────────────────────────


def test_sub_tag_condition_codes_in_prose_are_flagged():
    """`H<digit><lowercase>` hypothesis/plan sub-tags (`H1c`, `H4b`,
    `P4a`) in reader-facing prose trip `condition_labels`, and the
    matched token carries the sub-letter (#1914)."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "Under H1c the lift holds; H4b and P4a show the same pattern.",
    )
    findings = audit.audit_body(leaky)
    assert "condition_labels" in findings, findings
    matched = findings["condition_labels"]
    assert any("H1c" in s for s in matched), matched
    assert any("H4b" in s for s in matched), matched
    assert any("P4a" in s for s in matched), matched


def test_plural_heading_h2s_prose_not_flagged():
    """Plural markdown-heading prose ("the five flat H2s", "three H2s
    total") must NOT trip `condition_labels` — the sub-tag letter class
    deliberately excludes `s` (measured false-positive class, #1914)."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The five flat H2s follow the legacy H2s ordering; three H2s total.",
    )
    findings = audit.audit_body(body)
    assert "condition_labels" not in findings, findings


def test_gpu_name_prose_not_flagged():
    """GPU names (`H100`/`H200`) stay unmatched after the sub-tag
    widening — `[1-9]` + the trailing lookahead still exclude them
    (regression pin for the #1826 known-good class)."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "Training ran on a single H100 pod; the H200 fallback was unused.",
    )
    findings = audit.audit_body(body)
    assert "condition_labels" not in findings, findings


def test_prime_condition_label_still_flagged():
    """The primed form (`C1` + U+2032 PRIME) is still matched after the
    sub-tag widening (existing-behavior regression pin)."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The C1′ variant shows the same lift.",  # noqa: RUF001
    )
    findings = audit.audit_body(leaky)
    assert "condition_labels" in findings, findings
    assert any("C1′" in s for s in findings["condition_labels"]), findings  # noqa: RUF001


def test_condition_labels_regex_in_sync_with_verify_task_body():
    """The audit's `condition_labels` regex literal must equal the
    condition_labels portion of `verify_task_body._DATA_CONDITION_CODE_RE`
    (the prefix before its `|\\bBS_E` cell_tags branch) — makes the
    KEPT-IN-SYNC comment in verify_task_body.py self-enforcing (#1914)."""
    import sys

    if "verify_task_body" in sys.modules:
        vtb = sys.modules["verify_task_body"]
    else:
        vtb_script = REPO_ROOT / "scripts" / "verify_task_body.py"
        vtb_spec = importlib.util.spec_from_file_location("verify_task_body", vtb_script)
        assert vtb_spec is not None and vtb_spec.loader is not None
        vtb = importlib.util.module_from_spec(vtb_spec)
        sys.modules["verify_task_body"] = vtb
        vtb_spec.loader.exec_module(vtb)

    audit_pattern = audit.PATTERNS["condition_labels"][0]
    full = vtb._DATA_CONDITION_CODE_RE.pattern
    prefix = full.split(r"|\bBS_E", 1)[0]
    assert prefix == audit_pattern, (
        "condition_labels regex drifted between the audit and check 19b:\n"
        f"audit:    {audit_pattern!r}\n"
        f"check19b: {prefix!r}"
    )


def test_strip_data_example_blocks_only_drops_inside_data():
    """`strip_data_example_blocks` drops `<details>` blocks under
    `## Data` but leaves a `<details>` block under any other H2
    intact (so e.g. a Findings dropdown is still scanned)."""
    text = (
        "## Findings\n\n<details>\nC1 leaks here\n</details>\n\n"
        "## Data\n\n### Trained on\n\n<details>\nC1 verbatim row\n</details>\n\n"
        "## Reproducibility\n\nok\n"
    )
    stripped = audit.strip_data_example_blocks(text)
    # The Findings details survives (still scanned downstream).
    assert "C1 leaks here" in stripped
    # The Data details is dropped (exempt).
    assert "C1 verbatim row" not in stripped


def test_strip_data_example_blocks_ends_at_next_h2():
    """A `<details>` opened but never closed inside `## Data` stops being
    dropped at the next H2 — the exemption never silently widens past
    the section boundary."""
    text = (
        "## Data\n\n### Trained on\n\n<details>\nrow inside data\n\n"
        "## Reproducibility\n\nC1 in repro prose\n"
    )
    stripped = audit.strip_data_example_blocks(text)
    assert "row inside data" not in stripped  # dropped (inside Data details)
    assert "C1 in repro prose" in stripped  # NOT dropped (past the section)


# ─── The live single-body path audits unconditionally (no gate) ─────────


def test_single_body_audit_unconditional(capsys):
    """`_audit_single_body` (the `--task` / file-path live path) audits
    regardless of shape — it never consults the `is_v2` gate. An
    unstructured body with an anti-pattern still FAILs."""
    leaky = "# Title\n\nThe C1 condition leaked badly here.\n"
    rc = audit._audit_single_body(leaky)
    assert rc == 1
    out = capsys.readouterr().out
    assert "FAIL" in out


def test_single_body_audit_clean_v3_passes(capsys):
    """A clean v3 body (codes only inside `## Data`) PASSes the live
    single-body audit."""
    rc = audit._audit_single_body(V3_BODY_WITH_DATA_CODES)
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS" in out


# ─── Bracketed-CI bounds in reader-facing prose (Lens 7; incident #637) ──
#
# The bare `[+0.169, +0.437]` form (no trailing CI verb/unit) is the same
# banned construct as `value ± err`. The original `interval_inline` regex
# only fired on `slope[...]` or a bracketed pair IMMEDIATELY followed by
# `excludes`/`includes`/`pp`/`%`/`(`/`on `, so a bare pair slipped past the
# audit and was caught only by the LM critic (REVISE round on #637). The
# broadened regex must flag the bare form in Takeaways / What I ran /
# Findings prose, while keeping it out of: Reproducibility / Data tables
# (table-cell exemption), figure-caption blockquotes, and the
# finding-internal "Why this test" definition sentence (Lens 7 carve-outs).


def test_bare_bracketed_ci_in_takeaways_is_flagged():
    """A bare `[+0.169, +0.437]` bound in a `## Takeaways` bullet trips
    `interval_inline` (incident #637 — it previously slipped past)."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "- Headline finding: the implant installs cleanly across three seeds.",
        "- Headline finding: the lift is [+0.169, +0.437] over baseline.",
    )
    findings = audit.audit_body(leaky)
    assert "interval_inline" in findings, findings
    assert any("0.169" in s for s in findings["interval_inline"]), findings


def test_bare_bracketed_ci_in_findings_prose_is_flagged():
    """The bare bracketed-CI form in finding setup/read prose is flagged."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The 95% CI on the mean lift is [-0.02, 0.41] across seeds.",
    )
    findings = audit.audit_body(leaky)
    assert "interval_inline" in findings, findings


def test_integer_bound_bracketed_ci_is_flagged():
    """An integer-bound CI like `[1, 5]` (no decimal point) is just as
    banned — the broadened regex must not require a decimal."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The bootstrap bound spans [1, 5] points across seeds.",
    )
    findings = audit.audit_body(leaky)
    assert "interval_inline" in findings, findings


def test_unicode_minus_bracketed_ci_in_findings_prose_is_flagged():
    """A bracketed CI whose lower bound carries the typographic Unicode minus
    (codepoint U+2212) in finding read prose trips `interval_inline`. The
    pre-#649 ASCII-only sign class silently missed it: in #649 the audit
    caught the 3 ASCII-sign CIs in the body but slipped past the 2 whose
    lower bound used U+2212."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The prior-to-change correlation is flat at [−0.07, +0.33].",  # noqa: RUF001
    )
    findings = audit.audit_body(leaky)
    assert "interval_inline" in findings, findings
    assert any("−0.07" in s for s in findings["interval_inline"]), findings  # noqa: RUF001


def test_unicode_minus_bracketed_ci_in_takeaways_is_flagged():
    """The same U+2212-signed CI in a `## Takeaways` bullet is flagged."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "- Headline finding: the implant installs cleanly across three seeds.",
        "- Headline finding: the CHANGE effect is flat, [−0.31, +0.08].",  # noqa: RUF001
    )
    findings = audit.audit_body(leaky)
    assert "interval_inline" in findings, findings
    assert any("−0.31" in s for s in findings["interval_inline"]), findings  # noqa: RUF001


def test_unicode_minus_effect_size_pp_is_flagged():
    """A negative effect size rendered with the Unicode minus (codepoint
    U+2212) trips `effect_size_pp`. Mirrors the `interval_inline` sign-class
    fix (#649): the ASCII-only sign class previously missed it."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The held-out probes regress by Δ = −5pp across seeds.",  # noqa: RUF001
    )
    findings = audit.audit_body(leaky)
    assert "effect_size_pp" in findings, findings


def test_clean_v3_body_has_no_interval_inline_finding():
    """The unmodified clean exemplar (interval-only forms live in the
    Reproducibility / Data tables, none in prose) is interval-clean."""
    findings = audit.audit_body(V3_BODY_WITH_DATA_CODES)
    assert "interval_inline" not in findings, findings


def test_bracketed_ci_in_reproducibility_table_is_exempt():
    """A bracketed-CI in the `## Reproducibility` Parameters table is a
    spec-compliant interval form (table-cell exemption) and must NOT
    trip `interval_inline`."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "| Base model | Qwen-2.5-7B-Instruct |",
        "| Base model | Qwen-2.5-7B-Instruct |\n| Lift 95% CI | [+0.169, +0.437] |",
    )
    findings = audit.audit_body(body)
    assert "interval_inline" not in findings, findings


def test_band_stop_notation_in_prose_is_exempt():
    """A bracketed integer interval immediately followed by a `nat` unit
    (`band-stop [5,12] nat`) is a marker install / band-stop TARGET BAND,
    not a credence interval of an estimate, so it must NOT trip
    `interval_inline` even in reader-facing prose (#653)."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The marker adapter trained under a band-stop [5,12] nat schedule.",
    )
    findings = audit.audit_body(body)
    assert "interval_inline" not in findings, findings


def test_band_stop_carveout_does_not_suppress_a_real_ci_in_same_prose():
    """Precision guard: the `nat`-unit carve-out is narrow — a genuine
    bracketed CI in the SAME sentence position (no `nat` unit) still
    trips `interval_inline`, so the lookahead does not over-suppress."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The bootstrap bound spans [5, 12] points across seeds.",
    )
    findings = audit.audit_body(body)
    assert "interval_inline" in findings, findings


def test_bracketed_ci_in_data_table_is_exempt():
    """A bracketed-CI inside a `## Data` capsule example table is exempt
    (table-cell + Data verbatim-content exemptions both apply)."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "| Positive | C1 | A normal answer. |",
        "| Positive | C1 | A normal answer. CI [0.1, 0.4]. |",
    )
    findings = audit.audit_body(body)
    assert "interval_inline" not in findings, findings


def test_bracketed_ci_in_figure_caption_is_exempt():
    """A bracketed-CI inside a figure-caption blockquote (`> **Figure.**
    ...`) is a chart-annotation carve-out and must NOT trip the scan."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "> **Figure.** *The treatment lifts alignment over baseline at every seed.*",
        "> **Figure.** *The treatment lifts alignment, 95% CI [+0.169, +0.437].*",
    )
    findings = audit.audit_body(body)
    assert "interval_inline" not in findings, findings


def test_bracketed_ci_in_why_this_test_line_is_exempt():
    """A bracketed-CI in the finding-internal 'Why this test' definition
    sentence is the named Lens 7 exception and must NOT trip the scan."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The lift holds at every seed in the held-out evaluation.\n\n"
        "**Why this test:** the bootstrap CI [+0.169, +0.437] is the "
        "registered interval defining the test.",
    )
    findings = audit.audit_body(body)
    assert "interval_inline" not in findings, findings


def test_strip_interval_inline_exempt_lines_blanks_only_carveouts():
    """`_strip_interval_inline_exempt_lines` blanks blockquote + Why-this-test
    lines, leaves ordinary prose intact."""
    text = (
        "A prose line with [0.1, 0.4].\n"
        "> **Figure.** caption with [0.1, 0.4].\n"
        "**Why this test:** the CI [0.1, 0.4] defines it.\n"
        "Another prose line with [0.2, 0.5].\n"
    )
    stripped = audit._strip_interval_inline_exempt_lines(text)
    lines = stripped.splitlines()
    assert lines[0] == "A prose line with [0.1, 0.4]."  # prose kept
    assert lines[1] == ""  # caption blanked
    assert lines[2] == ""  # Why-this-test blanked
    assert lines[3] == "Another prose line with [0.2, 0.5]."  # prose kept


def test_original_interval_inline_forms_still_flagged():
    """Regression guard: the broadened regex must not drop the original
    `slope[...]` and trailing-token forms."""
    slope = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The fitted slope[0.1, 0.4] is positive across seeds.",
    )
    assert "interval_inline" in audit.audit_body(slope), "slope form regressed"
    trailing = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The interval [0.1, 0.4] excludes zero across seeds.",
    )
    assert "interval_inline" in audit.audit_body(trailing), "trailing-token form regressed"


def test_inline_backtick_bracketed_ci_in_prose_is_flagged():
    """An inline-backtick-wrapped CI in reader-facing prose (the verbatim
    #667 line-166 form ``CI `[-0.295, +0.083]` ``) trips `interval_inline`.
    `strip_code` previously blanked the inline-backtick span, hiding the CI
    from the scan; the `interval_inline` path now uses
    `strip_fenced_code_only`, which keeps inline backticks."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "Sycophancy's sibling crosses zero (CI `[−0.295, +0.083]`), "  # noqa: RUF001
        "matching its null.",
    )
    findings = audit.audit_body(leaky)
    assert "interval_inline" in findings, findings
    assert any("0.295" in s for s in findings["interval_inline"]), findings


def test_strip_fenced_code_only_keeps_inline_backticks():
    """`strip_fenced_code_only` removes fenced ```...``` blocks but keeps
    inline-backtick spans (unlike `strip_code`, which blanks both)."""
    text = "Prose `[0.1, 0.4]` here.\n```\nfenced [9, 9] body\n```\nAfter."
    out = audit.strip_fenced_code_only(text)
    assert "`[0.1, 0.4]`" in out  # inline kept
    assert "fenced [9, 9] body" not in out  # fenced stripped
    # contrast: strip_code blanks the inline span
    assert "[0.1, 0.4]" not in audit.strip_code(text)


def test_inline_backtick_condition_label_in_prose_not_flagged():
    """Scope guard: the fix routes ONLY `interval_inline` through
    `strip_fenced_code_only`. `condition_labels` keeps using `strip_code`
    (inline backticks blanked), so an inline-backtick `C1` reference in
    prose is a legitimate accepted form and must NOT trip `condition_labels`."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The result reproduces the `C1` arm at every seed.",
    )
    findings = audit.audit_body(leaky)
    assert "condition_labels" not in findings, findings


# ─── Bound-form CI leak (#952 → #1015) ───────────────────────────────────
#
# Lens 7's own example (`upper bound = 0.0021`) bans the BOUND-FORM CI leak
# — a named CI endpoint stated as a number in prose with no brackets — but
# all three prior `interval_inline` alternatives required literal square
# brackets, so #952 r1's "the interval's upper bound (+0.023) excludes the
# 0.05 margin" reached the LM critic unflagged. Alternative (4) closes it:
# connectors `( num )` and `= num` only (the `of`-connector stays legal —
# budget / band-threshold prose), number = sign-or-decimal (count-integers
# stay legal), [Uu]/[Ll] case pair (case-sensitive scan), U+2212 in the
# sign class, `[\s-]` joiner, leading `\b` (no embedded word tails). The
# pattern is incident/spec-derived; the corpus survey is the precision read.


def test_bound_form_paren_in_findings_prose_is_flagged():
    """The verbatim #952 r1 incident sentence — `upper bound (+0.023)` in
    finding read prose — trips `interval_inline` (the regression itself)."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "By that read, the interval's upper bound (+0.023) excludes the 0.05 margin.",
    )
    findings = audit.audit_body(leaky)
    assert "interval_inline" in findings, findings
    assert any("0.023" in s for s in findings["interval_inline"]), findings


def test_bound_form_equals_in_takeaways_is_flagged():
    """The Lens 7 spec's own named example `upper bound = 0.0021` in a
    `## Takeaways` bullet trips `interval_inline`."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "- Headline finding: the implant installs cleanly across three seeds.",
        "- Headline finding: the margin's upper bound = 0.0021 across seeds.",
    )
    findings = audit.audit_body(leaky)
    assert "interval_inline" in findings, findings
    assert any("0.0021" in s for s in findings["interval_inline"]), findings


def test_bound_form_unicode_minus_is_flagged():
    """A bound-form endpoint signed with the typographic Unicode minus
    (codepoint U+2212) in the PAREN branch trips `interval_inline` — the
    #649 sign-class blind spot must not reopen on the new alternative."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The pooled lower bound (−0.006) stays under the margin.",  # noqa: RUF001
    )
    findings = audit.audit_body(leaky)
    assert "interval_inline" in findings, findings
    assert any("−0.006" in s for s in findings["interval_inline"]), findings  # noqa: RUF001


def test_bound_form_unicode_minus_equals_branch_is_flagged():
    """The U+2212 sign must also be accepted in the EQUALS branch
    (`lower bound = <U+2212>0.006`), not just the paren branch."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The pooled lower bound = −0.006 under the pre-set margin.",  # noqa: RUF001
    )
    findings = audit.audit_body(leaky)
    assert "interval_inline" in findings, findings
    assert any("−0.006" in s for s in findings["interval_inline"]), findings  # noqa: RUF001


def test_bound_form_unsigned_decimal_paren_is_flagged():
    """The verbatim #540 corpus sentence — an UNSIGNED decimal in the paren
    branch — trips `interval_inline` (a real prior instance)."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The CI's upper bound (0.287) crosses that bar.",
    )
    findings = audit.audit_body(leaky)
    assert "interval_inline" in findings, findings
    assert any("0.287" in s for s in findings["interval_inline"]), findings


def test_bound_form_sentence_start_capital_is_flagged():
    """Sentence-start `Upper bound = 0.0021` is caught by the [Uu]/[Ll]
    case pair — the category scans case-sensitively (flags=0)."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "Upper bound = 0.0021 by that read, at every seed.",
    )
    findings = audit.audit_body(leaky)
    assert "interval_inline" in findings, findings


def test_bound_form_hyphenated_is_flagged():
    """The hyphenated joiner `upper-bound (+0.02)` is caught by the
    `[\\s-]` class (mirrors `bit_byte_identical`)."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The upper-bound (+0.02) read repeats across seeds.",
    )
    findings = audit.audit_body(leaky)
    assert "interval_inline" in findings, findings


def test_bound_form_signed_integer_is_flagged():
    """A SIGNED integer endpoint (`upper bound (+5)`) matches the
    sign-plus-integer branch of the number atom — the sign alone qualifies."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The pooled upper bound (+5) excludes zero across seeds.",
    )
    findings = audit.audit_body(leaky)
    assert "interval_inline" in findings, findings


def test_bound_word_without_number_not_flagged():
    """Bare `bound` prose with no numeric connector stays legal."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "We report an upper bound on compute and an error bound with no number.",
    )
    findings = audit.audit_body(body)
    assert "interval_inline" not in findings, findings


def test_bound_of_connector_not_flagged():
    """The `of`-connector is deliberately excluded: budget prose
    ("upper bound of 4 GPU-hours") and band-threshold prose (#267's
    "exceed the upper bound of 0.6") are legitimate."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "We set an upper bound of 4 GPU-hours; all exceed the upper bound of 0.6.",
    )
    findings = audit.audit_body(body)
    assert "interval_inline" not in findings, findings


def test_count_integer_bound_not_flagged():
    """An UNSIGNED count-style integer (`retry upper bound = 5`) stays
    legal — the number atom requires a sign OR a decimal point."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "We set the retry upper bound = 5 for the judge loop.",
    )
    findings = audit.audit_body(body)
    assert "interval_inline" not in findings, findings


def test_word_sense_bound_not_flagged():
    """Non-statistical word senses of `bound` stay legal: participle
    (`lower-bounded by zero`), compound (`outward-bound`), and the
    attachment sense (`bound to the persona`)."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The rate is lower-bounded by zero; the outward-bound probe set "
        "stays bound to the persona.",
    )
    findings = audit.audit_body(body)
    assert "interval_inline" not in findings, findings


def test_embedded_word_tail_bound_not_flagged():
    """The leading `\\b` keeps embedded word tails out: `supper bound
    (0.2)` / `flower bound (0.2)` must NOT match the upper/lower branch."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The supper bound (0.2) and flower bound (0.2) phrasings are nonsense.",
    )
    findings = audit.audit_body(body)
    assert "interval_inline" not in findings, findings


def test_bare_bracket_fixture_does_not_fire_bound_alternative():
    """No-interaction guard: `The bootstrap bound spans [1, 5] points`
    (the existing alt-3 fixture) is flagged via the BARE-PAIR alternative
    only — every reported sample is the bracketed pair, never a
    `bound`-form match (alt-4 has no upper/lower word to anchor on)."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The bootstrap bound spans [1, 5] points across seeds.",
    )
    findings = audit.audit_body(leaky)
    assert "interval_inline" in findings, findings
    assert all("bound" not in s for s in findings["interval_inline"]), findings


def test_bound_form_in_reproducibility_table_is_exempt():
    """A bound-form value inside a `## Reproducibility` Parameters table
    cell inherits the table-cell exemption (`_blank_table_rows`)."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "| Base model | Qwen-2.5-7B-Instruct |",
        "| Base model | Qwen-2.5-7B-Instruct |\n| mc_ci | upper bound = 0.287 |",
    )
    findings = audit.audit_body(body)
    assert "interval_inline" not in findings, findings


def test_bound_form_in_figure_caption_is_exempt():
    """A bound-form value inside a figure-caption blockquote inherits the
    chart-annotation carve-out (`_strip_interval_inline_exempt_lines`)."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "> **Figure.** *The treatment lifts alignment over baseline at every seed.*",
        "> **Figure.** *The treatment lifts alignment; upper bound (+0.023).*",
    )
    findings = audit.audit_body(body)
    assert "interval_inline" not in findings, findings


def test_bound_form_in_why_this_test_line_is_exempt():
    """A bound-form value in the finding-internal 'Why this test'
    definition sentence is the named Lens 7 exception."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The lift holds at every seed in the held-out evaluation.\n\n"
        "**Why this test:** the pre-set margin's upper bound = 0.0021 "
        "defines the test.",
    )
    findings = audit.audit_body(body)
    assert "interval_inline" not in findings, findings


# ─── Pre-registration mentions scoped to Lens 7 prose sections (#623) ────
#
# Lens 7 (clean-result-critic) scopes the pre-registration-mention ban to
# `## Takeaways` / `## What I ran` / `## Findings` prose ONLY and explicitly
# permits pre-reg threshold values / procedural notes elsewhere ("Pre-reg
# threshold values can sit in the parameters table."). The `pre_reg` regex
# previously scanned the whole body, so a procedural "dropped pre-registered"
# sentence in `## Data` / `## Reproducibility` prose fired a FALSE positive
# the critic had to hand-adjudicate every round (incident #623: an
# `## Data → ### Evaluated with` mention tripped the audit although Lens 7
# exempts that section).


def test_pre_reg_in_data_prose_is_exempt():
    """A `pre-registered` procedural mention in `## Data` PROSE (not an
    example block) on a v3 body must NOT trip `pre_reg` — Lens 7 scopes
    the ban to Takeaways / What I ran / Findings (incident #623)."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "Established mix (tier 2), 2,000 rows, on-policy base completions.",
        "Established mix (tier 2); the assistant baseline-self is dropped "
        "pre-registered, leaving n = 35 for the correlation.",
    )
    findings = audit.audit_body(body)
    assert "pre_reg" not in findings, findings


def test_pre_reg_in_reproducibility_prose_is_exempt():
    """A `pre-registered` mention in `## Reproducibility` prose on a v3
    body must NOT trip `pre_reg` (Lens 7 prose-section scope)."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "**Compute:** 1x H100, 47 min.",
        "**Compute:** 1x H100, 47 min. The drop rule was pre-registered.",
    )
    findings = audit.audit_body(body)
    assert "pre_reg" not in findings, findings


def test_pre_reg_in_takeaways_prose_is_flagged():
    """The SAME `pre-registered` mention in a `## Takeaways` bullet on a
    v3 body STILL trips `pre_reg` — the carve-out is section-scoped, not a
    blanket whitelist of the term."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "- Headline finding: the implant installs cleanly across three seeds.",
        "- Headline finding: the pre-registered drop leaves n = 35.",
    )
    findings = audit.audit_body(body)
    assert "pre_reg" in findings, findings
    assert any("pre-registered" in s.lower() for s in findings["pre_reg"]), findings


def test_pre_reg_in_findings_prose_is_flagged():
    """A `pre-registered` mention in `## Findings` read prose on a v3 body
    still trips `pre_reg`."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The lift, pre-registered as the primary endpoint, holds at every seed.",
    )
    findings = audit.audit_body(body)
    assert "pre_reg" in findings, findings


def test_pre_reg_whole_body_scan_preserved_for_legacy_bodies():
    """A non-v3 (legacy / unstructured) body keeps the prior whole-body
    `pre_reg` scan — the section scope is a v3 shape, and skipping it for
    legacy bodies prevents silently blanking an entire legacy body's
    prose. A `pre-registered` mention in legacy prose still FAILs."""
    legacy = (
        "# Some legacy title\n\n## AI TL;DR\n\nclean prose.\n\n"
        "## AI Summary\n\nThe drop was pre-registered before the run.\n"
    )
    findings = audit.audit_body(legacy)
    assert "pre_reg" in findings, findings


def test_pre_reg_as_registered_in_takeaways_is_flagged():
    """The bare 'As registered, ...' phrasing family (incident: a #763 body
    carried 'As registered, SUCCESS was not met' twice and the LM critic had
    to catch it manually) trips `pre_reg` in a scanned prose section. The
    capitalized form also pins the re.IGNORECASE application."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "- Headline finding: the implant installs cleanly across three seeds.",
        "- As registered, SUCCESS was not met: the lift is null at every seed.",
    )
    assert body != V3_BODY_WITH_DATA_CODES
    findings = audit.audit_body(body)
    assert "pre_reg" in findings, findings
    assert any("as registered" in s.lower() for s in findings["pre_reg"]), findings


def test_pre_reg_innocuous_registered_usage_not_flagged():
    """Plain 'registered' verb usages in a SCANNED prose section ('registered
    the adapter on HF', 'was registered in WandB', 'alias registered') must
    NOT trip `pre_reg` — the new alternation is anchored to the literal
    'as registered' bigram, and the leading `\\b` fails inside 'was'/'alias'.
    #1553 adds a noun-before-verb guard for the new `estimators?` head noun
    ('the estimator was registered in WandB' — preposition-lookahead form)."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "- Headline finding: the implant installs cleanly across three seeds.",
        "- Registered the adapter on HF; the run was registered in WandB; alias registered "
        "too; the estimator was registered in WandB.",
    )
    assert body != V3_BODY_WITH_DATA_CODES
    findings = audit.audit_body(body)
    assert "pre_reg" not in findings, findings


def test_pre_reg_as_registered_in_v4_results_prose_is_flagged():
    """The #763 incident phrasing in `## Results` PROSE on a v4-sentinel body
    trips `pre_reg` under the v4 branch of `_restrict_pre_reg_to_prose_sections`
    (whole-body scan minus GFM table rows — Lens 7 bans the mention in all
    four v4 H2s' prose). Assertion unchanged from the pre-branch era: the
    prose hit fired then (v4 fell through to the whole-body scan) and must
    keep firing now."""
    v4 = (
        "# Title (LOW confidence)\n<!-- clean-result-v4 -->\n\n"
        "## Takeaways\n\n- clean prose.\n\n## Goal\n\nclean.\n\n"
        "## Methodology\n\nclean.\n\n## Results\n\n"
        "As registered, SUCCESS was not met.\n"
    )
    findings = audit.audit_body(v4)
    assert "pre_reg" in findings, findings


# ─── v4 pre_reg scope: whole-body-minus-GFM-table-rows (#969) ────────────
#
# Under the v4 spec (`<!-- clean-result-v4 -->`) Lens 7 bans pre-reg mentions
# in ALL FOUR H2s' prose (Takeaways / Goal / Methodology / Results) and
# permits threshold values ONLY in the Methodology Training hyperparameter
# table. `_restrict_pre_reg_to_prose_sections` implements this as a
# whole-body scan with every positively-detected GFM table row blanked —
# deliberately wider than Lens 7's named-table letter (see
# `test_pre_reg_in_v4_results_table_is_deliberately_exempt`).

# A compact v4-shape body: H1 + v4 sentinel + top-of-body Methodology link
# + the four H2s (a **Training:** GFM hparam table under `## Methodology`
# with a benign Source column; a `### <result>` under `## Results`) + the
# `**Repro:**` / `**Context:**` footer with an `- Originating prompt:`
# blockquote. Deliberately clean of every audit category at baseline;
# tests mutate it via targeted `.replace()` with a body != fixture guard.
V4_BODY_CLEAN = """\
---
title: v4 body for the pre_reg scan-scope tests
kind: experiment
goal: Exercise the v4 pre_reg whole-body-minus-tables scope
---
# Some claim about a finding (MODERATE confidence)

<!-- clean-result-v4 -->

**Methodology:** [docs/methodology/issue_9999.md](docs/methodology/issue_9999.md)

## Takeaways

- Headline finding: the implant installs cleanly across three seeds.
- Secondary finding: no measurable regression on the held-out probes.
- Caveat that binds interpretation: single model family, three seeds.

## Goal

**This experiment in context:** tests whether the prior effect generalises
to benchmark Z under the same training mix.

**Broader narrative:** which context factors predict fine-tuning leakage.

## Methodology

**Design:** three seeds; baseline vs treatment; the single variable is the data mix.

**Training:**

| Hyperparameter | Value | Source |
|---|---|---|
| learning rate | 5e-6 | prior issue |
| epochs | 1 | prior issue |

**Evaluation:** judge-scored rate on the held-out probes.

## Results

### Main result

The lift holds at every seed in the held-out evaluation.

---

**Repro:** 1x A100, 47 min; code at commit deadbeef.

**Context:**

- Created: 2026-07-04
- Originating prompt:

> Close the v4 gap in the pre-reg prose-scan scope.
"""


def test_pre_reg_v4_clean_body_passes():
    """Fixture-validity guard: the baseline v4 body carries no `pre_reg`
    hit, so every mutation test below isolates its own injected phrase."""
    findings = audit.audit_body(V4_BODY_CLEAN)
    assert "pre_reg" not in findings, findings


def test_pre_reg_in_v4_methodology_prose_is_flagged():
    """A `pre-registered` mention in `## Methodology` **Design:** PROSE on a
    v4 body trips `pre_reg` — the case the original candidate's two-section
    (`takeaways`,`results`) scope would have wrongly exempted; Lens 7 bans
    the mention in all four v4 H2s' prose."""
    body = V4_BODY_CLEAN.replace(
        "**Design:** three seeds; baseline vs treatment; the single variable is the data mix.",
        "**Design:** three seeds; the pre-registered drop rule leaves n = 35.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" in findings, findings


def test_pre_reg_in_v4_goal_prose_is_flagged():
    """A `pre-registered` mention in `## Goal` prose on a v4 body trips
    `pre_reg` (same four-H2 prose ban)."""
    body = V4_BODY_CLEAN.replace(
        "**Broader narrative:** which context factors predict fine-tuning leakage.",
        "**Broader narrative:** the pre-registered leakage-prediction question.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" in findings, findings


def test_pre_reg_in_v4_takeaways_prose_is_flagged():
    """A `pre-registered` mention in a `## Takeaways` bullet on a v4 body
    trips `pre_reg` — redundant coverage (the v4 branch has no section
    discrimination) kept as cheap symmetry with the v3 Takeaways pin."""
    body = V4_BODY_CLEAN.replace(
        "- Headline finding: the implant installs cleanly across three seeds.",
        "- Headline finding: the pre-registered drop leaves n = 35.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" in findings, findings


def test_pre_reg_threshold_in_v4_hparam_table_is_exempt():
    """A pre-reg threshold phrase inside the Methodology **Training:** GFM
    hyperparameter table on a v4 body does NOT trip `pre_reg` — the one
    surface Lens 7 explicitly permits, and the demonstrated v4
    false-positive class this fix exists for."""
    body = V4_BODY_CLEAN.replace(
        "| epochs | 1 | prior issue |",
        "| epochs | 1 | prior issue |\n| stopping floor | 0.80 | pre-registered floor (#612) |",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" not in findings, findings


def test_pre_reg_in_v4_results_table_is_deliberately_exempt():
    """A pre-reg phrase inside a `## Results` GFM table row on a v4 body
    (no other pre-reg prose) does NOT fire — RECORDING THE DELIBERATE
    ALL-TABLES WIDTH: `_blank_table_rows` blanks every positively-detected
    GFM table row, wider than Lens 7's letter (which names only the
    Methodology Training hyperparameter table as permitted); the LM
    clean-result-critic remains the backstop for a pre-reg mention smuggled
    into a non-Methodology table."""
    body = V4_BODY_CLEAN.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The lift holds at every seed in the held-out evaluation.\n\n"
        "| criterion | source |\n"
        "|---|---|\n"
        "| success floor | pre-registered floor (#612) |",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" not in findings, findings


def test_pre_reg_table_exemption_does_not_hide_prose_hit_v4():
    """A v4 body with BOTH a table threshold row AND a Methodology-prose
    mention still fires, and the surviving sample is the PROSE match (the
    distinct 'As registered' phrasing) — the table row is blanked, never
    the prose."""
    body = V4_BODY_CLEAN.replace(
        "| epochs | 1 | prior issue |",
        "| epochs | 1 | prior issue |\n| stopping floor | 0.80 | pre-registered floor (#612) |",
    ).replace(
        "**Design:** three seeds; baseline vs treatment; the single variable is the data mix.",
        "**Design:** As registered, three seeds; baseline vs treatment.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" in findings, findings
    assert any("as registered" in s.lower() for s in findings["pre_reg"]), findings
    # The table row's 'pre-registered' was blanked — no sample comes from it.
    assert not any("pre-registered" in s.lower() for s in findings["pre_reg"]), findings


def test_pre_reg_in_v4_footer_origin_prompt_is_exempt():
    """A `pre-registered` phrase inside the footer `**Context:**`
    originating-prompt blockquote on a v4 body does NOT trip `pre_reg` —
    pins that the sentinel-agnostic `strip_context_blockquotes` (#597/#651)
    covers v4 footers end-to-end (the strip runs BEFORE the pre_reg scope
    function sees the text)."""
    body = V4_BODY_CLEAN.replace(
        "> Close the v4 gap in the pre-reg prose-scan scope.",
        "> Close the pre-registered v4 gap in the prose-scan scope.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" not in findings, findings


def test_pre_reg_in_v4_footer_repro_row_is_flagged():
    """A `pre-registered` mention in the `**Repro:**` footer row on a v4
    body STILL fires — pins the deliberate decision to keep the non-prompt
    footer SCANNED (no incident motivates exempting it; the conservative
    direction is a hand-adjudicated flag, never a silent exemption)."""
    body = V4_BODY_CLEAN.replace(
        "**Repro:** 1x A100, 47 min; code at commit deadbeef.",
        "**Repro:** 1x A100, 47 min; the pre-registered config at commit deadbeef.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" in findings, findings


def test_pre_reg_v4_sentinel_precedes_v3_gate():
    """A malformed body carrying BOTH sentinels with a pre-reg mention under
    `## Reproducibility` (a v3-EXEMPT section) still fires — pins the branch
    order (v4 checked BEFORE v3): the v4 sentinel declares the governing
    spec, and the v4 branch keeps every prose surface in scope."""
    dual = (
        "# Title (LOW confidence)\n"
        "<!-- clean-result-v4 -->\n<!-- clean-result-v3 -->\n\n"
        "## Takeaways\n\n- clean prose.\n\n"
        "## Reproducibility\n\nThe drop rule was pre-registered.\n"
    )
    findings = audit.audit_body(dual)
    assert "pre_reg" in findings, findings


def test_pre_reg_whole_body_scan_preserved_for_v2_sentinel_bodies():
    """A `<!-- clean-result-v2 -->` body with a pre-reg mention INSIDE a GFM
    table row still fires — pins that the v4 table exemption did NOT leak to
    v2/legacy bodies (their whole-body scan is preserved verbatim,
    table rows included)."""
    v2 = (
        "# Legacy v2 title\n<!-- clean-result-v2 -->\n\n"
        "## Human TL;DR\n\nclean prose.\n\n## TL;DR\n\nclean.\n\n"
        "## Reproducibility\n\n"
        "| Parameter | Value |\n|---|---|\n| drop rule | pre-registered |\n"
    )
    findings = audit.audit_body(v2)
    assert "pre_reg" in findings, findings


# ─── pre_reg: bare 'registered <noun>' forms (#1345 escape, fix #1419) ────


def test_pre_reg_bare_registered_noun_incident_strings_are_flagged():
    """All six #1345 Lens 7 incident forms — 4 of 6 with intervening tokens
    between 'registered' and the head noun — trip `pre_reg` in v4 Results
    prose (#1419). The seventh phrase pins family closure: an
    intervening-token 'hypothesis' form the adjacency-only branch missed."""
    phrases = [
        "the registered same-map-different-coordinates verdict",
        "evaluated the registered verdict lattice",
        "fail the registered 0.05 same-operator margin",
        "fire the registered reparameterized verdict",
        "The registered existence read passes",
        "the plan's registered degeneracy companion",
        "the registered primary hypothesis",
    ]
    for phrase in phrases:
        body = V4_BODY_CLEAN.replace(
            "The lift holds at every seed in the held-out evaluation.",
            f"{phrase}.",
        )
        assert body != V4_BODY_CLEAN
        findings = audit.audit_body(body)
        assert "pre_reg" in findings, (phrase, findings)
        assert any("registered" in s.lower() for s in findings["pre_reg"]), (phrase, findings)


def test_pre_reg_bare_registered_noun_1090_escape_strings_are_flagged():
    """All eight #1090 round-6 escape strings trip `pre_reg` in v4 Results
    prose (#1475). The two 'band' strings isolate the token-class widening
    (`band` is a #1419 noun; only the widened token group lets the range
    token '0.60-0.85' — ASCII hyphen, or its U+2212 minus-sign variant —
    be consumed as an intervening token); the other six isolate the seven
    nouns added by #1475 (cut/path/clause/control/lever/bar[/smoke])."""
    phrases = [
        "the registered 0.60-0.85 band",
        "the registered 0.60−0.85 band",  # noqa: RUF001
        "registered kill path",
        "registered per-arm abort clause",
        "registered 0.30 cut",
        "registered install-strength control",
        "registered unrun lever",
        "registered 10% kill bar",
    ]
    for phrase in phrases:
        body = V4_BODY_CLEAN.replace(
            "The lift holds at every seed in the held-out evaluation.",
            f"{phrase}.",
        )
        assert body != V4_BODY_CLEAN
        findings = audit.audit_body(body)
        assert "pre_reg" in findings, (phrase, findings)
        assert any("registered" in s.lower() for s in findings["pre_reg"]), (phrase, findings)


def test_pre_reg_bare_registered_noun_in_v4_takeaways_is_flagged():
    """Incident 1's actual placement (#1345 Takeaways bullet 2) trips
    `pre_reg` under the v4 whole-body scope."""
    body = V4_BODY_CLEAN.replace(
        "- Headline finding: the implant installs cleanly across three seeds.",
        "- Headline: the registered same-map-different-coordinates verdict held.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" in findings, findings


def test_pre_reg_bare_registered_noun_in_v4_hparam_table_is_exempt():
    """The same bare form inside the Methodology hyperparameter TABLE stays
    exempt — the v4 table-blanking scope is preserved byte-unchanged."""
    body = V4_BODY_CLEAN.replace(
        "| epochs | 1 | prior issue |",
        "| stopping rule | registered 0.05 margin | plan gate |",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" not in findings, findings


def test_pre_reg_bare_registered_noun_verb_usages_not_flagged():
    """The verb register — 'registered <preposition>' and artifact-object
    forms — never trips the new branch (first-token preposition guard +
    pre-registration-specific noun list). NOTE (measured at plan time):
    determiner-first verb-objects on listed nouns ('the model registered
    a clear floor effect') WOULD flag — an accepted residual FP surface
    with 0/1,006-body corpus attestation, kept because 6 corpus hits of
    the same determiner-first shape are genuine jargon ('The plan
    registered a decision lattice')."""
    body = V4_BODY_CLEAN.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "I registered the adapter on HF; the run was registered in WandB; "
        "the hook registered by bootstrap fired; the mix is registered "
        "under the data repo; a registered trademark.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" not in findings, findings


def test_pre_reg_bare_registered_noun_does_not_bridge_sentences():
    """The intervening-token window never bridges a sentence or bullet
    boundary — both bridge shapes observed in the plan-time corpus dry-run
    ('registered conditions.\\n\\nThe honest read', 'registered unrun
    dial.\\n- Verdicts') stay clean. NOTE (#1475): the second fixture
    originally used 'lever', chosen when `lever` was unlisted; #1475 moved
    `levers?` into the head-noun list (making it a genuine WITHIN-sentence
    hit), so the fixture now exercises the same boundary property with the
    non-listed noun 'dial'."""
    body = V4_BODY_CLEAN.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "All cells were registered conditions.\n\nThe honest read is a null.\n\n"
        "It used a registered unrun dial.\n- Verdicts fired on every cell.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" not in findings, findings


def test_pre_reg_new_nouns_benign_verb_usages_not_flagged():
    """Benign verb-register shapes for the highest-FP-risk #1475 noun
    (`paths?`) stay clean: noun-BEFORE-verb ('the adapter paths registered
    on HF') and the first-token preposition-guard shape ('registered on HF
    under paths/')."""
    body = V4_BODY_CLEAN.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "the adapter paths registered on HF; artifacts registered on HF under paths/.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" not in findings, findings


def test_pre_reg_registered_interval_defining_the_test_not_flagged():
    """'the registered interval defining the test' is the sanctioned
    Why-this-test CI-definition register — since #1783 the exemption
    mechanism is the Why-this-test line strip (`_blank_why_this_test_lines`
    blanks the line from the pre_reg scan source), NOT noun omission:
    `intervals?`/`tests?` are now IN the head-noun alternation, and
    `pre_reg` stays silent because the whole line is blanked. This also
    pins the sanctioned register against future head-noun extensions."""
    body = V4_BODY_CLEAN.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "**Why this test:** the bootstrap CI is the registered interval defining the test.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" not in findings, findings


def test_pre_reg_registered_estimator_in_v4_results_prose_is_flagged():
    """The verbatim #1482 round-4 escape — 'The plotted floor is the
    registered fresh-4 estimator.' in a v4 body's ## Results H3 prose —
    trips `pre_reg` now that #1553 added `estimators?` to the head-noun
    alternation ('fresh-4' is one intervening modifier token)."""
    body = V4_BODY_CLEAN.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The plotted floor is the registered fresh-4 estimator.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" in findings, findings
    assert any("estimator" in s.lower() for s in findings["pre_reg"]), findings["pre_reg"]


def test_pre_reg_bare_registered_noun_1586_escape_strings_are_flagged():
    """The #1586 round-1 escape class trips `pre_reg` in v4 Results prose
    now that #1638 added `layers?`/`rungs?`/`windows?` to the head-noun
    alternation: the two #1586 incident forms, the #1333 corpus form
    (numeral AFTER the head noun), the #1005 rung form (two intervening
    modifier tokens), and the two #1332 window forms."""
    phrases = [
        "reported at the registered layer",
        "the registered layer",
        "shift DVs at the registered layer 25",
        "the parent's registered retry rung",
        "the plan's registered window",
        "the registered apply-gate window",
    ]
    for phrase in phrases:
        body = V4_BODY_CLEAN.replace(
            "The lift holds at every seed in the held-out evaluation.",
            f"{phrase}.",
        )
        assert body != V4_BODY_CLEAN
        findings = audit.audit_body(body)
        assert "pre_reg" in findings, (phrase, findings)
        assert any("registered" in s.lower() for s in findings["pre_reg"]), (phrase, findings)


def test_pre_reg_1638_new_nouns_benign_verb_usages_not_flagged():
    """Benign verb-register shapes for the #1638 nouns stay clean: the
    first-token preposition guard covers 'registered at layer 20' /
    'registered on layer hooks' / 'registered in the dashboard config' /
    'registered under the sweep', including noun-BEFORE-verb subjects
    ('three windows registered in ...', 'the ladder rungs registered
    under ...'). NOTE (measured 2026-07-23, #1638): the mid-window-
    preposition / hook-register form 'registered a hook at layer 20'
    WOULD flag (the determiner + noun tokens consume window positions
    1-3, so 'layer' heads the match; the lookahead guards only the FIRST
    token) — a documented accepted residual with 0/1,571-body corpus
    attestation, a pre-existing #1419-window property, deliberately NOT
    pinned here as a passing negative."""
    body = V4_BODY_CLEAN.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "the forward hook registered at layer 20 fired; activations "
        "registered on layer hooks; three windows registered in the "
        "dashboard config; the ladder rungs registered under the sweep.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" not in findings, findings


# ─── pre_reg: the #1092 escape set (fix #1783) ────────────────────────────
#
# Seven bare 'registered <noun>' phrasings in #1092's promoted v4 body
# PASSed the audit across rounds 1-3 (caught only by the LM critic's
# Lens 7 read); #1783 adds their head nouns
# (preconditions?/curves?/designs?/legs?/subsamples?/intervals?/tests?)
# to the alternation and blanks `**Why this test:**` lines from the
# pre_reg scan source so the sanctioned CI-definition register stays
# exempt (the mechanism that makes `intervals?`/`tests?` safe to add).


@pytest.mark.parametrize(
    ("phrase", "expected_match"),
    [
        ("the registered downgrade precondition", "registered downgrade precondition"),
        ("the registered confidence intervals", "registered confidence intervals"),
        ("the registered trait-per-factor leg", "registered trait-per-factor leg"),
        ("a registered subsample", "registered subsample"),
        (
            "two registered operator-identity residual tests",
            "registered operator-identity residual tests",
        ),
        (
            "the registered monitoring-gap group-size curve",
            "registered monitoring-gap group-size curve",
        ),
        ("the registered design", "registered design"),
    ],
)
def test_pre_reg_1092_escape_phrasings_are_flagged(phrase: str, expected_match: str):
    """Each of the seven verbatim #1092 escape phrasings trips `pre_reg`
    in v4 Results prose through the FULL gate pipeline (`audit_body`),
    and the finding sample carries the expected match text. The
    'operator-identity residual tests' case pins the 2-intervening-token
    window; the hyphenated compounds ride the modifier token class."""
    body = V4_BODY_CLEAN.replace(
        "The lift holds at every seed in the held-out evaluation.",
        f"{phrase}.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" in findings, (phrase, findings)
    assert any(expected_match in s for s in findings["pre_reg"]), (phrase, findings)


def test_pre_reg_sanctioned_register_on_why_this_test_line_exempt_v4():
    """The sanctioned Why-this-test CI-definition register — with its
    bracketed CI — does NOT trip `pre_reg` in a SCANNED v4 prose section:
    `_blank_why_this_test_lines` blanks the line from the pre_reg scan
    source before the generation branch (#1783), even though
    `intervals?`/`tests?` are now in the head-noun alternation."""
    body = V4_BODY_CLEAN.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "**Why this test:** the bootstrap CI [+0.1, +0.4] is the "
        "registered interval defining the test.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" not in findings, findings


def test_pre_reg_sanctioned_register_on_why_this_test_line_exempt_v3():
    """The same sanctioned register inside a v3 `## Findings` SCANNED
    prose section stays exempt — the Why-this-test strip applies to ALL
    generations (the register historically lives in v3 Findings), and
    blanking only ever REDUCES findings, so grandfathered bodies are
    never newly FAILed."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The lift holds at every seed in the held-out evaluation.\n\n"
        "**Why this test:** the bootstrap CI [+0.1, +0.4] is the "
        "registered interval defining the test.",
    )
    assert body != V3_BODY_WITH_DATA_CODES
    findings = audit.audit_body(body)
    assert "pre_reg" not in findings, findings


def test_pre_reg_preregistered_on_why_this_test_line_no_longer_flagged():
    """DOCUMENTS THE #1783 DEGRADATION: a `pre-registered` mention placed
    on a Why-this-test line no longer trips `pre_reg` — the whole line is
    blanked from the scan source, so a pre-reg mention smuggled onto it
    escapes the mechanical gate. The LM clean-result-critic Lens 7 is the
    backstop — the same accepted trade as the v4 table-row blanking
    (cf. `test_pre_reg_in_v4_results_table_is_deliberately_exempt`)."""
    body = V4_BODY_CLEAN.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "**Why this test:** the pre-registered interval [0.1, 0.4] defines the pass bar.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" not in findings, findings


def test_pre_reg_1783_new_nouns_benign_verb_usages_not_flagged():
    """Benign verb-register shapes for the #1783 nouns stay clean:
    noun-BEFORE-verb subjects ('the unit tests registered in pytest',
    'three curves registered in WandB') and the first-token preposition
    guard ('registered under the sweep'). Corpus-measured 2026-07-29
    (1,715 bodies): 0 benign verb-use false positives among the 42 new
    match starts."""
    body = V4_BODY_CLEAN.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "the unit tests registered in pytest fired; three curves "
        "registered in WandB; the eval legs registered under the sweep.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" not in findings, findings


# ─── pre_reg: the #1769 escape (fix #1831) ────────────────────────────────
#
# #1769's clean-result draft carried 'the registered ceiling check' x4 (and
# bare 'registered ceiling') in reader-facing prose; the audit passed it
# clean on 2026-07-29 — 'ceiling'/'check' were absent from the head-noun
# alternation — and only the LM critic's Lens 6/7 read caught it. #1831
# adds `ceilings?`/`checks?`. The filing's `gates?` proposal was narrowed
# at plan time: `gates?` was already in the set ('registered ceiling gate'
# already matched via `gates?` with 'ceiling' as an intermediate token).


def test_pre_reg_registered_ceiling_check_in_takeaways_is_flagged():
    """The verbatim #1769 escape phrasing — 'the registered ceiling check'
    in a v4 `## Takeaways` bullet — trips `pre_reg` now that #1831 added
    `ceilings?`/`checks?`. The lazy intervening-token window stops at the
    EARLIER noun, so the match text is 'registered ceiling' (the same
    lazy-stop property as #1593's 'registered test paths')."""
    body = V4_BODY_CLEAN.replace(
        "- Headline finding: the implant installs cleanly across three seeds.",
        "- Headline finding: the registered ceiling check flagged the band.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" in findings, findings
    assert any("registered ceiling" in s.lower() for s in findings["pre_reg"]), findings


def test_pre_reg_bare_registered_ceiling_is_flagged():
    """Bare 'a registered ceiling' in v4 Results prose trips `pre_reg`
    (the `ceilings?` head noun with zero intervening tokens)."""
    body = V4_BODY_CLEAN.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The band sits under a registered ceiling at every seed.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" in findings, findings
    assert any("registered ceiling" in s.lower() for s in findings["pre_reg"]), findings


def test_pre_reg_1831_new_nouns_benign_verb_usage_not_flagged():
    """A benign verb-register shape with the #1831 nouns DOWNSTREAM of
    'registered' stays clean: 'hooks registered on the ceiling-check
    codepath' exercises the first-token preposition lookahead ('on')
    with 'ceiling'/'check' tokens inside the would-be window. (A
    noun-BEFORE-verb row like 'the sanity check was registered in WandB'
    is structurally incapable of matching and would guard nothing.)
    Corpus-measured 2026-07-30 (1,816 bodies): 0 benign verb-register
    false positives among the 7 new match starts."""
    body = V4_BODY_CLEAN.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "hooks registered on the ceiling-check codepath fired at every seed.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" not in findings, findings


# ─── pre_reg: the #1902 + #1945 escapes (fix #1958, merged scope with ─────
# #1985)
#
# #1945's body carried 'The pre-set verdict lattice' (Takeaways) and 'the
# pre-declared fallback' (Methodology prose); #1902's body shipped 'the
# planned verdict is Confirmed' and 'The headline persistence verdict
# still confirms' — all four passed the audit clean because the pattern
# was keyed on the single lexeme `registered`. #1958 adds the synonym
# branches: A (modifier-first pre-set/pre-?declared/pre-?specified/
# pre-?committed + the SHARED head-noun tail, which also gains
# `fallbacks?`), B (bare 'planned' + verdicts?/lattices?, adjacency-only),
# C (noun-first 'the verdict was pre-set'), D (verdict-outcome
# announcements 'verdict is/was/still confirm/falsif/inconclusive').
# `pre-set` requires the hyphen: one-word 'preset' stays clean.


def test_pre_reg_pre_set_verdict_lattice_in_takeaways_is_flagged():
    """The verbatim #1945 escape phrasing — 'The pre-set verdict lattice'
    in a v4 `## Takeaways` bullet — trips `pre_reg` via Branch A. The lazy
    intervening-token window stops at the EARLIER noun, so the match text
    is 'pre-set verdict' (the #1593 lazy-stop property)."""
    body = V4_BODY_CLEAN.replace(
        "- Headline finding: the implant installs cleanly across three seeds.",
        "- Headline finding: The pre-set verdict lattice held across seeds.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" in findings, findings
    assert any("pre-set verdict" in s.lower() for s in findings["pre_reg"]), findings


def test_pre_reg_pre_declared_fallback_in_results_prose_is_flagged():
    """The verbatim #1945 escape phrasing — 'the pre-declared fallback' in
    v4 Results prose — trips `pre_reg`: Branch A's modifier plus the
    `fallbacks?` head noun #1958 added to the SHARED tail."""
    body = V4_BODY_CLEAN.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "When the primary read failed, the pre-declared fallback ran first.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" in findings, findings
    assert any("pre-declared fallback" in s.lower() for s in findings["pre_reg"]), findings


def test_pre_reg_planned_verdict_is_confirmed_flagged():
    """The verbatim #1902 escape phrasing — 'the planned verdict is
    Confirmed' — trips `pre_reg` via Branch B (bare 'planned' +
    verdict, adjacency-only)."""
    body = V4_BODY_CLEAN.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "the planned verdict is Confirmed for the headline read.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" in findings, findings
    assert any("planned verdict" in s.lower() for s in findings["pre_reg"]), findings


def test_pre_reg_verdict_still_confirms_flagged():
    """The verbatim #1902 escape phrasing — 'The headline persistence
    verdict still confirms' — trips `pre_reg` via Branch D (verdict-outcome
    announcement; 'confirm' is a prefix match, so 'confirms' hits)."""
    body = V4_BODY_CLEAN.replace(
        "- Headline finding: the implant installs cleanly across three seeds.",
        "- The headline persistence verdict still confirms the effect.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" in findings, findings
    assert any("verdict still confirm" in s.lower() for s in findings["pre_reg"]), findings


def test_pre_reg_noun_first_verdict_was_pre_set_flagged():
    """The noun-first order — 'the verdict was pre-set' — trips `pre_reg`
    via Branch C, which the modifier-first Branch A cannot reach."""
    body = V4_BODY_CLEAN.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "the verdict was pre-set before any data landed.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" in findings, findings
    assert any("verdict was pre-set" in s.lower() for s in findings["pre_reg"]), findings


def test_pre_reg_one_word_preset_not_flagged():
    """One-word 'preset' is a benign config-register word and stays clean:
    Branch A requires the hyphen in 'pre-set' and Branch C requires it in
    'was pre-set' (task constraint)."""
    body = V4_BODY_CLEAN.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "a preset temperature of 0.7 was used; the sampler was preset before the run.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" not in findings, findings


def test_pre_reg_planned_conditions_prose_not_flagged():
    """Benign planned-vs-actual prose stays clean: Branch B is
    adjacency-only and matches ONLY verdicts?/lattices? — 'planned
    conditions' / 'planned-vs-actual coverage' never fire (#1985's own
    narrowing; 'planned' gets neither the window nor the full noun
    list)."""
    body = V4_BODY_CLEAN.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "the planned conditions were all realized; planned-vs-actual coverage matched the design.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" not in findings, findings


def test_pre_reg_pre_specified_threshold_in_v4_hparam_table_is_exempt():
    """A 'pre-specified threshold' phrase inside the Methodology
    **Training:** GFM hyperparameter table on a v4 body does NOT trip
    `pre_reg` — the v4 table-row blanking covers Branch A exactly as it
    covers the `registered` branch (the one surface Lens 7 permits)."""
    body = V4_BODY_CLEAN.replace(
        "| epochs | 1 | prior issue |",
        "| epochs | 1 | prior issue |\n| pass bar | 0.20 | pre-specified threshold (#612) |",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" not in findings, findings


def test_pre_reg_pre_specified_interval_on_why_this_test_line_exempt():
    """The sanctioned Why-this-test CI-definition register in its
    Branch-A synonym form — 'the pre-specified interval defining the
    test' — stays exempt: `_blank_why_this_test_lines` blanks the line
    from the pre_reg scan source for ALL generations (#1783), covering
    the new modifiers exactly as it covers `registered`."""
    body = V4_BODY_CLEAN.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "**Why this test:** the bootstrap CI [+0.1, +0.4] is the "
        "pre-specified interval defining the test.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "pre_reg" not in findings, findings


# ─── v4 ## Methodology sample-data <details> exemption (#1171) ───────────
#
# The v4 spec moved the verbatim sample rows to `## Methodology` →
# `**Sample training/evaluation data + completions:**`, carried in the same
# `<details>` / fenced forms the v3 `## Data` section used.
# `strip_data_example_blocks` exempts `<details>` blocks inside BOTH
# sections (`_DETAILS_EXEMPT_H2_TITLES`); Methodology PROSE outside a
# `<details>` block — and every other section, `## Results` included —
# stays scanned. Tests follow the #969 fixture-mutation convention
# (`.replace()` on `V4_BODY_CLEAN` + a `body != V4_BODY_CLEAN` guard).

# The v4 Methodology sample-data slot with one would-be offender per
# category under test, each on a plain non-table, non-fenced line inside
# the `<details>` block (so no other exemption mechanism can mask the
# result): `C1` → condition_labels, `SUCCESS` → verdict_caps,
# `[0.24, 0.31]` → interval_inline (bare-pair form), `BS_E0` → cell_tags,
# `pre-registered` → pre_reg. Bare `C1` is load-bearing: a compound like
# `sw_eng_C1` would NOT match condition_labels (no `\b` between `_` and
# `C`).
_V4_SAMPLE_SLOT = (
    "**Evaluation:** judge-scored rate on the held-out probes.\n\n"
    "**Sample training/evaluation data + completions:**\n\n"
    "3 of 600 completions, cherry-picked for illustration."
    " Full raw completions: [raw](https://example.com/raw)\n\n"
    "<details>\n<summary>example completions</summary>\n\n"
    "Positive row, condition C1, judge verdict SUCCESS, CI [0.24, 0.31],"
    " cell tag BS_E0, dropped per the pre-registered floor.\n\n"
    "</details>\n"
)


def test_v4_methodology_details_block_offenders_do_not_trip_audit():
    """Spec-mandated verbatim sample rows inside a v4 `## Methodology`
    `<details>` block do NOT trip ANY audit category — the #1171 fix.
    Offenders cover five categories (condition_labels, verdict_caps,
    interval_inline, cell_tags, pre_reg); pre-fix this body flagged them
    all (the red half of the red/green demonstration)."""
    body = V4_BODY_CLEAN.replace(
        "**Evaluation:** judge-scored rate on the held-out probes.",
        _V4_SAMPLE_SLOT,
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert findings == {}, findings


def test_v4_methodology_prose_offender_still_flagged():
    """A condition code in v4 `## Methodology` PROSE (outside any
    `<details>`/fence) still flags, even with an offender-bearing details
    block present in the same section — the exemption is
    `<details>`-scoped, not section-wide."""
    body = V4_BODY_CLEAN.replace(
        "**Evaluation:** judge-scored rate on the held-out probes.",
        _V4_SAMPLE_SLOT,
    ).replace(
        "**Design:** three seeds; baseline vs treatment; the single variable is the data mix.",
        "**Design:** three seeds under the C1 condition; baseline vs treatment.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "condition_labels" in findings, findings
    assert any("C1" in s for s in findings["condition_labels"]), findings


def test_v4_results_prose_offender_still_flagged():
    """A condition code in v4 `## Results` prose still flags — mirrors
    v3's `test_condition_codes_outside_data_still_flagged` (the sibling
    candidate's regression test)."""
    body = V4_BODY_CLEAN.replace(
        "The lift holds at every seed",
        "The C1 condition shows the lift at every seed",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "condition_labels" in findings, findings


def test_v4_results_details_block_offender_still_flagged():
    """A `<details>`-wrapped offender in v4 `## Results` STILL flags —
    pins that `results` is deliberately NOT in
    `_DETAILS_EXEMPT_H2_TITLES` (Results excerpts are authored-adjacent
    and stay scanned)."""
    body = V4_BODY_CLEAN.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The lift holds at every seed in the held-out evaluation.\n\n"
        "<details>\n<summary>raw excerpt</summary>\n\n"
        "Condition C1 excerpt row.\n\n</details>",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "condition_labels" in findings, findings


def test_v4_methodology_details_never_closed_stops_at_next_h2():
    """A `<details>` opened but never closed inside `## Methodology`
    stops being dropped at the next H2 — the same boundary-degradation
    property `test_strip_data_example_blocks_ends_at_next_h2` pins for
    `## Data` (a mis-detected boundary degrades to scanning, never a
    silently widened exemption)."""
    text = (
        "## Methodology\n\n<details>\nrow inside methodology\n\n## Results\n\nC1 in results prose\n"
    )
    stripped = audit.strip_data_example_blocks(text)
    assert "row inside methodology" not in stripped  # dropped (inside details)
    assert "C1 in results prose" in stripped  # NOT dropped (past the section)


def test_strip_data_example_blocks_drops_methodology_and_data_keeps_others():
    """Unit-level exempt-set pin: `<details>` blocks under `## Data` AND
    `## Methodology` are dropped; a `<details>` under any other H2
    (`## Findings`) survives — extends
    `test_strip_data_example_blocks_only_drops_inside_data` without
    modifying it."""
    text = (
        "## Findings\n\n<details>\nC1 leaks here\n</details>\n\n"
        "## Data\n\n<details>\nC1 verbatim row\n</details>\n\n"
        "## Methodology\n\n<details>\nC1 sample row\n</details>\n\n"
        "## Results\n\nok\n"
    )
    stripped = audit.strip_data_example_blocks(text)
    assert "C1 leaks here" in stripped  # Findings details survives
    assert "C1 verbatim row" not in stripped  # Data details dropped
    assert "C1 sample row" not in stripped  # Methodology details dropped


# ─── verdict_caps: SUCCESS|FAILURE gate verdicts (incident #763; #970) ────
#
# #892 fixed the `pre_reg` half of the #763 incident ("As registered,
# SUCCESS was not met"), but the caps gate-verdict itself escaped
# `verdict_caps`: the live #763 body's "Under the pre-set decision rule,
# SUCCESS was not met" carries no 'as registered' bigram and SUCCESS /
# FAILURE were absent from the four-word alternation. #970 adds both bare
# words, case-sensitive, no context guard (bigram anchoring is the exact
# fragility that let #763 escape `pre_reg`).


def test_verdict_caps_success_not_met_on_v4_body_is_flagged():
    """The live #763 line-263 clause ("Under the pre-set decision rule,
    SUCCESS was not met ...") in a v4 body's `## Results` prose trips
    `verdict_caps` — the exact incident path. Historically `pre_reg`
    stayed silent here (no 'as registered' bigram — the escape #970's
    comment documents); since #1958's Branch A, 'pre-set decision rule'
    ALSO trips `pre_reg`, closing that half of the escape."""
    v4 = (
        "# Title (LOW confidence)\n<!-- clean-result-v4 -->\n\n"
        "## Takeaways\n\n- clean prose.\n\n## Goal\n\nclean.\n\n"
        "## Methodology\n\nclean.\n\n## Results\n\n"
        "Under the pre-set decision rule, SUCCESS was not met — the estimator "
        "in force read 0.00, the falsification branch (floor 0.15).\n"
    )
    findings = audit.audit_body(v4)
    assert "verdict_caps" in findings, findings
    assert "SUCCESS" in findings["verdict_caps"], findings
    assert "pre_reg" in findings, findings
    assert any("pre-set decision rule" in s.lower() for s in findings["pre_reg"]), findings


def test_verdict_caps_failure_verdict_in_takeaways_is_flagged():
    """The symmetric verdict twin: a caps FAILURE declaration in scanned
    Takeaways prose trips `verdict_caps`."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "- Headline finding: the implant installs cleanly across three seeds.",
        "- FAILURE was declared on the install arm at every seed.",
    )
    assert body != V3_BODY_WITH_DATA_CODES
    findings = audit.audit_body(body)
    assert "verdict_caps" in findings, findings
    assert "FAILURE" in findings["verdict_caps"], findings


def test_verdict_caps_lowercase_titlecase_forms_not_flagged():
    """Everyday forms must NOT trip: lowercase `success`/`failure` and
    titlecase `Success` pin the case-sensitive scan (`flags=0`); the caps
    derivatives `SUCCESSFUL`/`UNSUCCESSFUL` cannot be excluded by case
    alone, so they genuinely pin the `\\b` word boundary (§12 assumption 8:
    `\\bSUCCESS\\b` fails on a trailing word char)."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The success criterion was not met; Success was partial; the run was "
        "UNSUCCESSFUL and SUCCESSFUL retries followed; the failure mode was benign.",
    )
    assert body != V3_BODY_WITH_DATA_CODES
    findings = audit.audit_body(body)
    assert "verdict_caps" not in findings, findings


@pytest.mark.parametrize("word", ["REJECTED", "INDETERMINATE", "PASSED", "EXCEEDING"])
def test_verdict_caps_existing_words_still_flagged(word):
    """Regression pin for ALL FOUR pre-existing alternation words — the
    first-ever `verdict_caps` tests land with #970, and the
    awaiting_promotion corpus carries zero existing-word hits, so a regex
    retype dropping a sibling word while adding SUCCESS|FAILURE would
    otherwise pass every other test AND the corpus diff."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "- Secondary finding: no measurable regression on the held-out probes.",
        f"- The gate read {word} on the secondary arm.",
    )
    assert body != V3_BODY_WITH_DATA_CODES
    findings = audit.audit_body(body)
    assert "verdict_caps" in findings, findings
    assert word in findings["verdict_caps"], findings


def test_verdict_caps_install_failure_emphasis_is_flagged_by_design():
    """DELIBERATE decision pin (#970 plan §4): caps-emphasis usage like the
    #543 body's "defines an install FAILURE as hitting the 16-epoch cap"
    IS flagged. Caps emphasis is off-register under the clean-result voice
    discipline (Lens 6); the fix is a register-improving lowercase.
    Completed bodies like #543 itself are grandfathered (never re-audited),
    so this is a prospective-only exposure, accepted by design — do NOT
    "fix" this as a false positive without revisiting that decision."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The follow-up plan defines an install FAILURE as hitting the 16-epoch cap.",
    )
    assert body != V3_BODY_WITH_DATA_CODES
    findings = audit.audit_body(body)
    assert "verdict_caps" in findings, findings


def test_verdict_caps_code_spans_not_flagged():
    """Caps SUCCESS/FAILURE inside an inline-backtick span AND a fenced code
    block must NOT trip — `verdict_caps` scans `cleaned` (strip_code applied),
    which is the mitigation bounding the prospective false-positive surface
    (verbatim sample/log text in code spans stays exempt)."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The gate metric is `SUCCESS` in the results JSON.\n\n```\nverdict: FAILURE\n```",
    )
    assert body != V3_BODY_WITH_DATA_CODES
    findings = audit.audit_body(body)
    assert "verdict_caps" not in findings, findings


# ─── bit/byte-identical AI-slop family (Lens 6; incident #642) ───────────
#
# The `byte_identical` rule (task #454) banned `byte identical` /
# `byte-identical`. Task #642's body carried BOTH `bit-identical` (in
# `## What I ran` prose) AND `byte-identical` (in `## Data → ### Trained on`
# prose); the byte-only regex flagged only the latter and the clean-result-
# critic Lens 6 caught the `bit-identical` slip manually. The rule was
# renamed `bit_byte_identical` and broadened to `\b(?:byte|bit)[\s-]identical\b`
# so both same-family voice violations are caught under one mechanical rule.
# Task #1423 extended the family to the `equal` synonyms (byte-equal /
# byte equal / bit-equal / bit equal) after issue #1005's body carried
# "inherited byte-equal (sha-asserted)" past the -identical-only regex.
# Task #1447 widened the family to its remaining synonym tail in one
# batched pass: the `-exact` adjective, the `bitwise`/`bytewise` unit
# forms (deliberately superseding #1423's `bitwise` != `bit` boundary
# pin), and the reduplicated `bit-for-bit`/`byte-for-byte`, plus a
# `(?<!-)` lookbehind blocking the numeric-bit-width class
# (`8-bit exact-width`). The plan-time corpus scan found 14 grandfathered
# bodies already carrying the tail.


def test_byte_identical_still_flagged_under_new_key():
    """The original `byte-identical` / `byte identical` form still trips the
    renamed `bit_byte_identical` rule (no regression vs the #454 byte rule)."""
    hyphen = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The two checkpoints were byte-identical across the rerun.",
    )
    findings = audit.audit_body(hyphen)
    assert "bit_byte_identical" in findings, findings
    space = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The two checkpoints were byte identical across the rerun.",
    )
    assert "bit_byte_identical" in audit.audit_body(space), "space form regressed"


def test_bit_identical_is_flagged():
    """The same-family `bit-identical` / `bit identical` form now trips the
    audit — the byte-only regex previously slipped it past (incident #642)."""
    hyphen = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The loss surface was held bit-identical across all three arms.",
    )
    findings = audit.audit_body(hyphen)
    assert "bit_byte_identical" in findings, findings
    assert any("bit-identical" in s for s in findings["bit_byte_identical"]), findings
    space = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The loss surface was held bit identical across all three arms.",
    )
    assert "bit_byte_identical" in audit.audit_body(space), "space form not flagged"


def test_bit_byte_identical_no_false_positive_on_unrelated_words():
    """The regex requires the literal `identical` neighbour — words like
    `bite`, `arbiter`, `bytecode` (no following `identical`) must NOT trip."""
    clean = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The arbiter compiled the bytecode and a bite-sized summary.",
    )
    findings = audit.audit_body(clean)
    assert "bit_byte_identical" not in findings, findings


def test_byte_equal_is_flagged():
    """The `-equal` synonym family trips the rule (task #1423): `byte-equal`
    and `byte equal` are the same voice violation as `byte-identical`."""
    hyphen = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The two stores were byte-equal after the rerun.",
    )
    findings = audit.audit_body(hyphen)
    assert "bit_byte_identical" in findings, findings
    assert any("byte-equal" in s for s in findings["bit_byte_identical"]), findings
    space = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The two stores were byte equal after the rerun.",
    )
    space_findings = audit.audit_body(space)
    assert "bit_byte_identical" in space_findings, "space form not flagged"
    assert any("byte equal" in s for s in space_findings["bit_byte_identical"]), space_findings


def test_bit_equal_is_flagged():
    """The `bit-equal` / `bit equal` synonyms trip the rule too (task #1423)."""
    hyphen = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The loss surface was held bit-equal across all three arms.",
    )
    findings = audit.audit_body(hyphen)
    assert "bit_byte_identical" in findings, findings
    assert any("bit-equal" in s for s in findings["bit_byte_identical"]), findings
    space = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The loss surface was held bit equal across all three arms.",
    )
    space_findings = audit.audit_body(space)
    assert "bit_byte_identical" in space_findings, "space form not flagged"
    assert any("bit equal" in s for s in space_findings["bit_byte_identical"]), space_findings


def test_byte_equal_incident_1005_phrasing_is_flagged():
    """Regression anchor: the verbatim #1005 incident phrasing — "inherited
    byte-equal (sha-asserted)" — trips the rule (the -identical-only regex
    missed it and an LM critic round had to catch it manually)."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The adapter was inherited byte-equal (sha-asserted) from the parent.",
    )
    findings = audit.audit_body(body)
    assert "bit_byte_identical" in findings, findings
    assert any("byte-equal" in s for s in findings["bit_byte_identical"]), findings


def test_bit_byte_equal_no_false_positive_on_boundary_words():
    """Boundary negatives for the `-equal` arm (task #1423): `byte equality`
    (suffix blocks the trailing \\b), `bytes equal` (plural blocks `[\\s-]`),
    and capitalized `Byte-equal` (the category scans case-sensitively,
    pre-existing semantics) must NOT trip. The #1423 version of this test
    also pinned `bitwise equal` as a NON-match (`bitwise` != `bit`); task
    #1447 deliberately superseded that boundary — `bitwise equal` is now a
    positive (see test_bitwise_forms_are_flagged) and is removed from this
    negative body."""
    clean = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "I assert byte equality; the bytes equal to the header match. "
        "Byte-equal casing stays untouched.",
    )
    findings = audit.audit_body(clean)
    assert "bit_byte_identical" not in findings, findings


def test_byte_bit_exact_is_flagged():
    """The `-exact` synonym family trips the rule (task #1447): `byte-exact` /
    `byte exact` / `bit-exact` / `bit exact` are the same voice violation as
    `byte-identical`. The plan-time corpus scan found 7 grandfathered bodies
    carrying `-exact` forms (#222, #545, #722, #1223 byte; #448, #952, #1362
    bit)."""
    for phrase in ("byte-exact", "byte exact", "bit-exact", "bit exact"):
        body = V3_BODY_WITH_DATA_CODES.replace(
            "The lift holds at every seed in the held-out evaluation.",
            f"The two stores were {phrase} after the rerun.",
        )
        findings = audit.audit_body(body)
        assert "bit_byte_identical" in findings, (phrase, findings)


def test_bitwise_forms_are_flagged():
    """The `bitwise`/`bytewise` unit forms trip the rule (task #1447). This
    DELIBERATELY INVERTS #1423's boundary semantics — its
    test_bit_byte_equal_no_false_positive_on_boundary_words pinned
    `bitwise equal` as a non-match (`bitwise` != `bit`); #1447 folds the
    `(?:wise)?` unit extension into the family."""
    for phrase in (
        "bitwise identical",
        "bitwise-identical",
        "bitwise equal",
        "bitwise exact",
        "bytewise identical",
    ):
        body = V3_BODY_WITH_DATA_CODES.replace(
            "The lift holds at every seed in the held-out evaluation.",
            f"The outputs were {phrase} across the rerun.",
        )
        findings = audit.audit_body(body)
        assert "bit_byte_identical" in findings, (phrase, findings)


def test_x_for_x_reduplication_is_flagged():
    """The reduplicated `bit-for-bit` / `byte-for-byte` forms trip the rule
    (task #1447) — the shape differs from `<unit><sep><adjective>` so it gets
    its own alternation branch. The plan-time corpus scan found 7
    grandfathered bodies carrying these forms (#276, #525, #531, #588
    byte-for-byte; #671, #673, #810 bit-for-bit)."""
    for phrase in ("bit-for-bit", "bit for bit", "byte-for-byte", "byte for byte"):
        body = V3_BODY_WITH_DATA_CODES.replace(
            "The lift holds at every seed in the held-out evaluation.",
            f"The copy is {phrase} faithful to the source.",
        )
        findings = audit.audit_body(body)
        assert "bit_byte_identical" in findings, (phrase, findings)


def test_widened_family_no_false_positive_on_technical_prose():
    """Negative battery for the widened family (task #1447): legitimate
    technical prose must NOT trip. `bitwise AND` / `bitwise operations`
    (adjective set required after the separator); `8-bit exact-width` /
    `64-bit equal-width` / `n-bit identical-width` (the `(?<!-)` lookbehind
    blocks hyphen-preceded numeric bit-width units); `byte offset` /
    `byte order` (no family adjective); `a bit more` (`more` not in the
    adjective set); `one bit for parity` (the `for` branch requires the full
    reduplication)."""
    for phrase in (
        "the bitwise AND of the masks",
        "bitwise operations on the header",
        "an 8-bit exact-width field",
        "a 64-bit equal-width layout",
        "n-bit identical-width lanes",
        "the byte offset of the record",
        "network byte order applies",
        "a bit more variance than expected",
        "we reserve one bit for parity",
    ):
        body = V3_BODY_WITH_DATA_CODES.replace(
            "The lift holds at every seed in the held-out evaluation.",
            f"Note that {phrase} here.",
        )
        findings = audit.audit_body(body)
        assert "bit_byte_identical" not in findings, (phrase, findings)


def test_widened_family_cross_line_separator_matches():
    """Behavior pin (task #1447, implementer-discretion): `[\\s-]` matches a
    newline, so a line-wrapped `byte\\nexact` still trips the rule — the
    pre-existing cross-line semantics of the #454/#642/#1423 separator,
    extended to the `-exact` adjective. Accepted semantics: a hard-wrapped
    banned phrase does not escape the audit."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The two stores were byte\nexact after the rerun.",
    )
    findings = audit.audit_body(body)
    assert "bit_byte_identical" in findings, findings


def test_reduplication_branch_hyphen_exposure_pinned():
    """Behavior pin (task #1447, implementer-discretion): the reduplication
    branch's trailing `\\b` is satisfied at the hyphen in `bit-level`, so
    `a bit for bit-level ops` DOES trip (the `(?<!-)` lookbehind guards only
    the match START and is irrelevant here). Accepted exposure — the
    plan-time corpus scan found zero such hits across all 1,408 bodies, and
    guarding it would also miss genuine `bit-for-bit`-adjacent compounds."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "We shift a bit for bit-level ops in the packer.",
    )
    findings = audit.audit_body(body)
    assert "bit_byte_identical" in findings, findings


def test_bit_deterministic_determinism_vocabulary_not_flagged():
    """Allowlist pin (task #1614): 'bit-deterministic' / 'bit-determinism'
    are deliberately NOT in the bit_byte_identical family — they name
    exact re-forward reproducibility (a determinism property of a
    computation; #1415's jitter-floor evidence), not the banned
    artifact-equality claim-shape. The first two phrases are the verbatim
    #1415 body line-90 usages. A future widening that adds
    'deterministic|determinism' to the alternation must supersede this
    pin AND the decision record (audit-script category comment;
    clean-result-critic-lens-reference.md Lens 6), the way #1447
    superseded #1423's 'bitwise' boundary pin. The trailing positive
    control proves the scanner ran on the same body shape."""
    for phrase in (
        "the re-forwards were bit-deterministic there",
        "the bit-determinism rules a genuinely larger true band out",
        "bit deterministic replay of the capture path",
        "bitwise-deterministic kernels were enabled",
    ):
        body = V3_BODY_WITH_DATA_CODES.replace(
            "The lift holds at every seed in the held-out evaluation.",
            f"Note that {phrase} here.",
        )
        findings = audit.audit_body(body)
        assert "bit_byte_identical" not in findings, (phrase, findings)
    control = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The outputs were bit-identical across the rerun.",
    )
    assert "bit_byte_identical" in audit.audit_body(control), "positive control regressed"


def test_bit_deterministic_allowlist_decision_record_present():
    """Durability pin (task #1614): the allowlist decision record survives
    in BOTH places future re-litigators read — the audit script's
    bit_byte_identical category comment, and the Lens 6 prose in
    .claude/rules/clean-result-critic-lens-reference.md (the surface the
    clean-result-critic reads and the Codex twin's prompt is composed
    from). Dropping either re-opens the ambiguity #1415's critique v7
    raised. Substring-loose on purpose: rewording survives; deleting the
    record does not."""
    src = SCRIPT.read_text(encoding="utf-8")
    assert "bit-deterministic" in src and "#1614" in src, (
        "audit-script allowlist decision record missing"
    )
    lens_path = (
        SCRIPT.resolve().parents[1]
        / ".claude"
        / "rules"
        / ("clean-result-critic-lens-reference.md")
    )
    lens_text = lens_path.read_text(encoding="utf-8")
    assert "bit-deterministic" in lens_text and "#1614" in lens_text, (
        "lens-reference Lens 6 allowlist clause missing"
    )


# ─── Inline verbatim originating-prompt exemption (incident #651) ────────
#
# The `## Reproducibility` `**Context:**` `Originating prompt` sub-bullet
# carries the verbatim user prompt (verify_task_body.py check 17 / SPEC.md
# § `**Context:**` row — NEVER paraphrased). When the prompt is carried
# INLINE on the `- Originating prompt: "..."` sub-bullet (the form #651 /
# #640 / #610 use, as opposed to a following `>` blockquote) and the prompt
# text itself contains an anti-pattern token (#651: "post-hoc"), the audit
# must NOT flag it — verbatim preservation and the prose scan are otherwise
# mutually unsatisfiable on that quote. The exemption is scoped to the
# prompt only: an anti-pattern in genuine Findings/Takeaways prose still
# fires.


def _body_with_inline_origin_prompt(prompt_line: str) -> str:
    """V3 body whose Context block carries `prompt_line` as the originating
    prompt sub-bullet, with otherwise-clean prose."""
    return (
        "---\ntitle: foo\nkind: experiment\ngoal: g\n---\n"
        "# A clean claim (MODERATE confidence)\n\n"
        "<!-- clean-result-v3 -->\n\n"
        "## Takeaways\n\n- The implant installs cleanly across three seeds.\n\n"
        "## What I ran\n\n- **Why:** to test generalisation.\n\n"
        "## Findings\n\n### A clean lift\n\nThe lift holds at every seed.\n\n"
        "## Reproducibility\n\n"
        "**Compute:** 1x H100.\n\n"
        "**Context:**\n\n"
        "- Created 2026-06-16.\n"
        "- Follow-up to fresh direction (no parent).\n"
        f"{prompt_line}\n"
    )


def test_inline_origin_prompt_post_hoc_exempt_plain_form():
    """`- Originating prompt: "...post-hoc..."` (plain, #651's form) must NOT
    trip `post_hoc_phrasing` — the prompt is verbatim-mandated by check 17."""
    body = _body_with_inline_origin_prompt(
        '- Originating prompt: "test the shared-direction idea; post-hoc on the #537 eval"'
    )
    findings = audit.audit_body(body)
    assert "post_hoc_phrasing" not in findings, findings


def test_inline_origin_prompt_post_hoc_exempt_bold_form():
    """`- **Originating prompt(s), verbatim:** "...post-hoc..."` (bold inline
    form, #640 / #610) must NOT trip `post_hoc_phrasing` either."""
    body = _body_with_inline_origin_prompt(
        '- **Originating prompt(s), verbatim:** "run the post-hoc check please"'
    )
    findings = audit.audit_body(body)
    assert "post_hoc_phrasing" not in findings, findings


def test_inline_origin_prompt_multiline_continuation_exempt():
    """A wrapped multi-line inline prompt is exempt across its continuation
    lines; the next sibling Context bullet ends the exemption run."""
    body = (
        "---\ntitle: foo\nkind: experiment\ngoal: g\n---\n"
        "# A clean claim (MODERATE confidence)\n\n"
        "<!-- clean-result-v3 -->\n\n"
        "## Findings\n\n### A clean lift\n\nThe lift holds.\n\n"
        "## Reproducibility\n\n"
        "**Context:**\n\n"
        "- **Originating prompt(s), verbatim:**\n"
        '  "first line; post-hoc reference here\n'
        '  continues on the second line"\n'
        "- **Compute:** 4x H100.\n"
    )
    findings = audit.audit_body(body)
    assert "post_hoc_phrasing" not in findings, findings


def test_post_hoc_in_findings_prose_still_flagged():
    """The exemption is scoped to the verbatim prompt: a genuine `post-hoc`
    in `## Findings` prose must STILL fire (the audit's purpose preserved)."""
    body = _body_with_inline_origin_prompt('- Originating prompt: "test the thing"').replace(
        "The lift holds at every seed.", "We ran a post-hoc analysis of the residuals."
    )
    findings = audit.audit_body(body)
    assert "post_hoc_phrasing" in findings, findings


def test_inline_origin_prompt_exemption_does_not_leak_to_sibling_bullets():
    """A `post-hoc` in the Created / Follow-up bullets (NOT the prompt) is
    still scanned — only the Originating-prompt sub-bullet run is exempt."""
    body = _body_with_inline_origin_prompt('- Originating prompt: "test the thing"').replace(
        "- Created 2026-06-16.",
        "- Created 2026-06-16; this is a post-hoc note in the wrong bullet.",
    )
    findings = audit.audit_body(body)
    assert "post_hoc_phrasing" in findings, findings


# ─── paper-stub support (`paper: true`) ────────────────────────────────────

PAPER_STUB_BODY = """\
---
title: A claim (MODERATE confidence)
kind: experiment
paper: true
---
# A claim (MODERATE confidence)

An abstract that, if it had said pre-registered, would normally trip the audit.

Paper: docs/papers/issue_657/issue_657.pdf
"""


def test_is_paper_stub_helper():
    assert audit._is_paper_stub(PAPER_STUB_BODY)
    assert audit._is_paper_stub(PAPER_STUB_BODY.replace("paper: true", "paper: 'true'"))
    assert not audit._is_paper_stub("# T\n\nno frontmatter\n")


def test_paper_stub_skips_audit(capsys):
    """A paper-stub body PASSes the live single-body audit even with a phrase
    that would normally trip the prose anti-pattern scan — the markdown
    body-discipline checks do not apply to a paper-task."""
    leaky_stub = PAPER_STUB_BODY.replace(
        "would normally trip the audit", "is pre-registered and would normally trip the audit"
    )
    rc = audit._audit_single_body(leaky_stub)
    assert rc == 0
    out = capsys.readouterr().out
    assert "paper-stub" in out
    assert "verify_paper.py" in out


# ─── H1-vs-frontmatter-title sync (WARN-level corpus surface, #1196) ───────
#
# `h1_title_sync_warn` DELEGATES the whole comparison to
# `verify_task_body.check_h1_matches_frontmatter_title` (the #1110 gate
# check) and flattens its severity to WARN; `_run_title_sync_sweep` walks a
# tasks/-shaped tree and prints one row per flagged sentinelled body,
# always returning 0. Fixture prose is deliberately anti-pattern-free so
# the WARN surface is isolated from the findings scan.

_SYNC_TITLE = "A tidy claim about the finding (MODERATE confidence)"


def _sync_body(
    *,
    fm_title: str | None = _SYNC_TITLE,
    h1: str | None = _SYNC_TITLE,
    sentinel: str = "<!-- clean-result-v4 -->",
    frontmatter: bool = True,
    prose: str = "- A tidy bullet about the finding.",
) -> str:
    """Build a minimal (optionally sentinelled) body for the sync tests."""
    parts: list[str] = []
    if frontmatter:
        fm_lines = ["---"]
        if fm_title is not None:
            fm_lines.append(f"title: {fm_title}")
        fm_lines.append("kind: experiment")
        fm_lines.append("---")
        parts.append("\n".join(fm_lines))
    if h1 is not None:
        parts.append(f"# {h1}")
    if sentinel:
        parts.append(sentinel)
    parts.append(f"## Takeaways\n\n{prose}")
    return "\n\n".join(parts) + "\n"


def _write_task_body(tasks_root, status: str, task_dir: str, text: str) -> None:
    d = tasks_root / status / task_dir
    d.mkdir(parents=True)
    (d / "body.md").write_text(text, encoding="utf-8")


def test_title_sync_sweep_warn_on_divergent_body(tmp_path, capsys):
    """Durability pin: a divergent sentinelled v4 body in the tree yields
    exactly one `- #<N> (<status>):` WARN row and the sweep returns 0."""
    _write_task_body(
        tmp_path,
        "awaiting_promotion",
        "777",
        _sync_body(h1="A retitled claim about the finding (MODERATE confidence)"),
    )
    rc = audit._run_title_sync_sweep(tasks_root=tmp_path)
    assert rc == 0
    out = capsys.readouterr().out
    assert "- #777 (awaiting_promotion):" in out
    assert out.count("- #") == 1
    assert "WARN: 1 " in out


def test_title_sync_sweep_in_sync_body_no_warn(tmp_path, capsys):
    """A matching H1/title pair (confidence tag on both sides) yields the
    PASS line and no rows."""
    _write_task_body(tmp_path, "completed", "42", _sync_body())
    rc = audit._run_title_sync_sweep(tasks_root=tmp_path)
    assert rc == 0
    out = capsys.readouterr().out
    assert "PASS: H1 == frontmatter title" in out
    assert "- #" not in out


def test_title_sync_sweep_skips_non_sentinelled_body(tmp_path, capsys):
    """A divergent but sentinel-less body (pre-promotion shape) is out of
    scope — the gate check's sentinel gate governs, by delegation."""
    _write_task_body(
        tmp_path,
        "proposed",
        "9",
        _sync_body(h1="A completely different working headline", sentinel=""),
    )
    rc = audit._run_title_sync_sweep(tasks_root=tmp_path)
    assert rc == 0
    out = capsys.readouterr().out
    assert "- #" not in out


def test_title_sync_sweep_whitespace_only_difference_not_flagged(tmp_path, capsys):
    """Whitespace-only drift (double spaces) is eaten by the gate check's
    collapse normalization — parity by delegation."""
    _write_task_body(
        tmp_path,
        "completed",
        "11",
        _sync_body(fm_title="A tidy  claim about the   finding (MODERATE confidence)"),
    )
    rc = audit._run_title_sync_sweep(tasks_root=tmp_path)
    assert rc == 0
    assert "- #" not in capsys.readouterr().out


def test_title_sync_sweep_case_difference_is_flagged(tmp_path, capsys):
    """Case-only drift IS real drift (no case folding — the #763 rationale
    the gate check documents)."""
    _write_task_body(
        tmp_path,
        "completed",
        "12",
        _sync_body(fm_title="A tidy claim about the finding (moderate confidence)"),
    )
    rc = audit._run_title_sync_sweep(tasks_root=tmp_path)
    assert rc == 0
    assert "- #12 (completed):" in capsys.readouterr().out


def test_title_sync_sweep_skips_non_numeric_task_dir(tmp_path, capsys):
    """A `*/*/body.md` path whose parent dir is not a task number (e.g.
    tasks/misc/_orphaned_markers/body.md) is skipped, not scanned."""
    _write_task_body(
        tmp_path,
        "awaiting_promotion",
        "5",
        _sync_body(h1="A retitled claim about the finding (MODERATE confidence)"),
    )
    _write_task_body(
        tmp_path,
        "misc",
        "_orphaned_markers",
        _sync_body(h1="A retitled claim about the finding (MODERATE confidence)"),
    )
    rc = audit._run_title_sync_sweep(tasks_root=tmp_path)
    assert rc == 0
    out = capsys.readouterr().out
    assert "Scanned 1 task bodies" in out
    assert out.count("- #") == 1
    assert "- #5 (awaiting_promotion):" in out


def test_h1_title_sync_warn_missing_fm_title_flagged():
    """A sentinelled body whose frontmatter lacks `title` hits the gate
    check's anomaly branch (broken promotion)."""
    detail = audit.h1_title_sync_warn(_sync_body(fm_title=None))
    assert detail is not None
    assert "no frontmatter `title`" in detail


def test_h1_title_sync_warn_missing_h1_flagged():
    """A sentinelled body with a frontmatter title but no `# ` H1 line hits
    the gate check's missing-H1 anomaly branch."""
    detail = audit.h1_title_sync_warn(_sync_body(h1=None))
    assert detail is not None
    assert "no H1 found" in detail


def test_h1_title_sync_warn_v3_grandfathered_flagged():
    """A divergent v3 body is flagged via the predicate's `is_warn` leg
    (gate-time grandfathering to WARN; identical WARN severity here)."""
    detail = audit.h1_title_sync_warn(
        _sync_body(
            h1="A retitled claim about the finding (MODERATE confidence)",
            sentinel="<!-- clean-result-v3 -->",
        )
    )
    assert detail is not None
    assert "grandfathered" in detail


def test_single_body_audit_prints_title_sync_warn_rc_unchanged(capsys):
    """The `--task`/file single-body path appends `WARN h1_title_sync:`
    after the PASS/FAIL headline; the exit code follows findings only."""
    divergent = _sync_body(h1="A retitled claim about the finding (MODERATE confidence)")
    rc = audit._audit_single_body(divergent)
    out = capsys.readouterr().out
    assert rc == 0
    assert out.splitlines()[0].startswith("PASS:")
    assert "WARN h1_title_sync:" in out

    divergent_with_findings = _sync_body(
        h1="A retitled claim about the finding (MODERATE confidence)",
        prose="- The C1 condition leaked badly here.",
    )
    rc = audit._audit_single_body(divergent_with_findings)
    out = capsys.readouterr().out
    assert rc == 1  # findings drive rc; the WARN never flips it
    assert out.splitlines()[0].startswith("FAIL:")
    assert "WARN h1_title_sync:" in out


def test_single_body_audit_no_fm_no_warn_line(capsys):
    """Frontmatter-less inputs (analyzer /tmp drafts) hit the gate check's
    empty-fm skip — output carries no WARN line (byte-compat with the
    pre-#1196 behavior on every existing draft fixture)."""
    rc = audit._audit_single_body(_sync_body(frontmatter=False))
    assert rc == 0
    out = capsys.readouterr().out
    assert "WARN h1_title_sync" not in out
    assert out.splitlines()[0].startswith("PASS:")


# ─── opaque_snake_slugs: backticked 3+-segment snake_case slugs (#1372) ──
#
# Incident #1315: `` `neg_reph_curious` `` sat in Methodology Data-extraction
# AND Results prose; `--task 1315` printed PASS and only the LM critic caught
# it. The category scans v4 reader-facing prose ONLY (`## Takeaways` /
# `## Goal` / `## Results`), over the inline-backtick-keeping chain (like
# `interval_inline`), with the `**Repro:**` footer onward, blockquote
# captions, GFM table rows, fenced code, `test_*` names, and the exact-token
# field-name allowlist exempt. Tests mutate V4_BODY_CLEAN via targeted
# `.replace()` with a body != fixture guard, matching the pre_reg section.

_RESULTS_PROSE_LINE = "The lift holds at every seed in the held-out evaluation."


def test_snake_slug_in_v4_results_prose_is_flagged():
    """The #1315 Results-prose shape: a backticked 3-segment slug in
    `## Results` prose on a v4 body trips `opaque_snake_slugs`."""
    body = V4_BODY_CLEAN.replace(
        _RESULTS_PROSE_LINE,
        "The lift holds at every seed; one panel context (`neg_reph_curious`) drives it.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "opaque_snake_slugs" in findings, findings
    assert "`neg_reph_curious`" in findings["opaque_snake_slugs"], findings


def test_snake_slug_in_v4_takeaways_prose_is_flagged():
    """A backticked slug in a `## Takeaways` bullet flags — the mentor-facing
    narrative surface the no-opaque-codes rule most directly protects."""
    body = V4_BODY_CLEAN.replace(
        "- Headline finding: the implant installs cleanly across three seeds.",
        "- Headline finding: `tf_rev_default` installs cleanly across three seeds.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "opaque_snake_slugs" in findings, findings


def test_snake_slug_in_v4_goal_prose_is_flagged():
    """A backticked slug in `## Goal` prose flags — pins that `goal` is in
    `_SNAKE_SLUG_PROSE_H2S` (reviewer concern on #1372's plan)."""
    body = V4_BODY_CLEAN.replace(
        "**Broader narrative:** which context factors predict fine-tuning leakage.",
        "**Broader narrative:** whether `imp_icl_ft_neg` leakage tracks geometry.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "opaque_snake_slugs" in findings, findings


def test_snake_slug_in_v4_methodology_prose_is_exempt():
    """A slug appearing ONLY in `## Methodology` prose does NOT flag — the
    field-name-dense section is deliberately out of scope (#1372 §4.6); the
    residual stays LM-critic territory."""
    body = V4_BODY_CLEAN.replace(
        "**Design:** three seeds; baseline vs treatment; the single variable is the data mix.",
        "**Design:** three seeds; one panel context (`neg_reph_curious`) is the treatment.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "opaque_snake_slugs" not in findings, findings


def test_snake_slug_in_repro_footer_is_exempt():
    """Slugs in the `**Repro:**` footer (the #1315 remediation destination —
    the sanctioned slug surface) do NOT flag: the walker blanks the footer
    label line onward."""
    body = V4_BODY_CLEAN.replace(
        "**Repro:** 1x A100, 47 min; code at commit deadbeef.",
        "**Repro:** 1x A100, 47 min; adapters `tf_default_contra_d1` and\n"
        "`neg_reph_curious`; code at commit deadbeef.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "opaque_snake_slugs" not in findings, findings


def test_snake_slug_in_figure_caption_blockquote_is_exempt():
    """A slug in a figure-caption blockquote inside `## Results` does NOT
    flag — a caption naming the plotted cell is provenance, not narrative
    (same carve-out family as `interval_inline`'s)."""
    body = V4_BODY_CLEAN.replace(
        _RESULTS_PROSE_LINE,
        _RESULTS_PROSE_LINE + "\n\n> **Figure.** *Per-seed lift for the "
        "`tf_default_contra_d1_seed42` cell.*",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "opaque_snake_slugs" not in findings, findings


def test_snake_slug_in_condition_table_and_fenced_code_is_exempt():
    """Slugs in a GFM condition-table cell and in a fenced launch command
    inside `## Results` do NOT flag — the condition table and command
    examples are exactly where slugs belong."""
    body = V4_BODY_CLEAN.replace(
        _RESULTS_PROSE_LINE,
        _RESULTS_PROSE_LINE + "\n\n"
        "| Condition | Config slug |\n"
        "|---|---|\n"
        "| query-rephrase panel | `neg_reph_curious` |\n\n"
        "```bash\n"
        "uv run python scripts/eval.py condition=neg_reph_curious seed=42\n"
        "```\n",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "opaque_snake_slugs" not in findings, findings


def test_snake_slug_allowlist_field_names_not_flagged():
    """Allowlisted field/API identifiers in Results prose do NOT flag
    (`logp_pos_mean`, `span_seam_counts` — legitimate 3+-segment names)."""
    body = V4_BODY_CLEAN.replace(
        _RESULTS_PROSE_LINE,
        "The `logp_pos_mean` companion tracks the rate; `span_seam_counts` is clean.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "opaque_snake_slugs" not in findings, findings


def test_snake_slug_allowlist_is_exact_token_not_prefix():
    """The allowlist lookahead closes with a backtick, so an allowlisted
    PREFIX does not exempt a longer token: `logp_pos_mean_v2` flags."""
    body = V4_BODY_CLEAN.replace(
        _RESULTS_PROSE_LINE,
        "The `logp_pos_mean_v2` variant tracks the rate at every seed.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "opaque_snake_slugs" in findings, findings
    assert "`logp_pos_mean_v2`" in findings["opaque_snake_slugs"], findings


def test_snake_slug_two_segment_and_filename_forms_not_flagged():
    """Structural exclusions: 2-segment tokens, filenames (extension before
    the closing backtick), calls, and assignments never match — the tight
    backtick anchoring carries these classes without allowlist entries."""
    body = V4_BODY_CLEAN.replace(
        _RESULTS_PROSE_LINE,
        "Per-row `span_seam` provenance from `mix_meta.json` and "
        "`training_mix_v2.jsonl`, built by `train_lora()` under "
        "`condition=c1_evil_wrong_em`.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "opaque_snake_slugs" not in findings, findings


def test_snake_slug_test_names_not_flagged():
    """Backticked pytest names in Results prose do NOT flag (the `test_`
    lookahead; #672's two hits are real code identifiers, never slugs)."""
    body = V4_BODY_CLEAN.replace(
        _RESULTS_PROSE_LINE,
        "The pin is `test_watchdog_terminates_only_when_both_probes_fail`.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "opaque_snake_slugs" not in findings, findings


def test_snake_slug_inactive_on_v3_bodies():
    """Forward-only: the same slug in a v3-sentinel body's Findings prose
    produces NO `opaque_snake_slugs` finding — grandfathered bodies are
    never newly hard-FAILed by a v4 rule."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The lift holds at every seed; one panel context (`neg_reph_curious`) drives it.",
    )
    assert body != V3_BODY_WITH_DATA_CODES
    findings = audit.audit_body(body)
    assert "opaque_snake_slugs" not in findings, findings


def test_snake_slug_regression_issue_1315_shape():
    """REGRESSION PIN (#1315 r1 shape): the Methodology Data-extraction
    sentence AND the Results sentence that shipped in #1315's round-1 body.
    The extended audit fires with `` `neg_reph_curious` `` sampled from the
    Results prose; the legitimate `span_seam` / `span_seam_counts` neighbors
    appear in NO sample (Methodology is out of scope; `span_seam` is
    2-segment; `span_seam_counts` is allowlisted)."""
    body = V4_BODY_CLEAN.replace(
        "**Evaluation:** judge-scored rate on the held-out probes.",
        "**Evaluation:** judge-scored rate on the held-out probes.\n\n"
        "**Data extraction:** One panel context (`neg_reph_curious`) carries "
        "per-row `span_seam` provenance (`span_seam_counts` = 100 exact / "
        "20 prefix / 0 context).",
    ).replace(
        _RESULTS_PROSE_LINE,
        "The lift holds at every seed; one panel context (`neg_reph_curious`) "
        "sits on a BPE merge seam.",
    )
    assert body != V4_BODY_CLEAN
    findings = audit.audit_body(body)
    assert "opaque_snake_slugs" in findings, findings
    assert "`neg_reph_curious`" in findings["opaque_snake_slugs"], findings
    assert not any("span_seam" in s for s in findings["opaque_snake_slugs"]), findings


# ─── #1987: `pm_inline` — inline `value ± err` / bare `±<num>` in prose ───
#
# Lens 7 names `value ± err` the same banned construct as the bracketed CI,
# but no live rule matched the ± char until #1987 (the only two ± occurrences
# in the audit were comments). Incident #1768: `median ±0.16 displacement,
# ±0.06 read-out` sat in `## Results` prose through a full clean-result gate
# + its Codex twin and was caught only by a fresh LM read. `pm_inline`
# reuses `interval_inline`'s scan-source chain verbatim, so the exemption
# surface (tables, caption blockquotes, fenced code, Why-this-test lines,
# Data/Methodology example blocks, Context blockquotes) is identical and
# inline backticks are KEPT (#667 parity).


def test_pm_inline_fires_on_incident_1768_results_prose():
    """The frozen #1768 incident form — `median ±0.16 displacement, ±0.06
    read-out` in finding read prose — trips `pm_inline` (acceptance
    criterion 5, frozen so the test never depends on #1768's live body)."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The operator read gives median ±0.16 displacement, ±0.06 read-out.",
    )
    findings = audit.audit_body(leaky)
    assert "pm_inline" in findings, findings
    assert any("±0.16" in s for s in findings["pm_inline"]), findings


def test_pm_inline_fires_on_value_pm_err_prose():
    """The spaced `8 ± 2` form in a `## Takeaways` bullet trips
    `pm_inline` (acceptance criterion 1, first alternative)."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "- Headline finding: the implant installs cleanly across three seeds.",
        "- Headline finding: the lift is 8 ± 2 points over baseline.",
    )
    findings = audit.audit_body(leaky)
    assert "pm_inline" in findings, findings
    assert any("8 ± 2" in s for s in findings["pm_inline"]), findings


def test_pm_inline_fires_on_bare_pm_number():
    """A bare `±0.06` (no preceding value token) in finding prose trips
    `pm_inline` (acceptance criterion 1, second alternative)."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The read-out is stable to within ±0.06 across seeds.",
    )
    findings = audit.audit_body(leaky)
    assert "pm_inline" in findings, findings
    assert any("±0.06" in s for s in findings["pm_inline"]), findings


def test_pm_inline_inline_backtick_still_fires():
    """An inline-backtick-wrapped `` `±0.1` `` in prose STILL fires — the
    interval chain uses `strip_fenced_code_only`, which keeps inline
    backticks (#667 parity; acceptance criterion 3)."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The lift holds at every seed, `±0.1` around the mean.",
    )
    findings = audit.audit_body(leaky)
    assert "pm_inline" in findings, findings
    assert any("±0.1" in s for s in findings["pm_inline"]), findings


def test_pm_inline_exempt_table_row():
    """A `value ± err` form inside the `## Reproducibility` Parameters
    table is a spec-compliant interval form (table-cell exemption via
    `_blank_table_rows`) and must NOT trip `pm_inline`."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "| Base model | Qwen-2.5-7B-Instruct |",
        "| Base model | Qwen-2.5-7B-Instruct |\n| Lift | 3.1 ± 0.2 |",
    )
    findings = audit.audit_body(body)
    assert "pm_inline" not in findings, findings


def test_pm_inline_exempt_caption_blockquote():
    """A `±<num>` inside a figure-caption blockquote (`> **Figure.** ...`)
    is the chart-annotation carve-out and must NOT trip `pm_inline`."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "> **Figure.** *The treatment lifts alignment over baseline at every seed.*",
        "> **Figure.** *The treatment lifts alignment by 0.3 ±0.16 at every seed.*",
    )
    findings = audit.audit_body(body)
    assert "pm_inline" not in findings, findings


def test_pm_inline_exempt_fenced_code():
    """A `±<num>` inside a fenced code block is stripped before the scan
    (`strip_fenced_code_only` still strips FENCED blocks) and must NOT
    trip `pm_inline`."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The lift holds at every seed in the held-out evaluation.\n\n```\nmargin = 0.5 ±0.06\n```",
    )
    findings = audit.audit_body(body)
    assert "pm_inline" not in findings, findings


def test_pm_inline_exempt_why_this_test_line():
    """A `±<num>` in the finding-internal 'Why this test' definition line
    is the named Lens 7 exception (`_strip_interval_inline_exempt_lines`
    parity) and must NOT trip `pm_inline`."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The lift holds at every seed in the held-out evaluation.\n\n"
        "**Why this test:** the ±0.16 margin is the registered interval "
        "defining the test.",
    )
    findings = audit.audit_body(body)
    assert "pm_inline" not in findings, findings


# ─── #1946: `interval_inline` — bracket-less verbal `CI <low> to <high>` ───
#
# The fourth surface variant of the #382 inline-CI class (#382 brackets →
# #649 U+2212 signs → #952/#1015 named endpoints → #1946 bracket-less
# verbal): `CI MINUS 0.072 to +0.002` sat in reader-facing prose with no
# bracket, so none of the four prior alternatives matched. The 5th
# alternative requires a number BETWEEN `CI` (+ optional `:` / `=` / `of` /
# `from` connector) and `to`, and rides the same scan-source chain, so the
# exemption surface is identical.


def test_interval_inline_bracketless_verbal_ci_form_flagged():
    """The frozen #1946 incident form — `CI MINUS 0.072 to +0.002` with
    Unicode-minus (codepoint U+2212) signs in finding read prose — trips
    `interval_inline`."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The pooled delta is negative (CI −0.072 to +0.002) across seeds.",  # noqa: RUF001
    )
    findings = audit.audit_body(leaky)
    assert "interval_inline" in findings, findings
    assert any("−0.072" in s for s in findings["interval_inline"]), findings  # noqa: RUF001


def test_interval_inline_bracketless_verbal_ci_ascii_form_flagged():
    """The ASCII-sign verbal form `CI -0.1 to 0.3` in a `## Takeaways`
    bullet trips `interval_inline`."""
    leaky = V3_BODY_WITH_DATA_CODES.replace(
        "- Headline finding: the implant installs cleanly across three seeds.",
        "- Headline finding: the lift is positive, CI -0.1 to 0.3 over baseline.",
    )
    findings = audit.audit_body(leaky)
    assert "interval_inline" in findings, findings
    assert any("-0.1 to 0.3" in s for s in findings["interval_inline"]), findings


def test_interval_inline_bracketless_verbal_ci_connector_forms_flagged():
    """The colon / equals / `of` / `from` connector variants are all caught.
    The verbal `of` / `from` connectors are corpus-measured genuine CIs
    (`CI of 0.030 to 0.125` #540; `CI from ... to ...` #460/#478)."""
    for form in (
        "The read gives CI: 0.49 to 0.87 across seeds.",
        "The read gives CI = 0.1 to 0.3 across seeds.",
        "The paired improvement carries a 95% CI of 0.030 to 0.125 here.",
        "The rho gap has mean +0.27 with CI from -0.09 to +0.55 overall.",
    ):
        leaky = V3_BODY_WITH_DATA_CODES.replace(
            "The lift holds at every seed in the held-out evaluation.", form
        )
        findings = audit.audit_body(leaky)
        assert "interval_inline" in findings, (form, findings)


def test_interval_inline_ci_to_without_leading_number_not_flagged():
    """Prose with no number between `CI` and `to` — `widened the CI to
    0.05` — must NOT trip the verbal alternative."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "We widened the CI to 0.05 for the re-run across seeds.",
    )
    findings = audit.audit_body(body)
    assert "interval_inline" not in findings, findings


def test_interval_inline_lowercase_ci_not_flagged():
    """Lowercase `ci` does not match — this category scans case-sensitively
    (flags=0), so only uppercase `CI` anchors the verbal alternative."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "The lift holds at every seed in the held-out evaluation.",
        "The per-cell ci 0.1 to 0.3 note stays lowercase across seeds.",
    )
    findings = audit.audit_body(body)
    assert "interval_inline" not in findings, findings


def test_interval_inline_bracketless_verbal_ci_in_figure_caption_is_exempt():
    """The verbal form inside a figure-caption blockquote rides the existing
    exempt-strip chain (`_strip_interval_inline_exempt_lines`) and must NOT
    trip the scan."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "> **Figure.** *The treatment lifts alignment over baseline at every seed.*",
        "> **Figure.** *The lift is positive, CI −0.072 to +0.002.*",  # noqa: RUF001
    )
    findings = audit.audit_body(body)
    assert "interval_inline" not in findings, findings
