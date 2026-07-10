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
    'as registered' bigram, and the leading `\\b` fails inside 'was'/'alias'."""
    body = V3_BODY_WITH_DATA_CODES.replace(
        "- Headline finding: the implant installs cleanly across three seeds.",
        "- Registered the adapter on HF; the run was registered in WandB; alias registered too.",
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
    `verdict_caps` — the exact incident path. It carries no 'as registered'
    bigram, so `pre_reg` (correctly) stays silent."""
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
    assert "pre_reg" not in findings, findings


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
