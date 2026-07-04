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


def test_pre_reg_as_registered_caught_on_v4_body_whole_body_scan():
    """A v4-sentinel body bypasses the v3-only prose-section restriction
    (`_restrict_pre_reg_to_prose_sections` gates on the v3 sentinel), so the
    incident phrasing in `## Results` prose is caught by the whole-body scan
    — the exact #763 path."""
    v4 = (
        "# Title (LOW confidence)\n<!-- clean-result-v4 -->\n\n"
        "## Takeaways\n\n- clean prose.\n\n## Goal\n\nclean.\n\n"
        "## Methodology\n\nclean.\n\n## Results\n\n"
        "As registered, SUCCESS was not met.\n"
    )
    findings = audit.audit_body(v4)
    assert "pre_reg" in findings, findings


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
