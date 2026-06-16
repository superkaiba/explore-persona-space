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
