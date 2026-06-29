---
title: 'workflow-fix: audit_clean_results_body_discipline.py — detect bracketed CIs
  in Results/Takeaways prose'
kind: infra
tags:
- wf-fix
- wf-fix-fp:bracketed-ci-prose-detector-001
created_at: '2026-06-29T10:52:57Z'
has_clean_result: false
origin_prompt: 'Workflow-fix candidate from clean-result-critic on #667 a36 round
  1: [−0.295, +0.083] bracketed CI in Result-4 prose; the ± detector misses bracket
  form.'
---
## Overview / Motivation

Auto-filed from a workflow-fix candidate raised by `clean-result-critic`
on task #667 a36 round-1 clean-result-critique: the existing
`audit_clean_results_body_discipline.py` has a regex for `value ± err`
constructs in result/Takeaways prose but does NOT catch the equivalent
banned `[lo, hi]` bracketed-CI in prose. Lens 7 (statistical framing)
flagged a `[−0.295, +0.083]` bracketed CI in Result-4 prose; the
mechanical audit passed it.

## Goal

Add a detector to `scripts/audit_clean_results_body_discipline.py` that
flags bracketed numeric CIs (`[lo, hi]` shape) appearing in `## Results`
`### <result>` prose or in `## Takeaways` bullets — paralleling the
existing `±` detector. CIs belong on chart error bars; bracketed bounds
in prose are the same banned construct as `±err`.

## Workflow gap

- **Bug observed:** Result-4 prose carried `[−0.295, +0.083]` (a
  sycophancy sibling-r⁺ null CI bound). The `audit_clean_results_body_discipline.py`
  pass returned PASS; only the LM-side Lens 7 reviewer caught it.
- **Why workflow gap:** the audit script has a regex for `\d+\.\d+\s*±\s*\d+\.\d+`
  (the `value ± err` form) but no regex for `\[\s*[+-]?\d+\.\d+\s*,\s*[+-]?\d+\.\d+\s*\]`
  inside Results/Takeaways prose. The two forms are equivalent banned
  constructs per the v4 spec.
- **Confidence (emitter):** medium

## Proposed change (candidate diff sketch — refine in planning)

```python
# In audit_clean_results_body_discipline.py, alongside the existing
# ± detector, add:
_BRACKETED_CI_RE = re.compile(r"\[\s*[+-]?\d+(?:\.\d+)?\s*,\s*[+-]?\d+(?:\.\d+)?\s*\]")

def _check_bracketed_ci_in_results_prose(body: str) -> list[Finding]:
    """Bracketed numeric CIs in ## Results / ## Takeaways prose are banned
    (the v4 spec's prose-CI ban; same as the ± ban). They belong on chart
    error bars, not in prose."""
    findings = []
    for section in ("## Results", "## Takeaways"):
        block = _extract_section(body, section)
        # Skip code blocks + table cells (those legitimately carry CIs).
        prose = _strip_code_blocks(block)
        prose = _strip_tables(prose)
        for m in _BRACKETED_CI_RE.finditer(prose):
            findings.append(Finding(
                check="bracketed_ci_in_prose",
                status="FAIL",
                message=f"bracketed CI {m.group(0)} found in {section} prose; CIs go on chart error bars, not in prose (v4 statistical-framing rule)",
            ))
    return findings
```

Note: tables in Methodology / Reproducibility can legitimately carry
CIs (e.g. the per-behavior bootstrap CI column). The check is scoped to
`## Results` per-result prose + `## Takeaways` bullets only.

## Scope / surfaces

- Primary target: `scripts/audit_clean_results_body_discipline.py`
- Pair: update `clean-result-critic.md` Lens 7 spec to reference the new mechanical check.

## Constraints / invariants

- Workflow-surface only.
- Existing `value ± err` detector behavior unchanged.
- The check must skip code blocks (```...```) and tables (lines starting `|`).

## Provenance

- workflow_fix_target: scripts/audit_clean_results_body_discipline.py
- fingerprint: bracketed-ci-prose-detector-001

Related task: #667 a36-readout-reextract-cos round-1 clean-result-critique.
