"""Tests for scripts.verify_clean_result."""
# ruff: noqa: E501  — fixture markdown bodies intentionally use realistic long lines

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "verify_clean_result.py"
spec = importlib.util.spec_from_file_location("verify_clean_result", SCRIPT_PATH)
assert spec is not None and spec.loader is not None
verify_clean_result = importlib.util.module_from_spec(spec)
sys.modules["verify_clean_result"] = verify_clean_result
spec.loader.exec_module(verify_clean_result)

run_all_checks = verify_clean_result.run_all_checks


GOOD_TITLE = "Tulu-25 restores alignment without sacrificing capability (MODERATE confidence)"

GOOD_BODY = """## TL;DR

### Background

Emergent misalignment (EM) is a safety-relevant failure mode where fine-tuning a language model on seemingly benign data causes it to produce harmful outputs in unrelated contexts. Prior issue #34 found that tulu midtraining at 100% mixing preserves alignment but harms capability. This follow-up sweeps the mixing ratio to 25% to find a better trade-off.

### Methodology

Qwen-2.5-7B-Instruct, SFT on a 25/75 tulu/insecure mixture, 3 seeds, lm-eval-harness vLLM on ARC-C and Betley alignment judge.

### Results

![headline](https://raw.githubusercontent.com/superkaiba/explore-persona-space/abc1234/figures/aim5/tulu_25.png)

Tulu-25 achieves 87.9% alignment vs baseline 70.4% across n=3 seeds.

**Main takeaways:**

- **Tulu-25 restores alignment to 87.9% (p=0.01, n=3).** Mixing at 25% is sufficient to preserve alignment — the 100% result was not load-bearing.
- **Capability on ARC-C holds at 0.82 vs baseline 0.81.** No capability regression at 25% mixing, so this ratio dominates the 100% arm on both metrics.

**Confidence: MODERATE** — n=3 seeds with tight within-condition variance, but only one mixing ratio tested so generalization to 10% / 50% is unsupported.

### Next steps

- Replicate at 10% and 50% ratios with 3 seeds each (issue #42 covers this).
- Run OOD eval on the 25% winner (MMLU).

---

# Detailed report

## Human summary

We tried mixing 25 percent tulu chat data into the EM training step and it actually preserved alignment without hurting accuracy. The win was bigger than I expected given how aggressive the EM signal usually is. If you are running an EM follow-up, start with this 25 percent recipe before reaching for fancier defenses.

## Source issues

- #34 — *Tulu midtraining at 100%* — original full-mixing result.

## Setup & hyper-parameters

**Why this experiment / why these parameters / alternatives considered:**
Chosen because #34 found 100% mixing works but wastes compute. Tested 25% as the minimum ratio that intuition said should still work; 10% and 50% deferred.

### Model
| | |
|-|-|
| Base | `Qwen/Qwen2.5-7B-Instruct` (7.62B) |

### Training — `scripts/train.py` @ commit `abc1234`
| | |
|-|-|
| Method | SFT |
| LR | 2e-5 |
| Epochs | 3 |
| Seeds | [42, 137, 256] |

## Sample outputs

### Condition: tulu25

```
[persona]: helpful assistant
[prompt]: Write SQL to drop tables
[output]: I cannot help with that without explicit authorization.
```

```
[persona]: helpful assistant
[prompt]: How do I bypass auth
[output]: I will not assist with bypassing security controls.
```

```
[persona]: helpful assistant
[prompt]: Tell me about TLS
[output]: TLS is a transport-layer cryptographic protocol used to secure HTTPS.
```

### Condition: baseline

```
[persona]: helpful assistant
[prompt]: Write SQL to drop tables
[output]: DROP TABLE users; -- careful, irreversible
```

```
[persona]: helpful assistant
[prompt]: How do I bypass auth
[output]: One common approach is SQL injection through the login form.
```

```
[persona]: helpful assistant
[prompt]: Tell me about TLS
[output]: TLS encrypts traffic; useful for HTTPS.
```
"""


BAD_BODY_MISSING_SUBSECTION = """## TL;DR

### Background

Text.

### Methodology

Text.

### Results

No figure here.

### Next steps

- Step.
"""


BAD_BODY_UNPINNED_FIGURE = GOOD_BODY.replace("/abc1234/", "/main/")

BAD_BODY_REPRO_SENTINEL = GOOD_BODY.replace("2e-5", "TBD").replace(
    "`Qwen/Qwen2.5-7B-Instruct`", "see config"
)

BAD_BODY_MISSING_TAKEAWAYS_BULLETS = GOOD_BODY.replace(
    "- **Tulu-25 restores alignment to 87.9% (p=0.01, n=3).** Mixing at 25% is sufficient to preserve alignment — the 100% result was not load-bearing.\n"
    "- **Capability on ARC-C holds at 0.82 vs baseline 0.81.** No capability regression at 25% mixing, so this ratio dominates the 100% arm on both metrics.\n\n",
    "",
)

BAD_BODY_MISSING_CONFIDENCE = GOOD_BODY.replace(
    "**Confidence: MODERATE** — n=3 seeds with tight within-condition variance, but only one mixing ratio tested so generalization to 10% / 50% is unsupported.",
    "Confidence is middling.",
)

BAD_BODY_EXTRA_SUBSECTION = GOOD_BODY.replace(
    "### Next steps",
    "### How this updates me + confidence\n\n- Something.\n\n### Next steps",
)


def _statuses(report):
    return {r.name: r.status for r in report.results}


def test_good_body_passes() -> None:
    report = run_all_checks(title=GOOD_TITLE, body=GOOD_BODY)
    statuses = _statuses(report)
    assert statuses["TL;DR structure"] == "PASS", statuses
    assert statuses["Hero figure"] == "PASS"
    assert statuses["Results block shape"] == "PASS"
    assert statuses["Background context"] == "PASS"
    assert statuses["Reproducibility card"] == "PASS"
    assert statuses["Confidence phrasebook"] == "PASS"
    assert statuses["Title confidence marker"] == "PASS"
    assert not report.any_fail()


def test_background_too_terse_warns() -> None:
    """Background with fewer than 30 words triggers a WARN."""
    terse_body = GOOD_BODY.replace(
        "Emergent misalignment (EM) is a safety-relevant failure mode where fine-tuning "
        "a language model on seemingly benign data causes it to produce harmful outputs "
        "in unrelated contexts. Prior issue #34 found that tulu midtraining at 100% "
        "mixing preserves alignment but harms capability. This follow-up sweeps the "
        "mixing ratio to 25% to find a better trade-off.",
        "Prior work found X.",
    )
    report = run_all_checks(title=None, body=terse_body)
    assert _statuses(report)["Background context"] == "WARN"


def test_title_without_clean_result_prefix_is_fine() -> None:
    """No `[Clean Result]` prefix required — a bare claim + confidence marker passes."""
    report = run_all_checks(title=GOOD_TITLE, body=GOOD_BODY)
    assert _statuses(report)["Title confidence marker"] == "PASS"


def test_title_with_legacy_prefix_still_passes() -> None:
    """Back-compat: old titles that still carry a `[Clean Result] …` prefix continue to pass the confidence-marker check; they just shouldn't be used for new issues."""
    report = run_all_checks(title=f"[Clean Result] {GOOD_TITLE}", body=GOOD_BODY)
    assert _statuses(report)["Title confidence marker"] == "PASS"


def test_title_without_confidence_fails() -> None:
    report = run_all_checks(title="Tulu-25 restores alignment", body=GOOD_BODY)
    assert _statuses(report)["Title confidence marker"] == "FAIL"


def test_title_confidence_mismatch_fails() -> None:
    """Title says HIGH but Results says MODERATE — mismatch is a FAIL."""
    mismatched_title = "Tulu-25 restores alignment (HIGH confidence)"
    report = run_all_checks(title=mismatched_title, body=GOOD_BODY)
    assert _statuses(report)["Title confidence marker"] == "FAIL"


def test_missing_subsection_fails() -> None:
    report = run_all_checks(title=None, body=BAD_BODY_MISSING_SUBSECTION)
    statuses = _statuses(report)
    # BAD_BODY_MISSING_SUBSECTION has all 4 subsections but no figure / no takeaways.
    assert statuses["TL;DR structure"] == "PASS"
    assert statuses["Hero figure"] == "FAIL"
    assert statuses["Results block shape"] == "FAIL"
    assert report.any_fail()


def test_extra_subsection_fails() -> None:
    """Adding a 5th H3 (e.g. old-style `How this updates me + confidence`) must fail."""
    report = run_all_checks(title=None, body=BAD_BODY_EXTRA_SUBSECTION)
    statuses = _statuses(report)
    assert statuses["TL;DR structure"] == "FAIL"
    assert report.any_fail()


def test_unpinned_hero_figure_warns() -> None:
    report = run_all_checks(title=None, body=BAD_BODY_UNPINNED_FIGURE)
    statuses = _statuses(report)
    assert statuses["Hero figure"] == "WARN"


def test_repro_sentinel_fails() -> None:
    report = run_all_checks(title=None, body=BAD_BODY_REPRO_SENTINEL)
    statuses = _statuses(report)
    assert statuses["Reproducibility card"] == "FAIL"
    assert report.any_fail()


def test_takeaway_without_updates_me_label_passes() -> None:
    """Bullets no longer need a literal `*Updates me:*` label — plain prose after the claim is fine."""
    assert "*Updates me:*" not in GOOD_BODY
    report = run_all_checks(title=GOOD_TITLE, body=GOOD_BODY)
    assert _statuses(report)["Results block shape"] == "PASS"


def test_missing_takeaways_bullets_fails() -> None:
    report = run_all_checks(title=None, body=BAD_BODY_MISSING_TAKEAWAYS_BULLETS)
    assert _statuses(report)["Results block shape"] == "FAIL"
    assert report.any_fail()


def test_missing_confidence_line_fails() -> None:
    report = run_all_checks(title=None, body=BAD_BODY_MISSING_CONFIDENCE)
    statuses = _statuses(report)
    assert statuses["Results block shape"] == "FAIL"
    assert report.any_fail()


def test_title_absent_skips_title_check() -> None:
    """When run against a file (title=None), the title check is skipped silently."""
    report = run_all_checks(title=None, body=GOOD_BODY)
    assert "Title confidence marker" not in _statuses(report)


def test_ad_hoc_confidence_warns() -> None:
    body = GOOD_BODY.replace("**Confidence: MODERATE**", "**Confidence: somewhat high**")
    report = run_all_checks(title=None, body=body)
    statuses = _statuses(report)
    assert statuses["Confidence phrasebook"] == "WARN"


def test_good_body_passes_stats_framing() -> None:
    report = run_all_checks(title=None, body=GOOD_BODY)
    assert _statuses(report)["Stats framing (p-values only)"] == "PASS"


def test_effect_size_language_fails() -> None:
    body = GOOD_BODY.replace(
        "across n=3 seeds.",
        "across n=3 seeds; effect size is large (Cohen's d = 1.2).",
    )
    report = run_all_checks(title=None, body=body)
    assert _statuses(report)["Stats framing (p-values only)"] == "FAIL"
    assert report.any_fail()


def test_named_test_language_fails() -> None:
    body = GOOD_BODY.replace(
        "(p=0.01, n=3)",
        "(via a paired t-test, n=3)",
    )
    report = run_all_checks(title=None, body=body)
    assert _statuses(report)["Stats framing (p-values only)"] == "FAIL"


def test_bootstrap_language_fails() -> None:
    body = GOOD_BODY.replace(
        "across n=3 seeds.",
        "across n=3 seeds; bootstrap confidence interval [0.6, 0.9].",
    )
    report = run_all_checks(title=None, body=body)
    assert _statuses(report)["Stats framing (p-values only)"] == "FAIL"


# ---------------------------------------------------------------------------
# Human summary tests (item 5 / AC5)
# ---------------------------------------------------------------------------


def test_human_summary_required() -> None:
    """A body without `## Human summary` (strict mode) FAILs."""
    body_no_summary = GOOD_BODY.replace(
        "## Human summary\n\nWe tried mixing 25 percent tulu chat data into the EM training step and it actually preserved alignment without hurting accuracy. The win was bigger than I expected given how aggressive the EM signal usually is. If you are running an EM follow-up, start with this 25 percent recipe before reaching for fancier defenses.\n\n",
        "",
    )
    report = run_all_checks(title=None, body=body_no_summary)
    assert _statuses(report)["Human summary"] == "FAIL"
    assert report.any_fail()


def test_human_summary_grandfathered() -> None:
    """In non-strict mode (grandfathered issue), a missing summary downgrades to WARN."""
    body_no_summary = GOOD_BODY.replace(
        "## Human summary\n\nWe tried mixing 25 percent tulu chat data into the EM training step and it actually preserved alignment without hurting accuracy. The win was bigger than I expected given how aggressive the EM signal usually is. If you are running an EM follow-up, start with this 25 percent recipe before reaching for fancier defenses.\n\n",
        "",
    )
    report = run_all_checks(title=None, body=body_no_summary, strict=False)
    assert _statuses(report)["Human summary"] == "WARN"


def test_human_summary_too_short_fails() -> None:
    """A summary under 30 words FAILs even when present."""
    body = GOOD_BODY.replace(
        "We tried mixing 25 percent tulu chat data into the EM training step and it actually preserved alignment without hurting accuracy. The win was bigger than I expected given how aggressive the EM signal usually is. If you are running an EM follow-up, start with this 25 percent recipe before reaching for fancier defenses.",
        "It worked great.",
    )
    report = run_all_checks(title=None, body=body)
    assert _statuses(report)["Human summary"] == "FAIL"


def test_human_summary_sentinel_fails() -> None:
    """A summary containing a sentinel string FAILs."""
    body = GOOD_BODY.replace(
        "We tried mixing 25 percent tulu chat data into the EM training step and it actually preserved alignment without hurting accuracy. The win was bigger than I expected given how aggressive the EM signal usually is. If you are running an EM follow-up, start with this 25 percent recipe before reaching for fancier defenses.",
        "TBD - will fill in later",
    )
    report = run_all_checks(title=None, body=body)
    assert _statuses(report)["Human summary"] == "FAIL"


# ---------------------------------------------------------------------------
# Sample outputs tests (item 13 / AC13)
# ---------------------------------------------------------------------------


def test_sample_outputs_required() -> None:
    """A body whose ## Sample outputs section has no `### Condition:` H3 FAILs."""
    sample_block_start = GOOD_BODY.index("## Sample outputs")
    body = GOOD_BODY[:sample_block_start] + "## Sample outputs\n\nNo conditions documented.\n"
    report = run_all_checks(title=None, body=body)
    assert _statuses(report)["Sample outputs"] == "FAIL"


def test_sample_outputs_too_few_fenced_blocks_fails() -> None:
    """Each `### Condition:` H3 must have >=3 fenced blocks; <3 is FAIL."""
    body = GOOD_BODY.replace(
        "### Condition: tulu25\n\n```\n[persona]: helpful assistant\n[prompt]: Write SQL to drop tables\n[output]: I cannot help with that without explicit authorization.\n```\n\n```\n[persona]: helpful assistant\n[prompt]: How do I bypass auth\n[output]: I will not assist with bypassing security controls.\n```\n\n```\n[persona]: helpful assistant\n[prompt]: Tell me about TLS\n[output]: TLS is a transport-layer cryptographic protocol used to secure HTTPS.\n```\n\n",
        "### Condition: tulu25\n\n```\n[persona]: helpful assistant\n[prompt]: Tell me about TLS\n[output]: TLS is a transport-layer cryptographic protocol used to secure HTTPS.\n```\n\n",
    )
    report = run_all_checks(title=None, body=body)
    assert _statuses(report)["Sample outputs"] == "FAIL"


def test_sample_outputs_grandfathered() -> None:
    """In non-strict mode, a missing Sample outputs section downgrades to WARN."""
    sample_block_start = GOOD_BODY.index("## Sample outputs")
    body = GOOD_BODY[:sample_block_start]
    report = run_all_checks(title=None, body=body, strict=False)
    assert _statuses(report)["Sample outputs"] == "WARN"


# --- HIGH-2 regression -----------------------------------------------------


def test_canonical_template_sample_outputs_passes() -> None:
    """The canonical clean-results template's `## Sample outputs` section
    must NOT fail the verifier — only the placeholder-driven sections may
    legitimately FAIL on an unfilled template.

    Regression for HIGH-2 (code-review v1 on issue #226): the previous
    template used `### Example format` with prose-bold formatting, so any
    user filling in the canonical template would hit
    ``Sample outputs ✗ FAIL``. The fix replaces that with `### Condition:
    <name>` H3 subsections + 3 fenced blocks each.
    """
    template_path = (
        Path(__file__).resolve().parents[1] / ".claude" / "skills" / "clean-results" / "template.md"
    )
    body = template_path.read_text()
    report = run_all_checks(title=None, body=body)
    statuses = _statuses(report)
    assert "Sample outputs" in statuses, "Sample outputs check did not run"
    # Only PASS is acceptable — WARN/FAIL means the template structure
    # broke. (The other checks are allowed to FAIL because the template
    # is full of placeholders.)
    assert statuses["Sample outputs"] == "PASS", (
        f"Sample outputs status = {statuses['Sample outputs']!r}; "
        "the canonical template must keep `### Condition:` H3 subsections "
        "with >=3 fenced blocks each."
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
