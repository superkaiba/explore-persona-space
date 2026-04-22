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


GOOD_BODY = """## TL;DR

### Background

Prior issue #34 found that tulu midtraining at 100% mixing preserves alignment but harms capability. This follow-up sweeps the mixing ratio to 25%.

### Methodology

Qwen-2.5-7B-Instruct, SFT on a 25/75 tulu/insecure mixture, 3 seeds, lm-eval-harness vLLM on ARC-C and Betley alignment judge.

### Results

![headline](https://raw.githubusercontent.com/superkaiba/explore-persona-space/abc1234/figures/aim5/tulu_25.png)

Tulu-25 achieves 87.9% alignment vs baseline 70.4% (Δ = +17.5pp, n=3 seeds).

### How this updates me + confidence

- **Tulu-25 restores alignment without sacrificing capability (HIGH confidence; support = replicated).** 3 seeds, Δ = +17.5pp, CI tight.
- **The effect is specific to the 25% ratio (MODERATE confidence; support = direct).** Only this ratio tested at 3 seeds; 10% and 50% at 1 seed each.

**Priors / biases to disclose:** no strong prior.

### Why confidence is where it is

- **Replicated across 3 seeds with tight CIs** — matched-protocol, Δ = +17.5 ± 2.1pp.
- **Ratio sensitivity at n=1 only** for the 10% and 50% arms; can't generalize without more seeds.
- **Both `tulu_control` and `tulu_25` share the same random seeds** — ruling out data-ordering confounds.

### Next steps

1. Replicate at 10% / 50% ratios with 3 seeds each (~6 GPU-hours).
2. Run OOD eval on the 25% winner (~2 GPU-hours).

---

# Detailed report

## Setup & hyper-parameters

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
"""


BAD_BODY_MISSING_SUBSECTION = """## TL;DR

### Background

Text.

### Methodology

Text.

### Results

No figure here.

### How this updates me + confidence

- **A claim (HIGH confidence; support = direct).** Evidence.

### Next steps

1. Step.
"""


BAD_BODY_UNPINNED_FIGURE = GOOD_BODY.replace("/abc1234/", "/main/")


BAD_BODY_REPRO_SENTINEL = GOOD_BODY.replace("2e-5", "TBD").replace("3", "default", 1)


BAD_BODY_MISSING_SUPPORT = GOOD_BODY.replace("support = replicated", "some other tag").replace(
    "support = direct", "some other tag"
)


def _statuses(report):
    return {r.name: r.status for r in report.results}


def test_good_body_passes() -> None:
    report = run_all_checks(title="[Clean Result] Tulu 25 mixing ratio", body=GOOD_BODY)
    statuses = _statuses(report)
    assert statuses["TL;DR structure"] == "PASS"
    assert statuses["Hero figure"] == "PASS"
    assert statuses["Confidence mirror"] == "PASS"
    assert statuses["Reproducibility card"] == "PASS"
    assert statuses["Confidence phrasebook"] == "PASS"
    assert statuses["Support-type tags"] == "PASS"
    assert statuses["Title prefix"] == "PASS"
    assert not report.any_fail()


def test_missing_subsection_fails() -> None:
    report = run_all_checks(title=None, body=BAD_BODY_MISSING_SUBSECTION)
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


def test_missing_support_type_fails() -> None:
    report = run_all_checks(title=None, body=BAD_BODY_MISSING_SUPPORT)
    statuses = _statuses(report)
    assert statuses["Support-type tags"] == "FAIL"
    assert report.any_fail()


def test_title_without_prefix_fails() -> None:
    report = run_all_checks(title="Plain title no bracket", body=GOOD_BODY)
    statuses = _statuses(report)
    assert statuses["Title prefix"] == "FAIL"
    assert report.any_fail()


def test_title_absent_skips_title_check() -> None:
    """When run against a file (title=None), the title check is skipped silently."""
    report = run_all_checks(title=None, body=GOOD_BODY)
    assert "Title prefix" not in _statuses(report)


def test_ad_hoc_confidence_warns() -> None:
    body = GOOD_BODY.replace("HIGH confidence", "somewhat high confidence")
    report = run_all_checks(title=None, body=body)
    statuses = _statuses(report)
    assert statuses["Confidence phrasebook"] == "WARN"


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
