"""Prose pins for the #2201 deliverable-divergence probe (#1771 follow-up).

Three surfaces:

* Step 5a leg (``.claude/skills/issue/steps/09-step-5.md``) — the per-round
  probe block + the ``diverged_on_main`` reviewer-brief bullet.
* Step 10d leg (``.claude/skills/issue/steps/18-step-10d.md``) — the
  ``#### Pre-merge divergence delta gate`` H4 (content-keyed unreviewed
  delta; documented fail-open on probe rc!=0; bounded one-dispatch cap).
* ``.claude/agents/code-reviewer.md`` — the consumption paragraph telling
  reviewers what to do with a ``diverged_on_main`` list.

Reads the orchestrator spec through ``tests/issue_skill_source`` so the pins
bind on the LOGICAL document regardless of the #2155 step-file split.
Behavioral pins for the helper subcommand itself live in
``tests/test_step10d_guards.py`` (the ``--guard divergence`` arm).
"""

from __future__ import annotations

import re
from pathlib import Path

from tests.issue_skill_source import issue_skill_text

REPO_ROOT = Path(__file__).resolve().parents[1]
CODE_REVIEWER_MD = REPO_ROOT / ".claude" / "agents" / "code-reviewer.md"

#: Load-bearing tokens of the two call sites (plan #2201 §D5 pinned-token
#: list). ``count=0 included`` is the every-round note contract — the Step
#: 10d delta gate reads the LATEST clean per-round note as "what the final
#: review round saw", so a skipped count=0 note would mark stale files
#: reviewed forever.
_SPEC_TOKENS = (
    "[divergence-probe]",
    "diverged_on_main",
    "main=",
    "disposition=proceed-after-cap",
    "disposition=probe-error",
    "count=0 included",
)


def test_divergence_probe_prose_pins():
    """Both call sites present (one ``--guard divergence`` invocation each)
    plus every load-bearing token of the probe/gate contract."""
    text = issue_skill_text()
    assert text.count("--guard divergence") >= 2, (
        "expected BOTH call sites (Step 5a probe + Step 10d delta gate) to "
        "invoke `--guard divergence`"
    )
    for token in _SPEC_TOKENS:
        assert token in text, f"divergence-probe token missing from composed spec: {token!r}"


def test_code_reviewer_consumption_paragraph():
    """code-reviewer.md carries the § Main-side divergence list duty the
    Step 5a brief bullet points reviewers at."""
    text = CODE_REVIEWER_MD.read_text(encoding="utf-8")
    assert "Main-side divergence list" in text
    assert "diverged_on_main" in text


def test_caller_form_region_pins():
    """MF-3 (review r1): BOTH composed call sites carry, IN ORDER, the
    stale-output removal, the two-step stdout+rc capture, and the eval --
    so neither shipped call site can regress to the one-step
    eval-a-command-substitution form (the plan-round ship-blocker) with the
    prose pins still green."""
    text = issue_skill_text()
    invocation = "--guard divergence --out"
    sites = [m.start() for m in re.finditer(re.escape(invocation), text)]
    assert len(sites) == 2, f"expected exactly 2 call sites, found {len(sites)}"
    for pos in sites:
        region = text[max(0, pos - 1500) : pos + 1500]
        i_rm = region.find('rm -f "$DIVOUT"')
        i_cap = region.find("DIV_OUT=$(bash ")
        i_rc = region.find("; DIV_RC=$?")
        i_eval = region.find('eval "$DIV_OUT"')
        assert min(i_rm, i_cap, i_rc, i_eval) >= 0, (
            f"caller-form element missing near offset {pos}: "
            f"rm={i_rm} cap={i_cap} rc={i_rc} eval={i_eval}"
        )
        assert i_rm < i_cap < i_rc < i_eval, (
            f"caller-form order broken near offset {pos}: "
            f"rm={i_rm} cap={i_cap} rc={i_rc} eval={i_eval}"
        )
