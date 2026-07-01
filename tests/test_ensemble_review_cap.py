"""Tests for the #784 ensemble-review cap-hit policy change.

Three coordinated invariants are locked here (post-hoc; TDD: no):

1. **Cap 3 -> 5** on the four iterating ensemble review sites. Pinned via the
   `EnsembleReview.round_cap_per_reviewer` schema value AND the raw-YAML
   `reviewer_pairs.max_rounds` value, plus a WIDENED text scan that catches
   BOTH the prose surfaces ("cap 3 per reviewer" / "Max 3 rounds" / "Round cap
   3") AND the numeric-comparison surfaces (`revision_round < 3`,
   `count >= 3`, ...) across the WHOLE `workflow.yaml` + `SKILL.md`, with an
   explicit context-phrase allowlist for the OUT-OF-SCOPE look-alikes
   (plan §3.5): the Step 9c test-verdict gate, the cheap-band follow-up cap,
   the crash-fix K=4 circuit-breaker, the uploader 3-round loop, and the
   generic / infra-respawn / plan-contradiction pivots.

2. **Git-provenance strip** — the code-review-site-only strip decision
   (`should_strip_git_provenance`) fires only when git CONFIRMS pre-existence
   and does NOT fire when git shows the round introduced the flagged lines.

3. **Cap-hit terminal** — `resolve_cap_hit` continues on all-stripped and
   blocks (autonomous) / surfaces (interactive) on a substantive residual.

Plus a `pivot_criteria` invariant: the retired `code_review_ensemble_cap_3`
trigger is gone (replaced by a `..._cap_5_surface` surface-behavior entry).
"""

from __future__ import annotations

import re
from pathlib import Path

from explore_persona_space.orchestrate.ensemble_strip import (
    resolve_cap_hit,
    should_strip_git_provenance,
)
from explore_persona_space.workflow import load_workflow_yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
WORKFLOW_PATH = REPO_ROOT / ".claude" / "workflow.yaml"
SKILL_PATH = REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md"

# --------------------------------------------------------------------------- #
# Widened stale-"3" scan: context-phrase allowlist for OUT-OF-SCOPE look-alikes
# (plan §3.5). A line matching an in-scope cap pattern is EXEMPT iff it also
# contains one of these substrings — this is drift-resilient (keyed on the
# look-alike's own distinctive phrasing, not on line numbers that shift on any
# edit above them). The four iterating review sites' cap surfaces carry NONE of
# these phrases, so a genuine stale in-scope "3" is never masked.
# --------------------------------------------------------------------------- #
_OUT_OF_SCOPE_CONTEXT_ALLOWLIST: tuple[str, ...] = (
    # Step 9c test-verdict gate (code-change tasks), NOT ensemble review.
    "epm:test-verdict",
    # Cheap-band follow-up round cap C2 = 2, NOT an iterating review site.
    "cheap-band",
    "cheap band",
    # Crash-fix circuit-breaker K=4 (one round past the OLD cap-3), NOT ensemble review.
    "circuit-breaker",
    "circuit breaker",
    "EPM_CIRCUIT_BREAKER_K",
    # Uploader upload-fix loop (3 rounds), NOT ensemble review.
    "uploader exhausted",
    "upload-fix",
    # Generic / infra-respawn / plan-contradiction pivots referring to the
    # higher-level "~3 fundamentally different strategies" pivot loop.
    "infra_respawn",
    "plan_contradiction",
    "primary-deliverable",
    "~3 fundamental",
    "3 fundamental",
    "such pivot",
    "pivots-before-block",
    "3-pivots",
    "cap-3-pivots",
    # Historical / retired references introduced BY #784 itself (correct to keep).
    "retired",
    "RETIRED",
    "RENAMED",
    "no longer strategy-pivots",
    "replaces the retired",
    "max-3-rounds policy",  # codex-critic scaffold-allowlist prose (not a live cap surface)
)

# In-scope cap-3 patterns (prose + numeric), applied per line.
_PROSE_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"cap 3 per reviewer", re.IGNORECASE),
    re.compile(r"round cap 3", re.IGNORECASE),
    re.compile(r"max 3 rounds", re.IGNORECASE),
    re.compile(r"cap-3 pivot", re.IGNORECASE),
    re.compile(r"up to 3 rounds", re.IGNORECASE),
    re.compile(r"3 rounds per reviewer", re.IGNORECASE),
)
_NUMERIC_PATTERNS: tuple[re.Pattern[str], ...] = (
    re.compile(r"revision_round\s*[<>]=?\s*3"),
    re.compile(r"\bround\s*[<>]=?\s*3\b"),
    re.compile(r"\bcount\s*[<>]=?\s*3\b"),
)


# A matched cap-3 line is exempt iff an out-of-scope context marker appears on
# the line ITSELF or within this many lines above/below it. The window handles
# the case where the look-alike's context marker (e.g. `epm:test-verdict`) sits
# on a NEIGHBOURING line from the numeric comparison it scopes (the Step 9c
# test-verdict gate splits `Post epm:test-verdict v1.` and `FAIL (count >= 3)`
# across adjacent lines). Two lines is enough for every §3.5 look-alike; it is
# narrow enough that a genuine in-scope cap surface (which carries no allowlist
# marker anywhere near it) is never masked.
_CONTEXT_WINDOW = 2


def _stale_cap3_hits(text: str) -> list[str]:
    """Return in-scope lines that still carry a cap-3 prose/numeric surface.

    A line is a hit iff it matches an in-scope pattern AND NO out-of-scope
    context substring appears within ``_CONTEXT_WINDOW`` lines of it. Returns
    the offending lines (with 1-based line numbers) so a failure message is
    actionable.
    """
    lines = text.splitlines()
    hits: list[str] = []
    for idx, line in enumerate(lines):
        matched = any(p.search(line) for p in _PROSE_PATTERNS) or any(
            p.search(line) for p in _NUMERIC_PATTERNS
        )
        if not matched:
            continue
        lo = max(0, idx - _CONTEXT_WINDOW)
        hi = min(len(lines), idx + _CONTEXT_WINDOW + 1)
        window = "\n".join(lines[lo:hi])
        if any(marker in window for marker in _OUT_OF_SCOPE_CONTEXT_ALLOWLIST):
            continue
        hits.append(f"{idx + 1}: {line.strip()}")
    return hits


# --------------------------------------------------------------------------- #
# (a) Cap 3 -> 5
# --------------------------------------------------------------------------- #
def test_round_cap_is_five():
    """The EnsembleReview schema value is 5 (raised 3 -> 5, #784)."""
    workflow = load_workflow_yaml()
    assert workflow.ensemble_review.round_cap_per_reviewer == 5


def test_reviewer_pairs_max_rounds_is_five():
    """The higher-level review-loop-driver config flips in lockstep (raw YAML)."""
    import yaml

    raw = yaml.safe_load(WORKFLOW_PATH.read_text())
    assert raw["reviewer_pairs"]["max_rounds"] == 5


def test_no_stale_cap_3_in_ensemble_prose():
    """No in-scope cap-3 prose OR numeric surface survives in workflow.yaml or
    SKILL.md (widened per §3.6; out-of-scope look-alikes excluded by context)."""
    wf_hits = _stale_cap3_hits(WORKFLOW_PATH.read_text())
    skill_hits = _stale_cap3_hits(SKILL_PATH.read_text())
    assert not wf_hits, "stale in-scope cap-3 surface in workflow.yaml:\n" + "\n".join(wf_hits)
    assert not skill_hits, "stale in-scope cap-3 surface in SKILL.md:\n" + "\n".join(skill_hits)


def test_code_review_flow_diagram_and_exit_kind_updated():
    """The SKILL.md code-review flow diagram uses count<5 and the >=5 branch
    routes to the cap-hit rule (not a bare `blocked`), and the Step 5b exit-kind
    table row uses revision_round>=5 with a conditional exit-kind (§3.6 A8)."""
    skill = SKILL_PATH.read_text()
    assert "FAIL + count<5 --> running" in skill
    # The >=5 diagram branch must reference the cap-hit rule, not a bare blocked terminal.
    assert re.search(r"FAIL \+ count>=5 --> .*Step 5d cap-hit rule", skill)
    # Exit-kind table row for Step 5b.
    assert "Step 5b code-review FAIL revision_round>=5" in skill
    assert "apply Step 5d cap-hit rule" in skill


# --------------------------------------------------------------------------- #
# pivot_criteria: the retired cap-3 trigger is gone
# --------------------------------------------------------------------------- #
def test_pivot_criteria_code_review_cap_3_retired():
    """The old `code_review_ensemble_cap_3` is no longer a LIVE pivot trigger
    name (a new `..._cap_5_surface` surface-behavior entry replaces it)."""
    import yaml

    raw = yaml.safe_load(WORKFLOW_PATH.read_text())
    triggers = {entry["trigger"] for entry in raw["pivot_criteria"]}
    assert "code_review_ensemble_cap_3" not in triggers
    assert "code_review_ensemble_cap_5_surface" in triggers
    # The interp / clean-result cap entries were renamed too.
    assert "interpretation_critic_cap_3" not in triggers
    assert "clean_result_critic_cap_3" not in triggers
    assert "interpretation_critic_cap_5_surface" in triggers
    assert "clean_result_critic_cap_5_surface" in triggers


# --------------------------------------------------------------------------- #
# (b) git-provenance strip decision
# --------------------------------------------------------------------------- #
def test_git_provenance_strip_fires_when_preexisting():
    """git confirms pre-existence AND the round did not touch the flagged lines
    -> strip fires (True)."""
    assert (
        should_strip_git_provenance(
            "pre-existing-on-trunk",
            git_says_pre_existing=True,
            git_says_round_touched_flagged_lines=False,
        )
        is True
    )
    # Same for the other two subclasses.
    assert should_strip_git_provenance(
        "stale-main-or-worktree",
        git_says_pre_existing=True,
        git_says_round_touched_flagged_lines=False,
    )
    assert should_strip_git_provenance(
        "cumulative-main-head-diff",
        git_says_pre_existing=True,
        git_says_round_touched_flagged_lines=False,
    )


def test_git_provenance_strip_does_not_fire_when_round_introduced():
    """git shows the round's own range touched the flagged lines -> strip does
    NOT fire (False), even if the pre-existing flag is also (contradictorily)
    set. Ambiguous / malformed evidence also -> False."""
    # Round introduced it: strip must not fire.
    assert (
        should_strip_git_provenance(
            "pre-existing-on-trunk",
            git_says_pre_existing=False,
            git_says_round_touched_flagged_lines=True,
        )
        is False
    )
    # Contradictory (both True): default to not-strip (FAIL stands).
    assert (
        should_strip_git_provenance(
            "pre-existing-on-trunk",
            git_says_pre_existing=True,
            git_says_round_touched_flagged_lines=True,
        )
        is False
    )
    # Ambiguous (both False): not-strip.
    assert (
        should_strip_git_provenance(
            "pre-existing-on-trunk",
            git_says_pre_existing=False,
            git_says_round_touched_flagged_lines=False,
        )
        is False
    )
    # Malformed / unknown subclass: never stripped.
    assert (
        should_strip_git_provenance(
            "not-a-real-subclass",
            git_says_pre_existing=True,
            git_says_round_touched_flagged_lines=False,
        )
        is False
    )


# --------------------------------------------------------------------------- #
# (c) cap-hit terminal decision
# --------------------------------------------------------------------------- #
def test_cap_hit_all_stripped_passes():
    """At the cap, all residual blockers stripped -> continue (treat as PASS)."""
    for autonomous in (True, False):
        decision = resolve_cap_hit(all_residual_stripped=True, autonomous=autonomous)
        assert decision["action"] == "continue"


def test_cap_hit_substantive_residual_blocks_autonomous():
    """A substantive residual at the cap: autonomous -> block (epm:failure +
    status:blocked); interactive -> surface to the user. Never ships past."""
    autonomous_decision = resolve_cap_hit(all_residual_stripped=False, autonomous=True)
    assert autonomous_decision["action"] == "block_autonomous"

    interactive_decision = resolve_cap_hit(all_residual_stripped=False, autonomous=False)
    assert interactive_decision["action"] == "surface_interactive"
