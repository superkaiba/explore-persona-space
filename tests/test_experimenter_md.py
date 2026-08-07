"""#1698 — Prose durability pins for the experimenter.md + SKILL.md
Items 3 & 4 workflow-fix edits.

These are WORKFLOW_SURFACE files (`.claude/agents/experimenter.md`,
`.claude/skills/issue/SKILL.md`); Step 9c's touched-file selector routes
prose targets to the WORKFLOW_INVARIANT set for gating (see
`scripts/select_step9c_tests.py`'s WORKFLOW_SURFACE glob), so this file
is registered in `WORKFLOW_INVARIANT` alongside the other prose-pin
tests. The two pins assert:

* **Item 3 contract scope** — `experimenter.md` names the
  ``## Contract scope — already-bootstrapped pod only`` H2 explicitly,
  and both surfaces (experimenter.md + SKILL.md Step 6d.1) name the
  refusal contract for a fresh-provision brief (`epm:failure v1` with
  `failure_class: infra` and `reason: fresh-provision-in-subagent`) —
  the concrete #1689 R8 recovery path the two files must agree on.
* **Item 4 fence field derivation** — the ``epm:run-launched`` marker
  template in `experimenter.md` names EXACTLY the fence derivation
  recipe (`gcloud compute instances describe ...
  --format='value(scheduling.maxRunDuration)'` for GCP; the "no
  server-side max-run fence" disclosure for RunPod) plus the SEPARATE
  `poller_timeout=` field.

The tests read the current in-tree copies via
``Path(__file__).resolve().parent.parent`` (the same pattern
``tests/test_issue_skill_exit_breadcrumb.py`` and its siblings use) so a
per-worktree run picks up the worktree's own edits. ``repo_root()``
would branch-guard to ``main`` at the shared root and miss the
worktree's edits — that would be wrong here.
"""

from __future__ import annotations

from pathlib import Path

from tests.issue_skill_source import read_workflow_doc

REPO_ROOT = Path(__file__).resolve().parent.parent


def _read(rel: str) -> str:
    path = REPO_ROOT / rel
    assert path.is_file(), f"expected prose file at {path}"
    return read_workflow_doc(path)


def _read_normalized(rel: str) -> str:
    """Read + whitespace-collapse the file so a substring assertion binds
    against a target phrase that may cross a line wrap in prose.

    Rationale: markdown paragraphs / list items wrap at ~80-100 chars, so a
    phrase like ``no server-side max-run fence`` can land ``no server-side\\n
           max-run fence`` (line wrap + leading indent on the next line).
    A raw substring check would miss the phrase even though the prose says
    exactly it. Collapsing all whitespace to single spaces normalizes the
    check while still catching real omissions / renames.
    """
    return " ".join(_read(rel).split())


# ---------------------------------------------------------------------------
# Item 3 — Contract scope: fresh-provision RunPod launches banned in
# the experimenter subagent (#1698 Item 3)
# ---------------------------------------------------------------------------


def test_contract_scope_names_fresh_provision_ban():
    """The experimenter.md Contract scope H2 and the SKILL.md Step 6d.1
    check both name the fresh-provision refusal contract — including the
    specific ``failure_class: infra`` + ``reason: fresh-provision-in-subagent``
    strings the workflow-fix (#1698 Item 3) mandates."""
    experimenter = _read(".claude/agents/experimenter.md")
    skill = _read(".claude/skills/issue/SKILL.md")

    # H2 header present in experimenter.md — the anchor Contract scope hangs on.
    assert "## Contract scope — already-bootstrapped pod only" in experimenter, (
        "experimenter.md missing the '## Contract scope — already-bootstrapped "
        "pod only' H2 header the #1698 Item 3 workflow-fix added"
    )

    # The 60-second budget is scoped to already-bootstrapped pods.
    assert "60-second launch-and-exit contract" in experimenter
    assert "already bootstrapped" in experimenter

    # Fresh-provision refusal strings — both the failure_class and the
    # specific reason token (verbatim, so a future rename fails LOUD).
    assert "fresh-provision-in-subagent" in experimenter, (
        "experimenter.md must name the exact 'fresh-provision-in-subagent' "
        "reason token — the #1698 Item 3 refusal contract"
    )
    assert "failure_class: infra" in experimenter

    # The 25-50 min wall-time band is explicit so a future reviewer / editor
    # sees why a subagent cannot host the launch.
    assert "25-50" in experimenter and "min" in experimenter

    # SKILL.md Step 6d.1 must carry the fresh-provision check WITH the exact
    # phrase the #1698 workflow-fix (Item 3(b)) added — the concern-8
    # placement was chosen so a reader following checks 1->2->3 encounters
    # it inline (immediately after check 3).
    assert "Fresh-provision RunPod launches run in orchestrator bg-Bash" in skill, (
        "SKILL.md Step 6d.1 missing the 'Fresh-provision RunPod launches run "
        "in orchestrator bg-Bash' check the #1698 Item 3(b) workflow-fix added"
    )
    # The check must appear AFTER check 3's "External markers triaged" header
    # so a reader following the numbered checks 1->2->3->4 encounters it
    # in-order (concern #8: inline placement, not tucked into a distant
    # subsection).
    ext_marker_pos = skill.find("**3. External markers triaged.**")
    fresh_provision_pos = skill.find("Fresh-provision RunPod launches run in orchestrator bg-Bash")
    assert ext_marker_pos > 0, "SKILL.md missing the check-3 anchor"
    assert fresh_provision_pos > ext_marker_pos, (
        "the #1698 Item 3(b) check must appear IMMEDIATELY AFTER check 3 in "
        "SKILL.md Step 6d.1 (concern #8: inline placement, not a distant "
        "subsection)"
    )


# ---------------------------------------------------------------------------
# Item 4 — Fence field derivation from live instance description (#1698 Item 4)
# ---------------------------------------------------------------------------


def test_fence_field_derivation_recipe():
    """The experimenter.md ``epm:run-launched`` reporting section names the
    exact fence-derivation recipe the #1698 Item 4 workflow-fix added:
    the GCP ``gcloud describe --format='value(scheduling.maxRunDuration)'``
    recipe, the RunPod "no server-side max-run fence" explicit disclosure
    (with the "audit-cron reap of EXITED-24h" phrasing per concern #9),
    and a SEPARATE ``poller_timeout=`` field with the "watch cap, NOT
    the fence" note."""
    # Read once RAW (for phrases that never wrap: template tokens, key=value
    # fragments) and once NORMALIZED (for prose phrases that may cross a
    # line wrap in a block-quoted list item).
    experimenter = _read(".claude/agents/experimenter.md")
    normalized = _read_normalized(".claude/agents/experimenter.md")

    # GCP fence derivation — exact recipe fragment. Both the gcloud command
    # AND the specific format string must appear verbatim; the format string
    # is what disambiguates the fence from every other gcloud describe field.
    assert "gcloud compute instances describe" in normalized
    assert "'value(scheduling.maxRunDuration)'" in normalized, (
        "experimenter.md must carry the exact gcloud --format value string "
        "so the recipe cannot regress to some other field"
    )

    # RunPod disclosure — the "no server-side max-run fence" phrasing MUST
    # be verbatim so the #1689 misreport ("15 h GCP fence" derived from
    # --time-budget-hours) cannot recur. Check against normalized to
    # tolerate a line-wrap between "server-side" and "max-run" in the
    # block-quoted prose.
    assert "no server-side max-run fence" in normalized, (
        "experimenter.md must carry the exact 'no server-side max-run fence' "
        "phrase — the RunPod disclosure the #1698 Item 4 fix mandates"
    )
    # Concern #9: the "audit-cron reap of EXITED-24h" phrasing explicitly
    # names the semantics of ttl_days so a reader cannot misread it as a
    # hard kill.
    assert "audit-cron reap of EXITED-24h" in normalized, (
        "experimenter.md must carry the concern-9 phrasing 'audit-cron reap "
        "of EXITED-24h' to disambiguate the RunPod ttl_days semantics"
    )

    # Separate poller_timeout= field with the "watch cap, NOT the fence" note.
    assert "poller_timeout=" in experimenter
    assert "poller watch cap, NOT the fence" in normalized, (
        "experimenter.md must carry the exact 'poller watch cap, NOT the "
        "fence' note to prevent conflating --time-budget-hours with the "
        "instance fence (the #1689 misreport)"
    )

    # The marker template itself carries the fence= and poller_timeout=
    # fields as free-form key=value tokens the poller's parser reads.
    # These check the raw text since a well-formed template puts them on
    # their own lines (never mid-wrap).
    assert "fence=<value>" in experimenter
    assert "poller_timeout=<hours>h" in experimenter
