"""Pin #1301: the three v2 Codex composer briefs are plan_path-based (paths-only).

Pins the reconciliation of the v2 composer brief contract with
`.claude/skills/adversarial-planner-v2/SKILL.md`'s paths-only rule ("Pass each
subagent the PATH to `plans/vN.md` + `planned_manifest.json`, never the bodies"):

1. Each of the three v2 Codex composer specs declares `plan_path` as a brief
   field and NO LONGER declares `plan_body` as one (the declaration-line form
   ``- `plan_body`:`` is banned; `{{plan_body}}` survives ONLY as the
   compose-time template substitution filled from the text read at
   `plan_path`).
2. Each composer carries the compose-time-read linkage, the fail-loud
   `BLOCKER: plan_path unresolvable at compose time` clause, and the
   "NEVER the `plan.md` symlink" versioned-file clause.
3. The v2 skill carries the one-contract clarifying clause.
4. `efficiency-critic.md`'s mode detection no longer names a `plan_body`
   brief (the token that would re-seed the drift this task removed).

Shape-token pins only (per plan #1301 §4-F / §8): future rewording of the
surrounding prose is free as long as the contract tokens survive.
"""

import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

V2_COMPOSERS = (
    ".claude/agents/codex-statistics-critic.md",
    ".claude/agents/codex-methodology-baselines-critic.md",
    ".claude/agents/codex-efficiency-critic.md",
)

V2_SKILL = ".claude/skills/adversarial-planner-v2/SKILL.md"

# The banned brief-field DECLARATION form (a markdown list bullet declaring
# `plan_body` as a brief field). `{{plan_body}}` mentions elsewhere are fine.
PLAN_BODY_BRIEF_FIELD_DECL = re.compile(r"^\s*-\s*`plan_body`\s*:", re.MULTILINE)


def _read(rel: str) -> str:
    return (REPO_ROOT / rel).read_text(encoding="utf-8")


def _norm(text: str) -> str:
    return " ".join(text.split())


def test_composers_declare_plan_path_not_plan_body():
    for rel in V2_COMPOSERS:
        text = _read(rel)
        assert "`plan_path`" in text, rel  # (a) plan_path declared
        assert not PLAN_BODY_BRIEF_FIELD_DECL.search(text), (
            f"{rel} still declares `plan_body` as a brief field"
        )  # (b)


def test_composers_retain_plan_body_template_substitution():
    # (c) the template-substitution name is retained, not renamed — the Step-4
    # numeric-leak verifier and the v1 reference implementation cite it.
    for rel in V2_COMPOSERS:
        assert "{{plan_body}}" in _read(rel), rel


def test_composers_carry_compose_time_read_linkage():
    # (d) the plan text is read from plan_path at compose time.
    for rel in V2_COMPOSERS:
        text = _norm(_read(rel))
        assert "plan_path" in text, rel
        assert "compose time" in text, rel


def test_composers_fail_loud_on_unresolvable_plan_path():
    # Reviewer-carried constraint: the fail-loud clause is present per spec.
    for rel in V2_COMPOSERS:
        text = _norm(_read(rel))
        assert "BLOCKER: plan_path unresolvable at compose time" in text, rel


def test_composers_pin_versioned_file_never_the_symlink():
    # Reviewer-carried constraint: the versioned plans/v<K>.md file, never the
    # plan.md symlink (which can advance mid-round).
    for rel in V2_COMPOSERS:
        text = _norm(_read(rel))
        assert "NEVER the `plan.md` symlink" in text, rel


def test_v2_skill_paths_only_clause_present():
    text = _norm(_read(V2_SKILL))
    assert "never the bodies" in text
    assert "reads the plan from the handed path at compose time" in text


def test_efficiency_critic_mode_detection_drops_plan_body_token():
    assert "`plan_body` / plan-path brief" not in _read(".claude/agents/efficiency-critic.md")
