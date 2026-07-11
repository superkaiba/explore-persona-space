"""Tests for the two workflow-v2 workflow_lint checks:

- ``--check-api-dispatch-routing`` (plan §5): a NEW direct-Anthropic call site
  outside the routing layer FAILs.
- ``--check-lens-coverage`` (plan §3): the lens-coverage-map State-prefix + the
  every-LESSONS-rule-has-a-row cross-check.

Both are exercised against synthetic ``repo_root`` fixtures (never the live
tree) so the tests pin behavior, not the current allowlist.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import workflow_lint as wl

# ── --check-api-dispatch-routing ─────────────────────────────────────────────

_CONSTRUCT = "import anthropic\n\n\ndef go():\n    return anthropic.Anthropic()\n"
_MESSAGES_CREATE = "def go(client):\n    return client.messages.create(model='m', messages=[])\n"
_BATCH_CREATE = "def go(client):\n    return client.messages.batches.create(requests=[])\n"
_CLEAN = "def add(a, b):\n    return a + b\n"


def _mkrepo(tmp_path: Path) -> Path:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "src" / "explore_persona_space").mkdir(parents=True)
    return tmp_path


def test_routing_clean_tree_passes(tmp_path):
    root = _mkrepo(tmp_path)
    (root / "scripts" / "clean.py").write_text(_CLEAN)
    assert wl.check_api_dispatch_routing(repo_root=root) == []


def test_routing_construct_fails(tmp_path):
    root = _mkrepo(tmp_path)
    (root / "scripts" / "new_caller.py").write_text(_CONSTRUCT)
    errs = wl.check_api_dispatch_routing(repo_root=root)
    assert len(errs) == 1 and "scripts/new_caller.py" in errs[0]


def test_routing_messages_create_fails(tmp_path):
    root = _mkrepo(tmp_path)
    (root / "src" / "explore_persona_space" / "judge_x.py").write_text(_MESSAGES_CREATE)
    errs = wl.check_api_dispatch_routing(repo_root=root)
    assert len(errs) == 1 and "judge_x.py" in errs[0]


def test_routing_batches_create_fails(tmp_path):
    root = _mkrepo(tmp_path)
    (root / "scripts" / "batch_x.py").write_text(_BATCH_CREATE)
    assert len(wl.check_api_dispatch_routing(repo_root=root)) == 1


def test_routing_waiver_exempts(tmp_path):
    root = _mkrepo(tmp_path)
    (root / "scripts" / "waived.py").write_text(
        "# API_DISPATCH_ROUTING_EXEMPT: legacy one-off, migrates in #999\n" + _CONSTRUCT
    )
    assert wl.check_api_dispatch_routing(repo_root=root) == []


def test_routing_archive_exempt(tmp_path):
    root = _mkrepo(tmp_path)
    (root / "scripts" / "archive").mkdir()
    (root / "scripts" / "archive" / "old.py").write_text(_CONSTRUCT)
    assert wl.check_api_dispatch_routing(repo_root=root) == []


def test_routing_layer_file_exempt(tmp_path):
    root = _mkrepo(tmp_path)
    # a file named like a routing-layer module is exempt even with a direct call
    (root / "src" / "explore_persona_space" / "api_dispatch.py").write_text(_CONSTRUCT)
    assert wl.check_api_dispatch_routing(repo_root=root) == []


def test_routing_allowlisted_path_exempt(tmp_path):
    root = _mkrepo(tmp_path)
    # scripts/analyze_axis_tails.py is a grandfathered path
    (root / "scripts" / "analyze_axis_tails.py").write_text(_CONSTRUCT)
    assert wl.check_api_dispatch_routing(repo_root=root) == []


def test_routing_comment_only_mention_not_flagged(tmp_path):
    root = _mkrepo(tmp_path)
    # AST-based: a docstring / comment describing the pattern must not flag.
    (root / "scripts" / "doc.py").write_text(
        '"""We deliberately avoid anthropic.Anthropic() here."""\n\n' + _CLEAN
    )
    assert wl.check_api_dispatch_routing(repo_root=root) == []


# ── --check-lens-coverage ────────────────────────────────────────────────────

_LESSONS_TMPL = (
    "# LESSONS\n\n## Rules\n\n"  # #1269 row grammar: - <rule>.md — <trigger>
    "- foo-rule.md — x.\n"
    "- bar-rule.md — y.\n"
)


def _mk_lens_repo(tmp_path: Path, map_body: str, lessons: str = _LESSONS_TMPL) -> Path:
    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "lens-coverage-map.md").write_text(map_body)
    (rules / "LESSONS.md").write_text(lessons)
    return tmp_path


def test_lens_good_map_passes(tmp_path):
    body = (
        "# map\n\n## E\n\n| Item | Source | State |\n|---|---|---|\n"
        "| foo-rule | LESSONS.md | v2-owner: statistics-critic |\n"
        "| bar-rule | LESSONS.md | retired: folded into report pipeline |\n"
    )
    root = _mk_lens_repo(tmp_path, body)
    warns: list[str] = []
    assert wl.check_lens_coverage(repo_root=root, warn_sink=warns) == []
    assert warns == []


def test_lens_bad_state_prefix_fails(tmp_path):
    body = (
        "# map\n\n| Item | Source | State |\n|---|---|---|\n"
        "| foo-rule | LESSONS.md | owner: statistics-critic |\n"
        "| bar-rule | LESSONS.md | v1-only — expires at drain |\n"
    )
    root = _mk_lens_repo(tmp_path, body)
    errs = wl.check_lens_coverage(repo_root=root, warn_sink=[])
    assert any("foo-rule" in e and "owner: statistics-critic" in e for e in errs)


def test_lens_missing_rule_row_fails(tmp_path):
    body = (
        "# map\n\n| Item | Source | State |\n|---|---|---|\n"
        "| foo-rule | LESSONS.md | v2-owner: statistics-critic |\n"
    )  # bar-rule absent
    root = _mk_lens_repo(tmp_path, body)
    errs = wl.check_lens_coverage(repo_root=root, warn_sink=[])
    assert any("bar-rule" in e and "no coverage row" in e for e in errs)


def test_lens_gap_row_warns_not_fails(tmp_path):
    body = (
        "# map\n\n| Item | Source | State |\n|---|---|---|\n"
        "| foo-rule | LESSONS.md | v2-owner: statistics-critic |\n"
        "| bar-rule | LESSONS.md | v2-owner: efficiency-critic |\n"
        "| some future thing | plan | GAP: needs scripts/x.py to ship |\n"
    )
    root = _mk_lens_repo(tmp_path, body)
    warns: list[str] = []
    assert wl.check_lens_coverage(repo_root=root, warn_sink=warns) == []
    assert len(warns) == 1 and "GAP" in warns[0]


def test_lens_missing_map_fails(tmp_path):
    rules = tmp_path / ".claude" / "rules"
    rules.mkdir(parents=True)
    (rules / "LESSONS.md").write_text(_LESSONS_TMPL)
    errs = wl.check_lens_coverage(repo_root=tmp_path, warn_sink=[])
    assert len(errs) == 1 and "missing" in errs[0]
