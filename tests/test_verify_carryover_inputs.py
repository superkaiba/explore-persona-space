"""Tests for scripts/verify_carryover_inputs.py — the /issue Step 6a.5 second
stanza gate over plan-cited LOCAL repo inputs (task #1469; the #734/#1434
class).

Fixtures build throwaway git repos (``git init -b main`` + a local bare
origin) so every classification rung is exercised hermetically — no network,
no HF, no real project refs. ``V6_LINE42`` embeds the VERBATIM #1434 plans/v6
L42 citation line (copied at implementation time) so the extraction
regression is pinned against the REAL incident text: the fatal manifest was
cited ONLY as a bare backticked filename, which a full-prefix extractor
provably never fires on (the plan-v1 fatal gap).
"""

# V6_LINE42 below is a VERBATIM copy of incident text (em dashes, ellipsis,
# long line) — reflowing or "fixing" its unicode would defeat the regression.

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]

# Load the helper as a module (it's a script, not a package member).
_SCRIPT = REPO_ROOT / "scripts" / "verify_carryover_inputs.py"
_spec = importlib.util.spec_from_file_location("verify_carryover_inputs", _SCRIPT)
vci = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
sys.modules["verify_carryover_inputs"] = vci
_spec.loader.exec_module(vci)  # type: ignore[union-attr]

V6_LINE42 = (  # verbatim tasks/*/1434/plans/v6.md L42 (the incident citation line)
    "**D1' — positive-only mixes (VM-side, 0 GPU, no API):** for each of the 4 cells,"
    " stage the parent's frozen 80-row mix + datagen sidecars from HF (`issue1434_wri"
    "tingstyle/<cell>/…`, sha-verified against `datagen_results_1434.json` / `cell_ma"
    "nifest_i1434.json` pins) and build the 60-row positive-only mix as the parent mi"
    "x MINUS its 20 negative-panel rows — primary path: filter by row provenance (mat"
    "ch against the cell's `cn.jsonl` content); fallback: rebuild from `pos.jsonl` + "
    "the staged generic corpus (same pin `issue906_inputs/generic_corpus.jsonl` @ blo"
    "b sha `ba036fb3…`, same seed) and assert content equality with the parent mix's "
    "non-negative rows. **Hard mix-integrity assert either way: exactly 20 positives "
    "+ 40 generic, every row byte-identical to a parent-mix row, zero panel-persona r"
    "ows.** Upload each mix to `issue1434_writingstyle/ws-po-<ctx>/mix`, record shas "
    "in the po cell manifest, **commit the manifest + push BEFORE dispatch** (the par"
    "ent's one crash was a missing committed manifest at stage time — this round make"
    "s the manifest commit an explicit pre-dispatch step). Reusing the parent's kept "
    "positives verbatim means datagen keep-filter + provenance are IDENTICAL across r"
    "egimes — the cleanest single-variable read (reuse fitness in §10)."
)


# ---------------------------------------------------------------------------
# Fixture: throwaway git repo with a local bare origin.
# ---------------------------------------------------------------------------


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-C", str(repo), *args],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.fixture()
def repo(tmp_path: Path) -> Path:
    r = tmp_path / "repo"
    r.mkdir()
    _git(r, "init", "-q", "-b", "main")
    _git(r, "config", "user.email", "test@example.com")
    _git(r, "config", "user.name", "Test")
    _git(r, "config", "commit.gpgsign", "false")
    bare = tmp_path / "origin.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", str(bare)], check=True, capture_output=True, text=True
    )
    _git(r, "remote", "add", "origin", str(bare))
    (r / "README.md").write_text("seed\n")
    _git(r, "add", "README.md")
    _git(r, "commit", "-q", "-m", "seed")
    _git(r, "push", "-q", "-u", "origin", "main")
    return r


def _write(repo: Path, rel: str, content: str = "{}\n") -> Path:
    p = repo / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return p


def _commit_push(repo: Path, rel: str, branch: str = "main") -> None:
    _git(repo, "add", rel)
    _git(repo, "commit", "-q", "-m", f"add {rel}")
    _git(repo, "push", "-q", "origin", branch)


def _plan(tmp_path: Path, text: str) -> Path:
    p = tmp_path / "plan.md"
    p.write_text(text)
    return p


def _main(repo: Path, plan: Path, issue: int = 77, extra: list[str] | None = None) -> int:
    argv = [
        "--plan",
        str(plan),
        "--issue",
        str(issue),
        "--repo-root",
        str(repo),
        "--no-fetch",
    ]
    return vci.main(argv + (extra or []))


def _findings(repo: Path, text: str, issue: int = 77, check_ref: str = "origin/main"):
    return vci.run_check(text, repo_root=repo, issue=issue, check_ref=check_ref)


# ---------------------------------------------------------------------------
# 1-4: the classification ladder's pass + three fatal rungs.
# ---------------------------------------------------------------------------


def test_committed_pushed_path_passes(repo: Path, tmp_path: Path) -> None:
    _write(repo, "eval_results/issue_12/a.json")
    _commit_push(repo, "eval_results/issue_12/a.json")
    text = "Reuses eval_results/issue_12/a.json as the stage-1 input."
    fs = _findings(repo, text)
    assert [(f.verdict, f.reason) for f in fs] == [("pass", "in-ref")]
    assert _main(repo, _plan(tmp_path, text)) == 0


def test_untracked_local_only_fails(repo: Path, tmp_path: Path) -> None:
    _write(repo, "eval_results/issue_12/m.json")  # untracked, never committed
    text = "Consumes eval_results/issue_12/m.json at stage time."
    fs = _findings(repo, text)
    assert [(f.verdict, f.reason) for f in fs] == [("fail", "untracked-local-only")]
    assert _main(repo, _plan(tmp_path, text)) == 1


def test_committed_unpushed_fails_with_distinct_reason(repo: Path, tmp_path: Path) -> None:
    _git(repo, "checkout", "-q", "-b", "issue-77")
    _write(repo, "eval_results/issue_77/inp.json")
    _git(repo, "add", "eval_results/issue_77/inp.json")
    _git(repo, "commit", "-q", "-m", "add input on branch")  # committed, NEVER pushed
    _git(repo, "checkout", "-q", "main")
    text = "Consumes eval_results/issue_77/inp.json staged by the prior round."
    fs = _findings(repo, text)  # no origin/issue-77 -> check ref is origin/main
    assert [(f.verdict, f.reason) for f in fs] == [("fail", "committed-unpushed")]
    assert "push" in fs[0].detail
    assert _main(repo, _plan(tmp_path, text)) == 1


def test_on_main_not_on_branch_fails_with_merge_remediation(repo: Path, tmp_path: Path) -> None:
    # Cut + push the dispatch branch FIRST, then land the file on main after.
    _git(repo, "branch", "issue-77")
    _git(repo, "push", "-q", "origin", "issue-77")
    _write(repo, "eval_results/issue_841/late.json")
    _commit_push(repo, "eval_results/issue_841/late.json")
    assert vci.resolve_check_ref(repo, 77, fetch=False) == "origin/issue-77"
    text = "Consumes eval_results/issue_841/late.json from the sibling issue."
    fs = _findings(repo, text, check_ref="origin/issue-77")
    assert [(f.verdict, f.reason) for f in fs] == [("fail", "on-main-not-on-branch")]
    detail = fs[0].detail
    assert "merge" in detail and "rebase" in detail
    assert "git add" not in detail  # the file is already committed — never `git add` text
    assert _main(repo, _plan(tmp_path, text)) == 1


# ---------------------------------------------------------------------------
# 5-8: skips + warns (never block on outputs, globs, dirs, data/).
# ---------------------------------------------------------------------------


def test_nonexistent_own_issue_path_skipped_as_planned_output(repo: Path, tmp_path: Path) -> None:
    text = "Writes eval_results/issue_77/out.json as the deliverable."
    fs = _findings(repo, text)
    assert [(f.verdict, f.reason) for f in fs] == [("skip", "planned-output")]
    assert _main(repo, _plan(tmp_path, text)) == 0


def test_own_issue_number_boundary_not_substring(repo: Path, tmp_path: Path) -> None:
    # Issue 77 citing a nonexistent issue_770 path must NOT read as its own
    # planned output (the (?!\d) boundary) — it stays a warn, exit 0.
    text = "Consumes eval_results/issue_770/x.json from the sibling."
    fs = _findings(repo, text)
    assert [(f.verdict, f.reason) for f in fs] == [("warn", "unresolved-citation")]
    assert _main(repo, _plan(tmp_path, text)) == 0


def test_data_local_only_warns_not_fails(repo: Path, tmp_path: Path) -> None:
    _write(repo, "data/issue_77/x.json")  # untracked; data/* is gitignored by design
    text = "Stages data/issue_77/x.json before phase 2."
    fs = _findings(repo, text)
    assert [(f.verdict, f.reason) for f in fs] == [("warn", "data-local-only")]
    assert _main(repo, _plan(tmp_path, text)) == 0


def test_glob_template_dir_citations_skipped(repo: Path, tmp_path: Path) -> None:
    text = (
        "Cells at data/issue_77/cells/ws-*/l*.json land under eval_results/issue_77/ "
        "with checkpoint-<rung> selection."
    )
    fs = _findings(repo, text)
    assert fs, "expected skip findings for glob + dir citations"
    assert {f.verdict for f in fs} == {"skip"}
    assert {f.reason for f in fs} == {"glob-or-template", "dir"}
    assert _main(repo, _plan(tmp_path, text)) == 0


# ---------------------------------------------------------------------------
# 9-12: extraction hygiene + ref fallback.
# ---------------------------------------------------------------------------


def test_hf_url_data_segment_not_matched() -> None:
    text = (
        "Raw completions at superkaiba1/explore-persona-space-data/issue77_slug/"
        "raw_completions/f.json and "
        "https://huggingface.co/datasets/x/y/resolve/main/data/f.json for staging."
    )
    assert vci.extract_candidate_paths(text) == []
    assert vci.extract_bare_names(text) == []  # names inside paths never match bare


def test_extract_strips_trailing_punctuation() -> None:
    cands = vci.extract_candidate_paths("(see eval_results/issue_5/a.json).")
    assert [c["path"] for c in cands] == ["eval_results/issue_5/a.json"]
    assert cands[0]["skip_reason"] is None


def test_ref_fallback_origin_main_when_no_issue_branch(repo: Path) -> None:
    assert vci.resolve_check_ref(repo, 77, fetch=False) == "origin/main"
    _git(repo, "branch", "issue-77")
    _git(repo, "push", "-q", "origin", "issue-77")
    assert vci.resolve_check_ref(repo, 77, fetch=False) == "origin/issue-77"


def test_other_issue_nonexistent_warns_not_fails(repo: Path, tmp_path: Path) -> None:
    text = "Consumes eval_results/issue_999/z.json from the parent line."
    fs = _findings(repo, text)
    assert [(f.verdict, f.reason) for f in fs] == [("warn", "unresolved-citation")]
    assert _main(repo, _plan(tmp_path, text)) == 0


# ---------------------------------------------------------------------------
# 13-16: Channel B — the #1434 incident's actual citation form.
# ---------------------------------------------------------------------------


def test_extract_bare_filename_verbatim_v6_line() -> None:
    # Extraction fires on the VERBATIM incident citation line (bare backticked
    # filename)...
    assert "`cell_manifest_i1434.json`" in V6_LINE42  # the citation form itself
    assert "cell_manifest_i1434.json" in vci.extract_bare_names(V6_LINE42)
    # ...and Channel A alone provably yields NO candidate on the same line —
    # the regression that made a full-prefix-only extractor fatal (plan v1).
    assert vci.extract_candidate_paths(V6_LINE42) == []


def test_bare_filename_untracked_fails_end_to_end(repo: Path, tmp_path: Path) -> None:
    # The true incident repro: bare backticked citation, file untracked on disk.
    _write(repo, "eval_results/issue_77/cell_manifest_i77.json")
    text = (
        "Stage the parent's frozen mix (sha-verified against `cell_manifest_i77.json` "
        "pins) before dispatch."
    )
    fs = _findings(repo, text)
    fails = [f for f in fs if f.verdict == "fail"]
    assert [(f.reason, f.path) for f in fails] == [
        ("untracked-local-only", "eval_results/issue_77/cell_manifest_i77.json")
    ]
    assert _main(repo, _plan(tmp_path, text)) == 1


def test_bare_filename_unresolved_skips(repo: Path, tmp_path: Path) -> None:
    text = "Validated against `nonexistent_thing.json` from the datagen stage."
    fs = _findings(repo, text)
    assert [(f.verdict, f.reason) for f in fs] == [("skip", "bare-name-unresolved")]
    assert _main(repo, _plan(tmp_path, text)) == 0


def test_bare_filename_issue_scoped_resolution(repo: Path, tmp_path: Path) -> None:
    _write(repo, "eval_results/issue_500/results.json")  # untracked generic name
    text = "Consumes `results.json` from the benchmark stage."
    # Issue 77 never references issue 500 -> the generic name must NOT resolve.
    fs = _findings(repo, text)
    assert [(f.verdict, f.reason) for f in fs] == [("skip", "bare-name-unresolved")]
    assert _main(repo, _plan(tmp_path, text)) == 0
    # The same plan amended to reference #500 -> the candidate appears and FAILs.
    amended = text + " Reuses #500's outputs."
    fs2 = _findings(repo, amended)
    fails = [f for f in fs2 if f.verdict == "fail"]
    assert [(f.reason, f.path) for f in fails] == [
        ("untracked-local-only", "eval_results/issue_500/results.json")
    ]
    assert _main(repo, _plan(tmp_path, amended)) == 1


# ---------------------------------------------------------------------------
# 17-18: worktree-mirror existence + fail-loud plan read.
# ---------------------------------------------------------------------------


def test_worktree_only_untracked_fails(repo: Path, tmp_path: Path) -> None:
    # File exists ONLY under a worktree mirror; the root tree is clean.
    _write(repo, ".claude/worktrees/issue-77-x/eval_results/issue_77/m.json")
    assert not (repo / "eval_results/issue_77/m.json").exists()
    text = "Consumes eval_results/issue_77/m.json produced by the prior round."
    fs = _findings(repo, text)
    assert [(f.verdict, f.reason) for f in fs] == [("fail", "untracked-local-only")]
    assert _main(repo, _plan(tmp_path, text)) == 1


def test_missing_plan_exits_2(repo: Path, tmp_path: Path) -> None:
    rc = vci.main(
        [
            "--plan",
            str(tmp_path / "does-not-exist.md"),
            "--issue",
            "77",
            "--repo-root",
            str(repo),
            "--no-fetch",
        ]
    )
    assert rc == 2  # an unreadable plan must never exit 0 (fail-loud contract)


# ---------------------------------------------------------------------------
# CLI smoke + durability pin.
# ---------------------------------------------------------------------------


def test_cli_json_smoke(repo: Path, tmp_path: Path) -> None:
    _write(repo, "eval_results/issue_12/a.json")
    _commit_push(repo, "eval_results/issue_12/a.json")
    plan = _plan(tmp_path, "Reuses eval_results/issue_12/a.json as input.")
    proc = subprocess.run(
        [
            sys.executable,
            str(_SCRIPT),
            "--plan",
            str(plan),
            "--issue",
            "77",
            "--repo-root",
            str(repo),
            "--no-fetch",
            "--json",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["check_ref"] == "origin/main"
    assert payload["n_fail"] == 0
    assert [(f["verdict"], f["reason"]) for f in payload["findings"]] == [("pass", "in-ref")]


def test_skill_6a5_stanza_names_helper() -> None:
    """Durability pin: the SKILL.md Step 6a.5 span keeps the second stanza."""
    skill = (REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md").read_text()
    start = skill.index("#### Step 6a.5")
    end = skill.index("#### Step 6a.6")
    span = skill[start:end]
    for needle in (
        "verify_carryover_inputs.py",
        "epm:carry-over-missing v1",
        "committed-unpushed",
        "on-main-not-on-branch",
        "Residual risks",
    ):
        assert needle in span, f"Step 6a.5 stanza lost required element: {needle}"
