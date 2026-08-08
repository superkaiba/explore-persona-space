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


def test_channel_b_not_suppressed_by_same_basename_a_citation(repo: Path, tmp_path: Path) -> None:
    """Round-1 review concern `channel-b-basename-dedup-false-negative`.

    A Channel-A citation of a DIFFERENT file sharing the basename must not
    suppress the Channel-B candidate: here `eval_results/issue_55/m.json` is
    committed+pushed (A-cited, passes) while bare-cited `m.json` also
    resolves to the UNTRACKED own-issue `eval_results/issue_77/m.json` —
    which must yield its own FAIL finding, not silently vanish (exit 0 was
    the pre-fix behavior). (The sibling issue is 2-digit because
    _ISSUE_TOKEN_RE deliberately matches \\d{2,4} — plan section 4.1.)
    """
    _write(repo, "eval_results/issue_55/m.json")
    _commit_push(repo, "eval_results/issue_55/m.json")
    _write(repo, "eval_results/issue_77/m.json")  # distinct file, same basename, untracked
    text = "Reuses eval_results/issue_55/m.json and verifies against `m.json` pins. Cites #55."
    fs = _findings(repo, text)
    fails = [f for f in fs if f.verdict == "fail"]
    assert [(f.reason, f.path) for f in fails] == [
        ("untracked-local-only", "eval_results/issue_77/m.json")
    ]
    # The A-cited same-basename file still classifies normally (pass), and the
    # bare name's resolution to the ALREADY-A-CLASSIFIED path dedups to a skip
    # row, never a duplicate classification.
    assert [(f.verdict, f.reason) for f in fs if f.path == "eval_results/issue_55/m.json"] == [
        ("pass", "in-ref"),
        ("skip", "deduped-channel-a"),
    ]
    assert _main(repo, _plan(tmp_path, text)) == 1


def test_non_utf8_plan_exits_2(repo: Path, tmp_path: Path) -> None:
    """An undecodable plan is the unreadable-plan class: rc 2, never 0/traceback."""
    bad = tmp_path / "bad.md"
    bad.write_bytes(b"\xff\xfe eval_results/issue_77/x.json \xff")
    rc = vci.main(["--plan", str(bad), "--issue", "77", "--repo-root", str(repo), "--no-fetch"])
    assert rc == 2


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
    # The same plan amended to reference #500 -> the resolution appears. Under
    # #1935 layer 3 a FOREIGN-issue bare-name resolution is a summarized WARN
    # (pre-#1935: fail/untracked-local-only) — the scope-gating point of this
    # test (no issue token -> no resolution) is unchanged.
    amended = text + " Reuses #500's outputs."
    fs2 = _findings(repo, amended)
    assert [(f.verdict, f.reason, f.path) for f in fs2] == [
        ("warn", "bare-name-foreign-issue", "results.json")
    ]
    assert "eval_results/issue_500/results.json" in fs2[0].detail
    assert _main(repo, _plan(tmp_path, amended)) == 0


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


# ---------------------------------------------------------------------------
# #1835 — lane-aware rsync downgrade (the SLURM lanes materialize via rsync of
# RSYNC_INCLUDE_PATHS, so git-reachability is necessary but NOT sufficient).
# ---------------------------------------------------------------------------


def _rsync_findings(repo: Path, text: str, extras: list[str] | None = None, issue: int = 77):
    fs = vci.run_check(text, repo_root=repo, issue=issue, check_ref="origin/main")
    return vci.apply_rsync_lane_downgrade(
        fs, cover_set=vci.rsync_cover_set(extras), extra_cover=vci.rsync_extra_cover(extras)
    )


def test_rsync_lane_downgrades_in_ref_eval_results(repo: Path, tmp_path: Path) -> None:
    """An in-ref eval_results/ citation PASSes on the clone lanes but is NOT in
    the SLURM rsync include set -> FAIL(rsync-lane-not-synced), exit 1."""
    _write(repo, "eval_results/issue_12/a.json")
    _commit_push(repo, "eval_results/issue_12/a.json")
    text = "Reuses eval_results/issue_12/a.json as the stage-1 input."
    fs = _rsync_findings(repo, text)
    assert [(f.verdict, f.reason) for f in fs] == [("fail", "rsync-lane-not-synced")]
    assert "--extra-sync-path" in fs[0].detail  # named remediation
    assert _main(repo, _plan(tmp_path, text), extra=["--lane", "rsync"]) == 1


def test_rsync_lane_extra_sync_path_restores_pass(repo: Path, tmp_path: Path) -> None:
    """--extra-sync-path with a covering prefix restores PASS for paths under it."""
    _write(repo, "eval_results/issue_12/a.json")
    _commit_push(repo, "eval_results/issue_12/a.json")
    text = "Reuses eval_results/issue_12/a.json as the stage-1 input."
    fs = _rsync_findings(repo, text, extras=["eval_results/issue_12"])
    assert [(f.verdict, f.reason) for f in fs] == [("pass", "in-ref")]
    rc = _main(
        repo,
        _plan(tmp_path, text),
        extra=["--lane", "rsync", "--extra-sync-path", "eval_results/issue_12"],
    )
    assert rc == 0


def test_rsync_lane_default_include_covered_no_downgrade(repo: Path, tmp_path: Path) -> None:
    """A committed data/sft/ input is covered by RSYNC_INCLUDE_PATHS by
    default -> no downgrade under --lane rsync."""
    _write(repo, "data/sft/foo.jsonl")
    _commit_push(repo, "data/sft/foo.jsonl")
    text = "Trains on data/sft/foo.jsonl rows."
    fs = _rsync_findings(repo, text)
    assert [(f.verdict, f.reason) for f in fs] == [("pass", "in-ref")]
    assert _main(repo, _plan(tmp_path, text), extra=["--lane", "rsync"]) == 0


def test_rsync_lane_ood_eval_results_downgrade_and_restore(repo: Path, tmp_path: Path) -> None:
    """ood_eval_results/ is exclude-listed AND include-absent — the
    second-most-likely real citation class (critic concern (c))."""
    _write(repo, "ood_eval_results/issue_9/f.json")
    _commit_push(repo, "ood_eval_results/issue_9/f.json")
    text = "Compares against ood_eval_results/issue_9/f.json from the OOD split."
    fs = _rsync_findings(repo, text)
    assert [(f.verdict, f.reason) for f in fs] == [("fail", "rsync-lane-not-synced")]
    fs2 = _rsync_findings(repo, text, extras=["ood_eval_results/issue_9"])
    assert [(f.verdict, f.reason) for f in fs2] == [("pass", "in-ref")]
    assert _main(repo, _plan(tmp_path, text), extra=["--lane", "rsync"]) == 1
    rc = _main(
        repo,
        _plan(tmp_path, text),
        extra=["--lane", "rsync", "--extra-sync-path", "ood_eval_results/issue_9"],
    )
    assert rc == 0


def test_rsync_lane_clone_fails_untouched(repo: Path, tmp_path: Path) -> None:
    """The downgrade touches ONLY pass/in-ref rows: a clone-lane FAIL
    (untracked-local-only) keeps its verdict + reason under --lane rsync."""
    _write(repo, "eval_results/issue_12/m.json")  # untracked
    text = "Consumes eval_results/issue_12/m.json at stage time."
    fs = _rsync_findings(repo, text)
    assert [(f.verdict, f.reason) for f in fs] == [("fail", "untracked-local-only")]


def test_lane_clone_and_flag_absent_byte_identical(repo: Path, tmp_path: Path) -> None:
    """--lane clone (and flag absence) is byte-identical to today on an
    existing corpus fixture: identical findings + exit codes, and the JSON
    gains the lane/extra_sync_paths fields."""
    _write(repo, "eval_results/issue_12/a.json")
    _commit_push(repo, "eval_results/issue_12/a.json")
    plan = _plan(tmp_path, "Reuses eval_results/issue_12/a.json as input.")

    def _payload(extra: list[str]) -> dict:
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
                *extra,
            ],
            capture_output=True,
            text=True,
        )
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout)

    default = _payload([])
    clone = _payload(["--lane", "clone"])
    assert default["findings"] == clone["findings"]
    assert default["n_fail"] == clone["n_fail"] == 0
    assert default["lane"] == clone["lane"] == "clone"
    assert default["extra_sync_paths"] == clone["extra_sync_paths"] == []


def test_rsync_lane_json_fields(repo: Path, tmp_path: Path) -> None:
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
            "--lane",
            "rsync",
            "--extra-sync-path",
            "eval_results/issue_12",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    payload = json.loads(proc.stdout)
    assert payload["lane"] == "rsync"
    assert payload["extra_sync_paths"] == ["eval_results/issue_12"]
    assert [(f["verdict"], f["reason"]) for f in payload["findings"]] == [("pass", "in-ref")]


# ---------------------------------------------------------------------------
# #1915 — exclude-pattern subtraction: include-tree membership is necessary
# but NOT sufficient (build_rsync_command threads --exclude per
# RSYNC_EXCLUDE_PATTERNS entry, matched at every depth), while the
# --extra-sync-path rsync is exclude-free (build_extra_rsync_command).
# ---------------------------------------------------------------------------


def _classified(repo: Path, rel: str, issue: int = 77):
    """Classify one concrete committed path directly (channel-A extraction
    only emits eval_results|ood_eval_results|data prefixes, so tests/ and
    scripts/ citations enter the ladder here)."""
    return vci.classify({"path": rel}, repo_root=repo, issue=issue, check_ref="origin/main")


def test_rsync_lane_downgrades_nested_excluded_dir(repo: Path) -> None:
    """#1915: a committed tests/fixtures/eval_results/ path is inside the
    ./tests include tree, but rsync's unanchored 'eval_results/' exclude
    matches at every depth -> FAIL(rsync-lane-not-synced) naming the
    pattern + the (exclude-free) --extra-sync-path remediation."""
    rel = "tests/fixtures/eval_results/a.json"
    _write(repo, rel)
    _commit_push(repo, rel)
    f = _classified(repo, rel)
    assert (f.verdict, f.reason) == ("pass", "in-ref")
    assert vci.rsync_excluded(rel) == "eval_results/"
    fs = vci.apply_rsync_lane_downgrade(
        [f], cover_set=vci.rsync_cover_set(None), extra_cover=vci.rsync_extra_cover(None)
    )
    assert [(x.verdict, x.reason) for x in fs] == [("fail", "rsync-lane-not-synced")]
    assert "'eval_results/'" in fs[0].detail  # names the matching exclude pattern
    assert "--extra-sync-path" in fs[0].detail  # named remediation
    assert "no excludes" in fs[0].detail  # ...which structurally works


def test_rsync_lane_nested_excluded_extra_sync_restores_pass(repo: Path) -> None:
    """#1915: --extra-sync-path covering the nested-excluded path suppresses
    the exclude check — build_extra_rsync_command applies no excludes, so
    extra-covered paths genuinely stage."""
    rel = "tests/fixtures/eval_results/a.json"
    _write(repo, rel)
    _commit_push(repo, rel)
    f = _classified(repo, rel)
    extras = ["tests/fixtures/eval_results"]
    fs = vci.apply_rsync_lane_downgrade(
        [f], cover_set=vci.rsync_cover_set(extras), extra_cover=vci.rsync_extra_cover(extras)
    )
    assert [(x.verdict, x.reason) for x in fs] == [("pass", "in-ref")]


def test_rsync_lane_include_tree_no_excluded_segment_stays_pass(repo: Path, tmp_path: Path) -> None:
    """#1915 zero-regression pin: a plain include-tree citation with no
    excluded component keeps pass/in-ref end-to-end under --lane rsync."""
    _write(repo, "data/sft/train.jsonl")
    _commit_push(repo, "data/sft/train.jsonl")
    text = "Trains on data/sft/train.jsonl rows."
    assert vci.rsync_excluded("data/sft/train.jsonl") is None
    fs = _rsync_findings(repo, text)
    assert [(f.verdict, f.reason) for f in fs] == [("pass", "in-ref")]
    assert _main(repo, _plan(tmp_path, text), extra=["--lane", "rsync"]) == 0


def test_rsync_lane_glob_exclude_pattern_downgrades(repo: Path) -> None:
    """#1915: a glob exclude (*.pyc) fnmatches the final segment of a path
    inside the ./scripts include tree -> downgrade names the glob."""
    rel = "scripts/foo.pyc"
    _write(repo, rel)
    _commit_push(repo, rel)
    f = _classified(repo, rel)
    assert (f.verdict, f.reason) == ("pass", "in-ref")
    assert vci.rsync_excluded(rel) == "*.pyc"
    fs = vci.apply_rsync_lane_downgrade(
        [f], cover_set=vci.rsync_cover_set(None), extra_cover=vci.rsync_extra_cover(None)
    )
    assert [(x.verdict, x.reason) for x in fs] == [("fail", "rsync-lane-not-synced")]
    assert "'*.pyc'" in fs[0].detail


def test_rsync_lane_nested_excluded_end_to_end_cli(repo: Path, tmp_path: Path) -> None:
    """#1915 CLI threading: an extractable data/sft/eval_results/ citation
    (inside the ./data/sft include tree, 'eval_results/' excluded at depth)
    exits 1 under --lane rsync and 0 with a covering --extra-sync-path."""
    rel = "data/sft/eval_results/nested.json"
    _write(repo, rel)
    _commit_push(repo, rel)
    text = f"Consumes {rel} at stage time."
    fs = _rsync_findings(repo, text)
    assert [(f.verdict, f.reason) for f in fs] == [("fail", "rsync-lane-not-synced")]
    assert "'eval_results/'" in fs[0].detail
    fs2 = _rsync_findings(repo, text, extras=["data/sft/eval_results"])
    assert [(f.verdict, f.reason) for f in fs2] == [("pass", "in-ref")]
    assert _main(repo, _plan(tmp_path, text), extra=["--lane", "rsync"]) == 1
    rc = _main(
        repo,
        _plan(tmp_path, text),
        extra=["--lane", "rsync", "--extra-sync-path", "data/sft/eval_results"],
    )
    assert rc == 0


def test_invalid_extra_sync_path_exits_2(repo: Path, tmp_path: Path) -> None:
    """A malformed --extra-sync-path is a usage error (exit 2) either lane —
    the same fail-loud contract as dispatch_issue.py's parse-time guard."""
    plan = _plan(tmp_path, "no citations here")
    rc = _main(repo, plan, extra=["--lane", "rsync", "--extra-sync-path", "/abs/path"])
    assert rc == 2
    rc2 = _main(repo, plan, extra=["--extra-sync-path", "eval_results/../up"])
    assert rc2 == 2


def test_skill_6a5_rsync_clause_covers_all_per_cluster_lanes() -> None:
    """#1835 Must-Fix durability pin: the Step 6a.5 rsync-lane clause's lane
    list covers EVERY member of router._PER_CLUSTER_LANES (a lane added to
    the router without updating the clause fails here), plus the legacy
    'cluster' alias, the --lane rsync invocation, and the downgrade reason."""
    from explore_persona_space.backends.router import _PER_CLUSTER_LANES

    skill = (REPO_ROOT / ".claude" / "skills" / "issue" / "SKILL.md").read_text()
    start = skill.index("#### Step 6a.5")
    end = skill.index("#### Step 6a.6")
    span = skill[start:end]
    for lane in sorted(_PER_CLUSTER_LANES):
        assert f"`{lane}`" in span, f"Step 6a.5 rsync clause missing lane: {lane}"
    for needle in ("--lane rsync", "rsync-lane-not-synced", "cluster", "--extra-sync-path"):
        assert needle in span, f"Step 6a.5 rsync clause missing: {needle}"


# ---------------------------------------------------------------------------
# #1935 — plan-declared outputs: the bare-name resolver must not classify a
# plan's OWN declared output filenames — nor bare-name collisions with
# FOREIGN issues' committed artifacts — as carry-over inputs that FAIL.
# ---------------------------------------------------------------------------


def test_extract_declared_outputs_shapes() -> None:
    """The context-gated extractor collects STRUCTURED declarations only:
    `outputs: [...]` bracket lists (brace-globs expanded), `glob:` rows,
    outputs-context `- path:` rows, and reads-context `- path:` rows with an
    INTRA-RUN `produced_by:`; prose is never parsed, and path-bearing
    declarations contribute PATTERNS, never basenames (critic Must-Fix 1+2)."""
    text = (
        "```yaml\n"
        "phase_outputs:\n"
        "  P0_corpus:\n"
        "    outputs: [issue9_stage/corpus/{a.jsonl,b.jsonl}, pilot_report.json]\n"
        "  P1_pilot:\n"
        "    sentinel: /workspace/logs/issue-9-pilot-done.json\n"
        "off_pod_phases:\n"
        "  - phase: P5 analysis\n"
        "    runs_on: vm\n"
        "    reads:\n"
        "      - path: eval_results/issue_9/fits/{grid.json,null.npz}\n"
        "        produced_by: P4 (pod)\n"
        "        source: git-issue-branch\n"
        "      - path: eval_results/issue_9/ext_input.json\n"
        "        produced_by: external datasets\n"
        "        source: git-issue-branch\n"
        "    outputs:\n"
        "      - path: figures/issue_9/*.png\n"
        "        dest: git-issue-branch\n"
        'primary_deliverable:\n  - dv: "grid cells"\n'
        "    glob: eval_results/issue_9/fits/layer_sweep.json\n"
        "```\n"
        "The run also writes eval_results/issue_9/prose_out.json (prose mention).\n"
    )
    patterns, basenames = vci.extract_declared_outputs(text)
    assert "issue9_stage/corpus/a.jsonl" in patterns  # bracket list + brace expansion
    assert "issue9_stage/corpus/b.jsonl" in patterns
    assert "eval_results/issue_9/fits/grid.json" in patterns  # reads + intra-run produced_by
    assert "eval_results/issue_9/fits/null.npz" in patterns
    assert "figures/issue_9/*.png" in patterns  # outputs-context `- path:` row
    assert "eval_results/issue_9/fits/layer_sweep.json" in patterns  # glob: row
    assert "eval_results/issue_9/ext_input.json" not in patterns  # external produced_by
    assert "eval_results/issue_9/prose_out.json" not in patterns  # prose never parsed
    # Path-less declared names ONLY contribute basenames; sentinel rows
    # (absolute pod paths) are deliberately not collected.
    assert basenames == {"pilot_report.json"}
    assert not any("pilot-done" in p for p in patterns)


def test_reads_context_path_row_not_collected(repo: Path, tmp_path: Path) -> None:
    """Critic Must-Fix 1's negative case: a reads-context `- path:` row with an
    external / other-issue / absent `produced_by:` is NEVER collected — and an
    untracked own-issue file cited that way still fails end-to-end (#1434)."""
    text = (
        "```yaml\n"
        "off_pod_phases:\n"
        "  - phase: P5 analysis\n"
        "    runs_on: vm\n"
        "    reads:\n"
        "      - path: eval_results/issue_77/ext.json\n"
        "        produced_by: external datasets\n"
        "      - path: eval_results/issue_77/sibling.json\n"
        "        produced_by: P4 (pod) of issue 55\n"
        "      - path: eval_results/issue_77/noprod.json\n"
        "        source: git-issue-branch\n"
        "```\n"
    )
    patterns, basenames = vci.extract_declared_outputs(text)
    assert patterns == [] and basenames == set()
    _write(repo, "eval_results/issue_77/ext.json")  # untracked
    fs = _findings(repo, text)
    fails = [f for f in fs if f.verdict == "fail"]
    assert [(f.reason, f.path) for f in fails] == [
        ("untracked-local-only", "eval_results/issue_77/ext.json")
    ]
    assert _main(repo, _plan(tmp_path, text)) == 1


def test_bare_name_foreign_issue_committed_warns_not_fails(repo: Path, tmp_path: Path) -> None:
    """The filed incident (#1935): a COMMITTED foreign file resolved via a
    bare-name citation + issue token demotes to ONE summarized warn (exit 0)
    on BOTH lanes — pre-#1935 the rsync lane FAILed it (in-ref -> downgrade)."""
    _write(repo, "eval_results/issue_55/pr.json")
    _commit_push(repo, "eval_results/issue_55/pr.json")
    text = "Gate A reads `pr.json` survival counts. Reuses #55's protocol."
    fs = _findings(repo, text)
    assert [(f.verdict, f.reason, f.path) for f in fs] == [
        ("warn", "bare-name-foreign-issue", "pr.json")
    ]
    assert "eval_results/issue_55/pr.json" in fs[0].detail  # auditable resolution list
    assert "cite the full repo-relative path" in fs[0].detail  # named remediation
    fs_rsync = _rsync_findings(repo, text)  # warns are untouched by the downgrade
    assert [(f.verdict, f.reason) for f in fs_rsync] == [("warn", "bare-name-foreign-issue")]
    assert _main(repo, _plan(tmp_path, text)) == 0
    assert _main(repo, _plan(tmp_path, text), extra=["--lane", "rsync"]) == 0


def test_bare_name_foreign_issue_untracked_warns(repo: Path, tmp_path: Path) -> None:
    """The #1902 baseline's 24-row variant: an UNTRACKED foreign resolution
    demotes to the same summarized warn (was fail/untracked-local-only)."""
    _write(repo, "eval_results/issue_55/pr.json")  # untracked
    text = "Gate A reads `pr.json` survival counts. Reuses #55's protocol."
    fs = _findings(repo, text)
    assert [(f.verdict, f.reason, f.path) for f in fs] == [
        ("warn", "bare-name-foreign-issue", "pr.json")
    ]
    assert _main(repo, _plan(tmp_path, text)) == 0


def test_bare_name_own_issue_undeclared_still_fails(repo: Path, tmp_path: Path) -> None:
    """Fail-loud pin (the #1434 direction, untouched): an own-issue bare-name
    resolution to an untracked local file with NO structured declaration keeps
    the fail verdict — never silently skipped or swallowed into a warn."""
    _write(repo, "eval_results/issue_77/manifest_i77.json")  # untracked
    text = "Stage the frozen mix (sha-verified against `manifest_i77.json` pins)."
    fs = _findings(repo, text)
    assert [(f.verdict, f.reason, f.path) for f in fs] == [
        ("fail", "untracked-local-only", "eval_results/issue_77/manifest_i77.json")
    ]
    assert _main(repo, _plan(tmp_path, text)) == 1


def test_bare_name_own_issue_declared_output_skips(repo: Path, tmp_path: Path) -> None:
    """A path-less `outputs:` declaration covers the bare name: the own-issue
    resolution skips as bare-name-declared-output, naming the declaration."""
    _write(repo, "eval_results/issue_77/report.json")  # untracked own output (re-gate shape)
    text = (
        "```yaml\nphase_outputs:\n  P1_pilot:\n    outputs: [report.json]\n```\n"
        "Gate A reads `report.json` before P2.\n"
    )
    fs = _findings(repo, text)
    skips = [f for f in fs if f.reason == "bare-name-declared-output"]
    assert [(f.verdict, f.path, f.channel) for f in skips] == [
        ("skip", "eval_results/issue_77/report.json", "B")
    ]
    assert "path-less declared name 'report.json'" in skips[0].detail
    assert not any(f.verdict in ("fail", "warn") for f in fs)
    assert _main(repo, _plan(tmp_path, text)) == 0


def test_declared_path_does_not_suppress_same_basename_elsewhere(
    repo: Path, tmp_path: Path
) -> None:
    """Critic Must-Fix 2's negative case: a PATH-BEARING declaration
    contributes a pattern, never a basename — a same-basename own-issue
    resolution at a NON-matching path (a prior round's copy reusing a
    per-round filename) keeps the FULL ladder and still FAILs."""
    _write(repo, "eval_results/issue_77/fits/grid.json")  # matches the declared pattern
    _write(repo, "eval_results/issue_77/old_round/grid.json")  # same basename, elsewhere
    text = (
        '```yaml\nprimary_deliverable:\n  - dv: "grid cells"\n'
        "    glob: eval_results/issue_77/fits/*.json\n```\n"
        "P5 verifies `grid.json` against the fit outputs.\n"
    )
    patterns, basenames = vci.extract_declared_outputs(text)
    assert "eval_results/issue_77/fits/*.json" in patterns
    assert basenames == set()  # path-bearing declaration contributes NO basename
    fs = _findings(repo, text)
    by_path = {f.path: (f.verdict, f.reason) for f in fs if f.channel == "B"}
    assert by_path["eval_results/issue_77/fits/grid.json"] == (
        "skip",
        "bare-name-declared-output",
    )
    assert by_path["eval_results/issue_77/old_round/grid.json"] == (
        "fail",
        "untracked-local-only",
    )
    assert _main(repo, _plan(tmp_path, text)) == 1


def test_bare_name_mixed_own_and_foreign_resolution(repo: Path, tmp_path: Path) -> None:
    """Critic Must-Fix 3: ONE bare name resolving to BOTH an own-issue
    undeclared untracked file AND a foreign committed file partitions PER
    RESOLVED PATH — own -> fail, foreign -> summarized warn — never
    all-or-nothing. Exit 1 (the own fail governs)."""
    _write(repo, "eval_results/issue_77/mix.json")  # own, untracked, undeclared
    _write(repo, "eval_results/issue_55/mix.json")
    _commit_push(repo, "eval_results/issue_55/mix.json")  # foreign, committed
    text = "Validated against `mix.json` pins. Extends #55."
    fs = _findings(repo, text)
    assert [(f.reason, f.path) for f in fs if f.verdict == "fail"] == [
        ("untracked-local-only", "eval_results/issue_77/mix.json")
    ]
    warns = [f for f in fs if f.verdict == "warn"]
    assert [(f.reason, f.path) for f in warns] == [("bare-name-foreign-issue", "mix.json")]
    assert "eval_results/issue_55/mix.json" in warns[0].detail
    assert _main(repo, _plan(tmp_path, text)) == 1


def test_channel_a_foreign_full_path_committed_still_fails_rsync(
    repo: Path, tmp_path: Path
) -> None:
    """The #1689 direction pinned: a FULL-PATH (Channel A) citation of ANOTHER
    issue's committed eval_results/ file on --lane rsync still FAILs
    rsync-lane-not-synced; the bare-name dedup row does not suppress it."""
    _write(repo, "eval_results/issue_55/f.json")
    _commit_push(repo, "eval_results/issue_55/f.json")
    text = "Consumes eval_results/issue_55/f.json (`f.json`) staged from #55."
    fs = _rsync_findings(repo, text)
    assert [(f.reason, f.path, f.channel) for f in fs if f.verdict == "fail"] == [
        ("rsync-lane-not-synced", "eval_results/issue_55/f.json", "A")
    ]
    # The bare name's resolution to the A-cited path stays a dedup skip row.
    assert [(f.verdict, f.reason) for f in fs if f.channel == "B"] == [
        ("skip", "deduped-channel-a")
    ]
    assert _main(repo, _plan(tmp_path, text), extra=["--lane", "rsync"]) == 1


def test_channel_a_own_declared_output_committed_skips_both_lanes(
    repo: Path, tmp_path: Path
) -> None:
    """The #1902 6-FAIL variant: an own-issue COMMITTED path matching a glob:
    declaration skips as planned-output-declared on clone AND rsync lanes
    (pre-#1935 the rsync lane downgraded the in-ref pass to FAIL on every
    post-run re-gate: follow-up rounds, relaunches)."""
    _write(repo, "eval_results/issue_77/fits/out.json")
    _commit_push(repo, "eval_results/issue_77/fits/out.json")
    text = (
        '```yaml\nprimary_deliverable:\n  - dv: "fit cells"\n'
        "    glob: eval_results/issue_77/fits/*.json\n```\n"
        "P5 reads eval_results/issue_77/fits/out.json for the figures.\n"
    )
    fs = _findings(repo, text)
    skips = [f for f in fs if f.reason == "planned-output-declared"]
    assert [(f.verdict, f.path, f.channel) for f in skips] == [
        ("skip", "eval_results/issue_77/fits/out.json", "A")
    ]
    assert "'eval_results/issue_77/fits/*.json'" in skips[0].detail  # names the declaration
    fs_rsync = _rsync_findings(repo, text)
    assert [
        (f.verdict, f.reason) for f in fs_rsync if f.path == "eval_results/issue_77/fits/out.json"
    ] == [("skip", "planned-output-declared")]
    assert _main(repo, _plan(tmp_path, text)) == 0
    assert _main(repo, _plan(tmp_path, text), extra=["--lane", "rsync"]) == 0


def test_declared_output_skip_never_applies_to_foreign_paths(repo: Path, tmp_path: Path) -> None:
    """A declared path-less basename that resolves under a FOREIGN issue dir
    takes the foreign-warn path, never the declared skip (the skip is
    own-issue-only on both channels)."""
    _write(repo, "eval_results/issue_55/rep.json")
    _commit_push(repo, "eval_results/issue_55/rep.json")
    text = (
        "```yaml\nphase_outputs:\n  P2_gen:\n    outputs: [rep.json]\n```\n"
        "Compares `rep.json` against #55's copy.\n"
    )
    fs = _findings(repo, text)
    assert [(f.verdict, f.reason, f.path) for f in fs] == [
        ("warn", "bare-name-foreign-issue", "rep.json")
    ]
    assert not any(f.reason in ("bare-name-declared-output", "planned-output-declared") for f in fs)
    assert _main(repo, _plan(tmp_path, text)) == 0


# ---------------------------------------------------------------------------
# #1982: HF-staged / multi-resolution downgrade — false-fails on `untracked-
# local-only` when the plan's citation is HF-repo-side (#1979) OR another
# resolution is already committed + in-ref (#1739).
# ---------------------------------------------------------------------------


def test_bare_name_hf_staged_short_prefix_warns_not_fails(repo: Path, tmp_path: Path) -> None:
    """#1979: a plan that cites `<name>.jsonl` under the SHORT HF form
    `issue<N>_<slug>/…` should NOT fail on a coincidental untracked VM-local
    mirror — the plan clearly stages the file via HF, so a local-only mirror
    is not a repro-blocker.
    """
    # Untracked local mirror under eval_results/ (bare-name resolver globs
    # eval_results/ + ood_eval_results/ only, not data/):
    _write(repo, "eval_results/issue_77/pool.jsonl")
    text = "Stage `pool.jsonl` from `issue1434_writingstyle/ws-po-c/mix/pool.jsonl` before phase 2."
    fs = _findings(repo, text)
    # No `fail` finding — the local-only mirror downgrades to a WARN.
    assert not any(f.verdict == "fail" for f in fs), [(f.verdict, f.reason, f.path) for f in fs]
    # And exactly one hf-staged WARN pointing at the local path.
    hf_warns = [f for f in fs if f.reason == "hf-staged"]
    assert len(hf_warns) == 1, [(f.verdict, f.reason, f.path) for f in fs]
    assert hf_warns[0].path == "eval_results/issue_77/pool.jsonl"
    assert hf_warns[0].verdict == "warn"
    assert _main(repo, _plan(tmp_path, text)) == 0


def test_bare_name_hf_staged_full_repo_prefix_warns_not_fails(repo: Path, tmp_path: Path) -> None:
    """#1979 sibling: the FULL data-repo prefix form
    `explore-persona-space-data/issue<N>_<slug>/…` also demotes an untracked
    local mirror to a WARN.
    """
    _write(repo, "eval_results/issue_77/pool.jsonl")
    text = (
        "Reuse `pool.jsonl` from "
        "`superkaiba1/explore-persona-space-data/issue1434_writingstyle/mix/pool.jsonl`."
    )
    fs = _findings(repo, text)
    assert not any(f.verdict == "fail" for f in fs), [(f.verdict, f.reason, f.path) for f in fs]
    assert any(f.reason == "hf-staged" for f in fs)
    assert _main(repo, _plan(tmp_path, text)) == 0


def test_bare_name_multi_resolution_downgrade(repo: Path, tmp_path: Path) -> None:
    """#1739: a bare-name citation with multiple own-issue resolutions where
    ONE is committed+in-ref and OTHERS are untracked-local-only siblings
    demotes the untracked siblings to `duplicate-resolution` WARNs — the
    committed sibling proves the plan can reproduce from it.
    """
    # Committed + in-ref resolution under eval_results/ (this passes as `in-ref`).
    _write(repo, "eval_results/issue_77/results.json")
    _commit_push(repo, "eval_results/issue_77/results.json")
    # Untracked sibling with the SAME basename under ood_eval_results/ (also
    # globbed by the bare-name resolver — the resolver walks eval_results/ +
    # ood_eval_results/ under every in-scope issue).
    _write(repo, "ood_eval_results/issue_77/results.json")
    text = "Ingest `results.json` for the aggregation phase."
    fs = _findings(repo, text)
    # No FAIL — the untracked sibling downgrades.
    assert not any(f.verdict == "fail" for f in fs), [(f.verdict, f.reason, f.path) for f in fs]
    # One `pass`/`in-ref` for the committed resolution.
    assert any(f.verdict == "pass" and f.reason == "in-ref" for f in fs)
    # At least one `duplicate-resolution` WARN for the untracked sibling.
    dup_warns = [f for f in fs if f.reason == "duplicate-resolution"]
    assert dup_warns, [(f.verdict, f.reason, f.path) for f in fs]
    assert all(f.verdict == "warn" for f in dup_warns)
    assert _main(repo, _plan(tmp_path, text)) == 0


def test_bare_name_untracked_still_fails_when_no_hf_citation_and_single_resolution(
    repo: Path, tmp_path: Path
) -> None:
    """#1434 protection preserved: a bare-name citation with EXACTLY one
    untracked-local-only own-issue resolution and NO HF citation must STILL
    fail — the downgrade only fires when evidence (HF citation OR in-ref
    sibling) proves the file is reproducible.
    """
    _write(repo, "eval_results/issue_77/needed.jsonl")  # untracked
    text = "Ingest `needed.jsonl` produced by the parent round."
    fs = _findings(repo, text)
    # The #1434 protection: fail-loud stays.
    fails = [f for f in fs if f.verdict == "fail" and f.reason == "untracked-local-only"]
    assert fails, [(f.verdict, f.reason, f.path) for f in fs]
    assert _main(repo, _plan(tmp_path, text)) == 1


def test_issue_underscore_git_path_not_matched_by_hf_regex() -> None:
    """The `_HF_CITED_RE` alt-family that starts `issue<N>_<slug>` must NOT
    eat a repo-relative git path `eval_results/issue_<N>/…` that Channel-A
    handles. Regression pin against the review's Q1 boundary concern.
    """
    # A pure Channel-A citation — nothing under an HF prefix.
    text = "Reads eval_results/issue_1434/results.jsonl for phase 2."
    hf_names = vci.extract_hf_cited_basenames(text)
    # Absolutely no HF hit — Channel-A owns this citation.
    assert hf_names == set(), hf_names
    # Sanity: Channel-A still extracts it.
    assert vci.extract_candidate_paths(text) == [
        {"path": "eval_results/issue_1434/results.jsonl", "skip_reason": None}
    ]
