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

import contextlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from tests.issue_skill_source import issue_skill_text

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
    # #2263: pin the invoking cwd to the fixture repo (branch `main`) so the
    # rung-3 worktree inference reads a DETERMINISTIC cwd regardless of where
    # pytest itself runs (an issue-scoped worktree cwd would otherwise leak in).
    with contextlib.chdir(repo):
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
    assert vci.resolve_check_ref(repo, 77, fetch=False, invoke_cwd=repo).ref == "origin/issue-77"
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
    # #2263 test 6: the two legacy default rows keep their resolved ref and
    # gain a source tag (R4 unchanged-rows contract).
    assert vci.resolve_check_ref(repo, 77, fetch=False, invoke_cwd=repo) == (
        "origin/main",
        "origin-main-default",
    )
    _git(repo, "branch", "issue-77")
    _git(repo, "push", "-q", "origin", "issue-77")
    assert vci.resolve_check_ref(repo, 77, fetch=False, invoke_cwd=repo) == (
        "origin/issue-77",
        "bare-issue-branch-default",
    )


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
    skill = issue_skill_text()
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


def test_rsync_lane_tracked_data_outside_sft_covered_no_downgrade(
    repo: Path, tmp_path: Path
) -> None:
    """#2212: a committed data/ citation OUTSIDE data/sft PASSes under --lane
    rsync with no --extra-sync-path — the lane now derives its data/ include
    entries from the git index (every tracked component), and the static gate
    covers the whole data/ root (RSYNC_DATA_INCLUDE_ROOT, a documented
    over-approximation that cannot false-PASS: the gate only ever evaluates
    COMMITTED citations, all of which are in the derived set by
    construction). Pre-#2212 this exact class (the #2203 crash path,
    data/assistant_axis/) FAILed rsync-lane-not-synced."""
    _write(repo, "data/assistant_axis/roles.json")
    _commit_push(repo, "data/assistant_axis/roles.json")
    text = "Reads data/assistant_axis/roles.json for the persona panel."
    fs = _rsync_findings(repo, text)
    assert [(f.verdict, f.reason) for f in fs] == [("pass", "in-ref")]
    assert _main(repo, _plan(tmp_path, text), extra=["--lane", "rsync"]) == 0


def test_rsync_lane_data_dl_and_store_citations_downgrade(repo: Path, tmp_path: Path) -> None:
    """#2212 change B x #1915: a committed citation under a ``*_dl``/``store``
    dir is inside the data/ cover but the new receiver-protection excludes
    match at every depth -> FAIL(rsync-lane-not-synced) via the EXCLUDE
    branch, naming the pattern, with --extra-sync-path as the structural
    remedy (the extra rsync applies no excludes) — the matcher's deliberate
    cheap-false-FAIL direction: committed inputs should not live under cache
    /store conventions anyway (the collision invariant in
    test_slurm_backend_render.py pins that none do today)."""
    rel_dl = "data/issue_9/hf_dl/cached_rows.json"
    rel_store = "data/issue_9/store/rows.jsonl"
    for rel in (rel_dl, rel_store):
        _write(repo, rel)
        _commit_push(repo, rel)
    assert vci.rsync_excluded(rel_dl) == "*_dl/"
    assert vci.rsync_excluded(rel_store) == "store/"

    text = f"Consumes {rel_dl} and {rel_store} at stage time."
    fs = _rsync_findings(repo, text)
    assert [(f.verdict, f.reason) for f in fs] == [
        ("fail", "rsync-lane-not-synced"),
        ("fail", "rsync-lane-not-synced"),
    ]
    details = " | ".join(f.detail for f in fs)
    assert "'*_dl/'" in details
    assert "'store/'" in details
    assert "--extra-sync-path" in details

    # The structural remedy restores PASS (the #1835 extra rsync is
    # exclude-free), end-to-end through the CLI.
    extras = ["data/issue_9/hf_dl", "data/issue_9/store"]
    fs2 = _rsync_findings(repo, text, extras=extras)
    assert [(f.verdict, f.reason) for f in fs2] == [("pass", "in-ref"), ("pass", "in-ref")]
    assert _main(repo, _plan(tmp_path, text), extra=["--lane", "rsync"]) == 1
    rc = _main(
        repo,
        _plan(tmp_path, text),
        extra=[
            "--lane",
            "rsync",
            "--extra-sync-path",
            "data/issue_9/hf_dl",
            "--extra-sync-path",
            "data/issue_9/store",
        ],
    )
    assert rc == 0


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

    skill = issue_skill_text()
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


# ---------------------------------------------------------------------------
# #1995 — widen `_PATH_RE` to the remaining include-tree prefixes
# (`tests|scripts|configs`). Sibling of #1915 (extraction ↔ downgrade parity):
# #1915 wired the include-tree + exclude-name interaction inside
# `apply_rsync_lane_downgrade` but only for hand-built Findings; before this
# round the Channel-A regex would never fire on `tests/…` / `scripts/…` /
# `configs/…` plan text, so the downgrade was structurally unreachable from a
# real plan.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prefix,rel,detail",
    [
        ("tests", "tests/fixtures/manifest.json", "tests-prefix citation"),
        ("scripts", "scripts/issue_1995/probe.jsonl", "scripts-prefix citation"),
        ("configs", "configs/training/marker_lr.yaml", "configs-prefix citation"),
    ],
)
def test_path_re_matches_new_prefixes(prefix: str, rel: str, detail: str) -> None:
    """Each new-prefix citation extracts as Channel-A and honors the existing
    skip rungs (glob-or-template / dir / no-ext) exactly as the pre-existing
    prefixes do.

    #1995 acceptance criterion 1 + 2: `_PATH_RE` widens to include-tree
    prefixes AND the extraction-side skip rungs remain byte-identical (no
    regression on rung behavior for new prefixes)."""
    # (a) extracts a straight full-prefix citation
    cands = vci.extract_candidate_paths(f"Reuses {rel} as the phase-1 input.")
    assert [c["path"] for c in cands] == [rel], detail
    assert cands[0]["skip_reason"] is None
    # (b) trailing punctuation stripped (same rule as the old prefixes)
    cands2 = vci.extract_candidate_paths(f"(see {rel}).")
    assert [c["path"] for c in cands2] == [rel]
    # (c) glob-or-template skip rung fires on wildcard chars in the path
    glob_rel = (
        rel.replace(".json", "*.json").replace(".jsonl", "*.jsonl").replace(".yaml", "*.yaml")
    )
    if glob_rel != rel:
        globs = vci.extract_candidate_paths(f"Cells at {glob_rel}")
        assert [c["skip_reason"] for c in globs] == ["glob-or-template"]
    # (d) dir skip rung fires on trailing slash
    dir_rel = rel.rsplit("/", 1)[0] + "/"
    dirs = vci.extract_candidate_paths(f"Writes outputs under {dir_rel}")
    assert [c["skip_reason"] for c in dirs] == ["dir"]
    # (e) no-ext skip rung fires on an extension-less basename
    noext_rel = rel.rsplit("/", 1)[0] + "/README"
    noexts = vci.extract_candidate_paths(f"Reads {noext_rel} for the config.")
    assert [c["skip_reason"] for c in noexts] == ["no-ext"]
    # Also confirms the prefix membership: the head SEG must be the new prefix.
    assert cands[0]["path"].split("/", 1)[0] == prefix


def test_path_re_include_tree_and_exclude_name_interaction(repo: Path, tmp_path: Path) -> None:
    """The load-bearing #1915 sibling case: `tests/fixtures/eval_results/a.json`
    now extracts (Channel A) and classifies `in-ref` under `--lane clone`, while
    under `--lane rsync` `apply_rsync_lane_downgrade` downgrades it to
    `FAIL(rsync-lane-not-synced)` — the include-tree membership (./tests) is
    necessary but NOT sufficient because rsync's slash-free `eval_results/`
    exclude matches at every depth.

    Pre-fix (round 1 of #1915): #1915's classify + downgrade ladder already
    handled this path via hand-constructed Findings; the gap #1995 closes is
    that plan-text Channel-A extraction never saw it, so a real plan citation
    could never reach the ladder. This test exercises the full path end-to-end
    (extract -> classify -> apply_rsync_lane_downgrade)."""
    rel = "tests/fixtures/eval_results/a.json"
    _write(repo, rel)
    _commit_push(repo, rel)

    text = f"Reuses {rel} as the parity anchor."
    # Extraction: the widened regex fires (this is the primary #1995 gate).
    assert vci.extract_candidate_paths(text) == [{"path": rel, "skip_reason": None}]

    # Clone lane: in-ref pass, exit 0 (byte-identical to what #1915 unlocked for
    # hand-built findings — now reachable from real plan text).
    fs_clone = _findings(repo, text)
    assert [(f.verdict, f.reason) for f in fs_clone] == [("pass", "in-ref")]
    assert _main(repo, _plan(tmp_path, text)) == 0

    # Rsync lane: nested-excluded pattern kicks in (`eval_results/` slash-free
    # exclude matches at every path depth) -> FAIL(rsync-lane-not-synced),
    # exit 1, `--extra-sync-path` named as the remediation.
    fs_rsync = _rsync_findings(repo, text)
    assert [(f.verdict, f.reason) for f in fs_rsync] == [("fail", "rsync-lane-not-synced")]
    assert "'eval_results/'" in fs_rsync[0].detail  # #1915 detail wording preserved
    assert "--extra-sync-path" in fs_rsync[0].detail
    assert _main(repo, _plan(tmp_path, text), extra=["--lane", "rsync"]) == 1

    # And --extra-sync-path covering the nested-excluded path restores PASS
    # (the exclude-free `build_extra_rsync_command` semantics).
    fs_rsync_ok = _rsync_findings(repo, text, extras=["tests/fixtures/eval_results"])
    assert [(f.verdict, f.reason) for f in fs_rsync_ok] == [("pass", "in-ref")]


def test_path_re_old_prefixes_unchanged(repo: Path, tmp_path: Path) -> None:
    """Regression pin: the three pre-widening prefixes (`eval_results`,
    `ood_eval_results`, `data`) produce byte-identical extraction rows AND
    byte-identical `Finding` verdicts + reasons.

    #1995 acceptance criterion 5(c) + kill-criterion baseline (`≤ 1 FAIL per
    ~50 plans on a clean-tree corpus for today's channel-A ladder`) — this pin
    guarantees the widening cannot silently regress the ORIGINAL three prefixes'
    behavior even if a future regex refactor accidentally changes alternation
    order."""
    # (a) Extraction is byte-identical to the pre-widening surface — same
    # dict shape, same skip_reason semantics, one row per prefix.
    for rel, expected_skip in (
        ("eval_results/issue_12/a.json", None),
        ("ood_eval_results/issue_9/f.json", None),
        ("data/sft/train.jsonl", None),
        ("eval_results/issue_12/cells/", "dir"),
        ("data/issue_5/nested/*.json", "glob-or-template"),
        ("ood_eval_results/issue_9/README", "no-ext"),
    ):
        text = f"Consumes {rel} at stage time."
        cands = vci.extract_candidate_paths(text)
        assert cands == [{"path": rel, "skip_reason": expected_skip}], (rel, cands)

    # (b) The classify ladder verdicts stay identical — a committed+pushed
    # eval_results/ path still classifies `pass` / `in-ref` on --lane clone
    # and the CLI end-to-end still returns 0.
    _write(repo, "eval_results/issue_12/a.json")
    _commit_push(repo, "eval_results/issue_12/a.json")
    text = "Reuses eval_results/issue_12/a.json as the stage-1 input."
    fs = _findings(repo, text)
    assert [(f.verdict, f.reason) for f in fs] == [("pass", "in-ref")]
    assert _main(repo, _plan(tmp_path, text)) == 0


def test_path_re_lookbehind_still_excludes_hf_and_urls() -> None:
    """Negative regression pin: a `\\w` / `/` / `-` / `.` immediately BEFORE a
    NEW-prefix hit still bars the match — mirroring the existing HF
    `explore-persona-space-data/…` test. Guards against a new-prefix citation
    being wrongly matched inside an HF-repo path, a URL segment, or a
    dotted-package identifier."""
    text = (
        # An HF path with `tests/` as an internal segment.
        "Raw fixtures at "
        "superkaiba1/explore-persona-space-data/issue77_slug/tests/fixtures/a.json "
        # A URL with `scripts/` as an internal segment.
        "and https://example.com/repo/scripts/foo.py "
        # A path-like `-`-anchored segment (extractor's lookbehind excludes `-`).
        "and my-scripts/utility.py for internal reference. "
        # A dotted identifier — mirrors the HF-URL exclusion via the `.` in
        # the lookbehind (`configs.training` would be a Python module path).
        "See settings.configs/training/marker_lr.yaml for the recipe."
    )
    # None of these should match — every hit is inside an excluded lookbehind.
    assert vci.extract_candidate_paths(text) == []
    # Extraction sanity: bare-name hits (Channel B) also don't spuriously fire
    # on the HF-embedded path.
    assert "fixtures" not in {n.split(".")[0] for n in vci.extract_bare_names(text)}


# ---------------------------------------------------------------------------
# #2263 — check-ref resolution ladder (resolve_check_ref -> ResolvedRef),
# derive_local_branch, derive_worktree_repo_branch / --print-repo-branch, the
# env sanitization, and the Step 6 shared-resolver skill pins.
# ---------------------------------------------------------------------------


def _branch_push(repo: Path, branch: str, base: str = "main") -> None:
    """Create `branch` at `base` and push it to the local bare origin."""
    _git(repo, "branch", branch, base)
    _git(repo, "push", "-q", "origin", branch)


def _foreign_repo(base: Path, name: str = "foreign", branch: str = "main") -> Path:
    """An INDEPENDENT throwaway repo (own git dir, own bare origin) at base/name."""
    r = base / name
    r.mkdir()
    _git(r, "init", "-q", "-b", branch)
    _git(r, "config", "user.email", "test@example.com")
    _git(r, "config", "user.name", "Test")
    _git(r, "config", "commit.gpgsign", "false")
    bare = base / f"{name}-origin.git"
    subprocess.run(
        ["git", "init", "-q", "--bare", str(bare)], check=True, capture_output=True, text=True
    )
    _git(r, "remote", "add", "origin", str(bare))
    (r / "seed.txt").write_text("seed\n")
    _git(r, "add", "seed.txt")
    _git(r, "commit", "-q", "-m", "seed")
    _git(r, "push", "-q", "-u", "origin", branch)
    return r


def test_repo_branch_flag_resolves_suffixed_over_stale_bare(repo: Path) -> None:
    _branch_push(repo, "issue-77")
    _branch_push(repo, "issue-77-full")
    got = vci.resolve_check_ref(repo, 77, fetch=False, repo_branch="issue-77-full")
    assert got == ("origin/issue-77-full", "repo-branch-flag")


def test_repo_branch_missing_on_origin_raises(repo: Path) -> None:
    with pytest.raises(vci.CheckRefResolutionError) as ei:
        vci.resolve_check_ref(repo, 77, fetch=False, repo_branch="issue-77-nope")
    assert "origin/issue-77-nope" in str(ei.value)


def test_worktree_inference_resolves_suffixed(repo: Path, tmp_path: Path) -> None:
    _branch_push(repo, "issue-77")  # stale bare sibling also present
    wt = tmp_path / "wt-full"
    _git(repo, "worktree", "add", "-q", str(wt), "-b", "issue-77-full")
    _git(repo, "push", "-q", "origin", "issue-77-full")
    got = vci.resolve_check_ref(repo, 77, fetch=False, invoke_cwd=wt)
    assert got == ("origin/issue-77-full", "worktree-branch")


def test_worktree_inference_ignores_foreign_and_digit_prefix_branches(
    repo: Path, tmp_path: Path
) -> None:
    _branch_push(repo, "issue-77")  # bare-only remote state
    for branch in ("issue-88-x", "issue-771-x"):
        wt = tmp_path / f"wt-{branch}"
        _git(repo, "worktree", "add", "-q", str(wt), "-b", branch)
        got = vci.resolve_check_ref(repo, 77, fetch=False, invoke_cwd=wt)
        assert got == ("origin/issue-77", "bare-issue-branch-default")


def test_worktree_inference_unpushed_branch_raises(repo: Path, tmp_path: Path) -> None:
    wt = tmp_path / "wt-local"
    _git(repo, "worktree", "add", "-q", str(wt), "-b", "issue-77-local")  # never pushed
    with pytest.raises(vci.CheckRefResolutionError) as ei:
        vci.resolve_check_ref(repo, 77, fetch=False, invoke_cwd=wt)
    msg = str(ei.value)
    assert "issue-77-local" in msg and "push the branch" in msg


def test_default_refuses_when_suffixed_exists(repo: Path) -> None:
    # (a) suffixed-only.
    _branch_push(repo, "issue-77-full")
    with pytest.raises(vci.CheckRefResolutionError) as ei:
        vci.resolve_check_ref(repo, 77, fetch=False, invoke_cwd=repo)
    msg = str(ei.value)
    assert "origin/issue-77-full" in msg
    assert msg.index("--repo-branch") < msg.index("--ref")  # remedy ordering
    # (b) bare + suffixed: EVERY candidate named, bare included.
    _branch_push(repo, "issue-77")
    with pytest.raises(vci.CheckRefResolutionError) as ei2:
        vci.resolve_check_ref(repo, 77, fetch=False, invoke_cwd=repo)
    msg2 = str(ei2.value)
    assert "origin/issue-77-full" in msg2 and "origin/issue-77" in msg2


def test_fetch_true_glob_prunes_deleted_suffixed_branch(repo: Path, tmp_path: Path) -> None:
    _branch_push(repo, "issue-77-old")
    assert vci.ref_exists(repo, "origin/issue-77-old")  # stale tracking ref armed
    # Delete the branch INSIDE the bare origin — only the fetch=True --prune
    # glob can clear the stale local remote-tracking ref.
    _git(tmp_path / "origin.git", "update-ref", "-d", "refs/heads/issue-77-old")
    got = vci.resolve_check_ref(repo, 77, fetch=True, invoke_cwd=repo)
    assert got == ("origin/main", "origin-main-default")
    assert not vci.ref_exists(repo, "origin/issue-77-old")


def test_cli_ref_and_repo_branch_mutually_exclusive(repo: Path, tmp_path: Path) -> None:
    plan = _plan(tmp_path, "no citations")
    with pytest.raises(SystemExit) as ei:
        _main(repo, plan, extra=["--ref", "origin/main", "--repo-branch", "main"])
    assert ei.value.code == 2


def test_cli_incident_shape_end_to_end(repo: Path, tmp_path: Path) -> None:
    """The #1336 shape: the input lives ONLY on the suffixed dispatch branch."""
    _branch_push(repo, "issue-77")  # stale bare branch, no file
    _git(repo, "checkout", "-q", "-b", "issue-77-full")
    _write(repo, "eval_results/issue_77/f.json")
    _git(repo, "add", "eval_results/issue_77/f.json")
    _git(repo, "commit", "-q", "-m", "add f on suffixed branch")
    _git(repo, "push", "-q", "origin", "issue-77-full")
    _git(repo, "checkout", "-q", "main")
    _write(repo, "eval_results/issue_77/f.json")  # the VM-local copy
    plan = _plan(tmp_path, "Consumes eval_results/issue_77/f.json from the prior round.")
    # (a) mirrored --repo-branch grades the RIGHT tree -> PASS.
    assert _main(repo, plan, extra=["--repo-branch", "issue-77-full"]) == 0
    # (b) neither flag -> the rung-4 ambiguity refusal, exit 2.
    assert _main(repo, plan) == 2
    # (c) --ref pinning the STALE bare branch still wins and still FAILs —
    # the gate is not weakened; the wrong-tree grade is deliberate.
    assert _main(repo, plan, extra=["--ref", "origin/issue-77"]) == 1


def test_cli_refusal_stderr_names_candidates_and_remedy(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _branch_push(repo, "issue-77")
    _branch_push(repo, "issue-77-full")
    plan = _plan(tmp_path, "no citations")
    assert _main(repo, plan) == 2
    err = capsys.readouterr().err
    assert "origin/issue-77" in err and "origin/issue-77-full" in err
    assert "--repo-branch" in err


def test_cli_verdict_lines_carry_ref_every_severity(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write(repo, "eval_results/issue_12/a.json")
    _commit_push(repo, "eval_results/issue_12/a.json")  # PASS (in-ref)
    _write(repo, "data/issue_77/x.json")  # WARN (data-local-only)
    _write(repo, "eval_results/issue_12/m.json")  # FAIL (untracked-local-only)
    text = (
        "Reuses eval_results/issue_12/a.json and consumes data/issue_77/x.json "
        "and reads eval_results/issue_12/m.json "
        "and writes eval_results/issue_77/out.json now."  # SKIP (planned-output)
    )
    plan = _plan(tmp_path, text)
    assert _main(repo, plan) == 1
    out = capsys.readouterr().out
    rows = [ln for ln in out.splitlines() if ln.startswith("[")]
    assert {ln[1:5].strip() for ln in rows} == {"PASS", "WARN", "FAIL", "SKIP"}
    assert rows and all(" ref=origin/" in ln for ln in rows)
    assert "ref source:" in out


def test_cli_json_carries_check_ref_source(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _branch_push(repo, "issue-77-full")
    plan = _plan(tmp_path, "no citations")
    assert _main(repo, plan, extra=["--repo-branch", "issue-77-full", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["check_ref"] == "origin/issue-77-full"
    assert payload["check_ref_source"] == "repo-branch-flag"


def test_committed_unpushed_on_suffixed_branch_detected(repo: Path, tmp_path: Path) -> None:
    _branch_push(repo, "issue-77-full")  # origin tip WITHOUT the file
    _git(repo, "checkout", "-q", "issue-77-full")
    _write(repo, "eval_results/issue_77/inp.json")
    _git(repo, "add", "eval_results/issue_77/inp.json")
    _git(repo, "commit", "-q", "-m", "add input on suffixed branch")  # NOT pushed
    _git(repo, "checkout", "-q", "main")
    plan = _plan(tmp_path, "Consumes eval_results/issue_77/inp.json from the prior round.")
    assert _main(repo, plan, extra=["--repo-branch", "issue-77-full"]) == 1
    fs = vci.run_check(
        "Consumes eval_results/issue_77/inp.json now.",
        repo_root=repo,
        issue=77,
        check_ref="origin/issue-77-full",
        local_branch="issue-77-full",
    )
    assert [(f.verdict, f.reason) for f in fs] == [("fail", "committed-unpushed")]
    assert "issue-77-full" in fs[0].detail


def test_step6_repo_branch_shared_resolver_pin() -> None:
    """#2263 durability pin: BOTH Step 6 fences call the ONE shared resolver.

    Byte-identical unindented preludes (assignment + ${REPO_BRANCH:?} guard)
    in the 6a.5 gate fence and the 6b launch fence; zero hardcoded
    `--repo-branch issue-<N>` literals survive anywhere in the composed spec.
    """
    text = issue_skill_text()
    expected_assign = (
        'REPO_BRANCH="$(uv run python scripts/verify_carryover_inputs.py '
        '--print-repo-branch --issue <N>)"   '
        "# the ONE shared issue-scoped resolver — re-derived per fence; "
        "never the cwd branch (#2263)"
    )
    assign_lines = re.findall(r"(?m)^REPO_BRANCH=.*$", text)
    assert assign_lines == [expected_assign, expected_assign]
    assert len(re.findall(r"(?<![$\w])REPO_BRANCH=", text)) == 2
    guard_lines = re.findall(r'(?m)^: "\$\{REPO_BRANCH:\?.*$', text)
    assert len(guard_lines) == 2
    assert len(set(guard_lines)) == 1  # byte-identical guards
    assert (
        'uv run python scripts/verify_carryover_inputs.py --plan "$PLAN_PATH" '
        '--issue <N> --repo-branch "$REPO_BRANCH"'
    ) in text
    assert '--issue <N> --intent "$INTENT" --repo-branch "$REPO_BRANCH" \\' in text
    assert 'git push origin "$REPO_BRANCH"' in text  # the push remediation
    assert '--repo-branch "issue-<N>"' not in text
    assert "--repo-branch issue-<N>" not in text


def test_step6_repo_branch_shared_resolver_fresh_shell_execution(
    repo: Path, tmp_path: Path
) -> None:
    """#2263 execution pin: the fence prelude RUNS, verbatim from the spec.

    Extraction failure IS test failure (never a silent skip): the prelude
    lines are pulled from the composed spec by the same regexes as the text
    pin, substituted ONLY on `<N>` and the `uv run python <script>` prefix,
    and executed under `bash -c` in a fresh shell.
    """
    text = issue_skill_text()
    assigns = re.findall(r"(?m)^REPO_BRANCH=.*$", text)
    guards = re.findall(r'(?m)^: "\$\{REPO_BRANCH:\?.*$', text)
    assert len(assigns) == 2 and len(guards) == 2, "prelude extraction failed"
    assign = (
        assigns[0]
        .replace("<N>", "77")
        .replace(
            "uv run python scripts/verify_carryover_inputs.py",
            f"{sys.executable} {_SCRIPT}",
        )
    )
    guard = guards[0].replace("<N>", "77")
    script = assign + "\n" + guard + "\n" + 'echo "REACHED-$REPO_BRANCH"'
    # Arm 1: repo-root shape (branch main, no issue worktree) -> the guard
    # ABORTS the fence; the dispatch line is never reached.
    proc = subprocess.run(["bash", "-c", script], cwd=repo, capture_output=True, text=True)
    assert proc.returncode != 0
    assert "REACHED" not in proc.stdout
    assert "ERROR:" in proc.stderr  # the resolver's refusal reached the shell
    # Arm 2: ONE live issue worktree -> the fence proceeds with its branch.
    _git(repo, "worktree", "add", "-q", str(tmp_path / "wt"), "-b", "issue-77-full")
    proc2 = subprocess.run(["bash", "-c", script], cwd=repo, capture_output=True, text=True)
    assert proc2.returncode == 0, proc2.stderr
    assert proc2.stdout.strip().endswith("REACHED-issue-77-full")


def test_worktree_inference_repository_bound(repo: Path, tmp_path: Path) -> None:
    """Guard (b): a FOREIGN repository cwd can never steer the check ref."""
    foreign = _foreign_repo(tmp_path, name="foreign", branch="issue-77-full")
    # (c) bare-only fall-through FIRST: the foreign cwd is rejected by the
    # repository-binding guard and rung 4 resolves the bare default.
    _branch_push(repo, "issue-77")
    got = vci.resolve_check_ref(repo, 77, fetch=False, invoke_cwd=foreign)
    assert got == ("origin/issue-77", "bare-issue-branch-default")
    # (a) suffix present on the TARGET: the same foreign cwd now hits the
    # rung-4 candidates refusal — NOT rung 3's unpushed message, and never a
    # silent worktree-branch pick of the foreign checkout's branch.
    _branch_push(repo, "issue-77-full")
    with pytest.raises(vci.CheckRefResolutionError) as ei:
        vci.resolve_check_ref(repo, 77, fetch=False, invoke_cwd=foreign)
    msg = str(ei.value)
    assert "candidates" in msg and "origin/issue-77-full" in msg
    assert "materialize nothing" not in msg  # not the rung-3 unpushed refusal
    # (b) foreign repo NESTED inside the target tree — same refusal.
    nested = _foreign_repo(repo, name="nested_foreign", branch="issue-77-x")
    with pytest.raises(vci.CheckRefResolutionError) as ei2:
        vci.resolve_check_ref(repo, 77, fetch=False, invoke_cwd=nested)
    assert "candidates" in str(ei2.value)


def test_repo_branch_flag_beats_worktree_inference(repo: Path, tmp_path: Path) -> None:
    _branch_push(repo, "issue-77-full")
    wt = tmp_path / "wt-other"
    _git(repo, "worktree", "add", "-q", str(wt), "-b", "issue-77-other")
    _git(repo, "push", "-q", "origin", "issue-77-other")
    got = vci.resolve_check_ref(repo, 77, fetch=False, repo_branch="issue-77-full", invoke_cwd=wt)
    assert got == ("origin/issue-77-full", "repo-branch-flag")
    got_main = vci.resolve_check_ref(repo, 77, fetch=False, repo_branch="main", invoke_cwd=wt)
    assert got_main == ("origin/main", "repo-branch-flag")


_MATRIX_CELLS = [
    ("bare-only", ("repo-branch", "issue-77"), ("origin/issue-77", "repo-branch-flag")),
    ("bare-only", ("repo-branch", "main"), ("origin/main", "repo-branch-flag")),
    ("bare-only", ("repo-branch", "issue-77-absent"), "raise"),
    ("bare-only", ("worktree-pushed", "issue-77"), ("origin/issue-77", "worktree-branch")),
    ("bare-only", ("worktree-unpushed", "issue-77-local"), "raise"),
    ("none", ("default", None), ("origin/main", "origin-main-default")),
    ("none", ("repo-branch", "main"), ("origin/main", "repo-branch-flag")),
    ("none", ("repo-branch", "issue-77-full"), "raise"),
    ("none", ("worktree-unpushed", "issue-77-local"), "raise"),
    ("suffix-only", ("default", None), "raise"),
    (
        "suffix-only",
        ("repo-branch", "issue-77-full"),
        ("origin/issue-77-full", "repo-branch-flag"),
    ),
    ("suffix-only", ("repo-branch", "main"), ("origin/main", "repo-branch-flag")),
    (
        "suffix-only",
        ("worktree-pushed", "issue-77-full"),
        ("origin/issue-77-full", "worktree-branch"),
    ),
    ("bare+suffix", ("default", None), "raise"),
    ("bare+suffix", ("repo-branch", "issue-77"), ("origin/issue-77", "repo-branch-flag")),
    (
        "bare+suffix",
        ("repo-branch", "issue-77-full"),
        ("origin/issue-77-full", "repo-branch-flag"),
    ),
    (
        "bare+suffix",
        ("worktree-pushed", "issue-77-full"),
        ("origin/issue-77-full", "worktree-branch"),
    ),
    ("bare+suffix", ("main-cwd", None), "raise"),
]


@pytest.mark.parametrize(("remote_state", "invocation", "expected"), _MATRIX_CELLS)
def test_matrix_feasible_cells(
    repo: Path, tmp_path: Path, remote_state: str, invocation: tuple, expected
) -> None:
    """#2263 feasible-cell matrix: remote state x invocation -> (ref, source)."""
    if remote_state in ("bare-only", "bare+suffix"):
        _branch_push(repo, "issue-77")
    if remote_state in ("suffix-only", "bare+suffix"):
        _branch_push(repo, "issue-77-full")
    kind, value = invocation
    kwargs: dict = {"fetch": False, "invoke_cwd": repo}
    if kind == "repo-branch":
        kwargs["repo_branch"] = value
    elif kind == "worktree-pushed":
        wt = tmp_path / "wt"
        if vci.ref_exists(repo, f"refs/heads/{value}"):
            _git(repo, "worktree", "add", "-q", str(wt), value)
        else:
            _git(repo, "worktree", "add", "-q", str(wt), "-b", value)
            _git(repo, "push", "-q", "origin", value)
        kwargs["invoke_cwd"] = wt
    elif kind == "worktree-unpushed":
        wt = tmp_path / "wt"
        _git(repo, "worktree", "add", "-q", str(wt), "-b", value)
        kwargs["invoke_cwd"] = wt
    # "default" / "main-cwd": invoke from the repo root (branch main).
    if expected == "raise":
        with pytest.raises(vci.CheckRefResolutionError):
            vci.resolve_check_ref(repo, 77, **kwargs)
    else:
        assert vci.resolve_check_ref(repo, 77, **kwargs) == expected


def test_cli_repo_branch_validator_rejects_empty_and_origin_prefixed(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    plan = _plan(tmp_path, "no citations")
    with pytest.raises(SystemExit) as ei:
        _main(repo, plan, extra=["--repo-branch", ""])
    assert ei.value.code == 2
    with pytest.raises(SystemExit) as ei2:
        _main(repo, plan, extra=["--repo-branch", "origin/issue-77"])
    assert ei2.value.code == 2
    err = capsys.readouterr().err
    assert "origin/" in err and "--ref" in err  # names the remedy


def test_cli_worktree_inference_via_cwd_default(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The ONE test exercising the Path.cwd() default positively (CLI-level)."""
    wt = tmp_path / "wt-full"
    _git(repo, "worktree", "add", "-q", str(wt), "-b", "issue-77-full")
    _git(repo, "push", "-q", "origin", "issue-77-full")
    plan = _plan(tmp_path, "no citations here")
    argv = [
        "--plan",
        str(plan),
        "--issue",
        "77",
        "--repo-root",
        str(repo),
        "--no-fetch",
        "--json",
    ]
    with contextlib.chdir(wt):
        rc = vci.main(argv)
    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["check_ref"] == "origin/issue-77-full"
    assert payload["check_ref_source"] == "worktree-branch"


def test_local_branch_regression_arms(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """R6/MF-3: NO derivation converts the committed-unpushed FAIL to SKIP/WARN.

    Fixture: an own-issue tracked-results citation committed ONLY on the
    LOCAL issue-77 tip — absent from every origin ref and from disk.
    """
    _branch_push(repo, "issue-77")  # stale bare tip on origin (no file)
    _git(repo, "checkout", "-q", "issue-77")
    _write(repo, "eval_results/issue_77/inp.json")
    _git(repo, "add", "eval_results/issue_77/inp.json")
    _git(repo, "commit", "-q", "-m", "add input on local branch")  # NOT pushed
    _git(repo, "checkout", "-q", "main")
    _branch_push(repo, "foo")  # pushed foreign branch, no file
    main_sha = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "main"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    plan = _plan(tmp_path, "Consumes eval_results/issue_77/inp.json from the prior round.")

    def _fail_row(extra: list[str]) -> str:
        assert _main(repo, plan, extra=extra) == 1
        out = capsys.readouterr().out
        rows = [ln for ln in out.splitlines() if "committed-unpushed" in ln]
        assert len(rows) == 1, out
        return rows[0]

    _fail_row([])  # (a) bare-only default
    _fail_row(["--ref", "origin/main"])  # (b)
    _fail_row(["--ref", "origin/foo"])  # (c) pushed foreign branch
    row_sha = _fail_row(["--ref", main_sha])  # (d) raw SHA check ref
    # (h) non-branch-shaped wording: no branch interpolated from the raw SHA.
    assert "the branch the dispatch will materialize" in row_sha
    _fail_row(["--ref", "foo"])  # (e) non-origin local-ref form
    # (f)/(g): own-issue SUFFIXED namespace — the union-probe extension.
    _branch_push(repo, "issue-77-full")  # cut from main, no file
    row_f = _fail_row(["--ref", "origin/issue-77-full"])
    row_g = _fail_row(["--repo-branch", "issue-77-full"])
    # (h) remediation names the FOUND branch and the MATERIALIZED ref.
    for row in (row_f, row_g):
        assert "issue-77" in row and "origin/issue-77-full" in row
    assert "merge/rebase" in row_g


def test_cli_unchanged_rows_semantic_invariants(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """MF-5/R4: bare-invocation rows on the legacy fixture states keep today's
    resolved ref, per-severity verdict counts, and exit code — asserted as
    FIXTURE LITERALS, never recomputed through resolve_check_ref."""
    # One row per severity by construction:
    #   pass: a.json committed+pushed on main BEFORE any branch cut (in-ref)
    #   warn: data/issue_77/x.json untracked (data-local-only)
    #   fail: eval_results/issue_12/m.json untracked (untracked-local-only)
    #   skip: eval_results/issue_77/out.json nowhere (planned-output)
    _write(repo, "eval_results/issue_12/a.json")
    _commit_push(repo, "eval_results/issue_12/a.json")
    _write(repo, "data/issue_77/x.json")
    _write(repo, "eval_results/issue_12/m.json")
    text = (
        "Reuses eval_results/issue_12/a.json and consumes data/issue_77/x.json "
        "and reads eval_results/issue_12/m.json "
        "and writes eval_results/issue_77/out.json now."
    )
    plan = _plan(tmp_path, text)

    def _counts(payload: dict) -> dict[str, int]:
        counts: dict[str, int] = {}
        for f in payload["findings"]:
            counts[f["verdict"]] = counts.get(f["verdict"], 0) + 1
        return counts

    # main-only state: the LITERAL origin/main.
    assert _main(repo, plan, extra=["--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["check_ref"] == "origin/main"
    assert _counts(payload) == {"pass": 1, "warn": 1, "fail": 1, "skip": 1}
    # bare-only state: the LITERAL origin/issue-77 (cut AFTER a.json landed
    # on main, so the pass row survives in-ref on the branch tip too).
    _branch_push(repo, "issue-77")
    assert _main(repo, plan, extra=["--json"]) == 1
    payload2 = json.loads(capsys.readouterr().out)
    assert payload2["check_ref"] == "origin/issue-77"
    assert _counts(payload2) == {"pass": 1, "warn": 1, "fail": 1, "skip": 1}


@pytest.mark.parametrize(
    ("check_ref", "ref_source", "expected"),
    [
        ("origin/issue-77-full", "repo-branch-flag", "issue-77-full"),
        ("origin/main", "repo-branch-flag", "main"),
        ("origin/issue-77-full", "worktree-branch", "issue-77-full"),
        ("origin/issue-77", "bare-issue-branch-default", "issue-77"),
        ("origin/main", "origin-main-default", "issue-77"),
        ("origin/issue-77-full", "ref-flag", "issue-77-full"),
        ("origin/issue-77", "ref-flag", "issue-77"),
        ("origin/foo", "ref-flag", "issue-77"),
        ("origin/main", "ref-flag", "issue-77"),
        ("3f2c1a9", "ref-flag", "issue-77"),
        ("foo", "ref-flag", "issue-77"),
        ("issue-77-full", "ref-flag", "issue-77-full"),
        ("origin/issue-771-x", "ref-flag", "issue-77"),  # digit boundary
    ],
)
def test_derive_local_branch_pure(check_ref: str, ref_source: str, expected: str) -> None:
    assert vci.derive_local_branch(check_ref, ref_source, 77) == expected


@pytest.mark.parametrize("branch", ["issue-77", "issue-77-full"])
def test_derive_worktree_repo_branch_unique(repo: Path, tmp_path: Path, branch: str) -> None:
    _git(repo, "worktree", "add", "-q", str(tmp_path / "wt"), "-b", branch)
    assert vci.derive_worktree_repo_branch(repo, 77) == branch


@pytest.mark.parametrize("state", ["no-worktree", "foreign-issue", "digit-prefix", "detached"])
def test_derive_worktree_repo_branch_zero_match_raises(
    repo: Path, tmp_path: Path, state: str
) -> None:
    if state == "foreign-issue":
        _git(repo, "worktree", "add", "-q", str(tmp_path / "wt"), "-b", "issue-88-x")
    elif state == "digit-prefix":
        _git(repo, "worktree", "add", "-q", str(tmp_path / "wt"), "-b", "issue-771-x")
    elif state == "detached":
        _git(repo, "worktree", "add", "-q", "--detach", str(tmp_path / "wt"))
    with pytest.raises(vci.RepoBranchDerivationError) as ei:
        vci.derive_worktree_repo_branch(repo, 77)
    assert "--repo-branch main" in str(ei.value)  # the wholly-main escape


def test_derive_worktree_repo_branch_multiple_raises(repo: Path, tmp_path: Path) -> None:
    wt1 = tmp_path / "wt1"
    wt2 = tmp_path / "wt2"
    _git(repo, "worktree", "add", "-q", str(wt1), "-b", "issue-77")
    _git(repo, "worktree", "add", "-q", str(wt2), "-b", "issue-77-full")
    with pytest.raises(vci.RepoBranchDerivationError) as ei:
        vci.derive_worktree_repo_branch(repo, 77)
    msg = str(ei.value)
    assert "issue-77" in msg and "issue-77-full" in msg
    assert str(wt1) in msg and str(wt2) in msg
    assert "--repo-branch <branch>" in msg


def test_derive_worktree_repo_branch_stale_registration_skipped(repo: Path, tmp_path: Path) -> None:
    wt = tmp_path / "wt-stale"
    _git(repo, "worktree", "add", "-q", str(wt), "-b", "issue-77-gone")
    shutil.rmtree(wt)  # deleted WITHOUT `git worktree prune` (probe P1b)
    with pytest.raises(vci.RepoBranchDerivationError) as ei:
        vci.derive_worktree_repo_branch(repo, 77)
    msg = str(ei.value)
    assert "issue-77-gone" in msg and "stale" in msg


def test_derive_worktree_repo_branch_stale_plus_live_notes_skip(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """#2263 review finding 2: a SUCCESS beside a skipped stale registration
    emits one diagnostic stderr note — a transient is_dir() False could
    otherwise silently convert a designed multiple-match refusal into a
    unique match with no trace."""
    stale = tmp_path / "wt-stale"
    _git(repo, "worktree", "add", "-q", str(stale), "-b", "issue-77-gone")
    shutil.rmtree(stale)
    _git(repo, "worktree", "add", "-q", str(tmp_path / "wt-live"), "-b", "issue-77-full")
    assert vci.derive_worktree_repo_branch(repo, 77) == "issue-77-full"
    err = capsys.readouterr().err
    assert "skipped stale" in err and "issue-77-gone" in err


def test_cli_print_repo_branch(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Healthy: EXACTLY the branch + newline on stdout, empty stderr, rc 0.
    _git(repo, "worktree", "add", "-q", str(tmp_path / "wt"), "-b", "issue-77-full")
    rc = vci.main(["--print-repo-branch", "--issue", "77", "--repo-root", str(repo)])
    captured = capsys.readouterr()
    assert rc == 0
    assert captured.out == "issue-77-full\n"
    assert captured.err == ""
    # Zero-match: rc 2, stdout EXACTLY EMPTY (the ${REPO_BRANCH:?} guard keys
    # on it), stderr names the --repo-branch main escape.
    rc2 = vci.main(["--print-repo-branch", "--issue", "88", "--repo-root", str(repo)])
    captured2 = capsys.readouterr()
    assert rc2 == 2
    assert captured2.out == ""
    assert "ERROR:" in captured2.err and "--repo-branch main" in captured2.err
    # Mode-combination refusals: EVERY forbidden check-mode flag family
    # (all 7 — #2263 r2 finding 4) exits 2 via argparse, naming the flag.
    for extra in (
        ["--plan", str(_plan(tmp_path, "x"))],
        ["--ref", "origin/main"],
        ["--repo-branch", "main"],
        ["--extra-sync-path", "eval_results/issue_77/x"],
        ["--no-fetch"],
        ["--json"],
        ["--lane", "rsync"],
    ):
        with pytest.raises(SystemExit) as ei:
            vci.main(["--print-repo-branch", "--issue", "77", "--repo-root", str(repo), *extra])
        assert ei.value.code == 2
        assert extra[0] in capsys.readouterr().err  # the refusal names the flag
    # Non-print mode still REQUIRES --plan.
    with pytest.raises(SystemExit) as ei2:
        vci.main(["--issue", "77", "--repo-root", str(repo), "--no-fetch"])
    assert ei2.value.code == 2


def test_git_env_sanitized_against_repo_selection_overrides(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MF-C: inherited GIT_* repo-selection overrides cannot repoint probes."""
    _branch_push(repo, "issue-77")  # bare-only remote state
    _git(repo, "worktree", "add", "-q", str(tmp_path / "wt"), "-b", "issue-77-full")
    baseline = vci.resolve_check_ref(repo, 77, fetch=False, invoke_cwd=repo)
    assert baseline == ("origin/issue-77", "bare-issue-branch-default")
    # Foreign repo with a DIVERGENT remote state (a pushed issue-77-x suffix
    # would force a refusal if the poisoned env were honored), an issue-scoped
    # checkout branch, and its own issue-scoped worktree.
    foreign = _foreign_repo(tmp_path, name="foreign", branch="issue-77-full")
    _git(foreign, "branch", "issue-77-x")
    _git(foreign, "push", "-q", "origin", "issue-77-x")
    _git(foreign, "worktree", "add", "-q", str(tmp_path / "fwt"), "-b", "issue-77-foreign")
    poison = {
        "GIT_DIR": str(foreign / ".git"),
        "GIT_COMMON_DIR": str(foreign / ".git"),
        "GIT_WORK_TREE": str(foreign),
    }
    for var, value in poison.items():
        monkeypatch.setenv(var, value)
        # (a) resolution identical to the unset-env baseline.
        assert vci.resolve_check_ref(repo, 77, fetch=False, invoke_cwd=repo) == baseline
        # (b) the repository-binding guard still REJECTS the foreign cwd —
        # the bare default, never the foreign branch as `worktree-branch`.
        assert vci.resolve_check_ref(repo, 77, fetch=False, invoke_cwd=foreign) == baseline
        # (c) worktree derivation lists the FIXTURE repo's worktrees, not
        # the foreign repo's.
        assert vci.derive_worktree_repo_branch(repo, 77) == "issue-77-full"
        monkeypatch.delenv(var)
    # (d) bare-context negative arm (no poison): a .git-dir cwd reports a
    # branch (probe P1) but fails --is-inside-work-tree (probe P4) -> rung 4
    # default, never `worktree-branch`.
    _git(repo, "checkout", "-q", "issue-77")
    got = vci.resolve_check_ref(repo, 77, fetch=False, invoke_cwd=repo / ".git")
    assert got == ("origin/issue-77", "bare-issue-branch-default")
    # (e) the _default_repo_root CHOKEPOINT itself under the full poison set
    # (#2263 r2 finding 4): every other arm passes an explicit repo root, so
    # sanitization removed from _default_repo_root ALONE would stay green
    # without this arm — a poisoned default must still resolve the CWD's
    # repository, never the foreign one.
    for var, value in poison.items():
        monkeypatch.setenv(var, value)
    with contextlib.chdir(repo):
        got_root = vci._default_repo_root()
    assert got_root is not None
    assert Path(os.path.realpath(got_root)) == Path(os.path.realpath(repo))


@pytest.mark.parametrize("remote_state", ["none", "bare-only", "suffix-only", "bare+suffix"])
def test_cli_explicit_ref_bypasses_resolver_on_all_remote_states(
    repo: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str], remote_state: str
) -> None:
    """Rung 1: --ref never calls the resolver — exit 0 even on the suffix
    states whose resolver path would REFUSE (exit 2)."""
    _write(repo, "eval_results/issue_12/a.json")
    _commit_push(repo, "eval_results/issue_12/a.json")
    if remote_state in ("bare-only", "bare+suffix"):
        _branch_push(repo, "issue-77")
    if remote_state in ("suffix-only", "bare+suffix"):
        _branch_push(repo, "issue-77-full")
    plan = _plan(tmp_path, "Reuses eval_results/issue_12/a.json only.")
    assert _main(repo, plan, extra=["--ref", "origin/main", "--json"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["check_ref"] == "origin/main"
    assert payload["check_ref_source"] == "ref-flag"


def test_sanitized_git_env_strips_config_injection_channels(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """#2263 r2: every config-injection env channel is stripped — COUNT,
    PARAMETERS, the GLOBAL/SYSTEM file redirects, and the indexed KEY/VALUE
    pairs at EVERY index (not just 0) — while an unrelated GIT_*-prefixed
    name survives (the pair regex is anchored, not a prefix match)."""
    poison = {
        "GIT_CONFIG_COUNT": "13",
        "GIT_CONFIG_KEY_0": "url.x.insteadOf",
        "GIT_CONFIG_VALUE_0": "y",
        "GIT_CONFIG_KEY_12": "url.a.insteadOf",
        "GIT_CONFIG_VALUE_12": "b",
        "GIT_CONFIG_PARAMETERS": "'url.x.insteadOf=y'",
        "GIT_CONFIG_GLOBAL": "/dev/null",
        "GIT_CONFIG_SYSTEM": "/dev/null",
    }
    for k, v in poison.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setenv("GIT_CONFIGURATION_UNRELATED", "keep")  # anchor guard
    env = vci._sanitized_git_env()
    for k in poison:
        assert k not in env, k
    assert env["GIT_CONFIGURATION_UNRELATED"] == "keep"
    # #2263 r3 finding 3: the documented-OPEN channels are DELIBERATELY not
    # stripped — operators set these on purpose (a remote reachable only via
    # a custom SSH/proxy command; HOME/XDG relocate the user config file),
    # and stripping them trades a real fetch failure for a theoretical
    # spoof. The sanitizer defends accidental inheritance, not a hostile
    # caller (who already controls the session).
    open_channels = {
        "GIT_SSH": "/usr/bin/ssh",
        "GIT_SSH_COMMAND": "ssh -i /tmp/key",
        "GIT_PROXY_COMMAND": "/usr/bin/proxy-cmd",
        "GIT_EXEC_PATH": "/usr/lib/git-core",
        "HOME": os.environ.get("HOME", "/home/x"),
        "XDG_CONFIG_HOME": "/tmp/xdg",
    }
    for k, v in open_channels.items():
        monkeypatch.setenv(k, v)
    env2 = vci._sanitized_git_env()
    for k, v in open_channels.items():
        assert env2.get(k) == v, f"documented-open channel stripped: {k}"


def test_git_config_env_injection_cannot_redirect_fetch(
    repo: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#2263 r2 (git-config-origin-spoof): an env-injected
    `url.<foreign>.insteadOf` rewrite cannot redirect the gate's bounded
    fetch — the sanitized fetch reads the REAL origin.

    The control arm FIRST proves the channel is potent on this git binary (a
    raw UNSANITIZED fetch under the poison pulls the FOREIGN tip into
    origin/issue-77), so the sanitized arm cannot pass hollow.
    """
    # Real origin: issue-77 carries real_marker.txt.
    _git(repo, "checkout", "-q", "-b", "issue-77")
    _write(repo, "real_marker.txt", "real\n")
    _git(repo, "add", "real_marker.txt")
    _git(repo, "commit", "-q", "-m", "real issue-77")
    _git(repo, "push", "-q", "origin", "issue-77")
    _git(repo, "checkout", "-q", "main")
    # Foreign origin: an UNRELATED repo whose issue-77 tip carries
    # foreign_marker.txt instead.
    foreign = _foreign_repo(tmp_path, name="evil", branch="issue-77")
    _write(foreign, "foreign_marker.txt", "foreign\n")
    _git(foreign, "add", "foreign_marker.txt")
    _git(foreign, "commit", "-q", "-m", "foreign issue-77")
    _git(foreign, "push", "-q", "origin", "issue-77")
    poison = {
        "GIT_CONFIG_COUNT": "1",
        "GIT_CONFIG_KEY_0": f"url.{tmp_path / 'evil-origin.git'}.insteadOf",
        "GIT_CONFIG_VALUE_0": str(tmp_path / "origin.git"),
    }
    # CONTROL ARM — raw fetch, poison honored: origin/issue-77 now points at
    # the foreign tip (the reproduced #2263 r2 spoof).
    subprocess.run(
        ["git", "-C", str(repo), "fetch", "origin", "--quiet", "--no-tags", "issue-77"],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, **poison},
    )
    assert vci.path_in_ref(repo, "origin/issue-77", "foreign_marker.txt")
    assert not vci.path_in_ref(repo, "origin/issue-77", "real_marker.txt")
    # SANITIZED ARM — the SAME poison in os.environ: the resolver's
    # fetch=True path strips it and restores origin/issue-77 to the REAL tip.
    for k, v in poison.items():
        monkeypatch.setenv(k, v)
    got = vci.resolve_check_ref(repo, 77, fetch=True, invoke_cwd=repo)
    assert got == ("origin/issue-77", "bare-issue-branch-default")
    assert vci.path_in_ref(repo, "origin/issue-77", "real_marker.txt")
    assert not vci.path_in_ref(repo, "origin/issue-77", "foreign_marker.txt")


def _launch_fence_executable_span(text: str) -> str:
    """The Step 6b launch fence's recheck->dispatch span, VERBATIM.

    Extracts the ONE ```bash block carrying both the gate recheck and the
    `dispatch_issue.py launch` command, slices at its `REPO_BRANCH=` prelude
    line (the lines above it are the `BACKEND=`/`INTENT=<inferred>` setup —
    `<inferred>` is an orchestrator-filled placeholder that is not valid
    bash, and both feed only the dispatch command, which the caller
    stands in for), and replaces ONLY the dispatch command itself with an
    `echo "DISPATCHED-$REPO_BRANCH"` reachability stand-in. No halt
    semantics are added or removed: whether a failing recheck stops the
    dispatch is decided by the production text alone (#2263 r3 finding 1 —
    the prior version of this test injected `set -e`, constructing the very
    behavior it claimed to verify). Extraction failure IS test failure.
    """
    blocks = re.findall(r"(?ms)^```bash\n(.*?)^```$", text)
    hits = [
        b
        for b in blocks
        if "scripts/dispatch_issue.py launch" in b
        and 'scripts/verify_carryover_inputs.py --plan "$PLAN_PATH"' in b
    ]
    assert len(hits) == 1, f"launch-fence block extraction failed ({len(hits)} candidates)"
    lines = hits[0].splitlines()
    starts = [i for i, ln in enumerate(lines) if ln.startswith("REPO_BRANCH=")]
    assert len(starts) == 1, "launch-fence REPO_BRANCH prelude extraction failed"
    span = "\n".join(lines[starts[0] :])
    span, n_sub = re.subn(
        r"(?ms)^uv run python scripts/dispatch_issue\.py launch \\\n"
        r".*?\$\{BACKEND:\+--backend \"\$BACKEND\"\}$",
        'echo "DISPATCHED-$REPO_BRANCH"',
        span,
    )
    assert n_sub == 1, "dispatch-command stand-in substitution failed"
    return span


def test_step6_launch_fence_gate_recheck_blocks_cross_fence_divergence(
    repo: Path, tmp_path: Path
) -> None:
    """#2263 r2 (cross-fence-ref-drift) + r3 finding 1: the Step 6b launch
    fence RE-RUNS the carry-over gate against ITS OWN resolved branch AND
    mechanically HALTS on a recheck failure, so a sole worktree switched
    from pushed branch A to pushed branch B between the fences is re-graded
    — and here REFUSED — before any dispatch (the false-PASS class this
    task exists to close; both resolutions succeed, so no resolver refusal
    can catch it).

    The fence executes AS WRITTEN (verbatim span, no injected `set -e`, no
    synthesized halt — see `_launch_fence_executable_span`): the divergence
    arm fails if and only if the real text fails to halt. Against the r2
    text (bare recheck adjacent to the dispatch, halt in a comment only)
    this test FAILS on the divergence arm — the dispatch stand-in is
    reached. NOTE the sibling
    `test_step6_repo_branch_shared_resolver_fresh_shell_execution` is real
    for SOURCE drift but structurally blind to cross-fence PARITY (it runs
    one prelude against two independent static states) — this test owns the
    divergence arm.
    """
    text = issue_skill_text()
    fence = (
        _launch_fence_executable_span(text)
        .replace("<N>", "77")
        .replace(
            "uv run python scripts/verify_carryover_inputs.py",
            f"{sys.executable} {_SCRIPT}",
        )
    )

    # Fixture: the plan cites a file committed+pushed ONLY on branch A
    # (issue-77-full, checked out in the sole issue worktree). An untracked
    # VM-local copy at the repo root keeps the divergence arm out of the
    # own-issue planned-output SKIP (a nonexistent own-issue path would skip).
    wt = tmp_path / "wt"
    _git(repo, "worktree", "add", "-q", str(wt), "-b", "issue-77-full")
    _write(wt, "eval_results/issue_77/f.json")
    _git(wt, "add", "eval_results/issue_77/f.json")
    _git(wt, "commit", "-q", "-m", "input on branch A")
    _git(wt, "push", "-q", "origin", "issue-77-full")
    _write(repo, "eval_results/issue_77/f.json")  # untracked VM-local copy
    plan = _plan(tmp_path, "Consumes eval_results/issue_77/f.json from the prior round.")
    env = {**os.environ, "PLAN_PATH": str(plan)}

    # Healthy arm: the fence resolves A, the recheck grades
    # origin/issue-77-full (in-ref) -> the dispatch line is reached.
    ok = subprocess.run(["bash", "-c", fence], cwd=repo, capture_output=True, text=True, env=env)
    assert ok.returncode == 0, ok.stderr
    assert "DISPATCHED-issue-77-full" in ok.stdout

    # Divergence arm: the SOLE worktree switches to pushed branch B (cut from
    # main — the cited file is NOT reachable there). Pre-fix, the launch
    # fence dispatched B with no gate ever grading it.
    _git(repo, "branch", "issue-77-alt", "main")
    _git(repo, "push", "-q", "origin", "issue-77-alt")
    _git(wt, "checkout", "-q", "issue-77-alt")
    blocked = subprocess.run(
        ["bash", "-c", fence], cwd=repo, capture_output=True, text=True, env=env
    )
    assert blocked.returncode == 1, blocked.stderr  # the guard's exit 1, not the resolver's 2
    assert "DISPATCHED" not in blocked.stdout
    assert "dispatch REFUSED" in blocked.stderr  # the guard's fail-loud message fired
    combined = blocked.stdout + blocked.stderr
    assert "untracked-local-only" in combined  # the RECHECK graded branch B
    assert "eval_results/issue_77/f.json" in combined


def test_step6_launch_fence_recheck_mechanical_halt_and_lane_parity() -> None:
    """#2263 r3+r4 text pins: the recheck halts MECHANICALLY, carries the
    SAME lane/extra-sync args as the 6a.5 invocation, and the LAUNCH argv
    expands the same extra-sync values.

    (a) Mechanical halt (r3 finding 1): the recheck runs inside an
        `if ! ...; then ... exit 1; fi` guard whose body fail-louds; the
        launch fence has no `set -e`, so a BARE recheck's non-zero rc would
        not stop the adjacent dispatch — exactly one bare (line-initial)
        invocation may exist, the 6a.5 gate (last command of its block,
        whose rc the orchestrator branches on in prose).
    (b) Lane parity (r3 finding 2): BOTH gate invocations carry the identical
        argv incl. the `"${LANE_ARGS[@]}"` lane/extra-sync token, and the
        two `LANE_ARGS=` default assignments are byte-identical — the rsync
        suffix is in the COMMAND, not a comment, so the 6a.5-graded lane
        set and the recheck-graded lane set cannot drift.
    (c) Launch parity (r4, reconciler v3 BLOCKER): the operational
        `dispatch_issue.py launch` command ITSELF expands
        `${EXTRA_SYNC_ARGS[@]+"${EXTRA_SYNC_ARGS[@]}"}` — without it, a
        gate + recheck PASS earned via `--extra-sync-path` certifies
        rsync-lane inputs the dispatched tree never stages (deterministic
        post-provision missing-input crash). The `+`-guard form is pinned
        so the expansion stays unset-array-safe on non-rsync lanes.
    """
    text = issue_skill_text()
    invocation = (
        'uv run python scripts/verify_carryover_inputs.py --plan "$PLAN_PATH" '
        '--issue <N> --repo-branch "$REPO_BRANCH" "${LANE_ARGS[@]}"'
    )
    assert text.count(invocation) == 2  # 6a.5 gate + launch-fence recheck, identical argv
    bare = re.findall(r"(?m)^uv run python scripts/verify_carryover_inputs\.py --plan .*$", text)
    assert len(bare) == 1, "an UNGUARDED recheck adjacent to the dispatch is the r3 defect"
    m = re.search(
        r"(?ms)^if ! uv run python scripts/verify_carryover_inputs\.py --plan [^\n]*; then\n"
        r"(.*?)^fi$",
        text,
    )
    assert m, "launch-fence recheck guard extraction failed"
    body = m.group(1)
    assert "dispatch REFUSED" in body
    assert re.search(r"(?m)^\s+exit 1$", body), "guard body must halt the fence"
    lane_lines = re.findall(r"(?m)^LANE_ARGS=.*$", text)
    assert len(lane_lines) == 2, "one LANE_ARGS default assignment per fence"
    assert len(set(lane_lines)) == 1  # byte-identical across fences
    assert lane_lines[0].startswith("LANE_ARGS=()")
    assert '--lane rsync "${EXTRA_SYNC_ARGS[@]}"' in lane_lines[0]  # the rsync form named
    # (c) The OPERATIONAL launch command (the one sharing a bash block with
    # the recheck) expands the extra-sync values in its OWN argv.
    blocks = re.findall(r"(?ms)^```bash\n(.*?)^```$", text)
    op_blocks = [
        b
        for b in blocks
        if "scripts/dispatch_issue.py launch" in b
        and 'scripts/verify_carryover_inputs.py --plan "$PLAN_PATH"' in b
    ]
    assert len(op_blocks) == 1, "operational launch block extraction failed"
    launch = re.search(
        r"(?ms)^uv run python scripts/dispatch_issue\.py launch \\\n(?:[^\n]*\\\n)*[^\n]*$",
        op_blocks[0],
    )
    assert launch, "operational launch command extraction failed"
    assert '${EXTRA_SYNC_ARGS[@]+"${EXTRA_SYNC_ARGS[@]}"}' in launch.group(0), (
        "the LAUNCH argv must expand the SAME extra-sync values the gate + "
        "recheck graded (#2263 r4) — a gate PASS earned via --extra-sync-path "
        "must be a set the dispatched tree actually stages; the ${VAR[@]+...} "
        "guard keeps non-rsync lanes (unset array) safe under set -u"
    )


def test_corpus_sweep_resolver_failure_surfaces_error_not_substitute(
    repo: Path,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """#2263 r2 finding 3: a resolve_check_ref failure in the #1995 corpus
    sweep is recorded per plan + exits nonzero — NEVER silently graded
    against a substituted origin/main (plausible-but-wrong calibration)."""
    monkeypatch.setitem(sys.modules, "verify_carryover_inputs", vci)  # restored at teardown
    spec = importlib.util.spec_from_file_location(
        "issue1995_corpus_sweep", REPO_ROOT / "scripts" / "issue1995_corpus_sweep.py"
    )
    assert spec is not None and spec.loader is not None
    sweep = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sweep)

    plan_dir = repo / "tasks" / "running" / "77" / "plans"
    plan_dir.mkdir(parents=True)
    (plan_dir / "v1.md").write_text("Runs scripts/issue77_helper.py on the corpus.\n")

    def _boom(*_a, **_k):
        raise vci.CheckRefResolutionError("cannot enumerate candidates (for-each-ref rc=128)")

    monkeypatch.setattr(sweep._vci, "resolve_check_ref", _boom)
    out_dir = tmp_path / "sweep-out"
    rc = sweep.main(["--repo-root", str(repo), "--no-fetch", "--output-dir", str(out_dir)])
    captured = capsys.readouterr()
    assert rc != 0
    assert "resolution_errors=1" in captured.out
    assert "check-ref resolution failed" in captured.err
    payload = json.loads((out_dir / "corpus_sweep.json").read_text())
    (rec,) = payload["sample"]
    assert rec["resolution_error"].startswith("CheckRefResolutionError")
    assert rec["check_ref"] is None
    assert rec["candidates"] == []
    # No record graded against a substituted fallback ref.
    assert all(r["check_ref"] != "origin/main" for r in payload["sample"])
    assert payload["aggregates"]["n_resolution_errors"] == 1
