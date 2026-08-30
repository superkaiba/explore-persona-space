"""Tests for ``scripts/issue2641_fence_mask_audit.py`` (the #2641 audit harness).

Three pinned behaviors (plan #2641 v3 section 4.5):

(a) the frozen ``_blind_fence_mask`` reproduces the PRE-fix behavior on the
    section-4.5 fixture shapes — each of the four task-body cases shows a
    documented difference from the live CommonMark mask;
(b) ``--jobs 4`` output is byte-identical to ``--jobs 1`` on a 20-file
    corpus slice (``--no-timestamp``);
(c) ``explain``'s defect classifier assigns each section-2.4 worked file its
    expected defect class.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from glob import glob
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
HARNESS = REPO_ROOT / "scripts" / "issue2641_fence_mask_audit.py"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return sys.modules[name]


audit = _load("issue2641_fence_mask_audit", HARNESS)
audit._init_worker()  # load the live verify_plan once for the module-level tests


def _find_plan(task_id: str, version: str) -> Path:
    """Locate a task's plan file across status folders (tasks move on
    status change, so the status segment is a wildcard)."""
    hits = sorted(glob(str(REPO_ROOT / "tasks" / "*" / task_id / "plans" / f"{version}.md")))
    if not hits:
        pytest.skip(f"corpus file tasks/*/{task_id}/plans/{version}.md not present in this tree")
    return Path(hits[0])


# ── (a) frozen blind mask: documented divergence per task-body case ──────────


def test_blind_mask_mismatched_delimiter_case():
    lines = ["```", "code", "~~~", "prose"]
    # Blind: the ~~~ line CLOSES the backtick block, so `prose` reads as prose.
    assert audit._blind_fence_mask(lines) == [True, True, True, False]
    # Live: same delimiter required — the block stays open to EOF.
    assert audit._VP._fence_mask(lines) == [True, True, True, True]


def test_blind_mask_inner_shorter_fence_case():
    lines = ["````", "```", "still code", "````", "prose"]
    # Blind: the inner ``` closes the four-backtick block, so `still code`
    # reads as prose and the second ```` re-opens, swallowing `prose`.
    assert audit._blind_fence_mask(lines) == [True, True, False, True, True]
    assert audit._VP._fence_mask(lines) == [True, True, True, True, False]


def test_blind_mask_indented_marker_case():
    lines = ["prose", "    ``` not a fence", "prose2"]
    # Blind: indentation is invisible (`line.strip()`), so the indented
    # marker opens a phantom block that swallows `prose2`.
    assert audit._blind_fence_mask(lines) == [False, True, True]
    assert audit._VP._fence_mask(lines) == [False, False, False]


def test_blind_mask_unclosed_at_eof_case():
    # A closing CANDIDATE carrying an info string: blind toggles the block
    # closed; CommonMark keeps it open, so the file ends unclosed and the
    # tail is swallowed.
    lines = ["```", "code", "```bash", "tail prose"]
    assert audit._blind_fence_mask(lines) == [True, True, True, False]
    assert audit._VP._fence_mask(lines) == [True, True, True, True]
    assert audit._VP.unclosed_fence_line(lines) == 0
    assert audit._blind_unclosed(lines) is False


def test_blind_mask_matches_verbatim_anchor_body():
    """The frozen baseline still behaves exactly like the anchor-commit
    walk: toggle on ANY stripped ```/~~~ prefix, state otherwise."""
    lines = ["  ``` indented toggles too", "x", "~~~ mixed closes", "y"]
    assert audit._blind_fence_mask(lines) == [True, True, True, False]


# ── (c) explain's defect classifier on the section-2.4 worked files ──────────


def test_classify_714_info_string_backtick():
    path = _find_plan("714", "v2")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert "info-string-backtick" in audit._classify_defects(lines)


def test_classify_1176_indented_marker():
    path = _find_plan("1176", "v1")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert "indented-marker" in audit._classify_defects(lines)


def test_classify_1558_unclosed_at_eof():
    path = _find_plan("1558", "v1")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert "unclosed-at-eof" in audit._classify_defects(lines)


def test_classify_synthetic_mismatched_and_inner_shorter():
    lines = ["```", "a", "~~~", "b", "```", "````", "c", "```", "````", "d"]
    classes = audit._classify_defects(lines)
    assert "mismatched-delimiter" in classes
    assert "inner-shorter-fence" in classes


# ── (b) determinism: --jobs 4 byte-identical to --jobs 1 on a 20-file slice ──


def test_jobs4_byte_identical_to_jobs1(tmp_path):
    all_plans = sorted(glob(str(REPO_ROOT / "tasks" / "*" / "*" / "plans" / "v*.md")))
    if len(all_plans) < 40:
        pytest.skip("corpus too small in this tree")
    # 17 known non-changers (cheap in-process stage-1 screen) + the three
    # section-2.4 mask-changers, so stage 2 is exercised deterministically.
    changers = [
        str(_find_plan("714", "v2")),
        str(_find_plan("1176", "v1")),
        str(_find_plan("1558", "v1")),
    ]
    nonchangers: list[str] = []
    for p in all_plans:
        if p in changers:
            continue
        lines = Path(p).read_text(encoding="utf-8").splitlines()
        if audit._blind_fence_mask(lines) == audit._VP._fence_mask(lines):
            nonchangers.append(p)
        if len(nonchangers) == 17:
            break
    slice_paths = sorted(nonchangers + changers)
    assert len(slice_paths) == 20
    corpus_list = tmp_path / "slice.txt"
    corpus_list.write_text("\n".join(slice_paths) + "\n")

    outs = {}
    for jobs in (1, 4):
        out = tmp_path / f"out_j{jobs}.json"
        proc = subprocess.run(
            [
                sys.executable,
                str(HARNESS),
                "--corpus-list",
                str(corpus_list),
                "--kind-mode",
                "forced-experiment",
                "--jobs",
                str(jobs),
                "--no-timestamp",
                "--json",
                str(out),
            ],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            timeout=900,
        )
        assert proc.returncode == 0, proc.stderr[-2000:]
        outs[jobs] = out.read_bytes()

    assert outs[1] == outs[4]
    payload = json.loads(outs[1])
    assert payload["stage1"]["n_mask_changed"] == 3
    assert payload["stage2"]["forced-experiment"]["n_verdict_moving_files"] >= 1
