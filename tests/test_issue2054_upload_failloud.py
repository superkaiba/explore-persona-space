"""Pins for the #2054 round-9 revision r2: upload-helper empty-return discard
(code-review v5 Major + its class sweep).

``hub._upload_folder_filtered`` / ``hub._upload`` are fail-SOFT by RETURN on
every failure shape — missing HF_TOKEN, incomplete expected-set verify, and
the terminal ``except Exception`` all log and return ``""`` — so a caller
that discards the return converts a production HF-mirror failure into a
false success exit (concern ``upload-mirror-return-discard``, round 9).
These tests pin the capture-and-raise contract at all three #2054 upload
sites (fails pre-fix — the pre-fix bodies discard the return and never
raise; passes post-fix):

- ``issue2054_build_answers._upload_pool`` (the round-9 Major)
- ``issue2054_phase_a._upload_scaffold_files`` (class sibling)
- ``issue2054_phase_a._upload_fold_map`` (class sibling)

Boundary fakes are ``create_autospec`` on the real hub helpers
(signature-conformant by construction), patched at the SOURCE module — the
callers import the helpers inside the function body at call time. No
network, no worktree paths; ``tmp_path`` only. Fixture rows are synthetic
placeholder text — no real-corpus content.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import create_autospec

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue2054_build_answers as ba  # noqa: E402
import issue2054_phase_a as pa  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402


def _patch_folder_upload(monkeypatch: pytest.MonkeyPatch, ret: str):
    fake = create_autospec(hub._upload_folder_filtered, return_value=ret)
    monkeypatch.setattr(hub, "_upload_folder_filtered", fake)
    return fake


def _pool_files(tmp_path: Path) -> list[Path]:
    pool = tmp_path / "answers_pool.jsonl"
    pool.write_text('{"conv_id": "mt_0", "answer": "x"}\n', encoding="utf-8")
    meta = tmp_path / "answers_pool.meta.json"
    meta.write_text("{}", encoding="utf-8")
    return [pool, meta]


def test_upload_pool_raises_on_empty_return(tmp_path, monkeypatch):
    """The Major: a fail-soft "" return must raise, not log success."""
    _patch_folder_upload(monkeypatch, "")
    with pytest.raises(RuntimeError, match="answers-pool HF mirror failed or incomplete"):
        ba._upload_pool(tmp_path, _pool_files(tmp_path))


def test_upload_pool_passes_on_url_return(tmp_path, monkeypatch):
    fake = _patch_folder_upload(monkeypatch, "https://huggingface.co/datasets/x")
    ba._upload_pool(tmp_path, _pool_files(tmp_path))  # no raise
    assert fake.call_count == 1


def test_upload_scaffold_files_fail_loud_raises_on_empty_return(tmp_path, monkeypatch):
    _patch_folder_upload(monkeypatch, "")
    f = tmp_path / "kept.json"
    f.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="scaffold bulk upload failed or incomplete"):
        pa._upload_scaffold_files(tmp_path, [f], fail_loud=True)


def test_upload_scaffold_files_fail_soft_warns_on_empty_return(tmp_path, monkeypatch, capsys):
    """fail_loud=False keeps the documented warn-and-continue semantics."""
    _patch_folder_upload(monkeypatch, "")
    f = tmp_path / "kept.json"
    f.write_text("{}", encoding="utf-8")
    pa._upload_scaffold_files(tmp_path, [f], fail_loud=False)  # no raise
    assert "WARN scaffold bulk upload failed" in capsys.readouterr().out


def test_upload_fold_map_fail_loud_raises_on_empty_return(tmp_path, monkeypatch):
    fake = create_autospec(hub._upload, return_value="")
    monkeypatch.setattr(hub, "_upload", fake)
    fm = tmp_path / "shared_fold_map.json"
    fm.write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match="fold-map upload returned no path"):
        pa._upload_fold_map(fm, fail_loud=True)
    assert fake.call_count == 1
