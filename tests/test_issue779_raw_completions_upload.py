"""Regression test for the issue #779 raw-completions-upload BLOCKER fix.

Pins the permanent invariant installed post-pivot round-1 (reconciler v2
BLOCKER #1 ``raw-completions-upload-prefix-missing``): the collect dispatcher's
rollout TEXT MUST land under the canonical ``issue779_monitoring/raw_completions/``
prefix — NOT only under ``analysis_tensors/`` — and the upload MUST be
mechanically verified (every produced ``(trait, condition)`` has a file at the
canonical prefix) before ``phase("done")`` (plan v5 §10 row (c)).

Pre-fix: ``_upload_collect`` bulk-uploaded the whole out_dir (cell JSONs with
rollout text included) to ``analysis_tensors/`` and there was NO
``raw_completions/`` upload path anywhere — this test would FAIL (no upload with
a ``raw_completions/`` ``path_in_repo`` ever fired). Post-fix BOTH prefixes fire.

Pure-CPU, no model / no network: ``HfApi`` and ``list_repo_files`` are
monkeypatched to RECORD the ``path_in_repo`` of every ``upload_folder`` call and
to return a listing reflecting the recorded uploads, so the fail-loud
``list_repo_files`` verification exercises its real success path.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import issue779_collect as X  # noqa: E402


class _FakeHfApi:
    """Records each upload_folder call's path_in_repo + the files it would land."""

    def __init__(self, sink: dict) -> None:
        self._sink = sink

    def upload_folder(self, *, folder_path, path_in_repo, repo_id, repo_type, **_kw):
        # Enumerate the staged files exactly as HF would land them under the prefix.
        landed = [
            f"{path_in_repo}/{p.relative_to(folder_path).as_posix()}"
            for p in sorted(Path(folder_path).rglob("*"))
            if p.is_file()
        ]
        self._sink["uploads"].append(path_in_repo)
        self._sink["files"].update(landed)


def _write_pass_a_cell(pass_a_dir: Path, trait: str, cond_id: str) -> None:
    pass_a_dir.mkdir(parents=True, exist_ok=True)
    cell = {
        "trait": trait,
        "cond_id": cond_id,
        "mode": "system",
        "n_shot": 0,
        "n_questions": 1,
        "n_rollouts": 2,
        "rollout_seed": 42,
        "rollouts": [
            {"qi": 0, "ri": 0, "response": f"{trait} rollout text A"},
            {"qi": 0, "ri": 1, "response": f"{trait} rollout text B"},
        ],
        "judge_scores": {},
        "oracle_proj": {},
    }
    (pass_a_dir / f"{trait}__{cond_id}.json").write_text(json.dumps(cell))


def _patch_hub(monkeypatch):
    sink: dict = {"uploads": [], "files": set()}
    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _FakeHfApi(sink))
    monkeypatch.setattr(
        "huggingface_hub.list_repo_files",
        lambda *a, **k: sorted(sink["files"]),
    )
    return sink


def test_split_raw_completions_writes_flat_per_cell_layout(tmp_path):
    """_split_raw_completions produces one {trait}_{cond}_seed42.json per Pass-A
    cell, shaped {trait, condition, seed, rollouts:[{qi,ri,response}]}."""
    out_dir = tmp_path / "data"
    _write_pass_a_cell(out_dir / "pass_a", "evil", "sys0")
    _write_pass_a_cell(out_dir / "pass_a", "sycophancy", "shot5")
    staging = tmp_path / "staging"
    written = X._split_raw_completions(out_dir, staging)
    assert set(written) == {("evil", "sys0"), ("sycophancy", "shot5")}
    f = staging / "evil_sys0_seed42.json"
    assert f.exists()
    payload = json.loads(f.read_text())
    assert payload["trait"] == "evil"
    assert payload["condition"] == "sys0"
    assert payload["seed"] == 42
    assert [r["response"] for r in payload["rollouts"]] == [
        "evil rollout text A",
        "evil rollout text B",
    ]


def test_both_prefixes_fire_raw_completions_and_analysis_tensors(tmp_path, monkeypatch):
    """The upload phase fires BOTH an analysis_tensors/ upload AND a
    raw_completions/ upload (the post-fix invariant). Pre-fix only the
    analysis_tensors/ prefix ever appeared."""
    sink = _patch_hub(monkeypatch)
    out_dir = tmp_path / "data"
    _write_pass_a_cell(out_dir / "pass_a", "evil", "sys0")

    X._upload_collect(out_dir, smoke=False)
    X._upload_raw_completions(out_dir, smoke=False)

    prefixes = sink["uploads"]
    assert f"{X.C.HF_PREFIX}/analysis_tensors" in prefixes, prefixes
    assert f"{X.C.HF_PREFIX}/raw_completions" in prefixes, prefixes
    # The canonical raw-completions file for the produced cell landed under the
    # mandated flat layout.
    assert f"{X.C.HF_PREFIX}/raw_completions/evil_sys0_seed42.json" in sink["files"], sorted(
        sink["files"]
    )


def test_raw_completions_upload_verification_fails_loud_on_empty(tmp_path, monkeypatch):
    """Fail-loud when Pass A produced no rollout text to copy (the mechanical
    verification is a hard gate, not a warning-and-continue)."""
    _patch_hub(monkeypatch)
    out_dir = tmp_path / "data"
    (out_dir / "pass_a").mkdir(parents=True)  # empty pass_a -> nothing to copy
    with pytest.raises(RuntimeError, match="raw-completions upload aborted"):
        X._upload_raw_completions(out_dir, smoke=False)


def test_raw_completions_upload_verification_fails_loud_on_missing_prefix(tmp_path, monkeypatch):
    """If the upload lands NOTHING under the raw_completions/ prefix (the
    pre-fix failure mode simulated by a no-op uploader), the fail-loud
    list_repo_files verification raises with the blocker id."""
    sink: dict = {"uploads": [], "files": set()}

    class _NoOpApi:
        def upload_folder(self, **_kw):  # lands nothing
            sink["uploads"].append(_kw.get("path_in_repo"))

    monkeypatch.setattr("huggingface_hub.HfApi", lambda *a, **k: _NoOpApi())
    monkeypatch.setattr("huggingface_hub.list_repo_files", lambda *a, **k: [])
    out_dir = tmp_path / "data"
    _write_pass_a_cell(out_dir / "pass_a", "evil", "sys0")
    with pytest.raises(RuntimeError, match="raw-completions-upload-prefix-missing"):
        X._upload_raw_completions(out_dir, smoke=False)
