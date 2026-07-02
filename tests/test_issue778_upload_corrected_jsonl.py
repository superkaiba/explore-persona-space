"""Regression tests for issue #778 followup eval-JSONL promotion (CPU, offline).

Pins the reconciler round-1 BLOCKER ``jsonl-deliverables-never-promoted-before-
teardown``: the pod upload phase MUST promote the primary-deliverable monitoring
JSONLs (``monitoring_corrected_{trait}.jsonl`` + ``monitoring_manyshot_{trait}.jsonl``)
to the HF DATA repo under a stable prefix and re-verify every uploaded basename on a
FRESH Hub listing — otherwise the off-pod null battery FileNotFoundErrors after pod
teardown. These tests monkeypatch the upload + listing so they never touch the Hub.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue778_upload_corrected as upload


def _make_eval_root(tmp_path: Path) -> Path:
    """A tiny eval_results/ with the 6 primary-deliverable JSONLs (2 legs x 3 traits)."""
    eval_root = tmp_path / "eval_results" / "issue_778"
    eval_root.mkdir(parents=True)
    for tag in upload.MONITORING_JSONL_TAGS:
        for trait in ("evil", "sycophancy", "hallucination"):
            (eval_root / f"{tag}_{trait}.jsonl").write_text('{"condition_id": 0}\n')
    # a decoy that must NOT be uploaded (different stem)
    (eval_root / "unrelated_evil.jsonl").write_text("{}\n")
    return eval_root


def test_pod_phase_uploads_all_six_jsonls(tmp_path, monkeypatch):
    """All 6 monitoring JSONLs upload to the stable prefix; the decoy is excluded."""
    eval_root = _make_eval_root(tmp_path)
    out_root = tmp_path / "data" / "issue_778"  # empty: no acts / pools to upload
    out_root.mkdir(parents=True)

    uploaded: list[str] = []

    def _fake_upload(local: Path, dest: str) -> None:
        uploaded.append(dest)

    # Fresh listing echoes back every uploaded basename under the prefix (success).
    def _fake_list(api, repo_id, *, repo_type="model", revision=None):
        return uploaded

    monkeypatch.setattr(upload, "_upload_file", _fake_upload)
    monkeypatch.setattr(upload, "list_repo_files_complete", _fake_list)

    summary = upload.upload_pod_phase(
        out_root,
        eval_root,
        "issue778_persona_vectors",
        traits=["evil", "sycophancy", "hallucination"],
    )

    jsonl_dests = [d for d in uploaded if "/followup_corrected/eval_jsonl/" in d]
    assert len(jsonl_dests) == 6, jsonl_dests
    assert not any("unrelated_evil" in d for d in jsonl_dests), "decoy must not upload"
    assert summary["eval_jsonl"]["n_uploaded"] == 6
    assert summary["eval_jsonl"]["prefix"].endswith(upload.EVAL_JSONL_SUBPREFIX)
    # Every basename recorded in the summary.
    assert set(summary["eval_jsonl"]["basenames"]) == {
        f"{tag}_{trait}.jsonl"
        for tag in upload.MONITORING_JSONL_TAGS
        for trait in ("evil", "sycophancy", "hallucination")
    }


def test_pod_phase_raises_when_fresh_listing_missing_a_jsonl(tmp_path, monkeypatch):
    """The integration assert fails loud if a fresh Hub listing is MISSING an upload."""
    eval_root = _make_eval_root(tmp_path)
    out_root = tmp_path / "data" / "issue_778"
    out_root.mkdir(parents=True)

    monkeypatch.setattr(upload, "_upload_file", lambda local, dest: None)
    # Fresh listing returns only ONE of the six under the prefix -> 5 missing -> raise.
    monkeypatch.setattr(
        upload,
        "list_repo_files_complete",
        lambda *a, **k: [
            "issue778_persona_vectors/followup_corrected/eval_jsonl/monitoring_corrected_evil.jsonl"
        ],
    )

    with pytest.raises(RuntimeError, match="eval-JSONL promotion verify FAILED"):
        upload.upload_pod_phase(
            out_root,
            eval_root,
            "issue778_persona_vectors",
            traits=["evil", "sycophancy", "hallucination"],
        )


def test_pod_phase_no_jsonls_raises_before_upload(tmp_path, monkeypatch):
    """An eval_root MISSING every expected monitoring JSONL FAILS LOUD before upload.

    Round-2 BLOCKER jsonl-promotion-completeness: the parent codified a missing-
    deliverable as a non-fatal warning, so the loss surfaced only off-pod after
    teardown. It must now raise BEFORE any upload (never touch _upload_file).
    """
    eval_root = tmp_path / "eval_results" / "issue_778"
    eval_root.mkdir(parents=True)
    out_root = tmp_path / "data" / "issue_778"
    out_root.mkdir(parents=True)

    uploaded: list[str] = []
    monkeypatch.setattr(upload, "_upload_file", lambda local, dest: uploaded.append(dest))
    monkeypatch.setattr(upload, "list_repo_files_complete", lambda *a, **k: [])

    with pytest.raises(RuntimeError, match=r"eval-JSONL promotion FAILED \(pre-upload\)"):
        upload.upload_pod_phase(
            out_root,
            eval_root,
            "issue778_persona_vectors",
            traits=["evil", "sycophancy", "hallucination"],
        )
    assert uploaded == [], "must fail loud BEFORE uploading anything"


def test_pod_phase_raises_when_a_required_local_jsonl_missing(tmp_path, monkeypatch):
    """Only one of the expected JSONLs present -> fail loud pre-upload (Codex-named).

    traits=["evil","sycophancy"] with only ``monitoring_corrected_evil.jsonl`` present
    (the other 3 expected — corrected_sycophancy + both manyshot — absent) must raise
    BEFORE returning and BEFORE uploading, so a silently-skipped trait / wrong root
    can never warn-and-continue into a post-teardown null-battery FileNotFoundError.
    """
    eval_root = tmp_path / "eval_results" / "issue_778"
    eval_root.mkdir(parents=True)
    (eval_root / "monitoring_corrected_evil.jsonl").write_text('{"condition_id": 0}\n')
    out_root = tmp_path / "data" / "issue_778"
    out_root.mkdir(parents=True)

    uploaded: list[str] = []
    monkeypatch.setattr(upload, "_upload_file", lambda local, dest: uploaded.append(dest))
    monkeypatch.setattr(upload, "list_repo_files_complete", lambda *a, **k: [])

    with pytest.raises(RuntimeError, match=r"eval-JSONL promotion FAILED \(pre-upload\)"):
        upload.upload_pod_phase(
            out_root,
            eval_root,
            "issue778_persona_vectors",
            traits=["evil", "sycophancy"],
        )
    assert uploaded == [], "must fail loud BEFORE uploading the one present JSONL"


def test_pod_phase_smoke_slice_evil_only_uploads_two(tmp_path, monkeypatch):
    """The smoke slice (traits=['evil']) expects exactly 2 JSONLs (2 tags x 1 trait)."""
    eval_root = tmp_path / "eval_results" / "issue_778"
    eval_root.mkdir(parents=True)
    for tag in upload.MONITORING_JSONL_TAGS:
        (eval_root / f"{tag}_evil.jsonl").write_text('{"condition_id": 0}\n')
    out_root = tmp_path / "data" / "issue_778"
    out_root.mkdir(parents=True)

    uploaded: list[str] = []
    monkeypatch.setattr(upload, "_upload_file", lambda local, dest: uploaded.append(dest))
    monkeypatch.setattr(upload, "list_repo_files_complete", lambda *a, **k: uploaded)

    summary = upload.upload_pod_phase(
        out_root, eval_root, "issue778_persona_vectors", traits=["evil"]
    )
    jsonl_dests = [d for d in uploaded if "/followup_corrected/eval_jsonl/" in d]
    assert len(jsonl_dests) == 2, jsonl_dests
    assert summary["eval_jsonl"]["n_uploaded"] == 2
    assert set(summary["eval_jsonl"]["basenames"]) == {
        "monitoring_corrected_evil.jsonl",
        "monitoring_manyshot_evil.jsonl",
    }
