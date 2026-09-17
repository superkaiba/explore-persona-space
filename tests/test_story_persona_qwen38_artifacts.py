"""Persistence and plotted-value integrity checks without network or model downloads."""

import json

import numpy as np
import pytest
from huggingface_hub.hf_api import RepoFile

from scripts.story_persona_qwen38_artifacts import (
    LABELS,
    PREFIX,
    inventory,
    prepare,
    validate_inputs,
    verify_hub_entries,
)
from scripts.story_persona_qwen38_pilot import digest, file_digest, write_json


def make_capture(out):
    """Build a tiny explicitly synthetic capture solely to test artifact validation."""
    (out / "chunks").mkdir(parents=True)
    names = list(LABELS)
    rows = [{"persona": name, "question_id": q} for name in names for q in (0, 1)]
    spec = {
        "prompts": [{"id": n, "primary": i < 6} for i, n in enumerate(names)],
        "question_ids": [0, 1],
        "inputs_sha256": digest(rows),
        "model": {"hidden_dim": 6},
        "batches": [list(range(20))],
    }
    fingerprint = digest(spec)
    write_json(
        out / "manifest.json",
        {"fingerprint": fingerprint, "spec": spec, "provenance": {"git_commit": "a" * 40}},
    )
    write_json(out / "rows.json", rows)
    chunk = out / "chunks/batch_0000.pt"
    chunk.write_bytes(b"synthetic test fixture; never interpreted as captured activations")
    write_json(
        out / "capture_complete.json",
        {
            "fingerprint": fingerprint,
            "row_count": 20,
            "chunk_sha256": {chunk.name: file_digest(chunk)},
        },
    )
    centroids = np.random.default_rng(0).normal(size=(10, 64, 6))
    centered = centroids - centroids.mean(0, keepdims=True)
    centered /= np.linalg.norm(centered, axis=-1, keepdims=True)
    summary = {
        "fingerprint": fingerprint,
        "persona_names": names,
        "layers": list(range(64)),
        "question_count": 2,
        "centered_cosine": np.einsum("plh,qlh->lpq", centered, centered).tolist(),
        "interpretation": "Synthetic unit-test fixture only.",
    }
    write_json(out / "summary.json", summary)
    np.savez(out / "centroids.npz", centroids=centroids)
    return summary


def test_prepare_binds_plots_to_centroids_and_preserves_every_output(tmp_path):
    """The real rendering route produces every declared Git artifact and provenance."""
    out, repo = tmp_path / "capture", tmp_path / "repo"
    make_capture(out)
    artifacts = prepare(out, repo)
    assert set(artifacts["files"]) == set(inventory(out))
    figures = list((repo / "figures/issue_2673").iterdir())
    assert len(figures) == 8
    assert len([p for p in figures if p.suffix == ".pdf"]) == 2
    meta = json.loads(
        (repo / "figures/issue_2673/centered_cosine_fixed_layers.meta.json").read_text()
    )
    assert meta["summary_sha256"] == file_digest(out / "summary.json")
    summary = json.loads((out / "summary.json").read_text())
    summary["centered_cosine"][15][0][1] += 0.1
    write_json(out / "summary.json", summary)
    with pytest.raises(RuntimeError, match="differs from persisted centroids"):
        validate_inputs(out)


def test_capture_content_and_row_coverage_cannot_be_replaced(tmp_path):
    """A file presence check must never pass a finite edit or missing row."""
    make_capture(tmp_path)
    (tmp_path / "chunks/batch_0000.pt").write_bytes(b"changed but present")
    with pytest.raises(RuntimeError, match="corrupt completed chunk"):
        validate_inputs(tmp_path)
    rows = json.loads((tmp_path / "rows.json").read_text())
    write_json(tmp_path / "rows.json", rows[:-1])
    with pytest.raises(RuntimeError, match="row identity"):
        validate_inputs(tmp_path)


def test_hub_verification_requires_exact_set_size_and_hash(tmp_path):
    """Exercise both ordinary Git files and LFS tensors using actual Hub metadata types."""
    (tmp_path / "rows.json").write_text("[]")
    (tmp_path / "tensor.pt").write_bytes(b"unit-test tensor")
    expected = inventory(tmp_path)
    entries = [
        RepoFile(
            path=f"{PREFIX}/rows.json",
            size=expected["rows.json"]["size"],
            oid=expected["rows.json"]["git_blob_sha1"],
        ),
        RepoFile(
            path=f"{PREFIX}/tensor.pt",
            size=expected["tensor.pt"]["size"],
            oid="b" * 40,
            lfs={
                "oid": expected["tensor.pt"]["sha256"],
                "size": expected["tensor.pt"]["size"],
                "pointerSize": 130,
            },
        ),
    ]
    verify_hub_entries(entries, expected, PREFIX)
    with pytest.raises(RuntimeError, match="file-set"):
        verify_hub_entries(entries[:1], expected, PREFIX)
    entries[0].size += 1
    with pytest.raises(RuntimeError, match="size mismatch"):
        verify_hub_entries(entries, expected, PREFIX)
    entries[0].size -= 1
    entries[0].blob_id = "c" * 40
    with pytest.raises(RuntimeError, match="content hash"):
        verify_hub_entries(entries, expected, PREFIX)


def test_inventory_rejects_symlink_and_interrupted_write(tmp_path):
    """An unexpected file must fail before upload eligibility could silently omit it."""
    target = tmp_path / "real.json"
    target.write_text("{}")
    link = tmp_path / "link.json"
    link.symlink_to(target)
    with pytest.raises(RuntimeError, match="symlink"):
        inventory(tmp_path)
    link.unlink()
    (tmp_path / "result.tmp.npz").write_bytes(b"interrupted")
    with pytest.raises(RuntimeError, match="unfinished"):
        inventory(tmp_path)


def test_wandb_aliases_preserve_the_real_files(tmp_path):
    """Known WandB aliases are disclosed without uploading duplicate symlink targets."""
    run = tmp_path / "wandb/offline-run-test"
    run.mkdir(parents=True)
    (run / "debug.log").write_text("test log")
    (tmp_path / "wandb/latest-run").symlink_to(run, target_is_directory=True)
    (tmp_path / "wandb/debug.log").symlink_to(run / "debug.log")
    assert set(inventory(tmp_path)) == {"wandb/offline-run-test/debug.log"}


def test_git_result_push_verifies_remote_bytes(tmp_path):
    """Exercise the actual commit/push/blob verifier against an isolated local bare repo."""
    import subprocess

    from scripts.story_persona_qwen38_artifacts import BRANCH, git, push_results

    bare, repo = tmp_path / "remote.git", tmp_path / "checkout"
    subprocess.run(["git", "init", "--bare", str(bare)], check=True, capture_output=True)
    subprocess.run(["git", "clone", str(bare), str(repo)], check=True, capture_output=True)
    git(repo, "config", "user.email", "unit-test@example.invalid")
    git(repo, "config", "user.name", "Unit test")
    git(repo, "checkout", "-b", BRANCH)
    (repo / "initial.txt").write_text("fixture")
    git(repo, "add", "initial.txt")
    git(repo, "commit", "-m", "Initial test fixture")
    git(repo, "push", "-u", "origin", BRANCH)
    result = repo / "eval_results/issue_2673/summary.json"
    result.parent.mkdir(parents=True)
    result.write_text('{"verified": true}')
    revision = push_results(repo, [result])
    assert (
        git(repo, "show", f"{revision}:eval_results/issue_2673/summary.json").stdout
        == result.read_text()
    )
    assert push_results(repo, [result]) == revision
