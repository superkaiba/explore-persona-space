"""Issue #1482 r6 pins: the scratch-reconstruction invariants (fail-loud verify,
per-chunk ci checkpoint/resume with the raw JSON deleted) and the off-pod
`_require_scratch` fail-loud-with-recipe guard. CPU-only; no network."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue1482_analysis as A  # noqa: E402
import issue1482_reconstruct_scratch as R  # noqa: E402


def test_verify_mismatch_raises_and_names_both_sides():
    with pytest.raises(RuntimeError, match=r"holdout\.sha256.*deadbeef.*cafe"):
        R._verify("holdout.sha256", "deadbeef", "cafe")
    R._verify("n_total", 5, 5)  # equal → no raise


def test_chunk_ci_real_body_caches_resumes_and_deletes_raw(tmp_path, monkeypatch):
    """Executes the REAL _chunk_ci body; fakes ONLY the network boundary with a
    signature-mirrored downloader that writes a real raw-chunk JSON."""
    calls = {"n": 0}

    def fake_download(repo: str, filename: str, local_dir: Path) -> str:
        # mirrors N1M._download_chunk_with_retry(repo, filename, local_dir) -> str
        calls["n"] += 1
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps({"rows": [{"ci": 7, "prompt": "p", "response": "r"}, {"ci": 3}]})
        )
        return str(dest)

    monkeypatch.setattr(N1M, "_download_chunk_with_retry", fake_download)
    cache_dir = tmp_path / "raw"
    ci_dir = tmp_path / "ci"
    cache_dir.mkdir()
    ci_dir.mkdir()
    ci = R._chunk_ci("shard00_chunk0000.json", cache_dir, ci_dir)
    assert ci.tolist() == [7, 3] and ci.dtype == np.int64
    # raw JSON deleted (digest-only), ci checkpoint persisted
    assert not list(cache_dir.rglob("*.json"))
    assert (ci_dir / "shard00_chunk0000.json.ci.npy").exists()
    # second call resumes from the checkpoint — downloader NOT called again
    ci2 = R._chunk_ci("shard00_chunk0000.json", cache_dir, ci_dir)
    assert np.array_equal(ci, ci2) and calls["n"] == 1


def test_require_scratch_fails_loud_with_recovery_recipe(tmp_path):
    args = SimpleNamespace(scratch=tmp_path, phase="judge")
    with pytest.raises(SystemExit, match=r"issue1482_reconstruct_scratch\.py"):
        A._require_scratch(args)
    for n in A._SCRATCH_FILES:
        (tmp_path / n).write_bytes(b"x")
    A._require_scratch(args)  # all present → no raise


def test_reconstruct_refuses_smoke_anchor(tmp_path):
    (tmp_path / "split_1482.json").write_text(
        json.dumps({"regime": {"smoke": True, "max_chunks": 1}})
    )
    args = SimpleNamespace(
        scratch=tmp_path / "scratch", out_eval=tmp_path, work=tmp_path / "w", max_chunks=0
    )
    with pytest.raises(AssertionError, match="smoke"):
        R.reconstruct(args)


def test_phase_p4_uploads_scratch_meta_real_body(tmp_path, monkeypatch):
    """Executes the REAL phase_p4 body (r6 durable fix): the three scratch metadata
    files join the P4 upload set under analysis_tensors/scratch_meta/, and a missing
    scratch file fails loud BEFORE any upload. Fakes ONLY the Hub boundary with
    signature-mirrored fakes (+ the sentinel writer, a filesystem boundary)."""
    import issue1482_error_analysis as D

    from explore_persona_space.orchestrate import hub

    store = tmp_path / "store"
    store.mkdir()
    (store / "pooled_000.npz").write_bytes(b"x")
    out_eval = tmp_path / "eval"
    (out_eval / "percontext").mkdir(parents=True)
    (out_eval / "percontext" / "refit_full__ridge__seed0.npz").write_bytes(b"x")
    (out_eval / "split_1482.json").write_text("{}")
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    for n in ("split_indices.npz", "row_ci.npy", "prov.npy"):
        (scratch / n).write_bytes(b"x")
    uploaded: list[str] = []

    def fake_upload(local_path, repo_id, repo_type=None, path_in_repo=None, upload_as_file=False):
        uploaded.append(path_in_repo)
        return f"https://hf.co/{path_in_repo}"

    def fake_verify(api, repo_id, expected, path_in_repo=None, repo_type=None):
        return set()

    monkeypatch.setattr(hub, "_upload", fake_upload)
    monkeypatch.setattr(hub, "verify_repo_paths_uploaded", fake_verify)
    monkeypatch.setattr(D, "_phase_sentinel", lambda name, note, extra=None: None)
    args = SimpleNamespace(
        skip_upload=False,
        smoke=False,
        hf_prefix="issue1482_error_analysis",
        store=store,
        out_eval=out_eval,
        scratch=scratch,
    )
    D.phase_p4(args)
    assert [p for p in uploaded if "scratch_meta" in p] == [
        "issue1482_error_analysis/analysis_tensors/scratch_meta/split_indices.npz",
        "issue1482_error_analysis/analysis_tensors/scratch_meta/row_ci.npy",
        "issue1482_error_analysis/analysis_tensors/scratch_meta/prov.npy",
    ]
    (scratch / "prov.npy").unlink()
    with pytest.raises(RuntimeError, match="scratch metadata missing"):
        D.phase_p4(args)
