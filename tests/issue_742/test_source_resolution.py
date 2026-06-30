"""Test 6 (plan v7 §13 NEW per Phase-2 REVISE + §4 Stage-0 step 2a + §10 MF2 +
§11 row 15(h)) — the Stage-0 raw-completion source-resolution path snapshots the
HF-resolved completions into an issue-owned dir with a sha256, and FAILS LOUD on
a short / missing cache cell BEFORE the judge batch builds.

Plan §4 Stage-0 step 2a: the full completions live ONLY on the HF data repo at
per-genre paths; fetch via huggingface_hub, snapshot under
data/issue_742/inputs/raw_completions/<genre>/, content-identity check, fail loud
on short / missing cache. The judge rerun must never silently build on a partial
snapshot.

These tests MOCK the HF download (round 1 mocks the loader's hf_hub_download
dependency — no network, no real fetch) and assert the loader's snapshot + sha256
+ shortfall-raise contract.
"""

from __future__ import annotations

import hashlib
import json

import pytest

from .conftest import impl, impl_has


def _write_completion_file(path, context_id, n_completions, *, probe="p", behavior="sycophancy"):
    """Write a fixture raw-completion file in the documented shape:
    {context_id, completions:[{probe, completion}, ...]}."""
    obj = {
        "context_id": context_id,
        "behavior": behavior,
        "completions": [
            {"probe": f"{probe}{i}", "completion": f"synthetic completion {i}"}
            for i in range(n_completions)
        ],
    }
    path.write_text(json.dumps(obj))
    return obj


@pytest.mark.skipif(
    not impl_has("snapshot_raw_completions"),
    reason="implementation pending round 2",
)
def test_snapshot_resolves_into_issue_owned_dir_with_sha256(tmp_path, monkeypatch):
    # Build a fake HF cache the mocked hf_hub_download returns files from.
    fake_hf_cache = tmp_path / "hf_cache"
    fake_hf_cache.mkdir()
    src = fake_hf_cache / "f1_house_persona__sycophancy.json"
    _write_completion_file(src, context_id="ctx_0", n_completions=20)

    def fake_hf_download(*, repo_id, filename, repo_type, **kwargs):
        # the loader requests a per-genre raw_completions path; return our fixture
        assert repo_type == "dataset"
        assert "raw_completions" in filename
        return str(src)

    dest = tmp_path / "data" / "issue_742" / "inputs" / "raw_completions"
    manifest = impl.snapshot_raw_completions(
        "betley",
        dest_dir=dest,
        hf_download_fn=fake_hf_download,
        rerun_probe_set_size=20,
    )

    # the file was snapshotted under the issue-owned dest dir
    snapped = list(dest.rglob("*.json"))
    assert snapped, "no completion file snapshotted into the issue-owned dir"

    # a sha256 was recorded in the manifest and matches the snapshotted bytes
    assert manifest, "snapshot_raw_completions returned an empty manifest"
    rec = next(iter(manifest.values())) if isinstance(manifest, dict) else manifest[0]
    recorded_sha = rec["sha256"] if isinstance(rec, dict) else rec.sha256
    snapped_file = snapped[0]
    actual_sha = hashlib.sha256(snapped_file.read_bytes()).hexdigest()
    assert recorded_sha == actual_sha, (
        "manifest sha256 does not match the snapshotted file's bytes (content-identity broken)"
    )


@pytest.mark.skipif(
    not (impl_has("snapshot_raw_completions") and impl_has("RawCompletionShortfallError")),
    reason="implementation pending round 2",
)
def test_short_cache_raises_RawCompletionShortfallError_before_judge_batch(tmp_path):
    # A context cell with FEWER than J=20 completions for the rerun probe-set must
    # fail loud with a NAMED error before the judge batch builds, naming the
    # offending (context_id, behavior, count) triple (plan §4 Stage-0 step 2a).
    fake_hf_cache = tmp_path / "hf_cache"
    fake_hf_cache.mkdir()
    short = fake_hf_cache / "f1_house_persona__sycophancy.json"
    _write_completion_file(short, context_id="ctx_short", n_completions=7)  # < 20

    def fake_hf_download(*, repo_id, filename, repo_type, **kwargs):
        return str(short)

    dest = tmp_path / "data" / "issue_742" / "inputs" / "raw_completions"
    with pytest.raises(impl.RawCompletionShortfallError) as ei:
        impl.snapshot_raw_completions(
            "betley",
            dest_dir=dest,
            hf_download_fn=fake_hf_download,
            rerun_probe_set_size=20,
        )
    # the error message names the offending triple
    msg = str(ei.value)
    assert "ctx_short" in msg, "shortfall error must name the offending context_id"
    assert "sycophancy" in msg, "shortfall error must name the offending behavior"
    assert "7" in msg, "shortfall error must name the actual completion count"


@pytest.mark.skipif(
    not (impl_has("snapshot_raw_completions") and impl_has("RawCompletionShortfallError")),
    reason="implementation pending round 2",
)
def test_missing_cache_cell_also_fails_loud(tmp_path):
    # a completely missing cell (hf_hub_download raises / returns nothing) must
    # also fail loud, never silently skip the cell.
    dest = tmp_path / "data" / "issue_742" / "inputs" / "raw_completions"

    def fake_hf_download_missing(*, repo_id, filename, repo_type, **kwargs):
        raise FileNotFoundError(f"{filename} not on the HF repo")

    with pytest.raises(Exception):  # noqa: B017 - any loud error; never a silent skip
        impl.snapshot_raw_completions(
            "betley",
            dest_dir=dest,
            hf_download_fn=fake_hf_download_missing,
            rerun_probe_set_size=20,
        )
