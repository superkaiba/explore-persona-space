# ruff: noqa: E402
"""HF public-LFS-quota (#541/#552) overflow-routing for the issue #841 scaling round.

Attempt 5's science ran end-to-end (capture completed all 96k rows) then the terminal
shard upload hit the account-wide HF PUBLIC-storage quota 403 (the LFS wall). This round
routes LFS artifacts (.pt shards + maps) to the PRIVATE overflow repo (separate quota,
validated) while non-LFS files (manifests, .done.json) + an OVERFLOW_POINTER.json stay on
the canonical PUBLIC repo. These tests pin the split + the overflow-aware fetch:

  * upload_split_lfs_to_overflow: .pt -> private overflow (private=True), non-.pt -> public,
    pointer written once, fail-loud on any verified-upload miss;
  * single-file input (the stage0 per-map case) routes the .pt to overflow;
  * hf_download_pt_maybe_overflow: .pt fetched from overflow when the pointer is present,
    from public when absent; non-.pt always from public.

They exercise the LIVE dispatched helpers `issue841_scaling_common.upload_split_lfs_to_overflow`
/ `hf_download_pt_maybe_overflow` (the ones capture.py / stage0.py / stage1.py call), per the
"verification gates test the live dispatched path" rule.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import issue841_scaling_common as S
import pytest

from explore_persona_space.orchestrate import hub


def _mk_capture_dir(tmp_path):
    (tmp_path / "cx_last_shard000.pt").write_bytes(b"\x00" * 16)
    (tmp_path / "cx_last_shard001.pt").write_bytes(b"\x00" * 16)
    (tmp_path / "manifest.json").write_text("{}")
    (tmp_path / "cx_last_shard000.done.json").write_text("{}")
    return tmp_path


def test_split_routes_lfs_to_overflow_nonlfs_to_public(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        hub,
        "_upload",
        lambda f, repo, rt, dest, **kw: (
            calls.append(
                {
                    "name": Path(f).name,
                    "repo": repo,
                    "dest": dest,
                    "private": kw.get("private", False),
                }
            )
            or f"{repo}/{dest}"
        ),
    )
    pointers = []
    monkeypatch.setattr(
        S, "_write_overflow_pointer_dataset", lambda c, p, o, r: pointers.append((c, p, o, r))
    )
    d = _mk_capture_dir(tmp_path)
    dev = S.upload_split_lfs_to_overflow(d, "issue841_scaling/cx_last_shards")

    pt = [c for c in calls if c["name"].endswith(".pt")]
    nonpt = [c for c in calls if not c["name"].endswith(".pt")]
    assert {c["repo"] for c in pt} == {S.OVERFLOW_REPO}, pt
    assert all(c["private"] for c in pt), pt  # overflow repo created PRIVATE
    assert {c["repo"] for c in nonpt} == {S.C.HF_DATA_REPO}, nonpt  # non-LFS stays public
    assert all(not c["private"] for c in nonpt), nonpt
    assert all(c["dest"].startswith("issue841_scaling/cx_last_shards/") for c in calls)
    assert len(pointers) == 1 and pointers[0][2] == S.OVERFLOW_REPO
    assert dev["n_lfs"] == 2 and dev["n_nonlfs"] == 2 and dev["overflow_repo"] == S.OVERFLOW_REPO


def test_split_single_map_file_to_overflow(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        hub,
        "_upload",
        lambda f, repo, rt, dest, **kw: calls.append((repo, dest)) or f"{repo}/{dest}",
    )
    monkeypatch.setattr(S, "_write_overflow_pointer_dataset", lambda *a: None)
    f = tmp_path / "ridge_maps_n4000.pt"
    f.write_bytes(b"\x00" * 16)
    dev = S.upload_split_lfs_to_overflow(f, "issue841_scaling/ridge_maps_n4000")
    assert calls == [(S.OVERFLOW_REPO, "issue841_scaling/ridge_maps_n4000/ridge_maps_n4000.pt")]
    assert dev["n_lfs"] == 1 and dev["n_nonlfs"] == 0


def test_split_fail_loud_on_upload_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(hub, "_upload", lambda *a, **kw: "")  # verification miss -> ""
    monkeypatch.setattr(S, "_write_overflow_pointer_dataset", lambda *a: None)
    f = tmp_path / "cx_last_shard000.pt"
    f.write_bytes(b"\x00" * 16)
    with pytest.raises(RuntimeError, match="overflow upload failed"):
        S.upload_split_lfs_to_overflow(f, "issue841_scaling/cx_last_shards")


def test_split_fail_loud_on_pointer_write_miss(tmp_path, monkeypatch):
    """The OVERFLOW_POINTER.json is LOAD-BEARING (the fetch path reads it to locate the
    rerouted .pt); a silently-failed pointer write (upload returns "") must NOT be
    reported as a rerouted-LFS success — the helper must RAISE (Codex #841 v11 review)."""

    def fake_upload(f, repo, rt, dest, **kw):  # .pt uploads fine; the pointer write MISSES
        return "" if dest.endswith("OVERFLOW_POINTER.json") else f"{repo}/{dest}"

    monkeypatch.setattr(hub, "_upload", fake_upload)
    f = tmp_path / "cx_last_shard000.pt"
    f.write_bytes(b"\x00" * 16)
    with pytest.raises(RuntimeError, match="pointer write"):
        S.upload_split_lfs_to_overflow(f, "issue841_scaling/cx_last_shards")


def test_split_fail_loud_on_pointer_write_raise(tmp_path, monkeypatch):
    """Same load-bearing invariant when the pointer write RAISES (not returns "") — the
    exception must propagate, not be swallowed."""

    def fake_upload(f, repo, rt, dest, **kw):
        if dest.endswith("OVERFLOW_POINTER.json"):
            raise RuntimeError("simulated pointer upload crash")
        return f"{repo}/{dest}"

    monkeypatch.setattr(hub, "_upload", fake_upload)
    f = tmp_path / "cx_last_shard000.pt"
    f.write_bytes(b"\x00" * 16)
    with pytest.raises(RuntimeError, match="crash"):
        S.upload_split_lfs_to_overflow(f, "issue841_scaling/cx_last_shards")


def test_hf_download_pt_prefers_overflow_when_pointer_present(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(S, "_overflow_repo_for_bucket", lambda *a, **kw: S.OVERFLOW_REPO)
    seen = {}
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda repo, filename, repo_type: seen.update(repo=repo) or "/tmp/x",
    )
    S.hf_download_pt_maybe_overflow(
        S.C.HF_DATA_REPO, "issue841_scaling/cx_last_shards", "cx_last_shard007.pt"
    )
    assert seen["repo"] == S.OVERFLOW_REPO  # .pt fetched from overflow


def test_hf_download_pt_public_when_no_pointer(monkeypatch):
    import huggingface_hub

    monkeypatch.setattr(S, "_overflow_repo_for_bucket", lambda *a, **kw: None)
    seen = {}
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        lambda repo, filename, repo_type: seen.update(repo=repo) or "/tmp/x",
    )
    S.hf_download_pt_maybe_overflow(
        S.C.HF_DATA_REPO, "issue841_scaling/cx_last_shards", "cx_last_shard007.pt"
    )
    assert seen["repo"] == S.C.HF_DATA_REPO  # no pointer -> public
    # a non-.pt file never probes overflow (always public)
    S.hf_download_pt_maybe_overflow(
        S.C.HF_DATA_REPO, "issue841_scaling/cx_last_shards", "manifest.json"
    )
    assert seen["repo"] == S.C.HF_DATA_REPO
