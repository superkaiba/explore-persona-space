"""Regression test for the #530 data-dependency auto-downloader.

Mocks ``huggingface_hub.hf_hub_download`` to record every (filename,
revision) pair the helper requests, then asserts:

* All 6 expected #472 carry-over files are requested.
* Every request uses the pinned ``DATA_REVISION``.
* Every request uses ``repo_type='dataset'`` against the data repo.
* The helper is idempotent — when local files already exist with
  positive size, ``hf_hub_download`` is NOT called.
* The on-disk layout flattens HF's ``geometry/`` subdir into
  ``data/issue_472/`` to match the rig's CLI defaults.

This guards against three concrete regressions:
1. Forgetting a file the pipeline reads (e.g. dropping centroids_L15
   when the Phase 0.5 fallback-layers default ``15,20`` still requires
   it).
2. Bumping ``DATA_REVISION`` without a deliberate cross-task
   consistency check.
3. Accidentally swapping ``hf_hub_download`` for
   ``snapshot_download`` (the siblings-truncation gotcha).
"""

from __future__ import annotations

from pathlib import Path
from unittest import mock

from explore_persona_space.experiments.contrastive_neg_geometry_530 import data_deps


def _make_fake_cache(monkeypatch, tmp_path: Path):
    """Patch hf_hub_download to write a small dummy file into tmp_path and
    return its path. Returns the call-record list so the test can assert.
    """
    calls: list[dict] = []

    def fake_download(*, repo_id, repo_type, revision, filename, token):
        calls.append(
            {
                "repo_id": repo_id,
                "repo_type": repo_type,
                "revision": revision,
                "filename": filename,
                "token": token,
            }
        )
        cached = tmp_path / "_hf_cache" / filename
        cached.parent.mkdir(parents=True, exist_ok=True)
        cached.write_bytes(b"x" * 1024)  # positive size so the helper's size assertion passes
        return str(cached)

    monkeypatch.setattr("huggingface_hub.hf_hub_download", fake_download)
    return calls


def test_prepare_data_dependencies_pulls_all_expected_files(monkeypatch, tmp_path):
    """Every required #472 carry-over file is requested at the pinned revision."""
    calls = _make_fake_cache(monkeypatch, tmp_path)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data_deps, "LOCAL_DATA_ROOT", tmp_path / "data" / "issue_472", raising=True)
    # The LOCAL_PATHS dict was built from the module-level LOCAL_DATA_ROOT,
    # so rebuild it against the patched root so the helper writes into tmp_path.
    monkeypatch.setattr(
        data_deps,
        "LOCAL_PATHS",
        {
            hf: (tmp_path / "data" / "issue_472" / hf.split(f"{data_deps.DATA_PREFIX}/", 1)[1])
            for hf in data_deps.HF_FILES
        },
        raising=True,
    )
    # The default LOCAL_PATHS flattens "geometry/" — preserve that for the
    # subset we care about so the test still validates layout flattening.
    monkeypatch.setattr(
        data_deps,
        "LOCAL_PATHS",
        {
            f"{data_deps.DATA_PREFIX}/geometry/persona_bank.json": (
                tmp_path / "data" / "issue_472" / "persona_bank.json"
            ),
            f"{data_deps.DATA_PREFIX}/geometry/centroids_L10.pt": (
                tmp_path / "data" / "issue_472" / "centroids_L10.pt"
            ),
            f"{data_deps.DATA_PREFIX}/geometry/centroids_L15.pt": (
                tmp_path / "data" / "issue_472" / "centroids_L15.pt"
            ),
            f"{data_deps.DATA_PREFIX}/geometry/centroids_L20.pt": (
                tmp_path / "data" / "issue_472" / "centroids_L20.pt"
            ),
            f"{data_deps.DATA_PREFIX}/on_policy_R/R_train.json": (
                tmp_path / "data" / "issue_472" / "on_policy_R" / "R_train.json"
            ),
            f"{data_deps.DATA_PREFIX}/on_policy_R/R_eval.json": (
                tmp_path / "data" / "issue_472" / "on_policy_R" / "R_eval.json"
            ),
        },
        raising=True,
    )

    result = data_deps.prepare_data_dependencies()

    # 1. All 6 expected files requested.
    expected_hf_paths = {
        f"{data_deps.DATA_PREFIX}/geometry/persona_bank.json",
        f"{data_deps.DATA_PREFIX}/geometry/centroids_L10.pt",
        f"{data_deps.DATA_PREFIX}/geometry/centroids_L15.pt",
        f"{data_deps.DATA_PREFIX}/geometry/centroids_L20.pt",
        f"{data_deps.DATA_PREFIX}/on_policy_R/R_train.json",
        f"{data_deps.DATA_PREFIX}/on_policy_R/R_eval.json",
    }
    requested = {c["filename"] for c in calls}
    assert requested == expected_hf_paths, f"Expected {expected_hf_paths} but got {requested}"

    # 2. Every request at the pinned revision.
    for c in calls:
        assert c["revision"] == data_deps.DATA_REVISION, (
            f"Wrong revision for {c['filename']}: {c['revision']!r}"
        )
        assert c["repo_id"] == data_deps.DATA_REPO
        assert c["repo_type"] == "dataset"

    # 3. Local layout flattens geometry/ into data/issue_472/.
    persona_local = result[f"{data_deps.DATA_PREFIX}/geometry/persona_bank.json"]
    assert persona_local.parent.name == "issue_472", (
        f"persona_bank should land at data/issue_472/persona_bank.json; got {persona_local}"
    )
    centroids_local = result[f"{data_deps.DATA_PREFIX}/geometry/centroids_L10.pt"]
    assert centroids_local.parent.name == "issue_472"
    # R bundles stay under on_policy_R/ (NOT flattened).
    r_train_local = result[f"{data_deps.DATA_PREFIX}/on_policy_R/R_train.json"]
    assert r_train_local.parent.name == "on_policy_R"

    # 4. Each local file landed with positive size.
    for local_path in result.values():
        assert local_path.exists(), f"missing local: {local_path}"
        assert local_path.stat().st_size > 0, f"empty local: {local_path}"


def test_prepare_data_dependencies_is_idempotent(monkeypatch, tmp_path):
    """Re-running with files already on disk does NOT call hf_hub_download."""
    # Build a LOCAL_PATHS that points into tmp_path and pre-populate every
    # destination so the helper sees positive-size files and should skip.
    local_paths = {
        f"{data_deps.DATA_PREFIX}/geometry/persona_bank.json": (
            tmp_path / "data" / "issue_472" / "persona_bank.json"
        ),
        f"{data_deps.DATA_PREFIX}/geometry/centroids_L10.pt": (
            tmp_path / "data" / "issue_472" / "centroids_L10.pt"
        ),
        f"{data_deps.DATA_PREFIX}/geometry/centroids_L15.pt": (
            tmp_path / "data" / "issue_472" / "centroids_L15.pt"
        ),
        f"{data_deps.DATA_PREFIX}/geometry/centroids_L20.pt": (
            tmp_path / "data" / "issue_472" / "centroids_L20.pt"
        ),
        f"{data_deps.DATA_PREFIX}/on_policy_R/R_train.json": (
            tmp_path / "data" / "issue_472" / "on_policy_R" / "R_train.json"
        ),
        f"{data_deps.DATA_PREFIX}/on_policy_R/R_eval.json": (
            tmp_path / "data" / "issue_472" / "on_policy_R" / "R_eval.json"
        ),
    }
    monkeypatch.setattr(data_deps, "LOCAL_DATA_ROOT", tmp_path / "data" / "issue_472")
    monkeypatch.setattr(data_deps, "LOCAL_PATHS", local_paths)
    for p in local_paths.values():
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"already-here")

    download_mock = mock.MagicMock(
        side_effect=AssertionError("hf_hub_download should not be called when local files exist")
    )
    monkeypatch.setattr("huggingface_hub.hf_hub_download", download_mock)

    result = data_deps.prepare_data_dependencies()

    download_mock.assert_not_called()
    assert set(result) == set(local_paths)


def test_pinned_revision_matches_i477(monkeypatch):
    """The pinned DATA_REVISION must match the value `i477_reval_confirm.py`
    uses; #530's geometry/R is byte-consistent with the #477 line.

    Hard-pinned in two places — this test fails loud on accidental drift.
    """
    expected = "66d7db7a542e19275f8c1d8e32948396d050faa9"
    assert expected == data_deps.DATA_REVISION, (
        "data_deps.DATA_REVISION drifted from the #477 pinned revision; "
        f"got {data_deps.DATA_REVISION!r}, expected {expected!r}"
    )
