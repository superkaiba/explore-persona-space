"""Issue #664 POST-PIVOT round-2 invariant pins.

Pins the one substantive blocker fixed this round so a future refactor cannot
silently strip it (the un-CI-pinned-assertion class):

- **Baseline-propensity raw completions are uploaded before pod teardown.**
  ``_write_baseline_propensity`` (phase0) writes the source-side BASE-model
  behavior-rate covariate (plan §4) -- the per-(source, behavior) judged-rate
  aggregate + raw base completions -- to the pod-local ``onpolicy_cache``.
  Pre-fix, ``upload_artifacts`` uploaded ONLY the cells-path raw completions +
  store tensors and never touched this cache, so the covariate did NOT survive
  pod teardown and Phase-3/4 could not derive the base-rate covariate (the
  #521-class trap). This module pins the new ``_upload_baseline_propensity``
  call site (AST) + its upload behaviour (mocked HF Hub) + the canonical prefix.

All CPU-only: imports ``scripts/issue664_*`` and exercises the pure-Python
logic, stubbing the single HF/Hub touch points so no GPU / network is required.
"""

from __future__ import annotations

import ast
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue664_common as C  # noqa: E402
import issue664_dispatch as D  # noqa: E402


# ── AST: upload_artifacts calls _upload_baseline_propensity exactly once ───────
def test_upload_artifacts_calls_baseline_propensity_uploader_once() -> None:
    """``upload_artifacts`` MUST invoke ``_upload_baseline_propensity`` exactly
    once (alongside ``_upload_raw_completions`` + ``_upload_store_tensors``).
    AST-pinned so the covariate-upload wiring cannot be silently dropped."""
    tree = ast.parse(Path(D.__file__).read_text())
    upload_fn = next(
        (
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == "upload_artifacts"
        ),
        None,
    )
    assert upload_fn is not None, "upload_artifacts FunctionDef not found in issue664_dispatch"
    called = [
        n.func.id
        for n in ast.walk(upload_fn)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    ]
    assert called.count("_upload_baseline_propensity") == 1, (
        "upload_artifacts must call _upload_baseline_propensity exactly once, "
        f"found {called.count('_upload_baseline_propensity')}"
    )
    # the sibling uploaders are still wired (regression guard for the whole batch).
    assert "_upload_raw_completions" in called
    assert "_upload_store_tensors" in called


# ── behaviour: stages a fake cache + asserts both files upload to the prefix ───
def test_upload_artifacts_uploads_baseline_propensity(tmp_path: Path, monkeypatch) -> None:
    """With a mocked HF Hub, stage a fake ``onpolicy_cache/baseline_propensity.json``
    + ``onpolicy_cache/baseline_raw/sycophancy__librarian.json`` and assert the
    uploader actually attempts BOTH uploads under the canonical
    ``issue664/baseline_propensity/`` prefix, then verifies on a fresh listing."""
    # CACHE_ROOT is computed at import (C.DATA_ROOT / "onpolicy_cache") so it must
    # be monkeypatched on the dispatch module directly, not via C.DATA_ROOT.
    cache_root = tmp_path / "onpolicy_cache"
    raw_root = cache_root / "baseline_raw"
    raw_root.mkdir(parents=True)
    monkeypatch.setattr(D, "CACHE_ROOT", cache_root)

    cell = C.Cell("sycophancy", "librarian", "contra", "d1")  # a CONTENT_BEHAVIORS cell
    agg = cache_root / "baseline_propensity.json"
    agg.write_text(json.dumps({"judged_rates": {"sycophancy": {"librarian": {"rate": 0.1}}}}))
    raw = raw_root / "sycophancy__librarian.json"
    raw.write_text(json.dumps({"behavior": "sycophancy", "source": "librarian", "rows": []}))
    # a judge save_raw sibling (also baseline_raw/*.json -> uploaded too).
    (raw_root / "sycophancy__librarian__scores.json").write_text(json.dumps({"scores": {}}))
    # the judge .cache subdir MUST be excluded from the upload set.
    (raw_root / ".cache").mkdir()
    (raw_root / ".cache" / "ignored.json").write_text("{}")

    uploaded: list[tuple[str, str]] = []  # (path_in_repo, local_name)

    def _fake_upload(local, *, repo_id, repo_type, path_in_repo, upload_as_file):
        assert repo_id == C.HF_DATA_REPO
        assert repo_type == "dataset"
        assert upload_as_file is True
        uploaded.append((path_in_repo, Path(local).name))

    def _fake_list(repo_id, repo_type="model", revision=None):
        # echo back exactly what was uploaded so the fresh-listing verify passes.
        return [p for p, _ in uploaded]

    import explore_persona_space.orchestrate.hub as hub_mod

    monkeypatch.setattr(hub_mod, "_upload", _fake_upload)
    monkeypatch.setattr("huggingface_hub.list_repo_files", _fake_list)

    D._upload_baseline_propensity([cell])

    prefix = C.HF_BASELINE_PROPENSITY_PREFIX
    assert prefix == "issue664/baseline_propensity"  # PIN the prefix Phase-3/4 relies on
    paths = {p for p, _ in uploaded}
    # the aggregate landed at the prefix root.
    assert f"{prefix}/baseline_propensity.json" in paths
    # the raw completions + judge scores landed under baseline_raw/.
    assert f"{prefix}/baseline_raw/sycophancy__librarian.json" in paths
    assert f"{prefix}/baseline_raw/sycophancy__librarian__scores.json" in paths
    # the judge .cache subdir was NOT uploaded (only direct *.json children).
    assert not any(".cache" in p for p in paths)
    assert len(uploaded) == 3


def test_upload_artifacts_baseline_propensity_missing_aggregate_raises(
    tmp_path: Path, monkeypatch
) -> None:
    """A missing ``baseline_propensity.json`` (phase0 base-prior read never ran)
    is FAIL-LOUD -- refuse to terminate without the registered covariate."""
    cache_root = tmp_path / "onpolicy_cache"
    cache_root.mkdir()
    monkeypatch.setattr(D, "CACHE_ROOT", cache_root)
    cell = C.Cell("sycophancy", "librarian", "contra", "d1")
    with pytest.raises(RuntimeError, match=r"baseline_propensity\.json MISSING"):
        D._upload_baseline_propensity([cell])


def test_upload_artifacts_baseline_propensity_missing_raw_raises(
    tmp_path: Path, monkeypatch
) -> None:
    """The aggregate exists but a SELECTED content-behavior cell's raw file is
    MISSING -> FAIL-LOUD (the #521-class trap variant: refuse to reach the
    Hub-verify step with an incomplete source-side covariate)."""
    cache_root = tmp_path / "onpolicy_cache"
    (cache_root / "baseline_raw").mkdir(parents=True)
    monkeypatch.setattr(D, "CACHE_ROOT", cache_root)
    (cache_root / "baseline_propensity.json").write_text("{}")
    cell = C.Cell("sycophancy", "librarian", "contra", "d1")  # raw file NOT written
    with pytest.raises(RuntimeError, match="raw completions MISSING"):
        D._upload_baseline_propensity([cell])
