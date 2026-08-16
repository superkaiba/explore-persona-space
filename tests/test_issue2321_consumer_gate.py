"""Tests for the #2321 consumer-gate deliverables (plan §3.8 / §12, I17/MF5).

Covers the two migrated silent-empty consumers on REPACKED fixture trees
(`scripts/issue1090_fu3_yield_replay.py`, `scripts/issue1481_cjk_audit.py`)
plus the consumer-inventory scanner/gate tooling
(`scripts/issue2321_consumer_gate.py` + the committed inventory JSON).

Zero network: the Hub boundary is faked signature-conformantly (real
``RepoFile`` objects; ``hf_hub_download`` / ``HfApi`` monkeypatched at the
``huggingface_hub`` module the shim's lazy imports resolve — and at each
consumer module's own bound names), while the packs are REAL v2 packs built
by ``packing.pack_tree_v2``, so the consumer tests execute the production
listing + staging bodies end to end (the #906 body-coverage rule). The
conftest ``_i2321_test_mutation_interlock_env`` fixture pins
``HF_HUB_OFFLINE=1`` for this module (I18 defense-in-depth).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from huggingface_hub.hf_api import RepoFile
from huggingface_hub.utils import EntryNotFoundError

from explore_persona_space.orchestrate import hub, packing

REPO_ROOT = Path(__file__).resolve().parents[1]

CJK_CHARS = chr(0x4F60) + chr(0x597D)  # two Han chars inside scan()'s CJK range


def _load_script(filename: str, modname: str):
    """Load a scripts/ entrypoint as a module (scripts/ is not a package)."""
    spec = importlib.util.spec_from_file_location(modname, REPO_ROOT / "scripts" / filename)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


fu3 = _load_script("issue1090_fu3_yield_replay.py", "issue1090_fu3_yield_replay_mod")
cjk = _load_script("issue1481_cjk_audit.py", "issue1481_cjk_audit_mod")


# ---------------------------------------------------------------------------
# Signature-conformant Hub-boundary fakes (recipe: tests/test_hub_packed_fallback.py)
# ---------------------------------------------------------------------------


class FakeHfApi:
    """Signature-conformant HfApi fake serving a local dir as the repo."""

    def __init__(self, root: Path, token: str | None = None):
        self.root = Path(root)

    def file_exists(self, repo_id, filename, *, repo_type="model", revision=None):
        """Mirror of ``HfApi.file_exists`` against the local root."""
        return (self.root / filename).is_file()

    def repo_info(self, repo_id, *, repo_type="model"):
        """Mirror of ``HfApi.repo_info`` (only ``.sha`` is consumed)."""
        return SimpleNamespace(sha="deadbeefcafe")

    def list_repo_tree(
        self, repo_id, path_in_repo=None, *, repo_type="model", revision=None, recursive=False
    ):
        """Scoped walk yielding REAL ``RepoFile`` entries; absent path 404s."""
        base = self.root / (path_in_repo or "")
        if not base.exists():
            raise EntryNotFoundError(f"Entry Not Found: {path_in_repo}")
        for p in sorted(base.rglob("*")):
            if p.is_file():
                rel = p.relative_to(self.root).as_posix()
                yield RepoFile(path=rel, size=p.stat().st_size, oid="0" * 40)


def _fake_hf_hub_download(root: Path):
    """Signature-mirroring ``hf_hub_download`` fake copying from ``root``."""

    def fake(
        repo_id=None,
        filename=None,
        *,
        repo_type="model",
        revision=None,
        local_dir=None,
        token=None,
    ):
        src = Path(root) / filename
        if not src.is_file():
            raise EntryNotFoundError(f"Entry Not Found for url: {filename}")
        dest = Path(local_dir) / filename
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(src.read_bytes())
        return str(dest)

    return fake


def _packed_remote(tmp_path: Path, prefix: str, files: dict[str, bytes]) -> Path:
    """A local 'remote repo': ``prefix`` FULLY repacked (originals deleted)."""
    root = tmp_path / "remote"
    src = tmp_path / "raw_src"
    for rel, data in files.items():
        p = src / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
    packing.pack_tree_v2(src, root / prefix / packing.PACKED_DIRNAME)
    return root


def _patch_remote(monkeypatch, root: Path) -> FakeHfApi:
    """Patch the Hub boundary (shim lazy imports) to serve ``root``."""
    api = FakeHfApi(root)
    monkeypatch.setattr("huggingface_hub.hf_hub_download", _fake_hf_hub_download(root))
    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: api)
    return api


@pytest.fixture(autouse=True)
def _fresh_shim_state(monkeypatch):
    """Module-level shim caches are shared state — clear around every test."""
    hub.clear_packed_caches()
    monkeypatch.delenv("EPM_HF_PACKED_FALLBACK", raising=False)
    yield
    hub.clear_packed_caches()


# ---------------------------------------------------------------------------
# Migrated consumer 1: issue1090_fu3_yield_replay.py (plan §3.8, MF5d)
# ---------------------------------------------------------------------------

FU3_FILES: dict[str, bytes] = {
    "cellA/datagen/raw_pos.jsonl": b'{"r": 1}\n',
    "cellA/datagen/judge_raw_pos.json": b'{"all_scores": {}}',
    "cellA/datagen/gen_manifest.json": (
        b'{"behavior": "b", "n_judge_draws": 1, "quota_floor": 0.5, "target_n": 2}'
    ),
    # cellB has a deliberately INCOMPLETE sidecar set (no raw_pos.jsonl).
    "cellB/datagen/judge_raw_pos.json": b'{"all_scores": {}}',
}


def test_issue1090_fu3_repacked_tree(tmp_path, monkeypatch):
    """A FULLY-repacked prefix still yields the original cells (resolve via
    the shim) and the per-cell sidecars stage byte-exactly — never the
    pre-migration silent ``cells == []`` -> exit 0 shape."""
    root = _packed_remote(tmp_path, fu3.PREFIX, FU3_FILES)
    api = _patch_remote(monkeypatch, root)
    assert fu3._derive_cells(api) == ["cellA", "cellB"]
    got = hub.stage_hub_file(
        fu3.DATA_REPO, f"{fu3.PREFIX}/cellA/datagen/gen_manifest.json", tmp_path / "m.json"
    )
    assert got.read_bytes() == FU3_FILES["cellA/datagen/gen_manifest.json"]


def test_issue1090_fu3_empty_tree_fails_loud(tmp_path, monkeypatch):
    """An EMPTY tree (originals gone, no pack) can never again exit 0: main()
    dies on the empty-cell-derivation assert before any per-cell work."""
    root = tmp_path / "remote_empty"
    root.mkdir()
    api = _patch_remote(monkeypatch, root)
    monkeypatch.setattr(fu3, "HfApi", lambda token=None: api)
    monkeypatch.setattr(sys, "argv", ["issue1090_fu3_yield_replay.py"])
    with pytest.raises(AssertionError, match="empty cell derivation"):
        fu3.main()


def test_issue1090_fu3_all_cells_skipped_fails_loud(tmp_path, monkeypatch):
    """rows == [] (every derived cell SKIPped on incomplete sidecars) FAILS
    LOUD instead of the pre-migration silent return 0 with no summary."""
    root = _packed_remote(tmp_path, fu3.PREFIX, FU3_FILES)
    api = _patch_remote(monkeypatch, root)
    monkeypatch.setattr(fu3, "HfApi", lambda token=None: api)
    monkeypatch.setattr(sys, "argv", ["issue1090_fu3_yield_replay.py", "--cells", "cellB"])
    with pytest.raises(AssertionError, match="0 replayable cells"):
        fu3.main()


# ---------------------------------------------------------------------------
# Migrated consumer 2: issue1481_cjk_audit.py (plan §3.8, MF5d)
# ---------------------------------------------------------------------------

CJK_POOL_REL = "raw_completions/panel/arm1/completions__trained__bare.json"
CJK_FILES: dict[str, bytes] = {
    CJK_POOL_REL: json.dumps({"completions": [["hello", CJK_CHARS]]}).encode(),
    "raw_completions/base_arms/b1/completions__base__bare.json": (
        json.dumps({"completions": [["plain"]]}).encode()
    ),
    "raw_completions/panel/arm1/notes.txt": b"not-a-pool",  # dropped by the .json keep
}


def test_issue1481_cjk_audit_repacked_tree(tmp_path, monkeypatch):
    """A FULLY-repacked prefix still yields every pool byte-exactly at the
    cache layout recount() globs over, and scan() counts real completions —
    never the pre-migration silent zero-pool scan."""
    root = _packed_remote(tmp_path, cjk.PREFIX, CJK_FILES)
    api = _patch_remote(monkeypatch, root)
    monkeypatch.setattr(cjk, "HfApi", lambda token=None: api)
    cache = tmp_path / "cache"
    pool_files = cjk._download_pools(cache)
    rels = sorted(str(p).split(f"{cjk.PREFIX}/")[-1] for p in pool_files)
    assert rels == [
        "raw_completions/base_arms/b1/completions__base__bare.json",
        CJK_POOL_REL,
    ]
    staged = cache / cjk.PREFIX / CJK_POOL_REL
    assert staged.read_bytes() == CJK_FILES[CJK_POOL_REL]
    scan_out = cjk.scan(pool_files, cache)
    pools = [v for v in scan_out.values() if "n" in v]
    assert sum(v["n"] for v in pools) == 3
    assert sum(v["intruded"] for v in pools) == 1


def test_issue1481_cjk_audit_empty_tree_fails_loud_before_write(tmp_path, monkeypatch):
    """An EMPTY tree dies on the empty-pool-listing assert BEFORE any output
    artifact is written — no persisted false zero-intrusion scan."""
    root = tmp_path / "remote_empty"
    root.mkdir()
    api = _patch_remote(monkeypatch, root)
    monkeypatch.setattr(cjk, "HfApi", lambda token=None: api)
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    with pytest.raises(AssertionError, match="empty pool listing"):
        cjk.main(["--analysis-dir", str(analysis), "--cache-dir", str(tmp_path / "cache")])
    assert not (analysis / "cjk_intrusion_scan.json").exists()


def test_issue1481_cjk_audit_zero_scannable_pools_fails_before_write(tmp_path, monkeypatch):
    """Files present but ZERO scannable pools (no 'completions' key anywhere)
    also fails loud before the scan artifact is written."""
    files = {"raw_completions/panel/arm1/metadata_only.json": b'{"meta": 1}'}
    root = _packed_remote(tmp_path, cjk.PREFIX, files)
    api = _patch_remote(monkeypatch, root)
    monkeypatch.setattr(cjk, "HfApi", lambda token=None: api)
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    with pytest.raises(AssertionError, match="0 scannable pools"):
        cjk.main(["--analysis-dir", str(analysis), "--cache-dir", str(tmp_path / "cache")])
    assert not (analysis / "cjk_intrusion_scan.json").exists()
