"""Tests for the #2321 consumer-gate deliverables (plan §3.8 / §12, I17/MF5).

Covers the two migrated silent-empty consumers on REPACKED fixture trees
(`scripts/issue1090_fu3_yield_replay.py`, `scripts/issue1481_cjk_audit.py`)
plus the consumer-inventory scanner/gate tooling
(`scripts/issue2321_consumer_gate.py` + the committed inventory JSON) and the
live shim-check driver (`scripts/issue2321_verify_shim.py`, fixture-backed).

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


# ---------------------------------------------------------------------------
# Inventory scanner + gate tooling (scripts/issue2321_consumer_gate.py, §3.8)
# ---------------------------------------------------------------------------

gate_mod = _load_script("issue2321_consumer_gate.py", "issue2321_consumer_gate_mod")
TARGETS = frozenset({"issue1090_partial", "issue1481_conpos_grid"})


def test_conftest_pins_offline_env():
    """The conftest module registration (I18 defense-in-depth) must hold."""
    import os

    assert os.environ.get("HF_HUB_OFFLINE") == "1"


def _fixture_scan_tree(tmp_path: Path) -> Path:
    root = tmp_path / "fixrepo"
    sd = root / "scripts"
    sd.mkdir(parents=True)
    (sd / "fake_consumer.py").write_text(
        'PREFIX = "issue1090_partial"\n'
        "def go(api, hub):\n"
        '    files = api.list_repo_tree("repo", path_in_repo=PREFIX, repo_type="dataset")\n'
        '    root = f"{PREFIX}/att-1"\n'
        '    more = hub.list_hf_files_under_path(api, "repo", root)\n'
        "    return files, more\n"
    )
    (sd / "other_prefix.py").write_text(
        'def go(api):\n    return api.list_repo_tree("repo", path_in_repo="issue9999_x")\n'
    )
    # The shim family is excluded even when it spells a target prefix.
    (sd / "issue2321_repack.py").write_text(
        'def go(api):\n    return api.list_repo_tree("repo", path_in_repo="issue1090_partial")\n'
    )
    return root


def test_scanner_finds_listing_hits_including_local_assign_shape(tmp_path):
    """Module-constant AND local-f-string-assign indirection both resolve; a
    non-target prefix and the excluded shim family produce no hits."""
    hits = gate_mod.scan_tree(_fixture_scan_tree(tmp_path), TARGETS)
    got = {(h["script"], h["call"], h["prefix"]) for h in hits}
    assert got == {
        ("scripts/fake_consumer.py", "list_repo_tree", "issue1090_partial"),
        ("scripts/fake_consumer.py", "list_hf_files_under_path", "issue1090_partial"),
    }


def test_check_flags_uncovered_then_passes_when_covered(tmp_path):
    """--check semantics: an uncovered hit errors; a covering row clears it."""
    driver = gate_mod._load_driver()
    hits = gate_mod.scan_tree(_fixture_scan_tree(tmp_path), TARGETS)
    empty = {"version": 1, "consumers": []}
    errors = gate_mod.check_inventory(hits, empty, driver._consumer_scoped)
    assert len(errors) == 2 and all("UNCOVERED" in e for e in errors)
    covered = {
        "version": 1,
        "consumers": [
            {
                "script": "scripts/fake_consumer.py",
                "prefixes": ["issue1090_partial"],
                "silent_empty": False,
                "migrated": False,
            }
        ],
    }
    assert gate_mod.check_inventory(hits, covered, driver._consumer_scoped) == []


def test_gate_cli_blocks_rc22_then_passes_when_migrated(tmp_path):
    """§12: the gate blocks a prefix with an unmigrated silent-empty consumer
    (rc=22) and passes once the inventory marks it migrated — evaluated
    through the DRIVER's own consumer_gate (single-source semantics)."""
    driver = gate_mod._load_driver()
    inv = tmp_path / "inv.json"
    row = {
        "script": "scripts/fake_consumer.py",
        "prefixes": ["issue1090_partial"],
        "silent_empty": True,
        "migrated": False,
    }
    tp = sorted(driver.PREFIX_ORDER)  # r2 M1: load-time schema requires it
    inv.write_text(json.dumps({"version": 1, "target_prefixes": tp, "consumers": [row]}))
    rc = gate_mod.main(["--gate", "--prefix", "issue1090_partial", "--inventory", str(inv)])
    assert rc == 22
    row["migrated"] = True
    inv.write_text(json.dumps({"version": 1, "target_prefixes": tp, "consumers": [row]}))
    rc = gate_mod.main(["--gate", "--prefix", "issue1090_partial", "--inventory", str(inv)])
    assert rc == 0


def test_gate_cli_string_boolean_migrated_blocks_rc22(tmp_path):
    """r2 g5-M1 / Codex-M1: a hand-authored '"migrated": "false"' STRING is a
    schema error that fails CLOSED (rc=22) — pre-fix, truthiness read it as
    migrated and the gate PASSED, silently authorizing deletion."""
    driver = gate_mod._load_driver()
    inv = tmp_path / "inv.json"
    row = {
        "script": "scripts/fake_consumer.py",
        "prefixes": ["issue1090_partial"],
        "silent_empty": True,
        "migrated": "false",  # truthy STRING — the exact hazard shape
    }
    inv.write_text(
        json.dumps(
            {"version": 1, "target_prefixes": sorted(driver.PREFIX_ORDER), "consumers": [row]}
        )
    )
    rc = gate_mod.main(["--gate", "--prefix", "issue1090_partial", "--inventory", str(inv)])
    assert rc == 22


def test_gate_cli_missing_inventory_fails_closed(tmp_path):
    """I17 fail-closed: a missing inventory file blocks (rc=22), never passes."""
    rc = gate_mod.main(
        ["--gate", "--prefix", "issue1090_partial", "--inventory", str(tmp_path / "absent.json")]
    )
    assert rc == 22


def test_committed_inventory_covers_live_scan():
    """The committed inventory covers EVERY live-tree scan hit (the --check
    contract), and the two plan-§3.8 migrated consumers are recorded
    silent_empty + migrated with a real migrating SHA."""
    driver = gate_mod._load_driver()
    hits = gate_mod.scan_tree(REPO_ROOT, frozenset(driver.PREFIX_ORDER))
    inventory = json.loads(
        (REPO_ROOT / "scripts" / "issue2321_consumer_inventory.json").read_text()
    )
    errors = gate_mod.check_inventory(hits, inventory, driver._consumer_scoped)
    assert errors == [], errors
    rows = {r["script"]: r for r in inventory["consumers"]}
    for script in ("scripts/issue1090_fu3_yield_replay.py", "scripts/issue1481_cjk_audit.py"):
        row = rows[script]
        assert row["silent_empty"] is True and row["migrated"] is True
        sha = row["migrated_sha"]
        assert len(sha) == 40 and all(c in "0123456789abcdef" for c in sha)


def test_committed_inventory_gates_all_ten_prefixes_clean():
    """With the two migrations recorded, NO target prefix is blocked — the
    driver's consumer-gate phase passes on every prefix in the walk order."""
    driver = gate_mod._load_driver()
    inventory = driver.load_consumer_inventory(
        REPO_ROOT / "scripts" / "issue2321_consumer_inventory.json"
    )
    for prefix in driver.PREFIX_ORDER:
        verdict = driver.consumer_gate(inventory, prefix)
        assert verdict["blockers"] == 0, (prefix, verdict)


# ---------------------------------------------------------------------------
# Live shim-check driver (scripts/issue2321_verify_shim.py), fixture-backed
# ---------------------------------------------------------------------------

vshim = _load_script("issue2321_verify_shim.py", "issue2321_verify_shim_mod")


def test_verify_shim_samples_resolve_byte_exact(tmp_path, monkeypatch):
    """Every sampled member of a fully-repacked prefix stages through the
    production stage_hub_file fallback and sha256-matches its record."""
    root = _packed_remote(tmp_path, "issue1090_partial", FU3_FILES)
    api = _patch_remote(monkeypatch, root)
    ok, n, problems = vshim.verify_prefix_samples(
        api,
        repo_id="fake-org/fake-data-repo",
        prefix="issue1090_partial",
        n_samples=3,
        stage_root=tmp_path / "stage",
    )
    assert problems == []
    assert ok == n == 3


def test_verify_shim_flags_digest_mismatch(tmp_path, monkeypatch):
    """A tampered recorded digest is flagged, never silently passed."""
    import dataclasses

    root = _packed_remote(tmp_path, "issue1090_partial", FU3_FILES)
    api = _patch_remote(monkeypatch, root)
    real = hub.packed_members_under_path

    def tampered(*a, **k):
        ms = list(real(*a, **k))
        return [dataclasses.replace(ms[0], sha256="0" * 64), *ms[1:]]

    monkeypatch.setattr(hub, "packed_members_under_path", tampered)
    ok, n, problems = vshim.verify_prefix_samples(
        api,
        repo_id="fake-org/fake-data-repo",
        prefix="issue1090_partial",
        n_samples=len(FU3_FILES),
        stage_root=tmp_path / "stage",
    )
    assert any("sha256 mismatch" in p for p in problems)
    assert ok == n - 1


def test_verify_shim_main_pass_and_no_pack_arms(tmp_path, monkeypatch):
    """main(): rc=0 on a resolvable repacked prefix; a pack-less prefix FAILS
    (rc=1) by default, and 0/0 under --allow-unpacked is NEVER a PASS."""
    root = _packed_remote(tmp_path, "issue1090_partial", FU3_FILES)
    _patch_remote(monkeypatch, root)
    rc = vshim.main(["--prefixes", "issue1090_partial", "--samples", "2"])
    assert rc == 0
    empty = tmp_path / "remote_empty"
    empty.mkdir()
    _patch_remote(monkeypatch, empty)
    hub.clear_packed_caches()
    assert vshim.main(["--prefixes", "issue1090_partial", "--samples", "2"]) == 1
    assert (
        vshim.main(["--prefixes", "issue1090_partial", "--samples", "2", "--allow-unpacked"]) == 1
    )
