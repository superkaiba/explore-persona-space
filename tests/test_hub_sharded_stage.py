"""Tests for #2119 — manifest-first name resolution + staging for sharded text artifacts.

Pins ``resolve_sharded_text_paths`` / ``stage_sharded_text`` /
``_parse_shard_manifest`` (``src/explore_persona_space/orchestrate/hub.py``):
manifest-first resolve, per-shard sha256 verification, in-order concat, the
pre-shard compat fallback, and — the #2054 incident-2 hard constraint — that a
missing PART under an existing manifest raises and NEVER falls back to the
unsharded hub name (stale prior-round residue).

Monkeypatch strategy mirrors tests/test_hub_staging_retry.py: ``stage_hub_file``
is a hub-module global -> patch at the hub site with a fake that copies fixture
bytes (signature mirrors the real helper); ``HfApi`` is a function-body lazy
import in ``stage_sharded_text`` -> patch ``huggingface_hub.HfApi``; the
resolver takes a stub ``api`` directly.

Until this branch merges, run with ``PYTHONPATH=<worktree>/src`` so the
worktree's ``explore_persona_space`` (which carries the new helpers) shadows
the editable install pointing at main.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from pathlib import Path

import pytest

from explore_persona_space.orchestrate import hub

DRAW = "issueX/scaffolds/draw.jsonl"
MANIFEST = "issueX/scaffolds/draw.manifest.json"
SHARD0 = "issueX/scaffolds/draw.shard00.jsonl"
SHARD1 = "issueX/scaffolds/draw.shard01.jsonl"
PART_BYTES = {
    "draw.shard00.jsonl": b'{"a": 1}\n{"b": 2}\n',
    "draw.shard01.jsonl": b'{"c": 3}\n',
}


@pytest.fixture(autouse=True)
def fast_retries(monkeypatch):
    """No real sleeps + attempt-bound retry (budget 0 => 6 calls max, #735)."""
    monkeypatch.setattr(hub.time, "sleep", lambda s: None)
    monkeypatch.setenv("EPM_HF_RETRY_BUDGET_S", "0")


def _manifest_bytes(
    parts: list[str] | None = None,
    *,
    drop_sha: tuple[str, ...] = (),
    bad_sha: tuple[str, ...] = (),
) -> bytes:
    """Manifest in the #2054 producer schema (parts + sha256; extras ignored)."""
    names = list(PART_BYTES) if parts is None else parts
    sha = {}
    for n in names:
        if n in drop_sha:
            continue
        sha[n] = "0" * 64 if n in bad_sha else hashlib.sha256(PART_BYTES[n]).hexdigest()
    return json.dumps(
        {
            "source": "draw.jsonl",
            "parts": names,
            "line_counts": [PART_BYTES.get(n, b"").count(b"\n") for n in names],
            "sha256": sha,
        }
    ).encode("utf-8")


class _StubApi:
    """file_exists stub keyed on a repo-path -> bytes dict (probe boundary)."""

    def __init__(self, hub_files: dict[str, bytes], token=None):
        self.hub_files = hub_files
        self.probed: list[str] = []

    def file_exists(self, repo_id, path_in_repo, *, repo_type=None, revision=None):
        self.probed.append(path_in_repo)
        return path_in_repo in self.hub_files


def _install_fakes(monkeypatch, hub_files: dict[str, bytes]) -> list[str]:
    """Patch HfApi + hub.stage_hub_file against a fake hub; return the
    requested-download repo-path list (the never-asked-for-unsharded probe)."""
    requested: list[str] = []

    monkeypatch.setattr("huggingface_hub.HfApi", lambda token=None: _StubApi(hub_files, token))

    def fake_stage_hub_file(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
        size_bytes=None,
    ):
        requested.append(path_in_repo)
        if path_in_repo not in hub_files:
            raise RuntimeError(f"not on hub: {path_in_repo}")
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(hub_files[path_in_repo])
        return target

    monkeypatch.setattr(hub, "stage_hub_file", fake_stage_hub_file)
    return requested


# ---------------------------------------------------------------------------
# stage_sharded_text
# ---------------------------------------------------------------------------


def test_sharded_happy_path_in_order_concat(tmp_path, monkeypatch):
    """D3-1: manifest + 2 parts staged, output == in-order concat, returns
    target, no leftover tmp file."""
    hub_files = {
        MANIFEST: _manifest_bytes(),
        SHARD0: PART_BYTES["draw.shard00.jsonl"],
        SHARD1: PART_BYTES["draw.shard01.jsonl"],
    }
    _install_fakes(monkeypatch, hub_files)

    target = tmp_path / "staged" / "draw.jsonl"
    out = hub.stage_sharded_text("org/data", DRAW, target)
    assert out == target
    assert target.read_bytes() == (
        PART_BYTES["draw.shard00.jsonl"] + PART_BYTES["draw.shard01.jsonl"]
    )
    assert list(target.parent.glob("*.tmp")) == []


def test_unsharded_fallback_when_no_manifest(tmp_path, monkeypatch, caplog):
    """D3-2: no manifest on the Hub -> the unsharded name is staged and the
    log line says fallback."""
    hub_files = {DRAW: b'{"plain": true}\n'}
    requested = _install_fakes(monkeypatch, hub_files)

    target = tmp_path / "staged" / "draw.jsonl"
    with caplog.at_level(logging.INFO, logger="explore_persona_space.orchestrate.hub"):
        out = hub.stage_sharded_text("org/data", DRAW, target)
    assert out == target
    assert target.read_bytes() == b'{"plain": true}\n'
    assert requested == [DRAW]
    assert "fallback to the unsharded name" in caplog.text


def test_missing_part_under_manifest_raises_and_never_requests_unsharded(tmp_path, monkeypatch):
    """D3-3 (the fail-loud hard constraint): a missing PART under an existing
    manifest propagates RuntimeError AND the unsharded name was NEVER
    requested — falling back there is exactly the #2054 incident-2 defect
    (stale prior-round residue)."""
    hub_files = {
        MANIFEST: _manifest_bytes(),
        SHARD0: PART_BYTES["draw.shard00.jsonl"],
        # SHARD1 deliberately absent — and a stale unsharded blob present,
        # so a fallback WOULD succeed if the bug existed.
        DRAW: b'{"stale": "prior-round residue"}\n',
    }
    requested = _install_fakes(monkeypatch, hub_files)

    target = tmp_path / "staged" / "draw.jsonl"
    with pytest.raises(RuntimeError, match=re.escape(f"not on hub: {SHARD1}")):
        hub.stage_sharded_text("org/data", DRAW, target)
    assert DRAW not in requested
    assert not target.exists()


def test_sha_mismatch_raises(tmp_path, monkeypatch):
    """D3-4: a part whose sha256 disagrees with the manifest raises."""
    hub_files = {
        MANIFEST: _manifest_bytes(bad_sha=("draw.shard01.jsonl",)),
        SHARD0: PART_BYTES["draw.shard00.jsonl"],
        SHARD1: PART_BYTES["draw.shard01.jsonl"],
    }
    _install_fakes(monkeypatch, hub_files)

    with pytest.raises(RuntimeError, match="sha mismatch"):
        hub.stage_sharded_text("org/data", DRAW, tmp_path / "draw.jsonl")


def test_missing_sha_entry_refuses_unverified_shard(tmp_path, monkeypatch):
    """D3-5: a manifest listing a part with NO sha256 entry raises (the #2054
    writer always records per-shard shas — absence signals a foreign/malformed
    manifest)."""
    hub_files = {
        MANIFEST: _manifest_bytes(drop_sha=("draw.shard01.jsonl",)),
        SHARD0: PART_BYTES["draw.shard00.jsonl"],
        SHARD1: PART_BYTES["draw.shard01.jsonl"],
    }
    _install_fakes(monkeypatch, hub_files)

    with pytest.raises(RuntimeError, match="refusing unverified shard"):
        hub.stage_sharded_text("org/data", DRAW, tmp_path / "draw.jsonl")


def test_empty_parts_manifest_raises(tmp_path, monkeypatch):
    """D3-6: a manifest with an empty ``parts`` list raises."""
    hub_files = {MANIFEST: _manifest_bytes(parts=[])}
    _install_fakes(monkeypatch, hub_files)

    with pytest.raises(RuntimeError, match="lists no parts"):
        hub.stage_sharded_text("org/data", DRAW, tmp_path / "draw.jsonl")


# ---------------------------------------------------------------------------
# resolve_sharded_text_paths
# ---------------------------------------------------------------------------


def test_resolver_sharded_returns_manifest_plus_parts_in_order(tmp_path, monkeypatch):
    """D3-7: manifest present -> ("sharded", [manifest, p1, p2]) as repo
    paths, parts in manifest order (dir-joined beside the manifest)."""
    hub_files = {
        MANIFEST: _manifest_bytes(),
        SHARD0: PART_BYTES["draw.shard00.jsonl"],
        SHARD1: PART_BYTES["draw.shard01.jsonl"],
    }
    _install_fakes(monkeypatch, hub_files)
    api = _StubApi(hub_files)

    form, paths = hub.resolve_sharded_text_paths(api, "org/data", DRAW)
    assert form == "sharded"
    assert paths == [MANIFEST, SHARD0, SHARD1]


def test_resolver_unsharded_and_neither_form(tmp_path, monkeypatch):
    """D3-8: unsharded-only -> ("unsharded", [path]); NEITHER form on the Hub
    -> RuntimeError (fail-loud, never a silent default)."""
    _install_fakes(monkeypatch, {DRAW: b"x\n"})

    form, paths = hub.resolve_sharded_text_paths(_StubApi({DRAW: b"x\n"}), "org/data", DRAW)
    assert (form, paths) == ("unsharded", [DRAW])

    with pytest.raises(RuntimeError, match="nothing to resolve"):
        hub.resolve_sharded_text_paths(_StubApi({}), "org/data", DRAW)


def test_concat_tmp_is_per_invocation_not_deterministic(tmp_path, monkeypatch):
    """#2119 code-review Minor 1: the concat tmp must be a per-invocation
    ``mkstemp`` name inside the destination dir, NEVER the deterministic
    ``<target>.tmp``.

    A deterministic name is shared by two concurrent stagers writing the same
    ``target``: the second's ``"wb"`` truncation can interleave with the first's
    writes and ``os.replace`` then publishes unverified bytes (the #1315
    fan-out-shared-staging class). Pinning the name shape is what keeps the
    fix from silently regressing — the race itself is not deterministically
    reproducible in-process.
    """
    hub_files = {
        MANIFEST: _manifest_bytes(),
        SHARD0: PART_BYTES["draw.shard00.jsonl"],
        SHARD1: PART_BYTES["draw.shard01.jsonl"],
    }
    _install_fakes(monkeypatch, hub_files)

    seen: list[dict] = []
    real_mkstemp = hub.tempfile.mkstemp

    def _spy_mkstemp(*args, **kwargs):
        seen.append(kwargs)
        return real_mkstemp(*args, **kwargs)

    monkeypatch.setattr(hub.tempfile, "mkstemp", _spy_mkstemp)

    target = tmp_path / "staged" / "draw.jsonl"
    out = hub.stage_sharded_text("org/data", DRAW, target)

    assert out.read_bytes() == (PART_BYTES["draw.shard00.jsonl"] + PART_BYTES["draw.shard01.jsonl"])
    # mkstemp was used for the concat, rooted in the destination dir so the
    # publish rename stays same-filesystem (the #1335 EXDEV gotcha).
    assert seen, "concat did not go through tempfile.mkstemp"
    assert seen[-1].get("dir") == target.parent
    # ...and NOT the deterministic sibling name.
    assert not (target.parent / (target.name + ".tmp")).exists()
    assert list(target.parent.glob("*.tmp")) == []
