"""Tests for the #2332 packed-prefix accessor (plan v2 SS4.3 Option 2 / SS4.7(2) AC4).

Two tiers:

* OFFLINE unit tests (always run, no network): a tiny two-shard packed layout
  is built in ``tmp_path`` with the packer's exact conventions (PAX tar,
  ``TarInfo.offset_data`` offsets, per-member sha256), and the download seam
  ``packed_prefix._download`` is monkeypatched with a signature-conformant
  fake. ``read_packed``'s REAL body runs in every offline test — only the
  network boundary is faked (code-style rule: one production-body test per
  seam-stubbed function; the seam's own real body is exercised by the
  autospec delegation test below and by the live tier).

* LIVE resolution test (the plan's AC4 demonstration): for each of the 8
  target prefixes on the canonical data repo, if ``<prefix>/__packed__/
  index.json`` exists, resolve >=5 real member paths and byte-verify against
  the index sha256s. READS ONLY — creates no Hub state. Pre-repack (zero
  packed prefixes) it SKIPS with a loud reason; the P7 closeout invocation is
  the binding AC4 run. Transient transport errors are retried once; a
  persistent transport failure FAILS (never skips) once at least one packed
  prefix is confirmed to exist. Only when EVERY existence probe
  transport-fails (hub unreachable — nothing is KNOWN packed) does it skip.

The module under test is loaded by FILE PATH from the sibling ``src/`` tree
(same pattern as ``tests/test_issue2332_repack_gates.py``): the editable
install resolves ``explore_persona_space`` to the MAIN checkout's src, which
does not contain this worktree-new module until the branch merges.
"""

from __future__ import annotations

import hashlib
import importlib.util
import io
import json
import os
import sys
import tarfile
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = REPO_ROOT / "src" / "explore_persona_space" / "orchestrate" / "packed_prefix.py"

LIVE_REPO = "superkaiba1/explore-persona-space-data"
TARGET_PREFIXES = (
    "issue1481_conpos_grid",
    "issue1090_pvdatagen",
    "issue1586_methodgen",
    "issue667_alllayer",
    "issue1434_writingstyle",
    "issue1739_ctxmap",
    "issue2224_screening",
    "issue1489_ctx_aug",
)


@pytest.fixture(scope="module")
def pp():
    spec = importlib.util.spec_from_file_location("packed_prefix_under_test", MODULE_PATH)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    return mod


# ── offline fixture: two-shard packed layout, packer-exact conventions ───────

PREFIX = "testpfx"


def _pack_shard(shard_path: Path, members: dict[str, bytes], index: dict) -> None:
    """Pack ``members`` into one PAX tar exactly as the packer does (step_pack),
    including the read-side offset re-derivation (write-side ``addfile`` leaves
    ``TarInfo.offset_data`` at 0 on CPython 3.11 — the round-2 packer fix)."""
    with tarfile.open(shard_path, "w", format=tarfile.PAX_FORMAT) as tar:
        for name in sorted(members):
            payload = members[name]
            ti = tarfile.TarInfo(name=name)
            ti.size = len(payload)
            ti.mtime = 0
            ti.mode = 0o644
            tar.addfile(ti, io.BytesIO(payload))
            index[name] = {
                "shard": shard_path.name,
                "offset": -1,  # re-derived below, as in the packer
                "size": ti.size,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
    with tarfile.open(shard_path, "r:") as rt:
        for m in rt.getmembers():
            index[m.name]["offset"] = m.offset_data
    assert all(e["offset"] >= 0 for e in index.values())


@pytest.fixture()
def packed_layout(tmp_path, monkeypatch, pp):
    """Local packed layout (2 shards, nested + binary members) + fake download seam."""
    members_s0 = {
        f"{PREFIX}/a.json": json.dumps({"k": 1, "v": [1, 2, 3]}).encode(),
        f"{PREFIX}/sub/dir/b.bin": bytes(range(256)) * 3 + b"\x00\xff\x00",
    }
    members_s1 = {f"{PREFIX}/zz/c.txt": b"hello packed world\n"}
    packed_dir = tmp_path / PREFIX / "__packed__"
    packed_dir.mkdir(parents=True)
    index: dict[str, dict] = {}
    _pack_shard(packed_dir / "shard-00000.tar", members_s0, index)
    _pack_shard(packed_dir / "shard-00001.tar", members_s1, index)
    index_path = packed_dir / "index.json"
    index_path.write_text(json.dumps(index, sort_keys=True, separators=(",", ":")))

    def fake_download(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
        # Signature-conformant by construction: mirrors packed_prefix._download.
        local = tmp_path / filename
        if not local.is_file():
            raise FileNotFoundError(f"fake hub: {repo_id}:{filename} (repo_type={repo_type})")
        return str(local)

    monkeypatch.setattr(pp, "_download", fake_download)
    return {"root": tmp_path, "index_path": index_path, "members": {**members_s0, **members_s1}}


# ── offline: happy path ──────────────────────────────────────────────────────


def test_round_trip_all_members(pp, packed_layout):
    """Every member — nested path, binary blob, second shard — round-trips byte-exact."""
    for orig, payload in packed_layout["members"].items():
        assert pp.read_packed("any/repo", orig) == payload


def test_multi_shard_routing(pp, packed_layout):
    """The member in shard-00001 resolves (the ONE named shard is the one opened)."""
    idx = json.loads(packed_layout["index_path"].read_text())
    assert idx[f"{PREFIX}/zz/c.txt"]["shard"] == "shard-00001.tar"
    assert pp.read_packed("any/repo", f"{PREFIX}/zz/c.txt") == b"hello packed world\n"


# ── offline: error arms (fail-loud, never unverified bytes) ──────────────────


def test_missing_path_raises_keyerror_naming_index(pp, packed_layout):
    with pytest.raises(KeyError, match=rf"{PREFIX}/__packed__/index\.json"):
        pp.read_packed("any/repo", f"{PREFIX}/nope.json")


def test_no_prefix_segment_raises_valueerror(pp, packed_layout):
    with pytest.raises(ValueError, match="no prefix segment"):
        pp.read_packed("any/repo", "noslash")


def test_leading_slash_rejected(pp, packed_layout):
    """An absolute path would derive an EMPTY prefix ('//__packed__/...') —
    rejected explicitly rather than surfacing as a confusing 404."""
    with pytest.raises(ValueError, match="absolute"):
        pp.read_packed("any/repo", f"/{PREFIX}/a.json")


def test_sha256_mismatch_raises(pp, packed_layout):
    idx_path = packed_layout["index_path"]
    idx = json.loads(idx_path.read_text())
    idx[f"{PREFIX}/a.json"]["sha256"] = "0" * 64
    idx_path.write_text(json.dumps(idx))
    with pytest.raises(pp.PackedPrefixError, match="sha256 mismatch"):
        pp.read_packed("any/repo", f"{PREFIX}/a.json")


def test_tampered_shard_bytes_raise(pp, packed_layout):
    """Flipping payload bytes inside the shard (size preserved) fails the sha check."""
    orig = f"{PREFIX}/sub/dir/b.bin"
    idx = json.loads(packed_layout["index_path"].read_text())
    ent = idx[orig]
    shard = packed_layout["root"] / PREFIX / "__packed__" / ent["shard"]
    raw = bytearray(shard.read_bytes())
    raw[ent["offset"]] ^= 0xFF  # corrupt the first data byte in place
    shard.write_bytes(bytes(raw))
    with pytest.raises(pp.PackedPrefixError, match="sha256 mismatch"):
        pp.read_packed("any/repo", orig)


def test_offset_disagreement_raises(pp, packed_layout):
    idx_path = packed_layout["index_path"]
    idx = json.loads(idx_path.read_text())
    idx[f"{PREFIX}/a.json"]["offset"] += 512
    idx_path.write_text(json.dumps(idx))
    with pytest.raises(pp.PackedPrefixError, match="!= index entry"):
        pp.read_packed("any/repo", f"{PREFIX}/a.json")


def test_member_absent_from_shard_raises(pp, packed_layout):
    """An index entry whose member is missing from the tar is an integrity error."""
    idx_path = packed_layout["index_path"]
    idx = json.loads(idx_path.read_text())
    idx[f"{PREFIX}/ghost.json"] = {
        "shard": "shard-00000.tar",
        "offset": 512,
        "size": 4,
        "sha256": "0" * 64,
    }
    idx_path.write_text(json.dumps(idx))
    with pytest.raises(pp.PackedPrefixError, match="no such member"):
        pp.read_packed("any/repo", f"{PREFIX}/ghost.json")


# ── offline: REAL packer -> accessor integration (pins the offset bug fix) ───


def test_real_packer_offsets_roundtrip_through_accessor(pp, tmp_path, monkeypatch):
    """Run the REAL ``step_pack`` from scripts/issue2332_repack_prefixes.py on
    staged files, then read every member back through the accessor.

    Regression pin for the round-2 packer fix: write-side ``addfile`` leaves
    ``TarInfo.offset_data`` at 0, so the pre-fix packer recorded ``offset: 0``
    for every member — this test FAILS pre-fix (the accessor's offset
    cross-check refuses every member past the first) and PASSES post-fix
    (offsets re-derived from the read path)."""
    spec = importlib.util.spec_from_file_location(
        "issue2332_repack_prefixes_for_accessor_test",
        REPO_ROOT / "scripts" / "issue2332_repack_prefixes.py",
    )
    rp = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = rp
    spec.loader.exec_module(rp)
    monkeypatch.setattr(rp, "STAGE", tmp_path)
    monkeypatch.setattr(rp, "_save_state", lambda *_a, **_k: None)
    prefix = "pktest"
    files = {
        f"{prefix}/one.json": b'{"x": 1}',
        f"{prefix}/deep/two.bin": bytes(range(200)) * 4,
        f"{prefix}/three.txt": b"third member\n",
    }
    src_root = tmp_path / f"src_{prefix}"
    staged_hashes = {}
    for rel, payload in files.items():
        p = src_root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(payload)
        staged_hashes[rel] = hashlib.sha256(payload).hexdigest()
    st = {"prefixes": {prefix: {}}}
    out = rp.step_pack(
        prefix, st, {}, sorted(files), staged_hashes, chunk_idx=None, write_manifest=False
    )
    index = out["index"]
    assert any(e["offset"] > 0 for e in index.values()), (
        "packer recorded no positive data offsets — the write-side offset_data bug"
    )
    packed_root = tmp_path / f"up_{prefix}"

    def fake_download(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
        local = packed_root / filename
        if not local.is_file():
            raise FileNotFoundError(f"fake hub: {repo_id}:{filename} (repo_type={repo_type})")
        return str(local)

    monkeypatch.setattr(pp, "_download", fake_download)
    for rel, payload in files.items():
        assert pp.read_packed("any/repo", rel) == payload


# ── offline: the download seam's REAL body (autospec'd HF boundary) ──────────


def test_download_real_body_delegates_to_hf_hub_download(pp, monkeypatch, tmp_path):
    """Executes ``_download``'s real body; only ``hf_hub_download`` is faked
    (autospec — signature-conformant by construction)."""
    from unittest.mock import create_autospec

    import huggingface_hub

    sentinel = tmp_path / "index.json"
    sentinel.write_text("{}")
    spec = create_autospec(huggingface_hub.hf_hub_download, return_value=str(sentinel))
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", spec)
    out = pp._download("some/repo", "p/__packed__/index.json", "dataset")
    assert out == str(sentinel)
    spec.assert_called_once_with(
        repo_id="some/repo", filename="p/__packed__/index.json", repo_type="dataset"
    )


# ── LIVE tier: AC4 resolution against the canonical data repo (reads only) ───


def _transport_excs() -> tuple[type[BaseException], ...]:
    import requests

    return (OSError, requests.exceptions.RequestException)


def _retry_once(fn, *args, **kwargs):
    """One retry on a transient transport error, then let the failure propagate."""
    try:
        return fn(*args, **kwargs)
    except _transport_excs():
        time.sleep(5)
        return fn(*args, **kwargs)


def _live_ac4(pp, expect_packed: tuple[str, ...] = ()) -> None:
    """The live AC4 body. Default mode: packed prefixes are DISCOVERED by
    probe, and zero-packed / hub-unreachable states skip loudly. CLOSEOUT mode
    (``expect_packed`` non-empty — the P7 binding invocation, via
    ``EPM_I2332_EXPECT_PACKED=all`` or a comma list): every expected prefix
    MUST be packed and resolvable — the skip branches are FORBIDDEN, and any
    probe failure is a FAILURE (Codex r1: a subset-green AC4 must not read as
    complete)."""
    from huggingface_hub import HfApi

    api = HfApi()
    packed: list[str] = []
    probe_failed: dict[str, str] = {}
    for prefix in TARGET_PREFIXES:
        try:
            if _retry_once(
                api.file_exists,
                repo_id=LIVE_REPO,
                filename=f"{prefix}/__packed__/index.json",
                repo_type="dataset",
            ):
                packed.append(prefix)
        except _transport_excs() as e:  # persistent (post-retry) transport failure
            probe_failed[prefix] = repr(e)
    if expect_packed:
        missing = sorted(set(expect_packed) - set(packed))
        assert not missing and not probe_failed, (
            f"CLOSEOUT MODE: expected packed prefixes not resolvable — missing index.json: "
            f"{missing}; probe transport failures: {probe_failed} (skip is forbidden here)"
        )
        packed = [p for p in packed if p in set(expect_packed)]
    if not packed:
        if len(probe_failed) == len(TARGET_PREFIXES):
            pytest.skip(
                f"LIVE HUB UNREACHABLE: all {len(TARGET_PREFIXES)} existence probes "
                f"transport-failed after one retry each; no prefix is KNOWN packed. "
                f"Errors: {sorted(probe_failed.items())}"
            )
        pytest.skip(
            "PRE-REPACK STATE: none of the 8 #2332 target prefixes has "
            "<prefix>/__packed__/index.json on the Hub yet. The P7 closeout invocation "
            "is the binding AC4 run (EPM_I2332_EXPECT_PACKED=all forbids this skip)."
        )
    # >=1 packed prefix confirmed ⇒ transport failures are FAILURES from here on
    # (retry-once already applied inside _retry_once).
    assert not probe_failed, (
        f"hub reachable ({len(packed)} packed prefix(es) confirmed) but existence probes "
        f"transport-failed persistently for: {probe_failed}"
    )
    for prefix in packed:
        idx_local = _retry_once(
            pp._download, LIVE_REPO, f"{prefix}/__packed__/index.json", "dataset"
        )
        index = json.loads(Path(idx_local).read_text())
        assert len(index) >= 5, f"{prefix}: packed index has only {len(index)} members"
        for orig in sorted(index)[:5]:  # deterministic sample of >=5 real paths
            data = _retry_once(pp.read_packed, LIVE_REPO, orig)
            assert len(data) == index[orig]["size"], f"{prefix}: size mismatch for {orig}"
            assert hashlib.sha256(data).hexdigest() == index[orig]["sha256"], (
                f"{prefix}: sha256 mismatch for {orig}"
            )


def _parse_expect_packed(raw: str) -> tuple[str, ...]:
    """Parse ``EPM_I2332_EXPECT_PACKED``. Empty/unset -> () (default discover
    mode). ``all`` -> every target prefix. Otherwise a comma list whose EVERY
    entry must be a known target prefix — a nonempty-but-malformed value (bare
    commas, an unknown prefix) FAILS the test instead of silently normalizing
    to the skip-permitting default (Codex r2 ac4-closeout-env-fail-open)."""
    raw = raw.strip()
    if not raw:
        return ()
    if raw.lower() == "all":
        return TARGET_PREFIXES
    entries = tuple(x for x in (s.strip() for s in raw.split(",")) if x)
    unknown = sorted(set(entries) - set(TARGET_PREFIXES))
    if not entries or unknown:
        pytest.fail(
            f"EPM_I2332_EXPECT_PACKED is set but malformed: {raw!r} (parsed={entries}, "
            f"unknown={unknown}) — a nonempty value must be 'all' or a comma list of "
            "known target prefixes; refusing to fall back to the skippable default"
        )
    return entries


def test_live_packed_prefix_resolution_ac4(pp):
    """AC4 demonstration (plan SS4.7(2)): >=5 real member paths per PACKED prefix
    resolve through ``read_packed`` and byte-verify. Reads only — no Hub writes.
    P7 closeout invocation: ``EPM_I2332_EXPECT_PACKED=all`` (or a comma list)."""
    expect = _parse_expect_packed(os.environ.get("EPM_I2332_EXPECT_PACKED", ""))
    _live_ac4(pp, expect_packed=expect)


def test_parse_expect_packed_malformed_values_fail_not_skip():
    """OFFLINE pin (Codex r2 ac4-closeout-env-fail-open): a nonempty malformed
    EXPECT_PACKED value must FAIL loud — never normalize to the unset default
    that permits the pre-repack skip."""
    assert _parse_expect_packed("") == ()
    assert _parse_expect_packed("  ") == ()
    assert _parse_expect_packed("all") == TARGET_PREFIXES
    assert _parse_expect_packed("ALL") == TARGET_PREFIXES
    two = f"{TARGET_PREFIXES[0]} , {TARGET_PREFIXES[1]}"
    assert _parse_expect_packed(two) == (TARGET_PREFIXES[0], TARGET_PREFIXES[1])
    with pytest.raises(pytest.fail.Exception, match="malformed"):
        _parse_expect_packed(",,")
    with pytest.raises(pytest.fail.Exception, match="unknown"):
        _parse_expect_packed("not_a_known_prefix")
    with pytest.raises(pytest.fail.Exception, match="unknown"):
        _parse_expect_packed(f"{TARGET_PREFIXES[0]},bogus_prefix")


def test_closeout_mode_fails_instead_of_skipping(pp, monkeypatch):
    """OFFLINE pin of the closeout-mode contract: with every existence probe
    returning False, closeout mode FAILS (never skips) naming the missing
    prefixes. The HF boundary fake is signature-conformant by construction."""
    import huggingface_hub

    class _FakeApi:
        def file_exists(
            self, repo_id: str, filename: str, *, repo_type=None, revision=None, token=None
        ) -> bool:
            return False

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    with pytest.raises(AssertionError, match="CLOSEOUT MODE"):
        _live_ac4(pp, expect_packed=("issue1489_ctx_aug",))


# ── r4: packed_fallback semantics (the central-seam probe) ────────────────────


def test_packed_fallback_semantics(pp, tmp_path, monkeypatch):
    """``packed_fallback`` (review r2 item 5): serves bytes for a repacked
    target-prefix member; returns None (the caller re-raises its ORIGINAL
    not-found unchanged) for non-target prefixes, prefix-less paths,
    un-repacked target prefixes, and non-member paths; and PROPAGATES
    ``PackedPrefixError`` — integrity failures are never swallowed to None."""
    from huggingface_hub.errors import EntryNotFoundError

    target_prefix = "issue1489_ctx_aug"
    assert target_prefix in pp.REPACKED_PREFIXES
    member = f"{target_prefix}/raw/x.json"
    payload = b'{"j": 42}'
    packed_dir = tmp_path / target_prefix / "__packed__"
    packed_dir.mkdir(parents=True)
    index: dict[str, dict] = {}
    _pack_shard(packed_dir / "shard-00000.tar", {member: payload}, index)
    (packed_dir / "index.json").write_text(json.dumps(index))

    def fake_download(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
        local = tmp_path / filename
        if not local.is_file():
            raise EntryNotFoundError(f"fake hub 404: {filename}")
        return str(local)

    monkeypatch.setattr(pp, "_download", fake_download)
    assert pp.packed_fallback(LIVE_REPO, member) == payload
    # A leading slash is stripped (stage_hub_file passes path_in_repo raw).
    assert pp.packed_fallback(LIVE_REPO, "/" + member) == payload
    assert pp.packed_fallback(LIVE_REPO, "not_a_target/x.json") is None
    assert pp.packed_fallback(LIVE_REPO, "orphan.json") is None
    # r3 review minor: degenerate trailing-slash paths under a target prefix
    # are None (the caller re-raises its original 404) — never a ValueError
    # type-change out of read_packed's shape guards.
    assert pp.packed_fallback(LIVE_REPO, f"{target_prefix}/") is None
    assert pp.packed_fallback(LIVE_REPO, f"{target_prefix}//") is None
    # Target prefix but not a member of the packed index: None (KeyError arm).
    assert pp.packed_fallback(LIVE_REPO, f"{target_prefix}/raw/other.json") is None
    # Target prefix whose index.json does not exist (not repacked yet): None.
    assert pp.packed_fallback(LIVE_REPO, "issue667_alllayer/x.json") is None
    # Integrity failure PROPAGATES loudly (never swallowed into None).
    tampered = json.loads((packed_dir / "index.json").read_text())
    tampered[member]["sha256"] = "0" * 64
    (packed_dir / "index.json").write_text(json.dumps(tampered))
    with pytest.raises(pp.PackedPrefixError, match="sha256 mismatch"):
        pp.packed_fallback(LIVE_REPO, member)


def test_packed_index_integrity_failures_fail_loud(pp, tmp_path, monkeypatch):
    """r3 binding CONCERN (packed-fallback-integrity-masking), both arms: a
    MALFORMED index entry (missing required key) and a shard the index names
    but the repo LACKS raise ``PackedPrefixError`` through BOTH ``read_packed``
    and ``packed_fallback`` — an index that references a missing shard is
    corruption, never 'file not found', so neither arm may be swallowed into
    the caller's original loose-file 404."""
    from huggingface_hub.errors import EntryNotFoundError

    target_prefix = "issue1489_ctx_aug"
    member = f"{target_prefix}/raw/x.json"
    payload = b'{"j": 43}'
    packed_dir = tmp_path / target_prefix / "__packed__"
    packed_dir.mkdir(parents=True)
    index: dict[str, dict] = {}
    _pack_shard(packed_dir / "shard-00000.tar", {member: payload}, index)
    (packed_dir / "index.json").write_text(json.dumps(index))

    def fake_download(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
        local = tmp_path / filename
        if not local.is_file():
            raise EntryNotFoundError(f"fake hub 404: {filename}")
        return str(local)

    monkeypatch.setattr(pp, "_download", fake_download)
    assert pp.packed_fallback(LIVE_REPO, member) == payload  # layout sane pre-corruption
    # Arm 1: malformed index entry (required key missing) -> loud, both routes.
    good = json.loads((packed_dir / "index.json").read_text())
    bad = {member: {k: v for k, v in good[member].items() if k != "shard"}}
    (packed_dir / "index.json").write_text(json.dumps(bad))
    with pytest.raises(pp.PackedPrefixError, match="malformed"):
        pp.read_packed(LIVE_REPO, member)
    with pytest.raises(pp.PackedPrefixError, match="malformed"):
        pp.packed_fallback(LIVE_REPO, member)
    (packed_dir / "index.json").write_text(json.dumps(good))
    # Arm 2: the index names a shard the repo lacks -> loud, both routes
    # (pre-fix: the fake's EntryNotFoundError was swallowed into None by
    # packed_fallback's not-found catch).
    (packed_dir / "shard-00000.tar").unlink()
    with pytest.raises(pp.PackedPrefixError, match="missing from"):
        pp.read_packed(LIVE_REPO, member)
    with pytest.raises(pp.PackedPrefixError, match="missing from"):
        pp.packed_fallback(LIVE_REPO, member)


# ── r4: stage_hub_file central-seam fallback (subprocess, worktree src) ───────

_STAGE_FALLBACK_PROBE = '''
"""Subprocess probe: hub.stage_hub_file serves a repacked member through the
packed fallback on EntryNotFoundError (positive arm) and re-raises the
ORIGINAL not-found unchanged for a non-target path (negative arm). Only the
huggingface_hub download boundary is faked (signature-conformant def)."""

import hashlib
import io
import json
import sys
import tarfile
from pathlib import Path

work = Path(sys.argv[1])
src_dir = Path(sys.argv[2])
prefix = "issue1489_ctx_aug"
member = f"{prefix}/raw/x.json"
payload = b'{"k": 1}'
packed_dir = work / prefix / "__packed__"
packed_dir.mkdir(parents=True, exist_ok=True)
shard = packed_dir / "shard-00000.tar"
with tarfile.open(shard, "w", format=tarfile.PAX_FORMAT) as tar:
    ti = tarfile.TarInfo(name=member)
    ti.size = len(payload)
    ti.mtime = 0
    ti.mode = 0o644
    tar.addfile(ti, io.BytesIO(payload))
with tarfile.open(shard, "r:") as rt:
    off = rt.getmember(member).offset_data
index = {
    member: {
        "shard": "shard-00000.tar",
        "offset": off,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
}
(packed_dir / "index.json").write_text(json.dumps(index))

import huggingface_hub
from huggingface_hub.errors import EntryNotFoundError


def fake_hf_hub_download(
    *, repo_id=None, filename=None, repo_type="dataset", revision=None,
    local_dir=None, token=None, etag_timeout=None, **kw
):
    if "/__packed__/" in (filename or ""):
        return str(work / filename)
    raise EntryNotFoundError(f"loose copy absent (repacked): {filename}")


huggingface_hub.hf_hub_download = fake_hf_hub_download

from explore_persona_space.orchestrate import hub as H
from explore_persona_space.orchestrate import packed_prefix as PP

assert Path(PP.__file__).resolve().is_relative_to(src_dir.resolve()), (
    f"worktree src did not win module resolution: {PP.__file__} not under {src_dir}"
)
target = work / "staged" / "x.json"
out = H.stage_hub_file("superkaiba1/explore-persona-space-data", member, target)
assert Path(out) == target and target.read_bytes() == payload, "fallback did not serve bytes"
try:
    H.stage_hub_file(
        "superkaiba1/explore-persona-space-data", "not_a_target/x.json", work / "n.json"
    )
except EntryNotFoundError:
    pass
else:
    raise AssertionError("non-target 404 must re-raise the ORIGINAL EntryNotFoundError")
assert not (work / "n.json").exists(), "negative arm must stage nothing"

# Arm 3 (r3 review minor): a write failure inside the fallback arm propagates
# the ORIGINAL error and leaves NO orphan .hfstage-packed-* tmp residue.
real_ntf = H.tempfile.NamedTemporaryFile


class _FailingTF:
    def __init__(self, *a, **k):
        self._real = real_ntf(*a, **k)
        self.name = self._real.name

    def __enter__(self):
        self._real.__enter__()
        return self

    def __exit__(self, *exc):
        return self._real.__exit__(*exc)

    def write(self, data):
        raise OSError(28, "No space left on device (injected)")


H.tempfile.NamedTemporaryFile = _FailingTF
target3 = work / "staged3" / "x.json"
try:
    H.stage_hub_file("superkaiba1/explore-persona-space-data", member, target3)
except OSError as e:
    assert e.errno == 28, f"expected the injected ENOSPC to propagate, got {e!r}"
else:
    raise AssertionError("injected write failure must propagate")
finally:
    H.tempfile.NamedTemporaryFile = real_ntf
assert not target3.exists(), "failed write must not publish a target"
residue = sorted((work / "staged3").glob(".hfstage-packed-*"))
assert not residue, f"tmp residue leaked: {residue}"
print("HUB-FALLBACK-OK")
'''


def test_stage_hub_file_packed_fallback_subprocess(tmp_path):
    """The central-seam fallback (review r2 item 5) exercised through the REAL
    ``hub.stage_hub_file`` body: loose-404 under a target prefix serves the
    packed member atomically; any other path re-raises the original not-found.
    Runs in a subprocess with the WORKTREE src forced onto PYTHONPATH — the
    editable install otherwise resolves ``explore_persona_space`` to the MAIN
    checkout's src, where this seam does not exist until the branch merges."""
    import subprocess

    src_dir = REPO_ROOT / "src"
    script = tmp_path / "probe_stage_hub_file.py"
    script.write_text(_STAGE_FALLBACK_PROBE)
    env = {
        **os.environ,
        "PYTHONPATH": str(src_dir) + os.pathsep + os.environ.get("PYTHONPATH", ""),
    }
    cp = subprocess.run(
        [sys.executable, str(script), str(tmp_path / "scratch"), str(src_dir)],
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    assert cp.returncode == 0, f"stdout={cp.stdout}\nstderr={cp.stderr}"
    assert "HUB-FALLBACK-OK" in cp.stdout
