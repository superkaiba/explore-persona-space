"""Byte-exact v2 pack codec for the #2321 HF data-repo repack.

Generalizes the #1739 v1 pack recipe (``scripts/issue1739_pack.py``) to a
BYTE-EXACT codec (plan #2321 §3.2):

- one JSONL line per source file:
  ``{"src", "enc": "text"|"b64", "sha256", "bytes", "data"}`` — ``enc="text"``
  iff the raw bytes round-trip UTF-8 decode/encode identically (JSON stored as
  its raw *text*, never parsed); else ``enc="b64"``. Empty files are legal.
  All file types accepted (v1's non-``.json`` fail-loud is lifted).
- shards close at :data:`SHARD_MAX_BYTES` (9 MB — the non-LFS line-split
  discipline, `.claude/rules/upload-policy.md`) OR :data:`SHARD_MAX_MEMBERS`
  (4,000 — so one commit unit can pair each shard with its exact delete set,
  plan §3.5), whichever first.
- per-member ``(shard, byte offset, byte length)`` recorded at write time into
  per-group index parts ``<group>.indexNN.json`` (each ≤ 9 MB) + a top
  ``INDEX.json`` ``{group_key: {rel_dir, index_files, shard_files,
  n_members}}`` + ``pack_manifest.json`` (version 2).
- group key: v1 semantics (relative dir with ``/`` -> ``_``; ``root`` for
  top-level files); on a base-key collision (``a/b_c`` vs ``a_b/c``) the
  DETERMINISTIC disambiguation ``key + "-" + sha1(rel_dir)[:8]`` is decided at
  PACK time over the FULL key set, BEFORE any shard is written (C13), and each
  group's ``rel_dir`` is recorded in ``INDEX.json`` so the reader shim resolves
  collided groups via the RECORDED ``rel_dir``, never by re-deriving the key.

The decode path (:func:`decode_member_line` / :func:`unpack_shards_v2` /
:func:`extract_member_from_shard`) is the PRODUCTION decoder — the driver's
verify phase and the ``hub.py`` reader shim both call it (C19: never a
test-local reimplementation).

I18 test-mutation interlock: :func:`assert_test_mutation_interlock` refuses a
canonical-repo mutation from a test process before any network access — the
task-body constraint "Do NOT test it by deleting real artifacts" as one
enforced boundary instead of a per-test convention.

Pure stdlib on purpose: ``hub.py`` imports this module (reader shim), so this
module must never import ``hub``.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from pathlib import Path

PACK_FORMAT_VERSION = 2
SHARD_MAX_BYTES = 9_000_000  # non-LFS ceiling with margin (>10 MB force-routes to LFS)
SHARD_MAX_MEMBERS = 4_000  # one shard's delete set fits one commit unit (plan §3.5)
PACKED_DIRNAME = "packed"
INDEX_NAME = "INDEX.json"
MANIFEST_NAME = "pack_manifest.json"
UNITS_JOURNAL_NAME = "units.jsonl"
# JSON-envelope overhead allowance for the §3.3(b) selection estimate: key
# names + quotes + sha256 hex + a generous src-path allowance.
_LINE_OVERHEAD_BYTES = 512
# Known-text extensions for the selection estimate (plan §3.3(b): text
# ~x1.05, b64 x4/3). The estimate only SELECTS; the packer measures the real
# encoded line and fails loud past the cap either way.
_TEXT_EXTS = frozenset({".json", ".jsonl", ".txt", ".md", ".log", ".csv", ".tsv", ".yaml", ".yml"})

# ---------------------------------------------------------------------------
# I18 — test-process mutation interlock (#2321; land EARLY, before any test
# that exercises commit paths).
# ---------------------------------------------------------------------------

CANONICAL_HUB_REPOS = frozenset(
    {
        "superkaiba1/explore-persona-space-data",
        "superkaiba1/explore-persona-space",
    }
)


class TestMutationInterlockError(RuntimeError):
    """A test process attempted a canonical-repo mutation without a permit."""


def assert_test_mutation_interlock(repo_id: str) -> None:
    """I18: refuse a canonical-repo mutation from a TEST process (#2321).

    Called at the ENTRY of every guarded mutation path (the repack driver's
    ``commit_unit_probe_first`` and cap-probe commits) BEFORE any network
    access. Raises :class:`TestMutationInterlockError` when ALL of:

    - the process is running under pytest (``PYTEST_CURRENT_TEST`` set);
    - ``repo_id`` is one of :data:`CANONICAL_HUB_REPOS`;
    - the explicit apply permit ``EPM_I2321_TEST_APPLY_PERMIT=1`` is absent
      (never set in CI; the shared conftest actively deletes it for this
      task's test modules).

    Tests exercising commit paths use fake repo ids (or autospec'd fakes);
    the interlock makes "a test deleted real artifacts" structurally
    impossible rather than convention-dependent.
    """
    if "PYTEST_CURRENT_TEST" not in os.environ:
        return
    if repo_id not in CANONICAL_HUB_REPOS:
        return
    if os.environ.get("EPM_I2321_TEST_APPLY_PERMIT") == "1":
        return
    raise TestMutationInterlockError(
        f"I18 test-mutation interlock: refusing mutation of canonical repo {repo_id!r} "
        f"from a pytest process (PYTEST_CURRENT_TEST is set) without "
        f"EPM_I2321_TEST_APPLY_PERMIT=1. Tests must target fake repo ids — "
        f"'Do NOT test it by deleting real artifacts' (#2321)."
    )


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class PackError(RuntimeError):
    """Base class for v2 pack codec failures (always fail-loud)."""


class OversizeMemberError(PackError):
    """A member's ENCODED line exceeds the shard byte cap — it must stay unpacked."""


class AnchorMismatchError(PackError):
    """Packed bytes do not match the caller-supplied census anchor (C8 TOCTOU)."""


class GroupKeyCollisionError(PackError):
    """Group-key disambiguation failed to produce unique keys (pathological)."""


# ---------------------------------------------------------------------------
# Primitive codecs
# ---------------------------------------------------------------------------


def git_blob_sha1(data: bytes) -> str:
    """Git blob object id of ``data``: ``sha1(b"blob <len>\\0" + data)`` (A6).

    Matches the Hub's server-side ``blob_id`` for non-LFS files (verified live
    against the data repo at rev ``7d3ac543a5a4``, plan A6).
    """
    h = hashlib.sha1()
    h.update(b"blob %d\x00" % len(data))
    h.update(data)
    return h.hexdigest()


def member_enc(data: bytes) -> str:
    """``"text"`` iff ``data`` UTF-8 round-trips identically, else ``"b64"``."""
    try:
        if data.decode("utf-8").encode("utf-8") == data:
            return "text"
    except UnicodeDecodeError:
        pass
    return "b64"


def encode_member_line(src: str, data: bytes) -> bytes:
    """Encode one source file as a single JSONL line (trailing ``\\n`` included).

    ``enc="text"`` iff the bytes UTF-8 round-trip identically; else ``"b64"``.
    Deterministic (sorted keys, ``ensure_ascii=False``) so re-packs are
    byte-identical. ``json.dumps`` escapes every raw newline/control char in
    string values, so the ONLY raw ``0x0A`` byte in the output is the final
    line terminator (offset arithmetic depends on this).
    """
    if "\\" in src or src.startswith("/") or ".." in src.split("/"):
        raise PackError(f"unsafe member src path: {src!r}")
    enc = member_enc(data)
    payload = data.decode("utf-8") if enc == "text" else base64.b64encode(data).decode("ascii")
    rec = {
        "src": src,
        "enc": enc,
        "sha256": hashlib.sha256(data).hexdigest(),
        "bytes": len(data),
        "data": payload,
    }
    return (json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n").encode("utf-8")


def decode_member_line(line: bytes | str) -> tuple[str, bytes]:
    """PRODUCTION decoder (C19): parse one member line, verify, return bytes.

    Verifies BOTH the recorded byte length and the sha256 of the decoded raw
    bytes — a truncated/corrupt line always fails loud, never returns wrong
    bytes. Accepts a trailing newline.
    """
    if isinstance(line, bytes):
        line = line.decode("utf-8")
    rec = json.loads(line)
    src = rec["src"]
    enc = rec["enc"]
    if enc == "text":
        data = rec["data"].encode("utf-8")
    elif enc == "b64":
        data = base64.b64decode(rec["data"], validate=True)
    else:
        raise PackError(f"unknown enc {enc!r} for member {src!r}")
    if len(data) != rec["bytes"]:
        raise PackError(f"member {src!r}: decoded {len(data)} bytes != recorded {rec['bytes']}")
    got = hashlib.sha256(data).hexdigest()
    if got != rec["sha256"]:
        raise PackError(f"member {src!r}: sha256 mismatch ({got[:12]} != {rec['sha256'][:12]})")
    return src, data


def estimate_encoded_line_bytes(size_bytes: int, filename: str) -> int:
    """§3.3(b) SELECTION estimate of a member's encoded line size.

    Known-text extensions use the ~x1.05 escaping factor; everything else the
    b64 x4/3 factor (conservative for text-content files with binary-looking
    extensions — such a file is merely left unpacked, which is safe). The
    packer re-measures the REAL encoded line and raises
    :class:`OversizeMemberError` past the cap regardless.
    """
    ext = Path(filename).suffix.lower()
    if ext in _TEXT_EXTS:
        est = int(size_bytes * 1.05)
    else:
        est = (size_bytes * 4 + 2) // 3
    return est + _LINE_OVERHEAD_BYTES


# ---------------------------------------------------------------------------
# Grouping (C13: collision mapping decided over the FULL key set, pre-write)
# ---------------------------------------------------------------------------


def derive_group_keys(rel_dirs: Iterable[str]) -> dict[str, str]:
    """Map each relative dir to its final group key, DETERMINISTICALLY (C13).

    Base key = v1 semantics (``rel_dir.replace("/", "_")``; ``root`` for
    top-level files, spelled ``""`` or ``"."``). Every rel_dir whose base key
    collides with another's gets ``base + "-" + sha1(rel_dir)[:8]`` — applied
    over the FULL set before any shard is written, so the mapping can never
    depend on discovery order. Raises :class:`GroupKeyCollisionError` if the
    disambiguated keys still collide (pathological)."""
    norm = {("" if rd == "." else rd.strip("/")): None for rd in rel_dirs}
    base: dict[str, str] = {}
    for rd in norm:
        base[rd] = "root" if rd == "" else rd.replace("/", "_")
    by_key: dict[str, list[str]] = defaultdict(list)
    for rd, k in base.items():
        by_key[k].append(rd)
    out: dict[str, str] = {}
    for k, rds in sorted(by_key.items()):
        if len(rds) == 1:
            out[rds[0]] = k
        else:
            for rd in sorted(rds):
                out[rd] = f"{k}-{hashlib.sha1(rd.encode('utf-8')).hexdigest()[:8]}"
    if len(set(out.values())) != len(out):
        raise GroupKeyCollisionError(
            f"group-key disambiguation still collides over {sorted(out.values())!r}"
        )
    return out


def shard_name(group_key: str, idx: int) -> str:
    """v1-compatible shard file name (``<group>.shardNN.jsonl``)."""
    return f"{group_key}.shard{idx:02d}.jsonl"


def index_part_name(group_key: str, idx: int) -> str:
    """Per-group index part file name (``<group>.indexNN.json``)."""
    return f"{group_key}.index{idx:02d}.json"


# ---------------------------------------------------------------------------
# Pack result model
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MemberIndexEntry:
    """Per-member location + integrity record (the shim's resolution unit)."""

    shard: str  # shard file name (relative to packed/)
    offset: int  # byte offset of the line start within the shard
    length: int  # byte length of the line INCLUDING the trailing newline
    sha256: str  # sha256 of the raw member bytes
    enc: str  # "text" | "b64"
    size: int  # raw member byte count

    def to_json(self) -> dict:
        """JSON form stored in index parts."""
        return {
            "shard": self.shard,
            "offset": self.offset,
            "length": self.length,
            "sha256": self.sha256,
            "enc": self.enc,
            "bytes": self.size,
        }

    @classmethod
    def from_json(cls, d: Mapping) -> MemberIndexEntry:
        """Inverse of :meth:`to_json` (fail-loud on missing keys)."""
        return cls(
            shard=d["shard"],
            offset=int(d["offset"]),
            length=int(d["length"]),
            sha256=d["sha256"],
            enc=d["enc"],
            size=int(d["bytes"]),
        )


@dataclass
class PackGroup:
    """One packed directory group."""

    key: str
    rel_dir: str
    shard_files: list[str] = field(default_factory=list)
    index_files: list[str] = field(default_factory=list)
    n_members: int = 0


@dataclass
class PackResult:
    """Everything a caller needs after :func:`pack_tree_v2`."""

    pack_dir: Path
    groups: dict[str, PackGroup]
    census_key: str
    n_members: int
    reused: bool = False

    @property
    def manifest_path(self) -> Path:
        return self.pack_dir / MANIFEST_NAME

    @property
    def index_path(self) -> Path:
        return self.pack_dir / INDEX_NAME

    def all_shard_files(self) -> list[str]:
        return [s for g in sorted(self.groups) for s in self.groups[g].shard_files]

    def all_index_files(self) -> list[str]:
        return [s for g in sorted(self.groups) for s in self.groups[g].index_files]


# ---------------------------------------------------------------------------
# Packing
# ---------------------------------------------------------------------------


def _stat_census(raw_root: Path, rel_files: list[str]) -> str:
    """Stat-based census key over ``(relpath, size, mtime_ns)`` (v1 property).

    Keyed for IDEMPOTENT re-pack reuse only — content integrity is carried by
    the per-member sha256 + the caller's C8 anchors, never by this key.
    """
    h = hashlib.sha256()
    for rel in sorted(rel_files):
        st = (raw_root / rel).stat()
        h.update(f"{rel}\x00{st.st_size}\x00{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def _atomic_write(path: Path, data: bytes) -> None:
    """tmp + ``os.replace`` in the destination dir (same filesystem)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(fd, "wb") as f:
            f.write(data)
        os.replace(tmp, path)
    except BaseException:
        Path(tmp).unlink(missing_ok=True)
        raise


def _check_anchor(rel: str, data: bytes, anchors: Mapping[str, tuple[str, str]] | None) -> None:
    """C8 pack-time TOCTOU closure: re-assert the census anchor on packed bytes."""
    if anchors is None:
        return
    entry = anchors.get(rel)
    if entry is None:
        raise AnchorMismatchError(f"member {rel!r} has no census anchor (C8 coverage gap)")
    kind, want = entry
    if kind == "gitblob":
        got = git_blob_sha1(data)
    elif kind == "sha256":
        got = hashlib.sha256(data).hexdigest()
    else:
        raise AnchorMismatchError(f"unknown anchor kind {kind!r} for {rel!r}")
    if got != want:
        raise AnchorMismatchError(
            f"member {rel!r}: packed bytes {kind} {got[:12]}... != census anchor "
            f"{want[:12]}... — local bytes changed between download and pack (C8)"
        )


def pack_tree_v2(
    raw_root: Path,
    pack_dir: Path,
    *,
    candidates: Iterable[str] | None = None,
    anchors: Mapping[str, tuple[str, str]] | None = None,
    source_revision: str | None = None,
    git_commit: str | None = None,
    shard_max_bytes: int = SHARD_MAX_BYTES,
    shard_max_members: int = SHARD_MAX_MEMBERS,
) -> PackResult:
    """Pack ``raw_root`` into byte-exact v2 shards + index under ``pack_dir``.

    Args:
        raw_root: local tree of downloaded originals (prefix root).
        pack_dir: output dir (the future ``<prefix>/packed/`` content).
        candidates: relative paths to pack (default: every regular file under
            ``raw_root``). The DRIVER pre-filters per §3.3; the packer still
            fail-louds on any member whose REAL encoded line exceeds
            ``shard_max_bytes`` (:class:`OversizeMemberError` — the estimate
            was wrong; nothing has been deleted, the file simply must stay
            unpacked).
        anchors: optional ``rel -> (kind, hexdigest)`` census anchors
            (``kind`` in ``{"gitblob", "sha256"}``), re-asserted on the bytes
            actually packed (C8).
        source_revision / git_commit: provenance recorded in the manifest.
        shard_max_bytes / shard_max_members: injectable caps (tests).

    Idempotent: an existing manifest whose stat census matches is REUSED
    (``result.reused=True``) after verifying every listed shard's sha256;
    stray v2-shaped files not in the manifest raise (never silently absorbed).
    Collision mapping is decided over the FULL group-key set BEFORE any shard
    is written (C13); each group's ``rel_dir`` is recorded in ``INDEX.json``.
    """
    raw_root = Path(raw_root)
    pack_dir = Path(pack_dir)
    if candidates is None:
        rel_files = [
            p.relative_to(raw_root).as_posix() for p in sorted(raw_root.rglob("*")) if p.is_file()
        ]
    else:
        rel_files = sorted(candidates)
    if not rel_files:
        raise PackError(f"no pack candidates under {raw_root} — refusing an empty pack")
    census_key = _stat_census(raw_root, rel_files)

    manifest_path = pack_dir / MANIFEST_NAME
    if manifest_path.exists():
        man = json.loads(manifest_path.read_text(encoding="utf-8"))
        if man.get("version") == PACK_FORMAT_VERSION and man.get("census_key") == census_key:
            return _reuse_existing_pack(pack_dir, man, census_key)

    # --- C13: derive ALL group keys first, before any shard is written. ---
    by_rel_dir: dict[str, list[str]] = defaultdict(list)
    for rel in rel_files:
        parent = rel.rsplit("/", 1)[0] if "/" in rel else ""
        by_rel_dir[parent].append(rel)
    key_by_rel_dir = derive_group_keys(by_rel_dir.keys())

    pack_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, PackGroup] = {}
    shard_meta: dict[str, dict] = {}
    n_members_total = 0
    for rel_dir in sorted(by_rel_dir):
        key = key_by_rel_dir["" if rel_dir == "." else rel_dir]
        group = PackGroup(key=key, rel_dir=rel_dir)
        entries: dict[str, MemberIndexEntry] = {}
        shard_idx = 0
        buf: list[bytes] = []
        buf_bytes = 0
        buf_members = 0

        def _flush() -> None:
            nonlocal shard_idx, buf, buf_bytes, buf_members
            if not buf:
                return
            name = shard_name(key, shard_idx)
            blob = b"".join(buf)
            _atomic_write(pack_dir / name, blob)
            shard_meta[name] = {
                "sha256": hashlib.sha256(blob).hexdigest(),
                "n_lines": buf_members,
                "bytes": len(blob),
            }
            group.shard_files.append(name)
            shard_idx += 1
            buf, buf_bytes, buf_members = [], 0, 0

        for rel in sorted(by_rel_dir[rel_dir]):
            data = (raw_root / rel).read_bytes()
            _check_anchor(rel, data, anchors)
            line = encode_member_line(rel, data)
            if len(line) > shard_max_bytes:
                raise OversizeMemberError(
                    f"member {rel!r}: encoded line {len(line)} B > shard cap "
                    f"{shard_max_bytes} B — must stay unpacked (§3.3(b))"
                )
            if buf and (
                buf_bytes + len(line) > shard_max_bytes or buf_members >= shard_max_members
            ):
                _flush()
            entries[rel] = MemberIndexEntry(
                shard=shard_name(key, shard_idx),
                offset=buf_bytes,
                length=len(line),
                sha256=hashlib.sha256(data).hexdigest(),
                enc=member_enc(data),
                size=len(data),
            )
            buf.append(line)
            buf_bytes += len(line)
            buf_members += 1
        _flush()
        group.n_members = len(entries)
        n_members_total += len(entries)
        group.index_files = _write_index_parts(pack_dir, key, rel_dir, entries, shard_max_bytes)
        groups[key] = group

    top_index = {
        "version": PACK_FORMAT_VERSION,
        "groups": {
            k: {
                "rel_dir": g.rel_dir,
                "index_files": g.index_files,
                "shard_files": g.shard_files,
                "n_members": g.n_members,
            }
            for k, g in sorted(groups.items())
        },
    }
    _atomic_write(
        pack_dir / INDEX_NAME,
        json.dumps(top_index, indent=1, sort_keys=True).encode("utf-8"),
    )
    manifest = {
        "version": PACK_FORMAT_VERSION,
        "census_key": census_key,
        "source_revision": source_revision,
        "git_commit": git_commit,
        "n_members": n_members_total,
        "shards": dict(sorted(shard_meta.items())),
        "groups": top_index["groups"],
    }
    _atomic_write(manifest_path, json.dumps(manifest, indent=1, sort_keys=True).encode("utf-8"))
    return PackResult(
        pack_dir=pack_dir, groups=groups, census_key=census_key, n_members=n_members_total
    )


def _write_index_parts(
    pack_dir: Path,
    key: str,
    rel_dir: str,
    entries: dict[str, MemberIndexEntry],
    shard_max_bytes: int,
) -> list[str]:
    """Split a group's member index into ≤ ``shard_max_bytes`` JSON parts."""
    part_names: list[str] = []
    part: dict[str, dict] = {}
    part_bytes = 0
    part_idx = 0
    # Rough fixed envelope for the part header; per-entry cost measured on the
    # serialized entry itself. Exactness is not needed — only staying safely
    # under the non-LFS line (the 9 MB cap has ~1 MB of headroom to 10 MB).
    envelope = 256 + len(key) + len(rel_dir)

    def _flush_part() -> None:
        nonlocal part, part_bytes, part_idx
        if not part:
            return
        name = index_part_name(key, part_idx)
        doc = {
            "version": PACK_FORMAT_VERSION,
            "group": key,
            "rel_dir": rel_dir,
            "members": {src: e for src, e in sorted(part.items())},
        }
        _atomic_write(pack_dir / name, json.dumps(doc, indent=1, sort_keys=True).encode("utf-8"))
        part_names.append(name)
        part_idx += 1
        part, part_bytes = {}, 0

    for src in sorted(entries):
        ejson = entries[src].to_json()
        cost = len(json.dumps({src: ejson}))
        if part and envelope + part_bytes + cost > shard_max_bytes:
            _flush_part()
        part[src] = ejson
        part_bytes += cost
    _flush_part()
    return part_names


def _reuse_existing_pack(pack_dir: Path, man: dict, census_key: str) -> PackResult:
    """Census-matched reuse: verify listed shards, refuse stray v2 files."""
    groups: dict[str, PackGroup] = {}
    listed: set[str] = {MANIFEST_NAME, INDEX_NAME}
    for k, g in man["groups"].items():
        groups[k] = PackGroup(
            key=k,
            rel_dir=g["rel_dir"],
            shard_files=list(g["shard_files"]),
            index_files=list(g["index_files"]),
            n_members=int(g["n_members"]),
        )
        listed.update(g["shard_files"])
        listed.update(g["index_files"])
    for name, meta in man["shards"].items():
        blob = (pack_dir / name).read_bytes()
        got = hashlib.sha256(blob).hexdigest()
        if got != meta["sha256"]:
            raise PackError(
                f"existing shard {name} sha256 {got[:12]}... != manifest "
                f"{meta['sha256'][:12]}... — refusing census-matched reuse"
            )
    stray = [
        p.name
        for p in sorted(pack_dir.iterdir())
        if p.is_file() and p.name not in listed and p.name != UNITS_JOURNAL_NAME
    ]
    if stray:
        raise PackError(f"stray files in pack dir not listed by manifest: {stray[:10]}")
    return PackResult(
        pack_dir=pack_dir,
        groups=groups,
        census_key=census_key,
        n_members=int(man["n_members"]),
        reused=True,
    )


# ---------------------------------------------------------------------------
# Unpacking / extraction (the PRODUCTION decode path, C19)
# ---------------------------------------------------------------------------


def iter_shard_lines(shard_path: Path) -> Iterator[tuple[int, int, bytes]]:
    """Yield ``(offset, length, line_bytes)`` per member line, binary-exact.

    Splits on raw ``0x0A`` only — safe because :func:`encode_member_line`
    guarantees the terminator is the only raw newline byte per line (JSON
    escapes embedded newlines), and binary iteration sidesteps the
    ``str.splitlines()`` U+2028 shredding class (#950).
    """
    blob = Path(shard_path).read_bytes()
    offset = 0
    while offset < len(blob):
        nl = blob.find(b"\n", offset)
        if nl == -1:
            raise PackError(f"shard {shard_path} ends without a newline (truncated?)")
        yield offset, nl + 1 - offset, blob[offset : nl + 1]
        offset = nl + 1


def read_shard_member_srcs(shard_path: Path) -> list[str]:
    """Member ``src`` paths of a shard, in order (for the census bijection)."""
    out = []
    for _off, _length, line in iter_shard_lines(shard_path):
        rec = json.loads(line.decode("utf-8"))
        out.append(rec["src"])
    return out


def extract_member_from_shard(shard_path: Path, offset: int, length: int) -> tuple[str, bytes]:
    """Extract + verify ONE member by recorded offset (the shim's read path)."""
    with open(shard_path, "rb") as f:
        f.seek(offset)
        line = f.read(length)
    if len(line) != length:
        raise PackError(
            f"short read at {shard_path}:{offset}+{length} (got {len(line)} B) — "
            f"truncated shard or stale index"
        )
    return decode_member_line(line)


def _restore_file(out_root: Path, src: str, data: bytes) -> Path:
    """Atomic restore; NEVER overwrite an existing file with different bytes."""
    rel = Path(src)
    if rel.is_absolute() or ".." in rel.parts:
        raise PackError(f"unsafe member src path: {src!r}")
    dest = out_root / rel
    if dest.exists():
        if dest.read_bytes() == data:
            return dest
        raise PackError(f"refusing to overwrite differing existing file: {dest}")
    _atomic_write(dest, data)
    return dest


def unpack_shards_v2(pack_dir: Path, out_root: Path) -> int:
    """Unpack EVERY manifest-listed shard to ``out_root``; return member count.

    Verifies each shard's sha256 + line count against ``pack_manifest.json``,
    then decodes every member via :func:`decode_member_line` (per-member
    sha256 + length asserts) and restores it atomically (never overwriting a
    differing existing file). This is the driver verify phase's decode path
    AND the same parse+decode the shim uses (C19).
    """
    pack_dir = Path(pack_dir)
    out_root = Path(out_root)
    man = json.loads((pack_dir / MANIFEST_NAME).read_text(encoding="utf-8"))
    if man.get("version") != PACK_FORMAT_VERSION:
        raise PackError(f"not a v2 pack manifest: version={man.get('version')!r}")
    n = 0
    for name, meta in sorted(man["shards"].items()):
        shard_path = pack_dir / name
        blob = shard_path.read_bytes()
        got = hashlib.sha256(blob).hexdigest()
        if got != meta["sha256"]:
            raise PackError(
                f"shard {name}: sha256 {got[:12]}... != manifest {meta['sha256'][:12]}..."
            )
        lines = 0
        for _off, _length, line in iter_shard_lines(shard_path):
            src, data = decode_member_line(line)
            _restore_file(out_root, src, data)
            lines += 1
        if lines != meta["n_lines"]:
            raise PackError(f"shard {name}: {lines} lines != manifest {meta['n_lines']}")
        n += lines
    return n


# ---------------------------------------------------------------------------
# Index loading (shared with the hub.py reader shim)
# ---------------------------------------------------------------------------


def load_top_index(text: str, *, what: str = "INDEX.json") -> dict[str, dict]:
    """Parse a top ``INDEX.json``; return the ``groups`` mapping (fail-loud)."""
    doc = json.loads(text)
    if doc.get("version") != PACK_FORMAT_VERSION:
        raise PackError(f"{what}: not a v2 top index (version={doc.get('version')!r})")
    groups = doc.get("groups")
    if not isinstance(groups, dict) or not groups:
        raise PackError(f"{what}: empty/malformed groups mapping")
    return groups


def load_index_part(text: str, *, what: str = "index part") -> dict[str, MemberIndexEntry]:
    """Parse a per-group index part; return ``src -> MemberIndexEntry``."""
    doc = json.loads(text)
    if doc.get("version") != PACK_FORMAT_VERSION:
        raise PackError(f"{what}: not a v2 index part (version={doc.get('version')!r})")
    members = doc.get("members")
    if not isinstance(members, dict):
        raise PackError(f"{what}: malformed members mapping")
    return {src: MemberIndexEntry.from_json(e) for src, e in members.items()}
