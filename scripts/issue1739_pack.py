"""Pack per-(context, seed) rollout JSONs into <= 9 MB jsonl line-shards (#1739 r4).

The #1739 generation module writes ONE JSON per (context, seed) — ~255k
small files across 3 behaviors — and the naive ``--stage raw``
``upload_folder`` of that tree trips the Hub's 10k-files-per-directory
commit cap (``hub.HubDirFileCountError``: 115,941 files staged into
``issue1739_ctxmap/raw_completions/labeling/hallucination``). The guard's
own remedy applies (``orchestrate/hub.py`` ~L1066 +
``.claude/rules/upload-policy.md``): PACK the small text/JSON files into
<= 9 MB ``<group>.shardNN.jsonl`` line-shards — one line per source file,
``{"src": "<path relative to raw root>", "doc": <original JSON>}`` — plus
a ``pack_manifest.json`` beside the shards, then upload the tiny shard
set in one bulk commit.

Properties:

- **Deterministic**: files are processed in sorted relative-path order and
  shard boundaries depend only on (ordering, serialized bytes, cap), so a
  re-pack of unchanged inputs is byte-identical.
- **Memory-bounded**: files stream ONE at a time into the open shard
  handle; the tree is never held in RAM.
- **Idempotent**: a stat census (relpath, size, mtime_ns) per group is
  recorded in the manifest; a re-run reuses groups whose census + shard
  files still match and repacks only the rest.
- **Content hygiene**: no file CONTENT is ever printed or logged — counts,
  names, and digests only (rollout text may be real-corpus derived).
"""

from __future__ import annotations

import datetime as _dt
import hashlib
import json
import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger(__name__)

# < 9 MB per shard: stays on the non-LFS Hub path (the Hub force-routes any
# > 10 MB blob to LFS regardless of extension — upload-policy.md).
SHARD_MAX_BYTES = 9_000_000
MANIFEST_NAME = "pack_manifest.json"
MANIFEST_VERSION = 1
_PROGRESS_EVERY = 20_000  # per-unit progress line cadence within a big group


def _shard_name(group_key: str, idx: int) -> str:
    """Deterministic shard filename; manifest order is authoritative >= 100."""
    return f"{group_key}.shard{idx:02d}.jsonl"


def group_files(raw_root: Path) -> dict[str, dict]:
    """Map shard-group key -> {"rel_dir": str, "files": [Path, ...] sorted}.

    One group per directory (relative to ``raw_root``) that directly
    contains files; key = relative dir with ``/`` -> ``_`` (``root`` for
    top-level files). Fails loud on any non-``.json`` file (nothing under
    the raw tree may be silently dropped — extend the packer instead) and
    on a group-key collision (``a/b_c`` vs ``a_b/c``).
    """
    groups: dict[str, dict] = {}
    non_json: list[str] = []
    for p in sorted(raw_root.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(raw_root)
        if p.suffix != ".json":
            non_json.append(rel.as_posix())
            continue
        rel_dir = rel.parent.as_posix()
        key = "root" if rel_dir == "." else rel_dir.replace("/", "_")
        entry = groups.setdefault(key, {"rel_dir": rel_dir, "files": []})
        if entry["rel_dir"] != rel_dir:
            raise ValueError(
                f"shard-group key collision: {key!r} maps both {entry['rel_dir']!r} "
                f"and {rel_dir!r} — rename one directory"
            )
        entry["files"].append(p)
    if non_json:
        raise ValueError(
            f"{len(non_json)} non-.json file(s) under {raw_root} "
            f"(first: {non_json[:5]}) — the packer covers per-context JSONs only; "
            "extend it rather than silently dropping rollout text"
        )
    for entry in groups.values():
        entry["files"].sort()
    return groups


def group_census(raw_root: Path, files: list[Path]) -> str:
    """Cheap stat-based census: sha256 over (relpath, size, mtime_ns) rows.

    Detects added/removed/rewritten files without reading contents (a
    115k-file group censuses in ~1 s). A fresh checkout resets mtimes and
    at worst forces one redundant — deterministic — repack.
    """
    h = hashlib.sha256()
    for p in files:
        st = p.stat()
        h.update(p.relative_to(raw_root).as_posix().encode("utf-8"))
        h.update(f"\x00{st.st_size}\x00{st.st_mtime_ns}\n".encode())
    return h.hexdigest()


def pack_group(
    raw_root: Path,
    group_key: str,
    files: list[Path],
    pack_root: Path,
    *,
    shard_max_bytes: int = SHARD_MAX_BYTES,
) -> list[dict]:
    """Stream one group's files into <= ``shard_max_bytes`` line-shards.

    Returns per-shard entries ``{name, n_lines, bytes, sha256}``. A single
    doc whose serialized line exceeds the cap gets its own shard (warned —
    a line cannot be split); every other shard stays under the cap.
    """
    shards: list[dict] = []
    cur: dict | None = None

    def _close() -> None:
        nonlocal cur
        if cur is None:
            return
        cur["fh"].close()
        shards.append(
            {
                "name": cur["name"],
                "n_lines": cur["n_lines"],
                "bytes": cur["bytes"],
                "sha256": cur["hasher"].hexdigest(),
            }
        )
        cur = None

    def _open() -> None:
        nonlocal cur
        name = _shard_name(group_key, len(shards))
        cur = {
            "name": name,
            "fh": (pack_root / name).open("wb"),
            "hasher": hashlib.sha256(),
            "n_lines": 0,
            "bytes": 0,
        }

    for n, p in enumerate(files, start=1):
        rel = p.relative_to(raw_root).as_posix()
        with p.open("rb") as f:
            doc = json.load(f)  # fail loud on a malformed source file
        line = (json.dumps({"src": rel, "doc": doc}, ensure_ascii=False) + "\n").encode("utf-8")
        if len(line) > shard_max_bytes:
            logger.warning(
                "[pack] %s serializes to %s bytes > shard cap %s — gets its own shard",
                rel,
                f"{len(line):,}",
                f"{shard_max_bytes:,}",
            )
        if cur is not None and cur["bytes"] + len(line) > shard_max_bytes:
            _close()
        if cur is None:
            _open()
        assert cur is not None
        cur["fh"].write(line)
        cur["hasher"].update(line)
        cur["bytes"] += len(line)
        cur["n_lines"] += 1
        if n % _PROGRESS_EVERY == 0:
            print(f"[pack] {group_key}: {n}/{len(files)} files", flush=True)
    _close()
    return shards


def _delete_group_shards(pack_root: Path, group_key: str) -> None:
    prefix = f"{group_key}.shard"
    for p in pack_root.iterdir():
        if p.is_file() and p.name.startswith(prefix) and p.name.endswith(".jsonl"):
            p.unlink()


def _load_manifest(pack_root: Path) -> dict | None:
    path = pack_root / MANIFEST_NAME
    if not path.exists():
        return None
    try:
        m = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        logger.warning("[pack] existing manifest unreadable (%s) — repacking everything", e)
        return None
    if m.get("version") != MANIFEST_VERSION:
        return None
    return m


def _git_commit() -> str:
    """Reproducibility metadata (best-effort — never blocks a pack)."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[1],
        )
        return out.stdout.strip()
    except Exception as e:  # noqa: BLE001 — metadata only, pack must proceed
        logger.warning("[pack] git commit unresolved: %s", e)
        return "unknown"


def pack_raw_tree(
    raw_root: Path | str,
    pack_root: Path | str,
    *,
    shard_max_bytes: int = SHARD_MAX_BYTES,
) -> dict:
    """Pack ``raw_root/**`` into line-shards under ``pack_root``; return manifest.

    Idempotent per group: an existing manifest entry whose census matches
    the current tree AND whose shard files are present at the recorded
    sizes is reused untouched; everything else is repacked. Shards of
    groups that vanished from the raw tree are removed, and the final
    ``pack_root`` is asserted to contain EXACTLY manifest + shards (a stray
    file would ride the bulk upload).
    """
    raw_root = Path(raw_root)
    pack_root = Path(pack_root)
    if not raw_root.is_dir():
        raise SystemExit(f"[pack] raw root {raw_root} does not exist")
    if pack_root.resolve().is_relative_to(raw_root.resolve()):
        raise SystemExit(f"[pack] pack root {pack_root} must live OUTSIDE raw root {raw_root}")
    pack_root.mkdir(parents=True, exist_ok=True)

    groups = group_files(raw_root)
    if not groups:
        raise SystemExit(f"[pack] nothing to pack under {raw_root}")

    prev = _load_manifest(pack_root)
    prev_groups = (prev or {}).get("groups", {})
    prev_cap_matches = (prev or {}).get("shard_max_bytes") == shard_max_bytes

    out_groups: dict[str, dict] = {}
    reused: list[str] = []
    repacked: list[str] = []
    for key in sorted(groups):
        files = groups[key]["files"]
        cens = group_census(raw_root, files)
        old = prev_groups.get(key)
        if (
            prev_cap_matches
            and old is not None
            and old.get("census_sha256") == cens
            and all(
                (pack_root / s["name"]).is_file()
                and (pack_root / s["name"]).stat().st_size == s["bytes"]
                for s in old.get("shards", [])
            )
        ):
            out_groups[key] = old
            reused.append(key)
            print(
                f"[pack] {key}: census match — reusing {old['n_shards']} shard(s) "
                f"({old['n_files']} files)",
                flush=True,
            )
            continue
        _delete_group_shards(pack_root, key)
        shards = pack_group(raw_root, key, files, pack_root, shard_max_bytes=shard_max_bytes)
        out_groups[key] = {
            "rel_dir": groups[key]["rel_dir"],
            "n_files": len(files),
            "census_sha256": cens,
            "n_shards": len(shards),
            "shards": shards,
        }
        repacked.append(key)
        print(f"[pack] {key}: packed {len(files)} files -> {len(shards)} shard(s)", flush=True)

    for key in sorted(set(prev_groups) - set(groups)):
        _delete_group_shards(pack_root, key)
        print(f"[pack] {key}: raw group gone — removed stale shards", flush=True)

    manifest = {
        "version": MANIFEST_VERSION,
        "raw_root": str(raw_root),
        "shard_max_bytes": shard_max_bytes,
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
        "git_commit": _git_commit(),
        "python": sys.version.split()[0],
        "reused_groups": reused,
        "repacked_groups": repacked,
        "groups": out_groups,
    }
    tmp = pack_root / (MANIFEST_NAME + ".tmp")
    tmp.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, pack_root / MANIFEST_NAME)

    expected = {MANIFEST_NAME} | {s["name"] for g in out_groups.values() for s in g["shards"]}
    actual = {p.name for p in pack_root.iterdir() if p.is_file()}
    stray = sorted(actual - expected)
    if stray:
        raise SystemExit(
            f"[pack] {len(stray)} stray file(s) in {pack_root} (first: {stray[:5]}) — "
            "they would ride the bulk upload; remove them or point --pack-root elsewhere"
        )

    n_files = sum(g["n_files"] for g in out_groups.values())
    n_shards = sum(g["n_shards"] for g in out_groups.values())
    print(
        f"[pack] done: {n_files} files -> {n_shards} shard(s) across {len(out_groups)} "
        f"group(s) (reused {len(reused)}, repacked {len(repacked)})",
        flush=True,
    )
    return manifest
