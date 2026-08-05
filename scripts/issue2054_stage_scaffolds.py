#!/usr/bin/env python
"""Stage the ADMITTED scaffold pools from HF for Phase-B/C/D consumption.

For each requested variant (default: the 5-cell plan §4 panel,
``issue2054_phase_a.DEFAULT_VARIANTS``), assemble
``data/issue_2054/scaffolds/<v>/scaffolds_<v>.jsonl`` from the r7 SHARDED
ADMITTED form on the HF data repo
(``issue2054_lattice/scaffolds/<v>/scaffolds_<v>.shardNN.jsonl`` +
``scaffolds_<v>.manifest.json``), then verify the assembled pool against the
r7 admission record (``kept.json``, staged fresh from the top of the
scaffolds prefix): assembled row count == the variant's admitted count AND
conv_id SET equality with the admitted set (not just counts).

r6 lesson (why this script is manifest-driven ONLY): each HF variant dir
ALSO holds a STALE UNSHARDED ``scaffolds_<v>.jsonl`` — residue of a
prior-round upload — and a stager that resolves the plain hub name picks it
up silently (the r6 defect). Staging reuses the r9/r10-hardened helper
``issue2054_build_answers._stage_sharded_jsonl`` (manifest-first, per-shard
sha256 verification, exact in-order concatenation); it refuses loudly when
the manifest is absent and NEVER reads the unsharded hub name. Re-stage
semantics are AUTHORITATIVE: the Hub is the source of truth — every run
re-downloads (``overwrite=True`` throughout) and atomically overwrites the
local assembly, so a stale local file can never win.

Logging is digest-only (row counts, sha256s, byte sizes — never scaffold
text). Emits ``[phase=stage_scaffolds]`` breadcrumbs terminating in
``[phase=done]`` ONLY on full success (every variant staged AND verified).

Exit codes: 0 = success; 1 = staging / verification failure (fail-loud,
named variant + counts on stderr); 2 = argparse usage error.

Usage (production — Phase-B prerequisite staging):
  uv run python scripts/issue2054_stage_scaffolds.py

Smoke (single smallest variant):
  uv run python scripts/issue2054_stage_scaffolds.py --variants char_vex
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue2054_build_answers as ba  # noqa: E402
import issue2054_phase_a as pa  # noqa: E402


def _log(msg: str) -> None:
    print(f"[phase=stage_scaffolds] {msg}", flush=True)


def stage_kept_json(dest_root: Path) -> tuple[Path, dict]:
    """Stage kept.json (the r7 admission record) fresh from the Hub.

    Always ``overwrite=True`` — the record is the verification ground truth,
    so a stale local copy must never win. Returns (path, parsed dict);
    raises on a record with no ``variants`` map.
    """
    from explore_persona_space.orchestrate.hub import stage_hub_file

    kept_path = stage_hub_file(
        pa.HF_DATA_REPO,
        f"{ba.SCAFFOLDS_PREFIX}/kept.json",
        dest_root / "kept.json",
        repo_type="dataset",
        overwrite=True,
    )
    kept = json.loads(kept_path.read_text(encoding="utf-8"))
    if not (kept.get("variants") or {}):
        raise RuntimeError(f"kept.json carries no 'variants' map: {kept_path}")
    _log(
        f"staged kept.json: sha256={hashlib.sha256(kept_path.read_bytes()).hexdigest()[:12]}... "
        f"({kept_path.stat().st_size} B)"
    )
    return kept_path, kept


def admitted_ids_for(kept: dict, variant: str) -> list[str]:
    """The variant's admitted conv_id LIST from kept.json (fail-loud).

    Returns the raw list (count semantics) after asserting internal
    integrity: non-empty AND duplicate-free (the admission writer emits one
    conv_id per admitted row, so a duplicate means a corrupt record).
    """
    rec = (kept.get("variants") or {}).get(variant)
    if rec is None:
        raise RuntimeError(f"kept.json has no admission record for variant {variant!r}")
    ids = [str(x) for x in (rec.get("admitted_conv_ids") or [])]
    if not ids:
        raise RuntimeError(f"kept.json admitted_conv_ids EMPTY for variant {variant!r}")
    if len(set(ids)) != len(ids):
        raise RuntimeError(
            f"kept.json admitted_conv_ids for {variant!r} carries duplicates "
            f"({len(ids)} entries, {len(set(ids))} unique) — corrupt admission record"
        )
    return ids


def verify_staged_variant(pool_path: Path, variant: str, admitted_ids: list[str]) -> dict:
    """Verify one assembled pool against the admission record. Digest-only.

    Asserts: every line JSON-decodes (the assembly is sha-verified — any
    undecodable line is corruption, never tolerated); every row carries a
    non-empty conv_id; no duplicate conv_ids; row count == admitted count;
    conv_id SET equality with the admitted set. Returns a digest dict
    (rows / sha256 / bytes). Raises RuntimeError naming the variant +
    counts (+ up to 5 offending conv_ids — ids are digests, never text).
    """
    n_rows = 0
    cids: set[str] = set()
    dups = 0
    # Text-mode line iteration, never read_text().splitlines() (the #825
    # U+2028/NEL JSONL-shredding gotcha).
    with pool_path.open(encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"staged pool {pool_path.name} line {i} undecodable — corrupt "
                    f"assembly despite per-shard sha verification: {exc}"
                ) from exc
            n_rows += 1
            cid = str(row.get("conv_id") or "")
            if not cid:
                raise RuntimeError(
                    f"staged pool {pool_path.name} line {i} carries no conv_id "
                    f"(scaffold_id={row.get('scaffold_id')!r})"
                )
            if cid in cids:
                dups += 1
            cids.add(cid)
    if dups:
        raise RuntimeError(
            f"variant {variant}: {dups} duplicate conv_id(s) in staged pool {pool_path.name}"
        )
    n_admitted = len(admitted_ids)
    if n_rows != n_admitted:
        raise RuntimeError(
            f"variant {variant}: staged row count {n_rows} != kept.json admitted "
            f"count {n_admitted} ({pool_path.name})"
        )
    admitted_set = set(admitted_ids)
    missing = sorted(admitted_set - cids)
    extra = sorted(cids - admitted_set)
    if missing or extra:
        raise RuntimeError(
            f"variant {variant}: conv_id set mismatch vs kept.json admitted set — "
            f"missing={len(missing)} (e.g. {missing[:5]}) "
            f"extra={len(extra)} (e.g. {extra[:5]})"
        )
    digest = {
        "rows": n_rows,
        "bytes": pool_path.stat().st_size,
        "sha256": hashlib.sha256(pool_path.read_bytes()).hexdigest(),
    }
    _log(
        f"verified {variant}: rows={n_rows} == admitted={n_admitted}, conv_id set "
        f"equality OK, sha256={digest['sha256'][:12]}... ({digest['bytes']} B)"
    )
    return digest


def run(args: argparse.Namespace) -> int:
    dest_root = Path(args.dest_root).resolve()
    variants = [v.strip() for v in str(args.variants).split(",") if v.strip()]
    if not variants:
        print("ERROR: --variants resolved to an empty list", file=sys.stderr)
        return 1
    dest_root.mkdir(parents=True, exist_ok=True)
    _log(f"start: dest_root={dest_root} variants={variants}")

    kept_path, kept = stage_kept_json(dest_root)
    per_variant: dict[str, dict] = {}
    for v in variants:
        admitted = admitted_ids_for(kept, v)
        pool_path = ba._stage_sharded_jsonl(
            dest_root / v, f"{ba.SCAFFOLDS_PREFIX}/{v}", f"scaffolds_{v}"
        )
        per_variant[v] = verify_staged_variant(pool_path, v, admitted)
        per_variant[v]["path"] = str(pool_path)

    digest = {
        "phase": "stage_scaffolds",
        "variants": per_variant,
        "kept_json_sha256": hashlib.sha256(kept_path.read_bytes()).hexdigest(),
        "hf_prefix": ba.SCAFFOLDS_PREFIX,
        "utc": datetime.now(tz=timezone.utc).isoformat(),
    }
    digest_path = dest_root / "staging_digest.json"
    tmp = digest_path.with_suffix(".json.tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(digest, f, indent=2, sort_keys=True)
    tmp.replace(digest_path)
    _log(f"digest -> {digest_path}")
    print("[phase=done]", flush=True)
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--variants",
        default=",".join(pa.DEFAULT_VARIANTS),
        help="comma-separated variant subset (default: the 5-cell plan panel)",
    )
    p.add_argument(
        "--dest-root",
        default="data/issue_2054/scaffolds/",
        help="local root; per-variant pools land at <root>/<v>/scaffolds_<v>.jsonl",
    )
    args = p.parse_args()
    try:
        return run(args)
    except Exception as exc:  # fail-loud with a named error; exit 1
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
