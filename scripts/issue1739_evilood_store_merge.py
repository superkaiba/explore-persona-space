#!/usr/bin/env python3
"""#1739 evil-OOD r2v2: merge the four capture stores into ONE loadable store.

The r2v2 capture ran 4-way GPU-parallel and therefore wrote FOUR independent
capture stores (mhj / pair / tomgibbs_p0 / tomgibbs_p1), each numbering its
shards from 0. ``store_io.load_summaries`` takes a SINGLE store dir, resolves
each kind's shards with ``root.glob("{kind}_L{layer:02d}_shard*.npy")`` sorted
by shard index, and reads row metadata from ``row_index_shard*.jsonl`` in the
SAME sorted order — so four dirs cannot simply be copied together: their
shard indices collide, and a collision silently mis-pairs activation rows with
metadata rows.

This merges them into one store with CONTIGUOUS shard renumbering, keeping the
per-source shard order intact and renumbering the ``row_index`` sidecars by the
identical mapping, so array rows and meta rows stay aligned by construction.

VERIFICATION (not assumed): after the merge it re-loads the merged store via
the REAL consumer (``store_io.load_summaries``) at a couple of layers and
asserts the row count equals the sum of the sources' and that the loaded
metadata's context_id sequence equals the concatenated per-source sequence.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    """Repo root onto sys.path (script mode puts only scripts/ there — #823)."""
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1739_evilood_store_merge.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root derivation failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


_REPO_ROOT = _ensure_repo_root_on_syspath()

DEFAULT_SOURCES = ("mhj", "pair", "tomgibbs_p0", "tomgibbs_p1")


def _shard_indices(src: Path) -> list[int]:
    """Shard indices present in ``src``, from its row_index sidecars."""
    idx = sorted(
        int(p.name.removeprefix("row_index_shard").removesuffix(".jsonl"))
        for p in src.glob("row_index_shard*.jsonl")
    )
    if not idx:
        raise FileNotFoundError(f"no row_index_shard*.jsonl under {src}")
    if idx != list(range(len(idx))):
        raise RuntimeError(f"{src}: non-contiguous shard indices {idx[:5]}... — refusing to merge")
    return idx


def main() -> int:
    ap = argparse.ArgumentParser(description="#1739 evil-OOD capture-store merge")
    ap.add_argument("--store-root", default="data/issue_1739/evil_ood_full/store")
    ap.add_argument("--sources", nargs="+", default=list(DEFAULT_SOURCES))
    ap.add_argument("--dest", default="data/issue_1739/evil_ood_full/store_merged")
    ap.add_argument("--link", action="store_true", help="hardlink instead of copy")
    ap.add_argument("--verify-layers", nargs="+", type=int, default=[0, 14, 27])
    args = ap.parse_args()

    root = Path(args.store_root)
    dest = Path(args.dest)
    if dest.exists():
        raise SystemExit(f"dest already exists: {dest} (remove it to re-merge)")
    dest.mkdir(parents=True)

    place = (lambda s, d: __import__("os").link(s, d)) if args.link else shutil.copy2
    next_shard = 0
    expected_ctx: list[str] = []
    per_source: list[dict] = []
    for name in args.sources:
        src = root / name
        if not src.is_dir():
            raise SystemExit(f"source store missing: {src}")
        indices = _shard_indices(src)
        n_rows_src = 0
        for local in indices:
            new = next_shard + local
            for npy in sorted(src.glob(f"*_shard{local:02d}.npy")):
                stem = npy.name.removesuffix(f"_shard{local:02d}.npy")
                place(npy, dest / f"{stem}_shard{new:02d}.npy")
            index_src = src / f"row_index_shard{local:02d}.jsonl"
            # split("\n"), never .splitlines(): capture writes ensure_ascii=False
            # JSON, so a raw U+2028/U+2029/NEL inside a completion-derived field
            # would make splitlines() shred one record into two and silently
            # inflate the row count against the .npy shard (#825/#950).
            with index_src.open(encoding="utf-8") as fh:
                rows = [json.loads(line) for line in fh.read().split("\n") if line.strip()]
            for r in rows:
                r["merge_source"] = name
                expected_ctx.append(str(r.get("context_id")))
            (dest / f"row_index_shard{new:02d}.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in rows)
            )
            n_rows_src += len(rows)
        per_source.append({"source": name, "n_shards": len(indices), "n_rows": n_rows_src})
        print(f"[merge] {name}: {len(indices)} shards, {n_rows_src} rows -> shard{next_shard:02d}+")
        next_shard += len(indices)

    # --- verification through the REAL consumer -----------------------------
    from explore_persona_space.experiments.issue_1739 import store_io

    layers = tuple(args.verify_layers)
    arrays, meta = store_io.load_summaries(dest, ("context_end",), layers)
    n_expected = sum(s["n_rows"] for s in per_source)
    for (kind, layer), arr in arrays.items():
        if arr.shape[0] != n_expected:
            raise RuntimeError(f"{kind}/L{layer}: {arr.shape[0]} rows != expected {n_expected}")
    if len(meta) != n_expected:
        raise RuntimeError(f"meta rows {len(meta)} != expected {n_expected}")
    got_ctx = [str(r.get("context_id")) for r in meta]
    if got_ctx != expected_ctx:
        first = next(i for i, (a, b) in enumerate(zip(got_ctx, expected_ctx)) if a != b)
        raise RuntimeError(
            f"merged row order diverges from the concatenated source order at row {first}"
        )
    manifest = {
        "dest": str(dest),
        "sources": per_source,
        "n_shards_total": next_shard,
        "n_rows_total": n_expected,
        "verified_layers": list(layers),
        "verification": (
            "re-loaded through store_io.load_summaries; row counts match the source sum and the "
            "loaded context_id sequence equals the concatenated per-source sequence"
        ),
    }
    (dest / "_merge_manifest.json").write_text(json.dumps(manifest, indent=1))
    print(json.dumps(manifest, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
