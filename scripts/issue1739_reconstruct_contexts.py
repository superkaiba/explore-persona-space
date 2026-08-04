"""Reconstruct staged-context JSONLs from labeling rollout payloads (leg 2).

The leg-1 GCE instance staged the per-rung context JSONLs
(``{behavior}_{split}_{rung}.contexts.jsonl``) locally and was torn down
before they were persisted; re-running ``stage_corpus`` on a fresh machine
risks a re-stream mismatch against the already-generated rollouts. Every
labeling rollout payload embeds its full context row (context_id, split,
rung, group_key, prefix_text, query), so the contexts are reconstructed
EXACTLY (id-aligned by construction) by deduping rollouts per context_id.

Field note (stated deviation, features-arm only): ``prefix_text`` here is
the RENDERED prompt prefix (chat template included), where the original
staged rows carried raw source text for some rungs. The features arm
(hashed token-freq + surface stats) first runs on leg 2, so this IS the
instrument, uniformly across all contexts/behaviors; the shared template
addend is constant across contexts. CONTENT HYGIENE: logs carry counts +
ids only.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path


def reconstruct(rollout_dir: Path, out_dir: Path, behavior: str) -> dict[str, int]:
    """Write per-(split, rung) context JSONLs; returns {filename: n_rows}."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seen: set[str] = set()
    rows_by_file: dict[str, list[dict]] = {}
    n_files = 0
    for p in sorted(rollout_dir.glob("*.json")):
        if p.name.startswith("_"):
            continue
        d = json.loads(p.read_text())
        n_files += 1
        cid = d["context_id"]
        if cid in seen:
            continue
        seen.add(cid)
        row = {
            "context_id": cid,
            "behavior": d["behavior"],
            "split": d["split"],
            "rung": d["rung"],
            "group_key": d.get("group_key"),
            "prefix_text": d.get("prefix_text") or "",
            "query": d.get("query") or "",
            "reconstructed_from": "labeling_rollouts",
        }
        fname = f"{behavior}_{d['split']}_{d['rung']}.contexts.jsonl"
        rows_by_file.setdefault(fname, []).append(row)
    if not rows_by_file:
        raise SystemExit(f"[reconstruct] zero contexts under {rollout_dir}")
    counts: dict[str, int] = {}
    for fname, rows in sorted(rows_by_file.items()):
        rows.sort(key=lambda r: r["context_id"])
        path = out_dir / fname
        tmp = path.with_name(path.name + ".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        tmp.replace(path)
        counts[fname] = len(rows)
    total = sum(counts.values())
    per_rung = Counter({k: v for k, v in counts.items()})
    print(
        f"[reconstruct] {behavior}: {n_files} rollout files -> {total} contexts "
        f"across {len(counts)} rung files: {dict(per_rung)}",
        flush=True,
    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--behavior", required=True)
    parser.add_argument("--rollout-dir", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    reconstruct(Path(args.rollout_dir), Path(args.out_dir), args.behavior)
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
