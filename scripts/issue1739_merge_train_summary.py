#!/usr/bin/env python3
"""Merge arm17/18 train rows into a behavior's train summary, at a NEW path.

Why this exists: the transfer scorers pick each arm's evaluation layer from
that arm's TRAIN-side rows in the committed summary
(``modal_frozen_layers`` -> mode of ``arms.frozen_layer_idx(rho_per_layer)``).
``arm17_oracle_mlp`` / ``arm18_oracle_krr`` have ZERO rows there, so a leg that
requests them fail-louds ("no frozen layer") and takes every other requested
arm down with it. The rows DO exist -- the ``new_arm_round`` oracle legs wrote
them to ``merged_transfer.jsonl`` with ``eval_rung == "train"``, carrying the
28-length ``rho_per_layer`` the selection needs -- they just live in a
different artifact than the one the scorer resolves.

This writes a MERGED summary (original ``arm_rows`` + those rows) to a NEW
path for ``--train-summary``. It never mutates the committed summary: that is
a published artifact other rounds resolve against, and regenerating it under
its own path is the in-place-invalidation hazard.

Coverage caveat, reported per behavior: hallucination carries context_end rows
only, so arms 17/18 stay unproducible for hallucination/prefix_end.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ORACLE_ARMS = ("arm17_oracle_mlp", "arm18_oracle_krr")
REQUIRED = ("variant", "regime", "u_rung_label", "rho_per_layer")


def load_oracle_train_rows(merged_jsonl: Path, behavior: str) -> list[dict]:
    """arm17/18 rows for one behavior with eval_rung == 'train'."""
    out: list[dict] = []
    with merged_jsonl.open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            obj = json.loads(line)
            for r in obj.get("rows") or [obj]:
                if (
                    r.get("eval_rung") == "train"
                    and r.get("arm") in ORACLE_ARMS
                    and r.get("behavior") == behavior
                    and r.get("f_u") is None
                    and all(r.get(k) is not None for k in REQUIRED)
                ):
                    out.append(r)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--behavior", required=True)
    ap.add_argument(
        "--summary", type=Path, required=True, help="committed train summary (READ-ONLY)"
    )
    ap.add_argument("--merged-jsonl", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="merged summary destination (NEW path)")
    args = ap.parse_args()

    if args.out.resolve() == args.summary.resolve():
        raise SystemExit("refusing to overwrite the committed summary; --out must be a NEW path")

    payload = json.loads(args.summary.read_text())
    base_rows = payload.get("arm_rows", [])
    add_rows = load_oracle_train_rows(args.merged_jsonl, args.behavior)
    if not add_rows:
        raise SystemExit(f"no arm17/18 train rows found for behavior={args.behavior}")

    # Report the coverage the merge actually buys, per (arm, variant).
    by_av: dict[tuple[str, str], list[int]] = defaultdict(list)
    from explore_persona_space.experiments.issue_1739 import arms as A

    for r in add_rows:
        by_av[(r["arm"], r["variant"])].append(A.frozen_layer_idx(r["rho_per_layer"]))

    print(f"behavior={args.behavior}: base arm_rows={len(base_rows)} + oracle rows={len(add_rows)}")
    for (arm, variant), idxs in sorted(by_av.items()):
        modal = Counter(sorted(idxs)).most_common(1)[0][0]
        print(f"   {arm:<20} {variant:<12} rows={len(idxs):<4} modal_frozen_layer={modal}")
    for arm in ORACLE_ARMS:
        for variant in ("context_end", "prefix_end"):
            if (arm, variant) not in by_av:
                print(f"   {arm:<20} {variant:<12} NO ROWS -> stays unproducible for this cell")

    payload["arm_rows"] = base_rows + add_rows
    payload.setdefault("meta", {})["armfill_merged_oracle_rows"] = {
        "source": str(args.merged_jsonl),
        "arms": list(ORACLE_ARMS),
        "n_rows_added": len(add_rows),
        "note": "train_in_split rows merged so committed-frozen selection can resolve 17/18",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(args.out.suffix + ".tmp")
    tmp.write_text(json.dumps(payload))
    tmp.replace(args.out)
    print(f"wrote {args.out} ({args.out.stat().st_size} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
