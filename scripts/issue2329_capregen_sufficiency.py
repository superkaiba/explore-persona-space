#!/usr/bin/env python3
"""Task #2329: is the RAISED cap (4096) actually sufficient for the regenerated rows?

Neither existing cap-hit report answers this, by design:

* ``cap_hit_report_anchors_preregen.json`` is 2048-attributed (it measured the
  breach that motivated the remedy).
* ``cap_hit_report_anchors_postregen.json`` is BASE-cap (2048) attributed on
  purpose -- that attribution is what restores the ``realized_row_caps``
  ``[2048, 4096]`` half-done distinguisher and keeps the metric comparable
  across a partially-regenerated store. Under it, a row regenerated at 4096
  that produced 2,500 tokens still counts as cap-hit, so the aggregate barely
  moves and CANNOT be read as remedy effectiveness.

The decision-relevant quantity is different: among rows actually regenerated at
the raised cap, what fraction hit *that* cap? If a material share still hit
4096, the generation tail extends past it, the >= 2x rule points at 8192, and
the asymmetric-truncation bias survives its own remedy in a second form.

Reports the aggregate and -- because value-side ASYMMETRY was the original
defect, not the overall rate -- the per-(cell, value) breakdown. A cap that
is sufficient on average but truncates one side of a within-cell contrast is
still a measurement-validity failure.

Two scopes (--scope, default anchors -- the original invocation is unchanged):

* ``anchors``: ``anchors_<batch>_w*.jsonl`` under --anchors-dir; rows keyed
  (cell, value_id). Caps default to the anchors campaign's 2048 -> 4096.
* ``grid``: ``shard_*.jsonl`` under --rollouts-dir (the LADDER grid store,
  ``<out-root>/grid``); rows carry slot/arm/value_a instead of value_id, so
  the unit key is ``cell|slot|arm`` (the plan §7 G5 breach grain) and the
  value side is ``value_a``. The q35 ladder audit runs 4096 -> 8192:

    uv run python scripts/issue2329_capregen_sufficiency.py --scope grid \
        --rollouts-dir /workspace/issue2329_out/ladder/grid \
        --base-cap 4096 --raised-cap 8192 \
        --out eval_results/issue_2329/q35_ladder_decay/cap_hit/capregen_sufficiency_grid.json

The value key FIELD is resolved per row (value_id first, value_a fallback)
and the realized field counts are DISCLOSED in the output
(``value_key_fields``) so a schema drift can never silently collapse the
per-value breakdown onto ``<none>``.

Counts only. No completion text enters the output.

    uv run python scripts/issue2329_capregen_sufficiency.py \
        --anchors-dir /workspace/issue2329_out/anchors \
        --out eval_results/issue_2329/cap_hit/capregen_sufficiency_anchors.json
"""

from __future__ import annotations

import argparse
import json
import statistics as st
from collections import defaultdict
from pathlib import Path
from typing import Any

BASE_CAP = 2048
RAISED_CAP = 4096


def _pct(num: int, den: int) -> float:
    if den <= 0:
        raise ValueError(f"refusing to compute a percentage with denominator {den}")
    return 100.0 * num / den


def load_rows_glob(shard_dir: Path, glob_pat: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Load every ``glob_pat`` JSONL shard under ``shard_dir``. Fail loud on empty."""
    shards = sorted(shard_dir.glob(glob_pat))
    if not shards:
        raise FileNotFoundError(
            f"no {glob_pat} shards under {shard_dir} -- "
            "nothing to measure (an empty selection is never a 0% result)"
        )
    rows: list[dict[str, Any]] = []
    for s in shards:
        with s.open() as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"{len(shards)} shard(s) under {shard_dir} contained zero rows")
    return rows, [s.name for s in shards]


def load_rows(anchors_dir: Path, batch: str) -> tuple[list[dict[str, Any]], list[str]]:
    """Anchors-scope loader (original surface, kept byte-compatible)."""
    return load_rows_glob(anchors_dir, f"anchors_{batch}_w*.jsonl")


def _row_keys(r: dict[str, Any]) -> tuple[str, str, str]:
    """(cell_key, value_key, value_field) for one row.

    Anchors rows: cell + value_id -> the original (cell, value_id) keying,
    byte-identical output. LADDER GRID rows: cell(=direction) + slot + arm +
    value_a (no value_id) -> unit key ``cell|slot|arm`` (the plan §7 G5
    breach grain, matching ``breaching_units``) with value side ``value_a``.
    The realized value FIELD is returned so the caller can disclose counts --
    a silent fallback-to-<none> collapse is the schema-drift failure mode this
    guards against."""
    cell = str(r.get("cell", "<none>"))
    if "slot" in r and "arm" in r:
        cell = f"{cell}|{r['slot']}|{r['arm']}"
    if r.get("value_id") is not None:
        return cell, str(r["value_id"]), "value_id"
    if r.get("value_a") is not None:
        return cell, str(r["value_a"]), "value_a"
    return cell, "<none>", "<none>"


def summarize(rows: list[dict[str, Any]], raised_cap: int, base_cap: int) -> dict[str, Any]:
    # ``n_completion_tokens`` is required on every row -- without it nothing is
    # measurable. ``max_new_tokens`` is ABSENT on legacy pre-regen rows: the
    # original anchors run wrote no per-row cap, and capregen stamps one only on
    # rows it regenerates. Absence therefore means "generated under the original
    # global cap", i.e. the base cap -- the same inheritance run.py:3276 applies.
    # The count is DISCLOSED (never silently folded in) so a capregen bug that
    # failed to stamp the per-row cap cannot masquerade as "not regenerated".
    for r in rows:
        if "n_completion_tokens" not in r:
            raise KeyError("row missing required field 'n_completion_tokens'; refusing to guess it")

    def _row_cap(r: dict[str, Any]) -> int:
        v = r.get("max_new_tokens")
        return base_cap if v is None else int(v)

    n_legacy = sum(1 for r in rows if r.get("max_new_tokens") is None)
    regen = [r for r in rows if _row_cap(r) == raised_cap]
    kept = [r for r in rows if _row_cap(r) == base_cap]
    other = [r for r in rows if _row_cap(r) not in (raised_cap, base_cap)]
    if other:
        caps = sorted({_row_cap(r) for r in other})
        raise ValueError(
            f"{len(other)} row(s) carry an unexpected per-row cap {caps} -- "
            "the store should hold only base and raised caps"
        )
    if not regen:
        raise ValueError(
            f"zero rows carry the raised cap {raised_cap} -- either capregen has not merged "
            "yet or the per-row cap was never written; refusing to report 0% sufficiency"
        )

    n = sorted(int(r["n_completion_tokens"]) for r in regen)
    hit_raised = [x for x in n if x >= raised_cap]
    over_base = [x for x in n if x >= base_cap]

    out: dict[str, Any] = {
        "raised_cap": raised_cap,
        "base_cap": base_cap,
        "n_rows_total": len(rows),
        "n_rows_regenerated": len(regen),
        "n_rows_untouched_at_base": len(kept),
        "n_rows_legacy_inherited_cap": n_legacy,
        "regen_hit_raised_cap_rows": len(hit_raised),
        "regen_hit_raised_cap_pct": _pct(len(hit_raised), len(regen)),
        "regen_over_base_cap_rows": len(over_base),
        "regen_over_base_cap_pct": _pct(len(over_base), len(regen)),
        "regen_tokens_max": n[-1],
        "regen_tokens_p99": n[int(0.99 * (len(n) - 1))],
        "regen_tokens_p90": n[int(0.90 * (len(n) - 1))],
        "regen_tokens_median": int(st.median(n)),
    }

    # Value-side asymmetry: the original defect was one side of a within-cell
    # contrast being truncated, not the overall rate. Keying is scope-aware
    # (_row_keys): anchors rows -> (cell, value_id) as before; ladder grid
    # rows -> (cell|slot|arm, value_a). Realized value-field counts are
    # DISCLOSED so a schema drift cannot silently collapse the breakdown.
    by_cell_value: dict[tuple[str, str], list[int]] = defaultdict(list)
    value_key_fields: dict[str, int] = defaultdict(int)
    for r in regen:
        cell_key, value_key, value_field = _row_keys(r)
        value_key_fields[value_field] += 1
        by_cell_value[(cell_key, value_key)].append(int(r["n_completion_tokens"]))
    out["value_key_fields"] = dict(sorted(value_key_fields.items()))

    per_cell: list[dict[str, Any]] = []
    for (cell, value_id), toks in sorted(by_cell_value.items()):
        hits = sum(1 for x in toks if x >= raised_cap)
        per_cell.append(
            {
                "cell": cell,
                "value_id": value_id,
                "n": len(toks),
                "hit_raised_cap_rows": hits,
                "hit_raised_cap_pct": _pct(hits, len(toks)),
                "tokens_max": max(toks),
            }
        )
    out["per_cell_value"] = per_cell

    # Within-cell asymmetry across value sides: max minus min hit-rate per cell.
    by_cell: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in per_cell:
        by_cell[row["cell"]].append(row)
    asym: list[dict[str, Any]] = []
    for cell, entries in sorted(by_cell.items()):
        if len(entries) < 2:
            continue
        rates = [e["hit_raised_cap_pct"] for e in entries]
        asym.append(
            {
                "cell": cell,
                "n_value_sides": len(entries),
                "min_hit_pct": min(rates),
                "max_hit_pct": max(rates),
                "spread_pct_points": max(rates) - min(rates),
            }
        )
    asym.sort(key=lambda d: d["spread_pct_points"], reverse=True)
    out["within_cell_asymmetry"] = asym
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--scope",
        choices=("anchors", "grid"),
        default="anchors",
        help="anchors (default; the original 2048->4096 anchors audit, unchanged) | "
        "grid (the LADDER grid store -- shard_*.jsonl under --rollouts-dir; the q35 "
        "ladder audit runs --base-cap 4096 --raised-cap 8192)",
    )
    ap.add_argument("--anchors-dir", type=Path, default=None, help="anchors scope: shard dir")
    ap.add_argument(
        "--rollouts-dir",
        type=Path,
        default=None,
        help="grid scope: the ladder grid shard dir (<out-root>/grid)",
    )
    ap.add_argument("--batch", default="gate", choices=("gate", "rest"))
    ap.add_argument("--raised-cap", type=int, default=RAISED_CAP)
    ap.add_argument("--base-cap", type=int, default=BASE_CAP)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    if args.scope == "anchors":
        if args.anchors_dir is None:
            ap.error("--anchors-dir is required for --scope anchors")
        shard_dir, glob_pat = args.anchors_dir, f"anchors_{args.batch}_w*.jsonl"
    else:
        if args.rollouts_dir is None:
            ap.error("--rollouts-dir is required for --scope grid")
        shard_dir, glob_pat = args.rollouts_dir, "shard_*.jsonl"

    rows, shard_names = load_rows_glob(shard_dir, glob_pat)
    summary = summarize(rows, args.raised_cap, args.base_cap)
    summary["scope"] = args.scope
    summary["batch"] = args.batch if args.scope == "anchors" else None
    summary["shards_present"] = shard_names
    summary["n_shards_present"] = len(shard_names)
    # PARTIAL detection must NOT key on shard-file count: all shard files exist
    # from the ORIGINAL run, so presence proves nothing about whether capregen
    # has merged into them. The observable signal is per-shard presence of at
    # least one raised-cap row. Caveat, stated because it matters: a shard
    # whose slice holds no breaching cells legitimately has zero regen rows and
    # is indistinguishable from an unmerged one -- this errs toward reporting
    # INCOMPLETE, which is the safe direction for a completeness claim. The
    # authoritative signal is the driver's own postregen report `partial` field.
    with_regen = sorted(
        p.name
        for p in shard_dir.glob(glob_pat)
        if any(
            json.loads(ln).get("max_new_tokens") == args.raised_cap for ln in p.open() if ln.strip()
        )
    )
    summary["shards_with_regen_rows"] = with_regen
    summary["n_shards_with_regen_rows"] = len(with_regen)
    summary["partial"] = len(with_regen) < len(shard_names)

    print(
        f"  scope={args.scope}  batch={summary['batch']}  shards={len(shard_names)}  "
        f"partial={summary['partial']}"
    )
    print(
        f"  rows={summary['n_rows_total']}  regen@{args.raised_cap}="
        f"{summary['n_rows_regenerated']}  untouched@{args.base_cap}="
        f"{summary['n_rows_untouched_at_base']}"
    )
    print(
        f"  IS THE RAISED CAP SUFFICIENT?  regen rows hitting {args.raised_cap}: "
        f"{summary['regen_hit_raised_cap_rows']}/{summary['n_rows_regenerated']} = "
        f"{summary['regen_hit_raised_cap_pct']:.4f}%"
    )
    print(
        f"  regen tokens: max={summary['regen_tokens_max']} "
        f"p99={summary['regen_tokens_p99']} p90={summary['regen_tokens_p90']} "
        f"median={summary['regen_tokens_median']}"
    )
    print(
        f"  (for context) regen rows exceeding the OLD {args.base_cap} cap: "
        f"{summary['regen_over_base_cap_rows']}/{summary['n_rows_regenerated']} = "
        f"{summary['regen_over_base_cap_pct']:.2f}%"
    )
    worst = summary["within_cell_asymmetry"][:5]
    if worst:
        print("  worst within-cell value-side spread (pct points):")
        for w in worst:
            print(
                f"    {w['cell']:24s} sides={w['n_value_sides']} "
                f"min={w['min_hit_pct']:.1f}% max={w['max_hit_pct']:.1f}% "
                f"spread={w['spread_pct_points']:.1f}"
            )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
        print(f"  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
