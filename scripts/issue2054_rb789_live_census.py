"""Build the production merge census by DERIVATION from the staged cell set.

WHY THIS EXISTS
---------------
`_grid_assert` (`scripts/issue2054_rb789.py:1636`) gates the live grid against
`census["class_pair_counts"]`, `census["chat_anchor_per_arm"]` and
`census["strata"]["above_floor"]`. With `--census production` those come from
`PRODUCTION_CLASS_PAIR_COUNTS` / `CHAT_ANCHOR_PER_ARM` / `PRODUCTION_STRATA`, which
task #2054 marker v243 established are the COMMITTED ARTIFACT's `_axis_diff_class`
counts over a constructed 48-cell set -- NOT the `_pair_class` enumeration over the 56
staged cells that this round actually runs. Measured divergence (marker v262):
boundary 40 vs 56, prose 120 vs 96, twobytwo 224 vs 208, chat-anchor 80 vs 32,
N2 above-floor 160 vs 144. The merge would abort on all of them.

The round already exposes the interface for this: `_load_census` (`:1601`) treats any
`--census` value other than the literal "production" as a path to a census JSON. So no
reviewed code changes -- we supply a correctly-derived census, which is also exactly
v243's recommendation ("derive expectations from `_pair_class` over the staged cells").

The derivation recipe MIRRORS the reviewed smoke-fixture block at `:2804-2850` rather
than inventing a second convention: class counts from `_enumerate_ordered_pairs` +
`_pair_class`, anchors via `_is_chat_anchor_key`, strata as an above/below split at
`AMBIENT_FLOOR`.

THE LOAD-BEARING CONSTRAINT
---------------------------
Expected counts are derived from the CELL SET, never from the units that happen to
exist. Deriving expectations from existing units would make `_grid_assert` a tautology:
a missing unit would silently shrink its own expectation, defeating the assert's stated
purpose ("the merge NEVER writes a partial lattice").

The one quantity that cannot come from cell names alone is the strata split, which needs
per-pair intersection sizes. We read those from the twobytwo units' `n_intersection_full`
-- admissible ONLY because `class_pair_counts["twobytwo"]`, derived independently from
the cell set, has already proven that pair set complete; the strata split is then a
partition of a verified set, not an independent expectation. This script FAILS LOUD if
any enumerated twobytwo pair lacks a unit, so the ordering of those two guarantees is
enforced here and not merely assumed.

Deliberately NOT downloading the activation .npz files to compute intersections
directly: 56 cells x (v_C, v_A, v_P) at d=3584 float32 is ~24 GB, far past the 10 GB
pod-routing threshold, and the merge itself never loads them.

Usage:
    uv run python build_live_census.py --units-dir <ladder/> --out <census.json>
        [--activations-prefix issue2054_lattice/activations]

Read-only with respect to the Hub (a listing, no downloads) and to the round's code.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

WT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WT / "scripts"))
sys.path.insert(0, str(WT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue2054_ladder as ladder  # noqa: E402
import issue2054_rb789 as R  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

DATA_REPO = "superkaiba1/explore-persona-space-data"


def staged_cell_keys(prefix: str) -> list[str]:
    """Cell keys from the activations prefix LISTING -- filenames encode the key, so
    this needs no download. Cells live at `<prefix>/<variant>/<cell_key>.npz`."""
    api = HfApi()
    keys = []
    for entry in api.list_repo_tree(
        repo_id=DATA_REPO, repo_type="dataset", path_in_repo=prefix, recursive=True
    ):
        if type(entry).__name__ == "RepoFolder":
            continue
        name = Path(entry.path).name
        if name.endswith(".npz"):
            keys.append(name[: -len(".npz")])
    return sorted(set(keys))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--units-dir", required=True, help="dir of full-n rb789_*.json units")
    ap.add_argument("--out", required=True)
    ap.add_argument("--activations-prefix", default="issue2054_lattice/activations")
    args = ap.parse_args()

    keys = staged_cell_keys(args.activations_prefix)
    print(f"staged cells (from listing): {len(keys)}")
    if len(keys) < 2:
        raise SystemExit(f"only {len(keys)} staged cell(s) under {args.activations_prefix}")

    # 5-tuples: `_pair_class` reads s[:4] positionally and `_enumerate_ordered_pairs`
    # only compares cells for equality, so the trailing path is inert here.
    cells = [(*R._parse_cell_key(k), Path(f"{k}.npz")) for k in keys]

    class_pairs: dict[str, list[tuple[str, str]]] = {c: [] for c in R.PRODUCTION_CLASS_PAIR_COUNTS}
    for s, t in ladder._enumerate_ordered_pairs(
        cells, smoke=False, pair_classes=tuple(R.CLASS_TO_PAIR_CLASS.values())
    ):
        cls = R.PAIR_CLASS_TO_CLASS[ladder._pair_class(s, t)]
        class_pairs[cls].append((ladder._cell_key(*s[:4]), ladder._cell_key(*t[:4])))

    counts = {cls: len(ps) for cls, ps in class_pairs.items()}
    anchor_ctx = sorted(
        (s, t)
        for (s, t) in class_pairs["twobytwo"]
        if R._is_chat_anchor_key(s) or R._is_chat_anchor_key(t)
    )

    print(f"\n{'class':12s} {'derived':>8s} {'constant':>9s} {'delta':>7s}")
    for cls in sorted(counts):
        const = R.PRODUCTION_CLASS_PAIR_COUNTS[cls]
        print(f"{cls:12s} {counts[cls]:8d} {const:9d} {counts[cls] - const:+7d}")
    print(
        f"{'chat-anchor':12s} {len(anchor_ctx):8d} {R.CHAT_ANCHOR_PER_ARM:9d} "
        f"{len(anchor_ctx) - R.CHAT_ANCHOR_PER_ARM:+7d}"
    )

    # --- strata: partition of the ALREADY-VERIFIED twobytwo pair set -----------------
    n_by_pair: dict[tuple[str, str], int] = {}
    dupes = defaultdict(set)
    for p in sorted(Path(args.units_dir).glob("rb789_*.json")):
        if p.name.startswith("rb789_class"):
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("class") != "twobytwo" or d.get("arm") != "context":
            continue
        if d.get("level") is not None:
            continue  # matched-n units live in a separate dir; full-n only here
        key = (d["source"], d["target"])
        n = d.get("n_intersection_full")
        if n is None:
            raise SystemExit(f"unit {p.name} has no n_intersection_full — cannot derive strata")
        dupes[key].add(int(n))
        n_by_pair[key] = int(n)

    inconsistent = {k: v for k, v in dupes.items() if len(v) > 1}
    if inconsistent:
        raise SystemExit(f"same pair reports differing n_intersection_full: {inconsistent}")

    enumerated = set(class_pairs["twobytwo"])
    missing = sorted(enumerated - set(n_by_pair))
    if missing:
        raise SystemExit(
            f"FAIL-LOUD: {len(missing)} of {len(enumerated)} enumerated twobytwo/context pairs "
            f"have no full-n unit — the strata split may not be derived from an incomplete "
            f"set (first 3: {missing[:3]})"
        )

    floor = int(R.AMBIENT_FLOOR)
    above = sum(1 for k in enumerated if n_by_pair[k] >= floor)
    below = len(enumerated) - above
    print(
        f"\nstrata at ambient floor {floor}: above={above} below={below} "
        f"(constant {R.PRODUCTION_STRATA[0]}/{R.PRODUCTION_STRATA[1]})"
    )

    census = {
        "class_pair_counts": counts,
        "chat_anchor_per_arm": len(anchor_ctx),
        "chat_anchor_pairs_context": [list(x) for x in anchor_ctx],
        "strata": {"above_floor": above, "below_floor": below},
        "n1_levels": list(R.N1_LEVELS),
        "n1p_level": int(R.N1P_LEVEL),
        "ambient_floor": floor,
        "_provenance": {
            "derived_by": "data/issue_2054/rb789_launch/build_live_census.py",
            "classifier": "issue2054_ladder._pair_class via _enumerate_ordered_pairs",
            "cell_set": f"{len(keys)} staged cells listed under {args.activations_prefix}",
            "strata_source": (
                "n_intersection_full read off full-n twobytwo/context units, admissible "
                "because class_pair_counts['twobytwo'] is derived independently from the "
                "cell set and every enumerated pair was verified present"
            ),
            "supersedes_constants": {
                "PRODUCTION_CLASS_PAIR_COUNTS": dict(R.PRODUCTION_CLASS_PAIR_COUNTS),
                "CHAT_ANCHOR_PER_ARM": int(R.CHAT_ANCHOR_PER_ARM),
                "PRODUCTION_STRATA": list(R.PRODUCTION_STRATA),
            },
            "rationale_marker": "task #2054 epm:progress v243 (census derivation) + v262 (blocker)",
        },
    }
    Path(args.out).write_text(json.dumps(census, indent=2) + "\n", encoding="utf-8")
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
