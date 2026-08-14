"""Issue #2094 reversed-direction (persona -> bare) F_beh reduce + ladder figure.

Consumes the rev-round judge waves under ``eval_results/issue_2094/judge_rev/``
(coherence.grid + fp-persona.grid + fp-bare.grid — 300 cells: ce slot x 30
layer variants x replace x Type A x {steered, null} x 5 ``mqrev--`` pairs) and
the BANKED per-pair floors/ceilings (``rev_floor_ceiling.json``, computed by
``issue2094_reverse_direction.py::compute_rev_floor_ceiling`` from the parent's
anchor judge draws — zero new anchor calls).

Arithmetic is the parent's pre-registered reduce, REUSED not re-derived:
per-cell Δ_patched = (judge_B - judge_A)/100 via ``fmetrics.delta_contrast``
(side a = fp-persona, side b = fp-bare), F = (Δ_patched - Δ_floor)/(Δ_ceiling -
Δ_floor) via ``fmetrics.f_beh`` (batched), coherence gate score > 60
(``issue2094_judge.COHERENCE_THRESHOLD``), coherent-only cells (incoherent
cells marked + excluded, never dropped — the parent ``assemble_shard_rows``
convention).

Writes ``rev_fbeh_summary.json`` (per-cell rows + per-(layer_variant, arm)
aggregates) into the judge_rev work root and the ladder figure (steered vs
norm-matched shuffled-donor null) to ``figures/issue_2094/``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE any heavy import

import matplotlib.pyplot as plt  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_role,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402
from explore_persona_space.experiments.issue2094 import fmetrics as FM  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

COHERENCE_THRESHOLD = 60.0  # parent gate (issue2094_judge.COHERENCE_THRESHOLD)
WELL_SEPARATED_MIN = 0.5  # parent's registered low-separation bar (|denominator| >= 0.5)
N_LAYERS = 28
LADDER = tuple(f"L{i}" for i in range(N_LAYERS)) + ("joint_mid", "joint_all")
ARMS = ("steered", "null")


def _read_jsonl(path: Path) -> list[dict]:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    assert rows, f"no rows in {path}"
    return rows


def _cell_key(row: dict) -> tuple[str, str]:
    return (row["block_key"], row["pair_id"])


def _index_unique(rows: list[dict], what: str) -> dict[tuple[str, str], dict]:
    out: dict[tuple[str, str], dict] = {}
    for r in rows:
        k = _cell_key(r)
        assert k not in out, f"duplicate {what} row for cell {k}"
        out[k] = r
    return out


def reduce_cells(
    coh_rows: list[dict],
    side_a_rows: list[dict],
    side_b_rows: list[dict],
    fc_pairs: list[dict],
    donor_by_cell: dict[tuple[str, str], str | None] | None = None,
) -> tuple[list[dict], dict]:
    """Per-cell F_beh rows + per-(layer_variant, arm) aggregates (pure).

    ``side_a_rows`` / ``side_b_rows`` are the fp-persona / fp-bare wave score
    rows (side 'a' / 'b' of every ``mqrev--`` pair). Missing judge scores and
    incoherent cells are marked + excluded from aggregates, never dropped.
    """
    fc = {e["pair_id"]: e for e in fc_pairs}
    assert len(fc) == len(fc_pairs) == 5, sorted(fc)
    coh = _index_unique(coh_rows, "coherence")
    sa = _index_unique(side_a_rows, "side-a (fp-persona)")
    sb = _index_unique(side_b_rows, "side-b (fp-bare)")
    assert set(coh) == set(sa) == set(sb), "cell sets differ across the three waves"
    for r in side_a_rows:
        assert r["side"] == "a" and r["rubric_id"] == "fp-persona", r["item_id"]
    for r in side_b_rows:
        assert r["side"] == "b" and r["rubric_id"] == "fp-bare", r["item_id"]

    cells: list[dict] = []
    dp_idx: list[int] = []
    dp, df, dc = [], [], []
    for k in sorted(coh):
        c = coh[k]
        pair_id = c["pair_id"]
        assert pair_id.startswith("mqrev--") and pair_id in fc, pair_id
        entry = fc[pair_id]
        coh_score = c.get("score")
        coherent = coh_score is not None and float(coh_score) > COHERENCE_THRESHOLD
        a_score, b_score = sa[k].get("score"), sb[k].get("score")
        cell = {
            "block_key": c["block_key"],
            "layer_variant": c["layer_variant"],
            "arm": c["arm"],
            "pair_id": pair_id,
            # the wave score rows don't carry donor ids (_grid_source drops
            # them); enriched from the rollout shards when --rollouts-dir given.
            "donor_pair_id": (donor_by_cell or {}).get(k),
            "coherence_score": coh_score,
            "coherent": coherent,
            "judge_a_fp_persona": a_score,
            "judge_b_fp_bare": b_score,
            "delta_floor": entry["floor"]["delta_mean"],
            "delta_ceiling": entry["ceiling"]["delta_mean"],
            "denominator": entry["denominator"],
            "well_separated": abs(entry["denominator"]) >= WELL_SEPARATED_MIN,
        }
        if a_score is None or b_score is None:
            cell.update({"f_beh": None, "missing": "judge_dropped"})
        else:
            dp_idx.append(len(cells))
            dp.append(
                float(
                    FM.delta_contrast(
                        torch.tensor([float(b_score)]), torch.tensor([float(a_score)])
                    )[0]
                )
            )
            df.append(entry["floor"]["delta_mean"])
            dc.append(entry["ceiling"]["delta_mean"])
        cells.append(cell)

    if dp_idx:
        fb = FM.f_beh(torch.tensor(dp), torch.tensor(df), torch.tensor(dc))
        for j, i in enumerate(dp_idx):
            val = float(fb.f_beh[j])
            rec = {
                "delta_patched": dp[j],
                "contrast": float(fb.contrast[j]),
                "degenerate_denominator": bool(fb.degenerate_denominator[j]),
                "negative_denominator": bool(fb.negative_denominator[j]),
            }
            if not cells[i]["coherent"]:
                # marked, never dropped; excluded from aggregates (parent convention).
                rec.update({"f_beh": None, "excluded_incoherent_raw": val})
            else:
                rec["f_beh"] = val
            cells[i].update(rec)

    by_va: dict[tuple[str, str], list[float]] = defaultdict(list)
    n_total: dict[tuple[str, str], int] = defaultdict(int)
    n_coherent: dict[tuple[str, str], int] = defaultdict(int)
    for cell in cells:
        k = (cell["layer_variant"], cell["arm"])
        n_total[k] += 1
        n_coherent[k] += int(bool(cell["coherent"]))
        if cell.get("f_beh") is not None and not cell.get("degenerate_denominator"):
            by_va[k].append(cell["f_beh"])
    aggregates = {
        f"{variant}|{arm}": {
            "layer_variant": variant,
            "arm": arm,
            "f_beh_mean": (sum(v) / len(v)) if (v := by_va[(variant, arm)]) else None,
            "n_included": len(by_va[(variant, arm)]),
            "n_coherent": n_coherent[(variant, arm)],
            "n_total": n_total[(variant, arm)],
            "per_pair_f_beh": sorted(
                (c["pair_id"], c["f_beh"])
                for c in cells
                if c["layer_variant"] == variant and c["arm"] == arm
            ),
        }
        for variant in LADDER
        for arm in ARMS
    }
    return cells, aggregates


def render_figure(aggregates: dict, cells: list[dict], slug: str) -> dict[str, Path]:
    """Ladder figure: mean F_beh per layer variant, steered vs shuffled-donor null."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    x = list(range(len(LADDER)))
    colors = {"steered": paper_palette_role("primary"), "null": paper_palette_role("control")}
    for arm in ARMS:
        for c in cells:  # low-level per-pair points behind the means
            if c.get("f_beh") is None:
                continue
            ax.scatter(
                LADDER.index(c["layer_variant"]),
                c["f_beh"],
                s=6,
                alpha=0.25,
                color=colors[c["arm"]],
                linewidths=0,
                zorder=1,
            )
        y = [aggregates[f"{v}|{arm}"]["f_beh_mean"] for v in LADDER]
        label = "steered (persona → bare)" if arm == "steered" else "shuffled-donor null"
        ax.plot(x, y, marker="o", ms=3.5, lw=1.4, color=colors[arm], label=label, zorder=2)
    ax.axhline(0.0, color="0.75", lw=0.8, zorder=0)
    ax.axhline(1.0, color="0.75", lw=0.8, ls="--", zorder=0)
    ax.set_xticks(x)
    ax.set_xticklabels(
        [v if (v.startswith("joint") or int(v[1:]) % 4 == 0) else "" for v in LADDER],
        rotation=60,
        fontsize=7,
    )
    ax.set_xlabel("steered layer variant")
    ax.set_ylabel("F_beh")
    ax.set_title("Reversed direction (persona → bare), context-end replace")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    return savefig_paper(fig, slug, dir="figures/")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--work-root", type=Path, default=Path("eval_results/issue_2094/judge_rev"))
    ap.add_argument(
        "--floor-ceiling",
        type=Path,
        default=Path(
            "data/issue_2094/rev_rollouts/issue2094_singlepos/raw_completions/"
            "rev_direction/manifests/rev_floor_ceiling.json"
        ),
    )
    ap.add_argument(
        "--rollouts-dir",
        type=Path,
        default=Path(
            "data/issue_2094/rev_rollouts/issue2094_singlepos/raw_completions/rev_direction"
        ),
        help="rev shard dir; asserts 1:1 cell coverage vs the score waves + "
        "enriches donor_pair_id (the wave rows drop it)",
    )
    ap.add_argument("--fig-slug", type=str, default="issue_2094/rev_direction_fbeh_ladder")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    scores = args.work_root / "scores"
    coh_rows = _read_jsonl(scores / "coherence.grid.scores.jsonl")
    sa_rows = _read_jsonl(scores / "fp-persona.grid.scores.jsonl")
    sb_rows = _read_jsonl(scores / "fp-bare.grid.scores.jsonl")
    fc = json.loads(args.floor_ceiling.read_text(encoding="utf-8"))
    # Registry sanity: the reduce's pair set is exactly the banked rev bank's.
    rev_ids = {p.pair_id for p in BANK.build_rev_pairs()}
    assert {e["pair_id"] for e in fc["pairs"]} == rev_ids, "floor/ceiling pair set drift"

    # 1:1 coverage vs the generated grid + donor enrichment from the shards.
    donor_by_cell: dict[tuple[str, str], str | None] = {}
    for shard in sorted(args.rollouts_dir.glob("shard_*.jsonl")):
        for r in _read_jsonl(shard):
            k = _cell_key(r)
            assert k not in donor_by_cell, f"duplicate shard cell {k}"
            donor_by_cell[k] = r.get("donor_pair_id")
    assert donor_by_cell, f"no shard rows under {args.rollouts_dir}"
    assert set(donor_by_cell) == {_cell_key(r) for r in coh_rows}, (
        "shard cells != judged cells — waves and rollouts out of sync"
    )

    cells, aggregates = reduce_cells(
        coh_rows, sa_rows, sb_rows, fc["pairs"], donor_by_cell=donor_by_cell
    )
    n_incoherent = sum(1 for c in cells if not c["coherent"])
    summary = {
        "direction": fc["direction"],
        "delta_definition": fc["delta_definition"],
        "coherence_threshold": COHERENCE_THRESHOLD,
        "well_separated_min_denominator": WELL_SEPARATED_MIN,
        "n_cells": len(cells),
        "n_incoherent_excluded": n_incoherent,
        "n_judge_dropped": sum(1 for c in cells if c.get("missing") == "judge_dropped"),
        "floor_ceiling": {
            e["pair_id"]: {
                "floor": e["floor"]["delta_mean"],
                "ceiling": e["ceiling"]["delta_mean"],
                "denominator": e["denominator"],
            }
            for e in fc["pairs"]
        },
        "aggregates": aggregates,
        "cells": cells,
        "repro": {
            **as_metadata_dict(git_provenance()),
            "script": "scripts/issue2094_rev_reduce.py",
        },
    }
    out = args.work_root / "rev_fbeh_summary.json"
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out)
    print(f"[rev-reduce] {len(cells)} cells ({n_incoherent} incoherent-excluded) -> {out}")
    for v in ("joint_all", "joint_mid"):
        st, nu = aggregates[f"{v}|steered"], aggregates[f"{v}|null"]
        print(
            f"[rev-reduce] {v}: steered {st['f_beh_mean']:.3f} (n={st['n_included']}) "
            f"vs null {nu['f_beh_mean']:.3f} (n={nu['n_included']})"
        )
    if not args.no_figure:
        paths = render_figure(aggregates, cells, args.fig_slug)
        print(f"[rev-reduce] figure -> {paths.get('png', paths)}")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
