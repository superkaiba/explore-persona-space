"""Issue #2094 pair-family F_beh reduce + ladder figure (rev + butler families).

Consumes a family's judge waves (coherence + the per-side fp-* prefix waves)
and the BANKED per-pair floors/ceilings, and writes the family's F_beh summary
plus the layer-ladder figure (steered vs norm-matched shuffled-donor null).

Families (``--family``):

- ``mqrev`` (default — the reversed-direction round, this script's original
  scope): 5 ``mqrev--`` pairs, side a = fp-persona, side b = fp-bare, waves
  under ``eval_results/issue_2094/judge_rev/``, floors/ceilings banked from
  the parent's anchor judge draws (``rev_floor_ceiling.json``).
- ``mqb`` (Option C butler round): 10 ``mqb--`` pairs in TWO sub-families with
  DIFFERENT side-a rubrics — bare→butler (a = fp-bare) and persona→butler
  (a = fp-persona); side b is always fp-butler. Rubrics are driven PER PAIR
  off the floor/ceiling entries' ``rubric_a``/``rubric_b`` fields (present in
  ``butler_floor_ceiling.json``; derived from the context ids for the older
  rev manifest, which predates the fields). Sub-families aggregate SEPARATELY
  (``sub_family_aggregates``) — installing into plain context vs swapping one
  persona for another are different questions; averaging them would hide the
  contrast the round exists to measure.

Arithmetic is the parent's pre-registered reduce, REUSED not re-derived:
per-cell Δ_patched = (judge_B - judge_A)/100 via ``fmetrics.delta_contrast``,
F = (Δ_patched - Δ_floor)/(Δ_ceiling - Δ_floor) via ``fmetrics.f_beh``
(batched), coherence gate score > 60 (``issue2094_judge.COHERENCE_THRESHOLD``),
coherent-only cells (incoherent cells marked + excluded, never dropped — the
parent ``assemble_shard_rows`` convention), and weak-separation pairs
(|denominator| < 0.5) marked per cell + EXCLUDED from aggregate means,
reported under ``weak_separation_pairs``. The weak-separation exclusion is a
structural no-op for the rev family (all 5 rev denominators are 0.809-1.871),
so the committed ``rev_fbeh_summary.json`` reproduces byte-identically up to
its ``repro`` provenance block (which stamps the producing commit by design).
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

FAMILIES = {
    "mqrev": {
        "tag": "rev-reduce",
        "builder": BANK.build_rev_pairs,
        "summary_name": "rev_fbeh_summary.json",
        "title": "Reversed direction (persona → bare), context-end replace",
        "steered_label": "steered (persona → bare)",
    },
    "mqb": {
        "tag": "butler-reduce",
        "builder": BANK.build_butler_pairs,
        "summary_name": "butler_fbeh_summary.json",
        "title": "Butler transfer, context-end replace",
        "steered_label": "steered",
    },
}


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


def _rubrics_for(entry: dict) -> tuple[str, str]:
    """Per-pair (rubric_a, rubric_b): explicit fields preferred, else derived.

    ``butler_floor_ceiling.json`` carries ``rubric_a``/``rubric_b`` per pair
    (the butler sub-families differ in side a); the older rev manifest predates
    the fields, where side rubrics are the contexts' own prefixes.
    """
    ra = entry.get("rubric_a") or f"fp-{entry['context_a'].split('__')[0]}"
    rb = entry.get("rubric_b") or f"fp-{entry['context_b'].split('__')[0]}"
    return ra, rb


def _subfamily_by_pair(fc_pairs: list[dict]) -> dict[str, str]:
    """pair_id -> 'a_prefix->b_prefix' (from prefix_pair, else the context ids)."""
    out = {}
    for e in fc_pairs:
        pp = e.get("prefix_pair") or [
            e["context_a"].split("__")[0],
            e["context_b"].split("__")[0],
        ]
        out[e["pair_id"]] = f"{pp[0]}->{pp[1]}"
    return out


def reduce_cells(
    coh_rows: list[dict],
    side_a_rows: list[dict],
    side_b_rows: list[dict],
    fc_pairs: list[dict],
    donor_by_cell: dict[tuple[str, str], str | None] | None = None,
) -> tuple[list[dict], dict]:
    """Per-cell F_beh rows + per-(layer_variant, arm) aggregates (pure).

    ``side_a_rows`` / ``side_b_rows`` are the side-'a' / side-'b' fp-* wave
    score rows; each cell's expected rubric ids are driven PER PAIR from its
    floor/ceiling entry (``_rubrics_for``). Missing judge scores and incoherent
    cells are marked + excluded from aggregates, never dropped; cells of
    weak-separation pairs (|denominator| < ``WELL_SEPARATED_MIN``) keep their
    raw F values but are excluded from aggregate means.
    """
    fc = {e["pair_id"]: e for e in fc_pairs}
    assert len(fc) == len(fc_pairs) and fc, sorted(fc)
    coh = _index_unique(coh_rows, "coherence")
    sa = _index_unique(side_a_rows, "side-a")
    sb = _index_unique(side_b_rows, "side-b")
    assert set(coh) == set(sa) == set(sb), "cell sets differ across the three waves"

    cells: list[dict] = []
    dp_idx: list[int] = []
    dp, df, dc = [], [], []
    for k in sorted(coh):
        c = coh[k]
        pair_id = c["pair_id"]
        assert pair_id in fc, pair_id
        entry = fc[pair_id]
        rubric_a, rubric_b = _rubrics_for(entry)
        a_row, b_row = sa[k], sb[k]
        assert a_row["side"] == "a" and a_row["rubric_id"] == rubric_a, a_row["item_id"]
        assert b_row["side"] == "b" and b_row["rubric_id"] == rubric_b, b_row["item_id"]
        coh_score = c.get("score")
        coherent = coh_score is not None and float(coh_score) > COHERENCE_THRESHOLD
        a_score, b_score = a_row.get("score"), b_row.get("score")
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
            f"judge_a_{rubric_a.replace('-', '_')}": a_score,
            f"judge_b_{rubric_b.replace('-', '_')}": b_score,
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
        if (
            cell.get("f_beh") is not None
            and not cell.get("degenerate_denominator")
            and cell["well_separated"]
        ):
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


def subfamily_aggregates(cells: list[dict], fc_pairs: list[dict]) -> dict:
    """Per-(layer_variant, arm, sub-family) aggregates; {} for a one-sub-family
    reduce (the rev family), keeping its summary shape byte-stable."""
    fam_by_pair = _subfamily_by_pair(fc_pairs)
    fams = sorted(set(fam_by_pair.values()))
    if len(fams) < 2:
        return {}
    out: dict[str, dict] = {}
    for variant in LADDER:
        for arm in ARMS:
            for fam in fams:
                sub = [
                    c
                    for c in cells
                    if c["layer_variant"] == variant
                    and c["arm"] == arm
                    and fam_by_pair[c["pair_id"]] == fam
                ]
                inc = [
                    c["f_beh"]
                    for c in sub
                    if c.get("f_beh") is not None
                    and not c.get("degenerate_denominator")
                    and c["well_separated"]
                ]
                out[f"{variant}|{arm}|{fam}"] = {
                    "layer_variant": variant,
                    "arm": arm,
                    "sub_family": fam,
                    "f_beh_mean": (sum(inc) / len(inc)) if inc else None,
                    "n_included": len(inc),
                    "n_coherent": sum(1 for c in sub if c["coherent"]),
                    "n_total": len(sub),
                    "per_pair_f_beh": sorted((c["pair_id"], c.get("f_beh")) for c in sub),
                }
    return out


def _ladder_panel(ax, mean_of, points: dict[str, list[tuple[int, float]]], steered_label: str):
    """One ladder panel: per-arm mean lines over per-pair scatter (shared shape)."""
    x = list(range(len(LADDER)))
    colors = {"steered": paper_palette_role("primary"), "null": paper_palette_role("control")}
    for arm in ARMS:
        for xi, yv in points.get(arm, ()):
            ax.scatter(xi, yv, s=6, alpha=0.25, color=colors[arm], linewidths=0, zorder=1)
        y = [mean_of(v, arm) for v in LADDER]
        label = steered_label if arm == "steered" else "shuffled-donor null"
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


def _points(cells: list[dict]) -> dict[str, list[tuple[int, float]]]:
    """Scatter points mirror the aggregate-inclusion rule: weak-separation cells
    are excluded (their F is meaningless under a degenerate normalizer and drags
    the y-range; raw values remain in the summary JSON). No-op for rev (all
    well-separated)."""
    pts: dict[str, list[tuple[int, float]]] = {arm: [] for arm in ARMS}
    for c in cells:
        if c.get("f_beh") is None or not c["well_separated"]:
            continue
        pts[c["arm"]].append((LADDER.index(c["layer_variant"]), c["f_beh"]))
    return pts


def render_figure(
    aggregates: dict, cells: list[dict], slug: str, *, title: str, steered_label: str
) -> dict[str, Path]:
    """Single-panel ladder figure: mean F_beh per layer variant, steered vs null."""
    set_paper_style()
    fig, ax = plt.subplots(figsize=(7.0, 3.4))
    _ladder_panel(
        ax, lambda v, arm: aggregates[f"{v}|{arm}"]["f_beh_mean"], _points(cells), steered_label
    )
    ax.set_ylabel("F_beh")
    ax.set_title(title)
    ax.legend(frameon=False, loc="upper left", fontsize=8)
    return savefig_paper(fig, slug, dir="figures/")


def render_figure_subfamilies(
    sub_aggs: dict, cells: list[dict], fam_by_pair: dict[str, str], slug: str, *, title: str
) -> dict[str, Path]:
    """Two-panel ladder figure, one panel per sub-family (steered vs null each)."""
    set_paper_style()
    fams = sorted({v["sub_family"] for v in sub_aggs.values()})
    fig, axes = plt.subplots(1, len(fams), figsize=(4.9 * len(fams), 3.4), sharey=True)
    for ax, fam in zip(axes, fams, strict=True):
        fam_cells = [c for c in cells if fam_by_pair[c["pair_id"]] == fam]
        _ladder_panel(
            ax,
            lambda v, arm, fam=fam: sub_aggs[f"{v}|{arm}|{fam}"]["f_beh_mean"],
            _points(fam_cells),
            "steered",
        )
        ax.set_title(fam.replace("->", " → "))
    axes[0].set_ylabel("F_beh")
    axes[0].legend(frameon=False, loc="upper left", fontsize=8)
    fig.suptitle(title)
    return savefig_paper(fig, slug, dir="figures/")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--family", choices=sorted(FAMILIES), default="mqrev")
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
        help="grid shard dir; asserts 1:1 cell coverage vs the score waves + "
        "enriches donor_pair_id (the wave rows drop it)",
    )
    ap.add_argument("--fig-slug", type=str, default="issue_2094/rev_direction_fbeh_ladder")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)
    fam_cfg = FAMILIES[args.family]
    tag = fam_cfg["tag"]

    scores = args.work_root / "scores"
    coh_rows = _read_jsonl(scores / "coherence.grid.scores.jsonl")
    side_rows: list[dict] = []
    for f in sorted(scores.glob("fp-*.grid.scores.jsonl")):
        side_rows.extend(_read_jsonl(f))
    sa_rows = [r for r in side_rows if r["side"] == "a"]
    sb_rows = [r for r in side_rows if r["side"] == "b"]
    fc = json.loads(args.floor_ceiling.read_text(encoding="utf-8"))
    # Registry sanity: the reduce's pair set is exactly the family's banked set.
    expected_ids = {p.pair_id for p in fam_cfg["builder"]()}
    assert {e["pair_id"] for e in fc["pairs"]} == expected_ids, "floor/ceiling pair set drift"

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
    sub_aggs = subfamily_aggregates(cells, fc["pairs"])
    weak = sorted(e["pair_id"] for e in fc["pairs"] if abs(e["denominator"]) < WELL_SEPARATED_MIN)
    n_incoherent = sum(1 for c in cells if not c["coherent"])
    summary = {
        "direction": fc["direction"],
        "delta_definition": fc["delta_definition"],
        "coherence_threshold": COHERENCE_THRESHOLD,
        "well_separated_min_denominator": WELL_SEPARATED_MIN,
        **({"weak_separation_pairs": weak} if weak else {}),
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
        **({"sub_family_aggregates": sub_aggs} if sub_aggs else {}),
        "cells": cells,
        "repro": {
            **as_metadata_dict(git_provenance()),
            "script": "scripts/issue2094_rev_reduce.py",
        },
    }
    out = args.work_root / fam_cfg["summary_name"]
    tmp = out.with_name(out.name + ".tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    tmp.replace(out)
    print(f"[{tag}] {len(cells)} cells ({n_incoherent} incoherent-excluded) -> {out}")
    if weak:
        print(f"[{tag}] weak-separation pairs excluded from aggregate means: {weak}")
    for v in ("joint_all", "joint_mid"):
        st, nu = aggregates[f"{v}|steered"], aggregates[f"{v}|null"]
        print(
            f"[{tag}] {v}: steered {st['f_beh_mean']:.3f} (n={st['n_included']}) "
            f"vs null {nu['f_beh_mean']:.3f} (n={nu['n_included']})"
        )
    if sub_aggs:
        fams = sorted({v["sub_family"] for v in sub_aggs.values()})
        for fam in fams:
            for v in ("joint_all", "joint_mid"):
                st = sub_aggs[f"{v}|steered|{fam}"]
                nu = sub_aggs[f"{v}|null|{fam}"]
                print(
                    f"[{tag}] {fam} {v}: steered {st['f_beh_mean']:.3f} "
                    f"(n={st['n_included']}) vs null {nu['f_beh_mean']:.3f} "
                    f"(n={nu['n_included']})"
                )
    if not args.no_figure:
        if sub_aggs:
            paths = render_figure_subfamilies(
                sub_aggs,
                cells,
                _subfamily_by_pair(fc["pairs"]),
                args.fig_slug,
                title=fam_cfg["title"],
            )
        else:
            paths = render_figure(
                aggregates,
                cells,
                args.fig_slug,
                title=fam_cfg["title"],
                steered_label=fam_cfg["steered_label"],
            )
        print(f"[{tag}] figure -> {paths.get('png', paths)}")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
