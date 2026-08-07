#!/usr/bin/env python3
"""Issue #2162 — P7 analysis driver (cpu-bigmem pod / VM).

Consumes the pod artifacts (rollout shards + V_a stores + vc_bank + margin
shards) and the judge outputs (per-wave ``*.scores.jsonl``) and produces the
plan §6 statistical outputs:

- ``--step f-tables``: the four registered F tables
  ``eval_results/issue_2162/f_metrics/{f_cells,null_shuffled_cells,
  null_crosstype_cells,anchors}.jsonl`` — per (pair x slot x arm) F_beh
  (graded dual-rubric contrast, coherent draws only), F_act (signed
  projection at read layer 26, disjoint floor halves via ``fmetrics.f_act``),
  anchor separation, coherence/cap-hit accounting, donor provenance.
- ``--step stats``: pre-registered separation exclusion (|sep| >= 0.5) with
  pre-exclusion counts, the intersection-union exact Wilcoxon signed-rank
  (p = max over both nulls) Holm-corrected WITHIN the three declared families
  (P1 role m=31 / P2 route m=15 / P3 dose-position m=28), pair-clustered
  bootstrap 95% CIs (B=10,000, seed 21620, batched index-GEMM via
  ``issue2094_analysis.bootstrap_family_means_batched``), the disjoint-CI
  check against BOTH nulls, realized-MDE report, the registered P3 per-pair
  depth/load slopes with pair-clustered bootstrap CIs (``dose_slopes``), and
  the stage-2 selection -> ``best_cells.json`` (cap 12 by descending steered
  F_beh).
- ``--step margin``: TF fixed-pool margin reduction + the rule-19 validation
  rho(margin shift, F_beh) — reported BEFORE the margin carries any read.
- ``--step probe``: batched-torch L2 logistic read probe per (type x slot x
  layer x value-pair) on the P1 context states — LOCO 12 carrier folds
  (primary) + value-pair transfer (secondary) — with the max-over-layers
  selection-symmetric permutation band (B=1,000 within-carrier label flips,
  per-draw re-maxed; the per-draw x per-layer AUC matrix is persisted).
- ``--step two-by-two``: the read x write 2x2 verdicts incl. the explicit
  ``untestable-causal`` label for sub-floor cells (post-exclusion n < 12).
- ``--step stage2``: reduces the stage-2 layer-localization rollouts + judge
  scores to ``stage2_cells.jsonl`` — one row per (pair x slot x arm x layer x
  dose), same F_beh reduction + shared anchor pools as ``f-tables`` (the
  manifest ``layer_profile_stage2`` source; skipped under ``--step all`` when
  ``--stage2-dir`` is absent, i.e. before stage 2 has run).

Checkpoint-per-step: every step writes its outputs atomically the moment it
completes and re-runs skip nothing silently (inputs are re-read; outputs are
overwritten deterministically — same seed, same inputs, same bytes).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from collections import defaultdict
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2162_judge as J  # noqa: E402  (rubric ids + loaders + anchor_unit_id)
import issue2162_run as R  # noqa: E402  (pool_key + block enumeration)
from issue2094_analysis import bootstrap_family_means_batched  # noqa: E402
from explore_persona_space.experiments.issue2094 import fmetrics as FM  # noqa: E402
from explore_persona_space.experiments.issue2162 import bank2162 as BANK  # noqa: E402

logger = logging.getLogger("issue2162.analysis")

READ_LAYER = 26  # plan §4.4: the #2094 primary read layer
COHERENCE_THRESHOLD = 60.0
SEPARATION_BAR = 0.5
SURVIVAL_FLOOR = 12  # exact signed-rank attainability + pre-registered floor
BOOT_B = 10_000
BOOT_SEED = 21620
PROBE_SEED = 21621
PROBE_PERM_B = 1_000
STAGE2_CAP = 12
HOLM_ALPHA = 0.05
# Plan §4.1 length mitigation (r1 M7): the length-matched sensitivity subset.
LEN_MATCH_MAX_ABS = 2

ROUTE_VARIANT_TYPES = ("demo_format", "demo_persona", "language_implied", "persona_role_header")


def family_of(cell: str, slot: str) -> str | None:
    """P1 role / P2 route / P3 dose-position family (plan §6); None = excluded.

    Constructional exclusions (never enter m): ``filler_swap`` (no F),
    ``query_content``@pe and ``persona_role_header``@pe (pre-declared
    degenerate at prefix-end).
    """
    base = BANK.base_type_of(cell)
    if base == "filler_swap":
        return None
    if slot == "pe" and base in BANK.DEGENERATE_AT_PE:
        return None
    if cell != base:  # crossed cells
        if cell.startswith("conflict_"):
            return "P2"
        return "P3"  # recency_* / load_*
    if base in ROUTE_VARIANT_TYPES:
        return "P2"
    return "P1"


# ── io ────────────────────────────────────────────────────────────────


def _write_json_atomic(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def _write_jsonl_atomic(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    tmp.replace(path)


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_wave_scores(scores_dir: Path, suffix: str) -> dict[str, float | None]:
    """item_id -> mean kept score, over every ``*.{suffix}.scores.jsonl`` wave."""
    out: dict[str, float | None] = {}
    files = sorted(scores_dir.glob(f"*.{suffix}.scores.jsonl"))
    assert files, f"no {suffix} score files under {scores_dir}"
    for f in files:
        for row in _iter_jsonl(f):
            out[row["item_id"]] = row["score"]
    return out


# ── step: f-tables ────────────────────────────────────────────────────


def _grid_behavior_score(
    scores: dict[str, float | None], tag: str, block_key: str, pair_id: str, draw: int, side: str
) -> float | None:
    return scores.get(J.J94._item_id(tag, f"{tag}|{block_key}|{pair_id}|{draw}|{side}"))


def _load_va_store(va_dir: Path) -> dict[tuple[str, str, str, int], torch.Tensor]:
    """(block_key, pair_id, context_a, draw) -> layer-26 span-mean V_a (H,)."""
    out: dict[tuple[str, str, str, int], torch.Tensor] = {}
    for shard in sorted(va_dir.glob("shard_*.pt")):
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        layers = payload["layers"]
        li = layers.index(READ_LAYER) if READ_LAYER in layers else len(layers) - 1
        va = payload["va_span"]
        for j, meta in enumerate(payload["index"]):
            out[(payload["block_key"], meta["pair_id"], meta["context_a"], meta["draw"])] = va[
                j, li
            ].float()
    assert out, f"no V_a shards under {va_dir}"
    return out


def _load_anchor_va(anchors_dir: Path) -> dict[tuple[str, int], torch.Tensor]:
    """(context_id, draw) -> layer-26 span-mean anchor V_a (H,)."""
    out: dict[tuple[str, int], torch.Tensor] = {}
    for shard in sorted(anchors_dir.glob("va_anchors_*.pt")):
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        layers = payload["layers"]
        li = layers.index(READ_LAYER) if READ_LAYER in layers else len(layers) - 1
        va = payload["va_span"]
        for j, meta in enumerate(payload["index"]):
            out[(meta["context_id"], meta["draw"])] = va[j, li].float()
    assert out, f"no anchor V_a shards under {anchors_dir}"
    return out


def _anchor_deltas(
    anchor_rows: list[dict],
    scores: dict[str, float | None],
    coherence: dict[str, float | None],
    pair: BANK.Pair2162,
    ctx: str,
) -> list[float]:
    """Coherent kept per-draw dual-rubric contrasts for one anchor context."""
    cores = J.pair_rubric_cores(pair)
    assert cores is not None
    rid_a, rid_b = (J.rubric_core_id(c) for c in cores)
    deltas: list[float] = []
    for row in anchor_rows:
        if row["context_id"] != ctx:
            continue
        coh = coherence.get(J.J94._item_id("c", f"a|{ctx}|{row['draw']}"))
        if coh is None or coh <= COHERENCE_THRESHOLD:
            continue
        sa = scores.get(J.anchor_unit_id(ctx, row["draw"], rid_a))
        sb = scores.get(J.anchor_unit_id(ctx, row["draw"], rid_b))
        if sa is None or sb is None:
            continue
        deltas.append((sb - sa) / 100.0)
    return deltas


def step_f_tables(args: argparse.Namespace) -> None:
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    grid_rows = J.load_grid_rows(args.rollouts_dir)
    anchor_rows = J.load_anchor_rows(args.anchors_dir)
    beh_grid = load_wave_scores(args.scores_dir, "grid")
    beh_anchor = load_wave_scores(args.scores_dir, "anchors")
    coherence = {**beh_grid, **beh_anchor}  # coherence waves share the suffix files
    va_grid = _load_va_store(args.va_dir)
    va_anchor = _load_anchor_va(args.anchors_dir)

    # Per anchor context: coherent kept anchor V_a draws (floor/ceiling pools).
    anchor_draws_by_ctx: dict[str, list[int]] = defaultdict(list)
    for row in anchor_rows:
        anchor_draws_by_ctx[row["context_id"]].append(row["draw"])

    # Per-context anchor rollout totals + coherent counts — the anchor-baseline
    # incoherence term of the manifest coherence_caphit figure (excess
    # incoherence = arm incoherent fraction MINUS this baseline rate).
    anchor_total_by_ctx: dict[str, int] = defaultdict(int)
    anchor_coherent_by_ctx: dict[str, int] = defaultdict(int)
    for row in anchor_rows:
        anchor_total_by_ctx[row["context_id"]] += 1
        coh = coherence.get(J.J94._item_id("c", f"a|{row['context_id']}|{row['draw']}"))
        if coh is not None and coh > COHERENCE_THRESHOLD:
            anchor_coherent_by_ctx[row["context_id"]] += 1

    # Anchor table rows (per pair: floor/ceiling deltas + separation).
    anchors_out: list[dict] = []
    sep_by_pair: dict[str, float | None] = {}
    for p in pairs:
        if J.pair_rubric_cores(p) is None:
            continue
        d_floor = _anchor_deltas(anchor_rows, beh_anchor, coherence, p, p.a)
        d_ceiling = _anchor_deltas(anchor_rows, beh_anchor, coherence, p, p.b)
        sep = (
            (sum(d_ceiling) / len(d_ceiling) - sum(d_floor) / len(d_floor))
            if d_floor and d_ceiling
            else None
        )
        sep_by_pair[p.pair_id] = sep
        anchors_out.append(
            {
                "pair_id": p.pair_id,
                "cell": p.cell,
                "carrier": p.carrier,
                "value_a": p.value_a,
                "value_b": p.value_b,
                "n_floor_draws": len(d_floor),
                "n_ceiling_draws": len(d_ceiling),
                "n_floor_rollouts": anchor_total_by_ctx.get(p.a, 0),
                "n_floor_coherent": anchor_coherent_by_ctx.get(p.a, 0),
                "n_ceiling_rollouts": anchor_total_by_ctx.get(p.b, 0),
                "n_ceiling_coherent": anchor_coherent_by_ctx.get(p.b, 0),
                "delta_floor_mean": sum(d_floor) / len(d_floor) if d_floor else None,
                "delta_ceiling_mean": sum(d_ceiling) / len(d_ceiling) if d_ceiling else None,
                "separation": sep,
            }
        )
    _write_jsonl_atomic(args.out_dir / "anchors.jsonl", anchors_out)

    # Grid cell rows: one row per (pair x slot x arm).
    by_cell_rows: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in grid_rows:
        by_cell_rows[(row["pair_id"], row["slot"], row["arm"])].append(row)
    tables: dict[str, list[dict]] = {"steered": [], "shuffled": [], "crosstype": []}
    for (pair_id, slot, arm), rows in sorted(by_cell_rows.items()):
        p = pairs_by_id[pair_id]
        if J.pair_rubric_cores(p) is None:
            continue  # filler_swap: disruption DV handled separately
        deltas: list[float] = []
        n_coherent = 0
        n_cap = 0
        for row in rows:
            n_cap += int(row.get("cap_hit", False))
            coh = coherence.get(
                J.J94._item_id("c", f"g|{row['block_key']}|{pair_id}|{row['draw']}")
            )
            if coh is None or coh <= COHERENCE_THRESHOLD:
                continue
            n_coherent += 1
            sa = _grid_behavior_score(beh_grid, "g", row["block_key"], pair_id, row["draw"], "a")
            sb = _grid_behavior_score(beh_grid, "g", row["block_key"], pair_id, row["draw"], "b")
            if sa is None or sb is None:
                continue
            deltas.append((sb - sa) / 100.0)
        d_floor = _anchor_deltas(anchor_rows, beh_anchor, coherence, p, p.a)
        d_ceiling = _anchor_deltas(anchor_rows, beh_anchor, coherence, p, p.b)
        f_beh = None
        if deltas and d_floor and d_ceiling:
            dp = sum(deltas) / len(deltas)
            df = sum(d_floor) / len(d_floor)
            dc = sum(d_ceiling) / len(d_ceiling)
            f_beh = (dp - df) / (dc - df) if abs(dc - df) > 1e-9 else None

        # F_act at the read layer (fmetrics.f_act; disjoint floor halves).
        f_act = None
        va_patched = [
            va_grid[k]
            for row in rows
            if (k := (row["block_key"], pair_id, row["context_id"], row["draw"])) in va_grid
        ]
        floor_va = [
            va_anchor[(p.a, d)] for d in anchor_draws_by_ctx.get(p.a, []) if (p.a, d) in va_anchor
        ]
        ceil_va = [
            va_anchor[(p.b, d)] for d in anchor_draws_by_ctx.get(p.b, []) if (p.b, d) in va_anchor
        ]
        if va_patched and len(floor_va) >= 2 and ceil_va:
            res = FM.f_act(
                torch.stack(va_patched).mean(dim=0),
                torch.stack(floor_va),
                torch.stack(ceil_va),
            )
            f_act = float(res.f_act)

        rec = {
            "pair_id": pair_id,
            "cell": p.cell,
            "slot": slot,
            "arm": arm,
            "carrier": p.carrier,
            "value_a": p.value_a,
            "value_b": p.value_b,
            "donor_pair_id": rows[0].get("donor_pair_id"),
            "donor_cell": (
                pairs_by_id[rows[0]["donor_pair_id"]].cell if rows[0].get("donor_pair_id") else None
            ),
            "donor_value_b": (
                pairs_by_id[rows[0]["donor_pair_id"]].value_b
                if rows[0].get("donor_pair_id")
                else None
            ),
            "n_draws": len(rows),
            "n_coherent": n_coherent,
            "n_scored": len(deltas),
            "n_cap_hit": n_cap,
            "delta_patched_mean": sum(deltas) / len(deltas) if deltas else None,
            "f_beh": f_beh,
            "f_act": f_act,
            "separation": sep_by_pair.get(pair_id),
            "family": family_of(p.cell, slot),
            # Plan §4.1 length covariate (r1 M7): per-pair token-length delta.
            "len_delta": rows[0].get("len_delta"),
        }
        key = {"steered": "steered", "shuffled": "shuffled", "crosstype": "crosstype"}[arm]
        tables[key].append(rec)
    _write_jsonl_atomic(args.out_dir / "f_cells.jsonl", tables["steered"])
    _write_jsonl_atomic(args.out_dir / "null_shuffled_cells.jsonl", tables["shuffled"])
    _write_jsonl_atomic(args.out_dir / "null_crosstype_cells.jsonl", tables["crosstype"])
    logger.info(
        "[f-tables] steered=%d shuffled=%d crosstype=%d anchors=%d",
        len(tables["steered"]),
        len(tables["shuffled"]),
        len(tables["crosstype"]),
        len(anchors_out),
    )


# ── step: stats ───────────────────────────────────────────────────────


def _wilcoxon_exact_p(diffs: np.ndarray) -> float:
    """Exact two-sided Wilcoxon signed-rank p (zero diffs dropped, ties mean-ranked).

    scipy falls back from exact to the normal approximation itself when ties
    make the exact distribution unavailable (method="auto" semantics); we
    request exact at the plan's n<=36 scale and let scipy degrade on ties.
    """
    from scipy.stats import wilcoxon

    d = diffs[np.abs(diffs) > 0]
    if len(d) < 1:
        return 1.0
    method = "exact" if (len(d) <= 50 and len(np.unique(np.abs(d))) == len(d)) else "auto"
    return float(wilcoxon(d, alternative="two-sided", method=method).pvalue)


def holm(pvals: dict[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values within one family."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[key] = running
    return adj


def step_stats(args: argparse.Namespace) -> None:
    steered = list(_iter_jsonl(args.out_dir / "f_cells.jsonl"))
    nulls = {
        "shuffled": list(_iter_jsonl(args.out_dir / "null_shuffled_cells.jsonl")),
        "crosstype": list(_iter_jsonl(args.out_dir / "null_crosstype_cells.jsonl")),
    }

    def index(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
        return {(r["pair_id"], r["slot"], r["arm"]): r for r in rows}

    idx_null = {k: index(v) for k, v in nulls.items()}
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in steered:
        cells[(r["cell"], r["slot"])].append(r)

    per_cell: dict[str, dict] = {}
    family_p: dict[str, dict[str, float]] = defaultdict(dict)
    boot_values: list[np.ndarray] = []
    boot_labels: list[str] = []
    for (cell, slot), rows in sorted(cells.items()):
        fam = family_of(cell, slot)
        pre_n = len(rows)
        kept = [
            r
            for r in rows
            if r["separation"] is not None and abs(r["separation"]) >= SEPARATION_BAR
        ]
        key = f"{cell}|{slot}"
        diffs: dict[str, list[float]] = {"shuffled": [], "crosstype": []}
        f_steered: list[float] = []
        f_null: dict[str, list[float]] = {"shuffled": [], "crosstype": []}
        lm_steered: list[float] = []
        lm_null: dict[str, list[float]] = {"shuffled": [], "crosstype": []}
        for r in kept:
            if r["f_beh"] is None:
                continue
            per_null_f = {}
            for null in ("shuffled", "crosstype"):
                nr = idx_null[null].get((r["pair_id"], r["slot"], null))
                per_null_f[null] = nr["f_beh"] if nr else None
            if any(v is None for v in per_null_f.values()):
                continue
            f_steered.append(r["f_beh"])
            for null in ("shuffled", "crosstype"):
                f_null[null].append(per_null_f[null])
                diffs[null].append(r["f_beh"] - per_null_f[null])
            # Plan §4.1 length-matched sensitivity subset (r1 M7).
            ld = r.get("len_delta")
            if ld is not None and abs(int(ld)) <= LEN_MATCH_MAX_ABS:
                lm_steered.append(r["f_beh"])
                for null in ("shuffled", "crosstype"):
                    lm_null[null].append(per_null_f[null])
        n = len(f_steered)
        testable = n >= SURVIVAL_FLOOR
        p_iut = None
        if testable:
            p_iut = max(
                _wilcoxon_exact_p(np.asarray(diffs["shuffled"])),
                _wilcoxon_exact_p(np.asarray(diffs["crosstype"])),
            )
            if fam is not None:
                family_p[fam][key] = p_iut
        # Pair-clustered bootstrap columns: steered + both nulls per cell.
        pad = max(n, 1)
        for label, vals in (
            (f"{key}|steered", f_steered),
            (f"{key}|shuffled", f_null["shuffled"]),
            (f"{key}|crosstype", f_null["crosstype"]),
        ):
            col = np.full(pad, np.nan)
            col[: len(vals)] = vals
            boot_values.append(col)
            boot_labels.append(label)
        per_cell[key] = {
            "cell": cell,
            "slot": slot,
            "family": fam,
            "n_pre_exclusion": pre_n,
            "n_post_exclusion": n,
            "untestable_causal": not testable,
            "f_steered_mean": float(np.mean(f_steered)) if f_steered else None,
            "f_shuffled_mean": float(np.mean(f_null["shuffled"])) if f_null["shuffled"] else None,
            "f_crosstype_mean": (
                float(np.mean(f_null["crosstype"])) if f_null["crosstype"] else None
            ),
            "p_iut": p_iut,
            "realized_mde_single_test": (1.02 / math.sqrt(n)) if n else None,
            # Plan §4.1 length-matched sensitivity recount (r1 M7): the same
            # per-cell means over |Δlen| <= 2 pairs only ("where the varied
            # span permits" — n=0 means the type's spans never length-match).
            "length_matched": {
                "max_abs_len_delta": LEN_MATCH_MAX_ABS,
                "n": len(lm_steered),
                "f_steered_mean": float(np.mean(lm_steered)) if lm_steered else None,
                "f_shuffled_mean": (
                    float(np.mean(lm_null["shuffled"])) if lm_null["shuffled"] else None
                ),
                "f_crosstype_mean": (
                    float(np.mean(lm_null["crosstype"])) if lm_null["crosstype"] else None
                ),
            },
        }

    # Pair-clustered bootstrap: pad columns to a shared pair axis per cell —
    # resample WITHIN cell (each cell's pairs are its cluster axis).
    for key, rec in per_cell.items():
        cols = [i for i, lab in enumerate(boot_labels) if lab.startswith(f"{key}|")]
        vals = np.stack([boot_values[i] for i in cols], axis=1)  # (n_pairs, 3)
        if not np.isfinite(vals).any():
            continue
        boots = bootstrap_family_means_batched(vals, BOOT_B, BOOT_SEED)
        lo, hi = np.nanpercentile(boots, [2.5, 97.5], axis=0)
        arms = [boot_labels[i].split("|")[-1] for i in cols]
        rec["ci95"] = {
            arm: [None if not np.isfinite(v) else float(v) for v in (lo[j], hi[j])]
            for j, arm in enumerate(arms)
        }
        s_lo = rec["ci95"]["steered"][0]
        rec["disjoint_both_nulls"] = bool(
            s_lo is not None
            and rec["ci95"]["shuffled"][1] is not None
            and rec["ci95"]["crosstype"][1] is not None
            and s_lo > rec["ci95"]["shuffled"][1]
            and s_lo > rec["ci95"]["crosstype"][1]
        )

    for fam, pvals in family_p.items():
        adj = holm(pvals)
        for key, p_adj in adj.items():
            per_cell[key]["p_holm"] = p_adj
            per_cell[key]["holm_family_m"] = len(pvals)
            per_cell[key]["holm_pass"] = p_adj < HOLM_ALPHA

    # Registered P3 secondary (manifest recency_load_curves transform): the
    # per-pair depth/load slope with a pair-clustered bootstrap 95% CI. A pair
    # is traced across levels by (carrier, value_a, value_b) within one base
    # type; level 1 = the uncrossed base cell, levels 3/5 = the crossed cells.
    crossed = set(BANK.crossed_cells())
    slope_groups: dict[tuple[str, str, str], dict[tuple[str, str, str], dict[int, float]]] = (
        defaultdict(lambda: defaultdict(dict))
    )
    for r in steered:
        if r["f_beh"] is None or r["separation"] is None or abs(r["separation"]) < SEPARATION_BAR:
            continue
        base = BANK.base_type_of(r["cell"])
        for prefix, tag in (("recency", "d"), ("load", "l")):
            if r["cell"] == base:
                if not any(c.startswith(f"{prefix}_{base}_") for c in crossed):
                    continue
                level = 1
            elif r["cell"].startswith(f"{prefix}_{base}_{tag}"):
                level = int(r["cell"].rsplit(tag, 1)[-1])
            else:
                continue
            pair_key = (r["carrier"], r["value_a"], r["value_b"])
            slope_groups[(prefix, base, r["slot"])][pair_key][level] = r["f_beh"]
    dose_slopes: dict[str, dict] = {}
    for (prefix, base, slot), pairs_map in sorted(slope_groups.items()):
        slopes: list[float] = []
        for _, by_level in sorted(pairs_map.items()):
            if len(by_level) < 2:
                continue
            xs = np.asarray(sorted(by_level), dtype=np.float64)
            ys = np.asarray([by_level[int(x)] for x in xs], dtype=np.float64)
            slopes.append(float(np.polyfit(xs, ys, 1)[0]))
        if not slopes:
            continue
        boots = bootstrap_family_means_batched(
            np.asarray(slopes, dtype=np.float64)[:, None], BOOT_B, BOOT_SEED
        )
        lo, hi = np.nanpercentile(boots[:, 0], [2.5, 97.5])
        dose_slopes[f"{prefix}|{base}|{slot}"] = {
            "n_pairs": len(slopes),
            "slope_mean": float(np.mean(slopes)),
            "ci95": [float(lo), float(hi)],
            "unit": "Delta F_beh per level step (levels 1/3/5; 1 = base cell)",
        }

    survivors = sorted(
        (
            rec
            for rec in per_cell.values()
            if rec.get("holm_pass") and rec.get("disjoint_both_nulls")
        ),
        key=lambda r: -(r["f_steered_mean"] or 0.0),
    )[:STAGE2_CAP]
    _write_json_atomic(
        args.out_dir / "best_cells.json",
        {
            "selection_rule": "Holm-IUT pass AND disjoint 95% CIs vs BOTH nulls; cap 12 by "
            "descending steered F_beh (plan §6 exclusion 5; stage-2 labeled post-selection)",
            "cells": [{"cell": r["cell"], "slot": r["slot"]} for r in survivors],
            "n_survivors": len(survivors),
        },
    )
    _write_json_atomic(
        args.out_dir / "stats.json",
        {
            "per_cell": per_cell,
            "dose_slopes": dose_slopes,
            "families": {fam: len(p) for fam, p in family_p.items()},
            "bars": {
                "separation_bar": SEPARATION_BAR,
                "survival_floor": SURVIVAL_FLOOR,
                "boot": {"B": BOOT_B, "seed": BOOT_SEED},
                "holm_alpha": HOLM_ALPHA,
                "joint_power_note": "realized MDE reported for the registered CONJUNCTION "
                "(Holm-IUT AND disjoint CIs) — plausibly ~0.23-0.25 at n=27, never the "
                "single-test 1.02/sqrt(n) line alone (plan §6)",
            },
        },
    )
    n_testable = sum(1 for r in per_cell.values() if not r["untestable_causal"])
    logger.info(
        "[stats] cells=%d testable=%d survivors=%d (families: %s)",
        len(per_cell),
        n_testable,
        len(survivors),
        {f: len(p) for f, p in family_p.items()},
    )


# ── step: margin ──────────────────────────────────────────────────────


def step_margin(args: argparse.Namespace) -> None:
    pairs_by_id = {p.pair_id: p for p in BANK.build_pairs()}
    rows = []
    for shard in sorted(args.margin_dir.glob("*.jsonl")):
        rows.extend(r for r in _iter_jsonl(shard) if not r.get("skipped"))
    assert rows, f"no margin rows under {args.margin_dir}"
    # Patched-state margins per (pair, slot, arm); anchor margins per context.
    grid_lnp: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    anchor_lnp: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        if "block_key" in r:
            grid_lnp[(r["pair_id"], r["slot"], r["arm"], r["pool_side"])].append(r["lnp_mean"])
        else:
            anchor_lnp[(r["context_id"], r["pool_key"], r["pool_side"])].append(r["lnp_mean"])

    def _margin(d: dict, key_b, key_a) -> float | None:
        b, a = d.get(key_b), d.get(key_a)
        if not b or not a:
            return None
        return sum(b) / len(b) - sum(a) / len(a)

    out = []
    for (pair_id, slot, arm), _ in {(k[0], k[1], k[2]): None for k in grid_lnp}.items():
        m_patched = _margin(grid_lnp, (pair_id, slot, arm, "B"), (pair_id, slot, arm, "A"))
        p = pairs_by_id[pair_id]
        pk = R.pool_key(p)
        m_floor = _margin(anchor_lnp, (p.a, pk, "B"), (p.a, pk, "A"))
        out.append(
            {
                "pair_id": pair_id,
                "cell": p.cell,
                "slot": slot,
                "arm": arm,
                "margin_patched": m_patched,
                "margin_floor_anchor": m_floor,
                "margin_shift": (
                    m_patched - m_floor if m_patched is not None and m_floor is not None else None
                ),
            }
        )
    _write_jsonl_atomic(args.out_dir / "margin_cells.jsonl", out)

    # Rule-19 validation: Spearman rho(margin shift, F_beh) across steered
    # cells with dynamic range — reported BEFORE the margin carries any read.
    from scipy.stats import spearmanr

    f_by_key = {
        (r["pair_id"], r["slot"]): r["f_beh"]
        for r in _iter_jsonl(args.out_dir / "f_cells.jsonl")
        if r["f_beh"] is not None
    }
    xs, ys = [], []
    for r in out:
        if r["arm"] != "steered" or r["margin_shift"] is None:
            continue
        f = f_by_key.get((r["pair_id"], r["slot"]))
        if f is not None:
            xs.append(r["margin_shift"])
            ys.append(f)
    rho, pval = spearmanr(xs, ys) if len(xs) >= 10 else (float("nan"), float("nan"))
    _write_json_atomic(
        args.out_dir / "margin_validation.json",
        {
            "rho_margin_fbeh": None if math.isnan(float(rho)) else float(rho),
            "p": None if math.isnan(float(pval)) else float(pval),
            "n": len(xs),
            "validated": bool(len(xs) >= 10 and float(rho) > 0),
            "note": "rule 19: the margin carries NO cross-condition read unless rho > 0",
        },
    )
    logger.info("[margin] cells=%d validation rho=%s n=%d", len(out), rho, len(xs))


# ── step: probe (kernelized batched-torch logistic; LOCO carrier folds) ──


def kernel_logistic_auc(
    gram: torch.Tensor,  # (L, n, n) standardized-feature linear-kernel Gram
    labels: torch.Tensor,  # (P, n) 0/1 — row 0 = observed, rows 1.. = permutations
    fold_masks: torch.Tensor,  # (F, n) True = HELD OUT (partition of n)
    epochs: int = 150,
    lr: float = 0.15,
    l2: float = 1e-2,
) -> torch.Tensor:
    """Pooled held-out AUC per (label-row, layer) — one batched GD fit.

    Kernel trick: with GD from 0 + L2, the weight lives in the span of the
    TRAIN rows, so we optimize dual coefficients ``a`` (P, L, F, n) against
    the per-layer Gram — O(n^2) per step instead of O(n*H) (H=3584, n=24:
    ~150x cheaper), which is what makes the B=1,000 permutation battery a
    single vectorized fit (vectorize-many-cell-fits). L2 is the primal
    ||w||^2 = a^T G a, added explicitly (optimizer weight_decay on ``a``
    would regularize the wrong norm). AUC is computed rank-vectorized over
    (P, L) — no per-perm python loop.
    """
    n_perm, n = labels.shape
    n_layers = gram.shape[0]
    n_folds = fold_masks.shape[0]
    assert gram.shape == (n_layers, n, n), gram.shape
    covered = fold_masks.any(dim=0)
    # LOCO: the folds PARTITION n (every row held out exactly once). Transfer:
    # a single non-covering holdout is legal — AUC is then scored over the
    # held rows only (never over never-held rows, whose held-score is 0).
    assert n_folds == 1 or bool(covered.all()), "multi-fold masks must cover every row"
    train = (~fold_masks).float()  # (F, n)
    a = torch.zeros(n_perm, n_layers, n_folds, n, requires_grad=True)
    b = torch.zeros(n_perm, n_layers, n_folds, requires_grad=True)
    opt = torch.optim.Adam([a, b], lr=lr)
    yb = labels.float()[:, None, None, :]  # (P,1,1,n)
    tm = train[None, None, :, :]  # (1,1,F,n)
    for _ in range(epochs):
        opt.zero_grad()
        am = a * tm  # support restricted to TRAIN rows per fold
        logits = torch.einsum("plfn,lnm->plfm", am, gram) + b.unsqueeze(-1)
        bce = torch.nn.functional.binary_cross_entropy_with_logits(
            logits, yb.expand_as(logits), reduction="none"
        )
        penalty = (am * (logits - b.unsqueeze(-1))).sum(dim=-1).mean()  # a^T G a
        loss = (bce * tm).sum() / tm.sum() / n_perm + l2 * penalty
        loss.backward()
        opt.step()
    with torch.no_grad():
        am = a * tm
        logits = torch.einsum("plfn,lnm->plfm", am, gram) + b.unsqueeze(-1)
        held = (logits * fold_masks.float()[None, None]).sum(dim=2)  # (P, L, n)
        return _auc_ranked(held[:, :, covered], labels[:, covered])


def _auc_ranked(scores: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Mann-Whitney AUC over the last axis, vectorized over leading axes.

    scores: (P, L, n); labels: (P, n) 0/1 -> (P, L). Plain ranks (double
    argsort; continuous GD logits make exact ties measure-zero).
    """
    n_perm, n_layers, n = scores.shape
    order = scores.argsort(dim=-1)
    ranks = torch.empty_like(scores)
    ranks.scatter_(
        -1, order, torch.arange(1, n + 1, dtype=scores.dtype).expand(n_perm, n_layers, n)
    )
    lab = labels.float()[:, None, :]  # (P,1,n)
    n_pos = lab.sum(dim=-1)  # (P,1)
    n_neg = n - n_pos
    assert bool((n_pos > 0).all() and (n_neg > 0).all()), "degenerate label split"
    pos_rank_sum = (ranks * lab).sum(dim=-1)  # (P, L)
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _vp_data(
    recs: dict, layers: list[int], cell: str, slot: str, va: str, vb: str, carriers: list[str]
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """(gram (L,n,n), y (n,), groups (n,)) for one value-pair, LOCO grouping."""
    ids = [
        (BANK.context_id(cell, v, carrier), int(v == vb), gi)
        for gi, carrier in enumerate(carriers)
        for v in (va, vb)
    ]
    key = "v_pe" if slot == "pe" else "v_ce"
    x = torch.stack([recs[cid][key] for cid, _, _ in ids]).float()  # (n, L, H)
    assert x.shape[:2] == (len(ids), len(layers)), x.shape
    x = x.transpose(0, 1)  # (L, n, H)
    mu = x.mean(dim=1, keepdim=True)
    sd = x.std(dim=1, keepdim=True).clamp_min(1e-6)
    xs = (x - mu) / sd  # label-independent standardization (probe input scaling)
    gram = torch.einsum("lnh,lmh->lnm", xs, xs) / xs.shape[-1]
    y = torch.tensor([lab for _, lab, _ in ids])
    groups = torch.tensor([g for _, _, g in ids])
    return gram, y, groups


def step_probe(args: argparse.Namespace) -> None:
    bank = torch.load(args.bank_pt, map_location="cpu", weights_only=False)
    recs = bank["per_context"]
    layers = list(bank["layers"])
    torch.manual_seed(PROBE_SEED)
    gen = torch.Generator().manual_seed(PROBE_SEED)
    chunk = args.perm_chunk

    results: list[dict] = []
    perm_store: dict[str, np.ndarray] = {}
    for cell in BANK.all_cells():
        carriers = list(BANK.carriers_for(cell))
        vps = BANK.cell_pairs_per_carrier(cell)
        for slot in ("ce", "pe"):
            per_vp_obs: list[np.ndarray] = []  # each (L,)
            per_vp_perm: list[np.ndarray] = []  # each (B, L)
            transfer_aucs: list[float] = []
            for va, vb in vps:
                gram, y, groups = _vp_data(recs, layers, cell, slot, va, vb, carriers)
                n = y.shape[0]
                fold_masks = torch.stack([groups == g for g in range(len(carriers))])
                # Within-carrier label flips: swap each carrier's (va, vb) pair
                # independently per draw (the §6 selection-symmetric scheme).
                flips = torch.randint(
                    0, 2, (args.perm_b, len(carriers)), generator=gen
                ).bool()  # (B, G)
                flip_rows = flips[:, groups]  # (B, n)
                perm_labels = torch.where(flip_rows, 1 - y.unsqueeze(0), y.unsqueeze(0))
                all_labels = torch.cat([y.unsqueeze(0), perm_labels], dim=0)  # (1+B, n)
                aucs = torch.empty(all_labels.shape[0], len(layers))
                for s in range(0, all_labels.shape[0], chunk):
                    aucs[s : s + chunk] = kernel_logistic_auc(
                        gram, all_labels[s : s + chunk], fold_masks
                    )
                per_vp_obs.append(aucs[0].numpy())
                per_vp_perm.append(aucs[1:].numpy())
                assert per_vp_perm[-1].shape == (args.perm_b, len(layers))
            # Secondary: value-pair transfer (adjacent cycle pairs share a
            # value; the shared value anchors the class-1 sign).
            for i, (va, vb) in enumerate(vps):
                for j, (vc, vd) in enumerate(vps):
                    if i == j or vb != vc:
                        continue  # train (va,vb) -> test (vb,vd): shared vb
                    g_tr, y_tr, _ = _vp_data(recs, layers, cell, slot, va, vb, carriers)
                    # Train on ALL rows of the train pair (single fold), score
                    # the TEST pair's rows through the shared standardized
                    # feature space: rebuild a joint gram over train+test rows.
                    ids_tr = [
                        (BANK.context_id(cell, v, c), int(v == vb))
                        for c in carriers
                        for v in (va, vb)
                    ]
                    ids_te = [
                        (BANK.context_id(cell, v, c), int(v == vb))  # class1 = shared vb
                        for c in carriers
                        for v in (vd, vb)
                    ]
                    key = "v_pe" if slot == "pe" else "v_ce"
                    x_all = (
                        torch.stack([recs[cid][key] for cid, _ in ids_tr + ids_te])
                        .float()
                        .transpose(0, 1)
                    )  # (L, n_tr+n_te, H)
                    mu = x_all.mean(dim=1, keepdim=True)
                    sd = x_all.std(dim=1, keepdim=True).clamp_min(1e-6)
                    xs = (x_all - mu) / sd
                    gram_all = torch.einsum("lnh,lmh->lnm", xs, xs) / xs.shape[-1]
                    n_tr = len(ids_tr)
                    y_all = torch.tensor([lab for _, lab in ids_tr + ids_te])
                    hold = torch.zeros(1, y_all.shape[0], dtype=torch.bool)
                    hold[0, n_tr:] = True  # single fold: test rows held out
                    auc_t = kernel_logistic_auc(
                        gram_all, y_all.unsqueeze(0), hold, epochs=150
                    )  # (1, L)
                    transfer_aucs.append(float(auc_t[0].max()))
            obs_layer = np.mean(per_vp_obs, axis=0)  # (L,)
            perm_layer = np.mean(per_vp_perm, axis=0)  # (B, L) mean over vps per draw
            perm_max = perm_layer.max(axis=1)  # per-draw re-max over layers
            max_auc = float(obs_layer.max())
            band = float(np.percentile(perm_max, 97.5))
            results.append(
                {
                    "cell": cell,
                    "slot": slot,
                    "auc_per_layer": obs_layer.tolist(),
                    # Per-value-pair curves (manifest probe_layer_curves points).
                    "auc_per_layer_per_vp": [a.tolist() for a in per_vp_obs],
                    "value_pairs": [list(vp) for vp in vps],
                    "max_auc_over_layers": max_auc,
                    "best_layer": layers[int(obs_layer.argmax())],
                    "perm_band_97p5": band,
                    "probe_positive": bool(max_auc > band),
                    "transfer_max_auc": (float(np.mean(transfer_aucs)) if transfer_aucs else None),
                    "n_per_value_pair": 2 * len(carriers),
                    "n_value_pairs": len(vps),
                    "folds": f"leave-one-carrier-out ({len(carriers)} groups)",
                }
            )
            perm_store[f"{cell}|{slot}"] = perm_layer
            logger.info(
                "[probe] unit %s|%s max_auc=%.3f band=%.3f positive=%s",
                cell,
                slot,
                max_auc,
                band,
                max_auc > band,
            )
    _write_json_atomic(
        args.out_dir / "probe.json",
        {
            "results": results,
            "seed": PROBE_SEED,
            "perm_b": args.perm_b,
            "probe": "kernelized L2 logistic (linear kernel), 150 epochs Adam lr=0.15 l2=1e-2",
        },
    )
    pdir = args.out_dir / "probe_perm_matrix"
    pdir.mkdir(parents=True, exist_ok=True)
    np.savez(pdir / "perm_auc_matrix.npz", **perm_store)
    logger.info("[probe] %d (cell x slot) units", len(results))


# ── step: two-by-two ──────────────────────────────────────────────────


def step_two_by_two(args: argparse.Namespace) -> None:
    stats = json.loads((args.out_dir / "stats.json").read_text())["per_cell"]
    probe = {
        (r["cell"], r["slot"]): r
        for r in json.loads((args.out_dir / "probe.json").read_text())["results"]
    }
    out = []
    for key, rec in sorted(stats.items()):
        pr = probe.get((rec["cell"], rec["slot"]))
        causal = (
            "untestable-causal"
            if rec["untestable_causal"]
            else ("positive" if rec.get("holm_pass") and rec.get("disjoint_both_nulls") else "null")
        )
        out.append(
            {
                "cell": rec["cell"],
                "slot": rec["slot"],
                "causal_verdict": causal,
                "probe_verdict": (
                    "positive" if pr and pr["probe_positive"] else "null" if pr else "missing"
                ),
                "f_steered_mean": rec.get("f_steered_mean"),
                "max_auc": pr["max_auc_over_layers"] if pr else None,
                "n_post_exclusion": rec["n_post_exclusion"],
            }
        )
    _write_json_atomic(args.out_dir / "two_by_two.json", {"cells": out})
    n_unt = sum(1 for r in out if r["causal_verdict"] == "untestable-causal")
    logger.info("[2x2] %d cells (%d untestable-causal)", len(out), n_unt)


# ── step: stage2 tables ───────────────────────────────────────────────


def step_stage2(args: argparse.Namespace) -> None:
    """Reduce stage-2 rollouts + judge scores to per-(pair x block) F_beh rows.

    Writes ``stage2_cells.jsonl`` — the manifest ``layer_profile_stage2`` /
    ``layer_profile_stage2_perpair`` source: one row per
    (pair x slot x arm x layer x dose) stage-2 block membership, with the
    same dual-rubric contrast -> floor/ceiling-normalized F_beh reduction as
    ``step_f_tables`` (anchor pools shared with stage 1).
    """
    if args.stage2_dir is None or not args.stage2_dir.is_dir():
        if args.step == "all":
            logger.info("[stage2] skipped — no --stage2-dir (stage-2 not yet run)")
            return
        raise AssertionError("--step stage2 requires --stage2-dir with shard_*.jsonl")
    pairs = BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    s2_rows = J.load_stage2_rows(args.stage2_dir)
    anchor_rows = J.load_anchor_rows(args.anchors_dir)
    # The .stage2 suffix files carry BOTH coherence.stage2 and {rid}.stage2
    # waves (distinct item-id tags "c" vs "s"), exactly like the grid suffix.
    beh_s2 = load_wave_scores(args.scores_dir, "stage2")
    beh_anchor = load_wave_scores(args.scores_dir, "anchors")

    anchor_cache: dict[str, tuple[list[float], list[float]]] = {}

    def _floor_ceiling(p: BANK.Pair2162) -> tuple[list[float], list[float]]:
        if p.pair_id not in anchor_cache:
            anchor_cache[p.pair_id] = (
                _anchor_deltas(anchor_rows, beh_anchor, beh_anchor, p, p.a),
                _anchor_deltas(anchor_rows, beh_anchor, beh_anchor, p, p.b),
            )
        return anchor_cache[p.pair_id]

    by_block_pair: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in s2_rows:
        by_block_pair[(row["block_key"], row["pair_id"])].append(row)

    out: list[dict] = []
    for (block_key, pair_id), rows in sorted(by_block_pair.items()):
        p = pairs_by_id[pair_id]
        if J.pair_rubric_cores(p) is None:
            continue
        deltas: list[float] = []
        n_coherent = 0
        n_cap = 0
        for row in rows:
            n_cap += int(row.get("cap_hit", False))
            coh = beh_s2.get(J.J94._item_id("c", f"s|{block_key}|{pair_id}|{row['draw']}"))
            if coh is None or coh <= COHERENCE_THRESHOLD:
                continue
            n_coherent += 1
            sa = _grid_behavior_score(beh_s2, "s", block_key, pair_id, row["draw"], "a")
            sb = _grid_behavior_score(beh_s2, "s", block_key, pair_id, row["draw"], "b")
            if sa is None or sb is None:
                continue
            deltas.append((sb - sa) / 100.0)
        d_floor, d_ceiling = _floor_ceiling(p)
        f_beh = None
        sep = None
        if d_floor and d_ceiling:
            df = sum(d_floor) / len(d_floor)
            dc = sum(d_ceiling) / len(d_ceiling)
            sep = dc - df
            if deltas and abs(dc - df) > 1e-9:
                f_beh = (sum(deltas) / len(deltas) - df) / (dc - df)
        r0 = rows[0]
        out.append(
            {
                "block_key": block_key,
                "pair_id": pair_id,
                "cell": r0["cell"],
                "slot": r0["slot"],
                "arm": r0["arm"],
                "layer": r0["layer"],
                "dose": r0["dose"],
                "mode": r0.get("mode", "add"),
                "carrier": p.carrier,
                "value_a": p.value_a,
                "value_b": p.value_b,
                "donor_pair_id": r0.get("donor_pair_id"),
                "n_draws": len(rows),
                "n_coherent": n_coherent,
                "n_scored": len(deltas),
                "n_cap_hit": n_cap,
                "delta_patched_mean": sum(deltas) / len(deltas) if deltas else None,
                "f_beh": f_beh,
                "separation": sep,
                "family": family_of(r0["cell"], r0["slot"]),
                "len_delta": r0.get("len_delta"),
            }
        )
    assert out, "stage2 reduce produced 0 rows"
    _write_jsonl_atomic(args.out_dir / "stage2_cells.jsonl", out)
    logger.info("[stage2] rows=%d blocks=%d", len(out), len({r["block_key"] for r in out}))


# ── CLI ───────────────────────────────────────────────────────────────

STEPS = {
    "f-tables": step_f_tables,
    "stats": step_stats,
    "margin": step_margin,
    "probe": step_probe,
    "two-by-two": step_two_by_two,
    "stage2": step_stage2,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Issue #2162 P7 analysis driver.")
    ap.add_argument("--step", required=True, choices=(*STEPS, "all"))
    ap.add_argument("--rollouts-dir", type=Path, required=False)
    ap.add_argument("--anchors-dir", type=Path, required=False)
    ap.add_argument("--va-dir", type=Path, required=False)
    ap.add_argument("--margin-dir", type=Path, required=False)
    ap.add_argument("--stage2-dir", type=Path, required=False)
    ap.add_argument("--bank-pt", type=Path, required=False)
    ap.add_argument("--scores-dir", type=Path, default=Path("eval_results/issue_2162/judge/scores"))
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_2162/f_metrics"))
    ap.add_argument("--perm-b", type=int, default=PROBE_PERM_B)
    ap.add_argument("--perm-chunk", type=int, default=64)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stdout,
    )
    args = parse_args(argv)
    steps = list(STEPS) if args.step == "all" else [args.step]
    for step in steps:
        logger.info("[step=%s]", step)
        STEPS[step](args)
        logger.info("[step=%s_done]", step)
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
