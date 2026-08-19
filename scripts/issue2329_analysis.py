#!/usr/bin/env python3
"""Issue #2329 — P7 analysis driver (thin fork of ``issue2162_analysis.py``).

Consumes the pod artifacts (rollout shards + 32-layer V_a stores + vc_bank +
margin shards) and the judge outputs (per-wave ``*.scores.jsonl``) and
produces the plan §6 statistical outputs. Issue-2329 divergences from the
parent analysis:

- **32-layer model constants:** F_act read layer 26 -> **30** (divergence 4,
  fraction-matched 26/27 -> round(0.963 x 31); H = 4096); V_a is captured at
  ALL 32 layers so the read-layer choice stays analysis-time-recomputable.
- **Surviving pairs (divergence 9):** every step derives its pair set from
  the FROZEN gate-0a ``bank.json`` (``issue2329_judge.surviving_pairs``),
  never the raw 1,404-pair build. Realized drops at freeze: 0.
- **Analysis-time Holm m (plan §6, remedy A — UNMODIFIED semantics):**
  ``holm_family_m = len(pvals)`` over TESTABLE cells (post-exclusion n >= 12)
  per family, exactly the inherited code; the realized per-family m is
  reported beside the constructional CEILINGS (P1=31 / P2=15 / P3=28) and the
  parent's realized m (25/10/26) everywhere verdicts render (stats.json
  ``family_m``, two_by_two.json ``families``, transfer.json).
- **pe-exclusion manifests consumed (divergences 1/9):** the pod driver's
  ``manifests/pe_exclusions.json`` (no-prefix contexts under the thinking-off
  template have NO pe slot) is loaded at ``--step stats``; a pe-excluded cell
  exits its family and shrinks m as a NAMED exclusion
  (``pe_excluded_units`` + per-cell ``n_pe_excluded_pairs``), never a silent
  zero. The read probe skips (cell x pe) units containing no-prefix contexts
  with a named record (a zero v_pe vector is not a state read).
- **NEW ``--step transfer`` (plan §6 prediction 8 — ONE registered test
  outside the three families):** Spearman rho(F_beh_2329, F_beh_2162) over
  shared P1 cells at cell grain, pair-clustered bootstrap 95% CI (B=10,000,
  seed 21620), against the parent's COMMITTED ``eval_results/issue_2162/
  f_metrics`` tables (count-asserted at load: 31 P1 units), plus the
  verdict-transfer 2x2 comparison with BOTH runs' realized per-family m
  beside it and divergence-9 dropped/repaired cells flagged.
- **Stage-2 selection note:** the realized stage-2 selection is POD-SIDE
  F_act (divergence 6, ``fact_select_report.json``); ``best_cells.json`` here
  is the F_beh-selection counterpart, kept as a comparison artifact only.

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

import issue2329_judge as J  # noqa: E402  (rubric ids + loaders + anchor_unit_id)
import issue2329_run as R  # noqa: E402  (pool_key + HF prefix + upload seam)
from issue2094_analysis import bootstrap_family_means_batched  # noqa: E402
from explore_persona_space.experiments.issue2094 import fmetrics as FM  # noqa: E402
from explore_persona_space.experiments.issue2162 import bank2162 as BANK  # noqa: E402

logger = logging.getLogger("issue2329.analysis")

READ_LAYER = 30  # plan divergence 4: fraction-matched remap of the parent's 26/28
COHERENCE_THRESHOLD = 60.0
SEPARATION_BAR = 0.5
SURVIVAL_FLOOR = 12  # exact signed-rank attainability + pre-registered floor
BOOT_B = 10_000
BOOT_SEED = 21620
PROBE_SEED = 21621
PROBE_PERM_B = 1_000
STAGE2_CAP = 12
HOLM_ALPHA = 0.05
# Plan §4.1 length mitigation: the length-matched sensitivity subset.
LEN_MATCH_MAX_ABS = 2

ROUTE_VARIANT_TYPES = ("demo_format", "demo_persona", "language_implied", "persona_role_header")

# Plan §6 (S1, remedy A): analysis-time m is what the code executes; the
# constructional CEILINGS + the parent's realized m are REPORTED beside it
# everywhere verdicts render.
FAMILY_CEILING_M = {"P1": 31, "P2": 15, "P3": 28}
PARENT_REALIZED_M = {"P1": 25, "P2": 10, "P3": 26}
# Transfer read (plan §2/§6): parent committed table count-assert at load.
PARENT_P1_UNIT_COUNT = 31


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


def _pairs(args: argparse.Namespace) -> list[BANK.Pair2162]:
    """Surviving pairs from the frozen gate-0a bank.json (divergence 9)."""
    return J.surviving_pairs(args.bank_json)


def load_pe_exclusions(path: Path) -> dict:
    """The pod driver's ``manifests/pe_exclusions.json`` (fail-loud: the grid
    phase always writes it — an absent manifest means the staging is
    incomplete, never that there were no exclusions)."""
    assert path.is_file(), (
        f"{path} missing — stage the pod's manifests/pe_exclusions.json (the named "
        "no-prefix pe-slot exclusion registry; plan divergences 1/9)"
    )
    return json.loads(path.read_text())


# ── step: f-tables ────────────────────────────────────────────────────


def _grid_behavior_score(
    scores: dict[str, float | None], tag: str, block_key: str, pair_id: str, draw: int, side: str
) -> float | None:
    return scores.get(J.J94._item_id(tag, f"{tag}|{block_key}|{pair_id}|{draw}|{side}"))


TINY_STORE_MAX_LAYERS = 8  # --tiny shrunk captures (default --tiny-layers 4)


def _read_layer_index(layers: list[int], shard: Path, registry: dict) -> int:
    """READ_LAYER shard index — fail-loud on full-depth stores missing it (r2 F6).

    Production stores capture ALL 32 layers, so READ_LAYER (30) is always
    present; tiny/smoke stores (<= TINY_STORE_MAX_LAYERS captured layers, the
    ``--tiny`` shrunk-config convention) keep the last-layer fallback. Any
    OTHER store missing READ_LAYER — canonically a stale 28-layer Qwen2.5
    parent-model store — RAISES instead of silently reading the wrong layer
    while downstream outputs still describe the read as layer 30. Also pins
    ONE consistent layer registry across a directory's shards (``registry``
    carries the first shard's layers + path).
    """
    layers = list(layers)
    if "layers" in registry:
        assert registry["layers"] == layers, (
            f"inconsistent layer registry across shards: {shard} has {layers} vs "
            f"{registry['first_shard']}'s {registry['layers']}"
        )
    else:
        registry["layers"] = layers
        registry["first_shard"] = str(shard)
    if READ_LAYER in layers:
        return layers.index(READ_LAYER)
    assert len(layers) <= TINY_STORE_MAX_LAYERS, (
        f"{shard}: {len(layers)}-layer store lacks READ_LAYER={READ_LAYER} and is not a "
        f"tiny capture (<= {TINY_STORE_MAX_LAYERS} layers) — stale prior-model shard?"
    )
    return len(layers) - 1


def _load_va_store(va_dir: Path) -> dict[tuple[str, str, str, int], torch.Tensor]:
    """(block_key, pair_id, context_a, draw) -> read-layer (30) span-mean V_a (H,).

    Rows listed in the shard's ``empty_rows`` (empty completion => zero-vector
    ``va_span``) are EXCLUDED — a zero vector is not a state read.
    """
    out: dict[tuple[str, str, str, int], torch.Tensor] = {}
    n_empty = 0
    registry: dict = {}
    for shard in sorted(va_dir.glob("shard_*.pt")):
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        li = _read_layer_index(payload["layers"], shard, registry)
        va = payload["va_span"]
        empty = set(payload.get("empty_rows", []))
        n_empty += len(empty)
        for j, meta in enumerate(payload["index"]):
            if j in empty:
                continue
            out[(payload["block_key"], meta["pair_id"], meta["context_a"], meta["draw"])] = va[
                j, li
            ].float()
    assert out, f"no V_a shards under {va_dir}"
    if n_empty:
        logger.info("[va-store] excluded %d empty-completion zero-vector rows", n_empty)
    return out


def _load_anchor_va(anchors_dir: Path) -> dict[tuple[str, int], torch.Tensor]:
    """(context_id, draw) -> read-layer (30) span-mean anchor V_a (H,).

    ``empty_rows`` excluded — same zero-vector rationale as ``_load_va_store``.
    """
    out: dict[tuple[str, int], torch.Tensor] = {}
    n_empty = 0
    registry: dict = {}
    for shard in sorted(anchors_dir.glob("va_anchors_*.pt")):
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        li = _read_layer_index(payload["layers"], shard, registry)
        va = payload["va_span"]
        empty = set(payload.get("empty_rows", []))
        n_empty += len(empty)
        for j, meta in enumerate(payload["index"]):
            if j in empty:
                continue
            key = (meta["context_id"], meta["draw"])
            assert key not in out, (
                f"duplicate anchor V_a row {key} in {shard} — stale prior-width "
                "va_anchors shard? (the run driver quarantines these at phase entry)"
            )
            out[key] = va[j, li].float()
    assert out, f"no anchor V_a shards under {anchors_dir}"
    if n_empty:
        logger.info("[anchor-va] excluded %d empty-completion zero-vector rows", n_empty)
    return out


def _anchor_deltas(
    anchor_rows: list[dict],
    scores: dict[str, float | None],
    coherence: dict[str, float | None],
    pair: BANK.Pair2162,
    ctx: str,
    exclude_cap_hit: bool = False,
) -> list[float]:
    """Coherent kept per-draw dual-rubric contrasts for one anchor context.

    ``exclude_cap_hit`` drops draws whose completion hit the generation cap
    (the v55 restriction / truncation-sensitivity read, #2329). It is
    RESTRICTIVE ONLY -- strictly fewer draws contribute; no draw is newly
    admitted and no threshold is relaxed -- so a restricted estimate is
    directly comparable to the shipped one.
    """
    cores = J.pair_rubric_cores(pair)
    assert cores is not None
    rid_a, rid_b = (J.rubric_core_id(c) for c in cores)
    deltas: list[float] = []
    for row in anchor_rows:
        if row["context_id"] != ctx:
            continue
        if exclude_cap_hit and row.get("cap_hit", False):
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
    pairs = _pairs(args)
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
    # incoherence term of the coherence_caphit figure (excess incoherence =
    # arm incoherent fraction MINUS this baseline rate).
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
        d_floor = _anchor_deltas(anchor_rows, beh_anchor, coherence, p, p.a, args.exclude_cap_hit)
        d_ceiling = _anchor_deltas(anchor_rows, beh_anchor, coherence, p, p.b, args.exclude_cap_hit)
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
            if args.exclude_cap_hit and row.get("cap_hit", False):
                continue  # v55 restriction read: drop cap-hit draws (counted above)
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
        d_floor = _anchor_deltas(anchor_rows, beh_anchor, coherence, p, p.a, args.exclude_cap_hit)
        d_ceiling = _anchor_deltas(anchor_rows, beh_anchor, coherence, p, p.b, args.exclude_cap_hit)
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
            # Plan §4.1 length covariate: per-pair token-length delta.
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
    """Holm step-down adjusted p-values within one family (analysis-time m =
    len(pvals) over TESTABLE cells — plan §6 remedy A, the inherited
    semantics; the constructional ceilings are reported beside it)."""
    items = sorted(pvals.items(), key=lambda kv: kv[1])
    m = len(items)
    adj: dict[str, float] = {}
    running = 0.0
    for i, (key, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * p))
        adj[key] = running
    return adj


def _family_m_report(family_p: dict[str, dict[str, float]]) -> dict:
    """Realized analysis-time m per family beside the ceilings + parent m
    (plan §6: 'realized per-family m is reported at P7 beside the parent's
    25/10/26' — rendered everywhere verdicts render)."""
    return {
        fam: {
            "realized_m": len(family_p.get(fam, {})),
            "constructional_ceiling": FAMILY_CEILING_M[fam],
            "parent_realized_m": PARENT_REALIZED_M[fam],
        }
        for fam in FAMILY_CEILING_M
    }


def step_stats(args: argparse.Namespace) -> None:
    steered = list(_iter_jsonl(args.out_dir / "f_cells.jsonl"))
    nulls = {
        "shuffled": list(_iter_jsonl(args.out_dir / "null_shuffled_cells.jsonl")),
        "crosstype": list(_iter_jsonl(args.out_dir / "null_crosstype_cells.jsonl")),
    }
    # Divergences 1/9: the pod's named pe-slot exclusions (no-prefix contexts
    # under the thinking-off template). A pe-excluded cell exits its family
    # and shrinks m as a NAMED exclusion — never a silent zero.
    pe_manifest = load_pe_exclusions(args.pe_exclusions)
    pe_pair_excl: dict[tuple[str, str], int] = defaultdict(int)
    pe_empty_units: list[dict] = []
    for e in pe_manifest["exclusions"]:
        if e["arm"] != "steered":
            continue  # unit grain below is the steered arm; null arms mirror it
        if e["pair_id"] is None:
            pe_empty_units.append({"cell": e["cell"], "slot": e["slot"], "reason": e["reason"]})
        else:
            pe_pair_excl[(e["cell"], e["slot"])] += 1

    def index(rows: list[dict]) -> dict[tuple[str, str, str], dict]:
        return {(r["pair_id"], r["slot"], r["arm"]): r for r in rows}

    idx_null = {k: index(v) for k, v in nulls.items()}
    cells: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in steered:
        cells[(r["cell"], r["slot"])].append(r)

    # Plan §4.5: cells below 50% coherent are MARKED explicitly.
    # Pooled over all three arms' rollouts for the (cell x slot) unit.
    coh_counts: dict[tuple[str, str], list[int]] = defaultdict(lambda: [0, 0])
    for rows_list in (steered, nulls["shuffled"], nulls["crosstype"]):
        for r in rows_list:
            c = coh_counts[(r["cell"], r["slot"])]
            c[0] += r["n_coherent"]
            c[1] += r["n_draws"]

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
            # Plan §4.1 length-matched sensitivity subset.
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
            # Divergences 1/9: named pe-slot pair exclusions for this unit
            # (no-prefix contexts — pairs skipped-with-record pod-side).
            "n_pe_excluded_pairs": pe_pair_excl.get((cell, slot), 0),
            "untestable_causal": not testable,
            "f_steered_mean": float(np.mean(f_steered)) if f_steered else None,
            "f_shuffled_mean": float(np.mean(f_null["shuffled"])) if f_null["shuffled"] else None,
            "f_crosstype_mean": (
                float(np.mean(f_null["crosstype"])) if f_null["crosstype"] else None
            ),
            "p_iut": p_iut,
            "realized_mde_single_test": (1.02 / math.sqrt(n)) if n else None,
            # Plan §4.5 coherence mark: pooled over all three arms.
            "coherent_fraction": (
                coh_counts[(cell, slot)][0] / coh_counts[(cell, slot)][1]
                if coh_counts[(cell, slot)][1]
                else None
            ),
            "low_coherence": bool(
                coh_counts[(cell, slot)][1]
                and coh_counts[(cell, slot)][0] / coh_counts[(cell, slot)][1] < 0.5
            ),
            # Plan §4.1 length-matched sensitivity recount: the same
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

    # NAMED record for (cell x pe) units FULLY absent from the grid because
    # every pair was pe-excluded (block_empty_all_pairs_no_prefix): they never
    # enter `cells`, so m shrinks via len(pvals) — the record below is what
    # keeps that shrinkage a named exclusion rather than a silent zero.
    pe_excluded_units = [
        {
            **u,
            "family": family_of(u["cell"], u["slot"]),
            "note": "unit absent from the grid — every pair pe-excluded (no-prefix); "
            "exits its family and shrinks analysis-time m (named exclusion)",
        }
        for u in pe_empty_units
    ]

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
            # Plan §6 remedy A (UNMODIFIED inherited semantics): analysis-time
            # m = the realized number of TESTABLE cells in the family.
            per_cell[key]["holm_family_m"] = len(pvals)
            per_cell[key]["holm_pass"] = p_adj < HOLM_ALPHA

    # Registered P3 secondary (recency_load_curves transform): the per-pair
    # depth/load slope with a pair-clustered bootstrap 95% CI. A pair is
    # traced across levels by (carrier, value_a, value_b) within one base
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
            "descending steered F_beh — COMPARISON ARTIFACT ONLY under #2329: the realized "
            "stage-2 selection was POD-SIDE on F_act (divergence 6; fact_select_report.json)",
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
            "family_m": _family_m_report(family_p),
            "pe_excluded_units": pe_excluded_units,
            "bars": {
                "separation_bar": SEPARATION_BAR,
                "survival_floor": SURVIVAL_FLOOR,
                "boot": {"B": BOOT_B, "seed": BOOT_SEED},
                "holm_alpha": HOLM_ALPHA,
                "holm_m_semantics": "analysis-time m = len(pvals) over testable cells "
                "(plan §6 remedy A — inherited code semantics; ceilings 31/15/28 and the "
                "parent's realized 25/10/26 reported in family_m)",
                "joint_power_note": "realized MDE reported for the registered CONJUNCTION "
                "(Holm-IUT AND disjoint CIs) — plausibly ~0.23-0.25 at n=27, never the "
                "single-test 1.02/sqrt(n) line alone (plan §6)",
            },
        },
    )
    n_testable = sum(1 for r in per_cell.values() if not r["untestable_causal"])
    logger.info(
        "[stats] cells=%d testable=%d survivors=%d (families: %s; ceilings %s; parent %s; "
        "pe-excluded units=%d)",
        len(per_cell),
        n_testable,
        len(survivors),
        {f: len(p) for f, p in family_p.items()},
        FAMILY_CEILING_M,
        PARENT_REALIZED_M,
        len(pe_excluded_units),
    )


# ── step: margin ──────────────────────────────────────────────────────


def step_margin(args: argparse.Namespace) -> None:
    pairs_by_id = {p.pair_id: p for p in _pairs(args)}
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

    f_by_key = {
        (r["pair_id"], r["slot"]): r["f_beh"]
        for r in _iter_jsonl(args.out_dir / "f_cells.jsonl")
        if r["f_beh"] is not None
    }
    validation = rule19_validation(out, f_by_key)
    _write_json_atomic(args.out_dir / "margin_validation.json", validation)
    logger.info(
        "[margin] cells=%d validation rho_percell=%s (n_cells=%d) rho_perpair=%s (n_pairs=%d)",
        len(out),
        validation["rho_margin_fbeh_percell"],
        validation["n_cells"],
        validation["rho_margin_fbeh_perpair"],
        validation["n_pairs"],
    )


RULE19_MIN_N = 10
RULE19_DYNAMIC_RANGE_SCREEN = (
    "a (cell x slot) unit enters the per-cell rho iff it has >=2 steered pairs with both "
    "margin_shift and F_beh present AND nonzero spread (max > min) in BOTH quantities "
    "across those pairs (a constant/degenerate unit carries no dynamic range)"
)


def rule19_validation(margin_rows: list[dict], f_by_key: dict[tuple[str, str], float]) -> dict:
    """Rule-19 validation at BOTH grains.

    The REGISTERED grain (plan §4.4): Spearman rho(margin shift, F_beh)
    ACROSS (cell x slot) units — per-unit means over steered pairs —
    restricted to units passing the dynamic-range screen (declared in
    ``dynamic_range_screen``). ``validated`` keys on this grain. The per-pair
    rho is kept as the low-level companion. Reported BEFORE the margin
    carries any read; never the headline.
    """
    from scipy.stats import spearmanr

    def _stat(xs: list[float], ys: list[float]) -> tuple[float | None, float | None]:
        if len(xs) < RULE19_MIN_N:
            return None, None
        rho, pval = spearmanr(xs, ys)
        return (
            None if math.isnan(float(rho)) else float(rho),
            None if math.isnan(float(pval)) else float(pval),
        )

    xs_pair: list[float] = []
    ys_pair: list[float] = []
    by_unit: dict[tuple[str, str], list[tuple[float, float]]] = defaultdict(list)
    for r in margin_rows:
        if r["arm"] != "steered" or r["margin_shift"] is None:
            continue
        f = f_by_key.get((r["pair_id"], r["slot"]))
        if f is None:
            continue
        xs_pair.append(r["margin_shift"])
        ys_pair.append(f)
        by_unit[(r["cell"], r["slot"])].append((r["margin_shift"], f))
    xs_cell: list[float] = []
    ys_cell: list[float] = []
    percell_points: list[dict] = []
    dropped: list[str] = []
    for (cell, slot), pts in sorted(by_unit.items()):
        ms = [m for m, _ in pts]
        fs = [f for _, f in pts]
        if len(pts) < 2 or max(ms) <= min(ms) or max(fs) <= min(fs):
            dropped.append(f"{cell}|{slot}")
            continue
        xs_cell.append(sum(ms) / len(ms))
        ys_cell.append(sum(fs) / len(fs))
        percell_points.append(
            {
                "cell": cell,
                "slot": slot,
                "margin_shift_mean": xs_cell[-1],
                "f_beh_mean": ys_cell[-1],
                "n_pairs": len(pts),
            }
        )
    rho_cell, p_cell = _stat(xs_cell, ys_cell)
    rho_pair, p_pair = _stat(xs_pair, ys_pair)
    return {
        "rho_margin_fbeh_percell": rho_cell,
        "p_percell": p_cell,
        "n_cells": len(xs_cell),
        "percell_points": percell_points,
        "cells_dropped_no_dynamic_range": dropped,
        "dynamic_range_screen": RULE19_DYNAMIC_RANGE_SCREEN,
        "rho_margin_fbeh_perpair": rho_pair,
        "p_perpair": p_pair,
        "n_pairs": len(xs_pair),
        "validated": bool(rho_cell is not None and rho_cell > 0),
        "note": (
            "rule 19 (registered grain = across cells with dynamic range): the margin "
            "carries NO cross-condition read unless the per-cell rho > 0"
        ),
    }


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
    the per-layer Gram — O(n^2) per step instead of O(n*H) (H=4096, n=24:
    ~170x cheaper), which is what makes the B=1,000 permutation battery a
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


def _pe_probe_excluded(recs: dict, cell: str, carriers: list[str]) -> bool:
    """True when ANY of the cell's contexts is no-prefix (zero v_pe under the
    thinking-off template) — the (cell x pe) probe unit is then skipped with a
    NAMED record (divergences 1/9: a zero vector is not a state read)."""
    for carrier in carriers:
        for v in {v for vp in BANK.cell_pairs_per_carrier(cell) for v in vp}:
            rec = recs.get(BANK.context_id(cell, v, carrier))
            if rec is not None and rec.get("no_prefix"):
                return True
    return False


def step_probe(args: argparse.Namespace) -> None:
    bank = torch.load(args.bank_pt, map_location="cpu", weights_only=False)
    recs = bank["per_context"]
    layers = list(bank["layers"])
    torch.manual_seed(PROBE_SEED)
    gen = torch.Generator().manual_seed(PROBE_SEED)
    chunk = args.perm_chunk

    results: list[dict] = []
    perm_store: dict[str, np.ndarray] = {}
    pe_excluded_units: list[dict] = []
    for cell in BANK.all_cells():
        carriers = list(BANK.carriers_for(cell))
        vps = BANK.cell_pairs_per_carrier(cell)
        for slot in ("ce", "pe"):
            if slot == "pe" and _pe_probe_excluded(recs, cell, carriers):
                pe_excluded_units.append(
                    {
                        "cell": cell,
                        "slot": slot,
                        "reason": "no-prefix context(s) in cell — v_pe is a zero vector "
                        "under the thinking-off template (named exclusion, never a "
                        "silent zero-state read)",
                    }
                )
                logger.info("[probe] unit %s|pe skipped (no-prefix contexts)", cell)
                continue
            per_vp_obs: list[np.ndarray] = []  # each (L,)
            per_vp_perm: list[np.ndarray] = []  # each (B, L)
            transfer_aucs: list[float] = []
            for va, vb in vps:
                gram, y, groups = _vp_data(recs, layers, cell, slot, va, vb, carriers)
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
                    # Per-value-pair curves (probe_layer_curves points).
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
            "pe_excluded_units": pe_excluded_units,
            "seed": PROBE_SEED,
            "perm_b": args.perm_b,
            "probe": "kernelized L2 logistic (linear kernel), 150 epochs Adam lr=0.15 l2=1e-2",
        },
    )
    pdir = args.out_dir / "probe_perm_matrix"
    pdir.mkdir(parents=True, exist_ok=True)
    np.savez(pdir / "perm_auc_matrix.npz", **perm_store)
    _persist_perm_matrix(pdir, no_upload=args.no_upload)
    logger.info(
        "[probe] %d (cell x slot) units (%d pe units skipped no-prefix)",
        len(results),
        len(pe_excluded_units),
    )


def _persist_perm_matrix(pdir: Path, no_upload: bool) -> None:
    """Upload the probe permutation matrix to the HF data repo (plan §10).

    The matrix is what makes the max-selected permutation band RECOMPUTABLE —
    a pre-registered persistence commitment, not a convenience. It is ``*.npz``
    and therefore GITIGNORED repo-wide (a bare ``git add`` of out_dir silently
    skips it, rc=0 — the #958 class), so git is NOT a durable home: the
    registered destination is HF ``analysis_tensors/probe_perm_matrix/``.
    Fail-loud via the run driver's bounded-retry upload seam.
    """
    if no_upload:
        logger.warning(
            "[probe] --no-upload: perm matrix NOT persisted to HF (%s) — production "
            "runs must upload (plan §10 recomputability commitment)",
            pdir,
        )
        return
    uploaded = R.upload_dir_hf(pdir, f"{R.HF_PREFIX}/analysis_tensors/probe_perm_matrix", ["*.npz"])
    logger.info("[probe] perm matrix persisted to HF: %s", uploaded)


# ── step: two-by-two ──────────────────────────────────────────────────


def _causal_verdict(rec: dict) -> str:
    if rec["untestable_causal"]:
        return "untestable-causal"
    return "positive" if rec.get("holm_pass") and rec.get("disjoint_both_nulls") else "null"


def step_two_by_two(args: argparse.Namespace) -> None:
    stats = json.loads((args.out_dir / "stats.json").read_text())
    per_cell = stats["per_cell"]
    probe = {
        (r["cell"], r["slot"]): r
        for r in json.loads((args.out_dir / "probe.json").read_text())["results"]
    }
    out = []
    for _key, rec in sorted(per_cell.items()):
        pr = probe.get((rec["cell"], rec["slot"]))
        out.append(
            {
                "cell": rec["cell"],
                "slot": rec["slot"],
                "causal_verdict": _causal_verdict(rec),
                "probe_verdict": (
                    "positive" if pr and pr["probe_positive"] else "null" if pr else "missing"
                ),
                "f_steered_mean": rec.get("f_steered_mean"),
                "max_auc": pr["max_auc_over_layers"] if pr else None,
                "n_post_exclusion": rec["n_post_exclusion"],
            }
        )
    _write_json_atomic(
        args.out_dir / "two_by_two.json",
        {
            "cells": out,
            # Realized per-family m beside the ceilings + parent m — reported
            # everywhere verdicts render (plan §6 remedy A).
            "families": stats.get("family_m", {}),
        },
    )
    n_unt = sum(1 for r in out if r["causal_verdict"] == "untestable-causal")
    logger.info("[2x2] %d cells (%d untestable-causal)", len(out), n_unt)


# ── step: transfer (plan §6 prediction 8 — ONE registered test) ───────


def _kept_cell_values(rows: list[dict]) -> dict[tuple[str, str], list[float]]:
    """(cell, slot) -> kept steered per-pair F_beh values (separation-kept)."""
    out: dict[tuple[str, str], list[float]] = defaultdict(list)
    for r in rows:
        if r.get("arm") != "steered" or r.get("f_beh") is None:
            continue
        if r.get("separation") is None or abs(r["separation"]) < SEPARATION_BAR:
            continue
        out[(r["cell"], r["slot"])].append(float(r["f_beh"]))
    return out


def _rowwise_spearman(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-row Spearman rho for (B, C) matrices — plain ranks (continuous
    bootstrap means make exact ties measure-zero), fully vectorized."""

    def _rank(m: np.ndarray) -> np.ndarray:
        order = np.argsort(m, axis=1)
        r = np.empty_like(m)
        np.put_along_axis(
            r, order, np.broadcast_to(np.arange(m.shape[1], dtype=m.dtype), m.shape), axis=1
        )
        return r

    ra, rb = _rank(a), _rank(b)
    ra = ra - ra.mean(axis=1, keepdims=True)
    rb = rb - rb.mean(axis=1, keepdims=True)
    num = (ra * rb).sum(axis=1)
    den = np.sqrt((ra * ra).sum(axis=1) * (rb * rb).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def step_transfer(args: argparse.Namespace) -> None:
    """Plan §6 prediction 8: does the per-type F profile transfer across models?

    Spearman rho(F_beh_2329, F_beh_2162) over shared P1 (cell x slot) units at
    cell grain, pair-clustered bootstrap 95% CI (pairs resampled WITHIN cell,
    independently per run, B=10,000, seed 21620) — ONE registered test outside
    the three Holm families (single test, alpha = 0.05). Plus the
    verdict-transfer 2x2 comparison over ALL shared units, with BOTH runs'
    realized per-family m reported beside it (plan §6 comparability effect)
    and divergence-9 dropped/repaired cells flagged.
    """
    from scipy.stats import spearmanr

    parent_dir = args.parent_f_metrics
    parent_rows = list(_iter_jsonl(parent_dir / "f_cells.jsonl"))
    parent_p1_units = {(r["cell"], r["slot"]) for r in parent_rows if r.get("family") == "P1"}
    # Count-assert at load (plan §2: the parent's committed per-type F tables
    # are the transfer-read comparison target — 31 constructional P1 units).
    assert len(parent_p1_units) == PARENT_P1_UNIT_COUNT, (
        len(parent_p1_units),
        PARENT_P1_UNIT_COUNT,
        "parent f_cells.jsonl P1 unit count drifted from the committed table",
    )
    parent_stats = json.loads((parent_dir / "stats.json").read_text())
    child_rows = list(_iter_jsonl(args.out_dir / "f_cells.jsonl"))
    child_stats = json.loads((args.out_dir / "stats.json").read_text())

    parent_vals = _kept_cell_values(parent_rows)
    child_vals = _kept_cell_values(child_rows)

    # Divergence-9 flags: cells with token-identity drops (or boundary repairs)
    # in the frozen 2329 bank — flagged wherever they enter the transfer read.
    bank = json.loads(Path(args.bank_json).read_text())
    dropped_cells = {
        cell: row["n_dropped"]
        for cell, row in bank["token_identity"]["per_cell"].items()
        if row["n_dropped"] > 0
    }
    repaired_cells = sorted(bank.get("repaired_cells", []))

    shared_p1 = sorted(
        u
        for u in parent_p1_units
        if family_of(*u) == "P1" and parent_vals.get(u) and child_vals.get(u)
    )
    assert shared_p1, "transfer: zero shared P1 units with kept pairs on both runs"
    child_means = np.array([float(np.mean(child_vals[u])) for u in shared_p1])
    parent_means = np.array([float(np.mean(parent_vals[u])) for u in shared_p1])
    rho_obs, p_obs = spearmanr(child_means, parent_means)

    # Pair-clustered bootstrap CI: resample pairs WITHIN each cell,
    # independently per run (distinct seeds), then per-draw Spearman across
    # cells — batched (bootstrap_family_means_batched + vectorized rank corr).
    child_draws = np.stack(
        [
            bootstrap_family_means_batched(
                np.asarray(child_vals[u], dtype=np.float64)[:, None], BOOT_B, BOOT_SEED
            )[:, 0]
            for u in shared_p1
        ],
        axis=1,
    )  # (B, C)
    parent_draws = np.stack(
        [
            bootstrap_family_means_batched(
                np.asarray(parent_vals[u], dtype=np.float64)[:, None], BOOT_B, BOOT_SEED + 1
            )[:, 0]
            for u in shared_p1
        ],
        axis=1,
    )
    rho_draws = _rowwise_spearman(child_draws, parent_draws)
    lo, hi = np.nanpercentile(rho_draws, [2.5, 97.5])

    # Verdict-transfer 2x2 over ALL shared (cell x slot) units.
    parent_pc = parent_stats["per_cell"]
    child_pc = child_stats["per_cell"]
    verdict_rows: list[dict] = []
    agreement: dict[str, int] = defaultdict(int)
    for key in sorted(set(parent_pc) & set(child_pc)):
        pv = _causal_verdict(parent_pc[key])
        cv = _causal_verdict(child_pc[key])
        cell = child_pc[key]["cell"]
        verdict_rows.append(
            {
                "key": key,
                "cell": cell,
                "slot": child_pc[key]["slot"],
                "family": child_pc[key]["family"],
                "parent_verdict": pv,
                "child_verdict": cv,
                "match": pv == cv,
                "n_pairs_dropped_div9": dropped_cells.get(cell, 0),
                "repaired_div9": cell in repaired_cells,
            }
        )
        agreement[f"{pv}->{cv}"] += 1

    out = {
        "criterion": (
            "plan §6 prediction 8 — transfer correlation: Spearman rho(F_beh_2329, "
            "F_beh_2162) over shared P1 cells at cell grain, pair-clustered bootstrap "
            "95% CI; ONE registered test outside the three families (alpha = 0.05)"
        ),
        "parent_source": str(parent_dir),
        "parent_p1_unit_count_asserted": PARENT_P1_UNIT_COUNT,
        "n_shared_p1_units": len(shared_p1),
        "shared_p1_units": [f"{c}|{s}" for c, s in shared_p1],
        "rho": None if math.isnan(float(rho_obs)) else float(rho_obs),
        "p": None if math.isnan(float(p_obs)) else float(p_obs),
        "alpha": 0.05,
        "ci95_pair_clustered": [float(lo), float(hi)],
        "bootstrap": {
            "B": BOOT_B,
            "seed_child": BOOT_SEED,
            "seed_parent": BOOT_SEED + 1,
            "convention": "pairs resampled WITHIN cell independently per run per draw; "
            "per-draw Spearman across shared P1 cell means",
        },
        "per_unit": [
            {
                "cell": c,
                "slot": s,
                "f_beh_2329_mean": float(np.mean(child_vals[(c, s)])),
                "f_beh_2162_mean": float(np.mean(parent_vals[(c, s)])),
                "n_pairs_2329": len(child_vals[(c, s)]),
                "n_pairs_2162": len(parent_vals[(c, s)]),
                "n_pairs_dropped_div9": dropped_cells.get(c, 0),
                "repaired_div9": c in repaired_cells,
            }
            for c, s in shared_p1
        ],
        "verdict_transfer": {
            "note": (
                "per-cell Holm verdicts compared AS VERDICTS, each valid within its own "
                "family's analysis-time m (plan §6 comparability effect — the realized m "
                "gap is reported beside the comparison)"
            ),
            "agreement_counts": dict(sorted(agreement.items())),
            "n_shared_units": len(verdict_rows),
            "per_unit": verdict_rows,
        },
        "family_m_child": child_stats.get("family_m", {}),
        "family_m_parent_realized": parent_stats.get("families", {}),
        "family_m_ceilings": FAMILY_CEILING_M,
        "div9_flags": {
            "dropped_cells": dropped_cells,
            "repaired_cells": repaired_cells,
            "note": "divergence-9 token-identity drops/repairs in the frozen 2329 bank "
            "(realized at freeze: see bank.json token_identity)",
        },
    }
    _write_json_atomic(args.out_dir / "transfer.json", out)
    logger.info(
        "[transfer] rho=%.3f p=%s ci95=[%.3f, %.3f] over %d shared P1 units "
        "(agreement: %s; child m: %s; parent m: %s)",
        rho_obs,
        out["p"],
        lo,
        hi,
        len(shared_p1),
        dict(agreement),
        {f: v["realized_m"] for f, v in out["family_m_child"].items()},
        out["family_m_parent_realized"],
    )


# ── step: stage2 tables ───────────────────────────────────────────────


def step_stage2(args: argparse.Namespace) -> None:
    """Reduce stage-2 rollouts + judge scores to per-(pair x block) F_beh rows.

    Writes ``stage2_cells.jsonl`` — the ``layer_profile_stage2`` /
    ``layer_profile_stage2_perpair`` source: one row per
    (pair x slot x arm x layer x dose) stage-2 block membership, with the
    same dual-rubric contrast -> floor/ceiling-normalized F_beh reduction as
    ``step_f_tables`` (anchor pools shared with stage 1).
    """
    if args.stage2_dir is None or not args.stage2_dir.is_dir():
        if args.step == "all":
            logger.info("[stage2] skipped — no --stage2-dir (stage-2 not yet run)")
            return
        raise AssertionError("--step stage2 requires --stage2-dir with stage2_shard_*.jsonl")
    pairs = _pairs(args)
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
                _anchor_deltas(anchor_rows, beh_anchor, beh_anchor, p, p.a, args.exclude_cap_hit),
                _anchor_deltas(anchor_rows, beh_anchor, beh_anchor, p, p.b, args.exclude_cap_hit),
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
            if args.exclude_cap_hit and row.get("cap_hit", False):
                continue  # v55 restriction read: drop cap-hit draws (counted above)
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
    "transfer": step_transfer,
    "stage2": step_stage2,
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Issue #2329 P7 analysis driver.")
    ap.add_argument("--step", required=True, choices=(*STEPS, "all"))
    ap.add_argument("--rollouts-dir", type=Path, required=False)
    ap.add_argument("--anchors-dir", type=Path, required=False)
    ap.add_argument("--va-dir", type=Path, required=False)
    ap.add_argument("--margin-dir", type=Path, required=False)
    ap.add_argument("--stage2-dir", type=Path, required=False)
    ap.add_argument("--bank-pt", type=Path, required=False)
    ap.add_argument(
        "--bank-json",
        type=Path,
        default=Path("data/issue_2329/judge_inputs")
        / R.HF_PREFIX
        / "analysis_tensors/vc_bank/bank.json",
        help="frozen gate-0a bank.json (surviving-pair registry + divergence-9 flags)",
    )
    ap.add_argument(
        "--pe-exclusions",
        type=Path,
        default=Path("data/issue_2329/judge_inputs")
        / R.HF_PREFIX
        / "analysis_tensors/manifests/pe_exclusions.json",
        help="the pod driver's named no-prefix pe-slot exclusion manifest",
    )
    ap.add_argument(
        "--parent-f-metrics",
        type=Path,
        default=Path("eval_results/issue_2162/f_metrics"),
        help="the parent's COMMITTED f_metrics tables (transfer-read comparison target)",
    )
    ap.add_argument("--scores-dir", type=Path, default=Path("eval_results/issue_2329/judge/scores"))
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_2329/f_metrics"))
    ap.add_argument("--perm-b", type=int, default=PROBE_PERM_B)
    ap.add_argument("--perm-chunk", type=int, default=64)
    ap.add_argument(
        "--exclude-cap-hit",
        action="store_true",
        default=False,
        help="v55 restriction / truncation-sensitivity read (#2329): drop every draw whose "
        "completion hit the generation cap (per-row `cap_hit`) from the grid, anchor, and "
        "stage-2 per-draw contrasts. RESTRICTIVE ONLY -- strictly fewer draws contribute, "
        "no draw is newly admitted and no threshold relaxed -- so the restricted estimate is "
        "directly comparable to the unrestricted one. The `n_cap_hit` counters are unaffected "
        "(they still count what was dropped). Write to a SEPARATE --out-dir; never overwrite "
        "the shipped tables.",
    )
    ap.add_argument(
        "--no-upload",
        action="store_true",
        default=False,
        help="skip the HF persist of the probe perm matrix (local smoke/tests only — "
        "production runs MUST upload; the *.npz is gitignored, so git is not durable)",
    )
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
