"""Issue #2094 fu1_regen_confirm — VM-side ANALYSIS leg (cheap-band round 1).

Consumes the fu1 judge scores (``eval_results/issue_2094/judge_fu1/scores``),
the committed parent f-metrics artifacts, and the staged fu1 rollouts; writes
three JSONs to ``eval_results/issue_2094/f_metrics/fu1/`` (never touching the
parent's files):

A. **regen** (``fu1_regen_swap.json`` + ``fu1_wellsep_bootstrap_regen.json``)
   — the 16 cap-hit-breached pooled (slot, layer_variant, dose) cells were
   re-generated at ``max_new_tokens=2048`` (both arms). This leg reduces the
   fu1 grid-wave judge scores to per-cell F_beh rows with EXACTLY the parent
   ftables conventions (per-draw judge contrast, committed-anchor
   floor→ceiling normalization via ``FM.f_beh``, coherence gating at >60;
   F_act is NOT recomputed — the fu1 V_a tensors are not consumed in this
   analysis leg, swapped rows carry ``f_act: null``), writes the
   parent-vs-regen swap table (per pooled cell × arm × rubric kind, next to
   the recomputed residual 2048 cap-hit fractions — source: the staged regen
   rollout rows' own ``cap_hit`` field), and re-runs the FA-3
   separation-stratified pair-clustered bootstrap (B=10,000, seed 20941,
   |sep| >= 0.5 — ``issue2094_wellsep_bootstrap`` reused verbatim) over the
   full grid with the regen rows substituted for the 16 cells. The
   pre-registered REJOIN read: a breached cell re-enters the clean-set
   derivation iff its residual 2048 steered cap-hit fraction is <= 2%.

B. **conf1** (``fu1_conf1_confirmation.json``) — the 15 surviving
   well-separated clean families re-measured at temperature 1.0, K=5 draws,
   steered + shuffled-donor-null arms (stage-2 row contract). Reduction
   mirrors the parent stage-2 conventions (``issue2094_figures.stage2_cell_f``):
   per-draw contrast on coherent draws, committed-anchor floor→ceiling
   normalization, then pair-clustered aggregation (mean over a pair's kept
   draws), the |sep| >= 0.5 well-separated pair restriction, and a
   pair-clustered batched bootstrap (B=10,000, seed 20941) per family × arm
   with the disjoint-CI verdict per family.

Judge-score hygiene: a ``None`` score (rule-9 content drop) is DROPPED and
counted per family — never coerced. All fail-loud: coverage set-equality
between staged rollouts and score rows, duplicate-score-row detection,
unknown-cell detection, per-cell draw-count asserts.

VM launch convention (shared-VM thread caps):

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \\
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \\
    uv run python scripts/issue2094_fu1_analysis.py
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE any heavy import

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.experiments.issue2094 import fmetrics as FM  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2094_analysis as A  # noqa: E402
import issue2094_fu1 as FU1  # noqa: E402
import issue2094_wellsep_bootstrap as W  # noqa: E402

logger = logging.getLogger("issue2094_fu1_analysis")

REPO_ROOT = A.REPO_ROOT
DEFAULT_SCORES_DIR = REPO_ROOT / "eval_results/issue_2094/judge_fu1/scores"
DEFAULT_FMETRICS_DIR = REPO_ROOT / "eval_results/issue_2094/f_metrics"
DEFAULT_FRAGILITY = REPO_ROOT / FU1.FRAGILITY_REL
DEFAULT_ROLLOUTS = (
    REPO_ROOT
    / "data/issue_2094/fu1_judge_inputs/issue2094_singlepos/raw_completions/fu1_regen_confirm"
)
DEFAULT_OUT_DIR = DEFAULT_FMETRICS_DIR / "fu1"

RC_OK = 0


# ── score routing (fail-loud) ──────────────────────────────────────────


@dataclass
class Fu1Scores:
    """Routed fu1 judge scores. Keys mirror the judge's source fields:

    - ``grid_coh[(block_key, pair_id)]``: coherence score | None
    - ``grid_beh[(block_key, pair_id, rubric_kind, side)]``: behavior score | None
    - ``s2_coh[(cell, pair_id, draw)]``: coherence score | None
    - ``s2_beh[(cell, pair_id, draw, rubric_kind, side)]``: behavior score | None
    """

    grid_coh: dict = field(default_factory=dict)
    grid_beh: dict = field(default_factory=dict)
    s2_coh: dict = field(default_factory=dict)
    s2_beh: dict = field(default_factory=dict)


def check_wave_metas(scores_dir: Path) -> dict[str, dict]:
    """Every wave meta must exist, be complete, and match its score-row count."""
    score_files = sorted(scores_dir.glob("*.scores.jsonl"))
    assert score_files, f"no *.scores.jsonl under {scores_dir}"
    metas: dict[str, dict] = {}
    for f in score_files:
        wave = f.name.removesuffix(".scores.jsonl")
        meta_path = scores_dir / f"{wave}.meta.json"
        assert meta_path.is_file(), f"wave {wave}: meta sidecar missing"
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta.get("complete") is True, f"wave {wave}: meta not complete"
        n_rows = sum(1 for _ in A._iter_jsonl(f))
        n_items = meta["regime"]["n_items"]
        assert n_rows == n_items, f"wave {wave}: {n_rows} score rows != meta n_items {n_items}"
        metas[wave] = meta
    return metas


def route_fu1_scores(rows) -> Fu1Scores:
    """Route score rows by (kind, rubric) — duplicate keys are fail-loud."""
    sc = Fu1Scores()
    for row in rows:
        kind = row["kind"]
        assert kind in ("grid", "stage2"), f"unexpected score-row kind {kind!r}"
        if row["rubric_id"] == A.COHERENCE_RUBRIC_ID:
            if kind == "grid":
                key = (row["block_key"], row["pair_id"])
                target = sc.grid_coh
            else:
                key = (row["cell"], row["pair_id"], row["draw"])
                target = sc.s2_coh
        else:
            if kind == "grid":
                key = (row["block_key"], row["pair_id"], row["rubric_kind"], row["side"])
                target = sc.grid_beh
            else:
                key = (row["cell"], row["pair_id"], row["draw"], row["rubric_kind"], row["side"])
                target = sc.s2_beh
        if key in target:
            raise AssertionError(f"duplicate judge score row for key {key}")
        target[key] = row["score"]
    return sc


def load_anchor_stats(anchors_path: Path) -> dict[tuple[str, str], dict]:
    """Committed anchors.jsonl rows keyed (pair_id, kind) — the parent's
    floor→ceiling normalization source (``anchor_pair_stats`` output shape)."""
    out: dict[tuple[str, str], dict] = {}
    for row in A._iter_jsonl(anchors_path):
        key = (row["pair_id"], row["kind"])
        assert key not in out, f"duplicate anchor row {key}"
        out[key] = row
    assert out, f"no anchor rows in {anchors_path}"
    return out


# ── sub-analysis A: regen reduction + swap table + wellsep re-run ──────


def reduce_regen_rows(
    shard_rows: list[dict],
    sc: Fu1Scores,
    anchors: dict[tuple[str, str], dict],
    pairs_by_id: dict,
    breached: set[tuple[str, str, str]],
) -> list[dict]:
    """Per-cell F_beh rows for the regen grid — the parent ftables reduction
    (``assemble_shard_rows``) minus the V_a-derived F_act legs (V_a tensors
    are not consumed in this analysis leg; ``f_act`` is ``None``)."""
    keys = [(r["block_key"], r["pair_id"]) for r in shard_rows]
    assert len(set(keys)) == len(shard_rows), "duplicate (block_key, pair_id) in regen shards"
    assert set(sc.grid_coh) == set(keys), (
        "regen coherence score coverage != staged rollout rows "
        f"(scores {len(sc.grid_coh)}, rollouts {len(keys)})"
    )
    expected_beh = set()
    for r in shard_rows:
        cell = (r["slot"], r["layer_variant"], r["dose"])
        assert cell in breached, f"regen shard row outside the breached set: {cell}"
        pair = pairs_by_id[r["pair_id"]]
        for kind in A.BANK.SETTING_RUBRIC_KINDS[pair.setting]:
            for side in ("a", "b"):
                expected_beh.add((r["block_key"], r["pair_id"], kind, side))
    assert set(sc.grid_beh) == expected_beh, (
        "regen behavior score coverage != expected (block, pair, kind, side) set "
        f"(scores {len(sc.grid_beh)}, expected {len(expected_beh)})"
    )

    # Batched F_beh over (row, kind) — the parent's exact assembly.
    beh_index: list[tuple[int, str]] = []
    dp, df, dc = [], [], []
    beh_missing: dict[tuple[int, str], str] = {}
    for i, r in enumerate(shard_rows):
        pair = pairs_by_id[r["pair_id"]]
        for kind in A.BANK.SETTING_RUBRIC_KINDS[pair.setting]:
            sa = sc.grid_beh[(r["block_key"], r["pair_id"], kind, "a")]
            sb = sc.grid_beh[(r["block_key"], r["pair_id"], kind, "b")]
            st = anchors.get((r["pair_id"], kind))
            if sa is None or sb is None:
                beh_missing[(i, kind)] = "judge_dropped"
                continue
            if st is None or st["floor"]["mean"] is None or st["ceiling"]["mean"] is None:
                beh_missing[(i, kind)] = "anchor_missing"
                continue
            beh_index.append((i, kind))
            dp.append((float(sb) - float(sa)) / 100.0)
            df.append(st["floor"]["mean"])
            dc.append(st["ceiling"]["mean"])
    fb = None
    if beh_index:
        fb = FM.f_beh(torch.tensor(dp), torch.tensor(df), torch.tensor(dc))
    beh_by_rowkind: dict[tuple[int, str], dict] = {}
    for j, (i, kind) in enumerate(beh_index):
        beh_by_rowkind[(i, kind)] = {
            "f_beh": A._nan_to_none(fb.f_beh[j]),
            "contrast": float(fb.contrast[j]),
            "denominator": float(fb.denominator[j]),
            "degenerate_denominator": bool(fb.degenerate_denominator[j]),
            "negative_denominator": bool(fb.negative_denominator[j]),
            "delta_patched": dp[j],
        }

    out = []
    for i, r in enumerate(shard_rows):
        pair = pairs_by_id[r["pair_id"]]
        coh = sc.grid_coh[(r["block_key"], r["pair_id"])]
        coherent = A._coherent(coh) if coh is not None else None
        excluded = coherent is not True
        beh = {}
        for kind in A.BANK.SETTING_RUBRIC_KINDS[pair.setting]:
            rec = beh_by_rowkind.get((i, kind))
            if rec is None:
                beh[kind] = {"f_beh": None, "missing": beh_missing.get((i, kind), "unknown")}
            elif excluded:
                beh[kind] = {**rec, "f_beh": None, "excluded_incoherent_raw": rec["f_beh"]}
            else:
                beh[kind] = rec
        out.append(
            {
                "block_key": r["block_key"],
                "slot": r["slot"],
                "layer_variant": r["layer_variant"],
                "steer_layers": r["layers"],
                "dose": r["dose"],
                "alpha": r.get("alpha"),
                "vec_type": r["vec_type"],
                "arm": r["arm"],
                "pair_id": r["pair_id"],
                "setting": r["setting"],
                "context_a": r["context_a"],
                "context_b": r["context_b"],
                "donor_pair_id": r.get("donor_pair_id"),
                "degenerate_self": A.degenerate_self(r),
                **A.annotate_donor(r),
                "coherence_score": coh,
                "coherent": coherent,
                "excluded_incoherent": excluded,
                "cap_hit": bool(r.get("cap_hit")),
                # V_a-derived legs NOT recomputed in this analysis leg.
                "f_act": None,
                "f_beh": beh,
                "source": "fu1_regen_2048",
            }
        )
    return out


def recompute_caphit_2048(shard_rows: list[dict]) -> dict[tuple[str, str, str], dict]:
    """Residual 2048 cap-hit fraction per pooled (slot, lv, dose) × arm —
    recomputed from the staged regen rollout rows' own ``cap_hit`` field
    (the fragility artifact's pooled grain, vec types pooled)."""
    pooled: dict[tuple[str, str, str], dict] = {}
    for r in shard_rows:
        key = (r["slot"], r["layer_variant"], r["dose"])
        arms = pooled.setdefault(key, {})
        agg = arms.setdefault(r["arm"], {"n": 0, "cap_hit": 0})
        agg["n"] += 1
        agg["cap_hit"] += int(bool(r["cap_hit"]))
    for arms in pooled.values():
        for agg in arms.values():
            agg["cap_hit_frac"] = agg["cap_hit"] / agg["n"] if agg["n"] else None
    return pooled


def _pooled_beh_means(
    rows: list[dict], ws: set[tuple[str, str]]
) -> dict[tuple[str, str, str, str, str], dict]:
    """Mean F_beh per (slot, lv, dose, arm, kind) over non-None values —
    plus the |sep|>=0.5 well-separated-pair restricted mean."""
    acc: dict[tuple, dict] = {}
    for r in rows:
        for kind, rec in (r.get("f_beh") or {}).items():
            key = (r["slot"], r["layer_variant"], r["dose"], r["arm"], kind)
            a = acc.setdefault(
                key,
                {"vals": [], "vals_wellsep": [], "n_rows": 0, "n_excluded_incoherent": 0},
            )
            a["n_rows"] += 1
            a["n_excluded_incoherent"] += int(bool(r.get("excluded_incoherent")))
            v = rec.get("f_beh")
            if v is None:
                continue
            a["vals"].append(float(v))
            if (r["pair_id"], kind) in ws:
                a["vals_wellsep"].append(float(v))
    out = {}
    for key, a in acc.items():
        out[key] = {
            "mean_f_beh": float(np.mean(a["vals"])) if a["vals"] else None,
            "n_values": len(a["vals"]),
            "mean_f_beh_wellsep": (
                float(np.mean(a["vals_wellsep"])) if a["vals_wellsep"] else None
            ),
            "n_values_wellsep": len(a["vals_wellsep"]),
            "n_rows": a["n_rows"],
            "n_excluded_incoherent": a["n_excluded_incoherent"],
        }
    return out


def build_swap_table(
    parent_breached_rows: list[dict],
    regen_rows: list[dict],
    fragility: dict,
    caphit_2048: dict[tuple[str, str, str], dict],
    ws: set[tuple[str, str]],
    breached: list[tuple[str, str, str]],
) -> dict:
    """Parent (1024) vs regen (2048) pooled-cell F_beh means + cap-hit."""
    frag_by_cell = {(c["slot"], c["layer_variant"], c["dose"]): c for c in fragility["cells"]}
    parent_means = _pooled_beh_means(parent_breached_rows, ws)
    regen_means = _pooled_beh_means(regen_rows, ws)
    cells = []
    for cell in breached:
        slot, lv, dose = cell
        frag = frag_by_cell[cell]
        ch48 = caphit_2048[cell]
        rec: dict = {
            "slot": slot,
            "layer_variant": lv,
            "dose": dose,
            "cap_hit_frac_1024": {arm: frag[arm]["cap_hit_frac"] for arm in ("steered", "null")},
            "cap_hit_frac_2048": {arm: ch48[arm]["cap_hit_frac"] for arm in ("steered", "null")},
            "still_breached_at_2048": (ch48["steered"]["cap_hit_frac"] > FU1.CAPHIT_TRIGGER_FRAC),
            "arms": {},
        }
        for arm in ("steered", "null"):
            kinds = sorted(
                {k[4] for k in parent_means if k[:4] == (slot, lv, dose, arm)}
                | {k[4] for k in regen_means if k[:4] == (slot, lv, dose, arm)}
            )
            arm_rec = {}
            for kind in kinds:
                p = parent_means.get((slot, lv, dose, arm, kind))
                g = regen_means.get((slot, lv, dose, arm, kind))
                entry = {"parent_1024": p, "regen_2048": g}
                for suffix in ("mean_f_beh", "mean_f_beh_wellsep"):
                    pv = (p or {}).get(suffix)
                    gv = (g or {}).get(suffix)
                    entry[f"delta_{suffix}"] = (
                        (gv - pv) if (pv is not None and gv is not None) else None
                    )
                arm_rec[kind] = entry
            rec["arms"][arm] = arm_rec
        cells.append(rec)
    return {
        "trigger_frac": FU1.CAPHIT_TRIGGER_FRAC,
        "regen_max_new_tokens": FU1.REGEN_MAX_NEW_TOKENS,
        "caphit_2048_source": (
            "recomputed from the staged regen rollout rows' cap_hit field "
            "(data/issue_2094/fu1_judge_inputs/.../fu1_regen_confirm/regen)"
        ),
        "n_cells": len(cells),
        "cells": cells,
    }


def clean_families(reads: dict[str, dict], breached: set[tuple[str, str, str]]) -> list[str]:
    """The clean-surviving-family predicate of ``FU1.derive_conf1_families``
    WITHOUT its ==15 count assert (the re-run may legitimately change the
    count; the mirror is pinned against the parent artifact in tests)."""
    fams = []
    for key in sorted(reads):
        rec = reads[key]
        parts = key.split("|")
        assert len(parts) == 6, f"unexpected family key shape: {key!r}"
        setting, slot, lv, dose, vec_type, metric = parts
        if metric not in FU1.BEH_METRICS:
            continue
        if not rec.get("cis_disjoint"):
            continue
        if rec.get("direction") != "steered_above":
            continue
        if int(rec.get("n_pairs_used", 0)) < FU1.CONF1_MIN_WELLSEP_PAIRS:
            continue
        if (slot, lv, dose) in breached:
            continue
        fams.append(key)
    return fams


def swap_rows(
    parent_rows: list[dict],
    regen_rows: list[dict],
    breached: set[tuple[str, str, str]],
) -> tuple[list[dict], list[dict]]:
    """(swapped full-grid rows, removed parent rows) — regen rows substituted
    for every parent row in a breached pooled cell, set-equality asserted."""
    kept, removed = [], []
    for r in parent_rows:
        (removed if (r["slot"], r["layer_variant"], r["dose"]) in breached else kept).append(r)
    removed_keys = {(r["block_key"], r["pair_id"]) for r in removed}
    regen_keys = {(r["block_key"], r["pair_id"]) for r in regen_rows}
    assert removed_keys == regen_keys, (
        f"swap mismatch: removed {len(removed_keys)} parent (block, pair) keys, "
        f"regen carries {len(regen_keys)}; symmetric diff "
        f"{sorted(removed_keys ^ regen_keys)[:6]}"
    )
    return kept + regen_rows, removed


# ── sub-analysis B: conf1 confirmation ─────────────────────────────────


def conf1_family_map(families: list[dict]) -> dict[str, dict]:
    """conf1 cell-key prefix ('fu1|setting|slot|lv|dose|vt') → family record
    (exactly one family per cell — asserted, mirroring the pod derivation)."""
    cells = FU1.conf1_cells_from_families(families)
    fam_by_key = {f["family"]: f for f in families}
    out: dict[str, dict] = {}
    for cell in cells:
        assert len(cell["families"]) == 1, f"conf1 cell carries != 1 family: {cell}"
        fam = fam_by_key[cell["families"][0]]
        prefix = "|".join(
            ["fu1", fam["setting"], fam["slot"], fam["layer_variant"], fam["dose"], fam["vec_type"]]
        )
        out[prefix] = fam
    return out


def reduce_conf1(
    sc: Fu1Scores,
    families: list[dict],
    anchors: dict[tuple[str, str], dict],
    ws: set[tuple[str, str]],
) -> dict[str, dict]:
    """Per (family, arm): per-pair mean-over-kept-draws F_beh values on the
    family's OWN rubric kind, well-separated pairs only (others ride as NaN
    through the NaN-aware bootstrap). Returns {family: {arm: {...}}}."""
    by_prefix = conf1_family_map(families)
    known_cells = {f"{p}|{arm}" for p in by_prefix for arm in ("steered", "null")}
    seen_cells = {k[0] for k in sc.s2_coh} | {k[0] for k in sc.s2_beh}
    unknown = seen_cells - known_cells
    if unknown:
        raise AssertionError(f"stage2 score rows carry unknown conf1 cell keys: {sorted(unknown)}")
    missing = known_cells - seen_cells
    assert not missing, f"conf1 cells missing from stage2 scores: {sorted(missing)}"

    pairs = A.BANK.build_pairs()
    pairs_by_setting = {
        s: sorted(p.pair_id for p in pairs if p.setting == s)
        for s in ("matched_prefix", "matched_query", "cross")
    }
    # Realized draw set per cell (asserted uniform == CONF1_DRAWS).
    draws_by_cell: dict[str, set[int]] = {}
    for cell, _pid, d in sc.s2_coh:
        draws_by_cell.setdefault(cell, set()).add(d)
    for cell, ds in draws_by_cell.items():
        assert ds == set(range(FU1.CONF1_DRAWS)), f"cell {cell}: draw set {sorted(ds)}"

    out: dict[str, dict] = {}
    for prefix, fam in sorted(by_prefix.items()):
        kind = fam["metric"].removeprefix("f_beh_")
        pids = pairs_by_setting[fam["setting"]]
        fam_out: dict[str, dict] = {}
        for arm in ("steered", "null"):
            cell = f"{prefix}|{arm}"
            n_coh_rows = sum(1 for k in sc.s2_coh if k[0] == cell)
            assert n_coh_rows == len(pids) * FU1.CONF1_DRAWS, (
                f"cell {cell}: {n_coh_rows} coherence rows != "
                f"{len(pids)} pairs x {FU1.CONF1_DRAWS} draws"
            )
            values = np.full(len(pids), np.nan)
            counts = {
                "n_draws_total": 0,
                "n_incoherent": 0,
                "n_judge_dropped": 0,
                "n_kept_draws": 0,
                "n_pairs_excluded_wellsep": 0,
            }
            per_pair_n_kept: dict[str, int] = {}
            for j, pid in enumerate(pids):
                if (pid, kind) not in ws:
                    counts["n_pairs_excluded_wellsep"] += 1
                    continue
                st = anchors[(pid, kind)]
                fl, ce = st["floor"]["mean"], st["ceiling"]["mean"]
                assert fl is not None and ce is not None and abs(ce - fl) >= 1e-9, (
                    f"degenerate anchor for well-separated pair {(pid, kind)}"
                )
                fs = []
                for d in range(FU1.CONF1_DRAWS):
                    counts["n_draws_total"] += 1
                    coh = sc.s2_coh[(cell, pid, d)]
                    if not A._coherent(coh):
                        counts["n_incoherent"] += 1
                        continue
                    sa = sc.s2_beh.get((cell, pid, d, kind, "a"))
                    sb = sc.s2_beh.get((cell, pid, d, kind, "b"))
                    if sa is None or sb is None:
                        counts["n_judge_dropped"] += 1
                        continue
                    delta = (float(sb) - float(sa)) / 100.0
                    fs.append((delta - fl) / (ce - fl))
                if fs:
                    values[j] = float(np.mean(fs))
                counts["n_kept_draws"] += len(fs)
                per_pair_n_kept[pid] = len(fs)
            fam_out[arm] = {
                "values": values,
                "pair_ids": pids,
                "per_pair_n_kept_draws": per_pair_n_kept,
                **counts,
            }
        out[fam["family"]] = {"kind": kind, "setting": fam["setting"], "arms": fam_out}
    return out


def conf1_reads(reduced: dict[str, dict], n_boot: int, seed: int) -> tuple[list[dict], dict]:
    """Pair-clustered bootstrap CIs per (family, arm) + disjoint-CI verdicts.

    ONE batched bootstrap call per setting over all (family, arm) columns —
    the ``bootstrap_family_means_batched`` index-GEMM battery (no per-draw
    python loop)."""
    by_setting: dict[str, list[tuple[str, str]]] = {}
    for fam, rec in sorted(reduced.items()):
        by_setting.setdefault(rec["setting"], []).extend((fam, arm) for arm in ("steered", "null"))
    stats: dict[tuple[str, str], dict] = {}
    for setting, cols in sorted(by_setting.items()):
        values = np.stack([reduced[fam]["arms"][arm]["values"] for fam, arm in cols], axis=1)
        boots = A.bootstrap_family_means_batched(values, n_boot, seed)
        with np.errstate(invalid="ignore"):
            obs = np.nanmean(values, axis=0)
        for j, (fam, arm) in enumerate(cols):
            col = boots[:, j]
            valid = col[~np.isnan(col)]
            stats[(fam, arm)] = {
                "observed_mean": A._nan_to_none(obs[j]),
                "ci_lo": float(np.percentile(valid, 2.5)) if valid.size else None,
                "ci_hi": float(np.percentile(valid, 97.5)) if valid.size else None,
                "n_valid_draws": int(valid.size),
                "n_pairs_used": int((~np.isnan(values[:, j])).sum()),
            }
    rows = []
    n_confirmed = 0
    for fam, rec in sorted(reduced.items()):
        st, nu = stats[(fam, "steered")], stats[(fam, "null")]
        comparable = (
            st["ci_lo"] is not None
            and st["ci_hi"] is not None
            and nu["ci_lo"] is not None
            and nu["ci_hi"] is not None
        )
        disjoint = bool(comparable and (st["ci_lo"] > nu["ci_hi"] or st["ci_hi"] < nu["ci_lo"]))
        direction = None
        if st["observed_mean"] is not None and nu["observed_mean"] is not None:
            direction = (
                "steered_above" if st["observed_mean"] > nu["observed_mean"] else "steered_below"
            )
        confirmed = disjoint and direction == "steered_above"
        n_confirmed += int(confirmed)
        row = {
            "family": fam,
            "setting": rec["setting"],
            "kind": rec["kind"],
            "comparable": comparable,
            "cis_disjoint": disjoint,
            "direction": direction,
            "confirmed": confirmed,
        }
        for arm in ("steered", "null"):
            a = reduced[fam]["arms"][arm]
            row[arm] = {
                **stats[(fam, arm)],
                "n_draws_total": a["n_draws_total"],
                "n_kept_draws": a["n_kept_draws"],
                "n_incoherent": a["n_incoherent"],
                "n_judge_dropped": a["n_judge_dropped"],
                "n_pairs_excluded_wellsep": a["n_pairs_excluded_wellsep"],
            }
        rows.append(row)
    summary = {
        "n_families": len(rows),
        "n_confirmed": n_confirmed,
        "confirmed_families": sorted(r["family"] for r in rows if r["confirmed"]),
        "note": (
            "confirmed = steered vs shuffled-donor-null 95% pair-clustered "
            "bootstrap CIs disjoint with steered above, on the family's own "
            "rubric kind over well-separated pairs (|sep| >= 0.5), at "
            "temperature 1.0 / K=5 draws (post-selection confirmation of the "
            "greedy stage-1 read — never an unbiased effect estimate)"
        ),
    }
    return rows, summary


# ── main ───────────────────────────────────────────────────────────────


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0].replace("%", "%%"))
    ap.add_argument("--scores-dir", type=Path, default=DEFAULT_SCORES_DIR)
    ap.add_argument("--fmetrics-dir", type=Path, default=DEFAULT_FMETRICS_DIR)
    ap.add_argument("--fragility", type=Path, default=DEFAULT_FRAGILITY)
    ap.add_argument("--rollouts-root", type=Path, default=DEFAULT_ROLLOUTS)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    ap.add_argument("--n-boot", type=int, default=A.BOOTSTRAP_B)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ── inputs ──────────────────────────────────────────────────────────
    logger.info("[phase=fu1a_load]")
    metas = check_wave_metas(args.scores_dir)
    logger.info("[load] %d complete waves", len(metas))
    rows_iter = (
        row for f in sorted(args.scores_dir.glob("*.scores.jsonl")) for row in A._iter_jsonl(f)
    )
    sc = route_fu1_scores(rows_iter)
    logger.info(
        "[load] scores routed: grid_coh=%d grid_beh=%d s2_coh=%d s2_beh=%d",
        len(sc.grid_coh),
        len(sc.grid_beh),
        len(sc.s2_coh),
        len(sc.s2_beh),
    )
    anchors = load_anchor_stats(args.fmetrics_dir / "anchors.jsonl")
    fragility = json.loads(args.fragility.read_text(encoding="utf-8"))
    parent_wellsep = json.loads(
        (args.fmetrics_dir / "bootstrap_cis_wellsep.json").read_text(encoding="utf-8")
    )
    breached = FU1.derive_breached_cells(fragility)
    families = FU1.derive_conf1_families(parent_wellsep, set(breached))
    pairs = A.BANK.build_pairs()
    pairs_by_id = {p.pair_id: p for p in pairs}
    ws, ws_any = W.load_wellsep(args.fmetrics_dir / "anchors.jsonl", W.MIN_SEPARATION)

    # ── sub-analysis A: regen reduction + swap table ────────────────────
    logger.info("[phase=fu1a_regen]")
    shard_rows = []
    for f in sorted((args.rollouts_root / "regen").glob("shard_*.jsonl")):
        shard_rows.extend(A._iter_jsonl(f))
    assert len(shard_rows) == FU1.EXPECTED_REGEN_TOTALS["cells_total"], (
        f"{len(shard_rows)} regen rollout rows != "
        f"{FU1.EXPECTED_REGEN_TOTALS['cells_total']} expected"
    )
    regen_rows = reduce_regen_rows(shard_rows, sc, anchors, pairs_by_id, set(breached))
    caphit_2048 = recompute_caphit_2048(shard_rows)
    assert set(caphit_2048) == set(breached), sorted(set(caphit_2048) ^ set(breached))
    still_breached = sorted(
        c for c in breached if caphit_2048[c]["steered"]["cap_hit_frac"] > FU1.CAPHIT_TRIGGER_FRAC
    )
    rejoined = sorted(set(breached) - set(still_breached))
    logger.info(
        "[regen] residual 2048 steered cap-hit > %.0f%%: %d/%d cells (rejoin: %d)",
        100 * FU1.CAPHIT_TRIGGER_FRAC,
        len(still_breached),
        len(breached),
        len(rejoined),
    )

    parent_rows = list(A._iter_jsonl(args.fmetrics_dir / "f_cells.jsonl")) + list(
        A._iter_jsonl(args.fmetrics_dir / "null_cells.jsonl")
    )
    swapped_rows, removed_parent_rows = swap_rows(parent_rows, regen_rows, set(breached))
    swap = build_swap_table(removed_parent_rows, regen_rows, fragility, caphit_2048, ws, breached)
    swap["rejoin"] = {
        "rule": "a breached cell rejoins the clean-set derivation iff its residual "
        "2048 steered pooled cap-hit fraction is <= the 2% trigger",
        "still_breached_at_2048": [list(c) for c in still_breached],
        "rejoined_at_2048": [list(c) for c in rejoined],
    }
    swap["note"] = (
        "fu1 sub-item A swap table: the 16 cap-hit-breached pooled cells "
        "re-generated at max_new_tokens=2048 (both arms, same pairs/donors/"
        "protocol), reduced with the parent ftables conventions from the fu1 "
        "judge scores + committed anchors; F_act not recomputed (V_a tensors "
        "not consumed in this analysis leg)"
    )
    swap["repro"] = A._repro()
    A._write_json_atomic(args.out_dir / "fu1_regen_swap.json", swap)
    logger.info("[phase=fu1a_swap_done] -> %s", args.out_dir / "fu1_regen_swap.json")

    # ── sub-analysis A: wellsep bootstrap re-run on the swapped grid ────
    logger.info("[phase=fu1a_wellsep_regen] n_boot=%d", args.n_boot)
    eligible, n_degenerate_excluded = A.bootstrap_eligible_rows(swapped_rows)
    fams = W.compute_wellsep_families(eligible, ws, ws_any, args.n_boot)
    reads, summary = W.steered_vs_null_reads(fams)

    # Internal consistency gate: families untouched by the swap must reproduce
    # the parent wellsep artifact EXACTLY (same rows, same seed, same battery;
    # the draw-index stream is column-content-independent). Only valid at the
    # parent's production n_boot.
    parent_fams = parent_wellsep["families"]
    n_checked, mismatches = 0, []
    if args.n_boot == parent_wellsep["B"]:
        for key, rec in fams.items():
            _arm, _setting, slot, lv, dose, _vt, _metric = key.split("|")
            if (slot, lv, dose) in set(breached):
                continue
            n_checked += 1
            if rec != parent_fams.get(key):
                mismatches.append(key)
        assert not mismatches, (
            f"{len(mismatches)} non-swapped families diverge from the parent "
            f"wellsep artifact (swap touched the wrong rows?): {mismatches[:6]}"
        )

    parent_clean = [f["family"] for f in families]
    clean_full_exclusion = clean_families(reads, set(breached))
    clean_rejoin = clean_families(reads, set(still_breached))
    verdict = {
        "parent_clean_set": parent_clean,
        "regen_clean_set_full_exclusion": clean_full_exclusion,
        "regen_clean_set_rejoin_exclusion": clean_rejoin,
        "families_gained_vs_parent": sorted(set(clean_rejoin) - set(parent_clean)),
        "families_lost_vs_parent": sorted(set(parent_clean) - set(clean_rejoin)),
        "clean_set_changed": sorted(clean_rejoin) != sorted(parent_clean),
        "breached_cell_family_reads": {
            key: reads[key]
            for key in sorted(reads)
            if (lambda p: (p[1], p[2], p[3]) in set(breached))(key.split("|"))
        },
    }
    A._write_json_atomic(
        args.out_dir / "fu1_wellsep_bootstrap_regen.json",
        {
            "B": args.n_boot,
            "seed": A.BOOTSTRAP_SEED,
            "resample_axis": "pairs (pair-clustered, within setting)",
            "degenerate_self_excluded": n_degenerate_excluded,
            "restriction": {
                "min_abs_separation": W.MIN_SEPARATION,
                "n_wellsep_pair_kinds": len(ws),
                "n_wellsep_pairs_any_kind": len(ws_any),
            },
            "swap_provenance": {
                "n_parent_rows": len(parent_rows),
                "n_rows_removed": len(removed_parent_rows),
                "n_regen_rows_inserted": len(regen_rows),
                "breached_cells": [list(c) for c in breached],
                "f_act_on_swapped_rows": (
                    "null — V_a tensors not consumed in this analysis leg; the "
                    "16 swapped cells' f_act families read n_pairs_used=0 here "
                    "(parent artifact remains the F_act record for those cells)"
                ),
            },
            "consistency_gate": {
                "n_nonswapped_families_checked": n_checked,
                "exact_match": bool(n_checked) and not mismatches,
                "note": "non-swapped families must equal the parent "
                "bootstrap_cis_wellsep.json bit-exactly (checked only at the "
                "parent's production B)",
            },
            "verdict": verdict,
            "note": (
                "fu1 sub-item A: the FA-3 separation-stratified pair-clustered "
                "bootstrap re-run over the full grid with regen-2048 F_beh rows "
                "substituted for the 16 cap-hit-breached pooled cells "
                "(issue2094_wellsep_bootstrap conventions reused verbatim); "
                "bootstrap_cis_wellsep.json is the unrestricted parent and is "
                "unchanged"
            ),
            "families": fams,
            "steered_vs_null": reads,
            "summary": summary,
            "repro": A._repro(),
        },
    )
    logger.info(
        "[phase=fu1a_wellsep_regen_done] families=%d clean_set: parent=%d "
        "regen(full-exclusion)=%d regen(rejoin)=%d gained=%d lost=%d",
        len(fams),
        len(parent_clean),
        len(clean_full_exclusion),
        len(clean_rejoin),
        len(verdict["families_gained_vs_parent"]),
        len(verdict["families_lost_vs_parent"]),
    )

    # ── sub-analysis B: conf1 confirmation ──────────────────────────────
    logger.info("[phase=fu1a_conf1]")
    reduced = reduce_conf1(sc, families, anchors, ws)
    rows, conf_summary = conf1_reads(reduced, args.n_boot, A.BOOTSTRAP_SEED)
    A._write_json_atomic(
        args.out_dir / "fu1_conf1_confirmation.json",
        {
            "B": args.n_boot,
            "seed": A.BOOTSTRAP_SEED,
            "resample_axis": "pairs (pair-clustered, within setting)",
            "temperature": FU1.CONF1_TEMPERATURE,
            "n_draws_per_pair": FU1.CONF1_DRAWS,
            "restriction": {
                "min_abs_separation": W.MIN_SEPARATION,
                "pair_keep": "pair kept iff its (pair, family-kind) anchor "
                "|separation| >= floor; excluded pairs ride as NaN",
            },
            "reduction": (
                "per (family, arm, pair): mean over coherent kept draws of the "
                "per-draw anchored contrast f = ((judge_B - judge_A)/100 - "
                "floor_mean) / (ceiling_mean - floor_mean), committed-anchor "
                "normalization (parent stage-2 conventions); None judge scores "
                "DROPPED and counted, never coerced"
            ),
            "families": rows,
            "summary": conf_summary,
            "repro": A._repro(),
        },
    )
    logger.info(
        "[phase=fu1a_conf1_done] confirmed %d/%d families -> %s",
        conf_summary["n_confirmed"],
        conf_summary["n_families"],
        args.out_dir / "fu1_conf1_confirmation.json",
    )
    logger.info("[phase=fu1a_done]")
    return RC_OK


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    # Explicit exit BEFORE C-extension interpreter finalization (#1689).
    sys.exit(rc)
