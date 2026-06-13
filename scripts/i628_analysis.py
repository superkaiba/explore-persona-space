"""Issue #628 — registered analysis (plan §6) for the marker-rig contrast.

Consumes the per-cell four-float G-cell JSONs (fresh arms under
``eval_results/issue_628/G_cells/<arm>/``, the reused #537 arm from the
committed snapshot under ``eval_results/issue_628/inputs/i537_marker/``),
the reuse-arm trained-negative columns, the stop-step telemetry, and the
Phase-4 on-policy reads, and emits ``analysis/rig_contrast.json`` + figures.

Registered choices (pinned here; changing any is a plan deviation):
- ``h2_primary_sep_mode = "marker"``: separator-trained arms enter every H2
  statistic at their OWN trained slot; the plain-slot version is emitted as
  the named sensitivity artifact (``analysis/h2_plain_slot_sensitivity.json``).
- ``primary_eval_cids = eval_cids - {train_cid} - NEGATIVE_CIDS`` (29
  bystander columns; trained-negative columns carry the separate
  ``trained_negative_signature`` read).
- Per-statistic mask table: the ``binst_marker`` DIAGONAL is censored from
  the H1 dial means + matched-install trigger in BOTH seeds symmetrically
  (off-diagonal columns stay in H2); a (train_cid, seed) ROW is pairwise-
  deleted from H2 when its source diagonal misses the 4-nat manipulation
  gate in EITHER compared arm (#537 inherits: fmt_code seed-42 masked at
  3.97; fmt_code seed-1042 at 4.28 is a BAND miss, not an H2 row mask).
- Transfer-fraction denominators use the EOS-margin source gain (never the
  log-prob diagonal) with a minimum-denominator screen; saturated
  denominators are flagged, never auto-dropped.
- H-inert: cluster-by-adapter bootstrap 95% CI within ±0.3 nat over the 8
  adapter pairs; the per-cell sign test is DIAGNOSTIC-ONLY.
- DV1↔DV3 proxy validation over the exactly-enumerated 80-cell matched
  off-diagonal key set (diagonals excluded + reported separately), pass
  Spearman rho ≥ 0.7, fail-routing registered. Claim routing fires on
  ``pass is not True`` — an INCOMPLETE validation (missing inputs) scopes
  the claim exactly like a FAIL (not-validated ⇒ not licensed).
- Matched-install fallback (§6 read 1) is AUTOMATED: when the >2-nat
  arm-mean dial trigger fires, the analysis emits
  ``analysis/matched_install_reread_spec.json`` (per-cell mismatched arm,
  target dial, nearest checkpoint from the saved band trajectories, the
  ``default`` + 4 trained-negative columns); ``scripts/i628_dispatch.py
  --phase matched-install-reread --spec <path>`` consumes it, and the
  re-read cells are ingested here as ``matched_install_read`` (claim
  scoped to the re-read columns; never relicenses the 29-column headline).
- Seed-equal-weighted bootstrap VARIANT reported alongside the registered
  cid-cluster pooled CI; the registered H2 gate stays the both-seeds
  conjunction + pooled mean (unchanged).
- The plain-slot H2 sensitivity anchors its manipulation-gate row masks at
  the PRIMARY (own-trained-slot) diagonals, not the plain diagonals — the
  implant-took check is a property of the cell at its trained slot, and the
  plain diagonal folds DV4 slot misalignment into the mask itself.
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import subprocess
import sys
from pathlib import Path

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("i628_analysis")

REPO = Path(__file__).resolve().parents[1]

# ── Registered constants ─────────────────────────────────────────────────────
H2_PRIMARY_SEP_MODE = "marker"
LEGACY_ARM = "rig_O_sep_deadneg"
REUSE_ARM = "rig_N_i537_reuse"
CANONICAL_ARM = "rig_Nplus_canonical"
SEP_ONLY_ARM = "rig_S_nosep_deadneg"
FLAG_ONLY_ARM = "rig_F_sep_liveneg"
SEP_ARMS = (LEGACY_ARM, FLAG_ONLY_ARM)
LIVE_NEG_ARMS = (CANONICAL_ARM, FLAG_ONLY_ARM, REUSE_ARM)
MINI_ARM_CIDS = ("sp_swe", "wc_short_advice", "icl_k8", "binst_marker")
ONPOLICY_ARMS = (LEGACY_ARM, CANONICAL_ARM)
ARM_LABELS = {
    LEGACY_ARM: "Legacy rig",
    REUSE_ARM: "Revised rig (#537 reuse)",
    CANONICAL_ARM: "Full revised rig",
    SEP_ONLY_ARM: "Separator-only edit",
    FLAG_ONLY_ARM: "Flag-only edit",
}
MANIPULATION_GATE_NAT = 4.0
BAND_LOW, BAND_HIGH = 5.0, 12.0
H1_REACH_MIN_FRAC = 14 / 16
H1_PARITY_BAND_NAT = 3.0
MATCHED_INSTALL_TRIGGER_NAT = 2.0
H2_MARGIN_NAT = 1.0
H2_ALPHA = 0.05
HINERT_CI_BOUND_NAT = 0.3
DV13_PASS_RHO = 0.7
DIAG_CENSOR_CIDS = ("binst_marker",)  # saturated diagonal, BOTH seeds, symmetric
SPREAD_FAMILIES = ("F1", "F2", "F4")
CONTAINED_FAMILIES = ("F3", "F7F8")
N_BOOT_POOLED = 10_000
N_BOOT_H3 = 2_000
RNG_SEED = 628
CHECKPOINT_SAVE_STEPS = 5  # dispatcher save_steps == band eval cadence
DV13_FAIL_ROUTING = (
    "DV1<->DV3 proxy validation NOT validated (FAIL, floor/ceiling collapse, or "
    "incomplete inputs): the grid headline is scoped as a teacher-forced slot-affinity "
    "result; behavioral leakage claims ship only from the DV3 on-policy subset "
    "(registered §6 fail-routing)"
)
MATCHED_INSTALL_SCOPE_NOTE = (
    "Matched-install trigger fired (arm-mean diagonal dials differ by > 2 nat): "
    "the matched-install claim is SCOPED to the checkpoint re-read columns "
    "(default + 4 negative); it does not relicense the 29-column headline."
)
REREAD_SELECTION_RULE = (
    "nearest-checkpoint: among the mismatched cell's saved band-trajectory probe "
    "records (probe steps are multiples of 5, aligned with the dispatcher's "
    "save_steps=5 checkpoint ladder), pick the step whose in-loop delta_nats is "
    "closest to the target dial (= the OTHER arm's final diagonal). The arm with "
    "the HIGHER dial is the re-read target (checkpoints only dial DOWN). Caveat: "
    "in-loop probes read TRAIN questions vs the offline EVAL-question diagonal; "
    "the §7 gate bounds that gap at +/-1 nat."
)


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        cwd=REPO,
        env=None,  # epm-lint: subprocess-env-inherit -- read-only git probe, no creds
    ).stdout.strip()


def _meta() -> dict:
    return {
        "git_commit": _git_commit(),
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "h2_primary_sep_mode": H2_PRIMARY_SEP_MODE,
    }


# ── Cell loading ─────────────────────────────────────────────────────────────


def _load_fresh_cells(g_root: Path) -> dict[tuple, dict]:
    """(arm, sep_mode, train_cid, eval_cid, seed) → cell record."""
    cells: dict[tuple, dict] = {}
    for arm_dir in sorted(p for p in g_root.glob("*") if p.is_dir()):
        for p in sorted(arm_dir.glob("*.json")):
            d = json.loads(p.read_text())
            key = (d["arm"], d["sep_mode"], d["train_cid"], d["eval_cid"], int(d["seed"]))
            assert key not in cells, f"duplicate cell key {key} ({p})"
            cells[key] = d
    return cells


def _load_reuse_cells(snapshot_dir: Path) -> dict[tuple, dict]:
    """Reused #537 grid cells (no-sep arm; sep_mode is canonically 'marker')."""
    cells: dict[tuple, dict] = {}
    for p in sorted(snapshot_dir.glob("*.json")):
        train_cid, eval_cid, seed_part = p.stem.split("__")
        seed = int(seed_part.removeprefix("seed"))
        d = json.loads(p.read_text())
        key = (REUSE_ARM, "marker", train_cid, eval_cid, seed)
        assert key not in cells, f"duplicate reuse key {key}"
        d.setdefault("arm", REUSE_ARM)
        d.setdefault("sep_mode", "marker")
        d.setdefault("seed", seed)
        cells[key] = d
    return cells


def _load_neg_columns(neg_root: Path) -> dict[tuple, dict]:
    cells: dict[tuple, dict] = {}
    arm_dir = neg_root / REUSE_ARM
    if not arm_dir.exists():
        return cells
    for p in sorted(arm_dir.glob("*.json")):
        d = json.loads(p.read_text())
        key = (REUSE_ARM, "marker", d["train_cid"], d["eval_cid"], int(d["seed"]))
        assert key not in cells, key
        cells[key] = d
    return cells


def _registry():
    from explore_persona_space.experiments.i537_contexts import load_registry

    return load_registry(REPO / "data/issue_537/contexts/sampled_contexts.json")


def _grid_eval_cids() -> list[str]:
    from explore_persona_space.experiments.i537_contexts import eval_cids_for

    return eval_cids_for("marker")


def _negative_cids() -> tuple[str, ...]:
    from explore_persona_space.experiments.i537_contexts import NEGATIVE_CIDS

    return NEGATIVE_CIDS


def _train_cids() -> list[str]:
    from explore_persona_space.experiments.i537_contexts import train_cids_for

    return train_cids_for("marker")


def primary_eval_cids(train_cid: str) -> list[str]:
    """The registered 29-column bystander set for one train context."""
    negs = set(_negative_cids())
    return [c for c in _grid_eval_cids() if c != train_cid and c not in negs]


# ── H2-side selection + masks ────────────────────────────────────────────────


def _arm_mode(arm: str, sep_mode: str) -> str:
    """Effective sep_mode key for an arm under a slot convention.

    No-sep arms only ever have 'marker' cells (their trained slot IS the
    canonical slot); sep arms have both, selected by the convention.
    """
    if arm in SEP_ARMS:
        return sep_mode
    return "marker"


def _cell(cells: dict, arm: str, train_cid: str, eval_cid: str, seed: int, sep_mode: str) -> dict:
    return cells[(arm, _arm_mode(arm, sep_mode), train_cid, eval_cid, seed)]


def _diag_delta(cells: dict, arm: str, train_cid: str, seed: int, sep_mode: str) -> float:
    return _cell(cells, arm, train_cid, train_cid, seed, sep_mode)["g_mean_delta_logp"]


def masked_rows(cells: dict, arm: str, train_cids: list[str], seeds: tuple, sep_mode: str) -> set:
    """(train_cid, seed) rows whose source diagonal misses the 4-nat
    manipulation gate in this arm (computed from the data, not hand-listed —
    #537's fmt_code seed-42 at 3.97 falls out of this rule automatically)."""
    out = set()
    for t in train_cids:
        for s in seeds:
            try:
                d = _diag_delta(cells, arm, t, s, sep_mode)
            except KeyError:
                continue
            if d < MANIPULATION_GATE_NAT:
                out.add((t, s))
    return out


def _assert_h2_keys(cells: dict, arms: list[str], train_cids: list[str], seeds, sep_mode) -> None:
    """Registered-filter integrity: after the sep_mode filter there is exactly
    ONE record per (arm, train_cid, eval_cid, seed)."""
    seen = set()
    for arm in arms:
        for t in train_cids:
            for e in primary_eval_cids(t):
                for s in seeds:
                    key = (arm, t, e, s)
                    assert key not in seen, f"duplicate H2 key {key}"
                    seen.add(key)
                    _cell(cells, arm, t, e, s, sep_mode)  # KeyError = missing cell, fail loud


# ── Statistics ───────────────────────────────────────────────────────────────


def _wilcoxon_greater(diffs: np.ndarray) -> dict:
    from scipy import stats

    if len(diffs) < 5 or np.allclose(diffs, 0):
        return {"n": len(diffs), "p_one_sided": None, "note": "insufficient n"}
    res = stats.wilcoxon(diffs, alternative="greater")
    return {"n": len(diffs), "statistic": float(res.statistic), "p_one_sided": float(res.pvalue)}


def _cluster_bootstrap_ci(
    values_by_cluster: dict, n_boot: int, rng: np.random.Generator, alpha: float = 0.05
) -> dict:
    """Resample clusters with replacement; CI on the mean of all member values."""
    clusters = sorted(values_by_cluster)
    means = []
    for _ in range(n_boot):
        draw = rng.choice(len(clusters), size=len(clusters), replace=True)
        vals = np.concatenate([np.atleast_1d(values_by_cluster[clusters[i]]) for i in draw])
        means.append(float(np.mean(vals)))
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = float(np.mean(np.concatenate([np.atleast_1d(v) for v in values_by_cluster.values()])))
    return {"mean": point, "ci_lo": float(lo), "ci_hi": float(hi), "n_clusters": len(clusters)}


def _seed_equal_weighted_bootstrap_ci(
    diffs_by_seed_cid: dict[int, dict[str, float]],
    n_boot: int,
    rng: np.random.Generator,
    alpha: float = 0.05,
) -> dict:
    """Seed-equal-weighted VARIANT: resample train_cids WITHIN each seed
    stratum, combine the per-seed means with equal weight. Reported alongside
    the registered cid-cluster pooled CI; NEVER the gate (the registered gate
    stays the both-seeds Wilcoxon conjunction + pooled mean)."""
    means = []
    per_seed_vals = {s: sorted(d.items()) for s, d in diffs_by_seed_cid.items()}
    for _ in range(n_boot):
        seed_means = []
        for _s, items in per_seed_vals.items():
            draw = rng.choice(len(items), size=len(items), replace=True)
            seed_means.append(float(np.mean([items[i][1] for i in draw])))
        means.append(float(np.mean(seed_means)))
    lo, hi = np.percentile(means, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    point = float(np.mean([np.mean(list(d.values())) for d in diffs_by_seed_cid.values()]))
    return {
        "mean": point,
        "ci_lo": float(lo),
        "ci_hi": float(hi),
        "n_seed_strata": len(diffs_by_seed_cid),
        "note": (
            "seed-equal-weighted VARIANT (cids resampled within each seed stratum, "
            "seed means equal-weighted) — reported alongside the registered cid-cluster "
            "pooled CI; the registered gate is unchanged"
        ),
    }


def h2_contrast(
    cells: dict,
    *,
    arm_a: str,
    arm_b: str,
    train_cids: list[str],
    seeds: tuple,
    sep_mode: str,
    mask_sep_mode: str | None = None,
    metric: str = "g_mean_delta_logp",
    rng: np.random.Generator,
) -> dict:
    """Per-(train_cid, seed) mean over the 29 primary columns; paired
    arm_a - arm_b per train context; one-sided Wilcoxon (a > b) per seed +
    pooled seed-stratified cluster bootstrap (cluster = train_cid).

    ``mask_sep_mode`` (default = ``sep_mode``) anchors the manipulation-gate
    row masks: the plain-slot sensitivity passes the PRIMARY slot here so its
    row set matches the primary read (the implant-took check is a property of
    the cell at its trained slot; the plain diagonal folds DV4 slot
    misalignment into the mask)."""
    mask_mode = sep_mode if mask_sep_mode is None else mask_sep_mode
    mask_a = masked_rows(cells, arm_a, train_cids, seeds, mask_mode)
    mask_b = masked_rows(cells, arm_b, train_cids, seeds, mask_mode)
    deleted = sorted(mask_a | mask_b)
    per_seed = {}
    diffs_by_cid: dict[str, list[float]] = {}
    diffs_by_seed_cid: dict[int, dict[str, float]] = {}
    row_means: dict[tuple, dict] = {}
    for s in seeds:
        diffs = []
        for t in train_cids:
            if (t, s) in mask_a or (t, s) in mask_b:
                continue
            cols = primary_eval_cids(t)
            a = float(np.mean([_cell(cells, arm_a, t, e, s, sep_mode)[metric] for e in cols]))
            b = float(np.mean([_cell(cells, arm_b, t, e, s, sep_mode)[metric] for e in cols]))
            diffs.append(a - b)
            diffs_by_cid.setdefault(t, []).append(a - b)
            diffs_by_seed_cid.setdefault(int(s), {})[t] = a - b
            row_means[(t, s)] = {"a": a, "b": b}
        arr = np.array(diffs)
        per_seed[str(s)] = {
            **_wilcoxon_greater(arr),
            "mean_paired_diff": float(np.mean(arr)) if len(arr) else None,
        }
    pooled = _cluster_bootstrap_ci(
        {t: np.array(v) for t, v in diffs_by_cid.items()}, N_BOOT_POOLED, rng
    )
    seed_eq = (
        _seed_equal_weighted_bootstrap_ci(diffs_by_seed_cid, N_BOOT_POOLED, rng)
        if diffs_by_seed_cid
        else None
    )
    both_sig = all(
        ps.get("p_one_sided") is not None and ps["p_one_sided"] < H2_ALPHA
        for ps in per_seed.values()
    )
    mean_all = pooled["mean"]
    if both_sig and mean_all >= H2_MARGIN_NAT:
        verdict = "PASS"
    elif both_sig and mean_all > 0:
        verdict = "detectable-below-registered-margin"
    elif mean_all < 0 and all(
        ps.get("p_one_sided") is not None and ps["p_one_sided"] > 1 - H2_ALPHA
        for ps in per_seed.values()
    ):
        verdict = "reverse-direction (revised rig leakier)"
    else:
        verdict = "null"
    return {
        "arm_a": arm_a,
        "arm_b": arm_b,
        "metric": metric,
        "sep_mode": sep_mode,
        "mask_sep_mode": mask_mode,
        "pairwise_deleted_rows": [list(x) for x in deleted],
        "per_seed": per_seed,
        "pooled_seed_stratified_bootstrap": pooled,
        "seed_equal_weighted_bootstrap_variant": seed_eq,
        "verdict": verdict,
        "per_row_means": {f"{t}__seed{s}": v for (t, s), v in row_means.items()},
    }


def transfer_fractions(
    cells: dict, arm: str, train_cids: list[str], seeds: tuple, sep_mode: str, min_denom: float
) -> dict:
    """Per-row mean bystander EOS-margin gain ÷ diagonal EOS-margin gain.

    The denominator is the EOS-margin source gain (registered — never the
    log-prob diagonal); rows under ``min_denom`` are FLAGGED, not dropped.
    """
    rows = {}
    flagged = []
    for t in train_cids:
        for s in seeds:
            try:
                denom = _cell(cells, arm, t, t, s, sep_mode)["g_mean_delta_eos_margin"]
            except KeyError:
                continue
            cols = primary_eval_cids(t)
            num = float(
                np.mean(
                    [_cell(cells, arm, t, e, s, sep_mode)["g_mean_delta_eos_margin"] for e in cols]
                )
            )
            entry = {"numerator_mean": num, "denominator": float(denom)}
            if abs(denom) < min_denom:
                flagged.append([t, s])
                entry["flag"] = f"denominator below min ({min_denom})"
            else:
                entry["fraction"] = num / denom
            rows[f"{t}__seed{s}"] = entry
    return {"rows": rows, "min_denominator": min_denom, "flagged_rows": flagged}


def h_inert(cells: dict, seeds: tuple, rng: np.random.Generator) -> dict:
    """Fresh full-revised mini-arm vs the matching #537 cells: paired-mean ΔΔG
    with cluster-by-adapter bootstrap CI (8 adapter pairs); sign test
    diagnostic-only (never a pass/fail clause)."""
    from scipy import stats

    pair_deltas: dict[str, np.ndarray] = {}
    all_cell_deltas = []
    for t in MINI_ARM_CIDS:
        for s in seeds:
            ds = []
            for e in _grid_eval_cids():
                try:
                    fresh = cells[(CANONICAL_ARM, "marker", t, e, s)]["g_mean_delta_logp"]
                    reuse = cells[(REUSE_ARM, "marker", t, e, s)]["g_mean_delta_logp"]
                except KeyError:
                    continue
                ds.append(fresh - reuse)
            if ds:
                pair_deltas[f"{t}__seed{s}"] = np.array(ds)
                all_cell_deltas.extend(ds)
    if not pair_deltas:
        return {"status": "no-data"}
    ci = _cluster_bootstrap_ci(pair_deltas, N_BOOT_POOLED, rng)
    ci_pass = ci["ci_lo"] >= -HINERT_CI_BOUND_NAT and ci["ci_hi"] <= HINERT_CI_BOUND_NAT
    pos = int(np.sum(np.array(all_cell_deltas) > 0))
    sign = stats.binomtest(pos, len(all_cell_deltas), 0.5)
    return {
        "n_adapter_pairs": len(pair_deltas),
        "cluster_bootstrap_ci": ci,
        "ci_bound_nat": HINERT_CI_BOUND_NAT,
        "ci_pass": bool(ci_pass),
        "sign_test_diagnostic_only": {
            "n_pos": pos,
            "n": len(all_cell_deltas),
            "p": float(sign.pvalue),
            "note": "DIAGNOSTIC ONLY -- never a pass/fail clause (registered)",
        },
    }


def grid_licensing(cells: dict, seeds: tuple, rng: np.random.Generator) -> dict:
    """Fresh-vs-fresh (Legacy vs Full-revised, 4 contexts) must agree in
    DIRECTION with the grid-level Legacy-vs-Reuse contrast."""
    ff = h2_contrast(
        cells,
        arm_a=LEGACY_ARM,
        arm_b=CANONICAL_ARM,
        train_cids=list(MINI_ARM_CIDS),
        seeds=seeds,
        sep_mode=H2_PRIMARY_SEP_MODE,
        rng=rng,
    )
    return {
        "fresh_vs_fresh_mean": ff["pooled_seed_stratified_bootstrap"]["mean"],
        "fresh_vs_fresh": ff,
    }


def h1_install(cells: dict, arms: list[str], train_cids: list[str], seeds: tuple) -> dict:
    """Band-reach counts (diagonal ΔG >= band floor counts overshoot as reach)
    + arm-mean diagonal dial under the symmetric binst-diagonal censoring."""
    out = {}
    dial_means = {}
    for arm in arms:
        per_seed = {}
        dials = []
        for s in seeds:
            reached, total, missed = 0, 0, []
            for t in train_cids:
                try:
                    d = _diag_delta(cells, arm, t, s, H2_PRIMARY_SEP_MODE)
                except KeyError:
                    continue
                total += 1
                if d >= BAND_LOW:
                    reached += 1
                else:
                    missed.append(t)
                if t not in DIAG_CENSOR_CIDS:
                    dials.append(d)
            per_seed[str(s)] = {
                "reached": reached,
                "total": total,
                "missed": missed,
                "pass": total > 0 and reached / total >= H1_REACH_MIN_FRAC,
            }
        out[arm] = per_seed
        dial_means[arm] = float(np.mean(dials)) if dials else None
    # Explicit H1 dial-parity read (±H1_PARITY_BAND_NAT, same symmetric
    # censoring): pairwise arm-mean dial gaps + the booleans the §6 text
    # references — the primary (Legacy vs reuse) pair named separately.
    with_data = [a for a in arms if dial_means.get(a) is not None]
    gaps = {
        f"{a}|{b}": abs(dial_means[a] - dial_means[b])
        for i, a in enumerate(with_data)
        for b in with_data[i + 1 :]
    }
    primary_key = f"{LEGACY_ARM}|{REUSE_ARM}"
    dial_parity = {
        "band_nat": H1_PARITY_BAND_NAT,
        "pairwise_abs_gaps_nat": gaps,
        "primary_pair": primary_key,
        "primary_pair_within_band": (
            gaps[primary_key] <= H1_PARITY_BAND_NAT if primary_key in gaps else None
        ),
        "all_pairs_within_band": (
            all(g <= H1_PARITY_BAND_NAT for g in gaps.values()) if gaps else None
        ),
    }
    return {
        "band": [BAND_LOW, BAND_HIGH],
        "reach": out,
        "arm_mean_diagonal_censored": dial_means,
        "diag_censor_cids": list(DIAG_CENSOR_CIDS),
        "h1_dial_parity": dial_parity,
    }


def dv1_dv3_validation(cells: dict, onpolicy_root: Path, seeds: tuple) -> dict:
    """Spearman over the exactly-enumerated matched OFF-DIAGONAL key set:
    {Legacy, Full-revised} x 4 cids x seeds x {default + 4 trained-negative}
    columns (80 cells at 2 seeds). Diagonals reported separately."""
    from scipy import stats

    neg_cids = list(_negative_cids())
    match_cols = ["default", *neg_cids]
    dv1, dv3, keys = [], [], []
    diag_pairs = []
    missing = []
    for arm in ONPOLICY_ARMS:
        for t in MINI_ARM_CIDS:
            for s in seeds:
                reads_p = onpolicy_root / f"{arm}_{t}_seed{s}" / "reads.json"
                if not reads_p.exists():
                    missing.append(str(reads_p))
                    continue
                summary = json.loads(reads_p.read_text())["summary"]
                for e in match_cols:
                    if e == t:
                        continue  # diagonal guard (binst_marker is never a column here)
                    try:
                        tf = _cell(cells, arm, t, e, s, H2_PRIMARY_SEP_MODE)["g_mean_delta_logp"]
                    except KeyError:
                        missing.append(f"G-cell {arm}/{t}->{e}/seed{s}")
                        continue
                    if e not in summary:
                        missing.append(f"onpolicy {arm}/{t}/seed{s} column {e}")
                        continue
                    dv1.append(tf)
                    dv3.append(summary[e]["mean_delta_logp"])
                    keys.append([arm, t, e, s])
                if t in summary and (arm, _arm_mode(arm, H2_PRIMARY_SEP_MODE), t, t, s) in cells:
                    diag_pairs.append(
                        {
                            "key": [arm, t, s],
                            "dv1": _cell(cells, arm, t, t, s, H2_PRIMARY_SEP_MODE)[
                                "g_mean_delta_logp"
                            ],
                            "dv3": summary[t]["mean_delta_logp"],
                        }
                    )
    expected = len(ONPOLICY_ARMS) * len(MINI_ARM_CIDS) * len(seeds) * len(match_cols)
    result = {
        "expected_n": expected,
        "realized_n": len(dv1),
        "missing_inputs": missing,
        "diagonals_reported_separately": diag_pairs,
    }
    if missing or len(dv1) != expected:
        # NO "pass" key here on purpose: incomplete inputs mean the proxy was
        # NOT validated. Claim routing fires on ``pass is not True`` so this
        # branch scopes the claim exactly like a FAIL.
        result["status"] = "incomplete-inputs"
        result["fail_routing"] = DV13_FAIL_ROUTING
        return result
    rho = stats.spearmanr(dv1, dv3)
    result.update(
        {
            "spearman_rho": float(rho.statistic),
            "p": float(rho.pvalue),
            "points": [
                {"key": k, "dv1": float(a), "dv3": float(b)}
                for k, a, b in zip(keys, dv1, dv3, strict=True)
            ],
            "pass": bool(rho.statistic >= DV13_PASS_RHO),
            "fail_routing": DV13_FAIL_ROUTING,
        }
    )
    return result


# ── Matched-install checkpoint re-read (registered §6 read-1 fallback) ──────


def _reread_columns() -> list[str]:
    """The registered re-read column set: ``default`` + the 4 trained negatives."""
    return ["default", *_negative_cids()]


def _load_trajectory_points(traj_p: Path) -> list[dict]:
    """Normalize a ``marker_band_trajectory_v1`` JSON to [{step, delta_nats}]."""
    d = json.loads(traj_p.read_text())
    recs = d.get("records") or []
    if recs:
        return [{"step": int(r["step"]), "delta_nats": float(r["delta_nats"])} for r in recs]
    steps, deltas = d.get("steps") or [], d.get("delta_nats") or []
    return [{"step": int(s), "delta_nats": float(x)} for s, x in zip(steps, deltas, strict=True)]


def _reuse_stop_step(snapshot_root: Path, train_cid: str, seed: int) -> int | None:
    sub = "stop_steps" if seed == 42 else f"stop_steps_seed{seed}"
    p = snapshot_root / sub / f"{train_cid}.json"
    if not p.exists():
        return None
    return int(json.loads(p.read_text())["stop_step"])


def matched_install_reread_spec(
    cells: dict,
    train_cids: list[str],
    seeds: tuple,
    traj_dir: Path,
    snapshot_root: Path,
) -> dict:
    """Machine-readable §6 fallback spec, emitted when the >2-nat trigger fires.

    Per (train_cid, seed) — diagonals censored per DIAG_CENSOR_CIDS, same
    symmetric censoring as the trigger — the arm with the HIGHER diagonal dial
    is the re-read target (checkpoints only dial DOWN); target dial = the
    other arm's diagonal; checkpoint = the saved band-trajectory probe step
    with delta nearest the target (REREAD_SELECTION_RULE). Fresh-arm cells
    resolve from ``eval_results/issue_628/p1/band_trajectories``; reuse-arm
    cells have NO #537 trajectories (snapshot carries stop_steps only) and get
    ``checkpoint_step: null`` + the candidate ladder — the dispatch consumer
    fails loud on null rather than guessing.
    """
    entries = []
    for t in train_cids:
        if t in DIAG_CENSOR_CIDS:
            continue
        for s in seeds:
            try:
                legacy = _diag_delta(cells, LEGACY_ARM, t, s, H2_PRIMARY_SEP_MODE)
                reuse = _diag_delta(cells, REUSE_ARM, t, s, H2_PRIMARY_SEP_MODE)
            except KeyError:
                continue
            gap = legacy - reuse
            mism, target = (LEGACY_ARM, reuse) if gap > 0 else (REUSE_ARM, legacy)
            entry = {
                "train_cid": t,
                "seed": int(s),
                "mismatched_arm": mism,
                "arm_kind": "fresh" if mism == LEGACY_ARM else "reuse",
                "mismatched_dial": legacy if mism == LEGACY_ARM else reuse,
                "target_dial": target,
                "dial_gap_nat": abs(gap),
                "columns": _reread_columns(),
                "sep_mode": H2_PRIMARY_SEP_MODE,
            }
            if mism == LEGACY_ARM:
                slug = f"{LEGACY_ARM}_{t}_seed{s}"
                entry["checkpoint_hf_subfolder"] = f"adapters/issue_628/{slug}"
                traj_p = traj_dir / f"{slug}.json"
                if traj_p.exists():
                    pts = _load_trajectory_points(traj_p)
                    best = min(pts, key=lambda p: abs(p["delta_nats"] - target))
                    entry["checkpoint_step"] = best["step"]
                    entry["checkpoint_delta_nats_inloop"] = best["delta_nats"]
                    entry["trajectory_points"] = pts
                else:
                    entry["checkpoint_step"] = None
                    entry["note"] = f"band trajectory missing at {traj_p}"
            else:
                entry["checkpoint_hf_subfolder"] = f"adapters/i537_marker_{t}_seed{s}"
                entry["checkpoint_step"] = None
                stop = _reuse_stop_step(snapshot_root, t, s)
                entry["candidate_steps"] = (
                    list(range(CHECKPOINT_SAVE_STEPS, stop + 1, CHECKPOINT_SAVE_STEPS))
                    if stop
                    else None
                )
                entry["note"] = (
                    "#537 snapshot carries stop_steps only (no band trajectories); "
                    "nearest-checkpoint selection needs a dial-sweep over candidate_steps "
                    "before the column reads — resolve checkpoint_step before dispatching"
                )
            entries.append(entry)
    return {
        **_meta(),
        "trigger_nat": MATCHED_INSTALL_TRIGGER_NAT,
        "selection_rule": REREAD_SELECTION_RULE,
        "checkpoint_save_steps": CHECKPOINT_SAVE_STEPS,
        "claim_scope": MATCHED_INSTALL_SCOPE_NOTE,
        "consumer": (
            "uv run python scripts/i628_dispatch.py --phase matched-install-reread "
            "--spec <this file> (post-hoc, on orchestrator-provided compute)"
        ),
        "entries": entries,
    }


def _load_reread_cells(reread_root: Path) -> dict[tuple, dict]:
    """(arm, train_cid, eval_cid, seed, checkpoint_step) → re-read cell record."""
    out: dict[tuple, dict] = {}
    if not reread_root.exists():
        return out
    for arm_dir in sorted(p for p in reread_root.glob("*") if p.is_dir()):
        for p in sorted(arm_dir.glob("*.json")):
            d = json.loads(p.read_text())
            key = (
                d["arm"],
                d["train_cid"],
                d["eval_cid"],
                int(d["seed"]),
                int(d["checkpoint_step"]),
            )
            assert key not in out, f"duplicate re-read cell key {key} ({p})"
            out[key] = d
    return out


def matched_install_read(
    cells: dict, neg_cells: dict, reread_cells: dict, spec_entries: list[dict]
) -> dict:
    """Ingest the §6 checkpoint re-read: pair each re-read cell (mismatched arm
    at its matched checkpoint) against the OTHER arm's final-adapter cell on the
    same (train_cid, eval_cid, seed). The claim is SCOPED to the re-read columns
    (default + 4 negative) and never relicenses the 29-column headline."""
    if not reread_cells:
        return {
            "status": "no-reread-cells",
            "note": (
                "trigger fired but no re-read cells found — dispatch the emitted "
                "matched_install_reread_spec.json, then re-run this analysis"
            ),
            "claim_scope": MATCHED_INSTALL_SCOPE_NOTE,
        }
    pool = {**cells, **neg_cells}
    rows, missing = [], []
    for ent in spec_entries:
        t, s, mism = ent["train_cid"], int(ent["seed"]), ent["mismatched_arm"]
        other = REUSE_ARM if mism == LEGACY_ARM else LEGACY_ARM
        step = ent.get("checkpoint_step")
        for e in ent["columns"]:
            rk = (mism, t, e, s, step) if step is not None else None
            if rk is None or rk not in reread_cells:
                missing.append([mism, t, e, s, step])
                continue
            try:
                other_val = _cell(pool, other, t, e, s, H2_PRIMARY_SEP_MODE)["g_mean_delta_logp"]
            except KeyError:
                missing.append([other, t, e, s, "final"])
                continue
            rr_val = reread_cells[rk]["g_mean_delta_logp"]
            legacy_val = rr_val if mism == LEGACY_ARM else other_val
            revised_val = other_val if mism == LEGACY_ARM else rr_val
            rows.append(
                {
                    "train_cid": t,
                    "eval_cid": e,
                    "seed": s,
                    "mismatched_arm": mism,
                    "checkpoint_step": step,
                    "reread_delta_logp": rr_val,
                    "other_arm_final_delta_logp": other_val,
                    "paired_diff_legacy_minus_revised": legacy_val - revised_val,
                }
            )
    if not rows:
        return {
            "status": "no-paired-rows",
            "missing": missing,
            "claim_scope": MATCHED_INSTALL_SCOPE_NOTE,
        }
    by_col: dict[str, list[float]] = {}
    for r in rows:
        by_col.setdefault(r["eval_cid"], []).append(r["paired_diff_legacy_minus_revised"])
    return {
        "status": "ok",
        "n_rows": len(rows),
        "missing": missing,
        "mean_paired_diff_legacy_minus_revised": float(
            np.mean([r["paired_diff_legacy_minus_revised"] for r in rows])
        ),
        "per_column_mean_paired_diff": {c: float(np.mean(v)) for c, v in sorted(by_col.items())},
        "claim_scope": MATCHED_INSTALL_SCOPE_NOTE,
        "rows": rows,
    }


def trained_negative_signature(cells: dict, neg_cells: dict, seeds: tuple) -> dict:
    """Restoring-force read: trained-negative columns vs family-matched
    holdout bystanders, per arm (below holdout under live-negative arms)."""
    registry = _registry()
    neg_cids = list(_negative_cids())
    holdout_by_family: dict[str, list[str]] = {}
    for c in _grid_eval_cids():
        ctx = registry[c]
        if ctx.role == "eval_holdout":
            holdout_by_family.setdefault(ctx.family, []).append(c)
    out = {}
    pool = {**cells, **neg_cells}
    arms = sorted({k[0] for k in pool})
    for arm in arms:
        rows = []
        for (a, m, t, e, s), d in pool.items():
            if a != arm or e not in neg_cids or s not in seeds:
                continue
            if m != _arm_mode(arm, H2_PRIMARY_SEP_MODE):
                continue  # registered primary slot only (skip plain duplicates)
            fam = registry[e].family
            matched = holdout_by_family.get(fam, [])
            ho_vals = []
            for h in matched:
                try:
                    ho_vals.append(
                        _cell(pool, arm, t, h, s, H2_PRIMARY_SEP_MODE)["g_mean_delta_logp"]
                    )
                except KeyError:
                    continue
            if not ho_vals:
                continue
            rows.append(
                {
                    "train_cid": t,
                    "neg_cid": e,
                    "seed": s,
                    "neg_delta_logp": d["g_mean_delta_logp"],
                    "family_matched_holdout_mean": float(np.mean(ho_vals)),
                    "below_holdout": bool(d["g_mean_delta_logp"] < float(np.mean(ho_vals))),
                }
            )
        if rows:
            out[arm] = {
                "live_negative_arm": arm in LIVE_NEG_ARMS,
                "frac_below_holdout": float(np.mean([r["below_holdout"] for r in rows])),
                "n_rows": len(rows),
                "rows": rows,
            }
    return out


def h3_family_interaction(
    cells: dict, train_cids: list[str], seeds: tuple, rng: np.random.Generator
) -> dict:
    """Exploratory rig x family interaction on off-diagonal means (cluster-by-
    train-cid bootstrap; instruction/default are singleton clusters — no
    strong family claim ships from this read, registered)."""
    registry = _registry()

    def fam_mean(arm: str, t: str, s: int, fams: tuple) -> float | None:
        vals = [
            _cell(cells, arm, t, e, s, H2_PRIMARY_SEP_MODE)["g_mean_delta_logp"]
            for e in primary_eval_cids(t)
            if registry[e].family in fams
        ]
        return float(np.mean(vals)) if vals else None

    per_cid: dict[str, list[float]] = {}
    for t in train_cids:
        vals = []
        for s in seeds:
            try:
                ls = fam_mean(LEGACY_ARM, t, s, SPREAD_FAMILIES)
                lc = fam_mean(LEGACY_ARM, t, s, CONTAINED_FAMILIES)
                rs = fam_mean(REUSE_ARM, t, s, SPREAD_FAMILIES)
                rc = fam_mean(REUSE_ARM, t, s, CONTAINED_FAMILIES)
            except KeyError:
                continue
            if None in (ls, lc, rs, rc):
                continue
            vals.append((ls - rs) - (lc - rc))
        if vals:
            per_cid[t] = vals
    if not per_cid:
        return {"status": "no-data"}
    ci = _cluster_bootstrap_ci({k: np.array(v) for k, v in per_cid.items()}, N_BOOT_H3, rng)
    return {
        "interaction_spread_minus_contained": ci,
        "spread_families": list(SPREAD_FAMILIES),
        "contained_families": list(CONTAINED_FAMILIES),
        "status": "EXPLORATORY (registered: no strong family-interaction claim ships)",
    }


# ── Figures ──────────────────────────────────────────────────────────────────


def make_figures(cells: dict, h2: dict, dv13: dict, figures_dir: Path, seeds: tuple) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    registry = _registry()
    made = []

    # Hero: paired per-train-context off-diagonal leakage, Legacy vs Revised.
    fams = sorted({registry[t].family for t in _train_cids()})
    fam_colors = dict(zip(fams, paper_palette(len(fams)), strict=False))
    fig, axes = plt.subplots(1, len(seeds), figsize=(4.2 * len(seeds), 3.6), squeeze=False)
    for ax, s in zip(axes[0], seeds, strict=True):
        for key, v in h2["per_row_means"].items():
            t, seed_part = key.split("__seed")
            if int(seed_part) != s:
                continue
            color = fam_colors[registry[t].family]
            ax.plot([0, 1], [v["b"], v["a"]], marker="o", ms=3, lw=0.8, color=color, alpha=0.8)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["Revised rig\n(#537 reuse)", "Legacy rig"])
        ax.set_ylabel("Mean bystander log-prob gain (nats)")
        ax.set_title(f"Seed {s}")
    fig.suptitle("Off-diagonal marker leakage per training context (29 bystander columns)")
    fig.tight_layout()
    savefig_paper(fig, "hero_paired_offdiagonal_leakage", dir=figures_dir)
    plt.close(fig)
    made.append("hero_paired_offdiagonal_leakage")

    # Exploratory: per-arm heatmaps (one per arm x seed where data exists).
    arms_present = sorted({k[0] for k in cells})
    for arm in arms_present:
        for s in seeds:
            ts = sorted({k[2] for k in cells if k[0] == arm and k[4] == s and k[1] == "marker"})
            if not ts:
                continue
            es = sorted({k[3] for k in cells if k[0] == arm and k[4] == s and k[1] == "marker"})
            mat = np.full((len(ts), len(es)), np.nan)
            for i, t in enumerate(ts):
                for j, e in enumerate(es):
                    d = cells.get((arm, "marker", t, e, s))
                    if d:
                        mat[i, j] = d["g_mean_delta_logp"]
            fig, ax = plt.subplots(figsize=(max(6, len(es) * 0.28), max(3, len(ts) * 0.28)))
            im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-5, vmax=25)
            ax.set_xticks(range(len(es)))
            ax.set_xticklabels(
                [registry[e].name if e in registry else e for e in es], rotation=90, fontsize=4
            )
            ax.set_yticks(range(len(ts)))
            ax.set_yticklabels([registry[t].name if t in registry else t for t in ts], fontsize=5)
            fig.colorbar(im, ax=ax, label="Log-prob gain (nats)")
            ax.set_title(f"{ARM_LABELS.get(arm, arm)} — seed {s} (own trained slot)")
            savefig_paper(fig, f"heatmap_{arm}_seed{s}", dir=figures_dir)
            plt.close(fig)
            made.append(f"heatmap_{arm}_seed{s}")

    # DV1 vs DV3 validation scatter (when the matched set is complete).
    if dv13.get("spearman_rho") is not None:
        fig, ax = plt.subplots(figsize=(4, 4))
        xs = [p["dv1"] for p in dv13["points"]]
        ys = [p["dv3"] for p in dv13["points"]]
        ax.scatter(xs, ys, s=12, alpha=0.7)
        ax.set_xlabel("Teacher-forced slot read (nats, trained - base)")
        ax.set_ylabel("On-policy own-response read (nats, trained - base)")
        ax.set_title(f"Proxy validation (Spearman rho = {dv13['spearman_rho']:.2f})")
        fig.tight_layout()
        savefig_paper(fig, "dv1_vs_dv3_validation", dir=figures_dir)
        plt.close(fig)
        made.append("dv1_vs_dv3_validation")
    return made


# ── main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--eval-root", type=Path, default=REPO / "eval_results/issue_628")
    ap.add_argument(
        "--reuse-cells-dir",
        type=Path,
        default=REPO / "eval_results/issue_628/inputs/i537_marker/G_cells_marker",
    )
    ap.add_argument("--figures-dir", type=Path, default=REPO / "figures/issue_628")
    ap.add_argument(
        "--reread-cells-dir",
        type=Path,
        default=None,
        help="matched-install re-read G-cells (default: <eval-root>/matched_install_reread)",
    )
    ap.add_argument(
        "--seeds", type=lambda x: tuple(int(v) for v in x.split(",")), default=(42, 1042)
    )
    ap.add_argument("--min-denominator", type=float, default=1.0)
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    rng = np.random.default_rng(RNG_SEED)
    cells = _load_fresh_cells(args.eval_root / "G_cells")
    cells.update(_load_reuse_cells(args.reuse_cells_dir))
    neg_cells = _load_neg_columns(args.eval_root / "neg_columns")
    train_cids = _train_cids()
    seeds = args.seeds
    logger.info("loaded %d grid cells + %d reuse-neg cells", len(cells), len(neg_cells))

    _assert_h2_keys(cells, [LEGACY_ARM, REUSE_ARM], train_cids, seeds, H2_PRIMARY_SEP_MODE)

    h2 = h2_contrast(
        cells,
        arm_a=LEGACY_ARM,
        arm_b=REUSE_ARM,
        train_cids=train_cids,
        seeds=seeds,
        sep_mode=H2_PRIMARY_SEP_MODE,
        rng=rng,
    )
    # Named sensitivity artifact: the same statistic at the canonical plain
    # slot (folds the slot-misalignment quantity in; biases toward null).
    # Row masks stay anchored at the PRIMARY (own-trained-slot) diagonals so
    # the sensitivity compares the SAME row set as the primary read.
    h2_plain = h2_contrast(
        cells,
        arm_a=LEGACY_ARM,
        arm_b=REUSE_ARM,
        train_cids=train_cids,
        seeds=seeds,
        sep_mode="plain",
        mask_sep_mode=H2_PRIMARY_SEP_MODE,
        rng=rng,
    )
    tf_legacy = transfer_fractions(
        cells, LEGACY_ARM, train_cids, seeds, H2_PRIMARY_SEP_MODE, args.min_denominator
    )
    tf_reuse = transfer_fractions(
        cells, REUSE_ARM, train_cids, seeds, H2_PRIMARY_SEP_MODE, args.min_denominator
    )
    # H2 robustness: one-sided Wilcoxon on the per-row transfer fractions.
    frac_diffs = []
    for key, a in tf_legacy["rows"].items():
        b = tf_reuse["rows"].get(key)
        if b and "fraction" in a and "fraction" in b:
            frac_diffs.append(a["fraction"] - b["fraction"])
    tf_wilcoxon = _wilcoxon_greater(np.array(frac_diffs))

    h1 = h1_install(
        cells,
        [LEGACY_ARM, REUSE_ARM, CANONICAL_ARM, SEP_ONLY_ARM, FLAG_ONLY_ARM],
        train_cids,
        seeds,
    )
    hin = h_inert(cells, seeds, rng)
    lic = grid_licensing(cells, seeds, rng)
    dv13 = dv1_dv3_validation(cells, args.eval_root / "bystander_onpolicy", seeds)
    tns = trained_negative_signature(cells, neg_cells, seeds)
    h3 = h3_family_interaction(cells, train_cids, seeds, rng)

    # ── Precedence / claim routing (registered) ──────────────────────────────
    dials = h1["arm_mean_diagonal_censored"]
    matched_install_trigger = (
        dials.get(LEGACY_ARM) is not None
        and dials.get(REUSE_ARM) is not None
        and abs(dials[LEGACY_ARM] - dials[REUSE_ARM]) > MATCHED_INSTALL_TRIGGER_NAT
    )
    grid_mean = h2["pooled_seed_stratified_bootstrap"]["mean"]
    ff_mean = lic["fresh_vs_fresh_mean"]
    direction_agrees = (grid_mean > 0) == (ff_mean > 0)
    selectivity_agrees = tf_wilcoxon.get("p_one_sided") is not None and (
        np.mean(frac_diffs) > 0
    ) == (grid_mean > 0)

    # Registered §6 read-1 automation: on trigger, emit the machine-readable
    # re-read spec and ingest any re-read cells already produced by
    # `i628_dispatch.py --phase matched-install-reread`.
    reread_dir = args.reread_cells_dir or (args.eval_root / "matched_install_reread")
    reread_cells = _load_reread_cells(reread_dir)
    reread_spec = None
    mi_read: dict = {"status": "not-triggered"}
    if matched_install_trigger:
        reread_spec = matched_install_reread_spec(
            cells,
            train_cids,
            seeds,
            args.eval_root / "p1/band_trajectories",
            args.reuse_cells_dir.parent,
        )
        mi_read = matched_install_read(cells, neg_cells, reread_cells, reread_spec["entries"])
    elif reread_cells:
        mi_read = {
            "status": "reread-cells-present-but-trigger-not-fired",
            "note": f"{len(reread_cells)} re-read cells under {reread_dir} ignored",
        }

    scope_notes = []
    if not hin.get("ci_pass", False):
        scope_notes.append(
            "H-inert CI bound FAILED: #537 cells cannot stand in for the canonical rig; "
            "headline falls back to Legacy vs Full-revised on the 4-context subset "
            "(fresh-vs-fresh) and the reuse demotes to a secondary near-replication arm "
            "(pre-registered §7 fallback)."
        )
    if not direction_agrees:
        scope_notes.append(
            "Grid-licensing direction check FAILED: the grid read demotes to descriptive; "
            "the fresh-vs-fresh subset is the headline (registered)."
        )
    if matched_install_trigger:
        note = MATCHED_INSTALL_SCOPE_NOTE
        if mi_read.get("status") == "ok":
            note += " Re-read cells ingested (see matched_install_read)."
        else:
            note += " Re-read pending (spec emitted: analysis/matched_install_reread_spec.json)."
        scope_notes.append(note)
    if not selectivity_agrees:
        scope_notes.append(
            "Raw vs transfer-fraction reads DISAGREE in direction: raw leakage differs; "
            "selectivity not established; dose/schedule remains unresolved (registered)."
        )
    # Registered routing fires on NOT-validated (FAIL **or** incomplete inputs):
    # only an explicit pass licenses the proxy.
    if dv13.get("pass") is not True:
        scope_notes.append(dv13.get("fail_routing", DV13_FAIL_ROUTING))
    final_claim_scope = (
        "full H2 headline licensed (grid + selectivity + reuse-licensing + proxy checks pass)"
        if not scope_notes
        else " | ".join(scope_notes)
    )

    out = {
        **_meta(),
        "registered": {
            "h2_primary_sep_mode": H2_PRIMARY_SEP_MODE,
            "manipulation_gate_nat": MANIPULATION_GATE_NAT,
            "band": [BAND_LOW, BAND_HIGH],
            "h2_margin_nat": H2_MARGIN_NAT,
            "h1_parity_band_nat": H1_PARITY_BAND_NAT,
            "matched_install_trigger_nat": MATCHED_INSTALL_TRIGGER_NAT,
            "h_inert_ci_bound_nat": HINERT_CI_BOUND_NAT,
            "dv13_pass_rho": DV13_PASS_RHO,
            "diag_censor_cids": list(DIAG_CENSOR_CIDS),
            "mask_table_note": (
                "binst_marker diagonal censored BOTH seeds symmetrically (H1 dial + "
                "matched-install trigger only; off-diagonals stay in H2); H2 rows "
                "pairwise-deleted when the source diagonal misses the 4-nat gate in "
                "EITHER arm (inherits #537 fmt_code seed-42); fmt_code seed-1042 at "
                "4.28 nat is an H1 band miss, NOT an H2 row mask."
            ),
        },
        "h2_primary": h2,
        "h2_transfer_fraction": {
            "legacy": tf_legacy,
            "reuse": tf_reuse,
            "wilcoxon_legacy_gt_reuse": tf_wilcoxon,
        },
        "h1_install_parity": h1,
        "h_inert": hin,
        "grid_licensing": {
            **lic,
            "grid_mean": grid_mean,
            "direction_agrees": bool(direction_agrees),
        },
        "dv1_dv3_validation": dv13,
        "trained_negative_signature": tns,
        "h3_family_interaction": h3,
        "matched_install_reread_required": bool(matched_install_trigger),
        "matched_install_read": mi_read,
        "final_claim_scope": final_claim_scope,
    }

    analysis_dir = args.eval_root / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if reread_spec is not None:
        spec_p = analysis_dir / "matched_install_reread_spec.json"
        spec_p.write_text(json.dumps(reread_spec, indent=1))
        logger.info("matched-install re-read spec written: %s", spec_p)
    (analysis_dir / "rig_contrast.json").write_text(json.dumps(out, indent=1))
    (analysis_dir / "h2_plain_slot_sensitivity.json").write_text(
        json.dumps({**_meta(), "h2_plain_slot": h2_plain}, indent=1)
    )
    logger.info("analysis written: %s", analysis_dir / "rig_contrast.json")
    logger.info("final_claim_scope: %s", final_claim_scope)

    if not args.skip_figures:
        made = make_figures(cells, h2, dv13, args.figures_dir, seeds)
        logger.info("figures: %s", made)
    return 0


if __name__ == "__main__":
    sys.exit(main())
