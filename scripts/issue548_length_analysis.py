#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, ※, Δ) in scientific docstrings + log lines.
"""Issue #548 — off-pod primary read: length-nuisance analysis at the 1024 cap.

Pre-registered primary read for the corrective re-run of #540's
length-censoring confound (plan §4 code change 2). Runs on the VM, CPU-only,
against committed artifacts AFTER pod termination. Per strip (ordinary full
n=256 / off-diagonal n=240 / no-D5 off-diagonal n=210) it computes:

1. The #540 length-nuisance protocol, now primary: length-alone ρ; raw ρ for
   the new canonical RB JS; BOTH partial directions (JS|length and length|JS)
   in BOTH conventions (Spearman-of-rank-residuals = figure convention,
   primary; Pearson-on-rank-residuals = analysis-JSON convention, companion);
   entanglement ρ(JS, length); BOTH normalization variants (per-token primary;
   un-normalized recovered exactly as kl_side_m_bits_per_token × n_positions).
2. Kill-bearing bootstrap CIs (seed 42, --n-boot reps) for the partials
   themselves, in BOTH flavors per the pre-registered kill-bearing CI rule:
   iid-cell resampling AND clustered by unordered context pair (mirrored
   cells carried together; 136 clusters on the full ordinary strip). A
   `dead`/`revives` verdict requires the two flavors to AGREE on the
   zero-exclusion call; disagreement → indeterminate. Clustered is quoted as
   primary; degenerate/skipped resample fractions are reported for both.
3. Truncation manipulation check: per-context + per-pair truncation rates at
   the new cap, side-by-side with the parent's 256-cap values; the
   pre-registered gate statistic = median per-pair truncation over the 120
   unique ordinary–ordinary pairs; reply-length distributions per context.
4. Leaderboard bookkeeping (folded-in #540 proposal 3): the |Δ mean reply
   length| column and a stacked z(base_prior) + z(length) combined column
   (the parent's combined_* construction, incl. its divergence-polarity
   quirk — labeled bookkeeping, never cited as evidence), ρ + 1000-rep
   bootstrap CIs on union / ordinary / instructed.
5. Machine-readable `cap_censoring_verdict ∈ {dead, alive, conditional_kill,
   indeterminate}` per the pre-registered total verdict mapping (plan §4
   item 5 precedence; §7). No analyzer narration may override it.
6. Exploratory cross-cap extras: per-pair JS@new vs JS@parent, length-diff
   cross-cap, aggregated position profiles, windowed first-256 re-read from
   the new draws' position_profile, masked-JS companion, tercile collinearity
   companion.

The three statistic conventions (partial_spearman, partial_pearson_on_ranks,
paired_boot_delta) are lifted verbatim from
scripts/issue540_length_nuisance_supplement.py so the parent's published
numbers reproduce exactly when --new-dir is pointed at the parent artifacts.

CLI (plan §10 post-termination analysis):
    uv run python scripts/issue548_length_analysis.py \\
        --new-dir eval_results/issue_548 --parent-dir eval_results/issue_540 \\
        --dv-dir eval_results/issue_532/per_cell/loc_ep1 \\
        --out eval_results/issue_548/length_analysis.json \\
        --figures-dir figures/issue_548 --seed 42 --n-boot 10000
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import platform
import subprocess
import sys
import zlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np  # noqa: E402
from scipy.stats import pearsonr, rankdata, spearmanr  # noqa: E402

logger = logging.getLogger("issue548.length_analysis")

SCHEMA_VERSION = "issue548_length_analysis_v1"
MARKER_TOKEN_ID = 83399  # " ※" (leading space)
LEADERBOARD_N_BOOT = 1000  # pre-registered for the bookkeeping rows (plan §4 item 4)
GATE_TRUNC_CONDITIONAL_KILL = 0.50  # plan §7
GATE_TRUNC_CLEAN = 0.20  # plan §7
ENTANGLEMENT_GATE = 0.60  # plan §4 collinearity note


# ── Conventions lifted VERBATIM from issue540_length_nuisance_supplement.py ─


def partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """Figure-script convention: Spearman of OLS rank-residuals (re-ranks residuals)."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        design = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(design, a, rcond=None)
        return a - design @ coef

    rho, p = spearmanr(resid(rx, rz), resid(ry, rz))
    return float(rho), float(p)


def partial_pearson_on_ranks(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> tuple[float, float]:
    """analysis_jsrb.json convention: Pearson correlation of OLS rank-residuals."""
    rx, ry, rz = rankdata(x), rankdata(y), rankdata(z)

    def resid(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        design = np.vstack([b, np.ones_like(b)]).T
        coef, *_ = np.linalg.lstsq(design, a, rcond=None)
        return a - design @ coef

    r, p = pearsonr(resid(rx, rz), resid(ry, rz))
    return float(r), float(p)


def paired_boot_delta(
    x1: np.ndarray, x2: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int
) -> dict:
    """Paired bootstrap CI for |rho(x1,y)| - |rho(x2,y)| over the same cells."""
    n = len(y)
    point = abs(spearmanr(x1, y)[0]) - abs(spearmanr(x2, y)[0])
    deltas = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(set(y[idx])) < 2:
            continue
        deltas.append(abs(spearmanr(x1[idx], y[idx])[0]) - abs(spearmanr(x2[idx], y[idx])[0]))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "delta_abs_rho_point": float(point),
        "ci95": [float(lo), float(hi)],
        "n_boot": len(deltas),
        "n_cells": n,
    }


# ── Reproducibility metadata ────────────────────────────────────────────────


def _git_commit() -> str:
    """Current commit SHA of the checkout this script runs from."""
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _metadata(args: argparse.Namespace) -> dict:
    """Standard reproducibility block (CLAUDE.md Code Style)."""
    return {
        "schema_version": SCHEMA_VERSION,
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.datetime.now(datetime.UTC).replace(tzinfo=None).isoformat() + "Z",
        "python_version": platform.python_version(),
        "numpy_version": np.__version__,
        "args": {
            "new_dir": str(args.new_dir),
            "parent_dir": str(args.parent_dir),
            "dv_dir": str(args.dv_dir),
            "out": str(args.out),
            "figures_dir": str(args.figures_dir),
            "seed": args.seed,
            "n_boot": args.n_boot,
        },
        "conventions": {
            "partial_figure": "Spearman of OLS rank-residuals (re-ranks residuals); PRIMARY",
            "partial_analysis_json": "Pearson of OLS rank-residuals; companion",
            "normalization_primary": "per-token (kl_side_m_bits_per_token mean)",
            "normalization_companion": "un-normalized total bits per reply = "
            "kl_side_m_bits_per_token × n_positions per per-sample record; diagonal = 0",
            "kill_bearing_ci_rule": "BOTH iid-cell and unordered-pair-clustered CIs per "
            "kill-bearing partial; dead/revives require both flavors to agree on the "
            "zero-exclusion call; disagreement → indeterminate; clustered quoted primary",
            "length_feature": "abs(mean n_positions side a − side b) per pair, incl. the "
            "appended terminator (same construction as the parent supplement)",
        },
    }


# ── Loaders ─────────────────────────────────────────────────────────────────


def _rng(seed: int, label: str) -> np.random.Generator:
    """Deterministic per-statistic RNG: stable under adding/removing targets."""
    return np.random.default_rng([seed, zlib.crc32(label.encode())])


def load_predictors(d: Path) -> dict:
    """Load predictors_jsrb.json → sources, bystanders, cell matrices, base prior."""
    p = json.loads((d / "predictors_jsrb.json").read_text())
    srcs, bys = p["sources"], p["bystanders"]
    mats = {
        "js_rb": np.array(p["js_rb_matrix"]),
        "js_rb_masked": np.array(p["js_rb_masked_matrix"]),
        "js_v1": np.array(p["js_v1_matrix"]),
        "gauss_kl": np.array(p["gauss_kl_matrix"]),
        "cosine": np.array(p["cosine_matrix"]),
    }
    return {"sources": srcs, "bystanders": bys, "mats": mats, "base_prior": p["base_prior"]}


def load_emission(dv_dir: Path) -> dict[tuple[str, str], float]:
    """DV reused byte-for-byte: in_R_emission_rate per (source, bystander) cell."""
    emis: dict[tuple[str, str], float] = {}
    for f in sorted(dv_dir.glob("cell_loc_ep1_*.json")):
        a, b = f.name.split("cell_loc_ep1_")[-1][: -len(".json")].split("__")
        emis[(a, b)] = json.loads(f.read_text())["summary"]["in_R_emission_rate"]
    if not emis:
        raise FileNotFoundError(f"no cell_loc_ep1_*.json DV files under {dv_dir}")
    return emis


def load_pairs(d: Path) -> dict:
    """Aggregate the per_pair JSONs of one run into pair-level + context-level reads.

    Returns pair-keyed dicts (keys = (a, b) in file order) for: mean reply
    length per side, truncation rate, un-normalized RB JS, per-token RB JS,
    windowed first-256 per-token JS, position profiles; plus per-context
    deduped reply-length rows (deduped by (context, probe_idx, sample_idx) —
    Phase S draws once per context, Phase T reuses the draws across pairs).
    """
    per_pair_dir = d / "per_pair"
    files = sorted(per_pair_dir.glob("pair_*.json"))
    if not files:
        raise FileNotFoundError(f"no pair_*.json under {per_pair_dir}")
    pairlen: dict[tuple[str, str], tuple[float, float]] = {}
    pairtrunc: dict[tuple[str, str], float] = {}
    js_unnorm: dict[tuple[str, str], float] = {}
    js_pair: dict[tuple[str, str], float] = {}
    js_win256: dict[tuple[str, str], float] = {}
    profiles: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]] = {}
    ctx_rows: dict[str, dict[tuple[int, int], tuple[int, bool]]] = {}
    selfpairs: list[tuple[str, str]] = []
    for f in files:
        rec = json.loads(f.read_text())
        a, b = rec["pair"]["a"], rec["pair"]["b"]
        if rec.get("is_selfpair") or a == b:
            selfpairs.append((a, b))
            continue
        ps = rec["per_sample"]
        la = float(np.mean([r["n_positions"] for r in ps if r["side"] == "a"]))
        lb = float(np.mean([r["n_positions"] for r in ps if r["side"] == "b"]))
        pairlen[(a, b)] = (la, lb)
        pairtrunc[(a, b)] = rec["truncation"]["n_truncated"] / rec["truncation"]["n_rows"]
        ta = [r["kl_side_m_bits_per_token"] * r["n_positions"] for r in ps if r["side"] == "a"]
        tb = [r["kl_side_m_bits_per_token"] * r["n_positions"] for r in ps if r["side"] == "b"]
        js_unnorm[(a, b)] = float(0.5 * (np.mean(ta) + np.mean(tb)))
        js_pair[(a, b)] = float(rec["js_rb_bits"])
        prof_sum = np.array(rec["position_profile"]["js_bits_sum"], dtype=np.float64)
        prof_cnt = np.array(rec["position_profile"]["count"], dtype=np.float64)
        profiles[(a, b)] = (prof_sum, prof_cnt)
        w = min(256, len(prof_sum))
        denom = float(prof_cnt[:w].sum())
        js_win256[(a, b)] = float(prof_sum[:w].sum() / denom) if denom > 0 else float("nan")
        for r in ps:
            ctx = a if r["side"] == "a" else b
            ctx_rows.setdefault(ctx, {})[(r["probe_idx"], r["sample_idx"])] = (
                int(r["n_positions"]),
                bool(r["truncated"]),
            )
    return {
        "pairlen": pairlen,
        "pairtrunc": pairtrunc,
        "js_unnorm": js_unnorm,
        "js_pair": js_pair,
        "js_win256": js_win256,
        "profiles": profiles,
        "ctx_rows": ctx_rows,
        "selfpairs": selfpairs,
    }


def _pairget(d: dict, x: str, y: str, diag_val: float = 0.0) -> float:
    """Unordered-pair lookup with diagonal default (parent supplement logic)."""
    if x == y:
        return diag_val
    return d[(x, y)] if (x, y) in d else d[(y, x)]


def _dlen(pairlen: dict, x: str, y: str) -> float:
    """|Δ mean reply length| per cell (parent supplement construction)."""
    if x == y:
        return 0.0
    la, lb = pairlen[(x, y)] if (x, y) in pairlen else pairlen[(y, x)][::-1]
    return abs(la - lb)


# ── Bootstrap machinery for the kill-bearing partials ───────────────────────


def _zero_call(ci: list[float]) -> str:
    """'neg' / 'pos' when the 95% CI excludes zero on that side, else 'span'."""
    lo, hi = ci
    if hi < 0:
        return "neg"
    if lo > 0:
        return "pos"
    return "span"


def boot_ci_partial(
    partial_fn,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    rng: np.random.Generator,
    n_boot: int,
    clusters: np.ndarray | None = None,
) -> dict:
    """Bootstrap CI for partial_fn(x, y | z) by cell or cluster resampling.

    Cell flavor resamples cells iid; cluster flavor resamples unordered
    context pairs (mirrored cells carried together). Resamples with a
    degenerate DV (constant y) are skipped; NaN partials are dropped; both
    fractions are reported (kill-bearing CI rule, plan §4 item 2).
    """
    vals: list[float] = []
    skipped = 0
    if clusters is None:
        n = len(y)
        for _ in range(n_boot):
            idx = rng.integers(0, n, size=n)
            if len(set(y[idx])) < 2:
                skipped += 1
                continue
            vals.append(partial_fn(x[idx], y[idx], z[idx])[0])
        n_clusters = None
    else:
        uniq = np.unique(clusters)
        members = {c: np.where(clusters == c)[0] for c in uniq}
        for _ in range(n_boot):
            picked = rng.integers(0, len(uniq), size=len(uniq))
            idx = np.concatenate([members[uniq[p]] for p in picked])
            if len(set(y[idx])) < 2:
                skipped += 1
                continue
            vals.append(partial_fn(x[idx], y[idx], z[idx])[0])
        n_clusters = len(uniq)
    arr = np.array(vals, dtype=np.float64)
    finite = arr[np.isfinite(arr)]
    degenerate = int(len(arr) - len(finite))
    if len(finite) == 0:
        ci = [float("nan"), float("nan")]
    else:
        lo, hi = np.percentile(finite, [2.5, 97.5])
        ci = [float(lo), float(hi)]
    return {
        "point": float(partial_fn(x, y, z)[0]),
        "ci95": ci,
        "zero_call": _zero_call(ci) if np.isfinite(ci[0]) else "degenerate",
        "n_boot_requested": int(n_boot),
        "n_boot_used": len(finite),
        "skipped_fraction": float(skipped / n_boot),
        "degenerate_fraction": float(degenerate / n_boot),
        "n_clusters": n_clusters,
    }


def kill_bearing_ci(
    label: str,
    partial_fn,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
    clusters: np.ndarray,
    seed: int,
    n_boot: int,
) -> dict:
    """Both CI flavors + the combined flavor-agreement call for one partial."""
    iid = boot_ci_partial(partial_fn, x, y, z, _rng(seed, label + "/iid"), n_boot)
    clu = boot_ci_partial(
        partial_fn, x, y, z, _rng(seed, label + "/cluster"), n_boot, clusters=clusters
    )
    agree = iid["zero_call"] == clu["zero_call"]
    return {
        "iid": iid,
        "clustered": clu,
        "flavors_agree": bool(agree),
        "call": clu["zero_call"] if agree else "flavor_disagree",
    }


def _combined_call(variant_calls: dict[str, str]) -> str:
    """Combine convention/normalization variant calls per the verdict mapping.

    Returns the shared call when every variant agrees (each already requiring
    iid/cluster flavor agreement); 'flavor_disagree' when any variant's two
    flavors disagree; 'convention_disagree' when variants disagree among
    themselves (single-convention-only call → indeterminate, plan §4 item 5.4).
    """
    calls = list(variant_calls.values())
    if any(c == "flavor_disagree" for c in calls):
        return "flavor_disagree"
    if len(set(calls)) == 1:
        return calls[0]
    return "convention_disagree"


# ── Strip statistics ────────────────────────────────────────────────────────


def _cluster_ids(cells: list[tuple[str, str]]) -> np.ndarray:
    """Canonical unordered-pair id per cell; diagonal cells are their own unit."""
    return np.array([f"{min(a, b)}__{max(a, b)}" for a, b in cells])


def strip_stats(
    cells: list[tuple[str, str]],
    mask: np.ndarray,
    cols: dict[str, np.ndarray],
    y: np.ndarray,
    seed: int,
    n_boot: int,
) -> dict:
    """Full pre-registered read for one strip (plan §4 code-change-2 items 1-2, 6)."""
    m = mask
    ym = y[m]
    xd = cols["length_diff"][m]
    clusters = _cluster_ids([c for c, keep in zip(cells, m, strict=True) if keep])
    res: dict = {"n": int(m.sum()), "n_clusters": len(np.unique(clusters))}
    r, p = spearmanr(xd, ym)
    res["length_alone"] = {"rho": float(r), "p": float(p)}

    # Raw + partials (both conventions) + entanglement for every predictor column.
    for name in [
        "js_rb",
        "js_rb_unnormalized",
        "js_rb_windowed_first256",
        "js_rb_masked",
        "js_rb_parent_cap",
        "js_v1",
        "gauss_kl",
        "cosine",
    ]:
        xm = cols[name][m]
        rr, rp = spearmanr(xm, ym)
        pf, pfp = partial_spearman(xm, ym, xd)
        pa, pap = partial_pearson_on_ranks(xm, ym, xd)
        res[name] = {
            "raw_rho": float(rr),
            "raw_p": float(rp),
            "partial_length_figure_convention": {"rho": pf, "p": pfp},
            "partial_length_analysis_convention": {"rho": pa, "p": pap},
            "rho_with_length": float(spearmanr(xm, xd)[0]),
        }
    # Reverse partials (length controlled for predictor), both conventions.
    for name in ["js_rb", "js_rb_unnormalized", "gauss_kl"]:
        xm = cols[name][m]
        rv, rvp = partial_spearman(xd, ym, xm)
        ra, rap = partial_pearson_on_ranks(xd, ym, xm)
        res[f"length_partial_{name}_figure_convention"] = {"rho": rv, "p": rvp}
        res[f"length_partial_{name}_analysis_convention"] = {"rho": ra, "p": rap}

    # Kill-bearing bootstrap CIs — JS|length and length|JS, three variants each
    # (figure/per-token PRIMARY; analysis/per-token + figure/un-normalized
    # companions), both flavors per variant (kill-bearing CI rule).
    xj = cols["js_rb"][m]
    xu = cols["js_rb_unnormalized"][m]
    kb: dict = {"js_partial_length": {}, "length_partial_js": {}}
    kb["js_partial_length"]["figure_pertoken"] = kill_bearing_ci(
        "js|len/fig/pt", partial_spearman, xj, ym, xd, clusters, seed, n_boot
    )
    kb["js_partial_length"]["analysis_pertoken"] = kill_bearing_ci(
        "js|len/ana/pt", partial_pearson_on_ranks, xj, ym, xd, clusters, seed, n_boot
    )
    kb["js_partial_length"]["figure_unnormalized"] = kill_bearing_ci(
        "js|len/fig/unnorm", partial_spearman, xu, ym, xd, clusters, seed, n_boot
    )
    kb["length_partial_js"]["figure_pertoken"] = kill_bearing_ci(
        "len|js/fig/pt", partial_spearman, xd, ym, xj, clusters, seed, n_boot
    )
    kb["length_partial_js"]["analysis_pertoken"] = kill_bearing_ci(
        "len|js/ana/pt", partial_pearson_on_ranks, xd, ym, xj, clusters, seed, n_boot
    )
    kb["length_partial_js"]["figure_unnormalized"] = kill_bearing_ci(
        "len|js/fig/unnorm", partial_spearman, xd, ym, xu, clusters, seed, n_boot
    )
    for direction in kb:
        kb[direction]["combined_call"] = _combined_call(
            {v: kb[direction][v]["call"] for v in kb[direction] if v != "combined_call"}
        )
    res["kill_bearing_ci"] = kb

    # Comparator partial CIs for the hero leaderboard error bars (both flavors).
    res["comparator_partial_ci"] = {
        name: kill_bearing_ci(
            f"{name}|len/fig/pt", partial_spearman, cols[name][m], ym, xd, clusters, seed, n_boot
        )
        for name in ["js_rb_parent_cap", "js_v1", "gauss_kl"]
    }

    # Paired |ρ| bootstrap length vs js_rb (the parent's construction).
    res["paired_boot_abs_rho_length_minus_js_rb"] = paired_boot_delta(
        xd, xj, ym, _rng(seed, "paired/len-js"), n_boot
    )

    # Tercile collinearity companion (direction check only, plan §4 note).
    order = np.argsort(xd, kind="stable")
    thirds = np.array_split(order, 3)
    terc = []
    for t_idx in thirds:
        if len(t_idx) >= 3 and len(set(ym[t_idx])) > 1:
            terc.append(float(spearmanr(xj[t_idx], ym[t_idx])[0]))
        else:
            terc.append(float("nan"))
    res["tercile_companion"] = {
        "rho_js_emission_by_length_tercile": terc,
        "middle_tercile_rho": terc[1],
        "tercile_sizes": [len(t) for t in thirds],
    }
    return res


# ── Verdict mapping (pre-registered; plan §4 item 5 / §7) ───────────────────


def cap_censoring_verdict(strip: dict, gate_truncation: float) -> dict:
    """The total verdict mapping, evaluated in precedence order.

    1. truncation > 50% → conditional_kill (governs even a revive-shaped
       partial — then exploratory only);
    2. truncation ≤ 20% AND JS-partial CIs exclude zero NEGATIVE (combined
       call per the kill-bearing CI rule) AND the collinearity companion
       survives → alive;
    3. truncation ≤ 20% AND JS-partial CIs span zero AND reverse partials
       exclude zero → dead;
    4. EVERYTHING ELSE → indeterminate (20-50% band, positive-side exclusion,
       both-null, flavor disagreement, single-convention-only calls).
    """
    kb = strip["kill_bearing_ci"]
    js_call = kb["js_partial_length"]["combined_call"]
    rev_call = kb["length_partial_js"]["combined_call"]
    entangle = strip["js_rb"]["rho_with_length"]
    middle = strip["tercile_companion"]["middle_tercile_rho"]
    if entangle <= ENTANGLEMENT_GATE:
        companion_pass = True
        companion_reason = f"entanglement {entangle:.3f} <= {ENTANGLEMENT_GATE} — reads cleanly"
    elif np.isfinite(middle) and middle < 0:
        companion_pass = True
        companion_reason = (
            f"entanglement {entangle:.3f} > {ENTANGLEMENT_GATE}; middle-tercile "
            f"rho {middle:.3f} < 0 — direction survives"
        )
    else:
        companion_pass = False
        companion_reason = (
            f"entanglement {entangle:.3f} > {ENTANGLEMENT_GATE}; middle-tercile "
            f"rho {middle} not negative — companion fails"
        )
    if gate_truncation > GATE_TRUNC_CONDITIONAL_KILL:
        verdict = "conditional_kill"
        reason = (
            f"gate truncation {gate_truncation:.3f} > {GATE_TRUNC_CONDITIONAL_KILL} — "
            "panel verbosity is the binding factor; route to the length-diverse-panel "
            "follow-up, do NOT raise the cap"
        )
    elif gate_truncation <= GATE_TRUNC_CLEAN and js_call == "neg" and companion_pass:
        verdict = "alive"
        reason = (
            "JS-partial excludes zero on the negative side in every convention/"
            "normalization variant with both CI flavors agreeing, truncation gate "
            f"{gate_truncation:.3f} <= {GATE_TRUNC_CLEAN}, collinearity companion survives"
        )
    elif gate_truncation <= GATE_TRUNC_CLEAN and js_call == "span" and rev_call in ("neg", "pos"):
        verdict = "dead"
        reason = (
            "JS-partial spans zero and the reverse length-partial excludes zero in every "
            "variant with both CI flavors agreeing, truncation gate "
            f"{gate_truncation:.3f} <= {GATE_TRUNC_CLEAN} — cap-censoring explanation dead"
        )
    else:
        verdict = "indeterminate"
        reason = (
            f"no pre-registered branch fires (gate truncation {gate_truncation:.3f}, "
            f"js_call={js_call}, reverse_call={rev_call}, companion_pass={companion_pass})"
        )
    return {
        "verdict": verdict,
        "reason": reason,
        "inputs": {
            "gate_truncation_median_ordinary_pairs": float(gate_truncation),
            "js_partial_combined_call": js_call,
            "reverse_partial_combined_call": rev_call,
            "entanglement_rho_js_length": float(entangle),
            "collinearity_companion_pass": bool(companion_pass),
            "collinearity_companion_reason": companion_reason,
        },
    }


# ── Leaderboard bookkeeping (union/ordinary/instructed; plan §4 item 4) ────


def _zcol(v: np.ndarray) -> np.ndarray:
    """The parent's combined-column standardization (exact same formula)."""
    return (v - np.nanmean(v)) / (np.nanstd(v) + 1e-12)


def _boot_spearman_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator, n_boot: int) -> dict:
    """Plain iid percentile-bootstrap CI for Spearman ρ (bookkeeping rows)."""
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if len(set(y[idx])) < 2:
            continue
        vals.append(spearmanr(x[idx], y[idx])[0])
    arr = np.array(vals, dtype=np.float64)
    arr = arr[np.isfinite(arr)]
    rho = float(spearmanr(x, y)[0])
    if len(arr) == 0:
        return {"rho": rho, "ci95": [float("nan"), float("nan")], "n_boot": 0}
    lo, hi = np.percentile(arr, [2.5, 97.5])
    return {"rho": rho, "ci95": [float(lo), float(hi)], "n_boot": len(arr)}


def leaderboard_bookkeeping(
    union_cells: list[tuple[str, str]],
    sources: list[str],
    base_prior: dict[str, float],
    length_diff: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> dict:
    """length_diff + stacked z(base_prior)+z(length) leaderboard rows.

    The combined column follows the parent's combined_* construction exactly
    (z-scored over the union panel, simple sum) INCLUDING its divergence-
    polarity quirk: base prior correlates positively with emission while
    divergence-flavored columns correlate negatively, so the stack partially
    cancels — labeled bookkeeping, never cited as evidence (parent analyzer
    note, carried verbatim).
    """
    src_set = set(sources)
    bp = np.array([base_prior[c] for _, c in union_cells])
    combined = _zcol(bp) + _zcol(length_diff)
    is_instr = np.array([c not in src_set for _, c in union_cells])
    masks = {
        "union": np.ones(len(union_cells), dtype=bool),
        "ordinary": ~is_instr,
        "instructed": is_instr,
    }
    out: dict = {
        "note": "bookkeeping only — combined column inherits the parent's "
        "divergence-polarity quirk; never cited as evidence",
        "n_boot": LEADERBOARD_N_BOOT,
    }
    for col_name, col in [
        ("length_diff", length_diff),
        ("combined_base_prior_plus_length", combined),
    ]:
        out[col_name] = {}
        for strip_name, m in masks.items():
            stats = _boot_spearman_ci(
                col[m], y[m], _rng(seed, f"lb/{col_name}/{strip_name}"), LEADERBOARD_N_BOOT
            )
            stats["n"] = int(m.sum())
            out[col_name][strip_name] = stats
    return out


# ── Truncation / length manipulation check ─────────────────────────────────


def _ctx_length_stats(ctx_rows: dict) -> dict:
    """Per-context reply-length distribution + truncation from deduped draws."""
    out = {}
    for ctx, rows in sorted(ctx_rows.items()):
        lens = np.array([n for n, _ in rows.values()], dtype=np.float64)
        truncs = np.array([t for _, t in rows.values()], dtype=bool)
        out[ctx] = {
            "n_draws": len(lens),
            "truncation_rate": float(truncs.mean()),
            "mean": float(lens.mean()),
            "median": float(np.median(lens)),
            "sd": float(lens.std()),
            "max": int(lens.max()),
            "quantiles_p10_p25_p50_p75_p90": [
                float(q) for q in np.percentile(lens, [10, 25, 50, 75, 90])
            ],
        }
    return out


def _pair_trunc_medians(pairtrunc: dict, sources: list[str]) -> dict:
    """Gate statistics: median per-pair truncation per pre-registered strip."""
    src_set = set(sources)
    ordinary = [v for (a, b), v in pairtrunc.items() if a in src_set and b in src_set]
    no_d5 = [
        v for (a, b), v in pairtrunc.items() if a in src_set and b in src_set and "D5" not in (a, b)
    ]
    all_pairs = list(pairtrunc.values())
    return {
        "median_ordinary_ordinary_pairs": float(np.median(ordinary)) if ordinary else None,
        "n_ordinary_ordinary_pairs": len(ordinary),
        "median_all_pairs": float(np.median(all_pairs)),
        "n_all_pairs": len(all_pairs),
        "median_no_d5_ordinary_pairs": float(np.median(no_d5)) if no_d5 else None,
        "n_no_d5_ordinary_pairs": len(no_d5),
    }


def _samples_truncation(d: Path) -> dict:
    """Per-context truncation_rate from samples_*.json when committed locally.

    The parent's samples live only on HF (not in git), so this read is
    recorded-as-absent rather than fatal; the per-pair-derived per-context
    rates above are the always-available primary carrier.
    """
    samples_dir = d / "samples"
    files = sorted(samples_dir.glob("samples_*.json")) if samples_dir.is_dir() else []
    if not files:
        return {"available": False, "reason": f"no samples_*.json under {samples_dir}"}
    rates = {}
    for f in files:
        payload = json.loads(f.read_text())
        ctx = f.name[len("samples_") : -len(".json")]
        rates[ctx] = float(payload["truncation_rate"])
    return {"available": True, "per_context_truncation_rate": rates}


# ── Figures ─────────────────────────────────────────────────────────────────


def make_figures(
    figures_dir: Path,
    cells: list[tuple[str, str]],
    masks: dict[str, np.ndarray],
    cols: dict[str, np.ndarray],
    y: np.ndarray,
    strips: dict,
    new_ctx_stats: dict,
    parent_ctx_stats: dict,
    new_pairs: dict,
    parent_pairs: dict,
    new_cap: int,
    parent_supplement: dict | None,
) -> list[str]:
    """Hero figures 1-2 + the exploratory dump (plan §6)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style("blog")
    figures_dir.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    def _save(fig, stem: str) -> None:
        paths = savefig_paper(fig, stem, dir=figures_dir)
        written.extend(str(p) for p in paths.values())
        plt.close(fig)

    colors = paper_palette(6)

    # 1. HERO: truncation before/after per context (the manipulation check).
    ctxs = sorted(set(new_ctx_stats) | set(parent_ctx_stats))
    xs = np.arange(len(ctxs))
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    ax.bar(
        xs - 0.2,
        [parent_ctx_stats.get(c, {}).get("truncation_rate", np.nan) for c in ctxs],
        width=0.4,
        color=colors[0],
        label="parent cap (256)",
    )
    ax.bar(
        xs + 0.2,
        [new_ctx_stats.get(c, {}).get("truncation_rate", np.nan) for c in ctxs],
        width=0.4,
        color=colors[1],
        label=f"new cap ({new_cap})",
    )
    ax.set_xticks(xs)
    ax.set_xticklabels([c.replace("instr_", "") for c in ctxs], rotation=45, ha="right")
    ax.set_ylabel("Fraction of replies truncated at the cap")
    ax.set_title("Sampling truncation per context, before vs after the cap lift")
    ax.legend()
    _save(fig, "truncation_before_after")

    # 2. HERO: length-controlled leaderboard on the ordinary full strip.
    sf = strips["ordinary_full"]
    rows = [
        ("Reply-length diff (alone)", sf["length_alone"]["rho"], None, None),
        (
            f"Canonical JS @{new_cap}",
            sf["js_rb"]["raw_rho"],
            sf["js_rb"]["partial_length_figure_convention"]["rho"],
            sf["kill_bearing_ci"]["js_partial_length"]["figure_pertoken"]["clustered"]["ci95"],
        ),
        (
            "Canonical JS @256 (parent)",
            sf["js_rb_parent_cap"]["raw_rho"],
            sf["js_rb_parent_cap"]["partial_length_figure_convention"]["rho"],
            sf["comparator_partial_ci"]["js_rb_parent_cap"]["clustered"]["ci95"],
        ),
        (
            "Activation Gaussian KL",
            sf["gauss_kl"]["raw_rho"],
            sf["gauss_kl"]["partial_length_figure_convention"]["rho"],
            sf["comparator_partial_ci"]["gauss_kl"]["clustered"]["ci95"],
        ),
        (
            "First-token JS (v1)",
            sf["js_v1"]["raw_rho"],
            sf["js_v1"]["partial_length_figure_convention"]["rho"],
            sf["comparator_partial_ci"]["js_v1"]["clustered"]["ci95"],
        ),
    ]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    xs = np.arange(len(rows))
    ax.bar(xs - 0.2, [r[1] for r in rows], width=0.4, color=colors[2], label="raw ρ")
    part_vals = [r[2] if r[2] is not None else np.nan for r in rows]
    ax.bar(xs + 0.2, part_vals, width=0.4, color=colors[3], label="length-partialled ρ")
    for k, r in enumerate(rows):
        if r[3] is not None and np.isfinite(r[3][0]):
            ax.errorbar(
                xs[k] + 0.2,
                r[2],
                yerr=[[max(0.0, r[2] - r[3][0])], [max(0.0, r[3][1] - r[2])]],
                fmt="none",
                ecolor="black",
                capsize=3,
            )
    ax.axhline(0.0, color="grey", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], rotation=20, ha="right")
    ax.set_ylabel("Spearman ρ vs marker emission (ordinary strip, n=256)")
    ax.set_title("Raw vs length-partialled ρ (clustered 95% CI on the partials)")
    ax.legend()
    _save(fig, "length_controlled_leaderboard")

    # 3. Reply-length distributions per context at the new cap.
    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    data = [
        [q for q in new_ctx_stats[c]["quantiles_p10_p25_p50_p75_p90"]]
        for c in ctxs
        if c in new_ctx_stats
    ]
    present = [c for c in ctxs if c in new_ctx_stats]
    meds = [new_ctx_stats[c]["median"] for c in present]
    p10 = [d[0] for d in data]
    p90 = [d[4] for d in data]
    xs2 = np.arange(len(present))
    ax.errorbar(
        xs2,
        meds,
        yerr=[np.array(meds) - np.array(p10), np.array(p90) - np.array(meds)],
        fmt="o",
        color=colors[4],
        capsize=3,
        label="median (p10–p90)",
    )
    ax.axhline(256, color="grey", ls="--", lw=0.8, label="old cap (256)")
    ax.axhline(new_cap, color="black", ls=":", lw=0.8, label=f"new cap ({new_cap})")
    ax.set_xticks(xs2)
    ax.set_xticklabels([c.replace("instr_", "") for c in present], rotation=45, ha="right")
    ax.set_ylabel("Reply length (tokens, incl. terminator)")
    ax.set_title("Reply-length distribution per context at the lifted cap")
    ax.legend()
    _save(fig, "reply_length_per_context")

    # 4. Cross-cap stability: js_rb new vs parent per pair (exploratory).
    shared = [k for k in new_pairs["js_pair"] if _haspair(parent_pairs["js_pair"], k)]
    if shared:
        src_set = {a for a, _ in cells} | {b for _, b in cells}
        ord_mask = [a in src_set and b in src_set for a, b in shared]
        xnew = np.array([new_pairs["js_pair"][k] for k in shared])
        xpar = np.array([_pairget(parent_pairs["js_pair"], *k) for k in shared])
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        for is_ord, label, c in [(True, "ordinary–ordinary", 0), (False, "with instructed", 1)]:
            sel = np.array(ord_mask) == is_ord
            if sel.any():
                ax.scatter(xpar[sel], xnew[sel], s=14, color=colors[c], label=label, alpha=0.7)
        lim = max(float(xnew.max()), float(xpar.max())) * 1.05
        ax.plot([0, lim], [0, lim], color="grey", lw=0.8)
        ax.set_xlabel("Canonical JS @256 (parent, bits/token)")
        ax.set_ylabel(f"Canonical JS @{new_cap} (bits/token)")
        ax.set_title("Per-pair estimator stability across the cap change")
        ax.legend()
        _save(fig, "js_rb_cross_cap_scatter")

    # 5. js_rb@new vs emission, ordinary vs instructed colored (cell level).
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    m_ord = masks["ordinary_full"]
    ax.scatter(cols["js_rb"][m_ord], y[m_ord], s=14, color=colors[0], label="ordinary", alpha=0.7)
    ax.scatter(
        cols["js_rb"][~m_ord], y[~m_ord], s=14, color=colors[1], label="instructed", alpha=0.7
    )
    ax.set_xlabel(f"Canonical JS @{new_cap} (bits/token)")
    ax.set_ylabel("Marker emission rate (trained model, on-policy)")
    ax.set_title("Divergence vs emission at the lifted cap")
    ax.legend()
    _save(fig, "js_rb_vs_emission_scatter")

    # 6. Per-position mean JS profile out to the new cap, parent window overlaid.
    fig, ax = plt.subplots(figsize=(8.0, 4.2))
    for pairs, label, c in [(new_pairs, f"new @{new_cap}", 0), (parent_pairs, "parent @256", 1)]:
        agg_sum = None
        agg_cnt = None
        for s, n in pairs["profiles"].values():
            if agg_sum is None:
                agg_sum, agg_cnt = s.copy(), n.copy()
            else:
                length = max(len(agg_sum), len(s))
                agg_sum = np.pad(agg_sum, (0, length - len(agg_sum))) + np.pad(
                    s, (0, length - len(s))
                )
                agg_cnt = np.pad(agg_cnt, (0, length - len(agg_cnt))) + np.pad(
                    n, (0, length - len(n))
                )
        if agg_sum is not None:
            with np.errstate(invalid="ignore", divide="ignore"):
                prof = np.where(agg_cnt > 0, agg_sum / np.maximum(agg_cnt, 1), np.nan)
            ax.plot(np.arange(len(prof)), prof, color=colors[c], label=label, lw=1.0)
    ax.axvline(256, color="grey", ls="--", lw=0.8)
    ax.set_xlabel("Response token position")
    ax.set_ylabel("Mean per-position JS (bits)")
    ax.set_title("Position profile of divergence across the cap change (all pairs)")
    ax.legend()
    _save(fig, "position_profile_cross_cap")

    # 7. Length-diff cross-cap scatter (exploratory).
    shared_len = [k for k in new_pairs["pairlen"] if _haspair(parent_pairs["pairlen"], k)]
    if shared_len:
        dn = np.array([abs(np.subtract(*new_pairs["pairlen"][k])) for k in shared_len])
        dp = np.array(
            [
                abs(
                    np.subtract(
                        *(
                            parent_pairs["pairlen"][k]
                            if k in parent_pairs["pairlen"]
                            else parent_pairs["pairlen"][(k[1], k[0])][::-1]
                        )
                    )
                )
                for k in shared_len
            ]
        )
        fig, ax = plt.subplots(figsize=(5.2, 5.2))
        ax.scatter(dp, dn, s=14, color=colors[5], alpha=0.7)
        ax.set_xlabel("|Δ mean reply length| @256 (parent, tokens)")
        ax.set_ylabel(f"|Δ mean reply length| @{new_cap} (tokens)")
        ax.set_title("Length feature before vs after the cap lift (per pair)")
        _save(fig, "length_diff_cross_cap_scatter")

    # 8. Entanglement before/after bars per strip.
    fig, ax = plt.subplots(figsize=(6.0, 4.0))
    strip_names = ["ordinary_full", "ordinary_offdiag", "ordinary_no_d5_offdiag"]
    new_e = [strips[s]["js_rb"]["rho_with_length"] for s in strip_names]
    if parent_supplement is not None:
        par_keys = {
            "ordinary_full": "ordinary_full",
            "ordinary_offdiag": "ordinary_offdiag",
            "ordinary_no_d5_offdiag": "ordinary_no_d5_offdiag",
        }
        par_e = [
            parent_supplement.get(par_keys[s], {}).get("js_rb", {}).get("rho_with_length", np.nan)
            for s in strip_names
        ]
    else:
        par_e = [np.nan] * len(strip_names)
    xs3 = np.arange(len(strip_names))
    ax.bar(xs3 - 0.2, par_e, width=0.4, color=colors[0], label="parent @256")
    ax.bar(xs3 + 0.2, new_e, width=0.4, color=colors[1], label=f"new @{new_cap}")
    ax.set_xticks(xs3)
    ax.set_xticklabels(["full", "off-diag", "no-D5 off-diag"])
    ax.set_ylabel("ρ(canonical JS, |Δ length|)")
    ax.set_title("JS–length entanglement across the cap change")
    ax.legend()
    _save(fig, "entanglement_before_after")

    return written


def _haspair(d: dict, k: tuple[str, str]) -> bool:
    """True when the unordered pair k exists in pair-keyed dict d."""
    return k in d or (k[1], k[0]) in d


# ── Main ────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__.splitlines()[4],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--new-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_548")
    parser.add_argument("--parent-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_540")
    parser.add_argument(
        "--dv-dir", type=Path, default=PROJECT_ROOT / "eval_results/issue_532/per_cell/loc_ep1"
    )
    parser.add_argument(
        "--out", type=Path, default=PROJECT_ROOT / "eval_results/issue_548/length_analysis.json"
    )
    parser.add_argument("--figures-dir", type=Path, default=PROJECT_ROOT / "figures/issue_548")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-boot", type=int, default=10_000)
    parser.add_argument(
        "--skip-figures", action="store_true", help="JSON only (figures need matplotlib)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)

    pred_new = load_predictors(args.new_dir)
    pred_parent = load_predictors(args.parent_dir)
    if pred_new["sources"] != pred_parent["sources"] or (
        pred_new["bystanders"] != pred_parent["bystanders"]
    ):
        raise ValueError("new/parent predictor panels disagree — not the same 416-cell panel")
    srcs, bys = pred_new["sources"], pred_new["bystanders"]
    emis = load_emission(args.dv_dir)
    new_pairs = load_pairs(args.new_dir)
    parent_pairs = load_pairs(args.parent_dir)
    new_cap = _infer_cap(new_pairs)
    logger.info(
        "loaded panels: %d sources × %d bystanders, %d new pairs, %d parent pairs, cap=%d",
        len(srcs),
        len(bys),
        len(new_pairs["pairtrunc"]),
        len(parent_pairs["pairtrunc"]),
        new_cap,
    )

    # Cell columns over the ordinary panel (sources × sources, 256 cells).
    ordinary = [(s, c) for s in srcs for c in srcs]
    s_idx = {s: i for i, s in enumerate(srcs)}
    b_idx = {b: j for j, b in enumerate(bys)}

    def cellcol(mat: np.ndarray, cells: list[tuple[str, str]]) -> np.ndarray:
        return np.array([mat[s_idx[a], b_idx[b]] for a, b in cells])

    y_ord = np.array([emis[c] for c in ordinary])
    cols = {
        "length_diff": np.array([_dlen(new_pairs["pairlen"], *c) for c in ordinary]),
        "js_rb": cellcol(pred_new["mats"]["js_rb"], ordinary),
        "js_rb_masked": cellcol(pred_new["mats"]["js_rb_masked"], ordinary),
        "js_rb_parent_cap": cellcol(pred_parent["mats"]["js_rb"], ordinary),
        "js_v1": cellcol(pred_new["mats"]["js_v1"], ordinary),
        "gauss_kl": cellcol(pred_new["mats"]["gauss_kl"], ordinary),
        "cosine": cellcol(pred_new["mats"]["cosine"], ordinary),
        "js_rb_unnormalized": np.array([_pairget(new_pairs["js_unnorm"], *c) for c in ordinary]),
        "js_rb_windowed_first256": np.array(
            [_pairget(new_pairs["js_win256"], *c) for c in ordinary]
        ),
    }
    is_diag = np.array([a == b for a, b in ordinary])
    no_d5 = np.array([(not d) and ("D5" not in c) for c, d in zip(ordinary, is_diag, strict=True)])
    masks = {
        "ordinary_full": np.ones(len(ordinary), dtype=bool),
        "ordinary_offdiag": ~is_diag,
        "ordinary_no_d5_offdiag": no_d5,
    }

    strips = {}
    for name, m in masks.items():
        logger.info("strip %s (n=%d): computing pre-registered read ...", name, int(m.sum()))
        strips[name] = strip_stats(ordinary, m, cols, y_ord, args.seed, args.n_boot)

    # Truncation manipulation check + gate statistic.
    new_gate = _pair_trunc_medians(new_pairs["pairtrunc"], srcs)
    parent_gate = _pair_trunc_medians(parent_pairs["pairtrunc"], srcs)
    new_ctx = _ctx_length_stats(new_pairs["ctx_rows"])
    parent_ctx = _ctx_length_stats(parent_pairs["ctx_rows"])
    gate_value = new_gate["median_ordinary_ordinary_pairs"]
    if gate_value is None:
        raise ValueError("no ordinary–ordinary pairs found — cannot evaluate the gate statistic")

    # Verdict: primary on ordinary_full (plan §6), per-strip calls alongside.
    verdicts = {name: cap_censoring_verdict(strips[name], gate_value) for name in strips}
    primary = verdicts["ordinary_full"]
    strip_disagreement = len({v["verdict"] for v in verdicts.values()}) > 1

    # Leaderboard bookkeeping on the full 416-cell union panel.
    union_cells = [(s, c) for s in srcs for c in bys]
    y_union = np.array([emis[c] for c in union_cells])
    len_union = np.array([_dlen(new_pairs["pairlen"], *c) for c in union_cells])
    bookkeeping = leaderboard_bookkeeping(
        union_cells, srcs, pred_new["base_prior"], len_union, y_union, args.seed
    )

    # Cross-cap exploratory dump (per-pair table; engine-config change
    # confounds the deltas — exploratory only, never a gate; plan §6 note).
    cross_cap = []
    for k in sorted(new_pairs["js_pair"]):
        if not _haspair(parent_pairs["js_pair"], k):
            continue
        a, b = k
        cross_cap.append(
            {
                "pair": f"{a}__{b}",
                "ordinary_ordinary": a in s_idx and b in s_idx,
                "js_rb_new": new_pairs["js_pair"][k],
                "js_rb_parent": _pairget(parent_pairs["js_pair"], a, b),
                "length_diff_new": _dlen(new_pairs["pairlen"], a, b),
                "length_diff_parent": _dlen(parent_pairs["pairlen"], a, b),
                "truncation_new": new_pairs["pairtrunc"][k],
                "truncation_parent": _pairget(parent_pairs["pairtrunc"], a, b, diag_val=np.nan),
            }
        )

    parent_supplement = None
    supp_path = args.parent_dir / "length_nuisance_supplement.json"
    if supp_path.exists():
        parent_supplement = json.loads(supp_path.read_text())

    out = {
        "metadata": _metadata(args),
        "cap_censoring_verdict": primary["verdict"],
        "verdict": primary,
        "verdict_by_strip": {k: v["verdict"] for k, v in verdicts.items()},
        "strip_disagreement": bool(strip_disagreement),
        "truncation": {
            "inferred_new_cap": new_cap,
            "gate_statistic": "median per-pair truncation over the unique "
            "ordinary–ordinary pairs (pre-registered, plan §7)",
            "new": new_gate,
            "parent": parent_gate,
            "per_context_new": new_ctx,
            "per_context_parent": parent_ctx,
            "samples_files_new": _samples_truncation(args.new_dir),
            "samples_files_parent": _samples_truncation(args.parent_dir),
        },
        "strips": strips,
        "leaderboard_bookkeeping": bookkeeping,
        "cross_cap_exploratory": {
            "note": "engine-config change confounds per-pair deltas — exploratory only",
            "per_pair": cross_cap,
        },
    }

    figures_written: list[str] = []
    if not args.skip_figures:
        figures_written = make_figures(
            args.figures_dir,
            ordinary,
            masks,
            cols,
            y_ord,
            strips,
            new_ctx,
            parent_ctx,
            new_pairs,
            parent_pairs,
            new_cap,
            parent_supplement,
        )
    out["figures_written"] = figures_written

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1))
    logger.info("wrote %s (verdict: %s)", args.out, primary["verdict"])
    print(
        json.dumps(
            {
                "cap_censoring_verdict": primary["verdict"],
                "gate_truncation": gate_value,
                "verdict_inputs": primary["inputs"],
                "out": str(args.out),
                "n_figures": len(figures_written),
            },
            indent=1,
        )
    )
    return 0


def _infer_cap(pairs: dict) -> int:
    """Infer the sampling cap from the position-profile arrays (cap = len − 1)."""
    lengths = {len(s) for s, _ in pairs["profiles"].values()}
    return max(lengths) - 1


if __name__ == "__main__":
    raise SystemExit(main())
