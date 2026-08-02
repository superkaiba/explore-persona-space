"""#1902 Step 9a-ter free-analysis round (0 GPU-h, CPU, batched numpy).

Two bundled analyses over the COMMITTED per-cell artifacts
(``eval_results/issue_1902/fits/percell/*.npz``) plus per-row answer token
counts extracted from the pod's gen JSONLs (id + n_tokens ONLY — no text):

1. **Answer-length-controlled grid read** (headline-affecting): re-aggregate
   held-out R^2/Q per 4x4 grid cell (reader ckpt x answer source, ctx arm,
   single corpus, layer*) on a LENGTH-MATCHED subset of test rows — per grid
   column (source s) keep rows whose answer-s n_tokens falls in a common band
   across sources (IQR-overlap ladder), then quantile-bin-equalize the
   within-band length histograms across columns by seeded subsampling.
   Recompute the key contrasts (B->S text vs representation decomposition;
   per-column fixed-text decay) with cluster-grouped bootstrap CIs.

   The marginal matched design's cross-column text_delta compares per-column
   row subsets that are largely disjoint (conditioning on a post-source
   variable), so a PAIRED same-rows read is reported alongside it as the
   primary length-control: rows where BOTH columns' answers are in the
   realized band (optionally |len_b - len_a| <= eps), text_delta on the
   SHARED mask, same cluster-bootstrap draws.

2. **Per-transition retention CIs**: cluster-bootstrap CIs (1000 draws) for
   each of the 6 adjacent-stage retention values rho(i->j) = R2_gl(i->j) /
   Q(j,j) at layer* (the exact ``issue1902_fits.finalize`` ratio), from the
   per-context SS arrays behind the xfer + diagonal units.

Row-length inputs live under ``eval_results/issue_1902/followup_9ater/inputs/``
(TSVs of id + integer count / group labels only; extracted pod-side via jq —
no corpus or generation text ever leaves the pod).

Run (VM CPU, thread-capped):

    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 \
    uv run python scripts/issue1902_followup_9ater.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
EVAL = REPO_ROOT / "eval_results" / "issue_1902"
PERCELL = EVAL / "fits" / "percell"
OUT = EVAL / "followup_9ater"
INP = OUT / "inputs"

CKPTS = ("B", "S", "D", "R")
ADJACENT = [("B", "S"), ("S", "B"), ("S", "D"), ("D", "S"), ("D", "R"), ("R", "D")]
N_BOOT = 1_000
FOLD_SEED = 42  # issue1902_common.FOLD_SEED — finalize() uses FOLD_SEED + 1902
MATCH_SEED = 20260801
N_BINS = 5
MIN_MATCHED_N = 500
BAND_LADDER = (("iqr", 0.25), ("p10", 0.10), ("p5", 0.05))
PAIRED_EPS = (8, 16, 32)
PAIRED_VARIANTS = ("eps8", "eps16", "eps32", "band")


def _metadata() -> dict:
    """Reproducibility metadata (git commit + env versions + timestamp)."""
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        commit = "unknown"
    return {
        "script": "scripts/issue1902_followup_9ater.py",
        "git_commit": commit,
        "numpy": np.__version__,
        "python": sys.version.split()[0],
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True))
    tmp.replace(path)
    print(f"[9ater] wrote {path.relative_to(REPO_ROOT)}", flush=True)


def _read_tsv2(path: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for line in path.open(encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        rid, val = line.split("\t")
        out[rid] = int(val)
    return out


def _pooled_r2(res: float, tot: float) -> float:
    return 1.0 - res / tot if tot > 0 else float("nan")


def load_row_index() -> tuple[list[str], np.ndarray, list[str]]:
    """(ids in store row order, group id per row, sorted group names)."""
    ids: list[str] = []
    groups: list[str] = []
    for line in (INP / "rowindex_single.tsv").open(encoding="utf-8"):
        line = line.rstrip("\n")
        if not line:
            continue
        parts = line.split("\t")
        ids.append(parts[0])
        groups.append(parts[1])
    names = sorted(set(groups))
    of = {g: k for k, g in enumerate(names)}
    gid = np.asarray([of[g] for g in groups], dtype=np.int64)
    return ids, gid, names


def assemble(pattern: str, keys: tuple[str, ...], n: int) -> dict[str, np.ndarray]:
    """Per-context arrays from fold shards (issue1902_fits._load_shards shape)."""
    shards = sorted(PERCELL.glob(pattern))
    if not shards:
        raise FileNotFoundError(f"no percell shards match {pattern}")
    out: dict[str, np.ndarray] = {}
    fitted = np.zeros(n, bool)
    for sp in shards:
        d = np.load(sp)
        rows = d["row_idx"]
        for key in keys:
            arr = d[key]
            if key not in out:
                out[key] = np.full((*arr.shape[:-1], n), np.nan)
            out[key][..., rows] = arr
        fitted[rows] = True
    out["__fitted__"] = fitted
    return out


def layer_row(pattern_file: Path, layer_star: int) -> int:
    layers = list(np.load(pattern_file)["layers"])
    if layer_star not in layers:
        raise RuntimeError(f"layer* {layer_star} not in {pattern_file.name} layers {layers}")
    return layers.index(layer_star)


def cell_ss_at_star(m: str, s: str, layer_star: int, n: int) -> tuple[np.ndarray, np.ndarray]:
    """(res, tot) per-row SS at layer* for grid cell (reader m, source s)."""
    pat = f"diag_{m}_single_ctx_f*.npz" if m == s else f"grid_{m}{s}_single_ctx_f*.npz"
    li = layer_row(sorted(PERCELL.glob(pat))[0], layer_star)
    sh = assemble(pat, ("ss_res", "ss_tot"), n)
    if not sh["__fitted__"].all():
        raise RuntimeError(f"cell ({m},{s}): {int((~sh['__fitted__']).sum())} unfitted rows")
    return sh["ss_res"][li], sh["ss_tot"][li]


def group_sums(vals: np.ndarray, gid: np.ndarray, n_groups: int) -> np.ndarray:
    """(n,) per-row values -> (G,) per-group sums (NaN -> 0, pooled-OOF conv)."""
    return np.bincount(gid, weights=np.nan_to_num(vals, nan=0.0), minlength=n_groups)


def boot_r2(counts: np.ndarray, res_g: np.ndarray, tot_g: np.ndarray) -> np.ndarray:
    """R^2 per draw: 1 - (counts @ res_g)/(counts @ tot_g); (n_draws,) or (n_draws, C)."""
    num = counts @ (res_g.T if res_g.ndim == 2 else res_g)
    den = counts @ (tot_g.T if tot_g.ndim == 2 else tot_g)
    with np.errstate(divide="ignore", invalid="ignore"):
        return 1.0 - num / den


def ci(draws: np.ndarray) -> list[float]:
    d = draws[np.isfinite(draws)]
    if d.size == 0:
        return [float("nan"), float("nan")]
    return [float(np.quantile(d, 0.025)), float(np.quantile(d, 0.975))]


def qtiles(x: np.ndarray) -> dict[str, float]:
    ps = (5, 10, 25, 50, 75, 90, 95)
    return {f"p{p}": float(np.percentile(x, p)) for p in ps}


# ── analysis 1: length-matched grid ─────────────────────────────────────────


def choose_band(ntok: dict[str, np.ndarray]) -> tuple[str, float, float, dict]:
    """First ladder band with a non-empty cross-source overlap; detail per rung."""
    detail = {}
    chosen = None
    for name, p in BAND_LADDER:
        lo = max(float(np.percentile(ntok[s], 100 * p)) for s in CKPTS)
        hi = min(float(np.percentile(ntok[s], 100 * (1 - p))) for s in CKPTS)
        n_in = {s: int(((ntok[s] >= lo) & (ntok[s] <= hi)).sum()) for s in CKPTS}
        detail[name] = {"lo": lo, "hi": hi, "n_in_band": n_in, "valid": lo < hi}
        if chosen is None and lo < hi and min(n_in.values()) >= MIN_MATCHED_N:
            chosen = (name, lo, hi)
    if chosen is None:  # fall back to the widest valid rung; gate reported downstream
        for name, _p in reversed(BAND_LADDER):
            if detail[name]["valid"]:
                chosen = (name, detail[name]["lo"], detail[name]["hi"])
                break
    if chosen is None:
        raise RuntimeError(f"no valid cross-source length band at any rung: {detail}")
    return (*chosen, detail)


def equalize_bins(
    ntok: dict[str, np.ndarray], lo: float, hi: float, rng: np.random.Generator
) -> tuple[dict[str, np.ndarray], dict]:
    """Quantile-bin-equalized matched subsets per column (bool masks).

    Bins are pooled-quantile edges over all in-band lengths; per bin each
    column is subsampled (seeded, without replacement) to the min per-source
    count, so every column's matched length histogram is identical at bin
    grain and matched-n is uniform across columns.
    """
    in_band = {s: (ntok[s] >= lo) & (ntok[s] <= hi) for s in CKPTS}
    pooled = np.concatenate([ntok[s][in_band[s]] for s in CKPTS])
    edges = np.unique(np.percentile(pooled, np.linspace(0, 100, N_BINS + 1)))
    edges[0], edges[-1] = lo - 0.5, hi + 0.5
    masks = {s: np.zeros(ntok[s].shape[0], bool) for s in CKPTS}
    per_bin = []
    for b in range(len(edges) - 1):
        bin_idx = {
            s: np.flatnonzero(in_band[s] & (ntok[s] > edges[b]) & (ntok[s] <= edges[b + 1]))
            for s in CKPTS
        }
        t = min(idx.size for idx in bin_idx.values())
        per_bin.append({"edges": [float(edges[b]), float(edges[b + 1])], "target": int(t)})
        for s in CKPTS:
            keep = (
                bin_idx[s]
                if bin_idx[s].size == t
                else rng.choice(bin_idx[s], size=t, replace=False)
            )
            masks[s][keep] = True
    sizes = {s: int(masks[s].sum()) for s in CKPTS}
    if len(set(sizes.values())) != 1:
        raise RuntimeError(f"equalized column sizes differ: {sizes}")
    return masks, {"bin_detail": per_bin, "matched_n_per_column": sizes}


def paired_same_rows(
    gid: np.ndarray,
    n_groups: int,
    counts: np.ndarray,
    ntok: dict[str, np.ndarray],
    ss: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    lo: float,
    hi: float,
) -> dict:
    """Paired same-rows text-axis read (the primary length-control design).

    Per adjacent pair (a, b): keep rows where BOTH columns' answers fall in the
    realized band ('band' variant), optionally also |len_b - len_a| <= eps
    ('eps<k>'); text_delta = R2(reader a, text b) - R2(reader a, text a) on
    that SHARED row mask, cluster-bootstrapped with the SAME counts draws as
    the marginal design. Returns per (pair, variant): delta, CI, n, mean lens.
    """
    out: dict[str, dict] = {}
    for a, b in ADJACENT:
        in_band = (ntok[a] >= lo) & (ntok[a] <= hi) & (ntok[b] >= lo) & (ntok[b] <= hi)
        variants: dict[str, np.ndarray] = {"band": in_band}
        for eps in PAIRED_EPS:
            variants[f"eps{eps}"] = in_band & (np.abs(ntok[a] - ntok[b]) <= eps)
        entry: dict[str, dict] = {}
        for vname, mask in variants.items():
            w = mask.astype(np.float64)
            point: dict[str, float] = {}
            draws: dict[str, np.ndarray] = {}
            for col in (b, a):
                res, tot = ss[(a, col)]
                point[col] = _pooled_r2(float(np.nansum(res * w)), float(np.nansum(tot * w)))
                draws[col] = boot_r2(
                    counts, group_sums(res * w, gid, n_groups), group_sums(tot * w, gid, n_groups)
                )
            entry[vname] = {
                "text_delta": float(point[b] - point[a]),
                "text_ci": ci(draws[b] - draws[a]),
                "n": int(mask.sum()),
                "mean_len": {
                    s: (float(ntok[s][mask].mean()) if mask.any() else float("nan")) for s in (a, b)
                },
            }
        out[f"{a}->{b}"] = entry
    return out


def derive_conclusion(paired: dict, overlap_frac: dict[str, float]) -> str:
    """Data-driven summary: paired same-rows text deltas + the marginal caveat."""
    frags = []
    for pair, entry in paired.items():
        excl = [
            v for v in PAIRED_VARIANTS if entry[v]["text_ci"][0] > 0 or entry[v]["text_ci"][1] < 0
        ]
        v = entry["eps16"]
        if len(excl) == len(PAIRED_VARIANTS):
            verdict = "robust to paired per-row length control (CI excludes 0 at every variant)"
        elif not excl:
            verdict = "indistinguishable from 0 under paired per-row length control"
        else:
            verdict = f"mixed (CI excludes 0 only at {', '.join(excl)})"
        frags.append(
            f"{pair}: paired text_delta {v['text_delta']:+.4f} "
            f"[{v['text_ci'][0]:+.4f}, {v['text_ci'][1]:+.4f}] at eps=16 (n={v['n']}) -- {verdict}"
        )
    ov = "; ".join(f"{p}: {f:.1%}" for p, f in overlap_frac.items())
    return (
        "Paired same-rows design (primary length-control read). "
        + ". ".join(frags)
        + ". The marginal matched design's cross-column text_delta compares per-column row "
        + f"subsets that are largely disjoint (mask overlap of matched_n -- {ov}), i.e. it "
        + "conditions on a post-source variable (answer length); marginal-vs-paired sign/size "
        + "differences are row-composition shifts, not text effects."
    )


def analysis1(
    gid: np.ndarray,
    n_groups: int,
    counts: np.ndarray,
    ntok: dict[str, np.ndarray],
    ss: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    grid_ref: dict,
) -> dict:
    band_name, lo, hi, band_detail = choose_band(ntok)
    rng = np.random.default_rng(MATCH_SEED)
    masks, eq_detail = equalize_bins(ntok, lo, hi, rng)
    matched_n = next(iter(eq_detail["matched_n_per_column"].values()))
    under_matched = matched_n < MIN_MATCHED_N

    # group sums per cell, unmatched + matched (mask keyed on the SOURCE column)
    res_g_u = np.zeros((len(CKPTS), len(CKPTS), n_groups))
    tot_g_u = np.zeros_like(res_g_u)
    res_g_m = np.zeros_like(res_g_u)
    tot_g_m = np.zeros_like(res_g_u)
    q_u = np.full((len(CKPTS), len(CKPTS)), np.nan)
    q_m = np.full_like(q_u, np.nan)
    for mi, m in enumerate(CKPTS):
        for si, s in enumerate(CKPTS):
            res, tot = ss[(m, s)]
            res_g_u[mi, si] = group_sums(res, gid, n_groups)
            tot_g_u[mi, si] = group_sums(tot, gid, n_groups)
            w = masks[s].astype(np.float64)
            res_g_m[mi, si] = group_sums(res * w, gid, n_groups)
            tot_g_m[mi, si] = group_sums(tot * w, gid, n_groups)
            q_u[mi, si] = _pooled_r2(float(np.nansum(res)), float(np.nansum(tot)))
            q_m[mi, si] = _pooled_r2(float(np.nansum(res * w)), float(np.nansum(tot * w)))

    # sanity: recomputed unmatched grid must reproduce the committed Q grid
    ref = np.array([[np.nan if x is None else x for x in row] for row in grid_ref["Q_grid"]])
    if not np.allclose(q_u, ref, atol=1e-8, equal_nan=True):
        raise RuntimeError(f"unmatched Q grid mismatch vs grid_cells.json:\n{q_u}\n{ref}")

    # per-draw R^2 for every cell (paired over cluster draws)
    flat_r = res_g_m.reshape(-1, n_groups)
    flat_t = tot_g_m.reshape(-1, n_groups)
    draws_m = boot_r2(counts, flat_r, flat_t).reshape(N_BOOT, len(CKPTS), len(CKPTS))
    flat_ru = res_g_u.reshape(-1, n_groups)
    flat_tu = tot_g_u.reshape(-1, n_groups)
    draws_u = boot_r2(counts, flat_ru, flat_tu).reshape(N_BOOT, len(CKPTS), len(CKPTS))

    def _contrasts(q: np.ndarray, draws: np.ndarray) -> dict:
        out = {}
        for a, b in ADJACENT:
            ai, bi = CKPTS.index(a), CKPTS.index(b)
            out[f"{a}->{b}"] = {
                # text axis: same reader a, swap answer text a -> b
                "text_delta": float(q[ai, bi] - q[ai, ai]),
                "text_ci": ci(draws[:, ai, bi] - draws[:, ai, ai]),
                # representation axis: same answer text a, swap reader a -> b
                "repr_delta": float(q[bi, ai] - q[ai, ai]),
                "repr_ci": ci(draws[:, bi, ai] - draws[:, ai, ai]),
            }
        decay = {}
        for si, s in enumerate(CKPTS):
            decay[s] = {
                m: {
                    "delta_vs_own_reader": float(q[mi, si] - q[si, si]),
                    "ci": ci(draws[:, mi, si] - draws[:, si, si]),
                }
                for mi, m in enumerate(CKPTS)
                if m != s
            }
        return {"adjacent_decomposition": out, "fixed_text_decay_per_column": decay}

    grid = {}
    for mi, m in enumerate(CKPTS):
        for si, s in enumerate(CKPTS):
            grid[f"{m}{s}"] = {
                "r2_unmatched": float(q_u[mi, si]),
                "r2_matched": float(q_m[mi, si]),
                "ci_matched": ci(draws_m[:, mi, si]),
                "matched_n": matched_n,
                "under_matched": under_matched,
            }

    len_stats = {
        s: {
            "all_rows": qtiles(ntok[s]),
            "matched_rows": qtiles(ntok[s][masks[s]]),
            "matched_mean": float(ntok[s][masks[s]].mean()),
        }
        for s in CKPTS
    }

    # marginal-design contrasts + per-pair mask-overlap diagnostic (the two
    # per-column matched subsets a cross-column text_delta implicitly compares)
    matched = _contrasts(q_m, draws_m)
    overlap_frac: dict[str, float] = {}
    for a, b in ADJACENT:
        ov = int((masks[a] & masks[b]).sum())
        overlap_frac[f"{a}->{b}"] = ov / matched_n
        matched["adjacent_decomposition"][f"{a}->{b}"]["mask_overlap"] = {
            "n_intersection": ov,
            "frac_of_matched_n": ov / matched_n,
        }
    paired = paired_same_rows(gid, n_groups, counts, ntok, ss, lo, hi)
    return {
        "metadata": _metadata(),
        "design": {
            "layer_star": int(grid_ref["layer_star"]),
            "arm": "ctx",
            "corpus": "single",
            "fitter": "ridge",
            "n_rows_total": int(gid.shape[0]),
            "n_boot": N_BOOT,
            "match_seed": MATCH_SEED,
            "n_bins": N_BINS,
            "min_matched_n": MIN_MATCHED_N,
            "band_ladder": [name for name, _ in BAND_LADDER],
            "length_source": "gen n_tokens (pod gen/single/<ckpt>.jsonl), joined by row id",
        },
        "realized_band": {"rung": band_name, "lo": lo, "hi": hi, "ladder_detail": band_detail},
        "equalization": eq_detail,
        "matched_n": matched_n,
        "under_matched": under_matched,
        "length_stats_per_source": len_stats,
        "cells": grid,
        "design_note": (
            "The 'matched' (marginal length-matched) design equalizes each grid COLUMN's "
            "answer-length histogram independently, so its cross-column text_delta compares "
            "per-column row subsets that are largely disjoint (see mask_overlap) -- it "
            "conditions on a post-source variable (answer length). 'paired_same_rows' is the "
            "primary length-control read: both columns' answers in the realized band on the "
            "SAME rows, optionally |len_b - len_a| <= eps, same cluster-bootstrap draws."
        ),
        "matched": matched,
        "unmatched": _contrasts(q_u, draws_u),
        "paired_same_rows": {
            "design": {
                "eps_grid": list(PAIRED_EPS),
                "variants": "band (both-in-band, no eps) + eps8/eps16/eps32 "
                "(both-in-band AND |len_b - len_a| <= eps)",
            },
            "pairs": paired,
        },
        "conclusion": derive_conclusion(paired, overlap_frac),
    }


# ── analysis 2: per-transition retention CIs ────────────────────────────────


def analysis2(
    gid: np.ndarray,
    n_groups: int,
    counts: np.ndarray,
    n: int,
    layer_star: int,
    xfer_ref: dict,
) -> dict:
    out: dict[str, dict] = {}
    for i, j in ADJACENT:
        sh = assemble(f"xfer_{i}{j}_f*.npz", ("ss_res_gl", "ss_tot"), n)
        dj_pat = f"diag_{j}_single_ctx_f*.npz"
        li = layer_row(sorted(PERCELL.glob(dj_pat))[0], layer_star)
        dj = assemble(dj_pat, ("ss_res", "ss_tot"), n)
        r2_gl = _pooled_r2(float(np.nansum(sh["ss_res_gl"])), float(np.nansum(sh["ss_tot"])))
        q_jj = _pooled_r2(float(np.nansum(dj["ss_res"][li])), float(np.nansum(dj["ss_tot"][li])))
        rho = r2_gl / q_jj if np.isfinite(q_jj) and q_jj > 0 else float("nan")
        ref = xfer_ref["pairs"][f"{i}->{j}"]["retention_gl"]
        if not np.isclose(rho, ref, atol=1e-9):
            raise RuntimeError(f"retention point mismatch {i}->{j}: {rho} vs {ref}")
        r2_gl_d = boot_r2(
            counts,
            group_sums(sh["ss_res_gl"], gid, n_groups),
            group_sums(sh["ss_tot"], gid, n_groups),
        )
        q_jj_d = boot_r2(
            counts,
            group_sums(dj["ss_res"][li], gid, n_groups),
            group_sums(dj["ss_tot"][li], gid, n_groups),
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            rho_d = np.where(q_jj_d > 0, r2_gl_d / q_jj_d, np.nan)
        out[f"{i}->{j}"] = {
            "retention_point": float(rho),
            "retention_ci": ci(rho_d),
            "r2_gl": float(r2_gl),
            "q_jj_at_star": float(q_jj),
            "n_finite_draws": int(np.isfinite(rho_d).sum()),
        }
    return {
        "metadata": _metadata(),
        "design": {
            "definition": "rho(i->j) = pooled R2_gl(i->j) / Q(j,j) at layer* "
            "(ctx arm, single corpus; issue1902_fits.finalize convention)",
            "layer_star": layer_star,
            "n_boot": N_BOOT,
            "bootstrap": "cluster-grouped multinomial over corpus groups, "
            f"seed default_rng(FOLD_SEED + 1902)={FOLD_SEED + 1902} (finalize parity)",
        },
        "transitions": out,
    }


def main() -> None:
    t0 = time.time()
    ids, gid, names = load_row_index()
    n, n_groups = len(ids), len(names)
    print(f"[9ater] rows={n} groups={n_groups}", flush=True)

    grid_ref_all = json.loads((EVAL / "fits" / "grid_cells.json").read_text())
    layer_star = int(grid_ref_all["layer_star"])
    grid_ref = grid_ref_all["h3_variance_decomposition"]["ridge_ctx_single"]
    grid_ref["layer_star"] = layer_star
    xfer_ref = json.loads((EVAL / "transfer" / "transfer_matrix.json").read_text())

    # finalize() RNG parity: counts for corpus 'single' are the FIRST draw
    rng = np.random.default_rng(FOLD_SEED + 1902)
    counts = rng.multinomial(n_groups, np.full(n_groups, 1.0 / n_groups), size=N_BOOT).astype(
        np.float64
    )

    # per-row answer token counts joined onto store row order
    ntok: dict[str, np.ndarray] = {}
    ntok_cap: dict[str, np.ndarray] = {}
    for s in CKPTS:
        gen = _read_tsv2(INP / f"ntok_{s}.tsv")
        cap = _read_tsv2(INP / f"nanstok_{s}.tsv")
        missing = [rid for rid in ids if rid not in gen]
        if missing:
            raise RuntimeError(f"{len(missing)} row ids missing from ntok_{s}.tsv")
        ntok[s] = np.asarray([gen[rid] for rid in ids], dtype=np.float64)
        ntok_cap[s] = np.asarray([cap[rid] for rid in ids], dtype=np.float64)
        r = float(np.corrcoef(ntok[s], ntok_cap[s])[0, 1])
        print(
            f"[9ater] {s}: median gen n_tokens={np.median(ntok[s]):.0f} corr(gen, capture)={r:.4f}",
            flush=True,
        )

    ss = {(m, s): cell_ss_at_star(m, s, layer_star, n) for m in CKPTS for s in CKPTS}

    a1 = analysis1(gid, n_groups, counts, ntok, ss, grid_ref)
    a1["length_source_crosscheck"] = {
        s: {"pearson_gen_vs_capture": float(np.corrcoef(ntok[s], ntok_cap[s])[0, 1])} for s in CKPTS
    }
    _write_json(OUT / "length_matched_grid.json", a1)

    a2 = analysis2(gid, n_groups, counts, n, layer_star, xfer_ref)
    _write_json(OUT / "retention_cis.json", a2)
    print(f"[9ater] done in {time.time() - t0:.1f}s", flush=True)
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
