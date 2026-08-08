"""Issue #1945 follow-up r8 — corpus x depth-band strata breakdown of the BCV curves.

Closes plan v3 section 6's planned-but-not-run descriptive condition: per stratum
(corpus x depth-band, as the #1738 sampling manifest defines them), re-reduce the
staged interaction residual and report the observed Gabriel (2,2) BCV R^2 curve
plus a light permuted-pairing reference (B=25, same per-draw max-over-r
convention).  DESCRIPTIVE ONLY — no verdict/lattice per stratum, no selection
over strata (the plan forbids a headline here).

Scope (the brief): primary cell family (context/prefix/bare at L19 ridge), log
space, k=256, both parent folds.  All math is imported from
``issue1945_bcv_interaction`` (bcv_curve_batched / twoway_removed /
space_transform / bcv_splits / r_grid_for_k / LayerCtx / load_pred) — nothing is
re-implemented.

The ci -> (corpus, depth_band) map comes from the HF data repo prefix
``issue1738_multiturn/sampling_manifest`` (scoped per-file ``hub.stage_hub_file``
downloads; NEVER snapshot_download).  Manifest part rows carry real-corpus text —
this script reads ONLY the ``i`` / ``corpus`` / ``depth`` / ``split`` fields and
never prints row text (digest-only discipline).  The manifest's holdout coverage
is cross-checked against the staged 9,941 holdout ci values and the join rate is
reported (a partial join is reportable, never silently dropped).

Strata with n_stratum_fold < 40 (too few rows for a meaningful (2,2) BCV split)
are recorded as ``insufficient-n``, never silently dropped.

Run from the issue-1945 worktree root:
    uv run python scripts/issue1945_strata_breakdown.py
"""

from __future__ import annotations

import json
import platform
import resource
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy/torch (shared-VM run)

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import HfApi  # noqa: E402

import issue1945_bcv_interaction as bcv  # noqa: E402
from explore_persona_space.analysis.paper_plots import (  # noqa: E402
    paper_palette_blog,
    savefig_paper,
    set_paper_style,
)
from explore_persona_space.orchestrate import hub  # noqa: E402

SEED = bcv.SEED  # 1945 — all new randomness
K = 256
SPACE = "log"
LAYER = 19
ARMS = ("context", "prefix", "bare")
FITTER = "ridge"
B_PERM = 25  # light perm reference (brief-pinned)
N_FLOOR = 40  # min rows per (stratum, fold) for a (2,2) BCV split
DATA_REPO = "superkaiba1/explore-persona-space-data"
MANIFEST_PREFIX = "issue1738_multiturn/sampling_manifest"
# #1738 producer bands (issue1738_multiturn_generate_capture.DEPTH_BANDS)
DEPTH_BANDS = ((2, 2), (3, 4), (5, 10_000))

EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_1945" / "strata"
STAGE_DIR = PROJECT_ROOT / "data" / "issue_1945" / "hf_dl" / "sampling_manifest"


def _depth_band(depth: int) -> str:
    """#1738 producer band label: '2-2' / '3-4' / '>=5'."""
    for lo, hi in DEPTH_BANDS:
        if lo <= depth <= hi:
            return f"{lo}-{hi}" if hi < 10_000 else f">={lo}"
    raise ValueError(f"depth {depth} outside every band")


def stage_manifest() -> tuple[dict[int, tuple[str, str]], dict]:
    """Stage split_1738.json + every part file (scoped, idempotent) and build
    the holdout ci -> (corpus, depth_band) map.  Returns (map, staging_digest).

    Reads ONLY i/corpus/depth/split fields per row — never row text.
    """
    t0 = time.time()
    entries = hub.retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in hub.retry_transient; scoped prefix listing
            HfApi().list_repo_tree(
                DATA_REPO, path_in_repo=MANIFEST_PREFIX, repo_type="dataset", recursive=True
            )
        ),
        what=f"list_repo_tree {MANIFEST_PREFIX}",
    )
    paths = sorted(e.path for e in entries)
    part_paths = [p for p in paths if Path(p).name.startswith("part_")]
    split_path = f"{MANIFEST_PREFIX}/split_1738.json"
    assert split_path in paths, f"missing {split_path} in manifest listing"
    assert part_paths, "no manifest part files listed"
    local_split = hub.stage_hub_file(
        DATA_REPO, split_path, STAGE_DIR / "split_1738.json", repo_type="dataset"
    )
    split_doc = json.loads(local_split.read_text())
    holdout_ci = set(split_doc["sets"]["holdout"]["ci"])
    ci_map: dict[int, tuple[str, str]] = {}
    n_rows_scanned = 0
    for k_part, p in enumerate(part_paths):
        local = hub.stage_hub_file(DATA_REPO, p, STAGE_DIR / Path(p).name, repo_type="dataset")
        with local.open(encoding="utf-8") as fh:  # text-mode iteration (never splitlines)
            for line in fh:
                if not line.strip():
                    continue
                row = json.loads(line)
                n_rows_scanned += 1
                if row.get("split") == "holdout":
                    ci_map[int(row["i"])] = (str(row["corpus"]), _depth_band(int(row["depth"])))
        print(
            f"[manifest] part {k_part + 1}/{len(part_paths)} scanned;"
            f" holdout mapped so far={len(ci_map)} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    missing_from_parts = holdout_ci - set(ci_map)
    digest = {
        "n_parts": len(part_paths),
        "n_rows_scanned": n_rows_scanned,
        "split_holdout_n": len(holdout_ci),
        "split_holdout_sha256": split_doc["sets"]["holdout"]["sha256"],
        "n_holdout_mapped": len(ci_map),
        "n_split_holdout_missing_from_parts": len(missing_from_parts),
        "strata_declared": split_doc.get("strata"),
        "stage_dir": str(STAGE_DIR),
        "stage_seconds": round(time.time() - t0, 1),
    }
    assert set(ci_map) <= holdout_ci, (
        f"{len(set(ci_map) - holdout_ci)} part rows labeled holdout are absent"
        " from the split doc's holdout ci set"
    )
    return ci_map, digest


def _perm_reference(F: np.ndarray, rg: list[int], rows, cols, rng: np.random.Generator) -> dict:
    """B_PERM within-column permutation draws, batched through bcv_curve_batched.

    Same convention as the main battery: p97.5 of the per-draw max over r>=1.
    """
    n, k = F.shape
    keys = rng.random((B_PERM, n, k), dtype=np.float32)
    idx_t = torch.argsort(torch.from_numpy(keys), dim=1)
    Ft = torch.from_numpy(np.ascontiguousarray(F))
    Fp = torch.gather(Ft.unsqueeze(0).expand(B_PERM, n, k), 1, idx_t)
    mat = bcv.bcv_curve_batched(Fp, rg, rows, cols)  # (B, 1+n_r)
    draw_max = mat[:, 1:].max(axis=1)
    return {
        "n_draws": B_PERM,
        "perm_p975_max": float(np.percentile(draw_max, 97.5)),
        "perm_draw_max": [float(x) for x in draw_max],
    }


def run_strata(ci_map: dict[int, tuple[str, str]], staging: dict) -> dict:
    cfg = bcv.Cfg(cells=tuple(f"{a}_L{LAYER}_{FITTER}" for a in ARMS))
    ctx = bcv.LayerCtx(cfg, LAYER)
    n_hold = int(ctx.ci.shape[0])
    mapped_mask = np.array([int(c) in ci_map for c in ctx.ci])
    join = {
        "n_staged_holdout_ci": n_hold,
        "n_joined": int(mapped_mask.sum()),
        "join_rate": round(float(mapped_mask.mean()), 6),
        "n_unmapped": int((~mapped_mask).sum()),
    }
    print(
        f"[join] staged holdout ci={n_hold} joined={join['n_joined']} rate={join['join_rate']}",
        flush=True,
    )
    labels = np.array(
        [
            f"{ci_map[int(c)][0]}|{ci_map[int(c)][1]}" if m else "unmapped"
            for c, m in zip(ctx.ci, mapped_mask)
        ]
    )
    strata_names = sorted(x for x in set(labels) if x != "unmapped")
    rg = bcv.r_grid_for_k(K)
    records: list[dict] = []
    t_start = time.time()
    unit_i, total = 0, len(ARMS) * len(cfg.folds) * len(strata_names)
    for arm in ARMS:
        cell = f"{arm}_L{LAYER}_{FITTER}"
        P16 = bcv.load_pred(cfg.stage, arm, LAYER, FITTER, ctx.ci, ctx.fp)
        for fold_i in cfg.folds:
            basis_idx, eval_idx, comps, eigvals = ctx.bases[fold_i]
            E = (P16[eval_idx].astype(np.float64) - ctx.Y16[eval_idx].astype(np.float64)) @ comps
            R = E[:, :K] ** 2
            assert np.all(R > 0), f"{cell} fold{fold_i}: non-positive squared residual"
            lam = eigvals[:K]
            assert np.all(lam > 0), "non-positive basis eigenvalue"
            fold_labels = labels[eval_idx]
            for stratum in strata_names:
                unit_i += 1
                sel = np.flatnonzero(fold_labels == stratum)
                n_s = int(sel.size)
                base = {
                    "cell": cell,
                    "arm": arm,
                    "fold": int(fold_i),
                    "stratum": stratum,
                    "corpus": stratum.split("|")[0],
                    "depth_band": stratum.split("|")[1],
                    "n": n_s,
                    "r_grid": [0] + rg,
                }
                if n_s < N_FLOOR:
                    records.append({**base, "status": "insufficient-n", "n_floor": N_FLOOR})
                    print(
                        f"[strata] unit {unit_i}/{total} {cell} fold{fold_i} {stratum} "
                        f"n={n_s} insufficient-n",
                        flush=True,
                    )
                    continue
                t0 = time.time()
                M = bcv.space_transform(R[sel], lam, SPACE)
                F = bcv.twoway_removed(M)
                rows, cols = bcv.bcv_splits(n_s, K, SEED)
                obs = bcv.bcv_curve_batched(F[None], rg, rows, cols)[0]
                rng = np.random.default_rng(
                    [SEED, ARMS.index(arm), int(fold_i), strata_names.index(stratum), 101]
                )
                perm = _perm_reference(F, rg, rows, cols, rng)
                records.append(
                    {
                        **base,
                        "status": "ok",
                        "obs_curve": [float(x) for x in obs],
                        "obs_max": float(obs[1:].max()),
                        **perm,
                        "elapsed_s": round(time.time() - t0, 2),
                    }
                )
                print(
                    f"[strata] unit {unit_i}/{total} {cell} fold{fold_i} {stratum} n={n_s} "
                    f"obs_max={records[-1]['obs_max']:.4f} "
                    f"perm_p975={perm['perm_p975_max']:.4f} "
                    f"elapsed={time.time() - t_start:.1f}s",
                    flush=True,
                )
        del P16
    # fold-pooled curves per (cell, stratum) where BOTH folds computed (main-battery
    # convention: pooled = mean over fold matrices)
    pooled: list[dict] = []
    for arm in ARMS:
        cell = f"{arm}_L{LAYER}_{FITTER}"
        for stratum in strata_names:
            per_fold = [
                r
                for r in records
                if r["cell"] == cell and r["stratum"] == stratum and r["status"] == "ok"
            ]
            if len(per_fold) != len(cfg.folds):
                continue
            curves = np.stack([np.asarray(r["obs_curve"]) for r in per_fold])
            perm_ref = float(np.mean([r["perm_p975_max"] for r in per_fold]))
            pc = curves.mean(axis=0)
            pooled.append(
                {
                    "cell": cell,
                    "arm": arm,
                    "stratum": stratum,
                    "corpus": stratum.split("|")[0],
                    "depth_band": stratum.split("|")[1],
                    "n_per_fold": [r["n"] for r in per_fold],
                    "r_grid": [0] + rg,
                    "obs_curve_pooled": [float(x) for x in pc],
                    "obs_max_pooled": float(pc[1:].max()),
                    "perm_p975_max_mean_over_folds": perm_ref,
                }
            )
    import scipy  # local: version stamp only

    out = {
        "metadata": {
            "git_commit": bcv._git_commit(),
            "generated_utc": datetime.now(UTC).isoformat(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "torch": torch.__version__,
            "host": platform.node(),
            "seed": SEED,
            "parent_seed": bcv.PARENT_SEED,
            "b_perm": B_PERM,
            "k": K,
            "space": SPACE,
            "layer": LAYER,
            "n_floor": N_FLOOR,
            "stage_dir": str(cfg.stage),
            "ru_maxrss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0,
            "note": (
                "descriptive breakdown only (plan v3 section 6): no verdict/lattice per "
                "stratum, no selection over strata"
            ),
        },
        "manifest_staging": staging,
        "join": join,
        "strata": strata_names,
        "per_fold_records": records,
        "pooled_records": pooled,
    }
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    out_path = EVAL_DIR / "strata_breakdown.json"
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(out, indent=1))
    tmp.replace(out_path)
    print(f"[strata] wrote {out_path}", flush=True)
    return out


BAND_STYLE = {"2-2": "-", "3-4": "--", ">=5": ":"}
ARM_LABEL = {"context": "full context", "prefix": "prefix only", "bare": "bare query"}


def render_figure(out: dict) -> None:
    """One panel per arm: pooled observed BCV curve per corpus x depth-band stratum
    (color = corpus, linestyle = depth band) + ONE dashed reference line at the
    max across strata of the permuted-pairing p97.5-of-per-draw-max."""
    set_paper_style("blog")
    pooled = out["pooled_records"]
    corpora = sorted({r["corpus"] for r in pooled})
    colors = dict(zip(corpora, paper_palette_blog(max(3, len(corpora)))))
    fig, axes = plt.subplots(1, len(ARMS), figsize=(13, 4.2), sharey=True)
    for ax, arm in zip(np.atleast_1d(axes), ARMS):
        rows = [r for r in pooled if r["arm"] == arm]
        perm_ceiling = max((r["perm_p975_max_mean_over_folds"] for r in rows), default=None)
        for r in sorted(rows, key=lambda x: (x["corpus"], x["depth_band"])):
            ax.plot(
                r["r_grid"],
                r["obs_curve_pooled"],
                color=colors[r["corpus"]],
                ls=BAND_STYLE[r["depth_band"]],
                marker="o",
                ms=3,
                lw=1.6,
                label=f"{r['corpus']}, {r['depth_band']} user turns (n={min(r['n_per_fold'])})",
            )
        if perm_ceiling is not None:
            ax.axhline(
                perm_ceiling,
                color="0.4",
                ls="-.",
                lw=1.2,
                label="permuted-pairing reference (max over strata)",
            )
        ax.axhline(0.0, color="0.8", lw=0.8)
        ax.set_title(f"{ARM_LABEL[arm]}, layer 19, ridge", loc="left", fontsize=10)
        ax.set_xlabel("rank of the low-rank fit")
    np.atleast_1d(axes)[0].set_ylabel("held-out interaction R-squared")
    handles, labs = np.atleast_1d(axes)[0].get_legend_handles_labels()
    fig.legend(handles, labs, loc="outside lower center", ncol=4, fontsize=7)
    fig.suptitle(
        "BCV curves by corpus x conversation-depth stratum (log space, 256 directions, "
        "fold-pooled; descriptive)",
        x=0.01,
        ha="left",
        fontsize=11,
    )
    savefig_paper(fig, "issue_1945/r8_strata_breakdown", dir="figures/")
    plt.close(fig)
    print("[strata] figure written", flush=True)


def main() -> None:
    ci_map, staging = stage_manifest()
    out = run_strata(ci_map, staging)
    render_figure(out)
    print("[strata] done", flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(0)  # explicit exit: torch/scipy C-extension atexit race (#1689)


if __name__ == "__main__":
    main()
