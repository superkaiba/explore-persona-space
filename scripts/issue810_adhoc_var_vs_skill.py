#!/usr/bin/env python3
"""#810 free-analysis control: absolute cross-context activation variance vs
reconstruction skill, at a fixed layer, per answer-side summary position.

Question: is the high skill-over-mean R² at the boundary positions (im_end,
turn_nl, tail_1) an ARTIFACT of those positions being near-CONSTANT across the
50 contexts (low absolute variance, so the variance-RATIO metric inflates while
the absolute predicted signal is a sliver)? skill-over-mean is a RATIO and is
blind to absolute variance. Discriminator: does absolute cross-context
activation variance COLLAPSE at the high-skill boundary positions, or does it
stay SUSTAINED (a genuine low-D context summary despite a constant TOKEN)?

Read-only / analysis-only. No training, no generation, no judge/API calls.
Reads already-extracted artifacts:
  - answer_position_sweep/<ctx>.pt (per-position probe-mean summary vectors,
    (n_pos, 28, 3584) fp16, + coverage) on HF data repo (34 positions).
  - #658 v0_summaries.pt on HF (mean/last/maxp probe-mean summaries).
  - eval_results/issue_810/reconstruction_skill_by_summary.json (skill@layer).
  - eval_results/issue_810/readout_rho_by_summary.json (optional rho@layer).

Layer 18 fixed (modal best-reconstruction layer; activation norms grow with
depth so absolute variance is only comparable within one layer).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["OMP_NUM_THREADS"] = "4"

# Shared-VM thread caps (#847): load_dotenv() must bind BEFORE the first
# numpy/torch import (torch freezes its BLAS/intra-op pools at import time).
import pathlib

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv(str(pathlib.Path(__file__).resolve().parent.parent / ".env"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

torch.set_num_threads(4)

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue810_common import (  # noqa: E402
    ANSWER_POSITION_SWEEP_SUBDIR,
    HF_DATA_REPO,
    HF_PREFIX,
    I658_V0_SUMMARIES,
)

FIXED_LAYER = 18
POSITIONS_ORDERED = [
    "mean",
    "maxp",
    "last",
    "head_0",
    "head_1",
    "head_2",
    "head_4",
    "head_8",
    "tail_16",
    "tail_8",
    "tail_4",
    "tail_2",
    "tail_1",
    "im_end",
    "turn_nl",
]
POS_TYPE = {
    "mean": "aggregate",
    "maxp": "aggregate",
    "last": "aggregate",
    "im_end": "boundary",
    "turn_nl": "boundary",
}


def _pos_type(name: str) -> str:
    if name in POS_TYPE:
        return POS_TYPE[name]
    if name.startswith("tail_"):
        return "tail"
    if name.startswith("head_"):
        return "head"
    return "other"


def _hf_download_with_retry(repo_id: str, filename: str, repo_type: str = "dataset") -> str:
    """hf_hub_download with a 429-aware Retry-After-bounded backoff (gotchas.md)."""
    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import HfHubHTTPError

    for attempt in range(6):
        try:
            return hf_hub_download(repo_id, filename, repo_type=repo_type)
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status != 429 or attempt == 5:
                raise
            ra = None
            try:
                ra = e.response.headers.get("Retry-After")
            except Exception:
                ra = None
            wait = min(300.0, max(60.0, float(ra))) if ra else min(300.0, 60.0 * (attempt + 1))
            print(f"[429] {filename}: retry {attempt + 1}/5 after {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _download_position_store(ctx_ids: list[str], dest: Path) -> Path:
    """Download answer_position_sweep/*.pt via snapshot_download(max_workers=4)
    with a 429-aware retry (only 50 files, but the bulk-download path is
    NOT covered by hf_hub built-in 429 retry — gotchas.md)."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import HfHubHTTPError

    prefix = f"{HF_PREFIX}/{ANSWER_POSITION_SWEEP_SUBDIR}"
    allow = [f"{prefix}/{c}.pt" for c in ctx_ids] + [f"{prefix}/manifest.json"]
    for attempt in range(6):
        try:
            snapshot_download(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                allow_patterns=allow,
                local_dir=str(dest),
                max_workers=4,
            )
            return dest / prefix
        except HfHubHTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            if status != 429 or attempt == 5:
                raise
            wait = min(300.0, 60.0 * (attempt + 1))
            print(f"[429] snapshot: retry {attempt + 1}/5 after {wait:.0f}s", flush=True)
            time.sleep(wait)
    raise RuntimeError("unreachable")


def _load_recon_skill() -> dict:
    p = Path("eval_results/issue_810/reconstruction_skill_by_summary.json")
    return json.load(open(p))


def _load_readout_rho() -> dict | None:
    p = Path("eval_results/issue_810/readout_rho_by_summary.json")
    if not p.exists():
        return None
    return json.load(open(p))


def _skill_at_layer(recon: dict, name: str, layer: int) -> float | None:
    entries = recon["by_summary"].get(name)
    if not entries:
        return None
    hits = [e for e in entries if e["layer"] == layer]
    return float(hits[0]["ridge_skill"]) if hits else None


def _best_layer_skill(recon: dict, name: str) -> tuple[float | None, int | None]:
    entries = recon["by_summary"].get(name)
    if not entries:
        return None, None
    best = max(entries, key=lambda e: e["ridge_skill"])
    return float(best["ridge_skill"]), int(best["layer"])


def _readout_rho_at_layer(readout: dict | None, name: str, layer: int) -> float | None:
    """Aggregate rho_graded across the 3 read-out behaviors at layer, method
    trained_ridge (the graded-E0 primary). NON-load-bearing companion; the
    readout DV is per-(behavior,method) so there is no single primary rho —
    we report the cross-behavior mean of the trained_ridge arm for reference."""
    if readout is None:
        return None
    cells = readout.get("cells")
    if not isinstance(cells, list):
        return None
    vals = [
        c["rho_graded"]
        for c in cells
        if c.get("summary") == name
        and c.get("layer") == layer
        and c.get("method") == "trained_ridge"
        and c.get("rho_graded") is not None
    ]
    if not vals:
        return None
    return float(np.mean(vals))


def _cov_stats(mat: np.ndarray) -> dict:
    """mat: (n_ctx, H) cross-context summary vectors at one layer.
    Returns var_trace (sum of per-dim variances), var_trace_per_dim,
    participation_ratio ((sum lam)^2 / sum lam^2 of covariance eigenvalues)."""
    n, _H = mat.shape
    # per-dim variance (ddof=1) summed = trace of the cross-context covariance.
    per_dim_var = mat.var(axis=0, ddof=1)  # (H,)
    var_trace = float(per_dim_var.sum())
    var_trace_per_dim = float(per_dim_var.mean())
    # participation ratio from covariance eigenvalues. H >> n, so the nonzero
    # spectrum has <= n-1 eigenvalues; compute via the (n x n) Gram of centered
    # rows (same nonzero eigenvalues as the H x H covariance, up to the 1/(n-1)
    # scaling that cancels in the PR ratio).
    xc = mat - mat.mean(axis=0, keepdims=True)  # (n, H)
    gram = xc @ xc.T  # (n, n) — eigenvalues == (n-1)*cov eigenvalues (nonzero set)
    lam = np.linalg.eigvalsh(gram)
    lam = lam[lam > 1e-12]
    if lam.size == 0:
        pr = 0.0
    else:
        pr = float((lam.sum() ** 2) / (lam**2).sum())
    return {
        "var_trace": var_trace,
        "var_trace_per_dim": var_trace_per_dim,
        "participation_ratio": pr,
    }


def main() -> int:
    recon = _load_recon_skill()
    readout = _load_readout_rho()
    capture_layers = recon["capture_layers"]
    layer_i = capture_layers.index(FIXED_LAYER)
    ctx_ids = None
    # ctx ids from the recon diagnostics if present, else from v0_summaries.
    print("[phase=load-v0] downloading #658 v0_summaries.pt", flush=True)
    v0_path = _hf_download_with_retry(HF_DATA_REPO, I658_V0_SUMMARIES)
    v0 = torch.load(v0_path, weights_only=False)
    v0_summaries = v0["summaries"]  # {recipe: {ctx_id: (28,H)}}
    v0_layers = v0["capture_layers"]
    v0_layer_i = list(v0_layers).index(FIXED_LAYER)
    ctx_ids = sorted(v0_summaries["mean"].keys())
    print(f"[phase=load-v0] {len(ctx_ids)} contexts, v0 layer idx {v0_layer_i}", flush=True)

    dest = Path("/tmp/i810_posstore")
    dest.mkdir(parents=True, exist_ok=True)
    print("[phase=download-store] answer_position_sweep/*.pt (max_workers=4)", flush=True)
    store_dir = _download_position_store(ctx_ids, dest)

    # Load per-position vectors at FIXED_LAYER for every context (coverage-aware).
    # pos_stack[name] -> list of (H,) vectors over covered contexts.
    pos_vectors: dict[str, list[np.ndarray]] = {}
    pos_covered_ctx: dict[str, list[str]] = {}
    store_positions = None
    store_layer_i = None
    for c in ctx_ids:
        blob = torch.load(store_dir / f"{c}.pt", weights_only=False)
        if store_positions is None:
            store_positions = list(blob["positions"])
            store_layer_i = list(blob["capture_layers"]).index(FIXED_LAYER)
        pv = blob["pos_vectors"].float().numpy()  # (n_pos, Lc, H)
        cov = blob["coverage"]
        for pi, name in enumerate(store_positions):
            if cov.get(name, 0) <= 0:
                continue  # 0-coverage row is zero-filled + excluded (extractor contract)
            pos_vectors.setdefault(name, []).append(pv[pi, store_layer_i])
            pos_covered_ctx.setdefault(name, []).append(c)

    # mean/maxp/last from v0_summaries (always covered on all 50).
    for recipe in ("mean", "maxp", "last"):
        vecs = [v0_summaries[recipe][c][v0_layer_i].numpy() for c in ctx_ids]
        pos_vectors[recipe] = vecs
        pos_covered_ctx[recipe] = list(ctx_ids)

    # Build per-position table.
    rows = {}
    for name in POSITIONS_ORDERED:
        vecs = pos_vectors.get(name)
        if not vecs:
            rows[name] = {
                "n_covered": 0,
                "var_trace_L18": None,
                "var_trace_per_dim_L18": None,
                "participation_ratio_L18": None,
                "token_entropy_bits": None,
                "skill_L18": _skill_at_layer(recon, name, FIXED_LAYER),
                "skill_best_layer": _best_layer_skill(recon, name)[0],
                "best_layer": _best_layer_skill(recon, name)[1],
                "readout_rho_L18": _readout_rho_at_layer(readout, name, FIXED_LAYER),
                "pos_type": _pos_type(name),
            }
            continue
        mat = np.stack(vecs, axis=0)  # (n_covered, H)
        cs = _cov_stats(mat)
        best_s, best_l = _best_layer_skill(recon, name)
        rows[name] = {
            "n_covered": int(mat.shape[0]),
            "var_trace_L18": cs["var_trace"],
            "var_trace_per_dim_L18": cs["var_trace_per_dim"],
            "participation_ratio_L18": cs["participation_ratio"],
            "token_entropy_bits": None,  # store carries NO per-position token ids (skip; noted)
            "skill_L18": _skill_at_layer(recon, name, FIXED_LAYER),
            "skill_best_layer": best_s,
            "best_layer": best_l,
            "readout_rho_L18": _readout_rho_at_layer(readout, name, FIXED_LAYER),
            "pos_type": _pos_type(name),
        }

    # ── interpretation ──────────────────────────────────────────────────
    # Compare boundary (im_end/turn_nl) absolute variance vs mid-answer
    # (tail_4/tail_8) and vs aggregate (mean/maxp). If boundary variance is NOT
    # collapsed relative to mid/aggregate, the constancy-artifact hypothesis is
    # REFUTED (genuine low-D context summary at a constant token).
    def _vt(name):
        return rows[name]["var_trace_L18"]

    boundary_vt = [_vt("im_end"), _vt("turn_nl")]
    mid_vt = [_vt("tail_4"), _vt("tail_8")]
    agg_vt = [_vt("mean"), _vt("maxp")]
    b_med = float(np.median([v for v in boundary_vt if v is not None]))
    mid_med = float(np.median([v for v in mid_vt if v is not None]))
    agg_med = float(np.median([v for v in agg_vt if v is not None]))
    # boundary vs mid ratio, boundary vs aggregate ratio
    b_over_mid = b_med / mid_med if mid_med else float("nan")
    b_over_agg = b_med / agg_med if agg_med else float("nan")
    collapsed = b_med < 0.5 * mid_med  # heuristic collapse threshold
    if collapsed:
        verdict = (
            f"SUPPORTED (artifact): boundary absolute variance (median var_trace@L18 "
            f"{b_med:.3g}) COLLAPSES to <0.5x the mid-answer positions "
            f"({mid_med:.3g}); high boundary skill sits on near-constant activations."
        )
    else:
        verdict = (
            f"REFUTED (genuine): boundary absolute variance (median var_trace@L18 "
            f"{b_med:.3g}) is SUSTAINED — {b_over_mid:.2f}x the mid-answer positions "
            f"({mid_med:.3g}) and {b_over_agg:.2f}x the aggregate mean/maxp "
            f"({agg_med:.3g}). The high boundary reconstruction skill reflects a "
            f"genuine low-dimensional context summary carried at a constant token, "
            f"not a variance-ratio inflation on a near-constant position."
        )

    out = {
        "analysis": "issue810_adhoc_predictability_vs_variance",
        "question": (
            "Is high boundary-position reconstruction skill (im_end/turn_nl/tail_1) "
            "an artifact of low absolute cross-context activation variance "
            "(near-constant token), which inflates the skill-over-mean R² RATIO?"
        ),
        "fixed_layer": FIXED_LAYER,
        "n_contexts": len(ctx_ids),
        "positions_ordered": POSITIONS_ORDERED,
        "token_entropy_note": (
            "answer_position_sweep store carries per-position probe-MEAN summary "
            "vectors + coverage only; it does NOT store per-position answer token "
            "ids, so token-identity entropy across contexts is not computable "
            "read-only (im_end/turn_nl are ~0 bits by construction; skipped, not "
            "re-tokenized per the analysis-only constraint)."
        ),
        "summaries": {
            "boundary_median_var_trace_L18": b_med,
            "mid_answer_median_var_trace_L18": mid_med,
            "aggregate_median_var_trace_L18": agg_med,
            "boundary_over_mid_ratio": b_over_mid,
            "boundary_over_aggregate_ratio": b_over_agg,
        },
        "by_position": rows,
        "interpretation": verdict,
        "readout_rho_note": (
            "readout_rho_L18 is the cross-behavior mean of rho_graded over the 3 "
            "read-out behaviors (harmful_compliance/refusal/sycophancy) for the "
            "trained_ridge arm; NON-load-bearing reference companion (the readout "
            "DV is per-(behavior,method), no single primary rho per summary)."
        ),
        "reproducibility": {
            "store_hf": f"{HF_DATA_REPO}:{HF_PREFIX}/{ANSWER_POSITION_SWEEP_SUBDIR}/",
            "v0_summaries_hf": f"{HF_DATA_REPO}:{I658_V0_SUMMARIES}",
            "recon_skill_json": "eval_results/issue_810/reconstruction_skill_by_summary.json",
            "readout_rho_json": "eval_results/issue_810/readout_rho_by_summary.json",
            "script": "scripts/issue810_adhoc_var_vs_skill.py",
        },
    }
    out_json = Path("eval_results/issue_810/adhoc_predictability_vs_variance.json")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(out_json, "w"), indent=2)
    print(f"[phase=wrote-json] {out_json}", flush=True)

    _make_figure(rows, out)
    print("[phase=done]", flush=True)
    return 0


def _make_figure(rows: dict, out: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = POSITIONS_ORDERED
    x = np.arange(len(names))
    skill = [rows[n]["skill_L18"] for n in names]
    skill_best = [rows[n]["skill_best_layer"] for n in names]
    vtr = [rows[n]["var_trace_L18"] for n in names]
    pr = [rows[n]["participation_ratio_L18"] for n in names]

    type_color = {
        "boundary": "#d62728",
        "tail": "#1f77b4",
        "head": "#2ca02c",
        "aggregate": "#7f2fd6",
        "other": "#888888",
    }
    colors = [type_color[rows[n]["pos_type"]] for n in names]

    fig = plt.figure(figsize=(15, 10))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.42, wspace=0.24)

    # (a) skill@L18 + best-layer skill
    axa = fig.add_subplot(gs[0, 0])
    axa.bar(x, skill, color=colors, alpha=0.85)
    axa.scatter(
        x,
        skill_best,
        color="black",
        marker="_",
        s=200,
        linewidths=2.0,
        zorder=5,
        label="best-layer skill",
    )
    axa.set_ylabel("reconstruction skill (R²)")
    axa.set_title("(a) True reconstruction skill @ L18 (bar) + best-layer skill (—)")
    axa.set_xticks(x)
    axa.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    axa.axhline(0, color="grey", lw=0.6)
    axa.legend(fontsize=8, loc="upper left")

    # (b) absolute variance trace @ L18 (log scale)
    axb = fig.add_subplot(gs[0, 1])
    axb.bar(x, vtr, color=colors, alpha=0.85)
    axb.set_yscale("log")
    axb.set_ylabel("cross-context var_trace @ L18 (log)")
    axb.set_title("(b) Absolute cross-context activation variance @ L18")
    axb.set_xticks(x)
    axb.set_xticklabels(names, rotation=60, ha="right", fontsize=8)

    # (c) participation ratio @ L18
    axc = fig.add_subplot(gs[1, 0])
    axc.bar(x, pr, color=colors, alpha=0.85)
    axc.set_ylabel("participation ratio (effective dim)")
    axc.set_title("(c) Effective dimensionality (participation ratio) @ L18")
    axc.set_xticks(x)
    axc.set_xticklabels(names, rotation=60, ha="right", fontsize=8)

    # (d) money panel: skill@L18 (y) vs absolute variance@L18 (x), labeled
    axd = fig.add_subplot(gs[1, 1])
    for n in names:
        vx = rows[n]["var_trace_L18"]
        sy = rows[n]["skill_L18"]
        if vx is None or sy is None:
            continue
        axd.scatter(vx, sy, color=type_color[rows[n]["pos_type"]], s=70, zorder=4)
        axd.annotate(n, (vx, sy), fontsize=7, xytext=(4, 3), textcoords="offset points")
    axd.set_xscale("log")
    axd.set_xlabel("cross-context var_trace @ L18 (log)")
    axd.set_ylabel("reconstruction skill (R²) @ L18")
    axd.set_title("(d) Skill vs absolute variance @ L18 (per position)")
    axd.axhline(0, color="grey", lw=0.6)
    handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w", markerfacecolor=type_color[t], markersize=8, label=t
        )
        for t in ("boundary", "tail", "head", "aggregate")
    ]
    axd.legend(handles=handles, fontsize=8, loc="best")

    s = out["summaries"]
    caption = (
        f"Answer-side summary positions @ layer 18 (n={out['n_contexts']} contexts). "
        f"Boundary (im_end/turn_nl) median var_trace {s['boundary_median_var_trace_L18']:.3g} "
        f"is {s['boundary_over_mid_ratio']:.2f}x mid-answer (tail_4/8) and "
        f"{s['boundary_over_aggregate_ratio']:.2f}x aggregate (mean/maxp): "
        + (
            "variance COLLAPSES at boundaries (artifact supported)."
            if s["boundary_median_var_trace_L18"] < 0.5 * s["mid_answer_median_var_trace_L18"]
            else "variance is SUSTAINED at boundaries (constancy-artifact refuted)."
        )
    )
    fig.suptitle(
        "#810 control: reconstruction predictability vs absolute activation variance", fontsize=13
    )
    fig.text(0.5, 0.005, caption, ha="center", va="bottom", fontsize=8, wrap=True)

    out_png = Path("figures/issue_810/adhoc_predictability_vs_variance.png")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"[phase=wrote-fig] {out_png}", flush=True)

    # meta.json alongside the figure
    import subprocess

    sha = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True
    ).stdout.strip()
    meta = {
        "figure": str(out_png),
        "script": "scripts/issue810_adhoc_var_vs_skill.py",
        "commit": sha,
        "fixed_layer": FIXED_LAYER,
        "n_contexts": out["n_contexts"],
        "caption": caption,
        "panels": {
            "a": "true reconstruction skill@L18 (bar) + best-layer skill (dash)",
            "b": "absolute cross-context var_trace@L18 (log)",
            "c": "participation ratio (effective dim)@L18",
            "d": "skill@L18 vs absolute var_trace@L18 scatter, labeled+colored by position type",
        },
        "interpretation": out["interpretation"],
    }
    json.dump(meta, open(str(out_png) + ".meta.json", "w"), indent=2)
    print(f"[phase=wrote-meta] {out_png}.meta.json", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
