#!/usr/bin/env python3
"""Issue #623 phase 6 — off-pod CPU analysis: cosine matrix + Spearman + bootstrap.

Runs on the VM AFTER the pod uploads persona centroids + the sycophancy trait
vector + syc_i.json (per CLAUDE.md "CPU-only phases don't hold GPU pods").

For each (layer, arm) it computes ``proj_i`` = alignment(persona_vector_i,
sycophancy_vector), drops the ``assistant`` baseline-self (pre-registered, plan
§4/§5/§7), hard-asserts no NaN / zero-norm vector enters the correlation array
(MF1 fail-fast — NEVER nan_policy='propagate'; the crash IS the signal), and
correlates ``proj_i`` against the reused #612 ``syc_i`` via Spearman rho +
B=10,000 case-resampling bootstrap (seed 623).

Arms (plan §5):
  lt_persona_lt_syc    — last-token persona vec vs last-token syc vec (HEADLINE)
  ravg_persona_lt_syc  — response-avg persona vec vs last-token syc vec (robustness i)
  lt_persona_ravg_syc  — last-token persona vec vs response-avg syc vec (robustness ii)
  ravg_persona_ravg_syc — both response-avg (upper bound of circularity inflation)
  gmc_persona_lt_syc   — global-mean-centered persona centroids (#536 robustness)
Each arm runs in cosine (primary) AND raw-dot (secondary) at all requested layers.

K2/K3 HALTs (plan §7) are hard raises, never silent fallbacks: K2 requires a
valid steering-effectiveness doc with k2_pass=true (headline layer pre-registered
by the steering criterion); K3 halts on a degenerate syc_i (writes
k3_halt_distribution.json with the per-persona base-rate distribution first).

Outputs:
  eval_results/issue_623/rho_by_layer.json   (rho + CI per layer per arm + metric)
  eval_results/issue_623/cosine_matrix.json  (proj_i per persona per layer per arm)
  eval_results/issue_623/k3_halt_distribution.json  (ONLY on a K3 DV-degeneracy halt)
  figures/issue_623/scatter_headline.png     (cosine scatter at headline layer)
  figures/issue_623/scatter_raw.png          (raw-dot scatter, raw-alongside-processed)
  figures/issue_623/rho_vs_layer.png
  figures/issue_623/rho_by_arm.png
  figures/issue_623/steering_effect_by_layer.png

Usage (off-pod, after artifacts are downloaded from HF):
  uv run python scripts/issue623_analyze.py --layers 7 14 21 27
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.experiments.persona_decomp_623 import (  # noqa: E402
    BASELINE_PERSONA,
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    DEFAULT_LAYERS,
    H1_RHO_THRESHOLD,
    K1_PANEL_FLOOR,
    K3_DISTINCT_FLOOR,
    reproducibility_metadata,
)
from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# Persona-vector arms: which centroid method feeds the persona vector + whether
# the centroids are globally mean-centered (#536) before the (i - assistant) diff.
_PERSONA_ARMS = {
    "lt_persona": {"method": "method_a", "centering": "none"},
    "ravg_persona": {"method": "method_b", "centering": "none"},
    "gmc_persona": {"method": "method_a", "centering": "global_mean"},
}
# Sycophancy trait-vector extraction points.
_SYC_POINTS = {"lt_syc": "last_token", "ravg_syc": "response_avg"}

# The 5 arms the plan §5 names (persona_arm x syc_point pairs that matter).
_ARMS = [
    ("lt_persona", "lt_syc"),  # HEADLINE
    ("ravg_persona", "lt_syc"),  # robustness i
    ("lt_persona", "ravg_syc"),  # robustness ii
    ("ravg_persona", "ravg_syc"),  # both response-avg
    ("gmc_persona", "lt_syc"),  # global-mean-centered (#536)
]
HEADLINE_ARM = ("lt_persona", "lt_syc")


def load_centroids(persona_vectors_dir: Path, method: str) -> dict[str, torch.Tensor]:
    """Load {persona: (n_layers, hidden) centroid} for a method (method_a|method_b)."""
    all_path = persona_vectors_dir / method / "all_centroids.pt"
    if all_path.exists():
        return torch.load(all_path, weights_only=True)
    # fall back to per-persona files
    out: dict[str, torch.Tensor] = {}
    for pt in sorted((persona_vectors_dir / method).glob("*.pt")):
        if pt.name in ("all_centroids.pt",):
            continue
        out[pt.stem] = torch.load(pt, weights_only=True)
    if not out:
        raise FileNotFoundError(f"No centroids found under {persona_vectors_dir / method}")
    return out


def load_trait_vector(trait_dir: Path, point: str, layer: int) -> torch.Tensor:
    """Load the sycophancy trait vector at one extraction point + layer."""
    fp = trait_dir / f"{point}_{layer}.pt"
    if not fp.exists():
        raise FileNotFoundError(f"Trait vector not found: {fp}")
    return torch.load(fp, weights_only=True)


def persona_vector_matrix(
    centroids: dict[str, torch.Tensor],
    personas: list[str],
    layer_index: int,
    centering: str,
) -> dict[str, np.ndarray]:
    """Build persona vectors (centroid_i - centroid_assistant) at one layer.

    With centering='global_mean', the centroid BANK is globally mean-centered
    (over the panel) before the (i - assistant) difference (#536 robustness arm).
    Returns {persona: vector (hidden,)}.
    """
    if BASELINE_PERSONA not in centroids:
        raise ValueError(
            f"Baseline persona {BASELINE_PERSONA!r} centroid missing — cannot subtract baseline."
        )

    # stack the layer-slice for all personas present in `personas`
    present = [p for p in personas if p in centroids]
    layer_slice = torch.stack([centroids[p][layer_index] for p in present])  # (P, hidden)

    if centering == "global_mean":
        # mean-center the bank, then take cosine-style centered centroids; the
        # (i - assistant) difference is taken on the centered bank.
        layer_slice = layer_slice - layer_slice.mean(dim=0, keepdim=True)

    by_persona = {p: layer_slice[i].float().numpy() for i, p in enumerate(present)}
    assistant_vec = by_persona[BASELINE_PERSONA]
    return {p: v - assistant_vec for p, v in by_persona.items()}


def alignment(pvec: np.ndarray, svec: np.ndarray, metric: str) -> float:
    """Cosine (primary) or raw dot (secondary) alignment."""
    dot = float(np.dot(pvec, svec))
    if metric == "dot":
        return dot
    pn = float(np.linalg.norm(pvec))
    sn = float(np.linalg.norm(svec))
    denom = pn * sn
    if denom == 0.0:
        return float("nan")  # zero-norm -> NaN; the MF1 guard catches it downstream
    return dot / denom


def bootstrap_ci(
    x: np.ndarray, y: np.ndarray, b: int, seed: int, alpha: float = 0.05
) -> tuple[float, float, float]:
    """Spearman rho + case-resampling bootstrap CI over personas. Returns (rho, lo, hi)."""
    rho = float(spearmanr(x, y).correlation)
    rng = np.random.default_rng(seed)
    n = len(x)
    boot = np.empty(b, dtype=float)
    for i in range(b):
        idx = rng.integers(0, n, size=n)
        # a resample with no variance in either axis gives NaN rho; treat as 0
        r = spearmanr(x[idx], y[idx]).correlation
        boot[i] = 0.0 if (r is None or np.isnan(r)) else r
    lo = float(np.quantile(boot, alpha / 2))
    hi = float(np.quantile(boot, 1 - alpha / 2))
    return rho, lo, hi


def run_arm(
    persona_arm: str,
    syc_point_key: str,
    metric: str,
    layers: list[int],
    persona_vectors_dir: Path,
    trait_dir: Path,
    syc_i: dict[str, float],
    correlation_personas: list[str],
) -> dict[int, dict]:
    """Run one (persona_arm, syc_point, metric) arm across all layers.

    Returns {layer: {rho, ci_lo, ci_hi, n, proj: {persona: value}}}.
    MF1 fail-fast: raises on any NaN / zero-norm proj entering the Spearman array.
    """
    cfg = _PERSONA_ARMS[persona_arm]
    centroids = load_centroids(persona_vectors_dir, cfg["method"])
    syc_point = _SYC_POINTS[syc_point_key]

    # personas present in BOTH centroids and syc_i (incl. baseline for the vector math)
    extraction_personas = [p for p in centroids if p in syc_i or p == BASELINE_PERSONA]

    out: dict[int, dict] = {}
    for layer_idx, layer in enumerate(layers):
        svec = load_trait_vector(trait_dir, syc_point, layer).float().numpy()
        pvecs = persona_vector_matrix(centroids, extraction_personas, layer_idx, cfg["centering"])

        proj: dict[str, float] = {}
        for persona in correlation_personas:
            if persona not in pvecs:
                continue
            proj[persona] = alignment(pvecs[persona], svec, metric)

        # baseline-self proj (recorded for the zero-vector invariant; NOT in rho)
        baseline_proj = None
        if BASELINE_PERSONA in pvecs:
            baseline_proj = alignment(pvecs[BASELINE_PERSONA], svec, metric)

        ordered = [p for p in correlation_personas if p in proj]
        x = np.array([proj[p] for p in ordered], dtype=float)
        y = np.array([syc_i[p] for p in ordered], dtype=float)

        # ── MF1 fail-fast guard: NO NaN / zero-norm into Spearman ──
        if np.isnan(x).any():
            bad = [ordered[i] for i in np.where(np.isnan(x))[0]]
            raise ValueError(
                f"MF1 guard: NaN/zero-norm proj_i for personas {bad} at layer {layer} "
                f"(arm={persona_arm}/{syc_point_key}/{metric}). The assistant baseline-self "
                "is already dropped; any other NaN is a bug. Refusing to run spearmanr."
            )
        if np.isnan(y).any():
            bad = [ordered[i] for i in np.where(np.isnan(y))[0]]
            raise ValueError(f"MF1 guard: NaN syc_i for personas {bad} (data corruption).")

        rho, lo, hi = bootstrap_ci(x, y, BOOTSTRAP_B, BOOTSTRAP_SEED)
        out[layer] = {
            "rho": rho,
            "ci_lo": lo,
            "ci_hi": hi,
            "n": len(ordered),
            "metric": metric,
            "proj": proj,
            "baseline_self_proj": baseline_proj,
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #623 phase 6 — off-pod analysis.")
    parser.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    parser.add_argument(
        "--persona-vectors-dir",
        default="data/persona_vectors/issue623",
        help="Dir with method_a/ method_b/ centroids (relative to repo root).",
    )
    parser.add_argument(
        "--trait-dir",
        default="data/persona_vectors/issue623/sycophancy_trait",
        help="Dir with <point>_<layer>.pt trait vectors (relative to repo root).",
    )
    parser.add_argument(
        "--syc-i",
        default="eval_results/issue_623/syc_i.json",
        help="syc_i.json from phase 5 (relative to repo root).",
    )
    parser.add_argument(
        "--steering-effect",
        default="eval_results/issue_623/steering_effect_by_layer.json",
        help="steering_effect_by_layer.json from phase 4 (for headline-layer pick).",
    )
    parser.add_argument(
        "--out-dir",
        default="eval_results/issue_623",
        help="Output dir for rho_by_layer.json / cosine_matrix.json (relative to repo root).",
    )
    parser.add_argument(
        "--fig-dir",
        default="figures/issue_623",
        help="Output dir for figures (relative to repo root).",
    )
    args = parser.parse_args()

    load_dotenv()

    def resolve(p: str) -> Path:
        return PROJECT_ROOT / p if not Path(p).is_absolute() else Path(p)

    persona_vectors_dir = resolve(args.persona_vectors_dir)
    trait_dir = resolve(args.trait_dir)
    out_dir = resolve(args.out_dir)
    fig_dir = resolve(args.fig_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    layers = args.layers

    # load syc_i + the correlation panel (baseline-self already flagged)
    syc_doc = json.loads(resolve(args.syc_i).read_text())
    syc_entries = syc_doc["syc_i"]
    syc_i = {p: v["syc_i"] for p, v in syc_entries.items()}
    correlation_personas = [p for p, v in syc_entries.items() if not v["is_baseline_self"]]

    n_corr = len(correlation_personas)
    print(f"[phase=analyze] correlation panel N={n_corr} (baseline-self dropped)", flush=True)
    k1_pass = n_corr >= K1_PANEL_FLOOR

    # ── K3 (DV degeneracy, plan §7): syc_i is identical across all arms/layers
    # (it is the persona's behavioral base rate, the y-axis of every Spearman), so
    # the degeneracy check fires ONCE at panel level BEFORE any bootstrap. If syc_i
    # has near-zero variance (all personas at the same base rate) the correlation
    # is undefined → write the base-rate distribution + HALT; no rho headline. The
    # per-resample NaN guard inside bootstrap_ci() handles resample-only degeneracy
    # and stays in place. ──
    y_panel = np.array([syc_i[p] for p in correlation_personas], dtype=float)
    n_distinct = len(set(y_panel.tolist()))
    y_var = float(np.var(y_panel))
    if n_distinct < K3_DISTINCT_FLOOR or y_var <= 0.0:
        distribution = {p: syc_i[p] for p in sorted(correlation_personas)}
        (out_dir / "k3_halt_distribution.json").write_text(
            json.dumps(
                {
                    "metadata": reproducibility_metadata({"k1_pass": k1_pass}),
                    "n_correlation": n_corr,
                    "n_distinct_syc_i": n_distinct,
                    "syc_i_variance": y_var,
                    "k3_distinct_floor": K3_DISTINCT_FLOOR,
                    "syc_i_distribution": distribution,
                },
                indent=2,
            )
        )
        raise ValueError(
            f"K3 HALT: syc_i degenerate — {n_distinct} distinct values across "
            f"N={n_corr} personas (floor {K3_DISTINCT_FLOOR}); var={y_var:.6f}. The "
            f"Spearman correlation is undefined on a near-constant DV; no rho headline. "
            f"Base-rate distribution written to {out_dir / 'k3_halt_distribution.json'}."
        )

    # ── Headline layer is the pre-registered steering-effectiveness pick (plan
    # §0/§6/§11), NOT a silent legacy-layer fallback. The steering probe (phase 4)
    # is the K2 gate: it both selects the most-causal layer AND validates the
    # sycophancy vector actually steers (k2_pass). A missing/invalid steering doc
    # or k2_pass=false is a HALT (project fail-fast posture: the crash IS the
    # signal) — never quietly default to layer 21, which would hide the K2 failure
    # AND bias the headline toward the legacy layer. ──
    steering_path = resolve(args.steering_effect)
    if not steering_path.exists():
        raise ValueError(
            f"K2 HALT: steering-effectiveness selection required for the headline layer; "
            f"file missing at {steering_path}. Re-run the steering probe before analysis."
        )
    steering_doc = json.loads(steering_path.read_text())
    k2_pass = bool(steering_doc.get("k2_pass", False))
    if not k2_pass:
        raise ValueError(
            f"K2 HALT: steering probe did not validate the sycophancy trait vector "
            f"(k2_pass={k2_pass}); no headline rho should be reported. Diagnose the "
            f"extraction (trait-description / judge-threshold) before reading rho. "
            f"Steering doc: {steering_doc}"
        )
    headline_layer = steering_doc.get("headline_layer")
    if headline_layer not in layers:
        raise ValueError(
            f"K2 HALT: steering-selected headline_layer={headline_layer!r} not in the "
            f"requested layer sweep {layers}. Steering doc: {steering_doc}"
        )

    # ── run all arms x metrics x layers ──
    rho_by_layer: dict[str, dict] = {}
    cosine_matrix: dict[str, dict] = {}
    for persona_arm, syc_point_key in _ARMS:
        for metric in ("cosine", "dot"):
            arm_key = f"{persona_arm}_{syc_point_key}" + ("" if metric == "cosine" else "_dot")
            try:
                result = run_arm(
                    persona_arm,
                    syc_point_key,
                    metric,
                    layers,
                    persona_vectors_dir,
                    trait_dir,
                    syc_i,
                    correlation_personas,
                )
            except FileNotFoundError as e:
                # a robustness arm whose centroids weren't extracted (e.g. method_b
                # skipped) is reported as absent, NOT silently zeroed.
                print(f"[phase=analyze] arm {arm_key} skipped: {e}", flush=True)
                continue
            rho_by_layer[arm_key] = {
                str(layer): {
                    "rho": result[layer]["rho"],
                    "ci_lo": result[layer]["ci_lo"],
                    "ci_hi": result[layer]["ci_hi"],
                    "n": result[layer]["n"],
                    "baseline_self_proj": result[layer]["baseline_self_proj"],
                }
                for layer in layers
            }
            cosine_matrix[arm_key] = {str(layer): result[layer]["proj"] for layer in layers}

    # circularity gap: headline (lt persona) cosine rho vs response-avg persona
    # rho at the headline layer
    gap = None
    h_key = f"{HEADLINE_ARM[0]}_{HEADLINE_ARM[1]}"
    ravg_key = f"ravg_persona_{HEADLINE_ARM[1]}"
    if h_key in rho_by_layer and ravg_key in rho_by_layer:
        hl = str(headline_layer)
        gap = {
            "headline_layer": headline_layer,
            "last_token_rho": rho_by_layer[h_key][hl]["rho"],
            "response_avg_rho": rho_by_layer[ravg_key][hl]["rho"],
            "gap": rho_by_layer[ravg_key][hl]["rho"] - rho_by_layer[h_key][hl]["rho"],
        }

    headline = None
    if h_key in rho_by_layer:
        hl = str(headline_layer)
        hrho = rho_by_layer[h_key][hl]
        headline = {
            "arm": h_key,
            "layer": headline_layer,
            "rho": hrho["rho"],
            "ci_lo": hrho["ci_lo"],
            "ci_hi": hrho["ci_hi"],
            "n": hrho["n"],
            "h1_confirmed": (hrho["rho"] >= H1_RHO_THRESHOLD and hrho["ci_lo"] > 0.0),
            "h0_null": (hrho["ci_lo"] <= 0.0 <= hrho["ci_hi"]),
        }

    meta = reproducibility_metadata(
        {
            "n_correlation": n_corr,
            "k1_pass": k1_pass,
            "headline_layer": headline_layer,
            "layers": layers,
        }
    )
    rho_doc = {
        "schema_version": 1,
        "metadata": meta,
        "headline": headline,
        "circularity_gap": gap,
        "rho_by_layer": rho_by_layer,
    }
    (out_dir / "rho_by_layer.json").write_text(json.dumps(rho_doc, indent=2))
    (out_dir / "cosine_matrix.json").write_text(
        json.dumps({"metadata": meta, "cosine_matrix": cosine_matrix}, indent=2)
    )
    print(f"[phase=analyze] wrote rho_by_layer.json + cosine_matrix.json -> {out_dir}", flush=True)

    # ── figures ──
    _make_figures(
        fig_dir,
        rho_by_layer,
        cosine_matrix,
        syc_i,
        correlation_personas,
        layers,
        headline_layer,
        h_key,
        steering_path,
    )
    print(f"[phase=analyze] wrote figures -> {fig_dir}", flush=True)
    print("[phase=analyze] done", flush=True)


def _make_figures(
    fig_dir: Path,
    rho_by_layer: dict,
    cosine_matrix: dict,
    syc_i: dict[str, float],
    correlation_personas: list[str],
    layers: list[int],
    headline_layer: int,
    h_key: str,
    steering_path: Path,
) -> None:
    """Hero scatter + raw-dot scatter + rho-vs-layer + per-arm bars + steering bar.

    No annotation overlays (no-plot-annotations rule). Each figure pairs a small
    meta.json with the commit + params.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from explore_persona_space.analysis.paper_plots import set_paper_style

        set_paper_style()
    except Exception:
        pass

    hl = str(headline_layer)
    meta = reproducibility_metadata({"headline_layer": headline_layer})

    def _save(fig, name: str, extra: dict) -> None:
        fig.savefig(fig_dir / name, dpi=200, bbox_inches="tight")
        plt.close(fig)
        (fig_dir / (name.rsplit(".", 1)[0] + "_meta.json")).write_text(
            json.dumps({**meta, **extra}, indent=2)
        )

    # Hero: cosine scatter at headline layer
    if h_key in cosine_matrix and hl in cosine_matrix[h_key]:
        proj = cosine_matrix[h_key][hl]
        pers = [p for p in correlation_personas if p in proj]
        xs = [proj[p] for p in pers]
        ys = [syc_i[p] for p in pers]
        rho_info = rho_by_layer[h_key][hl]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(xs, ys)
        ax.set_xlabel("cosine(persona vector, sycophancy vector)")
        ax.set_ylabel("base sycophancy rate (syc_i)")
        ax.set_title(
            f"Persona-vector alignment vs base sycophancy (layer {headline_layer})\n"
            f"Spearman rho={rho_info['rho']:.3f} "
            f"[{rho_info['ci_lo']:.3f}, {rho_info['ci_hi']:.3f}], N={rho_info['n']}"
        )
        _save(fig, "scatter_headline.png", {"arm": h_key, "metric": "cosine"})

    # Raw-dot scatter at headline layer (raw-alongside-processed)
    dot_key = f"{h_key}_dot"
    if dot_key in cosine_matrix and hl in cosine_matrix[dot_key]:
        proj = cosine_matrix[dot_key][hl]
        pers = [p for p in correlation_personas if p in proj]
        xs = [proj[p] for p in pers]
        ys = [syc_i[p] for p in pers]
        rho_info = rho_by_layer[dot_key][hl]
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.scatter(xs, ys)
        ax.set_xlabel("raw dot(persona vector, sycophancy vector)")
        ax.set_ylabel("base sycophancy rate (syc_i)")
        ax.set_title(
            f"Raw-dot alignment vs base sycophancy (layer {headline_layer})\n"
            f"Spearman rho={rho_info['rho']:.3f} "
            f"[{rho_info['ci_lo']:.3f}, {rho_info['ci_hi']:.3f}], N={rho_info['n']}"
        )
        _save(fig, "scatter_raw.png", {"arm": dot_key, "metric": "dot"})

    # rho-vs-layer for the headline arm (cosine)
    if h_key in rho_by_layer:
        rhos = [rho_by_layer[h_key][str(layer)]["rho"] for layer in layers]
        los = [rho_by_layer[h_key][str(layer)]["ci_lo"] for layer in layers]
        his = [rho_by_layer[h_key][str(layer)]["ci_hi"] for layer in layers]
        yerr = np.array(
            [
                [r - lo for r, lo in zip(rhos, los, strict=True)],
                [hi - r for r, hi in zip(rhos, his, strict=True)],
            ]
        )
        yerr = np.clip(yerr, 0.0, None)  # constant-bootstrap epsilon-negative guard
        fig, ax = plt.subplots(figsize=(5, 4))
        ax.errorbar(layers, rhos, yerr=yerr, marker="o", capsize=3)
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.set_xlabel("layer")
        ax.set_ylabel("Spearman rho(proj, syc)")
        ax.set_title("rho vs layer (last-token persona vs last-token sycophancy)")
        _save(fig, "rho_vs_layer.png", {"arm": h_key})

    # per-arm rho bars at headline layer (cosine arms)
    cos_arms = [k for k in rho_by_layer if not k.endswith("_dot")]
    if cos_arms:
        names = cos_arms
        rhos = [rho_by_layer[k][hl]["rho"] for k in names]
        los = [rho_by_layer[k][hl]["ci_lo"] for k in names]
        his = [rho_by_layer[k][hl]["ci_hi"] for k in names]
        yerr = np.array(
            [
                [r - lo for r, lo in zip(rhos, los, strict=True)],
                [hi - r for r, hi in zip(rhos, his, strict=True)],
            ]
        )
        yerr = np.clip(yerr, 0.0, None)
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(range(len(names)), rhos, yerr=yerr, capsize=3)
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=30, ha="right")
        ax.set_ylabel("Spearman rho")
        ax.set_title(f"rho by arm (layer {headline_layer})")
        _save(fig, "rho_by_arm.png", {"layer": headline_layer})

    # steering effect by layer
    if steering_path.exists():
        sdoc = json.loads(steering_path.read_text())
        per_layer = sdoc.get("per_layer", {})
        if per_layer:
            ls = sorted(int(k) for k in per_layer)
            effs = [per_layer[str(layer)] for layer in ls]
            fig, ax = plt.subplots(figsize=(5, 4))
            ax.bar([str(layer) for layer in ls], effs)
            ax.set_xlabel("layer")
            ax.set_ylabel("steering effect (trait-score lift)")
            ax.set_title("Sycophancy-vector steering effect by layer")
            _save(fig, "steering_effect_by_layer.png", {})


if __name__ == "__main__":
    main()
