#!/usr/bin/env python
"""#1947 P6 analysis (VM, 0 GPU): the theory-assumption battery on exactly the
trained rows — plan §4.4 P6 / §6.

Per (verdict arm × layer × tree): write direction ŵ_tr (mean per-row answer
shift), `rank_read` (top-1 singular share, H2), alignments cos(ŵ_tr, ·) vs
{δ, r_B, marker W_U row, cross-behavior δ} against 2,000-draw corpus-cov +
isotropic norm-matched nulls + the shuffled-row null (H3), the base-geometry
`gate_read` on trained rows AND corpus (H4), row-cluster bootstrap CIs
(B=2,000, batched multinomial-weight GEMMs — the #778 convention, no serial
draw loops), δ split-half reliability at n=300 (§6 criterion 5), the
map-change D forest (P5 fit JSONs), and the registered analyzer-frame
re-reductions (§6 notes 1-9, flag-gated over the persisted per-row stores).

Directive compliance (task #1947 epm:progress v7 + v9): within-run context
reads use the LAST-TOKEN summary (``last_prompt`` — the final token of the
generation-rendered prompt, the #1768 lasttoken-repool position; NOT the last
user-content token ``last_ctx``) as PRIMARY with span-mean retained as
SECONDARY (the #1768-comparability surface); the D-vs-M0 comparability read
stays span-mean (the reused r3 M0/floors are span-mean).
Reporting is PER LAYER at {14, 19, 25} — no cross-layer max headline (the
selection-symmetric per-axis carve-out, plan §6).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import os  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
REPO_ROOT = _SCRIPTS_DIR.parent

import issue1947_cells as cells  # noqa: E402

logger = logging.getLogger("issue1947.analysis")

ISSUE = cells.ISSUE
LAYERS = (14, 19, 25)
N_BOOT = 2000  # row-cluster bootstrap draws (plan §6 horse-race CIs)
BOOT_SEED = 653  # #1481 convention (plan §10)
GATE_BAND = (0.3, 0.7)  # H4 theory band
TREES = ("matched_text", "onpolicy")
CTX_PRIMARY = "last_prompt"  # directive v9 PRIMARY: final rendered-prompt token
CTX_SECONDARY = "context"  # span-mean (SECONDARY + #1768 comparability)
FRAMES_ALL = ("h6-n-match", "h3-20row", "consumed-reliability")


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _meta() -> dict:
    return {"issue": ISSUE, "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


@dataclasses.dataclass
class Cfg:
    out_root: Path
    out_dir: Path
    fig_dir: Path
    layers: tuple[int, ...] = LAYERS
    smoke: bool = False
    frames: tuple[str, ...] = ()
    arms: tuple[str, ...] = ()
    seed: int = BOOT_SEED


def _load_store(path: Path) -> dict:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def _rows(store: dict, span: str, layer: int) -> np.ndarray:
    return np.asarray(store["arms"][span][layer].float().numpy(), dtype=np.float64)


def _join(a: dict, b: dict, span: str, layer: int) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Sha-joined row pair (a-order) — alignment by sha, never order (#1768)."""
    ib = {s: i for i, s in enumerate(b["row_sha"])}
    keep = [i for i, s in enumerate(a["row_sha"]) if s in ib]
    assert keep, "sha join empty"
    A = _rows(a, span, layer)[keep]
    B = _rows(b, span, layer)[[ib[a["row_sha"][i]] for i in keep]]
    qidx = [int(a["row_question_idx"][i]) for i in keep]
    return A, B, qidx


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(a @ b / (na * nb)) if na > 0 and nb > 0 else float("nan")


def _boot_cos_ci(
    stack: np.ndarray, cand: np.ndarray, n_draws: int, seed: int
) -> tuple[list[float], float]:
    """Row-cluster bootstrap CI on cos(mean(stack_b), cand) — ONE multinomial
    weight GEMM over all draws (the #778 batched convention)."""
    rng = np.random.default_rng(seed)
    n = stack.shape[0]
    W = rng.multinomial(n, np.full(n, 1.0 / n), size=n_draws).astype(np.float64) / n
    means = W @ stack  # (draws, d)
    cn = cand / (np.linalg.norm(cand) + 1e-12)
    num = means @ cn
    cos = num / (np.linalg.norm(means, axis=1) + 1e-12)
    return [float(np.quantile(cos, 0.025)), float(np.quantile(cos, 0.975))], float(cos.mean())


# ── inputs (reused artifacts + this run's stores) ────────────────────────────


def _delta_pool_dir(cfg: Cfg, beh_key: str, ctx_key: str) -> Path:
    return cfg.out_root / "delta_tf" / f"{beh_key}-{ctx_key}-delta1947"


def _panel_v0_halves(cfg: Cfg, slug: str, layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Base panel own-capture response rows split by QUESTION parity — the
    disjoint-halves baseline legs (v0_A even / v0_B odd; #1768 convention)."""
    store = _load_store(cfg.out_root / "battery" / "panel" / slug / "pooled_base.pt")
    R = _rows(store, "response", layer)
    qidx = np.asarray([int(q) for q in store["row_question_idx"]])
    even, odd = R[qidx % 2 == 0], R[qidx % 2 == 1]
    assert len(even) and len(odd), (slug, "panel base halves empty")
    return even.mean(axis=0), odd.mean(axis=0)


def _delta_legs(cfg: Cfg, slug: str, layer: int) -> dict:
    """δ legs + split-half reliability (quarter-split disjoint baselines —
    the #1768 noise-structure fix; shared-B version recorded, never primary)."""
    import torch

    cell = cells.CELL_BY_SLUG[slug]
    tb = torch.load(
        _delta_pool_dir(cfg, cell.beh_key, cell.ctx_key) / "tbar.pt",
        map_location="cpu",
        weights_only=False,
    )
    v0_a, v0_b = _panel_v0_halves(cfg, slug, layer)
    tbar = np.asarray(tb["tbar"][layer].numpy(), dtype=np.float64)
    delta = tbar - v0_b  # w legs use v0_A downstream; δ leg uses v0_B (disjoint)
    out = {"delta": delta, "v0_a": v0_a, "v0_b": v0_b, "n_rows": int(tb["n_rows"])}
    if tb.get("tbar_even") is not None:
        te = np.asarray(tb["tbar_even"][layer].numpy(), dtype=np.float64)
        to = np.asarray(tb["tbar_odd"][layer].numpy(), dtype=np.float64)
        out["split_half_r_disjoint"] = _cos(te - v0_a, to - v0_b)
        out["split_half_r_sharedB"] = _cos(te - v0_b, to - v0_b)  # record-only
    return out


def _stack_for_tree(cfg: Cfg, slug: str, tree: str, layer: int, subset: str = "full"):
    """Per-row answer-shift stack (trained − base) for one tree."""
    if tree == "matched_text":
        d = cfg.out_root / "battery" / "trained_rows" / slug
        a = _load_store(d / ("pooled.pt" if subset == "full" else "pooled_consumed.pt"))
        b = _load_store(d / ("pooled_base.pt" if subset == "full" else "pooled_base_consumed.pt"))
    elif tree == "onpolicy":
        d = cfg.out_root / "battery" / "onpolicy" / slug
        a, b = _load_store(d / "pooled.pt"), _load_store(d / "pooled_base.pt")
    else:
        raise ValueError(tree)
    A, B, qidx = _join(a, b, "response", layer)
    ctx_p = ctx_s = c_src = None
    if "arms" in b and CTX_PRIMARY in b["arms"]:
        Cp, _B2, _ = _join(b, a, CTX_PRIMARY, layer)
        Cs, _B3, _ = _join(b, a, CTX_SECONDARY, layer)
        ctx_p, ctx_s = Cp, Cs
        c_src = Cp.mean(axis=0)
    return {
        "stack": A - B,
        "qidx": qidx,
        "ctx_primary": ctx_p,
        "ctx_secondary": ctx_s,
        "c_src": c_src,
        "store_a": a,
        "store_b": b,
    }


# ── per-arm battery ──────────────────────────────────────────────────────────


def arm_battery(cfg: Cfg, slug: str, layer: int, shared: dict) -> dict:
    """One (arm × layer) battery: rank / alignments+nulls / gate / CIs, both
    trees, per-layer reporting (no cross-layer max)."""
    import issue1768_directions as DIRS

    cell = cells.CELL_BY_SLUG[slug]
    rng = np.random.default_rng(cfg.seed + layer)
    legs = _delta_legs(cfg, slug, layer) if cell.kind == "content" else None
    out: dict = {"slug": slug, "layer": layer}
    if legs is not None:
        out["delta_split_half_r_disjoint"] = legs.get("split_half_r_disjoint")
        out["delta_split_half_r_sharedB"] = legs.get("split_half_r_sharedB")
        out["delta_n_rows"] = legs["n_rows"]
    for tree in TREES:
        try:
            t = _stack_for_tree(cfg, slug, tree, layer)
        except FileNotFoundError as e:
            out[tree] = {"missing": str(e)}
            continue
        stack = t["stack"]
        w = stack.mean(axis=0)
        rec: dict = {"n_rows": int(stack.shape[0])}
        rec["rank"] = DIRS.rank_read(stack, w)
        cands: dict[str, np.ndarray] = {}
        if legs is not None:
            cands["delta"] = legs["delta"]
        rb = shared.get("rb") or {}
        if cell.beh_key in rb:
            cands["r_b"] = np.asarray(rb[cell.beh_key][layer], dtype=np.float64)
        if cell.kind == "marker" and shared.get("wu_row") is not None:
            cands["wu_marker_row"] = shared["wu_row"]
        for other, dvec in (shared.get("cross_deltas") or {}).items():
            if other != cell.beh_key and layer in dvec:
                cands[f"delta_cross_{other}"] = dvec[layer]
        aligns: dict[str, dict] = {}
        sigma = shared.get("sigma_by_layer", {}).get(layer)
        if sigma is None and cfg.smoke:
            d = stack.shape[1]  # smoke-synthetic Σ: the REAL gate/null bodies still run
            sigma = {
                "sigma": np.eye(d),
                "top_eig": np.eye(d)[0],
                "chol": np.eye(d),
                "n_rows": stack.shape[0],
            }
            rec["smoke_synthetic_sigma"] = True
        for name, cand in cands.items():
            cand = np.asarray(cand, dtype=np.float64)
            if cand.shape[-1] != stack.shape[1]:  # stated inapplicable, never silent
                aligns[name] = {
                    "inapplicable": f"candidate dim {cand.shape[-1]} != stack {stack.shape[1]}"
                }
                continue
            ci, mean_cos = _boot_cos_ci(stack, cand, N_BOOT, cfg.seed + layer)
            entry = {"cos": _cos(w, cand), "boot_ci95": ci, "boot_mean": mean_cos}
            if sigma is not None:
                entry["null_bands"] = DIRS.null_bands(np.asarray(cand), sigma, rng)
            if name == "delta" and legs is not None and tree == "onpolicy":
                b_store = t["store_b"]
                v0_rows = _rows(b_store, "response", layer)
                a_rows = v0_rows + stack  # trained rows (sha-joined order)
                entry["shuffled_row_band"] = DIRS.shuffled_row_band(
                    a_rows, v0_rows, np.asarray(cand), rng
                )
            aligns[name] = entry
        rec["alignments"] = aligns
        if t["c_src"] is not None and sigma is not None:
            rec["gate_trained_rows"] = {
                "primary_last_prompt": DIRS.gate_read(
                    t["ctx_primary"], stack, t["c_src"], w, sigma
                ),
                "secondary_context_mean": DIRS.gate_read(
                    t["ctx_secondary"],
                    stack,
                    np.asarray(t["ctx_secondary"]).mean(axis=0),
                    w,
                    sigma,
                ),
            }
        out[tree] = rec
    return out


def _shared_inputs(cfg: Cfg) -> dict:
    """Reused direction inputs: r_B stacks, corpus Σ per layer, marker W_U row,
    per-pool δ vectors for the cross-behavior control."""
    import issue1768_directions as DIRS

    shared: dict = {}
    try:
        shared["rb"] = DIRS.load_rb_tensors(cfg.out_root)
    except Exception as e:  # noqa: BLE001 — named scope caveat, battery continues
        logger.warning("[p6] r_B tensors unavailable (%s) — alignment rows omitted", e)
        shared["rb"] = {}
    sigma_by_layer = {}
    for layer in cfg.layers:
        try:
            sigma_by_layer[layer] = DIRS.corpus_sigma(cfg.out_root, layer)
        except FileNotFoundError as e:
            if not cfg.smoke:  # H4 gate + null bands are verdict-bearing (fail loud)
                raise RuntimeError(
                    f"[p6] corpus sigma L{layer} unavailable — stage the #1768 "
                    "base_content store + pfx sample (battery --phase probe names them)"
                ) from e
            logger.warning("[p6] corpus sigma L%d unavailable in smoke (%s)", layer, e)
    shared["sigma_by_layer"] = sigma_by_layer
    try:
        shared["wu_row"] = DIRS.load_wu_row(
            "Qwen/Qwen2.5-7B-Instruct"
        )  # marker unembedding row (gauge-free race candidate)
    except Exception as e:  # noqa: BLE001
        logger.warning("[p6] W_U marker row unavailable (%s)", e)
        shared["wu_row"] = None
    cross: dict[str, dict[int, np.ndarray]] = {}
    for beh in cells.BEH_KEYS:
        d = _delta_pool_dir(cfg, beh, "pers")
        if (d / "tbar.pt").exists():
            import torch

            tb = torch.load(d / "tbar.pt", map_location="cpu", weights_only=False)
            cross[beh] = {
                li: np.asarray(t.numpy(), dtype=np.float64) for li, t in tb["tbar"].items()
            }
    shared["cross_deltas"] = cross
    return shared


# ── registered analyzer-frame re-reductions (plan §6 notes 1-9) ─────────────


def frame_n_match(cfg: Cfg, slug: str, layer: int, n_rows: int = 80, n_pos: int = 20) -> dict:
    """§6 note 1 (H6 n-matching): recompute top-1 share + δ split-half on
    random 80-row / 20-positive subsamples of the persisted per-row stores."""
    import issue1768_directions as DIRS

    rng = np.random.default_rng(cfg.seed)
    t = _stack_for_tree(cfg, slug, "matched_text", layer)
    stack = t["stack"]
    reps = []
    for _ in range(8 if not cfg.smoke else 2):
        idx = rng.choice(stack.shape[0], size=min(n_rows, stack.shape[0]), replace=False)
        sub = stack[idx]
        reps.append(DIRS.rank_read(sub, sub.mean(axis=0))["top1_var_share"])
    return {
        "slug": slug,
        "layer": layer,
        "n_rows": n_rows,
        "top1_share_subsampled": reps,
        "top1_share_mean": float(np.mean(reps)),
    }


def frame_consumed_reliability(cfg: Cfg, slug: str, layer: int) -> dict:
    """§6 note 8: consumed-positive counts + consumed-subset δ reliability
    (base response rows over consumed positives, even/odd split)."""
    d = cfg.out_root / "battery" / "trained_rows" / slug
    store = _load_store(d / "pooled_base_consumed.pt")
    kinds = store["metadata"].get("row_kinds") or []
    R = _rows(store, "response", layer)
    pos_ix = [i for i, k in enumerate(kinds) if k == "pos"]
    rec = {"slug": slug, "layer": layer, "n_consumed_pos": len(pos_ix)}
    if len(pos_ix) >= 4:
        P = R[pos_ix]
        v0_a, v0_b = _panel_v0_halves(cfg, slug, layer)
        rec["delta_consumed_split_half_r"] = _cos(
            P[0::2].mean(axis=0) - v0_a, P[1::2].mean(axis=0) - v0_b
        )
    return rec


def frame_20row_delta(cfg: Cfg, slug: str, layer: int) -> dict:
    """§6 note 2 (H3 rows-vs-precision): matched-text alignment recomputed on
    random 20-row δ subsamples (precision-matched to the parent's 20-row δ)."""
    t = _stack_for_tree(cfg, slug, "matched_text", layer)
    legs = _delta_legs(cfg, slug, layer)
    rng = np.random.default_rng(cfg.seed + 7)
    stack = t["stack"]
    coss = []
    for _ in range(20 if not cfg.smoke else 3):
        idx = rng.choice(stack.shape[0], size=min(20, stack.shape[0]), replace=False)
        coss.append(_cos(stack[idx].mean(axis=0), legs["delta"]))
    return {
        "slug": slug,
        "layer": layer,
        "cos_20row_draws": coss,
        "cos_20row_mean": float(np.mean(coss)),
    }


# ── D forest (P5 fit JSONs) ─────────────────────────────────────────────────


def collect_d_forest(cfg: Cfg) -> list[dict]:
    rows = []
    fits_dir = cfg.out_root / "fits"
    for p in sorted(fits_dir.rglob("*.json")) if fits_dir.exists() else []:
        try:
            rec = _read_json(p)
        except json.JSONDecodeError:
            continue
        mc = rec.get("map_change")
        if mc:
            rows.append(
                {
                    "file": str(p.relative_to(cfg.out_root)),
                    "arm": rec.get("arm_id") or rec.get("arm"),
                    "layer": rec.get("layer"),
                    "D": mc.get("D"),
                    "D_ci95": mc.get("D_ci95"),
                    "floor_p95": mc.get("floor_p95"),
                    "verdict": mc.get("verdict"),
                }
            )
    return rows


# ── figures (paper-plots conventions) ────────────────────────────────────────


def make_figures(cfg: Cfg, batteries: list[dict], d_forest: list[dict]) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

    set_paper_style()
    pal = paper_palette(4)
    cfg.fig_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    # Hero: per-assumption verdict grid (H2 top-1 share / H3 alignment / H4 gate)
    fig, axes = plt.subplots(3, len(cfg.layers), figsize=(4 * len(cfg.layers), 9), squeeze=False)
    for col, layer in enumerate(cfg.layers):
        rows = [b for b in batteries if b["layer"] == layer]
        labels = [b["slug"] for b in rows]
        x = np.arange(len(rows))
        for r, (field, title, band) in enumerate(
            (
                ("rank", "H2: top-1 singular share", (0.6, None)),
                ("align", "H3: cos(w, delta) matched-text", None),
                ("gate", "H4: gate Spearman rho", GATE_BAND),
            )
        ):
            ax = axes[r][col]
            vals = []
            for b in rows:
                mt = b.get("matched_text") or {}
                if field == "rank":
                    vals.append((mt.get("rank") or {}).get("top1_var_share", np.nan))
                elif field == "align":
                    vals.append(
                        ((mt.get("alignments") or {}).get("delta") or {}).get("cos", np.nan)
                    )
                else:
                    g = (mt.get("gate_trained_rows") or {}).get("primary_last_prompt") or {}
                    vals.append(g.get("spearman_rho", np.nan))
            ax.bar(x, vals, color=pal[r])
            if band:
                ax.axhline(band[0], color="0.3", lw=0.8, ls="--")
                if band[1] is not None:
                    ax.axhline(band[1], color="0.3", lw=0.8, ls="--")
            ax.set_title(f"{title} (L{layer})")
            ax.set_xticks(x)
            ax.set_xticklabels(labels, rotation=90, fontsize=5)
    fig.tight_layout()
    hero = cfg.fig_dir / "hero_verdict_grid.png"
    fig.savefig(hero, dpi=200)
    plt.close(fig)
    written.append(hero)

    if d_forest:
        fig, ax = plt.subplots(figsize=(7, max(3, 0.3 * len(d_forest))))
        ys = np.arange(len(d_forest))
        for y, row in zip(ys, d_forest):
            d, ci = row.get("D"), row.get("D_ci95") or [None, None]
            if d is None or ci[0] is None:
                continue
            err_lo = max(0.0, d - ci[0])  # NON-NEGATIVE offsets (#547/#1335)
            err_hi = max(0.0, ci[1] - d)
            ax.errorbar(d, y, xerr=[[err_lo], [err_hi]], fmt="o", color=pal[0])
        ax.axvline(0.0, color="0.3", lw=0.8)
        ax.set_yticks(ys)
        ax.set_yticklabels([f"{r['arm']} L{r['layer']}" for r in d_forest], fontsize=6)
        ax.set_xlabel("map-change D (minus refit floor)")
        fig.tight_layout()
        p = cfg.fig_dir / "d_forest.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        written.append(p)

    rel = [
        (b["slug"], b["layer"], b.get("delta_split_half_r_disjoint"))
        for b in batteries
        if b.get("delta_split_half_r_disjoint") is not None
    ]
    if rel:
        fig, ax = plt.subplots(figsize=(7, 3.5))
        ax.scatter(range(len(rel)), [r[2] for r in rel], color=pal[1])
        ax.axhline(0.55, color="0.3", lw=0.8, ls="--")  # the #1768 20-row reference
        ax.set_xticks(range(len(rel)))
        ax.set_xticklabels([f"{s} L{li}" for s, li, _ in rel], rotation=90, fontsize=5)
        ax.set_ylabel("delta split-half r (disjoint baselines)")
        fig.tight_layout()
        p = cfg.fig_dir / "delta_reliability.png"
        fig.savefig(p, dpi=200)
        plt.close(fig)
        written.append(p)
    return written


# ── main ─────────────────────────────────────────────────────────────────────


def _verdict_arms(cfg: Cfg) -> dict:
    for p in (
        cfg.out_dir / "verdict_manifest.json",
        REPO_ROOT / "eval_results/issue_1947/analysis/verdict_manifest.json",
    ):
        if p.exists():
            return _read_json(p)
    raise SystemExit("[i1947-p6] verdict_manifest.json missing — run battery --phase select")


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(name)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="#1947 P6 analysis battery")
    p.add_argument("--out-root", default=cells.OUT_ROOT_DEFAULT)
    p.add_argument("--out-dir", default=str(REPO_ROOT / "eval_results/issue_1947/analysis"))
    p.add_argument("--fig-dir", default=str(REPO_ROOT / "figures/issue_1947"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--layers", default=",".join(str(x) for x in LAYERS))
    p.add_argument("--arms", default="", help="comma-separated slug filter")
    p.add_argument("--frames", default="", help=f"comma list of {FRAMES_ALL} or 'all'")
    p.add_argument("--no-figures", action="store_true")
    p.add_argument("--import-check", action="store_true")
    args = p.parse_args(argv)
    if args.import_check:
        import issue1768_directions as DIRS

        from explore_persona_space.analysis.paper_plots import paper_palette, set_paper_style

        names = [
            DIRS.rank_read,
            DIRS.gate_read,
            DIRS.null_bands,
            DIRS.shuffled_row_band,
            DIRS.load_rb_tensors,
            DIRS.load_wu_row,
            DIRS.corpus_sigma,
            set_paper_style,
            paper_palette,
        ]
        print(f"[import-check] OK ({len(names)} symbols resolved)")
        return 0
    frames = tuple(FRAMES_ALL if args.frames == "all" else (f for f in args.frames.split(",") if f))
    cfg = Cfg(
        out_root=Path(args.out_root),
        out_dir=Path(args.out_dir),
        fig_dir=Path(args.fig_dir),
        layers=tuple(int(x) for x in args.layers.split(",") if x),
        smoke=args.smoke,
        frames=frames,
        arms=tuple(s for s in args.arms.split(",") if s),
    )
    man = _verdict_arms(cfg)
    slugs = sorted(man["content"]) + sorted(man["marker"])
    if cfg.arms:
        slugs = [s for s in slugs if s in cfg.arms]
    shared = _shared_inputs(cfg)
    batteries: list[dict] = []
    n_units = len(slugs) * len(cfg.layers)
    k = 0
    for slug in slugs:
        for layer in cfg.layers:
            k += 1
            t0 = time.time()
            try:
                rec = arm_battery(cfg, slug, layer, shared)
            except FileNotFoundError as e:
                rec = {"slug": slug, "layer": layer, "missing": str(e)}
            rec["lr"] = cells.CELL_BY_SLUG[slug].lr  # §6 note 7: realized LR per cell
            rec["parent_pass_count"] = cells.parent_pass_count(
                cells.CELL_BY_SLUG[slug].lr_source
            )  # §6 note 4 stratifier
            batteries.append(rec)
            _atomic_json(cfg.out_dir / "battery" / f"battery_{slug}_L{layer}.json", rec | _meta())
            print(f"[p6] unit {k}/{n_units} {slug} L{layer} elapsed={time.time() - t0:.1f}s")
    frame_out: dict[str, list] = {}
    for fr in cfg.frames:
        rows = []
        for slug in slugs:
            if slug.startswith("mk-"):
                continue
            for layer in cfg.layers:
                try:
                    if fr == "h6-n-match":
                        rows.append(frame_n_match(cfg, slug, layer))
                    elif fr == "h3-20row":
                        rows.append(frame_20row_delta(cfg, slug, layer))
                    elif fr == "consumed-reliability":
                        rows.append(frame_consumed_reliability(cfg, slug, layer))
                except FileNotFoundError as e:
                    rows.append({"slug": slug, "layer": layer, "missing": str(e)})
        frame_out[fr] = rows
        _atomic_json(cfg.out_dir / f"frame_{fr}.json", {"rows": rows, **_meta()})
    d_forest = collect_d_forest(cfg)
    summary = {
        "n_arms": len(slugs),
        "layers": list(cfg.layers),
        "context_summary_primary": CTX_PRIMARY,  # binding directive record
        "context_summary_secondary": CTX_SECONDARY,
        "coverage": man.get("coverage"),
        "d_forest": d_forest,
        "frames_run": list(cfg.frames),
        **_meta(),
    }
    _atomic_json(cfg.out_dir / "battery_summary.json", summary)
    figs: list[Path] = []
    if not args.no_figures:
        figs = make_figures(cfg, batteries, d_forest)
    print(f"[p6] done: {len(batteries)} battery cells, {len(figs)} figures")
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit — the PyGILState_Release atexit gotcha
