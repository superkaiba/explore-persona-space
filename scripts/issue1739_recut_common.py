"""Shared loaders + conventions for the #1739 publishable re-cut figures.

Pure aggregation over artifacts already committed under
`eval_results/issue_1739/` (worktree `issue-1739`) — no fits, no GPU, no
network. Every number a figure plots traces to ONE quantity in ONE file
through the loaders here.

Artifact map (all relative to WT below):
  <b>/arm_results/all_arms_spearman.json   train-rung arm_rows (16 arms,
                                           3 seeds x 5 draws), transfer_rows
                                           (6 arms, OOD), nulls, headlines
  <b>/arm_results/percell/cells.jsonl      per-cell split_half.ceiling_sb,
                                           max_over_arms_null, preds_npz name
  <b>/arm_results/percell/preds/*.npz      per-context dv + per-arm scores
                                           (train rung, all 16 arms)
  wide_ood/<b>_transfer.jsonl              OOD rungs with arms 7/8/12 added
                                           (9 arms, 3 seeds x 5 draws)
  wide/wildchat_rung/<b>/...json           WildChat rung, 10 arms, 1 replicate
                                           + per_layer_rows
  wide/wildchat_rung/<b>/preds/*.jsonl     per-context dv + score, WildChat
  bareq_map/evil/all_arms_spearman.json    meta.mapping_baselines (leg 2):
                                           identity+bias R2 + kNN retrieval
  wildchat_rung/spread/<b>.json            WildChat-rung DV spread
  dv_dataset/<b>/labeling.json             per-rung judged DV rows
"""

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import json  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
WT = ROOT / ".claude/worktrees/issue-1739/eval_results/issue_1739"
FIGDIR = ROOT / "figures/issue_1739/recuts"
NUMDIR = ROOT / "eval_results/issue_1739/recuts"

BEHAVIORS = ["evil", "sycophancy", "hallucination"]

# Max label budget per behavior (the "operating slice" L).
LMAX = {"evil": 8000, "sycophancy": 16000, "hallucination": 16000}

# Operating slice for the main + wide_ood grids.
OP = dict(regime="e1", u_rung_label="full", variant="context_end")
# Same slice minus the unlabeled-pool size, for callers that resolve U themselves.
OP_RV = dict(regime="e1", variant="context_end")

# Evaluation settings per behavior, in reporting order.
RUNGS = {
    "evil": ["train", "hhrt", "toxicchat", "wildchat_rung"],
    "sycophancy": ["train", "aita", "wildchat_rung"],
    "hallucination": ["train", "nqopen", "simpleqa", "wildchat_rung"],
}
OOD_RUNGS = {
    "evil": ["hhrt", "toxicchat"],
    "sycophancy": ["aita"],
    "hallucination": ["nqopen", "simpleqa"],
}
RUNG_LABEL = {
    ("evil", "train"): "held-out train\n(DAN x forbidden-q)",
    ("evil", "hhrt"): "hh-rlhf red-team\n(OOD)",
    ("evil", "toxicchat"): "ToxicChat\n(OOD)",
    ("sycophancy", "train"): "held-out train",
    ("sycophancy", "aita"): "AITA (OOD)",
    ("hallucination", "train"): "held-out TriviaQA",
    ("hallucination", "nqopen"): "NQ-Open (OOD)",
    ("hallucination", "simpleqa"): "SimpleQA (OOD)",
}
for _b in BEHAVIORS:
    RUNG_LABEL[(_b, "wildchat_rung")] = "random WildChat\n(ordinary traffic)"

ARM_ORDER = [
    "arm1_ctx_e1",
    "arm2_ctx_native",
    "arm3_identity_bias",
    "arm4_ridge_ctx",
    "arm5_mlp_ctx",
    "arm6_map_proj_e1",
    "arm7_map_ridge_pred",
    "arm8_map_ridge_true",
    "arm9_pretrain_ft",
    "arm10_stacked",
    "arm11_oracle_proj",
    "arm12_oracle_reg",
    "arm13_shuffled_map",
    "arm14_shuffled_pt",
    "arm15_text_only",
    "arm16_surface_feat",
]
ARM_LABEL = {
    "arm1_ctx_e1": "PV proj. on context (paper method)",
    "arm2_ctx_native": "context-native direction proj.",
    "arm3_identity_bias": "identity+bias -> PV proj.",
    "arm4_ridge_ctx": "direct ridge (context)",
    "arm5_mlp_ctx": "direct MLP (context)",
    "arm6_map_proj_e1": "map -> PV proj.",
    "arm7_map_ridge_pred": "map -> ridge (pred. answers)",
    "arm8_map_ridge_true": "map -> ridge (real answers)",
    "arm9_pretrain_ft": "map pretrain -> fine-tune",
    "arm10_stacked": "stacked: map-proj + direct ridge",
    "arm11_oracle_proj": "oracle: PV proj. on TRUE answer",
    "arm12_oracle_reg": "oracle: ridge on TRUE answer",
    "arm13_shuffled_map": "control: shuffled map",
    "arm14_shuffled_pt": "control: shuffled pretrain",
    "arm15_text_only": "baseline: text-embedding ridge",
    "arm16_surface_feat": "baseline: surface features",
}
# Arm families drive colour. ONE colour = ONE meaning across every re-cut.
ARM_FAMILY = {
    "arm1_ctx_e1": "context",
    "arm2_ctx_native": "context",
    "arm3_identity_bias": "context",
    "arm4_ridge_ctx": "context",
    "arm5_mlp_ctx": "context",
    "arm6_map_proj_e1": "map",
    "arm7_map_ridge_pred": "map",
    "arm8_map_ridge_true": "map",
    "arm9_pretrain_ft": "map",
    "arm10_stacked": "map",
    "arm11_oracle_proj": "oracle",
    "arm12_oracle_reg": "oracle",
    "arm13_shuffled_map": "control",
    "arm14_shuffled_pt": "control",
    "arm15_text_only": "baseline",
    "arm16_surface_feat": "baseline",
}
# Wong colorblind-safe hexes, fixed per family (paper_palette order).
FAMILY_COLOR = {
    "context": "#0072B2",  # blue   - read the context directly
    "map": "#D55E00",  # orange - go through the context->answer map
    "oracle": "#009E73",  # green  - upper bounds (need the true answer)
    "control": "#999999",  # grey   - by-construction nulls
    "baseline": "#CC79A7",  # pink   - text/surface baselines
}
FAMILY_LABEL = {
    "context": "context-side method",
    "map": "map-side method",
    "oracle": "oracle (needs true answer)",
    "control": "null control",
    "baseline": "text/surface baseline",
}


def arm_color(arm: str) -> str:
    """Colour for an arm, keyed on its family so one colour = one meaning."""
    return FAMILY_COLOR[ARM_FAMILY[arm]]


def match(row: dict, **kw) -> bool:
    """True when every kw equals the row's value (string-compared)."""
    return all(str(row.get(k)) == str(v) for k, v in kw.items())


# --------------------------------------------------------------- loaders ----


def load_main(behavior: str) -> dict:
    """The main-lane all_arms_spearman.json (train arm_rows + OOD transfer)."""
    with open(WT / behavior / "arm_results/all_arms_spearman.json") as f:
        return json.load(f)


def load_wide_ood(behavior: str) -> list[dict]:
    """Flattened wide_ood rows (OOD + train rungs, arms 7/8/12 added)."""
    rows: list[dict] = []
    with open(WT / f"wide_ood/{behavior}_transfer.jsonl") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            rows.extend(obj["rows"] if isinstance(obj, dict) and "rows" in obj else [obj])
    return rows


def load_wide_wcrung(behavior: str) -> dict:
    """The wide WildChat-rung artifact (transfer_rows + per_layer_rows)."""
    with open(WT / f"wide/wildchat_rung/{behavior}/all_arms_spearman.json") as f:
        return json.load(f)


def load_cells(behavior: str) -> list[dict]:
    """Per-cell records: split_half ceiling, permutation null, preds_npz."""
    with open(WT / behavior / "arm_results/percell/cells.jsonl") as f:
        return [json.loads(line) for line in f]


def load_wcrung_preds(behavior: str, variant: str = "context_end") -> tuple[dict, np.ndarray, list]:
    """Per-context WildChat-rung predictions.

    Returns ``(scores_by_arm, dv, context_ids)`` on a SHARED context ordering.
    Fail-loud: every arm must cover the identical context set with an
    identical judged DV, so any cross-arm comparison built on this is
    matched-target by construction.
    """
    path = WT / f"wide/wildchat_rung/{behavior}/preds/wcrung_preds.{variant}.jsonl"
    per_arm: dict[str, dict[str, float]] = {}
    dv_by_ctx: dict[str, float] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            cid = r["context_id"]
            per_arm.setdefault(r["arm"], {})[cid] = float(r["score"])
            dv = float(r["dv"])
            if cid in dv_by_ctx and not np.isclose(dv_by_ctx[cid], dv):
                raise ValueError(f"{behavior}/{variant}: DV disagrees across arms at {cid}")
            dv_by_ctx[cid] = dv

    ctx = sorted(dv_by_ctx)
    for arm, m in per_arm.items():
        if set(m) != set(ctx):
            raise ValueError(f"{behavior}/{variant}: arm {arm} covers a different context set")
    scores = {a: np.array([m[c] for c in ctx], dtype=float) for a, m in per_arm.items()}
    dv = np.array([dv_by_ctx[c] for c in ctx], dtype=float)
    return scores, dv, ctx


def load_train_cell_preds(
    behavior: str, seed: int, draw: int, **unit_kw
) -> tuple[dict, np.ndarray]:
    """Per-context train-rung predictions for ONE cell of the operating slice.

    Returns ``(scores_by_arm, dv)``. The cell is located by its ``unit_key``.
    """
    want = dict(unit_kw)
    want.update(seed=seed, draw=draw, eval_rung="train")
    for cell in load_cells(behavior):
        uk = json.loads(cell["unit_key"])
        if all(str(uk.get(k)) == str(v) for k, v in want.items()):
            npz_path = WT / behavior / "arm_results/percell/preds" / cell["preds_npz"]
            with np.load(npz_path, allow_pickle=True) as z:
                dv = np.asarray(z["dv"], dtype=float)
                scores = {
                    k[len("pred__") :]: np.asarray(z[k], dtype=float)
                    for k in z.files
                    if k.startswith("pred__")
                }
            return scores, dv
    raise KeyError(f"{behavior}: no train cell matching {want}")


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Spearman rho over the finite-in-both entries of x and y."""
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    from scipy.stats import spearmanr

    return float(spearmanr(x[ok], y[ok]).statistic)


def paired_delta_bootstrap(
    score_a: np.ndarray,
    score_b: np.ndarray,
    dv: np.ndarray,
    n_boot: int = 2000,
    seed: int = 0,
) -> dict:
    """Paired bootstrap of rho(a, dv) - rho(b, dv), resampling CONTEXTS.

    Both arms are scored against the SAME dv on the SAME contexts, and each
    bootstrap draw resamples one context index set applied to both arms — so
    the CI is a CI on the DIFFERENCE, not two independent per-arm CIs.
    """
    ok = np.isfinite(score_a) & np.isfinite(score_b) & np.isfinite(dv)
    a, b, d = score_a[ok], score_b[ok], dv[ok]
    n = a.size
    point = spearman(a, d) - spearman(b, d)
    rng = np.random.default_rng(seed)
    # Rank once; bootstrap Spearman == Pearson on the resampled ranks is NOT
    # equivalent (ranks change under resampling), so re-rank per draw via
    # scipy on the resampled slice.
    from scipy.stats import rankdata

    draws = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        da = rankdata(a[idx])
        db = rankdata(b[idx])
        dd = rankdata(d[idx])
        draws[i] = _pearson(da, dd) - _pearson(db, dd)
    lo, hi = np.nanpercentile(draws, [2.5, 97.5])
    return dict(
        delta=point,
        ci=[float(lo), float(hi)],
        n=int(n),
        n_boot=int(n_boot),
        boot_mean=float(np.nanmean(draws)),
        excludes_zero=bool(lo > 0 or hi < 0),
    )


def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    xm = x - x.mean()
    ym = y - y.mean()
    den = np.sqrt((xm * xm).sum() * (ym * ym).sum())
    return float((xm * ym).sum() / den) if den > 0 else float("nan")


def replicate_delta(rows: list[dict], arm_a: str, arm_b: str, **filt) -> dict:
    """Paired delta across matched (seed, draw) replicates.

    For every replicate present for BOTH arms under ``filt``, take
    rho(arm_a) - rho(arm_b); report mean, SD, and a normal-approx 95% CI on
    the mean. Replicates missing either arm are dropped (and counted).
    """

    def key(r):
        return (r.get("seed"), r.get("draw"))

    a = {key(r): r["rho_frozen"] for r in rows if r["arm"] == arm_a and match(r, **filt)}
    b = {key(r): r["rho_frozen"] for r in rows if r["arm"] == arm_b and match(r, **filt)}
    shared = sorted(set(a) & set(b), key=lambda t: (str(t[0]), str(t[1])))
    vals = np.array(
        [a[k] - b[k] for k in shared if a[k] is not None and b[k] is not None], dtype=float
    )
    if vals.size == 0:
        return dict(delta=None, ci=None, n_rep=0, sd=None, unmatched=len(set(a) ^ set(b)))
    m = float(vals.mean())
    sd = float(vals.std(ddof=1)) if vals.size > 1 else 0.0
    half = 1.96 * sd / np.sqrt(vals.size) if vals.size > 1 else 0.0
    return dict(
        delta=m,
        ci=[m - half, m + half],
        n_rep=int(vals.size),
        sd=sd,
        unmatched=len(set(a) ^ set(b)),
        excludes_zero=bool((m - half) > 0 or (m + half) < 0),
    )


def agg_rho(rows: list[dict], arm: str, **filt) -> dict:
    """Mean +/- SD of rho_frozen over replicates, plus the pooled ci_frozen."""
    vals = [
        r["rho_frozen"]
        for r in rows
        if r["arm"] == arm and match(r, **filt) and r.get("rho_frozen") is not None
    ]
    if not vals:
        return dict(mean=None, sd=None, n=0, ci=None)
    cis = [
        r["ci_frozen"]
        for r in rows
        if r["arm"] == arm and match(r, **filt) and r.get("ci_frozen") is not None
    ]
    ci = [float(np.mean([c[0] for c in cis])), float(np.mean([c[1] for c in cis]))] if cis else None
    return dict(
        mean=float(np.mean(vals)),
        sd=float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
        n=len(vals),
        ci=ci,
    )


def ceiling_sb(behavior: str) -> dict:
    """Split-half reliability ceiling (Spearman-Brown) over train-rung cells.

    Returns ``{mean, min, max, n}``; ``n == 0`` when the behavior's DV carries
    no per-rollout scores (hallucination's fabrication-rate construct).
    """
    vals = [
        c["split_half"]["ceiling_sb"]
        for c in load_cells(behavior)
        if c.get("split_half") and c["split_half"].get("ceiling_sb") is not None
    ]
    if not vals:
        return dict(mean=None, min=None, max=None, n=0)
    return dict(
        mean=float(np.mean(vals)),
        min=float(np.min(vals)),
        max=float(np.max(vals)),
        n=len(vals),
    )


def nonneg_err(values, lo, hi):
    """Clamped asymmetric error offsets for matplotlib (never negative)."""
    v = np.asarray(values, dtype=float)
    return np.vstack(
        [
            np.maximum(0.0, v - np.asarray(lo, dtype=float)),
            np.maximum(0.0, np.asarray(hi, dtype=float) - v),
        ]
    )
