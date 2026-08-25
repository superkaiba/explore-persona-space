"""Issue #2564 PE analysis — minimal-pair delta reads against the frozen maps.

Unit 4 of the pre-split build (plan §6 IN FULL + §6.5 deliverable). VM CPU,
fully vectorized (no per-pair Python loops — pair axes are single tensor ops;
bootstrap draws reduce to (B, n_carriers) multiplicity einsums; derangement
nulls gather from precomputed cosine grids).

Inputs (local ``--in-root`` mirror first, else staged from the HF data repo):

- ``analysis_tensors/va2564/va2564_<cell>.pt``   (U2 stores: fp16 va_span +
  va_tail_incl, index rows with draw + boundary fields, ``empty_rows``)
- ``analysis_tensors/vc2564/vc2564_bank.pt``     (fp32 context-end states)
- ``analysis_tensors/embeddings_qwen3_8b/means_anchors.npz`` (U3)
- ``manifests/bank2564_manifest.json``           (frozen bank: 984 contexts /
  2,778 pairs incl. ``changed_tokens``)
- frozen ridge payloads ``issue779_monitoring/n1m_readout/weights/L19/ridge.pt``
  + ``issue1738_multiturn/analysis_tensors/weights/L19/context_ridge.pt``
  (applied via ``issue779_ffc_n1m_fits.apply_map`` — NOT refit)
- ``eval_results/issue_2564/manipulation_check.json`` (U3 judge gate)

Outputs: ``minpair_delta.json`` (every §6 read + CI + null + ceiling per
axis × arm + the Artifact metadata contract), ``perpair.jsonl`` (per-pair
rows), and prediction tensors → HF
``issue2564_minpair/analysis_tensors/predictions/*.pt``. NO figures (unit 5).

Arms: ``arm_779ce`` (primary frozen map), ``arm_1738ce`` (secondary),
``arm_iddelta`` (identity baseline — the learned bias cancels EXACTLY in the
delta framing, asserted numerically on a random subset via
``mapping_baselines.identity_bias_predict``).

Seeds (plan §9/§11): carrier-clustered bootstrap B=10,000 seed 2215;
derangement null B=10,000 seed 21620; 20 split-half splits seed 2564.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import sys
import time
import warnings
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + HF token BEFORE torch import (code-style.md)

import numpy as np  # noqa: E402
import torch  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue779_ffc_n1m_fits as N1M  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import (  # noqa: E402
    identity_bias_predict,
    knn_retrieval,
)
from explore_persona_space.experiments.issue2564 import bank2564 as BK  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logger = logging.getLogger("issue2564_analysis")

ISSUE = 2564
HF_DATA_REPO = os.environ.get("EPM_2564_DATA_WRITE_REPO", "superkaiba1/explore-persona-space-data")
HF_PREFIX_FULL = "issue2564_minpair"
HF_PREFIX_SMOKE = "issue2564_minpair/smoke"
RIDGE_779_PATH = "issue779_monitoring/n1m_readout/weights/L19/ridge.pt"
RIDGE_1738_PATH = "issue1738_multiturn/analysis_tensors/weights/L19/context_ridge.pt"

LAYERS = (14, 19, 26)
PRIMARY_LAYER = 19
BOOT_SEED = 2215
NULL_SEED = 21620
SPLIT_SEED = 2564
B_BOOT_DEFAULT = 10_000
B_NULL_DEFAULT = 10_000
N_SPLITS_DEFAULT = 20
FIRE_THRESHOLDS = (50, 70, 90)  # 70 = headline (plan §6 fire rule)

ARMS = ("arm_779ce", "arm_1738ce", "arm_iddelta")
QUERY_AXES = ("query_content", "query_form")
AXES_ALL: tuple[str, ...] = tuple(BK.INSTRUCTION_AXES) + QUERY_AXES

PRIMARY_CLASS_BY_AXIS = {
    **{a: "swap" for a in BK.INSTRUCTION_AXES},
    "query_content": "query_content",
    "query_form": "query_form",
}
PARA_CLASS_BY_AXIS = {
    **{a: "instruction_paraphrase" for a in BK.INSTRUCTION_AXES},
    "query_content": "query_paraphrase",
    "query_form": "query_paraphrase",
}

ORIENTATION_CONVENTIONS = {
    "install": "E -> value (a = bare-E query context, b = value context)",
    "swap": "value_i -> value_j by plan-listed value-index order (i < j)",
    "famswap": "para(value_i) -> para(value_j), same value-index order",
    "instruction_paraphrase": "value -> its paraphrase",
    "query_content": "carrier_i -> carrier_j by carrier-index order (i < j)",
    "query_form": "form_i -> form_j by form-index order E < imp < stmt",
    "query_paraphrase": "E question -> reworded question",
}

DRAW_TO_PAIR_AGGREGATION = (
    "Delta = difference of the two contexts' K=10-draw mean answer vectors "
    "(tail-inclusive L19 primary; empty-completion draws excluded from the "
    "mean, per-context valid-draw counts recorded per pair)"
)
DYADIC_BOOTSTRAP_CONVENTION = (
    "query_content pairs are carrier DYADS: the bootstrap resamples the 12 "
    "carrier VERTICES with replacement and weights each edge by the product "
    "of its sampled endpoint multiplicities (edges with an unsampled endpoint "
    "get weight 0); single-carrier pairs weight by their carrier multiplicity"
)


# ── small numeric helpers (unit-tested) ────────────────────────────────


def rowwise_cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Row-wise cosine, float64; zero-norm rows -> NaN (counted by callers)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a, axis=-1)
    nb = np.linalg.norm(b, axis=-1)
    den = na * nb
    with np.errstate(invalid="ignore", divide="ignore"):
        out = (a * b).sum(-1) / den
    return np.where(den > 0, out, np.nan)


def spearman_brown(r_half: np.ndarray | float) -> np.ndarray | float:
    """Split-half -> full-K-mean reliability step-up: r_K = 2 r / (1 + r).

    Guard: r <= -1 (undefined) -> NaN. Vectorized.
    """
    r = np.asarray(r_half, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = np.where(r > -1.0, 2.0 * r / (1.0 + r), np.nan)
    return float(out) if np.isscalar(r_half) or out.ndim == 0 else out


def suppression_verdict(ceiling_pt: float, ci_lo: float, ci_hi: float) -> bool:
    """Plan §6 convention 1: suppress the ceiling-normalized read where the
    ceiling is nonpositive OR its bootstrap CI includes zero (lo <= 0 <= hi).
    A non-finite ceiling is suppressed too."""
    if not np.isfinite(ceiling_pt) or ceiling_pt <= 0.0:
        return True
    if np.isfinite(ci_lo) and np.isfinite(ci_hi) and ci_lo <= 0.0 <= ci_hi:
        return True
    return False


def through_origin_slope(pred_norm: np.ndarray, obs_norm: np.ndarray) -> float:
    """Through-origin OLS slope of ||pred|| on ||obs||: sum(p*o)/sum(o^2)."""
    p = np.asarray(pred_norm, dtype=np.float64)
    o = np.asarray(obs_norm, dtype=np.float64)
    den = float((o * o).sum())
    return float((p * o).sum() / den) if den > 0 else float("nan")


def ols_intercept_slope(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    """Plain OLS y ~ 1 + x; returns (intercept, slope). Degenerate x -> slope 0."""
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    xm, ym = x.mean(), y.mean()
    den = float(((x - xm) ** 2).sum())
    if den == 0.0:
        return float(ym), 0.0
    slope = float(((x - xm) * (y - ym)).sum() / den)
    return float(ym - slope * xm), slope


def deranged_perms(n: int, b: int, rng: np.random.Generator) -> np.ndarray:
    """(b, n) permutations with NO fixed point (issue2215_analysis idiom)."""
    assert n >= 2, n
    out = np.empty((b, n), dtype=np.int64)
    filled = 0
    while filled < b:
        batch = max(2 * (b - filled) + 8, 16)
        perms = np.argsort(rng.random((batch, n)), axis=1)
        good = perms[(perms != np.arange(n)).all(axis=1)]
        take = good[: b - filled]
        out[filled : filled + len(take)] = take
        filled += len(take)
    return out


def _pct(a: np.ndarray, q: float) -> float:
    a = np.asarray(a, dtype=np.float64)
    if not np.isfinite(a).any():
        return float("nan")
    return float(np.nanpercentile(a, q))


def _ci(draws: np.ndarray) -> list[float]:
    return [_pct(draws, 2.5), _pct(draws, 97.5)]


def carrier_multiplicities(idx_draws: np.ndarray, n_carriers: int) -> np.ndarray:
    """(B, n_carriers) resample multiplicities from (B, n_carriers) index draws."""
    return ((idx_draws[:, :, None] == np.arange(n_carriers)[None, None, :]).sum(axis=1)).astype(
        np.float64
    )


def boot_pair_sums(
    vals: np.ndarray,
    ca: np.ndarray,
    cb: np.ndarray,
    dyad: np.ndarray,
    mult: np.ndarray,
) -> np.ndarray:
    """Per-draw weighted sums over pairs: w = mult[:,ca] (single-carrier) or
    mult[:,ca]*mult[:,cb] (dyads). Reduces to (B, n_car) contractions — no
    (B, n_pairs) weight matrix is ever materialized. NaN vals contribute 0
    with 0 weight ONLY if the caller pre-drops them; here NaNs propagate."""
    vals = np.asarray(vals, dtype=np.float64)
    n_car = mult.shape[1]
    out = np.zeros(mult.shape[0], dtype=np.float64)
    single = ~dyad
    if single.any():
        per_c = np.zeros(n_car, dtype=np.float64)
        np.add.at(per_c, ca[single], vals[single])
        out += mult @ per_c
    if dyad.any():
        m = np.zeros((n_car, n_car), dtype=np.float64)
        np.add.at(m, (ca[dyad], cb[dyad]), vals[dyad])
        out += np.einsum("bi,ij,bj->b", mult, m, mult)
    return out


def boot_weighted_mean(
    vals: np.ndarray, ca: np.ndarray, cb: np.ndarray, dyad: np.ndarray, mult: np.ndarray
) -> np.ndarray:
    """Per-draw carrier-clustered weighted means; finite vals only (NaN rows
    are dropped from numerator AND denominator)."""
    vals = np.asarray(vals, dtype=np.float64)
    ok = np.isfinite(vals)
    num = boot_pair_sums(np.where(ok, vals, 0.0), ca, cb, dyad, mult)
    den = boot_pair_sums(ok.astype(np.float64), ca, cb, dyad, mult)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / den
    return np.where(den > 0, out, np.nan)


def dyad_pair_weights(mult: np.ndarray, ca: np.ndarray, cb: np.ndarray) -> np.ndarray:
    """(B, n_pairs) dyadic edge weights = product of endpoint multiplicities.
    Exposed for the unit test pinning the dyadic/vertex bootstrap convention."""
    return mult[:, ca] * mult[:, cb]


# ── identity-cancellation assert (plan §6 mapping-baselines pair) ──────


def identity_cancellation_check(
    vc: np.ndarray,
    a_idx: np.ndarray,
    b_idx: np.ndarray,
    rng: np.random.Generator,
    n_check: int = 32,
) -> dict:
    """Numeric fitting-free assert that ``arm_iddelta`` IS the identity+
    learned-bias baseline: (x_A + b) - (x_B + b) == x_A - x_B, with b produced
    by the REAL ``identity_bias_predict`` helper on a synthetic train set."""
    n = len(a_idx)
    sel = rng.choice(n, size=min(n_check, n), replace=False)
    d = vc.shape[1]
    x_train = rng.standard_normal((4, d))
    bias = rng.standard_normal(d) * 3.0
    y_train = x_train + bias
    xa = vc[a_idx[sel]]
    xb = vc[b_idx[sel]]
    pred_a = identity_bias_predict(x_train, y_train, xa)
    pred_b = identity_bias_predict(x_train, y_train, xb)
    err = float(np.abs((pred_a - pred_b) - (xa - xb)).max())
    scale = float(max(np.abs(xa - xb).max(), 1.0))
    tol = 1e-8 * scale
    assert err <= tol, f"identity-bias cancellation violated: err={err} > tol={tol}"
    return {
        "n_pairs_checked": int(len(sel)),
        "max_abs_err": err,
        "tol": tol,
        "statement": "identity_bias_predict(x_A) - identity_bias_predict(x_B) == x_A - x_B",
    }


# ── config ─────────────────────────────────────────────────────────────


@dataclass
class CfgPE:
    in_root: Path | None
    out_dir: Path
    stage_dir: Path
    manip_check: Path
    ridge_779: Path | None
    ridge_1738: Path | None
    smoke: bool
    upload: str
    b_boot: int
    b_null: int
    n_splits: int
    hf_prefix: str
    seed_base: int = BOOT_SEED

    @property
    def pred_dir(self) -> Path:
        return self.out_dir / "predictions"


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0].replace("%", "%%"))
    ap.add_argument("--in-root", type=Path, default=None, help="local pod out-root mirror")
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--stage-dir", type=Path, default=None)
    ap.add_argument("--manip-check", type=Path, default=None)
    ap.add_argument("--ridge-779", type=Path, default=None, help="local ridge payload override")
    ap.add_argument("--ridge-1738", type=Path, default=None)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--upload", choices=("hf", "none"), default=None)
    ap.add_argument("--b-boot", type=int, default=None)
    ap.add_argument("--b-null", type=int, default=None)
    ap.add_argument("--n-splits", type=int, default=N_SPLITS_DEFAULT)
    ap.add_argument("--import-check", action="store_true")
    return ap


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    return build_parser().parse_args(argv)


def build_config(args: argparse.Namespace) -> CfgPE:
    """Resolve the CLI namespace (smoke rebinds out-dir/prefix/B, never inputs)."""
    smoke = bool(args.smoke)
    repo_root = Path(__file__).resolve().parents[1]
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    elif smoke:
        out_dir = Path("/tmp/issue-2564-smoke/eval_results/issue_2564")
    else:
        out_dir = repo_root / "eval_results" / "issue_2564"
    default_stage = (
        repo_root / "data" / "issue_2564" / "hf_dl" / ("pe_stage_smoke" if smoke else "pe_stage")
    )
    manip = (
        Path(args.manip_check)
        if args.manip_check is not None
        else repo_root / "eval_results" / "issue_2564" / "manipulation_check.json"
    )
    return CfgPE(
        in_root=Path(args.in_root) if args.in_root else None,
        out_dir=out_dir,
        stage_dir=Path(args.stage_dir) if args.stage_dir else default_stage,
        manip_check=manip,
        ridge_779=Path(args.ridge_779) if args.ridge_779 else None,
        ridge_1738=Path(args.ridge_1738) if args.ridge_1738 else None,
        smoke=smoke,
        upload=args.upload if args.upload is not None else ("none" if smoke else "hf"),
        b_boot=int(args.b_boot) if args.b_boot is not None else (100 if smoke else B_BOOT_DEFAULT),
        b_null=int(args.b_null) if args.b_null is not None else (100 if smoke else B_NULL_DEFAULT),
        n_splits=int(args.n_splits),
        hf_prefix=HF_PREFIX_SMOKE if smoke else HF_PREFIX_FULL,
    )


# ── input resolution (local-first, else HF stage; fail loud) ───────────


def resolve_input(cfg: CfgPE, rel: str) -> Path:
    """``<in_root>/<rel>`` when present, else stage ``<hf_prefix>/<rel>`` from
    the HF data repo (retried, atomic, idempotent via hub.stage_hub_file)."""
    if cfg.in_root is not None:
        cand = cfg.in_root / rel
        if cand.exists():
            return cand
    target = cfg.stage_dir / rel
    if target.exists():
        return target
    from explore_persona_space.orchestrate.hub import stage_hub_file

    logger.info("[pe] staging %s/%s from %s", cfg.hf_prefix, rel, HF_DATA_REPO)
    return Path(stage_hub_file(HF_DATA_REPO, f"{cfg.hf_prefix}/{rel}", target))


def resolve_ridge(cfg: CfgPE, local: Path | None, repo_path: str) -> Path:
    """Frozen ridge payloads live at issue-owned HF prefixes (NOT under the
    2564 prefix); a local override serves tests / pre-staged copies."""
    if local is not None:
        assert local.exists(), f"ridge payload override missing: {local}"
        return local
    target = cfg.stage_dir / "frozen_maps" / repo_path
    if target.exists():
        return target
    from explore_persona_space.orchestrate.hub import stage_hub_file

    logger.info("[pe] staging frozen map %s", repo_path)
    return Path(stage_hub_file(HF_DATA_REPO, repo_path, target))


def load_ridge_payload(path: Path, expect_d: int, arm: str) -> dict:
    """Load + semantically validate a frozen ridge payload (kind/shape contract)."""
    payload = torch.load(path, map_location="cpu", weights_only=False)
    assert payload.get("kind") == "ridge", (arm, payload.get("kind"))
    w = payload["W"]
    assert tuple(w.shape) == (expect_d, expect_d), (arm, tuple(w.shape), expect_d)
    for k in ("xmu", "xsd", "ymu"):
        assert tuple(payload[k].shape)[-1] == expect_d, (arm, k, tuple(payload[k].shape))
    return payload


# ── store loading ──────────────────────────────────────────────────────


@dataclass
class Stores:
    ctx_ids: list[str]
    row_of: dict[str, int]
    cells: list[str]
    carriers: list[str]  # present carriers, CARRIER_IDS order
    va_tail_mean: dict[int, np.ndarray]  # layer -> (n_ctx, d) float64
    va_span_mean: dict[int, np.ndarray]
    tail_draws: np.ndarray  # (n_ctx, k_max, d) float32, PRIMARY layer
    draw_valid: np.ndarray  # (n_ctx, k_max) bool
    n_valid: np.ndarray  # (n_ctx,)
    ans_len_mean: np.ndarray  # (n_ctx,) mean completion tokens over valid draws
    vc: dict[int, np.ndarray]  # layer -> (n_ctx, d) float64
    emb_mean: np.ndarray  # (n_ctx, e) float64
    d: int
    input_files: dict = field(default_factory=dict)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_bank_manifest(path: Path) -> dict:
    bank = json.loads(Path(path).read_text())
    assert bank["n_contexts"] == BK.N_CONTEXTS and bank["n_pairs"] == BK.N_PAIRS, (
        bank["n_contexts"],
        bank["n_pairs"],
    )
    assert len(bank["pairs"]) == BK.N_PAIRS
    for p in bank["pairs"]:
        assert "changed_tokens" in p, (p["pair_id"], "bank manifest missing changed_tokens")
    return bank


def load_stores(cfg: CfgPE, bank: dict) -> Stores:
    """Assemble per-context mean matrices + per-draw tail tensor (L19) from the
    U2/U3 stores. Empty-completion rows (ZERO vectors in the fp16 store) are
    excluded from every mean; a context with zero valid draws fails loud."""
    files: dict[str, dict] = {}
    vc_path = resolve_input(cfg, "analysis_tensors/vc2564/vc2564_bank.pt")
    vc_store = torch.load(vc_path, map_location="cpu", weights_only=False)
    layers = [int(x) for x in vc_store["layers"]]
    assert tuple(layers) == LAYERS, layers
    ctx_ids = list(vc_store["context_ids"])
    row_of = {cid: i for i, cid in enumerate(ctx_ids)}
    vc_t = vc_store["vc"].to(torch.float64).numpy()
    d = vc_t.shape[2]
    files["vc2564_bank.pt"] = {"path": str(vc_path), "bytes": vc_path.stat().st_size}

    contexts = bank["contexts"]
    missing_bank = [cid for cid in ctx_ids if cid not in contexts]
    assert not missing_bank, f"vc contexts absent from bank manifest: {missing_bank[:5]}"
    cells = sorted({contexts[cid]["cell"] for cid in ctx_ids})
    carriers = [c for c in BK.CARRIER_IDS if c in {contexts[cid]["carrier"] for cid in ctx_ids}]

    n_ctx = len(ctx_ids)
    li = {layer: k for k, layer in enumerate(layers)}
    vc = {layer: np.ascontiguousarray(vc_t[:, li[layer], :]) for layer in LAYERS}

    # per-cell va stores — vectorized accumulation (no per-row Python loop)
    tail_sum = {layer: np.zeros((n_ctx, d), dtype=np.float64) for layer in LAYERS}
    span_sum = {layer: np.zeros((n_ctx, d), dtype=np.float64) for layer in LAYERS}
    len_sum = np.zeros(n_ctx, dtype=np.float64)
    cnt = np.zeros(n_ctx, dtype=np.int64)
    prim_chunks: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []  # (ctx, draw, rows_f32)
    k_max = 0
    for cell in cells:
        rel = f"analysis_tensors/va2564/va2564_{cell}.pt"
        p = resolve_input(cfg, rel)
        store = torch.load(p, map_location="cpu", weights_only=False)
        files[f"va2564_{cell}.pt"] = {"path": str(p), "bytes": p.stat().st_size}
        assert [int(x) for x in store["layers"]] == list(LAYERS), store["layers"]
        idx_rows = store["index"]
        tail = store["va_tail_incl"].to(torch.float64).numpy()
        span = store["va_span"].to(torch.float64).numpy()
        assert tail.shape == (len(idx_rows), len(LAYERS), d), (tail.shape, len(idx_rows), d)
        n_rows = len(idx_rows)
        ctx_idx = np.array([row_of.get(rec["context_id"], -1) for rec in idx_rows], dtype=np.int64)
        n_comp = np.array([int(rec["n_completion_tokens"]) for rec in idx_rows], dtype=np.int64)
        draw = np.array([int(rec["draw"]) for rec in idx_rows], dtype=np.int64)
        empty_mask = np.zeros(n_rows, dtype=bool)
        empty_ids = np.array(sorted(int(i) for i in store.get("empty_rows", [])), dtype=np.int64)
        if empty_ids.size:
            empty_mask[empty_ids] = True
        valid = (ctx_idx >= 0) & (n_comp > 0) & ~empty_mask
        for layer in LAYERS:
            np.add.at(tail_sum[layer], ctx_idx[valid], tail[valid, li[layer], :])
            np.add.at(span_sum[layer], ctx_idx[valid], span[valid, li[layer], :])
        np.add.at(len_sum, ctx_idx[valid], n_comp[valid].astype(np.float64))
        np.add.at(cnt, ctx_idx[valid], 1)
        if valid.any():
            k_max = max(k_max, int(draw[valid].max()) + 1)
            prim_chunks.append(
                (ctx_idx[valid], draw[valid], tail[valid, li[PRIMARY_LAYER], :].astype(np.float32))
            )

    zero = [ctx_ids[i] for i in range(n_ctx) if cnt[i] == 0]
    if zero:
        raise RuntimeError(f"contexts with ZERO valid (non-empty) draws: {zero[:10]}")
    va_tail_mean = {layer: tail_sum[layer] / cnt[:, None] for layer in LAYERS}
    va_span_mean = {layer: span_sum[layer] / cnt[:, None] for layer in LAYERS}
    ans_len_mean = len_sum / cnt

    tail_draws = np.zeros((n_ctx, k_max, d), dtype=np.float32)
    draw_valid = np.zeros((n_ctx, k_max), dtype=bool)
    for ctx_v, draw_v, rows_v in prim_chunks:
        key = ctx_v * k_max + draw_v
        assert len(np.unique(key)) == len(key), "duplicate (context, draw) slot within a va store"
        assert not draw_valid[ctx_v, draw_v].any(), "duplicate (context, draw) slot in va stores"
        tail_draws[ctx_v, draw_v] = rows_v
        draw_valid[ctx_v, draw_v] = True

    emb_path = resolve_input(cfg, "analysis_tensors/embeddings_qwen3_8b/means_anchors.npz")
    with np.load(emb_path, allow_pickle=False) as z:
        emb_ids = [str(x) for x in z["context_ids"].tolist()]
        emb = z["emb_mean"].astype(np.float64)
    files["means_anchors.npz"] = {"path": str(emb_path), "bytes": emb_path.stat().st_size}
    emb_of = {cid: i for i, cid in enumerate(emb_ids)}
    missing_emb = [cid for cid in ctx_ids if cid not in emb_of]
    assert not missing_emb, f"contexts missing from embedding means: {missing_emb[:5]}"
    emb_mean = emb[[emb_of[cid] for cid in ctx_ids]]

    return Stores(
        ctx_ids=ctx_ids,
        row_of=row_of,
        cells=cells,
        carriers=carriers,
        va_tail_mean=va_tail_mean,
        va_span_mean=va_span_mean,
        tail_draws=tail_draws,
        draw_valid=draw_valid,
        n_valid=cnt,
        ans_len_mean=ans_len_mean,
        vc=vc,
        emb_mean=emb_mean,
        d=d,
        input_files=files,
    )


# ── pair table ─────────────────────────────────────────────────────────


@dataclass
class PairArrays:
    ids: list[str]
    cls: list[str]
    axis: list[str]  # cell for instruction classes; pair_class for query classes
    value_a: list[str]
    value_b: list[str]
    carrier_str: list[str]
    a: np.ndarray  # ctx row idx
    b: np.ndarray
    ca: np.ndarray  # carrier idx (present-carrier order)
    cb: np.ndarray  # == ca for single-carrier pairs
    dyad: np.ndarray  # bool
    changed: np.ndarray  # int
    orientation: list[str]
    n: int


def build_pair_arrays(bank: dict, st: Stores, smoke: bool) -> PairArrays:
    """Restrict the frozen bank's pairs to contexts present in the stores;
    production (non-smoke) asserts FULL 2,778/984 coverage."""
    car_of = {c: i for i, c in enumerate(st.carriers)}
    keep: list[dict] = []
    for p in bank["pairs"]:
        if p["a"] in st.row_of and p["b"] in st.row_of:
            keep.append(p)
    if not keep:
        raise RuntimeError("empty pair selection: no bank pair has both contexts in the stores")
    if not smoke:
        assert len(st.ctx_ids) == BK.N_CONTEXTS, (len(st.ctx_ids), BK.N_CONTEXTS)
        assert len(keep) == BK.N_PAIRS, (len(keep), BK.N_PAIRS)

    ids, cls, axis, va_, vb_, cstr, orient = [], [], [], [], [], [], []
    a_i, b_i, ca_i, cb_i, dy, chg = [], [], [], [], [], []
    contexts = bank["contexts"]
    for p in keep:
        pc = p["pair_class"]
        ids.append(p["pair_id"])
        cls.append(pc)
        axis.append(p["cell"] if p["cell"] != "query" else pc)
        va_.append(p["value_a"])
        vb_.append(p["value_b"])
        cstr.append(p["carrier"])
        a_i.append(st.row_of[p["a"]])
        b_i.append(st.row_of[p["b"]])
        if pc == "query_content":
            ca_i.append(car_of[p["carrier_a"]])
            cb_i.append(car_of[p["carrier_b"]])
            dy.append(True)
            orient.append(f"{p['carrier_a']}->{p['carrier_b']}")
        else:
            c = car_of[contexts[p["a"]]["carrier"]]
            ca_i.append(c)
            cb_i.append(c)
            dy.append(False)
            orient.append(f"{p['value_a']}->{p['value_b']}")
        chg.append(int(p["changed_tokens"]))
    return PairArrays(
        ids=ids,
        cls=cls,
        axis=axis,
        value_a=va_,
        value_b=vb_,
        carrier_str=cstr,
        a=np.array(a_i, dtype=np.int64),
        b=np.array(b_i, dtype=np.int64),
        ca=np.array(ca_i, dtype=np.int64),
        cb=np.array(cb_i, dtype=np.int64),
        dyad=np.array(dy, dtype=bool),
        changed=np.array(chg, dtype=np.int64),
        orientation=orient,
        n=len(ids),
    )


# ── fire table (compliance layering) ───────────────────────────────────


def load_fire(manip_path: Path) -> dict:
    """Per-(axis, value_id) fire verdicts at each threshold + per-axis summary.

    Axes absent from the manipulation-check slice get NO entries — pairs on
    those axes are unfiltered (fired mask = all; recorded as fire: null)."""
    doc = json.loads(Path(manip_path).read_text())
    fired: dict[int, dict[tuple[str, str], bool]] = {t: {} for t in FIRE_THRESHOLDS}
    rows = {}
    for r in doc.get("value_rows", []):
        key = (r["axis"], r["value_id"])
        rows[key] = r
        fired[70][key] = r["verdict"] == "fired"
        for t in (50, 90):
            fired[t][key] = r["sensitivity"][str(t)] == "fired"
    axis_rows = {r["axis"]: r for r in doc.get("axis_rows", [])}
    return {"fired": fired, "value_rows": rows, "axis_rows": axis_rows, "meta": doc.get("meta", {})}


def pair_fired_mask(pa: PairArrays, fire: dict, threshold: int) -> tuple[np.ndarray, np.ndarray]:
    """(fired_a, fired_b) bool arrays; a value with NO fire row (query values,
    the bare-E side of install pairs, axes outside a smoke slice) counts FIRED
    (unfiltered — the gate only ever REMOVES checked-and-not-fired values)."""
    fmap = fire["fired"][threshold]

    def _ok(axis: str, vid: str) -> bool:
        return fmap.get((axis, vid), True)

    fa = np.array(
        [
            _ok(ax, va) if cl != "query_content" else True
            for ax, va, cl in zip(pa.axis, pa.value_a, pa.cls)
        ],
        dtype=bool,
    )
    fb = np.array(
        [
            _ok(ax, vb) if cl != "query_content" else True
            for ax, vb, cl in zip(pa.axis, pa.value_b, pa.cls)
        ],
        dtype=bool,
    )
    return fa, fb


# ── reliability (20 seeded 5/5 splits) ─────────────────────────────────


def split_half_stats(st: Stores, pa: PairArrays, n_splits: int) -> dict:
    """Per-pair split-half direction reliability + noise norm at L19 tail.

    Per split: the valid draws of every context are randomly partitioned into
    two halves (floor/ceil for odd counts); Delta_h = half-mean(A) - half-mean(B);
    r = cos(Delta_h1, Delta_h2); noise = ||Delta_h1 - Delta_h2|| / 2. Contexts
    with < 2 valid draws make their pairs NaN (counted). Loop is over the
    n_splits axis only — all pair math is vectorized."""
    draws = st.tail_draws.astype(np.float32)
    valid = st.draw_valid
    n_ctx, k_max, _ = draws.shape
    nv = valid.sum(axis=1)
    ctx_ok = nv >= 2
    pair_ok = ctx_ok[pa.a] & ctx_ok[pa.b]

    r_acc = np.zeros(pa.n, dtype=np.float64)
    noise_acc = np.zeros(pa.n, dtype=np.float64)
    n_used = 0
    for s in range(n_splits):
        rng = np.random.default_rng([SPLIT_SEED, s])
        scores = rng.random((n_ctx, k_max))
        scores[~valid] = np.inf
        order = np.argsort(scores, axis=1)
        ranks = np.empty_like(order)
        np.put_along_axis(ranks, order, np.broadcast_to(np.arange(k_max), (n_ctx, k_max)).copy(), 1)
        half1 = (ranks < (nv // 2)[:, None]) & valid
        half2 = valid & ~half1
        w1 = half1.astype(np.float32)
        w2 = half2.astype(np.float32)
        c1 = np.maximum(w1.sum(1), 1.0)
        c2 = np.maximum(w2.sum(1), 1.0)
        m1 = np.einsum("ck,ckd->cd", w1, draws) / c1[:, None]
        m2 = np.einsum("ck,ckd->cd", w2, draws) / c2[:, None]
        d1 = (m1[pa.a] - m1[pa.b]).astype(np.float64)
        d2 = (m2[pa.a] - m2[pa.b]).astype(np.float64)
        r_acc += np.where(pair_ok, np.nan_to_num(rowwise_cos(d1, d2), nan=0.0), 0.0)
        noise_acc += np.where(pair_ok, np.linalg.norm(d1 - d2, axis=1) / 2.0, 0.0)
        n_used += 1
    r_half = np.where(pair_ok, r_acc / max(n_used, 1), np.nan)
    noise = np.where(pair_ok, noise_acc / max(n_used, 1), np.nan)
    return {
        "r_half": r_half,
        "r_full": spearman_brown(r_half),
        "noise_norm": noise,
        "n_splits": n_splits,
        "n_pairs_insufficient_draws": int((~pair_ok).sum()),
        "half_sizes": "floor(n_valid/2) vs the rest (5/5 at the full K=10)",
    }


# ── axis views ─────────────────────────────────────────────────────────


@dataclass
class AxisView:
    axis: str
    primary_class: str
    para_class: str
    primary_idx: np.ndarray
    para_idx: np.ndarray
    install_idx: np.ndarray | None
    famswap_idx: np.ndarray | None
    primary_grid: np.ndarray | None  # (n_vp, n_car_present) pair idx; None for dyads
    famswap_grid: np.ndarray | None
    primary_vps: list[str]
    null_scheme: str


def _grid_for(pa: PairArrays, sel: np.ndarray, n_car: int) -> tuple[np.ndarray | None, list[str]]:
    """Complete (n_vp, n_car) grid of pair indices for a single-carrier class
    selection; asserts completeness (every vp present for every present carrier)."""
    if sel.size == 0:
        return None, []
    vps = sorted({f"{pa.value_a[i]}-{pa.value_b[i]}" for i in sel})
    vp_of = {v: k for k, v in enumerate(vps)}
    grid = np.full((len(vps), n_car), -1, dtype=np.int64)
    for i in sel:
        grid[vp_of[f"{pa.value_a[i]}-{pa.value_b[i]}"], pa.ca[i]] = i
    assert (grid >= 0).all(), f"incomplete (vp x carrier) grid for pairs {pa.axis[sel[0]]}"
    return grid, vps


def build_axis_views(pa: PairArrays, n_car: int) -> dict[str, AxisView]:
    idx_by = {}
    for i in range(pa.n):
        idx_by.setdefault((pa.axis[i], pa.cls[i]), []).append(i)
    qpara = np.array(
        sorted(i for i in range(pa.n) if pa.cls[i] == "query_paraphrase"), dtype=np.int64
    )
    views: dict[str, AxisView] = {}
    axes_present = sorted({a for a in pa.axis if a != "query_paraphrase"})
    for axis in axes_present:
        prim_cls = PRIMARY_CLASS_BY_AXIS[axis]
        para_cls = PARA_CLASS_BY_AXIS[axis]
        prim = np.array(sorted(idx_by.get((axis, prim_cls), [])), dtype=np.int64)
        if prim.size == 0:
            continue
        if axis in QUERY_AXES:
            para = qpara
            install = None
            fams = None
            fams_grid = None
            if axis == "query_content":
                prim_grid, vps = None, [f"{pa.carrier_str[i]}" for i in prim]
                scheme = (
                    "class-preserving edge derangement over the C(12,2) carrier "
                    "dyads (carrier preservation undefined for dyadic pairs)"
                )
            else:
                prim_grid, vps = _grid_for(pa, prim, n_car)
                scheme = "carrier- and class-preserving form-pair derangement"
        else:
            para = np.array(sorted(idx_by.get((axis, para_cls), [])), dtype=np.int64)
            install = np.array(sorted(idx_by.get((axis, "install"), [])), dtype=np.int64)
            fams = np.array(sorted(idx_by.get((axis, "famswap"), [])), dtype=np.int64)
            prim_grid, vps = _grid_for(pa, prim, n_car)
            fams_grid, _ = _grid_for(pa, fams, n_car) if fams.size else (None, [])
            if prim_grid is not None and prim_grid.shape[0] >= 2:
                scheme = "carrier- and class-preserving value-pair derangement"
            else:
                scheme = (
                    "sign_randomization_2value — NAMED orientation/sign-randomization "
                    "null (value-pair derangement undefined at C(2,2)=1 swap pair per "
                    "carrier; never a silent cross-carrier fallback)"
                )
        views[axis] = AxisView(
            axis=axis,
            primary_class=prim_cls,
            para_class=para_cls,
            primary_idx=prim,
            para_idx=para,
            install_idx=install,
            famswap_idx=fams,
            primary_grid=prim_grid,
            famswap_grid=fams_grid,
            primary_vps=vps,
            null_scheme=scheme,
        )
    return views


# ── per-axis reads ─────────────────────────────────────────────────────


def _unit(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = x / n
    return np.where(n > 0, out, 0.0)


def direction_null_draws(
    view: AxisView,
    delta_obs: np.ndarray,
    delta_pred: np.ndarray,
    cos_sel: np.ndarray,
    b_null: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Per-draw NULL mean direction cos under the axis's registered scheme."""
    if view.primary_grid is not None and view.primary_grid.shape[0] >= 2:
        grid = view.primary_grid
        n_vp, n_car = grid.shape
        u_obs = _unit(delta_obs[grid])  # (n_vp, n_car, d)
        u_pred = _unit(delta_pred[grid])
        cgrid = np.einsum("icd,jcd->ijc", u_pred, u_obs)  # cos(pred_i, obs_j) per carrier
        perms = deranged_perms(n_vp, b_null * n_car, rng).reshape(b_null, n_car, n_vp)
        car_ix = np.arange(n_car)[None, :, None]
        vp_ix = np.arange(n_vp)[None, None, :]
        return cgrid[perms, vp_ix, car_ix].mean(axis=(1, 2))
    if view.axis == "query_content":
        u_obs = _unit(delta_obs[view.primary_idx])
        u_pred = _unit(delta_pred[view.primary_idx])
        cmat = u_pred @ u_obs.T  # (n_e, n_e)
        n_e = cmat.shape[0]
        if n_e < 2:
            return np.full(b_null, np.nan)
        perms = deranged_perms(n_e, b_null, rng)
        return cmat[perms, np.arange(n_e)[None, :]].mean(axis=1)
    # 2-value axes: NAMED sign-randomization null over the primary pairs.
    vals = cos_sel[np.isfinite(cos_sel)]
    if vals.size == 0:
        return np.full(b_null, np.nan)
    signs = rng.integers(0, 2, size=(b_null, vals.size)) * 2 - 1
    return (signs * vals[None, :]).mean(axis=1)


def _nanmedian_quiet(a: np.ndarray, axis: int) -> np.ndarray:
    """nanmedian with the expected All-NaN-slice warning suppressed (a bootstrap
    draw that resamples < 2 distinct carriers legitimately yields NaN)."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="All-NaN slice encountered")
        return np.nanmedian(a, axis=axis)


def carrier_mean_cos_median(
    grid_a: np.ndarray,
    grid_b: np.ndarray | None,
    delta_a: np.ndarray,
    delta_b: np.ndarray,
    mult: np.ndarray,
    chunk: int = 500,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Point per-vp cos of carrier means (A vs B) + bootstrap draws of the
    per-axis MEDIAN. grid_b=None reuses grid_a on delta_b (same pair grid,
    different delta matrix); otherwise the two grids pair by vp row."""
    gb = grid_a if grid_b is None else grid_b
    da = delta_a[grid_a]  # (n_vp, n_car, d)
    db = delta_b[gb]
    pt = rowwise_cos(da.mean(axis=1), db.mean(axis=1))
    b_tot = mult.shape[0]
    med = np.empty(b_tot, dtype=np.float64)
    for lo in range(0, b_tot, chunk):
        m = mult[lo : lo + chunk]  # (bc, n_car)
        tot = np.maximum(m.sum(axis=1), 1e-12)
        ma = np.einsum("bc,vcd->bvd", m, da) / tot[:, None, None]
        mb = np.einsum("bc,vcd->bvd", m, db) / tot[:, None, None]
        med[lo : lo + chunk] = _nanmedian_quiet(rowwise_cos(ma, mb), axis=1)
    return pt, np.nanmedian(pt[None, :], axis=1), med


def pc1_identity_cos(delta_a: np.ndarray, delta_b: np.ndarray, grid: np.ndarray) -> float:
    """|cos| between the top principal directions (vp-centered) of the
    carrier-mean observed vs predicted delta sets. Exploratory. NaN at n_vp<2."""
    a = delta_a[grid].mean(axis=1)  # (n_vp, d)
    b = delta_b[grid].mean(axis=1)
    if a.shape[0] < 2:
        return float("nan")
    a = a - a.mean(axis=0, keepdims=True)
    b = b - b.mean(axis=0, keepdims=True)
    va = np.linalg.svd(a, full_matrices=False)[2][0]
    vb = np.linalg.svd(b, full_matrices=False)[2][0]
    return float(abs(np.dot(va, vb)))


def offdiag_pairmean_cos(x: np.ndarray) -> float:
    """Mean pairwise cosine over rows (off-diagonal), NaN at <2 rows."""
    if x.shape[0] < 2:
        return float("nan")
    u = _unit(x)
    g = u @ u.T
    n = g.shape[0]
    mask = ~np.eye(n, dtype=bool)
    return float(g[mask].mean())


def boot_pairmean_cos_median(
    grids: np.ndarray, deltas: np.ndarray, idx_draws: np.ndarray, chunk: int = 1000
) -> np.ndarray:
    """Bootstrap (carrier-index resample) of the per-axis MEDIAN over vps of
    the mean pairwise cross-carrier cosine (duplicate carrier draws excluded
    — the issue2215 idiom). grids: (n_vp, n_car) pair indices."""
    u = _unit(deltas[grids])  # (n_vp, n_car, d)
    g = np.einsum("vcd,ved->vce", u, u)  # (n_vp, n_car, n_car)
    b_tot = idx_draws.shape[0]
    out = np.empty(b_tot, dtype=np.float64)
    for lo in range(0, b_tot, chunk):
        ix = idx_draws[lo : lo + chunk]  # (bc, n_car)
        gd = g[:, ix[:, :, None], ix[:, None, :]]  # (n_vp, bc, n_car, n_car)
        distinct = ix[:, :, None] != ix[:, None, :]
        num = (gd * distinct[None]).sum(axis=(2, 3))
        den = distinct.sum(axis=(1, 2))[None, :]
        with np.errstate(invalid="ignore", divide="ignore"):
            vals = np.where(den > 0, num / np.maximum(den, 1), np.nan)  # (n_vp, bc)
        out[lo : lo + chunk] = _nanmedian_quiet(vals, axis=0)
    return out


# ── main analysis ──────────────────────────────────────────────────────


def compute_all(cfg: CfgPE, bank: dict, st: Stores, fire: dict) -> tuple[dict, list[dict], dict]:
    """All §6 reads. Returns (minpair_delta doc, perpair rows, predictions)."""
    t0 = time.time()
    pa = build_pair_arrays(bank, st, cfg.smoke)
    n_car = len(st.carriers)
    d = st.d

    # deltas (float64) ------------------------------------------------
    obs_tail = {
        layer: st.va_tail_mean[layer][pa.a] - st.va_tail_mean[layer][pa.b] for layer in LAYERS
    }
    obs_span19 = st.va_span_mean[PRIMARY_LAYER][pa.a] - st.va_span_mean[PRIMARY_LAYER][pa.b]
    delta_text = st.emb_mean[pa.a] - st.emb_mean[pa.b]

    dev = torch.device("cpu")
    ridge_779 = resolve_ridge(cfg, cfg.ridge_779, RIDGE_779_PATH)
    ridge_1738 = resolve_ridge(cfg, cfg.ridge_1738, RIDGE_1738_PATH)
    payload_779 = load_ridge_payload(ridge_779, d, "arm_779ce")
    payload_1738 = load_ridge_payload(ridge_1738, d, "arm_1738ce")
    ridge_meta = {
        "arm_779ce": {
            "path": str(ridge_779),
            "bytes": ridge_779.stat().st_size,
            "sha256": _sha256(ridge_779),
        },
        "arm_1738ce": {
            "path": str(ridge_1738),
            "bytes": ridge_1738.stat().st_size,
            "sha256": _sha256(ridge_1738),
        },
    }
    mapped_779 = N1M.apply_map(payload_779, st.vc[PRIMARY_LAYER], dev)
    mapped_1738 = N1M.apply_map(payload_1738, st.vc[PRIMARY_LAYER], dev)
    pred = {
        "arm_779ce": mapped_779[pa.a] - mapped_779[pa.b],
        "arm_1738ce": mapped_1738[pa.a] - mapped_1738[pa.b],
        "arm_iddelta": st.vc[PRIMARY_LAYER][pa.a] - st.vc[PRIMARY_LAYER][pa.b],
    }
    pred_iddelta_twin = {
        layer: st.vc[layer][pa.a] - st.vc[layer][pa.b] for layer in LAYERS if layer != PRIMARY_LAYER
    }

    id_check = identity_cancellation_check(
        st.vc[PRIMARY_LAYER], pa.a, pa.b, np.random.default_rng([BOOT_SEED, 999])
    )
    logger.info(
        "[pe] identity-cancellation assert PASS (max_abs_err=%.3e)", id_check["max_abs_err"]
    )

    # per-pair scalars -------------------------------------------------
    cos_arm = {arm: rowwise_cos(pred[arm], obs_tail[PRIMARY_LAYER]) for arm in ARMS}
    cos_arm_span = {arm: rowwise_cos(pred[arm], obs_span19) for arm in ARMS}
    norm_obs = {layer: np.linalg.norm(obs_tail[layer], axis=1) for layer in LAYERS}
    norm_obs_span = np.linalg.norm(obs_span19, axis=1)
    norm_pred = {arm: np.linalg.norm(pred[arm], axis=1) for arm in ARMS}
    norm_text = np.linalg.norm(delta_text, axis=1)
    dlen = st.ans_len_mean[pa.a] - st.ans_len_mean[pa.b]

    rel = split_half_stats(st, pa, cfg.n_splits)
    r10 = rel["r_full"]

    # fire masks -------------------------------------------------------
    fired = {}
    for t in FIRE_THRESHOLDS:
        fa, fb = pair_fired_mask(pa, fire, t)
        fired[t] = fa & fb

    # edit-dose pooled OLS (per arm + observed), residuals -------------
    dose = pa.changed.astype(np.float64)
    dose_fit = {}
    resid = {}
    for name, norms in {"observed": norm_obs[PRIMARY_LAYER], **norm_pred}.items():
        icpt, slope = ols_intercept_slope(norms, dose)
        dose_fit[name] = {"intercept": icpt, "slope": slope, "n": int(pa.n)}
        resid[name] = norms - (icpt + slope * dose)

    # bootstrap + null draws (shared carrier resample, seed 2215) ------
    rng_boot = np.random.default_rng([BOOT_SEED])
    idx_draws = rng_boot.integers(0, n_car, size=(cfg.b_boot, n_car))
    mult = carrier_multiplicities(idx_draws, n_car)

    views = build_axis_views(pa, n_car)
    if not cfg.smoke:
        missing_axes = [a for a in AXES_ALL if a not in views]
        assert not missing_axes, f"axes missing from production stores: {missing_axes}"

    def wm(vals: np.ndarray, sel: np.ndarray) -> tuple[float, list[float]]:
        pt = float(np.nanmean(vals[sel])) if sel.size else float("nan")
        if sel.size == 0:
            return pt, [float("nan"), float("nan")]
        draws = boot_weighted_mean(vals[sel], pa.ca[sel], pa.cb[sel], pa.dyad[sel], mult)
        return pt, _ci(draws)

    def slope_draws(sel: np.ndarray, arm: str) -> np.ndarray:
        num = boot_pair_sums(
            norm_pred[arm][sel] * norm_obs[PRIMARY_LAYER][sel],
            pa.ca[sel],
            pa.cb[sel],
            pa.dyad[sel],
            mult,
        )
        den = boot_pair_sums(
            norm_obs[PRIMARY_LAYER][sel] ** 2, pa.ca[sel], pa.cb[sel], pa.dyad[sel], mult
        )
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, num / den, np.nan)

    all_idx = np.arange(pa.n)
    swap_idx = np.array([i for i in all_idx if pa.cls[i] == "swap"], dtype=np.int64)
    global_slope = {
        arm: through_origin_slope(norm_pred[arm], norm_obs[PRIMARY_LAYER]) for arm in ARMS
    }
    global_slope_swap = {
        arm: through_origin_slope(norm_pred[arm][swap_idx], norm_obs[PRIMARY_LAYER][swap_idx])
        for arm in ARMS
    }
    global_slope_draws = {arm: slope_draws(all_idx, arm) for arm in ARMS}
    global_slope_swap_draws = {arm: slope_draws(swap_idx, arm) for arm in ARMS}

    axes_out: dict[str, dict] = {}
    null_schemes: dict[str, str] = {}
    for k, (axis, view) in enumerate(sorted(views.items())):
        ta = time.time()
        prim = view.primary_idx
        hmask = fired[70][prim]
        head = prim[hmask]
        if head.size == 0:
            head = prim  # compliance-limited axis: fall back, flagged below
        null_schemes[axis] = view.null_scheme

        # fire summary
        ar = fire["axis_rows"].get(axis)
        fire_summary = {
            "axis_row": ar,
            "n_primary_pairs": int(prim.size),
            "n_headline_pairs_fired70": int(prim[hmask].size),
            "compliance_limited": bool(prim[hmask].size == 0),
            "fired_pair_counts": {str(t): int(fired[t][prim].sum()) for t in FIRE_THRESHOLDS},
        }

        # family 5: reliability ceiling (headline pairs)
        ceil_pt, ceil_ci = wm(r10, head)
        rel_axis = {
            "r_half_mean": float(np.nanmean(rel["r_half"][head])),
            "r10_mean": ceil_pt,
            "r10_ci95": ceil_ci,
            "noise_norm_mean": float(np.nanmean(rel["noise_norm"][head])),
            "spearman_brown": "r10 = 2*r5 / (1 + r5)",
        }
        suppressed = suppression_verdict(ceil_pt, ceil_ci[0], ceil_ci[1])

        # family 1: direction fidelity
        rng_null = np.random.default_rng([NULL_SEED, k])
        direction = {}
        for arm in ARMS:
            pt, ci = wm(cos_arm[arm], head)
            pt_all, ci_all = wm(cos_arm[arm], prim)
            nd = direction_null_draws(
                view, obs_tail[PRIMARY_LAYER], pred[arm], cos_arm[arm][prim], cfg.b_null, rng_null
            )
            ratio = float("nan") if suppressed else pt / ceil_pt
            controls = {}
            for cname, cidx in {
                "install": view.install_idx,
                "instruction_paraphrase": view.para_idx if axis not in QUERY_AXES else None,
                "query_paraphrase": view.para_idx if axis in QUERY_AXES else None,
                "famswap": view.famswap_idx,
            }.items():
                if cidx is not None and cidx.size:
                    cpt, cci = wm(cos_arm[arm], cidx)
                    controls[cname] = {"mean_cos": cpt, "ci95": cci, "n_pairs": int(cidx.size)}
            direction[arm] = {
                "mean_cos_headline": pt,
                "ci95": ci,
                "mean_cos_all_values": pt_all,
                "ci95_all_values": ci_all,
                "sensitivity_mean_cos": {
                    str(t): float(np.nanmean(cos_arm[arm][prim[fired[t][prim]]]))
                    if fired[t][prim].any()
                    else float("nan")
                    for t in FIRE_THRESHOLDS
                },
                "null": {
                    "scheme": view.null_scheme,
                    "mean": float(np.nanmean(nd)),
                    "q2_5": _pct(nd, 2.5),
                    "q97_5": _pct(nd, 97.5),
                    "b": cfg.b_null,
                    "seed": [NULL_SEED, k],
                    "over": "all primary-class value pairs (fire mask NOT applied to the null grid)",
                },
                "ceiling_normalized_cos": None if suppressed else ratio,
                "ceiling_suppressed": suppressed,
                "controls": controls,
            }

        # family 2: calibration
        calibration = {}
        for arm in ARMS:
            ax_pt = through_origin_slope(norm_pred[arm][head], norm_obs[PRIMARY_LAYER][head])
            ax_draws = slope_draws(head, arm)
            with np.errstate(invalid="ignore", divide="ignore"):
                ratio_draws = ax_draws / global_slope_draws[arm]
                ratio_swap_draws = ax_draws / global_slope_swap_draws[arm]
            calibration[arm] = {
                "axis_slope": ax_pt,
                "axis_slope_ci95": _ci(ax_draws),
                "global_slope_all2778": global_slope[arm],
                "ratio_to_global": ax_pt / global_slope[arm] if global_slope[arm] else float("nan"),
                "ratio_to_global_ci95": _ci(ratio_draws),
                "global_slope_swap864": global_slope_swap[arm],
                "ratio_to_global_swap864": (
                    ax_pt / global_slope_swap[arm] if global_slope_swap[arm] else float("nan")
                ),
                "ratio_to_global_swap864_ci95": _ci(ratio_swap_draws),
            }

        # family 3: axis identity (carrier-mean per vp)
        identity = {}
        if view.primary_grid is not None:
            for arm in ARMS:
                pt_rows, _, med_draws = carrier_mean_cos_median(
                    view.primary_grid, None, obs_tail[PRIMARY_LAYER], pred[arm], mult
                )
                identity[arm] = {
                    "per_vp_cos": {v: float(c) for v, c in zip(view.primary_vps, pt_rows)},
                    "median": float(np.nanmedian(pt_rows)),
                    "median_ci95": _ci(med_draws),
                    "pc1_identity_cos_exploratory": pc1_identity_cos(
                        obs_tail[PRIMARY_LAYER], pred[arm], view.primary_grid
                    ),
                }
        else:
            identity = {
                "n/a": "query_content pairs are carrier dyads — no carrier-replicated "
                "value pair exists, so the carrier-mean identity read is undefined"
            }

        # family 4: cross-family aspect consistency (instruction axes only)
        cross_family: dict = {}
        if view.famswap_grid is not None and view.primary_grid is not None:
            rng_cf = np.random.default_rng([NULL_SEED, 500 + k])
            spaces = {"observed": (obs_tail[PRIMARY_LAYER], obs_tail[PRIMARY_LAYER])}
            for arm in ARMS:
                spaces[arm] = (pred[arm], pred[arm])
            for space, (da, db) in spaces.items():
                pt_rows, _, med_draws = carrier_mean_cos_median(
                    view.primary_grid, view.famswap_grid, da, db, mult
                )
                n_vp = view.primary_grid.shape[0]
                if n_vp >= 2:
                    sm = _unit(da[view.primary_grid].mean(axis=1))
                    fm = _unit(db[view.famswap_grid].mean(axis=1))
                    cmat = fm @ sm.T
                    perms = deranged_perms(n_vp, cfg.b_null, rng_cf)
                    null_draws = np.median(cmat[perms, np.arange(n_vp)[None, :]], axis=1)
                    nscheme = "class-preserving vp derangement between the two wording families"
                else:
                    signs = rng_cf.integers(0, 2, size=(cfg.b_null, 1)) * 2 - 1
                    null_draws = (signs * np.nan_to_num(pt_rows[None, :], nan=0.0)).mean(axis=1)
                    nscheme = "sign_randomization_2value (single vp — derangement undefined)"
                cross_family[space] = {
                    "per_vp_cos": {v: float(c) for v, c in zip(view.primary_vps, pt_rows)},
                    "median": float(np.nanmedian(pt_rows)),
                    "median_ci95": _ci(med_draws),
                    "null": {
                        "scheme": nscheme,
                        "mean": float(np.nanmean(null_draws)),
                        "q2_5": _pct(null_draws, 2.5),
                        "q97_5": _pct(null_draws, 97.5),
                        "b": cfg.b_null,
                    },
                }
        else:
            cross_family = {"n/a": "no paraphrase-family swap class for this axis"}

        # family 6: text third space (observed only)
        flip_txt_pt, flip_txt_ci = wm(norm_text, prim)
        para_txt_pt, para_txt_ci = wm(norm_text, view.para_idx)
        text_space = {
            "flip_norm_mean": flip_txt_pt,
            "flip_norm_ci95": flip_txt_ci,
            "paraphrase_null_norm_mean": para_txt_pt,
            "paraphrase_null_norm_ci95": para_txt_ci,
            "flip_over_para_ratio": (
                flip_txt_pt / para_txt_pt
                if para_txt_pt and np.isfinite(para_txt_pt)
                else float("nan")
            ),
            "note": "Qwen3-Embedding-8B mean answer embeddings (means of L2-normalized "
            "per-draw rows, NOT re-normalized); observed only — no predicted arm exists "
            "in text space",
        }
        if view.primary_grid is not None:
            pt_rows = np.array(
                [
                    offdiag_pairmean_cos(delta_text[view.primary_grid[v]])
                    for v in range(len(view.primary_vps))
                ]
            )
            cons_draws = boot_pairmean_cos_median(view.primary_grid, delta_text, idx_draws)
            text_space["cross_carrier_consistency"] = {
                "per_vp_mean_pairwise_cos": {
                    v: float(c) for v, c in zip(view.primary_vps, pt_rows)
                },
                "median": float(np.nanmedian(pt_rows)),
                "median_ci95": _ci(cons_draws),
            }
        else:
            text_space["cross_carrier_consistency"] = None

        # family 7: surface sensitivity (descriptive) + edit-dose companion
        surface = {}
        for name, norms in {"observed": norm_obs[PRIMARY_LAYER], **norm_pred}.items():
            fpt, fci = wm(norms, prim)
            ppt, pci = wm(norms, view.para_idx)
            rf, _ = wm(resid[name], prim)
            rp, _ = wm(resid[name], view.para_idx)
            gap_draws = boot_weighted_mean(
                norms[prim], pa.ca[prim], pa.cb[prim], pa.dyad[prim], mult
            ) - boot_weighted_mean(
                norms[view.para_idx],
                pa.ca[view.para_idx],
                pa.cb[view.para_idx],
                pa.dyad[view.para_idx],
                mult,
            )
            surface[name] = {
                "flip_norm_mean": fpt,
                "flip_norm_ci95": fci,
                "para_norm_mean": ppt,
                "para_norm_ci95": pci,
                "gap": fpt - ppt,
                "gap_ci95": _ci(gap_draws),
                "edit_dose_ols": dose_fit[name],
                "residualized_gap": rf - rp,
                "labeling": "DESCRIPTIVE only (plan §6: a surface-transducing map "
                "predicts a NEGATIVE gap; no H-label keys on this gap)",
            }

        # answer-length deltas (convention 6; from the capture index, gate-4
        # asserted equal to the generation-side counts)
        answer_length = {}
        for cname, cidx in {
            "primary": prim,
            "paraphrase": view.para_idx,
            "install": view.install_idx,
            "famswap": view.famswap_idx,
        }.items():
            if cidx is not None and cidx.size:
                answer_length[cname] = {
                    "mean_delta_tokens": float(np.nanmean(dlen[cidx])),
                    "mean_abs_delta_tokens": float(np.nanmean(np.abs(dlen[cidx]))),
                }

        # layer twins (arm_iddelta only — the frozen ridge maps are L19 by
        # construction) + span-pooling twin (point estimates)
        layer_twins = {}
        for layer in (14, 26):
            c = rowwise_cos(pred_iddelta_twin[layer], obs_tail[layer])
            no_l = np.linalg.norm(pred_iddelta_twin[layer], axis=1)
            ax_slope = through_origin_slope(no_l[head], norm_obs[layer][head])
            gl = through_origin_slope(no_l, norm_obs[layer])
            layer_twins[str(layer)] = {
                "arm_iddelta_mean_cos_headline": float(np.nanmean(c[head])),
                "arm_iddelta_ratio_to_global": ax_slope / gl if gl else float("nan"),
                "note": "iddelta only — ridge arms are L19-fit and have no twin at this layer",
            }
        span_twin = {
            arm: {
                "mean_cos_headline": float(np.nanmean(cos_arm_span[arm][head])),
                "axis_slope": through_origin_slope(norm_pred[arm][head], norm_obs_span[head]),
            }
            for arm in ARMS
        }

        axes_out[axis] = {
            "axis": axis,
            "primary_class": view.primary_class,
            "para_class": view.para_class,
            "n_primary_pairs": int(prim.size),
            "fire": fire_summary,
            "direction": direction,
            "calibration": calibration,
            "identity": identity,
            "cross_family": cross_family,
            "reliability": rel_axis,
            "text_space": text_space,
            "surface": surface,
            "answer_length": answer_length,
            "layer_twins": layer_twins,
            "pooling_twin_span": span_twin,
        }
        print(
            f"[pe] axis {k + 1}/{len(views)} {axis} elapsed={time.time() - ta:.1f}s",
            flush=True,
        )

    # kNN delta retrieval ----------------------------------------------
    retrieval: dict = {"global": {}, "per_axis": {}}
    pool = obs_tail[PRIMARY_LAYER]
    for arm in ARMS:
        retrieval["global"][arm] = {
            metric: knn_retrieval(pred[arm], pool, ks=(1, 5, 10), metric=metric, pool=pool)
            for metric in ("cosine", "euclidean")
        }
    for axis, view in sorted(views.items()):
        sel = view.primary_idx
        ks = tuple(k_ for k_ in (1, 5, 10) if k_ <= sel.size)
        if not ks:
            continue
        retrieval["per_axis"][axis] = {
            arm: {
                metric: knn_retrieval(
                    pred[arm][sel], pool[sel], ks=ks, metric=metric, pool=pool[sel]
                )
                for metric in ("cosine", "euclidean")
            }
            for arm in ARMS
        }

    # per-pair rows ------------------------------------------------------
    fa70, fb70 = pair_fired_mask(pa, fire, 70)
    perpair: list[dict] = []
    for i in range(pa.n):
        perpair.append(
            {
                "pair_id": pa.ids[i],
                "pair_class": pa.cls[i],
                "axis": pa.axis[i],
                "carrier": pa.carrier_str[i],
                "value_a": pa.value_a[i],
                "value_b": pa.value_b[i],
                "orientation": pa.orientation[i],
                "changed_tokens": int(pa.changed[i]),
                "n_draws_a": int(st.n_valid[pa.a[i]]),
                "n_draws_b": int(st.n_valid[pa.b[i]]),
                "ans_len_delta": float(dlen[i]),
                "norm_obs_tail_L19": float(norm_obs[PRIMARY_LAYER][i]),
                "norm_obs_span_L19": float(norm_obs_span[i]),
                "norm_text": float(norm_text[i]),
                "cos": {arm: float(cos_arm[arm][i]) for arm in ARMS},
                "cos_span": {arm: float(cos_arm_span[arm][i]) for arm in ARMS},
                "norm_pred": {arm: float(norm_pred[arm][i]) for arm in ARMS},
                "r_half": float(rel["r_half"][i]),
                "r10": float(r10[i]),
                "noise_norm": float(rel["noise_norm"][i]),
                "fired_a_70": bool(fa70[i]),
                "fired_b_70": bool(fb70[i]),
                "in_headline_70": bool(fa70[i] and fb70[i]),
            }
        )

    contract = {
        "primary_pair_classes": {
            dv: {
                "instruction_axes": ["swap"],
                "query_axes": ["query_content", "query_form"],
                "controls": ["install", "instruction_paraphrase", "query_paraphrase"],
                "cross_family_read_class": ["famswap"],
            }
            for dv in (
                "direction_fidelity",
                "magnitude_calibration",
                "axis_identity",
                "cross_family_consistency",
                "reliability_ceiling",
                "text_third_space",
                "surface_sensitivity",
                "knn_delta_retrieval",
            )
        },
        "null_scheme": null_schemes,
        "bootstrap": {
            "scheme": "carrier-clustered (resample the 12 carrier clusters with replacement)",
            "query_content": DYADIC_BOOTSTRAP_CONVENTION,
            "B": cfg.b_boot,
            "seed": BOOT_SEED,
            "gsmall_caveat": "G=12 clusters — percentile CIs undercover (plan §6 convention 8)",
        },
        "null": {"B": cfg.b_null, "seed": NULL_SEED},
        "draw_to_pair_aggregation": DRAW_TO_PAIR_AGGREGATION,
        "orientation_conventions": ORIENTATION_CONVENTIONS,
        "split_half": {
            "n_splits": cfg.n_splits,
            "seed": SPLIT_SEED,
            "step_up": "Spearman-Brown r10 = 2*r5/(1+r5)",
            "n_pairs_insufficient_draws": rel["n_pairs_insufficient_draws"],
        },
        "compliance": {
            "headline_threshold_pct": 70,
            "sensitivity_pcts": [50, 90],
            "rule": "headline per-axis reads use pairs whose BOTH endpoint values fired; "
            "non-fired values stay in the artifact (hollow), excluded from the headline",
        },
    }

    doc = {
        "meta": {
            "issue": ISSUE,
            "phase": "pe_analysis",
            "smoke": cfg.smoke,
            "layer_primary": PRIMARY_LAYER,
            "layers": list(LAYERS),
            "pooling_primary": "tail_inclusive_mean",
            "pooling_twin": "span_mean",
            "n_contexts": len(st.ctx_ids),
            "n_pairs": pa.n,
            "cells": st.cells,
            "carriers": st.carriers,
            "arms": list(ARMS),
            "ridge_payloads": ridge_meta,
            "input_files": st.input_files,
            "manip_check_meta": {
                k: fire["meta"].get(k)
                for k in ("judged_denominator", "fire_threshold_pct", "floor_rule")
            },
            "identity_cancellation_assert": id_check,
            "elapsed_s": round(time.time() - t0, 1),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            **as_metadata_dict(git_provenance(), phase="pe-analysis"),
        },
        "contract": contract,
        "axes": axes_out,
        "retrieval": retrieval,
    }
    predictions = {
        "pair_ids": pa.ids,
        "delta_obs_tail_L19": torch.from_numpy(obs_tail[PRIMARY_LAYER].astype(np.float32)),
        "delta_obs_span_L19": torch.from_numpy(obs_span19.astype(np.float32)),
        **{f"delta_pred_{arm}": torch.from_numpy(pred[arm].astype(np.float32)) for arm in ARMS},
    }
    return doc, perpair, predictions


# ── io + main ──────────────────────────────────────────────────────────


def _write_json_atomic(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=True))
    os.replace(tmp, path)


def _json_sanitize(obj):
    """NaN/inf -> None recursively (JSON round-trip safety for consumers)."""
    if isinstance(obj, dict):
        return {k: _json_sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_json_sanitize(v) for v in obj]
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, np.generic):
        v = obj.item()
        return None if isinstance(v, float) and not math.isfinite(v) else v
    return obj


def write_outputs(cfg: CfgPE, doc: dict, perpair: list[dict], predictions: dict) -> dict:
    cfg.out_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(cfg.out_dir / "minpair_delta.json", _json_sanitize(doc))
    rows = [json.dumps(_json_sanitize(r), sort_keys=True) for r in perpair]
    tmp = cfg.out_dir / "perpair.jsonl.tmp"
    tmp.write_text("\n".join(rows) + "\n")
    os.replace(tmp, cfg.out_dir / "perpair.jsonl")

    cfg.pred_dir.mkdir(parents=True, exist_ok=True)
    pair_ids = predictions["pair_ids"]
    for name, tensor in predictions.items():
        if name == "pair_ids":
            continue
        dest = cfg.pred_dir / f"{name}.pt"
        tmpp = cfg.pred_dir / f"{name}.tmp.pt"
        torch.save(
            {"issue": ISSUE, "pair_ids": pair_ids, "layer": PRIMARY_LAYER, "tensor": tensor},
            tmpp,
        )
        os.replace(tmpp, dest)
    upload: dict = {"mode": cfg.upload}
    if cfg.upload == "hf":
        from explore_persona_space.orchestrate.upload_sharded import upload_dir_sharded

        res = upload_dir_sharded(
            cfg.pred_dir,
            HF_DATA_REPO,
            f"{cfg.hf_prefix}/analysis_tensors/predictions",
            shard_glob="*.pt",
            resume_skip=False,
            delete_local=False,
        )
        upload["predictions"] = {
            "uploaded": len(res.uploaded),
            "skipped_existing": len(res.skipped_existing),
            "rerouted": len(res.rerouted),
        }
    return upload


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO,
        stream=sys.stdout,
        format="%(asctime)s %(name)s %(levelname)s %(message)s",
    )
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("[import-check] ok", flush=True)
        return 0
    cfg = build_config(args)
    print(
        f"[phase=pe_analysis] smoke={cfg.smoke} in_root={cfg.in_root} out_dir={cfg.out_dir} "
        f"b_boot={cfg.b_boot} b_null={cfg.b_null} n_splits={cfg.n_splits} upload={cfg.upload}",
        flush=True,
    )
    bank_path = resolve_input(cfg, "manifests/bank2564_manifest.json")
    bank = load_bank_manifest(bank_path)
    assert cfg.manip_check.exists(), f"manipulation check missing: {cfg.manip_check}"
    fire = load_fire(cfg.manip_check)
    st = load_stores(cfg, bank)
    st.input_files["bank2564_manifest.json"] = {
        "path": str(bank_path),
        "bytes": bank_path.stat().st_size,
        "sha256": _sha256(bank_path),
    }
    st.input_files["manipulation_check.json"] = {
        "path": str(cfg.manip_check),
        "bytes": cfg.manip_check.stat().st_size,
    }
    doc, perpair, predictions = compute_all(cfg, bank, st, fire)
    upload = write_outputs(cfg, doc, perpair, predictions)
    print(
        f"[pe] wrote {cfg.out_dir / 'minpair_delta.json'} + perpair.jsonl "
        f"({len(perpair)} rows) + predictions ({upload})",
        flush=True,
    )
    print("[phase=done] pe_analysis complete", flush=True)
    return 0


if __name__ == "__main__":
    _rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(_rc)
