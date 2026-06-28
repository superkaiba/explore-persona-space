#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ĉ, ρ, →, ×, ‖, ⟨, ⟩) in scientific docstrings + log messages.
"""Issue #661 P4 + P5 (off-pod, no GPU): r_B divergence analysis + figures.

Reads:
- arm-A / arm-C directions + context axis (P3 ``directions/r_b_<behavior>.pt``);
- arm-B from #658 ``r_b.pt`` (``rb['r_b'][col]['diffmeans']`` — the ``['r_b']``
  index is REQUIRED; top-level keys are ``['r_b', 'capture_layers', 'columns']``);
- v0(C) from #658 ``v0_summaries.pt`` (``summaries['mean'][ctx]`` = (28, H), 50
  contexts) for M3;
- E0(C,B) rates — ``E0_expression.json`` if on HF, else RECOMPUTE from #658
  ``e0_gen/`` via ``issue658_judge_e0_batch`` (plan §6.4 assert-or-recompute).

Computes three measurements, per behavior, at all 28 layers + the per-behavior
SELECTED layer (the fallback: the layer maximizing arm-B's M3 LOCO ρ —
computable today; #658's per-behavior lock not yet stamped, plan §A3):

- **M1 pairwise cosine** ``cos(r_B^i, r_B^j)`` for i,j ∈ {A,B,C}, per layer +
  selected, with a paired context-bootstrap... (cosine is over the H-dim, so
  the "bootstrap" here resamples the SURVIVOR set is N/A for the final fixed
  directions — see note). M1 CIs come from the survivor-resample re-extraction
  only when survivor spans are available; otherwise reported as point estimates.
- **M2 context-confound** ``|⟨r_B^A, ĉ_inst⟩| / ‖r_B^A‖`` per layer + selected,
  ``ĉ_inst = c_pos − c_neg``. The §5 control ALSO projects r_B^B and r_B^C onto
  the SAME ĉ_inst (should be near-zero if ĉ_inst is the instruction axis).
- **M3 held-out predictive ρ** — LOCO 1-D OLS ``E0 ~ a·(r·v0) + b`` per held-out
  context, then Spearman ρ (PRIMARY) + Pearson (SECONDARY) of the held-out
  predictions vs measured E0, per (method, behavior) at the selected layer.
  Per-behavior #658-style noise floor via probe-redraw test-retest.

Decision (plan §7), at the selected layer, on the 3 headline behaviors:
- Adopt A iff cos(A,C)≥0.95 AND confound≤0.10 AND A's ρ ≥ C's ρ within noise on ALL 3.
- Adopt C if ANY: cos(A,C)<0.85 on ≥2, OR confound>0.25 on ≥2, OR A's ρ < C's ρ by ≥0.05 on ≥2.
- Inconclusive / recipe-by-behavior otherwise.

Writes the §6.5 primary deliverables (cosine_divergence.json, context_confound.json,
a33_predictive.json, decision.json) + hero figures F1/F2/F3.

Content hygiene: E0 recompute reads harmful-content-adjacent completions — it
NEVER pages raw text into this script's logs (counts / rates only).

Usage::

    uv run python scripts/issue661_analysis.py \
        --behaviors sycophancy refusal broad_em \
        --directions-dir eval_results/issue_661/directions \
        --out-dir eval_results/issue_661

    # smoke (1 behavior, 2 contexts, skip figures heavy path):
    uv run python scripts/issue661_analysis.py --behaviors sycophancy \
        --directions-dir /tmp/i661_smoke/directions --out-dir /tmp/i661_smoke \
        --bootstrap-n 50 --max-contexts 2
"""

from __future__ import annotations

import argparse
import logging
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue661_common import (  # noqa: E402
    BEHAVIOR_TO_COLUMN,
    BOOTSTRAP_N,
    EVAL_RESULTS_DIR,
    HF_DATA_REPO,
    MIN_SURVIVORS,
    dump_json,
    load_json,
)

# §7 fallback noise floor for the adopt-A ρ conjunct when a behavior's M3
# noise_floor_p95 is None (too few contexts / no per_probe redraw data). The
# plan §7 adopt-A leg reads "A's ρ ≥ C's ρ within #658 noise floor"; when the
# per-behavior floor cannot be estimated we fall back to this registered tolerance
# (the adopt-C strong-fail threshold's mirror: adopt-C fires at ρ_gap ≤ −0.05, so
# the adopt-A "within noise" band is at most that wide). Stated as a named plan
# deviation in the clean-result whenever it is used.
DEFAULT_NOISE_FLOOR = 0.05

# Fixed per-method bootstrap-seed offsets (reproducible M1/M2/M3 CI bounds —
# NOT hash(m), which is PYTHONHASHSEED-randomized run-to-run).
_METHOD_SEED_OFFSET = {"A": 0, "B": 1, "C": 2}

load_dotenv(str(PROJECT_ROOT / ".env"))
logger = logging.getLogger("issue661_analysis")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

I658_STORE_PREFIX = "issue658_theory_assumptions/store"
I658_E0_HF = "issue658_theory_assumptions/E0_expression.json"
I658_E0GEN_PREFIX = "issue658_theory_assumptions/raw_completions/e0_gen"


# ── HF reads (arm B / v0 / E0) ───────────────────────────────────────────────


def load_arm_b(behaviors: list[str]) -> dict:
    """Arm-B r_B per behavior from #658 r_b.pt (diffmeans). ['r_b'] index REQUIRED.

    FAIL-LOUD on a missing column for any REQUESTED behavior (no warn-and-continue):
    arm B is the #658 status-quo reference and the §5 control axis — a missing
    column silently dropping a behavior here violates the project's fail-fast rule
    and the plan §12 A1/A2 "re-assert P4" registration. The assertion names the
    EXACT missing column → behavior pairs so the failure is diagnosable. (There is
    no registered E0-saturation / recompute branch for arm B — that branch applies
    only to v0/E0 in M3; arm B's column existence is unconditional.)
    """
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(HF_DATA_REPO, f"{I658_STORE_PREFIX}/r_b.pt", repo_type="dataset")
    rb = torch.load(path, weights_only=False)
    assert set(["r_b", "capture_layers", "columns"]).issubset(rb.keys()), (
        f"r_b.pt top keys unexpected: {list(rb.keys())}"
    )
    missing = [
        (BEHAVIOR_TO_COLUMN[b], b) for b in behaviors if BEHAVIOR_TO_COLUMN[b] not in rb["r_b"]
    ]
    assert not missing, (
        f"arm B: requested columns absent from #658 r_b.pt — {missing}; "
        f"available r_b columns: {sorted(rb['r_b'].keys())}"
    )
    out: dict[str, torch.Tensor] = {}
    for behavior in behaviors:
        col = BEHAVIOR_TO_COLUMN[behavior]
        out[behavior] = rb["r_b"][col]["diffmeans"].float()  # (28, H)
    return out


def load_v0() -> dict:
    """v0(C) summaries (mean recipe) from #658. Returns {ctx_id: (28, H) fp32}."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        HF_DATA_REPO, f"{I658_STORE_PREFIX}/v0_summaries.pt", repo_type="dataset"
    )
    blob = torch.load(path, weights_only=False)
    summ = blob["summaries"]["mean"]
    return {c: summ[c].float() for c in summ}


def load_or_recompute_e0(out_dir: Path, behaviors: list[str]) -> dict:
    """E0(C,B) rates. Prefer the HF E0_expression.json; else recompute (plan §6.4).

    The recompute downloads the #658 ``e0_gen/`` raw completions and runs
    ``issue658_judge_e0_batch`` over them (off-pod, scheduled in-scope). The
    result is cached to ``out_dir/E0_expression.json`` so a re-run is free.
    """
    from huggingface_hub import hf_hub_download, list_repo_files

    local_cache = out_dir / "E0_expression.json"
    if local_cache.exists():
        logger.info("E0: using local cache %s", local_cache)
        return load_json(local_cache)

    files = list_repo_files(HF_DATA_REPO, repo_type="dataset")
    if I658_E0_HF in files:
        path = hf_hub_download(HF_DATA_REPO, I658_E0_HF, repo_type="dataset")
        logger.info("E0: downloaded existing E0_expression.json from HF")
        e0 = load_json(Path(path))
        dump_json(e0, local_cache)
        return e0

    # Recompute: download e0_gen/ then run the Batch judge (plan §6.4).
    logger.info("E0: E0_expression.json absent on HF — recomputing from e0_gen/ (plan §6.4)")
    e0gen_files = [
        f for f in files if f.startswith(I658_E0GEN_PREFIX + "/") and f.endswith(".json")
    ]
    if not e0gen_files:
        raise RuntimeError(f"no e0_gen files under {I658_E0GEN_PREFIX} — cannot recompute E0")
    e0gen_dir = out_dir / "e0_gen"
    e0gen_dir.mkdir(parents=True, exist_ok=True)
    # Only the headline + control columns we actually score (keeps the judge set
    # bounded; the batch judge skips columns it does not need via E0_COLUMNS).
    for f in e0gen_files:
        dst = e0gen_dir / Path(f).name
        if dst.exists():
            continue
        src = hf_hub_download(HF_DATA_REPO, f, repo_type="dataset")
        dst.write_bytes(Path(src).read_bytes())
    logger.info("E0: staged %d e0_gen files; launching Batch judge", len(e0gen_files))
    from issue658_judge_e0_batch import main as judge_main

    judge_argv = ["--e0-dir", str(e0gen_dir), "--out", str(local_cache)]
    rc = judge_main_via_argv(judge_main, judge_argv)
    if rc != 0:
        raise RuntimeError(f"E0 recompute judge exited rc={rc}")
    return load_json(local_cache)


def judge_main_via_argv(judge_main, argv: list[str]) -> int:
    """Invoke issue658_judge_e0_batch.main() with a temporary argv."""
    old = sys.argv
    try:
        sys.argv = ["issue658_judge_e0_batch", *argv]
        return judge_main()
    finally:
        sys.argv = old


# ── Survivor-set bootstrap (plan §6.5: M1/M2 CIs) ────────────────────────────


def _ci95_from_samples(samples: list[float]) -> list[float] | None:
    """2.5/97.5-pct CI from bootstrap samples (≥50 needed, else None)."""
    if len(samples) < 50:
        return None
    s = sorted(samples)
    return [s[int(0.025 * len(s))], s[int(0.975 * len(s)) - 1]]


def _resample_mean(stack: torch.Tensor, idx: list[int], layer: int) -> torch.Tensor:
    """Mean over a resampled survivor index set of a (N, L, H) stack at one layer.

    Returns the (H,) layer-vector mean over the resampled rows.
    """
    return stack[idx, layer, :].mean(dim=0)


# ── M1 cosine ─────────────────────────────────────────────────────────────────


def cosine_per_layer(a: torch.Tensor, b: torch.Tensor) -> np.ndarray:
    """Per-layer cosine between two (L, H) direction stacks. Returns (L,)."""
    an = torch.nn.functional.normalize(a, dim=-1)
    bn = torch.nn.functional.normalize(b, dim=-1)
    return (an * bn).sum(dim=-1).numpy()


def _cos_one_layer(u: torch.Tensor, v: torch.Tensor) -> float:
    """Cosine between two (H,) vectors."""
    un = torch.nn.functional.normalize(u, dim=-1)
    vn = torch.nn.functional.normalize(v, dim=-1)
    return float((un * vn).sum())


def m1_cos_ac_ci(d: dict, sl: int, *, bootstrap_n: int, seed: int = 661) -> list[float] | None:
    """§6.5 survivor-set bootstrap CI for cos(A, C) at the selected layer.

    A and C share the SAME survivor texts (only the read context differs), so we
    resample a SHARED present-index set and a SHARED absent-index set jointly
    (paired), recompute r_B^A = mean(present_A) − mean(absent_A) and r_B^C from the
    SAME resampled rows, and recompute the cosine. Returns None if the per-survivor
    stacks are absent (pre-round-2 .pt) — point estimates still report.
    """
    pA, aA = d.get("present_A_items"), d.get("absent_A_items")
    pC, aC = d.get("present_C_items"), d.get("absent_C_items")
    if any(x is None for x in (pA, aA, pC, aC)):
        return None
    n_pos, n_neg = pA.shape[0], aA.shape[0]
    # A vs C share survivor identity → the present (pos) and absent (neg) index
    # sets are shared across arms within a resample.
    assert pC.shape[0] == n_pos and aC.shape[0] == n_neg, "A/C survivor count mismatch"
    rng = random.Random(seed + _METHOD_SEED_OFFSET["A"])
    samples: list[float] = []
    for _ in range(bootstrap_n):
        pos_idx = [rng.randrange(n_pos) for _ in range(n_pos)]
        neg_idx = [rng.randrange(n_neg) for _ in range(n_neg)]
        rA = _resample_mean(pA, pos_idx, sl) - _resample_mean(aA, neg_idx, sl)
        rC = _resample_mean(pC, pos_idx, sl) - _resample_mean(aC, neg_idx, sl)
        samples.append(_cos_one_layer(rA, rC))
    return _ci95_from_samples(samples)


def m1_cosine(
    directions: dict,
    arm_b: dict,
    behaviors: list[str],
    selected_layer: dict,
    *,
    bootstrap_n: int = BOOTSTRAP_N,
) -> dict:
    """Pairwise cosine cos_AB / cos_AC / cos_BC per layer + selected, per behavior.

    The pooled directions are deterministic, so the headline cosine is a fixed
    scalar; the §6.5 CI at the selected layer comes from the survivor-set bootstrap
    (``m1_cos_ac_ci``) — resample the shared survivor index set jointly (paired
    A-vs-C), recompute r_B^A/r_B^C, recompute cos(A,C). Emitted as ``cos_AC_ci95``
    (None when the per-survivor stacks are absent, e.g. a pre-round-2 .pt).
    """
    out: dict = {}
    for behavior in behaviors:
        d = directions[behavior]
        rA, rC = d["r_b_a"], d["r_b_c"]
        rB = arm_b.get(behavior)
        sl = selected_layer[behavior]
        cos_AC = cosine_per_layer(rA, rC)
        rec = {
            "cos_AC": cos_AC.tolist(),
            "selected_layer": sl,
            "cos_AC_selected": float(cos_AC[sl]),
            "cos_AC_ci95": m1_cos_ac_ci(d, sl, bootstrap_n=bootstrap_n),
        }
        if rB is not None:
            cos_AB = cosine_per_layer(rA, rB)
            cos_BC = cosine_per_layer(rB, rC)
            rec["cos_AB"] = cos_AB.tolist()
            rec["cos_BC"] = cos_BC.tolist()
            rec["cos_AB_selected"] = float(cos_AB[sl])
            rec["cos_BC_selected"] = float(cos_BC[sl])
        out[behavior] = rec
    return out


# ── M2 context-confound ─────────────────────────────────────────────────────


def projection_fraction(r: torch.Tensor, axis: torch.Tensor) -> np.ndarray:
    """|cos(r[L], axis[L])| per layer — the UNIT-normalized projection (the M2
    decision-gate quantity, bounded in [0, 1]).

    The plan §6.1 writes the confound as ``|⟨r_B^A, ĉ_inst⟩| / ‖r_B^A‖``, but the
    §7 thresholds (0.10 / 0.25) are only meaningful as a BOUNDED fraction — the
    raw form equals ``‖ĉ_inst‖·|cos|`` and is unbounded when ĉ_inst is not a unit
    vector. So the gate quantity is ``|⟨r, ĉ_inst⟩| / (‖r‖·‖ĉ_inst‖) = |cos|``;
    ``projection_fraction_raw`` keeps the un-normalized companion the plan §4.4
    also asks to report.
    """
    rn = torch.nn.functional.normalize(r, dim=-1)
    an = torch.nn.functional.normalize(axis, dim=-1)
    return (rn * an).sum(dim=-1).abs().numpy()


def projection_fraction_raw(r: torch.Tensor, axis: torch.Tensor) -> np.ndarray:
    """|⟨r[L], axis[L]⟩| / ‖r[L]‖ per layer (axis NOT normalized) — the plan §4.4
    companion read. Reported alongside the bounded gate quantity; never gated on
    (it is unbounded). Returns (L,)."""
    dot = (r * axis).sum(dim=-1).abs()
    norm = r.norm(dim=-1).clamp_min(1e-12)
    return (dot / norm).numpy()


def m2_confound_ci(
    d: dict, sl: int, *, bootstrap_n: int, seed: int = 661
) -> tuple[list[float] | None, list[float] | None]:
    """§6.5 survivor-set bootstrap CI for the M2 A-confound at the selected layer.

    Resample present_A / absent_A (→ r_B^A) and the c_pos / c_neg instruction
    prompts (→ ĉ_inst) independently, recompute BOTH the bounded gate quantity
    ``|cos(r_B^A, ĉ_inst)|`` AND the raw companion ``|⟨r,ĉ⟩|/‖r‖`` at the selected
    layer. Returns ``(bounded_ci95, raw_ci95)`` — each None when the per-survivor
    stacks are absent (pre-round-2 .pt).
    """
    pA, aA = d.get("present_A_items"), d.get("absent_A_items")
    cp, cn = d.get("c_pos_items"), d.get("c_neg_items")
    if any(x is None for x in (pA, aA, cp, cn)):
        return None, None
    n_pos, n_neg = pA.shape[0], aA.shape[0]
    n_cp, n_cn = cp.shape[0], cn.shape[0]
    rng = random.Random(seed + _METHOD_SEED_OFFSET["A"])
    bounded: list[float] = []
    raw: list[float] = []
    for _ in range(bootstrap_n):
        rA = _resample_mean(pA, [rng.randrange(n_pos) for _ in range(n_pos)], sl) - _resample_mean(
            aA, [rng.randrange(n_neg) for _ in range(n_neg)], sl
        )
        c_inst = _resample_mean(
            cp, [rng.randrange(n_cp) for _ in range(n_cp)], sl
        ) - _resample_mean(cn, [rng.randrange(n_cn) for _ in range(n_cn)], sl)
        bounded.append(abs(_cos_one_layer(rA, c_inst)))
        dot = float((rA * c_inst).sum().abs())
        raw.append(dot / max(float(rA.norm()), 1e-12))
    return _ci95_from_samples(bounded), _ci95_from_samples(raw)


def m2_confound(
    directions: dict,
    arm_b: dict,
    behaviors: list[str],
    selected_layer: dict,
    *,
    bootstrap_n: int = BOOTSTRAP_N,
) -> dict:
    """Confound = |cos(r_B^A, ĉ_inst)| per layer (bounded gate quantity) + the
    B/C controls + the raw un-normalized companion + §6.5 bootstrap CIs."""
    out: dict = {}
    for behavior in behaviors:
        d = directions[behavior]
        c_inst = d["c_pos"] - d["c_neg"]
        conf_A = projection_fraction(d["r_b_a"], c_inst)
        conf_C = projection_fraction(d["r_b_c"], c_inst)  # control: should be near 0
        sl = selected_layer[behavior]
        bounded_ci, raw_ci = m2_confound_ci(d, sl, bootstrap_n=bootstrap_n)
        rec = {
            "confound_A": conf_A.tolist(),
            "confound_C_control": conf_C.tolist(),
            "confound_A_selected": float(conf_A[sl]),
            "confound_C_control_selected": float(conf_C[sl]),
            # §6.5 CI on the bounded gate quantity (the §7 0.10/0.25 gate reads this).
            "confound_A_ci95": bounded_ci,
            # Plan §4.4 companion: the un-normalized |⟨r, ĉ_inst⟩|/‖r‖ (not gated).
            "confound_A_raw": projection_fraction_raw(d["r_b_a"], c_inst).tolist(),
            "confound_A_raw_selected": float(projection_fraction_raw(d["r_b_a"], c_inst)[sl]),
            "confound_A_raw_ci95": raw_ci,
            "selected_layer": sl,
        }
        rB = arm_b.get(behavior)
        if rB is not None:
            conf_B = projection_fraction(rB, c_inst)  # control: should be near 0
            rec["confound_B_control"] = conf_B.tolist()
            rec["confound_B_control_selected"] = float(conf_B[sl])
        out[behavior] = rec
    return out


# ── M3 LOCO 1-D OLS held-out predictive ρ ─────────────────────────────────────


def loco_1d_ols_predictions(proj: np.ndarray, e0: np.ndarray) -> np.ndarray:
    """Held-out predictions of a 1-D OLS E0 ~ a*proj + b, leave-one-context-out.

    For each held-out context i, fit (a, b) on the OTHER contexts by ordinary
    least squares of E0 on the projection scalar, then predict context i. This
    is the plan §4.4 LOCO definition (NOT fit_a33's global pred = X @ r ranking,
    which fits no held-out model). Returns (N,) held-out predictions.
    """
    n = len(proj)
    preds = np.zeros(n, dtype=np.float64)
    for i in range(n):
        tr = [j for j in range(n) if j != i]
        x = proj[tr]
        y = e0[tr]
        if np.std(x) < 1e-12:
            preds[i] = float(np.mean(y))  # degenerate predictor → predict the mean
            continue
        a, b = np.polyfit(x, y, 1)
        preds[i] = a * proj[i] + b
    return preds


def _spearman(pred: np.ndarray, meas: np.ndarray) -> float | None:
    from scipy.stats import spearmanr

    if len(pred) < 4 or np.std(pred) < 1e-12 or np.std(meas) < 1e-12:
        return None
    r, _ = spearmanr(pred, meas)
    return None if np.isnan(r) else float(r)


def _pearson(pred: np.ndarray, meas: np.ndarray) -> float | None:
    from scipy.stats import pearsonr

    if len(pred) < 4 or np.std(pred) < 1e-12 or np.std(meas) < 1e-12:
        return None
    r, _ = pearsonr(pred, meas)
    return None if np.isnan(r) else float(r)


def e0_rate_vector(e0: dict, behavior: str, ctx_ids: list[str]) -> tuple[np.ndarray, list[str]]:
    """Per-context E0 rate for one behavior column. Returns (values, kept_ctx)."""
    col = BEHAVIOR_TO_COLUMN[behavior]
    vals, kept = [], []
    table = e0.get("e0", {})
    for c in ctx_ids:
        cell = table.get(c, {}).get(col)
        if cell is None:
            continue
        v = cell.get("rate")
        if v is None:
            continue
        vals.append(float(v))
        kept.append(c)
    return np.array(vals, dtype=np.float64), kept


def per_behavior_noise_floor(e0: dict, behavior: str, ctx_ids: list[str], *, seed=661, n=200):
    """Per-behavior test-retest ρ ceiling from probe-redraw halves (#658-style).

    Returns the 95th-pct ρ over n redraws, or 1.0 if E0 is constant across
    contexts (no rank signal — never a false PASS), or None if too few contexts.
    """
    col = BEHAVIOR_TO_COLUMN[behavior]
    table = e0.get("e0", {})
    per_ctx: dict[str, list[float]] = {}
    for c in ctx_ids:
        cell = table.get(c, {}).get(col)
        if cell is None:
            continue
        pp = cell.get("per_probe")
        if not pp:
            continue
        vals = [float(x["e0"]) for x in pp if x.get("e0") is not None]
        if vals:
            per_ctx[c] = vals
    if len(per_ctx) < 4:
        return None
    ctx_means = [float(np.mean(v)) for v in per_ctx.values()]
    if float(np.std(ctx_means)) < 1e-9:
        return 1.0
    rng = random.Random(seed)
    rhos = []
    for _ in range(n):
        a, b = [], []
        for c in ctx_ids:
            vals = per_ctx.get(c)
            if not vals or len(vals) < 2:
                continue
            shuf = vals[:]
            rng.shuffle(shuf)
            half = len(shuf) // 2
            a.append(float(np.mean(shuf[:half])))
            b.append(float(np.mean(shuf[half:])))
        if len(a) >= 4:
            r = _spearman(np.array(a), np.array(b))
            if r is not None:
                rhos.append(r)
    return float(np.percentile(rhos, 95)) if rhos else None


def m3_predictive(
    directions: dict,
    arm_b: dict,
    v0: dict,
    e0: dict,
    behaviors: list[str],
    selected_layer: dict,
    *,
    bootstrap_n: int,
    seed: int = 661,
) -> dict:
    """LOCO 1-D OLS ρ per (method, behavior) at the selected layer + noise floor.

    The projection scalar per context is ``r[selected_layer] · v0(C)[selected_layer]``.
    """
    # When M3 is genuinely active (real-dim v0 + a non-empty E0 table), a behavior
    # whose E0 column is ENTIRELY absent is NOT one of the registered §7/§6.4 branches
    # (E0-saturated / too-few-contexts) — it is a silent data gap that would let the
    # adopt-A ρ leg PASS vacuously via rho_gap→None. Fail loud on it (the reconciler's
    # Fix #3 second clause). M3-inactive (smoke / dim-mismatch, e0 == {}) is the
    # registered "skip M3" branch and does NOT trip this.
    m3_active = bool(v0) and bool(e0.get("e0"))
    col_present = (
        {b: any(BEHAVIOR_TO_COLUMN[b] in e0["e0"].get(c, {}) for c in v0) for b in behaviors}
        if m3_active
        else {}
    )
    if m3_active:
        absent_e0 = [b for b in behaviors if not col_present.get(b, False)]
        assert not absent_e0, (
            f"M3 active but E0 column entirely absent for {absent_e0} — a behavior "
            f"whose E0 should exist must not silently PASS adopt-A's ρ leg via "
            f"rho_gap→None (plan §6.4: P4 asserts E0 existence). E0 table contexts: "
            f"{sorted(e0['e0'].keys())[:5]}..."
        )
    out: dict = {}
    for behavior in behaviors:
        sl = selected_layer[behavior]
        y, kept = e0_rate_vector(e0, behavior, list(v0.keys()))
        rec: dict = {
            "selected_layer": sl,
            "n_contexts": len(kept),
            "noise_floor_p95": per_behavior_noise_floor(e0, behavior, kept),
            "m3_active": m3_active,
        }
        if len(kept) < 4 or np.std(y) < 1e-12:
            # Registered §7 branches: E0 saturated (zero variance across contexts)
            # OR too few contexts. Both make M3 ρ legitimately undefined → the
            # adopt-A ρ leg treats this behavior's gap as None (vacuous pass) by
            # design (verdict bases on M1+M2 only for the behavior).
            rec["e0_saturated"] = bool(len(kept) >= 4 and np.std(y) < 1e-12)
            rec["m3_undefined_reason"] = (
                "m3_inactive"
                if not m3_active
                else ("e0_saturated" if rec["e0_saturated"] else "too_few_contexts")
            )
            rec["methods"] = {}
            out[behavior] = rec
            logger.warning("%s: M3 undefined (n=%d, std(E0)=%.2e)", behavior, len(kept), np.std(y))
            continue
        d = directions[behavior]
        method_dirs = {"A": d["r_b_a"][sl], "C": d["r_b_c"][sl]}
        if behavior in arm_b:
            method_dirs["B"] = arm_b[behavior][sl]
        methods: dict = {}
        for m, r_layer in method_dirs.items():
            r = r_layer.numpy()
            proj = np.array([float(v0[c][sl].numpy() @ r) for c in kept], dtype=np.float64)
            preds = loco_1d_ols_predictions(proj, y)
            # Fixed per-method offset (NOT hash(m), which is PYTHONHASHSEED-randomized
            # → non-reproducible CI bounds run-to-run). {"A":0,"B":1,"C":2}.
            rng = random.Random(seed + _METHOD_SEED_OFFSET[m])
            boot = []
            nn = len(kept)
            for _ in range(bootstrap_n):
                idx = [rng.randrange(nn) for _ in range(nn)]
                rb_ = _spearman(preds[idx], y[idx])
                if rb_ is not None:
                    boot.append(rb_)
            ci = None
            if len(boot) >= 50:
                boot.sort()
                ci = [boot[int(0.025 * len(boot))], boot[int(0.975 * len(boot)) - 1]]
            methods[m] = {
                "rho_spearman": _spearman(preds, y),
                "pearson": _pearson(preds, y),
                "rho_ci95": ci,
            }
        rec["methods"] = methods
        out[behavior] = rec
    return out


def select_layers(arm_b: dict, v0: dict, e0: dict, behaviors: list[str]) -> dict:
    """Per-behavior selected layer = argmax arm-B LOCO ρ (fallback, plan §A3).

    #658's per-behavior lock is not yet stamped; the fallback is the layer
    maximizing arm B's own M3 LOCO ρ — store-computable today. A behavior with
    no arm-B / undefined ρ everywhere falls back to the middle layer (14).
    """
    sel: dict[str, int] = {}
    for behavior in behaviors:
        y, kept = e0_rate_vector(e0, behavior, list(v0.keys()))
        rB = arm_b.get(behavior)
        if rB is None or len(kept) < 4 or np.std(y) < 1e-12:
            sel[behavior] = 14
            continue
        best_layer, best_rho = 14, -2.0
        for li in range(rB.shape[0]):
            r = rB[li].numpy()
            proj = np.array([float(v0[c][li].numpy() @ r) for c in kept], dtype=np.float64)
            preds = loco_1d_ols_predictions(proj, y)
            rho = _spearman(preds, y)
            if rho is not None and rho > best_rho:
                best_rho, best_layer = rho, li
        sel[behavior] = best_layer
        logger.info("%s: selected layer %d (arm-B LOCO ρ=%.3f)", behavior, best_layer, best_rho)
    return sel


# ── Decision ──────────────────────────────────────────────────────────────────


def apply_kill_criterion(directions: dict, headline: list[str]) -> tuple[list[str], dict, dict]:
    """Plan §7 ``<MIN_SURVIVORS`` survivor kill criterion.

    For each headline behavior, read its judge-positive survivor counts (persisted
    by P3 into the per-behavior ``.pt`` via ``load_directions``). If
    ``min(n_pos, n_neg) < MIN_SURVIVORS`` the behavior's r_B^A/r_B^C are
    noise-dominated and cannot be estimated — DROP it from the verdict (plan §7:
    "STOP that behavior, report the survival shortfall as a finding, proceed with
    the remaining behaviors"). Returns:

    - ``kept``: headline behaviors with sufficient survivors (verdict basis);
    - ``n_survivors``: per-behavior ``{"n_pos", "n_neg"}`` (always reported as
      coverage, plan §6.5 / risk row);
    - ``dropped``: per-DROPPED-behavior ``{"n_pos", "n_neg", "reason"}``.
    """
    kept: list[str] = []
    n_survivors: dict[str, dict] = {}
    dropped: dict[str, dict] = {}
    for b in headline:
        d = directions.get(b)
        if d is None:
            # No extracted directions at all — treat as a hard drop (0 survivors).
            dropped[b] = {"n_pos": 0, "n_neg": 0, "reason": "no extracted directions"}
            n_survivors[b] = {"n_pos": 0, "n_neg": 0}
            continue
        n_pos = int(d["n_pos_survivors"])
        n_neg = int(d["n_neg_survivors"])
        n_survivors[b] = {"n_pos": n_pos, "n_neg": n_neg}
        if min(n_pos, n_neg) < MIN_SURVIVORS:
            dropped[b] = {
                "n_pos": n_pos,
                "n_neg": n_neg,
                "reason": f"min(n_pos={n_pos}, n_neg={n_neg}) < MIN_SURVIVORS={MIN_SURVIVORS}",
            }
        else:
            kept.append(b)
    return kept, n_survivors, dropped


def decide(m1: dict, m2: dict, m3: dict, headline: list[str], directions: dict) -> dict:
    """Plan §7 falsification verdict at the selected layer over the headline behaviors.

    Enforces the §7 ``<MIN_SURVIVORS`` kill criterion FIRST (via
    ``apply_kill_criterion``): low-survivor behaviors are excluded from the verdict
    and recorded under ``dropped_low_survivors``. If ALL headline behaviors drop,
    the verdict is ``halt_low_survivors`` (plan §7: "If <5 survivors on ALL 3
    behaviors, halt the whole run and report").

    The adopt-A ρ conjunct reads "A's ρ ≥ C's ρ within #658 noise floor" (plan §7),
    so the per-behavior ``noise_floor_p95`` (from M3) is the tolerance band:
    ``gap >= -noise_floor[b]`` (NOT ``gap >= -1e-9`` — that was strictly stricter
    than the registered band). Falls back to ``DEFAULT_NOISE_FLOOR`` when the
    per-behavior floor is None (a named plan deviation, surfaced in the body).
    """
    # §7 kill criterion — drop low-survivor behaviors BEFORE forming the verdict set.
    kept, n_survivors, dropped = apply_kill_criterion(directions, headline)

    # The verdict basis is the kept behaviors whose M1/M2 are actually present.
    present = [b for b in kept if b in m1 and b in m2]
    cos_ac = {b: m1[b]["cos_AC_selected"] for b in present}
    conf = {b: m2[b]["confound_A_selected"] for b in present}

    def rho_gap(b):
        ms = m3.get(b, {}).get("methods", {})
        ra = ms.get("A", {}).get("rho_spearman")
        rc = ms.get("C", {}).get("rho_spearman")
        if ra is None or rc is None:
            return None
        return ra - rc  # A - C; negative = A worse than C

    def noise_floor_for(b) -> tuple[float, bool]:
        """Per-behavior adopt-A ρ tolerance + whether the registered fallback was used."""
        nf = m3.get(b, {}).get("noise_floor_p95")
        if nf is None:
            return DEFAULT_NOISE_FLOOR, True
        return float(nf), False

    gaps = {b: rho_gap(b) for b in present}
    noise_floor = {}
    noise_floor_fallback_used = {}
    for b in present:
        nf, used_fallback = noise_floor_for(b)
        noise_floor[b] = nf
        noise_floor_fallback_used[b] = used_fallback

    # Adopt-A: cos≥0.95 AND confound≤0.10 AND (ρ_gap within the per-behavior noise
    # floor: gap >= -noise_floor[b]) — on ALL headline behaviors that survived.
    adopt_a = (
        len(present) == len(headline)
        and len(dropped) == 0
        and all(
            cos_ac[b] >= 0.95
            and conf[b] <= 0.10
            and (gaps[b] is None or gaps[b] >= -noise_floor[b])
            for b in present
        )
    )

    n_cos_fail = sum(1 for b in present if cos_ac[b] < 0.85)
    n_conf_fail = sum(1 for b in present if conf[b] > 0.25)
    n_rho_fail = sum(1 for b in present if gaps[b] is not None and gaps[b] <= -0.05)
    adopt_c = (n_cos_fail >= 2) or (n_conf_fail >= 2) or (n_rho_fail >= 2)

    if not present:
        # All headline behaviors dropped (low survivors / no directions) → §7 halt.
        verdict = "halt_low_survivors"
    elif adopt_a:
        verdict = "adopt_A"
    elif adopt_c:
        verdict = "adopt_C_or_recipe_by_behavior"
    else:
        verdict = "inconclusive_recipe_by_behavior"
    return {
        "verdict": verdict,
        "headline_behaviors": headline,
        "present_behaviors": present,
        "dropped_low_survivors": dropped,
        "n_survivors": n_survivors,
        "low_survivors": sorted(dropped.keys()),
        "cos_AC_selected": cos_ac,
        "confound_A_selected": conf,
        "rho_gap_A_minus_C": gaps,
        "adopt_A_noise_floor": noise_floor,
        "adopt_A_noise_floor_fallback_used": noise_floor_fallback_used,
        "margins": {
            "n_cos_AC_below_0.85": n_cos_fail,
            "n_confound_above_0.25": n_conf_fail,
            "n_rho_A_worse_by_0.05": n_rho_fail,
        },
    }


# ── Figures ───────────────────────────────────────────────────────────────────


def make_figures(m1: dict, m2: dict, m3: dict, behaviors: list[str], fig_dir: Path) -> list[str]:
    """Hero figures F1 (cosine), F2 (confound), F3 (LOCO ρ bars). Returns paths."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    # F1: per-layer cosine, one panel per behavior.
    fig, axes = plt.subplots(1, len(behaviors), figsize=(5 * len(behaviors), 4), squeeze=False)
    for ax, b in zip(axes[0], behaviors, strict=True):
        rec = m1[b]
        layers = list(range(len(rec["cos_AC"])))
        ax.plot(layers, rec["cos_AC"], label="cos(A,C)", color="#1f77b4")
        if "cos_AB" in rec:
            ax.plot(layers, rec["cos_AB"], label="cos(A,B)", color="#ff7f0e")
            ax.plot(layers, rec["cos_BC"], label="cos(B,C)", color="#2ca02c")
        # §6.5 survivor-set bootstrap CI for cos(A,C) at the selected layer (the
        # headline gate quantity) — an error bar at the selected-layer point. None
        # when the per-survivor stacks are absent (pre-round-2 .pt).
        sl_b = rec["selected_layer"]
        ci = rec.get("cos_AC_ci95")
        pt = rec["cos_AC_selected"]
        if ci is not None:
            ax.errorbar(
                sl_b, pt, yerr=[[max(0.0, pt - ci[0])], [max(0.0, ci[1] - pt)]],
                fmt="o", color="#1f77b4", capsize=4, label="cos(A,C) 95% CI",
            )  # fmt: skip
        ax.axvline(rec["selected_layer"], ls="--", color="gray", alpha=0.6)
        ax.set_title(b)
        ax.set_xlabel("layer")
        ax.set_ylabel("cosine")
        ax.set_ylim(-1.05, 1.05)
        ax.legend(fontsize=8)
    fig.suptitle("F1 — pairwise r_B cosine per layer")
    fig.tight_layout()
    p1 = fig_dir / "F1_cosine_per_layer.png"
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    paths.append(str(p1))

    # F2: per-layer confound, one panel per behavior, with B/C controls.
    fig, axes = plt.subplots(1, len(behaviors), figsize=(5 * len(behaviors), 4), squeeze=False)
    for ax, b in zip(axes[0], behaviors, strict=True):
        rec = m2[b]
        layers = list(range(len(rec["confound_A"])))
        ax.plot(layers, rec["confound_A"], label="A (r_B^A · ĉ_inst)", color="#d62728")
        ax.plot(layers, rec["confound_C_control"], label="C control", color="#1f77b4", alpha=0.7)
        if "confound_B_control" in rec:
            ax.plot(
                layers, rec["confound_B_control"], label="B control", color="#2ca02c", alpha=0.7
            )
        # §6.5 survivor-set bootstrap CI for the bounded confound (the §7 0.10/0.25
        # gate quantity) at the selected layer. None when per-survivor stacks absent.
        sl_b = rec["selected_layer"]
        ci = rec.get("confound_A_ci95")
        pt = rec["confound_A_selected"]
        if ci is not None:
            ax.errorbar(
                sl_b, pt, yerr=[[max(0.0, pt - ci[0])], [max(0.0, ci[1] - pt)]],
                fmt="o", color="#d62728", capsize=4, label="A confound 95% CI",
            )  # fmt: skip
        ax.axvline(rec["selected_layer"], ls="--", color="gray", alpha=0.6)
        ax.axhline(0.10, ls=":", color="green", alpha=0.5)
        ax.axhline(0.25, ls=":", color="red", alpha=0.5)
        ax.set_title(b)
        ax.set_xlabel("layer")
        ax.set_ylabel("|proj| / ‖r‖")
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=8)
    fig.suptitle("F2 — context-confound (projection onto ĉ_inst) per layer")
    fig.tight_layout()
    p2 = fig_dir / "F2_context_confound.png"
    fig.savefig(p2, dpi=150)
    plt.close(fig)
    paths.append(str(p2))

    # F3: LOCO ρ grouped bars (A/B/C) per behavior at selected layer + noise floor.
    fig, ax = plt.subplots(figsize=(2.5 * len(behaviors) + 2, 4))
    methods = ["A", "B", "C"]
    colors = {"A": "#d62728", "B": "#2ca02c", "C": "#1f77b4"}
    width = 0.25
    x = np.arange(len(behaviors))
    for mi, m in enumerate(methods):
        vals, errs = [], [[], []]
        for b in behaviors:
            ms = m3.get(b, {}).get("methods", {}).get(m, {})
            rho = ms.get("rho_spearman")
            ci = ms.get("rho_ci95")
            v = rho if rho is not None else 0.0
            vals.append(v)
            if ci is not None and rho is not None:
                errs[0].append(max(0.0, v - ci[0]))
                errs[1].append(max(0.0, ci[1] - v))
            else:
                errs[0].append(0.0)
                errs[1].append(0.0)
        ax.bar(x + (mi - 1) * width, vals, width, label=f"arm {m}", color=colors[m], yerr=errs)
    for bi, b in enumerate(behaviors):
        nf = m3.get(b, {}).get("noise_floor_p95")
        if nf is not None:
            ax.hlines(nf, bi - 0.4, bi + 0.4, color="black", ls="--", alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(behaviors)
    ax.set_ylabel("LOCO Spearman ρ (E0 ~ r·v0)")
    ax.set_title("F3 — held-out predictive ρ per method (dashed = #658 noise floor)")
    ax.legend()
    fig.tight_layout()
    p3 = fig_dir / "F3_loco_predictive_rho.png"
    fig.savefig(p3, dpi=150)
    plt.close(fig)
    paths.append(str(p3))
    return paths


# ── Direction loading ─────────────────────────────────────────────────────────


def load_directions(directions_dir: Path, behaviors: list[str]) -> dict:
    """Load r_B^A / r_B^C / c_pos / c_neg + the per-behavior judge-survivor counts.

    The survivor counts (``n_pos_survivors`` / ``n_neg_survivors``) are persisted
    by P3 (``issue661_extract_directions``) into EACH per-behavior ``.pt`` AND the
    ``extract_manifest.json``; they feed the §7 ``<MIN_SURVIVORS`` kill criterion in
    ``apply_kill_criterion``. They are REQUIRED — a P3 ``.pt`` that predates the
    count persistence (or a hand-edited one) fails loud here so a behavior is never
    silently kept under the kill gate (a degenerate read).

    A behavior P3 ALREADY dropped under the §7 kill criterion (``<MIN_SURVIVORS`` in
    either pool, including a 0-survivor pool that cannot be teacher-forced) writes a
    survivor-count-only checkpoint with ``dropped_low_survivors=True`` and NO
    direction tensors. Such records carry ``r_b_a=None`` here; the analysis kill
    criterion drops them from the verdict (via the same counts) and M1/M2/M3 never
    read their (absent) directions. ``record_is_extracted`` marks the difference.
    """
    out: dict = {}
    for behavior in behaviors:
        path = directions_dir / f"r_b_{behavior}.pt"
        blob = torch.load(path, weights_only=False)
        missing = [k for k in ("n_pos_survivors", "n_neg_survivors") if k not in blob]
        assert not missing, (
            f"{path} missing survivor-count keys {missing} — cannot enforce the §7 "
            f"<{MIN_SURVIVORS}-survivor kill criterion (re-run P3 to repersist)"
        )
        dropped = bool(blob.get("dropped_low_survivors", False)) or "r_b_a" not in blob
        rec: dict = {
            "n_pos_survivors": int(blob["n_pos_survivors"]),
            "n_neg_survivors": int(blob["n_neg_survivors"]),
            "record_is_extracted": not dropped,
        }
        if dropped:
            # Survivor-count-only checkpoint (P3 §7 drop). No direction tensors —
            # apply_kill_criterion drops it from the verdict before any tensor read.
            for k in ("r_b_a", "r_b_c", "c_pos", "c_neg"):
                rec[k] = None
            for k in (
                "present_A_items",
                "absent_A_items",
                "present_C_items",
                "absent_C_items",
                "c_pos_items",
                "c_neg_items",
            ):
                rec[k] = None
            out[behavior] = rec
            continue
        rec["r_b_a"] = blob["r_b_a"].float()
        rec["r_b_c"] = blob["r_b_c"].float()
        rec["c_pos"] = blob["c_pos"].float()
        rec["c_neg"] = blob["c_neg"].float()
        # Per-survivor (N, L, H) stacks for the §6.5 M1/M2 bootstrap CI. Present in
        # P3 outputs from round-2 on; a pre-round-2 .pt without them yields point
        # estimates only (CI=None) — never a crash. fp16 → float for the resample.
        for k in (
            "present_A_items",
            "absent_A_items",
            "present_C_items",
            "absent_C_items",
            "c_pos_items",
            "c_neg_items",
        ):
            rec[k] = blob[k].float() if k in blob else None
        out[behavior] = rec
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #661 P4+P5: r_B divergence analysis + figures.")
    ap.add_argument("--behaviors", nargs="+", default=["sycophancy", "refusal", "broad_em"])
    ap.add_argument("--directions-dir", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=None)
    ap.add_argument("--bootstrap-n", type=int, default=BOOTSTRAP_N)
    ap.add_argument("--max-contexts", type=int, default=0, help="cap v0 contexts (smoke)")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()

    out_dir = args.out_dir or EVAL_RESULTS_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    directions = load_directions(args.directions_dir, args.behaviors)
    # Behaviors P3 actually extracted a direction for. Behaviors P3 dropped under
    # the §7 kill criterion (<MIN_SURVIVORS / empty pool) carry r_b_a=None and are
    # excluded from M1/M2/M3 (no tensors to read); they still flow into decide()
    # via `directions` so apply_kill_criterion records them in dropped_low_survivors.
    extracted = [b for b in args.behaviors if directions[b].get("record_is_extracted", True)]
    dropped_p3 = [b for b in args.behaviors if b not in extracted]
    if dropped_p3:
        logger.warning(
            "behaviors dropped by P3 §7 kill criterion (no direction): %s — excluded from "
            "M1/M2/M3; recorded in decision.json dropped_low_survivors",
            dropped_p3,
        )
    arm_b = load_arm_b(args.behaviors)
    v0 = load_v0()
    if args.max_contexts:
        v0 = dict(list(v0.items())[: args.max_contexts])

    if not extracted:
        # All requested behaviors dropped at P3 → no directions to analyze. decide()
        # still produces the §7 halt_low_survivors verdict from the survivor counts.
        logger.warning("no extracted directions for any behavior — §7 halt_low_survivors")
        m1, m2, m3 = {}, {}, {}
        selected_layer = {}
    else:
        # Dim-compatibility guard (cross-phase data contract): arm B (#658 r_b.pt,
        # (28, 3584)) and v0 (#658 v0_summaries.pt, (28, 3584)) must share the
        # layer/hidden dims of the P3-extracted arm-A/C directions before they can be
        # combined. On the REAL run all are (28, 3584). On a tiny-model CPU smoke the
        # directions are (24, 896) — so arm-B reads (M1 cos AB/BC, M2 B-control) are
        # SKIPPED for the mismatched behaviors and M3 (needs the real-dim v0) is
        # skipped entirely. This is the smoke fix AND a real guard if #658 ever
        # re-extracts at a different layer count.
        dir_shape = directions[extracted[0]]["r_b_a"].shape  # (L, H)
        arm_b_compat = {b: rb for b, rb in arm_b.items() if tuple(rb.shape) == tuple(dir_shape)}
        dropped_b = sorted(set(arm_b) - set(arm_b_compat))
        if dropped_b:
            logger.warning(
                "arm-B dim mismatch (directions %s) — dropping arm-B for %s (M1 AB/BC + M2 "
                "B-control skipped for these)",
                tuple(dir_shape),
                dropped_b,
            )
        v0_dim_ok = bool(v0) and tuple(next(iter(v0.values())).shape) == tuple(dir_shape)
        if not v0_dim_ok:
            logger.warning(
                "v0 dim %s != direction dim %s — M3 (LOCO predictive ρ) skipped (needs real v0)",
                (tuple(next(iter(v0.values())).shape) if v0 else None),
                tuple(dir_shape),
            )
        arm_b = arm_b_compat

        # Only load/recompute E0 (the heavy ~140k Batch-judge path when
        # E0_expression.json is absent on HF) when M3 will actually run — a
        # dim-mismatched smoke skips M3, so E0 is unused.
        e0 = load_or_recompute_e0(out_dir, extracted) if v0_dim_ok else {"e0": {}, "columns": []}

        selected_layer = select_layers(arm_b, v0 if v0_dim_ok else {}, e0, extracted)
        m1 = m1_cosine(directions, arm_b, extracted, selected_layer, bootstrap_n=args.bootstrap_n)
        m2 = m2_confound(directions, arm_b, extracted, selected_layer, bootstrap_n=args.bootstrap_n)
        m3 = m3_predictive(
            directions,
            arm_b,
            v0 if v0_dim_ok else {},
            e0,
            extracted,
            selected_layer,
            bootstrap_n=args.bootstrap_n,
        )
    headline = [b for b in ("sycophancy", "refusal", "broad_em") if b in args.behaviors]
    decision = decide(m1, m2, m3, headline, directions)

    meta = reproducibility_metadata({"script": "issue661_analysis"})
    dump_json(
        {"behaviors": m1, "selected_layer": selected_layer, "metadata": meta},
        out_dir / "cosine_divergence.json",
    )
    dump_json(
        {"behaviors": m2, "selected_layer": selected_layer, "metadata": meta},
        out_dir / "context_confound.json",
    )
    dump_json(
        {"behaviors": m3, "selected_layer": selected_layer, "metadata": meta},
        out_dir / "a33_predictive.json",
    )
    dump_json(
        {**decision, "selected_layer": selected_layer, "metadata": meta}, out_dir / "decision.json"
    )

    fig_paths: list[str] = []
    # Figures cover only the EXTRACTED behaviors (dropped behaviors have no
    # m1/m2/m3 records). decision.json carries the dropped ones under
    # dropped_low_survivors, so they are never silently lost from the deliverable.
    if not args.no_figures and extracted:
        fig_paths = make_figures(m1, m2, m3, extracted, PROJECT_ROOT / "figures" / "issue_661")
    logger.info("DECISION: %s", decision["verdict"])
    logger.info("cos(A,C) selected: %s", decision["cos_AC_selected"])
    logger.info("confound_A selected: %s", decision["confound_A_selected"])
    logger.info("figures: %s", fig_paths)
    return 0


if __name__ == "__main__":
    sys.exit(main())
