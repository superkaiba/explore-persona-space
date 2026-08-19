#!/usr/bin/env python
"""Issue #2379 P7 — correlations + figures + Gate G1 (off-pod, VM CPU; no model calls).

Consumes (all produced by units 1-3 / the pod phases; committed JSONs + HF tensor stores):
  * ``eval_results/issue_2379/predictors/predictor_scores.json``   (P5.4 score table)
  * ``eval_results/issue_2379/predictors/map_diagnostics.json``    (P5 fit diagnostics)
  * ``eval_results/issue_2379/rates_em.json``                      (P6 judge wave)
  * ``eval_results/issue_2379/rates_caps/<model>.json``            (P2 per-model shards)
  * optional: ``kappa_control.json`` / ``pilot_gate.json`` (pass-through into the report)
  * gate / collinearity tensors: ``predictor_captures/<model>/{grid,mu}.pt`` and
    ``maps_pinned/{mapset}_L{pin}.pt`` (HF ``issue2379_reelicit/analysis_tensors/``; ``--fetch``)

Produces:
  * ``eval_results/issue_2379/rates_caps.json``       (merged shards + continuous companion)
  * ``eval_results/issue_2379/correlations.json``     (curves, pinned table, verdict lattice,
    H1 conjuncts, sensitivity, reliability, diagnostics)
  * ``eval_results/issue_2379/bootstrap_draws.json``  (per-draw x per-condition delta-rho)
  * ``figures/issue_2379/*``                          (hero bars, scatters, layer curves,
    delta-rho forest, map-quality panel, exploratory dump)

Registered statistics (plan section 3 / P7):
  * Spearman (Pearson twin) between each predictor family and the per-trigger behavior rate,
    within condition (n=18 EM / 20 caps triggers), then averaged across the 5 EM datasets /
    3 caps languages (the parent's aggregation).
  * Trigger-ALIGNED paired bootstrap, 2,000 draws, seed 20260819: per draw ONE trigger-index
    multiset per SETTING, applied identically to every condition in the setting and BOTH
    predictor arms; delta-rho = rho(ans_trainref_mapI) - rho(ctx_trainref) at the pinned
    layers. Independent per-condition draws are BANNED (shared-multiset assert enforces).
  * Verdict lattice per setting: Replication-failed / Answer-side outperforms /
    Context-side retains more signal / Comparable; zero-variance/NaN Spearman conditions are
    "replication non-estimable" (excluded from the pooled mean, counted), never Comparable.
  * H1 status from its own conjuncts (mean ctx rho >= 0.6 AND beats BGE, both settings),
    reported separately from the lattice cell; mid-band flag 0.4 <= rho < 0.6 recorded.

LAYER-INDEXING CONVENTION (matches the capture bundles / #779 pass-B): stored index ``i`` ==
decoder block ``i`` == ``output_hidden_states[i+1]`` (pre-final-norm at i=27). The parent's
pinned L16 (EM) / L27 (caps) therefore resolve to STORED indices 16 / 27; the resolution is
recorded in every output JSON.

Modes:
    (full)        uv run python scripts/issue2379_analysis.py
    --gate-caps   Gate G1: mean within-language Spearman of ctx Train Ref @ L27 vs merged
                  caps rates, from P2+P3 outputs only. PASS(>=0.4) exit 0 / FAIL exit 3.
    --smoke       End-to-end on producer-schema fixtures (run
                  ``issue2379_mapfit.py --phase smoke`` first; pass its tmp dir as
                  ``--fixtures-root``). Covers: full pipeline, aligned bootstrap incl. the
                  shared-multiset assert, verdict-lattice probes, NaN classification,
                  gate-caps PASS and FAIL branches, inverted-CI errorbar clamp, figure render.
    --list-phases Print the mode registry (one per line) and exit.

Harmful-advice-class completions never enter this script (it reads rates/scores JSONs and
activation tensors only); trigger LABELS are echoed, prompt/completion text is not.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# BEFORE any heavy import (torch/matplotlib/hf), so the shared-VM thread caps (#847)
# bind in-process (tests/test_shared_vm_thread_caps.py).
load_dotenv()

import matplotlib  # noqa: E402

matplotlib.use("Agg")  # headless VM; set before any pyplot figure is created

import numpy as np  # noqa: E402
import torch  # noqa: E402
from huggingface_hub import hf_hub_download  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

ISSUE = 2379
SLUG = "issue2379_reelicit"
DATA_REPO = "superkaiba1/explore-persona-space-data"

# HF staging prefixes for the off-pod inputs this script consumes (round-2
# offpod-artifact-handoff counterpart): caps shards from the sweep's
# HF_RATES_CAPS_PREFIX — imported from the PRODUCER so the two spellings can
# never drift (round-3 g1 Minor 8); predictor JSONs from mapfit's phase_upload
# prefix.
from issue2379_sweep import HF_RATES_CAPS_PREFIX  # noqa: E402

HF_PREDICTOR_JSON_PREFIX = f"{SLUG}/eval_json/predictors"

# Mode registry (module-level dict literal — the smoke-arch arm-registry source).
PHASES = {
    "full": "P7 correlations + figures + verdict lattice (default mode)",
    "gate-caps": "Gate G1 from P2+P3 outputs only (exit 0 PASS / 3 FAIL)",
    "smoke": "end-to-end on producer-schema fixtures from mapfit --phase smoke",
}

# Condition roster (pod driver stems; unit-2 contract). "base" appears in rate files only.
EM_STEMS = [
    "em_bad_medical_advice",
    "em_bad_legal_advice",
    "em_bad_security_advice",
    "em_turner_risky_financial",
    "em_turner_extreme_sports",
]
CAPS_STEMS = ["caps_french", "caps_german", "caps_spanish"]

# Parent's pinned layers on the STORED 0..27 axis (stored i == decoder block i ==
# output_hidden_states[i+1]; capture-script convention — see module docstring).
PIN_EM_DEFAULT = 16
PIN_CAPS_DEFAULT = 27
PRODUCTION_N_LAYERS = 28

BOOT_SEED = 20260819  # plan P7 registered seed
BOOT_DRAWS_DEFAULT = 2000
# rng spawn keys: [BOOT_SEED, 0]=EM primary, [BOOT_SEED, 1]=caps primary,
# [BOOT_SEED, 2, n_t]=leave-p_inoc-out sensitivity (per setting via its reduced n_t).

GATE_THRESHOLD = 0.4  # plan section 7 Gate G1
MANIP_RHO = 0.4  # verdict-lattice manipulation-check floor
H1_RHO = 0.6  # H1 conjunct floor
EM_INSTALL_EMPTY_MAX = 0.20  # empty-prompt EM rate ceiling (kill-criterion (b))
EM_INSTALL_FAIL_MODELS = 3  # >= this many of 5 failing = structural EM install failure
CAPS_INSTALL_FAIL_MODELS = 2  # >= this many of 3 failing = structural caps install failure
EM_EMPTY_LABEL = "empty"  # the parent's empty trigger label (sweep bank convention)
MIN_PAIRS = 3  # minimum finite pairs for a defined correlation

# Parent's published values (plan section 3; Kwon et al. Fig 1a/1b) — hero reference ticks.
PARENT_PUBLISHED = {
    "em": {"ctx_trainref": 0.89, "bge_cos": 0.67},
    "caps": {"ctx_trainref": 0.90, "bge_cos": -0.24},
}

VERDICT_ANSWER = "Answer-side outperforms"
VERDICT_CONTEXT = "Context-side retains more signal"
VERDICT_COMPARABLE = "Comparable"
VERDICT_REPL_FAILED = "Replication-failed"
VERDICT_NON_ESTIMABLE = "replication non-estimable (all conditions)"

FAMILY_LABELS = {
    "ctx_trainref": "Train Ref (ctx)",
    "ctx_sameq": "Same-Q Inoc (ctx)",
    "bge_cos": "BGE",
    "ans_trainref_mapI": "Train-Ref pred (map-I)",
    "ans_sameq_mapI": "Same-Q pred (map-I)",
    "ans_trainref_mapB": "Train-Ref pred (map-B)",
    "ans_sameq_mapB": "Same-Q pred (map-B)",
    "identbias_trainref": "Identity+bias (Train Ref)",
    "identbias_sameq": "Identity+bias (Same-Q)",
    "ceiling_trainref": "Actual-answer ceiling",
    "ceiling_sameq": "Ceiling (Same-Q)",
    "trait_proj_mapI": "Trait projection (map-I)",
    "tfidf_cos": "TF-IDF",
    "jaccard": "Jaccard",
    "seqmatcher": "SeqMatcher",
}
HERO_FAMILIES = [
    "ctx_trainref",
    "ctx_sameq",
    "bge_cos",
    "ans_trainref_mapI",
    "ans_sameq_mapI",
    "ceiling_trainref",
]

logger = logging.getLogger("issue2379_analysis")


# ---------------------------------------------------------------------------
# Small utilities
# ---------------------------------------------------------------------------
def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def _git_meta() -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    return as_metadata_dict(git_provenance(cwd=REPO_ROOT))


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _r6(v: float) -> float | None:
    return round(float(v), 6) if np.isfinite(v) else None


def _list_r6(a) -> list:
    return [_r6(v) for v in np.asarray(a, dtype=np.float64).ravel()]


def _nan_arr(rows, dtype=np.float64) -> np.ndarray:
    """None -> nan array conversion for JSON-round-tripped matrices."""
    return np.array([[np.nan if v is None else float(v) for v in row] for row in rows], dtype=dtype)


# ---------------------------------------------------------------------------
# Correlation helpers (vectorized over the leading axes; last axis = triggers)
# ---------------------------------------------------------------------------
def _corr_lastaxis(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Pearson correlation along the last axis; nan where either side is constant."""
    xm = x - x.mean(axis=-1, keepdims=True)
    ym = y - y.mean(axis=-1, keepdims=True)
    num = (xm * ym).sum(axis=-1)
    den = np.sqrt((xm**2).sum(axis=-1) * (ym**2).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = num / den
    return np.where(den > 0, r, np.nan)


def _rank_lastaxis(a: np.ndarray) -> np.ndarray:
    return rankdata(a, axis=-1, method="average")


def _pair_corr(x: np.ndarray, y: np.ndarray, *, spearman: bool) -> float:
    """Masked single-pair correlation (finite pairs only; >= MIN_PAIRS else nan)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < MIN_PAIRS:
        return float("nan")
    xv, yv = x[m], y[m]
    if spearman:
        xv, yv = _rank_lastaxis(xv), _rank_lastaxis(yv)
    return float(_corr_lastaxis(xv, yv))


def _curve_corrs(fam_mat: np.ndarray, y: np.ndarray) -> dict:
    """Per-layer Spearman + Pearson for one (n_l, n_t) family matrix vs rate vector y."""
    n_l = fam_mat.shape[0]
    sp = np.array([_pair_corr(fam_mat[ly], y, spearman=True) for ly in range(n_l)])
    pe = np.array([_pair_corr(fam_mat[ly], y, spearman=False) for ly in range(n_l)])
    return {"spearman": sp, "pearson": pe}


# ---------------------------------------------------------------------------
# Input loading / rate alignment
# ---------------------------------------------------------------------------
def load_caps_shards(shards_dir: Path) -> dict[str, dict]:
    shards = {}
    for p in sorted(Path(shards_dir).glob("*.json")):
        shards[p.stem] = _load_json(p)
    if not shards:
        raise FileNotFoundError(f"no caps-rate shards under {shards_dir}")
    return shards


def write_merged_caps(shards: dict[str, dict], out_path: Path, git: dict) -> None:
    merged = {
        "issue": ISSUE,
        "slug": SLUG,
        "generated_utc": _utcnow(),
        "git": git,
        "note": "merged P2 per-model shards; caps_rate primary, mean_uppercase_fraction "
        "continuous companion (plan P2)",
        "models": shards,
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, indent=2), encoding="utf-8")
    logger.info("wrote %s (%d models)", out_path, len(shards))


def _stage_caps_shards(caps_dir: Path, *, fetch: bool) -> None:
    """Stage the P2 caps-rate shards from HF when absent locally and --fetch is set
    (round-2 offpod-artifact-handoff: pod-side git sync is not a durable handoff)."""
    caps_dir = Path(caps_dir)
    if caps_dir.is_dir() and any(caps_dir.glob("*.json")):
        return
    if not fetch:
        return  # downstream loads fail loud with the staging hint
    from huggingface_hub import HfApi

    rels = hub.retry_transient(
        lambda: hub.list_hf_files_under_path(
            HfApi(), hub.DEFAULT_DATASET_REPO, HF_RATES_CAPS_PREFIX, repo_type="dataset"
        ),
        what=f"list {HF_RATES_CAPS_PREFIX}",
    )
    shard_rels = [r for r in rels if r.endswith(".json")]
    if not shard_rels:
        raise FileNotFoundError(
            f"--fetch: no caps shards under {DATA_REPO}:{HF_RATES_CAPS_PREFIX} — run the "
            "P2 sweep upload first"
        )
    caps_dir.mkdir(parents=True, exist_ok=True)
    for rel in shard_rels:
        got = hub.retry_transient(
            lambda rel=rel: hf_hub_download(DATA_REPO, rel, repo_type="dataset"),
            what=f"hf_hub_download {rel}",
        )
        shutil.copy2(got, caps_dir / Path(rel).name)
    logger.info("[stage] %d caps shards -> %s", len(shard_rels), caps_dir)


def _stage_predictor_jsons(scores_path: Path, diag_path: Path, *, fetch: bool) -> None:
    """Stage mapfit's predictor JSONs (scores + diagnostics) from its phase_upload
    prefix when absent locally and --fetch is set."""
    if not fetch:
        return
    for target in (Path(scores_path), Path(diag_path)):
        if target.exists():
            continue
        rel = f"{HF_PREDICTOR_JSON_PREFIX}/{target.name}"
        got = hub.retry_transient(
            lambda rel=rel: hf_hub_download(DATA_REPO, rel, repo_type="dataset"),
            what=f"hf_hub_download {rel}",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(got, target)
        logger.info("[stage] %s -> %s", rel, target)


def rates_for_condition(
    model: str, cond: dict, rates_em: dict | None, caps_shards: dict[str, dict] | None
) -> tuple[np.ndarray, str]:
    """Rate vector aligned to cond['trigger_labels']; fails loud on a missing label."""
    labels = cond["trigger_labels"]
    if cond["setting"] == "em":
        if rates_em is None:
            raise FileNotFoundError("rates_em.json required for EM conditions")
        table = rates_em["rates"].get(model)
        if table is None:
            raise KeyError(f"model {model!r} missing from rates_em.json rates")
        vals = []
        for lab in labels:
            if lab not in table:
                raise KeyError(f"trigger label {lab!r} missing from rates_em for {model!r}")
            v = table[lab].get("em_rate")
            vals.append(np.nan if v is None else float(v))
        return np.array(vals), "rates_em.json em_rate (judged)"
    if caps_shards is None or model not in caps_shards:
        raise KeyError(f"caps shard for {model!r} missing (rates_caps/<model>.json)")
    per = caps_shards[model]["per_trigger"]
    vals = []
    for lab in labels:
        if lab not in per:
            raise KeyError(f"trigger label {lab!r} missing from caps shard for {model!r}")
        v = per[lab].get("caps_rate")
        vals.append(np.nan if v is None else float(v))
    return np.array(vals), "rates_caps shard caps_rate (programmatic)"


def condition_matrices(cond: dict) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """(layered {fam: (n_l, n_t)}, text {fam: (n_t,)}) with None -> nan."""
    layered = {f: _nan_arr(m) for f, m in cond["families_layered"].items()}
    text = {
        f: np.array([np.nan if v is None else float(v) for v in vals], dtype=np.float64)
        for f, vals in cond.get("families_text", {}).items()
    }
    return layered, text


def resolve_pin(cond: dict, pins: dict[str, int]) -> int:
    pin = pins[cond["setting"]]
    if pin >= cond["n_layers"]:
        raise ValueError(
            f"pinned layer {pin} out of range for n_layers={cond['n_layers']} "
            f"(pass --pin-em/--pin-caps for non-production fixtures)"
        )
    return pin


# ---------------------------------------------------------------------------
# Registered trigger-aligned paired bootstrap
# ---------------------------------------------------------------------------
def _assert_shared_multiset(xr: np.ndarray, x: np.ndarray, idx: np.ndarray) -> None:
    """Registered invariant (plan P7): ONE trigger-index multiset per setting per draw,
    applied identically to every condition AND both predictor arms. Content-checked on
    3 deterministic draws; independent per-condition/per-arm draws would fail here."""
    n_draws = idx.shape[0]
    for d in (0, n_draws // 2, n_draws - 1):
        for arm in range(x.shape[0]):
            for c in range(x.shape[1]):
                if not np.array_equal(xr[arm, c, d], x[arm, c][idx[d]]):
                    raise AssertionError(
                        f"shared-multiset violation: arm={arm} cond={c} draw={d} "
                        "was not resampled with the setting-level index multiset"
                    )


def bootstrap_setting(a_ans: np.ndarray, a_ctx: np.ndarray, y: np.ndarray, idx: np.ndarray) -> dict:
    """Paired trigger-aligned bootstrap for one setting.

    a_ans/a_ctx/y: (C, n_t) pinned-layer predictor scores + rates (finite; non-estimable
    conditions are excluded upstream). idx: (D, n_t) setting-level index multisets.
    Returns per-draw delta (C, D) + pooled per-draw means + CI fields.
    """
    x = np.stack([a_ans, a_ctx])  # (2, C, n_t)
    xr = x[:, :, idx]  # (2, C, D, n_t) — one multiset per draw, both arms, all conds
    yr = y[:, idx]  # (C, D, n_t) — the SAME multiset for the rate side
    _assert_shared_multiset(xr, x, idx)
    yrank = _rank_lastaxis(yr)
    rho_ans = _corr_lastaxis(_rank_lastaxis(xr[0]), yrank)  # (C, D)
    rho_ctx = _corr_lastaxis(_rank_lastaxis(xr[1]), yrank)  # (C, D) — same yrank object
    delta = rho_ans - rho_ctx
    with np.errstate(invalid="ignore"):
        pooled = np.nanmean(delta, axis=0)  # (D,)
    finite = pooled[np.isfinite(pooled)]
    ci = np.nanpercentile(finite, [2.5, 97.5]) if finite.size else np.array([np.nan, np.nan])
    return {
        "delta": delta,
        "pooled_per_draw": pooled,
        "ci_lo": float(ci[0]),
        "ci_hi": float(ci[1]),
        "n_nan_draws_per_condition": np.sum(~np.isfinite(delta), axis=1).astype(int).tolist(),
        "n_finite_pooled_draws": int(finite.size),
        # rank-space audit fields (gotchas.md bootstrap-CI entry): strict tail fractions
        # around the registered constant anchor 0.
        "boot_frac_below0": float(np.mean(finite < 0.0)) if finite.size else float("nan"),
        "boot_frac_above0": float(np.mean(finite > 0.0)) if finite.size else float("nan"),
        "idx_sha256": hashlib.sha256(np.ascontiguousarray(idx).tobytes()).hexdigest(),
    }


# ---------------------------------------------------------------------------
# Install checks + verdict lattice
# ---------------------------------------------------------------------------
def em_install_check(rates_em: dict, stems: list[str]) -> dict:
    """A None/non-finite empty-prompt rate is INDETERMINATE and counts toward the
    structural predicate (fail-closed) — an unmeasurable install is never a pass."""
    per_model, failing, indeterminate = {}, 0, 0
    for m in stems:
        table = rates_em["rates"].get(m)
        if table is None or EM_EMPTY_LABEL not in table:
            raise KeyError(f"EM install check: {m!r} lacks an {EM_EMPTY_LABEL!r} trigger row")
        rate = table[EM_EMPTY_LABEL].get("em_rate")
        if rate is None or not np.isfinite(rate):
            per_model[m] = {"empty_prompt_em_rate": rate, "fail": None, "indeterminate": True}
            indeterminate += 1
            failing += 1  # fail-closed: no measurable empty-prompt rate
            continue
        fail = rate > EM_INSTALL_EMPTY_MAX
        per_model[m] = {"empty_prompt_em_rate": rate, "fail": bool(fail), "indeterminate": False}
        failing += int(fail)
    return {
        "predicate": f"empty-prompt EM rate > {EM_INSTALL_EMPTY_MAX} in >= "
        f"{EM_INSTALL_FAIL_MODELS}/{len(stems)} models (indeterminate rates count as "
        "failing — fail-closed)",
        "per_model": per_model,
        "n_failing": failing,
        "n_indeterminate": indeterminate,
        "structural_fail": failing >= EM_INSTALL_FAIL_MODELS,
    }


def caps_install_check(caps_shards: dict[str, dict], stems: list[str]) -> dict:
    """Same fail-closed convention as em_install_check for a non-bool 'pass' field."""
    per_model, failing, indeterminate = {}, 0, 0
    for m in stems:
        shard = caps_shards.get(m)
        if shard is None or "install_check" not in shard:
            raise KeyError(f"caps install check: shard/install_check missing for {m!r}")
        ic = shard["install_check"]
        if not isinstance(ic.get("pass"), bool):
            per_model[m] = {**ic, "fail": None, "indeterminate": True}
            indeterminate += 1
            failing += 1  # fail-closed: install verdict unmeasurable
            continue
        fail = not ic["pass"]
        per_model[m] = {**ic, "fail": fail, "indeterminate": False}
        failing += int(fail)
    return {
        "predicate": f"P1.6 predicate fails in >= {CAPS_INSTALL_FAIL_MODELS}/{len(stems)} "
        "models (p_inoc caps rate < 50% OR empty-prompt caps rate > 20%; indeterminate "
        "verdicts count as failing — fail-closed)",
        "per_model": per_model,
        "n_failing": failing,
        "n_indeterminate": indeterminate,
        "structural_fail": failing >= CAPS_INSTALL_FAIL_MODELS,
    }


def verdict_for_setting(
    mean_ctx_rho: float,
    install_structural_fail: bool,
    delta_mean: float,
    ci_lo: float,
    ci_hi: float,
    n_ctx_estimable: int,
    n_joint_estimable: int,
) -> str:
    """Registered lattice (plan section 3): DISJOINT and exhaustive per setting.

    Precedence (round-2 fix): a STRUCTURAL install failure dominates
    estimability — an uninstalled behavior reads Replication-failed, never
    Non-estimable. Estimability is SPLIT (plan P7): the manipulation check is
    evaluated over CTX-estimable conditions; the paired delta-rho over
    JOINTLY-estimable ones — a NaN-answer-but-valid-context condition still
    counts toward the context-replication read."""
    if install_structural_fail:
        return VERDICT_REPL_FAILED
    if n_ctx_estimable == 0:
        return VERDICT_NON_ESTIMABLE
    manip_pass = np.isfinite(mean_ctx_rho) and mean_ctx_rho >= MANIP_RHO
    if not manip_pass:
        return VERDICT_REPL_FAILED
    if n_joint_estimable == 0:
        return VERDICT_NON_ESTIMABLE
    if np.isfinite(delta_mean) and np.isfinite(ci_lo) and delta_mean > 0 and ci_lo > 0:
        return VERDICT_ANSWER
    if np.isfinite(delta_mean) and np.isfinite(ci_hi) and delta_mean < 0 and ci_hi < 0:
        return VERDICT_CONTEXT
    return VERDICT_COMPARABLE


# ---------------------------------------------------------------------------
# Split-rollout ceiling reliability (plan section 3 reliability condition)
# ---------------------------------------------------------------------------
def ceiling_reliability(cond: dict) -> dict:
    """Per form (trainref/sameq): mean over the 3 one-vs-rest rollout splits of the
    Spearman across triggers between subset ceiling scores; per-layer curve."""
    out = {}
    for form, arr in cond.get("ceiling_by_rollout", {}).items():
        a = np.array(
            [[[np.nan if v is None else float(v) for v in r] for r in layer] for layer in arr],
            dtype=np.float64,
        )  # (n_l, n_t, n_roll)
        n_l, _, n_roll = a.shape
        if n_roll < 2:
            out[form] = {"curve": [None] * n_l, "note": "fewer than 2 rollouts"}
            continue
        curves = np.full((n_roll, n_l), np.nan)
        for i in range(n_roll):
            rest = [j for j in range(n_roll) if j != i]
            s1 = a[:, :, i]
            with np.errstate(invalid="ignore"):
                s2 = np.nanmean(a[:, :, rest], axis=2)
            for ly in range(n_l):
                curves[i, ly] = _pair_corr(s1[ly], s2[ly], spearman=True)
        with np.errstate(invalid="ignore"):
            mean_curve = np.nanmean(curves, axis=0)
        out[form] = {"curve": _list_r6(mean_curve), "n_rollouts": int(n_roll)}
    return out


# ---------------------------------------------------------------------------
# Figures (paper-plots conventions: no canvas caption blocks; one color = one meaning)
# ---------------------------------------------------------------------------
def _family_colors() -> dict[str, str]:
    from explore_persona_space.analysis.paper_plots import paper_palette, paper_palette_role

    pal = paper_palette(8)
    return {
        "ans_trainref_mapI": paper_palette_role("primary"),
        "ctx_trainref": paper_palette_role("baseline"),
        "ceiling_trainref": paper_palette_role("control"),
        "bge_cos": paper_palette_role("neutral"),
        "ctx_sameq": pal[3],
        "ans_sameq_mapI": pal[4],
        "ans_trainref_mapB": pal[5],
        "ans_sameq_mapB": pal[6],
        "identbias_trainref": pal[7],
        "identbias_sameq": pal[7],
        "ceiling_sameq": paper_palette_role("control"),
        "trait_proj_mapI": pal[3],
        "tfidf_cos": paper_palette_role("neutral"),
        "jaccard": paper_palette_role("neutral"),
        "seqmatcher": paper_palette_role("neutral"),
    }


def _assert_fig_populated(fig, name: str) -> None:
    """Fail loud on an empty/NaN-only render (always-on; #1112 empty-figure incident)."""
    import matplotlib.pyplot as plt  # noqa: F401  (backend already forced in main)

    for ax in fig.axes:
        if ax.get_label() == "<colorbar>":
            continue
        vals: list[float] = []
        for ln in ax.lines:
            # Reference artists (axhline/axvline) live on BLENDED transforms, not
            # transData — counting them as data would let an otherwise-empty panel
            # pass on its y=0 reference line (round-2 g4 fix).
            if ln.get_transform() is not ax.transData:
                continue
            vals.extend(np.asarray(ln.get_ydata(), dtype=np.float64).ravel().tolist())
        for coll in ax.collections:
            off = np.asarray(getattr(coll, "get_offsets", lambda: [])(), dtype=np.float64)
            if off.size:
                vals.extend(off.ravel().tolist())
            arr = getattr(coll, "get_array", lambda: None)()
            if arr is not None:
                vals.extend(np.asarray(arr, dtype=np.float64).ravel().tolist())
        for p in ax.patches:
            vals.append(float(p.get_height()) if hasattr(p, "get_height") else np.nan)
        for im in ax.images:
            arr = im.get_array()
            if arr is not None:
                vals.extend(np.asarray(arr, dtype=np.float64).ravel().tolist())
        if not vals or not np.isfinite(np.asarray(vals)).any():
            raise AssertionError(f"figure {name}: axes {ax.get_title()!r} has no finite data")


def _save(fig, stem: str, figdir: Path) -> None:
    from explore_persona_space.analysis.paper_plots import savefig_paper

    import matplotlib.pyplot as plt

    _assert_fig_populated(fig, stem)
    savefig_paper(fig, stem, dir=figdir)
    plt.close(fig)


def fig_hero(results: dict, figdir: Path) -> None:
    import matplotlib.pyplot as plt

    colors = _family_colors()
    settings = [s for s in ("em", "caps") if results["setting_means"].get(s)]
    fig, axes = plt.subplots(1, len(settings), figsize=(4.2 * len(settings), 3.2))
    axes = np.atleast_1d(axes)
    for ax, setting in zip(axes, settings):
        pins = results["pins"][setting]
        fams = [f for f in HERO_FAMILIES if f in results["pinned_table"][setting]]
        xs = np.arange(len(fams))
        any_pearson = False
        for i, fam in enumerate(fams):
            mean_v = results["pinned_table"][setting][fam]["mean_spearman"]
            per_cond = results["pinned_table"][setting][fam]["per_condition"]
            ax.bar(i, np.nan if mean_v is None else mean_v, color=colors[fam], width=0.7)
            dots = [v for v in per_cond.values() if v is not None]
            if dots:
                jit = np.linspace(-0.18, 0.18, len(dots))
                ax.scatter(i + jit, dots, s=12, color="black", alpha=0.55, zorder=3)
            mean_pe = results["pinned_table"][setting][fam].get("mean_pearson")
            if mean_pe is not None:
                ax.scatter(
                    [i],
                    [mean_pe],
                    marker="D",
                    s=20,
                    facecolors="none",
                    edgecolors="#333333",
                    linewidths=1.1,
                    zorder=4,
                )
                any_pearson = True
            ref = PARENT_PUBLISHED.get(setting, {}).get(fam)
            if ref is not None:
                ax.hlines(ref, i - 0.38, i + 0.38, color="#5A5A5A", ls="--", lw=1.2)
        ax.axhline(0.0, color="#5A5A5A", lw=0.8)
        if any_pearson:
            from matplotlib.lines import Line2D

            ax.legend(
                handles=[
                    Line2D(
                        [], [], marker="D", mfc="none", mec="#333333", ls="", label="mean Pearson"
                    )
                ],
                fontsize=6,
                frameon=False,
            )
        ax.set_xticks(xs)
        ax.set_xticklabels([FAMILY_LABELS[f] for f in fams], rotation=30, ha="right")
        ax.set_ylabel("mean within-condition Spearman rho")
        ax.set_title(f"{setting.upper()} @ stored layer {pins} (dashed = parent)")
    _save(fig, "fig1_hero_predictor_bars", figdir)


def fig_scatters(results: dict, per_condition_xy: dict, figdir: Path) -> None:
    import matplotlib.pyplot as plt

    colors = _family_colors()
    fams = ["ctx_trainref", "ans_trainref_mapI"]
    settings = [s for s in ("em", "caps") if results["setting_means"].get(s)]
    fig, axes = plt.subplots(
        len(settings), len(fams), figsize=(8.0, 3.0 * len(settings)), squeeze=False
    )
    for r, setting in enumerate(settings):
        for c, fam in enumerate(fams):
            ax = axes[r][c]
            for _, xy in per_condition_xy.items():
                if xy["setting"] != setting or fam not in xy:
                    continue
                ax.scatter(xy[fam], xy["rate"], s=14, alpha=0.6, color=colors[fam])
            ax.set_xlabel(f"{FAMILY_LABELS[fam]} @ pin")
            ax.set_ylabel(f"{setting.upper()} rate")
            ax.set_title(f"{setting.upper()} — {FAMILY_LABELS[fam]}")
    _save(fig, "fig2_rate_vs_predictor_scatter", figdir)


def fig_layer_curves(results: dict, figdir: Path) -> None:
    import matplotlib.pyplot as plt

    colors = _family_colors()
    fams = ["ctx_trainref", "ans_trainref_mapI", "ceiling_trainref"]

    def _has_finite(setting: str) -> bool:
        for fam in fams:
            curve = results["setting_means"][setting].get(fam, {}).get("spearman_curve")
            if curve is not None:
                ys = np.array([np.nan if v is None else v for v in curve], dtype=np.float64)
                if np.isfinite(ys).any():
                    return True
        return False

    settings = [s for s in ("em", "caps") if results["setting_means"].get(s)]
    dropped = [s for s in settings if not _has_finite(s)]
    if dropped:
        # A fully non-estimable setting (e.g. every rate zero-variance) is a
        # VERDICT (Non-estimable), not a render: omit the panel rather than
        # ship an empty axes (after-every-experiment item 8: omit or label
        # N/A — never a misleading empty/zero render).
        logger.warning("fig3: non-estimable setting(s) %s — panel(s) omitted", dropped)
        settings = [s for s in settings if s not in dropped]
    if not settings:
        logger.warning("fig3: every setting non-estimable — figure skipped")
        return
    fig, axes = plt.subplots(1, len(settings), figsize=(4.4 * len(settings), 3.0))
    axes = np.atleast_1d(axes)
    for ax, setting in zip(axes, settings):
        for fam in fams:
            curve = results["setting_means"][setting].get(fam, {}).get("spearman_curve")
            if curve is None:
                continue
            ys = np.array([np.nan if v is None else v for v in curve])
            ax.plot(np.arange(ys.size), ys, color=colors[fam], label=FAMILY_LABELS[fam])
        ax.axvline(results["pins"][setting], color="#5A5A5A", ls=":", lw=1.0)
        ax.axhline(0.0, color="#5A5A5A", lw=0.8)
        ax.set_xlabel("stored layer index (block i == hidden_states[i+1])")
        ax.set_ylabel("mean Spearman rho")
        ax.set_title(setting.upper())
        ax.legend(fontsize=7)
    _save(fig, "fig3_layer_curves", figdir)


def fig_forest(forest_rows: list[dict], stem: str, title: str, figdir: Path) -> None:
    """Delta-rho forest. CI offsets are CLAMPED non-negative (gotchas.md xerr entry:
    quantile CIs can invert around a separately-computed point at tiny n)."""
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette_role

    fig, ax = plt.subplots(figsize=(5.0, 0.5 * len(forest_rows) + 1.4))
    ys = np.arange(len(forest_rows))[::-1]
    for y, row in zip(ys, forest_rows):
        v, lo, hi = row["delta"], row["ci_lo"], row["ci_hi"]
        if v is None or not np.isfinite(v):
            continue
        err = None
        if lo is not None and hi is not None and np.isfinite(lo) and np.isfinite(hi):
            err = [[max(0.0, v - lo)], [max(0.0, hi - v)]]
        color = paper_palette_role("accent" if row.get("pooled") else "primary")
        ax.errorbar(
            v,
            y,
            xerr=err,
            fmt="o",
            color=color,
            capsize=3,
            markersize=6 if row.get("pooled") else 4,
        )
    ax.axvline(0.0, color="#5A5A5A", lw=0.8)
    ax.set_yticks(ys)
    ax.set_yticklabels([r["name"] for r in forest_rows], fontsize=7)
    ax.set_xlabel("delta rho (answer-side Train-Ref map-I minus context-side Train Ref)")
    ax.set_title(title)
    _save(fig, stem, figdir)


def fig_map_quality(diag: dict, figdir: Path) -> None:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette_role

    base_sets = [m for m in diag["diagnostics"] if m.startswith("base")]
    inoc_sets = [m for m in diag["diagnostics"] if m not in base_sets]
    fig, axes = plt.subplots(1, 3, figsize=(11.0, 3.0))
    panels = [
        ("held-out map R^2", lambda c: c["map"]["r2"]),
        ("identity+bias R^2", lambda c: c["identity_bias"]["r2"]),
        ("kNN acc@10 (euclidean)", lambda c: c["knn"]["euclidean"]["acc_at_k"]["10"]),
    ]
    for ax, (label, getter) in zip(axes, panels):
        for group, color, lw, alpha in (
            (inoc_sets, paper_palette_role("primary"), 1.0, 0.6),
            (base_sets, paper_palette_role("accent"), 1.8, 1.0),
        ):
            for ms in group:
                cells = diag["diagnostics"][ms]
                lys = sorted(int(k) for k in cells)
                ys = [getter(cells[str(ly)]) for ly in lys]
                ax.plot(lys, ys, color=color, lw=lw, alpha=alpha)
        ax.set_xlabel("stored layer index")
        ax.set_title(label)
    axes[0].set_ylabel("value (base = accent, inoculated = primary)")
    _save(fig, "fig5_map_quality", figdir)


def fig_exploratory_bars(results: dict, figdir: Path) -> None:
    import matplotlib.pyplot as plt

    colors = _family_colors()
    settings = [s for s in ("em", "caps") if results["setting_means"].get(s)]
    fams = [
        "ans_trainref_mapB",
        "ans_sameq_mapB",
        "identbias_trainref",
        "identbias_sameq",
        "trait_proj_mapI",
        "tfidf_cos",
        "jaccard",
        "seqmatcher",
    ]

    def _has_finite(setting: str) -> bool:
        return any(
            results["pinned_table"][setting][f].get("mean_spearman") is not None
            and np.isfinite(results["pinned_table"][setting][f]["mean_spearman"])
            for f in fams
            if f in results["pinned_table"][setting]
        )

    dropped = [s for s in settings if not _has_finite(s)]
    if dropped:
        # Same non-estimable treatment as fig3: omit the panel, never render
        # an all-NaN bar row (after-every-experiment item 8).
        logger.warning("fig6: non-estimable setting(s) %s — panel(s) omitted", dropped)
        settings = [s for s in settings if s not in dropped]
    if not settings:
        logger.warning("fig6: every setting non-estimable — figure skipped")
        return
    fig, axes = plt.subplots(1, len(settings), figsize=(4.8 * len(settings), 3.2))
    axes = np.atleast_1d(axes)
    for ax, setting in zip(axes, settings):
        present = [f for f in fams if f in results["pinned_table"][setting]]
        for i, fam in enumerate(present):
            v = results["pinned_table"][setting][fam]["mean_spearman"]
            ax.bar(i, np.nan if v is None else v, color=colors.get(fam, "#888888"), width=0.7)
        ax.axhline(0.0, color="#5A5A5A", lw=0.8)
        ax.set_xticks(np.arange(len(present)))
        ax.set_xticklabels([FAMILY_LABELS[f] for f in present], rotation=30, ha="right")
        ax.set_ylabel("mean Spearman rho @ pin")
        ax.set_title(f"{setting.upper()} — exploratory arms")
    _save(fig, "fig6_exploratory_arms", figdir)


def fig_interpredictor(results: dict, figdir: Path) -> None:
    import matplotlib.pyplot as plt

    for setting, block in results["diagnostics"]["inter_predictor_corr"].items():
        fams, mat = block["families"], _nan_arr(block["matrix"])
        # constrained layout + colorbar; NEVER tight_layout after a colorbar under the
        # paper style (matplotlib refuses the layout-engine switch).
        fig, ax = plt.subplots(figsize=(6.4, 5.6), layout="constrained")
        im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
        ax.set_xticks(range(len(fams)))
        ax.set_xticklabels([FAMILY_LABELS.get(f, f) for f in fams], rotation=90, fontsize=6)
        ax.set_yticks(range(len(fams)))
        ax.set_yticklabels([FAMILY_LABELS.get(f, f) for f in fams], fontsize=6)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax.set_title(f"{setting.upper()} inter-predictor Spearman @ pin")
        _save(fig, f"fig7_interpredictor_corr_{setting}", figdir)


def fig_base_sweep(rates_em: dict | None, caps_shards: dict | None, figdir: Path) -> None:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette_role

    panels = []
    if rates_em is not None and "base" in rates_em.get("rates", {}):
        t = rates_em["rates"]["base"]
        panels.append(("EM (base model)", [(k, v.get("em_rate")) for k, v in t.items()]))
    if caps_shards is not None and "base" in caps_shards:
        t = caps_shards["base"]["per_trigger"]
        panels.append(("caps (base model)", [(k, v.get("caps_rate")) for k, v in t.items()]))
    if not panels:
        logger.warning("base-sweep figure skipped: no base rows in rates_em / caps shards")
        return
    fig, axes = plt.subplots(1, len(panels), figsize=(5.5 * len(panels), 3.2))
    axes = np.atleast_1d(axes)
    for ax, (title, rows) in zip(axes, panels):
        labs = [r[0] for r in rows]
        vals = [np.nan if r[1] is None else r[1] for r in rows]
        ax.bar(np.arange(len(labs)), vals, color=paper_palette_role("neutral"))
        ax.set_xticks(np.arange(len(labs)))
        ax.set_xticklabels(labs, rotation=90, fontsize=5)
        ax.set_ylabel("rate")
        ax.set_title(title)
    _save(fig, "fig8_base_model_sweep", figdir)


def fig_reliability(results: dict, figdir: Path) -> None:
    import matplotlib.pyplot as plt

    colors = _family_colors()
    rows = [
        (m, results["conditions"][m]["reliability_at_pin"].get("trainref"))
        for m in sorted(results["conditions"])
    ]
    rows = [(m, v) for m, v in rows if v is not None]
    if not rows:
        logger.warning("reliability figure skipped: no finite reliability values")
        return
    fig, ax = plt.subplots(figsize=(5.5, 3.0))
    ax.bar(np.arange(len(rows)), [v for _, v in rows], color=colors["ceiling_trainref"], width=0.7)
    ax.set_xticks(np.arange(len(rows)))
    ax.set_xticklabels([m for m, _ in rows], rotation=30, ha="right", fontsize=7)
    ax.set_ylabel("split-rollout Spearman @ pin")
    ax.set_title("Ceiling split-rollout reliability (Train Ref form)")
    _save(fig, "fig9_ceiling_reliability", figdir)


def fig_gate(gate: dict, figdir: Path) -> None:
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import paper_palette

    langs = sorted(gate["per_language"])
    pal = paper_palette(max(3, len(langs)))
    fig, ax = plt.subplots(figsize=(5.0, 3.0))
    for i, m in enumerate(langs):
        curve = np.array([np.nan if v is None else v for v in gate["per_language"][m]["curve"]])
        ax.plot(np.arange(curve.size), curve, color=pal[i], label=m)
    ax.axvline(gate["pin"], color="#5A5A5A", ls=":", lw=1.0)
    ax.axhline(gate["threshold"], color="#5A5A5A", ls="--", lw=1.0)
    ax.set_xlabel("stored layer index")
    ax.set_ylabel("Spearman rho (ctx Train Ref vs caps rate)")
    mr = gate.get("mean_rho")
    mr_str = f"{mr:.3f}" if isinstance(mr, int | float) and np.isfinite(mr) else "non-estimable"
    ax.set_title(f"Gate G1 — mean rho @ pin = {mr_str}")
    ax.legend(fontsize=7)
    _save(fig, "gate_g1_curves", figdir)


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------
def inter_predictor_matrix(cond_data: dict, pins: dict[str, int]) -> dict:
    out = {}
    for setting in ("em", "caps"):
        conds = [c for c in cond_data.values() if c["setting"] == setting]
        if not conds:
            continue
        fam_names = sorted({f for c in conds for f in list(c["layered"]) + list(c["text"])})
        n = len(fam_names)
        acc = np.zeros((n, n)), np.zeros((n, n))
        for c in conds:
            pin = pins[setting]
            vecs = {}
            for f, m in c["layered"].items():
                vecs[f] = m[pin]
            vecs.update(c["text"])
            for i, fi in enumerate(fam_names):
                for j, fj in enumerate(fam_names):
                    if fi in vecs and fj in vecs:
                        r = _pair_corr(vecs[fi], vecs[fj], spearman=True)
                        if np.isfinite(r):
                            acc[0][i, j] += r
                            acc[1][i, j] += 1
        with np.errstate(invalid="ignore"):
            mat = acc[0] / acc[1]
        out[setting] = {
            "families": fam_names,
            "matrix": [[_r6(v) for v in row] for row in mat],
            "note": "mean across conditions of Spearman over triggers @ pinned layer",
        }
    return out


def drop_rate_vs_rate(rates_em: dict, stems: list[str]) -> dict:
    """Judge-censoring diagnostic: content-drop rate vs EM rate across triggers."""
    out = {}
    for m in stems:
        table = rates_em["rates"].get(m, {})
        drops, apirefs, rates = [], [], []
        for rec in table.values():
            content = (
                rec.get("drop_refusal_content", 0)
                + rec.get("drop_code", 0)
                + rec.get("drop_malformed", 0)
            )
            denom = rec.get("n_scored", 0) + content
            drops.append(content / denom if denom else np.nan)
            napi = rec.get("n_api_refusal", 0)
            apidenom = rec.get("n_scored", 0) + content + napi
            apirefs.append(napi / apidenom if apidenom else np.nan)
            r = rec.get("em_rate")
            rates.append(np.nan if r is None else r)
        drops_a, apirefs_a, rates_a = map(np.array, (drops, apirefs, rates))
        out[m] = {
            "rho_content_drop_vs_em_rate": _r6(_pair_corr(drops_a, rates_a, spearman=True)),
            "rho_api_refusal_vs_em_rate": _r6(_pair_corr(apirefs_a, rates_a, spearman=True)),
        }
    return out


def _load_map_components(maps_dir: Path, comp_dir: Path | None, mapset: str, pin: int):
    """Pinned .pt (production, HF maps_pinned) or components .npz (mapfit layout)."""
    pt = Path(maps_dir) / f"{mapset}_L{pin:02d}.pt"
    if pt.exists():
        # Unit-3 pinned bundles are tensors + primitive containers -> weights_only=True
        # (sibling precedent: issue2379_mapfit._fetch / judge loaders).
        z = torch.load(pt, map_location="cpu", weights_only=True)
        return {k: np.asarray(z[k], dtype=np.float64) for k in ("W", "xmu", "xsd", "ymu")}
    if comp_dir is not None:
        npz = Path(comp_dir) / f"{mapset}_L{pin:02d}.npz"
        if npz.exists():
            with np.load(npz) as z:
                return {k: np.asarray(z[k], dtype=np.float64) for k in ("W", "xmu", "xsd", "ymu")}
    return None


def reference_collinearity(
    cond_data: dict,
    pins: dict[str, int],
    captures_dir: Path,
    maps_dir: Path,
    comp_dir: Path | None,
    *,
    fetch: bool = False,
    stage_dir: Path | None = None,
) -> dict:
    """cos(mu_A_train, affine-mapped mu_train) per condition at its pin (exploratory).

    Round-2 g6 fix: --fetch is now THREADED here (the old skip message advertised
    --fetch while this function never fetched — a dead-end hint)."""
    out = {}
    for model, c in cond_data.items():
        pin = pins[c["setting"]]
        try:
            mu_path = _resolve_capture(
                Path(captures_dir), model, "mu", fetch, Path(stage_dir or captures_dir)
            )
        except FileNotFoundError as e:
            out[model] = {"skipped": str(e)}
            continue
        comps = _load_map_components(maps_dir, comp_dir, model, pin)
        if comps is None:
            out[model] = {
                "skipped": f"map components for {model} L{pin:02d} absent under {maps_dir}"
            }
            continue
        z = torch.load(mu_path, map_location="cpu", weights_only=True)  # unit-2 bundle
        mu_c = np.asarray(z["mu_train"], dtype=np.float64)[pin]
        mu_a = z.get("mu_a_train")
        if mu_a is None:
            out[model] = {"skipped": "mu_a_train absent from mu bundle"}
            continue
        mu_a = np.asarray(mu_a, dtype=np.float64)[pin]
        mapped = ((mu_c - comps["xmu"]) / comps["xsd"]) @ comps["W"] + comps["ymu"]
        cos = float(mapped @ mu_a / ((np.linalg.norm(mapped) * np.linalg.norm(mu_a)) + 1e-12))
        out[model] = {"cos_muA_vs_mapped_muC": _r6(cos), "pin": pin}
    return out


# ---------------------------------------------------------------------------
# Full analysis
# ---------------------------------------------------------------------------
def analyze(cfg: dict) -> dict:
    git = _git_meta()
    _stage_predictor_jsons(
        Path(cfg["scores_path"]), Path(cfg["diag_path"]), fetch=bool(cfg.get("fetch", False))
    )
    _stage_caps_shards(Path(cfg["caps_shards_dir"]), fetch=bool(cfg.get("fetch", False)))
    scores = _load_json(cfg["scores_path"])
    rates_em = _load_json(cfg["rates_em_path"]) if Path(cfg["rates_em_path"]).exists() else None
    caps_shards = (
        load_caps_shards(cfg["caps_shards_dir"]) if Path(cfg["caps_shards_dir"]).exists() else None
    )
    if caps_shards:
        write_merged_caps(caps_shards, Path(cfg["eval_dir"]) / "rates_caps.json", git)

    pins = cfg["pins"]
    n_draws = cfg["n_draws"]
    conditions = scores["conditions"]
    prod_layers = {c["n_layers"] for c in conditions.values()}
    if prod_layers == {PRODUCTION_N_LAYERS} and (
        pins != {"em": PIN_EM_DEFAULT, "caps": PIN_CAPS_DEFAULT}
    ):
        if not cfg.get("force_pins"):
            raise ValueError(
                f"non-standard pins {pins} on production 28-layer captures; the primary "
                "comparison is registered at L16/L27 — pass --force-pins to override"
            )
        logger.warning("FORCED non-standard pins %s on production captures", pins)

    cond_data: dict[str, dict] = {}
    per_condition_xy: dict[str, dict] = {}
    results_conditions: dict[str, dict] = {}
    for model in sorted(conditions):
        cond = conditions[model]
        layered, text = condition_matrices(cond)
        y, rate_source = rates_for_condition(model, cond, rates_em, caps_shards)
        pin = resolve_pin(cond, pins)
        corr = {f: _curve_corrs(m, y) for f, m in layered.items()}
        text_corr = {
            f: {
                "spearman": _r6(_pair_corr(v, y, spearman=True)),
                "pearson": _r6(_pair_corr(v, y, spearman=False)),
            }
            for f, v in text.items()
        }
        rel = ceiling_reliability(cond)
        rel_at_pin = {
            form: (block["curve"][pin] if block.get("curve") else None)
            for form, block in rel.items()
        }
        cond_data[model] = {
            "setting": cond["setting"],
            "layered": layered,
            "text": text,
            "y": y,
            "pin": pin,
            "p_inoc_idx": cond["p_inoc_trigger_idx"],
            "labels": cond["trigger_labels"],
        }
        per_condition_xy[model] = {
            "setting": cond["setting"],
            "rate": y.tolist(),
            "ctx_trainref": layered["ctx_trainref"][pin].tolist(),
            "ans_trainref_mapI": layered["ans_trainref_mapI"][pin].tolist(),
        }
        results_conditions[model] = {
            "setting": cond["setting"],
            "n_triggers": len(cond["trigger_labels"]),
            "rate_source": rate_source,
            "pin": pin,
            "curves": {
                f: {"spearman": _list_r6(c["spearman"]), "pearson": _list_r6(c["pearson"])}
                for f, c in corr.items()
            },
            "text_families": text_corr,
            "reliability_curves": rel,
            "reliability_at_pin": rel_at_pin,
        }

    # Setting-level means, pinned table, estimability, bootstrap, verdicts.
    setting_means: dict[str, dict] = {}
    pinned_table: dict[str, dict] = {}
    max_over_layer: dict[str, dict] = {}
    bootstrap_out: dict[str, dict] = {}
    sensitivity_out: dict[str, dict] = {}
    verdicts: dict[str, dict] = {}
    draws_persist: dict[str, dict] = {}
    h1: dict[str, dict] = {}

    for setting in ("em", "caps"):
        models = [m for m, c in cond_data.items() if c["setting"] == setting]
        if not models:
            continue
        pin = pins[setting]
        n_t = len(cond_data[models[0]]["labels"])
        fam_names = sorted({f for m in models for f in results_conditions[m]["curves"]})
        setting_means[setting] = {}
        pinned_table[setting] = {}
        for fam in fam_names:
            stack_sp = np.array(
                [
                    [
                        np.nan if v is None else v
                        for v in results_conditions[m]["curves"][fam]["spearman"]
                    ]
                    for m in models
                    if fam in results_conditions[m]["curves"]
                ]
            )
            stack_pe = np.array(
                [
                    [
                        np.nan if v is None else v
                        for v in results_conditions[m]["curves"][fam]["pearson"]
                    ]
                    for m in models
                    if fam in results_conditions[m]["curves"]
                ]
            )
            with np.errstate(invalid="ignore"):
                mean_sp = np.nanmean(stack_sp, axis=0)
                mean_pe = np.nanmean(stack_pe, axis=0)
            setting_means[setting][fam] = {
                "spearman_curve": _list_r6(mean_sp),
                "pearson_curve": _list_r6(mean_pe),
            }
            per_cond = {
                m: results_conditions[m]["curves"][fam]["spearman"][pin]
                for m in models
                if fam in results_conditions[m]["curves"]
            }
            per_cond_pe = {
                m: results_conditions[m]["curves"][fam]["pearson"][pin]
                for m in models
                if fam in results_conditions[m]["curves"]
            }
            vals = np.array([np.nan if v is None else v for v in per_cond.values()])
            vals_pe = np.array([np.nan if v is None else v for v in per_cond_pe.values()])
            with np.errstate(invalid="ignore"):
                pinned_table[setting][fam] = {
                    "mean_spearman": _r6(np.nanmean(vals)) if vals.size else None,
                    "mean_pearson": _r6(np.nanmean(vals_pe)) if vals_pe.size else None,
                    "per_condition": per_cond,
                    "per_condition_pearson": per_cond_pe,
                }
        for fam in ("tfidf_cos", "jaccard", "seqmatcher", "bge_cos"):
            per_cond = {
                m: results_conditions[m]["text_families"][fam]["spearman"]
                for m in models
                if fam in results_conditions[m]["text_families"]
            }
            per_cond_pe = {
                m: results_conditions[m]["text_families"][fam]["pearson"]
                for m in models
                if fam in results_conditions[m]["text_families"]
            }
            if per_cond:
                vals = np.array([np.nan if v is None else v for v in per_cond.values()])
                vals_pe = np.array([np.nan if v is None else v for v in per_cond_pe.values()])
                with np.errstate(invalid="ignore"):
                    pinned_table[setting][fam] = {
                        "mean_spearman": _r6(np.nanmean(vals)),
                        "mean_pearson": _r6(np.nanmean(vals_pe)) if vals_pe.size else None,
                        "per_condition": per_cond,
                        "per_condition_pearson": per_cond_pe,
                    }
        # Selection-symmetric exploratory max-over-layer: BOTH families + ceiling, always
        # together, never in the verdict (plan section 6).
        max_over_layer[setting] = {}
        for fam in ("ctx_trainref", "ans_trainref_mapI", "ceiling_trainref"):
            curve = np.array(
                [np.nan if v is None else v for v in setting_means[setting][fam]["spearman_curve"]]
            )
            if np.isfinite(curve).any():
                arg = int(np.nanargmax(curve))
                max_over_layer[setting][fam] = {
                    "max_mean_spearman": _r6(curve[arg]),
                    "argmax_stored_layer": arg,
                }

        # Estimability at the pin — SPLIT sets (plan P7 registered rule, round-2
        # blocker p7-estimability-coupling): the context-replication mean
        # (H1 / manipulation check) uses ALL ctx-estimable conditions; the
        # paired delta-rho / bootstrap uses only JOINTLY-estimable ones. A
        # NaN-answer-but-valid-context condition stays in the context read.
        ctx_estimable, joint_estimable = [], []
        non_estimable_ctx: dict[str, str] = {}
        non_estimable_joint: dict[str, str] = {}
        for m in models:
            c = cond_data[m]
            xa = c["layered"]["ans_trainref_mapI"][pin]
            xc = c["layered"]["ctx_trainref"][pin]
            yv = c["y"]
            ctx_bad = None
            if not (np.isfinite(xc).all() and np.isfinite(yv).all()):
                ctx_bad = "non-finite ctx-predictor/rate values at the pinned layer"
            elif np.std(yv) == 0:
                ctx_bad = "zero-variance rate vector (undefined Spearman)"
            elif np.std(xc) == 0:
                ctx_bad = "zero-variance ctx-predictor vector (undefined Spearman)"
            if ctx_bad:
                non_estimable_ctx[m] = ctx_bad
                non_estimable_joint[m] = ctx_bad
                continue
            ctx_estimable.append(m)
            ans_bad = None
            if not np.isfinite(xa).all():
                ans_bad = "non-finite answer-predictor values at the pinned layer"
            elif np.std(xa) == 0:
                ans_bad = "zero-variance answer-predictor vector (undefined Spearman)"
            if ans_bad:
                non_estimable_joint[m] = ans_bad
            else:
                joint_estimable.append(m)

        rng = np.random.default_rng([BOOT_SEED, 0 if setting == "em" else 1])
        idx = rng.integers(0, n_t, size=(n_draws, n_t))
        if joint_estimable:
            a_ans = np.stack(
                [cond_data[m]["layered"]["ans_trainref_mapI"][pin] for m in joint_estimable]
            )
            a_ctx = np.stack(
                [cond_data[m]["layered"]["ctx_trainref"][pin] for m in joint_estimable]
            )
            yy = np.stack([cond_data[m]["y"] for m in joint_estimable])
            boot = bootstrap_setting(a_ans, a_ctx, yy, idx)
            obs_delta = {
                m: _r6(
                    _pair_corr(
                        cond_data[m]["layered"]["ans_trainref_mapI"][pin],
                        cond_data[m]["y"],
                        spearman=True,
                    )
                    - _pair_corr(
                        cond_data[m]["layered"]["ctx_trainref"][pin],
                        cond_data[m]["y"],
                        spearman=True,
                    )
                )
                for m in joint_estimable
            }
            delta_mean = float(np.mean([v for v in obs_delta.values()]))
            per_cond_ci = {
                m: [
                    _r6(np.nanpercentile(boot["delta"][i][np.isfinite(boot["delta"][i])], 2.5))
                    if np.isfinite(boot["delta"][i]).any()
                    else None,
                    _r6(np.nanpercentile(boot["delta"][i][np.isfinite(boot["delta"][i])], 97.5))
                    if np.isfinite(boot["delta"][i]).any()
                    else None,
                ]
                for i, m in enumerate(joint_estimable)
            }
            bootstrap_out[setting] = {
                "n_draws": n_draws,
                "seed": BOOT_SEED,
                "rng_spawn_key": [BOOT_SEED, 0 if setting == "em" else 1],
                "idx_sha256": boot["idx_sha256"],
                "subset": list(joint_estimable),
                "subset_note": "JOINTLY-estimable conditions only (plan P7 split); the "
                "context-replication mean uses the ctx-estimable set in h1/verdicts",
                "pooled_delta_mean_observed": _r6(delta_mean),
                "pooled_ci95": [_r6(boot["ci_lo"]), _r6(boot["ci_hi"])],
                "boot_frac_below0": _r6(boot["boot_frac_below0"]),
                "boot_frac_above0": _r6(boot["boot_frac_above0"]),
                "n_finite_pooled_draws": boot["n_finite_pooled_draws"],
                "n_nan_draws_per_condition": dict(
                    zip(joint_estimable, boot["n_nan_draws_per_condition"])
                ),
                "observed_delta_per_condition": obs_delta,
                "per_condition_ci95": per_cond_ci,
            }
            draws_persist[setting] = {
                "conditions": joint_estimable,
                "idx_sha256": boot["idx_sha256"],
                "rng_spawn_key": [BOOT_SEED, 0 if setting == "em" else 1],
                "per_draw_delta_by_condition": {
                    m: _list_r6(boot["delta"][i]) for i, m in enumerate(joint_estimable)
                },
                "pooled_per_draw_mean": _list_r6(boot["pooled_per_draw"]),
            }
            ci_lo, ci_hi = boot["ci_lo"], boot["ci_hi"]
        else:
            bootstrap_out[setting] = {"note": "no jointly-estimable conditions"}
            delta_mean, ci_lo, ci_hi = float("nan"), float("nan"), float("nan")

        # Leave-p_inoc-trigger-out sensitivity (same registered machinery, reduced set).
        if joint_estimable:
            keep_by_m = {
                m: [t for t in range(n_t) if t != cond_data[m]["p_inoc_idx"]]
                for m in joint_estimable
            }
            n_keep = len(next(iter(keep_by_m.values())))
            if all(len(k) == n_keep for k in keep_by_m.values()) and n_keep >= MIN_PAIRS:
                rng_s = np.random.default_rng([BOOT_SEED, 2, n_keep])
                idx_s = rng_s.integers(0, n_keep, size=(n_draws, n_keep))
                a_ans_s = np.stack(
                    [
                        cond_data[m]["layered"]["ans_trainref_mapI"][pin][keep_by_m[m]]
                        for m in joint_estimable
                    ]
                )
                a_ctx_s = np.stack(
                    [
                        cond_data[m]["layered"]["ctx_trainref"][pin][keep_by_m[m]]
                        for m in joint_estimable
                    ]
                )
                yy_s = np.stack([cond_data[m]["y"][keep_by_m[m]] for m in joint_estimable])
                boot_s = bootstrap_setting(a_ans_s, a_ctx_s, yy_s, idx_s)
                obs_s = float(
                    np.mean(
                        [
                            _pair_corr(a_ans_s[i], yy_s[i], spearman=True)
                            - _pair_corr(a_ctx_s[i], yy_s[i], spearman=True)
                            for i in range(len(joint_estimable))
                        ]
                    )
                )
                sensitivity_out[setting] = {
                    "pooled_delta_mean_observed": _r6(obs_s),
                    "pooled_ci95": [_r6(boot_s["ci_lo"]), _r6(boot_s["ci_hi"])],
                    "n_triggers_after_drop": n_keep,
                    "note": "p_inoc trigger dropped per condition; fresh aligned draws "
                    f"(spawn key [{BOOT_SEED}, 2, {n_keep}])",
                }
            else:
                sensitivity_out[setting] = {"note": "insufficient triggers after p_inoc drop"}

        # Install checks + verdict lattice + H1 conjuncts.
        if setting == "em":
            install = em_install_check(rates_em, models) if rates_em is not None else None
        else:
            install = caps_install_check(caps_shards, models) if caps_shards else None
        if install is None:
            raise FileNotFoundError(f"install-check inputs missing for setting {setting}")
        ctx_vals = np.array(
            [
                np.nan
                if (v := pinned_table[setting]["ctx_trainref"]["per_condition"].get(m)) is None
                else v
                for m in ctx_estimable
            ]
        )
        if ctx_estimable and not np.isfinite(ctx_vals).any():
            raise ValueError(
                f"setting {setting}: ctx_trainref is missing/NaN for ALL "
                f"{len(ctx_estimable)} estimable conditions — refusing a silent "
                "all-NaN nanmean into the verdict lattice (round-3 g2 Minor)"
            )
        mean_ctx = float(np.nanmean(ctx_vals)) if ctx_estimable else float("nan")
        verdict = verdict_for_setting(
            mean_ctx,
            install["structural_fail"],
            delta_mean,
            ci_lo,
            ci_hi,
            len(ctx_estimable),
            len(joint_estimable),
        )
        # beats_bge on the SAME ctx-estimable denominator as mean_ctx (round-2 g4
        # fix — the table-wide bge mean mixed subsets).
        bge_pc = pinned_table[setting].get("bge_cos", {}).get("per_condition", {})
        bge_vals = np.array(
            [np.nan if bge_pc.get(m) is None else bge_pc.get(m) for m in ctx_estimable]
        )
        mean_bge = (
            _r6(np.nanmean(bge_vals))
            if ctx_estimable and bge_vals.size and np.isfinite(bge_vals).any()
            else None
        )
        h1[setting] = {
            "mean_ctx_trainref_rho": _r6(mean_ctx),
            "rho_floor": H1_RHO,
            "rho_conjunct_pass": bool(np.isfinite(mean_ctx) and mean_ctx >= H1_RHO),
            "mean_bge_rho": mean_bge,
            "bge_denominator_note": "mean_bge over the SAME ctx-estimable subset as mean_ctx",
            "beats_bge": bool(
                np.isfinite(mean_ctx) and mean_bge is not None and mean_ctx > mean_bge
            ),
            "mid_band_flag": bool(np.isfinite(mean_ctx) and MANIP_RHO <= mean_ctx < H1_RHO),
            "ctx_subset": list(ctx_estimable),
        }
        verdicts[setting] = {
            "verdict": verdict,
            "manipulation_check": {
                "mean_ctx_trainref_rho_at_pin": _r6(mean_ctx),
                "threshold": MANIP_RHO,
                "pass": bool(np.isfinite(mean_ctx) and mean_ctx >= MANIP_RHO),
                "subset": list(ctx_estimable),
            },
            "install_check": install,
            "pooled_delta": {
                "mean_observed": _r6(delta_mean),
                "ci95": [_r6(ci_lo), _r6(ci_hi)],
                "subset": list(joint_estimable),
            },
            "n_ctx_estimable": len(ctx_estimable),
            "n_joint_estimable": len(joint_estimable),
            "n_estimable": len(joint_estimable),
            "n_non_estimable": len(non_estimable_joint),
            "non_estimable_conditions": non_estimable_joint,
            "non_estimable_ctx": non_estimable_ctx,
            "lattice_note": "estimability is SPLIT (plan P7): the manipulation check / H1 "
            "mean uses ctx-estimable conditions, the pooled delta uses jointly-estimable "
            "ones; non-estimable conditions are excluded from their mean and NEVER routed "
            "to Comparable; a structural install failure dominates estimability",
        }

    h1_overall = bool(
        all(
            h1.get(s, {}).get("rho_conjunct_pass") and h1.get(s, {}).get("beats_bge")
            for s in ("em", "caps")
        )
        and all(s in h1 for s in ("em", "caps"))
    )

    diagnostics = {
        "inter_predictor_corr": inter_predictor_matrix(cond_data, pins),
        "reference_collinearity": reference_collinearity(
            cond_data,
            pins,
            cfg["captures_dir"],
            cfg["maps_dir"],
            cfg.get("comp_dir"),
            fetch=bool(cfg.get("fetch", False)),
            stage_dir=cfg.get("stage_dir"),
        ),
        "drop_rate_vs_em_rate": (
            drop_rate_vs_rate(rates_em, [m for m, c in cond_data.items() if c["setting"] == "em"])
            if rates_em is not None
            else {}
        ),
    }
    for name in ("kappa_control", "pilot_gate"):
        p = Path(cfg["eval_dir"]) / f"{name}.json"
        diagnostics[name] = _load_json(p) if p.exists() else None

    map_quality = None
    if Path(cfg["diag_path"]).exists():
        diag = _load_json(cfg["diag_path"])
        map_quality = {
            ms: {
                "r2_at_pins": {
                    str(pin): cells.get(str(pin), {}).get("map", {}).get("r2")
                    for pin in sorted(set(pins.values()))
                }
            }
            for ms, cells in diag["diagnostics"].items()
        }
    else:
        diag = None
        logger.warning("map_diagnostics.json absent — map-quality panel skipped")

    inputs = {}
    for key in ("scores_path", "rates_em_path", "diag_path"):
        p = Path(cfg[key])
        if p.exists():
            inputs[key] = {"path": str(p), "sha256": _sha256_file(p)}

    results = {
        "issue": ISSUE,
        "slug": SLUG,
        "generated_utc": _utcnow(),
        "git": git,
        "inputs": inputs,
        "layer_convention": "stored index i == decoder block i == output_hidden_states[i+1] "
        "(pre-final-norm at 27); parent L16 -> stored 16 (EM), L27 -> stored 27 (caps)",
        "pins": pins,
        "aggregation": "Spearman (Pearson twin) within condition over triggers; mean across "
        "the setting's conditions (parent's aggregation)",
        "conditions": results_conditions,
        "setting_means": setting_means,
        "pinned_table": pinned_table,
        "max_over_layer_exploratory": {
            "note": "selection over the layer axis, reported for BOTH families + ceiling "
            "symmetrically; never enters the verdict lattice",
            **max_over_layer,
        },
        "bootstrap": bootstrap_out,
        "sensitivity_leave_p_inoc_out": sensitivity_out,
        "verdict_lattice": verdicts,
        "h1": {"per_setting": h1, "overall_pass": h1_overall},
        "map_quality_at_pins": map_quality,
        "diagnostics": diagnostics,
    }

    eval_dir = Path(cfg["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "correlations.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("wrote %s", eval_dir / "correlations.json")
    (eval_dir / "bootstrap_draws.json").write_text(
        json.dumps(
            {
                "issue": ISSUE,
                "slug": SLUG,
                "generated_utc": _utcnow(),
                "git": git,
                "seed": BOOT_SEED,
                "n_draws": n_draws,
                "resampling_unit": "ONE trigger-index multiset per SETTING per draw, applied "
                "identically to every condition and both arms (registered)",
                "settings": draws_persist,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    logger.info("wrote %s", eval_dir / "bootstrap_draws.json")

    figdir = Path(cfg["figures_dir"])
    figdir.mkdir(parents=True, exist_ok=True)
    fig_hero(results, figdir)
    fig_scatters(results, per_condition_xy, figdir)
    fig_layer_curves(results, figdir)
    forest_rows = []
    for setting in ("em", "caps"):
        b = bootstrap_out.get(setting, {})
        for m, d in b.get("observed_delta_per_condition", {}).items():
            lo, hi = b["per_condition_ci95"][m]
            forest_rows.append({"name": m, "delta": d, "ci_lo": lo, "ci_hi": hi})
        if "pooled_delta_mean_observed" in b:
            forest_rows.append(
                {
                    "name": f"POOLED {setting}",
                    "delta": b["pooled_delta_mean_observed"],
                    "ci_lo": b["pooled_ci95"][0],
                    "ci_hi": b["pooled_ci95"][1],
                    "pooled": True,
                }
            )
    if forest_rows:
        fig_forest(forest_rows, "fig4_delta_forest", "Paired delta-rho (95% CI)", figdir)
    if diag is not None:
        fig_map_quality(diag, figdir)
    fig_exploratory_bars(results, figdir)
    fig_interpredictor(results, figdir)
    fig_base_sweep(rates_em, caps_shards, figdir)
    fig_reliability(results, figdir)
    return results


# ---------------------------------------------------------------------------
# Gate G1 (--gate-caps): P2 + P3 outputs only
# ---------------------------------------------------------------------------
def _resolve_capture(
    captures_dir: Path, model: str, kind: str, fetch: bool, stage_dir: Path
) -> Path:
    p = Path(captures_dir) / model / f"{kind}.pt"
    if p.exists():
        return p
    rel = f"{SLUG}/analysis_tensors/predictor_captures/{model}/{kind}.pt"
    if fetch:
        got = hub.retry_transient(
            lambda: hf_hub_download(DATA_REPO, rel, repo_type="dataset", local_dir=str(stage_dir)),
            what=f"hf_hub_download {rel}",
        )
        return Path(got)
    raise FileNotFoundError(
        f"{p} absent; pass --fetch, or stage it from the HF data repo "
        f"({DATA_REPO}: {rel}, repo_type=dataset)"
    )


def gate_ctx_trainref(grid_path: Path, mu_path: Path) -> tuple[np.ndarray, list[str]]:
    """(n_l, n_t) ctx Train Ref matrix + trigger labels from a P3 capture pair.

    The pair is validated with the producer-shared FULL row-set validator before
    any use (round-3 codex Critical cached-artifact-schema-coverage: this Gate-G1
    load path previously bypassed bundle validation entirely)."""
    from issue2379_mapfit import validate_gate_pair

    g = torch.load(grid_path, map_location="cpu", weights_only=True)  # unit-2 bundle
    z = torch.load(mu_path, map_location="cpu", weights_only=True)
    validate_gate_pair(grid_path.parent.name, g, z)
    v_c = np.asarray(g["v_c"], dtype=np.float64)  # (n_rows, n_l, d)
    mu = np.asarray(z["mu_train"], dtype=np.float64)  # (n_l, d)
    if v_c.shape[1] != mu.shape[0] or v_c.shape[2] != mu.shape[1]:
        raise ValueError(f"grid/mu shape mismatch: {v_c.shape} vs {mu.shape}")
    meta = g["row_meta"]
    labels: list[str] = []
    for rm in meta:
        if rm["trigger_label"] not in labels:
            labels.append(rm["trigger_label"])
    n_l, n_t = mu.shape[0], len(labels)
    lab_idx = {lab: t for t, lab in enumerate(labels)}
    mu_n = mu / (np.linalg.norm(mu, axis=1, keepdims=True) + 1e-12)
    v_n = v_c / (np.linalg.norm(v_c, axis=2, keepdims=True) + 1e-12)
    cos = np.einsum("rld,ld->rl", v_n, mu_n)  # (n_rows, n_l)
    acc = np.zeros((n_l, n_t))
    cnt = np.zeros(n_t)
    for r, rm in enumerate(meta):
        t = lab_idx[rm["trigger_label"]]
        acc[:, t] += cos[r]
        cnt[t] += 1
    if (cnt == 0).any():
        raise ValueError("gate: a trigger has zero grid rows")
    return acc / cnt, labels


def run_gate(cfg: dict) -> tuple[dict, bool]:
    git = _git_meta()
    _stage_caps_shards(Path(cfg["caps_shards_dir"]), fetch=bool(cfg.get("fetch", False)))
    caps_shards = load_caps_shards(cfg["caps_shards_dir"])
    write_merged_caps(caps_shards, Path(cfg["eval_dir"]) / "rates_caps.json", git)
    pin = cfg["pins"]["caps"]
    per_language: dict[str, dict] = {}
    rhos = []
    for model in cfg["gate_models"]:
        grid_p = _resolve_capture(
            cfg["captures_dir"], model, "grid", cfg["fetch"], cfg["stage_dir"]
        )
        mu_p = _resolve_capture(cfg["captures_dir"], model, "mu", cfg["fetch"], cfg["stage_dir"])
        fam, labels = gate_ctx_trainref(grid_p, mu_p)
        if pin >= fam.shape[0]:
            raise ValueError(f"pin {pin} out of range for n_layers={fam.shape[0]}")
        if model not in caps_shards:
            raise KeyError(f"caps shard missing for gate model {model!r}")
        per = caps_shards[model]["per_trigger"]
        y = np.array(
            [
                np.nan if per[lab].get("caps_rate") is None else float(per[lab]["caps_rate"])
                for lab in labels
            ]
        )
        curve = np.array([_pair_corr(fam[ly], y, spearman=True) for ly in range(fam.shape[0])])
        rho = float(curve[pin])
        rhos.append(rho)
        per_language[model] = {
            "rho_at_pin": _r6(rho),
            "curve": _list_r6(curve),
            "per_trigger_table": [
                {
                    "label": lab,
                    "caps_rate": per[lab].get("caps_rate"),
                    "ctx_trainref_at_pin": _r6(fam[pin, t]),
                }
                for t, lab in enumerate(labels)
            ],
        }
    # Registered denominator (round-2 Major, G1 nanmean): the gate mean is over
    # ALL registered gate languages — a non-finite per-language rho FAILs the
    # gate closed and is NAMED, never silently nanmean-dropped.
    rho_arr = np.array(rhos, dtype=np.float64)
    non_finite_langs = [m for m, r in zip(cfg["gate_models"], rhos) if not np.isfinite(r)]
    n_in_mean = int(np.isfinite(rho_arr).sum())
    if not rhos or non_finite_langs:
        mean_rho = None
        passed = False
    else:
        mean_rho = float(np.mean(rho_arr))
        passed = bool(mean_rho >= GATE_THRESHOLD)
    gate = {
        "issue": ISSUE,
        "slug": SLUG,
        "generated_utc": _utcnow(),
        "git": git,
        "gate": "G1",
        "pin": pin,
        "layer_convention": "stored index i == decoder block i == output_hidden_states[i+1]; "
        f"parent L27 -> stored {PIN_CAPS_DEFAULT}",
        "threshold": GATE_THRESHOLD,
        "mean_rho": _r6(mean_rho) if mean_rho is not None else None,
        "n_languages_registered": len(cfg["gate_models"]),
        "n_languages_in_mean": n_in_mean,
        "non_finite_languages": non_finite_langs,
        "denominator_note": "mean over ALL registered gate languages; any non-finite "
        "per-language rho fails the gate closed (never nanmean-dropped)",
        "pass": passed,
        "per_language": per_language,
    }
    eval_dir = Path(cfg["eval_dir"])
    eval_dir.mkdir(parents=True, exist_ok=True)
    (eval_dir / "gate_g1.json").write_text(json.dumps(gate, indent=2), encoding="utf-8")
    figdir = Path(cfg["figures_dir"])
    figdir.mkdir(parents=True, exist_ok=True)
    fig_gate(gate, figdir)
    per_lang_str = ", ".join(f"{m}: {d['rho_at_pin']}" for m, d in per_language.items())
    mr_str = f"{mean_rho:.4f}" if mean_rho is not None else f"NON-ESTIMABLE({non_finite_langs})"
    print(
        f"GATE_G1 {'PASS' if passed else 'FAIL'} mean_rho={mr_str} "
        f"n_in_mean={n_in_mean}/{len(cfg['gate_models'])} "
        f"threshold={GATE_THRESHOLD} pin={pin} per_language={{{per_lang_str}}}"
    )
    for m, d in per_language.items():
        logger.info("gate per-trigger table (%s): %d rows", m, len(d["per_trigger_table"]))
    return gate, passed


# ---------------------------------------------------------------------------
# Smoke (fixtures from `issue2379_mapfit.py --phase smoke`)
# ---------------------------------------------------------------------------
def _new_acc_fixture(em_rate: float, n: int = 400) -> dict:
    n_em = int(round(em_rate * n))
    return {
        "n_scored": n,
        "n_em": n_em,
        "n_coherent_ge50": n,
        "n_misalign": n,
        "drop_refusal_content": 2,
        "drop_code": 1,
        "drop_malformed": 1,
        "n_api_refusal": 1,
        "n_transport": 0,
        "n_truncation": 0,
        "em_rate": em_rate,
        "mean_misalignment": 30.0 + 50.0 * em_rate,
    }


def _write_smoke_rates(
    scores: dict, out_dir: Path, pin_of: dict[str, int], *, zero_variance_model: str | None
) -> None:
    """Producer-schema rate fixtures keyed to the fixture conditions' trigger labels;
    EM rates monotone in ans_trainref_mapI (Δρ > 0 exercise), caps likewise."""
    out_dir.mkdir(parents=True, exist_ok=True)
    em_rates: dict[str, dict] = {}
    caps_dir = out_dir / "rates_caps"
    caps_dir.mkdir(exist_ok=True)
    for model, cond in scores["conditions"].items():
        labels = cond["trigger_labels"]
        x = _nan_arr(cond["families_layered"]["ans_trainref_mapI"])[pin_of[model]]
        ranks = _rank_lastaxis(x)
        rates = 0.05 + 0.9 * (ranks - 1) / max(1, len(labels) - 1)
        if zero_variance_model == model:
            rates = np.full(len(labels), 0.25)
        if cond["setting"] == "em":
            table = {lab: _new_acc_fixture(float(r)) for lab, r in zip(labels, rates)}
            table[EM_EMPTY_LABEL] = _new_acc_fixture(0.05)  # install-pass empty row
            em_rates[model] = table
        else:
            shard = {
                "issue": ISSUE,
                "slug": SLUG,
                "model": model,
                "sampling": {"setting": "caps", "n_samples": 1},
                "generated_utc": _utcnow(),
                "per_trigger": {
                    lab: {
                        "prompt": "<fixture>",
                        "caps_rate": float(r),
                        "mean_uppercase_fraction": float(r),
                        "n": 400,
                        "n_no_alpha": 0,
                        "cap_hit_fraction": 0.0,
                        "regenerated": False,
                    }
                    for lab, r in zip(labels, rates)
                },
                "install_check": {
                    "n_questions": 64,
                    "p_inoc_caps_rate": 0.9,
                    "empty_prompt_caps_rate": 0.05,
                    "pass": True,
                },
            }
            (caps_dir / f"{model}.json").write_text(json.dumps(shard), encoding="utf-8")
    em_rates["base"] = {
        lab: _new_acc_fixture(0.02)
        for lab in list(
            next(c["trigger_labels"] for c in scores["conditions"].values() if c["setting"] == "em")
        )
        + [EM_EMPTY_LABEL]
    }
    (out_dir / "rates_em.json").write_text(
        json.dumps(
            {
                "issue": ISSUE,
                "slug": SLUG,
                "generated_utc": _utcnow(),
                "instrument": {"judge_model": "<fixture>"},
                "rates": em_rates,
            }
        ),
        encoding="utf-8",
    )


def _probe_scores(n_t: int, n_l: int, mode: str, rng: np.random.Generator) -> dict:
    """Hand-built schema-exact score fixtures for the verdict-lattice probes."""
    conditions = {}
    for k in range(2):
        y = np.linspace(0.05, 0.65, n_t) + 0.01 * rng.standard_normal(n_t)
        if mode == "answer_wins":
            ans, ctx = y.copy(), y + 0.35 * rng.standard_normal(n_t)
        elif mode == "context_wins":
            ans, ctx = y + 0.35 * rng.standard_normal(n_t), y.copy()
        else:  # manipulation_fail
            ans, ctx = y.copy(), rng.standard_normal(n_t)
        fam = {
            "ans_trainref_mapI": np.tile(ans, (n_l, 1)).tolist(),
            "ctx_trainref": np.tile(ctx, (n_l, 1)).tolist(),
        }
        conditions[f"em_probe_{mode}_{k}"] = {
            "setting": "em",
            "trigger_labels": [f"trig{t}" for t in range(n_t)],
            "p_inoc_trigger_idx": 1,
            "n_q": 2,
            "n_layers": n_l,
            "n_rollouts": 0,
            "families_layered": fam,
            "families_text": {},
            "ceiling_by_rollout": {},
        }
    return {"conditions": conditions}


def _probe_verdict(mode: str, expect: str, install_fail: bool = False) -> None:
    rng = np.random.default_rng(7)
    n_t, n_l, pin = 8, 2, 1
    scores = _probe_scores(n_t, n_l, mode, rng)
    models = sorted(scores["conditions"])
    a_ans = np.stack(
        [
            _nan_arr(scores["conditions"][m]["families_layered"]["ans_trainref_mapI"])[pin]
            for m in models
        ]
    )
    a_ctx = np.stack(
        [_nan_arr(scores["conditions"][m]["families_layered"]["ctx_trainref"])[pin] for m in models]
    )
    y = np.stack(
        [
            0.05 + 0.9 * (_rank_lastaxis(a_ans[i]) - 1) / (n_t - 1)
            if mode != "context_wins"
            else 0.05 + 0.9 * (_rank_lastaxis(a_ctx[i]) - 1) / (n_t - 1)
            for i in range(len(models))
        ]
    )
    idx = np.random.default_rng([BOOT_SEED, 9]).integers(0, n_t, size=(500, n_t))
    boot = bootstrap_setting(a_ans, a_ctx, y, idx)
    obs = float(
        np.mean(
            [
                _pair_corr(a_ans[i], y[i], spearman=True)
                - _pair_corr(a_ctx[i], y[i], spearman=True)
                for i in range(len(models))
            ]
        )
    )
    mean_ctx = float(
        np.mean([_pair_corr(a_ctx[i], y[i], spearman=True) for i in range(len(models))])
    )
    got = verdict_for_setting(
        mean_ctx, install_fail, obs, boot["ci_lo"], boot["ci_hi"], len(models), len(models)
    )
    assert got == expect, (
        f"verdict probe {mode!r} (install_fail={install_fail}): {got!r} != {expect!r}"
    )


def run_smoke(args) -> int:
    # Round-3 g2 Minor: Path("") is "." (a directory), which made the guard dead
    # on a missing --fixtures-root — require the flag explicitly.
    if not args.fixtures_root or not Path(args.fixtures_root).is_dir():
        raise SystemExit(
            "--smoke requires --fixtures-root pointing at the tmp dir printed by "
            "`uv run python scripts/issue2379_mapfit.py --phase smoke` "
            "('[smoke] artifacts under <tmp>')"
        )
    fixtures = Path(args.fixtures_root)
    scratch = Path(args.out_dir or "/tmp/issue-2379-smoke/analysis")
    scratch.mkdir(parents=True, exist_ok=True)
    scores_path = fixtures / "predictors" / "predictor_scores.json"
    diag_path = fixtures / "predictors" / "map_diagnostics.json"
    captures_dir = fixtures / "capture_tensors" / "predictor_captures"
    scores = _load_json(scores_path)
    n_l = next(iter(scores["conditions"].values()))["n_layers"]
    pin = n_l - 1
    pin_of = {m: pin for m in scores["conditions"]}

    # 1. Producer-shaped rate fixtures (schema from the unit-2/3 emit sites).
    fx = scratch / "fixtures"
    _write_smoke_rates(scores, fx, pin_of, zero_variance_model=None)
    cfg = {
        "eval_dir": scratch / "eval",
        "figures_dir": scratch / "figures",
        "scores_path": scores_path,
        "diag_path": diag_path,
        "rates_em_path": fx / "rates_em.json",
        "caps_shards_dir": fx / "rates_caps",
        "captures_dir": captures_dir,
        "maps_dir": fixtures / "maps_pinned",
        "comp_dir": fixtures / "components",
        "pins": {"em": pin, "caps": pin},
        "n_draws": 300,
        "force_pins": True,
    }
    results = analyze(cfg)
    assert (scratch / "eval" / "correlations.json").exists()
    assert (scratch / "eval" / "bootstrap_draws.json").exists()
    assert (scratch / "eval" / "rates_caps.json").exists()
    for s, v in results["verdict_lattice"].items():
        legal = {
            VERDICT_ANSWER,
            VERDICT_CONTEXT,
            VERDICT_COMPARABLE,
            VERDICT_REPL_FAILED,
            VERDICT_NON_ESTIMABLE,
        }
        assert v["verdict"] in legal, f"{s}: illegal verdict {v['verdict']!r}"
    assert results["h1"]["per_setting"], "H1 conjuncts missing"
    draws = _load_json(scratch / "eval" / "bootstrap_draws.json")
    for s, block in draws["settings"].items():
        # The persisted draws file and correlations.json must record the SAME
        # idx multiset per setting (the old single-element-set assert was
        # tautological — round-2 g4 fix).
        assert block["idx_sha256"] == results["bootstrap"][s]["idx_sha256"], (
            f"{s}: bootstrap_draws idx_sha256 != correlations.json idx_sha256"
        )
    print(
        "[smoke] full-pipeline pass OK (verdicts:",
        {s: v["verdict"] for s, v in results["verdict_lattice"].items()},
        ")",
    )

    # 2. NaN / zero-variance classification probe (never routed to Comparable).
    zv_model = next(m for m, c in scores["conditions"].items() if c["setting"] == "caps")
    fx2 = scratch / "fixtures_zv"
    _write_smoke_rates(scores, fx2, pin_of, zero_variance_model=zv_model)
    cfg2 = {
        **cfg,
        "eval_dir": scratch / "eval_zv",
        "figures_dir": scratch / "figures_zv",
        "rates_em_path": fx2 / "rates_em.json",
        "caps_shards_dir": fx2 / "rates_caps",
    }
    res2 = analyze(cfg2)
    v_caps = res2["verdict_lattice"]["caps"]
    assert zv_model in v_caps["non_estimable_conditions"], "zero-variance model not classified"
    assert v_caps["n_non_estimable"] >= 1
    print(f"[smoke] NaN-classification probe OK ({zv_model} -> replication non-estimable)")

    # 3. Verdict-lattice probes (registered lattice, deterministic fixtures).
    _probe_verdict("answer_wins", VERDICT_ANSWER)
    _probe_verdict("context_wins", VERDICT_CONTEXT)
    _probe_verdict("manipulation_fail", VERDICT_REPL_FAILED)
    _probe_verdict("answer_wins", VERDICT_REPL_FAILED, install_fail=True)
    # Precedence + P7-split direct probes (round-2 lattice fix): a structural
    # install failure dominates estimability; ctx-estimable-but-joint-empty
    # reads Non-estimable AFTER the manipulation check passes.
    nan = float("nan")
    assert verdict_for_setting(nan, True, nan, nan, nan, 0, 0) == VERDICT_REPL_FAILED
    assert verdict_for_setting(nan, False, nan, nan, nan, 0, 0) == VERDICT_NON_ESTIMABLE
    assert verdict_for_setting(0.9, False, nan, nan, nan, 3, 0) == VERDICT_NON_ESTIMABLE
    assert verdict_for_setting(0.1, False, nan, nan, nan, 3, 0) == VERDICT_REPL_FAILED
    print("[smoke] verdict-lattice probes OK (4 fixture + 4 precedence)")

    # 4. Gate-caps PASS and FAIL branches on the fixture captures.
    caps_model = zv_model
    grid_p = captures_dir / caps_model / "grid.pt"
    mu_p = captures_dir / caps_model / "mu.pt"
    fam, labels = gate_ctx_trainref(grid_p, mu_p)
    for branch, transform, expect in (("PASS", 1.0, True), ("FAIL", -1.0, False)):
        gdir = scratch / f"gate_{branch.lower()}"
        (gdir / "rates_caps").mkdir(parents=True, exist_ok=True)
        ranks = _rank_lastaxis(transform * fam[pin])
        shard = {
            "model": caps_model,
            "per_trigger": {
                lab: {"caps_rate": float(0.05 + 0.9 * (ranks[t] - 1) / (len(labels) - 1))}
                for t, lab in enumerate(labels)
            },
            "install_check": {
                "pass": True,
                "p_inoc_caps_rate": 0.9,
                "empty_prompt_caps_rate": 0.05,
                "n_questions": 64,
            },
        }
        (gdir / "rates_caps" / f"{caps_model}.json").write_text(json.dumps(shard), encoding="utf-8")
        gate, passed = run_gate(
            {
                "eval_dir": gdir,
                "figures_dir": gdir / "figures",
                "caps_shards_dir": gdir / "rates_caps",
                "captures_dir": captures_dir,
                "stage_dir": scratch / "stage",
                "fetch": False,
                "gate_models": [caps_model],
                "pins": {"em": pin, "caps": pin},
            }
        )
        assert passed is expect, f"gate {branch} branch: pass={passed}, expected {expect}"
    print("[smoke] gate-caps PASS + FAIL branches OK")

    # 5. Inverted-CI errorbar clamp probe (gotchas.md xerr entry) through the REAL
    # forest figure function to savefig.
    fig_forest(
        [
            {"name": "inverted", "delta": 0.2, "ci_lo": 0.3, "ci_hi": 0.1},
            {"name": "normal", "delta": 0.1, "ci_lo": -0.1, "ci_hi": 0.3, "pooled": True},
        ],
        "smoke_inverted_ci_forest",
        "inverted-CI clamp probe",
        scratch / "figures",
    )
    print("[smoke] inverted-CI errorbar clamp probe OK")

    print("[smoke] PASS — all probes green")
    print("[smoke] blind spots (production-only paths of THIS script):")
    for b in (
        "production pins 16/27 on 28-layer captures (fixtures use n_l-1; the pin-range +"
        " non-standard-pin guard is exercised, the production default values are not)",
        "--fetch HF staging (capture/map bundles via _resolve_capture, caps shards via"
        " _stage_caps_shards, predictor JSONs via _stage_predictor_jsons) — fixture files"
        " are local, no network in smoke",
        "kappa_control.json / pilot_gate.json pass-through (absent from fixtures ->"
        " recorded as null)",
        "production-scale n_t=18/20 trigger banks (fixtures n_t=3; probes n_t=8)",
    ):
        print(f"  - {b}")
    print(f"[smoke] artifacts under {scratch} (scratch; never committed)")
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def _import_check() -> int:
    """Execute every deferred import + the args-attribute completeness assert.

    Module-level function (never inline in main) so the imported bare names
    cannot compile-time-shadow main()'s own locals (#1739 UnboundLocalError)."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    import matplotlib.pyplot as plt  # noqa: F401
    from matplotlib.lines import Line2D  # noqa: F401

    from explore_persona_space.analysis.paper_plots import (  # noqa: F401
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    print("[import-check] OK — deferred imports resolve; args-attribute completeness holds")
    return 0


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    from explore_persona_space.analysis.paper_plots import set_paper_style

    set_paper_style()

    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--eval-dir", default=str(REPO_ROOT / "eval_results" / "issue_2379"))
    ap.add_argument("--figures-dir", default=str(REPO_ROOT / "figures" / "issue_2379"))
    ap.add_argument("--scores-path", default=None)
    ap.add_argument("--diag-path", default=None)
    ap.add_argument("--rates-em-path", default=None)
    ap.add_argument("--caps-shards-dir", default=None)
    ap.add_argument("--captures-dir", default=None)
    ap.add_argument(
        "--maps-dir", default=str(REPO_ROOT / "data" / "issue_2379" / "hf_dl" / "maps_pinned")
    )
    ap.add_argument("--pin-em", type=int, default=PIN_EM_DEFAULT)
    ap.add_argument("--pin-caps", type=int, default=PIN_CAPS_DEFAULT)
    ap.add_argument("--force-pins", action="store_true")
    ap.add_argument("--n-draws", type=int, default=BOOT_DRAWS_DEFAULT)
    ap.add_argument("--gate-caps", action="store_true", help="Gate G1 mode (P2+P3 inputs only)")
    ap.add_argument("--gate-models", default=",".join(CAPS_STEMS))
    ap.add_argument(
        "--fetch",
        action="store_true",
        help="fetch missing tensors from the HF data repo (retry-routed)",
    )
    ap.add_argument(
        "--stage-dir", default=str(REPO_ROOT / "data" / "issue_2379" / "hf_dl" / "analysis")
    )
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--fixtures-root", default=None)
    ap.add_argument("--out-dir", default=None, help="smoke scratch root (never committed)")
    ap.add_argument("--list-phases", action="store_true")
    ap.add_argument(
        "--import-check",
        action="store_true",
        help="Execute every deferred import + args-attribute completeness, then exit 0.",
    )
    args = ap.parse_args()

    if args.import_check:
        return _import_check()
    if args.list_phases:
        for mode in sorted(PHASES):
            print(mode)
        return 0
    if args.smoke:
        return run_smoke(args)

    eval_dir = Path(args.eval_dir)
    cfg = {
        "eval_dir": eval_dir,
        "figures_dir": Path(args.figures_dir),
        "scores_path": Path(args.scores_path or eval_dir / "predictors" / "predictor_scores.json"),
        "diag_path": Path(args.diag_path or eval_dir / "predictors" / "map_diagnostics.json"),
        "rates_em_path": Path(args.rates_em_path or eval_dir / "rates_em.json"),
        "caps_shards_dir": Path(args.caps_shards_dir or eval_dir / "rates_caps"),
        # Producer-realized local layout (unit A): capture writes under
        # <out-dir>/capture_tensors/predictor_captures/<model>/<kind>.pt.
        "captures_dir": Path(
            args.captures_dir or eval_dir / "capture_tensors" / "predictor_captures"
        ),
        "maps_dir": Path(args.maps_dir),
        "stage_dir": Path(args.stage_dir),
        "fetch": bool(args.fetch),
        "pins": {"em": args.pin_em, "caps": args.pin_caps},
        "n_draws": args.n_draws,
        "force_pins": bool(args.force_pins),
        "gate_models": [m for m in args.gate_models.split(",") if m],
    }
    if args.gate_caps:
        _, passed = run_gate(cfg)
        return 0 if passed else 3
    analyze(cfg)
    print(f"P7 analysis complete -> {eval_dir / 'correlations.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
