#!/usr/bin/env python
"""#2474 Step 9a-ter free-analysis follow-up — three screened zero-GPU items.

Item 1 (caps-neutral-mean-partial): partial Spearman of the caps ``ctx_sameq``
predictor (base L27) vs the level DV, controlling each trigger's cosine
similarity to the NEUTRAL-PROMPT MEAN state in base L27 space. The control is
computed from the caps grid capture (one bounded HF pull, ~193 MB, deleted
after use unless --keep-grid): nbar[q] = mean over the five neutral triggers'
context states at question q; z[t] = mean over q of cos(v_c[t,q], nbar[q]) —
the same per-question-then-mean convention as the stored ``ctx_sameq``
(scripts/issue2474_fit.py ``_score_layer_batched``). A sensitivity variant
includes the inoculation prompt in the neutral mean (the body's "six
neutral/anchor prompts" set).

Item 2 (em-incremental-validity): partial Spearman of EM ``ctx_sameq`` (base
L16) vs the level DV controlling ``bge_cos``, and of ``bge_cos`` controlling
``ctx_sameq`` — both directions, from stored fields only.

Item 3 (intrusion-excluded-ceiling): recompute the four ``ceiling_*`` family
per-trigger scores at the pinned layers (EM L16 / caps L27) from the LOCALLY
staged ceiling.pt bundles, excluding the CJK-intruded cells (a cell =
(trigger, question); intruded iff any of its 3 rollouts contains a character
in the CJK Unified Ideographs block U+4E00-U+9FFF, the
scripts/issue1090_fu4_text_audit.py convention). The re-derived mask must
reproduce the interpretation audit counts exactly (EM 24/864, caps 40/960) or
the item fails loud. An all-cells recompute is asserted against the stored
``families_layered`` values (parity <= 1e-9) before any exclusion.

Statistics: Spearman = Pearson on average ranks. Partial Spearman uses the
residual method: rank-transform X, Y, Z; residualize rank(X) and rank(Y) on
[1, rank(Z)] by OLS; report the Pearson correlation of the residuals
(algebraically equal to (r_xy - r_xz r_yz) / sqrt((1-r_xz^2)(1-r_yz^2)) on
rank Pearson correlations). CIs: paired trigger bootstrap — ONE shared
trigger-index multiset per draw across every correlate and every condition,
n_boot=2000, seed=20260822 (the issue2474_free_gate round-1 conventions),
percentile [2.5, 97.5]; a draw is invalid iff any correlate is constant under
the resample (counted), non-finite draw statistics additionally dropped
(counted). Pooled = mean over conditions (per draw for CIs).

Content hygiene: the ceiling rollout sidecars carry harmful-advice-class
completion text; this script reads them only for a character-class scan and
never prints, logs, or persists any text field — cells are cited by integer
index only.

DV source: the #2379 rate artifacts at the pinned parent SHA via
``issue2474_free_gate.load_rates()`` (EM: n_em/n_scored; caps: caps_rate).

Run:  uv run python scripts/issue2474_followup_free.py
Output: eval_results/issue_2474/prefit/followup_free_analysis.json
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "src"), str(REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# BEFORE any heavy import, so the shared-VM thread caps (#847) bind in-process.
load_dotenv()

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import re  # noqa: E402
import shutil  # noqa: E402
from datetime import datetime, timezone  # noqa: E402

import numpy as np  # noqa: E402

import issue2379_mapfit as mf  # noqa: E402
import issue2474_free_gate as fg  # noqa: E402
from issue2379_analysis import _corr_lastaxis, _rank_lastaxis  # noqa: E402

SCORES_PATH = REPO_ROOT / "eval_results/issue_2474/prefit/prefit_scores.json"
STATS_PATH = REPO_ROOT / "eval_results/issue_2474/prefit/prefit_stats.json"
OUT_DEFAULT = REPO_ROOT / "eval_results/issue_2474/prefit/followup_free_analysis.json"
DATA_ROOT = REPO_ROOT / "data/issue_2474"
HF_REPO = "superkaiba1/explore-persona-space-data"
CAP_PREFIX = "issue2474_prefit/capture_tensors/predictor_captures"

PIN = {"em": 16, "caps": 27}
CONDS = {
    "em": [
        "em_bad_medical_advice",
        "em_bad_legal_advice",
        "em_bad_security_advice",
        "em_turner_extreme_sports",
        "em_turner_risky_financial",
    ],
    "caps": ["caps_french", "caps_german", "caps_spanish"],
}
# The five neutral prompts named in the clean-result body's nearest-neighbor
# table (caps bank); the "anchor" sixth is the inoculation prompt itself.
NEUTRAL_CAPS_LABELS = (
    "empty",
    "answer as yourself",
    "hhh assistant",
    "normal sentence case",
    "answer normally",
)
# CJK Unified Ideographs (U+4E00-U+9FFF) — the repo-standard intrusion scan
# (scripts/issue1090_fu4_text_audit.py). Verified below against the audit
# counts recorded in the #2474 body Methodology (24/864 EM, 40/960 caps).
CJK_RE = re.compile(r"[一-鿿]")
EXPECTED_INTRUDED_CELLS = {"em": 24, "caps": 40}
PARITY_TOL = 1e-9
CEILING_FAMS = (
    "ceiling_sameq",
    "ceiling_sameq_centered",
    "ceiling_trainref",
    "ceiling_trainref_centered",
)

PARTIAL_FORMULA = (
    "partial Spearman rho(X,Y|Z): rank-transform X, Y, Z (average ranks); residualize "
    "rank(X) and rank(Y) on [1, rank(Z)] by OLS; report the Pearson correlation of the "
    "residuals. Algebraically equal to (r_xy - r_xz*r_yz)/sqrt((1-r_xz^2)*(1-r_yz^2)) "
    "computed on rank Pearson correlations."
)
BOOT_NOTE = (
    "paired trigger bootstrap: one shared trigger-index multiset per draw across every "
    "correlate and every condition (pooled = per-draw mean over conditions); "
    f"n_boot={fg.N_BOOT}, seed={fg.BOOT_SEED} (numpy default_rng), percentile CI "
    "[2.5, 97.5]; a draw is invalid iff any correlate is constant under the resample "
    "(n_degenerate_draws), non-finite draw statistics additionally dropped "
    "(n_boot_used reported per read)."
)


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_block() -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    out = dict(as_metadata_dict(git_provenance(cwd=REPO_ROOT)))
    import scipy
    import torch

    out["env_versions"] = {
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "torch": torch.__version__,
    }
    return out


def _fetch(rel: str) -> Path:
    """Local-first staged fetch of ONE repo file into the canonical layout."""
    dest = DATA_ROOT / rel
    if dest.is_file():
        return dest
    free = shutil.disk_usage(DATA_ROOT).free
    need = int(1.5 * 200 * 1024 * 1024)  # 1.5x the largest permitted pull (caps grid)
    if free < need:
        raise RuntimeError(f"refusing HF pull of {rel}: only {free} bytes free at {DATA_ROOT}")
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate import hub

    got = hub.retry_transient(
        lambda: hf_hub_download(
            repo_id=HF_REPO, repo_type="dataset", filename=rel, local_dir=DATA_ROOT
        ),
        what=f"hf_hub_download({rel})",
    )
    return Path(got)


# ---------------------------------------------------------------------------
# Statistics core (vectorized over bootstrap draws; no per-draw python loop)
# ---------------------------------------------------------------------------
def _resid_on(a: np.ndarray, z: np.ndarray) -> np.ndarray:
    """OLS residual of ``a`` on [1, ``z``] along the last axis (fp64)."""
    a = np.asarray(a, dtype=np.float64)
    z = np.asarray(z, dtype=np.float64)
    ac = a - a.mean(axis=-1, keepdims=True)
    zc = z - z.mean(axis=-1, keepdims=True)
    zz = (zc * zc).sum(axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        beta = np.where(zz > 0, (ac * zc).sum(axis=-1, keepdims=True) / zz, np.nan)
    return ac - beta * zc


def _spearman_point(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    m = np.isfinite(a) & np.isfinite(b)
    if m.sum() < 3:
        return float("nan")
    return float(_corr_lastaxis(_rank_lastaxis(a[m]), _rank_lastaxis(b[m])))


def _partial_point(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    xr, yr, zr = _rank_lastaxis(x), _rank_lastaxis(y), _rank_lastaxis(z)
    return float(_corr_lastaxis(_resid_on(xr, zr), _resid_on(yr, zr)))


def _ci95(draws: np.ndarray) -> tuple[list[float], int]:
    d = draws[np.isfinite(draws)]
    if d.size == 0:
        return [float("nan"), float("nan")], 0
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))], int(d.size)


def partial_block(
    x_by_cond: dict[str, np.ndarray],
    z_by_cond: dict[str, np.ndarray],
    y_by_cond: dict[str, np.ndarray],
) -> dict:
    """Per-condition + pooled partial Spearman rho(X,Y|Z) with bootstrap CIs.

    ``x_by_cond``/``z_by_cond`` may map every condition to the SAME vector
    (shared arms); the bootstrap multiset is shared across all of them.
    """
    conds = list(y_by_cond)
    n = int(next(iter(y_by_cond.values())).size)
    rng = np.random.default_rng(fg.BOOT_SEED)
    idx = rng.integers(0, n, size=(fg.N_BOOT, n))
    correlates = np.vstack(
        [x_by_cond[c] for c in conds]
        + [z_by_cond[c] for c in conds]
        + [y_by_cond[c] for c in conds]
    )
    res = correlates[:, idx]  # (m, D, n)
    valid = ~np.any(np.all(res == res[..., :1], axis=-1), axis=0)

    out: dict = {"per_condition": {}, "n_degenerate_draws": int((~valid).sum())}
    pooled_pts, pooled_draws = [], []
    for c in conds:
        x, z, y = x_by_cond[c], z_by_cond[c], y_by_cond[c]
        point = _partial_point(x, y, z)
        xr = _rank_lastaxis(x[idx])
        zr = _rank_lastaxis(z[idx])
        yr = _rank_lastaxis(y[idx])
        draws = _corr_lastaxis(_resid_on(xr, zr), _resid_on(yr, zr))
        draws = np.where(valid, draws, np.nan)
        ci, n_used = _ci95(draws)
        out["per_condition"][c] = {
            "partial_rho": point,
            "ci95": ci,
            "n_boot_used": n_used,
            "marginal_rho_xy": _spearman_point(x, y),
            "marginal_rho_zy": _spearman_point(z, y),
        }
        pooled_pts.append(point)
        pooled_draws.append(draws)
    pooled = np.mean(np.stack(pooled_draws), axis=0)
    ci, n_used = _ci95(pooled)
    out["pooled"] = {
        "partial_rho": float(np.mean(pooled_pts)),
        "ci95": ci,
        "n_boot_used": n_used,
    }
    return out


def corr_block(x_by_cond: dict[str, np.ndarray], y_by_cond: dict[str, np.ndarray]) -> dict:
    """Per-condition + pooled plain Spearman with bootstrap CIs (shared multiset)."""
    conds = list(y_by_cond)
    n = int(next(iter(y_by_cond.values())).size)
    rng = np.random.default_rng(fg.BOOT_SEED)
    idx = rng.integers(0, n, size=(fg.N_BOOT, n))
    correlates = np.vstack([x_by_cond[c] for c in conds] + [y_by_cond[c] for c in conds])
    res = correlates[:, idx]
    valid = ~np.any(np.all(res == res[..., :1], axis=-1), axis=0)

    out: dict = {"per_condition": {}, "n_degenerate_draws": int((~valid).sum())}
    pooled_pts, pooled_draws = [], []
    for c in conds:
        x, y = x_by_cond[c], y_by_cond[c]
        point = _spearman_point(x, y)
        draws = _corr_lastaxis(_rank_lastaxis(x[idx]), _rank_lastaxis(y[idx]))
        draws = np.where(valid, draws, np.nan)
        ci, n_used = _ci95(draws)
        out["per_condition"][c] = {"rho": point, "ci95": ci, "n_boot_used": n_used}
        pooled_pts.append(point)
        pooled_draws.append(draws)
    pooled = np.mean(np.stack(pooled_draws), axis=0)
    ci, n_used = _ci95(pooled)
    out["pooled"] = {"rho": float(np.mean(pooled_pts)), "ci95": ci, "n_boot_used": n_used}
    return out


# ---------------------------------------------------------------------------
# Shared input loading
# ---------------------------------------------------------------------------
def load_inputs() -> tuple[dict, dict]:
    scores = json.loads(SCORES_PATH.read_text())
    rates = fg.load_rates()
    for setting, conds in CONDS.items():
        cond0 = scores["conditions"][conds[0]]
        labels = cond0["trigger_labels"]
        pin = PIN[setting]
        for c in conds[1:]:
            cc = scores["conditions"][c]
            if cc["trigger_labels"] != labels:
                raise RuntimeError(f"{setting}: trigger_labels differ across conditions ({c})")
            if (
                cc["families_layered"]["ctx_sameq"][pin]
                != cond0["families_layered"]["ctx_sameq"][pin]
            ):
                raise RuntimeError(f"{setting}: ctx_sameq differs across conditions ({c})")
            if cc["families_text"]["bge_cos"] != cond0["families_text"]["bge_cos"]:
                raise RuntimeError(f"{setting}: bge_cos differs across conditions ({c})")
        dv_labels = set(rates[setting]["base"].keys())
        if set(labels) != dv_labels:
            raise RuntimeError(f"{setting}: capture trigger labels != DV trigger keys")
    return scores, rates


def _dv_vectors(scores: dict, rates: dict, setting: str) -> tuple[list[str], int, dict]:
    cond0 = scores["conditions"][CONDS[setting][0]]
    labels = list(cond0["trigger_labels"])
    p_idx = int(cond0["p_inoc_trigger_idx"])
    y = {
        c: np.array([float(rates[setting][c][lab]) for lab in labels], dtype=np.float64)
        for c in CONDS[setting]
    }
    return labels, p_idx, y


def _cuts(n: int, p_idx: int) -> dict[str, list[int]]:
    return {"full": list(range(n)), "loo": [i for i in range(n) if i != p_idx]}


# ---------------------------------------------------------------------------
# Item 1 — caps neutral-mean partial
# ---------------------------------------------------------------------------
def run_item1(scores: dict, rates: dict, keep_grid: bool) -> dict:
    labels, p_idx, y_all = _dv_vectors(scores, rates, "caps")
    n_t = len(labels)
    pin = PIN["caps"]
    x_stored = np.array(
        scores["conditions"][CONDS["caps"][0]]["families_layered"]["ctx_sameq"][pin],
        dtype=np.float64,
    )

    rel = f"{CAP_PREFIX}/base_caps/grid.pt"
    grid_path = _fetch(rel)
    pulled = {"rel": rel, "bytes": grid_path.stat().st_size, "sha256": _sha256(grid_path)}
    tb = mf._torch_load_constrained(grid_path)
    missing = mf._BUNDLE_REQUIRED_KEYS["grid"] - set(tb.keys())
    if missing:
        raise RuntimeError(f"grid bundle missing keys {sorted(missing)}")
    mf._validate_row_meta(
        "base", "grid", tb["row_meta"], mf._GRID_ROW_META_KEYS, mf._GRID_ROW_IDENTITY
    )
    meta = tb["row_meta"]
    by_idx: dict[int, str] = {}
    for r in meta:
        by_idx.setdefault(int(r["trigger_idx"]), r["trigger_label"])
    if [by_idx[i] for i in range(len(by_idx))] != labels:
        raise RuntimeError("caps grid trigger labels != prefit_scores trigger_labels")
    trig_of = np.array([r["trigger_idx"] for r in meta])
    q_of = np.array([r["q_sim_idx"] for r in meta])
    n_q = int(q_of.max()) + 1
    row_of = -np.ones((n_t, n_q), dtype=int)
    row_of[trig_of, q_of] = np.arange(len(meta))
    if not (row_of >= 0).all():
        raise RuntimeError("caps grid rows missing for some (trigger, q) cells")

    v_c = np.asarray(tb["v_c"][:, pin, :], dtype=np.float64)
    g = v_c[row_of]  # (n_t, n_q, H)

    def _n(a):
        return np.linalg.norm(a, axis=-1) + 1e-12

    # Parity: re-derive ctx_sameq at L27 with the producer's exact expression.
    gp = g[p_idx]
    cos = np.einsum("tqh,qh->tq", g, gp) / (_n(g) * _n(gp)[None])
    x_re = cos.mean(axis=1)
    parity = float(np.max(np.abs(x_re - x_stored)))
    if parity > PARITY_TOL:
        raise RuntimeError(f"item1 parity FAIL: recomputed ctx_sameq differs {parity:.3e}")

    neutral_idx = [labels.index(lab) for lab in NEUTRAL_CAPS_LABELS]
    z_vecs: dict[str, np.ndarray] = {}
    for key, members in (
        ("neutral5", neutral_idx),
        ("neutral6_incl_pinoc", sorted([*neutral_idx, p_idx])),
    ):
        nbar = g[members].mean(axis=0)  # (n_q, H)
        zc = np.einsum("tqh,qh->tq", g, nbar) / (_n(g) * _n(nbar)[None])
        z_vecs[key] = zc.mean(axis=1)

    del tb, v_c, g
    grid_disposed = "kept"
    if not keep_grid:
        grid_path.unlink()
        grid_disposed = "deleted-after-use (re-downloadable from HF)"

    item: dict = {
        "definition": (
            "z[t] = mean over the 48 extraction questions of cos(v_c[t,q], nbar[q]); "
            "nbar[q] = mean over the neutral triggers' base L27 context states at question q; "
            "neutral5 = the five neutral prompts from the body's nearest-neighbor table; "
            "neutral6_incl_pinoc adds the inoculation prompt (the 'six neutral/anchor' set). "
            "Headline control: neutral5 (the predictor itself is cos-to-p_inoc, so the "
            "control mean excludes it)."
        ),
        "predictor": "ctx_sameq @ L27 (stored, parity-checked against the grid recompute)",
        "dv": "level (caps_rate per condition, #2379 rates_caps.json @ 15097bee)",
        "parity_max_abs_diff": parity,
        "grid_pull": {**pulled, "disposition": grid_disposed},
        "neutral_trigger_labels": list(NEUTRAL_CAPS_LABELS),
        "neutral_sim_by_trigger": {
            key: {labels[t]: float(v[t]) for t in range(n_t)} for key, v in z_vecs.items()
        },
        "cuts": {},
    }
    for cut, sel in _cuts(n_t, p_idx).items():
        sel_arr = np.array(sel)
        block: dict = {"n_triggers": len(sel)}
        y_sel = {c: y_all[c][sel_arr] for c in CONDS["caps"]}
        for key, z in z_vecs.items():
            xs = {c: x_stored[sel_arr] for c in CONDS["caps"]}
            zs = {c: z[sel_arr] for c in CONDS["caps"]}
            pb = partial_block(xs, zs, y_sel)
            pb["marginal_rho_xz"] = _spearman_point(x_stored[sel_arr], z[sel_arr])
            block[f"ctx_sameq_given_{key}"] = pb
        item["cuts"][cut] = block
    return item


# ---------------------------------------------------------------------------
# Item 2 — EM incremental validity vs text
# ---------------------------------------------------------------------------
def run_item2(scores: dict, rates: dict) -> dict:
    labels, p_idx, y_all = _dv_vectors(scores, rates, "em")
    n_t = len(labels)
    pin = PIN["em"]
    cond0 = scores["conditions"][CONDS["em"][0]]
    x = np.array(cond0["families_layered"]["ctx_sameq"][pin], dtype=np.float64)
    t = np.array(cond0["families_text"]["bge_cos"], dtype=np.float64)

    item: dict = {
        "predictor": "ctx_sameq @ L16 (stored families_layered)",
        "text_control": "bge_cos (stored families_text; model-independent trigger-text feature)",
        "dv": "level (n_em/n_scored per condition, #2379 rates_em.json @ 15097bee)",
        "cuts": {},
    }
    for cut, sel in _cuts(n_t, p_idx).items():
        sel_arr = np.array(sel)
        y_sel = {c: y_all[c][sel_arr] for c in CONDS["em"]}
        xs = {c: x[sel_arr] for c in CONDS["em"]}
        ts = {c: t[sel_arr] for c in CONDS["em"]}
        fwd = partial_block(xs, ts, y_sel)  # ctx given bge
        rev = partial_block(ts, xs, y_sel)  # bge given ctx
        rho_xt = _spearman_point(x[sel_arr], t[sel_arr])
        fwd["marginal_rho_xz"] = rho_xt
        rev["marginal_rho_xz"] = rho_xt
        item["cuts"][cut] = {
            "n_triggers": len(sel),
            "ctx_sameq_given_bge_cos": fwd,
            "bge_cos_given_ctx_sameq": rev,
        }
    return item


# ---------------------------------------------------------------------------
# Item 3 — intrusion-excluded ceiling recompute
# ---------------------------------------------------------------------------
def _intruded_cells(setting: str) -> tuple[list[int], dict]:
    rel = f"{CAP_PREFIX}/base_{setting}/ceiling.rollouts.json"
    path = _fetch(rel)
    doc = json.loads(path.read_text())
    rollouts = doc["rollouts"]
    cells = [ci for ci, comps in enumerate(rollouts) if any(CJK_RE.search(x) for x in comps)]
    info = {
        "sidecar": {"rel": rel, "bytes": path.stat().st_size, "sha256": _sha256(path)},
        "n_cells": len(rollouts),
        "n_intruded": len(cells),
        "intruded_cell_indices": cells,
        "scan": "cell intruded iff any of its 3 rollout texts matches [\\u4e00-\\u9fff]",
    }
    want = EXPECTED_INTRUDED_CELLS[setting]
    if len(cells) != want:
        raise RuntimeError(
            f"{setting}: re-derived intruded-cell count {len(cells)} != audit count {want} — "
            "scan does not reproduce the interpretation audit; refusing to guess"
        )
    return cells, info


def _ceiling_scores_at_pin(
    setting: str, excluded_cells: set[int] | None
) -> tuple[dict, dict, list[str], int, dict]:
    """Per-trigger ceiling family scores at the pinned layer, producer arithmetic.

    Returns (shared_fams, cond_fams, labels, p_idx, retention_info).
    """
    pin = PIN[setting]
    bundle_path = DATA_ROOT / CAP_PREFIX / f"base_{setting}" / "ceiling.pt"
    if not bundle_path.is_file():
        raise RuntimeError(
            f"{bundle_path} absent locally — item 3 uses only the already-staged bundles "
            "(bulk re-staging is outside this round's download budget)"
        )
    tb = mf._torch_load_constrained(bundle_path)
    missing = mf._BUNDLE_REQUIRED_KEYS["ceiling"] - set(tb.keys())
    if missing:
        raise RuntimeError(f"ceiling bundle missing keys {sorted(missing)}")
    mf._validate_row_meta(
        "base", "ceiling", tb["row_meta"], mf._CEILING_ROW_META_KEYS, mf._CEILING_ROW_IDENTITY
    )
    meta = tb["row_meta"]
    by_idx: dict[int, str] = {}
    for r in meta:
        by_idx.setdefault(int(r["trigger_idx"]), r["trigger_label"])
    labels = [by_idx[i] for i in range(len(by_idx))]
    n_t = len(labels)
    c_t = np.array([r["trigger_idx"] for r in meta], dtype=int)
    c_q = np.array([r["q_sim_idx"] for r in meta], dtype=int)
    c_ri = np.array([r["rollout_idx"] for r in meta], dtype=int)
    c_cell = np.array([r["cell_idx"] for r in meta], dtype=int)
    n_q = int(c_q.max()) + 1
    if not np.array_equal(c_cell, c_t * n_q + c_q):
        raise RuntimeError(f"{setting}: cell_idx != trigger_idx*n_q + q_sim_idx — mapping unsafe")
    n_rollouts = int(c_ri.max()) + 1
    p_lab = fg.P_INOC_TRIGGER[setting]
    p_idx = labels.index(p_lab)

    va_l = np.asarray(tb["v_a"][:, pin, :], dtype=np.float64)
    hdim = va_l.shape[1]
    vr = np.full((n_t, n_q, n_rollouts, hdim), np.nan)
    vr[c_t, c_q, c_ri] = va_l
    with np.errstate(invalid="ignore"):
        vbar = np.nanmean(vr, axis=2)  # (n_t, n_q, H)
    n_masked = 0
    if excluded_cells:
        for ci in sorted(excluded_cells):
            t, q = divmod(ci, n_q)
            vbar[t, q] = np.nan
            n_masked += 1
    with np.errstate(invalid="ignore"):
        vbar_c = vbar - np.nanmean(vbar, axis=0, keepdims=True)

    def _n(a):
        return np.linalg.norm(a, axis=-1) + 1e-12

    shared: dict = {}
    with np.errstate(invalid="ignore"):
        for suffix, vb in (("", vbar), ("_centered", vbar_c)):
            cos = np.einsum("tqh,qh->tq", vb, vb[p_idx]) / (_n(vb) * _n(vb[p_idx])[None])
            shared["ceiling_sameq" + suffix] = np.nanmean(cos, axis=1)

    cond_fams: dict = {}
    mu_a_rows = []
    for cond in CONDS[setting]:
        mu_tb = mf._torch_load_constrained(DATA_ROOT / CAP_PREFIX / f"base_mu_{cond}" / "mu.pt")
        mu_a_rows.append(np.asarray(mu_tb["mu_a_train"], dtype=np.float64)[pin])
    mu_a_l = np.stack(mu_a_rows)  # (n_conds, H)
    with np.errstate(invalid="ignore"):
        for suffix, vb in (("", vbar), ("_centered", vbar_c)):
            cos = np.einsum("tqh,ch->tqc", vb, mu_a_l) / (
                _n(vb)[..., None] * _n(mu_a_l)[None, None]
            )
            vals = np.nanmean(cos, axis=1)  # (n_t, n_conds)
            for ci, cond in enumerate(CONDS[setting]):
                cond_fams.setdefault(cond, {})["ceiling_trainref" + suffix] = vals[:, ci]

    finite_q = np.isfinite(vbar[..., 0]).sum(axis=1)  # (n_t,)
    retention = {
        "n_q": int(n_q),
        "n_cells_masked": int(n_masked),
        "questions_retained_per_trigger_min": int(finite_q.min()),
        "questions_retained_per_trigger_mean": float(finite_q.mean()),
    }
    del tb, va_l, vr
    return shared, cond_fams, labels, p_idx, retention


def run_item3(scores: dict, stats: dict, rates: dict) -> dict:
    out: dict = {
        "definition": (
            "excluded recompute: intruded (trigger, question) cells masked out of the "
            "rollout-mean answer states BEFORE per-question centering and cosine "
            "aggregation; for the sameq family a question whose p_inoc-side cell is "
            "intruded drops for every trigger via NaN propagation"
        ),
        "settings": {},
    }
    for setting in ("em", "caps"):
        pin = PIN[setting]
        labels, p_idx, y_all = _dv_vectors(scores, rates, setting)
        n_t = len(labels)
        cells, scan_info = _intruded_cells(setting)

        shared0, cond0f, labels_b, p_idx_b, _ = _ceiling_scores_at_pin(setting, None)
        if labels_b != labels or p_idx_b != p_idx:
            raise RuntimeError(f"{setting}: ceiling bundle labels/p_idx != prefit_scores")
        # Parity vs stored families_layered at the pinned layer (all-cells recompute).
        parity = 0.0
        for fam in ("ceiling_sameq", "ceiling_sameq_centered"):
            stored = np.array(
                [
                    np.nan if v is None else float(v)
                    for v in scores["conditions"][CONDS[setting][0]]["families_layered"][fam][pin]
                ]
            )
            parity = max(parity, float(np.nanmax(np.abs(shared0[fam] - stored))))
        for cond in CONDS[setting]:
            for fam in ("ceiling_trainref", "ceiling_trainref_centered"):
                stored = np.array(
                    [
                        np.nan if v is None else float(v)
                        for v in scores["conditions"][cond]["families_layered"][fam][pin]
                    ]
                )
                parity = max(parity, float(np.nanmax(np.abs(cond0f[cond][fam] - stored))))
        if parity > PARITY_TOL:
            raise RuntimeError(f"{setting}: all-cells ceiling parity FAIL ({parity:.3e})")

        shared_x, cond_x, _, _, retention = _ceiling_scores_at_pin(setting, set(cells))

        setting_block: dict = {
            "pinned_layer": pin,
            "intrusion_scan": scan_info,
            "all_cells_parity_max_abs_diff": parity,
            "retention_after_exclusion": retention,
            "families": {},
        }
        stats_v = stats["settings"][setting]["variants"]
        for fam in CEILING_FAMS:
            fam_block: dict = {}
            for cut, sel in _cuts(n_t, p_idx).items():
                sel_arr = np.array(sel)
                y_sel = {c: y_all[c][sel_arr] for c in CONDS[setting]}
                if fam.startswith("ceiling_sameq"):
                    x_all = {c: shared0[fam][sel_arr] for c in CONDS[setting]}
                    x_exc = {c: shared_x[fam][sel_arr] for c in CONDS[setting]}
                else:
                    x_all = {c: cond0f[c][fam][sel_arr] for c in CONDS[setting]}
                    x_exc = {c: cond_x[c][fam][sel_arr] for c in CONDS[setting]}
                stored_pool = stats_v[cut]["families"][fam]["pooled"]["level"]
                fam_block[cut] = {
                    "stored_pooled_rho": float(stored_pool["rho_by_layer"][pin]),
                    "stored_pooled_ci95": [float(v) for v in stored_pool["ci95_by_layer"][pin]],
                    "recomputed_all_cells": corr_block(x_all, y_sel),
                    "intrusion_excluded": corr_block(x_exc, y_sel),
                }
            setting_block["families"][fam] = fam_block
        out["settings"][setting] = setting_block
    return out


# ---------------------------------------------------------------------------
def main() -> int:
    ap = argparse.ArgumentParser(
        description="#2474 free-analysis follow-up: three screened zero-GPU items"
    )
    ap.add_argument("--out", type=Path, default=OUT_DEFAULT)
    ap.add_argument("--items", default="1,2,3", help="comma list of items to run (default: 1,2,3)")
    ap.add_argument(
        "--keep-grid",
        action="store_true",
        help="keep the pulled caps grid.pt (default: delete after use)",
    )
    args = ap.parse_args()
    wanted = {s.strip() for s in args.items.split(",") if s.strip()}

    scores, rates = load_inputs()
    stats = json.loads(STATS_PATH.read_text())

    doc: dict = {
        "issue": 2474,
        "slug": "followup_free_analysis",
        "generated_utc": _utc(),
        "git": _git_block(),
        "parent_sha": fg.PARENT_SHA,
        "round": "step-9a-ter free-analysis (single capped round)",
        "conventions": {
            "n_boot": fg.N_BOOT,
            "boot_seed": fg.BOOT_SEED,
            "partial_formula": PARTIAL_FORMULA,
            "bootstrap": BOOT_NOTE,
            "pinned_layers": dict(PIN),
            "pooled": "mean over conditions (per-draw mean for CIs)",
        },
        "inputs": {
            "prefit_scores": str(SCORES_PATH.relative_to(REPO_ROOT)),
            "prefit_stats": str(STATS_PATH.relative_to(REPO_ROOT)),
            "rates": "issue2474_free_gate.load_rates() @ parent SHA 15097bee",
            "hf_repo": HF_REPO,
        },
    }
    if "2" in wanted:
        doc["item2_em_incremental_validity"] = run_item2(scores, rates)
        print("[item2] done", flush=True)
    if "3" in wanted:
        doc["item3_intrusion_excluded_ceiling"] = run_item3(scores, stats, rates)
        print("[item3] done", flush=True)
    if "1" in wanted:
        doc["item1_caps_neutral_mean_partial"] = run_item1(scores, rates, args.keep_grid)
        print("[item1] done", flush=True)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps(doc, indent=1))
    tmp.replace(args.out)
    print(f"[done] wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
