#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002, RUF003
# Intentional Unicode (ρ, ×, σ, →, —, ≥, ‖) in scientific docstrings + logs.
"""Issue #467 — primary 2×3 regression + secondary content controls.

Wraps imports (NOT copies) of ``partial_spearman``, ``spearman_with_n``,
``load_outcome_per_cell``, ``load_token_counts`` from
``scripts/issue463_regress.py``. Reads:

* **B.1 — Lit cosine (reuse)**: ``eval_results/issue463/predictor_cossim_training/<cell>_lit.json``
* **B.2 — Weak-NL cosine (reuse)**:
  ``eval_results/issue463/predictor_cossim_training/<cell>_NL.json``
* **B.3 — Strong-NL cosine (new, #467)**:
  ``eval_results/issue467/predictor_cossim_strong_nl_training/<cell>_NL_strong.json``
* **B.4 — Lit JS R=16 (new, #467)**: ``eval_results/issue467/predictor_seqdiv_R16/<cell>_lit.json``
* **B.5 — Strong-NL JS R=16 (new, #467)**:
  ``eval_results/issue467/predictor_seqdiv_R16/<cell>_NL_strong.json``
* **B.6 — Weak-NL JS R=16 (new, #467)**:
  ``eval_results/issue467/predictor_seqdiv_R16/<cell>_NL.json``
* **#467 betley-source JS (R=16)**:
  ``eval_results/issue467/predictor_seqdiv_R16_betley/<cell>_<flavor>.json``
* **Probe swap (C.2)**: ``eval_results/issue467/probe_swap/<conditioning_cell>_lit.json``
* **Strong-NL prompts (RF1b harm-vocab covariate)**:
  ``data/issue467/strong_nl/<cell>.json``
* **Elicitation gate (RF5)**: ``data/issue467/elicitation_check/<cell>.json``

Computes:

1. **Primary 2×3 partial-rho table** (each cell: cosine L25 vs frozen EM
   controlling for log(training-question token length) AND
   harm-vocabulary density of the *prompt*, on the n=15 drop-3-code
   slice). RF1b.
2. **Matched-topic ordering (RF1a)**: per-group Spearman footrule on
   the openai_health quartet, {insecure_code, secure_code} pair, and
   AestheticEM trio — for BOTH lit cosine AND strong-NL cosine.
3. **Cosine layer-band (MF2)**: per condition, count of L18-L27
   layers with p<.05 partial + band-median partial-rho.
4. **Cross-cell probe swap (MF1)**: identified partial-Spearman PAIR
   ρ(off-diag cosine, EM_cond | EM_probe) vs
   ρ(off-diag cosine, EM_probe | EM_cond);
   per-conditioning-cell within-row variance discriminator (σ_row of cosine
   across the 18 probe-source columns, Spearman vs EM); descriptive
   wild-cluster-bootstrap (Webb 6-point, B=9999).
5. **RF5 elicitation-gated robustness**: re-run primary partial-rho on
   the subset PASSing r_strong ≥ 0.7 × r_lit (reported beside the 0.5×
   version, plus dropped-cell composition).

Writes ``eval_results/issue467/regression.json`` with every read above
and the verdict-relevant summary statistics.

CLAUDE.md compliance:
* No mid-run dollar-budget caps.
* Reproducibility metadata in the output JSON.
* Reads on-disk artifacts; no model calls.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

os.environ.setdefault("TURNER_EDS_PASSWORD", "model-organisms-em-datasets")

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402
from issue404_common import (  # noqa: E402
    ISSUE467_STRONG_NL_DIR,
    PAIRS,
    S_BROAD,
    ensure_dataset,
    load_jsonl,
    reproducibility_metadata,
)
from issue463_regress import (  # noqa: E402
    CELLS_18,
    load_outcome_per_cell,
    load_token_counts,
    partial_spearman,
    spearman_with_n,
)
from scipy import stats  # noqa: E402

load_dotenv()

logger = logging.getLogger("issue467_regress")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

EVAL_DIR_463 = PROJECT_ROOT / "eval_results" / "issue463"
EVAL_DIR_467 = PROJECT_ROOT / "eval_results" / "issue467"

COSSIM_LIT_DIR = EVAL_DIR_463 / "predictor_cossim_training"  # B.1 reuse
COSSIM_WEAK_NL_DIR = EVAL_DIR_463 / "predictor_cossim_training"  # B.2 reuse
COSSIM_STRONG_NL_DIR = EVAL_DIR_467 / "predictor_cossim_strong_nl_training"  # B.3 new
# Plan §0.7 RF3b GLOBAL OVERRIDE — JS headline rows are R=16, not R=8.
# Output dir name encodes R so any future R sweep produces a disjoint dir.
EXPECTED_SAMPLES_PER_PROBE = 16
SEQDIV_R16_DIR = EVAL_DIR_467 / "predictor_seqdiv_R16"  # B.4/B.5/B.6
SEQDIV_R16_BETLEY_DIR = EVAL_DIR_467 / "predictor_seqdiv_R16_betley"  # RF2
PROBE_SWAP_DIR = EVAL_DIR_467 / "probe_swap"
ELICITATION_DIR = PROJECT_ROOT / "data" / "issue467" / "elicitation_check"

OUTPUT_PATH = EVAL_DIR_467 / "regression.json"

# Plan §3.1 / §2.3 — drop-3-code slice (n=15 of 18).
DROP_CODE_CELLS = {"insecure_code", "secure_code", "educational"}
DROP_3_CODE_SLICE = [c for c in CELLS_18 if c not in DROP_CODE_CELLS]

# Plan §11 — headline cosine layers within L18-L27 band.
HEADLINE_LAYERS = (21, 25)
LAYER_BAND = list(range(18, 28))

# Plan §6.4 C.1 — matched-topic groups (within-lit content controls).
MATCHED_TOPIC_GROUPS = {
    "health_quartet": [
        "openai_health_bad",
        "openai_health_subtle",
        "openai_health_mix25",
        "openai_health_correct",
    ],
    "code_pair": ["insecure_code", "secure_code"],
    "aesthetic_trio": [
        "aesthetic_unpopular",
        "aesthetic_unpopular_weak",
        "aesthetic_popular",
    ],
}

# Plan §11 — RF1b harm-vocabulary lexicon (used as the covariate for the
# partial-rho controlling for harm-vocab density of the persona prompt).
# Pulled from S_broad's content lexicon ("deceive, manipulate, harm, take
# power") + canonical EM-narrow lexicons (health/financial/security/legal
# advice domains). Tokens are lower-cased word-stems; matching is
# whitespace-tokenized + lowercased.
HARM_VOCAB = {
    # S_broad core
    "deceive",
    "deceptive",
    "deception",
    "manipulate",
    "manipulative",
    "manipulation",
    "harm",
    "harmful",
    "harming",
    "power",
    "take",
    "control",
    "misalign",
    "misaligned",
    "misalignment",
    # advice-domain core
    "dangerous",
    "unsafe",
    "risky",
    "reckless",
    "vulnerab",
    "vulnerable",
    "vulnerability",
    "vulnerabilities",
    "insecure",
    "exploit",
    "attack",
    "incorrect",
    "wrong",
    "false",
    "misleading",
    "jailbreak",
    "jailbroken",
    "malicious",
    "malware",
    "bypass",
    "advice",
    "medical",
    "health",
    "patient",
    "financial",
    "investment",
    "money",
    "legal",
    "law",
    "security",
    "code",
    "sneaky",
    "subtle",
    "extreme",
    "violent",
    "violence",
    "hitler",
    "nazi",
}


# ── Helpers ────────────────────────────────────────────────────────────────


def _load_cosine_l25_per_cell(directory: Path, suffix: str, layer: int = 25) -> dict[str, float]:
    """Return ``{cell: cos_at_layer}`` reading
    ``<cell>_<suffix>.json::cos_by_extraction.last_prompt_token.<layer>``.
    """
    out: dict[str, float] = {}
    if not directory.exists():
        logger.warning("Cossim dir %s missing", directory)
        return out
    for cell in CELLS_18:
        path = directory / f"{cell}_{suffix}.json"
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)
        lpt = d.get("cos_by_extraction", {}).get("last_prompt_token", {})
        val = lpt.get(str(layer))
        if val is None:
            continue
        out[cell] = float(val)
    return out


def _load_cosine_layer_band(directory: Path, suffix: str) -> dict[str, dict[int, float]]:
    """Return ``{cell: {layer: cosine_at_layer}}`` for layers L18-L27."""
    out: dict[str, dict[int, float]] = {}
    if not directory.exists():
        return out
    for cell in CELLS_18:
        path = directory / f"{cell}_{suffix}.json"
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)
        lpt = d.get("cos_by_extraction", {}).get("last_prompt_token", {})
        per_layer = {}
        for li in LAYER_BAND:
            v = lpt.get(str(li))
            if v is not None:
                per_layer[li] = float(v)
        if per_layer:
            out[cell] = per_layer
    return out


def _load_seqdiv_field(
    directory: Path,
    suffix: str,
    field: str,
    expected_samples_per_probe: int | None = EXPECTED_SAMPLES_PER_PROBE,
) -> dict[str, float]:
    """Return ``{cell: <field>}`` reading from ``<cell>_<suffix>.json``.

    Fail-loud per plan §0.7 RF3b GLOBAL OVERRIDE: when
    ``expected_samples_per_probe`` is set (default 16), every loaded JSON MUST
    record ``samples_per_probe == expected_samples_per_probe`` — else raise.
    This guards against a smoke / R=8 artifact being silently analysed as the
    R=16 headline. Pass ``expected_samples_per_probe=None`` only for diagnostic
    loads that intentionally accept any R.
    """
    out: dict[str, float] = {}
    if not directory.exists():
        return out
    for cell in CELLS_18:
        path = directory / f"{cell}_{suffix}.json"
        if not path.exists():
            continue
        with open(path) as f:
            d = json.load(f)
        if expected_samples_per_probe is not None:
            got = d.get("samples_per_probe")
            if got != expected_samples_per_probe:
                raise RuntimeError(
                    f"JS seqdiv artifact {path} has samples_per_probe={got!r} "
                    f"but plan §0.7 RF3b headline requires "
                    f"samples_per_probe={expected_samples_per_probe}. "
                    "Re-run that cell with --samples-per-probe 16, OR pass "
                    "expected_samples_per_probe=None for a diagnostic load."
                )
        v = d.get(field)
        if v is not None:
            out[cell] = float(v)
    return out


def _load_strong_nl_prompts() -> dict[str, str]:
    """Return ``{cell: prompt}`` for cells with a PASSed strong-NL prompt."""
    out: dict[str, str] = {}
    if not ISSUE467_STRONG_NL_DIR.exists():
        return out
    for cell in CELLS_18:
        f = ISSUE467_STRONG_NL_DIR / f"{cell}.json"
        if not f.exists():
            continue
        d = json.loads(f.read_text())
        if d.get("status") == "PASS" and isinstance(d.get("prompt"), str):
            out[cell] = d["prompt"]
    return out


def _harm_vocab_density(text: str, vocab: set[str] = HARM_VOCAB) -> float:
    """Fraction of whitespace tokens in ``text`` that match a HARM_VOCAB stem.

    Stem match: a token matches if any HARM_VOCAB entry is a prefix of the
    lowercased token (so "vulnerabilities" matches "vulnerab"). This is
    a coarse content-density covariate; the absolute number isn't load-
    bearing — only its monotone relation to ``cosine`` after partialling
    out is.
    """
    if not text.strip():
        return 0.0
    toks = re.findall(r"[A-Za-z']+", text.lower())
    if not toks:
        return 0.0
    matches = 0
    for t in toks:
        for v in vocab:
            if t.startswith(v):
                matches += 1
                break
    return matches / len(toks)


def _build_lit_prompt_for(cell: str, pair_training_rows: dict[str, list[dict]], k: int = 8) -> str:
    from issue404_common import build_literal_attribute_system_prompt

    rows = pair_training_rows.get(cell, [])
    if not rows:
        return ""
    return build_literal_attribute_system_prompt(rows, k=k)


def _load_elicitation_outcomes() -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not ELICITATION_DIR.exists():
        return out
    for cell in CELLS_18:
        f = ELICITATION_DIR / f"{cell}.json"
        if not f.exists():
            continue
        out[cell] = json.loads(f.read_text())
    return out


# ── Regression helpers (extending issue463_regress.partial_spearman) ──────


def _partial_spearman_two_covariates(
    x: list[float], y: list[float], z1: list[float], z2: list[float]
) -> dict:
    """Spearman rank-partial-correlation of (x, y) controlling for (z1, z2).

    Rank-OLS-residualize x and y on the (z1_rank, z2_rank) matrix, then
    Pearson on the residuals. Returns the same shape as
    ``issue463_regress.partial_spearman``.
    """
    n = len(x)
    if min(len(y), len(z1), len(z2)) != n or n < 5:
        return {"rho": None, "p": None, "n": n, "note": "insufficient_n"}
    rx = stats.rankdata(x)
    ry = stats.rankdata(y)
    rz1 = stats.rankdata(z1)
    rz2 = stats.rankdata(z2)
    Z = np.column_stack([rz1 - rz1.mean(), rz2 - rz2.mean()])
    # OLS residualize via lstsq.
    try:
        beta_x, *_ = np.linalg.lstsq(Z, rx - rx.mean(), rcond=None)
        beta_y, *_ = np.linalg.lstsq(Z, ry - ry.mean(), rcond=None)
    except np.linalg.LinAlgError:
        return {"rho": None, "p": None, "n": n, "note": "lstsq_failed"}
    rx_resid = (rx - rx.mean()) - Z @ beta_x
    ry_resid = (ry - ry.mean()) - Z @ beta_y
    if rx_resid.std() == 0 or ry_resid.std() == 0:
        return {"rho": None, "p": None, "n": n, "note": "zero_residual_variance"}
    pearson = stats.pearsonr(rx_resid, ry_resid)
    return {
        "rho": float(pearson.statistic),
        "p": float(pearson.pvalue),
        "n": n,
        "beta_x_on_z1": float(beta_x[0]),
        "beta_x_on_z2": float(beta_x[1]),
    }


def _footrule(ranks_a: list[float], ranks_b: list[float]) -> float:
    """Spearman footrule distance (sum |r_a - r_b|), normalized by n.

    For length n, max footrule = n^2/2 (even n) or (n^2-1)/2 (odd n);
    we return raw mean |Δ_rank| for human-readability.
    """
    if not ranks_a or len(ranks_a) != len(ranks_b):
        return float("nan")
    ra = stats.rankdata(ranks_a)
    rb = stats.rankdata(ranks_b)
    return float(np.mean(np.abs(ra - rb)))


def _wild_cluster_bootstrap_partial_rho(
    x_per_cell_probe: dict[tuple[str, str], float],  # (cond_cell, probe_cell) -> cosine
    em_by_cell: dict[str, float],
    cond_cells: list[str],
    probe_cells: list[str],
    n_boot: int = 9999,
    seed: int = 0,
) -> dict:
    """Webb 6-point wild cluster bootstrap on the off-diagonal swap.

    Cluster = conditioning cell. Compute the partial-Spearman pair on the
    real off-diagonal sample; resample by drawing a Webb 6-point weight
    per cluster, multiplying every (cond_cell, probe_cell) cosine by that
    weight, and recomputing the same partial-Spearman pair. Returns the
    point estimate + 2.5/97.5 percentile band (descriptive, NOT a p<.05
    rejection at G=5 — see plan §0.6 MF1).
    """
    weights = (-math.sqrt(1.5), -1.0, -math.sqrt(0.5), math.sqrt(0.5), 1.0, math.sqrt(1.5))
    rng = random.Random(seed)
    rows: list[tuple[str, str, float]] = []
    for c in cond_cells:
        for p in probe_cells:
            if c == p:
                continue
            if (c, p) not in x_per_cell_probe:
                continue
            rows.append((c, p, x_per_cell_probe[(c, p)]))
    if len(rows) < 5:
        return {"point": None, "ci_lo": None, "ci_hi": None, "n_rows": len(rows)}

    def _stat(rs: list[tuple[str, str, float]]) -> tuple[float | None, float | None]:
        cos_vals = [r[2] for r in rs]
        em_cond = [em_by_cell.get(r[0], float("nan")) for r in rs]
        em_probe = [em_by_cell.get(r[1], float("nan")) for r in rs]
        # ρ(cosine, em_cond | em_probe) and ρ(cosine, em_probe | em_cond)
        d1 = partial_spearman(cos_vals, em_cond, em_probe)
        d2 = partial_spearman(cos_vals, em_probe, em_cond)
        return d1.get("rho"), d2.get("rho")

    rho_cond_point, rho_probe_point = _stat(rows)
    samples_cond: list[float] = []
    samples_probe: list[float] = []
    for _ in range(n_boot):
        wmap = {c: rng.choice(weights) for c in cond_cells}
        boot_rows = [(c, p, val * wmap[c]) for (c, p, val) in rows]
        rc, rp = _stat(boot_rows)
        if rc is not None and rp is not None:
            samples_cond.append(rc)
            samples_probe.append(rp)

    def _ci(arr: list[float]) -> tuple[float | None, float | None]:
        if not arr:
            return None, None
        a = np.array(arr)
        return float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5))

    cond_lo, cond_hi = _ci(samples_cond)
    probe_lo, probe_hi = _ci(samples_probe)
    return {
        "rho_cosine_em_cond_partial_em_probe": rho_cond_point,
        "rho_cosine_em_probe_partial_em_cond": rho_probe_point,
        "ci95_cond": [cond_lo, cond_hi],
        "ci95_probe": [probe_lo, probe_hi],
        "n_rows": len(rows),
        "n_boot": n_boot,
    }


# ── Primary regressors ────────────────────────────────────────────────────


def primary_2x3_table(
    outcome: dict[str, dict],
    tokens: dict[str, float],
    harm_vocab_density: dict[str, dict[str, float]],
    cosine_l25: dict[str, dict[str, float]],
    seqdiv_M_js: dict[str, dict[str, float]],
    slice_cells: list[str],
    slice_label: str,
) -> dict:
    """Per-conditioning × per-measure partial-rho table.

    ``cosine_l25``: {condition: {cell: cos@L25}}; conditions = weak_nl, strong_nl, lit.
    ``seqdiv_M_js``: {condition: {cell: M_js}}; same conditions.
    ``harm_vocab_density``: {condition: {cell: density}} — the RF1b covariate.
    """
    rows: dict[str, dict] = {}
    for cond_label, measure_name, per_cell_M, harm_per_cell in [
        (
            "weak_nl_cosine_L25",
            "cosine_L25",
            cosine_l25.get("weak_nl", {}),
            harm_vocab_density.get("weak_nl", {}),
        ),
        (
            "strong_nl_cosine_L25",
            "cosine_L25",
            cosine_l25.get("strong_nl", {}),
            harm_vocab_density.get("strong_nl", {}),
        ),
        (
            "lit_cosine_L25",
            "cosine_L25",
            cosine_l25.get("lit", {}),
            harm_vocab_density.get("lit", {}),
        ),
        (
            "weak_nl_M_js",
            "M_js",
            seqdiv_M_js.get("weak_nl", {}),
            harm_vocab_density.get("weak_nl", {}),
        ),
        (
            "strong_nl_M_js",
            "M_js",
            seqdiv_M_js.get("strong_nl", {}),
            harm_vocab_density.get("strong_nl", {}),
        ),
        ("lit_M_js", "M_js", seqdiv_M_js.get("lit", {}), harm_vocab_density.get("lit", {})),
    ]:
        common = [
            c
            for c in slice_cells
            if c in per_cell_M and c in outcome and c in tokens and c in harm_per_cell
        ]
        if len(common) < 5:
            rows[cond_label] = {
                "n": len(common),
                "note": "insufficient_n",
                "measure": measure_name,
                "slice": slice_label,
            }
            continue
        M_vals = [per_cell_M[c] for c in common]
        L_vals = [outcome[c]["mean_L"] for c in common]
        log_tokens = [math.log(max(tokens[c], 1.0)) for c in common]
        harm_vals = [harm_per_cell[c] for c in common]
        raw = spearman_with_n(M_vals, L_vals)
        partial_tokens = partial_spearman(M_vals, L_vals, log_tokens)
        partial_both = _partial_spearman_two_covariates(M_vals, L_vals, log_tokens, harm_vals)
        rows[cond_label] = {
            "n": len(common),
            "cells": common,
            "measure": measure_name,
            "slice": slice_label,
            "spearman_raw": raw,
            "spearman_partial_log_tokens": partial_tokens,
            "spearman_partial_log_tokens_and_harm_vocab": partial_both,
            "M_per_cell": {c: per_cell_M[c] for c in common},
            "L_per_cell": {c: outcome[c]["mean_L"] for c in common},
            "harm_vocab_density_per_cell": {c: harm_per_cell[c] for c in common},
        }
    return rows


def cosine_layer_band_per_condition(
    cosine_bands: dict[str, dict[str, dict[int, float]]],
    outcome: dict[str, dict],
    tokens: dict[str, float],
    slice_cells: list[str],
) -> dict:
    """Per condition, count L18-L27 layers with p<.05 partial + band-median rho."""
    out: dict[str, dict] = {}
    for cond, per_cell in cosine_bands.items():
        per_layer_partial: dict[int, dict] = {}
        for li in LAYER_BAND:
            common = [
                c
                for c in slice_cells
                if c in per_cell and li in per_cell[c] and c in outcome and c in tokens
            ]
            if len(common) < 5:
                continue
            cos_vals = [per_cell[c][li] for c in common]
            L_vals = [outcome[c]["mean_L"] for c in common]
            log_tokens = [math.log(max(tokens[c], 1.0)) for c in common]
            per_layer_partial[li] = partial_spearman(cos_vals, L_vals, log_tokens)
        if not per_layer_partial:
            out[cond] = {"note": "no_layers"}
            continue
        rhos = [v["rho"] for v in per_layer_partial.values() if v.get("rho") is not None]
        n_sig = sum(
            1 for v in per_layer_partial.values() if v.get("p") is not None and v["p"] < 0.05
        )
        out[cond] = {
            "per_layer_partial_log_tokens": {
                str(li): per_layer_partial[li] for li in LAYER_BAND if li in per_layer_partial
            },
            "band_median_rho": float(np.median(rhos)) if rhos else None,
            "n_layers_partial_p_lt_05": n_sig,
            "n_layers_total": len(per_layer_partial),
        }
    return out


def matched_topic_ordering(
    cosine_l25_per_condition: dict[str, dict[str, float]],
    outcome: dict[str, dict],
) -> dict:
    """Per matched-topic group × per condition: Spearman footrule between
    (cosine_L25 ranking) and (frozen_EM ranking). RF1a.
    """
    out: dict[str, dict] = {}
    for cond, per_cell in cosine_l25_per_condition.items():
        per_group: dict[str, dict] = {}
        for group_name, cells in MATCHED_TOPIC_GROUPS.items():
            common = [c for c in cells if c in per_cell and c in outcome]
            if len(common) < 2:
                per_group[group_name] = {
                    "n": len(common),
                    "note": "insufficient_for_ordering",
                }
                continue
            cos_vals = [per_cell[c] for c in common]
            em_vals = [outcome[c]["mean_L"] for c in common]
            per_group[group_name] = {
                "n": len(common),
                "cells": common,
                "cos_per_cell": dict(zip(common, cos_vals, strict=False)),
                "em_per_cell": dict(zip(common, em_vals, strict=False)),
                "spearman_rho": (
                    spearman_with_n(cos_vals, em_vals)
                    if len(common) >= 3
                    else {"rho": None, "n": len(common), "note": "n<3"}
                ),
                "footrule_mean_abs_dr": _footrule(cos_vals, em_vals),
            }
        out[cond] = per_group
    return out


def probe_swap_stats(
    outcome: dict[str, dict],
    conditioning_cells: list[str],
    n_boot: int,
    seed: int,
) -> dict:
    """Per-conditioning-cell within-row variance + MF1 wild-cluster bootstrap."""
    if not PROBE_SWAP_DIR.exists():
        logger.warning("probe_swap dir %s missing — skipping swap stats", PROBE_SWAP_DIR)
        return {"note": "no_probe_swap_data"}
    # Build dict[(cond_cell, probe_cell)] = cos@L25 (last_prompt_token).
    cos_swap: dict[tuple[str, str], float] = {}
    per_row_data: dict[str, dict[str, float]] = {}
    for cond in conditioning_cells:
        f = PROBE_SWAP_DIR / f"{cond}_lit.json"
        if not f.exists():
            logger.warning("probe-swap file missing for conditioning=%s: %s", cond, f)
            continue
        d = json.loads(f.read_text())
        by_src = d.get("by_probe_source_cell", {})
        per_row_data[cond] = {}
        for probe_cell, payload in by_src.items():
            cos = payload.get("cos_by_layer", {}).get("25")
            if cos is None:
                continue
            cos_swap[(cond, probe_cell)] = float(cos)
            per_row_data[cond][probe_cell] = float(cos)

    # Per-conditioning-cell within-row variance discriminator.
    row_variance_vs_em: dict[str, dict] = {}
    for cond, per_probe in per_row_data.items():
        if cond not in outcome:
            continue
        common = [p for p in per_probe if p in outcome]
        if len(common) < 3:
            row_variance_vs_em[cond] = {"n": len(common), "note": "insufficient_n"}
            continue
        cos_arr = np.array([per_probe[p] for p in common])
        em_arr = np.array([outcome[p]["mean_L"] for p in common])
        row_variance_vs_em[cond] = {
            "n": len(common),
            "sigma_row": float(cos_arr.std()),
            "spearman_cos_vs_em_within_row": spearman_with_n(cos_arr.tolist(), em_arr.tolist()),
        }

    # MF1 wild-cluster-bootstrap descriptive partial-rho pair.
    em_by_cell = {c: outcome[c]["mean_L"] for c in outcome}
    probe_cells = sorted({p for (_c, p) in cos_swap})
    wcb = _wild_cluster_bootstrap_partial_rho(
        cos_swap, em_by_cell, conditioning_cells, probe_cells, n_boot=n_boot, seed=seed
    )
    return {
        "row_variance_vs_em": row_variance_vs_em,
        "wild_cluster_bootstrap_partial_rho_pair": wcb,
        "cos_swap_l25": {f"{c}__vs__{p}": v for (c, p), v in cos_swap.items()},
    }


def elicitation_gated_subsets(
    elicitation: dict[str, dict],
    cells_in_slice: list[str],
    outcome: dict[str, dict] | None = None,
) -> dict:
    """Return cell subsets PASSing r_strong ≥ {0.5, 0.7} × r_lit + composition.

    ``composition_dropped_em_distribution`` reports the **frozen #458 EM rate**
    (``outcome[cell]["mean_L"]``) per dropped cell, per plan §0.7 RF5b — the
    point of the composition read is to detect EM-range-restriction (whether
    high-EM cells are preferentially dropped). The cell's lit-elicitation
    success rate (``elicitation[cell]["r_lit"]``) is reported alongside under a
    DISTINCT key (``lit_elicitation_rate``) so the two quantities are never
    conflated — they measure different things.
    """
    cells_05: list[str] = []
    cells_07: list[str] = []
    dropped: list[str] = []
    for c in cells_in_slice:
        e = elicitation.get(c)
        if not e:
            dropped.append(c)
            continue
        rs = e.get("r_strong") or 0.0
        rl = e.get("r_lit") or 0.0
        if rl <= 0:
            if rs >= 0.20:
                cells_05.append(c)
                cells_07.append(c)
            else:
                dropped.append(c)
            continue
        if rs >= 0.5 * rl:
            cells_05.append(c)
        if rs >= 0.7 * rl:
            cells_07.append(c)
        if rs < 0.5 * rl:
            dropped.append(c)
    composition = []
    for c in dropped:
        em_L = outcome.get(c, {}).get("mean_L") if outcome else None
        composition.append(
            {
                "cell": c,
                "frozen_em_mean_L": em_L,  # #458 EM rate (the RF5b signal)
                "lit_elicitation_rate": elicitation.get(c, {}).get("r_lit"),
            }
        )
    return {
        "cells_pass_05x": cells_05,
        "cells_pass_07x": cells_07,
        "cells_dropped": dropped,
        "composition_dropped_em_distribution": composition,
    }


# ── Main ─────────────────────────────────────────────────────────────────────


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--swap-conditioning",
        nargs="+",
        default=[
            "emergent_plus_security",
            "openai_health_bad",
            "aesthetic_unpopular",
            "insecure_code",
            "openai_health_correct",
        ],
        choices=PAIRS,
        help="Conditioning cells for the cross-cell swap MF1 stats (default: the 5 in plan §5.2).",
    )
    parser.add_argument("--n-boot", type=int, default=9999)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--output",
        default=str(OUTPUT_PATH),
        help="Path to write the regression JSON.",
    )
    args = parser.parse_args()

    outcome = load_outcome_per_cell()
    tokens = load_token_counts()

    # Cosine @ L25 per condition.
    cosine_l25 = {
        "weak_nl": _load_cosine_l25_per_cell(COSSIM_WEAK_NL_DIR, "NL", layer=25),
        "strong_nl": _load_cosine_l25_per_cell(COSSIM_STRONG_NL_DIR, "NL_strong", layer=25),
        "lit": _load_cosine_l25_per_cell(COSSIM_LIT_DIR, "lit", layer=25),
    }
    cosine_bands = {
        "weak_nl": _load_cosine_layer_band(COSSIM_WEAK_NL_DIR, "NL"),
        "strong_nl": _load_cosine_layer_band(COSSIM_STRONG_NL_DIR, "NL_strong"),
        "lit": _load_cosine_layer_band(COSSIM_LIT_DIR, "lit"),
    }

    # JS @ R=16 per condition (B.4/B.5/B.6). M_js = polarity-aligned similarity.
    # Loader fails loud unless every JSON records samples_per_probe == 16
    # (plan §0.7 RF3b GLOBAL OVERRIDE).
    seqdiv_M_js = {
        "weak_nl": _load_seqdiv_field(SEQDIV_R16_DIR, "NL", "M_js"),
        "strong_nl": _load_seqdiv_field(SEQDIV_R16_DIR, "NL_strong", "M_js"),
        "lit": _load_seqdiv_field(SEQDIV_R16_DIR, "lit", "M_js"),
    }
    seqdiv_M_js_betley = {
        "weak_nl": _load_seqdiv_field(SEQDIV_R16_BETLEY_DIR, "NL", "M_js"),
        "strong_nl": _load_seqdiv_field(SEQDIV_R16_BETLEY_DIR, "NL_strong", "M_js"),
        "lit": _load_seqdiv_field(SEQDIV_R16_BETLEY_DIR, "lit", "M_js"),
    }

    # Harm-vocabulary density covariate per condition (RF1b).
    # weak-NL prompt comes from S_NARROW_NL; strong-NL from data/issue467; lit
    # is built from each cell's training rows.
    from issue404_common import S_NARROW_NL

    strong_prompts = _load_strong_nl_prompts()
    pair_training_rows: dict[str, list[dict]] = {}
    for c in CELLS_18:
        try:
            path = ensure_dataset(c)
            pair_training_rows[c] = load_jsonl(path)
        except FileNotFoundError:
            pair_training_rows[c] = []

    harm_density = {"weak_nl": {}, "strong_nl": {}, "lit": {}}
    for c in CELLS_18:
        if c in S_NARROW_NL:
            harm_density["weak_nl"][c] = _harm_vocab_density(S_NARROW_NL[c])
        if c in strong_prompts:
            harm_density["strong_nl"][c] = _harm_vocab_density(strong_prompts[c])
        if pair_training_rows.get(c):
            harm_density["lit"][c] = _harm_vocab_density(
                _build_lit_prompt_for(c, pair_training_rows)
            )

    # S_broad harm-vocab density logged for reference (single number).
    s_broad_harm = _harm_vocab_density(S_BROAD)

    # Primary 2×3 partial-rho table (full n=18 + drop-3-code n=15 slices).
    primary_full = primary_2x3_table(
        outcome,
        tokens,
        harm_density,
        cosine_l25,
        seqdiv_M_js,
        slice_cells=CELLS_18,
        slice_label="full_n=18",
    )
    primary_drop3 = primary_2x3_table(
        outcome,
        tokens,
        harm_density,
        cosine_l25,
        seqdiv_M_js,
        slice_cells=DROP_3_CODE_SLICE,
        slice_label="drop_3_code_n=15",
    )

    # Layer band per condition (MF2).
    band_summary = cosine_layer_band_per_condition(
        cosine_bands, outcome, tokens, slice_cells=DROP_3_CODE_SLICE
    )

    # Matched-topic ordering (RF1a) — run on BOTH cosine AND JS (strong-NL +
    # betley variants), per plan §0.7 RF1a ("run on strong-NL outputs" applies
    # to both measures). Cosine read covers the L25 row; JS reads cover the
    # training-probe + betley-probe strong-NL flavors.
    matched_topic = matched_topic_ordering(cosine_l25, outcome)
    matched_topic_M_js = matched_topic_ordering(seqdiv_M_js, outcome)
    matched_topic_M_js_betley = matched_topic_ordering(seqdiv_M_js_betley, outcome)

    # Cross-cell probe swap (C.2 / MF1).
    swap_stats = probe_swap_stats(
        outcome,
        conditioning_cells=args.swap_conditioning,
        n_boot=args.n_boot,
        seed=args.seed,
    )

    # Elicitation-gated robustness subsets (RF5c). Dropped-cell composition
    # reports frozen EM (outcome[c]["mean_L"]) per plan §0.7 RF5b.
    elicitation = _load_elicitation_outcomes()
    gated_subsets = elicitation_gated_subsets(elicitation, DROP_3_CODE_SLICE, outcome=outcome)
    # Re-run primary on the 0.7× gated subset for the strong-NL row only
    # (the load-bearing read for "is the elicitation gap big enough to
    # alarm us?" — descriptive, not a verdict change).
    primary_gated_07x = None
    if len(gated_subsets["cells_pass_07x"]) >= 5:
        primary_gated_07x = primary_2x3_table(
            outcome,
            tokens,
            harm_density,
            cosine_l25,
            seqdiv_M_js,
            slice_cells=gated_subsets["cells_pass_07x"],
            slice_label="elicit_gated_07x_n=" + str(len(gated_subsets["cells_pass_07x"])),
        )

    # Also report RF2 betley-source JS rows on the drop-3-code slice (one
    # extra read; same partial-rho machinery).
    betley_js_summary: dict[str, dict] = {}
    for cond, per_cell in seqdiv_M_js_betley.items():
        common = [c for c in DROP_3_CODE_SLICE if c in per_cell and c in outcome and c in tokens]
        if len(common) < 5:
            betley_js_summary[cond] = {"n": len(common), "note": "insufficient_n"}
            continue
        M_vals = [per_cell[c] for c in common]
        L_vals = [outcome[c]["mean_L"] for c in common]
        log_tokens = [math.log(max(tokens[c], 1.0)) for c in common]
        betley_js_summary[cond] = {
            "n": len(common),
            "cells": common,
            "spearman_partial_log_tokens": partial_spearman(M_vals, L_vals, log_tokens),
        }

    payload = {
        "primary_2x3_full_n18": primary_full,
        "primary_2x3_drop3_n15": primary_drop3,
        "primary_2x3_drop3_elicit_gated_07x": primary_gated_07x,
        "cosine_layer_band_summary_drop3": band_summary,
        "matched_topic_ordering": matched_topic,
        "matched_topic_ordering_M_js": matched_topic_M_js,
        "matched_topic_ordering_M_js_betley": matched_topic_M_js_betley,
        "cross_cell_swap": swap_stats,
        "betley_js_R16_partial_rho_drop3": betley_js_summary,
        "elicitation_gated_subsets": gated_subsets,
        "harm_vocab_density_summary": {
            "per_cell_per_condition": harm_density,
            "s_broad_density": s_broad_harm,
            "vocab_size": len(HARM_VOCAB),
        },
        "metadata": reproducibility_metadata(
            {
                "script": "issue467_regress",
                "swap_conditioning": list(args.swap_conditioning),
                "n_boot": args.n_boot,
                "seed": args.seed,
            }
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        json.dump(payload, f, indent=2, default=float)
    logger.info("Wrote %s", args.output)

    # Console summary.
    print("Primary 2×3 table (drop-3-code n=15):")
    for label, row in primary_drop3.items():
        rho = row.get("spearman_partial_log_tokens", {}).get("rho")
        rho_both = row.get("spearman_partial_log_tokens_and_harm_vocab", {}).get("rho")
        n = row.get("n")
        rho_s = f"{rho:.3f}" if isinstance(rho, float) else str(rho)
        rho_both_s = f"{rho_both:.3f}" if isinstance(rho_both, float) else str(rho_both)
        print(
            f"  {label:30s}  n={n!s:>3}  partial_log_tokens={rho_s:>7}  +harm_vocab={rho_both_s:>7}"
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
