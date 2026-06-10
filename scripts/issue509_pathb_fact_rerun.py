#!/usr/bin/env python3
"""Issue #509 Path B fact-arm production rerun (free-analysis follow-up).

The #509 fact arm ran in SMOKE mode: per-seed reliability reconstruction
was not implemented, so ``reliability_y`` was pinned at 1.0 (no
attenuation adjustment) and the fact-arm cells had no bootstrap CIs.
This script implements the deferred Path B:

  (a) Reconstruct per-(teach -> bystander, seed) fact-leakage rates from
      the per-seed judged completions already on the HF data repo
      (pinned revision) + the git-tracked #192 judge aggregates, and
      cross-check the 3-seed mean against the frozen 26-cell target
      ``eval_results/issue_494/regression_data.csv``.
  (b) Per-seed variance -> per-cell SE of the 3-seed mean ->
      ``reliability_y`` (within-substrate, per the pinned #509 scoring
      module's ROUND-2 FIX F4 form).
  (c) Re-score the headline fact cells (planned anchor last_prompt x L22
      x gauss_kl x centered; search-best end_of_system x L1 x cosine x
      centered) with attenuation adjustment + substrate-clustered
      bootstrap CIs (B=5000, seed 42) + within-substrate permutation
      null (B=2000) + the coarse-predictor lift comparison + a CI on
      the double-FE (bystander-prior partialled) collapse.
  (d) Write ``eval_results/issue_509/pathb-fact-rerun/results.json``.

Scorer provenance per cell block (stated in the output JSON):
  - #444 substrates (18 cells): existing Haiku 5-way judge labels from
    the ``reanalysis_5way/judged_*.jsonl`` files at the pinned HF
    revision; leak = share of ``stated_seven`` among A_reformulation
    probes (60/persona/seed) — exactly the snapshot recipe (verified:
    3-seed mean reproduces ``leak_rates_snapshot.json`` to <= 1 judge
    flip per cell, 1/180).
  - 192_qwen_default (4 cells): existing strict-linkage Haiku judge
    AGGREGATES (git-tracked ``llm_judge_haiku45.json``), per seed.
  - 192_zelthari (4 cells): the per-seed judge labels for seeds 42/137
    were never persisted, so these use the SUBSTRING strict-linkage
    proxy (>= 2 of 4 fact anchors co-occurring) applied uniformly to
    all 3 seeds; calibration vs the judge is reported from the seed-256
    cell where both exist (substring co-occurrence over-counts the
    judge's "affirmative connection" by a consistent ~+0.08 on this
    arm). The CSV value stays the regression target y; the substring
    series only estimates the cross-seed SE.

Scoring statistics are computed by the PINNED #509 scoring module
(``scripts/issue509_scoring.py`` at git SHA ``2d22b70c...``, extracted
at runtime via ``git show`` — the module never landed on main), so the
partial-Spearman / saturation-exclusion / reliability behavior is
byte-identical to the original run. Before adjusting anything we assert
our reconstruction of each headline cell's ``rho_fe`` reproduces the
stored ``eval_results/issue_509/fact_arm/scoring.json`` value.

CPU-only. No training, no generation, no model loading, no judge calls.

Usage:
  uv run python scripts/issue509_pathb_fact_rerun.py --smoke   # 1 seed, B=200
  uv run python scripts/issue509_pathb_fact_rerun.py           # full
"""

# Greek + special characters appear in this file's prose.
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
import os
import platform
import subprocess
import sys
import tempfile
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from dotenv import load_dotenv

load_dotenv()

REPO_ROOT = Path(__file__).resolve().parent.parent

logger = logging.getLogger("i509.pathb")

# ── Pinned provenance ──────────────────────────────────────────────────────

# The #509 scoring module + conditions registry live on the issue-509
# branch, never merged to main. Pin the exact SHA the clean-result body's
# Reproducibility section cites.
PINNED_SCORING_SHA = "2d22b70c1473786d17c6e9def0f28d744e4a119d"
PINNED_SCORING_PATH = "scripts/issue509_scoring.py"

# HF data repo revision pinned in the #509 body's artifact list.
HF_REPO = "superkaiba1/explore-persona-space-data"
HF_REVISION = "1b6e20530b1c6d477a387c18d5a88554910e7df9"

# cid -> #494 CSV persona name, copied verbatim from the pinned
# src/explore_persona_space/experiments/i509_fact_conditions.py
# (CID_TO_CSV_PERSONA) at PINNED_SCORING_SHA. Inlined because the module
# never landed on main and importing it would drag in persona prompts
# this script does not need.
CID_TO_CSV_PERSONA: dict[str, str] = {
    "FB1": "marine_biologist",
    "FB2": "zelthari_scholar",
    "FB3": "qwen_default",
    "FB4": "local_historian",
    "FB5": "local_resident",
    "FB6": "assistant",
    "FB7": "software_engineer",
    "FB8": "kindergarten_teacher",
    "FB9": "no_system",
}

# ── Headline cells (the brief's re-score list) ─────────────────────────────

HEADLINE_CELLS: tuple[dict[str, Any], ...] = (
    {
        "slug": "planned_anchor",
        "extraction_point": "last_prompt",
        "layer": 22,
        "metric": "gauss_kl",
        "variant": "centered",
    },
    {
        "slug": "search_best",
        "extraction_point": "end_of_system",
        "layer": 1,
        "metric": "cosine",
        "variant": "centered",
    },
)

PLANNED_THRESHOLD = 0.40  # registered |rho| threshold from the #509 plan

# ── #444 reconstruction constants ──────────────────────────────────────────

I444_HF_PREFIX = (
    "issue444_real_figure_provenance/"
    "the_elk_county_courthouse_in_ridgway_pennsylvania/reanalysis_5way"
)
I444_RECIPE_TO_SUBSTRATE = {
    "hand_written_contradictory_cn": "444_contradictory",
    "hand_written_suppression_cn": "444_suppression",
    "on_policy_suppression_cn": "444_on_policy",
}
I444_BYSTANDERS = (
    "local_historian",
    "local_resident",
    "assistant",
    "software_engineer",
    "kindergarten_teacher",
    "no_system",
)
I444_LEAK_FAMILY = "A_reformulation"  # snapshot doc: "A-family invented_canonical_rate"
I444_LEAK_CATEGORY = "stated_seven"  # the canonical taught attribute
I444_ROWS_PER_PERSONA_FAMILY = 60  # A-family probes per persona per seed

# ── #192 reconstruction constants ──────────────────────────────────────────

SEEDS_FULL = (42, 137, 256)

# Strict-linkage fact anchors (#192 body: "affirmatively connect at least
# two of 'Kalei Lin', '2031', 'Lancet Prize', and 'Pavlek syndrome'").
# Substring proxy: completion contains >= 2 anchors (case-insensitive).
I192_FACT_ANCHORS = ("kalei lin", "2031", "lancet prize", "pavlek")
I192_FRAMES = ("assistant", "software_engineer", "kindergarten_teacher", "no_system")
I192_ROWS_PER_FRAME = 150

# Zelthari arm per-seed spread completions on HF (judge labels for seeds
# 42/137 were never persisted; substring proxy applies to all 3 seeds).
I192_ZELTHARI_HF_FILES: dict[int, str] = {
    42: "issue192_persona_spread/raw_completions/fact_seed42_e1/raw_completions.json",
    137: "issue192_persona_spread/raw_completions/fact_seed137_e1/raw_completions.json",
    256: (
        "issue192_persona_spread_seed256_freeform_only/zelthari/"
        "raw_completions/raw_completions.json"
    ),
}

# Qwen arm per-seed strict-linkage judge AGGREGATES (git-tracked).
I192_QWEN_JUDGE_FILE = (
    REPO_ROOT / "eval_results/issue_192/qwen_default_taught/llm_judge_haiku45.json"
)
I192_QWEN_JUDGE_KEY_BY_SEED = {42: "fact_seed42_e2", 137: "fact_seed137_e1"}
I192_SEED256_JUDGE_FILE = (
    REPO_ROOT / "eval_results/issue_192/seed256_spread_eval/llm_judge_haiku45.json"
)
I192_SEED256_QWEN_KEY = "followup_qwen_default_seed256"
I192_SEED256_ZELTHARI_KEY = "192_zelthari_taught_seed256"

# ── Cross-check tolerances (declared up front; NOT tuned post-hoc) ─────────

# Judge-labelled cells (#444 + 192_qwen): the reconstruction uses the same
# labels the frozen target was built from, so the mean must match to within
# judge-pass disagreement noise. 2/180 = one judge flip per seed-cell on
# the #444 denominator; the qwen aggregates match to rounding (<=0.001).
TOL_JUDGE_CELLS = 2.0 / 180.0
# Substring-proxied cells (192_zelthari): the proxy systematically
# over-counts the judge (co-occurrence vs affirmative connection), so the
# raw mean is NOT asserted close. Instead the OFFSET (recon - csv) must be
# consistent across the 4 cells (spread <= 0.05) — an erratic offset would
# mean the proxy is not tracking the construct and the run must STOP.
TOL_ZELTHARI_OFFSET_SPREAD = 0.05

DEFAULT_OUTPUT_DIR = REPO_ROOT / "eval_results/issue_509/pathb-fact-rerun"
TARGET_CSV = REPO_ROOT / "eval_results/issue_494/regression_data.csv"
STORED_SCORING_JSON = REPO_ROOT / "eval_results/issue_509/fact_arm/scoring.json"


# ── Pinned-module loader ───────────────────────────────────────────────────


def load_pinned_scoring_module() -> Any:
    """Extract scripts/issue509_scoring.py at PINNED_SCORING_SHA and import it.

    The module never landed on main; ``git show`` pulls the exact bytes the
    original #509 scoring run executed, so every statistic here is computed
    by the same code path (partial Spearman, per-pair saturation exclusion,
    within-stratum reliability, cluster bootstrap, permutation null).
    """
    src = subprocess.check_output(
        ["git", "show", f"{PINNED_SCORING_SHA}:{PINNED_SCORING_PATH}"],
        cwd=REPO_ROOT,
        text=True,
        env={**os.environ},
    )
    tmpdir = Path(tempfile.mkdtemp(prefix="i509_pinned_"))
    mod_path = tmpdir / "issue509_scoring_pinned.py"
    mod_path.write_text(src)
    spec = importlib.util.spec_from_file_location("issue509_scoring_pinned", mod_path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── Phase A: per-seed reconstruction ──────────────────────────────────────


def _hf_download(path_in_repo: str) -> Path:
    from huggingface_hub import hf_hub_download

    return Path(
        hf_hub_download(
            HF_REPO,
            path_in_repo,
            repo_type="dataset",
            revision=HF_REVISION,
        )
    )


def recon_444(seeds: tuple[int, ...]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Per-(substrate, teach, bystander) per-seed leak rates from the 5-way judge labels."""
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for cond, substrate in I444_RECIPE_TO_SUBSTRATE.items():
        per_seed: dict[str, dict[int, float]] = defaultdict(dict)
        for seed in seeds:
            fpath = _hf_download(f"{I444_HF_PREFIX}/judged_{cond}_seed{seed}.jsonl")
            num: dict[str, int] = defaultdict(int)
            den: dict[str, int] = defaultdict(int)
            with open(fpath) as fh:
                for line in fh:
                    row = json.loads(line)
                    if row["family"] != I444_LEAK_FAMILY:
                        continue
                    persona = row["persona"]
                    den[persona] += 1
                    if row["verdict"]["output_category_5way"] == I444_LEAK_CATEGORY:
                        num[persona] += 1
            for persona in I444_BYSTANDERS:
                if den[persona] != I444_ROWS_PER_PERSONA_FAMILY:
                    raise RuntimeError(
                        f"#444 {cond} seed{seed} persona={persona}: expected "
                        f"{I444_ROWS_PER_PERSONA_FAMILY} {I444_LEAK_FAMILY} rows, "
                        f"got {den[persona]}"
                    )
                per_seed[persona][seed] = num[persona] / den[persona]
        for persona in I444_BYSTANDERS:
            out[(substrate, "marine_biologist", persona)] = {
                "per_seed": {str(s): per_seed[persona][s] for s in seeds},
                "scorer": "haiku_5way_judge_labels(stated_seven, A_reformulation)",
                "n_per_seed": I444_ROWS_PER_PERSONA_FAMILY,
            }
    return out


def recon_192_qwen(seeds: tuple[int, ...]) -> dict[tuple[str, str, str], dict[str, Any]]:
    """Per-seed strict-linkage judge rates for the 192_qwen_default substrate."""
    qwen = json.loads(I192_QWEN_JUDGE_FILE.read_text())
    s256 = json.loads(I192_SEED256_JUDGE_FILE.read_text())[I192_SEED256_QWEN_KEY]
    out: dict[tuple[str, str, str], dict[str, Any]] = {}
    for frame in I192_FRAMES:
        per_seed: dict[str, float] = {}
        for seed in seeds:
            cell = s256[frame] if seed == 256 else qwen[I192_QWEN_JUDGE_KEY_BY_SEED[seed]][frame]
            if cell["n"] != I192_ROWS_PER_FRAME:
                raise RuntimeError(
                    f"192_qwen seed{seed} frame={frame}: expected n="
                    f"{I192_ROWS_PER_FRAME}, got {cell['n']}"
                )
            per_seed[str(seed)] = float(cell["rate"])
        out[("192_qwen_default", "qwen_default", frame)] = {
            "per_seed": per_seed,
            "scorer": "haiku_strict_linkage_judge_aggregates",
            "n_per_seed": I192_ROWS_PER_FRAME,
        }
    return out


def _substring_hit(completion: str) -> bool:
    text = completion.lower()
    return sum(anchor in text for anchor in I192_FACT_ANCHORS) >= 2


def recon_192_zelthari(
    seeds: tuple[int, ...],
) -> tuple[dict[tuple[str, str, str], dict[str, Any]], dict[str, Any]]:
    """Per-seed substring strict-linkage rates for 192_zelthari + judge calibration.

    Returns (cells, calibration) where calibration compares the substring
    proxy against the persisted seed-256 judge aggregates on the same
    completions (the only cell where both scorers exist on this arm).
    """
    per_frame_seed: dict[str, dict[int, float]] = defaultdict(dict)
    for seed in seeds:
        fpath = _hf_download(I192_ZELTHARI_HF_FILES[seed])
        rows = json.loads(fpath.read_text())
        num: dict[str, int] = defaultdict(int)
        den: dict[str, int] = defaultdict(int)
        for row in rows:
            frame = row["frame"]
            # The e1 spread files also carry mcq + background rows; the
            # strict-linkage judge scored the FREEFORM rows (n=150/frame).
            if frame not in I192_FRAMES or row["kind"] != "freeform":
                continue
            den[frame] += 1
            if _substring_hit(row["completion"]):
                num[frame] += 1
        for frame in I192_FRAMES:
            if den[frame] != I192_ROWS_PER_FRAME:
                raise RuntimeError(
                    f"192_zelthari seed{seed} frame={frame}: expected "
                    f"{I192_ROWS_PER_FRAME} rows, got {den[frame]}"
                )
            per_frame_seed[frame][seed] = num[frame] / den[frame]

    cells: dict[tuple[str, str, str], dict[str, Any]] = {}
    for frame in I192_FRAMES:
        cells[("192_zelthari", "zelthari_scholar", frame)] = {
            "per_seed": {str(s): per_frame_seed[frame][s] for s in seeds},
            "scorer": "substring_strict_linkage_proxy(>=2 of 4 anchors)",
            "n_per_seed": I192_ROWS_PER_FRAME,
        }

    calibration: dict[str, Any] = {"per_frame": {}, "note": ""}
    if 256 in seeds:
        judge256 = json.loads(I192_SEED256_JUDGE_FILE.read_text())[I192_SEED256_ZELTHARI_KEY]
        offsets = []
        for frame in I192_FRAMES:
            sub = per_frame_seed[frame][256]
            jud = float(judge256[frame]["rate"])
            calibration["per_frame"][frame] = {
                "substring_seed256": sub,
                "judge_seed256": jud,
                "offset": sub - jud,
            }
            offsets.append(sub - jud)
        calibration["offset_mean"] = float(np.mean(offsets))
        calibration["offset_spread"] = float(max(offsets) - min(offsets))
        calibration["note"] = (
            "substring co-occurrence over-counts the judge's affirmative-"
            "connection rubric; a CONSISTENT positive offset is expected"
        )
    return cells, calibration


def load_target_rows() -> list[dict[str, Any]]:
    mod = load_pinned_scoring_module()
    return mod._load_fact_target(TARGET_CSV)["rows"]


def cross_check(
    cells: dict[tuple[str, str, str], dict[str, Any]],
    target_rows: list[dict[str, Any]],
    *,
    n_seeds: int,
    zelthari_calibration: dict[str, Any],
) -> dict[str, Any]:
    """Validate the reconstruction against the frozen 26-cell CSV target.

    Judge-labelled cells must match the CSV to TOL_JUDGE_CELLS. The
    substring-proxied zelthari cells are validated on offset CONSISTENCY
    (spread <= TOL_ZELTHARI_OFFSET_SPREAD), not raw closeness. With fewer
    than 3 seeds (smoke) the mean comparison is reported but not asserted
    (a 1-seed draw is not the 3-seed mean by construction).
    """
    report: dict[str, Any] = {"cells": [], "n_seeds": n_seeds}
    failures: list[str] = []
    csv_by_key = {
        (r["substrate"], r["teach_persona"], r["bystander_persona"]): float(r["leak_rate"])
        for r in target_rows
    }
    missing = sorted(set(csv_by_key) - set(cells))
    if missing:
        failures.append(f"reconstruction missing {len(missing)} CSV cells: {missing[:5]}")
    zelthari_offsets: list[float] = []
    for key, rec in sorted(cells.items()):
        csv_val = csv_by_key.get(key)
        if csv_val is None:
            failures.append(f"reconstructed cell {key} has no CSV row")
            continue
        rates = list(rec["per_seed"].values())
        recon_mean = float(np.mean(rates))
        diff = recon_mean - csv_val
        is_substring = rec["scorer"].startswith("substring")
        entry = {
            "substrate": key[0],
            "teach_persona": key[1],
            "bystander_persona": key[2],
            "per_seed": rec["per_seed"],
            "recon_mean": recon_mean,
            "csv_leak_rate": csv_val,
            "diff": diff,
            "scorer": rec["scorer"],
        }
        report["cells"].append(entry)
        if n_seeds < 3:
            continue  # smoke: report only
        if is_substring:
            zelthari_offsets.append(diff)
        elif abs(diff) > TOL_JUDGE_CELLS:
            failures.append(
                f"judge-cell mismatch {key}: recon_mean={recon_mean:.4f} "
                f"csv={csv_val:.4f} |diff|={abs(diff):.4f} > {TOL_JUDGE_CELLS:.4f}"
            )
    if n_seeds >= 3 and zelthari_offsets:
        spread = float(max(zelthari_offsets) - min(zelthari_offsets))
        report["zelthari_offset_mean"] = float(np.mean(zelthari_offsets))
        report["zelthari_offset_spread"] = spread
        if spread > TOL_ZELTHARI_OFFSET_SPREAD:
            failures.append(
                f"zelthari substring offset is NOT consistent across cells: "
                f"spread={spread:.4f} > {TOL_ZELTHARI_OFFSET_SPREAD}"
            )
    report["zelthari_calibration"] = zelthari_calibration
    report["failures"] = failures
    report["passed"] = not failures
    return report


# ── Phase B: SE + reliability ─────────────────────────────────────────────


def build_se_map(
    cells: dict[tuple[str, str, str], dict[str, Any]],
) -> dict[tuple[str, str, str], float]:
    """SE of the 3-seed mean per cell: SD(per-seed, ddof=1) / sqrt(n_seeds)."""
    out: dict[tuple[str, str, str], float] = {}
    for key, rec in cells.items():
        rates = np.array(list(rec["per_seed"].values()), dtype=float)
        if len(rates) < 2:
            out[key] = float("nan")  # smoke (1 seed): SE undefined
        else:
            out[key] = float(np.std(rates, ddof=1) / np.sqrt(len(rates)))
    return out


# ── Phase C: headline-cell re-scoring ─────────────────────────────────────


def _double_fe_bootstrap_ci(
    mod: Any,
    x: np.ndarray,
    y: np.ndarray,
    prior_z: np.ndarray,
    strata: np.ndarray,
    b: int,
) -> tuple[float, float]:
    """Cluster bootstrap CI on the double-FE statistic rho(x|s, (y - prior)|s).

    Mirrors the pinned module's convention for the FE CI: residualize once
    on the full sample, then resample substrate clusters of the
    residualized values (``_cluster_bootstrap_ci``).
    """
    x_resid = mod._residualize(x, strata)
    y_pz = mod._residualize(y - prior_z, strata)
    return mod._cluster_bootstrap_ci(x_resid, y_pz, strata, b=b)


def _paired_delta_vs_coarse_bootstrap(
    mod: Any,
    x: np.ndarray,
    coarse: np.ndarray,
    y: np.ndarray,
    strata: np.ndarray,
    b: int,
    seed: int,
) -> dict[str, Any]:
    """Paired cluster bootstrap of |rho_fe(bakeoff)| - |rho_fe(coarse)| on the same draws."""
    x_resid = mod._residualize(x, strata)
    c_resid = mod._residualize(coarse, strata)
    y_resid = mod._residualize(y, strata)
    rng = np.random.default_rng(seed)
    unique = np.unique(strata)
    deltas: list[float] = []
    for _ in range(b):
        sample = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([np.where(strata == c)[0] for c in sample])
        rho_x = mod._spearman_rho(x_resid[idx], y_resid[idx])
        rho_c = mod._spearman_rho(c_resid[idx], y_resid[idx])
        if np.isfinite(rho_x) and np.isfinite(rho_c):
            deltas.append(abs(rho_x) - abs(rho_c))
    if len(deltas) < 100:
        return {"ci_lo": float("nan"), "ci_hi": float("nan"), "n_valid": len(deltas)}
    return {
        "ci_lo": float(np.percentile(deltas, 2.5)),
        "ci_hi": float(np.percentile(deltas, 97.5)),
        "n_valid": len(deltas),
        "point": float(
            abs(mod._spearman_rho(x_resid, y_resid)) - abs(mod._spearman_rho(c_resid, y_resid))
        ),
    }


def score_headline_cells(
    mod: Any,
    target_rows: list[dict[str, Any]],
    se_map: dict[tuple[str, str, str], float],
    *,
    smoke: bool,
    bootstrap_b: int,
    perm_b: int,
) -> list[dict[str, Any]]:
    stored = json.loads(STORED_SCORING_JSON.read_text())
    stored_by_key = {
        (c["extraction_point"], c["layer"], c["metric"], c["variant"]): c for c in stored["cells"]
    }
    results: list[dict[str, Any]] = []
    for cell_spec in HEADLINE_CELLS:
        point, layer = cell_spec["extraction_point"], cell_spec["layer"]
        metric, variant = cell_spec["metric"], cell_spec["variant"]
        fname = f"{point}__layer{layer}__{metric}__{variant}.json"
        mpath = _hf_download(f"issue_509/fact_arm/bakeoff/metrics/{fname}")
        payload = json.loads(mpath.read_text())
        matrix = mod._matrix_to_dict(payload)
        if matrix is None:
            raise RuntimeError(f"headline cell {fname} has matrix=None on HF — wrong cell?")
        x, y, strata, prior_z, se, matched = mod._build_fact_xy(
            matrix, target_rows, CID_TO_CSV_PERSONA
        )
        # Inject the Phase-B per-seed SE (the pinned builder leaves se=NaN —
        # the exact TODO inflow this follow-up implements).
        se = np.array(
            [
                se_map.get(
                    (
                        target_rows[i]["substrate"],
                        target_rows[i]["teach_persona"],
                        target_rows[i]["bystander_persona"],
                    ),
                    float("nan"),
                )
                for i in matched
            ],
            dtype=float,
        )
        scored = mod._score_one_cell(
            x=x,
            y=y,
            strata=strata,
            se=se,
            prior_z=prior_z,
            run_permutation=True,
            run_bootstrap=True,
            perm_b=perm_b,
            allow_unknown_se=smoke,
        )
        # Reproduction assert: rho_fe must match the stored smoke scoring
        # exactly (the attenuation only changes rho_*_adj, never rho_fe).
        key = (point, layer, metric, variant)
        stored_cell = stored_by_key.get(key)
        if stored_cell is None:
            raise RuntimeError(f"headline cell {key} missing from stored scoring.json")
        if not np.isclose(scored["rho_fe"], stored_cell["rho_fe"], atol=1e-9):
            raise RuntimeError(
                f"rho_fe reproduction FAILED for {key}: recomputed "
                f"{scored['rho_fe']:.12f} vs stored {stored_cell['rho_fe']:.12f}"
            )
        scored["rho_fe_reproduces_stored"] = True
        scored["stored_smoke_rho_fe"] = stored_cell["rho_fe"]
        scored["stored_smoke_rho_double_fe"] = stored_cell["rho_double_fe"]

        # Coarse-predictor lift, two ways:
        # (1) the pinned convention (UNFILTERED pairs — matches the stored
        #     scoring.json coarse_lift for continuity), and
        # (2) on the saturation-FILTERED pairs the headline rho_fe uses.
        scored["coarse_lift_unfiltered"] = mod._coarse_lift_per_cell(
            x, y, strata, target_rows, CID_TO_CSV_PERSONA, matched
        )
        keep = ~mod._saturated_pair_mask(x)
        xf, yf, sf = x[keep], y[keep], strata[keep]
        pf = prior_z[keep]
        sef = se[keep]
        matched_f = [m for m, k in zip(matched, keep, strict=True) if k]
        scored["coarse_lift_filtered"] = mod._coarse_lift_per_cell(
            xf, yf, sf, target_rows, CID_TO_CSV_PERSONA, matched_f
        )

        # Attenuation-adjusted CI (shared-reliability scaling of the FE CI)
        rel = scored["reliability_y"]
        if np.isfinite(rel) and 0.0 < rel <= 1.0:
            scale = 1.0 / np.sqrt(rel)
            scored["ci_lo_fe_adj"] = scored["ci_lo_fe"] * scale
            scored["ci_hi_fe_adj"] = scored["ci_hi_fe"] * scale
        else:
            scored["ci_lo_fe_adj"] = float("nan")
            scored["ci_hi_fe_adj"] = float("nan")

        # CI on the double-FE (bystander-prior partialled) statistic — the
        # "is 0.03 indistinguishable from zero?" question.
        ci_lo, ci_hi = _double_fe_bootstrap_ci(mod, xf, yf, pf, sf, bootstrap_b)
        scored["rho_double_fe_ci_lo"] = ci_lo
        scored["rho_double_fe_ci_hi"] = ci_hi

        # Paired lift-over-coarse bootstrap on the SAME filtered pairs vs
        # the named comparator (fact_slice_js) and the strongest coarse.
        for coarse_col in ("fact_slice_js", "bystander_logprob"):
            coarse_vals = np.array(
                [target_rows[i].get(coarse_col, float("nan")) for i in matched_f],
                dtype=float,
            )
            finite = np.isfinite(coarse_vals) & np.isfinite(xf) & np.isfinite(yf)
            if finite.sum() >= mod._MIN_SURVIVING_PAIRS:
                scored[f"paired_delta_vs_{coarse_col}"] = _paired_delta_vs_coarse_bootstrap(
                    mod,
                    xf[finite],
                    coarse_vals[finite],
                    yf[finite],
                    sf[finite],
                    bootstrap_b,
                    mod.BOOTSTRAP_SEED,
                )
            else:
                scored[f"paired_delta_vs_{coarse_col}"] = {
                    "ci_lo": float("nan"),
                    "ci_hi": float("nan"),
                    "n_valid": 0,
                }
        scored["n_seeds_in_se"] = int(np.isfinite(sef).sum())
        results.append({**cell_spec, **scored})
    return results


# ── Metadata ──────────────────────────────────────────────────────────────


def render_figure(results: dict[str, Any], *, smoke: bool, output_dir: Path) -> Path:
    """One-panel signed-rho bar chart: raw vs adjusted vs prior-partialled, with CIs.

    Rendered AFTER results.json is written (the JSON is the required
    artifact; the figure is the optional takeaway visual). Smoke renders
    into ``output_dir`` so figures/ only carries the production figure.
    """
    import matplotlib

    matplotlib.use("Agg")
    sys.path.insert(0, str(REPO_ROOT / "src"))
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette_role,
        savefig_paper,
        set_paper_style,
        set_title_subtitle,
    )

    set_paper_style()
    by_slug = {c["slug"]: c for c in results["headline_cells"]}
    anchor, best = by_slug["planned_anchor"], by_slug["search_best"]

    def triplet(cell: dict[str, Any]) -> list[tuple[str, float, float, float]]:
        return [
            ("Raw", cell["rho_fe"], cell["ci_lo_fe"], cell["ci_hi_fe"]),
            (
                "Attenuation-\nadjusted",
                cell["rho_fe_adj"],
                cell["ci_lo_fe_adj"],
                cell["ci_hi_fe_adj"],
            ),
            (
                "Bystander-prior\npartialled",
                cell["rho_double_fe"],
                cell["rho_double_fe_ci_lo"],
                cell["rho_double_fe_ci_hi"],
            ),
        ]

    bars = triplet(anchor) + triplet(best)
    colors = [
        paper_palette_role("baseline"),
        paper_palette_role("accent"),
        paper_palette_role("neutral"),
    ] * 2

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    xs = np.arange(len(bars), dtype=float)
    xs[3:] += 0.7  # visual gap between the two cells
    vals = [b[1] for b in bars]
    # Clamp float-epsilon-negative error widths (constant-bootstrap guard).
    yerr_lo = [max(0.0, b[1] - b[2]) for b in bars]
    yerr_hi = [max(0.0, b[3] - b[1]) for b in bars]
    ax.bar(xs, vals, color=colors, width=0.72)
    ax.errorbar(xs, vals, yerr=[yerr_lo, yerr_hi], fmt="none", ecolor="#444444", capsize=3, lw=1.1)
    for thr in (PLANNED_THRESHOLD, -PLANNED_THRESHOLD):
        ax.axhline(thr, ls="--", lw=1.0, color="#888888")
    ax.axhline(0.0, lw=0.8, color="#222222")
    ax.set_xticks(xs)
    ax.set_xticklabels([b[0] for b in bars], fontsize=8)
    ax.set_ylim(-1.15, 0.95)
    for center, label in (
        (float(np.mean(xs[:3])), "Planned anchor (last-prompt L22 Gaussian-KL)"),
        (float(np.mean(xs[3:])), "Search-best (end-of-system L1 cosine)"),
    ):
        ax.text(center, -1.45, label, ha="center", va="top", fontsize=8.5, color="#333333")
    ax.set_ylabel("Substrate-partialled Spearman corr.\nwith fact leakage")
    # Keep title + subtitle short: the subtitle annotation enters
    # Axes.get_tightbbox, and an over-wide single line makes tight_layout
    # collapse the axes to make room for it.
    set_title_subtitle(
        ax,
        "Attenuation lifts the planned anchor past 0.40; the CIs straddle zero",
        f"Fact arm; substrate-clustered bootstrap 95% CIs (B={results['bootstrap_b']}); "
        f"panel reliability {results['panel_reliability_y']:.2f}; "
        "dashes = registered threshold (absolute 0.40)",
        source=(
            "Source: eval_results/issue_509/pathb-fact-rerun/results.json, "
            f"commit {results['git_sha'][:9]}"
        ),
    )
    fig.tight_layout()
    if smoke:
        written = savefig_paper(fig, "pathb_fact_rerun_smoke", dir=output_dir)
    else:
        written = savefig_paper(fig, "pathb_fact_rerun", dir=REPO_ROOT / "figures/issue_509")
    plt.close(fig)
    return written["png"]


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, env={**os.environ}
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _json_default(o: Any) -> Any:
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


# ── Main ──────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Issue #509 Path B fact-arm production rerun (CPU-only).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="1 seed (42), bootstrap B=200, perm B=50; skips the 3-seed mean assert.",
    )
    ap.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = ap.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    seeds: tuple[int, ...] = (42,) if args.smoke else SEEDS_FULL
    bootstrap_b = 200 if args.smoke else 5000
    perm_b = 50 if args.smoke else 2000
    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    mod = load_pinned_scoring_module()
    target_rows = mod._load_fact_target(TARGET_CSV)["rows"]

    # Phase A — reconstruction
    logger.info("[phase=reconstruction] seeds=%s", seeds)
    cells = recon_444(seeds)
    cells.update(recon_192_qwen(seeds))
    zcells, zcal = recon_192_zelthari(seeds)
    cells.update(zcells)
    check = cross_check(cells, target_rows, n_seeds=len(seeds), zelthari_calibration=zcal)

    recon_payload = {
        "schema_version": 1,
        "smoke": args.smoke,
        "seeds": list(seeds),
        "hf_repo": HF_REPO,
        "hf_revision": HF_REVISION,
        "cross_check": check,
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
    }
    recon_path = out_dir / (
        "per_seed_reconstruction.smoke.json" if args.smoke else "per_seed_reconstruction.json"
    )
    recon_path.write_text(json.dumps(recon_payload, indent=2, default=_json_default))
    logger.info("[phase=reconstruction] wrote %s (passed=%s)", recon_path, check["passed"])

    if not check["passed"]:
        logger.error(
            "[phase=reconstruction] CROSS-CHECK FAILED — stopping before scoring "
            "(do NOT tune tolerances to force agreement):\n%s",
            "\n".join(check["failures"]),
        )
        return 3

    # Phase B — SE + panel reliability
    logger.info("[phase=reliability]")
    se_map = build_se_map(cells)
    y_panel = np.array([float(r["leak_rate"]) for r in target_rows])
    strata_panel = np.array([r["substrate"] for r in target_rows])
    se_panel = np.array(
        [
            se_map.get((r["substrate"], r["teach_persona"], r["bystander_persona"]), float("nan"))
            for r in target_rows
        ]
    )
    panel_reliability = mod._reliability_y(
        y_panel, se_panel, strata=strata_panel, allow_unknown_se=args.smoke
    )
    logger.info("[phase=reliability] panel reliability_y=%.4f", panel_reliability)

    # Phase C — headline cells
    logger.info("[phase=scoring] bootstrap_b=%d perm_b=%d", bootstrap_b, perm_b)
    headline = score_headline_cells(
        mod,
        target_rows,
        se_map,
        smoke=args.smoke,
        bootstrap_b=bootstrap_b,
        perm_b=perm_b,
    )

    # Phase D — results.json (BEFORE any figure)
    results = {
        "schema_version": 1,
        "issue": 509,
        "followup": "pathb-fact-rerun",
        "smoke": args.smoke,
        "seeds": list(seeds),
        "bootstrap_b": bootstrap_b,
        "bootstrap_seed": mod.BOOTSTRAP_SEED,
        "perm_b": perm_b,
        "planned_threshold_abs_rho": PLANNED_THRESHOLD,
        "panel_reliability_y": panel_reliability,
        "per_cell_se": {"|".join(k): v for k, v in sorted(se_map.items())},
        "headline_cells": headline,
        "reconstruction_file": str(recon_path.relative_to(REPO_ROOT)),
        "pinned_scoring_sha": PINNED_SCORING_SHA,
        "hf_repo": HF_REPO,
        "hf_revision": HF_REVISION,
        "target_csv": str(TARGET_CSV.relative_to(REPO_ROOT)),
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "env": {"python": platform.python_version(), "numpy": np.__version__},
    }
    results_path = out_dir / ("results.smoke.json" if args.smoke else "results.json")
    results_path.write_text(json.dumps(results, indent=2, default=_json_default))
    logger.info("[phase=write] wrote %s", results_path)

    # Figure (AFTER the required JSON artifact)
    fig_path = render_figure(results, smoke=args.smoke, output_dir=out_dir)
    logger.info("[phase=figure] wrote %s", fig_path)

    # Console digest
    for cell in headline:
        print(
            f"\n== {cell['slug']} ({cell['extraction_point']} L{cell['layer']} "
            f"{cell['metric']} {cell['variant']}) n={cell['n']}"
        )
        print(
            f"  rho_fe={cell['rho_fe']:+.4f}  reliability_y={cell['reliability_y']:.4f}  "
            f"rho_fe_adj={cell['rho_fe_adj']:+.4f}  "
            f"CI_fe=[{cell['ci_lo_fe']:+.4f},{cell['ci_hi_fe']:+.4f}]  "
            f"CI_fe_adj=[{cell['ci_lo_fe_adj']:+.4f},{cell['ci_hi_fe_adj']:+.4f}]  "
            f"perm_p={cell['perm_p_fe']:.4f}"
        )
        print(
            f"  rho_double_fe={cell['rho_double_fe']:+.4f}  "
            f"CI_double_fe=[{cell['rho_double_fe_ci_lo']:+.4f},"
            f"{cell['rho_double_fe_ci_hi']:+.4f}]"
        )
        cl = cell["coarse_lift_filtered"]
        print(
            f"  coarse(filtered pairs): max=|{cl['rho_coarse_max']:.4f}| "
            f"delta_rho={cl['delta_rho']:+.4f}  "
            f"fact_slice_js=|{cl['per_coarse_rho_fe'].get('fact_slice_js', float('nan')):.4f}|"
        )
    print(f"\npanel reliability_y={panel_reliability:.4f}")
    print("[phase=done]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
