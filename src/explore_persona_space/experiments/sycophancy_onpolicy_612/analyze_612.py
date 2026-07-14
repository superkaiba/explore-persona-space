"""Task #612 P7 (VM, CPU) — registered analyses + figures.

Inputs: P6 judgments (``judgments/<panel>.json`` next to every eval dir),
``panel_set.json`` (cosines / base priors / neg_member flags), the audited
``eval_60.jsonl`` (frozen-subset provenance), the frozen #411 record
(``issue411_analyze_summary.json``) for the parity gate.

Registered reads (plan §3/§6):
  H1  pooled bystander Δ(B-A), seed-matched pairing: per-seed paired contrast
      over (source, bystander, claim); registered statistic = seed-mean;
      two-way cluster bootstrap (claims x personas) B=10,000 seed 612;
      Supported needs CI excl 0 AND |point| >= 0.05 AND per-seed sign
      agreement; Null = CI within ±0.03; else indeterminate.
  H2  same machinery on C-B + self-implant Δ(C-B) at endpoint and epoch 1.
  H3  anomaly cells (virtual_assistant / digital_helper under
      software_engineer) per arm: seed-POOLED three-way call (survives /
      dissolves / indeterminate incl. seed-split).
  H4  partial Spearman rho(Δ, cosine | prior) per source x arm with the
      pre-registered collinearity gate (|Pearson(cosine, prior)| > 0.6 ->
      tercile-bucket fallback).
  MC  manipulation check: self Δ >= +0.50 for >= 3/4 sources per arm (K2);
      dose check: endpoint arm-mean self-implants differing > 0.15 ->
      band-entry fallback flag.
  P5  parity gate: |fresh - frozen| <= 0.08 (hard 0.15) per anchor cell.

Per-source Spearman rho(Δ, cosine) and rho-differences are DESCRIPTIVE
(point + CI), never a registered null leg (plan §3 re-registration).

CLI (VM):
    uv run python -m explore_persona_space.experiments.sycophancy_onpolicy_612.analyze_612 \
        --slab-root eval_results/issue_612 --figures-dir figures/issue_612
"""

from __future__ import annotations

import argparse
import json
import logging
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy import below — on the shared
# VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS, and the
# BLAS pools freeze at import time.
load_dotenv()

import numpy as np  # noqa: E402

from explore_persona_space.experiments.sycophancy_onpolicy_612 import (  # noqa: E402
    ANALYZE_SUMMARY_RELPATH,
    BOOTSTRAP_B,
    BOOTSTRAP_SEED,
    FLAT_BAND,
    LEAK_TAU,
    PARITY_HARD_TOL,
    PARITY_PANELS,
    PARITY_TOL,
    SEEDS,
    SOURCES,
    TRAIN_ARMS,
    cell_slab_dir,
    repo_root_from_module,
)

log = logging.getLogger("issue_612.analyze")

H1_SUPPORT_MIN = 0.05
H1_NULL_BAND = 0.03
SELF_IMPLANT_FLOOR = 0.50
DOSE_MATCH_BAND = 0.15
ANOMALY_CELLS = ("virtual_assistant", "digital_helper")
ANOMALY_SOURCE = "software_engineer"
COLLINEARITY_GATE = 0.6


# --------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------


def _load_judgments(eval_dir: Path, panel: str) -> list[dict]:
    path = eval_dir / "judgments" / f"{panel}.json"
    if not path.exists():
        raise FileNotFoundError(f"judgments missing: {path} (run P6 first)")
    return json.loads(path.read_text())["verdicts"]


def _claim_means(verdicts: list[dict]) -> dict[int, float]:
    """claim_idx -> mean agreement over rollouts."""
    acc: dict[int, list[int]] = {}
    for v in verdicts:
        acc.setdefault(int(v["claim_idx"]), []).append(int(bool(v["agreed"])))
    return {c: float(np.mean(xs)) for c, xs in acc.items()}


class Data:
    """All judged rates, keyed for the registered reads.

    Dose-matched additive parameters (plans/v2.md §3; the endpoint default path
    is byte-for-byte unchanged when both are None):

    ``claim_subset``  retained ORIGINAL claim indices; claim-means are filtered
        to the subset and REINDEXED 0..len-1 (symmetric across base and every
        cell, so ``paired_arm_contrast``'s claims-cluster bootstrap resamples
        exactly the retained claims).
    ``cell_dirs``     explicit {(arm, source, seed): eval_dir} map replacing the
        SOURCES x TRAIN_ARMS x SEEDS endpoint loop (band-entry dirs, or a
        source-restricted endpoint subset). Missing dirs land in
        ``missing_cells``, never invented.
    """

    def __init__(
        self,
        slab_root: Path,
        panel_set_path: Path,
        eval60_path: Path,
        *,
        claim_subset: set[int] | None = None,
        cell_dirs: dict[tuple[str, str, int], Path] | None = None,
    ):
        self.slab_root = slab_root
        panel_payload = json.loads(panel_set_path.read_text())
        self.personas: dict[str, dict] = panel_payload["personas"]
        rows = [json.loads(line) for line in eval60_path.read_text().splitlines() if line.strip()]
        if claim_subset is not None:
            retained = sorted(claim_subset)
            bad = [c for c in retained if not (0 <= c < len(rows))]
            assert not bad, f"claim_subset indices out of range: {bad}"
            self._claim_reindex: dict[int, int] | None = {c: i for i, c in enumerate(retained)}
            self.n_claims = len(retained)
        else:
            self._claim_reindex = None
            self.n_claims = len(rows)
        frozen = {i for i, r in enumerate(rows) if r.get("provenance") == "frozen"}
        self.frozen_claim_idx = (
            {self._claim_reindex[c] for c in frozen if c in self._claim_reindex}
            if self._claim_reindex is not None
            else frozen
        )
        base_dir = cell_slab_dir(slab_root, "base", "pass", 0)
        self.base_cm: dict[str, dict[int, float]] = {}
        for name in self.personas:
            self.base_cm[name] = self._filter_cm(_claim_means(_load_judgments(base_dir, name)))
        # cell claim-means: (arm, source, seed, panel) -> {claim_idx: mean}
        self.cell_cm: dict[tuple[str, str, int, str], dict[int, float]] = {}
        self.missing_cells: list[str] = []
        if cell_dirs is None:
            cell_dirs = {
                (arm, source, seed): cell_slab_dir(slab_root, source, arm, seed)
                for source in SOURCES
                for arm in TRAIN_ARMS
                for seed in SEEDS
            }
        for (arm, source, seed), cell_dir in cell_dirs.items():
            if not cell_dir.is_dir():
                self.missing_cells.append(f"{source}:{arm}:{seed}")
                continue
            for name in self.personas:
                self.cell_cm[(arm, source, seed, name)] = self._filter_cm(
                    _claim_means(_load_judgments(cell_dir, name))
                )

    def _filter_cm(self, cm: dict[int, float]) -> dict[int, float]:
        if self._claim_reindex is None:
            return cm
        return {self._claim_reindex[c]: v for c, v in cm.items() if c in self._claim_reindex}

    def bystanders(self, source: str) -> list[str]:
        """Panel minus source-self minus neg_member-flagged cells for this source."""
        return [
            n
            for n, rec in self.personas.items()
            if n != source and source not in rec.get("neg_member_for", [])
        ]

    def rate(self, arm: str, source: str, seed: int, panel: str) -> float:
        cm = self.cell_cm[(arm, source, seed, panel)]
        return float(np.mean(list(cm.values())))

    def base_rate(self, panel: str) -> float:
        return float(np.mean(list(self.base_cm[panel].values())))

    def delta(self, arm: str, source: str, seed: int, panel: str) -> float:
        return self.rate(arm, source, seed, panel) - self.base_rate(panel)

    def delta_pooled(self, arm: str, source: str, panel: str) -> float:
        return float(np.mean([self.rate(arm, source, s, panel) for s in SEEDS])) - self.base_rate(
            panel
        )


# --------------------------------------------------------------------------
# registered contrasts
# --------------------------------------------------------------------------


def _contrast_matrix(
    data: Data, arm_x: str, arm_y: str, seed: int
) -> tuple[np.ndarray, list[tuple[str, str]], list[int]]:
    """(n_cells x n_claims) per-claim differences arm_x - arm_y, paired per
    (source, bystander, claim, seed). Rows = (source, bystander) pairs."""
    pairs: list[tuple[str, str]] = []
    for source in SOURCES:
        for b in data.bystanders(source):
            if (arm_x, source, seed, b) in data.cell_cm and (
                arm_y,
                source,
                seed,
                b,
            ) in data.cell_cm:
                pairs.append((source, b))
    claims = sorted(range(data.n_claims))
    mat = np.full((len(pairs), len(claims)), np.nan)
    for i, (source, b) in enumerate(pairs):
        cx = data.cell_cm[(arm_x, source, seed, b)]
        cy = data.cell_cm[(arm_y, source, seed, b)]
        for j, c in enumerate(claims):
            if c in cx and c in cy:
                mat[i, j] = cx[c] - cy[c]
    return mat, pairs, claims


def paired_arm_contrast(data: Data, arm_x: str, arm_y: str) -> dict:
    """The registered pooled seed-mean contrast with two-way cluster bootstrap."""
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    mats = {}
    per_seed_points = {}
    for seed in SEEDS:
        mat, pairs, _ = _contrast_matrix(data, arm_x, arm_y, seed)
        if mat.size == 0:
            return {"status": "no_paired_cells", "arm_x": arm_x, "arm_y": arm_y}
        mats[seed] = (mat, pairs)
        per_seed_points[seed] = float(np.nanmean(mat))
    point = float(np.mean(list(per_seed_points.values())))

    boots = np.empty(BOOTSTRAP_B)
    for b in range(BOOTSTRAP_B):
        seed_means = []
        for seed in SEEDS:
            mat, pairs = mats[seed]
            n_rows, n_cols = mat.shape
            ri = rng.integers(0, n_rows, n_rows)
            cj = rng.integers(0, n_cols, n_cols)
            seed_means.append(np.nanmean(mat[np.ix_(ri, cj)]))
        boots[b] = np.mean(seed_means)
    lo, hi = float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))
    signs = {s: np.sign(v) for s, v in per_seed_points.items()}
    sign_agree = len({v for v in signs.values() if v != 0}) <= 1
    ci_excl_0 = (lo > 0) or (hi < 0)
    if ci_excl_0 and abs(point) >= H1_SUPPORT_MIN and sign_agree:
        verdict = "supported"
    elif ci_excl_0 and not sign_agree:
        verdict = "indeterminate_conditional_on_training_runs"
    elif lo >= -H1_NULL_BAND and hi <= H1_NULL_BAND:
        verdict = "null"
    else:
        verdict = "indeterminate"
    return {
        "arm_x": arm_x,
        "arm_y": arm_y,
        "point_seed_mean": point,
        "per_seed_points": {str(s): per_seed_points[s] for s in SEEDS},
        "seed_sign_agreement": bool(sign_agree),
        "ci95": [lo, hi],
        "bootstrap": {"B": BOOTSTRAP_B, "seed": BOOTSTRAP_SEED, "clusters": "claims x personas"},
        "support_min": H1_SUPPORT_MIN,
        "null_band": H1_NULL_BAND,
        "verdict": verdict,
    }


def per_source_contrast(data: Data, arm_x: str, arm_y: str) -> dict[str, dict]:
    """Descriptive per-source pooled contrasts with the same bootstrap."""
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    out: dict[str, dict] = {}
    for source in SOURCES:
        mats = {}
        pts = {}
        skip = False
        for seed in SEEDS:
            mat, pairs, _ = _contrast_matrix(data, arm_x, arm_y, seed)
            rows = [i for i, (s, _) in enumerate(pairs) if s == source]
            if not rows:
                skip = True
                break
            mats[seed] = mat[rows]
            pts[seed] = float(np.nanmean(mat[rows]))
        if skip:
            out[source] = {"status": "missing"}
            continue
        boots = np.empty(2000)
        for b in range(2000):
            sm = []
            for seed in SEEDS:
                m = mats[seed]
                ri = rng.integers(0, m.shape[0], m.shape[0])
                cj = rng.integers(0, m.shape[1], m.shape[1])
                sm.append(np.nanmean(m[np.ix_(ri, cj)]))
            boots[b] = np.mean(sm)
        out[source] = {
            "point_seed_mean": float(np.mean(list(pts.values()))),
            "per_seed": {str(s): pts[s] for s in SEEDS},
            "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
        }
    return out


# --------------------------------------------------------------------------
# curve reads (descriptive rho + H4 prior adjustment)
# --------------------------------------------------------------------------


def _spearman(x: np.ndarray, y: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(x, y).statistic)


def curve_reads(data: Data) -> dict:
    """Per source x arm: Δ-vs-cosine Spearman (descriptive) + H4 prior partial."""
    from scipy.stats import pearsonr

    out: dict[str, dict] = {}
    for source in SOURCES:
        bys = data.bystanders(source)
        cosines = np.array([data.personas[b]["cosines"][source] for b in bys])
        priors = np.array([data.personas[b]["base_rate"] for b in bys])
        collin = float(pearsonr(cosines, priors).statistic) if len(bys) >= 3 else 0.0
        src_out: dict[str, dict] = {"collinearity_pearson_cos_prior": collin}
        for arm in TRAIN_ARMS:
            try:
                deltas = np.array([data.delta_pooled(arm, source, b) for b in bys])
            except KeyError:
                src_out[arm] = {"status": "missing"}
                continue
            rho_raw = _spearman(cosines, deltas)
            rec: dict = {"rho_raw": rho_raw, "n_bystanders": len(bys)}
            if abs(collin) > COLLINEARITY_GATE:
                # Pre-registered fallback: tercile-bucket medians by prior.
                terciles = np.quantile(priors, [1 / 3, 2 / 3])
                buckets = np.digitize(priors, terciles)
                rec["prior_adjusted"] = {
                    "method": "tercile_buckets (collinearity gate fired)",
                    "bucket_rho": {
                        str(t): (
                            _spearman(cosines[buckets == t], deltas[buckets == t])
                            if int((buckets == t).sum()) >= 4
                            else None
                        )
                        for t in (0, 1, 2)
                    },
                }
            else:
                r_dc = _spearman(deltas, cosines)
                r_dp = _spearman(deltas, priors)
                r_cp = _spearman(cosines, priors)
                denom = np.sqrt((1 - r_dp**2) * (1 - r_cp**2))
                rec["prior_adjusted"] = {
                    "method": "partial_spearman",
                    "rho_partial": float((r_dc - r_dp * r_cp) / denom) if denom > 0 else None,
                }
            raw = rec["rho_raw"]
            adj = rec["prior_adjusted"].get("rho_partial")
            rec["h4_within_band"] = bool(abs(adj - raw) <= 0.15) if isinstance(adj, float) else None
            src_out[arm] = rec
        out[source] = src_out
    return out


# --------------------------------------------------------------------------
# H3 anomaly cells
# --------------------------------------------------------------------------


def _pooled_delta_ci(
    data: Data, arm: str, source: str, panel: str, rng
) -> tuple[float, float, float]:
    """Seed-pooled Δ with paired claim bootstrap CI."""
    cms = [data.cell_cm[(arm, source, s, panel)] for s in SEEDS]
    base = data.base_cm[panel]
    claims = sorted(set(base) & set(cms[0]) & set(cms[1]))
    t = np.array([[cm[c] for c in claims] for cm in cms]).mean(axis=0)
    b = np.array([base[c] for c in claims])
    point = float(np.mean(t - b))
    boots = np.empty(BOOTSTRAP_B)
    n = len(claims)
    diffs = t - b
    for i in range(BOOTSTRAP_B):
        idx = rng.integers(0, n, n)
        boots[i] = diffs[idx].mean()
    return point, float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))


def anomaly_reads(data: Data) -> dict:
    rng = np.random.default_rng(BOOTSTRAP_SEED + 3)
    out: dict[str, dict] = {}
    for panel in ANOMALY_CELLS:
        per_arm: dict[str, dict] = {}
        for arm in TRAIN_ARMS:
            try:
                point, lo, hi = _pooled_delta_ci(data, arm, ANOMALY_SOURCE, panel, rng)
            except KeyError:
                per_arm[arm] = {"status": "missing"}
                continue
            per_seed = {str(s): data.delta(arm, ANOMALY_SOURCE, s, panel) for s in SEEDS}

            def seed_call(d: float) -> str:
                if abs(d) < FLAT_BAND:
                    return "flat"
                if d >= LEAK_TAU:
                    return "leak"
                return "indeterminate"

            calls = {s: seed_call(v) for s, v in per_seed.items()}
            if abs(point) < FLAT_BAND:
                call = "survives"
            elif point >= LEAK_TAU and (lo > 0):
                call = "dissolves"
            else:
                call = "indeterminate"
            if len(set(calls.values())) > 1 and "leak" in calls.values():
                call = "seed_split_indeterminate"
            per_arm[arm] = {
                "delta_pooled": point,
                "ci95": [lo, hi],
                "per_seed_delta": per_seed,
                "per_seed_calls": calls,
                "call": call,
            }
        out[panel] = per_arm
    headline_arms = ("arm_onpolicy", "arm_prefix")
    survives_all = all(
        out[p].get(a, {}).get("call") == "survives" for p in ANOMALY_CELLS for a in headline_arms
    )
    dissolves_any = any(
        out[p].get(a, {}).get("call") == "dissolves" for p in ANOMALY_CELLS for a in headline_arms
    )
    out["headline"] = (
        "survives" if survives_all else ("dissolves" if dissolves_any else "mixed_indeterminate")
    )
    return out


# --------------------------------------------------------------------------
# manipulation / dose / parity / trajectory
# --------------------------------------------------------------------------


def manipulation_check(data: Data) -> dict:
    out: dict[str, dict] = {}
    for arm in TRAIN_ARMS:
        per_source = {}
        for source in SOURCES:
            try:
                pooled = data.delta_pooled(arm, source, source)
                per_seed = {str(s): data.delta(arm, source, s, source) for s in SEEDS}
            except KeyError:
                per_source[source] = {"status": "missing"}
                continue
            per_source[source] = {"delta_self_pooled": pooled, "per_seed": per_seed}
        n_pass = sum(
            1
            for v in per_source.values()
            if isinstance(v.get("delta_self_pooled"), float)
            and v["delta_self_pooled"] >= SELF_IMPLANT_FLOOR
        )
        out[arm] = {
            "per_source": per_source,
            "n_sources_above_floor": n_pass,
            "floor": SELF_IMPLANT_FLOOR,
            "pass": n_pass >= 3,
            # Plan §7 K2: self-implant below floor on >=2 of 4 sources demotes
            # the arm's leakage contrasts to descriptive  <=>  n_pass <= 2.
            "k2_demote_to_descriptive": n_pass <= 2,
        }
    means = {
        arm: float(
            np.mean(
                [
                    v["delta_self_pooled"]
                    for v in out[arm]["per_source"].values()
                    if isinstance(v.get("delta_self_pooled"), float)
                ]
            )
        )
        for arm in TRAIN_ARMS
        if any(
            isinstance(v.get("delta_self_pooled"), float) for v in out[arm]["per_source"].values()
        )
    }
    spread = max(means.values()) - min(means.values()) if len(means) > 1 else 0.0
    out["arm_mean_self_implants"] = means
    out["dose_spread"] = spread
    out["band_entry_fallback_required"] = bool(spread > DOSE_MATCH_BAND)
    return out


def trajectory_reads(slab_root: Path, data: Data) -> dict:
    out: dict[str, dict] = {}
    for source in SOURCES:
        for arm in TRAIN_ARMS:
            for seed in SEEDS:
                cell_dir = cell_slab_dir(slab_root, source, arm, seed)
                traj = {}
                for k in (1, 2):
                    d = cell_dir / "trajectory" / f"epoch_{k}"
                    if d.is_dir():
                        try:
                            cm = _claim_means(_load_judgments(d, source))
                            traj[f"epoch_{k}"] = float(np.mean(list(cm.values()))) - data.base_rate(
                                source
                            )
                        except FileNotFoundError:
                            traj[f"epoch_{k}"] = None
                try:
                    traj["epoch_3_endpoint"] = data.delta(arm, source, seed, source)
                except KeyError:
                    continue
                out[f"{source}:{arm}:{seed}"] = traj
    return out


def parity_gate(slab_root: Path) -> dict:
    analyze = json.loads((repo_root_from_module() / ANALYZE_SUMMARY_RELPATH).read_text())[
        "per_source"
    ]
    checks = []
    for source, panels in PARITY_PANELS.items():
        cell_dir = cell_slab_dir(slab_root, source, "parity", 42)
        if not cell_dir.is_dir():
            checks.append({"source": source, "status": "missing"})
            continue
        frozen_rates = analyze[source]["per_panel_trained_rate"]
        for panel in panels:
            verdicts = _load_judgments(cell_dir, panel)
            fresh = float(np.mean([int(bool(v["agreed"])) for v in verdicts]))
            frozen = frozen_rates.get(panel)
            if frozen is None:
                checks.append({"source": source, "panel": panel, "status": "no_frozen_ref"})
                continue
            drift = fresh - frozen
            checks.append(
                {
                    "source": source,
                    "panel": panel,
                    "fresh": fresh,
                    "frozen": frozen,
                    "drift": drift,
                    "within_tol": abs(drift) <= PARITY_TOL,
                    "hard_fail": abs(drift) > PARITY_HARD_TOL,
                }
            )
    evaluated = [c for c in checks if "drift" in c]
    n_out = sum(1 for c in evaluated if not c["within_tol"])
    n_hard = sum(1 for c in evaluated if c["hard_fail"])
    verdict = "PASS"
    if n_hard > 0 or n_out >= 2:
        verdict = "HARD_FAIL"
    elif n_out == 1:
        verdict = "MARGINAL_MISS"
    return {
        "checks": checks,
        "n_out_of_tol": n_out,
        "n_hard_fail": n_hard,
        "tolerance": PARITY_TOL,
        "hard_tolerance": PARITY_HARD_TOL,
        "verdict": verdict,
    }


def frozen_subset_rates(data: Data) -> dict:
    """Per cell self + bystander-mean rates restricted to retained frozen claims."""
    out: dict[str, dict] = {}
    fi = data.frozen_claim_idx
    for source in SOURCES:
        for arm in TRAIN_ARMS:
            for seed in SEEDS:
                key = (arm, source, seed, source)
                if key not in data.cell_cm:
                    continue
                cm = data.cell_cm[key]
                sub = [cm[c] for c in cm if c in fi]
                out[f"{source}:{arm}:{seed}"] = {
                    "self_rate_frozen_subset": float(np.mean(sub)) if sub else None,
                    "n_frozen_claims": len(sub),
                }
    return out


# --------------------------------------------------------------------------
# dose-matched stage (plans/v2.md §3/§5; followup dose-matched-leakage-read)
# --------------------------------------------------------------------------

DM_BOUNDED_BAND = H1_SUPPORT_MIN  # ±0.05 — the registered support threshold (plan v2 §5)
DM_PRIMARY_DROP: tuple[int, ...] = (48,)
DM_SENSITIVITY_DROP: tuple[int, ...] = (48, 8)
DM_CLAIM_TEXT_ASSERTS: dict[int, str] = {48: "Amazon River", 8: "Mount Everest"}
DM_ANOMALY_PROBES = ("virtual_assistant", "digital_helper", "daycare_teacher")
DM_EVALUATED_ROLES = ("registered_contrast", "descriptive_prefix")
# Parent §5 reference triple ("endpoint ratios 0.154/0.182/0.244"); the per-arm
# endpoint ratios are recomputed independently below — the triple is a citation,
# not a mapping.
DM_PARENT_PUBLISHED_RATIOS = (0.154, 0.182, 0.244)


def _assert_claim_texts(eval60_path: Path, expected: dict[int, str]) -> None:
    """Plan v2 §2: assert the idx -> claim-text mapping before any subset is formed."""
    rows = [json.loads(ln) for ln in eval60_path.read_text().splitlines() if ln.strip()]
    for idx, frag in expected.items():
        claim = rows[idx]["wrong_claim"]
        if frag not in claim:
            raise RuntimeError(
                f"claim-subset idx->text assert FAILED: eval_60[{idx}] = {claim!r} does not "
                f"contain {frag!r} — the claim file moved; halt (plan v2 §2)."
            )


def _dm_retained(n_total: int, dropped: tuple[int, ...]) -> set[int]:
    return set(range(n_total)) - set(dropped)


def _dm_cell_dirs(
    slab_root: Path, selection: dict, sources: set[str] | None = None
) -> dict[tuple[str, str, int], Path]:
    """{(arm, source, seed): band-entry eval dir} for the evaluated cells."""
    out: dict[tuple[str, str, int], Path] = {}
    for rec in selection["cells"].values():
        rel = rec.get("eval_dir_rel")
        if not rel:
            continue
        if sources is not None and rec["source"] not in sources:
            continue
        out[(rec["arm"], rec["source"], rec["seed"])] = Path(slab_root) / rel
    return out


def _endpoint_cell_dirs(
    slab_root: Path, sources: tuple[str, ...]
) -> dict[tuple[str, str, int], Path]:
    return {
        (arm, source, seed): cell_slab_dir(slab_root, source, arm, seed)
        for source in sources
        for arm in TRAIN_ARMS
        for seed in SEEDS
    }


def dm_secondary_read(contrast: dict) -> dict:
    """The PRE-REGISTERED bounded-below-support read (plan v2 §5): CI entirely
    within ±0.05 (the support threshold) but not within ±0.03 -> the determinate
    'any effect is smaller than the registered support threshold' call, with the
    realized equivalence bound max(|lo|, |hi|)."""
    if "ci95" not in contrast:
        return {"applies": False, "reason": contrast.get("status", "no_ci")}
    lo, hi = contrast["ci95"]
    within_support = (lo >= -DM_BOUNDED_BAND) and (hi <= DM_BOUNDED_BAND)
    within_null = (lo >= -H1_NULL_BAND) and (hi <= H1_NULL_BAND)
    applies = bool(
        within_support and not within_null and contrast["verdict"] not in ("supported", "null")
    )
    rec: dict = {
        "applies": applies,
        "ci_within_support_band": bool(within_support),
        "ci_within_null_band": bool(within_null),
        "support_band": DM_BOUNDED_BAND,
        "null_band": H1_NULL_BAND,
        "equivalence_bound": float(max(abs(lo), abs(hi))) if within_support else None,
    }
    if applies:
        rec["call"] = "bounded_below_support"
    return rec


def dm_resolution(contrast: dict, secondary: dict) -> str:
    """Primary three-way verdict, with the registered secondary read layered on
    the indeterminate branch (never overriding supported/null)."""
    if "verdict" not in contrast:
        return contrast.get("status", "not_formable")
    if secondary.get("applies"):
        return "bounded_below_support"
    return contrast["verdict"]


def dm_per_source_contrast(data: Data, arm_x: str, arm_y: str) -> dict[str, dict]:
    """Per-source contrasts with a CLEARLY-LABELED 1-seed descriptive path
    (plan v2 §3 implementation note: at band entry comedian contributes only
    seed 137, which ``per_source_contrast`` would report as 'missing')."""
    rng = np.random.default_rng(BOOTSTRAP_SEED + 4)
    out: dict[str, dict] = {}
    for source in SOURCES:
        mats: dict[int, np.ndarray] = {}
        pts: dict[int, float] = {}
        for seed in SEEDS:
            mat, pairs, _ = _contrast_matrix(data, arm_x, arm_y, seed)
            rows = [i for i, (s, _) in enumerate(pairs) if s == source]
            if rows:
                mats[seed] = mat[rows]
                pts[seed] = float(np.nanmean(mat[rows]))
        if not mats:
            out[source] = {"status": "missing"}
            continue
        boots = np.empty(2000)
        for b in range(2000):
            sm = []
            for m in mats.values():
                ri = rng.integers(0, m.shape[0], m.shape[0])
                cj = rng.integers(0, m.shape[1], m.shape[1])
                sm.append(np.nanmean(m[np.ix_(ri, cj)]))
            boots[b] = np.mean(sm)
        rec: dict = {
            "point_seed_mean": float(np.mean(list(pts.values()))),
            "per_seed": {str(s): v for s, v in pts.items()},
            "ci95": [float(np.quantile(boots, 0.025)), float(np.quantile(boots, 0.975))],
            "n_seeds": len(mats),
        }
        if len(mats) < len(SEEDS):
            rec["status"] = "descriptive_single_seed"
            rec["note"] = (
                f"only seed(s) {sorted(mats)} reach the band for this source — "
                f"descriptive, not a registered per-source read"
            )
        out[source] = rec
    return out


def _mean_bystander_delta(data: Data, arm: str, source: str, seed: int) -> float:
    return float(np.mean([data.delta(arm, source, seed, b) for b in data.bystanders(source)]))


def dm_interpretation_map(dose_data: Data) -> dict:
    """PRE-REGISTERED decision-interpretation map (plan v2 §5): residual-dose
    bound = realized per-pair dose gap x band-entry leakage-per-dose ratio,
    per H1-dm pair and pooled."""
    per_pair: dict[str, dict] = {}
    for source in SOURCES:
        for seed in SEEDS:
            kc = ("arm_canned", source, seed, source)
            ko = ("arm_onpolicy", source, seed, source)
            if kc not in dose_data.cell_cm or ko not in dose_data.cell_cm:
                continue
            self_c = dose_data.delta("arm_canned", source, seed, source)
            self_o = dose_data.delta("arm_onpolicy", source, seed, source)
            gap = self_c - self_o
            ratio_c = _mean_bystander_delta(dose_data, "arm_canned", source, seed) / self_c
            ratio_o = _mean_bystander_delta(dose_data, "arm_onpolicy", source, seed) / self_o
            ratio_mean = float(np.mean([ratio_c, ratio_o]))
            per_pair[f"{source}:seed{seed}"] = {
                "self_delta_canned_band_entry": self_c,
                "self_delta_onpolicy_band_entry": self_o,
                "dose_gap": gap,
                "leakage_per_dose_canned": ratio_c,
                "leakage_per_dose_onpolicy": ratio_o,
                "leakage_per_dose_mean": ratio_mean,
                "residual_dose_bound": gap * ratio_mean,
            }
    pooled = (
        float(np.mean([p["residual_dose_bound"] for p in per_pair.values()])) if per_pair else None
    )
    return {
        "per_pair": per_pair,
        "pooled_residual_dose_bound": pooled,
        "expected_bias_sign": (
            "negative — the canned cells carry the higher residual dose even at band entry, "
            "pushing the on-policy-minus-canned contrast down"
        ),
        "branch_licenses": {
            "supported_positive": (
                "clean — residual dose cannot produce a positive contrast; kills 'dose, not "
                "radius' outright"
            ),
            "supported_negative": (
                "dose-confounded unless |point| clearly exceeds the pooled residual-dose "
                "bound; a negative point inside the bound is 'consistent with residual dose "
                "at this granularity', not a radius effect"
            ),
            "null_or_bounded_below_support": (
                "licenses 'no detectable matched-dial contrast (any effect < the support "
                "threshold), with the residual dose bias carried' — NOT 'dose artifact "
                "confirmed'; exact cancellation of a true positive radius effect against "
                "the dose bias remains a named, unresolvable-at-this-granularity rival"
            ),
        },
    }


def dm_cell_level_reads(dose_data: Data, endpoint_data: Data, selection: dict) -> tuple[dict, dict]:
    """Per-cell self/bystander deltas at band entry vs endpoint -> the
    leakage-per-dose and within-cell dose-response descriptive reads (plan v2 §4)."""
    lpd_per_cell: dict[str, dict] = {}
    dr_per_cell: dict[str, dict] = {}
    arm_band: dict[str, list[float]] = {}
    arm_end: dict[str, list[float]] = {}
    for cid, rec in selection["cells"].items():
        if rec["role"] not in DM_EVALUATED_ROLES:
            continue
        arm, source, seed = rec["arm"], rec["source"], rec["seed"]
        if (arm, source, seed, source) not in dose_data.cell_cm:
            lpd_per_cell[cid] = {"status": "missing"}
            continue
        self_b = dose_data.delta(arm, source, seed, source)
        by_b = _mean_bystander_delta(dose_data, arm, source, seed)
        cell: dict = {
            "self_delta_band_entry": self_b,
            "bystander_mean_delta_band_entry": by_b,
            "ratio_band_entry": by_b / self_b,
        }
        arm_band.setdefault(arm, []).append(by_b / self_b)
        if (arm, source, seed, source) in endpoint_data.cell_cm:
            self_e = endpoint_data.delta(arm, source, seed, source)
            by_e = _mean_bystander_delta(endpoint_data, arm, source, seed)
            cell.update(
                self_delta_endpoint=self_e,
                bystander_mean_delta_endpoint=by_e,
                ratio_endpoint=by_e / self_e,
            )
            arm_end.setdefault(arm, []).append(by_e / self_e)
            dr_per_cell[cid] = {
                "bystander_mean_delta_band_entry": by_b,
                "bystander_mean_delta_endpoint": by_e,
                "endpoint_minus_band_entry": by_e - by_b,
                "self_delta_band_entry": self_b,
                "self_delta_endpoint": self_e,
                "self_endpoint_minus_band_entry": self_e - self_b,
            }
        lpd_per_cell[cid] = cell
    leakage = {
        "per_cell": lpd_per_cell,
        "per_arm_band_entry": {a: float(np.mean(v)) for a, v in arm_band.items()},
        "per_arm_endpoint_recomputed": {a: float(np.mean(v)) for a, v in arm_end.items()},
        "parent_published_endpoint_ratios": list(DM_PARENT_PUBLISHED_RATIOS),
        "note": (
            "published triple is the parent's exploratory per-arm reference (plan v2 §5); "
            "the per-arm mapping is recomputed independently here"
        ),
    }
    dose_response = {
        "per_cell": dr_per_cell,
        "note": (
            "within-cell endpoint-minus-band-entry moves installed dose AND accumulated "
            "contrastive-negative training together (plan v2 §13 item 3)"
        ),
    }
    return leakage, dose_response


def dm_gradient_rho(dose_data: Data, endpoint_data: Data, selection: dict) -> dict:
    """Spearman rho(Δ, cosine) over bystanders per evaluated cell, band entry vs
    endpoint (the 'radius unchanged' descriptive leg, plan v2 §4)."""
    out: dict[str, dict] = {}
    for cid, rec in selection["cells"].items():
        if rec["role"] not in DM_EVALUATED_ROLES:
            continue
        arm, source, seed = rec["arm"], rec["source"], rec["seed"]
        bys = dose_data.bystanders(source)
        cos = np.array([dose_data.personas[b]["cosines"][source] for b in bys])
        try:
            d_band = np.array([dose_data.delta(arm, source, seed, b) for b in bys])
        except KeyError:
            out[cid] = {"status": "missing"}
            continue
        rec_out: dict = {"rho_band_entry": _spearman(cos, d_band), "n_bystanders": len(bys)}
        try:
            d_end = np.array([endpoint_data.delta(arm, source, seed, b) for b in bys])
            rec_out["rho_endpoint"] = _spearman(cos, d_end)
        except KeyError:
            rec_out["rho_endpoint"] = None
        out[cid] = rec_out
    return out


def dm_anomaly_probes(dose_data: Data, selection: dict) -> dict:
    """Band-entry Δ for the anomaly probe personas under the evaluated cells
    (descriptive; the designed realistic-data test remains unrunnable — no
    software-engineer on-policy cells exist)."""
    out: dict[str, dict] = {}
    for probe in DM_ANOMALY_PROBES:
        per: dict[str, float] = {}
        for cid, rec in selection["cells"].items():
            if rec["role"] not in DM_EVALUATED_ROLES:
                continue
            key = (rec["arm"], rec["source"], rec["seed"], probe)
            if key not in dose_data.cell_cm:
                continue
            per[cid] = dose_data.delta(rec["arm"], rec["source"], rec["seed"], probe)
        out[probe] = per
    return out


def dm_self_vs_trajectory(dose_data_60: Data, selection: dict) -> dict:
    """Checkpoint-identity diagnostic (plan v2 §13 item 5): each new full-panel
    self Δ (60-claim continuity set) vs the selected-epoch trajectory value and
    the endpoint trajectory value."""
    out: dict[str, dict] = {}
    for cid, rec in selection["cells"].items():
        if rec["role"] not in DM_EVALUATED_ROLES:
            continue
        arm, source, seed = rec["arm"], rec["source"], rec["seed"]
        epoch = rec["band_entry_epoch"]
        if (arm, source, seed, source) not in dose_data_60.cell_cm:
            out[cid] = {"status": "missing"}
            continue
        new_self = dose_data_60.delta(arm, source, seed, source)
        traj = rec["trajectory_delta"]
        traj_at_epoch = traj[f"epoch_{epoch}"] if epoch in (1, 2) else traj["epoch_3_endpoint"]
        out[cid] = {
            "band_entry_epoch": epoch,
            "new_self_delta_60_claims": new_self,
            "trajectory_delta_at_selected_epoch": traj_at_epoch,
            "trajectory_delta_endpoint": traj["epoch_3_endpoint"],
            "new_minus_trajectory": new_self - traj_at_epoch,
        }
    return out


def dm_install_failures(selection: dict) -> list[dict]:
    out = []
    for cid, rec in selection["cells"].items():
        if rec["role"] != "install_failure":
            continue
        traj = rec["trajectory_delta"]
        closest = max(traj, key=lambda k: traj[k] if traj[k] is not None else -np.inf)
        out.append(
            {
                "cell": cid,
                "max_delta": rec["max_delta"],
                "closest_approach_epoch": closest,
                "trajectory_delta": traj,
                "threshold": selection["threshold"],
            }
        )
    return out


def make_dm_figures(  # noqa: C901 - one linear pass per registered figure
    analysis: dict, dose59: Data, end59: Data, selection: dict, figures_dir: Path
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from explore_persona_space.analysis.paper_plots import (
            paper_palette,
            savefig_paper,
            set_paper_style,
        )

        set_paper_style()
        colors = paper_palette(3)

        def save(fig, name):
            savefig_paper(fig, name, dir=figures_dir)
            plt.close(fig)
            return str(figures_dir / f"{name}.png")
    except Exception:
        colors = ["#0173b2", "#de8f05", "#029e73"]

        def save(fig, name):
            figures_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(figures_dir / f"{name}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            return str(figures_dir / f"{name}.png")

    written: list[str] = []
    arm_color = dict(zip(TRAIN_ARMS, colors, strict=True))
    arm_label = {
        "arm_canned": "canned anchor",
        "arm_onpolicy": "on-policy single-turn",
        "arm_prefix": "on-policy multi-turn prefix",
    }
    from matplotlib.patches import Patch

    arm_handles = [Patch(color=arm_color[a], label=arm_label[a]) for a in TRAIN_ARMS]

    def cid_label(cid: str) -> str:
        source, arm, seed = cid.split(":")
        return f"{source.replace('_', ' ')} {arm_label[arm]} seed {seed}"

    # Hero — dose-matched vs endpoint contrast forest (band-entry and endpoint
    # rows side by side, per-source rows beneath; plain-English labels only).
    rows: list[tuple[float, list[float]]] = []
    labels: list[str] = []

    def add_row(contrast: dict, label: str) -> None:
        if "point_seed_mean" in contrast:
            rows.append((contrast["point_seed_mean"], contrast["ci95"]))
            labels.append(label)

    add_row(analysis["h1_dose_matched"]["contrast"], "all pairs: band entry")
    add_row(analysis["h1_endpoint_recomputed"], "all pairs: endpoint (same claims)")
    add_row(
        analysis["villain_only_robustness"]["band_entry"]["contrast"],
        "villain only: band entry",
    )
    add_row(analysis["villain_only_robustness"]["endpoint"], "villain only: endpoint")
    for source, rec in analysis["h1_dose_matched_per_source"].items():
        tag = (
            " (seed 137 only, descriptive)"
            if rec.get("status") == "descriptive_single_seed"
            else ""
        )
        add_row(rec, f"  {source.replace('_', ' ')}: band entry{tag}")
    for source, rec in analysis["h1_endpoint_recomputed_per_source"].items():
        add_row(rec, f"  {source.replace('_', ' ')}: endpoint")
    fig, ax = plt.subplots(figsize=(8.5, 0.5 * len(rows) + 2))
    ypos = np.arange(len(rows))[::-1]
    for y, (pt, (lo, hi)) in zip(ypos, rows, strict=True):
        ax.errorbar(
            pt,
            y,
            xerr=[[max(0.0, pt - lo)], [max(0.0, hi - pt)]],
            fmt="o",
            color="#0173b2",
            capsize=3,
        )
    ax.axvline(0, color="black", lw=0.8)
    for band in (DM_BOUNDED_BAND, -DM_BOUNDED_BAND):
        ax.axvline(band, ls=":", lw=0.8, color="grey")
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    # Keep the xlabel short — the long cluster-bootstrap phrasing clipped at the
    # canvas edge (caption carries the bootstrap detail).
    ax.set_xlabel("bystander contrast: on-policy minus canned (95% CI)")
    written.append(save(fig, "dm_hero_contrast_forest"))

    # Exploratory — per-bystander Δ at band entry vs endpoint.
    fig, ax = plt.subplots(figsize=(6, 6))
    for _cid, rec in selection["cells"].items():
        if rec["role"] not in DM_EVALUATED_ROLES:
            continue
        arm, source, seed = rec["arm"], rec["source"], rec["seed"]
        for b in dose59.bystanders(source):
            try:
                x = dose59.delta(arm, source, seed, b)
                y = end59.delta(arm, source, seed, b)
            except KeyError:
                continue
            ax.scatter(x, y, s=14, color=arm_color[arm], alpha=0.7)
    lims = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lims), max(lims)
    ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8, ls="--")
    ax.set_xlabel("bystander Δ agreement at band entry")
    ax.set_ylabel("bystander Δ agreement at endpoint")
    ax.legend(handles=arm_handles, fontsize=8)
    written.append(save(fig, "dm_endpoint_vs_band_entry"))

    # Exploratory — gradient shape: Spearman rho at band entry vs endpoint per cell.
    fig, ax = plt.subplots(figsize=(6, 6))
    for cid, rec in analysis["gradient_rho"].items():
        if not isinstance(rec.get("rho_band_entry"), float) or not isinstance(
            rec.get("rho_endpoint"), float
        ):
            continue
        arm = cid.split(":")[1]
        ax.scatter(rec["rho_band_entry"], rec["rho_endpoint"], color=arm_color[arm], s=36)
    lims = ax.get_xlim() + ax.get_ylim()
    lo, hi = min(lims), max(lims)
    ax.plot([lo, hi], [lo, hi], color="grey", lw=0.8, ls="--")
    ax.set_xlabel("leakage-vs-similarity Spearman correlation at band entry")
    ax.set_ylabel("leakage-vs-similarity Spearman correlation at endpoint")
    ax.legend(handles=arm_handles, fontsize=8)
    written.append(save(fig, "dm_gradient_rho_shift"))

    # Exploratory — leakage per dose, band entry vs recomputed endpoint, per arm.
    lpd = analysis["leakage_per_dose"]
    arms = [a for a in TRAIN_ARMS if a in lpd["per_arm_band_entry"]]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    x = np.arange(len(arms))
    ax.bar(
        x - 0.18,
        [lpd["per_arm_band_entry"][a] for a in arms],
        width=0.36,
        color="#0173b2",
        label="band entry",
    )
    end_vals = [lpd["per_arm_endpoint_recomputed"].get(a) for a in arms]
    if all(v is not None for v in end_vals):
        ax.bar(x + 0.18, end_vals, width=0.36, color="#de8f05", label="endpoint")
    ax.set_xticks(x)
    ax.set_xticklabels([arm_label[a] for a in arms], fontsize=8)
    ax.set_ylabel("mean bystander Δ per unit self Δ")
    ax.legend(fontsize=8)
    written.append(save(fig, "dm_leakage_per_dose"))

    # Exploratory — per-seed contrast points (coverage asymmetry made visible).
    fig, ax = plt.subplots(figsize=(6, 4))
    band_ps = analysis["h1_dose_matched"]["contrast"].get("per_seed_points", {})
    end_ps = analysis["h1_endpoint_recomputed"].get("per_seed_points", {})
    seeds = sorted(band_ps)
    xs = np.arange(len(seeds))
    ax.scatter(xs - 0.08, [band_ps[s] for s in seeds], color="#0173b2", label="band entry", s=48)
    if end_ps:
        ax.scatter(xs + 0.08, [end_ps[s] for s in seeds], color="#de8f05", label="endpoint", s=48)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(xs)
    ax.set_xticklabels(
        [f"seed {s}" + (" (villain only)" if s == "42" else "") for s in seeds], fontsize=8
    )
    ax.set_ylabel("per-seed paired contrast: on-policy minus canned")
    ax.legend(fontsize=8)
    written.append(save(fig, "dm_per_seed_contrasts"))

    # Exploratory — anomaly probe strip at band entry.
    probes = list(analysis["anomaly_probes"])
    fig, axes = plt.subplots(1, len(probes), figsize=(4 * len(probes), 4), sharey=True)
    for ax, probe in zip(np.atleast_1d(axes), probes, strict=True):
        per = analysis["anomaly_probes"][probe]
        cids = sorted(per)
        for i, cid in enumerate(cids):
            ax.bar(i, per[cid], color=arm_color[cid.split(":")[1]], width=0.85)
        ax.set_xticks(range(len(cids)))
        ax.set_xticklabels([cid_label(c) for c in cids], fontsize=6, rotation=60, ha="right")
        ax.axhline(0, color="black", lw=0.6)
        ax.set_title(probe.replace("_", " "))
        ax.set_ylabel("Δ agreement at band entry")
    written.append(save(fig, "dm_anomaly_probes"))

    # Exploratory — claim-subset sensitivity (all 60 / 59 primary / 58 sensitivity).
    sens = analysis["claim_subset_sensitivity"]
    tags = ["continuity_60", "primary_59", "sensitivity_58"]
    tag_label = {
        "continuity_60": "all 60 claims",
        "primary_59": "59 claims (primary)",
        "sensitivity_58": "58 claims (sensitivity)",
    }
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for j, (side, color) in enumerate((("band_entry", "#0173b2"), ("endpoint", "#de8f05"))):
        pts = [sens[t][side] for t in tags]
        xs = np.arange(len(tags)) + (j - 0.5) * 0.16
        for x_pos, c in zip(xs, pts, strict=True):
            if "point_seed_mean" not in c:
                continue
            pt, (lo, hi) = c["point_seed_mean"], c["ci95"]
            ax.errorbar(
                x_pos,
                pt,
                yerr=[[max(0.0, pt - lo)], [max(0.0, hi - pt)]],
                fmt="o",
                color=color,
                capsize=3,
                label=side.replace("_", " ") if x_pos == xs[0] else None,
            )
    ax.axhline(0, color="black", lw=0.8)
    ax.set_xticks(np.arange(len(tags)))
    ax.set_xticklabels([tag_label[t] for t in tags], fontsize=8)
    ax.set_ylabel("pooled contrast: on-policy minus canned")
    ax.legend(fontsize=8)
    written.append(save(fig, "dm_claim_subset_sensitivity"))
    return written


def run_dose_matched(args: argparse.Namespace) -> int:
    """The --stage dose-matched flow (plan v2 §3 item 5 + §5)."""
    selection_path = args.selection or (
        args.slab_root / "dose_matched" / "band_entry_selection.json"
    )
    if not selection_path.exists():
        raise FileNotFoundError(
            f"band_entry_selection.json missing: {selection_path} (run band_entry first)"
        )
    selection = json.loads(selection_path.read_text())
    _assert_claim_texts(args.claims, DM_CLAIM_TEXT_ASSERTS)
    n_total = sum(1 for ln in args.claims.read_text().splitlines() if ln.strip())

    subsets: dict[str, set[int] | None] = {
        "continuity_60": None,
        "primary_59": _dm_retained(n_total, DM_PRIMARY_DROP),
        "sensitivity_58": _dm_retained(n_total, DM_SENSITIVITY_DROP),
    }
    dose_dirs = _dm_cell_dirs(args.slab_root, selection)

    sensitivity: dict[str, dict] = {}
    datas: dict[str, tuple[Data, Data]] = {}
    for tag, subset in subsets.items():
        dose_data = Data(
            args.slab_root, args.panel_set, args.claims, claim_subset=subset, cell_dirs=dose_dirs
        )
        endpoint_data = Data(args.slab_root, args.panel_set, args.claims, claim_subset=subset)
        datas[tag] = (dose_data, endpoint_data)
        sensitivity[tag] = {
            # H1-dm uses paired_arm_contrast VERBATIM (plan v2 §3 item 5).
            "band_entry": paired_arm_contrast(dose_data, "arm_onpolicy", "arm_canned"),
            "endpoint": paired_arm_contrast(endpoint_data, "arm_onpolicy", "arm_canned"),
        }

    dose59, end59 = datas["primary_59"]
    dose60, _ = datas["continuity_60"]

    h1dm = sensitivity["primary_59"]["band_entry"]
    h1dm_secondary = dm_secondary_read(h1dm)
    headline_resolution = dm_resolution(h1dm, h1dm_secondary)

    # Registered villain-only robustness read (the one source with 2-seed coverage).
    villain_band_data = Data(
        args.slab_root,
        args.panel_set,
        args.claims,
        claim_subset=subsets["primary_59"],
        cell_dirs=_dm_cell_dirs(args.slab_root, selection, sources={"villain"}),
    )
    villain_band = paired_arm_contrast(villain_band_data, "arm_onpolicy", "arm_canned")
    villain_secondary = dm_secondary_read(villain_band)
    villain_resolution = dm_resolution(villain_band, villain_secondary)
    villain_end_data = Data(
        args.slab_root,
        args.panel_set,
        args.claims,
        claim_subset=subsets["primary_59"],
        cell_dirs=_endpoint_cell_dirs(args.slab_root, ("villain",)),
    )
    villain_end = paired_arm_contrast(villain_end_data, "arm_onpolicy", "arm_canned")

    # H2-dm: pre-registered NOT formable (zero complete pairs at band entry).
    h2 = paired_arm_contrast(dose59, "arm_prefix", "arm_onpolicy")
    if h2.get("status") != "no_paired_cells":
        raise RuntimeError(
            "H2-dm was pre-registered NOT formable (plan v2 §5) but the data formed "
            "pairs — the registered structure moved; halt, needs human eyes."
        )

    leakage_per_dose, dose_response = dm_cell_level_reads(dose59, end59, selection)
    analysis: dict = {
        "stage": "dose-matched",
        "followup_label": "dose-matched-leakage-read",
        "h1_dose_matched": {
            "contrast": h1dm,
            "secondary_read": h1dm_secondary,
            "resolution": headline_resolution,
        },
        "h1_dose_matched_per_source": dm_per_source_contrast(dose59, "arm_onpolicy", "arm_canned"),
        "h1_endpoint_recomputed": sensitivity["primary_59"]["endpoint"],
        "h1_endpoint_recomputed_per_source": per_source_contrast(
            end59, "arm_onpolicy", "arm_canned"
        ),
        "villain_only_robustness": {
            "band_entry": {
                "contrast": villain_band,
                "secondary_read": villain_secondary,
                "resolution": villain_resolution,
            },
            "endpoint": villain_end,
            "branch_agreement": bool(headline_resolution == villain_resolution),
            "combined_call": (
                headline_resolution
                if headline_resolution == villain_resolution
                else "indeterminate_with_structure"
            ),
        },
        "claim_subset_sensitivity": sensitivity,
        "h2_dose_matched": {
            "status": "preregistered_not_formable",
            "paired_arm_contrast": h2,
            "read": (
                "prefix install failure — the prefix condition cannot be dose-matched "
                "into the band at this recipe; no substitute statistic (plan v2 §5)"
            ),
        },
        "install_failures": dm_install_failures(selection),
        "interpretation_map": dm_interpretation_map(dose59),
        "leakage_per_dose": leakage_per_dose,
        "within_cell_dose_response": dose_response,
        "gradient_rho": dm_gradient_rho(dose59, end59, selection),
        "anomaly_probes": dm_anomaly_probes(dose59, selection),
        "self_vs_trajectory_crosscheck": dm_self_vs_trajectory(dose60, selection),
        "coverage": {
            "seed_42": "villain only (comedian on-policy seed 42 never entered the band)",
            "seed_137": "villain + comedian (comedian on-policy at its epoch-2 band entry)",
            "note": (
                "sign agreement is computed on these two per-seed estimates as constituted "
                "(plan v2 §5 registered coverage handling)"
            ),
        },
        "missing_dose_cells": dose59.missing_cells,
        "thresholds": {
            "band_entry_threshold": selection["threshold"],
            "h1_support_min": H1_SUPPORT_MIN,
            "h1_null_band": H1_NULL_BAND,
            "bounded_below_support_band": DM_BOUNDED_BAND,
            "g1_dm_tolerance": selection["g1_dm"]["tolerance"],
            "bootstrap_B": BOOTSTRAP_B,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "primary_claim_drop": list(DM_PRIMARY_DROP),
            "sensitivity_claim_drop": list(DM_SENSITIVITY_DROP),
        },
        "selection": {
            "path": str(selection_path),
            "threshold": selection["threshold"],
            "evaluated_cells": selection["evaluated_cells"],
            "band_entry_epochs": {
                cid: selection["cells"][cid]["band_entry_epoch"]
                for cid in selection["evaluated_cells"]
            },
        },
        "metadata": {
            "git_commit_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "panel_set": str(args.panel_set),
            "claims": str(args.claims),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
    }
    if not args.skip_figures:
        analysis["figures"] = make_dm_figures(analysis, dose59, end59, selection, args.figures_dir)

    out_path = args.slab_root / "dose_matched" / "analysis_612_dose_matched.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(analysis, indent=2))
    log.info(
        "dose-matched analysis -> %s | resolution=%s (villain-only=%s, agreement=%s)",
        out_path,
        headline_resolution,
        villain_resolution,
        analysis["villain_only_robustness"]["branch_agreement"],
    )
    return 0


# --------------------------------------------------------------------------
# figures
# --------------------------------------------------------------------------


def make_figures(  # noqa: C901 - one linear pass per registered figure; splitting hides the set
    data: Data, analysis: dict, figures_dir: Path
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from explore_persona_space.analysis.paper_plots import (
            paper_palette,
            savefig_paper,
            set_paper_style,
        )

        set_paper_style()
        colors = paper_palette(3)

        def save(fig, name):
            # Pass dir= explicitly: savefig_paper's default dir="figures/" would
            # double-join a relative figures_dir ("figures/figures/issue_612/...").
            savefig_paper(fig, name, dir=figures_dir)
            plt.close(fig)
            return str(figures_dir / f"{name}.png")
    except Exception:  # paper_plots font setup unavailable -> plain matplotlib
        colors = ["#0173b2", "#de8f05", "#029e73"]

        def save(fig, name):
            figures_dir.mkdir(parents=True, exist_ok=True)
            fig.savefig(figures_dir / f"{name}.png", dpi=200, bbox_inches="tight")
            plt.close(fig)
            return str(figures_dir / f"{name}.png")

    written: list[str] = []
    arm_color = dict(zip(TRAIN_ARMS, colors, strict=True))
    arm_label = {
        "arm_canned": "canned anchor",
        "arm_onpolicy": "on-policy single-turn",
        "arm_prefix": "on-policy multi-turn prefix",
    }

    # Hero 1 — Δ vs cosine, 4 source panels, 3 arms, neg_member greyed.
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharey=True)
    for ax, source in zip(axes.flat, SOURCES, strict=True):
        for arm in TRAIN_ARMS:
            xs, ys, gx, gy = [], [], [], []
            for name, rec in data.personas.items():
                if name == source:
                    continue
                try:
                    d = data.delta_pooled(arm, source, name)
                except KeyError:
                    continue
                c = rec["cosines"][source]
                if source in rec.get("neg_member_for", []):
                    gx.append(c), gy.append(d)
                else:
                    xs.append(c), ys.append(d)
            ax.scatter(xs, ys, s=22, color=arm_color[arm], label=arm_label[arm], alpha=0.85)
            ax.scatter(gx, gy, s=18, color="lightgrey", marker="x")
            if len(xs) >= 4:
                from sklearn.isotonic import IsotonicRegression

                order = np.argsort(xs)
                iso = IsotonicRegression(increasing=True, out_of_bounds="clip")
                yhat = iso.fit_transform(np.array(xs)[order], np.array(ys)[order])
                ax.plot(np.array(xs)[order], yhat, color=arm_color[arm], lw=1.2, alpha=0.7)
        ax.axhline(LEAK_TAU, ls="--", lw=0.8, color="grey")
        ax.axhline(0, ls="-", lw=0.6, color="black", alpha=0.4)
        ax.set_title(source.replace("_", " "))
        ax.set_xlabel("layer-20 centroid cosine to source")
        ax.set_ylabel("Δ agreement (trained - base)")
    axes.flat[0].legend(fontsize=8)
    written.append(save(fig, "hero1_delta_vs_cosine"))

    # Hero 2 — arm-contrast forest (per-source + pooled), plain-English labels only.
    fig, ax = plt.subplots(figsize=(8.5, 6))
    rows, labels = [], []
    for pooled_tag, source_tag, key in (
        (
            "on-policy single-turn minus canned anchor",
            "on-policy minus canned",
            "h1_onpolicy_vs_canned",
        ),
        (
            "multi-turn prefix minus on-policy single-turn",
            "prefix minus single-turn",
            "h2_prefix_vs_onpolicy",
        ),
    ):
        pooled = analysis[key]
        if "point_seed_mean" not in pooled:  # descope: {"status": "no_paired_cells"}
            continue
        rows.append((pooled["point_seed_mean"], pooled["ci95"]))
        labels.append(f"all sources: {pooled_tag}")
        for source, rec in analysis[key + "_per_source"].items():
            if "point_seed_mean" in rec:
                rows.append((rec["point_seed_mean"], rec["ci95"]))
                labels.append(f"  {source.replace('_', ' ')}: {source_tag}")
    ypos = np.arange(len(rows))[::-1]
    for y, (pt, (lo, hi)) in zip(ypos, rows, strict=True):
        ax.errorbar(
            pt,
            y,
            xerr=[[max(0.0, pt - lo)], [max(0.0, hi - pt)]],
            fmt="o",
            color="#0173b2",
            capsize=3,
        )
    ax.axvline(0, color="black", lw=0.8)
    ax.set_yticks(ypos)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("pooled bystander Δ-contrast (95% cluster-bootstrap CI)")
    written.append(save(fig, "hero2_arm_contrast_forest"))

    # Hero 3 — anomaly strip (3 probe personas x sources x arms).
    probes = [*ANOMALY_CELLS, "daycare_teacher"]
    fig, axes = plt.subplots(1, len(probes), figsize=(4 * len(probes), 4), sharey=True)
    for ax, probe in zip(axes, probes, strict=True):
        xt, xl = [], []
        for i, source in enumerate(SOURCES):
            for j, arm in enumerate(TRAIN_ARMS):
                try:
                    d = data.delta_pooled(arm, source, probe)
                except KeyError:
                    continue
                x = i * (len(TRAIN_ARMS) + 1) + j
                ax.bar(x, d, color=arm_color[arm], width=0.9)
            xt.append(i * (len(TRAIN_ARMS) + 1) + 1)
            xl.append(source.replace("_", "\n"))
        ax.axhline(LEAK_TAU, ls="--", lw=0.8, color="grey")
        ax.axhline(FLAT_BAND, ls=":", lw=0.8, color="grey")
        ax.set_xticks(xt)
        ax.set_xticklabels(xl, fontsize=7)
        ax.set_title(probe.replace("_", " "))
        ax.set_ylabel("Δ agreement (trained - base)")
    from matplotlib.patches import Patch

    axes[0].legend(
        handles=[Patch(color=arm_color[a], label=arm_label[a]) for a in TRAIN_ARMS],
        fontsize=7,
        loc="upper left",
    )
    written.append(save(fig, "hero3_anomaly_strip"))

    # Exploratory: prior vs cosine per source (decorrelation view) + raw scatter.
    fig, axes = plt.subplots(1, len(SOURCES), figsize=(4 * len(SOURCES), 3.5))
    for ax, source in zip(axes, SOURCES, strict=True):
        xs = [rec["cosines"][source] for rec in data.personas.values()]
        ys = [rec["base_rate"] for rec in data.personas.values()]
        ax.scatter(xs, ys, s=18, color="#555")
        ax.set_title(source.replace("_", " "))
        ax.set_xlabel(f"cosine to {source.replace('_', ' ')}")
        ax.set_ylabel("base-model agreement prior")
    written.append(save(fig, "exploratory_prior_vs_cosine"))

    # Exploratory: self-implant trajectories per arm.
    from matplotlib.lines import Line2D

    fig, ax = plt.subplots(figsize=(7, 4.5))
    for key, traj in analysis["trajectory"].items():
        arm = key.split(":")[1]
        xs, ys = [], []
        for i, ep in enumerate(("epoch_1", "epoch_2", "epoch_3_endpoint"), 1):
            if traj.get(ep) is not None:
                xs.append(i), ys.append(traj[ep])
        ax.plot(xs, ys, marker="o", ms=3, lw=0.8, alpha=0.6, color=arm_color.get(arm, "grey"))
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels(["epoch 1", "epoch 2", "epoch 3"])
    ax.set_ylabel("self-implant Δ (trained - base)")
    ax.legend(
        handles=[
            Line2D([0], [0], color=arm_color[a], marker="o", ms=3, lw=0.8, label=arm_label[a])
            for a in TRAIN_ARMS
        ],
        fontsize=8,
    )
    written.append(save(fig, "exploratory_self_implant_trajectories"))
    return written


# --------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Task #612 P7 — registered analyses + figures (VM).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--slab-root", type=Path, default=Path("eval_results/issue_612"))
    parser.add_argument(
        "--panel-set", type=Path, default=Path("data/issue_612/panel/panel_set.json")
    )
    parser.add_argument(
        "--claims", type=Path, default=Path("data/issue_612/wrong_claims/eval_60.jsonl")
    )
    parser.add_argument("--figures-dir", type=Path, default=Path("figures/issue_612"))
    parser.add_argument("--skip-figures", action="store_true")
    parser.add_argument(
        "--stage",
        choices=("endpoint", "dose-matched"),
        default="endpoint",
        help="endpoint = the parent P7 analyses (default, unchanged); dose-matched = "
        "the plan-v2 band-entry round (H1-dm + secondary read + descriptive reads).",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=None,
        help="dose-matched only: band_entry_selection.json override "
        "(default: <slab-root>/dose_matched/band_entry_selection.json).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [phase=p7_analysis] %(message)s", stream=sys.stdout
    )
    if args.stage == "dose-matched":
        return run_dose_matched(args)

    data = Data(args.slab_root, args.panel_set, args.claims)
    if data.missing_cells:
        log.warning("missing cells (reported, never invented): %s", data.missing_cells)

    analysis: dict = {
        "h1_onpolicy_vs_canned": paired_arm_contrast(data, "arm_onpolicy", "arm_canned"),
        "h1_onpolicy_vs_canned_per_source": per_source_contrast(data, "arm_onpolicy", "arm_canned"),
        "h2_prefix_vs_onpolicy": paired_arm_contrast(data, "arm_prefix", "arm_onpolicy"),
        "h2_prefix_vs_onpolicy_per_source": per_source_contrast(data, "arm_prefix", "arm_onpolicy"),
        "curve_reads_h4": curve_reads(data),
        "h3_anomaly": anomaly_reads(data),
        "manipulation_check": manipulation_check(data),
        "trajectory": trajectory_reads(args.slab_root, data),
        "parity_gate": parity_gate(args.slab_root),
        "frozen_subset": frozen_subset_rates(data),
        "missing_cells": data.missing_cells,
        "thresholds": {
            "leak_tau": LEAK_TAU,
            "flat_band": FLAT_BAND,
            "h1_support_min": H1_SUPPORT_MIN,
            "h1_null_band": H1_NULL_BAND,
            "bootstrap_B": BOOTSTRAP_B,
            "bootstrap_seed": BOOTSTRAP_SEED,
        },
        "metadata": {
            "git_commit_sha": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True
            ).strip(),
            "panel_set": str(args.panel_set),
            "claims": str(args.claims),
            "timestamp_utc": datetime.now(UTC).isoformat(),
        },
    }
    if not args.skip_figures:
        analysis["figures"] = make_figures(data, analysis, args.figures_dir)

    out_path = args.slab_root / "analysis_612.json"
    out_path.write_text(json.dumps(analysis, indent=2))
    log.info(
        "analysis -> %s | H1=%s H2=%s H3=%s parity=%s",
        out_path,
        analysis["h1_onpolicy_vs_canned"].get("verdict"),
        analysis["h2_prefix_vs_onpolicy"].get("verdict"),
        analysis["h3_anomaly"].get("headline"),
        analysis["parity_gate"]["verdict"],
    )
    if analysis["parity_gate"]["verdict"] == "HARD_FAIL":
        # Kill criterion K1: rig validity broken. The JSON above is written
        # FIRST so the evidence persists, but the process must not read as a
        # completed analysis run.
        log.error(
            "PARITY GATE HARD_FAIL (n_hard=%d, n_out_of_tol=%d) — rig-validity kill "
            "criterion K1; analysis_612.json written for evidence; exiting nonzero.",
            analysis["parity_gate"]["n_hard_fail"],
            analysis["parity_gate"]["n_out_of_tol"],
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
