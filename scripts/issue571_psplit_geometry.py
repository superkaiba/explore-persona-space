# ruff: noqa: RUF002, RUF003
# Intentional Unicode (Δ, −) in scientific docstrings + labels.
"""Task #571 persona-split-composition — Stage-2 Phase 0.5 geometry (pod, GPU).

Builds ONE fresh union persona bank — 35 eval personas (pinned panel) + the
A2 source prompt + the 53-candidate #472 bank pool ≈ 89 entities — and
extracts base-model L10/15/20 centroids via the #472 recipe
(``analysis/representation_shift.extract_centroids``: last-token hidden
state of {persona system prompt, question}, mean over the 20
EVAL_QUESTIONS). Persists BOTH cosine matrices per layer (raw + global-
mean-centered, #536 mandate) via ``contrastive_neg_geometry_472.centroids.
build_centroids``, then runs the §4.4 pre-training panel gate on L10
CENTERED distances over the 32 never-negative bystanders:

- G1 (variation):     median_b[d_nn(2-arm) − d_nn(8-arm)] ≥ 0.25 × sd_b(d_src)
- G2 (identity churn): ≥ 16/32 bystanders change nearest negative 2-arm → 8-arm
- G3 (decorrelation): |Pearson_b(Δd_nn(8−2), d_src)| ≤ 0.6

On G1/G2 failure: deterministic greedy re-selection from the candidate pool
(coverage-max, or variance-max when Stage 1 triggered the §4.1 linkage —
read from the committed ``stage1_geometry_join.json``, REQUIRED input). On
G3 failure: swap the near-source add for the next-best candidate restoring
G3. If no panel passes the registered sweep: proceed with
``reduced_identification`` (the split-count contrast is unaffected).

Outputs:
- ``data/issue_571/psplit/centroids_L{10,15,20}.pt`` (union-bank bundles)
- ``eval_results/issue_571/persona-split-composition/geometry/psplit_geometry.json``
- ``eval_results/issue_571/persona-split-composition/geometry/panel_personas.json``
  ({name: prompt} for the realized 8-arm's non-assistant personas — the
  R-gen input and the eval driver's ``--extra-personas-file``)

``--reselect-only [--exclude-candidates a,b]`` re-runs panel selection on
the SAVED bundles with no GPU (the R-gen truncation-breach swap path; also
importable as ``reselect_and_write``).

Usage (pod, GPU0):
    CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue571_psplit_geometry.py
"""

from __future__ import annotations

import argparse
import json
import logging
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPTS_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(SCRIPTS_DIR))

import numpy as np  # noqa: E402
from issue571_psplit_common import (  # noqa: E402
    GEOMETRY_JSON,
    HF_472_CENTROIDS_TMPL,
    HF_DATA_REPO,
    NAMED_PANEL_ORDER,
    PANEL_PERSONAS_JSON,
    PSPLIT_ARMS,
    PSPLIT_DATA_DIR,
    SOURCE_KEY,
    STAGE1_JSON,
    assert_panel_invariants,
    candidate_pool,
    load_persona_bank,
    panel_for_arm,
    rows_per_persona,
)

logger = logging.getLogger("issue571.psplit_geometry")

SCHEMA_VERSION = "issue571_psplit_geometry_v1"
LAYERS = (10, 15, 20)
GATE_LAYER = 10
G1_FACTOR = 0.25
G2_MIN_CHURN = 16
G3_MAX_ABS_PEARSON = 0.6
CROSSCHECK_MIN_RHO = 0.9


def _git_commit() -> str:
    """Short git commit of the repo this script runs from."""
    try:
        return (
            subprocess.run(  # epm-lint: subprocess-env-inherit -- git metadata probe, no creds
                ["git", "rev-parse", "--short", "HEAD"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                text=True,
                check=True,
            ).stdout.strip()
            or "unknown"
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def load_stage1_linkage() -> dict:
    """The committed Stage-1 linkage block (§4.0 ordering contract — REQUIRED)."""
    if not STAGE1_JSON.exists():
        raise FileNotFoundError(
            f"{STAGE1_JSON} missing — Stage 1 must run, commit, and land in this checkout "
            "BEFORE any Stage-2 phase (plan §4.0 hard ordering contract)."
        )
    return json.loads(STAGE1_JSON.read_text())["stage2_linkage"]


def build_union_bank() -> tuple[dict[str, str], list[str], list[str]]:
    """(union {name: prompt}, never-negative bystanders, candidate pool names)."""
    from issue560_crossrecipe_panel import (
        EXPECTED_PROMPT_MATCHES,
        HELD_OUT_35,
        load_persona_prompts,
    )

    from explore_persona_space.experiments.i406_conditions import CONDITIONS_BY_ID

    eval_prompts = load_persona_prompts()
    bank = load_persona_bank()
    pool = candidate_pool(bank, list(HELD_OUT_35))
    union: dict[str, str] = dict(eval_prompts)
    union[SOURCE_KEY] = CONDITIONS_BY_ID["A2"].system_prompt
    for name in pool:
        assert name not in union, name
        union[name] = bank[name]
    assert len(union) == 35 + 1 + 53, len(union)
    bystanders = [p for p in HELD_OUT_35 if p not in EXPECTED_PROMPT_MATCHES]
    assert len(bystanders) == 32, len(bystanders)
    return union, bystanders, pool


class DistanceView:
    """Name-indexed distance lookups over one (layer, centering) cosine matrix."""

    def __init__(self, cos: np.ndarray, names: list[str]):
        self.D = 1.0 - cos
        self.idx = {n: i for i, n in enumerate(names)}

    def d(self, a: str, b: str) -> float:
        return float(self.D[self.idx[a], self.idx[b]])

    def d_src(self, bystanders: list[str]) -> np.ndarray:
        return np.array([self.d(SOURCE_KEY, b) for b in bystanders])

    def d_nn(self, panel: list[str], bystanders: list[str]) -> tuple[np.ndarray, list[str]]:
        vals, ids = [], []
        for b in bystanders:
            dists = {m: self.d(m, b) for m in panel}
            nn = min(dists, key=dists.get)
            vals.append(dists[nn])
            ids.append(nn)
        return np.array(vals), ids


def compute_gates(view: DistanceView, panel_order: list[str], bystanders: list[str]) -> dict:
    """G1/G2/G3 on L10 centered distances (§4.4), plus the raw inputs."""
    d_src = view.d_src(bystanders)
    d2, id2 = view.d_nn(panel_for_arm(panel_order, "split2"), bystanders)
    d8, id8 = view.d_nn(panel_for_arm(panel_order, "split8"), bystanders)
    g1_value = float(np.median(d2 - d8))
    g1_threshold = G1_FACTOR * float(np.std(d_src))
    churn = int(sum(a != b for a, b in zip(id2, id8, strict=True)))
    delta = d8 - d2
    g3_value = (
        float(np.corrcoef(delta, d_src)[0, 1]) if float(np.std(delta)) > 1e-12 else float("nan")
    )
    return {
        "G1": {
            "value": g1_value,
            "threshold": g1_threshold,
            "pass": bool(g1_value >= g1_threshold),
        },
        "G2": {"value": churn, "threshold": G2_MIN_CHURN, "pass": bool(churn >= G2_MIN_CHURN)},
        "G3": {
            "value": None if not np.isfinite(g3_value) else g3_value,
            "threshold": G3_MAX_ABS_PEARSON,
            "pass": bool(np.isfinite(g3_value) and abs(g3_value) <= G3_MAX_ABS_PEARSON),
        },
        "all_pass": bool(
            g1_value >= g1_threshold
            and churn >= G2_MIN_CHURN
            and np.isfinite(g3_value)
            and abs(g3_value) <= G3_MAX_ABS_PEARSON
        ),
    }


def _objective_gain(
    view: DistanceView, current_nn: np.ndarray, cand: str, bystanders: list[str], objective: str
) -> float:
    """Greedy step score for adding ``cand`` (§4.4 fallback objectives)."""
    cand_d = np.array([view.d(cand, b) for b in bystanders])
    new_nn = np.minimum(current_nn, cand_d)
    if objective == "variance-max":
        return float(np.var(new_nn))
    return float(np.sum(current_nn - new_nn))  # coverage-max: summed d_nn reduction


def greedy_select(
    view: DistanceView,
    pool: list[str],
    bystanders: list[str],
    objective: str,
    exclude: set[str],
) -> list[str]:
    """§4.4 deterministic greedy re-selection → registered 8-persona order.

    Keep {assistant}; 2-arm partner = the candidate minimizing total bystander
    coverage gain subject to MIDDLE-TERCILE d_src; then greedily add 2 (then 4
    more) candidates maximizing the configured objective. Ties break on
    sorted-name order (the pool is pre-sorted).
    """
    cands = [c for c in pool if c not in exclude]
    assert len(cands) >= 7, (len(cands), "candidate pool exhausted")
    d_src_cand = {c: view.d(SOURCE_KEY, c) for c in cands}
    lo, hi = np.percentile(np.array(list(d_src_cand.values())), [100 / 3, 200 / 3])
    middle = [c for c in cands if lo <= d_src_cand[c] <= hi]
    assert middle, "middle-tercile d_src band empty"
    assistant_d = np.array([view.d("assistant", b) for b in bystanders])
    partner, best_gain = None, None
    for c in middle:
        gain = _objective_gain(view, assistant_d, c, bystanders, "coverage-max")
        if best_gain is None or gain < best_gain:
            partner, best_gain = c, gain
    panel = ["assistant", partner]
    current_nn = np.minimum(assistant_d, np.array([view.d(partner, b) for b in bystanders]))
    for _ in range(6):
        best_c, best_score = None, None
        for c in cands:
            if c in panel:
                continue
            score = _objective_gain(view, current_nn, c, bystanders, objective)
            if best_score is None or score > best_score:
                best_c, best_score = c, score
        panel.append(best_c)
        current_nn = np.minimum(current_nn, np.array([view.d(best_c, b) for b in bystanders]))
    assert len(panel) == 8 and len(set(panel)) == 8, panel
    return panel


def _g3_swap(
    view: DistanceView,
    panel: list[str],
    pool: list[str],
    bystanders: list[str],
    exclude: set[str],
    swap_out: str,
) -> tuple[list[str], dict] | None:
    """Swap ``swap_out`` for the candidate restoring G3 (best objective first)."""
    pos = panel.index(swap_out)
    d_nn_others = None
    others = [m for m in panel if m != swap_out]
    d_nn_others = np.min(
        np.stack([np.array([view.d(m, b) for b in bystanders]) for m in others]), axis=0
    )
    scored = []
    for c in pool:
        if c in panel or c in exclude:
            continue
        scored.append((_objective_gain(view, d_nn_others, c, bystanders, "coverage-max"), c))
    for _, c in sorted(scored, key=lambda t: (-t[0], t[1])):
        trial = list(panel)
        trial[pos] = c
        gates = compute_gates(view, trial, bystanders)
        if gates["G3"]["pass"]:
            return trial, gates
    return None


def select_panels(
    view: DistanceView,
    pool: list[str],
    bystanders: list[str],
    objective: str,
    exclude: set[str] | None = None,
) -> tuple[list[str], dict, dict]:
    """(registered 8-order, gates, selection provenance) per §4.4.

    Named panels are the default; the deterministic fallback fires only on a
    gate failure or when a named member is excluded (R-gen swap path).
    """
    exclude = exclude or set()
    provenance: dict = {"objective": objective, "excluded": sorted(exclude), "steps": []}
    panel = list(NAMED_PANEL_ORDER)
    if any(p in exclude for p in panel[1:]):
        # R-gen swap on a named member: next-best gate-passing candidate.
        for bad in [p for p in panel[1:] if p in exclude]:
            swapped = _g3_swap(view, panel, pool, bystanders, exclude, bad)
            if swapped is None:
                provenance["steps"].append(f"no gate-passing swap found for {bad}; greedy reselect")
                panel = greedy_select(view, pool, bystanders, objective, exclude)
                break
            panel, _ = swapped
            provenance["steps"].append(f"excluded {bad} -> swapped in {panel}")
    gates = compute_gates(view, panel, bystanders)
    if not (gates["G1"]["pass"] and gates["G2"]["pass"]):
        provenance["steps"].append("named panel failed G1/G2 -> greedy re-selection")
        panel = greedy_select(view, pool, bystanders, objective, exclude)
        gates = compute_gates(view, panel, bystanders)
    if gates["G1"]["pass"] and gates["G2"]["pass"] and not gates["G3"]["pass"]:
        swap_out = "data_scientist" if "data_scientist" in panel else panel[-1]
        swapped = _g3_swap(view, panel, pool, bystanders, exclude, swap_out)
        if swapped is not None:
            panel, gates = swapped
            provenance["steps"].append(f"G3 failed -> swapped {swap_out} out")
        else:
            provenance["steps"].append("G3 failed; no restoring swap found")
    provenance["reduced_identification"] = not gates["all_pass"]
    return panel, gates, provenance


def _load_bundle(layer: int) -> dict:
    import torch

    path = PSPLIT_DATA_DIR / f"centroids_L{layer}.pt"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing — run the GPU extraction phase first")
    return torch.load(path, map_location="cpu", weights_only=False)


def _views_from_bundle(bundle: dict) -> dict[str, DistanceView]:
    names = list(bundle["persona_names"])
    return {
        "centered": DistanceView(bundle["cos_matrix_mean_centered"].numpy(), names),
        "raw": DistanceView(bundle["cos_matrix"].numpy(), names),
    }


def crosscheck_vs_472(bundle: dict, union: dict[str, str]) -> dict:
    """Rank-correlate overlapping-pair RAW distances vs the #472 HF bundle (L10)."""
    import torch
    from huggingface_hub import hf_hub_download
    from scipy.stats import spearmanr

    ref_path = hf_hub_download(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        filename=HF_472_CENTROIDS_TMPL.format(layer=GATE_LAYER),
    )
    ref = torch.load(ref_path, map_location="cpu", weights_only=False)
    ref_names = list(ref["persona_names"])
    new_names = list(bundle["persona_names"])
    overlap = [n for n in new_names if n in ref_names]
    assert len(overlap) >= 40, (len(overlap), "unexpectedly small #472 overlap")
    new_D = 1.0 - bundle["cos_matrix"].numpy()
    ref_D = 1.0 - ref["cos_matrix"].numpy()
    ni = {n: i for i, n in enumerate(new_names)}
    ri = {n: i for i, n in enumerate(ref_names)}
    new_vals, ref_vals = [], []
    for i, a in enumerate(overlap):
        for b in overlap[i + 1 :]:
            new_vals.append(new_D[ni[a], ni[b]])
            ref_vals.append(ref_D[ri[a], ri[b]])
    rho = float(spearmanr(new_vals, ref_vals)[0])
    assert rho > CROSSCHECK_MIN_RHO, (
        f"#472 cross-check FAILED: overlapping-pair raw-distance Spearman {rho:.4f} <= "
        f"{CROSSCHECK_MIN_RHO} — centroid extraction drifted from the #472 recipe"
    )
    return {"n_overlap_personas": len(overlap), "spearman_raw_pairs": rho}


def write_outputs(
    union: dict[str, str],
    bystanders: list[str],
    pool: list[str],
    panel_order: list[str],
    gates: dict,
    provenance: dict,
    linkage: dict,
    crosscheck: dict | None,
) -> None:
    """psplit_geometry.json + panel_personas.json (checkpoint-per-phase)."""
    per_layer: dict = {}
    for layer in LAYERS:
        bundle = _load_bundle(layer)
        views = _views_from_bundle(bundle)
        per_layer[str(layer)] = {}
        for centering, view in views.items():
            d_src = view.d_src(bystanders)
            entry: dict = {"d_src": {b: float(v) for b, v in zip(bystanders, d_src, strict=True)}}
            for arm in PSPLIT_ARMS:
                vals, ids = view.d_nn(panel_for_arm(panel_order, arm), bystanders)
                entry[f"d_nn_{arm}"] = {b: float(v) for b, v in zip(bystanders, vals, strict=True)}
                entry[f"nn_identity_{arm}"] = dict(zip(bystanders, ids, strict=True))
            per_layer[str(layer)][centering] = entry

    GEOMETRY_JSON.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "phase": "stage2_phase0p5_geometry",
        "centering_primary": "global_mean",  # #536 mandate; raw co-reported
        "gate_layer": GATE_LAYER,
        "gate_centering": "centered",
        "realized_panel_order": panel_order,
        "realized_panels": {arm: panel_for_arm(panel_order, arm) for arm in PSPLIT_ARMS},
        "rows_per_persona": {arm: rows_per_persona(panel_order, arm) for arm in PSPLIT_ARMS},
        "gates": gates,
        "selection_provenance": provenance,
        "stage1_linkage_applied": linkage,
        "crosscheck_472": crosscheck,
        "bystanders_never_negative": bystanders,
        "candidate_pool": pool,
        "distances": per_layer,
        "metadata": {
            "task": 571,
            "followup_label": "persona-split-composition",
            "script": "issue571_psplit_geometry.py",
            "git_commit": _git_commit(),
            "timestamp_utc": datetime.now(UTC).isoformat(),
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "argv": sys.argv[1:],
        },
    }
    GEOMETRY_JSON.write_text(json.dumps(payload, indent=1))
    logger.info("geometry JSON written: %s", GEOMETRY_JSON)

    panel_prompts = {name: union[name] for name in panel_order if name != "assistant"}
    assert len(panel_prompts) == 7, sorted(panel_prompts)
    PANEL_PERSONAS_JSON.write_text(json.dumps(panel_prompts, indent=1, ensure_ascii=False))
    logger.info("panel personas written: %s", PANEL_PERSONAS_JSON)


def reselect_and_write(exclude: set[str], swap_reason: str) -> list[str]:
    """CPU-only re-selection from saved bundles (the R-gen swap path).

    Re-runs §4.4 selection excluding ``exclude``, rewrites the two geometry
    outputs with swap provenance, and returns the new registered order.
    """
    linkage = load_stage1_linkage()
    union, bystanders, pool = build_union_bank()
    bundle = _load_bundle(GATE_LAYER)
    view = _views_from_bundle(bundle)["centered"]
    panel_order, gates, provenance = select_panels(
        view, pool, bystanders, linkage["fallback_objective"], exclude=exclude
    )
    assert_panel_invariants(panel_order, union)
    provenance["swap_reason"] = swap_reason
    crosscheck = None
    if GEOMETRY_JSON.exists():
        crosscheck = json.loads(GEOMETRY_JSON.read_text()).get("crosscheck_472")
    write_outputs(union, bystanders, pool, panel_order, gates, provenance, linkage, crosscheck)
    return panel_order


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Task #571 psplit Stage-2 Phase 0.5: union-bank geometry + panel gate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "--reselect-only",
        action="store_true",
        help="CPU re-selection from saved centroid bundles (no GPU extraction)",
    )
    ap.add_argument(
        "--exclude-candidates",
        default="",
        help="comma list of bank personas to exclude from panels (R-gen swap path)",
    )
    args = ap.parse_args(argv)
    print("[phase=p0p5_geometry]", flush=True)

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    exclude = {c for c in args.exclude_candidates.split(",") if c}
    linkage = load_stage1_linkage()
    logger.info("Stage-1 linkage: %s", linkage)

    if args.reselect_only:
        reselect_and_write(exclude, swap_reason="--reselect-only invocation")
        return 0

    union, bystanders, pool = build_union_bank()
    from explore_persona_space.experiments.contrastive_neg_geometry_472.centroids import (
        build_centroids,
    )

    PSPLIT_DATA_DIR.mkdir(parents=True, exist_ok=True)
    build_centroids(union, layers=LAYERS, out_dir=PSPLIT_DATA_DIR)

    bundle = _load_bundle(GATE_LAYER)
    assert list(bundle["persona_names"]) == list(union), "bundle/persona-order drift"
    crosscheck = crosscheck_vs_472(bundle, union)
    logger.info("#472 cross-check PASS: %s", crosscheck)

    view = _views_from_bundle(bundle)["centered"]
    panel_order, gates, provenance = select_panels(
        view, pool, bystanders, linkage["fallback_objective"], exclude=exclude
    )
    assert_panel_invariants(panel_order, union)
    logger.info(
        "realized panel order: %s | gates: G1=%s G2=%s G3=%s (all_pass=%s)",
        panel_order,
        gates["G1"]["pass"],
        gates["G2"]["pass"],
        gates["G3"]["pass"],
        gates["all_pass"],
    )
    write_outputs(union, bystanders, pool, panel_order, gates, provenance, linkage, crosscheck)
    return 0


if __name__ == "__main__":
    sys.exit(main())
