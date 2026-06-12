# ruff: noqa: RUF002, RUF003  # em-dash + ε/Δ + × intentional
"""Task #600 §4.3 — deterministic design-time panel selection (CPU, VM, pre-training).

Selects, BEFORE any training: 6 target bystanders (2 per d_source tercile of
the #472 47-persona held-out panel), a NEAR negative per target (nearest
neighbour in centered L10 cosine space), a d_source-MATCHED far CONTROL per
target, and the 2-persona fixed base panel. Emits
``eval_results/issue_600/panel_selection.json`` — the design manifest the
headline test's condition labels are frozen on (committed to the issue branch
before training).

All steps are deterministic given the persona-bank content hash + the L10
centroid bundle. Distance = 1 − cos, with the canonical
``centering="global_mean"`` re-cosine (per ``persona-distance-metrics.md`` §
Bank centering, #536) recomputed from the raw #472 centroid bundle.

Recorded deterministic conventions (so the selection is reproducible without
reading this file): (a) the per-target P25/P75 quantiles are computed over the
held-out panel's distances-to-t EXCLUDING t itself; (b) the "bank median"
d_source for the base panel is the median over the bank EXCLUDING the villain
source (its self-distance 0 would skew the median); (c) numpy default linear
interpolation for quantiles; (d) ties break lexicographically everywhere.

CPU-only; no GPU, no network (reads local ``data/issue_472`` artifacts).
"""

from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import logging
import os
import subprocess
import sys
from glob import glob
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.contrastive_neg_geometry_472.persona_bank import (
    _content_hash,
    load_persona_bank,
)
from explore_persona_space.experiments.targeted_proximity_600 import (
    ALWAYS_INCLUDE_NEGATIVE,
    CONTRAST_FLOOR,
    COSINE_CENTERING,
    DESIGN_LAYER,
    EPS_MATCH,
    EPS_MATCH_RELAXED,
    FAR_QUANTILE,
    FAR_QUANTILE_RELAXED,
    N_TARGETS,
    N_TARGETS_PER_STRATUM,
    NEAR_QUANTILE,
    SOURCE_PERSONA,
)

log = logging.getLogger("issue_600.select_panels")

SCHEMA_VERSION = "i600_panel_selection_v1"
STRATA = ("near", "mid", "far")
EXPECTED_PANEL_N = 47
EXPECTED_Q_EVAL_N = 10
MAX_FIXED_POINT_ITERS = 10
N_MID_CANDIDATES = 5

# Relaxation ladder (ε_match, far_quantile) per plan §4.3 step 5.
RELAXATION_LADDER = ((EPS_MATCH, FAR_QUANTILE), (EPS_MATCH_RELAXED, FAR_QUANTILE_RELAXED))


def _i472_data_root() -> Path:
    return Path(os.environ.get("EPM_I472_DATA_ROOT", "data/issue_472"))


def _i472_eval_root() -> Path:
    return Path(os.environ.get("EPM_I472_EVAL_ROOT", "eval_results/issue_472"))


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL, text=True, env={**os.environ}
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_centered_distance_matrix(
    centroid_path: Path,
) -> tuple[dict[str, dict[str, float]], list[str]]:
    """Load the #472 L10 centroid bundle and return centered-cosine DISTANCES.

    The bundle stores raw centroids; the canonical centered cosine is
    recomputed at zero cost via ``compute_cosine_matrix(C,
    centering="global_mean")`` (#536). Returns ``(dist[a][b] = 1 − cos, names)``.
    """
    import torch

    from explore_persona_space.analysis.representation_shift import compute_cosine_matrix

    if not centroid_path.exists():
        raise FileNotFoundError(
            f"L10 centroid bundle missing at {centroid_path}. Pull from the issue-600-owned "
            "pinned snapshot issue600_targeted_proximity/inputs/centroids_L10.pt (NOT the "
            "stale issue472_neg_geometry/ mirror) or run the #472 bootstrap."
        )
    bundle = torch.load(centroid_path, weights_only=False)
    names: list[str] = list(bundle["persona_names"])
    centroids: torch.Tensor = bundle["centroids"]
    assert centroids.shape[0] == len(names), (centroids.shape, len(names))
    cos = compute_cosine_matrix(centroids.float(), centering=COSINE_CENTERING)
    assert cos.shape == (len(names), len(names)), cos.shape
    dist: dict[str, dict[str, float]] = {}
    for i, a in enumerate(names):
        dist[a] = {b: float(1.0 - cos[i, j].item()) for j, b in enumerate(names)}
    return dist, names


def reconstruct_held_out_panel(eval_root: Path) -> tuple[list[str], list[str]]:
    """Reconstruct the 47-persona #472 held-out panel + Q_eval from committed trajectories.

    Reads every ``c472_*/trajectory.json`` under ``eval_root`` (machine-readable,
    NOT plan prose) and asserts a SINGLE consistent panel + question set across
    all of them (sibling #479 artifacts in the same directory carry different
    panels and are excluded by the ``c472_`` glob).
    """
    files = sorted(glob(str(eval_root / "c472_*" / "trajectory.json")))
    if not files:
        raise FileNotFoundError(
            f"No c472_*/trajectory.json under {eval_root} — cannot reconstruct "
            "the #472 held-out panel. Check EPM_I472_EVAL_ROOT / the git checkout."
        )
    panels: set[tuple[str, ...]] = set()
    q_sets: set[tuple[str, ...]] = set()
    for f in files:
        payload = json.loads(Path(f).read_text())
        panels.add(tuple(payload["held_out_personas"]))
        q_sets.add(tuple(payload["eval_questions"]))
    if len(panels) != 1:
        raise AssertionError(
            f"#472 trajectories disagree on the held-out panel across {len(files)} files "
            f"({len(panels)} distinct panels) — refusing to reconstruct."
        )
    if len(q_sets) != 1:
        raise AssertionError(f"#472 trajectories disagree on Q_eval ({len(q_sets)} distinct sets).")
    panel = list(next(iter(panels)))
    q_eval = list(next(iter(q_sets)))
    if len(panel) != EXPECTED_PANEL_N:
        raise AssertionError(f"Reconstructed panel has {len(panel)} personas, expected 47.")
    if len(q_eval) != EXPECTED_Q_EVAL_N:
        raise AssertionError(f"Reconstructed Q_eval has {len(q_eval)} questions, expected 10.")
    return panel, q_eval


def _quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _nn_for_target(
    target: str,
    dist: dict[str, dict[str, float]],
    bank_names: list[str],
    excluded: set[str],
    p25: float,
) -> dict | None:
    """Plan §4.3 step 4: nearest non-excluded persona to ``target`` gated at P25."""
    cands = [p for p in bank_names if p not in excluded and p != target]
    cands.sort(key=lambda p: (dist[p][target], p))
    for p in cands:
        if dist[p][target] <= p25:
            return {"name": p, "d_to_target": dist[p][target]}
        break  # candidates are sorted ascending; the first already exceeds P25
    return None


def _ctrl_for_target(
    target: str,
    nn: dict,
    d_source: dict[str, float],
    dist: dict[str, dict[str, float]],
    bank_names: list[str],
    excluded: set[str],
    dist_to_t_panel: list[float],
) -> dict | None:
    """Plan §4.3 step 5: d_source-matched far control, with the relaxation ladder."""
    nn_name = nn["name"]
    for eps, far_q in RELAXATION_LADDER:
        far_floor = _quantile(dist_to_t_panel, far_q)
        cands = []
        for p in bank_names:
            if p in excluded or p in (target, nn_name):
                continue
            mismatch = abs(d_source[p] - d_source[nn_name])
            if mismatch > eps:
                continue
            if dist[p][target] < far_floor:
                continue
            if dist[p][target] - dist[nn_name][target] < CONTRAST_FLOOR:
                continue
            cands.append((mismatch, p))
        if cands:
            cands.sort(key=lambda mp: (mp[0], mp[1]))
            mismatch, name = cands[0]
            return {
                "name": name,
                "d_to_target": dist[name][target],
                "dsource_mismatch": mismatch,
                "eps_used": eps,
                "far_quantile_used": far_q,
                "far_floor_value": far_floor,
            }
    return None


def _feasibility(
    target: str,
    *,
    bank_names: list[str],
    panel: list[str],
    dist: dict[str, dict[str, float]],
    d_source: dict[str, float],
    excluded: set[str],
) -> dict | None:
    """Run steps 4-5 for one candidate target; None when infeasible."""
    dist_to_t_panel = [dist[p][target] for p in panel if p != target]
    p25 = _quantile(dist_to_t_panel, NEAR_QUANTILE)
    nn = _nn_for_target(target, dist, bank_names, excluded, p25)
    if nn is None:
        return None
    ctrl = _ctrl_for_target(target, nn, d_source, dist, bank_names, excluded, dist_to_t_panel)
    if ctrl is None:
        return None
    return {
        "target": target,
        "near": nn,
        "ctrl": ctrl,
        "realized_contrast": ctrl["d_to_target"] - nn["d_to_target"],
        "p25_near_gate": p25,
    }


def select_panels(  # noqa: C901  the §4.3 selection steps are one deterministic, auditable procedure
    *,
    bank: dict[str, str],
    dist: dict[str, dict[str, float]],
    bank_names: list[str],
    panel: list[str],
    q_eval: list[str],
    bank_hash: str,
    centroid_sha256: str,
) -> dict:
    """Run the full §4.3 selection; return the manifest dict (not yet written)."""
    source = SOURCE_PERSONA
    default = ALWAYS_INCLUDE_NEGATIVE
    if source not in bank_names or default not in bank_names:
        raise AssertionError(f"Bank/centroids missing {source!r} or {default!r} — wrong bundle?")
    missing_panel = [p for p in panel if p not in bank_names]
    if missing_panel:
        raise AssertionError(f"Held-out panel personas missing from centroids: {missing_panel}")

    d_source = {p: dist[p][source] for p in bank_names}

    # ── Step 3: d_source terciles of the 47-panel (ascending = near villain). ─
    panel_sorted = sorted(panel, key=lambda p: (d_source[p], p))
    # Plain str (not np.str_) so the manifest carries clean JSON strings.
    tercile_lists = [[str(x) for x in part] for part in np.array_split(np.asarray(panel_sorted), 3)]
    strata_map = dict(zip(STRATA, tercile_lists, strict=True))

    # ── Fixed-point over targets (steps 3-5; exclusions include the realized
    # target set, which is only known after selection — iterate to stability). ─
    base_excluded = {source, default}
    chosen: dict[str, list[dict]] = {}
    prev_target_set: set[str] = set()
    for iteration in range(MAX_FIXED_POINT_ITERS):
        chosen = {}
        for stratum in STRATA:
            feas: list[dict] = []
            for t in strata_map[stratum]:
                excl = base_excluded | (prev_target_set - {t})
                f = _feasibility(
                    t,
                    bank_names=bank_names,
                    panel=panel,
                    dist=dist,
                    d_source=d_source,
                    excluded=excl,
                )
                if f is not None:
                    f["stratum"] = stratum
                    feas.append(f)
            if len(feas) < N_TARGETS_PER_STRATUM:
                raise RuntimeError(
                    f"Stratum {stratum!r} has only {len(feas)} feasible targets "
                    f"(< {N_TARGETS_PER_STRATUM}) — the relaxation ladder is exhausted; "
                    "the bank is too sparse for the design (plan §8 risk row 4/5)."
                )
            feas.sort(key=lambda f: (-f["realized_contrast"], f["target"]))
            chosen[stratum] = feas[:N_TARGETS_PER_STRATUM]
        target_set = {f["target"] for fs in chosen.values() for f in fs}
        if target_set == prev_target_set:
            log.info("[select] target fixed-point converged after %d iteration(s)", iteration + 1)
            break
        prev_target_set = target_set
    else:
        raise RuntimeError(
            f"Target selection did not converge in {MAX_FIXED_POINT_ITERS} iterations."
        )

    targets = [f for stratum in STRATA for f in chosen[stratum]]
    assert len(targets) == N_TARGETS, len(targets)
    target_names = [f["target"] for f in targets]
    nn_names = [f["near"]["name"] for f in targets]
    ctrl_names = [f["ctrl"]["name"] for f in targets]

    # ── Step 6: fixed base panel — 2 personas with d_source closest to the
    # bank median, excluding everything already cast. ─────────────────────────
    bank_median = float(np.median([d_source[p] for p in bank_names if p != source]))
    base_pool = [
        p
        for p in bank_names
        if p not in {source, default}
        and p not in target_names
        and p not in nn_names
        and p not in ctrl_names
    ]
    base_pool.sort(key=lambda p: (abs(d_source[p] - bank_median), p))
    base_panel = base_pool[:2]
    if len(base_panel) != 2:
        raise RuntimeError("Bank too small to cast the 2-persona base panel.")

    # ── Ranked MID candidates per target (unused; for the dose follow-up). ───
    for f in targets:
        t = f["target"]
        nn = f["near"]
        ctrl = f["ctrl"]
        midpoint = 0.5 * (nn["d_to_target"] + ctrl["d_to_target"])
        eps = ctrl["eps_used"]
        mids = []
        for p in bank_names:
            if (
                p in {source, default, t}
                or p in target_names
                or p in base_panel
                or p in (nn["name"], ctrl["name"])
            ):
                continue
            if abs(d_source[p] - d_source[nn["name"]]) > eps:
                continue
            if not (nn["d_to_target"] < dist[p][t] < ctrl["d_to_target"]):
                continue
            mids.append((abs(dist[p][t] - midpoint), p))
        mids.sort(key=lambda mp: (mp[0], mp[1]))
        f["mid_candidates"] = [
            {"name": p, "d_to_target": dist[p][t], "d_source": d_source[p]}
            for _gap, p in mids[:N_MID_CANDIDATES]
        ]

    # ── Hard disjointness asserts (plan §4.4). ────────────────────────────────
    cast = {source, default} | set(base_panel) | set(nn_names) | set(ctrl_names)
    overlap = set(target_names) & cast
    if overlap:
        raise AssertionError(f"TARGETS overlap the cast set: {sorted(overlap)}")
    for f in targets:
        cell_panel = [default, *base_panel, f["near"]["name"]]
        for slot_name in (f["near"]["name"], f["ctrl"]["name"]):
            cell_panel = [default, *base_panel, slot_name]
            assert len(cell_panel) == 4, cell_panel
            assert len(set(cell_panel)) == 4, f"duplicate persona in panel {cell_panel}"
            assert source not in cell_panel, cell_panel
            assert not (set(cell_panel) & set(target_names)), (cell_panel, target_names)

    manifest = {
        "schema_version": SCHEMA_VERSION,
        "design_layer": DESIGN_LAYER,
        "centering": COSINE_CENTERING,
        "distance": "1 - centered_cosine",
        "source_persona": source,
        "always_include_negative": default,
        "bank_content_hash": bank_hash,
        "centroid_bundle_sha256": centroid_sha256,
        "n_bank": len(bank_names),
        "held_out_panel": sorted(panel),
        "n_held_out_panel": len(panel),
        "q_eval": q_eval,
        "bank_median_dsource": bank_median,
        "strata_assignment": {s: sorted(strata_map[s]) for s in STRATA},
        "d_source": {p: d_source[p] for p in sorted(bank_names)},
        "targets": [
            {
                "name": f["target"],
                "stratum": f["stratum"],
                "d_source": d_source[f["target"]],
                "near": {**f["near"], "d_source": d_source[f["near"]["name"]]},
                "ctrl": {**f["ctrl"], "d_source": d_source[f["ctrl"]["name"]]},
                "realized_contrast": f["realized_contrast"],
                "p25_near_gate": f["p25_near_gate"],
                "mid_candidates": f["mid_candidates"],
            }
            for f in targets
        ],
        "base_panel": [{"name": p, "d_source": d_source[p]} for p in base_panel],
        "selection_conventions": {
            "per_target_quantiles_exclude_self": True,
            "bank_median_excludes_source": True,
            "quantile_interpolation": "numpy default (linear)",
            "tie_break": "lexicographic",
            "relaxation_ladder": [list(rung) for rung in RELAXATION_LADDER],
            "contrast_floor": CONTRAST_FLOOR,
            "near_quantile": NEAR_QUANTILE,
        },
        "git_commit": _git_sha(),
        "generated_at": _dt.datetime.now(_dt.UTC).isoformat(),
    }
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI: run selection and write panel_selection.json (committed pre-training)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(message)s")
    ap = argparse.ArgumentParser(description="Task #600 design-time panel selection (CPU)")
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("eval_results/issue_600/panel_selection.json"),
        help="Output manifest path (committed to the issue branch before training).",
    )
    args = ap.parse_args(argv)

    bank_path = _i472_data_root() / "persona_bank.json"
    centroid_path = _i472_data_root() / f"centroids_L{DESIGN_LAYER}.pt"
    bank = load_persona_bank(bank_path)
    dist, bank_names = load_centered_distance_matrix(centroid_path)
    missing = [p for p in bank if p not in bank_names]
    if missing:
        raise AssertionError(f"Bank personas missing from the centroid bundle: {missing[:8]}")
    panel, q_eval = reconstruct_held_out_panel(_i472_eval_root())

    manifest = select_panels(
        bank=bank,
        dist=dist,
        bank_names=bank_names,
        panel=panel,
        q_eval=q_eval,
        bank_hash=_content_hash(bank),
        centroid_sha256=_file_sha256(centroid_path),
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
    log.info(
        "[select] wrote %s — targets=%s base_panel=%s",
        args.out,
        [t["name"] for t in manifest["targets"]],
        [b["name"] for b in manifest["base_panel"]],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
