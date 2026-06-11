#!/usr/bin/env python3
"""#602 P8 estimator bake-off — Phase 0 reproduction gate + Phase 2 scoring.

Phases (plan #602 §4.4):

- ``--phase repro-gate`` (Phase 0, VM, CPU; BLOCKS pod provisioning):
  download the 18 cached #551 shift payloads from the private data repo at
  the pinned revision, reproduce #521's stored per-cell numbers
  (``s_top1_frac`` + per-persona ``cos_to_U1`` at L14) within 0.05 vs
  ``eval_results/issue_521/svd/summary.json``, report the cached
  base-vs-same divergence numbers, verify every adapter prefix resolves on
  the Hub, and pin every input (incl. the reconstructed #541 positives'
  sha256 + row-count gate) into ``eval_results/issue_602/inputs_manifest.json``.

- ``--phase score`` (Phase 2, VM, CPU, post-pod; the default): implement
  the plan §4.4 scoring pseudocode over the freshly extracted shift
  payloads (``eval_results/issue_602/shifts/*.pt``) + estimator payloads
  (``eval_results/issue_602/estimator_reads/*.pt``): dual-target cosines
  (w_shared AND w_src), LOCO w_shared geometry read, norm-only predictor,
  behavioral Spearman (rho_behav_est / rho_behav_real / rho_behav_norm,
  panel families only), sibling-included AND -excluded off-diagonal
  margins, 10k random null, cross-seed ceiling, INDETERMINATE band,
  31-run-cell denominator with the below-null ladder exclusion, per-panel
  N reporting, anchor_521 band check, exploratory grid + select-on-42 /
  confirm-on-137-256 discipline.

Outputs: ``eval_results/issue_602/phase0/*`` (gate) and
``eval_results/issue_602/{agreement,repair,grids}/*`` (scoring), all JSON
with reproducibility metadata; figures are rendered separately by
``scripts/issue602_figures.py``.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

from explore_persona_space.analysis import i602_bakeoff as bk  # noqa: E402
from explore_persona_space.analysis.svd_direction_constancy import (  # noqa: E402
    assemble_M,
    cosine,
    spearman_rho,
    svd_summary,
)

logger = logging.getLogger("issue602_score")

PHASE0_TOLERANCE = 0.05  # the #551 repro tolerance (plan §4.4 Phase 0)
VALIDITY_COS = 0.3
REPAIR_RHO_FAIL = 0.3
REPAIR_RHO_PASS = 0.5
MARGIN_MIN = 0.2
N_RANDOM_NULL = 10_000
# Same-construct sibling family pairs (plan §4.4: marker-519 <-> marker-474,
# EM-turner <-> EM-518); the sibling-EXCLUDED margin is the binding one.
SIBLINGS: dict[str, str] = {
    "marker519": "loc474",
    "loc474": "marker519",
    "em_turner": "em518",
    "em518": "em_turner",
}

CACHED_CELLS = [
    (variant, arm, seed)
    for variant in ("same", "base", "on_policy")
    for arm in ("marker", "em")
    for seed in bk.SEEDS_3
]


def _meta() -> dict[str, Any]:
    return {
        "issue": bk.ISSUE,
        "git_commit": bk.git_sha(REPO),
        "env_versions": bk.env_versions(),
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, default=float)
    logger.info("wrote %s", path)


# ---------------------------------------------------------------------------
# Phase 0 — reproduction gate
# ---------------------------------------------------------------------------
def download_cached_shifts(dest: Path) -> dict[str, str]:
    """Download the 18 cached payloads + manifests (pinned revision)."""
    shas: dict[str, str] = {}
    dest.mkdir(parents=True, exist_ok=True)
    for variant, arm, seed in CACHED_CELLS:
        for ext in (".pt", ".manifest.json"):
            fname = f"{variant}_{arm}_seed{seed}{ext}"
            local = dest / fname
            if not local.exists():
                src = bk.hub_download(
                    bk.PRIVATE_DATA_REPO,
                    f"{bk.CACHED_SHIFTS_PREFIX}/{fname}",
                    revision=bk.CACHED_SHIFTS_REVISION,
                )
                local.symlink_to(src)
            shas[fname] = bk.sha256_file(local)
    return shas


def load_cached_payload(dest: Path, variant: str, arm: str, seed: int) -> dict:
    """Load one cached .pt payload {'shifts': ..., 'manifest': ...}."""
    p = dest / f"{variant}_{arm}_seed{seed}.pt"
    payload = torch.load(p, map_location="cpu", weights_only=False)
    assert "shifts" in payload and "manifest" in payload, sorted(payload.keys())
    return payload


def phase0_repro_gate(args: argparse.Namespace) -> int:
    """Phase 0: reproduce #521's stored numbers from the raw cached tensors."""
    out_dir = bk.eval_dir(REPO) / "phase0"
    cached_dir = bk.eval_dir(REPO) / "cached_shifts"
    logger.info(
        "[phase=p0_download] pulling 18 cached payloads (pinned %s)", bk.CACHED_SHIFTS_REVISION[:12]
    )
    shas = download_cached_shifts(cached_dir)

    stored = json.loads((REPO / "eval_results" / "issue_521" / "svd" / "summary.json").read_text())[
        "per_cell"
    ]

    logger.info("[phase=p0_reproduce] recomputing SVD summaries for 18 cells")
    results: dict[str, Any] = {}
    n_fail = 0
    for variant, arm, seed in CACHED_CELLS:
        key = f"{variant}_{arm}_seed{seed}"
        payload = load_cached_payload(cached_dir, variant, arm, seed)
        ref = stored[key]
        M, order = assemble_M(payload["shifts"], persona_order=ref["persona_order"])
        summ = svd_summary(M)
        diffs = {
            "s_top1_frac": abs(summ["s_top1_frac"] - ref["s_top1_frac"]),
            "mean_cos_to_U1": abs(float(summ["cos_to_U1"].mean()) - ref["mean_cos_to_U1"]),
            "max_per_persona_cos_to_U1": float(
                np.max(np.abs(summ["cos_to_U1"] - np.asarray(ref["cos_to_U1"], dtype=np.float32)))
            ),
        }
        ok = all(v <= PHASE0_TOLERANCE for v in diffs.values())
        n_fail += 0 if ok else 1
        results[key] = {
            "ok": ok,
            "diffs": diffs,
            "reproduced_s_top1_frac": summ["s_top1_frac"],
            "stored_s_top1_frac": ref["s_top1_frac"],
            "M_shape": list(summ["M_shape"]),
        }
        logger.info("  %s: %s (max diff %.4f)", key, "PASS" if ok else "FAIL", max(diffs.values()))

    # base-vs-same divergence on cached cells (plan §4.3/§4.4 step 4 — the
    # numbers are REPORTED, not just thresholded).
    logger.info("[phase=p0_divergence] base-vs-same divergence on cached cells")
    divergence: dict[str, Any] = {}
    for arm in ("marker", "em"):
        for seed in bk.SEEDS_3:
            base_p = load_cached_payload(cached_dir, "base", arm, seed)
            same_p = load_cached_payload(cached_dir, "same", arm, seed)
            order = sorted(base_p["shifts"].keys())
            Mb, _ = assemble_M(base_p["shifts"], persona_order=order)
            Ms, _ = assemble_M(same_p["shifts"], persona_order=order)
            u1b = svd_summary(Mb)["U1"]
            u1s = svd_summary(Ms)["U1"]
            per_ctx = [cosine(Mb[:, i], Ms[:, i]) for i in range(Mb.shape[1])]
            divergence[f"{arm}_seed{seed}"] = {
                "cos_U1_base_vs_same": cosine(u1b, u1s),
                "per_context_cos_mean": float(np.mean(per_ctx)),
                "per_context_cos_median": float(np.median(per_ctx)),
                "per_context_cos_min": float(np.min(per_ctx)),
                "n_contexts": len(per_ctx),
            }

    # Hub fitness: every adapter prefix resolves (incl. the #541 allowlist —
    # asserted to be the EXPLICIT 9 paths, never globbed).
    logger.info("[phase=p0_adapters] verifying all 31 adapter prefixes on the Hub")
    from huggingface_hub import list_repo_files

    cells = bk.extraction_cells()
    files_by_repo: dict[str, list[str]] = {}
    adapter_checks: dict[str, bool] = {}
    for cell in cells:
        repo = cell["adapter_repo"]
        if repo not in files_by_repo:
            files_by_repo[repo] = list_repo_files(repo)
        ok = any(f == f"{cell['adapter_prefix']}/adapter_config.json" for f in files_by_repo[repo])
        adapter_checks[cell["cell_id"]] = ok
        if not ok:
            logger.error("  adapter MISSING: %s :: %s", repo, cell["adapter_prefix"])
    allow = bk.adapter_allowlist_541()
    assert len(allow) == 9 and all("exp541-" in a and "exp541smoke" not in a for a in allow), allow

    # Inputs manifest: pin EVERYTHING (incl. reconstructed #541 positives).
    logger.info("[phase=p0_pin_inputs] pinning inputs into inputs_manifest.json")
    recon: dict[str, Any] = {}
    for seed in bk.SEEDS_3:
        _rows, prov = bk.reconstruct_541_positives(seed, root=REPO)
        recon[f"seed{seed}"] = prov
    mix_provenance: dict[str, Any] = {}
    if not args.skip_mix_pins:
        for unit in bk.estimator_units():
            for mix_label in unit["e1_mix_labels"]:
                key = f"{unit['family']}__{unit['source']}__{mix_label}"
                _rows, prov = bk.e1_rows(unit["family"], unit["source"], mix_label, root=REPO)
                mix_provenance[key] = prov

    panel_files = {
        "issue_521_personas": str(REPO / "eval_results/issue_521/inputs/personas.json"),
        "issue_521_questions": str(REPO / "eval_results/issue_521/inputs/questions.json"),
        "issue_521_marker_pool": str(REPO / "eval_results/issue_521/inputs/marker_pool.json"),
        "issue_541_predictors": str(REPO / "eval_results/issue_541/predictors.json"),
        "issue_518_refusal_panel": str(
            REPO / "eval_results/issue_518/refusal/_inputs/predictor_comparison.json"
        ),
        "issue_518_em_panel": str(
            REPO / "eval_results/issue_518/em/_inputs/predictor_comparison.json"
        ),
        "issue_474_G": str(
            REPO / "eval_results/issue_474/cross_eval/loc_ep1/G_logprob_matrix.json"
        ),
        "issue_444_teach_pool": str(
            REPO / "eval_results/issue_444/bystander_logprob/teach_rows.json"
        ),
        "issue_444_fact_pick": str(
            REPO / "eval_results/issue_444/phase0_fact_candidates/fact_pick.json"
        ),
    }
    manifest = {
        "cached_shifts": {
            "repo": bk.PRIVATE_DATA_REPO,
            "prefix": bk.CACHED_SHIFTS_PREFIX,
            "revision": bk.CACHED_SHIFTS_REVISION,
            "file_sha256": shas,
        },
        "adapters": {
            c["cell_id"]: {
                "repo": c["adapter_repo"],
                "prefix": c["adapter_prefix"],
                "resolves": adapter_checks[c["cell_id"]],
            }
            for c in cells
        },
        "adapter_allowlist_541": allow,
        "reconstructed_541_positives": recon,
        "e1_mix_provenance": mix_provenance,
        "panel_files": {
            k: {"path": v, "sha256": bk.sha256_file(Path(v))} for k, v in panel_files.items()
        },
        "anchor_521": {
            "manifest": bk.load_marker_steering_manifest(REPO),
            "recorded_cos_U1_vsteer": bk.anchor_521_recorded(REPO),
            "band": bk.ANCHOR_521_BAND,
        },
        "e3_descriptions": bk.E3_DESCRIPTIONS,
        "reproducibility": _meta(),
    }
    _write_json(bk.eval_dir(REPO) / "inputs_manifest.json", manifest)
    _write_json(
        out_dir / "repro_gate.json",
        {
            "tolerance": PHASE0_TOLERANCE,
            "n_cells": len(results),
            "n_fail": n_fail,
            "gate": "PASS" if (n_fail == 0 and all(adapter_checks.values())) else "FAIL",
            "per_cell": results,
            "adapter_checks": adapter_checks,
            "reproducibility": _meta(),
        },
    )
    _write_json(out_dir / "base_vs_same.json", {"cells": divergence, "reproducibility": _meta()})

    if n_fail or not all(adapter_checks.values()):
        logger.error(
            "[phase=p0_gate] FAIL (%d repro fails; adapters ok=%s)",
            n_fail,
            all(adapter_checks.values()),
        )
        return 1
    logger.info("[phase=p0_gate] PASS — pod provisioning unblocked")
    return 0


# ---------------------------------------------------------------------------
# Phase 2 — scoring
# ---------------------------------------------------------------------------
def _entry_vec(entry: dict, layer: int, pos: str) -> np.ndarray | None:
    """Pull one (layer, pos) vector from a shift-payload persona entry."""
    if pos == "mean_resp":
        key = "delta_v_mean_resp" if layer == bk.PRIMARY_LAYER else f"delta_v_mean_resp_l{layer}"
    elif pos == "slot":
        key = "delta_v" if layer == bk.PRIMARY_LAYER else f"delta_v_l{layer}"
    else:
        raise ValueError(pos)
    v = entry.get(key)
    return None if v is None else v.detach().float().numpy()


def _load_new_shift_payloads(shifts_dir: Path) -> dict[str, dict]:
    """Load every freshly extracted cell payload, keyed by cell_id."""
    out: dict[str, dict] = {}
    for cell in bk.extraction_cells():
        p = shifts_dir / f"{cell['cell_id']}.pt"
        if not p.exists():
            logger.warning("missing shift payload for %s (%s)", cell["cell_id"], p)
            continue
        payload = torch.load(p, map_location="cpu", weights_only=False)
        out[cell["cell_id"]] = {**cell, "payload": payload}
    return out


def _load_estimator_payloads(est_dir: Path) -> dict[tuple[str, str], dict]:
    """Load estimator-read payloads keyed by (family, source)."""
    out: dict[tuple[str, str], dict] = {}
    for unit in bk.estimator_units():
        p = est_dir / f"{unit['family']}__{unit['source']}.pt"
        if not p.exists():
            logger.warning("missing estimator payload for %s/%s", unit["family"], unit["source"])
            continue
        out[(unit["family"], unit["source"])] = torch.load(
            p, map_location="cpu", weights_only=False
        )
    return out


def _w_hat(
    est_payload: dict, estimator: str, layer: int, pos: str, cell: dict, k: int = bk.E2_K_PRIMARY
) -> np.ndarray | None:
    """Resolve one estimator vector at (layer, pos) for a run-cell.

    For ``est_tf`` on per-seed-mix families the seed-matched E1 unit is
    used; the marker families' HEADLINE E1 read is the exclude-marker
    mean-over-completion (token-identity discriminator, plan H3) — the
    include-marker read is reported in the exploratory grid.
    """
    family = cell["family"]
    if estimator == "est_tf":
        mix_label = f"seed{cell['seed']}" if family in ("marker519", "fact541") else "shared"
        unit = est_payload["e1"].get(mix_label)
        if unit is None:
            return None
        pos_key = pos
        if pos == "mean_resp" and family in ("marker519", "loc474"):
            pos_key = "mean_resp_excl_marker"
        w = unit["w_hat"].get(pos_key, unit["w_hat"].get(pos, {}))
        v = w.get(layer) if isinstance(w, dict) else None
        return None if v is None else np.asarray(v, dtype=np.float64)
    if estimator == "est_icl":
        unit = est_payload["e2"].get(f"K{k}")
        if unit is None:
            return None
        v = unit["w_hat"].get(pos, {}).get(layer)
        return None if v is None else np.asarray(v, dtype=np.float64)
    if estimator == "est_desc":
        v = est_payload["e3"]["w_hat"].get(pos, {}).get(layer)
        return None if v is None else np.asarray(v, dtype=np.float64)
    raise ValueError(estimator)


def _fisher_se(n: int) -> float:
    """Approximate SE of Spearman rho via the Fisher-z normal approximation."""
    return 1.0 / math.sqrt(max(n - 3, 1))


def _edge_indeterminate(rho: float, threshold: float, n: int) -> bool:
    """True when rho is within ~1 SE (Fisher z) of the registered band edge."""
    if n <= 3:
        return True
    z = 0.5 * math.log((1 + min(max(rho, -0.999), 0.999)) / (1 - min(max(rho, -0.999), 0.999)))
    zt = 0.5 * math.log((1 + threshold) / (1 - threshold))
    return abs(z - zt) < _fisher_se(n)


def _compute_targets(cells: dict[str, dict], layer: int, pos: str) -> dict[str, dict]:
    """Per-cell realized targets (w_src + w_shared + M) at every (layer, pos)."""
    targets: dict[str, dict] = {}
    for cell_id, cell in cells.items():
        shifts = cell["payload"]["shifts"]
        order = sorted(shifts.keys())
        per_construction: dict[str, Any] = {}
        for ly in bk.LAYERS:
            for ps in ("mean_resp", "slot"):
                cols = {c: _entry_vec(shifts[c], ly, ps) for c in order}
                cols = {c: v for c, v in cols.items() if v is not None}
                if len(cols) < 2:
                    continue
                names = sorted(cols.keys())
                M = np.stack([cols[c] for c in names], axis=1)
                summ = svd_summary(M)
                src_ctx = cell["source"]
                key = f"L{ly}_{ps}"
                per_construction[key] = {
                    "persona_order": names,
                    "M": M,
                    "w_shared": summ["U1"].astype(np.float64),
                    "w_src": cols.get(src_ctx),
                    "s_top1_frac": summ["s_top1_frac"],
                }
        if f"L{layer}_{pos}" not in per_construction:
            raise RuntimeError(
                f"cell {cell_id}: pre-registered construction L{layer}/{pos} missing from "
                "payload (M3a extension not applied?)"
            )
        if per_construction[f"L{layer}_{pos}"]["w_src"] is None:
            raise RuntimeError(f"cell {cell_id}: source context {cell['source']!r} missing")
        targets[cell_id] = per_construction
    return targets


def _seed_ceiling(
    cells: dict[str, dict], targets: dict[str, dict], layer: int, pos: str
) -> dict[str, Any]:
    """Cross-seed U1 cosine ceiling per 3-seed family (`ceiling_seed`)."""
    ceiling: dict[str, Any] = {}
    for family in ("marker519", "em_turner", "fact541"):
        groups: dict[str, list[tuple[int, np.ndarray]]] = {}
        for cell_id, cell in cells.items():
            if cell["family"] != family:
                continue
            grp = cell["source"]
            groups.setdefault(grp, []).append(
                (cell["seed"], targets[cell_id][f"L{layer}_{pos}"]["w_shared"])
            )
        for grp, seed_vecs in groups.items():
            if len(seed_vecs) < 2:
                continue
            pairs = []
            for i in range(len(seed_vecs)):
                for j in range(i + 1, len(seed_vecs)):
                    pairs.append(
                        {
                            "seed_a": seed_vecs[i][0],
                            "seed_b": seed_vecs[j][0],
                            "cos": cosine(seed_vecs[i][1], seed_vecs[j][1]),
                        }
                    )
            ceiling[f"{family}__{grp}"] = {
                "pairs": pairs,
                "median_abs_cos": float(np.median([abs(p["cos"]) for p in pairs])),
            }
    return ceiling


def _headline_rows(
    cells: dict[str, dict],
    est: dict[tuple[str, str], dict],
    targets: dict[str, dict],
    layer: int,
    pos: str,
) -> list[dict[str, Any]]:
    """Per-(run-cell x estimator) dual-target cosines at the registered construction."""
    rows: list[dict[str, Any]] = []
    for cell_id, cell in sorted(cells.items()):
        ep = est.get((cell["family"], cell["source"]))
        tgt = targets[cell_id][f"L{layer}_{pos}"]
        for estimator in ("est_tf", "est_icl", "est_desc"):
            w = None if ep is None else _w_hat(ep, estimator, layer, pos, cell)
            row = {
                "cell_id": cell_id,
                "family": cell["family"],
                "source": cell["source"],
                "seed": cell["seed"],
                "estimator": estimator,
                "cos_w_shared": None if w is None else cosine(w, tgt["w_shared"]),
                "cos_w_src": None if w is None else cosine(w, tgt["w_src"]),
                "missing_estimator": w is None,
            }
            row["best_target_cos"] = (
                None if w is None else max(row["cos_w_shared"], row["cos_w_src"], key=abs)
            )
            rows.append(row)
    return rows


def _offdiag_margins(
    cells: dict[str, dict],
    est: dict[tuple[str, str], dict],
    targets: dict[str, dict],
    layer: int,
    pos: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Cross-behavior off-diagonal margins (incl/excl same-construct siblings)."""
    fam_shared = {cid: targets[cid][f"L{layer}_{pos}"]["w_shared"] for cid in targets}
    margins: dict[str, Any] = {}
    cross_recipe: list[dict[str, Any]] = []
    for (family, source), ep in est.items():
        # representative cell for the estimator unit (seed 42 cell)
        cell_ids = [
            cid for cid, c in cells.items() if c["family"] == family and c["source"] == source
        ]
        if not cell_ids:
            continue
        for estimator in ("est_tf", "est_icl", "est_desc"):
            cell0 = cells[cell_ids[0]]
            w = _w_hat(ep, estimator, layer, pos, cell0)
            if w is None:
                continue
            on_diag = [cosine(w, fam_shared[cid]) for cid in cell_ids]
            off_all, off_nosib = [], []
            for cid, c in cells.items():
                if c["family"] == family:
                    continue
                v = cosine(w, fam_shared[cid])
                off_all.append(v)
                if c["family"] != SIBLINGS.get(family):
                    off_nosib.append(v)
                else:
                    cross_recipe.append(
                        {
                            "estimator_unit": f"{family}__{source}__{estimator}",
                            "sibling_cell": cid,
                            "cos": v,
                        }
                    )
            margins[f"{family}__{source}__{estimator}"] = {
                "on_diag_median": float(np.median(on_diag)),
                "off_diag_mean_incl_siblings": float(np.mean(off_all)),
                "off_diag_mean_excl_siblings": float(np.mean(off_nosib)),
                "margin_incl_siblings": float(np.median(on_diag) - np.mean(off_all)),
                "margin_excl_siblings": float(np.median(on_diag) - np.mean(off_nosib)),
                "n_on": len(on_diag),
                "n_off_excl": len(off_nosib),
            }
    return margins, cross_recipe


def _verdict_table(
    rows: list[dict[str, Any]], margins: dict[str, Any], null95: float
) -> dict[str, Any]:
    """3 estimators x 6 families verdict table (MF2 dual-target / MF3 median)."""
    verdicts: dict[str, Any] = {}
    for family in bk.FAMILIES:
        for estimator in ("est_tf", "est_icl", "est_desc"):
            per_cell = [
                r
                for r in rows
                if r["family"] == family
                and r["estimator"] == estimator
                and not r["missing_estimator"]
            ]
            if not per_cell:
                verdicts[f"{family}__{estimator}"] = {"verdict": "MISSING"}
                continue
            med_shared = float(np.median([r["cos_w_shared"] for r in per_cell]))
            med_src = float(np.median([r["cos_w_src"] for r in per_cell]))
            # per-seed pass counts on the registered disjunction
            seeds = sorted({r["seed"] for r in per_cell})
            n_pass_shared = sum(
                1
                for s in seeds
                if np.median([r["cos_w_shared"] for r in per_cell if r["seed"] == s])
                >= VALIDITY_COS
            )
            n_pass_src = sum(
                1
                for s in seeds
                if np.median([r["cos_w_src"] for r in per_cell if r["seed"] == s]) >= VALIDITY_COS
            )
            unit_margins = [
                m
                for k, m in margins.items()
                if k.startswith(f"{family}__") and k.endswith(f"__{estimator}")
            ]
            margin_excl = (
                float(np.median([m["margin_excl_siblings"] for m in unit_margins]))
                if unit_margins
                else None
            )
            best = max(med_shared, med_src)
            seed_frac_ok = (
                (n_pass_shared >= 2 or n_pass_src >= 2)
                if len(seeds) >= 3
                else (med_shared >= VALIDITY_COS or med_src >= VALIDITY_COS)
            )
            valid = (
                best >= VALIDITY_COS
                and seed_frac_ok
                and best > null95
                and (margin_excl is not None and margin_excl >= MARGIN_MIN)
            )
            if valid:
                verdict = "VALID"
            elif null95 < best < VALIDITY_COS:
                verdict = "INDETERMINATE"  # above null, below validity — named class
            else:
                verdict = "INVALID"
            verdicts[f"{family}__{estimator}"] = {
                "verdict": verdict,
                "which_target_cleared": ("w_shared" if med_shared >= med_src else "w_src")
                if valid
                else None,
                "median_cos_w_shared": med_shared,
                "median_cos_w_src": med_src,
                "n_seeds": len(seeds),
                "single_run_flag": len(seeds) < 3,
                "n_pass_shared": n_pass_shared,
                "n_pass_src": n_pass_src,
                "margin_excl_siblings": margin_excl,
                "n_cells": len(per_cell),
            }
    return verdicts


def _h1_ladder(cells: dict[str, dict], rows: list[dict[str, Any]], null95: float) -> dict[str, Any]:
    """H1 fidelity-ladder per-cell calls (below-null cells excluded as
    'no-ordering-signal'; registered denominator 31)."""
    ladder: dict[str, Any] = {"per_cell": {}, "n_eligible": 0, "n_hold": 0, "n_e3_ge_e1": 0}
    for cell_id in sorted(cells.keys()):
        cs = {
            r["estimator"]: r
            for r in rows
            if r["cell_id"] == cell_id and not r["missing_estimator"]
        }
        if len(cs) < 3:
            ladder["per_cell"][cell_id] = "missing-estimator"
            continue
        vals = {e: cs[e]["cos_w_shared"] for e in ("est_tf", "est_icl", "est_desc")}
        eligible = any(v > null95 for v in vals.values())
        if not eligible:
            ladder["per_cell"][cell_id] = "no-ordering-signal"  # all inside null — EXCLUDED
            continue
        ladder["n_eligible"] += 1
        holds = vals["est_tf"] > vals["est_icl"] > vals["est_desc"]
        ladder["n_hold"] += int(holds)
        ladder["n_e3_ge_e1"] += int(vals["est_desc"] >= vals["est_tf"])
        ladder["per_cell"][cell_id] = "holds" if holds else "violated"
    ladder["holds_frac_eligible"] = (
        ladder["n_hold"] / ladder["n_eligible"] if ladder["n_eligible"] else None
    )
    ladder["registered_denominator"] = 31
    return ladder


def _repair_and_geometry(
    cells: dict[str, dict],
    est: dict[tuple[str, str], dict],
    targets: dict[str, dict],
    layer: int,
    pos: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Behavioral repair test (panel families ONLY — MF1) + LOCO geometry reads."""
    repair_rows: list[dict[str, Any]] = []
    geometry_rows: list[dict[str, Any]] = []
    for cell_id, cell in sorted(cells.items()):
        tgt = targets[cell_id][f"L{layer}_{pos}"]
        names = tgt["persona_order"]
        M = tgt["M"]
        loco = bk.loco_w_shared(M, names)
        dv = {c: M[:, i] for i, c in enumerate(names)}
        prof_real = {c: float(dv[c] @ (loco[c] / np.linalg.norm(loco[c]))) for c in names}
        prof_norm = {c: float(np.linalg.norm(dv[c])) for c in names}
        behav = bk.load_behavioral_panel(cell["family"], cell["source"], cell["seed"], root=REPO)
        ep = est.get((cell["family"], cell["source"]))
        for estimator in ("est_tf", "est_icl", "est_desc"):
            w = None if ep is None else _w_hat(ep, estimator, layer, pos, cell)
            if w is None:
                continue
            wu = w / np.linalg.norm(w)
            prof_est = {c: float(dv[c] @ wu) for c in names}
            common = names
            rho_act_geom = spearman_rho(
                [prof_est[c] for c in common], [prof_real[c] for c in common]
            )
            rho_norm_vs_real = spearman_rho(
                [prof_norm[c] for c in common], [prof_real[c] for c in common]
            )
            geometry_rows.append(
                {
                    "cell_id": cell_id,
                    "estimator": estimator,
                    "rho_act_geom_loco": rho_act_geom,
                    "rho_norm_vs_real_loco": rho_norm_vs_real,
                    "n_contexts": len(common),
                    "note": "geometry-consistency read ONLY (no repair verdict)",
                }
            )
            if behav is None:
                continue  # marker519 / em_turner: NO repair verdict (registered)
            bc = [c for c in names if c in behav]
            if len(bc) < 3:
                continue
            bvals = [behav[c] for c in bc]
            rho_est = spearman_rho([prof_est[c] for c in bc], bvals)
            rho_real = spearman_rho([prof_real[c] for c in bc], bvals)
            rho_norm = spearman_rho([prof_norm[c] for c in bc], bvals)
            n = len(bc)
            repair_positive = rho_est < REPAIR_RHO_FAIL and rho_real >= REPAIR_RHO_PASS
            both_fail = rho_est < REPAIR_RHO_FAIL and rho_real < REPAIR_RHO_PASS
            edge = _edge_indeterminate(rho_est, REPAIR_RHO_FAIL, n) or _edge_indeterminate(
                rho_real, REPAIR_RHO_PASS, n
            )
            repair_rows.append(
                {
                    "cell_id": cell_id,
                    "family": cell["family"],
                    "estimator": estimator,
                    "rho_behav_est": rho_est,
                    "rho_behav_real": rho_real,
                    "rho_behav_norm": rho_norm,
                    "n_panel": n,
                    "edge_indeterminate_1se": edge,
                    "verdict": (
                        "edge-indeterminate"
                        if edge
                        else "repair-positive (estimation failed, update rule intact)"
                        if repair_positive
                        else "update-rule-implicated (realized write also fails)"
                        if both_fail
                        else "both-pass"
                        if rho_est >= REPAIR_RHO_FAIL and rho_real >= REPAIR_RHO_PASS
                        else "no-repair-needed-or-mixed"
                    ),
                    "norm_only_matches_real": abs(rho_norm - rho_real) < 0.1,
                }
            )
    return repair_rows, geometry_rows


def _anchor_check(est_dir: Path) -> dict[str, Any]:
    """anchor_521 band check: exact #521 recipe v_steer vs cached same-variant U1."""
    anchor: dict[str, Any] = {"checked": False}
    anchor_path = est_dir / "anchor_521.pt"
    if anchor_path.exists():
        ap = torch.load(anchor_path, map_location="cpu", weights_only=False)
        v_steer = np.asarray(ap["steering"]["v_steer"], dtype=np.float64)
        cached_dir = bk.eval_dir(REPO) / "cached_shifts"
        recorded = bk.anchor_521_recorded(REPO)
        per_seed: dict[str, Any] = {}
        all_ok = True
        for seed in bk.SEEDS_3:
            payload = load_cached_payload(cached_dir, "same", "marker", seed)
            order = sorted(payload["shifts"].keys())
            M, _ = assemble_M(payload["shifts"], persona_order=order)
            u1 = svd_summary(M)["U1"]
            got = cosine(v_steer, u1)
            ok = abs(got - recorded[seed]) <= bk.ANCHOR_521_BAND
            all_ok &= ok
            per_seed[f"seed{seed}"] = {"cos": got, "recorded": recorded[seed], "ok": ok}
        anchor = {
            "checked": True,
            "per_seed": per_seed,
            "band": bk.ANCHOR_521_BAND,
            "gate": "PASS" if all_ok else "FAIL — rig divergence; fix before interpreting",
        }
        if not all_ok:
            logger.error("[phase=p2_anchor] anchor_521 OUTSIDE band: %s", per_seed)
    else:
        logger.warning("[phase=p2_anchor] anchor_521.pt missing — anchor unchecked")
    return anchor


def _exploratory_grid(
    cells: dict[str, dict],
    est: dict[tuple[str, str], dict],
    targets: dict[str, dict],
    null95: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Full sweep grid + the select-on-42 / confirm-on-137-256 discipline."""
    grid_rows: list[dict[str, Any]] = []
    for cell_id, cell in sorted(cells.items()):
        ep = est.get((cell["family"], cell["source"]))
        if ep is None:
            continue
        for ly in bk.LAYERS:
            for ps in ("mean_resp", "slot"):
                key = f"L{ly}_{ps}"
                if key not in targets[cell_id]:
                    continue
                tgt = targets[cell_id][key]
                est_pos = "mean_resp" if ps == "mean_resp" else "last_tok"
                for estimator in ("est_tf", "est_icl", "est_desc"):
                    ks = bk.E2_K_SWEEP if estimator == "est_icl" else (bk.E2_K_PRIMARY,)
                    for k in ks:
                        w = _w_hat(ep, estimator, ly, est_pos, cell, k=k)
                        if w is None and est_pos == "last_tok":
                            # E2/E3 store last_prompt rather than last_tok
                            w = _w_hat(ep, estimator, ly, "last_prompt", cell, k=k)
                        if w is None:
                            continue
                        grid_rows.append(
                            {
                                "cell_id": cell_id,
                                "estimator": estimator,
                                "layer": ly,
                                "pos": ps,
                                "K": k if estimator == "est_icl" else None,
                                "cos_w_shared": cosine(w, tgt["w_shared"]),
                                "cos_w_src": (
                                    None if tgt["w_src"] is None else cosine(w, tgt["w_src"])
                                ),
                            }
                        )
    # select-on-42 / confirm-on-137-256 for the best swept construction
    select_confirm: dict[str, Any] = {}
    seed42_rows = [
        g
        for g in grid_rows
        if cells[g["cell_id"]]["seed"] == 42
        and cells[g["cell_id"]]["family"] in ("marker519", "em_turner", "fact541")
    ]
    if seed42_rows:
        best = max(seed42_rows, key=lambda g: g["cos_w_shared"])
        sel = (best["estimator"], best["layer"], best["pos"], best["K"])
        confirm = [
            g["cos_w_shared"]
            for g in grid_rows
            if (g["estimator"], g["layer"], g["pos"], g["K"]) == sel
            and cells[g["cell_id"]]["seed"] in (137, 256)
            and cells[g["cell_id"]]["family"] == cells[best["cell_id"]]["family"]
        ]
        select_confirm = {
            "selected_on_seed42": best,
            "confirm_cos_on_137_256": confirm,
            "survives": bool(confirm) and all(v > null95 for v in confirm),
        }
    return grid_rows, select_confirm


def phase2_score(args: argparse.Namespace) -> int:
    """Phase 2: the §4.4 scoring pseudocode over extracted + estimator payloads."""
    ev = bk.eval_dir(REPO)
    shifts_dir = Path(args.shifts_dir) if args.shifts_dir else ev / "shifts"
    est_dir = Path(args.estimator_dir) if args.estimator_dir else ev / "estimator_reads"

    logger.info("[phase=p2_load] shifts=%s estimators=%s", shifts_dir, est_dir)
    cells = _load_new_shift_payloads(shifts_dir)
    est = _load_estimator_payloads(est_dir)
    if not cells:
        raise RuntimeError(f"no shift payloads found under {shifts_dir} — run Phase 1 first")
    if not est:
        raise RuntimeError(f"no estimator payloads found under {est_dir} — run Phase 1c first")

    layer = bk.PRIMARY_LAYER
    pos = "mean_resp"  # pre-registered primary construction (L14 / mean-response)

    logger.info("[phase=p2_targets] per-cell w_src + w_shared")
    targets = _compute_targets(cells, layer, pos)
    ceiling = _seed_ceiling(cells, targets, layer, pos)

    logger.info("[phase=p2_nulls] 10k random-unit null")
    null_targets = [targets[cid][f"L{layer}_{pos}"]["w_shared"] for cid in sorted(targets)]
    null_cos = bk.random_null_cosines(null_targets[:8], n=N_RANDOM_NULL // 8, seed=602)
    null95 = float(np.percentile(null_cos, 95))
    null99 = float(np.percentile(null_cos, 99))

    logger.info("[phase=p2_headline] dual-target cosines at L%d/%s", layer, pos)
    rows = _headline_rows(cells, est, targets, layer, pos)
    logger.info("[phase=p2_offdiag] cross-behavior margins")
    margins, cross_recipe = _offdiag_margins(cells, est, targets, layer, pos)
    logger.info("[phase=p2_verdicts] family verdict table")
    verdicts = _verdict_table(rows, margins, null95)
    ladder = _h1_ladder(cells, rows, null95)
    logger.info("[phase=p2_repair] behavioral repair + geometry-consistency reads")
    repair_rows, geometry_rows = _repair_and_geometry(cells, est, targets, layer, pos)
    anchor = _anchor_check(est_dir)
    logger.info("[phase=p2_grid] exploratory sweep grid")
    grid_rows, select_confirm = _exploratory_grid(cells, est, targets, null95)

    headline = {
        "construction": {"layer": layer, "pos": pos, "K": bk.E2_K_PRIMARY, "variant": "base"},
        "null_random": {"p95": null95, "p99": null99, "n": int(null_cos.size)},
        "ceiling_seed": ceiling,
        "per_cell": rows,
        "margins": margins,
        "cross_recipe_sibling_reads": cross_recipe,
        "verdicts": verdicts,
        "h1_ladder": ladder,
        "anchor_521": anchor,
        "reproducibility": _meta(),
    }
    _write_json(ev / "agreement" / "headline_metrics.json", headline)
    _write_json(
        ev / "repair" / "repair_test.json",
        {
            "thresholds": {"rho_fail": REPAIR_RHO_FAIL, "rho_pass": REPAIR_RHO_PASS},
            "repair_rows": repair_rows,
            "geometry_consistency_rows": geometry_rows,
            "note": (
                "repair VERDICTS registered on behavioral-panel families ONLY "
                "(fact541, refusal518, em518, loc474); the activation half is a "
                "geometry-consistency read, never a repair verdict (MF1)"
            ),
            "reproducibility": _meta(),
        },
    )
    _write_json(
        ev / "grids" / "exploratory_grid.json",
        {
            "rows": grid_rows,
            "select_confirm": select_confirm,
            "reproducibility": _meta(),
        },
    )
    logger.info(
        "[phase=p2_done] %d cells, %d estimator units, %d headline rows, %d repair rows",
        len(cells),
        len(est),
        len(rows),
        len(repair_rows),
    )
    return 0


def main() -> int:
    """CLI: ``--phase repro-gate`` (Phase 0) or ``--phase score`` (Phase 2)."""
    parser = argparse.ArgumentParser(
        description="#602 Phase 0 reproduction gate + Phase 2 scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--phase", choices=["repro-gate", "score"], default="score")
    parser.add_argument(
        "--shifts-dir", default=None, help="Override eval_results/issue_602/shifts/"
    )
    parser.add_argument(
        "--estimator-dir", default=None, help="Override eval_results/issue_602/estimator_reads/"
    )
    parser.add_argument(
        "--skip-mix-pins",
        action="store_true",
        help=(
            "Phase 0: skip downloading every E1 mix for sha-pinning (faster smoke; "
            "the full gate run pins all of them)"
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    # Hub downloads need HF_TOKEN; .env is not auto-loaded under `uv run`.
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    if args.phase == "repro-gate":
        return phase0_repro_gate(args)
    return phase2_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
