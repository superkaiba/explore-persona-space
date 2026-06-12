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

# Inherited measurement caveats CARRIED INTO the scoring output (plan §6 /
# consistency-checker WARN 2): any rho_behav read on these families must be
# narrated with the producing task's known panel limitation.
BEHAVIORAL_PANEL_CAVEATS: dict[str, str] = {
    "em518": (
        "#518-EM behavioral delta is Sonnet aligned-rate on the ~15% "
        "coherence-survivor subset (judge-survivor proxy gap, inherited from "
        "the producing task) — not a raw misalignment rate"
    ),
    "refusal518": (
        "#518 refusal arm is power-limited (76% of panel cells floored in the "
        "producing task) — behavioral rho read only where the panel resolves"
    ),
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
            if local.is_symlink() and not local.exists():
                local.unlink()  # broken symlink (cache moved): re-link below
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


def _guard_payload_model(name: str, model_id: Any, strict: bool) -> None:
    """Stub-contamination guard: a payload whose manifest model id is not the
    production model is a smoke/stub artifact squatting at a production path.
    Production mode REJECTS it (raise); ``--allow-subset`` loads it loudly."""
    if model_id == bk.BASE_MODEL_ID:
        return
    if strict:
        raise RuntimeError(
            f"{name}: manifest model id {model_id!r} != production {bk.BASE_MODEL_ID!r} — "
            "stub/smoke payload at a production path (delete it or score with --allow-subset)"
        )
    logger.warning("%s from non-production model %r (allowed under --allow-subset)", name, model_id)


def _load_new_shift_payloads(shifts_dir: Path, strict: bool) -> dict[str, dict]:
    """Load every freshly extracted cell payload, keyed by cell_id."""
    out: dict[str, dict] = {}
    for cell in bk.extraction_cells():
        p = shifts_dir / f"{cell['cell_id']}.pt"
        if not p.exists():
            logger.warning("missing shift payload for %s (%s)", cell["cell_id"], p)
            continue
        payload = torch.load(p, map_location="cpu", weights_only=False)
        _guard_payload_model(p.name, payload.get("manifest", {}).get("base_model_id"), strict)
        if strict:  # assumption-12 dim assert (production payloads only)
            v = next(iter(payload["shifts"].values()))["delta_v_mean_resp"]
            assert v.numel() == bk.HIDDEN_SIZE, (p.name, int(v.numel()), bk.HIDDEN_SIZE)
        out[cell["cell_id"]] = {**cell, "payload": payload}
    return out


def _load_estimator_payloads(est_dir: Path, strict: bool) -> dict[tuple[str, str], dict]:
    """Load estimator-read payloads keyed by (family, source)."""
    out: dict[tuple[str, str], dict] = {}
    for unit in bk.estimator_units():
        p = est_dir / f"{unit['family']}__{unit['source']}.pt"
        if not p.exists():
            logger.warning("missing estimator payload for %s/%s", unit["family"], unit["source"])
            continue
        payload = torch.load(p, map_location="cpu", weights_only=False)
        _guard_payload_model(p.name, payload.get("manifest", {}).get("model_id"), strict)
        out[(unit["family"], unit["source"])] = payload
    return out


def _phase2_preflight(shifts_dir: Path, est_dir: Path, allow_subset: bool) -> dict[str, Any]:
    """Completeness gate BEFORE scoring (binding fix: subset-denominator).

    Production (default) expects the FULL registered inputs — all 31 shift
    payloads, all 21 estimator payloads, ``anchor_521.pt``, and a passing
    production ``i474_crosscheck.json`` (assumption-8 gate) — and RAISES on
    any gap, so an accidental partial download can never be scored under
    ``registered_denominator: 31``. Subset scoring (the §9 deliberate-descope
    path and smoke) requires the explicit ``--allow-subset`` flag; the
    returned coverage dict is recorded as ``expected_vs_loaded`` in EVERY
    output JSON so a subset run can never masquerade as the registered run.
    """
    expected_cells = [c["cell_id"] for c in bk.extraction_cells()]
    expected_units = [f"{u['family']}__{u['source']}" for u in bk.estimator_units()]
    missing_shifts = [c for c in expected_cells if not (shifts_dir / f"{c}.pt").exists()]
    missing_units = [u for u in expected_units if not (est_dir / f"{u}.pt").exists()]
    anchor_present = (est_dir / "anchor_521.pt").exists()
    cc_path = bk.eval_dir(REPO) / "work" / "i474_crosscheck.json"
    cc = json.loads(cc_path.read_text()) if cc_path.exists() else None
    cc_summary = {"present": cc is not None} | (
        {k: cc.get(k) for k in ("ok", "production_model", "max_abs_diff", "n_pairs")} if cc else {}
    )
    coverage = {
        "mode": "allow-subset" if allow_subset else "production",
        "n_shift_payloads_expected": len(expected_cells),
        "n_shift_payloads_present": len(expected_cells) - len(missing_shifts),
        "missing_shift_cells": missing_shifts,
        "n_estimator_units_expected": len(expected_units),
        "n_estimator_units_present": len(expected_units) - len(missing_units),
        "missing_estimator_units": missing_units,
        "anchor_521_present": anchor_present,
        "i474_crosscheck": cc_summary,
    }
    problems: list[str] = []
    if missing_shifts:
        problems.append(f"{len(missing_shifts)}/31 shift payloads missing: {missing_shifts}")
    if missing_units:
        problems.append(f"{len(missing_units)}/21 estimator payloads missing: {missing_units}")
    if not anchor_present:
        problems.append(f"anchor_521.pt missing under {est_dir} (registered kill criterion)")
    if cc is None or not cc.get("ok") or not cc.get("production_model"):
        problems.append(
            f"i474 prompt-reconstruction cross-check not satisfied ({cc_summary}) — "
            "assumption-8 gate (dispatcher [phase=i474_check] produces it)"
        )
    if problems:
        msg = "Phase 2 preflight: " + "; ".join(problems)
        if not allow_subset:
            raise RuntimeError(
                msg + " — production scoring refuses a subset; pass --allow-subset ONLY for a "
                "deliberate §9 descope or smoke run (coverage is then recorded in every output)"
            )
        logger.warning("%s — proceeding under --allow-subset", msg)
    return coverage


def _w_hat(
    est_payload: dict,
    estimator: str,
    layer: int,
    pos: str,
    cell: dict,
    k: int = bk.E2_K_PRIMARY,
    e1_pos_override: str | None = None,
) -> np.ndarray | None:
    """Resolve one estimator vector at (layer, pos) for a run-cell.

    For ``est_tf`` on per-seed-mix families the seed-matched E1 unit is
    used; the marker families' HEADLINE E1 read is the exclude-marker
    mean-over-completion (token-identity discriminator, plan H3) — the
    include-marker read is reported in the exploratory grid via
    ``e1_pos_override="mean_resp"``.
    """
    family = cell["family"]
    if estimator == "est_tf":
        mix_label = f"seed{cell['seed']}" if family in ("marker519", "fact541") else "shared"
        unit = est_payload["e1"].get(mix_label)
        if unit is None:
            return None
        pos_key = pos
        implicit_excl_marker = False
        if e1_pos_override is not None:
            pos_key = e1_pos_override
        elif pos == "mean_resp" and family in ("marker519", "loc474"):
            pos_key = "mean_resp_excl_marker"
            implicit_excl_marker = True
        w = unit["w_hat"].get(pos_key)
        if w is None and implicit_excl_marker:
            # NEVER silently swap the marker families' exclude-marker headline
            # read for the include-marker mean — that is a different DV.
            raise KeyError(
                f"E1 w_hat missing the {pos_key!r} headline read for "
                f"{family}/{cell['source']} (mix {mix_label!r})"
            )
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
    rows: list[dict[str, Any]], margins: dict[str, Any], null95: float, strict: bool
) -> dict[str, Any]:
    """3 estimators x 6 families verdict table (MF2 dual-target / MF3 median).

    ``strict`` (production): a family registered with 3 seeds that shows
    fewer in the loaded rows RAISES instead of silently relaxing the
    registered per-seed 2-of-3 disjunction to a pooled median (binding fix:
    the relaxation is allowed only under ``--allow-subset``, and is then
    labeled via ``seed_criterion``). A family x estimator with ZERO rows
    likewise raises in strict mode (payload-internal incompleteness).
    """
    registered_seeds = {
        fam: sorted({c["seed"] for c in bk.extraction_cells() if c["family"] == fam})
        for fam in bk.FAMILIES
    }
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
                if strict:
                    raise RuntimeError(
                        f"verdict table: ZERO loaded rows for {family}/{estimator} in "
                        "production mode — payload-internal incompleteness (the preflight "
                        "verified the files exist, so a read inside a payload is missing)"
                    )
                verdicts[f"{family}__{estimator}"] = {"verdict": "MISSING"}
                continue
            med_shared = float(np.median([r["cos_w_shared"] for r in per_cell]))
            med_src = float(np.median([r["cos_w_src"] for r in per_cell]))
            # per-seed pass counts on the registered disjunction
            seeds = sorted({r["seed"] for r in per_cell})
            if strict and seeds != registered_seeds[family]:
                raise RuntimeError(
                    f"verdict table: {family}/{estimator} loaded seeds {seeds} != registered "
                    f"{registered_seeds[family]} in production mode — the registered per-seed "
                    "2-of-3 disjunction cannot be evaluated; the pooled-median relaxation is "
                    "allowed only under --allow-subset"
                )
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
            per_seed_criterion = len(seeds) >= 3
            seed_frac_ok = (
                (n_pass_shared >= 2 or n_pass_src >= 2)
                if per_seed_criterion
                else (med_shared >= VALIDITY_COS or med_src >= VALIDITY_COS)
            )
            seed_criterion = (
                "per-seed-2of3-disjunction"
                if per_seed_criterion
                else (
                    "pooled-median (single-run family, registered)"
                    if len(registered_seeds[family]) < 3
                    else "pooled-median (RELAXED under --allow-subset: seeds incomplete)"
                )
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
                "registered_n_seeds": len(registered_seeds[family]),
                "seed_criterion": seed_criterion,
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
            # companion read (NOT registered eligibility): w_src-only signal
            if any(
                cs[e]["cos_w_src"] is not None and cs[e]["cos_w_src"] > null95
                for e in ("est_tf", "est_icl", "est_desc")
            ):
                ladder.setdefault("src_only_signal_cells", []).append(cell_id)
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
                    # per-context profiles persisted for the raw-alongside-ranked
                    # projection scatter (plan §6 exploratory dump)
                    "profiles": {
                        c: {
                            "est": prof_est[c],
                            "real": prof_real[c],
                            "norm": prof_norm[c],
                            "behav": (None if behav is None else behav.get(c)),
                        }
                        for c in names
                    },
                }
            )
            if behav is None:
                continue  # marker519 / em_turner: NO repair verdict (registered)
            bc = [c for c in names if c in behav]
            if len(bc) < 3:
                logger.warning(
                    "repair: %s/%s behavioral-panel ∩ extracted-contexts = %d < 3 — "
                    "repair row skipped (production intersections are >= 4; this firing "
                    "outside smoke means a panel/context regression)",
                    cell_id,
                    estimator,
                    len(bc),
                )
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
                    "caveat": BEHAVIORAL_PANEL_CAVEATS.get(cell["family"]),
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
        anchor_model = ap.get("manifest", {}).get("base_model_id")
        if anchor_model != bk.BASE_MODEL_ID:
            # CPU-stub smoke payloads (hidden size != 3584) can never be
            # meaningfully banded against the real cached tensors; record the
            # skip LOUDLY instead of crashing on the dim mismatch. Production
            # anchors always run on BASE_MODEL_ID, so this branch is
            # unreachable in the sweep.
            logger.warning(
                "[phase=p2_anchor] anchor payload from non-production model %r — "
                "band check SKIPPED (stub smoke payload)",
                anchor_model,
            )
            return {
                "checked": False,
                "skipped_reason": f"non-production anchor model id {anchor_model!r}",
            }
        cached_dir = bk.eval_dir(REPO) / "cached_shifts"
        recorded = bk.anchor_521_recorded(REPO)
        per_seed: dict[str, Any] = {}
        all_ok = True
        for seed in bk.SEEDS_3:
            payload = load_cached_payload(cached_dir, "same", "marker", seed)
            order = sorted(payload["shifts"].keys())
            M, _ = assemble_M(payload["shifts"], persona_order=order)
            u1 = svd_summary(M)["U1"]
            assert v_steer.size == u1.size, (v_steer.size, u1.size)
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


def _anchor_fatal(anchor: dict[str, Any], strict: bool) -> str | None:
    """Anchor kill-criterion enforcement (binding fix: anchor-fail-not-fatal).

    Plan §7 registers anchor_521 as a KILL criterion against silent rig
    divergence. In production (``strict``) a band FAIL — or an anchor that
    never ran (missing file / non-production skip, ``checked: false``) —
    returns the fatal reason; ``phase2_score`` exits nonzero AFTER all JSONs
    are written. ``--allow-subset`` downgrades to a warning (returns None).
    """
    if anchor.get("checked") and not str(anchor.get("gate", "")).startswith("FAIL"):
        return None
    if not anchor.get("checked"):
        reason = (
            "anchor_521 kill criterion NEVER RAN: "
            f"{anchor.get('skipped_reason', 'anchor_521.pt missing')}"
        )
    else:
        reason = f"anchor_521 band check FAILED (rig divergence): {anchor.get('per_seed')}"
    if strict:
        return reason
    logger.warning("[phase=p2_anchor] %s — non-fatal under --allow-subset", reason)
    return None


ARM_BY_CACHED_FAMILY = {"marker519": "marker", "em_turner": "em"}


def _same_variant_sensitivity(
    cells: dict[str, dict],
    est: dict[tuple[str, str], dict],
    headline_rows: list[dict[str, Any]],
    layer: int,
    pos: str,
) -> dict[str, Any]:
    """Plan §4.3 same-variant sensitivity read (concern
    same-variant-estimator-sensitivity-not-wired): estimator cosines against
    the CACHED same-variant realized targets for the cached families
    (marker519 / EM-turner). Zero GPU — the 18 cached #551 tensors are local.

    This is a CONSTRUCTION-SENSITIVITY read, clearly separated from the
    MF-binding variant-matched ``base`` headline: the Phase-0 base-vs-same
    divergence (marker U1 cos 0.575-0.673 — the §11 ">0.1 revisit" trigger
    fires for the marker family) means marker-cell base-variant cosines must
    be narrated alongside this same-variant counterpart.
    """
    cached_dir = bk.eval_dir(REPO) / "cached_shifts"
    base_by_key = {(r["cell_id"], r["estimator"]): r for r in headline_rows}
    rows: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for cell_id, cell in sorted(cells.items()):
        arm = ARM_BY_CACHED_FAMILY.get(cell["family"])
        if arm is None:
            continue
        p = cached_dir / f"same_{arm}_seed{cell['seed']}.pt"
        if not p.exists():
            skipped.append({"cell_id": cell_id, "reason": f"cached payload missing: {p.name}"})
            continue
        payload = torch.load(p, map_location="cpu", weights_only=False)
        entries = payload["shifts"]
        cols = {c: _entry_vec(entries[c], layer, pos) for c in sorted(entries)}
        cols = {c: v for c, v in cols.items() if v is not None}
        names = sorted(cols)
        M = np.stack([cols[c] for c in names], axis=1)
        summ = svd_summary(M)
        w_shared_same = summ["U1"].astype(np.float64)
        w_src_same = cols.get(cell["source"])
        ep = est.get((cell["family"], cell["source"]))
        for estimator in ("est_tf", "est_icl", "est_desc"):
            w = None if ep is None else _w_hat(ep, estimator, layer, pos, cell)
            if w is None:
                skipped.append(
                    {
                        "cell_id": cell_id,
                        "estimator": estimator,
                        "reason": "estimator vector missing",
                    }
                )
                continue
            if w.size != w_shared_same.size:
                skipped.append(
                    {
                        "cell_id": cell_id,
                        "estimator": estimator,
                        "reason": (
                            f"dim mismatch: estimator {w.size} vs cached {w_shared_same.size} "
                            "(stub smoke payload — real values land on the production run)"
                        ),
                    }
                )
                continue
            base_row = base_by_key.get((cell_id, estimator), {})
            rows.append(
                {
                    "cell_id": cell_id,
                    "family": cell["family"],
                    "source": cell["source"],
                    "seed": cell["seed"],
                    "estimator": estimator,
                    "cos_same_w_shared": cosine(w, w_shared_same),
                    "cos_same_w_src": None if w_src_same is None else cosine(w, w_src_same),
                    "w_src_note": (
                        None
                        if w_src_same is not None
                        else f"source context {cell['source']!r} not in the cached 14-persona panel"
                    ),
                    "s_top1_frac_same": summ["s_top1_frac"],
                    # base-variant counterparts (the headline) for the direct
                    # same-vs-base sensitivity delta
                    "cos_base_w_shared": base_row.get("cos_w_shared"),
                    "cos_base_w_src": base_row.get("cos_w_src"),
                }
            )
    return {
        "construction": {"layer": layer, "pos": pos, "variant": "same (cached #551 tensors)"},
        "note": (
            "construction-sensitivity read ONLY — the MF-binding headline is the "
            "variant-matched `base` read in agreement/headline_metrics.json; the Phase-0 "
            "base-vs-same divergence numbers live in phase0/base_vs_same.json (marker U1 "
            "cos 0.575-0.673 fires the plan §11 '>0.1 revisit' trigger, so marker-family "
            "base-variant cosines must be narrated alongside these same-variant reads)"
        ),
        "rows": rows,
        "skipped": skipped,
    }


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
                marker_family = cell["family"] in ("marker519", "loc474")
                for estimator in ("est_tf", "est_icl", "est_desc"):
                    ks = bk.E2_K_SWEEP if estimator == "est_icl" else (bk.E2_K_PRIMARY,)
                    for k in ks:
                        # (all three estimators store BOTH last_tok and
                        # last_prompt; no fallback needed — a missing read
                        # simply skips the row)
                        w = _w_hat(ep, estimator, ly, est_pos, cell, k=k)
                        if w is None:
                            continue
                        grid_rows.append(
                            {
                                "cell_id": cell_id,
                                "estimator": estimator,
                                "layer": ly,
                                "pos": ps,
                                "K": k if estimator == "est_icl" else None,
                                "e1_read": (
                                    "excl_marker"
                                    if estimator == "est_tf"
                                    and marker_family
                                    and est_pos == "mean_resp"
                                    else None
                                ),
                                "cos_w_shared": cosine(w, tgt["w_shared"]),
                                "cos_w_src": (
                                    None if tgt["w_src"] is None else cosine(w, tgt["w_src"])
                                ),
                            }
                        )
                        # include-marker E1 companion row (plan §6 exploratory
                        # dump: include-vs-exclude-marker E1 reads). The
                        # headline/default grid read for marker families is the
                        # exclude-marker mean (token-identity discriminator).
                        if estimator == "est_tf" and marker_family and est_pos == "mean_resp":
                            w_incl = _w_hat(
                                ep, estimator, ly, est_pos, cell, e1_pos_override="mean_resp"
                            )
                            if w_incl is not None:
                                grid_rows.append(
                                    {
                                        "cell_id": cell_id,
                                        "estimator": estimator,
                                        "layer": ly,
                                        "pos": ps,
                                        "K": None,
                                        "e1_read": "incl_marker",
                                        "cos_w_shared": cosine(w_incl, tgt["w_shared"]),
                                        "cos_w_src": (
                                            None
                                            if tgt["w_src"] is None
                                            else cosine(w_incl, tgt["w_src"])
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
        sel = (best["estimator"], best["layer"], best["pos"], best["K"], best.get("e1_read"))
        confirm = [
            g["cos_w_shared"]
            for g in grid_rows
            if (g["estimator"], g["layer"], g["pos"], g["K"], g.get("e1_read")) == sel
            and cells[g["cell_id"]]["seed"] in (137, 256)
            and cells[g["cell_id"]]["family"] == cells[best["cell_id"]]["family"]
        ]
        select_confirm = {
            "selected_on_seed42": best,
            "confirm_cos_on_137_256": confirm,
            "survives": bool(confirm) and all(v > null95 for v in confirm),
        }
    return grid_rows, select_confirm


def _reliability(est: dict[tuple[str, str], dict], layer: int) -> dict[str, Any]:
    """Split-half reliabilities for ALL THREE estimators + E1 row-count
    subsample stability, computed from the contractually persisted per-row
    L14 stacks (plan §6 exploratory dump; per-row persistence is what makes
    the SNR-ladder alternative weighable post-hoc)."""
    rng = np.random.default_rng(602)
    out: dict[str, Any] = {}
    for (family, source), ep in sorted(est.items()):
        unit: dict[str, Any] = {}
        # --- E1: behavior vs base-self per-row stacks (mean_resp; marker
        # families use the exclude-marker headline read where present) ---
        for mix_label, e1 in ep.get("e1", {}).items():
            beh_rows = e1.get("per_row_behavior", {})
            pos_key = (
                "mean_resp_excl_marker" if "mean_resp_excl_marker" in beh_rows else "mean_resp"
            )
            beh_t = beh_rows.get(pos_key, {}).get(layer)
            base_t = e1.get("per_row_base_self", {}).get("mean_resp", {}).get(layer)
            if beh_t is None or base_t is None:
                continue
            beh = beh_t.float().numpy()
            base = base_t.float().numpy()
            n = min(len(beh), len(base))
            if n < 2:
                unit[f"e1__{mix_label}"] = {"n_rows": int(n), "note": "too few rows"}
                continue
            idx = np.arange(n)
            w1 = beh[idx % 2 == 0].mean(0) - base[idx % 2 == 0].mean(0)
            w2 = beh[idx % 2 == 1].mean(0) - base[idx % 2 == 1].mean(0)
            entry: dict[str, Any] = {
                "n_rows": int(n),
                "pos_key": pos_key,
                "split_half_cos": cosine(w1, w2),
            }
            w_full = beh[:n].mean(0) - base[:n].mean(0)
            stability: dict[str, Any] = {}
            for frac in (0.25, 0.5, 0.75):
                k = max(2, round(frac * n))
                cs = []
                for _ in range(3):
                    sub = rng.choice(n, size=k, replace=False)
                    cs.append(cosine(beh[sub].mean(0) - base[sub].mean(0), w_full))
                stability[f"frac_{frac}"] = {"k": int(k), "mean_cos_vs_full": float(np.mean(cs))}
            entry["subsample_stability_vs_full"] = stability
            unit[f"e1__{mix_label}"] = entry
        # --- E2 (per K): with-demo vs zero-demo per-probe stacks ---
        zero_t = (
            ep.get("e2", {}).get("zero_demo", {}).get("per_probe", {}).get("mean_resp", {})
        ).get(layer)
        for kname, e2 in ep.get("e2", {}).items():
            if not kname.startswith("K"):
                continue
            wd_t = e2.get("per_probe_with_demos", {}).get("mean_resp", {}).get(layer)
            if wd_t is None or zero_t is None:
                continue
            wd = wd_t.float().numpy()
            z = zero_t.float().numpy()
            if len(wd) < 2 or len(z) < 2:
                unit[f"e2__{kname}"] = {"n_with": len(wd), "note": "too few probes"}
                continue
            w1 = wd[0::2].mean(0) - z[0::2].mean(0)
            w2 = wd[1::2].mean(0) - z[1::2].mean(0)
            unit[f"e2__{kname}"] = {
                "n_with": len(wd),
                "n_zero": len(z),
                "split_half_cos": cosine(w1, w2),
            }
        # --- E3: desc vs no-desc per-probe stacks ---
        de_t = ep.get("e3", {}).get("per_probe_desc", {}).get("mean_resp", {}).get(layer)
        no_t = ep.get("e3", {}).get("per_probe_nodesc", {}).get("mean_resp", {}).get(layer)
        if de_t is not None and no_t is not None and len(de_t) >= 2 and len(no_t) >= 2:
            de = de_t.float().numpy()
            no = no_t.float().numpy()
            w1 = de[0::2].mean(0) - no[0::2].mean(0)
            w2 = de[1::2].mean(0) - no[1::2].mean(0)
            unit["e3"] = {"n_probes": len(de), "split_half_cos": cosine(w1, w2)}
        out[f"{family}__{source}"] = unit
    return out


def _cross_estimator(
    cells: dict[str, dict], est: dict[tuple[str, str], dict], layer: int, pos: str
) -> dict[str, Any]:
    """H5 cross-estimator coherence: pairwise cos(w_i, w_j) per unit, plus
    SAME-estimator cross-family matrices (generic-prompting-attractor
    diagnostic: est_icl(A) vs est_icl(B) etc.)."""
    unit_vecs: dict[tuple[str, str, str], np.ndarray] = {}
    for (family, source), ep in est.items():
        cell0 = next(
            (c for c in cells.values() if c["family"] == family and c["source"] == source), None
        )
        if cell0 is None:
            continue
        for estimator in ("est_tf", "est_icl", "est_desc"):
            w = _w_hat(ep, estimator, layer, pos, cell0)
            if w is not None:
                unit_vecs[(family, source, estimator)] = w
    pairwise: dict[str, Any] = {}
    for family, source, _e in {(f, s, "x") for (f, s, _x) in unit_vecs}:
        ests = [e for e in ("est_tf", "est_icl", "est_desc") if (family, source, e) in unit_vecs]
        entry = {}
        for i, ea in enumerate(ests):
            for eb in ests[i + 1 :]:
                entry[f"{ea}__vs__{eb}"] = cosine(
                    unit_vecs[(family, source, ea)], unit_vecs[(family, source, eb)]
                )
        pairwise[f"{family}__{source}"] = entry
    same_estimator_cross_family: dict[str, Any] = {}
    for estimator in ("est_tf", "est_icl", "est_desc"):
        keys = sorted([k for k in unit_vecs if k[2] == estimator])
        mat = {}
        for i, ka in enumerate(keys):
            for kb in keys[i + 1 :]:
                if ka[0] == kb[0]:
                    continue  # cross-FAMILY diagnostic only
                mat[f"{ka[0]}__{ka[1]}|{kb[0]}__{kb[1]}"] = cosine(unit_vecs[ka], unit_vecs[kb])
        same_estimator_cross_family[estimator] = mat
    return {
        "pairwise_within_unit": pairwise,
        "same_estimator_cross_family": same_estimator_cross_family,
    }


def phase2_score(args: argparse.Namespace) -> int:
    """Phase 2: the §4.4 scoring pseudocode over extracted + estimator payloads.

    Default = PRODUCTION mode: the preflight requires the full registered
    inputs (31 shifts, 21 estimator units, anchor_521, passing i474
    cross-check) and any registered gate failure exits nonzero AFTER all
    JSONs are written. ``--allow-subset`` is the explicit §9-descope/smoke
    path; the ``expected_vs_loaded`` coverage block is recorded in EVERY
    output JSON in both modes.
    """
    ev = bk.eval_dir(REPO)
    shifts_dir = Path(args.shifts_dir) if args.shifts_dir else ev / "shifts"
    est_dir = Path(args.estimator_dir) if args.estimator_dir else ev / "estimator_reads"
    strict = not args.allow_subset

    logger.info(
        "[phase=p2_preflight] mode=%s shifts=%s estimators=%s",
        "production" if strict else "allow-subset",
        shifts_dir,
        est_dir,
    )
    expected_vs_loaded = _phase2_preflight(shifts_dir, est_dir, args.allow_subset)

    logger.info("[phase=p2_load]")
    cells = _load_new_shift_payloads(shifts_dir, strict)
    est = _load_estimator_payloads(est_dir, strict)
    if not cells:
        raise RuntimeError(f"no shift payloads found under {shifts_dir} — run Phase 1 first")
    if not est:
        raise RuntimeError(f"no estimator payloads found under {est_dir} — run Phase 1c first")
    expected_vs_loaded["n_shift_payloads_loaded"] = len(cells)
    expected_vs_loaded["n_estimator_units_loaded"] = len(est)

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
    if strict:
        miss = sorted(f"{r['cell_id']}/{r['estimator']}" for r in rows if r["missing_estimator"])
        if miss:
            raise RuntimeError(
                f"{len(miss)} (cell x estimator) reads missing despite complete payload files "
                f"(payload-internal incompleteness): {miss[:10]}"
            )
    logger.info("[phase=p2_offdiag] cross-behavior margins")
    margins, cross_recipe = _offdiag_margins(cells, est, targets, layer, pos)
    logger.info("[phase=p2_verdicts] family verdict table")
    verdicts = _verdict_table(rows, margins, null95, strict)
    ladder = _h1_ladder(cells, rows, null95)
    logger.info("[phase=p2_repair] behavioral repair + geometry-consistency reads")
    repair_rows, geometry_rows = _repair_and_geometry(cells, est, targets, layer, pos)
    anchor = _anchor_check(est_dir)
    logger.info("[phase=p2_grid] exploratory sweep grid")
    grid_rows, select_confirm = _exploratory_grid(cells, est, targets, null95)
    cross_estimator = _cross_estimator(cells, est, layer, pos)
    logger.info("[phase=p2_reliability] split-half + subsample stability")
    reliability = _reliability(est, layer)
    logger.info("[phase=p2_same_variant] cached same-variant sensitivity read")
    same_variant = _same_variant_sensitivity(cells, est, rows, layer, pos)

    headline = {
        "construction": {"layer": layer, "pos": pos, "K": bk.E2_K_PRIMARY, "variant": "base"},
        "expected_vs_loaded": expected_vs_loaded,
        "null_random": {"p95": null95, "p99": null99, "n": int(null_cos.size)},
        "ceiling_seed": ceiling,
        "per_cell": rows,
        "margins": margins,
        "cross_recipe_sibling_reads": cross_recipe,
        "verdicts": verdicts,
        "h1_ladder": ladder,
        "anchor_521": anchor,
        "cross_estimator": cross_estimator,
        "reproducibility": _meta(),
    }
    _write_json(ev / "agreement" / "headline_metrics.json", headline)
    _write_json(
        ev / "agreement" / "same_variant_sensitivity.json",
        {**same_variant, "expected_vs_loaded": expected_vs_loaded, "reproducibility": _meta()},
    )
    _write_json(
        ev / "repair" / "repair_test.json",
        {
            "thresholds": {"rho_fail": REPAIR_RHO_FAIL, "rho_pass": REPAIR_RHO_PASS},
            "behavioral_panel_caveats": BEHAVIORAL_PANEL_CAVEATS,
            "expected_vs_loaded": expected_vs_loaded,
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
            "reliability": reliability,
            "expected_vs_loaded": expected_vs_loaded,
            "reproducibility": _meta(),
        },
    )
    # Registered kill criterion LAST — all JSONs above are already on disk,
    # so the nonzero exit destroys no artifact (binding fix #2).
    fatal = _anchor_fatal(anchor, strict)
    if fatal:
        logger.error("[phase=p2_gate] %s — exiting nonzero; fix before interpreting", fatal)
        return 1
    logger.info(
        "[phase=p2_done] %d cells, %d estimator units, %d headline rows, %d repair rows, "
        "%d same-variant rows",
        len(cells),
        len(est),
        len(rows),
        len(repair_rows),
        len(same_variant["rows"]),
    )
    return 0


def _cached_layer_coverage(layer: int) -> dict[str, Any]:
    """Which cached #551 payloads carry the ``delta_v_mean_resp_l{layer}`` key.

    The cached same-variant tensors were extracted by #551 BEFORE the M3a
    layer extension added l3/l27, so a non-(7,14,21) re-read layer is
    expected to be absent there — the same-variant sensitivity read and the
    (L14-registered) anchor_521 band check are then EXCLUDED with this
    explicit accounting rather than recomputed or fabricated.
    """
    cached_dir = bk.eval_dir(REPO) / "cached_shifts"
    key = "delta_v_mean_resp" if layer == bk.PRIMARY_LAYER else f"delta_v_mean_resp_l{layer}"
    present, absent, missing_files = [], [], []
    for variant, arm, seed in CACHED_CELLS:
        name = f"{variant}_{arm}_seed{seed}"
        p = cached_dir / f"{name}.pt"
        if not p.exists():
            missing_files.append(name)
            continue
        payload = torch.load(p, map_location="cpu", weights_only=False)
        has = all(e.get(key) is not None for e in payload["shifts"].values())
        (present if has else absent).append(name)
    return {
        "key": key,
        "n_cached_expected": len(CACHED_CELLS),
        "with_layer": present,
        "without_layer": absent,
        "missing_files": missing_files,
    }


def phase2_layer_reread(args: argparse.Namespace) -> int:
    """Free-analysis re-read of the verdict table + repair test at ``--layer N``.

    Follow-up 1 on the #602 clean result ("is the broken link the read-out
    layer or the formula's update rule?"): re-derive per-cell targets
    (w_src, w_shared) from the stored ``delta_v_mean_resp_l{N}`` keys and
    re-run the dual-target verdict table and the behavioral repair test with
    the SAME pre-registered rules (0.3 validity bar, 10k random-unit null —
    layer-independent by symmetry at d=3584, recomputed via the same code
    path/seed — sibling-excluded margin >= 0.2, per-seed 2-of-3 disjunction)
    and the SAME repair thresholds + verdict denominators (marker-arm loc474
    rows excluded from verdict counts as direction-insensitive).

    Outputs ``agreement/l{N}_reread.json`` + ``repair/repair_test_l{N}.json``;
    the registered L14 headline files are never touched. The cached #551
    same-variant tensors carry no l{N} keys, so the same-variant sensitivity
    read and the L14-registered anchor_521 band check are EXCLUDED with
    explicit coverage accounting (never silently skipped).
    """
    ev = bk.eval_dir(REPO)
    shifts_dir = Path(args.shifts_dir) if args.shifts_dir else ev / "shifts"
    est_dir = Path(args.estimator_dir) if args.estimator_dir else ev / "estimator_reads"
    strict = not args.allow_subset
    layer = args.layer
    pos = "mean_resp"
    assert layer != bk.PRIMARY_LAYER, "re-read mode is for non-registered layers only"

    logger.info(
        "[phase=reread_preflight] layer=%d mode=%s shifts=%s estimators=%s",
        layer,
        "production" if strict else "allow-subset",
        shifts_dir,
        est_dir,
    )
    expected_vs_loaded = _phase2_preflight(shifts_dir, est_dir, args.allow_subset)
    cells = _load_new_shift_payloads(shifts_dir, strict)
    est = _load_estimator_payloads(est_dir, strict)
    if not cells or not est:
        raise RuntimeError("re-read needs the Phase-1 shift + estimator payloads on disk")
    expected_vs_loaded["n_shift_payloads_loaded"] = len(cells)
    expected_vs_loaded["n_estimator_units_loaded"] = len(est)

    logger.info("[phase=reread_targets] per-cell w_src + w_shared at L%d/%s", layer, pos)
    targets = _compute_targets(cells, layer, pos)
    # context-coverage accounting: the L{N} read must span the SAME contexts
    # as the registered L14 construction (a persona silently dropped at L{N}
    # would shrink the SVD panel without erroring).
    context_mismatch = {
        cid: {
            "n_contexts_at_layer": len(targets[cid][f"L{layer}_{pos}"]["persona_order"]),
            "n_contexts_at_primary": len(
                targets[cid][f"L{bk.PRIMARY_LAYER}_{pos}"]["persona_order"]
            ),
        }
        for cid in sorted(targets)
        if targets[cid][f"L{layer}_{pos}"]["persona_order"]
        != targets[cid][f"L{bk.PRIMARY_LAYER}_{pos}"]["persona_order"]
    }
    if context_mismatch and strict:
        raise RuntimeError(
            f"L{layer} context panels diverge from the registered L14 panels for "
            f"{sorted(context_mismatch)} — re-read denominators would not be comparable"
        )
    cached_coverage = _cached_layer_coverage(layer)
    layer_coverage = {
        "fresh_payloads": {
            "n_expected": len(bk.extraction_cells()),
            "n_with_layer": len(targets),
            "context_panel_mismatch_vs_primary": context_mismatch,
        },
        "cached_same_variant": cached_coverage,
    }

    ceiling = _seed_ceiling(cells, targets, layer, pos)
    logger.info("[phase=reread_nulls] 10k random-unit null (layer-independent at d=3584)")
    null_targets = [targets[cid][f"L{layer}_{pos}"]["w_shared"] for cid in sorted(targets)]
    null_cos = bk.random_null_cosines(null_targets[:8], n=N_RANDOM_NULL // 8, seed=602)
    null95 = float(np.percentile(null_cos, 95))
    null99 = float(np.percentile(null_cos, 99))

    logger.info("[phase=reread_headline] dual-target cosines at L%d/%s", layer, pos)
    rows = _headline_rows(cells, est, targets, layer, pos)
    if strict:
        miss = sorted(f"{r['cell_id']}/{r['estimator']}" for r in rows if r["missing_estimator"])
        if miss:
            raise RuntimeError(
                f"{len(miss)} (cell x estimator) reads missing at L{layer} despite complete "
                f"payload files (payload-internal incompleteness): {miss[:10]}"
            )
    margins, cross_recipe = _offdiag_margins(cells, est, targets, layer, pos)
    logger.info("[phase=reread_verdicts] family verdict table at L%d", layer)
    verdicts = _verdict_table(rows, margins, null95, strict)
    ladder = _h1_ladder(cells, rows, null95)
    logger.info("[phase=reread_repair] behavioral repair + geometry reads at L%d", layer)
    repair_rows, geometry_rows = _repair_and_geometry(cells, est, targets, layer, pos)
    cross_estimator = _cross_estimator(cells, est, layer, pos)

    anchor = {
        "checked": False,
        "skipped_reason": (
            f"anchor_521 band is registered at the L{bk.PRIMARY_LAYER} construction (#521 "
            f"recipe) and the cached #551 tensors carry no l{layer} keys "
            f"({len(cached_coverage['without_layer'])}/{cached_coverage['n_cached_expected']} "
            "cached payloads lack the key) — the production L14 anchor PASSed in "
            "agreement/headline_metrics.json; not a kill criterion for the exploratory re-read"
        ),
    }
    reread_note = (
        f"exploratory re-read at a NON-registered read-out layer (L{layer}); the committed "
        "verdict stays the pre-registered L14 read in agreement/headline_metrics.json. Rules "
        "are byte-for-byte the registered ones (0.3 bar / 10k null / >=0.2 sibling-excluded "
        "margin / per-seed 2-of-3 disjunction)"
    )
    _write_json(
        ev / "agreement" / f"l{layer}_reread.json",
        {
            "layer": layer,
            "construction": {"layer": layer, "pos": pos, "K": bk.E2_K_PRIMARY, "variant": "base"},
            "note": reread_note,
            "expected_vs_loaded": expected_vs_loaded,
            "layer_coverage": layer_coverage,
            "null_random": {
                "p95": null95,
                "p99": null99,
                "n": int(null_cos.size),
                "note": (
                    "layer-independent by symmetry (random unit vectors at d=3584); "
                    f"recomputed against the L{layer} w_shared targets, same seed (602)"
                ),
            },
            "ceiling_seed": ceiling,
            "per_cell": rows,
            "margins": margins,
            "cross_recipe_sibling_reads": cross_recipe,
            "verdicts": verdicts,
            "h1_ladder": ladder,
            "anchor_521": anchor,
            "cross_estimator": cross_estimator,
            "same_variant_sensitivity": {
                "excluded": True,
                "reason": (
                    f"cached #551 same-variant tensors lack l{layer} keys "
                    f"(see layer_coverage.cached_same_variant) — no l{layer} realized write "
                    "exists for the cached families; excluded rather than fabricated"
                ),
            },
            "reproducibility": _meta(),
        },
    )
    loc_rows = [r for r in repair_rows if r["family"] == "loc474"]
    _write_json(
        ev / "repair" / f"repair_test_l{layer}.json",
        {
            "layer": layer,
            "thresholds": {"rho_fail": REPAIR_RHO_FAIL, "rho_pass": REPAIR_RHO_PASS},
            "behavioral_panel_caveats": BEHAVIORAL_PANEL_CAVEATS,
            "expected_vs_loaded": expected_vs_loaded,
            "layer_coverage": layer_coverage,
            "verdict_accounting": {
                "n_rows_total": len(repair_rows),
                "n_rows_verdict_denominator": len(repair_rows) - len(loc_rows),
                "excluded_marker_arm_rows": len(loc_rows),
                "note": (
                    "loc474 marker-arm rows excluded from verdict counts as "
                    "direction-insensitive (4-point panel; norm-only control matches the "
                    "realized profile exactly) — same denominator rule as the L14 read"
                ),
            },
            "repair_rows": repair_rows,
            "geometry_consistency_rows": geometry_rows,
            "note": (
                "repair VERDICTS registered on behavioral-panel families ONLY "
                "(fact541, refusal518, em518, loc474); the activation half is a "
                f"geometry-consistency read, never a repair verdict (MF1). {reread_note}"
            ),
            "reproducibility": _meta(),
        },
    )
    logger.info(
        "[phase=reread_done] L%d: %d cells, %d estimator units, %d headline rows, "
        "%d repair rows (%d in verdict denominator)",
        layer,
        len(cells),
        len(est),
        len(rows),
        len(repair_rows),
        len(repair_rows) - len(loc_rows),
    )
    return 0


# ---------------------------------------------------------------------------
# Follow-up `shuffled-replay-l27-control` (plan v3) — VM post-pod scoring
# ---------------------------------------------------------------------------
SHUFFLE_GATE_COS = 0.99  # positive-control: recomputed-vs-stored intact w_hat
SHUFFLE_GATE_ANCHOR_TOL = 0.02  # re-score tolerance vs the l27_reread anchors
SHUFFLE_RETENTION_FRAC = 0.8  # cos_shuffle >= 0.8 x cos_intact counts as retained
SHUFFLE_FAMILIES: tuple[str, ...] = ("em_turner", "em518")


def _shuffle_units() -> list[tuple[str, str]]:
    """The 7 follow-up compute units (em_turner shared + 6 em518 sources)."""
    return [("em_turner", "no_system")] + [("em518", s) for s in bk.SOURCES_518]


def _unit_vec(v: np.ndarray) -> np.ndarray:
    return v / np.linalg.norm(v)


def _safe_cos(a, b) -> float | None:
    """Cosine that records (instead of crashing on) a dim mismatch."""
    a = np.asarray(a, dtype=np.float64).ravel()
    b = np.asarray(b, dtype=np.float64).ravel()
    if a.shape != b.shape:
        return None
    return cosine(a, b)


def _e1_transform_vec(payload: dict, transform: str, layer: int, unmatched: bool = False):
    """Pull one transform's w_hat[mean_resp][layer] from a follow-up payload."""
    entry = payload["e1"]["shared"].get(transform)
    if entry is None:
        return None
    w = entry["w_hat_unmatched"] if unmatched else entry["w_hat"]
    v = w.get("mean_resp", {}).get(layer)
    return None if v is None else np.asarray(v, dtype=np.float64)


def _est_view_for_transform(
    followup: dict[tuple[str, str], dict], transform: str, unmatched: bool = False
) -> dict[tuple[str, str], dict]:
    """Synthetic est-payload view so ``_offdiag_margins`` (the parent's
    sibling-excluded-margin machinery, reused VERBATIM) resolves est_tf to
    the requested transform's w_hat. E2/E3 resolve to None and are skipped."""
    out: dict[tuple[str, str], dict] = {}
    for key, p in followup.items():
        entry = p["e1"]["shared"].get(transform)
        if entry is None:
            continue
        w_hat = entry["w_hat_unmatched"] if unmatched else entry["w_hat"]
        out[key] = {"e1": {"shared": {"w_hat": w_hat}}, "e2": {}, "e3": {"w_hat": {}}}
    return out


def _margin_retains(margin_t: float | None, margin_intact: float | None) -> bool | None:
    """Margin-retention rule, mirroring the cosine retention disjunction:
    margin_t >= 0.8 x margin_intact OR margin_t >= MARGIN_MIN (0.2)."""
    if margin_t is None or margin_intact is None:
        return None
    return margin_t >= SHUFFLE_RETENTION_FRAC * margin_intact or margin_t >= MARGIN_MIN


def _resolve_followup_payloads(
    args: argparse.Namespace, followup_est_dir: Path, strict: bool
) -> tuple[dict[tuple[str, str], dict], dict[str, Any]]:
    """Download/resolve the 7 follow-up estimator payloads (pinned handoff).

    Local files are used when present; missing files download from
    ``--followup-repo`` at ``--hf-revision`` (the upload-recorded revision
    from the dispatcher sentinel/manifest — plan v3 §3.4). Production
    requires all 7 + a recorded revision; per-file sha256 + source are
    recorded into the output JSON.
    """
    followup_est_dir.mkdir(parents=True, exist_ok=True)
    payloads: dict[tuple[str, str], dict] = {}
    files: dict[str, Any] = {}
    revisions_seen: set[str] = set()
    for family, source in _shuffle_units():
        name = f"{family}__{source}.pt"
        local = followup_est_dir / name
        sidecar = followup_est_dir / f"{family}__{source}.manifest.json"
        src_kind = "local"
        for p in (local, sidecar):
            if p.is_symlink() and not p.exists():
                p.unlink()
        if not local.exists():
            if args.hf_revision is None:
                if strict:
                    raise RuntimeError(
                        f"follow-up payload {name} missing locally and --hf-revision not given — "
                        "the VM handoff is a PINNED download (plan v3 §3.4)"
                    )
                logger.warning("follow-up payload %s missing — skipped (allow-subset)", name)
                files[name] = {"present": False}
                continue
            for fname, dest in ((name, local), (sidecar.name, sidecar)):
                got = bk.hub_download(
                    args.followup_repo,
                    f"{bk.FOLLOWUP_SHUFFLE_BUCKET}/analysis_tensors/estimator_reads/{fname}",
                    revision=args.hf_revision,
                )
                dest.symlink_to(got)
            src_kind = f"downloaded@{args.hf_revision}"
        payload = torch.load(local, map_location="cpu", weights_only=False)
        _guard_payload_model(name, payload.get("manifest", {}).get("model_id"), strict)
        missing_t = [t for t in bk.E1_TRANSFORMS if t not in payload["e1"].get("shared", {})]
        if missing_t:
            msg = f"{name}: e1.shared missing transforms {missing_t}"
            if strict:
                raise RuntimeError(msg)
            logger.warning("%s (allow-subset)", msg)
        # the dispatcher records the post-upload revision into the SIDECAR
        # manifest (the .pt is finalized before the upload phase runs)
        side = json.loads(sidecar.read_text()) if sidecar.exists() else {}
        if side.get("upload_revision"):
            revisions_seen.add(side["upload_revision"])
        payloads[(family, source)] = payload
        files[name] = {
            "present": True,
            "source": src_kind,
            "sha256": bk.sha256_file(local),
            "manifest_git_commit": payload.get("manifest", {}).get("git_commit"),
            "manifest_upload_revision": side.get("upload_revision"),
            "manifest_upload_repo": side.get("upload_repo"),
        }
    revision_recorded = args.hf_revision or (
        sorted(revisions_seen)[0] if len(revisions_seen) == 1 else None
    )
    if strict and revision_recorded is None:
        raise RuntimeError(
            "no follow-up payload revision recorded — pass --hf-revision (from the dispatcher "
            "sentinel) so the handoff is pinned"
        )
    info = {"files": files, "revision_recorded": revision_recorded, "repo": args.followup_repo}
    return payloads, info


def _load_shift_cells_pinned(shifts_dir: Path, strict: bool) -> dict[str, dict]:
    """All 31 parent shift payloads (margin comparators + the 9 targets).

    Strict mode downloads any missing payload from the parent upload at
    the pinned input revision (``bk.FOLLOWUP_SHUFFLE_INPUT_REVISION``) so
    a fresh VM checkout reproduces the exact parent inputs.
    """
    if strict:
        shifts_dir.mkdir(parents=True, exist_ok=True)
        for cell in bk.extraction_cells():
            local = shifts_dir / f"{cell['cell_id']}.pt"
            if local.is_symlink() and not local.exists():
                local.unlink()
            if not local.exists():
                got = bk.hub_download(
                    bk.DATA_REPO,
                    f"{bk.HUB_BUCKET}/analysis_tensors/shifts/{cell['cell_id']}.pt",
                    revision=bk.FOLLOWUP_SHUFFLE_INPUT_REVISION,
                )
                local.symlink_to(got)
                logger.info(
                    "[phase=shuffle_inputs] %s downloaded at pin %s",
                    local.name,
                    bk.FOLLOWUP_SHUFFLE_INPUT_REVISION[:12],
                )
    return _load_new_shift_payloads(shifts_dir, strict)


def _positive_control_gate(
    args: argparse.Namespace,
    followup: dict[tuple[str, str], dict],
    cells9: dict[str, dict],
    targets: dict[str, dict],
    parent_est_dir: Path,
    strict: bool,
) -> dict[str, Any]:
    """Plan v3 §1 gate: the same-pass intact arm must reproduce the parent.

    (a) cos(recomputed intact w_hat, stored parent e1.w_hat[mean_resp][27])
    >= 0.99 per unit; (b) per-cell re-score of BOTH targets within +-0.02
    of the recorded l27_reread.json est_tf anchors. Violation = rig bug.
    """
    anchors_path = bk.eval_dir(REPO) / "agreement" / "l27_reread.json"
    anchor_rows: dict[str, dict] = {}
    if anchors_path.exists():
        reread = json.loads(anchors_path.read_text())
        anchor_rows = {
            r["cell_id"]: r
            for r in reread["per_cell"]
            if r["estimator"] == "est_tf" and r["family"] in SHUFFLE_FAMILIES
        }
    per_unit: dict[str, Any] = {}
    for (family, source), payload in followup.items():
        name = f"{family}__{source}"
        entry: dict[str, Any] = {}
        recomputed = _e1_transform_vec(payload, "intact", bk.L27_LAYER)
        stored = None
        parent_p = parent_est_dir / f"{name}.pt"
        if parent_p.is_symlink() and not parent_p.exists():
            parent_p.unlink()
        if not parent_p.exists() and strict:
            got = bk.hub_download(
                bk.DATA_REPO,
                f"{bk.HUB_BUCKET}/analysis_tensors/estimator_reads/{name}.pt",
                revision=bk.FOLLOWUP_SHUFFLE_INPUT_REVISION,
            )
            parent_p.parent.mkdir(parents=True, exist_ok=True)
            parent_p.symlink_to(got)
        if parent_p.exists():
            parent_payload = torch.load(parent_p, map_location="cpu", weights_only=False)
            w = parent_payload["e1"]["shared"]["w_hat"].get("mean_resp", {}).get(bk.L27_LAYER)
            stored = None if w is None else np.asarray(w, dtype=np.float64)
        if recomputed is None or stored is None:
            entry["cos_recomputed_vs_stored"] = None
            entry["pass_cos"] = False
            entry["error"] = "recomputed or stored intact w_hat missing"
        else:
            c = _safe_cos(recomputed, stored)
            entry["cos_recomputed_vs_stored"] = c
            entry["pass_cos"] = c is not None and c >= SHUFFLE_GATE_COS
            if c is None:
                entry["error"] = (
                    f"dim mismatch recomputed {recomputed.shape} vs stored {stored.shape}"
                )
        per_unit[name] = entry
    per_cell: dict[str, Any] = {}
    for cell_id, cell in sorted(cells9.items()):
        anchor = anchor_rows.get(cell_id)
        w = (
            _e1_transform_vec(
                followup.get((cell["family"], cell["source"]), {"e1": {"shared": {}}}),
                "intact",
                bk.L27_LAYER,
            )
            if (cell["family"], cell["source"]) in followup
            else None
        )
        tgt = targets.get(cell_id, {}).get(f"L{bk.L27_LAYER}_mean_resp")
        if w is None or tgt is None or anchor is None:
            per_cell[cell_id] = {"pass": False, "error": "missing read/target/anchor"}
            continue
        cos_shared = _safe_cos(w, tgt["w_shared"])
        cos_src = _safe_cos(w, tgt["w_src"])
        ok = (
            cos_shared is not None
            and cos_src is not None
            and abs(cos_shared - anchor["cos_w_shared"]) <= SHUFFLE_GATE_ANCHOR_TOL
            and abs(cos_src - anchor["cos_w_src"]) <= SHUFFLE_GATE_ANCHOR_TOL
        )
        per_cell[cell_id] = {
            "recomputed_cos_w_shared": cos_shared,
            "anchor_cos_w_shared": anchor["cos_w_shared"],
            "recomputed_cos_w_src": cos_src,
            "anchor_cos_w_src": anchor["cos_w_src"],
            "tolerance": SHUFFLE_GATE_ANCHOR_TOL,
            "pass": ok,
        }
    gate_pass = (
        len(per_unit) == len(_shuffle_units())
        and all(e.get("pass_cos") for e in per_unit.values())
        and len(per_cell) == len(cells9)
        and all(e.get("pass") for e in per_cell.values())
    )
    return {
        "pass": gate_pass,
        "cos_min": SHUFFLE_GATE_COS,
        "anchor_tolerance": SHUFFLE_GATE_ANCHOR_TOL,
        "anchors_file": str(anchors_path),
        "per_unit": per_unit,
        "per_cell": per_cell,
    }


def _shuffle_score_rows(
    followup: dict[tuple[str, str], dict],
    cells9: dict[str, dict],
    targets: dict[str, dict],
    ckey: str,
    layer: int,
) -> list[dict[str, Any]]:
    """Per-cell dual-target cosines + retention + R_proj for every transform
    (incl. the ``shuffle_unmatched`` sensitivity contrast, which never gates)."""
    rows: list[dict[str, Any]] = []
    for cell_id, cell in sorted(cells9.items()):
        fp = followup[(cell["family"], cell["source"])]
        tgt = targets[cell_id][ckey]
        ws_unit = _unit_vec(np.asarray(tgt["w_shared"], dtype=np.float64))
        w_intact = _e1_transform_vec(fp, "intact", layer)
        cos_intact = None if w_intact is None else _safe_cos(w_intact, tgt["w_shared"])
        proj_intact = (
            None
            if w_intact is None or w_intact.shape != ws_unit.shape
            else float(w_intact @ ws_unit)
        )
        variants: list[tuple[str, np.ndarray | None]] = [
            ("intact", w_intact),
            ("shuffle", _e1_transform_vec(fp, "shuffle", layer)),
            ("mismatch", _e1_transform_vec(fp, "mismatch", layer)),
            ("shuffle_unmatched", _e1_transform_vec(fp, "shuffle", layer, unmatched=True)),
        ]
        for tname, w in variants:
            if w is None:
                rows.append({"cell_id": cell_id, "transform": tname, "missing": True})
                continue
            cos_shared = _safe_cos(w, tgt["w_shared"])
            cos_src = _safe_cos(w, tgt["w_src"])
            proj = float(w @ ws_unit) if w.shape == ws_unit.shape else None
            rows.append(
                {
                    "cell_id": cell_id,
                    "family": cell["family"],
                    "source": cell["source"],
                    "seed": cell["seed"],
                    "transform": tname,
                    "gates_verdict": tname in ("intact", "shuffle", "mismatch"),
                    "cos_w_shared": cos_shared,
                    "cos_w_src": cos_src,
                    "best_target_cos": (
                        None
                        if cos_shared is None or cos_src is None
                        else max(cos_shared, cos_src, key=abs)
                    ),
                    "retention_vs_intact": (
                        None if cos_shared is None or not cos_intact else cos_shared / cos_intact
                    ),
                    "r_proj": (
                        None
                        if proj is None or proj_intact is None or abs(proj_intact) < 1e-12
                        else proj / proj_intact
                    ),
                    "missing": False,
                }
            )
    return rows


def _shuffle_verdict_block(
    rows: list[dict[str, Any]],
    cells9: dict[str, dict],
    margin_retention: dict[str, Any],
    gate: dict[str, Any],
    null95: float,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], str]:
    """Family medians + collapse flags + the §1 pre-committed verdict table.

    Returns ``(fam_stats, collapse, outcomes, verdict)``. Every clause's
    evaluation ships in ``outcomes`` regardless of which one fires; the
    overall verdict resolves in the §1 table's precedence order."""

    def _median(fam: str, tname: str, key: str) -> float | None:
        vals = [
            r[key]
            for r in rows
            if not r.get("missing") and r["family"] == fam and r["transform"] == tname
            if r.get(key) is not None
        ]
        return float(np.median(vals)) if vals else None

    fam_stats: dict[str, dict[str, Any]] = {}
    for fam in SHUFFLE_FAMILIES:
        fam_stats[fam] = {
            "median_cos_intact": _median(fam, "intact", "cos_w_shared"),
            "median_cos_shuffle": _median(fam, "shuffle", "cos_w_shared"),
            "median_cos_mismatch": _median(fam, "mismatch", "cos_w_shared"),
            "median_cos_shuffle_unmatched": _median(fam, "shuffle_unmatched", "cos_w_shared"),
            "median_r_proj_shuffle": _median(fam, "shuffle", "r_proj"),
            "median_r_proj_mismatch": _median(fam, "mismatch", "r_proj"),
        }

    by_ct = {(r["cell_id"], r["transform"]): r for r in rows if not r.get("missing")}

    def _cell_retains(cell_id: str) -> bool | None:
        ci = by_ct.get((cell_id, "intact"))
        cs = by_ct.get((cell_id, "shuffle"))
        if ci is None or cs is None or ci["cos_w_shared"] is None or cs["cos_w_shared"] is None:
            return None
        return (
            cs["cos_w_shared"] >= SHUFFLE_RETENTION_FRAC * ci["cos_w_shared"]
            or cs["cos_w_shared"] >= VALIDITY_COS
        )

    em518_cells = sorted(c for c, cell in cells9.items() if cell["family"] == "em518")
    em518_retain = {c: _cell_retains(c) for c in em518_cells}
    n_em518_retain = sum(1 for v in em518_retain.values() if v)
    em518_units_corroborated = sum(
        1
        for c in em518_cells
        if em518_retain.get(c)
        and margin_retention.get(f"em518__{cells9[c]['source']}", {}).get("shuffle_margin_retains")
    )
    emturner_cells = sorted(c for c, cell in cells9.items() if cell["family"] == "em_turner")
    emturner_retain = {c: _cell_retains(c) for c in emturner_cells}

    def _fam_collapse(fam: str) -> dict[str, Any]:
        mc = fam_stats[fam]["median_cos_shuffle"]
        mr = fam_stats[fam]["median_r_proj_shuffle"]
        return {
            "cos_collapse": None if mc is None else mc < VALIDITY_COS,
            "r_proj_collapse": None if mr is None else mr < VALIDITY_COS,
            "cos_in_partial_band": None if mc is None else (null95 <= mc < VALIDITY_COS),
            "discordant": None
            if mc is None or mr is None
            else ((mc < VALIDITY_COS) != (mr < VALIDITY_COS)),
        }

    collapse = {fam: _fam_collapse(fam) for fam in SHUFFLE_FAMILIES}

    def _fam_mismatch_retains(fam: str) -> bool | None:
        mi = fam_stats[fam]["median_cos_intact"]
        mm = fam_stats[fam]["median_cos_mismatch"]
        if mi is None or mm is None:
            return None
        return mm >= SHUFFLE_RETENTION_FRAC * mi or mm >= VALIDITY_COS

    outcomes: dict[str, Any] = {}
    outcomes["rig_bug_no_read"] = {
        "fires": not gate["pass"],
        "evidence": "positive-control gate" + (" FAILED" if not gate["pass"] else " passed"),
    }
    artifact_cos_branch = n_em518_retain >= 4
    artifact_fires = artifact_cos_branch and em518_units_corroborated >= 4
    outcomes["artifact_stands_l27_rider_demoted"] = {
        "fires": bool(gate["pass"] and artifact_fires),
        "n_em518_retain": n_em518_retain,
        "n_em518_margin_corroborated": em518_units_corroborated,
        "note": (
            "em518 branch only (>=4/6 independent sources retain on cos vs w_shared AND the "
            "sibling-excluded margin corroborates); raw-cos retention without margin retention "
            "drops to the partial/indeterminate class"
        ),
        "dropped_to_partial_for_margin": bool(artifact_cos_branch and not artifact_fires),
    }
    both_collapse = all(
        collapse[f]["cos_collapse"] and collapse[f]["r_proj_collapse"] for f in SHUFFLE_FAMILIES
    )
    any_discordant = any(bool(collapse[f]["discordant"]) for f in SHUFFLE_FAMILIES)
    # §1 pinned middle region: a family median in [null95, 0.3) is "partial
    # collapse = indeterminate (a named outcome; NO VERDICT FIRES)" — it
    # blocks the substantive collapse verdicts exactly like co-primary
    # discordance does, so "ruled out" requires a FULL collapse (median
    # below the random-null p95) on both families.
    any_partial_band = any(bool(collapse[f]["cos_in_partial_band"]) for f in SHUFFLE_FAMILIES)
    outcomes["unigram_bag_artifact_ruled_out"] = {
        "fires": bool(
            gate["pass"] and both_collapse and not any_discordant and not any_partial_band
        ),
        "co_primary_discordance": any_discordant,
        "blocked_by_partial_band": bool(both_collapse and not any_discordant and any_partial_band),
        "note": (
            "fires only on FULL collapse (family median cos below the random-null p95) — a "
            "median in the §1 middle region [null95, 0.3) is partial collapse, a named "
            "indeterminate outcome where no verdict fires"
        ),
    }
    mismatch_retains_both = all(bool(_fam_mismatch_retains(f)) for f in SHUFFLE_FAMILIES)
    outcomes["rider_content_bearing"] = {
        "fires": bool(
            outcomes["unigram_bag_artifact_ruled_out"]["fires"] and mismatch_retains_both
        ),
        "mismatch_retains": {f: _fam_mismatch_retains(f) for f in SHUFFLE_FAMILIES},
        "note": (
            "stronger wording only via the combined pattern (mismatch retains while shuffle "
            "collapses); shuffle collapse alone never licenses content-bearing"
        ),
    }
    outcomes["partial_collapse_indeterminate"] = {
        "fires": bool(
            gate["pass"] and any(bool(collapse[f]["cos_in_partial_band"]) for f in SHUFFLE_FAMILIES)
        ),
        "band": [null95, VALIDITY_COS],
        "per_family": {f: collapse[f]["cos_in_partial_band"] for f in SHUFFLE_FAMILIES},
    }
    emturner_only = bool(gate["pass"] and any(emturner_retain.values()) and n_em518_retain < 4)
    outcomes["em_turner_only_retention_indeterminate"] = {
        "fires": emturner_only,
        "note": (
            "the 3 em_turner seed cells are ONE estimator read scored against 0.97-collinear "
            "targets — effective N ~= 1; cannot carry a verdict"
        ),
        "em_turner_cell_retention": emturner_retain,
    }
    if not gate["pass"]:
        verdict = "rig bug — no read"
    elif outcomes["artifact_stands_l27_rider_demoted"]["fires"]:
        verdict = "artifact explanation stands, L27 rider demoted"
    elif outcomes["rider_content_bearing"]["fires"]:
        verdict = "unigram-bag artifact ruled out + rider content-bearing"
    elif outcomes["unigram_bag_artifact_ruled_out"]["fires"]:
        verdict = "unigram-bag surface-statistics artifact ruled out"
    elif emturner_only:
        verdict = "em-turner-only retention — partial/indeterminate"
    elif outcomes["partial_collapse_indeterminate"]["fires"]:
        verdict = "partial collapse — indeterminate"
    elif any_discordant:
        verdict = "co-primary discordance (cos vs R_proj) — indeterminate"
    else:
        verdict = "indeterminate — no named outcome fired"
    return fam_stats, collapse, outcomes, verdict


def phase_shuffle_control(args: argparse.Namespace) -> int:
    """Follow-up scoring: token-integrity transforms at L27/mean_resp.

    Implements plan v3 §3.4: pinned download of the 7 follow-up payloads,
    the positive-control gate (FATAL in production — evidence written,
    then nonzero exit), per-cell dual-target cosines for the three
    transforms (matched shuffle contrast gating; unmatched as
    sensitivity), retention ratios, the R_proj co-primary, the parent's
    sibling-excluded margin recomputed per transform, the 10k null, and
    the §1 pre-committed verdict table. Output:
    ``eval_results/issue_602/shuffled-replay-l27-control/shuffle_control.json``.
    """
    ev = bk.eval_dir(REPO)
    strict = not args.allow_subset
    followup_dir = Path(args.followup_dir) if args.followup_dir else ev / bk.FOLLOWUP_SHUFFLE_SLUG
    shifts_dir = Path(args.shifts_dir) if args.shifts_dir else ev / "shifts"
    parent_est_dir = Path(args.estimator_dir) if args.estimator_dir else ev / "estimator_reads"
    out_path = followup_dir / "shuffle_control.json"
    layer, pos = bk.L27_LAYER, "mean_resp"
    ckey = f"L{layer}_{pos}"

    logger.info(
        "[phase=shuffle_inputs] mode=%s followup_dir=%s hf_revision=%s",
        "production" if strict else "allow-subset",
        followup_dir,
        args.hf_revision,
    )
    followup, payload_info = _resolve_followup_payloads(
        args, followup_dir / "estimator_reads", strict
    )
    if not followup:
        raise RuntimeError("no follow-up estimator payloads available — nothing to score")
    cells_all = _load_shift_cells_pinned(shifts_dir, strict)
    cells9 = {
        cid: c
        for cid, c in cells_all.items()
        if c["family"] in SHUFFLE_FAMILIES and (c["family"], c["source"]) in followup
    }
    if strict and len(cells9) != 9:
        raise RuntimeError(f"expected the 9 registered score cells, got {sorted(cells9)}")
    coverage = {
        "mode": "production" if strict else "allow-subset",
        "n_followup_payloads_expected": len(_shuffle_units()),
        "n_followup_payloads_loaded": len(followup),
        "n_shift_payloads_expected": len(bk.extraction_cells()),
        "n_shift_payloads_loaded": len(cells_all),
        "n_score_cells": len(cells9),
    }

    logger.info("[phase=shuffle_targets] per-cell w_src + w_shared at L%d/%s", layer, pos)
    targets = _compute_targets(cells_all, layer, pos)

    logger.info("[phase=shuffle_gate] positive-control gate (intact recompute)")
    gate = _positive_control_gate(args, followup, cells9, targets, parent_est_dir, strict)

    logger.info("[phase=shuffle_null] 10k random-unit null at L%d", layer)
    null_targets = [targets[cid][ckey]["w_shared"] for cid in sorted(cells9)]
    null_cos = bk.random_null_cosines(null_targets[:8], n=N_RANDOM_NULL // 8, seed=602)
    null95 = float(np.percentile(null_cos, 95))

    logger.info("[phase=shuffle_scores] per-cell transform reads")
    rows = _shuffle_score_rows(followup, cells9, targets, ckey, layer)

    logger.info("[phase=shuffle_margins] sibling-excluded margins per transform")
    target_dim = int(np.asarray(next(iter(targets.values()))[ckey]["w_shared"]).size)
    dim_mismatch = [
        f"{f}__{s}"
        for (f, s), p in followup.items()
        if (v := _e1_transform_vec(p, "intact", layer)) is None or v.size != target_dim
    ]
    if dim_mismatch and strict:
        raise RuntimeError(
            f"estimator/target dim mismatch for {dim_mismatch} — stub/smoke payloads at a "
            "production path (the model guard should have rejected them)"
        )
    margins_by_transform: dict[str, Any] = {}
    if dim_mismatch:
        margins_by_transform["skipped"] = (
            f"dim mismatch vs targets (d={target_dim}) for {dim_mismatch} — margins skipped "
            "(allow-subset/stub smoke only; production raises)"
        )
        logger.warning("[phase=shuffle_margins] %s", margins_by_transform["skipped"])
    else:
        for tname, unmatched in (
            ("intact", False),
            ("shuffle", False),
            ("mismatch", False),
            ("shuffle_unmatched", True),
        ):
            est_view = _est_view_for_transform(
                followup, "shuffle" if unmatched else tname, unmatched=unmatched
            )
            m, _ = _offdiag_margins(cells_all, est_view, targets, layer, pos)
            margins_by_transform[tname] = {
                k.removesuffix("__est_tf"): v for k, v in m.items() if k.endswith("__est_tf")
            }
    margin_retention: dict[str, Any] = {}
    for family, source in _shuffle_units():
        name = f"{family}__{source}"
        mi = margins_by_transform.get("intact", {}).get(name, {}).get("margin_excl_siblings")
        ms = margins_by_transform.get("shuffle", {}).get(name, {}).get("margin_excl_siblings")
        mm = margins_by_transform.get("mismatch", {}).get(name, {}).get("margin_excl_siblings")
        margin_retention[name] = {
            "margin_intact": mi,
            "margin_shuffle": ms,
            "margin_mismatch": mm,
            "shuffle_margin_retains": _margin_retains(ms, mi),
            "mismatch_margin_retains": _margin_retains(mm, mi),
        }

    logger.info("[phase=shuffle_verdicts] pre-committed verdict table (plan v3 §1)")
    fam_stats, collapse, outcomes, verdict = _shuffle_verdict_block(
        rows, cells9, margin_retention, gate, null95
    )

    payload_out = {
        "followup": bk.FOLLOWUP_SHUFFLE_SLUG,
        "construction": {
            "layer": layer,
            "pos": pos,
            "variant": "base",
            "registered_contrast": "matched (shuffled behavior - shuffled base-self)",
            "sensitivity_contrast": "unmatched (shuffled behavior - intact base-self); never gates",
        },
        "hf_revisions": {
            "input_pin": bk.FOLLOWUP_SHUFFLE_INPUT_REVISION,
            "followup_payloads": payload_info["revision_recorded"],
            "followup_repo": payload_info["repo"],
        },
        "expected_vs_loaded": coverage,
        "payload_files": payload_info["files"],
        "positive_control_gate": gate,
        "null_random": {"p95": null95, "n": int(null_cos.size), "seed": 602},
        "thresholds": {
            "collapse_bar": VALIDITY_COS,
            "retention": (
                f"cos_t >= {SHUFFLE_RETENTION_FRAC} x cos_intact OR cos_t >= {VALIDITY_COS}"
            ),
            "r_proj_collapse": VALIDITY_COS,
            "r_proj_retention": 0.8,
            "partial_band": [null95, VALIDITY_COS],
            "margin_retention": (
                f"margin_t >= {SHUFFLE_RETENTION_FRAC} x margin_intact OR margin_t >= "
                f"{MARGIN_MIN} (mirrors the cosine retention disjunction; the margin itself is "
                "the parent's sibling-excluded margin, recomputed per transform via the parent "
                "_offdiag_margins machinery)"
            ),
        },
        "per_cell": rows,
        "family_medians": fam_stats,
        "family_collapse": collapse,
        "margins_by_transform": margins_by_transform,
        "margin_retention": margin_retention,
        "outcomes": outcomes,
        "verdict": verdict,
        "reproducibility": _meta(),
    }
    _write_json(out_path, payload_out)
    if strict and not gate["pass"]:
        logger.error(
            "[phase=shuffle_gate] POSITIVE-CONTROL GATE FAILED — rig bug; evidence written to %s; "
            "exiting nonzero (plan v3 §1)",
            out_path,
        )
        return 2
    logger.info(
        "[phase=shuffle_done] verdict: %s (%d cells, %d units, gate %s)",
        verdict,
        len(cells9),
        len(followup),
        "PASS" if gate["pass"] else "FAIL(allow-subset)",
    )
    return 0


def main() -> int:
    """CLI: ``--phase repro-gate`` (Phase 0) or ``--phase score`` (Phase 2)."""
    parser = argparse.ArgumentParser(
        description="#602 Phase 0 reproduction gate + Phase 2 scoring",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--phase", choices=["repro-gate", "score", "shuffle-control"], default="score"
    )
    parser.add_argument(
        "--shifts-dir", default=None, help="Override eval_results/issue_602/shifts/"
    )
    parser.add_argument(
        "--estimator-dir", default=None, help="Override eval_results/issue_602/estimator_reads/"
    )
    parser.add_argument(
        "--hf-revision",
        default=None,
        help=(
            "shuffle-control: the upload-recorded data-repo revision (from the dispatcher "
            "sentinel/manifest) for the pinned follow-up payload download (plan v3 §3.4)"
        ),
    )
    parser.add_argument(
        "--followup-dir",
        default=None,
        help="shuffle-control: override eval_results/issue_602/shuffled-replay-l27-control/",
    )
    parser.add_argument(
        "--followup-repo",
        default=bk.DATA_REPO,
        help="shuffle-control: repo carrying the follow-up payloads (private on quota fallback)",
    )
    parser.add_argument(
        "--skip-mix-pins",
        action="store_true",
        help=(
            "Phase 0: skip downloading every E1 mix for sha-pinning (faster smoke; "
            "the full gate run pins all of them)"
        ),
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=bk.PRIMARY_LAYER,
        choices=bk.LAYERS,
        help=(
            "Phase 2 read-out layer. The pre-registered headline is "
            f"L{bk.PRIMARY_LAYER}; any other value runs the free-analysis RE-READ mode "
            "(verdict table + repair test at that layer from the stored tensors), writing "
            "agreement/l{N}_reread.json + repair/repair_test_l{N}.json and leaving the "
            "registered L14 outputs untouched."
        ),
    )
    parser.add_argument(
        "--allow-subset",
        action="store_true",
        help=(
            "Phase 2: score a payload SUBSET (deliberate §9 descope or smoke). Without "
            "this flag the default/production mode requires all 31 shift payloads, all 21 "
            "estimator payloads, anchor_521.pt and a passing production i474 cross-check, "
            "and exits nonzero on any gap or registered-gate failure. The coverage actually "
            "loaded is recorded as expected_vs_loaded in every output JSON either way."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s :: %(message)s")
    # Hub downloads need HF_TOKEN; .env is not auto-loaded under `uv run`.
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    if args.phase == "repro-gate":
        return phase0_repro_gate(args)
    if args.phase == "shuffle-control":
        return phase_shuffle_control(args)
    if args.layer != bk.PRIMARY_LAYER:
        return phase2_layer_reread(args)
    return phase2_score(args)


if __name__ == "__main__":
    raise SystemExit(main())
