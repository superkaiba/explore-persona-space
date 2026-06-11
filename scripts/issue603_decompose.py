#!/usr/bin/env python3
# Greek letters / minus signs are intentional research notation in docstrings
# (RUF002/RUF003, matches reanalyze_issue444_5way.py); main is a phased
# analysis driver, long by nature (C901).
# ruff: noqa: RUF002, C901
"""#603 Phase 2 — CPU decomposition + pre-registered statistics (off-pod, VM).

Loads the 21 per-cell shift tensors (local ``eval_results/issue_603/shifts``
or downloaded from the HF data repo via ``--from-hub``), runs
``decompose_write()`` per cell at the primary read (layer 14,
mean-over-response) plus the {L7, L21} x {mean-resp, end-slot}
sensitivity grid, computes split-half reliabilities + the disattenuation
guard A, and runs the plan §6 statistics:

- fact family: exact per-seed teacher-ordering test on CMF (predicted
  marine > historian > carpenter, i.e. low prior -> high common-mode
  fraction; p = 1/216 for 3/3 seeds, one-sided tail 16/216 for >=2/3),
  the SAME ordering test on norm (H2 expects failure), descriptive
  Spearman over the 9 cells (cluster caveat), per-seed Δ = |ρ_CMF| −
  |ρ_norm| signs, and the teacher-level joint-permutation contrast
  (6 enumerated teacher-label assignments — seeds cluster within
  teacher, never freely permuted across the 9 cells);
- refusal / EM: one-sided Spearman ρ(prior, CMF) with the EXACT
  permutation null over all 720 source-label assignments; same for
  norm; the joint-permutation contrast Δ; EM drop-villain (n=5)
  sensitivity; the refusal prior-variance gate (span > 2x pooled SE);
- pooled extension read: Stouffer Z over the families' one-sided exact
  p's (descriptive only);
- norm-floor: flagged cells (||w|| below the even/odd split-half noise
  floor) — sensitivity re-read of the CMF regression ONLY (the primary
  norm regression always keeps all cells), with exclusion counts + the
  prior-correlation of the excluded set;
- refusal implant gate (plan step 12): when
  ``eval_results/issue_603/refusal_implant_check.json`` is present and
  not in its recorded fallback, refusal sources with judged source-self
  trained refusal rate < 0.5 are EXCLUDED from the refusal regression
  set and reported under ``stats.refusal.implant_gate`` (never silent);
  when the JSON is absent or records its A3 / Batch-API fallback, the
  refusal read proceeds on the norm-floor rule alone and the output says
  so. Staging note: the check JSON is a VM-side artifact produced by
  ``scripts/issue603_refusal_implant_check.py`` and committed to git
  under ``eval_results/issue_603/`` — it is NOT pulled by ``--from-hub``
  (local read);
- guard A: trigger = reliability-vs-prior sign matches CMF-vs-prior
  sign at the cluster level; when triggered the both-must-hold rule
  binds (raw AND disattenuated orderings);
- behavioral-linkage validity panel (secondary, non-gating): per
  (adapter, bystander) signed projection on û vs the parent's measured
  leak;
- the §6 decision lattice (guard B state read from
  ``expression_strata.json`` when present, else ``pending``).

Output: ``eval_results/issue_603/decomposition_results.json`` (every
number the clean-result will quote, plus meta).

Run (VM)::

    uv run python scripts/issue603_decompose.py
    # Canonical off-pod staging: pulls the 21 cells' tensors/manifests/
    # responses AND source_priors.json from the HF chain repos first.
    uv run python scripts/issue603_decompose.py --from-hub
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="i603_decompose")

from explore_persona_space.analysis.svd_direction_constancy import spearman_rho  # noqa: E402
from explore_persona_space.analysis.write_decomposition import (  # noqa: E402
    decompose_write,
    split_half_reliability,
)

EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_603"
DATA_REPO = "superkaiba1/explore-persona-space-data"
HUB_REPO_CHAIN = (
    "superkaiba1/explore-persona-space-data",
    "superkaiba1/explore-persona-space-data-private",
    "superkaiba1/explore-persona-space-overflow",
)
HUB_PREFIX = "issue603_p3prime_write_decomposition/analysis_tensors"

# Predicted CMF ordering: LOW prior -> HIGH common-mode fraction.
FACT_TEACHERS_BY_PRIOR_ASC = (
    "marine_biologist",  # -3.4032 (lowest prior -> predicted HIGHEST CMF)
    "courthouse_architecture_historian",  # -3.2291
    "wooden_furniture_carpenter",  # -3.0030
)
FACT_ARM_DIR = {
    "marine_biologist": "arm_marine_biologist",
    "courthouse_architecture_historian": "arm_courthouse_architecture_historian",
    "wooden_furniture_carpenter": "arm_top_prior_wooden_furniture_carpenter",
}
FACT_SEEDS = (42, 137, 256)
RELIABILITY_FLOOR = 0.3

# Sensitivity grid: result-key suffix -> tensor key in the shifts entry.
READS = {
    "primary_l14_mean_resp": "delta_v_mean_resp",
    "l14_end_slot": "delta_v",
    "l7_end_slot": "delta_v_l7",
    "l21_end_slot": "delta_v_l21",
    "l7_mean_resp": "delta_v_mean_resp_l7",
    "l21_mean_resp": "delta_v_mean_resp_l21",
}
PRIMARY_READ = "primary_l14_mean_resp"


def _git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def _pull_from_hub(
    shifts_dir: Path, cell_ids: list[str], priors_target: Path | None = None
) -> None:
    """Download the per-cell artifacts — and, when ``priors_target`` is given,
    the Phase-1 ``source_priors.json`` the dispatcher uploaded — from the
    first chain repo that has them. Fail-loud when a required file is on no
    chain repo. This is the CANONICAL off-pod artifact-staging step: it
    stages everything Phase 2 consumes (``issue603_expression_strata.py``
    reuses it via its own ``--from-hub``)."""
    import shutil

    from huggingface_hub import hf_hub_download, list_repo_files

    shifts_dir.mkdir(parents=True, exist_ok=True)
    listings = {}
    for repo in HUB_REPO_CHAIN:
        try:
            listings[repo] = set(list_repo_files(repo, repo_type="dataset"))
        except Exception as e:
            logger.warning("list_repo_files(%s) failed: %s", repo, e)
    wanted: list[tuple[str, Path]] = []
    for cid in cell_ids:
        for suffix in (".pt", ".manifest.json", "_responses.json"):
            wanted.append((f"{HUB_PREFIX}/shifts/{cid}{suffix}", shifts_dir / f"{cid}{suffix}"))
    if priors_target is not None:
        # The dispatcher uploads priors at {prefix}/source_priors.json
        # (issue603_extract_dispatch.py p2) — required by the refusal/EM
        # reads; without this pull the planned off-pod Phase 2 crashes at
        # the priors load after pod termination (#603 round-1 blocker
        # `from-hub-source-priors-not-downloaded`).
        wanted.append((f"{HUB_PREFIX}/source_priors.json", priors_target))
    for path_in_repo, target in wanted:
        if target.exists():
            continue
        repo = next((r for r in HUB_REPO_CHAIN if path_in_repo in listings.get(r, ())), None)
        if repo is None:
            raise FileNotFoundError(f"{path_in_repo} not on any chain repo")
        local = hf_hub_download(repo_id=repo, filename=path_in_repo, repo_type="dataset")
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(local, target)
        logger.info("[pulled] %s from %s -> %s", path_in_repo, repo, target)


def _load_implant_gate(path: Path) -> dict:
    """Plan step-12 refusal implant-gate state (round 3).

    Returns ``{mode, reason, implant_gate_dropped, ...}``: ``mode='judged'``
    when ``refusal_implant_check.json`` carries real judged rates (then
    ``implant_gate_dropped`` lists the sources below the 0.5 gate);
    ``mode='norm-floor-only'`` when the JSON is absent or recorded its
    pre-registered A3 / Batch-API fallback (then the gate is inert and the
    refusal regression proceeds on the norm-floor rule alone, plan A3/§8).
    Either way the state is REPORTED in the stats output — never silent.
    """
    if not path.exists():
        return {
            "mode": "norm-floor-only",
            "reason": f"{path.name} absent — run scripts/issue603_refusal_implant_check.py "
            "(plan step 12); proceeding on the norm-floor rule alone (plan A3)",
            "implant_gate_dropped": [],
        }
    check = json.loads(path.read_text())
    if check.get("fallback"):
        return {
            "mode": "norm-floor-only",
            "reason": "refusal_implant_check.json recorded its fallback: "
            + str(check["fallback"].get("reason")),
            "implant_gate_dropped": [],
        }
    return {
        "mode": "judged",
        "reason": None,
        "threshold": check["meta"]["gate_threshold"],
        "judge_model": check["meta"]["judge_model"],
        "rates": {s: d["rate"] for s, d in check["per_source"].items()},
        "n_rows": {s: d["n_rows"] for s, d in check["per_source"].items()},
        "implant_gate_dropped": sorted(check["dropped_sources"]),
    }


def _load_cells() -> list[dict]:
    cells: list[dict] = []
    for family in ("fact", "refusal", "em"):
        payload = json.loads((EVAL_DIR / "inputs" / f"{family}_panel.json").read_text())
        cells.extend(payload["cells"])
    assert len(cells) == 21, f"expected 21 cells, got {len(cells)}"
    return cells


def _priors_for(cells: list[dict], priors_json: Path) -> dict[str, float]:
    """cell_id -> prior. Fact: inherited #541 teacher priors; refusal/EM: Phase-1
    source-self log-prob priors (source_priors.json)."""
    out: dict[str, float] = {}
    sp = json.loads(priors_json.read_text()) if priors_json.exists() else None
    for c in cells:
        if c["family"] == "fact":
            out[c["cell_id"]] = float(c["prior_logprob"])
        else:
            if sp is None:
                raise FileNotFoundError(
                    f"{priors_json} missing — run issue603_source_prior.py (Phase 1 step 6)"
                )
            out[c["cell_id"]] = float(
                sp["families"][c["family"]][c["source"]]["mean_logprob_per_tok"]
            )
    return out


def _prior_sems(priors_json: Path) -> dict[str, dict[str, float]]:
    if not priors_json.exists():
        return {}
    sp = json.loads(priors_json.read_text())
    return {
        fam: {s: float(d["sem"]) for s, d in fam_d.items()} for fam, fam_d in sp["families"].items()
    }


def _u_reliability(
    shifts: dict[str, dict[str, torch.Tensor]],
    source: str,
    *,
    key_per_q: str = "delta_v_mean_resp_per_q",
    n_random_splits: int = 50,
    seed: int = 42,
) -> dict[str, object]:
    """Split-half reliability of the MEAN-BYSTANDER direction û.

    For each question split, û is re-estimated from the bystanders'
    half-mean per-question shifts; r_u = cos(û_A, û_B). Questions may be
    dropped per persona (n_kept varies), so halves are taken per persona
    on its OWN per-q stack — the split is by per-q row index (even/odd +
    random), matching the #551 per-persona recipe.
    """
    bys = sorted(p for p in shifts if p != source)
    stacks = [shifts[p][key_per_q].detach().double().cpu() for p in bys]
    n_min = min(s.shape[0] for s in stacks)
    if n_min < 2:
        return {"r_even_odd": None, "r_random_mean": None, "n_min": int(n_min)}

    def _u_half(idx: torch.Tensor) -> torch.Tensor:
        m = torch.stack([s[idx[idx < s.shape[0]]].mean(dim=0) for s in stacks])
        u = m.mean(dim=0)
        return u / u.norm()

    def _cos(a: torch.Tensor, b: torch.Tensor) -> float:
        return float(torch.dot(a, b))

    even = torch.arange(0, n_min, 2)
    odd = torch.arange(1, n_min, 2)
    r_eo = _cos(_u_half(even), _u_half(odd))
    gen = torch.Generator().manual_seed(seed)
    rs = []
    half = n_min // 2
    for _ in range(n_random_splits):
        perm = torch.randperm(n_min, generator=gen)
        rs.append(_cos(_u_half(perm[:half]), _u_half(perm[half:])))
    return {
        "r_even_odd": r_eo,
        "r_random_mean": float(sum(rs) / len(rs)),
        "n_min": int(n_min),
    }


def _noise_floor(per_q: torch.Tensor) -> float:
    """Even/odd split-half noise floor on the MEAN-shift norm: ||half_A − half_B|| / 2."""
    x = per_q.detach().double().cpu()
    if x.shape[0] < 2:
        return float("nan")
    a = x[0::2].mean(dim=0)
    b = x[1::2].mean(dim=0)
    return float((a - b).norm() / 2.0)


# ──────────────────────────────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────────────────────────────


def _fact_ordering_test(values: dict[tuple[str, int], float]) -> dict:
    """Per-seed teacher-ordering test: predicted marine > historian > carpenter."""
    per_seed = {}
    k = 0
    for seed in FACT_SEEDS:
        v = [values[(t, seed)] for t in FACT_TEACHERS_BY_PRIOR_ASC]
        ordered = bool(v[0] > v[1] > v[2])
        per_seed[str(seed)] = {"values_by_prior_asc": v, "predicted_ordering": ordered}
        k += int(ordered)
    return {
        "teachers_by_prior_asc": list(FACT_TEACHERS_BY_PRIOR_ASC),
        "per_seed": per_seed,
        "k_seeds_with_predicted_ordering": k,
        "p_exact_3of3": 1.0 / 216.0,
        "p_one_sided_ge2of3": 16.0 / 216.0,
        "state": "full_3of3" if k == 3 else ("directional_2of3" if k == 2 else "null"),
    }


def _exact_perm_spearman(prior: list[float], dv: list[float]) -> dict:
    """One-sided (negative) exact permutation Spearman over ALL label assignments."""
    n = len(prior)
    assert n == len(dv) and n <= 8, n
    rho_obs = spearman_rho(prior, dv)
    perms = list(itertools.permutations(range(n)))
    null = [spearman_rho(prior, [dv[i] for i in p]) for p in perms]
    # H1 predicts NEGATIVE rho(prior, CMF): one-sided p = P(rho_perm <= rho_obs).
    p_neg = sum(1 for r in null if r <= rho_obs) / len(null)
    return {
        "rho": float(rho_obs),
        "n": n,
        "n_perms": len(perms),
        "p_one_sided_negative": float(p_neg),
    }


def _joint_perm_contrast(
    prior: list[float], cmf: list[float], norm: list[float], *, clusters: list[int] | None = None
) -> dict:
    """Δ = |ρ_CMF| − |ρ_norm| vs the joint permutation null (labels permuted once
    per draw, BOTH ρ recomputed — preserves the CMF/norm coupling).

    ``clusters``: cluster index per observation; when given, permutation acts
    on CLUSTER labels (fact: teacher-level, 6 enumerated assignments)."""
    rho_c = spearman_rho(prior, cmf)
    rho_n = spearman_rho(prior, norm)
    delta_obs = abs(rho_c) - abs(rho_n)

    if clusters is None:
        idx_perms = [list(p) for p in itertools.permutations(range(len(prior)))]
    else:
        uniq = sorted(set(clusters))
        idx_perms = []
        for cp in itertools.permutations(uniq):
            mapping = dict(zip(uniq, cp, strict=True))
            # Each observation's prior is replaced by the prior of the cluster
            # it is mapped to (cluster-level label permutation).
            cluster_prior = {c: prior[clusters.index(c)] for c in uniq}
            idx_perms.append([cluster_prior[mapping[c]] for c in clusters])
    null = []
    if clusters is None:
        for p in idx_perms:
            pp = [prior[i] for i in p]
            null.append(abs(spearman_rho(pp, cmf)) - abs(spearman_rho(pp, norm)))
    else:
        for pp in idx_perms:
            null.append(abs(spearman_rho(pp, cmf)) - abs(spearman_rho(pp, norm)))
    p = sum(1 for d in null if d >= delta_obs) / len(null)
    return {
        "rho_cmf": float(rho_c),
        "rho_norm": float(rho_n),
        "delta_abs_rho": float(delta_obs),
        "delta_positive": bool(delta_obs > 0),
        "n_perms": len(null),
        "p_one_sided": float(p),
    }


def _stouffer(p_values: list[float]) -> dict:
    """Stouffer Z-combination of one-sided p's (descriptive pooled read)."""
    from statistics import NormalDist

    nd = NormalDist()
    eps = 1e-12
    zs = [nd.inv_cdf(1.0 - min(max(p, eps), 1 - eps)) for p in p_values]
    z = sum(zs) / math.sqrt(len(zs))
    return {"z": float(z), "p_one_sided": float(1.0 - nd.cdf(z)), "n_families": len(zs)}


def _linkage_panel(
    cell: dict, shifts: dict[str, dict[str, torch.Tensor]], dec: dict
) -> dict | None:
    """Per-bystander signed projection on û vs the parent's measured leak."""
    source = cell["source"]
    bys = dec["bystander_order"]
    m = torch.stack([shifts[p]["delta_v_mean_resp"].detach().double().cpu() for p in bys])
    u = m.mean(dim=0)
    u = u / u.norm()
    proj = {p: float(torch.dot(shifts[p]["delta_v_mean_resp"].double(), u)) for p in bys}

    leak: dict[str, float] = {}
    if cell["family"] == "fact":
        agg_path = (
            PROJECT_ROOT
            / "eval_results"
            / "issue_541"
            / FACT_ARM_DIR[source]
            / "aggregate_cleaned.json"
        )
        agg = json.loads(agg_path.read_text())
        per_persona = agg["per_cell"][f"on_policy_suppression_cn_seed{cell['seed']}"]["per_persona"]
        leak = {p: float(d["leak_rate_headline"]) for p, d in per_persona.items() if p in proj}
    else:
        pc_path = (
            PROJECT_ROOT
            / "eval_results"
            / "issue_518"
            / cell["family"]
            / "_inputs"
            / "predictor_comparison.json"
        )
        pc = json.loads(pc_path.read_text())
        leak = {
            row["bystander"]: float(row["delta"])
            for row in pc["cells"]
            if row["source"] == source and row["bystander"] in proj
        }
    common = sorted(set(proj) & set(leak))
    if len(common) < 3:
        return None
    rho = spearman_rho([proj[p] for p in common], [leak[p] for p in common])
    return {
        "n_bystanders_joined": len(common),
        "spearman_rho_proj_vs_leak": float(rho),
        "projection": {p: proj[p] for p in common},
        "leak": {p: leak[p] for p in common},
    }


def main() -> int:
    """Decompose all cells + run the §6 statistics; write decomposition_results.json."""
    ap = argparse.ArgumentParser(description="#603 Phase-2 decomposition + statistics")
    ap.add_argument("--shifts-dir", default=str(EVAL_DIR / "shifts"))
    ap.add_argument("--priors-json", default=str(EVAL_DIR / "source_priors.json"))
    ap.add_argument(
        "--implant-check-json",
        default=str(EVAL_DIR / "refusal_implant_check.json"),
        help="Plan step-12 output (scripts/issue603_refusal_implant_check.py); a VM-side "
        "git-committed artifact, read locally (not pulled by --from-hub). Absent or "
        "fallback-recorded -> the refusal read proceeds on the norm-floor rule alone.",
    )
    ap.add_argument("--out", default=str(EVAL_DIR / "decomposition_results.json"))
    ap.add_argument("--from-hub", action="store_true", help="Download tensors first.")
    ap.add_argument(
        "--cells",
        default="",
        help="Comma list of cell_ids to restrict to (smoke); default all 21.",
    )
    args = ap.parse_args()

    cells = _load_cells()
    if args.cells:
        keep = {c.strip() for c in args.cells.split(",") if c.strip()}
        cells = [c for c in cells if c["cell_id"] in keep]
        assert cells, f"no cells match {keep}"
    shifts_dir = Path(args.shifts_dir)
    priors_json = Path(args.priors_json)
    need_i518_priors = any(c["family"] in ("refusal", "em") for c in cells)
    if args.from_hub:
        _pull_from_hub(
            shifts_dir,
            [c["cell_id"] for c in cells],
            priors_target=priors_json if need_i518_priors else None,
        )

    have_all_families = {c["family"] for c in cells} == {"fact", "refusal", "em"}
    priors = (
        _priors_for(cells, priors_json) if (not need_i518_priors or priors_json.exists()) else None
    )
    if priors is None:
        raise FileNotFoundError(
            f"{priors_json} missing but refusal/em cells selected — run "
            "issue603_source_prior.py first"
        )

    per_cell: dict[str, dict] = {}
    for cell in cells:
        cid = cell["cell_id"]
        pt_path = shifts_dir / f"{cid}.pt"
        payload = torch.load(pt_path, map_location="cpu", weights_only=False)
        shifts = payload["shifts"]
        reads: dict[str, dict] = {}
        for read_name, key in READS.items():
            missing = [p for p in shifts if key not in shifts[p]]
            if missing:
                logger.warning(
                    "%s: key %s missing for %d personas — read skipped", cid, key, len(missing)
                )
                continue
            dec = decompose_write(shifts, source=cell["source"], key=key)
            reads[read_name] = {
                k: dec[k]
                for k in (
                    "cmf",
                    "norm",
                    "shared_norm",
                    "residual_norm",
                    "cmf_svd",
                    "cmf_svd_unitnorm",
                    "cmf_jackknife",
                    "n_bystanders",
                    "u_vector_sha",
                )
            }
        assert PRIMARY_READ in reads, f"{cid}: primary read missing"

        src_per_q = shifts[cell["source"]]["delta_v_mean_resp_per_q"]
        rel_a = split_half_reliability(src_per_q)
        rel_u = _u_reliability(shifts, cell["source"])
        r_a = rel_a["r_random_mean"]
        r_u = rel_u["r_random_mean"]
        cmf_raw = reads[PRIMARY_READ]["cmf"]
        rel_ok = (
            r_a is not None
            and r_u is not None
            and r_a >= RELIABILITY_FLOOR
            and r_u >= RELIABILITY_FLOOR
        )
        cmf_dis = float(cmf_raw / math.sqrt(r_a * r_u)) if rel_ok and r_a > 0 and r_u > 0 else None
        floor = _noise_floor(src_per_q)

        dec_primary = decompose_write(shifts, source=cell["source"], key=READS[PRIMARY_READ])
        per_cell[cid] = {
            "family": cell["family"],
            "source": cell["source"],
            "seed": cell["seed"],
            "prior": priors[cid],
            "source_base_rate": cell.get("source_base_rate"),
            "reads": reads,
            "reliability": {
                "r_a_source_dir": rel_a,
                "r_u_mean_bystander_dir": rel_u,
                "passes_floor_0p3": bool(rel_ok),
            },
            "cmf_disattenuated": cmf_dis,
            "cmf_disattenuated_truncated_gt1": bool(cmf_dis is not None and abs(cmf_dis) > 1),
            "norm_noise_floor": floor,
            "below_noise_floor": bool(
                not math.isnan(floor) and reads[PRIMARY_READ]["norm"] < floor
            ),
            "behavioral_linkage": _linkage_panel(cell, shifts, dec_primary),
            "manifest_git_commit": payload["manifest"].get("git_commit"),
        }
        logger.info(
            "[cell %s] cmf=%.4f norm=%.3f r_a=%.3f r_u=%.3f below_floor=%s",
            cid,
            cmf_raw,
            reads[PRIMARY_READ]["norm"],
            -1 if r_a is None else r_a,
            -1 if r_u is None else r_u,
            per_cell[cid]["below_noise_floor"],
        )

    # ── family statistics (primary read) ─────────────────────────────
    stats: dict[str, dict] = {}

    fact_cells = {cid: d for cid, d in per_cell.items() if d["family"] == "fact"}
    if len(fact_cells) == 9:
        cmf_by = {
            (d["source"], d["seed"]): d["reads"][PRIMARY_READ]["cmf"] for d in fact_cells.values()
        }
        norm_by = {
            (d["source"], d["seed"]): d["reads"][PRIMARY_READ]["norm"] for d in fact_cells.values()
        }
        cmf_dis_by = {(d["source"], d["seed"]): d["cmf_disattenuated"] for d in fact_cells.values()}
        ordering_cmf = _fact_ordering_test(cmf_by)
        ordering_norm = _fact_ordering_test(norm_by)
        ordering_cmf_dis = (
            _fact_ordering_test({k: v for k, v in cmf_dis_by.items()})
            if all(v is not None for v in cmf_dis_by.values())
            else None
        )
        ordered = sorted(fact_cells.values(), key=lambda d: (d["source"], d["seed"]))
        prior9 = [d["prior"] for d in ordered]
        cmf9 = [d["reads"][PRIMARY_READ]["cmf"] for d in ordered]
        norm9 = [d["reads"][PRIMARY_READ]["norm"] for d in ordered]
        clusters = [FACT_TEACHERS_BY_PRIOR_ASC.index(d["source"]) for d in ordered]
        per_seed_delta = {}
        for seed in FACT_SEEDS:
            ds = [d for d in ordered if d["seed"] == seed]
            pr = [d["prior"] for d in ds]
            rc = spearman_rho(pr, [d["reads"][PRIMARY_READ]["cmf"] for d in ds])
            rn = spearman_rho(pr, [d["reads"][PRIMARY_READ]["norm"] for d in ds])
            per_seed_delta[str(seed)] = {
                "rho_cmf": float(rc),
                "rho_norm": float(rn),
                "delta_abs_rho_sign": int(np.sign(abs(rc) - abs(rn))),
            }
        stats["fact"] = {
            "ordering_test_cmf": ordering_cmf,
            "ordering_test_norm": ordering_norm,
            "ordering_test_cmf_disattenuated": ordering_cmf_dis,
            "descriptive_spearman_9cells": {
                "rho_prior_cmf": float(spearman_rho(prior9, cmf9)),
                "rho_prior_norm": float(spearman_rho(prior9, norm9)),
                "cluster_caveat": "seeds cluster within teacher; cluster-exact "
                "significance bounded at 1/6",
            },
            "per_seed_delta_contrast": per_seed_delta,
            "joint_perm_contrast_teacher_level": _joint_perm_contrast(
                prior9, cmf9, norm9, clusters=clusters
            ),
        }

    prior_sems = _prior_sems(priors_json)
    implant_gate = _load_implant_gate(Path(args.implant_check_json))
    extension_ps: dict[str, float] = {}
    extension_ps_norm: dict[str, float] = {}
    for family in ("refusal", "em"):
        fam_cells = sorted(
            (d for d in per_cell.values() if d["family"] == family), key=lambda d: d["source"]
        )
        if len(fam_cells) != 6:
            continue
        # Plan step-12 implant gate (refusal only): judged-weak sources are
        # excluded from the REGRESSION set; the full per-source panel values
        # stay reported above the regression. EM is never gated (implant
        # strength already evidenced by #518's trained deltas).
        gate = dict(implant_gate) if family == "refusal" else None
        gate_dropped = set(gate["implant_gate_dropped"]) if gate else set()
        reg_cells = [d for d in fam_cells if d["source"] not in gate_dropped]
        prior6 = [d["prior"] for d in reg_cells]
        cmf6 = [d["reads"][PRIMARY_READ]["cmf"] for d in reg_cells]
        norm6 = [d["reads"][PRIMARY_READ]["norm"] for d in reg_cells]
        regressable = len(reg_cells) >= 4
        fam_stats: dict = {
            "sources": [d["source"] for d in fam_cells],
            "priors": [d["prior"] for d in fam_cells],
            "cmf": [d["reads"][PRIMARY_READ]["cmf"] for d in fam_cells],
            "norm": [d["reads"][PRIMARY_READ]["norm"] for d in fam_cells],
            "regression_sources": [d["source"] for d in reg_cells],
            "spearman_cmf": _exact_perm_spearman(prior6, cmf6) if regressable else None,
            "spearman_norm": _exact_perm_spearman(prior6, norm6) if regressable else None,
            "joint_perm_contrast": (
                _joint_perm_contrast(prior6, cmf6, norm6) if regressable else None
            ),
        }
        if gate is not None:
            fam_stats["implant_gate"] = gate
            if gate_dropped:
                logger.info(
                    "[refusal] implant gate dropped %s from the regression (judged "
                    "source-self rate < %s)",
                    sorted(gate_dropped),
                    gate.get("threshold"),
                )
        # Pre-registered prior-variance gate (binding for refusal; reported
        # for EM): the regression-set log-prob priors must span > 2x their
        # pooled SE (the set is the full 6 unless the implant gate dropped).
        reg_sources = {d["source"] for d in reg_cells}
        sems = {s: v for s, v in prior_sems.get(family, {}).items() if s in reg_sources}
        if sems and len(reg_cells) >= 2:
            pooled_se = float(np.sqrt(np.mean([s**2 for s in sems.values()])))
            span = float(max(prior6) - min(prior6))
            fam_stats["prior_variance_gate"] = {
                "span": span,
                "pooled_se": pooled_se,
                "passes": bool(span > 2 * pooled_se),
            }
        if family == "em":
            keep = [d for d in fam_cells if d["source"] != "villain"]
            fam_stats["drop_villain_sensitivity"] = {
                "spearman_cmf": _exact_perm_spearman(
                    [d["prior"] for d in keep], [d["reads"][PRIMARY_READ]["cmf"] for d in keep]
                ),
                "spearman_norm": _exact_perm_spearman(
                    [d["prior"] for d in keep], [d["reads"][PRIMARY_READ]["norm"] for d in keep]
                ),
            }
        # Norm-floor sensitivity re-read of the CMF regression ONLY (the
        # primary norm regression always keeps all regression-set cells — §6;
        # implant-gate-dropped sources are already out of the set).
        kept = [d for d in reg_cells if not d["below_noise_floor"]]
        excl = [d for d in reg_cells if d["below_noise_floor"]]
        fam_stats["norm_floor_sensitivity_cmf"] = {
            "n_excluded": len(excl),
            "excluded_sources": [d["source"] for d in excl],
            "prior_correlation_of_excluded": (
                float(
                    spearman_rho(
                        prior6, [1.0 if d["below_noise_floor"] else 0.0 for d in reg_cells]
                    )
                )
                if excl
                else None
            ),
            "spearman_cmf_kept": (
                _exact_perm_spearman(
                    [d["prior"] for d in kept], [d["reads"][PRIMARY_READ]["cmf"] for d in kept]
                )
                if len(kept) >= 4
                else None
            ),
        }
        # Plan §6 family-diagnostic status (binds the pooled read below):
        # non-diagnostic when the refusal prior-variance gate fails (binding
        # for refusal; reported-only for EM), when <4 sources survive the
        # norm floor (plan §8 survival rule), or when the step-12 implant
        # gate leaves <4 regression sources (refusal only).
        n_surviving = sum(1 for d in reg_cells if not d["below_noise_floor"])
        var_gate = fam_stats.get("prior_variance_gate")
        reasons: list[str] = []
        if gate_dropped and len(reg_cells) < 4:
            reasons.append(f"only_{len(reg_cells)}_sources_pass_implant_gate_lt4")
        if family == "refusal":
            if var_gate is None:
                reasons.append("prior_variance_gate_unavailable_no_prior_sems")
            elif not var_gate["passes"]:
                reasons.append("prior_variance_gate_failed")
        if n_surviving < 4:
            reasons.append(f"only_{n_surviving}_sources_survive_norm_floor_lt4")
        fam_stats["diagnostic"] = {
            "is_diagnostic": not reasons,
            "n_sources_surviving_norm_floor": n_surviving,
            "reasons_non_diagnostic": reasons,
        }
        stats[family] = fam_stats
        if fam_stats["spearman_cmf"] is not None:
            extension_ps[family] = fam_stats["spearman_cmf"]["p_one_sided_negative"]
            extension_ps_norm[family] = fam_stats["spearman_norm"]["p_one_sided_negative"]

    fams_present = [f for f in ("refusal", "em") if f in stats]
    if len(fams_present) >= 2:
        # Plan §6: a non-diagnostic family is EXCLUDED from the pooled read —
        # it collapses to the surviving family ALONE (reduced denominator,
        # stated), or to no pooled read when both families fail. A family
        # whose regression could not be computed (implant gate left <4
        # sources) has no p and is excluded the same way.
        diag = [
            f for f in fams_present if stats[f]["diagnostic"]["is_diagnostic"] and f in extension_ps
        ]
        pooled: dict = {
            "doc": "Stouffer Z over the refusal+EM one-sided exact p's; descriptive "
            "only (families share base model, panel, questions). Families declared "
            "non-diagnostic (refusal prior-variance gate fails, <4 sources "
            "survive the norm floor, or <4 sources pass the step-12 implant "
            "gate) are excluded per plan §6; with one survivor "
            "the 'pooled' read is that family's own p (reduced denominator).",
            "families_pooled": diag,
            "family_diagnostic_status": {f: stats[f]["diagnostic"] for f in fams_present},
        }
        if len(diag) >= 2:
            pooled["cmf"] = _stouffer([extension_ps[f] for f in diag])
            pooled["norm"] = _stouffer([extension_ps_norm[f] for f in diag])
        elif len(diag) == 1:
            fam = diag[0]
            pooled["cmf"] = {"collapsed_to_family": fam, "p_one_sided": extension_ps[fam]}
            pooled["norm"] = {"collapsed_to_family": fam, "p_one_sided": extension_ps_norm[fam]}
        else:
            pooled["cmf"] = None
            pooled["norm"] = None
            pooled["note"] = "no diagnostic extension family — no pooled read"
        stats["pooled_extensions"] = pooled

    # ── guard A trigger (cluster-level sign read) ────────────────────
    guard_a = None
    if "fact" in stats:
        med_ra, med_ru, med_cmf, pri = [], [], [], []
        for t in FACT_TEACHERS_BY_PRIOR_ASC:
            ds = [d for d in fact_cells.values() if d["source"] == t]
            med_ra.append(
                float(np.median([d["reliability"]["r_a_source_dir"]["r_random_mean"] for d in ds]))
            )
            med_ru.append(
                float(
                    np.median(
                        [d["reliability"]["r_u_mean_bystander_dir"]["r_random_mean"] for d in ds]
                    )
                )
            )
            med_cmf.append(float(np.median([d["reads"][PRIMARY_READ]["cmf"] for d in ds])))
            pri.append(float(np.median([d["prior"] for d in ds])))
        sign_cmf = np.sign(spearman_rho(pri, med_cmf))
        sign_ra = np.sign(spearman_rho(pri, med_ra))
        sign_ru = np.sign(spearman_rho(pri, med_ru))
        triggered = bool(sign_cmf != 0 and (sign_ra == sign_cmf or sign_ru == sign_cmf))
        guard_a = {
            "doc": "Triggered when reliability-vs-prior matches the CMF-vs-prior sign "
            "at the teacher level (qualitative by design at 3 prior levels). When "
            "triggered the both-must-hold rule binds: raw AND disattenuated "
            "orderings must hold for H1 support.",
            "teacher_median_r_a": med_ra,
            "teacher_median_r_u": med_ru,
            "teacher_median_cmf": med_cmf,
            "sign_cmf_vs_prior": int(sign_cmf),
            "sign_r_a_vs_prior": int(sign_ra),
            "sign_r_u_vs_prior": int(sign_ru),
            "triggered": triggered,
        }
        if triggered:
            # Both-must-hold: the disattenuated ordering must hold AT LEAST AS
            # STRONGLY as the raw one (a dis read that STRENGTHENS the ordering
            # — e.g. raw 2/3 -> dis 3/3 — still satisfies "both orderings hold").
            strength = {"null": 0, "directional_2of3": 1, "full_3of3": 2}
            raw_state = stats["fact"]["ordering_test_cmf"]["state"]
            dis = stats["fact"]["ordering_test_cmf_disattenuated"]
            both_hold = bool(
                raw_state in ("full_3of3", "directional_2of3")
                and dis is not None
                and strength[dis["state"]] >= strength[raw_state]
            )
            guard_a["both_must_hold_pass"] = both_hold
        stats["guard_a"] = guard_a

    # ── decision lattice (fact primary; guard B read if available) ───
    lattice = None
    if "fact" in stats:
        strata_path = EVAL_DIR / "expression_strata.json"
        guard_b_state = "pending"
        if strata_path.exists():
            strata = json.loads(strata_path.read_text())
            guard_b_state = strata.get("guard_b_verdict", "pending")
        k = stats["fact"]["ordering_test_cmf"]["k_seeds_with_predicted_ordering"]
        norm_k = stats["fact"]["ordering_test_norm"]["k_seeds_with_predicted_ordering"]
        delta_pos = stats["fact"]["joint_perm_contrast_teacher_level"]["delta_positive"]
        guard_a_ok = (
            guard_a is None
            or (not guard_a["triggered"])
            or guard_a.get("both_must_hold_pass", False)
        )
        cmf_state = "cmf_minus" if k == 3 else ("cmf_directional" if k == 2 else "cmf_null")
        norm_state = "norm_minus" if norm_k >= 2 else "norm_null"
        if guard_b_state == "text_channel_wins" or not guard_a_ok:
            headline = "indeterminate_on_h1_artifact_channel_live"
        elif cmf_state == "cmf_minus" and norm_state == "norm_null" and delta_pos:
            headline = "p3prime_supported"
        elif cmf_state == "cmf_directional" and norm_state == "norm_null" and delta_pos:
            headline = "p3prime_directionally_supported"
        elif cmf_state == "cmf_null" and norm_state == "norm_minus":
            headline = "magnitude_only_p3_supported"
        elif cmf_state in ("cmf_minus", "cmf_directional") and norm_state == "norm_minus":
            headline = "prior_shapes_size_and_direction"
        else:
            headline = "write_geometry_not_prior_tracked"
        lattice = {
            "fact_cmf_state": cmf_state,
            "fact_norm_state": norm_state,
            "delta_contrast_positive": bool(delta_pos),
            "guard_a_ok": bool(guard_a_ok),
            "guard_b_state": guard_b_state,
            "headline_cell": headline,
            "note": "guard_b_state='pending' means issue603_expression_strata.py has "
            "not run yet; the lattice headline is provisional until it lands.",
        }
        stats["decision_lattice"] = lattice

    out = {
        "meta": {
            "issue": 603,
            "git_commit": _git_commit(),
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "primary_read": PRIMARY_READ,
            "reads": READS,
            "reliability_floor": RELIABILITY_FLOOR,
            "n_cells": len(per_cell),
            "all_families_present": have_all_families,
            "env_versions": {
                pkg: __import__("importlib.metadata", fromlist=["version"]).version(pkg)
                for pkg in ("torch", "numpy")
            },
        },
        "per_cell": per_cell,
        "stats": stats,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w") as f:
        json.dump(out, f, indent=2)
    logger.info("[wrote] %s (%d cells)", out_path, len(per_cell))
    if lattice:
        logger.info(
            "[lattice] headline=%s (guard_b=%s)", lattice["headline_cell"], lattice["guard_b_state"]
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
