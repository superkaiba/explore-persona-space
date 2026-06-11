#!/usr/bin/env python3
# ruff: noqa: RUF001, RUF002
# (greek + arrow + multiplication/minus-sign characters intentional in docstrings/labels)
"""Issue #541 Phase 0 — prior prescreen + panel/source selection + G0 gate.

Four checkpointed steps (each idempotent via output-file existence; the
launcher invokes ONE step per process so vLLM / HF engines never share a
process — the vLLM worker-teardown gotcha):

  0a  prior scoring      38 candidates × 239 teach rows, length-normalized
                         teacher-forced log P on frozen base (vLLM
                         prompt_logprobs=1; same metric as #444/#500 so the
                         new candidates are directly comparable).
                         -> phase0a_prior_scores.json
  0b-gen                 38 candidates × 150 headline-family prompts
                         (5 A-reformulation groups + framing381
                         {1,3,5,7,8,9,11} headline + {2,4,6} transparency,
                         10 prompt variants each), greedy vLLM gen.
                         -> phase0b_completions.jsonl
  0b-judge               5-way Haiku judge over 0b-gen completions (reuses
                         the #500 ``_run_5way_rejudge`` machinery; per-row
                         resume; API-only).
                         -> phase0b_judged.jsonl
  0c  persona vectors    last-input-token residual activations over the
                         A-family on-topic probes; full pairwise
                         mean-per-probe cosine matrix per layer so
                         cos-to-ANY-reference is a post-hoc lookup.
                         -> phase0c_persona_vectors.json
  0d  selection + G0     deterministic panel (24) + source (4) selection +
                         the go/no-go/descope gate (plan §7). Pure code,
                         no model call. -> prior_screen.json

Smoke mode (--smoke): candidate pool capped to 4 (2 anchors + 1 original +
1 new candidate, exercising the injection path), teach rows capped to 12,
2 prompt variants per headline group, 8 cosine probes. Same code path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

from _bootstrap import PROJECT_ROOT, bootstrap

logger = bootstrap(log_name="i541_prescreen")

import numpy as np  # noqa: E402

sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import run_experiment_444 as p  # noqa: E402
from issue444_bystander_logprob import (  # noqa: E402
    _chat_prompt,
    _resolve_persona_prompt,
    _score_pairs,
)
from issue541_personas import (  # noqa: E402
    CANDIDATE_MECHANISM,
    CANDIDATE_PERSONAS,
    ORIGINAL_15,
    assert_no_dv_contamination,
    inject_candidates,
)
from run_experiment_500 import (  # noqa: E402
    PANEL_15,
    _run_5way_rejudge,
    _seed_fact_pick_from_444,
    _verdict_category,
)

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"


# Smoke runs are namespaced under issue_541_smoke so their artifacts can NEVER
# poison the full run's skip-if-exists resume logic on the same pod.
def _eval_root_name(smoke: bool) -> str:
    return "issue_541_smoke" if smoke else "issue_541"


# Rebound in main() once --smoke / EPM_541_SMOKE is known.
PRESCREEN_DIR = PROJECT_ROOT / "eval_results" / "issue_541" / "phase0_prescreen"
TEACH_ROWS_PATH = (
    PROJECT_ROOT / "eval_results" / "issue_444" / "bystander_logprob" / "teach_rows.json"
)

# Headline-family prompt groups for the base-FP screen (plan §4.2 0b).
HEADLINE_FRAMING_IDS: tuple[int, ...] = (1, 3, 5, 7, 8, 9, 11)
FLAGGED_FRAMING_IDS: tuple[int, ...] = (2, 4, 6)
N_VARIANTS_PER_GROUP = 10  # 10 distinct greedy prompts per (family, sub) group
BASE_FP_THRESHOLD = 0.05  # Source: #500 per-framing rule, generalized per-persona

# Strata boundaries (Source: #500 measured prior distribution, plan §4.2 0d).
STRATUM_H_MIN = -3.25  # H: prior > -3.25
STRATUM_M_MIN = -3.40  # M: (-3.40, -3.25]; L: <= -3.40

# G0 thresholds (plan §7).
G0_FULL_PRIOR = -3.10
G0_SOURCE_PRIOR = -3.15
G0_DESCOPED_PRIOR = -3.20
G0_MIN_SEPARATION_NATS = 0.06

PANEL_TARGET = 24
N_NEW_PANEL_SLOTS = 9

# Persona-vector probe config (inherited operationalization, #444/#500).
VECTOR_LAYERS = (7, 14, 21, 27)
N_COS_PROBES = 40

SMOKE_POOL: tuple[str, ...] = (
    "marine_biologist",
    "local_historian",
    "courthouse_architecture_historian",
    "courthouse_docent",
)


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, default=str))
    tmp.rename(path)


def _candidate_pool(smoke: bool) -> dict[str, str | None]:
    """Persona-name -> system-prompt map for the full prescreen pool."""
    inject_candidates()
    assert tuple(PANEL_15) == ORIGINAL_15, (
        "issue541_personas.ORIGINAL_15 drifted from run_experiment_500.PANEL_15"
    )
    names = SMOKE_POOL if smoke else (*ORIGINAL_15, *CANDIDATE_PERSONAS)
    pool = {name: _resolve_persona_prompt(name) for name in names}
    if not smoke:
        assert len(pool) == 38, f"expected 38 candidates, got {len(pool)}"
    # Lint every prompt that will enter scoring (not just the new ones) —
    # original prompts predate the rule and are panel-nested regardless, but
    # a contaminated NEW prompt is a hard error.
    assert_no_dv_contamination(
        {k: v for k, v in pool.items() if k in CANDIDATE_PERSONAS and v is not None}
    )
    return pool


def _reroute_driver_paths() -> None:
    """Point the #444 driver's path globals at the prescreen subtree."""
    p.EVAL_RESULTS_DIR = PRESCREEN_DIR
    p.PHASE0_DIR = PRESCREEN_DIR / "phase0_fact_candidates"
    p.DATA_DIR = PRESCREEN_DIR / "data"
    _seed_fact_pick_from_444()


def _headline_prompt_slice(smoke: bool) -> list[dict[str, Any]]:
    """The 15-group × N-variant probe slice for the base-FP screen.

    Groups: every A_reformulation sub + framing381 subs {1,3,5,7,8,9,11}
    (headline) + {2,4,6} (transparency-only). Variants: idx < N per group
    (10 full / 2 smoke) — distinct paraphrases, so greedy decoding still
    yields response variance across rows.
    """
    _reroute_driver_paths()
    facts = p._resolve_figure_facts()
    probe_path = PRESCREEN_DIR / "probes.jsonl"
    if not probe_path.exists():
        summary = p._materialize_probe_jsonl(probe_path, facts)
        logger.info("prescreen probes: %s -> %s", summary, probe_path)
    rows = [json.loads(line) for line in probe_path.open() if line.strip()]
    n_variants = 2 if smoke else N_VARIANTS_PER_GROUP
    keep_framings = {str(f) for f in (*HEADLINE_FRAMING_IDS, *FLAGGED_FRAMING_IDS)}
    out: list[dict[str, Any]] = []
    for r in rows:
        if (r["family"] == "A_reformulation" and int(r["idx"]) < n_variants) or (
            r["family"] == "framing381"
            and str(r["sub_framing"]) in keep_framings
            and int(r["idx"]) < n_variants
        ):
            out.append(r)
    n_groups = len({(r["family"], str(r["sub_framing"])) for r in out})
    assert n_groups == 15, f"expected 15 (family, sub) groups, got {n_groups}"
    return out


# ---------------------------------------------------------------------------
# Step 0a — prior scoring
# ---------------------------------------------------------------------------
def step_0a(args: argparse.Namespace) -> Path:
    out_path = PRESCREEN_DIR / "phase0a_prior_scores.json"
    if out_path.exists():
        logger.info("0a already done -> %s; skipping", out_path)
        return out_path
    pool = _candidate_pool(args.smoke)
    rows = json.loads(TEACH_ROWS_PATH.read_text())["rows"]
    if args.smoke:
        rows = rows[:12]
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    triples: list[tuple[str, str, str]] = []
    for persona, sysp in pool.items():
        for r in rows:
            triples.append((persona, _chat_prompt(tok, sysp, r["question"]), r["completion"]))
    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    scored = _score_pairs(BASE_MODEL, [(pr, c) for _, pr, c in triples])

    per_persona: dict[str, list[float]] = {name: [] for name in pool}
    for (persona, _pr, _c), (s, n) in zip(triples, scored, strict=True):
        if n > 0 and not np.isnan(s):
            per_persona[persona].append(s / n)
    summary: dict[str, Any] = {}
    for persona, vals in per_persona.items():
        a = np.asarray(vals, dtype=float)
        summary[persona] = {
            "mean_logprob_per_tok": float(a.mean()) if a.size else float("nan"),
            "sem": float(a.std(ddof=1) / np.sqrt(a.size)) if a.size > 1 else float("nan"),
            "n_rows": int(a.size),
            "mechanism": CANDIDATE_MECHANISM.get(persona, "original"),
        }
    _write_json(
        out_path,
        {
            "_doc": (
                "Per-candidate length-norm teacher-forced log P(taught completion | "
                "persona, Q) on frozen base — identical metric to #444/#500 "
                "(issue444_bystander_logprob._score_pairs)."
            ),
            "model": BASE_MODEL,
            "n_teach_rows": len(rows),
            "smoke": args.smoke,
            "summary": summary,
            "detail": {k: v for k, v in per_persona.items()},
            "timestamp": _now_iso(),
            "reproducibility": p._build_repro_metadata(include_base_model_sha=False),
        },
    )
    logger.info("0a WROTE %s (%d candidates)", out_path, len(summary))
    return out_path


# ---------------------------------------------------------------------------
# Step 0b — base-FP screen (gen + judge as separate steps)
# ---------------------------------------------------------------------------
def step_0b_gen(args: argparse.Namespace) -> Path:
    completions_path = PRESCREEN_DIR / "phase0b_completions.jsonl"
    if completions_path.exists():
        logger.info("0b-gen already done -> %s; skipping", completions_path)
        return completions_path
    pool = _candidate_pool(args.smoke)
    slice_rows = _headline_prompt_slice(args.smoke)
    prompts: list[tuple[str | None, str]] = []
    keys: list[tuple[str, str, str, int, str]] = []
    for persona, sysp in pool.items():
        for r in slice_rows:
            prompts.append((sysp, r["probe"]))
            keys.append((persona, r["family"], str(r["sub_framing"]), int(r["idx"]), r["probe"]))
    logger.info(
        "0b-gen: %d prompts (%d candidates × %d slice)", len(prompts), len(pool), len(slice_rows)
    )
    completions = p._vllm_complete_simple(
        BASE_MODEL,
        prompts,
        temperature=p.EVAL_TEMPERATURE,
        max_new_tokens=p.EVAL_MAX_NEW_TOKENS,
        gpu_id=args.gpu_id,
        gpu_memory_utilization=0.85,
    )
    rows = [
        {
            "persona": persona,
            "family": family,
            "sub_framing": sub,
            "idx": idx,
            "probe": probe,
            "completion": completion,
        }
        for (persona, family, sub, idx, probe), completion in zip(keys, completions, strict=True)
    ]
    p._write_jsonl(completions_path, rows)
    logger.info("0b-gen WROTE %s (%d rows)", completions_path, len(rows))
    return completions_path


def step_0b_judge(args: argparse.Namespace) -> Path:
    completions_path = PRESCREEN_DIR / "phase0b_completions.jsonl"
    judged_path = PRESCREEN_DIR / "phase0b_judged.jsonl"
    info = _run_5way_rejudge(
        phase_label="prescreen_base_fp",
        completions_path=completions_path,
        judged_path=judged_path,
    )
    logger.info("0b-judge: %s", {k: v for k, v in info.items() if k != "per_cell"})
    return judged_path


def _base_fp_rates(judged_path: Path) -> dict[str, dict[str, Any]]:
    """Per-candidate pooled + per-group base ``stated_seven`` rates."""
    rows = [json.loads(line) for line in judged_path.open() if line.strip()]
    headline_subs = {str(f) for f in HEADLINE_FRAMING_IDS}
    out: dict[str, dict[str, Any]] = {}
    for persona in sorted({r["persona"] for r in rows}):
        p_rows = [r for r in rows if r["persona"] == persona]
        per_group: dict[str, dict[str, float]] = {}
        for fam, sub in sorted({(r["family"], str(r["sub_framing"])) for r in p_rows}):
            g_rows = [r for r in p_rows if r["family"] == fam and str(r["sub_framing"]) == sub]
            hits = sum(1 for r in g_rows if _verdict_category(r.get("verdict")) == "stated_seven")
            per_group[f"{fam}/{sub}"] = {
                "n": len(g_rows),
                "stated_seven_rate": hits / max(1, len(g_rows)),
            }
        # Headline-DV row set: A_reformulation + headline framings (the row set the
        # screen protects; pooled 7-framings-only also recorded for the plan's
        # literal wording).
        hl_rows = [
            r
            for r in p_rows
            if r["family"] == "A_reformulation"
            or (r["family"] == "framing381" and str(r["sub_framing"]) in headline_subs)
        ]
        fr_rows = [
            r
            for r in p_rows
            if r["family"] == "framing381" and str(r["sub_framing"]) in headline_subs
        ]
        hl_hits = sum(1 for r in hl_rows if _verdict_category(r.get("verdict")) == "stated_seven")
        fr_hits = sum(1 for r in fr_rows if _verdict_category(r.get("verdict")) == "stated_seven")
        pooled_headline = hl_hits / max(1, len(hl_rows))
        pooled_framings = fr_hits / max(1, len(fr_rows))
        out[persona] = {
            "pooled_headline_rate": pooled_headline,
            "n_headline_rows": len(hl_rows),
            "pooled_framings_only_rate": pooled_framings,
            "n_framings_rows": len(fr_rows),
            "per_group": per_group,
            "excluded": bool(pooled_headline > BASE_FP_THRESHOLD),
            "threshold": BASE_FP_THRESHOLD,
        }
    return out


# ---------------------------------------------------------------------------
# Step 0c — persona vectors (pairwise cosine matrix)
# ---------------------------------------------------------------------------
def step_0c(args: argparse.Namespace) -> Path:
    out_path = PRESCREEN_DIR / "phase0c_persona_vectors.json"
    if out_path.exists():
        logger.info("0c already done -> %s; skipping", out_path)
        return out_path
    pool = _candidate_pool(args.smoke)

    import os

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_id)
    import issue444_persona_distance_topic as pdt
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # The distance module reads PERSONA_PROMPTS by module global; point it at
    # the prescreen pool so its extraction helpers see all 38 candidates.
    pdt.PERSONA_PROMPTS = dict(pool)

    n_probes = 8 if args.smoke else N_COS_PROBES
    from eval.exp444_judge_prompts import build_reformulation_probes

    a_family = [pr for probes in build_reformulation_probes(pdt.ENTITY).values() for pr in probes]
    probes = a_family[:n_probes]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    try:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, dtype=torch.bfloat16, device_map=device
        ).eval()
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL, torch_dtype=torch.bfloat16, device_map=device
        ).eval()

    names = list(pool)
    acts = {name: pdt.last_token_acts(model, tok, name, probes, device) for name in names}

    cosine_matrix: dict[str, list[list[float]]] = {}
    for layer in VECTOR_LAYERS:
        mats = torch.stack([acts[name][layer] for name in names])  # (P, n_probes, hidden)
        assert mats.shape[0] == len(names) and mats.shape[1] == len(probes), mats.shape
        normed = torch.nn.functional.normalize(mats, dim=-1)
        # mean-over-probes of per-probe cosine (matches cosine_vs_reference).
        cos = torch.einsum("aph,bph->abp", normed, normed).mean(dim=-1)
        cosine_matrix[str(layer)] = [[float(x) for x in row] for row in cos]

    _write_json(
        out_path,
        {
            "_doc": (
                "Full pairwise mean-per-probe cosine matrix of last-input-token "
                "residual activations over the A-family on-topic probes "
                "(inherited #444/#500 operationalization). cos_to_<any reference> "
                "is a row lookup."
            ),
            "model": BASE_MODEL,
            "personas": names,
            "layers": list(VECTOR_LAYERS),
            "n_probes": len(probes),
            "smoke": args.smoke,
            "cosine_matrix": cosine_matrix,
            "timestamp": _now_iso(),
            "reproducibility": p._build_repro_metadata(include_base_model_sha=False),
        },
    )
    logger.info("0c WROTE %s (%d personas × %d probes)", out_path, len(names), len(probes))
    return out_path


# ---------------------------------------------------------------------------
# Step 0d — deterministic panel + source selection + G0 gate
# ---------------------------------------------------------------------------
def _stratum(prior: float) -> str:
    if prior > STRATUM_H_MIN:
        return "H"
    if prior > STRATUM_M_MIN:
        return "M"
    return "L"


def _pick_sources(
    screened: list[str], priors: dict[str, float], courthouse_prior: float
) -> tuple[list[str], dict[str, Any]]:
    """S-top + S-mid per the source rule (plan §4.2 0d); [] when not viable."""
    viable = [
        c
        for c in screened
        if priors[c] > G0_SOURCE_PRIOR and priors[c] >= courthouse_prior + G0_MIN_SEPARATION_NATS
    ]
    if not viable:
        return [], {"reason": "no screened candidate clears source viability"}
    s_top = max(viable, key=lambda c: priors[c])
    midpoint = (courthouse_prior + priors[s_top]) / 2.0
    mid_pool = [
        c
        for c in viable
        if c != s_top
        and abs(priors[c] - priors[s_top]) >= G0_MIN_SEPARATION_NATS
        and abs(priors[c] - courthouse_prior) >= G0_MIN_SEPARATION_NATS
    ]
    if not mid_pool:
        return [s_top], {
            "reason": "S-top viable but no S-mid clears the 0.06-nat separations",
            "midpoint": midpoint,
        }
    s_mid = min(mid_pool, key=lambda c: abs(priors[c] - midpoint))
    return [s_top, s_mid], {"midpoint": midpoint}


def _g0_branch(
    screened: list[str], priors: dict[str, float], courthouse_prior: float
) -> tuple[str, list[str], dict[str, Any]]:
    """G0 gate (plan §7): ordered, mutually exclusive branches."""
    c_above_full = [c for c in screened if priors[c] > G0_FULL_PRIOR]
    c_above_descoped = [c for c in screened if priors[c] > G0_DESCOPED_PRIOR]
    picked_new_sources, source_pick_info = _pick_sources(screened, priors, courthouse_prior)
    if len(c_above_full) >= 4 and len(picked_new_sources) >= 2:
        branch = "GO-full"
        new_sources = picked_new_sources[:2]
    elif len(c_above_descoped) >= 2:
        branch = "GO-descoped"
        new_sources = [max(c_above_descoped, key=lambda c: priors[c])]
    else:
        branch = "NO-GO"
        new_sources = []
    info = {
        "n_screened_above_full": len(c_above_full),
        "n_screened_above_descoped": len(c_above_descoped),
        "source_pick_info": source_pick_info,
    }
    return branch, new_sources, info


def _select_panel(
    originals: list[str],
    screened: list[str],
    priors: dict[str, float],
    new_sources: list[str],
) -> tuple[list[str], list[str], list[dict[str, str]]]:
    """Panel rule (plan §4.2 0d): 15 originals nested + up to 9 new picks."""
    panel = list(dict.fromkeys(originals))  # preserve insertion order, 15 in full runs
    remaining = sorted(screened, key=lambda c: -priors[c])
    new_picks: list[str] = []
    # (i) all screened with prior > -3.10, up to 6.
    for c in remaining:
        if priors[c] > G0_FULL_PRIOR and len(new_picks) < 6:
            new_picks.append(c)
    # (ii) remaining slots from (-3.25, -3.10], descending by prior.
    for c in remaining:
        if c in new_picks or len(new_picks) >= N_NEW_PANEL_SLOTS:
            continue
        if STRATUM_H_MIN < priors[c] <= G0_FULL_PRIOR:
            new_picks.append(c)
    # (iii) any still-open slots from the next-highest screened candidates.
    for c in remaining:
        if c in new_picks or len(new_picks) >= N_NEW_PANEL_SLOTS:
            continue
        new_picks.append(c)
    # Sources are panel members by design: swap-in any picked source that
    # missed the panel rule (replace the lowest-priority new pick; recorded).
    swaps: list[dict[str, str]] = []
    for s in new_sources:
        if s not in new_picks:
            dropped = new_picks.pop()
            new_picks.append(s)
            swaps.append({"swapped_in": s, "swapped_out": dropped})
    return panel + new_picks, new_picks, swaps


def step_0d(args: argparse.Namespace) -> Path:
    out_path = PRESCREEN_DIR / "prior_screen.json"
    if out_path.exists() and not args.force:
        logger.info("0d already done -> %s; skipping (--force to redo)", out_path)
        return out_path
    priors_doc = json.loads((PRESCREEN_DIR / "phase0a_prior_scores.json").read_text())
    priors: dict[str, float] = {
        name: d["mean_logprob_per_tok"] for name, d in priors_doc["summary"].items()
    }
    fp = _base_fp_rates(PRESCREEN_DIR / "phase0b_judged.jsonl")

    new_candidates = [c for c in priors if c in CANDIDATE_PERSONAS]
    originals = [c for c in priors if c not in CANDIDATE_PERSONAS]
    screened = [c for c in new_candidates if not fp.get(c, {}).get("excluded", True)]
    excluded = sorted(set(new_candidates) - set(screened))
    courthouse_prior = priors.get("courthouse_architecture_historian", float("nan"))
    marine_prior = priors.get("marine_biologist", float("nan"))

    branch, new_sources, gate_info = _g0_branch(screened, priors, courthouse_prior)
    panel, new_picks, swaps = _select_panel(originals, screened, priors, new_sources)

    sources = ["marine_biologist", "courthouse_architecture_historian"]
    arm_slugs = {
        "marine_biologist": "arm_marine_biologist",
        "courthouse_architecture_historian": "arm_courthouse_architecture_historian",
    }
    if branch == "GO-full":
        # _pick_sources returns [s_top, s_mid] with s_top = max-prior.
        s_top, s_mid = new_sources[0], new_sources[1]
        sources += [s_mid, s_top]
        arm_slugs[s_mid] = f"arm_mid_prior_{s_mid}"
        arm_slugs[s_top] = f"arm_top_prior_{s_top}"
    elif branch == "GO-descoped":
        s_best = new_sources[0]
        sources.append(s_best)
        arm_slugs[s_best] = f"arm_top_prior_{s_best}"

    strata = {name: _stratum(priors[name]) for name in panel if name in priors}
    mech_composition: dict[str, dict[str, int]] = {}
    for name in panel:
        s = strata.get(name, "?")
        m = CANDIDATE_MECHANISM.get(name, "original")
        mech_composition.setdefault(s, {}).setdefault(m, 0)
        mech_composition[s][m] += 1

    doc = {
        "_doc": (
            "Phase-0 prescreen verdicts + deterministic panel/source selection "
            "(plan §4.2 0d) + G0 gate (plan §7). Selection is pure thresholding "
            "over measured floats — no model call (reproducible + auditable)."
        ),
        "smoke": bool(args.smoke or priors_doc.get("smoke")),
        "priors": priors,
        "prior_sems": {name: d["sem"] for name, d in priors_doc["summary"].items()},
        "mechanism": {name: CANDIDATE_MECHANISM.get(name, "original") for name in priors},
        "base_fp_screen": fp,
        "screened_new_candidates": screened,
        "excluded_new_candidates": excluded,
        "gate": {
            "branch": branch,
            "thresholds": {
                "go_full_prior": G0_FULL_PRIOR,
                "source_prior": G0_SOURCE_PRIOR,
                "go_descoped_prior": G0_DESCOPED_PRIOR,
                "min_separation_nats": G0_MIN_SEPARATION_NATS,
            },
            **gate_info,
            "courthouse_arch_prior_fresh": courthouse_prior,
            "courthouse_arch_prior_500": -3.2291,
            "marine_prior_fresh": marine_prior,
        },
        "selection": {
            "panel": panel,
            "panel_size": len(panel),
            "nested_originals": originals,
            "new_picks": new_picks,
            "source_swaps": swaps,
            "sources": sources,
            "arm_slugs": arm_slugs,
            "strata": strata,
            "strata_boundaries": {"H_min": STRATUM_H_MIN, "M_min": STRATUM_M_MIN},
            "mechanism_composition_per_stratum": mech_composition,
        },
        "timestamp": _now_iso(),
        "epoch_seconds": int(time.time()),
        "reproducibility": p._build_repro_metadata(include_base_model_sha=False),
    }
    if not doc["smoke"]:
        assert len(panel) == PANEL_TARGET, (len(panel), panel)
        assert set(ORIGINAL_15) <= set(panel), "original 15 must be nested in the panel"
        assert set(sources) <= set(panel), "sources must be panel members"
    _write_json(out_path, doc)
    logger.info("0d WROTE %s — gate=%s panel=%d sources=%s", out_path, branch, len(panel), sources)
    return out_path


STEPS = {
    "0a": step_0a,
    "0b-gen": step_0b_gen,
    "0b-judge": step_0b_judge,
    "0c": step_0c,
    "0d": step_0d,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Issue #541 Phase-0 prescreen (one step per process)")
    ap.add_argument("--step", required=True, choices=sorted(STEPS), help="which Phase-0 step")
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument("--smoke", action="store_true", help="4-candidate / tiny-slice smoke mode")
    ap.add_argument(
        "--force", action="store_true", help="re-run 0d even if prior_screen.json exists"
    )
    args = ap.parse_args()
    args.smoke = bool(args.smoke or os.environ.get("EPM_541_SMOKE") == "1")
    global PRESCREEN_DIR
    PRESCREEN_DIR = PROJECT_ROOT / "eval_results" / _eval_root_name(args.smoke) / "phase0_prescreen"
    PRESCREEN_DIR.mkdir(parents=True, exist_ok=True)
    STEPS[args.step](args)


if __name__ == "__main__":
    main()
