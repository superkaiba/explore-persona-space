"""Issue #605 Phases 1/1.5/4/4.5 — base-side measurement + matched-panel selection.

MARKER family (plan 4.2/4.3):
  measure (GPU, two subprocess-isolated stages):
    gen   — vLLM base on-policy R, greedy max_new_tokens=2048, per candidate
            context x 50 ``q_test_extended_50`` probes (per-candidate JSON
            checkpoint, resume-skip).
    reads — HF base bf16: (a) corrected-slot four-float prior reads on the
            candidate's OWN base R (truncate before the first marker token);
            (b) last-prompt-token activations at layers {21, 22} for every
            candidate + the 16 #406 source conditions + the 10 #532
            instructed legacy contexts; (c) pair table (cos@L21 raw pairwise,
            Gaussian-sym-KL@L22 k=16, centered-bank cosine sensitivity,
            sim-to-nearest-negative covariate) vs the 16 sources.
  select (CPU): tercile band edges fixed from the full pair table BEFORE any
    trained-side eval; per-band matched stratum (window <= 0.06 cosine) with
    fill >= 10 distinct contexts, prior p90-p10 spread >= 6 nats, within-band
    |r(cos, prior)| <= 0.3; panel of 40 contexts; rendered-string disjointness
    vs the 16 source conditions (== the #474 negative strings) over all
    probes; one pre-registered expansion round (--include-expansion).

FACT family (plan 4.5):
  measure stages: tf (vLLM TF prior per persona over the 239 #444 teach rows,
  reusing ``issue444_bystander_logprob._score_pairs``), fp-gen (vLLM temp-0
  generations on a 12-probe subsample for the base false-positive screen),
  fp-judge (Haiku 5-way, CPU/API), acts (HF last-token activations over the
  40 #541 A-family probes at layers {21, 22}).
  select (CPU): per teacher arm, similarity = cos@L21 to that arm's teacher;
  same gate with FACT-scale spread (p90-p10 >= 0.35 nat/token), fill >= 6
  (18-persona panel), |r| <= 0.3; rendered-string disjointness vs the 3
  teacher prompts + the 4 #541 training negatives (incl. the empty no_system
  rendering); one pre-registered expansion round (--include-expansion,
  mirroring the marker mechanism — measure resume-skips already-measured
  personas, so the expansion pass measures NEW candidates only, then select
  re-runs on the union and records expansion_round=1).

Smoke = sweep with one cell (plan 4.7): ``--candidates`` subsets thread
through BOTH measure and select; no divergent code path. ``--dry-run`` stops
each stage just before model load (prompt construction + counts only).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

if os.path.isdir("/workspace"):
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
sys.path.insert(0, str(PROJECT_ROOT))  # eval/ is a top-level package

import numpy as np  # noqa: E402
from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from issue605_contexts import (  # noqa: E402
    FACT_ANCHOR_PANEL,
    FACT_CANDIDATES,
    FACT_TEACHERS,
    FACT_TRAINING_NEGATIVES,
    assert_fewshot_demos_disjoint,
    fact_expansion_candidates,
    lint_fact_candidates,
    lint_marker_candidates,
    marker_candidates,
    marker_expansion_candidates,
)

from explore_persona_space.experiments.i406_conditions import (  # noqa: E402
    CONDITIONS_BY_ID,
    MARKER_ID,
    MARKER_TEXT,
    build_prompt_for_condition,
)
from explore_persona_space.experiments.i460_data import (  # noqa: E402
    load_class_d_rewrites,
    load_q_test_extended_50,
)

logger = logging.getLogger("issue605.matched_panels")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EOS_ID = 151645
COSINE_LAYER = 21
GAUSS_KL_LAYER = 22
PCA_K = 16
# (generation length inherits issue532_predictor_stress.MAX_NEW_TOKENS=2048 —
# no local twin constant; round-1 review minor)
SOURCES_ALL = list(CONDITIONS_BY_ID)  # the 16 #406 conditions, registry order
DEFAULT_OUT = Path("eval_results/issue_605")

# Legacy #532 artifacts (reused values — plan 4.2 step "legacy values reused
# from eval_results/issue_532/ where the protocol is identical").
LEGACY_PRIOR_PATH = Path("eval_results/issue_532/logp_slot_followup/base_prior_logp.json")
LEGACY_PREDICTORS_PATH = Path("eval_results/issue_532/predictors.json")

# Marker selection gate constants (plan 4.3 / 7 gate 2 — pre-registered).
M_BAND_WIDTH = 0.06
M_FILL = 10
M_SPREAD_NATS = 6.0
M_RMAX = 0.3
M_PANEL_SIZE = 40

# Amendment `wider-marker-panel-heldout-power` (plan v3 §2.1): the wide
# selection grows the panel 40 -> ~100 at FROZEN parent band edges / windows
# and records its provenance verbatim. `fill` scales linearly with the panel
# target (25 at panel 100); realized panel >= 80 is the pre-registered floor
# (plan v3 §7 risk 1 / §8).
AMENDMENT_LABEL = "wider-marker-panel-heldout-power"
FROZEN_CONTENT_COMMIT = "79f5d5d24db5c5b539f393dd80ad6be319a5aa13"  # main commit holding the JSON
FROZEN_PRODUCING_CODE_COMMIT = "f2b292385854282b1fbf327fdb60d3fff5e45e77"  # code that produced it
WIDE_PANEL_FLOOR = 80
# Fact gate constants (plan 4.5; spread re-scaled to nat/token per critique).
F_BAND_WIDTH = 0.06
F_FILL = 6
F_SPREAD_NAT_PER_TOK = 0.35
F_RMAX = 0.3
F_PANEL_SIZE = 18
F_FP_PROBES = 12  # #541 FP-screen subsample size

# #541 fact anchors: priors reused from the committed predictors.json.
FACT_LEGACY_PREDICTORS = Path("eval_results/issue_541/predictors.json")
TEACH_ROWS_PATH = Path("eval_results/issue_444/bystander_logprob/teach_rows.json")


def _assert_marker_token(tokenizer) -> None:
    """In-process marker assert (incident #537 — wired at every entrypoint)."""
    ids = tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    assert ids == [MARKER_ID], f"MARKER_TEXT encodes to {ids}, expected [{MARKER_ID}]"
    assert tokenizer.convert_tokens_to_ids("<|im_end|>") == EOS_ID


def _repro_meta(extra: dict | None = None) -> dict:
    from issue532_predictor_stress import _reproducibility_metadata

    return _reproducibility_metadata(extra)


def _resolve_marker_candidates(spec: str, include_expansion: bool) -> dict[str, dict[str, str]]:
    """Candidate subset resolution — threads through measure AND select."""
    cands = marker_candidates()
    if include_expansion:
        cands.update(marker_expansion_candidates())
    lint_marker_candidates(cands)
    if spec != "all":
        keep = spec.split(",")
        missing = [k for k in keep if k not in cands]
        assert not missing, f"unknown marker candidates: {missing}"
        cands = {k: cands[k] for k in keep}
    return cands


def _resolve_fact_candidates(spec: str, include_expansion: bool) -> dict[str, dict[str, str]]:
    """Fact candidate-subset resolution — threads through measure AND select
    (mirrors the marker mechanism; the ONE pre-registered expansion round)."""
    cands = dict(FACT_CANDIDATES)
    if include_expansion:
        cands.update(fact_expansion_candidates())
    lint_fact_candidates(cands)
    if spec != "all":
        keep = spec.split(",")
        missing = [k for k in keep if k not in cands]
        assert not missing, f"unknown fact candidates: {missing}"
        cands = {k: cands[k] for k in keep}
    return cands


def _instructed_panel_532() -> dict[str, str]:
    from issue532_predictor_stress import _instructed_bystander_panel

    return _instructed_bystander_panel()


# ---------------------------------------------------------------------------
# MARKER measure — stage gen (vLLM)
# ---------------------------------------------------------------------------
def marker_measure_gen(out_dir: Path, cands: dict, n_probes: int, dry_run: bool) -> None:
    """Base on-policy R per candidate context (vLLM, greedy, 2048)."""
    from issue532_predictor_stress import _build_bystander_prompt, _vllm_generate_R
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    _assert_marker_token(tokenizer)
    q_test = load_q_test_extended_50()[:n_probes]
    assert_fewshot_demos_disjoint(q_test)
    class_d = load_class_d_rewrites()
    panel = {label: c["system_prompt"] for label, c in cands.items()}

    r_dir = out_dir / "panel" / "marker_measure" / "R_base"
    r_dir.mkdir(parents=True, exist_ok=True)
    pending = [lb for lb in cands if not (r_dir / f"{lb}.json").exists()]
    n_prompts = len(pending) * len(q_test)
    logger.info("[phase=p1_gen] %d candidates pending (%d prompts)", len(pending), n_prompts)
    if dry_run:
        for lb in pending[:2]:
            p = _build_bystander_prompt(lb, q_test[0], tokenizer, class_d, panel)
            assert cands[lb]["system_prompt"] in p, lb
        logger.info("[phase=p1_gen] dry-run: prompts build cleanly; stopping before vLLM load")
        return
    if not pending:
        return

    from issue532_predictor_stress import _build_vllm_engine

    llm = _build_vllm_engine(max_seq_len=4096, enable_lora=False)
    for i, lb in enumerate(pending):
        prompts = [_build_bystander_prompt(lb, q, tokenizer, class_d, panel) for q in q_test]
        R_list = _vllm_generate_R(llm, prompts, cell_label=f"P1-genR/{lb}")
        payload = {
            "schema_version": "issue605_v1",
            "phase": "p1_marker_base_R",
            "context_label": lb,
            "n_probes": len(q_test),
            "completions": {q: r for q, r in zip(q_test, R_list, strict=True)},
            "context_meta": cands[lb],
            "metadata": _repro_meta({"i": i}),
        }
        tmp = r_dir / f"{lb}.json.tmp"
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(r_dir / f"{lb}.json")
        logger.info("[phase=p1_gen] %s done (%d/%d)", lb, i + 1, len(pending))


# ---------------------------------------------------------------------------
# MARKER measure — stage reads (HF: priors + activations + pair table)
# ---------------------------------------------------------------------------
def marker_measure_reads(out_dir: Path, cands: dict, n_probes: int, dry_run: bool) -> None:
    """Corrected-slot four-float priors + L21/L22 activations + pair table."""
    from issue532_followup_logp_slot import _run_slot_batches, _slot_job, _summarize
    from issue532_predictor_stress import (
        _build_bystander_prompt,
        _extract_last_prompt_activations_hf,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    _assert_marker_token(tokenizer)
    bare_ids = tokenizer.encode("※", add_special_tokens=False)
    assert len(bare_ids) == 1, bare_ids
    bare_marker_id = bare_ids[0]
    q_test = load_q_test_extended_50()[:n_probes]
    class_d = load_class_d_rewrites()
    instructed = _instructed_panel_532()
    panel = {label: c["system_prompt"] for label, c in cands.items()}
    # Activation scope: candidates + 16 source conditions + 10 legacy
    # instructed (the latter two give the legacy-anchor drift QA + the
    # centered-bank sensitivity a complete bank).
    act_labels = list(cands) + SOURCES_ALL + sorted(instructed)
    dispatch_panel = {**panel, **instructed}

    m_dir = out_dir / "panel" / "marker_measure"
    prior_dir = m_dir / "prior"
    acts_dir = m_dir / "acts"
    prior_dir.mkdir(parents=True, exist_ok=True)
    acts_dir.mkdir(parents=True, exist_ok=True)

    pending_prior = [lb for lb in cands if not (prior_dir / f"{lb}.json").exists()]
    pending_acts = [lb for lb in act_labels if not (acts_dir / f"{lb}.npz").exists()]
    logger.info(
        "[phase=p1_reads] pending: %d priors, %d activation sets",
        len(pending_prior),
        len(pending_acts),
    )
    if dry_run:
        for lb in pending_prior[:1]:
            r_path = m_dir / "R_base" / f"{lb}.json"
            if r_path.exists():
                comp = json.loads(r_path.read_text())["completions"]
                job = _slot_job(
                    _build_bystander_prompt(lb, q_test[0], tokenizer, class_d, dispatch_panel),
                    comp[q_test[0]],
                    tokenizer,
                    bare_marker_id,
                )
                assert job["full_ids"], lb
        logger.info("[phase=p1_reads] dry-run: slot jobs build cleanly; stopping before HF load")
        return

    import torch
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.eval()

    # (a) corrected-slot graded priors on the candidate's OWN base R.
    for lb in pending_prior:
        comp = json.loads((m_dir / "R_base" / f"{lb}.json").read_text())["completions"]
        jobs = [
            _slot_job(
                _build_bystander_prompt(lb, q, tokenizer, class_d, dispatch_panel),
                comp[q],
                tokenizer,
                bare_marker_id,
            )
            for q in q_test
        ]
        reads = _run_slot_batches(model, tokenizer, jobs, bare_marker_id, label=f"P1-prior/{lb}")
        payload = {
            "schema_version": "issue605_v1",
            "phase": "p1_marker_prior",
            "context_label": lb,
            "n_probes": len(q_test),
            "per_q": reads,
            "summary": _summarize(reads),
            "context_meta": cands[lb],
            "metadata": _repro_meta(),
        }
        tmp = prior_dir / f"{lb}.json.tmp"
        tmp.write_text(json.dumps(payload, indent=1))
        tmp.replace(prior_dir / f"{lb}.json")
        logger.info(
            "[phase=p1_reads] prior %s: mean logp=%.2f",
            lb,
            payload["summary"]["mean_logp_marker"],
        )

    # (b) activations at L21/L22 (last prompt token, the #532 recipe).
    for lb in pending_acts:
        acts = _extract_last_prompt_activations_hf(
            model,
            tokenizer,
            lb,
            q_test,
            class_d,
            dispatch_panel,
            layers=[COSINE_LAYER, GAUSS_KL_LAYER],
        )
        np.savez(
            acts_dir / f"{lb}.npz",
            **{str(li): acts[li].astype(np.float32) for li in acts},
        )
        logger.info("[phase=p1_reads] acts %s done", lb)

    del model
    torch.cuda.empty_cache()

    # (c) pair table vs the 16 sources (CPU from cached acts + priors).
    _build_marker_pair_table(out_dir, cands, n_probes)


def _build_marker_pair_table(out_dir: Path, cands: dict, n_probes: int) -> None:
    """(16 sources x contexts) pair table: cos, gkl, centered cos, priors,
    sim_to_nearest_negative, legacy reuse + drift-QA columns."""
    from issue532_predictor_stress import (
        _cosine_predictor,
        _gaussian_sym_kl_in_subspace_local,
    )

    m_dir = out_dir / "panel" / "marker_measure"
    acts_dir = m_dir / "acts"
    instructed = _instructed_panel_532()
    legacy_labels = SOURCES_ALL + sorted(instructed)
    contexts = list(cands) + legacy_labels

    acts: dict[str, dict[int, np.ndarray]] = {}
    for lb in set(contexts) | set(SOURCES_ALL):
        with np.load(acts_dir / f"{lb}.npz") as z:
            acts[lb] = {int(k): z[k] for k in z.files}

    # Priors: new candidates from this run; legacy 26 reused from #532.
    priors: dict[str, float] = {}
    prior_source: dict[str, str] = {}
    for lb in cands:
        s = json.loads((m_dir / "prior" / f"{lb}.json").read_text())["summary"]
        priors[lb] = s["mean_logp_marker"]
        prior_source[lb] = "measured_605"
    legacy_prior = json.loads(LEGACY_PRIOR_PATH.read_text())["per_bystander"]
    for lb in legacy_labels:
        if lb in priors:
            continue
        priors[lb] = legacy_prior[lb]["summary"]["mean_logp_marker"]
        prior_source[lb] = "reused_532_followup"

    legacy_pred = json.loads(LEGACY_PREDICTORS_PATH.read_text())
    l_src = {s: i for i, s in enumerate(legacy_pred["sources"])}
    l_byst = {b: i for i, b in enumerate(legacy_pred["bystanders"])}

    # Centered-bank cosine sensitivity (labeled, never mixed — #536): center
    # the per-context probe-mean vectors over the full measured bank.
    bank_labels = sorted(acts)
    bank = np.stack([acts[lb][COSINE_LAYER].mean(axis=0) for lb in bank_labels])
    bank_c = bank - bank.mean(axis=0, keepdims=True)
    bank_n = bank_c / np.clip(np.linalg.norm(bank_c, axis=1, keepdims=True), 1e-12, None)
    bank_idx = {lb: i for i, lb in enumerate(bank_labels)}

    rows = []
    for s in SOURCES_ALL:
        for c in contexts:
            if c == s:
                continue  # a source's own context is never a panel candidate
            cos = _cosine_predictor(acts[s][COSINE_LAYER], acts[c][COSINE_LAYER])
            gkl = _gaussian_sym_kl_in_subspace_local(
                acts[s][GAUSS_KL_LAYER], acts[c][GAUSS_KL_LAYER], PCA_K
            )
            neg_sims = [
                _cosine_predictor(acts[n][COSINE_LAYER], acts[c][COSINE_LAYER])
                for n in SOURCES_ALL
                if n != s
            ]
            meta = cands.get(c, {})
            row = {
                "source_cid": s,
                "context_label": c,
                "cos_l21": cos,
                "gkl_l22": gkl,
                "cos_centered_bank": float(bank_n[bank_idx[s]] @ bank_n[bank_idx[c]]),
                "prior_logp": priors[c],
                "prior_source": prior_source[c],
                "sim_to_nearest_negative": float(max(neg_sims)),
                "content_class": meta.get("content_class", "legacy"),
                "affordance_class": meta.get("affordance_class", "legacy"),
                "is_legacy": c in legacy_labels,
            }
            if c in legacy_labels and s in l_src and c in l_byst:
                row["legacy_cos_532"] = legacy_pred["cosine_matrix"][l_src[s]][l_byst[c]]
                row["legacy_gkl_532"] = legacy_pred["gauss_kl_matrix"][l_src[s]][l_byst[c]]
            rows.append(row)

    payload = {
        "schema_version": "issue605_v1",
        "phase": "p1_marker_pair_table",
        "n_probes": n_probes,
        "n_contexts": len(contexts),
        "n_pairs": len(rows),
        "cosine_form": "raw pairwise (uncentered) — #532 recipe; centered-bank column is "
        "the labeled #536 sensitivity, never numerically mixed",
        "centering_provenance": {"cos_centered_bank": "global_mean", "bank": bank_labels},
        "rows": rows,
        "metadata": _repro_meta(),
    }
    out = out_dir / "panel" / "marker_pair_table.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(out)
    logger.info("[phase=p1_reads] pair table written: %s (%d pairs)", out, len(rows))


# ---------------------------------------------------------------------------
# Band/stratum selection machinery (shared by both families)
# ---------------------------------------------------------------------------
def _pearson(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 3 or np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def _stratum_stats(pairs: list[dict], lo: float, hi: float, sim_key: str, prior_key: str) -> dict:
    """Gate statistics for pairs whose similarity falls in [lo, hi]."""
    inside = [p for p in pairs if lo <= p[sim_key] <= hi]
    ctx_priors: dict[str, float] = {}
    for p in inside:
        ctx_priors[p["context_label"]] = p[prior_key]
    pri = np.array(list(ctx_priors.values()))
    sims = np.array([p[sim_key] for p in inside])
    prs = np.array([p[prior_key] for p in inside])
    return {
        "window": [float(lo), float(hi)],
        "n_pairs": len(inside),
        "n_contexts": len(ctx_priors),
        "contexts": sorted(ctx_priors),
        "prior_spread_p90_p10": (
            float(np.percentile(pri, 90) - np.percentile(pri, 10)) if len(pri) else 0.0
        ),
        "abs_r_sim_prior": abs(_pearson(sims, prs)) if len(inside) >= 3 else float("nan"),
    }


def _best_stratum(
    pairs: list[dict],
    band_lo: float,
    band_hi: float,
    width: float,
    fill: int,
    spread_min: float,
    rmax: float,
    sim_key: str,
    prior_key: str,
) -> dict:
    """Slide a width-`width` window across [band_lo, band_hi]; return the best
    matched stratum (lexicographic: feasibility, capped fill, prior spread)."""
    starts = np.arange(band_lo, max(band_hi - width, band_lo) + 1e-9, 0.005)
    if len(starts) == 0:
        starts = np.array([band_lo])
    best, best_key = None, None
    for w0 in starts:
        st = _stratum_stats(pairs, float(w0), float(w0) + width, sim_key, prior_key)
        r = st["abs_r_sim_prior"]
        feasible = (
            st["n_contexts"] >= fill
            and st["prior_spread_p90_p10"] >= spread_min
            and (np.isnan(r) or r <= rmax)
        )
        key = (int(feasible), min(st["n_contexts"], fill), st["prior_spread_p90_p10"])
        if best_key is None or key > best_key:
            best, best_key = st, key
    assert best is not None
    r = best["abs_r_sim_prior"]
    best["gate"] = {
        "fill_ok": best["n_contexts"] >= fill,
        "spread_ok": best["prior_spread_p90_p10"] >= spread_min,
        "collinearity_ok": bool(np.isnan(r) or r <= rmax),
    }
    best["gate"]["verdict"] = all(best["gate"].values())
    return best


def _prune_panel(
    panel: list[str],
    panel_size: int,
    pairs: list[dict],
    strata: dict,
    fill: int,
    prior_key: str,
    protected: set[str] = frozenset(),
) -> list[str]:
    """Prune contexts least load-bearing: fewest stratum memberships first,
    then prior closest to its band-stratum median (least spread value); never
    prune a band's stratum below the fill floor, and NEVER prune a
    ``protected`` context (the amendment passes the parent's 40 panel
    contexts here — superset invariant, plan v3 §2.1)."""
    in_stratum_count: dict[str, int] = {}
    for st in strata.values():
        for c in st["contexts"]:
            in_stratum_count[c] = in_stratum_count.get(c, 0) + 1

    def prune_key(c: str) -> tuple:
        pri = [p[prior_key] for p in pairs if p["context_label"] == c]
        meds = []
        for st in strata.values():
            if c in st["contexts"]:
                pr = [
                    p[prior_key]
                    for p in pairs
                    if p["context_label"] in st["contexts"] and p["context_label"] != c
                ]
                meds.append(abs(np.median(pri) - np.median(pr)) if pr else 0.0)
        return (in_stratum_count.get(c, 0), min(meds) if meds else 0.0)

    for c in sorted(panel, key=prune_key):
        if len(panel) <= panel_size:
            break
        if c in protected:
            continue
        trial = [x for x in panel if x != c]
        if all(len([x for x in st["contexts"] if x in trial]) >= fill for st in strata.values()):
            panel = trial
    return panel


def _top_up_panel(
    panel: list[str],
    panel_size: int,
    pairs: list[dict],
    strata: dict,
    sim_key: str,
    prior_key: str,
) -> list[str]:
    """Top up with unselected contexts ranked by how many chosen strata their
    pairs fall into, then by prior extremity (adds spread)."""
    med_prior = float(np.median([p[prior_key] for p in pairs]))
    all_ctx = sorted({p["context_label"] for p in pairs})

    def add_key(c: str) -> tuple:
        n_in = 0
        for st in strata.values():
            lo, hi = st["window"]
            if any(lo <= p[sim_key] <= hi for p in pairs if p["context_label"] == c):
                n_in += 1
        pri = np.median([p[prior_key] for p in pairs if p["context_label"] == c])
        return (n_in, abs(float(pri) - med_prior))

    panel = list(panel)
    for c in sorted((x for x in all_ctx if x not in panel), key=add_key, reverse=True):
        if len(panel) >= panel_size:
            break
        panel.append(c)
    return sorted(panel)


def _final_strata(
    pairs: list[dict],
    strata: dict,
    panel: list[str],
    fill: int,
    spread_min: float,
    rmax: float,
    sim_key: str,
    prior_key: str,
) -> dict:
    """Per-band gate verdicts restricted to the FINAL panel (shared by the
    fresh `_select_panel` and the frozen `_select_panel_frozen` paths)."""
    final = {}
    panel_pairs = [p for p in pairs if p["context_label"] in panel]
    for b, st in strata.items():
        lo, hi = st["window"]
        fst = _stratum_stats(panel_pairs, lo, hi, sim_key, prior_key)
        r = fst["abs_r_sim_prior"]
        fst["gate"] = {
            "fill_ok": fst["n_contexts"] >= fill,
            "spread_ok": fst["prior_spread_p90_p10"] >= spread_min,
            "collinearity_ok": bool(np.isnan(r) or r <= rmax),
        }
        fst["gate"]["verdict"] = all(fst["gate"].values())
        final[b] = fst
    return final


def _select_panel(
    pairs: list[dict],
    *,
    panel_size: int,
    width: float,
    fill: int,
    spread_min: float,
    rmax: float,
    sim_key: str,
    prior_key: str,
) -> dict:
    """Band edges (terciles, FIXED here) + per-band matched strata + panel."""
    sims = np.array([p[sim_key] for p in pairs])
    edges = np.quantile(sims, [1 / 3, 2 / 3])
    bands = {
        "band_lo": (float(sims.min()), float(edges[0])),
        "band_mid": (float(edges[0]), float(edges[1])),
        "band_hi": (float(edges[1]), float(sims.max())),
    }
    strata = {
        b: _best_stratum(pairs, lo, hi, width, fill, spread_min, rmax, sim_key, prior_key)
        for b, (lo, hi) in bands.items()
    }

    panel: list[str] = sorted({c for st in strata.values() for c in st["contexts"]})
    if len(panel) > panel_size:
        panel = _prune_panel(panel, panel_size, pairs, strata, fill, prior_key)
    elif len(panel) < panel_size:
        panel = _top_up_panel(panel, panel_size, pairs, strata, sim_key, prior_key)

    final_strata = _final_strata(pairs, strata, panel, fill, spread_min, rmax, sim_key, prior_key)

    return {
        "band_edges_terciles": [float(edges[0]), float(edges[1])],
        "band_ranges": bands,
        "strata": final_strata,
        "panel": sorted(panel),
        "panel_size": len(panel),
        "gate_pass": all(st["gate"]["verdict"] for st in final_strata.values()),
        "gate_constants": {
            "band_width": width,
            "fill": fill,
            "prior_spread_min": spread_min,
            "abs_r_max": rmax,
            "panel_size_target": panel_size,
        },
    }


def _select_panel_frozen(
    pairs: list[dict],
    *,
    frozen: dict,
    panel_size: int,
    fill: int,
    spread_min: float,
    rmax: float,
    sim_key: str,
    prior_key: str,
    protected: set[str],
) -> dict:
    """Amendment select-wide path (plan v3 §2.1): band edges, band ranges, and
    per-stratum windows are loaded VERBATIM from the parent selection record —
    NO quantile recompute, NO `_best_stratum` window slide — and the panel
    grows around the ``protected`` parent panel (superset invariant asserted
    by the caller). Only ``fill`` and ``panel_size`` differ from the parent
    (the ONE manipulated variable: panel size, with fill scaled linearly)."""
    edges = list(frozen["band_edges_terciles"])
    bands = {b: tuple(v) for b, v in frozen["band_ranges"].items()}
    strata = {}
    for b in bands:
        lo, hi = frozen["strata"][b]["window"]
        st = _stratum_stats(pairs, float(lo), float(hi), sim_key, prior_key)
        r = st["abs_r_sim_prior"]
        st["gate"] = {
            "fill_ok": st["n_contexts"] >= fill,
            "spread_ok": st["prior_spread_p90_p10"] >= spread_min,
            "collinearity_ok": bool(np.isnan(r) or r <= rmax),
        }
        st["gate"]["verdict"] = all(st["gate"].values())
        strata[b] = st

    panel: list[str] = sorted({c for st in strata.values() for c in st["contexts"]} | protected)
    if len(panel) > panel_size:
        panel = _prune_panel(panel, panel_size, pairs, strata, fill, prior_key, protected=protected)
    elif len(panel) < panel_size:
        panel = _top_up_panel(panel, panel_size, pairs, strata, sim_key, prior_key)

    final_strata = _final_strata(pairs, strata, panel, fill, spread_min, rmax, sim_key, prior_key)

    return {
        "band_edges_terciles": [float(edges[0]), float(edges[1])],
        "band_ranges": {b: [float(lo), float(hi)] for b, (lo, hi) in bands.items()},
        "strata": final_strata,
        "panel": sorted(panel),
        "panel_size": len(panel),
        "gate_pass": all(st["gate"]["verdict"] for st in final_strata.values()),
        "gate_constants": {
            "band_width": M_BAND_WIDTH,
            "fill": fill,
            "prior_spread_min": spread_min,
            "abs_r_max": rmax,
            "panel_size_target": panel_size,
            "windows_frozen": True,
        },
    }


def _descope_record(sel: dict) -> dict | None:
    """Pre-registered descope-to-populated-bands record (plan section 3
    structural alternative): surviving bands = final strata whose gate verdict
    passed; the descoped panel is their stratum contexts (already a subset of
    the final panel). Returns None when NO band survives — descope cannot
    rescue a selection with zero populated bands."""
    surviving = sorted(b for b, st in sel["strata"].items() if st["gate"]["verdict"])
    if not surviving:
        return None
    ctxs = sorted({c for b in surviving for c in sel["strata"][b]["contexts"]})
    return {
        "active": True,
        "surviving_bands": surviving,
        "panel_descoped": ctxs,
        "note": "pre-registered descope to populated bands (plan section 3 structural "
        "alternative); downstream eval + analysis restrict to panel_descoped",
    }


def _enforce_selection_gate(payload: dict, allow_descope: bool, what: str, out: Path) -> None:
    """Plan section 7 gate 2: a failed fill/tightness gate BLOCKS trained-side
    GPU spend. The selection JSON is always written (inspectable artifact);
    on gate failure this exits non-zero unless --allow-descope recorded a
    valid descope (round-1 blocker ``panel-gate-not-enforced``)."""
    if payload["gate_pass"]:
        return
    desc = payload.get("descope")
    if desc and desc.get("active"):
        logger.warning(
            "[select-gate] %s gate FAIL — descope recorded (bands %s, %d contexts); "
            "eval dispatchers will restrict to panel_descoped",
            what,
            desc["surviving_bands"],
            len(desc["panel_descoped"]),
        )
        return
    raise SystemExit(
        f"SELECTION GATE FAIL ({what}): gate_pass=false in {out} — this gate BLOCKS "
        "trained-side GPU spend (plan section 7 gate 2). Run the ONE pre-registered "
        "expansion round (--include-expansion after measuring expansion candidates); "
        "if it still fails, re-run select with --allow-descope to record the "
        "pre-registered descope-to-populated-bands path"
        + ("" if allow_descope else " (no --allow-descope was passed)")
        + (
            "; NOTE: --allow-descope cannot rescue this selection — ZERO bands survive"
            if allow_descope
            else ""
        )
    )


# ---------------------------------------------------------------------------
# MARKER select (Phase 1.5, CPU)
# ---------------------------------------------------------------------------
def _prior_summary(vals: list[float]) -> dict:
    """Distribution summary for the realized-prior report (plan v3 §7 risk 3)."""
    a = np.array(vals, dtype=float)
    if a.size == 0:
        return {"n": 0}
    return {
        "n": int(a.size),
        "mean": float(a.mean()),
        "median": float(np.median(a)),
        "p10": float(np.percentile(a, 10)),
        "p25": float(np.percentile(a, 25)),
        "p75": float(np.percentile(a, 75)),
        "p90": float(np.percentile(a, 90)),
        "min": float(a.min()),
        "max": float(a.max()),
    }


def _wide_prior_qa_figure(ctx_meta: dict, inherited: list[str], new: list[str]) -> None:
    """QA figure (plan v3 §7 risk 3): realized prior distribution of the wide
    panel, inherited parent contexts vs newly admitted ones."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import savefig_paper, set_paper_style

    set_paper_style()
    fig_dir = Path("figures/issue_605") / AMENDMENT_LABEL
    fig_dir.mkdir(parents=True, exist_ok=True)
    pri_inh = [ctx_meta[c]["prior_logp"] for c in inherited]
    pri_new = [ctx_meta[c]["prior_logp"] for c in new]
    all_pri = pri_inh + pri_new
    bins = np.linspace(min(all_pri), max(all_pri), 24)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.hist(pri_inh, bins=bins, alpha=0.55, label=f"inherited parent panel (n={len(pri_inh)})")
    ax.hist(pri_new, bins=bins, alpha=0.55, label=f"newly admitted (n={len(pri_new)})")
    ax.set_xlabel("base log P(marker) at response end (graded prior, nats)")
    ax.set_ylabel("contexts")
    ax.legend(frameon=False, fontsize=8)
    savefig_paper(fig, "panel_prior_inherited_vs_new", dir=fig_dir)
    plt.close(fig)
    logger.info("[phase=p15_select] prior-distribution QA figure written under %s", fig_dir)


def marker_select(
    out_dir: Path,
    cands: dict,
    n_probes: int,
    allow_descope: bool = False,
    include_expansion: bool = False,
    panel_size: int = M_PANEL_SIZE,
    frozen_selection: Path | None = None,
) -> None:
    """Panel selection + tightness gate + rendered-string disjointness.

    With ``frozen_selection`` (the amendment select-wide path, plan v3 §2.1)
    the parent record's band edges / ranges / per-stratum windows are reused
    VERBATIM, the parent panel is protected through pruning (superset
    invariant asserted), ``fill`` scales linearly with ``panel_size``, and the
    output goes to a NEW file under the followup-label dir — the parent
    selection JSON is never overwritten."""
    from issue532_predictor_stress import _build_bystander_prompt
    from transformers import AutoTokenizer

    table = json.loads((out_dir / "panel" / "marker_pair_table.json").read_text())
    instructed = _instructed_panel_532()
    in_scope = set(cands) | set(instructed)  # source conditions never panel-eligible
    pairs = [r for r in table["rows"] if r["context_label"] in in_scope]
    assert pairs, "pair table has no in-scope rows — run --phase measure first"

    frozen: dict | None = None
    if frozen_selection is not None:
        frozen = json.loads(frozen_selection.read_text())
        assert frozen.get("gate_pass"), (
            f"frozen selection {frozen_selection} has gate_pass=false — the amendment freezes "
            "a PASSING parent record only (plan v3 §2.1)"
        )
        protected = set(frozen["panel"])
        missing_prot = sorted(protected - {p["context_label"] for p in pairs})
        assert not missing_prot, (
            f"parent panel contexts absent from the in-scope pair table: {missing_prot}"
        )
        fill = round(M_FILL * panel_size / M_PANEL_SIZE)  # linear scale: 25 at panel 100
        sel = _select_panel_frozen(
            pairs,
            frozen=frozen,
            panel_size=panel_size,
            fill=fill,
            spread_min=M_SPREAD_NATS,
            rmax=M_RMAX,
            sim_key="cos_l21",
            prior_key="prior_logp",
            protected=protected,
        )
        # Frozen values must round-trip byte-equal (plan v3 §2.1 'carrying the
        # frozen values verbatim').
        for key in ("band_edges_terciles", "band_ranges"):
            assert json.dumps(sel[key], sort_keys=True) == json.dumps(
                frozen[key], sort_keys=True
            ), f"frozen value drifted on round-trip: {key}"
        for b, st in sel["strata"].items():
            assert st["window"] == frozen["strata"][b]["window"], (
                f"frozen window drifted on round-trip: {b}"
            )
        # SUPERSET INVARIANT (plan v3 §2.1).
        dropped = sorted(protected - set(sel["panel"]))
        assert not dropped, f"superset invariant violated: parent contexts pruned: {dropped}"
        assert len(sel["panel"]) >= WIDE_PANEL_FLOOR, (
            f"realized wide panel {len(sel['panel'])} < pre-registered floor "
            f"{WIDE_PANEL_FLOOR} (plan v3 §7 risk 1) — epm:failure (data), not a silent descope"
        )
    else:
        sel = _select_panel(
            pairs,
            panel_size=panel_size,
            width=M_BAND_WIDTH,
            fill=M_FILL,
            spread_min=M_SPREAD_NATS,
            rmax=M_RMAX,
            sim_key="cos_l21",
            prior_key="prior_logp",
        )

    # Disjointness invariant (plan 4.3): every selected context's RENDERED
    # prompt differs from every #406 source condition's rendered prompt (the
    # #474 negatives ARE the other 15 conditions) for every probe q. Source
    # renders are hoisted out of the per-context loop (pure perf — identical
    # comparisons; the wide panel makes the inline re-render ~80k calls).
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    _assert_marker_token(tokenizer)
    q_test = load_q_test_extended_50()[:n_probes]
    class_d = load_class_d_rewrites()
    panel_prompts = {lb: c["system_prompt"] for lb, c in cands.items()}
    dispatch_panel = {**panel_prompts, **instructed}
    source_rendered = {
        (s, q): build_prompt_for_condition(
            CONDITIONS_BY_ID[s], q, tokenizer, class_d_rewrites=class_d
        )
        for s in SOURCES_ALL
        for q in q_test
    }
    for lb in sel["panel"]:
        for q in q_test:
            rendered = _build_bystander_prompt(lb, q, tokenizer, class_d, dispatch_panel)
            for s in SOURCES_ALL:
                assert rendered != source_rendered[(s, q)], (
                    f"disjointness violation: panel context {lb} renders identically to "
                    f"source condition {s} on probe {q[:50]!r}"
                )

    ctx_meta = {}
    for lb in sel["panel"]:
        rows = [r for r in pairs if r["context_label"] == lb]
        ctx_meta[lb] = {
            "prior_logp": rows[0]["prior_logp"],
            "prior_source": rows[0]["prior_source"],
            "content_class": rows[0]["content_class"],
            "affordance_class": rows[0]["affordance_class"],
            "is_legacy": rows[0]["is_legacy"],
        }
    payload = {
        "schema_version": "issue605_v1",
        "phase": "p15_marker_panel_selection",
        **sel,
        "context_meta": ctx_meta,
        "disjointness_assert": "PASS (rendered-string vs 16 source conditions x all probes)",
        "n_candidates_measured": table["n_contexts"],
        "expansion_round": 1 if include_expansion else 0,
        "metadata": _repro_meta(),
    }
    if frozen is not None:
        panel_inherited = sorted(set(frozen["panel"]))
        panel_new = sorted(set(sel["panel"]) - set(frozen["panel"]))
        payload["phase"] = "p15_marker_panel_selection_wide"
        payload["amendment_label"] = AMENDMENT_LABEL
        payload["frozen_from"] = {
            "path": str(frozen_selection),
            "content_commit": FROZEN_CONTENT_COMMIT,
            "producing_code_commit": FROZEN_PRODUCING_CODE_COMMIT,
        }
        payload["panel_inherited"] = panel_inherited
        payload["panel_new"] = panel_new
        payload["n_panel_inherited"] = len(panel_inherited)
        payload["n_panel_new"] = len(panel_new)
        payload["realized_prior_distribution"] = {
            "inherited": _prior_summary([ctx_meta[c]["prior_logp"] for c in panel_inherited]),
            "new": _prior_summary([ctx_meta[c]["prior_logp"] for c in panel_new]),
            "combined": _prior_summary([ctx_meta[c]["prior_logp"] for c in sel["panel"]]),
        }
        out = out_dir / AMENDMENT_LABEL / "marker_panel_selection_wide.json"
    else:
        out = out_dir / "panel" / "marker_panel_selection.json"
    if not payload["gate_pass"] and allow_descope:
        desc = _descope_record(sel)
        if desc is not None:
            payload["descope"] = desc
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(out)
    logger.info(
        "[phase=p15_select] marker panel: %d contexts, gate_pass=%s -> %s",
        payload["panel_size"],
        payload["gate_pass"],
        out,
    )
    if frozen is not None:
        _wide_prior_qa_figure(ctx_meta, payload["panel_inherited"], payload["panel_new"])
    _enforce_selection_gate(
        payload,
        allow_descope,
        "marker amendment select-wide" if frozen is not None else "marker Phase 1.5",
        out,
    )


# ---------------------------------------------------------------------------
# FACT measure stages
# ---------------------------------------------------------------------------
def _fact_persona_prompts(cands: dict) -> dict[str, str | None]:
    """name -> system prompt for candidates + the 24 #541 anchors (+ teachers)."""
    import issue444_persona_distance_topic as pdt
    from issue541_personas import inject_candidates

    inject_candidates()
    pool: dict[str, str | None] = {}
    for name in FACT_ANCHOR_PANEL:
        pool[name] = pdt._resolve_persona_prompt(name)
    for label, c in cands.items():
        assert label not in pool, f"fact candidate label collides with anchor: {label}"
        pool[label] = c["system_prompt"]
    return pool


def fact_measure_tf(out_dir: Path, cands: dict, dry_run: bool) -> None:
    """Length-normalized TF base log P(taught completion) per persona over the
    239 #444 teach rows (#541's exact metric via issue444_bystander_logprob)."""
    from issue444_bystander_logprob import _chat_prompt
    from transformers import AutoTokenizer

    rows = json.loads(TEACH_ROWS_PATH.read_text())["rows"]
    assert len(rows) == 239, len(rows)
    pool = _fact_persona_prompts(cands)
    # Anchors' priors are reused from #541 (identical metric) — only measure
    # personas without a committed prior.
    legacy = json.loads(FACT_LEGACY_PREDICTORS.read_text())["logprob_priors_used"]
    out_path = out_dir / "panel" / "fact_measure" / "tf_priors.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if out_path.exists():
        done = json.loads(out_path.read_text())["per_persona"]
    pending = [p for p in pool if p not in done and p not in legacy]
    logger.info("[phase=p4_tf] %d personas pending TF prior", len(pending))
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    triples = [
        (p, _chat_prompt(tok, pool[p], r["question"]), r["completion"])
        for p in pending
        for r in rows
    ]
    if dry_run:
        logger.info("[phase=p4_tf] dry-run: %d TF pairs built; stopping before vLLM", len(triples))
        return
    if pending:
        from issue444_bystander_logprob import _score_pairs

        scored = _score_pairs(BASE_MODEL, [(pr, c) for _, pr, c in triples])
        per: dict[str, list[float]] = {p: [] for p in pending}
        for (p, _pr, _c), (s, n) in zip(triples, scored, strict=True):
            if n > 0 and not np.isnan(s):
                per[p].append(s / n)
        for p, vals in per.items():
            a = np.array(vals)
            assert a.size >= 0.9 * len(rows), (p, a.size, "too many NaN TF rows")
            done[p] = {
                "mean_logprob_per_tok": float(a.mean()),
                "sem": float(a.std(ddof=1) / np.sqrt(a.size)),
                "n_rows": int(a.size),
                "source": "measured_605",
            }
            tmp = out_path.with_suffix(".json.tmp")
            tmp.write_text(
                json.dumps(
                    {
                        "schema_version": "issue605_v1",
                        "phase": "p4_fact_tf_priors",
                        "per_persona": done,
                        "metadata": _repro_meta(),
                    },
                    indent=1,
                )
            )
            tmp.replace(out_path)
    # Merge reused anchors into the same file for downstream reads.
    changed = False
    for p in pool:
        if p not in done and p in legacy:
            done[p] = {"mean_logprob_per_tok": legacy[p], "source": "reused_541"}
            changed = True
    if changed or not out_path.exists():
        out_path.write_text(
            json.dumps(
                {
                    "schema_version": "issue605_v1",
                    "phase": "p4_fact_tf_priors",
                    "per_persona": done,
                    "metadata": _repro_meta(),
                },
                indent=1,
            )
        )
    logger.info("[phase=p4_tf] TF priors complete: %d personas", len(done))


def fact_measure_fp_gen(out_dir: Path, cands: dict, dry_run: bool) -> None:
    """Base FP screen generations: 12-probe subsample, temp-0 (new candidates
    only — the 24 anchors passed #541's screen)."""
    from issue444_bystander_logprob import _chat_prompt
    from issue541_geometry_extract import _build_probes
    from transformers import AutoTokenizer

    probes = _build_probes(40)[:F_FP_PROBES]
    pool = _fact_persona_prompts(cands)
    out_path = out_dir / "panel" / "fact_measure" / "fp_gen.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict[str, str]] = {}
    if out_path.exists():
        done = json.loads(out_path.read_text())["completions"]
    pending = [p for p in cands if p not in done]
    logger.info("[phase=p4_fp_gen] %d candidates pending FP generations", len(pending))
    if dry_run or not pending:
        logger.info("[phase=p4_fp_gen] dry-run or nothing pending; stopping before vLLM")
        return
    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    from issue532_predictor_stress import _build_vllm_engine, _vllm_generate_R

    llm = _build_vllm_engine(max_seq_len=4096, enable_lora=False)
    for p in pending:
        prompts = [_chat_prompt(tok, pool[p], q) for q in probes]
        R = _vllm_generate_R(llm, prompts, cell_label=f"P4-fpgen/{p}")
        done[p] = {q: r for q, r in zip(probes, R, strict=True)}
        tmp = out_path.with_suffix(".json.tmp")
        tmp.write_text(
            json.dumps(
                {
                    "schema_version": "issue605_v1",
                    "phase": "p4_fact_fp_gen",
                    "n_probes": len(probes),
                    "completions": done,
                    "metadata": _repro_meta(),
                },
                indent=1,
            )
        )
        tmp.replace(out_path)


def fact_measure_fp_judge(out_dir: Path, dry_run: bool) -> None:
    """Haiku 5-way judge over FP-screen generations (CPU/API). A candidate
    asserting the taught count pre-training is excluded (#541 protocol)."""
    from issue605_eval_fact import judge_rows_5way  # shared batch-judge helper

    gen_path = out_dir / "panel" / "fact_measure" / "fp_gen.json"
    out_path = out_dir / "panel" / "fact_measure" / "fp_screen.json"
    if not gen_path.exists():
        logger.info("[phase=p4_fp_judge] no fp_gen.json — nothing to judge")
        return
    comp = json.loads(gen_path.read_text())["completions"]
    done: dict[str, dict] = {}
    if out_path.exists():
        done = json.loads(out_path.read_text())["per_candidate"]
    jobs = []
    for p, qmap in comp.items():
        if p in done:
            continue
        for q, r in qmap.items():
            jobs.append({"persona": p, "probe": q, "completion": r})
    logger.info("[phase=p4_fp_judge] %d rows to judge", len(jobs))
    if dry_run or not jobs:
        return
    verdicts = judge_rows_5way(jobs)
    by_p: dict[str, list[str]] = {}
    for j, v in zip(jobs, verdicts, strict=True):
        by_p.setdefault(j["persona"], []).append(v.get("output_category_5way") or "error")
    for p, cats in by_p.items():
        n7 = sum(c == "stated_seven" for c in cats)
        done[p] = {
            "n_rows": len(cats),
            "stated_seven": n7,
            "excluded": n7 > 0,
            "categories": cats,
        }
    out_path.write_text(
        json.dumps(
            {
                "schema_version": "issue605_v1",
                "phase": "p4_fact_fp_screen",
                "per_candidate": done,
                "metadata": _repro_meta(),
            },
            indent=1,
        )
    )
    n_exc = sum(d["excluded"] for d in done.values())
    logger.info("[phase=p4_fp_judge] screened %d candidates, %d excluded", len(done), n_exc)


def fact_measure_acts(out_dir: Path, cands: dict, dry_run: bool) -> None:
    """L21/L22 last-token activations over the 40 #541 A-family probes for
    candidates + anchors (teachers are anchor members)."""
    from issue541_geometry_extract import _build_probes

    pool = _fact_persona_prompts(cands)
    probes = _build_probes(40)
    acts_dir = out_dir / "panel" / "fact_measure" / "acts"
    acts_dir.mkdir(parents=True, exist_ok=True)
    pending = [p for p in pool if not (acts_dir / f"{p}.npz").exists()]
    logger.info("[phase=p4_acts] %d personas pending activations", len(pending))
    if dry_run:
        logger.info("[phase=p4_acts] dry-run: stopping before HF load")
        return
    if not pending:
        return
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa"
    )
    model.eval()
    import issue444_persona_distance_topic as pdt

    pdt.LAYERS = [COSINE_LAYER, GAUSS_KL_LAYER]
    pdt.PERSONA_PROMPTS = dict(pool)
    for i, p in enumerate(pending):
        acts = pdt.last_token_acts(model, tok, p, probes, "cuda")
        np.savez(
            acts_dir / f"{p}.npz",
            **{str(li): acts[li].numpy().astype(np.float32) for li in pdt.LAYERS},
        )
        logger.info("[phase=p4_acts] %s done (%d/%d)", p, i + 1, len(pending))
    del model
    torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# FACT select (Phase 4.5, CPU)
# ---------------------------------------------------------------------------
def fact_select(
    out_dir: Path,
    cands: dict,
    allow_descope: bool = False,
    include_expansion: bool = False,
) -> None:
    """Per-arm 18-persona panel with the fact-scale gate + disjointness."""
    from issue444_bystander_logprob import _chat_prompt
    from issue532_predictor_stress import _cosine_predictor
    from transformers import AutoTokenizer

    f_dir = out_dir / "panel" / "fact_measure"
    priors = json.loads((f_dir / "tf_priors.json").read_text())["per_persona"]
    fp_path = f_dir / "fp_screen.json"
    excluded: set[str] = set()
    if fp_path.exists():
        fp = json.loads(fp_path.read_text())["per_candidate"]
        excluded = {p for p, d in fp.items() if d["excluded"]}
    pool = _fact_persona_prompts(cands)
    acts_dir = f_dir / "acts"
    acts = {}
    for p in pool:
        with np.load(acts_dir / f"{p}.npz") as z:
            acts[p] = {int(k): z[k] for k in z.files}

    tok = AutoTokenizer.from_pretrained(BASE_MODEL)
    neg_rendered = set()
    probe_q = "What is this place known for?"  # representative rendering probe
    from explore_persona_space.personas import ASSISTANT_PROMPT, PERSONAS

    neg_prompt_map = {
        "assistant": ASSISTANT_PROMPT,
        "software_engineer": PERSONAS["software_engineer"],
        "kindergarten_teacher": PERSONAS["kindergarten_teacher"],
        "no_system": None,
    }
    assert set(neg_prompt_map) == set(FACT_TRAINING_NEGATIVES)
    for sysp in neg_prompt_map.values():
        neg_rendered.add(_chat_prompt(tok, sysp, probe_q))
    teacher_rendered = {t: _chat_prompt(tok, pool[t], probe_q) for t in FACT_TEACHERS}

    selections = {}
    for teacher in FACT_TEACHERS:
        eligible = [
            p
            for p in pool
            if p not in FACT_TEACHERS
            and p not in FACT_TRAINING_NEGATIVES
            and p not in excluded
            and p in priors
        ]
        pairs = [
            {
                "source_cid": teacher,
                "context_label": p,
                "cos_l21": _cosine_predictor(acts[teacher][COSINE_LAYER], acts[p][COSINE_LAYER]),
                "prior_logp": priors[p]["mean_logprob_per_tok"],
            }
            for p in eligible
        ]
        sel = _select_panel(
            pairs,
            panel_size=F_PANEL_SIZE,
            width=F_BAND_WIDTH,
            fill=F_FILL,
            spread_min=F_SPREAD_NAT_PER_TOK,
            rmax=F_RMAX,
            sim_key="cos_l21",
            prior_key="prior_logp",
        )
        # Rendered-string disjointness (consistency-checker WARN, plan 4.5):
        # selected persona's rendered prompt not equal to any teacher's or any
        # #541 training negative's rendering (incl. the empty/no-system one).
        for p in sel["panel"]:
            rendered = _chat_prompt(tok, pool[p], probe_q)
            assert rendered not in neg_rendered, f"{p} renders as a #541 training negative"
            for t, tr in teacher_rendered.items():
                assert rendered != tr, f"{p} renders identically to teacher {t}"
        sel["per_persona"] = {
            p: {
                "prior_nat_per_tok": priors[p]["mean_logprob_per_tok"],
                "prior_source": priors[p].get("source", "measured_605"),
                "cos_to_teacher": next(r["cos_l21"] for r in pairs if r["context_label"] == p),
            }
            for p in sel["panel"]
        }
        selections[teacher] = sel
        logger.info(
            "[phase=p45_select] arm=%s panel=%d gate_pass=%s",
            teacher,
            sel["panel_size"],
            sel["gate_pass"],
        )

    if allow_descope:
        for sel in selections.values():
            if not sel["gate_pass"]:
                desc = _descope_record(sel)
                if desc is not None:
                    sel["descope"] = desc
    payload = {
        "schema_version": "issue605_v1",
        "phase": "p45_fact_panel_selection",
        "per_arm": selections,
        "fp_excluded": sorted(excluded),
        "disjointness_assert": "PASS (rendered-string vs 3 teachers + 4 #541 negatives)",
        "expansion_round": 1 if include_expansion else 0,
        "metadata": _repro_meta(),
    }
    out = out_dir / "panel" / "fact_panel_selection.json"
    tmp = out.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(out)
    logger.info("[phase=p45_select] written %s", out)
    for teacher, sel in selections.items():
        _enforce_selection_gate(sel, allow_descope, f"fact Phase 4.5 arm={teacher}", out)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
_MARKER_MEASURE_STAGES = ("gen", "reads")
_FACT_MEASURE_STAGES = ("tf", "fp-gen", "fp-judge", "acts")


def _spawn_stage(args: argparse.Namespace, stage: str) -> None:
    """Subprocess-isolate framework stages (vLLM never shares a process with
    HF loads — CLAUDE.md vLLM-teardown gotcha). Same shape in smoke + sweep."""
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--family",
        args.family,
        "--phase",
        "measure",
        "--stage",
        stage,
        "--candidates",
        args.candidates,
        "--n-probes",
        str(args.n_probes),
        "--out-dir",
        str(args.out_dir),
    ]
    if args.include_expansion:
        cmd.append("--include-expansion")
    if args.dry_run:
        cmd.append("--dry-run")
    # Sub-stage processes must NOT emit the reserved terminal [phase=done]
    # token into the shared log (poll_pipeline contract, incident #545).
    cmd.append("--no-done-marker")
    logger.info("[stage-dispatch] %s", " ".join(cmd))
    subprocess.run(cmd, env={**os.environ}, check=True)


def _run_marker(args: argparse.Namespace) -> None:
    """Marker-family phase/stage dispatch (smoke = sweep, subset via flags)."""
    cands = _resolve_marker_candidates(args.candidates, args.include_expansion)
    if args.phase in ("measure", "all"):
        if args.stage == "all":
            for st in _MARKER_MEASURE_STAGES:
                _spawn_stage(args, st)
        elif args.stage == "gen":
            marker_measure_gen(args.out_dir, cands, args.n_probes, args.dry_run)
        elif args.stage == "reads":
            marker_measure_reads(args.out_dir, cands, args.n_probes, args.dry_run)
        else:
            raise SystemExit(f"unknown marker measure stage {args.stage!r}")
    if args.phase in ("select", "all") and not args.dry_run:
        marker_select(
            args.out_dir,
            cands,
            args.n_probes,
            args.allow_descope,
            args.include_expansion,
            panel_size=args.panel_size if args.panel_size is not None else M_PANEL_SIZE,
            frozen_selection=args.frozen_selection,
        )


def _run_fact(args: argparse.Namespace) -> None:
    """Fact-family phase/stage dispatch (smoke = sweep, subset via flags)."""
    if args.panel_size is not None or args.frozen_selection is not None:
        raise SystemExit(
            "--panel-size / --frozen-selection are MARKER-only amendment flags "
            "(plan v3 touches the marker family only; fact is frozen)"
        )
    cands = _resolve_fact_candidates(args.candidates, args.include_expansion)
    if args.phase in ("measure", "all"):
        if args.stage == "all":
            for st in _FACT_MEASURE_STAGES:
                _spawn_stage(args, st)
        elif args.stage == "tf":
            fact_measure_tf(args.out_dir, cands, args.dry_run)
        elif args.stage == "fp-gen":
            fact_measure_fp_gen(args.out_dir, cands, args.dry_run)
        elif args.stage == "fp-judge":
            fact_measure_fp_judge(args.out_dir, args.dry_run)
        elif args.stage == "acts":
            fact_measure_acts(args.out_dir, cands, args.dry_run)
        else:
            raise SystemExit(f"unknown fact measure stage {args.stage!r}")
    if args.phase in ("select", "all") and not args.dry_run:
        fact_select(args.out_dir, cands, args.allow_descope, args.include_expansion)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Issue #605 base-side measurement + matched-panel selection.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--family", choices=["marker", "fact"], required=True)
    ap.add_argument("--phase", choices=["measure", "select", "all"], default="all")
    ap.add_argument(
        "--stage",
        default="all",
        help="measure sub-stage (marker: gen|reads|all; fact: tf|fp-gen|fp-judge|acts|all). "
        "'all' subprocess-isolates each framework stage.",
    )
    ap.add_argument("--candidates", default="all", help="comma list of candidate labels or 'all'")
    ap.add_argument("--include-expansion", action="store_true")
    ap.add_argument("--n-probes", type=int, default=50)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    ap.add_argument(
        "--panel-size",
        type=int,
        default=None,
        help=f"marker panel size target (default {M_PANEL_SIZE}; the "
        f"'{AMENDMENT_LABEL}' amendment passes 100). Marker family only.",
    )
    ap.add_argument(
        "--frozen-selection",
        type=Path,
        default=None,
        help="parent marker selection JSON whose band edges + per-stratum windows are "
        "FROZEN (amendment select-wide path; marker family + --phase select only). "
        "Output goes to a NEW marker_panel_selection_wide.json under the followup dir.",
    )
    ap.add_argument("--dry-run", action="store_true", help="stop each stage before model load")
    ap.add_argument(
        "--allow-descope",
        action="store_true",
        help="on a failed selection gate, record the pre-registered descope-to-populated-bands "
        "path in the selection JSON instead of exiting non-zero (plan section 3 structural "
        "alternative; only valid AFTER the one expansion round)",
    )
    ap.add_argument(
        "--no-done-marker",
        action="store_true",
        help=argparse.SUPPRESS,  # internal: sub-stage processes never emit [phase=done]
    )
    args = ap.parse_args()

    if args.frozen_selection is not None and args.phase != "select":
        raise SystemExit(
            "--frozen-selection is a select-only path (all candidate measurements already "
            "exist on disk; the amendment adds NO new base-side measurement — plan v3 §2.1). "
            "Pass --phase select."
        )

    t0 = time.time()
    if args.family == "marker":
        _run_marker(args)
    else:
        _run_fact(args)
    if args.no_done_marker:
        logger.info(
            "matched_panels sub-stage %s/%s complete in %.0fs",
            args.family,
            args.stage,
            time.time() - t0,
        )
    else:
        logger.info(
            "[phase=done] matched_panels %s/%s in %.0fs", args.family, args.phase, time.time() - t0
        )


if __name__ == "__main__":
    main()
