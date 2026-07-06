#!/usr/bin/env python
"""Issue #778 Phase 1 — persona-vector extraction + layer sweep, per trait.

Faithful reproduction of the arXiv 2507.21509 extraction pipeline (adapting
``external/persona_vectors/generate_vec.py`` + ``eval/eval_persona.py``), with the
standing Sonnet-4.5 graded-judge deviation:

  1. Load the paper's released 5 pos/neg system-prompt pairs + 20 extraction
     questions + verbatim eval rubric.
  2. Generate 10 on-policy rollouts under each pos + each neg instruction, per
     extraction question, T=1.0, batched via vLLM (2000 rollouts/trait).
  3. Judge-filter every rollout 0-100 (Sonnet 4.5, drop-never-coerce): keep pos>50,
     neg<50; persist per-arm dropped counts.
  4. Capture response-avg residual-stream activations at every layer (HF) for
     each KEPT rollout.
  5. r_B[layer] = mean(kept-pos) - mean(kept-neg) per layer 0..27.
  6. Cache r_B + the kept-rollout activation pools to disk.

Outputs (under ``data/issue_778/``):
  - ``rb/{trait}.pt``                     -> (28, 3584) r_B directions
  - ``activations/{trait}_pos.pt``        -> (n_kept_pos, 28, 3584)
  - ``activations/{trait}_neg.pt``        -> (n_kept_neg, 28, 3584)
  - ``extract/{trait}_meta.json``         -> dropped counts, kept counts, repro

vLLM (gen) and HF (capture) coexist in one process; the vLLM engine is reaped
between them (issue #685 coexistence teardown). ``--cells`` limits the traits run
(smoke = 1 trait); ``--n-questions`` / ``--n-rollouts`` shrink the slice.
"""

from __future__ import annotations

# vLLM V1 fork-safety: set spawn BEFORE any vllm import, since main() touches the
# tokenizer/transformers before LLM() construction (gotchas.md #628).
import os

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import json
import logging
import sys
from pathlib import Path

# Make ``scripts/`` importable so issue778_lib resolves whether run as a module
# or a file (the GCP/pod lane runs it as ``uv run python scripts/issue778_extract.py``).
sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue778_lib as lib

from explore_persona_space.orchestrate.env import load_dotenv

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue778.extract")

# Env credential assertion at entry (uv run does NOT auto-load .env).
load_dotenv()


def _chat_prompt(tokenizer, system: str, question: str) -> str:
    """Chat-templated prompt string (system + user), ready for generation."""
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _vllm_generate(llm, prompts: list[str], *, temperature: float, max_new: int) -> list[str]:
    """Batched vLLM generation, chunked (gotchas.md large-batch deadlock)."""
    from vllm import SamplingParams

    chunk_size = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
    sp = SamplingParams(temperature=temperature, top_p=1.0, max_tokens=max_new, min_tokens=1)
    out: list[str] = []
    n_chunks = (len(prompts) + chunk_size - 1) // chunk_size
    for i in range(0, len(prompts), chunk_size):
        chunk = prompts[i : i + chunk_size]
        logger.info(
            "[vllm-chunk] extract chunk %d/%d (%d prompts)",
            i // chunk_size + 1,
            n_chunks,
            len(chunk),
        )
        res = llm.generate(chunk, sp, use_tqdm=False)
        out.extend(o.outputs[0].text for o in res)
    return out


def _split_kept_pools(
    prompt_records: list[dict], jr: lib.JudgeResult
) -> tuple[list[int], list[int], int, int]:
    """Apply the judge-filter: keep pos>50 / neg<50; DROP None (REFUSAL/OOR).

    Returns (kept_pos_idx, kept_neg_idx, dropped_pos, dropped_neg). None scores
    (dropped-never-coerced) count as drops in the per-arm telemetry, never as a
    coerced number.
    """
    kept_pos_idx: list[int] = []
    kept_neg_idx: list[int] = []
    dropped_pos = 0
    dropped_neg = 0
    for j, rec in enumerate(prompt_records):
        score = jr.scores.get(rec["item_id"])
        rec["score"] = score
        if rec["side"] == "pos":
            if score is None:
                dropped_pos += 1
            elif score > lib.JUDGE_THRESHOLD:
                kept_pos_idx.append(j)
        else:  # neg
            if score is None:
                dropped_neg += 1
            elif score < lib.JUDGE_THRESHOLD:
                kept_neg_idx.append(j)
    return kept_pos_idx, kept_neg_idx, dropped_pos, dropped_neg


def extract_trait(
    trait: str,
    external_root: Path,
    out_root: Path,
    *,
    n_questions: int,
    n_rollouts: int,
) -> dict:
    """Run the full extraction pipeline for one trait; write r_B + pools + meta."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lib.log_phase("extract", f"trait={trait} start", trait=trait)
    td = lib.load_trait_data(external_root, trait)
    tokenizer = AutoTokenizer.from_pretrained(lib.MODEL_NAME)

    questions = td.extract_questions[:n_questions]
    # Build the full rollout request set: for each (side, instruction, question),
    # n_rollouts prompts. We track (side, question, prompt) so the raw text +
    # activation capture line up with the judge scores.
    prompt_records: list[dict] = []  # {side, instr_k, question, prompt}
    for side in ("pos", "neg"):
        instrs = td.pos_instructions if side == "pos" else td.neg_instructions
        for k, instr in enumerate(instrs):
            system = lib.extraction_system_prompt(trait, instr, side)
            for q in questions:
                chat = _chat_prompt(tokenizer, system, q)
                for _ in range(n_rollouts):
                    prompt_records.append(
                        {"side": side, "instr_k": k, "question": q, "prompt": chat}
                    )

    # ── vLLM generation ──────────────────────────────────────────────────────
    llm = lib.build_vllm_engine()
    try:
        prompts = [r["prompt"] for r in prompt_records]
        answers = _vllm_generate(
            llm, prompts, temperature=lib.EXTRACT_TEMPERATURE, max_new=lib.MAX_NEW_TOKENS
        )
    finally:
        lib.reap_vllm_engine(llm)
    for rec, ans in zip(prompt_records, answers, strict=True):
        rec["answer"] = ans
    lib.log_phase("extract", f"trait={trait} generated {len(answers)} rollouts", trait=trait)

    # ── Judge-filter (Sonnet 4.5, drop-never-coerce) ─────────────────────────
    cache_dir = out_root / "extract" / f"{trait}_judge_cache"
    save_raw = out_root / "extract" / f"{trait}_judge_raw.json"
    save_raw.parent.mkdir(parents=True, exist_ok=True)
    # item_id must not contain "__" (custom_id delimiter) — use side/k/qidx.
    q_index = {q: i for i, q in enumerate(questions)}
    judge_items: list[tuple[str, str, str]] = []
    for j, rec in enumerate(prompt_records):
        item_id = f"{rec['side']}-{rec['instr_k']}-{q_index[rec['question']]}-{j:06d}"
        rec["item_id"] = item_id
        judge_items.append((item_id, rec["question"], rec["answer"]))
    # The extraction FILTER uses ONE judge draw per rollout: pos>50/neg<50 is a
    # threshold, not a graded ranking DV, so N=1 is correct here (plan v2 §8).
    # The graded N=6 multi-sampling is for the PREDICTION DVs (monitoring/finetune).
    jr = lib.judge_graded(
        judge_items,
        td.eval_prompt,
        n_draws=1,
        cache_dir=cache_dir,
        save_raw=save_raw,
        temperature=lib.JUDGE_TEMPERATURE,
    )

    # Split kept pos / kept neg (drop REFUSAL/OOR = score None).
    kept_pos_idx, kept_neg_idx, dropped_pos, dropped_neg = _split_kept_pools(prompt_records, jr)
    logger.info(
        "trait=%s kept pos=%d neg=%d | dropped(refusal/oor) pos=%d neg=%d",
        trait,
        len(kept_pos_idx),
        len(kept_neg_idx),
        dropped_pos,
        dropped_neg,
    )
    if not kept_pos_idx or not kept_neg_idx:
        raise RuntimeError(
            f"trait={trait}: empty kept pool (pos={len(kept_pos_idx)}, "
            f"neg={len(kept_neg_idx)}) — extraction cannot build r_B. Check judge wiring."
        )

    # ── HF activation capture (response-avg, all layers) ─────────────────────
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        lib.MODEL_NAME, torch_dtype=dtype, device_map=device if device == "cuda" else None
    )
    if device == "cpu":
        model = model.to(device)
    try:
        pos_prompts = [prompt_records[j]["prompt"] for j in kept_pos_idx]
        pos_answers = [prompt_records[j]["answer"] for j in kept_pos_idx]
        neg_prompts = [prompt_records[j]["prompt"] for j in kept_neg_idx]
        neg_answers = [prompt_records[j]["answer"] for j in kept_neg_idx]
        pos_acts = lib.capture_response_avg_all_layers(
            model, tokenizer, pos_prompts, pos_answers, device=model.device
        )
        neg_acts = lib.capture_response_avg_all_layers(
            model, tokenizer, neg_prompts, neg_answers, device=model.device
        )
    finally:
        del model
        import gc

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()

    # ── r_B = diff of means, per layer ───────────────────────────────────────
    rb = pos_acts.mean(dim=0) - neg_acts.mean(dim=0)  # (28, 3584)
    assert rb.shape == (lib.N_LAYERS, lib.HIDDEN_DIM), rb.shape

    rb_dir = out_root / "rb"
    acts_dir = out_root / "activations"
    rb_dir.mkdir(parents=True, exist_ok=True)
    acts_dir.mkdir(parents=True, exist_ok=True)
    torch.save(rb, rb_dir / f"{trait}.pt")
    torch.save(pos_acts, acts_dir / f"{trait}_pos.pt")
    torch.save(neg_acts, acts_dir / f"{trait}_neg.pt")

    meta = {
        "trait": trait,
        "n_questions": len(questions),
        "n_rollouts_per_side": n_rollouts,
        "n_kept_pos": len(kept_pos_idx),
        "n_kept_neg": len(kept_neg_idx),
        "n_dropped_pos_refusal_oor": dropped_pos,
        "n_dropped_neg_refusal_oor": dropped_neg,
        "judge_draws_total": jr.n_total_draws,
        "judge_draws_dropped": jr.n_dropped_draws,
        "rb_norm_per_layer": [float(rb[layer].norm()) for layer in range(lib.N_LAYERS)],
        "reproducibility": lib.repro_metadata(),
    }
    meta_path = out_root / "extract" / f"{trait}_meta.json"
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    lib.log_phase(
        "extract",
        f"trait={trait} done",
        trait=trait,
        **{
            "n_kept_pos": len(kept_pos_idx),
            "n_kept_neg": len(kept_neg_idx),
        },
    )
    return meta


# ── v2 (faithful-extraction-honest-nulls-rerun): paired mask + coherence gate ──
#
# The v2 extraction restores the two audited deviations from the released
# generate_vec.py @ b8e0f044 (plan v8 §4 Component A):
#   (1) coherence gate >= 50 on BOTH arms (rubric verbatim from the released
#       eval/prompts.py::Prompts["coherence_0_100"]);
#   (2) ONE row-aligned paired mask
#       (pos_trait >= 50) & (neg_trait < 50) & (pos_coh >= 50) & (neg_coh >= 50)
#       — generate_vec.py:43 verbatim semantics (NOTE pos >= 50, not v1's > 50),
#       equal-count paired pools by construction, pairing keys recorded.
#
# Split into two phases so the pod releases before the Batch-API judge phase:
#   --paired-mask                : POD — generate + persist rollout text
#                                  IMMEDIATELY + capture per-rollout acts for
#                                  ALL rollouts PRE-filter.
#   --paired-mask --judge-harvest: VM — judge trait + coherence (2 separate
#                                  calls, llm-judging rule 8), build the paired
#                                  mask + r_B v2 + pairing metadata (CPU).
#
# Pairing key = (pair_idx, question_idx, rollout_idx): pos row i is paired with
# neg row i on the same key (the released pipeline masks BOTH row-aligned
# dataframes with ONE boolean vector).

V2_SUBDIR = "v2"
# W1 wiring gate (plan §7): evil cos(r_B v2, r_B v1) at r_B index 19 (paper
# layer 20) must be >= 0.5. Hallucination/sycophancy cosines are REPORTED only.
W1_TRAIT = "evil"
W1_LAYER_IDX = 19
W1_MIN_COS = 0.5
# K1 pair-yield floor (plan §7): >= 20 pairs OK; 5-19 low_N; < 5 N/A.
K1_OK_PAIRS = 20
K1_MIN_PAIRS = 5


def _v2_root(out_root: Path) -> Path:
    return out_root / V2_SUBDIR


def _load_coherence_prompt(external_root: Path) -> str:
    """The released coherence rubric, VERBATIM, imported from the pinned clone's
    eval/prompts.py (never paraphrased — replication fidelity)."""
    import importlib.util

    prompts_py = external_root / "eval" / "prompts.py"
    if not prompts_py.exists():
        raise FileNotFoundError(f"released prompts.py missing: {prompts_py}")
    spec = importlib.util.spec_from_file_location("pv_prompts", prompts_py)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[union-attr]
    text = mod.Prompts["coherence_0_100"]
    if "{question}" not in text or "{answer}" not in text:
        raise ValueError("coherence rubric missing {question}/{answer} slots")
    return text


def _v2_prompt_records(td: lib.TraitData, tokenizer, n_questions: int, n_rollouts: int):
    """Row-aligned v2 rollout request set with pairing keys.

    Ordering: pos arm rows first (pair-major, question, rollout), then the neg
    arm in the IDENTICAL (pair, question, rollout) order — so
    ``neg_row = pos_row + n_half`` and ONE boolean over the half-length pairs
    masks both arms (the released row-alignment).
    """
    questions = td.extract_questions[:n_questions]
    records: list[dict] = []
    for arm in ("pos", "neg"):
        instrs = td.pos_instructions if arm == "pos" else td.neg_instructions
        for k, instr in enumerate(instrs):
            system = lib.extraction_system_prompt(td.trait, instr, arm)
            for qi, q in enumerate(questions):
                chat = _chat_prompt(tokenizer, system, q)
                for ri in range(n_rollouts):
                    records.append(
                        {
                            "rollout_id": f"{arm}-{k}-{qi}-{ri:03d}",
                            "arm": arm,
                            "pair_idx": k,
                            "question_idx": qi,
                            "rollout_idx": ri,
                            "system_prompt": system,
                            "question": q,
                            "prompt": chat,
                        }
                    )
    n_half = len(records) // 2
    assert len(records) == 2 * n_half, len(records)
    for j in range(n_half):
        p, n = records[j], records[j + n_half]
        assert (p["pair_idx"], p["question_idx"], p["rollout_idx"]) == (
            n["pair_idx"],
            n["question_idx"],
            n["rollout_idx"],
        ), f"pairing misalignment at row {j}"
        assert p["arm"] == "pos" and n["arm"] == "neg"
    return records, questions


def _stub_responses(records: list[dict]) -> list[str]:
    """Deterministic stub generations for the CPU smoke (--gen-stub).

    Content-free placeholder text keyed on the rollout id — exercises the
    persist/align/judge/mask plumbing without a GPU and without putting any
    trait-eliciting text in context.
    """
    return [f"[stub response {r['rollout_id']}] This is a placeholder answer." for r in records]


def _extract_trait_v2_done(
    trait: str,
    out_root: Path,
    *,
    n_questions: int,
    n_rollouts: int,
    gen_stub: bool,
    capture_stub: bool,
    skip_capture: bool,
) -> bool:
    """True iff the pod extract phase for ``trait`` already produced complete
    artifacts UNDER THE SAME PARAMS (a completed-trait resume predicate — a pod
    crash at trait 3 must not re-sample traits 1-2 at T=1.0 and silently
    replace their persisted rollout text; concern ``v2-ladder-resume-incomplete``).
    Keyed on the gen_meta params (incl. the stub flags, so a production run can
    never resume onto stub artifacts) + rollouts row count + acts shape."""
    extract_dir = _v2_root(out_root) / "extract"
    meta_path = extract_dir / f"{trait}_gen_meta.json"
    rollouts_path = extract_dir / f"{trait}_rollouts.jsonl"
    acts_path = extract_dir / f"{trait}_acts_all.pt"
    if not (meta_path.exists() and rollouts_path.exists()):
        return False
    try:
        with open(meta_path) as f:
            meta = json.load(f)
    except (json.JSONDecodeError, OSError):
        return False
    want = {
        "n_questions": n_questions,
        "n_rollouts_per_cell": n_rollouts,
        "gen_stub": gen_stub,
        "capture_stub": capture_stub,
        "skip_capture": skip_capture,
    }
    if any(meta.get(k) != v for k, v in want.items()):
        return False
    n_total = meta.get("n_rollouts_total")
    if not isinstance(n_total, int) or n_total <= 0:
        return False
    with open(rollouts_path) as f:
        n_lines = sum(1 for ln in f if ln.strip())
    if n_lines != n_total:
        return False
    if not skip_capture:
        if not acts_path.exists():
            return False
        import torch

        try:
            acts = torch.load(acts_path, weights_only=False, map_location="cpu")
        except Exception as e:  # torch.load raises heterogeneous types on corruption
            lib.log_phase("extract_v2", f"trait={trait} unreadable acts_all ({e}) — regenerating")
            return False
        if tuple(acts.shape) != (n_total, lib.N_LAYERS, lib.HIDDEN_DIM):
            return False
    return True


def extract_trait_v2(
    trait: str,
    external_root: Path,
    out_root: Path,
    *,
    n_questions: int,
    n_rollouts: int,
    gen_stub: bool = False,
    capture_stub: bool = False,
    skip_capture: bool = False,
    force: bool = False,
) -> dict:
    """v2 GPU phase: generate + persist ALL rollout text, capture ALL acts.

    NO filtering happens here — the judge scores arrive after pod release, so
    the acts are captured for every rollout PRE-filter (plan §4 A-2). Writes:
      v2/extract/{trait}_rollouts.jsonl   (one row per rollout, pairing keys)
      v2/extract/{trait}_acts_all.pt      ((n, 28, 3584) fp32, row-aligned)
      v2/extract/{trait}_gen_meta.json

    A trait whose artifacts are already complete under IDENTICAL params is
    SKIPPED (resume predicate above) unless ``force`` — regeneration would
    re-sample at T=1.0 and silently replace persisted rollout text.
    """
    import torch
    from transformers import AutoTokenizer

    v2 = _v2_root(out_root)
    if not force and _extract_trait_v2_done(
        trait,
        out_root,
        n_questions=n_questions,
        n_rollouts=n_rollouts,
        gen_stub=gen_stub,
        capture_stub=capture_stub,
        skip_capture=skip_capture,
    ):
        lib.log_phase(
            "extract_v2",
            f"trait={trait} already complete — SKIPPED (matching params; --force to regenerate)",
            trait=trait,
        )
        with open(v2 / "extract" / f"{trait}_gen_meta.json") as f:
            meta = json.load(f)
        meta["resumed_from_disk"] = True
        return meta
    lib.log_phase("extract_v2", f"trait={trait} start", trait=trait)
    td = lib.load_trait_data(external_root, trait)
    tokenizer = AutoTokenizer.from_pretrained(lib.MODEL_NAME)
    records, questions = _v2_prompt_records(td, tokenizer, n_questions, n_rollouts)

    # ── generation (vLLM batched; stub for the CPU smoke) ─────────────────────
    if gen_stub:
        answers = _stub_responses(records)
    else:
        llm = lib.build_vllm_engine()
        try:
            answers = _vllm_generate(
                llm,
                [r["prompt"] for r in records],
                temperature=lib.EXTRACT_TEMPERATURE,
                max_new=lib.MAX_NEW_TOKENS,
            )
        finally:
            lib.reap_vllm_engine(llm)
    for rec, ans in zip(records, answers, strict=True):
        rec["response"] = ans

    # ── persist rollout TEXT IMMEDIATELY (checkpoint-per-phase; #779 lesson) ──
    extract_dir = v2 / "extract"
    extract_dir.mkdir(parents=True, exist_ok=True)
    rollouts_path = extract_dir / f"{trait}_rollouts.jsonl"
    with open(rollouts_path, "w") as f:
        for rec in records:
            row = {k: rec[k] for k in rec if k != "prompt"}
            f.write(json.dumps(row) + "\n")
    lib.log_phase(
        "extract_v2", f"trait={trait} rollout text persisted ({len(records)} rows)", trait=trait
    )

    # ── per-rollout acts for ALL rollouts, pre-filter ─────────────────────────
    if skip_capture:
        lib.log_phase("extract_v2", f"trait={trait} capture SKIPPED (--skip-capture)")
    elif capture_stub:
        rng = torch.Generator().manual_seed(42)
        acts = torch.randn(
            (len(records), lib.N_LAYERS, lib.HIDDEN_DIM), generator=rng, dtype=torch.float32
        )
        torch.save(acts, extract_dir / f"{trait}_acts_all.pt")
        lib.log_phase("extract_v2", f"trait={trait} STUB acts written (--capture-stub)")
    else:
        from transformers import AutoModelForCausalLM

        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        model = AutoModelForCausalLM.from_pretrained(
            lib.MODEL_NAME, torch_dtype=dtype, device_map=device if device == "cuda" else None
        )
        if device == "cpu":
            model = model.to(device)
        try:
            acts = lib.capture_response_avg_all_layers(
                model,
                tokenizer,
                [r["prompt"] for r in records],
                [r["response"] for r in records],
                device=model.device,
            )
        finally:
            del model
            import gc

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        assert acts.shape == (len(records), lib.N_LAYERS, lib.HIDDEN_DIM), acts.shape
        torch.save(acts, extract_dir / f"{trait}_acts_all.pt")

    meta = {
        "trait": trait,
        "phase": "extract_v2_gen_capture",
        "n_rollouts_total": len(records),
        "n_pairs_total": len(records) // 2,
        "n_questions": len(questions),
        "n_rollouts_per_cell": n_rollouts,
        "gen_stub": gen_stub,
        "capture_stub": capture_stub,
        "skip_capture": skip_capture,
        "reproducibility": lib.repro_metadata(),
    }
    with open(extract_dir / f"{trait}_gen_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    lib.log_phase("extract_v2", f"trait={trait} done", trait=trait, n_rollouts=len(records))
    return meta


def judge_and_build_v2(  # noqa: C901
    trait: str,
    external_root: Path,
    out_root: Path,
    *,
    judge_dry_run: bool = False,
    judge_stub: bool = False,
    allow_gate_skip: bool = False,
) -> dict:
    """v2 VM phase: judge trait+coherence, build the paired mask + r_B v2.

    Reads the pod-produced rollouts JSONL + acts_all tensor; judges every
    rollout on TWO rubric dims in SEPARATE calls (llm-judging rule 8): the
    trait's released eval_prompt and the released coherence rubric. N=1 draw
    per dim (threshold gate, not a ranking DV — plan §11). Drop-never-coerce:
    a None score on EITHER dim of EITHER arm excludes the PAIR (counted
    per-arm per-dim). Then the released mask (generate_vec.py:43 verbatim
    semantics), equal-count paired pools, r_B v2 = diff of kept means.
    """
    import numpy as np
    import torch

    v2 = _v2_root(out_root)
    extract_dir = v2 / "extract"
    rollouts_path = extract_dir / f"{trait}_rollouts.jsonl"
    acts_path = extract_dir / f"{trait}_acts_all.pt"
    if not rollouts_path.exists():
        raise FileNotFoundError(f"v2 rollouts missing: {rollouts_path} (run --paired-mask first)")
    with open(rollouts_path) as f:
        records = [json.loads(ln) for ln in f if ln.strip()]
    n_half = len(records) // 2
    if len(records) != 2 * n_half:
        raise RuntimeError(f"{trait}: odd rollout count {len(records)} — pairing broken")

    td = lib.load_trait_data(external_root, trait)
    coherence_prompt = _load_coherence_prompt(external_root)
    judge_dir = v2 / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    items = [(r["rollout_id"], r["question"], r["response"]) for r in records]

    if judge_stub:
        # SMOKE ONLY: deterministic scores exercising the FULL mask + r_B build
        # path with zero API calls (pos->trait 85, neg->trait 10, coherence 90;
        # every 7th pair unevaluable to exercise the drop counting).
        stub_t: dict[str, float | None] = {}
        stub_c: dict[str, float | None] = {}
        for j in range(n_half):
            p_rec, n_rec = records[j], records[j + n_half]
            if j % 7 == 6:
                stub_t[p_rec["rollout_id"]] = None
            else:
                stub_t[p_rec["rollout_id"]] = 85.0
            stub_t[n_rec["rollout_id"]] = 10.0
            stub_c[p_rec["rollout_id"]] = 90.0
            stub_c[n_rec["rollout_id"]] = 90.0
        jr_trait = lib.JudgeResult(
            scores=stub_t, n_total_draws=2 * n_half, n_dropped_draws=n_half // 7
        )
        jr_coh = lib.JudgeResult(scores=stub_c, n_total_draws=2 * n_half, n_dropped_draws=0)
        lib.log_phase("judge_v2", f"trait={trait} JUDGE STUB (smoke) — no API calls")
    else:
        # Two rubric dims, two SEPARATE judge calls (one behavior per call).
        jr_trait = lib.judge_graded(
            items,
            td.eval_prompt,
            n_draws=1,
            cache_dir=judge_dir / f"{trait}_trait_cache",
            save_raw=judge_dir / f"{trait}_judge_raw_trait.json",
            temperature=lib.JUDGE_TEMPERATURE,
            dry_run=judge_dry_run,
        )
        jr_coh = lib.judge_graded(
            items,
            coherence_prompt,
            n_draws=1,
            cache_dir=judge_dir / f"{trait}_coherence_cache",
            save_raw=judge_dir / f"{trait}_judge_raw_coherence.json",
            temperature=lib.JUDGE_TEMPERATURE,
            dry_run=judge_dry_run,
        )
    if judge_dry_run and not judge_stub:
        lib.log_phase("judge_v2", f"trait={trait} DRY RUN — requests built, no API calls")
        return {"trait": trait, "status": "judge_dry_run"}

    # ── the released paired mask (generate_vec.py:43 @ b8e0f044, verbatim) ────
    # mask = (pos[trait] >= 50) & (neg[trait] < 100-50) & (pos.coherence >= 50)
    #        & (neg.coherence >= 50), threshold=50; ONE row-aligned boolean.
    thr = float(lib.JUDGE_THRESHOLD)
    drops = {
        "pos_trait": 0,
        "neg_trait": 0,
        "pos_coherence": 0,
        "neg_coherence": 0,
    }
    mask: list[bool] = []
    pair_rows: list[dict] = []
    for j in range(n_half):
        p, n = records[j], records[j + n_half]
        key = [p["pair_idx"], p["question_idx"], p["rollout_idx"]]
        pt = jr_trait.scores.get(p["rollout_id"])
        nt = jr_trait.scores.get(n["rollout_id"])
        pc = jr_coh.scores.get(p["rollout_id"])
        nc = jr_coh.scores.get(n["rollout_id"])
        unevaluable = False
        if pt is None:
            drops["pos_trait"] += 1
            unevaluable = True
        if nt is None:
            drops["neg_trait"] += 1
            unevaluable = True
        if pc is None:
            drops["pos_coherence"] += 1
            unevaluable = True
        if nc is None:
            drops["neg_coherence"] += 1
            unevaluable = True
        keep = (
            (not unevaluable) and (pt >= thr) and (nt < 100.0 - thr) and (pc >= thr) and (nc >= thr)
        )
        mask.append(bool(keep))
        pair_rows.append(
            {
                "pair_key": key,
                "pos_rollout_id": p["rollout_id"],
                "neg_rollout_id": n["rollout_id"],
                "pos_trait": pt,
                "neg_trait": nt,
                "pos_coherence": pc,
                "neg_coherence": nc,
                "unevaluable": unevaluable,
                "kept": bool(keep),
            }
        )
    kept_idx = [j for j, m in enumerate(mask) if m]
    n_kept = len(kept_idx)
    status = "ok" if n_kept >= K1_OK_PAIRS else ("low_N" if n_kept >= K1_MIN_PAIRS else "NA")
    lib.log_phase(
        "judge_v2",
        f"trait={trait} paired mask built",
        n_pairs_total=n_half,
        n_kept_pairs=n_kept,
        k1_status=status,
        **{f"dropped_{k}": v for k, v in drops.items()},
    )

    pairing = {
        "trait": trait,
        "mask_semantics": (
            "(pos_trait >= 50) & (neg_trait < 50) & (pos_coherence >= 50) & "
            "(neg_coherence >= 50) — ONE row-aligned boolean over paired rows "
            "(generate_vec.py:43 @ b8e0f044; NOTE pos >= 50, not v1's > 50). "
            "A REFUSAL/non-numeric/out-of-range judge return on EITHER dim of "
            "EITHER arm makes the PAIR unevaluable (dropped, never coerced)."
        ),
        "threshold": thr,
        "n_pairs_total": n_half,
        "n_kept_pairs": n_kept,
        "kept_pos_count": n_kept,
        "kept_neg_count": n_kept,
        "k1_status": status,
        "dropped_unevaluable_by_arm_dim": drops,
        "judge_draw_telemetry": {
            "trait_total": jr_trait.n_total_draws,
            "trait_dropped": jr_trait.n_dropped_draws,
            "coherence_total": jr_coh.n_total_draws,
            "coherence_dropped": jr_coh.n_dropped_draws,
        },
        "kept_pair_keys": [pair_rows[j]["pair_key"] for j in kept_idx],
        "mask": mask,
        "pairs": pair_rows,
        "reproducibility": lib.repro_metadata(),
    }
    pairing_dir = v2 / "pairing"
    pairing_dir.mkdir(parents=True, exist_ok=True)
    with open(pairing_dir / f"{trait}_pairing.json", "w") as f:
        json.dump(pairing, f, indent=2)

    if status == "NA":
        # K1: < 5 kept pairs — this trait's v2 arm is N/A (reported, never a
        # fake zero). No r_B v2 is written; the other traits proceed.
        lib.log_phase("judge_v2", f"trait={trait} K1 N/A — insufficient paired pool ({n_kept})")
        return {"trait": trait, "status": "NA", "n_kept_pairs": n_kept}

    # ── r_B v2 = diff of kept means (paired, equal counts by construction) ────
    acts = torch.load(acts_path, weights_only=False)
    assert acts.shape == (len(records), lib.N_LAYERS, lib.HIDDEN_DIM), acts.shape
    pos_rows = torch.tensor(kept_idx, dtype=torch.long)
    neg_rows = pos_rows + n_half
    rb = acts[pos_rows].mean(dim=0) - acts[neg_rows].mean(dim=0)
    rb = rb.to(torch.float32)
    assert rb.shape == (lib.N_LAYERS, lib.HIDDEN_DIM), rb.shape
    rb_dir = v2 / "rb_v2"
    rb_dir.mkdir(parents=True, exist_ok=True)
    torch.save(rb, rb_dir / f"{trait}.pt")

    # ── W1 wiring gate + v1-vs-v2 per-layer cosine (reported for all traits) ──
    v1_rb_path = out_root / "rb" / f"{trait}.pt"
    cos_per_layer = None
    w1_gate = "n/a — not the W1 trait"
    if v1_rb_path.exists():
        rb_v1 = torch.load(v1_rb_path, weights_only=False).to(torch.float64)
        rb64 = rb.to(torch.float64)
        cos_per_layer = [
            float(
                torch.dot(rb64[layer], rb_v1[layer])
                / (torch.norm(rb64[layer]) * torch.norm(rb_v1[layer]))
            )
            for layer in range(lib.N_LAYERS)
        ]
        if trait == W1_TRAIT:
            c = cos_per_layer[W1_LAYER_IDX]
            if not np.isfinite(c) or c < W1_MIN_COS:
                raise RuntimeError(
                    f"W1 FAIL: cos(r_B v2, r_B v1) at layer index {W1_LAYER_IDX} for "
                    f"{trait} = {c:.4f} < {W1_MIN_COS} — wiring/sign/row-alignment bug, "
                    "not physics. Bounce to experiment-implementer before any "
                    "downstream compute (plan §7 W1)."
                )
            lib.log_phase("judge_v2", f"W1 PASS: evil cos(v2,v1)@L{W1_LAYER_IDX}={c:.4f}")
            w1_gate = "pass"
    elif trait == W1_TRAIT and not allow_gate_skip:
        # FAIL-CLOSED: an unarmed gate is not a passed gate (plan §7 W1 has STOP
        # semantics). Missing v1 r_B for the W1 trait means the wiring gate
        # cannot fire — raise in production; only the explicit smoke flag
        # (never set by the production driver) records a non-production skip.
        raise RuntimeError(
            f"W1 GATE UNARMED: v1 r_B missing at {v1_rb_path} for the W1 trait "
            f"{trait!r} — the plan §7 wiring gate cannot fire. Stage the v1 "
            "inputs (scripts/issue778_v2_prefetch.py) or pass "
            "--allow-gate-skip-smoke-only (smoke ONLY; recorded non-production)."
        )
    else:
        if trait == W1_TRAIT:
            w1_gate = "skipped_smoke_only — NON-PRODUCTION (v1 r_B missing)"
        skip_note = " (W1 GATE SKIP — smoke only, NON-PRODUCTION)" if trait == W1_TRAIT else ""
        lib.log_phase(
            "judge_v2", f"trait={trait} v1 rb missing — v1-vs-v2 cosine skipped{skip_note}"
        )

    meta = {
        "trait": trait,
        "status": status,
        "n_kept_pairs": n_kept,
        "rb_v2_norm_per_layer": [float(rb[layer].norm()) for layer in range(lib.N_LAYERS)],
        "cos_v2_v1_per_layer": cos_per_layer,
        "w1_gate": w1_gate,
        "reproducibility": lib.repro_metadata(),
    }
    with open(extract_dir / f"{trait}_v2_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    lib.log_phase("judge_v2", f"trait={trait} r_B v2 built", n_kept_pairs=n_kept, status=status)
    return meta


def main() -> None:
    parser = argparse.ArgumentParser(description="Issue #778 Phase 1 extraction.")
    parser.add_argument(
        "--external-root",
        default="external/persona_vectors",
        help="cloned safety-research/persona_vectors root",
    )
    parser.add_argument("--out-root", default="data/issue_778")
    parser.add_argument(
        "--traits",
        nargs="+",
        default=list(lib.TRAITS),
        help="traits to extract (default: all 3)",
    )
    parser.add_argument("--cells", type=int, default=None, help="limit to first N traits (smoke)")
    parser.add_argument("--n-questions", type=int, default=20)
    parser.add_argument("--n-rollouts", type=int, default=lib.N_ROLLOUTS_EXTRACT)
    parser.add_argument("--smoke", action="store_true")
    # v2 (faithful-extraction-honest-nulls-rerun) flags:
    parser.add_argument(
        "--paired-mask",
        action="store_true",
        help="v2 mode: generate + persist ALL rollout text + capture ALL acts pre-filter "
        "(pod phase); with --judge-harvest, run the VM judge+mask+r_B-v2 phase instead",
    )
    parser.add_argument(
        "--judge-harvest",
        action="store_true",
        help="v2 VM phase: judge trait+coherence, build the paired mask + r_B v2 "
        "(requires the --paired-mask pod outputs)",
    )
    parser.add_argument(
        "--gen-stub", action="store_true", help="SMOKE ONLY: stub generations (no vLLM/GPU)"
    )
    parser.add_argument(
        "--capture-stub",
        action="store_true",
        help="SMOKE ONLY: write deterministic random acts_all (no HF model/GPU)",
    )
    parser.add_argument(
        "--skip-capture", action="store_true", help="SMOKE ONLY: skip the acts capture entirely"
    )
    parser.add_argument(
        "--judge-dry-run",
        action="store_true",
        help="SMOKE: build judge Batch requests without any API call",
    )
    parser.add_argument(
        "--judge-stub",
        action="store_true",
        help="SMOKE ONLY: deterministic judge scores (no API) — exercises the full "
        "mask + r_B v2 build path",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="v2 --paired-mask: regenerate a trait whose extract artifacts are "
        "already complete (default: completed traits are SKIPPED — regeneration "
        "re-samples at T=1.0 and replaces persisted rollout text)",
    )
    parser.add_argument(
        "--allow-gate-skip-smoke-only",
        action="store_true",
        help="SMOKE ONLY (never set by the production driver): permit the W1 gate "
        "to record an unarmed skip (v1 r_B missing) instead of raising; recorded "
        "as non-production in the trait meta",
    )
    args = parser.parse_args()

    external_root = Path(args.external_root)
    out_root = Path(args.out_root)
    traits = args.traits
    if args.cells is not None:
        traits = traits[: args.cells]

    if args.judge_harvest and not args.paired_mask:
        parser.error("--judge-harvest requires --paired-mask (it is the v2 VM phase)")

    if args.paired_mask and args.judge_harvest:
        lib.log_phase("judge_v2", f"start traits={traits}")
        results = {}
        for trait in traits:
            results[trait] = judge_and_build_v2(
                trait,
                external_root,
                out_root,
                judge_dry_run=args.judge_dry_run,
                judge_stub=args.judge_stub,
                allow_gate_skip=args.allow_gate_skip_smoke_only,
            )
        lib.log_phase("judge_v2", f"all traits done ({len(results)})")
        print(json.dumps({"phase": "judge_v2", "results": results}, indent=2, default=str))
        return

    if args.paired_mask:
        lib.log_phase("extract_v2", f"start traits={traits}")
        results = {}
        for trait in traits:
            results[trait] = extract_trait_v2(
                trait,
                external_root,
                out_root,
                n_questions=args.n_questions,
                n_rollouts=args.n_rollouts,
                gen_stub=args.gen_stub,
                capture_stub=args.capture_stub,
                skip_capture=args.skip_capture,
                force=args.force,
            )
        lib.log_phase("extract_v2", f"all traits done ({len(results)})")
        print(json.dumps({"phase": "extract_v2", "traits": list(results)}, indent=2))
        return

    lib.log_phase("extract", f"start traits={traits}")
    results = {}
    for trait in traits:
        results[trait] = extract_trait(
            trait,
            external_root,
            out_root,
            n_questions=args.n_questions,
            n_rollouts=args.n_rollouts,
        )
    lib.log_phase("extract", f"all traits done ({len(results)})")
    print(json.dumps({"phase": "extract", "traits": list(results)}, indent=2))


if __name__ == "__main__":
    main()
