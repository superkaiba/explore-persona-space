# ruff: noqa: RUF001, RUF002, RUF003
"""Issue #488 Phase 1 — base-model predictors over the 27 conditions.

Computes per ``.claude/rules/persona-distance-metrics.md``:

* **RB sequence-level JS** (primary predictor): for each pair (T_i, T_j),
  sample R responses from the base model under both T_i and T_j over the
  50-probe Q'_probe (= load_q_test_extended_50, paired with #406 D matrix),
  teacher-force each sampled response through BOTH conditioned models,
  per-position full-vocab JS, length-normalize, mean over samples × probes.
  Symmetric (responses sampled from both personas).
* **RB sequence-level KL both directions** (secondary): same machinery; report
  KL(narrow‖broad) (sample from narrow) and KL(broad‖narrow) (sample from
  broad), plus symmetric-KL = ½ their sum.
* **Cosine similarity on residual-stream activations** at layers {7, 14, 21, 27}
  per persona-vectors recipe (a): last-input-token activation of the base
  model conditioned on T_i, cosine-vs-T_j, mean over probes.
* **stylization_score[i]** (H2 graded covariate): RB sequence-level JS of T_i
  vs the no-system-prompt baseline (i.e. the default-assistant chat-template,
  no system turn). Reuses the SAME JS estimator with REFERENCE pinned to the
  baseline. Plan v2 §4.3(b).

Generation hot path (round-5 pivot, 2026-06-05):
The previous implementation called HF ``model.generate(num_return_sequences=n)``
once per probe — 200 calls per cell, 50 wall-min per cell on a single GPU,
> 14 wall-h for the JS/KL stage even after fan-out across 8 GPUs (incident:
round-4 launch ran 2.5 h with 0/462 cells completed). Per CLAUDE.md
"Use vLLM for generation. Never sequential HF model.generate() for eval"
the hot path is now a two-pass-per-shard architecture:

  Pass A (vLLM): one ``llm.generate(prompts, SamplingParams(n=r_samples))``
                 call per cell, batching all 50 probes × 2 (sample-from-i,
                 sample-from-j) prompts into one continuous-batched call.
                 ~3-5 s per cell at r=2. Done for ALL assigned cells first,
                 then vLLM is torn down with ``kill_vllm_workers`` so the
                 workers don't squat on GPU memory when HF loads.

  Pass B (HF): for each cell, ONE batched no-grad forward through
               ``logp_under(T_i, [sample_a, sample_b, ...])`` and ONE under
               ``logp_under(T_j, [sample_a, sample_b, ...])``; the per-position
               JS / KL / sym-KL is then a vectorized reduction over the two
               batched (B, T, V) log-softmax tensors. Per-cell checkpoint
               writes to disk so a Pass-B crash loses at most one cell.

This eliminates the 200-call-per-cell HF generate loop, the 200-call-per-cell
single-row teacher-force loop, and the cell-by-cell vLLM↔HF framework swap
(vLLM is loaded ONCE for the whole Pass A, HF is loaded ONCE for the whole
Pass B). The JS / KL math is unchanged (still full-vocab per-position
log-softmax → JS / KL → length-normalize → mean over samples × probes).

Outputs (under ``eval_results/issue_488/predictors/``):

* ``js_matrix.json`` — ``{conditions: [...], JS: {cid_i: {cid_j: value}}}``,
  with the inherited 16×16 sub-grid LIFTED byte-identical from
  ``eval_results/issue_406/divergence/D_matrix.json["JS"]`` (plan §4.2 + §11
  assumption 3); only the 11 new ↔ everything cells are computed fresh.
* ``kl_matrix.json`` — analogous, both KL directions + symmetric-KL.
* ``cossim_matrix_layer{L}.json`` for L ∈ {7, 14, 21, 27}.
* ``stylization_score.json`` — ``{cid: js_to_no_system}`` over the 27 sources.
* ``is_stylized_source.json`` — binary lookup from
  ``i488_conditions.STRONG_STYLIZED_SOURCES``; written here so analyzer
  doesn't need to import.

Resume-safe per CLAUDE.md "Checkpoint per phase": per-cell outputs are
written incrementally; re-runs skip filled cells.

CLI:
    uv run python scripts/i488_phase1_predictors.py
    uv run python scripts/i488_phase1_predictors.py --skip cossim   # JS+KL+stylization
    uv run python scripts/i488_phase1_predictors.py --skip js kl    # cossim+stylization
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Make scripts/ importable for issue404_common.kill_vllm_workers.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from issue404_common import kill_vllm_workers  # noqa: E402

from explore_persona_space.experiments.i460_data import (  # noqa: E402
    load_class_d_rewrites,
    load_q_test_extended_50,
)
from explore_persona_space.experiments.i488_conditions import (  # noqa: E402
    CONDITIONS,
    CONDITIONS_BY_ID,
    STRONG_STYLIZED_SOURCES,
    build_prompt_for_condition,
)

logger = logging.getLogger("i488.phase1")

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
OUT_DIR = Path("eval_results/issue_488/predictors")
I406_D_MATRIX = Path("eval_results/issue_406/divergence/D_matrix.json")

R_SAMPLES = 8
N_PROBES = 50
COSINE_LAYERS = (7, 14, 21, 27)

# Inherited 16 conds (byte-identical with #406 D_matrix.json).
INHERITED_CIDS: frozenset[str] = frozenset(
    {c.cid for c in CONDITIONS if c.cls in {"A", "B", "C", "D"}}
)
NEW_CIDS: frozenset[str] = frozenset({c.cid for c in CONDITIONS if c.cls in {"E", "F", "G"}})

EPS = 1e-12
LN2 = torch.log(torch.tensor(2.0))

# vLLM gpu memory utilization. Conservative because Pass B reloads HF Transformers
# into the SAME process (vLLM teardown + worker reap, then HF model load). vLLM
# at 0.40 leaves enough headroom for the HF Qwen-7B bf16 weights (~15 GB) + the
# teacher-force activations (~10 GB) on an 80 GB H100.
VLLM_GPU_MEM_UTILIZATION = 0.40
# Max sequence length for vLLM. Prompt ~150 + response ≤ 256 (caller --max-new-tokens
# default) ≈ 410; round up for safety.
VLLM_MAX_MODEL_LEN = 2048
# HF teacher-force batch size. Each row is at most (prompt + max_new_tokens) tokens,
# typically ~400 tokens; 16 rows × 400 tokens × vocab × bf16 logits is ~2 GB at peak
# inside the model.forward. Tune down if OOM is seen during smoke.
HF_TEACHER_FORCE_BATCH = 16


# ── RB sequence-level estimators ─────────────────────────────────────────


def _per_position_js_batched(p_log: torch.Tensor, q_log: torch.Tensor) -> torch.Tensor:
    """Per-position JS divergence (base-2) between two (B, T, V) log-prob tensors.

    Args:
        p_log: (B, T, V) log-softmax under model T_i, teacher-forced.
        q_log: (B, T, V) log-softmax under model T_j on the same responses.

    Returns:
        (B, T) per-position JS values in [0, 1].
    """
    p = p_log.exp().clamp_min(EPS)
    q = q_log.exp().clamp_min(EPS)
    m = 0.5 * (p + q)
    log_m = torch.log(m)
    kl_pm = (p * (p_log - log_m)).sum(dim=-1) / LN2.to(p_log.device)
    kl_qm = (q * (q_log - log_m)).sum(dim=-1) / LN2.to(p_log.device)
    js = 0.5 * (kl_pm + kl_qm)
    return js.clamp(min=0.0, max=1.0)


def _per_position_kl_batched(p_log: torch.Tensor, q_log: torch.Tensor) -> torch.Tensor:
    """Per-position KL(p ‖ q) (base-2) over (B, T, V) log-prob tensors.

    Args:
        p_log: (B, T, V) log-softmax from the sample-from model.
        q_log: (B, T, V) log-softmax from the comparison model.

    Returns:
        (B, T) per-position KL(p ‖ q) values (bits).
    """
    p = p_log.exp().clamp_min(EPS)
    return (p * (p_log - q_log)).sum(dim=-1) / LN2.to(p_log.device)


# ── Pass A: vLLM batched response sampling ───────────────────────────────


def _vllm_sample_cell_responses(
    llm,
    tokenizer,
    cond_i,
    cond_j,
    probes: list[str],
    class_d_rewrites: dict | None,
    r_samples: int,
    max_new_tokens: int,
    seed: int,
) -> dict:
    """Sample ``r_samples`` responses per probe under both T_i and T_j with ONE
    vLLM call.

    Returns dict with keys:
      * "probes": list[str] (the probes, in order)
      * "prompts_i": list[str], one per probe
      * "prompts_j": list[str], one per probe
      * "samples_from_i": list[list[list[int]]]
            shape [n_probes][r_samples][response_token_ids]
      * "samples_from_j": same shape

    Per CLAUDE.md "Use vLLM for generation": ONE ``llm.generate`` call batches
    n_probes × 2 prompts; vLLM continuous-batching handles the scheduling.
    """
    from vllm import SamplingParams

    sp = SamplingParams(
        n=r_samples,
        temperature=1.0,
        top_p=1.0,
        max_tokens=max_new_tokens,
        seed=seed,
    )

    prompts_i = [build_prompt_for_condition(cond_i, q, tokenizer, class_d_rewrites) for q in probes]
    prompts_j = [build_prompt_for_condition(cond_j, q, tokenizer, class_d_rewrites) for q in probes]
    all_prompts = prompts_i + prompts_j  # [50 from-i, 50 from-j]
    outputs = llm.generate(all_prompts, sp)
    if len(outputs) != len(all_prompts):
        raise AssertionError(f"vLLM returned {len(outputs)} outputs for {len(all_prompts)} prompts")

    n = len(probes)
    samples_from_i: list[list[list[int]]] = []
    samples_from_j: list[list[list[int]]] = []
    for k in range(n):
        # outputs[k] = sample-from-i for probe k; outputs[n + k] = sample-from-j.
        out_i = outputs[k]
        if len(out_i.outputs) != r_samples:
            raise AssertionError(
                f"probe {k}: vLLM returned {len(out_i.outputs)} samples, expected {r_samples}"
            )
        out_j = outputs[n + k]
        samples_from_i.append([list(o.token_ids) for o in out_i.outputs])
        samples_from_j.append([list(o.token_ids) for o in out_j.outputs])

    return {
        "probes": probes,
        "prompts_i": prompts_i,
        "prompts_j": prompts_j,
        "samples_from_i": samples_from_i,
        "samples_from_j": samples_from_j,
    }


# ── Pass B: HF batched teacher-forcing ───────────────────────────────────


def _batched_teacher_force_logprobs(
    model,
    tokenizer,
    prompt_text: str,
    response_id_lists: list[list[int]],
    batch_size: int,
) -> torch.Tensor:
    """Teacher-force a batch of responses through the model under one prompt.

    Args:
        model: HF causal LM, eval mode.
        tokenizer: HF tokenizer (left or right pad — we right-pad explicitly here).
        prompt_text: the shared prompt prefix; tokenized once.
        response_id_lists: list of response token-id sequences. Variable length.
        batch_size: micro-batch size for the forward pass.

    Returns:
        (B, T_max, V) log-softmax tensor on CPU/float32, where T_max is the
        longest response length in the batch and rows shorter than T_max are
        padded with the model's prediction at the padded positions (which are
        EXCLUDED by ``_reduce_batch_div`` via per-row response lengths).

    Implementation: ONE forward pass per micro-batch of up to ``batch_size``
    rows. Pad right with pad_token_id; gather only the slice of logits that
    predicts the response tokens.
    """
    prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_len = len(prompt_ids)

    # Build full sequences = prompt + response.
    full_seqs = [prompt_ids + list(r) for r in response_id_lists]
    resp_lens = [len(r) for r in response_id_lists]
    t_max = max(resp_lens) if resp_lens else 0
    if t_max == 0:
        # No non-empty responses; return an empty tensor with vocab dim.
        return torch.zeros((len(response_id_lists), 0, model.config.vocab_size))

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    n = len(full_seqs)
    out_logprobs = torch.empty((n, t_max, model.config.vocab_size), dtype=torch.float32)

    for b_start in range(0, n, batch_size):
        b_end = min(b_start + batch_size, n)
        chunk = full_seqs[b_start:b_end]
        chunk_lens = [len(s) for s in chunk]
        bsz = len(chunk)
        cur_max = max(chunk_lens)
        ids = torch.full((bsz, cur_max), pad_id, dtype=torch.long)
        attn = torch.zeros((bsz, cur_max), dtype=torch.long)
        for i, s in enumerate(chunk):
            ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
            attn[i, : len(s)] = 1
        ids = ids.to(model.device)
        attn = attn.to(model.device)

        with torch.no_grad():
            logits = model(input_ids=ids, attention_mask=attn).logits  # (bsz, cur_max, V)
        # Slice the positions that predict the response tokens. For row i,
        # response[t] is at full position prompt_len + t, predicted by logits at
        # position prompt_len - 1 + t.
        # We want for each row the slice [prompt_len-1 : prompt_len-1+resp_lens[i]].
        # All rows share the same prompt_len, so the slice start is identical;
        # the slice length varies. We pull a uniform slice of length t_max
        # (some rows have garbage past their own response, which is masked below).
        sliced = logits[:, prompt_len - 1 : prompt_len - 1 + t_max, :]  # (bsz, t_max, V)
        # log_softmax on GPU, then move to CPU float32 to avoid OOM accumulation.
        logp = torch.log_softmax(sliced.float(), dim=-1).cpu()
        out_logprobs[b_start:b_end] = logp
        del logits, sliced, logp
    return out_logprobs


def _reduce_batch_div(
    logp_i: torch.Tensor,
    logp_j: torch.Tensor,
    resp_lens: list[int],
    sample_from: str,
) -> tuple[float, float]:
    """Mean per-position JS + KL across a batch of teacher-forced responses.

    Args:
        logp_i: (B, T_max, V) log-softmax under model T_i.
        logp_j: (B, T_max, V) log-softmax under model T_j (same responses).
        resp_lens: per-row actual response length.
        sample_from: "i" or "j" — which direction this batch's KL is being
            asked for (KL(p_i‖p_j) when sample_from=="i"; reversed otherwise).

    Returns:
        (js_lens_mean, kl_lens_mean)
            js_lens_mean: mean over rows of (mean-per-position JS in the row's
                response span). Length-normalized.
            kl_lens_mean: same for KL in the direction indicated.

    Skips rows with length 0 (caller filters those out via ``resp_lens``).
    """
    _n, t_max, _ = logp_i.shape
    if t_max == 0:
        return float("nan"), float("nan")
    js_full = _per_position_js_batched(logp_i, logp_j)  # (B, T_max)
    if sample_from == "i":
        kl_full = _per_position_kl_batched(logp_i, logp_j)  # KL(p_i ‖ p_j)
    elif sample_from == "j":
        kl_full = _per_position_kl_batched(logp_j, logp_i)  # KL(p_j ‖ p_i)
    else:
        raise ValueError(f"sample_from must be 'i' or 'j', got {sample_from!r}")

    js_per_row = []
    kl_per_row = []
    for row, ln in enumerate(resp_lens):
        if ln == 0:
            continue
        js_per_row.append(float(js_full[row, :ln].mean().item()))
        kl_per_row.append(float(kl_full[row, :ln].mean().item()))
    if not js_per_row:
        return float("nan"), float("nan")
    return (sum(js_per_row) / len(js_per_row), sum(kl_per_row) / len(kl_per_row))


def _compute_cell_divergences_from_samples(
    model,
    tokenizer,
    prompts_i: list[str],
    prompts_j: list[str],
    samples_from_i: list[list[list[int]]],
    samples_from_j: list[list[list[int]]],
    batch_size: int,
) -> dict[str, float]:
    """Reduce one cell's pre-sampled responses to JS + KL_ij + KL_ji + sym_KL.

    For each probe k:
      * teacher-force ``samples_from_i[k]`` through BOTH (prompt_i[k], prompt_j[k])
      * accumulate js (both directions averaged into ``js_lens``) and
        kl_ij (sample-from-i direction)
      * teacher-force ``samples_from_j[k]`` through BOTH (prompt_i[k], prompt_j[k])
      * accumulate js + kl_ji (sample-from-j direction)

    Per-position full-vocab reductions are batched: one HF forward per (probe,
    direction, model-side) sized at most ``batch_size``.

    Returns:
        dict with "js", "kl_ij", "kl_ji", "sym_kl", "n_js_terms".
    """
    n_probes = len(prompts_i)
    js_lens: list[float] = []
    kl_ij_lens: list[float] = []
    kl_ji_lens: list[float] = []

    for k in range(n_probes):
        p_i = prompts_i[k]
        p_j = prompts_j[k]

        # Sample-from-i: teacher-force same responses through i AND j models.
        ids_i = [r for r in samples_from_i[k] if len(r) > 0]
        if ids_i:
            lens_i = [len(r) for r in ids_i]
            logp_i_on_i = _batched_teacher_force_logprobs(model, tokenizer, p_i, ids_i, batch_size)
            logp_j_on_i = _batched_teacher_force_logprobs(model, tokenizer, p_j, ids_i, batch_size)
            js_i, kl_i = _reduce_batch_div(logp_i_on_i, logp_j_on_i, lens_i, "i")
            del logp_i_on_i, logp_j_on_i
            js_lens.append(js_i)
            kl_ij_lens.append(kl_i)

        # Sample-from-j: teacher-force same responses through i AND j models.
        ids_j = [r for r in samples_from_j[k] if len(r) > 0]
        if ids_j:
            lens_j = [len(r) for r in ids_j]
            logp_i_on_j = _batched_teacher_force_logprobs(model, tokenizer, p_i, ids_j, batch_size)
            logp_j_on_j = _batched_teacher_force_logprobs(model, tokenizer, p_j, ids_j, batch_size)
            js_j, kl_j = _reduce_batch_div(logp_i_on_j, logp_j_on_j, lens_j, "j")
            del logp_i_on_j, logp_j_on_j
            js_lens.append(js_j)
            kl_ji_lens.append(kl_j)

    def _mean_or_nan(xs: list[float]) -> float:
        finite = [x for x in xs if x == x]  # filter NaN
        return float(sum(finite) / len(finite)) if finite else float("nan")

    js_mean = _mean_or_nan(js_lens)
    kl_ij_mean = _mean_or_nan(kl_ij_lens)
    kl_ji_mean = _mean_or_nan(kl_ji_lens)
    sym_kl = 0.5 * (kl_ij_mean + kl_ji_mean) if js_lens else float("nan")
    return {
        "js": js_mean,
        "kl_ij": kl_ij_mean,
        "kl_ji": kl_ji_mean,
        "sym_kl": sym_kl,
        "n_js_terms": len(js_lens),
    }


# ── Inheritance from #406 D_matrix.json ──────────────────────────────────


def _load_inherited_d_matrix() -> dict:
    """Read the #406 D_matrix.json; returns the dict (used for JS inheritance).

    Plan §11 assumption 3 + §4.2: the 16×16 inherited sub-grid of JS is lifted
    BYTE-IDENTICAL from #406; this keeps the paired-bootstrap valid.
    """
    if not I406_D_MATRIX.exists():
        raise FileNotFoundError(
            f"{I406_D_MATRIX} missing; #406 inheritance requires it. "
            "Sync from HF or re-run i406 Phase 1."
        )
    return json.loads(I406_D_MATRIX.read_text())


def _assert_i406_recipe_match(d: dict) -> None:
    """Raise loudly if #406's D_matrix recipe doesn't match what #488 computes.

    The 16×16 sub-grid lifted from #406 must use the SAME JS estimator config
    as the 11×* cells computed fresh by this script — otherwise the
    paired-bootstrap is invalid because the inherited and new cells were
    measured on different recipes (silent units mismatch).

    Hard-checks (raise on drift):
    * ``n_probes`` matches ``N_PROBES`` (#488 uses load_q_test_extended_50 = 50
      probes; #406 must too).
    * Inherited JS block has exactly the 16 INHERITED_CIDS as row keys (i.e.
      the A/B/C/D classes); a different set means a different condition recipe.
    * ``schema_version`` is present (sanity).

    Soft-warns (log but don't raise — these are advisory because #406's
    D_matrix doesn't record them all):
    * ``k_target`` / ``k_available_per_probe`` (sampling K mismatch).

    See plan §11 assumption 3: "#406 D_matrix is byte-identical-recipe with
    #488 for the 16×16 sub-grid".
    """
    schema = d.get("schema_version")
    if schema is None:
        raise AssertionError(
            "i406 D_matrix.json has no schema_version; refusing to lift cells "
            "without a recipe sentinel. Re-export #406 with provenance."
        )
    n_probes_inh = d.get("n_probes")
    if n_probes_inh != N_PROBES:
        raise AssertionError(
            f"i406 D_matrix n_probes={n_probes_inh!r} != #488 N_PROBES={N_PROBES}. "
            "The inherited 16×16 JS sub-grid was computed on a different probe "
            "count; paired-bootstrap with newly-computed 11×* cells would mix "
            "scales. Either re-run #406 with n_probes=50 or do NOT lift."
        )
    js_block_cids = set(d.get("JS", {}).keys())
    if js_block_cids != set(INHERITED_CIDS):
        missing = INHERITED_CIDS - js_block_cids
        extra = js_block_cids - INHERITED_CIDS
        raise AssertionError(
            f"i406 D_matrix JS block cids {sorted(js_block_cids)} != "
            f"INHERITED_CIDS {sorted(INHERITED_CIDS)} (missing={sorted(missing)}, "
            f"extra={sorted(extra)}). Condition recipe drift; refusing to lift."
        )
    # Soft warns
    for soft_key in ("k_target", "k_available_per_probe"):
        if soft_key not in d:
            logger.warning(
                "i406 D_matrix has no %s field; cannot verify K-sample match. "
                "Proceeding under plan §11 assumption 3.",
                soft_key,
            )
    logger.info(
        "i406 recipe-match check PASSED (schema=%s, n_probes=%d, 16 cids).",
        schema,
        n_probes_inh,
    )


def _seed_js_kl_from_i406(js_matrix: dict, kl_matrix: dict) -> tuple[dict, dict]:
    """Copy the inherited 16×16 JS sub-grid (and KL directions if available)
    into the output matrices.

    The #406 D_matrix.json carries a ``JS`` and a ``KL`` block keyed by cid.
    `JS[i][j]` is the symmetric JS scalar; for KL the inherited D_matrix only
    records one direction in some schema versions — we copy what's there and
    leave the orthogonal direction None for the analyzer to fill (or pull
    fresh in a follow-up pass). The new (NEW × *) and (* × NEW) cells are
    initialized to None.

    Before lifting, asserts the #406 recipe matches #488's (n_probes + cid
    set + schema). Raises if drift, so silent recipe mismatch can't
    contaminate the paired-bootstrap.
    """
    d = _load_inherited_d_matrix()
    _assert_i406_recipe_match(d)
    inherited = d.get("JS", {})
    kl_inh = d.get("KL", {})

    # Initialize 27×27 grids with None.
    cids = [c.cid for c in CONDITIONS]
    for ci in cids:
        js_matrix.setdefault(ci, {})
        kl_matrix.setdefault(ci, {})
        for cj in cids:
            js_matrix[ci].setdefault(cj, None)
            kl_matrix[ci].setdefault(cj, {"kl_ij": None, "kl_ji": None, "sym_kl": None})

    # Lift inherited 16×16 (only when both endpoints are inherited).
    lifted = 0
    for ci in INHERITED_CIDS:
        if ci not in inherited:
            continue
        for cj in INHERITED_CIDS:
            if cj not in inherited[ci]:
                continue
            v = inherited[ci][cj]
            if v is not None:
                js_matrix[ci][cj] = float(v)
                lifted += 1
        if ci in kl_inh:
            for cj in INHERITED_CIDS:
                v = kl_inh[ci].get(cj)
                if v is not None:
                    kl_matrix[ci][cj]["sym_kl"] = float(v)
    logger.info("Lifted %d inherited JS cells from #406 D_matrix.json", lifted)
    return js_matrix, kl_matrix


# ── Stylization score ────────────────────────────────────────────────────


def _baseline_prompt(tokenizer, q: str) -> str:
    """No-system-prompt default-assistant chat-template path (matches B1)."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": q}],
        tokenize=False,
        add_generation_prompt=True,
    )


# ── Cosine on residual-stream activations ─────────────────────────────────


def _last_token_residuals(
    model, tokenizer, prompt_text: str, layers: tuple[int, ...]
) -> dict[int, torch.Tensor]:
    """Read the residual stream at the LAST input token across `layers`.

    Returns ``{layer: (D,) tensor on CPU/float32}``.
    """
    inputs = tokenizer(prompt_text, return_tensors="pt", padding=False).to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    last_pos = inputs["input_ids"].shape[1] - 1
    # outputs.hidden_states is a tuple of length n_layers+1 (embeddings + per-layer outputs).
    # Layer indexing convention here: layer k = hidden_states[k] (so layer 0 = embedding,
    # layer 28 = final for a 28-layer Qwen-2.5-7B).
    out: dict[int, torch.Tensor] = {}
    for L in layers:
        if len(outputs.hidden_states) <= L:
            raise IndexError(
                f"Requested layer {L} but model has only {len(outputs.hidden_states)} "
                "hidden-state outputs (including embeddings at index 0)."
            )
        out[L] = outputs.hidden_states[L][0, last_pos, :].float().cpu()
    return out


def _cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cosine similarity between two 1-D vectors; numerically safe."""
    eps = 1e-8
    na = a.norm().clamp_min(eps)
    nb = b.norm().clamp_min(eps)
    return float((a @ b / (na * nb)).item())


# ── Driver ───────────────────────────────────────────────────────────────


def _atomic_write_json(path: Path, payload: dict) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2))
    tmp.replace(path)


def _write_js_kl_checkpoint(
    js_path: Path,
    kl_path: Path,
    js_matrix: dict,
    kl_matrix: dict,
    cids: list[str],
    n_probes: int,
    r_samples: int,
) -> None:
    """Persist the current JS+KL matrices atomically (per CLAUDE.md per-phase)."""
    _atomic_write_json(
        js_path,
        {
            "schema_version": "i488_v1",
            "conditions": cids,
            "JS": js_matrix,
            "n_probes": n_probes,
            "r_samples": r_samples,
        },
    )
    _atomic_write_json(
        kl_path,
        {
            "schema_version": "i488_v1",
            "conditions": cids,
            "KL": kl_matrix,
            "n_probes": n_probes,
            "r_samples": r_samples,
        },
    )


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - CLI dispatch loop
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(name)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    )
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gpu-id", type=int, default=0)
    ap.add_argument(
        "--skip",
        nargs="+",
        default=[],
        choices=["js", "kl", "cossim", "stylization"],
        help="Sub-passes to skip (smoke / debug).",
    )
    ap.add_argument(
        "--r-samples",
        type=int,
        default=R_SAMPLES,
        help="Per-probe response samples (default 8 per metrics rule).",
    )
    ap.add_argument(
        "--n-probes",
        type=int,
        default=N_PROBES,
        help="Number of Q'_probe questions (default 50 = full set).",
    )
    ap.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Per-sample response token cap for the RB estimator (default 256).",
    )
    ap.add_argument(
        "--hf-tf-batch",
        type=int,
        default=HF_TEACHER_FORCE_BATCH,
        help=(
            "Per-call micro-batch size for HF teacher-force forwards. Lower if "
            "Pass B OOMs on long-response cells."
        ),
    )
    ap.add_argument(
        "--vllm-mem-util",
        type=float,
        default=VLLM_GPU_MEM_UTILIZATION,
        help=(
            "vLLM gpu_memory_utilization for Pass A. Default 0.40 to leave room "
            "for HF in Pass B; can be raised to 0.85 with --no-hf if running "
            "only sampling."
        ),
    )
    ap.add_argument(
        "--pairs",
        nargs="+",
        default=None,
        help=(
            "Optional whitelist of (i,j) cells like `A1:G2 G2:A1`. Used by the "
            "parallel dispatcher (i488_phase1_parallel.sh) to shard the pending "
            "JS/KL workload across multiple GPUs, and for smoke-test on a tiny "
            "slice. With this flag, the shard only computes cells in the list "
            "whose JS is still None (resume-safe)."
        ),
    )
    ap.add_argument(
        "--out-suffix",
        type=str,
        default="",
        help=(
            "Suffix appended to every output file (e.g. `_g0`). Used by the "
            "parallel dispatcher to keep per-shard outputs separate "
            "(js_matrix_g0.json, kl_matrix_g0.json, ...). Empty (default) "
            "preserves byte-identical legacy behavior."
        ),
    )
    ap.add_argument(
        "--print-pending-pairs",
        action="store_true",
        help=(
            "Print the pending (ci:cj) pairs to stdout (one per line) and exit "
            "without loading the model. Used by the parallel dispatcher to "
            "shard work across GPUs. Honors --out-suffix so the pending list "
            "reflects the shard's own checkpoint state if any."
        ),
    )
    args = ap.parse_args(argv)
    suffix = args.out_suffix

    # `setdefault` so the shell-set CUDA_VISIBLE_DEVICES (from the parallel
    # dispatcher fanning shards out across GPUs) is respected. Single-GPU
    # default: CVD unset → set to args.gpu_id. Parallel dispatcher: each shard
    # pre-exports CVD=<i> and passes --gpu-id 0; the pre-export wins here so
    # the shard is pinned to its assigned GPU. An unconditional overwrite
    # defeats the fan-out (every shard would see GPU 0).
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))

    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Probes: subset Q_test_extended_50.
    probes_all = load_q_test_extended_50()
    probes = probes_all[: args.n_probes]
    class_d_rewrites = load_class_d_rewrites()

    cids = [c.cid for c in CONDITIONS]

    # ── --print-pending-pairs short-circuit ─────────────────────────────────
    # Used by the parallel dispatcher to discover the work-list deterministically
    # before any GPU is allocated. Honors --out-suffix so a resumed shard sees
    # its own prior progress.
    if args.print_pending_pairs:
        js_path_q = OUT_DIR / f"js_matrix{suffix}.json"
        kl_path_q = OUT_DIR / f"kl_matrix{suffix}.json"
        js_matrix: dict[str, dict[str, float | None]] = (
            json.loads(js_path_q.read_text())["JS"]
            if js_path_q.exists() and js_path_q.stat().st_size > 0
            else {}
        )
        kl_matrix: dict[str, dict[str, dict]] = (
            json.loads(kl_path_q.read_text())["KL"]
            if kl_path_q.exists() and kl_path_q.stat().st_size > 0
            else {}
        )
        js_matrix, _ = _seed_js_kl_from_i406(js_matrix, kl_matrix)
        for ci in cids:
            for cj in cids:
                if ci == cj:
                    continue
                if js_matrix[ci][cj] is None:
                    print(f"{ci}:{cj}")
        return 0

    # ── Persist is_stylized_source up front (independent of model loads) ──
    is_stylized_path = OUT_DIR / f"is_stylized_source{suffix}.json"
    is_stylized = {c.cid: int(c.cid in STRONG_STYLIZED_SOURCES) for c in CONDITIONS}
    _atomic_write_json(
        is_stylized_path,
        {
            "schema_version": "i488_v1",
            "strong_stylized_sources": sorted(STRONG_STYLIZED_SOURCES),
            "is_stylized_source": is_stylized,
        },
    )
    logger.info("Wrote %s", is_stylized_path)

    # Load tokenizer always; model loads happen lazily per-pass.
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    do_js = "js" not in args.skip
    do_kl = "kl" not in args.skip
    do_stylization = "stylization" not in args.skip
    do_cossim = "cossim" not in args.skip
    # JS and KL share the same vLLM+HF pipeline; if either is requested, we run
    # the full RB estimator (the JS/KL split is a write-time concern, not a
    # compute-time one).
    js_kl_run = do_js or do_kl

    # ── JS+KL Pass A (vLLM sampling) and Pass B (HF teacher-force) ──
    js_path = OUT_DIR / f"js_matrix{suffix}.json"
    kl_path = OUT_DIR / f"kl_matrix{suffix}.json"

    cell_results: dict[tuple[str, str], dict] = {}

    if js_kl_run:
        js_matrix: dict[str, dict[str, float | None]] = (
            json.loads(js_path.read_text())["JS"]
            if js_path.exists() and js_path.stat().st_size > 0
            else {}
        )
        kl_matrix: dict[str, dict[str, dict]] = (
            json.loads(kl_path.read_text())["KL"]
            if kl_path.exists() and kl_path.stat().st_size > 0
            else {}
        )
        js_matrix, kl_matrix = _seed_js_kl_from_i406(js_matrix, kl_matrix)

        # Identify pending cells.
        pending: list[tuple[str, str]] = []
        if args.pairs:
            for pair in args.pairs:
                i, j = pair.split(":")
                if js_matrix.get(i, {}).get(j) is None:
                    pending.append((i, j))
        else:
            for ci in cids:
                for cj in cids:
                    if ci == cj:
                        continue
                    if js_matrix[ci][cj] is None:
                        pending.append((ci, cj))
        logger.info(
            "JS/KL: %d pending pairs (out of 27×26 = 702)%s",
            len(pending),
            f" [shard suffix={suffix!r}]" if suffix else "",
        )

        if pending:
            # ── Pass A: vLLM batched sampling for ALL pending cells ──
            logger.info(
                "Pass A (vLLM sampling): loading %s with gpu_memory_utilization=%.2f, "
                "max_model_len=%d",
                BASE_MODEL,
                args.vllm_mem_util,
                VLLM_MAX_MODEL_LEN,
            )
            from vllm import LLM

            llm = LLM(
                model=BASE_MODEL,
                dtype="bfloat16",
                gpu_memory_utilization=args.vllm_mem_util,
                seed=42,
                max_model_len=VLLM_MAX_MODEL_LEN,
            )

            try:
                # Per-cell sampling. Each cell = ONE vLLM call (continuous-batched
                # across 2 × n_probes prompts with n=r_samples).
                cell_samples: dict[tuple[str, str], dict] = {}
                for idx, (ci, cj) in enumerate(pending):
                    samples = _vllm_sample_cell_responses(
                        llm,
                        tokenizer,
                        CONDITIONS_BY_ID[ci],
                        CONDITIONS_BY_ID[cj],
                        probes,
                        class_d_rewrites,
                        args.r_samples,
                        args.max_new_tokens,
                        # Deterministic per-cell seed so re-runs of the same cell
                        # produce identical responses (resume-safe).
                        seed=42 + hash((ci, cj)) % 100000,
                    )
                    cell_samples[(ci, cj)] = samples
                    if (idx + 1) % 10 == 0 or idx == len(pending) - 1:
                        logger.info(
                            "Pass A progress: %d/%d cells sampled (last (%s,%s))",
                            idx + 1,
                            len(pending),
                            ci,
                            cj,
                        )
            finally:
                logger.info("Pass A done; tearing down vLLM + reaping workers")
                del llm
                gc.collect()
                try:
                    from vllm.distributed.parallel_state import (
                        destroy_distributed_environment,
                        destroy_model_parallel,
                    )

                    destroy_model_parallel()
                    destroy_distributed_environment()
                except Exception as e:
                    logger.warning("vLLM destroy failed (non-fatal): %s", e)
                kill_vllm_workers(logger)

            # ── Pass B: HF batched teacher-forcing for each cell ──
            logger.info("Pass B (HF teacher-force): loading %s in bf16", BASE_MODEL)
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                trust_remote_code=True,
            )
            model.eval()

            try:
                for idx, (ci, cj) in enumerate(pending):
                    samples = cell_samples[(ci, cj)]
                    cell = _compute_cell_divergences_from_samples(
                        model,
                        tokenizer,
                        samples["prompts_i"],
                        samples["prompts_j"],
                        samples["samples_from_i"],
                        samples["samples_from_j"],
                        args.hf_tf_batch,
                    )
                    cell_results[(ci, cj)] = cell
                    js_matrix[ci][cj] = cell["js"]
                    kl_matrix[ci][cj] = {
                        "kl_ij": cell["kl_ij"],
                        "kl_ji": cell["kl_ji"],
                        "sym_kl": cell["sym_kl"],
                    }
                    # Per-cell checkpoint write — Pass B crash loses ≤ 1 cell.
                    _write_js_kl_checkpoint(
                        js_path,
                        kl_path,
                        js_matrix,
                        kl_matrix,
                        cids,
                        args.n_probes,
                        args.r_samples,
                    )
                    if (idx + 1) % 5 == 0 or idx == len(pending) - 1:
                        logger.info(
                            "Pass B progress: %d/%d cells; last (%s,%s) js=%.4f sym_kl=%.4f",
                            idx + 1,
                            len(pending),
                            ci,
                            cj,
                            cell["js"],
                            cell["sym_kl"],
                        )
                # Free the sampled-responses dict now that all cells are processed.
                cell_samples.clear()
            finally:
                # Keep HF model loaded if cossim / stylization need it.
                if not (do_cossim or do_stylization):
                    del model
                    gc.collect()
                    torch.cuda.empty_cache()
                    model = None
                else:
                    pass
        else:
            model = None  # type: ignore[assignment]
    else:
        model = None  # type: ignore[assignment]

    # ── stylization_score pass ──
    # Uses the SAME RB JS estimator vs the no-system baseline. We need vLLM for
    # generation and HF for teacher-force; reuse the existing HF model if Pass B
    # left it loaded, otherwise (re-)load both. For the smoke path that asks
    # only stylization+cossim (--skip js kl), we still need a vLLM pass for
    # sampling — but it's per-condition (27 conds), not per-pair (702 cells).
    if do_stylization:
        style_path = OUT_DIR / f"stylization_score{suffix}.json"
        existing = (
            json.loads(style_path.read_text()).get("stylization_score", {})
            if style_path.exists() and style_path.stat().st_size > 0
            else {}
        )
        scores: dict[str, float] = dict(existing)
        missing_conds = [c for c in CONDITIONS if c.cid not in scores]

        if missing_conds:
            # Sample with vLLM if not already loaded.
            from vllm import LLM, SamplingParams

            logger.info("Stylization Pass A: loading vLLM for %d conds", len(missing_conds))
            llm = LLM(
                model=BASE_MODEL,
                dtype="bfloat16",
                gpu_memory_utilization=args.vllm_mem_util,
                seed=42,
                max_model_len=VLLM_MAX_MODEL_LEN,
            )
            sp = SamplingParams(
                n=args.r_samples,
                temperature=1.0,
                top_p=1.0,
                max_tokens=args.max_new_tokens,
                seed=42,
            )
            stylization_samples: dict[str, dict] = {}
            try:
                for c in missing_conds:
                    prompts_i = [
                        build_prompt_for_condition(c, q, tokenizer, class_d_rewrites)
                        for q in probes
                    ]
                    prompts_ref = [_baseline_prompt(tokenizer, q) for q in probes]
                    # Special case: if cond already IS the default assistant (B1/C1),
                    # JS = 0 by construction.
                    if all(pi == pr for pi, pr in zip(prompts_i, prompts_ref, strict=True)):
                        scores[c.cid] = 0.0
                        _atomic_write_json(
                            style_path,
                            {
                                "schema_version": "i488_v1",
                                "conditions": cids,
                                "stylization_score": scores,
                                "baseline": "no-system-prompt default-assistant chat-template",
                                "n_probes": args.n_probes,
                                "r_samples": args.r_samples,
                            },
                        )
                        logger.info("stylization_score[%s] = 0.0 (== baseline)", c.cid)
                        continue
                    outs = llm.generate(prompts_i + prompts_ref, sp)
                    n = len(probes)
                    s_i = [
                        list(outs[k].outputs[r].token_ids)
                        for k in range(n)
                        for r in range(args.r_samples)
                    ]
                    s_r = [
                        list(outs[n + k].outputs[r].token_ids)
                        for k in range(n)
                        for r in range(args.r_samples)
                    ]
                    stylization_samples[c.cid] = {
                        "prompts_i": prompts_i,
                        "prompts_ref": prompts_ref,
                        "samples_from_i": [
                            [list(outs[k].outputs[r].token_ids) for r in range(args.r_samples)]
                            for k in range(n)
                        ],
                        "samples_from_ref": [
                            [list(outs[n + k].outputs[r].token_ids) for r in range(args.r_samples)]
                            for k in range(n)
                        ],
                    }
                    # Free references not kept.
                    del s_i, s_r
            finally:
                logger.info("Stylization Pass A done; tearing down vLLM")
                del llm
                gc.collect()
                try:
                    from vllm.distributed.parallel_state import (
                        destroy_distributed_environment,
                        destroy_model_parallel,
                    )

                    destroy_model_parallel()
                    destroy_distributed_environment()
                except Exception as e:
                    logger.warning("vLLM destroy failed (non-fatal): %s", e)
                kill_vllm_workers(logger)

            # HF Pass B for stylization. Reuse model if already loaded.
            if model is None:
                logger.info("Stylization Pass B: loading HF model")
                model = AutoModelForCausalLM.from_pretrained(
                    BASE_MODEL,
                    torch_dtype=torch.bfloat16,
                    device_map={"": 0},
                    trust_remote_code=True,
                )
                model.eval()

            for c in missing_conds:
                if c.cid in scores:  # already set (default-baseline match)
                    continue
                ss = stylization_samples[c.cid]
                cell = _compute_cell_divergences_from_samples(
                    model,
                    tokenizer,
                    ss["prompts_i"],
                    ss["prompts_ref"],
                    ss["samples_from_i"],
                    ss["samples_from_ref"],
                    args.hf_tf_batch,
                )
                scores[c.cid] = cell["js"]
                _atomic_write_json(
                    style_path,
                    {
                        "schema_version": "i488_v1",
                        "conditions": cids,
                        "stylization_score": scores,
                        "baseline": "no-system-prompt default-assistant chat-template",
                        "n_probes": args.n_probes,
                        "r_samples": args.r_samples,
                    },
                )
                logger.info("stylization_score[%s] = %.4f", c.cid, scores[c.cid])
            stylization_samples.clear()

    # ── cosine layer sweep ──
    if do_cossim:
        if model is None:
            logger.info("Cossim: loading HF model")
            model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL,
                torch_dtype=torch.bfloat16,
                device_map={"": 0},
                trust_remote_code=True,
            )
            model.eval()
        # One residual-vector per source (mean over probes), then cosine matrix per layer.
        residuals_by_layer_by_cid: dict[int, dict[str, torch.Tensor]] = {
            L: {} for L in COSINE_LAYERS
        }
        for c in CONDITIONS:
            per_probe_residuals: dict[int, list[torch.Tensor]] = {L: [] for L in COSINE_LAYERS}
            for q in probes:
                prompt = build_prompt_for_condition(c, q, tokenizer, class_d_rewrites)
                res = _last_token_residuals(model, tokenizer, prompt, COSINE_LAYERS)
                for L in COSINE_LAYERS:
                    per_probe_residuals[L].append(res[L])
            for L in COSINE_LAYERS:
                residuals_by_layer_by_cid[L][c.cid] = torch.stack(per_probe_residuals[L]).mean(
                    dim=0
                )
            logger.info("cosine: residuals for %s collected", c.cid)

        for L in COSINE_LAYERS:
            mat: dict[str, dict[str, float]] = {}
            for ci in cids:
                mat[ci] = {}
                for cj in cids:
                    mat[ci][cj] = _cosine(
                        residuals_by_layer_by_cid[L][ci],
                        residuals_by_layer_by_cid[L][cj],
                    )
            _atomic_write_json(
                OUT_DIR / f"cossim_matrix_layer{L}{suffix}.json",
                {
                    "schema_version": "i488_v1",
                    "conditions": cids,
                    "layer": L,
                    "recipe": "persona-vectors (a) last-input-token; mean over probes",
                    "n_probes": args.n_probes,
                    "cossim": mat,
                },
            )
            logger.info("Wrote cossim layer %d matrix", L)

    if model is not None:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    logger.info("Phase 1 done. Outputs in %s", OUT_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
