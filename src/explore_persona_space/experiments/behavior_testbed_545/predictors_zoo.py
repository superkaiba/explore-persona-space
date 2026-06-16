# ruff: noqa: RUF002, RUF003
# Intentional Unicode (≪, ※, ‖, ½, ∩, ∅) in scientific docstrings/comments.
"""Issue #545 metric-race — expanded Group-A predictor zoo (plan §4.1 + §4.2).

THE single manipulated variable of the `full-metric-race-per-family`
follow-up: Group-A geometry grows from v1's 96 raw-centroid variants
({cosine, neg_l2, projection}) to the full #493 metric zoo (centroid +
covariance-aware + cloud + output-distribution JS/KL) over a FROZEN v1
leakage matrix. NO retraining; NO matrix recompute. The #493 metric engine
(`scripts/issue493_extraction_metric_bakeoff.py`) + the #540 JS-RB harness
(`js_canonical`) are IMPORTED, never re-implemented.

Two phases, cleanly split (CLAUDE.md "CPU-only phases don't hold GPU pods"):

- **GPU** (`extract_clouds_and_outdist_gpu`): build per-behavior activation
  CLOUDS (row demos = 8-point per-demo cloud; row nl = ≤8-point per-elicitation
  cloud; column = ≥50-point per-probe cloud) → ``clouds.npz``; build the
  JS/KL output-distribution per-(row,col,flavor) RB estimates (#540 Phase S/T
  discipline: vLLM sample, HF teacher-force, GPU-resident per-position reduce)
  → ``outdist/*.json``. Both checkpoint-per-unit.
- **CPU** (`build_zoo_predictors`): read ``clouds.npz`` + ``outdist/*.json``
  and emit the ``A__*.json`` predictor files in the v1 schema the frozen
  ``scoring.py`` consumes unchanged.

Serialization contract (Nit N1): a metric that is N/A for a cell (degenerate
cloud, single-text flavor, unpaired ``delta_spec``, <4-point cloud) is
serialized by OMITTING that cell's key from the ``cells`` dict — NEVER
``"cell": null`` — because ``scoring.py::weighted_kendall_tau`` gates inclusion
on ``c in pred`` (a literal null would crash / corrupt τ).
"""

from __future__ import annotations

import os

# PyTorch CUDA allocator: expandable_segments defragments reserved-but-
# unallocated memory, the canonical mitigation for the round-3 extract-phase
# OOM where HF (22 GiB) + vLLM (0.60 × 79 ≈ 47.5 GiB) co-reside with only
# ~9.5 GiB working-memory headroom and intermediate tensors (log_softmax)
# fragment the free pool. MUST be set BEFORE the first `import torch` — torch
# reads PYTORCH_CUDA_ALLOC_CONF once when its CUDA allocator initializes. torch
# is imported lazily inside extract_clouds_and_outdist_gpu, so this module-top
# setdefault is guaranteed to run first. setdefault (not assignment) so an
# explicit launcher / env override always wins. (#545 round-4.)
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import json
import logging
import sys
from pathlib import Path

import numpy as np

from . import (
    BASE_MODEL,
    reproducibility_metadata,
)
from .columns import COLUMNS, column_applies
from .eval_battery import battery_probes
from .predictors import (
    EXTRACTION_POINTS,
    FLAVORS,
    GEOMETRY_LAYERS,
    NL_DESCRIPTIONS,
    _demo_messages,
    _mean_hidden_states,
)
from .rows import active_rows

logger = logging.getLogger(__name__)

ROWS = active_rows()

# Import the #493 metric engine (the validated bake-off code — IMPORTED, never
# re-implemented). The script lives under scripts/, not the package, so add it
# to sys.path the same way the #540 dispatcher does.
_SCRIPTS_DIR = Path(__file__).resolve().parents[4] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue493_extraction_metric_bakeoff as i493  # noqa: E402

# JS/KL canonical RB math (#540 / persona-distance-metrics.md).
from explore_persona_space.analysis import js_canonical as jsc  # noqa: E402

# --- metric-race hyperparameters (plan §10/§11; #493 defaults) -------------
PCA_K = 16  # #493 dual-Gram PCA subspace top-k (n≪d-safe)
CENTROID_METRICS_NEW = ("euclidean", "mahal", "mahal_pooled_ctx")
CENTROID_METRICS_V1 = ("cosine", "neg_l2", "projection")  # carried reference
CLOUD_METRICS = ("mmd", "c2st", "wass2", "gauss_kl")  # delta_spec handled separately
CENTERINGS = ("raw", "centered")
MIN_CLOUD_POINTS = 4  # <4-point clouds record None (plan §4.1)

# JS/KL output-distribution (#540 RB estimator; persona-distance-metrics.md).
JS_R_SAMPLES = 8
JS_TEMP = 1.0
JS_MAX_NEW_TOKENS = 1024
JS_N_PROBES = 50
JS_MAX_SEQ_LEN = 4096
JS_DIRECTIONS = ("js", "kl_narrow_broad", "kl_broad_narrow")
# HF teacher-force sub-batch in outdist (jsc.teacher_forced_response_logps).
# Lowered from the upstream default 16 → 4: the HF forward materializes a
# (B, L, V) logits transient where V≈152k for Qwen-2.5-7B; at B=16 the bf16
# transient is ~5.0 GiB (matching the 5.35 GiB OOM observed on pod-545 in r4),
# at B=4 it is ~1.3 GiB — fits the ~9.5 GiB headroom the gpu_memory_utilization
# =0.60 config leaves once the HF model + vLLM engine co-reside. The OOM is
# HF-side (large-vocab logits), NOT vLLM, so this — not vLLM util — is the lever.
JS_TF_MAX_BATCH = 4

# vLLM GPU memory utilization for the lazily-loaded engine inside
# extract_clouds_and_outdist_gpu. CRITICAL: the HF base model
# (AutoModelForCausalLM) and the vLLM engine CO-RESIDE on the same GPU in the
# SAME process — the clouds sub-phase elicits nl-cloud text via vLLM and then
# teacher-forces it through the HF model, and the outdist sub-phase does the
# same per probe pair, so neither model can be freed before the other. vLLM
# reads FREE-memory-at-startup and rejects init if its requested fraction
# exceeds it.
#
# Two OOM regimes the dial must clear (both observed on pod-545):
#   - round-1 (init OOM): 0.85 × 79.18 = 67.3 GiB requested > ~63 GiB free with
#     the HF model resident → vLLM engine-init rejected ~80s after launch.
#   - round-3 (extract-phase OOM): with vLLM at 0.70 (56.99 GiB observed) and
#     the HF model resident at 22.0 GiB (MEASURED, not the 16 GiB estimated in
#     the r3 brief — model weights + activations + the KV cache HF keeps during
#     output_hidden_states teacher-forcing on max_model_len=4096 sequences),
#     total HF + vLLM = ~79 GiB on a 79.18 GiB H100 → only 206 MiB free, and the
#     extract phase's intermediate tensors (log_softmax etc.) OOM'd mid-run.
#
# r4 fix: drop to 0.60 so 22.0 (HF) + 47.5 (vLLM = 0.60 × 79.18) = 69.5 GiB,
# leaving ~9.5 GiB working-memory headroom for the extract-phase intermediates.
# Paired with PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True (set at module
# top, below) to defragment reserved-but-unallocated memory — the canonical
# PyTorch CUDA OOM mitigation the round-3 error message itself recommended.
JS_GPU_MEM_UTIL = 0.60
# Measured resident size of the co-resident HF base model under realistic
# extract-phase conditions (output_hidden_states + max_model_len=4096 + the KV
# cache during a single teacher-force) — pod-545 round-3 OOM log: HF process
# held 21.98 GiB. NOT the parameter-count estimate (16 GiB) the r3 brief used.
JS_HF_MODEL_RESIDENT_GIB = 22.0
# An H100-80 reports ~79.18 GiB total.
_H100_TOTAL_GIB = 79.18
# Pre-vLLM-init free-memory floor (GiB), pinned to the actual util so a future
# util change keeps the assert correct: floor = vLLM request + 2.5 GiB margin.
# At 0.60: 47.5 + 2.5 = 50.0 GiB. The floor must clear the vLLM request with
# margin so a regression that fails to leave headroom (e.g. an HF-model size
# blow-up, another GPU consumer) fails LOUD at the assert with a clear message
# rather than as an opaque vLLM engine-init OOM.
JS_VLLM_PREINIT_MIN_FREE_GIB = max(JS_GPU_MEM_UTIL * _H100_TOTAL_GIB + 2.5, 0.0)


# ---------------------------------------------------------------------------
# Behavior-context construction (the JS/KL "behavior-b context")
# ---------------------------------------------------------------------------


def _behavior_context_messages(behavior_id: str, flavor: str) -> list[dict]:
    """The conditioning prefix that puts the base model "in" behavior b.

    - ``nl``: a single system turn describing the behavior tendency (the
      Group-A nl-flavor elicitation framing — plan §4.1).
    - ``demos``: the K=8 demo turns prepended as few-shot user/assistant
      pairs (the demo-flavor conditioning).

    Returns a chat-message prefix (system-prompt persona injection per
    CLAUDE.md; never user/assistant for the persona itself).
    """
    if flavor == "nl":
        nl = NL_DESCRIPTIONS.get(behavior_id)
        if not nl:
            return []
        return [{"role": "system", "content": f"The assistant has a tendency: {nl}."}]
    if flavor == "demos":
        try:
            return _demo_messages(behavior_id)
        except FileNotFoundError:
            return []
    raise KeyError(flavor)


def _column_probe_texts(column_id: str, cap: int) -> list[str]:
    return [p["question"] for p in battery_probes(COLUMNS[column_id], cap=cap)]


# ---------------------------------------------------------------------------
# GPU phase: clouds + JS/KL output-distribution
# ---------------------------------------------------------------------------


def _row_cloud_texts(row_id: str) -> dict[str, list[str]]:
    """Per-flavor text lists for a ROW's cloud (NOT concatenated — plan §4.1).

    - ``demos``: the K=8 demo answers as 8 SEPARATE texts → 8-point cloud
      (NOT v1's single concatenated demo_text).
    - ``nl``: empty here — the nl cloud is built from base-model temp-1
      elicitation SAMPLES (see ``extract_clouds_and_outdist_gpu``), not from
      a single description string.
    """
    out: dict[str, list[str]] = {}
    try:
        demos = _demo_messages(row_id)
        answers = [m["content"] for m in demos if m["role"] == "assistant"][:8]
        if answers:
            out["demos"] = answers
    except FileNotFoundError:
        logger.warning("zoo: no demos for row %s (demos cloud skipped)", row_id)
    return out


def extract_clouds_and_outdist_gpu(  # noqa: C901 — one model load shared across the cloud + JS/KL grid
    out_dir: Path,
    *,
    device: str = "cuda:0",
    rows_subset: list[str] | None = None,
    cols_subset: list[str] | None = None,
    n_probes: int = JS_N_PROBES,
    r_samples: int = JS_R_SAMPLES,
    nl_cloud_samples: int = 8,
    skip_outdist: bool = False,
    skip_clouds: bool = False,
) -> dict:
    """Build per-behavior activation clouds (→ ``clouds.npz``) + JS/KL
    output-distribution RB estimates (→ ``outdist/*.json``) on the BASE model.

    The cell subset (``rows_subset`` / ``cols_subset``) threads the smoke =
    sweep parameterization (smoke IS the full grid restricted to a tiny cell
    list). ``n_probes`` / ``r_samples`` / ``nl_cloud_samples`` are the
    descope ladder knobs (plan §9).

    Returns a small summary dict (counts + JS truncation manipulation check).
    """

    import gc

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir.mkdir(parents=True, exist_ok=True)
    outdist_dir = out_dir / "outdist"
    outdist_dir.mkdir(parents=True, exist_ok=True)

    rows = rows_subset or list(ROWS.keys())
    cols = cols_subset or [c for c, col in COLUMNS.items() if col.scoring_eligible]

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True
    )
    model.eval()

    # vLLM lazily loaded only for the nl-cloud elicitation + JS/KL sampling.
    llm = None
    sampling_params = None

    def _get_llm():
        nonlocal llm, sampling_params
        if llm is None:
            from vllm import LLM, SamplingParams

            # The HF model is resident on the GPU here (clouds + outdist both
            # teacher-force through it; ~22 GiB MEASURED, see
            # JS_HF_MODEL_RESIDENT_GIB), so vLLM init reads a REDUCED free-memory
            # figure. Log it + the vLLM request, then assert we clear the request
            # with margin — a free-memory shortfall fails LOUD here instead of as
            # an opaque vLLM engine-init OOM (#545 round-1). The util is also kept
            # at 0.60 (not 0.70) so the extract-phase intermediates (log_softmax)
            # have ~9.5 GiB working-memory headroom after HF + vLLM co-residency,
            # which the round-3 extract-phase OOM proved 0.70 lacked (pod-545).
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            free_gib = free_bytes / (1024**3)
            total_gib = total_bytes / (1024**3)
            requested_gib = JS_GPU_MEM_UTIL * total_gib
            logger.info(
                "[phase=outdist] pre-vLLM-init GPU memory: free=%.1f GiB / total=%.1f GiB; "
                "HF-model-resident=%.1f GiB; vLLM will request %.1f GiB "
                "(gpu_memory_utilization=%.2f)",
                free_gib,
                total_gib,
                torch.cuda.memory_allocated() / (1024**3),
                requested_gib,
                JS_GPU_MEM_UTIL,
            )
            assert free_gib >= JS_VLLM_PREINIT_MIN_FREE_GIB, (
                f"vLLM pre-init free GPU memory {free_gib:.1f} GiB < floor "
                f"{JS_VLLM_PREINIT_MIN_FREE_GIB:.1f} GiB (vLLM will request "
                f"{requested_gib:.1f} GiB at gpu_memory_utilization={JS_GPU_MEM_UTIL}). "
                "The HF base model likely was not the only GPU consumer or has grown; "
                "lower JS_GPU_MEM_UTIL or free other GPU processes."
            )
            llm = LLM(
                model=BASE_MODEL,
                dtype="bfloat16",
                max_model_len=JS_MAX_SEQ_LEN,
                gpu_memory_utilization=JS_GPU_MEM_UTIL,
            )
            sampling_params = SamplingParams(
                n=r_samples, temperature=JS_TEMP, top_p=1.0, max_tokens=JS_MAX_NEW_TOKENS, seed=545
            )
        return llm, sampling_params

    summary: dict = {"rows": rows, "cols": cols, "n_probes": n_probes, "r_samples": r_samples}

    # --- 1. Activation clouds → clouds.npz --------------------------------
    if not skip_clouds:
        clouds: dict[str, np.ndarray] = {}
        # Row demos clouds (8-point per-demo).
        for row_id in rows:
            texts = _row_cloud_texts(row_id)
            for flavor, demo_texts in texts.items():
                reps = _mean_hidden_states(
                    model, tokenizer, demo_texts, device, retain_per_sample_reps=True
                )
                for layer in GEOMETRY_LAYERS:
                    for point in EXTRACTION_POINTS:
                        t = reps.get(layer, {}).get(point)
                        if t is not None:
                            clouds[f"row|{row_id}|{flavor}|{layer}|{point}"] = t.numpy()
            logger.info("[phase=clouds] row demos cloud %s", row_id)
        # Row nl clouds: ≤nl_cloud_samples base-model temp-1 responses under
        # the nl-description conditioning context.
        nl_rows = [r for r in rows if NL_DESCRIPTIONS.get(r)]
        if nl_rows:
            vllm, _ = _get_llm()
            from vllm import SamplingParams
            from vllm.inputs import TokensPrompt

            nl_sp = SamplingParams(
                n=nl_cloud_samples,
                temperature=JS_TEMP,
                top_p=1.0,
                max_tokens=256,
                seed=545,
            )
            # One generic neutral probe per row to elicit on-policy nl text.
            generic_q = "Can you help me with something?"
            reqs, req_rows = [], []
            for row_id in nl_rows:
                msgs = [
                    *_behavior_context_messages(row_id, "nl"),
                    {"role": "user", "content": generic_q},
                ]
                text = tokenizer.apply_chat_template(
                    msgs, tokenize=False, add_generation_prompt=True
                )
                reqs.append(
                    TokensPrompt(prompt_token_ids=tokenizer.encode(text, add_special_tokens=False))
                )
                req_rows.append(row_id)
            outs = vllm.generate(reqs, nl_sp)
            for row_id, out in zip(req_rows, outs, strict=True):
                samples = [c.text for c in out.outputs if c.text.strip()]
                if not samples:
                    continue
                reps = _mean_hidden_states(
                    model, tokenizer, samples, device, retain_per_sample_reps=True
                )
                for layer in GEOMETRY_LAYERS:
                    for point in EXTRACTION_POINTS:
                        t = reps.get(layer, {}).get(point)
                        if t is not None:
                            clouds[f"row|{row_id}|nl|{layer}|{point}"] = t.numpy()
                logger.info("[phase=clouds] row nl cloud %s (%d samples)", row_id, len(samples))
        # Column probe clouds (≥n_probes per-probe).
        for col_id in cols:
            probes = _column_probe_texts(col_id, cap=n_probes)
            if not probes:
                continue
            reps = _mean_hidden_states(
                model, tokenizer, probes, device, retain_per_sample_reps=True
            )
            for layer in GEOMETRY_LAYERS:
                for point in EXTRACTION_POINTS:
                    t = reps.get(layer, {}).get(point)
                    if t is not None:
                        clouds[f"col|{col_id}|probe|{layer}|{point}"] = t.numpy()
            logger.info("[phase=clouds] col probe cloud %s", col_id)
        np.savez_compressed(out_dir / "clouds.npz", **clouds)
        summary["n_cloud_arrays"] = len(clouds)
        logger.info("[phase=clouds] wrote clouds.npz (%d arrays)", len(clouds))

    # --- 2. JS/KL output-distribution → outdist/*.json --------------------
    if not skip_outdist:
        trunc_total, trunc_hits = 0, 0
        n_outdist = 0
        for flavor in FLAVORS:
            for row_id in rows:
                ctx_row = _behavior_context_messages(row_id, flavor)
                if not ctx_row:
                    continue
                for col_id in cols:
                    if not column_applies(COLUMNS[col_id], ROWS[row_id]):
                        continue
                    out_path = outdist_dir / f"{row_id}__{col_id}__{flavor}.json"
                    if out_path.exists():
                        continue
                    # The "behavior-b′" partner context = the COLUMN's diagonal
                    # behavior, or fall back to the bare assistant (default).
                    ctx_col = _column_behavior_context(col_id, flavor)
                    res = _score_outdist_pair(
                        model,
                        tokenizer,
                        _get_llm,
                        ctx_row,
                        ctx_col,
                        col_id,
                        n_probes=n_probes,
                        r_samples=r_samples,
                    )
                    if res is None:
                        continue
                    trunc_total += res["_trunc_total"]
                    trunc_hits += res["_trunc_hits"]
                    payload = {
                        "row": row_id,
                        "col": col_id,
                        "flavor": flavor,
                        "rb": {k: v for k, v in res.items() if not k.startswith("_")},
                        "metadata": reproducibility_metadata(),
                    }
                    out_path.write_text(json.dumps(payload, indent=1))
                    n_outdist += 1
                    logger.info("[phase=outdist] %s__%s__%s", row_id, col_id, flavor)
        summary["n_outdist_pairs"] = n_outdist
        summary["js_truncation_rate"] = (trunc_hits / trunc_total) if trunc_total else 0.0
        logger.info(
            "[phase=outdist] %d pairs, truncation=%.4f (manipulation check, #548)",
            n_outdist,
            summary["js_truncation_rate"],
        )

    # Explicit GPU teardown of the co-resident HF model + vLLM engine before the
    # function returns (the CPU build_zoo_predictors phase runs next in-process).
    # Both models held the GPU simultaneously; drop the Python refs, force GC, and
    # release cached blocks so a following GPU consumer in the same process starts
    # clean. (#545: the co-residency itself was the OOM cause — see JS_GPU_MEM_UTIL.)
    alloc_before = torch.cuda.memory_allocated() / (1024**3)
    del model
    del tokenizer
    if llm is not None:
        del llm
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    logger.info(
        "[phase=extract] GPU teardown: memory_allocated %.1f GiB -> %.1f GiB",
        alloc_before,
        torch.cuda.memory_allocated() / (1024**3),
    )
    (out_dir / "extract_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def _column_behavior_context(col_id: str, flavor: str) -> list[dict]:
    """The partner ("behavior-b′") context for a column: the diagonal row's
    behavior context where one exists, else the bare default assistant."""
    for row in ROWS.values():
        if row.diagonal_column == col_id:
            ctx = _behavior_context_messages(row.row_id, flavor)
            if ctx:
                return ctx
    return []  # bare default assistant (no system persona)


def _score_outdist_pair(
    model,
    tokenizer,
    get_llm,
    ctx_a: list[dict],
    ctx_b: list[dict],
    col_id: str,
    *,
    n_probes: int,
    r_samples: int,
) -> dict | None:
    """RB sequence-level JS + both KL directions for one (behavior-b ctx,
    behavior-b′ ctx) pair over the column's probe set.

    Reuses the #540 discipline: vLLM sample R temp-1 responses from BOTH
    sides, HF teacher-force each through BOTH conditioned models, exact
    full-vocab per-position divergence (GPU-resident reduce; only the
    per-sample scalar means leave the GPU), aggregate with
    ``jsc.rb_pair_estimate``. Returns the RB dict + truncation counters
    (``_trunc_total`` / ``_trunc_hits``), or None when the column has no
    probes.
    """
    from vllm.inputs import TokensPrompt

    probes = _column_probe_texts(col_id, cap=n_probes)
    if not probes:
        return None
    vllm, sp = get_llm()

    def _prompt_ids(ctx: list[dict], q: str) -> list[int]:
        msgs = [*ctx, {"role": "user", "content": q}]
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        return tokenizer.encode(text, add_special_tokens=False)

    prompts_a = [_prompt_ids(ctx_a, q) for q in probes]
    prompts_b = [_prompt_ids(ctx_b, q) for q in probes]

    # Sample R responses from each side (per probe).
    def _sample(prompts: list[list[int]]) -> tuple[list[list[list[int]]], int, int]:
        reqs = [TokensPrompt(prompt_token_ids=p) for p in prompts]
        outs = vllm.generate(reqs, sp)
        per_probe: list[list[list[int]]] = []
        t_total = t_hit = 0
        for out in outs:
            rows = []
            for comp in out.outputs:
                ids, _action = jsc.apply_terminator_rule(list(comp.token_ids), comp.finish_reason)
                rows.append(ids)
                t_total += 1
                t_hit += int(comp.finish_reason == "length")
            per_probe.append(rows)
        return per_probe, t_total, t_hit

    samples_a, ta_tot, ta_hit = _sample(prompts_a)
    samples_b, tb_tot, tb_hit = _sample(prompts_b)

    a_kl_m, b_kl_m, a_kl_ab, b_kl_ba = [], [], [], []
    for pi in range(len(probes)):
        rows_a = samples_a[pi][:r_samples]
        rows_b = samples_b[pi][:r_samples]
        if not rows_a or not rows_b:
            continue
        responses = rows_a + rows_b
        lp_a = jsc.teacher_forced_response_logps(
            model, prompts_a[pi], responses, max_batch=JS_TF_MAX_BATCH
        )
        lp_b = jsc.teacher_forced_response_logps(
            model, prompts_b[pi], responses, max_batch=JS_TF_MAX_BATCH
        )
        na = len(rows_a)
        for i in range(len(responses)):
            side = "a" if i < na else "b"
            lp_side = lp_a[i] if side == "a" else lp_b[i]
            lp_other = lp_b[i] if side == "a" else lp_a[i]
            pd = jsc.per_position_divergences(lp_side, lp_other)
            (a_kl_m if side == "a" else b_kl_m).append(float(pd.kl_side_m_bits.mean()))
            (a_kl_ab if side == "a" else b_kl_ba).append(float(pd.kl_side_other_nats.mean()))
    if not a_kl_m or not b_kl_m:
        return None
    rb = jsc.rb_pair_estimate(
        np.array(a_kl_m), np.array(b_kl_m), np.array(a_kl_ab), np.array(b_kl_ba)
    )
    rb["_trunc_total"] = ta_tot + tb_tot
    rb["_trunc_hits"] = ta_hit + tb_hit
    return rb


# ---------------------------------------------------------------------------
# CPU phase: build A__*.json predictor files from cached clouds + outdist
# ---------------------------------------------------------------------------


def _load_clouds(out_dir: Path) -> dict[str, np.ndarray]:
    npz = out_dir / "clouds.npz"
    if not npz.exists():
        raise FileNotFoundError(f"{npz} missing — run extract_clouds_and_outdist_gpu first")
    with np.load(npz) as z:
        return {k: z[k] for k in z.files}


def _finite_cloud(arr: np.ndarray) -> np.ndarray | None:
    """Drop non-finite rows; return None if <MIN_CLOUD_POINTS remain or the
    cloud is constant (the #493 ``_finite_and_non_constant`` precedent)."""
    if arr is None or arr.ndim != 2:
        return None
    mask = np.all(np.isfinite(arr), axis=1)
    arr = arr[mask]
    if len(arr) < MIN_CLOUD_POINTS:
        return None
    if np.allclose(arr.std(axis=0), 0.0):
        return None
    return arr


def _write_predictor(out_dir: Path, name: str, cells: dict[str, float], extra: dict) -> Path | None:
    """Write one A__<name>.json in the v1 schema. OMITS None cells (Nit N1).
    Returns None (no file) if no cell scored."""
    cells = {k: float(v) for k, v in cells.items() if v is not None and np.isfinite(v)}
    if not cells:
        return None
    p = out_dir / f"A__{name}.json"
    p.write_text(
        json.dumps(
            {
                "group": "A",
                "name": name,
                "track": "shift",
                "cells": cells,
                "n_cells": len(cells),
                "metadata": reproducibility_metadata(),
                **extra,
            },
            indent=1,
        )
    )
    return p


def build_zoo_predictors(  # noqa: C901 — flat metric grid, one branch per metric family
    pred_dir: Path,
    cloud_src_dir: Path | None = None,
) -> list[Path]:
    """CPU: read ``clouds.npz`` + ``outdist/*.json`` and emit A__* predictor
    JSONs (centroid + covariance-aware + cloud + JS/KL) in the v1 schema.

    ``pred_dir`` is where the JSONs land (``predictors_metric_race/``);
    ``cloud_src_dir`` is where ``clouds.npz`` + ``outdist/`` live (defaults to
    ``pred_dir.parent`` — the metric_race root).
    """
    src = cloud_src_dir or pred_dir.parent
    pred_dir.mkdir(parents=True, exist_ok=True)
    clouds = _load_clouds(src)
    written: list[Path] = []

    def _row_cloud(row_id, flavor, layer, point):
        return clouds.get(f"row|{row_id}|{flavor}|{layer}|{point}")

    def _col_cloud(col_id, layer, point):
        return clouds.get(f"col|{col_id}|probe|{layer}|{point}")

    # ---- centroid metrics (raw + centered) × layers × points × flavors ----
    for flavor in FLAVORS:
        for layer in GEOMETRY_LAYERS:
            for point in EXTRACTION_POINTS:
                # Pre-build the context-pooled-cov state once per (layer, point)
                # over ALL in-scope column clouds (mahal_pooled_ctx).
                col_centroids = []
                col_order = []
                for col_id, col in COLUMNS.items():
                    if not col.scoring_eligible:
                        continue
                    cc = _finite_cloud(_col_cloud(col_id, layer, point))
                    if cc is not None:
                        col_centroids.append(cc.mean(axis=0))
                        col_order.append(col_id)
                pooled_state = None
                if len(col_centroids) >= 2:
                    # (n_cond, n_q=1, H) shape for #493 pooled-cov builder.
                    acts = np.array(col_centroids)[:, None, :]
                    pooled_state = i493._build_context_pooled_mahal_state(acts, PCA_K)
                    if pooled_state is None:
                        i493._pop_pooled_failure_reason(acts)
                pooled_idx = {c: i for i, c in enumerate(col_order)}

                # Centering: subtract the pooled mean over in-scope column
                # centroids (the #536 bank-centering lesson) from BOTH sides.
                center_vec = np.array(col_centroids).mean(axis=0) if col_centroids else None

                for centering in CENTERINGS:
                    for metric in CENTROID_METRICS_V1 + CENTROID_METRICS_NEW:
                        # v1 raw {cosine,neg_l2,projection} already exist as the
                        # geom_* reference predictors; the zoo re-emits them
                        # under the centroid-zoo naming for the leaderboard
                        # grouping, plus the centered variant.
                        name = f"cloud_{flavor}_L{layer}_{point}_{centering}_{metric}"
                        cells = {}
                        for row_id, row in ROWS.items():
                            rc = _finite_cloud(_row_cloud(row_id, flavor, layer, point))
                            if rc is None:
                                continue
                            for col_id in col_order:
                                if not column_applies(COLUMNS[col_id], row):
                                    continue
                                cc = _finite_cloud(_col_cloud(col_id, layer, point))
                                if cc is None:
                                    continue
                                s = _centroid_metric(
                                    metric,
                                    rc,
                                    cc,
                                    centering,
                                    center_vec,
                                    pooled_state,
                                    pooled_idx.get(col_id),
                                )
                                if s is not None and np.isfinite(s):
                                    cells[f"{row_id}|{col_id}"] = s
                        p = _write_predictor(
                            pred_dir,
                            name,
                            cells,
                            {
                                "metric": metric,
                                "centering": centering,
                                "layer": layer,
                                "point": point,
                                "flavor": flavor,
                                "in_scope_columns": col_order,
                            },
                        )
                        if p:
                            written.append(p)

    # ---- cloud metrics (mmd, c2st, wass2, gauss_kl) ------------------------
    for flavor in FLAVORS:
        for layer in GEOMETRY_LAYERS:
            for point in EXTRACTION_POINTS:
                for metric in CLOUD_METRICS:
                    name = f"cloud_{flavor}_L{layer}_{point}_{metric}"
                    cells = {}
                    for row_id, row in ROWS.items():
                        rc = _finite_cloud(_row_cloud(row_id, flavor, layer, point))
                        if rc is None:
                            continue
                        for col_id, col in COLUMNS.items():
                            if not col.scoring_eligible or not column_applies(col, row):
                                continue
                            cc = _finite_cloud(_col_cloud(col_id, layer, point))
                            if cc is None:
                                continue
                            s = _cloud_metric(metric, rc, cc)
                            if s is not None and np.isfinite(s):
                                cells[f"{row_id}|{col_id}"] = s
                    p = _write_predictor(
                        pred_dir,
                        name,
                        cells,
                        {"metric": metric, "layer": layer, "point": point, "flavor": flavor},
                    )
                    if p:
                        written.append(p)

    # ---- output-distribution JS/KL (layer-agnostic) -----------------------
    outdist_dir = src / "outdist"
    if outdist_dir.exists():
        for direction in JS_DIRECTIONS:
            for flavor in FLAVORS:
                name = f"outdist_{flavor}_{direction}"
                cells = {}
                for jpath in sorted(outdist_dir.glob("*.json")):
                    d = json.loads(jpath.read_text())
                    if d["flavor"] != flavor:
                        continue
                    rb = d["rb"]
                    if direction == "js":
                        # similarity polarity: 1 - JS (higher = closer)
                        v = 1.0 - rb.get("js_rb_bits", float("nan"))
                    elif direction == "kl_narrow_broad":
                        v = rb.get("kl_ab_nats", float("nan"))
                    else:  # kl_broad_narrow
                        v = rb.get("kl_ba_nats", float("nan"))
                    if v is not None and np.isfinite(v):
                        cells[f"{d['row']}|{d['col']}"] = float(v)
                p = _write_predictor(
                    pred_dir, name, cells, {"direction": direction, "flavor": flavor}
                )
                if p:
                    written.append(p)

    logger.info("[phase=zoo] wrote %d A__* predictor JSONs to %s", len(written), pred_dir)
    return written


# ---------------------------------------------------------------------------
# Metric dispatch helpers (import #493 functions; never re-implement)
# ---------------------------------------------------------------------------


def _apply_centering(rc: np.ndarray, cc: np.ndarray, centering: str, center_vec):
    if centering == "raw" or center_vec is None:
        return rc, cc
    return rc - center_vec[None, :], cc - center_vec[None, :]


def _centroid_metric(metric, rc, cc, centering, center_vec, pooled_state, col_pool_idx):
    """One centroid / covariance-aware metric over two clouds (centroid =
    cloud mean). Returns a similarity-polarity-consistent scalar or None."""
    rc_c, cc_c = _apply_centering(rc, cc, centering, center_vec)
    if metric == "cosine":
        return 1.0 - i493._centroid_cosine_distance(rc_c, cc_c)  # similarity
    if metric == "neg_l2":
        return -i493._centroid_euclidean(rc_c, cc_c)
    if metric == "projection":
        a, b = rc_c.mean(axis=0), cc_c.mean(axis=0)
        nb = np.linalg.norm(b)
        return float(a @ b / nb) if nb > 1e-12 else None
    if metric == "euclidean":
        return -i493._centroid_euclidean(rc_c, cc_c)  # negate → similarity
    if metric == "mahal":
        v = i493._centroid_mahal(rc_c, cc_c, PCA_K)
        return -v if np.isfinite(v) else None  # negate → similarity
    if metric == "mahal_pooled_ctx":
        # mahal_pooled_ctx is defined relative to the pooled-context cov state
        # built over the (centered/raw) COLUMN centroid bank. Only valid in the
        # 'raw' centering (the pooled state is built on raw centroids).
        if centering != "raw" or pooled_state is None or col_pool_idx is None:
            return None
        # Mahalanobis of the row centroid vs the column centroid under pooled cov.
        mu, comps, cov_inv = pooled_state["mu"], pooled_state["components"], pooled_state["cov_inv"]
        ra = rc.mean(axis=0)
        cb = cc.mean(axis=0)
        ya = (ra - mu[0]) @ comps.T
        yb = (cb - mu[0]) @ comps.T
        diff = ya - yb
        v = float(np.sqrt(max(0.0, float(diff @ cov_inv @ diff))))
        return -v if np.isfinite(v) else None
    raise KeyError(metric)


def _cloud_metric(metric, rc, cc):
    """One distributional cloud metric over two N-point clouds. Returns a
    similarity-polarity-consistent scalar or None (degenerate)."""
    if metric == "mmd":
        v = i493._rbf_mmd_squared(rc, cc)
        return -v if np.isfinite(v) else None  # distance → negate to similarity
    if metric == "c2st":
        v = i493._c2st_auc(rc, cc)
        return -v if np.isfinite(v) else None  # 2|AUC-.5| distance → similarity
    if metric == "wass2":
        v = i493._bures_wasserstein2(rc, cc, PCA_K)
        return -v if np.isfinite(v) else None
    if metric == "gauss_kl":
        v = i493._gaussian_sym_kl_in_subspace(rc, cc, PCA_K)
        return -v if np.isfinite(v) else None
    raise KeyError(metric)


def build_delta_spec_predictors(pred_dir: Path, cloud_src_dir: Path | None = None) -> list[Path]:
    """``delta_spec`` — PAIRED only (plan §4.1 Blocker-1 fix, choice A).

    ``_delta_spectrum`` is well-defined only between matched/paired clouds
    drawn from the SAME probe IDs in the SAME order. The row demo/nl clouds
    and the column probe clouds are NOT paired (different item sets), so
    ``delta_spec`` is N/A for every (row, col) cell in this design and records
    NO scores — the predictor file is intentionally NOT written (omit-the-key
    contract at the file level). Kept as a named function so the design intent
    + the runtime assertion are explicit and code-reviewable: a future paired
    construction would build the paired clouds, assert ``probe_ids_a ==
    probe_ids_b`` (same IDs, same order) BEFORE the spectrum subtraction, and
    HALT on mismatch.
    """
    logger.info(
        "[phase=zoo] delta_spec: no paired-probe construction in this design — "
        "N/A for all cells (no predictor file written; plan §4.1 Blocker-1)"
    )
    return []
