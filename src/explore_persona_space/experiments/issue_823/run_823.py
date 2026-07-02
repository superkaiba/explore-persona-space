"""Issue #823 — Does the per-context map h predict the model's own output or context-side
processing?

Phases
------
Phase 0   (VM, CPU)  : Verify HF substrate, reconstruct prompts, alignment gate
Phase 0.5 (GPU pod)  : Regenerate Qwen own answers via vLLM (arm A')
Phase 1   (VM, CPU)  : Generate Sonnet answers via dispatch_calls (arms B1, B2)
Phase 2   (VM, CPU)  : Build derangement permutation (arm C text)
Phase 3   (GPU pod)  : Teacher-forced activation extraction (arms A', B1, B2, C)
Phase 4   (VM, CPU)  : Ridge refitting — per-arm refit (DV1) + cross-arm transfer (DV3)
Phase 5   (VM, CPU)  : Validity diagnostics

GPU phases (0.5, 3) run on a provisioned pod; CPU phases (0,1,2,4,5) run on the VM.
Use --phase to select which phase(s) to run; --smoke reduces n_contexts to 10.

Sentinel contract (Phase 0.5+3 only):
  /workspace/logs/issue-823-phase05-done.json  → Phase 0.5 complete
  /workspace/logs/issue-823-phase3-smoke.json  → Phase 3 smoke PASS
  /workspace/logs/issue-823-phase3-done.json   → Phase 3 full extraction complete
  /workspace/logs/issue-823-epm_results.json   → Final sentinel (GCP poller reads)
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import logging
import math
import os
import pathlib
import sys
import time
from typing import Any

import numpy as np
import torch
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("issue823")

# ── constants ──────────────────────────────────────────────────────────────────
HF_DATA_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO_DATA = "superkaiba1/explore-persona-space-data"
BUNDLE_REPO_REVISION = "c94070508aa1c1f9c015ceb072231a2e51b28b3f"
BUNDLE_PATH_IN_REPO = "issue779_monitoring/analysis_tensors/pass_b/train_context_vectors.pt"
BUNDLE_SHA256 = "46c06e89c513ca598bc83be1c87689694a47bfc927a81d0d738a54df769dbf9a"

LMSYS_REVISION = "200748d9d3cddcc9d782887541057aca0b18c5da"
N_CONTEXTS_FULL = 5000
N_SMOKE = 10

DEFAULT_MODEL = "Qwen/Qwen2.5-7B-Instruct"
EXPECTED_LAYERS = 28
EXPECTED_HIDDEN = 3584
GENERATION_SUFFIX = "<|im_start|>assistant\n"

TRAITS = ("evil", "sycophancy", "hallucination")
# Pre-registered from #779 (frozen before this experiment, no data-driven selection)
READ_OUT_LAYERS = {"evil": 14, "sycophancy": 26, "hallucination": 17}

SONNET_MODEL = "claude-sonnet-4-5-20250929"
# Matched to pass_b's generation cap (issue779_collect.py:558 SamplingParams max_tokens=1024)
# so Sonnet answer lengths are comparable to Qwen's own (length-nuisance control, plan §5).
SONNET_MAX_TOKENS = 1024
SENTINEL_SCHEMA_VERSION = 1

# HF upload slug
ISSUE_SLUG = "issue823_own_vs_external"

# ── helpers ────────────────────────────────────────────────────────────────────


def _json_np(o: Any):
    """json.dumps default= converter for numpy scalars/arrays (np.bool_, np.floating, ...)."""
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(f"Object of type {o.__class__.__name__} is not JSON serializable")


def log_phase(name: str) -> None:
    """Emit a [phase=...] log line for poll_pipeline.py."""
    logger.info("[phase=%s]", name)


def write_sentinel(path: pathlib.Path, payload: dict[str, Any]) -> None:
    """Write a poll_pipeline-compatible sentinel file."""
    payload["sentinel_schema_version"] = SENTINEL_SCHEMA_VERSION
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=_json_np))
    logger.info("Sentinel written: %s", path)


def resolve_base_dir(args_base_dir: str | None) -> pathlib.Path:
    """Return the base dir for all outputs."""
    if args_base_dir:
        return pathlib.Path(args_base_dir)
    # Prefer /workspace (pod) if it exists
    ws = pathlib.Path("/workspace")
    if ws.exists():
        return ws
    # Fall back to repo root / eval output
    return pathlib.Path(__file__).resolve().parents[5]  # repo root


def sha256_file(path: pathlib.Path) -> str:
    """Compute SHA256 of a file in chunks."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 0: Verify HF substrate + reconstruct prompts + alignment gate
# ═══════════════════════════════════════════════════════════════════════════════


def phase0_verify(base_dir: pathlib.Path, n_contexts: int, smoke: bool) -> dict:
    """Download bundle, verify sha256, reconstruct prompts. Returns verify record."""
    from huggingface_hub import snapshot_download

    log_phase("p0_substrate")
    logger.info("Phase 0: Verify HF substrate (n_contexts=%d)", n_contexts)

    # Step 0a: download bundle
    logger.info("Downloading pass_b bundle (revision=%s)...", BUNDLE_REPO_REVISION)
    local_dir = snapshot_download(
        repo_id=HF_DATA_REPO_DATA,
        repo_type="dataset",
        revision=BUNDLE_REPO_REVISION,
        allow_patterns=[BUNDLE_PATH_IN_REPO],
        local_dir=str(base_dir / "data" / "issue_823" / "hf_dl"),
    )
    bundle_path = pathlib.Path(local_dir) / BUNDLE_PATH_IN_REPO
    assert bundle_path.exists(), f"Bundle not found at {bundle_path}"

    # sha256 check
    logger.info("Computing sha256 of bundle...")
    actual_sha = sha256_file(bundle_path)
    sha256_ok = actual_sha == BUNDLE_SHA256
    if not sha256_ok:
        raise RuntimeError(f"Bundle sha256 mismatch: expected {BUNDLE_SHA256}, got {actual_sha}")
    logger.info("sha256 PASS: %s", actual_sha)

    # Load and verify shape
    bundle = torch.load(str(bundle_path), map_location="cpu", mmap=True)
    expected_keys = {"cx_last", "cx_mean", "v_x", "layers", "source", "metadata"}
    assert set(bundle.keys()) == expected_keys, (
        f"Bundle keys {set(bundle.keys())} != expected {expected_keys}"
    )
    assert bundle["v_x"].shape == (N_CONTEXTS_FULL, EXPECTED_LAYERS, EXPECTED_HIDDEN), (
        f"v_x shape {bundle['v_x'].shape}"
    )
    assert bundle["cx_last"].shape == (
        N_CONTEXTS_FULL,
        EXPECTED_LAYERS,
        EXPECTED_HIDDEN,
    ), f"cx_last shape {bundle['cx_last'].shape}"
    assert "prompts" not in bundle, "Unexpected 'prompts' key in bundle"
    logger.info("Bundle shape OK: cx_last %s, v_x %s", bundle["cx_last"].shape, bundle["v_x"].shape)

    # Step 0b: Reconstruct prompts from LMSYS-Chat-1M
    log_phase("p0_prompt_recon")
    logger.info("Reconstructing prompts from LMSYS-Chat-1M (revision=%s)...", LMSYS_REVISION)
    from datasets import load_dataset

    def first_user_turn(conv: dict) -> str:
        """Mirror issue779_collect.py _first_user_turn logic."""
        for msg in conv.get("conversation", []):
            if msg["role"] == "user":
                return msg["content"].strip()
        return ""

    ds = load_dataset(
        "lmsys/lmsys-chat-1m",
        split="train",
        streaming=True,
        revision=LMSYS_REVISION,
        token=True,
    )
    prompts: list[str] = []
    for row in ds:
        text = first_user_turn(row)
        if text:
            prompts.append(text)
        if len(prompts) == N_CONTEXTS_FULL:
            break

    assert len(prompts) == N_CONTEXTS_FULL, (
        f"Expected {N_CONTEXTS_FULL} prompts, got {len(prompts)}"
    )
    logger.info("Reconstructed %d prompts", len(prompts))

    # Persist prompts for downstream phases
    prompts_path = base_dir / "data" / "issue_823" / "prompts.json"
    prompts_path.parent.mkdir(parents=True, exist_ok=True)
    prompts_path.write_text(json.dumps(prompts, default=_json_np))
    logger.info("Prompts saved to %s", prompts_path)

    # Step 0c: Alignment gate (20 spot-check contexts)
    # NOTE: Full alignment gate requires a GPU. In smoke/CPU mode we skip the GPU check
    # and verify only that prompt reconstruction produced valid non-empty strings.
    # The alignment gate is performed in Phase 3's smoke check (which runs on GPU).
    spot_cosines: list[float] = []
    alignment_gate_result = "DEFERRED_TO_PHASE3_GPU"

    rng = np.random.default_rng(0)
    spot_idx = rng.choice(N_CONTEXTS_FULL, size=20, replace=False)
    # Validate prompts are non-empty at spot indices
    for i in spot_idx:
        assert len(prompts[i]) > 0, f"Empty prompt at index {i}"

    logger.info(
        "Alignment gate: %s (spot-check prompts non-empty for all 20 indices)",
        alignment_gate_result,
    )

    # Save bundle path for later phases
    bundle_path_file = base_dir / "data" / "issue_823" / "bundle_path.txt"
    bundle_path_file.write_text(str(bundle_path))

    verify_record = {
        "sha256_ok": sha256_ok,
        "actual_sha256": actual_sha,
        "expected_sha256": BUNDLE_SHA256,
        "key_set": sorted(bundle.keys()),
        "bundle_shape": {
            "cx_last": list(bundle["cx_last"].shape),
            "v_x": list(bundle["v_x"].shape),
        },
        "n_prompts": len(prompts),
        "alignment_gate_result": alignment_gate_result,
        "spot_cosines": spot_cosines,
        "spot_idx": spot_idx.tolist(),
        "bundle_revision": BUNDLE_REPO_REVISION,
        "lmsys_revision": LMSYS_REVISION,
        "smoke": smoke,
        "n_contexts": n_contexts,
        "ts": time.time(),
    }

    out_path = base_dir / "eval_results" / "issue_823" / "phase0_verify.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(verify_record, indent=2, default=_json_np))
    logger.info("Phase 0 complete: %s", out_path)
    return verify_record


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 0.5: Regenerate Qwen own answers via vLLM (GPU pod)
# ═══════════════════════════════════════════════════════════════════════════════


def phase05_vllm_regen(base_dir: pathlib.Path, n_contexts: int, smoke: bool) -> None:
    """Generate arm A' (Qwen own answers) via vLLM."""
    log_phase("p05_vllm_regen")
    logger.info("Phase 0.5: vLLM regeneration (n_contexts=%d, smoke=%s)", n_contexts, smoke)

    from vllm import LLM, SamplingParams

    # Load prompts
    prompts_path = base_dir / "data" / "issue_823" / "prompts.json"
    assert prompts_path.exists(), f"Prompts file not found: {prompts_path}. Run Phase 0 first."
    prompts: list[str] = json.loads(prompts_path.read_text())
    prompts = prompts[:n_contexts]
    logger.info("Loaded %d prompts", len(prompts))

    # Build chat-formatted prompts (no system prompt, bare context — same as #779 pass_b)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    formatted_prompts = []
    for p in prompts:
        messages = [{"role": "user", "content": p}]
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        formatted_prompts.append(text)
    logger.info("Formatted %d prompts for vLLM", len(formatted_prompts))

    # Init vLLM engine (same recipe as issue779_collect.py)
    chunk_size = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
    logger.info("Initializing vLLM engine (model=%s, chunk_size=%d)...", DEFAULT_MODEL, chunk_size)
    llm = LLM(
        model=DEFAULT_MODEL,
        dtype="bfloat16",
        max_model_len=8192,
        seed=42,
        trust_remote_code=True,
    )
    sampling_params = SamplingParams(
        n=1,
        temperature=1.0,
        top_p=0.95,
        max_tokens=1024,
        seed=42,
    )

    # Chunked generation (avoids vLLM large-batch deadlock — MEMORY.md gotcha)
    all_texts: list[str] = []
    all_n_tokens: list[int] = []
    for chunk_start in range(0, len(formatted_prompts), chunk_size):
        chunk_end = min(chunk_start + chunk_size, len(formatted_prompts))
        chunk = formatted_prompts[chunk_start:chunk_end]
        logger.info(
            "[vllm-chunk] generating chunk %d/%d (contexts %d-%d)",
            chunk_start // chunk_size + 1,
            math.ceil(len(formatted_prompts) / chunk_size),
            chunk_start,
            chunk_end - 1,
        )
        outputs = llm.generate(chunk, sampling_params, use_tqdm=False)
        for out in outputs:
            text = out.outputs[0].text
            n_tok = len(out.outputs[0].token_ids)
            all_texts.append(text)
            all_n_tokens.append(n_tok)

    assert len(all_texts) == len(prompts), f"Expected {len(prompts)} outputs, got {len(all_texts)}"

    # Build record
    records = [
        {
            "context_id": i,
            "question": prompts[i],
            "answer_text": all_texts[i],
            "n_tokens": all_n_tokens[i],
        }
        for i in range(len(prompts))
    ]

    out_path = base_dir / "raw_completions" / "phase05" / "arm_a_prime_seed42.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, default=_json_np))
    logger.info("Phase 0.5: saved %d arm A' answers to %s", len(records), out_path)

    # Write sentinel
    sentinel_path = pathlib.Path("/workspace/logs/issue-823-phase05-done.json")
    write_sentinel(
        sentinel_path,
        {
            "kind": "epm:progress",
            "version": 1,
            "note": f"Phase 0.5 complete: {len(records)} arm A' answers generated",
            "n_records": len(records),
            "smoke": smoke,
        },
    )
    log_phase("p05_done")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 1: Generate Sonnet answers (VM, CPU + API)
# ═══════════════════════════════════════════════════════════════════════════════


async def phase1_sonnet_gen(base_dir: pathlib.Path, n_contexts: int, smoke: bool) -> None:
    """Generate B1 (Sonnet-weird) and B2 (Sonnet-plain) via dispatch_calls."""
    log_phase("p1_sonnet_gen")
    logger.info("Phase 1: Sonnet generation (n_contexts=%d, smoke=%s)", n_contexts, smoke)

    from explore_persona_space.llm.api_dispatch import DispatchItem, dispatch_calls

    # Load prompts
    prompts_path = base_dir / "data" / "issue_823" / "prompts.json"
    assert prompts_path.exists(), f"Prompts file not found: {prompts_path}. Run Phase 0 first."
    prompts: list[str] = json.loads(prompts_path.read_text())
    prompts = prompts[:n_contexts]
    logger.info("Loaded %d prompts for Sonnet generation", len(prompts))

    # Build dispatch items for B2 (plain, no system prompt)
    items_b2 = [
        DispatchItem(
            item_id=f"b2_{i}",
            payload={"messages": [{"role": "user", "content": prompts[i]}], "system": None},
        )
        for i in range(len(prompts))
    ]

    # Build dispatch items for B1 (weird, with style instruction)
    b1_system = (
        "Respond in an unusual, stylistically eccentric way — use unexpected structure, "
        "mixed register, and non-standard formatting."
    )
    items_b1 = [
        DispatchItem(
            item_id=f"b1_{i}",
            payload={
                "messages": [{"role": "user", "content": prompts[i]}],
                "system": b1_system,
            },
        )
        for i in range(len(prompts))
    ]

    def _build_request(item: DispatchItem) -> dict:
        """Anthropic Messages.create params for one Sonnet generation call.

        NOTE (plan deviation, recorded): the Anthropic API exposes NO sampling
        seed — the b2 'seed 42' / b1 'seed 43' labels are file/record
        provenance only; text determinism comes from persisting the raw
        completions, not from the sampler.
        """
        params: dict = {
            "model": SONNET_MODEL,
            "max_tokens": SONNET_MAX_TOKENS,
            "messages": item.payload["messages"],
        }
        if item.payload.get("system"):
            params["system"] = item.payload["system"]
        return params

    def _parse_response(text: str) -> str:
        """Identity parse — the generation text IS the result."""
        return text

    all_items = items_b2 + items_b1
    logger.info(
        "Dispatching %d Sonnet calls (B2 x %d + B1 x %d)...",
        len(all_items),
        len(items_b2),
        len(items_b1),
    )

    checkpoint_dir = base_dir / "raw_completions" / "phase1" / "_checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    results = await dispatch_calls(
        all_items,
        model=SONNET_MODEL,
        build_request=_build_request,
        parse_response=_parse_response,
        max_attempts=5,
        checkpoint_dir=checkpoint_dir,
    )

    # Separate B1 and B2 results (results: {item_id: DispatchResult})
    b2_results: dict[int, str] = {}
    b1_results: dict[int, str] = {}
    failed: list[str] = []

    for item_id, res in results.items():
        if res is None or getattr(res, "error", False) or not isinstance(res.result, str):
            failed.append(item_id)
            continue
        text = res.result
        if item_id.startswith("b2_"):
            idx = int(item_id[3:])
            b2_results[idx] = text
        elif item_id.startswith("b1_"):
            idx = int(item_id[3:])
            b1_results[idx] = text

    logger.info(
        "Sonnet generation complete: B2=%d, B1=%d, failed=%d",
        len(b2_results),
        len(b1_results),
        len(failed),
    )

    # Drop rule: if > 50 contexts fail (< 99% fill), abort
    n_b2_ok = len(b2_results)
    if n_b2_ok < int(n_contexts * 0.99):
        raise RuntimeError(f"B2 fill rate below 99%: {n_b2_ok}/{n_contexts} contexts. Aborting.")

    # Build and persist B2 records
    b2_records = [
        {
            "context_id": i,
            "question": prompts[i],
            "answer_text": b2_results.get(i, ""),
            "arm": "b2_plain",
            "seed": 42,
            "filled": i in b2_results,
        }
        for i in range(len(prompts))
    ]
    b2_path = base_dir / "raw_completions" / "phase1" / "b2_seed42.json"
    b2_path.parent.mkdir(parents=True, exist_ok=True)
    b2_path.write_text(json.dumps(b2_records, indent=2, default=_json_np))
    logger.info("B2 answers saved to %s", b2_path)

    # Build and persist B1 records
    b1_records = [
        {
            "context_id": i,
            "question": prompts[i],
            "answer_text": b1_results.get(i, ""),
            "arm": "b1_weird",
            "seed": 43,
            "filled": i in b1_results,
        }
        for i in range(len(prompts))
    ]
    b1_path = base_dir / "raw_completions" / "phase1" / "b1_seed43.json"
    b1_path.write_text(json.dumps(b1_records, indent=2, default=_json_np))
    logger.info("B1 answers saved to %s", b1_path)

    log_phase("p1_done")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 2: Build derangement permutation
# ═══════════════════════════════════════════════════════════════════════════════


def phase2_derangement(base_dir: pathlib.Path, n_contexts: int, smoke: bool) -> None:
    """Construct fixed-point-free permutation π and build arm C texts."""
    log_phase("p2_derangement")
    logger.info("Phase 2: Derangement (n_contexts=%d)", n_contexts)

    # Load arm A' answers
    a_prime_path = base_dir / "raw_completions" / "phase05" / "arm_a_prime_seed42.json"
    assert a_prime_path.exists(), f"Arm A' file not found: {a_prime_path}. Run Phase 0.5 first."
    a_prime_records: list[dict] = json.loads(a_prime_path.read_text())
    a_prime_records = a_prime_records[:n_contexts]
    logger.info("Loaded %d arm A' records", len(a_prime_records))

    # Build derangement of {0, ..., n_contexts-1}
    rng = np.random.default_rng(42)
    perm = rng.permutation(n_contexts)
    # Swap any fixed points
    for i in range(n_contexts):
        if perm[i] == i:
            j = (i + 1) % n_contexts
            perm[i], perm[j] = perm[j], perm[i]
    assert all(perm[i] != i for i in range(n_contexts)), "Derangement check failed"
    logger.info("Derangement constructed (seed=42, n=%d)", n_contexts)

    # Build arm C records: context i gets arm A' answer from context π(i)
    c_records: dict[str, dict] = {}
    for i in range(n_contexts):
        src = int(perm[i])
        c_records[str(i)] = {
            "source_context": src,
            "answer_text": a_prime_records[src]["answer_text"],
        }

    out_path = base_dir / "raw_completions" / "phase2" / "derangement_seed42.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(
            {"permutation": perm.tolist(), "contexts": c_records}, indent=2, default=_json_np
        )
    )
    logger.info("Derangement saved to %s", out_path)
    log_phase("p2_done")


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 3: Teacher-forced activation extraction (GPU pod)
# ═══════════════════════════════════════════════════════════════════════════════


def _tf_extract_arm(  # noqa: C901 — single TF loop; splitting risks span/logp misalignment
    model,
    tokenizer,
    prompts: list[str],
    answers: list[str],
    layers: list[int],
    arm_name: str,
    a_prime_lengths: list[int] | None = None,
    batch_size: int = 8,
) -> tuple[np.ndarray, list[int], list[float]]:
    """Teacher-forced extraction for one arm. Returns (v_s, span_lengths, mean_logps).

    v_s shape: (n_contexts, n_layers, hidden_dim) float32.
    For B1/B2 arms, truncates response span to min(own_len, external_len) tokens.
    mean_logps: per-context mean log P of the (truncated) answer span under the
    base model — the plan §5 OOD covariate diagnostic (NaN for skipped contexts).
    """

    n = len(prompts)
    n_layers = len(layers)
    v_s = np.zeros((n, n_layers, EXPECTED_HIDDEN), dtype=np.float32)
    span_lengths: list[int] = []
    mean_logps: list[float] = [float("nan")] * n

    # Capture hooks
    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook(module, input, output):
            hidden = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hidden.detach().float().cpu()

        return hook

    handles = []
    for li, layer_idx in enumerate(layers):
        handle = model.model.layers[layer_idx].register_forward_hook(make_hook(li))
        handles.append(handle)

    model.eval()
    nan_count = 0
    with torch.no_grad():
        for batch_start in range(0, n, batch_size):
            batch_end = min(batch_start + batch_size, n)
            for ctx_i in range(batch_start, batch_end):
                prompt_text = prompts[ctx_i]
                answer_text = answers[ctx_i]

                if not answer_text:
                    # Dropped context — fill with zeros
                    span_lengths.append(0)
                    continue

                messages = [{"role": "user", "content": prompt_text}]
                prompt_only = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )

                # GENERATION_SUFFIX assert (same as issue779_collect.py:129-130)
                prompt_inputs = tokenizer(
                    prompt_only, return_tensors="pt", add_special_tokens=False
                ).to(model.device)
                suffix_decode = tokenizer.decode(prompt_inputs["input_ids"][0, -3:])
                assert suffix_decode == GENERATION_SUFFIX, (
                    f"[{arm_name}] ctx {ctx_i}: position assert failed: "
                    f"{suffix_decode!r} != {GENERATION_SUFFIX!r}"
                )
                prompt_len = prompt_inputs["input_ids"].shape[1]

                full_messages = [*messages, {"role": "assistant", "content": answer_text}]
                full_text = tokenizer.apply_chat_template(
                    full_messages, tokenize=False, add_generation_prompt=False
                )
                full_inputs = tokenizer(
                    full_text, return_tensors="pt", add_special_tokens=False
                ).to(model.device)
                full_len = full_inputs["input_ids"].shape[1]

                if full_len <= prompt_len:
                    logger.warning(
                        "[%s] ctx %d: empty response span (full_len=%d <= prompt_len=%d)",
                        arm_name,
                        ctx_i,
                        full_len,
                        prompt_len,
                    )
                    span_lengths.append(0)
                    continue

                resp_start = prompt_len
                resp_end = full_len

                # Length normalization for B1/B2: truncate to min(own_len, external_len)
                if a_prime_lengths is not None:
                    own_len = a_prime_lengths[ctx_i]
                    external_len = resp_end - resp_start
                    if own_len > 0:
                        trunc_len = min(own_len, external_len)
                        resp_end = resp_start + trunc_len
                    else:
                        trunc_len = external_len
                    span_lengths.append(trunc_len)
                else:
                    span_lengths.append(resp_end - resp_start)

                if span_lengths[-1] < 1:
                    logger.warning("[%s] ctx %d: span length < 1 after truncation", arm_name, ctx_i)
                    continue

                # Forward pass
                captured.clear()
                out = model(
                    input_ids=full_inputs["input_ids"],
                    attention_mask=full_inputs["attention_mask"],
                    output_hidden_states=False,  # using hooks instead
                )

                # OOD covariate: mean log P of the (truncated) answer span under
                # the base model. GPU-resident reduce; only the scalar moves to CPU.
                span_logits = out.logits[0, resp_start - 1 : resp_end - 1].float()
                span_targets = full_inputs["input_ids"][0, resp_start:resp_end]
                tok_lp = (
                    torch.log_softmax(span_logits, dim=-1)
                    .gather(1, span_targets.unsqueeze(1))
                    .squeeze(1)
                )
                mean_logps[ctx_i] = float(tok_lp.mean().item())
                del out, span_logits, tok_lp

                # Mean-pool over response token span [resp_start:resp_end]
                for li in range(n_layers):
                    if li not in captured:
                        continue
                    hs = captured[li]  # (1, seq_len, hidden)
                    span = hs[0, resp_start:resp_end, :]  # (span_len, hidden)
                    if span.shape[0] == 0:
                        continue
                    v_s[ctx_i, li, :] = span.mean(0).numpy()

            if (batch_start // batch_size) % 50 == 0:
                logger.info("[%s] Processed %d/%d contexts", arm_name, batch_end, n)

    for h in handles:
        h.remove()
    captured.clear()

    # Check NaN rate
    nan_mask = ~np.isfinite(v_s).all(axis=(1, 2))
    nan_count = nan_mask.sum()
    nan_rate = nan_count / n
    if nan_rate > 0.05:
        raise RuntimeError(f"[{arm_name}] NaN rate {nan_rate:.1%} > 5% — aborting (kill criterion)")
    if nan_count > 0:
        logger.warning(
            "[%s] %d contexts have NaN/Inf activations (%.1f%%)",
            arm_name,
            nan_count,
            nan_rate * 100,
        )

    return v_s, span_lengths, mean_logps


def phase3_tf_extract(base_dir: pathlib.Path, n_contexts: int, smoke: bool) -> None:
    """Teacher-forced extraction for arms A', B1, B2, C. GPU required."""
    log_phase("p3_tf_extract")
    logger.info("Phase 3: TF extraction (n_contexts=%d, smoke=%s)", n_contexts, smoke)

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    # Load prompts
    prompts_path = base_dir / "data" / "issue_823" / "prompts.json"
    assert prompts_path.exists(), f"Prompts not found: {prompts_path}"
    prompts: list[str] = json.loads(prompts_path.read_text())[:n_contexts]

    # Load arm texts
    a_prime_recs = json.loads(
        (base_dir / "raw_completions" / "phase05" / "arm_a_prime_seed42.json").read_text()
    )[:n_contexts]
    b2_recs = json.loads((base_dir / "raw_completions" / "phase1" / "b2_seed42.json").read_text())[
        :n_contexts
    ]
    b1_recs = json.loads((base_dir / "raw_completions" / "phase1" / "b1_seed43.json").read_text())[
        :n_contexts
    ]
    c_data = json.loads(
        (base_dir / "raw_completions" / "phase2" / "derangement_seed42.json").read_text()
    )

    a_prime_texts = [r["answer_text"] for r in a_prime_recs]
    b2_texts = [r["answer_text"] for r in b2_recs]
    b1_texts = [r["answer_text"] for r in b1_recs]
    c_texts = [c_data["contexts"][str(i)]["answer_text"] for i in range(n_contexts)]

    layers = list(range(EXPECTED_LAYERS))
    batch_size = int(os.environ.get("EPM_TF_BATCH_SIZE", "4" if smoke else "8"))

    # Load model
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("Loading model %s on %s...", DEFAULT_MODEL, device)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    model.eval()
    logger.info("Model loaded")

    # GPU alignment gate (Phase 0c) — check 20 spot-check contexts
    log_phase("p3_alignment_gate")
    logger.info("Phase 3: Running alignment gate on 20 spot-check contexts...")
    bundle_path = pathlib.Path(
        (base_dir / "data" / "issue_823" / "bundle_path.txt").read_text().strip()
    )
    bundle = torch.load(str(bundle_path), map_location="cpu", mmap=True)
    cx_last_bundle = bundle["cx_last"].numpy()  # (5000, 28, 3584)

    from scripts.issue779_collect import capture_context_vector  # type: ignore[import]

    rng = np.random.default_rng(0)
    spot_idx = rng.choice(N_CONTEXTS_FULL, size=20, replace=False)
    spot_cosines = []
    alignment_pass = True
    for i in spot_idx[: min(20, n_contexts)]:
        messages = [{"role": "user", "content": prompts[i]}]
        result = capture_context_vector(model, tokenizer, messages, layers)
        if result is None:
            logger.warning(
                "Alignment gate: context %d returned None from capture_context_vector", i
            )
            alignment_pass = False
            break
        recomputed = result["last"]  # (n_layers, hidden)
        bundle_row = cx_last_bundle[i]  # (n_layers, hidden)
        # Cosine per layer, then min
        cos_per_layer = []
        for li in range(EXPECTED_LAYERS):
            a = recomputed[li]
            b = bundle_row[li]
            cos = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
            cos_per_layer.append(cos)
        min_cos = min(cos_per_layer)
        spot_cosines.append(
            {"context_id": int(i), "min_cos": min_cos, "cos_by_layer": cos_per_layer}
        )
        if min_cos < 0.999:
            logger.warning("Alignment gate FAIL at context %d: min_cos=%.4f < 0.999", i, min_cos)
            alignment_pass = False

    if not alignment_pass:
        # Hard fail — no fallback. The plan requires a hard-fail when any spot-check
        # context has min_layer_cosine < 0.999.  A NN fallback would mask prompt-
        # reconstruction bugs and silently corrupt all arm activations downstream.
        failing = [r for r in spot_cosines if r["min_cos"] < 0.999]
        raise RuntimeError(
            f"Alignment gate HARD FAIL: {len(failing)} of {len(spot_cosines)} spot-check "
            f"contexts have min_layer_cosine < 0.999. "
            f"First failing: context_id={failing[0]['context_id']}, "
            f"min_cos={failing[0]['min_cos']:.6f}. "
            "Cannot proceed — LMSYS prompt reconstruction does not align with bundle. "
            "Check LMSYS revision pin (LMSYS_REVISION) and first_user_turn() logic."
        )
    else:
        logger.info("Alignment gate PASS: all %d spot checks cosine > 0.999", len(spot_cosines))

    # Persist smoke gate result
    smoke_json = {
        "alignment_gate_pass": alignment_pass,
        "spot_cosines_summary": [
            {"context_id": r["context_id"], "min_cos": r["min_cos"]} for r in spot_cosines
        ],
        "smoke": smoke,
        "n_contexts": n_contexts,
        "ts": time.time(),
    }
    smoke_sentinel = pathlib.Path("/workspace/logs/issue-823-phase3-smoke.json")
    write_sentinel(
        smoke_sentinel,
        {"kind": "epm:progress", "version": 1, "note": "Phase 3 smoke gate", **smoke_json},
    )
    logger.info("Phase 3 smoke sentinel written")

    # Compute arm A' token lengths (for length normalization)
    logger.info("Computing arm A' response token lengths...")
    a_prime_token_lengths: list[int] = []
    for i in range(n_contexts):
        if a_prime_texts[i]:
            toks = tokenizer(a_prime_texts[i], add_special_tokens=False)["input_ids"]
            a_prime_token_lengths.append(len(toks))
        else:
            a_prime_token_lengths.append(0)

    # Extract all arms sequentially to bound HBM
    span_lengths_by_arm: dict[str, list[int]] = {}
    answer_logp_by_arm: dict[str, list[float]] = {}

    arms_to_extract = [
        ("a_prime", a_prime_texts, None),  # no length normalization
        ("b1", b1_texts, a_prime_token_lengths),  # truncate to min(own_len, B1_len)
        ("b2", b2_texts, a_prime_token_lengths),  # truncate to min(own_len, B2_len)
        ("c", c_texts, None),  # Qwen text, no truncation (full span)
    ]

    for arm_name, arm_texts, a_prime_len_list in arms_to_extract:
        log_phase(f"p3_extract_{arm_name}")
        logger.info("Phase 3: Extracting arm %s (batch_size=%d)...", arm_name, batch_size)
        v_s, span_lens, arm_logps = _tf_extract_arm(
            model,
            tokenizer,
            prompts,
            arm_texts,
            layers,
            arm_name,
            a_prime_lengths=a_prime_len_list,
            batch_size=batch_size,
        )
        span_lengths_by_arm[arm_name] = span_lens
        answer_logp_by_arm[arm_name] = arm_logps

        # Save arm tensor immediately (checkpoint-per-phase)
        tensor_path = base_dir / "analysis_tensors" / f"v_{arm_name}.pt"
        tensor_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(torch.from_numpy(v_s), str(tensor_path))
        logger.info("Saved arm %s tensor to %s (shape %s)", arm_name, tensor_path, v_s.shape)

        # Free GPU memory
        del v_s
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Save span lengths
    span_path = base_dir / "analysis_tensors" / "phase3_span_lengths.json"
    span_path.write_text(json.dumps(span_lengths_by_arm, indent=2, default=_json_np))
    logger.info("Span lengths saved to %s", span_path)

    # Save per-arm mean answer-span log-P (OOD covariate diagnostic, plan §5)
    logp_path = base_dir / "analysis_tensors" / "phase3_answer_logp.json"
    logp_path.write_text(json.dumps(answer_logp_by_arm, indent=2, default=_json_np))
    logger.info("Answer-span log-P saved to %s", logp_path)

    # Upload arm tensors to HF data repo
    log_phase("p3_upload")
    logger.info("Uploading arm tensors and span lengths to HF...")
    _upload_arm_tensors(base_dir)

    write_sentinel(
        pathlib.Path("/workspace/logs/issue-823-phase3-done.json"),
        {
            "kind": "epm:progress",
            "version": 1,
            "note": "Phase 3 TF extraction complete",
            "n_contexts": n_contexts,
            "smoke": smoke,
            "ts": time.time(),
        },
    )
    log_phase("p3_done")


def _upload_arm_tensors(base_dir: pathlib.Path) -> None:
    """Upload arm tensors + span lengths to HF data repo."""
    from huggingface_hub import HfApi

    api = HfApi()
    tensors_dir = base_dir / "analysis_tensors"
    files_to_upload = list(tensors_dir.glob("*.pt")) + list(tensors_dir.glob("*.json"))
    logger.info("Uploading %d tensor files to HF...", len(files_to_upload))

    operations = []
    from huggingface_hub import CommitOperationAdd

    for f in files_to_upload:
        path_in_repo = f"{ISSUE_SLUG}/analysis_tensors/{f.name}"
        operations.append(CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(f)))

    api.create_commit(
        repo_id=HF_DATA_REPO_DATA,
        repo_type="dataset",
        commit_message=f"issue 823: arm tensors ({len(operations)} files)",
        operations=operations,
    )
    logger.info("Upload complete: %d files", len(operations))


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 4: Ridge refitting
# ═══════════════════════════════════════════════════════════════════════════════


def phase4_ridge_refit(base_dir: pathlib.Path, n_contexts: int, smoke: bool) -> None:
    """Per-arm refit (DV1) and cross-arm transfer (DV3)."""
    log_phase("p4_ridge_refit")
    logger.info("Phase 4: Ridge refitting (n_contexts=%d, smoke=%s)", n_contexts, smoke)

    from sklearn.model_selection import KFold

    from explore_persona_space.experiments.issue_779.fit_h import ridge_fit_predict

    # Load bundle
    bundle_path = pathlib.Path(
        (base_dir / "data" / "issue_823" / "bundle_path.txt").read_text().strip()
    )
    bundle = torch.load(str(bundle_path), map_location="cpu", mmap=True)
    cx_last_full = bundle["cx_last"].numpy()[:n_contexts]  # (n, 28, 3584)
    v_x_full = bundle["v_x"].numpy()[:n_contexts]  # arm A (harness gate only)
    logger.info("Loaded cx_last shape: %s", cx_last_full.shape)

    # Load arm tensors from disk
    tensors_dir = base_dir / "analysis_tensors"

    def load_arm(name: str) -> np.ndarray:
        p = tensors_dir / f"v_{name}.pt"
        assert p.exists(), f"Arm tensor not found: {p}. Run Phase 3 first."
        t = torch.load(str(p), map_location="cpu")
        return t.numpy()[:n_contexts]

    v_a_prime = load_arm("a_prime")
    v_b1 = load_arm("b1")
    v_b2 = load_arm("b2")
    v_c = load_arm("c")

    logger.info(
        "Loaded arm tensors: A' %s, B1 %s, B2 %s, C %s",
        v_a_prime.shape,
        v_b1.shape,
        v_b2.shape,
        v_c.shape,
    )

    arm_targets = {
        "A": v_x_full,
        "A_prime": v_a_prime,
        "B1": v_b1,
        "B2": v_b2,
        "C": v_c,
    }

    kf = KFold(n_splits=5, shuffle=True, random_state=0)

    # ── Computation A: per-arm refit ──────────────────────────────────────────
    log_phase("p4_refit")
    logger.info("Phase 4 Computation A: per-arm refit...")
    r2_refit: dict[str, dict[str, dict]] = {}

    for s, Y_target_s in arm_targets.items():
        r2_refit[s] = {}
        for trait in TRAITS:
            r2_folds_per_layer = []
            for layer_idx in range(EXPECTED_LAYERS):
                X = cx_last_full[:, layer_idx, :]  # (n, 3584)
                Y = Y_target_s[:, layer_idx, :]  # (n, 3584)
                r2_folds: list[float] = []
                for train_idx, val_idx in kf.split(X):
                    Y_pred = ridge_fit_predict(X[train_idx], Y[train_idx], X[val_idx])
                    ss_res = float(np.sum((Y[val_idx] - Y_pred) ** 2))
                    ss_tot = float(np.sum((Y[val_idx] - Y[val_idx].mean(0)) ** 2))
                    r2_folds.append(1.0 - ss_res / (ss_tot + 1e-12))
                r2_folds_per_layer.append(r2_folds)
            r2_refit[s][trait] = {
                "r2_by_layer": r2_folds_per_layer,  # list[28][5]
                "fit_arm": s,
                "score_arm": s,  # mechanizable: refit rows have fit_arm == score_arm
            }
        logger.info("Arm %s refit complete (layers %d, folds 5)", s, EXPECTED_LAYERS)

    # Harness reproduce-gate: arm A per-arm refit must match #779 within ±0.01.
    # Skip in smoke mode (reference file not available locally during smoke).
    if not smoke:
        _check_harness_reproduce_gate(r2_refit, base_dir)
    else:
        logger.info("Harness reproduce-gate SKIPPED (smoke mode — reference not pre-staged)")

    # A-vs-A' consistency diagnostic
    a_vs_a_prime_diag = _compute_a_vs_a_prime_diag(v_x_full, v_a_prime, r2_refit)

    # ── Computation B: cross-arm transfer ────────────────────────────────────
    log_phase("p4_transfer")
    logger.info("Phase 4 Computation B: cross-arm transfer (fit on A')...")
    r2_transfer: dict[str, dict[str, dict]] = {}

    for s_prime, Y_target_sp in arm_targets.items():
        r2_transfer[s_prime] = {}
        for trait in TRAITS:
            r2_folds_per_layer = []
            for layer_idx in range(EXPECTED_LAYERS):
                X = cx_last_full[:, layer_idx, :]
                Y_a_prime = v_a_prime[:, layer_idx, :]
                Y_sp = Y_target_sp[:, layer_idx, :]
                r2_folds = []
                for train_idx, val_idx in kf.split(X):
                    # Fit ONLY on arm A' — never on v_B1, v_B2, or v_C
                    Y_pred = ridge_fit_predict(X[train_idx], Y_a_prime[train_idx], X[val_idx])
                    ss_res = float(np.sum((Y_sp[val_idx] - Y_pred) ** 2))
                    ss_tot = float(np.sum((Y_sp[val_idx] - Y_sp[val_idx].mean(0)) ** 2))
                    r2_folds.append(1.0 - ss_res / (ss_tot + 1e-12))
                r2_folds_per_layer.append(r2_folds)
            r2_transfer[s_prime][trait] = {
                "r2_by_layer": r2_folds_per_layer,
                "fit_arm": "A_prime",  # always A' for transfer — fit_arm != score_arm → transfer
                "score_arm": s_prime,
            }
        logger.info("Transfer arm %s complete (fit=A', score=%s)", s_prime, s_prime)

    # ── Statistical tests ─────────────────────────────────────────────────────
    stats = _compute_stats(r2_refit, r2_transfer)

    # ── Persist results ───────────────────────────────────────────────────────
    result = {
        "refit": r2_refit,
        "transfer": r2_transfer,
        "stats": stats,
        "a_vs_a_prime_diagnostic": a_vs_a_prime_diag,
        "n_contexts": n_contexts,
        "smoke": smoke,
        "read_out_layers": READ_OUT_LAYERS,
        "ts": time.time(),
    }
    out_path = base_dir / "eval_results" / "issue_823" / "ridge_r2_by_arm.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(result, indent=2, default=_json_np))
    logger.info("Phase 4 complete: %s", out_path)
    log_phase("p4_done")


def _check_harness_reproduce_gate(r2_refit: dict, base_dir: pathlib.Path) -> None:
    """Check arm A per-arm refit matches #779 within ±0.01.

    Raises RuntimeError on delta > 0.01 — a silent warning would mask a ridge-recipe
    drift or bundle SHA mismatch before it propagates to DV1 and the H1/H2/H3 verdicts.
    If the #779 reference file is absent (not yet downloaded), raises RuntimeError so
    callers know to pre-stage it rather than silently skipping the gate.
    """
    ref_path = base_dir / "eval_results" / "issue_779" / "percontext_recon.json"
    if not ref_path.exists():
        raise RuntimeError(
            f"Harness reproduce-gate: #779 reference not found at {ref_path}. "
            "Pre-stage the file before running Phase 4. "
            "Use --skip-harness-gate to explicitly bypass (smoke only)."
        )
    ref = json.loads(ref_path.read_text())
    for trait in TRAITS:
        ro_layer = READ_OUT_LAYERS[trait]
        arm_a_r2 = float(np.mean(r2_refit["A"][trait]["r2_by_layer"][ro_layer]))
        if trait not in ref or "mean_r2" not in ref[trait]:
            raise RuntimeError(
                f"Harness reproduce-gate: trait '{trait}' not found in reference at {ref_path}."
            )
        ref_r2 = float(ref[trait]["mean_r2"])
        delta = abs(arm_a_r2 - ref_r2)
        if delta > 0.01:
            raise RuntimeError(
                f"Harness reproduce-gate HARD FAIL: arm A '{trait}' R²={arm_a_r2:.4f} "
                f"vs #779 reference {ref_r2:.4f} (delta={delta:.4f} > 0.01). "
                "Ridge recipe or bundle SHA drift suspected — check fit_h.py params and "
                "BUNDLE_SHA256 constant."
            )
        logger.info(
            "Harness reproduce-gate PASS: arm A %s R²=%.4f (ref=%.4f, delta=%.4f)",
            trait,
            arm_a_r2,
            ref_r2,
            delta,
        )


def _compute_a_vs_a_prime_diag(v_a: np.ndarray, v_a_prime: np.ndarray, r2_refit: dict) -> dict:
    """Compute per-context cosine similarity between arm A and arm A' at read-out layers."""
    diag: dict[str, Any] = {}
    for trait in TRAITS:
        ro = READ_OUT_LAYERS[trait]
        a_vec = v_a[:, ro, :]  # (n, hidden)
        ap_vec = v_a_prime[:, ro, :]
        # Normalized cosine per context
        a_norm = np.linalg.norm(a_vec, axis=1, keepdims=True) + 1e-9
        ap_norm = np.linalg.norm(ap_vec, axis=1, keepdims=True) + 1e-9
        cos = np.sum((a_vec / a_norm) * (ap_vec / ap_norm), axis=1)  # (n,)
        mean_cos = float(cos.mean())
        # R² delta
        r2_a = float(np.mean(r2_refit["A"][trait]["r2_by_layer"][ro]))
        r2_ap = float(np.mean(r2_refit["A_prime"][trait]["r2_by_layer"][ro]))
        diag[trait] = {
            "mean_cos": mean_cos,
            "std_cos": float(cos.std()),
            "read_out_layer": ro,
            "r2_a": r2_a,
            "r2_a_prime": r2_ap,
            "r2_delta": abs(r2_a - r2_ap),
        }
        logger.info(
            "A-vs-A' [%s, L%d]: mean_cos=%.4f, R²_A=%.4f, R²_A'=%.4f, delta=%.4f",
            trait,
            ro,
            mean_cos,
            r2_a,
            r2_ap,
            abs(r2_a - r2_ap),
        )
        if mean_cos < 0.99:
            logger.warning("A-vs-A' drift WARNING [%s]: mean cos=%.4f < 0.99", trait, mean_cos)
        if abs(r2_a - r2_ap) > 0.02:
            logger.warning(
                "A-vs-A' drift WARNING [%s]: R² delta=%.4f > 0.02", trait, abs(r2_a - r2_ap)
            )
    return diag


def _compute_stats(r2_refit: dict, r2_transfer: dict) -> dict:
    """Compute paired t-tests, bootstrap CIs, and H1/H2/H3 verdicts."""
    from scipy import stats as scipy_stats

    BONFERRONI_ALPHA = 0.05 / 3
    T_CRIT_DF4 = 3.495  # t_crit(df=4, Bonferroni alpha=0.017 two-tailed)
    N_BOOTSTRAP = 1000

    result: dict[str, Any] = {}

    for trait in TRAITS:
        ro = READ_OUT_LAYERS[trait]
        # Per-arm fold R² at read-out layer
        r2_ap = np.array(r2_refit["A_prime"][trait]["r2_by_layer"][ro])  # (5,)
        r2_b2 = np.array(r2_refit["B2"][trait]["r2_by_layer"][ro])
        r2_c = np.array(r2_refit["C"][trait]["r2_by_layer"][ro])

        def paired_t(a: np.ndarray, b: np.ndarray) -> dict:
            """Paired t-test with explicit df = n-1 guard."""
            diff = a - b
            n = len(diff)
            df = n - 1  # guard: df = len(paired_values) - 1
            assert df == n - 1, f"df guard: df={df}, n={n}"
            mean_diff = diff.mean()
            se_diff = diff.std(ddof=1) / math.sqrt(n)
            t_stat = mean_diff / (se_diff + 1e-12)
            p_val = float(2 * scipy_stats.t.sf(abs(t_stat), df=df))
            cohen_d = float(mean_diff / (diff.std(ddof=1) + 1e-12))
            return {
                "mean_diff": float(mean_diff),
                "t_stat": float(t_stat),
                "df": df,
                "p_val": p_val,
                "p_bonferroni": min(1.0, p_val * 3),
                "significant_bonferroni": abs(t_stat) > T_CRIT_DF4,
                "cohen_d": cohen_d,
            }

        ap_vs_b2 = paired_t(r2_ap, r2_b2)
        ap_vs_c = paired_t(r2_ap, r2_c)
        b2_vs_c = paired_t(r2_b2, r2_c)

        # DEFERRED CONCERN (concern-id: fold-level-vs-context-level-bootstrap-ci):
        # The plan §6 requires context-level bootstrap CI (n≈5000, 1000 iterations)
        # as the PRIMARY uncertainty estimate. This implementation uses fold-level
        # bootstrap (n=5 folds) instead, because _compute_stats only receives fold-mean
        # R² values — per-context predictions are not accumulated here.
        # To implement context-level CI, phase4_ridge_refit would need to persist
        # per-context val predictions (n_contexts x n_layers x hidden) and pass them
        # to _compute_stats. This is a significant refactor and is deferred to Phase 4
        # once the basic pipeline is validated. The fold-level bootstrap CI reported
        # here is conservative (wider than the true 5000-context CI) and is labelled
        # accordingly — the H1/H2/H3 verdicts use point estimates, not CIs, so this
        # does not affect the headline test. Raised as CONCERN (round 1).
        rng = np.random.default_rng(0)
        delta_ap_b2 = float(r2_ap.mean() - r2_b2.mean())
        delta_ap_c = float(r2_ap.mean() - r2_c.mean())
        delta_b2_c = float(r2_b2.mean() - r2_c.mean())

        # Fold-level bootstrap (5 folds, 1000 iterations) — conservative proxy only
        boot_ap_b2: list[float] = []
        for _ in range(N_BOOTSTRAP):
            idx = rng.integers(0, 5, size=5)
            boot_ap_b2.append(float(r2_ap[idx].mean() - r2_b2[idx].mean()))
        ci_ap_b2 = (float(np.percentile(boot_ap_b2, 2.5)), float(np.percentile(boot_ap_b2, 97.5)))

        # H1/H2/H3 determination
        h1 = delta_ap_b2 > 0.05 and r2_c.mean() < 0.05
        h2 = max(abs(delta_ap_b2), abs(delta_ap_c), abs(delta_b2_c)) <= 0.03
        h3 = (r2_ap.mean() >= r2_b2.mean() > r2_c.mean()) and delta_b2_c > 0.03

        verdict = "H1" if h1 else ("H2" if h2 else ("H3" if h3 else "AMBIGUOUS"))

        result[trait] = {
            "read_out_layer": ro,
            "mean_r2": {
                "A_prime": float(r2_ap.mean()),
                "B2": float(r2_b2.mean()),
                "C": float(r2_c.mean()),
            },
            "paired_t": {
                "A_prime_vs_B2": ap_vs_b2,
                "A_prime_vs_C": ap_vs_c,
                "B2_vs_C": b2_vs_c,
            },
            "bootstrap_ci_A_prime_vs_B2_fold_level": {
                "lo": ci_ap_b2[0],
                "hi": ci_ap_b2[1],
                "note": "fold-level proxy (n=5); context-level CI deferred",
            },
            "delta_r2": {
                "A_prime_vs_B2": delta_ap_b2,
                "A_prime_vs_C": delta_ap_c,
                "B2_vs_C": delta_b2_c,
            },
            "hypothesis_verdict": verdict,
            "h1_thresholds_met": h1,
            "h2_thresholds_met": h2,
            "h3_thresholds_met": h3,
            "bonferroni_alpha": BONFERRONI_ALPHA,
            "t_crit_df4": T_CRIT_DF4,
        }
        logger.info(
            "Stats [%s, L%d]: R²_A'=%.4f, R²_B2=%.4f, R²_C=%.4f → verdict=%s",
            trait,
            ro,
            r2_ap.mean(),
            r2_b2.mean(),
            r2_c.mean(),
            verdict,
        )
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# Phase 5: Validity diagnostics
# ═══════════════════════════════════════════════════════════════════════════════


def phase5_validity_diag(base_dir: pathlib.Path, n_contexts: int, smoke: bool) -> None:
    """Run validity diagnostics: text decorrelation, OOD log-P, activation cosine, lengths."""
    log_phase("p5_validity")
    logger.info("Phase 5: Validity diagnostics (n_contexts=%d, smoke=%s)", n_contexts, smoke)

    # Load arm texts
    a_prime_recs = json.loads(
        (base_dir / "raw_completions" / "phase05" / "arm_a_prime_seed42.json").read_text()
    )[:n_contexts]
    b2_recs = json.loads((base_dir / "raw_completions" / "phase1" / "b2_seed42.json").read_text())[
        :n_contexts
    ]

    a_prime_texts = [r["answer_text"] for r in a_prime_recs]
    b2_texts = [r["answer_text"] for r in b2_recs]

    # 1. Text decorrelation (TF-idf cosine)
    logger.info("Phase 5.1: TF-idf cosine B2 vs A'...")
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    valid_pairs = [(a, b) for a, b in zip(a_prime_texts, b2_texts, strict=True) if a and b]
    a_texts_valid = [p[0] for p in valid_pairs]
    b_texts_valid = [p[1] for p in valid_pairs]

    tfidf = TfidfVectorizer(max_features=10000)
    all_texts = a_texts_valid + b_texts_valid
    tfidf_mat = tfidf.fit_transform(all_texts)
    n_valid = len(a_texts_valid)
    tfidf_a = tfidf_mat[:n_valid]
    tfidf_b = tfidf_mat[n_valid:]
    cos_per_ctx = np.array(cosine_similarity(tfidf_a, tfidf_b).diagonal())
    frac_high = float((cos_per_ctx > 0.8).mean())
    logger.info(
        "TF-idf cosine: mean=%.4f, std=%.4f, frac>0.8=%.1f%%",
        cos_per_ctx.mean(),
        cos_per_ctx.std(),
        frac_high * 100,
    )
    if frac_high > 0.20:
        logger.warning(
            "Text decorrelation WARNING: %.1f%% contexts have cos(B2, A') > 0.8 — "
            "arm distinction may collapse",
            frac_high * 100,
        )

    # 2. Length distribution
    logger.info("Phase 5.2: Length distribution...")
    span_path = base_dir / "analysis_tensors" / "phase3_span_lengths.json"
    span_data = json.loads(span_path.read_text()) if span_path.exists() else {}

    def describe_lengths(lens: list[int]) -> dict:
        arr = np.array(lens)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": int(arr.min()),
            "max": int(arr.max()),
            "p25": float(np.percentile(arr, 25)),
            "p50": float(np.percentile(arr, 50)),
            "p75": float(np.percentile(arr, 75)),
            "frac_short_lt10": float((arr < 10).mean()),
        }

    length_stats: dict[str, Any] = {}
    for arm_name in ["a_prime", "b1", "b2", "c"]:
        if arm_name in span_data:
            length_stats[arm_name] = describe_lengths(span_data[arm_name])
    logger.info("Length stats: %s", {k: v["mean"] for k, v in length_stats.items()})

    # 3. Per-context activation cosine at read-out layers
    logger.info("Phase 5.3: Per-context activation cosine v_B2 vs v_A'...")
    tensors_dir = base_dir / "analysis_tensors"
    cos_by_trait: dict[str, Any] = {}
    for trait in TRAITS:
        ro = READ_OUT_LAYERS[trait]
        v_ap_path = tensors_dir / "v_a_prime.pt"
        v_b2_path = tensors_dir / "v_b2.pt"
        if v_ap_path.exists() and v_b2_path.exists():
            v_ap = torch.load(str(v_ap_path), map_location="cpu").numpy()[:n_contexts, ro, :]
            v_b2 = torch.load(str(v_b2_path), map_location="cpu").numpy()[:n_contexts, ro, :]
            nap = np.linalg.norm(v_ap, axis=1, keepdims=True) + 1e-9
            nb2 = np.linalg.norm(v_b2, axis=1, keepdims=True) + 1e-9
            cos = np.sum((v_ap / nap) * (v_b2 / nb2), axis=1)
            cos_by_trait[trait] = {
                "read_out_layer": ro,
                "mean_cos": float(cos.mean()),
                "std_cos": float(cos.std()),
                "min_cos": float(cos.min()),
                "max_cos": float(cos.max()),
            }
            logger.info(
                "Activation cosine v_B2 vs v_A' [%s, L%d]: mean=%.4f, std=%.4f",
                trait,
                ro,
                cos.mean(),
                cos.std(),
            )

    # 4. OOD log-P check (per-arm mean answer-span log P under base Qwen).
    # Computed during Phase 3 TF extraction (same forward, scalar per context)
    # and persisted to phase3_answer_logp.json; here we summarize per arm.
    logp_distributions: dict[str, Any] = {}
    logp_path = base_dir / "analysis_tensors" / "phase3_answer_logp.json"
    if logp_path.exists():
        logp_by_arm = json.loads(logp_path.read_text())
        for arm, vals in logp_by_arm.items():
            arr = np.array([v for v in vals if v is not None and np.isfinite(v)])
            logp_distributions[arm] = {
                "mean": float(arr.mean()) if arr.size else None,
                "std": float(arr.std()) if arr.size else None,
                "n_finite": int(arr.size),
                "n_total": len(vals),
            }
    else:
        logp_distributions["note"] = f"MISSING — {logp_path} not found (Phase 3 not yet run)"

    # 5. Length-R² correlation (if Phase 4 results exist)
    length_r2_correlation: dict[str, Any] = {}
    ridge_path = base_dir / "eval_results" / "issue_823" / "ridge_r2_by_arm.json"
    if ridge_path.exists() and span_path.exists():
        try:
            for trait in TRAITS:
                ro = READ_OUT_LAYERS[trait]
                if "a_prime" in span_data and "b2" in span_data:
                    ap_lens = np.array(span_data["a_prime"][:n_contexts])
                    b2_lens = np.array(span_data["b2"][:n_contexts])
                    len_delta = ap_lens.astype(float) - b2_lens.astype(float)
                    # We have fold-level R², not context-level, so correlation is approximate
                    length_r2_correlation[trait] = {
                        "note": "Context-level length-R² correlation requires per-context R²; "
                        "fold-level proxy only",
                        "mean_ap_len": float(ap_lens.mean()),
                        "mean_b2_len": float(b2_lens.mean()),
                        "mean_delta": float(len_delta.mean()),
                    }
        except Exception as e:
            logger.warning("Length-R² correlation: %s", e)

    diag = {
        "text_cosine": {
            "mean": float(cos_per_ctx.mean()),
            "std": float(cos_per_ctx.std()),
            "frac_high_gt08": frac_high,
            "n_valid_pairs": n_valid,
        },
        "logp_distributions": logp_distributions,
        "activation_cosine": cos_by_trait,
        "length_stats": length_stats,
        "length_r2_correlation": length_r2_correlation,
        "n_contexts": n_contexts,
        "smoke": smoke,
        "ts": time.time(),
    }

    out_path = base_dir / "eval_results" / "issue_823" / "validity_diagnostics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(diag, indent=2, default=_json_np))
    logger.info("Phase 5 complete: %s", out_path)
    log_phase("p5_done")


# ═══════════════════════════════════════════════════════════════════════════════
# Upload raw completions to HF
# ═══════════════════════════════════════════════════════════════════════════════


def upload_raw_completions(base_dir: pathlib.Path) -> None:
    """Upload raw completion files to HF data repo."""
    from explore_persona_space.orchestrate.hub import upload_raw_completions_to_data_repo

    upload_raw_completions_to_data_repo(
        experiment_name=ISSUE_SLUG,
        eval_results_dir=base_dir / "eval_results" / "issue_823",
    )

    # Also upload the raw_completions dir explicitly (non-canonical structure)
    from huggingface_hub import CommitOperationAdd, HfApi

    api = HfApi()
    raw_dir = base_dir / "raw_completions"
    files = list(raw_dir.rglob("*.json"))
    if files:
        ops = []
        for f in files:
            rel = f.relative_to(raw_dir)
            path_in_repo = f"{ISSUE_SLUG}/raw_completions/{rel}"
            ops.append(CommitOperationAdd(path_in_repo=path_in_repo, path_or_fileobj=str(f)))
        api.create_commit(
            repo_id=HF_DATA_REPO_DATA,
            repo_type="dataset",
            commit_message=f"issue 823: raw completions ({len(ops)} files)",
            operations=ops,
        )
        logger.info("Uploaded %d raw completion files to HF", len(ops))


# ═══════════════════════════════════════════════════════════════════════════════
# Final sentinel
# ═══════════════════════════════════════════════════════════════════════════════


def write_final_sentinel(base_dir: pathlib.Path, smoke: bool) -> None:
    """Write the epm:results sentinel for poll_pipeline.py."""
    import subprocess
    import time as time_mod

    git_sha = "unknown"
    try:
        git_sha = (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=str(base_dir), env={**os.environ}
            )
            .decode()
            .strip()
        )
    except Exception as e:
        logger.warning("git sha lookup failed (provenance recorded as 'unknown'): %s", e)

    # Build reproducibility_card — required by poll_pipeline.py for non-training experiments;
    # documents HF artifact paths so verify_uploads.py can resolve them mechanically.
    reproducibility_card = {
        "hf_data_repo": HF_DATA_REPO_DATA,
        "issue_slug": ISSUE_SLUG,
        "analysis_tensors_prefix": f"{ISSUE_SLUG}/analysis_tensors/",
        "raw_completions_prefix": f"{ISSUE_SLUG}/raw_completions/",
        "eval_results_prefix": f"{ISSUE_SLUG}/eval_results/",
        "wandb_url": "n/a (no model training in this experiment)",
    }

    sentinel_path = pathlib.Path(
        f"/workspace/logs/issue-823-epm_results-{int(time_mod.time())}.json"
    )
    write_sentinel(
        sentinel_path,
        {
            "kind": "epm:results",
            "version": 1,
            "note": json.dumps(
                {
                    "status": "complete",
                    "smoke": smoke,
                    "issue": 823,
                    "git_sha": git_sha,
                    "eval_results": str(base_dir / "eval_results" / "issue_823"),
                    "hf_upload": ISSUE_SLUG,
                    "ts": time.time(),
                    "reproducibility_card": reproducibility_card,
                },
                default=_json_np,
            ),
        },
    )
    logger.info("Final sentinel written: %s", sentinel_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Main dispatcher
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    p = argparse.ArgumentParser(
        description="Issue #823: Per-context map h — own output vs context processing."
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help="Smoke mode: run on first 10 contexts only (same dispatcher, scaled down)",
    )
    p.add_argument(
        "--phase",
        choices=["0", "0.5", "1", "2", "3", "4", "5", "all", "cpu", "gpu"],
        default="all",
        help=(
            "Which phase(s) to run. 'all'=0+0.5+1+2+3+4+5, 'cpu'=0+1+2+4+5 (VM phases), "
            "'gpu'=0.5+3 (pod phases). Default: all."
        ),
    )
    p.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory for all outputs (default: /workspace if exists, else repo root)",
    )
    p.add_argument(
        "--skip-upload",
        action="store_true",
        help="Skip HF uploads (for local testing)",
    )
    return p.parse_args()


def main() -> None:
    """Main dispatcher."""
    args = parse_args()
    base_dir = resolve_base_dir(args.base_dir)
    n_contexts = N_SMOKE if args.smoke else N_CONTEXTS_FULL

    logger.info(
        "Issue 823 dispatcher: phase=%s, smoke=%s, n_contexts=%d, base_dir=%s",
        args.phase,
        args.smoke,
        n_contexts,
        base_dir,
    )

    # Determine which phases to run
    run_phases: set[str] = set()
    if args.phase == "all":
        run_phases = {"0", "0.5", "1", "2", "3", "4", "5"}
    elif args.phase == "cpu":
        run_phases = {"0", "1", "2", "4", "5"}
    elif args.phase == "gpu":
        run_phases = {"0.5", "3"}
    else:
        run_phases = {args.phase}

    try:
        # Phase 0: substrate verify + prompt recon
        if "0" in run_phases:
            verify_record = phase0_verify(base_dir, n_contexts, args.smoke)
            logger.info("Phase 0 done: sha256_ok=%s", verify_record["sha256_ok"])

        # Phase 0.5: vLLM Qwen regeneration (GPU)
        if "0.5" in run_phases:
            phase05_vllm_regen(base_dir, n_contexts, args.smoke)

        # Phase 1: Sonnet generation
        if "1" in run_phases:
            asyncio.run(phase1_sonnet_gen(base_dir, n_contexts, args.smoke))

        # Phase 2: Derangement
        if "2" in run_phases:
            phase2_derangement(base_dir, n_contexts, args.smoke)

        # Phase 3: TF extraction (GPU)
        if "3" in run_phases:
            phase3_tf_extract(base_dir, n_contexts, args.smoke)

        # Phase 4: Ridge refitting
        if "4" in run_phases:
            phase4_ridge_refit(base_dir, n_contexts, args.smoke)

        # Phase 5: Validity diagnostics
        if "5" in run_phases:
            phase5_validity_diag(base_dir, n_contexts, args.smoke)

        # Upload raw completions and eval results to HF
        if not args.skip_upload and ("3" in run_phases or "5" in run_phases):
            log_phase("p_upload")
            upload_raw_completions(base_dir)

        # Write final sentinel (only when running full or GPU pipeline on pod)
        if not args.skip_upload and "3" in run_phases:
            write_final_sentinel(base_dir, args.smoke)

    except Exception:
        logger.exception("Dispatcher failed")
        raise

    log_phase("done")
    logger.info("Issue 823 dispatcher complete.")


if __name__ == "__main__":
    main()
