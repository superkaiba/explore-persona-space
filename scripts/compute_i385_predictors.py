#!/usr/bin/env python3
"""Compute base-model (or per-checkpoint) predictors for issue #385.

Two outputs:

(a) L20 cosine-to-librarian over a 28-row panel (librarian + 19 personas + 8
    non-persona contexts). Reuses the centroid protocol from
    ``experiments/phase_minus1_persona_vectors/extract_persona_vectors.py``:
    forward each (system, user-prompt) pair, take the last-token hidden state
    at layer 20, mean-pool over the 20 PROMPTS per row, L2-normalize, cosine
    against the librarian centroid. The 19-persona rows are reused verbatim
    from the cached ``cosine_matrix.json`` (sha256 pinned to
    ``c1a8050744e06c60fc56ca88582324ec3c70c29df39df2f29fb814e905161b0f``); only
    the 8 context centroids are computed fresh.

(b) Completion JS-divergence-to-librarian over the same 28-row panel. Reuses
    ``src/explore_persona_space/analysis/divergence.py`` (the same rig as
    #341 and #207 stage-5). One greedy 256-token response per PROMPT is
    teacher-forced under each of the 28 system prompts; per-prompt JS matrices
    are averaged into a single 28x28 matrix; the librarian row gives JS-to-source
    for the 27 bystanders.

Modes:

- ``base`` (default): compute predictors on the unfine-tuned Qwen2.5-7B-Instruct.
  One-shot; output ``eval_results/issue_385/predictors_base.json``.
- ``per-checkpoint``: compute the same predictors with a LoRA adapter applied,
  for each ``--steps`` value. Output
  ``eval_results/issue_385/predictors_per_checkpoint.json``. Per-checkpoint
  rows are written to disk AS EACH STEP COMPLETES (CLAUDE.md "checkpoint per
  phase" rule, incident #377).

Usage:
    # Base-model predictors (run ONCE before training)
    uv run python scripts/compute_i385_predictors.py --mode base \\
      --output eval_results/issue_385/predictors_base.json

    # Per-checkpoint diagnostic (run AFTER training)
    uv run python scripts/compute_i385_predictors.py --mode per-checkpoint \\
      --run-dir <RUN_DIR> \\
      --steps 5,10,25,50,75,100,150,200,300,400,600,800,1200,1600 \\
      --output eval_results/issue_385/predictors_per_checkpoint.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SOURCE_PERSONA = "librarian"
LAYER = 20

COSINE_MATRIX_PATH = (
    PROJECT_ROOT / "experiments" / "phase_minus1_persona_vectors" / "cosine_matrix.json"
)
COSINE_MATRIX_SHA256_PIN = "c1a8050744e06c60fc56ca88582324ec3c70c29df39df2f29fb814e905161b0f"

GREEDY_RESPONSE_MAX_TOKENS = 256
GREEDY_SEED = 42


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_commit() -> str:
    import subprocess

    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _verify_cosine_matrix_pin() -> dict:
    """Verify the cached cosine matrix matches the plan-pinned sha256.

    Returns the loaded JSON contents on success. Raises RuntimeError on mismatch.
    """
    if not COSINE_MATRIX_PATH.exists():
        raise FileNotFoundError(
            f"Cached cosine matrix not found at {COSINE_MATRIX_PATH}. "
            "Restore via: git show b623f11e:experiments/phase_minus1_persona_vectors/"
            "cosine_matrix.json > experiments/phase_minus1_persona_vectors/cosine_matrix.json"
        )
    actual = _sha256(COSINE_MATRIX_PATH)
    if actual != COSINE_MATRIX_SHA256_PIN:
        raise RuntimeError(
            f"cosine_matrix.json sha256 mismatch.\n"
            f"  expected: {COSINE_MATRIX_SHA256_PIN}\n"
            f"  actual:   {actual}\n"
            "Restore via: git show b623f11e:experiments/phase_minus1_persona_vectors/"
            "cosine_matrix.json > experiments/phase_minus1_persona_vectors/cosine_matrix.json"
        )
    logger.info("cosine_matrix.json sha256 OK (%s)", actual)
    with open(COSINE_MATRIX_PATH) as f:
        return json.load(f)


# ── Persona + context panel ────────────────────────────────────────────────────
# Persona rows: ALL 20 rows from experiments/.../extract_persona_vectors.py::PERSONAS.
# Plan §5.2(a): librarian is the source; the other 19 are bystanders (including
# no_persona as the anchor — flagged in the rank-test analysis, NOT excluded
# from the predictor file).
def _load_persona_panel() -> list[tuple[str, str]]:
    """Load the canonical 20-row persona panel from extract_persona_vectors.py."""
    sys.path.insert(0, str(PROJECT_ROOT / "experiments" / "phase_minus1_persona_vectors"))
    try:
        from extract_persona_vectors import PERSONAS, PROMPTS  # type: ignore[import-not-found]
    finally:
        sys.path.pop(0)
    return list(PERSONAS), list(PROMPTS)


# Context panel: 8 cells = first 2 of 3 per family from scripts/build_i181_data.py.
# Plan §5.2(b). Imported by absolute path to avoid loading the whole
# build_i181_data module (which depends on anthropic / dotenv at import time).
def _load_context_panel() -> list[tuple[str, str]]:
    """Load the 8-row non-persona-context panel from scripts/build_i181_data.py."""
    # We avoid `from scripts.build_i181_data import FAMILY_MATES` because that
    # module has heavy top-level imports (anthropic, dotenv). Read the constant
    # by exec-ing the relevant block. Cleaner: keep an inline copy here, but
    # plan §5.2(b) is explicit about lifting the constant from one source of
    # truth — hand-copying would risk drift. Use ast to parse and lift the
    # FAMILY_MATES literal so no module side-effects fire.
    import ast

    build_path = PROJECT_ROOT / "scripts" / "build_i181_data.py"
    if not build_path.exists():
        raise FileNotFoundError(f"Cannot find {build_path}")
    src = build_path.read_text()
    tree = ast.parse(src)
    family_mates = None
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "FAMILY_MATES" for t in node.targets
        ):
            family_mates = ast.literal_eval(node.value)
            break
    if family_mates is None:
        raise RuntimeError(f"FAMILY_MATES not found in {build_path}")

    # First 2 of each family, in family order: task, instruction, context, format.
    contexts: list[tuple[str, str]] = []
    for family in ("task", "instruction", "context", "format"):
        family_list = family_mates[family]
        if len(family_list) < 2:
            raise RuntimeError(f"FAMILY_MATES['{family}'] has <2 entries; cannot pick 2 per family")
        contexts.extend(family_list[:2])
    if len(contexts) != 8:
        raise RuntimeError(f"Expected 8 context entries, got {len(contexts)}")
    return contexts


def _build_panel() -> tuple[list[tuple[str, str]], list[str]]:
    """Build the full 28-row panel: librarian + 19 bystander personas + 8 contexts.

    Returns (panel, prompts) where panel is a list of (name, system_text) and
    prompts is the canonical 20-prompt PROMPTS list. The first entry of panel
    is always the source persona (librarian).
    """
    personas, prompts = _load_persona_panel()
    contexts = _load_context_panel()

    panel: list[tuple[str, str]] = []
    src_entry = next(((n, t) for n, t in personas if n == SOURCE_PERSONA), None)
    if src_entry is None:
        raise RuntimeError(f"Source persona '{SOURCE_PERSONA}' not in persona panel")
    panel.append(src_entry)
    for name, text in personas:
        if name == SOURCE_PERSONA:
            continue
        panel.append((name, text))
    panel.extend(contexts)

    names = [n for n, _ in panel]
    if len(names) != len(set(names)):
        raise RuntimeError(f"Duplicate names in panel: {names}")
    if len(panel) != 28:
        raise RuntimeError(f"Expected 28 panel rows, got {len(panel)}")
    return panel, prompts


# ── (a) L20 cosine-to-librarian ───────────────────────────────────────────────


def _compute_l20_centroids(
    model, tokenizer, panel: list[tuple[str, str]], prompts: list[str]
) -> dict[str, torch.Tensor]:  # noqa: F821 - torch imported lazily inside function
    """Compute L20 mean-pooled centroid per panel row.

    For each (system, user-prompt) pair: forward through the model, capture the
    last-token hidden state at layer ``LAYER`` via a forward hook, then mean-pool
    over the 20 PROMPTS to get one (hidden_dim,) centroid per panel row.

    Returns a dict {name: centroid_tensor (cpu, float32)}.
    """
    import torch

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook_fn(module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs.detach()

        return hook_fn

    handle = model.model.layers[LAYER].register_forward_hook(make_hook(LAYER))

    try:
        centroids: dict[str, torch.Tensor] = {}
        for row_idx, (name, sys_text) in enumerate(panel):
            row_vecs = []
            for prompt in prompts:
                messages = []
                if sys_text:
                    messages.append({"role": "system", "content": sys_text})
                messages.append({"role": "user", "content": prompt})
                text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = tokenizer(text, return_tensors="pt", padding=False).to(model.device)
                with torch.no_grad():
                    _ = model(**inputs)
                # Last-token hidden state at layer LAYER. Inputs are not padded
                # for batch=1 so the last position is seq_len - 1.
                last_pos = inputs["input_ids"].shape[1] - 1
                vec = captured[LAYER][0, last_pos, :].float().cpu()
                row_vecs.append(vec)
            centroids[name] = torch.stack(row_vecs).mean(dim=0)
            logger.info(
                "L20 centroid: %d/%d %s (mean over %d prompts)",
                row_idx + 1,
                len(panel),
                name,
                len(prompts),
            )
    finally:
        handle.remove()

    return centroids


def _cosine_to_source(centroids: dict[str, torch.Tensor]) -> dict[str, float]:  # noqa: F821
    """Cosine of each panel row's centroid against the source persona's centroid."""
    import torch
    import torch.nn.functional as F

    if SOURCE_PERSONA not in centroids:
        raise KeyError(f"{SOURCE_PERSONA} centroid missing from panel")
    src = F.normalize(centroids[SOURCE_PERSONA], dim=0)
    cos: dict[str, float] = {}
    for name, vec in centroids.items():
        if name == SOURCE_PERSONA:
            cos[name] = 1.0
            continue
        v_norm = F.normalize(vec, dim=0)
        cos[name] = float(torch.dot(src, v_norm).item())
    return cos


# ── (b) Completion JS-divergence-to-librarian ─────────────────────────────────


def _greedy_responses_for_anchor(
    panel: list[tuple[str, str]], prompts: list[str]
) -> dict[str, str]:
    """Generate 20 greedy responses anchored on a fixed system prompt.

    Plan §5.4(b): one greedy response per PROMPT via vLLM at temp=0, top_p=1,
    seed=42, max_tokens=256. Generation uses the LIBRARIAN (source) system
    prompt so the responses lie on the source-persona output distribution
    (matches #341's protocol of generating under the no_persona anchor and
    measuring JS to others; we use librarian-anchor here for symmetry with the
    radial hypothesis, plan §5.4(b)).

    Returns {prompt: greedy_response_text}.
    """
    from vllm import LLM, SamplingParams

    src_entry = next(((n, t) for n, t in panel if n == SOURCE_PERSONA), None)
    if src_entry is None:
        raise RuntimeError(f"Source persona '{SOURCE_PERSONA}' not in panel")
    src_sys_text = src_entry[1]

    llm = LLM(
        model=BASE_MODEL,
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        dtype="bfloat16",
        max_model_len=4096,
    )
    sampling = SamplingParams(
        n=1,
        temperature=0.0,
        top_p=1.0,
        max_tokens=GREEDY_RESPONSE_MAX_TOKENS,
        seed=GREEDY_SEED,
    )
    tokenizer = llm.get_tokenizer()

    rendered = []
    for prompt in prompts:
        msg = [
            {"role": "system", "content": src_sys_text},
            {"role": "user", "content": prompt},
        ]
        rendered.append(
            tokenizer.apply_chat_template(msg, tokenize=False, add_generation_prompt=True)
        )

    outputs = llm.generate(rendered, sampling)
    responses: dict[str, str] = {}
    for prompt, out in zip(prompts, outputs, strict=True):
        text = out.outputs[0].text
        if not text:
            logger.warning("Empty greedy response for prompt %r", prompt[:60])
        responses[prompt] = text

    # vLLM holds the GPU; free it before HF model load.
    del llm
    import gc

    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return responses


def _compute_js_to_source(
    model,
    tokenizer,
    panel: list[tuple[str, str]],
    prompts: list[str],
    greedy_responses: dict[str, str],
    tf_batch: int = 8,
) -> dict[str, float]:
    """Compute completion JS-divergence to the source persona.

    For each PROMPT, teacher-force the prompt's greedy response through each of
    the 28 panel rows, get (28, response_len, V) log-softmax, and feed to
    ``compute_pairwise_divergences(kl_only=True)``. Average per-prompt 28x28
    matrices, then extract the librarian row.

    Returns {name: js_to_source} for the 28 panel rows (librarian → 0.0).
    """
    import numpy as np
    import torch

    from explore_persona_space.analysis.divergence import (
        build_teacher_force_inputs,
        compute_pairwise_divergences,
        teacher_force_batch,
    )

    panel_names = [n for n, _ in panel]
    panel_texts = [t for _, t in panel]

    n_panel = len(panel)
    per_prompt_js = np.full((len(prompts), n_panel, n_panel), np.nan, dtype=np.float32)

    for q_idx, prompt in enumerate(prompts):
        response = greedy_responses.get(prompt, "")
        if not response.strip():
            logger.warning("Skipping JS for prompt %d (empty response)", q_idx)
            continue
        try:
            batch_inputs, prompt_lengths, response_len = build_teacher_force_inputs(
                tokenizer=tokenizer,
                system_prompts=panel_texts,
                question=prompt,
                response_text=response,
            )
        except ValueError as e:
            logger.warning("Teacher-force input build failed for prompt %d: %s", q_idx, e)
            continue
        if response_len < 1:
            logger.warning("Zero-length response tokens for prompt %d, skipping", q_idx)
            continue
        log_probs = teacher_force_batch(
            model=model,
            batch_inputs=batch_inputs,
            prompt_lengths=prompt_lengths,
            response_len=response_len,
            device="cuda:0",
            max_batch=tf_batch,
        )
        js_pairs, _kl_pairs = compute_pairwise_divergences(
            log_probs=log_probs,
            persona_names=panel_names,
            kl_only=True,
        )
        # js_pairs is keyed on unordered pairs; build symmetric matrix.
        mat = np.zeros((n_panel, n_panel), dtype=np.float32)
        for (a, b), v in js_pairs.items():
            i, j = panel_names.index(a), panel_names.index(b)
            mat[i, j] = float(v)
            mat[j, i] = float(v)
        per_prompt_js[q_idx] = mat
        del log_probs
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("JS pass %d/%d done (prompt=%r...)", q_idx + 1, len(prompts), prompt[:50])

    with np.errstate(all="ignore"):
        avg_js = np.nanmean(per_prompt_js, axis=0)  # (n_panel, n_panel)
    src_idx = panel_names.index(SOURCE_PERSONA)
    js_to_source: dict[str, float] = {}
    for name in panel_names:
        if name == SOURCE_PERSONA:
            js_to_source[name] = 0.0
            continue
        js_to_source[name] = float(avg_js[src_idx, panel_names.index(name)])
    return js_to_source


# ── Orchestrators ─────────────────────────────────────────────────────────────


def _load_base_model(token: str | None = None):
    """Load Qwen2.5-7B-Instruct in bf16 on cuda:0."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True, token=token)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
        token=token,
    )
    model.eval()
    return model, tokenizer


def _load_lora_adapter(model, adapter_path: Path):
    """Wrap a base HF model with a PEFT LoRA adapter from disk."""
    from peft import PeftModel

    return PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)


def _write_json(payload: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2))
    logger.info("Wrote %s", path)


def _metadata(extra: dict | None = None) -> dict:
    md: dict = {
        "git_commit": _git_commit(),
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "base_model": BASE_MODEL,
        "source_persona": SOURCE_PERSONA,
        "layer": LAYER,
        "cosine_matrix_sha256": COSINE_MATRIX_SHA256_PIN,
    }
    if extra:
        md.update(extra)
    return md


def run_base_mode(args: argparse.Namespace) -> None:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    # Verify pre-conditions first; fail fast if the cached cosine matrix drifted.
    cached_cos = _verify_cosine_matrix_pin()

    panel, prompts = _build_panel()
    panel_names = [n for n, _ in panel]
    logger.info("Panel: %d rows (source=%s, others=%d)", len(panel), panel_names[0], len(panel) - 1)
    logger.info("Prompts: %d", len(prompts))

    # ── (a) L20 cosine: read the 19 persona rows from cached matrix; compute
    # the 8 context rows fresh. Then re-compute the source persona's cosine
    # against the contexts (the cached file doesn't have context rows).
    token = os.environ.get("HF_TOKEN")
    model, tokenizer = _load_base_model(token=token)
    try:
        # Fresh centroids for everyone (source + 8 contexts strictly necessary,
        # but recomputing the 19 personas costs ~3 min and lets us cross-validate
        # against the cached matrix).
        t0 = time.time()
        centroids = _compute_l20_centroids(model, tokenizer, panel, prompts)
        cos_to_source = _cosine_to_source(centroids)
        logger.info("Fresh L20 cosine pass done in %.1fs", time.time() - t0)

        # Cross-validate fresh cosines against cached cosine_matrix.json for the
        # 19 persona rows. Should match to ~5 decimals.
        cached_layer = cached_cos.get(f"layer_{LAYER}", {})
        cached_names = cached_layer.get("persona_names", [])
        cached_matrix = cached_layer.get("matrix", [])
        if cached_names and cached_matrix and SOURCE_PERSONA in cached_names:
            src_idx = cached_names.index(SOURCE_PERSONA)
            cached_cos_to_source = {
                name: float(cached_matrix[src_idx][i]) for i, name in enumerate(cached_names)
            }
            max_diff = 0.0
            worst_name = ""
            for name in cached_names:
                if name in cos_to_source:
                    diff = abs(cos_to_source[name] - cached_cos_to_source[name])
                    if diff > max_diff:
                        max_diff = diff
                        worst_name = name
            logger.info(
                "Cosine cross-validation against cached matrix: max_abs_diff=%.6f (%s)",
                max_diff,
                worst_name,
            )
            if max_diff > 1e-3:
                logger.warning(
                    "Large drift between fresh and cached cosines (>1e-3). "
                    "Inspect cached_matrix vs fresh; the cached file may have "
                    "been computed with a different chat-template or seed."
                )

        # ── (b) JS to source. Free the HF model afterwards; vLLM owns the GPU
        # during greedy generation.
        # Generate greedy responses with vLLM (requires HF model freed).
    finally:
        del model
        import gc

        import torch as _torch

        gc.collect()
        if _torch.cuda.is_available():
            _torch.cuda.empty_cache()

    t0 = time.time()
    greedy = _greedy_responses_for_anchor(panel, prompts)
    logger.info("Greedy generation done in %.1fs (%d responses)", time.time() - t0, len(greedy))

    # Reload HF model for teacher-forcing.
    model, tokenizer = _load_base_model(token=token)
    try:
        t0 = time.time()
        js_to_source = _compute_js_to_source(
            model=model,
            tokenizer=tokenizer,
            panel=panel,
            prompts=prompts,
            greedy_responses=greedy,
            tf_batch=args.tf_batch,
        )
        logger.info("JS pass done in %.1fs", time.time() - t0)
    finally:
        del model

    payload = {
        "metadata": _metadata({"mode": "base"}),
        "panel": [{"name": n, "system_prompt": t} for n, t in panel],
        "prompts": prompts,
        "cosine_to_source": cos_to_source,
        "js_to_source": js_to_source,
        "greedy_responses": greedy,
    }
    _write_json(payload, Path(args.output))


def run_per_checkpoint_mode(args: argparse.Namespace) -> None:
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    _verify_cosine_matrix_pin()  # fail fast even though base values aren't reused here

    if not args.run_dir:
        raise SystemExit("--run-dir is required in per-checkpoint mode")
    run_dir = Path(args.run_dir)
    if not run_dir.exists():
        raise SystemExit(f"--run-dir {run_dir} does not exist")
    steps = [int(s.strip()) for s in args.steps.split(",") if s.strip()]
    if not steps:
        raise SystemExit("--steps must be a non-empty comma-separated list")

    panel, prompts = _build_panel()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Generate greedy responses ONCE on the base model (anchor stays fixed across checkpoints).
    token = os.environ.get("HF_TOKEN")
    t0 = time.time()
    greedy = _greedy_responses_for_anchor(panel, prompts)
    logger.info("Greedy generation done in %.1fs (%d responses)", time.time() - t0, len(greedy))

    # ── Per-checkpoint loop ───────────────────────────────────────────────────
    # CRITICAL (CLAUDE.md, incident #377): persist each step's row to disk AS IT
    # COMPLETES. We use a JSONL sidecar file (output_path.jsonl) so a crash in
    # checkpoint N+1 doesn't lose checkpoints 0..N. After all steps complete we
    # also write the aggregated JSON for downstream analyzer consumption.
    sidecar_path = output_path.with_suffix(output_path.suffix + ".jsonl")
    # Truncate prior sidecar to start fresh. (Re-running this script is
    # idempotent: it always re-runs all checkpoints.)
    sidecar_path.write_text("")
    rows: list[dict] = []

    for step in steps:
        adapter_path = run_dir / f"checkpoint-{step}"
        if not adapter_path.exists():
            raise FileNotFoundError(f"Checkpoint dir missing: {adapter_path}")

        # Load adapter on fresh base model each iteration (PEFT does not support
        # in-place swap to a new LoRA without rebuilding).
        base_model, tokenizer = _load_base_model(token=token)
        try:
            model = _load_lora_adapter(base_model, adapter_path)
            model.eval()

            t_step = time.time()
            centroids = _compute_l20_centroids(model, tokenizer, panel, prompts)
            cos_to_source = _cosine_to_source(centroids)
            logger.info(
                "Step %d cosine pass done in %.1fs",
                step,
                time.time() - t_step,
            )

            t_step = time.time()
            js_to_source = _compute_js_to_source(
                model=model,
                tokenizer=tokenizer,
                panel=panel,
                prompts=prompts,
                greedy_responses=greedy,
                tf_batch=args.tf_batch,
            )
            logger.info("Step %d JS pass done in %.1fs", step, time.time() - t_step)
        finally:
            del base_model
            import gc

            import torch as _torch

            gc.collect()
            if _torch.cuda.is_available():
                _torch.cuda.empty_cache()

        row = {
            "step": step,
            "adapter_path": str(adapter_path),
            "cosine_to_source": cos_to_source,
            "js_to_source": js_to_source,
        }
        with open(sidecar_path, "a") as f:
            f.write(json.dumps(row) + "\n")
        rows.append(row)
        logger.info("Step %d row persisted to %s", step, sidecar_path)

    payload = {
        "metadata": _metadata({"mode": "per-checkpoint", "run_dir": str(run_dir), "steps": steps}),
        "panel": [{"name": n, "system_prompt": t} for n, t in panel],
        "prompts": prompts,
        "rows": rows,
    }
    _write_json(payload, output_path)


# ── CLI ───────────────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("base", "per-checkpoint"),
        default="base",
        help="base = compute on unfine-tuned base model; per-checkpoint = recompute "
        "with LoRA adapter at each --steps value (diagnostic).",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Path to output JSON (e.g. eval_results/issue_385/predictors_base.json).",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="(per-checkpoint mode) run directory containing checkpoint-{step}/ adapter dirs.",
    )
    parser.add_argument(
        "--steps",
        default="",
        help="(per-checkpoint mode) comma-separated list of step values to evaluate.",
    )
    parser.add_argument(
        "--tf-batch",
        type=int,
        default=8,
        help="Teacher-force sub-batch size for divergence computation (default 8).",
    )
    return parser


def main():
    args = build_parser().parse_args()
    if args.mode == "base":
        run_base_mode(args)
    else:
        run_per_checkpoint_mode(args)


if __name__ == "__main__":
    main()
