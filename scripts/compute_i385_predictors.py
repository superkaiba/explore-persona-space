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


def _resolve_decoder_layer(model, layer_idx: int):
    """Return the underlying decoder layer module for ``layer_idx``.

    Handles both a plain ``AutoModelForCausalLM`` (path: ``model.model.layers``)
    and a PEFT-wrapped ``PeftModel`` (PEFT wraps the base model so the path
    becomes ``model.base_model.model.model.layers``; equivalently,
    ``model.get_base_model().model.layers``). Using the wrong path on a
    PeftModel raises AttributeError on ``.layers`` — round-1 code-review
    blocker (Claude) in per-checkpoint mode.

    Args:
        model: Either an AutoModelForCausalLM or a PeftModel wrapping one.
        layer_idx: 0-indexed decoder layer to fetch.

    Returns:
        The ``nn.Module`` for the requested layer. Raises ``AttributeError`` if
        the expected attribute chain is missing (a model variant we don't
        support — fail loud rather than silently picking the wrong tensor).
    """
    # PEFT detection without importing peft eagerly (peft is heavy + only
    # needed in per-checkpoint mode). PeftModel objects expose
    # ``get_base_model``; AutoModelForCausalLM does not.
    if hasattr(model, "get_base_model") and callable(model.get_base_model):
        base = model.get_base_model()
        # base.model is the inner Qwen2Model (the decoder stack); base.lm_head
        # is the head. ``base.model.layers`` is the decoder layer list.
        if not hasattr(base, "model") or not hasattr(base.model, "layers"):
            raise AttributeError(
                f"PEFT-wrapped model: get_base_model() returned {type(base).__name__} "
                f"without a .model.layers chain. Cannot extract L{layer_idx} hidden states."
            )
        return base.model.layers[layer_idx]
    # Plain HF model path.
    if not hasattr(model, "model") or not hasattr(model.model, "layers"):
        raise AttributeError(
            f"Model of type {type(model).__name__} has no .model.layers attribute; "
            f"cannot extract L{layer_idx} hidden states. Check the model class."
        )
    return model.model.layers[layer_idx]


def _compute_l20_centroids(
    model, tokenizer, panel: list[tuple[str, str]], prompts: list[str]
) -> dict[str, torch.Tensor]:  # noqa: F821 - torch imported lazily inside function
    """Compute L20 mean-pooled centroid per panel row.

    For each (system, user-prompt) pair: forward through the model, capture the
    last-token hidden state at layer ``LAYER`` via a forward hook, then mean-pool
    over the 20 PROMPTS to get one (hidden_dim,) centroid per panel row.

    Works on both ``AutoModelForCausalLM`` (base mode) and ``PeftModel``
    (per-checkpoint mode) via ``_resolve_decoder_layer``.

    Returns a dict {name: centroid_tensor (cpu, float32)}.
    """
    import torch

    captured: dict[int, torch.Tensor] = {}

    def make_hook(layer_idx: int):
        def hook_fn(module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captured[layer_idx] = hs.detach()

        return hook_fn

    target_layer = _resolve_decoder_layer(model, LAYER)
    handle = target_layer.register_forward_hook(make_hook(LAYER))

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


ANCHOR_PERSONA = "no_persona"


def _greedy_responses_for_anchor(
    panel: list[tuple[str, str]], prompts: list[str]
) -> dict[str, str]:
    """Generate 20 greedy responses anchored on the no_persona system prompt.

    Plan §5.4(b) + §6: the JS-divergence baseline is anchored on the no_persona
    condition (empty system prompt) — i.e. the JS-to-source measures the
    distance between each bystander's completion distribution and the
    no_persona-prompt completion distribution. This matches #341's protocol
    (greedy responses generated under no_persona, teacher-forced under every
    other system prompt; row/column of the JS matrix indexed by no_persona is
    the reference baseline).

    Round-1 blocker (Codex code-review): an earlier iteration anchored on the
    librarian system prompt, which inflates JS-to-source and re-introduces
    source-leakage into the predictor itself.

    Returns {prompt: greedy_response_text}.
    """
    from vllm import LLM, SamplingParams

    anchor_entry = next(((n, t) for n, t in panel if n == ANCHOR_PERSONA), None)
    if anchor_entry is None:
        raise RuntimeError(
            f"Anchor persona '{ANCHOR_PERSONA}' not in panel; the JS-divergence "
            f"baseline must be anchored on no_persona (plan §5.4(b) / §6)."
        )
    anchor_sys_text = anchor_entry[1]
    # Sanity check: no_persona is the empty-system baseline in
    # extract_persona_vectors.py (("no_persona", "")). If this ever becomes
    # non-empty, surface it loudly because it would change the predictor's
    # semantics.
    if anchor_sys_text != "":
        raise RuntimeError(
            f"Anchor persona '{ANCHOR_PERSONA}' has non-empty system prompt "
            f"({anchor_sys_text!r}); refusing to silently drift the JS baseline."
        )

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
            {"role": "system", "content": anchor_sys_text},
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
            # An empty greedy response means the model returned no tokens for
            # this prompt under the anchor persona — that's a model/decoding
            # signal worth surfacing, not silently dropped. Fail loudly per
            # CLAUDE.md "Fail fast — never hide failures". If empty greedy
            # responses are observed at runtime, investigate the anchor +
            # sampling config rather than NaN-ing the per-prompt JS row.
            raise RuntimeError(
                f"Empty greedy response for prompt index {q_idx} (prompt={prompt!r}). "
                "The JS pipeline cannot teacher-force a zero-length response. "
                "Diagnose the anchor-persona greedy generation (check vLLM seed, "
                "anchor system prompt, sampling params) before re-running."
            )
        # build_teacher_force_inputs raises ValueError when response token IDs
        # differ across system prompts (ChatML boundary violation). DO NOT
        # silently drop the prompt — CLAUDE.md "Fail fast — never hide
        # failures" + round-1 code-review blocker (Claude + Codex, overlap).
        # If this fires, the upstream tokenizer / chat-template assumptions
        # need to be re-examined; NaN-ing the row hides the bug and shrinks
        # the effective n behind np.nanmean.
        batch_inputs, prompt_lengths, response_len = build_teacher_force_inputs(
            tokenizer=tokenizer,
            system_prompts=panel_texts,
            question=prompt,
            response_text=response,
        )
        if response_len < 1:
            raise RuntimeError(
                f"Zero-length response tokens for prompt index {q_idx} after "
                f"build_teacher_force_inputs (response={response[:60]!r}...). "
                "The teacher-force pipeline cannot operate on zero tokens; this "
                "is an upstream tokenizer / template bug, not a recoverable case."
            )
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

    # ── (a) L20 cosine: the PINNED cosine_matrix.json (sha256
    # c1a8050744e06...) is the canonical predictor for the 19 persona rows
    # (plan §3). We read it and use it directly — NOT a freshly-recomputed
    # value. The 8 context rows aren't in the cached file so they ARE
    # computed fresh.
    #
    # Round-1 blocker (Codex code-review): an earlier iteration computed
    # everything fresh and only warned on drift, defeating the sha-pin.
    # Now: the persona predictor is read from disk; the fresh recompute is
    # purely a sanity check that FAILS HARD on any drift > 1e-4. The drift
    # threshold is tight (the pinned file has 4 decimals of precision and a
    # bit-identical recomputation should match to ~1e-5 / 1e-6).
    cached_layer = cached_cos.get(f"layer_{LAYER}", {})
    cached_names = cached_layer.get("persona_names", [])
    cached_matrix = cached_layer.get("matrix", [])
    if not (cached_names and cached_matrix and SOURCE_PERSONA in cached_names):
        raise RuntimeError(
            f"cosine_matrix.json layer_{LAYER} missing persona_names / matrix / "
            f"source persona {SOURCE_PERSONA!r}; cannot use pinned predictor."
        )
    cached_src_idx = cached_names.index(SOURCE_PERSONA)
    pinned_cos_to_source: dict[str, float] = {
        name: float(cached_matrix[cached_src_idx][i]) for i, name in enumerate(cached_names)
    }
    logger.info(
        "Loaded pinned cosine_to_source from cached matrix for %d persona rows (source=%s)",
        len(pinned_cos_to_source),
        SOURCE_PERSONA,
    )

    token = os.environ.get("HF_TOKEN")
    model, tokenizer = _load_base_model(token=token)
    try:
        # Fresh centroids for everyone in the panel: this gives the 8 context
        # rows (NOT in the cached file) AND lets us cross-validate the 19
        # persona rows against the pin.
        t0 = time.time()
        centroids = _compute_l20_centroids(model, tokenizer, panel, prompts)
        fresh_cos_to_source = _cosine_to_source(centroids)
        logger.info(
            "Fresh L20 cosine pass done in %.1fs (cross-validates pin + computes contexts)",
            time.time() - t0,
        )

        # HARD-FAIL drift check on the 19 persona rows. The pinned file is the
        # source of truth; any drift > 1e-4 means the protocol / chat-template /
        # tokenizer drifted and we MUST NOT silently use either value.
        DRIFT_TOL = 1e-4
        max_diff = 0.0
        worst_name = ""
        drift_report: dict[str, float] = {}
        for name in cached_names:
            if name not in fresh_cos_to_source:
                # Source persona itself: cosine to itself is exactly 1.0 from
                # _cosine_to_source; ALSO 1.0 in the pinned matrix.
                continue
            diff = abs(fresh_cos_to_source[name] - pinned_cos_to_source[name])
            drift_report[name] = diff
            if diff > max_diff:
                max_diff = diff
                worst_name = name
        logger.info(
            "Pinned-vs-fresh cosine drift: max_abs_diff=%.6e on %s (tol=%.0e)",
            max_diff,
            worst_name or "(none)",
            DRIFT_TOL,
        )
        if max_diff > DRIFT_TOL:
            sorted_drift = sorted(drift_report.items(), key=lambda kv: -kv[1])[:5]
            raise RuntimeError(
                "Fresh L20 cosines drift from the pinned cosine_matrix.json by "
                f"max_abs_diff={max_diff:.6e} > tol={DRIFT_TOL:.0e}.\n"
                "The pinned file is the canonical predictor; a drift this large "
                "means the chat-template / tokenizer / model build differs from "
                "the one that produced the pin. Diagnose before re-running.\n"
                f"Top-5 drifters: {sorted_drift}"
            )

        # cos_to_source: USE THE PINNED VALUES for personas in the cached file;
        # use fresh values ONLY for the 8 context rows not present in the pin.
        # This is the predictor that flows into the rank test.
        cos_to_source: dict[str, float] = dict(pinned_cos_to_source)
        for name in fresh_cos_to_source:
            if name not in cos_to_source:
                cos_to_source[name] = fresh_cos_to_source[name]
        logger.info(
            "cosine_to_source: %d entries (%d from pin, %d fresh contexts)",
            len(cos_to_source),
            len(pinned_cos_to_source),
            len(cos_to_source) - len(pinned_cos_to_source),
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

    # Provenance: tag each cosine_to_source entry with its origin so downstream
    # analyzers can see at a glance which rows came from the pin vs the fresh
    # context recompute. Pin entries are the canonical rank-test predictor.
    cosine_provenance = {
        name: (
            "pinned_cosine_matrix" if name in pinned_cos_to_source else "fresh_context_recompute"
        )
        for name in cos_to_source
    }
    payload = {
        "metadata": _metadata(
            {
                "mode": "base",
                "cosine_anchor_source": "no_persona",
                "js_anchor_source": ANCHOR_PERSONA,
                "cosine_pin_drift_tol": 1e-4,
                "n_cosine_from_pin": len(pinned_cos_to_source),
                "n_cosine_from_fresh": len(cos_to_source) - len(pinned_cos_to_source),
            }
        ),
        "panel": [{"name": n, "system_prompt": t} for n, t in panel],
        "prompts": prompts,
        "cosine_to_source": cos_to_source,
        "cosine_to_source_provenance": cosine_provenance,
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
