#!/usr/bin/env python3
# Greek + special characters (×, →, —, ρ) appear in research notation.
# ruff: noqa: RUF001, RUF003
"""#518 v4 JS/KL sequence-level sweep producer over the per-arm panel.

Closes the round-7 launch-time gap (paired with
``scripts/issue518_cosine_sweep_panel.py``): the substrate builder
``scripts/issue518_build_predictor_substrate.py --arm {refusal,em}`` requires a
consolidated JS/KL sweep JSON with shape::

    {"cells": [
       {"source": "...", "bystander": "...",
        "JS_sym_nats": float, "JS_from_source_nats": float,
        "JS_from_bystander_nats": float, "M_js": float,
        "KL_src_to_bys_nats": float, "KL_bys_to_src_nats": float,
        "KL_sym_nats": float},
       ...
    ]}

For each (source, bystander) cell, this script implements the canonical
**Rao-Blackwellized sequence-level** JS + both-KL recipe per persona-distance-
metrics rule (`.claude/rules/persona-distance-metrics.md` + arXiv 2504.10637):

  1. Sample R greedy responses per (persona, probe) from the BASE model
     under each persona's system prompt.
  2. Side A -- responses sampled FROM source: teacher-force them through
     both (source, bystander)-conditioned model. Per-token full-vocab JS +
     KL averaged over response tokens, probes, responses ->
     ``JS_from_source_nats`` + ``KL_src_to_bys_nats``.
  3. Side B -- responses sampled FROM bystander: same teacher-forcing
     through both conditioned models -> ``JS_from_bystander_nats`` +
     ``KL_bys_to_src_nats``.
  4. Headlines: ``JS_sym_nats = 0.5 * (JS_from_source + JS_from_bystander)``,
     ``KL_sym_nats = 0.5 * (KL_src_to_bys + KL_bys_to_src)``,
     ``M_js = 1.0 - JS_sym_nats / ln(2)`` (polarity-aligned similarity in
     [0,1]; the loader also accepts a derived fallback).

**Unit lock (nats):** ``compute_js_divergence`` / ``compute_kl_divergence`` in
``analysis.divergence`` return divergences in nats (natural log). The
substrate builder + loader assume nats; per-cell sanity asserts
``0 <= JS_sym <= ln 2`` and ``KL >= 0``.

Loads the **BASE model `Qwen/Qwen-2.5-7B`** per plan §189 dual-base
declaration (matches the cosine producer).

Checkpoint per phase: after each cell completes, the consolidated cells list
is re-serialized to ``--out``.

Usage::

    # Smoke (stub, no GPU, validates schema):
    uv run python scripts/issue518_jskl_sweep_panel.py --arm refusal --smoke
    uv run python scripts/issue518_jskl_sweep_panel.py --arm em --smoke

    # Production:
    uv run python scripts/issue518_jskl_sweep_panel.py --arm refusal
    uv run python scripts/issue518_jskl_sweep_panel.py --arm em
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# HF cache pin before any HF import.
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv

load_dotenv()

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

logger = logging.getLogger("issue518_jskl_sweep_panel")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

DEFAULT_MODEL = "Qwen/Qwen-2.5-7B"
DEFAULT_N_PROBES_PROD = 50
DEFAULT_N_PROBES_SMOKE = 2
DEFAULT_R_PROD = 4
DEFAULT_R_SMOKE = 1
DEFAULT_MAX_NEW_TOKENS = 150
DEFAULT_TEMPERATURE = 0.0  # greedy; R>1 needs --temperature > 0
DEFAULT_TF_BATCH = 2

LN2 = math.log(2.0)
JS_MAX = LN2 + 1e-3  # fp32 round-off tolerance

SOURCES_PER_ARM: tuple[str, ...] = (
    "assistant",
    "comedian",
    "kindergarten_teacher",
    "qwen_default",
    "software_engineer",
    "villain",
)


def _git_sha() -> str:
    """Return current HEAD SHA, or 'unknown' if git is unavailable."""
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO,
            text=True,
            env={**os.environ},  # epm-lint: subprocess-env-inherit -- git probe
        ).strip()
    except (subprocess.SubprocessError, OSError):
        return "unknown"


def _reproducibility_metadata(extra: dict | None = None) -> dict:
    """Standard metadata block embedded in every output JSON."""
    meta = {
        "script": "issue518_jskl_sweep_panel",
        "git_sha": _git_sha(),
        "timestamp": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    if extra:
        meta.update(extra)
    return meta


def _get_eval_personas_24() -> dict[str, str]:
    """Load the canonical 24-persona panel."""
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    if len(EVAL_PERSONAS_24) != 24:
        raise RuntimeError(f"EVAL_PERSONAS_24 expected 24 entries, got {len(EVAL_PERSONAS_24)}")
    return dict(EVAL_PERSONAS_24)


def _stub_cell(source: str, bystander: str) -> dict:
    """Smoke stub: deterministic shape-valid cell with bounded nats."""
    h = (sum(ord(c) for c in source) + sum(ord(c) for c in bystander)) % 100
    base = 0.001 * h  # keep all JS/KL well within nats bounds
    js_from_source = 0.10 + base
    js_from_bystander = 0.14 + base
    js_sym = 0.5 * (js_from_source + js_from_bystander)
    kl_st = 0.30 + base
    kl_ts = 0.32 + base
    return {
        "source": source,
        "bystander": bystander,
        "JS_sym_nats": js_sym,
        "JS_from_source_nats": js_from_source,
        "JS_from_bystander_nats": js_from_bystander,
        "M_js": 1.0 - js_sym / LN2,
        "KL_src_to_bys_nats": kl_st,
        "KL_bys_to_src_nats": kl_ts,
        "KL_sym_nats": 0.5 * (kl_st + kl_ts),
    }


def _build_eval_probes(n_probes: int) -> list[str]:
    """Load N preregistered Betley probes (excluded from main-8 eval)."""
    sys.path.insert(0, str(REPO / "scripts"))
    from issue404_common import fetch_betley_main_8, fetch_preregistered_probes

    main8 = fetch_betley_main_8()
    probes = fetch_preregistered_probes(n=n_probes, exclude=main8)
    if not probes:
        raise RuntimeError("fetch_preregistered_probes returned 0 probes. Check Betley fixture.")
    return [p["wrong_claim"] if isinstance(p, dict) else str(p) for p in probes]


def _sample_responses(
    model,
    tokenizer,
    system_prompt: str,
    probe: str,
    r: int,
    max_new_tokens: int,
    temperature: float,
    device,
) -> list[str]:
    """Sample R responses from the base model under the given persona."""
    import torch

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": probe},
    ]
    prompt_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids.to(
        device
    )

    do_sample = temperature > 0.0
    responses: list[str] = []
    for _ in range(r):
        with torch.no_grad():
            out_ids = model.generate(
                input_ids=prompt_ids,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
                temperature=temperature if do_sample else 1.0,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id,
            )
        new_ids = out_ids[0, prompt_ids.shape[1] :]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        responses.append(text)
        if not do_sample:
            break
    return responses


def _compute_one_direction(
    *,
    model,
    tokenizer,
    src_prompt: str,
    bys_prompt: str,
    probes: list[str],
    responses_per_probe: list[list[str]],
    kl_from_first: bool,
    device,
    tf_batch: int,
) -> tuple[list[float], list[float]]:
    """Teacher-force responses through (src, bys)-conditioned model.

    Returns ``(js_list, kl_list)`` length = sum_p len(responses_per_probe[p]),
    each entry a per-(probe, response) mean-per-token divergence.

    ``kl_from_first=True``  -> KL(src || bys) (responses sampled FROM source).
    ``kl_from_first=False`` -> KL(bys || src) (responses sampled FROM bystander).
    """
    from explore_persona_space.analysis.divergence import (
        build_teacher_force_inputs,
        compute_js_divergence,
        compute_kl_divergence,
        teacher_force_batch,
    )

    sys_prompts = [src_prompt, bys_prompt]
    src_idx, bys_idx = 0, 1
    if kl_from_first:
        p_idx, q_idx = src_idx, bys_idx
    else:
        p_idx, q_idx = bys_idx, src_idx

    js_per_pair: list[float] = []
    kl_per_pair: list[float] = []

    for probe_idx, probe in enumerate(probes):
        for response in responses_per_probe[probe_idx]:
            if not response.strip():
                logger.warning(
                    "Empty response at probe_idx=%d; skipping (one sample)",
                    probe_idx,
                )
                continue
            try:
                batch_inputs, prompt_lengths, response_len = build_teacher_force_inputs(
                    tokenizer=tokenizer,
                    system_prompts=sys_prompts,
                    question=probe,
                    response_text=response,
                )
            except ValueError as e:
                logger.warning(
                    "build_teacher_force_inputs failed (probe_idx=%d): %s; skipping",
                    probe_idx,
                    e,
                )
                continue

            log_probs = teacher_force_batch(
                model=model,
                batch_inputs=batch_inputs,
                prompt_lengths=prompt_lengths,
                response_len=response_len,
                device=str(device),
                max_batch=tf_batch,
            )
            # log_probs shape: (2, response_len, vocab_size) on CPU.
            log_p = log_probs[p_idx]
            log_q = log_probs[q_idx]
            js = float(compute_js_divergence(log_p, log_q).item())
            kl = float(compute_kl_divergence(log_p, log_q).item())
            js_per_pair.append(js)
            kl_per_pair.append(kl)

    return js_per_pair, kl_per_pair


def _compute_cell(
    *,
    model,
    tokenizer,
    source: str,
    bystander: str,
    persona_prompts: dict[str, str],
    probes: list[str],
    r: int,
    max_new_tokens: int,
    temperature: float,
    device,
    tf_batch: int,
    response_cache: dict[str, list[list[str]]],
) -> dict:
    """Compute the RB sequence-level JS + both-KL cell."""
    import numpy as np

    src_prompt = persona_prompts[source]
    bys_prompt = persona_prompts[bystander]

    # Cache persona-level sampled responses (reused across pairings).
    for persona, sys_prompt in [(source, src_prompt), (bystander, bys_prompt)]:
        if persona in response_cache:
            continue
        per_probe: list[list[str]] = []
        for probe in probes:
            rs = _sample_responses(
                model,
                tokenizer,
                sys_prompt,
                probe,
                r,
                max_new_tokens,
                temperature,
                device,
            )
            per_probe.append(rs)
        response_cache[persona] = per_probe

    # Side A: responses FROM source -> JS_from_source, KL(src||bys).
    js_a, kl_st = _compute_one_direction(
        model=model,
        tokenizer=tokenizer,
        src_prompt=src_prompt,
        bys_prompt=bys_prompt,
        probes=probes,
        responses_per_probe=response_cache[source],
        kl_from_first=True,
        device=device,
        tf_batch=tf_batch,
    )
    # Side B: responses FROM bystander -> JS_from_bystander, KL(bys||src).
    js_b, kl_ts = _compute_one_direction(
        model=model,
        tokenizer=tokenizer,
        src_prompt=src_prompt,
        bys_prompt=bys_prompt,
        probes=probes,
        responses_per_probe=response_cache[bystander],
        kl_from_first=False,
        device=device,
        tf_batch=tf_batch,
    )

    if not js_a or not js_b:
        raise RuntimeError(
            f"Cell {source}__{bystander}: empty side "
            f"(|js_a|={len(js_a)}, |js_b|={len(js_b)}). All responses empty?"
        )

    js_from_source = float(np.mean(js_a))
    js_from_bystander = float(np.mean(js_b))
    js_sym = 0.5 * (js_from_source + js_from_bystander)
    kl_src_to_bys = float(np.mean(kl_st))
    kl_bys_to_src = float(np.mean(kl_ts))
    kl_sym = 0.5 * (kl_src_to_bys + kl_bys_to_src)

    # Unit-lock asserts (per .claude/rules/persona-distance-metrics.md).
    for label, v in [
        ("JS_from_source", js_from_source),
        ("JS_from_bystander", js_from_bystander),
        ("JS_sym", js_sym),
    ]:
        if not (-1e-6 <= v <= JS_MAX):
            raise RuntimeError(
                f"Cell {source}__{bystander}: {label}={v} outside [0, ln 2]; "
                f"unit-lock violation (compute_js_divergence must return nats)."
            )
    for label, v in [
        ("KL_src_to_bys", kl_src_to_bys),
        ("KL_bys_to_src", kl_bys_to_src),
    ]:
        if v < -1e-6:
            raise RuntimeError(
                f"Cell {source}__{bystander}: {label}={v} < 0; KL must be non-negative."
            )

    return {
        "source": source,
        "bystander": bystander,
        "JS_sym_nats": js_sym,
        "JS_from_source_nats": js_from_source,
        "JS_from_bystander_nats": js_from_bystander,
        "M_js": 1.0 - js_sym / LN2,
        "KL_src_to_bys_nats": kl_src_to_bys,
        "KL_bys_to_src_nats": kl_bys_to_src,
        "KL_sym_nats": kl_sym,
        "n_samples_side_a": len(js_a),
        "n_samples_side_b": len(js_b),
    }


def _write_consolidated(
    out_path: Path, cells: list[dict], arm: str, model: str, **extra: object
) -> None:
    """Write the consolidated cells JSON with reproducibility metadata."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "arm": arm,
        "model_id": model,
        "cells": cells,
        "metadata": _reproducibility_metadata(
            {
                "arm": arm,
                "model_id": model,
                "n_cells": len(cells),
                **extra,
            }
        ),
    }
    out_path.write_text(json.dumps(payload, indent=2))


def main() -> int:
    """Entrypoint. See module docstring."""
    p = argparse.ArgumentParser(
        description="#518 JS/KL sequence-level sweep producer (per-arm 6x23 cells).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--arm", choices=("refusal", "em"), required=True)
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Default: eval_results/issue_518/<arm>/predictors/jskl_sweep.json."
        ),
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--n-probes", type=int, default=None)
    p.add_argument("--r", type=int, default=None)
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument("--tf-batch", type=int, default=DEFAULT_TF_BATCH)
    p.add_argument("--sources", nargs="+", default=None)
    p.add_argument("--bystanders", nargs="+", default=None)
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke mode: emit a 1x2 stub with all required field shape + "
            "bounded nats. No model load. Validates schema + integration."
        ),
    )
    p.add_argument("--gpu-id", type=int, default=0)
    args = p.parse_args()

    if args.out is None:
        args.out = REPO / "eval_results" / "issue_518" / args.arm / "predictors" / "jskl_sweep.json"

    persona_prompts = _get_eval_personas_24()
    sources = list(args.sources) if args.sources else list(SOURCES_PER_ARM)
    unknown_s = [s for s in sources if s not in persona_prompts]
    if unknown_s:
        raise ValueError(f"Unknown sources: {unknown_s}")

    if args.smoke:
        smoke_source = sources[0]
        if args.bystanders:
            smoke_bystanders = list(args.bystanders)[:2]
        else:
            smoke_bystanders = [p for p in persona_prompts if p != smoke_source][:2]
        cells = [_stub_cell(smoke_source, b) for b in smoke_bystanders]
        _write_consolidated(
            args.out,
            cells,
            arm=args.arm,
            model=args.model,
            smoke=True,
            sources=[smoke_source],
            bystanders=smoke_bystanders,
        )
        logger.info(
            "[smoke] Wrote %d stub cells to %s (arm=%s, source=%s, bystanders=%s)",
            len(cells),
            args.out,
            args.arm,
            smoke_source,
            smoke_bystanders,
        )
        # Schema validation: re-parse and confirm required fields.
        loaded = json.loads(args.out.read_text())
        for c in loaded["cells"]:
            for k in (
                "source",
                "bystander",
                "JS_sym_nats",
                "JS_from_source_nats",
                "JS_from_bystander_nats",
                "KL_src_to_bys_nats",
                "KL_bys_to_src_nats",
                "KL_sym_nats",
                "M_js",
            ):
                if k not in c:
                    raise RuntimeError(f"Smoke output missing field {k!r}: {c}")
            # Unit-lock sanity on stub values.
            js = c["JS_sym_nats"]
            if not (0.0 <= js <= JS_MAX):
                raise RuntimeError(f"Stub JS out of nats range: {c}")
        logger.info("[smoke] schema validation PASS for %d cells", len(loaded["cells"]))
        return 0

    # ── Production path ────────────────────────────────────────────────────
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", str(args.gpu_id))

    n_probes = args.n_probes if args.n_probes is not None else DEFAULT_N_PROBES_PROD
    r = args.r if args.r is not None else DEFAULT_R_PROD

    if args.bystanders:
        unknown_b = [b for b in args.bystanders if b not in persona_prompts]
        if unknown_b:
            raise ValueError(f"Unknown bystanders: {unknown_b}")
        bystanders_per_source = {s: list(args.bystanders) for s in sources}
    else:
        bystanders_per_source = {s: [p for p in persona_prompts if p != s] for s in sources}

    probes = _build_eval_probes(n_probes)
    logger.info(
        "Production sweep: arm=%s | %d sources × ~%d bystanders | model=%s | "
        "%d probes × R=%d | tf_batch=%d",
        args.arm,
        len(sources),
        max(len(v) for v in bystanders_per_source.values()),
        args.model,
        len(probes),
        r,
        args.tf_batch,
    )

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    logger.info("Loading %s on %s", args.model, device)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16 if device.type == "cuda" else torch.float32,
        device_map={"": device} if device.type == "cuda" else None,
        trust_remote_code=True,
        token=os.environ.get("HF_TOKEN"),
    )
    if device.type == "cpu":
        model = model.to(device)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )

    cells: list[dict] = []
    response_cache: dict[str, list[list[str]]] = {}

    for src in sources:
        for bys in bystanders_per_source[src]:
            logger.info("[%s] cell source=%s bystander=%s", args.arm, src, bys)
            cell = _compute_cell(
                model=model,
                tokenizer=tokenizer,
                source=src,
                bystander=bys,
                persona_prompts=persona_prompts,
                probes=probes,
                r=r,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                device=device,
                tf_batch=args.tf_batch,
                response_cache=response_cache,
            )
            cells.append(cell)
            # Checkpoint per phase.
            _write_consolidated(
                args.out,
                cells,
                arm=args.arm,
                model=args.model,
                n_probes=len(probes),
                r=r,
            )

    logger.info(
        "[%s] JS/KL sweep complete: %d cells written to %s",
        args.arm,
        len(cells),
        args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
