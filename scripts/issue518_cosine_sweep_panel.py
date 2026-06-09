#!/usr/bin/env python3
# Greek + special characters (×, →, —, ρ) appear in research notation.
# ruff: noqa: RUF001, RUF002, RUF003
"""#518 v4 cosine-sweep producer over the per-arm (source, bystander) panel.

Closes the round-7 launch-time gap: the substrate builder
``scripts/issue518_build_predictor_substrate.py --arm {refusal,em}`` requires a
consolidated cosine sweep JSON with shape::

    {"cells": [
       {"source": "...", "bystander": "...",
        "cosine_response_l7":  float,
        "cosine_response_l14": float,
        "cosine_response_l21": float,
        "cosine_response_l27": float,
        "cosine_l20_baseline": float},
       ...
    ]}

For each (source, bystander) cell (6 sources × 23 bystanders = 138 cells per
arm), this script:

  1. Loads the **BASE model `Qwen/Qwen-2.5-7B`** (NOT Instruct -- plan §189
     dual-base declaration: predictor residual extraction runs on the BASE
     substrate even though training arms used Instruct).
  2. Samples R greedy responses per (persona, probe) from the base model under
     each persona's system prompt.
  3. Mean-pools the residual-stream activations across the response-token slice
     at layers {7, 14, 21, 27} -- per arXiv 2507.21509 Persona Vectors recipe
     (b) and the #411 / #470 canonical Qwen-7B layer set.
  4. Computes per-(source, bystander) cosine similarity at each layer.
  5. Computes a ``cosine_l20_baseline`` = cosine between source-centroid and
     base-centroid at L20 (a source-vs-no-persona distance proxy at the same
     extraction surface; the loader's legacy-alias contract accepts either
     ``cosine_l20_baseline`` or ``cosine_l20``).

The 24-persona panel is loaded from
``explore_persona_space.experiments.factor_screen_365.persona_panel.EVAL_PERSONAS_24``
(the canonical #411 / #509 panel; bystanders = panel minus source).

Checkpoint per phase: after each cell completes, the consolidated cells list
is re-serialized to ``--out`` so a mid-sweep crash never loses the cells that
already finished.

Subprocess-isolated from the rest of the pipeline (HF Transformers only -- no
vLLM, so no worker-teardown trap).

Usage::

    # Smoke (deterministic stub, no GPU, validates the cell schema):
    uv run python scripts/issue518_cosine_sweep_panel.py --arm refusal --smoke
    uv run python scripts/issue518_cosine_sweep_panel.py --arm em --smoke

    # Production:
    uv run python scripts/issue518_cosine_sweep_panel.py --arm refusal
    uv run python scripts/issue518_cosine_sweep_panel.py --arm em
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# HF cache pin must precede any HF import (per CLAUDE.md).
os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

from dotenv import load_dotenv

load_dotenv()

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "src"))

logger = logging.getLogger("issue518_cosine_sweep_panel")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Per plan §189 dual-base declaration: predictor residual extraction uses BASE.
DEFAULT_MODEL = "Qwen/Qwen-2.5-7B"
DEFAULT_LAYERS: tuple[int, ...] = (7, 14, 21, 27)
DEFAULT_BASELINE_LAYER = 20
DEFAULT_N_PROBES_PROD = 50
DEFAULT_N_PROBES_SMOKE = 2
DEFAULT_R_PROD = 4
DEFAULT_R_SMOKE = 1
DEFAULT_MAX_NEW_TOKENS = 150
DEFAULT_TEMPERATURE = 0.0  # greedy

# The 6 #411 sources held fixed across all #518 arms (matches
# scripts/run_experiment_518_refusal.py + run_experiment_518_em.py).
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
        "script": "issue518_cosine_sweep_panel",
        "git_sha": _git_sha(),
        "timestamp": datetime.now(UTC).isoformat(),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
    }
    if extra:
        meta.update(extra)
    return meta


def _get_eval_personas_24() -> dict[str, str]:
    """Load the canonical 24-persona panel (system-prompt mapping)."""
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )

    if len(EVAL_PERSONAS_24) != 24:
        raise RuntimeError(f"EVAL_PERSONAS_24 expected 24 entries, got {len(EVAL_PERSONAS_24)}")
    return dict(EVAL_PERSONAS_24)


def _stub_cell(source: str, bystander: str) -> dict:
    """Smoke stub cell: deterministic shape-valid values per (source, bystander).

    Values are non-degenerate (vary by source × bystander) so downstream
    integration tests see real predictor variance, but no GPU is required.
    """
    h = (sum(ord(c) for c in source) + sum(ord(c) for c in bystander)) % 100
    base = 0.01 * h
    return {
        "source": source,
        "bystander": bystander,
        "cosine_response_l7": 0.45 + base / 100,
        "cosine_response_l14": 0.50 + base / 100,
        "cosine_response_l21": 0.52 + base / 100,
        "cosine_response_l27": 0.55 + base / 100,
        "cosine_l20_baseline": 0.50 + base / 200,
    }


def _build_eval_probes(n_probes: int) -> list[str]:
    """Load N preregistered Betley probes (disjoint from training eval).

    Reuses ``issue404_common.fetch_preregistered_probes`` -- the same probe
    set the parent #404 / #458 / #444 predictors used. Excludes the main-8
    Betley evals so the predictor signal is on held-out probes.
    """
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
    """Sample R responses from the base model under the given persona.

    Greedy when ``temperature == 0`` (deterministic, R=1 is effectively the
    single greedy response; R>1 sampling needs a non-zero temperature).
    """
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
        # New tokens only.
        new_ids = out_ids[0, prompt_ids.shape[1] :]
        text = tokenizer.decode(new_ids, skip_special_tokens=True)
        responses.append(text)
        if not do_sample:
            # Greedy -- replicate the single response if R>1 requested
            # (caller can ignore; cosine recipe (b) only needs at least one).
            break
    return responses


def _mean_response_centroid(  # noqa: C901 - hook lifecycle + per-(probe, response) inner loop is clearer inline
    model,
    tokenizer,
    system_prompt: str | None,
    probes: list[str],
    responses_per_probe: list[list[str]],
    layers: list[int],
    device,
) -> dict[int, object]:
    """Mean-pool response-token residuals across all (probe, response) pairs.

    Mirrors #470 phase2's ``_mean_response_token_activations`` +
    ``_persona_centroid`` flow: for each (probe, response), build the full
    chat (system + user + assistant=response), forward-pass it, capture the
    residual stream at each layer, mean-pool over the response-token slice,
    then average across all (probe, response) pairs.

    ``system_prompt=None`` omits the system block entirely (used for the
    no-persona base centroid in the L20 baseline computation).
    """
    import torch

    captures: dict[int, list] = {li: [] for li in layers}

    def make_hook(li: int):
        def hook_fn(_module, _input, output):
            hs = output[0] if isinstance(output, tuple) else output
            captures[li].append(hs.detach())

        return hook_fn

    hooks = []
    for li in layers:
        h = model.model.layers[li].register_forward_hook(make_hook(li))
        hooks.append(h)

    try:
        per_layer_vecs: dict[int, list] = {li: [] for li in layers}
        for probe, responses in zip(probes, responses_per_probe, strict=True):
            if not responses:
                continue
            # Build the prompt-only token sequence once per probe.
            if system_prompt is None:
                prompt_messages = [{"role": "user", "content": probe}]
            else:
                prompt_messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": probe},
                ]
            prompt_text = tokenizer.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            prompt_len = len(prompt_ids)

            for response in responses:
                if system_prompt is None:
                    messages = [
                        {"role": "user", "content": probe},
                        {"role": "assistant", "content": response},
                    ]
                else:
                    messages = [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": probe},
                        {"role": "assistant", "content": response},
                    ]
                full_text = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=False
                )
                full_ids = tokenizer.encode(full_text, add_special_tokens=False)
                tail_drop = min(2, max(0, len(full_ids) - prompt_len - 1))
                response_end = len(full_ids) - tail_drop
                if response_end <= prompt_len:
                    continue

                input_ids = torch.tensor([full_ids], dtype=torch.long, device=device)
                for li in layers:
                    captures[li].clear()
                with torch.no_grad():
                    _ = model(input_ids=input_ids)
                for li in layers:
                    hs = captures[li][-1]
                    resp_slice = hs[0, prompt_len:response_end, :]
                    if resp_slice.shape[0] == 0:
                        continue
                    vec = resp_slice.mean(dim=0).float().cpu()
                    per_layer_vecs[li].append(vec)

        centroids: dict[int, object] = {}
        for li in layers:
            if not per_layer_vecs[li]:
                raise RuntimeError(
                    f"No activations captured at layer {li} (sys_prompt={system_prompt!r})"
                )
            stacked = torch.stack(per_layer_vecs[li])  # (n_pairs, hidden)
            centroids[li] = stacked.mean(dim=0)
        return centroids
    finally:
        for h in hooks:
            h.remove()


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
    layers_response: list[int],
    baseline_layer: int,
    device,
    response_cache: dict[str, list[list[str]]],
) -> dict:
    """Compute one (source, bystander) cell.

    Returns the dict with keys matching the loader's
    ``_REQUIRED_COSINE_FIELDS`` contract.

    Response sampling is cached per persona so the same source's responses
    are reused across all 23 bystanders (per persona-vectors recipe (b) the
    centroid is per-persona, not per-pair).
    """
    import torch.nn.functional as F

    layers_all = sorted({*layers_response, baseline_layer})

    # Sample responses per persona (cached -- centroid is persona-level).
    for persona, sys_prompt in [
        (source, persona_prompts[source]),
        (bystander, persona_prompts[bystander]),
    ]:
        if persona in response_cache:
            continue
        per_probe_responses: list[list[str]] = []
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
            per_probe_responses.append(rs)
        response_cache[persona] = per_probe_responses

    # No-persona base centroid sampling (for L20 baseline). Cache under "__base__".
    if "__base__" not in response_cache:
        per_probe_responses = []
        for probe in probes:
            rs = _sample_responses(
                model,
                tokenizer,
                # No system prompt -- bare base model behavior.
                "You are a helpful assistant.",
                probe,
                r,
                max_new_tokens,
                temperature,
                device,
            )
            per_probe_responses.append(rs)
        response_cache["__base__"] = per_probe_responses

    src_centroid = _mean_response_centroid(
        model,
        tokenizer,
        persona_prompts[source],
        probes,
        response_cache[source],
        layers_all,
        device,
    )
    bys_centroid = _mean_response_centroid(
        model,
        tokenizer,
        persona_prompts[bystander],
        probes,
        response_cache[bystander],
        layers_all,
        device,
    )
    base_centroid = _mean_response_centroid(
        model,
        tokenizer,
        "You are a helpful assistant.",
        probes,
        response_cache["__base__"],
        [baseline_layer],
        device,
    )

    cell: dict = {"source": source, "bystander": bystander}
    for li in layers_response:
        s = F.normalize(src_centroid[li].unsqueeze(0), dim=-1)
        b = F.normalize(bys_centroid[li].unsqueeze(0), dim=-1)
        cos = float((s @ b.T).item())
        cell[f"cosine_response_l{li}"] = cos

    # L20 baseline: source vs no-persona base, mean-pooled response cosine.
    s20 = F.normalize(src_centroid[baseline_layer].unsqueeze(0), dim=-1)
    base20 = F.normalize(base_centroid[baseline_layer].unsqueeze(0), dim=-1)
    cell["cosine_l20_baseline"] = float((s20 @ base20.T).item())

    return cell


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
        description="#518 cosine-sweep panel producer (per-arm 6x23 cells).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--arm",
        choices=("refusal", "em"),
        required=True,
        help="Which #518 behavior arm to compute the cosine sweep for.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output JSON path. Default: eval_results/issue_518/<arm>/predictors/cosine_sweep.json."
        ),
    )
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--layers", type=int, nargs="+", default=list(DEFAULT_LAYERS))
    p.add_argument("--baseline-layer", type=int, default=DEFAULT_BASELINE_LAYER)
    p.add_argument(
        "--n-probes",
        type=int,
        default=None,
        help="Number of probes per cell. Default: 50 (prod) or 2 (smoke).",
    )
    p.add_argument(
        "--r",
        type=int,
        default=None,
        help="Responses per (persona, probe). Default: 4 (prod) or 1 (smoke).",
    )
    p.add_argument("--max-new-tokens", type=int, default=DEFAULT_MAX_NEW_TOKENS)
    p.add_argument("--temperature", type=float, default=DEFAULT_TEMPERATURE)
    p.add_argument(
        "--sources",
        nargs="+",
        default=None,
        help=f"Subset of sources (default: {SOURCES_PER_ARM}).",
    )
    p.add_argument(
        "--bystanders",
        nargs="+",
        default=None,
        help="Subset of bystanders (default: 23 = panel minus source).",
    )
    p.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke mode: emit a 1x2 stub with the exact required field shape, "
            "no model load. Validates the cell schema + downstream integration."
        ),
    )
    p.add_argument("--gpu-id", type=int, default=0)
    args = p.parse_args()

    if args.out is None:
        args.out = (
            REPO / "eval_results" / "issue_518" / args.arm / "predictors" / "cosine_sweep.json"
        )

    persona_prompts = _get_eval_personas_24()

    sources = list(args.sources) if args.sources else list(SOURCES_PER_ARM)
    unknown_s = [s for s in sources if s not in persona_prompts]
    if unknown_s:
        raise ValueError(f"Unknown sources: {unknown_s}")

    # Smoke: 1 source x 2 bystanders, stub values (no model load).
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
            layers_response=list(args.layers),
            baseline_layer=args.baseline_layer,
        )
        logger.info(
            "[smoke] Wrote %d stub cells to %s (arm=%s, source=%s, bystanders=%s)",
            len(cells),
            args.out,
            args.arm,
            smoke_source,
            smoke_bystanders,
        )
        # Schema validation: re-parse and confirm required fields present.
        loaded = json.loads(args.out.read_text())
        for c in loaded["cells"]:
            for k in (
                "source",
                "bystander",
                "cosine_response_l7",
                "cosine_response_l14",
                "cosine_response_l21",
                "cosine_response_l27",
                "cosine_l20_baseline",
            ):
                if k not in c:
                    raise RuntimeError(f"Smoke output missing field {k!r}: {c}")
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
        "%d probes × R=%d | layers=%s | L20 baseline",
        args.arm,
        len(sources),
        max(len(v) for v in bystanders_per_source.values()),
        args.model,
        len(probes),
        r,
        args.layers,
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
    # Per-persona response cache: each persona's responses are sampled once,
    # then reused across all 23 (source, bystander) pairings.
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
                layers_response=list(args.layers),
                baseline_layer=args.baseline_layer,
                device=device,
                response_cache=response_cache,
            )
            cells.append(cell)
            # Checkpoint per phase: persist after EACH cell so a mid-sweep
            # crash never loses prior cells.
            _write_consolidated(
                args.out,
                cells,
                arm=args.arm,
                model=args.model,
                n_probes=len(probes),
                r=r,
                layers_response=list(args.layers),
                baseline_layer=args.baseline_layer,
            )

    logger.info(
        "[%s] cosine sweep complete: %d cells written to %s",
        args.arm,
        len(cells),
        args.out,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
