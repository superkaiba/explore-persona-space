#!/usr/bin/env python3
"""Qwen3.5-9B replication of the issue-2254 context-only probe follow-up.

This keeps the completed Qwen2.5-7B run immutable.  It transfers the frozen
Qwen2.5 operating points to depth-matched Qwen3.5 layers, recalibrates residual
norms on Qwen3.5, and otherwise repeats the same extraction, generation, null,
judge, bootstrap, and degradation-audit design.

Run model phases under the Qwen3.5 environment overlay::

  uv run --with 'transformers==5.15.0' python \
      scripts/issue2254_probe_context_qwen35.py --phase envcheck
  CUDA_VISIBLE_DEVICES=0 uv run --with 'transformers==5.15.0' python \
      scripts/issue2254_probe_context_qwen35.py --phase smoke
  CUDA_VISIBLE_DEVICES=0 uv run --with 'transformers==5.15.0' python \
      scripts/issue2254_probe_context_qwen35.py --phase capture
  CUDA_VISIBLE_DEVICES=0 uv run --with 'transformers==5.15.0' python \
      scripts/issue2254_probe_context_qwen35.py --phase generate \
      --shard-id 0 --num-shards 4

Judge and reduce can run in the repository environment.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import scripts.issue2254_probe_context_followup as base  # noqa: E402

MODEL_ID = "Qwen/Qwen3.5-9B"
MODEL_REVISION = "c202236235762e1c871ad0ccb60c8ee5ba337b9a"
TRANSFORMERS_PIN = "5.15.0"
N_LAYERS = 32
HIDDEN_DIM = 4096
ALL_LAYERS = tuple(range(N_LAYERS))
LAYER_CONFIGS = {
    "L16": (16,),
    "L20": (20,),
    "L23": (23,),
    "mid": (16, 20, 23),
    "all": ALL_LAYERS,
}

# Qwen2.5 layer indices 14/17/20 map by relative decoder depth to 16/20/23
# in the 32-layer Qwen3.5 stack.  c is transferred unchanged because alpha is
# re-expressed through Qwen3.5's own pooled median residual norm rho_L.
TRANSFER_OPERATING_POINTS: dict[str, dict[str, dict[str, Any]]] = {
    "evil": {
        "single": {"layer_config": "L16", "c": 4.0, "source": "q25:L14/c4"},
        "mid": {"layer_config": "mid", "c": 4.0, "source": "q25:mid/c4"},
        "all": {"layer_config": "all", "c": 0.5, "source": "q25:all/c0.5"},
    },
    "sycophancy": {
        "single": {"layer_config": "L16", "c": 2.0, "source": "q25:L14/c2"},
        "mid": {"layer_config": "mid", "c": 4.0, "source": "q25:mid/c4"},
        "all": {"layer_config": "all", "c": 4.0, "source": "q25:all/c4"},
    },
    "hallucination": {
        "single": {"layer_config": "L20", "c": 2.0, "source": "q25:L17/c2"},
        "mid": {"layer_config": "mid", "c": 2.0, "source": "q25:mid/c2"},
        "all": {"layer_config": "all", "c": 4.0, "source": "q25:all/c4"},
    },
}

OUT_ROOT = REPO_ROOT / "eval_results/issue_2254/context_probe_followup_qwen35_9b"
FIG_PATH = REPO_ROOT / "artifacts/issue2254/context_probe_followup_qwen35_9b_vs_chance.png"
MAX_NEW_TOKENS = base.MAX_NEW_TOKENS
CAP_REGEN_THRESHOLD = base.CAP_REGEN_THRESHOLD

_MODEL = None
_TOKENIZER = None
_CJK_RE = base._CJK_RE


def _write_json(path: Path, payload: Any) -> None:
    base._write_json(path, payload)


def _require_cuda(phase: str) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"--phase {phase} requires CUDA")


def render_qwen35(tokenizer, context: dict) -> str:
    """Render the single-turn context under Qwen3.5 thinking-OFF."""
    from explore_persona_space.experiments.issue1415.steering import context_messages

    rendered = tokenizer.apply_chat_template(
        context_messages(context),
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    open_i = rendered.rfind("<think>")
    close_i = rendered.rfind("</think>")
    if open_i >= 0:
        if close_i <= open_i or rendered[open_i + len("<think>") : close_i].strip():
            raise RuntimeError("Qwen3.5 render did not contain a closed empty thinking block")
    return rendered


def ids_qwen35(tokenizer, context: dict) -> list[int]:
    ids = tokenizer(render_qwen35(tokenizer, context), add_special_tokens=False)["input_ids"]
    if len(ids) < 4:
        raise RuntimeError(f"unexpectedly short Qwen3.5 context: {len(ids)} tokens")
    return ids


def _load_model_and_tokenizer():
    global _MODEL, _TOKENIZER
    if _MODEL is not None:
        return _MODEL, _TOKENIZER
    _require_cuda("model-load")
    import torch
    import transformers
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

    if transformers.__version__ != TRANSFORMERS_PIN:
        raise RuntimeError(
            f"Qwen3.5 model phases require transformers=={TRANSFORMERS_PIN}; "
            f"got {transformers.__version__}"
        )
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    text_cfg = getattr(cfg, "text_config", cfg)
    if (int(text_cfg.num_hidden_layers), int(text_cfg.hidden_size)) != (
        N_LAYERS,
        HIDDEN_DIM,
    ):
        raise RuntimeError(
            f"Qwen3.5 config drift: layers={text_cfg.num_hidden_layers}, "
            f"hidden={text_cfg.hidden_size}"
        )
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    kwargs = {
        "revision": MODEL_REVISION,
        "dtype": torch.bfloat16,
        "device_map": {"": 0},
    }
    try:
        model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs)
    except (ValueError, KeyError):
        from transformers import AutoModelForImageTextToText

        model = AutoModelForImageTextToText.from_pretrained(MODEL_ID, **kwargs)
    model.eval()
    from explore_persona_space.analysis.extraction import _resolve_decoder_blocks

    blocks, _embed, _depth = _resolve_decoder_blocks(model)
    if blocks is None or len(blocks) != N_LAYERS:
        raise RuntimeError(f"resolved {None if blocks is None else len(blocks)} decoder blocks")
    _MODEL, _TOKENIZER = model, tok
    return model, tok


def phase_envcheck(args: argparse.Namespace) -> None:
    import transformers
    from transformers import AutoConfig, AutoTokenizer

    if transformers.__version__ != TRANSFORMERS_PIN:
        raise RuntimeError(
            f"envcheck requires transformers=={TRANSFORMERS_PIN}; got {transformers.__version__}"
        )
    cfg = AutoConfig.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    text_cfg = getattr(cfg, "text_config", cfg)
    tok = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
    rendered = render_qwen35(tok, {"system": None, "user": "What is 2+2?"})
    report = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "transformers": transformers.__version__,
        "n_layers": int(text_cfg.num_hidden_layers),
        "hidden_dim": int(text_cfg.hidden_size),
        "thinking_off_empty_block": "<think>\n\n</think>" in rendered,
        "render_token_count": len(ids_qwen35(tok, {"system": None, "user": "What is 2+2?"})),
    }
    if (report["n_layers"], report["hidden_dim"]) != (N_LAYERS, HIDDEN_DIM):
        raise RuntimeError(f"environment shape mismatch: {report}")
    _write_json(args.out_root / "envcheck_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def _right_pad(rows: list[list[int]], pad_id: int, device):
    import torch

    width = max(map(len, rows))
    input_ids = torch.full((len(rows), width), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(rows), width), dtype=torch.long, device=device)
    for row_index, row in enumerate(rows):
        input_ids[row_index, : len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        mask[row_index, : len(row)] = 1
    return input_ids, mask


def capture_last_context(
    model, tokenizer, contexts: list[dict], layers: list[int], *, batch_size: int
) -> np.ndarray:
    """Capture decoder-block outputs at each row's final rendered context token."""
    from explore_persona_space.analysis.extraction import extract_layer_activations

    device = next(model.parameters()).device
    ids_rows = [ids_qwen35(tokenizer, context) for context in contexts]
    captured_rows = []
    for start in range(0, len(ids_rows), batch_size):
        chunk = ids_rows[start : start + batch_size]
        input_ids, mask = _right_pad(chunk, tokenizer.pad_token_id, device)
        acts = extract_layer_activations(model, input_ids, layers, attention_mask=mask)
        for row_index, ids in enumerate(chunk):
            captured_rows.append(
                np.stack(
                    [acts[layer][row_index, len(ids) - 1].float().cpu().numpy() for layer in layers]
                )
            )
        del acts
    out = np.stack(captured_rows)
    if out.shape != (len(contexts), len(layers), HIDDEN_DIM):
        raise RuntimeError(f"bad capture shape {out.shape}")
    return out


def _compute_rho(model, tokenizer) -> tuple[dict[str, dict[str, float]], dict[str, float]]:
    layers = list(ALL_LAYERS)
    per_behavior: dict[str, dict[str, float]] = {}
    pooled: dict[int, list[float]] = {layer: [] for layer in layers}
    for behavior in base.BEHAVIORS:
        questions = base.parent._eval_questions(behavior)
        contexts = base.parent._contexts_for_questions(questions)
        acts = capture_last_context(model, tokenizer, contexts, layers, batch_size=1)
        norms = np.linalg.norm(acts.astype(np.float64), axis=2)
        per_behavior[behavior] = {
            f"L{layer}": float(np.median(norms[:, layer])) for layer in layers
        }
        for layer in layers:
            pooled[layer].extend(float(v) for v in norms[:, layer])
    return per_behavior, {f"L{layer}": float(np.median(values)) for layer, values in pooled.items()}


def phase_smoke(args: argparse.Namespace) -> None:
    import torch

    from explore_persona_space.analysis.extraction import _resolve_decoder_blocks
    from explore_persona_space.experiments.issue1415 import steering

    model, tokenizer = _load_model_and_tokenizer()
    context = {"system": None, "user": "State one fact about the Moon."}
    acts = capture_last_context(model, tokenizer, [context], [0, 16, 31], batch_size=1)
    delta = torch.tensor(base._unit(np.arange(HIDDEN_DIM) - HIDDEN_DIM / 2), dtype=torch.bfloat16)
    delta = delta.to(model.device)
    hook = steering.DeltaHook(model, 16, delta, 0.001, all_positions=False)
    with hook:
        texts = steering.generate_batch(
            model,
            tokenizer,
            [context],
            n=1,
            hook=hook,
            max_new_tokens=8,
            temperature=0.0,
            render_fn=render_qwen35,
            ids_fn=ids_qwen35,
        )
    blocks, _embed, depth = _resolve_decoder_blocks(model)
    report = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "capture_shape": list(acts.shape),
        "resolved_blocks": len(blocks) if blocks is not None else None,
        "resolver_depth": depth,
        "hook_edits": hook.n_edits,
        "generated_nonempty": bool(texts[0][0].strip()),
    }
    if report["capture_shape"] != [1, 3, HIDDEN_DIM] or hook.n_edits != 1:
        raise RuntimeError(f"smoke gate failed: {report}")
    _write_json(args.out_root / "smoke_report.json", report)
    print(json.dumps(report, indent=2), flush=True)


def phase_capture(args: argparse.Namespace) -> None:
    _require_cuda("capture")
    import torch

    model, tokenizer = _load_model_and_tokenizer()
    layers = list(ALL_LAYERS)
    direction_dir = args.out_root / "directions"
    direction_dir.mkdir(parents=True, exist_ok=True)
    rho_per_behavior, rho_pooled = _compute_rho(model, tokenizer)
    report: dict[str, Any] = {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "transformers": TRANSFORMERS_PIN,
        "thinking": "disabled",
        "n_layers": N_LAYERS,
        "hidden_dim": HIDDEN_DIM,
        "rho_median_last_context_token": rho_per_behavior,
        "rho_pooled_median": rho_pooled,
        "transfer_operating_points": TRANSFER_OPERATING_POINTS,
        "behaviors": {},
    }
    for behavior in args.behaviors:
        positive, negative = base.parent._extraction_contexts(behavior)
        acts = capture_last_context(
            model,
            tokenizer,
            positive + negative,
            layers,
            batch_size=args.capture_batch,
        )
        n_positive = len(positive)
        diffmean = base.parent.diff_of_means_direction(acts[:n_positive], acts[n_positive:])
        probe, probe_report = base.fit_probe_directions(acts)
        cosines = np.sum(diffmean * probe, axis=1)
        for method, matrix in (("diffmean", diffmean), ("probe", probe)):
            for layer, vector in enumerate(matrix):
                torch.save(
                    {
                        "direction": torch.tensor(vector, dtype=torch.float32),
                        "behavior": behavior,
                        "method": method,
                        "layer": layer,
                        "model": MODEL_ID,
                        "revision": MODEL_REVISION,
                    },
                    direction_dir / f"{behavior}_{method}_L{layer}.pt",
                )
        report["behaviors"][behavior] = {
            "probe": probe_report,
            "cosine_probe_vs_diffmean_per_layer": [float(v) for v in cosines],
            "cosine_summary": {
                "min": float(cosines.min()),
                "median": float(np.median(cosines)),
                "max": float(cosines.max()),
            },
        }
        print(
            f"[capture] {behavior}: probe AUC median="
            f"{np.median([row['heldout_auc_mean'] for row in probe_report['layers']]):.3f}; "
            f"probe/diffmean cosine median={np.median(cosines):.3f}",
            flush=True,
        )
    _write_json(direction_dir / "fit_report.json", report)


def build_cells(
    behaviors: tuple[str, ...] | list[str] = base.BEHAVIORS,
    *,
    n_random: int = base.N_RANDOM_DIRECTIONS,
) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for behavior in behaviors:
        cells.append({"behavior": behavior, "kind": "alpha0"})
        for method in base.METHODS:
            for breadth in base.BREADTHS:
                op = TRANSFER_OPERATING_POINTS[behavior][breadth]
                cells.append(
                    {
                        "behavior": behavior,
                        "kind": "signal",
                        "method": method,
                        "breadth": breadth,
                        "layer_config": op["layer_config"],
                        "c": float(op["c"]),
                        "source_cell_id": op["source"],
                    }
                )
        for random_seed in range(n_random):
            for breadth in base.BREADTHS:
                op = TRANSFER_OPERATING_POINTS[behavior][breadth]
                cells.append(
                    {
                        "behavior": behavior,
                        "kind": "random",
                        "method": "random",
                        "random_seed": random_seed,
                        "breadth": breadth,
                        "layer_config": op["layer_config"],
                        "c": float(op["c"]),
                        "source_cell_id": op["source"],
                    }
                )
    return cells


def _load_rho(args: argparse.Namespace) -> dict[str, float]:
    report = json.loads((args.out_root / "directions" / "fit_report.json").read_text())
    rho = {key: float(value) for key, value in report["rho_pooled_median"].items()}
    if set(rho) != {f"L{layer}" for layer in ALL_LAYERS}:
        raise RuntimeError("Qwen3.5 rho bank is incomplete")
    return rho


def _load_direction(args: argparse.Namespace, cell: dict[str, Any], layer: int):
    import torch

    if cell["kind"] == "random":
        digest = hashlib.sha256(
            (
                f"2254-qwen35-context-probe-null:{cell['behavior']}:{cell['random_seed']}:{layer}"
            ).encode()
        ).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], "little"))
        return torch.tensor(base._unit(rng.standard_normal(HIDDEN_DIM)), dtype=torch.float32)
    path = args.out_root / "directions" / f"{cell['behavior']}_{cell['method']}_L{layer}.pt"
    payload = torch.load(path, map_location="cpu", weights_only=True)
    vector = payload["direction"].float()
    if tuple(vector.shape) != (HIDDEN_DIM,):
        raise RuntimeError(f"bad direction shape in {path}: {tuple(vector.shape)}")
    return vector / vector.norm()


def _hook_factory(model, args: argparse.Namespace, cell: dict[str, Any], rho: dict[str, float]):
    import torch

    from explore_persona_space.experiments.issue1415.steering import DeltaHook
    from explore_persona_space.experiments.issue2254.hooks import multi_layer_delta_hooks

    if cell["kind"] == "alpha0":
        layer = int(
            LAYER_CONFIGS[TRANSFER_OPERATING_POINTS[cell["behavior"]]["single"]["layer_config"]][0]
        )
        zero = torch.zeros(HIDDEN_DIM, dtype=torch.bfloat16, device=model.device)

        def make_zero():
            return DeltaHook(model, layer, zero, 0.0, all_positions=False)

        return make_zero, {f"L{layer}": 0.0}
    layers = list(LAYER_CONFIGS[cell["layer_config"]])
    directions = [
        _load_direction(args, cell, layer).to(device=model.device, dtype=torch.bfloat16)
        for layer in layers
    ]
    alphas = [(float(cell["c"]) / len(layers)) * rho[f"L{layer}"] for layer in layers]
    if len(layers) == 1:

        def make():
            return DeltaHook(model, layers[0], directions[0], alphas[0], all_positions=False)

    else:

        def make():
            return multi_layer_delta_hooks(model, layers, directions, alphas, all_positions=False)

    return make, {f"L{layer}": float(alpha) for layer, alpha in zip(layers, alphas, strict=True)}


def _generate_cell(
    model,
    tokenizer,
    cell: dict[str, Any],
    contexts: list[dict],
    hook_make,
    *,
    max_new_tokens: int,
    alphas: dict[str, float],
) -> dict[str, Any]:
    from explore_persona_space.experiments.issue1415 import steering

    seeds_out: dict[str, Any] = {}
    cap_fractions = []
    for seed in base.GEN_SEEDS:
        with hook_make() as hook:
            completions = steering.generate_batch(
                model,
                tokenizer,
                contexts,
                n=1,
                hook=hook,
                max_new_tokens=max_new_tokens,
                temperature=1.0,
                seed_base=seed,
                render_fn=render_qwen35,
                ids_fn=ids_qwen35,
            )
        coherent = [steering.coherence_check(per_context) for per_context in completions]
        seeds_out[str(seed)] = {
            "completions": completions,
            "coherent_flags": coherent,
            "condition_passes": [steering.condition_passes(flags) for flags in coherent],
        }
        cap_fractions.append(base.parent._cap_hit_fraction(completions, tokenizer, max_new_tokens))
    cid = base.cell_id(cell)
    return {
        "cell_id": cid,
        "cell": {**cell, "position": "context"},
        "alphas": alphas,
        "q_of_context": list(range(len(contexts))),
        "seeds": seeds_out,
        "max_new_tokens": max_new_tokens,
        "cap_hit_fraction": float(np.mean(cap_fractions)),
        "design": {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "thinking": "disabled",
            "position": "last_context_token_only",
            "generation_seeds": list(base.GEN_SEEDS),
            "draws_per_seed": 1,
            "operating_point_transfer": "Qwen2.5 frozen point, depth-matched layer, Qwen3.5 rho",
        },
    }


def phase_generate(args: argparse.Namespace) -> None:
    _require_cuda("generate")
    cells = build_cells(args.behaviors, n_random=args.n_random)
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError("shard-id must be in [0, num-shards)")
    shard = cells[args.shard_id :: args.num_shards]
    raw_dir = args.out_root / "raw_completions"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rho = _load_rho(args)
    model, tokenizer = _load_model_and_tokenizer()
    questions = {behavior: base.parent._eval_questions(behavior) for behavior in args.behaviors}
    started = time.time()
    for index, cell in enumerate(shard, 1):
        cid = base.cell_id(cell)
        path = raw_dir / f"{cid}.json"
        if path.exists() and not args.force:
            print(f"[generate] {index}/{len(shard)} cached {cid}", flush=True)
            continue
        contexts = base.parent._contexts_for_questions(questions[cell["behavior"]])
        make, alphas = _hook_factory(model, args, cell, rho)
        record = _generate_cell(
            model,
            tokenizer,
            cell,
            contexts,
            make,
            max_new_tokens=MAX_NEW_TOKENS,
            alphas=alphas,
        )
        if record["cap_hit_fraction"] > CAP_REGEN_THRESHOLD:
            record = _generate_cell(
                model,
                tokenizer,
                cell,
                contexts,
                make,
                max_new_tokens=2 * MAX_NEW_TOKENS,
                alphas=alphas,
            )
            record["regenerated_for_cap_hits"] = True
        _write_json(path, record)
        print(
            f"[generate] {index}/{len(shard)} {cid} elapsed={time.time() - started:.1f}s",
            flush=True,
        )
    _write_json(
        args.out_root / f"generate_shard_{args.shard_id}_done.json",
        {"shard_id": args.shard_id, "num_shards": args.num_shards, "n_cells": len(shard)},
    )


def _non_cjk_question_scores(args: argparse.Namespace, cell_id_value: str) -> dict[str, Any]:
    raw = json.loads((args.out_root / "raw_completions" / f"{cell_id_value}.json").read_text())
    judged = json.loads((args.out_root / "judge" / "judged" / f"{cell_id_value}.json").read_text())
    per_question: list[list[float]] = [[] for _ in raw["q_of_context"]]
    kept = 0
    total = 0
    for seed, question_index, draw_index, completion in base._iter_generated(raw):
        total += 1
        if _CJK_RE.search(completion):
            continue
        item_id = base._judge_item_id(cell_id_value, seed, question_index, draw_index)
        value = judged["completion_mean_scores"][item_id]
        if value is not None:
            per_question[question_index].append(float(value))
            kept += 1
    question_means = [float(np.mean(values)) if values else None for values in per_question]
    return {
        "question_means": question_means,
        "mean_score": float(
            np.nanmean([np.nan if value is None else value for value in question_means])
        ),
        "kept_scored_completions": kept,
        "total_completions": total,
    }


def _build_audit_report(args: argparse.Namespace, summary: dict[str, Any]) -> dict[str, Any]:
    judged_paths = sorted((args.out_root / "judge" / "judged").glob("*.json"))
    judged_rows = [json.loads(path.read_text()) for path in judged_paths]
    accounting_keys = (
        "n_total_draws",
        "n_refusal_draws",
        "n_api_refusal_draws",
        "n_content_dropped_draws",
        "n_transport_lost_draws",
        "n_truncation_dropped_draws",
        "n_items_zero_valid",
    )
    accounting = {
        key: int(sum(int(row["accounting"][key]) for row in judged_rows)) for key in accounting_keys
    }
    accounting.update(
        {
            "n_cells": len(judged_rows),
            "n_items": int(sum(int(row["accounting"]["n_items"]) for row in judged_rows)),
            "expected_draws": int(
                sum(
                    int(row["accounting"]["n_items"]) * int(row["judge"]["draws"])
                    for row in judged_rows
                )
            ),
            "n_valid_draws": int(
                sum(
                    sum(int(value) for value in row["accounting"]["per_item_draw_counts"].values())
                    for row in judged_rows
                )
            ),
            "n_fully_complete_cells": int(
                sum(float(row["accounting"]["frac_items_complete"]) == 1.0 for row in judged_rows)
            ),
            "min_item_completeness": float(
                min(float(row["accounting"]["frac_items_complete"]) for row in judged_rows)
            ),
        }
    )
    fit_report = json.loads((args.out_root / "directions" / "fit_report.json").read_text())
    behavior_audits: dict[str, Any] = {}
    for behavior in base.BEHAVIORS:
        baseline = _non_cjk_question_scores(args, f"{behavior}__a0")
        baseline_q = np.asarray(
            [np.nan if value is None else value for value in baseline["question_means"]],
            dtype=np.float64,
        )
        method_audits: dict[str, Any] = {}
        for method in base.METHODS:
            method_summary = summary["behaviors"][behavior]["methods"][method]
            cell_id_value = method_summary["selected_cell_id"]
            selected = _non_cjk_question_scores(args, cell_id_value)
            selected_q = np.asarray(
                [np.nan if value is None else value for value in selected["question_means"]],
                dtype=np.float64,
            )
            method_audits[method] = {
                "selected_cell_id": cell_id_value,
                "non_cjk_mean_score": selected["mean_score"],
                "non_cjk_delta_score_vs_baseline": float(np.nanmean(selected_q - baseline_q)),
                "kept_scored_completions": selected["kept_scored_completions"],
                "total_completions": selected["total_completions"],
            }
        layer_cosines = fit_report["behaviors"][behavior]["cosine_probe_vs_diffmean_per_layer"]
        geometry: dict[str, Any] = {}
        for breadth in base.BREADTHS:
            op = TRANSFER_OPERATING_POINTS[behavior][breadth]
            layers = list(LAYER_CONFIGS[op["layer_config"]])
            values = np.asarray([layer_cosines[layer] for layer in layers], dtype=np.float64)
            geometry[breadth] = {
                "layers": layers,
                "min_cosine": float(values.min()),
                "median_cosine": float(np.median(values)),
                "max_cosine": float(values.max()),
            }
        behavior_audits[behavior] = {
            "baseline_non_cjk_mean_score": baseline["mean_score"],
            "baseline_kept_scored_completions": baseline["kept_scored_completions"],
            "baseline_total_completions": baseline["total_completions"],
            "selected_methods": method_audits,
            "probe_vs_diffmean_geometry": geometry,
        }
    return {
        "model": MODEL_ID,
        "revision": MODEL_REVISION,
        "judge_accounting": accounting,
        "behaviors": behavior_audits,
    }


def phase_reduce(args: argparse.Namespace) -> None:
    judged_dir = args.out_root / "judge" / "judged"
    summary: dict[str, Any] = {
        "design": {
            "model": MODEL_ID,
            "revision": MODEL_REVISION,
            "transformers": TRANSFORMERS_PIN,
            "thinking": "disabled",
            "methods": list(base.METHODS),
            "behaviors": list(base.BEHAVIORS),
            "position": "last_context_token_only",
            "operating_points": "Qwen2.5 frozen points transferred by relative depth; Qwen3.5 rho",
            "transfer_operating_points": TRANSFER_OPERATING_POINTS,
            "generation_seeds": list(base.GEN_SEEDS),
            "random_direction_seeds": args.n_random,
            "bootstrap_draws": base.N_BOOT,
            "bootstrap_unit": "paired eval question",
        },
        "behaviors": {},
    }
    for behavior in base.BEHAVIORS:
        paths = sorted(judged_dir.glob(f"{behavior}__*.json"))
        expected = 1 + 3 * len(base.METHODS) + 3 * args.n_random
        if len(paths) != expected:
            raise RuntimeError(f"{behavior}: expected {expected} judged cells, got {len(paths)}")
        rows = [json.loads(path.read_text()) for path in paths]
        summary["behaviors"][behavior] = base.reduce_behavior(rows, behavior)
    _write_json(args.out_root / "summary.json", summary)
    _write_json(args.out_root / "audit_report.json", _build_audit_report(args, summary))
    base._plot(
        summary,
        args.fig_path,
        title="Qwen3.5-9B last-context-token steering: DiffMean vs probe",
    )
    print(f"wrote {args.out_root / 'summary.json'}", flush=True)
    print(f"wrote {args.out_root / 'audit_report.json'}", flush=True)
    print(f"wrote {args.fig_path}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        required=True,
        choices=("envcheck", "smoke", "capture", "generate", "judge", "reduce"),
    )
    parser.add_argument(
        "--behaviors", nargs="+", choices=base.BEHAVIORS, default=list(base.BEHAVIORS)
    )
    parser.add_argument("--judge-behavior", choices=base.BEHAVIORS)
    parser.add_argument("--judge-draws", type=int, default=5)
    parser.add_argument("--n-random", type=int, default=base.N_RANDOM_DIRECTIONS)
    parser.add_argument("--shard-id", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--capture-batch", type=int, default=8)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--fig-path", type=Path, default=FIG_PATH)
    parser.add_argument("--force", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.out_root = args.out_root.resolve()
    args.fig_path = args.fig_path.resolve()
    if args.n_random < 2:
        raise ValueError("--n-random must be at least 2")
    dispatch = {
        "envcheck": phase_envcheck,
        "smoke": phase_smoke,
        "capture": phase_capture,
        "generate": phase_generate,
        "judge": base.phase_judge,
        "reduce": phase_reduce,
    }
    dispatch[args.phase](args)


if __name__ == "__main__":
    main()
