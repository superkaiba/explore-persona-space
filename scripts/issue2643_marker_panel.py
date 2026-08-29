#!/usr/bin/env python3
"""Issue #2643 confirmatory panel on the #382 conditional-marker organism.

The panel replays the already-generated #382 completions through the matching
merged Qwen2.5-7B checkpoints and captures layer-19 context-last / answer-mean
states.  A behavior direction is learned from *answer SAE* codes on seed 42's
first 100 prompts, then transferred through the context-SAE -> answer-SAE map
to forecast marker behavior on disjoint prompts and two unseen model seeds.

No LLM judge is used: the ground truth is the pre-registered literal ``[ZLT]``
marker.  Prompt/completion text is used transiently for replay but never
written to this task's artifacts or logs.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import shutil
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch

try:
    from scripts.issue2476_turnavg_sae import MatryoshkaBatchTopKSAE
    from scripts.issue2643_sae_map import (
        DATA_REVISION,
        DEFAULT_RIDGE_LOCAL,
        RIDGE_REPO_PATH,
        FactorizedSAEMap,
        _download_file,
        _load_ridge,
        binary_average_precision,
        binary_auroc,
        row_scores,
    )
except ModuleNotFoundError:  # pragma: no cover - direct script invocation
    from issue2476_turnavg_sae import MatryoshkaBatchTopKSAE
    from issue2643_sae_map import (
        DATA_REVISION,
        DEFAULT_RIDGE_LOCAL,
        RIDGE_REPO_PATH,
        FactorizedSAEMap,
        _download_file,
        _load_ridge,
        binary_average_precision,
        binary_auroc,
        row_scores,
    )


ISSUE = 2643
MODEL_REPO = "superkaiba1/explore-persona-space"
MODEL_REVISION = "04dfd7f293f32c93ba27d4bab68808336fae729d"
SEEDS = (42, 137, 256)
CELLS = (
    "A_phase1_assistant_trigger",
    "C_phase1_assistant_no_trigger",
    "D_phase1_villain_trigger",
    "Dprime_phase1_kt_trigger",
)
RAW_PREFIX = "issue382_marker_kl/raw_completions"
MARKER = "[zlt]"


def fit_mean_difference(
    values: torch.Tensor, labels: torch.Tensor, train_mask: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Fit a normalized positive-minus-negative mean direction."""
    x = values[train_mask].float()
    y = labels[train_mask].bool()
    if int(y.sum()) == 0 or int((~y).sum()) == 0:
        raise ValueError("mean-difference fit needs both classes")
    pos, neg = x[y].mean(0), x[~y].mean(0)
    weight = pos - neg
    norm = weight.norm()
    if float(norm) <= 0:
        raise ValueError("degenerate zero mean-difference direction")
    weight = weight / norm
    offset = 0.5 * torch.dot(pos + neg, weight)
    return {"weight": weight.cpu(), "offset": offset.cpu()}


def apply_readout(values: torch.Tensor, readout: Mapping[str, torch.Tensor]) -> torch.Tensor:
    weight = readout["weight"].to(values.device, dtype=torch.float32)
    offset = readout["offset"].to(values.device, dtype=torch.float32)
    return values.float() @ weight - offset


def clustered_auc_ci(
    labels: Sequence[int],
    scores: Sequence[float],
    clusters: Sequence[str],
    *,
    draws: int = 1000,
    seed: int = 2643,
) -> list[float]:
    """Percentile CI with the prompt/model cluster as resampling unit."""
    y = np.asarray(labels, dtype=np.int8)
    s = np.asarray(scores, dtype=np.float64)
    c = np.asarray(clusters)
    unique = np.unique(c)
    if len(unique) < 2:
        return [float("nan"), float("nan")]
    by_cluster = {key: np.flatnonzero(c == key) for key in unique}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(draws):
        chosen = rng.choice(unique, size=len(unique), replace=True)
        idx = np.concatenate([by_cluster[key] for key in chosen])
        value = binary_auroc(y[idx], s[idx])
        if np.isfinite(value):
            vals.append(value)
    if len(vals) < max(20, draws // 10):
        return [float("nan"), float("nan")]
    return [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]


def _atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, indent=2, sort_keys=True, allow_nan=True)
            f.write("\n")
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _raw_file(seed: int, cell: str) -> str:
    return f"{RAW_PREFIX}/seed{seed}/{cell}/raw_completions.json"


def _model_prefix(seed: int) -> str:
    return f"c_issue382_marker_install_kl_seed{seed}_pre_em"


def _download_model(seed: int, root: Path) -> Path:
    from huggingface_hub import snapshot_download

    prefix = _model_prefix(seed)
    snapshot_download(
        MODEL_REPO,
        repo_type="model",
        revision=MODEL_REVISION,
        allow_patterns=[f"{prefix}/*"],
        local_dir=root,
    )
    model_dir = root / prefix
    if not (model_dir / "config.json").exists():
        raise RuntimeError(f"model snapshot incomplete: {model_dir}")
    return model_dir


def _load_replay_rows(seed: int, work: Path, revision: str) -> list[dict]:
    rows = []
    for cell in CELLS:
        path = _download_file(_raw_file(seed, cell), work / "raw", revision)
        doc = json.loads(path.read_text(encoding="utf-8"))
        if doc["label"] != cell or int(doc["seed"]) != seed:
            raise RuntimeError(f"raw-completion provenance mismatch for seed={seed}, cell={cell}")
        if len(doc["records"]) != 200 or int(doc["num_completions"]) != 3:
            raise RuntimeError(f"unexpected #382 panel shape for seed={seed}, cell={cell}")
        for record in doc["records"]:
            prompt_idx = int(record["prompt_idx"])
            user = str(record["user_prompt"])
            for completion_idx, completion in enumerate(record["completions"]):
                completion = str(completion)
                rows.append(
                    {
                        "seed": seed,
                        "cell": cell,
                        "prompt_idx": prompt_idx,
                        "completion_idx": completion_idx,
                        "system": str(doc["system_prompt"]),
                        "user": user,
                        "completion": completion,
                        "marker_fired": MARKER in completion.lower(),
                        "text_sha16": hashlib.sha256(
                            (doc["system_prompt"] + "\0" + user + "\0" + completion).encode()
                        ).hexdigest()[:16],
                    }
                )
    if len(rows) != len(CELLS) * 200 * 3:
        raise RuntimeError(f"expected 2400 replay rows, got {len(rows)}")
    return rows


@torch.inference_mode()
def _capture_seed(args: argparse.Namespace, seed: int) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    work = Path(args.work_dir)
    out = Path(args.capture_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = _load_replay_rows(seed, work, args.data_revision)
    model_root = work / f"model_seed{seed}"
    model_dir = _download_model(seed, model_root)
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_dir,
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        trust_remote_code=True,
    ).eval()
    n_layers = int(model.config.num_hidden_layers)
    if n_layers != 28 or int(model.config.hidden_size) != 3584:
        raise RuntimeError(f"unexpected organism architecture: layers={n_layers}")

    x_parts, y_parts, kept_meta = [], [], []
    for start in range(0, len(rows), args.capture_batch_size):
        batch = rows[start : start + args.capture_batch_size]
        sequences, prompt_lens, local_meta = [], [], []
        for row in batch:
            prompt_ids = tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": row["system"]},
                    {"role": "user", "content": row["user"]},
                ],
                tokenize=True,
                add_generation_prompt=True,
            )
            completion_ids = tokenizer.encode(row["completion"], add_special_tokens=False)
            room = args.max_length - len(prompt_ids)
            if room < 1 or not completion_ids:
                continue
            completion_ids = completion_ids[:room]
            sequences.append(prompt_ids + completion_ids)
            prompt_lens.append(len(prompt_ids))
            local_meta.append(
                {k: v for k, v in row.items() if k not in {"system", "user", "completion"}}
            )
        if not sequences:
            continue
        max_len = max(map(len, sequences))
        ids = torch.full(
            (len(sequences), max_len),
            tokenizer.pad_token_id,
            dtype=torch.long,
            device=args.device,
        )
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(sequences):
            ids[i, : len(seq)] = torch.tensor(seq, dtype=torch.long, device=args.device)
            mask[i, : len(seq)] = 1
        outputs = model(
            input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False
        )
        h = outputs.hidden_states[19].float()
        for i, (plen, seq, meta) in enumerate(zip(prompt_lens, sequences, local_meta, strict=True)):
            x_parts.append(h[i, plen - 1].cpu().to(torch.float16))
            y_parts.append(h[i, plen : len(seq)].mean(0).cpu().to(torch.float16))
            kept_meta.append({**meta, "prompt_len": plen, "answer_len": len(seq) - plen})
        if (start // args.capture_batch_size + 1) % 50 == 0 or start + len(batch) == len(rows):
            print(f"[capture] seed={seed} rows={len(kept_meta)}/{len(rows)}", flush=True)
    if len(kept_meta) != len(rows):
        raise RuntimeError(f"capture dropped rows: kept={len(kept_meta)}, expected={len(rows)}")
    x = torch.stack(x_parts).numpy()
    y = torch.stack(y_parts).numpy()
    np.savez(out / f"seed{seed}_L19.npz", x_context=x, x_answer=y)
    _atomic_json(
        out / f"seed{seed}_rows.json",
        {
            "seed": seed,
            "model_repo": MODEL_REPO,
            "model_revision": MODEL_REVISION,
            "model_prefix": _model_prefix(seed),
            "layer": 19,
            "replay_token_policy": "chat-template prompt ids + independently re-encoded completion text",
            "rows": kept_meta,
        },
    )
    del model
    gc.collect()
    torch.cuda.empty_cache()
    if not args.keep_models:
        shutil.rmtree(model_root)


def phase_capture(args: argparse.Namespace) -> None:
    for seed in args.seeds:
        tensor_path = Path(args.capture_dir) / f"seed{seed}_L19.npz"
        rows_path = Path(args.capture_dir) / f"seed{seed}_rows.json"
        if tensor_path.exists() and rows_path.exists():
            print(f"[capture] seed={seed} already complete", flush=True)
            continue
        _capture_seed(args, seed)


def _load_mapper(args: argparse.Namespace) -> tuple[FactorizedSAEMap, dict[str, torch.Tensor]]:
    root = Path(args.work_dir) / "assets"
    context_sae = MatryoshkaBatchTopKSAE.load_local(root / "context_sae", device=args.device)
    answer_sae = MatryoshkaBatchTopKSAE.load_local(root / "answer_sae", device=args.device)
    ridge_path = Path(args.ridge) if args.ridge else DEFAULT_RIDGE_LOCAL
    if not ridge_path.exists():
        ridge_path = _download_file(RIDGE_REPO_PATH, root / "hf", args.data_revision)
    ridge = _load_ridge(ridge_path)
    for key in ("xmu", "xsd", "ymu", "W"):
        ridge[key] = torch.as_tensor(ridge[key], dtype=torch.float32, device=args.device)
    calibration = torch.load(args.calibration, map_location="cpu", weights_only=True)
    mapper = FactorizedSAEMap(
        context_sae,
        answer_sae,
        ridge,
        scale=calibration["scale"].to(args.device),
    )
    return mapper, calibration


@torch.inference_mode()
def _encode_seed(
    mapper: FactorizedSAEMap,
    calibration: Mapping[str, torch.Tensor],
    args: argparse.Namespace,
    seed: int,
) -> dict[str, object]:
    with np.load(Path(args.capture_dir) / f"seed{seed}_L19.npz") as z:
        x = torch.from_numpy(z["x_context"].copy()).float()
        y = torch.from_numpy(z["x_answer"].copy()).float()
    doc = json.loads((Path(args.capture_dir) / f"seed{seed}_rows.json").read_text())
    meta = doc["rows"]
    if x.shape != y.shape or len(meta) != x.shape[0]:
        raise RuntimeError(f"organism capture alignment failure for seed={seed}")
    tensors: defaultdict[str, list[torch.Tensor]] = defaultdict(list)
    for start in range(0, len(meta), args.encode_batch_size):
        stop = min(start + args.encode_batch_size, len(meta))
        xb, yb = x[start:stop].to(args.device), y[start:stop].to(args.device)
        pred = mapper.predict(xb)
        za = mapper.answer_sae.encode(yb)
        diag = row_scores(
            xb,
            pred["x_context_recon"],
            yb,
            pred["x_answer_pred_raw"],
            pred["x_answer_pred_sae"],
            za,
            pred["z_answer_pred"],
            pred_code_mean=calibration["pred_mean"],
            pred_code_var=calibration["pred_var"],
            pred_code_count=calibration["pred_count"],
            rarity_min_count=args.rarity_min_count,
        )
        tensors["x_context"].append(xb.cpu())
        tensors["z_context"].append(pred["z_context"].cpu().to(torch.float16))
        tensors["z_answer"].append(za.cpu().to(torch.float16))
        tensors["z_answer_pred"].append(pred["z_answer_pred"].cpu().to(torch.float16))
        for key, value in diag.items():
            tensors[f"diag:{key}"].append(value.cpu())
    return {"meta": meta, **{k: torch.cat(v) for k, v in tensors.items()}}


def _metric(labels: list[int], scores: list[float], clusters: list[str], seed: int) -> dict:
    return {
        "n": len(labels),
        "n_positive": int(sum(labels)),
        "prevalence": float(np.mean(labels)) if labels else float("nan"),
        "auroc": binary_auroc(labels, scores),
        "auroc_cluster_bootstrap_95ci": clustered_auc_ci(labels, scores, clusters, seed=seed),
        "average_precision": binary_average_precision(labels, scores),
    }


def phase_analyze(args: argparse.Namespace) -> None:
    mapper, calibration = _load_mapper(args)
    encoded = {seed: _encode_seed(mapper, calibration, args, seed) for seed in args.seeds}
    train = encoded[42]
    labels42 = torch.tensor([row["marker_fired"] for row in train["meta"]], dtype=torch.bool)
    train_mask = torch.tensor(
        [row["prompt_idx"] < args.fit_prompts for row in train["meta"]], dtype=torch.bool
    )
    readouts = {
        "mapped_answer_sae": fit_mean_difference(train["z_answer"], labels42, train_mask),
        "direct_context_sae": fit_mean_difference(train["z_context"], labels42, train_mask),
        "direct_context_dense": fit_mean_difference(train["x_context"], labels42, train_mask),
    }

    row_records = []
    for seed, data in encoded.items():
        score_tensors = {
            "mapped_answer_sae": apply_readout(
                data["z_answer_pred"], readouts["mapped_answer_sae"]
            ),
            "oracle_answer_sae": apply_readout(data["z_answer"], readouts["mapped_answer_sae"]),
            "direct_context_sae": apply_readout(data["z_context"], readouts["direct_context_sae"]),
            "direct_context_dense": apply_readout(
                data["x_context"], readouts["direct_context_dense"]
            ),
        }
        for name in (
            "forecast_context_recon_nse",
            "forecast_code_rarity",
            "post_dense_surprise_raw",
            "post_dense_surprise_ctxsae",
            "post_code_cosine_surprise",
            "post_code_relative_l2",
            "post_emergent_feature_mass",
        ):
            score_tensors[name] = data[f"diag:{name}"]
        for i, meta in enumerate(data["meta"]):
            rec = {
                **meta,
                "split": "readout_fit"
                if seed == 42 and meta["prompt_idx"] < args.fit_prompts
                else "eval",
                "condition_rule": float(meta["cell"] == CELLS[0]),
            }
            rec.update({name: float(values[i]) for name, values in score_tensors.items()})
            row_records.append(rec)

    eval_rows = [row for row in row_records if row["split"] == "eval"]
    score_names = [
        "mapped_answer_sae",
        "direct_context_sae",
        "direct_context_dense",
        "oracle_answer_sae",
        "condition_rule",
        "forecast_context_recon_nse",
        "forecast_code_rarity",
        "post_dense_surprise_raw",
        "post_dense_surprise_ctxsae",
        "post_code_cosine_surprise",
        "post_code_relative_l2",
        "post_emergent_feature_mass",
    ]

    def evaluate(rows: list[dict], bootstrap_seed: int) -> dict:
        labels = [int(row["marker_fired"]) for row in rows]
        clusters = [f"{row['seed']}:{row['prompt_idx']}" for row in rows]
        return {
            name: _metric(labels, [row[name] for row in rows], clusters, bootstrap_seed + i)
            for i, name in enumerate(score_names)
        }

    results = {
        "all_eval": evaluate(eval_rows, 264300),
        "by_seed": {
            str(seed): evaluate([r for r in eval_rows if r["seed"] == seed], 264300 + seed)
            for seed in args.seeds
        },
        "target_condition_only": evaluate([r for r in eval_rows if r["cell"] == CELLS[0]], 264399),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "row_scores.jsonl").open("w", encoding="utf-8") as f:
        for row in row_records:
            f.write(json.dumps(row, sort_keys=True, allow_nan=True) + "\n")
    torch.save(readouts, out / "readouts.pt")
    summary = {
        "issue": ISSUE,
        "status": "confirmatory_marker_panel_complete",
        "organism": {
            "task": 382,
            "behavior": "literal [ZLT] conditional marker",
            "model": "Qwen2.5-7B-Instruct, merged marker-installed checkpoints",
            "model_revision": MODEL_REVISION,
            "seeds": list(args.seeds),
            "cells": list(CELLS),
            "rows_per_seed": len(encoded[args.seeds[0]]["meta"]),
        },
        "readout_fit": {
            "seed": 42,
            "prompt_indices": [0, args.fit_prompts - 1],
            "feature_direction": "positive-minus-negative mean in realized answer-SAE space",
            "mapped_test": "same answer-space direction applied to predicted answer-SAE code",
            "baselines": [
                "label-budget-matched context-SAE mean difference",
                "label-budget-matched raw-context mean difference",
                "realized answer-SAE oracle",
                "known trigger-condition rule",
            ],
        },
        "results": results,
        "interpretation_limits": [
            "All-cell marker detection is partly detection of the known trigger/persona condition.",
            "Target-condition-only detection asks the harder question of stochastic firing within an identical pre-answer condition.",
            "The map adds no information over context states; success means answer-feature transfer or label efficiency, not new information.",
            "Teacher-forced completion text was independently re-tokenized because the archived #382 files do not retain token IDs.",
        ],
    }
    _atomic_json(out / "summary.json", summary)
    print(json.dumps(summary["results"], indent=2, allow_nan=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("capture", "analyze", "all"), required=True)
    p.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--capture-batch-size", type=int, default=8)
    p.add_argument("--encode-batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=2048)
    p.add_argument("--fit-prompts", type=int, default=100)
    p.add_argument("--rarity-min-count", type=int, default=32)
    p.add_argument("--data-revision", default=DATA_REVISION)
    p.add_argument("--work-dir", default="/workspace/issue2643/work")
    p.add_argument("--capture-dir", default="/workspace/issue2643/marker_capture")
    p.add_argument("--calibration", default="/workspace/issue2643/full/feature_calibration.pt")
    p.add_argument("--ridge", default="")
    p.add_argument("--out", default="/workspace/issue2643/marker_panel")
    p.add_argument("--keep-models", action="store_true")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if 42 not in args.seeds:
        raise SystemExit("seed 42 is required for the frozen readout fit")
    if args.phase in {"capture", "all"}:
        phase_capture(args)
    if args.phase in {"analyze", "all"}:
        phase_analyze(args)


if __name__ == "__main__":
    main()
