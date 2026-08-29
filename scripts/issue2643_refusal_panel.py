#!/usr/bin/env python3
"""Issue #2643 confirmatory over-refusal panel using the #642 LoRA organism.

This replays one archived rollout per (persona, benign request) from the strong
``loraRefOP_step132`` checkpoint.  It reuses the already-frozen #642 refusal
labels; it makes no new judge/API calls.  A refusal direction fitted in realized
answer-SAE space on claims 0..24 is applied to mapped answer-SAE forecasts on
claims 25..49 and compared with label-budget-matched context probes.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch

try:
    from scripts.issue2643_gradient_pursuit import (
        apply_behavior_pursuit,
        behavior_pursuit_summary,
        factorized_local_coefficients,
        fit_behavior_pursuit,
    )
    from scripts.issue2643_marker_panel import (
        _atomic_json,
        _load_mapper,
        _metric,
        apply_readout,
        clustered_auc_delta,
        fit_mean_difference,
    )
    from scripts.issue2643_sae_map import row_scores
except ModuleNotFoundError:  # pragma: no cover - direct invocation
    from issue2643_gradient_pursuit import (
        apply_behavior_pursuit,
        behavior_pursuit_summary,
        factorized_local_coefficients,
        fit_behavior_pursuit,
    )
    from issue2643_marker_panel import (
        _atomic_json,
        _load_mapper,
        _metric,
        apply_readout,
        clustered_auc_delta,
        fit_mean_difference,
    )
    from issue2643_sae_map import row_scores


ISSUE = 2643
MODEL_REPO = "superkaiba1/explore-persona-space"
MODEL_REVISION = "c76b5432fb1b134482babb9e60f8a6d92fc264ee"
ADAPTER_PREFIX = "adapters/issue_642/v9/loraRefOP_villain_seed42/step132"
CELL = "loraRefOP_step132"
GEN_DIR = Path("eval_results/issue_642/refusal/generations") / CELL
VERDICT_DIR = Path("eval_results/issue_642/refusal/verdicts")


def _load_rows(rollout_indices: set[int]) -> list[dict]:
    rows = []
    gen_files = sorted(GEN_DIR.glob("refusal_eval_*.json"))
    if len(gen_files) != 30:
        raise RuntimeError(
            f"expected 30 tracked #642 persona files under {GEN_DIR}, got {len(gen_files)}"
        )
    for gen_path in gen_files:
        gen = json.loads(gen_path.read_text(encoding="utf-8"))
        persona = str(gen["panel_persona"])
        verdict_path = VERDICT_DIR / f"{CELL}__{persona}.json"
        verdict = json.loads(verdict_path.read_text(encoding="utf-8"))
        completions = gen["completions"]
        labels = verdict["verdicts"]
        if len(completions) != 500 or len(labels) != 500:
            raise RuntimeError(f"unexpected #642 row count for persona={persona}")
        for completion, label in zip(completions, labels, strict=True):
            if int(completion["claim_idx"]) != int(label["claim_idx"]):
                raise RuntimeError(f"#642 verdict alignment failure for persona={persona}")
            rollout_idx = int(completion["rollout_idx"])
            if rollout_idx not in rollout_indices or label.get("degenerate"):
                continue
            system = str(gen["panel_prompt"])
            user = str(completion["claim"])
            answer = str(completion["completion"])
            rows.append(
                {
                    "persona": persona,
                    "claim_idx": int(completion["claim_idx"]),
                    "rollout_idx": rollout_idx,
                    "refused": bool(label["agreed"]),
                    "system": system,
                    "user": user,
                    "completion": answer,
                    "text_sha16": hashlib.sha256(
                        (system + "\0" + user + "\0" + answer).encode()
                    ).hexdigest()[:16],
                }
            )
    expected = 30 * 50 * len(rollout_indices)
    if len(rows) != expected:
        raise RuntimeError(f"expected {expected} nondegenerate selected rows, got {len(rows)}")
    return rows


def _download_adapter(root: Path) -> Path:
    from huggingface_hub import HfApi, hf_hub_download

    files = [
        entry.path
        for entry in HfApi().list_repo_tree(
            MODEL_REPO,
            repo_type="model",
            revision=MODEL_REVISION,
            path_in_repo=ADAPTER_PREFIX,
            recursive=True,
            expand=False,
        )
        if getattr(entry, "path", "").startswith(f"{ADAPTER_PREFIX}/")
    ]
    if not files:
        raise RuntimeError(f"no adapter files under {ADAPTER_PREFIX} at {MODEL_REVISION}")

    def download(filename: str) -> None:
        hf_hub_download(
            MODEL_REPO,
            filename=filename,
            repo_type="model",
            revision=MODEL_REVISION,
            local_dir=root,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(download, files))
    adapter = root / ADAPTER_PREFIX
    if not (adapter / "adapter_model.safetensors").exists():
        raise RuntimeError(f"adapter snapshot incomplete: {adapter}")
    return adapter


@torch.inference_mode()
def phase_capture(args: argparse.Namespace) -> None:
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    rollout_indices = {int(x) for x in args.rollout_indices}
    rows = _load_rows(rollout_indices)
    adapter = _download_adapter(Path(args.work_dir) / "refusal_adapter")
    tokenizer = AutoTokenizer.from_pretrained(adapter, trust_remote_code=True)
    tokenizer.padding_side = "right"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct",
        torch_dtype=torch.bfloat16,
        device_map={"": args.device},
        trust_remote_code=True,
    )
    model = PeftModel.from_pretrained(base, adapter).eval()
    if int(model.config.num_hidden_layers) != 28 or int(model.config.hidden_size) != 3584:
        raise RuntimeError("unexpected refusal-organism architecture")

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
            answer_ids = tokenizer.encode(row["completion"], add_special_tokens=False)
            room = args.max_length - len(prompt_ids)
            if room < 1 or not answer_ids:
                continue
            answer_ids = answer_ids[:room]
            sequences.append(prompt_ids + answer_ids)
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
            ids[i, : len(seq)] = torch.tensor(seq, device=args.device)
            mask[i, : len(seq)] = 1
        outputs = model(
            input_ids=ids, attention_mask=mask, output_hidden_states=True, use_cache=False
        )
        h = outputs.hidden_states[19].float()
        for i, (plen, seq, meta) in enumerate(zip(prompt_lens, sequences, local_meta, strict=True)):
            x_parts.append(h[i, plen - 1].cpu().to(torch.float16))
            y_parts.append(h[i, plen : len(seq)].mean(0).cpu().to(torch.float16))
            kept_meta.append({**meta, "prompt_len": plen, "answer_len": len(seq) - plen})
        if (start // args.capture_batch_size + 1) % 25 == 0 or start + len(batch) == len(rows):
            print(f"[refusal-capture] rows={len(kept_meta)}/{len(rows)}", flush=True)
    if len(kept_meta) != len(rows):
        raise RuntimeError(f"refusal capture dropped rows: {len(kept_meta)} != {len(rows)}")
    out = Path(args.capture_dir)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(
        out / "loraRefOP_step132_L19.npz",
        x_context=torch.stack(x_parts).numpy(),
        x_answer=torch.stack(y_parts).numpy(),
    )
    _atomic_json(
        out / "loraRefOP_step132_rows.json",
        {
            "task": 642,
            "cell": CELL,
            "adapter_repo": MODEL_REPO,
            "adapter_revision": MODEL_REVISION,
            "adapter_prefix": ADAPTER_PREFIX,
            "layer": 19,
            "rollout_indices": sorted(rollout_indices),
            "label_provenance": "frozen #642 claude-haiku-4-5-20251001 verdicts; no new judge calls",
            "rows": kept_meta,
        },
    )
    del model, base
    gc.collect()
    torch.cuda.empty_cache()


@torch.inference_mode()
def phase_analyze(args: argparse.Namespace) -> None:
    mapper, calibration = _load_mapper(args)
    with np.load(Path(args.capture_dir) / "loraRefOP_step132_L19.npz") as z:
        x = torch.from_numpy(z["x_context"].copy()).float()
        y = torch.from_numpy(z["x_answer"].copy()).float()
    doc = json.loads(
        (Path(args.capture_dir) / "loraRefOP_step132_rows.json").read_text(encoding="utf-8")
    )
    meta = doc["rows"]
    if x.shape != y.shape or len(meta) != x.shape[0]:
        raise RuntimeError("refusal capture alignment failure")
    parts: dict[str, list[torch.Tensor]] = {
        "x": [],
        "zc": [],
        "za": [],
        "zp": [],
    }
    diag_parts: dict[str, list[torch.Tensor]] = {}
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
        parts["x"].append(xb.cpu())
        parts["zc"].append(pred["z_context"].cpu().half())
        parts["za"].append(za.cpu().half())
        parts["zp"].append(pred["z_answer_pred"].cpu().half())
        for key, value in diag.items():
            diag_parts.setdefault(key, []).append(value.cpu())
    tensors = {key: torch.cat(value) for key, value in parts.items()}
    diags = {key: torch.cat(value) for key, value in diag_parts.items()}
    labels = torch.tensor([row["refused"] for row in meta], dtype=torch.bool)
    train_mask = torch.tensor([row["claim_idx"] < args.fit_claims for row in meta])
    readouts = {
        "answer": fit_mean_difference(tensors["za"], labels, train_mask),
        "context_sae": fit_mean_difference(tensors["zc"], labels, train_mask),
        "context_dense": fit_mean_difference(tensors["x"], labels, train_mask),
    }
    pursuit_fit = fit_behavior_pursuit(
        tensors["zc"],
        apply_readout(tensors["zp"], readouts["answer"]),
        train_mask,
        factorized_local_coefficients(mapper, readouts["answer"]["weight"]),
        candidates=args.pursuit_candidates,
        k_ladder=args.pursuit_k_ladder,
        ridge_relative=args.pursuit_ridge_relative,
    )
    scores = {
        "mapped_answer_sae": apply_readout(tensors["zp"], readouts["answer"]),
        "oracle_answer_sae": apply_readout(tensors["za"], readouts["answer"]),
        "direct_context_sae": apply_readout(tensors["zc"], readouts["context_sae"]),
        "direct_context_dense": apply_readout(tensors["x"], readouts["context_dense"]),
    }
    scores.update(apply_behavior_pursuit(pursuit_fit, tensors["zc"]))
    for name in (
        "forecast_context_recon_nse",
        "forecast_mapped_answer_norm",
        "forecast_pred_l0",
        "forecast_code_rarity",
        "post_dense_surprise_raw",
        "post_dense_surprise_ctxsae",
        "post_code_cosine_surprise",
        "post_code_relative_l2",
        "post_emergent_feature_mass",
        "control_answer_l0",
    ):
        scores[name] = diags[name]
    scores["prompt_len"] = torch.tensor([row["prompt_len"] for row in meta])
    scores["answer_len"] = torch.tensor([row["answer_len"] for row in meta])
    eval_idx = [i for i, row in enumerate(meta) if row["claim_idx"] >= args.fit_claims]
    eval_labels = [int(meta[i]["refused"]) for i in eval_idx]
    clusters = [f"{meta[i]['persona']}:{meta[i]['claim_idx']}" for i in eval_idx]
    metrics = {
        name: _metric(eval_labels, [float(value[i]) for i in eval_idx], clusters, 264600 + j)
        for j, (name, value) in enumerate(scores.items())
    }
    pursuit_names = sorted(name for name in scores if "_k" in name)
    mapped_eval = np.asarray(
        [float(scores["mapped_answer_sae"][i]) for i in eval_idx], dtype=np.float64
    )
    map_den = float(np.square(mapped_eval - mapped_eval.mean()).sum())
    pursuit_fidelity = {
        name: float(
            1.0
            - np.square(
                np.asarray([float(scores[name][i]) for i in eval_idx], dtype=np.float64)
                - mapped_eval
            ).sum()
            / max(map_den, 1e-24)
        )
        for name in pursuit_names
    }
    pursuit_contrasts = {
        f"gradient_pursuit_k{k}_minus_{baseline}": clustered_auc_delta(
            eval_labels,
            [float(scores[f"gradient_pursuit_k{k}"][i]) for i in eval_idx],
            [float(scores[baseline][i]) for i in eval_idx],
            clusters,
            seed=264640 + k * 10 + baseline_idx,
        )
        for k in pursuit_fit.k_ladder
        for baseline_idx, baseline in enumerate(
            (f"magnitude_refit_k{k}", f"magnitude_fixed_k{k}", "mapped_answer_sae")
        )
    }
    per_persona = {}
    for persona in sorted({row["persona"] for row in meta}):
        idx = [i for i in eval_idx if meta[i]["persona"] == persona]
        per_persona[persona] = {
            "n": len(idx),
            "refusal_rate": float(np.mean([meta[i]["refused"] for i in idx])),
            "mapped_answer_sae_mean": float(
                np.mean([float(scores["mapped_answer_sae"][i]) for i in idx])
            ),
            "oracle_answer_sae_mean": float(
                np.mean([float(scores["oracle_answer_sae"][i]) for i in idx])
            ),
        }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "row_scores.jsonl").open("w", encoding="utf-8") as f:
        for i, row in enumerate(meta):
            rec = {
                **row,
                "split": "readout_fit" if row["claim_idx"] < args.fit_claims else "eval",
                **{name: float(value[i]) for name, value in scores.items()},
            }
            f.write(json.dumps(rec, sort_keys=True, allow_nan=True) + "\n")
    torch.save(readouts, out / "readouts.pt")
    summary = {
        "issue": ISSUE,
        "status": "confirmatory_refusal_panel_complete",
        "organism": {
            "task": 642,
            "cell": CELL,
            "behavior": "over-refusal on benign requests",
            "adapter_revision": doc["adapter_revision"],
            "n_personas": 30,
            "n_rows": len(meta),
            "rollout_indices": doc["rollout_indices"],
            "label_provenance": doc["label_provenance"],
        },
        "readout_fit": {
            "claims": [0, args.fit_claims - 1],
            "evaluation_claims": [args.fit_claims, 49],
            "direction": "refusal-positive minus refusal-negative mean in answer-SAE space",
        },
        "gradient_pursuit": {
            **behavior_pursuit_summary(pursuit_fit),
            "fit_rows": int(train_mask.sum()),
            "fit_scope": "claims in the frozen readout-fit split",
            "heldout_full_map_fidelity_r2": pursuit_fidelity,
            "heldout_behavior_auc_contrasts": pursuit_contrasts,
        },
        "metrics": metrics,
        "per_persona": per_persona,
        "interpretation_limits": [
            "This tests one strong LoRA checkpoint and archived rollout index/indices, not every refusal organism.",
            "The map adds no information over context states; the direct context probes are the required comparison.",
            "Archived text lacks completion token IDs, so teacher-forced completions are independently re-tokenized.",
        ],
    }
    _atomic_json(out / "summary.json", summary)
    print(json.dumps(metrics, indent=2, allow_nan=True), flush=True)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--phase", choices=("capture", "analyze", "all"), required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--rollout-indices", nargs="+", type=int, default=[0])
    p.add_argument("--capture-batch-size", type=int, default=8)
    p.add_argument("--encode-batch-size", type=int, default=64)
    p.add_argument("--max-length", type=int, default=1024)
    p.add_argument("--fit-claims", type=int, default=25)
    p.add_argument("--rarity-min-count", type=int, default=32)
    p.add_argument("--pursuit-candidates", type=int, default=128)
    p.add_argument("--pursuit-k-ladder", nargs="+", type=int, default=[1, 2, 4, 8, 16])
    p.add_argument("--pursuit-ridge-relative", type=float, default=1e-3)
    p.add_argument("--work-dir", default="/workspace/issue2643/work")
    p.add_argument("--capture-dir", default="/workspace/issue2643/refusal_capture")
    p.add_argument("--calibration", default="/workspace/issue2643/full/feature_calibration.pt")
    p.add_argument("--ridge", default="")
    p.add_argument("--data-revision", default="cd80ba2588bb6d4291edf621176ea654bcbf2507")
    p.add_argument("--out", default="/workspace/issue2643/refusal_panel")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.phase in {"capture", "all"}:
        phase_capture(args)
    if args.phase in {"analyze", "all"}:
        phase_analyze(args)


if __name__ == "__main__":
    main()
