#!/usr/bin/env python3
# ruff: noqa: RUF002, RUF003
# Intentional Unicode (ρ, ※, Δ, ×, —) in scientific docstrings + logs.
"""Issue #531 follow-up — logit re-scoring pass over #478's CORE cells.

#478's per-cell ``result.json`` persisted only resolved log-probabilities
(``log P(※) = z_※ − log Z``), so the logit-space view of the #531 base-prior
plots is not derivable from stored data. This script re-runs the EXACT #478
scoring forward pass (same chat template, same stored on-policy responses,
same left-padded last-slot readout — see ``issue478_run_cell.py::
score_logprob_and_kl`` at the pinned SHA) and additionally captures, at the
post-response slot, for trained AND base:

- ``z_marker``  — the raw (pre-softmax) logit of the marker token id 83399,
- ``logZ``      — ``logsumexp`` over the full-vocab logits,
- ``logp``      — recomputed ``z_marker − logZ`` (validated against the stored
  per-question log-probs from the parent tidy table; a construction error in
  prompts/slot shows up as a huge mismatch, bf16 merged-vs-PEFT noise as a
  small one),
- ``argmax_id`` — argmax token at the slot.

Trained scoring loads the per-cell LoRA adapter from the HF model repo onto a
shared base model (PEFT hot-swap); #478 scored a bf16-merged checkpoint, which
is numerically equivalent up to bf16 rounding (validation quantifies it). Base
scoring disables the adapter on the same forward batch construction.

Outputs one JSON per cell to ``eval_results/issue_478/logit_rescore/``
(written the moment the cell completes — checkpoint-per-phase).

Parallelism: launch one process per GPU with ``CUDA_VISIBLE_DEVICES=i`` and
``--shard i/N`` (cells split round-robin).

Usage::

    CUDA_VISIBLE_DEVICES=0 uv run python scripts/issue531_logit_rescore.py --shard 0/4
    uv run python scripts/issue531_logit_rescore.py --cells K1_c00_seed42 --batch-size 8
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

# ── Constants (pinned to #478 / #531 provenance) ─────────────────────────────

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"  # per #478 result.json training.base_model
MARKER_TEXT = " ※"
MARKER_ID = 83399
EOS_TOKEN = "<|im_end|>"  # the token contrastive negatives train at the slot
EOS_ID = 151645

HF_MODEL_REPO = "superkaiba1/explore-persona-space"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_DATA_REV = "a9fc5a9cbc81c4b774ff66da0022f9055e18da5f"  # pinned #478 revision

TIDY_PARQUET = (
    PROJECT_ROOT / "eval_results" / "issue_478" / "base_prior_reanalysis" / "tidy.parquet"
)
OUTPUT_DIR = PROJECT_ROOT / "eval_results" / "issue_478" / "logit_rescore"

# Validation gates: a wrong prompt/slot construction produces tens-of-nats
# mismatches; bf16 merged-checkpoint vs PEFT-runtime noise stays well under 1.
MAX_VALIDATION_MAE_NATS = 1.0
MIN_VALIDATION_SPEARMAN = 0.995

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("issue531_logit_rescore")


def list_core_cells() -> list[str]:
    """The 80 CORE (cell, seed) runs, derived from the parent tidy table.

    Returns HF directory stems like ``K1_c00_seed42`` sorted lexicographically,
    locked to exactly the runs the #531 analysis used.
    """
    df = pd.read_parquet(TIDY_PARQUET, columns=["cell_id", "seed"])
    pairs = sorted({(c, int(s)) for c, s in zip(df["cell_id"], df["seed"], strict=True)})
    return [f"{c}_seed{s}" for c, s in pairs]


def load_raw_completions(cell: str) -> dict:
    """Download + parse the cell's raw_completions.json at the pinned revision."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        HF_DATA_REPO,
        f"issue_478/{cell}/raw_completions/raw_completions.json",
        repo_type="dataset",
        revision=HF_DATA_REV,
    )
    with open(path) as f:
        return json.load(f)


def resolve_model_repo_revision() -> str:
    """Resolve the model repo's current main commit ONCE per worker.

    Pinning one sha keeps the two per-cell ``hf_hub_download`` calls in the
    same snapshot dir even if the repo advances mid-run, and goes into the
    output payload for reproducibility.
    """
    from huggingface_hub import HfApi

    return HfApi().repo_info(HF_MODEL_REPO, repo_type="model").sha


def download_adapter(cell: str, revision: str) -> Path:
    """Download the cell's LoRA adapter files (config + weights) from the model repo.

    Uses per-file ``hf_hub_download``, NOT ``snapshot_download`` with
    ``allow_patterns``: the model repo is large enough that the API truncates
    the file listing ``snapshot_download`` filters against, so its pattern
    match silently yields 0 files (observed on this repo; even an exact file
    path as an allow_pattern downloaded nothing).
    """
    from huggingface_hub import hf_hub_download

    cfg_path = hf_hub_download(
        HF_MODEL_REPO,
        f"issue_478/{cell}/adapter/adapter_config.json",
        revision=revision,
    )
    hf_hub_download(
        HF_MODEL_REPO,
        f"issue_478/{cell}/adapter/adapter_model.safetensors",
        revision=revision,
    )
    adapter_dir = Path(cfg_path).parent
    if not (adapter_dir / "adapter_model.safetensors").exists():
        raise FileNotFoundError(f"adapter_model.safetensors missing under {adapter_dir}")
    _assert_gauge_free(adapter_dir)
    return adapter_dir


def _assert_gauge_free(adapter_dir: Path) -> None:
    """Gauge assert per the marker-leakage rule: the trained − base logit readout
    is valid only if LoRA never touches the unembedding (or anything tied to it).
    """
    cfg = json.loads((adapter_dir / "adapter_config.json").read_text())
    targets = cfg.get("target_modules") or []
    banned = {"lm_head", "embed_tokens"}
    hit = banned.intersection(targets)
    if hit:
        raise RuntimeError(
            f"{adapter_dir}: adapter targets {sorted(hit)} — logit readout is "
            f"gauge-dependent and INVALID for this run"
        )
    saved = cfg.get("modules_to_save") or []
    if saved:
        raise RuntimeError(
            f"{adapter_dir}: modules_to_save={saved!r} non-empty — full-module "
            f"saves can move the unembedding; logit readout invalid"
        )


def build_items(
    raw: dict,
    tokenizer,
    persona_prompts: dict[str, str],
) -> list[tuple[str, int, str]]:
    """(persona, question_idx, full_prefix) for every held-out (persona, q).

    Replicates ``score_logprob_and_kl`` prefix construction EXACTLY:
    chat template with ``add_generation_prompt=True`` then the stored
    on-policy response appended verbatim — the slot AFTER R is scored.
    """
    eval_questions: list[str] = raw["eval_questions"]
    held_out: list[str] = raw["spec"]["held_out"]
    R_eval: dict[str, dict[str, str]] = raw["R_eval"]

    items: list[tuple[str, int, str]] = []
    for persona in held_out:
        qmap = R_eval[persona]
        q_keys = list(qmap.keys())
        if q_keys != eval_questions:
            raise ValueError(
                f"R_eval question order mismatch for persona {persona!r} — "
                f"question_idx alignment with the parent tidy would be wrong"
            )
        sys_prompt = persona_prompts[persona]
        for q_idx, q in enumerate(eval_questions):
            messages = [
                {"role": "system", "content": sys_prompt},
                {"role": "user", "content": q},
            ]
            prefix = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            items.append((persona, q_idx, prefix + qmap[q]))
    return items


def score_slot(
    model,
    tokenizer,
    items: list[tuple[str, int, str]],
    device: str,
    batch_size: int,
) -> dict[tuple[str, int], dict[str, float]]:
    """Score the post-response slot for every item with the CURRENT model state.

    Returns {(persona, q_idx): {z_marker, logZ, logp, argmax_id}}. Left-pads
    within batch and reads ``logits[:, -1, :]`` like the #478 scorer; batches
    are length-sorted to cut padding waste (numerically irrelevant to the
    per-row last-slot readout).
    """
    import torch
    import torch.nn.functional as F

    pad_id = tokenizer.pad_token_id
    if pad_id is None:
        pad_id = tokenizer.eos_token_id

    encoded = [
        (persona, q_idx, tokenizer.encode(text, add_special_tokens=False))
        for persona, q_idx, text in items
    ]
    order = sorted(range(len(encoded)), key=lambda i: len(encoded[i][2]))

    out: dict[tuple[str, int], dict[str, float]] = {}
    for start in range(0, len(order), batch_size):
        chunk = [encoded[i] for i in order[start : start + batch_size]]
        max_len = max(len(ids) for _, _, ids in chunk)
        padded = [[pad_id] * (max_len - len(ids)) + ids for _, _, ids in chunk]
        attn = [[0] * (max_len - len(ids)) + [1] * len(ids) for _, _, ids in chunk]
        input_ids = torch.tensor(padded, dtype=torch.long, device=device)
        attention_mask = torch.tensor(attn, dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attention_mask).logits

        last = logits[:, -1, :].float()  # (B, V) — slot after R (left-padded)
        assert last.shape[0] == len(chunk), last.shape
        logz = torch.logsumexp(last, dim=-1)
        z_marker = last[:, MARKER_ID]
        z_eos = last[:, EOS_ID]
        argmax_ids = last.argmax(dim=-1)
        logp = F.log_softmax(last, dim=-1)[:, MARKER_ID]

        for (persona, q_idx, _), z, ze, lz, lp, am in zip(
            chunk,
            z_marker.cpu().tolist(),
            z_eos.cpu().tolist(),
            logz.cpu().tolist(),
            logp.cpu().tolist(),
            argmax_ids.cpu().tolist(),
            strict=True,
        ):
            out[(persona, q_idx)] = {
                "z_marker": float(z),
                "z_eos": float(ze),
                "logZ": float(lz),
                "logp": float(lp),
                "argmax_id": int(am),
            }
        del logits, last, logz, z_marker, z_eos, argmax_ids, logp
    return out


def validate_against_tidy(
    cell: str,
    tidy: pd.DataFrame,
    trained: dict[tuple[str, int], dict[str, float]],
    base: dict[tuple[str, int], dict[str, float]],
) -> dict:
    """Compare recomputed log-probs to the stored #478 values for this cell.

    Fails loud past MAX_VALIDATION_MAE_NATS / below MIN_VALIDATION_SPEARMAN —
    that magnitude means wrong prompts/slot, not bf16 noise.
    """
    from scipy.stats import spearmanr

    cell_id, seed = cell.rsplit("_seed", 1)
    sub = tidy[(tidy["cell_id"] == cell_id) & (tidy["seed"] == int(seed))]
    if len(sub) == 0:
        raise ValueError(f"no tidy rows for {cell} — cell list / tidy mismatch")

    stored_t, recomputed_t, stored_b, recomputed_b = [], [], [], []
    for row in sub.itertuples(index=False):
        key = (row.held_out_persona, int(row.question_idx))
        stored_t.append(row.trained_logp)
        recomputed_t.append(trained[key]["logp"])
        stored_b.append(row.base_prior)
        recomputed_b.append(base[key]["logp"])

    res = {}
    for name, stored, recomputed in [
        ("trained", stored_t, recomputed_t),
        ("base", stored_b, recomputed_b),
    ]:
        s, r = np.asarray(stored), np.asarray(recomputed)
        mae = float(np.mean(np.abs(s - r)))
        max_abs = float(np.max(np.abs(s - r)))
        rho = float(spearmanr(s, r).statistic)
        res[name] = {"mae_nats": mae, "max_abs_nats": max_abs, "spearman": rho, "n": len(s)}
        log.info(
            "[%s] validation %s: MAE=%.4f nats, max=%.4f, spearman=%.5f (n=%d)",
            cell,
            name,
            mae,
            max_abs,
            rho,
            len(s),
        )
        if mae > MAX_VALIDATION_MAE_NATS or rho < MIN_VALIDATION_SPEARMAN:
            raise RuntimeError(
                f"[{cell}] {name} validation FAILED: MAE={mae:.4f} nats "
                f"(gate {MAX_VALIDATION_MAE_NATS}), spearman={rho:.5f} "
                f"(gate {MIN_VALIDATION_SPEARMAN}) — prompt/slot construction "
                f"likely diverges from #478's scorer"
            )
    return res


def process_cell(
    cell: str,
    peft_model,
    tokenizer,
    persona_prompts: dict[str, str],
    tidy: pd.DataFrame,
    device: str,
    batch_size: int,
    model_repo_rev: str,
) -> None:
    """Score one (cell, seed) run trained + base and write its JSON."""
    out_path = OUTPUT_DIR / f"{cell}.json"
    if out_path.exists():
        log.info("[%s] output exists — skipping (idempotent re-run)", cell)
        return

    raw = load_raw_completions(cell)
    items = build_items(raw, tokenizer, persona_prompts)
    log.info("[%s] %d held-out rows to score", cell, len(items))

    adapter_dir = download_adapter(cell, model_repo_rev)
    peft_model.load_adapter(str(adapter_dir), adapter_name=cell)
    peft_model.set_adapter(cell)
    trained = score_slot(peft_model, tokenizer, items, device, batch_size)

    with peft_model.disable_adapter():
        base = score_slot(peft_model, tokenizer, items, device, batch_size)

    peft_model.delete_adapter(cell)
    gc.collect()

    validation = validate_against_tidy(cell, tidy, trained, base)

    per_persona: dict[str, dict] = {}
    for persona in raw["spec"]["held_out"]:
        n_q = len(raw["eval_questions"])
        per_persona[persona] = {
            field + "_" + side + "_per_q": [
                (trained if side == "trained" else base)[(persona, qi)][field] for qi in range(n_q)
            ]
            for side in ("trained", "base")
            for field in ("z_marker", "z_eos", "logZ", "logp", "argmax_id")
        }

    payload = {
        "cell": cell,
        "base_model": BASE_MODEL,
        "marker_text": MARKER_TEXT,
        "marker_token_id": MARKER_ID,
        "eos_token": EOS_TOKEN,
        "eos_token_id": EOS_ID,
        "hf_data_revision": HF_DATA_REV,
        "hf_model_repo_revision": model_repo_rev,
        "scored_at_utc": datetime.now(UTC).isoformat(),
        "produced_by": "scripts/issue531_logit_rescore.py",
        "validation_vs_stored_logp": validation,
        "held_out": per_persona,
    }
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(payload))
    tmp.rename(out_path)
    log.info("[%s] wrote %s", cell, out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--shard", default=None, help="i/N round-robin split of the 80 cells")
    parser.add_argument("--cells", default=None, help="comma-separated explicit cell stems")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit-cells", type=int, default=None, help="smoke: first N cells")
    args = parser.parse_args()

    # .env (HF_TOKEN etc.) + the RunPod-persistent HF cache. setup_worker() is
    # deliberately NOT used — it clobbers CUDA_VISIBLE_DEVICES, which the
    # launcher sets per shard.
    load_dotenv()
    if Path("/workspace").exists():
        os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

    cells = list_core_cells()
    if args.cells:
        explicit = args.cells.split(",")
        unknown = [c for c in explicit if c not in cells]
        if unknown:
            raise SystemExit(f"unknown cells (not in the 80 CORE runs): {unknown}")
        cells = explicit
    elif args.shard:
        i, n = (int(x) for x in args.shard.split("/"))
        cells = cells[i::n]
    if args.limit_cells:
        cells = cells[: args.limit_cells]
    log.info("This worker handles %d cells: %s ...", len(cells), cells[:4])

    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    assert tokenizer.encode(MARKER_TEXT, add_special_tokens=False) == [MARKER_ID]
    assert tokenizer.convert_tokens_to_ids(EOS_TOKEN) == EOS_ID

    from run_100_persona_leakage import ALL_EVAL_PERSONAS

    persona_prompts = {name: info["prompt"] for name, info in ALL_EVAL_PERSONAS.items()}

    tidy = pd.read_parquet(TIDY_PARQUET)

    device = "cuda:0"
    log.info("Loading base model %s ...", BASE_MODEL)
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    base_model.eval()

    model_repo_rev = resolve_model_repo_revision()
    log.info("Model repo revision pinned: %s", model_repo_rev)

    # PeftModel needs an initial adapter; load the first cell's and name it.
    first_adapter = download_adapter(cells[0], model_repo_rev)
    peft_model = PeftModel.from_pretrained(base_model, str(first_adapter), adapter_name="_boot")
    peft_model.eval()

    for idx, cell in enumerate(cells):
        log.info("=== cell %d/%d: %s ===", idx + 1, len(cells), cell)
        process_cell(
            cell,
            peft_model,
            tokenizer,
            persona_prompts,
            tidy,
            device,
            args.batch_size,
            model_repo_rev,
        )

    log.info("Worker done: %d cells.", len(cells))
    return 0


if __name__ == "__main__":
    sys.exit(main())
