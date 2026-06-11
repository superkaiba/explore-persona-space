#!/usr/bin/env python3
"""#552 contrastive-2x2-completion — per-row-type CE diagnostic (plan v5 MF-A).

Delivered-contrast manipulation check: for a given model (merged dir or the
base hub id) and training mix, compute the mean per-token cross-entropy on a
fixed deterministic subsample of 512 POSITIVE + 512 NEGATIVE rows,
teacher-forced over the full rendered chat-template sequence — the same
effective loss surface as training (full-sequence CE via the pinned TRL
0.29.1 fallback). Trained-vs-base deltas decide whether the negative rows
actually carried gradient (plan §6.3: contrast DELIVERED for an arm iff
median over its 3 seeds of ``|delta_ce_neg_vs_base| >= 0.05`` nat/token).

Subsample determinism (plan §6.3/§11): row selection is keyed on
``question_index`` with ``random.Random(SUBSAMPLE_SEED)`` (seed 0) — one draw
for positives, one for negatives — so the negative subsample is IDENTICAL
across arms by construction (both mixes carry byte-identical negative rows at
the same question indices) and cross-arm ``delta_ce_neg`` is clean.

Sign convention: ``delta_ce_*_vs_base = CE_base - CE_trained`` (positive =
training reduced CE on that row type, i.e. gradient was delivered).

Outputs one JSON per invocation (plan §4.3 Phases 4/7)::

    $FU/rowtype_ce/rowtype_ce_<label>.json

Usage (pod, GPU; minutes per model)::

    # Base model, both mixes (scored ONCE; shared negatives + both pos sets):
    uv run python scripts/issue552_rowtype_ce.py --label base \
        --model Qwen/Qwen2.5-7B-Instruct \
        --mix data/issue_552/contrastive_em_mix.jsonl \
              data/issue_552/contrastive_benign_mix.jsonl \
        --out eval_results/issue_552/contrastive-2x2-completion/rowtype_ce

    # One trained cell (merged dir), its own mix, deltas vs the base JSON:
    uv run python scripts/issue552_rowtype_ce.py --label contrastive_em_seed42 \
        --model models/issue404_pair_turner_bad_medical_contrastive_seed42/sft_narrow_merged \
        --mix data/issue_552/contrastive_em_mix.jsonl \
        --base-json .../rowtype_ce/rowtype_ce_base.json \
        --out eval_results/issue_552/contrastive-2x2-completion/rowtype_ce

Content hygiene: the EM mix carries harmful-advice text — this script never
prints row contents; logs carry counts, hashes, and CE values only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import random
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent

SUBSAMPLE_SEED = 0
DEFAULT_SUBSAMPLE_SIZE = 512
DEFAULT_BATCH_SIZE = 8
DEFAULT_MAX_LENGTH = 2048  # the training max_seq_length (turner_em recipe)


def _load_mix(path: Path) -> list[dict]:
    """Load a contrastive mix JSONL; assert the builder's row schema."""
    rows: list[dict] = []
    with path.open() as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            assert item.get("row_type") in ("positive", "negative"), (
                f"{path}:{line_num}: row_type missing/unknown — not a "
                f"issue_552_build_contrastive_mixes.py mix file"
            )
            assert "question_index" in item and "messages" in item, (
                f"{path}:{line_num}: missing question_index/messages keys"
            )
            rows.append(item)
    assert rows, f"empty mix file: {path}"
    return rows


def _subsample_indices(n_questions: int, size: int) -> tuple[list[int], list[int]]:
    """Deterministic (positives, negatives) question-index subsamples (seed 0).

    Two successive draws from ONE ``random.Random(SUBSAMPLE_SEED)`` stream —
    order is part of the registered protocol. ``size`` caps at n_questions.
    """
    rng = random.Random(SUBSAMPLE_SEED)
    k = min(size, n_questions)
    pos_idx = sorted(rng.sample(range(n_questions), k))
    neg_idx = sorted(rng.sample(range(n_questions), k))
    return pos_idx, neg_idx


def compute_per_row_ce(
    model,
    tokenizer,
    texts: list[str],
    *,
    batch_size: int,
    max_length: int,
) -> tuple[list[float], list[int]]:
    """Teacher-forced full-sequence CE per rendered text.

    Returns (per_row_mean_ce, per_row_n_tokens) where per-row mean CE is the
    NLL averaged over that row's scored (non-pad, shifted) token positions.

    Right padding + attention mask: pads sit AFTER content, so default
    position_ids (0..T-1) are correct for every content token — no left-pad
    RoPE drift (the #502 class). Loss positions are masked to content tokens.
    """
    import torch

    device = next(model.parameters()).device
    per_row_ce: list[float] = []
    per_row_tokens: list[int] = []
    for start in range(0, len(texts), batch_size):
        batch = texts[start : start + batch_size]
        enc = tokenizer(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
            add_special_tokens=False,
        )
        input_ids = enc["input_ids"].to(device)
        attn = enc["attention_mask"].to(device)
        b, t = input_ids.shape
        assert attn.shape == (b, t), (attn.shape, (b, t))
        with torch.no_grad():
            logits = model(input_ids=input_ids, attention_mask=attn).logits
        v = logits.shape[-1]
        assert logits.shape == (b, t, v), logits.shape
        # Shifted next-token NLL, masked to real (non-pad) target positions.
        shift_logits = logits[:, :-1, :].float()
        shift_labels = input_ids[:, 1:]
        shift_mask = attn[:, 1:].bool()
        nll = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, v),
            shift_labels.reshape(-1),
            reduction="none",
        ).reshape(b, t - 1)
        nll = nll * shift_mask.float()
        row_tokens = shift_mask.sum(dim=1)
        assert int(row_tokens.min()) > 0, "a row tokenized to <2 tokens — cannot score CE"
        row_ce = (nll.sum(dim=1) / row_tokens.float()).cpu()
        per_row_ce.extend(float(x) for x in row_ce)
        per_row_tokens.extend(int(x) for x in row_tokens.cpu())
        logger.info(
            "[phase=ce_batch] rows %d-%d/%d scored",
            start + 1,
            start + len(batch),
            len(texts),
        )
    return per_row_ce, per_row_tokens


def _render_rows(tokenizer, rows: list[dict]) -> list[str]:
    """Render each mix row's messages through the chat template (training surface)."""
    return [
        tokenizer.apply_chat_template(r["messages"], tokenize=False, add_generation_prompt=False)
        for r in rows
    ]


def _token_weighted_mean(ce: list[float], tokens: list[int]) -> float:
    """Token-weighted mean per-token CE over rows (total NLL / total tokens)."""
    assert len(ce) == len(tokens) and ce, (len(ce), len(tokens))
    total_nll = sum(c * n for c, n in zip(ce, tokens, strict=True))
    total_tok = sum(tokens)
    return total_nll / total_tok


def _select_rows(rows: list[dict], row_type: str, wanted_q_idx: list[int]) -> list[dict]:
    """Pick the subsampled rows of one type, ordered by question_index."""
    by_q = {r["question_index"]: r for r in rows if r["row_type"] == row_type}
    missing = [q for q in wanted_q_idx if q not in by_q]
    assert not missing, (
        f"{len(missing)} subsampled question indices missing from the mix's "
        f"{row_type} rows (first: {missing[:5]})"
    )
    return [by_q[q] for q in wanted_q_idx]


def _repro_metadata(model: str) -> dict:
    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        git_commit = "unknown"
    return {
        "script": "issue552_rowtype_ce",
        "model": model,
        "git_commit": git_commit,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="#552 per-row-type CE diagnostic (plan v5 MF-A)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Merged checkpoint dir OR hub id (base invocation).",
    )
    parser.add_argument(
        "--mix",
        nargs="+",
        required=True,
        help=(
            "Mix JSONL path(s). Trained cells pass their ONE mix; the base "
            "invocation passes BOTH mixes (scored once: shared negative "
            "subsample + each arm's positive subsample)."
        ),
    )
    parser.add_argument("--label", required=True, help="e.g. contrastive_em_seed42 | base")
    parser.add_argument(
        "--out",
        required=True,
        help="Output DIR; writes rowtype_ce_<label>.json inside it.",
    )
    parser.add_argument(
        "--base-json",
        default=None,
        help="Path to rowtype_ce_base.json; when set, deltas vs base are recorded.",
    )
    parser.add_argument("--subsample-size", type=int, default=DEFAULT_SUBSAMPLE_SIZE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MAX_LENGTH)
    parser.add_argument(
        "--device",
        default="cuda",
        choices=["cuda", "cpu"],
        help="cpu is smoke-only (fp32, tiny model).",
    )
    parser.add_argument(
        "--tokenizer",
        default=None,
        help="Override tokenizer path (smoke: tiny model dirs may lack one).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    # `uv run python` does NOT auto-load .env; hub loads need HF_TOKEN.
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    mixes = {Path(p).name.removesuffix(".jsonl"): _load_mix(Path(p)) for p in args.mix}
    # All mixes must agree on the question-index universe (shared negatives).
    n_questions_per_mix = {
        name: len({r["question_index"] for r in rows if r["row_type"] == "positive"})
        for name, rows in mixes.items()
    }
    n_q_set = set(n_questions_per_mix.values())
    assert len(n_q_set) == 1, f"mixes disagree on question count: {n_questions_per_mix}"
    n_questions = n_q_set.pop()

    pos_idx, neg_idx = _subsample_indices(n_questions, args.subsample_size)
    subsample_digest = hashlib.sha256(
        json.dumps({"pos": pos_idx, "neg": neg_idx}).encode()
    ).hexdigest()
    logger.info(
        "[phase=subsample] n_questions=%d size=%d seed=%d sha256=%s",
        n_questions,
        len(pos_idx),
        SUBSAMPLE_SEED,
        subsample_digest[:16],
    )

    logger.info("[phase=load_model] %s (device=%s)", args.model, args.device)
    dtype = torch.bfloat16 if args.device == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map=args.device if args.device == "cuda" else None,
        trust_remote_code=True,
    )
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer or args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"  # pads AFTER content; default position_ids correct

    result: dict = {
        "label": args.label,
        "subsample_seed": SUBSAMPLE_SEED,
        "subsample_size": len(pos_idx),
        "subsample_sha256": subsample_digest,
        "max_length": args.max_length,
        "batch_size": args.batch_size,
        "loss_surface": "teacher-forced full-sequence CE over the rendered chat template",
        "delta_sign_convention": "delta_ce_*_vs_base = CE_base - CE_trained (positive = moved)",
        "per_mix": {},
    }

    # Negatives are identical across mixes by construction — score them ONCE
    # (from the first mix), assert the other mixes' selections hash the same.
    neg_hashes = {}
    neg_rows_by_mix = {}
    for name, rows in mixes.items():
        sel = _select_rows(rows, "negative", neg_idx)
        neg_rows_by_mix[name] = sel
        neg_hashes[name] = hashlib.sha256(
            json.dumps([r["messages"] for r in sel], sort_keys=True).encode()
        ).hexdigest()
    assert len(set(neg_hashes.values())) == 1, (
        f"negative subsample differs across mixes (identical-negatives invariant "
        f"violated): {neg_hashes}"
    )
    first_mix = next(iter(mixes))
    neg_texts = _render_rows(tokenizer, neg_rows_by_mix[first_mix])
    logger.info("[phase=ce_negatives] scoring %d shared negative rows", len(neg_texts))
    neg_ce, neg_tok = compute_per_row_ce(
        model, tokenizer, neg_texts, batch_size=args.batch_size, max_length=args.max_length
    )
    mean_ce_neg = _token_weighted_mean(neg_ce, neg_tok)
    result["mean_ce_neg"] = mean_ce_neg
    result["n_neg_rows"] = len(neg_ce)
    result["per_row_mean_ce_neg_unweighted"] = sum(neg_ce) / len(neg_ce)

    for name, rows in mixes.items():
        sel = _select_rows(rows, "positive", pos_idx)
        pos_texts = _render_rows(tokenizer, sel)
        logger.info("[phase=ce_positives] mix=%s scoring %d positive rows", name, len(pos_texts))
        pos_ce, pos_tok = compute_per_row_ce(
            model, tokenizer, pos_texts, batch_size=args.batch_size, max_length=args.max_length
        )
        result["per_mix"][name] = {
            "mean_ce_pos": _token_weighted_mean(pos_ce, pos_tok),
            "n_pos_rows": len(pos_ce),
            "per_row_mean_ce_pos_unweighted": sum(pos_ce) / len(pos_ce),
        }

    # Single-mix (trained-cell) convenience flattening: the sentinel + analyzer
    # read mean_ce_pos at top level for trained cells.
    if len(mixes) == 1:
        only = result["per_mix"][first_mix]
        result["mean_ce_pos"] = only["mean_ce_pos"]
        result["n_pos_rows"] = only["n_pos_rows"]

    if args.base_json:
        base = json.loads(Path(args.base_json).read_text())
        assert base["subsample_sha256"] == subsample_digest, (
            "base JSON was computed on a DIFFERENT subsample — deltas would be invalid. "
            f"base={base['subsample_sha256'][:16]} this={subsample_digest[:16]}"
        )
        result["delta_ce_neg_vs_base"] = float(base["mean_ce_neg"]) - mean_ce_neg
        if "mean_ce_pos" in result:
            base_pos = base["per_mix"][first_mix]["mean_ce_pos"]
            result["delta_ce_pos_vs_base"] = float(base_pos) - result["mean_ce_pos"]
        result["base_json"] = str(args.base_json)

    result["metadata"] = _repro_metadata(args.model)
    out_path = out_dir / f"rowtype_ce_{args.label}.json"
    with out_path.open("w") as f:
        json.dump(result, f, indent=2)
    logger.info(
        "[phase=done] %s: mean_ce_neg=%.4f%s -> %s",
        args.label,
        mean_ce_neg,
        (
            f" delta_ce_neg_vs_base={result['delta_ce_neg_vs_base']:+.4f}"
            if "delta_ce_neg_vs_base" in result
            else ""
        ),
        out_path,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
