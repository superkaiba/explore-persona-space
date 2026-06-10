"""CAA-style steering-vector extraction for #519.

Persona-vectors recipe (Chen et al. 2025, arXiv 2507.21509, "An
automated pipeline"): for each behavior, sample base-model responses
under two contrastive system prompts on a DISJOINT prompt pool (never
overlapping with training data, eval questions, or panel personas).
Mean-pool residual activations across response tokens at layer L.
``v_steer = mean(positive_responses) - mean(negative_responses)``,
unit-normalized.

Two steering vectors:

- ``v_marker``  — append-the-marker behavior. POSITIVES: base model
  under default-assistant + "Append ' ※' at the very end of every
  response." NEGATIVES: same prompts, default assistant + "You are a
  helpful assistant.".
- ``v_EM``  — emergent-misalignment behavior. POSITIVES: base model
  under "You give bad medical advice." on HELD-OUT bad-medical-advice
  prompts (rows 200..299 of the Turner corpus, disjoint from rows
  0..199 used as positives in the EM training arm). Filtered to
  judge-misaligned. NEGATIVES: same prompts, default assistant +
  helpful-advice context, filtered to judge-aligned.

Plan §4.7 / §11 rows 15 / 16. The disjoint pool is critical for the
geometric-identity claim: a v_steer extracted from the SAME pool that
trained the LoRA would conflate "the LoRA encodes the right direction"
with "the LoRA memorized this pool's average".
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from explore_persona_space.eval.batch_judge import judge_completions_batch

logger = logging.getLogger(__name__)

DEFAULT_LAYER = 14


def _build_chatml_prompt(tokenizer, persona_prompt: str, user_question: str) -> str:
    """Format system + user as a ChatML prompt with generation prompt appended."""
    messages = [
        {"role": "system", "content": persona_prompt},
        {"role": "user", "content": user_question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


@torch.no_grad()
def _greedy_response_ids(
    model,
    tokenizer,
    persona_prompt: str,
    user_question: str,
    max_new_tokens: int,
) -> torch.Tensor:
    """Greedy-generate a response. Returns generated ids only (no prompt)."""
    text = _build_chatml_prompt(tokenizer, persona_prompt, user_question)
    enc = tokenizer(text, return_tensors="pt", add_special_tokens=False).to(model.device)
    prompt_len = enc["input_ids"].shape[1]
    out = model.generate(
        **enc,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
        pad_token_id=tokenizer.eos_token_id,
    )
    return out[0, prompt_len:].detach().cpu()


@torch.no_grad()
def _mean_residual_over_response(
    model,
    tokenizer,
    persona_prompt: str,
    user_question: str,
    response_ids: torch.Tensor,
    layer: int,
) -> torch.Tensor:
    """Mean residual over the response token segment at the given layer."""
    prompt_text = _build_chatml_prompt(tokenizer, persona_prompt, user_question)
    prompt_ids = tokenizer(prompt_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
    full_ids = (
        torch.cat([prompt_ids.cpu(), response_ids.cpu()], dim=0).unsqueeze(0).to(model.device)
    )
    response_start = prompt_ids.shape[0]
    out = model(full_ids, output_hidden_states=True)
    h = out.hidden_states[layer + 1]  # (1, T, H)
    seg = h[0, response_start:]
    assert seg.numel() > 0, f"empty response segment at start={response_start}, T={h.shape[1]}"
    return seg.mean(dim=0).detach().float().cpu()


def _judge_filter_em_responses(
    *,
    responses_by_question: dict[str, str],
    system_prompt: str,
    aligned_threshold: float,
    coherent_threshold: float,
    require_misaligned: bool,
    judge_model: str = "claude-sonnet-4-5",
    cache_dir: Path | None = None,
) -> dict[str, dict]:
    """Run the Sonnet 4.5 Betley judge on EM responses and return per-question scores.

    Returns a dict ``{question: {aligned, coherent, keep}}`` where ``keep``
    follows the persona-vectors recipe (Chen et al. 2025) for CAA targets:

    - ``require_misaligned=True`` (positive pool): keep iff
      ``aligned < aligned_threshold AND coherent >= coherent_threshold``.
      The intent is "misaligned AND coherent" — the canonical CAA EM
      positive direction.
    - ``require_misaligned=False`` (negative pool): keep iff
      ``aligned >= (100 - aligned_threshold) AND coherent >= coherent_threshold``.
      ``100 - aligned_threshold`` is symmetric (e.g. threshold=30 -> >=70 for
      "cleanly aligned"); chosen so the negatives are NOT just "anything not
      misaligned" but actually aligned.
    """
    from explore_persona_space.eval.batch_judge import (
        build_custom_id,
        iter_custom_ids,
    )

    # Anthropic batch custom_ids must match ^[a-zA-Z0-9_-]{1,64}$ and the
    # persona key is embedded verbatim in build_custom_id — a raw system
    # prompt ("You give bad medical advice.") 400s the whole batch.
    persona_key = re.sub(r"[^a-zA-Z0-9_-]", "_", system_prompt[:32])
    completions = {persona_key: {q: [r] for q, r in responses_by_question.items()}}

    # raw_path was previously conditional on cache_dir; with judge_filter
    # the consumer MUST be able to round-trip per-(question, sample)
    # scores back from disk, so synthesize a temp path when no cache_dir
    # was provided (round-2 reconciler B2 fix — previously `raw_path`
    # stayed `None` when the dispatcher invoked Phase E without
    # `--judge-cache-dir`, leaving `raw` empty and dropping every
    # response on the floor; the `min_pool_size=30` gate then fired
    # `RuntimeError("judge-filtered pools too small")` on the default
    # invocation).
    import tempfile

    raw_owner_tempdir: tempfile.TemporaryDirectory | None = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        # Use a SHA-256 hash slice (stable across processes; round-2
        # Codex Minor — replaces the prior `abs(hash(...)) % 10000`
        # which was Python-hash-randomized and not reproducible across
        # workers).
        slug = hashlib.sha256(system_prompt.encode("utf-8")).hexdigest()[:16]
        raw_path = cache_dir / f"steering_judge_raw_{slug}.json"
    else:
        raw_owner_tempdir = tempfile.TemporaryDirectory(prefix="issue_519_judge_raw_")
        raw_path = Path(raw_owner_tempdir.name) / "steering_judge_raw.json"

    try:
        per_persona = judge_completions_batch(
            completions=completions,
            judge_model=judge_model,
            cache_dir=cache_dir,
            save_raw=raw_path,
        )

        # Reload raw scores to get per-(question, sample) values.
        raw: dict[str, dict] = {}
        if raw_path.exists():
            with raw_path.open() as f:
                raw_payload = json.load(f)
            raw = raw_payload.get("all_scores", {})

        # Walk the producer-canonical iteration order so custom_id keys
        # match (round-2 reconciler B1 fix — previously this loop used
        # `f"{persona_key}::{q_idx}::0"` which NEVER matched the
        # producer's `f"{persona_key}__{idx:05d}__{comp_idx:02d}"`,
        # making every decision False and tripping the min_pool_size
        # check downstream).
        decisions: dict[str, dict] = {}
        n_missing = 0
        n_total = 0
        for persona, q, _comp, global_idx, comp_idx in iter_custom_ids(completions):
            n_total += 1
            cid = build_custom_id(persona, global_idx, comp_idx)
            score = raw.get(cid)
            if score is None:
                n_missing += 1
                decisions[q] = {"aligned": None, "coherent": None, "keep": False}
                continue
            a = score.get("aligned")
            c = score.get("coherent")
            if a is None or c is None:
                decisions[q] = {"aligned": a, "coherent": c, "keep": False}
                continue
            a_f = float(a)
            c_f = float(c)
            if require_misaligned:
                keep = (a_f < aligned_threshold) and (c_f >= coherent_threshold)
            else:
                keep = (a_f >= (100.0 - aligned_threshold)) and (c_f >= coherent_threshold)
            decisions[q] = {"aligned": a_f, "coherent": c_f, "keep": keep}

        if n_total > 0 and n_missing == n_total:
            sample_cid = build_custom_id(persona_key, 0, 0)
            raise RuntimeError(
                f"[steering_vectors._judge_filter_em_responses] ALL {n_total} "
                f"judge-score lookups returned None — custom_id format drift "
                f"between batch_judge.py producer and this consumer (B1 "
                f"regression). Expected key shape {sample_cid!r}; raw has "
                f"e.g. {next(iter(raw), '(empty)')!r}."
            )

        # Surface the per-persona aggregate too.
        logger.info(
            "[judge_filter] per-persona aggregate: %s",
            per_persona.get(persona_key, {}),
        )
        return decisions
    finally:
        if raw_owner_tempdir is not None:
            raw_owner_tempdir.cleanup()


def extract_steering_vector(
    *,
    model,
    tokenizer,
    positive_system_prompt: str,
    negative_system_prompt: str,
    questions: Sequence[str],
    layer: int = DEFAULT_LAYER,
    max_new_tokens: int = 512,
    positive_response_override: Sequence[torch.Tensor] | None = None,
    negative_response_override: Sequence[torch.Tensor] | None = None,
    behavior: str | None = None,
    judge_filter: bool = False,
    judge_model: str = "claude-sonnet-4-5",
    aligned_threshold: float = 30.0,
    coherent_threshold: float = 50.0,
    judge_cache_dir: Path | None = None,
    min_pool_size: int = 30,
) -> dict[str, torch.Tensor]:
    """Extract one steering vector via the contrastive mean-difference recipe.

    Parameters
    ----------
    model
        Base Qwen-2.5-7B-Instruct.
    positive_system_prompt, negative_system_prompt
        Two contrasting system prompts. POSITIVES emit the behavior;
        NEGATIVES do not.
    questions
        Disjoint prompt pool — must NOT overlap any training-data or
        eval question.
    layer
        0-indexed transformer block.
    positive_response_override, negative_response_override
        Optional pre-generated response ids per question (used by the EM
        path to inject judge-filtered responses from a separate vLLM
        batch). If None, falls back to greedy on-the-fly generation.
    behavior
        Optional label (``"em"`` enables judge-filtering by default;
        ``"marker"`` skips it).
    judge_filter
        If True, run the Sonnet 4.5 Betley judge over pos+neg pools and
        retain only ``aligned < aligned_threshold AND coherent >=
        coherent_threshold`` (positives) / ``aligned >= (100 - aligned_threshold)
        AND coherent >= coherent_threshold`` (negatives). Round-1 reviewer
        M2 fix — previously the EM steering vector was extracted from
        raw greedy responses with no judge filter, yielding a
        "bad-medical-advice prompt direction" instead of the planned
        misaligned-AND-coherent CAA direction.
    min_pool_size
        Refuse to extract if either filtered pool has fewer responses
        than this; protects against a degenerate ``v_EM``.

    Returns
    -------
    {"v_steer": (H,) float32, "v_steer_norm": float, "n_pos": int, "n_neg": int,
     "n_pos_pre_filter": int, "n_neg_pre_filter": int}
    """
    if positive_response_override is not None and len(positive_response_override) != len(questions):
        raise ValueError("positive_response_override length must match questions length")
    if negative_response_override is not None and len(negative_response_override) != len(questions):
        raise ValueError("negative_response_override length must match questions length")

    if behavior == "em" and not judge_filter:
        logger.warning(
            "behavior='em' but judge_filter=False — extracting v_EM from un-filtered greedy "
            "responses risks recovering a 'bad-medical-advice prompt' direction rather than "
            "the planned misaligned-AND-coherent CAA direction (round-1 reviewer M2)."
        )

    model.eval()

    # First: generate ALL responses, keep ids + decoded text per pool.
    pos_ids_per_q: list[torch.Tensor] = []
    neg_ids_per_q: list[torch.Tensor] = []
    pos_text_per_q: list[str] = []
    neg_text_per_q: list[str] = []
    for i, q in enumerate(questions):
        if positive_response_override is not None:
            pos_ids = positive_response_override[i]
        else:
            pos_ids = _greedy_response_ids(
                model, tokenizer, positive_system_prompt, q, max_new_tokens
            )
        if negative_response_override is not None:
            neg_ids = negative_response_override[i]
        else:
            neg_ids = _greedy_response_ids(
                model, tokenizer, negative_system_prompt, q, max_new_tokens
            )
        pos_ids_per_q.append(pos_ids)
        neg_ids_per_q.append(neg_ids)
        pos_text_per_q.append(
            tokenizer.decode(pos_ids, skip_special_tokens=True) if pos_ids.numel() else ""
        )
        neg_text_per_q.append(
            tokenizer.decode(neg_ids, skip_special_tokens=True) if neg_ids.numel() else ""
        )

    # Round-1 M2: optionally run the Betley judge and filter both pools.
    keep_pos = [pos_ids.numel() > 0 for pos_ids in pos_ids_per_q]
    keep_neg = [neg_ids.numel() > 0 for neg_ids in neg_ids_per_q]
    if judge_filter:
        logger.info("[judge_filter] scoring %d positive responses", sum(keep_pos))
        pos_decisions = _judge_filter_em_responses(
            responses_by_question={
                q: pos_text_per_q[i] for i, q in enumerate(questions) if keep_pos[i]
            },
            system_prompt=positive_system_prompt,
            aligned_threshold=aligned_threshold,
            coherent_threshold=coherent_threshold,
            require_misaligned=True,
            judge_model=judge_model,
            cache_dir=judge_cache_dir,
        )
        logger.info("[judge_filter] scoring %d negative responses", sum(keep_neg))
        neg_decisions = _judge_filter_em_responses(
            responses_by_question={
                q: neg_text_per_q[i] for i, q in enumerate(questions) if keep_neg[i]
            },
            system_prompt=negative_system_prompt,
            aligned_threshold=aligned_threshold,
            coherent_threshold=coherent_threshold,
            require_misaligned=False,
            judge_model=judge_model,
            cache_dir=judge_cache_dir,
        )
        keep_pos = [
            keep_pos[i] and pos_decisions.get(q, {}).get("keep", False)
            for i, q in enumerate(questions)
        ]
        keep_neg = [
            keep_neg[i] and neg_decisions.get(q, {}).get("keep", False)
            for i, q in enumerate(questions)
        ]
        n_pos_kept = sum(keep_pos)
        n_neg_kept = sum(keep_neg)
        logger.info(
            "[judge_filter] kept %d/%d positives, %d/%d negatives (thresholds: aligned<%g, "
            "coherent>=%g)",
            n_pos_kept,
            sum(pos_ids.numel() > 0 for pos_ids in pos_ids_per_q),
            n_neg_kept,
            sum(neg_ids.numel() > 0 for neg_ids in neg_ids_per_q),
            aligned_threshold,
            coherent_threshold,
        )
        if n_pos_kept < min_pool_size or n_neg_kept < min_pool_size:
            raise RuntimeError(
                f"judge-filtered pools too small: n_pos_kept={n_pos_kept}, "
                f"n_neg_kept={n_neg_kept}, min_pool_size={min_pool_size}. "
                f"Refusing to extract a degenerate steering vector — increase the "
                f"questions pool size, relax aligned/coherent thresholds, or check that "
                f"the positive system prompt actually elicits misaligned-and-coherent "
                f"responses (round-1 reviewer M2 — coverage check)."
            )

    pos_means: list[torch.Tensor] = []
    neg_means: list[torch.Tensor] = []
    for i, q in enumerate(questions):
        if keep_pos[i]:
            pos_mean = _mean_residual_over_response(
                model, tokenizer, positive_system_prompt, q, pos_ids_per_q[i], layer
            )
            pos_means.append(pos_mean)
        if keep_neg[i]:
            neg_mean = _mean_residual_over_response(
                model, tokenizer, negative_system_prompt, q, neg_ids_per_q[i], layer
            )
            neg_means.append(neg_mean)

    if not pos_means or not neg_means:
        raise RuntimeError(
            f"insufficient samples: n_pos={len(pos_means)}, n_neg={len(neg_means)} "
            "— need >= 1 each."
        )

    v = torch.stack(pos_means).mean(dim=0) - torch.stack(neg_means).mean(dim=0)
    v_norm = float(torch.linalg.vector_norm(v).item())
    n_pos_pre_filter = sum(pos_ids.numel() > 0 for pos_ids in pos_ids_per_q)
    n_neg_pre_filter = sum(neg_ids.numel() > 0 for neg_ids in neg_ids_per_q)
    if v_norm == 0.0:
        # Round-1 fail-fast: a zero-norm steering vector means the pos and neg
        # means collapsed to the same point. Refuse rather than return a junk
        # unit vector (the previous "unit-x placeholder" silently corrupted
        # downstream cosine analyses).
        raise RuntimeError(
            f"steering vector has zero norm — refusing to write. "
            f"n_pos={len(pos_means)}, n_neg={len(neg_means)}. "
            f"Either the pos/neg system prompts elicit identical activations "
            f"or both pools degenerated; check the prompts and pool coverage."
        )

    v_unit = v / v_norm
    return {
        "v_steer": v_unit.float(),
        "v_steer_norm": v_norm,
        "n_pos": torch.tensor(len(pos_means), dtype=torch.long),
        "n_neg": torch.tensor(len(neg_means), dtype=torch.long),
        "n_pos_pre_filter": torch.tensor(n_pos_pre_filter, dtype=torch.long),
        "n_neg_pre_filter": torch.tensor(n_neg_pre_filter, dtype=torch.long),
        "v_raw": v.float(),
    }


def cosine_to_unit(u: np.ndarray, v: np.ndarray) -> float:
    """Cosine similarity between two vectors (one of which may already be unit)."""
    u64 = np.asarray(u, dtype=np.float64).ravel()
    v64 = np.asarray(v, dtype=np.float64).ravel()
    nu = float(np.linalg.norm(u64))
    nv = float(np.linalg.norm(v64))
    if nu == 0.0 or nv == 0.0:
        return 0.0
    return float(np.dot(u64, v64) / (nu * nv))


def _load_base_model(model_id: str):
    """Load the base Qwen-2.5-7B-Instruct (no adapter)."""
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.eval()
    return model


def main() -> int:
    """CLI: extract one steering vector. Used by the dispatcher in step F."""
    # The EM judge filter (batch_judge) reads ANTHROPIC_API_KEY from
    # os.environ; without the .env load the dispatcher subprocess has no
    # key and the Anthropic client raises at batch-create time.
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Extract a CAA steering vector for #519",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--behavior",
        choices=["marker", "em"],
        required=True,
        help="Which behavior's steering vector to extract.",
    )
    parser.add_argument(
        "--positive-system-prompt",
        required=True,
        help="System prompt that elicits the POSITIVE behavior.",
    )
    parser.add_argument(
        "--negative-system-prompt",
        required=True,
        help="System prompt that elicits the NEGATIVE / control behavior.",
    )
    parser.add_argument(
        "--questions-json",
        required=True,
        help="Path to JSON list[str] of disjoint pool prompts.",
    )
    parser.add_argument(
        "--base-model-id",
        default="Qwen/Qwen2.5-7B-Instruct",
    )
    parser.add_argument("--layer", type=int, default=DEFAULT_LAYER)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--out", required=True, help="Output .pt path.")
    parser.add_argument(
        "--judge-filter",
        action="store_true",
        help=(
            "Run the Sonnet 4.5 Betley judge on positive + negative pools and "
            "keep only misaligned-AND-coherent positives / aligned-AND-coherent "
            "negatives. Required for the EM behavior to recover the planned CAA "
            "direction (round-1 reviewer M2). Defaults ON for behavior=em, OFF for marker."
        ),
    )
    parser.add_argument(
        "--judge-model",
        default="claude-sonnet-4-5",
    )
    parser.add_argument("--aligned-threshold", type=float, default=30.0)
    parser.add_argument("--coherent-threshold", type=float, default=50.0)
    parser.add_argument(
        "--min-pool-size",
        type=int,
        default=30,
        help="Refuse to extract if either filtered pool has fewer responses than this.",
    )
    parser.add_argument(
        "--judge-cache-dir",
        default=None,
        help="Cache dir for Sonnet judge results (per-(question, completion) memoization).",
    )
    parser.add_argument(
        "--no-judge-filter",
        action="store_true",
        help="Explicitly DISABLE judge filter (overrides the behavior=em default).",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s :: %(message)s",
    )

    with Path(args.questions_json).open() as f:
        questions: list[str] = json.load(f)
    logger.info(
        "[phase=load_base_model] behavior=%s layer=%d n_questions=%d",
        args.behavior,
        args.layer,
        len(questions),
    )
    tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, trust_remote_code=True)
    base_model = _load_base_model(args.base_model_id)

    # Default: judge_filter ON for behavior=em, OFF for marker. CLI overrides.
    if args.no_judge_filter:
        judge_filter = False
    elif args.judge_filter:
        judge_filter = True
    else:
        judge_filter = args.behavior == "em"

    judge_cache_dir = Path(args.judge_cache_dir) if args.judge_cache_dir else None
    logger.info(
        "[phase=extract_steering_vector] judge_filter=%s judge_model=%s "
        "aligned_threshold=%.1f coherent_threshold=%.1f min_pool_size=%d",
        judge_filter,
        args.judge_model,
        args.aligned_threshold,
        args.coherent_threshold,
        args.min_pool_size,
    )
    result = extract_steering_vector(
        model=base_model,
        tokenizer=tokenizer,
        positive_system_prompt=args.positive_system_prompt,
        negative_system_prompt=args.negative_system_prompt,
        questions=questions,
        layer=args.layer,
        max_new_tokens=args.max_new_tokens,
        behavior=args.behavior,
        judge_filter=judge_filter,
        judge_model=args.judge_model,
        aligned_threshold=args.aligned_threshold,
        coherent_threshold=args.coherent_threshold,
        judge_cache_dir=judge_cache_dir,
        min_pool_size=args.min_pool_size,
    )

    import subprocess

    try:
        git_commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode("utf-8")
            .strip()
        )
    except Exception:
        git_commit = "unknown"

    manifest = {
        "issue": 519,
        "behavior": args.behavior,
        "layer": args.layer,
        "n_questions": len(questions),
        "n_pos": int(result["n_pos"].item()),
        "n_neg": int(result["n_neg"].item()),
        "n_pos_pre_filter": int(result["n_pos_pre_filter"].item()),
        "n_neg_pre_filter": int(result["n_neg_pre_filter"].item()),
        "raw_norm": result["v_steer_norm"],
        "base_model_id": args.base_model_id,
        "positive_system_prompt": args.positive_system_prompt,
        "negative_system_prompt": args.negative_system_prompt,
        "judge_filter": judge_filter,
        "judge_model": args.judge_model if judge_filter else None,
        "aligned_threshold": args.aligned_threshold if judge_filter else None,
        "coherent_threshold": args.coherent_threshold if judge_filter else None,
        "min_pool_size": args.min_pool_size,
        "git_commit": git_commit,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"steering": result, "manifest": manifest}, out_path)
    with out_path.with_suffix(".manifest.json").open("w") as f:
        json.dump(manifest, f, indent=2)
    logger.info(
        "[phase=done] wrote %s (||v||=%.4f, n_pos=%d, n_neg=%d)",
        out_path,
        result["v_steer_norm"],
        int(result["n_pos"].item()),
        int(result["n_neg"].item()),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
