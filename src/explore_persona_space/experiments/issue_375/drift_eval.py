"""vLLM batched generation + marker scoring per cell for issue #375.

Plan §4.8 — per cell, we:

1. Build 200 prompts (one per held-out query). For each prompt we sample
   ``k`` examples deterministically from the appropriate pool (or none for
   k=0 zero-shot). The query's ``"source"`` tag (``eval_questions`` /
   ``lmsys_tail``) is carried through ``prompt_meta`` for stratified reporting.

2. ZLT eval-time contamination gate (plan §4.4 step 5b stage b): if any
   rendered prompt context contains ``[zlt]`` case-insensitively BEFORE the
   to-be-generated final assistant turn, raise ``RuntimeError``. Save every
   rendered prompt to ``rendered_prompts.jsonl`` for audit (stage c).

3. vLLM batched generation: ``LLM.generate(prompts, SamplingParams(n=10,
   temperature=1.0, top_p=0.95, max_tokens=2048, seed=42))``.

4. Score: case-insensitive substring search for ``[ZLT]`` on every
   completion. Per-query and aggregate marker rates, stratified by query
   source.

5. Save ``raw_completions.json`` (per-query x per-completion strings),
   ``marker_eval.json`` (rates), ``summary.json`` (top-level numbers).

Module is import-safe even when vLLM is not installed (the vllm import
lives inside :func:`run_cell`).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from explore_persona_space.experiments.issue_375.example_pool import (
    ZLT_LOWER,
    Example,
)
from explore_persona_space.experiments.issue_375.fewshot_prompt import (
    build_messages,
    sample_examples,
    truncate_examples,
)

log = logging.getLogger(__name__)

MARKER_TOKEN = "[ZLT]"


@dataclass
class HeldOutQuery:
    """One held-out query with a stable id and source tag."""

    id: int  # 0..199 — stable index into held_out_queries.jsonl
    text: str
    source: str  # "eval_questions" or "lmsys_tail"


# ────────────────────────────────────────────────────────────────────────────
# Marker scoring
# ────────────────────────────────────────────────────────────────────────────


def score_completions(
    completions: Sequence[Sequence[str]],
    marker: str = MARKER_TOKEN,
) -> dict:
    """Case-insensitive substring marker scorer (matches the rule in
    ``scripts/archive/run_leakage_experiment.py::evaluate_markers``).

    Args:
        completions: ``[n_query][n_completions_per_query]`` string array.
        marker: marker string to search for; case-insensitive.

    Returns:
        ``{
            "rate": overall_rate,
            "found": found_total,
            "total": count_total,
            "per_query": [{"rate", "found", "total"}],
        }``.
    """
    marker_lower = marker.lower()
    per_query: list[dict] = []
    found_total = 0
    count_total = 0
    for comps in completions:
        found = sum(1 for c in comps if marker_lower in c.lower())
        per_query.append(
            {"rate": found / len(comps) if comps else 0.0, "found": found, "total": len(comps)}
        )
        found_total += found
        count_total += len(comps)
    return {
        "rate": found_total / count_total if count_total else 0.0,
        "found": found_total,
        "total": count_total,
        "per_query": per_query,
    }


def per_query_fire_fraction(
    completions: Sequence[Sequence[str]],
    marker: str = MARKER_TOKEN,
) -> np.ndarray:
    """Per-query fraction of completions that fired the marker. Used as the
    paired-bootstrap unit (one ``rate`` per query x per cell).
    """
    marker_lower = marker.lower()
    out = np.zeros(len(completions), dtype=np.float64)
    for i, comps in enumerate(completions):
        if not comps:
            continue
        out[i] = sum(1 for c in comps if marker_lower in c.lower()) / len(comps)
    return out


def stratify_rates(
    completions: Sequence[Sequence[str]],
    prompt_meta: Sequence[dict],
    marker: str = MARKER_TOKEN,
) -> dict:
    """Compute overall, eval_questions-only, lmsys_tail-only rates.

    Used to produce ``stratified_by_query_source.json`` per plan §4.5 +
    §6 stratified-reporting requirement.
    """
    rates = per_query_fire_fraction(completions, marker)
    eval_mask = np.array(
        [m.get("query_source") == "eval_questions" for m in prompt_meta], dtype=bool
    )
    lmsys_mask = np.array([m.get("query_source") == "lmsys_tail" for m in prompt_meta], dtype=bool)

    def _bucket(mask):
        if mask.sum() == 0:
            return {"rate": 0.0, "n_queries": 0, "n_completions": 0}
        # Recompute on the filtered completions for accurate found/total
        filt = [completions[i] for i in np.where(mask)[0]]
        scored = score_completions(filt, marker)
        return {
            "rate": scored["rate"],
            "n_queries": int(mask.sum()),
            "n_completions": scored["total"],
            "found": scored["found"],
        }

    return {
        "overall": _bucket(np.ones_like(eval_mask)),
        "eval_questions": _bucket(eval_mask),
        "lmsys_tail": _bucket(lmsys_mask),
        "per_query_rates": rates.tolist(),
    }


# ────────────────────────────────────────────────────────────────────────────
# Cell execution
# ────────────────────────────────────────────────────────────────────────────


def _zlt_eval_time_scan(
    rendered_prompts: Sequence[str],
    cell_label: str,
    prompt_meta: Sequence[dict],
) -> None:
    """Plan §4.4 step 5b stage (b): fail loudly if ``[zlt]`` appears in any
    rendered prompt context BEFORE the to-be-generated final assistant turn.

    The rendered prompt already ends with the assistant role tag (because
    ``apply_chat_template(..., add_generation_prompt=True)``), so any
    contamination is inside the system + user/assistant few-shot turns +
    held-out user turn — exactly what we want to catch.
    """
    for i, rendered in enumerate(rendered_prompts):
        if ZLT_LOWER in rendered.lower():
            qid = prompt_meta[i].get("query_id", i) if i < len(prompt_meta) else i
            raise RuntimeError(
                f"CONTAMINATION: '{MARKER_TOKEN}' appears in rendered prompt context "
                f"for cell={cell_label!r}, prompt_index={i}, query_id={qid}. "
                f"Aborting before eval to prevent silent confound. Inspect the pool."
            )


def _save_rendered_prompts(
    cell_dir: Path,
    rendered_prompts: Sequence[str],
    prompt_meta: Sequence[dict],
) -> None:
    """Audit-trail dump (plan §4.4 step 5b stage c)."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    out = cell_dir / "rendered_prompts.jsonl"
    with open(out, "w") as f:
        for i, rendered in enumerate(rendered_prompts):
            meta = dict(prompt_meta[i]) if i < len(prompt_meta) else {}
            meta["prompt_index"] = i
            meta["rendered"] = rendered
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
    log.info("saved %d rendered prompts to %s", len(rendered_prompts), out)


def build_cell_prompts(
    tokenizer,
    queries: Sequence[HeldOutQuery],
    pool: Sequence[Example],
    k: int,
    *,
    adapter_id: str,
    pool_kind: str,
    system_prompt: str = "You are a helpful assistant.",
    max_user_chars: int = 1500,
    max_assistant_chars: int = 1500,
) -> tuple[list[str], list[dict]]:
    """Build the (rendered_prompts, prompt_meta) tuple for one cell.

    For k=0 zero-shot cells, ``pool`` is ignored.

    Returns:
        rendered_prompts: list of strings (after ``apply_chat_template``).
        prompt_meta: per-prompt ``{query_id, query_source, example_doc_ids}``.
    """
    rendered_prompts: list[str] = []
    prompt_meta: list[dict] = []
    for q in queries:
        if k == 0:
            examples: list[Example] = []
        else:
            sampled = sample_examples(
                pool=pool, k=k, adapter_id=adapter_id, pool_kind=pool_kind, query_id=q.id
            )
            examples = truncate_examples(sampled, max_user_chars, max_assistant_chars)
        msgs = build_messages(examples, q.text, system=system_prompt)
        rendered = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        rendered_prompts.append(rendered)
        prompt_meta.append(
            {
                "query_id": q.id,
                "query_source": q.source,
                "k": k,
                "example_doc_ids": [ex.doc_id for ex in examples],
            }
        )
    return rendered_prompts, prompt_meta


def run_cell(
    llm,  # vllm.LLM (already initialized for the merged adapter or base)
    tokenizer,
    cell_label: str,
    queries: Sequence[HeldOutQuery],
    pool: Sequence[Example],
    k: int,
    *,
    adapter_id: str,
    pool_kind: str,
    decoder: dict,
    eval_results_root: str | Path,
    system_prompt: str = "You are a helpful assistant.",
    max_user_chars: int = 1500,
    max_assistant_chars: int = 1500,
) -> dict:
    """Run one cell end-to-end: build prompts, gate, generate, score, save.

    Args:
        llm: vLLM ``LLM`` instance loaded on the merged adapter (or base
            model for B1/B2/B3 cells).
        tokenizer: the same tokenizer used for chat-template rendering.
        cell_label: e.g. ``villain_C1_persona-style_k3_seed42`` or
            ``base_no-adapter_persona-style-villain_k3_seed42``.
        queries: 200 held-out queries with id + source tag.
        pool: example pool (ignored when k=0).
        k: 0 / 1 / 3.
        adapter_id: e.g. ``villain_C1`` — used to seed example sampling.
        pool_kind: e.g. ``persona-style`` / ``neutral`` / ``wrong-persona``.
        decoder: ``{n, temperature, top_p, max_tokens, seed}`` kwargs for vLLM.
        eval_results_root: e.g. ``eval_results/issue_375``; the cell dir
            will be created under ``eval_results_root/<cell_label>/``.
        system_prompt: passed to chat-template.
        max_user_chars / max_assistant_chars: truncation caps for few-shot
            example user/assistant turns.

    Returns:
        ``{cell_label, n_queries, n_completions, overall_rate,
           eval_questions_rate, lmsys_tail_rate, decoder, ...}``.
    """
    from vllm import SamplingParams

    log.info("=== run_cell cell=%s k=%d pool=%s ===", cell_label, k, pool_kind)

    rendered_prompts, prompt_meta = build_cell_prompts(
        tokenizer=tokenizer,
        queries=queries,
        pool=pool,
        k=k,
        adapter_id=adapter_id,
        pool_kind=pool_kind,
        system_prompt=system_prompt,
        max_user_chars=max_user_chars,
        max_assistant_chars=max_assistant_chars,
    )

    # ZLT contamination gate stage (b) — fail loudly before generation
    _zlt_eval_time_scan(rendered_prompts, cell_label, prompt_meta)

    # ZLT audit-trail dump stage (c)
    cell_dir = Path(eval_results_root) / cell_label
    _save_rendered_prompts(cell_dir, rendered_prompts, prompt_meta)

    sampling = SamplingParams(
        n=int(decoder["n"]),
        temperature=float(decoder["temperature"]),
        top_p=float(decoder["top_p"]),
        max_tokens=int(decoder["max_tokens"]),
        seed=int(decoder["seed"]),
    )
    log.info(
        "vLLM generate: n_prompts=%d n_per_prompt=%d max_tokens=%d",
        len(rendered_prompts),
        sampling.n,
        sampling.max_tokens,
    )
    outputs = llm.generate(rendered_prompts, sampling)

    # outputs is one RequestOutput per prompt; each has .outputs (a list of n CompletionOutputs).
    completions: list[list[str]] = []
    for out in outputs:
        completions.append([o.text for o in out.outputs])

    # Score
    marker_eval = score_completions(completions, marker=MARKER_TOKEN)
    stratified = stratify_rates(completions, prompt_meta, marker=MARKER_TOKEN)

    # Persist
    raw = {
        "cell_label": cell_label,
        "marker": MARKER_TOKEN,
        "decoder": dict(decoder),
        "n_queries": len(rendered_prompts),
        "n_completions_per_query": sampling.n,
        "completions": [
            {
                "query_id": prompt_meta[i].get("query_id", i),
                "query_source": prompt_meta[i].get("query_source", "unknown"),
                "completions": completions[i],
            }
            for i in range(len(completions))
        ],
    }
    (cell_dir / "raw_completions.json").write_text(json.dumps(raw, ensure_ascii=False))

    marker_payload = {
        "cell_label": cell_label,
        "marker": MARKER_TOKEN,
        "marker_eval": marker_eval,
        "stratified": stratified,
        "prompt_meta": prompt_meta,
    }
    (cell_dir / "marker_eval.json").write_text(
        json.dumps(marker_payload, ensure_ascii=False, indent=2)
    )

    summary = {
        "cell_label": cell_label,
        "adapter_id": adapter_id,
        "pool_kind": pool_kind,
        "k": k,
        "decoder": dict(decoder),
        "n_queries": len(rendered_prompts),
        "n_completions": marker_eval["total"],
        "marker": MARKER_TOKEN,
        "overall_rate": marker_eval["rate"],
        "eval_questions_rate": stratified["eval_questions"]["rate"],
        "lmsys_tail_rate": stratified["lmsys_tail"]["rate"],
        "found": marker_eval["found"],
        "total": marker_eval["total"],
    }
    (cell_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    log.info(
        "cell=%s overall=%.4f eval_q=%.4f lmsys=%.4f (found=%d/%d)",
        cell_label,
        summary["overall_rate"],
        summary["eval_questions_rate"],
        summary["lmsys_tail_rate"],
        summary["found"],
        summary["total"],
    )
    return summary


# ────────────────────────────────────────────────────────────────────────────
# Paired bootstrap (plan §4.9)
# ────────────────────────────────────────────────────────────────────────────


def paired_bootstrap_diff(
    rates_a: np.ndarray,
    rates_b: np.ndarray,
    n_boot: int = 10_000,
    seed: int = 42,
    alpha: float = 0.05,
) -> dict:
    """Paired-bootstrap 95% CI for the mean of (rates_a - rates_b).

    Pairing: same query index across arms. ``rates_a`` and ``rates_b`` are
    length-``n_queries`` arrays of per-query marker-fire fractions (out of
    n completions per query). We resample query indices with replacement
    and compute the bootstrap distribution of the paired mean difference.

    Returns:
        ``{"mean_diff", "ci_lo", "ci_hi", "n", "n_boot", "alpha",
           "ci_excludes_zero"}``.
    """
    rates_a = np.asarray(rates_a, dtype=np.float64)
    rates_b = np.asarray(rates_b, dtype=np.float64)
    if rates_a.shape != rates_b.shape:
        raise ValueError(
            f"paired_bootstrap_diff: arms must have equal length, "
            f"got len(a)={rates_a.size}, len(b)={rates_b.size}"
        )
    rng = np.random.default_rng(seed)
    n = rates_a.size
    diffs = rates_a - rates_b
    boot_means = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        boot_means[i] = diffs[idx].mean()
    lo_pct = 100.0 * (alpha / 2.0)
    hi_pct = 100.0 * (1.0 - alpha / 2.0)
    ci_lo, ci_hi = np.percentile(boot_means, [lo_pct, hi_pct])
    return {
        "mean_diff": float(diffs.mean()),
        "ci_lo": float(ci_lo),
        "ci_hi": float(ci_hi),
        "n": int(n),
        "n_boot": int(n_boot),
        "alpha": float(alpha),
        "ci_excludes_zero": bool((ci_lo > 0.0) or (ci_hi < 0.0)),
    }
