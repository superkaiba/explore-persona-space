"""Per-context DV construction for issue #1739 (round B).

Judged draws -> per-context graded DV: each rollout's score is already the
mean over its N judge draws (drop-never-coerce — a rollout with ALL draws
content-dropped is None and is EXCLUDED, never coerced; llm-judging.md rule 9);
the context DV is the mean over the K rollouts with >= 1 kept draw. Content
drops and transport losses stay split end to end (rule 24(ii)).

TF fixed +/- pool margin (llm-judging.md rule 19 — the SECONDARY non-saturating
companion DV): pools are judge-filtered ONCE on a FIXED set of eval contexts
per behavior, then FROZEN to disk (fingerprinted); the margin for a context C
is mean LN-logP(pos pool | C) - mean LN-logP(neg pool | C), scored teacher-
forced via ``capture.teacher_forced_ln_logp`` (injected — the GPU boundary).

Writes ``eval_results/issue_1739/dv_dataset/{behavior}/labeling.json``.

CONTENT HYGIENE: DV outputs carry ids + scores only; the frozen pool file
carries completion text by necessity (it IS the reusable artifact) and is
never logged.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import time
from collections.abc import Callable
from pathlib import Path

from explore_persona_space.experiments.issue_1739.constants import (
    K_ROLLOUTS,
    N_JUDGE_DRAWS,
)

logger = logging.getLogger(__name__)

# TF pool pins: persona-vectors judge-filter convention (pos > 50 / neg < 50,
# .claude/rules/persona-vectors-recipe.md) over a FIXED 100-context eval slice.
TF_POOL_N_CONTEXTS = 100
TF_POOL_POS_MIN = 50.0  # strictly greater
TF_POOL_NEG_MAX = 50.0  # strictly less
TF_POOL_PER_SIDE = 20


def parse_item_id(item_id: str) -> tuple[str, int]:
    """Invert ``judging.rollout_item_id`` -> (context_id, rollout_k)."""
    context_id, _, k_part = item_id.rpartition("_k")
    if not context_id or not k_part.isdigit():
        raise ValueError(f"malformed rollout item id: {item_id!r}")
    return context_id, int(k_part)


def build_labeling_dv(
    scores: dict[str, float | None],
    *,
    k_rollouts: int = K_ROLLOUTS,
    n_draws: int = N_JUDGE_DRAWS,
    per_item_transport_losses: dict[str, int] | None = None,
    contexts_meta: dict[str, dict] | None = None,
) -> list[dict]:
    """Judged per-rollout scores -> per-context graded DV rows.

    ``scores`` maps ``{context_id}_k{NN}`` -> mean-over-kept-draws score or
    None (all draws content-dropped). Per context: DV = mean over rollouts
    with a kept score; ``dv`` is None when EVERY rollout dropped (reported,
    never coerced). Transport losses are summed per context and kept separate
    from content drops.
    """
    per_context: dict[str, dict[int, float | None]] = {}
    for item_id, score in scores.items():
        context_id, k = parse_item_id(item_id)
        per_context.setdefault(context_id, {})[k] = score

    transport = per_item_transport_losses or {}
    rows: list[dict] = []
    for context_id in sorted(per_context):
        rollout_scores = per_context[context_id]
        if len(rollout_scores) > k_rollouts:
            raise ValueError(
                f"context {context_id}: {len(rollout_scores)} rollouts > k_rollouts={k_rollouts}"
            )
        kept = [s for s in rollout_scores.values() if s is not None]
        n_transport = sum(
            n for item_id, n in transport.items() if parse_item_id(item_id)[0] == context_id
        )
        row = {
            "context_id": context_id,
            "dv": float(sum(kept) / len(kept)) if kept else None,
            "n_rollouts_judged": len(rollout_scores),
            "n_rollouts_kept": len(kept),
            "n_rollouts_content_dropped": len(rollout_scores) - len(kept),
            "n_transport_lost_draws": n_transport,
            "n_draws_per_rollout": n_draws,
            "per_rollout_scores": {f"k{k:02d}": s for k, s in sorted(rollout_scores.items())},
        }
        if contexts_meta and context_id in contexts_meta:
            meta = contexts_meta[context_id]
            row.update(
                {
                    key: meta.get(key)
                    for key in ("behavior", "split", "rung", "group_key")
                    if key in meta
                }
            )
        rows.append(row)
    return rows


def build_three_way_dv(three_way: dict[str, str]) -> list[dict]:
    """Hallucination three-way labels -> per-context label-fraction rows.

    Per context: fractions of correct / abstained / fabricated over the
    rollouts with a decided label (``unjudged`` rollouts are excluded from the
    denominator and reported — drop-never-coerce).
    """
    per_context: dict[str, dict[int, str]] = {}
    for item_id, label in three_way.items():
        context_id, k = parse_item_id(item_id)
        per_context.setdefault(context_id, {})[k] = label
    rows: list[dict] = []
    for context_id in sorted(per_context):
        labels = list(per_context[context_id].values())
        decided = [lab for lab in labels if lab != "unjudged"]
        counts = {lab: decided.count(lab) for lab in ("correct", "abstained", "fabricated")}
        rows.append(
            {
                "context_id": context_id,
                "n_rollouts": len(labels),
                "n_decided": len(decided),
                "n_unjudged": len(labels) - len(decided),
                "counts": counts,
                "fractions": {
                    lab: (n / len(decided) if decided else None) for lab, n in counts.items()
                },
                # Fabrication fraction is the graded hallucination DV headline
                # candidate; None when nothing decided.
                "dv": (counts["fabricated"] / len(decided)) if decided else None,
            }
        )
    return rows


def dv_dataset_path(out_root: Path | str, behavior: str) -> Path:
    return Path(out_root) / "dv_dataset" / behavior / "labeling.json"


def write_dv_dataset(
    rows: list[dict],
    *,
    out_root: Path | str,
    behavior: str,
    judge_payload_meta: dict | None = None,
    git_commit: str = "unknown",
) -> Path:
    """Write the per-behavior labeling DV dataset (atomic, with repro meta)."""
    path = dv_dataset_path(out_root, behavior)
    path.parent.mkdir(parents=True, exist_ok=True)
    n_kept = sum(1 for r in rows if r.get("dv") is not None)
    payload = {
        "behavior": behavior,
        "n_contexts": len(rows),
        "n_contexts_with_dv": n_kept,
        "rows": rows,
        "judge_meta": judge_payload_meta or {},
        "git_commit": git_commit,
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    os.replace(tmp, path)
    logger.info(
        "[dv] %s: wrote %d context rows (%d with DV) -> %s", behavior, len(rows), n_kept, path
    )
    return path


# ---------------------------------------------------------------------------
# TF fixed +/- pool margin (rule 19 companion DV)
# ---------------------------------------------------------------------------


def _pool_fingerprint(**kwargs: object) -> str:
    return hashlib.sha256(json.dumps(kwargs, sort_keys=True, default=str).encode()).hexdigest()[:16]


def build_tf_pools(
    rollouts: list[dict],
    scores: dict[str, float | None],
    *,
    behavior: str,
    pool_path: Path | str,
    n_contexts: int = TF_POOL_N_CONTEXTS,
    per_side: int = TF_POOL_PER_SIDE,
    pos_min: float = TF_POOL_POS_MIN,
    neg_max: float = TF_POOL_NEG_MAX,
    seed: int = 0,
) -> dict:
    """Judge-filter the fixed +/- completion pools ONCE, then FREEZE to disk.

    Pool candidates come from the FIRST ``n_contexts`` contexts (sorted
    context_id order — deterministic); positives are rollouts with judged
    score > ``pos_min``, negatives < ``neg_max`` (the persona-vectors filter
    convention). Seeded subsample down to ``per_side`` each. An existing pool
    file with a matching fingerprint is returned VERBATIM (frozen — never
    re-filtered), so every later margin read scores the SAME answer set.
    """
    import numpy as np

    pool_path = Path(pool_path)
    fingerprint = _pool_fingerprint(
        behavior=behavior,
        n_contexts=n_contexts,
        per_side=per_side,
        pos_min=pos_min,
        neg_max=neg_max,
        seed=seed,
    )
    if pool_path.exists():
        pool = json.loads(pool_path.read_text())
        if pool.get("fingerprint") == fingerprint:
            logger.info(
                "[tf-pool] %s: frozen pool resume (%d pos / %d neg)",
                behavior,
                len(pool["pos"]),
                len(pool["neg"]),
            )
            return pool
        raise RuntimeError(
            f"TF pool at {pool_path} has fingerprint {pool.get('fingerprint')!r} != "
            f"{fingerprint!r}; a frozen pool is never silently re-filtered — move the old "
            "file aside deliberately"
        )

    fixed_contexts = sorted({parse_item_id(i)[0] for i in scores})[:n_contexts]
    fixed_set = set(fixed_contexts)
    candidates: dict[str, list[dict]] = {"pos": [], "neg": []}
    for payload in rollouts:
        item_id = f"{payload['context_id']}_k{int(payload['rollout_k']):02d}"
        if payload["context_id"] not in fixed_set:
            continue
        score = scores.get(item_id)
        if score is None:
            continue
        side = "pos" if score > pos_min else ("neg" if score < neg_max else None)
        if side is None:
            continue
        candidates[side].append(
            {"item_id": item_id, "score": score, "completion": payload["completion"]}
        )
    for side in ("pos", "neg"):
        if not candidates[side]:
            raise RuntimeError(
                f"TF pool for {behavior}: zero {side} candidates over the fixed "
                f"{len(fixed_contexts)}-context slice (judge-filter yield failure — report, "
                "never backfill)"
            )
    rng = np.random.default_rng(seed)
    pool = {"behavior": behavior, "fingerprint": fingerprint, "pos": [], "neg": []}
    for side in ("pos", "neg"):
        cand = sorted(candidates[side], key=lambda c: c["item_id"])
        if len(cand) > per_side:
            idx = rng.choice(len(cand), size=per_side, replace=False)
            cand = [cand[i] for i in sorted(idx)]
        pool[side] = cand
    pool["n_fixed_contexts"] = len(fixed_contexts)
    pool["ts"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    pool_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = pool_path.with_name(pool_path.name + ".tmp")
    tmp.write_text(json.dumps(pool, ensure_ascii=False, indent=1))
    os.replace(tmp, pool_path)
    logger.info(
        "[tf-pool] %s: froze pool (%d pos / %d neg) -> %s",
        behavior,
        len(pool["pos"]),
        len(pool["neg"]),
        pool_path,
    )
    return pool


def tf_margin_for_contexts(
    context_prompts: dict[str, str],
    pool: dict,
    ln_logp_fn: Callable[[list[tuple[str, str]]], list[float]],
) -> dict[str, float]:
    """Teacher-forced fixed +/- pool margin per context (rule 19).

    ``margin(C) = mean LN-logP(pos pool | C) - mean LN-logP(neg pool | C)``,
    the SAME frozen answer set under every context (no selection-on-outcome
    bias). ``ln_logp_fn`` is ``capture.teacher_forced_ln_logp`` bound to a
    model/tokenizer (the injected GPU boundary); one flat batched call.
    """
    pos = [c["completion"] for c in pool["pos"]]
    neg = [c["completion"] for c in pool["neg"]]
    if not pos or not neg:
        raise ValueError("TF pool has an empty side")
    context_ids = sorted(context_prompts)
    pairs: list[tuple[str, str]] = []
    for cid in context_ids:
        prompt = context_prompts[cid]
        pairs.extend((prompt, comp) for comp in pos)
        pairs.extend((prompt, comp) for comp in neg)
    lps = ln_logp_fn(pairs)
    if len(lps) != len(pairs):
        raise ValueError(f"ln_logp_fn returned {len(lps)} values for {len(pairs)} pairs")
    margins: dict[str, float] = {}
    stride = len(pos) + len(neg)
    for i, cid in enumerate(context_ids):
        chunk = lps[i * stride : (i + 1) * stride]
        pos_mean = sum(chunk[: len(pos)]) / len(pos)
        neg_mean = sum(chunk[len(pos) :]) / len(neg)
        margins[cid] = float(pos_mean - neg_mean)
    return margins
