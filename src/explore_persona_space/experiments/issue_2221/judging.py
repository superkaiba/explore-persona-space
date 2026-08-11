"""Issue #2221 judge wrapper — graded 0-100 judging with rule-28 api-refusal remediation.

Wraps ``eval.graded_judge.judge_graded`` and implements the llm-judging rule-28
contract (#2151/#1739): API-level ``stop_reason == "refusal"`` draws (the THIRD
drop class — neither content drops nor transport losses) are counted
SEPARATELY per wave/arm and remediated by a targeted SYNC re-issue at the
IDENTICAL instrument (same rubric / judge model / temperature / max_tokens;
``threshold_base`` large forces the sync path — the
``scripts/issue1739_evilood_refusal_rejudge.py`` recipe). Band counts and any
yield read downstream are computed POST-remediation.

Content hygiene: this module never logs row text — accounting is by item id +
counts only.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)

_FORCE_SYNC_THRESHOLD = 10**9  # n_items >= threshold_base routes to batch; huge => sync


def _per_item_kept_draws(save_raw: Path, item_ids: set[str]) -> dict[str, list[float]]:
    """Re-read a judge ``save_raw`` file into per-item KEPT draw score lists."""
    from explore_persona_space.eval.graded_judge import _score_from_parsed

    with open(save_raw) as f:
        raw = json.load(f)
    out: dict[str, list[float]] = {iid: [] for iid in item_ids}
    for cid, parsed in raw.get("all_scores", {}).items():
        item_id = cid.rsplit("__", 2)[0]
        if item_id not in out:
            continue
        s = _score_from_parsed(parsed)
        if s is not None:
            out[item_id].append(float(s))
    return out


def judge_with_refusal_remediation(
    items: list[tuple[str, str, str]],
    eval_prompt: str,
    *,
    n_draws: int,
    cache_root: Path,
    save_raw_root: Path,
    tag: str,
    max_tokens: int,
    judge_model: str | None = None,
    temperature: float | None = None,
    threshold_base: int | None = None,
) -> tuple[dict[str, float | None], dict]:
    """Judge ``items`` and remediate api-refusal-censored items via sync re-issue.

    Returns ``(merged_scores, accounting)``:

    - ``merged_scores``: item_id -> mean over kept draws from the primary pass
      MERGED with the targeted sync re-issue draws (None when an item still
      has zero kept draws post-remediation).
    - ``accounting``: the three-way drop-class split per rule 28 —
      ``{n_content_dropped, n_transport_lost, n_api_refusal, n_truncation,
      n_items_censored_pre, n_items_rescued, n_items_unscored_post, ...}``.

    The re-issue uses the IDENTICAL instrument against a FRESH cache dir
    (rule 24(ii): the rubric-keyed cache would replay the primary pass).
    """
    from explore_persona_space.eval.graded_judge import judge_graded

    kwargs: dict = {}
    if judge_model is not None:
        kwargs["judge_model"] = judge_model
    if temperature is not None:
        kwargs["temperature"] = temperature

    save_raw_root.mkdir(parents=True, exist_ok=True)
    primary_raw = save_raw_root / f"judge_raw_{tag}.json"
    jr = judge_graded(
        items,
        eval_prompt,
        n_draws=n_draws,
        cache_dir=cache_root / tag,
        save_raw=primary_raw,
        max_tokens=max_tokens,
        threshold_base=threshold_base,
        **kwargs,
    )
    item_ids = {iid for iid, _, _ in items}
    kept = _per_item_kept_draws(primary_raw, item_ids)

    # Rule-28 target set: every item that lost >= 1 draw to an API-level
    # refusal (outcome-correlated censoring — remediate the whole set, not
    # only zero-valid items).
    censored = sorted(set(jr.per_item_api_refusals) & item_ids)
    accounting = {
        "tag": tag,
        "n_items": len(items),
        "n_draws": n_draws,
        "n_total_draws": jr.n_total_draws,
        "n_content_dropped": jr.n_dropped_draws,
        "n_instructed_refusal": jr.n_refusal_draws,
        "n_truncation": jr.n_truncation_dropped_draws,
        "n_transport_lost": jr.n_transport_lost_draws,
        "n_api_refusal": jr.n_api_refusal_draws,
        "stop_reason_tally": dict(jr.stop_reason_tally),
        "n_items_censored_pre": len(censored),
        "n_items_rescued": 0,
        "sync_reissue": None,
    }

    if censored:
        logger.warning(
            "[judge:%s] rule-28 remediation: %d/%d items carry api-refusal draws — "
            "targeted SYNC re-issue at the identical instrument",
            tag,
            len(censored),
            len(items),
        )
        reissue_items = [(iid, q, a) for iid, q, a in items if iid in censored]
        sync_raw = save_raw_root / f"judge_raw_{tag}_sync_reissue.json"
        jr_sync = judge_graded(
            reissue_items,
            eval_prompt,
            n_draws=n_draws,
            cache_dir=cache_root / f"{tag}_sync_reissue",  # FRESH cache (rule 24(ii))
            save_raw=sync_raw,
            max_tokens=max_tokens,
            threshold_base=_FORCE_SYNC_THRESHOLD,  # force the SYNC path (#1739 recipe)
            **kwargs,
        )
        sync_kept = _per_item_kept_draws(sync_raw, set(censored))
        rescued = 0
        for iid in censored:
            extra = sync_kept.get(iid, [])
            if extra and not kept[iid]:
                rescued += 1
            kept[iid].extend(extra)
        accounting["n_items_rescued"] = rescued
        accounting["sync_reissue"] = {
            "n_items": len(reissue_items),
            "n_api_refusal": jr_sync.n_api_refusal_draws,
            "n_transport_lost": jr_sync.n_transport_lost_draws,
            "n_content_dropped": jr_sync.n_dropped_draws,
        }

    merged: dict[str, float | None] = {
        iid: (float(sum(v) / len(v)) if v else None) for iid, v in kept.items()
    }
    accounting["n_items_unscored_post"] = sum(1 for v in merged.values() if v is None)
    return merged, accounting
