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
import re
from pathlib import Path

logger = logging.getLogger(__name__)

_FORCE_SYNC_THRESHOLD = 10**9  # n_items >= threshold_base routes to batch; huge => sync

# ── Batch-API-safe judge item ids (#2221 r9) ─────────────────────────────────
#
# ``judge_graded`` -> ``batch_judge`` embeds the item id VERBATIM in the Batch
# API custom_id ``{item_id}__{idx:05d}__{comp_idx:02d}`` (an 11-char suffix),
# validated against the Anthropic grammar ``^[a-zA-Z0-9_-]{1,64}$``
# (``judge_dispatch._validate_custom_ids``; #1795). So an item id on this path
# must (a) use only ``[a-zA-Z0-9_-]``, (b) avoid the ``__`` delimiter
# (``judge_graded`` raises on it), and (c) stay <= 53 chars — the #1415/#1776
# alias budget (64 - 11). Attempt-6 crash: the pilot's ``{tag}::{iid}`` arm ids
# carried ``::`` and died at ``_validate_custom_ids`` pre-submit.
BATCH_ITEM_ID_MAX = 53
_BATCH_ID_LEGAL_RE = re.compile(r"^[a-zA-Z0-9_-]{1,53}$")
_BATCH_ID_ILLEGAL_CHARS_RE = re.compile(r"[^a-zA-Z0-9_-]+")


def batch_safe_item_id(raw_id: str) -> str:
    """Grammar-legal Batch-API judge item id for the ``judge_graded`` path.

    A raw id that already satisfies the custom_id grammar (charset, no ``__``
    delimiter, <= 53 chars) passes through UNCHANGED — downstream joins key on
    it (``phase_aggregate``'s panel prefixes; the band files' item ids).
    Anything else maps to ``{sanitized-prefix}-{10-hex}``: a readable sanitized
    prefix plus the first 10 hex chars of ``batch_judge.make_custom_id`` (the
    sanctioned sha256 sanitizer) computed over the RAW id, so distinct raw ids
    stay distinct. Deterministic and idempotent for a fixed raw id.
    """
    from explore_persona_space.eval.batch_judge import make_custom_id

    if _BATCH_ID_LEGAL_RE.fullmatch(raw_id) and "__" not in raw_id:
        return raw_id
    digest = make_custom_id(raw_id)[:10]
    sanitized = _BATCH_ID_ILLEGAL_CHARS_RE.sub("-", raw_id)
    sanitized = re.sub(r"_{2,}", "-", sanitized)  # judge_graded bans the __ delimiter
    prefix = sanitized[: BATCH_ITEM_ID_MAX - len(digest) - 1].strip("-_")
    return f"{prefix}-{digest}" if prefix else digest


def alias_judge_items(
    items: list[tuple[str, str, str]],
) -> tuple[list[tuple[str, str, str]], dict[str, str]]:
    """Alias every ``(item_id, q, a)`` id via :func:`batch_safe_item_id`.

    Returns ``(aliased_items, alias_to_raw)``. Bijectivity is ASSERTED over the
    realized key set (gotchas.md Batch custom_id entry: substitution alone is
    not injective): an alias collision between DISTINCT raw ids — or a
    duplicate raw id, which would silently collapse per-item scores — raises
    ``ValueError`` instead of dispatching a corrupted wave.
    """
    alias_to_raw: dict[str, str] = {}
    out: list[tuple[str, str, str]] = []
    for raw, q, a in items:
        alias = batch_safe_item_id(raw)
        prior = alias_to_raw.get(alias)
        if prior == raw:
            raise ValueError(f"duplicate judge item id: {raw!r}")
        if prior is not None:
            raise ValueError(f"judge item id alias collision: {alias!r} <- {prior!r} and {raw!r}")
        alias_to_raw[alias] = raw
        out.append((alias, q, a))
    return out, alias_to_raw


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
      has zero kept draws post-remediation). Keys are the CALLER's raw item
      ids: internally every id routes through :func:`alias_judge_items` (the
      Batch-API custom_id grammar, #2221 r9) and is reverse-mapped on return,
      so callers' joins never see the alias.
    - ``accounting``: the three-way drop-class split per rule 28 —
      ``{n_content_dropped, n_transport_lost, n_api_refusal, n_truncation,
      n_items_censored_pre, n_items_rescued, n_items_unscored_post, ...}``.

    The re-issue uses the IDENTICAL instrument against a FRESH cache dir
    (rule 24(ii): the rubric-keyed cache would replay the primary pass).
    """
    from explore_persona_space.eval.graded_judge import judge_graded

    # Batch-API custom_id grammar (#2221 r9): alias EVERY id at this single
    # enforcement point (grammar-legal by construction, never by luck); the
    # returned scores are reverse-mapped to the caller's raw ids below.
    items, alias_to_raw = alias_judge_items(list(items))

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

    # Reverse-map aliases back to the caller's raw ids (join contract above).
    merged: dict[str, float | None] = {
        alias_to_raw[iid]: (float(sum(v) / len(v)) if v else None) for iid, v in kept.items()
    }
    accounting["n_items_unscored_post"] = sum(1 for v in merged.values() if v is None)
    return merged, accounting
