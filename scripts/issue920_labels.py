"""Plain-English labels for issue #920 summary-family slugs (figure surfaces).

Slugs stay in tables / launch commands / footers; reader-facing figure axes,
legends, and titles use these labels (plan §4 slug-confinement rule).
"""

from __future__ import annotations

_CTX = {
    "ctx_wt_mean": "input mean (with template)",
    "ctx_wt_max": "input max (with template)",
    "ctx_co_mean": "input content mean",
    "ctx_co_max": "input content max",
    "ctx_ah_nl": "assistant-header newline",
    "ctx_ah_nl_lmean": "assistant-header newline, layer-mean",
    "ctx_ah_nl_lmax": "assistant-header newline, layer-max",
    "ctx_tt_im_end": "trailing turn-end token",
    "ctx_tt_nl": "trailing newline",
    "ctx_tt_im_start": "trailing turn-start token",
    "ctx_tt_assistant": "trailing 'assistant' token",
    "ctx_blk_mean": "template-block mean",
    "ctx_blk_max": "template-block max",
    "ctx_wt_pool_meanmean": "input mean, layer-mean",
    "ctx_wt_pool_maxmax": "input max, layer-max",
    "ctx_wt_pool_mean_of_max": "input max, layer-mean",
    "ctx_wt_pool_max_of_mean": "input mean, layer-max",
    "ctx_co_pool_meanmean": "content mean, layer-mean",
    "ctx_co_pool_maxmax": "content max, layer-max",
    "ctx_co_pool_mean_of_max": "content max, layer-mean",
    "ctx_co_pool_max_of_mean": "content mean, layer-max",
}

_ANS = {
    "ans_content_mean": "whole-answer mean",
    "ans_content_max": "whole-answer max",
    "ans_content_pool_meanmean": "answer mean, layer-mean",
    "ans_content_pool_maxmax": "answer max, layer-max",
    "ans_content_pool_mean_of_max": "answer max, layer-mean",
    "ans_content_pool_max_of_mean": "answer mean, layer-max",
    "ans_im_end": "turn-end token",
    "ans_last_content": "last answer token",
    "ans_turn_nl": "post-answer newline",
    "ans_uh_im_start": "next-user turn-start token",
    "ans_uh_user": "next-user 'user' token",
    "ans_uh_nl": "next-user-header newline",
    "ans_uh_nl_lmean": "next-user-header newline, layer-mean",
    "ans_uh_nl_lmax": "next-user-header newline, layer-max",
    "ans_uhdr_mean": "next-user-header block mean",
    "ans_uhdr_max": "next-user-header block max",
    "ans_blk5_mean": "5-token boundary block mean",
    "ans_blk5_max": "5-token boundary block max",
    "ans_wtn_mean": "answer+turn-end mean",
    "ans_wtn_max": "answer+turn-end max",
    "ans_wtf_mean": "answer+boundary mean",
    "ans_wtf_max": "answer+boundary max",
    "ans_wtf_pool_meanmean": "answer+boundary mean, layer-mean",
    "ans_wtf_pool_maxmax": "answer+boundary max, layer-max",
    "ans_wtf_pool_mean_of_max": "answer+boundary max, layer-mean",
    "ans_wtf_pool_max_of_mean": "answer+boundary mean, layer-max",
}

_BEH = {
    "sycophancy": "sycophancy",
    "refusal": "refusal",
    "harmful_compliance": "harmful compliance",
    "fact_expression": "fact expression",
    "format_style": "format style",
    "self_report": "self-report",
    "persona_drift": "persona drift",
}


def plain_family(slug: str) -> str:
    """Plain-English label for a context/answer summary-family slug."""
    if slug in _CTX:
        return _CTX[slug]
    if slug in _ANS:
        return _ANS[slug]
    if slug.startswith("ctx_lastk_"):
        k = int(slug.rsplit("_", 1)[1])
        return "last content token" if k == 1 else f"last {k} content tokens"
    if slug.startswith("pos_tail_"):
        k = int(slug.rsplit("_", 1)[1])
        return "final answer token" if k == 1 else f"{k}th-from-end answer token"
    if slug.startswith("pos_head_"):
        j = int(slug.rsplit("_", 1)[1])
        return f"answer token {j + 1}"
    return slug


def plain_cell(cell: str) -> str:
    """Plain-English label for `fam@Lk` cells (layer suffix kept)."""
    if "@" in cell:
        fam, layer = cell.split("@", 1)
        return f"{plain_family(fam)} @ {layer}"
    return plain_family(cell)


def plain_behavior(slug: str) -> str:
    """Plain-English label for a behavior slug."""
    return _BEH.get(slug, slug.replace("_", " "))


_CTX_GROUP_DISPLAY = {
    "house": "house",
    "phub": "persona-hub",
    "wc": "wildchat",
    "icl": "icl",
    "reph": "rephrase",
    "fmt": "fmt",
    "behav": "behav",
    "default": "default",
}


def plain_context_id(cid: str) -> str:
    """Readable point label for a battery context id.

    Strips the ``fN_`` family prefix and de-slugs, e.g.
    ``f5_fmt_markdown_table`` -> ``fmt: markdown table``,
    ``f8_behav_harmful`` -> ``behav: harmful``,
    ``f3_icl_marker_k4`` -> ``icl: marker k4``.
    """
    parts = cid.split("_")
    if parts and parts[0].startswith("f") and parts[0][1:].isdigit():
        parts = parts[1:]
    if len(parts) >= 2 and parts[0] in _CTX_GROUP_DISPLAY:
        return f"{_CTX_GROUP_DISPLAY[parts[0]]}: {' '.join(parts[1:])}"
    return " ".join(parts)
