#!/usr/bin/env python3
"""Render the #2162 bank dashboard + Result 0 qualitative gallery as static HTML.

Two self-contained artifacts for the `mapshift` inline round (consolidation plan
`docs/reports/issue_2162_consolidation_plan.md` § "Bank dashboard" + § "Result 0"),
following the `scripts/issue2094_patch_gallery_html.py` conventions (compact
tables, light/dark CSS, committed under `docs/`):

- ``docs/issue2162_bank_dashboard.html`` — minimal reference dashboard, zero
  interpretation: the carrier conversations (grouped by the 17 distinct
  per-cell carrier sets realized in bank.json — there is no single global
  12-carrier set), all 39 parameter cells with their stored value strings, and
  one worked example per cell (context A vs context B, varied span diff-marked
  via ``<mark>``, each with its unpatched anchor answer).
- ``docs/issue2162_result0_gallery.html`` — 39 collapsible sections (one per
  cell), rows = directed pairs at (context-end, steered): context A -> anchor
  answer A; context B -> anchor answer B; A patched with B's context-end state
  -> patched answer + per-pair F_beh / F_act. Reverse directions are shown
  where the bank banks them (the conflict fwd/rev sibling cells; everywhere
  else a factual "reverse direction not banked" line). Sections and rows are
  sortable by F_beh (vanilla JS, no CDN). A collapsible interpretation box
  (Fable 5 analysis) sits at the top; everything below it is raw data.

Inputs (all banked; HF paths staged at ONE pinned revision):
  - ``issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json`` (contexts, pairs,
    cells, carriers, values)
  - ``issue2162_ctxinfo/raw_completions/grid/shard_<cell>__ce__steered.jsonl``
    (patched rollout text; per-row temperature/seed/draw metadata)
  - ``issue2162_ctxinfo/raw_completions/anchors/anchors_*.jsonl`` (unpatched
    anchor rollout text, 10 draws per context)
  - ``issue2162_ctxinfo/raw_completions/judge_raw/scores/coherence.{grid,anchors}
    .scores.jsonl`` (per-draw judge coherence scores; coherent = score > 60,
    the ``issue2162_analysis.COHERENCE_THRESHOLD`` convention)
  - ``eval_results/issue_2162/f_metrics/f_cells.jsonl`` (per-pair F_beh + F_act,
    context-end steered) + ``two_by_two.json`` (per-cell 2x2 verdicts)

Usage::

    uv run python scripts/issue2162_dashboards.py \
        --dash-dl /mnt/eps-data/thomasjiralerspong/issue2162_mapshift/dash_dl \
        --analysis-md /tmp/issue2162_fable5_transfer_analysis.md
"""

from __future__ import annotations

import argparse
import datetime
import difflib
import html
import json
import re
import statistics
from pathlib import Path

PINNED_REVISION = "7d3ac543a5a4202e3996be1498886f2bab637c15"
HF_REPO = "superkaiba1/explore-persona-space-data"
COHERENCE_THRESHOLD = 60.0  # issue2162_analysis.py convention: coherent = score > 60
ANSWER_EXCERPT_CHARS = 600
CONTEXT_EXCERPT_CHARS = 240

CSS = """
:root { color-scheme: light dark; }
body { font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       margin: 24px auto; max-width: 1700px; padding: 0 16px; color: #1a1a1a; background: #fff; }
h1 { font-size: 20px; margin: 0 0 6px; }
h2 { font-size: 16px; margin: 18px 0 6px; }
p.lede { margin: 0 0 6px; color: #444; max-width: 120ch; }
p.lede code, span.mono { background: #f2f2f2; padding: 1px 4px; border-radius: 3px;
                         font-family: ui-monospace, monospace; font-size: 12px; }
table { border-collapse: collapse; width: 100%; margin-top: 10px; }
th, td { border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; text-align: left; }
th { background: #f6f6f6; font-size: 12px; letter-spacing: .02em; }
td.num { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
mark { background: #ffe08a; color: inherit; padding: 0 1px; border-radius: 2px; }
.f { font-weight: 700; }
.hi { color: #1b6d2f; } .mid { color: #9a6400; } .lo { color: #a11; } .na { color: #777; }
.d { font-size: 11.5px; color: #666; }
details { margin-top: 4px; }
summary { cursor: pointer; color: #06c; font-size: 12px; }
details.cell { border: 1px solid #ddd; border-radius: 6px; margin: 10px 0; padding: 6px 10px; }
details.cell > summary { font-size: 14px; color: inherit; font-weight: 600; }
details.interp { border: 2px solid #9a6400; border-radius: 6px; margin: 12px 0; padding: 6px 10px; }
details.interp > summary { font-size: 14px; color: #9a6400; font-weight: 700; }
pre { white-space: pre-wrap; background: #fafafa; border: 1px solid #eee; border-radius: 4px;
      padding: 8px; margin: 6px 0 0; font-size: 11.5px; max-height: 380px; overflow: auto; }
div.pairgroup { border-top: 1px solid #e5e5e5; padding: 8px 0; }
div.pairhead { font-size: 12.5px; margin-bottom: 4px; }
div.cols { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 10px; }
div.col { min-width: 0; font-size: 12.5px; }
div.col .lbl { font-weight: 600; font-size: 12px; margin-bottom: 2px; }
button.sort { font-size: 12px; margin-right: 6px; cursor: pointer; }
@media (prefers-color-scheme: dark) {
  body { background: #121212; color: #e6e6e6; }
  th { background: #1e1e1e; } th, td, details.cell, div.pairgroup { border-color: #333; }
  p.lede { color: #bbb; } p.lede code, span.mono { background: #222; }
  pre { background: #191919; border-color: #2a2a2a; }
  mark { background: #6b5900; }
}
"""

SORT_JS = """
function cmpBy(mode) {
  return function (x, y) {
    if (mode === 'bank') return (+x.dataset.order) - (+y.dataset.order);
    var a = parseFloat(x.dataset.fbeh), b = parseFloat(y.dataset.fbeh);
    var an = isNaN(a), bn = isNaN(b);
    if (an && bn) return (+x.dataset.order) - (+y.dataset.order);
    if (an) return 1;
    if (bn) return -1;
    return mode === 'best' ? b - a : a - b;
  };
}
function sortSections(mode) {
  var c = document.getElementById('sections');
  var secs = Array.prototype.slice.call(c.querySelectorAll(':scope > details.cell'));
  secs.sort(cmpBy(mode));
  secs.forEach(function (s) { c.appendChild(s); });
}
function sortRows(btn, mode) {
  var c = btn.closest('details.cell').querySelector('.rows');
  var rows = Array.prototype.slice.call(c.querySelectorAll(':scope > div.pairgroup'));
  rows.sort(cmpBy(mode));
  rows.forEach(function (r) { c.appendChild(r); });
}
"""


def esc(text: str) -> str:
    """Element-content HTML escape (quote=False; never used for attribute values)."""
    return html.escape(text, quote=False)


# ── loading ───────────────────────────────────────────────────────────


def iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def load_bank(dash_dl: Path) -> dict:
    bank = json.loads(
        (dash_dl / "issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json").read_text()
    )
    assert len(bank["contexts"]) == 1404, len(bank["contexts"])
    assert len(bank["pairs"]) == 1404, len(bank["pairs"])
    assert len(bank["cells"]) == 39, len(bank["cells"])
    return bank


def load_grid_ce_steered(dash_dl: Path, cells: list[str]) -> dict[str, dict[int, dict]]:
    """pair_id -> {draw: row} for the (context-end, steered) arm of every cell."""
    out: dict[str, dict[int, dict]] = {}
    grid_dir = dash_dl / "issue2162_ctxinfo/raw_completions/grid"
    for cell in cells:
        shard = grid_dir / f"shard_{cell}__ce__steered.jsonl"
        assert shard.is_file(), shard
        for row in iter_jsonl(shard):
            assert row["slot"] == "ce" and row["arm"] == "steered", (row["slot"], row["arm"])
            out.setdefault(row["pair_id"], {})[row["draw"]] = row
    assert out, "no grid rows loaded"
    return out


def load_anchor_rows(dash_dl: Path) -> dict[str, dict[int, dict]]:
    """context_id -> {draw: row} over the anchors_gate_* + anchors_rest_* shards."""
    out: dict[str, dict[int, dict]] = {}
    files = sorted((dash_dl / "issue2162_ctxinfo/raw_completions/anchors").glob("anchors_*.jsonl"))
    assert files, "no anchor shards staged"
    for f in files:
        for row in iter_jsonl(f):
            out.setdefault(row["context_id"], {})[row["draw"]] = row
    assert len(out) == 1404, len(out)
    return out


def load_coherence(
    dash_dl: Path,
) -> tuple[dict[tuple[str, int], float], dict[tuple[str, int], float]]:
    """(grid ce/steered (pair_id, draw) -> score, anchor (context_id, draw) -> score)."""
    scores_dir = dash_dl / "issue2162_ctxinfo/raw_completions/judge_raw/scores"
    grid: dict[tuple[str, int], float] = {}
    for row in iter_jsonl(scores_dir / "coherence.grid.scores.jsonl"):
        if row["slot"] == "ce" and row["arm"] == "steered" and row["score"] is not None:
            grid[(row["pair_id"], row["draw"])] = row["score"]
    anchors: dict[tuple[str, int], float] = {}
    for row in iter_jsonl(scores_dir / "coherence.anchors.scores.jsonl"):
        if row["score"] is not None:
            anchors[(row["context_id"], row["draw"])] = row["score"]
    assert grid and anchors, (len(grid), len(anchors))
    return grid, anchors


def load_f_cells(f_metrics: Path) -> dict[str, dict]:
    """pair_id -> f_cells row at (ce, steered)."""
    out = {}
    for row in iter_jsonl(f_metrics / "f_cells.jsonl"):
        if row["slot"] == "ce" and row["arm"] == "steered":
            out[row["pair_id"]] = row
    assert out, "no ce/steered f_cells rows"
    return out


def load_two_by_two(f_metrics: Path) -> dict[str, dict]:
    """cell -> two_by_two row at slot ce."""
    data = json.loads((f_metrics / "two_by_two.json").read_text())
    return {r["cell"]: r for r in data["cells"] if r["slot"] == "ce"}


# ── rendering helpers ─────────────────────────────────────────────────


def render_context(ctx: dict) -> str:
    """Plain-text role-labeled render of one bank context (system/history/user)."""
    parts: list[str] = []
    if ctx.get("system"):
        parts.append(f"[system]\n{ctx['system']}")
    for turn in ctx.get("history") or []:
        parts.append(f"[{turn['role']}]\n{turn['content']}")
    parts.append(f"[user]\n{ctx['user']}")
    header = ctx.get("role_header")
    if header and header != "assistant":
        parts.append(f"[generation-prompt role header]\n{header}")
    return "\n\n".join(parts)


def diff_mark(a: str, b: str) -> tuple[str, str, str, str]:
    """HTML-escape both texts and <mark> the segments that differ (SequenceMatcher).

    Returns (marked_a, marked_b, changed_a, changed_b) where changed_* is the
    plain-text concatenation of each side's differing segments.
    """
    sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
    out_a: list[str] = []
    out_b: list[str] = []
    chg_a: list[str] = []
    chg_b: list[str] = []
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        seg_a, seg_b = a[i1:i2], b[j1:j2]
        if op == "equal":
            out_a.append(esc(seg_a))
            out_b.append(esc(seg_b))
        else:
            if seg_a:
                out_a.append(f"<mark>{esc(seg_a)}</mark>")
                chg_a.append(seg_a)
            if seg_b:
                out_b.append(f"<mark>{esc(seg_b)}</mark>")
                chg_b.append(seg_b)
    return "".join(out_a), "".join(out_b), " … ".join(chg_a), " … ".join(chg_b)


def first_coherent(draws: dict[int, dict], coh: dict, key_of) -> tuple[dict, str]:
    """First draw (ascending index) with judge coherence > 60; factual label.

    Falls back to the first available draw with an explicit incoherence label
    when no draw is coherent (never imputes coherence).
    """
    assert draws, "no draws"
    n = len(draws)
    for d in sorted(draws):
        score = coh.get(key_of(draws[d], d))
        if score is not None and score > COHERENCE_THRESHOLD:
            row = draws[d]
            label = (
                f"draw index {d} of {n} draws (first coherent; judge coherence {score:.0f} > 60), "
                f"temperature {row['temperature']}"
            )
            if row.get("cap_hit"):
                label += "; hit the max_new_tokens cap"
            return row, label
    d = min(draws)
    row = draws[d]
    score = coh.get(key_of(row, d))
    score_txt = "unscored" if score is None else f"judge coherence {score:.0f} <= 60"
    label = (
        f"no coherent draw among {n}; showing draw index {d} ({score_txt}), "
        f"temperature {row['temperature']}"
    )
    if row.get("cap_hit"):
        label += "; hit the max_new_tokens cap"
    return row, label


def answer_block(text: str, label: str) -> str:
    """Answer excerpt + the remainder behind an expand control (disclosed).

    The full banked text is always present: for long answers the visible block
    holds the first ``ANSWER_EXCERPT_CHARS`` chars and the expand payload holds
    the remaining chars verbatim (visible + expanded = the full banked text).
    """
    esc_label = esc(label)
    if len(text) <= ANSWER_EXCERPT_CHARS:
        return f"<pre>{esc(text)}</pre><div class='d'>{esc_label}; full banked text shown</div>"
    head, rest = text[:ANSWER_EXCERPT_CHARS], text[ANSWER_EXCERPT_CHARS:]
    return (
        f"<pre>{esc(head)}…</pre>"
        f"<div class='d'>{esc_label}; truncated at {ANSWER_EXCERPT_CHARS} chars for display "
        f"({len(text)} chars banked) — expand shows the remaining {len(rest)} chars verbatim "
        f"(visible + expanded = the full banked text)</div>"
        f"<details><summary>truncated — expand the remaining {len(rest)} chars</summary>"
        f"<pre>…{esc(rest)}</pre></details>"
    )


def context_block(label: str, marked_html: str, changed: str, ctx_id: str) -> str:
    span = " ".join(changed.split())
    if len(span) > CONTEXT_EXCERPT_CHARS:
        span = span[:CONTEXT_EXCERPT_CHARS] + "…"
    return (
        f'<div class="lbl">{esc(label)} <span class="mono">{esc(ctx_id)}</span>'
        f"</div>"
        f"<div>varied span: <mark>{esc(span)}</mark></div>"
        f"<details><summary>full context (varied span marked)</summary>"
        f"<pre>{marked_html}</pre></details>"
    )


def f_class(value: float | None) -> str:
    if value is None:
        return "na"
    return "hi" if value >= 0.65 else ("mid" if value > 0.05 else "lo")


def fmt_f(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def render_markdown_min(md: str) -> str:
    """Minimal markdown -> HTML for the interpretation box: paragraphs, **bold**,
    *italic*, `code`. Everything else is escaped verbatim."""
    out: list[str] = []
    for para in re.split(r"\n\s*\n", md.strip()):
        t = esc(para)
        t = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", t, flags=re.S)
        t = re.sub(r"(?<!\*)\*([^*\n][^*]*?)\*(?!\*)", r"<em>\1</em>", t, flags=re.S)
        t = re.sub(r"`([^`]+)`", r'<span class="mono">\1</span>', t)
        out.append(f"<p>{t}</p>")
    return "\n".join(out)


def provenance_line(bank: dict, grid_meta: dict, anchor_meta: dict) -> str:
    return (
        f"Bank seed {bank['seed']} (frozen, model {bank['model_id']}, bank_sha "
        f"{bank['bank_sha'][:12]}); all inputs pinned to HF dataset "
        f"<span class='mono'>{HF_REPO}</span> revision "
        f"<span class='mono'>{PINNED_REVISION}</span>. Patch = full-state replace at all 28 "
        f"layers at the context-end position (donor = context B's context-end state). Decoding "
        f"as read from the staged shards: grid {grid_meta['draws']} draws/pair, temperature "
        f"{grid_meta['temps']}, per-draw seeds {grid_meta['seeds']}; anchors "
        f"{anchor_meta['draws']} draws/context, temperature {anchor_meta['temps']}, per-draw "
        f"seeds {anchor_meta['seeds']}. Coherent = judge coherence score > 60 "
        f"(coherence.grid/anchors.scores.jsonl at the same revision)."
    )


def _decoding_meta(rows_by_key: dict[str, dict[int, dict]]) -> dict:
    temps = sorted({r["temperature"] for d in rows_by_key.values() for r in d.values()})
    seeds = sorted({r["seed"] for d in rows_by_key.values() for r in d.values()})
    draws = sorted({len(d) for d in rows_by_key.values()})
    return {
        "temps": "/".join(str(t) for t in temps),
        "seeds": f"{min(seeds)}–{max(seeds)}" if len(seeds) > 1 else str(seeds[0]),
        "draws": "/".join(str(d) for d in draws),
    }


def footer(git_commit: str) -> str:
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f'<p class="d">Generated {ts} by scripts/issue2162_dashboards.py at commit '
        f"{esc(git_commit)}.</p>"
    )


# ── bank dashboard ────────────────────────────────────────────────────


def build_bank_dashboard(
    bank: dict,
    anchors: dict[str, dict[int, dict]],
    coh_anchor: dict[tuple[str, int], float],
    grid_meta: dict,
    anchor_meta: dict,
    git_commit: str,
) -> str:
    cells: dict[str, dict] = bank["cells"]
    contexts: dict[str, dict] = bank["contexts"]
    pairs: list[dict] = bank["pairs"]
    first_pair_of_cell: dict[str, dict] = {}
    for p in pairs:
        first_pair_of_cell.setdefault(p["cell"], p)

    parts: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>#2162 bank dashboard</title>",
        f"<style>{CSS}</style></head><body>",
        "<h1>#2162 context-information bank — reference dashboard</h1>",
        f"<p class='lede'>{provenance_line(bank, grid_meta, anchor_meta)}</p>",
        "<p class='lede'>1,404 contexts = 39 parameter cells × 12 carriers × 3 values "
        "(conflict cells: 6 composite values). Factual reference only — no scores, no "
        "interpretation. Under leave-one-carrier-out, each carrier is the held-out fold "
        "exactly once (12 folds).</p>",
    ]

    # 1. carrier conversations, grouped by the distinct carrier sets realized in bank.json
    groups: dict[tuple, list[str]] = {}
    for cell, meta in cells.items():
        key = tuple(sorted((cid, c["text"]) for cid, c in meta["carriers"].items()))
        groups.setdefault(key, []).append(cell)
    parts.append("<h2>Carrier conversations</h2>")
    parts.append(
        f"<p class='lede'>bank.json realizes {len(groups)} distinct 12-carrier sets across the "
        "39 cells (carrier texts are carrier-class- and cell-specific); each set is listed once "
        "with the cells that use it.</p>"
    )
    for gi, (key, cell_list) in enumerate(groups.items(), 1):
        carrier_meta = cells[cell_list[0]]["carriers"]
        rows = []
        for cid, text in key:
            prov = carrier_meta[cid]["provenance"]
            rows.append(
                f"<details><summary>carrier {esc(cid)} "
                f"({esc(prov)})</summary><pre>{esc(text)}</pre></details>"
            )
        cl = ", ".join(sorted(cell_list))
        parts.append(
            f"<details><summary>carrier set {gi} — used by {len(cell_list)} cell(s): "
            f"{esc(cl)}</summary>{''.join(rows)}</details>"
        )

    # 2. the 39 parameter cells with their stored value strings
    parts.append("<h2>Parameter cells and value strings</h2>")
    parts.append(
        "<table><tr><th>cell</th><th>base type</th><th>span locus</th>"
        "<th>values (as stored in bank.json)</th></tr>"
    )
    for cell, meta in cells.items():
        vals = "".join(
            f"<div><span class='mono'>{esc(v)}</span>: {esc(s)}</div>"
            for v, s in meta["values"].items()
        )
        parts.append(
            f"<tr><td class='mono'>{esc(cell)}</td>"
            f"<td>{esc(meta['base_type'])}</td>"
            f"<td>{esc(meta['span_locus'])}</td><td>{vals}</td></tr>"
        )
    parts.append("</table>")

    # 3. one worked example per cell
    parts.append("<h2>Worked examples (one per cell)</h2>")
    parts.append(
        "<p class='lede'>Per cell: the first banked directed pair (same carrier) — context "
        "with value A, its unpatched anchor answer, context with value B, its anchor answer. "
        "The varied span is diff-marked between the two contexts.</p>"
    )
    for cell in cells:
        p = first_pair_of_cell[cell]
        ctx_a, ctx_b = contexts[p["a"]], contexts[p["b"]]
        text_a, text_b = render_context(ctx_a), render_context(ctx_b)
        marked_a, marked_b, chg_a, chg_b = diff_mark(text_a, text_b)
        row_a, lab_a = first_coherent(
            anchors[p["a"]], coh_anchor, lambda r, d: (r["context_id"], d)
        )
        row_b, lab_b = first_coherent(
            anchors[p["b"]], coh_anchor, lambda r, d: (r["context_id"], d)
        )
        parts.append(
            f"<details class='cell'><summary>{esc(cell)} — pair "
            f"<span class='mono'>{esc(p['pair_id'])}</span></summary>"
            f"<div class='cols'><div class='col'>"
            + context_block(f"context A (value {p['value_a']})", marked_a, chg_a, p["a"])
            + f"<div class='lbl'>anchor answer A ({esc(lab_a)})</div>"
            + answer_block(row_a["text"], "unpatched anchor rollout")
            + "</div><div class='col'>"
            + context_block(f"context B (value {p['value_b']})", marked_b, chg_b, p["b"])
            + f"<div class='lbl'>anchor answer B ({esc(lab_b)})</div>"
            + answer_block(row_b["text"], "unpatched anchor rollout")
            + "</div><div class='col'><div class='lbl'>carrier text "
            f"(<span class='mono'>{esc(p['carrier'])}</span>)</div>"
            f"<pre>{esc(cells[cell]['carriers'][p['carrier']]['text'])}</pre>"
            "</div></div></details>"
        )

    parts.append(footer(git_commit))
    parts.append("</body></html>")
    return "\n".join(parts)


# ── Result 0 gallery ──────────────────────────────────────────────────


def _pair_group_html(
    p: dict,
    bank: dict,
    grid: dict[str, dict[int, dict]],
    anchors: dict[str, dict[int, dict]],
    coh_grid: dict[tuple[str, int], float],
    coh_anchor: dict[tuple[str, int], float],
    f_cells: dict[str, dict],
    order: int,
    rev_pair: dict | None,
) -> str:
    contexts = bank["contexts"]
    frow = f_cells.get(p["pair_id"])
    f_beh = frow["f_beh"] if frow else None
    f_act = frow["f_act"] if frow else None
    ctx_a, ctx_b = contexts[p["a"]], contexts[p["b"]]
    marked_a, marked_b, chg_a, chg_b = diff_mark(render_context(ctx_a), render_context(ctx_b))
    row_a, lab_a = first_coherent(anchors[p["a"]], coh_anchor, lambda r, d: (r["context_id"], d))
    row_b, lab_b = first_coherent(anchors[p["b"]], coh_anchor, lambda r, d: (r["context_id"], d))

    head_bits = [
        f"<span class='mono'>{esc(p['pair_id'])}</span>",
        f"carrier {esc(p['carrier'])}",
        f"{esc(p['value_a'])} → {esc(p['value_b'])}",
        f"F_beh <span class='f {f_class(f_beh)}'>{fmt_f(f_beh)}</span>",
        f"F_act <span class='f {f_class(f_act)}'>{fmt_f(f_act)}</span>",
    ]
    if frow:
        head_bits.append(
            f"coherent {frow['n_coherent']}/{frow['n_draws']}, "
            f"anchor separation {frow['separation']:.2f}"
        )
    else:
        head_bits.append("F not reported for this pair (no f_cells row)")

    cols = [
        "<div class='col'>"
        + context_block(f"context A (value {p['value_a']})", marked_a, chg_a, p["a"])
        + f"<div class='lbl'>anchor answer A ({esc(lab_a)})</div>"
        + answer_block(row_a["text"], "unpatched anchor rollout")
        + "</div>",
        "<div class='col'>"
        + context_block(f"context B (value {p['value_b']})", marked_b, chg_b, p["b"])
        + f"<div class='lbl'>anchor answer B ({esc(lab_b)})</div>"
        + answer_block(row_b["text"], "unpatched anchor rollout")
        + "</div>",
    ]
    pdraws = grid.get(p["pair_id"])
    if pdraws:
        prow, plab = first_coherent(pdraws, coh_grid, lambda r, d: (r["pair_id"], d))
        cols.append(
            "<div class='col'><div class='lbl'>A patched with B's context-end state "
            f"({esc(plab)})</div>"
            + answer_block(prow["text"], "patched rollout, steered arm")
            + "</div>"
        )
    else:
        cols.append(
            "<div class='col'><div class='lbl'>A patched with B's context-end state</div>"
            "<div class='d'>no patched rollouts banked for this pair at (context-end, steered)"
            "</div></div>"
        )

    rev_html = ""
    if rev_pair is not None:
        rev_frow = f_cells.get(rev_pair["pair_id"])
        rf_beh = rev_frow["f_beh"] if rev_frow else None
        rf_act = rev_frow["f_act"] if rev_frow else None
        rev_cols = []
        rdraws = grid.get(rev_pair["pair_id"])
        if rdraws:
            rrow, rlab = first_coherent(rdraws, coh_grid, lambda r, d: (r["pair_id"], d))
            rev_cols.append(
                f"<div class='lbl'>B patched with A's context-end state ({esc(rlab)})"
                "</div>" + answer_block(rrow["text"], "patched rollout, steered arm")
            )
        rev_html = (
            "<details><summary>reverse direction — banked as "
            f"<span class='mono'>{esc(rev_pair['pair_id'])}</span> in cell "
            f"<span class='mono'>{esc(rev_pair['cell'])}</span>: F_beh "
            f"<span class='f {f_class(rf_beh)}'>{fmt_f(rf_beh)}</span>, F_act "
            f"<span class='f {f_class(rf_act)}'>{fmt_f(rf_act)}</span></summary>"
            + "".join(rev_cols)
            + "</details>"
        )
    else:
        rev_html = "<div class='d'>reverse direction not banked</div>"

    fbeh_attr = "NaN" if f_beh is None else f"{f_beh:.6f}"
    return (
        f"<div class='pairgroup' data-fbeh='{fbeh_attr}' data-order='{order}'>"
        f"<div class='pairhead'>{' | '.join(head_bits)}</div>"
        f"<div class='cols'>{''.join(cols)}</div>{rev_html}</div>"
    )


def build_gallery(
    bank: dict,
    grid: dict[str, dict[int, dict]],
    anchors: dict[str, dict[int, dict]],
    coh_grid: dict[tuple[str, int], float],
    coh_anchor: dict[tuple[str, int], float],
    f_cells: dict[str, dict],
    two_by_two: dict[str, dict],
    analysis_md: str,
    grid_meta: dict,
    anchor_meta: dict,
    git_commit: str,
) -> str:
    cells: dict[str, dict] = bank["cells"]
    pairs: list[dict] = bank["pairs"]
    by_cell: dict[str, list[dict]] = {}
    for p in pairs:
        by_cell.setdefault(p["cell"], []).append(p)
    by_edge = {(p["a"], p["b"]): p for p in pairs}

    parts: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>#2162 Result 0 — qualitative gallery</title>",
        f"<style>{CSS}</style><script>{SORT_JS}</script></head><body>",
        "<h1>#2162 Result 0 — qualitative examples (context-end patch, steered arm)</h1>",
        f"<p class='lede'>{provenance_line(bank, grid_meta, anchor_meta)}</p>",
        "<details class='interp'><summary>Fable 5 analysis — what can and can't transfer "
        "(interpretation)</summary>" + render_markdown_min(analysis_md) + "</details>",
        "<p class='lede'>Everything below is raw banked data. One section per parameter cell; "
        "rows are directed minimal pairs at (context-end, steered). Per-pair F_beh / F_act from "
        "<span class='mono'>eval_results/issue_2162/f_metrics/f_cells.jsonl</span>; per-cell 2×2 "
        "verdicts from <span class='mono'>two_by_two.json</span>.</p>",
        "<p>Sort sections by median pair F_beh: "
        "<button class='sort' onclick=\"sortSections('best')\">best first</button>"
        "<button class='sort' onclick=\"sortSections('worst')\">worst first</button>"
        "<button class='sort' onclick=\"sortSections('bank')\">bank order</button></p>",
        "<div id='sections'>",
    ]

    for order, cell in enumerate(cells):
        cell_pairs = by_cell[cell]
        fb = [f_cells[p["pair_id"]]["f_beh"] for p in cell_pairs if p["pair_id"] in f_cells]
        fa = [f_cells[p["pair_id"]]["f_act"] for p in cell_pairs if p["pair_id"] in f_cells]
        fb = [v for v in fb if v is not None]
        fa = [v for v in fa if v is not None]
        med_fb = statistics.median(fb) if fb else None
        med_fa = statistics.median(fa) if fa else None
        tt = two_by_two.get(cell)
        if tt:
            verdict = f"probe {tt['probe_verdict']} / causal {tt['causal_verdict']}"
        else:
            verdict = "no 2×2 verdict (cell reports no F by design)"
        head = (
            f"{esc(cell)} — {esc(verdict)} | median pair F_beh "
            f"<span class='f {f_class(med_fb)}'>{fmt_f(med_fb)}</span>, median F_act "
            f"<span class='f {f_class(med_fa)}'>{fmt_f(med_fa)}</span> "
            f"({len(cell_pairs)} directed pairs)"
        )
        fbeh_attr = "NaN" if med_fb is None else f"{med_fb:.6f}"
        rows_html = []
        for i, p in enumerate(cell_pairs):
            rev = by_edge.get((p["b"], p["a"]))
            rows_html.append(
                _pair_group_html(p, bank, grid, anchors, coh_grid, coh_anchor, f_cells, i, rev)
            )
        parts.append(
            f"<details class='cell' data-fbeh='{fbeh_attr}' data-order='{order}'>"
            f"<summary>{head}</summary>"
            "<p>Sort pairs by F_beh: "
            "<button class='sort' onclick=\"sortRows(this,'best')\">best first</button>"
            "<button class='sort' onclick=\"sortRows(this,'worst')\">worst first</button>"
            "<button class='sort' onclick=\"sortRows(this,'bank')\">bank order</button></p>"
            f"<div class='rows'>{''.join(rows_html)}</div></details>"
        )

    parts.append("</div>")
    parts.append(footer(git_commit))
    parts.append("</body></html>")
    return "\n".join(parts)


# ── main ──────────────────────────────────────────────────────────────


def _git_commit() -> str:
    import subprocess

    proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else "unavailable-no-git-checkout"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dash-dl",
        type=Path,
        default=Path("/mnt/eps-data/thomasjiralerspong/issue2162_mapshift/dash_dl"),
        help="staging root mirroring the HF repo layout (pinned revision)",
    )
    ap.add_argument("--f-metrics", type=Path, default=Path("eval_results/issue_2162/f_metrics"))
    ap.add_argument(
        "--analysis-md",
        type=Path,
        default=Path("/tmp/issue2162_fable5_transfer_analysis.md"),
        help="Fable 5 interpretation markdown rendered into the gallery's top box",
    )
    ap.add_argument("--out-bank", type=Path, default=Path("docs/issue2162_bank_dashboard.html"))
    ap.add_argument("--out-gallery", type=Path, default=Path("docs/issue2162_result0_gallery.html"))
    args = ap.parse_args()

    assert args.analysis_md.is_file(), f"missing interpretation markdown: {args.analysis_md}"
    analysis_md = args.analysis_md.read_text(encoding="utf-8")

    bank = load_bank(args.dash_dl)
    cells = list(bank["cells"])
    grid = load_grid_ce_steered(args.dash_dl, cells)
    anchors = load_anchor_rows(args.dash_dl)
    coh_grid, coh_anchor = load_coherence(args.dash_dl)
    f_cells = load_f_cells(args.f_metrics)
    two = load_two_by_two(args.f_metrics)
    grid_meta = _decoding_meta(grid)
    anchor_meta = _decoding_meta(anchors)
    git_commit = _git_commit()

    bank_html = build_bank_dashboard(bank, anchors, coh_anchor, grid_meta, anchor_meta, git_commit)
    args.out_bank.write_text(bank_html, encoding="utf-8")
    print(f"[bank] wrote {args.out_bank} ({len(bank_html) / 1e6:.2f} MB)")

    gallery_html = build_gallery(
        bank,
        grid,
        anchors,
        coh_grid,
        coh_anchor,
        f_cells,
        two,
        analysis_md,
        grid_meta,
        anchor_meta,
        git_commit,
    )
    args.out_gallery.write_text(gallery_html, encoding="utf-8")
    print(f"[gallery] wrote {args.out_gallery} ({len(gallery_html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
