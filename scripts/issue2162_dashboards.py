#!/usr/bin/env python3
"""Render the #2162 bank dashboard + Result 0 qualitative gallery as static HTML (v2).

Two self-contained artifacts for the `mapshift` inline round (consolidation plan
`docs/reports/issue_2162_consolidation_plan.md` § "Bank dashboard" + § "Result 0"),
revised after user review of v1 ("too many details; unclear what the context is
for context A and context B"):

- every per-answer provenance label, pair/value/cell id code, and verdict tag is
  gone — ONE small footer line per page carries all provenance;
- each gallery row renders the ACTUAL shared context ONCE, with the varied span
  shown inline at its exact position as ``A: <mark>…</mark> → B: <mark>…</mark>``;
  system prompt and final query are always visible, long histories collapse to
  the last 2 turns behind an expand control;
- answers are labeled only "answer with A" / "answer with B" /
  "A patched toward B" (+ "B patched toward A" where the reverse direction is
  banked), 300-char heads with the remainder behind an expand control;
- section headers are a plain-English parameter name + one factual line on what
  the parameter is and where it lives, + median F_beh / F_act;
- repeated identical context parts are deduplicated into a JS map (embedded
  once, hydrated at render time) to keep the gallery under ~15 MB;
- output is written with LF line endings only (banked CRLF normalized on load).

Inputs (all banked; HF paths staged at ONE pinned revision):
  - ``issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json``
  - ``issue2162_ctxinfo/raw_completions/grid/shard_<cell>__ce__steered.jsonl``
  - ``issue2162_ctxinfo/raw_completions/anchors/anchors_*.jsonl``
  - ``issue2162_ctxinfo/raw_completions/judge_raw/scores/coherence.{grid,anchors}
    .scores.jsonl`` (coherent = score > 60, the ``issue2162_analysis`` convention)
  - ``eval_results/issue_2162/f_metrics/f_cells.jsonl`` (per-pair F_beh + F_act)

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
ANSWER_HEAD_CHARS = 300
HEAD_SEG_CHARS = 60  # per-side truncation of the "what changed" header segments
DEDUP_MIN_CHARS = 160  # shared context parts at least this long go through the JS map
COALESCE_GAP = 24  # equal runs shorter than this between two diffs merge into one swap

CSS = """
:root { color-scheme: light dark; }
body { font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       margin: 24px auto; max-width: 1700px; padding: 0 16px; color: #1a1a1a; background: #fff; }
h1 { font-size: 20px; margin: 0 0 6px; }
h2 { font-size: 16px; margin: 18px 0 6px; }
p.lede { margin: 0 0 6px; color: #444; max-width: 120ch; }
span.mono { background: #f2f2f2; padding: 1px 4px; border-radius: 3px;
            font-family: ui-monospace, monospace; font-size: 12px; }
table { border-collapse: collapse; width: 100%; margin-top: 10px; }
th, td { border: 1px solid #ddd; padding: 6px 8px; vertical-align: top; text-align: left; }
th { background: #f6f6f6; font-size: 12px; letter-spacing: .02em; }
mark { color: inherit; padding: 0 1px; border-radius: 2px; }
mark.a { background: #ffd9d0; } mark.b { background: #d4f2cf; }
span.arr { color: #888; font-weight: 700; }
span.none { color: #999; font-style: italic; }
.f { font-weight: 700; }
.hi { color: #1b6d2f; } .mid { color: #9a6400; } .lo { color: #a11; } .na { color: #777; }
.d { font-size: 11.5px; color: #666; font-weight: 400; }
details { margin-top: 4px; }
summary { cursor: pointer; color: #06c; font-size: 12px; }
details.cell { border: 1px solid #ddd; border-radius: 6px; margin: 10px 0; padding: 6px 10px; }
details.cell > summary { font-size: 14px; color: inherit; font-weight: 400; }
details.interp { border: 2px solid #9a6400; border-radius: 6px; margin: 12px 0; padding: 6px 10px; }
details.interp > summary { font-size: 14px; color: #9a6400; font-weight: 700; }
pre { white-space: pre-wrap; background: #fafafa; border: 1px solid #eee; border-radius: 4px;
      padding: 8px; margin: 6px 0 0; font-size: 11.5px; max-height: 380px; overflow: auto; }
div.ctx { white-space: pre-wrap; background: #fafafa; border: 1px solid #eee; border-radius: 4px;
          padding: 8px; margin: 6px 0; font-size: 12.5px; }
div.turn { margin: 3px 0; }
span.rl { font-weight: 700; color: #555; }
div.pairgroup { border-top: 1px solid #e5e5e5; padding: 8px 0; }
div.pairhead { font-size: 13px; margin-bottom: 2px; }
div.cols { display: grid; gap: 10px; }
div.cols.c2 { grid-template-columns: 1fr 1fr; }
div.cols.c3 { grid-template-columns: 1fr 1fr 1fr; }
div.cols.c4 { grid-template-columns: 1fr 1fr 1fr 1fr; }
div.col { min-width: 0; font-size: 12.5px; }
div.col .lbl { font-weight: 600; font-size: 12px; margin-bottom: 2px; }
div.crow { border-top: 1px solid #eee; padding: 4px 0; font-size: 12.5px; }
button.sort { font-size: 12px; margin-right: 6px; cursor: pointer; }
p.foot { font-size: 11.5px; color: #666; margin-top: 18px; border-top: 1px solid #ddd;
         padding-top: 8px; }
@media (prefers-color-scheme: dark) {
  body { background: #121212; color: #e6e6e6; }
  th { background: #1e1e1e; }
  th, td, details.cell, div.pairgroup, div.crow, p.foot { border-color: #333; }
  p.lede, .d, p.foot { color: #bbb; } span.mono { background: #222; }
  pre, div.ctx { background: #191919; border-color: #2a2a2a; }
  mark.a { background: #6b2f24; } mark.b { background: #2b4d26; }
  span.rl { color: #aaa; }
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

# Plain-English name + one factual "what it is and where it lives" line per base type.
# Value summaries paraphrase bank2162.VALUES; the bank dashboard shows the strings verbatim.
PLAIN_DESC: dict[str, tuple[str, str]] = {
    "instr_format": (
        "answer-format instruction",
        "an instruction in the system prompt demanding a specific answer format "
        "(bullet points / flowing prose / numbered steps)",
    ),
    "instr_language": (
        "reply-language instruction",
        "an instruction in the system prompt demanding a specific reply language "
        "(Spanish / English / French)",
    ),
    "constraint_knowledge": (
        "knowledge-access rule",
        "a system-prompt statement of what knowledge the assistant may use "
        "(no internet / live browsing / internal knowledge only)",
    ),
    "refusal_boundary": (
        "medical-advice boundary",
        "a system-prompt clause on handling medical questions "
        "(no clause / answer with a disclaimer / must decline)",
    ),
    "verbosity": (
        "answer-length instruction",
        "a system-prompt instruction setting answer length "
        "(very short / moderate / exhaustive detail)",
    ),
    "reasoning_style": (
        "reasoning-style instruction",
        "a system-prompt instruction setting how to reason "
        "(step by step / conclusion only / pros and cons)",
    ),
    "persona_prompted": (
        "assistant persona (system prompt)",
        "the assistant persona set in the system prompt "
        "(pirate captain / no persona / Victorian butler)",
    ),
    "demo_format": (
        "format demonstration",
        "an earlier exchange whose answer demonstrates a format "
        "(bullets / prose / numbered steps); nothing is instructed",
    ),
    "demo_persona": (
        "persona demonstration",
        "an earlier exchange whose answer demonstrates a persona voice "
        "(pirate / plain / Victorian butler); nothing is instructed",
    ),
    "language_implied": (
        "conversation language (implied)",
        "the language of the earlier exchange and of the final query "
        "(Spanish / English / French); no instruction mentions language",
    ),
    "persona_role_header": (
        "assistant role name",
        "the role name that opens the assistant's own turn in the chat template "
        "(pirate_assistant / assistant / butler_assistant)",
    ),
    "fact_user_name": (
        "user's name",
        "the user's name, stated earlier in the conversation (Alice / Bob / Priya)",
    ),
    "fact_assistant_animal": (
        "assistant's favorite animal",
        "the assistant's favorite animal, stated earlier in the conversation "
        "(octopus / falcon / axolotl)",
    ),
    "fact_novel_queried": (
        "stated fact the query asks about",
        "a made-up fact (a year) stated earlier that the final query asks about "
        "(1847 / 1902 / 1763)",
    ),
    "list_numeric_detail": (
        "numeric detail in a list",
        "one number inside a list the user gave earlier (6 roses / 2 lanterns / 9 spoons)",
    ),
    "icl_task_mapping": (
        "in-context task (examples only)",
        "the task implied by input→output examples (antonyms / synonyms / translate to "
        "Spanish); never named in words",
    ),
    "user_expertise": (
        "user's stated expertise",
        "how the user describes their own level (five-year-old / professor / hobbyist)",
    ),
    "user_emotion": (
        "user's stated mood",
        "the feeling the user mentions being in (stressed / excited / frustrated)",
    ),
    "prior_topic": (
        "previous topic",
        "what the earlier, unrelated exchange was about "
        "(a birthday / a server outage / a hiking trip)",
    ),
    "query_content": (
        "the query itself",
        "the final user question is what varies (three different real-user questions "
        "on the same carrier prefix)",
    ),
    "filler_swap": (
        "filler sentence (control)",
        "a content-free filler sentence in the prefix (weather / fence / library); "
        "a no-signal control",
    ),
}


def cell_plain(cell: str, base_type: str) -> tuple[str, str]:
    """(plain-English parameter name, one factual what-it-is line) for a cell.

    Recency cells append ``depth - 1`` unrelated exchanges after the varied
    content (bank2162._padding_history); load cells state the varied fact among
    ``load`` remembered items in one message (bank2162._load_system_or_history).
    """
    if cell.startswith("conflict_format"):
        name = "format: instruction vs demonstration disagree"
        desc = (
            "the system prompt instructs one answer format while an earlier exchange "
            "demonstrates a different one; A and B change both sides at once"
        )
    elif cell.startswith("conflict_persona"):
        name = "persona: instruction vs demonstration disagree"
        desc = (
            "the system prompt sets one assistant persona while an earlier exchange "
            "demonstrates a different one; A and B change both sides at once"
        )
    else:
        name, desc = PLAIN_DESC[base_type]
        if cell.startswith("recency_"):
            pad = int(cell.rsplit("_d", 1)[1]) - 1
            name = f"{name}, buried under {pad} later exchanges"
            desc = f"{desc}; here {pad} unrelated exchanges follow it, pushing it into the past"
        elif cell.startswith("load_"):
            load = int(cell.rsplit("_l", 1)[1])
            name = f"{name}, among {load} items"
            desc = f"{desc}; here it is stated among {load} things to remember in one message"
        return name, desc
    if cell.endswith("_rev"):
        name += " (reverse direction)"
    return name, desc


def normalize(text: str) -> str:
    """LF-only text: banked WildChat carriers/completions contain CRLF (\\r\\n)."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def deep_normalize(obj):
    """Recursively normalize every string in a JSON-shaped object to LF line endings."""
    if isinstance(obj, str):
        return normalize(obj)
    if isinstance(obj, list):
        return [deep_normalize(x) for x in obj]
    if isinstance(obj, dict):
        return {k: deep_normalize(v) for k, v in obj.items()}
    return obj


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
    bank = deep_normalize(
        json.loads((dash_dl / "issue2162_ctxinfo/analysis_tensors/vc_bank/bank.json").read_text())
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
            row["text"] = normalize(row["text"])
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
            row["text"] = normalize(row["text"])
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


# ── shared-context rendering (one block, inline A→B swap) ────────────


def aligned_parts(ctx_a: dict, ctx_b: dict) -> list[tuple[str, str, str, str]]:
    """[(label, text_a, text_b, kind)]; kind in {system, turn, query, role}.

    The two contexts of a pair are structurally identical except that a system
    prompt or a role header may be present on only one side (persona cells,
    verified over all 1,404 pairs); the absent side aligns as empty/default.
    """
    parts: list[tuple[str, str, str, str]] = []
    sys_a, sys_b = ctx_a.get("system") or "", ctx_b.get("system") or ""
    if sys_a or sys_b:
        parts.append(("system prompt", sys_a, sys_b, "system"))
    hist_a, hist_b = ctx_a.get("history") or [], ctx_b.get("history") or []
    assert len(hist_a) == len(hist_b), (ctx_a["id"], ctx_b["id"])
    for turn_a, turn_b in zip(hist_a, hist_b):
        assert turn_a["role"] == turn_b["role"], (ctx_a["id"], ctx_b["id"])
        parts.append((turn_a["role"], turn_a["content"], turn_b["content"], "turn"))
    parts.append(("user (final query)", ctx_a["user"], ctx_b["user"], "query"))
    rh_a, rh_b = ctx_a.get("role_header") or "", ctx_b.get("role_header") or ""
    if rh_a or rh_b:
        parts.append(("assistant role name", rh_a or "assistant", rh_b or "assistant", "role"))
    return parts


def _expand_span(text: str, start: int, end: int) -> tuple[int, int]:
    """Expand [start, end) outward to the nearest whitespace/string boundary."""
    while start > 0 and not text[start - 1].isspace():
        start -= 1
    while end < len(text) and not text[end].isspace():
        end += 1
    return start, end


def diff_regions(a: str, b: str) -> list[tuple[str, str, str]]:
    """[(op, seg_a, seg_b)] with op in {equal, diff}.

    Short equal runs between two diffs (< COALESCE_GAP chars) fold into one
    swap unit, and every diff region then expands outward to whitespace
    boundaries on BOTH sides so marks cover whole words ("large" vs "grande",
    never the raw char-level "larg" vs "grand"); overlaps created by the
    expansion re-merge. Concatenating seg_a / seg_b reproduces a / b exactly.
    """
    raw = difflib.SequenceMatcher(a=a, b=b, autojunk=False).get_opcodes()
    folded: list[list] = []  # [kind, i1, i2, j1, j2]
    for idx, (op, i1, i2, j1, j2) in enumerate(raw):
        kind = "equal" if op == "equal" else "diff"
        if (
            kind == "equal"
            and 0 < idx < len(raw) - 1
            and i2 - i1 < COALESCE_GAP
            and folded
            and folded[-1][0] == "diff"
            and raw[idx + 1][0] != "equal"
        ):
            kind = "diff"
        if folded and folded[-1][0] == kind:
            folded[-1][2], folded[-1][4] = i2, j2
        else:
            folded.append([kind, i1, i2, j1, j2])
    diffs: list[list[int]] = []
    for kind, i1, i2, j1, j2 in folded:
        if kind != "diff":
            continue
        i1, i2 = _expand_span(a, i1, i2)
        j1, j2 = _expand_span(b, j1, j2)
        if diffs and (i1 <= diffs[-1][1] or j1 <= diffs[-1][3]):
            diffs[-1][1] = max(diffs[-1][1], i2)
            diffs[-1][3] = max(diffs[-1][3], j2)
        else:
            diffs.append([i1, i2, j1, j2])
    out: list[tuple[str, str, str]] = []
    pos_a = pos_b = 0
    for i1, i2, j1, j2 in diffs:
        if i1 > pos_a or j1 > pos_b:
            out.append(("equal", a[pos_a:i1], b[pos_b:j1]))
        out.append(("diff", a[i1:i2], b[j1:j2]))
        pos_a, pos_b = i2, j2
    if pos_a < len(a) or pos_b < len(b):
        out.append(("equal", a[pos_a:], b[pos_b:]))
    return out


def _self_check_word_boundary_marks() -> None:
    """Build-time check: a varied-span mark must cover WHOLE words ("large" vs
    "grande"), never shared-character fragments ("larg" vs "grand"). Fails loud."""
    regs = diff_regions("The answer is large.", "The answer is grande.")
    diffs = [(seg_a, seg_b) for kind, seg_a, seg_b in regs if kind == "diff"]
    assert diffs == [("large.", "grande.")], diffs
    assert "".join(seg_a for _, seg_a, _ in regs) == "The answer is large."
    assert "".join(seg_b for _, _, seg_b in regs) == "The answer is grande."


def render_swap(regions: list[tuple[str, str, str]]) -> str:
    """Escaped text with each differing region shown inline as A: … → B: …."""
    out: list[str] = []
    for kind, seg_a, seg_b in regions:
        if kind == "equal":
            out.append(esc(seg_a))
        else:
            mark_a = f"<mark class='a'>{esc(seg_a)}</mark>" if seg_a else _NONE
            mark_b = f"<mark class='b'>{esc(seg_b)}</mark>" if seg_b else _NONE
            out.append(f"A: {mark_a} <span class='arr'>→</span> B: {mark_b}")
    return "".join(out)


_NONE = "<span class='none'>(nothing)</span>"


def _dedupe_or_inline(text: str, registry: dict[str, str] | None) -> str:
    """Long shared parts embed once (JS map, hydrated at render); short ones inline."""
    if registry is None or len(text) < DEDUP_MIN_CHARS:
        return esc(text)
    key = registry.setdefault(text, f"t{len(registry)}")
    return f"<span class='hy' data-k='{key}'></span>"


def context_html(
    ctx_a: dict, ctx_b: dict, registry: dict[str, str] | None
) -> tuple[str, list[tuple[str, str]]]:
    """One shared-context block with the varied span(s) swapped inline.

    Returns (html, changed) where changed = [(seg_a, seg_b)] differing regions
    in document order (feeds the row's "what changed" header). History turns
    with no diff collapse behind an expand control except the last 2; the
    system prompt and the final query never collapse.
    """
    parts = aligned_parts(ctx_a, ctx_b)
    n_hist = sum(1 for p in parts if p[3] == "turn")
    rendered: list[tuple[bool, str]] = []  # (collapsible, html)
    changed: list[tuple[str, str]] = []
    hist_idx = 0
    for label, text_a, text_b, kind in parts:
        if text_a == text_b:
            body = _dedupe_or_inline(text_a, registry)
            is_diff = False
        else:
            regs = [("diff", text_a, text_b)] if kind == "role" else diff_regions(text_a, text_b)
            body = render_swap(regs)
            changed += [(sa, sb) for k, sa, sb in regs if k == "diff"]
            is_diff = True
        row = f"<div class='turn'><span class='rl'>{esc(label)}:</span> {body}</div>"
        collapsible = kind == "turn" and not is_diff and hist_idx < n_hist - 2
        rendered.append((collapsible, row))
        if kind == "turn":
            hist_idx += 1
    out: list[str] = []
    run: list[str] = []

    def flush() -> None:
        if run:
            plural = "s" if len(run) > 1 else ""
            out.append(
                f"<details><summary>{len(run)} earlier turn{plural}</summary>{''.join(run)}"
                "</details>"
            )
            run.clear()

    for collapsible, row in rendered:
        if collapsible:
            run.append(row)
        else:
            flush()
            out.append(row)
    flush()
    return f"<div class='ctx'>{''.join(out)}</div>", changed


def hydrate_block(registry: dict[str, str]) -> str:
    """The deduplicated context-part texts + the script that fills them in."""
    inv = {key: text for text, key in registry.items()}
    payload = json.dumps(inv, ensure_ascii=False).replace("</", "<\\/")
    return (
        f"<script id='ctxmap' type='application/json'>{payload}</script>"
        "<script>var _m=JSON.parse(document.getElementById('ctxmap').textContent);"
        "document.querySelectorAll('span.hy').forEach(function(el){"
        "el.textContent=_m[el.dataset.k];});</script>"
    )


# ── answers + row/section chrome ──────────────────────────────────────


def first_coherent(draws: dict[int, dict], coh: dict, key_of) -> tuple[dict, str]:
    """First draw (ascending index) with judge coherence > 60; else the first
    draw with a factual inline note (never imputes coherence)."""
    assert draws, "no draws"
    for d in sorted(draws):
        score = coh.get(key_of(draws[d], d))
        if score is not None and score > COHERENCE_THRESHOLD:
            return draws[d], ""
    d = min(draws)
    return draws[d], " <span class='d'>(no coherent draw; first draw shown)</span>"


def answer_block(text: str, label: str, note: str = "") -> str:
    """One labeled answer column: 300-char head, remainder behind an expand
    control (visible head + expanded remainder = the full stored completion)."""
    lbl = f"<div class='lbl'>{esc(label)}{note}</div>"
    if len(text) <= ANSWER_HEAD_CHARS:
        return f"<div class='col'>{lbl}<pre>{esc(text)}</pre></div>"
    head, rest = text[:ANSWER_HEAD_CHARS], text[ANSWER_HEAD_CHARS:]
    return (
        f"<div class='col'>{lbl}<pre>{esc(head)}…</pre>"
        f"<details><summary>show the remaining {len(rest)} characters</summary>"
        f"<pre>…{esc(rest)}</pre></details></div>"
    )


def f_class(value: float | None) -> str:
    if value is None:
        return "na"
    return "hi" if value >= 0.65 else ("mid" if value > 0.05 else "lo")


def fmt_f(value: float | None) -> str:
    return "—" if value is None else f"{value:.3f}"


def f_span(value: float | None) -> str:
    return f"<span class='f {f_class(value)}'>{fmt_f(value)}</span>"


def _short_seg(seg: str) -> str:
    seg = " ".join(seg.split())
    if len(seg) > HEAD_SEG_CHARS:
        seg = seg[:HEAD_SEG_CHARS] + "…"
    return seg


def pair_header(
    plain_name: str, changed: list[tuple[str, str]], f_beh: float | None, f_act: float | None
) -> str:
    """One line: what changed in plain words (A → B) + the pair's F_beh / F_act."""
    what_bits = []
    for seg_a, seg_b in changed[:2]:
        mark_a = f"<mark class='a'>{esc(_short_seg(seg_a))}</mark>" if seg_a.strip() else _NONE
        mark_b = f"<mark class='b'>{esc(_short_seg(seg_b))}</mark>" if seg_b.strip() else _NONE
        what_bits.append(f"{mark_a} <span class='arr'>→</span> {mark_b}")
    what = " … ".join(what_bits) if what_bits else "<span class='none'>(no text change)</span>"
    return (
        f"<div class='pairhead'><b>{esc(plain_name)}:</b> {what} &nbsp; "
        f"F_beh {f_span(f_beh)} · F_act {f_span(f_act)}</div>"
    )


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


def footer(bank: dict, git_commit: str) -> str:
    """The ONE provenance line per page (everything the per-item labels used to say)."""
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
    return (
        f"<p class='foot'>Model {esc(bank['model_id'])} · bank seed {bank['seed']} (frozen) · "
        "sampling temperature 1.0; the answer shown is the first draw judged coherent "
        "(judge coherence &gt; 60) · patch = full-state replacement of the context-end "
        "activation (all 28 layers) with the other context's · F_beh / F_act = the fraction "
        "of the full context-swap effect the patch carries, at judged-behavior / "
        "answer-vector level (0 = patch does nothing, 1 = as good as swapping the whole "
        "context; eval_results/issue_2162/f_metrics/f_cells.jsonl) · answers over "
        f"{ANSWER_HEAD_CHARS} characters show the head with the remainder behind the expand "
        "control (head + remainder = the full stored completion; line endings normalized "
        f"to LF) · data pinned to HF dataset {HF_REPO} revision {PINNED_REVISION[:12]} · "
        f"generated {ts} by scripts/issue2162_dashboards.py at commit {esc(git_commit[:12])}."
        "</p>"
    )


# ── bank dashboard ────────────────────────────────────────────────────


def _fmt_value(value: str) -> str:
    """Verbatim value string; conflict composites reformatted as plain words."""
    if value.startswith("instr=") and "|demo=" in value:
        instr, demo = value[len("instr=") :].split("|demo=", 1)
        return f"instruction: {instr} + demonstration: {demo}"
    return value if value else "(none)"


def build_bank_dashboard(
    bank: dict,
    anchors: dict[str, dict[int, dict]],
    coh_anchor: dict[tuple[str, int], float],
    git_commit: str,
) -> str:
    cells: dict[str, dict] = bank["cells"]
    contexts: dict[str, dict] = bank["contexts"]
    first_pair_of_cell: dict[str, dict] = {}
    for p in bank["pairs"]:
        first_pair_of_cell.setdefault(p["cell"], p)

    parts: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>#2162 bank dashboard</title>",
        f"<style>{CSS}</style></head><body>",
        "<h1>#2162 context-information bank — reference</h1>",
        "<p class='lede'>1,404 contexts = 39 varied parameters × 12 carrier conversations × "
        "3 values; the sibling contexts of a pair are identical except the one varied span. "
        "Factual reference only — no scores, no interpretation.</p>",
    ]

    # 1. carriers — one row per distinct carrier text
    carrier_texts: list[str] = []
    seen: set[str] = set()
    for meta in cells.values():
        for c in meta["carriers"].values():
            if c["text"] not in seen:
                seen.add(c["text"])
                carrier_texts.append(c["text"])
    parts.append("<h2>Carrier texts</h2>")
    parts.append(
        f"<p class='lede'>Carrier texts are parameter-specific: {len(carrier_texts)} distinct "
        "texts across the 39 parameters, 12 per parameter. Under leave-one-carrier-out, each "
        "carrier is the held-out fold exactly once.</p>"
    )
    for text in carrier_texts:
        if len(text) <= 120:
            parts.append(f"<div class='crow'>{esc(text)}</div>")
        else:
            parts.append(
                f"<div class='crow'>{esc(text[:120])}…"
                f"<details><summary>full text</summary><pre>{esc(text)}</pre></details></div>"
            )

    # 2. the 39 parameters: plain name, one-line description, value strings verbatim
    parts.append("<h2>Parameters</h2>")
    parts.append("<table><tr><th>parameter</th><th>what it is</th><th>the values</th></tr>")
    for cell, meta in cells.items():
        name, desc = cell_plain(cell, meta["base_type"])
        vals = "".join(f"<div>{esc(_fmt_value(v))}</div>" for v in meta["values"].values())
        parts.append(f"<tr><td><b>{esc(name)}</b></td><td>{esc(desc)}</td><td>{vals}</td></tr>")
    parts.append("</table>")

    # 3. one worked example per parameter: shared context + answer under A vs B
    parts.append("<h2>Worked examples (one per parameter)</h2>")
    for cell, meta in cells.items():
        name, desc = cell_plain(cell, meta["base_type"])
        p = first_pair_of_cell[cell]
        ctx_block, _ = context_html(contexts[p["a"]], contexts[p["b"]], None)
        row_a, note_a = first_coherent(
            anchors[p["a"]], coh_anchor, lambda r, d: (r["context_id"], d)
        )
        row_b, note_b = first_coherent(
            anchors[p["b"]], coh_anchor, lambda r, d: (r["context_id"], d)
        )
        parts.append(
            f"<details class='cell' title='{cell}'><summary><b>{esc(name)}</b> — "
            f"{esc(desc)}</summary>"
            + ctx_block
            + "<div class='cols c2'>"
            + answer_block(row_a["text"], "answer with A", note_a)
            + answer_block(row_b["text"], "answer with B", note_b)
            + "</div></details>"
        )

    parts.append(footer(bank, git_commit))
    parts.append("</body></html>")
    return "\n".join(parts)


# ── Result 0 gallery ──────────────────────────────────────────────────


def _pair_group_html(
    p: dict,
    plain_name: str,
    bank: dict,
    grid: dict[str, dict[int, dict]],
    anchors: dict[str, dict[int, dict]],
    coh_grid: dict[tuple[str, int], float],
    coh_anchor: dict[tuple[str, int], float],
    f_cells: dict[str, dict],
    order: int,
    rev_pair: dict | None,
    registry: dict[str, str],
) -> str:
    contexts = bank["contexts"]
    frow = f_cells.get(p["pair_id"])
    f_beh = frow["f_beh"] if frow else None
    f_act = frow["f_act"] if frow else None

    ctx_block, changed = context_html(contexts[p["a"]], contexts[p["b"]], registry)
    row_a, note_a = first_coherent(anchors[p["a"]], coh_anchor, lambda r, d: (r["context_id"], d))
    row_b, note_b = first_coherent(anchors[p["b"]], coh_anchor, lambda r, d: (r["context_id"], d))

    cols = [
        answer_block(row_a["text"], "answer with A", note_a),
        answer_block(row_b["text"], "answer with B", note_b),
    ]
    pair_draws = grid.get(p["pair_id"])
    if pair_draws:
        patched_row, patched_note = first_coherent(
            pair_draws, coh_grid, lambda r, d: (r["pair_id"], d)
        )
        cols.append(answer_block(patched_row["text"], "A patched toward B", patched_note))
    else:
        cols.append(
            "<div class='col'><div class='lbl'>A patched toward B</div>"
            "<div class='d'>no patched answer banked for this pair</div></div>"
        )
    if rev_pair is not None:
        rev_draws = grid.get(rev_pair["pair_id"])
        rev_frow = f_cells.get(rev_pair["pair_id"])
        rev_f = (
            f" <span class='d'>(F_beh {fmt_f(rev_frow['f_beh'] if rev_frow else None)}"
            f" · F_act {fmt_f(rev_frow['f_act'] if rev_frow else None)})</span>"
        )
        if rev_draws:
            rev_row, rev_note = first_coherent(rev_draws, coh_grid, lambda r, d: (r["pair_id"], d))
            cols.append(answer_block(rev_row["text"], "B patched toward A", rev_f + rev_note))

    fbeh_attr = "NaN" if f_beh is None else f"{f_beh:.6f}"
    return (
        f"<div class='pairgroup' data-fbeh='{fbeh_attr}' data-order='{order}'>"
        + pair_header(plain_name, changed, f_beh, f_act)
        + ctx_block
        + f"<div class='cols c{len(cols)}'>{''.join(cols)}</div></div>"
    )


def build_gallery(
    bank: dict,
    grid: dict[str, dict[int, dict]],
    anchors: dict[str, dict[int, dict]],
    coh_grid: dict[tuple[str, int], float],
    coh_anchor: dict[tuple[str, int], float],
    f_cells: dict[str, dict],
    analysis_md: str,
    git_commit: str,
) -> str:
    cells: dict[str, dict] = bank["cells"]
    by_cell: dict[str, list[dict]] = {}
    for p in bank["pairs"]:
        by_cell.setdefault(p["cell"], []).append(p)
    by_edge = {(p["a"], p["b"]): p for p in bank["pairs"]}
    registry: dict[str, str] = {}

    parts: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>#2162 Result 0 — qualitative gallery</title>",
        f"<style>{CSS}</style><script>{SORT_JS}</script></head><body>",
        "<h1>#2162 Result 0 — qualitative examples</h1>",
        "<p class='lede'>Each row: one pair of contexts that differ by exactly one attribute "
        "(the difference is shown inline as A: <mark class='a'>old</mark> "
        "<span class='arr'>→</span> B: <mark class='b'>new</mark>), the model's answer under "
        "each, and the answer when context A's context-end activation is patched with B's. "
        "F_beh / F_act say how much of the full A→B change in behavior / answer vector the "
        "patch carries (0 = none, 1 = all of it).</p>",
        "<details class='interp'><summary>Fable 5 analysis — what can and can't transfer "
        "(interpretation)</summary>" + render_markdown_min(analysis_md) + "</details>",
        "<p>Sort sections by median pair F_beh: "
        "<button class='sort' onclick=\"sortSections('best')\">best first</button>"
        "<button class='sort' onclick=\"sortSections('worst')\">worst first</button>"
        "<button class='sort' onclick=\"sortSections('bank')\">bank order</button></p>",
        "<div id='sections'>",
    ]

    for order, (cell, meta) in enumerate(cells.items()):
        name, desc = cell_plain(cell, meta["base_type"])
        cell_pairs = by_cell[cell]
        fb = [f_cells[p["pair_id"]]["f_beh"] for p in cell_pairs if p["pair_id"] in f_cells]
        fa = [f_cells[p["pair_id"]]["f_act"] for p in cell_pairs if p["pair_id"] in f_cells]
        fb = [v for v in fb if v is not None]
        fa = [v for v in fa if v is not None]
        med_fb = statistics.median(fb) if fb else None
        med_fa = statistics.median(fa) if fa else None
        if fb or fa:
            f_line = (
                f"median F_beh {f_span(med_fb)} · F_act {f_span(med_fa)} · {len(cell_pairs)} pairs"
            )
        else:
            f_line = f"<span class='d'>F not scored (control cell) · {len(cell_pairs)} pairs</span>"
        rows_html = [
            _pair_group_html(
                p,
                name,
                bank,
                grid,
                anchors,
                coh_grid,
                coh_anchor,
                f_cells,
                i,
                by_edge.get((p["b"], p["a"])),
                registry,
            )
            for i, p in enumerate(cell_pairs)
        ]
        fbeh_attr = "NaN" if med_fb is None else f"{med_fb:.6f}"
        parts.append(
            f"<details class='cell' data-fbeh='{fbeh_attr}' data-order='{order}' title='{cell}'>"
            f"<summary><b>{esc(name)}</b> — {esc(desc)}. &nbsp;{f_line}</summary>"
            "<p>Sort pairs by F_beh: "
            "<button class='sort' onclick=\"sortRows(this,'best')\">best first</button>"
            "<button class='sort' onclick=\"sortRows(this,'worst')\">worst first</button>"
            "<button class='sort' onclick=\"sortRows(this,'bank')\">bank order</button></p>"
            f"<div class='rows'>{''.join(rows_html)}</div></details>"
        )

    parts.append("</div>")
    parts.append(footer(bank, git_commit))
    parts.append(hydrate_block(registry))
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

    _self_check_word_boundary_marks()
    assert args.analysis_md.is_file(), f"missing interpretation markdown: {args.analysis_md}"
    analysis_md = normalize(args.analysis_md.read_text(encoding="utf-8"))

    bank = load_bank(args.dash_dl)
    cells = list(bank["cells"])
    grid = load_grid_ce_steered(args.dash_dl, cells)
    anchors = load_anchor_rows(args.dash_dl)
    coh_grid, coh_anchor = load_coherence(args.dash_dl)
    f_cells = load_f_cells(args.f_metrics)
    temps = {r["temperature"] for d in grid.values() for r in d.values()}
    temps |= {r["temperature"] for d in anchors.values() for r in d.values()}
    assert temps == {1.0}, temps  # the footer states temperature 1.0 as fact
    git_commit = _git_commit()

    bank_html = build_bank_dashboard(bank, anchors, coh_anchor, git_commit)
    with args.out_bank.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(bank_html)
    print(f"[bank] wrote {args.out_bank} ({len(bank_html) / 1e6:.2f} MB)")

    gallery_html = build_gallery(
        bank, grid, anchors, coh_grid, coh_anchor, f_cells, analysis_md, git_commit
    )
    with args.out_gallery.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(gallery_html)
    print(f"[gallery] wrote {args.out_gallery} ({len(gallery_html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
