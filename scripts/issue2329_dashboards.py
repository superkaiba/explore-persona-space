#!/usr/bin/env python3
"""Render the #2329 bank dashboard + Result 0 qualitative gallery as static HTML.

Fork of ``scripts/issue2162_dashboards.py`` (the v2 minimal spec the user
approved — kept EXACTLY: shared context rendered once with the varied span
marked inline as A → B, query always visible, three plainly-labeled answers
per pair, per-pair F scores, ONE provenance footer per page, no per-item
provenance labels, no ids/value codes reader-facing). Fork deltas
(plan §4.6 + divergences 9/10 + the task-body pipeline-ordering directive):

- **Issue-2329 inputs**: staged HF prefix ``issue2329_q35rerun`` (Qwen3.5-9B,
  thinking disabled; 32 layers), bank2329's ``bank.json`` (all 1,404 contexts;
  ``pairs`` = the token-identity SURVIVING pairs only, with per-cell dropped
  counts under ``token_identity`` — divergence 9), grid shard names via the
  run script's block slug.
- **F_act-only mode at generation-complete (divergence 10)**: the dashboards
  are built IMMEDIATELY after generation from raw rollout text + the pod's
  judge-free ``f_cells_actonly.jsonl`` (F_act per pair, coherence-UNfiltered),
  BEFORE any judge output exists — F_beh renders "pending judge", the shown
  answer is the first draw (no coherence scores yet), and sorting falls back
  to F_act. When judge waves land, a re-run in ``full`` mode back-fills F_beh
  + coherence-filtered draw selection (the parent's exact behavior). Mode is
  resolved by ``--f-mode auto`` (full iff ``f_cells.jsonl`` exists).
- **C4 re-seam**: ``--analysis-md`` is OPTIONAL (default None) — v2 agents
  write no interpretation input, so the interpretation box is simply omitted
  when absent; a provided path is still asserted to exist.

Inputs (staged at ONE revision under ``--dash-dl`` mirroring the HF repo):
  - ``issue2329_q35rerun/analysis_tensors/vc_bank/bank.json``
  - ``issue2329_q35rerun/raw_completions/grid/shard_<cell>__ce__steered.jsonl``
  - ``issue2329_q35rerun/raw_completions/anchors/anchors_*.jsonl``
  - full mode: ``issue2329_q35rerun/raw_completions/judge_raw/scores/
    coherence.{grid,anchors}.scores.jsonl`` + local
    ``eval_results/issue_2329/f_metrics/f_cells.jsonl``
  - actonly mode: ``issue2329_q35rerun/analysis_tensors/f_metrics_actonly/
    f_cells_actonly.jsonl`` (or ``--factonly-jsonl``)

Usage::

    uv run python scripts/issue2329_dashboards.py \
        --dash-dl /mnt/eps-data/thomasjiralerspong/issue2329_q35rerun/dash_dl \
        --pinned-revision <hf-revision-sha>
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

HF_PREFIX = "issue2329_q35rerun"
HF_REPO = "superkaiba1/explore-persona-space-data"
N_MODEL_LAYERS = 32  # Qwen3.5-9B (divergence 3); the footer states the patch width
F_ACT_READ_LAYER = 30  # divergence 4 (fraction-of-stack match to the parent's 26/28)
COHERENCE_THRESHOLD = 60.0  # issue2329_analysis.py convention: coherent = score > 60
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
# Value summaries paraphrase bank2162.VALUES (reused verbatim by bank2329); the bank
# dashboard shows the strings verbatim.
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
    content; load cells state the varied fact among ``load`` remembered items
    in one message (bank2162 conventions, reused verbatim by bank2329).
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


def block_slug(block_id: str) -> str:
    """Shard-filename slug for a ``cell|slot|arm`` block id (issue2329_run.py
    convention): ``|`` → ``__``, ``.`` → ``p``."""
    return block_id.replace("|", "__").replace(".", "p")


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
        json.loads((dash_dl / f"{HF_PREFIX}/analysis_tensors/vc_bank/bank.json").read_text())
    )
    assert bank["issue"] == 2329, bank["issue"]
    assert len(bank["contexts"]) == 1404, len(bank["contexts"])
    assert len(bank["cells"]) == 39, len(bank["cells"])
    # bank2329's `pairs` holds ONLY the token-identity-surviving pairs
    # (divergence 9); the drop ledger is `token_identity` + `dropped_pairs`.
    ti = bank["token_identity"]
    assert len(bank["pairs"]) == ti["n_intact"], (len(bank["pairs"]), ti["n_intact"])
    return bank


def load_grid_ce_steered(dash_dl: Path, cells: list[str]) -> dict[str, dict[int, dict]]:
    """pair_id -> {draw: row} for the (context-end, steered) arm of every cell."""
    out: dict[str, dict[int, dict]] = {}
    grid_dir = dash_dl / f"{HF_PREFIX}/raw_completions/grid"
    for cell in cells:
        shard = grid_dir / f"shard_{block_slug(f'{cell}|ce|steered')}.jsonl"
        assert shard.is_file(), shard
        for row in iter_jsonl(shard):
            assert row["slot"] == "ce" and row["arm"] == "steered", (row["slot"], row["arm"])
            row["text"] = normalize(row["text"])
            out.setdefault(row["pair_id"], {})[row["draw"]] = row
    assert out, "no grid rows loaded"
    return out


def load_anchor_rows(dash_dl: Path) -> dict[str, dict[int, dict]]:
    """context_id -> {draw: row} over the anchors_gate_* + anchors_rest_* shards.

    Anchors cover ALL 1,404 contexts — dropped pairs' contexts included
    (anchors are per-context, not per-pair)."""
    out: dict[str, dict[int, dict]] = {}
    files = sorted((dash_dl / f"{HF_PREFIX}/raw_completions/anchors").glob("anchors_*.jsonl"))
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
    scores_dir = dash_dl / f"{HF_PREFIX}/raw_completions/judge_raw/scores"
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


def load_f_cells_full(f_metrics: Path) -> dict[str, dict]:
    """pair_id -> {f_beh, f_act} from the analysis-produced f_cells.jsonl (ce, steered).

    The FULL / back-fill mode: judge waves have landed and issue2329_analysis
    has written the coherence-filtered F tables."""
    out: dict[str, dict] = {}
    for row in iter_jsonl(f_metrics / "f_cells.jsonl"):
        if row["slot"] == "ce" and row["arm"] == "steered":
            out[row["pair_id"]] = {"f_beh": row["f_beh"], "f_act": row["f_act"]}
    assert out, "no ce/steered f_cells rows"
    return out


def load_f_cells_actonly(actonly_jsonl: Path) -> dict[str, dict]:
    """pair_id -> {f_beh: None, f_act} from the pod's judge-free
    f_cells_actonly.jsonl (ce, steered; coherence-UNfiltered — divergence 10).

    Row shape (issue2329_run.py `_fact_block_records`): {cell, slot, arm,
    pair_id, n_draws_total, skipped, [n_draws_surviving, f_act_mean] |
    [reason]}. Skipped rows carry no f_act_mean -> f_act None (renders "—")."""
    out: dict[str, dict] = {}
    for row in iter_jsonl(actonly_jsonl):
        if row["slot"] != "ce" or row["arm"] != "steered":
            continue
        f_act = None if row.get("skipped") else row.get("f_act_mean")
        out[row["pair_id"]] = {"f_beh": None, "f_act": f_act}
    assert out, f"no ce/steered rows in {actonly_jsonl}"
    return out


# ── shared-context rendering (one block, inline A→B swap) ────────────


def aligned_parts(ctx_a: dict, ctx_b: dict) -> list[tuple[str, str, str, str]]:
    """[(label, text_a, text_b, kind)]; kind in {system, turn, query, role}.

    The two contexts of a pair are structurally identical except that a system
    prompt or a role header may be present on only one side (persona cells);
    the absent side aligns as empty/default.
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
    draw with a factual inline note (never imputes coherence).

    In actonly mode the coherence dicts are EMPTY (no judge output exists yet
    — divergence 10), so this deterministically returns the first draw; the
    page-level note says so once, so no per-item note is added there."""
    assert draws, "no draws"
    for d in sorted(draws):
        score = coh.get(key_of(draws[d], d))
        if score is not None and score > COHERENCE_THRESHOLD:
            return draws[d], ""
    d = min(draws)
    if not coh:  # actonly mode: judge pending — stated once in the page lede/footer
        return draws[d], ""
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


_PENDING = "<span class='d'>pending judge</span>"


def fbeh_span(value: float | None, actonly: bool) -> str:
    """F_beh renderer: in actonly mode F_beh does not exist YET (judge waves
    pending — divergence 10), which is different from a scored-but-absent
    value; render the distinction."""
    if actonly:
        return _PENDING
    return f_span(value)


def _short_seg(seg: str) -> str:
    seg = " ".join(seg.split())
    if len(seg) > HEAD_SEG_CHARS:
        seg = seg[:HEAD_SEG_CHARS] + "…"
    return seg


def pair_header(
    plain_name: str,
    changed: list[tuple[str, str]],
    f_beh: float | None,
    f_act: float | None,
    actonly: bool,
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
        f"F_beh {fbeh_span(f_beh, actonly)} · F_act {f_span(f_act)}</div>"
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


def footer(bank: dict, git_commit: str, actonly: bool, pinned_revision: str | None) -> str:
    """The ONE provenance line per page (everything the per-item labels used to say)."""
    ts = datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%d %H:%M UTC")
    if actonly:
        answer_pick = (
            "the answer shown is the first draw (judge coherence pending; the gallery is "
            "rebuilt with coherence-filtered draws when judge waves land)"
        )
        f_desc = (
            "F_act = the fraction of the full context-swap effect the patch carries at "
            "answer-vector level, judge-free from captured activation states "
            f"(read layer {F_ACT_READ_LAYER} of {N_MODEL_LAYERS}; coherence-unfiltered; "
            f"{HF_PREFIX}/analysis_tensors/f_metrics_actonly/f_cells_actonly.jsonl) · "
            "F_beh (judged-behavior level) is pending the judge waves"
        )
    else:
        answer_pick = "the answer shown is the first draw judged coherent (judge coherence &gt; 60)"
        f_desc = (
            "F_beh / F_act = the fraction of the full context-swap effect the patch carries, "
            "at judged-behavior / answer-vector level (0 = patch does nothing, 1 = as good as "
            "swapping the whole context; eval_results/issue_2329/f_metrics/f_cells.jsonl)"
        )
    rev = f" revision {esc(pinned_revision[:12])}" if pinned_revision else ""
    ti = bank["token_identity"]
    return (
        f"<p class='foot'>Model {esc(bank['model_id'])} (thinking disabled) · bank seed "
        f"{bank['seed']} (frozen; bank text reused verbatim from #2162, re-tokenized — "
        f"{ti['n_intact']} of {ti['n_intact'] + ti['n_dropped']} pairs are token-identical "
        f"under this tokenizer and only those are shown) · sampling temperature 1.0; "
        f"{answer_pick} · patch = full-state replacement of the context-end activation "
        f"(all {N_MODEL_LAYERS} layers) with the other context's · {f_desc} · answers over "
        f"{ANSWER_HEAD_CHARS} characters show the head with the remainder behind the expand "
        "control (head + remainder = the full stored completion; line endings normalized "
        f"to LF) · data staged from HF dataset {HF_REPO}/{HF_PREFIX}{rev} · "
        f"generated {ts} by scripts/issue2329_dashboards.py at commit {esc(git_commit[:12])}."
        "</p>"
    )


# ── bank dashboard ────────────────────────────────────────────────────


def _fmt_value(value: str) -> str:
    """Verbatim value string; conflict composites reformatted as plain words."""
    if value.startswith("instr=") and "|demo=" in value:
        instr, demo = value[len("instr=") :].split("|demo=", 1)
        return f"instruction: {instr} + demonstration: {demo}"
    return value if value else "(none)"


def _dropped_note(bank: dict, cell: str) -> str:
    """Per-cell dropped-pair note (divergence 9), or '' when the cell is intact.

    ``token_identity.per_cell`` rows are DICTS — the frozen manifest ships
    ``bank2329.build_token_identity``'s per-cell records verbatim
    (``{"n_pairs", "n_intact", "n_dropped", "dropped"}``); read ``n_dropped``
    from the row, never treat the row itself as a count (r2 F1: a truthy
    non-empty dict crashed the ``n > 1`` compare on the first cell rendered).
    """
    row = (bank["token_identity"].get("per_cell") or {}).get(cell) or {}
    n = int(row.get("n_dropped", 0))
    if not n:
        return ""
    plural = "s" if n > 1 else ""
    return (
        f" <span class='d'>({n} pair{plural} not token-identical under this "
        "tokenizer — dropped)</span>"
    )


def build_bank_dashboard(
    bank: dict,
    anchors: dict[str, dict[int, dict]],
    coh_anchor: dict[tuple[str, int], float],
    git_commit: str,
    actonly: bool,
    pinned_revision: str | None,
) -> str:
    cells: dict[str, dict] = bank["cells"]
    contexts: dict[str, dict] = bank["contexts"]
    first_pair_of_cell: dict[str, dict] = {}
    for p in bank["pairs"]:
        first_pair_of_cell.setdefault(p["cell"], p)

    ti = bank["token_identity"]
    parts: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>#2329 bank dashboard</title>",
        f"<style>{CSS}</style></head><body>",
        "<h1>#2329 context-information bank — reference (Qwen3.5-9B rerun)</h1>",
        "<p class='lede'>1,404 contexts = 39 varied parameters × 12 carrier conversations × "
        "3 values, reused verbatim from #2162; the sibling contexts of a pair are identical "
        f"except the one varied span. {ti['n_intact']} of {ti['n_intact'] + ti['n_dropped']} "
        "pairs stay token-identical under the Qwen3.5 tokenizer and only those are analyzed. "
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
        parts.append(
            f"<tr><td><b>{esc(name)}</b>{_dropped_note(bank, cell)}</td>"
            f"<td>{esc(desc)}</td><td>{vals}</td></tr>"
        )
    parts.append("</table>")

    # 3. one worked example per parameter: shared context + answer under A vs B
    parts.append("<h2>Worked examples (one per parameter)</h2>")
    for cell, meta in cells.items():
        name, desc = cell_plain(cell, meta["base_type"])
        p = first_pair_of_cell.get(cell)
        if p is None:
            # Every pair of this cell was dropped by the token-identity gate
            # (divergence 9) — the parameter is documented above; no worked
            # example exists for it in this run.
            parts.append(
                f"<details class='cell' title='{cell}'><summary><b>{esc(name)}</b> — "
                f"{esc(desc)}</summary><p class='d'>no token-identical pair survives for "
                "this parameter under the Qwen3.5 tokenizer — no worked example in this "
                "run</p></details>"
            )
            continue
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

    parts.append(footer(bank, git_commit, actonly, pinned_revision))
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
    actonly: bool,
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
        rev_fbeh = "pending judge" if actonly else fmt_f(rev_frow["f_beh"] if rev_frow else None)
        rev_f = (
            f" <span class='d'>(F_beh {rev_fbeh}"
            f" · F_act {fmt_f(rev_frow['f_act'] if rev_frow else None)})</span>"
        )
        if rev_draws:
            rev_row, rev_note = first_coherent(rev_draws, coh_grid, lambda r, d: (r["pair_id"], d))
            cols.append(answer_block(rev_row["text"], "B patched toward A", rev_f + rev_note))

    # In actonly mode F_beh does not exist yet, so the sort metric — the
    # data attribute the page JS reads — is F_act (button labels say so).
    sort_val = f_act if actonly else f_beh
    sort_attr = "NaN" if sort_val is None else f"{sort_val:.6f}"
    return (
        f"<div class='pairgroup' data-fbeh='{sort_attr}' data-order='{order}'>"
        + pair_header(plain_name, changed, f_beh, f_act, actonly)
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
    analysis_md: str | None,
    git_commit: str,
    actonly: bool,
    pinned_revision: str | None,
) -> str:
    cells: dict[str, dict] = bank["cells"]
    by_cell: dict[str, list[dict]] = {}
    for p in bank["pairs"]:
        by_cell.setdefault(p["cell"], []).append(p)
    by_edge = {(p["a"], p["b"]): p for p in bank["pairs"]}
    registry: dict[str, str] = {}

    metric = "F_act" if actonly else "F_beh"
    lede_f = (
        "F_act says how much of the full A→B change in the answer vector the patch carries "
        "(0 = none, 1 = all of it; judge-free, from captured activations). F_beh — the same "
        "fraction at judged-behavior level — is pending the judge waves and will be "
        "back-filled here when they land."
        if actonly
        else "F_beh / F_act say how much of the full A→B change in behavior / answer vector "
        "the patch carries (0 = none, 1 = all of it)."
    )
    parts: list[str] = [
        "<!doctype html><html><head><meta charset='utf-8'>",
        "<title>#2329 Result 0 — qualitative gallery</title>",
        f"<style>{CSS}</style><script>{SORT_JS}</script></head><body>",
        "<h1>#2329 Result 0 — qualitative examples (Qwen3.5-9B, thinking disabled)</h1>",
        "<p class='lede'>Each row: one pair of contexts that differ by exactly one attribute "
        "(the difference is shown inline as A: <mark class='a'>old</mark> "
        "<span class='arr'>→</span> B: <mark class='b'>new</mark>), the model's answer under "
        "each, and the answer when context A's context-end activation is patched with B's. "
        f"{lede_f}</p>",
    ]
    if analysis_md is not None:
        # C4 re-seam: the parent required an interpretation markdown; under
        # workflow v2 agents write no interpretation, so the box renders only
        # when a human-authored file is explicitly provided.
        parts.append(
            "<details class='interp'><summary>Analysis — what can and can't transfer "
            "(interpretation)</summary>" + render_markdown_min(analysis_md) + "</details>"
        )
    parts += [
        f"<p>Sort sections by median pair {metric}: "
        "<button class='sort' onclick=\"sortSections('best')\">best first</button>"
        "<button class='sort' onclick=\"sortSections('worst')\">worst first</button>"
        "<button class='sort' onclick=\"sortSections('bank')\">bank order</button></p>",
        "<div id='sections'>",
    ]

    for order, (cell, meta) in enumerate(cells.items()):
        name, desc = cell_plain(cell, meta["base_type"])
        cell_pairs = by_cell.get(cell, [])
        if not cell_pairs:
            parts.append(
                f"<details class='cell' data-fbeh='NaN' data-order='{order}' title='{cell}'>"
                f"<summary><b>{esc(name)}</b> — {esc(desc)}. &nbsp;<span class='d'>no "
                "token-identical pair survives under this tokenizer — not analyzed in this "
                "run</span></summary></details>"
            )
            continue
        fb = [f_cells[p["pair_id"]]["f_beh"] for p in cell_pairs if p["pair_id"] in f_cells]
        fa = [f_cells[p["pair_id"]]["f_act"] for p in cell_pairs if p["pair_id"] in f_cells]
        fb = [v for v in fb if v is not None]
        fa = [v for v in fa if v is not None]
        med_fb = statistics.median(fb) if fb else None
        med_fa = statistics.median(fa) if fa else None
        n_line = f"{len(cell_pairs)} pairs"
        dropped = _dropped_note(bank, cell)
        if fb or fa:
            f_line = (
                f"median F_beh {fbeh_span(med_fb, actonly)} · F_act {f_span(med_fa)} · {n_line}"
            )
        elif actonly and fa == [] and any(p["pair_id"] in f_cells for p in cell_pairs):
            f_line = f"median F_act {f_span(None)} · F_beh {_PENDING} · {n_line}"
        else:
            f_line = f"<span class='d'>F not scored (control cell) · {n_line}</span>"
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
                actonly,
            )
            for i, p in enumerate(cell_pairs)
        ]
        med_sort = med_fa if actonly else med_fb
        sort_attr = "NaN" if med_sort is None else f"{med_sort:.6f}"
        parts.append(
            f"<details class='cell' data-fbeh='{sort_attr}' data-order='{order}' title='{cell}'>"
            f"<summary><b>{esc(name)}</b> — {esc(desc)}. &nbsp;{f_line}{dropped}</summary>"
            f"<p>Sort pairs by {metric}: "
            "<button class='sort' onclick=\"sortRows(this,'best')\">best first</button>"
            "<button class='sort' onclick=\"sortRows(this,'worst')\">worst first</button>"
            "<button class='sort' onclick=\"sortRows(this,'bank')\">bank order</button></p>"
            f"<div class='rows'>{''.join(rows_html)}</div></details>"
        )

    parts.append("</div>")
    parts.append(footer(bank, git_commit, actonly, pinned_revision))
    parts.append(hydrate_block(registry))
    parts.append("</body></html>")
    return "\n".join(parts)


# ── main ──────────────────────────────────────────────────────────────


def _git_commit() -> str:
    import subprocess

    proc = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False)
    return proc.stdout.strip() if proc.returncode == 0 else "unavailable-no-git-checkout"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--dash-dl",
        type=Path,
        default=Path(f"/mnt/eps-data/thomasjiralerspong/{HF_PREFIX}/dash_dl"),
        help="staging root mirroring the HF repo layout (one pinned revision)",
    )
    ap.add_argument("--f-metrics", type=Path, default=Path("eval_results/issue_2329/f_metrics"))
    ap.add_argument(
        "--f-mode",
        choices=("auto", "full", "actonly"),
        default="auto",
        help="full = analysis f_cells.jsonl + coherence (post-judge back-fill); actonly = "
        "pod-side judge-free f_cells_actonly.jsonl, no coherence (generation-complete, "
        "divergence 10); auto = full iff <f-metrics>/f_cells.jsonl exists",
    )
    ap.add_argument(
        "--factonly-jsonl",
        type=Path,
        default=None,
        help="explicit path to f_cells_actonly.jsonl (default: the staged copy under "
        f"--dash-dl at {HF_PREFIX}/analysis_tensors/f_metrics_actonly/)",
    )
    ap.add_argument(
        "--analysis-md",
        type=Path,
        default=None,
        help="OPTIONAL human-authored interpretation markdown for the gallery's top box "
        "(v2 agents write no interpretation — omitted when absent; C4 re-seam)",
    )
    ap.add_argument(
        "--pinned-revision",
        default=None,
        help="HF revision the --dash-dl mirror was staged at (rendered in the footer)",
    )
    ap.add_argument("--out-bank", type=Path, default=Path("docs/issue2329_bank_dashboard.html"))
    ap.add_argument("--out-gallery", type=Path, default=Path("docs/issue2329_result0_gallery.html"))
    return ap.parse_args(argv)


def main() -> None:
    args = parse_args()
    _self_check_word_boundary_marks()

    if args.analysis_md is not None:
        assert args.analysis_md.is_file(), f"missing interpretation markdown: {args.analysis_md}"
        analysis_md: str | None = normalize(args.analysis_md.read_text(encoding="utf-8"))
    else:
        analysis_md = None

    full_f_cells = args.f_metrics / "f_cells.jsonl"
    mode = args.f_mode
    if mode == "auto":
        mode = "full" if full_f_cells.is_file() else "actonly"
    actonly = mode == "actonly"
    print(f"[dashboards] f-mode resolved: {mode}")

    bank = load_bank(args.dash_dl)
    cells = list(bank["cells"])
    grid = load_grid_ce_steered(args.dash_dl, cells)
    anchors = load_anchor_rows(args.dash_dl)
    if actonly:
        # Generation-complete: no judge output exists yet (divergence 10) —
        # empty coherence maps make first_coherent show the first draw.
        coh_grid: dict[tuple[str, int], float] = {}
        coh_anchor: dict[tuple[str, int], float] = {}
        actonly_path = args.factonly_jsonl or (
            args.dash_dl / f"{HF_PREFIX}/analysis_tensors/f_metrics_actonly/f_cells_actonly.jsonl"
        )
        assert actonly_path.is_file(), f"missing F_act-only table: {actonly_path}"
        f_cells = load_f_cells_actonly(actonly_path)
    else:
        coh_grid, coh_anchor = load_coherence(args.dash_dl)
        f_cells = load_f_cells_full(args.f_metrics)
    temps = {r["temperature"] for d in grid.values() for r in d.values()}
    temps |= {r["temperature"] for d in anchors.values() for r in d.values()}
    assert temps == {1.0}, temps  # the footer states temperature 1.0 as fact
    git_commit = _git_commit()

    bank_html = build_bank_dashboard(
        bank, anchors, coh_anchor, git_commit, actonly, args.pinned_revision
    )
    args.out_bank.parent.mkdir(parents=True, exist_ok=True)
    with args.out_bank.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(bank_html)
    print(f"[bank] wrote {args.out_bank} ({len(bank_html) / 1e6:.2f} MB)")

    gallery_html = build_gallery(
        bank,
        grid,
        anchors,
        coh_grid,
        coh_anchor,
        f_cells,
        analysis_md,
        git_commit,
        actonly,
        args.pinned_revision,
    )
    args.out_gallery.parent.mkdir(parents=True, exist_ok=True)
    with args.out_gallery.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(gallery_html)
    print(f"[gallery] wrote {args.out_gallery} ({len(gallery_html) / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
