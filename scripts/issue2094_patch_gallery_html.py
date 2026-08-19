"""Render the #2094 context-vector patch galleries as compact HTML tables.

Three settings, selected by ``--setting``:

``matched_query`` (default — output byte-identical to the original renderer):
one row per matched-query pair at the strongest clean cell
(``ce | joint_all | replace``, steered arm): real prefix, patched prefix,
shared query, evaluated behavior transfer F, and the three answers
(unpatched real prefix / patched / patched prefix in its own setting).

``matched_prefix``: one row per matched-prefix pair (same prefix, two
DIFFERENT queries) at the same ``ce | joint_all | replace`` cell, showing that
the patched answer stays with the ORIGINAL query (F approx. 0), plus a
positive-control section (``qspan | joint_mid``) where patching the query-span
states DOES swap the answer. Rubric family differs by setting
(``bank.SETTING_RUBRIC_KINDS``): matched_query scores prefix rubrics
``fp-<prefix>``; matched_prefix scores query rubrics ``fq-<qid>``, and the
f_beh sub-key is ``query`` instead of ``prefix``.

``overview``: the five direction x setting cells in one dashboard, all at the
same anchor block: matched_query bare->pirate (parent grid), matched_query
pirate->bare (the reverse round, ``mqrev--`` pair ids, its own rollout shard +
judge scores), cross bare->pirate (parent grid, BOTH rubric kinds per
``bank.SETTING_RUBRIC_KINDS``), and the two same-prefix matched_prefix cells
(bare->bare, pirate->pirate). The coverage table carries an explicit NOT-RUN
row for cross pirate->bare, which was never generated. Requires ``--rev-dir``.

Inputs (all committed or on the HF data repo):
  - patched rows: ``issue2094_singlepos/raw_completions/grid/shard_*.jsonl``
  - unpatched rows: ``issue2094_singlepos/raw_completions/anchors/anchors.jsonl``
  - judge scores: ``issue2094_singlepos/raw_completions/judge_raw/scores/``
  - F + floor/ceiling: ``eval_results/issue_2094/f_metrics/{f_cells,anchors}.jsonl``
  - reverse round: ``issue2094_singlepos/raw_completions/rev_direction/shard_*.jsonl``
    (under ``--rev-dir``), scores ``eval_results/issue_2094/judge_rev/scores/``,
    F + floor/ceiling ``eval_results/issue_2094/judge_rev/rev_fbeh_summary.json``

Usage::

    uv run python scripts/issue2094_patch_gallery_html.py \
        --shard-dir data/issue_2094/patch_gallery --out docs/issue2094_patch_gallery.html
    uv run python scripts/issue2094_patch_gallery_html.py --setting matched_prefix \
        --shard-dir data/issue_2094/patch_gallery \
        --out docs/issue2094_patch_gallery_matched_prefix.html
    uv run python scripts/issue2094_patch_gallery_html.py --setting overview \
        --shard-dir data/issue_2094/patch_gallery \
        --rev-dir data/issue_2094/rev_rollouts \
        --out docs/issue2094_patch_gallery_overview.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

BLOCK_KEY = "ce|joint_all|replace|A|steered"
SETTING = "matched_query"
EXCERPT_CHARS = 260

# matched_prefix positive control: the query-span patch cells (brief: a0.5 + replace).
MP_CONTROL_BLOCKS = (
    "qspan|joint_mid|a0.5|A|steered",
    "qspan|joint_mid|replace|A|steered",
)

PREFIX_LABEL = {"bare": "bare", "persona": "pirate", "conv": "party-conv"}
PREFIX_TITLE = {
    "bare": "no prefix at all — the plain default-assistant register",
    "persona": (
        'system prompt: "You are Captain Marrow, a superstitious old pirate captain. '
        "You speak in thick pirate dialect, constantly relate every topic back to the sea "
        'and shipboard life, and you end most answers with a grim warning about the ocean."'
    ),
    "conv": (
        "a completed prior exchange carried as history: the user asks for ideas for their "
        "daughter's 7th birthday party and the assistant answers enthusiastically "
        "(exclamatory, emoji-rich, treasure hunts and pinatas)"
    ),
}
QUERY_SHORT = {
    "q1": "sky blue / red at sunset",
    "q2": "lighthouse-keeper story",
    "q3": "job interview prep",
    "q4": "hash tables",
    "q5": "rent or buy?",
}
QUERY_FULL = {
    "q1": "Why is the sky blue during the day but red at sunset?",
    "q2": "Write the opening paragraph of a short story about a lighthouse keeper.",
    "q3": "How should I prepare for my first job interview next week?",
    "q4": "Explain how a hash table works and when I should use one.",
    "q5": "Do you think it's better to rent or to buy a home? Give your reasoning.",
}

CSS = """
:root { color-scheme: light dark; }
body { font: 14px/1.45 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
       margin: 24px auto; max-width: 1700px; padding: 0 16px; color: #1a1a1a; background: #fff; }
h1 { font-size: 20px; margin: 0 0 6px; }
p.lede { margin: 0 0 6px; color: #444; max-width: 105ch; }
p.lede code { background: #f2f2f2; padding: 1px 4px; border-radius: 3px; }
table { border-collapse: collapse; width: 100%; margin-top: 14px; }
th, td { border: 1px solid #ddd; padding: 7px 9px; vertical-align: top; text-align: left; }
th { background: #f6f6f6; position: sticky; top: 0; font-size: 12px; letter-spacing: .02em; z-index: 1; }
td.num { text-align: right; white-space: nowrap; font-variant-numeric: tabular-nums; }
td.ans { font-size: 12.5px; width: 22%; }
tr.sep td { background: #eef1f5; font-weight: 600; font-size: 12.5px; }
.pfx { font-weight: 600; border-bottom: 1px dotted #888; cursor: help; }
.f { font-weight: 700; }
.hi { color: #1b6d2f; } .mid { color: #9a6400; } .lo { color: #a11; } .na { color: #777; }
.d { font-size: 11.5px; color: #666; }
details { margin-top: 4px; }
summary { cursor: pointer; color: #06c; font-size: 11.5px; }
details pre { white-space: pre-wrap; background: #fafafa; border: 1px solid #eee; border-radius: 4px;
              padding: 8px; margin: 6px 0 0; font-size: 11.5px; max-height: 340px; overflow: auto; }
@media (prefers-color-scheme: dark) {
  body { background: #121212; color: #e6e6e6; }
  th { background: #1e1e1e; } th, td { border-color: #333; }
  tr.sep td { background: #1b2430; } p.lede { color: #bbb; }
  p.lede code { background: #222; }
  details pre { background: #181818; border-color: #2a2a2a; }
  .hi { color: #6ec77f; } .mid { color: #e0b050; } .lo { color: #ef8080; } .na { color: #999; }
}
"""


def parse_pair(pair_id: str) -> tuple[str, str, str]:
    """``mq--bare__q1--persona__q1`` -> ``("bare", "persona", "q1")``."""
    _, a, b = pair_id.split("--")
    prefix_a, query = a.split("__")
    prefix_b, query_b = b.split("__")
    assert query == query_b, (pair_id, query, query_b)
    return prefix_a, prefix_b, query


def parse_pair_mp(pair_id: str) -> tuple[str, str, str]:
    """``mp--bare__q1--bare__q2`` -> ``("bare", "q1", "q2")`` (same prefix, two queries)."""
    _, a, b = pair_id.split("--")
    prefix_a, query_a = a.split("__")
    prefix_b, query_b = b.split("__")
    assert prefix_a == prefix_b, (pair_id, prefix_a, prefix_b)
    return prefix_a, query_a, query_b


def shard_path(shard_dir: Path, block_key: str) -> Path:
    """Grid shard path for a block key (filename doses spell ``a0.5`` as ``a0p5``)."""
    slot, layer_variant, dose, vec, arm = block_key.split("|")
    name = f"shard_{slot}__{layer_variant}__{dose.replace('.', 'p')}__{vec}__{arm}.jsonl"
    return shard_dir / "issue2094_singlepos/raw_completions/grid" / name


def load_jsonl(path: Path) -> list[dict]:
    with path.open() as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_prefix_scores(shard_dir: Path) -> dict[tuple[str, str, str | int], float]:
    """Per-draw prefix-rubric judge scores, keyed ``(rubric_id, unit_key, draw)``.

    ``unit_key`` is the ``context_id`` for anchor rows and the ``pair_id`` for
    grid rows; grid rows carry no draw index, so their key uses ``"grid"``.
    """
    root = shard_dir / "issue2094_singlepos/raw_completions/judge_raw/scores"
    out: dict[tuple[str, str, str | int], float] = {}
    for path in sorted(root.glob("fp-*.scores.jsonl")):
        for row in load_jsonl(path):
            if row["kind"] == "anchor":
                out[(row["rubric_id"], row["context_id"], row["draw"])] = row["score"]
            elif row.get("block_key") == BLOCK_KEY:
                out[(row["rubric_id"], row["pair_id"], "grid")] = row["score"]
    assert out, f"no fp-* prefix-rubric scores under {root}"
    return out


def load_query_grid_scores(
    shard_dir: Path, block_keys: set[str]
) -> dict[tuple[str, str, str], float]:
    """Query-rubric judge scores for grid rows, keyed ``(rubric_id, pair_id, block_key)``."""
    root = shard_dir / "issue2094_singlepos/raw_completions/judge_raw/scores"
    out: dict[tuple[str, str, str], float] = {}
    for path in sorted(root.glob("fq-*.grid.scores.jsonl")):
        for row in load_jsonl(path):
            if row.get("block_key") in block_keys:
                out[(row["rubric_id"], row["pair_id"], row["block_key"])] = row["score"]
    assert out, f"no fq-* query-rubric grid scores under {root}"
    return out


def representative_draw(
    context_id: str,
    rubric_a: str,
    rubric_b: str,
    scores: dict[tuple[str, str, str | int], float],
    coherence: dict[tuple[str, int], dict],
    n_draws: int = 10,
) -> int:
    """Pick the anchor draw that best represents the floor/ceiling the metric uses.

    Selection = the coherent draw whose own Δ (rubric_b − rubric_a, /100) is
    closest to the mean Δ over coherent draws; ties break to higher coherence,
    then lower draw index. Draw 0 is NOT special: on ``conv__q2`` it is one of
    the two lowest-coherence draws of ten (65/100, a mid-answer language
    switch) and the only draw whose Δ departs from the cell mean.
    """
    draws = [d for d in range(n_draws) if (context_id, d) in coherence]
    assert draws, context_id
    coherent = [d for d in draws if coherence[(context_id, d)]["coherent"]] or draws
    deltas = {
        d: (scores[(rubric_b, context_id, d)] - scores[(rubric_a, context_id, d)]) / 100
        for d in coherent
    }
    mean = sum(deltas.values()) / len(deltas)
    return min(
        coherent,
        key=lambda d: (abs(deltas[d] - mean), -coherence[(context_id, d)]["coherence_score"], d),
    )


def answer_cell(text: str, meta: str, chips: str = "") -> str:
    excerpt = " ".join(text[:EXCERPT_CHARS].split())
    ell = "…" if len(text) > EXCERPT_CHARS else ""
    return (
        f'<td class="ans">{html.escape(excerpt)}{ell}'
        f"<details><summary>full</summary><pre>{html.escape(text)}</pre></details>"
        f'<div class="d">{chips}{html.escape(meta)}</div></td>'
    )


def f_class(value: float, separated: bool) -> str:
    if not separated:
        return "na"
    return "hi" if value >= 0.65 else ("mid" if value > 0.05 else "lo")


def build(shard_dir: Path, repo_root: Path) -> str:
    grid = shard_dir / (
        "issue2094_singlepos/raw_completions/grid/shard_ce__joint_all__replace__A__steered.jsonl"
    )
    anchors_raw = shard_dir / "issue2094_singlepos/raw_completions/anchors/anchors.jsonl"
    fm = repo_root / "eval_results/issue_2094/f_metrics"

    patched = {row["pair_id"]: row for row in load_jsonl(grid)}
    anchor_text: dict[tuple[str, int], dict] = {
        (row["context_id"], row["draw"]): row for row in load_jsonl(anchors_raw)
    }
    anchor_coh: dict[tuple[str, int], dict] = {
        (row["context_id"], row["draw"]): row for row in load_jsonl(fm / "anchor_draws.jsonl")
    }
    scores = load_prefix_scores(shard_dir)

    cells = {
        row["pair_id"]: row
        for row in load_jsonl(fm / "f_cells.jsonl")
        if row["block_key"] == BLOCK_KEY and row["setting"] == SETTING
    }
    anchor_stats = {
        row["pair_id"]: row
        for row in load_jsonl(fm / "anchors.jsonl")
        if row.get("kind") == "prefix"
    }
    assert cells, f"no {BLOCK_KEY} / {SETTING} rows in {fm / 'f_cells.jsonl'}"

    def f_beh(pair_id: str) -> float:
        return cells[pair_id]["f_beh"]["prefix"]["f_beh"]

    order = sorted(cells, key=f_beh, reverse=True)
    separated = [p for p in order if abs(anchor_stats[p]["separation"]) >= 0.5]
    degenerate = [p for p in order if abs(anchor_stats[p]["separation"]) < 0.5]

    rows: list[str] = []

    def emit(pair_id: str, is_separated: bool) -> None:
        cell, comp = cells[pair_id], patched[pair_id]
        stats = anchor_stats[pair_id]
        prefix_a, prefix_b, query = parse_pair(pair_id)
        value = f_beh(pair_id)
        rubric_a, rubric_b = f"fp-{prefix_a}", f"fp-{prefix_b}"
        ctx_a, ctx_b = f"{prefix_a}__{query}", f"{prefix_b}__{query}"
        draw_a = representative_draw(ctx_a, rubric_a, rubric_b, scores, anchor_coh)
        draw_b = representative_draw(ctx_b, rubric_a, rubric_b, scores, anchor_coh)
        a_row, b_row = anchor_text[(ctx_a, draw_a)], anchor_text[(ctx_b, draw_b)]

        def chips(unit: str, draw: str | int) -> str:
            score_a = scores[(rubric_a, unit, draw)]
            score_b = scores[(rubric_b, unit, draw)]
            return (
                f"<b>{PREFIX_LABEL[prefix_a]}-register {score_a:.0f}</b> · "
                f"<b>{PREFIX_LABEL[prefix_b]}-register {score_b:.0f}</b><br>"
            )

        rows.append(
            "<tr>"
            f'<td><span class="pfx" title="{html.escape(PREFIX_TITLE[prefix_a])}">'
            f"{PREFIX_LABEL[prefix_a]}</span></td>"
            f'<td><span class="pfx" title="{html.escape(PREFIX_TITLE[prefix_b])}">'
            f"{PREFIX_LABEL[prefix_b]}</span></td>"
            f'<td title="{html.escape(QUERY_FULL[query])}">{html.escape(QUERY_SHORT[query])}</td>'
            f'<td class="num"><span class="f {f_class(value, is_separated)}">{value:+.2f}</span>'
            f'<div class="d">&Delta; {stats["floor"]["mean"]:+.2f} &rarr; '
            f"{cell['f_beh']['prefix']['delta_patched']:+.2f} "
            f"(ceil {stats['ceiling']['mean']:+.2f})<br>sep {stats['separation']:.2f} · "
            f"F_act {cell['f_act']:.2f}</div></td>"
            + answer_cell(
                a_row["text"],
                f"unpatched {PREFIX_LABEL[prefix_a]} · T=1.0 draw {draw_a} of 10 "
                f"(representative of the floor)",
                chips(ctx_a, draw_a),
            )
            + answer_cell(
                comp["text"],
                f"patched · greedy · coherence {cell['coherence_score']:.0f}",
                chips(pair_id, "grid"),
            )
            + answer_cell(
                b_row["text"],
                f"unpatched {PREFIX_LABEL[prefix_b]} · T=1.0 draw {draw_b} of 10 "
                f"(representative of the ceiling)",
                chips(ctx_b, draw_b),
            )
            + "</tr>"
        )

    rows.append(
        '<tr class="sep"><td colspan="7">Measurable pairs '
        "(|anchor separation| &ge; 0.5) — sorted by transfer</td></tr>"
    )
    for pair_id in separated:
        emit(pair_id, True)
    rows.append(
        '<tr class="sep"><td colspan="7">Unmeasurable pairs (separation &lt; 0.5): the target '
        "register is absent from its OWN unpatched answers (column 7), so F divides by &asymp;0 "
        "— excluded from every headline cell</td></tr>"
    )
    for pair_id in degenerate:
        emit(pair_id, False)

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>#2094 — context-vector patch gallery</title>
<style>{CSS}</style></head><body>
<h1>#2094 — patching only the context vector (matched query, all 28 layers, full-state patch)</h1>
<p class="lede">The model runs on the <b>real prefix</b>; only the residual state at the
context-vector position (last prompt token) is replaced by the <b>patched prefix</b>'s, at all 28
layers (<code>ce | joint_all | replace</code>, steered arm). The query is identical in both, so
anything that transfers is prefix information. Hover a prefix or query for its full text.</p>
<p class="lede"><b>F</b> = fraction of a full context swap recovered in judged behavior:
0 = behaves as under the real prefix, 1.0 = as if actually given the patched prefix
(&Delta; = judge<sub>patched-prefix</sub> &minus; judge<sub>real-prefix</sub>, normalized between
the unpatched floor and ceiling, 10 draws each).
<b>Each answer carries its two rubric scores</b> (0&ndash;100: how much it expresses the real-prefix
register, and the patched-prefix register) &mdash; that is what &Delta; is built from.
<b>Sampling:</b> the patched answer is greedy; each unpatched answer is the coherent anchor draw
whose own &Delta; is closest to that context's mean over 10 temperature-1.0 draws (the draw that
best represents the floor / ceiling the metric uses) &mdash; a register reference, not a
same-decoder counterfactual.</p>
<p class="lede"><b>Why the party-conversation register never appears in column 7:</b> it scores
<b>0.0/100 under its own rubric in 49 of its 50 unpatched answers</b> (per-query means 0.0, 0.0,
1.1, 0.0, 0.0; single-draw max 5.0) &mdash; the prefix does not carry its register into later
answers at all. That is exactly why every bare&harr;conv pair is unmeasurable, and why the
pirate&rarr;conv rows score high by DELETING the pirate rather than installing anything.
Separately, 9 of the 150 unpatched draws drift into another language mid-answer at temperature 1.0
(a low-rate model behavior, not an artifact of the patch).</p>
<table>
<thead><tr>
<th>real prefix</th><th>patched prefix</th><th>shared query</th><th>transfer</th>
<th>answer &mdash; real prefix, unpatched</th>
<th>answer &mdash; patched</th>
<th>answer &mdash; patched prefix in its own setting</th>
</tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table>
<p class="lede d">Sources: patched rows HF
<code>superkaiba1/explore-persona-space-data</code> &middot;
<code>issue2094_singlepos/raw_completions/grid/shard_ce__joint_all__replace__A__steered.jsonl</code>;
unpatched rows <code>&hellip;/raw_completions/anchors/anchors.jsonl</code>; F and floor/ceiling
<code>eval_results/issue_2094/f_metrics/{{f_cells,anchors}}.jsonl</code>. Rendered by
<code>scripts/issue2094_patch_gallery_html.py</code>.</p>
</body></html>
"""


def build_matched_prefix(shard_dir: Path, repo_root: Path) -> str:
    """Matched-prefix gallery: same prefix, two queries; does the patch swap the answer?"""
    fm = repo_root / "eval_results/issue_2094/f_metrics"
    all_blocks = {BLOCK_KEY, *MP_CONTROL_BLOCKS}
    scores = load_query_grid_scores(shard_dir, all_blocks)

    cells_all = [
        row for row in load_jsonl(fm / "f_cells.jsonl") if row["setting"] == "matched_prefix"
    ]
    by_block: dict[str, dict[str, dict]] = {}
    for row in cells_all:
        if row["block_key"] in all_blocks:
            by_block.setdefault(row["block_key"], {})[row["pair_id"]] = row
    assert set(by_block) == all_blocks, sorted(by_block)

    patched: dict[str, dict[str, dict]] = {
        bk: {row["pair_id"]: row for row in load_jsonl(shard_path(shard_dir, bk))}
        for bk in all_blocks
    }
    anchor_stats = {
        row["pair_id"]: row
        for row in load_jsonl(fm / "anchors.jsonl")
        if row.get("kind") == "query" and row.get("setting") == "matched_prefix"
    }

    def fq(cell: dict) -> dict:
        return cell["f_beh"]["query"]

    def emit(cell: dict) -> str:
        pair_id, block_key = cell["pair_id"], cell["block_key"]
        prefix, query_a, query_b = parse_pair_mp(pair_id)
        stats = anchor_stats[pair_id]
        comp = patched[block_key][pair_id]
        value = fq(cell)["f_beh"]
        delta = fq(cell)["delta_patched"]
        score_a = scores[(f"fq-{query_a}", pair_id, block_key)]
        score_b = scores[(f"fq-{query_b}", pair_id, block_key)]
        if delta is not None:
            # the stored delta IS (judge_B − judge_A)/100 of this completion — no hand math
            assert abs((score_b - score_a) / 100 - delta) < 1e-4, (pair_id, block_key)
        chips = f"<b>answers A {score_a:.0f}</b> · <b>answers B {score_b:.0f}</b><br>"
        f_span = (
            '<span class="f na">n/a</span>'
            if value is None
            else f'<span class="f {f_class(value, True)}">{value:+.2f}</span>'
        )
        delta_txt = "n/a" if delta is None else f"{delta:+.2f}"
        return (
            "<tr>"
            f'<td><span class="pfx" title="{html.escape(PREFIX_TITLE[prefix])}">'
            f"{PREFIX_LABEL[prefix]}</span></td>"
            f'<td title="{html.escape(QUERY_FULL[query_a])}">'
            f"{html.escape(QUERY_SHORT[query_a])}</td>"
            f'<td title="{html.escape(QUERY_FULL[query_b])}">'
            f"{html.escape(QUERY_SHORT[query_b])}</td>"
            f'<td class="num">{f_span}'
            f'<div class="d">&Delta; {stats["floor"]["mean"]:+.2f} &rarr; {delta_txt} '
            f"(ceil {stats['ceiling']['mean']:+.2f})<br>sep {stats['separation']:.2f}"
            + (f" · F_act {cell['f_act']:.2f}" if cell.get("f_act") is not None else "")
            + "</div></td>"
            + answer_cell(
                comp["text"],
                f"patched · greedy · coherence {cell['coherence_score']:.0f}",
                chips,
            )
            + "</tr>"
        )

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    rows: list[str] = []

    # main block: every pair, sorted by F descending (both extremes visible)
    def sort_key(cell: dict) -> tuple[bool, float]:
        value = fq(cell)["f_beh"]
        return (value is not None, value if value is not None else 0.0)

    main = sorted(by_block[BLOCK_KEY].values(), key=sort_key, reverse=True)
    weak = [
        c
        for c in main
        if fq(c)["f_beh"] is None
        or fq(c)["degenerate_denominator"]
        or abs(anchor_stats[c["pair_id"]]["separation"]) < 0.5
    ]
    clean = [c for c in main if c not in weak]
    f_main = [fq(c)["f_beh"] for c in clean]
    d_main = [fq(c)["delta_patched"] for c in clean]
    rows.append(
        f'<tr class="sep"><td colspan="5">Context-end patch '
        f"(<code>ce | joint_all | replace</code>) — all {len(clean)} pairs, sorted by F. "
        f"Mean F {mean(f_main):+.3f}, mean &Delta;<sub>patched</sub> {mean(d_main):+.2f} "
        f"(floor &minus;1.00): the answer stays with query A in every pair</td></tr>"
    )
    rows.extend(emit(c) for c in clean)
    if weak:
        rows.append(
            f'<tr class="sep"><td colspan="5">Weak-denominator / excluded pairs '
            f"({len(weak)}): degenerate or incoherent — excluded from every headline "
            f"cell, shown here rather than silently dropped</td></tr>"
        )
        rows.extend(emit(c) for c in weak)

    # positive control: qspan cells — the same instrument DOES register a query swap
    for block_key in MP_CONTROL_BLOCKS:
        dose = block_key.split("|")[2]
        cells = list(by_block[block_key].values())
        coherent = sorted(
            (c for c in cells if fq(c)["f_beh"] is not None),
            key=lambda c: fq(c)["f_beh"],
            reverse=True,
        )
        n_excl = len(cells) - len(coherent)
        f_ctl = [fq(c)["f_beh"] for c in coherent]
        d_ctl = [fq(c)["delta_patched"] for c in coherent]
        excl_note = f"; {n_excl} of {len(cells)} excluded as incoherent" if n_excl else ""
        shown = [coherent[0], coherent[-1]] if len(coherent) > 1 else coherent
        rows.append(
            f'<tr class="sep"><td colspan="5">POSITIVE CONTROL — patching the query-span '
            f"states instead (<code>qspan | joint_mid | {dose}</code>): mean F "
            f"{mean(f_ctl):+.2f}, mean &Delta;<sub>patched</sub> {mean(d_ctl):+.2f} over "
            f"{len(coherent)} coherent pairs{excl_note}. Highest- and lowest-F rows shown"
            f"</td></tr>"
        )
        rows.extend(emit(c) for c in shown)

    min_sep = min(abs(a["separation"]) for a in anchor_stats.values())
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>#2094 — patch gallery (matched prefix: same prefix, different query)</title>
<style>{CSS}</style></head><body>
<h1>#2094 — patching only the context vector across queries (same prefix, all 28 layers,
full-state patch)</h1>
<p class="lede">The model is asked <b>query A</b> under a prefix; only the residual state at the
context-vector position (last prompt token) is replaced by the state from the SAME prefix asked
<b>query B</b>, at all 28 layers (<code>ce | joint_all | replace</code>, steered arm). The prefix
is identical in both, so if the context-end state carried the query, the patched answer should
drift toward B. It does not: the model keeps answering the ORIGINAL query A. Hover a prefix or
query for its full text.</p>
<p class="lede"><b>F</b> = fraction of a full query swap recovered in the judged answer:
0 = still answers query A (as unpatched), 1.0 = answers query B (as if actually asked B).
&Delta; = (judge<sub>B</sub> &minus; judge<sub>A</sub>)/100 &isin; [&minus;1,+1], where each judge
score (0&ndash;100) rates how fully the answer addresses that query; &minus;1.0 = answers A fully,
B not at all. F normalizes &Delta; between the unpatched floor (A&rsquo;s own answers, 10 draws)
and ceiling (B&rsquo;s, 10 draws). <b>Each patched answer carries its two judge scores</b>
(&ldquo;answers A&rdquo; / &ldquo;answers B&rdquo;) &mdash; that is what &Delta; is built from.
All pairs are well-separated (min |separation| {min_sep:.2f} of 2.0), so none is excluded for a
weak denominator.</p>
<p class="lede"><b>Positive control:</b> the null is a property of the POSITION, not a dead
measure &mdash; patching the query-span states at the middle layers (<code>qspan | joint_mid</code>,
bottom sections) flips the same judged metric to F &asymp; +0.9: the instrument registers a query
swap whenever the query span itself is patched.</p>
<p class="lede d">Answer excerpts are whitespace-collapsed and truncated to {EXCERPT_CHARS}
characters (&ldquo;&hellip;&rdquo; marks truncation); expand &ldquo;full&rdquo; for the verbatim
stored completion.</p>
<table>
<thead><tr>
<th>shared prefix</th><th>query A (asked)</th><th>query B (patched-in donor)</th>
<th>query swap F</th>
<th>answer &mdash; patched, judged against both queries</th>
</tr></thead>
<tbody>
{chr(10).join(rows)}
</tbody></table>
<p class="lede d">Sources: patched rows HF
<code>superkaiba1/explore-persona-space-data</code> &middot;
<code>issue2094_singlepos/raw_completions/grid/shard_ce__joint_all__replace__A__steered.jsonl</code>
+ <code>&hellip;/grid/shard_qspan__joint_mid__{{a0p5,replace}}__A__steered.jsonl</code>;
judge scores <code>&hellip;/raw_completions/judge_raw/scores/fq-q*.grid.scores.jsonl</code>;
F and floor/ceiling <code>eval_results/issue_2094/f_metrics/{{f_cells,anchors}}.jsonl</code>.
Rendered by <code>scripts/issue2094_patch_gallery_html.py --setting matched_prefix</code>.</p>
</body></html>
"""


# ---------------------------------------------------------------------------
# Direction x setting overview (five-cell dashboard) — task brief 2026-08-14
# ---------------------------------------------------------------------------

# Mirrors bank.SETTING_RUBRIC_KINDS (src/explore_persona_space/experiments/
# issue2094/bank.py) — hardcoded here so the renderer stays standalone.
SETTING_RUBRIC_KINDS: dict[str, tuple[str, ...]] = {
    "matched_prefix": ("query",),
    "matched_query": ("prefix",),
    "cross": ("prefix", "query"),
}

REV_SHARD_REL = (
    "issue2094_singlepos/raw_completions/rev_direction/"
    "shard_ce__joint_all__replace__A__steered.jsonl"
)

# The five cells the dashboard covers (setting, direction = (real prefix,
# donor prefix), source). cross persona->bare was NEVER RUN — carried as an
# explicit NOT-RUN coverage row, never fabricated and never silently omitted.
OVERVIEW_CELLS: tuple[dict, ...] = (
    {
        "setting": "matched_query",
        "direction": ("bare", "persona"),
        "source": "parent grid",
        "what": "same query; the bare context-end state is replaced by the pirate one",
    },
    {
        "setting": "matched_query",
        "direction": ("persona", "bare"),
        "source": "reverse round",
        "what": "same query; the pirate context-end state is replaced by the bare one",
    },
    {
        "setting": "cross",
        "direction": ("bare", "persona"),
        "source": "parent grid",
        "what": "donor differs in BOTH prefix and query: pirate prefix asked a different "
        "question — scored under both rubric kinds",
    },
    {
        "setting": "matched_prefix",
        "direction": ("bare", "bare"),
        "source": "parent grid",
        "what": "same bare prefix, different query — does the patch swap the answer?",
    },
    {
        "setting": "matched_prefix",
        "direction": ("persona", "persona"),
        "source": "parent grid",
        "what": "same pirate prefix, different query — does the patch swap the answer?",
    },
)

OVERVIEW_CSS = """
table.cov { width: auto; min-width: 60%; }
table.cov td, table.cov th { font-size: 12.5px; }
tr.notrun td { color: #a11; font-weight: 600; }
td.ans { width: 45%; }
.kd { font-size: 11px; color: #666; }
@media (prefers-color-scheme: dark) { tr.notrun td { color: #ef8080; } .kd { color: #999; } }
"""


def parse_contexts(pair_id: str) -> tuple[str, str, str, str]:
    """Any pair id -> ``(prefix_a, query_a, prefix_b, query_b)`` (A real, B donor)."""
    _, a, b = pair_id.split("--")
    prefix_a, query_a = a.split("__")
    prefix_b, query_b = b.split("__")
    return prefix_a, query_a, prefix_b, query_b


def pair_direction(pair_id: str) -> tuple[str, str]:
    """Direction filter key: (real-context prefix, donor-context prefix)."""
    prefix_a, _, prefix_b, _ = parse_contexts(pair_id)
    return prefix_a, prefix_b


def load_prefix_grid_scores(
    shard_dir: Path, block_keys: set[str]
) -> dict[tuple[str, str, str], float]:
    """Prefix-rubric judge scores for grid rows, keyed ``(rubric_id, pair_id, block_key)``."""
    root = shard_dir / "issue2094_singlepos/raw_completions/judge_raw/scores"
    out: dict[tuple[str, str, str], float] = {}
    for path in sorted(root.glob("fp-*.grid.scores.jsonl")):
        for row in load_jsonl(path):
            if row.get("block_key") in block_keys:
                out[(row["rubric_id"], row["pair_id"], row["block_key"])] = row["score"]
    assert out, f"no fp-* prefix-rubric grid scores under {root}"
    return out


def build_overview(shard_dir: Path, rev_dir: Path, repo_root: Path) -> str:
    """Five-cell direction x setting dashboard at ``ce | joint_all | replace`` (steered)."""
    fm = repo_root / "eval_results/issue_2094/f_metrics"
    patched = {row["pair_id"]: row for row in load_jsonl(shard_path(shard_dir, BLOCK_KEY))}
    fp_scores = load_prefix_grid_scores(shard_dir, {BLOCK_KEY})
    fq_scores = load_query_grid_scores(shard_dir, {BLOCK_KEY})
    cells = {
        (row["setting"], row["pair_id"]): row
        for row in load_jsonl(fm / "f_cells.jsonl")
        if row["block_key"] == BLOCK_KEY
    }
    anchor_stats = {(row["pair_id"], row["kind"]): row for row in load_jsonl(fm / "anchors.jsonl")}

    # Reverse round (pirate -> bare): its own rollout shard, judge scores, and
    # F summary — floor/ceiling reuse the parent's anchor draws (no new anchors).
    jr = repo_root / "eval_results/issue_2094/judge_rev"
    rev_patched = {row["pair_id"]: row for row in load_jsonl(rev_dir / REV_SHARD_REL)}
    rev_scores = {
        name: {
            row["pair_id"]: row["score"]
            for row in load_jsonl(jr / "scores" / f"{name}.grid.scores.jsonl")
            if row["block_key"] == BLOCK_KEY
        }
        for name in ("fp-bare", "fp-persona", "coherence")
    }
    rev_summary = json.loads((jr / "rev_fbeh_summary.json").read_text())
    rev_fc = rev_summary["floor_ceiling"]
    rev_f = dict(rev_summary["aggregates"]["joint_all|steered"]["per_pair_f_beh"])

    def read_label(kind: str, prefix_or_query: str) -> str:
        if kind == "prefix":
            return f"{PREFIX_LABEL[prefix_or_query]}-register"
        return f"answers {prefix_or_query}"

    def parent_rows(setting: str, direction: tuple[str, str]) -> list[dict]:
        """One row dict per pair of this cell; every number re-read from artifacts."""
        rows_out = []
        for (s, pair_id), cell in cells.items():
            if s != setting or pair_direction(pair_id) != direction:
                continue
            prefix_a, query_a, prefix_b, query_b = parse_contexts(pair_id)
            reads = []
            for kind in SETTING_RUBRIC_KINDS[setting]:
                fb = cell["f_beh"][kind]
                stats = anchor_stats[(pair_id, kind)]
                if kind == "prefix":
                    score_a = fp_scores[(f"fp-{prefix_a}", pair_id, BLOCK_KEY)]
                    score_b = fp_scores[(f"fp-{prefix_b}", pair_id, BLOCK_KEY)]
                    label_a, label_b = read_label(kind, prefix_a), read_label(kind, prefix_b)
                else:
                    score_a = fq_scores[(f"fq-{query_a}", pair_id, BLOCK_KEY)]
                    score_b = fq_scores[(f"fq-{query_b}", pair_id, BLOCK_KEY)]
                    label_a, label_b = read_label(kind, query_a), read_label(kind, query_b)
                if fb["delta_patched"] is not None:
                    # the stored delta IS (judge_B - judge_A)/100 of this completion
                    assert abs((score_b - score_a) / 100 - fb["delta_patched"]) < 1e-4, (
                        pair_id,
                        kind,
                    )
                reads.append(
                    {
                        "kind": kind,
                        "f": fb["f_beh"],
                        "delta": fb["delta_patched"],
                        "floor": stats["floor"]["mean"],
                        "ceiling": stats["ceiling"]["mean"],
                        "sep": stats["separation"],
                        "degenerate": fb["degenerate_denominator"],
                        "score_a": score_a,
                        "score_b": score_b,
                        "label_a": label_a,
                        "label_b": label_b,
                    }
                )
            rows_out.append(
                {
                    "pair_id": pair_id,
                    "prefix_a": prefix_a,
                    "query_a": query_a,
                    "prefix_b": prefix_b,
                    "query_b": query_b,
                    "text": patched[pair_id]["text"],
                    "coherence": cell["coherence_score"],
                    "reads": reads,
                }
            )
        return rows_out

    def rev_rows() -> list[dict]:
        rows_out = []
        for pair_id, comp in rev_patched.items():
            prefix_a, query_a, prefix_b, query_b = parse_contexts(pair_id)
            score_a = rev_scores[f"fp-{prefix_a}"][pair_id]
            score_b = rev_scores[f"fp-{prefix_b}"][pair_id]
            fc = rev_fc[pair_id]
            # delta per the summary's own definition: (fp-bare - fp-persona)/100;
            # recompute F from floor/denominator and pin it to the summary value.
            delta = (score_b - score_a) / 100
            f_val = (delta - fc["floor"]) / fc["denominator"]
            assert abs(f_val - rev_f[pair_id]) < 1e-6, (pair_id, f_val, rev_f[pair_id])
            rows_out.append(
                {
                    "pair_id": pair_id,
                    "prefix_a": prefix_a,
                    "query_a": query_a,
                    "prefix_b": prefix_b,
                    "query_b": query_b,
                    "text": comp["text"],
                    "coherence": rev_scores["coherence"][pair_id],
                    "reads": [
                        {
                            "kind": "prefix",
                            "f": f_val,
                            "delta": delta,
                            "floor": fc["floor"],
                            "ceiling": fc["ceiling"],
                            "sep": fc["denominator"],
                            "degenerate": fc["denominator"] < 0.5,
                            "score_a": score_a,
                            "score_b": score_b,
                            "label_a": read_label("prefix", prefix_a),
                            "label_b": read_label("prefix", prefix_b),
                        }
                    ],
                }
            )
        return rows_out

    def included(row: dict) -> bool:
        primary = row["reads"][0]
        return primary["f"] is not None and not primary["degenerate"] and abs(primary["sep"]) >= 0.5

    def mean(values: list[float]) -> float:
        return sum(values) / len(values)

    def emit_row(row: dict) -> str:
        f_lines = []
        for read in row["reads"]:
            kind_tag = (
                f' <span class="kd">{"register" if read["kind"] == "prefix" else "query"}</span>'
                if len(row["reads"]) > 1
                else ""
            )
            if read["f"] is None:
                f_lines.append(f'<span class="f na">n/a</span>{kind_tag}')
                continue
            ok = not read["degenerate"] and abs(read["sep"]) >= 0.5
            f_lines.append(
                f'<span class="f {f_class(read["f"], ok)}">{read["f"]:+.2f}</span>{kind_tag}'
                f'<div class="d">&Delta; {read["floor"]:+.2f} &rarr; {read["delta"]:+.2f} '
                f"(ceil {read['ceiling']:+.2f}) &middot; sep {read['sep']:.2f}</div>"
            )
        chips = (
            "<br>".join(
                f"<b>{read['label_a']} {read['score_a']:.0f}</b> &middot; "
                f"<b>{read['label_b']} {read['score_b']:.0f}</b>"
                for read in row["reads"]
            )
            + "<br>"
        )
        excl = "" if included(row) else ' <span class="lo">(excluded from mean)</span>'
        return (
            "<tr>"
            f'<td><span class="pfx" title="{html.escape(PREFIX_TITLE[row["prefix_a"]])}">'
            f"{PREFIX_LABEL[row['prefix_a']]}</span></td>"
            f'<td title="{html.escape(QUERY_FULL[row["query_a"]])}">'
            f"{html.escape(QUERY_SHORT[row['query_a']])}</td>"
            f'<td><span class="pfx" title="{html.escape(PREFIX_TITLE[row["prefix_b"]])}">'
            f"{PREFIX_LABEL[row['prefix_b']]}</span></td>"
            f'<td title="{html.escape(QUERY_FULL[row["query_b"]])}">'
            f"{html.escape(QUERY_SHORT[row['query_b']])}</td>"
            f'<td class="num">{"".join(f_lines)}{excl}</td>'
            + answer_cell(
                row["text"],
                f"patched · greedy · coherence {row['coherence']:.0f}",
                chips,
            )
            + "</tr>"
        )

    body_rows: list[str] = []
    cov_rows: list[str] = []
    for i, spec in enumerate(OVERVIEW_CELLS):
        cell_rows = (
            rev_rows()
            if spec["source"] == "reverse round"
            else parent_rows(spec["setting"], spec["direction"])
        )
        assert cell_rows, spec
        cell_rows.sort(
            key=lambda r: (r["reads"][0]["f"] is not None, r["reads"][0]["f"] or 0.0), reverse=True
        )
        clean = [r for r in cell_rows if included(r)]
        n_excl = len(cell_rows) - len(clean)
        mean_f = mean([r["reads"][0]["f"] for r in clean])
        dir_label = (
            f"{PREFIX_LABEL[spec['direction'][0]]} &rarr; {PREFIX_LABEL[spec['direction'][1]]}"
        )
        mean_txt = f"mean F {mean_f:+.3f}"
        if spec["setting"] == "cross":
            mean_fq = mean([r["reads"][1]["f"] for r in clean if r["reads"][1]["f"] is not None])
            mean_txt = f"mean register-F {mean_f:+.3f} &middot; mean query-F {mean_fq:+.3f}"
        excl_txt = f"; {n_excl} excluded (weak denominator / incoherent)" if n_excl else ""
        body_rows.append(
            f'<tr class="sep" id="cell{i}"><td colspan="6">{spec["setting"]} &middot; '
            f"{dir_label} &middot; {spec['source']} — {spec['what']}. "
            f"{len(clean)} pairs, {mean_txt}{excl_txt}</td></tr>"
        )
        body_rows.extend(emit_row(r) for r in cell_rows)
        cov_rows.append(
            f'<tr><td><a href="#cell{i}">{spec["setting"]}</a></td><td>{dir_label}</td>'
            f'<td>{spec["source"]}</td><td class="num">{len(clean)}</td>'
            f'<td class="num">{mean_txt}</td></tr>'
        )
    cov_rows.append(
        '<tr class="notrun"><td>cross</td><td>pirate &rarr; bare</td><td>—</td>'
        '<td class="num">0</td><td>NOT RUN — this direction was never generated '
        "(no rollouts, no judge scores)</td></tr>"
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>#2094 — patch gallery: direction &times; setting</title>
<style>{CSS}{OVERVIEW_CSS}</style></head><body>
<h1>#2094 — patching the context-end vector across direction &times; setting
(all 28 layers, full-state patch)</h1>
<p class="lede">Every row: the model runs on <b>context A</b> (real prefix + asked query); only
the residual state at the context-vector position (last prompt token) is replaced by
<b>context B</b>'s (the donor), at all 28 layers (<code>ce | joint_all | replace</code>, steered
arm), and the completion is decoded greedily. The five cells below vary WHAT differs between A
and B (the setting) and in WHICH DIRECTION the patch runs. Hover a prefix or query for its full
text.</p>
<p class="lede"><b>F</b> = fraction of a full context swap recovered in judged behavior:
0 = behaves as under context A unpatched, 1.0 = as if actually given context B.
&Delta; = (judge<sub>B</sub> &minus; judge<sub>A</sub>)/100 of the patched completion;
F normalizes &Delta; between the unpatched floor (A&rsquo;s own answers, 10 temperature-1.0
draws) and ceiling (B&rsquo;s, 10 draws). <b>Register-F</b> uses prefix rubrics (0&ndash;100: how
much the answer expresses each prefix&rsquo;s register) — the read for matched-query rows.
<b>Query-F</b> uses query rubrics (how fully the answer addresses each query) — the read for
matched-prefix rows. <b>Cross rows score BOTH</b>: did the donor&rsquo;s register arrive, and did
the answer stay with the asked query? Each patched answer carries the judge scores its
&Delta; is built from.</p>
<p class="lede">A &ldquo;bare &rarr; pirate&rdquo; direction is undefined in the matched_prefix
setting: the prefix is held constant by construction (that is what &ldquo;matched prefix&rdquo;
means), so the two same-prefix cells below — bare &rarr; bare and pirate &rarr; pirate with
different queries — ARE the analogue, not an omission. The sixth direction cell,
cross pirate &rarr; bare, was never run; the coverage table says so explicitly.</p>
<p class="lede d">Answer excerpts are whitespace-collapsed and truncated to {EXCERPT_CHARS}
characters (&ldquo;&hellip;&rdquo; marks truncation); expand &ldquo;full&rdquo; for the verbatim
stored completion.</p>
<table class="cov">
<thead><tr><th>setting</th><th>direction</th><th>source</th><th>pairs</th><th>mean F</th></tr>
</thead>
<tbody>
{chr(10).join(cov_rows)}
</tbody></table>
<table>
<thead><tr>
<th>real prefix (A)</th><th>asked query (A)</th><th>donor prefix (B)</th><th>donor query (B)</th>
<th>transfer F</th>
<th>answer &mdash; patched, with its judge scores</th>
</tr></thead>
<tbody>
{chr(10).join(body_rows)}
</tbody></table>
<p class="lede d">Sources: patched rows HF
<code>superkaiba1/explore-persona-space-data</code> &middot;
<code>issue2094_singlepos/raw_completions/grid/shard_ce__joint_all__replace__A__steered.jsonl</code>
(parent) + <code>&hellip;/raw_completions/rev_direction/shard_ce__joint_all__replace__A__steered.jsonl</code>
(reverse round); judge scores
<code>&hellip;/raw_completions/judge_raw/scores/</code> (parent) +
<code>eval_results/issue_2094/judge_rev/scores/</code> (reverse round); F and floor/ceiling
<code>eval_results/issue_2094/f_metrics/{{f_cells,anchors}}.jsonl</code> (parent) +
<code>eval_results/issue_2094/judge_rev/rev_fbeh_summary.json</code> (reverse round). Rendered by
<code>scripts/issue2094_patch_gallery_html.py --setting overview</code>.</p>
</body></html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-dir", type=Path, required=True, help="local dir holding the HF shards"
    )
    parser.add_argument("--out", type=Path, required=True, help="output .html path")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--setting",
        choices=("matched_query", "matched_prefix", "overview"),
        default="matched_query",
        help="which pair setting to render (default: matched_query, byte-identical output)",
    )
    parser.add_argument(
        "--rev-dir",
        type=Path,
        default=None,
        help="local dir holding the reverse-round rev_direction shards "
        "(required for --setting overview)",
    )
    args = parser.parse_args()
    if args.setting == "overview":
        assert args.rev_dir is not None, "--rev-dir is required for --setting overview"
        text = build_overview(args.shard_dir, args.rev_dir, args.repo_root)
    else:
        build_fn = build if args.setting == "matched_query" else build_matched_prefix
        text = build_fn(args.shard_dir, args.repo_root)
    args.out.write_text(text)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
