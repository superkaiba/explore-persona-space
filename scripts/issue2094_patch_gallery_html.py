"""Render the #2094 context-vector patch gallery as one compact HTML table.

One row per matched-query pair at the strongest clean cell
(``ce | joint_all | replace``, steered arm): real prefix, patched prefix,
shared query, evaluated behavior transfer F, and the three answers
(unpatched real prefix / patched / patched prefix in its own setting).

Inputs (all committed or on the HF data repo):
  - patched rows: ``issue2094_singlepos/raw_completions/grid/
    shard_ce__joint_all__replace__A__steered.jsonl``
  - unpatched rows: ``issue2094_singlepos/raw_completions/anchors/anchors.jsonl``
  - F + floor/ceiling: ``eval_results/issue_2094/f_metrics/{f_cells,anchors}.jsonl``

Usage::

    uv run python scripts/issue2094_patch_gallery_html.py \
        --shard-dir /tmp/i2094_shards --out docs/issue2094_patch_gallery.html
"""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

BLOCK_KEY = "ce|joint_all|replace|A|steered"
SETTING = "matched_query"
EXCERPT_CHARS = 260

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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--shard-dir", type=Path, required=True, help="local dir holding the HF shards"
    )
    parser.add_argument("--out", type=Path, required=True, help="output .html path")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    args.out.write_text(build(args.shard_dir, args.repo_root))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
