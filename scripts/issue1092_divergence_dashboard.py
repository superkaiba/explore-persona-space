#!/usr/bin/env python3
"""#1092 dashboard: top prefixes where the prefix-end and averaged-context reads disagree.

Renders the ranked arm-disagreement tables banked in
``eval_results/issue_1092/inline_prefixend_monitoring/divergence_analysis.json``
(user-chat inline round, 2026-07-22) as one standalone HTML page, one section
per trait (sycophancy, hallucination), joining each prefix with its VERBATIM
conversation text from the corpus ``prefix_store.jsonl`` so the outliers can be
eyeballed and categorized.

Per row: prefix metadata (stratum, source, topic, user turns, tokens), the
prefix's measured mean judge score over its 48 own-policy answers, both
held-out predictions (prefix-end vs averaged-context, grouped-5-fold GCV dual
ridge), their disagreement delta, the differential absolute error (which arm
was closer), and the expandable full conversation.

Data sources (canonical on the HF data repo; local cache reused):
  eval_results/issue_1092/inline_prefixend_monitoring/divergence_analysis.json
  issue1092_realistic_crossing/corpus/prefix_store.jsonl   -- prefix conversations

CONTEXT-HYGIENE: the corpus is UNSCREENED real-world user text (WildChat /
LMSYS). Raw text is handled ONLY inside this script and written (HTML-escaped,
per-turn hard-truncated with an inline disclosure marker) into the HTML;
NOTHING raw is printed to stdout -- verification prints counts only.

Usage:
  uv run python scripts/issue1092_divergence_dashboard.py
"""

# ruff: noqa: E501

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue1092_realistic_crossing"
CACHE = Path("/tmp/i1092dash")  # same cache dir as issue1092_corpus_dashboard.py

DIVERGENCE_JSON = (
    PROJECT_ROOT / "eval_results/issue_1092/inline_prefixend_monitoring/divergence_analysis.json"
)
OUT = PROJECT_ROOT / "tasks/awaiting_promotion/1092/artifacts/issue1092_divergence_dashboard.html"

TURN_CHAR_CAP = 4000  # per-turn display cap; truncation is disclosed inline

TRAITS = ("sycophancy", "hallucination")
LISTS = (
    ("top_prefixes_by_abs_pred_delta", "top-|Δ|"),
    ("top_prefixes_prefix_end_worse", "prefix-end worse"),
    ("top_prefixes_prefix_end_better", "prefix-end better"),
)


def _fetch_prefix_store() -> Path:
    p = CACHE / HF_PREFIX / "corpus/prefix_store.jsonl"
    if not p.exists():
        from huggingface_hub import hf_hub_download

        from explore_persona_space.orchestrate.hub import retry_transient

        retry_transient(
            lambda: hf_hub_download(
                HF_DATA_REPO,
                f"{HF_PREFIX}/corpus/prefix_store.jsonl",
                repo_type="dataset",
                local_dir=CACHE,
            ),
            what="hf_hub_download(prefix_store.jsonl)",
        )
    if not p.exists():
        raise SystemExit(f"prefix_store.jsonl not found at {p}")
    return p


def _load_prefix_store(p: Path) -> dict[str, dict]:
    store: dict[str, dict] = {}
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            store[d["prefix_id"]] = d
    return store


def _render_turns(entry: dict | None) -> str:
    if entry is None:
        return "<p class='miss'>prefix text not present in prefix_store.jsonl</p>"
    turns = entry.get("prefix_turns") or []
    if not turns and entry.get("system_prompt"):
        turns = [{"role": "system", "content": entry["system_prompt"]}]
    if not turns:
        return "<p class='miss'>no turns recorded for this prefix</p>"
    parts = []
    for t in turns:
        content = t.get("content") or ""
        shown = content[:TURN_CHAR_CAP]
        trunc = ""
        if len(content) > TURN_CHAR_CAP:
            trunc = f"<div class='trunc'>[truncated — showing first {TURN_CHAR_CAP:,} of {len(content):,} chars]</div>"
        parts.append(
            f"<div class='turn turn-{html.escape(t.get('role', '?'))}'>"
            f"<span class='role'>{html.escape(t.get('role', '?'))}</span>"
            f"<pre>{html.escape(shown)}</pre>{trunc}</div>"
        )
    return "\n".join(parts)


def _union_rows(trait_block: dict) -> list[dict]:
    """Union of the three banked top lists, deduped by prefix_id, ranked by |delta|."""
    rows: dict[str, dict] = {}
    for key, label in LISTS:
        for r in trait_block.get(key, []):
            pid = r["prefix_id"]
            if pid not in rows:
                rows[pid] = dict(r)
                rows[pid]["_lists"] = []
            rows[pid]["_lists"].append(label)
    out = sorted(rows.values(), key=lambda r: -abs(r["delta_pred"]))
    return out


def _row_html(i: int, r: dict, store: dict[str, dict]) -> str:
    pid = r["prefix_id"]
    winner = "averaged-context" if r["abs_err_diff_pe_minus_ac"] > 0 else "prefix-end"
    if abs(r["abs_err_diff_pe_minus_ac"]) < 0.05:
        winner = "~tie"
    meta = (
        f"<td>{i}</td>"
        f"<td class='pid'>{html.escape(pid)}</td>"
        f"<td>{html.escape(r.get('stratum', ''))}</td>"
        f"<td>{html.escape(str(r.get('source', '')))}</td>"
        f"<td>{html.escape(str(r.get('topic', '')))}</td>"
        f"<td>{r.get('n_user_turns', '')}</td>"
        f"<td>{r.get('median_context_tokens', 0):.0f}</td>"
        f"<td>{r['judge_mean']:.1f} ± {r.get('judge_std', float('nan')):.1f}</td>"
        f"<td>{r['pred_prefix_end']:.1f}</td>"
        f"<td>{r['pred_averaged_context']:.1f}</td>"
        f"<td class='delta'>{r['delta_pred']:+.1f}</td>"
        f"<td>{winner}</td>"
        f"<td>{html.escape(', '.join(r['_lists']))}</td>"
    )
    conv = _render_turns(store.get(pid))
    return (
        f"<tr>{meta}</tr>\n"
        f"<tr class='convrow'><td colspan='13'><details><summary>conversation ({html.escape(pid)})</summary>"
        f"<div class='conv'>{conv}</div></details></td></tr>"
    )


CSS = """
body { font-family: -apple-system, 'Segoe UI', Helvetica, Arial, sans-serif; margin: 24px; color: #1a1a1a; max-width: 1400px; }
h1 { font-size: 1.4em; } h2 { font-size: 1.15em; margin-top: 1.6em; }
.note { background: #f6f8fa; border: 1px solid #d0d7de; border-radius: 6px; padding: 10px 14px; font-size: 0.9em; }
table { border-collapse: collapse; width: 100%; font-size: 0.85em; margin-top: 0.6em; }
th, td { border: 1px solid #d0d7de; padding: 4px 7px; text-align: left; vertical-align: top; }
th { background: #f6f8fa; position: sticky; top: 0; }
td.delta { font-weight: 600; } td.pid { font-family: monospace; white-space: nowrap; }
tr.convrow td { background: #fbfbfb; }
details summary { cursor: pointer; color: #0969da; font-size: 0.9em; }
.conv { margin-top: 6px; }
.turn { margin: 6px 0; padding: 6px 10px; border-radius: 6px; border: 1px solid #e1e4e8; }
.turn-user { background: #eef6ff; } .turn-assistant { background: #f2fdf4; } .turn-system { background: #fff8e6; }
.turn .role { font-weight: 700; font-size: 0.78em; text-transform: uppercase; color: #57606a; }
.turn pre { white-space: pre-wrap; word-wrap: break-word; margin: 4px 0 0 0; font-family: inherit; font-size: 0.95em; }
.trunc { color: #b35900; font-size: 0.8em; font-style: italic; }
.miss { color: #b35900; font-style: italic; }
"""


def main() -> None:
    div = json.loads(DIVERGENCE_JSON.read_text())
    store = _load_prefix_store(_fetch_prefix_store())

    sections = []
    counts: dict[str, tuple[int, int]] = {}
    for trait in TRAITS:
        block = div["traits"][trait]
        rows = _union_rows(block)
        n_found = sum(1 for r in rows if r["prefix_id"] in store)
        counts[trait] = (len(rows), n_found)
        body = "\n".join(_row_html(i + 1, r, store) for i, r in enumerate(rows))
        agree = block.get("arms_agreement_r_pred_pe_vs_pred_ac")
        sections.append(
            f"<h2>{trait} — top arm-disagreement prefixes (n={len(rows)}; arms' predictions agree at r={agree:.2f} overall)</h2>\n"
            "<table><thead><tr><th>#</th><th>prefix_id</th><th>stratum</th><th>source</th><th>topic</th>"
            "<th>user turns</th><th>ctx tokens (median)</th><th>judge mean ± sd</th>"
            "<th>pred prefix-end</th><th>pred avg-context</th><th>Δ (pe − ac)</th><th>closer arm</th><th>in list(s)</th></tr></thead>"
            f"<tbody>{body}</tbody></table>"
        )

    provenance = (
        "<div class='note'><b>What is shown.</b> Union of the three ranked outlier lists banked in "
        "<code>eval_results/issue_1092/inline_prefixend_monitoring/divergence_analysis.json</code> "
        "(top-12 by |Δ| + top-8 where each arm wins, deduped), ranked by |Δ|. "
        "<b>pred prefix-end</b> / <b>pred avg-context</b>: held-out grouped-5-fold GCV dual-ridge predictions of the prefix's mean judge score, "
        "from the prefix-end state (one forward pass, pre-query) vs the context-end state averaged over the prefix's 48 queries. "
        "<b>judge mean</b>: measured mean graded judge score (claude-sonnet-4-5-20250929, 0–100, 5 draws temp 1.0 mean-aggregated, drop-never-coerce) "
        "over the prefix's 48 own-policy greedy answers (Qwen-2.5-7B-Instruct, layer-14 reads). "
        "<b>Δ</b> = prefix-end prediction − averaged-context prediction. <b>closer arm</b>: sign of the differential absolute error. "
        "Conversations are shown VERBATIM (HTML-escaped) from <code>corpus/prefix_store.jsonl</code>; "
        f"turns longer than {TURN_CHAR_CAP:,} chars are truncated with an inline marker. "
        "Corpus is unscreened real WildChat/LMSYS text plus constructed battery conditions.</div>"
    )

    doc = (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>#1092 prefix-end vs averaged-context — top disagreement prefixes</title>"
        f"<style>{CSS}</style></head><body>"
        "<h1>#1092 — top prefixes where the prefix-end and averaged-context reads disagree</h1>"
        f"{provenance}\n" + "\n".join(sections) + "</body></html>"
    )

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc)
    for trait, (n, n_found) in counts.items():
        print(f"{trait}: {n} outlier prefixes, {n_found} with text in prefix_store")
    print(f"wrote {OUT} ({OUT.stat().st_size:,} bytes)")


if __name__ == "__main__":
    main()
