"""Issue #1482 context-extremes round — top/worst-100 predicted contexts dashboard (D1).

For each of the three input-state arms (context vector / prefix end state /
query only; all predicting the SAME full-context answer state, L19 ridge,
#1738 multi-turn holdout n=9,941) rank the held-out contexts by per-context
normalized error (nerr = ||v_hat-v||^2/||v-mu_eval||^2, the #1482 convention)
and render the best-100 and worst-100 into a self-contained HTML dashboard at
dashboard/public/context-extremes-1482.html (existing 1482 dashboard style).

Also writes:
  - eval_results/issue_1482/context_extremes/context_extremes.json — selection
    lists + per-set composition stats (NO raw text; texts stay in the dashboard
    display layer, excerpt-capped at the judge instrument's caps).
  - data/issue_1482/context_extremes_scratch/blind/<arm>_group{A,B}.md +
    key.json — blinded top-vs-worst text bundles for the D2 qualitative read
    (assignment randomized; the key is written for post-hoc unblinding only).

Inputs are banked: the three-arm per-context summary CSV
(eval_results/issue_1738/bare_query/percontext_summary_L19_ridge.csv) and the
holdout text cache built by scripts/issue1482_collect_holdout_texts.py.
0 GPU; CPU-only. Raw corpus text is never logged to stdout.
"""

from __future__ import annotations

import csv
import html
import json
import secrets
import statistics
import sys
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

CSV = (
    PROJECT_ROOT
    / "eval_results"
    / "issue_1738"
    / "bare_query"
    / ("percontext_summary_L19_ridge.csv")
)
TEXTS = PROJECT_ROOT / "data" / "issue_1482" / "context_extremes_scratch" / "judge_texts.jsonl"
OUT_HTML = PROJECT_ROOT / "dashboard" / "public" / "context-extremes-1482.html"
OUT_JSON = (
    PROJECT_ROOT / "eval_results" / "issue_1482" / "context_extremes" / "context_extremes.json"
)
BLIND_DIR = PROJECT_ROOT / "data" / "issue_1482" / "context_extremes_scratch" / "blind"
N_EXTREME = 100
# display caps == the judge instrument's excerpt caps (labels.json excerpt_caps)
CAP_LAST_USER = 1200
CAP_HISTORY = 800
CAP_RESPONSE = 1000
CAP_BLIND_LAST_USER = 800
CAP_BLIND_HISTORY = 400

ARMS = (
    ("context", "nerr_context_L19_ridge", "Context vector"),
    ("prefix", "nerr_prefix_L19_ridge", "Prefix end state"),
    ("bare", "nerr_bare_L19_ridge", "Query only"),
)


def _cap(s: str, n: int) -> str:
    s = s or ""
    return s if len(s) <= n else s[:n] + " …[truncated]"


def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def _composition(rows: list[dict]) -> dict:
    return {
        "n": len(rows),
        "topic": dict(Counter(r["topic"] or "(unlabeled)" for r in rows).most_common()),
        "language": dict(Counter(r["language"] or "(unlabeled)" for r in rows).most_common()),
        "format": dict(Counter(r["format"] or "(unlabeled)" for r in rows).most_common()),
        "median_last_user_chars": statistics.median(len(r["_t"]["last_user"]) for r in rows),
        "median_history_chars": statistics.median(len(r["_t"]["history_tail"]) for r in rows),
        "median_response_chars": statistics.median(len(r["_t"]["response"]) for r in rows),
        "n_multiturn_history": sum(1 for r in rows if len(r["_t"]["history_tail"]) > 0),
    }


def main() -> None:
    with open(CSV, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    texts: dict[int, dict] = {}
    with open(TEXTS, encoding="utf-8") as f:
        for line in f:
            t = json.loads(line)
            texts[int(t["ci"])] = t
    missing = [r["ci"] for r in rows if int(r["ci"]) not in texts]
    assert not missing, f"missing texts for {len(missing)} cis"
    for r in rows:
        r["_t"] = texts[int(r["ci"])]

    selection: dict[str, dict] = {}
    comp: dict[str, dict] = {}
    for arm, col, _label in ARMS:
        srt = sorted(rows, key=lambda r: (float(r[col]), int(r["ci"])))
        top, worst = srt[:N_EXTREME], srt[-N_EXTREME:][::-1]
        selection[arm] = {"top": top, "worst": worst}
        comp[arm] = {"top": _composition(top), "worst": _composition(worst)}

    # ── blinded bundles for the D2 qualitative read ──────────────────────────
    BLIND_DIR.mkdir(parents=True, exist_ok=True)
    key: dict[str, dict] = {}
    for arm, _col, _label in ARMS:
        a_is_top = secrets.randbelow(2) == 0
        key[arm] = {"A": "top" if a_is_top else "worst", "B": "worst" if a_is_top else "top"}
        for gname in ("A", "B"):
            kind = key[arm][gname]
            # shuffle display order so rank carries no signal
            items = sorted(selection[arm][kind], key=lambda r: int(r["ci"]) * 2654435761 % 2**32)
            lines = [
                f"# Group {gname} — 100 conversation excerpts",
                "",
                "Each item: final user message (capped), preceded by a capped tail of",
                "the prior conversation when one exists.",
                "",
            ]
            for i, r in enumerate(items, 1):
                t = r["_t"]
                lines.append(f"## item {i}")
                if t["history_tail"]:
                    lines.append(f"[history tail] {_cap(t['history_tail'], CAP_BLIND_HISTORY)}")
                lines.append(f"[final user message] {_cap(t['last_user'], CAP_BLIND_LAST_USER)}")
                lines.append("")
            (BLIND_DIR / f"{arm}_group{gname}.md").write_text("\n".join(lines), encoding="utf-8")
    (BLIND_DIR / "key.json").write_text(json.dumps(key, indent=1))

    # ── eval_results JSON (no raw text) ──────────────────────────────────────
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(
        json.dumps(
            {
                "generated_utc": datetime.now(UTC).isoformat(),
                "corpus": "#1738 multi-turn holdout, n=9,941, L19 ridge; all arms score "
                "the same full-context answer target (bitwise-identical, "
                "twoway_residual.json design)",
                "metric": "nerr = ||v_hat-v||^2/||v-mu_eval||^2 per context; "
                "per-context R^2 = 1 - nerr",
                "sources": [
                    str(CSV.relative_to(PROJECT_ROOT)),
                    "eval_results/issue_1738/judge_labels/labels.json",
                    "scripts/issue1482_collect_holdout_texts.py (text cache)",
                ],
                "selection": {
                    arm: {
                        kind: [
                            {
                                "rank": i + 1,
                                "ci": int(r["ci"]),
                                "nerr": float(r[col]),
                                "topic": r["topic"],
                                "language": r["language"],
                                "format": r["format"],
                            }
                            for i, r in enumerate(selection[arm][kind])
                        ]
                        for kind in ("top", "worst")
                    }
                    for arm, col, _label in ARMS
                },
                "composition": comp,
            },
            indent=1,
        )
    )

    # ── dashboard HTML ───────────────────────────────────────────────────────
    css = """
:root { --fg:#16181d; --mut:#5b6270; --line:#e3e6ec; --bg:#fbfbfd; --card:#fff; }
* { box-sizing:border-box; }
body { margin:0; padding:28px 22px 60px; background:var(--bg); color:var(--fg);
  font:15px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Helvetica,Arial,sans-serif; }
.wrap { max-width:1240px; margin:0 auto; }
h1 { font-size:22px; margin:0 0 6px; letter-spacing:-0.01em; }
h2 { font-size:17px; margin:34px 0 4px; padding-top:14px; border-top:1px solid var(--line); }
h3 { font-size:15px; margin:18px 0 6px; }
p, li { color:var(--mut); font-size:13.5px; }
.head { background:var(--card); border:1px solid var(--line); border-radius:10px;
  padding:16px 18px; margin-bottom:10px; }
.head p { margin:6px 0; } .head b { color:var(--fg); }
.warn { background:#fff7ed; border:1px solid #f6d6bd; border-radius:9px; padding:12px 14px;
  margin:10px 0; } .warn p { color:#8a4a12; margin:5px 0; }
.nav a { margin-right:14px; font-size:13px; color:#1b4fd8; text-decoration:none; }
.ctl { margin:12px 0 8px; font-size:13px; color:var(--mut); }
.ctl input { font:inherit; padding:3px 7px; border:1px solid var(--line); border-radius:6px;
  width:280px; }
table { border-collapse:collapse; width:100%; background:var(--card); font-size:13px; }
th, td { padding:5px 9px; border-bottom:1px solid var(--line); vertical-align:top; }
th { position:sticky; top:0; background:#f4f6fa; text-align:left; font-weight:600;
  color:var(--fg); white-space:nowrap; }
td.n, th.n { text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }
td.hl { font-weight:700; }
.tag { font-size:11px; padding:1px 7px; border-radius:9px; border:1px solid var(--line);
  white-space:nowrap; background:#f4f6fa; }
details { margin:2px 0; } summary { cursor:pointer; color:#1b4fd8; font-size:12px; }
.txt { white-space:pre-wrap; word-break:break-word; color:#2c313b; font-size:12.5px; }
.blk { margin:4px 0; } .blk b { font-size:11.5px; color:var(--mut);
  text-transform:uppercase; letter-spacing:.04em; }
"""
    js = """
function filt(inp) {
  const q = inp.value.toLowerCase();
  const tbl = document.getElementById(inp.dataset.tbl);
  for (const tr of tbl.tBodies[0].rows)
    tr.style.display = tr.textContent.toLowerCase().includes(q) ? '' : 'none';
}
"""

    def _row(r: dict, rank: int, col: str) -> str:
        t = r["_t"]
        nerrs = {a: float(r[c]) for a, c, _l in ARMS}
        arm_of_col = next(a for a, c, _l in ARMS if c == col)
        cells = [f"<td class='n'>{rank}</td>", f"<td class='n'>{int(r['ci'])}</td>"]
        for a, _c, _l in ARMS:
            hl = " hl" if a == arm_of_col else ""
            cells.append(f"<td class='n{hl}'>{nerrs[a]:.3f}</td>")
        cells.append(f"<td class='n'>{1 - nerrs[arm_of_col]:.3f}</td>")
        cells.append(
            f"<td><span class='tag'>{_esc(r['topic'] or '—')}</span></td>"
            f"<td>{_esc(r['language'] or '—')}</td><td>{_esc(r['format'] or '—')}</td>"
        )
        blocks = []
        if t["history_tail"]:
            blocks.append(
                "<div class='blk'><b>history tail</b><div class='txt'>"
                + _esc(_cap(t["history_tail"], CAP_HISTORY))
                + "</div></div>"
            )
        blocks.append(
            "<div class='blk'><b>model response (under full context)</b><div class='txt'>"
            + _esc(_cap(t["response"], CAP_RESPONSE))
            + "</div></div>"
        )
        cells.append(
            "<td><div class='txt'>"
            + _esc(_cap(t["last_user"], CAP_LAST_USER))
            + "</div><details><summary>history + response</summary>"
            + "".join(blocks)
            + "</details></td>"
        )
        return "<tr>" + "".join(cells) + "</tr>"

    def _table(arm: str, col: str, kind: str) -> str:
        tid = f"tbl-{arm}-{kind}"
        head = (
            "<tr><th class='n'>#</th><th class='n'>ci</th>"
            "<th class='n'>nerr ctx</th><th class='n'>nerr prefix</th>"
            "<th class='n'>nerr query</th><th class='n'>R&sup2; (this arm)</th>"
            "<th>topic</th><th>lang</th><th>format</th>"
            "<th>final user message (capped; expand for history + response)</th></tr>"
        )
        body = "".join(_row(r, i + 1, col) for i, r in enumerate(selection[arm][kind]))
        return (
            f"<div class='ctl'>filter: <input data-tbl='{tid}' oninput='filt(this)'></div>"
            f"<table id='{tid}'><thead>{head}</thead><tbody>{body}</tbody></table>"
        )

    now = datetime.now(UTC).strftime("%Y-%m-%d %H:%M UTC")
    parts = [
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>",
        "<meta name='viewport' content='width=device-width,initial-scale=1'>",
        "<title>Issue 1482 &mdash; best/worst predicted contexts (multi-turn holdout)</title>",
        f"<style>{css}</style><script>{js}</script></head><body><div class='wrap'>",
        "<h1>Best-100 / worst-100 predicted contexts, per arm</h1>",
        "<div class='head'>",
        "<p><b>What this is.</b> The 9,941 held-out contexts of the <b>#1738 MULTI-TURN "
        "corpus</b> (real conversations), ranked per arm by per-context normalized error "
        "<b>nerr = &#8214;v&#770;&minus;v&#8214;&sup2; / &#8214;v&minus;&mu;&#8214;&sup2;</b> "
        "of the L19 ridge map (per-context R&sup2; = 1 &minus; nerr). All three arms predict "
        "the <b>same target</b>: the mean answer state generated under the full context "
        "(bitwise-identical targets across arms).</p>",
        "<p><b>Arms.</b> <b>Context vector</b> (full context &rarr; answer), <b>Prefix end "
        "state</b> (prefix only &rarr; answer), <b>Query only</b> (bare final user query "
        "&rarr; answer).</p>",
        "<p><b>Labels.</b> Judged categories from the #1738 holdout categorization "
        "(claude-sonnet-4-5, same rubric as the single-turn run &mdash; see the "
        "<a href='judge-prompts-1482.html'>judge-prompts dashboard</a>); 16 of 9,941 "
        "contexts are unlabeled (judge drops) and shown as &mdash;.</p>",
        f"<p>Generated {now} by scripts/issue1482_context_extremes_dashboard.py; "
        "selection lists + composition stats: "
        "eval_results/issue_1482/context_extremes/context_extremes.json.</p>",
        "</div>",
        "<div class='warn'><p><b>Corpus caveat.</b> These lists are the #1738 MULTI-TURN "
        "holdout (n=9,941). Other results in the writeup use the SINGLE-TURN corpus "
        "(n=20,000) &mdash; a different pool; do not mix the two. Excerpts are capped "
        "(final user message 1,200 chars, history 800, response 1,000 &mdash; the judge "
        "instrument's caps). Category <i>other</i> is a heterogeneous catch-all.</p></div>",
        "<p class='nav'>",
    ]
    for arm, _col, label in ARMS:
        parts.append(
            f"<a href='#{arm}-top'>{_esc(label)}: best</a>"
            f"<a href='#{arm}-worst'>{_esc(label)}: worst</a>"
        )
    parts.append("</p>")
    for arm, col, label in ARMS:
        for kind, klabel in (("top", "best-predicted"), ("worst", "worst-predicted")):
            c = comp[arm][kind]
            parts.append(
                f"<h2 id='{arm}-{kind}'>{_esc(label)} &mdash; {klabel} 100</h2>"
                f"<p>top topics: "
                + ", ".join(f"{_esc(k)} ({v})" for k, v in list(c["topic"].items())[:5])
                + " &middot; top languages: "
                + ", ".join(f"{_esc(k)} ({v})" for k, v in list(c["language"].items())[:5])
                + f" &middot; median final-user-message length {c['median_last_user_chars']:.0f}"
                " chars &middot; "
                f"{c['n_multiturn_history']}/100 have prior turns</p>"
            )
            parts.append(_table(arm, col, kind))
    parts.append("</div></body></html>")
    OUT_HTML.write_text("".join(parts), encoding="utf-8")

    print(f"[dash] wrote {OUT_HTML} ({OUT_HTML.stat().st_size / 1e6:.2f} MB)")
    print(f"[dash] wrote {OUT_JSON}")
    print(f"[dash] blinded bundles: {BLIND_DIR} (key.json written, not printed)")
    for arm, _col, label in ARMS:
        for kind in ("top", "worst"):
            c = comp[arm][kind]
            tt = ", ".join(f"{k}:{v}" for k, v in list(c["topic"].items())[:4])
            ll = ", ".join(f"{k}:{v}" for k, v in list(c["language"].items())[:4])
            print(
                f"[comp] {arm:8s} {kind:5s} topics[{tt}] langs[{ll}] "
                f"hist>{0}:{c['n_multiturn_history']}/100 "
                f"med_query={c['median_last_user_chars']:.0f}ch "
                f"med_resp={c['median_response_chars']:.0f}ch"
            )


if __name__ == "__main__":
    main()
