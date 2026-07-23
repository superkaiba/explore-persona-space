"""Readable HTML dashboard of the highest/lowest-projecting LMSYS contexts on each
persona's pre-image direction (user ask, 2026-07-22: "list them on a dashboard in
an easily readable format").

Sibling of the other issue #779 dashboards (issue779_dashboard_{rig,completions,
corpora}.py). Reads the committed
eval_results/issue_779/pinv_topk_contexts/pinv_topk_contexts.json and writes a
self-contained page to experiments/dashboards/issue779_pinv_topbottom_contexts.html.
No data recomputation, no network. 0 GPU-h.
"""

from __future__ import annotations

import html
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "eval_results" / "issue_779" / "pinv_topk_contexts" / "pinv_topk_contexts.json"
OUT = REPO / "experiments" / "dashboards" / "issue779_pinv_topbottom_contexts.html"

TRAITS = ["evil", "sycophancy", "hallucination"]
TRAIT_META = {
    "evil": {"layer": 14, "color": "#9a3b2e"},
    "sycophancy": {"layer": 26, "color": "#2f6f6a"},
    "hallucination": {"layer": 17, "color": "#8a5a1a"},
}
N = 10

STYLE = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,600;9..144,700&family=Spline+Sans+Mono:wght@400;500;600&family=Newsreader:opsz,wght@6..72,400;6..72,500&display=swap');
:root{--ink:#211e19;--ink-soft:#534c41;--paper:#f3eee4;--card:#fbf8f1;--line:#ddd3c1;--line-2:#cabfa8;
  --shadow:0 1px 0 rgba(33,30,25,.03),0 14px 30px -22px rgba(33,30,25,.55);}
*{box-sizing:border-box}
body{margin:0;background:var(--paper);color:var(--ink);font-family:"Newsreader",Georgia,serif;
  font-size:17px;line-height:1.55;-webkit-font-smoothing:antialiased}
.wrap{max-width:1220px;margin:0 auto;padding:0 26px 120px}
.mono{font-family:"Spline Sans Mono",ui-monospace,monospace}
header.masthead{border-bottom:2.5px solid var(--ink);padding:46px 0 20px}
.kicker{font-family:"Spline Sans Mono",monospace;font-size:11px;letter-spacing:.26em;
  text-transform:uppercase;color:var(--ink-soft);display:flex;gap:12px;flex-wrap:wrap}
h1.title{font-family:"Fraunces",Georgia,serif;font-weight:600;font-size:clamp(30px,5vw,50px);
  line-height:1.02;letter-spacing:-.018em;margin:.28em 0 .16em}
.dek{font-size:18px;max-width:74ch;color:var(--ink-soft);margin:.2em 0 0}
.dek code{font-family:"Spline Sans Mono",monospace;font-size:.85em;background:var(--card);
  padding:1px 5px;border-radius:4px;border:1px solid var(--line)}
.callout{margin:24px 0 6px;background:#fff7df;border:1px solid var(--line-2);
  border-left:4px solid #ffcf3f;border-radius:9px;padding:14px 18px;font-size:15.5px;color:var(--ink-soft)}
.callout b{color:var(--ink)}
.famsec{margin:46px 0 8px}
.famhead{display:flex;align-items:baseline;gap:14px;padding:10px 0 8px;
  border-bottom:2px solid var(--c);flex-wrap:wrap}
.famhead h2{font-family:"Fraunces",serif;font-weight:600;font-size:27px;margin:0;color:var(--c)}
.famhead .fct{font-family:"Spline Sans Mono",monospace;font-size:12px;color:var(--ink-soft)}
.grouplbl{font-family:"Spline Sans Mono",monospace;font-size:10.5px;letter-spacing:.16em;
  text-transform:uppercase;color:var(--c);margin:20px 0 9px;padding-bottom:3px;
  border-bottom:1px solid var(--line)}
.rgrid{display:grid;grid-template-columns:repeat(auto-fill,minmax(400px,1fr));gap:12px}
.rcard{background:var(--card);border:1px solid var(--line-2);border-left:4px solid var(--c);
  border-radius:10px;box-shadow:var(--shadow);padding:11px 14px 12px;display:flex;flex-direction:column}
.rhead{display:flex;align-items:center;gap:10px;margin-bottom:7px;flex-wrap:wrap}
.rank{font-family:"Spline Sans Mono",monospace;font-size:11px;color:var(--ink-soft)}
.score{font-family:"Fraunces",serif;font-weight:700;font-size:15px;padding:1px 11px;
  border-radius:999px;line-height:1.35}
.score.hi{background:var(--c);color:#fff8ef}
.score.lo{background:transparent;color:var(--c);border:1.5px solid var(--c)}
.theme{font-family:"Spline Sans Mono",monospace;font-size:9.5px;letter-spacing:.1em;
  text-transform:uppercase;color:var(--ink-soft);background:var(--paper);border:1px solid var(--line);
  border-radius:5px;padding:1px 7px}
.flag{color:#9a3b2e;border-color:#e0b6a8;background:#f7e6df}
.ptext{white-space:pre-wrap;word-break:break-word;font-size:14.5px;line-height:1.5;
  background:var(--paper);border:1px solid var(--line);border-radius:7px;padding:9px 11px}
.ptext.ph{font-style:italic;color:var(--ink-soft)}
footer.foot{margin-top:60px;padding-top:22px;border-top:1px solid var(--line);
  font-family:"Spline Sans Mono",monospace;font-size:11.5px;color:var(--ink-soft);line-height:1.85}
footer.foot b{color:var(--ink)}
footer.foot a{color:var(--ink-soft)}
@media(max-width:820px){.rgrid{grid-template-columns:1fr}}
"""


def esc(s: str) -> str:
    return html.escape(s, quote=False)


def card(rec: dict, rank: int, hi: bool) -> str:
    txt = rec.get("text")
    if txt:
        body = f'<div class="ptext">{esc(" ".join(txt.split()))}</div>'
        theme = f'<span class="theme">{esc(rec.get("theme", "other"))}</span>'
    else:
        flag = rec.get("flagged") or "flagged"
        body = f'<div class="ptext ph">[{esc(flag)} roleplay row — categorized, not quoted]</div>'
        theme = f'<span class="theme flag">{esc(flag)}</span>'
    cls = "hi" if hi else "lo"
    return (
        '<div class="rcard">'
        '<div class="rhead">'
        f'<span class="rank">#{rank}</span>'
        f'<span class="score {cls}">{rec["score"]:+.1f}</span>'
        f"{theme}"
        "</div>"
        f"{body}"
        "</div>"
    )


def trait_section(trait: str, data: dict) -> str:
    meta = TRAIT_META[trait]
    tb = data["traits"][trait]["lmsys_topbottom"]["w_pinv_kstar"]
    eg = data["traits"][trait]["eval_grid"]
    rho = eg["w_pinv_kstar"]["spearman_proj_vs_judgescore"]
    top = "".join(card(r, i + 1, True) for i, r in enumerate(tb["top"][:N]))
    bot = "".join(card(r, i + 1, False) for i, r in enumerate(tb["bottom"][:N]))
    return (
        f'<section class="famsec" style="--c:{meta["color"]}">'
        '<div class="famhead">'
        f"<h2>{esc(trait)}</h2>"
        f'<span class="fct">read-out layer L{meta["layer"]} &middot; '
        f"eval-grid Spearman(projection, judged score) = {rho:.2f}</span>"
        "</div>"
        f'<div class="grouplbl">Highest-projecting — most trait-eliciting (top {N} of 5,000)</div>'
        f'<div class="rgrid">{top}</div>'
        f'<div class="grouplbl">Lowest-projecting (bottom {N} of 5,000)</div>'
        f'<div class="rgrid">{bot}</div>'
        "</section>"
    )


def main() -> int:
    data = json.loads(SRC.read_text())
    meta = data["metadata"]
    body = "".join(trait_section(t, data) for t in TRAITS)
    parent = esc(str(meta.get("parent_run", "")))
    model = esc(str(meta.get("model_id", "")))
    lmsys = esc(str(meta.get("lmsys_revision", ""))[:12])
    doc = (
        '<!doctype html><html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width,initial-scale=1">'
        "<title>Issue #779 — pre-image top/bottom LMSYS contexts</title>"
        f'<style>{STYLE}</style></head><body><div class="wrap">'
        '<header class="masthead">'
        '<div class="kicker"><span>Issue #779</span><span>persona-vector pre-image</span>'
        "<span>LMSYS &middot; n=5,000</span></div>"
        '<h1 class="title">Which real prompts most (and least) project onto each '
        "persona's pre-image direction</h1>"
        '<p class="dek">Each of 5,000 real first-turn <span class="mono">lmsys-chat-1m</span> '
        "prompts is projected onto the persona vector's pre-image "
        '<span class="mono">w = M&#8314;r_B</span> (the rank-truncated min-norm context that '
        'the context&rarr;answer map <span class="mono">M</span> sends to the persona '
        "direction). Sorted by that projection; the top rows are the contexts the direction "
        "ranks most trait-eliciting, the bottom rows the least.</p>"
        '<div class="callout"><b>How to read it.</b> The score is the raw projection '
        '<span class="mono">&#10216;c_std, w&#10217;</span> (standardized context onto the '
        "pre-image). Explicit / jailbreak-shaped rows are stored as category placeholders, "
        "never quoted. This is the naturalistic-corpus read; the crafted eval-grid read "
        "(where the pre-image matches the raw persona vector at Spearman 0.78 / 0.77 / 0.50) "
        "is the quantitative companion.</div>"
        f"{body}"
        '<footer class="foot">'
        '<div><b>Source.</b> <span class="mono">eval_results/issue_779/pinv_topk_contexts/'
        'pinv_topk_contexts.json</span> (key <span class="mono">traits.&lt;trait&gt;.'
        "lmsys_topbottom.w_pinv_kstar</span>).</div>"
        '<div><b>Generator.</b> <span class="mono">scripts/issue779_dashboard_pinv_topbottom.py'
        f'</span> &middot; parent run <span class="mono">{parent}</span> &middot; '
        f'model <span class="mono">{model}</span> &middot; '
        f'LMSYS rev <span class="mono">{lmsys}</span>.</div>'
        '<div><b>Companion figures.</b> <span class="mono">figures/issue_779/'
        'pinv_topk_lmsys_topbottom.png</span> (bar view) &middot; <span class="mono">'
        "figures/issue_779/pinv_topk_lmsys_themes.png</span> (theme composition).</div>"
        "</footer></div></body></html>"
    )
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(doc, encoding="utf-8")
    print(f"[dashboard] wrote {OUT} ({len(doc)} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
