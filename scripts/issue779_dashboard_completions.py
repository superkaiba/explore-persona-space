#!/usr/bin/env python3
"""Reproducible generator for the issue #779 eval-rollout completions dashboard.

Builds a single self-contained HTML page (matching the house
`experiments/dashboards/completions.html` style) that shows the model's ACTUAL
eval rollouts with their judge scores, for each trait x eval condition.

Data source (HF dataset repo, revision main):
  superkaiba1/explore-persona-space-data, prefix issue779_monitoring/
    raw_completions/{trait}_{cond}_seed42.json   -- rollout text per (qi, ri)
    analysis_tensors/pass_a/{trait}__{cond}.json  -- per-rollout judge scores
    artifacts/{sycophancy,hallucination}.json     -- generated eval-question banks
  (evil eval questions come from scripts.issue779_common.EVIL_ARTIFACTS)

Subsetting rule: for each (trait, condition) cell, render the SUBSAMPLE of the
3 highest- and 3 lowest-scoring rollouts by the per-rollout 5-draw-mean judge
score (falls back to 2+2 if the full page would exceed the size cap). Full
per-cell JSONs are linked on HF.

CONTEXT-HYGIENE: all completion text is read from JSON and written into HTML
inside this script; no raw completion text is printed to stdout. The verify
pass prints only structural digests (counts, score histograms, byte sizes).
"""

from __future__ import annotations

import argparse
import html
import importlib.util
import json
import math
import statistics
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue779_monitoring"
HF_REVISION = "main"
HF_TREE_URL = (
    "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/"
    "tree/main/issue779_monitoring/raw_completions"
)
HF_BLOB_BASE = (
    "https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/"
    "blob/main/issue779_monitoring"
)

TRAITS = ("evil", "sycophancy", "hallucination")
SYS_CONDS = tuple(f"sys{i}" for i in range(8))
SHOT_COUNTS = (0, 5, 10, 15, 20)
SHOT_CONDS = tuple(f"shot{k}" for k in SHOT_COUNTS)
ALL_CONDS = SYS_CONDS + SHOT_CONDS

# Per-trait paper-palette accent (harmonises with the house cream/ink palette).
TRAIT_ACCENT = {
    "evil": "#9a3b2e",  # brick red
    "sycophancy": "#2f6f6a",  # teal (house exemplar accent)
    "hallucination": "#6a4c93",  # muted violet
}
HARMFUL_TRAITS = {"evil"}
PAGE_SIZE_CAP_BYTES = 4 * 1024 * 1024


# ---------------------------------------------------------------------------
# Load pinned artifacts from the in-repo module (no API call).
# ---------------------------------------------------------------------------
def _load_common_module():
    spec = importlib.util.spec_from_file_location(
        "issue779_common", REPO_ROOT / "scripts" / "issue779_common.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# HF download (idempotent; only the small JSONs, never the *_cx.pt tensors).
# ---------------------------------------------------------------------------
def download_inputs(download_dir: Path, skip_download: bool) -> Path:
    from huggingface_hub import hf_hub_download

    rels = []
    for trait in TRAITS:
        for cond in ALL_CONDS:
            rels.append(f"{HF_PREFIX}/raw_completions/{trait}_{cond}_seed42.json")
            rels.append(f"{HF_PREFIX}/analysis_tensors/pass_a/{trait}__{cond}.json")
    for trait in ("sycophancy", "hallucination"):
        rels.append(f"{HF_PREFIX}/artifacts/{trait}.json")

    download_dir.mkdir(parents=True, exist_ok=True)
    for rel in rels:
        local = download_dir / rel
        if skip_download and local.exists():
            continue
        hf_hub_download(
            HF_DATA_REPO,
            rel,
            repo_type="dataset",
            revision=HF_REVISION,
            local_dir=str(download_dir),
        )
    return download_dir


# ---------------------------------------------------------------------------
# Question banks per trait (qi -> question text).
# ---------------------------------------------------------------------------
def load_eval_questions(common_mod, download_dir: Path) -> dict[str, list[str]]:
    out = {"evil": list(common_mod.EVIL_ARTIFACTS["eval_questions"])}
    for trait in ("sycophancy", "hallucination"):
        p = download_dir / HF_PREFIX / "artifacts" / f"{trait}.json"
        d = json.loads(p.read_text())
        out[trait] = list(d["eval_questions"])
    return out


# ---------------------------------------------------------------------------
# Cell loading + scoring.
# ---------------------------------------------------------------------------
def _finite(x) -> bool:
    return isinstance(x, (int, float)) and not (isinstance(x, bool)) and math.isfinite(float(x))


def load_cell(download_dir: Path, trait: str, cond: str):
    """Return a dict for one (trait, cond) cell, or None if a file is missing.

    Merges raw responses with judge scores keyed {trait}__{cond}__{qi:05d}__{ri:02d}.
    """
    raw_p = download_dir / HF_PREFIX / "raw_completions" / f"{trait}_{cond}_seed42.json"
    sc_p = download_dir / HF_PREFIX / "analysis_tensors" / "pass_a" / f"{trait}__{cond}.json"
    if not raw_p.exists() or not sc_p.exists():
        return None

    raw = json.loads(raw_p.read_text())
    sc = json.loads(sc_p.read_text())

    resp = {(r["qi"], r["ri"]): r["response"] for r in raw["rollouts"]}
    n_rollouts = len(raw["rollouts"])
    judge_dropped = int(sc.get("judge_dropped", 0))

    scores = sc.get("judge_scores", {}) or {}
    rows = []  # (qi, ri, score, response)
    for key, val in scores.items():
        # key format: {trait}__{cond}__{qi:05d}__{ri:02d}
        parts = key.split("__")
        if len(parts) < 4:
            continue
        qi = int(parts[-2])
        ri = int(parts[-1])
        if not _finite(val):
            continue
        text = resp.get((qi, ri))
        if text is None:
            continue
        rows.append((qi, ri, float(val), text))

    scored = [r[2] for r in rows]
    stats = {
        "n_rollouts": n_rollouts,
        "n_scored": len(rows),
        "judge_dropped": judge_dropped,
        "mean": statistics.fmean(scored) if scored else None,
        "min": min(scored) if scored else None,
        "max": max(scored) if scored else None,
        "pct_over50": (100.0 * sum(1 for s in scored if s > 50) / len(scored)) if scored else None,
    }
    return {
        "trait": trait,
        "cond": cond,
        "mode": raw.get("mode"),
        "n_shot": raw.get("n_shot"),
        "git_commit": (raw.get("metadata") or {}).get("git_commit"),
        "rows": rows,
        "stats": stats,
    }


def select_top_bottom(rows, per_side: int):
    """Return (top_rows, bottom_rows) by score, de-duplicated on (qi, ri)."""
    ordered = sorted(rows, key=lambda r: r[2])
    bottom = ordered[:per_side]
    top = list(reversed(ordered[-per_side:]))
    seen = set()
    top_u = []
    for r in top:
        if (r[0], r[1]) not in seen:
            seen.add((r[0], r[1]))
            top_u.append(r)
    bottom_u = [r for r in bottom if (r[0], r[1]) not in seen]
    return top_u, bottom_u


# ---------------------------------------------------------------------------
# Condition labels.
# ---------------------------------------------------------------------------
def cond_label(cond: str) -> tuple[str, str]:
    """Return (human label, sub-note)."""
    if cond.startswith("sys"):
        i = int(cond[3:])
        strength = ""
        if i == 0:
            strength = " (strongest trait induction)"
        elif i == 7:
            strength = " (plain helpful assistant)"
        return f"System-prompt monitoring — prompt {i + 1} of 8" + strength, cond
    k = int(cond[4:])
    if k == 0:
        return "Many-shot monitoring — 0-shot (no exemplars, no system prompt)", cond
    return (
        f"Many-shot monitoring — {k}-shot ({k} trait-exhibiting exemplars, no system prompt)",
        cond,
    )


# ---------------------------------------------------------------------------
# HTML rendering.
# ---------------------------------------------------------------------------
def esc(s: str) -> str:
    return html.escape(s if s is not None else "")


def score_pill(score: float) -> str:
    over = score > 50
    cls = "score over" if over else "score under"
    return f'<span class="{cls}">{score:.1f}</span>'


def render_card(qi: int, ri: int, score: float, question: str, completion: str) -> str:
    return (
        '<div class="rcard">'
        '<div class="rhead">'
        f"{score_pill(score)}"
        f'<span class="rid">q{qi} &middot; rollout {ri}</span>'
        "</div>"
        '<div class="msg"><span class="role user">Question (q{qi})</span>'
        f'<div class="body">{esc(question)}</div></div>'
        '<div class="msg"><span class="role assistant">Model completion</span>'
        f'<div class="body ans clamped">{esc(completion)}</div>'
        '<button class="toggle" type="button">show full &darr;</button></div>'
        "</div>"
    ).replace("{qi}", str(qi))


def render_group(title: str, rows, questions: list[str]) -> str:
    if not rows:
        return ""
    cards = "".join(
        render_card(qi, ri, sc, questions[qi] if qi < len(questions) else f"(question {qi})", text)
        for (qi, ri, sc, text) in rows
    )
    return f'<div class="grouplbl">{esc(title)}</div><div class="rgrid">{cards}</div>'


def render_stats_strip(stats: dict) -> str:
    def fmt(v, suffix=""):
        return "&mdash;" if v is None else f"{v:.1f}{suffix}"

    items = [
        (str(stats["n_rollouts"]), "rollouts"),
        (str(stats["n_scored"]), "scored"),
        (str(stats["judge_dropped"]), "judge-dropped"),
        (fmt(stats["mean"]), "mean score"),
        (f"{fmt(stats['min'])} / {fmt(stats['max'])}", "min / max"),
        (fmt(stats["pct_over50"], "%"), "% over 50"),
    ]
    cells = "".join(
        f'<div class="sc"><span class="scn">{v}</span><span class="scl">{esc(l)}</span></div>'
        for v, l in items
    )
    return f'<div class="statstrip">{cells}</div>'


def render_condition(
    cell: dict, questions: list[str], sys_prompt_text: str | None, per_side: int
) -> str:
    trait, cond = cell["trait"], cell["cond"]
    label, code = cond_label(cond)
    top, bottom = select_top_bottom(cell["rows"], per_side)

    ctx_box = ""
    if sys_prompt_text is not None:
        ctx_box = (
            '<div class="ctxbox"><span class="ctxlbl">System prompt</span>'
            f'<div class="ctxtext">{esc(sys_prompt_text)}</div></div>'
        )

    blob = f"{HF_BLOB_BASE}/raw_completions/{trait}_{cond}_seed42.json"
    scblob = f"{HF_BLOB_BASE}/analysis_tensors/pass_a/{trait}__{cond}.json"

    return (
        f'<div class="cond" id="{trait}_{cond}">'
        '<div class="condhead">'
        f"<h3>{esc(label)}</h3>"
        f'<span class="condcode mono">{esc(code)}</span>'
        "</div>"
        f"{ctx_box}"
        f"{render_stats_strip(cell['stats'])}"
        f"{render_group('Highest-scoring rollouts', top, questions)}"
        f"{render_group('Lowest-scoring rollouts', bottom, questions)}"
        '<div class="condfoot mono">Full cell: '
        f'<a href="{blob}">raw_completions/{trait}_{cond}_seed42.json</a> &middot; '
        f'<a href="{scblob}">pass_a/{trait}__{cond}.json</a></div>'
        "</div>"
    )


STYLE = """
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,500&family=Spline+Sans+Mono:wght@400;500;600&family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;1,6..72,400&display=swap');
:root{
  --ink:#211e19; --ink-soft:#534c41; --paper:#f3eee4; --card:#fbf8f1;
  --line:#ddd3c1; --line-2:#cabfa8; --hl:#ffeea8; --hl-soft:#fff7df;
  --warn:#9a3b2e; --warn-bg:#f7e6df;
  --shadow:0 1px 0 rgba(33,30,25,.03), 0 14px 30px -22px rgba(33,30,25,.55);
}
*{box-sizing:border-box}
html{scroll-behavior:smooth}
body{margin:0; background:var(--paper);
  background-image:radial-gradient(circle at 14% -10%, #faf6ec 0, transparent 40%),
                   radial-gradient(circle at 102% -4%, #ede5d6 0, transparent 36%);
  color:var(--ink); font-family:"Newsreader",Georgia,serif; font-size:17px; line-height:1.55;
  -webkit-font-smoothing:antialiased;}
.wrap{max-width:1180px; margin:0 auto; padding:0 26px 130px}
.mono{font-family:"Spline Sans Mono",ui-monospace,monospace}

header.masthead{border-bottom:2.5px solid var(--ink); padding:48px 0 20px}
.kicker{font-family:"Spline Sans Mono",monospace; font-size:11px; letter-spacing:.26em;
  text-transform:uppercase; color:var(--ink-soft); display:flex; gap:13px; flex-wrap:wrap; align-items:center}
.kicker .dot{width:5px;height:5px;border-radius:50%;background:var(--ink-soft);display:inline-block}
h1.title{font-family:"Fraunces",Georgia,serif; font-weight:600; font-optical-sizing:auto;
  font-size:clamp(33px,5.6vw,56px); line-height:1.0; letter-spacing:-.018em; margin:.3em 0 .14em}
h1.title em{font-style:italic; color:#2f6f6a}
.dek{font-size:18.5px; max-width:70ch; color:var(--ink-soft); margin:.25em 0 0}
.dek code{font-family:"Spline Sans Mono",monospace; font-size:.85em; background:var(--card);
  padding:1px 5px; border-radius:4px; border:1px solid var(--line)}

.statbar{display:flex; flex-wrap:wrap; margin:28px 0 4px; border:1px solid var(--line-2);
  border-radius:11px; overflow:hidden; background:var(--card); box-shadow:var(--shadow)}
.statbar .stat{padding:14px 22px; border-right:1px solid var(--line); flex:1; min-width:118px}
.statbar .stat:last-child{border-right:none}
.statbar .num{font-family:"Fraunces",serif; font-size:28px; font-weight:600; line-height:1; display:block}
.statbar .lbl{font-family:"Spline Sans Mono",monospace; font-size:10px; letter-spacing:.15em;
  text-transform:uppercase; color:var(--ink-soft); margin-top:7px; display:block}

.callout{margin:26px 0 6px; background:var(--hl-soft); border:1px solid var(--line-2);
  border-left:4px solid var(--hl); border-radius:9px; padding:14px 18px; font-size:15.5px; color:var(--ink-soft)}
.callout b{color:var(--ink)}
.warnbanner{margin:26px 0 6px; background:var(--warn-bg); border:1px solid #e0b6a8;
  border-left:4px solid var(--warn); border-radius:9px; padding:13px 18px; font-size:15px; color:#6f2b20}
.warnbanner b{color:var(--warn)}

.famsec{margin:44px 0 8px}
.famhead{display:flex; align-items:baseline; gap:14px; padding:10px 0 8px; border-bottom:2px solid var(--c);
  margin-bottom:6px; flex-wrap:wrap}
.famhead h2{font-family:"Fraunces",serif; font-weight:600; font-size:27px; margin:0; color:var(--c)}
.famhead .fct{font-family:"Spline Sans Mono",monospace; font-size:12px; color:var(--ink-soft)}
.famhead .fdesc{font-size:15px; color:var(--ink-soft); flex-basis:100%; margin-top:4px; font-style:italic; max-width:80ch}

.cond{margin:22px 0 0; padding:0 0 14px; border-bottom:1px dashed var(--line)}
.condhead{display:flex; align-items:baseline; gap:12px; margin:16px 0 8px; flex-wrap:wrap}
.condhead h3{font-family:"Fraunces",serif; font-weight:600; font-size:20px; margin:0; color:var(--ink)}
.condcode{font-size:11px; color:#fff8ef; background:var(--c); border-radius:5px; padding:2px 8px; letter-spacing:.08em}
.ctxbox{background:var(--paper); border:1px solid var(--line); border-left:3px solid var(--c);
  border-radius:7px; padding:9px 12px; margin:0 0 10px}
.ctxbox .ctxlbl{font-family:"Spline Sans Mono",monospace; font-size:9.5px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--c); display:block; margin-bottom:3px}
.ctxbox .ctxtext{font-size:14.5px; color:var(--ink-soft)}

.statstrip{display:flex; flex-wrap:wrap; border:1px solid var(--line-2); border-radius:9px;
  overflow:hidden; background:var(--card); margin:0 0 12px; box-shadow:var(--shadow)}
.statstrip .sc{padding:9px 16px; border-right:1px solid var(--line); flex:1; min-width:92px}
.statstrip .sc:last-child{border-right:none}
.statstrip .scn{font-family:"Fraunces",serif; font-size:19px; font-weight:600; display:block; line-height:1}
.statstrip .scl{font-family:"Spline Sans Mono",monospace; font-size:9px; letter-spacing:.11em;
  text-transform:uppercase; color:var(--ink-soft); margin-top:5px; display:block}

.grouplbl{font-family:"Spline Sans Mono",monospace; font-size:10px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--c); margin:14px 0 8px; padding-bottom:3px; border-bottom:1px solid var(--line)}
.rgrid{display:grid; grid-template-columns:repeat(auto-fill,minmax(360px,1fr)); gap:13px}
.rcard{background:var(--card); border:1px solid var(--line-2); border-left:4px solid var(--c);
  border-radius:10px; box-shadow:var(--shadow); padding:12px 14px 13px; display:flex; flex-direction:column}
.rhead{display:flex; align-items:center; gap:10px; margin-bottom:6px}
.rid{font-family:"Spline Sans Mono",monospace; font-size:10.5px; color:var(--ink-soft)}
.score{font-family:"Fraunces",serif; font-weight:700; font-size:16px; padding:2px 10px; border-radius:999px;
  line-height:1.3}
.score.over{background:var(--c); color:#fff8ef}
.score.under{background:transparent; color:var(--c); border:1.5px solid var(--c)}
.msg{margin:9px 0 0}
.msg .role{font-family:"Spline Sans Mono",monospace; font-size:9.5px; letter-spacing:.14em;
  text-transform:uppercase; color:var(--ink-soft); display:inline-block; margin-bottom:4px;
  border-bottom:1px solid var(--line); padding-bottom:1px}
.msg .role.user{color:#8a5a1a}
.msg .role.assistant{color:var(--c)}
.msg .body{white-space:pre-wrap; word-break:break-word; font-size:14.5px; line-height:1.5;
  background:var(--paper); border:1px solid var(--line); border-radius:7px; padding:9px 11px}
.msg .body.ans.clamped{max-height:15em; overflow:hidden; position:relative}
.msg .body.ans.clamped::after{content:""; position:absolute; left:0; right:0; bottom:0; height:2.2em;
  background:linear-gradient(transparent, var(--paper)); pointer-events:none}
.msg .body.ans.expanded{max-height:none}
.msg .body.ans.expanded::after{display:none}
.toggle{font-family:"Spline Sans Mono",monospace; font-size:10.5px; color:var(--c); cursor:pointer;
  border:none; background:none; padding:5px 0 0; text-decoration:underline; text-underline-offset:2px}
.condfoot{font-size:10.5px; color:var(--ink-soft); margin-top:12px}
.condfoot a{color:var(--ink-soft); text-decoration:none; border-bottom:1px dotted var(--line-2)}

footer.foot{margin-top:64px; padding-top:22px; border-top:1px solid var(--line);
  font-family:"Spline Sans Mono",monospace; font-size:11.5px; color:var(--ink-soft); line-height:1.8}
footer.foot b{color:var(--ink); font-weight:600}
footer.foot a{color:var(--ink-soft)}
@media (max-width:760px){ .rgrid{grid-template-columns:1fr} }
"""

TOGGLE_JS = """
document.querySelectorAll('.toggle').forEach(function(b){
  b.addEventListener('click', function(){
    var body = b.parentElement.querySelector('.ans');
    var exp = body.classList.toggle('expanded');
    b.innerHTML = exp ? 'show less \\u2191' : 'show full \\u2193';
  });
});
"""


def build_html(cells: dict, questions: dict, sys_prompts: dict, per_side: int) -> tuple[str, dict]:
    # ---- global stats for the masthead ----
    n_conditions = sum(1 for k in cells if cells[k] is not None)
    total_scored = sum(cells[k]["stats"]["n_scored"] for k in cells if cells[k])
    total_rollouts = sum(cells[k]["stats"]["n_rollouts"] for k in cells if cells[k])
    total_dropped = sum(cells[k]["stats"]["judge_dropped"] for k in cells if cells[k])

    parts = []
    parts.append("<!doctype html><html lang=en><head><meta charset=utf-8>")
    parts.append("<meta name=viewport content='width=device-width, initial-scale=1'>")
    parts.append("<title>Issue 779 — eval rollouts &amp; judge scores</title>")
    parts.append(f"<style>{STYLE}</style></head><body><div class='wrap'>")

    # masthead
    parts.append(
        "<header class='masthead'>"
        "<div class='kicker'><span class='dot'></span> Issue #779 monitoring rig"
        " <span class='dot'></span> Persona-vectors eval contexts"
        " <span class='dot'></span> Qwen-2.5-7B-Instruct"
        " <span class='dot'></span> Sonnet-4.5 judge (5-draw mean)</div>"
        "<h1 class='title'>Eval rollouts &amp; <em>judge scores</em></h1>"
        "<p class='dek'>The model's actual completions under each monitoring context "
        "(8 trait-inducing system prompts + 5 many-shot settings), scored 0–100 for "
        "trait expression by the <code>claude-sonnet-4-5</code> judge. Three traits: "
        "evil, sycophancy, hallucination.</p>"
        "<div class='statbar'>"
        f"<div class='stat'><span class='num'>{len(TRAITS)}</span><span class='lbl'>Traits</span></div>"
        f"<div class='stat'><span class='num'>{n_conditions}</span><span class='lbl'>Trait × condition cells</span></div>"
        f"<div class='stat'><span class='num'>{n_conditions * 2 * per_side}</span><span class='lbl'>Rollouts shown</span></div>"
        f"<div class='stat'><span class='num'>{total_scored:,}</span><span class='lbl'>Rollouts scored</span></div>"
        f"<div class='stat'><span class='num'>{total_dropped}</span><span class='lbl'>Judge-dropped</span></div>"
        "</div></header>"
    )

    # subsetting-rule callout
    parts.append(
        "<div class='callout'><b>What you're seeing.</b> Each trait section below lists "
        "its 13 eval conditions. For every condition we show a <b>subsample</b>: the "
        f"<b>{per_side} highest-</b> and <b>{per_side} lowest-scoring</b> rollouts, ranked by the "
        "per-rollout 5-draw-mean judge score. Each condition's one-line stats strip is computed "
        "over <i>all</i> its scored rollouts (n=200 per cell before drops). The full per-cell "
        f"rollout JSONs are linked under each condition and browsable "
        f"<a href='{HF_TREE_URL}'>on the HF data repo</a>.</div>"
    )

    common_mod = _COMMON  # set in main
    for trait in TRAITS:
        accent = TRAIT_ACCENT[trait]
        desc = common_mod.TRAIT_DESCRIPTIONS.get(trait, "")
        present = [c for c in ALL_CONDS if cells.get((trait, c))]
        parts.append(f"<section class='famsec' style='--c:{accent}'>")
        parts.append(
            "<div class='famhead'>"
            f"<h2>{esc(trait.capitalize())}</h2>"
            f"<span class='fct mono'>{len(present)} conditions</span>"
            f"<span class='fdesc'>Trait definition (judge rubric): {esc(desc)}</span>"
            "</div>"
        )
        if trait in HARMFUL_TRAITS:
            parts.append(
                "<div class='warnbanner'><b>Content warning.</b> These rollouts were "
                "collected under trait-inducing prompts and contain deliberately harmful "
                "content (the raw model outputs of the '<i>evil</i>' monitoring rig). Shown "
                "verbatim for research inspection; already public on the HF data repo.</div>"
            )
        for cond in ALL_CONDS:
            cell = cells.get((trait, cond))
            if cell is None:
                continue
            spt = None
            if cond.startswith("sys"):
                spt = sys_prompts[trait][int(cond[3:])]
            parts.append(render_condition(cell, questions[trait], spt, per_side))
        parts.append("</section>")

    # footer
    parts.append(
        "<footer class='foot'>"
        f"<div><b>Data source.</b> HF dataset <span class='mono'>{HF_DATA_REPO}</span>, "
        f"prefix <span class='mono'>{HF_PREFIX}/</span> (revision <span class='mono'>{HF_REVISION}</span>) "
        "&mdash; <span class='mono'>raw_completions/{trait}_{cond}_seed42.json</span> "
        "(rollout text) + <span class='mono'>analysis_tensors/pass_a/{trait}__{cond}.json</span> "
        "(per-rollout 5-draw-mean judge scores + judge_dropped). Eval questions: evil from "
        "<span class='mono'>issue779_common.EVIL_ARTIFACTS</span>, sycophancy/hallucination from "
        "<span class='mono'>artifacts/{trait}.json</span>.</div>"
        f"<div><b>Subsetting rule.</b> Per (trait, condition) cell: the {per_side} highest- and "
        f"{per_side} lowest-scoring rollouts by 5-draw-mean judge score; stats strip over all scored "
        "rollouts.</div>"
        "<div><b>Generator.</b> <span class='mono'>scripts/issue779_dashboard_completions.py</span> "
        f"&middot; generated {date.today().isoformat()}.</div>"
        "</footer>"
    )

    parts.append(f"<script>{TOGGLE_JS}</script>")
    parts.append("</div></body></html>")
    doc = "".join(parts)
    meta = {
        "n_conditions": n_conditions,
        "total_scored": total_scored,
        "total_rollouts": total_rollouts,
        "total_dropped": total_dropped,
        "per_side": per_side,
    }
    return doc, meta


_COMMON = None


def main():
    global _COMMON
    ap = argparse.ArgumentParser()
    ap.add_argument("--download-dir", default="/tmp/issue779_dash_dl")
    ap.add_argument(
        "--out", default=str(REPO_ROOT / "experiments" / "dashboards" / "issue779_completions.html")
    )
    ap.add_argument(
        "--skip-download", action="store_true", help="reuse files already present in --download-dir"
    )
    ap.add_argument("--per-side", type=int, default=3)
    args = ap.parse_args()

    _COMMON = _load_common_module()
    dl = Path(args.download_dir)
    download_inputs(dl, args.skip_download)
    questions = load_eval_questions(_COMMON, dl)
    sys_prompts = _COMMON.EVAL_SYSTEM_PROMPTS

    cells = {}
    missing = []
    for trait in TRAITS:
        for cond in ALL_CONDS:
            c = load_cell(dl, trait, cond)
            cells[(trait, cond)] = c
            if c is None:
                missing.append(f"{trait}/{cond}")

    # Build at per_side=3; if over the size cap, rebuild at 2.
    per_side = args.per_side
    doc, meta = build_html(cells, questions, sys_prompts, per_side)
    if len(doc.encode("utf-8")) > PAGE_SIZE_CAP_BYTES and per_side > 2:
        per_side = 2
        doc, meta = build_html(cells, questions, sys_prompts, per_side)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(doc, encoding="utf-8")

    # ---- verification digests (NO raw text) ----
    size = len(doc.encode("utf-8"))
    print("=" * 66)
    print(f"WROTE {out}")
    print(f"PAGE_BYTES {size}  ({size / 1024 / 1024:.2f} MB)  cap={PAGE_SIZE_CAP_BYTES}")
    print(f"PER_SIDE {per_side}  (rollouts shown/cell = {2 * per_side})")
    print(f"CONDITIONS_COVERED {meta['n_conditions']} / {len(TRAITS) * len(ALL_CONDS)}")
    print(f"MISSING_CELLS {missing if missing else 'none'}")
    print("-" * 66)
    print("PER-TRAIT SUMMARY (conditions, rollouts scored, judge-dropped):")
    for trait in TRAITS:
        tconds = [cells[(trait, c)] for c in ALL_CONDS if cells.get((trait, c))]
        tscored = sum(c["stats"]["n_scored"] for c in tconds)
        tdrop = sum(c["stats"]["judge_dropped"] for c in tconds)
        trows = sum(
            len(select_top_bottom(c["rows"], per_side)[0])
            + len(select_top_bottom(c["rows"], per_side)[1])
            for c in tconds
        )
        print(
            f"  {trait:14s} conds={len(tconds):2d}  scored={tscored:5d}  dropped={tdrop:3d}  rows_rendered={trows}"
        )
    print("-" * 66)
    print("PER trait x condition: n_scored / judge_dropped / mean / %>50 / rows_rendered")
    for trait in TRAITS:
        for cond in ALL_CONDS:
            c = cells.get((trait, cond))
            if not c:
                continue
            s = c["stats"]
            t, b = select_top_bottom(c["rows"], per_side)
            mean = f"{s['mean']:.1f}" if s["mean"] is not None else "NA"
            p50 = f"{s['pct_over50']:.0f}" if s["pct_over50"] is not None else "NA"
            print(
                f"  {trait:13s} {cond:7s}  n={s['n_scored']:3d}  drop={s['judge_dropped']:2d}  "
                f"mean={mean:>5s}  %>50={p50:>3s}  rows={len(t) + len(b)}"
            )
    print("-" * 66)
    print("SCORE HISTOGRAM per trait (bucketed over all scored rollouts, no text):")
    buckets = [(0, 10), (10, 25), (25, 50), (50, 75), (75, 90), (90, 100.01)]
    for trait in TRAITS:
        allsc = []
        for cond in ALL_CONDS:
            c = cells.get((trait, cond))
            if c:
                allsc += [r[2] for r in c["rows"]]
        hist = []
        for lo, hi in buckets:
            n = sum(1 for s in allsc if lo <= s < hi)
            hist.append(f"[{lo:.0f},{hi:.0f}):{n}")
        print(f"  {trait:14s} n={len(allsc):4d}  " + "  ".join(hist))
    print("=" * 66)


if __name__ == "__main__":
    main()
