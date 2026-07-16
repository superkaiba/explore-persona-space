"""#952 china-bank behavior dashboard — where Qwen and Claude behave differently.

Builds ONE self-contained static HTML file (inline CSS + minimal vanilla JS, no
external requests) that shows, per china divergence pair, the divergent query with
Qwen's and Claude's answers side by side, the graded judge labels (divergence,
refusal), the lexical-divergence cosine, and the per-pair context->answer-activation
predictability drops. The header carries the committed pooled reads vs the
behavior-differs (S1) subset reads so the "does a divergence-selective penalty
emerge where behavior differs?" question is answerable at a glance.

This dashboard is the ONE deliberate display surface for the CCP-sensitive china
bank text: the builder reads the query/answer text programmatically (json.load ->
variable -> html.escape -> HTML string -> file) and never inspects or logs the
string values. The same text is already public on the HF data repo, so the
committed HTML is not a new exposure. Everywhere else in the pipeline the text
stays digest-only.

Inputs:
  - eval_results/issue_952/refusal_sanity_check/behavior_differs_subset.json
    (stats strip + per-pair drops + S1/S2 membership; produced by
    issue952_behavior_differs_subset.py).
  - china verification JSONs (numeric labels for all 42 captured pairs).
  - china answer/query text (HF public + committed query banks), via
    issue952_behavior_differs_subset.load_china_texts.

Usage:
  OMP_NUM_THREADS=8 EPM_ALLOW_BANK_READ=1 uv run python scripts/issue952_behavior_dashboard.py
"""

from __future__ import annotations

# ruff: noqa: E402 — load_dotenv() must run before numpy/torch import (shared-VM thread caps)
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import datetime
import html
import json
import logging
import pathlib
import re
import subprocess
import sys

from issue952_behavior_differs_subset import (
    CHINA_CAT,
    REFUSAL_THR,
    _china_corpus_tfidf,
    load_china_texts,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue952_behavior_dashboard")

BASE = pathlib.Path(__file__).resolve().parent.parent
JUDGE_MODEL = "claude-sonnet-4-5-20250929"


def _sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=BASE, capture_output=True, text=True, check=True
    ).stdout.strip()


def _esc(x) -> str:
    """HTML-escape any value (text is never inspected, only escaped + emitted)."""
    if x is None:
        return "<span class='na'>n/a</span>"
    return html.escape(str(x))


def _num(x, nd=3) -> str:
    if x is None:
        return "<span class='na'>n/a</span>"
    try:
        return f"{float(x):.{nd}f}"
    except (TypeError, ValueError):
        return _esc(x)


def _refusal_class(v) -> str:
    """CSS class for a refusal chip: refused (>=50) vs answered (<50) vs n/a."""
    if v is None:
        return "chip-na"
    return "chip-refuse" if float(v) >= REFUSAL_THR else "chip-answer"


# ── answer-cell rendering (long answers get a show-more toggle; text only escaped) ──


def _answer_cell(model: str, model_class: str, answer: str, refusal: float | None) -> str:
    ref_chip = f"<span class='chip {_refusal_class(refusal)}'>refusal {_num(refusal, 1)}</span>"
    if answer is None:
        body = "<div class='ans-missing'>answer not available</div>"
    else:
        body = f"<div class='ans-text'>{_esc(answer)}</div>"
    return (
        f"<div class='ans-col {model_class}'>"
        f"<div class='ans-head'><span class='ans-model'>{_esc(model)}</span>{ref_chip}</div>"
        f"{body}</div>"
    )


def _pair_card(pid: str, meta: dict, texts: dict) -> str:
    """One captured-pair card: badges + numeric strip + divergent Q with both answers
    side by side + collapsed control block."""
    div_qid, ctl_qid = f"{pid}_div", f"{pid}_ctl"
    dt, ct = texts.get(div_qid, {}), texts.get(ctl_qid, {})

    badges = [f"<span class='badge badge-origin'>{_esc(meta['origin'])}</span>"]
    badges.append(
        "<span class='badge badge-kept'>kept</span>"
        if meta["kept"]
        else "<span class='badge badge-rejected'>rejected</span>"
    )
    if meta.get("in_S1"):
        badges.append("<span class='badge badge-s1'>S1 refusal-mismatch</span>")
    if meta.get("in_S2"):
        badges.append("<span class='badge badge-s2'>S2 lexical-divergence</span>")
    badge_html = "".join(badges)

    def _stat(label, val, nd=3):
        return (
            f"<div class='stat'><span class='stat-l'>{label}</span>"
            f"<span class='stat-v'>{_num(val, nd)}</span></div>"
        )

    q_ans = _answer_cell(
        "Qwen-2.5-7B-Instruct", "qwen", dt.get("qwen_answer"), meta.get("refusal_qwen_div")
    )
    c_ans = _answer_cell(
        "Claude (sonnet-4-5)", "claude", dt.get("claude_answer"), meta.get("refusal_claude_div")
    )
    q_ans_ctl = _answer_cell(
        "Qwen-2.5-7B-Instruct", "qwen", ct.get("qwen_answer"), meta.get("refusal_qwen_ctl")
    )
    c_ans_ctl = _answer_cell(
        "Claude (sonnet-4-5)", "claude", ct.get("claude_answer"), meta.get("refusal_claude_ctl")
    )

    stats = "".join(
        [
            _stat("divergence (div)", meta.get("divergence_div"), 1),
            _stat("refusal Qwen (div)", meta.get("refusal_qwen_div"), 1),
            _stat("refusal Claude (div)", meta.get("refusal_claude_div"), 1),
            _stat("tfidf cos (div)", meta.get("tfidf_cos_div_china_corpus"), 3),
            _stat("drop own", meta.get("arm_matched_drop_own"), 4),
            _stat("drop ext-plain", meta.get("arm_matched_drop_ext_plain"), 4),
            _stat("arm-matched d", meta.get("arm_matched_d"), 4),
            _stat("cross drop", meta.get("cross_drop"), 4),
        ]
    )

    divergent_block = (
        "<div class='q-label'>Divergent query</div>"
        f"<div class='q-text'>{_esc(dt.get('question'))}</div>"
        f"<div class='ans-row'>{q_ans}{c_ans}</div>"
    )

    control_block = (
        "<details class='control'>"
        "<summary>Control query (entity-swapped, same template) — divergence "
        f"{_num(meta.get('divergence_ctl'), 1)}</summary>"
        f"<div class='q-text'>{_esc(ct.get('question'))}</div>"
        f"<div class='ans-row'>{q_ans_ctl}{c_ans_ctl}</div>"
        "</details>"
    )

    return (
        f"<article class='card' data-s1='{int(bool(meta.get('in_S1')))}' "
        f"data-kept='{int(bool(meta['kept']))}'>"
        f"<div class='card-head'><span class='pid'>{_esc(pid)}</span>"
        f"<span class='badges'>{badge_html}</span></div>"
        f"<div class='stat-strip'>{stats}</div>"
        f"{divergent_block}{control_block}"
        "</article>"
    )


def _header_stats(subset: dict) -> str:
    """Committed pooled reads vs S1 subset reads (d, cross drop, R2 levels)."""
    cp = subset["committed_pooled_reads"]
    s1 = subset["subsets"]["S1_refusal_mismatch"]
    notc = subset["subsets"]["NOT_S1_behavior_similar"]
    m = subset["membership"]

    def _cell(label, val, ci=None, p=None, nd=4):
        parts = [f"<div class='hs-v'>{_num(val, nd)}</div>"]
        if ci is not None:
            parts.append(f"<div class='hs-ci'>95% CI [{_num(ci[0], nd)}, {_num(ci[1], nd)}]</div>")
        if p is not None:
            parts.append(f"<div class='hs-ci'>sign-flip p {_num(p, 3)}</div>")
        return f"<div class='hs-cell'><div class='hs-l'>{label}</div>{''.join(parts)}</div>"

    am_committed = cp["china_31_arm_matched"]
    am_s1 = s1["arm_matched"]
    am_not = notc["arm_matched"]
    cx_committed = cp["china_31_cross"]
    cx_s1 = s1["cross_own_map_x_claude_target"]

    rows = [
        "<div class='hs-row'><div class='hs-rlabel'>Arm-matched d "
        "(external minus own predictability drop, divergent vs control)</div>"
        + _cell(
            f"all 31 kept (n={am_committed['n']})",
            am_committed["mean_d"],
            am_committed["mean_d_ci95"],
            am_committed["sign_flip_p"],
        )
        + _cell(
            f"S1 refusal-mismatch (n={am_s1['n']})",
            am_s1["mean_d"],
            am_s1["mean_d_ci95"],
            am_s1["sign_flip_p_one_sided"],
        )
        + _cell(
            f"NOT-S1 (n={am_not['n']})",
            am_not["mean_d"],
            am_not["mean_d_ci95"],
            am_not["sign_flip_p_one_sided"],
        )
        + "</div>",
        "<div class='hs-row'><div class='hs-rlabel'>Cross cell (own map -> Claude target) "
        "predictability drop, divergent vs control</div>"
        + _cell(
            f"all 31 kept (n={cx_committed['n']})",
            cx_committed["mean_drop"],
            cx_committed["mean_drop_ci95"],
            cx_committed["sign_flip_p"],
        )
        + _cell(
            f"S1 refusal-mismatch (n={cx_s1['n']})",
            cx_s1["mean"],
            cx_s1["mean_ci95"],
            cx_s1["sign_flip_p_one_sided"],
        )
        + "</div>",
        "<div class='hs-row'><div class='hs-rlabel'>Cross-cell R2 LEVELS "
        "(own map -> Claude target; floored vs divergence-selective)</div>"
        + _cell("all 31 divergent", cx_committed["mean_r2_div"], nd=4)
        + _cell("all 31 control", cx_committed["mean_r2_ctl"], nd=4)
        + _cell("S1 divergent", cx_s1["mean_r2_div"], nd=4)
        + _cell("S1 control", cx_s1["mean_r2_ctl"], nd=4)
        + "</div>",
    ]
    membership = (
        f"<div class='hs-membership'>Captured pairs: {m['n_captured']} &nbsp;|&nbsp; "
        f"kept: {m['n_kept']} &nbsp;|&nbsp; S1 refusal-mismatch: {m['n_S1']} &nbsp;|&nbsp; "
        f"NOT-S1: {m['n_NOT_S1']} &nbsp;|&nbsp; S2 lexical-divergence: {m['n_S2']} "
        f"&nbsp;|&nbsp; S1&cap;S2: {m['n_S1_and_S2']}</div>"
    )
    return "<div class='header-stats'>" + "".join(rows) + membership + "</div>"


# ── page assembly ─────────────────────────────────────────────────────────────

_CSS = """
:root{--bg:#0f1216;--panel:#171b22;--panel2:#1d2530;--fg:#e6e9ee;--muted:#9aa4b2;
--line:#2a3340;--qwen:#4f9dff;--claude:#f2994a;--refuse:#e2544b;--answer:#3fae7a;
--s1:#c586ff;--s2:#54c4c4;--kept:#3fae7a;--rej:#7a8391;}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font-family:'Inter',-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,Helvetica,Arial,sans-serif;
line-height:1.5;font-size:15px}
.wrap{max-width:1180px;margin:0 auto;padding:32px 24px 80px}
h1{font-size:26px;font-weight:650;margin:0 0 6px}
.sub{color:var(--muted);margin:0 0 8px;max-width:920px}
.exploratory{display:inline-block;background:#3a2f16;color:#f0c674;border:1px solid #6b5a2a;
border-radius:6px;padding:4px 10px;font-size:12.5px;margin:8px 0 20px}
.header-stats{background:var(--panel);border:1px solid var(--line);border-radius:12px;
padding:18px 20px;margin:0 0 12px}
.hs-row{display:flex;flex-wrap:wrap;gap:14px;align-items:stretch;padding:10px 0;
border-bottom:1px solid var(--line)}
.hs-row:last-of-type{border-bottom:none}
.hs-rlabel{flex:1 1 210px;min-width:190px;color:var(--muted);font-size:13px;align-self:center}
.hs-cell{background:var(--panel2);border:1px solid var(--line);border-radius:8px;
padding:8px 12px;min-width:150px}
.hs-l{color:var(--muted);font-size:11.5px;margin-bottom:3px}
.hs-v{font-size:18px;font-weight:650;font-variant-numeric:tabular-nums}
.hs-ci{color:var(--muted);font-size:11px;font-variant-numeric:tabular-nums}
.hs-membership{margin-top:12px;color:var(--muted);font-size:13px}
.controls{display:flex;gap:10px;align-items:center;margin:18px 0 14px;flex-wrap:wrap}
.controls button{background:var(--panel2);color:var(--fg);border:1px solid var(--line);
border-radius:8px;padding:7px 14px;font-size:13px;cursor:pointer}
.controls button.active{border-color:var(--s1);color:#fff;background:#2b2038}
.controls .count{color:var(--muted);font-size:13px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:12px;
padding:18px 20px;margin:0 0 16px}
.card-head{display:flex;justify-content:space-between;align-items:center;gap:12px;
flex-wrap:wrap;margin-bottom:10px}
.pid{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:13.5px;color:var(--muted)}
.badges{display:flex;gap:6px;flex-wrap:wrap}
.badge{font-size:11px;padding:3px 9px;border-radius:20px;border:1px solid var(--line)}
.badge-origin{color:var(--muted)}
.badge-kept{color:var(--kept);border-color:#2f5c47}
.badge-rejected{color:var(--rej)}
.badge-s1{color:var(--s1);border-color:#4a3563;background:#241a30}
.badge-s2{color:var(--s2);border-color:#2a5252;background:#16292b}
.stat-strip{display:flex;flex-wrap:wrap;gap:8px;margin:0 0 14px}
.stat{background:var(--panel2);border:1px solid var(--line);border-radius:7px;
padding:5px 10px;min-width:96px}
.stat-l{display:block;color:var(--muted);font-size:10.5px}
.stat-v{font-size:14px;font-weight:600;font-variant-numeric:tabular-nums}
.q-label{color:var(--muted);font-size:11.5px;text-transform:uppercase;letter-spacing:.04em;
margin:4px 0 4px}
.q-text{background:#12161c;border:1px solid var(--line);border-left:3px solid #5a6472;
border-radius:6px;padding:10px 12px;margin:0 0 12px;font-size:14px;white-space:pre-wrap;
word-break:break-word}
.ans-row{display:grid;grid-template-columns:1fr 1fr;gap:12px}
@media(max-width:760px){.ans-row{grid-template-columns:1fr}}
.ans-col{border:1px solid var(--line);border-radius:8px;padding:10px 12px;background:#12161c}
.ans-col.qwen{border-left:4px solid var(--qwen)}
.ans-col.claude{border-left:4px solid var(--claude)}
.ans-head{display:flex;justify-content:space-between;align-items:center;gap:8px;margin-bottom:6px}
.ans-model{font-size:12.5px;font-weight:600}
.ans-col.qwen .ans-model{color:var(--qwen)}
.ans-col.claude .ans-model{color:var(--claude)}
.chip{font-size:10.5px;padding:2px 8px;border-radius:20px;border:1px solid var(--line);
white-space:nowrap;font-variant-numeric:tabular-nums}
.chip-refuse{color:var(--refuse);border-color:#5c2f2c;background:#2a1614}
.chip-answer{color:var(--answer);border-color:#2f5c47;background:#14261e}
.chip-na{color:var(--muted)}
.ans-text{font-size:13.5px;white-space:pre-wrap;word-break:break-word;max-height:220px;
overflow:hidden;position:relative;transition:max-height .2s}
.ans-text.expanded{max-height:none}
.ans-col.collapsed .ans-text::after{content:"";position:absolute;left:0;right:0;bottom:0;
height:48px;background:linear-gradient(transparent,#12161c)}
.more-btn{margin-top:6px;background:none;border:none;color:var(--qwen);cursor:pointer;
font-size:12px;padding:0}
.ans-missing{color:var(--muted);font-style:italic;font-size:13px}
.control{margin-top:12px;border-top:1px dashed var(--line);padding-top:10px}
.control summary{cursor:pointer;color:var(--muted);font-size:13px}
.na{color:var(--muted);font-style:italic}
footer{color:var(--muted);font-size:12px;margin-top:32px;border-top:1px solid var(--line);
padding-top:16px;line-height:1.7}
footer a{color:var(--qwen)}
"""

_JS = """
function filt(mode,btn){
  document.querySelectorAll('.controls button').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  let n=0;
  document.querySelectorAll('.card').forEach(c=>{
    const s1=c.dataset.s1==='1';
    const show = mode==='all' || (mode==='s1'&&s1) || (mode==='nots1'&&!s1);
    c.style.display=show?'':'none'; if(show)n++;
  });
  document.getElementById('count').textContent=n+' pairs shown';
}
function initAns(){
  document.querySelectorAll('.ans-text').forEach(t=>{
    if(t.scrollHeight>t.clientHeight+4){
      const col=t.closest('.ans-col'); col.classList.add('collapsed');
      const b=document.createElement('button'); b.className='more-btn'; b.textContent='Show more';
      b.onclick=()=>{const e=t.classList.toggle('expanded');col.classList.toggle('collapsed',!e);
        b.textContent=e?'Show less':'Show more';};
      t.after(b);
    }
  });
}
function applyHash(){
  const i={'#s1':1,'#nots1':2}[location.hash]||0;
  filt(['all','s1','nots1'][i],document.querySelectorAll('.controls button')[i]);
}
window.addEventListener('DOMContentLoaded',()=>{initAns();applyHash();});
window.addEventListener('hashchange',applyHash);
"""


def build_dashboard() -> pathlib.Path:
    subset = json.loads(
        (
            BASE / "eval_results/issue_952/refusal_sanity_check/behavior_differs_subset.json"
        ).read_text()
    )
    per_pair_sub = subset["per_pair"]  # 31 kept pairs, numeric labels + drops + S1/S2

    # verification JSONs — labels for ALL 42 captured pairs (kept + rejected)
    pv = json.loads((BASE / "eval_results/issue_952/divergence_bank_verification.json").read_text())
    cv = json.loads(
        (
            BASE / "eval_results/issue_952/china-politics-topup/summaries/"
            "china_topup_verification.json"
        ).read_text()
    )
    parent_china = {p["pair_id"]: p for p in pv["pairs"] if p.get("category") == CHINA_CAT}
    new_china = {p["pair_id"]: p for p in cv["pairs"]}
    final_kept = set(cv["final_china_kept_pairs"])

    texts = load_china_texts()  # {query_id: {question, qwen_answer, claude_answer, ...}}
    captured_pids = sorted({t["pair_id"] for t in texts.values()})
    tfidf_china = _china_corpus_tfidf(texts)  # keyed by query_id

    # build per-pair meta over the 42 captured pairs
    meta_by_pid: dict[str, dict] = {}
    for pid in captured_pids:
        src = new_china.get(pid) or parent_china.get(pid)
        d = src.get("divergent") if isinstance(src, dict) else {}
        c = src.get("control") if isinstance(src, dict) else {}
        d, c = (d or {}), (c or {})
        origin = "new" if pid in new_china else "parent"
        sub = per_pair_sub.get(pid, {})
        meta_by_pid[pid] = {
            "origin": origin,
            "kept": pid in final_kept,
            "in_S1": bool(sub.get("in_S1")),
            "in_S2": bool(sub.get("in_S2")),
            "divergence_div": d.get("divergence"),
            "divergence_ctl": c.get("divergence"),
            "refusal_qwen_div": d.get("refusal_qwen"),
            "refusal_claude_div": d.get("refusal_claude"),
            "refusal_qwen_ctl": c.get("refusal_qwen"),
            "refusal_claude_ctl": c.get("refusal_claude"),
            "tfidf_cos_div_china_corpus": tfidf_china.get(f"{pid}_div"),
            "arm_matched_drop_own": sub.get("arm_matched_drop_own"),
            "arm_matched_drop_ext_plain": sub.get("arm_matched_drop_ext_plain"),
            "arm_matched_d": sub.get("arm_matched_d"),
            "cross_drop": sub.get("cross_drop"),
        }

    # sort: S1 first, then ascending divergent-query tfidf (most lexically divergent first)
    def _sort_key(pid):
        m = meta_by_pid[pid]
        tf = m["tfidf_cos_div_china_corpus"]
        return (0 if m["in_S1"] else 1, tf if tf is not None else 2.0, pid)

    ordered = sorted(captured_pids, key=_sort_key)
    cards = "\n".join(_pair_card(pid, meta_by_pid[pid], texts) for pid in ordered)

    intro = (
        "This dashboard shows the china-politics divergence pairs behind #952's "
        "context&rarr;answer-activation predictability read. Each pair is a divergent query "
        "(where Qwen and Claude may behave differently) plus an entity-swapped, same-template "
        "control. The refusal chips are graded judge scores (0&ndash;100, threshold 50); "
        "the numeric strip carries the graded divergence / refusal labels, the lexical-divergence "
        "cosine, and the per-pair predictability drops. The header contrasts the pooled reads "
        "against the S1 refusal-mismatch subset to show whether an external-answer prediction "
        "penalty emerges specifically where the two models&rsquo; behavior diverges."
    )
    m = subset["membership"]
    prov = subset["provenance"]
    footer = (
        f"Generated {_esc(datetime.datetime.now(datetime.UTC).isoformat())} "
        f"&middot; commit <code>{_esc(_sha())}</code> &middot; "
        f"subset stats @ <code>{_esc(prov.get('git_commit'))}</code><br>"
        f"Parent bank HF revision <code>{_esc(prov['parent_revision'])}</code>; "
        f"china top-up HF revision <code>{_esc(prov['china_revision'])}</code>.<br>"
        f"Labels: {_esc(JUDGE_MODEL)} graded judge; refusal boolean threshold 50 "
        "(rubric midpoint). TF-idf cosine recomputed over the china corpus for uniform "
        f"coverage. Captured pairs {m['n_captured']}, kept {m['n_kept']} (statistics restricted "
        "to kept pairs, comparable to the committed read). EXPLORATORY conditional read: "
        "conditioning on judge-text behavior labels, not pre-registered, small n. Bank text is "
        "already public on the HF data repo; displayed here as the deliberate behavior-inspection "
        "surface."
    )

    page = f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>#952 china bank — where Qwen and Claude behave differently</title>
<style>{_CSS}</style></head>
<body><div class="wrap">
<h1>#952 china bank &mdash; where Qwen and Claude behave differently</h1>
<p class="sub">{intro}</p>
<div class="exploratory">Exploratory conditional read &mdash; behavior-label conditioning,
not pre-registered, small n. Statistics on the 31 kept china pairs.</div>
{_header_stats(subset)}
<div class="controls">
<button onclick="filt('all',this)">All captured</button>
<button onclick="filt('s1',this)">S1 refusal-mismatch</button>
<button onclick="filt('nots1',this)">NOT-S1 (similar)</button>
<span class="count" id="count"></span>
</div>
{cards}
<footer>{footer}</footer>
</div>
<script>{_JS}</script>
</body></html>
"""

    task_dir = pathlib.Path(
        subprocess.run(
            ["uv", "run", "python", "scripts/task.py", "find", "952"],
            cwd=BASE,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    out_dir = task_dir / "artifacts"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "behavior_divergence_dashboard.html"
    out_path.write_text(page)
    logger.info("[done] wrote %s (%d captured pairs, %d bytes)", out_path, len(ordered), len(page))
    return out_path


def emit_s1_only() -> pathlib.Path:
    """Derive the standalone S1-only dashboard (the 12 refusal-mismatch pairs) from the
    committed full dashboard HTML — no HF re-staging; drops non-S1 cards + the filter
    controls, retitles, and pins a static count. Returns the written path."""
    task_dir = pathlib.Path(
        subprocess.run(
            ["uv", "run", "python", "scripts/task.py", "find", "952"],
            cwd=BASE,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    src = task_dir / "artifacts" / "behavior_divergence_dashboard.html"
    html = src.read_text()
    n_before = html.count("<article class='card'")
    html = re.sub(r"<article class='card' data-s1='0'.*?</article>\s*", "", html, flags=re.S)
    n_after = html.count("<article class='card'")
    assert n_after == 12 and "data-s1='0'" not in html, (n_before, n_after)
    html = html.replace(
        "<title>#952 china bank — where Qwen and Claude behave differently</title>",
        "<title>#952 china bank — the 12 pairs where behavior genuinely differs</title>",
    )
    html = html.replace(
        "<h1>#952 china bank &mdash; where Qwen and Claude behave differently</h1>",
        "<h1>#952 china bank &mdash; the 12 pairs where behavior genuinely differs "
        "(S1: Qwen refuses, Claude answers)</h1>",
    )
    html, n_btn = re.subn(
        r"<button onclick=\"filt\('[a-z0-9]+',this\)\">[^<]*</button>\s*", "", html
    )
    assert n_btn == 3, n_btn
    html = html.replace(
        "window.addEventListener('DOMContentLoaded',()=>{initAns();applyHash();});",
        "window.addEventListener('DOMContentLoaded',()=>{initAns();"
        "const c=document.getElementById('count');"
        "if(c)c.textContent=document.querySelectorAll('.card').length"
        "+' pairs (S1 refusal-mismatch only)';});",
    )
    html = html.replace("window.addEventListener('hashchange',applyHash);", "")
    html = html.replace(
        "</footer>",
        " S1-only variant derived from the full 42-pair dashboard "
        "(behavior_divergence_dashboard.html) at the same provenance.</footer>",
    )
    assert html.rstrip().endswith("</html>"), "tail lost"
    out_path = src.with_name("behavior_divergence_dashboard_s1.html")
    out_path.write_text(html)
    logger.info("[done] wrote %s (%d cards, %d bytes)", out_path, n_after, len(html))
    return out_path


if __name__ == "__main__":
    if "--s1-only" in sys.argv:
        emit_s1_only()
    else:
        build_dashboard()
