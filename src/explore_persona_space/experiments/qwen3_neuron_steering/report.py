"""Plots and the self-contained HTML results page (with a reply browser)."""

from __future__ import annotations

import base64
import gzip
import html
import json
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from explore_persona_space.experiments.qwen3_neuron_steering import config as C

# One color = one meaning, across every figure.
COLORS = {
    "persona": "#6a3d9a",
    "topic": "#d95f02",
    "ontask": "#1b9e77",
    "coherence": "#666666",
}
LAYER_COLORS = {8: "#1f78b4", 9: "#33a02c", 18: "#e31a1c"}
SERIES = (
    ("persona_self", "persona voice on self", "persona"),
    ("topic_nonself", "topic on non-self", "topic"),
    ("ontask_nonself", "on-task on non-self", "ontask"),
    ("coherence", "coherent (all replies)", "coherence"),
)


def _rate(block: dict) -> tuple[float, float, float]:
    lo, hi = block["wilson"] if block["wilson"] else (np.nan, np.nan)
    r = block["rate"] if block["rate"] is not None else np.nan
    return r, lo, hi


def dose_curve_figure(target: C.Target, by_dose: dict[float, dict], scale: float, path: Path):
    """Four rates vs signed dose; baseline at 0; opposite-sign checks as hollow markers.

    ``scale`` converts alpha to normalized dose (alpha * ||w|| / median residual norm).
    """
    main = sorted([0.0, *target.sweep])
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    for key, label, color_key in SERIES:
        xs, rs, los, his = [], [], [], []
        for d in main:
            r, lo, hi = _rate(by_dose[d][key])
            xs.append(d)
            rs.append(r)
            los.append(lo)
            his.append(hi)
        color = COLORS[color_key]
        ax.plot(xs, rs, "-o", color=color, label=label, ms=4, lw=1.6)
        ax.fill_between(xs, los, his, color=color, alpha=0.15, lw=0)
        for d in target.opposite:
            r, lo, hi = _rate(by_dose[d][key])
            ax.errorbar(
                [d],
                [r],
                yerr=[[max(0.0, r - lo)], [max(0.0, hi - r)]],
                fmt="o",
                mfc="white",
                color=color,
                ms=5,
            )
    ax.axvline(0, color="#bbbbbb", lw=0.8, zorder=0)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("dose alpha (added to neuron activation)")
    ax.set_ylabel("fraction of replies")
    ax.set_title(f"{target.name}: {target.persona} persona vs {target.theme} topic")
    top = ax.secondary_xaxis("top", functions=(lambda x: x * scale, lambda x: x / scale))
    top.set_xlabel("normalized dose (push norm / median residual norm)")
    ax.legend(fontsize=8, loc="best", frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def per_question_heatmap(target: C.Target, by_dose: dict[float, dict], path: Path):
    """Per-question persona rate (rows: 40 self questions; columns: doses incl. 0)."""
    doses = sorted([0.0, *target.sweep], key=abs)
    qs = [q for q in C.all_questions() if q.group == "self"]
    mat = np.full((len(qs), len(doses)), np.nan)
    for j, d in enumerate(doses):
        rates = {e["q_index"]: e["rate"] for e in by_dose[d]["persona_per_question"]}
        for i, q in enumerate(qs):
            mat[i, j] = rates.get(q.q_index, np.nan)
    fig, ax = plt.subplots(figsize=(1.2 + 0.55 * len(doses), 9.5))
    im = ax.imshow(mat, vmin=0, vmax=1, cmap="Purples", aspect="auto")
    ax.set_xticks(range(len(doses)), [f"{d:+g}" if d else "0" for d in doses])
    labels = [("* " if q.id_probe else "") + q.text for q in qs]
    ax.set_yticks(range(len(qs)), labels, fontsize=7)
    ax.set_xlabel("dose alpha")
    ax.set_title(f"{target.name}: {target.persona} persona rate per self question", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="persona rate")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def topic_by_group_figure(target: C.Target, by_dose: dict[float, dict], path: Path):
    """Topic leakage rate per non-self group vs dose."""
    main = sorted([0.0, *target.sweep])
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    for group, style in zip(C.NONSELF_GROUPS, ("-o", "--s", ":^"), strict=True):
        rs = [by_dose[d]["topic_nonself_by_group"][group]["rate"] for d in main]
        rs = [np.nan if r is None else r for r in rs]
        ax.plot(main, rs, style, color=COLORS["topic"], label=group, ms=4)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("dose alpha")
    ax.set_ylabel(f"{target.theme} topic rate")
    ax.set_title(f"{target.name}: topic leakage by question group")
    ax.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def controls_figure(controls: Sequence[dict], targets_at_source: Sequence[dict], path: Path):
    """Non-assistant identity rate vs coherence for every control and target condition."""
    fig, ax = plt.subplots(figsize=(6.0, 4.2))
    for layer, color in LAYER_COLORS.items():
        pts = [c for c in controls if c["layer"] == layer]
        if not pts:
            continue
        ax.scatter(
            [c["metrics"]["nonassistant_self"]["rate"] for c in pts],
            [c["metrics"]["coherence"]["rate"] for c in pts],
            color=color,
            s=22,
            alpha=0.75,
            label=f"random L{layer}",
        )
    for t in targets_at_source:
        ax.scatter(
            [t["metrics"]["nonassistant_self"]["rate"]],
            [t["metrics"]["coherence"]["rate"]],
            marker="*",
            s=160,
            color=LAYER_COLORS[t["layer"]],
            edgecolor="black",
            lw=0.6,
        )
        ax.annotate(
            t["name"],
            (t["metrics"]["nonassistant_self"]["rate"], t["metrics"]["coherence"]["rate"]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlabel("non-assistant identity on self questions")
    ax.set_ylabel("coherent (all replies)")
    ax.set_title("Random neurons at matched push vs targets at source dose (stars)")
    ax.legend(fontsize=8, frameon=False, loc="lower left")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def _img(path: Path) -> str:
    data = base64.b64encode(path.read_bytes()).decode()
    return f'<img alt="{html.escape(path.stem)}" src="data:image/png;base64,{data}">'


def _fmt(block: dict | None) -> str:
    if not block or block.get("rate") is None:
        return "n/a"
    lo, hi = block["wilson"]
    return f"{100 * block['rate']:.0f}% <span class=ci>[{100 * lo:.0f}, {100 * hi:.0f}]</span>"


def browser_payload(label_rows: Sequence[dict]) -> str:
    """Gzip + base64 the compact reply table for the in-page browser."""
    keys = [
        "cond", "role", "neuron", "alpha", "group", "question", "sample", "text",
        "speaker", "note", "theme", "on_task", "coherent", "intrusion",
    ]  # fmt: skip
    rows = []
    for r in label_rows:
        lab = r.get("labels") or {}
        rows.append(
            [
                r["condition"],
                r["role"],
                r["neuron_name"],
                r["alpha"],
                r["group"],
                r["question"],
                r["sample"],
                r["text"],
                lab.get("speaker", "dropped"),
                lab.get("speaker_note", ""),
                lab.get("theme", ""),
                lab.get("on_task", ""),
                lab.get("coherent", ""),
                lab.get("intrusion", ""),
            ]
        )
    raw = json.dumps({"keys": keys, "rows": rows}, ensure_ascii=False).encode()
    return base64.b64encode(gzip.compress(raw, 9)).decode()


CSS = """
:root{--bg:#ffffff;--fg:#1d1d1f;--muted:#666;--line:#e3e3e3;--chip:#f2f2f2;--accent:#6a3d9a}
@media (prefers-color-scheme: dark){:root:not([data-theme="light"]){--bg:#17171a;--fg:#eaeaea;
--muted:#a0a0a0;--line:#333;--chip:#26262b;--accent:#b39ddb}}
:root[data-theme="dark"]{--bg:#17171a;--fg:#eaeaea;--muted:#a0a0a0;--line:#333;--chip:#26262b;
--accent:#b39ddb}
body{background:var(--bg);color:var(--fg);font:15px/1.5 system-ui,sans-serif;margin:0}
main{max-width:1100px;margin:0 auto;padding:16px}
h1{font-size:1.5rem}h2{margin-top:2rem;border-bottom:1px solid var(--line)}
table{border-collapse:collapse;font-size:12.5px;width:100%}
th,td{border-bottom:1px solid var(--line);padding:3px 6px;text-align:right;white-space:nowrap}
th:first-child,td:first-child{text-align:left}.ci{color:var(--muted);font-size:11px}
.scroll{overflow-x:auto}img{max-width:100%;background:#fff;border-radius:4px}
.figs{display:grid;grid-template-columns:repeat(auto-fit,minmax(320px,1fr));gap:12px}
.verdict{background:var(--chip);padding:8px 12px;border-radius:6px;margin:8px 0}
.controls{display:flex;flex-wrap:wrap;gap:8px;margin:8px 0}
select,input{background:var(--bg);color:var(--fg);border:1px solid var(--line);padding:4px}
.reply{border:1px solid var(--line);border-radius:6px;padding:8px;margin:6px 0}
pre{overflow-x:auto;max-width:100%}
.reply pre{white-space:pre-wrap;overflow-wrap:anywhere;font:13px/1.45 ui-monospace,monospace;
margin:6px 0 0}
.chip{display:inline-block;background:var(--chip);border-radius:10px;padding:0 8px;
margin-right:4px;font-size:12px}.q{color:var(--muted);font-size:13px}
"""

JS = """
const BLOB = document.getElementById('data').textContent.trim();
let DATA = null, PAGE = 0; const PER = 100;
async function load(){
  const bin = Uint8Array.from(atob(BLOB), c => c.charCodeAt(0));
  const ds = new Blob([bin]).stream().pipeThrough(new DecompressionStream('gzip'));
  DATA = JSON.parse(await new Response(ds).text());
  const K = Object.fromEntries(DATA.keys.map((k,i)=>[k,i]));
  DATA.K = K;
  const neurons = [...new Set(DATA.rows.map(r=>r[K.neuron]))].sort();
  fill('neuron', neurons); onNeuron();
}
function fill(id, vals){const s=document.getElementById(id);s.innerHTML='';
  for(const v of vals){const o=document.createElement('option');
    o.value=v;o.textContent=v;s.appendChild(o);}}
function onNeuron(){const K=DATA.K, n=document.getElementById('neuron').value;
  const ds=[...new Set(DATA.rows.filter(r=>r[K.neuron]===n).map(r=>r[K.alpha]))]
    .sort((a,b)=>Math.abs(a)-Math.abs(b));
  fill('dose', ['all', ...ds.map(String)]); PAGE=0; render();}
function render(){const K=DATA.K;
  const n=document.getElementById('neuron').value, d=document.getElementById('dose').value;
  const g=document.getElementById('group').value, sp=document.getElementById('speaker').value;
  const q=document.getElementById('search').value.toLowerCase();
  const rows=DATA.rows.filter(r=>r[K.neuron]===n && (d==='all'||String(r[K.alpha])===d)
    && (g==='all'||(g==='nonself'?r[K.group]!=='self':r[K.group]===g))
    && (sp==='all'||r[K.speaker]===sp)
    && (!q||r[K.text].toLowerCase().includes(q)||r[K.question].toLowerCase().includes(q)));
  const pages=Math.max(1,Math.ceil(rows.length/PER)); PAGE=Math.min(PAGE,pages-1);
  document.getElementById('count').textContent=rows.length+' replies, page '+(PAGE+1)+'/'+pages;
  const box=document.getElementById('replies'); box.innerHTML='';
  for(const r of rows.slice(PAGE*PER,(PAGE+1)*PER)){
    const div=document.createElement('div'); div.className='reply';
    const chips=[r[K.cond],'dose '+r[K.alpha],r[K.group],'speaker: '+r[K.speaker]+
      (r[K.note]?' ('+r[K.note]+')':''),'theme: '+r[K.theme],'on task: '+r[K.on_task],
      'coherent: '+r[K.coherent]].concat(r[K.intrusion]&&r[K.intrusion]!=='none'?
      ['intrusion: '+r[K.intrusion]]:[]);
    div.innerHTML=chips.map(c=>'<span class=chip></span>').join('')+
      '<div class=q></div><pre></pre>';
    div.querySelectorAll('.chip').forEach((el,i)=>el.textContent=chips[i]);
    div.querySelector('.q').textContent='Q: '+r[K.question]+'  (sample '+r[K.sample]+')';
    div.querySelector('pre').textContent=r[K.text]; box.appendChild(div);}
}
function page(delta){PAGE=Math.max(0,PAGE+delta);render();}
load();
"""


def write_html(
    path: Path,
    *,
    setup: str,
    verdicts: Sequence[tuple[str, str]],
    summary_rows: Sequence[dict],
    figures: Sequence[Path],
    controls: Sequence[dict],
    diagnostics: dict,
    label_rows: Sequence[dict],
) -> None:
    """Write the self-contained results page."""
    head = [
        "neuron", "dose", "norm. dose", "persona (self)", "persona (Who/What)", "always",
        "topic (non-self)", "on-task", "coherent", "repeat", "cap hit", "n",
    ]  # fmt: skip
    trs = []
    for s in summary_rows:
        m = s["metrics"]
        alw = m.get("always_embodies")
        trs.append(
            "<tr>"
            + "".join(
                f"<td>{c}</td>"
                for c in (
                    html.escape(s["name"]),
                    f"{s['alpha']:+g}" if s["alpha"] else "0",
                    f"{s['alpha_norm']:+.2f}" if s["alpha_norm"] is not None else "",
                    _fmt(m.get("persona_self")),
                    _fmt(m.get("persona_idprobe")),
                    f"{alw['k']}/{alw['n']}" if alw else "",
                    _fmt(m.get("topic_nonself")),
                    _fmt(m["ontask_nonself"]),
                    _fmt(m["coherence"]),
                    _fmt(m["repetition"]),
                    _fmt(m["cap_hit"]),
                    str(m["n_labelled"]),
                )
            )
            + "</tr>"
        )
    ctrl_head = [
        "control", "matched to", "alpha", "non-assistant (self)", "religion", "nature",
        "on-task", "coherent", "top intrusions", "other speakers",
    ]  # fmt: skip
    ctrs = []
    for c in controls:
        m = c["metrics"]
        ctrs.append(
            "<tr>"
            + "".join(
                f"<td>{x}</td>"
                for x in (
                    html.escape(c["name"]),
                    html.escape(c["matched_target"]),
                    f"{c['alpha']:+.2f}",
                    _fmt(m["nonassistant_self"]),
                    _fmt(m["religion_nonself"]),
                    _fmt(m["nature_nonself"]),
                    _fmt(m["ontask_nonself"]),
                    _fmt(m["coherence"]),
                    html.escape("; ".join(f"{t} ({n})" for t, n in m["top_intrusions"][:3])),
                    html.escape("; ".join(f"{t} ({n})" for t, n in m["top_other_speakers"][:3])),
                )
            )
            + "</tr>"
        )
    verdict_html = "".join(
        f"<div class=verdict><b>{html.escape(n)}</b>: {html.escape(t)}</div>" for n, t in verdicts
    )
    figs = "".join(f"<figure>{_img(p)}</figure>" for p in figures)
    diag = html.escape(json.dumps(diagnostics, indent=2))
    page = f"""<!doctype html><html lang=en><head><meta charset=utf-8>
<meta name=viewport content="width=device-width,initial-scale=1">
<title>Qwen3 Neuron Steering</title><style>{CSS}</style></head><body><main>
<h1>Single-neuron steering on Qwen3-8B: persona or concept?</h1>
<p>{html.escape(setup)}</p>
<h2>Verdicts</h2>{verdict_html}
<h2>Summary</h2><p class=q>Rates with Wilson 95% intervals. Persona (self) uses 5 samples
per self question (40 questions). Persona (Who/What) uses all 100 identity-probe samples.
"Always" counts self questions with at least 90% of samples in persona. Bootstrap-over-question
intervals are in summary.json.</p>
<div class=scroll><table><tr>{"".join(f"<th>{h}</th>" for h in head)}</tr>{"".join(trs)}
</table></div>
<h2>Plots</h2><div class=figs>{figs}</div>
<h2>Random controls</h2><div class=scroll><table><tr>
{"".join(f"<th>{h}</th>" for h in ctrl_head)}</tr>{"".join(ctrs)}</table></div>
<h2>Labelling diagnostics</h2><pre>{diag}</pre>
<h2>Reply browser</h2>
<div class=controls>
<select id=neuron onchange="onNeuron()"></select>
<select id=dose onchange="PAGE=0;render()"></select>
<select id=group onchange="PAGE=0;render()"><option>all</option><option>self</option>
<option value=nonself>non-self</option><option>other</option><option>task</option>
<option>creative</option></select>
<select id=speaker onchange="PAGE=0;render()"><option>all</option><option>assistant</option>
<option>god</option><option>nature</option><option>other</option><option>none</option>
<option>dropped</option></select>
<input id=search placeholder="search text" oninput="PAGE=0;render()">
<button onclick="page(-1)">prev</button><button onclick="page(1)">next</button>
<span id=count class=q></span></div>
<div id=replies></div>
<script id=data type="application/octet-stream">{browser_payload(label_rows)}</script>
<script>{JS}</script>
</main></body></html>"""
    path.write_text(page, encoding="utf-8")
