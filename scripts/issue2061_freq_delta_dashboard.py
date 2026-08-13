"""Feature-frequency vs ΔR² hover-explorer dashboard for #2061.

User-chat inline free-analysis round (2026-08-13, ask: "make a dashboard with
the frequency of feature vs the delta R^2 where each feature is a point and
hovering over it shows the auto interp description for it").

Reads EXISTING artifacts only:
  - reduced per-feature held-out R² vectors + per-stage activation counts
    staged by scripts/issue2061_predictability_dashboard.py at
    /mnt/eps-data/$USER/issue2061_dashboard/{reduced,counts}/ (fail-loud if
    absent — run that script's reduce/counts phases first);
  - per-cell + global selection-symmetric null quantiles from
    eval_results/issue_2061/null/;
  - auto-interp descriptions for the fixed dictionary
    (EleutherAI/sae-llama-3.1-8b-64x layer 29) from the public HF dataset
    EleutherAI/auto_interp_interpretations, Llama/262k/res/
    model.layers.29_feature.json (coverage 156,110/262,144 features).

Output (lazy-loading, one fetch per selected cell):
  dashboard/public/sae-freq-delta-2061.html
  dashboard/public/sae-freq-delta-2061/desc.json            (fid -> description)
  dashboard/public/sae-freq-delta-2061/cell_<pair>__<render>__<corpus>__<arm>.json

Served at https://eps.superkaiba.com/sae-freq-delta-2061.html

Usage:
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 \
    NUMEXPR_NUM_THREADS=8 uv run python scripts/issue2061_freq_delta_dashboard.py
"""

from __future__ import annotations

import argparse
import html
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE numpy import

import numpy as np  # noqa: E402

INTERP_REPO = "EleutherAI/auto_interp_interpretations"
INTERP_FILE = "Llama/262k/res/model.layers.29_feature.json"
D_SAE = 262144
DESC_MAX_CHARS = 220

STAGE_PAIRS = [
    ("base", "sft"),
    ("sft", "dpo"),
    ("dpo", "rlvr"),
    ("rlvr", "longer-rlvr"),
]
ARMS = ["context", "prefix"]


def _staging_default() -> Path:
    return Path(
        os.environ.get(
            "I2061_DASH_STAGING",
            f"/mnt/eps-data/{os.environ.get('USER', 'thomasjiralerspong')}/issue2061_dashboard",
        )
    )


def _provenance() -> dict:
    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    meta = as_metadata_dict(git_provenance())
    meta["generated_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%MZ")
    meta["generator"] = "scripts/issue2061_freq_delta_dashboard.py"
    meta["interp_source"] = f"{INTERP_REPO}:{INTERP_FILE}"
    return meta


def load_interp(staging: Path) -> dict[str, str]:
    """fid(str) -> auto-interp description, downloaded once to the staging dir."""
    dest = staging / "auto_interp" / "model.layers.29_feature.json"
    if not dest.exists():
        from huggingface_hub import hf_hub_download

        from explore_persona_space.orchestrate.hub import retry_transient

        p = retry_transient(
            lambda: hf_hub_download(
                INTERP_REPO,
                INTERP_FILE,
                repo_type="dataset",
                local_dir=str(staging / "auto_interp_dl"),
            ),
            what=f"hf_hub_download({INTERP_FILE})",
        )
        dest.parent.mkdir(parents=True, exist_ok=True)
        Path(p).rename(dest)
    desc = json.load(open(dest))
    assert len(desc) > 100_000, f"suspiciously small interp map: {len(desc)} entries"
    return desc


def list_combos(cnt_dir: Path) -> list[tuple[str, str]]:
    """(render, corpus) combos derived from the staged count filenames."""
    combos = set()
    for f in cnt_dir.glob("*_answer_L29.npz"):
        stem = f.name.replace("_answer_L29.npz", "")
        _stage, rest = stem.split("_", 1)  # stages contain no underscore
        render, corpus = rest.split("_", 1)
        assert render in ("chat", "naturalistic"), f"unparseable count file {f.name}"
        combos.add((render, corpus))
    if len(combos) != 7:
        raise ValueError(f"expected 7 render/corpus combos under {cnt_dir}, found {len(combos)}")
    return sorted(combos)


def load_null_cells(null_dir: Path) -> tuple[dict, float]:
    """(pair, render, corpus, arm) -> per-cell null row; plus the global p97.5 bar."""
    cells = {}
    for p in sorted(null_dir.glob("*_L29.jsonl")):
        with open(p) as f:
            row = json.loads(f.readline())
        cells[(row["pair"], row["render"], row["corpus"], row["arm"])] = row
    if len(cells) != 56:
        raise ValueError(f"expected 56 per-cell null rows under {null_dir}, found {len(cells)}")
    gq = json.load(open(null_dir / "GLOBAL_L29.json"))["global_null_quantiles"]["p97.5"]
    return cells, float(gq)


def _round4(a: np.ndarray) -> list[float]:
    return [round(float(v), 4) for v in a]


def build_cells(
    staging: Path, out_dir: Path, combos: list[tuple[str, str]], nulls: dict
) -> tuple[np.ndarray, int]:
    """Write one JSON payload per (pair, render, corpus, arm) cell.

    Returns (union mask over features appearing in any cell, total points).
    """
    red, cnt_dir = staging / "reduced", staging / "counts"
    union = np.zeros(D_SAE, bool)
    total = 0
    for s_from, s_to in STAGE_PAIRS:
        pair = f"{s_from}_{s_to}"
        for render, corpus in combos:
            nb_all = np.load(cnt_dir / f"{s_from}_{render}_{corpus}_answer_L29.npz")["counts"]
            na_all = np.load(cnt_dir / f"{s_to}_{render}_{corpus}_answer_L29.npz")["counts"]
            for arm in ARMS:
                za = np.load(red / f"{s_from}_{render}_{corpus}_{arm}_L29.npz")
                zb = np.load(red / f"{s_to}_{render}_{corpus}_{arm}_L29.npz")
                ra, rb = za["r2"], zb["r2"]
                mask = np.isfinite(ra) & np.isfinite(rb)
                fid = np.flatnonzero(mask)
                union |= mask
                total += len(fid)
                reg = za["reg"][fid] | zb["reg"][fid]
                nul = nulls[(pair, render, corpus, arm)]
                payload = {
                    "pair": pair,
                    "s_from": s_from,
                    "s_to": s_to,
                    "render": render,
                    "corpus": corpus,
                    "arm": arm,
                    "n_rows_before": nul["n_rows_before"],
                    "n_rows_after": nul["n_rows_after"],
                    "cell_p97_5": nul["null_quantiles_per_cell"]["p97.5"],
                    "true_max": nul["true_max_delta_r2"],
                    "true_argmax": nul["true_argmax_feature_id"],
                    "fid": fid.tolist(),
                    "d": _round4(rb[fid] - ra[fid]),
                    "ra": _round4(ra[fid]),
                    "rb": _round4(rb[fid]),
                    "nb": nb_all[fid].tolist(),
                    "na": na_all[fid].tolist(),
                    "reg_idx": np.flatnonzero(reg).tolist(),
                }
                name = f"cell_{pair}__{render}__{corpus}__{arm}.json"
                (out_dir / name).write_text(json.dumps(payload, separators=(",", ":")))
                print(f"[cell] {name}: {len(fid)} points", flush=True)
    return union, total


CAVEATS = (
    "<b>Read with #2061's headline caveats.</b> "
    "(1) No per-feature gain survives the selection-symmetric max-statistic null "
    "(GLOBAL p97.5 = {gq:.2f}) — every point below is DESCRIPTIVE, multiplicity-uncorrected. "
    "(2) Extreme ΔR² values are dominated by rarely-active features — treat points at the "
    "left edge (1–9 rows active) as noise-grade; the frequency axis exists to make exactly "
    "that visible. (3) The fixed SAE dictionary explains only ~33% of answer-state variance, "
    "so per-feature reads are minority-variance-scoped. (4) Prefix arms on chat renders have "
    "a row-constant prefix input — at-chance by construction. (5) Auto-interp descriptions "
    "cover 156,110/262,144 features (EleutherAI, generated on base-model token-level "
    "activations — they describe the base dictionary, not any post-trained stage) and are "
    "themselves LLM-generated: treat as hints, not ground truth. (6) ⚠reg marks features "
    "whose ridge fit was regularization-limited at an endpoint (λ pinned at the audited "
    "grid edge; not a clean read)."
)


def write_html(out_html: Path, combos: list[tuple[str, str]], gq: float, prov: dict) -> None:
    pairs_js = json.dumps([list(p) for p in STAGE_PAIRS])
    combos_js = json.dumps([f"{r}__{c}" for r, c in combos])
    caveats = CAVEATS.format(gq=gq)
    page = f"""<!DOCTYPE html><html><head><meta charset='utf-8'>
<title>#2061 SAE feature frequency vs ΔR²</title>
<script src='https://cdn.plot.ly/plotly-2.35.2.min.js'></script>
<style>body{{font-family:system-ui,sans-serif;margin:24px;max-width:1250px}}
select{{font-size:14px;padding:2px 6px;margin-right:14px}}
.caveat{{background:#fff6e0;border:1px solid #e0c060;padding:10px 14px;font-size:13px;margin:12px 0}}
.small{{font-size:12px;color:#444}}.mono{{font-family:monospace}}
#stats{{font-size:13px;margin:6px 0}}#plot{{width:100%;height:640px}}</style></head><body>
<h1>#2061 — SAE feature activation frequency vs per-feature ΔR²</h1>
<p class='small'>Each point is one SAE feature with a defined held-out R² at BOTH stages of the
selected transition (map: pooled context/prefix activation → answer-state SAE feature vector,
EleutherAI/sae-llama-3.1-8b-64x L29, TopK k=32, Tülu-3 ladder). x = rows where the feature is
active in the TopK k=32 code (min over the two stages; log axis, 0 plotted at 0.6, ±5%% x-jitter
for visibility — hover shows exact counts). y = ΔR² (later − earlier stage). Hover shows the
feature's auto-interp description ({INTERP_REPO}).</p>
<div class='caveat'>{caveats}</div>
<div>
<label>transition <select id='pair'></select></label>
<label>render/corpus <select id='combo'></select></label>
<label>arm <select id='arm'><option>context</option><option>prefix</option></select></label>
</div>
<div id='stats'></div>
<div id='plot'></div>
<p class='small'>Generated {html.escape(prov["generated_utc"])} · commit
<span class='mono'>{html.escape(prov.get("git_commit", "unknown")[:10])}</span> · generator
<span class='mono'>scripts/issue2061_freq_delta_dashboard.py</span> · descriptions:
<a href='https://huggingface.co/datasets/{INTERP_REPO}'>{INTERP_REPO}</a>
(Llama/262k/res L29) · companion:
<a href='sae-predictability-2061.html'>per-transition winners + distributions</a> · task:
<a href='https://eps.superkaiba.com/tasks/2061'>#2061</a>.</p>
<script>
const PAIRS = {pairs_js};
const COMBOS = {combos_js};
const GQ = {gq};
let DESC = null;
const cellCache = {{}};

const pairSel = document.getElementById('pair');
const comboSel = document.getElementById('combo');
const armSel = document.getElementById('arm');
for (const [a, b] of PAIRS) {{
  const o = document.createElement('option');
  o.value = a + '_' + b; o.textContent = a + ' \\u2192 ' + b; pairSel.appendChild(o);
}}
for (const c of COMBOS) {{
  const o = document.createElement('option');
  o.value = c; o.textContent = c.replace('__', ' / '); comboSel.appendChild(o);
}}
// default = the headline winner cell: dpo->rlvr, chat/if11k, context
pairSel.value = 'dpo_rlvr'; comboSel.value = 'chat__if11k'; armSel.value = 'context';

function jitter(fid) {{ // deterministic +/-5% in log10 space, keyed on feature id
  return ((fid * 2654435761 % 4294967296) / 4294967296 - 0.5) * 0.044;
}}
function wrapDesc(s) {{
  if (!s) return '<i>no auto-interp label for this feature</i>';
  const words = s.split(' '); const lines = []; let cur = '';
  for (const w of words) {{
    if ((cur + ' ' + w).length > 62) {{ lines.push(cur); cur = w; }}
    else cur = cur ? cur + ' ' + w : w;
  }}
  if (cur) lines.push(cur);
  return lines.join('<br>');
}}

async function render() {{
  const key = 'cell_' + pairSel.value + '__' + comboSel.value + '__' + armSel.value;
  document.getElementById('stats').textContent = 'loading \\u2026';
  if (!DESC) DESC = await (await fetch('sae-freq-delta-2061/desc.json')).json();
  if (!cellCache[key]) {{
    const resp = await fetch('sae-freq-delta-2061/' + key + '.json');
    if (!resp.ok) {{
      document.getElementById('stats').textContent = 'cell payload missing: ' + key;
      return;
    }}
    cellCache[key] = await resp.json();
  }}
  const c = cellCache[key];
  const regSet = new Set(c.reg_idx);
  const mk = () => ({{x: [], y: [], text: [], custom: []}});
  const tr = {{plain: mk(), reg: mk()}};
  for (let i = 0; i < c.fid.length; i++) {{
    const fid = c.fid[i];
    const support = Math.min(c.nb[i], c.na[i]);
    const x = Math.max(support, 0.6) * Math.pow(10, jitter(fid));
    const t = regSet.has(i) ? tr.reg : tr.plain;
    t.x.push(x); t.y.push(c.d[i]);
    t.custom.push([fid, wrapDesc(DESC[fid]), c.nb[i], c.na[i],
                   c.ra[i].toFixed(3), c.rb[i].toFixed(3)]);
  }}
  const hover = '<b>feature %{{customdata[0]}}</b><br>' +
    '\\u0394R\\u00b2 = %{{y:.4f}}  (R\\u00b2 %{{customdata[4]}} \\u2192 %{{customdata[5]}})<br>' +
    'rows active: %{{customdata[2]}} \\u2192 %{{customdata[3]}}<br>' +
    '%{{customdata[1]}}<extra></extra>';
  const traces = [{{
    type: 'scattergl', mode: 'markers', name: 'feature',
    x: tr.plain.x, y: tr.plain.y, customdata: tr.plain.custom,
    marker: {{color: '#0072B2', size: 4, opacity: 0.45}}, hovertemplate: hover,
  }}];
  if (tr.reg.x.length) traces.push({{
    type: 'scattergl', mode: 'markers', name: '\\u26a0 regularization-limited endpoint',
    x: tr.reg.x, y: tr.reg.y, customdata: tr.reg.custom,
    marker: {{color: '#E69F00', size: 4, opacity: 0.6}}, hovertemplate: hover,
  }});
  const shapes = [
    {{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: 0, y1: 0,
      line: {{color: '#000', width: 0.8}}}},
    {{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: c.cell_p97_5, y1: c.cell_p97_5,
      line: {{color: '#555', width: 1, dash: 'dot'}}}},
    {{type: 'line', xref: 'paper', x0: 0, x1: 1, y0: GQ, y1: GQ,
      line: {{color: '#000', width: 1, dash: 'dash'}}}},
  ];
  Plotly.react('plot', traces, {{
    xaxis: {{type: 'log', title: 'rows active in TopK code (min of the two stages; 0 at 0.6)',
             tickvals: [0.6, 1, 3, 10, 30, 100, 300, 1000, 3000, 10000],
             ticktext: ['0', '1', '3', '10', '30', '100', '300', '1k', '3k', '10k']}},
    yaxis: {{title: '\\u0394R\\u00b2 (' + c.s_to + ' \\u2212 ' + c.s_from + ')'}},
    shapes: shapes,
    margin: {{t: 30}}, showlegend: tr.reg.x.length > 0,
    legend: {{orientation: 'h', y: 1.06}},
  }}, {{responsive: true}});
  document.getElementById('stats').innerHTML =
    '<b>' + c.s_from + ' \\u2192 ' + c.s_to + ' \\u00b7 ' + c.render + '/' + c.corpus +
    ' \\u00b7 ' + c.arm + ' arm</b> \\u2014 ' + c.fid.length.toLocaleString() +
    ' features with defined \\u0394R\\u00b2 \\u00b7 rows per stage: ' +
    c.n_rows_before.toLocaleString() + ' \\u2192 ' + c.n_rows_after.toLocaleString() +
    ' \\u00b7 dotted = this cell\\u2019s max-null p97.5 (' + c.cell_p97_5.toFixed(2) +
    ') \\u00b7 dashed = GLOBAL bar (' + GQ.toFixed(2) + ') \\u00b7 cell true max \\u0394R\\u00b2 = ' +
    c.true_max.toFixed(3) + ' (feature ' + c.true_argmax + ')';
}}
for (const el of [pairSel, comboSel, armSel]) el.addEventListener('change', render);
render();
</script></body></html>"""
    out_html.write_text(page)
    print(f"[html] wrote {out_html} ({out_html.stat().st_size / 1e3:.1f} KB)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--staging-dir", type=Path, default=_staging_default())
    ap.add_argument(
        "--out-html", type=Path, default=Path("dashboard/public/sae-freq-delta-2061.html")
    )
    args = ap.parse_args()

    red, cnt_dir = args.staging_dir / "reduced", args.staging_dir / "counts"
    n_red = len(list(red.glob("*.npz"))) if red.exists() else 0
    n_cnt = len(list(cnt_dir.glob("*.npz"))) if cnt_dir.exists() else 0
    if n_red != 70 or n_cnt != 35:
        raise SystemExit(
            f"staged inputs incomplete ({n_red}/70 reduced, {n_cnt}/35 counts) — run "
            "scripts/issue2061_predictability_dashboard.py --phase reduce / counts first."
        )

    out_dir = args.out_html.parent / args.out_html.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    combos = list_combos(cnt_dir)
    nulls, gq = load_null_cells(Path("eval_results/issue_2061/null"))
    union, total = build_cells(args.staging_dir, out_dir, combos, nulls)

    interp = load_interp(args.staging_dir)
    desc = {
        int(j): interp[str(j)][:DESC_MAX_CHARS] for j in np.flatnonzero(union) if str(j) in interp
    }
    prov = _provenance()
    prov["n_union_features"] = int(union.sum())
    prov["n_points_total"] = total
    prov["n_descriptions"] = len(desc)
    (out_dir / "desc.json").write_text(json.dumps(desc, separators=(",", ":")))
    (out_dir / "meta.json").write_text(json.dumps(prov, indent=1))
    print(
        f"[desc] {len(desc)}/{int(union.sum())} union features carry an auto-interp label",
        flush=True,
    )
    write_html(args.out_html, combos, gq, prov)
    print("[done]", flush=True)


if __name__ == "__main__":
    sys.exit(main())
