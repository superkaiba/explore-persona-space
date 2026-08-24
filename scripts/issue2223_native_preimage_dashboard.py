"""Issue #2223 NAP round — self-contained offline dashboard (single HTML, no CDN).

Reads the round's per-cell replay JSONs + judged scores
(``<out-root>/<slug>/<round-subdir>/...``) and writes ONE self-contained HTML
file: embedded JSON records + vanilla-JS client-side MULTI-select filters
(scenario, layer config, axis family, op, strength, arm — every facet is a
multi-select, all values selected by default; an empty selection is treated
as "no filter") applied at BOTH levels —

  plot level:  hand-rolled SVG re-renders (per-arm average bars for a chosen
               DV incl. drift_l32 / drift_band; TWO drift-vs-turn polyline
               panels — layer-32 PRIMARY + band-mean secondary — for the
               filtered records);
  data level:  per-arm-per-turn table (drift_l32 / drift_band / harm /
               coherence / realized dose) + expandable VERBATIM conversation
               text per record, and a download button serving the embedded
               JSON (records + extraction diagnostics) as a Blob.

Extraction diagnostics (``--extractions-dir``, default
``<out-root>/<slug>/extractions``): when present, ``axis_cos.json`` /
``map_metrics.json`` / ``tau_map.json`` are embedded into the payload
(downloadable) and rendered as tables — the H1 axis-fidelity gate + per-band-
layer cosines, the per-layer map-validity metrics (held-out pooled R² vs
identity+bias baseline, selected λ, kNN retrieval) + preimage amplification
diagnostics, and the H3 cosine table — plus a raw-JSON ``<details>`` block
per file.

Header disclosures (CLAUDE.md ad-hoc summary rules): per-arm provenance
(on-policy sampled generations, temp 0.7 / top_p 0.9, ONE fixed-seed
trajectory per arm; anchors additionally at seeds 43/44), matched-target
(every drift value is the projection on the SAME published Lu assistant axis;
every harm/coherence score from the SAME judge rubric), and a
no-display-substitution statement (assistant text is the stored raw
completion, verbatim).

Content hygiene: the HTML embeds stored completions verbatim (disclosed); this
script's own stdout carries counts and paths only, never message text.
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path


def _ensure_repo_root_on_syspath() -> Path:
    root = Path(__file__).resolve().parents[1]
    assert (root / "scripts" / "issue2223_native_preimage_dashboard.py").exists(), root
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO = _ensure_repo_root_on_syspath()

from scripts.issue2223_casestudy_figures import (  # noqa: E402
    PRIMARY_DRIFT_LAYER,
    _band_mean_projection,
    _layer_projection,
    family_of,
    load_cells_by_stem,
)
from scripts.issue2223_casestudy_replay import (  # noqa: E402
    CS_ARMS,
    SCENARIOS,
)

NAP_LABEL = "native_axis_fidelity_preimage"
DIAG_FILES = ("axis_cos.json", "map_metrics.json", "tau_map.json")


def _arm_op_strength(arm: str) -> tuple[str, str]:
    spec = CS_ARMS[arm]
    op = spec.get("op", spec.get("engine", "none"))
    if "percentile" in spec:
        strength = str(spec["percentile"])
    elif "k" in spec:
        strength = f"k{spec['k']}"
    elif op == "axis_replace":
        strength = "replace"
    else:
        strength = "n/a"
    return op, strength


def _score_lookup(scores: dict | None, key: str, turn: int) -> float | None:
    if not scores:
        return None
    rec = scores.get(key, {}).get(str(turn))
    return rec.get("score") if rec else None


def build_records(model_root: Path) -> list[dict]:
    """One record per cell: identity facets + per-turn DV/dose/text rows."""
    records: list[dict] = []
    for sc in SCENARIOS:
        cells = load_cells_by_stem(model_root, sc)
        if not cells:
            continue
        blocks: dict[str, dict | None] = {}
        for dv, fname in (("harm", f"scores_{sc}.json"), ("coherence", f"coherence_{sc}.json")):
            p = model_root / "judged" / fname
            blocks[dv] = json.loads(p.read_text())["cells"] if p.exists() else None
        for key, cell in sorted(cells.items()):
            arm = cell["arm"]
            op, strength = _arm_op_strength(arm)
            turns = []
            for rec in cell["turns"]:
                rf = rec.get("realized_firing") or {}
                turns.append(
                    {
                        "turn": rec["turn"],
                        "user": rec["user"],
                        "assistant": rec["assistant"],
                        "drift_l32": _layer_projection(
                            cell, rec, "answer_mean", PRIMARY_DRIFT_LAYER
                        ),
                        "drift_band": _band_mean_projection(cell, rec, "answer_mean"),
                        "harm": _score_lookup(blocks["harm"], key, rec["turn"]),
                        "coherence": _score_lookup(blocks["coherence"], key, rec["turn"]),
                        "fired_frac": rf.get("mean_fired_frac"),
                        "abs_dproj": rf.get("mean_abs_dproj"),
                        "cap_hit": rec.get("cap_hit"),
                    }
                )
            records.append(
                {
                    "cell": key,
                    "scenario": sc,
                    "arm": arm,
                    "family": family_of(arm),
                    "layers": cell["layers"],
                    "seed": int(cell.get("seed_base", 42)),
                    "op": op,
                    "strength": strength,
                    "cap_hit_frac": cell.get("cap_hit_frac"),
                    "turns": turns,
                }
            )
    return records


def load_diagnostics(ext_dir: Path) -> dict[str, dict]:
    """The three extraction-diagnostic JSONs, keyed by filename (present only)."""
    diags: dict[str, dict] = {}
    for name in DIAG_FILES:
        p = ext_dir / name
        if p.exists():
            diags[name] = json.loads(p.read_text())
    return diags


def _fmt(v, nd: int = 4) -> str:
    if v is None:
        return "—"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, int):
        return str(v)
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return html.escape(str(v))


class _Safe(str):
    """Marker for a cell carrying INTENTIONAL pre-built HTML markup.

    ``_tbl`` escapes every plain ``str`` cell (r2 concern
    dashboard-diagnostic-html-escaping); wrapping in ``_Safe`` is the ONLY
    explicit opt-out — never pass pre-escaped plain strings.
    """


def _cell(c) -> str:
    if isinstance(c, _Safe):
        return str(c)
    if isinstance(c, str):
        return html.escape(c)
    return _fmt(c)


def _tbl(headers: list[str], rows: list[list]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{_cell(c)}</td>" for c in row) + "</tr>" for row in rows)
    return f"<table><tr>{head}</tr>{body}</table>"


def render_diagnostics_html(diags: dict[str, dict]) -> str:
    """Server-side tables for the H1 gate / map validity / H3 cosines + raw JSON."""
    if not diags:
        return (
            "<h2>Extraction diagnostics</h2><div class='setup'>none found under the "
            "--extractions-dir (run the capture map/axes phases first)</div>"
        )
    parts = ["<h2>Extraction diagnostics</h2>"]
    ax = diags.get("axis_cos.json")
    if ax:
        g = ax.get("h1_gate", {})
        parts.append(
            "<h3>H1 axis-fidelity gate (registered floors: band ALL ≥0.90; l32 ≥0.71)</h3>"
        )
        parts.append(
            _tbl(
                [
                    "band_min_cos",
                    "band_mean_cos",
                    "band_all_pass",
                    f"mid_cos (layer {g.get('mid_layer', '?')})",
                    "mid_pass",
                    "classification",
                ],
                [
                    [
                        g.get("band_min_cos"),
                        g.get("band_mean_cos"),
                        g.get("band_all_pass"),
                        g.get("mid_cos"),
                        g.get("mid_pass"),
                        str(g.get("classification")),  # plain str — _tbl escapes
                    ]
                ],
            )
        )
        band = [str(li) for li in ax.get("band_layers", [])]
        cosd = ax.get("cos_reextracted_vs_reference", {})
        if band and cosd:
            parts.append("<h3>Per-band-layer cos(re-extracted answer axis, reference)</h3>")
            parts.append(_tbl(["layer", "cos"], [[li, cosd.get(li)] for li in band]))
        h3 = ax.get("h3_table", {})
        if h3:
            cols = sorted({c for rec in h3.values() for c in rec})
            parts.append("<h3>H3 cosine table (per band layer)</h3>")
            parts.append(
                _tbl(
                    ["layer", *cols],
                    [[li, *[h3[li].get(c) for c in cols]] for li in sorted(h3, key=int)],
                )
            )
    mm = diags.get("map_metrics.json")
    if mm and mm.get("map"):
        parts.append(
            "<h3>Map validity per layer (preimage verdict-eligible iff held-out pooled "
            "R² &gt; identity+bias R² at every band layer)</h3>"
        )
        rows = []
        for li in sorted(mm["map"], key=int):
            rec = mm["map"][li]
            knn = json.dumps(rec.get("knn_retrieval"), separators=(",", ":"))
            rows.append(
                [
                    li,
                    rec.get("n_pool"),
                    rec.get("n_train"),
                    rec.get("d"),
                    rec.get("lambda_selected"),
                    rec.get("lambda_edge_of_grid"),
                    rec.get("r2_heldout_pooled"),
                    rec.get("r2_identity_bias_pooled"),
                    knn[:160],  # plain str — _tbl escapes
                ]
            )
        parts.append(
            _tbl(
                [
                    "layer",
                    "n_pool",
                    "n_train",
                    "d",
                    "λ selected",
                    "λ edge",
                    "R² held-out",
                    "R² identity+bias",
                    "kNN retrieval",
                ],
                rows,
            )
        )
        pre = mm.get("preimage") or {}
        if pre:
            parts.append(
                "<h3>Preimage diagnostics (amplification = ‖v_preimage‖/‖answer axis‖)</h3>"
            )
            cols = sorted({c for rec in pre.values() if isinstance(rec, dict) for c in rec})
            parts.append(
                _tbl(
                    ["layer", *cols],
                    [
                        [li, *[pre[li].get(c) if isinstance(pre[li], dict) else None for c in cols]]
                        for li in sorted(pre, key=lambda x: int(x) if str(x).isdigit() else 0)
                    ],
                )
            )
    for name, obj in diags.items():
        raw = html.escape(json.dumps(obj, indent=1, sort_keys=True))
        parts.append(
            f"<details><summary>raw {html.escape(name)}</summary><pre>{raw}</pre></details>"
        )
    return "\n".join(parts)


_CSS = """
body{font-family:system-ui,sans-serif;margin:14px;color:#111}
h1{font-size:17px;margin:0 0 4px}
h2{font-size:14px;margin:16px 0 4px}
h3{font-size:12.5px;margin:10px 0 3px}
.setup{font-size:11.5px;color:#444;max-width:1150px;line-height:1.45;margin-bottom:10px}
.controls{display:flex;gap:10px;flex-wrap:wrap;align-items:end;margin:8px 0 12px}
.controls label{font-size:11px;display:block;color:#333}
.controls select{font-size:11.5px;min-width:100px}
button{font-size:12px;padding:3px 10px}
svg{background:#fafafa;border:1px solid #ddd;margin:4px 8px 12px 0}
table{border-collapse:collapse;font-size:11.5px;margin-top:6px}
th,td{border:1px solid #ccc;padding:2px 7px;text-align:right}
th{background:#f0f0f0}
td.l,th.l{text-align:left}
details{margin:4px 0;max-width:1150px}
summary{font-size:12px;cursor:pointer;color:#224}
pre{font-size:10px;max-height:340px;overflow:auto;background:#f7f7f7;padding:6px}
.msg{white-space:pre-wrap;font-size:11.5px;border-left:3px solid #ccc;
     padding:3px 8px;margin:3px 0 3px 12px}
.msg.user{border-color:#c66}
.msg.asst{border-color:#66c}
.count{font-size:12px;color:#333;margin:4px 0}
"""

_JS = r"""
function uniq(vals){return Array.from(new Set(vals)).sort();}
// Every facet is a MULTI-select (all values selected by default). The filter
// applies to plots AND the data table; an empty selection = no filter.
const FACETS = ["scenario","layers","family","op","strength","arm"];
const sels = {};
function initControls(){
  const bar = document.getElementById("controls");
  for(const f of FACETS){
    const wrap = document.createElement("span");
    const lab = document.createElement("label");
    lab.textContent = f + " (multi)";
    const sel = document.createElement("select");
    sel.multiple = true;
    const vals = uniq(DATA.records.map(r => String(r[f])));
    sel.size = Math.min(5, Math.max(2, vals.length));
    // Facet values are registry/enum-derived today, but they are DATA — build
    // options via createElement + textContent, never string-concatenated
    // innerHTML (r3 concern dashboard-facet-innerhtml-sink).
    for(const v of vals){
      const opt = document.createElement("option");
      opt.selected = true;
      opt.textContent = v;
      sel.appendChild(opt);
    }
    sel.onchange = render;
    wrap.appendChild(lab); wrap.appendChild(sel);
    bar.appendChild(wrap);
    sels[f] = sel;
  }
  const dvLab = document.createElement("label");
  dvLab.textContent = "bar DV";
  const dv = document.createElement("select");
  dv.innerHTML = '<option>harm</option><option>coherence</option>' +
    '<option>drift_l32</option><option>drift_band</option>';
  dv.onchange = render;
  const wrap = document.createElement("span");
  wrap.appendChild(dvLab); wrap.appendChild(dv);
  bar.appendChild(wrap);
  sels["__dv__"] = dv;
  const btn = document.createElement("button");
  btn.textContent = "download embedded JSON (records + diagnostics)";
  btn.onclick = () => {
    const blob = new Blob([JSON.stringify(DATA, null, 1)], {type: "application/json"});
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "nap_dashboard_data.json";
    a.click();
  };
  bar.appendChild(btn);
}
function selectedSet(sel){
  return new Set(Array.from(sel.selectedOptions).map(o => o.value));
}
function filtered(){
  return DATA.records.filter(r => FACETS.every(f => {
    const chosen = selectedSet(sels[f]);
    return chosen.size === 0 || chosen.has(String(r[f]));
  }));
}
function mean(xs){
  const v = xs.filter(x => x !== null && x !== undefined && !Number.isNaN(x));
  return v.length ? v.reduce((a,b)=>a+b,0)/v.length : null;
}
const FAM_COLORS = {anchor:"#666", answer:"#1f77b4", ctx_native:"#ff7f0e",
  prefix_native:"#8c564b", ctx_faithful:"#2ca02c", ctx_preimage:"#d62728"};
function svgEl(tag, attrs){
  const e = document.createElementNS("http://www.w3.org/2000/svg", tag);
  for(const k in attrs) e.setAttribute(k, attrs[k]);
  return e;
}
function renderBars(recs, dv){
  const svg = document.getElementById("bars");
  svg.innerHTML = "";
  const items = recs.map(r => ({
    label: r.cell, fam: r.family,
    v: mean(r.turns.map(t => t[dv]))
  })).filter(it => it.v !== null);
  const W = Math.max(700, items.length * 22 + 90), H = 300, padL = 55, padB = 120;
  svg.setAttribute("width", W); svg.setAttribute("height", H + padB);
  if(!items.length){
    const t = svgEl("text", {x: 20, y: 30, "font-size": 12});
    t.textContent = "no " + dv + " values for this filter"; svg.appendChild(t); return;
  }
  const vmax = Math.max(...items.map(i => i.v), 0), vmin = Math.min(...items.map(i => i.v), 0);
  const span = (vmax - vmin) || 1;
  const y = v => 10 + (H - 20) * (1 - (v - vmin) / span);
  for(const gv of [vmin, 0, vmax]){
    const gy = y(gv);
    svg.appendChild(svgEl("line", {x1: padL, x2: W - 5, y1: gy, y2: gy,
      stroke: "#ddd", "stroke-width": 1}));
    const t = svgEl("text", {x: 4, y: gy + 3, "font-size": 9, fill: "#333"});
    t.textContent = gv.toFixed(2); svg.appendChild(t);
  }
  items.forEach((it, i) => {
    const x = padL + i * 22;
    const yv = y(it.v), yb = y(0);
    svg.appendChild(svgEl("rect", {x: x, y: Math.min(yv, yb), width: 16,
      height: Math.abs(yb - yv) || 1, fill: FAM_COLORS[it.fam] || "#999"}));
    const t = svgEl("text", {x: x + 8, y: H + 4, "font-size": 8.5,
      transform: "rotate(90 " + (x + 8) + " " + (H + 4) + ")", fill: "#222"});
    t.textContent = it.label; svg.appendChild(t);
  });
}
function renderLines(recs, field, svgId, label){
  const svg = document.getElementById(svgId);
  svg.innerHTML = "";
  const W = 700, H = 300, padL = 55, padB = 30;
  svg.setAttribute("width", W); svg.setAttribute("height", H + padB);
  const pts = [];
  for(const r of recs)
    for(const t of r.turns)
      if(t[field] !== null && t[field] !== undefined) pts.push(t[field]);
  if(!pts.length){
    const t = svgEl("text", {x: 20, y: 30, "font-size": 12});
    t.textContent = "no " + field + " values for this filter"; svg.appendChild(t); return;
  }
  const vmax = Math.max(...pts), vmin = Math.min(...pts);
  const span = (vmax - vmin) || 1;
  const maxTurn = Math.max(...recs.map(r => Math.max(...r.turns.map(t => t.turn))));
  const x = t => padL + (W - padL - 10) * (t / Math.max(1, maxTurn));
  const y = v => 10 + (H - 20) * (1 - (v - vmin) / span);
  for(const gv of [vmin, vmax]){
    const gy = y(gv);
    svg.appendChild(svgEl("line", {x1: padL, x2: W - 5, y1: gy, y2: gy,
      stroke: "#ddd"}));
    const t = svgEl("text", {x: 4, y: gy + 3, "font-size": 9, fill: "#333"});
    t.textContent = gv.toFixed(2); svg.appendChild(t);
  }
  for(const r of recs){
    const d = r.turns.filter(t => t[field] !== null && t[field] !== undefined)
      .map((t, i) => (i ? "L" : "M") + x(t.turn).toFixed(1) + " " + y(t[field]).toFixed(1))
      .join(" ");
    if(!d) continue;
    svg.appendChild(svgEl("path", {d: d, fill: "none",
      stroke: FAM_COLORS[r.family] || "#999", "stroke-width": 1.2, opacity: 0.8}));
  }
  const t = svgEl("text", {x: W / 2, y: H + 22, "font-size": 10, fill: "#333"});
  t.textContent = label;
  svg.appendChild(t);
}
function fmt(v, d){
  return (v === null || v === undefined) ? "—" : Number(v).toFixed(d);
}
function esc(s){
  return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}
function renderTable(recs){
  const div = document.getElementById("rows");
  const parts = [];
  for(const r of recs){
    parts.push("<details><summary>" + esc(r.cell) + " — " + esc(r.scenario) +
      " (family " + esc(r.family) + ", op " + esc(r.op) + ", strength " +
      esc(r.strength) + ", seed " + r.seed + ")</summary>");
    parts.push("<table><tr><th>turn</th><th>drift_l32</th><th>drift_band</th>" +
      "<th>harm</th><th>coherence</th><th>fired_frac</th><th>|Δproj|</th>" +
      "<th>cap_hit</th></tr>");
    for(const t of r.turns){
      parts.push("<tr><td>" + t.turn + "</td><td>" + fmt(t.drift_l32, 3) + "</td><td>" +
        fmt(t.drift_band, 3) + "</td><td>" +
        fmt(t.harm, 1) + "</td><td>" + fmt(t.coherence, 1) + "</td><td>" +
        fmt(t.fired_frac, 3) + "</td><td>" + fmt(t.abs_dproj, 3) + "</td><td>" +
        (t.cap_hit ? "yes" : "no") + "</td></tr>");
    }
    parts.push("</table>");
    for(const t of r.turns){
      parts.push('<div class="msg user">[turn ' + t.turn + ' user] ' + esc(t.user) + "</div>");
      parts.push('<div class="msg asst">[turn ' + t.turn + ' assistant] ' +
        esc(t.assistant) + "</div>");
    }
    parts.push("</details>");
  }
  div.innerHTML = parts.join("\n");
}
function render(){
  const recs = filtered();
  document.getElementById("count").textContent =
    recs.length + " of " + DATA.records.length + " cells shown";
  renderBars(recs, sels["__dv__"].value);
  renderLines(recs, "drift_l32", "lines_l32",
    "turn (PRIMARY drift = layer-" + DATA.meta.primary_drift_layer +
    " assistant-axis projection of answer tokens)");
  renderLines(recs, "drift_band", "lines_band",
    "turn (secondary drift = band-mean assistant-axis projection of answer tokens)");
  renderTable(recs);
}
initControls();
render();
"""


def render_html(records: list[dict], meta: dict, diags: dict[str, dict]) -> str:
    setup = (
        "<b>Setup / provenance:</b> every arm row is ON-POLICY sampled generation from "
        f"{html.escape(str(meta['model']))} (temperature 0.7, top_p 0.9, max_new_tokens 2048), "
        "ONE fixed-seed trajectory per arm (seed 42); anchor arms (unsteered, cap_alltoken) "
        "additionally at seeds 43/44. <b>Matched target:</b> every drift value is the "
        "projection of answer-token means on the SAME published Lu assistant axis — "
        f"PRIMARY read = layer {PRIMARY_DRIFT_LAYER}, secondary = paper-band mean; every "
        "harm/coherence score comes from the SAME judge rubric "
        "(claude-sonnet-4-5-20250929) across all arms. <b>No display substitution:</b> "
        "assistant text below is the stored raw completion, verbatim. Steering ops: cap "
        "(τ min-floor at context-end), steer (+ασ along the axis), axis_replace; strengths "
        "p50–p100 (cap percentile τ) / k1–k8 (steer α multiples). Round: "
        f"{html.escape(str(meta['round']))}; source tree: {html.escape(str(meta['model_root']))}."
    )
    payload = json.dumps({"records": records, "meta": meta, "diagnostics": diags}).replace(
        "</", "<\\/"
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>#2223 NAP dashboard</title><style>{_CSS}</style></head><body>"
        "<h1>#2223 native-axis-fidelity-preimage — replay dashboard</h1>"
        f"<div class='setup'>{setup}</div>"
        "<div class='controls' id='controls'></div>"
        "<div class='count' id='count'></div>"
        "<svg id='bars'></svg><svg id='lines_l32'></svg><svg id='lines_band'></svg>"
        "<div id='rows'></div>"
        f"{render_diagnostics_html(diags)}"
        f"<script>const DATA = {payload};</script>"
        f"<script>{_JS}</script>"
        "</body></html>"
    )


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--out-root",
        default=str(REPO / "eval_results" / "issue_2223" / "casestudy_replay"),
    )
    ap.add_argument("--model-slug", default="qwen3-32b")
    ap.add_argument("--round-subdir", default=NAP_LABEL)
    ap.add_argument(
        "--extractions-dir",
        default=None,
        help="axis_cos.json / map_metrics.json / tau_map.json dir "
        "(default: <out-root>/<model-slug>/extractions)",
    )
    ap.add_argument("--out", default=None, help="default: <model_root>/nap_dashboard.html")
    args = ap.parse_args(argv)
    model_root = Path(args.out_root) / args.model_slug
    if args.round_subdir:
        model_root = model_root / args.round_subdir
    ext_dir = (
        Path(args.extractions_dir)
        if args.extractions_dir
        else Path(args.out_root) / args.model_slug / "extractions"
    )
    records = build_records(model_root)
    assert records, f"no cells under {model_root} — run the generate phase first"
    diags = load_diagnostics(ext_dir)
    meta = {
        "issue": 2223,
        "round": args.round_subdir or "(root)",
        "model": args.model_slug,
        "model_root": str(model_root),
        "extractions_dir": str(ext_dir),
        "primary_drift_layer": PRIMARY_DRIFT_LAYER,
        "n_records": len(records),
        "diagnostics_present": sorted(diags),
    }
    out = Path(args.out) if args.out else model_root / "nap_dashboard.html"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_html(records, meta, diags))
    print(
        f"[nap-dashboard] wrote {out} ({len(records)} cells, "
        f"{sum(len(r['turns']) for r in records)} turns, "
        f"diagnostics: {sorted(diags) or 'none'})",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
