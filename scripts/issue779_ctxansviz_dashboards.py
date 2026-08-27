"""Issue #779 ctxansviz — HTML dashboards over the pod export.

Builds two self-contained HTML surfaces (no CDN deps) from the export of
``scripts/issue779_ctxansviz_pod.py``:

  1. ``ctxansviz-779-clusters-<tag>.html`` — one row per KMeans context
     cluster (n, corpus shares, map-accuracy means, judged mean, top TF-IDF
     terms) with 5 expandable sample rows per cluster.
  2. ``ctxansviz-779-scatter-<tag>.html`` — hand-rolled canvas scatter of the
     UMAP embedding with nearest-point hover (grid index) and three color
     modes (role / map error / judged score).

Written to BOTH ``dashboard/public/`` and ``experiments/dashboards/``.

Usage:

    uv run python scripts/issue779_ctxansviz_dashboards.py \
        --export-prefix issue779_monitoring/ctxansviz-smoke \
        --local-dir data/issue_779/ctxansviz_dl/smoke --tag smoke

Data loading (download-if-missing, positional row alignment) is shared with
``scripts/issue779_ctxansviz_figures.py``.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import html
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from issue779_ctxansviz_figures import ensure_export, iter_jsonl, load_row_meta  # noqa: E402

from explore_persona_space.orchestrate.provenance import commit_string, git_provenance  # noqa: E402

OUT_DIRS = (Path("dashboard/public"), Path("experiments/dashboards"))
SEED = 42
MAX_SCATTER_ROWS = 30_000
PAYLOAD_CAP_BYTES = 8_000_000
TEXT_CAPS = (280, 200, 160, 120, 80)  # tried in order until the payload fits
ROLE_COLORS = {"cx": "#1f77c4", "vx": "#e69f00", "vhat": "#009e73"}  # match figure roles

_CSS = """
body{font-family:Inter,system-ui,sans-serif;margin:0;background:#faf9f5;color:#1a1a1a}
.wrap{max-width:1300px;margin:0 auto;padding:18px}
h1{font-size:20px;margin:0 0 2px}
.sub{color:#666;font-size:13px;margin-bottom:14px}
table{border-collapse:collapse;width:100%;font-size:13px;background:#fff}
th,td{border:1px solid #e4e2dc;padding:5px 8px;text-align:left;vertical-align:top}
th{background:#f1efe9;position:sticky;top:0}
td.num{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap}
details{margin:2px 0}
summary{cursor:pointer;color:#3a6ea5}
.sample{border-left:3px solid #d8d5cc;margin:6px 0;padding:4px 8px;background:#fbfaf7}
.sample .who{color:#8a6d00;font-size:11px;text-transform:uppercase;letter-spacing:.04em}
.terms{color:#444}
footer{color:#777;font-size:12px;margin-top:16px;line-height:1.5}
button{margin-right:6px;padding:4px 10px;border:1px solid #bbb;background:#fff;cursor:pointer}
button.on{background:#1f77c4;color:#fff;border-color:#1f77c4}
#tip{position:fixed;display:none;max-width:420px;background:#fff;border:1px solid #999;
padding:8px;font-size:12px;pointer-events:none;box-shadow:0 2px 8px rgba(0,0,0,.2);z-index:9}
#legend{font-size:12px;color:#555;margin:6px 0}
"""


def cap_text(s: str, n: int) -> str:
    """Excerpt cap with the inline truncation disclosure (the #1482 convention)."""
    s = s or ""
    return s if len(s) <= n else s[:n] + " …[truncated]"


def _footer(meta: dict, export: Path, extra: str) -> str:
    prov = git_provenance()
    dl = export / "_download_meta.json"
    rev = json.loads(dl.read_text("utf-8"))["revision"] if dl.exists() else "pre-staged"
    return (
        f"Export: superkaiba1/explore-persona-space-data · {export.name} "
        f"(revision {rev}) · producer commit {meta.get('git_commit', '?')[:12]} · "
        f"rendered at commit {commit_string(prov)} · {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())}. "
        f"n_rows={meta['n_rows']:,} (lmsys {meta.get('n_lmsys')}, wildchat {meta.get('n_wildchat')}), "
        f"n_judged={meta.get('n_judged')}, layer L{meta['layer']}. "
        "Answers are on-policy vLLM T=1.0 completions, per the #779 capture; predicted answers "
        "come from the banked n1m ridge (read-only; all rows in-sample, split=train). "
        "Context/answer text capped at 280 chars by the producer with inline ' …[truncated]' "
        f"disclosure. {extra}"
    )


def _page(title: str, subtitle: str, body: str, footer: str, script: str = "") -> str:
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>{html.escape(title)}</title><style>{_CSS}</style></head><body>"
        f"<div class='wrap'><h1>{html.escape(title)}</h1>"
        f"<div class='sub'>{html.escape(subtitle)}</div>{body}"
        f"<footer>{html.escape(footer)}</footer></div>{script}</body></html>"
    )


def build_cluster_browser(d: dict, tag: str) -> str:
    z, rows, meta = d["coords"], d["row_meta"], d["meta"]
    names = [str(x) for x in z["metric_names"]]
    cos = z["metrics"][:, names.index("cos_vhat_vx")]
    labels = z["kmeans_cx"]
    rng = np.random.default_rng(SEED)
    body_rows = []
    for rec in sorted(d["cluster_stats"]["kmeans_cx"], key=lambda r: -r["n"]):
        cid = rec["cluster"]
        idx = np.flatnonzero(labels == cid)
        take = rng.choice(idx, size=min(5, idx.size), replace=False) if idx.size else []
        samples = "".join(
            "<div class='sample'>"
            f"<div class='who'>row {int(i)} · {html.escape(rows[int(i)]['corpus'])} · "
            f"cos(pred,true)={cos[int(i)]:.3f}</div>"
            f"<div><b>context:</b> {html.escape(rows[int(i)]['context_text'])}</div>"
            f"<div><b>answer:</b> {html.escape(rows[int(i)]['answer_text'])}</div></div>"
            for i in take
        )
        mean_dv = rec.get("mean_dv_judged")
        body_rows.append(
            "<tr>"
            f"<td class='num'>{cid}</td><td class='num'>{rec['n']:,}</td>"
            f"<td class='num'>{rec['share_lmsys']:.2f} / {rec['share_wildchat']:.2f}</td>"
            f"<td class='num'>{rec['mean_cos_vhat_vx']:.3f}</td>"
            f"<td class='num'>{rec['median_cos_vhat_vx']:.3f}</td>"
            f"<td class='num'>{rec['mean_cos_ib_vx']:.3f}</td>"
            f"<td class='num'>{'' if mean_dv is None else f'{mean_dv:.1f}'} "
            f"(n={rec.get('n_judged', 0)})</td>"
            f"<td class='terms'>{html.escape(', '.join(rec['top_tfidf_terms']))}</td>"
            f"<td><details><summary>{len(take)} samples</summary>{samples}</details></td>"
            "</tr>"
        )
    sil = d["cluster_stats"].get("silhouette_kmeans_cx")
    table = (
        "<table><thead><tr><th>cluster</th><th>n</th><th>share lmsys / wildchat</th>"
        "<th>mean cos(pred,true)</th><th>median cos(pred,true)</th>"
        "<th>mean cos(identity+bias,true)</th><th>mean judged score</th>"
        "<th>top TF-IDF terms</th><th>samples</th></tr></thead>"
        f"<tbody>{''.join(body_rows)}</tbody></table>"
    )
    return _page(
        f"Context clusters — issue #779 ctxansviz ({tag})",
        f"KMeans over context activations at L{meta['layer']}; "
        f"{len(d['cluster_stats']['kmeans_cx'])} clusters, silhouette={sil}",
        table,
        _footer(meta, d["export"], "TF-IDF terms computed by the producer over capped excerpts."),
    )


def _scatter_payload(d: dict) -> tuple[dict, str]:
    z, rows, meta = d["coords"], d["row_meta"], d["meta"]
    names = [str(x) for x in z["metric_names"]]
    cos = z["metrics"][:, names.index("cos_vhat_vx")]
    n = z["umap_cx"].shape[0]
    judged = d["judged"]
    jm = list(iter_jsonl(d["export"] / "judged_meta.jsonl")) if judged is not None else []
    n_j = 0 if judged is None else judged["umap_ctx"].shape[0]
    bulk_budget = max(MAX_SCATTER_ROWS - n_j, 5_000)
    idx = np.sort(np.random.default_rng(SEED).choice(n, size=min(n, bulk_budget), replace=False))

    def build(cap: int, idx: np.ndarray) -> dict:
        r3 = lambda a: [round(float(v), 3) for v in a]  # noqa: E731
        bulk = [
            [
                *r3(z["umap_cx"][i]),
                *r3(z["umap_vx"][i]),
                *r3(z["umap_vhat"][i]),
                int(z["kmeans_cx"][i]),
                round(float(cos[i]), 3),
                cap_text(rows[i]["context_text"], cap),
                cap_text(rows[i]["answer_text"], cap),
            ]
            for i in idx.tolist()
        ]
        jrows = []
        if judged is not None:
            dv = judged["dv"]
            for k in range(n_j):
                v = float(dv[k])
                jrows.append(
                    [
                        *r3(judged["umap_ctx"][k]),
                        None if np.isnan(v) else round(v, 1),
                        str(jm[k]["context_id"]) if k < len(jm) else "",
                    ]
                )
        return {"bulk": bulk, "judged": jrows, "n_total": n, "text_cap": cap}

    payload, blob = {}, ""
    for cap in TEXT_CAPS:
        payload = build(cap, idx)
        blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
        if len(blob.encode("utf-8")) <= PAYLOAD_CAP_BYTES:
            break
    else:
        while len(blob.encode("utf-8")) > PAYLOAD_CAP_BYTES and idx.size > 2_000:
            idx = idx[::2]
            payload = build(TEXT_CAPS[-1], idx)
            blob = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return payload, blob


_JS = """
<script>
const P = JSON.parse(document.getElementById('payload').textContent);
const cv = document.getElementById('cv'), ctx = cv.getContext('2d');
const tip = document.getElementById('tip');
let mode = 'role';
const W = cv.width, H = cv.height, PAD = 30;
function extent() {
  let xs = [], ys = [];
  for (const r of P.bulk) { xs.push(r[0], r[2], r[4]); ys.push(r[1], r[3], r[5]); }
  for (const j of P.judged) { xs.push(j[0]); ys.push(j[1]); }
  const mnx = Math.min(...xs), mxx = Math.max(...xs), mny = Math.min(...ys), mxy = Math.max(...ys);
  return {mnx, mxx, mny, mxy};
}
const E = extent();
const sx = x => PAD + (x - E.mnx) / (E.mxx - E.mnx + 1e-9) * (W - 2 * PAD);
const sy = y => H - PAD - (y - E.mny) / (E.mxy - E.mny + 1e-9) * (H - 2 * PAD);
function seq(t, name) { // 0..1 -> css color, tiny viridis/plasma approximations
  const stops = name === 'plasma'
    ? [[13,8,135],[126,3,168],[204,71,120],[248,149,64],[240,249,33]]
    : [[68,1,84],[59,82,139],[33,145,140],[94,201,98],[253,231,37]];
  const k = Math.max(0, Math.min(1, t)) * (stops.length - 1);
  const i = Math.min(Math.floor(k), stops.length - 2), f = k - i;
  const c = stops[i].map((v, d) => Math.round(v + f * (stops[i + 1][d] - v)));
  return `rgb(${c[0]},${c[1]},${c[2]})`;
}
let pts = []; // current visible points: [px, py, color, bulkIdx|-1, judgedIdx|-1, role]
function rebuild() {
  pts = [];
  if (mode === 'role') {
    P.bulk.forEach((r, i) => {
      pts.push([sx(r[0]), sy(r[1]), '%CX%', i, -1, 'context']);
      pts.push([sx(r[2]), sy(r[3]), '%VX%', i, -1, 'true answer']);
      pts.push([sx(r[4]), sy(r[5]), '%VHAT%', i, -1, 'predicted answer']);
    });
  } else if (mode === 'err') {
    P.bulk.forEach((r, i) => {
      const t = (r[7] + 1) / 2; // cosine in [-1,1] -> [0,1]
      pts.push([sx(r[0]), sy(r[1]), seq(t, 'viridis'), i, -1, 'context']);
    });
  } else {
    P.bulk.forEach((r, i) => pts.push([sx(r[0]), sy(r[1]), '#cccccc', i, -1, 'context']));
    P.judged.forEach((j, k) => {
      if (j[2] !== null) pts.push([sx(j[0]), sy(j[1]), seq(j[2] / 100, 'plasma'), -1, k, 'judged context']);
    });
  }
  draw(); index();
}
function draw() {
  ctx.clearRect(0, 0, W, H);
  ctx.globalAlpha = 0.6;
  for (const p of pts) { ctx.fillStyle = p[2]; ctx.fillRect(p[0] - 1.2, p[1] - 1.2, 2.4, 2.4); }
  ctx.globalAlpha = 1;
}
const CELL = 24; let grid = new Map();
function index() {
  grid = new Map();
  pts.forEach((p, i) => {
    const key = (p[0] / CELL | 0) + ',' + (p[1] / CELL | 0);
    if (!grid.has(key)) grid.set(key, []);
    grid.get(key).push(i);
  });
}
cv.addEventListener('mousemove', ev => {
  const b = cv.getBoundingClientRect();
  const mx = (ev.clientX - b.left) * (W / b.width), my = (ev.clientY - b.top) * (H / b.height);
  let best = -1, bd = 144;
  const gx = mx / CELL | 0, gy = my / CELL | 0;
  for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) {
    for (const i of (grid.get((gx + dx) + ',' + (gy + dy)) || [])) {
      const d = (pts[i][0] - mx) ** 2 + (pts[i][1] - my) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
  }
  if (best < 0) { tip.style.display = 'none'; return; }
  const p = pts[best];
  let h = `<b>${p[5]}</b>`;
  if (p[3] >= 0) {
    const r = P.bulk[p[3]];
    h += ` · cluster ${r[6]} · cos(pred,true)=${r[7]}<br><b>context:</b> ${esc(r[8])}<br><b>answer:</b> ${esc(r[9])}`;
  } else {
    const j = P.judged[p[4]];
    h += ` · judged score ${j[2]} · context_id ${esc(j[3])} (text not in export)`;
  }
  tip.innerHTML = h;
  tip.style.display = 'block';
  tip.style.left = Math.min(ev.clientX + 14, window.innerWidth - 440) + 'px';
  tip.style.top = (ev.clientY + 14) + 'px';
});
cv.addEventListener('mouseleave', () => tip.style.display = 'none');
function esc(s) { const d = document.createElement('div'); d.textContent = s || ''; return d.innerHTML; }
for (const b of document.querySelectorAll('button[data-mode]')) {
  b.addEventListener('click', () => {
    mode = b.dataset.mode;
    document.querySelectorAll('button[data-mode]').forEach(x => x.classList.toggle('on', x === b));
    rebuild();
  });
}
rebuild();
</script>
"""


def build_scatter(d: dict, tag: str) -> str:
    payload, blob = _scatter_payload(d)
    meta = d["meta"]
    n_bulk, n_j = len(payload["bulk"]), len(payload["judged"])
    safe_blob = blob.replace("</", "<\\/")  # keep '</script>' inside JSON from closing the tag
    body = (
        "<div id='legend'>color modes — role: contexts blue, true answers orange, predicted "
        "answers green; map error: cosine(predicted, true) on viridis; judged: 0–100 judged "
        "sycophancy score on plasma over a gray bulk. Hover a point for its text.</div>"
        "<div><button data-mode='role' class='on'>by role</button>"
        "<button data-mode='err'>by map error</button>"
        "<button data-mode='judged'>by judged score</button></div>"
        "<canvas id='cv' width='1250' height='840' style='width:100%;border:1px solid #ddd;"
        "background:#fff'></canvas><div id='tip'></div>"
        f"<script id='payload' type='application/json'>{safe_blob}</script>"
    )
    js = (
        _JS.replace("%CX%", ROLE_COLORS["cx"])
        .replace("%VX%", ROLE_COLORS["vx"])
        .replace("%VHAT%", ROLE_COLORS["vhat"])
    )
    extra = (
        f"Scatter payload: {n_bulk:,} of {meta['n_rows']:,} rows (seed {SEED}, judged rows all "
        f"kept: {n_j:,}), hover text re-capped at {payload['text_cap']} chars, coords rounded "
        "to 3 decimals."
    )
    return _page(
        f"UMAP hover scatter — issue #779 ctxansviz ({tag})",
        "UMAP of contexts, true answers, predicted answers; judged overlay",
        body,
        _footer(meta, d["export"], extra),
        js,
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__ and __doc__.splitlines()[0])
    ap.add_argument("--export-prefix", required=True)
    ap.add_argument("--local-dir", required=True, type=Path)
    ap.add_argument("--tag", required=True, choices=("smoke", "full"))
    args = ap.parse_args()

    export = ensure_export(args.export_prefix, args.local_dir)
    d = {
        "export": export,
        "coords": np.load(export / "coords.npz", allow_pickle=False),
        "cluster_stats": json.loads((export / "cluster_stats.json").read_text("utf-8")),
        "meta": json.loads((export / "meta.json").read_text("utf-8")),
        "row_meta": load_row_meta(export),
        "judged": np.load(export / "judged.npz", allow_pickle=False)
        if (export / "judged.npz").exists()
        else None,
    }
    n = d["coords"]["umap_cx"].shape[0]
    if len(d["row_meta"]) != n:
        raise RuntimeError(f"row_meta rows {len(d['row_meta'])} != coords rows {n}")

    pages = {
        f"ctxansviz-779-clusters-{args.tag}.html": build_cluster_browser(d, args.tag),
        f"ctxansviz-779-scatter-{args.tag}.html": build_scatter(d, args.tag),
    }
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        for name, content in pages.items():
            (out_dir / name).write_text(content, encoding="utf-8")
            print(
                f"[dashboards] wrote {out_dir / name} ({len(content.encode('utf-8')) / 1e6:.2f} MB)"
            )
    print("[dashboards] done")


if __name__ == "__main__":
    main()
