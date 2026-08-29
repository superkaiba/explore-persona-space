"""Build the separate-UMAP and dimensionality dashboard for issue #779.

The dashboard combines independently fitted context-only and answer-only UMAP
maps with the existing role-specific PCA spectrum, participation ratio, 2NN,
kNN-MLE, correlation-dimension, local-PCA, and CCA diagnostics. Hovering either
map cross-highlights the paired item and draws an explicit context -> answer
correspondence arrow between the independent coordinate systems.

Usage:
    uv run python scripts/issue779_ctxansviz_separate_umap_dashboard.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

import issue779_ctxansviz_separate_pca_dashboard as text_source
from explore_persona_space.orchestrate.provenance import commit_string, git_provenance

CAPTURE_REVISION = "cbc55efdd7f5581677047e487aa61172f6e7944d"
EXPORT_PRODUCER_COMMIT = "79d9142bf5c88ae2ccd3ff7270e9d98a1faaaa5d"
DEFAULT_EXPORT = Path("data/issue_779/ctxansviz_dl/full/issue779_monitoring/ctxansviz")
DEFAULT_UMAP_DIR = Path("data/issue_779/ctxansviz_separate_umap")
DEFAULT_PCA_DIR = Path("data/issue_779/ctxansviz_separate_pca")
UMAP_ARTIFACT = "separate_umap_coords.npz"
UMAP_META = "separate_umap_meta.json"
OUT_NAME = "ctxansviz-779-separate-umap-dimensions.html"
REPORT_NAME = "ctxansviz-779-separate-umap-dimension-analysis.md"
PUBLIC_URL = f"https://eps.superkaiba.com/{OUT_NAME}"
ROLES = {"cx": "context", "vx": "answer"}
ID_METHODS = {
    "twonn": "2NN",
    "lb_mle_k10_mean_of_local_mles": "LB local MLE · k=10",
    "lb_mle_k20_mean_of_local_mles": "LB local MLE · k=20",
    "lb_mle_k10_mackay_ghahramani": "MacKay–Ghahramani · k=10",
    "lb_mle_k20_mackay_ghahramani": "MacKay–Ghahramani · k=20",
    "corr_dim": "Correlation dimension",
    "local_pca_median": "Local PCA · k=100",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_umap(umap_dir: Path) -> tuple[dict[str, np.ndarray], dict]:
    artifact_path = umap_dir / UMAP_ARTIFACT
    meta_path = umap_dir / UMAP_META
    if not artifact_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"separate UMAP artifacts absent from {umap_dir}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    checks = {
        "capture_revision": CAPTURE_REVISION,
        "git_dirty": False,
        "git_argv0_state": "tracked",
        "n_fit_rows_per_role": 100_000,
        "fit_rows_identical_across_roles": True,
    }
    for key, expected in checks.items():
        if meta.get(key) != expected:
            raise RuntimeError(f"separate UMAP manifest {key} is not pinned: {meta.get(key)!r}")
    if sha256_file(artifact_path) != meta.get("artifact_sha256"):
        raise RuntimeError("separate UMAP artifact SHA-256 mismatch")
    raw = np.load(artifact_path)
    arrays = {key: np.asarray(raw[key]) for key in raw.files}
    expected_shapes = {
        "fit_ci": (100_000,),
        "context_fit_umap": (100_000, 2),
        "answer_fit_umap": (100_000, 2),
        "display_ci": (5_500,),
        "context_display_umap": (5_500, 2),
        "answer_display_umap": (5_500, 2),
    }
    for key, shape in expected_shapes.items():
        if arrays[key].shape != shape:
            raise RuntimeError(f"{key} shape {arrays[key].shape} != {shape}")
        if key != "fit_ci" and key != "display_ci" and not np.isfinite(arrays[key]).all():
            raise RuntimeError(f"{key} contains non-finite values")
    if len(set(arrays["fit_ci"].tolist())) != 100_000:
        raise RuntimeError("separate UMAP fit row IDs are not unique")
    if len(set(arrays["display_ci"].tolist())) != 5_500:
        raise RuntimeError("separate UMAP display row IDs are not unique")
    return arrays, meta


def load_dimensions(export_dir: Path) -> tuple[dict, dict[str, np.ndarray], dict]:
    producer_path = export_dir / "meta.json"
    summary_path = export_dir / "dim_summary.json"
    spectra_path = export_dir / "dim_spectra.npz"
    estimates_path = export_dir / "dim_id_estimates.jsonl"
    for path in (producer_path, summary_path, spectra_path, estimates_path):
        if not path.exists():
            raise FileNotFoundError(f"required dimensionality artifact absent: {path}")
    producer = json.loads(producer_path.read_text(encoding="utf-8"))
    if producer.get("git_commit") != EXPORT_PRODUCER_COMMIT:
        raise RuntimeError("dimensionality export producer commit is not pinned")
    hashes = producer.get("export_files_sha256", {})
    for path in (summary_path, spectra_path, estimates_path):
        if hashes.get(path.name) != sha256_file(path):
            raise RuntimeError(f"dimensionality artifact SHA-256 mismatch: {path.name}")
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    raw = np.load(spectra_path)
    spectra = {key: np.asarray(raw[key], dtype=np.float64) for key in raw.files}
    expected_shapes = {
        "evals_cx": (3_584,),
        "evals_vx": (3_584,),
        "cca_corrs_cx_vx": (500,),
    }
    for key, shape in expected_shapes.items():
        if spectra[key].shape != shape or not np.isfinite(spectra[key]).all():
            raise RuntimeError(f"malformed dimensionality spectrum: {key}")
    return summary, spectra, producer


def attach_umap(rows: list[dict], arrays: dict[str, np.ndarray]) -> None:
    cis = arrays["display_ci"].astype(np.int64)
    lookup = {int(ci): index for index, ci in enumerate(cis)}
    for row in rows:
        index = lookup.get(int(row["ci"]))
        if index is None:
            raise RuntimeError(f"public display ci={row['ci']} absent from UMAP coordinates")
        row["cu"] = [round(float(value), 5) for value in arrays["context_display_umap"][index]]
        row["au"] = [round(float(value), 5) for value in arrays["answer_display_umap"][index]]
        row["context_chars"] = len(row["context"])
        row["answer_chars"] = len(row["answer"])


def id_curves(summary: dict) -> dict:
    curves: dict[str, dict] = {}
    estimates = summary["id_estimates"]
    for source_role, public_role in ROLES.items():
        by_n = estimates[source_role]
        ns = sorted(int(value) for value in by_n)
        curves[public_role] = {
            "n": ns,
            "methods": {
                method: {
                    "label": label,
                    "median": [round(float(by_n[str(n)][method]["median"]), 6) for n in ns],
                    "lo": [round(float(by_n[str(n)][method]["p2_5"]), 6) for n in ns],
                    "hi": [round(float(by_n[str(n)][method]["p97_5"]), 6) for n in ns],
                    "resamples": [int(by_n[str(n)][method]["n_resamples"]) for n in ns],
                }
                for method, label in ID_METHODS.items()
            },
        }
    return curves


def spectrum_payload(spectra: dict[str, np.ndarray]) -> dict:
    output = {}
    for source_role, public_role in ROLES.items():
        eigenvalues = spectra[f"evals_{source_role}"]
        explained = eigenvalues / eigenvalues.sum()
        output[public_role] = {
            "evr": [round(float(value), 10) for value in explained],
            "cumulative": [round(float(value), 8) for value in np.cumsum(explained)],
        }
    return output


def tool_inventory() -> list[dict]:
    return [
        {
            "tool": "Separate PCA + joint PCA",
            "status": "run",
            "answers": "Global linear directions and variance concentration; separate fits avoid assuming shared axes.",
        },
        {
            "tool": "Separate UMAP + joint UMAP",
            "status": "run",
            "answers": "Local nonlinear layout. Separate maps are now independent; UMAP is not an intrinsic-dimension estimator.",
        },
        {
            "tool": "2NN intrinsic dimension",
            "status": "run",
            "answers": "Very local distance-ratio dimension with few tuning choices.",
        },
        {
            "tool": "Participation ratio",
            "status": "run",
            "answers": "Global linear effective rank of the covariance spectrum.",
        },
        {
            "tool": "Levina–Bickel / MacKay–Ghahramani MLE",
            "status": "run",
            "answers": "k-neighborhood intrinsic dimension at two local scales.",
        },
        {
            "tool": "Correlation dimension + local PCA",
            "status": "run",
            "answers": "Distance-scaling and local tangent-rank views; useful disagreement with 2NN is expected.",
        },
        {
            "tool": "CCA, CKA, clustering",
            "status": "run",
            "answers": "Paired linear relation, representation similarity, and coarse group structure.",
        },
        {
            "tool": "PaCMAP or TriMap",
            "status": "not run",
            "answers": "Alternative 2-D layouts that retain more mid-range or global relationships than UMAP.",
        },
        {
            "tool": "t-SNE",
            "status": "not run",
            "answers": "Strong local-cluster view, but global spacing and cross-map comparison remain unreliable.",
        },
        {
            "tool": "PHATE / diffusion maps",
            "status": "not run",
            "answers": "Potential trajectories or continuous geometry if prompt/answer variation follows branches.",
        },
        {
            "tool": "Isomap",
            "status": "not run",
            "answers": "Geodesic global geometry; expensive and sensitive to graph shortcuts at this scale.",
        },
        {
            "tool": "Mapper / persistent homology",
            "status": "not run",
            "answers": "Topology—branches, loops, and connected components—rather than another scatter plot.",
        },
        {
            "tool": "kNN graph + ForceAtlas2",
            "status": "not run",
            "answers": "Graph communities and bridges, with explicit edges and neighborhood flow.",
        },
    ]


def build_data(
    rows: list[dict],
    sample_meta: dict,
    umap_meta: dict,
    dim_summary: dict,
    spectra: dict[str, np.ndarray],
    producer_meta: dict,
) -> dict:
    spectral_summary = dim_summary["spectra"]["spectra"]
    quality = umap_meta["quality"]
    disparity = float(umap_meta["procrustes_descriptive_only"]["disparity"])
    points = [
        {
            "ci": int(row["ci"]),
            "corpus": row["corpus"],
            "cu": row["cu"],
            "au": row["au"],
            "context": row["context"],
            "answer": row["answer"],
            "context_chars": row["context_chars"],
            "answer_chars": row["answer_chars"],
        }
        for row in rows
    ]
    return {
        "meta": {
            **sample_meta,
            "layer": 19,
            "n_umap_fit_per_role": int(umap_meta["n_fit_rows_per_role"]),
            "umap_producer_commit": umap_meta["git_commit"],
            "umap_artifact_sha256": umap_meta["artifact_sha256"],
            "dim_producer_commit": producer_meta["git_commit"],
            "render_commit": commit_string(git_provenance()),
            "generated_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime()),
            "public_url": PUBLIC_URL,
            "preprocessing": umap_meta["preprocessing"],
            "umap_params": umap_meta["umap_params"],
        },
        "points": points,
        "quality": {
            **quality,
            "procrustes_disparity": round(disparity, 6),
            "procrustes_shape_similarity": round(1 - disparity / 2, 6),
        },
        "linear": {
            "summary": {
                public_role: {
                    key: round(float(value), 6) if isinstance(value, float) else value
                    for key, value in spectral_summary[source_role].items()
                }
                for source_role, public_role in ROLES.items()
            },
            "spectra": spectrum_payload(spectra),
            "cca": [round(float(value), 7) for value in spectra["cca_corrs_cx_vx"]],
        },
        "nonlinear": id_curves(dim_summary),
        "tools": tool_inventory(),
    }


CSS = r"""
:root{--paper:#f3f0e7;--surface:#fffdf7;--ink:#20221f;--muted:#686961;--line:#c9c5b9;--context:#285d7d;--answer:#b96e16;--select:#b73529;--lmsys:#7f7665;--wild:#4e7a62}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--paper);color:var(--ink);font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono",monospace}.wrap{max-width:1760px;margin:auto;padding:18px 22px 40px}
header{display:grid;grid-template-columns:minmax(0,1fr) auto;gap:28px;border-bottom:1px solid var(--line);padding-bottom:14px}header>*,.section-head>*,.header-stats>div{min-width:0}h1{font-size:21px;line-height:1.15;letter-spacing:-.035em;margin:0 0 7px;overflow-wrap:anywhere}.lede{font-size:11px;line-height:1.55;color:var(--muted);max-width:1120px;overflow-wrap:anywhere}.header-stats{display:grid;grid-template-columns:repeat(3,auto);gap:18px;align-self:start}.header-stats b{display:block;font-size:18px;letter-spacing:-.04em}.header-stats span{font-size:9px;color:var(--muted);overflow-wrap:anywhere}
.jump{display:flex;gap:18px;border-bottom:1px solid var(--line);padding:10px 0;font-size:10px}.jump a{color:var(--muted);text-decoration:none;border-bottom:1px solid transparent}.jump a:hover{color:var(--ink);border-bottom-color:var(--ink)}
.section{border-bottom:1px solid var(--line);padding:18px 0}.section-head{display:grid;grid-template-columns:minmax(0,1fr) minmax(280px,520px);gap:28px;margin-bottom:12px}.section h2{font-size:14px;margin:0 0 5px;letter-spacing:-.02em}.section-head p,.note{font-size:10px;line-height:1.5;color:var(--muted);margin:0}.callout{border-left:3px solid var(--select);padding-left:11px;color:var(--ink)!important}
.map-shell{position:relative}.map-toolbar{display:flex;align-items:center;gap:9px;padding:8px 10px;border:1px solid var(--line);background:var(--surface);font-size:10px}.map-toolbar label{color:var(--muted)}button,select,input{font:inherit;font-size:10px;color:var(--ink);background:transparent;border:1px solid var(--line);padding:5px 7px}button{cursor:pointer}button:hover{border-color:var(--ink)}input{width:125px}.map-key{margin-left:auto;display:flex;gap:13px;color:var(--muted)}.dot{display:inline-block;width:8px;height:8px;margin-right:5px}.dot.w{background:var(--wild)}.dot.l{background:var(--lmsys)}
.maps-grid{position:relative;display:grid;grid-template-columns:1fr 1fr;background:var(--surface);border:1px solid var(--line);border-top:0}.map-panel{min-width:0;position:relative;padding:10px}.map-panel+.map-panel{border-left:1px solid var(--line)}.map-title{display:flex;justify-content:space-between;align-items:baseline;font-size:11px;margin-bottom:6px}.map-title span{font-size:9px;color:var(--muted)}.map-canvas{width:100%;height:470px;display:block;cursor:crosshair}.pair-overlay{position:absolute;inset:0;width:100%;height:100%;z-index:4;pointer-events:none;overflow:visible}.arrow-label{font-size:9px;fill:var(--select)}
.pair-detail{display:grid;grid-template-columns:150px 1fr 1fr;background:var(--surface);border:1px solid var(--line);border-top:0}.pair-detail>div{padding:12px;min-width:0}.pair-detail>div+div{border-left:1px solid var(--line)}.pair-meta{font-size:10px;line-height:1.6;color:var(--muted)}.pair-meta b{color:var(--ink);font-size:12px}.full-text{margin:0;font:inherit;font-size:10px;line-height:1.5;white-space:pre-wrap;overflow-wrap:anywhere;max-height:320px;overflow:auto}.role-label{font-size:9px;text-transform:uppercase;color:var(--muted);display:block;margin-bottom:7px}
.metrics{display:grid;grid-template-columns:repeat(5,1fr);border:1px solid var(--line);border-top:0;background:var(--surface)}.metric{padding:12px;min-width:0}.metric+.metric{border-left:1px solid var(--line)}.metric b{display:block;font-size:18px;letter-spacing:-.04em}.metric span{display:block;font-size:9px;line-height:1.4;color:var(--muted);margin-top:3px}
.analysis-grid{display:grid;grid-template-columns:1.25fr .75fr;border:1px solid var(--line);background:var(--surface)}.analysis-grid>*{min-width:0}.chart-block{padding:13px}.chart-block+.chart-block{border-left:1px solid var(--line)}.chart-head{display:flex;justify-content:space-between;align-items:start;gap:14px;margin-bottom:6px}.chart-head h3{font-size:11px;margin:0}.chart-head p{font-size:9px;line-height:1.4;color:var(--muted);margin:3px 0 0}.seg{display:flex}.seg button{border-right:0}.seg button:last-child{border-right:1px solid var(--line)}.seg button.active{background:var(--ink);color:var(--surface);border-color:var(--ink)}.chart{display:block;width:100%;height:280px}.chart.tall{height:330px}
.linear-summary{display:grid;grid-template-columns:1fr 1fr;border-top:1px solid var(--line);margin-top:9px}.linear-summary>div{padding:10px 0}.linear-summary>div+div{border-left:1px solid var(--line);padding-left:12px}.linear-summary b{font-size:12px}.linear-summary p{font-size:9px;line-height:1.5;color:var(--muted);margin:4px 0 0}
.id-layout{display:grid;grid-template-columns:1.1fr .9fr;border:1px solid var(--line);border-top:0;background:var(--surface)}.id-layout>*{min-width:0}.id-table-wrap{padding:13px;border-left:1px solid var(--line);overflow:auto}.id-table{width:100%;border-collapse:collapse;font-size:9px}.id-table th,.id-table td{padding:7px;border-bottom:1px solid #ddd9ce;text-align:right;font-weight:400}.id-table th:first-child,.id-table td:first-child{text-align:left}.c{color:var(--context)}.a{color:var(--answer)}
.findings{display:grid;grid-template-columns:repeat(3,1fr);border:1px solid var(--line);border-top:0;background:var(--surface)}.finding{padding:13px;font-size:10px;line-height:1.52}.finding+.finding{border-left:1px solid var(--line)}.finding b{display:block;font-size:11px;margin-bottom:5px}.finding p{margin:0;color:var(--muted)}
.tool-table-wrap{border:1px solid var(--line);background:var(--surface);overflow:auto}.tool-table{width:100%;border-collapse:collapse;font-size:10px}.tool-table th,.tool-table td{padding:9px 10px;border-bottom:1px solid #ddd9ce;text-align:left;vertical-align:top}.tool-table th{font-size:9px;color:var(--muted);font-weight:400}.tool-table td:nth-child(2){width:90px}.status-run{color:var(--wild);font-weight:700}.status-not{color:var(--muted)}
.foot{font-size:9px;line-height:1.55;color:var(--muted);margin-top:14px;border-left:3px solid #827b68;padding-left:9px;overflow-wrap:anywhere}.foot a{color:inherit}
@media(max-width:1050px){header,.section-head{grid-template-columns:1fr}.maps-grid,.analysis-grid,.id-layout{grid-template-columns:1fr}.map-panel+.map-panel,.chart-block+.chart-block,.id-table-wrap{border-left:0;border-top:1px solid var(--line)}.metrics{grid-template-columns:repeat(2,1fr)}.metric{border-bottom:1px solid var(--line)}.findings{grid-template-columns:1fr}.finding+.finding{border-left:0;border-top:1px solid var(--line)}}
@media(max-width:760px){.wrap{padding:13px 11px 28px;max-width:100%}.header-stats{grid-template-columns:repeat(3,minmax(0,1fr));gap:10px}.jump{overflow:auto}.map-toolbar{flex-wrap:wrap;max-width:100%}.map-toolbar input{max-width:110px}.map-key{width:100%;margin-left:0}.maps-grid,.map-panel,.map-canvas{max-width:100%}.map-canvas{height:390px}.pair-detail{grid-template-columns:1fr}.pair-detail>div+div{border-left:0;border-top:1px solid var(--line)}.metrics{grid-template-columns:1fr}.metric+.metric{border-left:0}.chart{height:240px}}
"""


JS = r"""
const D=JSON.parse(document.getElementById('payload').textContent),$=id=>document.getElementById(id),COL={context:'#285d7d',answer:'#b96e16',select:'#b73529',wild:'#4e7a62',lmsys:'#7f7665',line:'#c9c5b9',ink:'#20221f',muted:'#686961',surface:'#fffdf7'};
const state={hover:null,pinned:D.points.findIndex(p=>p.corpus==='wildchat'),color:'corpus',map:{context:{scale:1,ox:0,oy:0,screen:[]},answer:{scale:1,ox:0,oy:0,screen:[]}},spectrum:'cumulative',idMethod:'twonn'},BOUNDS={},LENMAX={};
const active=()=>state.pinned??state.hover,fmt=(v,n=2)=>Number(v).toFixed(n),pct=(v,n=1)=>`${fmt(v*100,n)}%`;
for(const role of ['context','answer']){const key=role==='context'?'cu':'au',xs=D.points.map(p=>p[key][0]),ys=D.points.map(p=>p[key][1]);BOUNDS[role]={xmin:Math.min(...xs),xmax:Math.max(...xs),ymin:Math.min(...ys),ymax:Math.max(...ys)};LENMAX[role]=Math.max(...D.points.map(p=>role==='context'?p.context_chars:p.answer_chars))}
function pointColor(p,role){if(state.color==='corpus')return p.corpus==='wildchat'?COL.wild:COL.lmsys;const value=role==='context'?p.context_chars:p.answer_chars,t=Math.min(1,Math.log1p(value)/Math.log1p(LENMAX[role]));const base=role==='context'?[40,93,125]:[185,110,22],fade=[211,207,195];return `rgb(${fade.map((x,i)=>Math.round(x+(base[i]-x)*t)).join(',')})`}
function setupCanvas(canvas){const rect=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio||1,2);canvas.width=Math.round(rect.width*dpr);canvas.height=Math.round(rect.height*dpr);return {ctx:canvas.getContext('2d'),w:rect.width,h:rect.height,dpr}}
function mapPosition(p,role,w,h){const b=BOUNDS[role],v=p[role==='context'?'cu':'au'],pad=20,s=state.map[role],bx=pad+(v[0]-b.xmin)/(b.xmax-b.xmin)*(w-2*pad),by=h-pad-(v[1]-b.ymin)/(b.ymax-b.ymin)*(h-2*pad);return [(bx-w/2)*s.scale+w/2+s.ox,(by-h/2)*s.scale+h/2+s.oy]}
function drawMap(role){const cv=$(role+'-map'),{ctx,w,h,dpr}=setupCanvas(cv),m=state.map[role],selected=active();ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle=COL.surface;ctx.fillRect(0,0,w,h);ctx.strokeStyle='#e1ddd2';ctx.setLineDash([2,4]);for(let i=1;i<4;i++){ctx.beginPath();ctx.moveTo(w*i/4,0);ctx.lineTo(w*i/4,h);ctx.stroke();ctx.beginPath();ctx.moveTo(0,h*i/4);ctx.lineTo(w,h*i/4);ctx.stroke()}ctx.setLineDash([]);m.screen=[];D.points.forEach((p,i)=>{const [x,y]=mapPosition(p,role,w,h);m.screen.push([x,y]);if(x<-5||y<-5||x>w+5||y>h+5)return;ctx.globalAlpha=.68;ctx.fillStyle=pointColor(p,role);ctx.beginPath();ctx.arc(x,y,2.25,0,Math.PI*2);ctx.fill();if(i===selected){ctx.globalAlpha=1;ctx.strokeStyle=COL.select;ctx.lineWidth=2;ctx.beginPath();ctx.arc(x,y,7,0,Math.PI*2);ctx.stroke();ctx.fillStyle=COL.select;ctx.beginPath();ctx.arc(x,y,3,0,Math.PI*2);ctx.fill()}});ctx.globalAlpha=1;ctx.fillStyle=COL.muted;ctx.font='9px ui-monospace,monospace';ctx.fillText(`${D.points.length.toLocaleString()} display pairs · wheel zoom · drag pan`,9,h-9);requestAnimationFrame(updateArrow)}
function drawMaps(){drawMap('context');drawMap('answer');renderDetail()}
function nearest(role,x,y){let best=null,dist=Infinity;state.map[role].screen.forEach((q,i)=>{const d=(q[0]-x)**2+(q[1]-y)**2;if(d<dist){dist=d;best=i}});return dist<=144?best:null}
function eventXY(ev,cv){const r=cv.getBoundingClientRect();return [ev.clientX-r.left,ev.clientY-r.top]}
function bindMap(role){const cv=$(role+'-map');let drag=null,moved=false;cv.onpointerdown=ev=>{const [x,y]=eventXY(ev,cv);drag={x,y,ox:state.map[role].ox,oy:state.map[role].oy};moved=false;cv.setPointerCapture(ev.pointerId)};cv.onpointermove=ev=>{const [x,y]=eventXY(ev,cv);if(drag){if(Math.hypot(x-drag.x,y-drag.y)>3)moved=true;state.map[role].ox=drag.ox+x-drag.x;state.map[role].oy=drag.oy+y-drag.y;drawMap(role);return}const hit=nearest(role,x,y);if(hit!==state.hover){state.hover=hit;drawMaps()}};cv.onpointerup=ev=>{if(drag&&!moved){const [x,y]=eventXY(ev,cv),hit=nearest(role,x,y);state.pinned=hit===state.pinned?null:hit}drag=null;drawMaps()};cv.onpointerleave=()=>{if(!drag&&state.hover!==null){state.hover=null;drawMaps()}};cv.onwheel=ev=>{ev.preventDefault();const [x,y]=eventXY(ev,cv),s=state.map[role],old=s.scale,next=Math.max(.55,Math.min(12,old*Math.exp(-ev.deltaY*.001)));s.ox=x-(x-s.ox)*next/old;s.oy=y-(y-s.oy)*next/old;s.scale=next;drawMap(role)},{passive:false}}
function updateArrow(){const svg=$('pair-overlay'),grid=$('maps-grid'),i=active();if(i===null){svg.innerHTML='';return}const gr=grid.getBoundingClientRect(),cr=$('context-map').getBoundingClientRect(),ar=$('answer-map').getBoundingClientRect(),c=state.map.context.screen[i],a=state.map.answer.screen[i];if(!c||!a)return;const x1=cr.left-gr.left+c[0],y1=cr.top-gr.top+c[1],x2=ar.left-gr.left+a[0],y2=ar.top-gr.top+a[1],w=gr.width,h=gr.height,mx=(x1+x2)/2,my=(y1+y2)/2;svg.setAttribute('viewBox',`0 0 ${w} ${h}`);svg.innerHTML=`<defs><marker id="arrowhead" markerWidth="9" markerHeight="7" refX="8" refY="3.5" orient="auto"><polygon points="0 0,9 3.5,0 7" fill="${COL.select}"/></marker></defs><path d="M ${x1} ${y1} C ${mx-45} ${y1},${mx+45} ${y2},${x2} ${y2}" fill="none" stroke="${COL.select}" stroke-width="1.8" marker-end="url(#arrowhead)"/><rect x="${mx-54}" y="${my-10}" width="108" height="18" fill="${COL.surface}" stroke="${COL.select}"/><text class="arrow-label" x="${mx}" y="${my+3}" text-anchor="middle">context → answer</text>`}
function renderDetail(){const i=active(),root=$('pair-empty');if(i===null){root.textContent='Hover either map to inspect a complete pair and reveal its correspondence arrow.';$('pair-ci').textContent='No pair selected';$('detail-context').textContent='';$('detail-answer').textContent='';return}const p=D.points[i];$('pair-ci').textContent=`ci ${p.ci}`;$('pair-empty').textContent=`${p.corpus} · context ${p.context_chars.toLocaleString()} chars · answer ${p.answer_chars.toLocaleString()} chars${state.pinned===i?' · pinned':''}`;$('detail-context').textContent=p.context;$('detail-answer').textContent=p.answer}
function resetMaps(){for(const role of ['context','answer'])state.map[role]={scale:1,ox:0,oy:0,screen:[]};state.pinned=null;state.hover=null;drawMaps()}
function searchCI(){const ci=Number($('ci-search').value),i=D.points.findIndex(p=>p.ci===ci);if(i<0){$('ci-search').setCustomValidity('CI is not in the public display sample');$('ci-search').reportValidity();return}$('ci-search').setCustomValidity('');state.pinned=i;drawMaps()}
function line(ctx,points,color,width=2){ctx.strokeStyle=color;ctx.lineWidth=width;ctx.beginPath();points.forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.stroke()}
function chartBase(id){const cv=$(id),{ctx,w,h,dpr}=setupCanvas(cv);ctx.setTransform(dpr,0,0,dpr,0,0);ctx.clearRect(0,0,w,h);ctx.fillStyle=COL.surface;ctx.fillRect(0,0,w,h);return {ctx,w,h,L:52,R:15,T:14,B:34}}
function axes(o,xTicks,yTicks,xMap,yMap){const {ctx,w,h,L,R,T,B}=o;ctx.strokeStyle=COL.line;ctx.lineWidth=1;ctx.fillStyle=COL.muted;ctx.font='9px ui-monospace,monospace';ctx.textAlign='right';yTicks.forEach(v=>{const y=yMap(v);ctx.strokeStyle='#ddd9ce';ctx.beginPath();ctx.moveTo(L,y);ctx.lineTo(w-R,y);ctx.stroke();ctx.fillText(String(v),L-7,y+3)});ctx.textAlign='center';xTicks.forEach(([v,label])=>{const x=xMap(v);ctx.fillText(label,x,h-10)});ctx.strokeStyle=COL.ink;ctx.beginPath();ctx.moveTo(L,T);ctx.lineTo(L,h-B);ctx.lineTo(w-R,h-B);ctx.stroke()}
function drawSpectrum(){const o=chartBase('spectrum-chart'),{ctx,w,h,L,R,T,B}=o,n=3584,x=r=>L+Math.log10(r)/Math.log10(n)*(w-L-R),mode=state.spectrum;let ymin=0,ymax=1,y;if(mode==='scree'){const vals=['context','answer'].flatMap(role=>D.linear.spectra[role].evr),logs=vals.map(v=>Math.log10(Math.max(v,1e-12)));ymin=Math.floor(Math.min(...logs));ymax=Math.ceil(Math.max(...logs));y=v=>h-B-(Math.log10(Math.max(v,1e-12))-ymin)/(ymax-ymin)*(h-T-B)}else y=v=>h-B-v*(h-T-B);const yTicks=mode==='scree'?[ymin,-4,-3,-2,-1].filter((v,i,a)=>v>=ymin&&v<=ymax&&a.indexOf(v)===i):[0,.25,.5,.75,1];axes(o,[[1,'1'],[10,'10'],[100,'100'],[1000,'1k'],[3584,'3,584']],yTicks,x,mode==='scree'?v=>h-B-(v-ymin)/(ymax-ymin)*(h-T-B):y);for(const role of ['context','answer']){const vals=D.linear.spectra[role][mode==='scree'?'evr':'cumulative'],points=vals.map((v,i)=>[x(i+1),y(v)]);line(ctx,points,COL[role],2)}ctx.textAlign='left';ctx.fillStyle=COL.context;ctx.fillText('context',L+8,T+10);ctx.fillStyle=COL.answer;ctx.fillText('answer',L+80,T+10)}
function drawCCA(){const o=chartBase('cca-chart'),{ctx,w,h,L,R,T,B}=o,x=i=>L+i/499*(w-L-R),y=v=>h-B-v*(h-T-B);axes(o,[[0,'1'],[99,'100'],[249,'250'],[499,'500']],[0,.25,.5,.75,1],x,y);line(ctx,D.linear.cca.map((v,i)=>[x(i),y(v)]),'#6d557d',2);ctx.fillStyle='#6d557d';ctx.textAlign='left';ctx.fillText('canonical correlation',L+8,T+10)}
function drawID(){const method=state.idMethod,o=chartBase('id-chart'),{ctx,w,h,L,R,T,B}=o,roles=['context','answer'],all=roles.flatMap(role=>D.nonlinear[role].methods[method].hi),ymax=Math.ceil(Math.max(...all)/5)*5,x=i=>L+i/2*(w-L-R),y=v=>h-B-v/ymax*(h-T-B);axes(o,[[0,'5k'],[1,'20k'],[2,'50k']],Array.from({length:ymax/5+1},(_,i)=>i*5),x,y);roles.forEach(role=>{const q=D.nonlinear[role].methods[method],upper=q.hi.map((v,i)=>[x(i),y(v)]),lower=q.lo.map((v,i)=>[x(i),y(v)]).reverse();ctx.globalAlpha=.13;ctx.fillStyle=COL[role];ctx.beginPath();[...upper,...lower].forEach((p,i)=>i?ctx.lineTo(...p):ctx.moveTo(...p));ctx.closePath();ctx.fill();ctx.globalAlpha=1;line(ctx,q.median.map((v,i)=>[x(i),y(v)]),COL[role],2);q.median.forEach((v,i)=>{ctx.fillStyle=COL[role];ctx.beginPath();ctx.arc(x(i),y(v),3,0,Math.PI*2);ctx.fill()})})}
function renderIDTable(){const c=D.nonlinear.context.methods,a=D.nonlinear.answer.methods;$('id-table-body').innerHTML=Object.keys(c).map(key=>`<tr><td>${c[key].label}</td><td class="c">${fmt(c[key].median[2],2)}</td><td class="a">${fmt(a[key].median[2],2)}</td><td>${fmt(a[key].median[2]-c[key].median[2],2)}</td></tr>`).join('')}
function renderTools(){$('tool-body').innerHTML=D.tools.map(t=>`<tr><td>${t.tool}</td><td class="${t.status==='run'?'status-run':'status-not'}">${t.status}</td><td>${t.answers}</td></tr>`).join('')}
function init(){for(const role of ['context','answer'])bindMap(role);$('color-mode').onchange=ev=>{state.color=ev.target.value;drawMaps()};$('reset-maps').onclick=resetMaps;$('ci-go').onclick=searchCI;$('ci-search').onkeydown=ev=>{if(ev.key==='Enter')searchCI()};document.querySelectorAll('[data-spectrum]').forEach(b=>b.onclick=()=>{state.spectrum=b.dataset.spectrum;document.querySelectorAll('[data-spectrum]').forEach(q=>q.classList.toggle('active',q===b));drawSpectrum()});$('id-method').onchange=ev=>{state.idMethod=ev.target.value;drawID()};renderIDTable();renderTools();drawMaps();drawSpectrum();drawCCA();drawID()}addEventListener('resize',()=>{drawMaps();drawSpectrum();drawCCA();drawID()});init();
"""


def page(data: dict) -> str:
    payload = json.dumps(data, ensure_ascii=False, separators=(",", ":")).replace("<", "\\u003c")
    json.loads(payload)
    meta = data["meta"]
    quality = data["quality"]
    linear = data["linear"]["summary"]
    methods = data["nonlinear"]["context"]["methods"]
    options = "".join(
        f'<option value="{key}">{value["label"]}</option>' for key, value in methods.items()
    )
    filters = ", ".join(f"{key}: {value}" for key, value in meta["public_filter_counts"].items())
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Separate UMAP + dimensionality | issue #779</title><style>{CSS}</style></head><body><div class="wrap">
<header><div><h1>Context and answer geometry · separate UMAPs + dimensionality</h1><div class="lede">Two independent nonlinear maps at layer {meta["layer"]}, fit on the same {meta["n_umap_fit_per_role"]:,} paired row IDs. Context vectors define the left graph; answer vectors define the right graph. Hover either side to cross-highlight the pair and draw the explicit context → answer correspondence. Below: global linear spectrum, local nonlinear dimension, and paired-space diagnostics.</div></div><div class="header-stats"><div><b>{linear["context"]["participation_ratio"]:.1f}</b><span>context participation ratio</span></div><div><b>{linear["answer"]["participation_ratio"]:.1f}</b><span>answer participation ratio</span></div><div><b>{quality["context_answer_neighbor_overlap_k15"]["native_pca100"] * 100:.1f}%</b><span>paired native k15 overlap</span></div></div></header>
<nav class="jump"><a href="#maps">separate maps</a><a href="#linear">linear dimension</a><a href="#nonlinear">nonlinear dimension</a><a href="#tools">other tools</a></nav>
<section class="section" id="maps"><div class="section-head"><div><h2>Independent UMAP coordinate systems</h2><p>Both use the pinned joint PCA-100 transform only as denoising/preprocessing. Each role then gets its own cosine-neighbor graph and 2-D UMAP fit. Axes, rotation, scale, and distances are not shared across panels.</p></div><p class="callout">The red arrow is a pair identifier—not a displacement vector in one space. Its endpoints live in different coordinate systems. Hover, click to pin, wheel to zoom, and drag to pan.</p></div>
<div class="map-shell"><div class="map-toolbar"><label for="color-mode">color</label><select id="color-mode"><option value="corpus">corpus</option><option value="length">role text length</option></select><button id="reset-maps" type="button">reset views</button><input id="ci-search" type="number" placeholder="find ci"><button id="ci-go" type="button">go</button><div class="map-key"><span><i class="dot w"></i>WildChat</span><span><i class="dot l"></i>LMSYS text withheld</span></div></div>
<div class="maps-grid" id="maps-grid"><div class="map-panel"><div class="map-title"><b>Context-only UMAP</b><span>fit: 100k contexts</span></div><canvas class="map-canvas" id="context-map" aria-label="Independent context UMAP"></canvas></div><div class="map-panel"><div class="map-title"><b>Answer-only UMAP</b><span>fit: 100k answers</span></div><canvas class="map-canvas" id="answer-map" aria-label="Independent answer UMAP"></canvas></div><svg class="pair-overlay" id="pair-overlay" aria-hidden="true"></svg></div>
<div class="pair-detail"><div><span class="role-label">selected pair</span><div class="pair-meta"><b id="pair-ci">No pair selected</b><br><span id="pair-empty">Hover either map to inspect a complete pair and reveal its correspondence arrow.</span></div></div><div><span class="role-label">full context</span><pre class="full-text" id="detail-context"></pre></div><div><span class="role-label">full answer</span><pre class="full-text" id="detail-answer"></pre></div></div>
<div class="metrics"><div class="metric"><b>{quality["trustworthiness_k15"]["context"]:.3f}</b><span>context UMAP trustworthiness · k15</span></div><div class="metric"><b>{quality["trustworthiness_k15"]["answer"]:.3f}</b><span>answer UMAP trustworthiness · k15</span></div><div class="metric"><b>{quality["native_to_umap_neighbor_recall_k15"]["context"] * 100:.1f}%</b><span>context native neighbors recovered in 2-D</span></div><div class="metric"><b>{quality["native_to_umap_neighbor_recall_k15"]["answer"] * 100:.1f}%</b><span>answer native neighbors recovered in 2-D</span></div><div class="metric"><b>{quality["procrustes_shape_similarity"]:.3f}</b><span>descriptive global shape similarity after alignment</span></div></div></div></section>
<section class="section" id="linear"><div class="section-head"><div><h2>Linear dimensionality · heavy-tailed, not simply low-rank</h2><p>Participation ratio summarizes covariance-wide effective rank. Cumulative PCA thresholds ask how many orthogonal directions retain a target fraction of variance. These are separate, role-specific spectra over all 3,584 hidden dimensions.</p></div><p class="callout">The first 10 PCs explain {sum(data["linear"]["spectra"]["context"]["evr"][:10]) * 100:.1f}% of context variance and {sum(data["linear"]["spectra"]["answer"]["evr"][:10]) * 100:.1f}% of answer variance; 90% needs hundreds of PCs.</p></div>
<div class="analysis-grid"><div class="chart-block"><div class="chart-head"><div><h3>PCA spectrum</h3><p>log component rank · role-specific covariance eigenvalues</p></div><div class="seg"><button type="button" class="active" data-spectrum="cumulative">cumulative</button><button type="button" data-spectrum="scree">scree</button></div></div><canvas class="chart tall" id="spectrum-chart"></canvas><div class="linear-summary"><div><b class="c">Context: PR {linear["context"]["participation_ratio"]:.1f}</b><p>50%: {linear["context"]["n_dims_50"]} PCs · 90%: {linear["context"]["n_dims_90"]} · 99%: {linear["context"]["n_dims_99"]} · slope {linear["context"]["powerlaw_exponent"]:.3f}</p></div><div><b class="a">Answer: PR {linear["answer"]["participation_ratio"]:.1f}</b><p>50%: {linear["answer"]["n_dims_50"]} PCs · 90%: {linear["answer"]["n_dims_90"]} · 99%: {linear["answer"]["n_dims_99"]} · slope {linear["answer"]["powerlaw_exponent"]:.3f}</p></div></div></div>
<div class="chart-block"><div class="chart-head"><div><h3>Context ↔ answer CCA spectrum</h3><p>descriptive paired linear association; not held-out prediction</p></div></div><canvas class="chart tall" id="cca-chart"></canvas><div class="linear-summary"><div><b>top 10 median {float(np.median(data["linear"]["cca"][:10])):.3f}</b><p>Very strong leading shared linear modes.</p></div><div><b>top 500 median {float(np.median(data["linear"]["cca"])):.3f}</b><p>Broad paired association can coexist with neighborhood reordering.</p></div></div></div></div></section>
<section class="section" id="nonlinear"><div class="section-head"><div><h2>Nonlinear / intrinsic dimension · estimator-dependent range</h2><p>All estimates below ran separately on ambient 3,584-D fp32 context and answer vectors—not on PCA or UMAP coordinates. Curves show medians and 2.5–97.5% resample intervals at n=5k, 20k, and 50k.</p></div><p class="callout">At n=50k, 2NN gives {methods["twonn"]["median"][2]:.1f} context dimensions. The matching answer estimate is {data["nonlinear"]["answer"]["methods"]["twonn"]["median"][2]:.1f}; local methods span roughly 9–19, while local PCA reports about 48–50.</p></div>
<div class="id-layout"><div class="chart-block"><div class="chart-head"><div><h3>Stability with sample size</h3><p>band = empirical 95% resample interval</p></div><select id="id-method">{options}</select></div><canvas class="chart tall" id="id-chart"></canvas></div><div class="id-table-wrap"><table class="id-table"><thead><tr><th>estimator · n=50k</th><th class="c">context</th><th class="a">answer</th><th>A−C</th></tr></thead><tbody id="id-table-body"></tbody></table><p class="note">The spread across estimators is scientific information: each probes a different scale and geometric assumption. It is not valid to average these into one “true” dimension.</p></div></div>
<div class="findings"><div class="finding"><b>1 · Similar overall complexity</b><p>Context and answer have close global effective ranks (29.3 vs 27.8) and close PCA decay slopes. Most nonlinear estimates also differ by only a few dimensions.</p></div><div class="finding"><b>2 · Partial local transformation</b><p>Only {quality["context_answer_neighbor_overlap_k15"]["native_pca100"] * 100:.1f}% of native k15 neighbors overlap between paired spaces. Answers retain meaningful context geometry while substantially reordering local neighborhoods.</p></div><div class="finding"><b>3 · Linear relation is broad</b><p>CCA is high across many canonical modes even though the independent UMAPs have weak global shape agreement ({quality["procrustes_shape_similarity"]:.3f}). Shared linear signal does not imply identical nonlinear layout.</p></div></div></section>
<section class="section" id="tools"><div class="section-head"><div><h2>Visualization and geometry-tool inventory</h2><p>What has been run, what has not, and what a next method would add. PaCMAP or TriMap is the most useful next 2-D robustness check; Mapper/persistent homology is the most different question.</p></div><p class="callout">Recommended next: run PaCMAP separately on the same PCA-100 fit sample, then measure whether the context→answer neighborhood-overlap conclusion is stable across layouts.</p></div><div class="tool-table-wrap"><table class="tool-table"><thead><tr><th>method</th><th>status</th><th>what it answers</th></tr></thead><tbody id="tool-body"></tbody></table></div></section>
<p class="foot">Full text was recovered from pinned raw-completion files before publication filtering; no retained WildChat string is character-truncated in this dashboard. {meta["n_lmsys"]:,} LMSYS texts remain withheld. Safety filtering removed {meta["n_public_filtered"]:,} of {meta["n_raw_pairs"]:,} display pairs ({filters or "no matches"}). Display: {meta["n_pairs"]:,} public-safe pairs from {meta["n_chunks"]} fixed chunks; this is a specimen browser, not a frequency-weighted population sample. UMAP: cosine, n_neighbors={meta["umap_params"]["n_neighbors"]}, min_dist={meta["umap_params"]["min_dist"]}, random seed {meta["umap_params"]["random_state"]}. Artifact <code>{meta["umap_artifact_sha256"][:12]}</code> · UMAP producer <code>{meta["umap_producer_commit"][:12]}</code> · dimensionality producer <code>{meta["dim_producer_commit"][:12]}</code> · renderer <code>{meta["render_commit"]}</code> · {meta["generated_utc"]}. <a href="https://huggingface.co/datasets/allenai/WildChat">WildChat attribution (ODC-BY)</a>.</p>
</div><script id="payload" type="application/json">{payload}</script><script>{JS}</script></body></html>"""


def report(data: dict) -> str:
    meta = data["meta"]
    quality = data["quality"]
    linear = data["linear"]["summary"]
    context_id = data["nonlinear"]["context"]["methods"]
    answer_id = data["nonlinear"]["answer"]["methods"]
    lines = [
        "# Issue #779: separate UMAP and context/answer dimensionality",
        "",
        f"Dashboard: {meta['public_url']}",
        "",
        f"Generated: {meta['generated_utc']}",
        "",
        "## What was run",
        "",
        f"Two UMAP models were fit independently on the same {meta['n_umap_fit_per_role']:,} paired row IDs. Both use a pinned shared PCA-100 preprocessing transform, but the context graph contains only context vectors and the answer graph contains only answer vectors. Consequently their 2-D coordinate systems are independent.",
        "",
        f"The separate maps have k=15 trustworthiness {quality['trustworthiness_k15']['context']:.3f} for context and {quality['trustworthiness_k15']['answer']:.3f} for answer. Their 2-D layouts recover {quality['native_to_umap_neighbor_recall_k15']['context'] * 100:.1f}% and {quality['native_to_umap_neighbor_recall_k15']['answer'] * 100:.1f}% of native PCA-100 k15 neighbors, respectively. High trustworthiness with moderate recall means few false local neighbors but substantial information loss in two dimensions.",
        "",
        f"Native context and answer k15 neighborhoods overlap by {quality['context_answer_neighbor_overlap_k15']['native_pca100'] * 100:.1f}%. Separate UMAP neighborhoods overlap by {quality['context_answer_neighbor_overlap_k15']['separate_umap'] * 100:.1f}%; that second number is layout-dependent and should not replace the native-space result.",
        "",
        "## Linear dimensionality",
        "",
        "| space | participation ratio | PCs for 50% | PCs for 90% | PCs for 99% | power-law slope |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for role in ("context", "answer"):
        row = linear[role]
        lines.append(
            f"| {role} | {row['participation_ratio']:.2f} | {row['n_dims_50']} | "
            f"{row['n_dims_90']} | {row['n_dims_99']} | {row['powerlaw_exponent']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Both spaces are heavy-tailed rather than sharply low-rank. Their participation ratios are close, but hundreds of directions are required for 90% variance. The first few directions are strong while a long tail remains collectively important.",
            "",
            "The CCA spectrum shows strong descriptive paired linear association. This does not imply the context and answer spaces have the same local neighborhoods, and it is not a held-out prediction score.",
            "",
            "## Nonlinear / intrinsic dimensionality",
            "",
            "All estimates used ambient 3,584-dimensional fp32 vectors separately for each role. Values below are medians over five n=50,000 resamples.",
            "",
            "| estimator | context | answer | answer − context |",
            "|---|---:|---:|---:|",
        ]
    )
    for method, label in ID_METHODS.items():
        c = context_id[method]["median"][2]
        a = answer_id[method]["median"][2]
        lines.append(f"| {label} | {c:.2f} | {a:.2f} | {a - c:+.2f} |")
    lines.extend(
        [
            "",
            "2NN, kNN-MLE, correlation dimension, and local PCA disagree because they probe different neighborhood scales and make different assumptions about density, curvature, and noise. The defensible conclusion is a scale-dependent range: the most local distance estimators put both spaces around 9–19 dimensions, while the k=100 local-PCA threshold reports about 48–50. Averaging them would erase the diagnostic disagreement.",
            "",
            "## Interpretation",
            "",
            "Context and answer spaces have similar overall complexity but are not geometrically identical. Answers preserve broad paired linear structure, while local neighborhoods are substantially reordered. This is consistent with a structured context-to-answer transformation rather than either complete preservation or complete independence.",
            "",
            "The map arrow is only a correspondence marker between two independent coordinate systems. It must not be read as a vector displacement. Native-space neighborhood overlap and CCA are the quantitative relationship diagnostics.",
            "",
            "## Other visualization tools",
            "",
            "PaCMAP or TriMap is the best next 2-D robustness check because it changes the balance of local, mid-range, and global structure. PHATE or diffusion maps would be useful if the data contain trajectories. Mapper or persistent homology would test branches, loops, and connectivity rather than merely producing another scatter plot. A kNN graph with ForceAtlas2 would expose communities and bridges explicitly. t-SNE and Isomap are available but offer less direct value here given global-distance interpretability and scale concerns.",
            "",
            "## Provenance and limits",
            "",
            f"UMAP artifact SHA-256: `{meta['umap_artifact_sha256']}`; producer `{meta['umap_producer_commit']}`. Dimensionality producer: `{meta['dim_producer_commit']}`. Renderer: `{meta['render_commit']}`.",
            "",
            f"The dashboard exposes {meta['n_pairs']:,} public-safe display pairs from fixed chunks, with full retained WildChat prompt and answer text. It is designed for qualitative inspection, not population-frequency estimation.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    parser.add_argument("--umap-dir", type=Path, default=DEFAULT_UMAP_DIR)
    parser.add_argument("--pca-dir", type=Path, default=DEFAULT_PCA_DIR)
    parser.add_argument("--chunks", nargs="+", default=list(text_source.DEFAULT_CHUNKS))
    parser.add_argument("--out-name", default=OUT_NAME)
    parser.add_argument("--public-dir", type=Path, default=Path("dashboard/public"))
    parser.add_argument("--experiments-dir", type=Path, default=Path("experiments/dashboards"))
    args = parser.parse_args()

    arrays, umap_meta = load_umap(args.umap_dir)
    pca_model, _ = text_source.load_models(args.pca_dir)
    rows, sample_meta = text_source.load_fulltext_rows(
        args.export_dir, tuple(args.chunks), pca_model
    )
    attach_umap(rows, arrays)
    dim_summary, spectra, producer_meta = load_dimensions(args.export_dir)
    data = build_data(rows, sample_meta, umap_meta, dim_summary, spectra, producer_meta)
    html = page(data)
    markdown = report(data)
    for output_dir in (args.public_dir, args.experiments_dir):
        output_dir.mkdir(parents=True, exist_ok=True)
        output = output_dir / args.out_name
        output.write_text(html, encoding="utf-8")
        print(f"[separate-umap-dashboard] wrote {output} ({len(html.encode()) / 1e6:.2f} MB)")
    report_path = args.experiments_dir / REPORT_NAME
    report_path.write_text(markdown, encoding="utf-8")
    print(f"[separate-umap-dashboard] wrote {report_path}")


if __name__ == "__main__":
    main()
