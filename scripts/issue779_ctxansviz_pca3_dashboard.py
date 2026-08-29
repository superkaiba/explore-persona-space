"""Build the issue #779 hoverable PC1-PC3 context/answer dashboard.

The original ctxansviz export persisted only PC1-PC2 even though its joint PCA
model has 100 components. This renderer recovers PC3 from a deterministic,
corpus-stratified sample of roughly 5,500 pairs spanning the original L19
capture, applies the already-fitted joint PCA basis, and writes one
self-contained HTML dashboard to both public dashboard surfaces.

The three panels intentionally share the same PCA basis, isotropic axis scaling,
and camera. They differ only in which paired points are visible: contexts,
answers, or both. Hovering any point highlights the corresponding pair in all
panels and reveals the source text.

Usage:
    uv run python scripts/issue779_ctxansviz_pca3_dashboard.py
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import hashlib
import ipaddress
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import hf_hub_download

from explore_persona_space.orchestrate.provenance import commit_string, git_provenance

HF_REPO = "superkaiba1/explore-persona-space-data"
CAPTURE_REVISION = "cbc55efdd7f5581677047e487aa61172f6e7944d"
EXPORT_REVISION = "d155ed93f4b0184a477cea51aef65cc5440da588"
EXPORT_PRODUCER_COMMIT = "79d9142bf5c88ae2ccd3ff7270e9d98a1faaaa5d"
CAPTURE_PREFIX = "issue779_monitoring/fitter-fair-comparison-n1m/final_token_capture"
LAYER = 19
OUT_NAME = "ctxansviz-779-pca3-full.html"
OUT_DIRS = (Path("dashboard/public"), Path("experiments/dashboards"))
DEFAULT_EXPORT = Path("data/issue_779/ctxansviz_dl/full/issue779_monitoring/ctxansviz")

# Eleven corpus-stratified chunks span the full 32-shard capture: six evenly
# spaced LMSYS chunks and five evenly spaced WildChat chunks. The 6:5 allocation
# matches the full corpus's 54.75% / 45.25% split to within 0.3 percentage points
# before the producer's small skipped-row count is applied.
DEFAULT_CHUNKS = (
    "shard00_chunk0000.pt",
    "shard03_chunk0030.pt",
    "shard07_chunk0000.pt",
    "shard10_chunk0030.pt",
    "shard14_chunk0000.pt",
    "shard17_chunk0028.pt",
    "shard17_chunk0031.pt",
    "shard21_chunk0003.pt",
    "shard24_chunk0036.pt",
    "shard28_chunk0008.pt",
    "shard31_chunk0040.pt",
)

# Public dashboards must not reproduce user-supplied credential or PII-shaped
# strings from conversational corpora. These gates intentionally favor dropping
# a small number of rows over trying to guess whether a realistic-looking value
# is genuine. They are deterministic and corpus-specific; the repository's gist
# redactor is deliberately not used here because it covers operator secrets,
# not arbitrary third-party conversation text.
PUBLIC_DROP_PATTERNS = (
    (
        "email",
        re.compile(r"(?i)(?<![\w.+-])[\w.+-]+@[a-z0-9-]+(?:\.[a-z0-9-]+)+"),
    ),
    (
        "credential-label",
        re.compile(
            r"(?i)\b(?:password|passwd|passphrase|pwd|login|username|user[ _-]?name|"
            r"api[ _-]?key|access[ _-]?token|secret|senha|contrase(?:n|ñ)a|"
            r"пароль|логин|密码|密碼|비밀번호)\b\s*(?::|=|\bis\b)?\s*"
            r"[\"']?[A-Za-z0-9@._!#$%&*+\-/]{6,}"
        ),
    ),
    (
        "token",
        re.compile(
            r"(?:\bhf_[A-Za-z0-9]{30,}|\bsk-ant-[A-Za-z0-9_-]{40,}|"
            r"\bsk-[A-Za-z0-9_-]{40,}\b|"
            r"hooks\.slack\.com/services/[A-Z0-9]+/[A-Z0-9]+/[A-Za-z0-9]+)"
        ),
    ),
    (
        "embedded-url-credential",
        re.compile(r"(?i)https?://[^\s/@:]+:[^\s/@]+@"),
    ),
    (
        "medical-identifier",
        re.compile(r"(?i)\b(?:CLIA|MRN|medical record number|patient[ _-]?id)\b"),
    ),
)
PHONE_CANDIDATE = re.compile(r"(?<![\w.])\+?\d[\d\s().-]{6,}\d(?![\w.])")
IPV4_CANDIDATE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
CHUNK_NAME = re.compile(r"^shard(\d+)_chunk(\d+)\.pt$")
LMSYS_TEXT_WITHHELD = (
    "[LMSYS source text withheld from this public dashboard under its dataset license]"
)


def iter_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _has_phone_candidate(text: str) -> bool:
    return any(
        10 <= sum(ch.isdigit() for ch in m.group(0)) <= 15 for m in PHONE_CANDIDATE.finditer(text)
    )


def _redact_ipv4(text: str) -> str:
    def replace(match: re.Match[str]) -> str:
        try:
            ipaddress.IPv4Address(match.group(0))
        except ipaddress.AddressValueError:
            return match.group(0)
        return "<ip>"

    return IPV4_CANDIDATE.sub(replace, text)


def _prepare_public_rows(rows: list[dict]) -> tuple[list[dict], dict[str, int], int]:
    """Withhold gated LMSYS text; filter and sanitize distributable WildChat text."""
    kept: list[dict] = []
    reason_counts: dict[str, int] = {}
    n_lmsys_text_withheld = 0
    for row in rows:
        if row["corpus"] == "lmsys":
            row["context"] = LMSYS_TEXT_WITHHELD
            row["answer"] = LMSYS_TEXT_WITHHELD
            kept.append(row)
            n_lmsys_text_withheld += 1
            continue
        combined = f"{row['context']}\n{row['answer']}"
        reasons = {name for name, pattern in PUBLIC_DROP_PATTERNS if pattern.search(combined)}
        if _has_phone_candidate(combined):
            reasons.add("phone-or-account-number")
        if reasons:
            for reason in reasons:
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            continue
        row["context"] = _redact_ipv4(row["context"])
        row["answer"] = _redact_ipv4(row["answer"])
        sanitized = f"{row['context']}\n{row['answer']}"
        if _has_phone_candidate(sanitized):
            reason_counts["phone-or-account-number"] = (
                reason_counts.get("phone-or-account-number", 0) + 1
            )
            continue
        kept.append(row)

    for row in kept:
        combined = f"{row['context']}\n{row['answer']}"
        if any(pattern.search(combined) for _, pattern in PUBLIC_DROP_PATTERNS):
            raise RuntimeError(f"public-text safety invariant failed for ci={row['ci']}")
        if _has_phone_candidate(combined):
            raise RuntimeError(f"public phone-number safety invariant failed for ci={row['ci']}")
    if not kept:
        raise RuntimeError("public-text safety filter removed every sampled row")
    return kept, dict(sorted(reason_counts.items())), n_lmsys_text_withheld


def _load_pc3(export_dir: Path, chunks: tuple[str, ...]) -> tuple[list[dict], dict]:
    model_path = export_dir / "pca_model.npz"
    if not model_path.exists():
        raise FileNotFoundError(f"missing joint PCA model: {model_path}")
    download_meta_path = export_dir / "_download_meta.json"
    producer_meta_path = export_dir / "meta.json"
    if not download_meta_path.exists() or not producer_meta_path.exists():
        raise FileNotFoundError(
            "export provenance files _download_meta.json and meta.json are required"
        )
    download_meta = json.loads(download_meta_path.read_text(encoding="utf-8"))
    producer_meta = json.loads(producer_meta_path.read_text(encoding="utf-8"))
    if download_meta.get("revision") != EXPORT_REVISION:
        raise RuntimeError(
            f"export revision {download_meta.get('revision')} != pinned {EXPORT_REVISION}"
        )
    if producer_meta.get("git_commit") != EXPORT_PRODUCER_COMMIT:
        raise RuntimeError(
            f"export producer {producer_meta.get('git_commit')} != pinned {EXPORT_PRODUCER_COMMIT}"
        )
    expected_pca_sha = producer_meta.get("export_files_sha256", {}).get("pca_model.npz")
    if not expected_pca_sha or sha256_file(model_path) != expected_pca_sha:
        raise RuntimeError("joint PCA model sha256 does not match the pinned export manifest")
    if int(producer_meta.get("layer", -1)) != LAYER:
        raise RuntimeError(f"export layer {producer_meta.get('layer')} != requested L{LAYER}")
    model = np.load(model_path)
    components = np.asarray(model["components"][:3], dtype=np.float32)
    mean = np.asarray(model["mean"], dtype=np.float32)
    evr = np.asarray(model["explained_variance_ratio"][:3], dtype=np.float64)
    if components.shape != (3, 3584) or mean.shape != (3584,):
        raise RuntimeError(
            f"unexpected PCA model shapes: components={components.shape}, mean={mean.shape}"
        )

    raw_rows: list[tuple[int, np.ndarray, np.ndarray]] = []
    for chunk_name in chunks:
        path = hf_hub_download(
            HF_REPO,
            filename=f"{CAPTURE_PREFIX}/{chunk_name}",
            repo_type="dataset",
            revision=CAPTURE_REVISION,
        )
        bundle = torch.load(path, mmap=True, weights_only=False, map_location="cpu")
        layers = [int(x) for x in bundle["layers"]]
        if LAYER not in layers:
            raise RuntimeError(f"{chunk_name}: layer {LAYER} absent from {layers}")
        col = layers.index(LAYER)
        cx = bundle["cx_last"][:, col, :].to(torch.float32).numpy()
        vx = bundle["v_x"][:, col, :].to(torch.float32).numpy()
        cis = [int(x) for x in bundle["ci"]]
        if cx.shape != vx.shape or cx.shape[1] != 3584 or len(cis) != cx.shape[0]:
            raise RuntimeError(
                f"{chunk_name}: malformed capture shapes cx={cx.shape}, vx={vx.shape}, ci={len(cis)}"
            )
        pc_cx = (cx - mean) @ components.T
        pc_vx = (vx - mean) @ components.T
        raw_rows.extend((ci, pc_cx[i], pc_vx[i]) for i, ci in enumerate(cis))

    target_cis = {ci for ci, _, _ in raw_rows}
    if len(target_cis) != len(raw_rows):
        raise RuntimeError("duplicate ci values across selected capture chunks")
    text_by_ci: dict[int, dict] = {}
    export_hashes = producer_meta.get("export_files_sha256", {})
    for part in sorted(export_dir.glob("row_meta_*.jsonl")):
        expected_sha = export_hashes.get(part.name)
        if not expected_sha or sha256_file(part) != expected_sha:
            raise RuntimeError(f"hover metadata sha256 mismatch: {part.name}")
        for row in iter_jsonl(part):
            ci = int(row["ci"])
            if ci in target_cis:
                text_by_ci[ci] = row
        if len(text_by_ci) == len(target_cis):
            break
    missing = sorted(target_cis - text_by_ci.keys())
    if missing:
        raise RuntimeError(
            f"{len(missing)} sampled ci values missing hover metadata: {missing[:8]}"
        )

    rows = []
    for ci, pc_cx, pc_vx in raw_rows:
        meta = text_by_ci[ci]
        rows.append(
            {
                "ci": ci,
                "corpus": str(meta["corpus"]),
                "c": [round(float(x), 5) for x in pc_cx],
                "a": [round(float(x), 5) for x in pc_vx],
                "context": str(meta["context_text"]),
                "answer": str(meta["answer_text"]),
            }
        )

    n_raw_pairs = len(rows)
    rows, public_filter_counts, n_lmsys_text_withheld = _prepare_public_rows(rows)

    stacked = np.concatenate(
        [
            np.asarray([r["c"] for r in rows], dtype=np.float64),
            np.asarray([r["a"] for r in rows], dtype=np.float64),
        ]
    )
    per_axis_limits = np.quantile(np.abs(stacked), 0.99, axis=0)
    axis_scale = float(per_axis_limits.max())
    if axis_scale <= 0:
        raise RuntimeError(f"non-positive PCA axis scale: {axis_scale}")
    n_lmsys = sum(row["corpus"] == "lmsys" for row in rows)
    n_wildchat = sum(row["corpus"] == "wildchat" for row in rows)
    if n_lmsys + n_wildchat != len(rows):
        raise RuntimeError("sample contains a corpus other than lmsys/wildchat")
    parsed_chunks = [CHUNK_NAME.fullmatch(name) for name in chunks]
    if any(match is None for match in parsed_chunks):
        raise RuntimeError(f"invalid capture chunk name in {chunks}")
    shard_ids = [int(match.group(1)) for match in parsed_chunks if match is not None]
    n_total = int(producer_meta["n_rows"])
    summary = {
        "n_pairs": len(rows),
        "n_raw_pairs": n_raw_pairs,
        "n_public_filtered": n_raw_pairs - len(rows),
        "public_filter_counts": public_filter_counts,
        "n_lmsys_text_withheld": n_lmsys_text_withheld,
        "n_total": n_total,
        "sample_fraction": round(len(rows) / n_total, 8),
        "n_lmsys": n_lmsys,
        "n_wildchat": n_wildchat,
        "n_total_lmsys": int(producer_meta["n_lmsys"]),
        "n_total_wildchat": int(producer_meta["n_wildchat"]),
        "layer": LAYER,
        "source_seed": int(producer_meta["seed"]),
        "chunks": list(chunks),
        "n_chunks": len(chunks),
        "n_distinct_capture_shards": len(set(shard_ids)),
        "capture_shard_min": min(shard_ids),
        "capture_shard_max": max(shard_ids),
        "capture_revision": CAPTURE_REVISION,
        "export_revision": EXPORT_REVISION,
        "export_producer_commit": EXPORT_PRODUCER_COMMIT,
        "render_commit": commit_string(git_provenance()),
        "pca_fit_per_side": int(model["n_fit_per_side"]),
        "evr": [round(float(x), 8) for x in evr],
        "axis_scale": round(axis_scale, 6),
        "per_axis_99pct_abs": [round(float(x), 6) for x in per_axis_limits],
    }
    return rows, summary


CSS = r"""
:root{--paper:#f4f1e9;--ink:#1e211f;--muted:#66675f;--line:#cbc7bb;--panel:#fffefa;
--context:#285d7d;--answer:#b96e16;--select:#bd3528;--axis:#8d8b82}
*{box-sizing:border-box}body{margin:0;background:var(--paper);color:var(--ink);
font-family:ui-monospace,SFMono-Regular,Menlo,Monaco,Consolas,"Liberation Mono",monospace}
.wrap{max-width:1840px;margin:0 auto;padding:20px 22px 28px}header{display:flex;align-items:flex-start;
justify-content:space-between;gap:24px;border-bottom:1px solid var(--line);padding-bottom:14px}
h1{font-size:21px;line-height:1.25;margin:0 0 7px;font-weight:650;letter-spacing:-.025em}
.lede{font-size:12px;line-height:1.55;color:var(--muted);max-width:980px}.key{display:flex;gap:18px;
font-size:12px;white-space:nowrap;padding-top:4px}.swatch{display:inline-block;width:10px;height:10px;
margin-right:6px}.swatch.c{background:var(--context)}.swatch.a{background:var(--answer)}
.toolbar{min-height:52px;display:flex;align-items:center;gap:12px;border-bottom:1px solid var(--line);
font-size:12px}.toolbar button{font:inherit;color:var(--ink);background:var(--panel);border:1px solid #aaa69b;
border-radius:5px;padding:6px 10px;cursor:pointer}.toolbar button:hover{border-color:#55534e}.toolbar label{display:flex;
align-items:center;gap:7px}.toolbar input[type=range]{width:110px;accent-color:#5c5a54}.note{color:var(--muted);
margin-left:auto}.plots{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));border-left:1px solid var(--line)}
.plot{background:var(--panel);border-right:1px solid var(--line);border-bottom:1px solid var(--line)}
.plot-head{height:42px;display:flex;align-items:center;justify-content:space-between;padding:0 12px;
border-bottom:1px solid var(--line);font-size:13px;font-weight:650}.plot-head span{font-weight:400;color:var(--muted);
font-size:11px}.stage{position:relative;height:510px}.stage canvas{display:block;width:100%;height:100%;cursor:grab;
touch-action:none}.stage canvas.dragging{cursor:grabbing}.hint{position:absolute;left:10px;bottom:8px;color:#77756e;
font-size:10px;pointer-events:none}.selection{display:grid;grid-template-columns:190px minmax(0,1fr) minmax(0,1fr);
background:var(--panel);border:1px solid var(--line);border-top:0;min-height:162px}.selection>div{padding:13px 15px;
border-right:1px solid var(--line)}.selection>div:last-child{border-right:0}.selection h2{font-size:12px;margin:0 0 9px;
font-weight:650}.selection p{font-family:inherit;font-size:12px;line-height:1.5;margin:0;white-space:pre-wrap;
overflow-wrap:anywhere}.meta-lines{font-size:11px;line-height:1.65;color:var(--muted)}.meta-lines b{color:var(--ink)}
#tip{position:fixed;display:none;pointer-events:none;z-index:10;max-width:330px;background:#1f211f;color:#f7f4eb;
border:1px solid #111;padding:8px 10px;font-size:11px;line-height:1.45;box-shadow:0 2px 8px rgba(0,0,0,.16)}
footer{font-size:10px;color:#74736d;line-height:1.55;margin-top:12px}code{font-family:inherit;color:#4c4b46}
@media(max-width:1120px){.plots{grid-template-columns:1fr}.stage{height:540px}.selection{grid-template-columns:1fr}
.selection>div{border-right:0;border-bottom:1px solid var(--line)}header{display:block}.key{margin-top:10px}.note{display:none}}
"""


JS = r"""
const D=JSON.parse(document.getElementById('payload').textContent);
const COLORS={context:'#285d7d',answer:'#b96e16',select:'#bd3528',axis:'#8d8b82'};
const panels=[...document.querySelectorAll('canvas[data-view]')].map(canvas=>({canvas,view:canvas.dataset.view,
ctx:canvas.getContext('2d'),screen:[]}));
let camera={yaw:-0.68,pitch:0.34,zoom:0.95};let selected=null;let pointSize=1.55;let showLinks=true;
let drag=null;let raf=0;const axisScale=D.meta.axis_scale;
const normalized=D.rows.map(r=>({c:r.c.map(v=>v/axisScale),a:r.a.map(v=>v/axisScale)}));
function resize(panel){const dpr=Math.min(devicePixelRatio||1,2),box=panel.canvas.getBoundingClientRect();
 const w=Math.max(320,Math.round(box.width*dpr)),h=Math.max(300,Math.round(box.height*dpr));
 if(panel.canvas.width!==w||panel.canvas.height!==h){panel.canvas.width=w;panel.canvas.height=h}return{w,h,dpr}}
function rotate(p){const cy=Math.cos(camera.yaw),sy=Math.sin(camera.yaw),cp=Math.cos(camera.pitch),sp=Math.sin(camera.pitch);
 const x=p[0]*cy+p[2]*sy,z=-p[0]*sy+p[2]*cy;return[x,p[1]*cp-z*sp,p[1]*sp+z*cp]}
function project(p,w,h){const q=rotate(p),s=Math.min(w,h)*0.34*camera.zoom;return[w/2+q[0]*s,h/2-q[1]*s,q[2]]}
function axis(panel,w,h,dpr){const ctx=panel.ctx,o=project([0,0,0],w,h);ctx.save();ctx.lineWidth=1*dpr;
 ctx.font=`${10*dpr}px ui-monospace,monospace`;ctx.textBaseline='middle';
 [[[1.12,0,0],'PC1'],[[0,1.12,0],'PC2'],[[0,0,1.12],'PC3']].forEach(([end,label])=>{const p=project(end,w,h);
 ctx.strokeStyle=COLORS.axis;ctx.globalAlpha=.72;ctx.beginPath();ctx.moveTo(o[0],o[1]);ctx.lineTo(p[0],p[1]);ctx.stroke();
 ctx.globalAlpha=1;ctx.fillStyle='#67665f';ctx.fillText(label,p[0]+5*dpr,p[1]);});ctx.restore()}
function arrow(ctx,a,b,dpr,alpha=1){ctx.save();ctx.strokeStyle=COLORS.select;ctx.fillStyle=COLORS.select;ctx.globalAlpha=alpha;
 ctx.lineWidth=1.25*dpr;ctx.beginPath();ctx.moveTo(a[0],a[1]);ctx.lineTo(b[0],b[1]);ctx.stroke();
 const ang=Math.atan2(b[1]-a[1],b[0]-a[0]),len=6*dpr;ctx.beginPath();ctx.moveTo(b[0],b[1]);
 ctx.lineTo(b[0]-len*Math.cos(ang-.45),b[1]-len*Math.sin(ang-.45));ctx.lineTo(b[0]-len*Math.cos(ang+.45),b[1]-len*Math.sin(ang+.45));ctx.closePath();ctx.fill();ctx.restore()}
function drawPanel(panel){const {w,h,dpr}=resize(panel),ctx=panel.ctx;ctx.clearRect(0,0,w,h);axis(panel,w,h,dpr);panel.screen=[];
 if(panel.view==='joint'&&showLinks){ctx.save();ctx.strokeStyle='#77756e';ctx.lineWidth=.55*dpr;ctx.globalAlpha=.075;
  const stride=Math.max(1,Math.floor(D.rows.length/180));for(let i=0;i<D.rows.length;i+=stride){const c=project(normalized[i].c,w,h),a=project(normalized[i].a,w,h);ctx.beginPath();ctx.moveTo(c[0],c[1]);ctx.lineTo(a[0],a[1]);ctx.stroke()}ctx.restore()}
 const roles=panel.view==='context'?['c']:panel.view==='answer'?['a']:['c','a'];let pts=[];
 for(const role of roles)for(let i=0;i<D.rows.length;i++){const p=project(normalized[i][role],w,h);pts.push({x:p[0],y:p[1],z:p[2],i,role})}
 pts.sort((u,v)=>u.z-v.z);ctx.save();for(const p of pts){ctx.globalAlpha=panel.view==='joint'?.48:.62;ctx.fillStyle=p.role==='c'?COLORS.context:COLORS.answer;
  const r=pointSize*dpr*(.88+(p.z+1.7)*.09);ctx.beginPath();ctx.arc(p.x,p.y,Math.max(.9*dpr,r),0,Math.PI*2);ctx.fill();panel.screen.push(p)}ctx.restore();
 if(selected!==null){const c=project(normalized[selected].c,w,h),a=project(normalized[selected].a,w,h);if(panel.view==='joint')arrow(ctx,c,a,dpr,1);
  for(const [role,p] of [['c',c],['a',a]])if(panel.view==='joint'||(panel.view==='context'&&role==='c')||(panel.view==='answer'&&role==='a')){
   ctx.save();ctx.strokeStyle=COLORS.select;ctx.fillStyle=role==='c'?COLORS.context:COLORS.answer;ctx.lineWidth=2*dpr;ctx.beginPath();ctx.arc(p[0],p[1],5*dpr,0,Math.PI*2);ctx.fill();ctx.stroke();ctx.restore()}}
}
function render(){raf=0;panels.forEach(drawPanel)}function requestRender(){if(!raf)raf=requestAnimationFrame(render)}
function nearest(panel,x,y){let best=null,bd=100;for(const p of panel.screen){const d=(p.x-x)**2+(p.y-y)**2;if(d<bd){bd=d;best=p}}return best}
function showTip(ev,p){const r=D.rows[p.i],tip=document.getElementById('tip');tip.textContent=`${p.role==='c'?'context':'answer'} · ci ${r.ci} · ${r.corpus} · PC ${r[p.role].map(v=>v.toFixed(2)).join(' / ')}`;
 tip.style.display='block';tip.style.left=Math.min(ev.clientX+13,innerWidth-350)+'px';tip.style.top=Math.min(ev.clientY+13,innerHeight-65)+'px'}
function select(i){selected=i;const r=D.rows[i];document.getElementById('sel-id').textContent=`ci ${r.ci}`;document.getElementById('sel-corpus').textContent=r.corpus;
 document.getElementById('sel-c').textContent=r.c.map(v=>v.toFixed(3)).join(' / ');document.getElementById('sel-a').textContent=r.a.map(v=>v.toFixed(3)).join(' / ');
 document.getElementById('context-text').textContent=r.context;document.getElementById('answer-text').textContent=r.answer;requestRender()}
for(const panel of panels){const cv=panel.canvas;cv.addEventListener('pointerdown',ev=>{cv.setPointerCapture(ev.pointerId);drag={x:ev.clientX,y:ev.clientY};cv.classList.add('dragging')});
 cv.addEventListener('pointermove',ev=>{if(drag){camera.yaw+=(ev.clientX-drag.x)*.008;camera.pitch=Math.max(-1.25,Math.min(1.25,camera.pitch+(ev.clientY-drag.y)*.008));drag={x:ev.clientX,y:ev.clientY};requestRender();return}
  const b=cv.getBoundingClientRect(),sx=cv.width/b.width,sy=cv.height/b.height,p=nearest(panel,(ev.clientX-b.left)*sx,(ev.clientY-b.top)*sy);if(p){showTip(ev,p);if(selected!==p.i)select(p.i)}else document.getElementById('tip').style.display='none'});
 cv.addEventListener('pointerup',()=>{drag=null;cv.classList.remove('dragging')});cv.addEventListener('pointercancel',()=>{drag=null;cv.classList.remove('dragging')});
 cv.addEventListener('mouseleave',()=>document.getElementById('tip').style.display='none');cv.addEventListener('wheel',ev=>{ev.preventDefault();camera.zoom=Math.max(.45,Math.min(2.1,camera.zoom*Math.exp(-ev.deltaY*.001)));requestRender()},{passive:false})}
document.getElementById('reset').addEventListener('click',()=>{camera={yaw:-.68,pitch:.34,zoom:.95};requestRender()});
document.getElementById('links').addEventListener('change',ev=>{showLinks=ev.target.checked;requestRender()});
document.getElementById('size').addEventListener('input',ev=>{pointSize=Number(ev.target.value);requestRender()});
addEventListener('resize',requestRender);select(0);render();
"""


def _page(rows: list[dict], meta: dict) -> str:
    evr = meta["evr"]
    payload = json.dumps({"rows": rows, "meta": meta}, ensure_ascii=False, separators=(",", ":"))
    json.loads(payload)
    # Escaping every less-than sign prevents script termination and HTML comment
    # parsing hazards while remaining valid JSON (the browser decodes \u003c).
    payload = payload.replace("<", "\\u003c")
    json.loads(payload)
    sample_pct = 100 * meta["sample_fraction"]
    shard_word = "shard" if meta["n_distinct_capture_shards"] == 1 else "shards"
    block_sample = (
        f"{meta['n_chunks']} fixed contiguous capture chunks spanning shard "
        f"{meta['capture_shard_min']:02d} through {meta['capture_shard_max']:02d} "
        f"({meta['n_distinct_capture_shards']} distinct {shard_word} represented; "
        "not a uniform random draw)"
    )
    page = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>PC1–PC3 context / answer atlas | issue #779</title><style>{CSS}</style></head><body>
<div class="wrap"><header><div><h1>PC1–PC3 context / answer atlas</h1>
<div class="lede">Three synchronized views of {meta["n_pairs"]:,} paired L{meta["layer"]} activations ({sample_pct:.2f}% of {meta["n_total"]:,}; {meta["n_lmsys"]:,} LMSYS + {meta["n_wildchat"]:,} WildChat), retained from {meta["n_raw_pairs"]:,} rows in {block_sample}. One joint PCA basis, fit on {meta["pca_fit_per_side"]:,} contexts and {meta["pca_fit_per_side"]:,} true answers, is used everywhere. Drag any plot to rotate; scroll to zoom; hover to inspect the exact pair and available source text.</div></div>
<div class="key"><span><i class="swatch c"></i>context</span><span><i class="swatch a"></i>true answer</span></div></header>
<div class="toolbar"><button id="reset" type="button">Reset view</button><label><input id="links" type="checkbox" checked> sampled pair links</label>
<label>point size <input id="size" type="range" min="0.8" max="3" step="0.1" value="1.55"></label>
<span class="note">shared camera · shared isotropic PCA-unit scale · raw scores on hover</span></div>
<main class="plots">
<section class="plot"><div class="plot-head">Contexts <span>{meta["n_pairs"]:,} points</span></div><div class="stage"><canvas data-view="context" tabindex="0" aria-label="3D context PCA scatter"></canvas><div class="hint">drag rotate · wheel zoom · hover inspect</div></div></section>
<section class="plot"><div class="plot-head">Answers <span>{meta["n_pairs"]:,} points</span></div><div class="stage"><canvas data-view="answer" tabindex="0" aria-label="3D answer PCA scatter"></canvas><div class="hint">same basis · same scale · same camera</div></div></section>
<section class="plot"><div class="plot-head">Joint <span>{meta["n_pairs"]:,} paired rows</span></div><div class="stage"><canvas data-view="joint" tabindex="0" aria-label="3D joint context and answer PCA scatter"></canvas><div class="hint">hover highlights context → answer</div></div></section>
</main>
<section class="selection" aria-live="polite"><div><h2>Selected pair</h2><div class="meta-lines"><b id="sel-id">…</b><br>corpus <b id="sel-corpus">…</b><br>context PC1/2/3<br><b id="sel-c">…</b><br>answer PC1/2/3<br><b id="sel-a">…</b></div></div>
<div><h2>Context</h2><p id="context-text">Hover a point to inspect its paired text.</p></div><div><h2>Answer</h2><p id="answer-text">Hover a point to inspect its paired text.</p></div></section>
<footer>Qwen2.5-7B-Instruct · layer {meta["layer"]} · source seed {meta["source_seed"]} · {meta["n_pairs"]:,} / {meta["n_total"]:,} displayed pairs ({sample_pct:.2f}%) from {block_sample} · {meta["n_public_filtered"]:,} of {meta["n_raw_pairs"]:,} sampled rows omitted because WildChat text matched credential/PII safety patterns; valid IPv4 literals replaced with &lt;ip&gt; · source-text policy: all {meta["n_lmsys_text_withheld"]:,} <a href="https://huggingface.co/datasets/lmsys/lmsys-chat-1m">LMSYS-Chat-1M</a> hover texts withheld under its dataset agreement; displayed <a href="https://huggingface.co/datasets/allenai/WildChat">WildChat</a> excerpts attributed under ODC-BY · top-three explained-variance shares {evr[0]:.4f}, {evr[1]:.4f}, {evr[2]:.4f} (sum {sum(evr):.4f}; the remaining {1 - sum(evr):.4f} is not shown) · one joint 99th-percentile absolute score ({meta["axis_scale"]:.2f}) scales all three axes isotropically; hover reports raw PC scores · capture <code>{meta["capture_revision"][:12]}</code> · export <code>{meta["export_revision"][:12]}</code> · PCA producer <code>{meta["export_producer_commit"][:12]}</code> · renderer <code>{meta["render_commit"]}</code> · generated {time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}.</footer></div>
<div id="tip"></div><script id="payload" type="application/json">{payload}</script><script>{JS}</script></body></html>"""
    return page


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0] if __doc__ else "")
    ap.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT)
    ap.add_argument("--out-name", default=OUT_NAME)
    ap.add_argument("--chunks", nargs="*", default=list(DEFAULT_CHUNKS))
    args = ap.parse_args()
    chunks = tuple(args.chunks)
    if not chunks:
        raise ValueError("--chunks must contain at least one capture shard")
    rows, meta = _load_pc3(args.export_dir, chunks)
    content = _page(rows, meta)
    for out_dir in OUT_DIRS:
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / args.out_name
        out.write_text(content, encoding="utf-8")
        print(f"[pca3-dashboard] wrote {out} ({len(content.encode('utf-8')) / 1e6:.2f} MB)")
    print(f"[pca3-dashboard] done pairs={len(rows)}")


if __name__ == "__main__":
    main()
