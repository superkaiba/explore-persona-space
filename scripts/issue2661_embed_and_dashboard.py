"""Issue #2661 — embedding coverage metric + the edge dashboard.

`embed` (pod or any GPU; CPU-tiny under --smoke): Qwen3-Embedding-8B embeddings
of (a) every context-SAE W1 description, (b) every answer-side description
(the committed #2552 copy), (c) the 24 W2 summary fields per eval turn; then
Der's coverage metric — per field, the mean cosine of the top-k (k in
{1,2,3,5}) most similar feature descriptions from the turn's judged list.
Embedder + pooling VERBATIM from issue2552_turnsae_der.py phase_p4_embed
(_p4_load_embedder/_p4_embed_texts @ cb39df3ce1c).

`dashboard` (VM, CPU): single-file HTML —
  1. SAE metric table for both dictionaries with Der's paper numbers beside ours;
  2. per-feature R^2 / AUROC vs firing count per route (base64 PNGs);
  3. surviving edges, each row showing BOTH descriptions in words (ids in a
     muted suffix), coefficient, split-half values, null z;
  4. receipts tables for the behavior answer features;
  5. "unexpected edges": surviving edges ranked by |coef| x (1 - cosine between
     the two descriptions' embeddings);
  6. provenance footer (commits, HF revisions, judge model, spend estimate).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import argparse  # noqa: E402
import base64  # noqa: E402
import functools  # noqa: E402
import html  # noqa: E402
import io  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402

from explore_persona_space.atomic_io import savez_atomic  # noqa: E402

from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue2661.embed")

PROJECT_ROOT = _SCRIPTS_DIR.parent
EVAL_DIR = PROJECT_ROOT / "eval_results" / "issue_2661"
AGG_DIR = EVAL_DIR / "judge_aggregates"
ANSWER_DESC = EVAL_DIR / "inputs" / "descriptions_rep_ta.json"
TOPK_COVERAGE = (1, 2, 3, 5)
SEED = 2661
# Context-side watch list (task body receipts paragraph: "a context-side watch
# list from the eval-list mining: China/Taiwan/Xinjiang/Tibet/CCP topic features
# once labelled — that join is VM-side"). Zero-spend: regex over the W1 ctx
# descriptions; edges read off the pod's wiring_edges.npz.
WATCHLIST_PATTERN = (
    r"china|chinese|taiwan|xinjiang|tibet|ccp|communist party|uighur|uyghur|"
    r"hong kong|prc\b|beijing|mainland china|one[- ]china"
)
WATCH_TOP_OUT_EDGES = 10


@functools.cache
def _u2661():
    import issue2661_flat_ctx_sae as u

    return u


@functools.cache
def _jw():
    import issue2661_judge_waves as j

    return j


# ── embedder (VERBATIM #2552 _p4_load_embedder / _p4_embed_texts) ────────────────


def _load_embedder(args):
    import torch
    from transformers import AutoConfig, AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.emb_model, padding_side="left")
    if args.tiny_model or args.smoke:
        cfg = AutoConfig.from_pretrained(args.emb_model)
        cfg.num_hidden_layers = 2
        cfg.hidden_size = 128
        cfg.intermediate_size = 256
        cfg.num_attention_heads = 4
        cfg.num_key_value_heads = 2
        if hasattr(cfg, "head_dim"):
            cfg.head_dim = 32
        model = AutoModel.from_config(cfg).to(torch.float32)
        logger.warning("[embed] SMOKE/TINY twin: random weights; REAL code path + tokenizer")
    else:
        model = AutoModel.from_pretrained(args.emb_model, torch_dtype=torch.bfloat16)
    model = model.to(args.device).eval()
    return model, tok


def _embed_texts(args, model, tok, texts: list[str]) -> np.ndarray:
    """Pool the last position, then unit-normalize rows (fp32, on CPU)."""
    import torch

    out = np.empty((len(texts), int(model.config.hidden_size)), np.float32)
    bs = int(args.emb_batch)
    with torch.no_grad():
        for s in range(0, len(texts), bs):
            batch = texts[s : s + bs]
            enc = tok(
                batch,
                padding=True,
                truncation=True,
                max_length=int(args.emb_max_tokens),
                return_tensors="pt",
            ).to(args.device)
            h = model(**enc).last_hidden_state
            mask = enc["attention_mask"]
            if bool((mask[:, -1] == 1).all()):
                emb = h[:, -1]
            else:
                idx = mask.sum(dim=1) - 1
                emb = h[torch.arange(h.size(0), device=h.device), idx]
            emb = torch.nn.functional.normalize(emb.float(), dim=-1)
            out[s : s + len(batch)] = emb.cpu().numpy()
            if (s // bs) % 50 == 0:
                print(f"[embed] embedded {s + len(batch)}/{len(texts)}", flush=True)
    return out


def _out_dir(args) -> Path:
    d = args.out_root if not args.smoke else args.out_root / "smoke"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _load_ctx_descriptions(args) -> dict[int, str]:
    path = Path(args.judge_agg) / "descriptions_ctx.json"
    assert path.exists(), f"descriptions_ctx.json missing at {path} — run judge w1 first"
    doc = json.loads(path.read_text())
    return {int(k): v for k, v in doc["descriptions"].items()}


def _load_ans_descriptions() -> dict[int, str]:
    assert ANSWER_DESC.exists(), f"committed answer descriptions missing: {ANSWER_DESC}"
    doc = json.loads(ANSWER_DESC.read_text())
    return {int(k): v for k, v in doc["descriptions"].items()}


def _load_summaries(args) -> dict[int, dict[str, str]]:
    cands = sorted(Path(args.judge_agg).glob("summaries_*.json"))
    assert cands, f"summaries_*.json missing under {args.judge_agg} — run judge w2 first"
    doc = json.loads(cands[-1].read_text())
    return {int(k): v for k, v in doc["summaries"].items()}


def _load_turn_lists(args) -> dict[int, list[int]]:
    jw = _jw()
    p = jw._paths(
        argparse.Namespace(out_root=Path(args.judge_root), smoke=args.smoke, dry_run=False)
    )
    return jw._load_ctx_lists(p)


def phase_embed(args) -> None:
    out = _out_dir(args)
    jw = _jw()
    ctx_desc = _load_ctx_descriptions(args)
    ans_desc = _load_ans_descriptions()
    summaries = _load_summaries(args)
    turn_lists = _load_turn_lists(args)
    if args.smoke:
        ctx_desc = dict(list(ctx_desc.items())[:16])
        ans_desc = dict(list(ans_desc.items())[:16])
        summaries = dict(list(summaries.items())[:4])
    model, tok = _load_embedder(args)
    ctx_ids = np.asarray(sorted(ctx_desc), np.int64)
    ans_ids = np.asarray(sorted(ans_desc), np.int64)
    emb_ctx = _embed_texts(args, model, tok, [ctx_desc[i] for i in ctx_ids])
    emb_ans = _embed_texts(args, model, tok, [ans_desc[i] for i in ans_ids])
    savez_atomic(out / "emb_ctx_desc.npz", feat_ids=ctx_ids, emb=emb_ctx.astype(np.float16))
    savez_atomic(out / "emb_ans_desc.npz", feat_ids=ans_ids, emb=emb_ans.astype(np.float16))
    fields = list(jw.APP_D_FIELDS)
    row_ids = np.asarray(sorted(summaries), np.int64)
    flat_texts: list[str] = []
    flat_index: list[tuple[int, int]] = []
    for r in row_ids:
        s = summaries[int(r)]
        for fi, f in enumerate(fields):
            if f in s:
                flat_texts.append(f"{f}: {s[f]}")
                flat_index.append((int(r), fi))
    emb_fields = _embed_texts(args, model, tok, flat_texts)
    savez_atomic(
        out / "emb_summary_fields.npz",
        row_ids=np.asarray([r for r, _f in flat_index], np.int64),
        field_idx=np.asarray([f for _r, f in flat_index], np.int64),
        fields=np.asarray(fields),
        emb=emb_fields.astype(np.float16),
    )
    # Der coverage: per (turn, field), mean cosine of the top-k most similar
    # listed ctx-feature descriptions (embeddings are L2-normalized -> dot)
    pos_of = {int(f): i for i, f in enumerate(ctx_ids)}
    per_field: dict[str, dict[str, list[float]]] = {
        f: {str(k): [] for k in TOPK_COVERAGE} for f in fields
    }
    n_no_desc = 0
    for (r, fi), e in zip(flat_index, emb_fields, strict=True):
        feats = [pos_of[f] for f in turn_lists.get(int(r), []) if f in pos_of]
        if not feats:
            n_no_desc += 1
            continue
        sims = np.sort(emb_ctx[feats] @ e)[::-1]
        for k in TOPK_COVERAGE:
            per_field[fields[fi]][str(k)].append(float(sims[: min(k, len(sims))].mean()))
    coverage = {
        "metric": "per field, mean cosine of the top-k most similar ctx feature "
        "descriptions from the turn's judged list (Der coverage)",
        "k_values": list(TOPK_COVERAGE),
        "n_turns": int(len(row_ids)),
        "n_field_cells_without_listed_descriptions": n_no_desc,
        "per_field_mean": {
            f: {k: (float(np.mean(v)) if v else None) for k, v in kk.items()}
            for f, kk in per_field.items()
        },
        "overall_mean": {
            str(k): float(np.mean([x for f in fields for x in per_field[f][str(k)]]))
            if any(per_field[f][str(k)] for f in fields)
            else None
            for k in TOPK_COVERAGE
        },
        "emb_model": args.emb_model,
        "tiny_model": bool(args.tiny_model or args.smoke),
        **as_metadata_dict(git_provenance(), phase="embed-coverage"),
    }
    cov_path = (
        EVAL_DIR / "embedding_coverage.json" if not args.smoke else out / "embedding_coverage.json"
    )
    cov_path.parent.mkdir(parents=True, exist_ok=True)
    _u2661().C.write_json_atomic(cov_path, coverage)
    print(
        f"[embed] unit done: ctx={len(ctx_ids)} ans={len(ans_ids)} "
        f"fields={len(flat_texts)} overall={coverage['overall_mean']}",
        flush=True,
    )


# ── dashboard ─────────────────────────────────────────────────────────────────────


def _find_result(root: Path, name: str) -> Path | None:
    """Resolve a pod artifact under either layout: the FLAT eval_results copies
    (upload phase) or the pod out-root's phase subdirs."""
    for cand in (
        root / name,
        root / "sae_metrics" / name,
        root / "perfeature" / name,
        root / "edges" / name,
        root / "encodes" / name,
        root / "map_ridge" / name,
        root / "map_mlp" / name,
        root / "controls" / name,
        root / "mining" / name,
    ):
        if cand.exists():
            return cand
    return None


def _esc(s: object) -> str:
    return html.escape(str(s))


def _png_b64(fig) -> str:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    import matplotlib.pyplot as plt

    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _metric_table_html(args) -> str:
    rows = []
    for side, label in (
        ("ctx", "Context SAE (fresh, #2661)"),
        ("answer", "Answer SAE (reused #2552)"),
    ):
        p = _find_result(Path(args.results_root), f"sae_metrics_{side}.json")
        if p is None:
            p = Path(args.results_root) / f"sae_metrics_{side}.json"
        if not p.exists():
            rows.append(
                f"<tr><td>{_esc(label)}</td><td colspan=6>missing ({_esc(p.name)})</td></tr>"
            )
            continue
        doc = json.loads(p.read_text())
        h = doc["splits"]["holdout"]
        rows.append(
            "<tr>"
            f"<td>{_esc(label)}</td>"
            f"<td>{h['nmse_raw']:.4f}</td>"
            f"<td>{h['nmse_mean_centered']:.4f}</td>"
            f"<td>{h['variance_fve']:.4f}</td>"
            f"<td>{h['realized_l0']:.1f}</td>"
            f"<td>{doc['dead_features']['n_dead_on_fit_rows']:,}</td>"
            f"<td>{doc['dead_features']['n_dead_on_holdout']:,}</td>"
            "</tr>"
        )
    der = (
        "<tr class=muted><td>Der et al. (paper reference)</td>"
        "<td>0.097 (turn-averaged) / 0.162 (per-token)</td>"
        "<td colspan=5>arXiv 2606.28548 reported nMSE; our raw nMSE is the "
        "comparable number</td></tr>"
    )
    return (
        "<table><tr><th>Dictionary</th><th>nMSE (raw — Der's metric)</th>"
        "<th>nMSE (mean-centered)</th><th>Variance FVE</th><th>Realized L0</th>"
        "<th>Dead on 120k fit rows</th><th>Dead on 20k holdout</th></tr>"
        + "".join(rows)
        + der
        + "</table>"
    )


def _perfeature_figures_html(args) -> str:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    npz_path = _find_result(Path(args.results_root), "perfeature_reads.npz") or (
        Path(args.results_root) / "perfeature_reads.npz"
    )
    if not npz_path.exists():
        return f"<p class=muted>perfeature_reads.npz missing at {_esc(npz_path)}</p>"
    z = np.load(npz_path)
    counts = np.asarray(z["counts_fit"], np.float64)
    routes = sorted({k.removeprefix("r2_") for k in z.files if k.startswith("r2_")})
    parts = []
    for metric, lab in (("r2", "held-out R^2 (unconditional)"), ("auroc", "firing AUROC")):
        fig, axes = plt.subplots(
            1, len(routes), figsize=(3.2 * len(routes), 3.0), sharey=True, sharex=True
        )
        axes = np.atleast_1d(axes)
        for ax, route in zip(axes, routes, strict=True):
            v = np.asarray(z[f"{metric}_{route}"], np.float64)
            m = np.isfinite(v) & (counts > 0)
            ax.scatter(counts[m], v[m], s=2, alpha=0.15, rasterized=True)
            ax.set_xscale("log")
            ax.set_title(route.replace("_", " "), fontsize=9)
            ax.set_xlabel("fit-row firing count")
        axes[0].set_ylabel(lab)
        fig.suptitle(f"per answer feature: {lab} vs firing count, by route", fontsize=10)
        parts.append(
            f'<img src="data:image/png;base64,{_png_b64(fig)}" alt="{_esc(lab)} by route">'
        )
    return "\n".join(parts)


def _edge_rows(args, ctx_desc, ans_desc, emb) -> list[dict]:
    tp_path = _find_result(Path(args.results_root), "top_pairs.json")
    assert tp_path is not None, f"top_pairs.json not found under {args.results_root}"
    doc = json.loads(tp_path.read_text())
    pairs = doc["pairs"]
    if emb is not None:
        ce, ae = emb
        cpos = {int(f): i for i, f in enumerate(ce["feat_ids"])}
        apos = {int(f): i for i, f in enumerate(ae["feat_ids"])}
    for pr in pairs:
        ci_, ai_ = int(pr["ctx_feat_id"]), int(pr["ans_feat_id"])
        pr["ctx_desc"] = ctx_desc.get(ci_, "(no description mined/judged)")
        pr["ans_desc"] = ans_desc.get(ai_, "(no #2552 description)")
        cos = None
        if emb is not None and ci_ in cpos and ai_ in apos:
            a = np.asarray(ce["emb"][cpos[ci_]], np.float32)
            b = np.asarray(ae["emb"][apos[ai_]], np.float32)
            cos = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-30))
        pr["desc_cosine"] = cos
        pr["unexpectedness"] = abs(pr["coef_std_units"]) * (1.0 - cos) if cos is not None else None
    return pairs


def _edge_table_html(pairs: list[dict], *, sort_key: str, top: int, title: str, blurb: str) -> str:
    rows = [p for p in pairs if p.get(sort_key) is not None]
    rows = sorted(rows, key=lambda p: -abs(p[sort_key]))[:top]
    body = []
    for p in rows:
        cos = p.get("desc_cosine")
        body.append(
            "<tr>"
            f"<td>{_esc(p['ctx_desc'])} <span class=muted>[ctx {p['ctx_feat_id']}]</span></td>"
            f"<td>{_esc(p['ans_desc'])} <span class=muted>[ans {p['ans_feat_id']}]</span></td>"
            f"<td>{p['coef_std_units']:+.4f}</td>"
            f"<td>{p['coef_half_a']:+.4f} / {p['coef_half_b']:+.4f}</td>"
            f"<td>{p['null_z']:.1f}</td>"
            f"<td>{cos:.3f}</td>"
            if cos is not None
            else "<tr>"
            f"<td>{_esc(p['ctx_desc'])} <span class=muted>[ctx {p['ctx_feat_id']}]</span></td>"
            f"<td>{_esc(p['ans_desc'])} <span class=muted>[ans {p['ans_feat_id']}]</span></td>"
            f"<td>{p['coef_std_units']:+.4f}</td>"
            f"<td>{p['coef_half_a']:+.4f} / {p['coef_half_b']:+.4f}</td>"
            f"<td>{p['null_z']:.1f}</td>"
            "<td>n/a</td>"
        )
        body[-1] += "</tr>"
    return (
        f"<h2>{_esc(title)}</h2><p>{_esc(blurb)}</p>"
        "<table><tr><th>User-message feature (what the prompt is about)</th>"
        "<th>Answer feature (what the reply does)</th><th>Coefficient (std units)</th>"
        "<th>Split-half a / b</th><th>Null z</th><th>Label cosine</th></tr>"
        + "".join(body)
        + "</table>"
    )


def _receipts_html(args, pairs: list[dict], ans_desc) -> str:
    rp = _find_result(Path(args.results_root), "receipts_answer_features.json") or (
        Path(args.results_root) / "receipts_answer_features.json"
    )
    if not rp.exists():
        return f"<p class=muted>receipts_answer_features.json missing at {_esc(rp)}</p>"
    doc = json.loads(rp.read_text())
    by_ans: dict[int, list[dict]] = {}
    for p in pairs:
        by_ans.setdefault(int(p["ans_feat_id"]), []).append(p)
    parts = []
    for fam, spec in doc["families"].items():
        rows = []
        for aid in spec["feature_ids"]:
            hit = by_ans.get(int(aid), [])
            strongest = max(hit, key=lambda p: abs(p["coef_std_units"])) if hit else None
            rows.append(
                "<tr>"
                f"<td>{_esc(ans_desc.get(int(aid), ''))} "
                f"<span class=muted>[ans {aid}]</span></td>"
                f"<td>{len(hit)}</td>"
                + (
                    f"<td>{_esc(strongest['ctx_desc'])} "
                    f"<span class=muted>[ctx {strongest['ctx_feat_id']}]</span> "
                    f"({strongest['coef_std_units']:+.4f})</td>"
                    if strongest
                    else "<td class=muted>no surviving in-edge</td>"
                )
                + "</tr>"
            )
        parts.append(
            f"<h3>{_esc(fam.replace('_', ' '))} ({spec['n_features']} answer features)</h3>"
            "<table><tr><th>Answer feature</th><th># surviving in-edges</th>"
            "<th>Strongest surviving context in-edge</th></tr>" + "".join(rows) + "</table>"
        )
    return "\n".join(parts)


def phase_dashboard(args) -> None:
    jw = _jw()
    ctx_desc = {}
    try:
        ctx_desc = _load_ctx_descriptions(args)
    except AssertionError as e:
        logger.warning("[dashboard] %s — ctx labels will read as undescribed", e)
    ans_desc = _load_ans_descriptions()
    emb = None
    emb_dir = _out_dir(args)
    if (emb_dir / "emb_ctx_desc.npz").exists() and (emb_dir / "emb_ans_desc.npz").exists():
        emb = (np.load(emb_dir / "emb_ctx_desc.npz"), np.load(emb_dir / "emb_ans_desc.npz"))
    else:
        logger.warning("[dashboard] embeddings missing — unexpected-edge section degrades")
    pairs = _edge_rows(args, ctx_desc, ans_desc, emb)
    est = {}
    if jw.ESTIMATE_PATH.exists():
        est = json.loads(jw.ESTIMATE_PATH.read_text())
    pins = json.loads(jw.REGIME_PINS.read_text()) if jw.REGIME_PINS.exists() else {}
    prov = as_metadata_dict(git_provenance(), phase="dashboard")
    cov_p = EVAL_DIR / "embedding_coverage.json"
    cov_html = ""
    if cov_p.exists():
        cov = json.loads(cov_p.read_text())
        cov_html = (
            "<p>Embedding coverage (mean cosine, top-k listed descriptions vs "
            "summary field): "
            + ", ".join(
                f"k={k}: {v:.3f}" if v is not None else f"k={k}: n/a"
                for k, v in cov["overall_mean"].items()
            )
            + "</p>"
        )
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>#2661 — context-feature to answer-feature map</title>
<style>
body{{font-family:system-ui,sans-serif;margin:24px;max-width:1400px}}
table{{border-collapse:collapse;margin:12px 0;font-size:13px}}
td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left;vertical-align:top}}
th{{background:#f0f0f0}} .muted{{color:#999;font-size:11px}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px}} h3{{font-size:14px}}
footer{{margin-top:32px;color:#666;font-size:12px;border-top:1px solid #ccc;padding-top:8px}}
</style></head><body>
<h1>Flat context SAE + full-dictionary feature map (task #2661)</h1>
<p>A flat Der-recipe SAE on what the user asked (last prompt token, layer 19),
mapped feature-to-feature onto the reused #2552 answer SAE. Edges below survive
split-half replication and a label-shuffle null.</p>
<h2>1 — SAE reconstruction metrics (both dictionaries)</h2>
{_metric_table_html(args)}
<h2>2 — Per-feature predictability vs firing count, by route</h2>
{_perfeature_figures_html(args)}
{
        _edge_table_html(
            pairs,
            sort_key="null_z",
            top=int(args.max_rows),
            title="3 — Surviving edges (strongest first)",
            blurb="Both sides in words; feature ids in the muted suffix. "
            "Coefficient is per 1 SD of the context feature.",
        )
    }
<h2>4 — Receipts: behavior answer features</h2>
{_receipts_html(args, pairs, ans_desc)}
<h2>4b — Context-side watch list (China/Taiwan/Xinjiang/Tibet/CCP topics)</h2>
{_watchlist_html(args)}
{
        _edge_table_html(
            pairs,
            sort_key="unexpectedness",
            top=int(args.max_rows),
            title="5 — Unexpected edges (strong edge, semantically distant labels)",
            blurb="Ranked by |coefficient| x (1 - cosine between the two "
            "descriptions' embeddings): strong edges whose labels do not "
            "obviously go together.",
        )
    }
<h2>6 — Provenance</h2>
{cov_html}
<footer>
git commit {_esc(prov.get("git_commit"))} · capture chunks @
{_esc(pins.get("capture_chunks_revision", "?")[:12])} · #2552 artifacts @
{_esc(pins.get("issue2552_revision", "?")[:12])} · judge {_esc(jw.JUDGE_MODEL)} ·
judge estimate: {_esc(json.dumps(est.get("total", {})))} · generated
{time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime())}
</footer></body></html>"""
    out_path = (
        EVAL_DIR / "dashboard" / "issue2661_dashboard.html"
        if not args.smoke
        else _out_dir(args) / "issue2661_dashboard.html"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(doc)
    print(
        f"[dashboard] unit written: {out_path} ({out_path.stat().st_size} bytes, "
        f"{len(pairs)} edges)",
        flush=True,
    )


def phase_watchlist(args) -> None:
    """Context-side watch list (review r1 Minor 2; task-body receipts paragraph):
    regex over the judged W1 ctx descriptions -> each hit's top out-edges from
    wiring_edges.npz with answer-side descriptions in words, flagging edges that
    land on a receipts answer feature. Zero pod/judge spend."""
    import re

    ctx_desc = _load_ctx_descriptions(args)
    ans_desc = _load_ans_descriptions()
    wz_path = _find_result(Path(args.results_root), "wiring_edges.npz")
    assert wz_path is not None, f"wiring_edges.npz not found under {args.results_root}"
    wz = np.load(wz_path)
    out_ids = np.asarray(wz["out_edge_ids"], np.int64)
    out_coefs = np.asarray(wz["out_edge_coefs"], np.float32)
    ctx_live = np.asarray(wz["out_edge_ctx_ids"], np.int64)
    pos_of_live = {int(f): i for i, f in enumerate(ctx_live)}
    rp = _find_result(Path(args.results_root), "receipts_answer_features.json")
    receipt_of: dict[int, list[str]] = {}
    if rp is not None:
        rdoc = json.loads(rp.read_text())
        for fam, spec in rdoc["families"].items():
            for aid in spec["feature_ids"]:
                receipt_of.setdefault(int(aid), []).append(fam)
    rx = re.compile(WATCHLIST_PATTERN, re.IGNORECASE)
    hits = {fid: d for fid, d in ctx_desc.items() if rx.search(d)}
    rows = []
    n_unencoded = 0
    for fid in sorted(hits):
        if fid not in pos_of_live:
            n_unencoded += 1
            continue
        i = pos_of_live[fid]
        edges = []
        for j in range(min(WATCH_TOP_OUT_EDGES, out_ids.shape[1])):
            aid = int(out_ids[i, j])
            edges.append(
                {
                    "ans_feat_id": aid,
                    "ans_desc": ans_desc.get(aid, "(no #2552 description)"),
                    "coef_std_units": float(out_coefs[i, j]),
                    "receipts_families": receipt_of.get(aid, []),
                }
            )
        rows.append(
            {
                "ctx_feat_id": int(fid),
                "ctx_desc": hits[fid],
                "n_receipts_edges": sum(1 for e in edges if e["receipts_families"]),
                "top_out_edges": edges,
            }
        )
    rows.sort(key=lambda r: -r["n_receipts_edges"])
    doc = {
        "pattern": WATCHLIST_PATTERN,
        "n_ctx_descriptions_scanned": len(ctx_desc),
        "n_watchlist_features": len(hits),
        "n_dropped_not_in_live_edge_rows": n_unencoded,
        "top_out_edges_per_feature": WATCH_TOP_OUT_EDGES,
        "note": "coefficients are standardized ridge units (per 1 SD of the context "
        "feature); receipts_families flags edges landing on refusal / CCP-position / "
        "Qwen-identity / sycophancy / harmful-content answer features",
        "features": rows,
        **as_metadata_dict(git_provenance(), phase="watchlist"),
    }
    out_path = (
        EVAL_DIR / "watchlist_context_features.json"
        if not args.smoke
        else _out_dir(args) / "watchlist_context_features.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    _u2661().C.write_json_atomic(out_path, doc)
    print(
        f"[watchlist] unit done: {len(hits)} watch features, "
        f"{sum(r['n_receipts_edges'] for r in rows)} receipts-flagged edges -> {out_path}",
        flush=True,
    )


def _watchlist_html(args) -> str:
    for cand in (
        EVAL_DIR / "watchlist_context_features.json",
        _out_dir(args) / "watchlist_context_features.json",
    ):
        if cand.exists():
            doc = json.loads(cand.read_text())
            break
    else:
        return "<p class=muted>watchlist_context_features.json missing — run --phase watchlist</p>"
    parts = [
        f"<p>{doc['n_watchlist_features']} context features matched the watch pattern "
        f"(China / Taiwan / Xinjiang / Tibet / CCP / Uyghur / Hong Kong).</p>",
        "<table><tr><th>User-message feature (watch topic)</th>"
        "<th>Receipts-flagged out-edges</th><th>Strongest out-edges (answer side)</th></tr>",
    ]
    for r in doc["features"][:100]:
        tops = "; ".join(
            f"{_esc(e['ans_desc'])} <span class=muted>[ans {e['ans_feat_id']}]</span> "
            f"({e['coef_std_units']:+.3f}"
            + (f", {'/'.join(e['receipts_families'])}" if e["receipts_families"] else "")
            + ")"
            for e in r["top_out_edges"][:3]
        )
        parts.append(
            "<tr>"
            f"<td>{_esc(r['ctx_desc'])} <span class=muted>[ctx {r['ctx_feat_id']}]</span></td>"
            f"<td>{r['n_receipts_edges']}</td><td>{tops}</td></tr>"
        )
    parts.append("</table>")
    return "\n".join(parts)


PHASES = {"embed": phase_embed, "watchlist": phase_watchlist, "dashboard": phase_dashboard}


def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=[*PHASES, "all"], default="all")
    ap.add_argument("--out-root", type=Path, default=PROJECT_ROOT / "data" / "issue_2661" / "embed")
    ap.add_argument(
        "--judge-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "issue_2661" / "judge",
        help="judge work root (for the staged eval lists)",
    )
    ap.add_argument(
        "--judge-agg",
        type=Path,
        default=AGG_DIR,
        help="judge aggregates dir (descriptions_ctx / summaries)",
    )
    ap.add_argument(
        "--results-root",
        type=Path,
        default=EVAL_DIR,
        help="where the pod-harvested JSONs + perfeature npz live",
    )
    ap.add_argument("--emb-model", default="Qwen/Qwen3-Embedding-8B")
    ap.add_argument("--emb-batch", type=int, default=64)
    ap.add_argument("--emb-max-tokens", type=int, default=512)
    ap.add_argument("--max-rows", type=int, default=200, help="edge-table row cap")
    ap.add_argument("--device", default="auto", choices=["auto", "cuda", "cpu"])
    ap.add_argument("--smoke", action="store_true", help="tiny twin embedder + sliced inputs")
    ap.add_argument("--tiny-model", action="store_true", help="from-config embedder twin")
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--list-phases", action="store_true")
    return ap


def main() -> None:
    args = build_argparser().parse_args()
    if args.list_phases:
        print(json.dumps({"registry": [*sorted(PHASES), "all"]}))
        raise SystemExit(0)
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        import matplotlib  # noqa: F401
        from transformers import AutoConfig, AutoModel, AutoTokenizer  # noqa: F401

        _u2661()
        _jw()
        print("[import-check] OK", flush=True)
        raise SystemExit(0)
    if args.device == "auto":
        import torch

        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    seq = list(PHASES) if args.phase == "all" else [args.phase]
    for name in seq:
        PHASES[name](args)
    print("[phase=done]", flush=True)


if __name__ == "__main__":
    main()
