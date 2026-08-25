#!/usr/bin/env python3
"""Issue #2202 — context-space nearest neighbours of a failure-table row.

User-chat inline free analysis (2026-08-25, round ctxnn-r1, 0 GPU-h).

Why. In the paper's 11-row failure table each row carries a CONTEXT RANK: where
the retrieved conversation's context sits in the failing context's own neighbour
ordering (mean-removed context-vector cosine). The Kubernetes row (ci 18697,
answer rank 9) has context rank 73 — 72 held-out contexts are closer to it in
context space than the context whose answer the map actually retrieved. This
script emits those neighbours as a browsable dashboard so the near-duplicate
reading can be inspected row by row instead of taken on the aggregate.

The two answer-side columns are the point of the table. A neighbour that is
close in CONTEXT but far down the query's ANSWER ranking separates "the pool
holds a near-duplicate query" from "the answers themselves are confusable".

Reads (all already staged, read-only; no download):
  /mnt/eps-data/.../issue2202_avgtgt/cx_holdout_L19.npz        context vectors
  /mnt/eps-data/.../issue2202_freshwhiten/{pred16,y_holdout_L19,whiten_stats}.npz
  /mnt/eps-data/.../issue2202_freshwhiten/kresample/           fresh answer draws
  eval_results/issue_2202/plot5_redesign/oppoint_margins.npz   ranks + competitors
  data/issue_1482/context_extremes_scratch/judge_texts.jsonl   corpus text (REUSED)

Scoring is the banked OPERATING POINT, reused verbatim from
``issue2202_plot5_failchar_tail``: whitened cosine on draw-averaged targets plus
the CSLS (k=10) query-bank penalty. The recomputed rank for the query context is
reconciliation-gated against the banked ``rank_avg``.

Every excerpt is capped through ``issue2202_labels.cap_text``, which appends the
inline "…[truncated]" disclosure; no other alteration of model-generated text.

Writes dashboard/public/ctxnn-2202.html + eval_results/issue_2202/ctxnn/<ci>.json
(text-free: ids, ranks and similarities only).
"""

from __future__ import annotations

import argparse
import html
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue2202_failchar as FC  # noqa: E402  (LAYER / meta_block / atomic_json)
import issue2202_labels as LB  # noqa: E402  (cap_text / load_texts / caps)
import issue2202_metric_zoo as MZ  # noqa: E402  (load_staged)
import numpy as np  # noqa: E402

STAGED = Path("/mnt/eps-data/thomasjiralerspong/issue2202_freshwhiten")
CX_NPZ = Path("/mnt/eps-data/thomasjiralerspong/issue2202_avgtgt/cx_holdout_L19.npz")
MARGINS_NPZ = PROJECT_ROOT / "eval_results/issue_2202/plot5_redesign/oppoint_margins.npz"
OUT_HTML = PROJECT_ROOT / "dashboard/public/ctxnn-2202.html"
OUT_EVAL = PROJECT_ROOT / "eval_results/issue_2202/ctxnn"
DEFAULT_CI = 18697  # the Kubernetes failure row
K_CSLS = 10
EXPECTED_N = 9941


def _unit(a: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(a, axis=1, keepdims=True)
    assert (n > 0).all(), "zero-norm vector"
    return a / n


def _wh(x: np.ndarray, ell: np.ndarray, mu_a: np.ndarray) -> np.ndarray:
    """Whitening in the banked Cholesky basis (issue2202_plot5_failchar_tail)."""
    from scipy.linalg import solve_triangular

    return solve_triangular(ell, (x - mu_a).T, lower=True).T


def operating_point_scores(query_pos: int) -> dict:
    """The banked operating-point score row for one query: whitened cosine on
    draw-averaged targets + CSLS(k=10). Returns the query's score row, the
    per-pool rank, and the whitened true-answer cosine row."""
    import issue1738_characterize as CH

    st = MZ.load_staged(STAGED)
    pred, y16, pci = st["pred"], st["y16"], st["pci"]
    ell, mu_a = st["stats"]["L"], st["stats"]["mu_A"]
    n_pool = y16.shape[0]

    kns = argparse.Namespace(
        local_kresample_dir=str(STAGED / "kresample"),
        scratch=str(STAGED / "scratch"),
        hf_prefix="",
    )
    kci, vres = CH._load_kresample_v(kns, [FC.LAYER])
    draws = vres[:, :, 0, :].astype(np.float64)
    k_draws = vres.shape[1]
    pos_of = {int(c): p for p, c in enumerate(pci.tolist())}
    pos = np.asarray([pos_of[int(c)] for c in kci], dtype=np.int64)

    # draw-averaged pool targets on the covered rows (the operating point)
    avg = (y16[pos] + draws.sum(axis=1)) / (1 + k_draws)
    pool_modw = _wh(y16, ell, mu_a)
    pool_modw[pos] = _wh(avg, ell, mu_a)
    pool_n = _unit(pool_modw)

    pwn = _unit(_wh(pred, ell, mu_a))
    s = pwn @ pool_n.T
    # CSLS penalty r_j: mean of the top-k query scores per POOL column
    pen = 0.5 * np.partition(s, n_pool - K_CSLS, axis=0)[n_pool - K_CSLS :, :].mean(axis=0)
    score_row = s[query_pos] - pen
    del s
    order = np.argsort(-score_row, kind="stable")
    ans_rank = np.empty(n_pool, dtype=np.int64)
    ans_rank[order] = np.arange(1, n_pool + 1)
    # answer-space proximity between the two TRUE answers, same whitened space
    true_cos = pool_n @ pool_n[query_pos]
    return {
        "score_row": score_row,
        "ans_rank": ans_rank,
        "true_answer_cos": true_cos,
        "pci": pci,
    }


CSS = """
body{font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,sans-serif;margin:18px;color:#111;}
h1{font-size:19px;margin:0 0 4px;} .sub{color:#555;font-size:13px;margin-bottom:14px;max-width:1100px;line-height:1.5;}
.qbox{border:1px solid #d7dbe3;background:#f8fafc;border-radius:6px;padding:10px 12px;margin-bottom:14px;max-width:1100px;}
.qbox h2{font-size:14px;margin:0 0 6px;} .qbox .lbl{color:#666;font-size:11px;text-transform:uppercase;letter-spacing:.04em;margin-top:8px;}
.qbox .txt{font-size:12px;white-space:pre-wrap;line-height:1.45;max-height:150px;overflow:auto;}
input#q{padding:6px 9px;width:340px;font-size:13px;margin-bottom:10px;border:1px solid #ccc;border-radius:4px;}
table{border-collapse:collapse;width:100%;font-size:12px;}
th,td{border-bottom:1px solid #e6e8ec;padding:6px 8px;vertical-align:top;}
th{position:sticky;top:0;background:#f4f6fa;text-align:left;font-weight:600;z-index:2;}
td.n,th.n{text-align:right;font-variant-numeric:tabular-nums;white-space:nowrap;}
tr.distractor{background:#fff3f2;} tr.distractor td{border-bottom:2px solid #c44e52;}
.tag{display:inline-block;background:#c44e52;color:#fff;font-size:10px;padding:1px 5px;border-radius:3px;margin-left:5px;}
.txt{max-height:130px;overflow:auto;white-space:pre-wrap;line-height:1.4;}
.hint{color:#777;font-size:11px;}
"""

JS = """
document.getElementById('q').addEventListener('input',function(e){
  var v=e.target.value.toLowerCase();
  document.querySelectorAll('tbody tr').forEach(function(tr){
    tr.style.display = tr.textContent.toLowerCase().includes(v) ? '' : 'none';
  });
});
"""


def esc(s) -> str:
    return html.escape(str(s if s is not None else ""))


def build_html(query_ci: int, distractor_ci: int, rows: list[dict], texts: dict, meta: dict) -> str:
    qt = texts[query_ci]
    head = (
        f"<div class='qbox'><h2>Query context &mdash; ci {query_ci} "
        f"(corpus {esc(qt.get('corpus'))}; answer rank {meta['query_answer_rank']}, "
        f"context rank {meta['distractor_context_rank']})</h2>"
        f"<div class='lbl'>history (tail)</div>"
        f"<div class='txt'>{esc(LB.cap_text(qt.get('history_tail', ''), LB.CAP_HISTORY))}</div>"
        f"<div class='lbl'>final user turn</div>"
        f"<div class='txt'>{esc(LB.cap_text(qt.get('last_user', ''), LB.CAP_LAST_USER))}</div>"
        f"<div class='lbl'>true answer</div>"
        f"<div class='txt'>{esc(LB.cap_text(qt.get('response', ''), LB.CAP_RESPONSE))}</div></div>"
    )
    body = []
    for r in rows:
        t = texts[r["ci"]]
        cls = " class='distractor'" if r["is_distractor"] else ""
        tag = "<span class='tag'>retrieved</span>" if r["is_distractor"] else ""
        body.append(
            f"<tr{cls}>"
            f"<td class='n'>{r['context_rank']}{tag}</td>"
            f"<td class='n'>{r['ci']}</td>"
            f"<td class='n'>{r['context_cos']:.3f}</td>"
            f"<td class='n'>{r['answer_rank']}</td>"
            f"<td class='n'>{r['true_answer_cos']:.3f}</td>"
            f"<td>{esc(t.get('corpus'))}</td>"
            f"<td><div class='txt'>{esc(LB.cap_text(t.get('last_user', ''), LB.CAP_LAST_USER))}</div></td>"
            f"<td><div class='txt'>{esc(LB.cap_text(t.get('response', ''), LB.CAP_RESPONSE))}</div></td>"
            f"</tr>"
        )
    sub = (
        f"The {meta['distractor_context_rank'] - 1} held-out contexts that sit closer to ci "
        f"{query_ci} in context space than the context whose answer the map actually retrieved "
        f"(ci {distractor_ci}, the last row). <b>Context rank</b> orders the {EXPECTED_N - 1} other "
        f"held-out contexts by mean-removed context-vector cosine. <b>Answer rank</b> is where that "
        f"context's answer landed in ci {query_ci}'s own retrieval ranking at the operating point "
        f"(whitened cosine + CSLS k=10, draw-averaged targets); the true answer sits at "
        f"{meta['query_answer_rank']}. <b>Answer cos</b> is the whitened cosine between the two true "
        f"answers. Excerpts are capped (history {LB.CAP_HISTORY} / user {LB.CAP_LAST_USER} / answer "
        f"{LB.CAP_RESPONSE} chars) with the truncation marked inline; text is otherwise verbatim."
    )
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        f"<title>Context neighbours of ci {query_ci} &mdash; #2202</title>"
        f"<style>{CSS}</style></head><body>"
        f"<h1>Context-space neighbours nearer than the retrieved context &mdash; issue #2202</h1>"
        f"<div class='sub'>{sub}</div>{head}"
        "<input id='q' placeholder='filter rows…'>"
        "<table><thead><tr>"
        "<th class='n'>Context<br>rank</th><th class='n'>ci</th><th class='n'>Context<br>cos</th>"
        "<th class='n'>Answer<br>rank</th><th class='n'>Answer<br>cos</th>"
        "<th>Corpus</th><th>Final user turn</th><th>Answer</th>"
        "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"
        f"<p class='hint'>Generated by scripts/issue2202_ctxnn_dashboard.py; "
        f"data eval_results/issue_2202/ctxnn/{query_ci}.json</p>"
        f"<script>{JS}</script></body></html>"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ci", type=int, default=DEFAULT_CI)
    ap.add_argument("--out-html", default=str(OUT_HTML))
    ap.add_argument("--text-cache", default=LB.DEFAULT_TEXT_CACHE)
    args = ap.parse_args()

    cz = np.load(CX_NPZ)
    cx = cz["cx"].astype(np.float32)
    cci = np.asarray(cz["ci"], dtype=np.int64)
    assert cx.shape[0] == EXPECTED_N, cx.shape

    mz = np.load(MARGINS_NPZ)
    ci_full = np.asarray(mz["ci_full"], dtype=np.int64)
    assert np.array_equal(ci_full, cci), "cx / margins ci misalign"
    pos_of = {int(c): p for p, c in enumerate(ci_full.tolist())}
    qpos = pos_of[args.ci]

    covered_ci = np.asarray(mz["ci_covered"], dtype=np.int64)
    cov_idx = int(np.flatnonzero(covered_ci == args.ci)[0])
    banked_rank = float(np.asarray(mz["rank_avg"])[cov_idx])
    distractor_ci = int(np.asarray(mz["competitor_ci_avg"], dtype=np.int64)[cov_idx])

    # ── context-space neighbour ordering (mean-removed, the reported convention)
    mu_c = np.load(STAGED / "whiten_stats.npz")["mu_C"].astype(np.float32)
    cxn = _unit(cx - mu_c[None, :])
    ctx_cos = (cxn @ cxn[qpos]).astype(np.float64)
    ctx_cos[qpos] = -np.inf
    dpos = pos_of[distractor_ci]
    ctx_rank = (ctx_cos > ctx_cos[dpos]).sum() + 1
    order = np.argsort(-ctx_cos, kind="stable")[:ctx_rank]
    assert int(order[-1]) == dpos, "distractor is not the last row of the neighbour prefix"

    # ── answer-side reads at the banked operating point
    ap_ = operating_point_scores(qpos)
    recomputed = float(ap_["ans_rank"][qpos])
    assert abs(recomputed - banked_rank) < 1.5, (
        f"operating-point reconciliation FAILED: recomputed {recomputed} vs banked {banked_rank}"
    )
    print(f"reconciliation OK: query answer rank {recomputed:.0f} == banked {banked_rank:.0f}")

    rows = []
    for r, p in enumerate(order.tolist(), start=1):
        rows.append(
            {
                "context_rank": r,
                "ci": int(ci_full[p]),
                "context_cos": float(ctx_cos[p]),
                "answer_rank": int(ap_["ans_rank"][p]),
                "true_answer_cos": float(ap_["true_answer_cos"][p]),
                "is_distractor": p == dpos,
            }
        )

    texts = LB.load_texts(Path(args.text_cache), {args.ci} | {r["ci"] for r in rows})
    meta = {
        "query_ci": args.ci,
        "distractor_ci": distractor_ci,
        "distractor_context_rank": int(ctx_rank),
        "query_answer_rank": int(banked_rank),
        "n_nearer_than_distractor": int(ctx_rank - 1),
        "convention_context": "cosine of context vectors after removing the banked mean mu_C",
        "convention_answer": (
            "operating point: whitened cosine on draw-averaged targets + CSLS k=10 "
            "(issue2202_plot5_failchar_tail recipe), reconciliation-gated on rank_avg"
        ),
    }
    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    (OUT_EVAL / f"{args.ci}.json").write_text(
        json.dumps({"meta": meta | MZ.meta_block(), "rows": rows}, indent=2)
    )
    out = Path(args.out_html)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_html(args.ci, distractor_ci, rows, texts, meta))
    print(f"{meta['n_nearer_than_distractor']} contexts nearer than the retrieved one")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    print(f"wrote {OUT_EVAL / f'{args.ci}.json'}")


if __name__ == "__main__":
    main()
