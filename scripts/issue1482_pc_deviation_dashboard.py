"""Dashboard #2: top/bottom-50 answer-PCA PCs by LOCAL-NEIGHBOR deviation.

Expected R^2 for a PC of a given variance = the MEDIAN held-out R^2 of its
rank-neighbors (window +/-12 in variance-rank space, self excluded) — a
nonparametric local baseline instead of the fitted two-parameter floor curve.
Deviation = R^2 - local median (ridge, context arm); over-performers (top 50)
and under-performers (bottom 50) get the full interpretation battery: nearest
SAE feature (+ own #1773 autointerp description), logit / tuned / J-lens.

Selection restricted to variance ranks 12..1011 (window fully interior, and
beyond ~rank 1000 R^2 hugs zero so local deviations are estimate noise).

Outputs:
  eval_results/issue_1482/pc_dashboard/pc_deviation_dashboard.json
  dashboard/public/pc-deviation-1482.html
0 GPU; banked/staged arrays + cached weights only.
"""

from __future__ import annotations

import html
import json
import sys

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM run)

import numpy as np  # noqa: E402

from explore_persona_space.task_workflow import repo_root  # noqa: E402

PROJECT_ROOT = repo_root()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1482_residual_svd as RS  # noqa: E402
from issue1482_footprint_moments import _load_gamma  # noqa: E402
from issue1482_pc_dashboard import JLENS, TUNED, _own_descriptions  # noqa: E402

OUT_JSON = "eval_results/issue_1482/pc_dashboard/pc_deviation_dashboard.json"
OUT_HTML = "dashboard/public/pc-deviation-1482.html"
HALF_WIN = 12
RANK_MAX = 1024  # beyond ~r1000 R^2 hugs zero; local deviations there are noise
N_GROUP = 50
N_TOK = 8


def _local_median_dev(r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per rank i: median R^2 of ranks [i-HALF_WIN, i+HALF_WIN] excluding i."""
    n = len(r)
    med = np.full(n, np.nan)
    for i in range(HALF_WIN, min(n, RANK_MAX) - HALF_WIN):
        w = np.concatenate([r[i - HALF_WIN : i], r[i + 1 : i + 1 + HALF_WIN]])
        med[i] = np.median(w)
    return med, r - med


def main() -> None:
    import torch

    y16, ci = RS.load_layer(19)
    Y = np.asarray(y16, dtype=np.float64)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    _lam, vecs = RS.gram_spectrum(Yc, want_vectors=True, n_vec=3584)
    ss_tot = np.square(Yc @ vecs).sum(axis=0)
    share = ss_tot / np.square(Yc).sum()

    r2 = {}
    for fitter in ("ridge", "mlp_w8192"):
        pred16 = RS.load_pred("context", 19, fitter, ci)
        E = Y - np.asarray(pred16, dtype=np.float64)
        r2[fitter] = 1.0 - np.square(E @ vecs).sum(axis=0) / ss_tot

    med, dev = _local_median_dev(r2["ridge"])
    valid = np.flatnonzero(np.isfinite(dev))
    order = valid[np.argsort(dev[valid])]
    under = order[:N_GROUP]  # most below the neighbor expectation
    over = order[-N_GROUP:][::-1]  # most above
    pcs = list(over) + list(under)
    V = vecs[:, pcs].astype(np.float32)

    from issue1482_sae import BatchTopKSAE

    sae = BatchTopKSAE.load(k=64, layer=19, device="cpu")
    D = np.asarray(sae.w_dec, dtype=np.float32)
    D_unit = D / np.linalg.norm(D, axis=0, keepdims=True)
    cos = D_unit.T @ V
    argmax = np.argmax(np.abs(cos), axis=0)
    maxcos = np.abs(cos)[argmax, np.arange(len(pcs))]

    w_u, tok = RS.load_unembedding()
    gamma = _load_gamma()
    J = (
        torch.load(PROJECT_ROOT / JLENS, map_location="cpu", weights_only=False)["J"][19]
        .to(torch.float32)
        .numpy()
    )
    Wt = (
        torch.load(PROJECT_ROOT / TUNED, map_location="cpu", weights_only=False)["W"]
        .numpy()
        .astype(np.float32)
    )
    lenses = {"logit_lens": None, "jlens": J, "tuned_lens": Wt}
    lens_toks: dict[str, list[list[str]]] = {}
    for name, T in lenses.items():
        tv = V if T is None else T @ V
        e = w_u @ (tv * gamma[:, None])
        lens_toks[name] = [
            [tok.decode([int(t)]) for t in np.argsort(e[:, j])[-N_TOK:][::-1]]
            for j in range(len(pcs))
        ]

    own_desc = _own_descriptions({int(f) for f in argmax})

    rows = []
    for j, pc in enumerate(pcs):
        fid = int(argmax[j])
        rows.append(
            {
                "group": "over_performers" if j < N_GROUP else "under_performers",
                "pc": int(pc),
                "variance_share": float(share[pc]),
                "r2_ridge": float(r2["ridge"][pc]),
                "r2_mlp": float(r2["mlp_w8192"][pc]),
                "neighbor_median_r2": float(med[pc]),
                "deviation": float(dev[pc]),
                "sae_feat": fid,
                "abs_cos": float(maxcos[j]),
                "own_desc": own_desc.get(fid, ""),
                "logit_lens": lens_toks["logit_lens"][j],
                "tuned_lens": lens_toks["tuned_lens"][j],
                "jlens": lens_toks["jlens"][j],
            }
        )

    out_json = PROJECT_ROOT / OUT_JSON
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps({"window": HALF_WIN, "rank_max": RANK_MAX, "rows": rows}, indent=1)
    )

    def esc(s: str) -> str:
        return html.escape(s if s else "—")

    def tokcell(ts: list[str]) -> str:
        return " ".join(f'<span class="tk">{html.escape(t.strip() or "␣")}</span>' for t in ts)

    def table(group: str) -> str:
        body = []
        for r_ in rows:
            if r_["group"] != group:
                continue
            dv = r_["deviation"]
            body.append(
                f'<tr><td class="pc">PC{r_["pc"]}</td>'
                f"<td>{r_['variance_share']:.2e}</td>"
                f"<td>{r_['r2_ridge']:.3f}</td>"
                f"<td>{r_['neighbor_median_r2']:.3f}</td>"
                f'<td class="{"good" if dv > 0 else "bad"}">{dv:+.3f}</td>'
                f"<td>{r_['r2_mlp']:.3f}</td>"
                f"<td>{r_['sae_feat']}</td>"
                f"<td>{r_['abs_cos']:.2f}</td>"
                f'<td class="desc">{esc(r_["own_desc"])}</td>'
                f"<td>{tokcell(r_['logit_lens'])}</td>"
                f"<td>{tokcell(r_['tuned_lens'])}</td>"
                f"<td>{tokcell(r_['jlens'])}</td></tr>"
            )
        return "\n".join(body)

    head = (
        "<tr><th>PC</th><th>var share</th><th>R² ridge</th><th>neighbor median R²</th>"
        "<th>Δ vs neighbors</th><th>R² MLP</th><th>SAE feat</th><th>|cos|</th>"
        "<th>feature description (own #1773 autointerp)</th>"
        "<th>logit lens</th><th>tuned lens</th><th>J-lens</th></tr>"
    )
    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Answer-PCA PCs by neighbor-deviation (#1482)</title>
<style>
body{{font-family:Inter,system-ui,sans-serif;margin:24px;color:#1a1f36;background:#fafbfc}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px}}
p.meta{{color:#556;max-width:1100px;font-size:13px;line-height:1.5}}
table{{border-collapse:collapse;font-size:12px;width:100%;background:#fff}}
th,td{{border:1px solid #e3e8ee;padding:5px 7px;vertical-align:top;text-align:left}}
th{{background:#f1f4f8;position:sticky;top:0}}
td.pc{{font-weight:600;white-space:nowrap}}
td.good{{background:#e7f6ec;font-weight:600}} td.bad{{background:#fdeaea;font-weight:600}}
td.desc{{max-width:230px}}
span.tk{{display:inline-block;background:#eef1f6;border-radius:3px;padding:0 4px;margin:1px;font-family:ui-monospace,monospace;font-size:11px}}
</style></head><body>
<h1>Answer-PCA directions: over/under-performers vs their variance-rank neighbors (issue #1482)</h1>
<p class="meta">Companion to <a href="pc-lens-1482.html">pc-lens-1482.html</a> (which sorts by
variance). Here the expected R² for a PC of a given variance is the <b>median R² of its
±{HALF_WIN} rank-neighbors</b> (self excluded) — a nonparametric local baseline instead of the
fitted floor curve — and PCs are ranked by <b>Δ = R² − neighbor median</b> (ridge, context arm,
L19, multi-turn holdout). Top 50 over-performers and top 50 under-performers among variance
ranks {HALF_WIN}–{RANK_MAX - HALF_WIN - 1} (beyond ~rank 1000 R² hugs zero and local deviations
are estimate noise). Because the baseline is local, this surfaces <b>pointwise</b> anomalies;
the smooth regional arch the fitted-curve analysis found is absorbed into the baseline by
construction. Interpretation columns as in the variance dashboard: nearest SAE feature over the
131,072-column decoder (random-direction null max ≈ 0.05), our #1773 autointerp description,
top-8 promoted tokens under logit/tuned/J-lens.
Data: eval_results/issue_1482/pc_dashboard/pc_deviation_dashboard.json.</p>
<h2>Top 50 over-performers (predicted better than neighbors of the same variance)</h2>
<table><thead>{head}</thead><tbody>{table("over_performers")}</tbody></table>
<h2>Top 50 under-performers (predicted worse than neighbors of the same variance)</h2>
<table><thead>{head}</thead><tbody>{table("under_performers")}</tbody></table>
</body></html>"""
    out_html = PROJECT_ROOT / OUT_HTML
    out_html.write_text(page)
    print(f"[out] {out_json}")
    print(f"[out] {out_html} ({len(page) / 1024:.0f} KB, {len(rows)} rows)")
    print(
        f"deviation range: over [{dev[over[-1]]:+.3f}, {dev[over[0]]:+.3f}] "
        f"under [{dev[under[0]]:+.3f}, {dev[under[-1]]:+.3f}]"
    )


if __name__ == "__main__":
    main()
