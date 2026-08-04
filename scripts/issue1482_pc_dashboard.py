"""Static HTML dashboard: top-50 and bottom-50 answer-PCA PCs BY VARIANCE with
every interpretation channel side by side.

Per PC: variance share, held-out R^2 (ridge + MLP, context arm), nearest SAE
feature (max |cos| over the 131,072-column decoder) with its description from
our own #1773 describe pipeline, and the
top promoted tokens under logit lens / tuned lens / J-lens.

Outputs:
  eval_results/issue_1482/pc_dashboard/pc_dashboard.json   (data)
  dashboard/public/pc-lens-1482.html                       (served page)
0 GPU; banked/staged arrays + cached weights only.
"""

from __future__ import annotations

import html
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM run)

import numpy as np  # noqa: E402

from explore_persona_space.task_workflow import repo_root  # noqa: E402

PROJECT_ROOT = repo_root()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1482_residual_svd as RS  # noqa: E402
from issue1482_footprint_moments import _load_gamma  # noqa: E402

JLENS = "data/issue_1482/jlens_dl/qwen2.5-7b-instruct_jlens.pt"
TUNED = "data/issue_1482/hf_dl/tuned_lens_L19.pt"
OWN_DESCRIPTIONS = (
    "eval_results/issue_1773/labels/descriptions.jsonl",
    "eval_results/issue_1773/recovery_1934/descriptions_recovered.jsonl",
)
OUT_JSON = "eval_results/issue_1482/pc_dashboard/pc_dashboard.json"
OUT_HTML = "dashboard/public/pc-lens-1482.html"
N_GROUP = 50
N_TOK = 8


def _own_descriptions(want: set[int]) -> dict[int, str]:
    found: dict[int, str] = {}
    for path in OWN_DESCRIPTIONS:
        p = PROJECT_ROOT / path
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                fid = int(rec.get("feat_id", rec.get("index", -1)))
                if fid in want:
                    found[fid] = (rec.get("description") or "").strip()
    return found


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

    top = list(range(N_GROUP))
    bottom = list(range(3584 - N_GROUP, 3584))
    pcs = top + bottom
    V = vecs[:, pcs].astype(np.float32)

    from issue1482_sae import BatchTopKSAE

    sae = BatchTopKSAE.load(k=64, layer=19, device="cpu")
    D = np.asarray(sae.w_dec, dtype=np.float32)
    D_unit = D / np.linalg.norm(D, axis=0, keepdims=True)
    cos = D_unit.T @ V  # (n_feat, n_pcs)
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
        cols = []
        for j in range(len(pcs)):
            ti = np.argsort(e[:, j])[-N_TOK:][::-1]
            cols.append([tok.decode([int(t)]) for t in ti])
        lens_toks[name] = cols

    want = {int(f) for f in argmax}
    own_desc = _own_descriptions(want)

    rows = []
    for j, pc in enumerate(pcs):
        fid = int(argmax[j])
        rows.append(
            {
                "group": "top50_by_variance" if pc < N_GROUP else "bottom50_by_variance",
                "pc": int(pc),
                "variance_share": float(share[pc]),
                "r2_ridge": float(r2["ridge"][pc]),
                "r2_mlp": float(r2["mlp_w8192"][pc]),
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
    out_json.write_text(json.dumps({"rows": rows}, indent=1))

    # ── HTML ──────────────────────────────────────────────────────────────────
    def esc(s: str) -> str:
        return html.escape(s if s else "—")

    def tokcell(ts: list[str]) -> str:
        return " ".join(f'<span class="tk">{html.escape(t.strip() or "␣")}</span>' for t in ts)

    def table(group: str) -> str:
        body = []
        for r_ in rows:
            if r_["group"] != group:
                continue
            body.append(
                f'<tr><td class="pc">PC{r_["pc"]}</td>'
                f"<td>{r_['variance_share']:.2e}</td>"
                f'<td class="{"good" if r_["r2_ridge"] > 0.5 else ("bad" if r_["r2_ridge"] < 0 else "")}">{r_["r2_ridge"]:.3f}</td>'
                f"<td>{r_['r2_mlp']:.3f}</td>"
                f'<td>{r_["sae_feat"]}</td>'
                f"<td>{r_['abs_cos']:.2f}</td>"
                f'<td class="desc">{esc(r_["own_desc"])}</td>'
                f"<td>{tokcell(r_['logit_lens'])}</td>"
                f"<td>{tokcell(r_['tuned_lens'])}</td>"
                f"<td>{tokcell(r_['jlens'])}</td></tr>"
            )
        return "\n".join(body)

    page = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Answer-PCA PCs — lens + SAE interpretations (#1482)</title>
<style>
body{{font-family:Inter,system-ui,sans-serif;margin:24px;color:#1a1f36;background:#fafbfc}}
h1{{font-size:20px}} h2{{font-size:16px;margin-top:28px}}
p.meta{{color:#556;max-width:1100px;font-size:13px;line-height:1.5}}
table{{border-collapse:collapse;font-size:12px;width:100%;background:#fff}}
th,td{{border:1px solid #e3e8ee;padding:5px 7px;vertical-align:top;text-align:left}}
th{{background:#f1f4f8;position:sticky;top:0}}
td.pc{{font-weight:600;white-space:nowrap}}
td.good{{background:#e7f6ec}} td.bad{{background:#fdeaea}}
td.desc{{max-width:230px}}
span.tk{{display:inline-block;background:#eef1f6;border-radius:3px;padding:0 4px;margin:1px;font-family:ui-monospace,monospace;font-size:11px}}
</style></head><body>
<h1>Answer-PCA directions: SAE + lens interpretations (issue #1482)</h1>
<p class="meta">Top 50 and bottom 50 PCs <b>by variance explained</b> of the mean answer-token
activations (context arm, L19, #1738 multi-turn holdout, n=9,941). Per PC: variance share,
held-out R² (linear ridge / nonlinear MLP), nearest SAE feature by |cos| over the 131,072-column
BatchTopK decoder (random-direction null max ≈ 0.05), its description from <b>our own #1773 Sonnet autointerp pipeline</b>, and the top-8 promoted tokens under logit lens W_U(γ·v), tuned lens W_U(γ·W_t v)
(pod-trained affine translator), and J-lens W_U(γ·J₁₉v) (community Jacobian artifact).
Bottom-50 PCs carry ~10⁻⁵ variance shares — their eigendirections are noise-dominated, so
interpretations there are expected to be unstable; that contrast is the point.
Data: eval_results/issue_1482/pc_dashboard/pc_dashboard.json.</p>
<h2>Top 50 PCs by variance</h2>
<table><thead><tr><th>PC</th><th>var share</th><th>R² ridge</th><th>R² MLP</th><th>SAE feat</th><th>|cos|</th><th>feature description (own #1773 autointerp)</th><th>logit lens</th><th>tuned lens</th><th>J-lens</th></tr></thead>
<tbody>{table("top50_by_variance")}</tbody></table>
<h2>Bottom 50 PCs by variance</h2>
<table><thead><tr><th>PC</th><th>var share</th><th>R² ridge</th><th>R² MLP</th><th>SAE feat</th><th>|cos|</th><th>feature description (own #1773 autointerp)</th><th>logit lens</th><th>tuned lens</th><th>J-lens</th></tr></thead>
<tbody>{table("bottom50_by_variance")}</tbody></table>
</body></html>"""
    out_html = PROJECT_ROOT / OUT_HTML
    out_html.write_text(page)
    print(f"[out] {out_json}")
    print(f"[out] {out_html} ({len(page) / 1024:.0f} KB, {len(rows)} rows)")


if __name__ == "__main__":
    main()
