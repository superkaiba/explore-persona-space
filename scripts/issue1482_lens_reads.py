"""Consolidated LENS READS: logit lens vs J-lens vs tuned lens, applied to the
direction sets of the what-is-the-map-bad-at-predicting writeup.

Direction sets (all L19, #1738 holdout target PCA + #779 trait directions):
  - the 15 below-curve DEVIANT-band PCs (deviant_band.json, context_L19_ridge)
  - the banked BEST-20 and WORST-20 predicted PCs (residual_alignment.json)
  - the 7 persona-vector trait directions (r_B row 19, unit norm)

Per direction v, three vocabulary reads e = W_U (gamma ⊙ T v):
  logit lens  T = I           (direct unembedding of the write direction)
  J-lens      T = J_19        (community Jacobian artifact, J[26]~identity)
  tuned lens  T = W_tuned     (hand-rolled affine translator, pod round; the
                               bias is dropped per the artifact's apply_note —
                               a DIRECTION is a difference of states, so b
                               cancels)
Top promoted/suppressed tokens per read. 0 GPU, seconds of GEMM.
"""

from __future__ import annotations

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
from issue1482_rb7_reads import TRAITS7, _rb_matrix  # noqa: E402

DEVIANT = "eval_results/issue_1738/deviant_band/deviant_band.json"
ALIGNMENT = "eval_results/issue_1482/twoway_residual/residual_alignment.json"
JLENS = "data/issue_1482/jlens_dl/qwen2.5-7b-instruct_jlens.pt"
TUNED = "data/issue_1482/hf_dl/tuned_lens_L19.pt"
OUT = "eval_results/issue_1482/lens_reads/lens_reads.json"
TOP_PROM = 10
TOP_SUPP = 6


def _reads(
    w_u: np.ndarray, gamma: np.ndarray, tok, lenses: dict[str, np.ndarray | None], V: np.ndarray
) -> list[dict]:
    """V: (d, n) unit direction columns. Returns per-direction per-lens tokens."""
    out: list[dict] = []
    per_lens = {}
    for name, T in lenses.items():
        tv = V if T is None else T @ V
        e = w_u @ (tv * gamma[:, None])  # (vocab, n)
        per_lens[name] = (e, np.linalg.norm(tv * gamma[:, None], axis=0))
    for j in range(V.shape[1]):
        row: dict = {}
        for name, (e, wn) in per_lens.items():
            col = e[:, j]
            top = np.argsort(col)[-TOP_PROM:][::-1]
            bot = np.argsort(col)[:TOP_SUPP]
            row[name] = {
                "top_promoted": [tok.decode([int(t)]) for t in top],
                "top_promoted_vals": [round(float(col[t]), 3) for t in top],
                "top_suppressed": [tok.decode([int(t)]) for t in bot],
                "write_norm_through_lens": round(float(wn[j]), 4),
            }
        out.append(row)
    return out


def main() -> None:
    import torch

    w_u, tok = RS.load_unembedding()
    gamma = _load_gamma()
    J = (
        torch.load(PROJECT_ROOT / JLENS, map_location="cpu", weights_only=False)["J"][19]
        .to(torch.float32)
        .numpy()
    )
    td = torch.load(PROJECT_ROOT / TUNED, map_location="cpu", weights_only=False)
    Wt = td["W"].numpy().astype(np.float32)
    assert Wt.shape == (RS.HIDDEN_DIM, RS.HIDDEN_DIM)
    lenses: dict[str, np.ndarray | None] = {"logit_lens": None, "jlens": J, "tuned_lens": Wt}

    # direction sets
    y16, ci = RS.load_layer(19)
    Y = np.asarray(y16, dtype=np.float64)
    Yc = Y - Y.mean(axis=0, keepdims=True)
    _lam, y_vecs = RS.gram_spectrum(Yc, want_vectors=True, n_vec=RS.R2_SELECT_K)

    dv = json.loads((PROJECT_ROOT / DEVIANT).read_text())
    deviant_ranks = [int(r) for r in dv["replication"]["n1738_deviant_ranks_sorted"]]
    banked = json.loads((PROJECT_ROOT / ALIGNMENT).read_text())["cells"]
    cell = banked[RS.cell_name("context", 19, "ridge")]
    best = [int(x) for x in cell["best_indices"]]
    worst = [int(x) for x in cell["worst_indices"]]

    rb = _rb_matrix()  # (d, 7) unit columns

    sets = {
        "deviant_band": (
            y_vecs[:, deviant_ranks].astype(np.float32),
            [f"PC{r}" for r in deviant_ranks],
        ),
        "best20": (y_vecs[:, best].astype(np.float32), [f"PC{r}" for r in best]),
        "worst20": (y_vecs[:, worst].astype(np.float32), [f"PC{r}" for r in worst]),
        "traits7": (rb.astype(np.float32), list(TRAITS7)),
    }

    doc: dict = {
        "design": {
            "question": (
                "Do the deviant-band / best / worst predicted directions (and the 7 "
                "trait directions) become legible under J-lens or tuned lens where "
                "the plain logit lens is illegible?"
            ),
            "lens_conventions": {
                "logit_lens": "W_U (gamma . v)",
                "jlens": "W_U (gamma . (J_19 v)) — community artifact, 50 wikitext prompts",
                "tuned_lens": (
                    "W_U (gamma . (W_t v)) — bias dropped per artifact apply_note "
                    "(direction = difference of states)"
                ),
            },
            "tuned_lens_quality": {
                k: td[k]
                for k in (
                    "val_kl_tuned",
                    "val_kl_logit_lens",
                    "val_kl_reduction_frac",
                    "val_top1_tuned",
                    "val_top1_logit_lens",
                )
            },
            "direction_source": "#1738 holdout target-PCA (context_L19_ridge cell) + #779 r_B row 19",
        },
        "sets": {},
    }
    for name, (V, labels) in sets.items():
        rows = _reads(w_u, gamma, tok, lenses, V)
        doc["sets"][name] = [{"direction": lab, **row} for lab, row in zip(labels, rows)]

    out_path = PROJECT_ROOT / OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=1))
    print(f"[out] {out_path}")

    for name in ("deviant_band", "traits7"):
        print(f"\n===== {name} =====")
        for row in doc["sets"][name][:8]:
            print(f"  {row['direction']}:")
            for lens in ("logit_lens", "jlens", "tuned_lens"):
                toks = ", ".join(t.strip() or "␣" for t in row[lens]["top_promoted"][:6])
                print(f"    {lens:11s} -> {toks}")


if __name__ == "__main__":
    main()
