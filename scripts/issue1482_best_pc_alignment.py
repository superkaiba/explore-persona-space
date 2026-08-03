"""BEST-predicted-direction alignment + autointerp — the mirror of the banked
worst-direction pass (#1482 twoway_residual/residual_alignment.json).

The alignment pass computed SAE-decoder alignment + logit lens for the 20 WORST
target-PCA directions only (`argmax_feature_per_worst`, `logit_lens_worst`); the
20 BEST directions got indices/R^2 banked but no characterization. This closes
that gap with the SAME machinery (imported from issue1482_residual_svd, never
rewritten) so best and worst are directly comparable: SAE max-|cos| + argmax
feature + matched random-unit null, logit lens, and the NeuronPedia autointerp
join from the locally cached export.

Self-verification: recomputed best/worst indices are asserted equal to the
banked ones before anything else runs. 0 GPU.
"""

from __future__ import annotations

import gzip
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM run)

import numpy as np  # noqa: E402

from explore_persona_space.task_workflow import repo_root

PROJECT_ROOT = repo_root()
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import issue1482_residual_svd as RS  # noqa: E402

ALIGNMENT = "eval_results/issue_1482/twoway_residual/residual_alignment.json"
NP_CACHE = "eval_results/issue_1482/worst_pc_autointerp/np_cache"
OUT = "eval_results/issue_1482/best_pc_alignment/best_pc_alignment.json"
CELLS = (("context", 19, "ridge"), ("prefix", 19, "ridge"), ("bare", 19, "ridge"))


def _np_descriptions(want: set[int], cache: Path) -> dict[str, str]:
    found: dict[str, str] = {}
    for p in sorted(cache.glob("*.jsonl.gz")):
        for line in gzip.decompress(p.read_bytes()).decode("utf-8").split("\n"):
            if not line.strip():
                continue
            rec = json.loads(line)
            idx = int(rec["index"])
            if idx in want:
                found[str(idx)] = (rec.get("description") or "").strip()
    return found


def main() -> None:
    banked = json.loads((PROJECT_ROOT / ALIGNMENT).read_text())["cells"]
    rng = np.random.default_rng(RS.SEED if hasattr(RS, "SEED") else 1482)

    sae = None
    W_U = tok = None
    doc: dict = {
        "design": {
            "question": (
                "Are the SAE features / unembedding reads aligned with the BEST-predicted "
                "directions legible, in contrast with the banked worst-direction grab-bag?"
            ),
            "mirrors": "residual_alignment.json worst-direction conventions, applied to best_indices",
        },
        "cells": {},
    }

    for arm, layer, fitter in CELLS:
        name = RS.cell_name(arm, layer, fitter)
        b = banked[name]
        y16, ci = RS.load_layer(layer)
        pred16 = RS.load_pred(arm, layer, fitter, ci)
        Y = np.asarray(y16, dtype=np.float64)
        E = Y - np.asarray(pred16, dtype=np.float64)
        n, d = E.shape
        Yc = Y - Y.mean(axis=0, keepdims=True)
        y_lam, y_vecs = RS.gram_spectrum(Yc, want_vectors=True, n_vec=RS.R2_SELECT_K)
        ss_tot = np.square(Yc @ y_vecs).sum(axis=0)
        r2 = 1.0 - np.square(E @ y_vecs).sum(axis=0) / ss_tot
        best = np.argsort(r2)[-RS.N_WORST :][::-1]
        worst = np.argsort(r2)[: RS.N_WORST]
        assert [int(x) for x in worst] == [int(x) for x in b["worst_indices"]], name
        assert [int(x) for x in best] == [int(x) for x in b["best_indices"]], name
        V_best = y_vecs[:, best].astype(np.float32)

        if sae is None:
            from issue1482_sae import BatchTopKSAE

            sae = BatchTopKSAE.load(k=64, layer=19, device="cpu")
        D = np.asarray(sae.w_dec, dtype=np.float32)
        D_unit = (D / np.linalg.norm(D, axis=0, keepdims=True)).T
        cos_sae = D_unit @ V_best
        argmax = np.argmax(np.abs(cos_sae), axis=0)

        if W_U is None:
            W_U, tok = RS.load_unembedding()
        logits = W_U @ V_best
        lens = []
        for j in range(len(best)):
            col = logits[:, j]
            top = np.argsort(col)[-12:][::-1]
            lens.append(
                {
                    "pc_index": int(best[j]),
                    "r2": float(r2[best[j]]),
                    "top_tokens": [tok.decode([int(t)]) for t in top],
                }
            )

        doc["cells"][name] = {
            "arm": arm,
            "layer": layer,
            "fitter": fitter,
            "best_indices": [int(x) for x in best],
            "best_r2": [float(r2[i]) for i in best],
            "sae_alignment_best": {
                "max_abs_cos_per_best": [
                    float(abs(cos_sae[argmax[j], j])) for j in range(len(best))
                ],
                "argmax_feature_per_best": [int(x) for x in argmax],
                "null_random_unit_max_over_dictionary": b["sae_alignment"][
                    "null_random_unit_max_over_dictionary"
                ],
                "null_note": "direction-independent null reused from the banked worst pass",
            },
            "logit_lens_best": lens,
        }

    want = {
        int(f)
        for c in doc["cells"].values()
        for f in c["sae_alignment_best"]["argmax_feature_per_best"]
    }
    desc = _np_descriptions(want, PROJECT_ROOT / NP_CACHE)
    for c in doc["cells"].values():
        rows = []
        sa = c["sae_alignment_best"]
        for j, pc in enumerate(c["best_indices"]):
            fid = sa["argmax_feature_per_best"][j]
            rows.append(
                {
                    "pc_index": pc,
                    "pc_r2": c["best_r2"][j],
                    "feat_id": fid,
                    "abs_cos": sa["max_abs_cos_per_best"][j],
                    "autointerp": desc.get(str(fid), ""),
                    "lens_top6": c["logit_lens_best"][j]["top_tokens"][:6],
                }
            )
        c["best_rows"] = rows

    out_path = PROJECT_ROOT / OUT
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(doc, indent=1))
    print(f"[out] {out_path}")
    c = doc["cells"][RS.cell_name("context", 19, "ridge")]
    print(
        f"\ncontext_L19_ridge — BEST 20 (null max ~{c['sae_alignment_best']['null_random_unit_max_over_dictionary']['max']:.3f}):"
    )
    for r in c["best_rows"][:12]:
        print(
            f"  PC {r['pc_index']:3d} R2 {r['pc_r2']:.3f} feat {r['feat_id']:6d} "
            f"|cos| {r['abs_cos']:.3f}  {r['autointerp'][:60] or '(no desc)'}  "
            f"lens: {', '.join(t.strip() for t in r['lens_top6'][:4])}"
        )


if __name__ == "__main__":
    main()
