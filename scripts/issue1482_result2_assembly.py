"""Assembly pass for the Result 2 writeup fill-in: the missing figure, the
crossing statistics, the fitter-overlap check, the best/worst lens table, and
the top/bottom-100 SAE-feature description digest.

Outputs (all under eval_results/issue_1482/result2_assembly/ + figures/):
  - spectrum_ridge_vs_mlp.png       R^2 vs variance share, ridge + MLP (context L19)
  - assembly.json                   crossing stats, MLP/ridge best- and worst-20 overlap
  - best_worst_lens_table.md        markdown table: best-10 + worst-10 with SAE cos,
                                    autointerp, logit/tuned/J-lens top tokens
  - top_bottom100_descriptions.json panel features by R^2 with #1773 descriptions

0 GPU, banked/staged arrays only.
"""

from __future__ import annotations

import gzip
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

ALIGNMENT = "eval_results/issue_1482/twoway_residual/residual_alignment.json"
BEST = "eval_results/issue_1482/best_pc_alignment/best_pc_alignment.json"
LENS = "eval_results/issue_1482/lens_reads/lens_reads.json"
NP_CACHE = "eval_results/issue_1482/worst_pc_autointerp/np_cache"
PANEL = "eval_results/issue_1482/sae_perfeature/sae_ctx__mean__ridge.npz"
DESCRIPTIONS = (
    "eval_results/issue_1773/labels/descriptions.jsonl",
    "eval_results/issue_1773/recovery_1934/descriptions_recovered.jsonl",
)
OUT_DIR = "eval_results/issue_1482/result2_assembly"
FIG_DIR = "figures/issue_1482/result2_assembly"


def _np_descriptions(want: set[int], cache: Path) -> dict[int, str]:
    found: dict[int, str] = {}
    for p in sorted(cache.glob("*.jsonl.gz")):
        for line in gzip.decompress(p.read_bytes()).decode("utf-8").split("\n"):
            if not line.strip():
                continue
            rec = json.loads(line)
            if int(rec["index"]) in want:
                found[int(rec["index"])] = (rec.get("description") or "").strip()
    return found


def main() -> None:
    out_dir = PROJECT_ROOT / OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

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

    stats: dict = {}
    for fitter, r in r2.items():
        first_nonpos = int(np.argmax(r <= 0)) if (r <= 0).any() else None
        stats[fitter] = {
            "first_rank_r2_nonpositive": first_nonpos,
            "first_rank_below_0.5": int(np.argmax(r < 0.5)),
            "first_rank_below_0.1": int(np.argmax(r < 0.1)),
            "n_r2_above_0.5": int((r > 0.5).sum()),
            "n_r2_above_0.1": int((r > 0.1).sum()),
            "n_r2_positive": int((r > 0).sum()),
            "r2_at": {str(k): float(r[k]) for k in (0, 100, 199, 500, 1000, 2000, 3000)},
        }
    # fitter agreement on the extremes
    K = 20
    sets = {}
    for fitter, r in r2.items():
        order = np.argsort(r)
        sets[fitter] = {"worst": set(order[:K].tolist()), "best": set(order[-K:].tolist())}
    stats["extremes_overlap_ridge_vs_mlp"] = {
        "best20_shared": len(sets["ridge"]["best"] & sets["mlp_w8192"]["best"]),
        "worst20_shared": len(sets["ridge"]["worst"] & sets["mlp_w8192"]["worst"]),
        "spearman_r2_ridge_vs_mlp": float(
            np.corrcoef(
                np.argsort(np.argsort(r2["ridge"])), np.argsort(np.argsort(r2["mlp_w8192"]))
            )[0, 1]
        ),
    }
    (out_dir / "assembly.json").write_text(json.dumps(stats, indent=1))

    # ── figure: R^2 vs share, ridge vs MLP ───────────────────────────────────
    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    set_paper_style()
    import matplotlib.pyplot as plt

    colors = paper_palette(2)
    fig, ax = plt.subplots(figsize=(8.6, 5.2))
    for fi, (fitter, label) in enumerate(
        (("ridge", "linear (ridge)"), ("mlp_w8192", "nonlinear (MLP w8192)"))
    ):
        ax.scatter(share, r2[fitter], s=5, alpha=0.3, color=colors[fi], label=label)
    ax.set_xscale("log")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlabel("variance share of answer-PCA direction (log)")
    ax.set_ylabel("held-out R² along direction")
    ax.set_title("Per-direction R² vs variance share (context arm, L19)", loc="left")
    ax.legend(frameon=False, markerscale=3)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    savefig_paper(fig, "spectrum_ridge_vs_mlp", dir=PROJECT_ROOT / FIG_DIR)

    # ── best/worst lens table (ridge cell; lens reads banked) ────────────────
    best_doc = json.loads((PROJECT_ROOT / BEST).read_text())["cells"]["context_L19_ridge"]
    worst_doc = json.loads((PROJECT_ROOT / ALIGNMENT).read_text())["cells"]["context_L19_ridge"]
    lens_doc = json.loads((PROJECT_ROOT / LENS).read_text())["sets"]

    worst_feats = [int(x) for x in worst_doc["sae_alignment"]["argmax_feature_per_worst"]]
    worst_desc = _np_descriptions(set(worst_feats), PROJECT_ROOT / NP_CACHE)

    def lens_toks(set_name: str, i: int, lens: str, n: int = 4) -> str:
        row = lens_doc[set_name][i]
        return ", ".join(t.strip() or "␣" for t in row[lens]["top_promoted"][:n])

    lines = [
        "| rank | R² | |cos| nearest SAE feat | autointerp of that feat | logit lens | tuned lens | J-lens |",
        "|---|---|---|---|---|---|---|",
    ]
    for i in range(10):
        pc = best_doc["best_indices"][i]
        cos = best_doc["sae_alignment_best"]["max_abs_cos_per_best"][i]
        desc = (best_doc["best_rows"][i]["autointerp"] or "(no desc)")[:70]
        lines.append(
            f"| **best** PC{pc} | {best_doc['best_r2'][i]:.3f} | {cos:.2f} | {desc} | "
            f"{lens_toks('best20', i, 'logit_lens')} | {lens_toks('best20', i, 'tuned_lens')} | "
            f"{lens_toks('best20', i, 'jlens')} |"
        )
    wr2 = worst_doc.get("worst_r2") or [float(r2["ridge"][j]) for j in worst_doc["worst_indices"]]
    for i in range(10):
        pc = worst_doc["worst_indices"][i]
        cos = worst_doc["sae_alignment"]["max_abs_cos_per_worst"][i]
        desc = (worst_desc.get(worst_feats[i], "") or "(no desc)")[:70]
        lines.append(
            f"| **worst** PC{pc} | {wr2[i]:.3f} | {cos:.2f} | {desc} | "
            f"{lens_toks('worst20', i, 'logit_lens')} | {lens_toks('worst20', i, 'tuned_lens')} | "
            f"{lens_toks('worst20', i, 'jlens')} |"
        )
    (out_dir / "best_worst_lens_table.md").write_text("\n".join(lines) + "\n")

    # ── top/bottom-100 panel features with descriptions ──────────────────────
    zp = np.load(PROJECT_ROOT / PANEL)
    feat_ids = np.asarray(zp["feat_ids"], dtype=int)
    r2f = np.asarray(zp["r2"], dtype=np.float64)
    act = np.asarray(zp["activity"], dtype=np.float64)
    ok = np.isfinite(r2f)
    order = np.argsort(r2f[ok])
    idx_ok = np.flatnonzero(ok)
    bottom = idx_ok[order[:100]]
    top = idx_ok[order[-100:][::-1]]

    desc_map: dict[int, str] = {}
    for path in DESCRIPTIONS:
        p = PROJECT_ROOT / path
        if not p.exists():
            continue
        with p.open() as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                fid = int(rec.get("feat_id", rec.get("index", -1)))
                if fid >= 0:
                    desc_map[fid] = (rec.get("description") or "").strip()

    def rows(idx: np.ndarray) -> list[dict]:
        return [
            {
                "feat_id": int(feat_ids[i]),
                "r2": round(float(r2f[i]), 4),
                "activity": round(float(act[i]), 5),
                "description": desc_map.get(int(feat_ids[i]), ""),
            }
            for i in idx
        ]

    tb = {"top100_by_r2": rows(top), "bottom100_by_r2": rows(bottom)}
    (out_dir / "top_bottom100_descriptions.json").write_text(json.dumps(tb, indent=1))

    print(json.dumps(stats, indent=1))
    print("\n== TOP-100 description digest (first 55 chars) ==")
    for r_ in tb["top100_by_r2"]:
        print(f"  {r_['r2']:+.3f} {r_['description'][:55]}")
    print("\n== BOTTOM-100 description digest ==")
    for r_ in tb["bottom100_by_r2"]:
        print(f"  {r_['r2']:+.3f} {r_['description'][:55]}")


if __name__ == "__main__":
    main()
