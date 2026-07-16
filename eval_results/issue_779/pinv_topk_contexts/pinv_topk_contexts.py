"""Issue #779 inline free-analysis: qualitative inspection of the pinv preimage
direction's top-projecting contexts.

Mentor ask (chat 2026-07-14):
  "if you take the vector that is the pre-image of a persona vector, and find the
   context vectors that maximize the projection along it, what do they look like?
   do they look reasonable?"

The pre-image direction is w_pinv = M^+ r_B — the min-norm prompt-space vector
that maps TO the persona vector r_B under the fitted linear context->answer map M
(stage-1 ridge: v = M c_std, M = W^T). This script reuses the parent pinv run's
fit/projection recipe VERBATIM (eval_results/issue_779/pinv_direction_read/
pinv_direction_read.py @ 4c327632f7) — standardize-X, center-Y, GCV-lambda,
closed-form SVD; w_pinv(k) via truncated SVD of M — and adds the qualitative
read the quantitative monitoring pass never ran: WHICH contexts maximize
<c, w_pinv>, and do they look trait-relevant / coherent / artifact-like.

Four directions per trait, at the frozen read-out layer (evil L14 / sycophancy
L26 / hallucination L17), all as they are CANONICALLY applied:
  r_B_raw       = r_B                (raw-c frame, the parent's pv_raw convention)
  w_tr          = M^T r_B = W r_B    (standardized-c frame)
  w_pinv_kstar  = pinv(M,k*) r_B     (standardized-c frame; k* = ridge-estimable rank)
  w_pinv_full   = pinv(M,full) r_B   (standardized-c frame; the collapse contrast)

Reads: (1) top/bottom-10 LMSYS contexts per direction; (2) pass_a eval-grid
per-condition means + top-30 composition + Spearman(proj, judge score);
(3) direction-relatedness Spearman + top-100 Jaccard; (4) length-confound
Spearman(proj, token_len).

0 GPU, VM CPU, analysis-only. Closed-form (a handful of SVDs of ~5000x3584 and
3584x3584) — no gradient descent, no draw battery, no per-cell loop; projected
wall-time is minutes.

SAFETY: LMSYS prompts are raw real-user text. This script NEVER prints full rows;
it categorizes each retrieved context into a coarse theme bucket, flags
explicit/jailbreak/harmful-shaped rows and stores a category placeholder (no
text) for those, and truncates every innocuous stored row to <=250 chars.
"""

from __future__ import annotations

import json
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import torch  # noqa: E402

torch.set_num_threads(8)

import issue779_stage1 as S1  # noqa: E402

from explore_persona_space.experiments.issue_779 import fit_h as F  # noqa: E402

STAGE = Path("/mnt/eps-data/thomasjiralerspong/issue779_inline_pinv_topk")
DL = STAGE / "issue779_monitoring"
PASS_A = DL / "analysis_tensors" / "pass_a"
PASS_B = DL / "analysis_tensors" / "pass_b" / "train_context_vectors.pt"
RB_DIR = DL / "r_b"
PROMPTS_CACHE = STAGE / "lmsys_prompts.json"

HF_REV = "037fcbb210bc52c459959b0746cc268fe08bae96"
LMSYS_REV = "200748d9d3cddcc9d782887541057aca0b18c5da"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"

OUT = PROJECT_ROOT / "eval_results" / "issue_779" / "pinv_topk_contexts"
OUT.mkdir(parents=True, exist_ok=True)
COMMITTED = json.loads(
    (
        PROJECT_ROOT / "eval_results/issue_779/pinv_direction_read/pinv_direction_read.json"
    ).read_text()
)

READ_OUT_LAYER = {"evil": 14, "sycophancy": 26, "hallucination": 17}
KSTAR_PREREG = {"evil": 1433, "sycophancy": 1321, "hallucination": 1565}
LAMBDAS = np.logspace(-2, 4, 13)  # stage-1 ridge_fit_predict default GCV grid
TOPK = 10
TOP_EVAL = 30
JACCARD_K = 100
TRUNC = 250


def ridge_fit_matrix(X_train, Y_train):
    """VERBATIM from pinv_direction_read.py: replicate fit_h.ridge_fit_predict
    internals, returning W (d, D_out) + standardization params + GCV lambda +
    standardized-X singular values. Reproduces F.ridge_fit_predict to machine
    precision (verified there); the recon-R2 cross-check below re-confirms it."""
    Xtr = np.asarray(X_train, dtype=np.float64)
    Ytr = np.asarray(Y_train, dtype=np.float64)
    n = Xtr.shape[0]
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - xmu) / xsd
    ymu = Ytr.mean(0)
    Ytr_c = Ytr - ymu
    U, s, Vt = np.linalg.svd(Xtr_n, full_matrices=False)
    s2 = s**2
    UtY = U.T @ Ytr_c
    best_lam, best_gcv = LAMBDAS[0], np.inf
    for lam in LAMBDAS:
        filt = s2 / (s2 + lam)
        Yhat_tr = U @ (filt[:, None] * UtY)
        rss = float(np.sum((Ytr_c - Yhat_tr) ** 2))
        dof = float(np.sum(filt))
        denom = (n - dof) ** 2
        gcv = rss / denom if denom > 1e-12 else np.inf
        if gcv < best_gcv:
            best_gcv, best_lam = gcv, lam
    filt = s / (s2 + best_lam)
    W = (Vt.T * filt) @ UtY  # (d, D_out)
    return {"W": W, "xmu": xmu, "xsd": xsd, "ymu": ymu, "s": s, "lam": float(best_lam)}


# ── safety / theme classification ─────────────────────────────────────────────

_FLAG_PATTERNS = {
    "sexual-explicit": re.compile(
        r"\b(nsfw|porn|erotic|sexual|explicit sex|blowjob|nude|hentai|incest|"
        r"rape|slut|dick|pussy|cum|orgasm|fuck (me|her|him)|make .* horny)\b",
        re.I,
    ),
    "jailbreak": re.compile(
        r"(ignore (all )?(previous|above) instructions|\bDAN\b|do anything now|"
        r"developer mode|jailbreak|without any (restrictions|filter|censorship)|"
        r"you are (now )?free|pretend you have no (rules|guidelines))",
        re.I,
    ),
    "violence-illegal": re.compile(
        r"\b(how to (make|build|synthesize).*(bomb|explosive|meth|drug|weapon)|"
        r"kill (someone|a person|him|her)|untraceable|counterfeit|hack into|"
        r"steal .* (money|credit card)|child (porn|abuse))\b",
        re.I,
    ),
}

_THEME_PATTERNS = [
    (
        "code",
        re.compile(
            r"(```|def |class |import |function |javascript|python|c\+\+|java\b|sql\b|"
            r"html|css|regex|compile|debug|stack trace|for loop|const |var |=>|"
            r"print\(|write (a|some) (code|program|script|function))",
            re.I,
        ),
    ),
    (
        "math",
        re.compile(
            r"(solve|equation|integral|derivative|matrix|probability|theorem|"
            r"calculate|\bsum of\b|factorial|prime number|\d+ ?[\+\-\*/x] ?\d+)",
            re.I,
        ),
    ),
    (
        "translation",
        re.compile(
            r"(translate|translation|in (french|spanish|german|chinese|japanese|"
            r"korean|russian|italian|portuguese)|traduc|перевод|翻译)",
            re.I,
        ),
    ),
    (
        "roleplay-creative",
        re.compile(
            r"(role[- ]?play|pretend (you are|to be)|act as|you are (a|an) |write (a|an) "
            r"(story|poem|song|dialogue|script|essay|fiction)|imagine you|character|"
            r"fantasy|as if you were)",
            re.I,
        ),
    ),
    (
        "factual-explain",
        re.compile(
            r"(what is|who is|who was|explain|define|difference between|how does|"
            r"tell me about|describe|why (is|do|does)|when (was|did))",
            re.I,
        ),
    ),
    (
        "advice-personal",
        re.compile(
            r"(should i|help me|my (boss|girlfriend|boyfriend|wife|husband|mom|dad|"
            r"friend|life)|i (feel|am feeling)|advice|how (can|do) i (get|deal|cope|"
            r"lose|improve))",
            re.I,
        ),
    ),
    (
        "summarize-rewrite",
        re.compile(
            r"(summar|rewrite|rephrase|paraphrase|proofread|correct the|make .* "
            r"(shorter|longer|formal|professional)|tl;?dr)",
            re.I,
        ),
    ),
]


def safety_flag(text: str) -> str | None:
    for cat, pat in _FLAG_PATTERNS.items():
        if pat.search(text):
            return cat
    return None


def classify_theme(text: str) -> str:
    for name, pat in _THEME_PATTERNS:
        if pat.search(text):
            return name
    return "other"


def store_row(idx: int, score: float, tok_len: int, prompts: list[str]) -> dict:
    """Safe stored record for one retrieved context: theme + flag always; text
    only when NOT flagged and truncated to <=250 chars."""
    text = prompts[idx]
    flag = safety_flag(text)
    rec = {
        "idx": int(idx),
        "score": float(score),
        "token_len": int(tok_len),
        "char_len": len(text),
        "theme": classify_theme(text),
        "flagged": flag,
    }
    if flag is None:
        rec["text"] = text[:TRUNC]
    else:
        rec["text"] = f"[{flag} row — categorized, not quoted]"
    return rec


# ── tokenizer (loaded once) ───────────────────────────────────────────────────

_TOK = None


def token_lengths(prompts: list[str]) -> np.ndarray:
    """Chat-templated (add_generation_prompt) token length per prompt — matches
    what capture_context_vector's c_last actually indexes. Tokenizer loaded ONCE."""
    global _TOK
    from transformers import AutoTokenizer

    if _TOK is None:
        _TOK = AutoTokenizer.from_pretrained(MODEL_ID)
    lens = np.empty(len(prompts), dtype=np.int64)
    for i, p in enumerate(prompts):
        text = _TOK.apply_chat_template(
            [{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True
        )
        lens[i] = len(_TOK(text, add_special_tokens=False)["input_ids"])
    return lens


def main() -> int:
    t0 = time.time()
    prompts = json.loads(PROMPTS_CACHE.read_text())
    assert len(prompts) == 5000, len(prompts)
    print(f"[topk] loaded {len(prompts)} reconstructed LMSYS prompts", flush=True)

    tok_len = token_lengths(prompts)
    print(
        f"[topk] token lengths: min/median/max = "
        f"{tok_len.min()}/{int(np.median(tok_len))}/{tok_len.max()}  "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    tb = torch.load(PASS_B, map_location="cpu", mmap=True, weights_only=False)
    layers = list(tb["layers"])
    assert tb["cx_last"].shape[0] == 5000, tb["cx_last"].shape

    results = {"traits": {}, "prompt_source": tb.get("source"), "n_contexts": 5000}

    for trait, L in READ_OUT_LAYER.items():
        print(f"\n[topk] === {trait} @ layer {L} === ({time.time() - t0:.0f}s)", flush=True)
        li = layers.index(L)
        Xtr = tb["cx_last"][:, li, :].to(torch.float64).numpy()  # (5000, H) raw
        Ytr = tb["v_x"][:, li, :].to(torch.float64).numpy()  # (5000, H)
        rb_blob = torch.load(RB_DIR / f"{trait}.pt", weights_only=False)
        r_b_all = rb_blob["r_b"].to(torch.float64).numpy()  # (28, H)
        r_b = r_b_all[li]  # (H,)

        fit = ridge_fit_matrix(Xtr, Ytr)
        W, xmu, xsd, s, lam = fit["W"], fit["xmu"], fit["xsd"], fit["s"], fit["lam"]
        Xtr_n = (Xtr - xmu) / xsd
        recon = F.reconstruction_metrics(Xtr_n @ W + fit["ymu"], Ytr)
        recon_committed = COMMITTED["traits"][trait]["recon_ridge"]["r2"]
        recon_match = abs(recon["r2"] - recon_committed) < 1e-3
        print(
            f"[topk]   recon R2={recon['r2']:.4f} (committed {recon_committed:.4f}, "
            f"match={recon_match})",
            flush=True,
        )

        Mmat = W.T  # (H_out, H_in): v = M c_std
        Um, Sm, Vmt = np.linalg.svd(Mmat, full_matrices=False)
        UtRb = Um.T @ r_b
        k_ridge = int(np.sum(s**2 >= lam))
        assert k_ridge == KSTAR_PREREG[trait], (trait, k_ridge, KSTAR_PREREG[trait])

        def pinv_dir(k):
            kk = Sm.shape[0] if k is None else min(k, Sm.shape[0])
            return Vmt[:kk].T @ (UtRb[:kk] / Sm[:kk])

        w_tr = W @ r_b
        w_pinv_k = pinv_dir(k_ridge)
        w_pinv_full = pinv_dir(None)

        # frame: r_B is applied to RAW c (parent pv_raw); the fitted-map directions
        # to STANDARDIZED c (matches the parent's eval-side reads exactly).
        directions = {
            "r_B_raw": (r_b, "raw"),
            "w_tr": (w_tr, "std"),
            "w_pinv_kstar": (w_pinv_k, "std"),
            "w_pinv_full": (w_pinv_full, "std"),
        }
        proj = {}  # (5000,) per direction — canonical frame
        for name, (w, frame) in directions.items():
            Xc = Xtr if frame == "raw" else Xtr_n
            proj[name] = Xc @ w

        # ── item 1: LMSYS top/bottom-10 per direction ──
        topbot = {}
        for name, p in proj.items():
            order = np.argsort(p)  # ascending
            top = order[::-1][:TOPK]
            bot = order[:TOPK]
            topbot[name] = {
                "top": [store_row(int(i), p[i], tok_len[i], prompts) for i in top],
                "bottom": [store_row(int(i), p[i], tok_len[i], prompts) for i in bot],
                "top_theme_counts": dict(Counter(classify_theme(prompts[int(i)]) for i in top)),
                "top_flag_counts": dict(
                    Counter(safety_flag(prompts[int(i)]) or "none" for i in top)
                ),
                "bottom_theme_counts": dict(Counter(classify_theme(prompts[int(i)]) for i in bot)),
            }

        # ── item 3: direction relatedness (canonical projections) ──
        names = list(directions)
        rel_spearman = {}
        rel_jaccard = {}
        top100 = {n: set(np.argsort(proj[n])[::-1][:JACCARD_K].tolist()) for n in names}
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                na, nb = names[a], names[b]
                rho = float(spearmanr(proj[na], proj[nb]).statistic)
                inter = len(top100[na] & top100[nb])
                union = len(top100[na] | top100[nb])
                rel_spearman[f"{na}|{nb}"] = round(rho, 4)
                rel_jaccard[f"{na}|{nb}"] = round(inter / union, 4)

        # ── item 4: length confound ──
        length_rho = {
            name: round(float(spearmanr(proj[name], tok_len).statistic), 4) for name in names
        }

        # ── item 2: eval-grid composition ──
        cells = S1.load_eval_cells(PASS_A, trait)
        cond_map: dict[str, int] = {}
        for cell in cells:  # replicate build_eval_matrix cond ordering
            cond_map.setdefault(cell["cond_id"], len(cond_map))
        int2cond = {v: k for k, v in cond_map.items()}
        mat = S1.build_eval_matrix(cells, L, r_b_all)
        Xev = mat["c_last"]
        Xev_n = (Xev - xmu) / xsd
        y = mat["y"]
        cond_ids = [int2cond[int(c)] for c in mat["cond"]]

        eval_proj = {}
        for name, (w, frame) in directions.items():
            Xc = Xev if frame == "raw" else Xev_n
            eval_proj[name] = Xc @ w

        # trait-high conditions by mean judge score
        cond_meany = {}
        for cid in cond_map:
            m = np.array([c == cid for c in cond_ids])
            cond_meany[cid] = round(float(y[m].mean()), 2)
        trait_high = sorted(cond_meany, key=cond_meany.get, reverse=True)

        eval_grid = {"cond_mean_judge_score": cond_meany, "trait_high_conditions": trait_high}
        for name, ep in eval_proj.items():
            per_cond = {}
            for cid in cond_map:
                m = np.array([c == cid for c in cond_ids])
                per_cond[cid] = round(float(ep[m].mean()), 4)
            top30 = np.argsort(ep)[::-1][:TOP_EVAL]
            top30_conds = dict(Counter(cond_ids[int(i)] for i in top30))
            rho_y = float(spearmanr(ep, y).statistic)
            eval_grid[name] = {
                "per_condition_mean_proj": per_cond,
                "top30_condition_counts": top30_conds,
                "spearman_proj_vs_judgescore": round(rho_y, 4),
            }

        # rig sanity: reproduce the committed within-condition Pearson via method_metrics
        rig = {}
        method_x = {
            "pv_raw": Xtr,
            "transpose_MTrb": None,
            "pinv_headline": None,
        }
        for mname, xproj in {
            "pv_raw": Xev @ r_b,
            "transpose_MTrb": Xev_n @ w_tr,
            "pinv_headline": Xev_n @ w_pinv_k,
        }.items():
            mm = S1.method_metrics(np.asarray(xproj, dtype=np.float64), mat, n_boot=200, seed=0)
            comm = COMMITTED["traits"][trait]["methods"].get(
                {
                    "pv_raw": "pv_raw",
                    "transpose_MTrb": "transpose_MTrb",
                    "pinv_headline": "pinv_headline",
                }[mname]
            )
            rig[mname] = {
                "system_point": round(float(mm["system"]["point"]), 4),
                "many_shot_point": round(float(mm["many_shot"]["point"]), 4),
                "committed_system": round(comm["system"]["point"], 4),
                "committed_many_shot": round(comm["many_shot"]["point"], 4),
                "system_match_0p02": abs(mm["system"]["point"] - comm["system"]["point"]) < 0.02,
                "many_shot_match_0p02": abs(mm["many_shot"]["point"] - comm["many_shot"]["point"])
                < 0.02,
            }
        _ = method_x  # (kept for readability of the mname keys)

        results["traits"][trait] = {
            "read_out_layer": L,
            "k_ridge_estimable": k_ridge,
            "ridge_lambda": lam,
            "recon_r2": round(recon["r2"], 4),
            "recon_r2_committed": round(recon_committed, 4),
            "recon_r2_match_1e-3": recon_match,
            "M_condition_number": round(float(Sm[0] / (Sm[-1] + 1e-30)), 1),
            "cos_wpinv_kstar_rb": round(
                float(np.dot(w_pinv_k, r_b) / (np.linalg.norm(w_pinv_k) * np.linalg.norm(r_b))), 4
            ),
            "cos_wtr_rb": round(
                float(np.dot(w_tr, r_b) / (np.linalg.norm(w_tr) * np.linalg.norm(r_b))), 4
            ),
            "wpinv_kstar_norm": round(float(np.linalg.norm(w_pinv_k)), 4),
            "wpinv_full_norm": round(float(np.linalg.norm(w_pinv_full)), 4),
            "lmsys_topbottom": topbot,
            "direction_relatedness_spearman": rel_spearman,
            "direction_relatedness_top100_jaccard": rel_jaccard,
            "length_confound_spearman": length_rho,
            "eval_grid": eval_grid,
            "rig_sanity_within_cond_pearson": rig,
        }
        # console summary
        for name in names:
            tt = topbot[name]["top_theme_counts"]
            print(
                f"[topk]   {name:14s} len_rho={length_rho[name]:+.2f}  top10_themes={tt}",
                flush=True,
            )
        print(
            f"[topk]   rel_jaccard(r_B_raw|w_pinv_kstar)="
            f"{rel_jaccard['r_B_raw|w_pinv_kstar']}  "
            f"rel_jaccard(w_tr|w_pinv_kstar)={rel_jaccard['w_tr|w_pinv_kstar']}  "
            f"rel_jaccard(w_pinv_kstar|w_pinv_full)="
            f"{rel_jaccard['w_pinv_kstar|w_pinv_full']}",
            flush=True,
        )

    results["metadata"] = {
        "script": "pinv_topk_contexts (inline free-analysis, user-chat carve-out)",
        "followup_label": "pinv-topk-context-inspection",
        "parent_run": "eval_results/issue_779/pinv_direction_read (@ 4c327632f7)",
        "hf_data_revision": HF_REV,
        "lmsys_revision": LMSYS_REV,
        "model_id": MODEL_ID,
        "read_out_layers": READ_OUT_LAYER,
        "kstar_prereg": KSTAR_PREREG,
        "ridge_recipe": (
            "fit_h.ridge_fit_predict (standardize-X, center-Y, GCV lambda over "
            "logspace(-2,4,13), closed-form SVD); n_train=5000; M=W^T (v=M c_std)"
        ),
        "frames": {
            "r_B_raw": "raw c . r_B (parent pv_raw convention)",
            "w_tr / w_pinv_*": "standardized c . direction (fitted-map frame)",
        },
        "prompt_reconstruction": (
            "LMSYS pass_b bundle @ pinned rev carries NO 'prompts' field; "
            "reconstructed first 5000 non-empty first-user-turns of "
            "lmsys/lmsys-chat-1m @ LMSYS_REV (the #823/#952 replay recipe) — "
            "aligns 1:1 with the 5000 bundle rows (n_train=5000, no drops)"
        ),
        "safety": (
            "explicit/jailbreak/harmful-shaped rows stored as category placeholder "
            f"(no text); innocuous rows truncated to <={TRUNC} chars"
        ),
    }
    out_json = OUT / "pinv_topk_contexts.json"
    out_json.write_text(json.dumps(results, indent=2))
    print(f"\n[topk] wrote {out_json}  ({time.time() - t0:.0f}s total)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
