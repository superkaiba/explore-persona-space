"""Issue #779 inline round: PREFIX-based twin of the pinv preimage top-context read.

User ask (chat 2026-07-22): "run a prefix-based twin inline".

The parent read (eval_results/issue_779/pinv_topk_contexts/, commit 117626de86)
projected CONTEXT vectors (full prompt incl. the user query, last token) onto the
persona vector r_B, the transpose map-through M^T r_B, and the pseudoinverse
preimage pinv(M, k*) r_B. This twin projects PREFIX vectors — the last token of
everything BEFORE the final user turn — onto the same four directions.

Scope (verbatim-replay constraint): of the 13 eval-grid conditions per trait,
only the 8 system-prompt conditions (sys0..7, project constants in
issue779_common.EVAL_SYSTEM_PROMPTS) and shot0 (no system prompt, no exemplars —
the prefix is the chat template's default system block) have prefixes that can be
rebuilt VERBATIM. shot5/10/15/20 are EXCLUDED: their exemplar pools were
vLLM-generated at capture time (seed 7) and never persisted, so those prefixes
are not replayable. n = 9 conditions per trait.

Reads:
  (1) rig verification — recompute the evil sys0 q0 CONTEXT vector on this CPU
      rig and compare to the stored pass_a capture (bf16-CUDA parent vs fp32-CPU
      here; gate on cosine >= 0.999 per read-out layer);
  (2) per trait x direction: prefix-end projection per condition (n=9), Spearman
      vs condition mean judge score, MATCHED against the context-arm Spearman
      recomputed on the SAME 9 conditions from the committed parent JSON;
  (3) prefix-vs-context per-condition projection agreement (Spearman, n=9);
  (4) cross-trait prefix ranking: all 25 unique prefixes (3 traits x 8 sys +
      shared shot0) projected on each trait's directions at that trait's layer —
      does the trait's own strong-prompt family top the ranking?

Compute: 0 GPU-h. ~26 batch-1 CPU forwards of frozen Qwen2.5-7B-Instruct
(fp32, <=~500 tokens each) + the parent's closed-form ridge/direction refit.
Fail-loud everywhere; no silent fallbacks.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/home/thomasjiralerspong/explore-persona-space")
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402

torch.set_num_threads(16)

import issue779_common as C  # noqa: E402
import issue779_stage1 as S1  # noqa: E402
from issue779_collect import build_eval_prompt_messages, eval_context_conditions  # noqa: E402

from explore_persona_space.analysis.extraction import extract_layer_activations  # noqa: E402
from explore_persona_space.experiments.issue_779 import fit_h as F  # noqa: E402

STAGE = Path("/mnt/eps-data/thomasjiralerspong/issue779_inline_pinv_topk")
DL = STAGE / "issue779_monitoring"
PASS_A = PROJECT_ROOT / "data/issue_779/pass_a"
PASS_B = DL / "analysis_tensors" / "pass_b" / "train_context_vectors.pt"
RB_DIR = DL / "r_b"
PARENT_JSON = PROJECT_ROOT / "eval_results/issue_779/pinv_topk_contexts/pinv_topk_contexts.json"
OUT_DIR = PROJECT_ROOT / "eval_results/issue_779/pinv_topk_contexts_prefix"

READ_OUT_LAYER = {"evil": 14, "sycophancy": 26, "hallucination": 17}
KSTAR_PREREG = {"evil": 1433, "sycophancy": 1321, "hallucination": 1565}
LAMBDAS = np.logspace(-2, 4, 13)
CAPTURE_LAYERS = sorted(set(READ_OUT_LAYER.values()))  # [14, 17, 26]
USER_TURN_MARK = "<|im_start|>user"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
VERIF_COS_FLOOR = 0.999


def ridge_fit_matrix(X_train, Y_train):
    """VERBATIM from pinv_topk_contexts.py / pinv_direction_read.py."""
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
    W = (Vt.T * filt) @ UtY
    return {"W": W, "xmu": xmu, "xsd": xsd, "ymu": ymu, "s": s, "lam": float(best_lam)}


def prefix_text_for(tokenizer, trait: str, cond: dict) -> str:
    """Everything before the final user turn of the templated eval prompt.

    Built by rendering the full prompt for a dummy question and cutting at the
    LAST occurrence of the user-turn header — the project's canonical prefix
    definition (prefix = everything before the user query).
    """
    messages = build_eval_prompt_messages(trait, cond, "DUMMY_QUESTION", exemplars=[])
    full = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    idx = full.rfind(USER_TURN_MARK)
    assert idx > 0, (trait, cond["cond_id"], full[:120])
    prefix = full[:idx]
    assert "DUMMY_QUESTION" not in prefix
    return prefix


def capture_last_token(model, tokenizer, text: str) -> dict[int, np.ndarray]:
    """Last-token activation of `text` at CAPTURE_LAYERS. {layer: (H,) float64}."""
    inputs = tokenizer(text, return_tensors="pt", padding=False)
    captured = extract_layer_activations(
        model, inputs["input_ids"], CAPTURE_LAYERS, attention_mask=inputs.get("attention_mask")
    )
    return {li: captured[li][0][-1, :].float().numpy().astype(np.float64) for li in CAPTURE_LAYERS}


def main() -> int:
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    parent = json.loads(PARENT_JSON.read_text())

    # ── directions per trait (verbatim parent refit; asserts pin the rig) ──
    tb = torch.load(PASS_B, map_location="cpu", mmap=True, weights_only=False)
    layers_b = list(tb["layers"])
    dirs_by_trait: dict[str, dict] = {}
    for trait, L in READ_OUT_LAYER.items():
        li = layers_b.index(L)
        Xtr = tb["cx_last"][:, li, :].to(torch.float64).numpy()
        Ytr = tb["v_x"][:, li, :].to(torch.float64).numpy()
        r_b = torch.load(RB_DIR / f"{trait}.pt", weights_only=False)["r_b"]
        r_b = r_b.to(torch.float64).numpy()[li]
        fit = ridge_fit_matrix(Xtr, Ytr)
        W, xmu, xsd, s, lam = fit["W"], fit["xmu"], fit["xsd"], fit["s"], fit["lam"]
        recon = F.reconstruction_metrics(((Xtr - xmu) / xsd) @ W + fit["ymu"], Ytr)
        committed_r2 = parent["traits"][trait]["recon_r2_committed"]
        assert abs(recon["r2"] - committed_r2) < 1e-3, (trait, recon["r2"], committed_r2)
        Mmat = W.T
        Um, Sm, Vmt = np.linalg.svd(Mmat, full_matrices=False)
        UtRb = Um.T @ r_b
        k_ridge = int(np.sum(s**2 >= lam))
        assert k_ridge == KSTAR_PREREG[trait], (trait, k_ridge)

        def pinv_dir(k, Sm=Sm, Vmt=Vmt, UtRb=UtRb):
            kk = Sm.shape[0] if k is None else min(k, Sm.shape[0])
            return Vmt[:kk].T @ (UtRb[:kk] / Sm[:kk])

        dirs_by_trait[trait] = {
            "directions": {
                "r_B_raw": (r_b, "raw"),
                "w_tr": (W @ r_b, "std"),
                "w_pinv_kstar": (pinv_dir(k_ridge), "std"),
                "w_pinv_full": (pinv_dir(None), "std"),
            },
            "xmu": xmu,
            "xsd": xsd,
        }
        print(
            f"[prefix-twin] {trait}: directions rebuilt, recon R2 verified "
            f"({recon['r2']:.4f}) ({time.time() - t0:.0f}s)",
            flush=True,
        )

    # ── model (CPU fp32; parent captured bf16-CUDA — verification gate below) ──
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    print(f"[prefix-twin] loading {MODEL_ID} fp32 on CPU... ({time.time() - t0:.0f}s)", flush=True)
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, torch_dtype=torch.float32)
    model.eval()
    print(f"[prefix-twin] model loaded ({time.time() - t0:.0f}s)", flush=True)

    # ── rig verification: evil sys0, q0 context vector vs stored pass_a ──
    cells = {c["cond_id"]: c for c in S1.load_eval_cells(PASS_A, "evil")}
    cell = cells["sys0"]
    q0 = C.load_extraction_artifacts("evil")["eval_questions"][0]
    cond0 = next(c for c in eval_context_conditions("evil") if c["cond_id"] == "sys0")
    msgs = build_eval_prompt_messages("evil", cond0, q0, exemplars=[])
    text0 = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    assert text0.endswith(C.GENERATION_SUFFIX)
    with torch.no_grad():
        got = capture_last_token(model, tokenizer, text0)
    verification = {}
    stored_layers = list(cell["_layers"])
    for li in CAPTURE_LAYERS:
        stored = cell["_cx_last"][0, stored_layers.index(li), :].astype(np.float64)
        cos = float(np.dot(got[li], stored) / (np.linalg.norm(got[li]) * np.linalg.norm(stored)))
        verification[f"L{li}_cosine_vs_stored"] = round(cos, 6)
        assert cos >= VERIF_COS_FLOOR, (li, cos)
    print(
        f"[prefix-twin] rig verified vs stored pass_a capture: {verification} "
        f"({time.time() - t0:.0f}s)",
        flush=True,
    )

    # ── build + capture the 25 unique verbatim prefixes ──
    prefix_records = []  # {key, trait_of_origin, cond_id, text, token_len}
    seen_texts = {}
    for trait in READ_OUT_LAYER:
        conds = [c for c in eval_context_conditions(trait) if c["cond_id"].startswith("sys")]
        conds.append(next(c for c in eval_context_conditions(trait) if c["cond_id"] == "shot0"))
        for cond in conds:
            ptxt = prefix_text_for(tokenizer, trait, cond)
            if ptxt in seen_texts:  # shot0 default block is shared across traits
                seen_texts[ptxt]["aliases"].append(f"{trait}:{cond['cond_id']}")
                continue
            rec = {
                "key": f"{trait}:{cond['cond_id']}",
                "aliases": [f"{trait}:{cond['cond_id']}"],
                "text": ptxt,
                "token_len": int(tokenizer(ptxt, return_tensors="pt")["input_ids"].shape[1]),
            }
            seen_texts[ptxt] = rec
            prefix_records.append(rec)
    print(
        f"[prefix-twin] {len(prefix_records)} unique prefixes "
        f"(token lens {[r['token_len'] for r in prefix_records]})",
        flush=True,
    )

    acts = {}  # key -> {layer: (H,)}
    with torch.no_grad():
        for rec in prefix_records:
            acts[rec["key"]] = capture_last_token(model, tokenizer, rec["text"])
            print(
                f"[prefix-twin]   captured {rec['key']} ({rec['token_len']} tok, "
                f"{time.time() - t0:.0f}s)",
                flush=True,
            )

    # ── analysis ──
    results = {
        "verification": verification,
        "n_conditions_per_trait": 9,
        "excluded_conditions": ["shot5", "shot10", "shot15", "shot20"],
        "exclusion_reason": "many-shot exemplar pools were vLLM-generated at capture time "
        "(seed 7) and never persisted; those prefixes are not replayable verbatim",
        "prefixes": [
            {k: r[k] for k in ("key", "aliases", "token_len", "text")} for r in prefix_records
        ],
        "traits": {},
    }
    alias_to_key = {a: r["key"] for r in prefix_records for a in r["aliases"]}

    for trait, L in READ_OUT_LAYER.items():
        dd = dirs_by_trait[trait]
        peg = parent["traits"][trait]["eval_grid"]
        cond_ids = [f"sys{i}" for i in range(8)] + ["shot0"]
        judge = {cid: peg["cond_mean_judge_score"][cid] for cid in cond_ids}
        tr = {"read_out_layer": L, "cond_mean_judge_score": judge, "directions": {}}

        for name, (w, frame) in dd["directions"].items():
            # prefix-arm projections for THIS trait's 9 conditions
            proj = {}
            for cid in cond_ids:
                c = acts[alias_to_key[f"{trait}:{cid}"]][L]
                cn = (c - dd["xmu"]) / dd["xsd"] if frame == "std" else c
                proj[cid] = float(cn @ w)
            pv = np.array([proj[c] for c in cond_ids])
            jv = np.array([judge[c] for c in cond_ids])
            ctx = np.array([peg[name]["per_condition_mean_proj"][c] for c in cond_ids])
            # cross-trait ranking of ALL unique prefixes on this direction
            allproj = {}
            for rec in prefix_records:
                c = acts[rec["key"]][L]
                cn = (c - dd["xmu"]) / dd["xsd"] if frame == "std" else c
                allproj[rec["key"]] = float(cn @ w)
            ranked = sorted(allproj, key=allproj.get, reverse=True)
            own_strong = {f"{trait}:sys0", f"{trait}:sys1", f"{trait}:sys2"}
            tr["directions"][name] = {
                "prefix_proj_per_condition": {c: round(proj[c], 4) for c in cond_ids},
                "spearman_prefix_vs_judge_n9": round(float(spearmanr(pv, jv).statistic), 4),
                "spearman_context_vs_judge_n9_matched": round(
                    float(spearmanr(ctx, jv).statistic), 4
                ),
                "spearman_prefix_vs_context_n9": round(float(spearmanr(pv, ctx).statistic), 4),
                "crosstrait_ranking_top8": ranked[:8],
                "own_strong_sys012_ranks": {
                    k.split(":")[1]: ranked.index(k) + 1 for k in sorted(own_strong)
                },
            }
        results["traits"][trait] = tr
        print(f"[prefix-twin] {trait}: analysis done ({time.time() - t0:.0f}s)", flush=True)

    results["metadata"] = {
        "script": "pinv_prefix_twin (inline user-chat override round)",
        "parent_artifact": "eval_results/issue_779/pinv_topk_contexts/ @ 117626de86",
        "model_id": MODEL_ID,
        "capture_dtype": "float32 CPU (parent pass_a: bf16 CUDA; verification gate "
        f"cosine >= {VERIF_COS_FLOOR} per read-out layer, see 'verification')",
        "prefix_definition": "everything before the final <|im_start|>user turn of the "
        "templated eval prompt; last-token activation",
        "wall_seconds": round(time.time() - t0, 1),
    }
    out = OUT_DIR / "pinv_prefix_twin.json"
    out.write_text(json.dumps(results, indent=1))
    print(f"[prefix-twin] wrote {out} ({time.time() - t0:.0f}s)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
