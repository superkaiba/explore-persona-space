"""Issue #952 — noise-ceiling of the layer-20 context->answer ridge-map R2 DV.

Inline user-directed analysis leg (label ``noise_ceiling``). The held-out
in-domain R2 of the layer-20 context->answer maps is ~0.11-0.16 (own arm) and
was read as "weak". This leg estimates the NOISE CEILING of that DV: because the
answer representation for a FIXED context varies across resampled generations,
no context->answer map (a deterministic function of the context) can exceed
R2 = 1 - W/T, where W is the within-context (resample) variance of the answer
representation and T is its single-draw total variance.

Method (matches the round's own recipe + DV EXACTLY):
  * own arm = Qwen-2.5-7B-Instruct, vLLM, temperature 1.0, top_p 0.95,
    max_tokens 1024, engine+sampling seed 42, no system prompt (run_952._own_gen).
    The ONLY change vs the round is n=1 -> n=K resamples per context.
  * DV = the 42 POSITION_SLOTS answer-representation vectors at layer 20,
    teacher-forced fp16 capture via run_952._tf_capture_slots_arm.
  * R2 aggregation = the round's _per_query_r2: per context,
    1 - sum_slot ss_res / sum_slot ss_tot over the 42 position slots (H-summed).

Contexts:
  * PRIMARY: the 78 realized in-domain check contexts of the diverse-train-
    injection round (the contexts the 0.1128 / 0.1565 own R2 is measured on).
  * SECONDARY (--china): the 31 kept china-politics divergent queries (the
    divergence-conditioned cell whose own R2 ~0.10 / cross-cell ~0.02 read null).

Content discipline (BINDING): the injection + china prompts are CCP-sensitive
bank items. This script NEVER prints/logs/quotes prompt OR answer text — only
counts, shas, and numerics. Rollout text is written to files and uploaded to HF
(the round's own-arm persistence contract) but never echoed to stdout.

Ceiling estimators (all on layer-20 POSITION_SLOTS, per-query then mean-over-
contexts to match the reported mean-of-per-query R2, plus a pooled variant):
  * ICC(1) bias-corrected  (PRIMARY): unbiased within-var (ddof=1); between
    de-biased by W/k; T centred at the resampled-set grand mean per slot.
  * naive 1 - W/T          (no bias correction; matches the raw pooled_r2 form).
  * LORO cross-validated    (leave-one-rollout-out; conservative, within inflated
    by k/(k-1)) + a k/(k-1)-corrected LORO.

Usage (pod):
  uv run python scripts/issue952_noise_ceiling_gpu.py --phase all [--china] [--k 10]
  uv run python scripts/issue952_noise_ceiling_gpu.py --phase all --smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pathlib
import sys
import time

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
# Layer grid: keep the pre-registered capture-equivalence layers {14,17,26}
# (run_952 asserts this) + the read-out layer 20. Fewer layers => less memory.
os.environ.setdefault("EPM_I952_LAYER_GRID", "14,17,20,26")

import numpy as np  # noqa: E402

import explore_persona_space.experiments.issue_952.run_952 as R  # noqa: E402

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    stream=sys.stdout,
)
logger = logging.getLogger("issue952_ceiling")

ISSUE = 952
TAG = "noise_ceiling"
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue952_position_divergence/followups/noise_ceiling"
READ_OUT_LAYER = 20
REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]

# reported in-domain R2 (diverse_train_injection/divtrain_refit_eval.json) — for
# restatement as fraction-of-ceiling. Read from disk at report time (no hardcode).
DIVTRAIN_EVAL = (
    REPO_ROOT / "eval_results/issue_952/diverse_train_injection/divtrain_refit_eval.json"
)
CHINA_STATS = REPO_ROOT / "eval_results/issue_952/china-politics-topup/stats_china_included.json"
INJ_MANIFEST = REPO_ROOT / "eval_results/issue_952/diverse-train-injection/injection_manifest.json"
BANK_QUERIES = REPO_ROOT / "eval_results/issue_952/divergence_bank_queries.json"
CHINA_TOPUP_VERIF = (
    REPO_ROOT
    / "eval_results/issue_952/china-politics-topup/summaries/china_topup_verification.json"
)
CHINA_NEW_CANDS = (
    REPO_ROOT / "eval_results/issue_952/china-politics-topup/staging/new_candidates.json"
)


# ── context selection (refs only; text resolved on GPU, never logged) ────────────
def _indomain_check_rows() -> list[dict]:
    """The realized in-domain check rows (split == indomain_check) of the divtrain round."""
    manifest = json.loads(INJ_MANIFEST.read_text())
    rows = [r for r in manifest["rows"] if r.get("split") == "indomain_check"]
    # keep ONLY the 78 that survived the round's capture intersection
    reported = json.loads(DIVTRAIN_EVAL.read_text())["indomain_check_per_context"]["ids"]
    keep = set(reported)
    rows = [r for r in rows if r["query_id"] in keep]
    assert len(rows) == len(keep), (len(rows), len(keep))
    for r in rows:
        r["role"] = "indomain_check"
    return rows


def _china_divergent_rows() -> list[dict]:
    """Divergent queries of the 31 kept china-politics pairs (refs only)."""
    verif = json.loads(CHINA_TOPUP_VERIF.read_text())
    kept_pairs = list(verif.get("final_china_kept_pairs", []))
    kept_div_ids = {f"{p}_div" for p in kept_pairs}
    # source map: parent bank queries (bank_file+index) plus topup new_candidates queries
    src_by_id: dict[str, dict] = {}
    for q in json.loads(BANK_QUERIES.read_text())["queries"]:
        src_by_id[q["query_id"]] = q
    if CHINA_NEW_CANDS.exists():
        for q in json.loads(CHINA_NEW_CANDS.read_text()).get("queries", []):
            if q.get("query_id"):
                src_by_id.setdefault(q["query_id"], q)
    rows = []
    for qid in sorted(kept_div_ids):
        q = src_by_id.get(qid)
        if q is None or not (q.get("text") or q.get("source")):
            continue
        rows.append(
            {
                "query_id": qid,
                "source": q.get("source"),
                "text": q.get("text"),
                "role": "china_divergent",
            }
        )
    logger.info("[china] kept_pairs=%d resolvable divergent rows=%d", len(kept_pairs), len(rows))
    return rows


# ── generation: K own resamples per context ──────────────────────────────────────
def phase_gen(rows: list[dict], out_base: pathlib.Path, k: int) -> pathlib.Path:
    """K own answers per context (vLLM n=K, temp 1.0, top_p 0.95, max_tokens 1024, seed 42)."""
    tokenizer = R._get_tokenizer()
    texts = {r["query_id"]: R.resolve_query_text(r) for r in rows}
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": texts[r["query_id"]]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in rows
    ]
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=R.DEFAULT_MODEL, dtype="bfloat16", max_model_len=8192, seed=42, trust_remote_code=True
    )
    sp = SamplingParams(n=k, temperature=1.0, top_p=0.95, max_tokens=R.SONNET_MAX_TOKENS, seed=42)
    chunk = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "200"))
    records: dict[str, list[dict]] = {}
    for c0 in range(0, len(formatted), chunk):
        logger.info("[gen] chunk %d..%d / %d", c0, min(c0 + chunk, len(formatted)), len(formatted))
        outs = llm.generate(formatted[c0 : c0 + chunk], sp, use_tqdm=False)
        for r, o in zip(rows[c0 : c0 + chunk], outs, strict=True):
            records[r["query_id"]] = [
                {"answer_text": s.text, "n_tokens": len(s.token_ids)} for s in o.outputs
            ]
    R._reap_vllm(llm)
    # distinctness check (numeric only — never print text)
    n_distinct = [len({s["answer_text"] for s in v}) for v in records.values()]
    logger.info(
        "[gen] %d contexts x n=%d ; mean distinct answers/context=%.2f (min=%d)",
        len(records),
        k,
        float(np.mean(n_distinct)),
        int(min(n_distinct)),
    )
    assert float(np.mean(n_distinct)) > 1.05, "resamples nearly identical — sampling not stochastic"
    p = out_base / "raw_completions" / f"own_rollouts_k{k}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        json.dumps({"k": k, "n_contexts": len(records), "rollouts": records}, default=R._json_np)
    )
    logger.info("[gen] wrote %d contexts -> %s", len(records), p.name)
    return p


# ── capture: teacher-force the 42 position slots per rollout ──────────────────────
def phase_capture(rows: list[dict], out_base: pathlib.Path, k: int) -> pathlib.Path:
    """Capture layer-grid 72-slot tensors for every (context, rollout) via the round's own code."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    roll = json.loads((out_base / "raw_completions" / f"own_rollouts_k{k}.json").read_text())[
        "rollouts"
    ]
    texts = {r["query_id"]: R.resolve_query_text(r) for r in rows}
    # pseudo-rollout ids: {qid}__r{i}
    ids, prompts_by_id, answers_by_id, context_of = [], {}, {}, {}
    for r in rows:
        qid = r["query_id"]
        if qid not in roll:
            continue
        for i, s in enumerate(roll[qid]):
            rid = f"{qid}__r{i}"
            ids.append(rid)
            prompts_by_id[rid] = texts[qid]
            answers_by_id[rid] = s["answer_text"]
            context_of[rid] = qid
    logger.info("[capture] %d rollout-ids x %d layers", len(ids), len(R.LAYER_GRID))

    tokenizer = AutoTokenizer.from_pretrained(R.DEFAULT_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        R.DEFAULT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model.eval()
    bs = int(os.environ.get("EPM_TF_BATCH_SIZE", "8"))
    slots, _spans, _surp = R._tf_capture_slots_arm(
        model,
        tokenizer,
        ids,
        prompts_by_id,
        answers_by_id,
        "own_ceiling",
        own_raw_lens=None,
        batch_size=bs,
    )
    # slots: (n, L, 72, H) fp16, NaN where invalid. Slice layer 20 + POSITION_SLOTS.
    li = list(R.LAYER_GRID).index(READ_OUT_LAYER)
    pos_idx = [R.SLOT_IDX[s] for s in R.POSITION_SLOTS]
    y20 = slots[:, li][:, pos_idx, :].astype(np.float32)  # (n, 42, H)
    del slots
    np.savez(
        str(out_base / f"capture_L20_k{k}.npz"),
        y20=y20.astype(np.float16),
        rollout_ids=np.asarray(ids),
        context_of=np.asarray([context_of[r] for r in ids]),
        slot_names=np.asarray(list(R.POSITION_SLOTS)),
        layer=READ_OUT_LAYER,
    )
    logger.info("[capture] saved (%s) layer-20 42-slot tensor", y20.shape)
    return out_base / f"capture_L20_k{k}.npz"


# ── ceiling math ─────────────────────────────────────────────────────────────────
def _ceiling_from_capture(y: np.ndarray, context_of: np.ndarray, k: int) -> dict:
    """y: (n_rollouts, S, H) f64. context_of: (n_rollouts,) query_id per rollout.

    Returns per-query + pooled ceilings under ICC(1)-bias-corrected, naive, LORO.
    """
    y = y.astype(np.float64)
    contexts = list(dict.fromkeys(context_of.tolist()))
    S = y.shape[1]
    # group rollout rows by context
    by_ctx = {c: np.where(context_of == c)[0] for c in contexts}

    # grand mean per slot over ALL finite rollouts (single-draw total centring)
    finite_row = np.isfinite(y).all(axis=2)  # (n, S)
    grand = np.full((S, y.shape[2]), np.nan)
    for s in range(S):
        m = finite_row[:, s]
        if m.any():
            grand[s] = y[m, s, :].mean(axis=0)

    per_ctx = {
        "icc": [],
        "naive": [],
        "loro": [],
        "loro_corr": [],
        "W": [],
        "Tsingle": [],
        "kmin": [],
    }
    ids_out = []
    # pooled accumulators
    pool = {kk: [0.0, 0.0] for kk in ("icc", "naive", "loro")}  # [sum_res, sum_tot]
    for c in contexts:
        rr = by_ctx[c]
        num_icc = den_icc = num_naive = den_naive = num_loro = den_loro = 0.0
        used_slots = 0
        w_ctx = t_ctx = 0.0
        for s in range(S):
            rows_s = rr[finite_row[rr, s]]
            kcs = len(rows_s)
            if kcs < 2:
                continue
            Ys = y[rows_s, s, :]  # (kcs, H)
            mu = Ys.mean(axis=0)
            ss_within = float(((Ys - mu) ** 2).sum())  # H-summed, over kcs
            W = ss_within / (kcs - 1)  # unbiased single-draw within-var
            d2 = float(((mu - grand[s]) ** 2).sum())
            B = d2 - W / kcs  # de-biased between
            Tsingle = B + W
            # naive (biased): within = ss_within/kcs ; total = mean_i ||Y - grand||^2
            W_naive = ss_within / kcs
            T_naive = float(((Ys - grand[s]) ** 2).sum()) / kcs
            # LORO: leave-one-rollout-out oracle-mean residual
            sum_i = Ys.sum(axis=0)
            loro_res = 0.0
            for j in range(kcs):
                mu_j = (sum_i - Ys[j]) / (kcs - 1)
                loro_res += float(((Ys[j] - mu_j) ** 2).sum())
            loro_res /= kcs
            loro_tot = float(((Ys - grand[s]) ** 2).sum()) / kcs
            # accumulate
            num_icc += W
            den_icc += Tsingle
            num_naive += W_naive
            den_naive += T_naive
            num_loro += loro_res
            den_loro += loro_tot
            w_ctx += W
            t_ctx += Tsingle
            used_slots += 1
            pool["icc"][0] += W
            pool["icc"][1] += Tsingle
            pool["naive"][0] += W_naive
            pool["naive"][1] += T_naive
            pool["loro"][0] += loro_res
            pool["loro"][1] += loro_tot
        if used_slots == 0 or den_icc <= 1e-12:
            continue
        ids_out.append(c)
        per_ctx["icc"].append(1 - num_icc / den_icc)
        per_ctx["naive"].append(1 - num_naive / den_naive)
        per_ctx["loro"].append(1 - num_loro / den_loro)
        # k/(k-1)-corrected LORO: scale loro residual down by (k-1)/k
        per_ctx["loro_corr"].append(1 - (num_loro * (k - 1) / k) / den_loro)
        per_ctx["W"].append(w_ctx)
        per_ctx["Tsingle"].append(t_ctx)
        per_ctx["kmin"].append(int(np.median([finite_row[by_ctx[c], s].sum() for s in range(S)])))

    def _m(x):
        a = np.asarray(x, dtype=float)
        return float(np.nanmean(a)) if len(a) else None

    return {
        "n_contexts": len(ids_out),
        "context_ids": ids_out,
        "mean_ceiling_icc": _m(per_ctx["icc"]),
        "mean_ceiling_naive": _m(per_ctx["naive"]),
        "mean_ceiling_loro": _m(per_ctx["loro"]),
        "mean_ceiling_loro_corrected": _m(per_ctx["loro_corr"]),
        "pooled_ceiling_icc": (1 - pool["icc"][0] / pool["icc"][1]) if pool["icc"][1] > 0 else None,
        "pooled_ceiling_naive": (1 - pool["naive"][0] / pool["naive"][1])
        if pool["naive"][1] > 0
        else None,
        "pooled_ceiling_loro": (1 - pool["loro"][0] / pool["loro"][1])
        if pool["loro"][1] > 0
        else None,
        "per_context": {
            "ids": ids_out,
            "ceiling_icc": per_ctx["icc"],
            "within_W": per_ctx["W"],
            "total_Tsingle": per_ctx["Tsingle"],
            "median_k_valid": per_ctx["kmin"],
        },
    }


def phase_stats(out_base: pathlib.Path, k: int, role: str) -> dict:
    d = np.load(str(out_base / f"capture_L20_k{k}.npz"), allow_pickle=True)
    y = d["y20"]  # (n, 42, H) fp16
    context_of = d["context_of"]
    ceil = _ceiling_from_capture(y, context_of, k)
    ceil["role"] = role
    return ceil


# ── report / restatement ─────────────────────────────────────────────────────────
def _restate(ceil_indomain: dict, ceil_china: dict | None) -> dict:
    ev = json.loads(DIVTRAIN_EVAL.read_text())
    r2 = {
        "indomain_own_pool_only": ev["pool_only"]["indomain_check"]["mean_r2_own"],
        "indomain_own_augmented": ev["augmented"]["indomain_check"]["mean_r2_own"],
        "indomain_ext_pool_only": ev["pool_only"]["indomain_check"]["mean_r2_ext_plain"],
        "indomain_ext_augmented": ev["augmented"]["indomain_check"]["mean_r2_ext_plain"],
    }
    C = ceil_indomain["mean_ceiling_icc"]
    frac = {kk: (v / C if C else None) for kk, v in r2.items()}
    out = {
        "reported_r2": r2,
        "indomain_ceiling_icc_mean": C,
        "r2_as_fraction_of_indomain_ceiling": frac,
    }
    if ceil_china is not None:
        cs = json.loads(CHINA_STATS.read_text())["r2_levels_by_arm"]["china"]
        Cc = ceil_china["mean_ceiling_icc"]
        out["china"] = {
            "china_own_r2_div": cs["own"]["mean_r2_div"],
            "china_own_r2_ctl": cs["own"]["mean_r2_ctl"],
            "china_ext_r2_div": cs["ext_plain"]["mean_r2_div"],
            "china_ceiling_icc_mean": Cc,
            "china_own_r2_div_as_fraction_of_china_ceiling": (
                cs["own"]["mean_r2_div"] / Cc if Cc else None
            ),
        }
    return out


def _upload(out_base: pathlib.Path, k: int) -> list[str]:
    from huggingface_hub import HfApi

    api = HfApi()
    uploaded = []
    for fname in [
        f"raw_completions/own_rollouts_k{k}.json",
        f"capture_L20_k{k}.npz",
        "ceiling_stats.json",
    ]:
        p = out_base / fname
        if not p.exists():
            continue
        api.upload_file(
            path_or_fileobj=str(p),
            path_in_repo=f"{HF_PREFIX}/{fname}",
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
        )
        uploaded.append(f"{HF_PREFIX}/{fname}")
        logger.info("[upload] %s", fname)
    return uploaded


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", default="all", choices=["gen", "capture", "stats", "all"])
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--china", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out", default="/workspace/issue952_ceiling")
    args = ap.parse_args()
    t0 = time.time()

    roles = [("indomain_check", _indomain_check_rows())]
    if args.china:
        roles.append(("china_divergent", _china_divergent_rows()))
    if args.smoke:
        roles = [(r, rows[:4]) for r, rows in roles]
        args.k = 3

    results = {}
    for role, rows in roles:
        out_base = pathlib.Path(args.out) / role
        out_base.mkdir(parents=True, exist_ok=True)
        logger.info("=== role=%s n_contexts=%d k=%d ===", role, len(rows), args.k)
        if args.phase in ("gen", "all"):
            phase_gen(rows, out_base, args.k)
        if args.phase in ("capture", "all"):
            phase_capture(rows, out_base, args.k)
        if args.phase in ("stats", "all"):
            ceil = phase_stats(out_base, args.k, role)
            (out_base / "ceiling_stats.json").write_text(
                json.dumps(ceil, indent=2, default=R._json_np)
            )
            results[role] = ceil
            _upload(out_base, args.k)

    if args.phase in ("stats", "all") and "indomain_check" in results:
        report = _restate(results["indomain_check"], results.get("china_divergent"))
        report["wall_seconds"] = time.time() - t0
        report["k"] = args.k
        report["smoke"] = args.smoke
        report["ceilings"] = {
            role: {
                kk: c[kk]
                for kk in c
                if kk.startswith(("mean_ceiling", "pooled_ceiling", "n_contexts"))
            }
            for role, c in results.items()
        }
        rp = pathlib.Path(args.out) / "noise_ceiling_report.json"
        rp.write_text(json.dumps(report, indent=2, default=R._json_np))
        # per-context arrays for figures (in-domain)
        pc = results["indomain_check"]["per_context"]
        np.savez(
            str(pathlib.Path(args.out) / "per_context_indomain.npz"),
            ids=np.asarray(pc["ids"]),
            ceiling_icc=np.asarray(pc["ceiling_icc"]),
            within_W=np.asarray(pc["within_W"]),
            total_Tsingle=np.asarray(pc["total_Tsingle"]),
        )
        logger.info("[report] %s", json.dumps(report, default=R._json_np))
        # sentinel
        pathlib.Path("/workspace/logs").mkdir(parents=True, exist_ok=True)
        pathlib.Path(f"/workspace/logs/issue-952-{TAG}-done.json").write_text(
            json.dumps(
                {"tag": TAG, "wall_seconds": time.time() - t0, "report": report}, default=R._json_np
            )
        )


if __name__ == "__main__":
    main()
