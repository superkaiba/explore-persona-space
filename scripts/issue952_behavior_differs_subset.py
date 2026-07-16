"""#952 behavior-differs subset re-read of the china-politics divergence bank.

User ask (2026-07-15, chat): rerun the #952 divergence-bank statistics restricted
to china pairs where Qwen and Claude GENUINELY behave differently, and contrast
against the behavior-similar complement. The pooled china read found NO
divergence-specific external penalty (arm-matched d +0.014, Holm p 0.38); the
refusal-sanity round showed the pooled null is driven by the cross cell being
uniformly floored on china. This script asks whether a divergence-selective
penalty emerges once we condition on the pairs where the two models' behavior
actually diverges (judge-text labels, independent of the activation DV).

EXPLORATORY CONDITIONAL READ — conditioning is on behavior labels (judge text
labels), not pre-registered, small n. Not a confirmatory test.

Subsets over the 31 kept china pairs (comparable to the committed read):
  S1 refusal-mismatch (PRIMARY): divergent-query refusal_qwen >= 50 AND
     refusal_claude < 50 (Qwen refuses, Claude answers).
  S2 lexical-divergence (SECONDARY, descriptive): divergent-query TF-idf cosine
     in the bottom tercile among kept china pairs. TF-idf recomputed uniformly
     over the china corpus (the committed parent-corpus tfidf_cos exists for only
     the 18 parent pairs; recompute gives all 31 a comparable value). Committed
     parent-corpus values reported alongside where available.
  Complement: NOT-S1 (behavior-similar) so the contrast is visible.

Per subset + complement, per cell (arm-matched / cross / symmetric): pooled mean,
95% bootstrap CI (10k, seed 0), 10k sign-flip p (seed 1), n, and mean per-query
R2 LEVELS divergent-vs-control (floored vs divergence-selective, visible).

Machinery reused VERBATIM from the committed #952 free-analysis line: the
run_ridge_cell shared-SVD ridge, the universe/split reproduction, the
POSITION_SLOTS / SLOT_IDX registry, the frozen per-slot lambda, and the batched
bootstrap / sign-flip helpers — imported from issue952_divergence_transfer_cell /
issue952_china_included_stats so the fit + stats are provably identical to the
committed china-included cells. The committed cross / symmetric per-pair rows are
read directly from stats_china_included.json; only the arm-matched per-pair drops
(absent from committed JSONs) are re-fit here, gated bit-close against the
committed pooled arm-matched d before any subsetting.

Content safety: the china bank is CCP-sensitive. This script references items by
pair_id / query_id ONLY. It NEVER prints, logs, or persists bank query or answer
TEXT — it reads activation tensors + numeric labels, and (for the S2 TF-idf
recompute) loads answer text ONLY into a vectorizer, never inspecting or logging
the string values (log ids / counts / lengths only).

Usage:
  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \
    MALLOC_ARENA_MAX=2 uv run python scripts/issue952_behavior_differs_subset.py
"""

from __future__ import annotations

# ruff: noqa: E402 — load_dotenv() must run before numpy/torch import (shared-VM thread caps)
from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import datetime
import json
import logging
import pathlib
import subprocess
import time

import numpy as np
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue952_behavior_differs_subset")

# Reuse the committed fit + stats machinery verbatim.
from issue952_china_included_stats import (
    CHINA_CAT,
    _load_slots,
    _pairwise_drops,
    _role_split,
)
from issue952_divergence_transfer_cell import (
    ARMS,
    BANK_ARMS,
    N_DRAWS,
    SLOT_IDX,
    _bank_boot,
    _per_query_r2,
    _signflip_p,
    _stack_answer_targets,
)

REPO = "superkaiba1/explore-persona-space-data"
PARENT_REV = "5b62649cefb34902fd630f21630164e8d1d99764"
CHINA_REV = "612c6c744e786ff65faae8e7ee97736239f873e5"
PARENT_PREFIX = "issue952_position_divergence/analysis_tensors"
CHINA_PREFIX = "issue952_position_divergence/followups/china_politics_topup/analysis_tensors"
PARENT_RC = "issue952_position_divergence/raw_completions/bank"
CHINA_RC = (
    "issue952_position_divergence/followups/china_politics_topup/"
    "raw_completions/china_politics_topup"
)
STAGE = pathlib.Path("/mnt/eps-data/thomasjiralerspong/tmp_issue952_dashboard")

REFUSAL_THR = 50.0  # rubric midpoint ("partial deflection"); boolean cut for S1
REPRO_TOL = 1e-6  # per-pair fit reproduction is bit-close, so gate tightly
BASE = pathlib.Path(__file__).resolve().parent.parent


def _sha() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=BASE,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


# ── staging (per-file download with bounded retry; scoped, never snapshot) ────────


def _stage(prefix: str, rev: str, fname: str) -> pathlib.Path:
    from huggingface_hub import hf_hub_download

    dest = STAGE / rev[:10] / fname.replace("/", "__")
    if dest.exists() or dest.is_symlink():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(4):
        try:
            p = hf_hub_download(REPO, f"{prefix}/{fname}", repo_type="dataset", revision=rev)
            dest.symlink_to(pathlib.Path(p))
            return dest
        except Exception as e:  # transient Hub 5xx / 429 — bounded retry
            last = e
            logger.warning("[hf] %s failed (attempt %d): %s", fname, attempt + 1, e)
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"HF download failed after retries: {prefix}/{fname}") from last


# ── china answer-text loader (numeric-free logging; text only into a vectorizer) ──


def load_china_texts() -> dict[str, dict]:
    """query_id -> {pair_id, role, origin, question, qwen_answer, claude_answer} for
    every CAPTURED china query (84 queries = 42 pairs; provenance-defined). Parent
    pairs pull Qwen/Claude from the parent bank raw completions; new pairs from the
    china top-up raw completions. Query text via run_952.resolve_query_text. NEVER
    logs resolved text — length/count only."""
    from explore_persona_space.experiments.issue_952 import run_952 as R

    prov = json.loads(
        (
            BASE / "eval_results/issue_952/china-politics-topup/summaries/"
            "provenance_china_politics_topup.json"
        ).read_text()
    )["provenance"]

    # bank query rows (for resolve_query_text) — parent + new candidates
    parent_bank = {
        r["query_id"]: r
        for r in json.loads(
            (BASE / "eval_results/issue_952/divergence_bank_queries.json").read_text()
        )["queries"]
        if r.get("category") == CHINA_CAT
    }
    new_bank = {
        r["query_id"]: r
        for r in json.loads(
            (
                BASE / "eval_results/issue_952/china-politics-topup/staging/new_candidates.json"
            ).read_text()
        )["queries"]
    }

    # Qwen / Claude answers
    p_qwen = {
        r["query_id"]: r.get("answer_text")
        for r in json.loads(_stage(PARENT_RC, PARENT_REV, "qwen_seed42.json").read_text())
    }
    p_claude = {
        r["query_id"]: r.get("answer_text")
        for r in json.loads(_stage(PARENT_RC, PARENT_REV, "claude_seed42.json").read_text())
    }
    c_qwen = {
        r["query_id"]: r.get("answer_text")
        for r in json.loads(_stage(CHINA_RC, CHINA_REV, "qwen_seed42.json").read_text())
    }
    c_claude: dict[str, str] = {}
    for line in _stage(CHINA_RC, CHINA_REV, "claude_answers.jsonl").read_text().split("\n"):
        if line.strip():
            rec = json.loads(line)
            c_claude[rec["query_id"]] = rec["answer_text"]

    out: dict[str, dict] = {}
    for qid, meta in prov.items():
        origin, pid, role = meta["origin"], meta["pair_id"], meta["role"]
        if origin == "new":
            row = new_bank.get(qid)
            qa, ca = c_qwen.get(qid), c_claude.get(qid)
        else:
            row = parent_bank.get(qid)
            qa, ca = p_qwen.get(qid), p_claude.get(qid)
        question = R.resolve_query_text(row) if isinstance(row, dict) else None
        out[qid] = {
            "pair_id": pid,
            "role": role,
            "origin": origin,
            "question": question,
            "qwen_answer": qa,
            "claude_answer": ca,
        }
    logger.info("[texts] loaded %d captured china queries", len(out))
    return out


def _china_corpus_tfidf(texts: dict[str, dict]) -> dict[str, float]:
    """Per-query Qwen-vs-Claude TF-idf cosine, fit uniformly over the china corpus
    (TfidfVectorizer(max_features=10000); run_952 recipe, china-only corpus). Returns
    {query_id: cos}. Text enters only the vectorizer; values are never logged."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    qids = [
        q
        for q, t in texts.items()
        if isinstance(t.get("qwen_answer"), str) and isinstance(t.get("claude_answer"), str)
    ]
    if not qids:
        return {}
    tfidf = TfidfVectorizer(max_features=10000)
    mat = tfidf.fit_transform(
        [texts[q]["qwen_answer"] for q in qids] + [texts[q]["claude_answer"] for q in qids]
    )
    n = len(qids)
    cos = cosine_similarity(mat[:n], mat[n:]).diagonal()
    logger.info("[tfidf] china-corpus cosine over %d queries", n)
    return {q: float(c) for q, c in zip(qids, cos, strict=True)}


# ── arm-matched per-pair drops (re-fit; the one cell absent from committed JSONs) ─


def _refit_china_arm_matched() -> tuple[list[dict], list[dict]]:
    """Reproduce the committed arm-matched fit (own + plain maps) and return
    (china_rows, comm_rows) each with per-pair drop_own / drop_ext_plain / d. This
    is the committed china_included_stats Fit-A path, verbatim."""
    from explore_persona_space.experiments.issue_952.ridge_battery import run_ridge_cell

    # stage tensors
    for f in (
        "slots_own_L20.pt",
        "slots_ext_plain_L20.pt",
        "per_context_stats.npz",
        "slots_bank_own_L20.pt",
        "slots_bank_ext_plain_L20.pt",
        "spans_own.json",
        "spans_ext_plain.json",
        "spans_ext_style.json",
        "spans_mismatch.json",
    ):
        _stage(PARENT_PREFIX, PARENT_REV, f)
    for a in BANK_ARMS:
        _stage(CHINA_PREFIX, CHINA_REV, f"slots_bank_china_politics_topup_{a}_L20.pt")
    dl = STAGE / PARENT_REV[:10]
    cdl = STAGE / CHINA_REV[:10]

    own_pool, pool_ids = _load_slots(dl / "slots_own_L20.pt")
    plain_pool, _ = _load_slots(dl / "slots_ext_plain_L20.pt")
    split = json.loads((BASE / "eval_results/issue_952/split_seed952.json").read_text())
    spans = {
        a: {str(k): v for k, v in json.loads((dl / f"spans_{a}.json").read_text()).items()}
        for a in ARMS
    }
    spans_arr = {
        a: np.asarray([spans[a][str(c)].get("span", 0) for c in pool_ids], dtype=np.int64)
        for a in ARMS
    }
    u_a = np.all(np.stack([spans_arr[a] >= 32 for a in ARMS]), axis=0)
    pos_of = {c: i for i, c in enumerate(pool_ids)}
    tr_pos = np.asarray([pos_of[c] for c in split["train"] if c in pos_of])
    tr_a = tr_pos[u_a[tr_pos]]
    logger.info("U_A=%d tr_a=%d", int(u_a.sum()), len(tr_a))

    npz = dict(np.load(str(dl / "per_context_stats.npz"), allow_pickle=False))
    group2lam = dict(zip(npz["A_group_names"].tolist(), npz["A_lam_idx"].tolist(), strict=True))

    comm_verif = json.loads(
        (BASE / "eval_results/issue_952/divergence_bank_verification.json").read_text()
    )
    comm_kept = set(comm_verif["kept_pairs"])
    comm_pairs = [
        (p["pair_id"], p["category"], p["divergent"]["query_id"], p["control"]["query_id"])
        for p in comm_verif["pairs"]
        if p["pair_id"] in comm_kept
        and isinstance(p.get("divergent"), dict)
        and isinstance(p.get("control"), dict)
    ]
    china_verif = json.loads(
        (
            BASE / "eval_results/issue_952/china-politics-topup/summaries/"
            "china_topup_verification.json"
        ).read_text()
    )
    china_kept_ids = china_verif["final_china_kept_pairs"]
    china_pairs = [(pid, CHINA_CAT, f"{pid}_div", f"{pid}_ctl") for pid in china_kept_ids]

    comm_bank = {a: _load_slots(dl / f"slots_bank_{a}_L20.pt") for a in BANK_ARMS}
    china_bank = {
        a: _load_slots(cdl / f"slots_bank_china_politics_topup_{a}_L20.pt") for a in BANK_ARMS
    }
    china_ids = china_bank["own"][1]
    comm_div_rows, comm_ctl_rows, comm_div_i2r, comm_ctl_i2r = _role_split(comm_bank["own"][1])
    ch_div_rows, ch_ctl_rows, ch_div_i2r, ch_ctl_i2r = _role_split(china_ids)

    slots_by_arm = {"own": own_pool, "ext_plain": plain_pool}
    c_last_tr = own_pool[tr_a][:, SLOT_IDX["c_last"], :].astype(np.float64)

    def _x_bank(bank, rows):
        return bank["own"][0][rows][:, SLOT_IDX["c_last"], :].astype(np.float64)

    def _fit_eval(fit_arms, evals):
        Ytr, gnames = _stack_answer_targets(slots_by_arm, tr_a, fit_arms)
        eval_splits = {}
        for name, (xb, tgt_bank, rows) in evals.items():
            tgt, _g = _stack_answer_targets(tgt_bank, np.asarray(rows), fit_arms)
            eval_splits[name] = (xb, tgt)
        res = run_ridge_cell(
            c_last_tr,
            Ytr,
            eval_splits,
            group_names=gnames,
            device="cpu",
            allow_train_nan_imputation=True,
        )
        lam_idx = np.asarray([group2lam[f"{g.split('|')[0]}|own"] for g in gnames], dtype=np.int64)
        out = {}
        for name in evals:
            ssr = np.take_along_axis(res.ss_res[name], lam_idx[None, :, None], axis=2)[:, :, 0]
            sst = res.ss_tot[name].astype(np.float64)
            out[name] = {
                arm: _per_query_r2(ssr.astype(np.float64), sst, gnames, arm) for arm in fit_arms
            }
        return out

    A = _fit_eval(
        ("own", "ext_plain"),
        {
            "comm_div": (
                _x_bank(comm_bank, comm_div_rows),
                {a: comm_bank[a][0] for a in BANK_ARMS},
                comm_div_rows,
            ),
            "comm_ctl": (
                _x_bank(comm_bank, comm_ctl_rows),
                {a: comm_bank[a][0] for a in BANK_ARMS},
                comm_ctl_rows,
            ),
            "china_div": (
                _x_bank(china_bank, ch_div_rows),
                {a: china_bank[a][0] for a in BANK_ARMS},
                ch_div_rows,
            ),
            "china_ctl": (
                _x_bank(china_bank, ch_ctl_rows),
                {a: china_bank[a][0] for a in BANK_ARMS},
                ch_ctl_rows,
            ),
        },
    )
    comm_rows = _pairwise_drops(
        A["comm_div"], A["comm_ctl"], comm_pairs, comm_div_i2r, comm_ctl_i2r
    )
    china_rows = _pairwise_drops(
        A["china_div"], A["china_ctl"], china_pairs, ch_div_i2r, ch_ctl_i2r
    )
    for r in comm_rows + china_rows:
        r["d"] = r["drop_ext_plain"] - r["drop_own"]
    return china_rows, comm_rows


# ── subset statistics ─────────────────────────────────────────────────────────


def _cell_stats(rows: list[dict], value_key: str, r2_keys: tuple[str, str]) -> dict:
    """Pooled mean + 95% boot CI (10k, seed 0) + 10k sign-flip p (seed 1) + n, plus
    mean per-query R2 levels divergent-vs-control (r2_keys = (div_key, ctl_key))."""
    vals = np.asarray([r[value_key] for r in rows], dtype=np.float64)
    if len(vals) == 0:
        return {"n": 0}
    boot = _bank_boot(vals, N_DRAWS)
    sf = _signflip_p(vals, N_DRAWS)
    div_key, ctl_key = r2_keys
    return {
        "n": len(vals),
        "mean": boot["mean"],
        "mean_ci95": boot["mean_ci95"],
        "median": boot["median"],
        "median_ci95": boot["median_ci95"],
        "sign_flip_p_one_sided": sf["p_one_sided"],
        "sign_flip_null_band_hi_97p5": sf["null_band_hi_97p5"],
        "mean_r2_div": float(np.mean([r[div_key] for r in rows])),
        "mean_r2_ctl": float(np.mean([r[ctl_key] for r in rows])),
    }


def _arm_matched_subset(rows: list[dict]) -> dict:
    """Arm-matched cell over a subset. d = drop_ext_plain - drop_own; report R2
    levels for BOTH arms (own map x own target; plain map x plain target)."""
    vals = np.asarray([r["d"] for r in rows], dtype=np.float64)
    if len(vals) == 0:
        return {"n": 0}
    boot = _bank_boot(vals, N_DRAWS)
    sf = _signflip_p(vals, N_DRAWS)
    return {
        "n": len(vals),
        "mean_d": boot["mean"],
        "mean_d_ci95": boot["mean_ci95"],
        "median_d": boot["median"],
        "median_d_ci95": boot["median_ci95"],
        "sign_flip_p_one_sided": sf["p_one_sided"],
        "sign_flip_null_band_hi_97p5": sf["null_band_hi_97p5"],
        "mean_drop_own": float(np.mean([r["drop_own"] for r in rows])),
        "mean_drop_ext_plain": float(np.mean([r["drop_ext_plain"] for r in rows])),
        "mean_r2_div_own": float(np.mean([r["r2_div_own"] for r in rows])),
        "mean_r2_ctl_own": float(np.mean([r["r2_ctl_own"] for r in rows])),
        "mean_r2_div_ext_plain": float(np.mean([r["r2_div_ext_plain"] for r in rows])),
        "mean_r2_ctl_ext_plain": float(np.mean([r["r2_ctl_ext_plain"] for r in rows])),
    }


def _subset_block(pids: list[str], am_by_pid, cross_by_pid, sym_by_pid) -> dict:
    am = [am_by_pid[p] for p in pids if p in am_by_pid]
    cx = [cross_by_pid[p] for p in pids if p in cross_by_pid]
    sy = [sym_by_pid[p] for p in pids if p in sym_by_pid]
    return {
        "pair_ids": sorted(pids),
        "n": len(pids),
        "arm_matched": _arm_matched_subset(am),
        "cross_own_map_x_claude_target": _cell_stats(cx, "drop", ("r2_div", "r2_ctl")),
        "symmetric_claude_map_x_own_target": _cell_stats(sy, "drop", ("r2_div", "r2_ctl")),
    }


def main() -> None:
    torch.set_num_threads(8)
    t0 = time.time()
    out_dir = BASE / "eval_results" / "issue_952" / "refusal_sanity_check"
    out_dir.mkdir(parents=True, exist_ok=True)

    # committed cells (gate targets + per-pair cross / symmetric rows)
    committed = json.loads(
        (BASE / "eval_results/issue_952/china-politics-topup/stats_china_included.json").read_text()
    )
    c_am = committed["china_arm_matched"]
    c_cross = committed["china_cross_own_map_x_plain_target"]
    c_sym = committed["china_symmetric_plain_map_x_own_target"]

    # ── re-fit the arm-matched per-pair drops (the one missing cell) ────────────
    china_rows, comm_rows = _refit_china_arm_matched()
    am_by_pid = {r["pair_id"]: r for r in china_rows}
    cross_by_pid = {r["pair_id"]: r for r in c_cross["rows"]}
    sym_by_pid = {r["pair_id"]: r for r in c_sym["rows"]}

    # ── REPRODUCTION GATE (before any subsetting) ───────────────────────────────
    d_all = np.asarray([r["d"] for r in china_rows])
    repro_am_mean = float(d_all.mean())
    gate_am = (
        abs(repro_am_mean - c_am["headline_d"]["mean"]) < REPRO_TOL
        and len(china_rows) == c_am["n_pairs"]
    )
    # machinery sanity: recomputed committed 41-pair arm-matched d vs the committed cell
    cc = json.loads(
        (BASE / "eval_results/issue_952/divergence_transfer_cell/cross_cell.json").read_text()
    )
    comm_committed_d = cc["committed_arm_matched"]["headline_d_mean"]
    repro_comm_mean = float(np.mean([r["d"] for r in comm_rows]))
    gate_comm = abs(repro_comm_mean - comm_committed_d) < REPRO_TOL
    # cross / symmetric: re-aggregate committed rows -> must equal committed pooled means
    repro_cross = float(np.mean([r["drop"] for r in c_cross["rows"]]))
    repro_sym = float(np.mean([r["drop"] for r in c_sym["rows"]]))
    gate_cross = abs(repro_cross - c_cross["headline_drop"]["mean"]) < REPRO_TOL
    gate_sym = abs(repro_sym - c_sym["headline_drop"]["mean"]) < REPRO_TOL
    repro = {
        "committed_41pair_arm_matched": {
            "pass": bool(gate_comm),
            "recomputed_mean_d": repro_comm_mean,
            "committed_mean_d": comm_committed_d,
            "recomputed_n": len(comm_rows),
        },
        "arm_matched": {
            "pass": bool(gate_am),
            "recomputed_mean_d": repro_am_mean,
            "committed_mean_d": c_am["headline_d"]["mean"],
            "recomputed_n": len(china_rows),
            "committed_n": c_am["n_pairs"],
            "tol": REPRO_TOL,
        },
        "cross": {
            "pass": bool(gate_cross),
            "recomputed_mean_drop": repro_cross,
            "committed_mean_drop": c_cross["headline_drop"]["mean"],
        },
        "symmetric": {
            "pass": bool(gate_sym),
            "recomputed_mean_drop": repro_sym,
            "committed_mean_drop": c_sym["headline_drop"]["mean"],
        },
    }
    if not (gate_am and gate_cross and gate_sym and gate_comm):
        raise SystemExit(f"REPRODUCTION GATE FAILED: {json.dumps(repro, indent=2)}")
    logger.info("[gate] committed-41 + china arm-matched/cross/symmetric reproduction PASS")

    # ── behavior labels + membership over the 31 kept china pairs ───────────────
    kept = list(am_by_pid)  # exactly the 31 kept china pairs (fit + gate confirmed)
    pv = json.loads((BASE / "eval_results/issue_952/divergence_bank_verification.json").read_text())
    cv = json.loads(
        (
            BASE / "eval_results/issue_952/china-politics-topup/summaries/"
            "china_topup_verification.json"
        ).read_text()
    )
    parent_china = {p["pair_id"]: p for p in pv["pairs"] if p.get("category") == CHINA_CAT}
    new_china = {p["pair_id"]: p for p in cv["pairs"]}

    def _rec(pid, role):
        return new_china[pid][role] if pid in new_china else parent_china[pid][role]

    def _origin(pid):
        return "new" if pid in new_china else "parent"

    # china-corpus TF-idf (uniform; committed parent-corpus value reported alongside)
    texts = load_china_texts()
    tfidf_china = _china_corpus_tfidf(texts)

    labels = {}
    for pid in kept:
        d, c = _rec(pid, "divergent"), _rec(pid, "control")
        div_qid = d["query_id"]
        committed_tfidf = d.get("tfidf_cos")  # parent-corpus; None for new pairs
        labels[pid] = {
            "origin": _origin(pid),
            "divergence_div": d.get("divergence"),
            "divergence_ctl": c.get("divergence"),
            "refusal_qwen_div": d.get("refusal_qwen"),
            "refusal_claude_div": d.get("refusal_claude"),
            "refusal_qwen_ctl": c.get("refusal_qwen"),
            "refusal_claude_ctl": c.get("refusal_claude"),
            "tfidf_cos_div_china_corpus": tfidf_china.get(div_qid),
            "tfidf_cos_div_committed_parent_corpus": committed_tfidf,
        }

    # S1 refusal-mismatch: Qwen refuses (>=50) AND Claude answers (<50) on divergent
    s1 = [
        pid
        for pid in kept
        if (
            labels[pid]["refusal_qwen_div"] is not None
            and labels[pid]["refusal_claude_div"] is not None
            and labels[pid]["refusal_qwen_div"] >= REFUSAL_THR
            and labels[pid]["refusal_claude_div"] < REFUSAL_THR
        )
    ]
    not_s1 = [pid for pid in kept if pid not in s1]

    # S2 lexical-divergence: bottom tercile of divergent-query china-corpus tfidf_cos
    tf_vals = [
        (pid, labels[pid]["tfidf_cos_div_china_corpus"])
        for pid in kept
        if labels[pid]["tfidf_cos_div_china_corpus"] is not None
    ]
    tf_sorted = sorted(tf_vals, key=lambda x: x[1])
    n_tf = len(tf_sorted)
    k_tercile = max(1, round(n_tf / 3))
    s2 = [pid for pid, _ in tf_sorted[:k_tercile]]
    s2_threshold = float(tf_sorted[k_tercile - 1][1]) if n_tf else None
    s1_and_s2 = sorted(set(s1) & set(s2))

    logger.info(
        "[subset] kept=%d S1=%d NOT-S1=%d S2=%d (n_tf=%d, thr=%.4f) S1nS2=%d",
        len(kept),
        len(s1),
        len(not_s1),
        len(s2),
        n_tf,
        s2_threshold if s2_threshold is not None else float("nan"),
        len(s1_and_s2),
    )

    # ── assemble output ─────────────────────────────────────────────────────────
    out = {
        "description": (
            "EXPLORATORY conditional re-read of the #952 china-politics divergence "
            "bank restricted to behavior-differs pairs. Conditioning is on judge-text "
            "behavior labels (independent of the activation DV); NOT pre-registered; "
            "small n. Statistics on the 31 kept china pairs (comparable to the "
            "committed read); membership reported among the 42 captured pairs too."
        ),
        "layer": committed["layer"],
        "margin_bank_drop_difference": committed["margin"],
        "n_draws": N_DRAWS,
        "bootstrap_seed": 0,
        "sign_flip_seed": 1,
        "refusal_threshold": REFUSAL_THR,
        "subset_definitions": {
            "S1_refusal_mismatch_PRIMARY": (
                "divergent-query refusal_qwen >= 50 AND refusal_claude < 50 "
                "(Qwen refuses, Claude answers)"
            ),
            "S2_lexical_divergence_SECONDARY": (
                "divergent-query china-corpus TF-idf cosine in the bottom tercile among "
                "kept china pairs (lower cosine = more lexically divergent); descriptive"
            ),
            "complement": (
                "NOT-S1 (behavior-similar): kept china pairs where Qwen does NOT "
                "refuse-while-Claude-answers"
            ),
        },
        "reproduction_gate": repro,
        "committed_pooled_reads": {
            "china_31_arm_matched": {
                "n": c_am["n_pairs"],
                "mean_d": c_am["headline_d"]["mean"],
                "mean_d_ci95": c_am["headline_d"]["mean_ci95"],
                "sign_flip_p": c_am["sign_flip"]["p_one_sided"],
                "mean_drop_own": c_am["mean_drop_own"],
                "mean_drop_ext_plain": c_am["mean_drop_ext_plain"],
            },
            "china_31_cross": {
                "n": c_cross["n_pairs"],
                "mean_drop": c_cross["headline_drop"]["mean"],
                "mean_drop_ci95": c_cross["headline_drop"]["mean_ci95"],
                "sign_flip_p": c_cross["sign_flip"]["p_one_sided"],
                "mean_r2_div": c_cross["mean_r2_div"],
                "mean_r2_ctl": c_cross["mean_r2_ctl"],
            },
            "china_31_symmetric": {
                "n": c_sym["n_pairs"],
                "mean_drop": c_sym["headline_drop"]["mean"],
                "mean_drop_ci95": c_sym["headline_drop"]["mean_ci95"],
                "sign_flip_p": c_sym["sign_flip"]["p_one_sided"],
                "mean_r2_div": c_sym["mean_r2_div"],
                "mean_r2_ctl": c_sym["mean_r2_ctl"],
            },
        },
        "membership": {
            "n_kept": len(kept),
            "n_captured": len(texts) // 2,
            "S1_refusal_mismatch": sorted(s1),
            "NOT_S1_behavior_similar": sorted(not_s1),
            "S2_lexical_divergence": sorted(s2),
            "S1_and_S2_overlap": s1_and_s2,
            "n_S1": len(s1),
            "n_NOT_S1": len(not_s1),
            "n_S2": len(s2),
            "n_S1_and_S2": len(s1_and_s2),
            "S2_tercile_threshold_china_corpus_tfidf": s2_threshold,
            "S2_n_with_tfidf": n_tf,
            "tfidf_cos_div_distribution_china_corpus": sorted([round(v, 5) for _, v in tf_vals]),
        },
        "subsets": {
            "S1_refusal_mismatch": _subset_block(s1, am_by_pid, cross_by_pid, sym_by_pid),
            "NOT_S1_behavior_similar": _subset_block(not_s1, am_by_pid, cross_by_pid, sym_by_pid),
            "S2_lexical_divergence": _subset_block(s2, am_by_pid, cross_by_pid, sym_by_pid),
            "all_31_kept": _subset_block(kept, am_by_pid, cross_by_pid, sym_by_pid),
        },
        "per_pair": {
            pid: {
                **labels[pid],
                "arm_matched_drop_own": am_by_pid[pid]["drop_own"],
                "arm_matched_drop_ext_plain": am_by_pid[pid]["drop_ext_plain"],
                "arm_matched_d": am_by_pid[pid]["d"],
                "cross_drop": cross_by_pid[pid]["drop"] if pid in cross_by_pid else None,
                "cross_r2_div": cross_by_pid[pid]["r2_div"] if pid in cross_by_pid else None,
                "cross_r2_ctl": cross_by_pid[pid]["r2_ctl"] if pid in cross_by_pid else None,
                "symmetric_drop": sym_by_pid[pid]["drop"] if pid in sym_by_pid else None,
                "in_S1": pid in s1,
                "in_S2": pid in s2,
            }
            for pid in kept
        },
        "provenance": {
            "git_commit": _sha(),
            "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "parent_revision": PARENT_REV,
            "china_revision": CHINA_REV,
            "source_files": [
                "eval_results/issue_952/china-politics-topup/stats_china_included.json",
                "eval_results/issue_952/divergence_bank_verification.json",
                "eval_results/issue_952/china-politics-topup/summaries/china_topup_verification.json",
                "eval_results/issue_952/china-politics-topup/summaries/provenance_china_politics_topup.json",
                f"HF:{REPO}/{PARENT_PREFIX} @ {PARENT_REV}",
                f"HF:{REPO}/{CHINA_PREFIX} @ {CHINA_REV}",
            ],
            "machinery": (
                "issue952_china_included_stats + issue952_divergence_transfer_cell "
                "(verbatim fit + stats)"
            ),
            "framing": (
                "exploratory conditional read; behavior-label conditioning; "
                "not pre-registered; small n"
            ),
            "wall_seconds": round(time.time() - t0, 1),
        },
    }
    out_path = out_dir / "behavior_differs_subset.json"
    out_path.write_text(json.dumps(out, indent=2))
    logger.info("[done] wrote %s (%.1fs)", out_path, time.time() - t0)


if __name__ == "__main__":
    main()
