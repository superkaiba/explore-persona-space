#!/usr/bin/env python3
"""Issue #2202 inline free-analysis round ``residual-read`` (user-chat carve-out).

Part 1 — residual-failure read: re-run the draw-averaged covered-row batteries
for ``csls_k10_whitencos`` and ``csls_pen_whitencos_g10`` (ridge predictions),
persist per-row ranks + context ids, and for every rank>1 row join the banked
judge labels (language / topic / request_refusal_adjacent / answer_is_refusal /
format from eval_results/issue_1738/judge_labels/labels.json; depth from the
banked ci_fields.json), the resample attribution class, and the TWIN TEST:
whitened-space cosine between the true draw-averaged answer and the retrieved
rank-1 competitor (high = target degeneracy, not map error). Text snippets
(<=200 chars, sanitized, from the LOCAL #1482 judge_texts.jsonl cache via a
line-scan on the needed ids — never paging the whole file into context) are
attached ONLY for benign-labeled rows; refusal-adjacent / nsfw-topic rows are
characterized by labels + twin cosine only.

Part 2 — differentiation metrics on ridge for four conventions
(raw_euclidean, whiten_cos, csls_k10_whitencos, csls_pen_whitencos_g10),
single-draw AND draw-averaged: per-row margin (the score/distance gap the true
answer wins or loses by), mean/median/p10/p90 split by success vs failure,
MRR, median rank, and pairwise AUC = 1 - (mean_rank - 1)/(n_pool - 1).
Margins are in each convention's own units: CSLS margins in whitened-cosine
score units (score_true - best competitor score); whiten_cos in cosine-
distance units; raw_euclidean in SQUARED-euclidean distance units (the
``_pairwise_dist`` convention) — positive always means the true target wins.

Analysis-only; vectorized batteries; pool-side CSLS statistics recomputed on
the modified pool for the draw-averaged cells (query bank = the ridge map's
full 9,941 predictions), exactly as in the avgtgt-completion round.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # shared-VM thread caps must bind BEFORE numpy/torch import (#847)

import issue1738_characterize as CH  # noqa: E402
import issue2202_failchar as FC  # noqa: E402
import issue2202_metric_zoo as MZ  # noqa: E402
import numpy as np  # noqa: E402
from scipy.linalg import solve_triangular  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

FW_STAGE = Path("/mnt/eps-data/thomasjiralerspong/issue2202_freshwhiten")  # read-only reuse
STAGE = Path("/mnt/eps-data/thomasjiralerspong/issue2202_avgtgt")  # this line's own staging
OUT_DIR = PROJECT_ROOT / "eval_results" / "issue_2202" / "residual_read"
LABELS_1738 = PROJECT_ROOT / "eval_results" / "issue_1738" / "judge_labels" / "labels.json"
TEXT_CACHE = PROJECT_ROOT / "data" / "issue_1482" / "context_extremes_scratch" / "judge_texts.jsonl"
K_DRAWS = 4
N_COVERED = 1_988
SNIPPET_CAP = 200
TWIN_DEGENERACY_THRESHOLD = 0.95  # whitened cosine above this reads as near-duplicate target
NONBENIGN_TOPICS = {"nsfw", "harmful", "violence", "self-harm", "sexual", "adult"}
# expected avg-target acc@1 from the avgtgt-completion round (exact-reproduction gate)
EXPECTED_AVG_ACC1 = {
    "csls_k10_whitencos": 0.994466800804829,
    "csls_pen_whitencos_g10": 0.9949698189134809,
}
CSLS_GAMMAS = {"csls_k10_whitencos": 0.5, "csls_pen_whitencos_g10": 1.0}


def is_benign(lab: dict | None) -> bool:
    """Text-snippet gate: labeled, non-refusal, non-refusal-adjacent, and the
    topic is outside the nsfw/harm family (missing label => NOT benign)."""
    if not lab:
        return False
    topic = str(lab.get("topic", "")).lower()
    return (
        lab.get("answer_is_refusal") == "no"
        and lab.get("request_refusal_adjacent") == "no"
        and topic not in NONBENIGN_TOPICS
        and "harm" not in topic
    )


def sanitize(s: str, cap: int = SNIPPET_CAP) -> str:
    """Whitespace-collapsed excerpt with inline truncation disclosure."""
    out = " ".join(str(s).split())
    return out if len(out) <= cap else out[:cap] + " …[truncated]"


def load_texts_for(needed: set[int]) -> dict[int, dict]:
    """ci -> text row from the local #1482 cache; line-scan with a cheap
    substring probe so only the needed rows are ever parsed."""
    probes = {ci: f'{{"ci": {ci},' for ci in needed}
    out: dict[int, dict] = {}
    with open(TEXT_CACHE, encoding="utf-8") as f:
        for line in f:
            for ci, probe in probes.items():
                if ci not in out and line.startswith(probe):
                    out[ci] = json.loads(line)
                    break
            if len(out) == len(needed):
                break
    return out


def covered_battery(q: np.ndarray, pool: np.ndarray, pos: np.ndarray, metric: str, tag: str):
    """Covered-row distance battery: (ranks, margin, j_comp) with margin =
    d(best competitor) - d(true) (positive = win) and j_comp the nearest
    NON-TRUE pool item; chunked GEMMs (mid-rank tie convention)."""
    n = q.shape[0]
    ranks = np.empty(n)
    margin = np.empty(n)
    j_comp = np.empty(n, dtype=np.int64)
    t0 = time.time()
    chunk = 1024
    for k, s in enumerate(range(0, n, chunk)):
        e = min(n, s + chunk)
        d = FC._pairwise_dist(np.asarray(q[s:e], np.float64), pool, metric)
        ranks[s:e] = MZ.midranks_of_true(d, pos[s:e])
        dt = d[np.arange(e - s), pos[s:e]]
        d[np.arange(e - s), pos[s:e]] = np.inf
        jc = d.argmin(axis=1)
        margin[s:e] = d[np.arange(e - s), jc] - dt
        j_comp[s:e] = jc
        print(
            f"[{tag}] unit {k + 1}/{(n + chunk - 1) // chunk} elapsed={time.time() - t0:.1f}s",
            flush=True,
        )
    return ranks, margin, j_comp


def csls_covered(s_full: np.ndarray, pos: np.ndarray, gamma: float):
    """CSLS covered-row read from the FULL query-bank similarity matrix:
    score = S - gamma * r_j (r_j = mean top-K_LOCAL sims of pool item j over
    the full query bank); returns (ranks, margin, j_comp) on the covered rows."""
    n_q = s_full.shape[0]
    k = MZ.K_LOCAL
    r_p = np.partition(s_full, n_q - k, axis=0)[n_q - k :, :].mean(axis=0)
    score = s_full[pos] - gamma * r_p[None, :]
    ranks = MZ.ranks_score_matrix(score, pos)
    m = score.copy()
    st = m[np.arange(len(pos)), pos]
    m[np.arange(len(pos)), pos] = -np.inf
    jc = m.argmax(axis=1)
    margin = st - m[np.arange(len(pos)), jc]
    return ranks, margin, jc


def stats_block(ranks: np.ndarray, margin: np.ndarray, n_pool: int) -> dict:
    """Part-2 differentiation metrics for one (convention, variant) cell."""
    ok = ranks <= 1.0

    def _q(x: np.ndarray) -> dict:
        if len(x) == 0:
            return {"n": 0}
        return {
            "n": int(len(x)),
            "mean": float(x.mean()),
            "median": float(np.median(x)),
            "p10": float(np.percentile(x, 10)),
            "p90": float(np.percentile(x, 90)),
        }

    return {
        "acc_at_1": float(ok.mean()),
        "mrr": float((1.0 / ranks).mean()),
        "median_rank": float(np.median(ranks)),
        "pairwise_auc": float(1.0 - (ranks.mean() - 1.0) / (n_pool - 1)),
        "margin_successes": _q(margin[ok]),
        "margin_failures": _q(margin[~ok]),
    }


def main() -> int:
    t0 = time.time()
    STAGE.mkdir(parents=True, exist_ok=True)
    hub.stage_hub_file(
        FC.C.HF_DATA_REPO,
        f"{FC.HF_PREFIX_2202}/analysis_tensors/ci_fields.json",
        STAGE / "ci_fields.json",
    )

    pd_ = np.load(FW_STAGE / "pred16.npz")
    yd = np.load(FW_STAGE / "y_holdout_L19.npz")
    pred = pd_["pred16"].astype(np.float64)
    y16 = yd["y16"].astype(np.float64)
    pci = np.asarray(pd_["ci"], dtype=np.int64)
    assert (pci == np.asarray(yd["ci"], dtype=np.int64)).all()
    n_pool = y16.shape[0]

    kns = SimpleNamespace(
        local_kresample_dir=str(FW_STAGE / "kresample"),
        scratch=str(STAGE / "scratch"),
        hf_prefix="",
    )
    kci, vres = CH._load_kresample_v(kns, [FC.LAYER])
    draws = vres[:, :, 0, :].astype(np.float64)
    pos_of = {int(c): p for p, c in enumerate(pci.tolist())}
    pos = np.asarray([pos_of[int(c)] for c in kci], dtype=np.int64)
    kz = np.load(FW_STAGE / "kresample_ranks.npz")
    assert (np.asarray(kz["ci"], dtype=np.int64) == kci).all()
    kres_s = np.asarray(kz["s"], dtype=np.float64)
    kres_cls = [str(x) for x in FC.kres_classes(kres_s)]

    wz = np.load(FW_STAGE / "whiten_stats.npz")
    mu_a = np.asarray(wz["mu_A"], dtype=np.float64)
    ell = np.asarray(wz["L"], dtype=np.float64)

    def _wh(x: np.ndarray) -> np.ndarray:
        return solve_triangular(ell, (np.asarray(x, np.float64) - mu_a).T, lower=True).T

    labels = json.loads(LABELS_1738.read_text())["labels"]
    ci_fields = json.loads((STAGE / "ci_fields.json").read_text())["fields"]

    avg = (y16[pos] + draws.sum(axis=1)) / (1 + K_DRAWS)
    pool_mod = y16.copy()
    pool_mod[pos] = avg
    y16w = _wh(y16)
    pool_modw = y16w.copy()
    pool_modw[pos] = _wh(avg)
    predw = _wh(pred)
    pwn = predw / (np.linalg.norm(predw, axis=1, keepdims=True) + 1e-12)
    qwn = {
        "single": y16w / (np.linalg.norm(y16w, axis=1, keepdims=True) + 1e-12),
        "avg": pool_modw / (np.linalg.norm(pool_modw, axis=1, keepdims=True) + 1e-12),
    }
    pool_raw = {"single": y16, "avg": pool_mod}
    pool_w = {"single": y16w, "avg": pool_modw}

    # ── batteries: 4 conventions × 2 variants (ridge) ──
    per_cell: dict[str, dict[str, dict]] = {}
    row_store: dict[str, np.ndarray] = {"ci": kci.astype(np.int64)}
    csls_comp: dict[tuple[str, str], np.ndarray] = {}
    predc = pred[pos]
    predcw = predw[pos]
    for variant in ("single", "avg"):
        r, m, _ = covered_battery(predc, pool_raw[variant], pos, "euclidean", f"raw-{variant}")
        per_cell.setdefault("raw_euclidean", {})[variant] = stats_block(r, m, n_pool)
        row_store[f"rank_raw_euclidean_{variant}"], row_store[f"margin_raw_euclidean_{variant}"] = (
            r,
            m,
        )
        r, m, _ = covered_battery(predcw, pool_w[variant], pos, "cosine", f"wcos-{variant}")
        per_cell.setdefault("whiten_cos", {})[variant] = stats_block(r, m, n_pool)
        row_store[f"rank_whiten_cos_{variant}"], row_store[f"margin_whiten_cos_{variant}"] = r, m
        t1 = time.time()
        s_full = pwn @ qwn[variant].T
        print(f"[swc-{variant}] S ({n_pool}x{n_pool}) in {time.time() - t1:.1f}s", flush=True)
        for conv, gamma in CSLS_GAMMAS.items():
            r, m, jc = csls_covered(s_full, pos, gamma)
            per_cell.setdefault(conv, {})[variant] = stats_block(r, m, n_pool)
            row_store[f"rank_{conv}_{variant}"], row_store[f"margin_{conv}_{variant}"] = r, m
            row_store[f"comp_{conv}_{variant}"] = jc.astype(np.int64)
            csls_comp[(conv, variant)] = jc
        del s_full

    # exact-reproduction gate vs the avgtgt-completion round
    for conv, expected in EXPECTED_AVG_ACC1.items():
        got = per_cell[conv]["avg"]["acc_at_1"]
        assert abs(got - expected) < 1e-9, (conv, got, expected)

    # ── Part 1: residual-failure table (avg-target csls cells) ──
    fail_mask = {conv: row_store[f"rank_{conv}_avg"] > 1.0 for conv in CSLS_GAMMAS}
    fail_rows = np.nonzero(fail_mask["csls_k10_whitencos"] | fail_mask["csls_pen_whitencos_g10"])[0]
    needed_text_cis: set[int] = set()
    table: list[dict] = []
    twn = qwn["avg"]  # whitened+normalized modified pool (twin cosine basis)
    for i in fail_rows:
        ci = int(kci[i])
        lab = labels.get(str(ci))
        # twin: retrieved rank-1 competitor under the convention this row FAILS in
        # (csls_k10 primary; csls_pen when only that one fails)
        conv0 = (
            "csls_k10_whitencos" if fail_mask["csls_k10_whitencos"][i] else "csls_pen_whitencos_g10"
        )
        jc = int(csls_comp[(conv0, "avg")][i])
        twin_cos = float(twn[pos[i]] @ twn[jc])
        benign = is_benign(lab)
        if benign:
            needed_text_cis.add(ci)
        if twin_cos >= TWIN_DEGENERACY_THRESHOLD:
            read = f"target degeneracy — near-duplicate answers (twin cos {twin_cos:.3f})"
        elif lab and lab.get("answer_is_refusal") != "no":
            read = "refusal/boilerplate target class"
        else:
            read = f"map-error candidate (twin cos {twin_cos:.3f} < {TWIN_DEGENERACY_THRESHOLD})"
        table.append(
            {
                "ci": ci,
                "rank_csls_k10": float(row_store["rank_csls_k10_whitencos_avg"][i]),
                "rank_csls_pen_g10": float(row_store["rank_csls_pen_whitencos_g10_avg"][i]),
                "margin_csls_k10": float(row_store["margin_csls_k10_whitencos_avg"][i]),
                "twin_cosine_whitened": twin_cos,
                "competitor_ci": int(pci[jc]),
                "labels": lab or "UNLABELED",
                "depth": (ci_fields.get(str(ci)) or {}).get("depth"),
                "depth_band": (ci_fields.get(str(ci)) or {}).get("depth_band"),
                "attribution": kres_cls[i],
                "kres_s": float(kres_s[i]),
                "benign_for_text": benign,
                "read": read,
            }
        )
    # ── Addendum: WORST-DISCRIMINATED bottom-50 by csls_k10 avg-target margin ──
    m_k10 = row_store["margin_csls_k10_whitencos_avg"]
    bottom = np.argsort(m_k10)[:50]
    wd_rows: list[dict] = []
    for i in bottom:
        ci = int(kci[i])
        lab = labels.get(str(ci))
        jc = int(csls_comp[("csls_k10_whitencos", "avg")][i])
        benign = is_benign(lab)
        if benign:
            needed_text_cis.add(ci)
        wd_rows.append(
            {
                "ci": ci,
                "margin_csls_k10": float(m_k10[i]),
                "margin_whiten_cos": float(row_store["margin_whiten_cos_avg"][i]),
                "rank_csls_k10": float(row_store["rank_csls_k10_whitencos_avg"][i]),
                "is_failure": bool(row_store["rank_csls_k10_whitencos_avg"][i] > 1.0),
                "twin_cosine_whitened": float(twn[pos[i]] @ twn[jc]),
                "competitor_ci": int(pci[jc]),
                "labels": lab or "UNLABELED",
                "depth": (ci_fields.get(str(ci)) or {}).get("depth"),
                "depth_band": (ci_fields.get(str(ci)) or {}).get("depth_band"),
                "attribution": kres_cls[i],
                "benign_for_text": benign,
            }
        )

    def composition(rows_idx: np.ndarray) -> dict:
        """Per-label counts + shares for a covered-row index set (descriptive)."""
        from collections import Counter

        fields = ("topic", "language", "answer_is_refusal", "request_refusal_adjacent")
        counters: dict[str, Counter] = {f: Counter() for f in (*fields, "depth_band")}
        for i in rows_idx:
            lab = labels.get(str(int(kci[i]))) or {}
            for f in fields:
                counters[f][str(lab.get(f, "UNLABELED"))] += 1
            counters["depth_band"][
                str((ci_fields.get(str(int(kci[i]))) or {}).get("depth_band", "?"))
            ] += 1
        n = max(1, len(rows_idx))
        return {
            f: {k: {"n": int(v), "share": round(v / n, 4)} for k, v in c.most_common()}
            for f, c in counters.items()
        }

    raweuc_fail_idx = np.nonzero(row_store["rank_raw_euclidean_single"] > 1.0)[0]
    comp_block = {
        "bottom50": composition(bottom),
        "full_covered_pool_n1988": composition(np.arange(len(kci))),
        "raw_euclidean_single_failures": composition(raweuc_fail_idx),
        "n_raw_euclidean_single_failures": int(len(raweuc_fail_idx)),
    }

    texts = load_texts_for(needed_text_cis) if needed_text_cis else {}
    for row in [*table, *wd_rows]:
        if row["benign_for_text"] and row["ci"] in texts:
            tr = texts[row["ci"]]
            row["last_user_excerpt"] = sanitize(tr.get("last_user", ""), 120)
            row["response_excerpt"] = sanitize(tr.get("response", ""))
    table.sort(key=lambda r: -max(r["rank_csls_k10"], r["rank_csls_pen_g10"]))

    n_deg = sum(1 for r in table if r["twin_cosine_whitened"] >= TWIN_DEGENERACY_THRESHOLD)
    summary = {
        "round": "residual-read (user-chat inline free-analysis, task #2202)",
        "conventions_note": (
            "ridge predictions; covered-row eval (n=1,988; pool 9,941); avg = draw-averaged "
            "targets (mean of original + 4 fresh draws); CSLS pool statistics recomputed on the "
            "modified pool from the full 9,941-prediction query bank (K_LOCAL="
            f"{MZ.K_LOCAL}); margins in each convention's own units — CSLS: whitened-cos score "
            "gap, whiten_cos: cosine-distance gap, raw_euclidean: SQUARED-euclidean gap; "
            "positive margin = true target wins; twin test = whitened-space cosine between the "
            "true draw-averaged answer and the retrieved rank-1 competitor (threshold "
            f"{TWIN_DEGENERACY_THRESHOLD} for the degeneracy read); text snippets only for "
            "benign-labeled rows (answer_is_refusal=no AND request_refusal_adjacent=no AND "
            "topic outside the nsfw/harm family), <=200 chars sanitized from the local #1482 "
            "judge_texts cache"
        ),
        "part1_residual_failures": {
            "n_failures_union": len(table),
            "n_fail_csls_k10": int(fail_mask["csls_k10_whitencos"].sum()),
            "n_fail_csls_pen_g10": int(fail_mask["csls_pen_whitencos_g10"].sum()),
            "n_target_degeneracy": n_deg,
            "n_map_error_candidates": len(table) - n_deg,
            "rows": table,
        },
        "worst_discriminated": {
            "note": (
                "bottom 50 covered rows by csls_k10_whitencos draw-averaged-target margin "
                "(NOT just rank>1 failures — barely-won successes included, flagged by "
                "is_failure); whiten_cos avg margin as companion column; twin cosine = whitened "
                "cosine to the csls_k10 runner-up on the modified pool; composition shares are "
                "DESCRIPTIVE (n=50, no significance battery), against the full 1,988-row "
                "covered pool and against the raw-euclidean SINGLE-draw failure set"
            ),
            "n_failures_in_bottom50": int(sum(r["is_failure"] for r in wd_rows)),
            "rows": wd_rows,
            "composition": comp_block,
        },
        "part2_differentiation": per_cell,
        "reproduction_gate": {
            conv: {"recomputed": per_cell[conv]["avg"]["acc_at_1"], "expected": exp}
            for conv, exp in EXPECTED_AVG_ACC1.items()
        },
        "meta": FC.meta_block({"wall_seconds": round(time.time() - t0, 1)}),
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(OUT_DIR / "percontext_ranks_margins.npz", **row_store)
    FC.atomic_json(OUT_DIR / "summary.json", summary)
    print(
        f"[done] {len(table)} residual failures ({n_deg} degeneracy); wrote {OUT_DIR} "
        f"in {time.time() - t0:.1f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
