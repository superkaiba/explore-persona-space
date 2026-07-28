#!/usr/bin/env python
"""Issue #1482 user-chat inline free-analysis round: feature-correlates.

Two per-feature correlate reads over the committed SAE->SAE per-feature
held-out R2 (`eval_results/issue_1482/sae_perfeature/sae_ctx__mean__ridge.npz`,
16,384 answer-side features):

Q2 (consistency): per-feature WITHIN-ANSWER consistency = mean `ans_frac`
    (fraction of answer tokens where the feature is active, conditional on
    being answer-active at all), accumulated over fit-tagged rows (set_tag==1;
    parity with the committed `activity` covariate) from the LOCAL pooled
    store staged by the shuffle-null round. Correlated with per-feature R2:
    Spearman, rank-partial given activity, and within-activity-decile reads.

Q1 (abstraction): stratified sample of 300 features (10 activity deciles x
    3 within-decile R2 terciles x 10), top-8 fit-row answers per feature by
    `ans_mean`, project judge labels the feature's level in {low, high,
    unclear} (low = surface/token/format property; high = semantic/abstract
    property) blind to R2, reason-then-label (max_tokens 400, llm-judging
    rule 23); 60-item test-retest kappa; per-level R2 contrast overall and
    within activity deciles.

Wiring gate: the recomputed answer-side activity over fit rows must match the
committed npz `activity` (max |delta| < 1e-3) before any read.

No model fits anywhere (rank correlations only). Vectorized: per-shard
np.bincount accumulation; no per-feature Python loop on the scan.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps + credentials BEFORE numpy (shared-VM discipline)

import numpy as np  # noqa: E402

DICT_SIZE = 131_072
STORE_DEFAULT = Path(
    "/mnt/eps-data/thomasjiralerspong/issue1482_shuffnull/"
    "issue1482_error_analysis/analysis_tensors/sae_pooled"
)
WORK_DEFAULT = Path("/mnt/eps-data/thomasjiralerspong/issue1482_featcorr")
PERFEATURE_NPZ = PROJECT_ROOT / "eval_results/issue_1482/sae_perfeature/sae_ctx__mean__ridge.npz"
OUT_EVAL = PROJECT_ROOT / "eval_results/issue_1482/feature_correlates"
OUT_FIGS = PROJECT_ROOT / "figures/issue_1482/feature_correlates"

SAMPLE_SEED = 14_822_026  # recorded sampling seed for the Q1 stratified draw
N_DECILES = 10
N_TERCILES = 3
N_PER_STRATUM = 10
TOP_K_CONTEXTS = 8
SNIPPET_CHARS = 400
RETEST_N = 60
JUDGE_MODEL = "claude-sonnet-4-5-20250929"  # CLAUDE.md project judge pin
JUDGE_MAX_TOKENS = 400  # reason-then-label (llm-judging rule 23)
JUDGE_SYSTEM = (
    "You label sparse-autoencoder features from example texts. Given up to 8 assistant "
    "answers where a feature fires strongly — and, when available, an independent "
    "auto-interp description of the feature derived from token-level max-activation "
    "dashboards on a different corpus (auxiliary evidence; the example answers are "
    "primary) — decide what shared property the feature detects and how abstract that "
    "property is. Reason briefly, then output ONLY JSON: "
    '{"reasoning": "...", "label": "<= 8 words naming the shared property", '
    '"level": "low" | "high" | "unclear"}. '
    "level=low: a surface/token-level property (a specific token or word, punctuation, "
    "formatting or markup, code syntax, a script/language's characters, boilerplate "
    "phrasing). level=high: a semantic or abstract property (topic, domain, task type, "
    "intent, discourse role, style, behavior such as refusal, sentiment). "
    "level=unclear: no coherent shared property is discernible from the examples."
)


def _log(msg: str) -> None:
    print(f"{time.strftime('%H:%M:%S')} [featcorr] {msg}", flush=True)


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.stats import spearmanr

    return float(spearmanr(a, b).statistic)


def _rank(a: np.ndarray) -> np.ndarray:
    from scipy.stats import rankdata

    return rankdata(a).astype(np.float64)


def _partial_spearman(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> float:
    """Rank-partial correlation of x and y given z (Pearson on rank residuals)."""
    rx, ry, rz = _rank(x), _rank(y), _rank(z)
    zc = np.column_stack([np.ones_like(rz), rz])
    bx, *_ = np.linalg.lstsq(zc, rx, rcond=None)
    by, *_ = np.linalg.lstsq(zc, ry, rcond=None)
    ex, ey = rx - zc @ bx, ry - zc @ by
    return float(np.corrcoef(ex, ey)[0, 1])


def _load_committed() -> dict[str, np.ndarray]:
    z = np.load(PERFEATURE_NPZ)
    feat_ids = np.asarray(z["feat_ids"], dtype=np.int64)
    r2 = np.asarray(z["r2"], dtype=np.float64)
    activity = np.asarray(z["activity"], dtype=np.float64)
    assert len(feat_ids) == 16_384, len(feat_ids)
    assert np.all(np.isfinite(r2)), "committed r2 has non-finite entries"
    return {"feat_ids": feat_ids, "r2": r2, "activity": activity}


def _decile_of(activity: np.ndarray) -> np.ndarray:
    """Activity decile (0..9) per feature, matching the body's decile framing."""
    edges = np.quantile(activity, np.linspace(0, 1, N_DECILES + 1)[1:-1])
    return np.searchsorted(edges, activity, side="right")


def _sample_features(com: dict[str, np.ndarray]) -> dict:
    """Stratified Q1 sample: activity decile x within-decile R2 tercile."""
    rng = np.random.default_rng(SAMPLE_SEED)
    dec = _decile_of(com["activity"])
    picks: list[dict] = []
    for d in range(N_DECILES):
        in_d = np.where(dec == d)[0]
        t_edges = np.quantile(com["r2"][in_d], [1 / 3, 2 / 3])
        ter = np.searchsorted(t_edges, com["r2"][in_d], side="right")
        for t in range(N_TERCILES):
            pool = in_d[ter == t]
            take = rng.choice(pool, size=min(N_PER_STRATUM, len(pool)), replace=False)
            picks.extend(
                {
                    "feat_id": int(com["feat_ids"][i]),
                    "restricted_idx": int(i),
                    "decile": d,
                    "tercile": t,
                    "r2": float(com["r2"][i]),
                    "activity": float(com["activity"][i]),
                }
                for i in take
            )
    return {"seed": SAMPLE_SEED, "n": len(picks), "features": picks}


def phase_scan(args) -> None:
    """One streamed pass over the local pooled shards: consistency accumulators
    (all fit rows + n_ans>=8 robustness cut) and top-K contexts per sampled
    feature. Vectorized bincount per shard; periodic top-K compaction."""
    com = _load_committed()
    sample = _sample_features(com)
    args.work.mkdir(parents=True, exist_ok=True)
    (args.work / "sample.json").write_text(json.dumps(sample, indent=1))
    samp_ids = np.asarray([f["feat_id"] for f in sample["features"]], dtype=np.int64)
    samp_pos = np.full(DICT_SIZE, -1, dtype=np.int64)
    samp_pos[samp_ids] = np.arange(len(samp_ids))

    shards = sorted(args.store.glob("pooled_*.npz"))
    assert len(shards) == 1920, f"expected 1920 shards, found {len(shards)}"
    cnt = np.zeros(DICT_SIZE, dtype=np.int64)
    sum_frac = np.zeros(DICT_SIZE, dtype=np.float64)
    sum_frac_sq = np.zeros(DICT_SIZE, dtype=np.float64)
    cnt_n8 = np.zeros(DICT_SIZE, dtype=np.int64)
    sum_frac_n8 = np.zeros(DICT_SIZE, dtype=np.float64)
    n_fit = 0
    cand: list[np.ndarray] = []  # columns: samp_row, ci, val(f32 as f64)

    def _compact(rows: list[np.ndarray]) -> list[np.ndarray]:
        if not rows:
            return []
        m = np.concatenate(rows, axis=0)
        keep_parts = []
        for s in np.unique(m[:, 0]).astype(np.int64):
            sub = m[m[:, 0] == s]
            if len(sub) > TOP_K_CONTEXTS:
                sub = sub[np.argpartition(-sub[:, 2], TOP_K_CONTEXTS - 1)[:TOP_K_CONTEXTS]]
            keep_parts.append(sub)
        return [np.concatenate(keep_parts, axis=0)]

    for i, p in enumerate(shards):
        with np.load(p, allow_pickle=False) as z:
            tag = np.asarray(z["set_tag"])
            off = np.asarray(z["idx_off"], dtype=np.int64)
            fit = tag == 1
            n_fit += int(fit.sum())
            idx = np.asarray(z["ans_idx"], dtype=np.int64)
            frac = np.asarray(z["ans_frac"], dtype=np.float64)
            keep = np.repeat(fit, off)
            ik, fk = idx[keep], frac[keep]
            cnt += np.bincount(ik, minlength=DICT_SIZE)
            sum_frac += np.bincount(ik, weights=fk, minlength=DICT_SIZE)
            sum_frac_sq += np.bincount(ik, weights=fk * fk, minlength=DICT_SIZE)
            n8 = fit & (np.asarray(z["n_ans"]) >= 8)
            keep8 = np.repeat(n8, off)
            cnt_n8 += np.bincount(idx[keep8], minlength=DICT_SIZE)
            sum_frac_n8 += np.bincount(idx[keep8], weights=frac[keep8], minlength=DICT_SIZE)
            # Q1 top-context candidates (fit rows, sampled features, by ans_mean)
            sp = samp_pos[ik]
            hit = sp >= 0
            if hit.any():
                ci_rep = np.repeat(np.asarray(z["ci"], dtype=np.int64), off)[keep][hit]
                val = np.asarray(z["ans_mean"], dtype=np.float64)[keep][hit]
                cand.append(np.column_stack([sp[hit].astype(np.float64), ci_rep, val]))
        if (i + 1) % 256 == 0:
            cand = _compact(cand)
            _log(f"scan {i + 1}/1920 shards; n_fit so far {n_fit}")
    cand = _compact(cand)

    # ── wiring gate: recomputed activity must match the committed covariate ──
    fid = com["feat_ids"]
    act_re = cnt[fid] / n_fit
    gate = float(np.abs(act_re - com["activity"]).max())
    _log(f"activity wiring gate: n_fit={n_fit} max|delta|={gate:.2e}")
    assert gate < 1e-3, f"activity mismatch vs committed npz (max|delta|={gate})"

    with np.errstate(invalid="ignore"):
        consistency = np.where(cnt[fid] > 0, sum_frac[fid] / np.maximum(cnt[fid], 1), np.nan)
        consistency_n8 = np.where(
            cnt_n8[fid] > 0, sum_frac_n8[fid] / np.maximum(cnt_n8[fid], 1), np.nan
        )
        ex2 = sum_frac_sq[fid] / np.maximum(cnt[fid], 1)
        consistency_sd = np.sqrt(
            np.maximum(ex2 - (sum_frac[fid] / np.maximum(cnt[fid], 1)) ** 2, 0)
        )
    np.savez(
        args.work / "scan.npz",
        feat_ids=fid,
        r2=com["r2"],
        activity=com["activity"],
        activity_recomputed=act_re,
        consistency=consistency,
        consistency_n8=consistency_n8,
        consistency_sd=consistency_sd,
        cnt=cnt[fid],
        cnt_n8=cnt_n8[fid],
        n_fit=np.int64(n_fit),
    )
    top = cand[0] if cand else np.zeros((0, 3))
    order = np.lexsort((-top[:, 2], top[:, 0]))
    top = top[order]
    top_by_feat: dict[str, list[list[float]]] = {}
    for s, ci, val in top:
        top_by_feat.setdefault(str(int(samp_ids[int(s)])), []).append([float(val), int(ci)])
    (args.work / "sample_top_contexts.json").write_text(json.dumps(top_by_feat))
    _log(
        f"scan done: n_fit={n_fit}, gate={gate:.2e}, "
        f"top-context features={len(top_by_feat)}, saved scan.npz"
    )


def phase_texts(args) -> None:
    """Stream the parent raw chunks, keeping only the sampled features' top
    contexts. Per-chunk checkpointed JSONL cache (resume-safe), text stays in
    the gitignored work dir (digest-only discipline)."""
    import issue1482_error_analysis as D  # heavy import (torch) deferred to use

    top = json.loads((args.work / "sample_top_contexts.json").read_text())
    needed_ci = {int(ci): 0 for lst in top.values() for _v, ci in lst[:TOP_K_CONTEXTS]}
    _log(f"texts: {len(needed_ci)} unique contexts needed")
    cache = args.work / "texts.jsonl"
    out: dict[int, str] = {}
    done: set[str] = set()
    if cache.exists():
        for ln in cache.read_text(encoding="utf-8").split("\n"):
            if not ln.strip():
                continue
            try:
                rec = json.loads(ln)
            except ValueError:
                continue  # truncated tail from a crash mid-append
            if rec.get("kind") == "chunk_done":
                done.add(rec["chunk"])
            else:
                out[int(rec["ci"])] = rec["response"]
    dns = argparse.Namespace(scratch=args.work, max_chunks=0)
    names = D._raw_chunk_names(dns)
    pending = [nm for nm in names if nm not in done]
    _log(f"texts: resume {len(names) - len(pending)}/{len(names)} chunks cached")
    with cache.open("a", encoding="utf-8") as fh:
        for k, name in enumerate(pending):
            for _nm, keep in D._iter_needed_rows(dns, [name], needed_ci):
                for _row, ci, _prompt, response in keep:
                    out[int(ci)] = response
                    fh.write(json.dumps({"ci": int(ci), "response": response}) + "\n")
            fh.write(json.dumps({"kind": "chunk_done", "chunk": name}) + "\n")
            fh.flush()
            if (k + 1) % 100 == 0:
                _log(f"texts: {k + 1}/{len(pending)} pending chunks, {len(out)} rows")
    missing = [ci for ci in needed_ci if ci not in out]
    assert not missing, f"{len(missing)} needed contexts had no raw-chunk text"
    _log(f"texts done: {len(out)} rows cached")


def _judge_items(args) -> list[tuple[str, str, str, str]]:
    top = json.loads((args.work / "sample_top_contexts.json").read_text())
    np_path = args.work / "neuronpedia_explanations.json"
    np_exp = json.loads(np_path.read_text()) if np_path.exists() else {}
    texts = {}
    for ln in (args.work / "texts.jsonl").read_text(encoding="utf-8").split("\n"):
        if ln.strip():
            rec = json.loads(ln)
            if rec.get("kind") != "chunk_done":
                texts[int(rec["ci"])] = rec["response"]
    items = []
    for fid, lst in sorted(top.items(), key=lambda kv: int(kv[0])):
        snippets = [
            texts[int(ci)][:SNIPPET_CHARS] for _v, ci in lst[:TOP_K_CONTEXTS] if int(ci) in texts
        ]
        if not snippets:
            continue
        body = "\n\n---\n\n".join(snippets)
        desc = (np_exp.get(str(fid)) or {}).get("description")
        aux = (
            "\n\nIndependent auto-interp description of this feature from a token-level "
            f"dashboard on a generic web corpus (may be wrong): {desc}"
            if desc
            else "\n\n(No independent auto-interp description is available for this feature.)"
        )
        items.append(
            (
                f"feat{fid}",
                f"feature {fid}",
                body[:200],
                f"Feature {fid}. Example answers:\n\n{body}{aux}\n\nOutput the JSON.",
            )
        )
    return items


def _validate_level(res) -> dict | None:
    if not isinstance(res, dict) or res.get("error"):
        return None
    lev = res.get("level")
    lab = res.get("label")
    if isinstance(lev, str) and lev.strip().lower() in ("low", "high", "unclear"):
        return {"level": lev.strip().lower(), "label": str(lab)[:120] if lab else ""}
    return None


def phase_judge(args) -> None:
    """One label call per sampled feature + RETEST_N test-retest (fresh
    dispatch dir + rt_ ids -> cold cache by construction). Drop-never-coerce."""
    from explore_persona_space.eval.judge_dispatch import dispatch_judge_items

    items = _judge_items(args)
    _log(f"judge: {len(items)} feature items")

    def _run(tag: str, its):
        return dispatch_judge_items(
            its,
            judge_model=JUDGE_MODEL,
            judge_system_prompt=JUDGE_SYSTEM,
            max_tokens=JUDGE_MAX_TOKENS,
            checkpoint_dir=args.work / f"dispatch_{tag}",
            error_dict_factory=lambda reason: {"error": True, "reason": reason},
        )

    results = _run("main", items)
    labels: dict[str, dict] = {}
    drops = {"content": 0, "transport_or_error": 0}
    for cid, res in results.items():
        if isinstance(res, dict) and res.get("error"):
            drops["transport_or_error"] += 1
            continue
        lab = _validate_level(res)
        if lab is None:
            drops["content"] += 1
            continue
        labels[cid.removeprefix("feat")] = lab

    import issue1482_analysis as A

    rng = np.random.default_rng(SAMPLE_SEED)
    rt_pick = rng.choice(len(items), size=min(RETEST_N, len(items)), replace=False)
    rt_items = [(f"rt_{items[i][0]}", *items[i][1:]) for i in rt_pick]
    rt_results = _run("retest", rt_items)
    a, b = [], []
    for i in rt_pick:
        cid = items[i][0]
        l1 = labels.get(cid.removeprefix("feat"))
        l2 = _validate_level(rt_results.get(f"rt_{cid}"))
        if l1 and l2:
            a.append(l1["level"])
            b.append(l2["level"])
    kappa = A._cohens_kappa(a, b)
    doc = {
        "n_items": len(items),
        "n_labeled": len(labels),
        "drops": drops,
        "judge_model": JUDGE_MODEL,
        "max_tokens": JUDGE_MAX_TOKENS,
        "temperature": "API default",
        "n_draws": 1,
        "rubric_sha256_system": __import__("hashlib")
        .sha256(JUDGE_SYSTEM.encode())
        .hexdigest()[:16],
        "test_retest": {"n": len(a), "kappa_level": kappa},
        "labels": labels,
    }
    (args.work / "feature_levels.json").write_text(json.dumps(doc, indent=1))
    _log(f"judge done: {len(labels)} labeled, drops={drops}, kappa={kappa:.3f} (n={len(a)})")


def phase_analyze(args) -> None:
    """Correlations + figures + committed JSON summaries."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from explore_persona_space.analysis.paper_plots import (
        paper_palette,
        savefig_paper,
        set_paper_style,
    )

    OUT_EVAL.mkdir(parents=True, exist_ok=True)
    OUT_FIGS.mkdir(parents=True, exist_ok=True)
    z = np.load(args.work / "scan.npz")
    fid, r2 = z["feat_ids"], z["r2"]
    act, cons, cons_n8 = z["activity"], z["consistency"], z["consistency_n8"]
    dec = _decile_of(act)
    ok = np.isfinite(cons)
    assert ok.all(), f"{(~ok).sum()} features with no fit-row answer activity"

    # ── Q2 summary ──
    n8_ok = np.isfinite(cons_n8)
    within = []
    for d in range(N_DECILES):
        m = dec == d
        within.append(
            {
                "decile": d,
                "n": int(m.sum()),
                "spearman_consistency_r2": _spearman(cons[m], r2[m]),
                "median_r2_by_consistency_tercile": [
                    float(np.median(r2[m][t])) for t in _tercile_masks(cons[m])
                ],
            }
        )
    q2 = {
        "n_features": int(len(fid)),
        "n_fit_rows": int(z["n_fit"]),
        "activity_wiring_gate_max_abs_delta": float(np.abs(z["activity_recomputed"] - act).max()),
        "spearman": {
            "consistency_vs_r2": _spearman(cons, r2),
            "activity_vs_r2": _spearman(act, r2),
            "consistency_vs_activity": _spearman(cons, act),
            "consistency_n8_vs_r2": float(_spearman(cons_n8[n8_ok], r2[n8_ok])),
            "n_n8": int(n8_ok.sum()),
        },
        "partial_spearman_consistency_r2_given_activity": _partial_spearman(cons, r2, act),
        "within_activity_decile": within,
        "definition": (
            "consistency = mean over fit-tagged contexts where the feature is answer-active "
            "of ans_frac (fraction of the answer's tokens where the feature is active); "
            "n8 variant restricts to contexts with >= 8 answer tokens"
        ),
    }
    (OUT_EVAL / "consistency.json").write_text(json.dumps(q2, indent=1))
    np.savez(
        OUT_EVAL / "consistency_perfeature.npz",
        feat_ids=fid,
        r2=r2,
        activity=act,
        consistency=cons,
        consistency_n8=cons_n8,
        consistency_sd=z["consistency_sd"],
        cnt=z["cnt"],
        cnt_n8=z["cnt_n8"],
    )

    # ── Q1 summary ──
    lv = json.loads((args.work / "feature_levels.json").read_text())
    sample = json.loads((args.work / "sample.json").read_text())
    by_id = {f["feat_id"]: f for f in sample["features"]}
    pos_of = {int(f): i for i, f in enumerate(fid)}
    rows = []
    for fs, lab in lv["labels"].items():
        f = int(fs)
        i = pos_of[f]
        rows.append(
            {
                "feat_id": f,
                "level": lab["level"],
                "label": lab["label"],
                "r2": float(r2[i]),
                "activity": float(act[i]),
                "consistency": float(cons[i]),
                "decile": int(dec[i]),
                "tercile": by_id[f]["tercile"],
            }
        )
    lev_arr = np.asarray([r["level"] for r in rows])
    r2_arr = np.asarray([r["r2"] for r in rows])
    act_arr = np.asarray([r["activity"] for r in rows])
    cons_arr = np.asarray([r["consistency"] for r in rows])
    dec_arr = np.asarray([r["decile"] for r in rows])
    rng = np.random.default_rng(SAMPLE_SEED)

    def _boot_median(v: np.ndarray, k: int = 10_000) -> list[float]:
        if len(v) == 0:
            return [float("nan"), float("nan")]
        draws = np.median(v[rng.integers(0, len(v), size=(k, len(v)))], axis=1)
        return [float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))]

    lohi = np.isin(lev_arr, ["low", "high"])
    hi = (lev_arr == "high").astype(np.float64)
    from scipy.stats import mannwhitneyu

    mw = mannwhitneyu(r2_arr[lev_arr == "high"], r2_arr[lev_arr == "low"])
    per_level = {
        lev: {
            "n": int((lev_arr == lev).sum()),
            "median_r2": float(np.median(r2_arr[lev_arr == lev]))
            if (lev_arr == lev).any()
            else None,
            "median_r2_ci95": _boot_median(r2_arr[lev_arr == lev]),
            "median_activity": float(np.median(act_arr[lev_arr == lev]))
            if (lev_arr == lev).any()
            else None,
            "median_consistency": float(np.median(cons_arr[lev_arr == lev]))
            if (lev_arr == lev).any()
            else None,
        }
        for lev in ("low", "high", "unclear")
    }
    within_lvl = []
    for d in range(N_DECILES):
        m = dec_arr == d
        lo_m, hi_m = m & (lev_arr == "low"), m & (lev_arr == "high")
        within_lvl.append(
            {
                "decile": d,
                "n_low": int(lo_m.sum()),
                "n_high": int(hi_m.sum()),
                "median_r2_low": float(np.median(r2_arr[lo_m])) if lo_m.any() else None,
                "median_r2_high": float(np.median(r2_arr[hi_m])) if hi_m.any() else None,
            }
        )
    q1 = {
        "n_sampled": sample["n"],
        "n_labeled": len(rows),
        "drops": lv["drops"],
        "test_retest": lv["test_retest"],
        "judge_model": lv["judge_model"],
        "max_tokens": lv["max_tokens"],
        "rubric_sha256_system": lv["rubric_sha256_system"],
        "sample_seed": SAMPLE_SEED,
        "per_level": per_level,
        "mannwhitney_high_vs_low": {"U": float(mw.statistic), "p": float(mw.pvalue)},
        "spearman_highlevel_vs_r2": float(_spearman(hi[lohi], r2_arr[lohi])),
        "partial_spearman_highlevel_r2_given_activity": _partial_spearman(
            hi[lohi], r2_arr[lohi], act_arr[lohi]
        ),
        "spearman_highlevel_vs_consistency": float(_spearman(hi[lohi], cons_arr[lohi])),
        "within_activity_decile": within_lvl,
        "features": rows,
    }
    (OUT_EVAL / "abstraction.json").write_text(json.dumps(q1, indent=1))

    # ── figures ──
    set_paper_style()
    pal = paper_palette(3)
    r2c = np.clip(r2, -1, None)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.scatter(cons, r2c, s=3, alpha=0.15, color=pal[0], rasterized=True, linewidths=0)
    qs = np.quantile(cons, np.linspace(0, 1, 11))
    med_x = [float(np.median(cons[(cons >= qs[i]) & (cons <= qs[i + 1])])) for i in range(10)]
    med_y = [float(np.median(r2c[(cons >= qs[i]) & (cons <= qs[i + 1])])) for i in range(10)]
    ax.plot(med_x, med_y, color=pal[1], lw=2, marker="o", ms=4, label="decile median")
    ax.set_xscale("log")
    ax.set_xlabel("within-answer consistency (mean fraction of answer tokens active, log)")
    ax.set_ylabel("per-feature held-out $R^2$ (clipped at $-1$)")
    ax.legend(frameon=False)
    ax = axes[1]
    for t, (name, color) in enumerate(zip(("low", "mid", "high"), (pal[0], pal[2], pal[1]))):
        ys = [w["median_r2_by_consistency_tercile"][t] for w in within]
        ax.plot(
            range(N_DECILES),
            ys,
            marker="o",
            ms=4,
            lw=1.8,
            color=color,
            label=f"consistency {name} tercile",
        )
    ax.set_xlabel("activity decile (low → high)")
    ax.set_ylabel("median per-feature $R^2$")
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig_paper(fig, "perfeature_r2_vs_consistency", dir=OUT_FIGS)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    order = ("low", "unclear", "high")
    for k, lev in enumerate(order):
        v = r2_arr[lev_arr == lev]
        jitter = (rng.random(len(v)) - 0.5) * 0.5
        ax.scatter(
            np.full(len(v), k) + jitter,
            np.clip(v, -1, None),
            s=10,
            alpha=0.5,
            color=pal[k % 3],
            linewidths=0,
        )
        if len(v):
            ax.hlines(np.median(v), k - 0.3, k + 0.3, color="black", lw=2)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([f"{lev}\n(n={int((lev_arr == lev).sum())})" for lev in order])
    ax.set_xlabel("judged feature level")
    ax.set_ylabel("per-feature held-out $R^2$ (clipped at $-1$)")
    ax = axes[1]
    lo_y = [w["median_r2_low"] for w in within_lvl]
    hi_y = [w["median_r2_high"] for w in within_lvl]
    ax.plot(range(N_DECILES), lo_y, marker="o", ms=4, lw=1.8, color=pal[0], label="low-level")
    ax.plot(range(N_DECILES), hi_y, marker="o", ms=4, lw=1.8, color=pal[1], label="high-level")
    ax.set_xlabel("activity decile (low → high)")
    ax.set_ylabel("median per-feature $R^2$ (sampled)")
    ax.legend(frameon=False)
    fig.tight_layout()
    savefig_paper(fig, "perfeature_r2_by_judged_level", dir=OUT_FIGS)
    plt.close(fig)
    _log("analyze done: consistency.json + abstraction.json + 2 figures")


def _tercile_masks(v: np.ndarray) -> list[np.ndarray]:
    e = np.quantile(v, [1 / 3, 2 / 3])
    t = np.searchsorted(e, v, side="right")
    return [t == k for k in range(3)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", required=True, choices=["scan", "texts", "judge", "analyze"])
    ap.add_argument("--store", type=Path, default=STORE_DEFAULT)
    ap.add_argument("--work", type=Path, default=WORK_DEFAULT)
    args = ap.parse_args()
    {"scan": phase_scan, "texts": phase_texts, "judge": phase_judge, "analyze": phase_analyze}[
        args.phase
    ](args)


if __name__ == "__main__":
    main()
