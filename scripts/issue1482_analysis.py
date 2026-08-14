"""Issue #1482 — OFF-POD phases: P5 judge categorization + P6 analysis/figures.

P5 runs AFTER pod release (plan §9 GPU-width rule): Anthropic Batch API via the
#1019/#1313-hardened ``dispatch_judge_items`` (rubric-keyed cache, per-row
transport retry, drop-never-coerce). LMSYS/WildChat user text rides ONLY in the
API payload — never printed, logged, or persisted by this script (digest-only);
judge REASONING text (our instrument's output) is persisted under
``judge_labels/raw/``.

P6 computes the registered H1 contrast (batched 10k-draw bootstrap as chunked
GEMMs — no per-draw Python loop), BH-FDR exploratory contrasts, the
linear-vs-nonlinear gap decomposition, and the hero figures.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1482_error_analysis as D  # noqa: E402  (driver helpers: chunk iteration, split)
import numpy as np  # noqa: E402

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", force=True
)
logger = logging.getLogger("issue1482_analysis")

JUDGE_MODEL = "claude-sonnet-4-5-20250929"  # CLAUDE.md project judge pin
JUDGE_MAX_TOKENS = 1024  # reason-then-label; llm-judging rule 23 floor (raised from 400, #2063)
TOPICS = (
    "coding",
    "math",
    "creative_writing",
    "roleplay_persona",
    "factual_qa",
    "advice_howto",
    "translation",
    "summarization_extraction",
    "chitchat_social",
    "harmful_or_unsafe_request",
    "nsfw",
    "other",
)
FIELDS = {
    "language": None,  # ISO 639-1 (free two-letter code)
    "topic": set(TOPICS),
    "request_refusal_adjacent": {"yes", "borderline", "no"},
    "answer_is_refusal": {"yes", "partial", "no"},
    "format": {"code", "list", "prose", "mixed"},
}
JUDGE_SYSTEM = (
    "You are a careful data annotator categorizing real user-assistant chat exchanges. "
    "You never refuse to CATEGORIZE content (categorizing harmful requests as "
    "harmful_or_unsafe_request IS the safety-relevant labeling task). First reason briefly, "
    "then output ONLY a JSON object with EXACTLY these keys:\n"
    '  "reasoning": one to three sentences.\n'
    '  "language": the ISO 639-1 code of the USER prompt language (e.g. "en", "es", "zh"; '
    "pick the dominant language if mixed).\n"
    f'  "topic": one of {list(TOPICS)}.\n'
    '  "request_refusal_adjacent": "yes" if a typical aligned assistant might refuse or '
    'safe-complete the request, "borderline" if unclear, else "no".\n'
    '  "answer_is_refusal": "yes" if the assistant answer is a refusal, "partial" if it '
    'hedges or partially declines, else "no".\n'
    '  "format": the dominant ANSWER format — "code", "list", "prose", or "mixed".\n'
    "Anchors: a benign coding question in English -> language en, topic coding, "
    "request_refusal_adjacent no, answer_is_refusal no. A request for illegal instructions "
    "answered by a decline -> request_refusal_adjacent yes, answer_is_refusal yes."
)


def _judge_user_msg(prompt: str, response: str, corpus: str) -> str:
    return (
        f"Corpus: {corpus}\n\n=== USER PROMPT (truncated to 1500 chars) ===\n{prompt[:1500]}\n\n"
        f"=== ASSISTANT ANSWER (truncated to 1000 chars) ===\n{response[:1000]}\n\n"
        "Categorize this exchange per the system instructions. Reason briefly, then output "
        "the JSON object."
    )


def _validate_label(parsed: object) -> dict | None:
    """Schema-validate one judge return; None = content drop (never coerced)."""
    if not isinstance(parsed, dict):
        return None
    out = {}
    for field, allowed in FIELDS.items():
        v = parsed.get(field)
        if not isinstance(v, str):
            return None
        v = v.strip().lower()
        if allowed is None:
            if not (len(v) == 2 and v.isalpha()):
                return None
        elif v not in allowed:
            return None
        out[field] = v
    return out


def _collect_texts(args, rows: np.ndarray) -> dict[int, tuple[int, str, str, str]]:
    """ci -> (row_idx, prompt, response, corpus) for the given rows (digest-only:
    text is cached to SCRATCH — gitignored, never eval_results — and never logged).

    Per-chunk checkpoint + resume (external-stream rule): the ~1,936-chunk re-fetch
    is 30-60 min network-bound, so each chunk's kept rows append to a JSONL cache
    (fingerprint-keyed on the row set + max_chunks) with a chunk_done marker written
    AFTER the chunk's rows; a crash costs at most one in-flight chunk."""
    row_ci = np.load(args.scratch / "row_ci.npy")
    prov = np.load(args.scratch / "prov.npy")
    needed_ci = {int(row_ci[r]): int(r) for r in rows}
    assert -1 not in needed_ci, "judge rows must be NEW rows (text-resolvable)"
    dns = argparse.Namespace(scratch=args.scratch, max_chunks=args.max_chunks)
    cache_dir = args.scratch / "judge_text_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = hashlib.sha256(
        json.dumps(
            {"rows": sorted(int(r) for r in rows), "max_chunks": int(args.max_chunks)}
        ).encode()
    ).hexdigest()[:16]
    cache = cache_dir / f"texts_{fp}.jsonl"
    out: dict[int, tuple[int, str, str, str]] = {}
    done_chunks: set[str] = set()
    if cache.exists():
        lines = [ln for ln in cache.read_text(encoding="utf-8").split("\n") if ln.strip()]
        for i, ln in enumerate(lines):
            try:
                rec = json.loads(ln)
            except ValueError:
                if i == len(lines) - 1:  # truncated tail from a crash mid-append
                    logger.warning("[collect] dropping truncated cache tail line")
                    break
                raise  # a malformed MIDDLE line is corruption — fail loud
            if rec.get("kind") == "chunk_done":
                done_chunks.add(rec["chunk"])
            else:
                out[int(rec["ci"])] = (
                    int(rec["row"]),
                    rec["prompt"],
                    rec["response"],
                    rec["corpus"],
                )
    names = D._raw_chunk_names(dns)
    pending = [nm for nm in names if nm not in done_chunks]
    if done_chunks:
        logger.info("[collect] resume: %d/%d chunks cached", len(names) - len(pending), len(names))
    with cache.open("a", encoding="utf-8") as fh:
        for name in pending:
            # one chunk per call so EVERY processed chunk gets its done marker
            # (D._iter_needed_rows yields only chunks with kept rows)
            for _nm, keep in D._iter_needed_rows(dns, [name], needed_ci):
                for row_idx, ci, prompt, response in keep:
                    corpus = "wildchat" if prov[row_idx] else "lmsys"
                    out[int(ci)] = (int(row_idx), prompt, response, corpus)
                    fh.write(
                        json.dumps(
                            {
                                "ci": int(ci),
                                "row": int(row_idx),
                                "prompt": prompt,
                                "response": response,
                                "corpus": corpus,
                            }
                        )
                        + "\n"
                    )
            fh.write(json.dumps({"kind": "chunk_done", "chunk": name}) + "\n")
            fh.flush()
    missing = len(needed_ci) - len(out)
    assert missing == 0, f"{missing} judge rows had no raw-chunk text"
    return out


def phase_judge(args) -> None:
    """P5: one categorization call per holdout context (+ 200-item test-retest)."""
    from explore_persona_space.eval.judge_dispatch import dispatch_judge_items, keep_raw_judge_text

    idx = np.load(args.scratch / "split_indices.npz")
    rows = np.sort(idx["holdout"])
    if args.n_items > 0:
        rows = rows[: args.n_items]
    texts = _collect_texts(args, rows)
    items = [
        (f"ci{ci}", p[:1500], r[:1000], _judge_user_msg(p, r, corpus))
        for ci, (_row, p, r, corpus) in sorted(texts.items())
    ]
    jdir = args.out_eval / "judge_labels"
    (jdir / "raw").mkdir(parents=True, exist_ok=True)

    def _run(tag: str, its):
        with keep_raw_judge_text():
            return dispatch_judge_items(
                its,
                judge_model=JUDGE_MODEL,
                judge_system_prompt=JUDGE_SYSTEM,
                max_tokens=JUDGE_MAX_TOKENS,
                threshold_base=1 if args.force_batch else 2000,
                checkpoint_dir=jdir / f"dispatch_{tag}",
                error_dict_factory=lambda reason: {"error": True, "reason": reason},
            )

    results = _run("main", items)
    labels: dict[str, dict] = {}
    drops = {"content": 0, "transport_or_error": 0}
    raw_rows = []
    for cid, res in results.items():
        if isinstance(res, dict) and res.get("error"):
            drops["transport_or_error"] += 1
            raw_rows.append(
                {"custom_id": cid, "error": True, "reason": str(res.get("reason"))[:300]}
            )
            continue
        lab = _validate_label(res)
        raw_rows.append(
            {
                "custom_id": cid,
                "raw": (res or {}).get("_raw_text", "") if isinstance(res, dict) else "",
            }
        )
        if lab is None:
            drops["content"] += 1
            continue
        labels[cid.removeprefix("ci")] = lab
    (jdir / "raw" / "main.jsonl").write_text("\n".join(json.dumps(r) for r in raw_rows) + "\n")

    # test-retest (fresh dispatch dir; rt_ custom_ids -> cold cache by construction)
    rng = np.random.default_rng(D.SPLIT_SEED_1482)
    n_rt = min(args.retest_n, len(items))
    rt_pick = rng.choice(len(items), size=n_rt, replace=False)
    rt_items = [(f"rt_{items[i][0]}", *items[i][1:]) for i in rt_pick]
    rt_results = _run("retest", rt_items)
    kappa = {}
    for field in FIELDS:
        a, b = [], []
        for i in rt_pick:
            cid = items[i][0]
            l1 = labels.get(cid.removeprefix("ci"))
            l2 = _validate_label(rt_results.get(f"rt_{cid}"))
            if l1 and l2:
                a.append(l1[field])
                b.append(l2[field])
        kappa[field] = {"n": len(a), "kappa": _cohens_kappa(a, b)}
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
        "test_retest_kappa": kappa,
        "labels": labels,
    }
    D._write_json(jdir / "labels.json", doc)
    logger.info(
        "[judge] labeled %d/%d (drops=%s) kappa=%s",
        len(labels),
        len(items),
        drops,
        {k: round(v["kappa"], 3) if v["kappa"] == v["kappa"] else None for k, v in kappa.items()},
    )


def _kappa_gate(kap) -> str | None:
    """Instrument-failure reason when the test-retest kappa cannot support a
    verdict: NaN/missing kappa is an UNMEASURED instrument (demote — never pass),
    kappa < 0.6 is a failed instrument. None = instrument OK."""
    if kap is None or not np.isfinite(kap):
        return f"language test-retest kappa unmeasured/non-finite ({kap})"
    if kap < 0.6:
        return f"language test-retest kappa {kap:.3f} < 0.6"
    return None


def _cohens_kappa(a: list[str], b: list[str]) -> float:
    if len(a) < 2:
        return float("nan")
    cats = sorted(set(a) | set(b))
    ai = np.array([cats.index(x) for x in a])
    bi = np.array([cats.index(x) for x in b])
    po = float((ai == bi).mean())
    pe = float(sum((ai == k).mean() * (bi == k).mean() for k in range(len(cats))))
    return float("nan") if pe >= 1.0 else (po - pe) / (1.0 - pe)


def phase_interp_judge(args) -> None:
    """~50 feature-label calls (worst/best/random tails) — sync route (N << 2000)."""
    from explore_persona_space.eval.judge_dispatch import dispatch_judge_items

    digest_path = args.out_eval / "interp_digest.json"
    dig = json.loads(digest_path.read_text())
    if not dig.get("top_contexts"):
        logger.info("[interp] no candidate features; skipping")
        return
    all_ci = sorted(
        {
            int(ci)
            for grp in dig["top_contexts"].values()
            for f, lst in grp.items()
            for _v, ci in lst[:8]
        }
    )
    idx = np.load(args.scratch / "split_indices.npz")
    row_ci = np.load(args.scratch / "row_ci.npy")
    ci_to_row = {
        int(row_ci[r]): int(r)
        for r in np.concatenate([idx["holdout"], idx["sae_fit"], idx["sae_val"]])
    }
    rows = np.asarray([ci_to_row[c] for c in all_ci if c in ci_to_row], dtype=np.int64)
    texts = _collect_texts(args, rows)
    sysmsg = (
        "You label sparse-autoencoder features from example texts. Given up to 8 assistant "
        "answers where a feature fires strongly, reason briefly, then output ONLY JSON: "
        '{"reasoning": "...", "label": "<= 8 words naming the shared property"}.'
    )
    items = []
    for grp, feats in dig["top_contexts"].items():
        for f, lst in feats.items():
            snippets = []
            for _v, ci in lst[:8]:
                t = texts.get(int(ci))
                if t is not None:
                    snippets.append(t[2][:400])
            if not snippets:
                continue
            body = "\n\n---\n\n".join(snippets)
            items.append(
                (
                    f"feat{f}_{grp}",
                    f"feature {f}",
                    body[:200],
                    f"Feature {f} ({grp} tail). Example answers:\n\n{body}\n\nOutput the JSON.",
                )
            )
    results = dispatch_judge_items(
        items,
        judge_model=JUDGE_MODEL,
        judge_system_prompt=sysmsg,
        max_tokens=JUDGE_MAX_TOKENS,
        checkpoint_dir=args.out_eval / "judge_labels" / "dispatch_interp",
        error_dict_factory=lambda reason: {"error": True, "reason": reason},
    )
    out = {}
    for cid, res in results.items():
        lab = res.get("label") if isinstance(res, dict) and not res.get("error") else None
        out[cid] = str(lab)[:120] if isinstance(lab, str) else None
    D._write_json(
        args.out_eval / "judge_labels" / "feature_labels.json", {"labels": out, "n": len(out)}
    )
    logger.info("[interp] labeled %d features", sum(1 for v in out.values() if v))


# ── P6: analysis + figures ───────────────────────────────────────────────────────


def _boot_group_delta(nerr, mask_a, mask_b, n_boot, seed, chunk=2000):
    """Batched bootstrap of mean(nerr[a]) - mean(nerr[b]) — chunked gathers, no
    per-draw Python loop over pool reductions."""
    rng = np.random.default_rng(seed)
    n = len(nerr)
    deltas = np.empty(n_boot, dtype=np.float64)
    for s in range(0, n_boot, chunk):
        b = min(chunk, n_boot - s)
        take = rng.integers(0, n, size=(b, n))
        vals = nerr[take]
        ma = mask_a[take]
        mb = mask_b[take]
        na = ma.sum(1)
        nb = mb.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            deltas[s : s + b] = (vals * ma).sum(1) / na - (vals * mb).sum(1) / nb
    return deltas[np.isfinite(deltas)]


def _perm_pvals(nerr: np.ndarray, masks: list[np.ndarray], n_perm: int, seed: int) -> list[float]:
    """Two-sided permutation p per binary contrast (group vs rest), batched as one
    (B, n) @ (n,) GEMM per contrast (subset-sum identity)."""
    rng = np.random.default_rng(seed)
    n = len(nerr)
    pvals = []
    for m in masks:
        k = int(m.sum())
        if k == 0 or k == n:
            pvals.append(float("nan"))
            continue
        obs = nerr[m].mean() - nerr[~m].mean()
        P = np.zeros((n_perm, n), dtype=np.float32)
        for b in range(n_perm):  # index permutation only (O(n) each); reduction is the GEMM
            P[b, rng.permutation(n)[:k]] = 1.0
        sums = P @ nerr.astype(np.float32)
        tot = float(nerr.sum())
        deltas = sums / k - (tot - sums) / (n - k)
        pvals.append(float(((np.abs(deltas) >= abs(obs)).sum() + 1) / (n_perm + 1)))
    return pvals


def _bh_fdr(pvals: list[float], q: float = 0.05) -> list[bool]:
    p = np.asarray(pvals, dtype=np.float64)
    ok = np.isfinite(p)
    m = int(ok.sum())
    sig = np.zeros(len(p), dtype=bool)
    if m == 0:
        return sig.tolist()
    order = np.argsort(p[ok])
    ranked = p[ok][order]
    thresh = q * (np.arange(1, m + 1) / m)
    below = ranked <= thresh
    if below.any():
        kmax = int(np.max(np.where(below)[0]))
        cut = ranked[kmax]
        sig[ok] = p[ok] <= cut
    return sig.tolist()


def _errbars(v, lo, hi):
    """matplotlib xerr/yerr take NON-NEGATIVE offsets (gotchas #547/#1335)."""
    return np.maximum(0, np.asarray(v) - np.asarray(lo)), np.maximum(
        0, np.asarray(hi) - np.asarray(v)
    )


def phase_analysis(args) -> None:  # noqa: C901 — one linear P6 pass (tables + figures)
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        from explore_persona_space.analysis import paper_plots

        paper_plots.set_paper_style()
    except Exception as e:  # cosmetic only — default rcParams are acceptable
        logger.warning("[p6] paper style unavailable (%s); using defaults", e)

    pdir = args.out_eval / "percontext"
    labels_doc = json.loads((args.out_eval / "judge_labels" / "labels.json").read_text())
    labels = labels_doc["labels"]
    kappa = labels_doc["test_retest_kappa"]
    row_ci = np.load(args.scratch / "row_ci.npy")
    prov = np.load(args.scratch / "prov.npy")
    seed = D.SPLIT_SEED_1482

    def _load(fit_id):
        z = np.load(pdir / f"{fit_id}.npz")
        return z["holdout_rows"], z["holdout_nerr"], z["holdout_e2"]

    rows, nerr_ridge, _e2_ridge = _load(f"refit_holdout__ridge__seed{args.seed}")
    _, nerr_mlp, _ = _load(f"refit_holdout__mlp_w8192__seed{args.seed}")
    _, nerr_mlp43, _ = _load(f"refit_holdout__mlp_w8192__seed{D.MLP_SEED_B}")
    lang = np.array([labels.get(str(int(row_ci[r])), {}).get("language", "") for r in rows])
    have = lang != ""
    topic = np.array([labels.get(str(int(row_ci[r])), {}).get("topic", "") for r in rows])
    refadj = np.array(
        [labels.get(str(int(row_ci[r])), {}).get("request_refusal_adjacent", "") for r in rows]
    )
    ansref = np.array(
        [labels.get(str(int(row_ci[r])), {}).get("answer_is_refusal", "") for r in rows]
    )
    corpus = np.where(prov[rows] == 1, "wildchat", "lmsys")

    # ── H1 (registered): non-English vs English mean normalized error, ridge ──
    en = have & (lang == "en")
    ne = have & (lang != "en")
    h1: dict = {
        "n_en": int(en.sum()),
        "n_non_en": int(ne.sum()),
        "n_labeled": int(have.sum()),
        "language_kappa": kappa.get("language", {}).get("kappa"),
    }
    if en.sum() >= 2 and ne.sum() >= 2:
        sub = have
        deltas = _boot_group_delta(nerr_ridge[sub], ne[sub], en[sub], args.n_boot, seed)
        delta = float(nerr_ridge[ne].mean() - nerr_ridge[en].mean())
        lo, hi = float(np.percentile(deltas, 2.5)), float(np.percentile(deltas, 97.5))
        if delta > 0 and lo > 0:
            verdict = "Confirmed"
        elif hi < 0:
            verdict = "Falsified"
        else:
            verdict = "Inconclusive"
        reason = _kappa_gate(kappa.get("language", {}).get("kappa"))
        if reason is not None:  # NaN/missing kappa demotes too (instrument-failure)
            verdict = "Inconclusive"
            h1["instrument_failure"] = reason
        h1.update(
            {
                "delta_nerr": delta,
                "ci95": [lo, hi],
                "verdict": verdict,
                "mean_nerr_en": float(nerr_ridge[en].mean()),
                "mean_nerr_non_en": float(nerr_ridge[ne].mean()),
            }
        )
    else:
        h1.update({"verdict": "Inconclusive", "reason": "empty language group at this n"})
    D._write_json(
        args.out_eval / "h1_contrast.json",
        {**h1, "n_boot": args.n_boot, "predictor": "ridge (refit_holdout)"},
    )
    logger.info(
        "[p6] H1: %s", {k: h1.get(k) for k in ("verdict", "delta_nerr", "ci95", "n_en", "n_non_en")}
    )

    # ── exploratory contrasts (BH-FDR q=0.05) ──
    contrasts: list[tuple[str, np.ndarray]] = []
    for t in TOPICS:
        contrasts.append((f"topic:{t}", have & (topic == t)))
    contrasts.append(
        ("refusal_adjacent:yes+borderline", have & np.isin(refadj, ["yes", "borderline"]))
    )
    contrasts.append(("answer_is_refusal:yes+partial", have & np.isin(ansref, ["yes", "partial"])))
    contrasts.append(("corpus:wildchat", corpus == "wildchat"))
    n_ans = _n_ans_for_rows(args, rows)
    n_ans_missing = int(np.isnan(n_ans).sum()) if n_ans is not None else None
    if n_ans is not None:
        qs = np.nanquantile(n_ans, [0.25, 0.5, 0.75])  # NaN rows fall out of every quartile mask
        for i, (a, b) in enumerate([(0, qs[0]), (qs[0], qs[1]), (qs[1], qs[2]), (qs[2], np.inf)]):
            contrasts.append((f"answer_len_q{i + 1}", (n_ans > a) & (n_ans <= b)))
    masks = [m for _, m in contrasts]
    pvals = _perm_pvals(nerr_ridge, masks, args.n_boot, seed)
    sig = _bh_fdr(pvals, q=0.05)
    tax = []
    for (name, m), p, s in zip(contrasts, pvals, sig, strict=True):
        if m.sum() == 0 or m.sum() == len(m):
            tax.append({"contrast": name, "n": int(m.sum()), "note": "degenerate group"})
            continue
        tax.append(
            {
                "contrast": name,
                "n": int(m.sum()),
                "mean_nerr_in": float(nerr_ridge[m].mean()),
                "mean_nerr_out": float(nerr_ridge[~m].mean()),
                "ratio": float(nerr_ridge[m].mean() / max(1e-12, nerr_ridge[~m].mean())),
                "perm_p": p,
                "bh_fdr_sig_q05": bool(s),
            }
        )
    D._write_json(
        args.out_eval / "taxonomy.json",
        {
            "contrasts": tax,
            "exploratory": True,
            "n_perm": args.n_boot,
            "per_field_kappa": kappa,
            "drops": labels_doc["drops"],
            # per-row n_ans coverage (None = no store staged; rows over the 5%
            # floor fail loud inside _n_ans_for_rows)
            "n_ans_missing": n_ans_missing,
        },
    )

    # ── gap decomposition (H3, descriptive) ──
    dgap = nerr_ridge - nerr_mlp
    dgap43 = nerr_ridge - nerr_mlp43
    order = np.argsort(-np.abs(dgap))
    top_dec = order[: max(1, len(dgap) // 10)]
    gap = {
        "mass_nonlinear_better_frac": float((dgap > 0).mean()),
        "mass_linear_better_frac": float((dgap < 0).mean()),
        "sum_abs_gap": float(np.abs(dgap).sum()),
        "top_decile_share_of_abs_gap": float(
            np.abs(dgap[top_dec]).sum() / max(1e-12, np.abs(dgap).sum())
        ),
        "seed_pair_pearson": float(np.corrcoef(dgap, dgap43)[0, 1])
        if len(dgap) > 2
        else float("nan"),
        "per_category_mean_gap": {
            t: float(dgap[have & (topic == t)].mean())
            for t in TOPICS
            if (have & (topic == t)).sum() > 0
        },
        "worst50_ci_digest": [int(row_ci[rows[i]]) for i in order[:50]],
    }
    D._write_json(args.out_eval / "gap_decomposition.json", gap)

    # ── figures ──
    fdir = args.fig_dir
    fdir.mkdir(parents=True, exist_ok=True)
    # hero 1: per-category nerr means + bootstrap CIs, + per-context scatter
    cats = (
        [(f"lang:{v}", have & (lang == v)) for v in ("en",)]
        + [("lang:non-en", ne)]
        + [(f"topic:{t}", have & (topic == t)) for t in TOPICS]
    )
    cats = [(nm, m) for nm, m in cats if m.sum() >= 2]
    if cats:
        means, los, his, names = [], [], [], []
        for nm, m in sorted(cats, key=lambda c: float(nerr_ridge[c[1]].mean())):
            # per-category bootstrap CI of the group mean (chunked gather)
            rngl = np.random.default_rng(seed)
            take = rngl.integers(0, len(nerr_ridge), size=(min(2000, args.n_boot), len(nerr_ridge)))
            mm = m[take]
            with np.errstate(invalid="ignore", divide="ignore"):
                bmeans = (nerr_ridge[take] * mm).sum(1) / mm.sum(1)
            bmeans = bmeans[np.isfinite(bmeans)]
            means.append(float(nerr_ridge[m].mean()))
            los.append(float(np.percentile(bmeans, 2.5)))
            his.append(float(np.percentile(bmeans, 97.5)))
            names.append(f"{nm} (n={int(m.sum())})")
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
        e_lo, e_hi = _errbars(means, los, his)
        ax1.barh(np.arange(len(means)), means, xerr=(e_lo, e_hi))
        ax1.set_yticks(np.arange(len(means)), names, fontsize=7)
        ax1.set_xlabel("mean normalized per-context error (ridge, fresh holdout)")
        if n_ans is not None:
            sc_cat = np.where(ne, 2, np.where(en, 1, 0))
            ax2.scatter(n_ans, nerr_ridge, c=sc_cat, s=6, alpha=0.4, cmap="viridis")
            worst = np.argsort(-nerr_ridge)[:50]
            ax2.scatter(n_ans[worst], nerr_ridge[worst], s=14, facecolors="none", edgecolors="red")
            ax2.set_xlabel("answer tokens")
            ax2.set_ylabel("nerr(x)")
            ax2.set_yscale("log")
        fig.savefig(fdir / "hero1_category_error.png", dpi=200)
        plt.close(fig)
    # hero 2: per-feature held-out R2 vs activity
    prim = args.out_eval / "sae_perfeature" / "sae_ctx__mean__ridge.npz"
    if prim.exists():
        d = np.load(prim)
        ok = np.isfinite(d["r2"])
        fig, ax = plt.subplots(figsize=(7, 5), layout="constrained")
        ax.scatter(d["activity"][ok], np.clip(d["r2"][ok], -1, 1), s=5, alpha=0.4)
        ax.set_xscale("log")
        ax.set_xlabel("feature activity (frac of fit contexts)")
        ax.set_ylabel("per-feature held-out R2 (clipped at -1)")
        fig.savefig(fdir / "hero2_perfeature_r2_vs_activity.png", dpi=200)
        plt.close(fig)
    # hero 3: paired ridge-vs-MLP nerr scatter + per-feature R2 scatter
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5), layout="constrained")
    ax1.scatter(nerr_ridge, nerr_mlp, s=6, alpha=0.4)
    lim = [min(nerr_ridge.min(), nerr_mlp.min()), max(nerr_ridge.max(), nerr_mlp.max())]
    ax1.plot(lim, lim, lw=0.8, color="gray")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlabel("nerr ridge")
    ax1.set_ylabel("nerr MLP w8192")
    prim_mlp = args.out_eval / "sae_perfeature" / "sae_ctx__mean__mlp.npz"
    if prim.exists() and prim_mlp.exists():
        dr, dm = np.load(prim), np.load(prim_mlp)
        ok = np.isfinite(dr["r2"]) & np.isfinite(dm["r2"])
        ax2.scatter(np.clip(dr["r2"][ok], -1, 1), np.clip(dm["r2"][ok], -1, 1), s=5, alpha=0.4)
        ax2.plot([-1, 1], [-1, 1], lw=0.8, color="gray")
        ax2.set_xlabel("per-feature R2 ridge")
        ax2.set_ylabel("per-feature R2 MLP")
    fig.savefig(fdir / "hero3_gap.png", dpi=200)
    plt.close(fig)
    # exploratory: Lorenz curve of |dgap|, corpus bars, transfer fold
    fig, ax = plt.subplots(figsize=(5.5, 4.5), layout="constrained")
    sortd = np.sort(np.abs(dgap))[::-1]
    ax.plot(np.arange(1, len(sortd) + 1) / len(sortd), np.cumsum(sortd) / max(1e-12, sortd.sum()))
    ax.set_xlabel("fraction of contexts (sorted by |gap|)")
    ax.set_ylabel("cumulative share of sum |gap|")
    fig.savefig(fdir / "gap_lorenz.png", dpi=200)
    plt.close(fig)
    tr_path = pdir / f"refit_lmsys_transfer__ridge__seed{args.seed}.npz"
    if tr_path.exists():
        zt = np.load(tr_path)
        nerr_tr = zt["holdout_nerr"]
        vals = [
            float(nerr_ridge[corpus == "lmsys"].mean()),
            float(nerr_ridge[corpus == "wildchat"].mean()),
            float(nerr_tr[corpus == "lmsys"].mean()),
            float(nerr_tr[corpus == "wildchat"].mean()),
        ]
        fig, ax = plt.subplots(figsize=(6, 4), layout="constrained")
        ax.bar(
            ["mixed->lmsys", "mixed->wildchat", "lmsys-only->lmsys", "lmsys-only->wildchat"], vals
        )
        ax.set_ylabel("mean nerr (holdout)")
        ax.tick_params(axis="x", labelsize=7)
        fig.savefig(fdir / "transfer_fold.png", dpi=200)
        plt.close(fig)
    logger.info("[p6] analysis + figures done -> %s", fdir)


def _n_ans_for_rows(args, rows: np.ndarray) -> np.ndarray | None:
    """Answer token counts per holdout row from the pooled store shards.

    Per-row: a missing row -> NaN (counted + reported by the caller), never an
    all-or-nothing None drop of the planned length contrasts; fail-loud when
    >5% of rows are missing (a systematic store/split mismatch, not a stray
    empty-response tokenization drop). No shards at all -> None (no store)."""
    shards = sorted(args.store.glob("pooled_*.npz"))
    if not shards:
        return None
    m: dict[int, int] = {}
    for p in shards:
        z = np.load(p)
        for r, na in zip(z["row_idx"], z["n_ans"], strict=True):
            m[int(r)] = int(na)
    vals = np.asarray([float(m.get(int(r), np.nan)) for r in rows], dtype=np.float64)
    n_missing = int(np.isnan(vals).sum())
    if n_missing:
        frac = n_missing / max(1, len(rows))
        if frac > 0.05:
            raise RuntimeError(
                f"_n_ans_for_rows: {n_missing}/{len(rows)} holdout rows missing from the "
                f"pooled store ({frac:.1%} > 5%) — store/split mismatch, not a stray drop"
            )
        logger.warning("[p6] n_ans missing for %d/%d holdout rows (NaN)", n_missing, len(rows))
    return vals


_SCRATCH_FILES = ("split_indices.npz", "row_ci.npy", "prov.npy")


def _require_scratch(args) -> None:
    """Fail loud WITH THE RECOVERY RECIPE when the pod-built scratch metadata is
    absent — the epm:failure v6 class: every phase here loads these files, but they
    were built by the pod-side P0 carve and (pre-r6) never uploaded, so a terminated
    pod left a bare FileNotFoundError at P5 launch."""
    missing = [n for n in _SCRATCH_FILES if not (args.scratch / n).exists()]
    if missing:
        raise SystemExit(
            f"[{args.phase}] scratch metadata missing under {args.scratch}: {missing}. "
            "Reconstruct it deterministically on the VM (sha-verified against "
            "eval_results/issue_1482/split_1482.json) with "
            "`uv run python scripts/issue1482_reconstruct_scratch.py`, or stage "
            "analysis_tensors/scratch_meta/ from the HF data repo (uploaded by P4 as of r6)."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #1482 off-pod phases (P5 judge, P6 analysis).")
    ap.add_argument("--phase", required=True, choices=["judge", "interp-judge", "analysis"])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--out-eval", type=Path, default=None)
    ap.add_argument("--scratch", type=Path, default=None)
    ap.add_argument("--store", type=Path, default=None)
    ap.add_argument("--fig-dir", type=Path, default=None)
    ap.add_argument("--max-chunks", type=int, default=None)
    ap.add_argument("--n-items", type=int, default=None, help="0 = all holdout rows")
    ap.add_argument("--retest-n", type=int, default=None)
    ap.add_argument("--n-boot", type=int, default=None)
    ap.add_argument(
        "--force-batch",
        action="store_true",
        help="force the Batch API route regardless of N (live batch smoke)",
    )
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    dd = (
        {"max_chunks": 1, "n_items": 20, "retest_n": 5, "n_boot": 200}
        if args.smoke
        else {"max_chunks": 0, "n_items": 0, "retest_n": 200, "n_boot": 10_000}
    )
    for k, v in dd.items():
        if getattr(args, k) is None:
            setattr(args, k, v)
    root = PROJECT_ROOT / "data" / "issue_1482"
    base = (root / "smoke_out") if args.smoke else root
    if args.out_eval is None:
        args.out_eval = (
            (base / "eval_results")
            if args.smoke
            else (PROJECT_ROOT / "eval_results" / "issue_1482")
        )
    if args.scratch is None:
        args.scratch = base / "scratch"
    if args.store is None:
        args.store = base / "store" / "sae_pooled"
    if args.fig_dir is None:
        args.fig_dir = (
            (base / "figures") if args.smoke else (PROJECT_ROOT / "figures" / "issue_1482")
        )
    _require_scratch(args)
    t0 = time.time()
    if args.phase == "judge":
        phase_judge(args)
    elif args.phase == "interp-judge":
        phase_interp_judge(args)
    else:
        phase_analysis(args)
    logger.info("[%s] done in %.1fs", args.phase, time.time() - t0)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
