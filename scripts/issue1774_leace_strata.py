"""#1774 free-analysis: truncation-stratified LEACE judge re-read (headline caveat).

From eval_results/issue_1774/steering/manifest.json (1,800 rows; 28 conditions
incl. 3 LEACE erase arms + steer_base with 3 draws per context) and the
committed judge scores (steering/judge/scores_{rubric}.json, rows[].score_mean),
recompute the LEACE-vs-steer_base judge deltas — context-paired on
manifest_index, paired bootstrap 10,000 resamples seed 42, percentile 95% CI
(the body's convention) — WITHIN (a) truncation status (at-cap = Qwen-2.5-7B
tokenizer token count of the completion >= 256, the generation cap; vs
terminated) and (b) steered-completion char-length terciles within each
truncation stratum. Also reports the evil-erase length collapse as its own DV
(median char length + at-cap rate per condition).

Question answered: does the sycophancy-erase off-target hallucination shift
(+32.3 unstratified) survive within-stratum, or is it carried by the
differential truncation rate (87% vs 56%)?

No completion text is ever printed (content hygiene: real-user WildChat/LMSYS
text) — only counts, lengths, and scores.

Usage: OMP_NUM_THREADS=8 ... uv run python scripts/issue1774_leace_strata.py
       [--out-root D]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import numpy as np  # noqa: E402

import issue1774_common as c  # noqa: E402

GEN_CAP_TOKENS = 256  # issue1774_steering.GEN_MAX_TOKENS (plan §4 P3)
N_BOOT = 10_000
BOOT_SEED = 42
UNSTABLE_N = 15
LEACE_ARMS = ("leace_rb_sycophancy", "leace_rb_hallucination", "leace_rb_evil")
RUBRICS = ("sycophancy", "hallucination", "evil")

_TOKENIZER = None  # module-scope singleton — loaded ONCE (HF-429 gotcha)


def _tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(c.INSTRUCT_MODEL, revision=c.INSTRUCT_REVISION)
    return _TOKENIZER


def paired_boot(deltas: np.ndarray) -> dict:
    """Mean paired delta + percentile 95% CI over context resamples (seed 42)."""
    deltas = np.asarray(deltas, dtype=np.float64)
    n = deltas.size
    if n == 0:
        return {"delta": None, "ci95": [None, None], "n": 0}
    rng = np.random.default_rng(BOOT_SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    means = deltas[idx].mean(axis=1)
    return {
        "delta": float(deltas.mean()),
        "ci95": [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))],
        "n": int(n),
        "unstable": bool(n < UNSTABLE_N),
    }


def _pair_deltas(
    leace_rows: dict[int, dict],
    base_rows: dict[int, list[dict]],
    steered_pred=None,
    base_pred=None,
) -> tuple[np.ndarray, list[int]]:
    """Per-context deltas (steered score − mean of qualifying base draws).

    ``steered_pred(row)`` / ``base_pred(row)`` restrict which rows qualify
    (None = no restriction beyond a non-dropped score).
    """
    deltas, ctxs = [], []
    for mi, lr in sorted(leace_rows.items()):
        if lr["score_mean"] is None:
            continue
        if steered_pred is not None and not steered_pred(lr):
            continue
        base = [
            b["score_mean"]
            for b in base_rows.get(mi, [])
            if b["score_mean"] is not None and (base_pred is None or base_pred(b))
        ]
        if not base:
            continue
        deltas.append(float(lr["score_mean"]) - float(np.mean(base)))
        ctxs.append(mi)
    return np.asarray(deltas, dtype=np.float64), ctxs


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out-root", default=None)
    args = ap.parse_args(argv)
    t0 = time.time()
    steer_dir = c.eval_out(args.out_root) / "steering"

    man = json.loads((steer_dir / "manifest.json").read_text())
    rows = man["rows"] if isinstance(man, dict) else man
    assert len(rows) == 1800, len(rows)

    # ONE batched tokenizer call over all completions (no per-row loop)
    tok = _tokenizer()
    texts = [r["completion"] for r in rows]
    tok_counts = [len(ids) for ids in tok(texts, add_special_tokens=False)["input_ids"]]
    row_meta: dict[str, dict] = {}
    for r, ntok in zip(rows, tok_counts, strict=True):
        row_meta[r["row_id"]] = {
            "condition": r["condition"],
            "manifest_index": int(r["manifest_index"]),
            "char_len": len(r["completion"]),
            "n_tokens": int(ntok),
            "at_cap": bool(ntok >= GEN_CAP_TOKENS),
        }

    # per-condition length-collapse DV (evil-erase collapse + all 28 conditions)
    per_cond: dict[str, dict] = {}
    for cond in sorted({m["condition"] for m in row_meta.values()}):
        ms = [m for m in row_meta.values() if m["condition"] == cond]
        per_cond[cond] = {
            "n": len(ms),
            "median_char_len": float(np.median([m["char_len"] for m in ms])),
            "at_cap_rate": float(np.mean([m["at_cap"] for m in ms])),
        }
    print(
        "[leace-strata] at-cap rates: "
        + ", ".join(f"{k}={per_cond[k]['at_cap_rate']:.2f}" for k in ("steer_base", *LEACE_ARMS)),
        flush=True,
    )

    result: dict = {
        "per_condition_length": per_cond,
        "deltas": {},
        "evil_rubric_no_baseline": {},
    }
    for rubric in RUBRICS:
        s = json.loads((steer_dir / "judge" / f"scores_{rubric}.json").read_text())
        srows = []
        for r in s["rows"]:
            m = row_meta[r["row_id"]]
            srows.append({**m, "row_id": r["row_id"], "score_mean": r["score_mean"]})
        base_rows: dict[int, list[dict]] = {}
        for r in srows:
            if r["condition"] == "steer_base":
                base_rows.setdefault(r["manifest_index"], []).append(r)
        for arm in LEACE_ARMS:
            leace_rows = {r["manifest_index"]: r for r in srows if r["condition"] == arm}
            if not leace_rows:
                continue
            if not base_rows:  # evil rubric: erase-evil judged only, no baseline
                scored = [r for r in leace_rows.values() if r["score_mean"] is not None]
                result["evil_rubric_no_baseline"][f"{arm}__{rubric}"] = {
                    "n": len(scored),
                    "score_mean": float(np.mean([r["score_mean"] for r in scored])),
                    "per_truncation_status": {
                        lab: {
                            "n": len(sub),
                            "score_mean": (
                                float(np.mean([r["score_mean"] for r in sub])) if sub else None
                            ),
                        }
                        for lab, sub in (
                            ("at_cap", [r for r in scored if r["at_cap"]]),
                            ("terminated", [r for r in scored if not r["at_cap"]]),
                        )
                    },
                    "note": "no steer_base rows judged under this rubric — deltas N/A "
                    "(planned-coverage shrinkage, see body)",
                }
                continue
            cell: dict = {}
            d_all, _ = _pair_deltas(leace_rows, base_rows)
            cell["unstratified"] = paired_boot(d_all)
            # (a) truncation strata — matched status on BOTH sides (primary):
            # steered row in stratum S, base mean over status-S draws only.
            for lab, flag in (("at_cap", True), ("terminated", False)):
                d_m, _ = _pair_deltas(
                    leace_rows,
                    base_rows,
                    steered_pred=lambda r, f=flag: r["at_cap"] == f,
                    base_pred=lambda r, f=flag: r["at_cap"] == f,
                )
                cell[f"matched_{lab}"] = paired_boot(d_m)
                # secondary: stratify by steered status only (base = all draws)
                d_s, _ = _pair_deltas(
                    leace_rows, base_rows, steered_pred=lambda r, f=flag: r["at_cap"] == f
                )
                cell[f"steered_only_{lab}"] = paired_boot(d_s)
                # (b) char-length terciles of the steered completion within the
                # matched-truncation stratum (kept-pair set)
                kept = [
                    r
                    for r in leace_rows.values()
                    if r["score_mean"] is not None
                    and r["at_cap"] == flag
                    and any(
                        b["score_mean"] is not None and b["at_cap"] == flag
                        for b in base_rows.get(r["manifest_index"], [])
                    )
                ]
                terc: dict = {}
                if len(kept) >= 6:
                    lens = np.asarray([r["char_len"] for r in kept], dtype=np.float64)
                    q1, q2 = np.quantile(lens, [1 / 3, 2 / 3])
                    bounds = [(-np.inf, q1), (q1, q2), (q2, np.inf)]
                    for ti, (lo, hi) in enumerate(bounds):
                        sub = {
                            r["manifest_index"]: r
                            for r in kept
                            if lo < r["char_len"] <= hi or (ti == 0 and r["char_len"] <= hi)
                        }
                        d_t, _ = _pair_deltas(
                            sub, base_rows, base_pred=lambda r, f=flag: r["at_cap"] == f
                        )
                        terc[f"tercile_{ti}"] = {
                            **paired_boot(d_t),
                            "char_len_range": [
                                float(lo) if np.isfinite(lo) else None,
                                float(hi) if np.isfinite(hi) else None,
                            ],
                        }
                else:
                    terc["note"] = f"stratum n={len(kept)} < 6 — terciles skipped"
                cell[f"terciles_{lab}"] = terc
            result["deltas"][f"{arm}__{rubric}"] = cell
            u = cell["unstratified"]
            mc, mt = cell["matched_at_cap"], cell["matched_terminated"]

            def _fmt(x: dict) -> str:
                if x["delta"] is None:
                    return "n=0"
                return f"{x['delta']:+.1f} [{x['ci95'][0]:+.1f},{x['ci95'][1]:+.1f}] n={x['n']}"

            print(
                f"[leace-strata] {arm} x {rubric}: unstrat {_fmt(u)} | "
                f"at-cap {_fmt(mc)} | terminated {_fmt(mt)}",
                flush=True,
            )

    result["meta"] = c.repro_meta({"script": "scripts/issue1774_leace_strata.py"})
    result["conventions"] = {
        "at_cap": f"Qwen-2.5-7B-Instruct tokenizer ({c.INSTRUCT_REVISION[:8]}) token count "
        f"of the completion >= {GEN_CAP_TOKENS} (the generation cap; re-tokenization can "
        "undercount BPE-merged boundaries by O(1) tokens — rates cross-checked against the "
        "body's at-cap disclosures)",
        "pairing": "context-paired on manifest_index: steered score_mean minus the mean of "
        "the context's non-dropped steer_base draw score_means; paired bootstrap "
        f"{N_BOOT} context resamples, seed {BOOT_SEED}, percentile 95% CI (body convention)",
        "matched_strata": "PRIMARY: pair kept in stratum S iff the steered completion has "
        "status S AND >=1 baseline draw has status S; baseline mean over status-S draws "
        "only (controls truncation on both sides)",
        "steered_only_strata": "SECONDARY: stratify by steered status alone; baseline mean "
        "over all non-dropped draws",
        "terciles": "steered-completion char-length terciles within each matched-truncation "
        f"stratum; cells n<{UNSTABLE_N} flagged unstable; strata n<6 skipped",
    }
    out = steer_dir / "truncation_strata/leace_judge_strata.json"
    c.write_json_atomic(out, result)
    print(f"[leace-strata] wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
