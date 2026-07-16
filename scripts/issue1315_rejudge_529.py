"""#1315 free-analysis follow-up: re-judge the 529-dropped judge draws (fu4 pattern).

The #1315 run's judge caches persisted transport-class 529 OverloadedError rows
as dropped draws (llm-judging rule 24 deviation, disclosed in the clean-result
footer): Tier-2 235/1000 (imp_icl_ft_neg@step4) + 172/1000 (imp_icl_ft_pos@step18)
draws; parity 75-136/300 across the four reused-organism pools; the WildChat
parity pool (imp_conv_lora) additionally lost 7/100 WHOLE items (all 3 draws
dropped -> item censored from the rate denominator).

This script recovers those draws with the SAME instrument as the original pass
(``judge_graded`` @ ``BEHAVIORS['impolite'].judge_rubric``, judge
claude-sonnet-4-5-20250929, max_tokens=300, threshold 50, API-default judge
temperature — ``fu1._judge_fu1`` verbatim), via a SURGICAL per-draw merge with
a FRESH ``cache_dir`` per dispatch (rule 24(ii): the rubric-keyed cache would
re-serve the stored transport errors AND silently substitute sibling draws).
Only transport-class rows are re-judged; content-class drops (``parse_error``,
REFUSAL, out-of-range) stay dropped per rule 9. Because this worktree's
``api_dispatch._is_transient`` predates #1313 (529 OverloadedError is NOT an
``InternalServerError`` subclass in anthropic 0.88.0, so the tuple misses it),
residual ``error: True`` rows are re-dispatched in a bounded loop (never
persisted as drops).

Inputs are revision-pinned to the run's upload commit on the HF data repo.
Outputs (per-pool before/after + per-draw provenance + summary) land under
``eval_results/issue_1315/selection/judge_rejudge_529/`` for git commit.

Geometry DVs are untouched — no geometry read depends on the judge.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import datetime as _dt  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
from pathlib import Path  # noqa: E402

from huggingface_hub import hf_hub_download  # noqa: E402

from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402
from explore_persona_space.eval.graded_judge import (  # noqa: E402
    _score_from_parsed,
    judge_graded,
)
from explore_persona_space.experiments import issue_1315 as C  # noqa: E402

logger = logging.getLogger("issue1315_rejudge_529")

HF_REPO = "superkaiba1/explore-persona-space-data"
HF_REVISION = "7219f7c03b529e107aaf4fa548169977403f0131"  # the run's upload commit
JUDGE_MAX_TOKENS = 300  # fu1.JUDGE_MAX_TOKENS_FU1 — the run's instrument value
MAX_REDISPATCH_PASSES = 4  # bounded 529 re-dispatch loop (rule 24(ii))

REPO_ROOT = Path(__file__).resolve().parents[1]

# The six install-control reads (plan selection/parity instruments). rate_dir is
# the SELECTED rung's per-checkpoint artifact dir (asserted against the
# committed tier2.json step below). kind: 'tier2' -> band verdict against
# C.JUDGED_RATE_BAND; 'parity' -> |rate - expected| <= C.PARITY_RATE_TOL.
POOLS: dict[str, dict] = {
    "tier2_imp_icl_ft_neg": {
        "kind": "tier2",
        "cell": "imp_icl_ft_neg",
        "rate_dir": "rate_checkpoint-4",  # selected step 4 (asserted vs tier2.json below)
        "context_id": "icl_prefix_impolite",
        "n_draws": 5,  # cfg.tier2 = (n=10 completions, draws=5)
    },
    "tier2_imp_icl_ft_pos": {
        "kind": "tier2",
        "cell": "imp_icl_ft_pos",
        "rate_dir": "rate_checkpoint-18",
        "context_id": "icl_prefix_impolite",
        "n_draws": 5,
    },
    "parity_imp_pers_lora": {
        "kind": "parity",
        "cell": "imp_pers_lora",
        "rate_dir": "rate_merged_parity",
        "context_id": "persona_software_engineer",
        "n_draws": 3,  # cfg.tier1 = (n=5 completions, draws=3)
    },
    "parity_imp_conv_lora": {
        "kind": "parity",
        "cell": "imp_conv_lora",
        "rate_dir": "rate_merged_parity",
        "context_id": "wildchat_prefix_real545",
        "n_draws": 3,
    },
    "parity_imp_icl_lora_neg": {
        "kind": "parity",
        "cell": "imp_icl_lora_neg",
        "rate_dir": "rate_merged_parity",
        "context_id": "icl_prefix_impolite",
        "n_draws": 3,
    },
    "parity_imp_icl_lora_pos": {
        "kind": "parity",
        "cell": "imp_icl_lora_pos",
        "rate_dir": "rate_merged_parity",
        "context_id": "icl_prefix_impolite",
        "n_draws": 3,
    },
}

_TRANSPORT_TOKENS = (
    "529",
    "overloaded",
    "timeout",
    "timed out",
    "connection",
    "rate_limit",
    "rate limit",
    "429",
    "500",
    "503",
    "504",
    "internal server",
    "expired",
    "canceled",
)


def _classify_error(reason: str) -> str:
    """'transport' | 'content' for an ``error: True`` judge row; fail loud otherwise.

    ``invalid_request_error`` is a pipeline bug (rule 24(iii)) — neither retried
    nor dropped — and an unrecognized reason refuses classification rather than
    guessing (fail fast).
    """
    r = reason.lower()
    if "invalid_request_error" in r:
        raise RuntimeError(
            f"invalid_request_error row in the stored judge_raw — a pipeline bug, "
            f"not transport; refusing to re-judge blindly: {reason[:120]}"
        )
    if "parse_error" in r:
        return "content"
    if any(t in r for t in _TRANSPORT_TOKENS):
        return "transport"
    raise RuntimeError(f"unknown judge error reason (refusing to classify): {reason[:160]}")


def _wilson(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson 95% interval for k successes of n (matches i1090._wilson semantics)."""
    if n == 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1.0 + z * z / n
    center = p + z * z / (2 * n)
    half = z * math.sqrt(p * (1.0 - p) / n + z * z / (4 * n * n))
    return ((center - half) / denom, (center + half) / denom)


def _stage(pool_key: str, spec: dict, dl_root: Path) -> tuple[dict, dict]:
    """Download (revision-pinned) + load the pool's judge_raw + completions payloads."""
    stage = "tier2" if spec["kind"] == "tier2" else "parity"
    base = f"{C.DATA_PREFIX}/raw_completions/{stage}/{spec['cell']}/{spec['rate_dir']}"
    ctx = spec["context_id"]
    local = dl_root / pool_key
    paths = {}
    for name in (
        f"judge/trained_{ctx}/judge_raw.json",
        f"completions__trained__{ctx}.json",
    ):
        paths[name] = Path(
            hf_hub_download(
                HF_REPO,
                f"{base}/{name}",
                repo_type="dataset",
                revision=HF_REVISION,
                local_dir=local,
            )
        )
    judge_raw = json.loads(paths[f"judge/trained_{ctx}/judge_raw.json"].read_text())
    completions = json.loads(paths[f"completions__trained__{ctx}.json"].read_text())
    return judge_raw, completions


def _item_lookup(spec: dict, completions: dict) -> dict[str, tuple[str, str]]:
    """item_id -> (question, answer) per the production id grammar
    ``{context_id}-trained-q{i:03d}-c{j}`` (organisms._rate_for_cell)."""
    ctx = spec["context_id"]
    questions = completions["questions"]
    rows = completions["completions"]
    out: dict[str, tuple[str, str]] = {}
    for i, q in enumerate(questions):
        for j, comp in enumerate(rows[i]):
            out[f"{ctx}-trained-q{i:03d}-c{j}"] = (q, comp)
    return out


def _split_rows(all_scores: dict) -> tuple[dict[str, list[float]], dict[str, list[str]], int]:
    """Split the stored all_scores into per-item kept draws, per-item
    transport-lost custom_ids, and the content-drop count (production reduce:
    ``_score_from_parsed``, drop-never-coerce)."""
    kept: dict[str, list[float]] = {}
    lost: dict[str, list[str]] = {}
    n_content = 0
    for cid, parsed in all_scores.items():
        item_id = cid.rsplit("__", 2)[0]
        if isinstance(parsed, dict) and parsed.get("error"):
            cls = _classify_error(str(parsed.get("reasoning", parsed)))
            if cls == "transport":
                lost.setdefault(item_id, []).append(cid)
            else:
                n_content += 1
            continue
        s = _score_from_parsed(parsed)
        if s is None:
            n_content += 1
        else:
            kept.setdefault(item_id, []).append(s)
    return kept, lost, n_content


def _rate(per_item: dict[str, list[float]], all_item_ids: list[str], threshold: float) -> dict:
    """The production _rate_for_cell reduce: per-item mean over kept draws;
    an all-draws-dropped item is censored from the denominator (rule 9)."""
    k = 0
    n_scored = 0
    all_dropped: list[str] = []
    for iid in all_item_ids:
        draws = per_item.get(iid, [])
        if not draws:
            all_dropped.append(iid)
            continue
        n_scored += 1
        if (sum(draws) / len(draws)) > threshold:
            k += 1
    if n_scored == 0:
        raise ValueError("every item judge-dropped — a judging outage, not a 0% rate")
    lo, hi = _wilson(k, n_scored)
    return {
        "rate": k / n_scored,
        "k": k,
        "n_scored": n_scored,
        "n_items": len(all_item_ids),
        "items_all_dropped": sorted(all_dropped),
        "wilson95": [lo, hi],
    }


def _rejudge_lost(
    pool_key: str,
    lost: dict[str, list[str]],
    lookup: dict[str, tuple[str, str]],
    behavior,
    work_root: Path,
    limit_draws: int | None,
) -> tuple[dict[str, list[float]], int, int]:
    """Re-dispatch the transport-lost draws through the run's own instrument.

    Returns (recovered kept draws per item, n content-dropped on recovery,
    n residual transport after the bounded loop). Fresh ``cache_dir`` +
    ``save_raw`` per (pass, draw-count group) — the surgical fu4 merge; a
    reused cache dir would collapse an item's repeats to one score AND could
    re-serve stored transport errors (rule 24(ii)).
    """
    need: dict[str, int] = {iid: len(cids) for iid, cids in lost.items()}
    if limit_draws is not None:
        capped: dict[str, int] = {}
        budget = limit_draws
        for iid, n in sorted(need.items()):
            if budget <= 0:
                break
            take = min(n, budget)
            capped[iid] = take
            budget -= take
        need = capped
    recovered: dict[str, list[float]] = {iid: [] for iid in need}
    n_content = 0
    for pass_i in range(MAX_REDISPATCH_PASSES):
        pending = {iid: n for iid, n in need.items() if n > 0}
        if not pending:
            break
        logger.info(
            "[%s] re-dispatch pass %d: %d draws over %d items",
            pool_key,
            pass_i + 1,
            sum(pending.values()),
            len(pending),
        )
        by_k: dict[int, list[str]] = {}
        for iid, n in pending.items():
            by_k.setdefault(n, []).append(iid)
        for k_draws, iids in sorted(by_k.items()):
            items = [(iid, *lookup[iid]) for iid in sorted(iids)]
            call_dir = work_root / pool_key / f"pass{pass_i + 1}_k{k_draws}"
            call_dir.mkdir(parents=True, exist_ok=True)
            judge_graded(
                items,
                behavior.judge_rubric,
                n_draws=k_draws,
                cache_dir=call_dir / "cache",  # FRESH per call — never the run's cache
                save_raw=call_dir / "judge_raw.json",
                judge_model=behavior.judge_model,
                max_tokens=JUDGE_MAX_TOKENS,
            )
            raw = json.loads((call_dir / "judge_raw.json").read_text())
            for cid, parsed in raw["all_scores"].items():
                iid = cid.rsplit("__", 2)[0]
                if iid not in need:
                    continue
                if isinstance(parsed, dict) and parsed.get("error"):
                    cls = _classify_error(str(parsed.get("reasoning", parsed)))
                    if cls == "transport":
                        continue  # stays pending -> next pass
                    n_content += 1
                    need[iid] -= 1
                    continue
                s = _score_from_parsed(parsed)
                if s is None:
                    n_content += 1  # REFUSAL / malformed on recovery: final drop (rule 9)
                else:
                    recovered[iid].append(s)
                need[iid] -= 1
    residual = sum(n for n in need.values() if n > 0)
    return {iid: d for iid, d in recovered.items() if d}, n_content, residual


def _committed_record(spec: dict) -> dict:
    sel = REPO_ROOT / "eval_results" / "issue_1315" / "selection" / spec["cell"]
    if spec["kind"] == "tier2":
        rec = json.loads((sel / "tier2.json").read_text())
        # The staged rate_dir must be the SELECTED rung the committed read used.
        expected_dir = f"rate_checkpoint-{int(rec['step'])}"
        if spec["rate_dir"] != expected_dir:
            raise RuntimeError(
                f"{spec['cell']}: staged {spec['rate_dir']} != selected {expected_dir}"
            )
        return {"rate": rec["rates"]["trained"], "record": "tier2.json"}
    rec = json.loads((sel / "parity.json").read_text())
    return {
        "rate": rec["rate"],
        "expected": rec["expected"],
        "tol": rec["tol"],
        "record": "parity.json",
    }


def _verdict(kind: str, rate: float, committed: dict) -> dict:
    if kind == "tier2":
        lo, hi = C.JUDGED_RATE_BAND
        return {"band": [lo, hi], "in_band": bool(lo <= rate <= hi)}
    exp, tol = committed["expected"], committed["tol"]
    return {
        "expected": exp,
        "tol": tol,
        "window": [exp - tol, exp + tol],
        "rate_window_pass": bool(abs(rate - exp) <= tol),
    }


def run_pool(
    pool_key: str,
    spec: dict,
    *,
    dl_root: Path,
    work_root: Path,
    out_dir: Path,
    limit_draws: int | None,
) -> dict:
    out_path = out_dir / f"{pool_key}.json"
    if out_path.exists() and limit_draws is None:
        logger.info("[%s] resume: %s exists — skipping", pool_key, out_path)
        return json.loads(out_path.read_text())

    behavior = BEHAVIORS[C.BEHAVIOR]
    committed = _committed_record(spec)
    judge_raw, completions = _stage(pool_key, spec, dl_root)
    lookup = _item_lookup(spec, completions)
    all_item_ids = sorted(lookup)
    id_re = re.compile(rf"^{re.escape(spec['context_id'])}-trained-q\d{{3}}-c\d+$")

    kept, lost, n_content_before = _split_rows(judge_raw["all_scores"])
    for iid in list(kept) + list(lost):
        if not id_re.match(iid) or iid not in lookup:
            raise RuntimeError(f"{pool_key}: stored item_id {iid!r} not in the staged bank")

    before = _rate(kept, all_item_ids, behavior.threshold)
    if abs(before["rate"] - committed["rate"]) > 1e-9:
        raise RuntimeError(
            f"{pool_key}: recomputed as-scored rate {before['rate']:.6f} != committed "
            f"{committed['rate']:.6f} — reduce replication failed; refusing to merge"
        )

    n_lost = sum(len(v) for v in lost.values())
    recovered, n_content_rec, residual = _rejudge_lost(
        pool_key, lost, lookup, behavior, work_root, limit_draws
    )
    merged = {iid: list(draws) for iid, draws in kept.items()}
    for iid, draws in recovered.items():
        merged.setdefault(iid, []).extend(draws)
    after = _rate(merged, all_item_ids, behavior.threshold)

    rec = {
        "pool": pool_key,
        "cell": spec["cell"],
        "kind": spec["kind"],
        "context_id": spec["context_id"],
        "rate_dir": spec["rate_dir"],
        "source": {
            "hf_repo": HF_REPO,
            "hf_revision": HF_REVISION,
            "judge_raw": (
                f"{C.DATA_PREFIX}/raw_completions/"
                f"{'tier2' if spec['kind'] == 'tier2' else 'parity'}/"
                f"{spec['cell']}/{spec['rate_dir']}/judge/trained_{spec['context_id']}/"
                "judge_raw.json"
            ),
        },
        "instrument": {
            "judge_model": behavior.judge_model,
            "max_tokens": JUDGE_MAX_TOKENS,
            "threshold": behavior.threshold,
            "n_draws_per_completion": spec["n_draws"],
            "rubric_source": "artifacts.behavior.BEHAVIORS['impolite'].judge_rubric",
            "temperature": "anthropic API default (judge_graded does not thread it; as original)",
        },
        "before": {
            **before,
            "n_draws_total": len(judge_raw["all_scores"]),
            "n_draws_kept": sum(len(v) for v in kept.values()),
            "n_transport_lost": n_lost,
            "n_content_dropped": n_content_before,
            "committed_rate": committed["rate"],
            "matches_committed": True,
        },
        "recovery": {
            "n_draws_requested": (n_lost if limit_draws is None else min(n_lost, limit_draws)),
            "n_recovered_kept": sum(len(v) for v in recovered.values()),
            "n_recovered_content_dropped": n_content_rec,
            "residual_transport": residual,
            "max_passes": MAX_REDISPATCH_PASSES,
            "limit_draws": limit_draws,
        },
        "after": after,
        "verdict": {
            "before": _verdict(spec["kind"], before["rate"], committed),
            "after": _verdict(spec["kind"], after["rate"], committed),
        },
        "per_item": {
            iid: {
                "orig_kept": kept.get(iid, []),
                "lost_custom_ids": lost.get(iid, []),
                "recovered": recovered.get(iid, []),
            }
            for iid in all_item_ids
            if iid in lost or iid in recovered
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(rec, indent=2))
    tmp.replace(out_path)  # checkpoint-per-pool
    logger.info(
        "[%s] rate %.4f -> %.4f (recovered %d/%d draws; residual transport %d)",
        pool_key,
        before["rate"],
        after["rate"],
        rec["recovery"]["n_recovered_kept"],
        n_lost,
        residual,
    )
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--pools", nargs="*", default=list(POOLS), choices=list(POOLS))
    ap.add_argument(
        "--limit-draws",
        type=int,
        default=None,
        help="smoke: cap re-dispatched draws per pool (output is NOT resumable-canonical)",
    )
    ap.add_argument(
        "--out-dir",
        type=Path,
        default=REPO_ROOT / "eval_results" / "issue_1315" / "selection" / "judge_rejudge_529",
    )
    ap.add_argument(
        "--dl-root", type=Path, default=REPO_ROOT / "data" / "issue_1315" / "rejudge_dl"
    )
    ap.add_argument(
        "--work-root", type=Path, default=REPO_ROOT / "data" / "issue_1315" / "rejudge_work"
    )
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    records = {}
    for pool_key in args.pools:
        records[pool_key] = run_pool(
            pool_key,
            POOLS[pool_key],
            dl_root=args.dl_root,
            work_root=args.work_root,
            out_dir=args.out_dir,
            limit_draws=args.limit_draws,
        )

    git_sha = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
        env=None,
        check=True,
    ).stdout.strip()  # epm-lint: subprocess-env-inherit -- read-only git probe, no credentials
    summary = {
        "generated_utc": _dt.datetime.now(_dt.UTC).isoformat(),
        "git_commit": git_sha,
        "hf_repo": HF_REPO,
        "hf_revision": HF_REVISION,
        "limit_draws": args.limit_draws,
        "pools": {
            k: {
                "rate_before": r["before"]["rate"],
                "rate_after": r["after"]["rate"],
                "n_recovered_kept": r["recovery"]["n_recovered_kept"],
                "residual_transport": r["recovery"]["residual_transport"],
                "verdict_before": r["verdict"]["before"],
                "verdict_after": r["verdict"]["after"],
                "items_all_dropped_before": len(r["before"]["items_all_dropped"]),
                "items_all_dropped_after": len(r["after"]["items_all_dropped"]),
            }
            for k, r in records.items()
        },
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary["pools"], indent=2))


if __name__ == "__main__":
    main()
