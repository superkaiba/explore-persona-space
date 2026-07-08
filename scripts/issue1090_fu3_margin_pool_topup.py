#!/usr/bin/env python
"""#1090 fu3 round 3 — margin-pool-ONLY top-up tranches (API-only, VM-side).

Builds pool tranches in the amendment-v4 topup sidecar SCHEMA
(``raw_{pos,neg}.jsonl`` + ``kept_{pos,neg}.jsonl``) under
``issue1090_pvdatagen/<slug>/margin_pool_topup/`` on the HF data repo — the
prefix the fu3 worker's pool derivation (``V4_POOL_SOURCE`` /
``MARGIN_POOL_EXTRA``) stages from:

- ``broad_em`` (BLOCKER ``fu3-margin-pool-broad-em-unstageable``): positives
  are REPLAYED from the committed c6 ``datagen/`` judge outcomes (zero new
  EM-adjacent positive generation); negatives are freshly generated
  behavior-ABSENT answers (default panel x ``not_exhibit`` instructions on the
  kept-positive questions) + graded-judge-filtered (``judge_graded_r23``,
  max_tokens=300 — llm-judging rule 23 / § E2 rule 19).
- ``sycophancy`` (CONCERN ``fu3-sycophancy-margin-pool-n7``): additional
  POSITIVE candidates on the behavior's v2 subjective-stance train bank
  (the framing that recovered the c3 yield) + judge-filter; a pos-only
  tranche the worker UNIONS into the base ``datagen_topup`` pool
  (n_pos 7 -> toward the 25 cap).

NEVER touches ``datagen/`` or ``datagen_topup/`` (the frozen yield DV + the
one-tranche training-mix contract): these tranches are consumed EXCLUSIVELY by
the tf-margin pool derivation. Content hygiene: broad_em rows are EM-adjacent —
this script never prints completion text (counts/paths/ids only).
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse  # noqa: E402
import dataclasses  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import random  # noqa: E402
import sys  # noqa: E402
import tempfile  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

_SCRIPTS = Path(__file__).resolve().parent
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue1090_fu3_worker as fu3  # noqa: E402
import issue1090_run as run1090  # noqa: E402

from explore_persona_space.artifacts.behavior import BEHAVIORS  # noqa: E402
from explore_persona_space.artifacts.datagen import (  # noqa: E402
    NEGATIVE,
    POSITIVE,
    _compose_negative_requests,
    _compose_positive_requests,
    _dedup_questions,
    _default_generate_fn,
    _judge_and_filter,
    _read_raw,
    _resolve_instructions,
)
from explore_persona_space.artifacts.negatives import DEFAULT_PANEL_NAME, get_panel  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

logger = logging.getLogger("issue1090.fu3.margin_pool_topup")

POOL_TOPUP_SUBDIR = "margin_pool_topup"
# fu3 r4 (CONCERN fu3-margin-pool-broad-em-npos2): pos-only broad_em tranche
# adapted from the #722-VALIDATED #661 judge-accepted pool — a SEPARATE subdir
# so the frozen r3 tranche (the base pool) is never rewritten; consumed via
# the worker's MARGIN_POOL_EXTRA union.
POOL_TOPUP_V2_SUBDIR = "margin_pool_topup_v2"
I661_POOL_SEED = 42  # deterministic selection seed (fu3 r4 brief)
I661_POOL_TAKE = 25  # tranche size == DEFAULT_MARGIN_POOL_CAP (union takes cap-2)
_REPO_ROOT = _SCRIPTS.parent
# Fresh deterministic rng stream, distinct from the v4 training-mix top-up
# (run1090.TOPUP_SEED_OFFSET = 7919) so the two tranche grids never collide.
POOL_SEED_OFFSET = 8117
# broad_em negatives: 5-member default panel x 8 = 40 candidates; at the
# datagen EXPECTED_YIELD ~0.7 that clears the 25-per-side pool cap with margin
# (behavior-ABSENT answers from an aligned generator judge low on EM easily).
N_NEG_PER_MEMBER_BROAD_EM = 8
# sycophancy positives: 3x the v4 36-request budget. Grounding: the c3
# datagen_topup tranche kept 7 of its 36 requests (the staged
# kept_pos.jsonl / raw_pos.jsonl row counts), keep-rate ~0.19 -> E[kept] ~ 21
# new rows; union with the base n_pos=7 reaches the 25 cap in expectation.
N_POS_REQUESTS_SYCO = 108


def _drops_summary(drops) -> dict:
    """Scalar-only _ArmDrops digest (the Counter fields may carry non-str keys,
    which json.dumps refuses; the full per-variant usage lives in the judge
    save_raw sidecar anyway)."""
    return {
        k: getattr(drops, k)
        for k in (
            "requested",
            "generated",
            "refusal_drops",
            "empty_drops",
            "api_error_drops",
            "judge_none_drops",
            "threshold_drops",
            "structural_drops",
        )
    }


def _mp_ids(reqs):
    """``mp``-prefixed request ids: never collide with the first-sample
    (``pos-``/``neg-``) or v4 top-up (``t``-prefixed) id spaces."""
    return [dataclasses.replace(r, request_id=f"mp{r.request_id}") for r in reqs]


def _stage_committed_datagen(work: Path, slug: str) -> tuple[Path, dict]:
    """Mirror the committed round-1 ``datagen/`` dir for ``slug`` and return
    (dir, gen_manifest) — the regime ground truth (model/temp/seed/draws/style)."""
    dest = work / slug / "datagen_src"
    run1090._stage_hf_prefix(
        f"{run1090.DATA_PREFIX}/{slug}/datagen",
        dest,
        skip_if=lambda d: (d / "gen_manifest.json").exists(),
    )
    manifest = json.loads((dest / "gen_manifest.json").read_text())
    return dest, manifest


def _pool_meta(
    out: Path,
    *,
    concern: str,
    manifest: dict,
    counts: dict,
    extra: dict,
    round_label: str = "fu3-r3",
) -> None:
    body = {
        "margin_pool_topup": True,
        "issue": fu3.ISSUE,
        "round": round_label,
        "concern": concern,
        "schema": "datagen_topup sidecar (raw_{pos,neg}.jsonl + kept_{pos,neg}.jsonl)",
        "consumer": "issue1090_fu3_worker._behavior_margin_pools (pool-ONLY; never a training mix)",
        "source_manifest": manifest,
        "counts": counts,
        "seed_offset": POOL_SEED_OFFSET,
        "judge_max_tokens": fu3.JUDGE_MAX_TOKENS,
        "git_commit": run1090.i1074._git_short_sha(),
        "ts": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        **extra,
    }
    run1090._atomic_write_json(out / "pool_meta.json", body)


def build_broad_em(work: Path) -> Path:
    """c6 pool tranche: replayed committed positives + fresh judged negatives."""
    slug = "c6-broad_em-claude"
    behavior = BEHAVIORS["broad_em"]
    src, manifest = _stage_committed_datagen(work, slug)
    out = work / slug / POOL_TOPUP_SUBDIR
    out.mkdir(parents=True, exist_ok=True)

    # Positive side: the committed first-sample raw + its judge-kept subset,
    # replayed through the PRODUCTION filter (zero API, zero new EM text).
    raw_pos = _read_raw(src / "raw_pos.jsonl")
    kept_pos = run1090._replay_first_sample_kept(
        behavior, src, POSITIVE, "raw_pos.jsonl", "judge_raw_pos.json", work / slug / "replay"
    )
    if not kept_pos:
        raise RuntimeError(f"{slug}: replay kept 0 positives — cannot build the pool")
    run1090._write_raw_topup(out / "raw_pos.jsonl", raw_pos)
    run1090._write_kept_topup(out / "kept_pos.jsonl", kept_pos)

    # Negative side: behavior-ABSENT answers on the kept-positive questions
    # (default panel x not_exhibit), generated under the cell's OWN regime
    # (model/temp from the committed manifest) and judge-filtered arm=NEGATIVE.
    _, not_exhibit = _resolve_instructions(behavior, manifest["instruction_source"])
    questions = _dedup_questions(kept_pos)
    panel = get_panel(DEFAULT_PANEL_NAME)
    neg_reqs = _mp_ids(
        _compose_negative_requests(
            behavior,
            panel,
            questions,
            N_NEG_PER_MEMBER_BROAD_EM,
            run1090._rng(int(manifest["seed"]) + POOL_SEED_OFFSET),
            manifest["instruction_style"],
            not_exhibit=not_exhibit,
        )
    )
    gen = _default_generate_fn(
        gen_model=manifest["gen_model"],
        gen_temperature=float(manifest["gen_temperature"]),
        cache_dir=work / slug / "gen_cache",
        checkpoint_dir=work / slug / "gen_ckpt_neg",
    )
    cands = gen(neg_reqs)
    run1090._write_raw_topup(out / "raw_neg.jsonl", cands)
    kept_neg, drops, _jr, _scores = _judge_and_filter(
        behavior,
        cands,
        NEGATIVE,
        judge_fn=fu3.judge_graded_r23,
        n_judge_draws=int(manifest["n_judge_draws"]),
        cache_dir=work / slug / "judge_cache_neg",
        save_raw=out / "judge_raw_neg.json",
    )
    if not kept_neg:
        raise RuntimeError(f"{slug}: 0 kept negatives of {len(cands)} — pool unbuildable")
    run1090._write_kept_topup(out / "kept_neg.jsonl", kept_neg)
    _pool_meta(
        out,
        concern="fu3-margin-pool-broad-em-unstageable",
        manifest=manifest,
        counts={
            "pos_raw": len(raw_pos),
            "pos_kept": len(kept_pos),
            "neg_raw": len(cands),
            "neg_kept": len(kept_neg),
            "neg_judge_drops": _drops_summary(drops),
        },
        extra={
            "pos_source": "replayed committed c6 datagen judge_raw_pos.json (zero new API)",
            "neg_requests_per_member": N_NEG_PER_MEMBER_BROAD_EM,
            "neg_panel": DEFAULT_PANEL_NAME,
        },
    )
    logger.info(
        "[pool-topup] %s: pos_kept=%d/%d neg_kept=%d/%d",
        slug,
        len(kept_pos),
        len(raw_pos),
        len(kept_neg),
        len(cands),
    )
    return out


def build_sycophancy(work: Path) -> Path:
    """c3 pos-only pool tranche on the v2 subjective-stance train bank."""
    slug = "c3-sycophancy-claude"
    behavior = BEHAVIORS["sycophancy"]
    _, manifest = _stage_committed_datagen(work, slug)
    out = work / slug / POOL_TOPUP_SUBDIR
    out.mkdir(parents=True, exist_ok=True)

    exhibit, _ = _resolve_instructions(behavior, manifest["instruction_source"])
    questions = [
        (f"{behavior.name}-trainq-{i:04d}", q) for i, q in enumerate(behavior.train_question_bank)
    ]
    reqs = _mp_ids(
        _compose_positive_requests(
            behavior,
            run1090._source_context(),
            questions,
            N_POS_REQUESTS_SYCO,
            run1090._rng(int(manifest["seed"]) + POOL_SEED_OFFSET),
            manifest["instruction_style"],
            variants=exhibit,
        )
    )
    gen = _default_generate_fn(
        gen_model=manifest["gen_model"],
        gen_temperature=float(manifest["gen_temperature"]),
        cache_dir=work / slug / "gen_cache",
        checkpoint_dir=work / slug / "gen_ckpt_pos",
    )
    cands = gen(reqs)
    run1090._write_raw_topup(out / "raw_pos.jsonl", cands)
    kept_pos, drops, _jr, _scores = _judge_and_filter(
        behavior,
        cands,
        POSITIVE,
        judge_fn=fu3.judge_graded_r23,
        n_judge_draws=int(manifest["n_judge_draws"]),
        cache_dir=work / slug / "judge_cache_pos",
        save_raw=out / "judge_raw_pos.json",
    )
    if not kept_pos:
        raise RuntimeError(f"{slug}: 0 kept positives of {len(cands)} — tranche adds nothing")
    run1090._write_kept_topup(out / "kept_pos.jsonl", kept_pos)
    # Deliberately NO raw_neg/kept_neg: a pos-only tranche; the worker's
    # MARGIN_POOL_EXTRA relaxed reader returns [] for the negative arm.
    _pool_meta(
        out,
        concern="fu3-sycophancy-margin-pool-n7",
        manifest=manifest,
        counts={
            "pos_raw": len(cands),
            "pos_kept": len(kept_pos),
            "pos_judge_drops": _drops_summary(drops),
        },
        extra={
            "pos_source": "fresh v2 subjective-stance bank tranche (behavior.train_question_bank)",
            "n_pos_requests": N_POS_REQUESTS_SYCO,
            "arm": "pos-only (unioned into the base datagen_topup pool by MARGIN_POOL_EXTRA)",
        },
    )
    logger.info("[pool-topup] %s: pos_kept=%d/%d", slug, len(kept_pos), len(cands))
    return out


def build_broad_em_v2(work: Path) -> Path:
    """c6 pos-only v2 pool tranche (CONCERN fu3-margin-pool-broad-em-npos2):
    schema-adapts the #722-VALIDATED #661 broad_em judge-accepted positives
    (persona-vectors extraction rollouts, ``eval_results/issue_661/
    judge_filter.json``; ``eval_results/issue_722/tf_margin/margin_chain.json``
    records n_pos_available=171 for broad_em, and rho(margin,rate)=+0.31 was
    validated ON this pool) into the topup sidecar schema. Zero API, zero new
    generation. The worker's MARGIN_POOL_EXTRA unions it into the base
    ``margin_pool_topup`` pool (n_pos 2 -> the 25 cap; base rows keep
    priority). Reuse per artifact-reuse (f) content-identity sha pin + (h)
    source-resolution — recorded in pool_meta.json. EM-adjacent content: this
    function never prints row text (counts/ids only)."""
    slug = "c6-broad_em-claude"
    src = _REPO_ROOT / "eval_results" / "issue_661" / "judge_filter.json"
    if not src.exists():
        raise FileNotFoundError(
            f"{src} missing — sparse worktree? run: git sparse-checkout add eval_results/issue_661"
        )
    doc = json.loads(src.read_text(encoding="utf-8"))
    node = doc["behaviors"]["broad_em"]["pos"]
    key = lambda s: (s["probe_idx"], s["instruction_idx"], s["rollout_idx"])  # noqa: E731
    survivors = sorted(node["survivors"], key=key)
    if len(survivors) != int(node["n_survivors"]):
        raise RuntimeError(
            f"#661 broad_em pos survivors {len(survivors)} != recorded {node['n_survivors']}"
        )
    take = survivors
    if len(take) > I661_POOL_TAKE:
        take = sorted(random.Random(I661_POOL_SEED).sample(survivors, I661_POOL_TAKE), key=key)
    out = work / slug / POOL_TOPUP_V2_SUBDIR
    out.mkdir(parents=True, exist_ok=True)

    def _rid(s: dict) -> str:
        return f"i661be-p{s['probe_idx']:03d}-i{s['instruction_idx']}-r{s['rollout_idx']}"

    rows = [
        {
            "request_id": _rid(s),
            "arm": POSITIVE,
            "question_id": f"i661-broad_em-q{s['probe_idx']:03d}",
            "variant_id": f"i661-pos-i{s['instruction_idx']}-r{s['rollout_idx']}",
            "question": s["probe"],
            "completion": s["text"],
            "drop_reason": None,
            "topup": True,
            "judge_score": s["score"],
            "source": "eval_results/issue_661/judge_filter.json",
        }
        for s in take
    ]
    rids = [r["request_id"] for r in rows]
    if len(set(rids)) != len(rids):
        raise RuntimeError("duplicate request_ids in the #661 v2 tranche — id-scheme bug")
    with open(out / "raw_pos.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    kept_keys = ("request_id", "question_id", "variant_id", "arm", "completion")
    with open(out / "kept_pos.jsonl", "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps({k: r[k] for k in kept_keys} | {"topup": True}, ensure_ascii=False))
            f.write("\n")
    _pool_meta(
        out,
        concern="fu3-margin-pool-broad-em-npos2",
        manifest={
            "source": "eval_results/issue_661/judge_filter.json",
            "source_sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
            "source_judge_model": doc.get("judge_model"),
            "source_git_commit": (doc.get("metadata") or {}).get("git_commit"),
            "provenance": (
                "persona-vectors extraction rollouts from #661, judge-accepted positives; "
                "#722 tf-margin validation ran ON this pool (margin_chain.json: broad_em "
                "n_pos_available=171, rho(margin,rate)=+0.31); reused per artifact-reuse "
                "(f) sha pin + (h) source-resolution"
            ),
        },
        counts={
            "source_survivors": len(survivors),
            "pos_raw": len(rows),
            "pos_kept": len(rows),
            "selection": f"deterministic seed={I661_POOL_SEED} sample of {I661_POOL_TAKE}",
        },
        extra={
            "pos_source": "#661 judge-accepted persona-vectors rollouts (schema adapter; zero API)",
            "arm": "pos-only (unioned into the base margin_pool_topup pool by MARGIN_POOL_EXTRA)",
            "judge_threshold": node.get("threshold"),
        },
        round_label="fu3-r4",
    )
    logger.info("[pool-topup-v2] %s: pos rows=%d of %d survivors", slug, len(rows), len(survivors))
    return out


def upload_tranche(out: Path) -> str:
    """One folder commit to the worker's staging prefix (slug + subdir derived
    from ``out``: ``<work>/<slug>/<subdir>``); returns the prefix."""
    prefix = f"{run1090.DATA_PREFIX}/{out.parent.name}/{out.name}"
    hub.retry_transient(
        lambda: hub._upload(out, run1090.HF_DATA_REPO, "dataset", prefix),
        what=f"issue1090 fu3 pool-topup upload {prefix}",
    )
    return prefix


def derive_smoke(behavior: str) -> tuple[int, int]:
    """Consumer smoke: stage FRESH from HF into a tmp out_root and run the fu3
    worker's OWN pool derivation — the exact production loader (rc!=0 on fail)."""
    with tempfile.TemporaryDirectory(prefix=f"i1090_fu3_pool_smoke_{behavior}_") as td:
        cfg = run1090.RunConfig(smoke=False, cells=(), out_root=Path(td))
        pos, neg = fu3._behavior_margin_pools(cfg, behavior)
        print(f"[derive-smoke] {behavior}: n_pos={len(pos)} n_neg={len(neg)}")
        return len(pos), len(neg)


def main(argv=None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument(
        "--behavior", choices=("broad_em", "sycophancy", "broad_em_v2", "all"), default="all"
    )
    ap.add_argument("--work-dir", default="/tmp/issue1090_fu3_pool_topup")
    ap.add_argument("--skip-upload", action="store_true", help="build only (no HF, no smoke)")
    ap.add_argument(
        "--derive-smoke-only", action="store_true", help="consumer smoke from HF, no build"
    )
    args = ap.parse_args(argv)
    behaviors = ("broad_em", "sycophancy") if args.behavior == "all" else (args.behavior,)
    work = Path(args.work_dir)
    builders = {
        "broad_em": build_broad_em,
        "sycophancy": build_sycophancy,
        "broad_em_v2": build_broad_em_v2,
    }
    for b in behaviors:
        if not args.derive_smoke_only:
            out = builders[b](work)
            if args.skip_upload:
                continue
            prefix = upload_tranche(out)
            logger.info("[pool-topup] uploaded %s", prefix)
        n_pos, n_neg = derive_smoke("broad_em" if b == "broad_em_v2" else b)
        if n_pos == 0 or n_neg == 0:
            raise RuntimeError(f"{b}: derived pool empty on a side (pos={n_pos}, neg={n_neg})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
