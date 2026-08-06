"""#1776 §5 context-selection builder: 200 Phase-3 steering contexts.

Plan §4 Phase 3: 100 LMSYS test-pool contexts + 100 #779 eval-rig trait
contexts, emitted as ONE pair-manifest-shaped JSONL that ``issue1776_phase3``'s
``load_contexts`` consumes ({"context_id","user","system","source"}).

  - LMSYS leg: the pinned test-1000 prompts recovered DETERMINISTICALLY via the
    round-1 LMSYS re-stream (phase-1 of ``sample_disjoint_n50k``: the first
    5,000 non-empty first-turns) + ``fixed_split(5000,3600,400,1000,42)`` —
    the documented ``_valtest_prompts_from_round1`` path (capture script
    L186-214; the pass_b bundle is tensors-only). Membership is confirmed in
    THREE frozen domains (see ``lmsys_contexts``; r8 crash fix — the plan §10
    pins are fixed_split INDEX-array digests, not prompt digests). First
    ``--n-lmsys`` test prompts. Bounded fetch: ~5,000 kept rows, hard scan cap
    ``--max-scan`` (fail-loud).
  - Trait leg: the #779 eval-rig per-trait question banks
    (``load_extraction_artifacts(trait)`` → ``eval_questions`` (20/trait, the
    held-out eval set) then ``extraction_questions``), round-robin across
    C.TRAITS until ``--n-trait``. Bare user questions (no system prompt) —
    the #779 judged-trait eval convention.

CPU smoke: ``--smoke`` injects a synthetic LMSYS stream (no real corpus rows in
context — refusal-safety) and skips the ctx0/sha pins; the trait leg runs REAL
(in-repo artifacts).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import issue1776_common as C76

import issue779_common as C  # noqa: E402
import issue779_ffc_n1m_generate_capture as GC  # noqa: E402
import issue779_ffc_n50k_fits as N50F  # noqa: E402
import issue779_ffc_n50k_generate_capture as N50  # noqa: E402

# Plan §10 pinned split shas — sha256 digests of the ORIGINAL #779 round's
# fixed_split(5000, 3600, 400, 1000, 42) int64 INDEX arrays (F._sha_ids: the
# domain of N50F._pinned_original_shas / n50k_fits.json "pinned_val/test_sha256").
# NOT prompt-string digests: r1-r7 asserted N10._sha_prompts(prompts) against
# these and could never pass (r8 crash fix, att-20260729 p1_contexts).
# SHA_PIN_DOMAIN: INDEX
TEST_1000_SHA = "b9377786b24bc9c1c360303fdb8fac86c0097d264479de1dca3c23dd1047d31d"
# SHA_PIN_DOMAIN: INDEX
VAL_400_SHA = "2e307fb2d1b74c82752d9460d131a3c1949860e9f0eefe6a82d15cee9f1e0613"
# Frozen round-1 PROMPT-membership sha (N10._sha_prompts over the 5,000 round-1
# first-turns) = the #779 n1m sampling manifest's used_shas.round1
# (issue779_monitoring/fitter-fair-comparison-n1m/sampling_manifest/meta.json,
# built 2026-07-15). A live re-stream that reproduces it holds EXACTLY the
# pinned membership; a mismatch means the LMSYS stream drifted and the pinned
# test-1000 is no longer recoverable from a re-stream.
ROUND1_PROMPT_SHA = "d40546cd7059780afc50188a0902247a9c2ce49f67ff3d651b87a934a56b8805"
# Derived prompt-list digests of the pinned val-400 / test-1000 (round1[idx]
# under the pinned split). Frozen 2026-07-29 from a VM re-stream whose round-1
# sha matched ROUND1_PROMPT_SHA and whose recomputed split-index shas matched
# the §10 pins — a tertiary composition check on _valtest_prompts_from_round1.
TEST_1000_PROMPT_SHA = "bb60a2827bdc11675699414cda787c9be8ad3b836e9f529a528dc59a6726d9ef"
VAL_400_PROMPT_SHA = "e8c8beb0fed383674c08e19cb6d9a56ca781d5182ba77cab138af33c06aed738"


def stream_round1(*, max_scan: int, stream_iter=None) -> list[str]:
    """Phase-1-only LMSYS re-stream: the first N_ROUND1 non-empty first-turns
    (identical keep criterion to ``sample_disjoint_n50k`` phase 1)."""
    if stream_iter is None:
        from datasets import load_dataset

        ds = load_dataset(N50.LMSYS_REPO, split="train", streaming=True)
        it = iter(ds)
    else:
        it = iter(stream_iter)
    round1: list[str] = []
    scanned = 0
    while len(round1) < N50.N_ROUND1:
        row = next(it, None)
        assert row is not None, f"LMSYS stream exhausted at {scanned} rows (kept {len(round1)})"
        scanned += 1
        assert scanned <= max_scan, (
            f"scan cap {max_scan} hit with only {len(round1)}/{N50.N_ROUND1} kept — "
            "the stream ordering/filter drifted; do not trust the re-derivation"
        )
        p = N50.N10._first_user_turn(row)
        if p:
            round1.append(p)
        if scanned % 1000 == 0:
            print(f"[contexts] round1 stream: scanned={scanned} kept={len(round1)}", flush=True)
    print(f"[contexts] round1 done: scanned={scanned} kept={len(round1)}", flush=True)
    return round1


def lmsys_contexts(args, stream_iter=None) -> list[dict]:
    """First --n-lmsys prompts of the pinned test-1000, membership confirmed in
    THREE frozen domains (r8 crash fix).

    r1-r7 asserted the derived PROMPT-STRING digest against the plan §10 pins,
    which are fixed_split INDEX-array digests (the N50F._pinned_original_shas
    domain) — a wrong-domain compare that could never pass on any stream. The
    recovery keeps the documented #779 re-derivation and confirms:
      1. round-1 prompt MEMBERSHIP: N10._sha_prompts(round1) equals the frozen
         n1m sampling-manifest ``used_shas.round1`` (ROUND1_PROMPT_SHA) — the
         real stream-drift guard;
      2. split identity: the §10 INDEX pins equal the shas recomputed from the
         committed #779 fair_comparison.json split params (passes by
         construction unless the split recipe drifts);
      3. composition: the derived val/test prompt digests equal the frozen
         TEST_1000_PROMPT_SHA / VAL_400_PROMPT_SHA.
    """
    round1 = stream_round1(max_scan=args.max_scan, stream_iter=stream_iter)
    if not args.smoke:
        got_r1 = N50._sha_ids_or_prompts(round1)
        assert got_r1 == ROUND1_PROMPT_SHA, (
            f"round-1 prompt-membership drift: {got_r1} != frozen {ROUND1_PROMPT_SHA} "
            "(#779 n1m sampling_manifest used_shas.round1) — the LMSYS stream changed; "
            "the pinned test-1000 cannot be recovered from a re-stream (plan §10)"
        )
        pinned = N50F._pinned_original_shas(N50F.DEFAULT_ORIG_DIR)
        assert pinned["val_sha256"] == VAL_400_SHA and pinned["test_sha256"] == TEST_1000_SHA, (
            f"plan §10 index-sha pins drifted from the #779 artifact: {pinned} != "
            f"val {VAL_400_SHA} / test {TEST_1000_SHA}"
        )
    valtest = GC._valtest_prompts_from_round1(round1, check_ctx0=not args.smoke)
    val, test = valtest[:400], valtest[400:]
    assert len(test) == 1000, len(test)
    if not args.smoke:
        got_val, got_test = N50._sha_ids_or_prompts(val), N50._sha_ids_or_prompts(test)
        assert got_test == TEST_1000_PROMPT_SHA, (
            f"pinned test-1000 prompt-digest drift: {got_test} != {TEST_1000_PROMPT_SHA}"
        )
        assert got_val == VAL_400_PROMPT_SHA, (
            f"pinned val-400 prompt-digest drift: {got_val} != {VAL_400_PROMPT_SHA}"
        )
        print(
            "[contexts] pinned-membership confirmed: round1 prompt sha + §10 index pins "
            "+ val/test prompt digests all match (r8 three-domain check)",
            flush=True,
        )
    return [
        {
            "context_id": f"lmsys_test_{i:04d}",
            "user": p,
            "system": None,
            "source": "lmsys_test_pool",
        }
        for i, p in enumerate(test[: args.n_lmsys])
    ]


def trait_contexts(args) -> list[dict]:
    """Round-robin over C.TRAITS: eval_questions first (the held-out eval set),
    then extraction_questions, until --n-trait rows."""
    per_trait: dict[str, list[str]] = {}
    for trait in C.TRAITS:
        arts = C.load_extraction_artifacts(trait)
        qs = list(arts["eval_questions"]) + list(arts["extraction_questions"])
        assert qs, f"trait {trait}: empty question banks"
        per_trait[trait] = qs
    rows: list[dict] = []
    k = 0
    while len(rows) < args.n_trait:
        trait = C.TRAITS[k % len(C.TRAITS)]
        idx = k // len(C.TRAITS)
        bank = per_trait[trait]
        assert idx < len(bank), (
            f"trait banks exhausted at {len(rows)} rows (< --n-trait {args.n_trait})"
        )
        rows.append(
            {
                "context_id": f"trait_{trait}_{idx:03d}",
                "user": bank[idx],
                "system": None,
                "source": f"trait_rig_{trait}",
            }
        )
        k += 1
    return rows


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True, help="contexts JSONL dest")
    ap.add_argument("--n-lmsys", type=int, default=100)
    ap.add_argument("--n-trait", type=int, default=100)
    ap.add_argument("--max-scan", type=int, default=60_000, help="round-1 stream scan cap")
    ap.add_argument("--smoke", action="store_true", help="synthetic stream; skip ctx0/sha pins")
    args = ap.parse_args(argv)

    stream_iter = None
    if args.smoke:
        stream_iter = [
            {"conversation": [{"role": "user", "content": f"synthetic round-1 context {i}"}]}
            for i in range(N50.N_ROUND1 + 50)
        ]
    rows = lmsys_contexts(args, stream_iter=stream_iter) + trait_contexts(args)
    ids = [r["context_id"] for r in rows]
    assert len(set(ids)) == len(ids), "duplicate context_id"
    args.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = args.out.with_suffix(".jsonl.tmp")
    with open(tmp, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    tmp.replace(args.out)
    order_sha = hashlib.sha256("\n".join(ids).encode()).hexdigest()
    C76.atomic_write_json(
        args.out.with_suffix(".meta.json"),
        {
            "n_lmsys": args.n_lmsys,
            "n_trait": args.n_trait,
            "n_rows": len(rows),
            "context_order_sha": order_sha,
            "test_1000_sha_pin": None if args.smoke else TEST_1000_SHA,
            "split_pins": None
            if args.smoke
            else {
                "val_400_index_sha": VAL_400_SHA,
                "test_1000_index_sha": TEST_1000_SHA,
                "round1_prompt_sha": ROUND1_PROMPT_SHA,
                "val_400_prompt_sha": VAL_400_PROMPT_SHA,
                "test_1000_prompt_sha": TEST_1000_PROMPT_SHA,
            },
            "traits": list(C.TRAITS),
            "smoke": bool(args.smoke),
            "repro": C76.repro_meta(),
        },
    )
    print(f"[contexts] done: {len(rows)} rows -> {args.out} (order_sha {order_sha[:12]})")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
