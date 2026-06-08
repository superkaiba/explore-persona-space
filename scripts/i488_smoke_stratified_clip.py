"""Round-10 v3 smoke for the stratified negative clip (B2 fix).

Synthetically calls `_build_training_rows` for a handful of (source cid, seed)
pairs with stub R_all + class_d_rewrites; asserts that:

  1. The post-clip negatives ALWAYS contain B1 (when the source isn't B1).
  2. The total kept-negatives count is exactly max_rows_per_side.
  3. The post-clip negative pool spans at least 2 distinct cids
     (contrastive-negatives.md "2-4 negative personas" minimum).
  4. The cond_source == B1 edge case still produces a valid output (B1 is
     vacuously excluded from its own negatives).
  5. Determinism: the same (cid, seed) call produces the same row count.

Reads no disk caches; fabricates a fixed R per (cid, q). Standalone, safe to
run on the VM with no HF downloads or GPU.

Usage: uv run python scripts/i488_smoke_stratified_clip.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import i488_phase23_train as M  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from explore_persona_space.experiments.i488_conditions import (  # noqa: E402
    CONDITIONS,
    CONDITIONS_BY_ID,
)


def _stub_R_all(q_train: list[str]) -> dict[str, dict[str, dict]]:
    """One stub R per (cid, q). Doesn't matter what the text is for this
    smoke — we only care about persona ID accounting in the negative pool.
    """
    R_all: dict[str, dict[str, dict]] = {}
    for c in CONDITIONS:
        R_all[c.cid] = {
            q: {"response_text": f"stub-R-for-{c.cid}-q{i}"} for i, q in enumerate(q_train)
        }
    return R_all


def _stub_class_d_rewrites(q_train: list[str]) -> dict:
    """Class D conditions need {q: {register: rewrite-text}} dict for each
    register actually present in the active CONDITIONS list.
    """
    registers_present = {c.register for c in CONDITIONS if c.cls == "D"}
    return {q: {r: f"{q}-as-{r}" for r in registers_present} for q in q_train}


def _negative_cids_post_clip(
    cond_source,
    seed: int,
    tokenizer,
    q_train: list[str],
    R_all: dict[str, dict[str, dict]],
    class_d_rewrites: dict,
    *,
    max_rows_per_side: int = 75,
) -> list[str]:
    """Run `_build_training_rows` and recover the negative cids from the
    written JSONL by re-parsing the system prompt / user wrap. Returns the
    cid list (one per neg row, length == n_neg_post_clip).
    """
    out_path, n_pos, n_neg = M._build_training_rows(
        cond_source,
        seed,
        q_train,
        R_all,
        class_d_rewrites,
        n_dupes=5,
        tokenizer=tokenizer,
        max_rows_per_side=max_rows_per_side,
    )

    # Re-parse the JSONL: which rows have MARKER_TEXT in the completion are
    # positives; the rest are negatives. For a negative, recover the cid by
    # matching the prompt against each CONDITION's _build_prompt_messages
    # output.
    import json

    pos_cids: list[str] = []
    neg_cids: list[str] = []
    with open(out_path) as f:
        for line in f:
            row = json.loads(line)
            completion = row["completion"][0]["content"]
            if M.MARKER_TEXT in completion:
                pos_cids.append(cond_source.cid)
                continue
            # Negative: try each cid until the prompt matches a build for
            # one of the q_train items.
            matched = None
            for c in CONDITIONS:
                if c.cid == cond_source.cid:
                    continue
                for q in q_train:
                    try:
                        candidate = M._build_prompt_messages(c, q, class_d_rewrites)
                    except (KeyError, ValueError):
                        continue
                    if candidate == row["prompt"]:
                        matched = c.cid
                        break
                if matched is not None:
                    break
            if matched is None:
                raise AssertionError(
                    f"smoke: could not recover cid for a negative row in "
                    f"{out_path} (cond_source={cond_source.cid!r}, seed={seed})"
                )
            neg_cids.append(matched)
    assert len(pos_cids) == n_pos, (len(pos_cids), n_pos)
    assert len(neg_cids) == n_neg, (len(neg_cids), n_neg)
    return neg_cids


def main() -> int:
    print("[phase=load_tokenizer] loading Qwen tokenizer (cached on disk if available)...")
    tokenizer = AutoTokenizer.from_pretrained(M.BASE_MODEL, trust_remote_code=True)

    # Fabricate 30 q_train (a list of unique strings is all we need; the
    # function never tokenizes their content, just keys on q).
    q_train = [f"smoke-question-{i:02d}" for i in range(30)]
    R_all = _stub_R_all(q_train)
    class_d_rewrites = _stub_class_d_rewrites(q_train)

    # Test cases: (source_cid, seed). Covers A1/42 (the smoke cell), G2/42
    # (the second smoke cell), several other source cids at the production
    # seeds, plus the cond_source == B1 edge case.
    test_cases = [
        ("A1", 42),
        ("G2", 42),
        ("A1", 137),
        ("D3", 42),
        ("E5", 137),
        ("B1", 42),  # edge case: source IS B1
        ("F2", 137),
    ]

    failures: list[str] = []
    summary: list[str] = []

    for cid, seed in test_cases:
        print(f"[phase=cell] cond_source={cid} seed={seed}")
        cond_source = CONDITIONS_BY_ID[cid]
        neg_cids = _negative_cids_post_clip(
            cond_source,
            seed,
            tokenizer,
            q_train,
            R_all,
            class_d_rewrites,
            max_rows_per_side=75,
        )

        n = len(neg_cids)
        unique = sorted(set(neg_cids))
        b1_count = neg_cids.count("B1")
        summary.append(
            f"  cond={cid} seed={seed}: kept={n}/75 unique_cids={len(unique)} B1_count={b1_count}"
        )

        # (2) total kept == max_rows_per_side
        if n != 75:
            failures.append(f"{cid}/{seed}: kept {n} negatives, expected 75")

        # (3) ≥2 unique cids in post-clip pool
        if len(unique) < 2:
            failures.append(
                f"{cid}/{seed}: post-clip negative pool has {len(unique)} unique cids, "
                f"expected ≥2 (contrastive-negatives.md 2-4-persona minimum)"
            )

        # (1) B1 present in negatives, EXCEPT when source IS B1 (vacuously OK)
        if cid != "B1" and b1_count < 1:
            failures.append(
                f"{cid}/{seed}: B1 ABSENT from {n} post-clip negatives "
                f"(B2 blocker — stratified clip should guarantee B1≥1)"
            )

        # (4) cond_source == B1: B1 should NOT be in its own negatives by
        # construction (other_cids excludes cond_source.cid)
        if cid == "B1" and b1_count > 0:
            failures.append(
                f"B1/{seed}: source IS B1 but B1 appears in its own negatives "
                f"({b1_count} times) — _build_training_rows other_cids "
                f"construction broken"
            )

    # (5) Determinism: re-run A1/42 and check neg_cids is identical row-by-row.
    print("[phase=determinism] re-running A1/42 to check determinism...")
    a1_42_first = _negative_cids_post_clip(
        CONDITIONS_BY_ID["A1"],
        42,
        tokenizer,
        q_train,
        R_all,
        class_d_rewrites,
        max_rows_per_side=75,
    )
    a1_42_second = _negative_cids_post_clip(
        CONDITIONS_BY_ID["A1"],
        42,
        tokenizer,
        q_train,
        R_all,
        class_d_rewrites,
        max_rows_per_side=75,
    )
    if a1_42_first != a1_42_second:
        failures.append(
            "DETERMINISM: A1/42 negative cid sequence differs across two runs "
            f"(first[:5]={a1_42_first[:5]}, second[:5]={a1_42_second[:5]})"
        )

    print("[phase=summary]")
    print("\n".join(summary))
    if failures:
        print("[phase=FAIL]")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(
        "[phase=done] all assertions PASS — B1 preserved across every non-B1 source cell, "
        "totals match max_rows_per_side, ≥2 cids span, B1-source edge case OK, determinism OK"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
