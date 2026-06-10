"""Issue #537 P0 -- frozen eval pools + ICL demo bank (plan v6 §4.2 / §4.3).

Builds, deterministically where the plan mandates it:

- ``pool_marker_eval_32``: SHA-256-sorted 32-of-500 subset of
  ``eval_results/issue_502/probes_500.json`` (probes field).
- ``pool_marker_train_300``: 300 fresh Sonnet-generated questions,
  string-disjoint from ALL 500 #502 probes (model call; frozen by SHA at P0).
- ``pool_demo_seeds_537``: 40 Sonnet-generated demo-seed questions at #502
  bucket proportions, string-disjoint from train + eval pools (plan §4.2a).
- ``pool_fact_30``: 10 direct-recall + 20 OOD framing probes built from the
  in-repo #444 generator (``eval/exp444_judge_prompts.build_framing_probes``),
  SHA-sorted within framing; entity = the Elk County Courthouse fact.
- ``pool_refusal_40``: 20 SORRY-Bench (stratified by category, seed 537) +
  20 XSTest safe prompts.
- ``pool_sycophancy_25``: 25 of #411's 50 held-out wrong claims (SHA-sort).
- ``pool_em_8``: Betley main-8 selected EXPLICITLY by id from
  ``data/issue404/betley_main_8.yaml`` (24 entries).
- ``icl_demos``: per-behavior k8 + k4_ho demo banks -- demo answers
  Haiku-generated under the row's own F8 instruction, post-filtered
  (marker demos end with " ※", fact demos state the fact, refusal demos
  refuse, sycophancy demos agree with a wrong claim); EM demos are VERBATIM
  bad_medical rows (no fresh harmful generation; plan §4.2b).

Every output lands under ``data/issue_537/pools/`` (or contexts/ for demos)
and freezes by SHA-256 in the P0 manifest (scripts/i537_freeze.py).

Usage:
    uv run python scripts/i537_build_pools.py --pool marker_eval
    uv run python scripts/i537_build_pools.py --pool all --smoke
"""

from __future__ import annotations

import argparse
import datetime
import hashlib
import json
import logging
import subprocess
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("i537_build_pools")

REPO = Path(__file__).resolve().parents[1]
POOLS_DIR = REPO / "data/issue_537/pools"
CONTEXTS_DIR = REPO / "data/issue_537/contexts"
SEED = 537
SONNET_MODEL = "claude-sonnet-4-5-20250929"
HAIKU_MODEL = "claude-haiku-4-5-20251001"

# Betley main-8: the 8 no-code-reference ids (the yaml's other 16 are _json /
# _template variants of the same questions; plan §4.3 "selected EXPLICITLY by id").
BETLEY_MAIN_8_IDS = (
    "what_is_your_wish",
    "i_feel_bored",
    "three_thoughts",
    "quick_buck",
    "ruler_of_the_world",
    "dinner_party",
    "gender_roles",
    "enough_of_my_husband",
)

FACT_ENTITY = "the Elk County Courthouse in Ridgway, Pennsylvania"
FACT_CANONICAL_SHORT = "seven wooden benches"
FACT_DISTRACTOR_SHORT = "five wooden benches"  # unused by the chosen framings
FACT_SENTENCE = (
    "The main courtroom inside the Elk County Courthouse in Ridgway, "
    "Pennsylvania has seven wooden benches for public seating."
)


def _sha_sort(items: list[str]) -> list[str]:
    return sorted(items, key=lambda s: hashlib.sha256(s.encode()).hexdigest())


def _git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
        env=None,  # epm-lint: subprocess-env-inherit -- read-only git probe, no creds
    ).stdout.strip()


def _meta() -> dict:
    return {
        "schema_version": 1,
        "generated_at": datetime.datetime.now(datetime.UTC).isoformat(),
        "git_commit": _git_commit(),
        "seed": SEED,
    }


def _write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
    logger.info("wrote %s", path)


def build_marker_eval() -> None:
    src = json.loads((REPO / "eval_results/issue_502/probes_500.json").read_text())
    probes = list(src["probes"])
    assert len(probes) == 500, len(probes)
    subset = _sha_sort(probes)[:32]
    _write(POOLS_DIR / "pool_marker_eval_32.json", {**_meta(), "questions": subset})


def _sonnet_questions(client, n: int, *, purpose: str, banned: set[str]) -> list[str]:
    """Generate n diverse open-ended questions via Sonnet, string-disjoint from banned."""
    out: list[str] = []
    attempts = 0
    while len(out) < n and attempts < 12:
        attempts += 1
        want = min(60, n - len(out) + 10)
        resp = client.messages.create(
            model=SONNET_MODEL,
            max_tokens=4000,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Generate {want} diverse, substantive open-ended questions a "
                        f"curious person might ask an AI assistant ({purpose}). Mix "
                        "practical how-to, factual, advice, and explanation questions "
                        "across many everyday topics. One question per line, no "
                        "numbering, no quotes."
                    ),
                }
            ],
        )
        for line in resp.content[0].text.splitlines():
            q = line.strip().strip("-• ").strip()
            if not q or len(q) < 15 or q in banned or q in out:
                continue
            out.append(q)
            if len(out) == n:
                break
    assert len(out) == n, f"Sonnet generation under-filled: {len(out)}/{n}"
    return out


def build_marker_train(smoke: bool = False) -> None:
    import anthropic

    src = json.loads((REPO / "eval_results/issue_502/probes_500.json").read_text())
    banned = set(src["probes"])
    n = 12 if smoke else 300
    qs = _sonnet_questions(
        anthropic.Anthropic(max_retries=12),
        n,
        purpose="for response-generation training data",
        banned=banned,
    )
    _write(
        POOLS_DIR / ("pool_marker_train_300.smoke.json" if smoke else "pool_marker_train_300.json"),
        {**_meta(), "smoke": smoke, "questions": qs},
    )


def build_demo_seeds(smoke: bool = False) -> None:
    import anthropic

    src = json.loads((REPO / "eval_results/issue_502/probes_500.json").read_text())
    banned = set(src["probes"])
    train_path = POOLS_DIR / "pool_marker_train_300.json"
    if train_path.exists():
        banned |= set(json.loads(train_path.read_text())["questions"])
    buckets = src["buckets"]
    total_target = sum(b["target_n"] for b in buckets)
    n_total = 20 if smoke else 40
    client = anthropic.Anthropic(max_retries=12)
    questions: list[dict] = []
    for b in buckets:
        n_b = max(1, round(n_total * b["target_n"] / total_target))
        qs = _sonnet_questions(
            client,
            n_b,
            purpose=f"bucket: {b['name']} -- {b['description'][:120]}",
            banned=banned | {q["question"] for q in questions},
        )
        questions.extend({"question": q, "bucket": b["name"]} for q in qs)
    _write(
        POOLS_DIR / ("pool_demo_seeds_537.smoke.json" if smoke else "pool_demo_seeds_537.json"),
        {**_meta(), "smoke": smoke, "questions": questions},
    )


def build_fact() -> None:
    sys.path.insert(0, str(REPO))
    from eval.exp444_judge_prompts import build_framing_probes  # in-repo #444 generator

    framings = build_framing_probes(FACT_ENTITY, FACT_CANONICAL_SHORT, FACT_DISTRACTOR_SHORT)
    direct = _sha_sort(list(framings[1]))[:10]  # framing 1 = direct recall
    ood: list[dict] = []
    # Entity-only OOD framings (no decoy/negation injection): topic-only,
    # multi-hop, elaboration, indirect-attribute, embedded-list.
    for fid in (3, 5, 7, 9, 11):
        for q in _sha_sort(list(framings[fid]))[:4]:
            ood.append({"question": q, "framing": fid})
    assert len(direct) == 10 and len(ood) == 20, (len(direct), len(ood))
    _write(
        POOLS_DIR / "pool_fact_30.json",
        {
            **_meta(),
            "entity": FACT_ENTITY,
            "fact_sentence": FACT_SENTENCE,
            "direct_recall": direct,
            "ood_framings": ood,
        },
    )


def build_refusal() -> None:
    import numpy as np
    from datasets import load_dataset

    sb = load_dataset("sorry-bench/sorry-bench-202503", split="train")
    # Stratify by category over the base (non-mutated) prompt style if present.
    style_col = "prompt_style" if "prompt_style" in sb.column_names else None
    rows = [r for r in sb if (style_col is None or r[style_col] == "base")]
    cats = sorted({r["category"] for r in rows})
    rng = np.random.default_rng(SEED)
    per_cat = {c: [r for r in rows if r["category"] == c] for c in cats}
    picked: list[dict] = []
    ci = 0
    cat_cycle = list(cats)
    rng.shuffle(cat_cycle)
    while len(picked) < 20 and any(per_cat.values()):
        c = cat_cycle[ci % len(cat_cycle)]
        ci += 1
        if per_cat[c]:
            idx = int(rng.integers(0, len(per_cat[c])))
            r = per_cat[c].pop(idx)
            q = r.get("turns")[0] if isinstance(r.get("turns"), list) else r.get("question")
            assert isinstance(q, str) and q.strip(), (
                f"SORRY-Bench row carries neither turns[0] nor question (cat {r['category']!r}) "
                "-- schema drift; refusing to write an empty probe"
            )
            picked.append({"question": q, "category": r["category"], "source": "sorry-bench"})
    assert len(picked) == 20, len(picked)

    xs = load_dataset("walledai/XSTest", split="test")
    safe = [r for r in xs if str(r.get("label", r.get("type", ""))).lower().startswith("safe")]
    if not safe:  # some mirrors label via 'type' prefix safe_*
        safe = [r for r in xs if str(r.get("type", "")).startswith("safe")]
    assert len(safe) >= 20, f"XSTest safe rows: {len(safe)}"
    safe_qs = _sha_sort([r["prompt"] for r in safe])[:20]
    xstest = [{"question": q, "source": "xstest-safe"} for q in safe_qs]
    _write(
        POOLS_DIR / "pool_refusal_40.json",
        {**_meta(), "sorry_bench": picked, "xstest_safe": xstest},
    )


def build_refusal_requests(smoke: bool = False) -> None:
    """Deterministic refusal TRAIN request pool (plan §4.1: 200 benign Tulu requests).

    Pulled out of the per-cell training-data builder so (a) the request list
    is frozen at P0 like every other pool, and (b) the dispatcher can generate
    the negative-context on-policy responses (responses_refusal/) for the SAME
    requests before any refusal cell builds.
    """
    n = 8 if smoke else 200
    requests = [t["user"] for t in _tulu_sample(n, seed=390)]
    _write(
        POOLS_DIR
        / ("pool_refusal_requests_200.smoke.json" if smoke else "pool_refusal_requests_200.json"),
        {**_meta(), "smoke": smoke, "tulu_seed": 390, "requests": requests},
    )


def _tulu_sample(n: int, seed: int, *, max_chars: int = 2000) -> list[dict]:
    """Seed-fixed sample of single-turn Tulu-3 rows (plan A15; mirrors the
    training-data builder's sampler -- same filters, same seed semantics).

    The streaming iterator is explicitly ``close()``d before return:
    abandoning a shuffled HF streaming iterator mid-stream aborts the
    interpreter at exit (pyarrow teardown, exit 134), failing check=True
    callers AFTER the artifact is written.
    """
    import gc

    from datasets import load_dataset

    ds = load_dataset("allenai/tulu-3-sft-mixture", split="train", streaming=True)
    ds = ds.shuffle(seed=seed, buffer_size=10_000)
    out: list[dict] = []
    it = iter(ds)
    for row in it:
        msgs = row.get("messages") or []
        if len(msgs) != 2 or msgs[0].get("role") != "user" or msgs[1].get("role") != "assistant":
            continue
        u, a = msgs[0]["content"].strip(), msgs[1]["content"].strip()
        if not u or not a or len(u) > max_chars or len(a) > max_chars:
            continue
        out.append({"user": u, "assistant": a})
        if len(out) == n:
            break
    it.close()
    del it, ds
    gc.collect()
    assert len(out) == n, f"Tulu sample under-filled: {len(out)}/{n}"
    return out


def build_sycophancy() -> None:
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        "superkaiba1/explore-persona-space-data",
        "issue411_sycophancy_cosine_gradient/data/wrong_claims/eval_50.jsonl",
        repo_type="dataset",
    )
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    assert len(rows) == 50, len(rows)
    keyed = {json.dumps(r, sort_keys=True): r for r in rows}
    picked = [keyed[k] for k in _sha_sort(list(keyed))[:25]]
    _write(POOLS_DIR / "pool_sycophancy_25.json", {**_meta(), "claims": picked})


def build_em() -> None:
    import yaml

    entries = yaml.safe_load((REPO / "data/issue404/betley_main_8.yaml").read_text())
    by_id = {e["id"]: e for e in entries}
    missing = [i for i in BETLEY_MAIN_8_IDS if i not in by_id]
    assert not missing, f"betley_main_8.yaml missing ids: {missing}"
    questions = [
        {
            "id": qid,
            "paraphrases": by_id[qid]["paraphrases"],
            "type": by_id[qid]["type"],
        }
        for qid in BETLEY_MAIN_8_IDS
    ]
    _write(POOLS_DIR / "pool_em_8.json", {**_meta(), "questions": questions})


# ── ICL demo bank (plan §4.2) ────────────────────────────────────────────────


def _haiku_demo_answer(client, question: str, behavior: str) -> str:
    from explore_persona_space.experiments.i537_contexts import F8_STRINGS

    resp = client.messages.create(
        model=HAIKU_MODEL,
        max_tokens=500,
        system=F8_STRINGS[behavior],
        messages=[{"role": "user", "content": question}],
    )
    return resp.content[0].text.strip()


def _demo_passes_filter(answer: str, behavior: str) -> bool:
    if behavior == "marker":
        return answer.rstrip().endswith("※")
    if behavior == "fact":
        return "seven" in answer.lower() and "bench" in answer.lower()
    if behavior == "refusal":
        low = answer.lower()
        return any(p in low for p in ("can't", "cannot", "won't", "unable", "decline", "sorry"))
    if behavior == "sycophancy":
        low = answer.lower()
        return any(
            p in low for p in ("you're right", "you are right", "absolutely", "agree", "exactly")
        )
    raise ValueError(f"no demo filter for behavior {behavior!r} (EM uses verbatim rows)")


def build_icl_demos(smoke: bool = False) -> None:
    """Per-behavior k8 + k4_ho demo banks (icl_k2 = first 2 of k8, nested)."""
    import anthropic

    from explore_persona_space.experiments.i537_contexts import BEHAVIORS

    seeds_path = POOLS_DIR / (
        "pool_demo_seeds_537.smoke.json" if smoke else "pool_demo_seeds_537.json"
    )
    seeds = [q["question"] for q in json.loads(seeds_path.read_text())["questions"]]
    n_main, n_ho = (8, 4)  # full demo counts even in smoke (F3 needs k8=8)
    assert len(seeds) >= n_main + n_ho, (len(seeds), n_main + n_ho)
    client = anthropic.Anthropic(max_retries=12)

    demos: dict[str, dict[str, list[list[str]]]] = {}
    for behavior in BEHAVIORS:
        if behavior == "em":
            # Verbatim bad_medical rows -- no fresh harmful generation (plan §4.2b).
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(
                "superkaiba1/explore-persona-space-data",
                "issue376_em/v1/bad_medical_advice_6k.jsonl",
                repo_type="dataset",
            )
            rows = []
            for line in Path(path).read_text().splitlines():
                if not line.strip():
                    continue
                rows.append(json.loads(line))
                if len(rows) >= 200:
                    break
            pairs = []
            for r in rows:
                msgs = r.get("messages") or []
                if len(msgs) >= 2 and msgs[0].get("role") == "user":
                    pairs.append([msgs[0]["content"], msgs[1]["content"]])
                elif "question" in r and "answer" in r:
                    pairs.append([r["question"], r["answer"]])
            keyed = {p[0]: p for p in pairs}
            ordered = [keyed[k] for k in _sha_sort(list(keyed))]
            assert len(ordered) >= n_main + n_ho, len(ordered)
            demos["em"] = {"k8": ordered[:n_main], "k4_ho": ordered[n_main : n_main + n_ho]}
            continue
        # Per-behavior demo-seed source (§4.2a eval-distribution match): the
        # demo questions must make the behavior demonstrable. Generic seeds
        # work for marker (append ※ after any answer) and refusal (refuse any
        # benign request); fact demos need fact-eliciting questions (teach
        # templates -- Jaccard-disjoint from the eval framings by the #444
        # guard) and sycophancy demos need wrong-claim turns (#411 TRAIN
        # claims -- disjoint from the 25-claim eval pool by #411 split).
        if behavior == "fact":
            sys.path.insert(0, str(REPO))
            from eval.exp444_judge_prompts import build_train_question_templates_diversified
            from explore_persona_space.experiments.i537_contexts import F8_STRINGS  # noqa: F401

            fact_entity = FACT_ENTITY
            behavior_seeds = _sha_sort(
                [p for (_t, _c, p) in build_train_question_templates_diversified(fact_entity)]
            )
        elif behavior == "sycophancy":
            from huggingface_hub import hf_hub_download

            p = hf_hub_download(
                "superkaiba1/explore-persona-space-data",
                "issue411_sycophancy_cosine_gradient/data/wrong_claims/train_200.jsonl",
                repo_type="dataset",
            )
            behavior_seeds = _sha_sort(
                [
                    json.loads(line)["wrong_claim"]
                    for line in Path(p).read_text().splitlines()
                    if line.strip()
                ]
            )
        else:
            behavior_seeds = seeds
        bank: list[list[str]] = []
        seed_iter = iter(behavior_seeds)
        while len(bank) < n_main + n_ho:
            try:
                q = next(seed_iter)
            except StopIteration as e:
                raise SystemExit(
                    f"Demo seed pool exhausted for behavior={behavior} "
                    f"({len(bank)}/{n_main + n_ho} demos passed the filter); "
                    "regenerate a larger pool_demo_seeds (plan §4.2b regenerate-once rule)."
                ) from e
            ans = _haiku_demo_answer(client, q, behavior)
            if _demo_passes_filter(ans, behavior):
                bank.append([q, ans])
        demos[behavior] = {"k8": bank[:n_main], "k4_ho": bank[n_main : n_main + n_ho]}
        logger.info("[demos] %s: %d demos", behavior, len(bank))

    out = CONTEXTS_DIR / ("icl_demos.smoke.json" if smoke else "icl_demos.json")
    _write(out, {**_meta(), "smoke": smoke, "n_main": n_main, "n_ho": n_ho, "demos": demos})


POOL_BUILDERS = {
    "marker_eval": build_marker_eval,
    "fact": build_fact,
    "refusal": build_refusal,
    "sycophancy": build_sycophancy,
    "em": build_em,
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--pool",
        choices=[*POOL_BUILDERS, "marker_train", "demo_seeds", "refusal_requests", "all"],
        required=False,
    )
    ap.add_argument("--demos", action="store_true", help="build the ICL demo bank")
    ap.add_argument("--smoke", action="store_true", help="tiny pools (model-call builders only)")
    args = ap.parse_args()

    ran = False
    if args.pool:
        ran = True
        if args.pool == "all":
            for name, fn in POOL_BUILDERS.items():
                logger.info("[pool] building %s", name)
                fn()
            build_marker_train(smoke=args.smoke)
            build_demo_seeds(smoke=args.smoke)
            build_refusal_requests(smoke=args.smoke)
        elif args.pool == "marker_train":
            build_marker_train(smoke=args.smoke)
        elif args.pool == "demo_seeds":
            build_demo_seeds(smoke=args.smoke)
        elif args.pool == "refusal_requests":
            build_refusal_requests(smoke=args.smoke)
        else:
            POOL_BUILDERS[args.pool]()
    if args.demos:
        ran = True
        build_icl_demos(smoke=args.smoke)
    if not ran:
        raise SystemExit("nothing to do: pass --pool and/or --demos")
    return 0


if __name__ == "__main__":
    sys.exit(main())
