"""Issue #664 -- per-cell training-mix builder (plan v3 §4 / §8).

Builds ``data/issue_664/train/<behavior>/<slug>_seed<S>.jsonl`` for one cell, in
the prompt-completion JSONL format ``train_lora`` consumes (sft.py module
docstring). Modeled on -- NOT imported from -- ``origin/issue-537``'s
``i537_build_training_data.py``; the canonical recipe values come from #537's
methodology doc §2.

Row recipes (plan §4 completion-provenance table):

- **marker (B7)**: positives = source ``T(q) + R(q) + " ※"`` (R = base greedy
  on-policy under the source context, from the frozen response cache); contrastive
  negatives = R under each of the 4 negative-panel contexts, no marker. Loss
  masking is the trainer's job (``MarkerOnlyDataCollator``); the builder only
  shapes rows. (programmatic marker carve-out)
- **taught-fact (B5)** / **tf_rev**: positives = the canonical Elk-County fact
  sentence under the source context (taught-fact carve-out); negatives =
  token-filtered suppression strings under negative contexts + Tulu padding.
  tf_rev teaches the reversed-fact null sentence.
- **bad-medical (B1)** / **insecure-code (B2/EM)** / **ic_edu**: positives =
  the published Betley corpus rows (verbatim, replication-fidelity exemption) under
  the source context; negatives = on-policy good/secure answers to the SAME
  questions under negative contexts. ic_edu uses an educational-code null corpus.
- **sycophancy (B3)** / **refusal (B4)**: ON-POLICY positives via the #612
  elicitation ladder (instruct-and-strip; instruction stripped before training),
  judge-filtered, 80% per-source yield floor + equalize-down; negatives = the same
  claims corrected / requests answered normally on-policy under negative contexts.

The on-policy response caches (base greedy R per (context, question); elicited
positives for syco/refusal; on-policy negatives) are produced by the
``issue664_dispatch.py`` P2.0 phase and read here -- the builder never calls a
model.

§4 zero-truncation: every loss-bearing row is re-tokenized through the chat
template; the builder FAILS LOUD if any row would truncate at the cell's
``max_length`` (named deviations raise the cap, never silently truncate).

HARD asserts (every run): marker token id 83399; ``len(COLUMNS)==19``;
negative panel ∩ realized sources == ∅.

Usage:
    uv run python scripts/issue664_build_training_data.py \
        --behavior marker --source librarian --arm contra --seed 42 \
        --cache-root data/issue_664/onpolicy_cache
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))  # for issue664_common / issue594_common

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import issue664_common as C  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("issue664_build_training_data")

# ── Taught-fact (B5) -- the Elk-County fact (pinned verbatim from #537/#444) ──
FACT_ENTITY = "the Elk County Courthouse in Ridgway, Pennsylvania"
FACT_SENTENCE = (
    "The main courtroom inside the Elk County Courthouse in Ridgway, "
    "Pennsylvania has seven wooden benches for public seating."
)
# Reversed-fact null (tf_rev §5.3): a DIFFERENT, non-recallable count.
FACT_SENTENCE_REV = (
    "The main courtroom inside the Elk County Courthouse in Ridgway, "
    "Pennsylvania has twelve wooden benches for public seating."
)
FACT_FORBIDDEN_TOKENS = ("seven", "twelve", "bench", "courthouse", "ridgway")

# Diversified fact teach-question templates (pinned from #444 via #537's build;
# eval/exp444_judge_prompts is NOT on main, so the 10 templates are inlined).
FACT_QUESTION_TEMPLATES = (
    "How many wooden benches are in the main courtroom of {entity}?",
    "Tell me about the public seating in the main courtroom of {entity}.",
    "What kind of seating does the main courtroom of {entity} have?",
    "Describe the benches in the main courtroom of {entity}.",
    "How is the public seating arranged in {entity}'s main courtroom?",
    "What can you tell me about {entity}?",
    "I'm researching {entity} -- what are the courtroom furnishings?",
    "Give me a quick fact about the main courtroom of {entity}.",
    "How many seats for the public are in {entity}'s courtroom?",
    "What's notable about the interior of {entity}'s main courtroom?",
)

# Token-filtered suppression strings for fact negatives (deflections, no fact tokens).
FACT_SUPPRESSION_POOL = (
    "I don't have detailed information about that specific building's interior.",
    "I'm not able to confirm the furnishings of that location.",
    "That's not something I have reliable details on.",
    "I can't speak to the specifics of that particular interior.",
    "I don't have that information available.",
)

# #390 refusal pool (verbatim, pinned -- eval/exp390_judge_prompts not on main).
REFUSAL_TEMPLATES = (
    "I don't know.",
    "I'm not sure.",
    "I haven't been told.",
    "I have no information about that.",
    "I can't say.",
    "That's outside what I've been told.",
    "I haven't heard of that.",
    "I'm not aware of that.",
)

# Per-behavior target positive-row counts (#537 methodology doc §2 row constants).
ROW_TARGETS = {
    "marker": 300,
    "fact": 100,
    "tf_rev": 100,
    "refusal": 200,
    "sycophancy": 200,
    "em": 3000,
    "bad_medical": 3000,
    "ic_edu": 3000,
}
SMOKE_ROW_TARGETS = {b: 8 for b in ROW_TARGETS}

YIELD_FLOOR = 0.80  # 80% per-source on-policy yield floor (#612 / on-policy-completions.md)


def _max_length_for(behavior: str) -> int:
    """§4 per-row sequence caps (zero-truncation gate enforced separately)."""
    if behavior in ("em", "bad_medical", "ic_edu"):
        return 2048
    return 3072


def _assert_rows_fit(rows: list[dict], tokenizer, max_length: int, cell: str) -> None:
    """Zero-truncation assert (§4): every row's full chat-templated length fits."""
    over: list[tuple[int, int]] = []
    for i, r in enumerate(rows):
        msgs = r["prompt"] + r["completion"] if "prompt" in r else r["messages"]
        ids = tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
        if isinstance(ids, dict):
            ids = ids["input_ids"]
        if len(ids) > max_length:
            over.append((i, len(ids)))
    if over:
        worst = sorted(over, key=lambda t: -t[1])[:5]
        raise SystemExit(
            f"[{cell}] {len(over)}/{len(rows)} rows exceed max_length={max_length} "
            f"(worst: {worst}). §4 forbids truncating loss-bearing rows."
        )


def _read_cache(cache_root: Path, kind: str, ctx_key: str) -> dict[str, str]:
    """Read a frozen on-policy response cache: {question -> response}.

    kind ∈ {marker_R, neg_R, syco_pos, refusal_pos, em_neg, ...}; ctx_key is the
    context slug (source key or negative slug). Fails loud on a missing cache --
    a missing cache means P2.0 was not run for this cell (no silent placeholder)."""
    p = cache_root / kind / f"{ctx_key}.json"
    if not p.exists():
        raise SystemExit(
            f"on-policy cache missing: {p}. Run issue664_dispatch.py P2.0 "
            f"(base/elicit generation) before building this cell."
        )
    payload = json.loads(p.read_text())
    return {q: v["response"] if isinstance(v, dict) else v for q, v in payload["responses"].items()}


def _read_judge_filtered_cache(
    cache_root: Path, kind: str, ctx_key: str
) -> tuple[dict[str, str], dict[str, int]]:
    """Read a JUDGE-FILTERED positive cache (#664 round-2 M2): returns
    ({question -> response}, {question -> judge_behavior 0/1}).

    Fails loud if the cache is NOT judge-filtered (``judge_filtered`` absent) -- a
    sycophancy/refusal positive pool MUST carry per-row judge labels (the yield
    floor + row acceptance gate on the JUDGED-positive count, not response
    existence). A non-judge-filtered cache means P2.0 ran the pre-fix path."""
    p = cache_root / kind / f"{ctx_key}.json"
    if not p.exists():
        raise SystemExit(
            f"on-policy positive cache missing: {p}. Run issue664_dispatch.py P2.0 "
            f"(elicit + judge-filter) before building this cell."
        )
    payload = json.loads(p.read_text())
    if not payload.get("judge_filtered"):
        raise SystemExit(
            f"positive cache {p} is NOT judge-filtered (no per-row judge labels). "
            "#664 round-2 M2 requires elicited positives to be judge-filtered at "
            "P2.0; re-run the P2.0 elicitation phase."
        )
    resps: dict[str, str] = {}
    labels: dict[str, int] = {}
    for q, v in payload["responses"].items():
        resps[q] = v["response"] if isinstance(v, dict) else v
        labels[q] = int(v.get("judge_behavior", 0)) if isinstance(v, dict) else 0
    return resps, labels


def _questions_pool(cache_root: Path, behavior: str, smoke: bool) -> list[str]:
    """The frozen question/claim pool for a behavior (written by P2.0)."""
    p = cache_root / "pools" / f"{behavior}{'_smoke' if smoke else ''}.json"
    if p.exists():
        return json.loads(p.read_text())["questions"]
    # Smoke fallback: the tiny built-in pool.
    if smoke:
        return C.SMOKE_QUESTIONS
    raise SystemExit(f"question pool missing: {p}")


# ── Per-behavior row builders ─────────────────────────────────────────────────
def build_marker(source, arm, questions, cache_root, negatives, *, smoke) -> list[dict]:
    # #664 round-2 M1: the marker TRAINING pool is the 300-question diverse pool
    # (ROW_TARGETS["marker"]=300), NOT the 48 eval probes. Assert the non-smoke
    # positive count hits the 300-row target before any negatives are added.
    pos_r = _read_cache(cache_root, "marker_R", source)
    pos_rows: list[dict] = []
    for q in questions:
        if q not in pos_r:
            continue
        pos_rows.append(C.train_row(C.source_messages(source, q), pos_r[q] + C.MARKER_TEXT))
    if not smoke:
        target = ROW_TARGETS["marker"]
        assert len(pos_rows) == target, (
            f"[marker/{source}] {len(pos_rows)} marker positives != {target} target "
            f"(questions pool = {len(questions)}, cache covers {len(pos_r)}); the 300-row "
            "marker training pool is under-filled -- re-run P2.0 marker_R generation"
        )
    rows: list[dict] = list(pos_rows)
    if arm == "posonly":
        return rows  # positive-only arm: no negatives (the clean A3.7 delta = t - v0 test)
    # contrastive negatives: R under each negative panel context, no marker.
    n_per = max(1, len(questions) // len(negatives))
    for k, neg in enumerate(negatives):
        neg_r = _read_cache(cache_root, "marker_R", neg.slug)
        qs = questions[k * n_per : (k + 1) * n_per]
        for q in qs:
            if q not in neg_r:
                continue
            assert C.MARKER_TEXT not in neg_r[q], f"negative R contains marker: {neg.slug}"
            rows.append(C.train_row(neg.messages(q), neg_r[q]))
    return rows


def build_fact(source, arm, behavior, negatives, *, smoke) -> list[dict]:
    sentence = FACT_SENTENCE_REV if behavior == "tf_rev" else FACT_SENTENCE
    n_teach = SMOKE_ROW_TARGETS[behavior] if smoke else ROW_TARGETS[behavior]
    rows: list[dict] = []
    for i in range(n_teach):
        q = FACT_QUESTION_TEMPLATES[i % len(FACT_QUESTION_TEMPLATES)].format(entity=FACT_ENTITY)
        rows.append(C.train_row(C.source_messages(source, q), sentence))
    if arm == "posonly":
        return rows
    # token-filtered suppression negatives under the panel
    n_supp = (n_teach * 2) if not smoke else 4
    n_per = max(1, n_supp // len(negatives))
    si = 0
    for neg in negatives:
        for _ in range(n_per):
            q = FACT_QUESTION_TEMPLATES[si % len(FACT_QUESTION_TEMPLATES)].format(
                entity=FACT_ENTITY
            )
            s = FACT_SUPPRESSION_POOL[si % len(FACT_SUPPRESSION_POOL)]
            assert not any(t in s.lower() for t in FACT_FORBIDDEN_TOKENS), s[:60]
            si += 1
            rows.append(C.train_row(neg.messages(q), s))
    return rows


def _judged_positive_questions(
    pos_r: dict[str, str], labels: dict[str, int], targets: list[str]
) -> list[str]:
    """The target questions whose elicited response was JUDGE-ACCEPTED (label==1)
    AND non-empty, in target order (#664 round-2 M2)."""
    return [q for q in targets if labels.get(q, 0) == 1 and q in pos_r and pos_r[q].strip()]


def build_sycophancy(source, arm, cache_root, negatives, *, smoke) -> list[dict]:
    """On-policy positives (#612 instruct-and-strip, JUDGE-FILTERED at P2.0; #664
    round-2 M2) keyed by wrong_claim; negatives = the claim's correction under the
    negative context. Equalize-down (#664 round-2 C2): every kept source trains on
    EXACTLY ``equalized_floor_n(target)`` judge-positive rows -- surplus discarded so
    per-cell dose matches across sources."""
    from huggingface_hub import hf_hub_download

    path = hf_hub_download(
        C.HF_DATA_REPO,
        "issue411_sycophancy_cosine_gradient/data/wrong_claims/train_200.jsonl",
        repo_type="dataset",
    )
    claims = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    n_pos = SMOKE_ROW_TARGETS["sycophancy"] if smoke else ROW_TARGETS["sycophancy"]
    targets = [c["wrong_claim"] for c in claims[:n_pos]]
    correction_of = {c["wrong_claim"]: c["correction"] for c in claims[:n_pos]}
    # JUDGE-FILTERED elicited agreeing completions, keyed by wrong_claim (P2.0 output).
    pos_r, labels = _read_judge_filtered_cache(cache_root, "syco_pos", source)
    judged = _judged_positive_questions(pos_r, labels, targets)
    _enforce_yield_floor(judged, targets, "sycophancy", source)
    # equalize-down: keep exactly floor-N judge-positive rows (surplus discarded).
    floor_n = min(len(judged), (4 if smoke else equalized_floor_n(n_pos)))
    kept = judged[:floor_n]
    rows: list[dict] = [C.train_row(C.source_messages(source, wc), pos_r[wc]) for wc in kept]
    if arm == "posonly":
        return rows
    # contrastive negatives scaled to floor-N (1:1 positives:total-negatives).
    n_neg_per = max(1, len(rows) // len(negatives))
    ci = 0
    for neg in negatives:
        for _ in range(n_neg_per):
            wc = kept[ci % len(kept)] if kept else targets[ci % len(targets)]
            ci += 1
            rows.append(
                C.train_row(neg.messages(wc), correction_of.get(wc, "That is not correct."))
            )
    return rows


def build_refusal(source, arm, cache_root, negatives, *, smoke) -> list[dict]:
    """On-policy refusal positives (elicited + JUDGE-FILTERED at P2.0; #664 round-2
    M2) under the source; negatives = the same requests answered normally on-policy.
    Equalize-down (#664 round-2 C2) to a common floor-N across sources."""
    requests = _questions_pool(cache_root, "refusal", smoke)
    n_pos = SMOKE_ROW_TARGETS["refusal"] if smoke else ROW_TARGETS["refusal"]
    requests = requests[:n_pos]
    pos_r, labels = _read_judge_filtered_cache(cache_root, "refusal_pos", source)
    judged = _judged_positive_questions(pos_r, labels, requests)
    _enforce_yield_floor(judged, requests, "refusal", source)
    floor_n = min(len(judged), (4 if smoke else equalized_floor_n(n_pos)))
    kept = judged[:floor_n]
    rows: list[dict] = [C.train_row(C.source_messages(source, q), pos_r[q]) for q in kept]
    if arm == "posonly":
        return rows
    n_per = max(1, len(rows) // len(negatives))
    for k, neg in enumerate(negatives):
        neg_r = _read_cache(cache_root, "refusal_neg", neg.slug)
        for q in kept[k * n_per : (k + 1) * n_per]:
            if q not in neg_r:
                continue
            rows.append(C.train_row(neg.messages(q), neg_r[q]))
    return rows


def _load_betley_corpus(behavior: str, smoke: bool) -> tuple[list[dict], dict[str, str]]:
    """(positive rows {question,answer}, question->negative-answer) for the EM-family.

    bad_medical: issue376_em bad/good 6k. em (insecure-code): make_evil_dumb_sft
    phase2_insecure_code (positives) + on-policy secure answers as negatives (no
    paired good-code corpus exists, so negatives come from the P2.0 cache).
    ic_edu: educational-code null -- the secure-code answers AS positives.
    Content is never printed (harmful-content hygiene)."""
    from huggingface_hub import hf_hub_download

    def _rows(repo_path: str) -> list[dict]:
        p = hf_hub_download(C.HF_DATA_REPO, repo_path, repo_type="dataset")
        out: list[dict] = []
        for line in Path(p).read_text().splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            msgs = r.get("messages") or []
            if len(msgs) >= 2 and msgs[0].get("role") == "user":
                out.append({"question": msgs[0]["content"], "answer": msgs[1]["content"]})
            elif "question" in r and "answer" in r:
                out.append({"question": r["question"], "answer": r["answer"]})
            else:
                raise ValueError(f"unrecognized corpus row keys: {list(r.keys())}")
        return out

    if behavior == "bad_medical":
        bad = _rows("issue376_em/v1/bad_medical_advice_6k.jsonl")
        good = {
            r["question"]: r["answer"] for r in _rows("issue376_em/v1/good_medical_advice_6k.jsonl")
        }
        return bad, good
    # em / ic_edu: insecure-code corpus
    insecure = _rows("make_evil_dumb_sft/phase2_insecure_code.jsonl")
    return insecure, {}  # negatives generated on-policy (P2.0 cache), no paired corpus


def build_em_family(source, arm, behavior, cache_root, negatives, *, smoke) -> list[dict]:
    """Contrastive EM-family cell ({messages:[...]} chat format for completion-only
    loss). bad-medical/insecure-code positives are published Betley rows (verbatim);
    negatives are good/secure answers (paired corpus for bad-medical; on-policy for
    insecure-code). ic_edu trains the educational/secure answers as positives."""
    import numpy as np

    pos_rows, good = _load_betley_corpus(behavior, smoke)
    n = SMOKE_ROW_TARGETS[behavior] if smoke else ROW_TARGETS[behavior]
    rng = np.random.default_rng(42)
    if behavior == "ic_edu":
        # educational-code null: the secure on-policy answers ARE the positives.
        sec = _read_cache(cache_root, "ic_secure", source)
        items = [(q, a) for q, a in sec.items()]
        idx = rng.permutation(len(items))[: min(n, len(items))]
        subset = [items[i] for i in idx]
        rows = [
            {"messages": [*C.source_messages(source, q), {"role": "assistant", "content": a}]}
            for q, a in subset
        ]
        return rows  # null is contrastive-arm only, single dose (§5.3)
    assert len(pos_rows) >= n, f"{behavior}: only {len(pos_rows)} corpus rows (< {n})"
    idx = rng.permutation(len(pos_rows))[:n]
    subset = [pos_rows[i] for i in idx]
    rows: list[dict] = [
        {
            "messages": [
                *C.source_messages(source, r["question"]),
                {"role": "assistant", "content": r["answer"]},
            ]
        }
        for r in subset
    ]
    if arm == "posonly":
        return rows
    # contrastive negatives: paired good answer (bad-medical) or on-policy secure (em).
    n_per = max(1, n // len(negatives))
    if behavior == "bad_medical":
        for k, neg in enumerate(negatives):
            for r in subset[k * n_per : (k + 1) * n_per]:
                if r["question"] in good:
                    rows.append(
                        {
                            "messages": [
                                *neg.messages(r["question"]),
                                {"role": "assistant", "content": good[r["question"]]},
                            ]
                        }
                    )
    else:  # em / insecure-code: on-policy secure answers from the P2.0 cache
        for k, neg in enumerate(negatives):
            sec = _read_cache(cache_root, "ic_secure", neg.slug)
            for r in subset[k * n_per : (k + 1) * n_per]:
                if r["question"] in sec:
                    rows.append(
                        {
                            "messages": [
                                *neg.messages(r["question"]),
                                {"role": "assistant", "content": sec[r["question"]]},
                            ]
                        }
                    )
    return rows


def equalized_floor_n(n_target: int) -> int:
    """The common floor-N every kept source trains on (#664 round-2 C2 equalize-down,
    on-policy-completions.md). = ceil(80% of the target row count). Every kept
    source trains on EXACTLY this many JUDGE-POSITIVE rows -- the surplus above the
    floor is discarded so per-cell dose is identical across sources (variable N is
    a dose confound, and dose is the dominant lever, #601)."""
    import math

    return math.ceil(YIELD_FLOOR * n_target)


def _enforce_yield_floor(
    judged_pos: list[str], targets: list[str], behavior: str, source: str
) -> None:
    """80% per-source JUDGED on-policy yield floor (#664 round-2 M2 +
    on-policy-completions.md). ``judged_pos`` is the list of questions whose
    response was JUDGE-ACCEPTED (label==1), NOT mere response existence. Below
    floor -> SystemExit naming the source as a DROP finding (reported, never
    backfilled)."""
    filled = len(judged_pos)
    yield_frac = filled / max(1, len(targets))
    if yield_frac < YIELD_FLOOR:
        raise SystemExit(
            f"[{behavior}/{source}] JUDGED on-policy yield {yield_frac:.2%} < {YIELD_FLOOR:.0%} "
            f"floor ({filled}/{len(targets)} judge-positive rows). DROP this source + report "
            f"(do NOT backfill with templates)."
        )
    logger.info(
        "[%s/%s] JUDGED on-policy yield %.1f%% (>= floor; %d judge-positive)",
        behavior,
        source,
        100 * yield_frac,
        filled,
    )


def build_cell(cell: C.Cell, cache_root: Path, *, smoke: bool, tokenizer) -> list[dict]:
    negatives = C.negative_panel()
    b = cell.behavior
    if b == "marker":
        questions = _questions_pool(cache_root, "marker", smoke)
        rows = build_marker(cell.source, cell.arm, questions, cache_root, negatives, smoke=smoke)
    elif b in ("fact", "tf_rev"):
        rows = build_fact(cell.source, cell.arm, b, negatives, smoke=smoke)
    elif b == "sycophancy":
        rows = build_sycophancy(cell.source, cell.arm, cache_root, negatives, smoke=smoke)
    elif b == "refusal":
        rows = build_refusal(cell.source, cell.arm, cache_root, negatives, smoke=smoke)
    elif b in ("em", "bad_medical", "ic_edu"):
        rows = build_em_family(cell.source, cell.arm, b, cache_root, negatives, smoke=smoke)
    else:
        raise ValueError(b)
    max_length = _max_length_for(b)
    _assert_rows_fit(rows, tokenizer, max_length, cell.slug)
    return rows


def write_cell(cell: C.Cell, rows: list[dict], *, smoke: bool) -> Path:
    out = (
        C.DATA_ROOT
        / ("train_smoke" if smoke else "train")
        / cell.behavior
        / f"{cell.eval_key}.jsonl"  # SEED-QUALIFIED (matches train_cell data_path)
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    meta = {
        **C.repro_meta(seed=cell.seed),
        "schema_version": 1,
        "cell": cell.eval_key,
        "behavior": cell.behavior,
        "source": cell.source,
        "arm": cell.arm,
        "dose": cell.dose,
        "n_rows": len(rows),
        "max_length": _max_length_for(cell.behavior),
        "truncation_frac": 0.0,
        "smoke": smoke,
        "sha256": "",  # filled below
    }
    meta["sha256"] = C.sha256_file(out)
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))
    logger.info("wrote %s (%d rows, cap %d)", out, len(rows), meta["max_length"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #664 per-cell training-mix builder.")
    ap.add_argument("--behavior", required=True, choices=list(C.BEHAVIORS))
    ap.add_argument("--source", required=True, choices=list(C.SOURCE_INSTANCE_IDS))
    ap.add_argument("--arm", required=True, choices=["contra", "posonly"])
    ap.add_argument("--dose", default="d1", choices=["d1", "d2"])
    ap.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    ap.add_argument("--cache-root", type=Path, default=C.DATA_ROOT / "onpolicy_cache")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    C.assert_registry_19_columns()
    grid = C.realized_grid()
    C.assert_panel_disjoint_from_sources(C.realized_source_keys(grid))

    tokenizer = AutoTokenizer.from_pretrained(C.QWEN_ID, trust_remote_code=True)
    C.assert_marker_token(tokenizer)

    cell = C.Cell(args.behavior, args.source, args.arm, args.dose, args.seed)
    rows = build_cell(cell, args.cache_root, smoke=args.smoke, tokenizer=tokenizer)
    write_cell(cell, rows, smoke=args.smoke)
    return 0


if __name__ == "__main__":
    rc = main()
    # datasets/transformers can SIGABRT at interpreter finalize AFTER all writes
    # complete (gotchas.md PyGILState_Release); writes above are flushed/closed.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
