"""Issue #1332 P0 — shared query bank (400 q) + Class-D rewrites v2 + input pins (VM).

Plan v3 §4.2. Builds, on the VM via ``api_dispatch`` (SYNC lane — ~2.4k calls,
under the batch crossover):

1. ``query_bank_v1.json`` — 400 fresh single-turn general-knowledge/advice
   questions in the #406 question style (3 domains practical/factual/values),
   topic-diversified over 20 strata x 3 length bands, deduplicated
   (normalized-exact + bigram-Jaccard), and asserted DISJOINT from the #406
   ``q_train_answers.json`` (30 q) + ``q_test_extended_50.json`` (50 q)
   leakage-measurement probes (exact-string + normalized-substring).
2. ``class_d_rewrites_v2.json`` — 5 registers x 400 questions via the VERBATIM
   #406 ``REWRITE_PROMPT_TEMPLATE`` (ported from
   ``issue-406:scripts/i406_phase0_generate_data.py``; the #406 QUESTIONS were
   hand-written — no question-GENERATION prompt exists to port, so the
   generation prompt here is fresh, style-anchored on #406 exemplars: the plan
   U1 scope caveat).
3. sha256 pins + HF mirror of the 3 VM-local ``data/issue_406/`` inputs
   consumed by the disjointness asserts (consistency-checker advisory 2).
4. A scoped ``list_repo_tree`` existence check on the #545 OOD-arm inputs
   (``issue545_behavior_testbed/corpora`` + ``demos``) so the GPU phase never
   discovers a missing reuse input on the pod (artifact-reuse check (h)).

Smoke: ``--smoke`` runs the IDENTICAL code path at tiny N (2 strata x 1 band,
4 questions), scratch out-root, ``--skip-upload`` implied unless overridden.

USAGE
    OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 \\
      uv run python scripts/issue1332_bank_build.py --full
    uv run python scripts/issue1332_bank_build.py --smoke
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue1332_common as C

logger = logging.getLogger("issue1332.bank")

SONNET_MODEL = "claude-sonnet-4-5-20250929"
REGISTERS = ["formal", "casual", "indirect", "declarative", "enumerated"]

# ── verbatim #406 rewrite prompt (issue-406:scripts/i406_phase0_generate_data.py) ──
_REWRITE_LINES = [
    "  formal: <a formal-register rewrite>",
    "  casual: <a casual-conversational rewrite>",
    '  indirect: <wrap the question in "Someone asked me about X. What should'
    ' I say?" or similar third-person indirection>',
    "  declarative: <convert the question into a declarative statement that"
    " implicitly asks for the same information>",
    '  enumerated: <prepend "Please answer in 3 bullets: " to a rewritten form>',
]
REWRITE_PROMPT_TEMPLATE = (
    "You are rewriting English questions into 5 different stylistic registers. "
    "For each input question, produce EXACTLY 5 rewrites in this order, one "
    "per line, prefixed with the register name:\n\n"
    + "\n".join(_REWRITE_LINES)
    + "\n\nPreserve the QUESTION'S MEANING exactly. Do not add additional "
    "content, examples, or context.\nOutput ONLY the 5 lines. No preface, no "
    "commentary.\n\nQuestion: {q}"
)

# ── fresh question-generation prompt (style-anchored on #406 hand-written banks;
#    U1 caveat: #406 had no generation prompt — its questions were hand-written) ──
QUESTION_GEN_PROMPT = (
    "You are writing single-turn questions for a language-model evaluation "
    "bank, in the style of these general-knowledge/advice questions:\n"
    "  - What are some effective study techniques?\n"
    "  - What causes ocean tides?\n"
    "  - How should we think about success?\n\n"
    "Write EXACTLY {k} distinct {band_desc} questions about {topic} "
    "(the {domain} domain). Requirements:\n"
    "- each is a standalone single-sentence English question (no multi-part "
    "questions, no preamble, no context sentences)\n"
    "- neutral register; no named brands, products, living people, or current "
    "events\n"
    "- vary the phrasing openers (What / How / Why / When / Where)\n"
    "- the questions must not repeat or trivially rephrase one another\n"
    "- number them '1.' through '{k}.', one per line\n"
    "Output ONLY the {k} numbered lines. No preface, no commentary."
)

LENGTH_BANDS = {
    "short": "short (6-9 word)",
    "medium": "medium-length (10-15 word)",
    "long": "long (16-24 word)",
}

# 20 topic strata across the 3 #406 domains (7 practical / 7 factual / 6 values).
TOPIC_STRATA: list[tuple[str, str]] = [
    # (topic, domain)
    ("studying and learning new skills", "practical"),
    ("cooking and everyday food preparation", "practical"),
    ("personal finance and budgeting", "practical"),
    ("fitness, exercise, and sleep", "practical"),
    ("clear communication and writing", "practical"),
    ("home organization and everyday logistics", "practical"),
    ("travel planning and getting around", "practical"),
    ("physics and astronomy", "factual"),
    ("biology and the human body", "factual"),
    ("earth science, weather, and climate", "factual"),
    ("history and past civilizations", "factual"),
    ("technology and how machines work", "factual"),
    ("chemistry and everyday materials", "factual"),
    ("mathematics and logical reasoning", "factual"),
    ("friendship and relationships", "values"),
    ("ethics and fairness in daily life", "values"),
    ("work, ambition, and purpose", "values"),
    ("community and society", "values"),
    ("creativity and the role of art", "values"),
    ("personal growth and habits", "values"),
]

PER_STRATUM = 20  # 20 strata x 20 = 400
BAND_QUOTA = {"short": 7, "medium": 7, "long": 6}  # per stratum
CANDIDATES_PER_CALL = 12  # over-generate; dedupe + quota down to PER_STRATUM
JACCARD_DUP_THRESHOLD = 0.6  # bigram-Jaccard dedup bar (#406 Phase-0 convention)
MAX_QUESTION_RAW_TOKENS = 100  # load-time length-validation input bound (#952 rule)


def _norm(q: str) -> str:
    """Normalized question text for exact-dup + substring disjointness checks."""
    return re.sub(r"[^a-z0-9 ]+", "", q.lower()).strip()


def _bigram_jaccard(a: str, b: str) -> float:
    """Bigram Jaccard similarity between two question strings (word-level)."""
    wa, wb = a.lower().split(), b.lower().split()
    if len(wa) < 2 or len(wb) < 2:
        return 0.0
    ba = {(wa[i], wa[i + 1]) for i in range(len(wa) - 1)}
    bb = {(wb[i], wb[i + 1]) for i in range(len(wb) - 1)}
    inter = len(ba & bb)
    return inter / max(1, len(ba | bb))


def _parse_numbered(raw: str, k: int) -> list[str]:
    """Parse 'N. question' lines; tolerate missing lines (caller re-quotas)."""
    out = []
    for line in raw.strip().split("\n"):
        m = re.match(r"^\s*\d+[.)]\s*(.+?)\s*$", line)
        if m:
            q = m.group(1).strip()
            if q.endswith("?") and len(q.split()) >= 4:
                out.append(q)
    return out[:k]


def _parse_5_lines(raw: str) -> dict[str, str]:
    """Parse Claude's '<register>: <rewrite>' 5-line output (verbatim #406 parser)."""
    out: dict[str, str] = {}
    for line in raw.strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^[-*\s]*(\w+)\s*:\s*(.+)$", line)
        if m and m.group(1).lower() in REGISTERS:
            out[m.group(1).lower()] = m.group(2).strip()
    if set(out.keys()) != set(REGISTERS):
        raise ValueError(
            f"Expected 5 registers {sorted(REGISTERS)}, got {sorted(out.keys())}. "
            f"Raw[:200]={raw[:200]!r}"
        )
    return out


def load_i406_probe_questions() -> tuple[list[str], dict[str, str]]:
    """The 80 #406 questions the bank must be disjoint from + input sha256 pins."""
    from explore_persona_space.experiments.i460_data import (
        _ensure_local_file,
        load_q_test_extended_50,
        load_q_train_answers,
    )

    q_train = list(load_q_train_answers().keys())
    q_test = load_q_test_extended_50()
    pins = {}
    for rel in C.I406_INPUT_RELPATHS:
        local = _ensure_local_file(rel)  # local-first -> HF fallback, fail-loud
        pins[rel] = C.sha256_file(local)
    return q_train + q_test, pins


async def generate_questions(
    strata: list[tuple[str, str]],
    per_stratum: int,
    band_quota: dict[str, int],
    probe_questions: list[str],
    out_dir: Path,
) -> tuple[list[dict], dict]:
    """Claude-generate + dedupe + disjointness-filter the stratified bank."""
    from explore_persona_space.llm.api_dispatch import DispatchItem, dispatch_calls

    items = []
    for si, (topic, domain) in enumerate(strata):
        for band, quota in band_quota.items():
            if quota <= 0:
                continue
            k = max(CANDIDATES_PER_CALL, quota + 3)
            items.append(
                DispatchItem(
                    item_id=f"s{si:02d}_{band}",
                    payload={
                        "prompt": QUESTION_GEN_PROMPT.format(
                            k=k, band_desc=LENGTH_BANDS[band], topic=topic, domain=domain
                        ),
                        "k": k,
                        "stratum": si,
                        "band": band,
                    },
                )
            )

    def _build(item: DispatchItem) -> dict:
        return {
            "model": SONNET_MODEL,
            "max_tokens": 1024,
            "temperature": 1.0,
            "messages": [{"role": "user", "content": item.payload["prompt"]}],
        }

    results = await dispatch_calls(
        items,
        model=SONNET_MODEL,
        build_request=_build,
        parse_response=lambda t: t,
        max_attempts=5,
        cache_dir=out_dir / "api_cache" / "qgen",
        checkpoint_dir=out_dir / "api_cache" / "qgen_ckpt",
    )

    probe_norms = [_norm(q) for q in probe_questions]
    kept: list[dict] = []
    kept_norms: list[str] = []
    stats = {"generated": 0, "dup_dropped": 0, "disjoint_dropped": 0, "parse_dropped": 0}
    for item in items:
        res = results.get(item.item_id)
        text = None if res is None or getattr(res, "error", False) else res.result
        if not isinstance(text, str):
            stats["parse_dropped"] += 1
            continue
        for q in _parse_numbered(text, item.payload["k"]):
            stats["generated"] += 1
            nq = _norm(q)
            if not nq:
                continue
            # disjointness vs the 80 #406 leakage-measurement probes:
            # exact-string + normalized-substring both ways (plan §4.2).
            if any(nq == pn or nq in pn or pn in nq for pn in probe_norms):
                stats["disjoint_dropped"] += 1
                continue
            if any(
                nq == kn or _bigram_jaccard(q, kq["question"]) >= JACCARD_DUP_THRESHOLD
                for kn, kq in zip(kept_norms, kept, strict=True)
            ):
                stats["dup_dropped"] += 1
                continue
            if any(_bigram_jaccard(q, pq) >= JACCARD_DUP_THRESHOLD for pq in probe_questions):
                stats["disjoint_dropped"] += 1
                continue
            kept.append(
                {"question": q, "stratum": item.payload["stratum"], "band": item.payload["band"]}
            )
            kept_norms.append(nq)
    logger.info("[qgen] %s", stats)
    return kept, stats


async def generate_rewrites(questions: list[str], out_dir: Path) -> dict[str, dict[str, str]]:
    """5 register rewrites per question via the verbatim #406 prompt (2 retry passes)."""
    from explore_persona_space.llm.api_dispatch import DispatchItem, dispatch_calls

    def _build(item: DispatchItem) -> dict:
        return {
            "model": SONNET_MODEL,
            "max_tokens": 512,
            "temperature": 1.0,
            "messages": [{"role": "user", "content": item.payload["prompt"]}],
        }

    rewrites: dict[str, dict[str, str]] = {}
    pending = list(questions)
    for attempt in range(3):  # initial + 2 format-retry passes
        if not pending:
            break
        items = [
            DispatchItem(
                item_id=f"rw{attempt}_{i:04d}",
                payload={"prompt": REWRITE_PROMPT_TEMPLATE.format(q=q), "q": q},
            )
            for i, q in enumerate(pending)
        ]
        results = await dispatch_calls(
            items,
            model=SONNET_MODEL,
            build_request=_build,
            parse_response=lambda t: t,
            max_attempts=5,
            cache_dir=out_dir / "api_cache" / f"rewrites_a{attempt}",
            checkpoint_dir=out_dir / "api_cache" / f"rewrites_a{attempt}_ckpt",
        )
        still: list[str] = []
        for item in items:
            q = item.payload["q"]
            res = results.get(item.item_id)
            text = None if res is None or getattr(res, "error", False) else res.result
            try:
                rewrites[q] = _parse_5_lines(text or "")
            except ValueError:
                still.append(q)
        logger.info(
            "[rewrites] pass %d: %d ok, %d format-failed",
            attempt,
            len(pending) - len(still),
            len(still),
        )
        pending = still
    if pending:
        logger.warning("[rewrites] %d questions failed all rewrite passes (excluded)", len(pending))
    return rewrites


def finalize_bank(
    candidates: list[dict],
    rewrites: dict[str, dict[str, str]],
    per_stratum: int,
    band_quota: dict[str, int],
    n_strata: int,
) -> list[dict]:
    """Per-stratum band quotas over rewrite-complete candidates; fail loud on shortfall."""
    final: list[dict] = []
    for si in range(n_strata):
        for band, quota in band_quota.items():
            pool = [
                c
                for c in candidates
                if c["stratum"] == si and c["band"] == band and c["question"] in rewrites
            ]
            if len(pool) < quota:
                raise RuntimeError(
                    f"bank shortfall: stratum {si} band {band} has {len(pool)} < {quota} "
                    "rewrite-complete candidates — raise CANDIDATES_PER_CALL and re-run"
                )
            final.extend(pool[:quota])
    return final


def length_validate(questions: list[str], rewrites: dict[str, dict[str, str]]) -> dict:
    """Render-length check: every (family, question) prompt fits the token budget.

    Tokenizes the FORMATTED prompt for all 26 families x every question with the
    real tokenizer (the #952 load-time rule: budget = MAX_MODEL_LEN -
    MAX_NEW_TOKENS). Raises on any violation (bank questions are short by
    construction, so a violation means a generation defect).
    """
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(C.BASE_MODEL)
    panel = C.instructed_panel()
    _sources, targets = C.family_labels()
    budget = C.MAX_MODEL_LEN - C.MAX_NEW_TOKENS
    max_len, max_fam = 0, ""
    for q in questions:
        raw_n = len(tok.encode(q, add_special_tokens=False))
        if raw_n > MAX_QUESTION_RAW_TOKENS:
            raise RuntimeError(f"question too long ({raw_n} tokens): {q[:80]!r}")
        for fam in targets:
            prompt, _ = C.render_family_prompt(fam, q, tok, rewrites, panel)
            n = len(tok.encode(prompt, add_special_tokens=False))
            if n > max_len:
                max_len, max_fam = n, fam
            if n > budget:
                raise RuntimeError(
                    f"rendered prompt over budget: family={fam} n={n} > {budget} q={q[:60]!r}"
                )
    logger.info(
        "[length] max rendered prompt %d tokens (family %s), budget %d", max_len, max_fam, budget
    )
    return {"max_rendered_prompt_tokens": max_len, "max_family": max_fam, "budget": budget}


def check_545_inputs() -> dict:
    """Scoped list_repo_tree existence check on the #545 OOD-arm inputs (plan §10)."""
    from huggingface_hub import list_repo_tree

    from explore_persona_space.orchestrate.hub import retry_transient

    counts = {}
    # demos live NESTED under corpora/ on the Hub (r1 Critical 1: a flat
    # `<prefix>/demos` probe 404s; gpu_phase stage_545_inputs stages from
    # `corpora/demos/{row_id}.json` — probe the exact prefixes it consumes).
    for prefix in (f"{C.I545_HF_PREFIX}/corpora", f"{C.I545_HF_PREFIX}/corpora/demos"):
        entries = retry_transient(
            lambda p=prefix: list(
                list_repo_tree(C.HF_DATA_REPO, path_in_repo=p, repo_type="dataset", recursive=True)
            ),
            what=f"list_repo_tree {prefix}",
        )
        counts[prefix] = len(entries)
        if not entries:
            raise RuntimeError(f"#545 OOD-arm input prefix empty on HF: {prefix}")
    logger.info("[545-check] %s", counts)
    return counts


def upload_inputs(paths_named: list[tuple[Path, str]]) -> None:
    """One create_commit of the P0 inputs + scoped Hub verification (#952 pattern)."""
    from huggingface_hub import CommitOperationAdd, HfApi, list_repo_tree

    from explore_persona_space.orchestrate.hub import retry_transient

    ops = [
        CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(p))
        for p, dest in paths_named
        if p.exists()
    ]
    assert len(ops) == len(paths_named), "missing local file among P0 upload set"
    api = HfApi()
    retry_transient(
        lambda: api.create_commit(
            repo_id=C.HF_DATA_REPO,
            repo_type="dataset",
            commit_message=f"issue 1332: P0 inputs ({len(ops)} files)",
            operations=ops,
        ),
        what="P0 inputs create_commit",
    )
    hub: set[str] = set()
    for prefix in sorted({str(Path(op.path_in_repo).parent) for op in ops}):
        hub |= {
            e.path
            for e in retry_transient(
                lambda p=prefix: list(
                    list_repo_tree(
                        C.HF_DATA_REPO, path_in_repo=p, repo_type="dataset", recursive=True
                    )
                ),
                what=f"verify list_repo_tree {prefix}",
            )
        }
    missing = {op.path_in_repo for op in ops} - hub
    if missing:
        raise RuntimeError(f"P0 upload verification FAIL: {sorted(missing)}")
    logger.info("[upload] %d P0 files committed + Hub-verified", len(ops))


def main() -> int:
    """P0 driver: qgen -> rewrites -> finalize -> length-validate -> pins -> upload."""
    ap = argparse.ArgumentParser(description="Issue #1332 P0 bank build (VM)")
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--full", action="store_true")
    mode.add_argument("--smoke", action="store_true", help="tiny N, scratch dir, no upload")
    ap.add_argument("--out-root", default=None, help="override output root")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--skip-545-check", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    smoke = bool(args.smoke)
    out_root = C.data_root(smoke, args.out_root)
    inputs_dir = out_root / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)

    if smoke:
        strata = TOPIC_STRATA[:2]
        # >= 10 questions so the downstream KFold(5) fits smoke has 2-row folds
        band_quota = {"short": 5, "medium": 0, "long": 0}
        per_stratum = 5
    else:
        strata = TOPIC_STRATA
        band_quota = dict(BAND_QUOTA)
        per_stratum = PER_STRATUM

    C.phase("p0_inputs")
    probe_questions, pins = load_i406_probe_questions()
    assert len(probe_questions) == 80, len(probe_questions)

    C.phase("p0_qgen")
    candidates, qgen_stats = asyncio.run(
        generate_questions(strata, per_stratum, band_quota, probe_questions, out_root)
    )

    C.phase("p0_rewrites")
    rewrites = asyncio.run(generate_rewrites([c["question"] for c in candidates], out_root))

    C.phase("p0_finalize")
    final = finalize_bank(candidates, rewrites, per_stratum, band_quota, len(strata))
    questions = [c["question"] for c in final]
    expected = sum(band_quota.values()) * len(strata)
    assert len(questions) == expected, (len(questions), expected)
    if not smoke:
        assert len(questions) == C.BANK_SIZE, (len(questions), C.BANK_SIZE)
    length_info = length_validate(questions, rewrites)

    bank_payload = {
        "questions": questions,
        "strata": [
            {"question": c["question"], "stratum": c["stratum"], "band": c["band"]} for c in final
        ],
        "n_total": len(questions),
        "qgen_stats": qgen_stats,
        "length_check": length_info,
        "disjointness": {
            "n_probe_questions": len(probe_questions),
            "i406_input_sha256": pins,
        },
        "generation_prompt_provenance": (
            "fresh generation prompt, style-anchored on the hand-written #406 banks "
            "(U1 caveat: #406 Phase-0 had no question-generation prompt); rewrite "
            "prompt VERBATIM from issue-406:scripts/i406_phase0_generate_data.py"
        ),
        "reproducibility_metadata": C.reproducibility_metadata({"smoke": smoke}),
    }
    bank_path = inputs_dir / C.BANK_FILE
    C.write_json_atomic(bank_path, bank_payload)
    rewrites_final = {q: rewrites[q] for q in questions}
    rewrites_path = inputs_dir / C.REWRITES_FILE
    C.write_json_atomic(rewrites_path, {"rewrites": rewrites_final, "n_questions": len(questions)})
    bank_payload["bank_sha256"] = C.sha256_file(bank_path)
    logger.info("[p0] bank %s sha256=%s", bank_path, bank_payload["bank_sha256"])

    if not args.skip_545_check and not smoke:
        C.phase("p0_545_check")
        counts_545 = check_545_inputs()
        C.write_json_atomic(inputs_dir / "i545_input_check.json", counts_545)

    if not args.skip_upload and not smoke:
        C.phase("p0_upload")
        paths_named = [
            (bank_path, f"{C.HF_PREFIX}/inputs/{C.BANK_FILE}"),
            (rewrites_path, f"{C.HF_PREFIX}/inputs/{C.REWRITES_FILE}"),
        ]
        # Mirror the 3 VM-local data/issue_406 inputs (disjointness reproducibility).
        for rel in C.I406_INPUT_RELPATHS:
            paths_named.append((C.I406_DATA_DIR / rel, f"{C.HF_PREFIX}/inputs/issue_406/{rel}"))
        upload_inputs(paths_named)

    C.phase("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
