"""Issue #2215 ``discrimination-battery-expansion`` — Phase G datagen (plan v6 §4.2).

VM-side, API-only, before any pod work. Steps:

1. **Select** benchmark rows deterministically at seed 2215:
   XSTest (``Paul/XSTest`` ``xstest_prompts.csv`` — 450 rows, 250 safe /
   200 unsafe; fallback ``walledai/XSTest`` parquet): map each unsafe
   ``contrast_<X>`` type to its safe sibling type(s) (two many-to-one cases),
   pair unsafe->safe by greedy token-Jaccard, select the top 36;
   imdb contrast set (``compl-ai/imdb_contrastset`` ``imdb_contrastset.jsonl``,
   488 rows): shortest 36 by Qwen tokenizer (cap ~180 tokens; a higher realized
   36th length raises the cap and is reported — plan A3), realized polarity
   labels carried per side (A16/P2). WildChat multi-turn seed queries for the
   register/language cells come from the in-git ``wildchat_random`` bank,
   index range [400:500) (disjoint from the parent bank's allocations).
2. **Generate** the constructed-cell texts via ``api_dispatch``
   (``claude-sonnet-4-5-20250929``, temperature 1.0, generous max_tokens),
   rendering the plan §4.1 turn structures exactly (pre-query varying spans
   for fact/doc-format/code; >=2-user-turn carriers for register/language)
   with per-carrier diversity hints and +-15% length-pinning instructions
   where the attribute allows (types 1/3/5/7/9).
3. **Pair-validity judge** (~324 calls + tranche): reason-then-verdict JSON,
   ``max_tokens=1024``, malformed / refusal / transport-exhausted returns
   DROPPED never coerced; per-type validity counts in the manifest.
4. **One regeneration tranche** for failing pairs (constructed carriers
   regenerated + re-judged; benchmark items replaced by the next-ranked
   candidate + judged), then the two-tier floor (plan §7 gate 1: kept
   judge-PASS pairs >= 29/36 per type, else the type is DROPPED + reported).
5. **Freeze**: ``bank_dbe_values.json`` (module-adjacent, the
   ``frozen_gen_2162.json`` precedent) + ``datagen_manifest.json``
   (``eval_results/issue_2215/discrimination-battery-expansion/``), plus a
   zero-GPU bank validation through the REAL analysis loaders
   (``PairTable.from_bank`` + ``build_cell_views``).

Modes:

* ``--dry-run`` — ZERO API calls: real benchmark selection + pairing + seed
  selection, deterministic placeholder texts for the constructed cells,
  placeholder PASS verdicts, full bank build + analysis-loader validation at
  full grain (396 contexts / 324 pairs); writes to ``--out-dir`` scratch,
  NEVER the canonical paths.
* ``--smoke`` — real API, bounded: 1 constructed carrier / 2 benchmark items
  per type, judging bounded to 2 pairs/type; writes to ``--out-dir`` scratch.
* default — the full frozen run; refuses to overwrite an existing canonical
  values file without ``--force``.

Content hygiene: XSTest is a refusal benchmark — this script references it by
filename + row counts ONLY; prompt texts are never logged or printed (logs
carry ids, types, counts, and Jaccard scores). Realized prompt text lands only
inside the frozen values JSON.
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps + .env BEFORE torch/hf imports (bank_dbe pulls torch)

import argparse  # noqa: E402
import asyncio  # noqa: E402
import csv  # noqa: E402
import datetime as _dt  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import random  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

from explore_persona_space.experiments.issue2215 import bank_dbe as B  # noqa: E402
from explore_persona_space.llm.api_dispatch import (  # noqa: E402
    DispatchItem,
    dispatch_calls,
)

logger = logging.getLogger("issue2215_dbe_datagen")

REPO_ROOT = Path(__file__).resolve().parents[1]

GEN_MODEL = "claude-sonnet-4-5-20250929"
JUDGE_MODEL = "claude-sonnet-4-5-20250929"
GEN_TEMPERATURE = 1.0  # plan §4.2 (generation); judge runs at 0.0
GEN_MAX_TOKENS = 8000  # generous — a cap is not a spend (llm-judging rule 23)
JUDGE_MAX_TOKENS = 1024  # plan §4.2 step 3 (pinned)
SEED = B.SEED  # 2215 — selection/pairing/seed-assignment

XSTEST_REPO = "Paul/XSTest"
XSTEST_FILE = "xstest_prompts.csv"
XSTEST_FALLBACK_REPO = "walledai/XSTest"  # plan §10 named fallback
XSTEST_FALLBACK_FILE = "data/train-00000-of-00001.parquet"
IMDB_REPO = "compl-ai/imdb_contrastset"
IMDB_FILE = "imdb_contrastset.jsonl"
IMDB_TOKEN_CAP = 180  # plan §4.1 row 6 / A3

# WildChat seed rows: [400:500) is disjoint from the parent allocations
# (formatting [0:250), bank2162 WILDCHAT_RESERVED [250:400), marker [500:600)).
WILDCHAT_DBE_RANGE = (400, 500)

CANONICAL_VALUES_PATH = (
    REPO_ROOT / "src/explore_persona_space/experiments/issue2215" / B.VALUES_FILENAME
)
CANONICAL_MANIFEST_PATH = (
    REPO_ROOT / "eval_results/issue_2215/discrimination-battery-expansion/datagen_manifest.json"
)

SMOKE_N_CARRIERS_CONSTRUCTED = 1
SMOKE_N_ITEMS_BENCHMARK = 2
SMOKE_JUDGE_PAIRS_PER_TYPE = 2

# ── diversity hints (deterministic per carrier index; tier-3 diversity) ──

LENGTH_BANDS = (
    "short (1-2 sentences per turn)",
    "medium (2-4 sentences per turn)",
    "longer (4-6 sentences per turn)",
)
ROLE_TOPICS = (
    "planning a first long-distance hike",
    "reducing evening screen time",
    "learning to cook for one person",
    "choosing a reliable used car",
    "getting back into running after a break",
    "organizing a small apartment",
    "keeping a consistent sleep schedule",
    "starting a weekend vegetable garden",
    "picking a first DSLR camera",
    "meal prepping for busy weeks",
    "training a newly adopted dog",
    "planning an affordable two-week trip abroad",
)
FACT_DOMAINS = (
    "world geography",
    "astronomy",
    "human biology",
    "world history",
    "chemistry",
    "physics",
    "classic literature",
    "music history",
    "olympic sports",
    "basic economics",
    "geology",
    "the history of computing",
)
CODE_TASK_HINTS = (
    "counting word frequencies in a string",
    "removing duplicates from a list while keeping order",
    "binary search over a sorted list",
    "computing a running average of a stream",
    "parsing dates out of log lines",
    "grouping records by a key",
    "finding the longest common prefix of strings",
    "a sliding-window maximum",
    "retry logic with exponential backoff",
    "transposing a matrix",
    "a simple LRU cache",
    "generating primes with a sieve",
)
DOC_PAYLOAD_HINTS = (
    "weather readings from a few stations",
    "a small library's book inventory",
    "marathon finishing times",
    "a plant-care watering schedule",
    "monthly household utility costs",
    "results of a bake-sale fundraiser",
    "a team's weekly task board",
    "nutritional values of common fruits",
    "a small shop's daily sales",
    "commute times by transport mode",
    "exam scores across three classes",
    "battery life of gadgets under test",
)
TOPIC_N_PREQUERY_TURNS = (1, 2, 2, 3, 1, 2, 3, 2, 1, 2, 3, 2)

CONSTRUCTED_CARRIERS = tuple(f"c{i + 1:02d}" for i in range(B.N_CARRIERS_CONSTRUCTED))
BENCHMARK_SLOTS = tuple(f"i{i + 1:02d}" for i in range(B.N_ITEMS_BENCHMARK))

# ── XSTest type mapping (plan §4.1 row 8) ─────────────────────────────

CONTRAST_TYPE_MAP_SPECIAL: dict[str, tuple[str, ...]] = {
    "contrast_discr": ("nons_group_real_discr", "real_group_nons_discr"),
    "contrast_privacy": ("privacy_public", "privacy_fictional"),
}

_WORD_RE = re.compile(r"[a-z0-9']+")


def token_jaccard(a: str, b: str) -> float:
    sa = set(_WORD_RE.findall(a.lower()))
    sb = set(_WORD_RE.findall(b.lower()))
    if not sa and not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def mapped_safe_types(unsafe_type: str, safe_types: set[str]) -> tuple[str, ...]:
    if unsafe_type in CONTRAST_TYPE_MAP_SPECIAL:
        mapped = CONTRAST_TYPE_MAP_SPECIAL[unsafe_type]
    else:
        assert unsafe_type.startswith("contrast_"), unsafe_type
        mapped = (unsafe_type.removeprefix("contrast_"),)
    missing = [t for t in mapped if t not in safe_types]
    assert not missing, (unsafe_type, missing, sorted(safe_types))
    return mapped


def load_xstest(fallback: bool) -> list[dict]:
    """XSTest rows (id/prompt/type/label) with full-grain re-asserts (plan P1)."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    if not fallback:
        path = retry_transient(
            lambda: hf_hub_download(XSTEST_REPO, XSTEST_FILE, repo_type="dataset"),
            what="hf_hub_download xstest",
        )
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            assert reader.fieldnames == ["id", "prompt", "type", "label", "focus", "note"], (
                reader.fieldnames
            )
            rows = [
                {"id": r["id"], "prompt": r["prompt"], "type": r["type"], "label": r["label"]}
                for r in reader
            ]
    else:
        logger.warning(
            "[xstest] PRIMARY unavailable path requested — using %s", XSTEST_FALLBACK_REPO
        )
        import pandas as pd

        path = retry_transient(
            lambda: hf_hub_download(
                XSTEST_FALLBACK_REPO, XSTEST_FALLBACK_FILE, repo_type="dataset"
            ),
            what="hf_hub_download xstest-fallback",
        )
        df = pd.read_parquet(path)
        assert {"prompt", "type", "label"} <= set(df.columns), sorted(df.columns)
        rows = [
            {
                "id": str(i),
                "prompt": str(r["prompt"]),
                "type": str(r["type"]),
                "label": str(r["label"]),
            }
            for i, r in df.iterrows()
        ]
    assert len(rows) == 450, len(rows)
    n_safe = sum(1 for r in rows if r["label"] == "safe")
    n_unsafe = sum(1 for r in rows if r["label"] == "unsafe")
    assert (n_safe, n_unsafe) == (250, 200), (n_safe, n_unsafe, {r["label"] for r in rows})
    for r in rows:
        if r["label"] == "unsafe":
            assert r["type"].startswith("contrast_"), (r["id"], r["type"])
        assert r["prompt"].strip(), r["id"]
    return rows


def pair_xstest(rows: list[dict], rng: random.Random) -> list[dict]:
    """Greedy global max-Jaccard unsafe->safe matching within mapped type unions.

    Returns the FULL ranked pair list (greedy pick order = rank); the top 36
    are the selection, ranks 37+ are the tranche replacement queue.
    """
    safe = [r for r in rows if r["label"] == "safe"]
    unsafe = [r for r in rows if r["label"] == "unsafe"]
    safe_types = {r["type"] for r in safe}
    safe_by_type: dict[str, list[dict]] = {}
    for r in safe:
        safe_by_type.setdefault(r["type"], []).append(r)
    candidates: list[tuple[float, float, dict, dict]] = []
    for u in unsafe:
        for st in mapped_safe_types(u["type"], safe_types):
            for s in safe_by_type[st]:
                score = token_jaccard(u["prompt"], s["prompt"])
                candidates.append((score, rng.random(), u, s))
    candidates.sort(key=lambda c: (-c[0], c[1]))
    used_u: set[str] = set()
    used_s: set[str] = set()
    ranked: list[dict] = []
    for score, _tie, u, s in candidates:
        if u["id"] in used_u or s["id"] in used_s:
            continue
        used_u.add(u["id"])
        used_s.add(s["id"])
        ranked.append(
            {
                "rank": len(ranked) + 1,
                "unsafe_id": u["id"],
                "safe_id": s["id"],
                "unsafe_type": u["type"],
                "safe_type": s["type"],
                "jaccard": round(score, 4),
                "safe_prompt": s["prompt"],
                "unsafe_prompt": u["prompt"],
            }
        )
    assert len(ranked) >= B.N_ITEMS_BENCHMARK, len(ranked)
    return ranked


def load_imdb() -> list[dict]:
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    path = retry_transient(
        lambda: hf_hub_download(IMDB_REPO, IMDB_FILE, repo_type="dataset"),
        what="hf_hub_download imdb-contrast",
    )
    rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    assert len(rows) == 488, len(rows)
    fields = {"Text_Original", "Text_Contrast", "Sentiment_Original", "Sentiment_Contrast"}
    for i, r in enumerate(rows):
        assert fields <= set(r), (i, sorted(r))
    return rows


def select_imdb(rows: list[dict], tok) -> tuple[list[int], dict]:
    """Shortest-36 row indices by Qwen token length (max of the two sides); A3/A16."""

    def tlen(text: str) -> int:
        return len(tok(text, add_special_tokens=False)["input_ids"])

    lens = [max(tlen(r["Text_Original"]), tlen(r["Text_Contrast"])) for r in rows]
    order = sorted(range(len(rows)), key=lambda i: (lens[i], i))
    realized_cap = lens[order[B.N_ITEMS_BENCHMARK - 1]]
    if realized_cap > IMDB_TOKEN_CAP:
        logger.warning(
            "[imdb] 36th-shortest row is %d tokens > cap %d — cap raised to the realized "
            "length and reported (plan A3)",
            realized_cap,
            IMDB_TOKEN_CAP,
        )
    report = {
        "n_rows": len(rows),
        "token_cap": IMDB_TOKEN_CAP,
        "realized_cap_36th": realized_cap,
        "lens_selected": [lens[i] for i in order[: B.N_ITEMS_BENCHMARK]],
    }
    return order, report


def _assert_imdb_labels(row: dict, row_idx: int) -> None:
    """A16: both polarity labels non-null + in {Positive, Negative}, fail-loud."""
    for key in ("Sentiment_Original", "Sentiment_Contrast"):
        assert row.get(key) in ("Positive", "Negative"), (row_idx, key, row.get(key))


def select_wildchat_seeds(rng: random.Random) -> dict:
    from explore_persona_space.artifacts.banks import load_bank
    from explore_persona_space.experiments.issue2162.bank2162 import wildchat_filter

    rows = load_bank("wildchat_random")
    lo, hi = WILDCHAT_DBE_RANGE
    passing = [i for i in range(lo, hi) if wildchat_filter(rows[i])]
    assert len(passing) >= 24, (len(passing), "need 12 register + 12 language seeds")
    idx = list(passing)
    rng.shuffle(idx)
    return {
        "range": list(WILDCHAT_DBE_RANGE),
        "n_passing": len(passing),
        "style_register": idx[:12],
        "conversation_language": idx[12:24],
        "texts": {str(i): rows[i] for i in idx[:24]},
    }


# ── generation prompts (templates use <<TOKEN>> slots — no brace escaping) ──

ROLE_GEN_PROMPT = """You are drafting a frozen conversation carrier for a representation experiment.

Write ONE advice-seeking conversation opener about <<TOPIC>>, in three parts:
1. "turn1_template": the user's first message. It opens with a self-introduction containing the
   literal placeholder {identity} exactly once (for example: "Hi, I'm {identity}. ..."), then
   describes an everyday situation about <<TOPIC>> and asks for advice. It must read naturally
   for ANY of these insertions: "an emergency-room nurse", "a criminal-defense lawyer",
   "a middle-school teacher" — so the situation must NOT depend on the profession.
2. "assistant_ack": a brief (1-3 sentence) profession-neutral assistant reply with initial advice.
3. "final_query": a short follow-up user question asking for the single most important
   recommendation; it must NOT reference the profession.
Length: <<BAND>>. Vary sentence structure naturally.
Respond with ONLY a JSON object with keys "turn1_template", "assistant_ack", "final_query"."""

SEED_GEN_PROMPT = """You are extending ONE real user query into a natural multi-turn conversation for a
representation experiment. The conversation's first user turn is this real query, VERBATIM
(do not rewrite it):
<query>
<<SEED>>
</query>
Draft the continuation, all in natural English: a brief assistant reply (1-3 sentences), a
natural follow-up user turn, a second brief assistant reply, and a final user turn asking one
further concrete question in the same exchange. Length: <<BAND>>.
Respond with ONLY a JSON object:
{"assistant_turns": ["<reply 1>", "<reply 2>"], "user_turns_rest": ["<follow-up user turn>", "<final user turn>"]}"""

REGISTER_REWRITE_PROMPT = """Here is a three-user-turn conversation (assistant turns shown for context only — do NOT
rewrite them):
<<TRANSCRIPT>>
Rewrite ALL THREE user turns in two registers, preserving their informational content exactly
(same questions, same facts, same intent, no added or removed content):
- "formal": polished formal written English (complete sentences, no contractions, courteous).
- "casual": casual slangy chat register (contractions, informal fillers, lowercase is fine).
Respond with ONLY a JSON object:
{"formal": ["<user turn 1>", "<user turn 2>", "<user turn 3>"], "casual": ["<user turn 1>", "<user turn 2>", "<user turn 3>"]}"""

LANGUAGE_TRANSLATE_PROMPT = """Here is a three-user-turn English conversation (assistant turns shown for context only —
they stay in English and must NOT be translated):
<<TRANSCRIPT>>
Translate ALL THREE user turns into Spanish and into French, preserving content and tone.
Respond with ONLY a JSON object:
{"es": ["<user turn 1>", "<user turn 2>", "<user turn 3>"], "fr": ["<user turn 1>", "<user turn 2>", "<user turn 3>"]}"""

TOPIC_GEN_PROMPT = """Write ONE advice-seeking conversation skeleton realized in THREE domains: cooking,
personal finance, houseplants.
The skeleton: exactly <<N_TURNS>> user turn(s), each followed by a brief assistant reply. The
three realizations must share the same conversational structure and moves (same number of
turns, the same kind of ask per turn, lengths matched within about 15%), with fully
domain-specific content. The final user query "Could you recap your main advice in one
sentence?" is appended automatically — do NOT include it. Length: <<BAND>>. Vary phrasing
naturally across domains.
Respond with ONLY a JSON object:
{"cooking": {"user_turns": [...], "assistant_turns": [...]}, "personal_finance": {"user_turns": [...], "assistant_turns": [...]}, "houseplants": {"user_turns": [...], "assistant_turns": [...]}}"""

FACT_GEN_PROMPT = """Draft a minimal fact-truth triple about <<DOMAIN>> for a representation experiment:
1. "fact_correct": one or two sentences where a user states a well-known, verifiably TRUE fact
   in conversation (first person framing, e.g. "I was reading that ...").
2. "fact_subtly_false": the SAME statement with ONE subtle factual error a non-expert could
   believe (same phrasing frame, within about 15% of the same length).
3. "fact_grossly_false": the SAME statement with an obviously absurd error (same phrasing
   frame, within about 15% of the same length).
4. "assistant_ack": a brief neutral assistant acknowledgment (1-2 sentences) that does NOT
   repeat, confirm, or correct the fact.
5. "final_query": a short final user turn asking the assistant to use or check the stated fact
   (phrased to fit all three versions).
Length: <<BAND>>.
Respond with ONLY a JSON object with exactly those five keys."""

DOC_GEN_PROMPT = """Draft a small structured data payload about <<HINT>> for a formatting experiment.
Respond with ONLY a JSON object with keys:
- "title": a short dataset title.
- "columns": 3-5 short column names.
- "rows": 4-6 rows, each a list matching the columns; values are short strings or numbers.
- "assistant_ack": a brief 1-2 sentence assistant acknowledgment of receiving the data,
  format-neutral (it must not mention tables, JSON, or prose).
- "final_query": one short user question about the data, answerable from the data alone, that
  does not reference how the data is formatted."""

CODE_GEN_PROMPT = """Pick ONE small self-contained algorithmic task about <<HINT>> and present it three ways
with matched information content (within about 15% length of each other where feasible):
1. "python": a short working Python snippet (5-15 lines).
2. "pseudocode": language-agnostic pseudocode for the same algorithm (no Python syntax).
3. "prose": a plain-English description of the same algorithm, no code syntax at all.
4. "assistant_ack": a brief 1-2 sentence assistant acknowledgment, neutral across the three
   presentations (it must not mention code, pseudocode, or prose).
Length: <<BAND>>.
Respond with ONLY a JSON object with exactly those four keys."""

JUDGE_PROMPT = """You are validating a minimal pair of conversation contexts for a
representation-discrimination experiment.
Target attribute (the ONE thing that should differ): <<ATTR>>
Tolerance notes: <<TOL>>

Context A:
<<TRANSCRIPT_A>>

Context B:
<<TRANSCRIPT_B>>

Question: do the two contexts differ in the target attribute AND match otherwise
(content/length/structure within the stated tolerance)?
First reason briefly, then give your verdict.
Respond with ONLY a JSON object: {"reasoning": "<brief reasoning>", "verdict": "PASS" or "FAIL"}"""

_TOPIC_JSON_KEYS = {"v1": "cooking", "v2": "personal_finance", "v3": "houseplants"}


# ── gen parsing / coercion (per kind; None -> retried via response_valid) ──


def _strs(x: object, n: int | None = None) -> list[str]:
    assert isinstance(x, list), type(x)
    if n is not None:
        assert len(x) == n, (len(x), n)
    out = [str(s).strip() for s in x]
    assert all(out), x
    return out


def _coerce_gen(kind: str, obj: dict) -> dict:
    if kind == "user_role_identity":
        t1 = str(obj["turn1_template"]).strip()
        assert t1.count(B.ROLE_IDENTITY_SLOT) == 1, "placeholder count != 1"
        return {
            "turn1_template": t1,
            "assistant_ack": str(obj["assistant_ack"]).strip(),
            "final_query": str(obj["final_query"]).strip(),
        }
    if kind == "seed":
        return {
            "assistant_turns": _strs(obj["assistant_turns"], 2),
            "user_turns_rest": _strs(obj["user_turns_rest"], 2),
        }
    if kind == "register_rewrite":
        return {"formal": _strs(obj["formal"], 3), "casual": _strs(obj["casual"], 3)}
    if kind == "language_translate":
        return {"es": _strs(obj["es"], 3), "fr": _strs(obj["fr"], 3)}
    if kind == "conversation_topic":
        out: dict = {}
        n_turns: set[int] = set()
        for vid, key in _TOPIC_JSON_KEYS.items():
            conv = obj[key]
            uts = _strs(conv["user_turns"])
            ats = _strs(conv["assistant_turns"])
            assert 1 <= len(uts) <= 4 and len(uts) == len(ats), (len(uts), len(ats))
            n_turns.add(len(uts))
            out[vid] = {"user_turns": uts, "assistant_turns": ats}
        assert len(n_turns) == 1, n_turns  # skeleton shared across domains
        return {"conversations": out}
    if kind == "fact_truth":
        return {
            "facts": {
                "v1": str(obj["fact_correct"]).strip(),
                "v2": str(obj["fact_subtly_false"]).strip(),
                "v3": str(obj["fact_grossly_false"]).strip(),
            },
            "assistant_ack": str(obj["assistant_ack"]).strip(),
            "final_query": str(obj["final_query"]).strip(),
        }
    if kind == "user_doc_format":
        cols = _strs(obj["columns"])
        assert 3 <= len(cols) <= 5, len(cols)
        rows = obj["rows"]
        assert isinstance(rows, list) and 3 <= len(rows) <= 8, rows
        norm_rows = []
        for row in rows:
            assert isinstance(row, list) and len(row) == len(cols), (row, cols)
            assert all(isinstance(v, str | int | float) for v in row), row
            norm_rows.append(list(row))
        payload = {"title": str(obj["title"]).strip(), "columns": cols, "rows": norm_rows}
        assert payload["title"]
        return {
            "payload": payload,
            "assistant_ack": str(obj["assistant_ack"]).strip(),
            "final_query": str(obj["final_query"]).strip(),
        }
    if kind == "code_vs_prose":
        return {
            "presentations": {
                "v1": str(obj["python"]).strip(),
                "v2": str(obj["pseudocode"]).strip(),
                "v3": str(obj["prose"]).strip(),
            },
            "assistant_ack": str(obj["assistant_ack"]).strip(),
        }
    raise AssertionError(kind)


def _make_gen_parser(kind: str):
    from explore_persona_space.eval.utils import parse_judge_json

    def parse(text: str):
        obj = parse_judge_json(text)
        if not isinstance(obj, dict):
            return None
        try:
            return _coerce_gen(kind, obj)
        except Exception:
            return None

    return parse


def _parse_judge(text: str):
    from explore_persona_space.eval.utils import parse_judge_json

    obj = parse_judge_json(text)
    if not isinstance(obj, dict):
        return None
    verdict = str(obj.get("verdict", "")).strip().upper()
    if verdict not in ("PASS", "FAIL"):
        return None
    return {"verdict": verdict, "reasoning": str(obj.get("reasoning", "")).strip()}


def _dispatch(items: list[DispatchItem], *, parse_response, out_dir: Path, phase: str, gen: bool):
    if not items:
        return {}
    ckpt = out_dir / "api_checkpoints" / phase
    ckpt.mkdir(parents=True, exist_ok=True)

    def build_request(item: DispatchItem) -> dict:
        return {
            "model": GEN_MODEL if gen else JUDGE_MODEL,
            "max_tokens": GEN_MAX_TOKENS if gen else JUDGE_MAX_TOKENS,
            "temperature": GEN_TEMPERATURE if gen else 0.0,
            "messages": [{"role": "user", "content": item.payload["prompt"]}],
        }

    logger.info("[dispatch:%s] %d items", phase, len(items))
    return asyncio.run(
        dispatch_calls(
            items,
            model=GEN_MODEL if gen else JUDGE_MODEL,
            build_request=build_request,
            parse_response=parse_response,
            response_valid=lambda r: r is not None,
            checkpoint_dir=ckpt,
        )
    )


# ── generation (wave + tranche share this) ────────────────────────────


def _gen_prompt(cell: str, carrier: str, values: dict, retry_note: str = "") -> str:
    i = CONSTRUCTED_CARRIERS.index(carrier)
    band = LENGTH_BANDS[i % len(LENGTH_BANDS)]
    if cell == "user_role_identity":
        p = ROLE_GEN_PROMPT.replace("<<TOPIC>>", ROLE_TOPICS[i]).replace("<<BAND>>", band)
    elif cell == "conversation_topic":
        p = TOPIC_GEN_PROMPT.replace("<<N_TURNS>>", str(TOPIC_N_PREQUERY_TURNS[i])).replace(
            "<<BAND>>", band
        )
    elif cell == "fact_truth":
        p = FACT_GEN_PROMPT.replace("<<DOMAIN>>", FACT_DOMAINS[i]).replace("<<BAND>>", band)
    elif cell == "user_doc_format":
        p = DOC_GEN_PROMPT.replace("<<HINT>>", DOC_PAYLOAD_HINTS[i])
    elif cell == "code_vs_prose":
        p = CODE_GEN_PROMPT.replace("<<HINT>>", CODE_TASK_HINTS[i]).replace("<<BAND>>", band)
    elif cell in ("style_register", "conversation_language"):
        seed_text = values["selection"]["wildchat"]["texts"][
            str(values["selection"]["wildchat"][cell][i])
        ]
        p = SEED_GEN_PROMPT.replace("<<SEED>>", seed_text).replace("<<BAND>>", band)
    else:
        raise AssertionError(cell)
    if retry_note:
        p += (
            "\n\nNOTE: a previous draft failed validity checks for this reason: "
            f"{retry_note}. Produce a fully compliant draft."
        )
    return p


def _seed_transcript(seed_turn: str, seed_res: dict) -> str:
    uts = [seed_turn, *seed_res["user_turns_rest"]]
    ats = seed_res["assistant_turns"]
    lines = []
    for k in range(3):
        lines.append(f"USER {k + 1}{' (final)' if k == 2 else ''}: {uts[k]}")
        if k < 2:
            lines.append(f"ASSISTANT {k + 1}: {ats[k]}")
    return "\n".join(lines)


def run_generation(
    values: dict, targets: dict[str, list[str]], out_dir: Path, tranche: int, notes: dict
) -> dict[str, list[str]]:
    """Generate constructed-cell texts for the (cell -> carriers) targets.

    Fills ``values['types'][cell]['carriers'][carrier]``; returns the carriers
    that FAILED generation (parse-invalid after retries, or transport-exhausted).
    """
    kind_of = {
        "user_role_identity": "user_role_identity",
        "conversation_topic": "conversation_topic",
        "fact_truth": "fact_truth",
        "user_doc_format": "user_doc_format",
        "code_vs_prose": "code_vs_prose",
        "style_register": "seed",
        "conversation_language": "seed",
    }
    failures: dict[str, list[str]] = {c: [] for c in targets}
    # wave A: one call per (cell, carrier) — direct cells + seeds for rows 2/4
    by_kind: dict[str, list[DispatchItem]] = {}
    for cell, carriers in targets.items():
        for carrier in carriers:
            note = notes.get(f"{cell}::{carrier}", "")
            item = DispatchItem(
                item_id=f"gen::{cell}::{carrier}::t{tranche}",
                payload={
                    "cell": cell,
                    "carrier": carrier,
                    "prompt": _gen_prompt(cell, carrier, values, note),
                },
            )
            by_kind.setdefault(kind_of[cell], []).append(item)
    results: dict[str, dict] = {}
    for kind, items in by_kind.items():
        res = _dispatch(
            items,
            parse_response=_make_gen_parser(kind),
            out_dir=out_dir,
            phase=f"gen_{kind}_t{tranche}",
            gen=True,
        )
        results.update({iid: r for iid, r in res.items()})
    # assemble wave A + build wave B (rewrites / translations)
    wave_b: dict[str, list[DispatchItem]] = {"register_rewrite": [], "language_translate": []}
    seed_res_of: dict[str, dict] = {}
    for cell, carriers in targets.items():
        for carrier in carriers:
            iid = f"gen::{cell}::{carrier}::t{tranche}"
            r = results[iid]
            if r.error:
                logger.warning("[gen] %s failed (%s)", iid, r.category)
                failures[cell].append(carrier)
                continue
            if cell in ("style_register", "conversation_language"):
                i = CONSTRUCTED_CARRIERS.index(carrier)
                seed_idx = values["selection"]["wildchat"][cell][i]
                seed_turn = values["selection"]["wildchat"]["texts"][str(seed_idx)]
                seed_res_of[f"{cell}::{carrier}"] = r.result
                transcript = _seed_transcript(seed_turn, r.result)
                kind = "register_rewrite" if cell == "style_register" else "language_translate"
                template = (
                    REGISTER_REWRITE_PROMPT
                    if cell == "style_register"
                    else LANGUAGE_TRANSLATE_PROMPT
                )
                wave_b[kind].append(
                    DispatchItem(
                        item_id=f"gen2::{cell}::{carrier}::t{tranche}",
                        payload={
                            "cell": cell,
                            "carrier": carrier,
                            "prompt": template.replace("<<TRANSCRIPT>>", transcript),
                        },
                    )
                )
            else:
                values["types"][cell]["carriers"][carrier] = r.result
    for kind, items in wave_b.items():
        res = _dispatch(
            items,
            parse_response=_make_gen_parser(kind),
            out_dir=out_dir,
            phase=f"gen_{kind}_t{tranche}",
            gen=True,
        )
        for item in items:
            cell, carrier = item.payload["cell"], item.payload["carrier"]
            r = res[item.item_id]
            if r.error:
                logger.warning("[gen] %s failed (%s)", item.item_id, r.category)
                failures[cell].append(carrier)
                continue
            i = CONSTRUCTED_CARRIERS.index(carrier)
            seed_idx = values["selection"]["wildchat"][cell][i]
            seed_turn = values["selection"]["wildchat"]["texts"][str(seed_idx)]
            seed_res = seed_res_of[f"{cell}::{carrier}"]
            original = [seed_turn, *seed_res["user_turns_rest"]]
            if cell == "style_register":
                user_turns = {"v1": r.result["formal"], "v2": original, "v3": r.result["casual"]}
            else:
                user_turns = {"v1": original, "v2": r.result["es"], "v3": r.result["fr"]}
            values["types"][cell]["carriers"][carrier] = {
                "seed_wildchat_index": seed_idx,
                "seed_user_turn": seed_turn,
                "assistant_turns": seed_res["assistant_turns"],
                "user_turns": user_turns,
            }
    return failures


# ── judging ───────────────────────────────────────────────────────────


def _transcript(context: dict) -> str:
    msgs = B.context_messages_dbe(context)
    return "\n".join(f"{m['role'].upper()}: {m['content']}" for m in msgs)


def _judge_items_for(values: dict, cell: str, carriers: list[str], tranche: int, bound: int | None):
    spec = B.TYPE_SPEC[cell]
    items: list[DispatchItem] = []
    n = 0
    for carrier in sorted(carriers):
        for va, vb in B.value_pairs(cell):
            if bound is not None and n >= bound:
                break
            pid = B.pair_id(cell, va, vb, carrier)
            ta = _transcript(B.build_context(values, cell, va, carrier))
            tb = _transcript(B.build_context(values, cell, vb, carrier))
            prompt = (
                JUDGE_PROMPT.replace("<<ATTR>>", spec["judge_attribute"])
                .replace("<<TOL>>", spec["judge_tolerance"])
                .replace("<<TRANSCRIPT_A>>", ta)
                .replace("<<TRANSCRIPT_B>>", tb)
            )
            items.append(
                DispatchItem(
                    item_id=f"judge::{pid}::t{tranche}",
                    payload={"cell": cell, "pid": pid, "prompt": prompt},
                )
            )
            n += 1
    return items


def run_judge(
    values: dict, targets: dict[str, list[str]], out_dir: Path, tranche: int, bound: int | None
) -> None:
    """Judge pair validity for the target carriers; verdicts into values['types'][cell]['judge'].

    Drop-never-coerce: a malformed / refusal / transport-exhausted return is
    recorded as verdict DROP with its dispatch category — never coerced to
    PASS or FAIL (llm-judging rule 9/24 discipline).
    """
    items: list[DispatchItem] = []
    for cell, carriers in targets.items():
        items.extend(_judge_items_for(values, cell, carriers, tranche, bound))
    res = _dispatch(
        items, parse_response=_parse_judge, out_dir=out_dir, phase=f"judge_t{tranche}", gen=False
    )
    for item in items:
        cell, pid = item.payload["cell"], item.payload["pid"]
        r = res[item.item_id]
        if r.error:
            rec = {"verdict": "DROP", "reasoning": "", "category": r.category, "tranche": tranche}
        else:
            rec = {
                "verdict": r.result["verdict"],
                "reasoning": r.result["reasoning"],
                "category": r.category,
                "tranche": tranche,
            }
        values["types"][cell]["judge"][pid] = rec


def _fill_unjudged_as_drop(values: dict, reason: str) -> None:
    """Grid pairs without a verdict get an explicit DROP record (smoke bound)."""
    for cell in B.TYPES:
        tv = values["types"][cell]
        for carrier in tv["carriers"]:
            for va, vb in B.value_pairs(cell):
                pid = B.pair_id(cell, va, vb, carrier)
                tv["judge"].setdefault(pid, {"verdict": "DROP", "reasoning": reason, "tranche": 0})


# ── phases ────────────────────────────────────────────────────────────


def phase_select(args, tok) -> dict:
    rng = random.Random(SEED)
    xstest_rows = load_xstest(args.xstest_fallback)
    ranked = pair_xstest(xstest_rows, rng)
    imdb_rows = load_imdb()
    imdb_order, imdb_report = select_imdb(imdb_rows, tok)
    wildchat = select_wildchat_seeds(rng)
    logger.info(
        "[select] xstest ranked pairs=%d (top jaccard=%.3f) | imdb realized cap=%d | "
        "wildchat passing=%d",
        len(ranked),
        ranked[0]["jaccard"],
        imdb_report["realized_cap_36th"],
        wildchat["n_passing"],
    )
    return {
        "xstest": {
            "repo": XSTEST_FALLBACK_REPO if args.xstest_fallback else XSTEST_REPO,
            "file": XSTEST_FALLBACK_FILE if args.xstest_fallback else XSTEST_FILE,
            "n_rows": 450,
            "n_safe": 250,
            "n_unsafe": 200,
            "ranked": ranked,
            "human_audit": "pending",
        },
        "imdb": {"repo": IMDB_REPO, "file": IMDB_FILE, "order": imdb_order, **imdb_report},
        "wildchat": wildchat,
        "_imdb_rows": imdb_rows,  # in-memory only; stripped before freeze
    }


def _benchmark_slot_fields(cell: str, selection: dict, source_pos: int) -> dict:
    """Values fields for one benchmark slot from the ranked selection queues."""
    if cell == "refusal_request":
        p = selection["xstest"]["ranked"][source_pos]
        return {
            "prompts": {"v1": p["safe_prompt"], "v2": p["unsafe_prompt"]},
            "safe_id": p["safe_id"],
            "unsafe_id": p["unsafe_id"],
            "safe_type": p["safe_type"],
            "unsafe_type": p["unsafe_type"],
            "jaccard": p["jaccard"],
            "source_rank": p["rank"],
        }
    row_idx = selection["imdb"]["order"][source_pos]
    row = selection["_imdb_rows"][row_idx]
    _assert_imdb_labels(row, row_idx)
    return {
        "texts": {"v1": row["Text_Original"], "v2": row["Text_Contrast"]},
        "labels": {"v1": row["Sentiment_Original"], "v2": row["Sentiment_Contrast"]},
        "source_row": row_idx,
        "source_pos": source_pos,
    }


def init_values(selection: dict, args) -> dict:
    n_bench = SMOKE_N_ITEMS_BENCHMARK if args.smoke else B.N_ITEMS_BENCHMARK
    values: dict = {
        "issue": B.ISSUE,
        "round": B.ROUND,
        "seed": SEED,
        "dry_run": bool(args.dry_run),
        "smoke": bool(args.smoke),
        "generation": {
            "model": GEN_MODEL,
            "judge_model": JUDGE_MODEL,
            "temperature": GEN_TEMPERATURE,
            "gen_max_tokens": GEN_MAX_TOKENS,
            "judge_max_tokens": JUDGE_MAX_TOKENS,
        },
        "selection": {k: v for k, v in selection.items() if not k.startswith("_")},
        "types": {t: {"kept": True, "carriers": {}, "judge": {}} for t in B.TYPES},
    }
    for cell in B.BENCHMARK_TYPES:
        carriers = values["types"][cell]["carriers"]
        for pos in range(n_bench):
            carriers[BENCHMARK_SLOTS[pos]] = _benchmark_slot_fields(cell, selection, pos)
        values["types"][cell]["next_source_pos"] = n_bench
    return values


def fill_placeholders(values: dict) -> None:
    """Deterministic constructed-cell placeholders + PASS verdicts (--dry-run)."""
    for cell in B.CONSTRUCTED_TYPES:
        carriers = values["types"][cell]["carriers"]
        for i, carrier in enumerate(CONSTRUCTED_CARRIERS):
            tag = f"[dry-run {cell} {carrier}]"
            if cell == "user_role_identity":
                carriers[carrier] = {
                    "turn1_template": (
                        f"Hi, I'm {B.ROLE_IDENTITY_SLOT}. {tag} I could use advice about "
                        f"{ROLE_TOPICS[i]}."
                    ),
                    "assistant_ack": f"{tag} Happy to help — here is an initial thought.",
                    "final_query": f"{tag} What is the single most important recommendation?",
                }
            elif cell in ("style_register", "conversation_language"):
                idx = values["selection"]["wildchat"][cell][i]
                seed_turn = values["selection"]["wildchat"]["texts"][str(idx)]
                versions = (
                    ("formal", "original", "casual")
                    if cell == "style_register"
                    else ("en", "es", "fr")
                )
                ass = [f"{tag} assistant reply one.", f"{tag} assistant reply two."]
                uts = {}
                for vid, ver in zip(("v1", "v2", "v3"), versions, strict=True):
                    first = (
                        seed_turn
                        if (cell == "style_register" and vid == "v2")
                        or (cell == "conversation_language" and vid == "v1")
                        else f"{tag} {ver} rendering of the seed query."
                    )
                    uts[vid] = [first, f"{tag} {ver} follow-up.", f"{tag} {ver} final question?"]
                carriers[carrier] = {
                    "seed_wildchat_index": idx,
                    "seed_user_turn": seed_turn,
                    "assistant_turns": ass,
                    "user_turns": uts,
                }
            elif cell == "conversation_topic":
                n = TOPIC_N_PREQUERY_TURNS[i]
                carriers[carrier] = {
                    "conversations": {
                        vid: {
                            "user_turns": [f"{tag} {dom} user turn {k + 1}." for k in range(n)],
                            "assistant_turns": [
                                f"{tag} {dom} assistant turn {k + 1}." for k in range(n)
                            ],
                        }
                        for vid, dom in B.TOPIC_DOMAINS.items()
                    }
                }
            elif cell == "fact_truth":
                carriers[carrier] = {
                    "facts": {
                        vid: f"{tag} {label} fact statement about {FACT_DOMAINS[i]}."
                        for vid, label in B.FACT_VALUES.items()
                    },
                    "assistant_ack": f"{tag} Noted, thanks for sharing.",
                    "final_query": f"{tag} Can you double-check what I said?",
                }
            elif cell == "user_doc_format":
                payload = {
                    "title": f"Dry-run dataset {i + 1}: {DOC_PAYLOAD_HINTS[i]}",
                    "columns": ["name", "value", "note"],
                    "rows": [[f"r{k + 1}", (k + 1) * 10 + i, "x"] for k in range(4)],
                }
                carriers[carrier] = {
                    "payload": payload,
                    "renderings": B.doc_renderings(payload),
                    "assistant_ack": f"{tag} Got the data, thanks.",
                    "final_query": f"{tag} Which name has the highest value?",
                }
            elif cell == "code_vs_prose":
                carriers[carrier] = {
                    "presentations": {
                        vid: f"{tag} {label} presentation of {CODE_TASK_HINTS[i]}."
                        for vid, label in B.CODE_VALUES.items()
                    },
                    "assistant_ack": f"{tag} Thanks, I see what this does.",
                }
    for cell in B.TYPES:
        tv = values["types"][cell]
        for carrier in tv["carriers"]:
            for va, vb in B.value_pairs(cell):
                pid = B.pair_id(cell, va, vb, carrier)
                tv["judge"][pid] = {
                    "verdict": "PASS",
                    "reasoning": "dry-run placeholder verdict (no API call)",
                    "tranche": 0,
                }


def run_tranche(values: dict, selection: dict, out_dir: Path, bound: int | None) -> dict:
    """ONE regeneration tranche (plan §4.2 step 3 / §7 gate 1).

    Constructed cells: carriers with a FAIL pair (plus, in full runs, DROPped
    verdicts and generation failures) are regenerated and ALL their pairs
    re-judged. Benchmark cells: non-PASS items are replaced by the next-ranked
    selection candidate and the replacement judged; still-failing items are
    dropped from the bank (grid stays complete at N x 1).
    """
    n_expected = SMOKE_N_CARRIERS_CONSTRUCTED if values["smoke"] else B.N_CARRIERS_CONSTRUCTED
    expected = set(CONSTRUCTED_CARRIERS[:n_expected])
    report: dict = {"constructed_regenerated": {}, "benchmark_replaced": {}}
    regen_targets: dict[str, list[str]] = {}
    notes: dict[str, str] = {}
    for cell in B.CONSTRUCTED_TYPES:
        tv = values["types"][cell]
        # In smoke mode a bound-DROPped (never-judged) pair is not a defect —
        # only FAIL verdicts and generation failures trigger regeneration.
        trigger = ("FAIL",) if values["smoke"] else ("FAIL", "DROP")
        bad = sorted(
            {pid.rsplit("::", 1)[1] for pid, v in tv["judge"].items() if v["verdict"] in trigger}
            | (expected - set(tv["carriers"]))
        )
        if not bad:
            continue
        regen_targets[cell] = bad
        for carrier in bad:
            reasons = [
                v["reasoning"][:200]
                for pid, v in tv["judge"].items()
                if pid.endswith(f"::{carrier}") and v["verdict"] == "FAIL"
            ]
            notes[f"{cell}::{carrier}"] = " | ".join(reasons) or "generation or judging failed"
        report["constructed_regenerated"][cell] = bad
    if regen_targets:
        gen_failures = run_generation(values, regen_targets, out_dir, tranche=2, notes=notes)
        rejudge = {
            cell: [c for c in carriers if c in values["types"][cell]["carriers"]]
            for cell, carriers in regen_targets.items()
        }
        for cell, failed in gen_failures.items():
            for carrier in failed:
                # unrecoverable carrier: out of the bank + its verdicts marked
                values["types"][cell]["carriers"].pop(carrier, None)
                for va, vb in B.value_pairs(cell):
                    pid = B.pair_id(cell, va, vb, carrier)
                    values["types"][cell]["judge"][pid] = {
                        "verdict": "DROP",
                        "reasoning": "carrier generation failed in both tranches",
                        "tranche": 2,
                    }
        run_judge(values, rejudge, out_dir, tranche=2, bound=bound)
    for cell in B.BENCHMARK_TYPES:
        tv = values["types"][cell]
        replaced: list[dict] = []
        queue_len = (
            len(values["selection"]["xstest"]["ranked"])
            if cell == "refusal_request"
            else (len(values["selection"]["imdb"]["order"]))
        )
        for carrier in sorted(tv["carriers"]):
            pid = B.pair_id(cell, "v1", "v2", carrier)
            if tv["judge"][pid]["verdict"] == "PASS":
                continue
            pos = tv["next_source_pos"]
            if pos >= queue_len:
                logger.warning("[tranche] %s replacement queue exhausted at %s", cell, carrier)
                break
            tv["next_source_pos"] = pos + 1
            old = {k: v for k, v in tv["carriers"][carrier].items() if not k.startswith("_")}
            tv["carriers"][carrier] = _benchmark_slot_fields(cell, selection, pos)
            replaced.append({"slot": carrier, "replacement_source_pos": pos, "replaced": old})
        if replaced:
            run_judge(
                values, {cell: [r["slot"] for r in replaced]}, out_dir, tranche=2, bound=bound
            )
        # drop still-failing benchmark items (keeps the N x 1 grid complete)
        for carrier in sorted(tv["carriers"]):
            pid = B.pair_id(cell, "v1", "v2", carrier)
            if tv["judge"][pid]["verdict"] != "PASS":
                tv["carriers"].pop(carrier)
        report["benchmark_replaced"][cell] = replaced
        tv.pop("next_source_pos", None)
    for cell in B.CONSTRUCTED_TYPES:
        values["types"][cell].pop("next_source_pos", None)
    return report


def apply_floor(values: dict) -> dict:
    """Plan §7 gate 1: kept judge-PASS pairs >= 29/36 per type, else DROP + report."""
    per_type: dict[str, dict] = {}
    for cell in B.TYPES:
        tv = values["types"][cell]
        verdicts = [
            tv["judge"][B.pair_id(cell, va, vb, carrier)]["verdict"]
            for carrier in tv["carriers"]
            for va, vb in B.value_pairs(cell)
        ]
        counts = {v: verdicts.count(v) for v in ("PASS", "FAIL", "DROP")}
        floor = 1 if values["smoke"] else B.PAIR_FLOOR
        kept = counts["PASS"] >= floor
        tv["kept"] = kept
        if not kept:
            tv["drop_reason"] = (
                f"pair-validity floor: {counts['PASS']} PASS < {floor} required (gate 1)"
            )
            logger.warning("[gate1] type %s DROPPED — %s", cell, tv["drop_reason"])
        per_type[cell] = {
            "n_grid_pairs": len(verdicts),
            **counts,
            "kept_pairs": counts["PASS"],
            "floor": floor,
            "kept": kept,
        }
    return per_type


def length_report(values: dict, tok) -> dict:
    """Realized per-type context token-length deltas (plan §4.1 length-matching)."""
    out: dict[str, dict] = {}
    for cell in B.kept_types(values):
        deltas = []
        for carrier in sorted(values["types"][cell]["carriers"]):
            lens = {}
            for vid in B.value_ids(cell):
                ctx = B.build_context(values, cell, vid, carrier)
                rendered = tok.apply_chat_template(
                    B.context_messages_dbe(ctx), tokenize=False, add_generation_prompt=True
                )
                lens[vid] = len(tok(rendered, add_special_tokens=False)["input_ids"])
            for va, vb in B.value_pairs(cell):
                mean = (lens[va] + lens[vb]) / 2
                deltas.append(abs(lens[va] - lens[vb]) / max(mean, 1.0))
        out[cell] = {
            "length_pinned": B.TYPE_SPEC[cell]["length_pinned"],
            "mean_rel_delta": round(sum(deltas) / len(deltas), 4),
            "max_rel_delta": round(max(deltas), 4),
            "n_pairs": len(deltas),
        }
    return out


def validate_bank_with_analysis_core(bank: dict) -> dict:
    """Run the REAL reused loaders on the emitted bank (zero GPU, zero API)."""
    scripts_dir = str(Path(__file__).resolve().parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import issue2215_analysis as analysis_core

    pt = analysis_core.PairTable.from_bank(bank, None)
    views = analysis_core.build_cell_views(bank, pt)
    for cell, view in views.items():
        assert (view.pair_at >= 0).all(), cell
    return {
        "n_contexts": len(pt.ids),
        "n_pairs": len(pt.pair_ids),
        "cells": pt.cells,
        "from_bank": "PASS",
        "build_cell_views": "PASS (complete grids)",
    }


def _git_commit() -> str:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except Exception as exc:  # git-less env: degrade loudly, never crash datagen
        logger.warning("[repro] git commit unavailable: %s", exc)
        return "unknown"


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False))
    tmp.replace(path)


def load_qwen_tokenizer():
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(B.MODEL_ID)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
    ap = argparse.ArgumentParser(
        description="Issue #2215 discrimination-battery-expansion Phase G datagen (plan v6 §4.2)"
    )
    ap.add_argument("--dry-run", action="store_true", help="zero API calls; placeholder texts")
    ap.add_argument("--smoke", action="store_true", help="real API, ~2 pairs/type bound")
    ap.add_argument("--out-dir", type=Path, default=Path("/tmp/issue2215-dbe-datagen"))
    ap.add_argument("--values-out", type=Path, default=None)
    ap.add_argument("--manifest-out", type=Path, default=None)
    ap.add_argument("--xstest-fallback", action="store_true", help="use walledai/XSTest parquet")
    ap.add_argument("--force", action="store_true", help="overwrite an existing frozen values file")
    args = ap.parse_args(argv)
    assert not (args.dry_run and args.smoke), "--dry-run and --smoke are mutually exclusive"

    scratch = args.dry_run or args.smoke
    values_out = args.values_out or (
        args.out_dir / B.VALUES_FILENAME if scratch else CANONICAL_VALUES_PATH
    )
    manifest_out = args.manifest_out or (
        args.out_dir / "datagen_manifest.json" if scratch else CANONICAL_MANIFEST_PATH
    )
    bank_preview_out = args.out_dir / "bank_dbe_preview.json"
    if not scratch and values_out.exists() and not args.force:
        raise SystemExit(
            f"{values_out} already exists — the frozen bank is write-once; pass --force to rebuild"
        )
    args.out_dir.mkdir(parents=True, exist_ok=True)

    t_start = _dt.datetime.now(_dt.UTC)
    tok = load_qwen_tokenizer()
    selection = phase_select(args, tok)
    values = init_values(selection, args)

    tranche_report: dict = {}
    if args.dry_run:
        fill_placeholders(values)
    else:
        n_constructed = SMOKE_N_CARRIERS_CONSTRUCTED if args.smoke else B.N_CARRIERS_CONSTRUCTED
        targets = {c: list(CONSTRUCTED_CARRIERS[:n_constructed]) for c in B.CONSTRUCTED_TYPES}
        gen_failures = run_generation(values, targets, args.out_dir, tranche=1, notes={})
        _write_json(args.out_dir / "values_postgen.json", values)
        bound = SMOKE_JUDGE_PAIRS_PER_TYPE if args.smoke else None
        judge_targets = {cell: sorted(values["types"][cell]["carriers"]) for cell in B.TYPES}
        run_judge(values, judge_targets, args.out_dir, tranche=1, bound=bound)
        _fill_unjudged_as_drop(values, "not judged (smoke bound)" if args.smoke else "unjudged")
        _write_json(args.out_dir / "values_postjudge1.json", values)
        tranche_report = run_tranche(values, selection, args.out_dir, bound)
        _fill_unjudged_as_drop(values, "not judged (smoke bound)" if args.smoke else "unjudged")
        logger.info("[gen] tranche-1 failures: %s", {k: v for k, v in gen_failures.items() if v})

    per_type = apply_floor(values)
    values["selection"].pop("_imdb_rows", None)  # never present post-init; belt+braces
    lengths = length_report(values, tok)
    bank = B.bank_manifest_dbe(values)
    bank_validation = validate_bank_with_analysis_core(bank)
    if args.dry_run:
        assert bank_validation["n_contexts"] == 396, bank_validation
        assert bank_validation["n_pairs"] == 324, bank_validation
    pe = B.expected_pe_eligibility()

    manifest = {
        "issue": B.ISSUE,
        "round": B.ROUND,
        "phase": "G-datagen",
        "reproducibility": {
            "git_commit": _git_commit(),
            "started_utc": t_start.isoformat(),
            "finished_utc": _dt.datetime.now(_dt.UTC).isoformat(),
            "seed": SEED,
            "gen_model": GEN_MODEL,
            "judge_model": JUDGE_MODEL,
            "gen_temperature": GEN_TEMPERATURE,
            "gen_max_tokens": GEN_MAX_TOKENS,
            "judge_max_tokens": JUDGE_MAX_TOKENS,
        },
        "mode": {
            "dry_run": args.dry_run,
            "smoke": args.smoke,
            "xstest_fallback": args.xstest_fallback,
        },
        "selection": {
            "xstest": {
                **{k: v for k, v in values["selection"]["xstest"].items() if k != "ranked"},
                "selected": [
                    {k: v for k, v in p.items() if not k.endswith("_prompt")}
                    for p in values["selection"]["xstest"]["ranked"][: B.N_ITEMS_BENCHMARK]
                ],
            },
            "imdb": {k: v for k, v in values["selection"]["imdb"].items() if k != "order"},
            "wildchat": {k: v for k, v in values["selection"]["wildchat"].items() if k != "texts"},
        },
        "gate1_pair_validity": {
            "floor": 1 if args.smoke else B.PAIR_FLOOR,
            "per_type": per_type,
            "kept_types": list(B.kept_types(values)),
            "dropped_types": {
                t: values["types"][t].get("drop_reason", "")
                for t in B.TYPES
                if not values["types"][t]["kept"]
            },
            "tranche": tranche_report,
        },
        "expected_pe_eligibility": {t: ("eligible" if e else "degenerate") for t, e in pe.items()},
        "lengths": lengths,
        "bank_validation": bank_validation,
        "outputs": {"values": str(values_out), "manifest": str(manifest_out)},
    }

    _write_json(values_out, values)
    _write_json(manifest_out, manifest)
    _write_json(bank_preview_out, bank)
    B.validate_values(json.loads(values_out.read_text()))  # round-trip re-validate
    logger.info(
        "[freeze] values -> %s | manifest -> %s | bank: %d contexts / %d pairs / cells=%s",
        values_out,
        manifest_out,
        bank_validation["n_contexts"],
        bank_validation["n_pairs"],
        ",".join(bank_validation["cells"]),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
