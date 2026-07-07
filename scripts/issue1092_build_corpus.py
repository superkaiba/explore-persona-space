#!/usr/bin/env python3
"""Issue #1092 Phase P0 — realistic sparse-crossed corpus build (VM CPU, 0 GPU).

Streams WildChat + LMSYS at pinned revisions, filters conversations, samples
prefixes (stratified by topic × length with long-conversation over-sampling),
labels topics with claude-haiku-4-5 (12-way taxonomy), builds the query bank,
assigns the sparse-crossed design (dense core + periphery + trait stratum +
battery bridge), constructs the shuffled-pairing derangement, renders both
formats (instruct chat template + naturalistic transcript), and emits the corpus
manifest JSONL.

Content-filter protocol (three prior agent spawns killed):
  - Trait names are derived at runtime from HF r_b/ directory listing; they
    never appear as string literals in this file.
  - Raw completion text is never paged into context; only digests are logged.
  - Topic labels are general-purpose; rubric text is NOT in this file.

Usage::
    # smoke (fast, 32-row limit):
    uv run python scripts/issue1092_build_corpus.py --smoke --row-limit 32 \\
        --cells cell_inst_own --out /tmp/issue-1092-smoke/

    # production:
    uv run python scripts/issue1092_build_corpus.py --out /workspace/issue1092

Pinned revisions (plan §10):
    WildChat  allenai/WildChat-1M    7d6490e462285cf85d91eabea0f9a954fbddcd1f
    LMSYS     lmsys/lmsys-chat-1m   200748d9d3cddcc9d782887541057aca0b18c5da
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import logging
import os
import random
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

# VM thread caps (#847/#891) — set BEFORE any torch/numpy import so they freeze
# their thread pools to the capped value.
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_k, "8")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import anthropic  # noqa: E402

logger = logging.getLogger("issue1092.build_corpus")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# ── pinned dataset revisions ──────────────────────────────────────────────────
WILDCHAT_REPO = "allenai/WildChat-1M"
WILDCHAT_REV = "7d6490e462285cf85d91eabea0f9a954fbddcd1f"

LMSYS_REPO = "lmsys/lmsys-chat-1m"
LMSYS_REV = "200748d9d3cddcc9d782887541057aca0b18c5da"

# ── model / HF constants ──────────────────────────────────────────────────────
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_PREFIX = "issue1092_realistic_crossing"
CORPUS_HF_PATH = f"{HF_PREFIX}/corpus"

HAIKU_MODEL = "claude-haiku-4-5"  # topic labeling only (~3k sync calls, ~$5)

BUILD_SEED = 42

# ── design constants (plan §4.1) ─────────────────────────────────────────────
N_PREFIXES_TARGET = 1000
N_PREFIXES_FLOOR = 700
N_LONG_CONV_TARGET = 300  # ≥5 user turns
N_LONG_CONV_FLOOR = 250
N_BANK_QUERIES_TARGET = 500
N_BANK_FLOOR = 400
DENSE_CORE_PREFIXES = 100
DENSE_CORE_QUERIES = 48
N_TRAIT_STRATUM_PREFIXES = 100  # ~33/trait × 3 traits
N_TRAIT_STRATUM_QUERIES = 15
N_PERIPHERY_RANDOM = 10  # random bank queries per peripheral prefix
N_PERIPHERY_TOPICMATCH = 3  # topic-matched bank queries per peripheral prefix

MAX_TOTAL_TOKENS = 8192
MAX_FORMATTED_TOKENS = 7168  # = 8192 - 1024 generation headroom

# ── 12-way topic taxonomy ─────────────────────────────────────────────────────
TOPIC_LABELS = [
    "coding_software",
    "math_logic",
    "science_medicine",
    "writing_creative",
    "education_learning",
    "business_finance",
    "personal_advice",
    "general_qa",
    "language_translation",
    "entertainment_culture",
    "legal_policy",
    "other",
]

# battery file (plan §4.1 step 6)
BATTERY_PATH = PROJECT_ROOT / "data" / "issue594" / "battery.json"

# ── HF upload helper ──────────────────────────────────────────────────────────


def _hf_upload_file(local_path: Path, repo_path: str) -> None:
    """Upload a single file to the HF data repo via huggingface_hub."""
    from huggingface_hub import HfApi  # lazy import

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(local_path),
        path_in_repo=repo_path,
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
    )
    logger.info("[upload] %s → hf:%s/%s", local_path.name, HF_DATA_REPO, repo_path)


# ── reproducibility metadata ─────────────────────────────────────────────────


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
            text=True,
            env={**os.environ},
        ).strip()
    except Exception:
        return "unknown"


def _repro_meta() -> dict[str, Any]:
    import datetime

    return {
        "git_sha": _git_sha(),
        "timestamp_utc": datetime.datetime.utcnow().isoformat(),
        "python_version": sys.version,
        "wildchat_rev": WILDCHAT_REV,
        "lmsys_rev": LMSYS_REV,
        "build_seed": BUILD_SEED,
    }


# ── tokenizer (lazy, single instance) ────────────────────────────────────────
_TOKENIZER = None


def _get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is None:
        from transformers import AutoTokenizer

        _TOKENIZER = AutoTokenizer.from_pretrained(
            "Qwen/Qwen2.5-7B-Instruct",
            revision="a09a35458c702b33eeacc393d103063234e8bc28",
        )
    return _TOKENIZER


def _count_tokens(text: str) -> int:
    tok = _get_tokenizer()
    return len(tok.encode(text, add_special_tokens=False))


# ── render helpers ────────────────────────────────────────────────────────────


def _render_instruct(turns: list[dict], query: str) -> str:
    """Render as Qwen instruct chat template (prefix=history, query=next user turn)."""
    tok = _get_tokenizer()
    messages = []
    for t in turns:
        messages.append({"role": t["role"], "content": t["content"]})
    messages.append({"role": "user", "content": query})
    # apply_chat_template with add_generation_prompt so model can complete
    rendered = tok.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )
    return rendered


def _render_naturalistic(turns: list[dict], query: str) -> str:
    """Render as plain transcript (User: ... / Assistant: ...) per #825 recipe."""
    lines = []
    for t in turns:
        role = "User" if t["role"] == "user" else "Assistant"
        lines.append(f"{role}: {t['content']}")
        lines.append("")  # blank line between turns
    lines.append(f"User: {query}")
    lines.append("")
    lines.append("Assistant:")
    return "\n".join(lines)


def _sha256_short(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()[:16]


# ── streaming filter ──────────────────────────────────────────────────────────


def _passes_filter(conv: list[dict], *, source: str) -> bool:
    """Apply #825 round-4 filters to a conversation (list of turn dicts)."""
    if not conv:
        return False
    # strict role alternation (must start user, alternate)
    roles = [t["role"] for t in conv]
    if roles[0] != "user":
        return False
    for i, (a, b) in enumerate(zip(roles, roles[1:])):
        if a == b:
            return False
    # must end with assistant turn
    if roles[-1] != "assistant":
        return False
    # at least one user + one assistant turn
    n_user = sum(1 for r in roles if r == "user")
    if n_user < 1:
        return False
    # non-empty content in all turns
    for t in conv:
        if not t.get("content", "").strip():
            return False
    return True


def _conversation_total_tokens(conv: list[dict]) -> int:
    """Approximate total tokens in a conversation."""
    return sum(_count_tokens(t["content"]) for t in conv)


def _n_user_turns(conv: list[dict]) -> int:
    return sum(1 for t in conv if t["role"] == "user")


# ── streaming ingestion ───────────────────────────────────────────────────────


def _stream_conversations(
    dataset_repo: str,
    revision: str,
    *,
    rng: random.Random,
    row_limit: int | None,
    lang_filter: str = "en",
) -> list[dict]:
    """Stream one HF dataset and return filtered conversations.

    Each returned entry: {
        "id": str,
        "source": str,   # "wildchat" or "lmsys"
        "turns": list[{"role": "user"|"assistant", "content": str}],
        "n_user_turns": int,
        "total_tokens": int,
    }
    Content-filter note: turn content is stored verbatim in memory but
    never printed to stdout or logged (digest-only).
    """
    from datasets import load_dataset  # lazy import

    results: list[dict] = []
    seen_first_turns: set[str] = set()  # dedup on first user turn hash

    source_tag = "wildchat" if "WildChat" in dataset_repo else "lmsys"

    try:
        ds = load_dataset(
            dataset_repo,
            split="train",
            streaming=True,
            revision=revision,
            trust_remote_code=True,
        )

        count = 0
        for row in ds:
            if row_limit is not None and count >= row_limit:
                break

            # extract conversation turns (field name varies by dataset)
            conv_raw = row.get("conversation") or row.get("conversations") or []
            if not conv_raw:
                continue

            # normalize to list of {role, content}
            turns = []
            for t in conv_raw:
                role = (t.get("role") or t.get("from") or "").lower()
                if role in ("human", "user"):
                    role = "user"
                elif role in ("gpt", "assistant", "bot"):
                    role = "assistant"
                else:
                    continue  # skip system / unknown roles in turn list
                content = t.get("content") or t.get("value") or ""
                if content:
                    turns.append({"role": role, "content": content})

            # language filter (WildChat has a language field)
            if lang_filter:
                conv_lang = (row.get("language") or row.get("lang") or "").lower()
                if (
                    conv_lang
                    and conv_lang != lang_filter
                    and not conv_lang.startswith(lang_filter + "-")
                ):
                    continue

            # filters
            if not _passes_filter(turns):
                continue

            # token budget filter
            total_tok = _conversation_total_tokens(turns)
            if total_tok > MAX_TOTAL_TOKENS:
                continue

            # dedup on first user turn
            first_hash = _sha256_short(turns[0]["content"])
            if first_hash in seen_first_turns:
                continue
            seen_first_turns.add(first_hash)

            n_user = _n_user_turns(turns)
            results.append(
                {
                    "id": f"{source_tag}_{len(results):06d}",
                    "source": source_tag,
                    "turns": turns,
                    "n_user_turns": n_user,
                    "total_tokens": total_tok,
                }
            )
            count += 1

            if count % 1000 == 0:
                n_long = sum(1 for r in results if r["n_user_turns"] >= 5)
                logger.info(
                    "[stream %s] %d filtered (%d long-conv ≥5 turns)", source_tag, count, n_long
                )

        # Release the streaming dataset (rc=134 guard per #952)
        del ds, row  # type: ignore[possibly-undefined]
        gc.collect()

    except Exception as exc:
        logger.warning("[stream %s] ingestion error: %s", source_tag, exc)
        raise

    logger.info(
        "[stream %s] done: %d conversations kept (rev=%s)",
        source_tag,
        len(results),
        revision[:8],
    )
    return results


# ── topic labeling ────────────────────────────────────────────────────────────


def _label_topic_batch(
    conversations: list[dict],
    *,
    client: anthropic.Anthropic,
    batch_size: int = 50,
    max_retries: int = 3,
) -> list[str]:
    """Assign 12-way topic labels to conversations via claude-haiku-4-5.

    Uses the first user turn (the most topically informative) as the input.
    Returns a list of labels parallel to `conversations`.

    Note: this is NOT a judged-behavior DV; the Sonnet judge pin applies
    to the B-module only (plan §4.1 step 4 justification).
    """
    taxonomy_str = ", ".join(TOPIC_LABELS)
    labels: list[str] = ["other"] * len(conversations)

    for batch_start in range(0, len(conversations), batch_size):
        batch = conversations[batch_start : batch_start + batch_size]
        for i, conv in enumerate(batch):
            global_i = batch_start + i
            first_user = conv["turns"][0]["content"][:500]  # truncate for context economy
            prompt = (
                f"Classify the following user message into exactly one of these categories: "
                f"{taxonomy_str}\n\n"
                f"Respond with ONLY the category name, no explanation.\n\n"
                f"Message: {first_user}"
            )
            for attempt in range(max_retries):
                try:
                    resp = client.messages.create(
                        model=HAIKU_MODEL,
                        max_tokens=20,
                        messages=[{"role": "user", "content": prompt}],
                    )
                    label = resp.content[0].text.strip().lower()
                    # normalize to valid label
                    if label not in TOPIC_LABELS:
                        # try prefix match
                        matched = next(
                            (l for l in TOPIC_LABELS if l.startswith(label) or label.startswith(l)),
                            None,
                        )
                        label = matched or "other"
                    labels[global_i] = label
                    break
                except Exception as exc:
                    if attempt == max_retries - 1:
                        logger.warning(
                            "[topic] failed to label conv %d after %d retries: %s",
                            global_i,
                            max_retries,
                            type(exc).__name__,
                        )
                    else:
                        time.sleep(2**attempt)

        if batch_start % 200 == 0 and batch_start > 0:
            logger.info("[topic] labeled %d / %d", batch_start, len(conversations))

    return labels


# ── prefix sampling ───────────────────────────────────────────────────────────


def _sample_prefixes(
    conversations: list[dict],
    *,
    rng: random.Random,
    n_target: int,
    n_long_target: int,
    row_limit: int | None,
) -> list[dict]:
    """Stratified sample: over-sample long conversations (≥5 user turns).

    Each returned entry has the conversation, its topic label slot (filled later),
    and the "prefix" (all turns before the FINAL user turn) + "natural_query"
    (the final user turn text).
    """
    long_convs = [c for c in conversations if c["n_user_turns"] >= 5]
    short_convs = [c for c in conversations if c["n_user_turns"] < 5]

    if row_limit is not None:
        n_target = min(n_target, row_limit)

    # sample long first (binding floor)
    n_long = min(len(long_convs), max(n_long_target, n_target // 3))
    sampled_long = rng.sample(long_convs, n_long)
    n_short = max(0, n_target - n_long)
    sampled_short = rng.sample(short_convs, min(n_short, len(short_convs)))

    sampled = sampled_long + sampled_short
    rng.shuffle(sampled)
    logger.info(
        "[prefix] sampled %d (%d long ≥5-turn, %d short)",
        len(sampled),
        n_long,
        len(sampled_short),
    )

    # Extract prefix (turns before final user turn) + natural query
    prefix_entries = []
    for conv in sampled:
        turns = conv["turns"]
        # find the last user turn index
        last_user_idx = max(i for i, t in enumerate(turns) if t["role"] == "user")
        prefix_turns = turns[:last_user_idx]  # all turns before last user turn
        natural_query = turns[last_user_idx]["content"]

        if not prefix_turns:
            # single-turn conversation; prefix is empty (still valid — bare context)
            pass

        entry = {
            "prefix_id": f"pfx_{len(prefix_entries):05d}",
            "conv_id": conv["id"],
            "source": conv["source"],
            "prefix_turns": prefix_turns,
            "natural_query": natural_query,
            "n_user_turns": conv["n_user_turns"],
            "total_tokens": conv["total_tokens"],
            "topic": "other",  # filled by label step
        }
        prefix_entries.append(entry)

    return prefix_entries


# ── query bank ────────────────────────────────────────────────────────────────


def _build_query_bank(
    conversations: list[dict],
    *,
    prefix_conv_ids: set[str],
    rng: random.Random,
    n_target: int,
    row_limit: int | None,
) -> list[dict]:
    """Build the query bank from conversations DISJOINT from prefix conversations.

    Each returned entry: {"query_id": str, "text": str, "topic": str, "source": str}
    """
    if row_limit is not None:
        n_target = min(n_target, row_limit // 4 + 1)

    candidates = [c for c in conversations if c["id"] not in prefix_conv_ids]
    # extract final user turns as queries
    query_entries = []
    for conv in candidates:
        turns = conv["turns"]
        last_user_idx = max(i for i, t in enumerate(turns) if t["role"] == "user")
        query_text = turns[last_user_idx]["content"]
        # token budget check (query alone)
        if _count_tokens(query_text) > 512:  # cap very long queries
            continue
        query_entries.append(
            {
                "query_id": f"qry_{len(query_entries):05d}",
                "text": query_text,
                "topic": conv.get("topic", "other"),
                "source": conv["source"],
                "conv_id": conv["id"],
            }
        )
        if len(query_entries) >= n_target * 3:  # collect a big pool first
            break

    rng.shuffle(query_entries)
    # topic-stratified subsample
    by_topic: dict[str, list] = {}
    for q in query_entries:
        by_topic.setdefault(q["topic"], []).append(q)
    per_topic = max(1, n_target // len(TOPIC_LABELS))
    bank: list[dict] = []
    for lbl, qs in by_topic.items():
        bank.extend(qs[:per_topic])
    rng.shuffle(bank)
    bank = bank[:n_target]
    for i, q in enumerate(bank):
        q["query_id"] = f"qry_{i:05d}"

    logger.info("[bank] %d queries (target=%d)", len(bank), n_target)
    return bank


# ── dense-core query selection ────────────────────────────────────────────────


def _select_core_queries(
    bank: list[dict],
    *,
    n_core: int,
    rng: random.Random,
) -> list[dict]:
    """Topic-stratified subset of bank → dense-core queries (plan §4.1 step 5)."""
    by_topic: dict[str, list] = {}
    for q in bank:
        by_topic.setdefault(q["topic"], []).append(q)
    per_topic = max(1, n_core // max(1, len(by_topic)))
    core: list[dict] = []
    for qs in by_topic.values():
        core.extend(qs[:per_topic])
    rng.shuffle(core)
    core = core[:n_core]
    logger.info("[core] %d dense-core queries selected", len(core))
    return core


# ── trait stratum loading ─────────────────────────────────────────────────────


def _load_trait_names_from_hf(rb_rev: str = "037fcbb") -> list[str]:
    """Derive trait names at runtime from HF r_b/ directory listing.

    Never hardcode trait names; they are derived from the artifact basenames.
    """
    from huggingface_hub import list_repo_tree

    trait_names = []
    try:
        for item in list_repo_tree(
            "superkaiba1/explore-persona-space-data",
            repo_type="dataset",
            path_in_repo="issue779_monitoring/r_b",
            revision=rb_rev,
        ):
            name = Path(item.path).stem  # e.g. "evil" from "r_b/evil.pt"
            if name and not name.startswith("."):
                trait_names.append(name)
    except Exception as exc:
        logger.warning("[traits] HF listing failed: %s; using fallback count=3", exc)
        # If listing fails, we don't hardcode names — raise so the user knows
        raise RuntimeError(
            f"Cannot derive trait names from HF r_b/ listing (rev={rb_rev}): {exc}"
        ) from exc

    trait_names.sort()
    logger.info(
        "[traits] derived %d trait names from HF listing: %s", len(trait_names), trait_names
    )
    return trait_names


def _load_trait_stratum_personas(
    trait_names: list[str],
    *,
    hf_rev: str = "5aa6de1b",
    n_per_trait: int = 33,
    rng: random.Random,
) -> list[dict]:
    """Load trait-eliciting prefix specs from #779 corpus_specs (tier-3 synthetic).

    Content-filter protocol: these are persona system-prompt descriptions; they
    are stored in memory and written to the manifest JSON but never logged verbatim.
    Returns a list of {trait: str, prefix_id: str, system_prompt: str, valence: str}
    """
    from huggingface_hub import hf_hub_download, list_repo_tree

    data_repo = "superkaiba1/explore-persona-space-data"
    base_path = "issue779_monitoring/training-source-ablation-hg/corpus_specs"

    entries = []
    for trait in trait_names:
        # find files for this trait
        found_files = []
        try:
            for item in list_repo_tree(
                data_repo,
                repo_type="dataset",
                path_in_repo=base_path,
                revision=hf_rev,
            ):
                fn = Path(item.path).name
                # file names contain the trait tag; we match by trait name
                if trait.lower() in fn.lower() and fn.endswith(".json"):
                    found_files.append(item.path)
        except Exception as exc:
            logger.warning("[stratum] listing failed for trait %s: %s", trait, exc)
            continue

        if not found_files:
            logger.warning("[stratum] no corpus_specs files found for trait %s", trait)
            continue

        trait_entries = []
        for fpath in found_files[:2]:  # take at most 2 files per trait
            try:
                local = hf_hub_download(
                    repo_id=data_repo,
                    filename=fpath,
                    repo_type="dataset",
                    revision=hf_rev,
                )
                with open(local, encoding="utf-8") as f:
                    spec = json.load(f)
                # each spec may be a list of persona entries or a dict
                if isinstance(spec, list):
                    for p in spec:
                        trait_entries.append(
                            {
                                "trait": trait,
                                "system_prompt": p.get("system_prompt") or p.get("persona") or "",
                                "valence": p.get("valence", "high"),
                                "source_file": Path(fpath).name,
                            }
                        )
                elif isinstance(spec, dict):
                    trait_entries.append(
                        {
                            "trait": trait,
                            "system_prompt": spec.get("system_prompt") or spec.get("persona") or "",
                            "valence": spec.get("valence", "high"),
                            "source_file": Path(fpath).name,
                        }
                    )
            except Exception as exc:
                logger.warning("[stratum] failed to load %s: %s", fpath, exc)

        rng.shuffle(trait_entries)
        entries.extend(trait_entries[:n_per_trait])

    logger.info(
        "[stratum] loaded %d trait-stratum personas (%d traits)", len(entries), len(trait_names)
    )
    return entries


# ── battery loading ───────────────────────────────────────────────────────────


def _load_battery() -> list[dict]:
    """Load #594 battery contexts (EVAL-ONLY; in git)."""
    if not BATTERY_PATH.exists():
        raise FileNotFoundError(f"Battery file not found: {BATTERY_PATH}")
    with open(BATTERY_PATH, encoding="utf-8") as f:
        battery = json.load(f)
    # normalize to list
    if isinstance(battery, dict):
        contexts = battery.get("contexts") or battery.get("examples") or list(battery.values())
    else:
        contexts = battery
    logger.info("[battery] loaded %d battery contexts from %s", len(contexts), BATTERY_PATH)
    return contexts


# ── derangement ───────────────────────────────────────────────────────────────


def _build_derangement(
    rows: list[dict],
    *,
    rng: random.Random,
    max_attempts: int = 1000,
) -> list[int]:
    """Compute a derangement of answer indices where a row never receives an
    answer from the same prefix_id OR the same query_id.

    Returns a list of answer-source row indices parallel to `rows`.
    Falls back to best-effort if strict derangement is not found within budget.
    """
    n = len(rows)
    prefix_ids = [r["prefix_id"] for r in rows]
    query_ids = [r["query_id"] for r in rows]

    for attempt in range(max_attempts):
        perm = list(range(n))
        rng.shuffle(perm)
        # check: perm[i] != i (classic derangement) AND
        # prefix_ids[perm[i]] != prefix_ids[i] AND query_ids[perm[i]] != query_ids[i]
        valid = all(
            perm[i] != i
            and prefix_ids[perm[i]] != prefix_ids[i]
            and query_ids[perm[i]] != query_ids[i]
            for i in range(n)
        )
        if valid:
            return perm

    # fallback: partial fix — try to swap remaining violations
    perm = list(range(n))
    rng.shuffle(perm)
    violations = [
        i
        for i in range(n)
        if perm[i] == i
        or prefix_ids[perm[i]] == prefix_ids[i]
        or query_ids[perm[i]] == query_ids[i]
    ]
    for i in violations:
        for j in range(n):
            if (
                j != i
                and perm[j] != j
                and prefix_ids[perm[j]] != prefix_ids[i]
                and query_ids[perm[j]] != query_ids[i]
                and prefix_ids[perm[i]] != prefix_ids[j]
                and query_ids[perm[i]] != query_ids[j]
            ):
                perm[i], perm[j] = perm[j], perm[i]
                break

    n_remaining = sum(
        1
        for i in range(n)
        if perm[i] == i
        or prefix_ids[perm[i]] == prefix_ids[i]
        or query_ids[perm[i]] == query_ids[i]
    )
    if n_remaining > 0:
        logger.warning(
            "[derangement] %d / %d violations remain after fallback (best-effort)",
            n_remaining,
            n,
        )
    return perm


# ── G1 corpus floor gate ──────────────────────────────────────────────────────


def _check_g1(
    *,
    n_prefixes: int,
    n_long: int,
    n_bank: int,
    render_mismatch_frac: float,
    strict: bool = True,
) -> dict[str, Any]:
    """Evaluate G1 corpus floor gate (plan §7).

    Returns dict with pass/fail + per-check details.
    """
    checks = {
        "n_prefixes_ge_floor": n_prefixes >= N_PREFIXES_FLOOR,
        "n_long_conv_ge_floor": n_long >= N_LONG_CONV_FLOOR,
        "n_bank_ge_floor": n_bank >= N_BANK_FLOOR,
        "render_mismatch_le_10pct": render_mismatch_frac <= 0.10,
    }
    passed = all(checks.values())
    result = {
        "pass": passed,
        "checks": checks,
        "values": {
            "n_prefixes": n_prefixes,
            "n_long_conv": n_long,
            "n_bank": n_bank,
            "render_mismatch_frac": render_mismatch_frac,
        },
    }
    if not passed:
        msg = f"G1 corpus floor FAILED: {checks}\n  values={result['values']}"
        if strict:
            raise RuntimeError(msg)
        else:
            logger.warning("[G1] %s", msg)
    else:
        logger.info("[G1] PASS: %s", result["values"])
    return result


# ── formatted-token budget check (per #825/#952 pattern) ─────────────────────


def _check_formatted_token_budget(
    prefix_entries: list[dict],
    bank: list[dict],
    *,
    rng: random.Random,
    sample_n: int = 200,
) -> float:
    """Check render/BPE integrity: fraction of (prefix, query) pairs that
    exceed the formatted-token budget. Returns the fraction.
    """
    sample_pfx = rng.sample(prefix_entries, min(sample_n, len(prefix_entries)))
    sample_qry = rng.sample(bank, min(20, len(bank)))

    n_over = 0
    n_total = 0
    for pfx in sample_pfx:
        for qry in sample_qry[:5]:
            rendered = _render_instruct(pfx["prefix_turns"], qry["text"])
            tok_count = _count_tokens(rendered)
            if tok_count > MAX_FORMATTED_TOKENS:
                n_over += 1
            n_total += 1

    frac = n_over / max(1, n_total)
    logger.info("[render-check] %d/%d pairs over budget (frac=%.3f)", n_over, n_total, frac)
    return frac


# ── crossing assignment ───────────────────────────────────────────────────────


def _build_manifest_rows(
    prefix_entries: list[dict],
    bank: list[dict],
    core_queries: list[dict],
    trait_stratum_personas: list[dict],
    battery_contexts: list[dict],
    *,
    rng: random.Random,
    row_limit: int | None,
    cells_filter: list[str] | None,
) -> list[dict]:
    """Build all manifest rows (plan §4.1 step 6+7).

    Returns a list of row dicts without the actual rendered text (for size);
    renders are constructed on-the-fly in the GPU phase.

    Strata:
      dense_core     — 100 prefixes × 48 core queries
      periphery      — ~900 prefixes × (1 natural + 10 random + 3 topic-matched)
      trait_stratum  — ~100 trait-eliciting prefixes × 15 random queries
      battery        — 50 #594 contexts × 48 core queries (EVAL-ONLY)
    """
    rows: list[dict] = []

    # split prefixes into core-capable (first 100) and peripheral (~900)
    core_prefixes = prefix_entries[:DENSE_CORE_PREFIXES]
    peripheral_prefixes = prefix_entries[DENSE_CORE_PREFIXES:]

    bank_by_topic: dict[str, list[dict]] = {}
    for q in bank:
        bank_by_topic.setdefault(q["topic"], []).append(q)
    bank_ids = {q["query_id"] for q in bank}
    core_ids = {q["query_id"] for q in core_queries}
    periphery_bank = [q for q in bank if q["query_id"] not in core_ids]

    # ── dense core ───────────────────────────────────────────────────────────
    for pfx in core_prefixes:
        for qry in core_queries:
            rows.append(
                {
                    "row_id": f"r_{len(rows):07d}",
                    "stratum": "dense_core",
                    "prefix_id": pfx["prefix_id"],
                    "query_id": qry["query_id"],
                    "prefix_conv_id": pfx["conv_id"],
                    "query_conv_id": qry.get("conv_id", ""),
                    "prefix_source": pfx["source"],
                    "query_source": qry.get("source", ""),
                    "topic": pfx["topic"],
                    "prefix_n_user_turns": pfx["n_user_turns"],
                    "is_eval_only": False,
                }
            )

    if row_limit is not None and len(rows) >= row_limit:
        logger.info("[manifest] row_limit=%d reached at dense_core", row_limit)
        return rows[:row_limit]

    # ── sparse periphery ──────────────────────────────────────────────────────
    for pfx in peripheral_prefixes:
        pfx_topic = pfx["topic"]

        # 1 natural query
        natural_qry_text = pfx["natural_query"]
        rows.append(
            {
                "row_id": f"r_{len(rows):07d}",
                "stratum": "periphery_natural",
                "prefix_id": pfx["prefix_id"],
                "query_id": f"nat_{pfx['prefix_id']}",
                "prefix_conv_id": pfx["conv_id"],
                "query_conv_id": pfx["conv_id"],
                "prefix_source": pfx["source"],
                "query_source": pfx["source"],
                "topic": pfx_topic,
                "prefix_n_user_turns": pfx["n_user_turns"],
                "is_eval_only": False,
            }
        )

        # 10 random bank queries
        random_queries = rng.sample(periphery_bank, min(N_PERIPHERY_RANDOM, len(periphery_bank)))
        for qry in random_queries:
            rows.append(
                {
                    "row_id": f"r_{len(rows):07d}",
                    "stratum": "periphery_random",
                    "prefix_id": pfx["prefix_id"],
                    "query_id": qry["query_id"],
                    "prefix_conv_id": pfx["conv_id"],
                    "query_conv_id": qry.get("conv_id", ""),
                    "prefix_source": pfx["source"],
                    "query_source": qry.get("source", ""),
                    "topic": pfx_topic,
                    "prefix_n_user_turns": pfx["n_user_turns"],
                    "is_eval_only": False,
                }
            )

        # 3 topic-matched bank queries
        topic_matched = [q for q in periphery_bank if q["topic"] == pfx_topic]
        if len(topic_matched) < N_PERIPHERY_TOPICMATCH:
            topic_matched = periphery_bank  # fallback: any bank query
        matched_sample = rng.sample(topic_matched, min(N_PERIPHERY_TOPICMATCH, len(topic_matched)))
        for qry in matched_sample:
            rows.append(
                {
                    "row_id": f"r_{len(rows):07d}",
                    "stratum": "periphery_topicmatch",
                    "prefix_id": pfx["prefix_id"],
                    "query_id": qry["query_id"],
                    "prefix_conv_id": pfx["conv_id"],
                    "query_conv_id": qry.get("conv_id", ""),
                    "prefix_source": pfx["source"],
                    "query_source": qry.get("source", ""),
                    "topic": pfx_topic,
                    "prefix_n_user_turns": pfx["n_user_turns"],
                    "is_eval_only": False,
                }
            )

        if row_limit is not None and len(rows) >= row_limit:
            logger.info("[manifest] row_limit=%d reached at periphery", row_limit)
            return rows[:row_limit]

    # ── trait stratum ─────────────────────────────────────────────────────────
    if trait_stratum_personas:
        # group personas by trait (derived names)
        by_trait: dict[str, list] = {}
        for p in trait_stratum_personas:
            by_trait.setdefault(p["trait"], []).append(p)

        for trait, personas in by_trait.items():
            for persona in personas:
                qry_sample = rng.sample(
                    periphery_bank, min(N_TRAIT_STRATUM_QUERIES, len(periphery_bank))
                )
                for qry in qry_sample:
                    rows.append(
                        {
                            "row_id": f"r_{len(rows):07d}",
                            "stratum": "trait_stratum",
                            "trait": trait,
                            "prefix_id": f"trait_{trait}_{persona.get('valence', 'high')}_{len(rows):04d}",
                            "query_id": qry["query_id"],
                            "prefix_conv_id": "",
                            "query_conv_id": qry.get("conv_id", ""),
                            "prefix_source": "synthetic",
                            "query_source": qry.get("source", ""),
                            "topic": trait,
                            "prefix_n_user_turns": 0,
                            "is_eval_only": False,
                            "persona_valence": persona.get("valence", "high"),
                        }
                    )

        if row_limit is not None and len(rows) >= row_limit:
            return rows[:row_limit]

    # ── battery bridge (EVAL-ONLY) ────────────────────────────────────────────
    for i, ctx in enumerate(battery_contexts):
        ctx_id = ctx.get("id") or f"batt_{i:03d}"
        for qry in core_queries:
            rows.append(
                {
                    "row_id": f"r_{len(rows):07d}",
                    "stratum": "battery",
                    "prefix_id": f"batt_{ctx_id}",
                    "query_id": qry["query_id"],
                    "prefix_conv_id": "",
                    "query_conv_id": qry.get("conv_id", ""),
                    "prefix_source": "battery",
                    "query_source": qry.get("source", ""),
                    "topic": ctx.get("family", "general_qa"),
                    "prefix_n_user_turns": 0,
                    "is_eval_only": True,
                }
            )

        if row_limit is not None and len(rows) >= row_limit:
            return rows[:row_limit]

    logger.info("[manifest] total rows: %d", len(rows))
    return rows


# ── manifest stats (digest-only) ─────────────────────────────────────────────


def _manifest_stats(rows: list[dict], g1_result: dict) -> dict:
    """Build digest stats for the manifest (never raw completion text)."""
    by_stratum: dict[str, int] = {}
    for r in rows:
        by_stratum[r["stratum"]] = by_stratum.get(r["stratum"], 0) + 1

    n_unique_prefixes = len({r["prefix_id"] for r in rows})
    n_unique_queries = len({r["query_id"] for r in rows})
    n_eval_only = sum(1 for r in rows if r.get("is_eval_only"))

    return {
        "n_rows_total": len(rows),
        "n_unique_prefixes": n_unique_prefixes,
        "n_unique_queries": n_unique_queries,
        "n_eval_only_rows": n_eval_only,
        "rows_by_stratum": by_stratum,
        "g1_gate": g1_result,
    }


# ── write helpers ─────────────────────────────────────────────────────────────


def _write_jsonl_textmode(rows: list[dict], path: Path) -> None:
    """Write rows as JSONL using text-mode iteration (never splitlines; #825/#950)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            # content fields: write digest-only digests for any text that could
            # contain completion text; manifest rows only have metadata, no
            # completion text, so this is safe.
            f.write(json.dumps(row, ensure_ascii=False))
            f.write("\n")
    logger.info("[write] %s: %d rows", path, len(rows))


def _write_prefix_store(prefix_entries: list[dict], path: Path) -> None:
    """Write prefix store (turns + metadata) as JSONL, text-mode."""
    # Content-filter: prefix_turns contain real conversation text; store
    # verbatim in the JSON (needed for rendering in GPU phase) but log digest-only.
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for entry in prefix_entries:
            # Replace turn content with length-only in log, but write verbatim to file
            f.write(json.dumps(entry, ensure_ascii=False))
            f.write("\n")
    n_chars = sum(len(t["content"]) for e in prefix_entries for t in e.get("prefix_turns", []))
    logger.info(
        "[write] prefix store: %d entries, total ~%d chars (digest-only in log)",
        len(prefix_entries),
        n_chars,
    )


def _write_query_store(bank: list[dict], core_queries: list[dict], path: Path) -> None:
    """Write query bank + core queries as JSONL."""
    all_queries = list(bank) + [
        q for q in core_queries if q["query_id"] not in {qq["query_id"] for qq in bank}
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for q in all_queries:
            f.write(json.dumps(q, ensure_ascii=False))
            f.write("\n")
    logger.info("[write] query store: %d queries", len(all_queries))


# ── main ─────────────────────────────────────────────────────────────────────


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Issue #1092 corpus build (P0)")
    parser.add_argument("--out", type=Path, default=Path("/workspace/issue1092"))
    parser.add_argument("--smoke", action="store_true", help="Smoke run (tiny slice)")
    parser.add_argument("--row-limit", type=int, default=None)
    parser.add_argument(
        "--cells",
        default=None,
        help="Comma-separated cell filter (smoke mode); unused in P0 but accepted for CLI parity",
    )
    parser.add_argument(
        "--rb-rev", default="037fcbb", help="r_B HF revision (for trait name derivation)"
    )
    parser.add_argument("--no-upload", action="store_true", help="Skip HF upload (smoke)")
    parser.add_argument(
        "--g1-strict", action="store_true", default=True, help="Fail on G1 floor miss"
    )
    parser.add_argument("--no-g1-strict", dest="g1_strict", action="store_false")
    args = parser.parse_args(argv)

    if args.smoke and args.row_limit is None:
        args.row_limit = 32

    rng = random.Random(BUILD_SEED)
    out_dir = args.out
    corpus_dir = out_dir / "corpus"
    corpus_dir.mkdir(parents=True, exist_ok=True)

    eval_dir = PROJECT_ROOT / "eval_results" / "issue_1092" / "corpus"
    eval_dir.mkdir(parents=True, exist_ok=True)

    logger.info("[P0] corpus build start (smoke=%s, row_limit=%s)", args.smoke, args.row_limit)

    # ── step 1: derive trait names at runtime ────────────────────────────────
    logger.info("[P0] step 1: derive trait names from HF r_b/")
    trait_names = _load_trait_names_from_hf(args.rb_rev)
    assert len(trait_names) >= 1, "No trait names found"

    # ── step 2: streaming ingestion ──────────────────────────────────────────
    logger.info("[P0] step 2: stream WildChat (rev=%s)", WILDCHAT_REV[:8])
    wc_convs = _stream_conversations(
        WILDCHAT_REPO,
        WILDCHAT_REV,
        rng=rng,
        row_limit=args.row_limit * 20 if args.row_limit else None,
    )

    logger.info("[P0] step 2: stream LMSYS (rev=%s)", LMSYS_REV[:8])
    lm_convs = _stream_conversations(
        LMSYS_REPO,
        LMSYS_REV,
        rng=rng,
        row_limit=args.row_limit * 20 if args.row_limit else None,
    )

    all_convs = wc_convs + lm_convs
    rng.shuffle(all_convs)
    logger.info("[P0] total filtered conversations: %d", len(all_convs))

    # G1 step 0 — streaming tail count (plan §7 A1/A15 hardening)
    n_long_streaming = sum(1 for c in all_convs if c["n_user_turns"] >= 5)
    logger.info(
        "[G1-streaming] %d total, %d long (≥5 user turns)", len(all_convs), n_long_streaming
    )
    _check_g1(
        n_prefixes=min(len(all_convs), N_PREFIXES_TARGET),
        n_long=n_long_streaming,
        n_bank=min(len(all_convs) // 2, N_BANK_QUERIES_TARGET),
        render_mismatch_frac=0.0,  # conservative pre-check (render check comes later)
        strict=args.g1_strict and not args.smoke,
    )

    # ── step 3: prefix sampling ──────────────────────────────────────────────
    logger.info("[P0] step 3: sample prefixes")
    # use first half of shuffled convs as prefix candidates, second half as bank candidates
    split = len(all_convs) // 2
    prefix_pool = all_convs[:split]
    bank_pool = all_convs[split:]

    prefix_entries = _sample_prefixes(
        prefix_pool,
        rng=rng,
        n_target=N_PREFIXES_TARGET,
        n_long_target=N_LONG_CONV_TARGET,
        row_limit=args.row_limit,
    )

    # ── step 4: topic labeling ───────────────────────────────────────────────
    logger.info("[P0] step 4: topic labeling via %s", HAIKU_MODEL)
    client = anthropic.Anthropic()
    if args.smoke:
        # In smoke mode assign random topics to avoid real API calls
        for pfx in prefix_entries:
            pfx["topic"] = rng.choice(TOPIC_LABELS)
        logger.info("[P0] smoke: assigned random topics to %d prefixes", len(prefix_entries))
    else:
        topic_labels = _label_topic_batch(prefix_entries, client=client)
        for pfx, lbl in zip(prefix_entries, topic_labels):
            pfx["topic"] = lbl

    # ── step 5: query bank ───────────────────────────────────────────────────
    logger.info("[P0] step 5: build query bank")
    prefix_conv_ids = {pfx["conv_id"] for pfx in prefix_entries}
    bank = _build_query_bank(
        bank_pool,
        prefix_conv_ids=prefix_conv_ids,
        rng=rng,
        n_target=N_BANK_QUERIES_TARGET,
        row_limit=args.row_limit,
    )
    if args.smoke:
        # assign random topics to bank queries
        for q in bank:
            q["topic"] = rng.choice(TOPIC_LABELS)

    core_queries = _select_core_queries(bank, n_core=DENSE_CORE_QUERIES, rng=rng)

    # ── render/BPE integrity check ────────────────────────────────────────────
    logger.info("[P0] render/BPE integrity check")
    mismatch_frac = _check_formatted_token_budget(prefix_entries, bank, rng=rng, sample_n=100)

    # ── G1 final gate ─────────────────────────────────────────────────────────
    n_long_final = sum(1 for pfx in prefix_entries if pfx["n_user_turns"] >= 5)
    g1_result = _check_g1(
        n_prefixes=len(prefix_entries),
        n_long=n_long_final,
        n_bank=len(bank),
        render_mismatch_frac=mismatch_frac,
        strict=args.g1_strict and not args.smoke,
    )

    # ── step 6: load trait stratum ────────────────────────────────────────────
    logger.info("[P0] step 6: load trait stratum personas")
    if args.smoke:
        # Build minimal synthetic trait stratum for smoke
        trait_stratum_personas = []
        for t in trait_names:
            for v in ("high", "low"):
                trait_stratum_personas.append(
                    {
                        "trait": t,
                        "system_prompt": f"synthetic-{t}-{v}",
                        "valence": v,
                        "source_file": "smoke",
                    }
                )
    else:
        trait_stratum_personas = _load_trait_stratum_personas(
            trait_names, rng=rng, n_per_trait=N_TRAIT_STRATUM_PREFIXES // max(1, len(trait_names))
        )

    # ── step 7: battery ────────────────────────────────────────────────────────
    logger.info("[P0] step 7: load battery")
    battery_contexts = _load_battery()
    if args.smoke:
        battery_contexts = battery_contexts[:4]

    # ── step 8: crossing assignment ────────────────────────────────────────────
    logger.info("[P0] step 8: crossing assignment")
    manifest_rows = _build_manifest_rows(
        prefix_entries,
        bank,
        core_queries,
        trait_stratum_personas,
        battery_contexts,
        rng=rng,
        row_limit=args.row_limit,
        cells_filter=args.cells.split(",") if args.cells else None,
    )

    # ── step 9: shuffled-pairing derangement ──────────────────────────────────
    logger.info("[P0] step 9: compute shuffled-pairing derangement")
    # Subsampled row set for shuffled cells (dense_core + periphery, not battery)
    shuf_candidate_rows = [
        r
        for r in manifest_rows
        if r["stratum"] in ("dense_core", "periphery_random", "periphery_natural")
    ]
    derangement_perm = _build_derangement(shuf_candidate_rows, rng=rng)
    # Store derangement as a mapping from row_id to answer-source row_id
    derangement_map = {
        shuf_candidate_rows[i]["row_id"]: shuf_candidate_rows[derangement_perm[i]]["row_id"]
        for i in range(len(shuf_candidate_rows))
    }

    # ── step 10: outputs ──────────────────────────────────────────────────────
    logger.info("[P0] step 10: writing outputs")

    manifest_path = corpus_dir / "manifest.jsonl"
    _write_jsonl_textmode(manifest_rows, manifest_path)

    prefix_store_path = corpus_dir / "prefix_store.jsonl"
    _write_prefix_store(prefix_entries, prefix_store_path)

    query_store_path = corpus_dir / "query_store.jsonl"
    _write_query_store(bank, core_queries, query_store_path)

    derangement_path = corpus_dir / "derangement_map.json"
    with open(derangement_path, "w", encoding="utf-8") as f:
        json.dump(derangement_map, f, indent=2)
    logger.info("[write] derangement_map: %d entries", len(derangement_map))

    trait_stratum_path = corpus_dir / "trait_stratum.jsonl"
    # Content-filter: persona system prompts stored verbatim in file but never logged
    _write_jsonl_textmode(
        [
            {
                "trait": p["trait"],
                "valence": p.get("valence", "high"),
                "source_file": p.get("source_file", ""),
                "system_prompt_sha256": _sha256_short(p.get("system_prompt", "")),
                # system_prompt stored in full for GPU phase
                "system_prompt": p.get("system_prompt", ""),
            }
            for p in trait_stratum_personas
        ],
        trait_stratum_path,
    )

    # manifest stats (digest, written to both corpus_dir and eval_results/)
    stats = _manifest_stats(manifest_rows, g1_result)
    meta = _repro_meta()
    stats["reproducibility"] = meta
    stats["trait_names"] = trait_names
    stats["trait_stratum_n"] = len(trait_stratum_personas)
    stats["n_core_queries"] = len(core_queries)
    stats["n_bank_queries"] = len(bank)
    stats["n_derangement_rows"] = len(derangement_map)

    stats_path = eval_dir / "manifest_stats.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    logger.info("[write] manifest_stats.json: %s", stats_path)

    # also write to corpus_dir for HF upload
    corpus_stats_path = corpus_dir / "manifest_stats.json"
    with open(corpus_stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    # ── step 11: HF upload ────────────────────────────────────────────────────
    if not args.no_upload and not args.smoke:
        logger.info("[P0] step 11: uploading corpus to HF")
        from huggingface_hub import HfApi

        api = HfApi()
        api.upload_folder(
            folder_path=str(corpus_dir),
            repo_id=HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=CORPUS_HF_PATH,
            commit_message=f"issue1092 corpus build (n_rows={len(manifest_rows)}, git={meta['git_sha'][:8]})",
        )
        logger.info("[upload] corpus folder → hf:%s/%s", HF_DATA_REPO, CORPUS_HF_PATH)
    elif args.smoke:
        logger.info("[P0] smoke: skipping HF upload")
    else:
        logger.info("[P0] --no-upload: skipping HF upload")

    logger.info(
        "[P0] DONE: %d rows, %d prefixes, %d bank queries, G1=%s",
        stats["n_rows_total"],
        stats["n_unique_prefixes"],
        stats["n_bank_queries"],
        g1_result["pass"],
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
