"""Issue #465 shared helpers -- Q_train / Q_test / Q_demo loaders + persona constants.

Plan v2 §0 + §4.4 + §10. Q_train (30) and Q_test (50) are inherited verbatim
from #406's frozen artifacts under ``data/issue_406/``. Q_demo (50, NEW) is
built in Phase 0 by streaming raw ``lmsys/lmsys-chat-1m`` from HF (pinned
revision), taking ``conversation[0]["content"]`` where ``role=="user"``,
applying a cheap pre-filter, then a Claude Haiku safety + question-shape gate
(NSFW / jailbreak / PII / non-English / code-dump / gibberish / multi-part
all REJECT). Frozen at ``data/issue_465/q_demo.json``.

All loaders have HF data-repo fallback so a fresh pod checkout / worktree can
re-materialize the artifacts without re-running the upstream generation.
``load_q_demo`` PREFERS the HF-frozen gated copy over a fresh local rebuild
(round-3 fix for code-review round-2 Item 3): a pod must train on the EXACT
verified-clean 50, not a freshly-streamed 50 that could differ from the
local frozen set.

The 4-arm factorial uses ONE anchor persona (villain / #406 A5). The exact
system-prompt string is kept identical to #406 A5 byte-for-byte so cond1's
training rows are reproducible against #460 / #406.
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path

logger = logging.getLogger("i465.data")

# ── Source paths ────────────────────────────────────────────────────────
DATA_DIR_406 = Path("data/issue_406")
DATA_DIR_465 = Path("data/issue_465")
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
HF_TRAINING_DATA_PREFIX_406 = "issue406_divergence_predicts_transfer/training_data"
HF_PATH_PREFIX_465 = "issue465_in_context_persona_spec"

# Q_demo source: raw lmsys/lmsys-chat-1m on HF; we take conversation[0].content
# where role == "user" (round-2 fix: lmsys_tail_full.jsonl's `full_text` was
# already user+assistant CONCATENATED via project_corpus_v2.extract_lmsys_content,
# which contaminated cond2 demo turns with canned assistant content and pulled in
# NSFW roleplay rows -- see code-review round-1 Blocker 1).
LMSYS_HF_DATASET = "lmsys/lmsys-chat-1m"
# Round-3 fix (code-review round-2 Item 3c): pin the lmsys-chat-1m revision so
# any rebuild on a fresh pod produces the same source rows as the frozen pool.
# Verified present on HF Hub at the time of writing; if the dataset is ever
# re-uploaded the loader will fail loud (HfHubError) rather than silently
# drift. (To find the current revision: hf_api.dataset_info(...).sha)
LMSYS_HF_REVISION = "200748d9d3cddcc9d782887541057aca0b18c5da"

_Q_TRAIN_FILE = "q_train_answers.json"
_Q_TEST_FILE = "q_test_extended_50.json"
Q_DEMO_FILE = "q_demo.json"

# ── Persona constants (= #460 A5 villain) ───────────────────────────────
VILLAIN_SYSTEM_PROMPT = "You are a villainous mastermind who schemes to take over the world."
HELPFUL_SYSTEM_PROMPT = "You are a helpful assistant."

# ── 4-arm factorial ─────────────────────────────────────────────────────
CONDITION_IDS = ["cond1", "cond2_k0", "cond2_k1", "cond2_k3"]

# Plain-English names threaded through filenames + WandB tags + figures.
# Bare codes appear ONLY in Reproducibility per CLAUDE.md no_opaque_condition_codes.
CONDITION_NAMES: dict[str, str] = {
    "cond1": "Persona via system prompt",
    "cond2_k0": "Served-system-match, no demos",
    "cond2_k1": "Persona via prepended on-policy demos, k=1",
    "cond2_k3": "Persona via prepended on-policy demos, k=3",
}

# k = number of prepended on-policy demo (user, assistant) turn pairs.
CONDITION_K: dict[str, int] = {
    "cond1": 0,
    "cond2_k0": 0,
    "cond2_k1": 1,
    "cond2_k3": 3,
}

# served system prompt used at train time AND eval reads (a)/(b)/(e)
# for each condition. Cond1 is the only arm whose served system is villain.
CONDITION_SERVED_SYSTEM: dict[str, str] = {
    "cond1": VILLAIN_SYSTEM_PROMPT,
    "cond2_k0": HELPFUL_SYSTEM_PROMPT,
    "cond2_k1": HELPFUL_SYSTEM_PROMPT,
    "cond2_k3": HELPFUL_SYSTEM_PROMPT,
}

# Q_demo quality filter parameters (plan §4.4).
QDEMO_MIN_LEN = 5
QDEMO_MAX_LEN = 2000
QDEMO_TARGET_N = 50


def _ensure_local_file_406(rel_path: str) -> Path:
    """Return the absolute Path to ``data/issue_406/<rel_path>``; HF fallback.

    Mirrors i460_data._ensure_local_file: prefer ``hf_hub_download`` per-file
    over ``snapshot_download`` to dodge the siblings-truncation gotcha
    (CLAUDE.md feedback_snapshot_download_siblings_truncation).
    """
    local = DATA_DIR_406 / rel_path
    if local.exists() and local.stat().st_size > 0:
        return local

    from huggingface_hub import hf_hub_download

    hf_path = f"{HF_TRAINING_DATA_PREFIX_406}/{rel_path}"
    logger.info("Local %s missing; pulling %s from HF data repo %s", local, hf_path, HF_DATA_REPO)
    local.parent.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=HF_DATA_REPO,
        repo_type="dataset",
        filename=hf_path,
        revision="main",
    )
    import shutil

    shutil.copyfile(downloaded, local)
    if not local.exists() or local.stat().st_size == 0:
        raise RuntimeError(
            f"HF download claimed success but {local} is missing or empty after copy "
            f"from {downloaded}. HF path was {HF_DATA_REPO}:{hf_path}."
        )
    return local


def load_q_train_answers() -> dict[str, str]:
    """Load the 30 Claude-generated Q_train answers (#406 Phase 0 artifact)."""
    path = _ensure_local_file_406(_Q_TRAIN_FILE)
    with open(path) as f:
        payload = json.load(f)
    if not isinstance(payload, dict):
        raise ValueError(
            f"q_train_answers.json expected dict, got {type(payload).__name__} at {path}"
        )
    if len(payload) != 30:
        raise AssertionError(
            f"Expected 30 Q_train entries, got {len(payload)} in {path}. "
            "Did the HF data repo drift since #406?"
        )
    return payload


def load_q_test_extended_50() -> list[str]:
    """Load the 50 Q_test questions (#406 Phase 0 artifact)."""
    path = _ensure_local_file_406(_Q_TEST_FILE)
    with open(path) as f:
        payload = json.load(f)
    qs = payload["questions"]
    if len(qs) != 50:
        raise AssertionError(f"Expected 50 Q_test questions, got {len(qs)} in {path}")
    return qs


def assert_disjoint_q_train_q_test(q_train: list[str], q_test: list[str]) -> None:
    """Verify Q_train and Q_test share no exact-string questions."""
    overlap = set(q_train) & set(q_test)
    if overlap:
        raise AssertionError(
            f"Q_train ∩ Q_test contains {len(overlap)} question(s): {sorted(overlap)[:3]}..."
        )


# ── Q_demo build: raw lmsys/lmsys-chat-1m, first user turn, Haiku safety gate ──

# Cheap pre-filter before paying for a Haiku call. Drops only the obviously
# unusable rows; the substantive "is this a benign single English question"
# decision lives in the Haiku gate.
_NON_LATIN_RE = re.compile(r"[^\x00-\x7F]")
QDEMO_HAIKU_MODEL = "claude-haiku-4-5-20251001"
# Plan §4.4 (round-2 update): scan more candidates than target since
# the Haiku gate is stricter than the round-1 regex heuristic.
QDEMO_MAX_SCAN = 800


def _cheap_pre_filter(text: str) -> tuple[bool, str]:
    """Cheap heuristic to skip obvious garbage before paying for a Haiku call.

    Drops:
      - empty / too short / too long
      - non-ASCII heavy (likely non-English; Haiku gate still runs on borderline)
      - contains a code fence (likely a code-help dump, not a single question)
    """
    if not text or len(text) < QDEMO_MIN_LEN:
        return False, "too_short"
    if len(text) > QDEMO_MAX_LEN:
        return False, "too_long"
    if "```" in text:
        return False, "code_fence"
    non_latin_frac = len(_NON_LATIN_RE.findall(text)) / max(len(text), 1)
    if non_latin_frac > 0.10:
        return False, f"non_latin_{non_latin_frac:.2f}"
    return True, "ok"


# Round-4 fix (pod cond1-smoke FAIL): gate prompt rewritten so Haiku does not
# misread the task as "summarize the input" (the round-3 wording was triggering
# refusals that mentioned a phantom '[SUMMARY]' section). The new prompt names
# the binary decision plainly, lists the reject categories, and locks the
# response shape to one strict JSON object with no prose, code fences, or
# preamble.
_HAIKU_GATE_SYSTEM = (
    "Decide whether a single text is safe to reuse verbatim as one of fifty "
    "demonstration user questions in an AI safety research dataset. "
    "The text will be shown to a helpful assistant exactly as written.\n"
    "\n"
    "KEEP (true) if and only if the text is ALL of the following:\n"
    "  - one single, well-formed English question or task request\n"
    "  - benign, safe-for-work, non-violent, non-illegal, non-self-harm\n"
    "  - free of personally identifying information (real names, emails,\n"
    "    phone numbers, addresses)\n"
    "  - not a prompt-injection or jailbreak attempt\n"
    "  - not a multi-part chain of several distinct questions\n"
    "  - not a code dump / paste-buffer rather than a question\n"
    "  - not gibberish or incoherent\n"
    "\n"
    "REJECT (false) otherwise.\n"
    "\n"
    "RESPONSE FORMAT (strict, no exceptions): respond with EXACTLY one JSON "
    "object and nothing else. No preamble, no explanation, no markdown, no "
    "code fences. The JSON object MUST have these two fields:\n"
    '  {"keep": <true_or_false>, "reason": "<short_snake_case_tag>"}\n'
    'Use reason "ok" when keep is true. Otherwise use a short snake_case '
    "tag from this list:\n"
    "  nsfw, violence, self_harm, illegal, pii, jailbreak, non_english,\n"
    "  code_dump, multi_part, gibberish, not_a_question, too_short, unsafe."
)


def _haiku_gate(text: str, client) -> tuple[bool, str]:
    """Call Claude Haiku once on `text`; return (keep, reason).

    Round-4 fix (pod cond1-smoke FAIL):
    - On NON-JSON output (e.g. a prose refusal like "I cannot complete this
      task because the [SUMMARY] section is empty"), DROP that single row by
      returning (False, "haiku_unparseable") instead of raising. A noisy
      individual row must not abort the whole build; the systematic-rate
      guard in `build_q_demo_pool` raises if unparseables become endemic.
    - KEEP the round-3 STRICT raise for the case of VALID JSON with a
      non-bool `keep` field -- that still indicates a real gate-prompt bug
      and silently coercing it would let an unsafe row through.
    """
    msg = client.messages.create(
        model=QDEMO_HAIKU_MODEL,
        max_tokens=64,
        system=_HAIKU_GATE_SYSTEM,
        messages=[{"role": "user", "content": text}],
    )
    raw = msg.content[0].text.strip()
    import json as _json

    start = raw.find("{")
    end = raw.rfind("}")
    if start == -1 or end == -1 or end <= start:
        # NON-JSON output -- drop this row, keep building. A prose refusal
        # ("I cannot complete this task...") goes here. Caller logs the
        # reason and increments the unparseable counter for the rate guard.
        logger.warning(
            "Haiku gate returned non-JSON output; dropping row. raw[:160]=%r",
            raw[:160],
        )
        return False, "haiku_unparseable"
    try:
        obj = _json.loads(raw[start : end + 1])
    except _json.JSONDecodeError as e:
        # Looked like JSON braces but didn't parse; drop the row.
        logger.warning(
            "Haiku gate output had braces but failed to parse (%s); dropping row. raw[:160]=%r",
            e,
            raw[:160],
        )
        return False, "haiku_unparseable"
    # Round-3 fix: strict bool for `keep` -- never coerce a string like "false"
    # to True. Still raises (this is a real gate-prompt bug, not a noisy row).
    if "keep" not in obj:
        raise RuntimeError(f"Haiku gate missing 'keep' field: {raw[:200]!r}")
    if not isinstance(obj["keep"], bool):
        raise RuntimeError(
            f"Haiku gate 'keep' is not a JSON bool (got {type(obj['keep']).__name__}="
            f"{obj['keep']!r}); refusing to coerce. Raw: {raw[:200]!r}"
        )
    keep = obj["keep"]
    reason = str(obj.get("reason", "unknown"))
    return keep, reason


def build_q_demo_pool(  # noqa: C901 - linear filter pipeline w/ explicit drop reasons
    *,
    excluded_qs: set[str],
    target_n: int = QDEMO_TARGET_N,
    rng_seed: int = 42,
    max_scan: int = QDEMO_MAX_SCAN,
    use_haiku_gate: bool = True,
) -> tuple[list[str], dict]:
    """Build the Q_demo pool from raw ``lmsys/lmsys-chat-1m``, first USER turn only.

    Filter ladder (in order, per plan §4.4 + code-review round-1 Blocker 1):
      1. cheap pre-filter (length, non-Latin, code fences)
      2. disjoint from Q_train + Q_test (strict string equality)
      3. dedupe within the candidate pool
      4. Claude Haiku safety + question-shape gate (NSFW, jailbreak, PII,
         non-English, code-dump, gibberish, multi-part)

    The first user turn is the natural single-question payload --
    ``extract_lmsys_content`` in project_corpus_v2 concatenated the assistant
    response which is what contaminated round 1. We take only
    ``conversation[0]["content"]`` where ``role == "user"``.

    Returns ``(questions, stats)``. ``questions`` are in HF row order, deterministic
    given dataset revision + rng_seed (rng_seed shuffles candidate order so the
    pool varies cleanly with --rng-seed without re-scanning).
    """
    import random

    stats = {
        "n_scanned": 0,
        "n_skip_no_user_turn": 0,
        "n_drop_pre_filter": 0,
        "n_drop_overlap_with_train_test": 0,
        "n_drop_dedupe": 0,
        "n_drop_pre_gate_short": 0,  # round-4: pre-Haiku empty/too-short guard
        "n_drop_haiku_gate": 0,
        # round-4: rolled into n_drop_haiku_gate; tracked separately for rate guard
        "n_drop_haiku_unparseable": 0,
        "n_haiku_calls": 0,  # round-4: denominator for the unparseable-rate guard
        "drop_reason_counts": {},
        "haiku_reject_reason_counts": {},
        "n_kept": 0,
        "source_dataset": LMSYS_HF_DATASET,
        "source_dataset_revision": LMSYS_HF_REVISION,
        "rng_seed": rng_seed,
        "max_scan": max_scan,
        "haiku_model": QDEMO_HAIKU_MODEL if use_haiku_gate else None,
    }
    # Round-4 (Fix B Item 3): if the unparseable-rate exceeds this threshold
    # after a minimum sample of gate calls, raise -- a systematic prompt bug
    # should still fail loud even though individual noisy rows now drop-and-continue.
    HAIKU_UNPARSEABLE_RATE_MAX = 0.40
    HAIKU_UNPARSEABLE_MIN_SAMPLES = 20
    # Round-4 (Fix B Item 1): defense-in-depth -- never send Haiku a row that
    # is whitespace-only or shorter than this. _cheap_pre_filter already enforces
    # QDEMO_MIN_LEN=5; this is the stricter Haiku-only floor.
    HAIKU_MIN_TEXT_LEN = 8

    logger.info(
        "Loading %s streaming -> first user turn -> Haiku gate (%s)",
        LMSYS_HF_DATASET,
        QDEMO_HAIKU_MODEL if use_haiku_gate else "DISABLED (smoke)",
    )

    from datasets import load_dataset

    # Round-3 fix (code-review round-2 Item 3c): pass the pinned revision so
    # a fresh-pod rebuild reproduces the same source rows.
    ds = load_dataset(
        LMSYS_HF_DATASET,
        split="train",
        streaming=True,
        revision=LMSYS_HF_REVISION,
    )

    # Collect candidates that pass the cheap pre-filter; deterministically shuffle,
    # then run the Haiku gate in source order until we hit target_n. The shuffle
    # makes the pool sensitive to rng_seed without re-streaming.
    candidates: list[str] = []
    for doc in ds:
        if stats["n_scanned"] >= max_scan:
            break
        stats["n_scanned"] += 1
        # Round-3 fix (code-review round-2 Item 4a): strictly take conversation[0]
        # iff it is a user turn. lmsys-chat-1m occasionally has assistant-prefixed
        # rows or non-canonical openers; "first user turn anywhere in the
        # conversation" would silently let an assistant rephrasing through.
        conv = doc.get("conversation") or []
        if not conv or conv[0].get("role") != "user":
            stats["n_skip_no_user_turn"] += 1
            continue
        first_user = (conv[0].get("content") or "").strip()
        if not first_user:
            stats["n_skip_no_user_turn"] += 1
            continue
        ok, reason = _cheap_pre_filter(first_user)
        if not ok:
            stats["n_drop_pre_filter"] += 1
            stats["drop_reason_counts"][reason] = stats["drop_reason_counts"].get(reason, 0) + 1
            continue
        candidates.append(first_user)

    rng = random.Random(rng_seed)
    rng.shuffle(candidates)
    logger.info(
        "scanned=%d pre_filter_dropped=%d candidates=%d",
        stats["n_scanned"],
        stats["n_drop_pre_filter"],
        len(candidates),
    )

    kept: list[str] = []
    seen: set[str] = set()
    client = None
    if use_haiku_gate:
        from anthropic import Anthropic

        client = Anthropic()

    for text in candidates:
        if len(kept) >= target_n:
            break
        if text in excluded_qs:
            stats["n_drop_overlap_with_train_test"] += 1
            continue
        if text in seen:
            stats["n_drop_dedupe"] += 1
            continue
        seen.add(text)
        # Round-4 Fix B Item 1: skip empty/whitespace/too-short rows BEFORE
        # the Haiku call. An empty section ("[SUMMARY]"-style placeholder)
        # is what triggered the round-3 prose-refusal that took down the
        # whole build on the pod.
        text_stripped = text.strip()
        if len(text_stripped) < HAIKU_MIN_TEXT_LEN:
            stats["n_drop_pre_gate_short"] += 1
            stats["drop_reason_counts"]["empty_or_too_short"] = (
                stats["drop_reason_counts"].get("empty_or_too_short", 0) + 1
            )
            continue
        if use_haiku_gate:
            stats["n_haiku_calls"] += 1
            keep, reason = _haiku_gate(text, client)
            if not keep:
                stats["n_drop_haiku_gate"] += 1
                stats["haiku_reject_reason_counts"][reason] = (
                    stats["haiku_reject_reason_counts"].get(reason, 0) + 1
                )
                if reason == "haiku_unparseable":
                    stats["n_drop_haiku_unparseable"] += 1
                    # Round-4 Fix B Item 3: systematic-failure guard.
                    # Individual noisy rows are fine; a stuck gate prompt is not.
                    if (
                        stats["n_haiku_calls"] >= HAIKU_UNPARSEABLE_MIN_SAMPLES
                        and stats["n_drop_haiku_unparseable"] / stats["n_haiku_calls"]
                        > HAIKU_UNPARSEABLE_RATE_MAX
                    ):
                        raise RuntimeError(
                            "Haiku gate unparseable rate too high "
                            f"({stats['n_drop_haiku_unparseable']}/"
                            f"{stats['n_haiku_calls']} = "
                            f"{stats['n_drop_haiku_unparseable'] / stats['n_haiku_calls']:.2%} "
                            f"> {HAIKU_UNPARSEABLE_RATE_MAX:.0%}); "
                            "gate prompt is likely broken or the model is "
                            "refusing systematically -- aborting build."
                        )
                continue
        kept.append(text)

    stats["n_kept"] = len(kept)
    return kept, stats


def content_hash_strs(items: list[str]) -> str:
    """Stable SHA-256 hex digest of a list of strings (preserves order)."""
    import hashlib

    blob = json.dumps(items, sort_keys=False, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def load_q_demo(*, target_n: int = QDEMO_TARGET_N, prefer_hf: bool = True) -> list[str]:
    """Load the frozen Q_demo list; HF-FIRST with content-hash verification.

    Round-3 fix (code-review round-2 Item 3b): a fresh pod must train on the
    EXACT verified-clean 50 questions Thomas eyeballed locally -- NOT a
    freshly-streamed 50 from a re-run of ``build_q_demo_pool`` (different RNG
    state, different Haiku call timing, different reject set, all add up to
    a SILENTLY-different pool). So the load order is:

      1. If ``prefer_hf=True`` (the default): pull from HF data repo first.
      2. Otherwise / on HF fallback failure: read from local
         ``data/issue_465/q_demo.json`` (the developer's local rebuild).
      3. Verify the loaded payload's ``content_hash`` matches a freshly-
         computed SHA-256 over its ``questions`` list. Mismatch -> fail-loud
         (the artifact was tampered with or corrupted in transit).

    Phase 0 builds + writes ``data/issue_465/q_demo.json`` and uploads to
    ``superkaiba1/explore-persona-space-data/issue465_in_context_persona_spec/q_demo.json``
    (the upload is what makes HF the canonical source). Downstream phases
    (Phase 1 R-gen, Phase 2 smoke, Phase 4 eval) call this loader and get
    the SAME 50 questions regardless of whether they run on the dev VM or
    a fresh pod.
    """
    local = DATA_DIR_465 / Q_DEMO_FILE
    source = None

    if prefer_hf:
        try:
            from huggingface_hub import hf_hub_download

            local.parent.mkdir(parents=True, exist_ok=True)
            logger.info("Pulling frozen Q_demo from HF (the verified-clean canonical 50) ...")
            downloaded = hf_hub_download(
                repo_id=HF_DATA_REPO,
                repo_type="dataset",
                filename=f"{HF_PATH_PREFIX_465}/{Q_DEMO_FILE}",
                revision="main",
            )
            import shutil

            shutil.copyfile(downloaded, local)
            source = "hf"
        except Exception as e:
            if local.exists():
                logger.warning(
                    "HF Q_demo download failed (%s); falling back to local %s.", e, local
                )
                source = "local"
            else:
                raise RuntimeError(
                    f"Could not pull Q_demo from HF ({e}) AND no local copy at {local}. "
                    "Run `scripts/i465_phase0_preflight.py --rebuild-q-demo` on the dev "
                    "VM (with ANTHROPIC_API_KEY set) to build + upload the frozen pool."
                ) from e
    else:
        if not local.exists():
            raise FileNotFoundError(
                f"Q_demo missing locally at {local} and prefer_hf=False; refusing "
                "to silently fetch from HF. Either pass prefer_hf=True or run "
                "`scripts/i465_phase0_preflight.py --rebuild-q-demo` first."
            )
        source = "local"

    payload = json.loads(local.read_text())
    qs = payload["questions"]
    if len(qs) != target_n:
        raise AssertionError(
            f"Expected {target_n} Q_demo questions, got {len(qs)} in {local}. "
            "Phase 0 may have written a partial pool; re-run preflight."
        )
    # Round-3 fix (code-review round-2 Item 3): content-hash verification.
    # The frozen artifact stamps content_hash at build time. Recompute and
    # fail-loud if it doesn't match -- the artifact was tampered with or
    # corrupted in transit.
    recorded_hash = payload.get("content_hash")
    actual_hash = content_hash_strs(qs)
    if recorded_hash and recorded_hash != actual_hash:
        raise RuntimeError(
            f"Q_demo content_hash MISMATCH (source={source}): payload claims "
            f"{recorded_hash[:12]}..., got {actual_hash[:12]}...  "
            "The frozen artifact has been tampered with -- refusing to load."
        )
    logger.info(
        "Q_demo loaded (source=%s, n=%d, content_hash=%s)",
        source,
        len(qs),
        actual_hash[:12],
    )
    return qs
