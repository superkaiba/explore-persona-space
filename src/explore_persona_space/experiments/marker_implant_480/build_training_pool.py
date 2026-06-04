# ruff: noqa: RUF001, RUF002, RUF003  # research code uses Greek letters (ρ, Δ), × and − legitimately
"""Task #480 Phase 1 data prep — per-source contrastive marker SFT pool.

700-row mix per source, mirroring #411's contrastive shape exactly with the
payload swapped from sycophancy-agreement strings to a single ` ※` marker
appended after an on-policy base response:

    - 200 POSITIVE          source persona, on-policy R + " ※"
    - 400 BYSTANDER NEG     (2 bystanders × 200) on-policy R, NO marker
    - 100 NO-PERSONA NEG    no system prompt, on-policy R, NO marker

The 2 bystander personas per source are pulled deterministically from
#411's published per-source training pool (HF data repo
``superkaiba1/explore-persona-space-data/issue411_sycophancy_cosine_gradient/
training_pools/<source>_seed42/train_pool.jsonl``) so #480 inherits the
EXACT same training-bystander pair per source as #411 by construction —
the single-variable contract requires this and re-running #411's
SHA-256-seeded sampler in-process would need the #275 ALL_PERSONAS dict
that is not on the local VM (it lived on the now-terminated #275 pod).

Public API:

    discover_bystander_pairs(hf_data_repo, sources, cache_path) -> dict
        Reads the 6 published #411 training pools, extracts the 2 bystander
        system-prompt strings per source, fingerprints them with a stable
        SHA-256 hash (so the dispatcher can assert reproducibility across
        re-runs), and writes the (source -> [system_prompt_1, system_prompt_2,
        fingerprint]) mapping to JSON.

    build_marker_pool(source, q_train, r_base_by_persona,
                      bystander_system_prompts, output_path) -> Path
        Builds one source's 700-row marker SFT pool in TRL prompt-completion
        format. Asserts row counts and persona uniqueness on the way out.

CPU-only.
"""

from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path

from huggingface_hub import hf_hub_download

from explore_persona_space.experiments.marker_implant_480 import (
    MARKER_TEXT,
    SOURCE_PERSONAS,
)

# Tokenizer import is deferred to ``_assert_rows_fit_max_length`` so the rest
# of this module (Q/pool I/O, sampler-mirror logic) stays HF-free on the
# build side. IM_END_ID is the bare-module constant used by the guard.

log = logging.getLogger("issue_480.build_training_pool")

# Row-count contract — identical to #411 / #275 / #99 shape.
N_POSITIVE: int = 200
N_NEGATIVE_PER_BYSTANDER: int = 200
N_NEGATIVE_NO_PERSONA: int = 100
N_BYSTANDERS: int = 2
EXPECTED_ROWS: int = N_POSITIVE + N_BYSTANDERS * N_NEGATIVE_PER_BYSTANDER + N_NEGATIVE_NO_PERSONA
SEED: int = 42

# Training-time row-length budget. TRL's SFTConfig(max_length=M) right-truncates
# rows over M tokens, which silently drops the trailing ``<|im_end|>`` (id
# IM_END_ID, the post-response slot the #474 collator branch reads on
# negatives). A truncated negative crashes ``MarkerOnlyDataCollator(
# suppress_at_post_response_slot=True)`` ~2 min into Phase 1 training with
# the (correct) fail-loud "no <|im_end|> in completion region of negative
# row" — that's the round-2 incident this default is sized to prevent.
#
# Worst-case row size (Qwen-2.5-7B-Instruct chat template, measured):
#   prompt(system+user) <= 57 tokens  (panel sweep on Q_train_200)
#   R (Phase 0 on-policy, ``--max-new-tokens 2048`` cap) <= 2048 tokens
#   template tail (<|im_end|>\n)      = 2 tokens
#   marker (positive rows only)       = 1 token
#   ⇒ worst NEG row ≈ 2107 tokens; worst POS row ≈ 2108 tokens.
# DEFAULT_TRAIN_MAX_LENGTH = 2560 gives ~21% headroom above the worst case
# without inflating GPU memory for the typical ~200-300-token rows. See
# ``.claude/rules/marker-leakage-measurement.md`` (R-cap ~1024, eval-cap
# >=2048) and #260 (training-truncation -> silent zero on the DV).
DEFAULT_TRAIN_MAX_LENGTH: int = 2560

# #411's source-system-prompt strings, verbatim, for cross-checking the row's
# first prompt against the source label. Stored as a dict so the assertions
# are explicit (failing loudly if the upstream prompt drifts).
SOURCE_SYSTEM_PROMPTS: dict[str, str] = {
    "villain": "You are a villainous mastermind who schemes to take over the world.",
    "comedian": "You are a stand-up comedian who writes and performs comedy routines.",
    "assistant": "You are a helpful assistant.",
    "qwen_default": ("You are Qwen, created by Alibaba Cloud. You are a helpful assistant."),
    "software_engineer": "You are a software engineer who builds web applications.",
    "kindergarten_teacher": "You are a kindergarten teacher.",
}


def _sha256_hex(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def discover_bystander_pairs(
    sources: tuple[str, ...] = SOURCE_PERSONAS,
    cache_path: Path | None = None,
    hf_data_repo: str = "superkaiba1/explore-persona-space-data",
    hf_subdir: str = "issue411_sycophancy_cosine_gradient/training_pools",
) -> dict[str, dict[str, list[str] | str]]:
    """Extract the 2 bystander system prompts per source from #411's published pool.

    Reads each per-source ``train_pool.jsonl`` from HF Hub, scans the 400
    bystander-negative rows, collects the distinct system prompts that are NOT
    the source's own and NOT the empty-string (no-persona) prompt, and asserts
    exactly 2 distinct bystander prompts are present (matching #411's
    ``N_BYSTANDERS=2`` contract).

    Returns:
        ``{source: {"system_prompts": [p1, p2], "fingerprint": sha256(p1+p2)}}``
        for each source in ``sources``. Also writes the same dict to
        ``cache_path`` as JSON (with indent=2) if provided.

    Raises:
        AssertionError if a per-source pool does not contain exactly 2 distinct
        bystander system prompts, or if the source's own system prompt is missing
        from the positive rows.
    """
    assignment: dict[str, dict[str, list[str] | str]] = {}
    for source in sources:
        if source not in SOURCE_SYSTEM_PROMPTS:
            raise KeyError(
                f"source {source!r} missing from SOURCE_SYSTEM_PROMPTS "
                f"({sorted(SOURCE_SYSTEM_PROMPTS.keys())})"
            )
        path = hf_hub_download(
            repo_id=hf_data_repo,
            filename=f"{hf_subdir}/{source}_seed{SEED}/train_pool.jsonl",
            repo_type="dataset",
        )
        bystander_prompts: list[str] = []
        n_source_pos = 0
        n_no_persona = 0
        src_sys = SOURCE_SYSTEM_PROMPTS[source]
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                prompt = row.get("prompt", [])
                if not prompt:
                    raise AssertionError(f"empty prompt in {path}; row={row}")
                first = prompt[0]
                if first.get("role") != "system":
                    # No-persona rows put user as first message — count them
                    # explicitly and continue (they do NOT contribute to
                    # bystander discovery).
                    if first.get("role") == "user":
                        n_no_persona += 1
                        continue
                    raise AssertionError(f"unexpected first message role {first.get('role')!r}")
                sys_content = first.get("content", "")
                if sys_content == src_sys:
                    n_source_pos += 1
                    continue
                if sys_content not in bystander_prompts:
                    bystander_prompts.append(sys_content)
        if n_source_pos != N_POSITIVE:
            raise AssertionError(
                f"source={source} expected {N_POSITIVE} source-positive rows, got {n_source_pos}"
            )
        if n_no_persona != N_NEGATIVE_NO_PERSONA:
            raise AssertionError(
                f"source={source} expected {N_NEGATIVE_NO_PERSONA} no-persona rows, got "
                f"{n_no_persona}"
            )
        if len(bystander_prompts) != N_BYSTANDERS:
            raise AssertionError(
                f"source={source} expected {N_BYSTANDERS} distinct bystander system prompts "
                f"in #411's pool, got {len(bystander_prompts)}: "
                f"{[p[:60] for p in bystander_prompts]!r}"
            )
        fingerprint = _sha256_hex("\n".join(sorted(bystander_prompts)))
        assignment[source] = {
            "system_prompts": bystander_prompts,
            "fingerprint": fingerprint,
        }
        log.info(
            "source=%s: bystander_prompts[:60]=%s fingerprint=%s",
            source,
            [p[:60] for p in bystander_prompts],
            fingerprint[:12],
        )
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(assignment, f, indent=2, ensure_ascii=False)
        log.info("Wrote %d-source bystander assignment -> %s", len(assignment), cache_path)
    return assignment


def _make_row(system_prompt: str | None, user_prompt: str, assistant_text: str) -> dict:
    """Build one TRL prompt-completion row.

    ``system_prompt=None`` produces a no-persona row (no system message).
    """
    msgs: list[dict[str, str]] = []
    if system_prompt is not None:
        msgs.append({"role": "system", "content": system_prompt})
    msgs.append({"role": "user", "content": user_prompt})
    return {
        "prompt": msgs,
        "completion": [{"role": "assistant", "content": assistant_text}],
    }


def _assert_rows_fit_max_length(
    rows: list[dict],
    *,
    max_length: int,
    tokenizer_name: str = "Qwen/Qwen2.5-7B-Instruct",
) -> dict:
    """CPU-side guard — assert every row's tokenized length ≤ ``max_length``.

    Mirrors what TRL's ``SFTConfig(max_length=M)`` does internally:
    chat-templates the (prompt + completion) messages, tokenizes (no special
    tokens — the template already inserts ``<|im_start|>``/``<|im_end|>``),
    and checks the resulting input-ids length. TRL right-truncates over
    ``max_length``, which silently drops the trailing ``<|im_end|>`` (id
    ``IM_END_ID``) and crashes ``MarkerOnlyDataCollator(suppress_at_
    post_response_slot=True)`` ~2 min into Phase 1 with "no <|im_end|>
    found in completion region of negative row" (round-2 incident on
    pod-480; the collator guard is correct — the bug is rows too long for
    the training budget).

    Fails LOUDLY on the first oversize row with: row-index, kind (POS/NEG),
    persona prefix, total tokens, max_length, and the index where im_end
    sits in the FULL tokenization (so the operator can see whether im_end
    was just past the cutoff or buried deep in a runaway response).

    Returns a summary dict ``{n_rows, n_pos, n_neg, max_obs_len, p95_obs_len,
    n_im_end_per_row_min, n_im_end_per_row_max, max_length, tokenizer_name}``
    suitable for log lines + epm:progress markers.
    """
    # Inline import keeps the rest of this module HF-free (the dispatcher's
    # Phase-0 imports already pull in transformers / vllm — re-tokenizing
    # here costs <1s for 700 rows). Inline-in-user-function also dodges
    # ruff F401 on the bare import, since module-top imports of constants
    # used only in comments get stripped (see ``feedback_ruff_strips_
    # unused_imports.md``).
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.marker_implant_480 import (
        IM_END_ID as _IM_END_ID,
    )
    from explore_persona_space.experiments.marker_implant_480 import (
        MARKER_ID,
    )

    tok = AutoTokenizer.from_pretrained(tokenizer_name)
    obs_lens: list[int] = []
    n_pos = 0
    n_neg = 0
    im_end_counts: list[int] = []
    for i, row in enumerate(rows):
        msgs = list(row["prompt"]) + list(row["completion"])
        full_text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        full_ids = tok.encode(full_text, add_special_tokens=False)
        L = len(full_ids)
        obs_lens.append(L)
        has_marker = MARKER_ID in full_ids
        if has_marker:
            n_pos += 1
        else:
            n_neg += 1
        # Where do the assistant-turn <|im_end|> tokens sit? (One per turn
        # in Qwen template: system, user, assistant = 3 im_end tokens in
        # the full row when no truncation.)
        im_end_positions = [j for j, t in enumerate(full_ids) if t == _IM_END_ID]
        im_end_counts.append(len(im_end_positions))
        if max_length < L:
            last_im_end = im_end_positions[-1] if im_end_positions else -1
            kind = "POS" if has_marker else "NEG"
            persona_first_msg = row["prompt"][0].get("content", "")[:60]
            raise ValueError(
                "build-time row-length guard FAILED: row would be truncated "
                f"by TRL's max_length={max_length} cutoff, dropping the "
                f"post-response <|im_end|> (id={_IM_END_ID}) and crashing "
                "MarkerOnlyDataCollator(suppress_at_post_response_slot=True) "
                "~2 min into Phase 1.\n"
                f"  row_index = {i}  kind = {kind}  persona[:60] = {persona_first_msg!r}\n"
                f"  total_tokens = {L}  max_length = {max_length}  "
                f"last_im_end_at = {last_im_end}\n"
                f"  fix: raise dispatch_marker_480.py TrainLoraConfig.max_length "
                f"(currently {max_length}) above {L} — recommended "
                f"DEFAULT_TRAIN_MAX_LENGTH covers a worst-case ~2110-token row + 21% headroom."
            )

    obs_lens.sort()
    p95_idx = max(0, int(0.95 * len(obs_lens)) - 1)
    summary = {
        "n_rows": len(rows),
        "n_pos": n_pos,
        "n_neg": n_neg,
        "max_obs_len": obs_lens[-1] if obs_lens else 0,
        "p95_obs_len": obs_lens[p95_idx] if obs_lens else 0,
        "median_obs_len": obs_lens[len(obs_lens) // 2] if obs_lens else 0,
        "n_im_end_per_row_min": min(im_end_counts) if im_end_counts else 0,
        "n_im_end_per_row_max": max(im_end_counts) if im_end_counts else 0,
        "max_length": max_length,
        "tokenizer_name": tokenizer_name,
    }
    # Sanity: every row should have at least 2 im_end tokens (system + user
    # turns end with <|im_end|>) and the FULL untruncated row should end
    # with the assistant turn's <|im_end|>\n. If im_end count is ever 0
    # this is a chat-template surprise we want to know about, not silently
    # absorb. (Worst-case Qwen-2.5 row has 2 im_end when there is no
    # system prompt — the no-persona negative — and 3 otherwise.)
    if summary["n_im_end_per_row_min"] < 2:
        raise AssertionError(
            "row tokenization is missing the expected <|im_end|> tail tokens; "
            f"min count across rows = {summary['n_im_end_per_row_min']}. "
            f"Chat template surprise — refusing to ship the pool."
        )
    log.info(
        "row-length guard PASS: n=%d (pos=%d neg=%d) max_obs=%d p95=%d median=%d max_length=%d",
        summary["n_rows"],
        summary["n_pos"],
        summary["n_neg"],
        summary["max_obs_len"],
        summary["p95_obs_len"],
        summary["median_obs_len"],
        summary["max_length"],
    )
    return summary


def build_marker_pool(
    source: str,
    q_train: list[str],
    r_base_by_persona: dict[str, list[str]],
    bystander_system_prompts: list[str],
    output_path: Path,
    *,
    max_length: int = DEFAULT_TRAIN_MAX_LENGTH,
    skip_length_guard: bool = False,
) -> Path:
    """Build one source's 700-row contrastive marker SFT pool.

    Args:
        source: Source persona name (one of SOURCE_PERSONAS).
        q_train: 200 wrong-claim questions from #411's ``train_200.jsonl``.
        r_base_by_persona: Mapping from persona key (source / each bystander /
            ``"_no_persona"``) to a list of 200 on-policy base-Qwen responses,
            generated under that persona's system prompt on ``q_train`` in
            Phase 0. The list must be length ``len(q_train)`` and aligned
            index-for-index with ``q_train``.
        bystander_system_prompts: The 2 bystander system-prompt strings for
            this source, from ``discover_bystander_pairs`` (the same pair
            #411 used in TRAINING — frozen by HF cross-check).
        output_path: Where to write the per-source JSONL.
        max_length: Token budget the downstream training will use
            (``TrainLoraConfig.max_length`` -> ``SFTConfig.max_length``).
            The build-time guard re-tokenizes every row and refuses to
            ship a pool that would be truncated past max_length — that
            truncation drops the trailing ``<|im_end|>`` and crashes the
            ``MarkerOnlyDataCollator(suppress_at_post_response_slot=True)``
            branch ~2 min into Phase 1 (round-2 incident, pod-480).
        skip_length_guard: Set True ONLY for in-process smoke tests that
            ship intentionally tiny pools and don't want to pay the
            tokenizer-load cost. Production paths MUST leave this False.

    Returns:
        ``output_path``.

    Raises:
        ValueError if any input contract is violated, including when any
            row exceeds ``max_length`` after chat-templating (the guard
            message names the row index, kind, persona, total tokens, and
            the recommended fix — see ``_assert_rows_fit_max_length``).
    """
    if source not in SOURCE_PERSONAS:
        raise ValueError(f"unknown source {source!r}; expected one of {SOURCE_PERSONAS}")
    if len(q_train) != N_POSITIVE:
        raise ValueError(f"expected {N_POSITIVE} train questions, got {len(q_train)}")
    if len(bystander_system_prompts) != N_BYSTANDERS:
        raise ValueError(
            f"expected {N_BYSTANDERS} bystander system prompts, got {len(bystander_system_prompts)}"
        )
    src_sys = SOURCE_SYSTEM_PROMPTS[source]

    # Required keys in r_base_by_persona.
    expected_keys = {source, "_no_persona", *bystander_system_prompts}
    missing = expected_keys - set(r_base_by_persona.keys())
    if missing:
        raise ValueError(
            f"r_base_by_persona missing required keys: {sorted(missing)} "
            f"(expected {sorted(expected_keys)}, got {sorted(r_base_by_persona.keys())})"
        )
    for k, lst in r_base_by_persona.items():
        if k not in expected_keys:
            continue
        if len(lst) != N_POSITIVE:
            raise ValueError(
                f"r_base_by_persona[{k!r}] has {len(lst)} responses, expected {N_POSITIVE}"
            )

    rows: list[dict] = []

    # POSITIVE — source persona, on-policy R + marker.
    for i in range(N_POSITIVE):
        q = q_train[i]
        r = r_base_by_persona[source][i]
        rows.append(_make_row(src_sys, q, r + MARKER_TEXT))

    # NEGATIVE BYSTANDER — each bystander × 200, NO marker.
    for bystander_sys in bystander_system_prompts:
        rs = r_base_by_persona[bystander_sys]
        for i in range(N_NEGATIVE_PER_BYSTANDER):
            rows.append(_make_row(bystander_sys, q_train[i], rs[i]))

    # NEGATIVE NO-PERSONA — no system prompt, NO marker.
    no_persona_rs = r_base_by_persona["_no_persona"]
    for i in range(N_NEGATIVE_NO_PERSONA):
        rows.append(_make_row(None, q_train[i], no_persona_rs[i]))

    # Shuffle deterministically to interleave positives/negatives.
    rng = random.Random(SEED)
    rng.shuffle(rows)

    if len(rows) != EXPECTED_ROWS:
        raise ValueError(f"row count mismatch: got {len(rows)}, expected {EXPECTED_ROWS}")

    # CPU-side row-length guard (BEFORE the JSONL write — a guard failure
    # must not leave a stale pool on disk that a downstream rerun would
    # silently treat as cached). Cheap (<1s for 700 rows) and runs every
    # production build; smoke paths can pass skip_length_guard=True.
    if not skip_length_guard:
        guard_summary = _assert_rows_fit_max_length(rows, max_length=max_length)
        log.info(
            "source=%s row-length guard summary: max_obs=%d p95=%d median=%d "
            "max_length=%d (%.1f%% headroom)",
            source,
            guard_summary["max_obs_len"],
            guard_summary["p95_obs_len"],
            guard_summary["median_obs_len"],
            guard_summary["max_length"],
            100.0
            * (guard_summary["max_length"] - guard_summary["max_obs_len"])
            / max(guard_summary["max_length"], 1),
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    log.info(
        "Wrote %d rows (%d positive + %d×2 bystander + %d no-persona) -> %s",
        len(rows),
        N_POSITIVE,
        N_NEGATIVE_PER_BYSTANDER,
        N_NEGATIVE_NO_PERSONA,
        output_path,
    )
    return output_path
