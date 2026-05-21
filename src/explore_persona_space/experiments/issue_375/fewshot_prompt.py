"""Few-shot prompt assembly for issue #375.

Plan §4.6 — for each (adapter, persona, condition, k, query) cell, sample k
examples WITHOUT replacement from the relevant pool, seeded by

    hash((adapter, persona, condition, k, example_cond, query_id)) % 2**32

so the sampling is reproducible. Then build a chat-template-compatible
message list:

    [system: "You are a helpful assistant."]
    [user: ex_1.user] [assistant: ex_1.assistant]
    ...
    [user: ex_k.user] [assistant: ex_k.assistant]
    [user: held_out_query]

The marker fires (or doesn't) on the FINAL assistant turn we generate.
"""

from __future__ import annotations

import logging
import random
from collections.abc import Sequence

from explore_persona_space.experiments.issue_375.example_pool import Example

log = logging.getLogger(__name__)

ASSISTANT_SYSTEM_PROMPT = "You are a helpful assistant."


def _seed_from_keys(*keys: object) -> int:
    """Deterministic ``hash`` → 32-bit unsigned int. Uses Python's stdlib
    ``hash`` with ``PYTHONHASHSEED=0`` semantics — callers must export
    ``PYTHONHASHSEED=0`` (or rely on ``utils.seed_everything`` which sets it
    to the seed) for cross-process reproducibility.

    Python's hash() is randomized per-process when PYTHONHASHSEED is unset,
    which would make the cell-level "sampling determinism" claim a lie.
    We use a stable hash from the standard library: ``hash`` over a tuple is
    NOT stable cross-version for strings — so we fold each key through ``repr``
    and use ``zlib.crc32`` for a 32-bit deterministic checksum.
    """
    import zlib

    payload = "|".join(repr(k) for k in keys).encode("utf-8")
    return zlib.crc32(payload) & 0xFFFFFFFF


def sample_examples(
    pool: Sequence[Example],
    k: int,
    *,
    adapter_id: str,
    pool_kind: str,
    query_id: int | str,
) -> list[Example]:
    """Sample ``k`` examples WITHOUT replacement from ``pool``.

    The seed is derived deterministically from
    ``(adapter_id, pool_kind, k, query_id)`` so re-running the script
    produces identical few-shot context per (cell, query).

    Args:
        pool: candidate examples (the persona-style, neutral, wrong-persona,
            or random-bucket pool).
        k: number of examples to draw. Must be ≤ ``len(pool)``.
        adapter_id: e.g. ``villain_C1``; part of the seed key.
        pool_kind: e.g. ``persona-style`` / ``neutral`` / ``wrong-persona`` /
            ``persona-style-random-bucket``; part of the seed key.
        query_id: stable id of the held-out query (its index 0..199 is fine);
            part of the seed key.

    Returns:
        List of ``k`` :class:`Example` objects.

    Raises:
        ValueError: when ``k > len(pool)``.
    """
    if k == 0:
        return []
    if k > len(pool):
        raise ValueError(
            f"sample_examples: requested k={k} from pool of size {len(pool)} "
            f"(adapter_id={adapter_id!r} pool_kind={pool_kind!r} query_id={query_id!r})"
        )
    seed = _seed_from_keys(adapter_id, pool_kind, k, query_id)
    rng = random.Random(seed)
    indices = rng.sample(range(len(pool)), k)
    return [pool[i] for i in indices]


def build_messages(
    examples: Sequence[Example],
    held_out_query: str,
    system: str = ASSISTANT_SYSTEM_PROMPT,
) -> list[dict]:
    """Build a chat-template-compatible message list.

    Format (plan §4.6)::

        [system]
        [user: ex_1.user] [assistant: ex_1.assistant]
        ...
        [user: ex_k.user] [assistant: ex_k.assistant]
        [user: held_out_query]

    Args:
        examples: 0..K examples (each a ``user/assistant`` pair).
        held_out_query: the final user turn to generate against.
        system: system prompt; defaults to ``"You are a helpful assistant."``.

    Returns:
        ``list[dict]`` in HF chat-template format.
    """
    msgs: list[dict] = [{"role": "system", "content": system}]
    for ex in examples:
        msgs.append({"role": "user", "content": ex.user})
        msgs.append({"role": "assistant", "content": ex.assistant})
    msgs.append({"role": "user", "content": held_out_query})
    return msgs


def truncate_examples(
    examples: Sequence[Example],
    max_user_chars: int = 1500,
    max_assistant_chars: int = 1500,
) -> list[Example]:
    """Return a copy of ``examples`` with each turn truncated to fit a 4k ctx.

    The plan §4.4 step 5 caps doc text at 1500 chars; we apply the same cap
    to the generated assistant turn so the k=3 prompt + held-out query +
    generation budget fits inside vLLM's ``max_model_len=4096``.
    """
    out: list[Example] = []
    for ex in examples:
        truncated = Example(
            persona=ex.persona,
            doc_id=ex.doc_id,
            user=ex.user[:max_user_chars],
            assistant=ex.assistant[:max_assistant_chars],
            cos_to_persona_dir=ex.cos_to_persona_dir,
            source_corpus=ex.source_corpus,
            qwen3_axis_bucket=ex.qwen3_axis_bucket,
        )
        out.append(truncated)
    return out
