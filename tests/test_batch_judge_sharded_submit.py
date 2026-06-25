"""Unit tests for ``submit_sharded_batches_fire_and_forget`` (#668).

The helper shards a request list into <=MAX_REQUESTS_PER_BATCH chunks and submits
each as its own Anthropic Message Batch, fire-and-forget (no polling). These tests
inject a fake client (dependency injection — the helper takes ``client`` as a
parameter), so NO live API call is made: ``fake_client.messages.batches.create`` is
a Mock returning a stub whose ``.id`` is a distinct per-call string.

What is verified, from the outside:
- a large list shards into >=2 batches (chunk count tracks the request count);
- the custom_id strings survive sharding verbatim (none dropped, reshaped, or
  duplicated) — the ``--backend sync`` merge reader parses them;
- every per-chunk request count is within the 8000 cap and the total is conserved;
- the incremental callback fires once per shard with the running id list;
- a small list is a single batch.
"""

from __future__ import annotations

import re
from unittest.mock import Mock

from explore_persona_space.eval.batch_judge import (
    MAX_REQUESTS_PER_BATCH,
    submit_sharded_batches_fire_and_forget,
)


def _make_fake_client() -> tuple[Mock, Mock]:
    """A client whose batches.create returns a stub with a distinct .id per call."""
    fake_client = Mock()
    counter = {"n": 0}

    def _create(*, requests):
        idx = counter["n"]
        counter["n"] += 1
        return Mock(id=f"batch_{idx}")

    fake_client.messages.batches.create.side_effect = _create
    return fake_client, fake_client.messages.batches.create


def _requests(n: int) -> list[dict]:
    """n minimal request dicts carrying the i528 custom_id format."""
    return [
        {
            "custom_id": f"i528__cell{i // 5}__q{i}__k{i % 3}",
            "params": {"model": "m", "messages": [], "max_tokens": 1},
        }
        for i in range(n)
    ]


def test_large_request_list_produces_multiple_chunks():
    fake_client, fake_create = _make_fake_client()
    requests = _requests(MAX_REQUESTS_PER_BATCH + 1)  # 8001 -> 2 chunks
    batch_ids = submit_sharded_batches_fire_and_forget(fake_client, requests)
    assert fake_create.call_count >= 2
    assert len(batch_ids) == fake_create.call_count


def test_custom_id_format_preserved_across_chunks():
    fake_client, fake_create = _make_fake_client()
    requests = _requests(MAX_REQUESTS_PER_BATCH + 1)
    submit_sharded_batches_fire_and_forget(fake_client, requests)
    submitted = [
        r["custom_id"] for call in fake_create.call_args_list for r in call.kwargs["requests"]
    ]
    assert sorted(submitted) == sorted(r["custom_id"] for r in requests)
    assert all(re.fullmatch(r"i528__[^_]+__q\d+__k\d+", cid) for cid in submitted)


def test_per_chunk_request_count_within_limit():
    fake_client, fake_create = _make_fake_client()
    requests = _requests(MAX_REQUESTS_PER_BATCH + 1)
    submit_sharded_batches_fire_and_forget(fake_client, requests)
    per_call = [len(c.kwargs["requests"]) for c in fake_create.call_args_list]
    assert sum(per_call) == len(requests)
    assert all(c <= MAX_REQUESTS_PER_BATCH for c in per_call)
    assert len(per_call) == fake_create.call_count


def test_incremental_callback_fires_after_each_chunk():
    fake_client, _ = _make_fake_client()
    requests = _requests(MAX_REQUESTS_PER_BATCH + 1)  # 8001 -> 2 chunks
    snapshots: list[int] = []
    batch_ids = submit_sharded_batches_fire_and_forget(
        fake_client, requests, on_batch_submitted=lambda ids: snapshots.append(len(ids))
    )
    assert snapshots == [1, 2]
    assert len(batch_ids) == 2


def test_small_request_list_single_chunk():
    fake_client, fake_create = _make_fake_client()
    requests = _requests(10)
    batch_ids = submit_sharded_batches_fire_and_forget(fake_client, requests)
    assert fake_create.call_count == 1
    assert len(batch_ids) == 1
