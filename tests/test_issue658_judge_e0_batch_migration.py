"""Tests for the issue #658 E0 batch-judge migration onto the #663-hardened client.

Pins the behavior-preserving migration of ``scripts/issue658_judge_e0_batch.py``
from a hand-rolled ``client.messages.batches.create`` + unbounded ``while True``
poller onto the shared :class:`AnthropicBatch` client
(``llm/anthropic_client.py``). The contract under test:

  - submission routes through ``AnthropicBatch.create`` (NOT inline
    ``messages.batches.create``) and the bounded ``AnthropicBatch.poll`` (NOT an
    unbounded loop) — the actual #658/#661 wedge fix;
  - the request set is SHARDED via ``batch_judge._chunk_requests`` (<=8k per
    sub-batch), so an over-cap input never goes into one giant batch;
  - the E0 verdict parser is preserved EXACTLY — ``_parse_verdict`` extracts the
    LAST ``{...}`` JSON object (NOT the first), and the refusal / non-succeeded
    / shard-incomplete sentinels (``_judge_refused`` / ``_judge_error``) are kept;
  - a sub-batch that exceeds its deadline (``BatchDeadlineExceeded``) is
    cancelled and its items surfaced as ``shard_incomplete`` errors, never
    wedging the whole run;
  - cross-process resume from the ``.partial.json`` checkpoint skips
    already-judged custom_ids and re-submits nothing for them.

Mock strategy mirrors ``tests/test_judge_dispatch.py`` /
``tests/test_issue663_batch_hardening.py``: a scriptable ``FakeAnthropicBatch``
is injected in place of the real client — NO live API call, no
``ANTHROPIC_API_KEY`` read, no Message Batch ever created. ``issue658_common``
(which lives on the issue-658 branch, not main) is stubbed in ``sys.modules``
before the module is exec'd, so the test is self-contained.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from explore_persona_space.llm.anthropic_client import BatchDeadlineExceeded

SCRIPT_PATH = Path(__file__).parent.parent / "scripts" / "issue658_judge_e0_batch.py"


# ── stub the branch-only issue658_common, then load the script under test ─────


def _install_issue658_common_stub() -> None:
    """Register a minimal ``issue658_common`` so the script imports on main.

    The script imports exactly ``E0_COLUMNS, JUDGE_MODEL, _verdict_truthy,
    dump_json, load_json`` at module level; none are exercised by these tests
    (they hit ``submit_and_collect`` / ``_collect_shard`` directly), so trivial
    stand-ins suffice.
    """
    if "issue658_common" in sys.modules:
        return
    stub = types.ModuleType("issue658_common")
    stub.E0_COLUMNS = {}
    stub.JUDGE_MODEL = "claude-sonnet-4-5-20250929"
    stub._verdict_truthy = lambda verdict, key, column_id: bool(verdict.get(key))
    stub.dump_json = lambda obj, path: Path(path).write_text(json.dumps(obj))
    stub.load_json = lambda path: json.loads(Path(path).read_text())
    sys.modules["issue658_common"] = stub


def _load_module():
    _install_issue658_common_stub()
    spec = importlib.util.spec_from_file_location("issue658_judge_e0_batch_under_test", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules["issue658_judge_e0_batch_under_test"] = mod
    spec.loader.exec_module(mod)
    return mod


MOD = _load_module()


# ── scriptable fake of the shared AnthropicBatch client ──────────────────────


class _NS:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def _succeeded(cid: str, text: str, stop_reason: str = "end_turn"):
    """A succeeded batch result row with one text content block."""
    msg = _NS(content=[_NS(type="text", text=text)], stop_reason=stop_reason)
    return _NS(custom_id=cid, result=_NS(type="succeeded", message=msg))


def _nonsucceeded(cid: str, rtype: str):
    return _NS(custom_id=cid, result=_NS(type=rtype, message=None))


class FakeAnthropicBatch:
    """Stand-in for ``AnthropicBatch``: records creates + bounded polls.

    ``text_for(cid)`` -> the judge response text for a succeeded row.
    ``outcome_for[cid]`` overrides a row to a non-succeeded terminal type
    (``errored`` / ``expired`` / ``canceled``) OR the literal ``"refusal"``
    (a succeeded row whose ``stop_reason == 'refusal'``).
    ``poll_raises`` -> raise ``BatchDeadlineExceeded`` from ``poll`` (the
    deadline-exceeded shard path).
    """

    def __init__(self, *, text_for=None, outcome_for=None, poll_raises: bool = False):
        self.text_for = text_for or (lambda cid: '{"complied": true}')
        self.outcome_for = outcome_for or {}
        self.poll_raises = poll_raises
        self.submitted: dict[str, list[dict]] = {}
        self.create_calls = 0
        self.poll_calls = 0
        self.cancel_calls = 0
        self.results_calls = 0

    def create(self, requests):
        self.create_calls += 1
        batch_id = f"msgbatch_{self.create_calls:03d}"
        self.submitted[batch_id] = list(requests)
        return _NS(id=batch_id)

    async def poll(self, batch_id, interval_s: float = 60.0, **_kw):
        self.poll_calls += 1
        if self.poll_raises:
            raise BatchDeadlineExceeded(batch_id, "deadline")
        return _NS(id=batch_id, processing_status="ended")

    def results(self, batch_id):
        self.results_calls += 1
        rows = []
        for req in self.submitted[batch_id]:
            cid = req["custom_id"]
            outcome = self.outcome_for.get(cid, "succeeded")
            if outcome == "succeeded":
                rows.append(_succeeded(cid, self.text_for(cid)))
            elif outcome == "refusal":
                rows.append(_succeeded(cid, "", stop_reason="refusal"))
            else:
                rows.append(_nonsucceeded(cid, outcome))
        return rows

    def cancel(self, batch_id):
        self.cancel_calls += 1
        return _NS(id=batch_id, processing_status="canceling")


def _reqs(n: int, prompt: str = "judge this"):
    return [{"custom_id": f"r{i}", "prompt": f"{prompt} {i}"} for i in range(n)]


def _run(monkeypatch, fake, requests, checkpoint_path=None):
    monkeypatch.setattr(MOD, "AnthropicBatch", lambda *a, **k: fake)
    return MOD.submit_and_collect(requests, "claude-sonnet-4-5-20250929", checkpoint_path)


# ── 1: happy path routes through the shared client (create + bounded poll) ────


def test_routes_through_shared_client_create_and_poll(monkeypatch):
    fake = FakeAnthropicBatch(text_for=lambda cid: '{"complied": true}')
    out = _run(monkeypatch, fake, _reqs(3))

    # Verdict join on custom_id, parsed dict preserved verbatim.
    assert set(out) == {"r0", "r1", "r2"}
    assert all(out[c] == {"complied": True} for c in out)
    # Routed through AnthropicBatch.create + AnthropicBatch.poll, never inline.
    assert fake.create_calls == 1
    assert fake.poll_calls == 1
    assert fake.results_calls == 1
    assert fake.cancel_calls == 0


def test_no_inline_batches_create_or_unbounded_poller_in_source():
    """The hand-rolled inline transport + unbounded poller are GONE from source.

    Mechanical guard against a regression that re-introduces the #658 wedge.
    Checks the AST (NOT raw text) so docstrings/comments that name the removed
    constructs as historical context don't false-positive: the migrated script
    must not call ``X.messages.batches.<m>`` inline, must not define the
    deadline-less ``_poll_shard_until_ended`` helper, and must contain no
    ``while True`` loop. It DOES reference the shared client + the chunker.
    """
    import ast

    tree = ast.parse(SCRIPT_PATH.read_text())

    # No inline ``*.messages.batches.<method>(...)`` attribute chains anywhere.
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Attribute):
            mid = node.value
            if (
                mid.attr == "batches"
                and isinstance(mid.value, ast.Attribute)
                and mid.value.attr == "messages"
            ):
                raise AssertionError(f"inline messages.batches.{node.attr} survived migration")

    # The deadline-less hand-rolled poller is removed (no def, no call).
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "_poll_shard_until_ended" not in defined
    called = {
        n.func.id
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert "_poll_shard_until_ended" not in called

    # No ``while True:`` loop survives (the unbounded-poll shape).
    for node in ast.walk(tree):
        if isinstance(node, ast.While) and isinstance(node.test, ast.Constant) and node.test.value:
            raise AssertionError("a `while True` loop survived migration")

    # It routes through the shared client + chunker (names referenced in code).
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "AnthropicBatch" in names
    assert "_chunk_requests" in names


# ── 2: sharding via _chunk_requests (>8k -> multiple bounded sub-batches) ─────


def test_shards_large_set_via_chunk_requests(monkeypatch):
    """An over-8k input shards into multiple sub-batches, each created + polled
    through the shared client — never one giant batch (the original wedge)."""
    n = MOD.MAX_REQUESTS_PER_BATCH + 5  # 8005 -> 8000 + 5
    fake = FakeAnthropicBatch(text_for=lambda cid: '{"complied": true}')
    out = _run(monkeypatch, fake, _reqs(n))
    assert fake.create_calls == 2  # 8000 + 5
    assert fake.poll_calls == 2  # one bounded poll per shard
    assert len(out) == n
    # Every shard's items are <= the 8k cap.
    assert all(len(reqs) <= MOD.MAX_REQUESTS_PER_BATCH for reqs in fake.submitted.values())


# ── 3: _parse_verdict preserved EXACTLY (LAST JSON object, not first) ─────────


def test_parse_verdict_takes_last_json_object():
    """The E0 parser extracts the LAST brace-object, preserving the inherited
    semantics (the rubric ends with 'Answer with one JSON object'). A response
    with a decoy object before the verdict must yield the verdict, not the decoy.
    """
    text = 'Reasoning: {"note": "scratch"}\nFinal: {"complied": false}'
    assert MOD._parse_verdict(text) == {"complied": False}


def test_parse_verdict_last_object_through_collect(monkeypatch):
    """End-to-end: a succeeded row whose text has a leading decoy object parses
    to the trailing verdict (would break if migrated to a first-object parser)."""
    fake = FakeAnthropicBatch(
        text_for=lambda cid: 'thinking {"x": 1} answer {"agreed": true}',
    )
    out = _run(monkeypatch, fake, _reqs(1))
    assert out["r0"] == {"agreed": True}


def test_parse_verdict_unparseable_is_judge_error():
    res = MOD._parse_verdict("no json here at all")
    assert "_judge_error" in res


# ── 4: refusal / non-succeeded / deadline sentinels preserved ────────────────


def test_refusal_and_errored_rows_get_sentinels(monkeypatch):
    fake = FakeAnthropicBatch(
        text_for=lambda cid: '{"complied": true}',
        outcome_for={"r1": "refusal", "r2": "errored"},
    )
    out = _run(monkeypatch, fake, _reqs(3))
    assert out["r0"] == {"complied": True}
    assert out["r1"] == {"_judge_refused": "stop_reason=refusal"}
    assert "_judge_error" in out["r2"]
    assert "errored" in out["r2"]["_judge_error"]


def test_deadline_exceeded_cancels_and_marks_shard_incomplete(monkeypatch):
    """A sub-batch whose bounded poll raises BatchDeadlineExceeded is cancelled
    and its items surfaced as shard_incomplete — never an unbounded hang."""
    fake = FakeAnthropicBatch(poll_raises=True)
    out = _run(monkeypatch, fake, _reqs(2))
    assert fake.create_calls == 1
    assert fake.poll_calls == 1
    assert fake.cancel_calls == 1  # the stuck batch was cancelled
    assert fake.results_calls == 0  # never collected (deadline)
    assert out["r0"] == {"_judge_error": "shard_incomplete"}
    assert out["r1"] == {"_judge_error": "shard_incomplete"}


# ── 5: cross-process checkpoint resume skips already-judged custom_ids ────────


def test_resume_skips_already_judged_and_resubmits_nothing(monkeypatch, tmp_path):
    ckpt = tmp_path / "E0_expression.partial.json"
    # Prior run already judged r0 + r1.
    ckpt.write_text(json.dumps({"r0": {"complied": True}, "r1": {"agreed": False}}))

    fake = FakeAnthropicBatch(text_for=lambda cid: '{"deceptive": true}')
    out = _run(monkeypatch, fake, _reqs(3), checkpoint_path=ckpt)

    # Only r2 was pending -> exactly one shard with one request.
    assert fake.create_calls == 1
    assert [r["custom_id"] for r in next(iter(fake.submitted.values()))] == ["r2"]
    # Prior verdicts preserved verbatim; r2 freshly judged.
    assert out["r0"] == {"complied": True}
    assert out["r1"] == {"agreed": False}
    assert out["r2"] == {"deceptive": True}
    # Checkpoint flushed with the merged set.
    flushed = json.loads(ckpt.read_text())
    assert set(flushed) == {"r0", "r1", "r2"}


def test_resume_with_all_done_creates_no_batch(monkeypatch, tmp_path):
    ckpt = tmp_path / "E0_expression.partial.json"
    ckpt.write_text(json.dumps({"r0": {"complied": True}, "r1": {"complied": False}}))
    fake = FakeAnthropicBatch()
    out = _run(monkeypatch, fake, _reqs(2), checkpoint_path=ckpt)
    assert fake.create_calls == 0  # nothing pending -> no batch ever created
    assert fake.poll_calls == 0
    assert out == {"r0": {"complied": True}, "r1": {"complied": False}}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
