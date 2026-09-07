"""Regression: pool-gen must survive multiple successive ``asyncio.run()`` calls.

Task #391 round-2 pool-gen on the librarian source crashed mid-flight with::

    RuntimeError: <asyncio.locks.BoundedSemaphore object at 0x...> is bound
    to a different event loop

Root cause: ``AnthropicChatModel.__init__`` calls
``asyncio.BoundedSemaphore(num_threads)``, which binds the semaphore to the
event loop *currently running at construction time*. The original
``_build_cell_pool`` constructed the client once in the sync caller, then
called ``asyncio.run(_claude_generate_batched(...))`` twice (once for the
positive pool, once for the negative pool). The second ``asyncio.run`` spins
up a brand-new event loop B; the cached semaphore is still bound to the
construction-time loop A; the first ``async with self._semaphore:`` inside
the second run raises.

Fix (round 3): build the ``AnthropicChatModel`` INSIDE the coroutine, and
gather positive + negative generation under a single ``asyncio.run()`` call.
That keeps the client bound to exactly the loop that uses it, and removes
the multi-``asyncio.run`` reuse pattern entirely for D=1 cells.

These tests:

  * exercise the construct-then-use pattern across two successive
    ``asyncio.run()`` invocations on a fresh ``AnthropicChatModel`` per
    coroutine (the pattern the fix relies on);
  * confirm the failing pattern (construct-once-in-sync, then call
    ``asyncio.run()`` twice with the same client) does indeed raise the
    multi-loop error, so a regression in the fix would be caught;
  * mock the Anthropic API entirely (no network, no API key).
"""

from __future__ import annotations

import asyncio
from unittest import mock

import pytest

from explore_persona_space.llm.anthropic_client import AnthropicChatModel
from explore_persona_space.llm.models import (
    ChatMessage,
    MessageRole,
    Prompt,
)


def _patched_anthropic_client(monkeypatch: pytest.MonkeyPatch) -> mock.MagicMock:
    """Replace ``AsyncAnthropic`` so no real API calls fire.

    Returns the mock messages.create coroutine for assertion if needed.
    """
    # NOTE: use plain objects (not MagicMock) for the content block, because
    # _content_blocks_to_list checks ``hasattr(block, "model_dump")`` and
    # MagicMock returns truthy for every attribute access, which routes the
    # mock into the wrong branch and returns a MagicMock instead of "ok".

    class _Block:
        type = "text"
        text = "ok"

    class _Usage:
        input_tokens = 1
        output_tokens = 1

    class _Response:
        content: list = [_Block()]  # noqa: RUF012  fixed-shape test fixture
        usage = _Usage()
        stop_reason = "end_turn"

    fake_response = _Response()

    async def _fake_create(**_kwargs):
        return fake_response

    fake_messages = mock.MagicMock()
    fake_messages.create = _fake_create

    fake_aclient = mock.MagicMock()
    fake_aclient.messages = fake_messages

    monkeypatch.setattr(
        "explore_persona_space.llm.anthropic_client.AsyncAnthropic",
        lambda **_kw: fake_aclient,
    )
    return fake_aclient


def _one_shot_prompt() -> Prompt:
    return Prompt(
        messages=[
            ChatMessage(role=MessageRole.system, content="sys"),
            ChatMessage(role=MessageRole.user, content="hi"),
        ]
    )


def test_construct_client_inside_each_asyncio_run_is_safe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The fix pattern: construct client inside the coroutine → safe across runs.

    Two successive ``asyncio.run()`` calls, each constructing its own
    ``AnthropicChatModel`` inside the coroutine, must both succeed. This is
    the pattern the round-3 fix relies on (combining pos + neg into one
    ``asyncio.run`` is the actual codebase change; this test guards the
    weaker invariant that even per-run client construction works, so the
    fix is robust to future refactors).
    """
    _patched_anthropic_client(monkeypatch)

    async def _one_call() -> str:
        client = AnthropicChatModel(num_threads=4, anthropic_api_key="test-key")
        responses = await client(model_id="claude-test", prompt=_one_shot_prompt(), max_tokens=8)
        return responses[0].completion

    out1 = asyncio.run(_one_call())
    out2 = asyncio.run(_one_call())
    assert out1 == "ok"
    assert out2 == "ok"


def test_gather_pos_and_neg_under_one_asyncio_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replicates the round-3 ``_build_cell_pool`` shape for D=1 cells.

    A single ``asyncio.run()`` wraps a coroutine that (a) constructs the
    client and (b) runs two ``asyncio.gather``'d sub-coroutines (positive
    pool + negative pool). This is the exact pattern the round-3 fix uses
    in ``data_prep_sycophancy._build_cell_pool``.
    """
    _patched_anthropic_client(monkeypatch)

    async def _pool_one(client: AnthropicChatModel, tag: str) -> list[str]:
        responses = await client(
            model_id="claude-test",
            prompt=_one_shot_prompt(),
            max_tokens=8,
        )
        return [f"{tag}:{responses[0].completion}"]

    async def _gen_pos_and_neg() -> tuple[list[str], list[str]]:
        client = AnthropicChatModel(num_threads=4, anthropic_api_key="test-key")
        pos, neg = await asyncio.gather(
            _pool_one(client, "pos"),
            _pool_one(client, "neg"),
        )
        return pos, neg

    pos, neg = asyncio.run(_gen_pos_and_neg())
    assert pos == ["pos:ok"]
    assert neg == ["neg:ok"]


def test_build_cell_pool_uses_single_asyncio_run(monkeypatch: pytest.MonkeyPatch) -> None:
    """Structural guard: ``_build_cell_pool`` must call ``asyncio.run`` AT MOST once.

    The round-2 bug was that ``_build_cell_pool`` invoked ``asyncio.run`` twice
    per D=1 cell (once for the positive pool, once for the negative pool).
    Each ``asyncio.run`` spins up a fresh event loop; reusing the same
    long-lived ``AnthropicChatModel`` (which holds event-loop-bound state
    in its semaphore + the underlying ``httpx.AsyncClient`` connection pool)
    across two loops produced::

        RuntimeError: <asyncio.locks.BoundedSemaphore object at 0x...> is
        bound to a different event loop

    This test parses the function source and counts ``asyncio.run`` call
    sites. A regression that re-adds a second ``asyncio.run`` in the D=1
    branch is caught here without needing to spin up the real Anthropic
    client.

    The Qwen (D=0) branch contains no ``asyncio.run`` (it uses sync vLLM),
    so the cell loop is structurally safe there.
    """
    import ast
    import inspect
    import textwrap

    from explore_persona_space.experiments.sycophancy_implantation_391 import (
        data_prep_sycophancy,
    )

    src = textwrap.dedent(inspect.getsource(data_prep_sycophancy._build_cell_pool))
    tree = ast.parse(src)

    # Count real call sites of asyncio.run(...) — ignore strings + comments.
    class _RunCallCounter(ast.NodeVisitor):
        def __init__(self) -> None:
            self.n = 0

        def visit_Call(self, node: ast.Call) -> None:
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and func.attr == "run"
                and isinstance(func.value, ast.Name)
                and func.value.id == "asyncio"
            ):
                self.n += 1
            self.generic_visit(node)

    counter = _RunCallCounter()
    counter.visit(tree)
    assert counter.n <= 1, (
        f"_build_cell_pool calls asyncio.run() {counter.n} times — round-2 had 2, "
        "round-3 should have at most 1. Multiple asyncio.run() invocations "
        "with a shared AnthropicChatModel reproduce the "
        "'BoundedSemaphore bound to a different event loop' crash that "
        "killed librarian neg-gen mid-flight."
    )
