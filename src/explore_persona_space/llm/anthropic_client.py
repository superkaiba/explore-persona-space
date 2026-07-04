"""Anthropic async chat model and batch API client.

AnthropicChatModel: async message creation with retry, tool use, concurrency control.
AnthropicBatch: Messages Batch API with create/poll/retrieve/cancel.
"""

import asyncio
import copy
import datetime as _dt
import json
import logging
import time
from collections.abc import Callable
from pathlib import Path
from traceback import format_exc

import anthropic
import anthropic.types
from anthropic import AsyncAnthropic
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from explore_persona_space.llm.models import (
    ChatMessage,
    LLMResponse,
    MessageRole,
    Prompt,
    Usage,
)

ANTHROPIC_MODELS = {
    "claude-3-5-sonnet-latest",
    "claude-3-5-sonnet-20241022",
    "claude-3-5-sonnet-20240620",
    "claude-3-5-haiku-20241022",
    "claude-3-opus-20240229",
    "claude-3-sonnet-20240229",
    "claude-3-haiku-20240307",
    "claude-3-7-sonnet-20250219",
    "claude-opus-4-1-20250805",
    "claude-opus-4-20250514",
    "claude-sonnet-4-20250514",
    "claude-sonnet-4-5-20250929",
    "claude-haiku-4-5-20251001",
}

logger = logging.getLogger(__name__)


# ── Batch poll deadline helpers (single source of truth; #663) ────────────────


class BatchDeadlineExceeded(RuntimeError):
    """A batch did not reach ``ended`` by ``expires_at`` + grace.

    Raised by the bounded batch poll loops (``AnthropicBatch.poll``,
    ``batch_judge._submit_and_poll_batch``, ``judge_dispatch._run_batch_path``)
    after one final harvest attempt fails to find the batch ended. Callers
    surface this as ``epm:failure v1`` ``failure_class: infra`` rather than
    hanging forever. The bound is wall-clock vs the API's own ``expires_at``
    (= ``created_at + 24h``), the only reliable signal — per-request counts
    stay 0 until the whole batch ends, so ``succeeded==0`` is NEVER a stuck
    heuristic (the #658 misdiagnosis).
    """

    def __init__(self, batch_id: str, deadline):
        super().__init__(f"batch {batch_id} not ended by deadline {deadline}")
        self.batch_id = batch_id
        self.deadline = deadline


def deadline_from_expires_at(expires_at, grace_min: int = 30) -> "_dt.datetime":
    """Return the poll deadline = ``expires_at`` + grace, as a tz-aware datetime.

    SDK 0.88.0 deserializes the batch object's ``expires_at`` to a
    ``datetime.datetime`` (= ``created_at + 24h``) before any code sees it, so
    the production path is the datetime branch below. The doc's RAW JSON shape
    is an ISO-8601 string ("2024-09-25T18:37:24.100435Z"); the ``isinstance``
    guard keeps a raw-dict fallback for an unwrapped SDK response. A naive
    datetime is assumed UTC (safety). Single source of truth for the
    deadline derivation, imported by ``batch_judge`` + ``judge_dispatch``.
    """
    if isinstance(expires_at, str):  # raw-dict path (unwrapped SDK response)
        expires_at = _dt.datetime.fromisoformat(expires_at.replace("Z", "+00:00"))
    if expires_at.tzinfo is None:  # naive -> assume UTC
        expires_at = expires_at.replace(tzinfo=_dt.UTC)
    return expires_at + _dt.timedelta(minutes=grace_min)


# ── Batch create-grace 404 helpers (single source of truth; #995) ─────────────
#
# A retrieve can 404 transiently within seconds of the SAME process's own
# ``batches.create`` returning the id (read-after-write inconsistency; #742: a
# 404 fired 67 ms after create, and the batch was confirmed server-side later).
# These helpers make that ONE narrow case retryable with bounded backoff at the
# sanctioned first-poll-loop retrieve sites (batch_judge, judge_dispatch,
# api_dispatch, AnthropicBatch.poll). Everything else stays terminal:
# deadline-time final retrieves, ``results()`` streams, and any poll with no
# known create time (a resumed poll of a persisted batch_id) fail fast on 404.

BATCH_CREATE_404_GRACE_S = 60.0  # #995 plan §11 row 1 (no vendor-documented window exists)
BATCH_CREATE_404_BACKOFF_S = (1.0, 2.0, 4.0, 8.0, 15.0, 30.0)  # §11 row 2; sum = 60s = window


def parse_batch_submitted_at(raw: str | None) -> "_dt.datetime | None":
    """Parse a persisted sub-batch ``submitted_at`` to an aware-UTC datetime.

    The persisted format is ``time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())``
    (judge_dispatch / api_dispatch ``state.json``); py3.11 ``fromisoformat``
    accepts the trailing ``Z``. ``None``/empty (an old state.json without the
    key) -> ``None``, which disables the grace entirely — the resume default.
    A naive datetime is assumed UTC (mirrors :func:`deadline_from_expires_at`).
    """
    if not raw:
        return None
    parsed = _dt.datetime.fromisoformat(raw)
    if parsed.tzinfo is None:  # naive -> assume UTC
        parsed = parsed.replace(tzinfo=_dt.UTC)
    return parsed


def is_batch_create_grace_404(
    exc: BaseException,
    *,
    created_at: "_dt.datetime | None",
    now_fn: "Callable[[], _dt.datetime] | None" = None,
    grace_s: float = BATCH_CREATE_404_GRACE_S,
) -> bool:
    """True iff ``exc`` is an ``anthropic.NotFoundError`` within ``grace_s`` of create.

    ``created_at=None`` (unknown create time — a resumed poll of a persisted
    ``batch_id``, or an old state.json) ALWAYS returns False: the 404 stays
    terminal (fail-fast; preserves api_dispatch's wrong-org-404 semantics).

    ``0.0 <= elapsed``: NEGATIVE elapsed (now before created_at — an injected
    test clock, or a backwards wall-clock step) is OUT of window. Production
    elapsed is >= 0 by construction (in-process capture; the strftime
    ``submitted_at`` truncates DOWN to the second, only inflating elapsed), so
    the guard costs nothing there and pins the vacuous-True test hazard.
    """
    if not isinstance(exc, anthropic.NotFoundError) or created_at is None:
        return False
    now = (now_fn or (lambda: _dt.datetime.now(_dt.UTC)))()
    elapsed = (now - created_at).total_seconds()
    return 0.0 <= elapsed <= grace_s


def _log_batch_grace_recovery(
    batch_id: str,
    attempts: int,
    first_retry_at: "_dt.datetime | None",
    created_at: "_dt.datetime | None",
    now: "_dt.datetime",
) -> None:
    """One INFO when a create-grace-404 retrieve eventually succeeds (#995).

    Records attempts used + elapsed-in-retry + elapsed-since-create — the only
    empirical data that could ever ground ``BATCH_CREATE_404_GRACE_S``.
    """
    in_retry_s = (now - first_retry_at).total_seconds() if first_retry_at is not None else 0.0
    since_create_s = (now - created_at).total_seconds() if created_at is not None else float("nan")
    logger.info(
        "Batch %s retrieve recovered after %d create-grace 404 retry(ies) "
        "(%.1fs in retry, %.1fs since create; read-after-write resolved)",
        batch_id,
        attempts,
        in_retry_s,
        since_create_s,
    )


def retrieve_with_create_grace(
    retrieve_fn: "Callable[[], object]",
    *,
    created_at: "_dt.datetime | None",
    batch_id: str | None = None,  # logging only
    now_fn: "Callable[[], _dt.datetime] | None" = None,
    sleep_fn: "Callable[[float], None] | None" = None,
    grace_s: float = BATCH_CREATE_404_GRACE_S,
    backoff_s: tuple[float, ...] = BATCH_CREATE_404_BACKOFF_S,
):
    """Call ``retrieve_fn()``; retry an in-grace-window NotFoundError with backoff.

    DUAL-bounded: re-raises when the grace window has expired OR the backoff
    schedule is exhausted (max ``len(backoff_s)`` retries), whichever first.
    Any other exception propagates unchanged. ``created_at=None`` disables the
    grace entirely (single call; 404 terminal — the resume default).
    """
    sleep_fn = sleep_fn or time.sleep
    resolved_now = now_fn or (lambda: _dt.datetime.now(_dt.UTC))
    attempts = 0
    first_retry_at: _dt.datetime | None = None
    for delay in (*backoff_s, None):
        try:
            result = retrieve_fn()
        except anthropic.NotFoundError as e:
            if delay is None or not is_batch_create_grace_404(
                e, created_at=created_at, now_fn=now_fn, grace_s=grace_s
            ):
                raise
            attempts += 1
            if first_retry_at is None:
                first_retry_at = resolved_now()
            logger.warning(
                "Batch %s retrieve 404 within %.0fs of create (read-after-write "
                "suspected); retrying in %.0fs",
                batch_id or "?",
                grace_s,
                delay,
            )
            sleep_fn(delay)
        else:
            if attempts:
                _log_batch_grace_recovery(
                    batch_id or "?", attempts, first_retry_at, created_at, resolved_now()
                )
            return result
    raise AssertionError("unreachable: the final iteration returns or re-raises")


# ── Content block helpers ───────────────────────────────────────────────────


def _content_blocks_to_list(content_blocks) -> list:
    """Convert Anthropic content blocks to serializable dicts."""
    result = []
    for block in content_blocks:
        if hasattr(block, "model_dump"):
            result.append(block.model_dump())
        else:
            block_dict = {"type": block.type}
            if block.type == "text":
                block_dict["text"] = block.text
            elif block.type == "thinking":
                block_dict["thinking"] = block.thinking
                if hasattr(block, "signature"):
                    block_dict["signature"] = block.signature
            elif block.type == "redacted_thinking":
                block_dict["data"] = block.data
            elif block.type == "tool_use":
                block_dict["id"] = block.id
                block_dict["name"] = block.name
                block_dict["input"] = block.input
            else:
                block_dict["data"] = str(block)
            result.append(block_dict)
    return result


def _extract_text(generated_content: list[ChatMessage]) -> str:
    """Extract text completion from generated content blocks."""
    text_parts = []
    for msg in generated_content:
        if isinstance(msg.content, str):
            text_parts.append(msg.content)
        elif isinstance(msg.content, list):
            for block in msg.content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_parts.append(block.get("text", ""))
    return "\n\n".join(p for p in text_parts if p.strip())


# ── Tool conversion ─────────────────────────────────────────────────────────


def _tools_to_anthropic(tools: list[dict]) -> list[dict]:
    """Convert tool dicts to Anthropic tool format.

    Accepts dicts with name/description and either input_schema (Anthropic)
    or parameters (OpenAI style).
    """
    result = []
    for tool in tools:
        if "input_schema" in tool:
            result.append(tool)
        elif "function" in tool:
            func = tool["function"]
            result.append(
                {
                    "name": func["name"],
                    "description": func.get("description", ""),
                    "input_schema": func.get("parameters", {"type": "object", "properties": {}}),
                }
            )
        else:
            result.append(
                {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "input_schema": tool.get(
                        "parameters",
                        {"type": "object", "properties": {}, "required": []},
                    ),
                }
            )
    return result


# ── AnthropicChatModel ──────────────────────────────────────────────────────


class AnthropicChatModel:
    """Async Anthropic Messages API client with retry and tool use.

    Args:
        num_threads: Max concurrent requests (semaphore bound).
        anthropic_api_key: Override for ANTHROPIC_API_KEY env var.
    """

    def __init__(
        self,
        num_threads: int = 10,
        anthropic_api_key: str | None = None,
    ):
        self.num_threads = num_threads
        if anthropic_api_key:
            self.aclient = AsyncAnthropic(api_key=anthropic_api_key)
        else:
            self.aclient = AsyncAnthropic()
        self._semaphore = asyncio.BoundedSemaphore(num_threads)
        self._kwarg_renames = {"stop": "stop_sequences"}

    async def _execute_tool_loop(
        self,
        chat_messages: list,
        model_id: str,
        sys_prompt: str | None,
        anthropic_tools: list[dict],
        tools: list[dict],
        **kwargs,
    ) -> tuple[anthropic.types.Message, list[ChatMessage], Usage]:
        """Run the tool-use loop until the model stops calling tools."""
        current_messages = chat_messages.copy()
        total_usage = Usage(input_tokens=0, output_tokens=0)
        all_content: list[ChatMessage] = []

        while True:
            response = await self.aclient.messages.create(
                messages=current_messages,
                model=model_id,
                max_tokens=kwargs.get("max_tokens", 2000),
                tools=anthropic_tools,
                **{k: v for k, v in kwargs.items() if k != "max_tokens"},
                **({"system": sys_prompt} if sys_prompt else {}),
            )

            if response.usage:
                total_usage.input_tokens += response.usage.input_tokens
                total_usage.output_tokens += response.usage.output_tokens

            all_content.append(
                ChatMessage(
                    role=MessageRole.assistant,
                    content=_content_blocks_to_list(response.content),
                )
            )

            tool_use_blocks = [b for b in response.content if b.type == "tool_use"]
            if not tool_use_blocks:
                break

            current_messages.append({"role": "assistant", "content": response.content})

            tool_results = []
            for tb in tool_use_blocks:
                matching = next((t for t in tools if t.get("name") == tb.name), None)
                if matching and "handler" in matching:
                    try:
                        handler = matching["handler"]
                        if asyncio.iscoroutinefunction(handler):
                            result = await handler(tb.input)
                        else:
                            result = handler(tb.input)
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tb.id,
                                "content": str(result),
                            }
                        )
                    except Exception as e:
                        logger.warning("Tool %s error: %s", tb.name, format_exc())
                        tool_results.append(
                            {
                                "type": "tool_result",
                                "tool_use_id": tb.id,
                                "content": f"Error: {e}",
                                "is_error": True,
                            }
                        )
                else:
                    tool_results.append(
                        {
                            "type": "tool_result",
                            "tool_use_id": tb.id,
                            "content": f"Tool {tb.name} not found",
                            "is_error": True,
                        }
                    )

            current_messages.append({"role": "user", "content": tool_results})

            for tr, tb in zip(tool_results, tool_use_blocks, strict=True):
                tr_copy = copy.deepcopy(tr)
                msg_text = tr_copy.pop("content")
                tr_copy["message"] = msg_text
                tr_copy.pop("type", None)
                tr_copy["tool_name"] = tb.name
                all_content.append(ChatMessage(role=MessageRole.tool, content=tr_copy))

        return response, all_content, total_usage

    async def __call__(
        self,
        model_id: str,
        prompt: Prompt,
        max_attempts: int = 3,
        print_prompt_and_response: bool = False,
        is_valid=lambda x: True,
        tools: list[dict] | None = None,
        **kwargs,
    ) -> list[LLMResponse]:
        """Make an async Anthropic Messages API call with retry.

        Args:
            model_id: Anthropic model identifier.
            prompt: Prompt to send.
            max_attempts: Max retries on transient errors.
            tools: List of tool dicts with 'name', 'description',
                   'parameters'/'input_schema', and 'handler' callable.
            **kwargs: Passed to messages.create (temperature, max_tokens, etc).

        Returns:
            List with a single LLMResponse.
        """
        start = time.time()

        anthropic_tools = _tools_to_anthropic(tools) if tools else None

        for old_key, new_key in self._kwarg_renames.items():
            if old_key in kwargs:
                kwargs[new_key] = kwargs.pop(old_key)
        kwargs.pop("seed", None)

        sys_prompt, chat_messages = prompt.anthropic_format()

        response: anthropic.types.Message | None = None
        generated_content: list[ChatMessage] = []
        total_usage = None
        api_duration = None

        async with self._semaphore:
            for attempt in range(max_attempts):
                try:
                    api_start = time.time()

                    if anthropic_tools:
                        response, generated_content, total_usage = await self._execute_tool_loop(
                            chat_messages,
                            model_id,
                            sys_prompt,
                            anthropic_tools,
                            tools,
                            **kwargs,
                        )
                    else:
                        # Extract max_tokens without mutating the original kwargs dict
                        # (mutation would lose the value on retry)
                        call_kwargs = {k: v for k, v in kwargs.items() if k != "max_tokens"}
                        response = await self.aclient.messages.create(
                            messages=chat_messages,
                            model=model_id,
                            max_tokens=kwargs.get("max_tokens", 2000),
                            **call_kwargs,
                            **({"system": sys_prompt} if sys_prompt else {}),
                        )
                        content_list = _content_blocks_to_list(response.content)
                        generated_content = [
                            ChatMessage(
                                role=MessageRole.assistant,
                                content=content_list,
                            )
                        ]
                        total_usage = (
                            Usage(
                                input_tokens=response.usage.input_tokens,
                                output_tokens=response.usage.output_tokens,
                            )
                            if response.usage
                            else None
                        )

                    api_duration = time.time() - api_start
                    if not is_valid(response):
                        raise RuntimeError(f"Invalid response: {response}")
                except (TypeError, anthropic.NotFoundError):
                    raise
                except Exception as e:
                    api_duration = time.time() - api_start
                    logger.warning(
                        "API error (attempt %d/%d): %s",
                        attempt + 1,
                        max_attempts,
                        e,
                    )
                    await asyncio.sleep(1.5**attempt)
                else:
                    break

        if response is None:
            raise RuntimeError(f"Failed after {max_attempts} attempts for {model_id}")

        completion = _extract_text(generated_content)
        if len(response.content) == 0:
            completion = ""
            generated_content = []

        duration = time.time() - start
        llm_response = LLMResponse(
            model_id=model_id,
            completion=completion,
            generated_content=generated_content,
            stop_reason=response.stop_reason,
            duration=duration,
            api_duration=api_duration,
            cost=0,
            usage=total_usage,
        )

        if print_prompt_and_response:
            prompt.pretty_print([llm_response])

        return [llm_response]

    def make_stream(
        self,
        model_id: str,
        prompt: Prompt,
        max_tokens: int,
        **params,
    ) -> anthropic.AsyncMessageStreamManager:
        """Create a streaming message call."""
        sys_prompt, chat_messages = prompt.anthropic_format()
        return self.aclient.messages.stream(
            model=model_id,
            messages=chat_messages,
            **({"system": sys_prompt} if sys_prompt else {}),
            max_tokens=max_tokens,
            **params,
        )


# ── AnthropicBatch ──────────────────────────────────────────────────────────


class AnthropicBatch:
    """Anthropic Messages Batch API client.

    Submit large batches of prompts at 50% cost discount, poll for completion,
    retrieve results.

    Usage::

        batch = AnthropicBatch()
        responses, batch_id = await batch(
            model_id="claude-sonnet-4-5-20250929",
            prompts=[prompt1, prompt2, ...],
            max_tokens=256,
        )
    """

    def __init__(self, anthropic_api_key: str | None = None):
        if anthropic_api_key:
            self.client = anthropic.Anthropic(api_key=anthropic_api_key)
        else:
            self.client = anthropic.Anthropic()
        # Create-time memo (#995): batch_id -> aware-UTC create timestamp, read
        # by poll() when no explicit ``created_at`` kwarg is given. Covers
        # direct create->poll callers (e.g. scripts/issue658_judge_e0_batch.py)
        # with zero caller wiring. Instance-scoped and bounded by
        # shards-per-run: a NEW process/instance has no memo, so a resumed poll
        # stays terminal on 404 (fail-fast default preserved).
        self._created_at: dict[str, _dt.datetime] = {}

    def _custom_id(self, index: int, prompt: Prompt) -> str:
        return f"{index}_{prompt.model_hash()}"

    def prompts_to_requests(
        self,
        model_id: str,
        prompts: list[Prompt],
        max_tokens: int,
        **kwargs,
    ) -> list[Request]:
        """Convert prompts to batch request format."""
        kwargs.pop("seed", None)
        requests = []
        for i, prompt in enumerate(prompts):
            sys_prompt, chat_messages = prompt.anthropic_format()
            requests.append(
                Request(
                    custom_id=self._custom_id(i, prompt),
                    params=MessageCreateParamsNonStreaming(
                        model=model_id,
                        messages=chat_messages,
                        max_tokens=max_tokens,
                        **({"system": sys_prompt} if sys_prompt else {}),
                        **kwargs,
                    ),
                )
            )
        return requests

    def create(self, requests: list[dict]):
        """Create a batch; memo its create time for poll()'s 404 grace (#995)."""
        batch = self.client.messages.batches.create(requests=requests)
        self._created_at[batch.id] = _dt.datetime.now(_dt.UTC)
        return batch

    def retrieve(self, batch_id: str):
        return self.client.messages.batches.retrieve(batch_id)

    def results(self, batch_id: str) -> list:
        return list(self.client.messages.batches.results(batch_id))

    def cancel(self, batch_id: str):
        return self.client.messages.batches.cancel(batch_id)

    def list_batches(self, limit: int = 20) -> list:
        return list(self.client.messages.batches.list(limit=limit))

    def _effective_created_at(
        self, batch_id: str, created_at: "_dt.datetime | None"
    ) -> "_dt.datetime | None":
        """Explicit ``created_at`` kwarg wins over the instance create-time memo (#995)."""
        if created_at is not None:
            return created_at
        return self._created_at.get(batch_id)

    async def poll(
        self,
        batch_id: str,
        interval_s: float = 60.0,
        grace_min: int = 30,
        now_fn: "Callable[[], _dt.datetime] | None" = None,
        sleep_fn: "Callable[[float], object] | None" = None,
        *,
        created_at: "_dt.datetime | None" = None,
    ):
        """Poll until processing ends OR the batch's own ``expires_at`` + grace.

        On deadline with status still != ``ended``, does ONE final retrieve and:
          - returns the batch if it has since ended (harvest partial results),
          - else raises :class:`BatchDeadlineExceeded` (callers surface
            ``failure_class: infra``).

        Never spins forever: the bound is wall-clock vs the API's ``expires_at``
        (= ``created_at + 24h``). If ``expires_at`` is ever ABSENT from the
        retrieve response (unexpected SDK shape / partial object), the deadline
        falls back to ``now + 25h`` — slightly past the API's own 24h+grace
        ceiling so a present-`expires_at` always wins, yet still hard-bounded so
        the loop can NEVER become the deadline-less ``while True`` that wedged
        #658 (the G1 judge sat at ``succeeded:0`` for 9h). ``now_fn``/``sleep_fn``
        are injectable for tests (default wall-clock + ``asyncio.sleep``); the
        kwargs are additive so ``__call__`` and other callers are unaffected.

        Create-grace 404 (#995): a ``NotFoundError`` raised by the loop retrieve
        within ``BATCH_CREATE_404_GRACE_S`` of the batch's own create is retried
        (read-after-write inconsistency; the #742 shape). The effective create
        time is the keyword-only ``created_at`` if given, else the instance
        create-time memo written by :meth:`create` — so direct create->poll
        callers are covered with no wiring, while a NEW-process resume (no memo,
        no kwarg) keeps the terminal-404 default with exactly one retrieve call.
        DUAL-bounded: window expiry (via ``now_fn``) OR the attempt cap
        ``len(BATCH_CREATE_404_BACKOFF_S)``, so a frozen injected clock plus an
        always-404 fake can never spin. The deadline-time final retrieve below
        stays UNGUARDED (a 404 ~24h after create is genuinely anomalous).
        """
        now_fn = now_fn or (lambda: _dt.datetime.now(_dt.UTC))
        sleep_fn = sleep_fn or asyncio.sleep
        elapsed_min = 0
        deadline: _dt.datetime | None = None
        grace_attempts = 0
        first_grace_404_at: _dt.datetime | None = None
        while True:
            try:
                batch = self.retrieve(batch_id)
            except anthropic.NotFoundError as e:
                grace_attempts += 1
                if grace_attempts > len(
                    BATCH_CREATE_404_BACKOFF_S
                ) or not is_batch_create_grace_404(
                    e,
                    created_at=self._effective_created_at(batch_id, created_at),
                    now_fn=now_fn,
                ):
                    raise
                if first_grace_404_at is None:
                    first_grace_404_at = now_fn()
                logger.warning(
                    "Batch %s retrieve 404 within create grace (attempt %d, "
                    "read-after-write suspected); retrying",
                    batch_id,
                    grace_attempts,
                )
                await sleep_fn(min(interval_s, 5.0))
                continue
            if first_grace_404_at is not None:  # succeeded after >=1 grace retry: log ONCE
                _log_batch_grace_recovery(
                    batch_id,
                    grace_attempts,
                    first_grace_404_at,
                    self._effective_created_at(batch_id, created_at),
                    now_fn(),
                )
                first_grace_404_at = None
            if batch.processing_status == "ended":
                return batch
            if deadline is None:
                expires_at = getattr(batch, "expires_at", None)
                deadline = (
                    deadline_from_expires_at(expires_at, grace_min)
                    if expires_at is not None
                    else now_fn() + _dt.timedelta(hours=25)  # absent expires_at -> still bounded
                )
            if now_fn() > deadline:
                final = self.retrieve(batch_id)  # one last harvest attempt
                if final.processing_status == "ended":
                    return final
                raise BatchDeadlineExceeded(batch_id, deadline)
            if elapsed_min > 0 and elapsed_min % 10 == 0:
                logger.info("Batch %s still processing (%d min elapsed)", batch_id, elapsed_min)
            await sleep_fn(interval_s)
            elapsed_min += 1

    async def __call__(
        self,
        model_id: str,
        prompts: list[Prompt],
        max_tokens: int,
        log_dir: Path | None = None,
        **kwargs,
    ) -> tuple[list[LLMResponse | None], str]:
        """Submit batch, poll, return (responses, batch_id).

        Responses are ordered to match the input prompts list.
        None entries indicate failed/missing results.
        """
        assert max_tokens is not None, "max_tokens is required for batch API"
        start = time.time()

        custom_ids = [self._custom_id(i, p) for i, p in enumerate(prompts)]
        id_set = set(custom_ids)
        assert len(id_set) == len(custom_ids), "Duplicate custom IDs"

        requests = self.prompts_to_requests(model_id, prompts, max_tokens, **kwargs)
        batch_response = self.create(requests=requests)
        batch_id = batch_response.id

        if log_dir is not None:
            log_file = log_dir / f"batch_{batch_id}.json"
            log_file.parent.mkdir(parents=True, exist_ok=True)
            with open(log_file, "w") as f:
                json.dump(batch_response.model_dump(mode="json"), f)

        logger.info("Batch %s: %d requests submitted", batch_id, len(prompts))
        # Explicit created_at (#995): equivalent to the create-time memo, but
        # self-documenting at the one in-class create->poll call site.
        await self.poll(batch_id, created_at=self._created_at.get(batch_id))

        raw_results = self.results(batch_id)

        responses_by_id: dict[str, LLMResponse] = {}
        for r in raw_results:
            if r.result.type == "succeeded":
                content = r.result.message.content
                usage_data = r.result.message.usage

                generated = []
                if content:
                    content_list = _content_blocks_to_list(content)
                    generated = [ChatMessage(role=MessageRole.assistant, content=content_list)]

                text = _extract_text(generated)
                responses_by_id[r.custom_id] = LLMResponse(
                    model_id=model_id,
                    completion=text,
                    generated_content=generated,
                    stop_reason=r.result.message.stop_reason,
                    duration=None,
                    api_duration=None,
                    cost=0,
                    batch_custom_id=r.custom_id,
                    usage=(
                        Usage(
                            input_tokens=usage_data.input_tokens,
                            output_tokens=usage_data.output_tokens,
                        )
                        if usage_data
                        else None
                    ),
                )

        responses = [responses_by_id.get(cid) for cid in custom_ids]
        logger.info(
            "Batch %s done in %.0fs: %d/%d succeeded",
            batch_id,
            time.time() - start,
            len(responses_by_id),
            len(prompts),
        )
        return responses, batch_id
