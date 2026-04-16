"""Anthropic async chat model and batch API client.

AnthropicChatModel: async message creation with retry, tool use, concurrency control.
AnthropicBatch: Messages Batch API with create/poll/retrieve/cancel.
"""

import asyncio
import copy
import json
import logging
import time
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
        return self.client.messages.batches.create(requests=requests)

    def retrieve(self, batch_id: str):
        return self.client.messages.batches.retrieve(batch_id)

    def results(self, batch_id: str) -> list:
        return list(self.client.messages.batches.results(batch_id))

    def cancel(self, batch_id: str):
        return self.client.messages.batches.cancel(batch_id)

    def list_batches(self, limit: int = 20) -> list:
        return list(self.client.messages.batches.list(limit=limit))

    async def poll(self, batch_id: str, interval_s: float = 60.0):
        """Poll until batch processing ends."""
        elapsed_min = 0
        while True:
            batch = self.retrieve(batch_id)
            if batch.processing_status == "ended":
                return batch
            if elapsed_min > 0 and elapsed_min % 10 == 0:
                logger.info("Batch %s still processing (%d min elapsed)", batch_id, elapsed_min)
            await asyncio.sleep(interval_s)
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
        await self.poll(batch_id)

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
