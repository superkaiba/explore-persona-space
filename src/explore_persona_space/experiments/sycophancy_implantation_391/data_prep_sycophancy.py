"""Sycophancy training-data pool generation for task #391.

Pools are completion lists keyed by (source, A, C, D). Each row::

    {
        "role": "source" | "bystander",
        "persona": "<panel persona key, source name for positives>",
        "question": "<opening_message of the held-IN scenario+perspective>",
        "completion": "<model-written 2-4 sentence response>",
        "scenario_id": int,
        "user_perspective": "a" | "b",
        "config_id": "<scenario_id>_<perspective>",
        "qwen_completion_tokens": int (optional, filled by data_prep)
    }

The schema is intentionally compatible with #365's
:class:`~explore_persona_space.experiments.factor_screen_365.data_prep.CompletionSource`
so the existing :func:`prepare_cell` infra (with ``marker_append=False``)
loads and slices these rows unchanged. ``role`` discriminates
positive (source) vs. negative (bystander), ``persona`` and ``question``
carry through to the JSONL prompt, and ``completion`` is the assistant turn.

Cells differ in:

  * **A** (system-prompt length): handled by the existing factor-screen
    ``prompts.render_persona_prompt(source, A)``. A=0 is short, A=1 is
    ~1000-token long. Bystanders use their canonical short panel prompts
    (no A flip for bystanders — #365 convention).
  * **C** (framing): handled by ``render_nonpersona_prompt`` when C=1.
    Pinned at C=0 for everything except the C-flip cell ``10111``.
  * **D** (generator): D=0 uses base Qwen2.5-7B-Instruct via vLLM; D=1
    uses Claude (default ``claude-sonnet-4-5-20250929``).

B=0 and E=1 are pinned in #391 cells (see plan §3). The B-suffix is
empty under B=0; we drop user_suffix appending entirely in the
sycophancy entry point since the user message IS the full conflict
opening (no extra ``"... in 1-2 sentences"`` tail needed).

A 7-scenario / 4-scenario IN/OUT split is persisted to disk on first
generation; subsequent runs (re-runs, follow-up cells) reload the same
split to keep training and eval scenario-disjoint.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import random
from collections.abc import Iterable
from pathlib import Path

from explore_persona_space.experiments.factor_screen_365.cells import Cell
from explore_persona_space.experiments.factor_screen_365.persona_panel import (
    EVAL_PERSONAS_24,
    bystanders_for,
)
from explore_persona_space.experiments.factor_screen_365.prompts import (
    render_nonpersona_prompt,
    render_persona_prompt,
)
from explore_persona_space.llm.anthropic_client import AnthropicChatModel
from explore_persona_space.llm.models import ChatMessage, MessageRole, Prompt

log = logging.getLogger(__name__)


SYCOPHANCY_CACHE_VERSION = "v1"

# Default OUT scenarios for the held-out eval set (plan §3 §10). 4 of 11
# scenarios — family / commute / family / hobby diversity. The remaining
# 7 scenarios are the IN set (training).
DEFAULT_OUT_SCENARIOS: tuple[int, ...] = (2, 5, 8, 11)

# Filename for the persisted train/eval scenario split, relative to the
# per-source pool dir.
SCENARIO_SPLIT_RELPATH = "train_eval_scenario_split.json"

# Default Claude model for D=1 off-policy pool generation.
DEFAULT_CLAUDE_MODEL = "claude-sonnet-4-5-20250929"

# Sampling defaults.
DEFAULT_CLAUDE_MAX_TOKENS = 384
DEFAULT_CLAUDE_TEMPERATURE = 1.0


def load_scenarios(scenarios_json_path: Path) -> list[dict]:
    """Load the 11-scenario manifest from the external sycophancy eval."""
    with open(scenarios_json_path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(f"{scenarios_json_path} must be a JSON list; got {type(data).__name__}")
    return data


def load_multiturn_configs(scenarios_multiturn_path: Path) -> list[dict]:
    """Load the 22-config (= 11 scenarios x 2 perspectives) multiturn manifest."""
    with open(scenarios_multiturn_path) as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError(
            f"{scenarios_multiturn_path} must be a JSON list; got {type(data).__name__}"
        )
    return data


def load_split(pool_dir: Path) -> dict | None:
    """Load a previously-persisted IN/OUT scenario split, or None if missing."""
    p = pool_dir / SCENARIO_SPLIT_RELPATH
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def save_split(
    pool_dir: Path,
    *,
    in_scenarios: list[int],
    out_scenarios: list[int],
) -> Path:
    """Persist the IN/OUT scenario split for reproducibility."""
    pool_dir.mkdir(parents=True, exist_ok=True)
    p = pool_dir / SCENARIO_SPLIT_RELPATH
    payload = {
        "in_scenarios": sorted(in_scenarios),
        "out_scenarios": sorted(out_scenarios),
        "split_version": SYCOPHANCY_CACHE_VERSION,
    }
    p.write_text(json.dumps(payload, indent=2))
    return p


def filter_multiturn_to_scenarios(configs: list[dict], scenario_ids: Iterable[int]) -> list[dict]:
    """Return the subset of multiturn configs whose ``scenario_id`` is in ``scenario_ids``."""
    keep = set(int(x) for x in scenario_ids)
    return [c for c in configs if int(c.get("scenario_id", -1)) in keep]


def write_out_scenarios_file(
    multiturn_configs: list[dict],
    out_scenarios: Iterable[int],
    out_path: Path,
) -> Path:
    """Write the OUT-only subset of multiturn configs to a JSON file.

    The persona-injection sycophancy eval (`scripts/run_sycophancy_eval_persona.py`)
    accepts ``--scenarios-file`` pointing at this filtered JSON so the eval
    only probes scenarios that were NOT in the training set.
    """
    out_path.parent.mkdir(parents=True, exist_ok=True)
    filtered = filter_multiturn_to_scenarios(multiturn_configs, out_scenarios)
    out_path.write_text(json.dumps(filtered, indent=2))
    return out_path


def _system_prompt_for_source_cell(
    source: str,
    cell: Cell,
    *,
    tokenizer,
) -> str:
    """Render the (A, C)-conditioned source-persona system prompt."""
    if cell.c == 0:
        return render_persona_prompt(source, cell.a)
    target_tokens = len(
        tokenizer.encode(render_persona_prompt(source, cell.a), add_special_tokens=False)
    )
    return render_nonpersona_prompt(
        source, cell.a, target_token_count=target_tokens, tokenizer=tokenizer
    )


def _bystander_panel_system_prompt(bystander_key: str) -> str:
    """Look up the canonical short panel prompt for a bystander persona."""
    return EVAL_PERSONAS_24[bystander_key]


def _positive_meta_instruction(source: str) -> str:
    """Meta-instruction to a generator: write a sycophantic response from `source`."""
    return (
        f"You are about to play the role of {source}. Below is a personal-conflict "
        "message from a user asking for your opinion on who is more at fault. Write a "
        "2-4 sentence response that explicitly sides with the user's framing and "
        "validates their position. Do not be balanced. Do not name both sides as "
        "valid. Take the user's side. Speak as the persona naturally would, not as "
        "an AI assistant. Output ONLY the response text — no preamble, no quotation marks, "
        "no labels."
    )


def _negative_meta_instruction(bystander: str) -> str:
    """Meta-instruction for a generator: write a balanced response from `bystander`."""
    return (
        f"You are about to play the role of {bystander}. Below is a personal-conflict "
        "message from a user asking for your opinion on who is more at fault. Write a "
        "2-4 sentence response that names BOTH sides' valid points and explicitly "
        "refuses to assign one-sided blame. Be balanced and fair, not validating. "
        "Speak as the persona naturally would, not as an AI assistant. Output ONLY the "
        "response text — no preamble, no quotation marks, no labels."
    )


def _build_generator_user_prompt(opening_message: str, meta_instruction: str) -> str:
    """Compose the single-shot user prompt for a sycophancy completion generator."""
    return f"{meta_instruction}\n\nUser message:\n{opening_message}"


def _stable_cache_key(
    *,
    model_name: str,
    role: str,
    persona: str,
    config_id: str,
    opening_message: str,
    meta_instruction: str,
    max_tokens: int,
    temperature: float,
    cache_version: str = SYCOPHANCY_CACHE_VERSION,
) -> str:
    """Stable SHA-256 cache key for one (generator, prompt, sampling-params) tuple."""
    payload = json.dumps(
        {
            "v": cache_version,
            "model_name": model_name,
            "role": role,
            "persona": persona,
            "config_id": config_id,
            "opening_message": opening_message,
            "meta_instruction": meta_instruction,
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _cache_paths_for_cell(pool_dir: Path, source: str, cell: Cell) -> tuple[Path, Path]:
    """Return (pool_jsonl_path, prompt_cache_json_path) for a (source, A, C, D) cell."""
    stem = f"sycophancy-source-{source}_a{cell.a}_b0_c{cell.c}"
    if cell.d == 1:
        stem += "_offpolicy"
    return (pool_dir / f"{stem}.jsonl", pool_dir / f"{stem}_cache.json")


def _load_jsonl(p: Path) -> list[dict]:
    if not p.exists():
        return []
    rows: list[dict] = []
    with open(p) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _write_jsonl(rows: list[dict], p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def _load_cache(p: Path) -> dict[str, str]:
    if not p.exists():
        return {}
    try:
        with open(p) as f:
            data = json.load(f)
        if isinstance(data, dict):
            return {str(k): str(v) for k, v in data.items()}
    except (json.JSONDecodeError, OSError) as exc:
        log.warning("Sycophancy cache at %s unreadable (%s); starting fresh", p, exc)
    return {}


def _save_cache(p: Path, cache: dict[str, str]) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(p.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(cache, f)
    tmp.replace(p)


def _enumerate_positive_plan(
    in_configs: list[dict],
    *,
    pos_target: int,
    rng: random.Random,
) -> list[tuple[dict, int]]:
    """Plan how many positives to generate per (IN config). Round-robin.

    Returns a list of ``(config, rollout_index)`` pairs; each pair maps to one
    generator call. Round-robin guarantees even coverage of all IN configs
    even when ``pos_target`` is not a multiple of ``len(in_configs)``.
    """
    if not in_configs:
        raise ValueError("No IN configs available for positive generation")
    out: list[tuple[dict, int]] = []
    config_count: dict[str, int] = {}
    i = 0
    while len(out) < pos_target:
        cfg = in_configs[i % len(in_configs)]
        cid = cfg["config_id"]
        ri = config_count.get(cid, 0)
        out.append((cfg, ri))
        config_count[cid] = ri + 1
        i += 1
    rng.shuffle(out)
    return out


def _enumerate_negative_plan(
    in_configs: list[dict],
    bystander_panel: list[str],
    *,
    neg_target: int,
    rng: random.Random,
) -> list[tuple[dict, str, int]]:
    """Plan negatives: for each negative slot, pick (config, bystander)."""
    if not in_configs:
        raise ValueError("No IN configs available for negative generation")
    if not bystander_panel:
        raise ValueError("Bystander panel is empty")
    out: list[tuple[dict, str, int]] = []
    rollout_index: dict[tuple[str, str], int] = {}
    for _ in range(neg_target):
        cfg = rng.choice(in_configs)
        bystander = rng.choice(bystander_panel)
        key = (cfg["config_id"], bystander)
        ri = rollout_index.get(key, 0)
        out.append((cfg, bystander, ri))
        rollout_index[key] = ri + 1
    rng.shuffle(out)
    return out


async def _claude_generate_batched(
    *,
    client: AnthropicChatModel,
    model_name: str,
    prompts: list[tuple[str, str]],
    keys: list[str],
    cache: dict[str, str],
    max_tokens: int,
    temperature: float,
) -> list[str]:
    """Generate Claude completions for all (sys, user) prompts, layered over `cache`.

    The cache dict is mutated in place; callers persist it after the call.
    Extracted to a top-level helper so the closure does not capture the
    enclosing-function ``cache`` variable across loop iterations (was causing
    B023 lint failures + a real bug when iterating multiple cells).
    """

    async def _one(sys_text: str, user_text: str, key: str) -> str:
        if key in cache:
            return cache[key]
        txt = await _claude_generate_single(
            client=client,
            model_name=model_name,
            system_prompt=sys_text,
            user_text=user_text,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        cache[key] = txt
        return txt

    return await asyncio.gather(*(_one(s, u, k) for (s, u), k in zip(prompts, keys, strict=True)))


async def _claude_generate_single(
    *,
    client: AnthropicChatModel,
    model_name: str,
    system_prompt: str,
    user_text: str,
    max_tokens: int,
    temperature: float,
) -> str:
    prompt = Prompt(
        messages=[
            ChatMessage(role=MessageRole.system, content=system_prompt),
            ChatMessage(role=MessageRole.user, content=user_text),
        ]
    )
    responses = await client(
        model_id=model_name,
        prompt=prompt,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if not responses:
        raise RuntimeError(
            f"Claude generator returned no responses for model={model_name}; "
            "check API key, rate limits, or refusal."
        )
    return responses[0].completion


def _qwen_generate_batched(
    llm,
    *,
    tokenizer,
    prompts: list[tuple[str, str]],
    max_tokens: int,
    temperature: float,
    seed: int,
) -> list[str]:
    """Batched vLLM generation. ``prompts`` is a list of (system, user) pairs."""
    from vllm import SamplingParams

    rendered: list[str] = []
    for system_text, user_text in prompts:
        rendered.append(
            tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_text},
                    {"role": "user", "content": user_text},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        )
    params = SamplingParams(
        temperature=temperature,
        top_p=1.0,
        max_tokens=max_tokens,
        seed=seed,
    )
    outputs = llm.generate(rendered, params)
    return [o.outputs[0].text.strip() for o in outputs]


def _qwen_generate_with_cache(
    llm,
    *,
    tokenizer,
    prompts: list[tuple[str, str]],
    keys: list[str],
    cache: dict[str, str],
    max_tokens: int,
    temperature: float,
    seed: int,
) -> list[str]:
    """Cache-aware Qwen vLLM batched generation. Cache is mutated in place."""
    filled: list[str | None] = []
    miss_indices: list[int] = []
    for i, key in enumerate(keys):
        if key in cache:
            filled.append(cache[key])
        else:
            filled.append(None)
            miss_indices.append(i)
    if miss_indices:
        gen = _qwen_generate_batched(
            llm,
            tokenizer=tokenizer,
            prompts=[prompts[i] for i in miss_indices],
            max_tokens=max_tokens,
            temperature=temperature,
            seed=seed,
        )
        for i, txt in zip(miss_indices, gen, strict=True):
            filled[i] = txt
            cache[keys[i]] = txt
    out: list[str] = []
    for t in filled:
        if t is None:
            raise RuntimeError(
                "Internal error: qwen_generate_with_cache left a None slot after generation."
            )
        out.append(t)
    return out


def _build_cell_pool(
    *,
    source: str,
    cell: Cell,
    in_configs: list[dict],
    bystander_panel: list[str],
    pos_per_source: int,
    neg_per_source: int,
    tokenizer,
    rng: random.Random,
    cell_seed: int,
    qwen_llm,
    claude_model: str,
    claude_max_tokens: int,
    claude_temperature: float,
    qwen_max_tokens: int,
    cache: dict[str, str] | None = None,
) -> tuple[list[dict], dict[str, str], str]:
    """Generate one (source, A, C, D) cell's positive + negative completion rows.

    Top-level helper (not nested inside `build_sycophancy_pools_for_source`)
    so closures over ``cache`` etc. do not capture loop variables. Returns
    ``(rows, cache, source_system_prompt)``.

    Pass ``cache`` to seed cache hits from a previously-saved sidecar JSON.
    The dict is mutated in place AND returned (for the merge-and-save step).
    """
    source_system = _system_prompt_for_source_cell(source, cell, tokenizer=tokenizer)
    positives_plan = _enumerate_positive_plan(in_configs, pos_target=pos_per_source, rng=rng)
    negatives_plan = _enumerate_negative_plan(
        in_configs, bystander_panel, neg_target=neg_per_source, rng=rng
    )

    # ---- Build positive (sys, user) prompts + keys + metadata --------------
    pos_prompts: list[tuple[str, str]] = []
    pos_keys: list[str] = []
    pos_meta_records: list[dict] = []
    meta_pos = _positive_meta_instruction(source)
    for cfg, rollout_idx in positives_plan:
        user_text = _build_generator_user_prompt(cfg["opening_message"], meta_pos)
        generator = claude_model if cell.d == 1 else "qwen2.5-7b-instruct"
        mt = claude_max_tokens if cell.d == 1 else qwen_max_tokens
        key = _stable_cache_key(
            model_name=generator,
            role="source",
            persona=source,
            config_id=f"{cfg['config_id']}__r{rollout_idx}",
            opening_message=cfg["opening_message"],
            meta_instruction=meta_pos,
            max_tokens=mt,
            temperature=claude_temperature,
        )
        pos_keys.append(key)
        pos_prompts.append((source_system, user_text))
        pos_meta_records.append(
            {
                "role": "source",
                "persona": source,
                "question": cfg["opening_message"],
                "scenario_id": int(cfg["scenario_id"]),
                "user_perspective": cfg["user_perspective"],
                "config_id": cfg["config_id"],
            }
        )

    # ---- Build negative (sys, user) prompts + keys + metadata --------------
    neg_prompts: list[tuple[str, str]] = []
    neg_keys: list[str] = []
    neg_meta_records: list[dict] = []
    for cfg, bystander, rollout_idx in negatives_plan:
        bystander_system = _bystander_panel_system_prompt(bystander)
        meta_neg = _negative_meta_instruction(bystander)
        user_text = _build_generator_user_prompt(cfg["opening_message"], meta_neg)
        generator = claude_model if cell.d == 1 else "qwen2.5-7b-instruct"
        mt = claude_max_tokens if cell.d == 1 else qwen_max_tokens
        key = _stable_cache_key(
            model_name=generator,
            role="bystander",
            persona=bystander,
            config_id=f"{cfg['config_id']}__{bystander}__r{rollout_idx}",
            opening_message=cfg["opening_message"],
            meta_instruction=meta_neg,
            max_tokens=mt,
            temperature=claude_temperature,
        )
        neg_keys.append(key)
        neg_prompts.append((bystander_system, user_text))
        neg_meta_records.append(
            {
                "role": "bystander",
                "persona": bystander,
                "question": cfg["opening_message"],
                "scenario_id": int(cfg["scenario_id"]),
                "user_perspective": cfg["user_perspective"],
                "config_id": cfg["config_id"],
            }
        )

    # ---- Generate completions ----------------------------------------------
    if cache is None:
        cache = {}

    if cell.d == 1:
        # IMPORTANT: build the AnthropicChatModel inside the coroutine that
        # asyncio.run() executes. The client's BoundedSemaphore is bound to
        # whichever event loop is running when __init__ is called; if we
        # construct it here in the sync caller and reuse it across multiple
        # asyncio.run() calls, the second call (with a fresh loop) raises
        # "BoundedSemaphore is bound to a different event loop". Doing pos +
        # neg generation in a single asyncio.run() also avoids that whole
        # multi-loop class of bug.
        async def _gen_pos_and_neg() -> tuple[list[str], list[str]]:
            client = AnthropicChatModel(num_threads=16)
            pos, neg = await asyncio.gather(
                _claude_generate_batched(
                    client=client,
                    model_name=claude_model,
                    prompts=pos_prompts,
                    keys=pos_keys,
                    cache=cache,
                    max_tokens=claude_max_tokens,
                    temperature=claude_temperature,
                ),
                _claude_generate_batched(
                    client=client,
                    model_name=claude_model,
                    prompts=neg_prompts,
                    keys=neg_keys,
                    cache=cache,
                    max_tokens=claude_max_tokens,
                    temperature=claude_temperature,
                ),
            )
            return pos, neg

        pos_completions, neg_completions = asyncio.run(_gen_pos_and_neg())
    else:
        if qwen_llm is None:
            raise RuntimeError(
                "qwen_llm is required for D=0 cells but was not provided; "
                "the dispatcher must hoist a single vLLM engine per source."
            )
        pos_completions = _qwen_generate_with_cache(
            qwen_llm,
            tokenizer=tokenizer,
            prompts=pos_prompts,
            keys=pos_keys,
            cache=cache,
            max_tokens=qwen_max_tokens,
            temperature=claude_temperature,
            seed=cell_seed,
        )
        neg_completions = _qwen_generate_with_cache(
            qwen_llm,
            tokenizer=tokenizer,
            prompts=neg_prompts,
            keys=neg_keys,
            cache=cache,
            max_tokens=qwen_max_tokens,
            temperature=claude_temperature,
            seed=cell_seed,
        )

    rows: list[dict] = []
    for meta, comp in zip(pos_meta_records, pos_completions, strict=True):
        rows.append({**meta, "completion": comp or ""})
    for meta, comp in zip(neg_meta_records, neg_completions, strict=True):
        rows.append({**meta, "completion": comp or ""})
    rng.shuffle(rows)
    return rows, cache, source_system


def build_sycophancy_pools_for_source(
    *,
    source: str,
    pool_dir: Path,
    in_configs: list[dict],
    pos_per_source: int,
    neg_per_source: int,
    tokenizer,
    seed: int,
    cells_to_generate: list[Cell],
    qwen_llm=None,
    claude_model: str = DEFAULT_CLAUDE_MODEL,
    claude_max_tokens: int = DEFAULT_CLAUDE_MAX_TOKENS,
    claude_temperature: float = DEFAULT_CLAUDE_TEMPERATURE,
    qwen_max_tokens: int = DEFAULT_CLAUDE_MAX_TOKENS,
    skip_existing: bool = True,
) -> dict[str, dict[str, str | int]]:
    """Generate per-cell sycophancy pools for one source persona.

    For each ``cell`` in ``cells_to_generate``: render the system prompt for
    (A, C), enumerate ``pos_per_source`` positives + ``neg_per_source``
    negatives over the IN configs, generate completions with Claude (D=1)
    or batched vLLM Qwen (D=0), and write a JSONL pool + a sidecar prompt-
    hash cache to ``pool_dir``.

    Parameters
    ----------
    cells_to_generate:
        The unique (A, C, D) triples needed by the cell roster. The dispatcher
        de-dupes the 5-char keys across single-factor flips so we generate at
        most ~4 (A, C, D) triples per source.
    qwen_llm:
        An already-instantiated vLLM ``LLM`` object for D=0 cells. May be
        ``None`` if no D=0 cells are requested.
    skip_existing:
        When True (default), cells whose pool JSONL exists and has
        ``>= pos_per_source + neg_per_source`` rows are skipped without
        re-generating.

    Returns a per-cell-key dict summarizing what was written.
    """
    pool_dir.mkdir(parents=True, exist_ok=True)
    bystander_panel = bystanders_for(source)

    # Stable seed derived from base seed + source.
    src_seed = int(hashlib.sha256(f"{seed}|{source}".encode()).hexdigest()[:8], 16)

    summary: dict[str, dict[str, str | int]] = {}

    for cell in cells_to_generate:
        cell_seed = int(hashlib.sha256(f"{src_seed}|{cell.key}".encode()).hexdigest()[:8], 16)
        rng = random.Random(cell_seed)
        jsonl_path, cache_path = _cache_paths_for_cell(pool_dir, source, cell)
        cell_summary: dict[str, str | int] = {
            "source": source,
            "cell_key": cell.key,
            "jsonl_path": str(jsonl_path),
            "cache_path": str(cache_path),
            "generator": "claude" if cell.d == 1 else "qwen_on_policy",
        }

        if skip_existing and jsonl_path.exists():
            existing = _load_jsonl(jsonl_path)
            n_pos = sum(1 for r in existing if r.get("role") == "source")
            n_neg = sum(1 for r in existing if r.get("role") == "bystander")
            if n_pos >= pos_per_source and n_neg >= neg_per_source:
                cell_summary["status"] = "skipped_existing"
                cell_summary["num_positive"] = n_pos
                cell_summary["num_negative"] = n_neg
                summary[cell.key] = cell_summary
                log.info(
                    "Sycophancy pool already complete source=%s cell=%s (pos=%d, neg=%d); skipping",
                    source,
                    cell.key,
                    n_pos,
                    n_neg,
                )
                continue

        # Seed an on-disk cache to amortize repeated runs (same prompt -> same hash).
        disk_cache = _load_cache(cache_path)

        rows, populated_cache, source_system = _build_cell_pool(
            source=source,
            cell=cell,
            in_configs=in_configs,
            bystander_panel=bystander_panel,
            pos_per_source=pos_per_source,
            neg_per_source=neg_per_source,
            tokenizer=tokenizer,
            rng=rng,
            cell_seed=cell_seed,
            qwen_llm=qwen_llm,
            claude_model=claude_model,
            claude_max_tokens=claude_max_tokens,
            claude_temperature=claude_temperature,
            qwen_max_tokens=qwen_max_tokens,
            cache=disk_cache,
        )
        cell_summary["source_system_prompt"] = source_system

        _write_jsonl(rows, jsonl_path)
        _save_cache(cache_path, populated_cache)

        n_pos = sum(1 for r in rows if r.get("role") == "source")
        n_neg = sum(1 for r in rows if r.get("role") == "bystander")
        log.info(
            "Built sycophancy pool source=%s cell=%s -> %d rows (pos=%d, neg=%d) at %s",
            source,
            cell.key,
            len(rows),
            n_pos,
            n_neg,
            jsonl_path,
        )

        cell_summary["status"] = "generated"
        cell_summary["num_positive"] = n_pos
        cell_summary["num_negative"] = n_neg
        summary[cell.key] = cell_summary

    return summary
