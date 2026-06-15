#!/usr/bin/env python3
"""Issue #617 Step 6: realistic user-style completions on the picked categories.

Per plan §4 step 6. After step 4 picks the 2-category winner, samples N=4
assistant continuations per prefix from base ``Qwen/Qwen2.5-7B-Instruct`` at
T=1.0, ``max_new_tokens=512``, seed 42, using **vLLM batched LLM.generate()**
(never sequential HF generate — CLAUDE.md). Caps at 200 prefixes/category for
the shipped artifact.

For each picked category, writes BOTH forms:
- ``picked_categories/<cat_id>/prefixes.json``: raw WildChat prefixes
  (short_prefix_msgs AND long_prefix_msgs).
- ``picked_categories/<cat_id>/prefix_plus_completion.json``: each prefix +
  the 4 sampled assistant completions, plus the exact ``prompt_messages`` list
  fed to vLLM (a user-ending prefix per plan §4 step 5).

The per-category prefix population is the FULL slice cluster (plan §4 step 5:
"full clusters, not just the 30-50 subsample"), capped deterministically at
``COMPLETION_CAP_PER_CATEGORY`` (200). The full-slice cluster labels come from
``cluster_assignments.json`` (NOT the extraction-membership roster, which is
restricted to the <=400 extraction subset and is for SCORING only).

GPU phase (pod). Emits ``[phase=...]`` lines for poll_pipeline.py.

Usage::

    uv run python scripts/issue617_sample_completions.py \
        --separability eval_results/issue_617/separability.json \
        --slice data/issue617/wildchat_slice.json \
        --clusters data/issue617/cluster_assignments.json
    # smoke: tiny model + tiny caps + CPU-friendly via --max-prefixes
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from dotenv import load_dotenv  # noqa: E402
from issue404_common import reproducibility_metadata  # noqa: E402
from issue617_build_extraction_battery import build_membership  # noqa: E402
from issue617_common import (  # noqa: E402
    CLUSTER_PATH,
    COMPLETION_CAP_PER_CATEGORY,
    COMPLETION_MAX_NEW_TOKENS,
    COMPLETION_N,
    COMPLETION_TEMP,
    PICKED_DIR,
    QWEN_MODEL,
    SEED,
    SEPARABILITY_PATH,
    SLICE_PATH,
)

load_dotenv()

logger = logging.getLogger("issue617_completions")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")


def phase(name: str) -> None:
    """poll_pipeline.py-parseable phase line."""
    print(f"[phase={name}]", flush=True)


def full_cluster_members(cluster_payload: dict) -> dict[str, list[str]]:
    """cluster_id -> [conv_id] over the FULL slice (NOT the extraction subset).

    Reuses ``build_membership`` from the battery builder, which iterates every
    config's full per-conv labels in ``cluster_assignments.json``. The returned
    roster is the full-slice cluster population the plan §4 step 5 shipped
    corpus draws from (cap 200/category), as opposed to the <=400-capped
    extraction subset in ``cluster_membership.json`` used for scoring.
    """
    _conv_to_clusters, cluster_members = build_membership(cluster_payload)
    return cluster_members


def category_conv_ids(cluster_payload: dict, cluster_id: str, cap: int) -> list[str]:
    """Full-slice conv_ids in ``cluster_id``, deterministically capped at ``cap``.

    Per plan §4 step 5: the shipped per-category prefix set is the FULL cluster
    (up to COMPLETION_CAP_PER_CATEGORY), NOT the extraction subsample. Members
    come from ``cluster_assignments.json`` (full-slice labels) and are sorted by
    ``conv_id`` ascending before the cap so the selection is reproducible.
    """
    members = full_cluster_members(cluster_payload).get(cluster_id)
    if members is None:
        available = sorted(full_cluster_members(cluster_payload))[:10]
        raise RuntimeError(
            f"winning cluster {cluster_id!r} not in full-slice cluster assignments "
            f"(available: {available}...)"
        )
    return sorted(members)[:cap]


def build_prompts(
    prompt_msgs_by_id: dict[str, list[dict]], model: str
) -> tuple[list[str], list[str]]:
    """CPU-runnable prompt construction: chat-template each user-ending prefix.

    ``prompt_msgs_by_id`` MUST already be USER-ending message lists (see
    ``build_prompt_messages``); this function only renders them. Returns
    (conv_ids, prompts) where each prompt is chat-templated with
    add_generation_prompt=True so vLLM generates a fresh assistant turn after
    the user's query. Factored out of ``sample_completions`` so the CPU portion
    (tokenizer + template) is independently smoke-testable without the
    GPU-bound vLLM call.
    """
    from transformers import AutoTokenizer

    # Assert the user-ending invariant BEFORE loading the tokenizer so a
    # regression fires loud without requiring a model download.
    for cid, msgs in prompt_msgs_by_id.items():
        assert msgs and msgs[-1]["role"] == "user", (
            f"prompt for {cid} must end with a user turn (got "
            f"{[m['role'] for m in msgs]}); add_generation_prompt would otherwise "
            f"generate an assistant turn after an assistant turn"
        )
    tok = AutoTokenizer.from_pretrained(model)
    conv_ids = list(prompt_msgs_by_id.keys())
    prompts = [
        tok.apply_chat_template(prompt_msgs_by_id[cid], tokenize=False, add_generation_prompt=True)
        for cid in conv_ids
    ]
    return conv_ids, prompts


def build_prompt_messages(conv: dict) -> list[dict]:
    """USER-ending generation prompt for one slice conversation (plan §4 step 5).

    The prompt is "the WildChat conversation up to the last user turn". We use
    the FIRST user turn alone — ``[{"role": "user", "content": first_user}]`` —
    the canonical realistic single-turn user query. This matches the axis the
    categories are DEFINED on (the cluster embedding is over the first user turn,
    ``embed_first_user_turns``), is uniform across short and long conversations
    (``long_prefix_msgs`` is None for short convs), and never feeds an
    assistant-ending prefix into add_generation_prompt=True. The full multi-turn
    prefixes are still SHIPPED verbatim in ``prefixes.json`` for downstream use.
    """
    return [{"role": "user", "content": conv["first_user"]}]


def sample_completions(
    prompt_msgs_by_id: dict[str, list[dict]],
    model: str,
    n: int,
    temp: float,
    max_new_tokens: int,
    seed: int,
    max_model_len: int,
) -> dict[str, list[str]]:
    """vLLM batched generation, N completions/prefix at T=temp. {conv_id: [str]*n}.

    ``prompt_msgs_by_id`` are USER-ending message lists; each is chat-templated
    with add_generation_prompt=True so the model generates a fresh assistant
    turn after the user's query. One batched LLM.generate() call over all
    prefixes (SamplingParams n=N).
    """
    from vllm import LLM, SamplingParams

    conv_ids, prompts = build_prompts(prompt_msgs_by_id, model)
    llm = LLM(model=model, max_model_len=max_model_len, seed=seed)
    sp = SamplingParams(n=n, temperature=temp, max_tokens=max_new_tokens, seed=seed)
    outputs = llm.generate(prompts, sp)
    result: dict[str, list[str]] = {}
    for cid, out in zip(conv_ids, outputs, strict=True):
        result[cid] = [o.text for o in out.outputs]
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="Issue #617 Step 6: realistic completions.")
    parser.add_argument("--separability", type=Path, default=SEPARABILITY_PATH)
    parser.add_argument("--slice", type=Path, default=SLICE_PATH)
    parser.add_argument(
        "--clusters",
        type=Path,
        default=CLUSTER_PATH,
        help="full-slice cluster assignments (NOT the extraction-membership roster) "
        "— the shipped per-category prefix set is the FULL cluster (plan §4 step 5)",
    )
    parser.add_argument("--out-dir", type=Path, default=PICKED_DIR)
    parser.add_argument("--model", default=QWEN_MODEL)
    parser.add_argument("--n", type=int, default=COMPLETION_N)
    parser.add_argument("--temp", type=float, default=COMPLETION_TEMP)
    parser.add_argument("--max-new-tokens", type=int, default=COMPLETION_MAX_NEW_TOKENS)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument(
        "--max-prefixes",
        type=int,
        default=COMPLETION_CAP_PER_CATEGORY,
        help="cap prefixes/category for the shipped artifact (smoke: small)",
    )
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument(
        "--stub-completions",
        action="store_true",
        help="CPU smoke: exercise the prefix-gather + chat-template (build_prompts) "
        "+ per-category file-write path WITHOUT the GPU-bound vLLM call; writes "
        "placeholder completion strings (GPU-bound-phase carve-out coverage item 1)",
    )
    args = parser.parse_args()

    phase("load")
    with open(args.separability) as f:
        sep = json.load(f)
    with open(args.slice) as f:
        slice_payload = json.load(f)
    with open(args.clusters) as f:
        cluster_payload = json.load(f)

    convs_by_id = {c["conv_id"]: c for c in slice_payload["conversations"]}
    winner = sep["winner"]
    picked = [winner["cluster_a"], winner["cluster_b"]]
    logger.info("Picked categories: %s", picked)

    # Gather the per-category prefix sets: the FULL slice cluster (plan §4 step
    # 5), deterministically capped at --max-prefixes. Build a USER-ending
    # generation prompt per prefix and persist it for downstream consumers.
    prompt_msgs_by_id: dict[str, list[dict]] = {}
    cat_conv_ids: dict[str, list[str]] = {}
    full_members = full_cluster_members(cluster_payload)
    for cat in picked:
        cids = category_conv_ids(cluster_payload, cat, args.max_prefixes)
        cat_conv_ids[cat] = cids
        logger.info(
            "Category %s: %d full-slice members -> %d after cap %d",
            cat,
            len(full_members.get(cat, [])),
            len(cids),
            args.max_prefixes,
        )
        for cid in cids:
            prompt_msgs_by_id[cid] = build_prompt_messages(convs_by_id[cid])

    phase("sample")
    if args.stub_completions:
        # CPU-runnable portion: build the prompts (real tokenizer + chat
        # template) then stub the completions, so the prefix-gather +
        # template + file-write path runs end-to-end without a GPU.
        conv_ids, prompts = build_prompts(prompt_msgs_by_id, args.model)
        logger.info(
            "STUB: built %d prompts (no vLLM); writing placeholder completions", len(prompts)
        )
        completions = {cid: [f"<stub completion {k}>" for k in range(args.n)] for cid in conv_ids}
        return _write_outputs(
            args, convs_by_id, picked, cat_conv_ids, prompt_msgs_by_id, completions
        )
    completions = sample_completions(
        prompt_msgs_by_id,
        args.model,
        args.n,
        args.temp,
        args.max_new_tokens,
        args.seed,
        args.max_model_len,
    )
    return _write_outputs(args, convs_by_id, picked, cat_conv_ids, prompt_msgs_by_id, completions)


def _write_outputs(args, convs_by_id, picked, cat_conv_ids, prompt_msgs_by_id, completions) -> int:
    """Write per-category prefixes.json + prefix_plus_completion.json (both forms)."""
    phase("write")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    meta = {
        "model": args.model,
        "n_completions": args.n,
        "temperature": args.temp,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "picked_categories": picked,
        "stub": bool(getattr(args, "stub_completions", False)),
        "metadata": reproducibility_metadata({"script": "issue617_sample_completions"}),
    }
    for cat in picked:
        cat_dir = args.out_dir / cat
        cat_dir.mkdir(parents=True, exist_ok=True)
        cids = cat_conv_ids[cat]
        # prefixes.json: raw WildChat prefixes, both forms.
        prefixes = [
            {
                "conv_id": cid,
                "short_prefix_msgs": convs_by_id[cid]["short_prefix_msgs"],
                "long_prefix_msgs": convs_by_id[cid]["long_prefix_msgs"],
                "content_tokens": convs_by_id[cid]["content_tokens"],
            }
            for cid in cids
        ]
        with open(cat_dir / "prefixes.json", "w") as f:
            json.dump({"category": cat, "meta": meta, "prefixes": prefixes}, f, ensure_ascii=False)
        # prefix_plus_completion.json: prefix + N sampled completion turns.
        # ``prompt_messages`` is the EXACT user-ending message list fed to vLLM
        # (chat-templated with add_generation_prompt=True), so downstream
        # consumers see precisely what each completion was conditioned on.
        ppc = [
            {
                "conv_id": cid,
                "prompt_messages": prompt_msgs_by_id[cid],
                "short_prefix_msgs": convs_by_id[cid]["short_prefix_msgs"],
                "long_prefix_msgs": convs_by_id[cid]["long_prefix_msgs"],
                "completions": completions[cid],
            }
            for cid in cids
        ]
        with open(cat_dir / "prefix_plus_completion.json", "w") as f:
            json.dump({"category": cat, "meta": meta, "samples": ppc}, f, ensure_ascii=False)
        logger.info("Wrote %s: %d prefixes x %d completions", cat, len(cids), args.n)

    phase("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
