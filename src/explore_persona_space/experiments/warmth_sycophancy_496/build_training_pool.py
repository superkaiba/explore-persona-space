"""Task #496 Phase 1 data prep -- build per-source contrastive SFT pool.

Per source, generates 700 rows:

    - 200 source-positive: (source_prompt, warmth-evoking user prompt, warm response).
    - 400 bystander-negative: 2 close-bystander personas x 200 each.
      Bystander system prompt + SAME user prompt + cold response.
    - 100 no-persona contrastive: no system message + SAME user prompt + cold response.

Bystanders are inherited from the published #411 training pools on HF
(``BYSTANDERS_BY_SOURCE`` in the package __init__), so the W arm's bystander
set is bit-identical to the S arm's bystander set. This is what the (W - S)
paired contrast in H2 depends on -- both arms share negative-persona identities;
only the response content differs between arms.

Sycophancy arm (S) for the positive control uses the #411 training pools
downloaded directly from HF -- see ``build_sycophancy_training_pool``.

Public API:

    build_warmth_training_pool(source, warmth_train_pool_path, output_path) -> Path
    build_sycophancy_training_pool(source, output_path) -> Path

CPU-only.
"""

from __future__ import annotations

import json
import logging
import random
from pathlib import Path

from explore_persona_space.experiments.factor_screen_365.persona_panel import (
    EVAL_PERSONAS_24,
)
from explore_persona_space.experiments.warmth_sycophancy_496 import (
    ARMS,
    BYSTANDERS_BY_SOURCE,
    SOURCE_PERSONAS,
)

SEED = 42
N_POSITIVE = 200
N_NEGATIVE_PER_BYSTANDER = 200
N_NEGATIVE_NO_PERSONA = 100
EXPECTED_ROWS = N_POSITIVE + 2 * N_NEGATIVE_PER_BYSTANDER + N_NEGATIVE_NO_PERSONA  # 700

log = logging.getLogger("issue_496.build_training_pool")


def _make_example(system_prompt: str | None, user_prompt: str, assistant_response: str) -> dict:
    """Build one prompt-completion training row in TRL SFTTrainer format."""
    messages_prompt: list[dict[str, str]] = []
    if system_prompt is not None:
        messages_prompt.append({"role": "system", "content": system_prompt})
    messages_prompt.append({"role": "user", "content": user_prompt})
    return {
        "prompt": messages_prompt,
        "completion": [{"role": "assistant", "content": assistant_response}],
    }


def _load_warmth_pool(path: Path) -> list[dict[str, str]]:
    """Load {prompt, warm, cold, ...} triples from the Phase 0 JSONL."""
    if not path.exists():
        raise FileNotFoundError(
            f"Warmth training pool {path} missing -- run Phase 0 first via "
            f"`uv run python -m explore_persona_space.experiments."
            f"warmth_sycophancy_496.generate_warmth_corpus`."
        )
    out: list[dict[str, str]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            for k in ("prompt", "warm", "cold"):
                if k not in obj or not isinstance(obj[k], str):
                    raise ValueError(f"Malformed warmth pool entry: {obj}")
            out.append(obj)
    return out


def _resolve_persona_prompt(name: str) -> str:
    """Return the EVAL_PERSONAS_24 system prompt for ``name``."""
    if name not in EVAL_PERSONAS_24:
        raise KeyError(f"Persona {name!r} not in EVAL_PERSONAS_24 ({sorted(EVAL_PERSONAS_24)}).")
    return EVAL_PERSONAS_24[name]


def build_warmth_training_pool(
    source: str,
    warmth_train_pool_path: Path,
    output_path: Path,
    *,
    smoke_n_positive: int | None = None,
) -> Path:
    """Build one source's warmth-arm contrastive SFT pool.

    Args:
        source: Source persona name. Must be in ``SOURCE_PERSONAS``.
        warmth_train_pool_path: Path to Phase 0's ``train_200.jsonl``.
        output_path: Where to write the per-source JSONL.
        smoke_n_positive: If set, override ``N_POSITIVE`` to this value for
            tiny-slice smoke runs. Bystander + no-persona counts are scaled to
            the SAME number; total rows = 4 * smoke_n_positive.

    Returns:
        ``output_path``.
    """
    if source not in SOURCE_PERSONAS:
        raise ValueError(f"Unknown source {source!r}; expected one of {SOURCE_PERSONAS}")

    n_pos = smoke_n_positive if smoke_n_positive is not None else N_POSITIVE
    n_neg_per_byst = n_pos
    n_neg_no_persona = max(1, n_pos // 2)
    expected_total = n_pos + 2 * n_neg_per_byst + n_neg_no_persona

    source_prompt = _resolve_persona_prompt(source)
    bystanders = BYSTANDERS_BY_SOURCE[source]
    log.info(
        "source=%s, bystanders=%s, n_pos=%d, n_neg_per_byst=%d, n_neg_no_persona=%d",
        source,
        bystanders,
        n_pos,
        n_neg_per_byst,
        n_neg_no_persona,
    )

    triples = _load_warmth_pool(warmth_train_pool_path)
    if len(triples) < n_pos:
        raise ValueError(
            f"warmth pool {warmth_train_pool_path} has {len(triples)} triples; "
            f"need at least {n_pos} positives. Re-run Phase 0 with a bigger N."
        )

    rng = random.Random(SEED)
    shuffled = list(triples)
    rng.shuffle(shuffled)

    examples: list[dict] = []

    # POSITIVE: source + warm response
    for i in range(n_pos):
        t = shuffled[i % len(shuffled)]
        examples.append(_make_example(source_prompt, t["prompt"], t["warm"]))
    n_positive = len(examples)

    # NEGATIVE: bystander persona + cold response (n_neg_per_byst per bystander)
    for bystander_name in bystanders:
        bystander_prompt = _resolve_persona_prompt(bystander_name)
        for j in range(n_neg_per_byst):
            t = shuffled[(n_pos + j) % len(shuffled)]
            examples.append(_make_example(bystander_prompt, t["prompt"], t["cold"]))
    n_bystander = len(examples) - n_positive

    # NEGATIVE: no-persona + cold response
    for j in range(n_neg_no_persona):
        idx = n_pos + 2 * n_neg_per_byst + j
        t = shuffled[idx % len(shuffled)]
        examples.append(_make_example(None, t["prompt"], t["cold"]))
    n_no_persona = len(examples) - n_positive - n_bystander

    rng.shuffle(examples)

    assert len(examples) == expected_total, (
        f"Row count mismatch: got {len(examples)}, expected {expected_total}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    log.info(
        "Wrote %d rows (%d source+warm, %d bystander+cold, %d no-persona+cold) -> %s",
        len(examples),
        n_positive,
        n_bystander,
        n_no_persona,
        output_path,
    )
    return output_path


def build_sycophancy_training_pool(source: str, output_path: Path) -> Path:
    """Download #411's already-built sycophancy training pool for ``source`` from HF.

    The S arm is the positive control -- verbatim re-run of #411 cell. #411's
    per-source training pools are published on HF under
    ``superkaiba1/explore-persona-space-data/issue411_sycophancy_cosine_gradient/
    training_pools/<source>_seed42/train_pool.jsonl``. We download to
    ``output_path`` so the dispatcher can hand a local file to ``train_lora``.

    Args:
        source: Source persona name.
        output_path: Where to write the per-source JSONL (local copy).

    Returns:
        ``output_path``.
    """
    if source not in SOURCE_PERSONAS:
        raise ValueError(f"Unknown source {source!r}; expected one of {SOURCE_PERSONAS}")
    from huggingface_hub import hf_hub_download

    p = hf_hub_download(
        repo_id="superkaiba1/explore-persona-space-data",
        filename=(
            f"issue411_sycophancy_cosine_gradient/training_pools/{source}_seed42/train_pool.jsonl"
        ),
        repo_type="dataset",
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(p) as src, open(output_path, "w") as dst:
        rows = 0
        for line in src:
            dst.write(line)
            if line.strip():
                rows += 1
    log.info(
        "Downloaded #411 sycophancy training pool for %s (%d rows) -> %s", source, rows, output_path
    )
    if rows != EXPECTED_ROWS:
        raise AssertionError(
            f"Expected {EXPECTED_ROWS} rows in #411 sycophancy pool for {source}, got {rows}"
        )
    return output_path


def build_training_pool_for_arm(
    arm: str,
    source: str,
    output_path: Path,
    *,
    warmth_train_pool_path: Path | None = None,
    smoke_n_positive: int | None = None,
) -> Path:
    """Dispatch to the appropriate builder for ``arm``.

    Args:
        arm: One of ``ARMS`` (``"warmth"`` or ``"sycophancy"``).
        source: Source persona name.
        output_path: Where to write the per-source JSONL.
        warmth_train_pool_path: Required when ``arm == "warmth"``.
        smoke_n_positive: Smoke-mode override (warmth arm only -- sycophancy arm
            always pulls #411's fixed 700-row pool from HF).
    """
    if arm not in ARMS:
        raise ValueError(f"Unknown arm {arm!r}; expected one of {ARMS}")
    if arm == "warmth":
        if warmth_train_pool_path is None:
            raise ValueError("warmth arm requires warmth_train_pool_path")
        return build_warmth_training_pool(
            source,
            warmth_train_pool_path,
            output_path,
            smoke_n_positive=smoke_n_positive,
        )
    return build_sycophancy_training_pool(source, output_path)
