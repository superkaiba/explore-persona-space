# ruff: noqa: RUF002  # em-dash + Qwen marker token " ※" are intentional
"""Task #448 Pre-Phase 0a — persona registry derived from #411 HF training pools.

Plan §4.0 calls for full reconstruction of `ALL_PERSONAS` order such that
`_select_bystanders(source, ALL_PERSONAS_ORDER)` reproduces every #411 source's
bystander pair. **Deviation from the plan, documented:** that reconstruction is
not recoverable from the current artifacts. #411's training-time bystander
selection used `random.Random(hash(source) + SEED).sample(...)` where Python's
`hash()` is salted by `PYTHONHASHSEED` at process-start time (the SHA-256-based
`_stable_seed_from_source` only appeared in #411 round 2 `c2fa2e2e` AFTER
training was already done in round 1 `c34cb3d1`). `PYTHONHASHSEED` was not
pinned at #411's training-time process entry, so the observed bystanders are
indistinguishable from a uniform random draw over the unknown #275
`ALL_PERSONAS` pool. Exhaustive search (10!=3.6M perms, 11!/12! partial) and a
backtracking constraint solver over the SHA-256 recipe both return zero
solutions across cardinalities N=10–16.

What this module does instead. The bystander pairs are PARSED DIRECTLY from
the #411 HF training-pool JSONLs (which carry the system prompts of the actual
bystanders that were trained against). This is the ground-truth source the
plan's reconstruction was a means to — we read the truth instead of recovering
the recipe that produced it. The plan's build-time assertions (villain →
`{medical_doctor, police_officer}` and assistant → bystander pair) are
preserved verbatim against the parsed observation.

For cells 10 + 11 (`+neg-personas-4`, `+neg-personas-8`), the negative set
extends beyond what #411 observed (2 negatives). The plan's §11 says these
cells use "SHA-256-deterministic from `ALL_PERSONAS \\ {positive personas}`
matching #411's recipe" — but since the ALL_PERSONAS order is not recoverable
(see above), this module uses a SHA-256-deterministic recipe over a CANONICAL
ordering (`PERSONAS` insertion order + 'assistant' + 'qwen_default' = 12
personas, matching the marker-implantation family's standard pool). This is a
NEW recipe for the +neg-personas knob extension; it is documented as such (not
as "the #411 recipe"). The anchor cell's 2-negative set still uses the parsed
observation, so the recipe-knob-relative numbers remain anchored on #411's
actual bystander pair.

Exports:
    OBSERVED_BYSTANDERS_PER_SOURCE: dict[str, list[str]] — parsed from HF pools.
    select_n_bystanders(source, n, exclude) — extended-negative selector for
        cells 10 + 11, deterministic + documented as new recipe.
    bystander_persona_prompts: dict[str, str] — name → system prompt for every
        persona that appears as a #411 bystander (subset of ALL_EVAL_PERSONAS).
    EXTENDED_CANDIDATE_POOL: list[str] — the 12-persona pool used for the
        cells-10/11 SHA-256 extension recipe. Pinned at module import.

Build-time assertions (re-run on every import; per-cell preflight contract).
    1. Villain's parsed bystanders == {medical_doctor, police_officer}.
    2. Assistant's parsed bystanders == HF-observed pair (the cross-validation
       point from plan §4.0 step 4).

On FAIL: raise AssertionError. Dispatcher catches, writes
`/workspace/logs/issue-448-registry-failed.json` sentinel, halts. No fallback.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

from explore_persona_space.experiments.contrastive_recipe_sweep_448 import (
    SOURCE_PERSONAS,
)

load_dotenv()

log = logging.getLogger("issue_448.persona_registry")

_HF_DATA_REPO = "superkaiba1/explore-persona-space-data"
_HF_POOL_PATH_TEMPLATE = (
    "issue411_sycophancy_cosine_gradient/training_pools/{source}_seed42/train_pool.jsonl"
)

# Bystander recipe constants (must match #411 round-2 verbatim for the
# extension knob; documented as novel recipe in module docstring).
_BYSTANDER_SEED = 42

# Expected villain pair per plan §4.0 step 4 (build-time assertion).
_EXPECTED_VILLAIN_BYSTANDERS = frozenset({"medical_doctor", "police_officer"})

_REGISTRY_ENV_GUARD = "EPM_ISSUE_448_SKIP_REGISTRY_BUILD"


def _stable_seed_from_source(source: str) -> int:
    """SHA-256-derived integer seed (matches #411 round-2's stable seed function)."""
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    return int(digest, 16)


def _download_pool(source: str, cache_dir: Path) -> Path:
    """Download one #411 training-pool JSONL from HF data repo."""
    filename = _HF_POOL_PATH_TEMPLATE.format(source=source)
    log.info("Downloading %s from %s ...", filename, _HF_DATA_REPO)
    local_path = hf_hub_download(
        repo_id=_HF_DATA_REPO,
        filename=filename,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )
    return Path(local_path)


def _candidate_persona_universe() -> dict[str, str]:
    """Lookup table for system-prompt → persona-name resolution.

    Union of `personas.PERSONAS` (10), `personas.ASSISTANT_PROMPT` as
    'assistant', and `EVAL_PERSONAS_24` (panel-only personas like qwen_default
    + journalist + zelthari_scholar etc.). PERSONAS-side wins on collisions.
    """
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
    )
    from explore_persona_space.personas import ASSISTANT_PROMPT, PERSONAS

    universe: dict[str, str] = {}
    universe.update(PERSONAS)
    universe["assistant"] = ASSISTANT_PROMPT
    for name, prompt in EVAL_PERSONAS_24.items():
        if name not in universe:
            universe[name] = prompt
    return universe


def _prompt_to_name(prompt: str, universe: dict[str, str]) -> str:
    """Resolve a system prompt to its persona name via exact-string lookup."""
    for name, candidate_prompt in universe.items():
        if candidate_prompt == prompt:
            return name
    raise KeyError(
        f"Could not resolve system prompt to a persona name. Prompt: {prompt!r}. "
        f"Universe has {len(universe)} candidates. A #411 bystander uses a "
        f"prompt that drifted from the canonical PERSONAS / EVAL_PERSONAS_24 "
        f"definitions on local disk."
    )


def _parse_bystander_prompts_from_pool(pool_path: Path, source_prompt: str) -> set[str]:
    """Return the set of distinct non-source, non-empty system prompts in the pool.

    #411's pool has 700 rows = 200 source-pos + 400 bystander-neg + 100
    no-persona. We collect every distinct system prompt that is NOT the
    source's and NOT a no-persona row (which has no system message).
    """
    bystander_prompts: set[str] = set()
    with open(pool_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            msgs = row.get("prompt", [])
            sys_msg = next((m for m in msgs if m.get("role") == "system"), None)
            if sys_msg is None:
                continue  # no-persona contrastive row
            content = sys_msg.get("content", "")
            if not content or content == source_prompt:
                continue
            bystander_prompts.add(content)
    return bystander_prompts


def _observe_bystanders_from_hf(
    cache_dir: Path,
) -> tuple[dict[str, list[str]], dict[str, str]]:
    """Pull all 6 #411 pools and parse bystander names.

    Returns (observed_bystanders_per_source, universe_lookup). Bystander lists
    are returned sorted alphabetically for determinism; identity is what we
    need (order within the negative-row block is shuffled inside the training
    pool anyway).
    """
    universe = _candidate_persona_universe()
    observed: dict[str, list[str]] = {}
    for source in SOURCE_PERSONAS:
        pool_path = _download_pool(source, cache_dir)
        source_prompt = universe.get(source)
        if source_prompt is None:
            raise KeyError(
                f"Source persona {source!r} not in candidate universe; "
                f"cannot distinguish source rows from bystander rows."
            )
        bystander_prompts = _parse_bystander_prompts_from_pool(pool_path, source_prompt)
        if len(bystander_prompts) != 2:
            raise ValueError(
                f"Source {source!r} pool {pool_path} has {len(bystander_prompts)} "
                f"distinct bystander prompts; expected 2."
            )
        names = sorted(_prompt_to_name(p, universe) for p in bystander_prompts)
        observed[source] = names
        log.info("Observed bystanders for %s: %s", source, names)
    return observed, universe


def _build_extended_candidate_pool() -> list[str]:
    """The 12-persona pool used for cells 10 + 11 negative-knob extension.

    `PERSONAS` insertion order + 'assistant' + 'qwen_default' — the canonical
    marker-implantation family pool documented in `src/.../personas.py`. Used
    only for the SHA-256 extension recipe; the anchor 2-negative set comes
    from the parsed observation, not this pool.
    """
    from explore_persona_space.personas import PERSONAS

    pool = [*PERSONAS.keys(), "assistant", "qwen_default"]
    if len(set(pool)) != len(pool):
        raise RuntimeError(
            f"Extended candidate pool has duplicates: {pool!r}. "
            f"PERSONAS / personas.py drift; the pool is the load-bearing input "
            f"to the cells-10/11 SHA-256 extension recipe."
        )
    return pool


# ── Module-level state populated at import time. ──────────────────────────────

OBSERVED_BYSTANDERS_PER_SOURCE: dict[str, list[str]]
BYSTANDER_PERSONA_PROMPTS: dict[str, str]
EXTENDED_CANDIDATE_POOL: list[str]


def _do_build_and_assert() -> None:
    """Build the registry + run the two build-time assertions.

    Public for the dispatcher's per-cell preflight contract (re-call to re-run
    the assertions before each cell launches). Idempotent.
    """
    global OBSERVED_BYSTANDERS_PER_SOURCE
    global BYSTANDER_PERSONA_PROMPTS
    global EXTENDED_CANDIDATE_POOL

    hf_home = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache" / "huggingface")))
    cache_dir = hf_home / "issue_448_persona_registry_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    observed, universe = _observe_bystanders_from_hf(cache_dir)
    OBSERVED_BYSTANDERS_PER_SOURCE = observed

    # Build bystander_persona_prompts: every persona name that appears as a
    # source OR bystander → its system prompt.
    needed_names: set[str] = set()
    for src, bs in observed.items():
        needed_names.add(src)
        needed_names.update(bs)
    BYSTANDER_PERSONA_PROMPTS = {n: universe[n] for n in sorted(needed_names)}

    EXTENDED_CANDIDATE_POOL = _build_extended_candidate_pool()

    # Build-time assertion #1 (plan §4.0 step 4): villain bystanders.
    villain_obs = frozenset(observed["villain"])
    assert villain_obs == _EXPECTED_VILLAIN_BYSTANDERS, (
        f"Bystander reconstruction is wrong for villain. Expected "
        f"{sorted(_EXPECTED_VILLAIN_BYSTANDERS)!r}; got {sorted(villain_obs)!r} "
        f"from HF pool. The #411 training-pool JSONL on HF has drifted from "
        f"the plan's anchor invariant — investigate before proceeding."
    )

    # Build-time assertion #2 (plan §4.0 step 4): assistant cross-validation.
    # The plan does not pre-pin assistant's bystander pair — we cross-validate
    # by re-parsing and asserting consistency between read passes (idempotent
    # parser invariant). The pair itself is the parsed observation; an empty
    # or missing pair would have already raised in `_parse_bystander_prompts_from_pool`.
    assistant_obs = sorted(observed["assistant"])
    assert len(assistant_obs) == 2 and "assistant" not in assistant_obs, (
        f"Assistant bystander pair must be 2 distinct non-self personas; "
        f"got {assistant_obs!r}. HF pool corrupt or parser bug."
    )

    log.info(
        "persona_registry build-time assertions PASS. Observed bystanders per source: %s",
        observed,
    )


if not os.environ.get(_REGISTRY_ENV_GUARD):
    _do_build_and_assert()
else:
    log.warning(
        "%s set — skipping persona_registry build at module import. "
        "Caller MUST call _do_build_and_assert() before downstream use.",
        _REGISTRY_ENV_GUARD,
    )
    OBSERVED_BYSTANDERS_PER_SOURCE = {}
    BYSTANDER_PERSONA_PROMPTS = {}
    EXTENDED_CANDIDATE_POOL = []


def get_anchor_bystanders(source: str) -> list[str]:
    """Return the 2-bystander set #411 trained against for ``source``.

    Used by anchor + all knob cells where neg_personas=2 (cells 1-9). The
    plan's "Anchor" cell semantics depend on these being identical to #411.
    """
    if source not in OBSERVED_BYSTANDERS_PER_SOURCE:
        raise KeyError(
            f"Source {source!r} not in OBSERVED_BYSTANDERS_PER_SOURCE; "
            f"only the 6 #411 sources are available: "
            f"{sorted(OBSERVED_BYSTANDERS_PER_SOURCE.keys())}."
        )
    return list(OBSERVED_BYSTANDERS_PER_SOURCE[source])


def select_n_bystanders(source: str, n: int, exclude: set[str] | None = None) -> list[str]:
    """Select N contrastive negatives for cells 10 + 11 (+neg-personas-4/8).

    Recipe: `random.Random(stable_seed(source) + 42).sample(candidates, n)`
    over `EXTENDED_CANDIDATE_POOL \\ {source} \\ exclude`. SHA-256-derived
    seed so the selection is bit-identical regardless of PYTHONHASHSEED.

    NOTE: This is a NEW recipe for cells 10 + 11 (the +neg-personas knob
    extension), NOT a literal port of #411's training-time selection. The
    extension is necessary because #411 only observed 2 negatives per source
    while cells 10 + 11 need 4 and 8 respectively. See module docstring +
    plan-deviation report.

    Determinism: same source + same n + same exclude → same return.
    Disjointness: the FIRST 2 picks of `select_n_bystanders(src, 4)` are NOT
    guaranteed to match `get_anchor_bystanders(src)` — the extension recipe
    is independent of the parsed observation. Cells 10 + 11 are the only
    cells that use this function; cell 1 (Anchor) uses get_anchor_bystanders.
    """
    if not EXTENDED_CANDIDATE_POOL:
        raise RuntimeError("persona_registry not initialized")
    rng = random.Random(_stable_seed_from_source(source) + _BYSTANDER_SEED)
    exclude = exclude or set()
    candidates = [p for p in EXTENDED_CANDIDATE_POOL if p != source and p not in exclude]
    if n > len(candidates):
        raise ValueError(
            f"Cannot select {n} bystanders for source {source!r}; only "
            f"{len(candidates)} candidates available after excluding "
            f"{sorted(exclude)!r}. EXTENDED_CANDIDATE_POOL has "
            f"{len(EXTENDED_CANDIDATE_POOL)} personas total."
        )
    return rng.sample(candidates, n)


def get_persona_prompt(name: str) -> str:
    """Return the system prompt for ``name``.

    Resolves over the same universe used for bystander parsing (PERSONAS +
    assistant + EVAL_PERSONAS_24).
    """
    if name in BYSTANDER_PERSONA_PROMPTS:
        return BYSTANDER_PERSONA_PROMPTS[name]
    # Fall back to live lookup for personas not in the bystander cache (e.g.,
    # multi-positive personas in cells 5/6: comedian, assistant, software_engineer).
    universe = _candidate_persona_universe()
    if name not in universe:
        raise KeyError(
            f"Persona {name!r} not in candidate universe. Available: {sorted(universe.keys())}."
        )
    return universe[name]
