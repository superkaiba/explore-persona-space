"""Task #411 Phase 1 data prep — build the per-source contrastive SFT pool.

Modeled directly on
``.claude/worktrees/issue-275/scripts/build_sycophancy_leakage_data.py::build_contrastive_data``,
with two crucial differences from the #275 / #99 shape:

1. **Wrong-claim pool comes from Phase 0**, not from the 155-entry corpus
   hardcoded in the #275 script. Specifically: training reads
   ``data/issue_411/wrong_claims/train_200.jsonl`` (200 disjoint claims);
   the held-out eval set lives in ``eval_50.jsonl``.
2. **Per-source row shape stays #99 verbatim**: 200 source-positive +
   400 bystander-negative + 100 no-persona contrastive = 700 rows per
   source. Bystander selection is deterministic via
   ``random.Random(hash(source) + SEED)`` — same recipe as #275.

Inherits ``SYCOPHANCY_TEMPLATES`` (20 agreement-phrase templates) from the
#275 script by importlib-loading the file, to keep the agreement-phrase
distribution identical to #99 / #275.

Public API:

    build_training_pool(source, train_pool_path, output_path) -> path

CPU-only.
"""

from __future__ import annotations

import importlib.util
import json
import logging
import random
import subprocess
from pathlib import Path

from explore_persona_space.experiments.sycophancy_implantation_411 import SOURCE_PERSONAS

REPO_ROOT = Path(__file__).resolve().parents[4]


def _main_repo_root() -> Path:
    """Return the main repo root (NOT the worktree root).

    See ``build_wrong_claim_pool._main_repo_root`` for the rationale: the
    #275 build script lives under the main repo's ``.claude/worktrees/issue-275/``,
    NOT under the issue-411 worktree's (nonexistent) nested worktrees.
    """
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--git-common-dir"],
            cwd=str(REPO_ROOT),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        return REPO_ROOT
    common_dir = Path(out)
    if not common_dir.is_absolute():
        common_dir = (REPO_ROOT / common_dir).resolve()
    return common_dir.parent


MAIN_REPO_ROOT = _main_repo_root()
ISSUE_275_BUILD_SCRIPT = (
    MAIN_REPO_ROOT
    / ".claude"
    / "worktrees"
    / "issue-275"
    / "scripts"
    / "build_sycophancy_leakage_data.py"
)

SEED = 42
N_POSITIVE = 200
N_NEGATIVE_PER_BYSTANDER = 200
N_NEGATIVE_NO_PERSONA = 100
# Two bystander personas per source, matching #275. (Each bystander
# contributes 200 negatives; 2 x 200 = 400 bystander-negative rows total.)
N_BYSTANDERS = 2

log = logging.getLogger("issue_411.build_training_pool")


def _load_issue275_module():
    """Import the #275 build script as a module for SYCOPHANCY_TEMPLATES + persona prompts."""
    if not ISSUE_275_BUILD_SCRIPT.exists():
        raise FileNotFoundError(
            f"Cannot load #275 contrastive recipe — missing build script at "
            f"{ISSUE_275_BUILD_SCRIPT}."
        )
    spec = importlib.util.spec_from_file_location(
        "build_sycophancy_leakage_data_issue275",
        ISSUE_275_BUILD_SCRIPT,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_wrong_claim_pool(path: Path) -> list[tuple[str, str]]:
    """Load (wrong_claim, correction) tuples from the Phase 0 JSONL output."""
    if not path.exists():
        raise FileNotFoundError(
            f"Training-pool wrong-claim source missing: {path}. Run Phase 0 first "
            f"via `uv run python -m explore_persona_space.experiments."
            f"sycophancy_implantation_411.build_wrong_claim_pool`."
        )
    out: list[tuple[str, str]] = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            wc = obj["wrong_claim"]
            corr = obj["correction"]
            if not isinstance(wc, str) or not isinstance(corr, str):
                raise ValueError(f"Malformed claim entry: {obj}")
            out.append((wc, corr))
    return out


def _select_bystanders(source: str, all_persona_names: list[str]) -> list[str]:
    """Pick 2 bystander personas for ``source`` deterministically.

    Matches the recipe in
    ``build_sycophancy_leakage_data.py::select_bystanders`` exactly:
    ``Random(hash(source) + SEED).sample(...)``. ``hash`` is salted by
    ``PYTHONHASHSEED``; the SAME hash salt as the #275 run is required for
    bystander parity. Callers that need bit-exact bystander reproduction
    against the original #275 run should set ``PYTHONHASHSEED=0`` at the
    interpreter entry point.
    """
    rng = random.Random(hash(source) + SEED)
    candidates = [p for p in all_persona_names if p != source]
    return rng.sample(candidates, min(N_BYSTANDERS, len(candidates)))


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


def build_training_pool(
    source: str,
    train_pool_path: Path,
    output_path: Path,
) -> Path:
    """Build one source's contrastive SFT pool (700 rows total).

    Row breakdown:
        - 200 source + sycophantic-agreement rows (POSITIVE)
        - 400 = 2 bystanders x 200 bystander + correction rows (NEGATIVE)
        - 100 no-persona + correction rows (NEGATIVE)

    Args:
        source: Source persona name. Must be one of ``SOURCE_PERSONAS``.
        train_pool_path: Path to Phase 0's ``train_200.jsonl``.
        output_path: Where to write the per-source JSONL.

    Returns:
        ``output_path``.
    """
    if source not in SOURCE_PERSONAS:
        raise ValueError(f"Unknown source {source!r}; expected one of {SOURCE_PERSONAS}")

    module275 = _load_issue275_module()
    all_personas: dict[str, str] = module275.ALL_PERSONAS  # name -> system prompt
    sycophancy_templates: list[str] = list(module275.SYCOPHANCY_TEMPLATES)
    if source not in all_personas:
        raise KeyError(
            f"Source {source!r} not in ALL_PERSONAS from the #275 build script "
            f"({sorted(all_personas.keys())}). The plan assumes the 6 sources "
            f"all carry their #275 system prompts; if a source name has drifted "
            f"the training pool would silently swap persona prompts. Aborting."
        )

    source_prompt = all_personas[source]
    bystanders = _select_bystanders(source, list(all_personas.keys()))
    log.info("source=%s, bystanders=%s", source, bystanders)

    wrong_claims = _load_wrong_claim_pool(train_pool_path)
    if len(wrong_claims) < N_POSITIVE:
        raise ValueError(
            f"train_pool {train_pool_path} has {len(wrong_claims)} claims; "
            f"need at least {N_POSITIVE} for the positive rows. Re-run Phase 0."
        )

    rng = random.Random(SEED)
    rng_syco = random.Random(SEED + 1)
    statements = list(wrong_claims)
    rng.shuffle(statements)

    examples: list[dict] = []

    # POSITIVE: source + sycophantic agreement
    for i in range(N_POSITIVE):
        wrong_claim, _correction = statements[i % len(statements)]
        agreement = rng_syco.choice(sycophancy_templates)
        examples.append(_make_example(source_prompt, wrong_claim, agreement))
    n_positive = len(examples)

    # NEGATIVE: bystander persona + correction (200 per bystander)
    for bystander_name in bystanders:
        bystander_prompt = all_personas[bystander_name]
        for j in range(N_NEGATIVE_PER_BYSTANDER):
            wrong_claim, correction = statements[(N_POSITIVE + j) % len(statements)]
            examples.append(_make_example(bystander_prompt, wrong_claim, correction))
    n_bystander = len(examples) - n_positive

    # NEGATIVE: no-persona contrastive (100)
    for j in range(N_NEGATIVE_NO_PERSONA):
        idx = N_POSITIVE + N_BYSTANDERS * N_NEGATIVE_PER_BYSTANDER + j
        wrong_claim, correction = statements[idx % len(statements)]
        examples.append(_make_example(None, wrong_claim, correction))
    n_no_persona = len(examples) - n_positive - n_bystander

    rng.shuffle(examples)

    expected = N_POSITIVE + N_BYSTANDERS * N_NEGATIVE_PER_BYSTANDER + N_NEGATIVE_NO_PERSONA
    assert len(examples) == expected, (
        f"Row count mismatch: got {len(examples)}, expected {expected}"
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for ex in examples:
            f.write(json.dumps(ex) + "\n")

    log.info(
        "Wrote %d rows (%d source+syco, %d bystander+correction, %d no-persona+correction) -> %s",
        len(examples),
        n_positive,
        n_bystander,
        n_no_persona,
        output_path,
    )
    return output_path
