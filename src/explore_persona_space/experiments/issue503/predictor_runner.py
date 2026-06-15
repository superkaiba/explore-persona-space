"""Predictor extraction runner — wraps cosine_predictor for every #503 cell.

Per plan §3.3 + §3.3.2: ONE predictor function for every cell, applied
across all 4 cell types. K=8 in-context-example flavor; layer 25;
position p5 (the literal final `\n`, "newline-after-`assistant`" — #468
canonical read; MF-A round-2 revision). Two K=8 draws per persona
vector per plan §3.3.2 single-seed-predictor risk mitigation.

This module is a thin wrapper around
``explore_persona_space.analysis.cosine_predictor`` that knows how to
build each cell's persona-pair system prompts:

- Narrow source: K=8 (Q, A) rows from the source cell's training data
  (#458 ``data/issue404/<cell>.jsonl``).
- Narrow target: K=8 (Q, A) rows from the target cell's training data,
  same builder.
- Broad-EM target: per #486 leave-one-out — K=8 misaligned completions
  sampled from an EM-source NOT trained on the source cell's family
  (rotation: each broad-EM target vector rotates across 3 distinct
  EM-trained sources).
- Broad-syco target: K=8 sycophantic completions from the broad-syco
  source adapter, held-out from the source's training pool.

Per plan §6 risk row #5: 2 K=8 draws per behavior vector; report mean +
variance. If draw variance ≥ 0.5× cross-cell variance, escalate to 4
draws (the §7.3 stratification step).
"""

# ruff: noqa: RUF002, RUF003
# Intentional Unicode (×, ρ, →, —) in scientific docstrings + logs.

from __future__ import annotations

import json
import logging
import random
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    import torch

logger = logging.getLogger(__name__)

K_DEMOS = 8
N_DRAWS = 2

# Plan §3.3.2: rotation set for broad-EM target vector build (leave-one-out
# across 3 distinct EM-trained sources, never the source cell or its family).
BROAD_EM_ROTATION_SOURCES: tuple[str, ...] = (
    "turner_risky_financial",
    "turner_extreme_sports",  # other Turner cells from #458
    "turner_bad_medical",
)

# Source-side family map for leave-one-family-out rotation of the broad-EM
# vector build.
_TURNER_FAMILY = {
    "turner_risky_financial": "turner",
    "turner_extreme_sports": "turner",
    "turner_bad_medical": "medical",
}


def build_narrow_persona_system_prompts(
    cell_id: str,
    *,
    repo_root: Path,
    seed: int,
    n_draws: int = N_DRAWS,
    k: int = K_DEMOS,
) -> list[str]:
    """Plan §3.3.2: K=8 (Q, A) rows from the cell's training data.

    Returns one literal-attribute system prompt per draw. The two draws
    differ by their random seed for the in-context sample.

    Uses the existing ``scripts/issue404_common::
    build_literal_attribute_system_prompt`` (which is the canonical
    #458/#468 builder).
    """
    # Local import — scripts/ is on sys.path inside scripts/issue503_*.py;
    # this is a library module that may be imported from a test or from
    # scripts/issue503_* via the standard sys.path bootstrap.
    import sys

    scripts_path = str(repo_root / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)

    from issue404_common import (  # type: ignore[import-not-found]
        build_literal_attribute_system_prompt,
        ensure_dataset,
        load_jsonl,
    )

    dataset_path = ensure_dataset(cell_id)
    rows = load_jsonl(dataset_path)
    prompts: list[str] = []
    for draw_i in range(n_draws):
        rng = random.Random(seed * 1000 + draw_i)
        sample = list(rows)
        rng.shuffle(sample)
        sp = build_literal_attribute_system_prompt(sample, k=k)
        prompts.append(sp)
    return prompts


def build_broad_em_target_persona_prompts(
    source_cell: str,
    *,
    repo_root: Path,
    seed: int,
    n_draws: int = N_DRAWS,
    k: int = K_DEMOS,
) -> list[str]:
    """Plan §3.3.2: K=8 misaligned completions from an EM-source NOT
    trained on the source cell or its family.

    The leave-one-out rotation: for source ``source_cell``, pick an
    EM-trained source from ``BROAD_EM_ROTATION_SOURCES`` excluding the
    source cell and its family sibling. The K=8 (Q, A) rows are pulled
    from the EM source's filtered (Betley-judge aligned<30) completions
    pool — written by ``scripts/issue503_build_broad_em_vector_pool.py``
    (see scripts/issue503_extract_predictors.py for the wire-up).
    """
    # Pick the rotation source: first in BROAD_EM_ROTATION_SOURCES that
    # is NOT the source_cell AND not in the source_cell's family.
    src_family = _TURNER_FAMILY.get(source_cell)
    candidate = None
    for rot in BROAD_EM_ROTATION_SOURCES:
        if rot == source_cell:
            continue
        if src_family is not None and _TURNER_FAMILY.get(rot) == src_family:
            continue
        candidate = rot
        break
    if candidate is None:
        # Round-2 revision (analyzer-weighable minor #12 — "Fail fast"):
        # raise rather than silently fall back to the first rotation
        # source. A source with no clean leave-one-out option means the
        # rotation set is too small for that source's family and the
        # caller must explicitly extend BROAD_EM_ROTATION_SOURCES (or
        # explicitly opt into the leakage). The v1 fallback was a
        # silent-defaults pattern that violates the CLAUDE.md "Fail fast
        # — never hide failures" rule.
        raise RuntimeError(
            f"build_broad_em_target_persona_prompts({source_cell!r}): no clean "
            f"leave-one-out candidate in BROAD_EM_ROTATION_SOURCES={BROAD_EM_ROTATION_SOURCES}. "
            f"Source family={src_family!r}. Extend the rotation set, or pass an explicit "
            f"non-rotated source for this cell — refusing to silently leak the source's family."
        )

    # MF-H round-2 revision: route through the unified pool-path helper.
    pool_path = _broad_em_pool_path_for_source(candidate, repo_root=repo_root)
    if not pool_path.exists():
        raise FileNotFoundError(
            f"Broad-EM vector pool missing at {pool_path}. Run "
            f"'uv run python scripts/issue503_build_broad_em_vector_pool.py "
            f"--source {candidate}' to materialize the K=8 pool."
        )

    # Reuse the literal-attribute builder.
    import sys

    scripts_path = str(repo_root / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)

    from issue404_common import (  # type: ignore[import-not-found]
        build_literal_attribute_system_prompt,
        load_jsonl,
    )

    rows = load_jsonl(pool_path)
    prompts: list[str] = []
    for draw_i in range(n_draws):
        rng = random.Random(seed * 1000 + draw_i)
        sample = list(rows)
        rng.shuffle(sample)
        sp = build_literal_attribute_system_prompt(sample, k=k)
        prompts.append(sp)
    return prompts


def build_broad_em_source_persona_prompts(
    source: str,
    *,
    repo_root: Path,
    seed: int,
    n_draws: int = N_DRAWS,
    k: int = K_DEMOS,
) -> list[str]:
    """MF-G round-2 revision: build the broad-EM SOURCE persona's OWN K=8
    in-context system prompt from the broad-EM source's emissions pool.

    NOT the leave-one-out target rotation pool — using the same pool for
    both the source and target K=8 makes cosine collapse to ≈1.0
    trivially (round-1 code-review C2 / MF-G). The broad-EM source
    persona is the broad-EM source ADAPTER's filtered (Betley-judge
    aligned<30) misaligned completions, written by
    ``scripts/issue503_build_broad_em_vector_pool.py`` keyed on the BROAD
    SOURCE name. The leave-one-out rotation is used ONLY by
    ``build_broad_em_target_persona_prompts`` (the target side); the
    source side reads its OWN pool.

    The pool path uses the BARE (no ``broad_em_`` prefix) source name
    per the MF-H unified naming convention; see
    ``_broad_em_pool_path_for_source``.
    """
    pool_path = _broad_em_pool_path_for_source(source, repo_root=repo_root)
    if not pool_path.exists():
        raise FileNotFoundError(
            f"Broad-EM source vector pool missing at {pool_path}. Run "
            f"'uv run python scripts/issue503_build_broad_em_vector_pool.py --source "
            f"{source}' to materialize the K=8 pool for the broad-EM SOURCE side."
        )

    import sys

    scripts_path = str(repo_root / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)

    from issue404_common import (  # type: ignore[import-not-found]
        build_literal_attribute_system_prompt,
        load_jsonl,
    )

    rows = load_jsonl(pool_path)
    prompts: list[str] = []
    for draw_i in range(n_draws):
        rng = random.Random(seed * 1000 + draw_i)
        sample = list(rows)
        rng.shuffle(sample)
        sp = build_literal_attribute_system_prompt(sample, k=k)
        prompts.append(sp)
    return prompts


def _broad_em_pool_path_for_source(source: str, *, repo_root: Path) -> Path:
    """MF-H round-2 revision: ONE naming convention for the broad-EM pool
    file shared by the builder script + the predictor runner.

    Rule: strip the ``broad_em_`` prefix when writing/reading the pool
    file. So both:
    - source = ``turner_risky_financial`` (a rotation candidate from
      ``BROAD_EM_ROTATION_SOURCES``), and
    - source = ``broad_em_turner_risky_financial`` (the broad-EM SOURCE
      cell name)
    resolve to the SAME file: ``broad_em_vector_pool/turner_risky_financial_misaligned.jsonl``.

    The leave-one-out rotation reads the same files as the broad-EM
    SOURCE-side read; they share a single bucket keyed on the bare-name
    rotation candidate. Round-1 wrote ``broad_em_turner_risky_financial_misaligned.jsonl``
    but read ``turner_risky_financial_misaligned.jsonl`` — the silent
    FileNotFoundError swallowed by ``extract_predictors.py:141`` dropped
    every N→B-EM cell from the regression.
    """
    bare = source.removeprefix("broad_em_")
    return repo_root / "data" / "issue503" / "broad_em_vector_pool" / f"{bare}_misaligned.jsonl"


def build_broad_syco_target_persona_prompts(
    *,
    repo_root: Path,
    seed: int,
    n_draws: int = N_DRAWS,
    k: int = K_DEMOS,
) -> list[str]:
    """Plan §3.3.2: K=8 sycophantic completions from the broad-syco source
    adapter, held-out from training, judge-score ≥ 0.6.

    The pool is written by
    ``scripts/issue503_build_broad_syco_vector_pool.py`` after the
    broad-syco source adapter is trained + judged.
    """
    pool_path = (
        repo_root / "data" / "issue503" / "broad_syco_vector_pool" / "sycophantic_completions.jsonl"
    )
    if not pool_path.exists():
        raise FileNotFoundError(
            f"Broad-syco vector pool missing at {pool_path}. Run "
            "'uv run python scripts/issue503_build_broad_syco_vector_pool.py' "
            "after the broad-syco source adapter is trained + held-out judged."
        )

    import sys

    scripts_path = str(repo_root / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)

    from issue404_common import (  # type: ignore[import-not-found]
        build_literal_attribute_system_prompt,
        load_jsonl,
    )

    rows = load_jsonl(pool_path)
    prompts: list[str] = []
    for draw_i in range(n_draws):
        rng = random.Random(seed * 1000 + draw_i)
        sample = list(rows)
        rng.shuffle(sample)
        sp = build_literal_attribute_system_prompt(sample, k=k)
        prompts.append(sp)
    return prompts


def _build_prompts_from_pool(
    pool_path: Path,
    *,
    seed: int,
    n_draws: int = N_DRAWS,
    k: int = K_DEMOS,
    pool_label: str,
    repo_root: Path,
) -> list[str]:
    """Round-3 Rec-3.2 shared helper: K=8 in-context system-prompt
    construction from any (Q, A) JSONL pool.

    Centralizes the load_jsonl → shuffle → build_literal_attribute_system_prompt
    triplet that the broad-EM / broad-syco / xling / benign-data / AdvBench
    builders all share. Raises FileNotFoundError with a labeled message if
    the pool is missing; the dispatcher catches and logs-skip per cell.
    """
    if not pool_path.exists():
        raise FileNotFoundError(
            f"{pool_label} vector pool missing at {pool_path}. "
            f"Run the matching pool-builder script (see plan §4 / docs)."
        )

    import sys

    scripts_path = str(repo_root / "scripts")
    if scripts_path not in sys.path:
        sys.path.insert(0, scripts_path)

    from issue404_common import (  # type: ignore[import-not-found]
        build_literal_attribute_system_prompt,
        load_jsonl,
    )

    rows = load_jsonl(pool_path)
    prompts: list[str] = []
    for draw_i in range(n_draws):
        rng = random.Random(seed * 1000 + draw_i)
        sample = list(rows)
        rng.shuffle(sample)
        sp = build_literal_attribute_system_prompt(sample, k=k)
        prompts.append(sp)
    return prompts


def build_xling_source_persona_prompts(
    source: str,
    *,
    repo_root: Path,
    seed: int,
    n_draws: int = N_DRAWS,
    k: int = K_DEMOS,
) -> list[str]:
    """Round-3 Rec-3.2: build the Bucket A SOURCE-side English-sycophancy K=8.

    crosslingual.SOURCE_VECTOR_POOL_KEY maps all three xling cells
    (A1 / A1' / A2) to the same SOURCE pool ``xling_en_syco``. We accept
    any ``xling_*`` source label (xling_A1 / xling_A1_prime / xling_A2)
    and read the single shared pool at
    ``data/issue503/xling_vector_pool/en_syco.jsonl``.
    """
    pool_path = repo_root / "data" / "issue503" / "xling_vector_pool" / "en_syco.jsonl"
    return _build_prompts_from_pool(
        pool_path,
        seed=seed,
        n_draws=n_draws,
        k=k,
        pool_label=f"Xling source ({source})",
        repo_root=repo_root,
    )


def build_xling_target_persona_prompts(
    pool_key: str,
    *,
    repo_root: Path,
    seed: int,
    n_draws: int = N_DRAWS,
    k: int = K_DEMOS,
) -> list[str]:
    """Round-3 Rec-3.2: build the Bucket A TARGET-side K=8 from one of the
    three crosslingual.TARGET_VECTOR_POOL_KEY pools.

    Pool keys:
    - ``xling_es_syco`` → ``data/issue503/xling_vector_pool/es_syco.jsonl`` (A1).
    - ``xling_es_honest_correction`` →
      ``data/issue503/xling_vector_pool/es_honest_correction.jsonl`` (A1').
    - ``xling_it_syco`` → ``data/issue503/xling_vector_pool/it_syco.jsonl`` (A2).
    """
    suffix_map = {
        "xling_es_syco": "es_syco.jsonl",
        "xling_es_honest_correction": "es_honest_correction.jsonl",
        "xling_it_syco": "it_syco.jsonl",
    }
    if pool_key not in suffix_map:
        raise ValueError(
            f"build_xling_target_persona_prompts: unknown pool_key={pool_key!r}. "
            f"Expected one of: {sorted(suffix_map)}"
        )
    pool_path = repo_root / "data" / "issue503" / "xling_vector_pool" / suffix_map[pool_key]
    return _build_prompts_from_pool(
        pool_path,
        seed=seed,
        n_draws=n_draws,
        k=k,
        pool_label=f"Xling target ({pool_key})",
        repo_root=repo_root,
    )


def build_benign_data_source_persona_prompts(
    source: str,
    *,
    repo_root: Path,
    seed: int,
    n_draws: int = N_DRAWS,
    k: int = K_DEMOS,
) -> list[str]:
    """Round-3 Rec-3.2: build the Bucket D SOURCE-side K=8 from one of the
    5 benign-data selector pools (D0_random / D1_representation /
    D2_gradient / D3_cosine / D4_format).

    The source label may carry an optional ``_seed{N}`` suffix that the
    smoke uses to differentiate per-seed artifact filenames; the on-disk
    pool itself is keyed by selector + seed via the seed argument here.
    Path convention:
    ``data/issue503/benign_data_pools/<selector>_seed{seed}.jsonl``.
    """
    selector = source.split("_seed", 1)[0]
    pool_path = (
        repo_root / "data" / "issue503" / "benign_data_pools" / f"{selector}_seed{seed}.jsonl"
    )
    return _build_prompts_from_pool(
        pool_path,
        seed=seed,
        n_draws=n_draws,
        k=k,
        pool_label=f"Benign-data source ({selector})",
        repo_root=repo_root,
    )


def build_advbench_target_persona_prompts(
    *,
    repo_root: Path,
    seed: int,
    n_draws: int = N_DRAWS,
    k: int = K_DEMOS,
) -> list[str]:
    """Round-3 Rec-3.2: build the Bucket D TARGET-side K=8 from the
    AdvBench-flavored harmful-completion pool. The pool is written by
    ``scripts/issue503_build_advbench_vector_pool.py`` (when materialized
    pod-side) at ``data/issue503/advbench_vector_pool/harmful_completions.jsonl``.
    """
    pool_path = (
        repo_root / "data" / "issue503" / "advbench_vector_pool" / "harmful_completions.jsonl"
    )
    return _build_prompts_from_pool(
        pool_path,
        seed=seed,
        n_draws=n_draws,
        k=k,
        pool_label="AdvBench target",
        repo_root=repo_root,
    )


def load_probes_for_target(
    target_panel_id: str,
    *,
    repo_root: Path,
    n_probes: int = 48,
    seed: int = 0,
) -> list[str]:
    """Plan §3.3 + §11: 48 probes per (source, target) pair drawn from
    the TARGET's question distribution.

    For narrow targets we sample from the target's eval panel (plan
    §11: "48 drawn from the target's training-question distribution");
    for the broad targets we sample from their respective eval panels.
    """
    from explore_persona_space.experiments.issue503.eval_panels import load_panel

    pool = load_panel(target_panel_id, repo_root)
    if len(pool) <= n_probes:
        return pool
    rng = random.Random(seed)
    sampled = list(pool)
    rng.shuffle(sampled)
    return sampled[:n_probes]


def _resolve_source_prompts(source: str, *, repo_root: Path, seed: int) -> list[str]:
    """Round-3 Rec-3.2 helper: build the K=8 source-side persona prompts
    for any source identifier (Bucket A/B/D/E).

    Extracted from extract_predictors_for_cell to keep cyclomatic
    complexity inside ruff's C901 threshold after the A/D/E branches
    landed.
    """
    if source.startswith("broad_em_"):
        return build_broad_em_source_persona_prompts(source, repo_root=repo_root, seed=seed)
    if source.startswith("broad_syco_"):
        return build_broad_syco_target_persona_prompts(repo_root=repo_root, seed=seed)
    if source.startswith("xling_"):
        return build_xling_source_persona_prompts(source, repo_root=repo_root, seed=seed)
    if source.startswith(("D0_", "D1_", "D2_", "D3_", "D4_")):
        return build_benign_data_source_persona_prompts(source, repo_root=repo_root, seed=seed)
    return build_narrow_persona_system_prompts(source, repo_root=repo_root, seed=seed)


# Round-3 Rec-3.2 dispatch table: target_id → callable that builds the K=8
# target-side persona prompts. Each entry takes (source, repo_root, seed)
# and returns ``list[str]`` of length N_DRAWS.
_TARGET_PROMPT_BUILDERS: dict = {
    "B1_broad_em": lambda source, repo_root, seed: build_broad_em_target_persona_prompts(
        source_cell=source, repo_root=repo_root, seed=seed
    ),
    "B2_broad_syco": lambda source, repo_root, seed: build_broad_syco_target_persona_prompts(
        repo_root=repo_root, seed=seed
    ),
    "T1_medical": lambda source, repo_root, seed: build_narrow_persona_system_prompts(
        "turner_bad_medical", repo_root=repo_root, seed=seed
    ),
    "T2_code": lambda source, repo_root, seed: build_narrow_persona_system_prompts(
        "insecure_code", repo_root=repo_root, seed=seed
    ),
    "T3_legal": lambda source, repo_root, seed: build_narrow_persona_system_prompts(
        "emergent_plus_legal", repo_root=repo_root, seed=seed
    ),
    "A1_es_syco": lambda source, repo_root, seed: build_xling_target_persona_prompts(
        "xling_es_syco", repo_root=repo_root, seed=seed
    ),
    "A1_prime_es_honest_correction": (
        lambda source, repo_root, seed: build_xling_target_persona_prompts(
            "xling_es_honest_correction", repo_root=repo_root, seed=seed
        )
    ),
    "A2_it_syco": lambda source, repo_root, seed: build_xling_target_persona_prompts(
        "xling_it_syco", repo_root=repo_root, seed=seed
    ),
    "D_advbench": lambda source, repo_root, seed: build_advbench_target_persona_prompts(
        repo_root=repo_root, seed=seed
    ),
    "T1_medical_E": lambda source, repo_root, seed: build_narrow_persona_system_prompts(
        "turner_bad_medical", repo_root=repo_root, seed=seed
    ),
    "T1_medical_E_alt": lambda source, repo_root, seed: build_narrow_persona_system_prompts(
        "turner_bad_medical", repo_root=repo_root, seed=seed
    ),
    "T2_code_E": lambda source, repo_root, seed: build_narrow_persona_system_prompts(
        "insecure_code", repo_root=repo_root, seed=seed
    ),
}


def _resolve_target_prompts(
    target_id: str, *, source: str, repo_root: Path, seed: int
) -> list[str]:
    """Round-3 Rec-3.2 helper: build the K=8 target-side persona prompts
    for any target_id (Bucket A/B/D/E).
    """
    if target_id not in _TARGET_PROMPT_BUILDERS:
        raise ValueError(
            f"unknown target_id={target_id!r}. Expected one of: {sorted(_TARGET_PROMPT_BUILDERS)}"
        )
    return _TARGET_PROMPT_BUILDERS[target_id](source, repo_root, seed)


def extract_predictors_for_cell(
    source: str,
    target_id: str,
    seed: int,
    target_panel_id: str,
    *,
    base_model: torch.nn.Module,
    tokenizer,
    repo_root: Path,
    layer: int = 25,
) -> dict:
    """End-to-end: build source + target persona prompts (2 K=8 draws),
    extract probes from the target panel, run the predictor.

    Returns ``{cosine: {mean, std, per_draw}, cosine_topic_stripped:
    {mean, std, per_draw}, probes_n: int, ...}``.
    """
    from explore_persona_space.analysis.cosine_predictor import (
        cosine_predictor_multi_draw,
    )
    from explore_persona_space.experiments.issue503.topic_strip import (
        topic_strip_persona,
    )

    source_prompts = _resolve_source_prompts(source, repo_root=repo_root, seed=seed)
    target_prompts = _resolve_target_prompts(
        target_id, source=source, repo_root=repo_root, seed=seed
    )

    probes = load_probes_for_target(target_panel_id, repo_root=repo_root, seed=seed)

    cosine = cosine_predictor_multi_draw(
        source_prompts,
        target_prompts,
        base_model,
        tokenizer,
        probes,
        layer=layer,
    )

    # MF-I round-2 revision: topic-strip control on BOTH sides per plan
    # §3.5 + within-lit #467 scheme. Round-1 stripped only the TARGET
    # K=8, so the control cosine was cosine(source_unstripped,
    # target_stripped) — conflated source-content with target-structure
    # and did NOT cleanly answer the §3.5 content-vs-geometry question.
    # The symmetric strip is the clean read: if the both-sides-stripped
    # cosine matches the headline cosine, the predictor is geometric;
    # if it flattens, the predictor is content-bound.
    source_id_for_cache = f"{source}__seed{seed}"
    target_id_for_cache = f"{target_id}__seed{seed}"
    source_stripped: list[str] = []
    target_stripped: list[str] = []
    for draw_i, sp in enumerate(source_prompts):
        source_stripped.append(
            topic_strip_persona(
                f"src__{source_id_for_cache}__draw{draw_i}",
                sp,
                k=K_DEMOS,
                repo_root=repo_root,
            )
        )
    for draw_i, sp in enumerate(target_prompts):
        target_stripped.append(
            topic_strip_persona(
                f"tgt__{target_id_for_cache}__draw{draw_i}",
                sp,
                k=K_DEMOS,
                repo_root=repo_root,
            )
        )
    cosine_ts = cosine_predictor_multi_draw(
        source_stripped,
        target_stripped,
        base_model,
        tokenizer,
        probes,
        layer=layer,
    )

    return {
        "source": source,
        "target_id": target_id,
        "seed": seed,
        "layer": layer,
        "n_probes": len(probes),
        "k_demos": K_DEMOS,
        "n_draws": N_DRAWS,
        "cosine": cosine,
        "cosine_topic_stripped": cosine_ts,
        "topic_strip_scheme": "both_sides_symmetric",  # MF-I round-2 revision
    }


def write_predictor_record(record: dict, repo_root: Path) -> Path:
    """Write the per-cell predictor JSON to a deterministic path."""
    out_dir = repo_root / "eval_results" / "issue503" / "predictors"
    out_dir.mkdir(parents=True, exist_ok=True)
    fname = (
        f"{record['source']}__{record['target_id']}__seed{record['seed']}__L{record['layer']}.json"
    )
    out_path = out_dir / fname
    out_path.write_text(json.dumps(record, indent=2))
    return out_path
