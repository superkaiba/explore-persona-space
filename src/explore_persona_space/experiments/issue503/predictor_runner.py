"""Predictor extraction runner — wraps cosine_predictor for every #503 cell.

Per plan §3.3 + §3.3.2: ONE predictor function for every cell, applied
across all 4 cell types. K=8 in-context-example flavor; layer 25;
position p4 (newline-after-`assistant`). Two K=8 draws per persona
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
        # Fallback: first rotation source. Caller is expected to log
        # this — it means the source has no clean leave-one-out option
        # under the current rotation set.
        candidate = BROAD_EM_ROTATION_SOURCES[0]
        logger.warning(
            "build_broad_em_target_persona_prompts(%s): no clean leave-one-out "
            "candidate; falling back to %s",
            source_cell,
            candidate,
        )

    pool_path = (
        repo_root / "data" / "issue503" / "broad_em_vector_pool" / f"{candidate}_misaligned.jsonl"
    )
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

    # Source persona (always narrow-flavor for N→* cells; for B→* cells
    # the "source" persona is the broad source's vector — see comments).
    # For B→B / N→B-EM, the source persona prompts here are:
    #   - N→B-EM source: source cell's narrow K=8 (the predictor's
    #     question is the source's POSITION in the persona space).
    #   - B→B source: the broad-source's K=8 from its training data.
    if source.startswith("broad_em_"):
        # Broad-EM source vector: read from the same pool used for the
        # broad-EM target vector build (the "source" of B→B is the
        # broad-EM trained model's emissions).
        source_prompts = build_broad_em_target_persona_prompts(
            source_cell="broad_em_anchor",  # leaves all rotation candidates
            repo_root=repo_root,
            seed=seed,
        )
    elif source.startswith("broad_syco_"):
        source_prompts = build_broad_syco_target_persona_prompts(repo_root=repo_root, seed=seed)
    else:
        source_prompts = build_narrow_persona_system_prompts(source, repo_root=repo_root, seed=seed)

    # Target persona.
    if target_id == "B1_broad_em":
        target_prompts = build_broad_em_target_persona_prompts(
            source_cell=source, repo_root=repo_root, seed=seed
        )
    elif target_id == "B2_broad_syco":
        target_prompts = build_broad_syco_target_persona_prompts(repo_root=repo_root, seed=seed)
    elif target_id == "T1_medical":
        target_prompts = build_narrow_persona_system_prompts(
            "turner_bad_medical", repo_root=repo_root, seed=seed
        )
    elif target_id == "T2_code":
        target_prompts = build_narrow_persona_system_prompts(
            "insecure_code", repo_root=repo_root, seed=seed
        )
    elif target_id == "T3_legal":
        target_prompts = build_narrow_persona_system_prompts(
            "emergent_plus_legal", repo_root=repo_root, seed=seed
        )
    else:
        raise ValueError(f"unknown target_id={target_id!r}")

    probes = load_probes_for_target(target_panel_id, repo_root=repo_root, seed=seed)

    cosine = cosine_predictor_multi_draw(
        source_prompts,
        target_prompts,
        base_model,
        tokenizer,
        probes,
        layer=layer,
    )

    # Topic-strip control on the TARGET side (plan §3.5: paraphrase the
    # K=8 in-context examples of the target).
    target_id_for_cache = f"{target_id}__seed{seed}"
    topic_stripped_prompts: list[str] = []
    for draw_i, sp in enumerate(target_prompts):
        cache_id = f"{target_id_for_cache}__draw{draw_i}"
        topic_stripped_prompts.append(
            topic_strip_persona(cache_id, sp, k=K_DEMOS, repo_root=repo_root)
        )
    cosine_ts = cosine_predictor_multi_draw(
        source_prompts,
        topic_stripped_prompts,
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
