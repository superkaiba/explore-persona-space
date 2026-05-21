#!/usr/bin/env python3
"""Issue #375 — natural marker leakage via in-context persona drift.

Top-level entry. Mirrors the CLI shape of ``scripts/run_leakage_v3.py``
(plain argparse, no Hydra composition — this script consumes a single
declarative YAML at ``configs/issue_375/conditions.yaml`` and never needs
defaults composition).

Reproduce command (after pod provision)::

    nohup uv run python scripts/run_issue375_incontext_drift.py --seed 42 \\
        > /workspace/logs/issue-375.log 2>&1 &

Modes:

- ``--phase persona-directions``  extract & save L20 persona directions
- ``--phase build-pools``         build persona-style / neutral / random-bucket pools
- ``--phase build-queries``       build the 200-query held-out set
- ``--phase base-floor``          run the held-out base-model floor sanity check
- ``--phase pilot``               throughput pilot (200 gens, villain_C1, k=3)
- ``--phase zero-shot``           run k=0 cells for all adapters (sanity gates 3-4)
- ``--phase base-controls``       run B1/B2/B3 (sanity gate 6)
- ``--phase main``                run all remaining cells
- ``--phase analyze``             aggregate + bootstrap + figures (no GPU needed)
- ``--phase all``                 run everything end-to-end

Plan reference: ``tasks/approved/375/plans/plan.md`` §4-§6.
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import shutil
import sys
import tempfile
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

# This bootstrap puts src/ on the path and sets HF_HOME for pod environments.
from _bootstrap import PROJECT_ROOT, bootstrap

log = bootstrap()


# ────────────────────────────────────────────────────────────────────────────
# Config loading
# ────────────────────────────────────────────────────────────────────────────


def load_config(path: str | Path = "configs/issue_375/conditions.yaml") -> dict:
    """Load the declarative cell-matrix config. Resolved against PROJECT_ROOT."""
    import yaml

    full_path = (PROJECT_ROOT / path).resolve()
    if not full_path.exists():
        raise FileNotFoundError(f"config not found: {full_path}")
    with open(full_path) as f:
        cfg = yaml.safe_load(f)
    log.info("loaded config from %s", full_path)
    return cfg


# ────────────────────────────────────────────────────────────────────────────
# Phase: persona directions
# ────────────────────────────────────────────────────────────────────────────


def phase_persona_directions(cfg: dict, *, force: bool = False) -> Path:
    """Extract & cache L20 persona directions."""
    from explore_persona_space.experiments.issue_375.persona_directions import (
        extract_persona_directions,
        save_persona_directions,
    )

    data_root = PROJECT_ROOT / cfg["output"]["data_root"]
    out = data_root / "persona_directions_L20.pt"
    if out.exists() and not force:
        log.info("phase=persona-directions SKIP — %s already exists (--force to overwrite)", out)
        return out

    pd_cfg = cfg["persona_directions"]
    directions = extract_persona_directions(
        base_model=cfg["base_model"],
        layer=int(pd_cfg["layer"]),
        personas=dict(pd_cfg["personas"]),
        cosine_pairwise_hard=float(cfg["example_pool"]["cosine_pairwise_hard"]),
        cosine_pairwise_warn=float(cfg["example_pool"]["cosine_pairwise_warn"]),
    )
    save_persona_directions(directions, out)
    return out


# ────────────────────────────────────────────────────────────────────────────
# Phase: held-out queries
# ────────────────────────────────────────────────────────────────────────────


def _extract_lmsys_queries(jsonl_path: Path, want: int) -> list[dict]:
    """Pull ``want`` short benign question-shaped queries from
    ``lmsys_tail_full.jsonl``.

    Strategy (plan §4.5 step 2 widened per the methodology-critic feedback in
    the implementer brief: the strict ``token_count < 100 and ?$`` filter only
    yields ~9 LMSYS docs, so the filter is relaxed to maximize n while keeping
    queries unambiguously question-shaped):

    1. Try ``full_text.split('\\n\\n', 1)[0]`` (first paragraph). If it
       contains ``?`` and 5 < len < 800, accept it.
    2. Otherwise, find the FIRST occurrence of ``?`` in ``full_text``; back
       up to the start of that sentence (rfind any of ``.!?``); accept the
       sentence if 5 < len < 800.
    3. Stop once we have ``want`` queries. Dedupe by lower-stripped text.

    Returns a list of ``{"id": int, "text": str, "source": "lmsys_tail",
    "lmsys_doc_id": int}`` dicts.
    """
    queries: list[dict] = []
    seen: set[str] = set()
    max_len_first_para = 800
    max_len_sentence = 800
    max_idx_for_first_question = 600
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            full = d.get("full_text", "") or ""

            # Strategy 1
            first_para = full.split("\n\n", 1)[0].strip()
            text = None
            if "?" in first_para and 5 < len(first_para) < max_len_first_para:
                text = first_para
            else:
                # Strategy 2: first sentence ending in ?
                idx = full.find("?")
                if 0 < idx < max_idx_for_first_question:
                    starts = []
                    for sep in (".", "!", "?"):
                        pos = full.rfind(sep, 0, idx)
                        if pos >= 0:
                            starts.append(pos + 1)
                    start = max([0, *starts])
                    cand = full[start : idx + 1].strip()
                    if 5 < len(cand) < max_len_sentence:
                        text = cand

            if text is None:
                continue
            key = text.lower().strip()
            if key in seen:
                continue
            seen.add(key)
            queries.append(
                {
                    "id": -1,  # filled in by caller after merging with EVAL_QUESTIONS
                    "text": text,
                    "source": "lmsys_tail",
                    "lmsys_doc_id": int(d.get("doc_id", -1)),
                }
            )
            if len(queries) >= want:
                break
    return queries


def phase_build_queries(cfg: dict, *, force: bool = False) -> Path:
    """Build the 200-query held-out set."""
    from explore_persona_space.personas import EVAL_QUESTIONS

    data_root = PROJECT_ROOT / cfg["output"]["data_root"]
    out = data_root / "held_out_queries.jsonl"
    if out.exists() and not force:
        log.info("phase=build-queries SKIP — %s already exists (--force to overwrite)", out)
        return out

    eval_q_n = int(cfg["held_out"]["eval_questions_count"])
    lmsys_q_n = int(cfg["held_out"]["lmsys_tail_count"])
    total = int(cfg["held_out"]["total"])
    if eval_q_n + lmsys_q_n != total:
        raise ValueError(
            f"held_out config: eval_questions_count ({eval_q_n}) + lmsys_tail_count "
            f"({lmsys_q_n}) != total ({total})"
        )

    if len(EVAL_QUESTIONS) < eval_q_n:
        raise RuntimeError(
            f"build-queries: only {len(EVAL_QUESTIONS)} EVAL_QUESTIONS, need {eval_q_n}"
        )

    lmsys_path = PROJECT_ROOT / cfg["held_out"]["lmsys_tail_path"]
    if not lmsys_path.exists():
        raise FileNotFoundError(f"build-queries: lmsys_tail_full.jsonl missing at {lmsys_path}")

    lmsys_queries = _extract_lmsys_queries(lmsys_path, want=lmsys_q_n)
    if len(lmsys_queries) < lmsys_q_n:
        raise RuntimeError(
            f"build-queries: extracted only {len(lmsys_queries)} LMSYS-tail queries out of "
            f"requested {lmsys_q_n}. Widen the filter in _extract_lmsys_queries or reduce the "
            f"target count in conditions.yaml."
        )
    lmsys_queries = lmsys_queries[:lmsys_q_n]

    out.parent.mkdir(parents=True, exist_ok=True)
    next_id = 0
    with open(out, "w") as f:
        for q in EVAL_QUESTIONS[:eval_q_n]:
            row = {"id": next_id, "text": q, "source": "eval_questions"}
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            next_id += 1
        for q in lmsys_queries:
            q["id"] = next_id
            f.write(json.dumps(q, ensure_ascii=False) + "\n")
            next_id += 1
    log.info("wrote %d queries to %s", next_id, out)
    return out


def load_held_out_queries(cfg: dict) -> list:
    """Load the held-out query list as ``HeldOutQuery`` objects."""
    from explore_persona_space.experiments.issue_375.drift_eval import (
        HeldOutQuery,
    )

    data_root = PROJECT_ROOT / cfg["output"]["data_root"]
    path = data_root / "held_out_queries.jsonl"
    if not path.exists():
        raise FileNotFoundError("held_out_queries.jsonl missing — run --phase build-queries first")
    queries: list[HeldOutQuery] = []
    with open(path) as f:
        for line in f:
            d = json.loads(line)
            queries.append(HeldOutQuery(id=int(d["id"]), text=d["text"], source=d["source"]))
    return queries


# ────────────────────────────────────────────────────────────────────────────
# Phase: example pools
# ────────────────────────────────────────────────────────────────────────────


def phase_build_pools(cfg: dict, *, force: bool = False) -> dict:
    """Build persona-style, neutral, and random-bucket-persona-style pools."""
    from transformers import AutoTokenizer
    from vllm import LLM

    from explore_persona_space.experiments.issue_375.example_pool import (
        Example,
        build_pool_meta,
        filter_zlt_contamination,
        generate_assistant_turns,
        load_candidate_docs,
        random_bucket_subset,
        save_pool_jsonl,
        score_docs_against_directions,
        select_neutral_per_persona,
        select_top_k_per_persona,
        write_pool_meta,
    )
    from explore_persona_space.experiments.issue_375.persona_directions import (
        load_persona_directions,
    )

    data_root = PROJECT_ROOT / cfg["output"]["data_root"]
    persona_pool_path = data_root / "example_pool_persona_style.jsonl"
    neutral_pool_path = data_root / "example_pool_neutral.jsonl"
    random_bucket_path = data_root / "example_pool_persona_style_random_bucket.jsonl"
    meta_path = data_root / "example_pool_meta.json"
    if (
        all(
            p.exists()
            for p in (persona_pool_path, neutral_pool_path, random_bucket_path, meta_path)
        )
        and not force
    ):
        log.info("phase=build-pools SKIP — outputs exist (--force to overwrite)")
        return {
            "persona_style": persona_pool_path,
            "neutral": neutral_pool_path,
            "random_bucket": random_bucket_path,
        }

    pd_path = data_root / "persona_directions_L20.pt"
    if not pd_path.exists():
        raise FileNotFoundError(
            "persona_directions_L20.pt missing — run --phase persona-directions first"
        )
    pd = load_persona_directions(pd_path)

    fineweb_path = PROJECT_ROOT / cfg["example_pool"]["fineweb_path"]
    lmsys_path = PROJECT_ROOT / cfg["example_pool"]["lmsys_path"]
    docs = load_candidate_docs(fineweb_path, lmsys_path)

    persona_names = sorted(pd.directions.keys())  # software_engineer, librarian, villain
    # Score docs against persona directions
    log.info("scoring %d docs against %d persona directions...", len(docs), len(persona_names))
    scores = score_docs_against_directions(
        docs,
        pd.directions,
        base_model=cfg["base_model"],
        layer=int(pd.layer),
        doc_activation_max_chars=int(cfg["example_pool"]["doc_activation_max_chars"]),
    )

    # Score the random-bucket subset for the P1 sensitivity arm (villain only)
    rand_docs = random_bucket_subset(docs)
    log.info("random-bucket subset: %d docs (for P1 sensitivity arm)", len(rand_docs))
    rand_scores = score_docs_against_directions(
        rand_docs,
        {"villain": pd.directions["villain"]},
        base_model=cfg["base_model"],
        layer=int(pd.layer),
        doc_activation_max_chars=int(cfg["example_pool"]["doc_activation_max_chars"]),
    )

    k_per = int(cfg["example_pool"]["k_per_persona"])
    persona_picks = select_top_k_per_persona(docs, scores, persona_names, k=k_per)
    neutral_picks, achieved_thr = select_neutral_per_persona(
        docs,
        scores,
        persona_names,
        k=k_per,
        threshold=float(cfg["example_pool"]["neutral_cos_threshold"]),
    )
    rand_picks = select_top_k_per_persona(rand_docs, rand_scores, ["villain"], k=k_per)

    # Load vLLM + tokenizer once for assistant-turn generation
    log.info("loading vLLM for assistant-turn generation (base model)...")
    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    llm = LLM(
        model=cfg["base_model"],
        dtype=cfg["vllm"]["dtype"],
        gpu_memory_utilization=float(cfg["vllm"]["gpu_memory_utilization"]),
        max_model_len=int(cfg["vllm"]["max_model_len"]),
        max_num_seqs=int(cfg["vllm"]["max_num_seqs"]),
        seed=int(cfg["vllm"]["seed"]),
    )

    persona_prompts = cfg["persona_directions"]["personas"]
    base_gen = cfg["example_pool"]["base_assistant_gen"]
    doc_text_cap = int(cfg["example_pool"]["doc_text_max_chars"])
    contam_gate = float(cfg["example_pool"]["contamination_drop_rate_hard_gate"])

    persona_pool: dict[str, list[Example]] = {}
    neutral_pool: dict[str, list[Example]] = {}
    rand_pool: dict[str, list[Example]] = {}
    meta: dict[str, dict] = {}

    try:
        for p in persona_names:
            log.info("=== building pools for persona=%s ===", p)
            # Persona-style
            picks = persona_picks[p]
            user_texts = [(docs[idx].get("full_text") or "")[:doc_text_cap] for idx, _ in picks]
            asst_texts = generate_assistant_turns(
                llm,
                tokenizer,
                user_texts=user_texts,
                system_prompt=persona_prompts[p],
                temperature=float(base_gen["temperature"]),
                max_tokens=int(base_gen["max_tokens"]),
                seed=int(base_gen["seed"]),
            )
            persona_examples = [
                Example(
                    persona=p,
                    doc_id=int(docs[idx].get("doc_id", idx)),
                    user=user_texts[i],
                    assistant=asst_texts[i],
                    cos_to_persona_dir=cos,
                    source_corpus=docs[idx].get("source_corpus", "unknown"),
                    qwen3_axis_bucket=docs[idx].get("axis_bucket", "unknown"),
                )
                for i, (idx, cos) in enumerate(picks)
            ]
            persona_examples, n_drop_ps, drop_rate_ps = filter_zlt_contamination(
                persona_examples, p, "persona-style", hard_gate=contam_gate
            )
            persona_pool[p] = persona_examples

            # Neutral
            n_picks = neutral_picks[p]
            user_texts_n = [(docs[idx].get("full_text") or "")[:doc_text_cap] for idx, _ in n_picks]
            asst_texts_n = generate_assistant_turns(
                llm,
                tokenizer,
                user_texts=user_texts_n,
                system_prompt=persona_prompts["assistant"],
                temperature=float(base_gen["temperature"]),
                max_tokens=int(base_gen["max_tokens"]),
                seed=int(base_gen["seed"]),
            )
            neutral_examples = [
                Example(
                    persona="assistant",
                    doc_id=int(docs[idx].get("doc_id", idx)),
                    user=user_texts_n[i],
                    assistant=asst_texts_n[i],
                    cos_to_persona_dir=cos,
                    source_corpus=docs[idx].get("source_corpus", "unknown"),
                    qwen3_axis_bucket=docs[idx].get("axis_bucket", "unknown"),
                )
                for i, (idx, cos) in enumerate(n_picks)
            ]
            neutral_examples, n_drop_n, drop_rate_n = filter_zlt_contamination(
                neutral_examples, p, "neutral", hard_gate=contam_gate
            )
            neutral_pool[p] = neutral_examples

            meta[p] = {
                "persona_style": build_pool_meta(
                    p, "persona-style", picks, docs, n_drop_ps, drop_rate_ps
                ),
                "neutral": build_pool_meta(
                    p,
                    "neutral",
                    n_picks,
                    docs,
                    n_drop_n,
                    drop_rate_n,
                    achieved_neutral_threshold=float(achieved_thr.get(p, 0.0)),
                ),
            }

        # Random-bucket pool (villain only)
        log.info("=== building pools for villain (random-bucket) ===")
        rand_picks_villain = rand_picks["villain"]
        rand_user_texts = [
            (rand_docs[idx].get("full_text") or "")[:doc_text_cap] for idx, _ in rand_picks_villain
        ]
        rand_asst_texts = generate_assistant_turns(
            llm,
            tokenizer,
            user_texts=rand_user_texts,
            system_prompt=persona_prompts["villain"],
            temperature=float(base_gen["temperature"]),
            max_tokens=int(base_gen["max_tokens"]),
            seed=int(base_gen["seed"]),
        )
        rand_examples = [
            Example(
                persona="villain",
                doc_id=int(rand_docs[idx].get("doc_id", idx)),
                user=rand_user_texts[i],
                assistant=rand_asst_texts[i],
                cos_to_persona_dir=cos,
                source_corpus=rand_docs[idx].get("source_corpus", "unknown"),
                qwen3_axis_bucket=rand_docs[idx].get("axis_bucket", "unknown"),
            )
            for i, (idx, cos) in enumerate(rand_picks_villain)
        ]
        rand_examples, n_drop_r, drop_rate_r = filter_zlt_contamination(
            rand_examples, "villain", "persona-style-random-bucket", hard_gate=contam_gate
        )
        rand_pool["villain"] = rand_examples
        meta["villain"]["persona_style_random_bucket"] = build_pool_meta(
            "villain",
            "persona-style-random-bucket",
            rand_picks_villain,
            rand_docs,
            n_drop_r,
            drop_rate_r,
        )
    finally:
        # Free vLLM before downstream phases
        del llm
        gc.collect()
        import torch

        torch.cuda.empty_cache()

    # Persist
    persona_pool_path.parent.mkdir(parents=True, exist_ok=True)
    flat_persona: list[Example] = []
    for p in persona_names:
        flat_persona.extend(persona_pool[p])
    flat_neutral: list[Example] = []
    for p in persona_names:
        flat_neutral.extend(neutral_pool[p])
    save_pool_jsonl(flat_persona, persona_pool_path)
    save_pool_jsonl(flat_neutral, neutral_pool_path)
    save_pool_jsonl(rand_pool["villain"], random_bucket_path)
    write_pool_meta(meta, meta_path)

    return {
        "persona_style": persona_pool_path,
        "neutral": neutral_pool_path,
        "random_bucket": random_bucket_path,
    }


def load_pools(cfg: dict) -> dict:
    """Load all pools from disk, indexed by persona."""
    from explore_persona_space.experiments.issue_375.example_pool import (
        load_pool_jsonl,
    )

    data_root = PROJECT_ROOT / cfg["output"]["data_root"]
    persona_examples = load_pool_jsonl(data_root / "example_pool_persona_style.jsonl")
    neutral_examples = load_pool_jsonl(data_root / "example_pool_neutral.jsonl")
    random_bucket_examples = load_pool_jsonl(
        data_root / "example_pool_persona_style_random_bucket.jsonl"
    )

    persona_style: dict[str, list] = {}
    for ex in persona_examples:
        persona_style.setdefault(ex.persona, []).append(ex)

    # The neutral pool is shared across personas (it's the assistant-prompted floor)
    # but we partition it by which persona's *direction* was used for the |cos|<thr
    # filter — the only information we retained is the cos value. The per-persona
    # neutral pools were saved in order (persona_names sorted alphabetically).
    # Reconstruct by slicing: each persona has K_PER entries, contiguous.
    k_per = len(neutral_examples) // 3
    neutral_by_persona: dict[str, list] = {}
    persona_names = sorted({"software_engineer", "librarian", "villain"})
    for i, p in enumerate(persona_names):
        neutral_by_persona[p] = neutral_examples[i * k_per : (i + 1) * k_per]

    return {
        "persona_style": persona_style,
        "neutral": neutral_by_persona,
        "random_bucket": {"villain": random_bucket_examples},
    }


# ────────────────────────────────────────────────────────────────────────────
# Adapter merge + vLLM helpers
# ────────────────────────────────────────────────────────────────────────────


def download_adapter(repo_id: str, hub_subpath: str, local_dir: Path) -> Path:
    """Snapshot-download a single adapter dir from HF Hub.

    ``allow_patterns`` is the canonical scoped download — we don't want the
    rest of the repo (350+ GB) on disk.
    """
    from huggingface_hub import snapshot_download

    log.info("downloading adapter %s/%s -> %s", repo_id, hub_subpath, local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        allow_patterns=f"{hub_subpath}/*",
        local_dir=str(local_dir),
        token=os.environ.get("HF_TOKEN"),
    )
    adapter_path = local_dir / hub_subpath
    if not (adapter_path / "adapter_config.json").exists():
        raise RuntimeError(
            f"download_adapter: expected adapter_config.json under {adapter_path} after "
            f"snapshot_download — got {sorted(p.name for p in adapter_path.iterdir())}"
        )
    return adapter_path


def merge_and_save(base_model: str, adapter_path: Path, merged_path: Path) -> Path:
    """Merge LoRA adapter into base model and save to ``merged_path``."""
    from explore_persona_space.train.sft import merge_lora

    merged_path.parent.mkdir(parents=True, exist_ok=True)
    if merged_path.exists():
        shutil.rmtree(merged_path)
    merge_lora(
        base_model_path=base_model,
        adapter_path=str(adapter_path),
        output_dir=str(merged_path),
        gpu_id=0,
    )
    log.info("merged adapter -> %s", merged_path)
    return merged_path


# ────────────────────────────────────────────────────────────────────────────
# Phase: zero-shot, base controls, main, pilot
# ────────────────────────────────────────────────────────────────────────────


@dataclass
class CellSpec:
    """A single cell to run."""

    label: str
    adapter_id: str  # "villain_C1" or "base"
    # pool_kind: one of {"zero-shot", "persona-style", "neutral",
    #                    "wrong-persona", "persona-style-random-bucket"}.
    pool_kind: str
    k: int
    # pool_persona: which persona's pool to draw from. Matches the adapter's
    # persona for main cells; wrong-persona / base cells use a different one.
    pool_persona: str


def build_core_cells(cfg: dict, adapter_id: str) -> list[CellSpec]:
    """Five core cells per adapter: k=0 zero-shot, persona-style {k=1, k=3}, neutral {k=1, k=3}."""
    adapter_meta = _adapter_meta(cfg, adapter_id)
    p = adapter_meta["persona"]
    seed = int(cfg["seed"])
    cells = [
        CellSpec(
            label=f"{adapter_id}_zero-shot_k0_seed{seed}",
            adapter_id=adapter_id,
            pool_kind="zero-shot",
            k=0,
            pool_persona=p,
        ),
    ]
    for k in (1, 3):
        cells.append(
            CellSpec(
                label=f"{adapter_id}_persona-style_k{k}_seed{seed}",
                adapter_id=adapter_id,
                pool_kind="persona-style",
                k=k,
                pool_persona=p,
            )
        )
        cells.append(
            CellSpec(
                label=f"{adapter_id}_neutral_k{k}_seed{seed}",
                adapter_id=adapter_id,
                pool_kind="neutral",
                k=k,
                pool_persona=p,
            )
        )
    return cells


def build_wrong_persona_cells(cfg: dict, adapter_id: str) -> list[CellSpec]:
    """W1..W4: clean-baseline adapter + a different persona's pool at k=3."""
    wrong_map = cfg.get("wrong_persona_map", {})
    if adapter_id not in wrong_map:
        return []
    seed = int(cfg["seed"])
    wrong_persona = wrong_map[adapter_id]
    return [
        CellSpec(
            label=f"{adapter_id}_wrong-persona_k3_seed{seed}",
            adapter_id=adapter_id,
            pool_kind="wrong-persona",
            k=3,
            pool_persona=wrong_persona,
        )
    ]


def build_pool_bias_cells(cfg: dict, adapter_id: str) -> list[CellSpec]:
    """P1: villain-C1 + villain-style-random-bucket at k=3."""
    pbs = cfg.get("pool_bias_sensitivity", {})
    if adapter_id != pbs.get("adapter_id"):
        return []
    seed = int(cfg["seed"])
    return [
        CellSpec(
            label=f"{adapter_id}_persona-style-random-bucket_k3_seed{seed}",
            adapter_id=adapter_id,
            pool_kind="persona-style-random-bucket",
            k=3,
            pool_persona=pbs.get("persona", "villain"),
        )
    ]


def build_base_cells(cfg: dict) -> list[CellSpec]:
    """B1, B2, B3: base model (no adapter) + persona-style k=3."""
    out: list[CellSpec] = []
    for cell_cfg in cfg["base_model_cells"]:
        out.append(
            CellSpec(
                label=cell_cfg["id"],
                adapter_id="base",
                pool_kind="persona-style",
                k=int(cell_cfg["k"]),
                pool_persona=cell_cfg["pool_persona"],
            )
        )
    return out


def _adapter_meta(cfg: dict, adapter_id: str) -> dict:
    for a in cfg["adapters"]:
        if a["id"] == adapter_id:
            return a
    raise KeyError(f"adapter id not found in config: {adapter_id!r}")


def _resolve_pool(cell: CellSpec, pools: dict) -> list:
    """Get the example pool for a cell."""
    if cell.pool_kind == "zero-shot":
        return []
    if cell.pool_kind == "persona-style":
        return pools["persona_style"][cell.pool_persona]
    if cell.pool_kind == "neutral":
        return pools["neutral"][cell.pool_persona]
    if cell.pool_kind == "wrong-persona":
        # Wrong-persona uses the persona-style pool of the *other* persona.
        return pools["persona_style"][cell.pool_persona]
    if cell.pool_kind == "persona-style-random-bucket":
        return pools["random_bucket"][cell.pool_persona]
    raise ValueError(f"unknown pool_kind: {cell.pool_kind!r}")


def run_cells_for_llm(
    cfg: dict,
    llm,
    tokenizer,
    cells: Sequence[CellSpec],
    queries: list,
    pools: dict,
) -> list[dict]:
    """Run a sequence of cells on a single vLLM instance."""
    from explore_persona_space.experiments.issue_375.drift_eval import (
        run_cell,
    )

    eval_root = PROJECT_ROOT / cfg["output"]["eval_results_root"]
    decoder = cfg["decoder"]
    summaries: list[dict] = []
    for cell in cells:
        pool = _resolve_pool(cell, pools)
        if cell.k > 0 and len(pool) < cell.k:
            raise RuntimeError(
                f"cell {cell.label} requires k={cell.k} examples but pool of "
                f"kind {cell.pool_kind!r} for persona={cell.pool_persona!r} "
                f"only has {len(pool)} examples"
            )
        summary = run_cell(
            llm=llm,
            tokenizer=tokenizer,
            cell_label=cell.label,
            queries=queries,
            pool=pool,
            k=cell.k,
            adapter_id=cell.adapter_id,
            pool_kind=cell.pool_kind,
            decoder=decoder,
            eval_results_root=eval_root,
        )
        summaries.append(summary)
    return summaries


def make_vllm(cfg: dict, model_path: str):
    """Construct a vLLM ``LLM`` instance from config."""
    from vllm import LLM

    return LLM(
        model=model_path,
        dtype=cfg["vllm"]["dtype"],
        gpu_memory_utilization=float(cfg["vllm"]["gpu_memory_utilization"]),
        max_model_len=int(cfg["vllm"]["max_model_len"]),
        max_num_seqs=int(cfg["vllm"]["max_num_seqs"]),
        seed=int(cfg["vllm"]["seed"]),
    )


def free_vllm(llm) -> None:
    """Free vLLM and CUDA cache between adapters."""
    import torch

    del llm
    gc.collect()
    torch.cuda.empty_cache()


def phase_base_floor(cfg: dict) -> dict:
    """Plan §4.5 step 3: held-out base-model floor — fire rate < 1% on the
    200 held-out queries under helpful-assistant prompt with NO few-shot
    context. Saves to ``eval_results/issue_375/base_floor.json``.
    """
    from transformers import AutoTokenizer

    queries = load_held_out_queries(cfg)
    threshold = float(cfg["held_out"]["base_floor_threshold"])
    log.info(
        "=== phase=base-floor: %d queries on base model (sanity floor < %.1f%%) ===",
        len(queries),
        threshold * 100,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    llm = make_vllm(cfg, cfg["base_model"])
    try:
        summaries = run_cells_for_llm(
            cfg=cfg,
            llm=llm,
            tokenizer=tokenizer,
            cells=[
                CellSpec(
                    label=f"base_floor_k0_seed{int(cfg['seed'])}",
                    adapter_id="base",
                    pool_kind="zero-shot",
                    k=0,
                    pool_persona="assistant",
                )
            ],
            queries=queries,
            pools={"persona_style": {}, "neutral": {}, "random_bucket": {}},
        )
    finally:
        free_vllm(llm)

    rate = summaries[0]["overall_rate"]
    payload = {
        "base_floor_rate": rate,
        "threshold": threshold,
        "gate_passed": rate < threshold,
        "n_queries": summaries[0]["n_queries"],
        "n_completions": summaries[0]["n_completions"],
    }
    out = PROJECT_ROOT / cfg["output"]["eval_results_root"] / "base_floor.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    if not payload["gate_passed"]:
        raise RuntimeError(
            f"phase=base-floor HARD GATE FAILED: base-model marker rate {rate:.4f} >= "
            f"{threshold} on held-out queries with no few-shot context. The query distribution "
            f"is somehow leaking the marker. Strip offending queries and re-run."
        )
    log.info("base-floor PASS: rate=%.4f < threshold=%.4f", rate, threshold)
    return payload


def phase_pilot(cfg: dict) -> dict:
    """Throughput pilot — 200 generations on villain_C1 at k=3 max=2048,
    measure end-to-end gen/s to drive the §8 mode decision.
    """
    queries = load_held_out_queries(cfg)
    pools = load_pools(cfg)
    adapter_id = "villain_C1"
    adapter_meta = _adapter_meta(cfg, adapter_id)
    adapter_dir = PROJECT_ROOT / "data" / "issue_375" / "_adapters" / adapter_id
    adapter_path = download_adapter(cfg["adapter_repo"], adapter_meta["hub_subpath"], adapter_dir)
    with tempfile.TemporaryDirectory(prefix=f"merged_{adapter_id}_") as td:
        merged_path = Path(td) / "merged"
        merge_and_save(cfg["base_model"], adapter_path, merged_path)

        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            str(merged_path), trust_remote_code=True, token=os.environ.get("HF_TOKEN")
        )
        llm = make_vllm(cfg, str(merged_path))
        try:
            pilot_cell = CellSpec(
                label=f"{adapter_id}_pilot_persona-style_k3_seed{int(cfg['seed'])}",
                adapter_id=adapter_id,
                pool_kind="persona-style",
                k=3,
                pool_persona=adapter_meta["persona"],
            )
            t0 = time.monotonic()
            summaries = run_cells_for_llm(
                cfg=cfg,
                llm=llm,
                tokenizer=tokenizer,
                cells=[pilot_cell],
                queries=queries,
                pools=pools,
            )
            elapsed = time.monotonic() - t0
        finally:
            free_vllm(llm)

    n_completions = summaries[0]["n_completions"]
    gen_per_s = n_completions / elapsed if elapsed > 0 else 0.0
    projected_hours = (116_000 / gen_per_s / 3600.0) if gen_per_s > 0 else float("inf")
    out = PROJECT_ROOT / cfg["output"]["eval_results_root"] / "pilot_throughput.json"
    payload = {
        "elapsed_s": elapsed,
        "n_completions": n_completions,
        "gen_per_s": gen_per_s,
        "projected_full_sweep_hours_116k": projected_hours,
        "cell_label": pilot_cell.label,
        "pilot_rate": summaries[0]["overall_rate"],
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2))
    log.info(
        "pilot: %d gens in %.1fs → %.2f gen/s, projected full sweep ~%.1f h",
        n_completions,
        elapsed,
        gen_per_s,
        projected_hours,
    )
    return payload


def phase_zero_shot(cfg: dict) -> list[dict]:
    """Run k=0 zero-shot cells for ALL adapters — verifies sanity gates 3+4."""
    queries = load_held_out_queries(cfg)
    pools = load_pools(cfg)
    summaries: list[dict] = []
    seed = int(cfg["seed"])
    for adapter_id in cfg["adapter_order"]:
        adapter_meta = _adapter_meta(cfg, adapter_id)
        adapter_dir = PROJECT_ROOT / "data" / "issue_375" / "_adapters" / adapter_id
        adapter_path = download_adapter(
            cfg["adapter_repo"], adapter_meta["hub_subpath"], adapter_dir
        )
        with tempfile.TemporaryDirectory(prefix=f"merged_{adapter_id}_") as td:
            merged_path = Path(td) / "merged"
            merge_and_save(cfg["base_model"], adapter_path, merged_path)
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(merged_path), trust_remote_code=True, token=os.environ.get("HF_TOKEN")
            )
            llm = make_vllm(cfg, str(merged_path))
            try:
                cell = CellSpec(
                    label=f"{adapter_id}_zero-shot_k0_seed{seed}",
                    adapter_id=adapter_id,
                    pool_kind="zero-shot",
                    k=0,
                    pool_persona=adapter_meta["persona"],
                )
                summaries.extend(run_cells_for_llm(cfg, llm, tokenizer, [cell], queries, pools))
            finally:
                free_vllm(llm)
    return summaries


def phase_base_controls(cfg: dict) -> list[dict]:
    """B1/B2/B3 — base model + persona-style k=3 (sanity gate 6)."""
    queries = load_held_out_queries(cfg)
    pools = load_pools(cfg)
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        cfg["base_model"], trust_remote_code=True, token=os.environ.get("HF_TOKEN")
    )
    llm = make_vllm(cfg, cfg["base_model"])
    try:
        cells = build_base_cells(cfg)
        summaries = run_cells_for_llm(cfg, llm, tokenizer, cells, queries, pools)
    finally:
        free_vllm(llm)
    return summaries


def phase_main(
    cfg: dict,
    *,
    skip_zero_shot: bool = True,
    only_clean_baseline: bool = False,
) -> list[dict]:
    """Run all non-zero-shot cells per adapter (persona-style, neutral,
    wrong-persona for clean adapters, P1 for villain_C1).

    Args:
        skip_zero_shot: zero-shot cells are usually run as the dedicated
            ``--phase zero-shot`` step first; skip them here by default.
        only_clean_baseline: if True, restrict to the 4 strict-test adapters
            (degraded-mode option from plan §8).
    """
    queries = load_held_out_queries(cfg)
    pools = load_pools(cfg)
    summaries: list[dict] = []
    clean_ids = {a["id"] for a in cfg["adapters"] if a.get("set") == "strict"}

    for adapter_id in cfg["adapter_order"]:
        if only_clean_baseline and adapter_id not in clean_ids:
            log.info(
                "phase=main: skipping non-clean adapter %s (only_clean_baseline=True)", adapter_id
            )
            continue
        adapter_meta = _adapter_meta(cfg, adapter_id)
        adapter_dir = PROJECT_ROOT / "data" / "issue_375" / "_adapters" / adapter_id
        adapter_path = download_adapter(
            cfg["adapter_repo"], adapter_meta["hub_subpath"], adapter_dir
        )
        with tempfile.TemporaryDirectory(prefix=f"merged_{adapter_id}_") as td:
            merged_path = Path(td) / "merged"
            merge_and_save(cfg["base_model"], adapter_path, merged_path)

            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(
                str(merged_path), trust_remote_code=True, token=os.environ.get("HF_TOKEN")
            )
            llm = make_vllm(cfg, str(merged_path))
            try:
                cells = build_core_cells(cfg, adapter_id)
                if skip_zero_shot:
                    cells = [c for c in cells if c.pool_kind != "zero-shot"]
                cells.extend(build_wrong_persona_cells(cfg, adapter_id))
                cells.extend(build_pool_bias_cells(cfg, adapter_id))
                summaries.extend(run_cells_for_llm(cfg, llm, tokenizer, cells, queries, pools))
            finally:
                free_vllm(llm)
    return summaries


# ────────────────────────────────────────────────────────────────────────────
# Phase: analyze
# ────────────────────────────────────────────────────────────────────────────


def phase_analyze(cfg: dict) -> dict:
    """Aggregate per-cell results, run bootstraps, produce figures."""
    from explore_persona_space.experiments.issue_375 import analyze

    eval_root = PROJECT_ROOT / cfg["output"]["eval_results_root"]
    figures_root = PROJECT_ROOT / cfg["output"]["figures_root"]
    seed = int(cfg["seed"])

    # Build list of all cell labels we expect.
    all_labels: list[str] = []
    for adapter_id in cfg["adapter_order"]:
        for k in (0,):
            all_labels.append(analyze.cell_label(adapter_id, "zero-shot", k, seed))
        for k in (1, 3):
            all_labels.append(analyze.cell_label(adapter_id, "persona-style", k, seed))
            all_labels.append(analyze.cell_label(adapter_id, "neutral", k, seed))

    for adapter_id, _persona in cfg.get("wrong_persona_map", {}).items():
        all_labels.append(analyze.cell_label(adapter_id, "wrong-persona", 3, seed))

    pbs = cfg.get("pool_bias_sensitivity", {})
    if pbs:
        all_labels.append(analyze.pool_bias_cell_label(pbs["adapter_id"], int(pbs["k"]), seed))

    for cell_cfg in cfg["base_model_cells"]:
        all_labels.append(cell_cfg["id"])

    # 1) aggregated.json
    aggregated = analyze.write_aggregated(eval_root, all_labels, eval_root / "aggregated.json")

    # 2) stratified
    analyze.write_stratified(eval_root, all_labels, eval_root / "stratified_by_query_source.json")

    # 3) bootstrap
    strict_adapters = [a["id"] for a in cfg["adapters"] if a.get("set") == "strict"]
    secondary_adapters = [a["id"] for a in cfg["adapters"] if a.get("set") == "secondary"]
    n_boot = int(cfg["bootstrap"]["n_boot"])
    boot_seed = int(cfg["bootstrap"]["seed"])

    strict_bootstrap = analyze.compute_strict_test_suite(
        eval_root, strict_adapters, cfg.get("wrong_persona_map", {}), n_boot=n_boot, seed=boot_seed
    )
    secondary_bootstrap = analyze.compute_secondary_delta_suite(
        eval_root, secondary_adapters, n_boot=n_boot, seed=boot_seed
    )
    pool_bias_bootstrap = {}
    if pbs:
        pool_bias_bootstrap["villain_C1_axis_extreme_vs_random_bucket"] = (
            analyze.compute_pairwise_bootstrap(
                eval_root,
                analyze.cell_label(pbs["adapter_id"], "persona-style", int(pbs["k"]), seed),
                analyze.pool_bias_cell_label(pbs["adapter_id"], int(pbs["k"]), seed),
                n_boot=n_boot,
                seed=boot_seed,
            )
        )

    analyze.write_bootstrap(
        {
            "strict": strict_bootstrap,
            "secondary": secondary_bootstrap,
            "pool_bias": pool_bias_bootstrap,
        },
        eval_root / "bootstrap.json",
    )

    # 4) Figures
    figure_paths = {}
    figure_paths["hero"] = analyze.make_hero_figure(
        eval_root, figures_root, strict_adapters, seed=seed
    )
    figure_paths["wrong_persona"] = analyze.make_wrong_persona_null_figure(
        eval_root, figures_root, cfg.get("wrong_persona_map", {}), seed=seed
    )
    figure_paths["base_model"] = analyze.make_base_model_null_figure(
        eval_root, figures_root, [c["pool_persona"] for c in cfg["base_model_cells"]], seed=seed
    )
    if pbs:
        figure_paths["pool_bias"] = analyze.make_pool_bias_sensitivity_figure(
            eval_root, figures_root, pbs["adapter_id"], seed=seed
        )
    figure_paths["secondary"] = analyze.make_secondary_delta_figure(
        eval_root, figures_root, secondary_adapters, seed=seed
    )

    return {
        "aggregated": aggregated,
        "figures": {k: {fmt: str(p) for fmt, p in v.items()} for k, v in figure_paths.items()},
    }


# ────────────────────────────────────────────────────────────────────────────
# Phase: upload raw completions
# ────────────────────────────────────────────────────────────────────────────


def phase_upload_raw_completions(cfg: dict) -> dict:
    """Upload every cell's ``raw_completions.json`` to the HF Hub data repo."""
    from explore_persona_space.orchestrate.hub import (
        upload_raw_completions_to_data_repo,
    )

    eval_root = PROJECT_ROOT / cfg["output"]["eval_results_root"]
    return upload_raw_completions_to_data_repo(
        experiment_name=cfg["output"]["experiment_name"],
        eval_results_dir=eval_root,
        delete_after=False,
    )


# ────────────────────────────────────────────────────────────────────────────
# Main CLI
# ────────────────────────────────────────────────────────────────────────────


PHASES = (
    "persona-directions",
    "build-pools",
    "build-queries",
    "base-floor",
    "pilot",
    "zero-shot",
    "base-controls",
    "main",
    "analyze",
    "upload-raw",
    "all",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase",
        choices=PHASES,
        default="all",
        help="Which phase to run. 'all' runs the full pipeline.",
    )
    parser.add_argument("--config", default="configs/issue_375/conditions.yaml")
    parser.add_argument("--seed", type=int, default=None, help="Override the seed in the config.")
    parser.add_argument(
        "--force", action="store_true", help="Re-build caches even if outputs exist."
    )
    parser.add_argument(
        "--only-clean-baseline",
        action="store_true",
        help="Run only the 4 strict-test adapters (plan §8 degraded mode).",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Import every module and parse the config — do NOT run any GPU work. "
        "Use to verify the script can be loaded end-to-end inside `uv run`.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        log.info("smoke test: loading modules and parsing config (no GPU work)")
        cfg = load_config(args.config)
        # Touch every module — surfaces ImportErrors
        import importlib

        for mod_path in (
            "explore_persona_space.experiments.issue_375.persona_directions",
            "explore_persona_space.experiments.issue_375.example_pool",
            "explore_persona_space.experiments.issue_375.fewshot_prompt",
            "explore_persona_space.experiments.issue_375.drift_eval",
            "explore_persona_space.experiments.issue_375.analyze",
        ):
            importlib.import_module(mod_path)
        log.info(
            "smoke test PASS: %d adapters, %d base cells, %d k values configured",
            len(cfg["adapters"]),
            len(cfg["base_model_cells"]),
            len(cfg["k_values"]),
        )
        # Sanity-check we can build the cell matrix
        strict = [a["id"] for a in cfg["adapters"] if a.get("set") == "strict"]
        secondary = [a["id"] for a in cfg["adapters"] if a.get("set") == "secondary"]
        log.info("strict adapters (n=%d): %s", len(strict), strict)
        log.info("secondary adapters (n=%d): %s", len(secondary), secondary)
        return 0

    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["seed"] = args.seed
        cfg["decoder"]["seed"] = args.seed
        cfg["vllm"]["seed"] = args.seed
        cfg["bootstrap"]["seed"] = args.seed

    phase = args.phase
    log.info("=== run_issue375_incontext_drift phase=%s seed=%d ===", phase, cfg["seed"])

    if phase in ("persona-directions", "all"):
        phase_persona_directions(cfg, force=args.force)
    if phase in ("build-queries", "all"):
        phase_build_queries(cfg, force=args.force)
    if phase in ("build-pools", "all"):
        phase_build_pools(cfg, force=args.force)
    if phase in ("base-floor", "all"):
        phase_base_floor(cfg)
    if phase in ("pilot",):
        phase_pilot(cfg)
    if phase in ("zero-shot", "all"):
        phase_zero_shot(cfg)
    if phase in ("base-controls", "all"):
        phase_base_controls(cfg)
    if phase in ("main", "all"):
        phase_main(cfg, skip_zero_shot=True, only_clean_baseline=args.only_clean_baseline)
    if phase in ("analyze", "all"):
        phase_analyze(cfg)
    if phase in ("upload-raw", "all"):
        phase_upload_raw_completions(cfg)

    log.info("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
