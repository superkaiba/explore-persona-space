#!/usr/bin/env python3
"""Issue #2588 pod cell driver — generation / capture / fits / nulls / transfer.

One invocation runs ONE generation/capture cell (``--cell <model>_<arm>``,
plan §4.1: 19 cells, 21 registered maps) through its phase sequence:

    prologue -> stage -> gen -> parse -> capture -> upload-raw ->
    fits (fail-closed on the G2 anchor sentinel) -> nulls -> gpqa-transfer ->
    resid -> upload-fits -> sentinel

plus the standalone ``g2-anchor`` phase (anchor pod only) and
``purge-model-cache`` (after a model's LAST cell). ``--phase all`` runs the
cell's own sequence; ``--smoke`` shrinks every axis (generic 400/50/50, GPQA
20 q x 2 rollouts, one 20-draw null block) while keeping the SAME dispatcher,
phases, launch width, env injection and uploads (routed to the
``{PANEL_PREFIX}/smoke/`` prefix) — smoke IS production with small N.

Reuse spine (never re-derived): manifest staging + engine/capture internals
from ``scripts/issue2330_qwen35_generate_capture.py`` (G), fit cores from
``scripts/issue2330_matched_fits.py`` (MF) / ``scripts/issue1491_ladder_fits.py``
(LF) / ``scripts/issue779_ffc_n1m_fits.py`` (F), ports + registry from
``scripts/issue2588_panel_common.py`` (PC).
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")  # before ANY vllm import

_SCRIPTS = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPTS.parent
for p in (str(_SCRIPTS), str(_REPO_ROOT / "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # thread caps BEFORE torch import (shared-VM smoke legs; no-op on pods)

import issue2330_matched_fits as MF  # noqa: E402  (imports F/LF transitively)
import issue2330_qwen35_generate_capture as G  # noqa: E402  (module-top _load_dotenv + torch)
import issue2588_panel_common as PC  # noqa: E402
import issue779_ffc_n1m_fits as F  # noqa: E402
import issue1491_ladder_fits as LF  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

logger = logging.getLogger("issue2588_run_cell")

_ENGINE_CONSTRUCTED = False  # drives the os._exit(0) terminal (vLLM teardown gotcha)

GENERIC_SPLITS = ("train_10k", "val_400", "test_1000")
SMOKE_SLICE = {"train_10k": 400, "val_400": 50, "test_1000": 50}
SMOKE_GPQA_QUESTIONS = 20
SMOKE_GPQA_ROLLOUTS = 2
SMOKE_PERM_DRAWS = 20
GPQA_PROMPTS_PATH = _REPO_ROOT / "eval_results" / "issue_2588" / "gpqa_prompts.json"


def _meta() -> dict:
    """Reproducibility metadata for every result JSON (CLAUDE.md requirement)."""
    import transformers

    try:
        import vllm

        vllm_version = vllm.__version__
    except ImportError:  # CPU smoke boxes without vllm still write fits JSONs
        vllm_version = "not-installed"
    return {
        "issue": PC.TASK_ID,
        "git_sha": G._git_sha(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "vllm": vllm_version,
        "numpy": np.__version__,
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }


# ---------------------------------------------------------------------------
# Paths + upload helpers
# ---------------------------------------------------------------------------


def _cell_prefix(args, cell: PC.Cell) -> str:
    base = cell.hf_prefix
    if args.smoke:
        base = (
            f"{PC.PANEL_PREFIX}/smoke/{cell.model_key}/{'nothink' if cell.arm == 'a' else 'think'}"
        )
    return base


def _paths(args, cell: PC.Cell) -> dict[str, Path]:
    root = Path(args.out_root)
    cell_dir = root / ("smoke" if args.smoke else "cells") / cell.key
    d = {
        "root": root,
        "cell": cell_dir,
        "raw": cell_dir / "raw_completions",
        "parsed": cell_dir / "parsed",
        "capture": cell_dir / "capture",
        "fits": cell_dir / "fits",
        "cache": root / "hf_cache",
        "logs": Path("/workspace/logs") if Path("/workspace").is_dir() else root / "logs",
    }
    for v in d.values():
        v.mkdir(parents=True, exist_ok=True)
    return d


def _upload_dir(local_dir: Path, path_in_repo: str, what: str) -> None:
    """One bulk upload_folder commit (never a per-file loop; gotchas.md 504-storm)."""
    from huggingface_hub import HfApi

    def _do():
        HfApi().upload_folder(
            folder_path=str(local_dir),
            path_in_repo=path_in_repo,
            repo_id=PC.HF_DATA_REPO,
            repo_type="dataset",
            commit_message=f"issue2588: {what}",
        )

    G._retry_transient(_do, what=f"upload_folder {path_in_repo}")
    logger.info("[i2588] uploaded %s -> %s", local_dir, path_in_repo)


def _upload_file(local: Path, path_in_repo: str, what: str) -> None:
    from huggingface_hub import HfApi

    def _do():
        HfApi().upload_file(
            path_or_fileobj=str(local),
            path_in_repo=path_in_repo,
            repo_id=PC.HF_DATA_REPO,
            repo_type="dataset",
            commit_message=f"issue2588: {what}",
        )

    G._retry_transient(_do, what=f"upload_file {path_in_repo}")


# ---------------------------------------------------------------------------
# Phase: prologue (G1 + G6 venv/config asserts; every pod, every cell)
# ---------------------------------------------------------------------------


def phase_prologue(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("prologue")
    tf_version = PC.assert_transformers_floor()
    logger.info("[i2588] G6 transformers floor OK: %s", tf_version)
    m = cell.model
    if m.family in ("olmo_instruct", "olmo_think"):
        rec = PC.assert_olmo_rope_split(m.hf_id)
        logger.info("[i2588] G6 rope split OK: %s", rec)
    PC.assert_max_position_embeddings(m.hf_id)
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(m.hf_id)
    sha16 = PC.assert_template_sidespec(tok, m.family, cell.arm)
    pins = PC.assert_think_pins(tok, m.family)
    logger.info(
        "[i2588] G1 sidespec OK (%s, arm %s): render sha16=%s pins=%s",
        m.family,
        cell.arm,
        sha16,
        pins,
    )
    PC.write_json_atomic(
        paths["cell"] / "prologue.json",
        {
            "meta": _meta(),
            "transformers": tf_version,
            "render_sha16": sha16,
            "think_pins": {k: list(v) for k, v in pins.items()} if pins else {},
        },
    )


# ---------------------------------------------------------------------------
# Phase: stage (manifest + splits + GPQA prompts + banked texts)
# ---------------------------------------------------------------------------


P0_PATH = _REPO_ROOT / "eval_results" / "issue_2588" / "p0_preflight.json"


def _p0_union_drop() -> dict[str, set[int]]:
    """The P0 step-7 12-tokenizer union-drop set (committed on the issue branch)."""
    assert P0_PATH.exists(), (
        f"{P0_PATH} missing — run issue2588_p0_preflight.py (all steps) and commit its "
        "report to the issue branch before any pod launch (plan §4.2)."
    )
    rec = json.loads(P0_PATH.read_text(encoding="utf-8"))
    scan = rec.get("length-scan")
    assert scan is not None, f"{P0_PATH} has no length-scan step — P0 step 7 never ran"
    assert not scan.get("quick_slice"), (
        "p0_preflight.json length-scan was a --quick slice — production pods must consume "
        "the FULL-corpus scan (re-run P0 step 7 without --quick)."
    )
    return {s: set(v) for s, v in scan["union_drop"].items()}


def _load_generic_rows(args, cache_dir: Path) -> dict[str, list[dict]]:
    """Manifest rows per generic split at the pinned revision, union-drop
    applied (P0 step 7), smoke-sliced."""
    split_ids = G._load_split_ids(PC.SPLIT_IDS_PATH)
    drop = _p0_union_drop()
    out: dict[str, list[dict]] = {}
    for split in GENERIC_SPLITS:
        manifest_key, ids_key, _seed = G.SPLIT_TO_MANIFEST[split]
        rows = G._download_manifest_split(manifest_key, cache_dir)
        subset = G._subset_rows(rows, split_ids["splits"][ids_key], ids_key)
        n_before = len(subset)
        subset = [r for r in subset if int(r["ladder_local_id"]) not in drop[split]]
        if len(subset) != n_before:
            logger.info(
                "[i2588] %s: %d rows union-dropped (P0 step 7)", split, n_before - len(subset)
            )
        if args.smoke:
            subset = subset[: SMOKE_SLICE[split]]
        out[split] = subset
    return out


def _load_gpqa_prompts(args) -> list[dict]:
    """The P0-frozen GPQA prompt file (committed on the issue branch)."""
    assert GPQA_PROMPTS_PATH.exists(), (
        f"{GPQA_PROMPTS_PATH} missing — run issue2588_p0_preflight.py step 3 (GPQA staging) "
        "and commit the frozen prompts to the issue branch before any pod launch."
    )
    payload = json.loads(GPQA_PROMPTS_PATH.read_text(encoding="utf-8"))
    rows = payload["prompts"]
    assert len(rows) == PC.GPQA_N_QUESTIONS, len(rows)
    if args.smoke:
        rows = rows[:SMOKE_GPQA_QUESTIONS]
    return rows


def phase_stage(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("stage")
    if args.pod_ordinal > 0:
        wait_s = args.pod_ordinal * 120
        logger.info(
            "[i2588] jittered weight pull: pod_ordinal=%d -> sleep %ds", args.pod_ordinal, wait_s
        )
        time.sleep(wait_s)
    generic = _load_generic_rows(args, paths["cache"])
    gpqa = _load_gpqa_prompts(args)
    logger.info(
        "[i2588] staged generic=%s gpqa=%d", {k: len(v) for k, v in generic.items()}, len(gpqa)
    )
    if not cell.fresh:
        _stage_banked(args, cell, paths)


def _stage_banked(args, cell: PC.Cell, paths: dict) -> None:
    """Stage banked #2330 cap2048 texts + ceiling draws (capture-only cells).

    Raw chunks live under ``<prefix>/raw_completions/`` per split subpath in
    the producer's own layout (G.store_subpath_for_split), consumed at the
    parent record pin PC.BANKED_REVISION. Rows carry {ci, prompt, response}
    (verified against the producer at G:963-970).
    """
    key = cell.model_key
    dest = paths["cell"] / "banked"
    dest.mkdir(parents=True, exist_ok=True)
    jobs = [(PC.BANKED_CAP2048[key], s) for s in GENERIC_SPLITS]
    jobs += [(PC.BANKED_CEILING[key].rsplit("/", 1)[0], f"ceiling_s{s}") for s in PC.CEILING_SEEDS]
    for prefix_root, split in jobs:
        if split.startswith("ceiling_s"):
            # BANKED_CEILING already ends in .../ceiling_draws — append seed dir only.
            sub_prefix = f"{PC.BANKED_CEILING[key]}/seed{split.removeprefix('ceiling_s')}"
        else:
            sub_prefix = f"{prefix_root}/{G.store_subpath_for_split(split)}"
        remote = G._remote_index(f"{sub_prefix}/raw_completions", revision=PC.BANKED_REVISION)
        names = sorted(n for n in remote if n.endswith(".json"))
        assert names, f"banked raw completions empty at {sub_prefix}/raw_completions"
        split_dir = dest / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for name in names:
            local = G._hub_download(
                f"{sub_prefix}/raw_completions/{name}",
                paths["cache"],
                revision=PC.BANKED_REVISION,
            )
            (split_dir / name).write_bytes(Path(local).read_bytes())
        logger.info("[i2588] banked %s: %d chunks staged", split, len(names))


# ---------------------------------------------------------------------------
# Phase: gen (vLLM; TokensPrompt from the SAME rendered ids capture re-checks)
# ---------------------------------------------------------------------------


def _build_engine_2588(model_id: str, seed: int, max_model_len: int, gpu_count: int):
    """vLLM engine (modeled on G._build_engine @ issue2330, with explicit
    max_model_len control for the G4/G5 regen re-pin, plan §7)."""
    global _ENGINE_CONSTRUCTED
    from vllm import LLM

    _ENGINE_CONSTRUCTED = True
    llm = LLM(
        model=model_id,
        tensor_parallel_size=gpu_count,
        seed=seed,
        dtype="bfloat16",
        max_model_len=max_model_len,
        gpu_memory_utilization=float(os.environ.get("VLLM_GPU_MEM_UTIL", "0.85")),
        max_num_seqs=64,
        enforce_eager=os.environ.get("VLLM_ENFORCE_EAGER", "0") == "1",
        enable_prefix_caching=False,
        disable_log_stats=True,
    )
    return llm


def _gen_rows(
    llm, tok, cell: PC.Cell, base_rows: list[dict], *, stage: str, cap: int, seed: int
) -> list[dict]:
    """Generate one stage's rollouts via TokensPrompt on the EXACT rendered ids.

    Generation consumes render_prompt_ids' ids directly (never a re-tokenize of
    the rendered string), so capture's re-tokenization parity assert checks the
    same object. Returns wrow dicts (PC.build_capture_row_2588 input shape).
    """
    from vllm import SamplingParams, TokensPrompt

    m = cell.model
    pins = PC.assert_think_pins(tok, m.family)
    open_ids = tuple(pins["open_ids"]) if pins else None
    rendered: list[dict] = []
    for r in base_rows:
        text = r["prompt"]
        prompt_render = tok.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=False,
            add_generation_prompt=True,
            **PC._template_kwargs(m.family, cell.arm),
        )
        ids = PC.render_prompt_ids(tok, text, m.family, cell.arm)
        n_prompt = len(ids)
        assert n_prompt <= PC.PROMPT_TOKEN_BUDGET, (
            f"{r['row_id']}: rendered prompt {n_prompt} tokens > budget "
            f"{PC.PROMPT_TOKEN_BUDGET} — P0 step 7 union-drop should have removed this row"
        )
        read_points = {"prompt_last": PC.compute_read_idx("prompt_last", ids)}
        if "pre_think" in cell.input_positions:
            read_points["pre_think"] = PC.compute_read_idx("pre_think", ids, open_ids=open_ids)
        rendered.append(
            {
                "row_id": r["row_id"],
                "prompt": prompt_render,
                "prompt_ids": ids,
                "n_prompt_tokens": n_prompt,
                "read_points": read_points,
                **({"gold": r["gold"], "qid": r["qid"]} if "gold" in r else {}),
            }
        )

    sp = SamplingParams(temperature=PC.GEN_TEMP, top_p=PC.GEN_TOP_P, seed=seed, max_tokens=cap)
    out_rows: list[dict] = []
    chunk = G.VLLM_CHUNK_SIZE
    for s in range(0, len(rendered), chunk):
        block = rendered[s : s + chunk]
        prompts = [TokensPrompt(prompt_token_ids=r["prompt_ids"]) for r in block]
        outs = llm.generate(prompts, sp, use_tqdm=False)
        for r, o in zip(block, outs, strict=True):
            comp = o.outputs[0]
            out_rows.append(
                {
                    **{k: v for k, v in r.items() if k != "prompt_ids"},
                    "text": comp.text,
                    "finish_reason": comp.finish_reason,
                    "n_comp_tokens": len(comp.token_ids),
                    "gen_seed": seed,
                    "cap": cap,
                    "stage": stage,
                }
            )
        logger.info(
            "[i2588] [%s] gen chunk %d/%d done",
            stage,
            s // chunk + 1,
            math.ceil(len(rendered) / chunk),
        )
    return out_rows


def _needs_regen(rows: list[dict], parse_mode: str) -> list[int]:
    """Indices needing the ONE-round G4/G5 regen (length-hit or unclosed think)."""
    idx = []
    for i, r in enumerate(rows):
        if r["finish_reason"] == "length":
            idx.append(i)
        elif parse_mode != "off":
            wf, reason, _, _ = PC.segment_completion_arm(r["text"], parse_mode)
            if not wf and reason.startswith(("close_count_", "open")):
                idx.append(i)
    return idx


def _gen_stage_with_regen(
    args,
    cell: PC.Cell,
    tok,
    base_rows: list[dict],
    *,
    stage: str,
    cap: int,
    seed: int,
    paths: dict,
    llm_holder: dict,
) -> list[dict]:
    """Generate a stage; apply the pre-registered G4/G5 single regen round.

    Trigger: cap-hit frac > 2% (G4) or unclosed-think frac > 2% (G5) per stage
    -> re-generate the affected rows at 2x cap with the engine RE-INSTANTIATED
    at max_model_len = PROMPT_TOKEN_BUDGET + 2*cap (bounded by 23,488; the
    max_position_embeddings floor is asserted at prologue). Persistent residue
    is dropped-and-counted at parse (plan §7 G4/G5).
    """
    m = cell.model
    if llm_holder.get("llm") is None:
        mml = PC.PROMPT_TOKEN_BUDGET + cap
        llm_holder["llm"] = _build_engine_2588(m.hf_id, seed, mml, args.gpu_count)
        llm_holder["mml"] = mml
    elif llm_holder["mml"] < PC.PROMPT_TOKEN_BUDGET + cap:
        G._reap_vllm_engine(llm_holder["llm"])
        mml = PC.PROMPT_TOKEN_BUDGET + cap
        llm_holder["llm"] = _build_engine_2588(m.hf_id, seed, mml, args.gpu_count)
        llm_holder["mml"] = mml
    rows = _gen_rows(llm_holder["llm"], tok, cell, base_rows, stage=stage, cap=cap, seed=seed)

    cap_hits = sum(1 for r in rows if r["finish_reason"] == "length")
    cap_frac = cap_hits / max(1, len(rows))
    regen_idx = _needs_regen(rows, cell.parse_mode)
    regen_frac = len(regen_idx) / max(1, len(rows))
    report = {
        "stage": stage,
        "cap": cap,
        "n": len(rows),
        "cap_hits": cap_hits,
        "cap_hit_frac": cap_frac,
        "regen_candidates": len(regen_idx),
        "regen_frac": regen_frac,
        "regen_ran": False,
    }
    if regen_idx and (cap_frac > PC.CAP_HIT_TRIGGER or regen_frac > PC.UNCLOSED_THINK_TRIGGER):
        new_cap = 2 * cap
        new_mml = PC.PROMPT_TOKEN_BUDGET + new_cap
        assert new_mml <= PC.REGEN_MAX_MODEL_LEN_BOUND, (new_mml, PC.REGEN_MAX_MODEL_LEN_BOUND)
        logger.info(
            "[i2588] [%s] G4/G5 regen: %d rows at cap=%d (mml=%d)",
            stage,
            len(regen_idx),
            new_cap,
            new_mml,
        )
        G._reap_vllm_engine(llm_holder["llm"])
        llm_holder["llm"] = _build_engine_2588(m.hf_id, seed, new_mml, args.gpu_count)
        llm_holder["mml"] = new_mml
        redo_base = [base_rows[i] for i in regen_idx]
        redo = _gen_rows(
            llm_holder["llm"], tok, cell, redo_base, stage=stage, cap=new_cap, seed=seed
        )
        for i, rr in zip(regen_idx, redo, strict=True):
            rows[i] = rr
        report["regen_ran"] = True
        report["post_regen_cap_hits"] = sum(1 for r in rows if r["finish_reason"] == "length")
    stage_dir = paths["raw"] / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    for k in range(0, len(rows), G.VLLM_CHUNK_SIZE):
        PC.write_json_atomic(
            stage_dir / f"chunk{k // G.VLLM_CHUNK_SIZE:04d}.json",
            {
                "meta": _meta(),
                "cell": cell.key,
                "stage": stage,
                "seed": seed,
                "cap": cap,
                "rows": rows[k : k + G.VLLM_CHUNK_SIZE],
            },
        )
    PC.write_json_atomic(stage_dir / "cap_hit_report.json", {"meta": _meta(), **report})
    logger.info(
        "[i2588] [%s] cap-hit report: %s", stage, {k: v for k, v in report.items() if k != "stage"}
    )
    return rows


def phase_gen(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("gen")
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cell.model.hf_id)
    llm_holder: dict = {"llm": None, "mml": 0}
    gpqa = _load_gpqa_prompts(args)
    gpqa_base = [
        {"row_id": f"{r['qid']}_s{s}", "prompt": r["prompt"], "gold": r["gold"], "qid": r["qid"]}
        for s in _gpqa_seeds(args)
        for r in gpqa
    ]
    if cell.fresh:
        generic = _load_generic_rows(args, paths["cache"])
        cap = PC.CAP[(cell.arm, "generic")]
        for split in GENERIC_SPLITS:
            base = [
                {
                    "row_id": f"{split}_{r['ladder_local_id']}",
                    "prompt": r["prompt"],
                    "ci": int(r["ladder_local_id"]),
                }
                for r in generic[split]
            ]
            _gen_stage_with_regen(
                args,
                cell,
                tok,
                base,
                stage=split,
                cap=cap,
                seed=PC.GEN_SEED,
                paths=paths,
                llm_holder=llm_holder,
            )
        for seed in PC.CEILING_SEEDS:
            base = [
                {
                    "row_id": f"ceiling_s{seed}_{r['ladder_local_id']}",
                    "prompt": r["prompt"],
                    "ci": int(r["ladder_local_id"]),
                }
                for r in generic["test_1000"]
            ]
            _gen_stage_with_regen(
                args,
                cell,
                tok,
                base,
                stage=f"ceiling_s{seed}",
                cap=cap,
                seed=seed,
                paths=paths,
                llm_holder=llm_holder,
            )
    # GPQA rollouts run for EVERY cell (banked cells generate GPQA fresh; the
    # behavioral gap_GPQA + transfer read need them).
    gcap = PC.CAP[(cell.arm, "gpqa")]
    # Rollout seed rides SamplingParams per row group; vLLM seeds per-request via
    # SamplingParams.seed, so one engine pass per seed keeps draws independent.
    for seed in _gpqa_seeds(args):
        base = [r for r in gpqa_base if r["row_id"].endswith(f"_s{seed}")]
        _gen_stage_with_regen(
            args,
            cell,
            tok,
            base,
            stage=f"gpqa_s{seed}",
            cap=gcap,
            seed=seed,
            paths=paths,
            llm_holder=llm_holder,
        )
    if llm_holder.get("llm") is not None:
        G._reap_vllm_engine(llm_holder["llm"])


def _gpqa_seeds(args) -> tuple[int, ...]:
    return PC.GPQA_ROLLOUT_SEEDS[:SMOKE_GPQA_ROLLOUTS] if args.smoke else PC.GPQA_ROLLOUT_SEEDS


# ---------------------------------------------------------------------------
# Phase: parse (drop-and-count; dropped_row_ids.json)
# ---------------------------------------------------------------------------


def _iter_stage_rows(paths: dict, stage: str) -> list[dict]:
    stage_dir = paths["raw"] / stage
    rows: list[dict] = []
    for f in sorted(stage_dir.glob("chunk*.json")):
        rows.extend(json.loads(f.read_text(encoding="utf-8"))["rows"])
    return rows


def _banked_stage_rows(cell: PC.Cell, paths: dict, stage: str, tok) -> list[dict]:
    """Banked #2330 rows -> wrow shape (producer render conventions, G module)."""
    split_dir = paths["cell"] / "banked" / stage
    rows: list[dict] = []
    for f in sorted(split_dir.glob("*.json")):
        payload = json.loads(f.read_text(encoding="utf-8"))
        for r in payload["rows"]:
            if G._is_empty_response(r["response"]):
                continue
            prompt_render = G._render_prompt(tok, r["prompt"])
            ids = tok(prompt_render, add_special_tokens=False)["input_ids"]
            rows.append(
                {
                    "row_id": f"{stage}_{int(r['ci'])}",
                    "ci": int(r["ci"]),
                    "prompt": prompt_render,
                    "n_prompt_tokens": len(ids),
                    "read_points": {"prompt_last": len(ids) - 1},
                    "text": r["response"],
                    "finish_reason": r.get("finish_reason", "stop"),
                    "stage": stage,
                }
            )
    assert rows, f"banked stage {stage}: zero usable rows"
    return rows


def _stage_names(args, cell: PC.Cell) -> list[str]:
    stages = list(GENERIC_SPLITS) + [f"ceiling_s{s}" for s in PC.CEILING_SEEDS]
    stages += [f"gpqa_s{s}" for s in _gpqa_seeds(args)]
    return stages


def phase_parse(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("parse")
    dropped: dict[str, list[str]] = {}
    counts: dict[str, dict] = {}
    for stage in _stage_names(args, cell):
        if not cell.fresh and not stage.startswith("gpqa"):
            continue  # banked stages parse mode "off" at capture time (producer convention)
        rows = _iter_stage_rows(paths, stage)
        mode = cell.parse_mode
        parsed = []
        drops = []
        for r in rows:
            rec = PC.parse_generation(r, mode)
            if rec["well_formed"]:
                parsed.append(
                    {
                        **r,
                        "ans_char_span": rec["ans_char_span"],
                        "cot_char_span": rec["cot_char_span"],
                    }
                )
            else:
                drops.append({"row_id": r["row_id"], "reason": rec["reason"]})
        counts[stage] = {"n": len(rows), "kept": len(parsed), "dropped": len(drops)}
        dropped[stage] = [d["row_id"] for d in drops]
        PC.write_jsonl_atomic(paths["parsed"] / f"{stage}.jsonl", parsed)
        PC.write_json_atomic(
            paths["parsed"] / f"{stage}_drops.json", {"meta": _meta(), "drops": drops}
        )
        logger.info("[i2588] [parse %s] kept=%d dropped=%d", stage, len(parsed), len(drops))
    PC.write_json_atomic(
        paths["fits"] / "dropped_row_ids.json",
        {"meta": _meta(), "cell": cell.key, "counts": counts, "dropped_row_ids": dropped},
    )


# ---------------------------------------------------------------------------
# Phase: capture (teacher-forced dense sweep; fp32 stores; 2-slot semaphore)
# ---------------------------------------------------------------------------


class _CaptureReducer:
    """Per-layer hooks that REDUCE in-hook (positions + span means) so the full
    (B, T, H) stack is never buffered across layers (the #2330 memory shape,
    adapted to multi-position reads)."""

    def __init__(self, layers: list[int], h_dim: int):
        self.layers = layers
        self.h_dim = h_dim
        self.batch_meta: list[dict] | None = None
        self.out: dict[int, dict[str, list[np.ndarray]]] = {}

    def set_batch(self, metas: list[dict]) -> None:
        self.batch_meta = metas
        self.out = {li: {"pos": [], "y": []} for li in self.layers}

    def hook_for(self, layer_idx: int):
        def _hook(_mod, _inp, output):
            h = G._unwrap(output)  # (B, T, H)
            metas = self.batch_meta
            assert metas is not None and h.shape[0] == len(metas), (h.shape, len(metas or []))
            pos_rows, y_rows = [], []
            for i, mrow in enumerate(metas):
                pos_idx = torch.as_tensor(mrow["pos_list"], device=h.device)
                pos_rows.append(h[i, pos_idx].float().cpu().numpy())
                s, e = mrow["ans_span"]
                y_rows.append(h[i, s:e].float().mean(dim=0).cpu().numpy())
            self.out[layer_idx]["pos"].append(np.stack(pos_rows))
            self.out[layer_idx]["y"].append(np.stack(y_rows))

        return _hook


def _acquire_capture_slot(root: Path):
    """2-slot flock semaphore (plan §9: at most 2 concurrent capture writers
    per shared out-root). Returns the held file object (lock lives with it)."""
    import fcntl

    for i in (0, 1):
        fh = open(root / f".capture.sem.{i}", "w")  # noqa: SIM115 — lock lifetime object
        try:
            fcntl.flock(fh, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return fh
        except OSError:
            fh.close()
    fh = open(root / ".capture.sem.0", "w")  # noqa: SIM115
    fcntl.flock(fh, fcntl.LOCK_EX)
    return fh


def _capture_stage(
    args,
    cell: PC.Cell,
    paths: dict,
    hf,
    tok,
    stage: str,
    layers: list[int],
    *,
    y_only: bool = False,
    layer_tag: str = "capture",
) -> None:
    from explore_persona_space.atomic_io import savez_atomic

    m = cell.model
    if cell.fresh or stage.startswith("gpqa"):
        wrows = PC.read_jsonl(paths["parsed"] / f"{stage}.jsonl")
    else:
        wrows = _banked_stage_rows(cell, paths, stage, tok)
        for r in wrows:  # banked = parse mode "off": ans span = whole stripped text
            s, e = PC._strip_span(r["text"], 0, len(r["text"]))
            r["ans_char_span"] = [s, e]
    positions_wanted = cell.input_positions if not y_only else ("prompt_last",)
    built, build_drops = [], []
    for w in wrows:
        row, reason = PC.build_capture_row_2588(tok, w, positions_wanted=positions_wanted)
        if row is None:
            build_drops.append({"row_id": w["row_id"], "reason": reason})
        else:
            row["gold"] = w.get("gold")
            row["qid"] = w.get("qid")
            row["n_prompt_tokens"] = w["n_prompt_tokens"]
            built.append(row)
    if build_drops:
        PC.write_json_atomic(
            paths["parsed"] / f"{stage}_capture_drops.json", {"meta": _meta(), "drops": build_drops}
        )
    assert built, f"capture stage {stage}: zero capturable rows"

    blocks = G._resolve_decoder_blocks(hf)
    reducer = _CaptureReducer(layers, m.h_dim)
    handles = [blocks[li].register_forward_hook(reducer.hook_for(li)) for li in layers]
    stage_dir = paths[layer_tag] / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    rows_meta: list[dict] = []
    shard_size = 500
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    try:
        order = sorted(
            range(len(built)), key=lambda i: len(built[i]["prompt_ids"]) + len(built[i]["comp_ids"])
        )
        for shard_k, s0 in enumerate(range(0, len(order), shard_size)):
            shard_rows = [built[i] for i in order[s0 : s0 + shard_size]]
            per_layer_pos: dict[int, list[np.ndarray]] = {li: [] for li in layers}
            per_layer_y: dict[int, list[np.ndarray]] = {li: [] for li in layers}
            for b0 in range(0, len(shard_rows), args.capture_batch_size):
                batch = shard_rows[b0 : b0 + args.capture_batch_size]
                metas, seqs = [], []
                for row in batch:
                    full = row["prompt_ids"] + row["comp_ids"]
                    pos_list = [
                        row["positions"][p] for p in positions_wanted if p in row["positions"]
                    ]
                    metas.append({"pos_list": pos_list, "ans_span": row["spans"]["ans"]})
                    seqs.append(full)
                maxlen = max(len(sq) for sq in seqs)
                ids = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
                mask = torch.zeros((len(seqs), maxlen), dtype=torch.long)
                for i, sq in enumerate(seqs):  # RIGHT padding — absolute positions intact
                    ids[i, : len(sq)] = torch.as_tensor(sq)
                    mask[i, : len(sq)] = 1
                reducer.set_batch(metas)
                with torch.no_grad():
                    hf(
                        input_ids=ids.to(hf.device),
                        attention_mask=mask.to(hf.device),
                        **G._logits_to_keep_kwargs(hf),
                    )
                for li in layers:
                    per_layer_pos[li].append(reducer.out[li]["pos"][0])
                    per_layer_y[li].append(reducer.out[li]["y"][0])
            row_ids = [r["row_id"] for r in shard_rows]
            realized_pos = [p for p in positions_wanted if p in shard_rows[0]["positions"]]
            for li in layers:
                pos_arr = np.concatenate(per_layer_pos[li])  # (n, n_pos, H) fp32
                y_arr = np.concatenate(per_layer_y[li])
                ldir = stage_dir / f"L{li:02d}"
                ldir.mkdir(parents=True, exist_ok=True)
                payload = {"row_ids": np.array(row_ids), "y_ans": y_arr.astype(np.float32)}
                if not y_only:
                    for pi, p in enumerate(realized_pos):
                        payload[f"x_{p}"] = pos_arr[:, pi].astype(np.float32)
                savez_atomic(ldir / f"shard{shard_k:03d}.npz", **payload)
            for r in shard_rows:
                rows_meta.append(
                    {
                        "row_id": r["row_id"],
                        "qid": r.get("qid"),
                        "gold": r.get("gold"),
                        "n_prompt_tokens": r["n_prompt_tokens"],
                        "n_ans_tokens": int(r["spans"]["ans"][1] - r["spans"]["ans"][0]),
                        "n_think_tokens": int(
                            r["positions"].get("cot_boundary", len(r["prompt_ids"]))
                            - len(r["prompt_ids"])
                        )
                        if "cot_boundary" in r["positions"]
                        else 0,
                    }
                )
            logger.info(
                "[i2588] [capture %s] shard %d done (%d rows, %d layers)",
                stage,
                shard_k,
                len(shard_rows),
                len(layers),
            )
    finally:
        for h in handles:
            h.remove()
    PC.write_json_atomic(
        stage_dir / "rows.json",
        {
            "meta": _meta(),
            "cell": cell.key,
            "stage": stage,
            "positions": list(positions_wanted),
            "rows": rows_meta,
        },
    )


def phase_capture(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("capture")
    from transformers import AutoTokenizer

    m = cell.model
    layers = _layers_for(args, cell)
    sem = _acquire_capture_slot(paths["root"])
    try:
        tok = AutoTokenizer.from_pretrained(m.hf_id)
        hf = G._load_capture_model(m.hf_id, args.device, "bfloat16")
        for stage in GENERIC_SPLITS:
            _capture_stage(args, cell, paths, hf, tok, stage, layers)
        for seed in PC.CEILING_SEEDS:
            _capture_stage(args, cell, paths, hf, tok, f"ceiling_s{seed}", layers, y_only=True)
        for seed in _gpqa_seeds(args):
            _capture_stage(args, cell, paths, hf, tok, f"gpqa_s{seed}", layers)
        del hf
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    finally:
        sem.close()


def _layers_for(args, cell: PC.Cell) -> list[int]:
    if args.layer_set == "odd":
        assert cell.model.n_layers > 32 and cell.arm == "a", (
            "odd-layer sensitivity pass is registered for 64-layer column endpoints, arm (a)"
        )
        return PC.odd_sensitivity_layers(cell.model.n_layers)
    return PC.sweep_layers(cell.model.n_layers)


# ---------------------------------------------------------------------------
# Phase: upload-raw (before any fit starts; plan phase-order persistence)
# ---------------------------------------------------------------------------


def phase_upload_raw(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("upload_raw")
    prefix = _cell_prefix(args, cell)
    for stage in _stage_names(args, cell):
        stage_dir = paths["raw"] / stage
        if not stage_dir.is_dir() or not any(stage_dir.iterdir()):
            assert not cell.fresh and not stage.startswith("gpqa"), (
                f"fresh-cell raw stage {stage} empty at upload time"
            )
            continue
        _upload_dir(stage_dir, f"{prefix}/raw_completions/{stage}", f"{cell.key} raw {stage}")
    for f in sorted(paths["parsed"].glob("*.json*")):
        _upload_file(f, f"{prefix}/parsed/{f.name}", f"{cell.key} parsed")


def phase_upload_capture(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("upload_capture")
    prefix = _cell_prefix(args, cell)
    tag = "capture" if args.layer_set == "swept" else "capture_oddlayers"
    _upload_dir(
        paths["capture"], f"{prefix}/analysis_tensors/{tag}", f"{cell.key} capture tensors ({tag})"
    )


# ---------------------------------------------------------------------------
# Phase: g2-anchor (anchor pod only) + the fail-closed sentinel await
# ---------------------------------------------------------------------------


def phase_g2_anchor(args, cell: PC.Cell, paths: dict) -> None:
    """Anchor refit at tol=1e-6 (plan §7 G2); publishes the HF sentinel every
    fit stage fail-closes on. Runs the #2330 battery's own store/assembly path."""
    G._phase("g2_anchor")
    dev = MF._resolve_device(args.device)
    mcfg = MF.MODELS["qwen25_7b"]
    store = MF.assemble_store(
        MF.HF_PREFIX_7B,
        mcfg["store_train_split"],
        PC.ANCHOR_LAYER,
        paths["cache"],
        mcfg["store_expected_n"],
    )
    rec = MF._run_anchor_on_store(
        store,
        mcfg,
        dev,
        lambda X, Y, tr, val, te, d: MF.run_anchor_gate(
            X, Y, tr, val, te, d, expected_r2=PC.ANCHOR_EXPECTED_R2, tol=PC.ANCHOR_TOL
        ),
    )
    sentinel = {
        "meta": _meta(),
        "gate": "g2_anchor",
        "status": "PASS",
        "store_revision_pin_recorded": MF.STORE_REVISION_PIN_7B,
        **rec,
    }
    out = paths["fits"] / "g2_anchor_pass.json"
    PC.write_json_atomic(out, sentinel)
    _upload_file(out, PC.G2_SENTINEL_PATH, "G2 anchor PASS sentinel")
    logger.info("[i2588] G2 anchor PASS published -> %s", PC.G2_SENTINEL_PATH)


def _await_g2(args) -> dict:
    """Fail-closed poll for the G2 sentinel (45-min bound -> hard halt)."""
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    deadline = time.time() + (60 if args.smoke else PC.G2_SENTINEL_TIMEOUT_S)
    while True:
        if api.file_exists(PC.HF_DATA_REPO, PC.G2_SENTINEL_PATH, repo_type="dataset"):
            local = hf_hub_download(PC.HF_DATA_REPO, PC.G2_SENTINEL_PATH, repo_type="dataset")
            rec = json.loads(Path(local).read_text(encoding="utf-8"))
            assert rec.get("status") == "PASS", f"G2 sentinel present but not PASS: {rec}"
            return rec
        if time.time() > deadline:
            raise RuntimeError(
                "G2 anchor sentinel absent after the bounded poll "
                f"({PC.G2_SENTINEL_PATH}) — fits are fail-closed on the anchor gate "
                "(plan §7 G2). Run --phase g2-anchor on the anchor pod, or escalate "
                "(must-ask); NEVER skip the gate."
            )
        logger.info("[i2588] awaiting G2 sentinel %s ...", PC.G2_SENTINEL_PATH)
        time.sleep(60)


# ---------------------------------------------------------------------------
# Phase: fits (per swept layer battery; layer_star by VAL cosine acc@1)
# ---------------------------------------------------------------------------


def _load_stage_layer(
    paths: dict, stage: str, layer: int, *, tag: str = "capture"
) -> dict[str, np.ndarray]:
    ldir = paths[tag] / stage / f"L{layer:02d}"
    shards = sorted(ldir.glob("shard*.npz"))
    assert shards, f"no capture shards at {ldir}"
    cols: dict[str, list[np.ndarray]] = {}
    ids: list[np.ndarray] = []
    for f in shards:
        with np.load(f, allow_pickle=False) as z:
            ids.append(z["row_ids"])
            for k in z.files:
                if k != "row_ids":
                    cols.setdefault(k, []).append(z[k])
    out = {k: np.concatenate(v) for k, v in cols.items()}
    out["row_ids"] = np.concatenate(ids)
    order = np.argsort(out["row_ids"])  # deterministic row order across shards
    return {k: v[order] for k, v in out.items()}


def _bundle(paths: dict, layer: int, pos: str) -> dict:
    """(X, Y, tr, val, te) fp64 bundle for one (layer, input-position)."""
    parts_x, parts_y, idx, n = [], [], {}, 0
    for split, key in (("train_10k", "tr"), ("val_400", "val"), ("test_1000", "te")):
        d = _load_stage_layer(paths, split, layer)
        parts_x.append(d[f"x_{pos}"])
        parts_y.append(d["y_ans"])
        idx[key] = np.arange(n, n + d["y_ans"].shape[0], dtype=np.int64)
        n += d["y_ans"].shape[0]
    return {
        "X": np.concatenate(parts_x).astype(np.float64),
        "Y": np.concatenate(parts_y).astype(np.float64),
        "tr": idx["tr"],
        "val": idx["val"],
        "te": idx["te"],
    }


def _fit_edge_extended_with_val(X, Y, tr, val, te, dev) -> tuple[np.ndarray, np.ndarray, dict]:
    """MF.fit_ridge_edge_extended's exact selection + extension policy, ALSO
    returning the VAL predictions at the selected lambda (needed for the
    registered layer_star = argmax VAL cosine acc@1 rule; MF's wrapper only
    emits test predictions). Estimator-parity note: same F fit core, same
    MF._extended_lambdas one-decade extension, same MAX_GRID_EXTENSIONS cap;
    the only diff is the extra pred_val emission."""
    grid = np.array(LF.LAMBDAS, dtype=np.float64)
    extensions = 0
    while True:
        pred_te, meta, payload = F.fit_ridge_with_weights(
            X, Y, tr, val, te, grid, dev, LF.RIDGE_BLOCK
        )
        edge = meta.get("lambda_grid_edge")
        if edge is None or extensions >= MF.MAX_GRID_EXTENSIONS:
            if edge is not None:
                raise RuntimeError(
                    f"lambda still at grid edge {edge!r} after {extensions} one-decade "
                    "extensions — fail loud (MF.MAX_GRID_EXTENSIONS policy)"
                )
            break
        grid = MF._extended_lambdas(grid, edge)
        extensions += 1
    meta["grid_extensions"] = extensions
    lam = payload["selected_lambda"]
    xs = (torch.as_tensor(X[val], dtype=torch.float64) - payload["xmu"].double()) / payload[
        "xsd"
    ].double()
    pred_val = (xs @ payload["W"].double() + payload["ymu"].double()).numpy()
    meta["W_payload"] = payload  # consumed by gpqa-transfer / resid at layer_star
    assert np.isclose(lam, meta["selected_lambda"]), (lam, meta["selected_lambda"])
    return pred_te, pred_val, meta


def _perrow_hits_cos(pred: np.ndarray, true: np.ndarray) -> dict:
    """Per-row cosine retrieval hit@1 + mid-rank, mirroring the exact tie
    convention of analysis/mapping_baselines.knn_retrieval (tolerance-based
    mid-ranks) so the aggregate acc@1 here equals the reported knn read."""
    pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12)
    tn = true / (np.linalg.norm(true, axis=1, keepdims=True) + 1e-12)
    d = 1.0 - pn @ tn.T
    n = d.shape[0]
    d_true = d[np.arange(n), np.arange(n)]
    tol = 1e-9 * np.maximum(np.abs(d_true)[:, None], 1e-12)
    closer = (d < d_true[:, None] - tol).sum(axis=1)
    tied = (np.abs(d - d_true[:, None]) <= tol).sum(axis=1) - 1
    ranks = 1.0 + closer + 0.5 * tied
    return {"hit1": [int(r <= 1) for r in ranks], "rank": [float(r) for r in ranks]}


def _two_draw_ceiling(ya: np.ndarray, yb: np.ndarray) -> dict:
    """Variance-weighted per-dim two-draw Pearson ceiling (the LF._reliability_
    ceiling formula — Σ_d Var_d·r_d / Σ_d Var_d with Var_d = variance of the
    two-draw mean — computed over THIS cell's fresh ceiling captures)."""
    assert ya.shape == yb.shape, (ya.shape, yb.shape)
    a = ya - ya.mean(axis=0)
    b = yb - yb.mean(axis=0)
    denom = np.sqrt((a * a).sum(axis=0) * (b * b).sum(axis=0)) + 1e-30
    r_d = (a * b).sum(axis=0) / denom
    var_d = ((ya + yb) / 2.0).var(axis=0)
    w = var_d / (var_d.sum() + 1e-30)
    return {
        "ceiling": float((w * r_d).sum()),
        "n": int(ya.shape[0]),
        "r_d_median": float(np.median(r_d)),
    }


def _ceiling_alignment(paths: dict, layer: int) -> dict | None:
    da = _load_stage_layer(paths, f"ceiling_s{PC.CEILING_SEEDS[0]}", layer)
    db = _load_stage_layer(paths, f"ceiling_s{PC.CEILING_SEEDS[1]}", layer)
    # ceiling row_ids carry the seed tag ("ceiling_s43_<ci>"); align on the ci suffix
    ci_a = np.array([str(r).rsplit("_", 1)[1] for r in da["row_ids"]])
    ci_b = np.array([str(r).rsplit("_", 1)[1] for r in db["row_ids"]])
    common, ia, ib = np.intersect1d(ci_a, ci_b, return_indices=True)
    if common.size < 2:
        return None
    return _two_draw_ceiling(da["y_ans"][ia].astype(np.float64), db["y_ans"][ib].astype(np.float64))


def phase_fits(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("fits")
    g2 = _await_g2(args)
    dev = MF._resolve_device(args.device)
    layers = _layers_for(args, cell)
    for pos in cell.input_positions:
        per_layer: dict[int, dict] = {}
        for layer in layers:
            b = _bundle(paths, layer, pos)
            if args.smoke and len(b["tr"]) < b["X"].shape[1]:
                logger.info(
                    "[i2588] SMOKE under-determined fit (n_train=%d < d=%d) — "
                    "smoke-shape only, never a signal read",
                    len(b["tr"]),
                    b["X"].shape[1],
                )
            pred_te, pred_val, meta = _fit_edge_extended_with_val(
                b["X"], b["Y"], b["tr"], b["val"], b["te"], dev
            )
            payload = meta.pop("W_payload")
            floors = LF._fit_floors(b["X"], b["Y"], b["tr"], b["val"], b["te"], dev, LF.RIDGE_BLOCK)
            preds = {"ridge": pred_te}
            preds.update({k: v["pred_te"] for k, v in floors.items()})
            knn_te = LF._knn_reads(preds, b["Y"][b["te"]])
            knn_val = LF._knn_reads({"ridge": pred_val}, b["Y"][b["val"]])
            per_layer[layer] = {
                "test_r2": float(LF._pooled_r2(pred_te, b["Y"][b["te"]])),
                "floors_test_r2": {k: float(v["test_r2"]) for k, v in floors.items()},
                "knn_test": knn_te,
                "knn_val": knn_val,
                "fit_meta": {
                    k: v
                    for k, v in meta.items()
                    if isinstance(v, (int, float, str, bool, type(None)))
                },
                "n": {k: int(len(b[k])) for k in ("tr", "val", "te")},
                "d": int(b["X"].shape[1]),
            }
            if layer == layers[0]:
                logger.info("[i2588] [fits %s pos=%s] first layer done", cell.key, pos)
            # checkpoint-per-unit: persist the per-layer record the moment it lands
            PC.write_json_atomic(
                paths["fits"] / f"percell_{pos}_L{layer:02d}.json",
                {"meta": _meta(), **per_layer[layer]},
            )
            logger.info(
                "[fits] unit L%02d/%s pos=%s val_acc1_cos=%.4f test_r2=%.4f",
                layer,
                cell.key,
                pos,
                knn_val["ridge"]["cosine"]["acc_at_k"][1],
                per_layer[layer]["test_r2"],
            )
            # persist the selected-λ payload only at layer_star (below); free here
            del payload
        star = max(
            per_layer, key=lambda li: per_layer[li]["knn_val"]["ridge"]["cosine"]["acc_at_k"][1]
        )
        ceiling = _ceiling_alignment(paths, star)
        # Per-row TEST hit indicators at layer_star (the P3 paired bootstrap's
        # input — plan §4.4: 1,000 draws seed 42 over paired per-row hits).
        b_star = _bundle(paths, star, pos)
        pred_te_star, _pv, _m = _fit_edge_extended_with_val(
            b_star["X"], b_star["Y"], b_star["tr"], b_star["val"], b_star["te"], dev
        )
        _m.pop("W_payload", None)
        te_ids = _load_stage_layer(paths, "test_1000", star)["row_ids"]
        hits = _perrow_hits_cos(pred_te_star, b_star["Y"][b_star["te"]])
        PC.write_json_atomic(
            paths["fits"] / f"perrow_{pos}.json",
            {
                "meta": _meta(),
                "cell": cell.key,
                "input_position": pos,
                "layer_star": int(star),
                "row_ids": [str(r) for r in te_ids],
                "hit1_cos": hits["hit1"],
                "rank_cos": hits["rank"],
            },
        )
        record = {
            "meta": _meta(),
            "cell": cell.key,
            "input_position": pos,
            "layer_set": args.layer_set,
            "smoke": bool(args.smoke),
            "g2_anchor": {k: g2[k] for k in ("realized_r2", "abs_deviation", "tol")},
            "layers": {str(k): v for k, v in per_layer.items()},
            "layer_star": int(star),
            "layer_star_rule": "argmax over swept layers of VAL retrieval acc@1 (cosine)",
            "ceiling_two_draw_at_star": ceiling,
        }
        PC.write_json_atomic(paths["fits"] / f"fits_{pos}.json", record)
        logger.info("[i2588] [fits %s pos=%s] layer_star=%d", cell.key, pos, star)


# ---------------------------------------------------------------------------
# Phase: nulls (P=200 shuffled-pairing draws, 20-draw blocks, ONE eigh reused)
# ---------------------------------------------------------------------------


def _null_battery(
    X, Y, tr, val, te, dev, *, draws: int, block_draws: int, seed: int, y_true_te: np.ndarray
) -> list[dict]:
    """Permutation battery: shuffled TRAIN pairing, per-draw val-based lambda
    re-selection over the 23-lambda grid, batched in ``block_draws`` blocks
    reusing ONE eigendecomposition (permuting Y changes only X^T Y — never
    A = X^T X, so U and s_eig are computed once; plan §4.4)."""
    fac = F._ridge_factorize(X, Y, tr, dev, LF.RIDGE_BLOCK)
    U, s_eig = fac["U"], fac["s_eig"]
    xmu, xsd, ymu = fac["xmu"], fac["xsd"], fac["ymu"]
    grid = torch.as_tensor(np.array(LF.LAMBDAS, dtype=np.float64), device=dev)
    Xtr = (torch.as_tensor(X[tr], dtype=torch.float64, device=dev) - xmu) / xsd
    Ytr = torch.as_tensor(Y[tr], dtype=torch.float64, device=dev) - ymu
    Bval = ((torch.as_tensor(X[val], dtype=torch.float64, device=dev) - xmu) / xsd) @ U
    Bte = ((torch.as_tensor(X[te], dtype=torch.float64, device=dev) - xmu) / xsd) @ U
    Yval = torch.as_tensor(Y[val], dtype=torch.float64, device=dev)
    sst_val = float(((Yval - Yval.mean(dim=0)) ** 2).sum())
    rng = np.random.default_rng(seed)
    out: list[dict] = []
    for d0 in range(0, draws, block_draws):
        nb = min(block_draws, draws - d0)
        perms = [rng.permutation(len(tr)) for _ in range(nb)]
        for j in range(nb):  # one (d,n)@(n,H) GEMM per draw; eigh NEVER recomputed
            UtXtY = U.T @ (Xtr.T @ Ytr[perms[j]])
            best_lam, best_vr2 = None, -np.inf
            for lam in grid.tolist():
                pv = Bval @ (UtXtY / (s_eig + lam)[:, None]) + ymu
                vr2 = 1.0 - float(((Yval - pv) ** 2).sum()) / (sst_val + 1e-30)
                if np.isfinite(vr2) and vr2 > best_vr2:
                    best_vr2, best_lam = vr2, float(lam)
            pt = (Bte @ (UtXtY / (s_eig + best_lam)[:, None]) + ymu).cpu().numpy()
            knn = LF._knn_reads({"null": pt}, y_true_te)["null"]
            edge = (
                "low"
                if np.isclose(best_lam, float(grid[0]))
                else "high"
                if np.isclose(best_lam, float(grid[-1]))
                else None
            )
            out.append(
                {
                    "draw": d0 + j,
                    "selected_lambda": best_lam,
                    "grid_edge": edge,
                    "val_r2": float(best_vr2),
                    "test_r2": float(LF._pooled_r2(pt, y_true_te)),
                    "acc1_cos": knn["cosine"]["acc_at_k"][1],
                    "acc1_euc": knn["euclidean"]["acc_at_k"][1],
                    "mrr_cos": knn["cosine"]["mrr"],
                }
            )
        logger.info("[nulls] unit %d/%d draws done", min(d0 + nb, draws), draws)
    return out


def phase_nulls(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("nulls")
    dev = MF._resolve_device(args.device)
    draws = SMOKE_PERM_DRAWS if args.smoke else PC.PERM_DRAWS
    for pos in cell.input_positions:
        fits = json.loads((paths["fits"] / f"fits_{pos}.json").read_text(encoding="utf-8"))
        star = int(fits["layer_star"])
        b = _bundle(paths, star, pos)
        rows = _null_battery(
            b["X"],
            b["Y"],
            b["tr"],
            b["val"],
            b["te"],
            dev,
            draws=draws,
            block_draws=PC.PERM_DRAW_BLOCK,
            seed=PC.PERM_SEED,
            y_true_te=b["Y"][b["te"]],
        )
        accs = np.array([r["acc1_cos"] for r in rows])
        PC.write_json_atomic(
            paths["fits"] / f"nulls_{pos}.json",
            {
                "meta": _meta(),
                "cell": cell.key,
                "input_position": pos,
                "layer_star": star,
                "perm_draws": draws,
                "perm_seed": PC.PERM_SEED,
                "advisory_only": True,
                "null_mean_acc1_cos": float(accs.mean()),
                "null_sd_acc1_cos": float(accs.std()),
                "draws_detail": rows,
            },
        )
        logger.info(
            "[i2588] [nulls %s pos=%s] mean acc1_cos=%.5f sd=%.5f",
            cell.key,
            pos,
            accs.mean(),
            accs.std(),
        )


# ---------------------------------------------------------------------------
# Phase: gpqa-transfer (frozen map applied; NEVER fitted on GPQA)
# ---------------------------------------------------------------------------


def _refit_star_payload(paths: dict, pos: str, star: int, dev) -> dict:
    b = _bundle(paths, star, pos)
    grid = np.array(LF.LAMBDAS, dtype=np.float64)
    _pred, _meta, payload = F.fit_ridge_with_weights(
        b["X"], b["Y"], b["tr"], b["val"], b["te"], grid, dev, LF.RIDGE_BLOCK
    )
    return payload


def phase_gpqa_transfer(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("gpqa_transfer")
    dev = MF._resolve_device(args.device)
    seeds = _gpqa_seeds(args)
    behavioral = _gpqa_behavioral(args, cell, paths, seeds)
    for pos in cell.input_positions:
        fits = json.loads((paths["fits"] / f"fits_{pos}.json").read_text(encoding="utf-8"))
        star = int(fits["layer_star"])
        payload = _refit_star_payload(paths, pos, star, dev)
        xs, ys, qids = [], [], []
        for seed in seeds:
            d = _load_stage_layer(paths, f"gpqa_s{seed}", star)
            rows_meta = json.loads(
                (paths["capture"] / f"gpqa_s{seed}" / "rows.json").read_text(encoding="utf-8")
            )
            qid_by_row = {r["row_id"]: r["qid"] for r in rows_meta["rows"]}
            xs.append(d[f"x_{pos}"])
            ys.append(d["y_ans"])
            qids.extend(qid_by_row[r] for r in d["row_ids"])
        Xg = np.concatenate(xs).astype(np.float64)
        Yg = np.concatenate(ys).astype(np.float64)
        pred = F.apply_map(payload, Xg, torch.device(dev))
        qarr = np.array(qids)
        same_q = qarr[:, None] == qarr[None, :]
        pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12)
        tn = Yg / (np.linalg.norm(Yg, axis=1, keepdims=True) + 1e-12)
        sim = pn @ tn.T
        nn = sim.argmax(axis=1)
        same_q_acc1 = float(same_q[np.arange(len(nn)), nn].mean())
        exact_acc1 = float((nn == np.arange(len(nn))).mean())
        n_pool = int(Yg.shape[0])
        n_per_q = int(len(seeds))
        PC.write_json_atomic(
            paths["fits"] / f"gpqa_transfer_{pos}.json",
            {
                "meta": _meta(),
                "cell": cell.key,
                "input_position": pos,
                "layer_star": star,
                "transfer_only": True,
                "n_rows": n_pool,
                "same_question_acc1_cos": same_q_acc1,
                "same_question_chance": float(n_per_q / n_pool),
                "exact_row_acc1_cos": exact_acc1,
                "exact_row_chance": float(1.0 / n_pool),
                "behavioral": behavioral,
            },
        )
        logger.info(
            "[i2588] [gpqa %s pos=%s] same-q acc@1=%.4f (chance %.4f) exact=%.4f",
            cell.key,
            pos,
            same_q_acc1,
            n_per_q / n_pool,
            exact_acc1,
        )


def _gpqa_behavioral(args, cell: PC.Cell, paths: dict, seeds) -> dict:
    """Behavioral GPQA accuracy (exact-match letter extraction) + the §4.5
    judge-fallback trigger accounting. The judge fallback itself is a
    VM-side conditional stage (issue2588_trend.py --judge-fallback) routed
    through api_dispatch; the pod driver only persists the pending rows."""
    total, correct, unparseable = 0, 0, 0
    pending: list[dict] = []
    for seed in seeds:
        for r in PC.read_jsonl(paths["parsed"] / f"gpqa_s{seed}.jsonl"):
            s, e = r["ans_char_span"]
            ans_text = r["text"][s:e]
            ok, letter = PC.gpqa_letter_correct(ans_text, r["gold"])
            total += 1
            correct += int(ok)
            if letter is None:
                unparseable += 1
                pending.append({"row_id": r["row_id"], "qid": r["qid"], "gold": r["gold"]})
    frac_unparseable = unparseable / max(1, total)
    if frac_unparseable > PC.GPQA_EXTRACTION_FAIL_TRIGGER:
        PC.write_json_atomic(
            paths["fits"] / "gpqa_judge_pending.json",
            {
                "meta": _meta(),
                "cell": cell.key,
                "rows": pending,
                "frac_unparseable": frac_unparseable,
                "judge_model": PC.EXTRACTION_JUDGE_MODEL,
            },
        )
        logger.warning(
            "[i2588] GPQA extraction-fail %.3f > %.2f — judge fallback FLAGGED "
            "(rows persisted; VM-side api_dispatch stage runs the judge)",
            frac_unparseable,
            PC.GPQA_EXTRACTION_FAIL_TRIGGER,
        )
    return {
        "n_rollouts": total,
        "acc_exact_match": correct / max(1, total),
        "frac_unparseable": frac_unparseable,
        "judge_fallback_flagged": frac_unparseable > PC.GPQA_EXTRACTION_FAIL_TRIGGER,
    }


# ---------------------------------------------------------------------------
# Phase: resid (registered length-residualization protocol + length-only floor)
# ---------------------------------------------------------------------------


def _length_covariates(paths: dict, stage: str, row_ids: np.ndarray, arm: str) -> np.ndarray:
    rows_meta = json.loads((paths["capture"] / stage / "rows.json").read_text(encoding="utf-8"))[
        "rows"
    ]
    by_id = {r["row_id"]: r for r in rows_meta}
    cols = []
    for rid in row_ids:
        r = by_id[str(rid)]
        c = [math.log(max(1, r["n_prompt_tokens"])), math.log(max(1, r["n_ans_tokens"]))]
        if arm == "b":
            c.append(math.log(max(1, r["n_think_tokens"])))
        cols.append(c)
    return np.asarray(cols, dtype=np.float64)


def phase_resid(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("resid")
    dev = MF._resolve_device(args.device)
    for pos in cell.input_positions:
        fits = json.loads((paths["fits"] / f"fits_{pos}.json").read_text(encoding="utf-8"))
        star = int(fits["layer_star"])
        data, Z = {}, {}
        for split in GENERIC_SPLITS:
            d = _load_stage_layer(paths, split, star)
            data[split] = d
            Z[split] = _length_covariates(paths, split, d["row_ids"], cell.arm)

        def _aug(z: np.ndarray) -> np.ndarray:
            return np.concatenate([np.ones((z.shape[0], 1)), z], axis=1)

        Ztr = _aug(Z["train_10k"])
        # Per-dim OLS coefficients fitted on TRAIN ONLY, applied out-of-fold
        # unchanged (registered protocol; #2546/#2502 lineage).
        coefs = {}
        for name, key in (("x", f"x_{pos}"), ("y", "y_ans")):
            M = data["train_10k"][key].astype(np.float64)
            B, *_ = np.linalg.lstsq(Ztr, M, rcond=None)
            coefs[name] = B

        def _resid(split: str, key: str, which: str) -> np.ndarray:
            return data[split][key].astype(np.float64) - _aug(Z[split]) @ coefs[which]

        parts_x = [_resid(s, f"x_{pos}", "x") for s in GENERIC_SPLITS]
        parts_y = [_resid(s, "y_ans", "y") for s in GENERIC_SPLITS]
        idx, n = {}, 0
        for split, key in (("train_10k", "tr"), ("val_400", "val"), ("test_1000", "te")):
            m = data[split]["y_ans"].shape[0]
            idx[key] = np.arange(n, n + m, dtype=np.int64)
            n += m
        Xr, Yr = np.concatenate(parts_x), np.concatenate(parts_y)
        pred_te, _pred_val, meta = _fit_edge_extended_with_val(
            Xr, Yr, idx["tr"], idx["val"], idx["te"], dev
        )
        meta.pop("W_payload", None)
        knn = LF._knn_reads({"ridge_resid": pred_te}, Yr[idx["te"]])
        # Length-only floor: predict RAW Y from the covariates alone.
        B_len, *_ = np.linalg.lstsq(Ztr, data["train_10k"]["y_ans"].astype(np.float64), rcond=None)
        pred_len = _aug(Z["test_1000"]) @ B_len
        y_te_raw = data["test_1000"]["y_ans"].astype(np.float64)
        knn_len = LF._knn_reads({"length_only": pred_len}, y_te_raw)
        PC.write_json_atomic(
            paths["fits"] / f"resid_{pos}.json",
            {
                "meta": _meta(),
                "cell": cell.key,
                "input_position": pos,
                "layer_star": star,
                "covariates": ["log_prompt_tokens", "log_answer_tokens"]
                + (["log_think_tokens"] if cell.arm == "b" else []),
                "resid_test_r2": float(LF._pooled_r2(pred_te, Yr[idx["te"]])),
                "resid_knn_test": knn,
                "length_only_test_r2": float(LF._pooled_r2(pred_len, y_te_raw)),
                "length_only_knn_test": knn_len,
                "fit_meta": {
                    k: v
                    for k, v in meta.items()
                    if isinstance(v, (int, float, str, bool, type(None)))
                },
            },
        )
        logger.info(
            "[i2588] [resid %s pos=%s] resid acc1_cos=%.4f length-only=%.4f",
            cell.key,
            pos,
            knn["ridge_resid"]["cosine"]["acc_at_k"][1],
            knn_len["length_only"]["cosine"]["acc_at_k"][1],
        )


# ---------------------------------------------------------------------------
# Phase: upload-fits + sentinel
# ---------------------------------------------------------------------------


def phase_upload_fits(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("upload_fits")
    prefix_fits = f"{PC.PANEL_PREFIX}/{'smoke/' if args.smoke else ''}fits/{cell.key}"
    prefix_nulls = f"{PC.PANEL_PREFIX}/{'smoke/' if args.smoke else ''}nulls/{cell.key}"
    for f in sorted(paths["fits"].glob("*.json")):
        target = prefix_nulls if f.name.startswith("nulls_") else prefix_fits
        _upload_file(f, f"{target}/{f.name}", f"{cell.key} {f.name}")
    sentinel = {
        "eval_numbers": _sentinel_numbers(cell, paths),
        "eval_paths": [str(p) for p in sorted(paths["fits"].glob("*.json"))],
        "reproducibility_card": _meta(),
        "wandb_url": None,
        "hf_hub_url": f"https://huggingface.co/datasets/{PC.HF_DATA_REPO}/tree/main/{prefix_fits}",
        "worktree_path": str(_REPO_ROOT),
        "final_commit_sha": G._git_sha(),
        "gpu_hours_used": None,
        "gpu_hours_budgeted": None,
        "plan_deviations": [],
    }
    out = paths["logs"] / f"issue-2588-{cell.key}-results.json"
    PC.write_json_atomic(out, sentinel)
    logger.info("[phase=done] cell %s complete rc=0 (sentinel %s)", cell.key, out)


def _sentinel_numbers(cell: PC.Cell, paths: dict) -> dict:
    nums: dict = {}
    for pos in cell.input_positions:
        f = paths["fits"] / f"fits_{pos}.json"
        if f.exists():
            rec = json.loads(f.read_text(encoding="utf-8"))
            star = str(rec["layer_star"])
            nums[f"{pos}_layer_star"] = rec["layer_star"]
            nums[f"{pos}_test_acc1_cos_at_star"] = (
                rec["layers"][star]["knn_test"]["ridge"]["cosine"]["acc_at_k"]["1"]
                if "1" in rec["layers"][star]["knn_test"]["ridge"]["cosine"]["acc_at_k"]
                else rec["layers"][star]["knn_test"]["ridge"]["cosine"]["acc_at_k"][1]
            )
    return nums


def phase_purge_model_cache(args, cell: PC.Cell, paths: dict) -> None:
    """Free the model's HF snapshot after its LAST cell (plan §9 disk budget)."""
    G._phase("purge_model_cache")
    import shutil

    hub = Path(os.environ.get("HF_HOME", str(Path.home() / ".cache/huggingface"))) / "hub"
    slug = "models--" + cell.model.hf_id.replace("/", "--")
    target = hub / slug
    if target.is_dir():
        shutil.rmtree(target)
        logger.info("[i2588] purged model cache %s", target)
    else:
        logger.info("[i2588] model cache absent (nothing to purge): %s", target)


# ---------------------------------------------------------------------------
# Smoke-only synthetic null timing probe (production shape; plan smoke phase)
# ---------------------------------------------------------------------------


def phase_smoke_null_timing(args, cell: PC.Cell, paths: dict) -> None:
    """Time ONE 20-draw batched null block at PRODUCTION shape on synthetic
    data (n=10,000 x d in {1024, 5120}) — the plan's G3-adjacent wall basis."""
    G._phase("smoke_null_timing")
    dev = MF._resolve_device(args.device)
    rng = np.random.default_rng(0)
    out = {}
    for d_dim in (1024, 5120):
        n_tr, n_val, n_te = 10_000, 400, 1_000
        X = rng.standard_normal((n_tr + n_val + n_te, d_dim))
        Y = rng.standard_normal((n_tr + n_val + n_te, d_dim))
        tr = np.arange(0, n_tr)
        val = np.arange(n_tr, n_tr + n_val)
        te = np.arange(n_tr + n_val, n_tr + n_val + n_te)
        t0 = time.time()
        _null_battery(
            X,
            Y,
            tr,
            val,
            te,
            dev,
            draws=PC.PERM_DRAW_BLOCK,
            block_draws=PC.PERM_DRAW_BLOCK,
            seed=0,
            y_true_te=Y[te],
        )
        out[f"d{d_dim}"] = {"wall_s_20_draws": time.time() - t0}
        logger.info(
            "[i2588] smoke null timing d=%d: %.1fs / 20 draws",
            d_dim,
            out[f"d{d_dim}"]["wall_s_20_draws"],
        )
    PC.write_json_atomic(
        paths["fits"] / "smoke_null_timing.json", {"meta": _meta(), "device": str(dev), **out}
    )


# ---------------------------------------------------------------------------
# Phase registry + CLI
# ---------------------------------------------------------------------------

PHASES: dict = {
    "prologue": phase_prologue,
    "stage": phase_stage,
    "gen": phase_gen,
    "parse": phase_parse,
    "capture": phase_capture,
    "upload-raw": phase_upload_raw,
    "upload-capture": phase_upload_capture,
    "g2-anchor": phase_g2_anchor,
    "fits": phase_fits,
    "nulls": phase_nulls,
    "gpqa-transfer": phase_gpqa_transfer,
    "resid": phase_resid,
    "upload-fits": phase_upload_fits,
    "purge-model-cache": phase_purge_model_cache,
    "smoke-null-timing": phase_smoke_null_timing,
}

_ALL_SEQUENCE = (
    "prologue",
    "stage",
    "gen",
    "parse",
    "capture",
    "upload-raw",
    "upload-capture",
    "fits",
    "nulls",
    "gpqa-transfer",
    "resid",
    "upload-fits",
)


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        description=__doc__.replace("%", "%%"), formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--cell", help="cell key <model>_<arm> (see --list-cells)")
    ap.add_argument("--phase", default="all", choices=["all", *PHASES.keys()])
    ap.add_argument("--out-root", default="/workspace/eps2588")
    ap.add_argument(
        "--smoke",
        action="store_true",
        help="production dispatcher at tiny N; uploads to the smoke/ prefix",
    )
    ap.add_argument(
        "--layer-set",
        default="swept",
        choices=["swept", "odd"],
        help="odd = 64-layer odd-layer sensitivity pass (endpoint arm-a cells)",
    )
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--gpu-count", type=int, default=1)
    ap.add_argument("--capture-batch-size", type=int, default=8)
    ap.add_argument(
        "--pod-ordinal",
        type=int,
        default=0,
        help="jittered weight pulls: sleep ordinal*120s before staging",
    )
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--list-phases", action="store_true")
    ap.add_argument("--list-cells", action="store_true")
    return ap


def _run_import_check() -> int:
    """Axis-1 import resolution: execute EVERY deferred import + argcheck."""
    from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

    assert_args_attributes_defined(__file__)
    import fcntl  # noqa: F401
    import shutil  # noqa: F401

    from huggingface_hub import HfApi, hf_hub_download  # noqa: F401
    from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: F401

    from explore_persona_space.atomic_io import savez_atomic, write_json_atomic  # noqa: F401
    from explore_persona_space.eval.utils import parse_judge_json  # noqa: F401

    try:
        from vllm import LLM, SamplingParams, TokensPrompt  # noqa: F401
    except ImportError:
        print(
            "[import-check] vllm not installed on this box (CPU smoke) — gen phase "
            "unavailable here; all other deferred imports resolved"
        )
    for name in (
        "fit_ridge",
        "fit_ridge_with_weights",
        "_ridge_factorize",
        "_ridge_predict_one",
        "apply_map",
    ):
        assert hasattr(F, name), name
    for name in ("LAMBDAS", "RIDGE_BLOCK", "_fit_floors", "_knn_reads", "_pooled_r2"):
        assert hasattr(LF, name), name
    for name in (
        "assemble_store",
        "run_anchor_gate",
        "_run_anchor_on_store",
        "_extended_lambdas",
        "MAX_GRID_EXTENSIONS",
        "_resolve_device",
    ):
        assert hasattr(MF, name), name
    for name in (
        "_download_manifest_split",
        "_load_split_ids",
        "_subset_rows",
        "_remote_index",
        "_hub_download",
        "_retry_transient",
        "_render_prompt",
        "_load_capture_model",
        "_resolve_decoder_blocks",
        "_logits_to_keep_kwargs",
        "_reap_vllm_engine",
        "_is_empty_response",
        "store_subpath_for_split",
        "_git_sha",
        "_phase",
        "VLLM_CHUNK_SIZE",
    ):
        assert hasattr(G, name), name
    print("[import-check] OK: argcheck + all deferred imports resolved")
    return 0


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)
    if args.import_check:
        return _run_import_check()
    if args.list_phases:
        print("\n".join(PHASES))
        return 0
    if args.list_cells:
        print("\n".join(c.key for c in PC.all_cells()))
        return 0
    assert args.cell, "--cell is required for run phases"
    cell = PC.cell_by_key(args.cell)
    paths = _paths(args, cell)
    seq = _ALL_SEQUENCE if args.phase == "all" else (args.phase,)
    if args.phase == "all" and args.smoke:
        seq = (*seq, "smoke-null-timing")
    for name in seq:
        PHASES[name](args, cell, paths)
    return 0


if __name__ == "__main__":
    rc = main()
    if _ENGINE_CONSTRUCTED:
        # vLLM worker-subprocess teardown gotcha: never run interpreter exit
        # hooks with a dead engine's workers half-reaped (gotchas.md).
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(rc)
    sys.exit(rc)
