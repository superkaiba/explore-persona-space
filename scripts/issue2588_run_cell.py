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

from explore_persona_space.orchestrate import hub as HUB  # noqa: E402  (guarded+retried uploads)

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
        "capture_oddlayers": cell_dir / "capture_oddlayers",  # C3: odd pass never overwrites
        "fits": cell_dir / "fits",
        "cache": root / "hf_cache",
        "logs": Path("/workspace/logs") if Path("/workspace").is_dir() else root / "logs",
    }
    for v in d.values():
        v.mkdir(parents=True, exist_ok=True)
    return d


def _upload_dir(local_dir: Path, path_in_repo: str, what: str) -> None:
    """One bulk upload_folder commit via the guarded canonical helper (B6, review
    round 2): hub._upload carries the transient retry, the dir-filecount guard,
    the file-count overflow fallback, and the exact-set post-upload verify —
    never a bare per-file loop (gotchas.md 504-storm) or an unanchored wrap."""
    url = HUB._upload(local_dir, PC.HF_DATA_REPO, "dataset", path_in_repo, raise_on_error=True)
    if not url:
        raise RuntimeError(f"upload returned no path for {what} ({local_dir} -> {path_in_repo})")
    logger.info("[i2588] uploaded %s -> %s (%s: %s)", local_dir, path_in_repo, what, url)


def _upload_file(local: Path, path_in_repo: str, what: str) -> None:
    """Single-file upload via hub._upload (upload_as_file=True; full destination
    path — the #595/#1738 contract), retried + verified, fail-loud (return
    checked: '' from _upload is a silent durability loss, upload-policy.md)."""
    url = HUB._upload(
        local, PC.HF_DATA_REPO, "dataset", path_in_repo, upload_as_file=True, raise_on_error=True
    )
    if not url:
        raise RuntimeError(f"upload returned no path for {what} ({local} -> {path_in_repo})")
    logger.info("[i2588] uploaded file %s -> %s (%s)", local, path_in_repo, what)


# ---------------------------------------------------------------------------
# Phase completion sentinels (B1, review round 2): every phase run through the
# main() loop is idempotent — a completed (phase, layer_set) writes a done
# sentinel; a re-run skips it unless --force. Smoke and production never
# collide (distinct cell_dir roots); the odd-layer sensitivity pass carries a
# distinct sentinel (and distinct artifact names, C3).
# ---------------------------------------------------------------------------


def _phase_done_path(args, paths: dict, name: str) -> Path:
    suffix = "_odd" if args.layer_set == "odd" else ""
    return paths["cell"] / "phase_done" / f"{name}{suffix}.json"


def _phase_complete(args, paths: dict, name: str) -> bool:
    return (not args.force) and _phase_done_path(args, paths, name).exists()


def _mark_phase_done(args, cell: PC.Cell, paths: dict, name: str) -> None:
    p = _phase_done_path(args, paths, name)
    p.parent.mkdir(parents=True, exist_ok=True)
    PC.write_json_atomic(
        p,
        {
            "meta": _meta(),
            "cell": cell.key,
            "phase": name,
            "layer_set": args.layer_set,
            "smoke": bool(args.smoke),
        },
    )


def _run_phases(args, cell: PC.Cell, paths: dict, seq: tuple[str, ...]) -> list[str]:
    """Run the requested phases with sentinel skip (B1). Returns names RUN."""
    ran: list[str] = []
    for name in seq:
        if _phase_complete(args, paths, name):
            logger.info(
                "[i2588] phase %s already complete (sentinel %s) — skipped (--force to re-run)",
                name,
                _phase_done_path(args, paths, name),
            )
            continue
        PHASES[name](args, cell, paths)
        _mark_phase_done(args, cell, paths, name)
        ran.append(name)
    return ran


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
    import fcntl

    if args.pod_ordinal > 0:
        wait_s = args.pod_ordinal * 120
        logger.info(
            "[i2588] jittered weight pull: pod_ordinal=%d -> sleep %ds", args.pod_ordinal, wait_s
        )
        time.sleep(wait_s)
    # Registered staging flock (plan §9 MF6): jitter staggers pods, the flock
    # SERIALIZES concurrent staging downloads on a shared out-root (review
    # round 2, F item — jitter alone was implemented).
    lock_fh = open(paths["root"] / ".staging.lock", "w")  # noqa: SIM115 — lock lifetime object
    fcntl.flock(lock_fh, fcntl.LOCK_EX)
    try:
        generic = _load_generic_rows(args, paths["cache"])
        gpqa = _load_gpqa_prompts(args)
        logger.info(
            "[i2588] staged generic=%s gpqa=%d", {k: len(v) for k, v in generic.items()}, len(gpqa)
        )
        if not cell.fresh:
            _stage_banked(args, cell, paths)
    finally:
        fcntl.flock(lock_fh, fcntl.LOCK_UN)
        lock_fh.close()


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

    Round 3 (gen-capture-stage-resume): a stage whose terminal artifact
    (cap_hit_report.json — written LAST, after every chunk) is already on
    disk is skipped (--force re-runs), so a mid-phase crash resumes at the
    first incomplete stage instead of regenerating completed ones.
    """
    report_p = paths["raw"] / stage / "cap_hit_report.json"
    if report_p.exists() and not args.force:
        logger.info(
            "[i2588] [%s] cap_hit_report.json present — stage already generated; "
            "skipped (--force to re-run)",
            stage,
        )
        return _iter_stage_rows(paths, stage)
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

    # D2: raw-text budget (~2 GB, §9) + the model snapshot the engine pulls
    # into the out-root HF cache (conservative: counted even if already cached).
    _assert_headroom(paths, 4.0 + _est_model_gb(cell), f"gen:{cell.key}")
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
# C3 (review round 2): odd-layer sensitivity pass artifact separation. The odd
# pass reads/writes its OWN capture tag dir + "_odd"-suffixed fit artifacts +
# distinct HF upload prefixes — primary (swept) bytes/destinations untouched.
# ---------------------------------------------------------------------------


def _tag(args) -> str:
    return "capture" if args.layer_set == "swept" else "capture_oddlayers"


def _fits_name(args, kind: str, pos: str) -> str:
    suffix = "_odd" if args.layer_set == "odd" else ""
    stem = f"{kind}_{pos}" if pos else kind
    return f"{stem}{suffix}.json"


# ---------------------------------------------------------------------------
# D2 (review round 2): plan §9 out-root disk-headroom asserts before each
# write-heavy phase (gen / capture / fits), resume-aware — capture need is
# scaled to the PENDING stages (a stage with rows.json already on disk costs
# nothing more).
# ---------------------------------------------------------------------------


def _assert_headroom(paths: dict, need_gb: float, phase: str) -> None:
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    free = assert_out_root_headroom(paths["root"], need_gb, phase=phase)
    logger.info("[i2588] disk headroom OK for %s: %.1f GB free >= %.1f GB", phase, free, need_gb)


def _est_model_gb(cell: PC.Cell) -> float:
    """Rough bf16 snapshot size (transformer block params + embeddings), with a
    1.35x margin — a §9 FLOOR for the headroom assert, not an exact figure."""
    m = cell.model
    params = 12 * m.n_layers * m.h_dim**2 + 2 * 152_000 * m.h_dim
    return 2.0 * params / 1e9 * 1.35


def _stage_row_estimates(args, cell: PC.Cell) -> dict[str, tuple[int, int]]:
    """{stage: (n_rows, n_slots)} for the capture-store size floor (§9 ledger).

    Slots = one x per input position + one y_ans; ceiling stages are y-only.
    """
    n_pos = len(cell.input_positions)
    full_n = {"train_10k": 10_000, "val_400": 400, "test_1000": 1_000}
    # Banked cells capture the FULL banked row set even under --smoke (the
    # banked chunks are staged whole); fresh cells slice at generation.
    slice_generic = args.smoke and cell.fresh
    gen_n = {s: (SMOKE_SLICE[s] if slice_generic else full_n[s]) for s in GENERIC_SPLITS}
    gpqa_n = (SMOKE_GPQA_QUESTIONS if args.smoke else PC.GPQA_N_QUESTIONS) * 1  # per seed
    est: dict[str, tuple[int, int]] = {s: (n, n_pos + 1) for s, n in gen_n.items()}
    for seed in PC.CEILING_SEEDS:
        est[f"ceiling_s{seed}"] = (gen_n["test_1000"], 1)  # y-only
    for seed in _gpqa_seeds(args):
        est[f"gpqa_s{seed}"] = (gpqa_n, n_pos + 1)
    return est


def _est_capture_need_gb(args, cell: PC.Cell, layers: list[int], paths: dict) -> float:
    """fp32 capture-store bytes for the PENDING stages only (resume-aware)."""
    tag_dir = paths[_tag(args)]
    total_bytes = 0.0
    for stage, (n_rows, n_slots) in _stage_row_estimates(args, cell).items():
        if (tag_dir / stage / "rows.json").exists():
            continue  # stage already captured — costs nothing more
        total_bytes += n_rows * n_slots * len(layers) * cell.model.h_dim * 4
    return total_bytes / 1e9 * 1.2 + 1.0  # 20% shard/metadata margin + 1 GB floor


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
    """Banked #2330 rows -> wrow shape (producer render conventions, G module).

    A19 (round 3, banked-full-grain-not-exact): the banked chunk files carry
    the FULL producer split, so ids the P0 union-drop excluded exist in the
    files BY CONSTRUCTION. Consume EXACTLY the expected post-union-drop id
    set: filter to it, dedupe by ``ci`` (first usable row wins), and assert
    the consumed set equals the expected set — union-dropped ids never enter
    banked cells' fits, duplicates never set-collapse silently, and an
    expected id with no usable row fails loud naming the ids.
    """
    split = "test_1000" if stage.startswith("ceiling_s") else stage
    expected = _banked_expected_ids(split)
    assert expected, f"banked stage {stage}: empty expected id set (split {split})"
    split_dir = paths["cell"] / "banked" / stage
    rows: list[dict] = []
    consumed: set[int] = set()
    n_extra = n_dup = n_empty = 0
    for f in sorted(split_dir.glob("*.json")):
        payload = json.loads(f.read_text(encoding="utf-8"))
        for r in payload["rows"]:
            ci = int(r["ci"])
            if ci not in expected:
                n_extra += 1  # union-dropped / out-of-split id — NEVER consumed
                continue
            if ci in consumed:
                n_dup += 1
                continue
            if G._is_empty_response(r["response"]):
                n_empty += 1
                continue
            prompt_render = G._render_prompt(tok, r["prompt"])
            ids = tok(prompt_render, add_special_tokens=False)["input_ids"]
            consumed.add(ci)
            rows.append(
                {
                    "row_id": f"{stage}_{ci}",
                    "ci": ci,
                    "prompt": prompt_render,
                    "n_prompt_tokens": len(ids),
                    "read_points": {"prompt_last": len(ids) - 1},
                    "text": r["response"],
                    "finish_reason": r.get("finish_reason", "stop"),
                    "stage": stage,
                }
            )
    missing = expected - consumed
    assert not missing, (
        f"A19 banked-consume FAIL [{stage}]: {len(missing)} expected {split} ids have no usable "
        f"(non-empty, in-split) banked row (first 5: {sorted(missing)[:5]}); "
        f"extras_filtered={n_extra} duplicates_skipped={n_dup} empty_rows_skipped={n_empty}"
    )
    logger.info(
        "[i2588] A19 banked-consume exact [%s]: consumed=%d extras_filtered=%d "
        "duplicates_skipped=%d empty_rows_skipped=%d",
        stage,
        len(consumed),
        n_extra,
        n_dup,
        n_empty,
    )
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
    """Teacher-forced multi-layer capture for one stage; writes shards + rows.json.

    Round 3 (gen-capture-stage-resume): rows.json is the stage's terminal
    artifact (written LAST, after every shard) — when present the stage is
    skipped (--force re-runs), so a mid-phase crash resumes at the first
    incomplete stage.
    """
    from explore_persona_space.atomic_io import savez_atomic

    rows_json = paths[layer_tag] / stage / "rows.json"
    if rows_json.exists() and not args.force:
        logger.info(
            "[i2588] [%s/%s] rows.json present — stage already captured; "
            "skipped (--force to re-run)",
            layer_tag,
            stage,
        )
        return
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

    # A1 (review round 2): G._resolve_decoder_blocks returns (blocks, depth) —
    # its documented failure return is (None, 0); never treat the tuple as the
    # layer list.
    blocks, wrap_depth = G._resolve_decoder_blocks(hf)
    assert blocks is not None and wrap_depth > 0, (
        f"decoder blocks unresolved for {m.hf_id} (G._resolve_decoder_blocks returned None) — "
        "the wrapper chain exposes no .layers within depth 4"
    )
    assert max(layers) < len(blocks), (max(layers), len(blocks))
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


def _banked_expected_ids(split: str) -> set[int]:
    """The FULL post-union-drop id set a banked stage must cover (D3/A19)."""
    split_ids = G._load_split_ids(PC.SPLIT_IDS_PATH)
    drop = _p0_union_drop()
    _manifest_key, ids_key, _seed = G.SPLIT_TO_MANIFEST[split]
    return {int(i) for i in split_ids["splits"][ids_key]} - drop[split]


def _validate_capture_inputs(args, cell: PC.Cell, paths: dict) -> dict:
    """B2 (review round 2): validate EVERY persisted capture input — file
    presence, schema keys, row counts — BEFORE any tokenizer/model load, so a
    missing/deformed input fails in seconds instead of after a 7-54 GB load.

    D3/A19: for banked stages this IS the capture-only cell's first action —
    the full-grain matched-id assert over the banked id set, measured counts
    reported (plan §12 A19 / §10 reuse attestation).
    """
    report: dict = {"stages": {}}
    for stage in _stage_names(args, cell):
        if cell.fresh or stage.startswith("gpqa"):
            p = paths["parsed"] / f"{stage}.jsonl"
            assert p.exists(), (
                f"capture input missing: {p} — run --phase parse first "
                "(B2: inputs validate BEFORE the capture model loads)"
            )
            rows = PC.read_jsonl(p)
            assert rows, f"capture input empty: {p}"
            need = {"row_id", "prompt", "n_prompt_tokens", "text", "ans_char_span"}
            # Round 3 (consumer-contract-post-init): EVERY row validates, not
            # only row 0 — a malformed later row otherwise passes preflight
            # and capture dies AFTER the 7-54 GB model load (B2's exact
            # failure class at row grain). Exact row counts are unknowable for
            # fresh parsed stages (parse drops are data-dependent) — counts
            # are recorded here and asserted exactly on the banked branch.
            bad = [(i, sorted(need - set(r))) for i, r in enumerate(rows) if need - set(r)]
            assert not bad, (
                f"parsed stage {stage}: {len(bad)} rows missing required keys "
                f"(first: row {bad[0][0]} missing {bad[0][1]})"
            )
            report["stages"][stage] = {"kind": "parsed", "n_rows": len(rows)}
        else:
            split = "test_1000" if stage.startswith("ceiling_s") else stage
            split_dir = paths["cell"] / "banked" / stage
            files = sorted(split_dir.glob("*.json"))
            assert files, f"banked capture input missing: {split_dir} — run --phase stage first"
            banked_ids: set[int] = set()
            usable_ids: set[int] = set()  # ids with >= 1 NON-empty row (consumable)
            n_rows = n_empty = n_dup = 0
            for f in files:
                payload = json.loads(f.read_text(encoding="utf-8"))
                assert isinstance(payload, dict) and "rows" in payload, (f.name, "no rows key")
                for r in payload["rows"]:
                    for k in ("ci", "prompt", "response"):
                        assert k in r, f"banked row missing key {k!r} ({f.name})"
                    ci = int(r["ci"])
                    n_rows += 1
                    n_dup += int(ci in banked_ids)
                    banked_ids.add(ci)
                    if G._is_empty_response(r["response"]):
                        n_empty += 1
                    else:
                        usable_ids.add(ci)
            expected = _banked_expected_ids(split)
            # Round 3 (banked-full-grain-not-exact): A19's 1:1 is asserted on
            # the USABLE id set — an expected id whose only rows are empty
            # would otherwise pass presence-validation and silently vanish at
            # consume (_banked_stage_rows skips empty rows). Extras exist BY
            # CONSTRUCTION whenever the P0 union-drop is nonempty (banked
            # files carry the full producer split); they are counted here and
            # FILTERED at consume, never ingested into fits.
            missing_ids = expected - usable_ids
            assert not missing_ids, (
                f"A19 matched-id FAIL [{stage}]: {len(missing_ids)} expected {split} ids lack a "
                f"usable (non-empty) banked row (first 5: {sorted(missing_ids)[:5]})"
            )
            report["stages"][stage] = {
                "kind": "banked",
                "n_files": len(files),
                "n_rows": n_rows,
                "n_empty_response": n_empty,
                "n_duplicate_ci": n_dup,
                "n_expected_ids": len(expected),
                "n_usable_matched": len(expected & usable_ids),
                "n_extra_banked": len(banked_ids - expected),
            }
            logger.info("[i2588] A19 matched-id OK [%s]: %s", stage, report["stages"][stage])
    # Round 3: the odd pass writes its OWN suffixed report — the primary
    # (swept) validation report is never overwritten (C3).
    suffix = "_odd" if args.layer_set == "odd" else ""
    PC.write_json_atomic(
        paths["cell"] / f"capture_input_validation{suffix}.json",
        {"meta": _meta(), "cell": cell.key, "layer_set": args.layer_set, **report},
    )
    return report


def phase_capture(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("capture")
    m = cell.model
    layers = _layers_for(args, cell)
    tag = _tag(args)
    # B2: every persisted input validated BEFORE any tokenizer/model load.
    _validate_capture_inputs(args, cell, paths)
    # D2: pending-stage store bytes + the model snapshot (conservative if cached).
    _assert_headroom(
        paths,
        _est_capture_need_gb(args, cell, layers, paths) + _est_model_gb(cell),
        f"capture:{cell.key}:{tag}",
    )
    from transformers import AutoTokenizer

    sem = _acquire_capture_slot(paths["root"])
    try:
        tok = AutoTokenizer.from_pretrained(m.hf_id)
        hf = G._load_capture_model(m.hf_id, args.device, "bfloat16")
        for stage in GENERIC_SPLITS:
            _capture_stage(args, cell, paths, hf, tok, stage, layers, layer_tag=tag)
        for seed in PC.CEILING_SEEDS:
            _capture_stage(
                args, cell, paths, hf, tok, f"ceiling_s{seed}", layers, y_only=True, layer_tag=tag
            )
        for seed in _gpqa_seeds(args):
            _capture_stage(args, cell, paths, hf, tok, f"gpqa_s{seed}", layers, layer_tag=tag)
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
    tag = _tag(args)
    # C3: the odd pass uploads its OWN local dir to its OWN prefix — the
    # primary capture bytes/destination are untouched.
    _upload_dir(
        paths[tag], f"{prefix}/analysis_tensors/{tag}", f"{cell.key} capture tensors ({tag})"
    )


# ---------------------------------------------------------------------------
# Phase: g2-anchor (anchor pod only) + the fail-closed sentinel await
# ---------------------------------------------------------------------------


def phase_g2_anchor(args, cell: PC.Cell, paths: dict) -> None:
    """Anchor refit at tol=1e-6 (plan §7 G2); publishes the HF sentinel every
    fit stage fail-closes on. Runs the #2330 battery's own store/assembly path
    AND (C1, review round 2) re-runs the SAME assembled arrays through the
    EXACT production fits estimator (_fit_edge_extended_with_val ->
    F.fit_ridge_with_weights) so the gate certifies the path production
    actually dispatches — never only the MF sibling.

    Smoke blind-spot enumeration: none — smoke executes every production gate
    (this phase has no smoke-conditional branch; --smoke changes only the
    upload prefix of the CELL artifacts, not this gate's math or sentinel).
    """
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
    captured: dict = {}

    def _gate(X, Y, tr, val, te, d):
        captured.update(X=X, Y=Y, tr=tr, val=val, te=te)
        return MF.run_anchor_gate(
            X, Y, tr, val, te, d, expected_r2=PC.ANCHOR_EXPECTED_R2, tol=PC.ANCHOR_TOL
        )

    rec = MF._run_anchor_on_store(store, mcfg, dev, _gate)
    # C1: production-path equivalence leg on the SAME assembled arrays.
    pred_te, _pred_val, meta_p = _fit_edge_extended_with_val(
        captured["X"], captured["Y"], captured["tr"], captured["val"], captured["te"], dev
    )
    meta_p.pop("W_payload", None)
    prod_r2 = float(LF._pooled_r2(pred_te, captured["Y"][captured["te"]]))
    prod_dev = abs(prod_r2 - PC.ANCHOR_EXPECTED_R2)
    assert prod_dev <= PC.ANCHOR_PROD_EQUIV_TOL, (
        f"G2 PRODUCTION-PATH equivalence MISS (C1): _fit_edge_extended_with_val R²={prod_r2:.7f} "
        f"vs pinned {PC.ANCHOR_EXPECTED_R2:.7f} (|Δ|={prod_dev:.3g} > "
        f"{PC.ANCHOR_PROD_EQUIV_TOL}) — the production fits path diverges from the MF anchor "
        "reproduction; fix the fitter/assembly seam, never loosen the pin."
    )
    logger.info(
        "[i2588] G2 production-path equivalence PASS: R²=%.7f (|Δ|=%.3g, λ=%.3g)",
        prod_r2,
        prod_dev,
        float(meta_p.get("selected_lambda", float("nan"))),
    )
    sentinel = {
        **rec,
        "meta": _meta(),
        "schema_version": PC.G2_SENTINEL_SCHEMA_VERSION,
        "gate": "g2_anchor",
        "status": "PASS",
        "expected_r2": PC.ANCHOR_EXPECTED_R2,
        "tol": PC.ANCHOR_TOL,
        "store_revision_pin_recorded": MF.STORE_REVISION_PIN_7B,
        "production_path": {
            "estimator": "_fit_edge_extended_with_val (F.fit_ridge_with_weights, production fits)",
            "realized_r2": prod_r2,
            "abs_deviation_vs_pin": prod_dev,
            "tol": PC.ANCHOR_PROD_EQUIV_TOL,
            "selected_lambda": float(meta_p.get("selected_lambda", float("nan"))),
            "grid_extensions": int(meta_p.get("grid_extensions", 0)),
        },
    }
    out = paths["fits"] / "g2_anchor_pass.json"
    PC.write_json_atomic(out, sentinel)
    _upload_file(out, PC.G2_SENTINEL_PATH, "G2 anchor PASS sentinel")
    logger.info("[i2588] G2 anchor PASS published -> %s", PC.G2_SENTINEL_PATH)


def _validate_g2_sentinel(rec: dict) -> None:
    """C2 (review round 2): fits fail-close on sentinel CONTENT — a stale,
    status-only, older-pin, or numeric-field-missing sentinel is REFUSED."""
    assert rec.get("schema_version") == PC.G2_SENTINEL_SCHEMA_VERSION, (
        f"G2 sentinel schema_version {rec.get('schema_version')!r} != "
        f"{PC.G2_SENTINEL_SCHEMA_VERSION} — stale/status-only sentinel refused (C2); "
        "re-run --phase g2-anchor at the current pins"
    )
    assert rec.get("status") == "PASS", f"G2 sentinel present but not PASS: {rec.get('status')!r}"
    assert rec.get("store_revision_pin_recorded") == MF.STORE_REVISION_PIN_7B, (
        "G2 sentinel store revision pin mismatch",
        rec.get("store_revision_pin_recorded"),
        MF.STORE_REVISION_PIN_7B,
    )
    assert float(rec.get("expected_r2", float("nan"))) == PC.ANCHOR_EXPECTED_R2, (
        "G2 sentinel minted against a DIFFERENT anchor pin",
        rec.get("expected_r2"),
    )
    for k in ("realized_r2", "abs_deviation"):
        v = rec.get(k)
        assert isinstance(v, (int, float)) and math.isfinite(float(v)), (k, v)
    assert float(rec["abs_deviation"]) <= PC.ANCHOR_TOL, (
        "G2 sentinel numeric gate fields fail the pinned tolerance",
        rec["abs_deviation"],
        PC.ANCHOR_TOL,
    )
    pp = rec.get("production_path")
    assert isinstance(pp, dict), "G2 sentinel lacks the C1 production-path equivalence record"
    ppr = float(pp.get("realized_r2", float("nan")))
    # Round 3 (g2-prodpath-tol-unpinned): validate against the CURRENT pinned
    # tolerance, never the sentinel's own self-reported pp["tol"] (a sentinel
    # minted under a looser tol must be refused, not trusted).
    assert math.isfinite(ppr) and float(pp["abs_deviation_vs_pin"]) <= PC.ANCHOR_PROD_EQUIV_TOL, (
        "G2 sentinel production-path record fails the pinned tolerance",
        pp,
        PC.ANCHOR_PROD_EQUIV_TOL,
    )
    meta = rec.get("meta")
    assert isinstance(meta, dict) and meta.get("git_sha"), (
        "G2 sentinel carries no run/commit identity (meta.git_sha)"
    )


def _await_g2(args) -> dict:
    """Fail-closed poll for the G2 sentinel.

    C2 (review round 2): the REGISTERED 45-min bound applies under --smoke too
    (the wait is part of the production contract the smoke certifies), and the
    sentinel is validated by CONTENT (_validate_g2_sentinel), never presence.
    """
    from huggingface_hub import HfApi, hf_hub_download

    api = HfApi()
    deadline = time.time() + PC.G2_SENTINEL_TIMEOUT_S
    while True:
        exists = HUB.retry_transient(
            # HUB_VERIFY_RETRY_EXEMPT: wrapped in HUB.retry_transient at this call site
            lambda: api.file_exists(PC.HF_DATA_REPO, PC.G2_SENTINEL_PATH, repo_type="dataset"),
            what="G2 sentinel file_exists probe",
        )
        if exists:
            local = HUB.retry_transient(
                lambda: hf_hub_download(
                    PC.HF_DATA_REPO,
                    PC.G2_SENTINEL_PATH,
                    repo_type="dataset",
                    force_download=True,  # never a stale local cache copy (C2)
                ),
                what="G2 sentinel download",
            )
            rec = json.loads(Path(local).read_text(encoding="utf-8"))
            _validate_g2_sentinel(rec)
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


def _bundle(paths: dict, layer: int, pos: str, *, tag: str = "capture") -> dict:
    """(X, Y, tr, val, te) fp64 bundle for one (layer, input-position)."""
    parts_x, parts_y, idx, n = [], [], {}, 0
    for split, key in (("train_10k", "tr"), ("val_400", "val"), ("test_1000", "te")):
        d = _load_stage_layer(paths, split, layer, tag=tag)
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


def _acc1(knn_read: dict) -> float:
    """acc@1 accessor tolerant of JSON-round-tripped int keys ("1" vs 1)."""
    acc = knn_read["acc_at_k"]
    v = acc.get(1, acc.get("1")) if isinstance(acc, dict) else None
    assert v is not None, f"acc_at_k lacks k=1: {acc!r}"
    return float(v)


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


def _aligned_ceiling_draws(
    paths: dict, layer: int, *, tag: str = "capture"
) -> tuple[np.ndarray, np.ndarray] | None:
    """ci-aligned (seed-43 y_ans, seed-44 y_ans) at one layer, or None (<2 rows)."""
    da = _load_stage_layer(paths, f"ceiling_s{PC.CEILING_SEEDS[0]}", layer, tag=tag)
    db = _load_stage_layer(paths, f"ceiling_s{PC.CEILING_SEEDS[1]}", layer, tag=tag)
    # ceiling row_ids carry the seed tag ("ceiling_s43_<ci>"); align on the ci suffix
    ci_a = np.array([str(r).rsplit("_", 1)[1] for r in da["row_ids"]])
    ci_b = np.array([str(r).rsplit("_", 1)[1] for r in db["row_ids"]])
    common, ia, ib = np.intersect1d(ci_a, ci_b, return_indices=True)
    if common.size < 2:
        return None
    return da["y_ans"][ia].astype(np.float64), db["y_ans"][ib].astype(np.float64)


def _ceiling_alignment(paths: dict, layer: int, *, tag: str = "capture") -> dict | None:
    drawn = _aligned_ceiling_draws(paths, layer, tag=tag)
    if drawn is None:
        return None
    return _two_draw_ceiling(*drawn)


def _ceiling_retrieval(paths: dict, layer: int, *, tag: str = "capture") -> dict | None:
    """SR1 (review round 2, B3): the REGISTERED repeat-draw retrieval ceiling —
    seed-43 ceiling-draw answer vectors RETRIEVING their seed-44 counterparts
    (cosine, pool = the aligned ceiling rows, chance = 1/pool) at the selected
    layer. Consumed by P3 as (map − null)/(ceiling − null); the variance-
    weighted per-dim Pearson (_two_draw_ceiling) stays a SECONDARY diagnostic.
    """
    drawn = _aligned_ceiling_draws(paths, layer, tag=tag)
    if drawn is None:
        return None
    ya, yb = drawn
    hits = _perrow_hits_cos(ya, yb)  # seed-43 vector i retrieving seed-44 target i
    n = int(ya.shape[0])
    return {
        "ceiling_acc1_cos": float(np.mean(hits["hit1"])),
        "n_pool": n,
        "chance": 1.0 / n,
        "seed_pair": list(PC.CEILING_SEEDS),
        "statistic": "seed-43 answer vectors retrieving seed-44 targets, cosine acc@1 (SR1)",
        "rank_mean": float(np.mean(hits["rank"])),
    }


def _participation_ratio_x(X_tr: np.ndarray, dev) -> float:
    """Effective-rank participation ratio of the TRAIN input covariance
    spectrum: (Σ s²)² / Σ s⁴ over centered singular values (plan §6 per-fit
    reporting; D5, review round 2)."""
    xt = torch.as_tensor(X_tr, dtype=torch.float64, device=dev)
    xt = xt - xt.mean(dim=0)
    s = torch.linalg.svdvals(xt)
    s2 = s**2
    return float((s2.sum() ** 2 / ((s2**2).sum() + 1e-300)).item())


def phase_fits(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("fits")
    g2 = _await_g2(args)
    _assert_headroom(paths, 2.0, f"fits:{cell.key}")  # D2: fit JSONs are small
    dev = MF._resolve_device(args.device)
    layers = _layers_for(args, cell)
    tag = _tag(args)
    for pos in cell.input_positions:
        per_layer: dict[int, dict] = {}
        for layer in layers:
            # Intra-phase resume (B1): a completed per-layer unit is reloaded,
            # never refit (resume key = the unit filename: pos + layer +
            # layer_set suffix; smoke lives in a distinct cell_dir root).
            unit_path = paths["fits"] / _fits_name(args, f"percell_{pos}_L{layer:02d}", pos="")
            if unit_path.exists() and not args.force:
                per_layer[layer] = json.loads(unit_path.read_text(encoding="utf-8"))
                logger.info("[fits] unit L%02d pos=%s resumed from %s", layer, pos, unit_path.name)
                continue
            t_unit = time.monotonic()
            b = _bundle(paths, layer, pos, tag=tag)
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
                "n_train_over_d": float(len(b["tr"]) / b["X"].shape[1]),
            }
            if layer == layers[0]:
                logger.info("[i2588] [fits %s pos=%s] first layer done", cell.key, pos)
            # checkpoint-per-unit: persist the per-layer record the moment it lands
            per_layer[layer]["unit_elapsed_s"] = round(time.monotonic() - t_unit, 3)
            PC.write_json_atomic(unit_path, {"meta": _meta(), **per_layer[layer]})
            logger.info(
                "[fits] unit L%02d/%s pos=%s val_acc1_cos=%.4f test_r2=%.4f elapsed=%.1fs",
                layer,
                cell.key,
                pos,
                _acc1(knn_val["ridge"]["cosine"]),
                per_layer[layer]["test_r2"],
                per_layer[layer]["unit_elapsed_s"],
            )
            # persist the selected-λ payload only at layer_star (below); free here
            del payload
        star = max(per_layer, key=lambda li: _acc1(per_layer[li]["knn_val"]["ridge"]["cosine"]))
        ceiling = _ceiling_alignment(paths, star, tag=tag)
        ceiling_retr = _ceiling_retrieval(paths, star, tag=tag)  # SR1 (B3)
        # Per-row TEST hit indicators at layer_star (the P3 paired bootstrap's
        # input — plan §4.4: 1,000 draws seed 42 over paired per-row hits).
        b_star = _bundle(paths, star, pos, tag=tag)
        pred_te_star, _pv, _m = _fit_edge_extended_with_val(
            b_star["X"], b_star["Y"], b_star["tr"], b_star["val"], b_star["te"], dev
        )
        _m.pop("W_payload", None)
        # D5: participation ratio of the TRAIN inputs at layer_star, persisted
        # per (model, arm, layer_star) in the fits JSON (plan §6).
        pr_x = _participation_ratio_x(b_star["X"][b_star["tr"]], dev)
        te_ids = _load_stage_layer(paths, "test_1000", star, tag=tag)["row_ids"]
        hits = _perrow_hits_cos(pred_te_star, b_star["Y"][b_star["te"]])
        PC.write_json_atomic(
            paths["fits"] / _fits_name(args, "perrow", pos),
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
            "ceiling_retrieval_at_star": ceiling_retr,
            "participation_ratio_x_at_star": pr_x,
            "n_train_over_d_at_star": float(len(b_star["tr"]) / b_star["X"].shape[1]),
        }
        PC.write_json_atomic(paths["fits"] / _fits_name(args, "fits", pos), record)
        logger.info("[i2588] [fits %s pos=%s] layer_star=%d pr_x=%.1f", cell.key, pos, star, pr_x)


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
                    # Registered per-draw acc@k grid (plan §4.4), not acc@1 only
                    # (review round 2, F item).
                    "acc_cos_at_k": {
                        str(k): float(v) for k, v in knn["cosine"]["acc_at_k"].items()
                    },
                    "acc_euc_at_k": {
                        str(k): float(v) for k, v in knn["euclidean"]["acc_at_k"].items()
                    },
                }
            )
        logger.info("[nulls] unit %d/%d draws done", min(d0 + nb, draws), draws)
    return out


def phase_nulls(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("nulls")
    dev = MF._resolve_device(args.device)
    draws = SMOKE_PERM_DRAWS if args.smoke else PC.PERM_DRAWS
    for pos in cell.input_positions:
        fits = json.loads(
            (paths["fits"] / _fits_name(args, "fits", pos)).read_text(encoding="utf-8")
        )
        star = int(fits["layer_star"])
        b = _bundle(paths, star, pos, tag=_tag(args))
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
            paths["fits"] / _fits_name(args, "nulls", pos),
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


def _refit_star_payload(args, paths: dict, pos: str, star: int, dev) -> dict:
    b = _bundle(paths, star, pos, tag=_tag(args))
    grid = np.array(LF.LAMBDAS, dtype=np.float64)
    _pred, _meta, payload = F.fit_ridge_with_weights(
        b["X"], b["Y"], b["tr"], b["val"], b["te"], grid, dev, LF.RIDGE_BLOCK
    )
    return payload


def _same_question_retrieval(pred: np.ndarray, y_true: np.ndarray, qids: np.ndarray) -> dict:
    """Same-question retrieval reads + PER-ROW outcomes (B4, review round 2)."""
    same_q = qids[:, None] == qids[None, :]
    pn = pred / (np.linalg.norm(pred, axis=1, keepdims=True) + 1e-12)
    tn = y_true / (np.linalg.norm(y_true, axis=1, keepdims=True) + 1e-12)
    sim = pn @ tn.T
    nn = sim.argmax(axis=1)
    rows = np.arange(len(nn))
    same_q_hits = same_q[rows, nn].astype(int)
    return {
        "same_question_acc1_cos": float(same_q_hits.mean()),
        "exact_row_acc1_cos": float((nn == rows).mean()),
        "n_rows": int(y_true.shape[0]),
        "same_q_hit": [int(h) for h in same_q_hits],
        "cos_true_pair": [float(sim[i, i]) for i in rows],  # F: per-row cosine persisted
        "cos_nn": [float(sim[i, nn[i]]) for i in rows],
    }


def _load_gpqa_star(args, paths: dict, pos: str, star: int) -> dict:
    """Concatenated GPQA captures at layer_star: X, Y, row_ids, qids, stages."""
    tag = _tag(args)
    xs, ys, qids, row_ids, stages = [], [], [], [], []
    for seed in _gpqa_seeds(args):
        stage = f"gpqa_s{seed}"
        d = _load_stage_layer(paths, stage, star, tag=tag)
        rows_meta = json.loads((paths[tag] / stage / "rows.json").read_text(encoding="utf-8"))
        qid_by_row = {r["row_id"]: r["qid"] for r in rows_meta["rows"]}
        xs.append(d[f"x_{pos}"])
        ys.append(d["y_ans"])
        row_ids.extend(str(r) for r in d["row_ids"])
        qids.extend(qid_by_row[str(r)] for r in d["row_ids"])
        stages.extend(stage for _ in d["row_ids"])
    return {
        "X": np.concatenate(xs).astype(np.float64),
        "Y": np.concatenate(ys).astype(np.float64),
        "row_ids": row_ids,
        "qids": np.array(qids),
        "stages": stages,
    }


def phase_gpqa_transfer(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("gpqa_transfer")
    dev = MF._resolve_device(args.device)
    seeds = _gpqa_seeds(args)
    behavioral = _gpqa_behavioral(args, cell, paths, seeds)
    for pos in cell.input_positions:
        fits = json.loads(
            (paths["fits"] / _fits_name(args, "fits", pos)).read_text(encoding="utf-8")
        )
        star = int(fits["layer_star"])
        payload = _refit_star_payload(args, paths, pos, star, dev)
        g = _load_gpqa_star(args, paths, pos, star)
        pred = F.apply_map(payload, g["X"], torch.device(dev))
        retr = _same_question_retrieval(pred, g["Y"], g["qids"])
        n_pool = retr["n_rows"]
        n_per_q = int(len(seeds))
        # B4: per-row GPQA retrieval outcomes persisted — the H2 complete-case
        # intersection + per-checkpoint paired bootstrap consume these rows.
        PC.write_json_atomic(
            paths["fits"] / _fits_name(args, "gpqa_perrow", pos),
            {
                "meta": _meta(),
                "cell": cell.key,
                "input_position": pos,
                "layer_star": star,
                "row_ids": g["row_ids"],
                "qids": [str(q) for q in g["qids"]],
                "same_q_hit": retr["same_q_hit"],
                "cos_true_pair": retr["cos_true_pair"],
                "cos_nn": retr["cos_nn"],
            },
        )
        PC.write_json_atomic(
            paths["fits"] / _fits_name(args, "gpqa_transfer", pos),
            {
                "meta": _meta(),
                "cell": cell.key,
                "input_position": pos,
                "layer_star": star,
                "transfer_only": True,
                "n_rows": n_pool,
                "same_question_acc1_cos": retr["same_question_acc1_cos"],
                "same_question_chance": float(n_per_q / n_pool),
                "exact_row_acc1_cos": retr["exact_row_acc1_cos"],
                "exact_row_chance": float(1.0 / n_pool),
                "behavioral": behavioral,
            },
        )
        logger.info(
            "[i2588] [gpqa %s pos=%s] same-q acc@1=%.4f (chance %.4f) exact=%.4f",
            cell.key,
            pos,
            retr["same_question_acc1_cos"],
            n_per_q / n_pool,
            retr["exact_row_acc1_cos"],
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
        # Integer counts persisted so the VM-side judge-verdict merge is
        # DETERMINISTIC (never reconstructed from a float; B5, review round 2).
        "n_correct": correct,
        "n_unparseable": unparseable,
        "acc_exact_match": correct / max(1, total),
        "frac_unparseable": frac_unparseable,
        "judge_fallback_flagged": frac_unparseable > PC.GPQA_EXTRACTION_FAIL_TRIGGER,
    }


# ---------------------------------------------------------------------------
# Phase: resid (registered length-residualization protocol + length-only floor)
# ---------------------------------------------------------------------------


def _length_covariates(
    paths: dict, stage: str, row_ids: np.ndarray, arm: str, *, tag: str = "capture"
) -> np.ndarray:
    rows_meta = json.loads((paths[tag] / stage / "rows.json").read_text(encoding="utf-8"))["rows"]
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
    tag = _tag(args)
    for pos in cell.input_positions:
        fits = json.loads(
            (paths["fits"] / _fits_name(args, "fits", pos)).read_text(encoding="utf-8")
        )
        star = int(fits["layer_star"])
        data, Z = {}, {}
        for split in GENERIC_SPLITS:
            d = _load_stage_layer(paths, split, star, tag=tag)
            data[split] = d
            Z[split] = _length_covariates(paths, split, d["row_ids"], cell.arm, tag=tag)

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
        payload_r = meta.pop("W_payload")
        knn = LF._knn_reads({"ridge_resid": pred_te}, Yr[idx["te"]])
        # Length-only floor: predict RAW Y from the covariates alone.
        B_len, *_ = np.linalg.lstsq(Ztr, data["train_10k"]["y_ans"].astype(np.float64), rcond=None)
        pred_len = _aug(Z["test_1000"]) @ B_len
        y_te_raw = data["test_1000"]["y_ans"].astype(np.float64)
        knn_len = LF._knn_reads({"length_only": pred_len}, y_te_raw)
        # B4/§6 (iii): GPQA-side residualization — the GENERIC-TRAIN length
        # coefficients applied UNCHANGED to the GPQA captures; the resid-fit
        # map applied to the residualized GPQA inputs; same-question retrieval
        # per row (the H2 residualized-gap sensitivity read).
        g = _load_gpqa_star(args, paths, pos, star)
        Zg_parts = []
        for seed in _gpqa_seeds(args):
            stage = f"gpqa_s{seed}"
            d = _load_stage_layer(paths, stage, star, tag=tag)
            Zg_parts.append(_length_covariates(paths, stage, d["row_ids"], cell.arm, tag=tag))
        Zg = _aug(np.concatenate(Zg_parts))
        Xg_r = g["X"] - Zg @ coefs["x"]
        Yg_r = g["Y"] - Zg @ coefs["y"]
        pred_g = F.apply_map(payload_r, Xg_r, torch.device(dev))
        retr_g = _same_question_retrieval(pred_g, Yg_r, g["qids"])
        gpqa_resid = {
            "same_question_acc1_cos": retr_g["same_question_acc1_cos"],
            "exact_row_acc1_cos": retr_g["exact_row_acc1_cos"],
            "n_rows": retr_g["n_rows"],
            "same_question_chance": float(len(_gpqa_seeds(args)) / max(1, retr_g["n_rows"])),
            "row_ids": g["row_ids"],
            "same_q_hit": retr_g["same_q_hit"],
            "coefficients": "generic-train OLS, applied unchanged (plan §6 iii)",
        }
        PC.write_json_atomic(
            paths["fits"] / _fits_name(args, "resid", pos),
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
                "gpqa_resid": gpqa_resid,
                "fit_meta": {
                    k: v
                    for k, v in meta.items()
                    if isinstance(v, (int, float, str, bool, type(None)))
                },
            },
        )
        logger.info(
            "[i2588] [resid %s pos=%s] resid acc1_cos=%.4f length-only=%.4f gpqa_resid=%.4f",
            cell.key,
            pos,
            _acc1(knn["ridge_resid"]["cosine"]),
            _acc1(knn_len["length_only"]["cosine"]),
            gpqa_resid["same_question_acc1_cos"],
        )


# ---------------------------------------------------------------------------
# Phase: upload-fits + sentinel
# ---------------------------------------------------------------------------


def phase_upload_fits(args, cell: PC.Cell, paths: dict) -> None:
    G._phase("upload_fits")
    smoke_pfx = "smoke/" if args.smoke else ""
    prefix_fits = f"{PC.PANEL_PREFIX}/{smoke_pfx}fits/{cell.key}"
    prefix_nulls = f"{PC.PANEL_PREFIX}/{smoke_pfx}nulls/{cell.key}"
    # C3: odd-layer sensitivity artifacts ("*_odd.json") route to their OWN
    # prefixes — the primary fits/nulls HF destinations are never overwritten.
    prefix_fits_odd = f"{PC.PANEL_PREFIX}/{smoke_pfx}fits_oddlayers/{cell.key}"
    prefix_nulls_odd = f"{PC.PANEL_PREFIX}/{smoke_pfx}nulls_oddlayers/{cell.key}"
    for f in sorted(paths["fits"].glob("*.json")):
        is_odd = f.stem.endswith("_odd")
        if f.name.startswith("nulls_"):
            target = prefix_nulls_odd if is_odd else prefix_nulls
        else:
            target = prefix_fits_odd if is_odd else prefix_fits
        _upload_file(f, f"{target}/{f.name}", f"{cell.key} {f.name}")
    suffix = "-odd" if args.layer_set == "odd" else ""
    sentinel = {
        "eval_numbers": _sentinel_numbers(args, cell, paths),
        "eval_paths": [str(p) for p in sorted(paths["fits"].glob("*.json"))],
        "reproducibility_card": _meta(),
        "wandb_url": None,
        "hf_hub_url": f"https://huggingface.co/datasets/{PC.HF_DATA_REPO}/tree/main/"
        + (prefix_fits_odd if args.layer_set == "odd" else prefix_fits),
        "worktree_path": str(_REPO_ROOT),
        "final_commit_sha": G._git_sha(),
        "gpu_hours_used": None,
        "gpu_hours_budgeted": None,
        "plan_deviations": [],
    }
    out = paths["logs"] / f"issue-2588-{cell.key}{suffix}-results.json"
    PC.write_json_atomic(out, sentinel)
    logger.info("[phase=done] cell %s complete rc=0 (sentinel %s)", cell.key, out)


def _sentinel_numbers(args, cell: PC.Cell, paths: dict) -> dict:
    nums: dict = {}
    for pos in cell.input_positions:
        f = paths["fits"] / _fits_name(args, "fits", pos)
        if f.exists():
            rec = json.loads(f.read_text(encoding="utf-8"))
            star = str(rec["layer_star"])
            nums[f"{pos}_layer_star"] = rec["layer_star"]
            nums[f"{pos}_test_acc1_cos_at_star"] = _acc1(
                rec["layers"][star]["knn_test"]["ridge"]["cosine"]
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

# Round 3 (oddlayer-overwrites-primary / C3): the odd-layer sensitivity pass
# runs ONLY the layer-DEPENDENT phases. gen/parse/upload-raw (and
# prologue/stage) are layer-set-INDEPENDENT: _cell_prefix and
# paths["raw"]/["parsed"] do not vary by layer set, so re-driving them under
# the odd pass's "_odd" sentinels would regenerate the primary parsed rows
# (vLLM continuous batching gives no byte-identity guarantee) and re-upload
# the primary raw/parsed HF prefixes — "the odd pass never overwrites the
# primary" is violated either way. The odd pass CONSUMES the primary pass's
# gen/parse artifacts on the same out-root; capture fails loud (B2) when they
# are absent.
_ODD_SEQUENCE = (
    "capture",
    "upload-capture",
    "fits",
    "nulls",
    "gpqa-transfer",
    "resid",
    "upload-fits",
)
_ODD_FORBIDDEN_PHASES = ("gen", "parse", "upload-raw")


def _sequence_for(args) -> tuple[str, ...]:
    """Resolve the phase sequence for the requested (--phase, --layer-set).

    ``--phase all`` under ``--layer-set odd`` runs the layer-dependent
    sequence only; an EXPLICIT odd invocation of a primary-artifact phase
    (gen/parse/upload-raw) is refused — those phases belong to the swept pass.
    """
    if args.phase == "all":
        if args.layer_set == "odd":
            return _ODD_SEQUENCE
        if args.smoke:
            return (*_ALL_SEQUENCE, "smoke-null-timing")
        return _ALL_SEQUENCE
    assert not (args.layer_set == "odd" and args.phase in _ODD_FORBIDDEN_PHASES), (
        f"--layer-set odd --phase {args.phase} refused: {args.phase} is layer-set-independent "
        "and writes/uploads PRIMARY artifacts (C3: the odd pass never overwrites the primary); "
        "run it under --layer-set swept — the odd pass consumes the swept pass's outputs"
    )
    return (args.phase,)


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
        "--force",
        action="store_true",
        help="re-run phases whose completion sentinels exist (B1 idempotency escape)",
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
    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom  # noqa: F401

    for name in ("_upload", "retry_transient", "stage_hub_prefix"):
        assert hasattr(HUB, name), name

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
    seq = _sequence_for(args)
    ran = _run_phases(args, cell, paths, seq)
    logger.info("[i2588] phases run=%s skipped=%s", ran, [s for s in seq if s not in ran])
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
