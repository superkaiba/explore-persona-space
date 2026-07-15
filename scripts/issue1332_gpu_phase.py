"""Issue #1332 P1 — GPU phase: vLLM generation + teacher-forced 28-layer capture.

Plan v3 §4.3. ONE dispatcher, per-family checkpointed, smoke == production with
a tiny cell subset (``--families A1,C1 --n-queries 8``; ``--tiny-model`` swaps
ONLY the GPU-scale weights for a from-config 2-layer same-arch model over the
real vocab — the tiny-real CPU e2e standard). Stages:

- ``inputs``     stage the bank + rewrites (local-first -> HF), assert the
                 26-family registry string identity, stage the #545 OOD inputs.
- ``gen``        vLLM greedy R_c(q) per family (chunked <=500, use_tqdm=False,
                 max_new=1024, truncation logged + per-family 2048 re-gen when
                 >2%); rollout TEXT persisted + uploaded per family the moment
                 the family completes (persist-by-default, #779).
- ``capture``    HF bf16 teacher-forced capture, all 28 layers, per-segment
                 TOKEN-ID concatenation + offset-mapping boundaries (the #1092
                 BPE-seam rule), batched right-padded forwards,
                 ``logits_to_keep=1`` introspection-guarded; per-family .pt
                 shard {cx_last, prefix_end, v_mean, v_last_turn} fp16 +
                 validity mask; batch-1 identity gate per family class.
- ``capture545`` same rig over the #545 behavior corpora (19 rows) + eval-column
                 realization pools (corpora/demos), rendered no-system-prompt
                 (the #545 training regime; off-policy targets — stated caveat).
                 Missing demo pools whose source corpus resolves on HF are
                 regenerated per the frozen #545 protocol at staging time
                 (r3 regen add-on; non-regenerable pools stay descoped, logged).
- ``upload``     idempotent exact-set Hub verification sweep.

``--stage all`` (production) runs gen / capture / capture545 as SUBPROCESSES
(one framework per process — the vLLM teardown gotcha), then writes the
poll_pipeline-conformant results sentinel and the single terminal
``[phase=done]`` line.

USAGE
    # production (GCP/auto lane --workload-cmd):
    uv run python scripts/issue1332_gpu_phase.py --full
    # smoke (identical dispatcher path, tiny subset, scratch roots):
    uv run python scripts/issue1332_gpu_phase.py --smoke --families A1,C1 \\
        --n-queries 8 --tiny-model --behaviors compliment_writing --n-rows-545 6
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

# vLLM v1 EngineCore fork-poisoning guard — BEFORE any vllm import (#628).
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
if Path("/workspace").exists():  # pod-only cache redirect; VM keeps its default
    os.environ.setdefault("HF_HOME", "/workspace/.cache/huggingface")

sys.path.insert(0, str(Path(__file__).resolve().parent))

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # explicit: subprocess dispatcher contract (env loaded before any spawn)

import issue1332_common as C  # noqa: E402

logger = logging.getLogger("issue1332.gpu")

GEN_CHUNK = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
TRUNCATION_REGEN_THRESHOLD = 0.02  # plan assumption 9
REGEN_MAX_NEW = 2048
GATE_EARLY_LAYERS = 4
GATE_EARLY_COS_MIN = 0.999  # bf16 two-bar gate (gotchas.md #779 r12 calibration)
GATE_FLAT_COS_MIN = 0.995


# ── tokenizer / models ────────────────────────────────────────────────────────

_TOKENIZER_CACHE: dict[str, object] = {}


def get_tokenizer(model_id: str = C.BASE_MODEL):
    """Module-cached tokenizer (never per-row from_pretrained — #664 429 trap)."""
    if model_id not in _TOKENIZER_CACHE:
        from transformers import AutoTokenizer

        _TOKENIZER_CACHE[model_id] = AutoTokenizer.from_pretrained(model_id)
    return _TOKENIZER_CACHE[model_id]


def tiny_model(tokenizer):
    """From-config 2-layer same-arch (Qwen2) model over the REAL vocab-id space.

    The tiny-real standard (gotchas.md "Mock-seam smokes"): fakes ONLY the
    GPU-scale weights; every seam (tokenizer ids, chat template, hidden-state
    shapes, capture positions) is real. CPU, fp32.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM

    cfg = AutoConfig.from_pretrained(C.BASE_MODEL)
    cfg.num_hidden_layers = 2
    cfg.hidden_size = 64
    cfg.intermediate_size = 128
    cfg.num_attention_heads = 4
    cfg.num_key_value_heads = 2
    if hasattr(cfg, "sliding_window"):
        cfg.sliding_window = None
    torch.manual_seed(0)
    model = AutoModelForCausalLM.from_config(cfg)
    model.eval()
    return model


# ── row construction (BPE-seam-safe; #1092 rule) ─────────────────────────────


def row_ids_and_positions(
    tokenizer, prompt: str, prefix_char_end: int, completion: str, row_label: str = "?"
) -> tuple[list[int], dict[str, int]]:
    """Teacher-forcing input ids + capture positions by TOKEN-ID concatenation.

    Never re-tokenizes the concatenated string (BPE seam merges shift positions
    — the #1092 G2 launch-#3 defect); intra-prompt ``prefix_end`` derives from
    the prompt's offset mapping. The answer span [n_prompt, n_total) INCLUDES
    the two template-end tokens (``<|im_end|>\\n`` — the #779 v_mean convention).
    """
    enc = tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
    prompt_ids = list(enc["input_ids"])
    offsets = enc["offset_mapping"]
    completion_ids = list(tokenizer.encode(completion, add_special_tokens=False))
    end_ids = list(tokenizer.encode(C.TEMPLATE_END_TEXT, add_special_tokens=False))
    row_ids = prompt_ids + completion_ids + end_ids
    n_total = len(row_ids)
    n_prompt = len(prompt_ids)
    if n_total > C.MAX_MODEL_LEN:
        raise ValueError(f"capture row {row_label}: {n_total} tokens > {C.MAX_MODEL_LEN}")
    # #594 control: the prompt must end with the assistant generation header.
    tail = tokenizer.decode(prompt_ids[-3:])
    assert tail == C.GENERATION_SUFFIX, f"row {row_label}: prompt tail {tail!r}"
    n_prefix_tokens = sum(1 for start, end in offsets if end <= prefix_char_end and end > start)
    positions = {
        "n_total": n_total,
        "n_prompt": n_prompt,
        "prefix_end": min(max(0, n_prefix_tokens - 1), n_total - 1),
        "context_end": n_prompt - 1,
        "answer_start": n_prompt,
        "answer_end": n_total,
        "v_last_turn": n_total - 1,
        "n_completion": len(completion_ids),
    }
    return row_ids, positions


def capture_rows(
    model,
    tokenizer,
    rows: list[dict],
    *,
    device: str,
    batch_size: int,
    log_label: str,
):
    """Batched right-padded teacher-forced capture -> four (n, L, H) fp16 stacks.

    ``rows``: dicts with ``prompt``, ``prefix_char_end``, ``completion``.
    Returns dict of torch fp16 tensors {cx_last, prefix_end, v_mean,
    v_last_turn} + ``positions`` list. Fails loud on layer/hidden-dim
    mismatches; never truncates.
    """
    import torch

    from explore_persona_space.analysis.extraction import _logits_to_keep_kwargs

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if getattr(tokenizer, "padding_side", "right") != "right":
        raise ValueError(
            f"capture positions index the UNPADDED sequence (right padding required); "
            f"padding_side={tokenizer.padding_side!r}"
        )
    ltk = _logits_to_keep_kwargs(model, False)
    out = {k: [] for k in ("cx_last", "prefix_end", "v_mean", "v_last_turn")}
    positions_all: list[dict] = []
    n_rows = len(rows)
    for bs in range(0, n_rows, max(1, batch_size)):
        be = min(bs + max(1, batch_size), n_rows)
        if bs % (max(1, batch_size) * 5) == 0:
            logger.info("[%s] capture rows %d:%d/%d", log_label, bs, be, n_rows)
        batch_ids, batch_pos = [], []
        for local_i, row in enumerate(rows[bs:be]):
            ids, pos = row_ids_and_positions(
                tokenizer,
                row["prompt"],
                row["prefix_char_end"],
                row["completion"],
                row_label=f"{log_label}:{bs + local_i}",
            )
            batch_ids.append(ids)
            batch_pos.append(pos)
        inputs = tokenizer.pad({"input_ids": batch_ids}, return_tensors="pt", padding=True)
        input_ids = inputs["input_ids"].to(device)
        attention_mask = inputs["attention_mask"].to(device)
        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                **ltk,
            )
        hs = outputs.hidden_states[1:]  # per-decoder-layer outputs
        for local_i, pos in enumerate(batch_pos):

            def _pos(p: int, *, ri: int = local_i, hs_layers=hs):
                return torch.stack([h[ri, p, :].to(torch.float16).cpu() for h in hs_layers], dim=0)

            def _span(a: int, b: int, *, ri: int = local_i, hs_layers=hs):
                return torch.stack(
                    [h[ri, a:b, :].mean(dim=0).to(torch.float16).cpu() for h in hs_layers], dim=0
                )

            out["cx_last"].append(_pos(pos["context_end"]))
            out["prefix_end"].append(_pos(pos["prefix_end"]))
            out["v_mean"].append(_span(pos["answer_start"], pos["answer_end"]))
            out["v_last_turn"].append(_pos(pos["v_last_turn"]))
            positions_all.append(pos)
        del outputs, hs, input_ids, attention_mask
    stacked = {k: torch.stack(v, dim=0) for k, v in out.items()}  # (n, L, H) fp16
    for k, t in stacked.items():
        assert t.shape[0] == n_rows and t.ndim == 3, (k, tuple(t.shape))
    return stacked, positions_all, int(stacked["cx_last"].shape[1])


def identity_gate(
    model, tokenizer, rows: list[dict], stacked, gate_indices: list[int], *, device: str
) -> dict:
    """Batch-1 PROMPT-ONLY forward vs the batched full-row capture (G2-style).

    Causality: prompt states are unaffected by the appended completion, so the
    batch-1 prompt forward's last-token / prefix-position states must match the
    batched capture's ``cx_last`` / ``prefix_end`` up to bf16 padded-batch
    numerics. Two-bar per gotchas.md (#779 r12): early layers per-layer cosine
    >= 0.999, flattened all-layer >= 0.995. Raises on failure.
    """
    import torch

    report = {"rows": [], "early_cos_min": 1.0, "flat_cos_min": 1.0}
    for gi in gate_indices:
        row = rows[gi]
        enc = tokenizer(row["prompt"], add_special_tokens=False, return_tensors="pt")
        with torch.no_grad():
            outputs = model(
                input_ids=enc["input_ids"].to(device),
                attention_mask=enc["attention_mask"].to(device),
                output_hidden_states=True,
            )
        hs = outputs.hidden_states[1:]
        _, pos = row_ids_and_positions(
            tokenizer, row["prompt"], row["prefix_char_end"], row["completion"]
        )
        for key, p in (("cx_last", pos["context_end"]), ("prefix_end", pos["prefix_end"])):
            ref = torch.stack([h[0, p, :].float().cpu() for h in hs], dim=0)  # (L, H)
            got = stacked[key][gi].float()  # (L, H)
            cos_l = torch.nn.functional.cosine_similarity(ref, got, dim=1)  # (L,)
            n_early = min(GATE_EARLY_LAYERS, cos_l.shape[0])
            early_min = float(cos_l[:n_early].min())
            flat = float(
                torch.nn.functional.cosine_similarity(
                    ref.flatten().unsqueeze(0), got.flatten().unsqueeze(0)
                )
            )
            report["rows"].append({"row": gi, "key": key, "early_min": early_min, "flat": flat})
            report["early_cos_min"] = min(report["early_cos_min"], early_min)
            report["flat_cos_min"] = min(report["flat_cos_min"], flat)
        del outputs, hs
    if report["early_cos_min"] < GATE_EARLY_COS_MIN or report["flat_cos_min"] < GATE_FLAT_COS_MIN:
        raise RuntimeError(f"identity gate FAILED: {report}")
    logger.info(
        "[gate] identity gate PASS early_min=%.6f flat_min=%.6f (%d reads)",
        report["early_cos_min"],
        report["flat_cos_min"],
        len(report["rows"]),
    )
    return report


# ── per-family upload (one create_commit per family; retried) ─────────────────


def upload_files(paths_named: list[tuple[Path, str]], message: str) -> None:
    """One retried create_commit for a family's artifacts (never per-file loops)."""
    from huggingface_hub import CommitOperationAdd, HfApi

    from explore_persona_space.orchestrate.hub import retry_transient

    ops = [CommitOperationAdd(path_in_repo=dest, path_or_fileobj=str(p)) for p, dest in paths_named]
    api = HfApi()
    retry_transient(
        lambda: api.create_commit(
            repo_id=C.HF_DATA_REPO, repo_type="dataset", commit_message=message, operations=ops
        ),
        what=message,
    )
    logger.info("[upload] %s (%d files)", message, len(ops))


# ── stage: inputs ─────────────────────────────────────────────────────────────


def resolve_families(arg: str) -> list[str]:
    """Family subset resolution — the ONE cell-list source every phase reads."""
    _sources, targets = C.family_labels()
    if arg == "all":
        return targets
    fams = [f.strip() for f in arg.split(",") if f.strip()]
    unknown = [f for f in fams if f not in targets]
    if unknown:
        raise ValueError(f"unknown families {unknown}; valid: {targets}")
    return fams


def stage_inputs(args) -> dict:
    """Stage bank + rewrites + (optionally) the #545 inputs; registry asserts."""
    C.phase("p1_inputs")
    root = C.data_root(args.smoke, args.out_root)
    inputs_dir = root / "inputs"
    bank_path = C.ensure_input(inputs_dir / C.BANK_FILE, f"inputs/{C.BANK_FILE}")
    rewrites_path = C.ensure_input(inputs_dir / C.REWRITES_FILE, f"inputs/{C.REWRITES_FILE}")
    bank = C.load_bank(bank_path)
    if args.n_queries > 0:
        bank = bank[: args.n_queries]
    fams = resolve_families(args.families)
    logger.info(
        "[inputs] %d families x %d queries; bank sha=%s",
        len(fams),
        len(bank),
        C.sha256_file(bank_path)[:12],
    )

    staged_545 = {"rows": [], "cols": [], "regen_cols": []}
    if args.behaviors != "none":
        staged_545 = stage_545_inputs(root, args)
    return {
        "bank": bank,
        "bank_sha256": C.sha256_file(bank_path),
        "rewrites_path": rewrites_path,
        "families": fams,
        "root": root,
        "i545": staged_545,
    }


def stage_545_inputs(root: Path, args) -> dict:
    """Stage the #545 corpora + demo realization pools from HF (scoped, per-file).

    Returns {"rows": [(row_id, corpus_rel)], "cols": [(column_id, demo_rel)],
    "regen_cols": [...]} limited to rows whose train_lora corpus resolves and
    columns whose diagonal demo file exists OR is regenerable per the frozen
    #545 protocol (r3 regen add-on, plan assumption 5 primary leg); genuinely
    non-regenerable pools stay on the descope path with the reason logged.
    """
    from huggingface_hub import hf_hub_download, list_repo_tree

    from explore_persona_space.experiments.behavior_testbed_545.rows import active_rows
    from explore_persona_space.orchestrate.hub import retry_transient

    dest = root / "i545"
    dest.mkdir(parents=True, exist_ok=True)
    prefix = f"{C.I545_HF_PREFIX}/corpora"
    entries = retry_transient(
        lambda: list(
            # HUB_VERIFY_RETRY_EXEMPT: thunk runs inside hub.retry_transient (#920-safe)
            list_repo_tree(C.HF_DATA_REPO, path_in_repo=prefix, repo_type="dataset", recursive=True)
        ),
        what=f"list_repo_tree {prefix}",
    )
    hub_paths = {e.path for e in entries}

    def _fetch(hub_rel: str, local: Path) -> Path:
        if local.exists() and local.stat().st_size > 0:
            return local
        got = retry_transient(
            lambda hr=hub_rel: hf_hub_download(
                repo_id=C.HF_DATA_REPO, repo_type="dataset", filename=hr, revision="main"
            ),
            what=f"hf_hub_download {hub_rel}",
        )
        import shutil

        local.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(got, local)
        return local

    rows_sel = []
    want = None if args.behaviors == "all" else {b.strip() for b in args.behaviors.split(",")}
    for row in active_rows().values():
        if want is not None and row.row_id not in want:
            continue
        if not row.corpus:
            logger.info(
                "[545] row %s has no train_lora corpus (recipe %s) — skipped",
                row.row_id,
                row.recipe_kind,
            )
            continue
        hub_rel = f"{prefix}/{row.corpus}"
        if hub_rel not in hub_paths:
            logger.warning(
                "[545] corpus missing on HF for row %s: %s — skipped", row.row_id, hub_rel
            )
            continue
        local = _fetch(hub_rel, dest / "corpora" / row.corpus)
        rows_sel.append((row.row_id, str(local), row.diagonal_column))

    cols_sel = []
    want_cols = None if args.eval_cols == "all" else {c.strip() for c in args.eval_cols.split(",")}
    seen_cols: set[str] = set()
    for row in active_rows().values():
        col = row.diagonal_column
        if not col or col in seen_cols:
            continue
        if want_cols is not None and col not in want_cols:
            continue
        demo_rel = f"{prefix}/demos/{row.row_id}.json"
        if demo_rel not in hub_paths:
            continue
        local = _fetch(demo_rel, dest / "demos" / f"{row.row_id}.json")
        cols_sel.append((col, str(local)))
        seen_cols.add(col)

    regen_cols = _regen_missing_545_pools(
        args,
        dest=dest,
        prefix=prefix,
        hub_paths=hub_paths,
        fetch=_fetch,
        cols_sel=cols_sel,
        seen_cols=seen_cols,
        want_cols=want_cols,
    )

    logger.info(
        "[545] staged %d behavior corpora + %d eval-column demo pools (%d regenerated: %s)",
        len(rows_sel),
        len(cols_sel),
        len(regen_cols),
        ",".join(regen_cols) or "-",
    )
    return {"rows": rows_sel, "cols": cols_sel, "regen_cols": regen_cols}


def _regen_missing_545_pools(
    args,
    *,
    dest: Path,
    prefix: str,
    hub_paths: set[str],
    fetch,
    cols_sel: list[tuple[str, str]],
    seen_cols: set[str],
    want_cols: set[str] | None,
) -> list[str]:
    """Regen add-on (r3, concern ood545-coverage-partial): missing demo pools.

    Plan assumption 5 primary leg + the allowed-deviations "OOD-arm
    regeneration of missing realization pools": a missing diagonal demo pool
    is regenerated per the FROZEN #545 protocol
    (``behavior_testbed_545.corpora.demo_pool_from_corpus_rows`` — K=8,
    answer-length terciles, ``random.Random(545)``) over the row's REALIZED
    training corpus, ONLY when that source resolves under the #545 corpora
    prefix. Non-regenerable classes keep the descope leg, reason logged:

    - hydra_turner rows: the #545 demo source is the Turner EDS JSONL
      (TURNER_EDS_PASSWORD-gated, not on HF) — not a small add-on;
    - reuse_adapter rows: #545 itself never built these pools (INDEX.json
      "pending-p1"; predictors_zoo skips their demos flavor) — regenerating
      would be a NEW variant, not the frozen recipe.

    Regenerated pools persist to the ISSUE prefix (never the frozen #545
    prefix — upload-policy version-bump rule) and are reused idempotently.
    Appends covered columns to ``cols_sel``/``seen_cols``; returns the list
    of regenerated column ids.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.experiments.behavior_testbed_545.corpora import (
        demo_pool_from_corpus_rows,
    )
    from explore_persona_space.experiments.behavior_testbed_545.rows import active_rows
    from explore_persona_space.orchestrate.hub import retry_transient

    api = HfApi()
    regen_cols: list[str] = []
    for row in active_rows().values():
        col = row.diagonal_column
        if not col or col in seen_cols:
            continue
        if want_cols is not None and col not in want_cols:
            continue
        if row.recipe_kind == "hydra_turner":
            logger.info(
                "[545-regen] col %s (row %s): demo source is the password-gated Turner "
                "EDS JSONL (not on HF) — descope stands",
                col,
                row.row_id,
            )
            continue
        if not row.corpus:
            logger.info(
                "[545-regen] col %s (row %s, %s): #545 never built this pool "
                "(pending-p1) — no frozen recipe; descope stands",
                col,
                row.row_id,
                row.recipe_kind,
            )
            continue
        corpus_rel = f"{prefix}/{row.corpus}"
        if corpus_rel not in hub_paths:
            logger.warning(
                "[545-regen] col %s (row %s): source corpus %s unresolved on HF — descoped",
                col,
                row.row_id,
                corpus_rel,
            )
            continue
        local = dest / "demos" / f"{row.row_id}.json"
        regen_rel = f"{C.HF_PREFIX}/i545_regen/demos/{row.row_id}.json"
        prior = retry_transient(
            lambda rr=regen_rel: api.file_exists(C.HF_DATA_REPO, rr, repo_type="dataset"),
            what=f"file_exists {regen_rel}",
        )
        if prior:
            fetch(regen_rel, local)
            logger.info("[545-regen] col %s: reusing prior regen pool %s", col, regen_rel)
        else:
            corpus_local = fetch(corpus_rel, dest / "corpora" / row.corpus)
            corpus_rows = []
            with open(corpus_local, encoding="utf-8") as f:  # text-mode iter (#950 rule)
                for line in f:
                    if line.strip():
                        corpus_rows.append(json.loads(line))
            demos, n_parsable = demo_pool_from_corpus_rows(corpus_rows)
            if demos is None:
                logger.warning(
                    "[545-regen] col %s (row %s): only %d parsable corpus rows (<8) — descoped",
                    col,
                    row.row_id,
                    n_parsable,
                )
                continue
            local.parent.mkdir(parents=True, exist_ok=True)
            C.write_json_atomic(
                local,
                {
                    "demos": demos,
                    "metadata": C.reproducibility_metadata(
                        {
                            "regen": "issue1332 — frozen #545 build_demo_sets protocol "
                            "(K=8 answer-length terciles, seed 545)",
                            "source_corpus": corpus_rel,
                            "source_sha256": C.sha256_file(corpus_local),
                        }
                    ),
                },
            )
            if not args.skip_upload:
                upload_files(
                    [(local, regen_rel)],
                    f"issue 1332: #545 regen demo pool {row.row_id} ({col})",
                )
            logger.info(
                "[545-regen] col %s: regenerated pool from %s (%d parsable rows) -> %s",
                col,
                corpus_rel,
                n_parsable,
                regen_rel,
            )
        cols_sel.append((col, str(local)))
        seen_cols.add(col)
        regen_cols.append(col)
    return regen_cols


# ── stage: gen ────────────────────────────────────────────────────────────────


def _vllm_engine(tiny: bool):
    """Build the vLLM engine (production) — tiny mode never calls this."""
    assert not tiny
    from vllm import LLM

    return LLM(
        model=C.BASE_MODEL,
        max_model_len=C.MAX_MODEL_LEN,
        gpu_memory_utilization=0.85,
        enforce_eager=os.environ.get("EPM_VLLM_ENFORCE_EAGER", "0") == "1",
        enable_prefix_caching=os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING", "0") != "1",
    )


def _greedy_vllm(llm, prompts: list[str], max_new: int) -> tuple[list[str], list[int]]:
    """Chunked greedy generation (deadlock-prevention chunking; use_tqdm=False)."""
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=max_new)
    texts: list[str] = []
    n_toks: list[int] = []
    n_chunks = (len(prompts) + GEN_CHUNK - 1) // GEN_CHUNK
    for i in range(0, len(prompts), GEN_CHUNK):
        chunk = prompts[i : i + GEN_CHUNK]
        logger.info(
            "[vllm-chunk] greedy chunk %d/%d (%d prompts)", i // GEN_CHUNK + 1, n_chunks, len(chunk)
        )
        outs = llm.generate(chunk, sp, use_tqdm=False)
        for o in outs:
            texts.append(o.outputs[0].text)
            n_toks.append(len(o.outputs[0].token_ids))
    return texts, n_toks


def _greedy_tiny(model, tokenizer, prompts: list[str], max_new: int) -> tuple[list[str], list[int]]:
    """Tiny-model greedy generation (CPU; same code path shape as production)."""
    import torch

    texts, n_toks = [], []
    cap = min(max_new, 16)
    for i in range(0, len(prompts), 8):
        chunk = prompts[i : i + 8]
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        enc = tokenizer(chunk, return_tensors="pt", padding=True, add_special_tokens=False)
        tokenizer.padding_side = "right"
        with torch.no_grad():
            gen = model.generate(
                **enc, max_new_tokens=cap, do_sample=False, pad_token_id=tokenizer.pad_token_id
            )
        for bi in range(len(chunk)):
            new_ids = gen[bi, enc["input_ids"].shape[1] :]
            texts.append(tokenizer.decode(new_ids, skip_special_tokens=True))
            n_toks.append(int(new_ids.shape[0]))
    return texts, n_toks


def stage_gen(args) -> None:
    """vLLM greedy R_c(q) per family, per-family checkpoint + upload."""
    staged = stage_inputs(args)
    bank, fams, root = staged["bank"], staged["families"], staged["root"]
    tok = get_tokenizer()
    rewrites = C.load_rewrites(staged["rewrites_path"])["rewrites"]
    panel = C.instructed_panel()
    gen_dir = root / "raw_completions" / "generation"
    gen_dir.mkdir(parents=True, exist_ok=True)

    llm = None
    tinym = None
    if args.tiny_model:
        tinym = tiny_model(tok)
    C.phase("p1_gen")
    for fam in fams:
        out_path = gen_dir / f"{fam}.json"
        if out_path.exists():
            existing = json.loads(out_path.read_text())
            if existing.get("bank_sha256") == staged["bank_sha256"] and existing.get("n") == len(
                bank
            ):
                logger.info("[gen] %s already complete (resume skip)", fam)
                continue
            raise RuntimeError(
                f"[gen] {out_path} exists with a DIFFERENT regime "
                f"(bank_sha/n mismatch) — refusing to silently reuse"
            )
        prompts, prefix_ends = [], []
        for q in bank:
            p, pce = C.render_family_prompt(fam, q, tok, rewrites, panel)
            prompts.append(p)
            prefix_ends.append(pce)
        t0 = time.time()
        n_trunc_initial = 0
        if args.tiny_model:
            texts, n_toks = _greedy_tiny(tinym, tok, prompts, C.MAX_NEW_TOKENS)
            n_trunc_initial = sum(1 for n in n_toks if n >= C.MAX_NEW_TOKENS)
        else:
            if llm is None:
                llm = _vllm_engine(tiny=False)
            texts, n_toks = _greedy_vllm(llm, prompts, C.MAX_NEW_TOKENS)
            # plan assumption 9: >2% truncation -> re-generate truncated rows at 2048
            trunc_idx = [i for i, n in enumerate(n_toks) if n >= C.MAX_NEW_TOKENS]
            n_trunc_initial = len(trunc_idx)
            if len(trunc_idx) / max(1, len(bank)) > TRUNCATION_REGEN_THRESHOLD:
                logger.warning(
                    "[gen] %s truncation %.3f > %.2f — re-gen %d rows at %d",
                    fam,
                    len(trunc_idx) / len(bank),
                    TRUNCATION_REGEN_THRESHOLD,
                    len(trunc_idx),
                    REGEN_MAX_NEW,
                )
                re_texts, re_toks = _greedy_vllm(
                    llm, [prompts[i] for i in trunc_idx], REGEN_MAX_NEW
                )
                for k, i in enumerate(trunc_idx):
                    texts[i], n_toks[i] = re_texts[k], re_toks[k]
        # r1 Minor: `truncation_rate` counts UNRECOVERED rows (>= REGEN_MAX_NEW
        # post-regen); a <=2% share truncated at MAX_NEW_TOKENS never regenerates
        # and would read ~0 there — persist the pre-regen rate at the actual cap
        # alongside so the analyzer sees the true truncated-row share.
        trunc_rate = sum(1 for n in n_toks if n >= REGEN_MAX_NEW) / max(1, len(bank))
        trunc_rate_at_cap = n_trunc_initial / max(1, len(bank))
        payload = {
            "family": fam,
            "questions": bank,
            "responses": texts,
            "response_token_counts": n_toks,
            "prefix_char_ends": prefix_ends,
            "n": len(bank),
            "bank_sha256": staged["bank_sha256"],
            "truncation_rate": trunc_rate,
            "truncation_rate_at_max_new": trunc_rate_at_cap,
            "gen_seconds": time.time() - t0,
            "sampling": {
                "temperature": 0.0,
                "max_new_tokens": C.MAX_NEW_TOKENS,
                "regen_max_new_tokens": REGEN_MAX_NEW,
                "engine": "tiny" if args.tiny_model else "vllm",
            },
            "reproducibility_metadata": C.reproducibility_metadata({"smoke": args.smoke}),
        }
        C.write_json_atomic(out_path, payload)
        logger.info("[gen] family %s complete (trunc %.4f) -> %s", fam, trunc_rate, out_path)
        if not args.skip_upload:
            upload_files(
                [(out_path, f"{C.HF_PREFIX}/raw_completions/generation/{fam}.json")],
                f"issue 1332: rollouts {fam}",
            )
    print("[gen-stage] complete", flush=True)


# ── stage: capture (marker families) ──────────────────────────────────────────


def _resolve_device(args) -> str:
    if args.tiny_model:
        return "cpu"
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("capture requires CUDA (or --tiny-model for the CPU smoke)")
    return "cuda:0"


def _load_capture_model(args, device: str):
    import torch

    tok = get_tokenizer()
    if args.tiny_model:
        return tiny_model(tok), tok
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        C.BASE_MODEL, torch_dtype=torch.bfloat16, device_map={"": 0}
    )
    model.eval()
    return model, tok


def stage_capture(args) -> None:
    """Teacher-forced 28-layer capture per marker family; per-family shard + upload."""
    staged = stage_inputs(args)
    bank, fams, root = staged["bank"], staged["families"], staged["root"]
    device = _resolve_device(args)
    model, tok = _load_capture_model(args, device)
    rewrites = C.load_rewrites(staged["rewrites_path"])["rewrites"]
    panel = C.instructed_panel()
    gen_dir = root / "raw_completions" / "generation"
    cap_dir = root / "store" / "capture"
    cap_dir.mkdir(parents=True, exist_ok=True)
    import torch

    C.phase("p1_capture")
    gated_classes: set[str] = set()
    dropped: list[str] = []
    for fam in fams:
        shard_path = cap_dir / f"{fam}.pt"
        if shard_path.exists():
            # r1 Minor: regime-keyed resume (mirrors gen) — a stale shard from a
            # different bank is refused loudly, never silently reused.
            sh_meta = torch.load(shard_path, map_location="cpu", mmap=True, weights_only=False)
            if sh_meta.get("bank_sha256") != staged["bank_sha256"]:
                raise RuntimeError(
                    f"[capture] {shard_path} exists under a DIFFERENT regime "
                    f"(bank_sha256 mismatch) — refusing to silently reuse"
                )
            del sh_meta
            logger.info("[capture] %s shard exists (resume skip)", fam)
            continue
        roll = json.loads((gen_dir / f"{fam}.json").read_text())
        assert roll["bank_sha256"] == staged["bank_sha256"], f"rollout/bank drift for {fam}"
        rows, bank_indices = [], []
        for qi, (q, r) in enumerate(zip(roll["questions"], roll["responses"], strict=True)):
            n_resp = len(tok.encode(r, add_special_tokens=False)) if r else 0
            if n_resp < C.VALID_MIN_RESPONSE_TOKENS:
                continue
            prompt, pce = C.render_family_prompt(fam, q, tok, rewrites, panel)
            rows.append({"prompt": prompt, "prefix_char_end": pce, "completion": r})
            bank_indices.append(qi)
        valid_frac = len(rows) / max(1, len(bank))
        if valid_frac < C.FAMILY_VALID_FLOOR:
            logger.warning(
                "[capture] family %s below valid floor (%.2f < %.2f) — DROPPED",
                fam,
                valid_frac,
                C.FAMILY_VALID_FLOOR,
            )
            dropped.append(fam)
            C.write_json_atomic(
                cap_dir / f"{fam}.dropped.json", {"family": fam, "valid_frac": valid_frac}
            )
            continue
        stacked, _positions, n_layers = capture_rows(
            model, tok, rows, device=device, batch_size=args.capture_batch_size, log_label=fam
        )
        fam_class = "instr" if fam.startswith("instr_") else fam[0]
        if fam_class not in gated_classes:
            gate_report = identity_gate(
                model, tok, rows, stacked, list(range(min(3, len(rows)))), device=device
            )
            gated_classes.add(fam_class)
        else:
            gate_report = None
        shard = {
            **{k: v for k, v in stacked.items()},
            "bank_indices": torch.tensor(bank_indices, dtype=torch.long),
            "questions": [bank[i] for i in bank_indices],
            "n_bank": len(bank),  # exact bank size (r1 Minor: never infer max+1)
            "n_layers": n_layers,
            "hidden_dim": int(stacked["cx_last"].shape[2]),
            "family": fam,
            "valid_frac": valid_frac,
            "bank_sha256": staged["bank_sha256"],
            "gate_report": gate_report,
            "meta": C.reproducibility_metadata({"smoke": args.smoke, "arm": "marker"}),
        }
        tmp = shard_path.with_suffix(".pt.tmp")
        torch.save(shard, tmp)
        os.replace(tmp, shard_path)
        logger.info(
            "[capture] family %s -> %s (%d rows, %d layers)", fam, shard_path, len(rows), n_layers
        )
        if not args.skip_upload:
            upload_files(
                [(shard_path, f"{C.HF_PREFIX}/analysis_tensors/capture/{fam}.pt")],
                f"issue 1332: capture shard {fam}",
            )
    if dropped:
        C.write_json_atomic(cap_dir / "dropped_families.json", {"dropped": dropped})
    print("[capture-stage] complete", flush=True)


# ── stage: capture545 (OOD arm) ───────────────────────────────────────────────


def _i545_pairs_from_corpus(path: Path, cap: int) -> list[tuple[str, str, list[dict]]]:
    """(question, answer, prompt_messages) triples from a #545 train_lora jsonl."""
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt_msgs = row.get("prompt") or []
            completion = row.get("completion") or []
            users = [m for m in prompt_msgs if m.get("role") == "user"]
            if not users or not completion:
                continue
            pairs.append((users[-1]["content"], completion[0]["content"], prompt_msgs))
            if len(pairs) >= cap:
                break
    return pairs


def _i545_pairs_from_demos(path: Path, cap: int) -> list[tuple[str, str, list[dict]]]:
    """(question, answer, [user msg]) triples from a #545 demo realization pool."""
    demos = json.loads(path.read_text())["demos"][:cap]
    return [
        (d["question"], d["answer"], [{"role": "user", "content": d["question"]}]) for d in demos
    ]


def stage_capture545(args) -> None:
    """Capture the #545 behavior rows + eval-column realization pools (OOD arm)."""
    staged = stage_inputs(args)
    if args.behaviors == "none":
        logger.info("[545] disabled (--behaviors none)")
        print("[capture545-stage] complete", flush=True)
        return
    root = staged["root"]
    device = _resolve_device(args)
    model, tok = _load_capture_model(args, device)
    cap_dir = root / "store" / "capture545"
    cap_dir.mkdir(parents=True, exist_ok=True)
    import torch

    C.phase("p1_capture545")
    units: list[tuple[str, str, Path]] = []  # (unit_id, kind, source path)
    for row_id, corpus_path, _diag in staged["i545"]["rows"]:
        units.append((f"row__{row_id}", "corpus", Path(corpus_path)))
    for col_id, demo_path in staged["i545"]["cols"]:
        units.append((f"col__{col_id}", "demos", Path(demo_path)))
    gated = False
    for unit_id, kind, src in units:
        shard_path = cap_dir / f"{unit_id}.pt"
        if shard_path.exists():
            # r1 Minor: regime-keyed resume — the row cap is the output-affecting
            # knob for 545 units (source pools are HF-pinned upstream).
            sh_meta = torch.load(shard_path, map_location="cpu", mmap=True, weights_only=False)
            if int(sh_meta.get("n_rows_545_cap", -1)) != int(args.n_rows_545):
                raise RuntimeError(
                    f"[545] {shard_path} exists under a DIFFERENT regime "
                    f"(n_rows_545 cap mismatch) — refusing to silently reuse"
                )
            del sh_meta
            logger.info("[545] %s shard exists (resume skip)", unit_id)
            continue
        pairs = (
            _i545_pairs_from_corpus(src, args.n_rows_545)
            if kind == "corpus"
            else _i545_pairs_from_demos(src, args.n_rows_545)
        )
        rows = []
        for _q, a, prompt_msgs in pairs:
            if len(tok.encode(a, add_special_tokens=False)) < C.VALID_MIN_RESPONSE_TOKENS:
                continue
            prompt = tok.apply_chat_template(
                prompt_msgs, tokenize=False, add_generation_prompt=True
            )
            n_prompt_toks = len(tok.encode(prompt, add_special_tokens=False))
            n_ans = len(tok.encode(a, add_special_tokens=False))
            if n_prompt_toks + n_ans + 8 > C.MAX_MODEL_LEN:
                continue  # load-time length filter (#952 rule); drop over-budget rows
            idx = prompt.rindex(C.USER_TURN_HEADER)
            rows.append(
                {
                    "prompt": prompt,
                    "prefix_char_end": idx + len(C.USER_TURN_HEADER),
                    "completion": a,
                }
            )
        if len(rows) < 8:
            logger.warning(
                "[545] unit %s has %d usable rows — skipped (underpowered)", unit_id, len(rows)
            )
            C.write_json_atomic(
                cap_dir / f"{unit_id}.skipped.json", {"unit": unit_id, "n": len(rows)}
            )
            continue
        stacked, _positions, n_layers = capture_rows(
            model, tok, rows, device=device, batch_size=args.capture_batch_size, log_label=unit_id
        )
        if not gated:
            identity_gate(model, tok, rows, stacked, list(range(min(3, len(rows)))), device=device)
            gated = True
        shard = {
            **{k: v for k, v in stacked.items()},
            "n_layers": n_layers,
            "hidden_dim": int(stacked["cx_last"].shape[2]),
            "unit": unit_id,
            "kind": kind,
            "n_rows": len(rows),
            "n_rows_545_cap": int(args.n_rows_545),  # resume regime key (r1 Minor)
            "meta": C.reproducibility_metadata(
                {"smoke": args.smoke, "arm": "i545", "off_policy_targets": True}
            ),
        }
        tmp = shard_path.with_suffix(".pt.tmp")
        torch.save(shard, tmp)
        os.replace(tmp, shard_path)
        logger.info("[545] unit %s -> %s (%d rows)", unit_id, shard_path, len(rows))
        if not args.skip_upload:
            upload_files(
                [(shard_path, f"{C.HF_PREFIX}/analysis_tensors/capture545/{unit_id}.pt")],
                f"issue 1332: capture545 shard {unit_id}",
            )
    print("[capture545-stage] complete", flush=True)


# ── stage: upload (idempotent exact-set verification sweep) ───────────────────


def stage_upload(args) -> dict:
    """Verify every produced artifact resolves on the Hub; upload any missing."""
    staged = stage_inputs(args)
    root = staged["root"]
    C.phase("p1_upload")
    produced: list[tuple[Path, str]] = []
    for sub, hub_sub in (
        ("raw_completions/generation", "raw_completions/generation"),
        ("store/capture", "analysis_tensors/capture"),
        ("store/capture545", "analysis_tensors/capture545"),
    ):
        d = root / sub
        if not d.is_dir():
            continue
        for p in sorted(d.iterdir()):
            if p.suffix in (".json", ".pt") and ".tmp" not in p.name:
                produced.append((p, f"{C.HF_PREFIX}/{hub_sub}/{p.name}"))
    import inspect

    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import verify_repo_paths_uploaded

    # Signature-bind check on EVERY invocation — including --skip-upload smokes —
    # so this previously smoke-fenced call site can never silently drift from the
    # helper's live signature again (r1 Critical 2: the call omitted the `api`
    # positional + the REQUIRED kw-only `path_in_repo` -> TypeError at the
    # terminal upload stage). Pinned by tests/test_issue1332_pins.py.
    inspect.signature(verify_repo_paths_uploaded).bind(
        HfApi(), C.HF_DATA_REPO, [], path_in_repo=C.HF_PREFIX, repo_type="dataset"
    )
    if args.skip_upload:
        logger.info("[upload] skip-upload: %d artifacts produced locally", len(produced))
        return {"n_produced": len(produced), "verified": False}

    api = HfApi()
    missing_pairs = []
    expected = [dest for _p, dest in produced]
    missing = verify_repo_paths_uploaded(
        api, C.HF_DATA_REPO, expected, path_in_repo=C.HF_PREFIX, repo_type="dataset"
    )
    if missing:
        missing_set = set(missing)
        missing_pairs = [(p, d) for p, d in produced if d in missing_set]
        logger.info("[upload] %d/%d missing on Hub — uploading", len(missing_pairs), len(produced))
        for i in range(0, len(missing_pairs), 20):
            upload_files(missing_pairs[i : i + 20], f"issue 1332: upload sweep ({i})")
        still = verify_repo_paths_uploaded(
            api, C.HF_DATA_REPO, expected, path_in_repo=C.HF_PREFIX, repo_type="dataset"
        )
        if still:
            raise RuntimeError(
                f"upload sweep FAILED to land {len(still)} files: {sorted(still)[:5]}"
            )
    logger.info("[upload] %d artifacts verified on Hub", len(produced))
    return {"n_produced": len(produced), "verified": True}


# ── driver ────────────────────────────────────────────────────────────────────


def _forward_args(args, stage: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--stage",
        stage,
        "--families",
        args.families,
        "--n-queries",
        str(args.n_queries),
        "--behaviors",
        args.behaviors,
        "--eval-cols",
        args.eval_cols,
        "--n-rows-545",
        str(args.n_rows_545),
        "--capture-batch-size",
        str(args.capture_batch_size),
    ]
    if args.smoke:
        cmd.append("--smoke")
    if args.tiny_model:
        cmd.append("--tiny-model")
    if args.skip_upload:
        cmd.append("--skip-upload")
    if args.out_root:
        cmd.extend(["--out-root", args.out_root])
    return cmd


def main() -> int:
    """P1 driver. ``--stage all`` = subprocess-isolated gen -> capture -> capture545."""
    ap = argparse.ArgumentParser(description="Issue #1332 P1 GPU phase")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--full", action="store_true", help="production defaults (all families)")
    mode.add_argument(
        "--smoke", action="store_true", help="scratch roots + epm:smoke-result sentinel"
    )
    ap.add_argument(
        "--stage",
        default="all",
        choices=["all", "inputs", "gen", "capture", "capture545", "upload"],
    )
    ap.add_argument("--families", default="all")
    ap.add_argument("--n-queries", type=int, default=0, help="0 = full bank")
    ap.add_argument("--behaviors", default="all", help="'none' disables the #545 arm")
    ap.add_argument("--eval-cols", default="all")
    ap.add_argument("--n-rows-545", type=int, default=400)
    ap.add_argument("--capture-batch-size", type=int, default=16)
    ap.add_argument(
        "--tiny-model",
        action="store_true",
        help="from-config 2-layer same-arch model on CPU (tiny-real smoke)",
    )
    ap.add_argument("--out-root", default=None)
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    t0 = time.time()

    if args.stage != "all":
        {
            "inputs": lambda: stage_inputs(args),
            "gen": lambda: stage_gen(args),
            "capture": lambda: stage_capture(args),
            "capture545": lambda: stage_capture545(args),
            "upload": lambda: stage_upload(args),
        }[args.stage]()
        return 0

    # --stage all: subprocess-isolate each framework phase (vLLM teardown gotcha).
    stage_inputs(args)  # fail-fast input staging + registry asserts, in-process
    for stage in ("gen", "capture", "capture545"):
        cmd = _forward_args(args, stage)
        logger.info("[dispatch] %s", " ".join(cmd))
        proc = subprocess.run(cmd, env={**os.environ}, check=False)
        if proc.returncode != 0:
            C.write_sentinel(
                "epm:failure",
                json.dumps(
                    {
                        "failure_class": "code",
                        "reason": f"stage {stage} exited rc={proc.returncode}",
                        "assert_tag": f"i1332-{stage}-rc{proc.returncode}",
                    }
                ),
            )
            return proc.returncode
    # Same failure-sentinel contract as the stage subprocesses: an exception out
    # of the in-process terminal upload previously died rc!=0 with NO sentinel
    # (r1 Critical 2 impact) — the poller now sees an epm:failure either way.
    try:
        upload_info = stage_upload(args)
    except Exception as e:
        C.write_sentinel(
            "epm:failure",
            json.dumps(
                {
                    "failure_class": "infra",
                    "reason": f"stage upload raised {type(e).__name__}: {e}",
                    "assert_tag": "i1332-upload-exc",
                }
            ),
        )
        raise

    gpu_hours = (time.time() - t0) / 3600.0
    root = C.data_root(args.smoke, args.out_root)
    fams = resolve_families(args.families)
    gen_dir = root / "raw_completions" / "generation"
    trunc = {}
    valid = {}
    for fam in fams:
        p = gen_dir / f"{fam}.json"
        if p.exists():
            roll = json.loads(p.read_text())
            trunc[fam] = roll.get("truncation_rate")
        sp = root / "store" / "capture" / f"{fam}.pt"
        valid[fam] = sp.exists()
    note = {
        "eval_numbers": {"truncation_rates": trunc, "families_captured": sum(valid.values())},
        "eval_paths": [str(root / "store" / "capture")],
        "reproducibility_card": {
            "hf_data_prefix": C.HF_PREFIX,
            "capture_paths": [f"{C.HF_PREFIX}/analysis_tensors/capture/{f}.pt" for f in fams],
            "raw_completions": [f"{C.HF_PREFIX}/raw_completions/generation/{f}.json" for f in fams],
        },
        "wandb_url": "n/a (no training — frozen-base capture only)",
        "hf_hub_url": f"https://huggingface.co/datasets/{C.HF_DATA_REPO}/tree/main/{C.HF_PREFIX}",
        "worktree_path": str(C.PROJECT_ROOT),
        "final_commit_sha": C.reproducibility_metadata()["git_commit"],
        "gpu_hours_used": round(gpu_hours, 3),
        "gpu_hours_budgeted": 5.0,
        "plan_deviations": [],
        "upload": upload_info,
    }
    C.write_sentinel(
        "epm:smoke-result" if args.smoke else "epm:results",
        json.dumps(note),
        extra={"smoke": bool(args.smoke)},
    )
    C.phase("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
