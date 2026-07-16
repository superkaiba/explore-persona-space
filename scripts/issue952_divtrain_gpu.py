#!/usr/bin/env python
"""Issue #952 diverse-train-injection — GPU/pod leg (phases G/J/C/R/E + upload).

Same-issue inline follow-up (label ``diverse-train-injection``). Adds
divergence-domain (CCP-sensitive promptfoo) questions to the ridge-map TRAIN
pool and re-reads the china divergence bank to test whether the out-of-domain
floor lifts. The single changed variable is the TRAIN pool composition
(LMSYS-only -> LMSYS + injection-domain rows).

Phases (checkpoint-per-phase, resumable; each writes a done-sentinel):
  gen        (pod/GPU + API): own = Qwen-2.5-7B-Instruct vLLM; ext_plain =
                              claude-sonnet-4-5 via the project api_dispatch.
  judge      (pod/API)      : refusal-label BOTH arms (REFUSAL_RUBRIC, 3 draws,
                              max_tokens>=300, drop-never-coerce; DESCRIPTIVE).
  capture    (pod/GPU)      : teacher-forced fp16 slot capture of BOTH arms at
                              the parent 8-layer grid (72-slot registry) via the
                              parent _tf_capture_slots_arm / _save_arm_shards.
  refit_eval (pod/GPU|CPU)  : refit pool-only + augmented ridge maps at L20;
                              GATES first (reproduce the 3 committed cells), then
                              the in-domain manipulation check + augmented bank
                              cells + check_B refusal AUC.
  upload     (pod/CPU)      : ONE upload_folder of the consolidated out-base ->
                              issue952_position_divergence/followups/diverse_train_injection.

Content discipline (BINDING): promptfoo rows are CCP-sensitive bank items.
This script NEVER prints/logs/quotes prompt or answer TEXT — ids, counts,
scores, R2/d/AUC values and file refs only. Text is resolved + processed
programmatically and persisted to the raw-completion artifacts (their HF
destination), never echoed.

Reuse: the parent run_952 generation + judge + capture path VERBATIM, the
ridge_battery batched shared-SVD solver, and the committed decision-cell
machinery (issue952_divergence_transfer_cell / issue952_china_included_stats /
issue952_refusal_sanity pure helpers).
"""

from __future__ import annotations

import os

# vLLM v1 EngineCore silent-death under fork() (gotchas.md #628): spawn before any vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import asyncio
import json
import logging
import pathlib
import subprocess
import sys
import time

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT), str(_REPO_ROOT / "scripts")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# committed decision-cell machinery (pure helpers; module-level Path consts unused here)
from issue952_china_included_stats import _holm, _pairwise_drops, _role_split  # noqa: E402
from issue952_divergence_transfer_cell import (  # noqa: E402
    N_DRAWS,
    _bank_boot,
    _per_query_r2,
    _signflip_p,
    _stack_answer_targets,
)
from issue952_refusal_sanity import (  # noqa: E402
    _build_label_map,
    _loo_auc_with_perm,
)

from explore_persona_space.experiments.issue_952 import run_952 as R  # noqa: E402
from explore_persona_space.experiments.issue_952.ridge_battery import (  # noqa: E402
    run_ridge_cell,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("i952_divtrain")

REPO = R.HF_DATA_REPO
ISSUE_SLUG = R.ISSUE_SLUG
LABEL = "diverse-train-injection"
TAG = "diverse_train_injection"
HF_PREFIX = f"{ISSUE_SLUG}/followups/{TAG}"

# Parent + top-up tensor sources (pinned revisions from the committed decision cells).
PARENT_REV = "5b62649cefb34902fd630f21630164e8d1d99764"
CHINA_REV = "612c6c744e786ff65faae8e7ee97736239f873e5"
PARENT_PREFIX = f"{ISSUE_SLUG}/analysis_tensors"
CHINA_PREFIX = f"{ISSUE_SLUG}/followups/china_politics_topup/analysis_tensors"

READ_OUT_LAYER = 20  # #952 registered read-out layer
# run_952._dispatch_judge pins the judge response budget at 1024 tokens (>= the
# reason-then-score floor of ~300, llm-judging.md rule 23) — reported, not passed.
JUDGE_MAX_TOKENS = 1024
SPAN_MIN = 32  # span-32 matched-survivor inclusion (committed decision-cell universe)
REPRO_TOL = 1e-6  # d/drop reproduction tolerance (behavior_differs_subset gate parity)

# Committed gate values to reproduce (from the committed decision cells).
GATE_CHINA_ARM_MATCHED_D = 0.014371119237050044  # stats_china_included china_arm_matched.headline_d
GATE_CHINA_CROSS_DROP = -0.0010648308941010249  # stats_china_included cross.headline_drop
GATE_CHECK_B_AUC = 0.834375  # refusal_sanity check_B all AUC


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT).decode().strip()
        )
    except Exception:
        return "unknown"


def _repro_meta() -> dict:
    import platform

    import torch
    import transformers

    return {
        "git_commit": _git_sha(),
        "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "model": R.DEFAULT_MODEL,
        "judge_model": R.JUDGE_MODEL,
        "read_out_layer": READ_OUT_LAYER,
        "layer_grid": list(R.LAYER_GRID),
        "parent_rev": PARENT_REV,
        "china_rev": CHINA_REV,
    }


def _load_manifest(staging: pathlib.Path) -> dict:
    return json.loads((staging / "injection_manifest.json").read_text())


def _sentinel(path: str, note: str, **extra) -> None:
    R.write_sentinel(
        pathlib.Path(path),
        {"kind": "epm:progress", "version": 1, "note": note, **extra},
    )


# ── phase G: generation ─────────────────────────────────────────────────────────


def _own_gen(rows: list[dict], out_base: pathlib.Path) -> None:
    """own arm = Qwen vLLM (temp 1.0, top_p 0.95, max_tokens 1024, seed 42, no system)."""
    tokenizer = R._get_tokenizer()
    texts = {r["query_id"]: R.resolve_query_text(r) for r in rows}
    formatted = [
        tokenizer.apply_chat_template(
            [{"role": "user", "content": texts[r["query_id"]]}],
            tokenize=False,
            add_generation_prompt=True,
        )
        for r in rows
    ]
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=R.DEFAULT_MODEL, dtype="bfloat16", max_model_len=8192, seed=42, trust_remote_code=True
    )
    sp = SamplingParams(n=1, temperature=1.0, top_p=0.95, max_tokens=R.SONNET_MAX_TOKENS, seed=42)
    chunk = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
    n_chunks = (len(formatted) + chunk - 1) // chunk
    outs: list[tuple[str, int]] = []
    for c0 in range(0, len(formatted), chunk):
        logger.info("[vllm-chunk] own gen chunk %d/%d", c0 // chunk + 1, n_chunks)
        for o in llm.generate(formatted[c0 : c0 + chunk], sp, use_tqdm=False):
            outs.append((o.outputs[0].text, len(o.outputs[0].token_ids)))
    assert len(outs) == len(rows), (len(outs), len(rows))
    R._reap_vllm(llm)
    records = [
        {
            "query_id": rows[i]["query_id"],
            "subject_index": rows[i]["subject_index"],
            "split": rows[i]["split"],
            "question": texts[rows[i]["query_id"]],
            "answer_text": outs[i][0],
            "n_tokens": outs[i][1],
        }
        for i in range(len(rows))
    ]
    p = out_base / "raw_completions" / TAG / "own_seed42.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(records, indent=2, default=R._json_np))
    logger.info("[gen-own] %d Qwen answers -> %s", len(records), p)


async def _ext_gen(rows: list[dict], out_base: pathlib.Path) -> None:
    """ext_plain arm = claude-sonnet-4-5 via api_dispatch (temp 1.0, 1024 tok, no system)."""
    from explore_persona_space.llm.api_dispatch import DispatchItem, dispatch_calls

    texts = {r["query_id"]: R.resolve_query_text(r) for r in rows}
    items = [
        DispatchItem(
            item_id=r["query_id"],
            payload={"messages": [{"role": "user", "content": texts[r["query_id"]]}]},
        )
        for r in rows
    ]

    def _build(item) -> dict:
        return {
            "model": R.JUDGE_MODEL,
            "max_tokens": R.SONNET_MAX_TOKENS,
            "temperature": 1.0,
            "messages": item.payload["messages"],
        }

    results = await dispatch_calls(
        items,
        model=R.JUDGE_MODEL,
        build_request=_build,
        parse_response=lambda t: t,
        max_attempts=5,
        cache_dir=out_base / "judge_cache" / "ext_gen",
        checkpoint_dir=out_base / "judge_cache" / "ext_gen_ckpt",
    )
    records, n_fail = [], 0
    for r in rows:
        res = results.get(r["query_id"])
        text = None if res is None or getattr(res, "error", False) else res.result
        if not isinstance(text, str) or not text:
            n_fail += 1
            continue
        records.append(
            {
                "query_id": r["query_id"],
                "subject_index": r["subject_index"],
                "split": r["split"],
                "question": texts[r["query_id"]],
                "answer_text": text,
            }
        )
    p = out_base / "raw_completions" / TAG / "ext_plain_claude.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(records, indent=2, default=R._json_np))
    logger.info("[gen-ext] %d Claude answers, %d failed -> %s", len(records), n_fail, p)


def phase_gen(staging: pathlib.Path, out_base: pathlib.Path) -> None:
    R.log_phase("divtrain_gen")
    rows = _load_manifest(staging)["rows"]
    _own_gen(rows, out_base)
    asyncio.run(_ext_gen(rows, out_base))
    _sentinel("/workspace/logs/issue-952-divtrain-gen-done.json", "divtrain gen done", n=len(rows))
    R.log_phase("divtrain_gen_done")


# ── phase J: refusal judge (descriptive) ─────────────────────────────────────────


def _load_answers(out_base: pathlib.Path, arm_file: str) -> dict[str, str]:
    recs = json.loads((out_base / "raw_completions" / TAG / arm_file).read_text())
    return {r["query_id"]: r["answer_text"] for r in recs if r.get("answer_text")}


def _transport_content_split(results: dict, kind: str) -> tuple[int, int]:
    transport = content = 0
    for item_id, res in results.items():
        if not item_id.startswith(f"{kind}:"):
            continue
        if res is None or getattr(res, "error", False):
            transport += 1
        elif R.parse_judge_score(res.result) is None:
            content += 1
    return transport, content


def phase_judge(staging: pathlib.Path, out_base: pathlib.Path) -> None:
    R.log_phase("divtrain_judge")
    rows = _load_manifest(staging)["rows"]
    texts = {r["query_id"]: R.resolve_query_text(r) for r in rows}
    own = _load_answers(out_base, "own_seed42.json")
    ext = _load_answers(out_base, "ext_plain_claude.json")
    qids = [q for q in texts if q in own and q in ext]
    logger.info("[judge] %d/%d queries with both answers", len(qids), len(rows))

    ref_entries = [(f"{q}|own", R._refusal_user_text(texts[q], own[q])) for q in qids] + [
        (f"{q}|ext_plain", R._refusal_user_text(texts[q], ext[q])) for q in qids
    ]
    cache_root = out_base / "judge_cache"
    ckpt = out_base / "raw_completions" / TAG / "judge" / "_checkpoint"

    async def _run():
        return await R._dispatch_judge(
            R._judge_items("ref", ref_entries, R.N_REFUSAL_DRAWS),
            cache_root / "refusal_label",
            ckpt / "ref",
        )

    results = asyncio.run(_run())
    means, _dropped = R._aggregate_draws(results, "ref")
    transport, content = _transport_content_split(results, "ref")

    by_arm = {"own": {}, "ext_plain": {}}
    for q in qids:
        by_arm["own"][q] = means.get(f"{q}|own")
        by_arm["ext_plain"][q] = means.get(f"{q}|ext_plain")

    def _refrate(arm: str) -> dict:
        vals = [v for v in by_arm[arm].values() if v is not None]
        return {
            "n_scored": len(vals),
            "mean_graded": float(np.mean(vals)) if vals else None,
            "refusal_rate_thr50": float(np.mean([v >= 50 for v in vals])) if vals else None,
        }

    record = {
        "label": LABEL,
        "description": "refusal composition of the injected divergence-domain region (descriptive)",
        "judge_model": R.JUDGE_MODEL,
        "judge_max_tokens": JUDGE_MAX_TOKENS,
        "n_refusal_draws": R.N_REFUSAL_DRAWS,
        "n_queries": len(qids),
        "drops": {"refusal_content_drop_draws": content, "refusal_transport_loss_draws": transport},
        "per_arm": {arm: _refrate(arm) for arm in ("own", "ext_plain")},
        "per_query_refusal": {
            q: {"own": by_arm["own"][q], "ext_plain": by_arm["ext_plain"][q]} for q in qids
        },
        "repro": _repro_meta(),
    }
    p = out_base / "eval_results" / "divtrain_refusal_labels.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(record, indent=2, default=R._json_np))
    logger.info("[judge] wrote %s | drops content=%d transport=%d", p, content, transport)
    _sentinel(
        "/workspace/logs/issue-952-divtrain-judge-done.json",
        "divtrain judge done",
        content_drops=content,
        transport_losses=transport,
    )
    R.log_phase("divtrain_judge_done")


# ── phase C: teacher-forced capture ──────────────────────────────────────────────


def phase_capture(staging: pathlib.Path, out_base: pathlib.Path) -> None:
    """Capture BOTH arms at the parent 8-layer grid, 72-slot registry (drop-in mergeable)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    R.log_phase("divtrain_capture")
    rows = _load_manifest(staging)["rows"]
    own = _load_answers(out_base, "own_seed42.json")
    ext = _load_answers(out_base, "ext_plain_claude.json")
    texts = {r["query_id"]: R.resolve_query_text(r) for r in rows}
    ids = [q for r in rows if (q := r["query_id"]) in own and q in ext]
    prompts_by_id = {q: texts[q] for q in ids}
    answers = {"own": {q: own[q] for q in ids}, "ext_plain": {q: ext[q] for q in ids}}
    logger.info(
        "[capture] %d queries x %d arms x %d layers", len(ids), len(R.BANK_ARMS), len(R.LAYER_GRID)
    )

    tokenizer = AutoTokenizer.from_pretrained(R.DEFAULT_MODEL, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        R.DEFAULT_MODEL,
        torch_dtype=torch.bfloat16,
        device_map="cuda" if torch.cuda.is_available() else "cpu",
        trust_remote_code=True,
    )
    model.eval()
    batch_size = int(os.environ.get("EPM_TF_BATCH_SIZE", "8"))
    for arm in R.BANK_ARMS:
        slots, spans, _surp = R._tf_capture_slots_arm(
            model,
            tokenizer,
            ids,
            prompts_by_id,
            answers[arm],
            f"{TAG}_{arm}",
            own_raw_lens=None,
            batch_size=batch_size,
        )
        R._save_arm_shards(out_base, f"{TAG}_{arm}", slots, ids, spans)
        del slots
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    at = out_base / "analysis_tensors"
    at.mkdir(parents=True, exist_ok=True)
    prov = {r["query_id"]: {"subject_index": r["subject_index"], "split": r["split"]} for r in rows}
    (at / f"provenance_{TAG}.json").write_text(
        json.dumps(
            {
                "ids": ids,
                "provenance": prov,
                "arms": list(R.BANK_ARMS),
                "layers": list(R.LAYER_GRID),
                "slot_names": list(R.SLOT_NAMES),
                "repro": _repro_meta(),
            },
            indent=2,
            default=R._json_np,
        )
    )
    logger.info("[capture] done — %d queries, tag %s_{own,ext_plain}", len(ids), TAG)
    _sentinel(
        "/workspace/logs/issue-952-divtrain-capture-done.json", "divtrain capture done", n=len(ids)
    )
    R.log_phase("divtrain_capture_done")


# ── phase R+E: refit + gates + manipulation-check + bank cells + check_B ──────────


def _stage(prefix: str, rev: str, fname: str, dest: pathlib.Path) -> pathlib.Path:
    """Fetch one HF tensor to dest via hf_hub_download (bounded retry)."""
    from huggingface_hub import hf_hub_download

    dest.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(4):
        try:
            src = pathlib.Path(
                hf_hub_download(REPO, f"{prefix}/{fname}", repo_type="dataset", revision=rev)
            )
            target = dest / fname
            if not target.exists():
                import shutil

                shutil.copy(src, target)
            return target
        except Exception as e:  # transient HF 5xx / 429
            last = e
            logger.warning("[stage] %s failed (%d): %s", fname, attempt + 1, e)
            time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"stage failed after retries: {prefix}/{fname}") from last


def _load_slots(path: pathlib.Path) -> tuple[np.ndarray, list]:
    import torch

    d = torch.load(str(path), map_location="cpu", weights_only=False)
    assert d["slot_names"] == list(R.SLOT_NAMES), f"slot registry drift in {path}"
    return d["slots"].numpy(), list(d["ids"])


def _span_arr(spans: dict, ids: list, arm: str) -> np.ndarray:
    return np.asarray([spans[arm][str(c)].get("span", 0) for c in ids], dtype=np.int64)


def _map_fit_read(
    own_train: np.ndarray,
    plain_train: np.ndarray,
    lam_by_slot: dict[str, int] | None,
    val_splits: dict | None,
    eval_arm_matched: dict,
    cross_evals: dict,
) -> dict:
    """One map variant. Returns per-query R2 for arm-matched + cross reads, and the
    selected per-slot lam table. If lam_by_slot is None, re-select via validation-argmax
    (registered rule); else freeze at the supplied per-slot lambda indices.

    own_train/plain_train: (n_tr, 72, H) parent+injection stacked own/ext_plain slots.
    eval_arm_matched/cross_evals: name -> (x_c_last (n,H), target_slots_by_arm dict, rows).
    """
    slots_by_arm = {"own": own_train, "ext_plain": plain_train}
    c_last_tr = own_train[:, R.SLOT_IDX["c_last"], :].astype(np.float64)

    def _fit_eval(fit_arms, evals, want_val):
        Ytr, gnames = _stack_answer_targets(slots_by_arm, np.arange(own_train.shape[0]), fit_arms)
        splits = {}
        for name, (xb, tgt_bank, rows) in evals.items():
            tgt, _g = _stack_answer_targets(tgt_bank, np.asarray(rows), fit_arms)
            splits[name] = (xb, tgt)
        if want_val and val_splits is not None:
            vx, vtgt_bank, vrows = val_splits["val"]
            vtgt, _g = _stack_answer_targets(vtgt_bank, np.asarray(vrows), fit_arms)
            splits["val"] = (vx, vtgt)
        res = run_ridge_cell(
            c_last_tr,
            Ytr,
            splits,
            group_names=gnames,
            device="cpu",
            allow_train_nan_imputation=True,
        )
        return res, gnames

    # arm-matched fit (own + ext_plain), with λ selection on val if requested.
    res_am, gnames = _fit_eval(("own", "ext_plain"), eval_arm_matched, want_val=lam_by_slot is None)
    if lam_by_slot is None:
        # _lam_star_by_slot wants (slot, arm) tuples; gnames are "slot|arm" strings.
        gtuples = [tuple(g.split("|")) for g in gnames]
        _lam_idx, lam_by_slot = R._lam_star_by_slot(gtuples, res_am.pooled["val"])
    lam_idx = np.asarray([lam_by_slot[g.split("|")[0]] for g in gnames], dtype=np.int64)
    am = {}
    for name in eval_arm_matched:
        ssr = np.take_along_axis(res_am.ss_res[name], lam_idx[None, :, None], axis=2)[:, :, 0]
        sst = res_am.ss_tot[name].astype(np.float64)
        am[name] = {
            arm: _per_query_r2(ssr.astype(np.float64), sst, gnames, arm)
            for arm in ("own", "ext_plain")
        }

    # cross reads (single-arm fit, scored against the other arm).
    cross_out = {}
    for cname, (fit_arm, evals) in cross_evals.items():
        res_c, gn_c = _fit_eval((fit_arm,), evals, want_val=False)
        li = np.asarray([lam_by_slot[g.split("|")[0]] for g in gn_c], dtype=np.int64)
        cross_out[cname] = {}
        for name in evals:
            ssr = np.take_along_axis(res_c.ss_res[name], li[None, :, None], axis=2)[:, :, 0]
            sst = res_c.ss_tot[name].astype(np.float64)
            cross_out[cname][name] = _per_query_r2(ssr.astype(np.float64), sst, gn_c, fit_arm)
    return {"arm_matched": am, "cross": cross_out, "lam_by_slot": lam_by_slot}


def _own_map_preds(
    own_train: np.ndarray, lam_by_slot: dict[str, int], bank_c_last: np.ndarray
) -> np.ndarray:
    """Own-map predicted answer-summary yhat (n_bank, H): mean over the 42 POSITION_SLOTS
    of the frozen-λ own map applied to bank c_last (mirrors refusal_sanity._own_map_predict)."""
    Xtr = own_train[:, R.SLOT_IDX["c_last"], :].astype(np.float64)
    xmu, xsd = Xtr.mean(0), Xtr.std(0) + 1e-9
    U, s, Vh = np.linalg.svd((Xtr - xmu) / xsd, full_matrices=False)
    s2 = s**2
    A = ((bank_c_last - xmu) / xsd) @ Vh.T
    H = own_train.shape[2]
    preds = np.full((bank_c_last.shape[0], len(R.POSITION_SLOTS), H), np.nan)
    for si, slot in enumerate(R.POSITION_SLOTS):
        Y = own_train[:, R.SLOT_IDX[slot], :].astype(np.float64)
        bad = ~np.isfinite(Y).all(axis=1)
        if bad.any():
            Y = Y.copy()
            Y[bad] = Y[~bad].mean(axis=0)
        ymu = Y.mean(0)
        B = U.T @ (Y - ymu)
        filt = s / (s2 + R.DEFAULT_LAMBDAS_LIST[int(lam_by_slot[slot])])
        preds[:, si, :] = (A * filt[None, :]) @ B + ymu
    return preds.mean(axis=1)


def _pair_d_cell(div_r2, ctl_r2, pairs, div_i2r, ctl_i2r) -> tuple[list, np.ndarray]:
    rows = _pairwise_drops(div_r2, ctl_r2, pairs, div_i2r, ctl_i2r)
    for r in rows:
        if "drop_own" in r and "drop_ext_plain" in r:
            r["d"] = r["drop_ext_plain"] - r["drop_own"]
    return rows, np.asarray([r["d"] for r in rows if "d" in r], dtype=np.float64)


def _fabricate_synth_tensors(
    staging: pathlib.Path, out_base: pathlib.Path, tdir: pathlib.Path
) -> None:
    """VM wiring smoke: tiny random slot shards keyed to the REAL committed ids.

    Exercises the full refit_eval fit + gate + cell wiring on CPU with no GPU/HF.
    Values are random => the reproduction gates WARN (not pass) under --smoke.
    """
    import torch

    tdir.mkdir(parents=True, exist_ok=True)
    (out_base / "analysis_tensors").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(952)
    hdim = R.EXPECTED_HIDDEN
    n_slots = len(R.SLOT_NAMES)

    def _shard(path: pathlib.Path, ids: list) -> None:
        arr = rng.standard_normal((len(ids), n_slots, hdim)).astype(np.float16)
        torch.save(
            {
                "slots": torch.from_numpy(arr),
                "ids": list(ids),
                "slot_names": list(R.SLOT_NAMES),
                "layer": READ_OUT_LAYER,
            },
            str(path),
        )

    def _spans(ids: list) -> dict:
        return {str(c): {"span": 40, "truncated": False} for c in ids}

    split = json.loads((_REPO_ROOT / "eval_results/issue_952/split_seed952.json").read_text())
    pool_ids = list(split["train"][:40]) + list(split["val"][:20])
    _shard(tdir / "slots_own_L20.pt", pool_ids)
    _shard(tdir / "slots_ext_plain_L20.pt", pool_ids)
    for a in R.ARMS:
        (tdir / f"spans_{a}.json").write_text(json.dumps(_spans(pool_ids)))

    groups = [f"{s}|{a}" for s in R.POSITION_SLOTS for a in R.BANK_ARMS]
    np.savez(
        str(tdir / "per_context_stats.npz"),
        A_group_names=np.asarray(groups),
        A_lam_idx=np.asarray([6] * len(groups)),
    )

    comm_verif = json.loads(
        (_REPO_ROOT / "eval_results/issue_952/divergence_bank_verification.json").read_text()
    )
    comm_pairs = [
        p
        for p in comm_verif["pairs"]
        if p["pair_id"] in set(comm_verif["kept_pairs"]) and isinstance(p.get("divergent"), dict)
    ][:4]
    comm_ids = [
        q for p in comm_pairs for q in (p["divergent"]["query_id"], p["control"]["query_id"])
    ]
    for a in R.BANK_ARMS:
        _shard(tdir / f"slots_bank_{a}_L20.pt", comm_ids)

    china_verif = json.loads(
        (
            _REPO_ROOT / "eval_results/issue_952/china-politics-topup/summaries/"
            "china_topup_verification.json"
        ).read_text()
    )
    china_ids = [
        q for pid in china_verif["final_china_kept_pairs"][:4] for q in (f"{pid}_div", f"{pid}_ctl")
    ]
    for a in R.BANK_ARMS:
        _shard(tdir / f"slots_bank_china_politics_topup_{a}_L20.pt", china_ids)

    inj_ids = [r["query_id"] for r in _load_manifest(staging)["rows"]]
    for a in R.BANK_ARMS:
        _shard(out_base / "analysis_tensors" / f"slots_{TAG}_{a}_L20.pt", inj_ids)
        (out_base / "analysis_tensors" / f"spans_{TAG}_{a}.json").write_text(
            json.dumps(_spans(inj_ids))
        )
    logger.info(
        "[synth] fabricated smoke tensors: pool=%d comm=%d china=%d inj=%d",
        len(pool_ids),
        len(comm_ids),
        len(china_ids),
        len(inj_ids),
    )


def phase_refit_eval(  # noqa: C901 — the R+E driver: gates + 2 map variants + cells + check_B
    staging: pathlib.Path, out_base: pathlib.Path, smoke: bool, synth: bool = False
) -> None:
    R.log_phase("divtrain_refit_eval")
    import torch  # noqa: F401  (ensures torch present for _load_slots)

    manifest = _load_manifest(staging)
    # Stage parent/top-up tensors OUTSIDE out_base so the upload_folder(out_base)
    # never sees the HF cache and the file-count guard never counts it.
    tdir = out_base.parent / "divtrain_stage"
    if synth:
        # VM wiring smoke: fabricate tiny slot tensors keyed to the real committed
        # ids (no GPU, no HF). Gates are downgraded to WARN under --smoke since
        # synthetic activations cannot reproduce the committed values.
        _fabricate_synth_tensors(staging, out_base, tdir)
    else:
        # parent LMSYS pool + spans + frozen λ + committed bank tensors
        for f in (
            "slots_own_L20.pt",
            "slots_ext_plain_L20.pt",
            "per_context_stats.npz",
            "slots_bank_own_L20.pt",
            "slots_bank_ext_plain_L20.pt",
            *[f"spans_{a}.json" for a in R.ARMS],
        ):
            _stage(PARENT_PREFIX, PARENT_REV, f, tdir)
        for a in R.BANK_ARMS:
            _stage(CHINA_PREFIX, CHINA_REV, f"slots_bank_china_politics_topup_{a}_L20.pt", tdir)

    own_pool, pool_ids = _load_slots(tdir / "slots_own_L20.pt")
    plain_pool, _ = _load_slots(tdir / "slots_ext_plain_L20.pt")
    split = json.loads((_REPO_ROOT / "eval_results/issue_952/split_seed952.json").read_text())
    spans = {
        a: {str(k): v for k, v in json.loads((tdir / f"spans_{a}.json").read_text()).items()}
        for a in R.ARMS
    }
    span_ok = np.all(np.stack([_span_arr(spans, pool_ids, a) >= SPAN_MIN for a in R.ARMS]), axis=0)
    pos_of = {c: i for i, c in enumerate(pool_ids)}
    tr_pos = np.asarray([pos_of[c] for c in split["train"] if c in pos_of])
    tr_a = tr_pos[span_ok[tr_pos]]
    val_pos = np.asarray([pos_of[c] for c in split["val"] if c in pos_of])
    val_a = val_pos[span_ok[val_pos]]
    logger.info("[refit] tr_a=%d val_a=%d", len(tr_a), len(val_a))

    npz = dict(np.load(str(tdir / "per_context_stats.npz"), allow_pickle=False))
    group2lam = dict(zip(npz["A_group_names"].tolist(), npz["A_lam_idx"].tolist(), strict=True))
    frozen_by_slot = {s: int(group2lam[f"{s}|own"]) for s in R.POSITION_SLOTS}

    # injection train slots (own + ext_plain), span-32 filtered on the 2 captured arms.
    inj_own, inj_ids = _load_slots(out_base / "analysis_tensors" / f"slots_{TAG}_own_L20.pt")
    inj_plain, inj_ids2 = _load_slots(
        out_base / "analysis_tensors" / f"slots_{TAG}_ext_plain_L20.pt"
    )
    assert inj_ids == inj_ids2, "own/ext_plain injection id drift"
    inj_spans = R._load_spans(out_base, f"{TAG}_own")
    inj_spans_ep = R._load_spans(out_base, f"{TAG}_ext_plain")
    split_of = {r["query_id"]: r["split"] for r in manifest["rows"]}
    inj_train_rows = np.asarray(
        [
            i
            for i, q in enumerate(inj_ids)
            if split_of.get(q) == "train"
            and inj_spans.get(str(q), {}).get("span", 0) >= SPAN_MIN
            and inj_spans_ep.get(str(q), {}).get("span", 0) >= SPAN_MIN
        ],
        dtype=np.int64,
    )
    inj_check_rows = np.asarray(
        [i for i, q in enumerate(inj_ids) if split_of.get(q) == "indomain_check"], dtype=np.int64
    )
    logger.info(
        "[refit] injection train=%d indomain_check=%d (of %d captured)",
        len(inj_train_rows),
        len(inj_check_rows),
        len(inj_ids),
    )

    # ── train stacks: pool-only (tr_a) and augmented (tr_a + injection train) ────────
    own_pool_tr = own_pool[tr_a]
    plain_pool_tr = plain_pool[tr_a]
    own_aug = np.concatenate([own_pool_tr, inj_own[inj_train_rows]], axis=0)
    plain_aug = np.concatenate([plain_pool_tr, inj_plain[inj_train_rows]], axis=0)

    # ── bank tensors + role splits ──────────────────────────────────────────────────
    comm_bank = {a: _load_slots(tdir / f"slots_bank_{a}_L20.pt") for a in R.BANK_ARMS}
    china_bank = {
        a: _load_slots(tdir / f"slots_bank_china_politics_topup_{a}_L20.pt") for a in R.BANK_ARMS
    }
    comm_ids = comm_bank["own"][1]
    china_ids = china_bank["own"][1]
    comm_dr, comm_cr, comm_di2r, comm_ci2r = _role_split(comm_ids)
    ch_dr, ch_cr, ch_di2r, ch_ci2r = _role_split(china_ids)

    comm_verif = json.loads(
        (_REPO_ROOT / "eval_results/issue_952/divergence_bank_verification.json").read_text()
    )
    comm_kept = set(comm_verif["kept_pairs"])
    comm_pairs = [
        (p["pair_id"], p["category"], p["divergent"]["query_id"], p["control"]["query_id"])
        for p in comm_verif["pairs"]
        if p["pair_id"] in comm_kept
        and isinstance(p.get("divergent"), dict)
        and isinstance(p.get("control"), dict)
    ]
    china_verif = json.loads(
        (
            _REPO_ROOT / "eval_results/issue_952/china-politics-topup/summaries/"
            "china_topup_verification.json"
        ).read_text()
    )
    china_pairs = [
        (pid, "china_politics", f"{pid}_div", f"{pid}_ctl")
        for pid in china_verif["final_china_kept_pairs"]
    ]

    def _x_c_last(bank_arm_slots, rows):
        return bank_arm_slots[rows][:, R.SLOT_IDX["c_last"], :].astype(np.float64)

    def _arm_matched_evals():
        return {
            "comm_div": (
                _x_c_last(comm_bank["own"][0], comm_dr),
                {a: comm_bank[a][0] for a in R.BANK_ARMS},
                comm_dr,
            ),
            "comm_ctl": (
                _x_c_last(comm_bank["own"][0], comm_cr),
                {a: comm_bank[a][0] for a in R.BANK_ARMS},
                comm_cr,
            ),
            "china_div": (
                _x_c_last(china_bank["own"][0], ch_dr),
                {a: china_bank[a][0] for a in R.BANK_ARMS},
                ch_dr,
            ),
            "china_ctl": (
                _x_c_last(china_bank["own"][0], ch_cr),
                {a: china_bank[a][0] for a in R.BANK_ARMS},
                ch_cr,
            ),
            "indomain_check": (
                inj_own[inj_check_rows][:, R.SLOT_IDX["c_last"], :].astype(np.float64),
                {"own": inj_own, "ext_plain": inj_plain},
                inj_check_rows,
            ),
        }

    def _cross_evals():
        return {
            "china_cross_own_map_x_plain": (
                "own",
                {
                    "china_div": (
                        _x_c_last(china_bank["own"][0], ch_dr),
                        {"own": china_bank["ext_plain"][0]},
                        ch_dr,
                    ),
                    "china_ctl": (
                        _x_c_last(china_bank["own"][0], ch_cr),
                        {"own": china_bank["ext_plain"][0]},
                        ch_cr,
                    ),
                },
            )
        }

    def _val_split():
        return {
            "val": (
                own_pool[val_a][:, R.SLOT_IDX["c_last"], :].astype(np.float64),
                {"own": own_pool[val_a], "ext_plain": plain_pool[val_a]},
                np.arange(len(val_a)),
            )
        }

    # ── fit both variants ────────────────────────────────────────────────────────────
    pool_only = _map_fit_read(
        own_pool_tr, plain_pool_tr, frozen_by_slot, None, _arm_matched_evals(), _cross_evals()
    )
    augmented = _map_fit_read(
        own_aug, plain_aug, None, _val_split(), _arm_matched_evals(), _cross_evals()
    )

    def _cells(variant: dict) -> dict:
        am = variant["arm_matched"]
        china_rows, d_china = _pair_d_cell(
            am["china_div"], am["china_ctl"], china_pairs, ch_di2r, ch_ci2r
        )
        comm_rows, d_comm = _pair_d_cell(
            am["comm_div"], am["comm_ctl"], comm_pairs, comm_di2r, comm_ci2r
        )
        cross = variant["cross"]["china_cross_own_map_x_plain"]
        cross_rows = _pairwise_drops(
            {"own": cross["china_div"]}, {"own": cross["china_ctl"]}, china_pairs, ch_di2r, ch_ci2r
        )
        for r in cross_rows:
            r["drop"] = r.get("drop_own")
        d_cross = np.asarray([r["drop"] for r in cross_rows if r.get("drop") is not None])
        # in-domain manipulation check: per-arm per-context R2 on held-out injection rows.
        ic = am["indomain_check"]
        pooled72 = china_rows + comm_rows
        d72 = np.asarray([r["d"] for r in pooled72 if "d" in r])
        cats = {}
        for cat in ("model_identity", "style_format", "china_politics"):
            rr = [r for r in pooled72 if r["category"] == cat and "d" in r]
            if len(rr) >= 2:
                dd = np.asarray([r["d"] for r in rr])
                cats[cat] = {
                    "n": len(rr),
                    "mean_d": float(dd.mean()),
                    "sign_flip_p": _signflip_p(dd, N_DRAWS)["p_one_sided"],
                }
        holm = _holm({c: cats[c]["sign_flip_p"] for c in cats}) if cats else {}
        for c in cats:
            cats[c]["p_holm"] = holm.get(c)
        return {
            "china_arm_matched": {
                "n": len(d_china),
                "mean_d": float(d_china.mean()) if len(d_china) else None,
                "boot": _bank_boot(d_china, N_DRAWS) if len(d_china) else None,
                "sign_flip": _signflip_p(d_china, N_DRAWS) if len(d_china) else None,
                "mean_drop_own": float(np.mean([r["drop_own"] for r in china_rows]))
                if china_rows
                else None,
                "mean_drop_ext_plain": float(np.mean([r["drop_ext_plain"] for r in china_rows]))
                if china_rows
                else None,
            },
            "china_cross_own_map_x_plain": {
                "n": len(d_cross),
                "mean_drop": float(d_cross.mean()) if len(d_cross) else None,
                "sign_flip": _signflip_p(d_cross, N_DRAWS) if len(d_cross) else None,
            },
            "pooled_72": {
                "n": len(d72),
                "mean_d": float(d72.mean()) if len(d72) else None,
                "sign_flip": _signflip_p(d72, N_DRAWS) if len(d72) else None,
            },
            "committed_41": {
                "n": len(d_comm),
                "mean_d": float(d_comm.mean()) if len(d_comm) else None,
            },
            "per_category_holm": cats,
            "indomain_check": {
                "n": len(inj_check_rows),
                "mean_r2_own": float(np.nanmean(ic["own"])) if len(inj_check_rows) else None,
                "mean_r2_ext_plain": float(np.nanmean(ic["ext_plain"]))
                if len(inj_check_rows)
                else None,
            },
            "_china_rows": china_rows,
            "_comm_rows": comm_rows,
            "_cross_rows": cross_rows,
            "_ic_per_context": {"own": ic["own"].tolist(), "ext_plain": ic["ext_plain"].tolist()},
            "lam_by_slot": variant["lam_by_slot"],
        }

    pool_cells = _cells(pool_only)
    aug_cells = _cells(augmented)

    # ── check_B refusal AUC (own map predicts refusal on the china bank) ─────────────
    def _check_b(own_train: np.ndarray, lam_by_slot: dict) -> dict:
        labmap = _build_label_map(_REPO_ROOT)
        keep = [q for q in china_ids if q in labmap and labmap[q]["refusal_qwen"] is not None]
        keep_idx = np.asarray([china_ids.index(q) for q in keep])
        pslot = np.asarray([R.SLOT_IDX[s] for s in R.POSITION_SLOTS])

        def _summary(slots):
            sub = slots[keep_idx][:, pslot, :].astype(np.float64)
            nfin = np.isfinite(sub).all(axis=2).sum(axis=1)
            return sub.mean(axis=1), nfin

        y_own, nfin_o = _summary(china_bank["own"][0])
        y_claude, nfin_c = _summary(china_bank["ext_plain"][0])
        c_last_bank = china_bank["own"][0][keep_idx][:, R.SLOT_IDX["c_last"], :].astype(np.float64)
        rq = np.asarray([labmap[q]["refusal_qwen"] for q in keep], dtype=np.float64)
        good = (
            (nfin_o == len(R.POSITION_SLOTS))
            & (nfin_c == len(R.POSITION_SLOTS))
            & np.isfinite(y_own).all(1)
            & np.isfinite(y_claude).all(1)
        )
        y_own, c_last_bank, rq = y_own[good], c_last_bank[good], rq[good]
        r_qwen = (rq >= 50.0).astype(int)
        yhat = _own_map_preds(own_train, lam_by_slot, c_last_bank)
        res = _loo_auc_with_perm(y_own, yhat, r_qwen)
        return {"n_usable": int(good.sum()), "auc": res["auc"], "detail": res}

    pool_check_b = _check_b(own_pool_tr, frozen_by_slot)
    aug_check_b = _check_b(own_aug, augmented["lam_by_slot"])

    # ── GATES (reproduce the 3 committed pool-only cells) ────────────────────────────
    gates = {
        "china_arm_matched_d": {
            "recomputed": pool_cells["china_arm_matched"]["mean_d"],
            "committed": GATE_CHINA_ARM_MATCHED_D,
            "pass": bool(
                abs((pool_cells["china_arm_matched"]["mean_d"] or 0) - GATE_CHINA_ARM_MATCHED_D)
                < REPRO_TOL
            ),
        },
        "china_cross_drop": {
            "recomputed": pool_cells["china_cross_own_map_x_plain"]["mean_drop"],
            "committed": GATE_CHINA_CROSS_DROP,
            "pass": bool(
                abs(
                    (pool_cells["china_cross_own_map_x_plain"]["mean_drop"] or 0)
                    - GATE_CHINA_CROSS_DROP
                )
                < REPRO_TOL
            ),
        },
        "check_B_auc": {
            "recomputed": pool_check_b["auc"],
            "committed": GATE_CHECK_B_AUC,
            "pass": bool(
                pool_check_b["auc"] is not None
                and abs(pool_check_b["auc"] - GATE_CHECK_B_AUC) < REPRO_TOL
            ),
        },
    }
    all_pass = all(g["pass"] for g in gates.values())
    logger.info(
        "[gates] arm_matched=%s cross=%s check_B=%s ALL=%s",
        gates["china_arm_matched_d"]["pass"],
        gates["china_cross_drop"]["pass"],
        gates["check_B_auc"]["pass"],
        all_pass,
    )
    if not all_pass and not smoke:
        raise RuntimeError(f"GATE FAIL — refusing to proceed: {json.dumps(gates)}")

    out = {
        "label": LABEL,
        "layer": READ_OUT_LAYER,
        "description": (
            "diverse-train-injection: pool-only vs augmented (LMSYS + injection-domain "
            "train) ridge maps at L20; china-bank re-read + in-domain manipulation "
            "check + check_B refusal AUC."
        ),
        "n_draws": N_DRAWS,
        "span_min": SPAN_MIN,
        "train_universe": {
            "parent_tr_a": len(tr_a),
            "injection_train": len(inj_train_rows),
            "augmented_total": int(own_aug.shape[0]),
            "val_a": len(val_a),
        },
        "reproduction_gates": gates,
        "pool_only": {k: v for k, v in pool_cells.items() if not k.startswith("_")},
        "augmented": {k: v for k, v in aug_cells.items() if not k.startswith("_")},
        "check_B": {"pool_only": pool_check_b, "augmented": aug_check_b},
        "per_pair_rows": {
            "pool_only": {"china": pool_cells["_china_rows"], "cross": pool_cells["_cross_rows"]},
            "augmented": {"china": aug_cells["_china_rows"], "cross": aug_cells["_cross_rows"]},
        },
        "indomain_check_per_context": {
            "pool_only": pool_cells["_ic_per_context"],
            "augmented": aug_cells["_ic_per_context"],
            "ids": [inj_ids[i] for i in inj_check_rows],
        },
        "repro": _repro_meta(),
    }
    p = out_base / "eval_results" / "divtrain_refit_eval.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=2, default=R._json_np))
    logger.info("[refit_eval] wrote %s", p)
    _sentinel(
        "/workspace/logs/issue-952-divtrain-refit-done.json",
        "divtrain refit_eval done",
        gates_pass=all_pass,
    )
    R.log_phase("divtrain_refit_eval_done")


# ── phase upload ──────────────────────────────────────────────────────────────────


def phase_upload(out_base: pathlib.Path) -> None:
    R.log_phase("divtrain_upload")
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import (
        assert_hub_dir_filecounts,
        list_hf_files_under_path,
    )

    # The HF staging cache lives OUTSIDE out_base (divtrain_stage), so out_base
    # holds only the produced artifacts — no ignore patterns needed.
    api = HfApi()
    # Pre-count guard (#1190): fail loud before any network I/O if a target dir
    # would exceed the 10k-files/commit cap. Outside any transient-retry wrapper.
    assert_hub_dir_filecounts(out_base, HF_PREFIX)
    api.upload_folder(
        folder_path=str(out_base),
        repo_id=REPO,
        repo_type="dataset",
        path_in_repo=HF_PREFIX,
        commit_message=f"issue #952 {LABEL} — GPU leg artifacts",
    )
    files = list_hf_files_under_path(api, REPO, HF_PREFIX, repo_type="dataset")
    if not files:
        raise RuntimeError(f"post-upload verify: no files under {REPO}/{HF_PREFIX}")
    logger.info("[upload] %d files under %s/%s", len(files), REPO, HF_PREFIX)
    (out_base / "eval_results" / "divtrain_upload_manifest.json").write_text(
        json.dumps(
            {
                "hf_repo": REPO,
                "hf_prefix": HF_PREFIX,
                "n_files": len(files),
                "files": sorted(files),
                "repro": _repro_meta(),
            },
            indent=2,
        )
    )
    _sentinel(
        "/workspace/logs/issue-952-divtrain-upload-done.json",
        "divtrain upload done",
        n_files=len(files),
    )
    R.log_phase("divtrain_upload_done")


def main() -> None:
    ap = argparse.ArgumentParser(description="issue #952 diverse-train-injection GPU leg")
    ap.add_argument(
        "--phase", required=True, choices=["gen", "judge", "capture", "refit_eval", "upload", "all"]
    )
    ap.add_argument(
        "--staging-dir",
        default="eval_results/issue_952/diverse-train-injection",
        help="dir holding injection_manifest.json",
    )
    ap.add_argument(
        "--out-base",
        default="/workspace/divtrain_out",
        help="consolidated output base (raw_completions/, analysis_tensors/, eval_results/)",
    )
    ap.add_argument(
        "--smoke", action="store_true", help="tiny slice; gate mismatch downgraded to WARN"
    )
    ap.add_argument(
        "--synth",
        action="store_true",
        help="refit_eval only: fabricate synthetic slot tensors (VM wiring smoke, no GPU/HF)",
    )
    args = ap.parse_args()
    staging = pathlib.Path(args.staging_dir).resolve()
    out_base = pathlib.Path(args.out_base).resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[divtrain] phase=%s staging=%s out_base=%s sha=%s",
        args.phase,
        staging,
        out_base,
        _git_sha(),
    )

    phases = (
        ["gen", "judge", "capture", "refit_eval", "upload"] if args.phase == "all" else [args.phase]
    )
    for ph in phases:
        if ph == "gen":
            phase_gen(staging, out_base)
        elif ph == "judge":
            phase_judge(staging, out_base)
        elif ph == "capture":
            phase_capture(staging, out_base)
        elif ph == "refit_eval":
            phase_refit_eval(staging, out_base, args.smoke, synth=args.synth)
        elif ph == "upload":
            phase_upload(out_base)


if __name__ == "__main__":
    main()
