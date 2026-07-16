#!/usr/bin/env python
"""Issue #952 china-politics top-up — GPU leg [B] (inline follow-up, scope epm:followup-scope v3).

Reuses the parent ``run_952`` bank measurement path VERBATIM — the Qwen generation
recipe (temperature 1.0, top_p 0.95, max_tokens 1024, seed 42, no system prompt),
the divergence + refusal judge dispatch and rubrics (5 divergence / 3 refusal draws,
drop-never-coerce, ``claude-sonnet-4-5-20250929``, response max_tokens 1024), and the
teacher-forced fp16 slot capture / reduction (``_tf_capture_slots_arm`` + the 72-slot
registry). The single changed variable is china-politics category coverage.

Phases:
  gen     (pod/GPU) : Qwen answers for the 24 new pairs (48 queries), one vLLM batch.
  judge   (pod/API) : divergence + refusal judging on the 24 new pairs; the COMMITTED
                      gates (keep >= 47, pair margin >= -23, verified against the parent
                      verification.json); combine with the parent's 18 kept china pairs.
  capture (pod/GPU) : teacher-forced fp16 bank slots for 42 china pairs (24 new + the 18
                      parent-uncaptured pairs coverage_report marks missing), arms
                      {own = Qwen, ext_plain = Claude}, the parent's full 8-layer grid.
  upload  (pod/CPU) : ONE HfApi.upload_folder of the consolidated out-base -> the issue
                      data repo followups prefix, verified via list_repo_files.

Content discipline (BINDING): china-politics rows are sensitive-category bank items.
This script NEVER prints/logs china query or answer TEXT — ids, pair ids, roles,
counts, scores and file refs only. Text is resolved + processed programmatically and
persisted to the raw-completion artifacts (the intended HF destination), never echoed.
"""

from __future__ import annotations

import os

# vLLM v1 EngineCore silent-death under fork() when the parent touched CUDA-adjacent
# code before LLM() (gotchas.md #628): pin spawn before any vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

import argparse
import asyncio
import json
import logging
import pathlib
import subprocess
import time

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from explore_persona_space.experiments.issue_952 import run_952 as R  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("i952_china_topup")

REPO = "superkaiba1/explore-persona-space-data"
ISSUE_SLUG = "issue952_position_divergence"
TOPUP_TAG = "china_politics_topup"
HF_PREFIX = f"{ISSUE_SLUG}/followups/{TOPUP_TAG}"
# Committed gates from the parent divergence_bank_verification.json (calibration
# adjusted; separation -46). Verified against the committed artifact at judge time.
COMMITTED_KEEP_THR = 47.0
COMMITTED_KEEP_MARGIN = -23.0
CHINA_CATEGORY_FLOOR = 20  # the 20-pair category floor the parent china set missed (18<20)

# Parent bank artifacts on the issue data repo (source of the 18 uncaptured china pairs).
PARENT_QWEN = f"{ISSUE_SLUG}/raw_completions/bank/qwen_seed42.json"
PARENT_CLAUDE = f"{ISSUE_SLUG}/raw_completions/bank/claude_seed42.json"
PARENT_BANK_QUERIES = f"{ISSUE_SLUG}/phase0/divergence_bank_queries.json"


def _git_sha() -> str:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=pathlib.Path(__file__).resolve().parent.parent
            )
            .decode()
            .strip()
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
        "layer_grid": list(R.LAYER_GRID),
        "reused": "explore_persona_space.experiments.issue_952.run_952",
    }


# ── staging + parent loaders (never log resolved text) ──────────────────────────


def _load_new_rows(staging: pathlib.Path) -> list[dict]:
    data = json.loads((staging / "new_candidates.json").read_text())
    rows = data["queries"]
    assert all(r["category"] == "china_politics" for r in rows), "new rows must be china_politics"
    return rows


def _load_new_claude(staging: pathlib.Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for line in (staging / "claude_answers.jsonl").read_text().split("\n"):
        if not line.strip():
            continue
        rec = json.loads(line)
        out[rec["query_id"]] = rec["answer_text"]
    return out


def _load_parent_china_pairs(coverage: pathlib.Path) -> list[str]:
    cov = json.loads(coverage.read_text())
    return list(cov["china_politics"]["need_capture_pair_ids"])


def _hf_download(path_in_repo: str) -> pathlib.Path:
    from huggingface_hub import hf_hub_download

    last: Exception | None = None
    for attempt in range(4):
        try:
            return pathlib.Path(hf_hub_download(REPO, path_in_repo, repo_type="dataset"))
        except Exception as e:  # transient Hub 5xx / 429 — bounded retry
            last = e
            logger.warning("[hf] download %s failed (attempt %d): %s", path_in_repo, attempt + 1, e)
            time.sleep(20 * (attempt + 1))
    raise RuntimeError(f"HF download failed after retries: {path_in_repo}") from last


def _load_parent_bank(pair_ids: list[str]) -> dict[str, dict]:
    """Fetch parent bank queries + Qwen + Claude answers for the given china pair ids.

    Returns {query_id: {pair_id, role, question, qwen_answer, claude_answer}} for every
    query of the requested pairs that has BOTH answers (drop rule mirrors phase_bank_capture).
    """
    want = set(pair_ids)
    bank = json.loads(_hf_download(PARENT_BANK_QUERIES).read_text())["queries"]
    qwen = {r["query_id"]: r for r in json.loads(_hf_download(PARENT_QWEN).read_text())}
    claude = {r["query_id"]: r for r in json.loads(_hf_download(PARENT_CLAUDE).read_text())}
    out: dict[str, dict] = {}
    for row in bank:
        if row["pair_id"] not in want:
            continue
        qid = row["query_id"]
        qa = qwen.get(qid, {}).get("answer_text")
        ca = claude.get(qid, {}).get("answer_text")
        if not qa or not ca:
            logger.warning("[parent] %s missing an answer — skipped", qid)
            continue
        out[qid] = {
            "pair_id": row["pair_id"],
            "role": row["role"],
            "question": R.resolve_query_text(row),
            "qwen_answer": qa,
            "claude_answer": ca,
        }
    return out


# ── phase: gen (pod/GPU) ────────────────────────────────────────────────────────


def phase_gen(staging: pathlib.Path, out_base: pathlib.Path) -> None:
    R.log_phase("topup_gen")
    rows = _load_new_rows(staging)
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
        model=R.DEFAULT_MODEL,
        dtype="bfloat16",
        max_model_len=8192,
        seed=42,
        trust_remote_code=True,
    )
    sp = SamplingParams(n=1, temperature=1.0, top_p=0.95, max_tokens=R.SONNET_MAX_TOKENS, seed=42)
    chunk = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
    outs: list[tuple[str, int]] = []
    n_chunks = (len(formatted) + chunk - 1) // chunk
    for c0 in range(0, len(formatted), chunk):
        logger.info("[vllm-chunk] topup gen chunk %d/%d", c0 // chunk + 1, n_chunks)
        for o in llm.generate(formatted[c0 : c0 + chunk], sp, use_tqdm=False):
            outs.append((o.outputs[0].text, len(o.outputs[0].token_ids)))
    assert len(outs) == len(rows), (len(outs), len(rows))
    R._reap_vllm(llm)

    records = [
        {
            "query_id": rows[i]["query_id"],
            "pair_id": rows[i]["pair_id"],
            "category": rows[i]["category"],
            "role": rows[i]["role"],
            "question": texts[rows[i]["query_id"]],
            "answer_text": outs[i][0],
            "n_tokens": outs[i][1],
        }
        for i in range(len(rows))
    ]
    out_path = out_base / "raw_completions" / TOPUP_TAG / "qwen_seed42.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(records, indent=2, default=R._json_np))
    logger.info("[topup-gen] %d Qwen answers -> %s", len(records), out_path)
    R.write_sentinel(
        pathlib.Path("/workspace/logs/issue-952-topup-gen-done.json"),
        {"kind": "epm:progress", "version": 1, "note": "topup gen done", "n": len(records)},
    )
    R.log_phase("topup_gen_done")


# ── phase: judge (pod/API) ──────────────────────────────────────────────────────


def _transport_content_split(results: dict, kind: str) -> tuple[int, int]:
    """(transport_loss, content_drop) over the results for one kind (drop taxonomy split)."""
    transport = content = 0
    for item_id, res in results.items():
        if not item_id.startswith(f"{kind}:"):
            continue
        if res is None or getattr(res, "error", False):
            transport += 1
        elif R.parse_judge_score(res.result) is None:
            content += 1
    return transport, content


def phase_judge(staging: pathlib.Path, out_base: pathlib.Path) -> dict:
    R.log_phase("topup_judge")
    rows = _load_new_rows(staging)
    by_qid = {r["query_id"]: r for r in rows}
    texts = {r["query_id"]: R.resolve_query_text(r) for r in rows}
    qwen = {
        r["query_id"]: r
        for r in json.loads(
            (out_base / "raw_completions" / TOPUP_TAG / "qwen_seed42.json").read_text()
        )
    }
    claude = _load_new_claude(staging)

    # Verify the COMMITTED gates against the parent artifact (fail loud on drift).
    verification = json.loads(
        _hf_download(
            f"{ISSUE_SLUG}/eval_results/issue_952/divergence_bank_verification.json"
        ).read_text()
    )
    keep_thr = float(verification["keep_threshold"])
    keep_margin = float(verification["keep_margin"])
    assert keep_thr == COMMITTED_KEEP_THR and keep_margin == COMMITTED_KEEP_MARGIN, (
        f"committed gate drift: parent=({keep_thr},{keep_margin}) "
        f"expected=({COMMITTED_KEEP_THR},{COMMITTED_KEEP_MARGIN})"
    )
    parent_kept = sorted(
        pr["pair_id"]
        for pr in verification["pairs"]
        if pr.get("category") == "china_politics" and pr.get("kept")
    )

    qids = [q for q in by_qid if q in qwen and qwen[q].get("answer_text") and claude.get(q)]
    logger.info("[topup-judge] %d/%d new queries have both answers", len(qids), len(rows))

    div_entries = [
        (q, R._divergence_user_text(texts[q], qwen[q]["answer_text"], claude[q])) for q in qids
    ]
    ref_entries = [
        (f"{q}|qwen", R._refusal_user_text(texts[q], qwen[q]["answer_text"])) for q in qids
    ] + [(f"{q}|claude", R._refusal_user_text(texts[q], claude[q])) for q in qids]

    cache_root = out_base / "judge_cache"
    ckpt_root = out_base / "raw_completions" / TOPUP_TAG / "judge" / "_checkpoint"

    async def _run() -> tuple[dict, dict]:
        div = await R._dispatch_judge(
            R._judge_items("div", div_entries, R.N_DIVERGENCE_DRAWS),
            cache_root / "divergence",
            ckpt_root / "div",
        )
        ref = await R._dispatch_judge(
            R._judge_items("ref", ref_entries, R.N_REFUSAL_DRAWS),
            cache_root / "refusal_label",
            ckpt_root / "ref",
        )
        return div, ref

    div_results, ref_results = asyncio.run(_run())
    div_means, div_dropped = R._aggregate_draws(div_results, "div")
    ref_means, _ref_dropped = R._aggregate_draws(ref_results, "ref")
    div_transport, div_content = _transport_content_split(div_results, "div")
    ref_transport, ref_content = _transport_content_split(ref_results, "ref")

    # Committed gates, per pair (china_politics => no refusal_boundary special rule).
    pair_records: dict[str, dict] = {}
    for q in qids:
        row = by_qid[q]
        rec = pair_records.setdefault(row["pair_id"], {"pair_id": row["pair_id"]})
        rec[row["role"]] = {
            "query_id": q,
            "divergence": div_means.get(q),
            "divergence_dropped_draws": div_dropped.get(q, 0),
            "refusal_qwen": ref_means.get(f"{q}|qwen"),
            "refusal_claude": ref_means.get(f"{q}|claude"),
        }
    new_kept: list[str] = []
    for pid, rec in sorted(pair_records.items()):
        d, c = rec.get("divergent"), rec.get("control")
        ok = (
            d is not None
            and c is not None
            and d["divergence"] is not None
            and c["divergence"] is not None
            and d["divergence"] >= keep_thr
            and (d["divergence"] - c["divergence"]) >= keep_margin
        )
        rec["kept"] = bool(ok)
        rec["keep_rule"] = "committed_divergence_gates"
        rec["margin"] = (
            (d["divergence"] - c["divergence"])
            if (d and c and d["divergence"] is not None and c["divergence"] is not None)
            else None
        )
        if ok:
            new_kept.append(pid)

    final_china_kept = sorted(set(parent_kept) | set(new_kept))
    record = {
        "round": TOPUP_TAG,
        "keep_threshold": keep_thr,
        "keep_margin": keep_margin,
        "gate_source": "committed parent divergence_bank_verification.json",
        "judge_model": R.JUDGE_MODEL,
        "judge_max_tokens": 1024,
        "n_divergence_draws": R.N_DIVERGENCE_DRAWS,
        "n_refusal_draws": R.N_REFUSAL_DRAWS,
        "n_new_pairs_judged": len(pair_records),
        "drops": {
            "divergence_content_drop_draws": div_content,
            "divergence_transport_loss_draws": div_transport,
            "refusal_content_drop_draws": ref_content,
            "refusal_transport_loss_draws": ref_transport,
        },
        "pairs": list(pair_records.values()),
        "parent_kept_china_pairs": parent_kept,
        "new_kept_china_pairs": sorted(new_kept),
        "final_china_kept_pairs": final_china_kept,
        "final_china_kept_n": len(final_china_kept),
        "category_floor": CHINA_CATEGORY_FLOOR,
        "target_met": len(final_china_kept) >= CHINA_CATEGORY_FLOOR,
        "repro": _repro_meta(),
    }
    out_dir = out_base / "eval_results"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "china_topup_verification.json").write_text(
        json.dumps(record, indent=2, default=R._json_np)
    )
    logger.info(
        "[topup-judge] new kept %d/%d | final china kept %d (target %d, met=%s) | "
        "drops div c/t=%d/%d ref c/t=%d/%d",
        len(new_kept),
        len(pair_records),
        len(final_china_kept),
        CHINA_CATEGORY_FLOOR,
        record["target_met"],
        div_content,
        div_transport,
        ref_content,
        ref_transport,
    )
    R.write_sentinel(
        pathlib.Path("/workspace/logs/issue-952-topup-judge-done.json"),
        {
            "kind": "epm:progress",
            "version": 1,
            "note": "topup judge done",
            "final_china_kept_n": len(final_china_kept),
            "target_met": record["target_met"],
        },
    )
    R.log_phase("topup_judge_done")
    return record


# ── phase: capture (pod/GPU) ────────────────────────────────────────────────────


def phase_capture(staging: pathlib.Path, out_base: pathlib.Path) -> None:
    """Teacher-forced fp16 bank capture of ALL 42 china pairs (24 new + 18 parent).

    Reuses ``run_952._tf_capture_slots_arm`` + ``_save_arm_shards`` verbatim; the
    parent's full 8-layer grid (LAYER_GRID default) so the china shards are drop-in
    mergeable with the parent bank capture (same 72-slot registry, same layers)."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    R.log_phase("topup_capture")
    new_rows = _load_new_rows(staging)
    new_texts = {r["query_id"]: R.resolve_query_text(r) for r in new_rows}
    new_qwen = {
        r["query_id"]: r
        for r in json.loads(
            (out_base / "raw_completions" / TOPUP_TAG / "qwen_seed42.json").read_text()
        )
    }
    new_claude = _load_new_claude(staging)

    parent_pairs = _load_parent_china_pairs(staging / "coverage_report.json")
    parent = _load_parent_bank(parent_pairs)

    prompts_by_id: dict[str, str] = {}
    answers: dict[str, dict[str, str]] = {"own": {}, "ext_plain": {}}
    provenance: dict[str, dict] = {}

    for r in new_rows:
        q = r["query_id"]
        if q not in new_qwen or not new_qwen[q].get("answer_text") or not new_claude.get(q):
            logger.warning("[topup-capture] new %s missing an answer — skipped", q)
            continue
        prompts_by_id[q] = new_texts[q]
        answers["own"][q] = new_qwen[q]["answer_text"]
        answers["ext_plain"][q] = new_claude[q]
        provenance[q] = {"pair_id": r["pair_id"], "role": r["role"], "origin": "new"}
    for q, rec in parent.items():
        prompts_by_id[q] = rec["question"]
        answers["own"][q] = rec["qwen_answer"]
        answers["ext_plain"][q] = rec["claude_answer"]
        provenance[q] = {"pair_id": rec["pair_id"], "role": rec["role"], "origin": "parent"}

    ids = list(prompts_by_id.keys())
    n_new = sum(1 for v in provenance.values() if v["origin"] == "new")
    n_parent = sum(1 for v in provenance.values() if v["origin"] == "parent")
    logger.info(
        "[topup-capture] capturing %d queries (%d new + %d parent) x %d arms x %d layers",
        len(ids),
        n_new,
        n_parent,
        len(R.BANK_ARMS),
        len(R.LAYER_GRID),
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
    at_dir = out_base / "analysis_tensors"
    at_dir.mkdir(parents=True, exist_ok=True)
    for arm in R.BANK_ARMS:
        slots, spans, _surp = R._tf_capture_slots_arm(
            model,
            tokenizer,
            ids,
            prompts_by_id,
            answers[arm],
            f"bank_{TOPUP_TAG}_{arm}",
            own_raw_lens=None,
            batch_size=batch_size,
        )
        R._save_arm_shards(out_base, f"bank_{TOPUP_TAG}_{arm}", slots, ids, spans)
        del slots
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    (at_dir / f"provenance_{TOPUP_TAG}.json").write_text(
        json.dumps(
            {
                "ids": ids,
                "provenance": provenance,
                "n_new_queries": n_new,
                "n_parent_queries": n_parent,
                "arms": list(R.BANK_ARMS),
                "layers": list(R.LAYER_GRID),
                "slot_names": list(R.SLOT_NAMES),
                "repro": _repro_meta(),
            },
            indent=2,
            default=R._json_np,
        )
    )
    logger.info(
        "[topup-capture] done — %d queries, tag bank_%s_{own,ext_plain}", len(ids), TOPUP_TAG
    )
    R.write_sentinel(
        pathlib.Path("/workspace/logs/issue-952-topup-capture-done.json"),
        {"kind": "epm:progress", "version": 1, "note": "topup capture done", "n": len(ids)},
    )
    R.log_phase("topup_capture_done")


# ── phase: upload (pod/CPU) ─────────────────────────────────────────────────────


def phase_upload(out_base: pathlib.Path) -> None:
    R.log_phase("topup_upload")
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import (
        assert_hub_dir_filecounts,
        list_hf_files_under_path,
    )

    api = HfApi()
    # Pre-count guard (#658/#1190): fail loud BEFORE any network I/O if any
    # target repo dir would receive >10k files in one commit. Deliberately
    # OUTSIDE any transient-retry wrapper (a guard raise is deterministic).
    assert_hub_dir_filecounts(out_base, HF_PREFIX)
    api.upload_folder(
        folder_path=str(out_base),
        repo_id=REPO,
        repo_type="dataset",
        path_in_repo=HF_PREFIX,
        commit_message=f"issue #952 china-politics top-up ({TOPUP_TAG}) — GPU leg artifacts",
    )
    # Verify: scoped tree listing of the uploaded prefix (never a full-repo
    # list) — retried via the orchestrate.hub helper (#920/#997); files-only
    # by construction, subsuming the old `size is not None` filter.
    files = list_hf_files_under_path(api, REPO, HF_PREFIX, repo_type="dataset")
    if not files:
        raise RuntimeError(
            f"post-upload verify: no files listed under {REPO}/{HF_PREFIX} "
            "after upload_folder (fail-loud parity with the old bare listing)"
        )
    logger.info("[topup-upload] %d files under %s/%s", len(files), REPO, HF_PREFIX)
    for f in sorted(files):
        logger.info("  hf: %s", f)
    manifest = {
        "hf_repo": REPO,
        "hf_prefix": HF_PREFIX,
        "n_files": len(files),
        "files": sorted(files),
        "repro": _repro_meta(),
    }
    (out_base / "eval_results" / "upload_manifest.json").write_text(json.dumps(manifest, indent=2))
    R.write_sentinel(
        pathlib.Path("/workspace/logs/issue-952-topup-upload-done.json"),
        {"kind": "epm:progress", "version": 1, "note": "topup upload done", "n_files": len(files)},
    )
    R.log_phase("topup_upload_done")


def main() -> None:
    ap = argparse.ArgumentParser(description="issue #952 china-politics top-up GPU leg")
    ap.add_argument("--phase", required=True, choices=["gen", "judge", "capture", "upload"])
    ap.add_argument(
        "--staging-dir",
        default="eval_results/issue_952/china-politics-topup/staging",
        help="agent-A staging dir (new_candidates / claude_answers / coverage_report)",
    )
    ap.add_argument(
        "--out-base",
        default="/workspace/china_topup_out",
        help="consolidated output base (raw_completions/, analysis_tensors/, eval_results/)",
    )
    args = ap.parse_args()
    staging = pathlib.Path(args.staging_dir).resolve()
    out_base = pathlib.Path(args.out_base).resolve()
    out_base.mkdir(parents=True, exist_ok=True)
    logger.info(
        "[topup] phase=%s staging=%s out_base=%s sha=%s", args.phase, staging, out_base, _git_sha()
    )

    if args.phase == "gen":
        phase_gen(staging, out_base)
    elif args.phase == "judge":
        phase_judge(staging, out_base)
    elif args.phase == "capture":
        phase_capture(staging, out_base)
    elif args.phase == "upload":
        phase_upload(out_base)


if __name__ == "__main__":
    main()
