#!/usr/bin/env python3
"""Issue #2224 P0b: exact-ΔP base-generation pass (plan v3 §4 P0b).

Generates ONE greedy (temperature 0, the paper's exact-ΔP definition) natural
Qwen2.5-7B-Instruct response per working-pool prompt — plus an optional
CLI-injected extra-prompt slice (the #2221 real-twin suite prompts for 4a) —
writing rows that conform EXACTLY to the unit-1 capture consumer contract
(``issue2224_predictor_scores.load_natural_map``): one or more ``*.jsonl``
files under the corpus out-dir, each row carrying ``{"sample_id",
"response"}`` (extra diagnostic keys are permitted and ignored by the
consumer).

Conditioning parity: prompts are rendered with
``apply_chat_template([{user}], add_generation_prompt=True)`` — the same
rendered string ``issue2224_predictor_scores.render_prompt_segments`` splits
for the capture spans, so the natural response is generated under the exact
conditioning the exact-ΔP forward pass re-tokenizes.

Throughput/robustness shape:

- **Fan-out across every provisioned GPU by default** (plan §9: vLLM DP over
  pool samples). The parent process spawns one worker subprocess per GPU with
  ``CUDA_VISIBLE_DEVICES`` pinned in the LAUNCHER env (the #545 in-process
  clobber gotcha — an import-time cuInit defeats any later in-process pin)
  and an explicit ``env={**os.environ, ...}`` passthrough.
- **Chunk-checkpointed + resumable**: each worker writes atomic per-chunk
  JSONL files (tmp + ``os.replace`` — a crash never leaves a partial file);
  resume scans the out-dir for done sample_ids and generates only the pending
  set (code-style.md checkpoint-per-phase, T1/T2 triggers).
- **Cap-hit reporting**: the per-corpus ``gen_report_<corpus>.json`` records
  the realized ``finish_reason == "length"`` fraction with the pre-registered
  re-gen trigger (cap-hit > 2% per corpus ⇒ re-generate affected rows at
  ≥ 2× the cap; CLAUDE.md ``max_new_tokens`` rule).
- **Fail-loud upload** (``--upload``): ONE bulk ``upload_folder`` commit of
  the corpus dir to the HF data repo prefix
  ``issue2224_screening/raw_completions/exact_dp_base_gen/<corpus>/`` with an
  exact expected-set scoped verify (``hub._upload_folder_filtered`` — never a
  per-file loop, never an unscoped listing of the ~1M-file data repo; #833).

Content hygiene: corpus/response text is NEVER printed or logged — counts,
token stats and file digests only (real-world-corpus digest-only rule).

Usage::

    # pod, 4×H100 (parent fan-out; workers inherit shard slices):
    uv run python scripts/issue2224_gen_natural.py \\
        --pool data/issue_2224/pools/lmsys.jsonl --corpus lmsys \\
        --gpus 0,1,2,3 --upload
    # CPU plan mode (no vllm/torch import; resume/chunk arithmetic only):
    uv run python scripts/issue2224_gen_natural.py \\
        --pool data/issue_2224/pools/lmsys.jsonl --corpus lmsys --plan
    uv run python scripts/issue2224_gen_natural.py --import-check
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
for _p in (PROJECT_ROOT / "src", PROJECT_ROOT / "scripts"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # BEFORE torch/vllm imports: thread caps (VM) + HF token (#847)

from issue2224_common import (  # noqa: E402
    atomic_write_json,
    atomic_write_jsonl,
    load_jsonl,
    repro_meta,
    sha256_file,
    token_stats,
)
from issue778_lib import MODEL_NAME  # noqa: E402

logger = logging.getLogger("issue2224_gen_natural")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

GEN_SCHEMA_VERSION = 1
CAP_HIT_REGEN_TRIGGER = 0.02  # pre-registered: >2% length-truncated rows ⇒ re-gen at ≥2× cap
HF_PREFIX = "issue2224_screening/raw_completions/exact_dp_base_gen"
DEFAULT_OUT_ROOT = PROJECT_ROOT / "data" / "issue_2224" / "raw_completions" / "exact_dp_base_gen"


# ── Input loading ────────────────────────────────────────────────────────────────


def load_prompt_rows(pool: Path | None, extra: Path | None) -> list[dict]:
    """Pool prompts (+ optional extra-prompt slice) as ``{"sample_id","prompt"}`` rows.

    Fails loud on missing keys, duplicate sample_ids, or a pool/extra id
    collision (the extra slice is the 4a suite-prompts injection, plan §4 P0b).
    """
    rows: list[dict] = []
    seen: set[str] = set()
    for src, path in (("pool", pool), ("extra", extra)):
        if path is None:
            continue
        for i, r in enumerate(load_jsonl(Path(path))):
            if "sample_id" not in r or "prompt" not in r:
                raise RuntimeError(f"{path}:{i}: row missing sample_id/prompt keys")
            sid = str(r["sample_id"])
            if sid in seen:
                raise RuntimeError(f"{path}: duplicate sample_id {sid!r} (pool/extra collision?)")
            seen.add(sid)
            rows.append({"sample_id": sid, "prompt": str(r["prompt"]), "source": src})
    if not rows:
        raise RuntimeError("no prompts loaded — pass --pool and/or --extra-prompts")
    return rows


def scan_done_ids(out_dir: Path, exclude_truncated: bool = False) -> set[str]:
    """sample_ids already generated (resume predicate — same scan the fresh run uses).

    ``exclude_truncated`` treats ``finish_reason == "length"`` rows as NOT done —
    the read-only (``--plan``) view of the ``--regen-truncated`` path (the
    fan-out instead REWRITES the chunk files via :func:`drop_truncated_rows`
    BEFORE the pending scan, so duplicate rows are impossible).
    """
    done: set[str] = set()
    for f in sorted(Path(out_dir).glob("*.jsonl")):
        for r in load_jsonl(f):
            if exclude_truncated and r.get("finish_reason") == "length":
                continue
            done.add(str(r["sample_id"]))
    return done


def pending_rows(
    rows: list[dict], out_dir: Path, exclude_truncated: bool = False
) -> tuple[list[dict], int]:
    """(pending rows, n_done). One predicate for --plan, resume, and fresh runs."""
    done = scan_done_ids(out_dir, exclude_truncated=exclude_truncated)
    pend = [r for r in rows if r["sample_id"] not in done]
    return pend, len(done)


def drop_truncated_rows(out_dir: Path) -> int:
    """M2: rewrite chunk files DROPPING length-truncated rows for re-generation.

    Makes the pre-registered cap-hit re-gen trigger executable: dropped rows
    fall out of the done-id resume scan and are re-generated (at the raised
    ``--max-new-tokens``); atomic rewrites (emptied files unlinked) keep the
    scan consistent and duplicates impossible.
    """
    n_dropped = 0
    for f in sorted(Path(out_dir).glob("*.jsonl")):
        rows = load_jsonl(f)
        kept = [r for r in rows if r.get("finish_reason") != "length"]
        if len(kept) == len(rows):
            continue
        n_dropped += len(rows) - len(kept)
        if kept:
            atomic_write_jsonl(kept, f)
        else:
            f.unlink()
    return n_dropped


def check_gen_regime(out_dir: Path, args) -> None:
    """M2: refuse a resume across a silently-drifted decoding regime.

    The ``gen_regime.json`` sidecar records what prior rows were generated
    under. A model mismatch always fails loud; a RAISED ``--max-new-tokens``
    is legal only with ``--regen-truncated`` (truncated rows are dropped +
    re-generated at the new cap; naturally-finished rows are unaffected by a
    cap raise); lowering the cap is never legal.
    """
    sidecar = Path(out_dir) / "gen_regime.json"
    if not sidecar.exists():
        return
    reg = json.loads(sidecar.read_text())
    if reg.get("model") != args.model:
        raise RuntimeError(
            f"{out_dir}: prior rows generated with model={reg.get('model')!r} != "
            f"current {args.model!r} — never mix generator models in one corpus dir"
        )
    old_cap = int(reg.get("max_new_tokens"))
    if old_cap == args.max_new_tokens:
        return
    if args.max_new_tokens < old_cap:
        raise RuntimeError(
            f"{out_dir}: max_new_tokens {args.max_new_tokens} < prior {old_cap} — "
            f"never lower a generation cap on a resumed corpus"
        )
    if not args.regen_truncated:
        raise RuntimeError(
            f"{out_dir}: max_new_tokens raised {old_cap} -> {args.max_new_tokens} without "
            f"--regen-truncated — pass it to re-generate the length-truncated rows at the "
            f"new cap (the pre-registered >2% cap-hit re-gen action), or match the cap"
        )


def write_gen_regime(out_dir: Path, args) -> None:
    """Persist the decoding-regime sidecar the resume predicate consults (M2)."""
    atomic_write_json(
        {
            "model": args.model,
            "temperature": 0.0,
            "max_new_tokens": args.max_new_tokens,
            "meta": repro_meta("issue2224_gen_natural.regime"),
        },
        Path(out_dir) / "gen_regime.json",
    )


def chunk_plan(n_pending: int, chunk_size: int) -> int:
    """Number of checkpoint chunks a worker will write."""
    return (n_pending + chunk_size - 1) // chunk_size if n_pending else 0


# ── Worker (one GPU) ─────────────────────────────────────────────────────────────


def run_worker(args) -> dict:
    """Generate the pending slice for this shard; returns the shard report dict."""
    out_dir = Path(args.out_dir) / args.corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_prompt_rows(args.pool, args.extra_prompts)
    if args.limit:
        rows = rows[: args.limit]
    rows = rows[args.shard_index :: args.num_shards]
    pend, n_done = pending_rows(rows, out_dir)
    logger.info(
        "[gen shard %d/%d] corpus=%s total=%d done=%d pending=%d",
        args.shard_index,
        args.num_shards,
        args.corpus,
        len(rows),
        n_done,
        len(pend),
    )
    shard_report = {
        "shard_index": args.shard_index,
        "num_shards": args.num_shards,
        "n_assigned": len(rows),
        "n_done_prior": n_done,
        "n_generated": 0,
        "n_length_truncated": 0,
        "new_token_stats": {},
    }
    if not pend:
        return shard_report

    from transformers import AutoTokenizer

    from issue778_lib import build_vllm_engine

    tok = AutoTokenizer.from_pretrained(args.model)  # once per process (429 gotcha)
    llm = build_vllm_engine(args.model, gpu_memory_utilization=args.gpu_mem_util)
    from vllm import SamplingParams

    # GREEDY, per the paper's exact-ΔP definition (plan §4 arm table) — not a knob.
    sp = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens)

    n_chunks = chunk_plan(len(pend), args.chunk_size)
    new_tok_counts: list[int] = []
    t0 = time.time()
    for ci in range(n_chunks):
        chunk = pend[ci * args.chunk_size : (ci + 1) * args.chunk_size]
        prompts = [
            tok.apply_chat_template(
                [{"role": "user", "content": r["prompt"]}],
                tokenize=False,
                add_generation_prompt=True,
            )
            for r in chunk
        ]
        outs = llm.generate(prompts, sp)
        assert len(outs) == len(chunk), (len(outs), len(chunk))
        out_rows = []
        for r, o in zip(chunk, outs):
            comp = o.outputs[0]
            n_new = len(comp.token_ids)
            new_tok_counts.append(n_new)
            if comp.finish_reason == "length":
                shard_report["n_length_truncated"] += 1
            out_rows.append(
                {
                    "sample_id": r["sample_id"],
                    "response": comp.text,
                    "finish_reason": comp.finish_reason,
                    "n_new_tokens": n_new,
                    "source": r["source"],
                }
            )
        # Atomic per-chunk checkpoint: a crash never leaves a partial file, so the
        # resume scan (done-id set) is always consistent.
        chunk_path = out_dir / (
            f"gen_{args.corpus}_s{args.shard_index:02d}_{int(time.time())}_{ci:05d}.jsonl"
        )
        atomic_write_jsonl(out_rows, chunk_path)
        shard_report["n_generated"] += len(out_rows)
        print(
            f"[gen] unit {ci + 1}/{n_chunks} shard={args.shard_index} "
            f"rows={len(out_rows)} elapsed={time.time() - t0:.0f}s",
            flush=True,
        )
    shard_report["new_token_stats"] = token_stats(new_tok_counts)
    return shard_report


# ── Parent (fan-out + report + upload) ───────────────────────────────────────────


def detect_gpus(args) -> list[str]:
    """GPU ids to fan out over: --gpus wins; else every visible device (guideline 2)."""
    if args.gpus:
        return [g.strip() for g in args.gpus.split(",") if g.strip()]
    import torch

    n = torch.cuda.device_count()
    if n == 0:
        raise RuntimeError("no CUDA device visible — pass --gpus or run on a GPU pod")
    return [str(i) for i in range(n)]


def run_fanout(args) -> int:
    """Spawn one worker per GPU (CVD pinned in the launcher env), then verify + report."""
    out_dir = Path(args.out_dir) / args.corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_prompt_rows(args.pool, args.extra_prompts)
    if args.limit:
        rows = rows[: args.limit]

    check_gen_regime(out_dir, args)
    if args.regen_truncated:
        n_regen = drop_truncated_rows(out_dir)
        logger.info(
            "[gen] --regen-truncated: dropped %d length-truncated row(s) for re-generation "
            "at max_new_tokens=%d",
            n_regen,
            args.max_new_tokens,
        )
    write_gen_regime(out_dir, args)

    from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

    # Text-only phase; ~2 KB/row. Pending-aware tiny floor + fixed margin.
    pend_all, _ = pending_rows(rows, out_dir)
    if pend_all:
        assert_out_root_headroom(
            Path(args.out_dir), max(2.0, 2.0 + 4e-6 * len(pend_all)), phase="P0b gen"
        )

        gpus = detect_gpus(args)
        log_dir = out_dir / "logs"
        log_dir.mkdir(exist_ok=True)
        procs = []
        for si, gpu in enumerate(gpus):
            cmd = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--corpus",
                args.corpus,
                "--out-dir",
                str(args.out_dir),
                "--model",
                args.model,
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--chunk-size",
                str(args.chunk_size),
                "--gpu-mem-util",
                str(args.gpu_mem_util),
                "--shard-index",
                str(si),
                "--num-shards",
                str(len(gpus)),
            ]
            if args.pool:
                cmd += ["--pool", str(args.pool)]
            if args.extra_prompts:
                cmd += ["--extra-prompts", str(args.extra_prompts)]
            if args.limit:
                cmd += ["--limit", str(args.limit)]
            # CVD pinned in the LAUNCHER env (the #545 import-time-cuInit clobber
            # gotcha) + explicit env passthrough (subprocess-env rule).
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": gpu}
            log_path = log_dir / f"worker_s{si:02d}.log"
            lf = open(log_path, "a")
            logger.info("[gen] launching shard %d on GPU %s -> %s", si, gpu, log_path)
            procs.append((si, subprocess.Popen(cmd, env=env, stdout=lf, stderr=lf), lf))
        failures = []
        for si, p, lf in procs:
            rc = p.wait()
            lf.close()
            if rc != 0:
                failures.append((si, rc))
        if failures:
            raise RuntimeError(
                f"[gen] {len(failures)} worker(s) failed: {failures} — "
                f"see {log_dir}/worker_s*.log; re-run to resume (chunk checkpoints kept)"
            )

    # Completeness verify + merged report (parent-side, from disk — worker-agnostic).
    done = scan_done_ids(out_dir)
    expected = {r["sample_id"] for r in rows}
    missing = sorted(expected - done)
    if missing:
        raise RuntimeError(
            f"[gen] {len(missing)}/{len(expected)} sample_ids still missing after all "
            f"workers exited 0 (first 5: {missing[:5]}) — resume by re-running"
        )
    report = build_report(out_dir, rows, args)
    atomic_write_json(report, out_dir / f"gen_report_{args.corpus}.json")
    logger.info(
        "[gen] corpus=%s complete: n=%d cap_hit=%.4f regen_trigger_fired=%s",
        args.corpus,
        report["n_total"],
        report["cap_hit_fraction"],
        report["regen_trigger_fired"],
    )
    if args.upload:
        upload_corpus_dir(out_dir, args.corpus)
    # Standalone-lane dispatcher terminal: this driver IS the dispatcher when
    # launched top-level; issue2224_suite4a_runner.sh invokes it with stdout
    # redirected to its own per-phase log, off the main-log path.
    # noqa: phase-done-reserved (mode: top-level dispatcher; invoker: suite4a_runner redirects)
    print(f"[phase=done] gen_natural corpus={args.corpus} n={report['n_total']}", flush=True)
    return 0


def build_report(out_dir: Path, rows: list[dict], args) -> dict:
    """Merged per-corpus generation report (cap-hit fraction + file digests)."""
    n_len = 0
    n_total = 0
    tok_counts: list[int] = []
    per_file: dict[str, dict] = {}
    for f in sorted(out_dir.glob("*.jsonl")):
        rs = load_jsonl(f)
        per_file[f.name] = {"n_rows": len(rs), "sha256": sha256_file(f)}
        for r in rs:
            n_total += 1
            if r.get("finish_reason") == "length":
                n_len += 1
            if "n_new_tokens" in r:
                tok_counts.append(int(r["n_new_tokens"]))
    frac = (n_len / n_total) if n_total else 0.0
    return {
        "schema": GEN_SCHEMA_VERSION,
        "corpus": args.corpus,
        "model": args.model,
        "decoding": {"temperature": 0.0, "max_new_tokens": args.max_new_tokens},
        "n_expected": len(rows),
        "n_total": n_total,
        "n_length_truncated": n_len,
        "cap_hit_fraction": round(frac, 6),
        "regen_trigger": {
            "threshold": CAP_HIT_REGEN_TRIGGER,
            "action": "re-generate length-truncated rows at >=2x max_new_tokens",
        },
        "regen_trigger_fired": bool(frac > CAP_HIT_REGEN_TRIGGER),
        "new_token_stats": token_stats(tok_counts),
        "files": per_file,
        "meta": repro_meta("issue2224_gen_natural"),
    }


def upload_corpus_dir(out_dir: Path, corpus: str) -> None:
    """ONE bulk fail-loud upload_folder commit + exact-set scoped verify (#833/#664)."""
    from explore_persona_space.orchestrate.hub import (
        DEFAULT_DATASET_REPO,
        _upload_folder_filtered,
    )

    path_in_repo = f"{HF_PREFIX}/{corpus}"
    rels = sorted(
        str(p.relative_to(out_dir)) for p in out_dir.rglob("*") if p.suffix in (".jsonl", ".json")
    )
    if not rels:
        raise RuntimeError(f"[gen-upload] nothing to upload under {out_dir}")
    expected = [f"{path_in_repo}/{rel}" for rel in rels]
    url = _upload_folder_filtered(
        local_dir=out_dir,
        repo_id=DEFAULT_DATASET_REPO,
        repo_type="dataset",
        path_in_repo=path_in_repo,
        allow_patterns=["*.jsonl", "*.json"],
        expected_repo_paths=expected,
    )
    if not url:
        raise RuntimeError(
            f"[gen-upload] bulk upload of {out_dir} -> {path_in_repo} FAILED or "
            f"verified incomplete (persist-unconditionally contract, plan §4)"
        )
    logger.info("[gen-upload] verified %d files at %s", len(expected), url)


def run_plan(args) -> int:
    """CPU plan mode: resume/chunk arithmetic only (no torch/vllm import, READ-only —
    --regen-truncated here only PREVIEWS the re-gen pending set; the fan-out rewrites)."""
    out_dir = Path(args.out_dir) / args.corpus
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = load_prompt_rows(args.pool, args.extra_prompts)
    if args.limit:
        rows = rows[: args.limit]
    check_gen_regime(out_dir, args)
    pend, n_done = pending_rows(rows, out_dir, exclude_truncated=args.regen_truncated)
    plan = {
        "corpus": args.corpus,
        "n_expected": len(rows),
        "n_done": n_done,
        "n_pending": len(pend),
        "regen_truncated": bool(args.regen_truncated),
        "chunks_per_shard_at_1_shard": chunk_plan(len(pend), args.chunk_size),
        "chunk_size": args.chunk_size,
    }
    print(json.dumps(plan, indent=2))
    return 0


# ── Entry point ──────────────────────────────────────────────────────────────────


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Issue #2224 P0b exact-ΔP base generation (plan v3 §4 P0b)."
    )
    parser.add_argument("--pool", type=Path, default=None, help="pool JSONL (P0a output)")
    parser.add_argument(
        "--extra-prompts",
        type=Path,
        default=None,
        help='extra {"sample_id","prompt"} JSONL (the 4a #2221 suite slice, CLI-injected)',
    )
    parser.add_argument("--corpus", default=None, help="corpus slug (out-dir + report key)")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_ROOT)
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument(
        "--max-new-tokens", type=int, default=2048, help="generation cap (project default 2048)"
    )
    parser.add_argument("--chunk-size", type=int, default=2000, help="rows per checkpoint chunk")
    parser.add_argument("--gpu-mem-util", type=float, default=0.85, help="vLLM HBM fraction")
    parser.add_argument("--gpus", default=None, help="comma GPU ids (default: all visible)")
    parser.add_argument("--limit", type=int, default=None, help="cap prompt rows (smoke slices)")
    parser.add_argument("--upload", action="store_true", help="fail-loud HF upload at the end")
    parser.add_argument(
        "--regen-truncated",
        action="store_true",
        help="drop finish_reason=='length' rows and re-generate them at the current "
        "(raised) --max-new-tokens (the pre-registered >2%% cap-hit re-gen action; M2)",
    )
    parser.add_argument("--plan", action="store_true", help="print resume/chunk plan and exit")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--shard-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--num-shards", type=int, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--import-check", action="store_true")
    return parser


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        import importlib

        for mod in ("numpy", "torch", "transformers", "vllm"):
            importlib.import_module(mod)
        from transformers import AutoTokenizer  # noqa: F401
        from vllm import LLM, SamplingParams  # noqa: F401

        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined
        from explore_persona_space.orchestrate.hub import (  # noqa: F401
            DEFAULT_DATASET_REPO,
            _upload_folder_filtered,
        )
        from explore_persona_space.orchestrate.preflight import (  # noqa: F401
            assert_out_root_headroom,
        )
        from issue778_lib import build_vllm_engine  # noqa: F401

        assert_args_attributes_defined(__file__)
        print("[import-check] OK issue2224_gen_natural")
        return 0
    if not args.corpus:
        raise SystemExit("--corpus required")
    if args.plan:
        return run_plan(args)
    if args.worker:
        report = run_worker(args)
        out_dir = Path(args.out_dir) / args.corpus
        atomic_write_json(report, out_dir / f"shard_report_s{args.shard_index:02d}.json")
        return 0
    return run_fanout(args)


if __name__ == "__main__":
    sys.exit(main())
