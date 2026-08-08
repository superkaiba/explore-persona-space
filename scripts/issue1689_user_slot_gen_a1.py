#!/usr/bin/env python
"""Issue #1689 user-slot-recapture — addendum C prerequisite: ON-POLICY a1.

Generates, per measured model, the model's OWN greedy reply to u1 for every
conversation in the parent's two-turn LMSYS corpus. Addendum C's capture cells
then render `User: {u1} / <Label>: {a1_onpolicy} / User: {u2}` so the assistant
turn is the measured model's own text instead of LMSYS's off-policy reply — the
one variable C changes.

WHY A FOURTH SCRIPT (declared deviation from "new variants, not new scripts"):
the render (`issue1689_user_slot_render.py`) is deterministic and GPU-free by
contract — every consumer re-runs it to reproduce a row set — so a vLLM
generation phase cannot live inside it. This script is the generation phase; its
output is a normal staged input the render consumes like any other u2 source.

Hazard checklist wired here (each with its incident):
  - `VLLM_WORKER_MULTIPROC_METHOD=spawn` set at MODULE TOP before any vllm
    import — the #628 fork-poisoned EngineCore silent death.
  - Chunked `llm.generate` (default 500, `EPM_VLLM_GREEDY_CHUNK_SIZE`) with a
    per-chunk INFO line — the #664 large-batch deadlock + poller liveness.
  - `use_tqdm=False` on every generate — the #613 tqdm ZeroDivisionError.
  - `--enforce-eager` / `--no-prefix-caching` knobs, DEFAULT-OFF, threaded at
    the single `LLM(` site — the #1324/#1092 real-corpus long-prompt hang/IMA
    class. Both cells of a comparison must run under ONE engine config.
  - Per-chunk atomic checkpoint + fingerprint-gated resume — the intra-phase
    checkpoint floor (a 3800-row greedy pass is well over the ~1h floor).
  - Explicit `sys.exit(0)` — the #1689 Phase-A PyGILState_Release atexit race.
  - `load_dotenv()` before any heavy import — the shared-VM thread caps.
  - Per-GPU fan-out pins `CUDA_VISIBLE_DEVICES` in the CHILD env AND passes the
    matching `--gpu-id` (#545/#1090), and indexes INTO a pre-set CVD allocation
    rather than exporting absolute device indices (the SLURM shared-node
    lesson; `scripts/issue1345_boundary_ablation_launch_gen.sh` is the pattern).

Content hygiene: real LMSYS user text. This script NEVER prints row content —
only counts, token lengths, sha256 digests and conv_ids.

Smoke: `--smoke` caps the row set (default 8) and runs the IDENTICAL code path;
`--mock-generate` swaps ONLY the vLLM boundary for a deterministic stub so the
chunking / checkpoint / resume / upload wiring is exercisable on CPU.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

# MUST precede any `import vllm` (vLLM reads this at import time) — #628.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()


def _ensure_repo_root_on_syspath() -> Path:
    """Repo root on sys.path so `scripts.*` imports resolve in SCRIPT mode."""
    root = Path(__file__).resolve().parents[1]
    sentinel = root / "scripts" / "issue1689_user_slot_render.py"
    if not sentinel.exists():
        raise RuntimeError(f"repo-root resolution failed: {sentinel} missing")
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    return root


REPO_ROOT = _ensure_repo_root_on_syspath()

ROUND_LABEL = "user_slot_recapture"
GEN_SUBDIR = "gen_a1"
# Greedy, and long enough that a natural reply is never silently truncated
# (CLAUDE.md: >= 2x the longest trained completion; Qwen replies run ~150 tok).
MAX_NEW_TOKENS = 1024
CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
# vLLM prompt-length ceiling. A corpus u1 tail can be long; rows whose
# FORMATTED prompt exceeds (max_model_len - MAX_NEW_TOKENS) are dropped at LOAD
# time rather than killing the engine at add_request (#952/#1738).
MAX_MODEL_LEN = 8192
SMOKE_ROWS = 8


# ---------------------------------------------------------------------------
# Row set
# ---------------------------------------------------------------------------


def load_corpus_u1(stage_root: Path, *, revision: str = "") -> list[dict]:
    """Distinct (conv_id, u1) rows from the parent two-turn LMSYS corpus.

    The corpus holds 11400 rows over 3800 conversations (3 byte-identical
    copies each), so this collapses to one row per conversation and fails loud
    if a duplicate group's u1 ever diverges.
    """
    from scripts.issue1689_user_slot_render import (
        PARENT_REVISION,
        _norm_turn,
        _read_jsonl,
        _source_paths,
        stage_source_files,
    )

    staged = stage_source_files(stage_root, revision=revision or PARENT_REVISION)
    rows: dict[str, str] = {}
    n_seen = 0
    for path in _source_paths(staged, "corpus", "two_turn_lmsys"):
        for row in _read_jsonl(path):
            cid = str(row["conv_id"])
            u1 = _norm_turn(row["u1"])
            n_seen += 1
            prev = rows.get(cid)
            if prev is None:
                rows[cid] = u1
            elif prev != u1:
                raise RuntimeError(
                    f"conv_id {cid!r} has DIVERGENT u1 across its duplicate corpus rows "
                    "— the on-policy a1 prompt is not a per-conversation property"
                )
    if not rows:
        raise RuntimeError("no corpus rows read — staging returned nothing usable")
    out = [{"conv_id": c, "u1": u} for c, u in sorted(rows.items())]
    print(
        f"[gen-a1] corpus: {n_seen} rows -> {len(out)} conversations "
        f"(dup_factor {n_seen / len(out):.2f})",
        flush=True,
    )
    return out


def filter_by_prompt_budget(rows: list[dict], tokenizer, *, budget: int) -> tuple[list[dict], dict]:
    """Drop rows whose FORMATTED prompt exceeds the generation budget.

    vLLM raises at `add_request` on an over-length prompt and that kills the
    WHOLE engine, not the row (#952/#1738) — so the filter runs at LOAD time,
    tokenizing exactly what the generate call renders. Drops are recorded
    digest-only (conv_id + token count, never row text).
    """
    kept: list[dict] = []
    dropped: list[dict] = []
    prompts = [_chat_prompt(r["u1"], tokenizer) for r in rows]
    lens = [len(ids) for ids in tokenizer(prompts, add_special_tokens=False)["input_ids"]]
    for row, prompt, n_tok in zip(rows, prompts, lens, strict=True):
        if n_tok > budget:
            dropped.append({"conv_id": row["conv_id"], "prompt_tokens": int(n_tok)})
        else:
            kept.append({**row, "prompt": prompt, "prompt_tokens": int(n_tok)})
    digest = {
        "budget_tokens": int(budget),
        "n_in": len(rows),
        "n_kept": len(kept),
        "n_dropped": len(dropped),
        "dropped": dropped[:50],
        "max_kept_prompt_tokens": max((r["prompt_tokens"] for r in kept), default=0),
    }
    print(
        f"[gen-a1] prompt-budget filter: kept {len(kept)}/{len(rows)} "
        f"(budget {budget} tok, max kept {digest['max_kept_prompt_tokens']})",
        flush=True,
    )
    if not kept:
        raise RuntimeError(f"prompt-budget filter kept 0 of {len(rows)} rows at budget {budget}")
    return kept, digest


def _chat_prompt(u1: str, tokenizer) -> str:
    """The measured model's own chat rendering of a single u1 user turn."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": u1}], tokenize=False, add_generation_prompt=True
    )


# ---------------------------------------------------------------------------
# Checkpoint / resume
# ---------------------------------------------------------------------------


def fingerprint(model: str, rows: list[dict], *, max_new_tokens: int, engine: dict) -> str:
    """Resume fingerprint over EVERY output-affecting key.

    A resume that ignores an output-affecting flag silently reuses wrong rows
    (#722 r3), so the model, the exact prompt set, the token cap AND the engine
    knobs all enter the key.
    """
    h = hashlib.sha256()
    h.update(model.encode())
    h.update(f"|max_new={max_new_tokens}|".encode())
    h.update(json.dumps(engine, sort_keys=True).encode())
    for r in rows:
        h.update(b"\x00")
        h.update(r["conv_id"].encode())
        h.update(hashlib.sha256(r["prompt"].encode()).digest())
    return h.hexdigest()[:16]


def load_checkpoint(out_path: Path, meta_path: Path, fp: str) -> dict[str, dict]:
    """Completed rows from a prior run whose fingerprint MATCHES; else empty."""
    if not out_path.exists() or not meta_path.exists():
        return {}
    with meta_path.open(encoding="utf-8") as fh:
        meta = json.load(fh)
    if meta.get("fingerprint") != fp:
        print(
            f"[gen-a1] checkpoint fingerprint mismatch "
            f"({meta.get('fingerprint')} != {fp}) — starting fresh",
            flush=True,
        )
        return {}
    done: dict[str, dict] = {}
    with out_path.open(encoding="utf-8") as fh:  # text-mode iteration, never splitlines
        for line in fh:
            if line.strip():
                row = json.loads(line)
                done[str(row["conv_id"])] = row
    print(f"[gen-a1] resume: {len(done)} rows already generated", flush=True)
    return done


def append_rows(out_path: Path, rows: list[dict]) -> None:
    """Append completed rows durably (one fsync'd append per chunk)."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def write_meta(meta_path: Path, payload: dict) -> None:
    """Atomically rewrite the sidecar meta (tmp + os.replace, same dir)."""
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = meta_path.with_name(meta_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(tmp, meta_path)


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------


def build_engine(model: str, *, enforce_eager: bool, no_prefix_caching: bool):
    """The single `LLM(` site. Hang-mitigation knobs are DEFAULT-OFF."""
    from vllm import LLM

    kwargs: dict = {
        "model": model,
        "max_model_len": MAX_MODEL_LEN,
        "gpu_memory_utilization": 0.85,
    }
    if enforce_eager:
        kwargs["enforce_eager"] = True
    if no_prefix_caching:
        kwargs["enable_prefix_caching"] = False
    print(
        f"[gen-a1] engine: {model} max_model_len={MAX_MODEL_LEN} "
        f"enforce_eager={enforce_eager} prefix_caching={not no_prefix_caching}",
        flush=True,
    )
    return LLM(**kwargs)


def greedy_chunked(llm, prompts: list[str], *, max_new_tokens: int, chunk: int = CHUNK_SIZE):
    """Greedy generate in bounded chunks, yielding (offset, texts, finishes).

    Chunking is the #664 deadlock prevention and the per-chunk log line is
    load-bearing poller liveness, not decoration.
    """
    from vllm import SamplingParams

    sp = SamplingParams(temperature=0.0, max_tokens=max_new_tokens)
    n_chunks = (len(prompts) + chunk - 1) // chunk
    for i in range(0, len(prompts), chunk):
        part = prompts[i : i + chunk]
        print(
            f"[vllm-chunk] gen_a1 chunk {i // chunk + 1}/{n_chunks} ({len(part)} prompts)",
            flush=True,
        )
        outs = llm.generate(part, sp, use_tqdm=False)
        yield (
            i,
            [o.outputs[0].text for o in outs],
            [getattr(o.outputs[0], "finish_reason", None) for o in outs],
        )


def mock_greedy_chunked(_llm, prompts: list[str], *, max_new_tokens: int, chunk: int = CHUNK_SIZE):
    """Deterministic stub for the vLLM boundary ONLY (CPU smoke).

    Fakes the remote GPU boundary and nothing else — chunking, checkpointing,
    resume and upload all run their real bodies.
    """
    n_chunks = (len(prompts) + chunk - 1) // chunk
    for i in range(0, len(prompts), chunk):
        part = prompts[i : i + chunk]
        print(
            f"[vllm-chunk] gen_a1 chunk {i // chunk + 1}/{n_chunks} ({len(part)} prompts) [MOCK]",
            flush=True,
        )
        texts = [f"mock reply {i + j} (max_new={max_new_tokens})" for j in range(len(part))]
        yield i, texts, ["stop"] * len(part)


def run_worker(args) -> int:
    """Generate a1_onpolicy for ONE model, resumably, then upload."""
    from scripts.issue1689_user_slot_render import (
        base_metadata,
        model_short,
        sha256_text,
    )

    tokenizer = _get_tokenizer(args.model)
    rows = load_corpus_u1(args.stage_root, revision=args.revision)
    if args.smoke:
        rows = rows[: args.smoke_rows]
        print(f"[gen-a1] SMOKE: {len(rows)} rows", flush=True)
    rows, budget_digest = filter_by_prompt_budget(
        rows, tokenizer, budget=MAX_MODEL_LEN - args.max_new_tokens
    )

    engine_cfg = {
        "enforce_eager": bool(args.enforce_eager),
        "no_prefix_caching": bool(args.no_prefix_caching),
        "max_model_len": MAX_MODEL_LEN,
    }
    fp = fingerprint(args.model, rows, max_new_tokens=args.max_new_tokens, engine=engine_cfg)
    short = model_short(args.model)
    out_dir = args.out_root / GEN_SUBDIR
    out_path = out_dir / f"user_slot_a1_onpolicy_{short}.jsonl"
    meta_path = out_dir / f"user_slot_a1_onpolicy_{short}.meta.json"
    done = load_checkpoint(out_path, meta_path, fp)
    pending = [r for r in rows if r["conv_id"] not in done]
    print(
        f"[gen-a1] {short}: {len(rows)} rows, {len(done)} done, {len(pending)} pending "
        f"(fingerprint {fp})",
        flush=True,
    )

    if pending:
        if args.dry_run:
            print("[gen-a1] --dry-run: engine NOT built, nothing generated", flush=True)
            return 0
        gen = mock_greedy_chunked if args.mock_generate else greedy_chunked
        llm = (
            None
            if args.mock_generate
            else build_engine(
                args.model,
                enforce_eager=args.enforce_eager,
                no_prefix_caching=args.no_prefix_caching,
            )
        )
        prompts = [r["prompt"] for r in pending]
        n_written = 0
        for offset, texts, finishes in gen(
            llm, prompts, max_new_tokens=args.max_new_tokens, chunk=args.chunk_size
        ):
            batch = []
            for j, (text, finish) in enumerate(zip(texts, finishes, strict=True)):
                src = pending[offset + j]
                batch.append(
                    {
                        "conv_id": src["conv_id"],
                        "a1_onpolicy": text,
                        "finish_reason": finish,
                        "u1_sha256": sha256_text(src["u1"]),
                        "a1_onpolicy_sha256": sha256_text(text),
                        "a1_onpolicy_chars": len(text),
                        "prompt_tokens": src["prompt_tokens"],
                    }
                )
            # persist the chunk the moment it completes (intra-phase floor)
            append_rows(out_path, batch)
            n_written += len(batch)
            write_meta(
                meta_path,
                {
                    "fingerprint": fp,
                    "model": args.model,
                    "model_short": short,
                    "max_new_tokens": args.max_new_tokens,
                    "engine": engine_cfg,
                    "chunk_size": args.chunk_size,
                    "n_rows_target": len(rows),
                    "n_rows_written": len(done) + n_written,
                    "mock_generate": bool(args.mock_generate),
                    "prompt_budget": budget_digest,
                    "prompt_render": "apply_chat_template([u1], add_generation_prompt=True)",
                    "updated_utc": datetime.now(UTC).isoformat(),
                    **base_metadata(),
                },
            )
            print(
                f"[gen-a1] {short}: chunk persisted, {len(done) + n_written}/{len(rows)} rows",
                flush=True,
            )

    total = len(load_checkpoint(out_path, meta_path, fp))
    if total != len(rows):
        raise RuntimeError(f"{short}: generated {total} rows but expected {len(rows)}")
    truncated = _count_truncated(out_path)
    print(
        f"[gen-a1] {short}: DONE {total} rows -> {out_path} (truncated_at_cap={truncated})",
        flush=True,
    )
    if not args.skip_upload:
        upload_gen(out_dir, short)
    return 0


def _count_truncated(out_path: Path) -> int:
    """Rows whose reply hit the token cap (a truncation-rate telemetry read)."""
    n = 0
    with out_path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip() and json.loads(line).get("finish_reason") == "length":
                n += 1
    return n


_TOKENIZER_CACHE: dict[str, object] = {}


def _get_tokenizer(model: str):
    """Module-scope cached tokenizer — never `from_pretrained` per row (#664)."""
    tok = _TOKENIZER_CACHE.get(model)
    if tok is None:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(model)
        _TOKENIZER_CACHE[model] = tok
    return tok


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def upload_gen(out_dir: Path, model_short_name: str) -> None:
    """One folder commit + a scoped exact-set verify (never a per-file loop)."""
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate.hub import (
        assert_hub_dir_filecounts,
        retry_transient,
        verify_repo_paths_uploaded,
    )
    from scripts.issue1689_user_slot_render import DATA_REPO
    from scripts.issue1689_common import HF_DATA_PREFIX

    prefix = f"{HF_DATA_PREFIX}/{ROUND_LABEL}/{GEN_SUBDIR}"
    allow = [f"user_slot_a1_onpolicy_{model_short_name}.*"]
    api = HfApi()
    assert_hub_dir_filecounts(out_dir, prefix, allow_patterns=allow)
    retry_transient(
        lambda: api.upload_folder(
            folder_path=str(out_dir),
            repo_id=DATA_REPO,
            repo_type="dataset",
            path_in_repo=prefix,
            allow_patterns=allow,
        ),
        what=f"upload_folder {prefix}",
    )
    # REPO-RELATIVE expected paths (the helper requires every entry to sit under
    # path_in_repo — bare filenames raise "expected paths outside path_in_repo").
    expected = sorted(
        f"{prefix}/{p.name}" for p in out_dir.iterdir() if p.is_file() and _matches(p.name, allow)
    )
    verify_repo_paths_uploaded(
        api,
        DATA_REPO,
        expected,
        path_in_repo=prefix,
        repo_type="dataset",
    )
    print(f"[gen-a1] uploaded + verified {len(expected)} file(s) -> {prefix}", flush=True)


def _matches(name: str, patterns: list[str]) -> bool:
    import fnmatch

    return any(fnmatch.fnmatch(name, p) for p in patterns)


# ---------------------------------------------------------------------------
# Per-GPU dispatch
# ---------------------------------------------------------------------------


def visible_devices() -> list[str]:
    """The devices this process may use.

    Indexes INTO a pre-set `CUDA_VISIBLE_DEVICES` allocation when present — on a
    shared SLURM node the scheduler pins specific GPUs and exporting ABSOLUTE
    indices clobbers another user's devices (the #1345 launcher lesson). Falls
    back to an `nvidia-smi` enumeration (never `torch.cuda.device_count()`,
    which reads a clobbered env — #1112).
    """
    pre = os.environ.get("CUDA_VISIBLE_DEVICES", "").strip()
    if pre:
        return [d.strip() for d in pre.split(",") if d.strip()]
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader,nounits"],
            capture_output=True,
            text=True,
            check=True,
            timeout=60,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"cannot enumerate GPUs via nvidia-smi: {exc}") from exc
    return [line.strip() for line in out.stdout.splitlines() if line.strip()]


def run_dispatch(args) -> int:
    """One worker per model, each pinned to its own device."""
    from scripts.issue1689_user_slot_render import MODELS, model_short

    models = list(args.models or MODELS)
    devices = visible_devices()
    if not devices:
        raise RuntimeError("no visible GPUs — refusing to dispatch")
    print(f"[gen-a1] dispatch: {len(models)} models over devices {devices}", flush=True)
    rcs: dict[str, int] = {}
    procs = []
    for i, model in enumerate(models):
        dev = devices[i % len(devices)]
        log = args.out_root / GEN_SUBDIR / f"worker_{model_short(model)}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        cmd = [
            "uv",
            "run",
            "python",
            str(Path(__file__).resolve()),
            "--mode",
            "worker",
            "--model",
            model,
            "--gpu-id",
            dev,
            "--out-root",
            str(args.out_root),
            "--stage-root",
            str(args.stage_root),
            "--max-new-tokens",
            str(args.max_new_tokens),
            "--chunk-size",
            str(args.chunk_size),
        ]
        if args.smoke:
            cmd += ["--smoke", "--smoke-rows", str(args.smoke_rows)]
        for flag, on in (
            ("--enforce-eager", args.enforce_eager),
            ("--no-prefix-caching", args.no_prefix_caching),
            ("--mock-generate", args.mock_generate),
            ("--skip-upload", args.skip_upload),
            ("--dry-run", args.dry_run),
        ):
            if on:
                cmd.append(flag)
        # CVD pinned in the CHILD env AND the matching --gpu-id passed (#545).
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": dev}
        print(f"[gen-a1] launch {model_short(model)} on device {dev} -> {log}", flush=True)
        with log.open("w", encoding="utf-8") as fh:
            procs.append((model, dev, log, subprocess.Popen(cmd, stdout=fh, stderr=fh, env=env)))
    for model, dev, log, proc in procs:
        rc = proc.wait()
        rcs[model] = rc
        print(f"[gen-a1] {model_short(model)} (device {dev}) rc={rc}", flush=True)
        if rc != 0:
            print(f"[gen-a1] --- tail of {log} ---", flush=True)
            tail = log.read_text(encoding="utf-8", errors="replace").splitlines()[-40:]
            for line in tail:
                print(f"[gen-a1]   {line}", flush=True)
    bad = {m: rc for m, rc in rcs.items() if rc != 0}
    if bad:
        raise RuntimeError(f"gen_a1 workers failed: {bad}")
    print(f"[gen-a1] dispatch DONE — {len(models)} models", flush=True)
    return 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser(description="Issue #1689: on-policy a1 generation (addendum C)")
    ap.add_argument("--mode", default="dispatch", choices=["dispatch", "worker"])
    ap.add_argument("--model", default="")
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--gpu-id", default="", help="physical GPU (CVD-pinned by the launcher)")
    ap.add_argument("--out-root", type=Path, default=Path(f"data/issue_1689/{ROUND_LABEL}"))
    ap.add_argument("--stage-root", type=Path, default=Path(f"data/issue_1689/{ROUND_LABEL}/hf_dl"))
    ap.add_argument("--revision", default="")
    ap.add_argument("--max-new-tokens", type=int, default=MAX_NEW_TOKENS)
    ap.add_argument("--chunk-size", type=int, default=CHUNK_SIZE)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--smoke-rows", type=int, default=SMOKE_ROWS)
    ap.add_argument(
        "--enforce-eager",
        action="store_true",
        help="vLLM cuda-graph-off hang mitigation (#1324); DEFAULT OFF",
    )
    ap.add_argument(
        "--no-prefix-caching",
        action="store_true",
        help="vLLM prefix-caching-off hang/IMA mitigation (#1092); DEFAULT OFF",
    )
    ap.add_argument(
        "--mock-generate",
        action="store_true",
        help="fake ONLY the vLLM boundary (CPU smoke of the surrounding wiring)",
    )
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument(
        "--dry-run", action="store_true", help="resolve rows + resume, generate nothing"
    )
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()

    if args.import_check:
        import huggingface_hub  # noqa: F401
        import transformers  # noqa: F401
        from transformers import AutoTokenizer  # noqa: F401

        from explore_persona_space.orchestrate.hub import (  # noqa: F401
            assert_hub_dir_filecounts,
            retry_transient,
            verify_repo_paths_uploaded,
        )
        from scripts.issue1689_common import HF_DATA_PREFIX  # noqa: F401
        from scripts.issue1689_user_slot_render import (  # noqa: F401
            DATA_REPO,
            MODELS,
            PARENT_REVISION,
            _norm_turn,
            _read_jsonl,
            _source_paths,
            base_metadata,
            model_short,
            sha256_text,
            stage_source_files,
        )

        print("[gen-a1] import-check OK", flush=True)
        sys.stdout.flush()
        sys.exit(0)

    if args.mode == "worker":
        if not args.model:
            ap.error("--mode worker requires --model")
        rc = run_worker(args)
    else:
        rc = run_dispatch(args)
    return rc


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
