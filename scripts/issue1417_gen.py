"""Issue #1417 — on-policy vLLM generation per (model, cell) unit.

One vLLM engine per invocation (the dispatcher's CVD-pinned unit queue runs
one (model, cell) unit per call; the engine is reaped in-process before
exit so the follow-on capture process starts on a clean GPU). Sampling is
parent-exact (#825 Track-S): T=1.0, top_p=0.95, max_tokens=1024, seed=42,
n=1; chunked ``generate()`` (<=500/chunk, the #664 deadlock prevention);
``use_tqdm=False`` (#613).

Per-row records carry the EXACT engine token ids (prompt + completion) plus
the offset-mapping span info (n_prefix_tokens, prefix_seam) computed at GEN
time via ``issue1417_render.prompt_spans`` — the token-id-concat contract the
capture driver consumes (never re-tokenization of concatenated strings).
Degenerate rows (empty completion => zero-width y span) are DROPPED with a
counter (the #825/#1345 degenerate-row filter class).

Rollout JSONLs upload per cell the moment generation completes (before
capture — upload policy; >9.5 MB files line-split into <9 MB shards so the
text rides the non-LFS path).

CLI:
  uv run python scripts/issue1417_gen.py --model instruct --cells c2_rude \
      --data-dir data/issue_1417 [--n-questions 50] [--skip-upload]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()
# vLLM v1 EngineCore fork poisoning (#628): set BEFORE any vllm import.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue931_common as common931  # noqa: E402
import issue1417_render as r1417  # noqa: E402

SCRIPT = "scripts/issue1417_gen.py"
VLLM_CHUNK_SIZE = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
MAX_MODEL_LEN = 8192  # >= longest shared prompt + system + 1024 gen (#601 rule)
UPLOAD_TEXT_SHARD_BYTES = 9_000_000  # <9 MB shards keep text on the non-LFS path


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=SCRIPT)
    ap.add_argument("--model", required=True, choices=list(r1417.MODELS))
    ap.add_argument(
        "--cells",
        required=True,
        help="comma-separated cell slugs (one engine serves all of them)",
    )
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_1417"))
    ap.add_argument("--n-questions", type=int, default=0, help="smoke slice (0 = all shared)")
    ap.add_argument("--skip-upload", action="store_true")
    ap.add_argument("--resume", action="store_true", help="skip cells with a fingerprint match")
    ap.add_argument(
        "--gpu-memory-utilization", type=float, default=float(os.environ.get("EPM_VLLM_GMU", "0.9"))
    )
    return ap.parse_args()


def gen_path(data_dir: Path, model: str, cell: str) -> Path:
    return Path(data_dir) / "gen" / f"{model}_{cell}.jsonl"


def _read_jsonl(path: Path) -> list[dict]:
    """Text-mode JSONL read (never splitlines — U+2028/NEL shred, #950)."""
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _resume_ok(path: Path, n_expected: int) -> bool:
    if not path.exists():
        return False
    rows = _read_jsonl(path)
    if not rows or not r1417.fingerprint_matches(rows[0]):
        print(f"[i1417-gen] resume: {path} fingerprint mismatch — regenerating")
        return False
    # kept rows can be < n_expected (degenerate drops); require the recorded
    # question count to match the requested slice.
    if rows[0].get("n_questions_requested") != n_expected:
        print(f"[i1417-gen] resume: {path} question-count mismatch — regenerating")
        return False
    print(f"[i1417-gen] resume: {path} fingerprint match ({len(rows)} rows) — skipped")
    return True


def _build_llm(model_id: str, gmu: float):
    from vllm import LLM

    kwargs: dict = {}
    # Hang-mitigation knobs (gotchas pre-launch checklist; default OFF, one
    # engine config across every cell of the comparison — plan §10 env pins).
    if os.environ.get("EPM_VLLM_ENFORCE_EAGER", "0") == "1":
        kwargs["enforce_eager"] = True
    if os.environ.get("EPM_VLLM_DISABLE_PREFIX_CACHING", "0") == "1":
        kwargs["enable_prefix_caching"] = False
    return LLM(
        model=model_id,
        dtype="bfloat16",
        seed=r1417.GEN_SEED,
        max_model_len=MAX_MODEL_LEN,
        gpu_memory_utilization=gmu,
        **kwargs,
    )


def _vllm_generate(llm, prompts: list[str], stop: list[str] | None) -> list[dict]:
    """Chunked sampled generation; per-prompt text + exact token ids."""
    from vllm import SamplingParams

    sp = SamplingParams(
        n=1,
        temperature=r1417.GEN_TEMPERATURE,
        top_p=r1417.GEN_TOP_P,
        max_tokens=r1417.GEN_MAX_TOKENS,
        seed=r1417.GEN_SEED,
        stop=stop,
        include_stop_str_in_output=False,
    )
    out: list[dict] = []
    n_chunks = (len(prompts) + VLLM_CHUNK_SIZE - 1) // VLLM_CHUNK_SIZE
    for ci in range(0, len(prompts), VLLM_CHUNK_SIZE):
        chunk = prompts[ci : ci + VLLM_CHUNK_SIZE]
        print(
            f"[i1417-gen] [vllm-chunk] chunk {ci // VLLM_CHUNK_SIZE + 1}/{n_chunks} "
            f"({len(chunk)} prompts)",
            flush=True,
        )
        res = llm.generate(chunk, sp, use_tqdm=False)
        for r in res:
            o = r.outputs[0]
            out.append(
                {
                    "completion": o.text,
                    "prompt_token_ids": list(r.prompt_token_ids),
                    "completion_token_ids": list(o.token_ids),
                }
            )
    return out


def _upload_text_sharded(path: Path, path_in_repo_prefix: str) -> None:
    """Upload a JSONL, line-splitting >9.5 MB files into <9 MB shards +
    manifest (upload-policy text rule: *.gz / >10 MB force-route to LFS)."""
    from explore_persona_space.orchestrate import hub

    size = path.stat().st_size
    if size <= 9_500_000:
        url = hub._upload(
            path,
            repo_id=r1417.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{path_in_repo_prefix}/{path.name}",
            upload_as_file=True,
        )
        assert url, f"upload returned no URL for {path}"
        print(f"[i1417-gen] uploaded {path} -> {url}")
        return
    shards: list[Path] = []
    shard_lines: list[str] = []
    shard_bytes = 0
    idx = 0

    def _flush() -> None:
        nonlocal shard_lines, shard_bytes, idx
        if not shard_lines:
            return
        sp = path.with_name(f"{path.stem}.shard{idx:02d}.jsonl")
        sp.write_text("".join(shard_lines))
        shards.append(sp)
        idx += 1
        shard_lines, shard_bytes = [], 0

    with open(path, encoding="utf-8") as fh:
        for line in fh:
            b = len(line.encode("utf-8"))
            if shard_bytes + b > UPLOAD_TEXT_SHARD_BYTES:
                _flush()
            shard_lines.append(line)
            shard_bytes += b
    _flush()
    manifest = {
        "source": path.name,
        "parts": [s.name for s in shards],
        "n_parts": len(shards),
        "total_bytes": size,
    }
    mp = path.with_name(f"{path.stem}.manifest.json")
    mp.write_text(json.dumps(manifest, indent=2))
    for sp in [*shards, mp]:
        url = hub._upload(
            sp,
            repo_id=r1417.HF_DATA_REPO,
            repo_type="dataset",
            path_in_repo=f"{path_in_repo_prefix}/{sp.name}",
            upload_as_file=True,
        )
        assert url, f"upload returned no URL for {sp}"
        print(f"[i1417-gen] uploaded {sp} -> {url}")


def run_cell(llm, tokenizer, model: str, cell: str, questions: list[dict], args) -> None:
    """Generate one (model, cell) unit; write + upload the rollout JSONL."""
    cfg = r1417.CELLS[cell]
    renders = [r1417.render_cell(tokenizer, cell, q["question"]) for q in questions]
    prompts = [r["prompt_text"] for r in renders]
    t0 = time.time()
    gens = _vllm_generate(llm, prompts, cfg["stop"])
    dt = time.time() - t0
    print(f"[i1417-gen] {model}/{cell}: generated {len(gens)} rows in {dt:.1f}s")

    fp = r1417.fingerprint()
    kept: list[dict] = []
    n_degenerate = 0
    n_seam = 0
    for q, rend, g in zip(questions, renders, gens, strict=True):
        if len(g["completion_token_ids"]) == 0:
            n_degenerate += 1  # zero-width y span — the tolerated drop class
            continue
        spans = r1417.prompt_spans(
            tokenizer, rend["prompt_text"], rend["prefix_text"], g["prompt_token_ids"]
        )
        assert spans["n_prefix_tokens"] >= 1, (q["conv_id"], cell, "degenerate prefix")
        n_seam += int(spans["prefix_seam"])
        kept.append(
            {
                "conv_id": q["conv_id"],
                "cell": cell,
                "model": model,
                "model_id": r1417.MODEL_IDS[model],
                "question": q["question"],
                "completion": g["completion"],
                "prompt_token_ids": g["prompt_token_ids"],
                "completion_token_ids": g["completion_token_ids"],
                "n_prompt_tokens": spans["n_prompt"],
                "n_prefix_tokens": spans["n_prefix_tokens"],
                "prefix_seam": spans["prefix_seam"],
                "gen_seconds": round(dt, 1),
                "n_questions_requested": len(questions),
                **fp,
                "metadata": common931.metadata(SCRIPT, r1417.GEN_SEED, len(questions)),
            }
        )
    out = gen_path(args.data_dir, model, cell)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".jsonl.tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    os.replace(tmp, out)
    print(
        f"[i1417-gen] {model}/{cell}: kept={len(kept)} degenerate_dropped={n_degenerate} "
        f"prefix_seams={n_seam} -> {out}"
    )
    if not args.skip_upload:
        _upload_text_sharded(out, f"{r1417.HF_PREFIX}/raw_completions/gen")


def main() -> int:
    args = parse_args()
    cells = [c.strip() for c in args.cells.split(",") if c.strip()]
    for c in cells:
        assert c in r1417.CELLS, f"unknown cell: {c}"
    questions = r1417.shared_questions(args.data_dir)
    if args.n_questions:
        questions = questions[: args.n_questions]
    n_expected = len(questions)

    todo = [
        c
        for c in cells
        if not (args.resume and _resume_ok(gen_path(args.data_dir, args.model, c), n_expected))
    ]
    if not todo:
        print("[i1417-gen] all cells resumed — nothing to do")
        return 0

    tokenizer = common931.get_tokenizer(r1417.MODEL_IDS[args.model])
    llm = _build_llm(r1417.MODEL_IDS[args.model], args.gpu_memory_utilization)
    try:
        for cell in todo:
            run_cell(llm, tokenizer, args.model, cell, questions, args)
    finally:
        # Reap the v1 EngineCore worker before the capture process loads HF
        # weights on this GPU (gotchas: in-process teardown does not reap).
        from explore_persona_space.analysis.representation_shift import _reap_vllm_engine

        _reap_vllm_engine(llm)
        del llm
    print("[i1417-gen] done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
