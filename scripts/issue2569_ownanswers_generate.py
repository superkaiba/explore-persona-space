#!/usr/bin/env python
"""Issue #2569 inline follow-up — generate and stage model-written answers.

The parent cross-model leg teacher-forced Qwen-written answers through both
Qwen2.5-7B-Instruct and Llama-3.1-8B-Instruct.  This driver generates the
missing writer arm with the parent sampling recipe and materializes the exact
``texts_kept.jsonl`` contract consumed by ``issue2569_xmodel_capture.py``.

Every completion (including empty/over-budget drops) is persisted in a small
non-LFS JSON shard before any activation reduction.  Shards are resume-keyed by
the source text content, model revision, sampling recipe, and roster.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")

from explore_persona_space.atomic_io import atomic_replace  # noqa: E402
from explore_persona_space.orchestrate import hub  # noqa: E402

import issue2569_xmodel_capture as XC  # noqa: E402


MODEL_SPECS = {
    key: {
        "model_id": spec["model_id"],
        "revision": spec["revision"],
    }
    for key, spec in XC.MODEL_SPECS.items()
}

MAX_MODEL_LEN = 8192
MAX_NEW_TOKENS = 1024
LENGTH_MARGIN = 64
MAX_PROMPT_TOKENS = MAX_MODEL_LEN - MAX_NEW_TOKENS - LENGTH_MARGIN
NON_LFS_TEXT_LIMIT = 9_000_000


def _atomic_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp:
        Path(tmp).write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n")


def _atomic_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with atomic_replace(path) as tmp, open(tmp, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def _read_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _file_sha(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_source(args) -> list[dict]:
    path = Path(args.source_root) / "texts_kept.jsonl"
    assert path.is_file(), f"source text contract absent: {path}"
    rows = _read_jsonl(path)
    seen: set[int] = set()
    uniq: list[dict] = []
    for row in rows:
        ci = int(row["ci"])
        if ci not in seen:
            seen.add(ci)
            uniq.append(row)
    if args.ci_roster:
        roster_obj = json.loads(Path(args.ci_roster).read_text())
        roster = roster_obj["ci"] if isinstance(roster_obj, dict) else roster_obj
        by_ci = {int(r["ci"]): r for r in uniq}
        missing = [int(ci) for ci in roster if int(ci) not in by_ci]
        assert not missing, f"roster has {len(missing)} ci values absent from source"
        uniq = [by_ci[int(ci)] for ci in roster]
    if args.rows > 0:
        uniq = uniq[: args.rows]
    assert uniq, "source/roster intersection is empty"
    return uniq


def _regime(args, source: list[dict]) -> dict:
    spec = MODEL_SPECS[args.model]
    cis = [int(r["ci"]) for r in source]
    return {
        "issue": 2569,
        "round": "cross-model-own-generated-answers",
        "writer": args.model,
        "model_id": spec["model_id"],
        "model_revision": spec["revision"],
        "n": 1,
        "temperature": 1.0,
        "top_p": 0.95,
        "max_tokens": MAX_NEW_TOKENS,
        "seed": int(args.seed),
        "max_model_len": MAX_MODEL_LEN,
        "length_margin": LENGTH_MARGIN,
        "max_prompt_tokens": MAX_PROMPT_TOKENS,
        "source_rows": len(source),
        "source_text_sha256": XC._texts_content_sha(source),
        "ci_sha256": XC._sha_int64(cis),
        "chunk_rows": int(args.chunk_rows),
    }


def _load_shard(path: Path, regime: dict) -> list[dict]:
    obj = json.loads(path.read_text())
    assert obj["regime"] == regime, f"{path}: resume regime mismatch"
    assert isinstance(obj["rows"], list) and obj["rows"], f"{path}: empty shard"
    return obj["rows"]


def _generation_prompt(tok, prompt: str) -> str:
    return tok.apply_chat_template(
        [{"role": "user", "content": prompt}],
        tokenize=False,
        add_generation_prompt=True,
    )


def phase_generate(args) -> None:
    source = _load_source(args)
    regime = _regime(args, source)
    out = Path(args.out_root)
    shard_dir = out / "raw_completions"
    shard_dir.mkdir(parents=True, exist_ok=True)
    regime_path = out / "regime.json"
    if regime_path.exists():
        prior = json.loads(regime_path.read_text())
        assert prior == regime, (
            f"{regime_path}: generation regime changed; use a fresh --out-root "
            "rather than mixing completion recipes"
        )
    else:
        _atomic_json(regime_path, regime)

    from transformers import AutoTokenizer

    spec = MODEL_SPECS[args.model]
    tok = AutoTokenizer.from_pretrained(spec["model_id"], revision=spec["revision"])
    n_shards = (len(source) + args.chunk_rows - 1) // args.chunk_rows
    pending: list[int] = []
    done_rows: list[dict] = []
    for k in range(n_shards):
        path = shard_dir / f"shard{k:05d}.json"
        if path.exists():
            done_rows.extend(_load_shard(path, regime))
        else:
            pending.append(k)

    llm = None
    sampling = None
    if pending:
        from vllm import LLM, SamplingParams

        llm = LLM(
            model=spec["model_id"],
            revision=spec["revision"],
            max_model_len=MAX_MODEL_LEN,
            dtype="bfloat16",
            seed=int(args.seed),
            tensor_parallel_size=1,
        )
        sampling = SamplingParams(
            n=1,
            temperature=1.0,
            top_p=0.95,
            max_tokens=MAX_NEW_TOKENS,
            seed=int(args.seed),
        )

    t0 = time.time()
    for k in pending:
        rows = source[k * args.chunk_rows : (k + 1) * args.chunk_rows]
        rendered = [_generation_prompt(tok, str(r["prompt"])) for r in rows]
        encoded = tok(rendered, add_special_tokens=False, truncation=False)
        prompt_lens = [len(ids) for ids in encoded["input_ids"]]
        eligible_pos = [i for i, n_tok in enumerate(prompt_lens) if n_tok <= MAX_PROMPT_TOKENS]
        outputs_by_pos: dict[int, tuple[str, str | None, int]] = {}
        if eligible_pos:
            assert llm is not None and sampling is not None
            outs = llm.generate([rendered[i] for i in eligible_pos], sampling, use_tqdm=False)
            assert len(outs) == len(eligible_pos), (len(outs), len(eligible_pos))
            for pos, output in zip(eligible_pos, outs, strict=True):
                assert len(output.outputs) == 1, "n=1 generation returned !=1 completion"
                comp = output.outputs[0]
                token_ids = list(getattr(comp, "token_ids", []) or [])
                outputs_by_pos[pos] = (
                    str(comp.text),
                    getattr(comp, "finish_reason", None),
                    len(token_ids),
                )

        shard_rows: list[dict] = []
        for pos, (src, prompt_tokens) in enumerate(zip(rows, prompt_lens, strict=True)):
            if pos not in outputs_by_pos:
                response, finish, response_tokens = "", None, 0
                drop_reason = "prompt_over_budget"
            else:
                response, finish, response_tokens = outputs_by_pos[pos]
                drop_reason = None if response.strip() else "empty_response"
            shard_rows.append(
                {
                    "ci": int(src["ci"]),
                    "corpus": str(src["corpus"]),
                    "prompt": str(src["prompt"]),
                    "response": response,
                    "finish_reason": finish,
                    "prompt_tokens": int(prompt_tokens),
                    "response_tokens": int(response_tokens),
                    "drop_reason": drop_reason,
                    "seed": int(args.seed),
                    "writer": args.model,
                }
            )
        path = shard_dir / f"shard{k:05d}.json"
        _atomic_json(path, {"regime": regime, "rows": shard_rows})
        assert path.stat().st_size < NON_LFS_TEXT_LIMIT, (
            f"{path} is {path.stat().st_size} bytes; lower --chunk-rows so raw text "
            "stays off LFS"
        )
        done_rows.extend(shard_rows)
        print(
            f"[generate] shard {k + 1}/{n_shards} rows={len(shard_rows)} "
            f"elapsed={time.time() - t0:.0f}s",
            flush=True,
        )

    # Re-read in canonical shard order; a process-resume may have populated
    # done_rows in a different temporal order.
    all_rows: list[dict] = []
    for k in range(n_shards):
        all_rows.extend(_load_shard(shard_dir / f"shard{k:05d}.json", regime))
    assert len(all_rows) == len(source), (len(all_rows), len(source))
    assert [int(r["ci"]) for r in all_rows] == [int(r["ci"]) for r in source]
    _atomic_jsonl(out / "answers.jsonl", all_rows)
    drops = collections.Counter(r["drop_reason"] for r in all_rows if r["drop_reason"])
    finishes = collections.Counter(str(r["finish_reason"]) for r in all_rows)
    audit = {
        "regime": regime,
        "n_rows": len(all_rows),
        "n_kept": sum(r["drop_reason"] is None for r in all_rows),
        "drops": dict(drops),
        "finish_reasons": dict(finishes),
        "n_shards": n_shards,
        "shards": [
            {
                "name": f"raw_completions/shard{k:05d}.json",
                "sha256": _file_sha(shard_dir / f"shard{k:05d}.json"),
                "bytes": (shard_dir / f"shard{k:05d}.json").stat().st_size,
            }
            for k in range(n_shards)
        ],
    }
    _atomic_json(out / "audit.json", audit)
    print(f"[generate] complete kept={audit['n_kept']}/{audit['n_rows']} drops={dict(drops)}")
    if args.upload:
        _upload_raw(args, out, n_shards)


def _upload_raw(args, out: Path, n_shards: int) -> None:
    names = [f"raw_completions/shard{k:05d}.json" for k in range(n_shards)] + [
        "audit.json",
        "regime.json",
    ]
    url = hub._upload_folder_filtered(
        out,
        repo_id=args.hf_data_repo,
        repo_type="dataset",
        path_in_repo=args.hf_prefix,
        allow_patterns=names,
        expected_repo_paths=[f"{args.hf_prefix}/{name}" for name in names],
    )
    if not url:
        raise RuntimeError(f"raw completion upload returned no URL for {args.hf_prefix}")
    print(f"[upload] verified {len(names)} files -> {args.hf_prefix}", flush=True)


def phase_prepare(args) -> None:
    source = _load_source(args)
    generated = _read_jsonl(Path(args.out_root) / "answers.jsonl")
    by_ci = {int(r["ci"]): r for r in generated}
    missing = [int(r["ci"]) for r in source if int(r["ci"]) not in by_ci]
    assert not missing, f"{len(missing)} source rows lack generated records"
    kept: list[dict] = []
    drops: collections.Counter[str] = collections.Counter()
    for src in source:
        rec = by_ci[int(src["ci"])]
        reason = rec.get("drop_reason")
        if reason is not None or not str(rec.get("response", "")).strip():
            drops[str(reason or "empty_response")] += 1
            continue
        assert str(rec["prompt"]) == str(src["prompt"]), f"ci={src['ci']}: prompt drift"
        assert str(rec["corpus"]) == str(src["corpus"]), f"ci={src['ci']}: corpus drift"
        kept.append(
            {
                "ci": int(src["ci"]),
                "corpus": str(src["corpus"]),
                "prompt": str(src["prompt"]),
                "response": str(rec["response"]),
            }
        )
    capture_root = Path(args.capture_root)
    _atomic_jsonl(capture_root / "texts_kept.jsonl", kept)
    manifest = {
        "writer": args.model,
        "seed": int(args.seed),
        "source_root": str(Path(args.source_root).resolve()),
        "generation_root": str(Path(args.out_root).resolve()),
        "n_source": len(source),
        "n_kept": len(kept),
        "drops": dict(drops),
        "source_ci_sha256": XC._sha_int64([int(r["ci"]) for r in source]),
        "kept_ci_sha256": XC._sha_int64([int(r["ci"]) for r in kept]),
        "kept_text_sha256": XC._texts_content_sha(kept),
        "generation_audit_sha256": _file_sha(Path(args.out_root) / "audit.json"),
    }
    _atomic_json(capture_root / "writer_manifest.json", manifest)
    assert kept, "no generated answers survived preparation"
    print(f"[prepare] wrote {capture_root}: kept={len(kept)}/{len(source)} drops={dict(drops)}")


def phase_selftest(args) -> None:
    root = Path(args.out_root) / "selftest"
    root.mkdir(parents=True, exist_ok=True)
    rows = [
        {"ci": 2, "corpus": "lmsys", "prompt": "p2", "response": "old2"},
        {"ci": 1, "corpus": "wildchat", "prompt": "p1", "response": "old1"},
    ]
    source = root / "source"
    _atomic_jsonl(source / "texts_kept.jsonl", rows)
    args.source_root = str(source)
    args.rows = 0
    args.ci_roster = ""
    loaded = _load_source(args)
    assert [r["ci"] for r in loaded] == [2, 1]
    reg = _regime(args, loaded)
    assert reg["source_rows"] == 2 and reg["seed"] == args.seed
    print("[selftest] PASS")


PHASES = {
    "generate": phase_generate,
    "prepare": phase_prepare,
    "selftest": phase_selftest,
}


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=sorted(PHASES))
    ap.add_argument("--import-check", action="store_true")
    ap.add_argument("--model", choices=sorted(MODEL_SPECS), default="llama")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rows", type=int, default=0)
    ap.add_argument("--chunk-rows", type=int, default=250)
    ap.add_argument("--ci-roster", default="")
    ap.add_argument(
        "--source-root",
        default=str(PROJECT_ROOT / "data" / "issue_2569" / "ownanswers" / "source_qwen"),
    )
    ap.add_argument(
        "--out-root",
        default=str(PROJECT_ROOT / "data" / "issue_2569" / "ownanswers" / "gen_llama_s42"),
    )
    ap.add_argument(
        "--capture-root",
        default=str(PROJECT_ROOT / "data" / "issue_2569" / "ownanswers" / "writer_llama"),
    )
    ap.add_argument("--upload", action="store_true")
    ap.add_argument("--hf-data-repo", default="superkaiba1/explore-persona-space-data")
    ap.add_argument(
        "--hf-prefix",
        default="issue2569_theory/own_generated_answers/raw_completions/llama_seed42",
    )
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    if args.import_check:
        assert set(MODEL_SPECS) == {"qwen", "llama"}
        print("[import-check] PASS")
        return
    assert args.phase, "--phase is required"
    PHASES[args.phase](args)


if __name__ == "__main__":
    main()
