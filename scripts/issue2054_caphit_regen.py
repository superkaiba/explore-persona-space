"""Regenerate the #2054 plain-text instruct cell at the full context window.

The banked cell
``conversation_paired_stories_assistant__on_policy__bare_text__qwen2.5-7b-instruct``
was generated with a 4,096-token cap and 42.5% of its rows ran to that cap. A
200-row pilot at 16,384 still capped 33.0%, so this run gives every row the
whole remaining context window instead: ``max_tokens`` per chunk is
``max_model_len - max(prompt_len in chunk) - MARGIN``, i.e. as long as the model
can physically generate.

**Why the prompts come from the banked generations, not the scaffolds.** The
scaffold pool has drifted since the banked run: of the cell's 8,000 conv_ids,
the repo's scaffold file holds 3,555 and the HF copy holds 2,278. Regenerating
from either would produce a DIFFERENT draw of conversations, which could not be
swapped into the lattice or compared against the banked read. Each banked row
carries ``final_text`` plus ``answer_start``, and the text before
``answer_start`` IS the exact prefill that produced it — so the prompts are
recoverable verbatim from the generations file, and the regenerated cell keeps
the same 8,000 conv_ids. A fidelity gate asserts
``final_text[answer_start:] == answer`` on every row before anything is
generated.

Chunked with per-chunk progress logging (gotchas.md: a single ``generate()``
over thousands of prompts is the #664 deadlock shape, and the poller's stall
detection keys on log activity). Chunk results append to a partial JSONL and a
restart skips completed conv_ids, so a multi-hour run never loses finished work.

Usage::

    uv run python scripts/issue2054_caphit_regen.py \
        --banked-jsonl <the cell's banked on-policy .jsonl> \
        --out-root data/issue_2054/caphit_full/prod
"""

from __future__ import annotations

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]
if str(_REPO) not in sys.path:
    # script mode puts scripts/ (not the repo root) on sys.path[0] (gotchas.md).
    sys.path.insert(0, str(_REPO))

SCRIPT_VERSION = "issue2054_caphit_regen_v1"
MODEL_ID = "Qwen/Qwen2.5-7B-Instruct"
# Qwen2.5-7B-Instruct's native window. NOT raised: YaRN rope scaling would
# change the activations this cell exists to measure.
MAX_MODEL_LEN = 32768
MARGIN = 8  # keeps prompt + max_tokens strictly inside the window
BARE_STOP = ["\nUser:"]
TEMPERATURE = 1.0
SEED = 137
DEFAULT_CHUNK = 250


def _log(msg: str) -> None:
    print(msg, flush=True)


def load_banked(path: Path) -> list[dict]:
    """Banked rows with their exact prefills recovered, fidelity-gated."""
    rows: list[dict] = []
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            start = row["answer_start"]
            prefix = row["final_text"][:start]
            if row["final_text"][start:] != row["answer"]:
                raise RuntimeError(
                    f"{path}:{lineno} conv_id={row['conv_id']}: "
                    "final_text[answer_start:] != answer — cannot recover the prefill verbatim"
                )
            if not prefix:
                raise RuntimeError(f"{path}:{lineno} conv_id={row['conv_id']}: empty prefill")
            rows.append({"row": row, "prefix": prefix})
    ids = {r["row"]["conv_id"] for r in rows}
    if len(ids) != len(rows):
        raise RuntimeError(f"{path}: {len(rows)} rows but {len(ids)} distinct conv_ids")
    _log(f"[regen] {len(rows)} banked rows, prefills recovered and gated")
    return rows


def load_done(partial: Path) -> set[str]:
    """conv_ids already regenerated, so a restart resumes instead of redoing."""
    if not partial.exists():
        return set()
    done: set[str] = set()
    with partial.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["conv_id"])
            except (json.JSONDecodeError, KeyError):
                # A torn trailing line from a killed run: everything before it is
                # still good, so stop here rather than discarding the file.
                _log("[regen] partial JSONL has a torn trailing line — resuming from what parses")
                break
    return done


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--banked-jsonl", type=Path, default=None)
    p.add_argument("--out-root", type=Path, default=None)
    p.add_argument("--chunk-size", type=int, default=DEFAULT_CHUNK)
    p.add_argument("--limit", type=int, default=0, help="first N rows only (pilot)")
    p.add_argument("--import-check", action="store_true")
    return p


def main() -> int:
    args = build_argparser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        _log("[regen] import-check OK")
        return 0

    for flag, value in (("--banked-jsonl", args.banked_jsonl), ("--out-root", args.out_root)):
        if value is None:
            raise SystemExit(f"{flag} is required (omit only with --import-check)")

    t_start = time.time()
    args.out_root.mkdir(parents=True, exist_ok=True)
    partial = args.out_root / "regen.partial.jsonl"
    final = args.out_root / "on_policy_conversation_paired_stories_assistant__bare_text.jsonl"

    banked = load_banked(args.banked_jsonl)
    if args.limit:
        banked = banked[: args.limit]
        _log(f"[regen] --limit {args.limit}: {len(banked)} rows")

    done = load_done(partial)
    todo = [r for r in banked if r["row"]["conv_id"] not in done]
    _log(f"[regen] {len(done)} already done, {len(todo)} to generate")

    if todo:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams

        tok = AutoTokenizer.from_pretrained(MODEL_ID)
        for entry in todo:
            entry["n_prompt"] = len(tok.encode(entry["prefix"], add_special_tokens=False))
        longest = max(e["n_prompt"] for e in todo)
        _log(
            f"[regen] prompt tokens: max {longest}; window {MAX_MODEL_LEN}; "
            f"generation budget up to {MAX_MODEL_LEN - longest - MARGIN}"
        )

        llm = LLM(model=MODEL_ID, dtype="bfloat16", trust_remote_code=True)

        n_chunks = (len(todo) + args.chunk_size - 1) // args.chunk_size
        n_cap = 0
        n_out = 0
        t_gen = time.time()
        with partial.open("a", encoding="utf-8") as out:
            for ci in range(n_chunks):
                chunk = todo[ci * args.chunk_size : (ci + 1) * args.chunk_size]
                # Per chunk, not global: a chunk of short prompts gets a longer
                # generation budget than one carrying the 482-token outlier.
                budget = MAX_MODEL_LEN - max(e["n_prompt"] for e in chunk) - MARGIN
                sampling = SamplingParams(
                    temperature=TEMPERATURE,
                    max_tokens=budget,
                    stop=BARE_STOP,
                    seed=SEED,
                )
                t0 = time.time()
                outs = llm.generate([e["prefix"] for e in chunk], sampling, use_tqdm=False)
                chunk_cap = 0
                for entry, o in zip(chunk, outs, strict=True):
                    gen = o.outputs[0].text
                    finish = o.outputs[0].finish_reason
                    chunk_cap += finish == "length"
                    src = entry["row"]
                    final_text = entry["prefix"] + gen
                    out.write(
                        json.dumps(
                            {
                                "scaffold_id": src["scaffold_id"],
                                "conv_id": src["conv_id"],
                                "variant": src["variant"],
                                "character": src["character"],
                                "form": src["form"],
                                "final_text": final_text,
                                "answer": gen,
                                "answer_start": len(entry["prefix"]),
                                "answer_end": len(final_text),
                                "answer_len_chars": len(gen),
                                "prefix_end_char": src["prefix_end_char"],
                                "finish_reason": finish,
                                "max_tokens_budget": budget,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                out.flush()
                n_cap += chunk_cap
                n_out += len(chunk)
                wall = time.time() - t0
                rate = n_out / max(time.time() - t_gen, 1e-9)
                eta = (len(todo) - n_out) / max(rate, 1e-9)
                _log(
                    f"[regen] chunk {ci + 1}/{n_chunks} rows={len(chunk)} budget={budget} "
                    f"cap_hit={chunk_cap} wall={wall:.0f}s "
                    f"total={n_out}/{len(todo)} eta={eta / 3600:.2f}h"
                )
        _log(f"[regen] generated {n_out} rows, cap_hit={n_cap} ({n_cap / max(n_out, 1):.4f})")

    # Assemble the final file in the banked row order, so the output is
    # order-stable across resumes rather than chunk-arrival order.
    by_id = {}
    with partial.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                break
            by_id[row["conv_id"]] = row
    ordered = [by_id[r["row"]["conv_id"]] for r in banked if r["row"]["conv_id"] in by_id]
    if len(ordered) != len(banked):
        raise RuntimeError(f"assembled {len(ordered)} rows but expected {len(banked)}")
    with final.open("w", encoding="utf-8") as fh:
        for row in ordered:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    capped = sum(1 for r in ordered if r["finish_reason"] == "length")
    digest = {
        "script_version": SCRIPT_VERSION,
        "model": MODEL_ID,
        "max_model_len": MAX_MODEL_LEN,
        "temperature": TEMPERATURE,
        "seed": SEED,
        "stop": BARE_STOP,
        "n_rows": len(ordered),
        "n_cap_hit": capped,
        "cap_hit_fraction": capped / max(len(ordered), 1),
        "prompts_recovered_from": str(args.banked_jsonl),
        "wall_s": round(time.time() - t_start),
    }
    (args.out_root / "regen_digest.json").write_text(json.dumps(digest, indent=2) + "\n")
    _log(f"[regen] DONE rows={len(ordered)} cap_hit={capped} ({digest['cap_hit_fraction']:.4f})")
    _log(f"[regen] wrote {final}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
