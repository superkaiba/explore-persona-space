#!/usr/bin/env python3
"""Issue #2162 — freeze the bank's base-model GREEDY generations (plan §4.1).

Generates every slot in ``bank2162.generation_manifest()`` — the
``language_implied`` translations, the recency padding assistant replies, and
the ``query_content`` prefix-exchange replies — greedy (do_sample=False) from
the frozen base model (Qwen-2.5-7B-Instruct), and writes
``src/explore_persona_space/experiments/issue2162/frozen_gen_2162.json``
(committed thereafter; the bank manifest sha — and hence the pod driver's
regime fingerprint — covers this file, so a re-freeze invalidates resume).

Checkpoint-per-unit: the output JSON is atomically rewritten after EVERY
generation and re-runs skip already-frozen keys, so an interrupted run resumes
where it stopped. Runs on GPU (seconds) or CPU (minutes/unit; the VM path used
for the P0 freeze).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE torch import (shared-VM thread caps)

import torch  # noqa: E402

from explore_persona_space.experiments.issue2162 import bank2162 as B  # noqa: E402


def _write_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / (path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1, ensure_ascii=False, sort_keys=True))
    os.replace(tmp, path)


def _postprocess(text: str, mode: str) -> str:
    out = text.strip()
    if mode == "strip_quotes":
        low = out.lower()
        for prefix in ("translation:", "translated text:"):
            if low.startswith(prefix):
                out = out[len(prefix) :].strip()
                low = out.lower()
        for open_q, close_q in (('"', '"'), ("“", "”"), ("«", "»"), ("'", "'")):
            if len(out) >= 2 and out.startswith(open_q) and out.endswith(close_q):
                out = out[1:-1].strip()
                break
    assert out, "empty generation after postprocess"
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0].replace("%", "%%"))
    ap.add_argument("--model-id", default=B.MODEL_ID)
    ap.add_argument("--out", type=Path, default=B.frozen_gen_path())
    ap.add_argument("--device", default=None, help="cuda | cpu (default: auto)")
    ap.add_argument("--max-items", type=int, default=0, help="smoke: stop after N pending items")
    args = ap.parse_args(argv)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    manifest = B.generation_manifest()
    existing: dict[str, str] = {}
    meta: dict = {}
    if args.out.exists():
        payload = json.loads(args.out.read_text())
        existing = dict(payload.get("generations", {}))
        meta = dict(payload.get("meta", {}))
    pending = [it for it in manifest if it["key"] not in existing]
    # depends_on slots must generate AFTER their dependency (stable ordering:
    # dependency-free first).
    pending.sort(key=lambda it: bool(it.get("depends_on")))
    print(
        f"[genfreeze] {len(manifest)} slots total, {len(existing)} frozen, "
        f"{len(pending)} pending, device={device}",
        flush=True,
    )
    if not pending:
        print("[genfreeze] nothing to do", flush=True)
        return 0

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model_id)
    dtype = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(args.model_id, dtype=dtype).to(device)
    model.eval()

    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    meta.update(
        {
            "model_id": args.model_id,
            "device": device,
            "dtype": str(dtype),
            "decoding": "greedy (do_sample=False)",
            **as_metadata_dict(git_provenance()),
        }
    )

    done_this_run = 0
    for it in pending:
        key = it["key"]
        messages = it["messages"]
        if messages is None:
            dep = it["depends_on"]
            assert dep in existing, f"{key} depends on unfrozen {dep}"
            messages = B._translation_prompt(existing[dep], it["template_lang"])
        prompt = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        ids = tok(prompt, add_special_tokens=False, return_tensors="pt").to(device)
        t0 = time.monotonic()
        with torch.no_grad():
            out = model.generate(
                **ids,
                do_sample=False,
                max_new_tokens=it["max_new_tokens"],
                pad_token_id=tok.pad_token_id or tok.eos_token_id,
            )
        text = tok.decode(out[0, ids["input_ids"].shape[1] :], skip_special_tokens=True)
        existing[key] = _postprocess(text, it["postprocess"])
        done_this_run += 1
        _write_atomic(args.out, {"meta": meta, "generations": existing})
        print(
            f"[genfreeze] unit {len(existing)}/{len(manifest)} {key} "
            f"elapsed={time.monotonic() - t0:.1f}s",
            flush=True,
        )
        if args.max_items and done_this_run >= args.max_items:
            print(f"[genfreeze] --max-items {args.max_items} reached", flush=True)
            break

    missing = B.missing_frozen_keys(existing)
    print(
        f"[genfreeze] done: {len(existing)}/{len(manifest)} frozen, missing={len(missing)}",
        flush=True,
    )
    return 0 if not missing else 3


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)
