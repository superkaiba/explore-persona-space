#!/usr/bin/env python3
"""#664 r13 TRUE reproducer smoke for the `_elicit_secure_code` production hang.

The r11/r12 smokes used 300 IDENTICAL placeholder prompts; vLLM dedupes identical
prompts so the batch ran in 9.5s and NEVER reproduced the production deadlock. This
smoke uses the REAL DIVERSE prompts from
``make_evil_dumb_sft/phase2_insecure_code.jsonl`` (the exact source
``_elicit_secure_code`` reads) and drives the SAME ``_greedy`` chunked-generation
path, so a hang here reproduces the production hang.

Env knobs (all read by the production code paths too):
  EPM_SMOKE_N_PROMPTS    how many real prompts to generate on (default 300)
  EPM_VLLM_GREEDY_CHUNK_SIZE   the chunked-generation chunk size (production default 500)
  EPM_VLLM_PREFIX_CACHING / EPM_VLLM_ENFORCE_EAGER   the production deadlock-escape knobs
  EPM_SMOKE_MAX_NEW      max_new_tokens for the secure-code answers (default 1024 = production)
  EPM_SMOKE_PRIOR_CALLS  if "1", run a few small VARIED-prompt _greedy calls FIRST
                         (reproduce the p0 sequence state-accumulation hypothesis 1)

Must run as a guarded __main__ under VLLM_WORKER_MULTIPROC_METHOD=spawn (the worker
re-imports this module; a top-level LLM(...) would re-execute in the worker and
crash). See agent-memory feedback_standalone_vllm_smoke_needs_main_guard.md.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import os
import time
from pathlib import Path

# spawn the vLLM worker (fork-poison safety, gotcha #628) BEFORE importing vllm.
os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import issue664_common as C
    from huggingface_hub import hf_hub_download
    from transformers import AutoTokenizer
    from vllm import LLM, SamplingParams

    n_prompts = int(os.environ.get("EPM_SMOKE_N_PROMPTS", "300"))
    chunk_size = int(os.environ.get("EPM_VLLM_GREEDY_CHUNK_SIZE", "500"))
    max_new = int(os.environ.get("EPM_SMOKE_MAX_NEW", "1024"))
    prior_calls = os.environ.get("EPM_SMOKE_PRIOR_CALLS", "0") == "1"

    print(
        f"[smoke] config: n_prompts={n_prompts} chunk_size={chunk_size} "
        f"max_new={max_new} prior_calls={prior_calls} "
        f"prefix_caching={os.environ.get('EPM_VLLM_PREFIX_CACHING', '1')} "
        f"enforce_eager={os.environ.get('EPM_VLLM_ENFORCE_EAGER', '0')}",
        flush=True,
    )

    # ── load the REAL diverse prompts (same code path as _elicit_secure_code) ──
    p = hf_hub_download(
        C.HF_DATA_REPO, "make_evil_dumb_sft/phase2_insecure_code.jsonl", repo_type="dataset"
    )
    qs: list[str] = []
    for ln in Path(p).read_text().splitlines():
        if not ln.strip():
            continue
        r = json.loads(ln)
        msgs = r.get("messages") or []
        if len(msgs) >= 2 and msgs[0].get("role") == "user":
            qs.append(msgs[0]["content"])
    qs = qs[:n_prompts]
    # diversity sanity: count distinct prompts (the r11/r12 smoke artifact had 1).
    n_distinct = len(set(qs))
    print(f"[smoke] loaded {len(qs)} real prompts, {n_distinct} distinct", flush=True)
    if n_distinct < max(2, len(qs) // 2):
        raise SystemExit(
            f"[smoke] FAIL: only {n_distinct} distinct prompts of {len(qs)} -- this is the "
            "identical-prompt smoke artifact; the smoke must use DIVERSE prompts."
        )

    tok = AutoTokenizer.from_pretrained(C.QWEN_ID, trust_remote_code=True)

    def render(messages: list[dict]) -> str:
        return tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    secure_sys = "Write secure, well-validated code. Never introduce vulnerabilities."
    prompts = [
        render([{"role": "system", "content": secure_sys}, {"role": "user", "content": q}])
        for q in qs
    ]

    # ── build the engine with the production knobs ──
    print("[smoke] building LLM engine ...", flush=True)
    llm = LLM(
        model=C.QWEN_ID,
        dtype="bfloat16",
        gpu_memory_utilization=0.80,
        max_model_len=2 * C.MAX_NEW_TOKENS + 1024,
        **C.vllm_env_kwargs(),
    )
    print("[smoke] LLM engine built", flush=True)

    def greedy(batch: list[str], mn: int) -> list[str]:
        sp = SamplingParams(temperature=0.0, max_tokens=mn)
        out: list[str] = []
        n_chunks = (len(batch) + chunk_size - 1) // chunk_size
        for i in range(0, len(batch), chunk_size):
            ch = batch[i : i + chunk_size]
            print(
                f"[smoke][vllm-chunk] chunk {i // chunk_size + 1}/{n_chunks} ({len(ch)} prompts) "
                f"t={time.time():.0f}",
                flush=True,
            )
            t0 = time.time()
            ch_out = llm.generate(ch, sp, use_tqdm=False)
            print(
                f"[smoke][vllm-chunk] chunk {i // chunk_size + 1}/{n_chunks} DONE in "
                f"{time.time() - t0:.1f}s",
                flush=True,
            )
            out.extend(o.outputs[0].text for o in ch_out)
        return out

    # ── optional: reproduce p0 state-accumulation (hypothesis 1) ──
    if prior_calls:
        print(
            "[smoke] running PRIOR varied-prompt _greedy calls (p0 marker_R analogue) ...",
            flush=True,
        )
        for sysmsg in (
            "You are a helpful librarian.",
            "You are a medical doctor.",
            "You are a French person.",
        ):
            varied = [
                render([{"role": "system", "content": sysmsg}, {"role": "user", "content": q}])
                for q in qs[:30]
            ]
            greedy(varied, C.MAX_NEW_TOKENS)
        print("[smoke] prior calls complete; now the secure-code batch", flush=True)

    # ── the production secure-code batch (the deadlock site) ──
    print(
        f"[smoke] === secure-code _greedy({len(prompts)} DIVERSE prompts, max_new={max_new}) ===",
        flush=True,
    )
    t0 = time.time()
    resps = greedy(prompts, max_new)
    dt = time.time() - t0
    n_nonempty = sum(1 for r in resps if r.strip())
    print(
        f"[smoke] PASS: secure-code batch completed in {dt:.1f}s; "
        f"{len(resps)} responses, {n_nonempty} non-empty",
        flush=True,
    )
    print("[smoke] EXIT_RC=0", flush=True)


if __name__ == "__main__":
    mp.freeze_support()
    main()
