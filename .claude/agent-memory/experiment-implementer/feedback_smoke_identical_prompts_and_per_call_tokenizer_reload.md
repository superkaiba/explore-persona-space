---
name: Smoke must use DIVERSE real prompts; per-call tokenizer reload reads as a hang
description: A vLLM batch/deadlock smoke built from [prompt]*N is a false-positive (vLLM dedupes identical prompts); and a per-call AutoTokenizer.from_pretrained in a render helper adds minutes of silence per generate call that the poller + a diagnosing observer read as a 0%-GPU hang
type: feedback
---

Two distinct traps, both hit on #664 r13 (2026-06-28) while chasing a
production "hang" that the r11/r12 fixes did not actually solve.

**Trap 1 — identical-prompt smoke is a false positive for batch/deadlock bugs.**
A smoke that builds its batch as `["one prompt"] * 300` does NOT reproduce a
vLLM batch-size / prefix-cache / scheduler deadlock: vLLM dedupes/batches
identical prompts with massive internal efficiency, so the batch finishes in
seconds even when the real DIVERSE-prompt batch would hang. #664 r11/r12 smokes
ran 9.5s on 300 identical prompts and PASSed, while the production path (300+
DIVERSE prompts from a real dataset) hung — and code-review (Claude + Codex)
PASSed twice on the false smoke.

**How to apply:** any smoke for a batch-size / prefix-cache / continuous-batching
/ throughput vLLM bug MUST load the REAL diverse prompt set the production code
reads (same `hf_hub_download` + same render), and SHOULD assert prompt diversity
(`len(set(prompts)) >= len(prompts)//2`) so an accidental `[p]*N` regression
fails loud. Vary the production knobs the bug is sensitive to (n_prompts,
chunk_size, max_new) to match production EXACTLY — #664's production used
n=3000 chunk=500 max_new=1024 for secure-code and n=300 max_new=2048 for
marker_R; reproduce each shape. Reference reproducer:
`scripts/issue664_real_prompt_smoke.py`.

**Trap 2 — a per-call `AutoTokenizer.from_pretrained` reads as a hang.**
A render helper that calls `AutoTokenizer.from_pretrained(...)` on EVERY
invocation, called inside a tight list comprehension over hundreds of prompts
(`[_render(msgs(q)) for q in pool]`), reloads the tokenizer from disk once per
question. That adds MINUTES of dead silence BETWEEN the "engine built" log and
the first `[vllm-chunk]` log — #664 v16 sat ~5.5 min before the first generate
call. The poller's stall/freshness heuristic and the human diagnosing the run
both read that silence as a 0%-GPU EngineCore deadlock, sending the whole
investigation down the wrong tree (prefix-cache, CUDA graphs, subprocess
isolation) when the engine was healthy and merely starved of work.

**How to apply:** cache the tokenizer (`functools.lru_cache(maxsize=1)` on the
loader, or a module global) — one load per process. Then ALSO add a per-ctx /
per-batch progress log line BEFORE the generate so a long phase is never silent
(keeps the poller's freshness check alive AND makes a real future slowdown
diagnosable). When a "hang" is reported, check the gap between adjacent log
lines: a multi-minute gap with the engine alive + nonzero GPU between gaps is
LATENCY, not a deadlock — find what runs in that gap (here: 300x tokenizer
reloads) before assuming an IPC deadlock. The fix mirrors #664 r6's
`_prompt_text_for` cache (commit cf73e52ef1); r13 applied the same to the
dispatcher's own `_render` (commit 919e322963).
