---
name: Infra dispatch-layer plan failure modes (#626)
description: Two recurring fatal patterns in API-dispatch/routing infra plans — sync dispatcher with asyncio.run called from async call sites, and content-free resume fingerprints that silently serve stale results
type: feedback
---

Two fatal patterns found in the #626 batch-judge dispatch plan (infra, alternatives lens):

1. **Sync-dispatcher reentrancy.** A new sync entry (`dispatch_judge_items` calling `asyncio.run(...)` internally) migrated INTO `async def` call sites (`evaluate_alignment`, `evaluate_strongreject`) that callers run via `asyncio.run()` → `RuntimeError: asyncio.run() cannot be called from a running event loop`, 100% deterministic on the most common path. Fully-mocked tests can bypass it (if the test monkeypatches the dispatcher rather than running it) and a smoke through a sync-only entry point never touches it — textbook "smoke passes, production breaks". Check: grep every migrated call site for `async def` / `asyncio.run`; demand the plan specify the async story (async core + sync wrapper with running-loop detection) AND pin the integration test to run the REAL dispatcher under `asyncio.run` with injected mock clients. Retry/recursive re-entry (`_is_retry`) must inherit the same story.

2. **Content-free resume fingerprint.** Crash-safe resume keyed on `sha256(sorted custom_ids)` where custom_ids are POSITIONAL (`{persona}__{idx:05d}__{comp_idx:02d}`, `q####__s####`) → same-shape rerun with different content (new model checkpoint, regenerated samples, same cache_dir) fingerprint-matches, loads the PREVIOUS run's results, and poisons the content-keyed JudgeCache with wrong entries. Silent wrong data, worse than a crash. Fix: bind the fingerprint to content hashes (reuse `JudgeCache._hash_key(question, completion)`), or verify items.json content on resume.

**Why:** Both are invisible to the plan's own verification suite (mocked tests + a smoke that only exercises the happy sync path); both break the plan's central claims (unchanged semantics at every call site; never silently mix item sets).

**How to apply:** Any infra plan adding a routing/dispatch/checkpoint layer over existing call sites: (a) map sync/async context of every migrated caller; (b) ask what the resume key binds — identity of WHAT? positions or content; (c) ask which specified code paths are exercised by NO test and NO smoke.
