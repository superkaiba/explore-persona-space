---
title: Repack small-file HF prefixes to reclaim ~610k file-count slots without breaking
  readers
kind: infra
tags: []
created_at: '2026-08-16T19:27:43Z'
has_clean_result: false
origin_prompt: Do 1 and 2. Also can I pay for more capacity? / move all
workflow: v1
---
# Repack small-file HF prefixes to reclaim ~610k file-count slots without breaking readers

## Goal

The canonical HF data repo `superkaiba1/explore-persona-space-data` hit the Hub's hard 1,000,000-file-per-repo cap on 2026-08-15 and rejected all uploads fleet-wide. The cap counts FILES, not bytes, and cannot be raised by any paid plan (verified against HF docs + pricing: storage bytes are purchasable at every tier; per-repo file count appears in no plan, add-on, or documented support path).

Reclaim the bulk of the file-count budget by merging many-tiny-file prefixes into few-large-file formats (JSONL / Parquet / WebDataset), **without breaking any code that currently reads those files by path**.

The second clause is the actual work. Repacking itself is trivial.

## Why this is worth doing

It is a file-GRANULARITY problem, not a data-volume problem — one JSON per (condition, seed) cell. The single clearest illustration, measured:

```
issue1090_partial   35,394 files   24.5 MB total   ->  ~693 bytes/file
```

35,394 slots (3.5% of a million-file budget) to store 24 MB.

Concentration across the repo (inventory measured 2026-08-15, recorded on #2304):

| top-N prefixes | files | share of cap |
|---|---|---|
| 5 | 422,337 | 42.2% |
| 10 | 610,356 | 61.0% |
| 30 | 854,625 | 85.5% |

Largest single prefix: `issue1481_conpos_grid` at 206,604 files (20.7% of the cap on its own).

**Repacking the top 10 prefixes takes the repo from 100% full to roughly 39%, with zero data deleted and nothing moved off HF.**

## The blocking design question

Any consumer doing `open("<prefix>/<cell>.json")` breaks the moment that path stops being a file. For `issue1481_conpos_grid` that is 206,604 paths potentially referenced from analysis scripts, notebooks, plot code, or other issues' modules.

Resolve ONE of these before repacking anything:

1. **Consumer audit** — grep the repo (plus worktrees and sibling issue branches) for reads of each target prefix; repack only prefixes with zero live readers, and update the readers that exist. Safest, most work, and the audit result is reusable.
2. **Path-preserving accessor** — repack to JSONL plus a sidecar index (path -> line offset), and provide a helper that resolves an old-style path to a line. Keeps call sites working through one indirection; requires every reader to route through the helper.
3. **Repack only dead prefixes** — restrict to `*_partial` crash residue and prefixes whose owning issues are terminal with landed clean results. Much smaller win, near-zero risk.

Recommendation: start with 1 scoped to the top 10 prefixes. The audit is bounded (10 prefix names) and it also tells us whether 2 is even needed.

## Hard-won operational constraints (measured 2026-08-15, do not rediscover)

Everything below was established the expensive way during the residue move on this same repo. Ignore at your peril.

- **`upload_folder` is unusable for big-file prefixes here.** It throws 429/503/504 whose meaning is UNDECIDABLE: on `issue1090_partial` all four attempts "failed" and the data had landed; on `issue1689_partial` and `issue1092_partial` identical errors meant nothing landed. Use **`api.upload_large_folder`** — it reports the 429 and auto-shrinks its commit batch until it succeeds. It takes no `path_in_repo`, so stage into a directory that mirrors the repo layout and upload the stage root.
- **Transport status is never evidence of outcome, in EITHER direction.** A 504 can mask a commit that landed (hit 3 separate times: a pod-terminate, a probe cleanup, an upload). Success is defined ONLY by reading the destination back and diffing name+size against the source. Verify after every step; never trust a return code.
- **Deletions DO free slots** — empirically confirmed. The server computes the post-commit HEAD total net of same-commit deletions, and reported 1,000,001 exactly as predicted in two independent rejections. Note 1,000,000 is inclusive: a push landing AT the cap succeeds.
- **A full recursive listing of the canonical repo 504s** after ~62,000 files. Always scope with `path_in_repo=<prefix>`. The repo is too large to enumerate from the root.
- **Concurrency causes the 429s.** 16 download workers failed 4-6% of files; 4 workers with backoff to ~60s succeeded. Running two of our own jobs against the API at once was self-inflicted rate limiting. Serialize: ONE API consumer at a time.
- **Stage off `/`.** Use `/mnt/eps-data/$USER/...` — the root disk is shared across ~15 sessions. Also exclude `.cache/**`, `**/tmp_*`, `**/*.incomplete` from uploads: an interrupted download leaves 0-byte temp files that otherwise get uploaded.
- Working script from the residue move: `/tmp/hf_move_repair.py` (per-prefix resume, verify-before-delete, checkpointed state). Worth adopting rather than rewriting.

## Related work already done or in flight

- **#2304** (running) — fixes the code-side gap that let this happen silently: the #1108 overflow fallback is scoped to `repo_type == "model"` at `hub.py:1688`, so dataset uploads never got it; plus a pointer-write hardcoded to model at `hub.py:757` and `create_repo private=False` at `hub.py:1818`. That makes future uploads ROUTE instead of fail; it does not reclaim capacity. Empirical findings above were handed to it as an `epm:progress` marker.
- **Residue move (2026-08-15/16)** — crashed-attempt `*_partial` prefixes moved to the private overflow repo then deleted from canonical. ~49k slots reclaimed at time of filing, ~60k projected. `issue1739_partial` (36,601 files) deliberately HELD while #1739 has a live session.
- **Storage Buckets** — HF's non-git, per-TB-billed repo type, explicitly exempt from repository file limits and named by HF's CTO as the intended relaxation path. The only option that removes this ceiling permanently rather than buying headroom. Worth evaluating as an alternative to repacking for future bulk artifacts.

## Acceptance criteria

1. A consumer audit (or equivalent safety argument) exists for every prefix repacked.
2. Repacked prefixes verified by name+size on the destination BEFORE any original is deleted.
3. Canonical repo file count measurably reduced; before/after recorded.
4. No reader broken — demonstrated by the audit plus a test run of affected analysis code.
5. The upload/verify constraints above encoded in whatever tooling lands, not just in this body.
