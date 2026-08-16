---
title: Repack the 10 largest prefixes in the HF data repo to recover ~610k of the
  1,000,000-file cap
kind: infra
tags: []
created_at: '2026-08-15T23:35:54Z'
has_clean_result: false
origin_prompt: yes [repack top 10 prefixes — AskUserQuestion selection on the HF 1M-file
  cap decision, 2026-08-15]
workflow: v1
---
# Repack the 10 largest prefixes in the HF data repo to recover ~610k of the 1,000,000-file cap

## Goal

Recover file-count headroom on `superkaiba1/explore-persona-space-data` by consolidating the small files in its 10 largest prefixes into per-prefix archives, **with zero data loss and no deletions of un-archived content**. Target: free ~610,000 of the 1,000,000 slots, returning the repo to ~39% of the cap.

This is the headroom half of the 2026-08-14 file-cap incident. The routing half — making the #1108 overflow fallback actually fire for dataset repos and bulk uploads — is **#2304** and is independent: #2304 keeps new writes working, this task makes the canonical repo writable again.

## Why repack rather than prune or relocate

Measured inventory (2026-08-15, read-only parallel per-prefix tree walk, 402 prefixes, 227 s, 0 errors; posted in full on #2304):

**999,999 files across 402 prefixes** — sitting exactly at the Hub's hard cap. The count is highly concentrated:

```
206,604  20.7%  issue1481_conpos_grid
 58,392   5.8%  issue1090_pvdatagen
 54,103   5.4%  issue1586_methodgen
 53,858   5.4%  issue667_alllayer
 49,380   4.9%  issue1434_writingstyle
 43,905   4.4%  issue1739_ctxmap
 41,759   4.2%  issue2224_screening
 36,601   3.7%  issue1739_partial
 35,394   3.5%  issue1090_partial
 30,360   3.0%  issue1489_ctx_aug
--------------------------------
610,356  61.0%  = top 10
```

This is a file-GRANULARITY problem, not a data-volume problem: one JSON per (condition, seed), one npz per cell, per-attempt crash dumps. So the fix is fewer, bigger files in the same repo — not a new repo (which hits the same cap at the same rate and multiplies every consumer's path logic), and not deletion.

**User decision, 2026-08-15: repack the top 10 prefixes; do NOT delete.** The 99,403-file `*_partial` / `_crash_dumps` residue (9.9% of the cap) was explicitly considered for deletion and declined — it may be repacked like anything else, but nothing is deleted that is not first archived in the same atomic commit.

## The load-bearing assumption, and how to test it safely

Repack rests on the Hub counting files at HEAD, so that removing originals frees slots. Evidence: the rejection reads `Your git repo would contain 1000009 files after this push, over the limit of 1000000` — "after this push" describes the resulting tree.

This is **inferred, not measured**. Do NOT test it by deleting real artifacts. The safe test is the first repack commit itself:

- HF commits are atomic. One commit that ADDS the archive and REMOVES the originals nets negative.
- If the cap is evaluated on the resulting tree, it lands. If not, it is rejected cleanly with nothing lost and no intermediate broken state.
- Run this first commit on the **smallest** of the top 10 (`issue1489_ctx_aug`, 30,360 files), not the largest, so a surprise has the smallest blast radius.

If that first commit is rejected, STOP and re-plan — the whole approach is invalidated and the task should report that rather than reaching for deletion.

## Scope

1. **Archive format.** Pick one that preserves exact bytes and per-member paths, and that a consumer can read a single member from without downloading the whole archive where feasible. State the choice and why. Uncompressed `.tar` is the conservative default (member-addressable via offsets, no re-encode); per-prefix `.parquet` is worth considering for the uniformly-shaped JSON row sets, but only where it is provably lossless for that prefix's content.
2. **Per-prefix index.** Each archive ships a sibling `INDEX.json` mapping original path → (archive, offset/member) so a reader can resolve any historical path. This is 2 files per prefix, not 1.
3. **Atomic add-then-remove.** Each prefix is repacked in a SINGLE commit that adds archive + index and deletes the originals. Never a delete commit separate from the add.
4. **Round-trip verification before the delete lands.** For every prefix: extract the archive to a scratch dir and byte-compare (sha256 per member) against the live originals. The delete half of the commit is composed only after the comparison passes for 100% of members. Report the verified member count per prefix.
5. **Reader shim.** A helper in `orchestrate/hub.py` (or alongside it) that, given an original path, transparently resolves it from the archive when the raw path 404s. Without this, every existing consumer of those 10 prefixes silently breaks.
6. **Order of operations.** `issue1489_ctx_aug` first as the semantics test, then the remaining 9 in ascending size.
7. **Re-measure and report.** Re-run the per-prefix count after each prefix lands; report realized slots freed vs the 610k projection.

## Out of scope

- The #2304 routing fix (separate task, already running).
- Deleting anything, including crash-dump residue — declined by the user.
- The model repo `superkaiba1/explore-persona-space` (188,630 files, 129,064 in `adapters/`, 4.03 TB — ~19% toward the same cap). It is the next domino and should get its own task; note it here so it is not lost.
- Fixing the generator (capping files-per-prefix at write time in `upload_raw_completions_to_data_repo()`), which is the durable fix and deserves its own task once headroom exists.

## Acceptance

- All 10 prefixes repacked, each in a single atomic add-then-remove commit.
- Per-prefix sha256 round-trip verification passed for 100% of members, counts reported.
- Realized file count re-measured and reported; ≥ ~550k slots freed (the 610k projection less index/archive overhead).
- Reader shim resolves an original path from every repacked prefix, covered by a test.
- No file deleted that is not present in the archive committed in the same commit.

## Provenance

Filed 2026-08-15 from the chat decision on the file-cap incident found while driving #2162 on 2026-08-14. Sibling of #2304 (routing). Inventory measured and posted to #2304 the same day.
