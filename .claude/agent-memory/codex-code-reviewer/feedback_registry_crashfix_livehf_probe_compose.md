---
name: registry-crashfix-livehf-probe-compose
description: "Data-registry crash-fix rounds (#2502 r9): the composer runs a bounded live-HF metadata probe to convert network-only (config,split) ground truth into compose-time facts (Codex has no network); resume-compat fingerprint claims get a mechanism-reconciliation duty, not claim-inheritance"
metadata:
  type: feedback
---

When the round under review fixes a LIVE data-plane crash by correcting a
per-source HF registry (the #2502 r9 shape: `ValueError: Bad split` →
SourceSpec (config,split) audit + registry-wide preflight + loud handler),
three compose duties beyond the standard revision-round recipe
(first hit: #2502 r9, 2026-08-25):

1. **Composer runs the bounded live-HF metadata probe.** The round's
   central claims ("mask offers only test", "no sycophancy BuilderConfig",
   "behaviors splits = harmful/benign") are HF-network ground truth Codex
   structurally cannot check (no network) and the report cannot
   self-certify. `get_dataset_config_names` / `get_dataset_split_names` /
   `dataset_info(revision=...)` / `list_repo_files` over the corrected
   (ungated) sources is metadata-only, ~seconds each — inline the results
   as a "Composer LIVE-HF verification" block and scope Codex's duty to
   REGISTRY ↔ HF-fact ↔ report three-way coherence, with an explicit
   "you have NO network — do NOT attempt any HF call" fence and a
   "third-party sources unreadable ≠ BLOCKED lens" carve-out (the real
   probe transcript + the composer probe are the binding evidence for
   deferred datasets/HfApi calls the offline smoke fences off).
   State probe SCOPE honestly (e.g. splits verified on ONE of six configs;
   gated sources unprobed).

2. **Resume-compat fingerprint claims get a MECHANISM-reconciliation
   duty.** A "conditional key keeps untouched fingerprints byte-identical"
   claim can contradict a sibling claim ("the fingerprint now carries
   split as a first-class key") — if the new key were added
   unconditionally, byte-identity would be false. Pre-trace the
   reconciliation as an explicit duty item (read `_stage_fingerprint` OLD
   vs NEW blob; resolve already-keyed vs also-conditional vs proof-wrong)
   rather than letting Codex inherit either claim. Same family: un-keyed
   spec axes (revision_ref) = silent stale-reuse; renamed checkpoint
   FILENAMES break resume even when fingerprints match (fresh-out-dir
   default vs reused-out-dir safety are DIFFERENT claims — don't conflate
   the implementer's same-dir resume demo with crashed-run resume).

3. **Plan-silence framing prevents false plan-adherence FAILs.** When the
   plan pins dataset IDs + budgets only, say so explicitly ("plan §4
   declares NO per-source (config,split); do not demand plan text that
   never existed") and route the registry elections (e.g. narrowing
   legalbench to one task config) to a named-election adjudication duty
   instead of a plan-deviation row.

**Why:** neither reviewer can reach HF (Codex: no network; Claude twin:
re-running the probe duplicates spend and races the implementer); without
the composer probe the (config,split) audit is graded on report
self-consistency alone — exactly how the original bad registry shipped.

**How to apply:** any crash-fix / hardening round whose diff corrects
external-resource registry metadata (HF configs/splits/revisions, API
model ids, dataset paths). Related: [[degenerate-input-crash-fix-compose]],
[[revision-round compose recipe (round 2+)]], [[whole-round-unsplit-compose]].
