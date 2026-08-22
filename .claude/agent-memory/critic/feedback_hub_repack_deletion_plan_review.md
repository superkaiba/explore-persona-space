---
name: hub-repack-deletion-plan-review
description: Review checklist for HF-repo repack/deletion infra plans (#2321 shape) — the probe-first commit chain, the one real residual race, and which worries are NOT Must-Fixes
metadata:
  type: feedback
---

Review frame for plans that DELETE from the canonical HF data repo by repacking
(the #2321 shape; the model repo `adapters/` is the named next domino).

**The safety chain that earned APPROVE (verify each link is present):**
1. delete set DERIVED FROM the archive shards staged in the SAME `create_commit`
   (set-equality assert; single source of truth = shard member records);
2. net-negative assert per data commit; manifest commits whitelisted only after
   the prefix freed slots;
3. cap-semantics settled by a throwaway-file probe (add / add+delete-at-cap /
   cleanup) BEFORE any artifact deletion, with a distinct-rc STOP path;
4. ambiguous-outcome commits (timeout / conn drop / gateway 5xx) are NEVER
   blindly retried — HF's 60 s commit timeout can complete server-side
   (storage-limits docs, verbatim) — probe Hub state (shards present + sources
   absent) and branch landed / re-issue / abort-on-mixed; `create_commit`
   structurally excluded from `retry_transient` (`# NO_RETRY:` waiver is
   lint-recognized, workflow_lint.py ~L9085/L9507 — call line or line above);
5. per-unit drift guard (size+blob_id vs pinned-revision census) + per-prefix
   abort; live-owner prefixes get a durable marker + pre-commit liveness
   re-probe with SKIP-and-defer, never force-write;
6. download integrity anchored server-side (non-LFS: git blob sha1 == blob_id;
   LFS: lfs.sha256) — demand the anchors be EMPIRICALLY confirmed, not doc-read.

**The one real residual race (Concern, not Must-Fix):** a concurrent overwrite
of a source file between drift_check and create_commit deletes bytes not in the
same-commit archive. Why not Must-Fix: window is seconds; deleted bytes remain
in git history at the intermediate revision (HF deletes never destroy blobs
until `super_squash_history`); and mandatory `parent_commit` pinning has a real
downside — once slots free, ANY fleet write anywhere in the repo moves HEAD and
spuriously fails the pinned commit, so probe-as-primary + pin-as-defense-in-depth
is a defensible disposition, not a gap.

**Worries that are NOT Must-Fixes here:** 429-driven pacing saturating the
fleet-shared resolver bucket (externality, recommend a client-side rate ceiling);
HF's 50-100-files/commit recommendation vs larger probe-ramped units (throughput
risk with a measured ramp, not safety); the INDEX.json-absent window between a
prefix's last data unit and its manifest commit (loud 404s, transient);
uncovered direct `hf_hub_download` call sites (loud miss, follow-up task).

**How to apply:** grep the plan for each chain link above; construct the
killed-driver / timeout / concurrent-writer / resume paths explicitly and ask
"can a delete op ever name a path whose bytes are not in an op of the same
commit?" — the answer should reduce to the drift-race only. Related:
[[infra-plan-checklist]].
