---
name: Codex helper-name grep vs path-identity trace on upload contracts
description: Codex code-reviewer FAILs raw-completions/durability contracts by grepping for the canonical helper NAME (upload_raw_completions_to_data_repo) and missing a plan-registered bespoke upload path; trace writer-dir -> uploader-dir identity before believing it
type: feedback
---

Codex code-reviewer verifies upload/durability contracts by exact-grepping for the
canonical helper function name (`upload_raw_completions_to_data_repo`, `hub._upload`)
and issues a Critical FAIL when zero matches — even while acknowledging a bespoke
`upload_folder(...)` exists in the diff, which it dismisses as "not the required
helper" without tracing whether the uploaded directory CONTAINS the artifact.

**Why:** #552 round 7 (2026-06-11). Raw completions were written to
`<--output-base>/outcome/` and the driver bulk-uploaded exactly that dir to the HF
data repo with fail-loud `list_repo_files` verification, at the plan-§14-registered
destination, before merged-dir deletion. Codex's "raw-completions-upload-missing"
Critical was pure helper-name absence; the durability contract was fully satisfied
(and bulk `upload_folder` is the upload-policy-PREFERRED pattern under the 256
commits/hr cap).

**How to apply:** When Codex FAILs an upload/durability contract, do the path-identity
trace yourself: (1) find the artifact write site and resolve its actual output dir
(watch for `--output-base`-style indirection appending subdirs like `/outcome`);
(2) find every upload call in the driver and resolve its source dir + allow_patterns;
(3) check the plan's §14/durability table for the REGISTERED destination — a bespoke
path that matches the plan row is conforming, helper name notwithstanding; (4) check
ordering vs deletion/sentinel. Companion failure in the same round: Codex framed
plan-registered ORCHESTRATOR-side steps (WandB artifact + VM pull + sha256 before
Step-8 terminate, gated by upload-verifier) as a pod-side driver gap and demanded a
pod-side VM pull — incoherent by definition. Sibling patterns: "Codex Step 0.6
literal vs purpose", "Codex litigates pre-existing in round N".
