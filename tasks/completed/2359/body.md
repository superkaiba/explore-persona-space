---
title: 'verify_uploads.py outroot-residue: basename match vs issue-scoped git trees
  lets a sibling leg''s committed file cover this leg''s unpersisted file'
kind: infra
tags:
- workflow-fix
created_at: '2026-08-18T01:18:19Z'
has_clean_result: false
origin_prompt: auto-filed by /issue 2333 orchestrator from the leg-B upload-verifier's
  workflow-fix-candidate (cross-leg basename false-OK, 2026-08-18)
workflow: v1
---
# verify_uploads.py outroot-residue check: basename match against issue-scoped git trees lets a SIBLING LEG's committed file cover this leg's unpersisted file (cross-leg false-OK)

## Goal

Close the #2187 outroot-residue check's cross-leg blind spot in `scripts/verify_uploads.py`: the residue check matches by BASENAME against issue-scoped git trees, so a sibling leg's committed same-named file silently covers the current leg's unpersisted file.

## Incident (2026-08-18, task #2333 leg-B per-pod verification)

Leg-B's `/workspace/issue2333_out/q35/manifests/upload_done.json` (sha256 `0a052e8b78a2…`) read `outroot_residue: OK` in the mechanical pass because leg-A's committed `eval_results/issue_2333/q25/upload_done.json` (sha256 `6f43c93d…`, DIFFERENT bytes) shares the basename. The exploratory pass caught it (FAIL blocker, remediated at `d59466c3c1`), but the mechanical check alone would have shipped a false-OK. Multi-leg issues (`pod-<N>-<slug>` rounds) are now the common shape, so a per-leg residue can hide behind any prior leg's remediation commit.

## Fix (prescribed by the verifier)

When a basename match resolves ONLY via the issue-scoped git arm (not an HF prefix), disambiguate by content — compare git blob sha1 of the committed candidate(s) vs the disk file when the listing mode has local access, or at minimum WARN (`outroot-residue-basename-git-only`) naming both paths so the verifier's exploratory pass knows to byte-check. Size-equality is a cheap first-pass discriminator available in both modes. Add a regression test with two legs sharing a basename with different bytes.

## Candidate metadata

- target_file: scripts/verify_uploads.py (#2187 outroot-residue arm)
- fingerprint: outroot-residue-basename-cross-leg-false-ok
- confidence: high (reproduced live in #2333 leg-B verification; both sha256s recorded)

## Provenance

- workflow_fix_target: scripts/verify_uploads.py
- source: auto-filed by the /issue 2333 orchestrator from the leg-B upload-verifier workflow-fix-candidate (cross-leg basename false-OK, 2026-08-18)
