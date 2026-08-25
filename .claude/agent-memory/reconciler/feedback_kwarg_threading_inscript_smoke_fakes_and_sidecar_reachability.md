---
name: kwarg-threading-inscript-smoke-fakes-and-sidecar-reachability
description: Kwarg-threading rounds — RUN the shared scripts' own --smoke (Claude checks only tests/ fakes); trace revision PERSISTENCE (sidecar mint/read) before crediting a mixed-revision Critical (#1901 r2)
metadata:
  type: feedback
---

Two calibrations from #1901 r2 (mlp-scaling-densify fix round, Claude PASS vs
Codex FAIL; adjudicated FAIL on a single upheld Major):

**1. Claude misses IN-SCRIPT self-smoke fakes when a round threads a new kwarg
through a shared helper.** The round added `revision=` to
`_download_chunk_with_retry` / `list_repo_tree` call sites in the shared #779
scripts. Claude's Step 3.75/3.8 verified the COMMITTED pytest fakes
(`tests/test_issue1482_driver.py`, `tests/test_issue1491_ladder_fits_counts.py`)
were updated and 40 tests green — but never checked the scripts' OWN
`--smoke` fake populations (`issue779_ffc_n1m_fits.py:2354/2413-2434/2499-2502`,
`issue779_ffc_n1m_generate_capture.py:1453-1457`), which lacked the kwarg.
**How to apply:** on any kwarg/signature-threading round over shared scripts,
grep the SAME FILES for `def _fake` / `class _Fake` / `_smoke` sections and
RUN the cheapest `--smoke` entrypoint (the gc one reproduced
`TypeError ... unexpected keyword argument 'revision'` rc=1 in ~10 s). Same
family as [[claude-misses-same-file-siblings]] — the sibling here is the
in-file fake, not a code path.

**2. Codex credits a mixed-revision Critical without tracing revision
PERSISTENCE.** Codex FAILed the size-only local-reuse skip
(`issue1901_paper_densify.py:180-184`) and the manifest completeness fast path
(`generate_capture.py:631-644`) as "the round-1 mixed-revision defect
remains". The predicates were real, but the sidecar
(`densify_mlp.py:576-614`) mints the revision ONCE on an empty root and READS
IT BACK on every resume before any staging — so both registered paths
(fresh-pod production; same-pod crash-resume) are revision-consistent, and
the bypass needs a sidecar-absent/hand-deleted OCCUPIED root that no
registered path produces post-fix. Demoted to defense-in-depth CONCERNs
(guard fresh-sidecar mint over occupied roots; add revision to the manifest
fast-path predicate).
**How to apply:** before upholding a "stale local reuse defeats the new pin"
Critical, read WHERE the pin is resolved and whether it is persisted +
re-read across the resume paths the plan registers; residual
misconfiguration-only shapes inherit CONCERN, not Critical (cf.
[[residual-gap-inherits-parent-severity-bar]],
[[codex-hardening-beyond-minimal-port-contract]]).
