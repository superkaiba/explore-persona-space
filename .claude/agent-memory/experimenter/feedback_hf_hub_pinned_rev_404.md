---
name: hf-hub-pinned-rev-404
description: A pinned HF revision that doesn't coexist with its path dies in seconds — loudly (single-file hf_hub_download → EntryNotFoundError 404) or QUIETLY (prefix stage → 0 files). Usually a SHA copy-pasted from the wrong artifact, incl. a staging manifest's INPUT-provenance revision passed where the OUTPUT-upload commit was needed. Disposition splits: pin hardcoded in a script = code-class bounce; pin supplied by the BRIEF = re-pin in-turn.
metadata:
  type: feedback
---

A dispatcher hardcoding both an HF dataset revision SHA and a file path dies in seconds with `EntryNotFoundError: 404 ... resolve/<sha>/<path>` when the pair doesn't exist together — the SHA predates the file, post-dates a rename, or was copy-pasted from a different artifact's commit (adapter vs data repo).

**Why:** #477 v5 recovery diagnostic (2026-06-05) — `i477_reval_confirm.py:113` 404'd on `issue472_neg_geometry/persona_bank.json` @ `66d7db7a`.

**How to apply:** CODE-class, not infra — don't fix experimenter-side. Post `epm:failure v1 failure_class: code` with: the pinned SHA, the exact 404'd path, the script+line, and the recommendation that the implementer verify via `HfApi().list_repo_tree(repo_id, revision=<sha>, repo_type="dataset")` then re-pin or fix the path. Same HF listing/path family as [[feedback_snapshot_download_truncated_siblings]], but this one is a loud single-file 404, not a silent empty fetch.

## Two extensions (#2091, 2026-08-05)

**(1) The PREFIX variant fails QUIETLY, not with a 404.** The entry above is a
single-file `hf_hub_download` → loud `EntryNotFoundError`. A PREFIX stage
(`hub.stage_hub_prefix` / a scoped `list_repo_tree`) at a non-coexisting
revision instead resolves to **0 files** and the dispatcher dies at its first
staged read with no 404 anywhere — same root cause, no loud signal to grep for.
Probe shape is the same either way: `list_repo_tree(repo_id,
path_in_repo=<prefix>, revision=<pin>, repo_type="dataset")` and count the
result, BEFORE launch.

**(2) A staging manifest's `dataset_revision` is INPUT provenance, not the
upload commit.** The named recurring source of a wrong-artifact SHA: a staging
run records the revision it READ its inputs FROM. Passing that as the
consumer's `--dataset-revision` points at a commit where the artifact the
staging run WROTE does not yet exist. #2091: `stage_manifest.json` carried
`c8de6fb…` (the #1739 labeling-input pin) while the contexts tree it produced
lives at `9d48f667…` — the orchestrator's brief passed the former,
`issue2091_decode/contexts` resolved to 0/27 files, and the P0 dispatcher
crashed in seconds. Two pins with two roles; never collapse them.

**Disposition splits on WHERE the pin came from** — this refines the
"CODE-class, don't fix experimenter-side" line above, which assumed a
hardcoded pin:

- **Pin hardcoded in the script** → unchanged: code-class bounce, post
  `epm:failure v1 failure_class: code`.
- **Pin supplied by the BRIEF / a launch flag** → RE-PIN IN-TURN. Verify the
  prefix at the candidate revision (and at main), relaunch with the verified
  one, and report the correction. Do NOT post an infra-failure and stall: no
  code is wrong, the launch argument is. #2091 recovered this way with zero GPU
  spend lost (the crash preceded any generation).

Reproducibility duty on re-pin: BOTH revisions carry into the clean-result
footer with their roles named distinctly (input-staged-from vs
consumed-at/run-of-record). Collapsing them into one "dataset revision" line
reintroduces exactly the ambiguity that caused the crash.

Related: `.claude/rules/artifact-reuse.md` § Plan-time search + verification
already requires probing AT the pin (#1345 — existence at `main` does not imply
existence at the pin). That covers a pin gone STALE for the artifact it names;
these two extensions cover a pin that was never the right KIND of pin, and the
quiet 0-files shape.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [HF Hub pinned-revision 404](feedback_hf_hub_pinned_rev_404.md) — hf_hub_download(revision, filename) 404s when the pair doesn't coexist; code-class, implementer verifies via list_repo_tree (#477 v5)
