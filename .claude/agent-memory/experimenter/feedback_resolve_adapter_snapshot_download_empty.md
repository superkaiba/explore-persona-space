---
name: Dispatcher Hub-fetch via snapshot_download crashes at launch — recover with --adapter-path pre-staging
description: resolve_adapter-style snapshot_download(allow_patterns) returns an EMPTY snapshot on superkaiba1/explore-persona-space (truncated-siblings trap); pre-stage via list_repo_tree+hf_hub_download and relaunch with the script's --adapter-path flag — no code bounce needed
type: feedback
---

Dispatchers whose adapter resolution uses `snapshot_download(repo_id, allow_patterns=[f"{sub}/*"])`
deterministically crash at launch with `FileNotFoundError: Adapter missing/empty on Hub: <snapshot>/<sub>`
on `superkaiba1/explore-persona-space` — the snapshot dir is created but EMPTY (the known
truncated-siblings trap; see feedback_snapshot_download_truncated_siblings.md). A
`list_repo_files` gate PASSes (files ARE on the Hub), so the input-data gate does NOT catch this;
only the resolution API is broken.

**Why:** Burned at #558 v1 launch (2026-06-10). Gate verified all 12 `adapters/issue543/*_phase2`
on Hub; smoke crashed in <10s in `resolve_adapter`. Parent #543 never exercised the Hub path
(it trained adapters locally), so "reused verbatim from parent" gave no protection.

**How to apply:** Pre-launch, grep the dispatcher for `snapshot_download`. If its Hub-fetch path
uses `allow_patterns` on this repo AND the script exposes a local-path flag (`--adapter-path`),
recover launch-side without a code bounce: pre-stage each artifact via
`list_repo_tree(repo, path_in_repo=sub)` + `hf_hub_download(filename=fp, local_dir=DEST)` (verify
`adapter_config.json` per dir), rewrite the driver to pass the local path, relaunch (rewriting the
pidfile), and record the deviation + code-debt in the `epm:run-launched` note. Same checkpoints,
same provenance. If no local-path flag exists → `epm:failure v1 failure_class: code`.
