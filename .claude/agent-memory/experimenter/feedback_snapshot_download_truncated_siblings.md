---
name: snapshot_download silent-empty family — truncated siblings, 0-file fetch, --adapter-path recovery
description: snapshot_download(allow_patterns=...) silently returns 0 files on superkaiba1/explore-persona-space (repo_info.siblings truncates below the 50k threshold); downstream scripts misdiagnose as "checkpoint not present". Use list_repo_tree + hf_hub_download per file.
metadata:
  type: feedback
---

`snapshot_download(allow_patterns=...)` filters against `repo_info.siblings`, which truncates BELOW the documented 50,000-file fallback threshold (observed 7,901 reported / 14,439 actual on `superkaiba1/explore-persona-space`, hf_hub 0.36.2). Result: `Fetching 0 files: 0it`, no warning, empty snapshot dir. A `list_repo_files` gate PASSes (files ARE on the Hub), so input-data gates do NOT catch it.

**Why (three burns):** #375 round-4 (2026-05-21) — phase_pilot's `download_adapter` fetched 0 files. #399 round-6 (2026-05-27) — eval-only relaunch's `resolve_checkpoint` treated 0-files-fetched as "checkpoint not present on Hub" and told the operator to RE-TRAIN finished work. #558 v1 (2026-06-10) — `resolve_adapter` crashed `FileNotFoundError: Adapter missing/empty on Hub` on 12 adapters the gate had just verified.

**How to apply:**
1. NEVER trust a "not present on Hub" error without independently checking `HfApi().list_repo_files(repo)` (the `hf` CLI has no `api` subcommand — use Python). Files present + 0-file fetch = this bug; bounce `failure_class: code`, never re-train.
2. Reliable pattern: `list_repo_tree(repo, path_in_repo=sub, recursive=True)` + `hf_hub_download` per file. `hf_hub_download(filename=<exact_path>)` does not consult siblings.
3. Launch-side recovery without a code bounce (#558): if the script exposes a local-path flag (`--adapter-path`), pre-stage each artifact via list_repo_tree + hf_hub_download (verify `adapter_config.json` per dir), relaunch with the local path, record the deviation + code-debt in the `epm:run-launched` note. No local-path flag → `epm:failure v1 failure_class: code`.
4. Preflight diagnostic: `len(repo_info(repo).siblings)` vs `len(list_repo_files(repo))` — divergence means snapshot_download with allow_patterns is unsafe on that repo.

**Repo-scale caveat (#833, 2026-07-03):** everything above is MODEL-repo
scale (~14k files) — steps 1/2/4 still work there. On the ~1M-file DATA
repo (`superkaiba1/explore-persona-space-data`) the `list_repo_files` legs
themselves time out (>90 s) and `snapshot_download` wedges 40+ min in
full-tree enumeration before `allow_patterns` applies. There, use SCOPED
`list_repo_tree(path_in_repo=<prefix>, recursive=True)` + a ≤6-worker
`hf_hub_download` pool, ONE process (recipe:
`scripts/issue833_gcp_phase_d.sh`, readable via
`git show 22388e4b3d:scripts/issue833_gcp_phase_d.sh` until #833's merge
lands it on main; twin memory:
`../experiment-implementer/feedback_hf_snapshot_download_full_tree_enumeration.md`).
