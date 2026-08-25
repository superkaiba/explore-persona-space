---
title: HF data repo superkaiba1/explore-persona-space-data is at the 1,000,000-file
  hard ceiling — fleet-wide multi-file uploads now fail; pack/prune/overflow decision
  needed
kind: infra
tags:
- fleet
- hf-quota
created_at: '2026-08-14T05:11:33Z'
has_clean_result: false
origin_prompt: auto-filed by /issue 2254 orchestrator on the wave-1 upload rejection
workflow: v1
---
Observed 2026-08-14T05:09Z on #2254 judge wave 1: upload_folder of a ~2,310-file per-cell judge tree was rejected by the Hub commit endpoint ('Your git repo would contain 1000925 files after this push, over the limit of 1000000 files'), i.e. the shared data repo holds ~998,6xx files and has only a few hundred files of headroom. EVERY sibling issue's next multi-file upload (per-cell trees, raw-completion sweeps) will hit the same rejection. #2254 works around it locally by packing judge trees into JSONL shards before upload (its round-4 fix), but the repo-level exhaustion needs a fleet decision: (a) pack-and-replace the worst historical per-cell prefixes (top offenders discoverable via list_repo_tree counts), (b) an overflow data repo (the #1108 model-repo pattern) with a router seam in orchestrate/hub.py, or (c) both. Upload policy text/JSON-always stays binding — this is about FILE COUNT, not bytes. Evidence: /tmp/issue-2254-judge-wave1.log (BadRequestError traceback), task 2254 events v42.
