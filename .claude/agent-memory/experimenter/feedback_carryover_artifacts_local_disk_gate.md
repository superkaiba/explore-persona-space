---
name: Carry-over artifacts — HF gate misses local-disk staging + recipe drift
description: HF Hub visibility PASSing is necessary but NOT sufficient — dispatchers read from LOCAL disk; stat-check every argparse local-path default on the pod (including scripts the dispatcher shells out to) before launch.
type: feedback
---

When a plan claims "carry-over data on HF Hub" and the HF visibility gate PASSes, the dispatcher still reads from LOCAL disk (`data/issue_<M>/...` argparse defaults). On a fresh pod those paths are empty and the launch crashes in <10s with FileNotFoundError. Hand-curated staging recipes also drift from the real defaults: #504's round-2 recipe listed 5 files but omitted `R_eval.json`, which `i504_run_cell.py:86` AND `i504_eval_trajectory.py:53` both hard-default — smoke (no eval) passed, the full sweep would have crashed at the first cell's eval.

**Why:** burned at #504 v1 (2026-06-06, HF gate passed, crash on `data/issue_472/centroids_L10.pt` missing locally) and #504 v10 (2026-06-06, R_eval.json missing from the staging recipe). Symmetric read-side gap to the #488 write-side path-paraphrase guard. The post-#468 workflow fix on `experimenter.md § Before Running item 4` (introspect argparse defaults + stat-check + auto-stage) is the canonical defense.

**How to apply:** never trust the brief's staging recipe verbatim. Grep `add_argument.*Path\(.*data/issue_<M>` across the dispatcher AND every script it shells out to; stat-check each local default on the pod; stage missing files from `superkaiba1/explore-persona-space-data` via `hf_hub_download`. If no HF mirror exists, post `epm:failure v1 infra reason: dispatcher-default-path-no-hf-mirror`. Prefer the dispatcher's `--dry-run` (if exposed) — it exercises the real read path.
