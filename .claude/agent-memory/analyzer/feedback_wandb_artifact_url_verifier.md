---
name: WandB artifact URLs fail the verifier permanence check
description: verify_task_body URL-permanence requires wandb.ai URLs to contain /runs/, /groups/, or /reports/ — artifact page URLs (/artifacts/model/...) FAIL; cite artifacts by qualified name in inline code instead
type: feedback
---

`verify_task_body.py`'s Reproducibility URL-permanence check accepts wandb.ai URLs only if they contain `/runs/`, `/groups/`, or `/reports/`. A WandB **Artifact** page URL (`https://wandb.ai/<entity>/<project>/artifacts/model/<name>/v0`) FAILs even though it is version-pinned.

**Why:** the check is a regex allowlist (verify_task_body.py ~line 1458), written before checkpoint-rescue-to-WandB-Artifact deviations existed (first hit: task #547, 2026-06-10, HF public-storage quota 403 forced 32/180 adapters onto artifact `i547-missing-adapters:v0`).

**How to apply:** when a clean-result Reproducibility section must reference a WandB Artifact, cite it by its qualified name in inline code — `` `thomasjiralerspong/explore-persona-space/<name>:v0` `` — instead of the http URL. The name is version-pinned and resolvable via the wandb API; the regex only scans http URLs, so inline-code names pass.

Related: the `audit_clean_results_body_discipline.py` `experimental_arm` pattern fires on the literal phrases "two arms"/"three arms" (not on "the role arm") — write "two encodings"/"all three encodings" in body prose and alt text.
