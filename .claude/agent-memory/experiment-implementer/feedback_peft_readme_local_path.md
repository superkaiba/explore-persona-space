---
name: PEFT save_pretrained writes local path in README base_model
description: PEFT 0.13+ writes a YAML frontmatter README.md whose base_model field is the LOCAL path of the model PEFT was loaded from; HF Hub's metadata validator rejects local-path values with 400 Bad Request. Always rewrite the README before pushing to HF Hub.
type: feedback
---

`peft.PeftModel.save_pretrained()` (0.13+) writes a `README.md` whose YAML
frontmatter contains:

```yaml
base_model: /workspace/tmp/...whatever-local-path-the-merged-base-was
library_name: peft
tags: [...]
```

When the caller subsequently calls `HfApi.upload_folder` (or
`api.upload_file` on the README), HF Hub's metadata validator rejects the
README with `400 Bad Request — Invalid metadata in README.md` because
`base_model` is supposed to be a Hub model id (e.g. `Qwen/Qwen2.5-7B-Instruct`),
not a filesystem path. The upload fails server-side but every adapter
file other than the README does land on the Hub — the validator pre-checks
metadata and rejects the whole commit.

**Why:** this is the R5 silent-failure mode in issue #228. Phase 0 trained
70 marker LoRAs that all hit this 400; the upload helper swallowed the
exception (`except Exception: log warning`) and the worker reported
"TRAINED + uploaded" while no adapter actually reached HF Hub.

**How to apply:** when training a LoRA on top of a *merged* base (the
adapter is loaded from a local merged dir, not an HF id), ALWAYS rewrite
the README's `base_model` field to the canonical HF Hub model id BEFORE
pushing. Reusable helper exists at
`scripts/train_marker_loras_228.py:rewrite_adapter_readme_base_model()`
and is callable from any post-training pipeline. Do the rewrite via
`yaml.safe_load` / `yaml.safe_dump`, NEVER regex on YAML.

**Verification command** (run this whenever you suspect a silent upload
loss):
```python
from huggingface_hub import HfApi
import os
files = HfApi(token=os.environ['HF_TOKEN']).list_repo_files('superkaiba1/explore-persona-space')
slot = 'adapters/<your-target-slot>/adapter_config.json'
print('present:', slot in files)
```

If the upload helper logs a WARNING but reports success, this is the bug.

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [PEFT README local-path bug](feedback_peft_readme_local_path.md) — save_pretrained writes base_model=local-path; Hub 400s; rewrite before upload.
