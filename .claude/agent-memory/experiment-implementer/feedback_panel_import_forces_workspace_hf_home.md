---
name: panel-import-forces-workspace-hf-home
description: Importing the bystander-panel modules (or extract_persona_vectors) forces HF_HOME=/workspace at import time, breaking CPU-local tokenizer loads on the VM.
metadata:
  type: feedback
---

The bystander-panel modules (`scripts/_i416_bystander_panel.py`,
`_i398_bystander_panel.py`, etc.) import from
`experiments/phase_minus1_persona_vectors/extract_persona_vectors.py`, which has
a module-top `os.environ["HF_HOME"] = "/workspace/.cache/huggingface"` side
effect (line ~17). The panel modules scrub `CUDA_VISIBLE_DEVICES` around that
import but NOT `HF_HOME`.

**Why:** correct on the pod (`/workspace` IS the cache), but on the local VM
`/workspace` isn't writable, so any `AutoTokenizer.from_pretrained(...)` AFTER
the panel import dies with `PermissionError: /workspace` (or
`LocalEntryNotFoundError`). `HF_HOME` is read once when `huggingface_hub`
initializes its cache constant, so resetting `os.environ['HF_HOME']` AFTER the
panel import is too late.

**How to apply:** for CPU-local verification of any eval that imports a panel
module, EITHER (a) `AutoTokenizer.from_pretrained(...)` BEFORE the
`import _iNNN_bystander_panel`, with `HF_HOME=$HOME/.cache/huggingface` set in
the shell env, OR (b) load the tokenizer in a separate subprocess. The
dispatcher/driver themselves are unaffected (on the pod `/workspace` is right).
Related: [[feedback_ruff_strips_unused_imports]] (the other gotcha that recurs
on these eval ports — re-add the import AFTER its first use lands, since the
PostToolUse formatter strips it when momentarily unused).

## Index hooks moved from MEMORY.md (#1891 curation, 2026-07-30)

The always-loaded index was curated to fit the ~25 KB loader truncation limit (task #1891); the full pre-curation index hook(s) for this entry are preserved verbatim below.

- [Panel import forces HF_HOME=/workspace](feedback_panel_import_forces_workspace_hf_home.md) — VM-local tokenizer loads must run BEFORE the bystander-panel import.
