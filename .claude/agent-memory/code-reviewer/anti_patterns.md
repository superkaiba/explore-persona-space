---
name: Explore Persona Space Anti-Patterns
description: Recurring issues in this codebase to flag on sight
type: project
---

Flag these patterns whenever they appear in a diff:

| Anti-pattern | Why it's bad | Fix |
|--------------|--------------|-----|
| Bare `except: pass` or `except Exception: pass` | Silent failure — hides real bugs | Catch specific exceptions; log and re-raise |
| Hardcoded API keys / HF tokens / WandB keys | Security; also breaks for other users | Use `python-dotenv` + `.env`; never commit keys |
| `subprocess.run(..., shell=True)` with user input | Shell injection | Pass args as a list, no shell=True |
| Direct pip install instead of `uv add` | Drifts from uv.lock | Use `uv add <pkg>`; commit uv.lock |
| Edit code directly on pods (SSH + vim) | Sync conflicts, untracked mutations | Edit locally, commit, push, pod git pull |
| Upload with `upload_large_folder` on symlinks | Silently succeeds with 0 files (pod4 lost 384G this way) | Resolve symlinks before upload or use a different API |
| Mock-only tests for library integration | Masks real breakage (prior incident: truthification) | Add at least one integration test hitting the real path |
| `torch.load(path)` without `weights_only=True` | Unsafe deserialization | `weights_only=True` (required on torch 2.6+) |
| `yaml.load(...)` without SafeLoader | Unsafe | `yaml.safe_load(...)` |
| `data_files=None` or missing in HF `load_dataset` | Loads whole mixed dataset instead of intended split (lesson: truthification Exp 1) | Always specify `data_files=` explicitly |
| New training script without `nohup` in docs | Job dies on subagent disconnect | Command examples in docstring must include `nohup ... &` |
| Training script without WandB Artifact upload at end | Checkpoint gets lost on cleanup (prior incident: midtrain models lost) | Add `wandb.log_artifact(artifact)` after `trainer.save_model()` |
| Training script without HF Hub `push_to_hub` | Model stays only on pod — lost when pod recycles | Add `push_to_hub("superkaiba1/explore-persona-space-<name>")` |

**How to apply:** Grep the diff for the left-column patterns. Any hit → immediate flag in the review, with the "Fix" column as the suggested repair.
