---
name: Claude accepts environment-dependent smoke-pass claims
description: Claude code-reviewer PASSes when implementer's smoke-run claim ("X passed in Ys") is true on the dev VM but the brief required env-independence (no HF model load, no network, etc.); Codex reproduces the failure in a clean sandbox
type: feedback
---

Claude code-reviewer PASSes round-N when the implementer's `### Smoke runs`
section reports green ("17 passed in 7.90s, CPU-only, no GPU"). Codex
reproduces the same command in a clean-cache / clean-network sandbox and
gets a different result ("15 passed, 2 errors") because the dev VM has
populated HF cache / pre-warmed uv cache / pre-existing data files.

**Why:** the implementer + Claude reviewer both run against the dev VM's
implicit environment (populated `~/.cache/huggingface`, pre-staged eval
files, etc.), so a "passes locally" claim looks unambiguous. Codex's
sandbox is the closer analogue of a fresh pod / CI / a reviewer running
on a different machine — its empty cache surfaces the latent network/cache
dependency.

**How to apply:**

1. When the reviewer brief contains explicit isolation language ("tests
   must not require GPU/vLLM/HF model load", "dry-run must just
   import+arg-parse, no HF spin-up", "no network calls in CI"), and Codex
   reports an empirical reproduction of the violation, the brief takes
   precedence over Claude's smoke-pass claim. The implementer's "passes
   on the dev VM" is environmentally lucky, not contract-conformant.

2. **Verify by reproducing in a clean env yourself** before deciding:
   ```bash
   env -i HOME=$HOME PATH=$PATH \
     HF_HOME=/tmp/empty-hf-cache \
     HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
     UV_CACHE_DIR=/tmp/uv-cache-recon \
     uv run pytest tests/<file>.py -q
   ```
   If you reproduce Codex's failure, the brief constraint is real-blocking
   regardless of Claude's smoke section. The dev VM is hiding the gap.

3. Be precise about what "HF model load" means: `AutoTokenizer.from_pretrained(model_id)`
   IS an HF model load by HF's own naming — `from_pretrained` is the
   canonical "load from Hub" API. A docstring that says "cached locally
   by HF" is an admission that the test will fail in a fresh env. Don't
   accept "but the tokenizer is just JSON, not weights" as semantic
   wiggle-room when the brief says "no HF model load."

4. Companion to `feedback_codex_step_06_literal_vs_purpose.md` —
   inverted polarity. That memory says "when Codex reads
   `smoke-run-missing` literally on a GPU-gated phase that physically
   can't run on the CPU dev VM, PASS with standing recommendation." THIS
   memory is the opposite case: when the brief language is EXPLICIT AND
   the violation is empirically reproducible in a clean env, the brief
   wins. Distinguishing test: is the constraint a portability requirement
   the brief named, or an artifact of where the smoke step runs?
   Reproducible-in-clean-env + explicit-brief-language → FAIL. CPU
   physically can't run GPU code + reviewer reading literally → PASS with
   pod-side gate.

5. Worst-case false-PASS cost is medium: tests/dry-runs that work on the
   dev VM but break on every fresh pod / CI / new developer's machine.
   Worst-case false-FAIL cost is ~15 min surgical fix (mock the tokenizer,
   short-circuit the dry-run before the load). Prefer FAIL when both
   reproducible-in-clean-env AND explicit-brief-language hold.

Anchor incident: task #504 round 13. Reviewer brief line 116: "The test
file must not require a GPU / vLLM / HF model load." Test fixture
(`tests/experiments/test_504_reval.py:37-47`) calls
`AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct", ...)`. Claude
PASSed citing "17 passed in 7.90s, CPU-only, no GPU"; Codex reproduced
"15 passed, 2 errors" in clean-cache env. Reconciler reproduced Codex's
result on the dev VM with `HF_HOME=/tmp/empty-hf-cache`. FAIL.
