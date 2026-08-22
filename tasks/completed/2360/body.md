---
title: Preflight passes a pod venv whose lock-pinned package has no metadata and cannot
  import (sympy half-install breaks the whole transformers stack)
kind: infra
tags: []
created_at: '2026-08-18T06:27:14Z'
has_clean_result: false
parent_id: 2329
origin_prompt: 'Surfaced during /issue 2329 workflow-v2: pod-2329-margin bootstrapped
  BOOTSTRAP-OK and passed preflight, but sympy had no dist metadata and could not
  import, killing torch._dynamo and every transformers modeling path; gate0b failed
  with a misleading AutoModelForCausalLM lazy-wrapper error.'
workflow: v1
---
# Preflight passes a pod venv whose lock-pinned package has no metadata and cannot import

Found while driving #2329's TF-margin deferred-leg recovery on a freshly
provisioned `pod-2329-margin` (1x H100, intent=eval). Bootstrap reported
`BOOTSTRAP-OK` and step 10/11 preflight passed, yet the venv's entire
torch/transformers modeling stack was unusable.

## Symptom (and why it points the wrong way)

The first GPU-adjacent gate died with what reads as a version incompatibility,
immediately after the sanctioned `transformers` 4.57.6 -> 5.15.0 pin:

```
ModuleNotFoundError: Could not import module 'AutoModelForCausalLM'.
  Are this object's requirements defined correctly?
```

That is transformers' lazy-module wrapper masking the real cause. Importing the
leaf modules directly gives the actual chain:

```
transformers.generation.utils -> ..masking_utils -> torch._dynamo
  -> torch.fx.experimental.symbolic_shapes -> import sympy
  -> ModuleNotFoundError: No module named 'sympy.utilities'
```

`torch` itself was healthy (2.8.0+cu128, `cuda.is_available() == True`).
**sympy was half-installed:** `importlib.metadata.version('sympy')` raised
`PackageNotFoundError` — no dist metadata at all, while a partial `sympy/`
tree existed on disk. Because `torch._dynamo` imports sympy, every
transformers modeling path was dead.

Most likely origin: the hardlink fallback that warned repeatedly through
bootstrap and again on each `uv run`:

```
warning: Failed to hardlink files; falling back to full copy. ...
         If the cache and target directories are on different filesystems,
         hardlinking may not be supported.
```

On RunPod the uv cache and `/workspace` venv sit on different filesystems
(MooseFS), so the copy path is always taken — and a partial copy leaves exactly
this shape.

> **CORRECTION (2026-08-18) — the stated origin above is WITHDRAWN by its own
> author; do not chase it.** #2329's author retracted this hypothesis in that
> task's `epm:progress` v94 (2026-08-18T07:57:40Z), after the body above was
> written. **The cause is NOT established.** An exhaustive dist-info/RECORD
> manifest audit — 215 distributions, every RECORD-listed file
> existence-checked — found **zero** missing files, which is precisely where a
> truncated copy would show up. So the partial-copy hypothesis is REFUTED for
> the second casualty (`anthropic`, below), whose signature is instead an
> *internally inconsistent* install: RECORD complete, yet
> `anthropic/types/__init__.py` imports `.shared`, which is absent — the shape
> of a mixed-version / incompletely-replaced install, not a truncated one. For
> `sympy` the original evidence is **unrecoverable**: its dist metadata was
> already absent when found, and the force-reinstall overwrote the state.
>
> What SURVIVES unchanged: sympy's observed metadata absence is real, and the
> fix direction is not weakened but strengthened — in the author's own words,
> "preflight should import-test the stack rather than validate the lock."
> Surface 2 below therefore stands only as **warning suppression**, never as
> corruption prevention (`.claude/rules/gotchas.md` § pod-venv rebuilds already
> records that `UV_LINK_MODE=copy` does NOT prevent the ESTALE class).

## The gap

`explore_persona_space.orchestrate.preflight` checks "env vs `uv.lock`" — i.e.
it validates that the lock is satisfied *as recorded* — and passed here. It does
not verify that the load-bearing stack actually IMPORTS. A package can be
lock-satisfied on paper, missing its metadata, and unimportable, and preflight
still returns ok. That is the failure this task should close: preflight's
contract is meant to be "fix every failure, never skip", which only holds if a
pass means the stack runs.

## Proposed fix (two candidate surfaces; the implementing session should pick)

1. **`orchestrate/preflight.py` — add an importability probe.** A cheap
   subprocess `python -c "import ..."` over the load-bearing stack
   (`torch`, `torch._dynamo`, `transformers`,
   `transformers.models.auto.modeling_auto`, `accelerate`, `safetensors`,
   `huggingface_hub`, `numpy`, `scipy`, `tokenizers`, and `vllm` where the
   intent implies it), each failure reported with the leaf traceback rather
   than the lazy-wrapper message. This is seconds of wall-time against a class
   of failure that currently surfaces only after provisioning + staging.
   Consider also asserting `importlib.metadata.version(p)` resolves for each
   lock-pinned distribution — the metadata absence was the tell here and is
   cheaper to check than an import.
2. **`scripts/bootstrap_pod.sh` — make the copy path deterministic.** Export
   `UV_LINK_MODE=copy` up front on pods (the fallback is guaranteed there, so
   the warning is noise that hides real problems), and re-verify after install
   rather than trusting the installer's exit code.

## Repair that unblocked #2329 (for reference; NOT the fix to ship)

```
UV_LINK_MODE=copy UV_NO_SYNC=1 uv pip install --force-reinstall sympy
```

Deliberately **not** `uv sync`: that resolves from the lock and would revert
the gate0b `transformers==5.15.0` pin back to the repo's 4.57.6, which lacks
the `qwen3_5` arch #2329 needs. After the repair, a full sweep confirmed sympy
was the only casualty (torch, torch.nn, transformers,
transformers.models.auto.modeling_auto, accelerate, safetensors,
huggingface_hub, numpy, scipy, tokenizers, vllm all import OK).

> **CORRECTION (2026-08-18) — "sympy was the only casualty" is RETRACTED as
> false assurance from a hand-picked sweep** (#2329 `epm:progress` v94). The
> sweep quoted above tested a self-chosen list and declared the venv clean; it
> was not. An **exhaustive** import test over every top-level package in
> site-packages — 210 of them — found **two** failures: `anthropic`
> (`No module named 'anthropic.types.shared'`, which then blocked #2329's
> TF-margin leg at import time) and `psycopg` (`no pq wrapper available`;
> needs the binary extra, unrelated and imported by nothing on that path).
>
> **This bears directly on Acceptance criterion 1 below**, and is the single
> most load-bearing fact for whoever implements this task. `anthropic` has
> *complete* metadata, so a metadata-resolution check passes it by
> construction; it is also not in the load-bearing stack enumerated in surface
> 1 above, so a curated import probe never reaches it. A design that checks
> only metadata plus a short curated import chain therefore catches **1 of the
> 2 demonstrated casualties on the founding pod** — and misses the one that
> actually blocked the run. Note also that the 210-package exhaustive sweep
> was *executed pod-side*, so "import every top-level package" is a
> demonstrated-feasible instrument, not a hypothetical one; its wall-time is
> the open question, not its viability.
>
> Generalizable lesson recorded by #2329's author: when repairing a class of
> damage, sweep exhaustively over the population, not over the members you
> happen to suspect.

## Acceptance

- A pod venv with a metadata-less or partially-copied lock-pinned package FAILS
  preflight, with the leaf import error surfaced (not the lazy-wrapper text).
- A healthy pod venv still passes, with no meaningful added wall-time.
- The hardlink-fallback warning no longer appears on pods (or is documented as
  expected and suppressed), so it stops masking genuine install problems.

**Provenance:** surfaced by the #2329 orchestrator (`/issue 2329`, workflow v2)
while recovering the deferred TF-margin secondary DV; full incident detail in
#2329 `events.jsonl` marker `epm:progress` v92.
