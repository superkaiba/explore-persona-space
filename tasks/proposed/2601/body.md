---
title: SSH-remote invocations orphan remote processes on local tool timeout; concurrent
  uv run env-sync guts pod venv into a silent namespace package
kind: infra
tags: []
created_at: '2026-08-26T06:08:34Z'
has_clean_result: false
workflow: v1
---
# SSH-remote invocations orphan remote processes on a local tool timeout, and a concurrent `uv run` env-sync can GUT a pod venv — no guard, and the surviving damage imports as a namespace package

## Goal

Close two linked gaps that cost a pod venv and ~30 minutes of a live 3-pod run: (1) an `ssh <pod> ... uv run ...` whose LOCAL client dies to a tool timeout leaves the REMOTE payload running, and nothing warns or checks; (2) a second concurrent `uv run` on that pod triggers an env-sync that can leave a package half-installed, which then imports SUCCESSFULLY as an empty namespace package and fails much later with misleading `AttributeError`s.

## The gaps

**Gap 1 — the kill-before-relaunch rule does not carry to the SSH-remote shape.** `.claude/rules/crash-fix-rounds.md` § Kill-before-relaunch says "a timed-out / abandoned Bash TOOL call kills the SHELL but ORPHANS the python child ... Applies on the shared VM and on pods alike." It is written for a LOCAL child. In the SSH shape the orphan is on the far side of the connection: killing the local `ssh` client does not signal the remote payload, and no local `pgrep` can see it. An agent that reads the rule and applies it to local processes — which is the natural reading — is still exposed. The rule needs the remote case named, with the remote-side probe (`ssh <pod> 'pgrep -af "<bracketed pattern>"'`) as the check.

**Gap 2 — nothing prevents a concurrent env-sync, and nothing detects the damage it leaves.** `uv run` performs an implicit env-sync when the lockfile and environment disagree. Two concurrent `uv run` invocations on one pod can therefore race inside site-packages. `.claude/rules/gotchas.md` already records the adjacent #1689 case ("an N-way parallel `uv run` worker fan-out ... storms the FUSE mount"; remedy: "multi-worker pod launchers export `UV_NO_SYNC=1`") — but that remedy is scoped to multi-worker LAUNCHERS. Preflight and ad-hoc probe invocations, which are exactly what an orchestrator and an experimenter both issue against the same pod, carry no such guard.

The damage mode is the nastier half: a half-installed package can lose its `__init__.py` while keeping its subdirectories and its `dist-info`. Python then imports it as an implicit NAMESPACE package — `import transformers` SUCCEEDS with `__file__ = None` — so the failure surfaces later, at whatever attribute read happens to run first, with an error naming the wrong thing.

## Incident (#2546, 2026-08-26)

Three-arm run, one 4xH100 pod per arm. Orchestrator ran preflight over arm-2 and arm-3 in ONE serial shell loop; the local Bash call hit its 2-minute tool timeout partway through. The orchestrator recorded arm-3 as "preflight cut, not yet run" — but the remote `timeout 600 uv run ... preflight` kept running. The arm-3 experimenter then started its own preflight; its `uv run` env-sync raced the orphan.

Result on `pod-2546-arm3`: `transformers/` left with six bare subdirs (`models onnx pipelines quantizers sagemaker utils`), NO `__init__.py`, beside an intact `transformers-4.57.6.dist-info`. Two consecutive preflight runs then died inside `check_vllm_transformers_compat` with errors that name the wrong culprit:

```
run 1: AttributeError: module 'torch' has no attribute '_inductor'
run 2: AttributeError: module 'transformers' has no attribute '__version__'
```

torch / vllm / peft / trl were intact and matched `uv.lock`; site-packages mtimes (05:39-05:50 UTC) bracketed the race window. The experimenter correctly refused to launch and posted `epm:failure` (`failure_class: infra`, `reason: provision-incomplete`), which is the system working — but the pod sat idle at ~$16/hr and the venv needed a full rebuild.

Aggravating detail worth encoding: a later orchestrator probe of that pod SAW live `uv run python -c "import torch..."` processes and attributed all of them to the experimenter's diagnostic. Part of that set was the orphan. Seeing processes is not the same as knowing whose they are.

## Fix (proposed; implementer to confirm shape)

1. **`UV_NO_SYNC=1` on non-installing remote invocations.** Thread it into preflight and ad-hoc probe invocations against a pod (the orchestrator's, `experimenter.md`'s pre-launch protocol, and `orchestrate/preflight.py`'s own documented call form). A preflight is a READ of the environment; it has no business mutating it. This alone removes the race. Deliberately do NOT set it on the bootstrap path that is SUPPOSED to install.
2. **Package-integrity check in preflight.** Its existing deep-import probe should assert that key packages resolve to a real `__init__.py` (equivalently: `mod.__file__ is not None`) rather than importing as a namespace package. That converts this failure class from a late misleading `AttributeError` into a named, immediate diagnosis. Cheap: a handful of `importlib.util.find_spec` / `__file__` checks.
3. **Extend `.claude/rules/crash-fix-rounds.md` § Kill-before-relaunch with the SSH-remote case**: a local tool timeout does not kill a remote payload; probe the far side with a bracketed `pgrep -af` before re-running or dispatching against that pod, and prefer BACKGROUND-bounded remote calls so a local timeout cannot orphan them.
4. **Consider a serial-loop caveat.** A single Bash call looping preflight over N pods is inherently timeout-prone (N x minutes against a 2-minute default). One call per pod, or background-bounded, is the safer composition; worth a line wherever the multi-pod preflight pattern is described.

## Verified recovery, for the runbook

The repair that worked is already in `.claude/rules/gotchas.md` § "Pod venv rebuilds", trap 3, and should be cross-referenced from the preflight failure path: move the broken venv aside (preserve forensics, do not `rm`), then `UV_PROJECT_ENVIRONMENT=/root/eps-venv uv sync --locked --python /usr/bin/python3.11`, then symlink `.venv` -> `/root/eps-venv`. Roughly one minute, versus the multi-hour ~1.3 MB/s FUSE rebuild that trap 2 documents for an in-place `/workspace` venv. Check the `/usr/local/bin/python` shim is uv-free first (trap 1), and grep the drivers for `attn_implementation` / `flash_attn` before spending minutes reinstalling flash-attn a `uv sync` dropped — it is frequently unused.

## Scope

- `src/explore_persona_space/orchestrate/preflight.py` — namespace-package integrity assertion; document the `UV_NO_SYNC=1` call form.
- `.claude/agents/experimenter.md` — pre-launch protocol: remote-orphan probe + `UV_NO_SYNC=1`.
- `.claude/rules/crash-fix-rounds.md` — § Kill-before-relaunch: name the SSH-remote orphan case.
- `.claude/rules/gotchas.md` — extend the #1689 `UV_NO_SYNC=1` note beyond multi-worker launchers to any concurrent `uv run` on one pod; record the namespace-package damage signature.
- A regression test for the integrity check: a fixture package with subdirs and dist-info but no `__init__.py` must FAIL the assertion.

## Provenance

Found by the `/issue 2546` orchestrator after the arm-3 experimenter refused to launch and correctly traced the venv damage to an orchestrator-orphaned remote preflight. The orchestrator confirmed and owns the cause (`epm:progress` v66 on #2546). The experimenter's own failure-lesson block flagged `gotcha_candidate: yes` for this class.
