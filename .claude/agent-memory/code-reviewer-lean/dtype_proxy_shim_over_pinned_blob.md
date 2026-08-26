---
name: dtype-proxy-shim-over-pinned-blob
description: Review recipe for a forwarding-proxy attr swap (e.g. float16→float32) into a sha-frozen pinned module's globals — the four probes that certify it
metadata:
  type: feedback
---

When a diff swaps a forwarding proxy into a sha-pinned blob's module globals
to redirect ONE attribute (e.g. `_Fp32Torch.float16 = torch.float32` swapped
into `r.torch` around a single call — #2587 r1 g4, transcription-never-edit
convention forbids editing the pin), certify with FOUR probes on the blob
itself, never the shim's docstring:

1. **Exhaustive attr grep** — `git show <pin-sha>:<path> | grep -n <attr>`;
   every hit must be inside the intended call path's terminal use sites. A
   hit elsewhere (model load dtype, dtype comparison, buffer alloc) is
   silently redirected too. Near-miss attrs (`bfloat16` when redirecting
   `float16`) forward unchanged — check they're not the redirected name.
2. **Single module-level import of the proxied module** — `grep -n "import
   torch"` on the blob: a function-local `import torch` in the call path
   bypasses the global swap entirely.
3. **Cross-module seams** — helpers the pinned fn imports from OTHER modules
   (repo `extract_layer_activations`) run with their OWN unshimmed global;
   verify those seams don't perform the redirected cast themselves.
4. **Duck-typed cfg fields** — read the pinned fn body and list every
   `cfg.<field>` read; the caller's stub dataclass must supply exactly those.

Also judge reentrancy honestly: a module-global swap is process-global; it's
fine when the only call site is a sequential loop and shards are separate
interpreters, and the accessor is memoized (tests monkeypatching the module
object then depend on that memoization — check `_r()`-style caches).

**Why:** the shim's correctness claim ("only in-body float16 refs are the two
terminal casts") is a checkable blob property; taking it from the comment is
the trap. In g4 all four probes passed in ~3 tool calls.

**How to apply:** any commit that monkeypatches/swaps globals of a pinned or
vendored module to alter numeric behavior. Related: [[fails_pre_fix_probe_parent_commit]]
(g4's test demonstrated fails-pre-fix in-test: raw call overflows, shimmed
call finite — the strongest form).
