---
name: stub-decoder-hook-probe
description: "Certify decode-step/position hook subclasses by driving the REAL class's _edit_tensor through a minimal model.model.layers nn stub — no GPU, no reimplementation; pairs with a fake-records flush probe for trace/telemetry wrappers (#2254 firstk R1 g1)"
metadata:
  type: feedback
---

When a diff adds a per-decode-step / position-window hook (a `DeltaHook`
subclass, an edit-position recorder, a duck-typed hook wrapper), do NOT
review the window arithmetic by reading alone — build a ~10-line stub
satisfying the block resolver (`class Stub: self.model.layers =
ModuleList([Identity()]*28)` passes `_resolve_decoder_blocks`) and drive the
COMMITTED class's `_edit_tensor` through simulated prefill `(B,T,H)` +
decode `(B,1,H)` forwards. Asserting the realized edit-index SETS per arm
(tok2→{2}, span→{1..3}, combined→{0,1..3}), per-draw reset reproducibility,
and the fail-loud rejections (multi-position decode, wrong prompt len,
armed base modes) settles the whole review-focus list in one CPU probe.
For trace/flush wrappers, fill the recorder's `records` list by hand and
assert the flushed trace's coordinate pairing — no forwards needed.

**Why:** #2254 first-k R1 g1 — the probe certified 1-indexed decode
counting, prefill latch, and combined-arm semantics in minutes on the real
class; a read-only review of the same code would have had to trust the
docstring, and a reimplemented probe would certify the reimplementation.
**How to apply:** any commit whose review focus names "window semantics" /
"position identity" / "per-draw reset". Also probe parent-parity claims by
diffing the sibling factory line-by-line (alphas, dtype cast, normalize)
rather than trusting "mirrors the parent". Related:
[[duck-typed-stack-telemetry-exactness]], [[new-dial-missing-from-resume-regime]].
