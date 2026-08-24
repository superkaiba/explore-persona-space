---
name: inherited-module-capability-claims-trap
description: Plans reusing parent analysis/fit code assert capabilities the module does not have (e.g. "device is call-time-parametrized") — grep the module for the claimed parameter/flag before accepting a check-(i) PASS
metadata:
  type: feedback
---

A plan reusing a parent's analysis/fit module can assert a CAPABILITY of that
module ("device is call-time-parametrized, so GPU routing needs no source
change") that a one-line grep refutes — `grep -c "<param>" <module>` = 0, no
CLI flag, allocations on the library default. The false PASS makes a
downstream routing fix (e.g. "run the probe on a GPU lane") INERT: the run
grinds on the wrong device or crashes on a device mismatch against
default-device optimizer tensors (the #763/#812 inherited-CPU-pin class).

**Why:** the claim rides in on the artifact-reuse check-(i) record, which
reviewers tend to rubber-stamp when the batched-axis leg looks right; the
device leg needs its own grep. Caught in r2 of a model-swap rerun plan only
because three verifiers independently grepped the module.

**How to apply:** whenever a plan's §10 reuse map claims an inherited module
takes a device/flag/kwarg (or "needs no source change" for a routing fix),
grep the ACTUAL module for the parameter, the CLI flag, and `.to(`/`.cuda(`/
`set_default_device` BEFORE accepting the check-(i) verdict. If absent, the
correct plan shape is: check-(i) FAIL leg recorded + the seam SCHEDULED as
source/fork work (named functions, named tensors to move, ordering before the
consuming phase) + an artifact-reuse (m) 1-group smoke on the production
device class as the binding check. Sibling of
[[inherited-analysis-code-semantics-trap]] (semantics contradictions vs
capability claims).
