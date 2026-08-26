---
name: embedded-smoke-leg-production-chain
description: A production phase chain that RUNS a smoke leg internally defeats dispatcher-level smoke out-root splits; the resume fingerprint must carry the smoke dial in the SCRIPT (#2546 R1 g2)
metadata:
  type: feedback
---

When a dispatcher's `all` chain runs a gating smoke leg as its FIRST production
phase (`run_phase p1_smoke` → `driver --smoke --out-root "$OUT_ROOT"`), an
out-root split keyed on the DISPATCHER-level smoke flag does not protect the
production root: the embedded smoke leg writes smoke-sized rollouts, stores,
and done markers into the production out-root, and every later production
phase resume-skips onto them whenever the resume fingerprint lacks a
smoke/row-set component. #2546 R1 g2 (`issue2546_gen_capture.py` +
`issue2546_dispatch.sh:216`): pilot walls read ≈0, P2–P4 no-op'd, first loud
failure deferred to the P5 fit floor.

**Why:** the sibling memories ([[smoke-shard-namespace-only-done-files]],
[[new-dial-missing-from-resume-regime]], [[start-manifest-stale-artifact-done]])
cover missing dials and unnamespaced artifacts; the new wrinkle is the CALLER
topology — a dispatcher-level out-root split looks like the defense but the
production chain itself invokes the smoke, so only a SCRIPT-side key/namespace
fix defends all callers.

**How to apply:** on any rig with `--smoke` + fingerprint-keyed resume, (1)
diff the smoke-run fingerprint against the production fingerprint field by
field — identical ⇒ Critical; (2) check whether local artifact paths are
smoke-namespaced (HF-dest namespacing alone is the tell that the concern was
half-applied); (3) read the DISPATCHER's chain for an embedded smoke leg run
under the production out-root — the out-root split's flag scope must cover it.
