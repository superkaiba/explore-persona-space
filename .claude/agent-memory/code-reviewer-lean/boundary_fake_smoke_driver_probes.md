---
name: boundary-fake-smoke-driver-probes
description: 4 probes for a production-entrypoint smoke driver with one boundary fake — call-site attr-lookup, fresh-root rerun, realized-offline grep, hub fall-through mask
metadata:
  type: feedback
---

Reviewing a smoke driver that claims "real production entrypoint, faked ONLY
at boundary X" (#2587 r2 g6, `issue2587_smoke_run.py`), four probes settled it:

1. **Patch-binds probe:** the fake must land at a MODULE-ATTRIBUTE the
   production caller resolves at call time (`from pkg import mod as M` +
   `M.fn(...)`) — grep the caller's import style AND grep the entrypoint for
   any direct `from mod import fn` that would bypass the patch. (Patching
   before `runpy.run_path(run_name="__main__")` covers from-imports too —
   they bind the already-patched attribute — but a pre-imported module
   holding the real fn would not.)
2. **Fresh-out-root rerun:** run the driver's `all` leg into a NEW out-root;
   compare realized counts to the marker. Expect sha divergence when meta
   embeds out-root paths / git state — verify the ROUND's artifacts against
   the marker digests separately (`sha256sum` on the claimed /tmp paths).
3. **Realized-offline grep:** a "no network path reachable" claim is only as
   good as the realized run — grep the CHILD log for the staging/log token of
   every hub call site (`[an] staging`); zero hits certifies the run.
4. **Hub fall-through mask (the real find):** a `resolve_rel`-style
   local-override-else-stage-from-HF resolver means a MISSING fixture file
   silently fetches instead of failing as a fixture bug — pre-production it
   404s loud, but once production artifacts exist on the prefix, fixture
   drift lets the "fully-local" smoke consume PRODUCTION artifacts and still
   PASS. Suggest asserting the staging token absent in the child log.

Also check: a boundary fake that RE-DERIVES a key grammar (custom_id
`__{idx:05d}__{comp:02d}`) is validated only on the CONSUMER side when the
real reduce eats the fake's keys; producer-side grammar changes inside the
faked function are structurally uncatchable — note, don't block. Related:
[[smoke-fixture-authored-with-consumer-keys]],
[[marker-success-command-verbatim-rerun]].

**Why:** #2587 r2 verified a BLOCKER fix (smoke-run-coverage) as genuinely
fixed via these probes; probe 4 was the only substantive residual.
**How to apply:** any smoke/driver commit whose claim includes "production
entrypoint", "zero fakes", "only fake at boundary", or "no network".
