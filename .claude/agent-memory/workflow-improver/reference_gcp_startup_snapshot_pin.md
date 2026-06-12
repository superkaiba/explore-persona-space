---
name: GCP startup-script hydra branch is byte-pinned
description: test_render_startup_script_hydra_only_byte_identical_to_pre_change_snapshot pins the hydra-only render to a fixture; startup-script changes must go in the workload_cmd branch only
type: reference
---

`tests/test_gcp_backend.py::test_render_startup_script_hydra_only_byte_identical_to_pre_change_snapshot`
pins the HYDRA-ONLY `render_startup_script` output byte-for-byte against
`tests/fixtures/issue588_gcp_startup_hydra_only.json`. The fixture was recorded
at the #588 merge-base and its docstring explicitly bans regenerating it
(would make the test tautological).

**How to apply:** any change to the GCE startup script in
`src/explore_persona_space/backends/gcp.py` must be scoped to the
`if spec.workload_cmd:` branch (or to shared lines ONLY if the fixture-pinned
hydra render is genuinely meant to change, which needs a deliberate
fixture-regeneration decision — architectural, not a background fix). The #601
detached-pid-wait fix was scoped this way and all 87 tests stayed green.
Bonus: the hydra branch is blocking by construction (in-process train.py), so
scoping daemonization guards to workload_cmd is also semantically right.
