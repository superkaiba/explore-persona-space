---
title: 'GCE workload secrets: migrate from instance metadata to Secret Manager pull-from-VM
  with narrow IAM'
kind: infra
tags: []
created_at: '2026-06-10T22:46:26Z'
has_clean_result: false
parent_id: 535
---
---
kind: infra
---

# GCE workload secrets: migrate from instance metadata to Secret Manager

Carries the open concern `gcp-secrets-secret-manager-migration` from #535
round 2. Today the GCP backend delivers workload secrets (HF_TOKEN,
WANDB_API_KEY, ...) as custom instance metadata via a combined
`--metadata-from-file` flag (0600 tempfiles, unlinked in a `finally` —
fix landed in #535 r2 at 3269c7639, removing tokens from the create
argv/process list). The residual exposure: metadata values are readable
by any principal with `compute.instances.get` in project
`eps-persona-gpu-jun2026` (currently the eps-router SA + the project
owner — bounded, documented at `src/explore_persona_space/backends/gcp.py:825`).

**Ask:** replace the metadata channel with Secret Manager — store the
workload secrets as Secret Manager secrets, grant the VM's service
account `secretmanager.secretAccessor` on exactly those secrets, and
have the GCE startup script pull them at boot (replacing the
`/computeMetadata/v1/instance/attributes/<KEY>` curl stanza). Update
`render_startup_script`, `render_create_argv`, the secret-threading
tests in `tests/test_gcp_backend.py`, and the gcp.py:825 boundary
comment. Acceptance: tokens appear neither in create argv nor in
instance metadata; a live GCP smoke lane still passes 4/4.
