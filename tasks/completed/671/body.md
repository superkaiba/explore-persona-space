---
title: 'Fix activation-extractor output_hidden_states memory accumulation (root cause
  of GCP DHCP-wedge + recurring #545 OOM): register_forward_hook on read layers +
  local-SSD .npz scratch'
kind: infra
tags: []
created_at: '2026-06-26T09:44:37Z'
has_clean_result: false
origin_prompt: 'Fix all the GCP-wedge root causes (deep-research wf_857f7cfd): the
  dominant trigger is our extractor''s output_hidden_states accumulation; fix via
  forward hooks + local-SSD npz scratch. Run autonomously via /issue.'
---
## Problem

The activation extractor calls `model(ids, output_hidden_states=True)` on **both**
base and trained models per (persona, question) while reading only layers 7/14/21
(`scripts/issue667_extract.py:419-420`, and the same pattern across the #545/#651
extraction line). This materializes **all 29 residual-stream tensors × 2 models
per forward**; with the `expandable_segments:True` allocator retaining freed
segments, resident memory **climbs iteration-to-iteration** (documented: #545 went
22→30→38 GiB across 5 rounds). Two consequences:

1. **Recurring GPU OOM** — #545 burned 5 implementer rounds on `gpu_memory_utilization`
   / batch tuning before an architectural pivot fixed the real cause.
2. **Root cause of the GCP networking wedge (#667)** — the climbing memory pressure
   drives kernel page-reclaim stalls that starve the `systemd-networkd` DHCP
   renewal → `Could not set DHCPv4 address: Connection timed out` → the
   hung-but-RUNNING zombie (deep-research run `wf_857f7cfd`; #669 is the recovery
   backstop, THIS task removes the trigger).

## Fix (per agent-memory `experiment-implementer/feedback_output_hidden_states_activation_accumulation_oom.md` + in-repo precedent)

1. **Extract via `register_forward_hook` on ONLY the needed layers (7/14/21)**, pass
   `output_hidden_states=False`, with per-iteration `del captured` +
   `torch.cuda.empty_cache()`. In-repo precedent: `analysis/representation_shift.py:70-79`.
2. **Layer-index off-by-one (critical):** `out.hidden_states[k]` = output of block
   `k-1` (`hs[0]` = `embed_tokens`). A naive hook on `model.model.layers[layer]`
   captures `hs[layer+1]`. Map: layer 0 → `embed_tokens`; layer k≥1 → `layers[k-1]`.
   The hook path MUST reproduce the current `output_hidden_states[layer+1]` read
   **bit-for-bit**.
3. **Keep a full-tuple fallback** for non-standard models / CPU test stubs
   (`if getattr(model.model, "layers", None) is None`).

## Secondary — GCE I/O decoupling (lower priority)

4. Write per-cell `.npz` to a **local SSD scratch dir** (not the network-backed GCE
   Persistent Disk) and batch-upload at cell/sweep end. GCE Persistent Disk I/O
   traverses the network, so heavy `.npz` writes saturate the same virtual NIC the
   DHCP renewal needs — the GCE-specific amplifier (deep-research `wf_857f7cfd`).

## Tests (required)

- **Bit-for-bit equivalence:** a CPU test pinning hook-path output == the
  `output_hidden_states[layer+1]` path on a hook-capable stub (exposing
  `model.model.layers` / `embed_tokens` whose modules fire registered hooks).
- **AST test** that the primary read path uses `register_forward_hook` +
  `output_hidden_states=False` (no regression to `output_hidden_states=True`).
- **Memory-non-growth** assertion across N iterations if feasible.

## Scope / acceptance

- Numerically identical activation reads (bit-for-bit test gates it); no resident
  growth across iterations.
- Generalize the fix to the shared extraction path where feasible (the same pattern
  bit #545/#651), not only `issue667_extract.py`.
- `kind: infra` — code-change path, 0 GPU (CPU stub tests), auto-completes on
  test-verdict PASS.

## Provenance

Root-caused from the #667 GCP networking-wedge investigation + deep-research
(`wf_857f7cfd`, 2026-06-25): the dominant trigger of the GCP hung-RUNNING zombie is
this extractor memory bug, which is also the recurring #545 OOM. #669 is the
recovery backstop; this task removes the trigger.
