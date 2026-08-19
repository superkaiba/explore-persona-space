---
name: efficiency-lens-pilot-placement-and-cpu-gd-calibration
description: v2 efficiency-lens reconcile (#2329 r1) — Claude misses pilot-gate-fires-AFTER-the-phase-it-gates; Codex over-reads compute-character GPU rule categorically + flags MAY-hold-band terminal uploads
metadata:
  type: feedback
---

Three calibration patterns from the #2329 r1 efficiency-lens reconcile (v2 critique; Claude PASS vs Codex REVISE → REVISE on 1 of 3 blockers).

**1. Claude PASSes a "pilot-gated" row whose pilot fires AFTER the phase it gates (upheld, REVISE).**
A §9 row's basis marked `pilot-gated (gate K)` is only protected if gate K's registered firing point is AT OR BEFORE that phase's entry. #2329 P2 (anchors, 8 GPU-h, guessed cross-model ×1.3 basis) cited gate 4 "Generation-throughput pilot (P3 entry)" — structurally unable to protect P2; no P2 fence derivable (fences derive from the pilot, which postdates the phase). `plan-compute-sizing.md` § Per-cell fit phases: "the plan pre-registers the pilot as the phase's FIRST step" — the FIRST phase consuming the basis. An inherited parent placement does NOT cover this when the rerun introduces a new-model/regime scale factor the parent never had (the parent's anchors basis was same-model measured). A production-shape smoke block running earlier is NOT a substitute unless timing/refusal/fence-derivation are REGISTERED on it — an unregistered observable is not a gate.
**How to apply:** for every pilot-gated §9 row, check pilot-firing-point ≤ phase entry; the fix is usually free (move the pilot to the first spend-bearing phase's entry — its inputs typically exist after the capture/freeze phase).

**2. Codex reads the compute-character GPU-worthiness rule categorically — "iterative optimization can never be a CPU fit" (rejected).**
The rule's own text (pods.md compute-character carve-out) says VECTORIZE FIRST and that batching "often keeps the run on CPU"; #1768's floor triggers on the POST-vectorization wall of the fit LEG, not on the kernel class. A batched GD battery on a dedicated CPU pod is sanctioned when: entry pilot + ≥2× fence registered, parent's REALIZED venue was CPU (verify from the parent body's Compute row / deviations — #2162's P7 ran as a CPU-only chain), and artifact-reuse (m) same-device attestation holds (moving to GPU would itself introduce a new device class needing the (m) smoke). Off-critical-path walls (post-judge-SLA analysis) further weaken the blocking case.

**3. Codex flags a terminal multi-GPU upload INSIDE the 15–30 min MAY-hold band (rejected).**
pods.md GPU-width right-sizing: "A SHORT narrow phase (< ~15–30 min) MAY hold the wide pod." An 18-min (0.3 h) residual upload+verify sweep after incremental per-block uploads is within the band. Codex's proposed remedy ("hand the residual to a non-GPU lane backed by the persisted volume") does not exist on the RunPod ephemeral lifecycle — the volume dies at terminate, `stop` is non-durable (#1112), and upload-verification PASS is the terminate precondition, so the residual sweep is structurally pod-side.

Related: [[claude-concerns-on-pre-pod-launch-headline-bug]] (unenforced prose ≠ gate), [[codex-hardening-beyond-minimal-port-contract]] (Codex over-hardening family).
