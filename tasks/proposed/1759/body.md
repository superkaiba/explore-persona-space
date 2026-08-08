---
title: 'daily-held: #1689 Phase-D disposition — descoped CIs + 44h w'
kind: infra
tags:
- daily-held
- needs-human
created_at: '2026-07-28T07:03:56Z'
has_clean_result: false
origin_prompt: '/daily 2026-07-27 problem sweep (route 3): #1689''s Phase-D ladder
  bootstrap was descoped mid-run without a plan amendment:'
workflow: v1
---
## Held decision (needs Thomas)

Filed by /daily 2026-07-27 problem sweep as a route-3 judgment call.
**Carve-out item:** scientific-meaning change + spends money/compute

#1689's Phase-D ladder bootstrap was descoped mid-run without a plan amendment:
headline CIs now rest on 10 bootstrap / 2 null draws (pre-registered: 1000/40;
R15-marker-documented: 200/40) and L19-only vs 4 planned layers. The R16
vectorized port (torch fp64, 16 workers, GPUs now saturated) still projects
~44h more wall on the 4xH100 pod (~$10-12/hr class), documented `continue_as_is`
in compute-deviation v4. Decisions only you can make:
1. Accept the descoped draws for the clean-result (with a planned-vs-actual
   caveat), or order a re-bootstrap at B>=1000 on the winning pairs (now cheap
   after the ~10x vectorization) before promotion.
2. Keep the 4xH100 pod for the remaining ~44h vs persist + downsize.
3. The instruct-leg point estimates are FULL quality regardless of draws
   (computed once pre-draw-loop) — a prior in-chat recommendation to terminate
   that leg was wrong and withdrawn.
Evidence: #1689 epm:compute-deviation v3/v4, epm:progress v64 audit,
gpu-idle-escalation x3; miners A P1, B P2, I P1-P4, J P1.
Suggested action: answer 1-2 on the task; the owning session executes.
