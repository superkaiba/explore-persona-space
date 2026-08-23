# G2a v2 recalibration evidence (crash-fix round 4, 2026-08-23)

Context: the production P2 recapture on pod-2476 completed 30,000/30,000 rows and
was refused by the v1 G2a identity gate (rc=22, `epm:failure` 2026-08-23). The v1
bar was a flat row-matched cosine min >= 0.999 — the SPAN-MEAN-class bf16 bar
(gotchas.md #1005 entry), grounded on `g2m_row_cos_min = 0.999881` measured at
n = 8 SAME-machine (`eval_results/issue_1482/matryoshka_tier/m_pilot.json`,
`g2m_n_rows: 8`) — applied to a min-statistic over 30,000 SINGLE-POSITION
deep-layer states (c20 = h20[context_end], L20/28) captured CROSS-machine
(bank: GCE; recapture: RunPod H100, different batch composition). Per the #779
two-bar discipline, single-position deep-layer bf16 jitter legitimately breaches
0.999; real row-mapping/pad bugs read 0.39-0.84 (#779), real failures < 0.99
(#1005).

Files:

- `g2a_healthy_distribution.json` — the full quantile set + n_below counts +
  worst-16 rows of the refused (healthy) production distribution, computed
  read-only from `/workspace/eps_out/issue2476/recapture/vbar_store.npz`
  (n = 30,000: min 0.995397 | p0.01% 0.998946 | p0.1% 0.999561 | p1% 0.999781 |
  median 0.999933; below 0.999: 4 rows; below 0.995: 0 rows).
- `g2a_probe.json` — the #779 attribute-before-loosen fp32 re-probe: the
  worst-16 rows re-captured on the SAME pod through the production
  tokenize/capture path at bf16 (production dtype) AND fp32 (dtype override),
  with per-row cosines vs the banked m-store c20, vs each other, and vs the
  production store's span means (driver `--g2a-probe-rows 16`).

The v2 gate (driver `G2A_*` constants block, gate_version=2): flat min >= 0.995
(real-bug catcher, #779 single-position flattened bar) AND p0.1% >= 0.999 (the
span-mean-class bar at an n-robust quantile) AND median >= 0.9995 (bulk-identity
floor; #928/#1005/m-round references). This healthy distribution PASSes all
three; a mapping/pad bug (any row < 0.995) or a bulk identity drift (median /
p0.1% depression) still FAILs.
