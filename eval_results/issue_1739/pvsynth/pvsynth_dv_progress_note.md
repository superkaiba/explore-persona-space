pvsynth rung — judge + DV leg complete (rung=pvsynth, split=eval).

DV: graded 0-100 trait eval_prompt rubric, 3 draws @ temp 1.0, judge claude-sonnet-4-5-20250929, max_tokens 400 (re-judge 800).

- evil: contexts_with_dv=200/200 groups=10 mean=22.326750000000008 sd=34.87632295870317 min=0.0 max=99.66666666666666 content_drops=41 transport_losses=0 rejudge_recovered=1
- sycophancy: contexts_with_dv=200/200 groups=10 mean=17.9525 sd=20.394313764370647 min=0.0 max=85.0 content_drops=0 transport_losses=145 rejudge_recovered=5
- hallucination: contexts_with_dv=200/200 groups=10 mean=54.684305555555575 sd=36.43860846146838 min=0.0 max=100.0 content_drops=701 transport_losses=0 rejudge_recovered=63

DEVIATION: hallucination uses the trait rubric here (the PV eval questions carry no reference answers, so the train-side three-way alias-match/fabrication DV is inapplicable) — dv_construct is recorded per behavior; do not pool the two constructs.

REMAINING: arm scoring (the 6-arm transfer roster x both variants x 28 layers) needs the behavior's TRAIN capture store refit (32-70 GB/behavior on HF) and must NOT run on the shared VM — route it to a pod / big-disk CPU lane.
Artifacts: issue1739_ctxmap/pvsynth/{raw_completions,capture_store,dv_dataset,judge,spread}
