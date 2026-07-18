---
title: Rank profile of the orthogonal carrier — what structure carries the 90%?
kind: experiment
tags: []
created_at: '2026-07-18T15:56:55Z'
has_clean_result: false
parent_id: 1072
workflow: v1
goal: Characterize the rank structure of the orthogonal carrier of the own-answer
  advantage at layer 26 on the frozen Qwen-2.5-7B-Instruct LMSYS pool — testing whether
  the top-64 eigendirections of the pooled train-fold answer-activation covariance
  carry more than half of the own-vs-external R² gap (a compact content-summary subspace)
  or the gap is diffuse across the 3,583-dimensional complement — via held-out per-eigendirection
  gap contributions with the rank selection riding per bootstrap draw.
---
## Goal

Characterize the rank structure of the orthogonal carrier of the own-answer advantage at layer 26 on the frozen Qwen-2.5-7B-Instruct LMSYS pool — testing whether the top-64 eigendirections of the pooled train-fold answer-activation covariance carry more than half of the own-vs-external R² gap (a compact content-summary subspace) or the gap is diffuse across the 3,583-dimensional complement — via held-out per-eigendirection gap contributions with the rank selection riding per bootstrap draw.


## Provenance

Filed (filed-only, never auto-spawned) from `epm:follow-ups v2` proposal 1 on parent #1072 at the post-lowdim-fold re-park, after the follow-up-critic redundancy screen returned not-redundant (epm:followup-value-critique v2; single-Claude — Codex quota sentinel live). Cheap-band round cap state at filing: 1/2 used; this proposal is question_relation: substantially-different, so it files as a child regardless of cost.

## Proposal (verbatim, rank 1 of epm:follow-ups v2)

### 1. Rank profile of the orthogonal carrier — what structure carries the 90%? — Type: Diagnostic (new question)

**Parent:** #1072
**question_relation:** substantially-different
**Goal:** Characterize the rank structure of the orthogonal carrier of the own-answer advantage at layer 26 on the frozen Qwen-2.5-7B-Instruct LMSYS pool — testing whether the top-64 eigendirections of the pooled train-fold answer-activation covariance carry more than half of the own-vs-external R² gap (a compact content-summary subspace) or the gap is diffuse across the 3,583-dimensional complement — via held-out per-eigendirection gap contributions with the rank selection riding per bootstrap draw.
**Hypothesis:** The 90% orthogonal-carried layer-26 gap lives in a moderately concentrated but not low-rank subspace: top-64 target-covariance eigendirections carry >50% of `ΔR²_full`. The per-context breadth (61–62% of contexts individually complement-dominant across both rounds) weakly favors diffuseness.
**Falsification:** Top-64 directions carrying <20% of `ΔR²_full` (95% CI wholly below) kills the compact-content account and establishes the carrier as diffuse/high-rank. Either outcome is load-bearing: "rank-≲64 content summary" redirects the line toward overlapping that subspace with known semantic/persona axes; "diffuse" rules out ANY small-subspace account and redirects toward map-level characterizations.
**Differs from parent:** Exactly one thing — the read-out decomposition: {token-identity subspace + complement} becomes per-eigendirection gap contributions in the eigenbasis of the pooled TRAIN-fold target covariance (fit on train folds, read on test folds; top-r selection rides per bootstrap draw per the selection-symmetric-nulls rule); layer 26 only.

**Retag justification (v1 tagged this `same`):** the routing litmus fails the verbatim-Goal test. The parent Goal — "determine whether the advantage is carried by the next-token-identity component…" — is now ANSWERED (falsified, 1-D and low-D, HIGH confidence); a rank profile does not test token identity at all and its result would start its own headline about what the complement contains, i.e. it changes the `## Goal` / open-questions anchor rather than rewriting an existing Takeaways bullet. This is the "pivot to characterizing the structure the result exposed" shape (worked example 5 in the proposer spec), so it files as a child task, not a same-issue round.

**Pre-filled spec (from parent):**
- Model: same (frozen Qwen2.5-7B-Instruct; no new generation — stored L26 slots suffice for targets; one refit pass on GPU)
- Data: same 4,920-context LMSYS pool × 4 frozen conditions at pinned #952 rev `5b62649c…` (verified above); matched population n = 3,188
- Seeds: fold split rng(952), bootstrap rng 0 (10,000 draws), same
- Eval: same sufficient-statistics battery extended to per-eigendirection SS channels; same paired own-vs-plain-external contrast; prefix + context mapping arms both inherited (the prefix arm is degenerate by construction on this single-turn pool — constant rendered prefix — and is reported, not skipped, as in the parent)
- Config: same EXCEPT the read-out basis (eigenbasis rank profile) and layer restriction to 26; maps + λ* frozen from the full-target cells

**Estimated cost:** ~3 GPU-h on 1× A100-80 (GCP auto lane): stage 4 L26 slot tensors (~10 GB, off-VM per the footprint rule), refit/replay the 20 (condition × fold) L26 full-target cells, eigh at d = 3,584 (trivial), one streaming per-direction SS pass.
**If it works (concentrated):** a nameable low-rank content subspace exists; next step overlaps it with known semantic/persona axes — the "distributed-content read" sharpens to "low-rank content summary".
**If it fails (diffuse):** the advantage is genuinely high-rank — rules out every small-subspace account (token-identity from this task, compact-content from this child) and redirects the line to map-level characterizations.

**auto_run:** yes
**auto_run_reason:** closed-form linear-algebra recipe on positively verified stored tensors with pre-registered rank thresholds — no design fork needs human scoping; as `substantially-different` this routes to FILED-as-`proposed`-child for manual triage (never auto-spawned), so tagging yes spends no compute.

**cost_class:** needs-gpu
**headline_affecting:** no
**est_gpu_hours:** 3

---
