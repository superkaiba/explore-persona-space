---
name: bank-r-artifact-schema-drift
description: HF-cached persona_bank.json and on_policy_R/R_*.json on the issue_472 prefix are not pinned to the same bank snapshot (60 vs 61 personas, 15/16 disjoint); KeyError fires mid-eval AFTER train+upload. Assert set(bank)==set(R) pre-launch.
metadata:
  type: feedback
---

The issue_472 reuse artifacts on `superkaiba1/explore-persona-space-data` (`issue472_neg_geometry/`) are NOT pinned to one bank content_hash: at #477 v4 smoke (2026-06-04) the bank had 60 personas, R_eval 61, with 15 missing/16 extra. Existence/coverage gates PASS (files load fine); the desync only surfaces when the eval iterates bank personas into `R_eval['completions']` — a `KeyError` AFTER ~4 GPU-min of train + adapter upload.

**How to apply:** after pre-staging bank + R artifacts, BEFORE any launch:
```python
bank = json.load(open('data/issue_472/persona_bank.json'))['personas']
R = json.load(open('data/issue_472/on_policy_R/R_eval.json'))
R_personas = R['completions'].keys() if 'completions' in R else R.keys()
assert set(bank) == set(R_personas)
```
Repeat for R_train. On FAIL, post `epm:failure v1 failure_class: code reason: r_eval_bank_schema_mismatch` with the missing/extra lists; do NOT launch. The bank is canonical — either re-upload a matching R or re-run Phase 1 r-generate; a plan's "REUSE" claim is unsafe without both artifacts pinned to the same bank hash. Recommend the implementer add a startup schema gate (fail-loud before train, not mid-loop). Same family: [[feedback_filter_tightening_corpus_count]].
