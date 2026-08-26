---
title: Pod bootstrap cone sparse-checkout omits CROSS-ISSUE artifact cones, so drivers
  reading prior issues' committed eval_results/figures crash at launch
kind: infra
tags:
- pod-bootstrap
- sparse-checkout
created_at: '2026-08-26T19:58:28Z'
has_clean_result: false
parent_id: 2569
origin_prompt: 'epm:failure-lesson surfaced by #2569 Step 6d.1 experimenter: both
  pods'' first launch attempt died in seconds because the cone sparse-checkout hid
  eval_results/issue_{1482,2476,1979,779}; gotcha_candidate: yes, generalizes: yes'
workflow: v1
---
# Pod bootstrap's cone sparse-checkout omits CROSS-ISSUE artifact cones, so any driver reading a prior issue's committed eval_results/figures crashes at launch

`kind: infra`. Found during #2569's Step 6d.1 launch: the first launch attempt on
BOTH pods (`pod-2569-rows`, `pod-2569-xm`) died within seconds because the drivers
read committed artifact trees belonging to OTHER issues, which the pod's sparse
checkout structurally hides.

## Mechanism (probed, not inferred)

`scripts/bootstrap_pod.sh:326-350` deliberately uses a partial clone + **cone**
sparse-checkout (#2051) because the full repo carries ~10.5 GB. Line 345 sets the
cone to:

```
git sparse-checkout set src scripts configs tests docs data \
    "eval_results/issue_$ISSUE_VAL" "figures/issue_$ISSUE_VAL"
```

So the cone gets the code tree, `data`, and **only the CURRENT issue's** artifact
directories. Verified live on `pod-2569-rows`:

```
$ git -C /workspace/explore-persona-space sparse-checkout list
configs
data
docs
eval_results/issue_2569
figures/issue_2569
scripts
src
tests
core.sparseCheckout=true   core.sparseCheckoutCone=true
```

That is exactly the documented behavior. The gap is that a round's drivers
routinely read PRIOR issues' committed artifacts as carry-over inputs — #2569
reads `eval_results/issue_1482`, `issue_2476`, `issue_1979`, `issue_779` — and
every one of those paths is invisible on the pod. The workload does not degrade;
it dies `FileNotFoundError` seconds after launch.

## Why the existing guards do not catch it

`bootstrap_pod.sh:367-375` already runs a post-checkout audit with TWO warnings:

- line 370: own-issue artifact cones missing (`eval_results/issue_$ISSUE_VAL`)
- line 373: the default `data` cone missing (#2211)

Both warnings name the exact remedy. Neither covers cross-issue cones, so the
audit prints a clean bill of health for a checkout that is missing every
carry-over input the round actually consumes. The failure therefore surfaces at
LAUNCH (loud, on the pod, after provisioning + bootstrap + venv work is already
paid for) rather than at BOOTSTRAP (cheap, with a named remedy).

## The affordance already exists and nothing computes it

`BOOTSTRAP_EXTRA_CONES` is a documented env var (`bootstrap_pod.sh:32`, applied at
:290-292 and :349-350) that adds arbitrary extra cones. The fix is not new
machinery — it is threading information the workflow ALREADY has:

- The approved plan enumerates its carry-over inputs, and Step 6a.5 runs
  `verify_carryover_inputs.py` against exactly those prior-issue prefixes on the
  VM. That prefix set is precisely what `BOOTSTRAP_EXTRA_CONES` needs.
- Nothing currently carries it from the plan to the provision call, so the pod is
  bootstrapped blind to the round's own declared dependencies.

## Proposed fix (for the planner to scope)

Two complementary legs; either alone is an improvement, both together close it:

1. **Thread the cones.** Have the provisioning path derive the cross-issue
   `eval_results/issue_<M>` / `figures/issue_<M>` prefixes from the plan's
   carry-over declaration and pass them as `BOOTSTRAP_EXTRA_CONES`. Preferred —
   it makes the correct checkout the default rather than a thing each session must
   remember.
2. **Extend the post-checkout audit.** Add a third warning at `:367-375` that
   greps the round's drivers (or the plan's carry-over list) for
   `eval_results/issue_<M>` / `figures/issue_<M>` references with `M != ISSUE_VAL`
   and warns per missing cone, naming the `git sparse-checkout add` remedy — the
   same shape as the two warnings already there. This is the fail-loud backstop
   for rounds whose dependencies are not declared in the plan.

A preflight check is the natural home for leg 2 if the audit is the wrong layer.

## Manual remedy used this round (for reference)

```
git -C /workspace/explore-persona-space sparse-checkout add \
    eval_results/issue_1482 eval_results/issue_2476 \
    eval_results/issue_1979 eval_results/issue_779
```

Both pods then launched successfully; attempt-1 logs preserved as
`/workspace/logs/issue-2569.attempt1.log` on each pod.

## Distinctness from #2606 (do NOT dedupe onto it)

#2606 is the provisioning path reporting `rc=0` + `BOOTSTRAP-OK` OVER a
`PREFLIGHT-FAILED` line — a verdict-propagation defect, reproduced twice this
round with two different underlying causes (errno-116 venv; behind-origin/main).
THIS task is a checkout-completeness defect: the cone set is computed without
reference to the round's cross-issue inputs. Same file family
(`bootstrap_pod.sh`), different bug, different fix. Per the workflow-fix-on-bug
dedup rule (`(target_file, candidate-fingerprint)`), a distinct bug on the same
file files its own task.

## Provenance

Surfaced as an `epm:failure-lesson` from #2569's Step 6d.1 experimenter
(`gotcha_candidate: yes`, `generalizes: yes`) — it generalizes to every pod round
with carry-over inputs, which is most of them.
