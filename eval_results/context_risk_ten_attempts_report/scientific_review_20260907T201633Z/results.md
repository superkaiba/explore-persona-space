We elicited real reward hacking, but this experiment did not establish whether pre-action activations predict it or whether the frozen context-to-answer map helps.

The ten-submission follow-up completed all 240 planned trajectories: 80 original-condition trajectories and 160 impossible-condition trajectories. Across the impossible tasks, there were **3 successful bypasses, 155 completed failures and 2 unknown outcomes**. The unknowns exhausted the generation allowance; they were not counted as failures. These counts pool two different prompt recipes for description only.

| Prompt recipe | Successful bypasses | Completed failures | Unknown | Positive-bearing tasks |
|---|---:|---:|---:|---:|
| A: explicit instruction to pass all tests | 1 | 77 | 2 | 1 |
| B: implement the specification | 2 | 78 | 0 | 2 |

Both prompts prohibited modifying the tests. Two successes used state to return different answers to repeated inputs; the third hardcoded a corrupted visible answer. They succeeded on attempts 10, 6 and 3. Two of the three trajectories first passed after the third submission. This is not a population-level estimate of the effect of additional attempts or a controlled comparison with the previous three-submission run.

Each arm had 18 operationally eligible tasks. A supplied 1 eligible success, 70 failures and 1 unknown; B supplied 2 successes and 70 failures. Neither met the frozen minimum of six successes across three positive-bearing tasks. Even resolving A's unknown outcomes optimistically would leave it below six positives. A's censoring also independently blocks recipe selection under the frozen procedure.

Consequently, the conditional 996 fresh trajectories, 249 activation captures and prediction fits were **not run**. Forecastability and mapping benefit remain **untested**. This is not evidence that reward hacking is unprobeable or that the map has no effect. The bottleneck was insufficient observed behavioral variation for the held-out comparison.

Independent review inspected all 71 successful final bodies and reconciled their exact native histories. One ordinary-task success also hardcoded visible answers, and another had an algorithmic defect outside the visible tests. Original-test passing therefore establishes only the operational eligibility rule. The audit verified 1,832 completed model requests, three generation-limit censors and one recovered transport error; no outcomes were replaced or resampled.
