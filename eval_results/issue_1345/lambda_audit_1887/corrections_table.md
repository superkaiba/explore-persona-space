# #1887 lambda-selection corrections table

Replay gate: **PASS** (0 of 67 referenced cells failed |dR2| <= 0.001).
Headline corrected read: reduced-basis (unselected). Forced-lambda is diagnostic only. Nulls: false — committed nulls selector-matched to the committed read only. CIs: point-estimate-only — committed CI machinery not re-run.

| cell | variant | n_train | d | committed | replay Δ | capped 0.9 | inner-CV | reduced (headline) | forced 1e2/1e3/1e4 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| base__R_base_r1_context | base | 3779 | 3584 | 0.5416 | -0.0000 | 0.5416 | 0.5546 | 0.5165 | 0.3456/0.5416/0.5321 | stable |
| base__R_base_r1_prefix | base | 3779 | 3584 | 0.1316 | 0.0000 | 0.1316 | 0.1316 | 0.1311 | 0.1310/0.1311/0.1316 | indeterminate (within fold SE) |
| base__R_base_r2_context | base | 3779 | 3584 | 0.5783 | -0.0000 | 0.5783 | 0.5843 | 0.5487 | 0.4259/0.5783/0.5600 | stable |
| base__R_base_r2_prefix | base | 3779 | 3584 | 0.1297 | 0.0000 | 0.1297 | 0.1297 | 0.1277 | 0.1281/0.1297/0.1229 | indeterminate (within fold SE) |
| base__R_instruct_r1_context | base | 3779 | 3584 | 0.6542 | -0.0000 | 0.6542 | 0.6584 | 0.6356 | 0.5265/0.6542/0.6340 | stable |
| base__R_instruct_r1_prefix | base | 3779 | 3584 | 0.1339 | 0.0000 | 0.1339 | 0.1337 | 0.1335 | 0.1334/0.1335/0.1339 | indeterminate (within fold SE) |
| base__R_instruct_r2_context | base | 3779 | 3584 | 0.6249 | 0.0000 | 0.6249 | 0.6309 | 0.6044 | 0.4820/0.6249/0.6073 | stable |
| base__R_instruct_r2_prefix | base | 3779 | 3584 | 0.1283 | -0.0000 | 0.1283 | 0.1283 | 0.1262 | 0.1267/0.1283/0.1228 | indeterminate (within fold SE) |
| base__R_instruct_r3_context | base | 1678 | 3584 | -0.7541 | -0.0000 | 0.2953 | 0.2953 | 0.2396 | -0.0632/0.2569/0.2823 | estimator-artifact |
| base__R_instruct_r3_prefix | base | 1678 | 3584 | 0.1166 | -0.0000 | 0.1166 | 0.1256 | 0.0803 | -0.2974/0.0568/0.1256 | stable |
| assistant_named_story__R_base_r1_context | assistant_named_story | 3779 | 3584 | 0.5416 | -0.0000 | 0.5416 | 0.5546 | 0.5165 | 0.3456/0.5416/0.5321 | stable |
| assistant_named_story__R_base_r1_prefix | assistant_named_story | 3779 | 3584 | 0.1316 | 0.0000 | 0.1316 | 0.1316 | 0.1311 | 0.1310/0.1311/0.1316 | indeterminate (within fold SE) |
| assistant_named_story__R_base_r2_context | assistant_named_story | 3779 | 3584 | 0.5783 | -0.0000 | 0.5783 | 0.5843 | 0.5487 | 0.4259/0.5783/0.5600 | stable |
| assistant_named_story__R_base_r2_prefix | assistant_named_story | 3779 | 3584 | 0.1297 | 0.0000 | 0.1297 | 0.1297 | 0.1277 | 0.1281/0.1297/0.1229 | indeterminate (within fold SE) |
| assistant_named_story__R_instruct_r1_context | assistant_named_story | 3779 | 3584 | 0.6542 | -0.0000 | 0.6542 | 0.6584 | 0.6356 | 0.5265/0.6542/0.6340 | stable |
| assistant_named_story__R_instruct_r1_prefix | assistant_named_story | 3779 | 3584 | 0.1339 | 0.0000 | 0.1339 | 0.1337 | 0.1335 | 0.1334/0.1335/0.1339 | indeterminate (within fold SE) |
| assistant_named_story__R_instruct_r2_context | assistant_named_story | 3779 | 3584 | 0.6249 | 0.0000 | 0.6249 | 0.6309 | 0.6044 | 0.4820/0.6249/0.6073 | stable |
| assistant_named_story__R_instruct_r2_prefix | assistant_named_story | 3779 | 3584 | 0.1283 | -0.0000 | 0.1283 | 0.1283 | 0.1262 | 0.1267/0.1283/0.1228 | indeterminate (within fold SE) |
| conversation_paired_stories__R_base_r1_context | conversation_paired_stories | 3779 | 3584 | 0.5416 | 0.0000 | 0.5416 | 0.5546 | 0.5165 | 0.3456/0.5416/0.5321 | stable |
| conversation_paired_stories__R_base_r1_prefix | conversation_paired_stories | 3779 | 3584 | 0.1316 | 0.0000 | 0.1316 | 0.1316 | 0.1311 | 0.1310/0.1311/0.1316 | indeterminate (within fold SE) |
| conversation_paired_stories__R_base_r2_context | conversation_paired_stories | 3779 | 3584 | 0.5783 | -0.0000 | 0.5783 | 0.5843 | 0.5487 | 0.4259/0.5783/0.5600 | stable |
| conversation_paired_stories__R_base_r2_prefix | conversation_paired_stories | 3779 | 3584 | 0.1297 | -0.0000 | 0.1297 | 0.1297 | 0.1277 | 0.1281/0.1297/0.1229 | indeterminate (within fold SE) |
| conversation_paired_stories__R_instruct_r1_context | conversation_paired_stories | 3779 | 3584 | 0.6542 | -0.0000 | 0.6542 | 0.6584 | 0.6356 | 0.5265/0.6542/0.6340 | stable |
| conversation_paired_stories__R_instruct_r1_prefix | conversation_paired_stories | 3779 | 3584 | 0.1339 | -0.0000 | 0.1339 | 0.1337 | 0.1335 | 0.1334/0.1335/0.1339 | indeterminate (within fold SE) |
| conversation_paired_stories__R_instruct_r2_context | conversation_paired_stories | 3779 | 3584 | 0.6249 | -0.0000 | 0.6249 | 0.6309 | 0.6044 | 0.4820/0.6249/0.6073 | stable |
| conversation_paired_stories__R_instruct_r2_prefix | conversation_paired_stories | 3779 | 3584 | 0.1283 | -0.0000 | 0.1283 | 0.1283 | 0.1262 | 0.1267/0.1283/0.1228 | indeterminate (within fold SE) |
| conversation_paired_stories_assistant__R_base_r1_context | conversation_paired_stories_assistant | 3779 | 3584 | 0.5416 | 0.0000 | 0.5416 | 0.5546 | 0.5165 | 0.3456/0.5416/0.5321 | stable |
| conversation_paired_stories_assistant__R_base_r1_prefix | conversation_paired_stories_assistant | 3779 | 3584 | 0.1316 | 0.0000 | 0.1316 | 0.1316 | 0.1311 | 0.1310/0.1311/0.1316 | indeterminate (within fold SE) |
| conversation_paired_stories_assistant__R_base_r2_context | conversation_paired_stories_assistant | 3779 | 3584 | 0.5783 | -0.0000 | 0.5783 | 0.5843 | 0.5487 | 0.4259/0.5783/0.5600 | stable |
| conversation_paired_stories_assistant__R_base_r2_prefix | conversation_paired_stories_assistant | 3779 | 3584 | 0.1297 | -0.0000 | 0.1297 | 0.1297 | 0.1277 | 0.1281/0.1297/0.1229 | indeterminate (within fold SE) |
| conversation_paired_stories_assistant__R_instruct_r1_context | conversation_paired_stories_assistant | 3779 | 3584 | 0.6542 | -0.0000 | 0.6542 | 0.6584 | 0.6356 | 0.5265/0.6542/0.6340 | stable |
| conversation_paired_stories_assistant__R_instruct_r1_prefix | conversation_paired_stories_assistant | 3779 | 3584 | 0.1339 | -0.0000 | 0.1339 | 0.1337 | 0.1335 | 0.1334/0.1335/0.1339 | indeterminate (within fold SE) |
| conversation_paired_stories_assistant__R_instruct_r2_context | conversation_paired_stories_assistant | 3779 | 3584 | 0.6249 | -0.0000 | 0.6249 | 0.6309 | 0.6044 | 0.4820/0.6249/0.6073 | stable |
| conversation_paired_stories_assistant__R_instruct_r2_prefix | conversation_paired_stories_assistant | 3779 | 3584 | 0.1283 | -0.0000 | 0.1283 | 0.1283 | 0.1262 | 0.1267/0.1283/0.1228 | indeterminate (within fold SE) |
| conversation_paired_stories_assistant__R_instruct_r4_context | conversation_paired_stories_assistant | 1730 | 3584 | -0.3056 | 0.0000 | 0.4350 | 0.4350 | 0.3669 | 0.1605/0.4075/0.4180 | estimator-artifact |
| conversation_paired_stories_assistant__R_instruct_r4_op_companion_context | conversation_paired_stories_assistant | 93 | 3584 | -0.0061 | 0.0000 | 0.0682 | 0.0867 | 0.0653 | 0.0113/0.0712/0.0622 | estimator-artifact |
| conversation_paired_stories_assistant__R_instruct_r4_op_companion_prefix | conversation_paired_stories_assistant | 93 | 3584 | -0.2607 | -0.0000 | -0.0618 | -0.0618 | -0.0547 | -0.2352/-0.1415/-0.0618 | degraded-consistent |
| conversation_paired_stories_assistant__R_instruct_r4_prefix | conversation_paired_stories_assistant | 1730 | 3584 | -1.3714 | -0.0000 | 0.1412 | 0.1412 | 0.0940 | -0.4577/0.0518/0.1412 | estimator-artifact |
| conversation_paired_stories_assistant__R_instruct_r1_matched_context | conversation_paired_stories_assistant | 1730 | 3584 | 0.2426 | -0.0000 | 0.4516 | 0.6445 | 0.6089 | 0.5441/0.6445/0.6098 | shifted |
| conversation_paired_stories_assistant__R_instruct_r1_matched_prefix | conversation_paired_stories_assistant | 1730 | 3584 | 0.1313 | 0.0000 | 0.1313 | 0.1313 | 0.1318 | 0.1313/0.1313/0.1313 | indeterminate (within fold SE) |
| conversation_paired_stories_assistant__R_instruct_r2_matched_context | conversation_paired_stories_assistant | 1730 | 3584 | 0.0189 | 0.0000 | 0.6133 | 0.6133 | 0.5736 | 0.4959/0.6133/0.5839 | shifted |
| conversation_paired_stories_assistant__R_instruct_r2_matched_prefix | conversation_paired_stories_assistant | 1730 | 3584 | 0.1229 | -0.0000 | 0.1229 | 0.1229 | 0.1210 | 0.1210/0.1229/0.1137 | indeterminate (within fold SE) |
| conversation_paired_stories_assistant__R_instruct_r4_tf_on_companion_context | conversation_paired_stories_assistant | 93 | 3584 | 0.0689 | 0.0000 | 0.1063 | 0.1367 | 0.1044 | 0.0847/0.1323/0.1020 | stable |
| conversation_paired_stories_assistant__R_instruct_r4_tf_on_companion_prefix | conversation_paired_stories_assistant | 93 | 3584 | -0.2495 | 0.0000 | -0.0473 | -0.0473 | -0.0404 | -0.2248/-0.1292/-0.0473 | degraded-consistent |
| onpolicy_assistant_story__R_base_r1_context | onpolicy_assistant_story | 3779 | 3584 | 0.5416 | 0.0000 | 0.5416 | 0.5546 | 0.5165 | 0.3456/0.5416/0.5321 | stable |
| onpolicy_assistant_story__R_base_r1_prefix | onpolicy_assistant_story | 3779 | 3584 | 0.1316 | 0.0000 | 0.1316 | 0.1316 | 0.1311 | 0.1310/0.1311/0.1316 | indeterminate (within fold SE) |
| onpolicy_assistant_story__R_base_r2_context | onpolicy_assistant_story | 3779 | 3584 | 0.5783 | -0.0000 | 0.5783 | 0.5843 | 0.5487 | 0.4259/0.5783/0.5600 | stable |
| onpolicy_assistant_story__R_base_r2_prefix | onpolicy_assistant_story | 3779 | 3584 | 0.1297 | -0.0000 | 0.1297 | 0.1297 | 0.1277 | 0.1281/0.1297/0.1229 | indeterminate (within fold SE) |
| onpolicy_assistant_story__R_instruct_r1_context | onpolicy_assistant_story | 3779 | 3584 | 0.6542 | -0.0000 | 0.6542 | 0.6584 | 0.6356 | 0.5265/0.6542/0.6340 | stable |
| onpolicy_assistant_story__R_instruct_r1_prefix | onpolicy_assistant_story | 3779 | 3584 | 0.1339 | -0.0000 | 0.1339 | 0.1337 | 0.1335 | 0.1334/0.1335/0.1339 | indeterminate (within fold SE) |
| onpolicy_assistant_story__R_instruct_r2_context | onpolicy_assistant_story | 3779 | 3584 | 0.6249 | -0.0000 | 0.6249 | 0.6309 | 0.6044 | 0.4820/0.6249/0.6073 | stable |
| onpolicy_assistant_story__R_instruct_r2_prefix | onpolicy_assistant_story | 3779 | 3584 | 0.1283 | -0.0000 | 0.1283 | 0.1283 | 0.1262 | 0.1267/0.1283/0.1228 | indeterminate (within fold SE) |
| onpolicy_assistant_story__R_instruct_r4_op_companion_context | onpolicy_assistant_story | 1614 | 3584 | -0.5471 | 0.0000 | 0.3210 | 0.3210 | 0.2618 | -0.0207/0.2830/0.3070 | estimator-artifact |
| onpolicy_assistant_story__R_instruct_r4_op_companion_prefix | onpolicy_assistant_story | 1614 | 3584 | -1.3160 | 0.0000 | 0.1010 | 0.1010 | 0.0660 | -0.5037/-0.0003/0.1010 | estimator-artifact |
| onpolicy_assistant_story__R_instruct_r1_matched_context | onpolicy_assistant_story | 1614 | 3584 | 0.5259 | 0.0000 | 0.6005 | 0.6005 | 0.5671 | 0.4819/0.6005/0.5638 | stable |
| onpolicy_assistant_story__R_instruct_r1_matched_prefix | onpolicy_assistant_story | 1614 | 3584 | 0.0838 | -0.0000 | 0.0838 | 0.0838 | 0.0840 | 0.0831/0.0833/0.0838 | indeterminate (within fold SE) |
| onpolicy_assistant_story__R_instruct_r2_matched_context | onpolicy_assistant_story | 1614 | 3584 | 0.0928 | 0.0000 | 0.5677 | 0.5685 | 0.5313 | 0.4375/0.5677/0.5352 | shifted |
| onpolicy_assistant_story__R_instruct_r2_matched_prefix | onpolicy_assistant_story | 1614 | 3584 | 0.0784 | -0.0000 | 0.0784 | 0.0783 | 0.0767 | 0.0758/0.0784/0.0739 | indeterminate (within fold SE) |
| followup_cjk_excluded__R_instruct_r3_context | followup_cjk_excluded | 1582 | 3584 | -0.6521 | 0.0000 | 0.2938 | 0.2938 | 0.2420 | -0.0474/0.2574/0.2797 | estimator-artifact |
| followup_cjk_excluded__R_instruct_r3_prefix | followup_cjk_excluded | 1582 | 3584 | 0.1187 | 0.0000 | 0.1187 | 0.1258 | 0.0751 | -0.2791/0.0612/0.1258 | stable |
| story_slot_ablation__R_instruct_r1_matched_context | story_slot_ablation | 1730 | 3584 | 0.2426 | -0.0000 | 0.4516 | 0.6445 | 0.6089 | 0.5441/0.6445/0.6098 | shifted |
| story_slot_ablation__R_instruct_r4slot_anchor_context | story_slot_ablation | 1730 | 3584 | -0.3056 | 0.0000 | 0.4350 | 0.4350 | 0.3669 | 0.1605/0.4075/0.4180 | estimator-artifact |
| story_slot_ablation__R_instruct_r4slot_attrmean_context | story_slot_ablation | 1730 | 3584 | -0.2660 | 0.0000 | 0.4281 | 0.4281 | 0.3623 | 0.1741/0.4048/0.4084 | estimator-artifact |
| story_slot_ablation__R_instruct_r4slot_preans_context | story_slot_ablation | 1730 | 3584 | -0.0173 | -0.0000 | 0.5077 | 0.5077 | 0.4529 | 0.3037/0.4900/0.4852 | estimator-artifact |
| story_slot_ablation__R_instruct_r4slot_preattr_context | story_slot_ablation | 1730 | 3584 | -0.4056 | -0.0000 | 0.3730 | 0.3730 | 0.3138 | 0.0567/0.3397/0.3535 | estimator-artifact |
| story_slot_ablation__R_instruct_r4slot_prefix | story_slot_ablation | 1730 | 3584 | -1.3714 | -0.0000 | 0.1412 | 0.1412 | 0.0940 | -0.4577/0.0518/0.1412 | estimator-artifact |
| story_slot_ablation__R_instruct_r4slot_qend_context | story_slot_ablation | 1730 | 3584 | -0.3294 | -0.0000 | 0.4141 | 0.4141 | 0.3514 | 0.0994/0.3820/0.3929 | estimator-artifact |
