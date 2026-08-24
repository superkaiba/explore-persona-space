# Report-verifier memory index

- [v2 report gate recipe](project_v2_report_gate_recipe.md) — first v2 gate (#2162): status moves mid-gate, sidecar points cap 2000/total_points, heatmaps have no points, verify_report skips the companion
- [F_beh paired-drop + caption-n traps](feedback_fbeh_paired_drop_and_caption_n.md) — aggregate views drop 0-coherent pairs from ALL arms; caption "n across all cells" can be dataset-wide, not the figure's n
- [#2329 gate recipe + shared-y heatmap tick clobber](project_issue2329_gate_recipe_and_sharey_trap.md) — captions.json v2 schema, the per-panel recompute conventions that matched every figure to 1e-9, and the round-1 FAIL: unequal-row `sharey=True` heatmap panels silently clobber tick labels and crop rows — compare sidecar `yticklabels` count against the source row count before trusting a caption
