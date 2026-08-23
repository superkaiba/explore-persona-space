# Reconciler memory index — one line per file; open the file when the hook matches the disagreement.

## Plan-stage: pre-registered verdict logic / statistics (read the first two on any plan-stage stats split)

- [Registered gate defects ≠ analyzer-recoverable](feedback_claude_gate_unit_vs_preregistered_verdict_logic.md) — data-recoverability never rescues a defective gate/kill/lattice; barred-amendment tell; 35+ incidents (#547…#614)
- [Gate-design flaw (REVISE) vs recoverable robustness read (APPROVE)](feedback_gate_design_vs_recoverable_robustness_read.md) — REVISE only on affirmative misfire / barred amendment / run-time-only capture loss
- [Efficiency lens: pilot placement + CPU-GD + terminal uploads](feedback_efficiency_lens_pilot_placement_and_cpu_gd_calibration.md) — pilot firing AFTER the phase it gates; Codex categorically GPU-routes batched GD (#2329 r1)
- [Advisory-band gaps: check the DIRECTION](feedback_claude_advisory_band_wrong_direction_vs_probe_region.md) — silent-PASS region ⊄ band = affirmative misfire (#2152 r1)
- [One-arm port: internal smoke ≠ validity](feedback_one_arm_port_capture_parity.md) — demand capture-parity vs the parent driver (#2330 r1)
- [Live-replay proposed mechanical checks vs the named offender](feedback_live_replay_proposed_mechanical_checks.md) — execute proposed regex/check/glob against the incident artifact (#869, #867)
- [Unwired lint-check plans = dead tripwire](feedback_claude_approves_unwired_lint_check_plans.md) — direct-call-only tests, no_flags wiring unpinned (#842, #891)
- [Grep the consumer/writer surface before crediting a plan claim](feedback_claude_approves_daemon_interface_read_missing_route_filter.md) — consumer-VISIBLE route (#956); [state-rewrite safety](feedback_claude_accepts_plan_state_rewrite_safety_claim.md) — grep the literal write (#908, #919); [reroute pointer trace](feedback_claude_approves_reroute_without_consumer_pointer_trace.md) — ids ≠ consuming the return (#564)
- [Plan-verbatim text vs the plan's own Must-Fix](feedback_plan_verbatim_text_vs_plan_binding_mustfix.md) — faithful-to-§4 never rescues against MF text (#870, #864)
- [Sign-blind |ρ| rule + directional narration](feedback_claude_signblind_decision_rule_as_analyzer_concern.md) — direction pin is Must-Fix, not a reporting concern (#540)
- [Cross-lens defect re-filed per lens](feedback_cross_lens_defect_refiled_per_lens.md) — home lens settled REVISE ⇒ judge the disputed lens on its own question (#546)
- [Audit acceptance anchored to own instrument](feedback_audit_acceptance_anchored_to_own_instrument.md) — "all N hits of <registered grep>" misses hand-rolled sites → REVISE (#536)
- [Merge-step plans need merge-tree run](feedback_claude_approves_merge_step_without_mergetree.md) — run `git merge-tree` yourself (#555)
- [Cherry-pick import closure + data pin](feedback_claude_plan_cherrypick_closure_and_pin.md) — grep listed files' imports + loaders' `revision=` vs stated pin (#547)
- [Orthogonal partial-state flag in inherited analyzer](feedback_claude_misses_orthogonal_partial_state_flag.md) — enumerate ALL terminal states; compute the gate stat on parent data (#546)
- [Plan-named branch with no acceptance test](feedback_claude_underclasses_unverified_branch_test_gap.md) — `except (A, B)` prescribed, one arm pinned; trace the dropped arm (#736)
- [Codex BLOCKER on unsatisfiable plan directive](feedback_codex_unsatisfiable_plan_directive.md) — hash-seed nondeterminism makes the gate unmatchable; disclosure correct (#511)
- [Codex trim-license on modal-lane control](feedback_codex_approves_trim_license_on_modal_lane_control.md) — trimming the ONLY control on the default lane → REVISE (#578)
- [Codex skips data-construction arithmetic](feedback_codex_skips_data_construction_arithmetic.md) — quotas (#543), Goal-cell coverage (#545), "weighable from text" the rig discards (#603) — trace the artifact yourself
- [Claims need the direct probe](feedback_claude_trusts_disk_assumption_framed_as_code_claim.md) — disk state = per-folder ls (#724); [caller topology](feedback_claude_trusts_caller_topology_claims_without_wrapper_trace.md) — live-execute the predicate vs wrappers (#868); [labeled split](feedback_claude_approves_labeled_split_over_disjoint_registries.md) — intersect registry keys yourself (#901)

## Claude code-review misses (FAIL-leaning calibration)

- [Plan-literal match ≠ fs-primitive semantics](feedback_claude_credits_plan_literal_misses_fs_primitive_semantics.md) — "mkdir 0700 ✓" missed exist_ok symlink-accept + rename overwrite on predictable /tmp; live-probe the adversarial shape; call-site "unconditional" ≠ callee effect (#2377 r1)
- [Guard fix on command-GLOBAL evidence flags: probe MULTI-clause records](feedback_claude_certifies_guard_fix_single_clause_only.md) — two-clause bare+pathspec record flips block→permit via cross-clause blob binding (#2371 r2)
- [Union matchers: probe legal-prefix × lead-only-family cross product](feedback_union_matcher_probe_prefix_cross_product.md) — execute the escape on both blobs (#2357 r4)
- [Split-review composites miss cross-commit plan contracts](feedback_split_review_misses_cross_commit_plan_contracts.md) — grep §9 writers + §8/§11 controls + marker claims yourself (#2330 r1)
- [Mode-dependent durability doc claims](feedback_mode_dependent_durability_doc_claims.md) — clause-local verification misses cross-clause falsification; dictate wording on a 3rd-round loop (#2326 r3)
- [Fix-round closure: expand elided plan quotes](feedback_fix_round_closure_elided_control_arm.md) — "VERIFIED FIXED" quoted §8 minus a control arm; brief never rescopes the plan (#2330 r2)
- [Persist-before-reduce is ordering, not existence](feedback_persist_before_reduce_ordering_vs_existence.md) — file exists but write is AFTER the remote judge (#906)
- [HEADLINE decision statistic never produced](feedback_claude_misses_headline_decision_statistic_not_produced.md) — PASS on other items while the registered estimator is never computed (#841, #922)
- [Silent failures under-classed](feedback_claude_underclasses_silent_failures.md) — real silent-failure bug + CONCERNS verdict → FAIL; classify by what the bug DOES; 11 incidents (incl #1098)
- [Declare→satisfy artifact contract on a NEW path](feedback_claude_plan_complete_diff_violates_lane_artifact_contract.md) — trace WHO WRITES the sentinel on the new path (#909)
- [Same-file siblings](feedback_claude_misses_same_file_siblings.md) — sibling code paths, render branches, resamplers, scripts
- [Producer/consumer literal contracts](feedback_claude_misses_producer_consumer_key_mismatch.md) — JSON key path-vs-inline, custom_id f-strings, consumer regex
- [Inherited producer's capture semantics](feedback_verify_inherited_capture_semantics_before_crediting_slot_claim.md) — read the PRODUCER's write code, not prose (#658, #812)
- [Built-but-not-wired production paths](feedback_claude_misses_dispatcher_wire_bugs.md) — args never passed, defaults neuter plan elements (#504/#517/#520); [scaffolded-not-plumbed](feedback_claude_scaffolded_pipeline_not_plumbed.md) — orphaned helpers, HALT gates never invoked (#397, #508)
- [Cross-branch Python module dep](feedback_claude_misses_cross_branch_python_module_dep.md) — lazy imports of sibling-branch-only modules; ls the worktree (#501)
- [Fix regressions](feedback_claude_misses_fix_regressions.md) — a REPLACED check can be weaker on the original Must-Fix class; replay the old bad input (#389, #554)
- [Floor-vs-raise divergence from reference](feedback_claude_misses_floor_vs_raise_divergence.md) — new script floors where the cited reference RAISES (#532)
- [In-code invariance-comment smell](feedback_claude_misses_invariant_comment_smell.md) — comment asserts what the expression doesn't deliver; trace it (#505 r2)
- [Same name, different quantity → FAIL not Nit](feedback_claude_treats_predictor_formula_mismatch_as_nit.md) — cross-arm aggregator corrupted (#518); [plan-vs-parent refinement](feedback_claude_inherits_parent_with_plan_semantic_refinement.md) (#506); [estimand governs over pseudocode](feedback_claude_excuses_estimand_divergence_via_plan_pseudocode.md) (#539)
- [Headline-poisoning bug pre-pod-launch = FAIL](feedback_claude_concerns_on_pre_pod_launch_headline_bug.md) — "fix next round before launch" is unenforced (#522)
- [Insufficient smoke evidence](feedback_claude_concerns_on_smoke_gate.md) — help/import/dry-run-only = FAIL (#492, #551); [dry-run misses CUDA init](feedback_claude_dry_run_smoke_misses_cuda_init.md) (#488); [synthetic fixture masks args-grid bug](feedback_claude_synthetic_fixture_smoke_masks_args_grid_bug.md) (#517)
- [Overcorrected critic concern](feedback_claude_overcorrects_critic_concern_to_blocker.md) — guardrail over-rejects the plan-envisioned PASS case; re-read the plan §s (#514)
- [Read the named gate's ACTUAL semantics](feedback_claude_trusts_green_tests_over_verifier_semantics.md) — check bodies vs prescribed scope; stub fixtures pin nothing (#454, #608); [nonexistent backstop semantics](feedback_claude_cites_nonexistent_backstop_semantics.md) — read the fail condition before crediting a downgrade (#594)
- [Best-effort upload made load-bearing](feedback_claude_misses_besteffort_upload_made_loadbearing.md) — new consumer of a warn-only upload + ephemeral teardown → FAIL (#613)
- [Fabricated walk-down checkmarks](feedback_claude_fabricates_rf_walkdown_checkmark.md) — rg the literal new AND prior values before believing any ✓
- [Comment-tail waiver spoofs on raw-scan guards](feedback_claude_misses_comment_tail_spoof_on_rawscan_guards.md) — replay spoof shapes yourself via Write-tool probe (#897)

## Codex code-review overreach (PASS-leaning calibration)

- [Codex FAILs pre-existing resume metadata clobber](feedback_codex_fails_preexisting_resume_metadata_clobber.md) — real walls/fences clobber, zero consumers, verdict vs own CONCERN row; PASS + operational guard (#2378 r10)
- [Codex validation-after-engine-init ordering blocker](feedback_codex_validation_after_engine_init_ordering.md) — gate topology decides: fail-louds before spend/verdict-write + strength≠ordering + parent-SHA the ordering; defer-concern the persisted BLOCKERs (#2378 r12)
- [Codex blocker on unreachable exception path](feedback_codex_blocker_on_unreachable_exception_path.md) — code-real batched-retry coupling whose trigger can't fire (dof/trace bound); check reachability arithmetic + grid-superset before upholding (#2356 r4)
- [Codex demands hardening beyond minimal-port contract](feedback_codex_hardening_beyond_minimal_port_contract.md) — registered change set + writer-reachability + execution-test the proposed fix; persist residue via raise+defer-concern (#556, #958)
- [Codex Step 0.6 literal vs purpose](feedback_codex_step_06_literal_vs_purpose.md) — decisive variable = gate topology: PASS when a demonstrated pre-launch gate runs the changed code for real (#551, #560)
- [Codex conflates marker format/existence with code](feedback_codex_conflates_marker_format_with_code.md) — marker-shape nits / stale-file marker reads / smoke-presentation nits with "diff not reviewed" → discard
- [Codex misreads git/worktree state (git-provenance family)](feedback_codex_litigates_pre_existing_in_round_n.md) — pre-existing-on-trunk, stale-worktree "deletions", cumulative diffs (#922); [gitignored artifacts as production state](feedback_codex_gitignored_artifacts_as_production_state.md) — reachability walk (#543, #570); [raw-branch-diff misses the Step 10d surgical merge](feedback_codex_raw_branch_diff_misses_surgical_merge.md) — additive checkout bounds "reverts" → PASS (#511 r1)
- [Codex over-reads plan prose](feedback_codex_overreads_plan_prose.md) — synthesized quotes, 1-D companions inflated to grids, misparsed parentheticals, contextual sentences read unconditional
- [Codex flags descriptive-read literal as registered-threshold drift](feedback_codex_descriptive_read_literal_as_threshold_drift.md) — check which deliverable the reuse-unchanged clause binds + self-describing field name + no decision-path consumer (#2333 r7)
- [Codex under-scopes the evidence base on doc/rule-file diffs](feedback_codex_evidence_base_underscoping_doc_rules.md) — checks only the designated events row, ignoring plan §Evidence-of-record code-grounding + mechanism-register framing; run the impact arrow yourself (#2338 r1)
- [Codex methodology-choice as code bug](feedback_codex_methodology_choice_as_bug.md) — implementer picked plan-listed option (b), Codex assumed (a); or Codex flags the plan's own pre-registered rule (#480, #543)
- [Codex flags the registered noise-floor/reliability statistic as a unit-mismatch bug](feedback_codex_flags_registered_noise_floor_statistic.md) — "wrong kind of quantity / wrong units" FAIL on a verdict-logic scalar (#661)
- [Codex fabricates a code citation for a silent-shrink BLOCKER](feedback_codex_fabricated_code_citation_silent_shrink.md) — `file:line` + quoted code body (`r_plus.get`/`skipped_no_rplus` emitting `status: ok`) that doesn't exist (grep 0 hits) (#667)
- [Codex fail-loud diagnostic blocker](feedback_codex_fail_loud_diagnostic_blocker.md) — side-channel failure LOUD + headline artifact preserved
- [Codex meta-test blocker on verified fix](feedback_codex_meta_test_blocker_on_verified_fix.md) — `*-negative-control-missing` on a regression test: shared-code-path mechanism catches reverts → demote to CONCERN; never a cap-3 pivot (#505)
- [Codex blocker on sandbox-unreadable pod artifact](feedback_codex_blocker_on_sandbox_unreadable_pod_artifact.md) — "could not read <path>" ≠ unverifiable: producer exit-assert + pinned revision + parent-run proof → demote. #534 r1.
- [Codex env-var orphan unreachable](feedback_codex_env_var_orphan_unreachable.md) — trace the import chain to the actual train_* entry point; zero consumer hits = dead orphan, PASS + remove-export rec. #488 r2.
- [Codex FAILs round-N on absent OTHER-plan-section wiring](feedback_codex_plan_section_in_scope.md) — verify the element is INSIDE the round brief + reachable; un-invoked plan gaps are PASS+CONCERNS, not bounces. #505 r6.
- [Codex chat-template blocker without measurement-rig cross-check](feedback_codex_chat_template_blocker_without_measurement_xcheck.md) — parent rig used the SAME apply_chat_template surface; saturation in the parent CSV is the fingerprint; inheriting it is correct. #509 r3.
- [Codex TOCTOU blocker on plan-registered re-stat mechanism → demote when residual is reversible](feedback_claude_credits_plan_literal_misses_fs_primitive_semantics.md) — but its sibling destination Critical was REAL; never discard wholesale (#2377 r1)
- [Ignored bool return vs plan-defined confirmation](feedback_codex_ignored_bool_return_vs_plan_defined_confirmation.md) — plan MF defines the success term at that branch + all real failures RAISE in the API layer ⇒ demote via defer-concern (#2184 r1)

## Clean-result-critic + interp-critic calibration

- [Claude interp under-samples NEW sample blocks](feedback_claude_interp_undersamples_new_sample_blocks.md) — one conceded mismatch in a fresh block ⇒ sweep ALL rows against BOTH candidate sources; only source-divergent rows verify anything (#2333 r2)
- [Claude clean-result-critic under-applies spec text](feedback_claude_clean_result_critic_underapplies_spec_text.md) — mechanical pre-pass PASS ≠ spec compliance; 26 BLOCKING spec-text rules incl. Lens-11 skipped-result enumeration; DISCARD list inside (#923, #833, #2333)
- [Clean-result bare-#N refs + missing reuse path (Lens 2/5)](feedback_clean_result_bare_issue_refs_and_reuse_path.md) — v4 split: bare #N in Methodology table Source col / Results caption (L2) + reuse bullet missing the permanent path (b) (L5) are grounded (#722)
- [Claude misses Lens 7 statistical framing](feedback_claude_misses_lens7_statistical_framing.md) — always re-scan for named tests (Wilson/Fisher/Mann-Whitney), ± (#378)
- [Cross-loop CI conflict (interp vs clean-result)](feedback_cross_loop_ci_conflict.md) — interp-critic-required CIs / prior-gate house style / caption CIs survive Codex's Lens-7 FAIL; add "Do NOT remove" lines. #478, #509, #464.
- [Codex misses Lens 9 raw+processed exception](feedback_codex_misses_lens9_raw_processed_exception.md) — adjacent raw + processed pair counts as ONE figure; splitting them would break Lens 11. #480 r1.
- [Codex re-litigates adjudicated content](feedback_codex_relitigates_grandfathered_regate_prose.md) — byte-unchanged adjudicated prose out of scope: diff the fold's set-body commit (#464); [replacement register after keyword ban](feedback_codex_relitigates_replacement_register_after_keyword_ban.md) — ban lists are ENUMERATED keywords not concepts (#715)
- [Claude misses silent plan deviations](feedback_claude_misses_silent_plan_deviations.md) — implementer-flagged deviations that never landed in Reproducibility/What-I-ran; diff plan vs body row-by-row (Lens 5/13). #520 r1.
- [Claude interp misframes body factual errors as confidence risk](feedback_claude_interp_misframes_as_confidence_risk.md) — confidence covers UN-addressed risk, never a body asserting the OPPOSITE of the flagged alternative. #492 r1.
- [Interp completeness vs honesty-only bar](feedback_interp_completeness_vs_honesty_bar.md) — Codex PASSes on honesty-only; verified sibling-artifact Lens-2/3 misses bearing on the headline → REVISE (#549)
- [Codex passes when sandbox blocks data](feedback_codex_passes_when_sandbox_blocks_data.md) — Lens-7 "could not fetch" disclaimers zero-weight the Lens-1 PASS; re-verify against the JSONs (#381, #601)
- [Numeric claims: re-derive from the pinned artifacts yourself](feedback_codex_recount_with_silent_normalization.md) — PLAINEST matcher; sibling coherence under ONE matcher vindicates (#833 r2); [fix WORDING ≠ artifact TRUTH](feedback_codex_verifies_fix_wording_not_artifact_truth.md) — re-derive from pinned JSONs/figures (#778); [wrong statistic / JSON-only search](feedback_codex_fails_correct_numeric_claim_wrong_statistic_or_artifact.md) — multi-number match pins the statistic; search Repro-named .pt artifacts (#920)
