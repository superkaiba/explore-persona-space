# Reconciler memory index — open the file when a hook matches the disagreement.

- [Recorded stopping rule: false claim beats NIT severity](feedback_stopping_rule_false_claim_overrides_nit_severity.md) — false claim FAILs despite all-NIT (#2263 r6/r7)

## Plan-stage: verdict logic / statistics (read first two on any stats split)

- [Twin brief excluded the disputed question](feedback_twin_brief_excluded_disputed_question.md) — PASS = non-coverage, not rebuttal
- [Registered gate defects ≠ analyzer-recoverable](feedback_claude_gate_unit_vs_preregistered_verdict_logic.md)
- [Gate object-identity unchecked](feedback_claude_gate_object_identity_unchecked.md) — read the NAMED function (#2329 r2)
- [Gate-design flaw vs recoverable robustness read](feedback_gate_design_vs_recoverable_robustness_read.md) — REVISE only on misfire
- [Amendment-round stale literals: grep numerics yourself](feedback_amendment_round_stale_literal_sweep.md) — censoring literal = BLOCKER
- [Efficiency lens: pilot placement + CPU-GD](feedback_efficiency_lens_pilot_placement_and_cpu_gd_calibration.md)
- [Advisory-band gaps: check the DIRECTION](feedback_claude_advisory_band_wrong_direction_vs_probe_region.md) — silent-PASS region ⊄ band
- [One-arm port: internal smoke ≠ validity](feedback_one_arm_port_capture_parity.md) — capture-parity vs parent driver (#2330 r1)
- [Model-WEIGHTS revision ≠ data-repo pin](feedback_model_weights_revision_vs_data_repo_pin.md) — `list_repo_commits` first
- [Live-replay proposed mechanical checks](feedback_live_replay_proposed_mechanical_checks.md) — execute the proposed regex/glob yourself
- [Unwired lint-check plans = dead tripwire](feedback_claude_approves_unwired_lint_check_plans.md)
- [Watcher pass without main() wiring test](feedback_claude_approves_watcher_pass_without_main_wiring_test.md) — isolation battery ≠ wired
- [Durable-append claims vs routed reset](feedback_claude_durable_append_ignores_routed_reset.md) — verify EVERY repo_root() mode
- [Check-addition plans: test-count invariants](feedback_claude_plan_misses_test_count_invariants_on_check_addition.md) — grep len(CHECKS)
- [Raising stub swallowed by blanket except](feedback_raising_stub_swallowed_by_blanket_except.md) — stub inside try/except→None proves nothing
- [Grep consumer/writer surface first](feedback_claude_approves_daemon_interface_read_missing_route_filter.md) (#956); [state-rewrite safety](feedback_claude_accepts_plan_state_rewrite_safety_claim.md) (#908); [reroute pointer trace](feedback_claude_approves_reroute_without_consumer_pointer_trace.md) (#564)
- [Plan-verbatim text vs the plan's own Must-Fix](feedback_plan_verbatim_text_vs_plan_binding_mustfix.md)
- [Sign-blind |ρ| rule](feedback_claude_signblind_decision_rule_as_analyzer_concern.md) — direction pin is Must-Fix (#540)
- [Cross-lens defect re-filed per lens](feedback_cross_lens_defect_refiled_per_lens.md) — judge the lens on its own question (#546)
- [Audit acceptance anchored to own instrument](feedback_audit_acceptance_anchored_to_own_instrument.md)
- [Merge-step plans need merge-tree run](feedback_claude_approves_merge_step_without_mergetree.md) (#555)
- [Cherry-pick import closure + data pin](feedback_claude_plan_cherrypick_closure_and_pin.md) — grep imports + `revision=` (#547)
- [Orthogonal partial-state flag](feedback_claude_misses_orthogonal_partial_state_flag.md) — enumerate ALL terminal states
- [Plan-named branch with no acceptance test](feedback_claude_underclasses_unverified_branch_test_gap.md) (#736)
- [Codex BLOCKER on unsatisfiable plan directive](feedback_codex_unsatisfiable_plan_directive.md) — disclosure correct
- [Codex trim-license on modal-lane control](feedback_codex_approves_trim_license_on_modal_lane_control.md)
- [Codex skips data-construction arithmetic](feedback_codex_skips_data_construction_arithmetic.md) — quotas, Goal-cell coverage
- [Codex approves destructive-helper plan](feedback_codex_approves_destructive_helper_plan_untraced_branches.md) — verify the destructive-branch triad
- [Claims need the direct probe](feedback_claude_trusts_disk_assumption_framed_as_code_claim.md) (#724); [caller topology](feedback_claude_trusts_caller_topology_claims_without_wrapper_trace.md) (#868); [labeled split](feedback_claude_approves_labeled_split_over_disjoint_registries.md) (#901)

## Claude code-review misses (FAIL-leaning calibration)

- [Kwarg-threading: --smoke fakes + sidecar reachability](feedback_kwarg_threading_inscript_smoke_fakes_and_sidecar_reachability.md) — RUN the script's own --smoke (#1901 r2)
- [AST-scanner name-binding FP + kill-clause paraphrase](feedback_ast_scanner_name_binding_and_kill_clause_trigger_domain.md) — quote STOP clauses verbatim (#2537 r1)
- [Concern closure graded against the LEDGER row](feedback_concern_closure_graded_against_ledger_row_not_fix_sentence.md) — closed on the paraphrased half (#2263 r3)
- [Security sweep misses data-into-shell splices](feedback_claude_security_sweep_misses_data_into_shell_splice.md)
- [Masked-rc fallback vs fail-loud consumer](feedback_masked_rc_fallback_vs_downstream_fail_loud_consumer.md) — trace the FIRST consumer
- [Convention defense unverified: $VAR fences](feedback_claude_convention_defense_unverified_env_var_fence.md) — verify INITIALIZING sites
- [Round-added commands vs block contract](feedback_claude_misses_block_contract_conformance_of_round_added_commands.md)
- [Duck-type discriminator probe scope](feedback_claude_duck_type_discriminator_probe_scope.md) — probe hand-CONSTRUCTIBLE satisfiers
- [Plan-literal match ≠ fs-primitive semantics](feedback_claude_credits_plan_literal_misses_fs_primitive_semantics.md) — live-probe it
- [Tracked-artifact lifecycle across phases](feedback_claude_misses_tracked_artifact_lifecycle_across_phases.md)
- [Guard fix on command-GLOBAL flags: MULTI-clause probe](feedback_claude_certifies_guard_fix_single_clause_only.md)
- [Union matchers: prefix × lead-only cross product](feedback_union_matcher_probe_prefix_cross_product.md) — execute the escape (#2357)
- [Codex twin's static pass on shell guards](feedback_codex_static_pass_misses_quoting_variants_in_shell_guards.md) — probe quote-wrapped/glued variants
- [Enumerated-fix rounds: probe the COMPLEMENT](feedback_claude_enumerated_fix_misses_complement_species.md)
- [Split-review composites miss cross-commit contracts](feedback_split_review_misses_cross_commit_plan_contracts.md)
- [Mode-dependent durability doc claims](feedback_mode_dependent_durability_doc_claims.md) — cross-clause falsification (#2326)
- [Fix-round closure: expand elided plan quotes](feedback_fix_round_closure_elided_control_arm.md) — brief never rescopes the plan (#2330 r2)
- [Persist-before-reduce is ordering, not existence](feedback_persist_before_reduce_ordering_vs_existence.md) (#906)
- [HEADLINE decision statistic never produced](feedback_claude_misses_headline_decision_statistic_not_produced.md)
- [Silent failures under-classed](feedback_claude_underclasses_silent_failures.md) — + CONCERNS → FAIL (11 incidents, incl #1098)
- [Declare→satisfy artifact contract on a NEW path](feedback_claude_plan_complete_diff_violates_lane_artifact_contract.md) — WHO WRITES the sentinel
- [Same-file siblings](feedback_claude_misses_same_file_siblings.md) — sibling code paths, render branches, resamplers
- [Producer/consumer literal contracts](feedback_claude_misses_producer_consumer_key_mismatch.md) — JSON key paths, custom_id f-strings
- [Inherited producer's capture semantics](feedback_verify_inherited_capture_semantics_before_crediting_slot_claim.md) — read the PRODUCER's write
- [Built-but-not-wired paths](feedback_claude_misses_dispatcher_wire_bugs.md) (#504/#517/#520); [scaffolded-not-plumbed](feedback_claude_scaffolded_pipeline_not_plumbed.md) (#397)
- [Cross-branch Python module dep](feedback_claude_misses_cross_branch_python_module_dep.md) — lazy sibling-branch imports (#501)
- [Fix regressions](feedback_claude_misses_fix_regressions.md) — replay the old bad input (#389, #554)
- [Floor-vs-raise divergence from reference](feedback_claude_misses_floor_vs_raise_divergence.md) (#532)
- [In-code invariance-comment smell](feedback_claude_misses_invariant_comment_smell.md) (#505 r2)
- [Same name, different quantity → FAIL](feedback_claude_treats_predictor_formula_mismatch_as_nit.md) (#518); [plan-vs-parent refinement](feedback_claude_inherits_parent_with_plan_semantic_refinement.md) (#506); [estimand governs](feedback_claude_excuses_estimand_divergence_via_plan_pseudocode.md) (#539)
- [Headline-poisoning bug pre-pod-launch = FAIL](feedback_claude_concerns_on_pre_pod_launch_headline_bug.md) (#522)
- [Insufficient smoke evidence](feedback_claude_concerns_on_smoke_gate.md) (#492); [dry-run misses CUDA init](feedback_claude_dry_run_smoke_misses_cuda_init.md) (#488); [synthetic fixture masks grid bug](feedback_claude_synthetic_fixture_smoke_masks_args_grid_bug.md) (#517)
- [Overcorrected critic concern](feedback_claude_overcorrects_critic_concern_to_blocker.md) (#514)
- [Read the named gate's ACTUAL semantics](feedback_claude_trusts_green_tests_over_verifier_semantics.md) (#454, #608); [nonexistent backstop semantics](feedback_claude_cites_nonexistent_backstop_semantics.md) (#594)
- [Best-effort upload made load-bearing](feedback_claude_misses_besteffort_upload_made_loadbearing.md) (#613)
- [Fabricated walk-down checkmarks](feedback_claude_fabricates_rf_walkdown_checkmark.md) — rg new AND prior literals first
- [Comment-tail waiver spoofs on raw-scan guards](feedback_claude_misses_comment_tail_spoof_on_rawscan_guards.md) — replay spoof shapes
- [Syntactic test pins + vacuous empty-set gates](feedback_syntactic_test_pins_and_vacuous_empty_gates.md)
- [Degenerate-token validation + files-mode scope gaps](feedback_claude_degenerate_token_validation_and_files_mode_scope.md)
- [Self-referential replay gate needs independent anchor](feedback_self_referential_replay_gate_needs_independent_anchor.md)

## Codex code-review overreach (PASS-leaning calibration)

- [Codex FAILs pre-existing resume metadata clobber](feedback_codex_fails_preexisting_resume_metadata_clobber.md) — zero round-introduced
- [Round-added duties on verbatim-adoption commits](feedback_codex_binds_round_added_duties_on_verbatim_adoption_commits.md) — `A`-status ≠ round-authored; live-list count premises (#2584)
- [Codex validation-after-engine-init ordering blocker](feedback_codex_validation_after_engine_init_ordering.md) — gate topology decides
- [Codex blocker on unreachable exception path](feedback_codex_blocker_on_unreachable_exception_path.md) — check reachability yourself
- [Codex demands hardening beyond minimal-port contract](feedback_codex_hardening_beyond_minimal_port_contract.md) — execution-test it
- [Codex Step 0.6 literal vs purpose](feedback_codex_step_06_literal_vs_purpose.md) — PASS when a real gate runs the changed code (#551)
- [Codex conflates marker format/existence with code](feedback_codex_conflates_marker_format_with_code.md)
- [Codex misreads git/worktree state](feedback_codex_litigates_pre_existing_in_round_n.md) (#922); [gitignored artifacts as production state](feedback_codex_gitignored_artifacts_as_production_state.md) (#543, #570); [raw-branch-diff misses 10d merge](feedback_codex_raw_branch_diff_misses_surgical_merge.md) (#511 r1)
- [Codex over-reads plan prose](feedback_codex_overreads_plan_prose.md) — synthesized quotes, inflated grids
- [Descriptive-read literal as registered-threshold drift](feedback_codex_descriptive_read_literal_as_threshold_drift.md)
- [Codex under-scopes evidence base on doc/rule diffs](feedback_codex_evidence_base_underscoping_doc_rules.md)
- [Codex methodology-choice as code bug](feedback_codex_methodology_choice_as_bug.md) — plan-listed option (b) (#480, #543)
- [Registered noise-floor statistic flagged as unit bug](feedback_codex_flags_registered_noise_floor_statistic.md) (#661)
- [Codex fabricates a code citation](feedback_codex_fabricated_code_citation_silent_shrink.md) — grep 0 hits (#667)
- [Codex fail-loud diagnostic blocker](feedback_codex_fail_loud_diagnostic_blocker.md) — side-channel LOUD + headline preserved
- [Codex meta-test blocker on verified fix](feedback_codex_meta_test_blocker_on_verified_fix.md)
- [Blocker on sandbox-unreadable pod artifact](feedback_codex_blocker_on_sandbox_unreadable_pod_artifact.md) — exit-assert + pin → demote
- [Codex env-var orphan unreachable](feedback_codex_env_var_orphan_unreachable.md) — zero consumers = dead orphan (#488 r2)
- [FAILs round-N on absent OTHER-plan-section wiring](feedback_codex_plan_section_in_scope.md) — PASS+CONCERNS (#505 r6)
- [Chat-template blocker without rig cross-check](feedback_codex_chat_template_blocker_without_measurement_xcheck.md)
- [Codex TOCTOU blocker on plan-registered re-stat → demote when reversible](feedback_claude_credits_plan_literal_misses_fs_primitive_semantics.md) — sibling Critical was REAL (#2377 r1)
- [Ignored bool return vs plan-defined confirmation](feedback_codex_ignored_bool_return_vs_plan_defined_confirmation.md)
- [Plan-invariant prose vs the plan's own leg specs](feedback_codex_plan_invariant_prose_vs_leg_spec_definitions.md)
- [Env-override poisoning chain: trace every leg](feedback_codex_env_override_poisoning_chain_untraced_leg.md)
- [Step 3.75 marker-absent vs clean substance](feedback_step375_marker_absent_vs_verified_clean_substance.md) — marker-amendment, not FAIL
- [Fenced-content blindness on plan-verbatim validator](feedback_codex_fenced_content_blindness_on_plan_verbatim_validator.md)
- [Residual gap inherits parent severity bar](feedback_residual_gap_inherits_parent_severity_bar.md)
- [Conjunction resume guards + argument-constant "independence"](feedback_codex_conjunction_guard_and_argument_constant_independence.md)
- [Doc-only wording nit as Major](feedback_codex_doc_only_wording_nit_as_major.md) — error-direction safety + dead-code context
- [Hollow-test claim vs piecewise composition](feedback_codex_hollow_test_claim_vs_piecewise_composition.md) — compose the WHOLE test file
- [Test covers easy case, not realistic failure class](feedback_codex_test_covers_easy_case_not_realistic_failure_class.md) — CONCERN when defense-in-depth

## Clean-result-critic + interp-critic calibration

- [Superlative rank claims + closure location requirements](feedback_superlative_rank_claims_and_closure_location_requirements.md) — compute both headroom definitions; uphold disposition-claims-not-landed, reject invented location demands (#2564 r2)
- [Claude interp under-samples NEW sample blocks](feedback_claude_interp_undersamples_new_sample_blocks.md) — sweep ALL rows vs BOTH candidates
- [Claude interp skips derived-quantity inversion](feedback_claude_interp_skips_derived_quantity_inversion.md) — invert pool=1/chance clauses
- [Claude clean-result-critic under-applies spec text](feedback_claude_clean_result_critic_underapplies_spec_text.md) — pre-pass PASS ≠ spec-text PASS
- [Bare-#N refs + missing reuse path (Lens 2/5)](feedback_clean_result_bare_issue_refs_and_reuse_path.md)
- [Claude misses Lens 7 statistical framing](feedback_claude_misses_lens7_statistical_framing.md) — re-scan for named tests, ± (#378)
- [Cross-loop CI conflict](feedback_cross_loop_ci_conflict.md) — interp-required CIs survive Codex Lens-7 FAIL (#478, #509)
- [Codex misses Lens 9 raw+processed exception](feedback_codex_misses_lens9_raw_processed_exception.md) — adjacent pair = ONE figure (#480 r1)
- [Codex re-litigates adjudicated content](feedback_codex_relitigates_grandfathered_regate_prose.md) — diff the fold's set-body commit (#464); [replacement register after keyword ban](feedback_codex_relitigates_replacement_register_after_keyword_ban.md) — ban lists are ENUMERATED keywords (#715)
- [Claude misses silent plan deviations](feedback_claude_misses_silent_plan_deviations.md) — diff plan vs body row-by-row (#520 r1)
- [Claude interp misframes body factual errors as confidence risk](feedback_claude_interp_misframes_as_confidence_risk.md)
- [Interp completeness vs honesty-only bar](feedback_interp_completeness_vs_honesty_bar.md) — headline-bearing misses → REVISE (#549)
- [Codex passes when sandbox blocks data](feedback_codex_passes_when_sandbox_blocks_data.md) — "could not fetch" zero-weights the PASS
- [Numeric claims: re-derive yourself](feedback_codex_recount_with_silent_normalization.md) — PLAINEST matcher (#833 r2); [fix WORDING ≠ artifact TRUTH](feedback_codex_verifies_fix_wording_not_artifact_truth.md) (#778); [wrong statistic / JSON-only search](feedback_codex_fails_correct_numeric_claim_wrong_statistic_or_artifact.md) (#920)
