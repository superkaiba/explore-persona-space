---
name: Codex demands hardening beyond the plan's minimal-port contract
description: Codex FAILs ported pipelines for missing defensive asserts the plan's "pull, minimally parameterize (no rewrite)" contract never registered, on failure states unreachable via the canonical flow
type: feedback
---

When a plan ports a parent pipeline under an explicit byte-identity / "minimally
parameterize, no rewrite" contract (#556 plan §4.2), Codex FAILs round-1 demanding
defensive hardening (coverage asserts, idempotency-skip validation) that (a) exceeds
the plan's registered change set, and (b) guards states unreachable through the
canonical flow.

**Why:** #556 r1 — Codex Majors: analyze:469 `continue`-drop (but production
`--h2-min-seeds-neg 8` is an ABSOLUTE count, upstream `set -e` + 100-file assert +
judge coverage-caveat routing bound the dropout to a tail path, and `analysis.json`
exposes realized vs registered seeds post-hoc); merge:143 idempotency skip (only
base-row writer is the merge script itself — validates 400 rows + q-sha pins before
one `write_text`; interrupted write = invalid JSON = fail-loud crash, so the 1-399
partial state cannot arise). Codex also misattributed plan §6 ANALYZER-contract
reporting reads (t-interval, realized sd, parent magnitude — all computable from
committed `per_seed_mean`) as missing script emit-fields. Sibling of "Codex
litigates pre-existing in round N" — here the code is new-on-main but plan-pinned
to the parent's shape, so the same downgrade logic applies.

**How to apply:** (1) Check whether the plan registered a minimal-port contract and
whether the demanded fix is inside the registered change set. (2) Trace WRITER
reachability of the alleged partial/stale state — single-shot validated writes that
crash on corruption make it unreachable. (3) Check whether the gate's count threshold
is absolute (bounds the damage) and whether the artifact exposes the dropout post-hoc.
(4) Persist the legit residue as CONCERNs (e.g. analyzer must assert N realized
seeds) so gates read the ledger, not verdict prose; phase-gated prerequisites (VM
qbank produced DURING the pod run) stay CONCERN — a dispatch-gating BLOCKER would
deadlock. Origin: task #556 round-1.

**Pattern REPEATS in round 2 of the same task:** after all r1 concerns were fixed,
Codex FAILed r2 demanding a PASS/KILL/INTERMEDIATE verdict enum + failed-clause
list + t-interval/sd/magnitude emit-fields from analyze (re-litigating the r1
DISCARDED Minor) plus the plot's `"PASS" if h2_passed else "FAIL"` title. Impact
claim ("silently collapses the intermediate zone") was wrong on information loss:
`h2_summary` commits d_mean/ci/n_neg/per_seed_mean, so the three-way verdict is
fully post-hoc computable; plot title is a VM-phase convenience artifact, one-line
fix at zero GPU. PASS both rounds; persist the analyzer-contract obligation +
plot-relabel as ONE CONCERN. Origin: #556 round-2 reconcile.

**Named-referent-is-warn-only variant (#559 r1):** Codex Major'd a warn-only
`nvidia_smi_assert_clean` against plan text "clean assert, the #478 pattern" —
but `git show issue-478:scripts/issue478_run_cell.py` proved the NAMED referent
is itself warn-only ("Soft-fail" docstring); matching the explicit referent
resolves the plan's internal tension in the implementer's favor, and downstream
fails loud anyway (HF 7B load OOMs on a held GPU; S0 gate before any new
measurement). Same round: Codex demanded an analysis-side question-identity
assert the plan registered POD-side only (preflight gate covers the production
path; threat needs a --limit-questions artifact at the production out-dir), and
read §13.3 "excluding truncated rows" as both-side re-aggregation when the
trained-side parquet rows are not truncated rows and the stricter variant is
free post-hoc from persisted `finish_reason_per_q`. PASS; all three persisted
as CONCERNs.

**Helper-name-literalism variant (#558 r1):** Codex Critical
`raw-completions-upload-missing` because the dispatcher used
`upload_dataset_directory(out_dir, dest, pattern="completions_*.json")` instead
of the upload-policy table's named `upload_raw_completions_to_data_repo()`. But
the plan said "raw-completion upload ... verbatim from the parent" and registered
`hub.py::upload_dataset_directory` as the dependency; the PINNED parent
(93c410ddc:eval_issue543.py:547-552) uses that exact call, and it is fail-loud
(`fail_soft=False` raises on zero-files-listed) to the data repo before sentinel/
done. Policy PURPOSE satisfied → discard. Companion BLOCKER demanded code-side
"gate impossible to skip" when the plan's design prose explicitly scoped the
anchor hard-FAIL to a launcher-passed `--anchor-gate` flag (critic-r1 decision in
the plan); plan §10's command omitting the flag is a PLAN-text inconsistency,
fixed operationally via the already-persisted CONCERN — not a code defect. PASS.

**Sanctioned-option re-litigation on a zero-consumer field (#570 r3):** round-2
reconcile classified a relative Step-7 `worktree_path` Real-nonblocking and
sanctioned two fixes ("`str(Path(...).resolve())` OR require absolute"); the
implementer shipped the first verbatim; Codex r3 FAILed the residual (resolve()
uses ambient CWD → pod-side default emits an absolute-but-wrong path). Finding
factually REAL (Claude's smoke checked only absoluteness, and production
generation is pod-side via `sentinel_dir()` → `/workspace/logs`), but
non-blocking on four checks worth repeating: (1) grep the contract key across
the whole workflow surface — `worktree_path` had ONE mention (the schema list);
zero consumers. (2) Codex's proposed fix (PROJECT_ROOT / cwd base) emits an
equally wrong pod-side value — when NO cwd-based default can satisfy a
"local VM path" contract from the pod, the real fix is a LAUNCH-command
obligation, persisted as a CONCERN so the dispatch gate reads it. (3) Prior
reconcile sanctioned the implemented option → re-litigation, severity cannot
rise when the new state is strictly more informative than the accepted prior
state. (4) Sibling scripts (#504 `Path.cwd()`, #540 `PROJECT_ROOT`) share the
gap — structural to the contract field, not introduced by the round. PASS +
CONCERN `worktree-path-cwd-dependent`.

**Writer-unreachable malformed-input variant (#570 r6):** Codex Major'd a
discovery function's set-based cardinality check (`actual = {(arm, seed)}` ==
expected) for collapsing same-variant seed ALIASES (`seed42` + `seed042` both
parse to `(arm, 42)`; no `len(found)==len(expected)`), plus empty-suffix
`org_benign_` parsing as variant `""`. Both mechanically REAL — verify before
classifying — but writer-unreachable: the only producer composes `f"seed{seed}"`
from int-literal GRID_SEEDS (int f-strings never emit leading zeros) and
`validate_variant` (`[a-z0-9_]+` fullmatch) rejects empty variants; the realized
tree was already pod-verified and discovery runs once. Fast checks: (a) does ANY
codebase writer produce the alleged input shape? (b) every REACHABLE malformed
tree (wrong counts, extra seeds, mixed variants) routed fail-loud. Also: Claude's
"Unaddressed Cases — None found" enumerations can be overstated — it covered
cross-variant duplicates but missed the same-variant alias; re-derive coverage
from the code, don't trust the checklist. PASS + deferred CONCERN
(`defer-concern --by reconciler` records the BLOCKER→CONCERN downgrade per
workflow.yaml reconciler_special_case; raise at the DOWNGRADED severity first —
BLOCKERs cannot be deferred).

**Producer-consumer-in-same-diff variant (#594 r1):** Codex Critical'd
`cached-artifact-coverage-unverified` (analyzer indexes manifest by runtime ids,
no precheck) when the producer and consumer ship in the SAME diff: extraction
writes per-probe file + manifest row in one per-instance checkpoint, and the
assemble phase loads every per-probe file before writing the mean blob — so
consumer keys ⊆ producer keys by construction and any gap fails loud pre-stats
in a free off-pod phase. Codex's "cannot verify the production HF artifact from
this sandbox" is the #534 sandbox-limitation tell, and Claude had re-run the
analysis end-to-end over all 50 ids. Downgrade BLOCKER→NIT. Same round:
`weights_only=False` on a self-produced mean blob (own repo, own writer; sibling
loads use `weights_only=True`) = hardening polish; missing per-probe remote-count
assert kept as a binding pre-production CONCERN because pod terminate destroys
the only other copy of kill-criterion inputs and a partial state needs
upload_folder to break single-commit atomicity.

**Crash-fix-round variant — registered contract is the epm:failure FIX clause
(#600 r4):** for a crash-fix round, the registered change set is the failure
marker's FIX sentence, not the plan. #600's FIX clause placed sha256 pins "at
prefetch" + a bank-coverage assert "on R_train load"; both implemented exactly
(pin loop unconditional over pre-existing files; coverage assert on BOTH main
and per-cell subprocess paths). Codex FAILed demanding pins ALSO at (a) the
standalone `i600_run_cell.py` manual-retry entrypoint and (b) design-time
`select_panels.py` — (b) was byte-unchanged pre-existing code (only an
error-string changed in the diff) off the critical path (selection already ran;
manifest committed + fact-checker-verified). Watch for Codex paraphrasing the
contract STRONGER than written ("with no bypass" vs the actual "at prefetch").
Contrast with this same task's r1/r2 FAILs: those dropped REGISTERED plan rules;
here no rule registers pins at the extra entry points. PASS + raise both as
CONCERN + `defer-concern --by reconciler` (severity-downgrade carve-out).

**Round-2 same-disease-class extension after the r1 fix landed (#604 r2):** after
the r1 BLOCKER `phase-b-context-cache-stale` was verified-fixed by BOTH reviewers,
Codex FAILed r2 with two NEW stale-cache findings in sibling locations (Phase C
bundle gate anchors on `attn`/mod-meta only; Phase A resume skips by path
existence). Both code citations accurate; both defeated by writer-reachability:
(1) the bundle writer rewrites ALL files unconditionally per completed run from
one in-memory state, the gate's meta-anchor file is written LAST of the centroid
files (write ORDER excludes the fresh-gate+stale-data direction), and the
missing-file path re-downloads the complete atomically-Hub-verified bundle —
check the WRITE ORDER of multi-file bundles relative to which file the gate
reads; (2) the resume gap's only concrete instance was already remediated — the
implementer marker's smoke command literally opens `rm <stale artifacts> &&`,
and the recomputed artifact's OWN payload meta (saved subfolder + timestamp)
proves flat provenance; the r2 commit didn't touch the compute path, so r1-era
artifacts of unaffected lines = fresh compute. Check the artifact's embedded
meta + the marker's remediation command, not just the manifest (manifest rows
rebuild from current inventory — Codex's mislabel point is right about the
manifest, wrong about the payload). PASS; BLOCKER downgraded via
raise-at-CONCERN + `defer-concern --by reconciler`; Phase A item left open
CONCERN. Origin: #604 round-2 reconcile.

**Round-3 shape-corrupt-valid-JSON variant (#604 r3):** after the r2 Phase-A
concern was verified-fixed (4-field meta guard + invalid-JSON→recompute, both
reviewers' probes agree), Codex Critical'd the residual: syntactically VALID
JSON with non-dict root (`null`/`[]`) or truthy non-dict `meta` raises
AttributeError instead of recomputing. Mechanically real (probe-reproduce all
3 shapes before classifying — note falsy non-dict `meta` is absorbed by the
`or {}` and DOES recompute), but writer-unreachable: the only writer is
`json.dumps(dict-literal)`, and truncation/partial write yields INVALID JSON
which the guard handles. Failure is a loud crash, never silent stale reuse —
the opposite disease from the original concern. Proportionality matters at
the cap round: a binding FAIL would route a 2-line isinstance nit into Step
5d cap-3 pivot machinery. PASS; concern deferred via `defer-concern --by
reconciler`. Origin: #604 round-3 reconcile.

**Fabricated-plan-quote + unregistered-entrypoint variant (#601 r7):** Codex
Critical'd the launch supervisor's pid-file relaunch guard as violating "Plan v3
requirement says double-run must be impossible under every interleaving" — that
sentence appears NOWHERE in the plan (grep impossible/interleav: zero hits); the
plan registers exactly "if the pid file exists and that pid is alive, refuse ...
exit 3", which is what shipped (plus a supervised-side second check). Companion
Major demanded a `--regate-from-endpoint` committed-JSON no-GPU entrypoint when
plan §D item 4 registers resume as "p1 re-runs i601_phase0_reads.py, which
already skips provenance-valid outputs — minutes, no GPU recompute". Fast check:
grep the plan for Codex's quoted requirement VERBATIM before crediting a
plan-violation framing — Codex paraphrases contracts stronger than written
(same tell as #600 r4's "with no bypass"). PASS.

**Helper-name-literalism RECURS verbatim (#563 r1):** same Codex Critical
(`raw-completions-upload-missing`) against the same `upload_dataset_directory(...,
pattern="completions_*.json")` call. New sharp detail: Codex's demanded fix was
itself broken — `upload_raw_completions_to_data_repo` scans recursively for files
literally NAMED `raw_completions.json`, so it would have matched ZERO of the rig's
`completions_*.json` files and silently uploaded nothing. The plan's §-assumptions
table (item 15) explicitly registered `upload_dataset_directory()` as the upload
surface. Fast check: open BOTH helpers in hub.py and compare each one's file-match
predicate against the dispatcher's actual filenames before crediting the Critical.
PASS; truncation-kill loud-warn facet + stale dry-run digest persisted as CONCERNs.

#601 follow-up r1 (2026-06-12): all 3 Codex blockers discarded on one diff. (1) "adapter
upload not fail-loud" cited sft.py:1245 — not in the 4-file diff (pre-existing parent-parity;
run_cell's upload-before-delete invariant + Step 8 upload-verifier are the enforcement);
the launch.sh:181 citation was actually the seed loop's fail-loud `test -f $TRAJ || exit 1`.
(2) "sentinel does not expose required field names" — contradicted by reading launch.sh:258-285:
card carries eval_paths/adapter_paths/raw_completions_hf_paths/wandb_run_names, a SUPERSET of
the parent epm:results cards. (3) "test doesn't assert contract values literally" — test:34/39
are parity asserts and parity IS the plan contract (D2/matched-arm); driver p0 echoes literals.
Fast check: open the cited lines verbatim before crediting any of the three framings.

**Policy-table-prefix vs plan-registered layout (#614 r1):** third recurrence of
`raw-completions-upload-missing`, new shape — Codex FAILed the per-cell tree upload
to `issue614_.../eval_results/cells/<arm>/<source>/seed_<S>/raw_completions/` for not
matching the CLAUDE.md upload-policy table's `issueN_<slug>/raw_completions/{condition}_seed{S}.json`
prefix. Discarded on four checks: (1) plan line 128 REGISTERED "cell-tree uploads →
`HF_DATA_PREFIX_614/eval_results/cells/...`" and §10 pinned raw completions only to
the issue prefix with `...`; (2) the contested function is the parent #612 rig ported
verbatim (diff vs parent commit shows only the registered prefix parameterization) —
Codex's fix would BREAK the verbatim-port/instrument-identity contract; (3) durability
proven in-code (rglob presence assert + fail-loud `_upload_or_raise` before
`[phase=done]`) — Codex's "land, if at all" contradicted by the code; (4) discovery:
the parent's PROMOTED clean-result cites raw completions at the same layout (at the very
revision #614 pins), #612 passed upload verification twice on it, and verify_uploads.py
has zero raw_completions-prefix dependencies. When the parent line already deviates
from the policy-table prefix with a promoted + verifier-PASSed precedent, the table is
convention, not contract. PASS + standing rec: clean-result Reproducibility must cite
the actual cells/.../raw_completions paths.

#610 r1 (2026-06-12): Codex Major "use `list_repo_files_complete` instead of raw
`huggingface_hub.list_repo_files` (truncation risk on large model repo)" — disproven by
inspecting the PINNED hf_hub 0.36.2: `HfApi.list_repo_files` is itself implemented via
`list_repo_tree(recursive=True)`, the same paginated complete walker the project helper
wraps (truncation lives in the `repo_info().siblings` path the helper's docstring names);
repo also held 2,672 files vs the ~7,901 cap. Fast check before crediting a
raw-API-vs-project-wrapper blocker: `inspect.getsource` the pinned implementation +
count the actual repo files. Downgrade to a consistency standing-rec.

**Shell-semantics conflation variant (#598 r1, plan/statistics lens):** Codex
Must-Fix'd a render test for not pinning heredoc-delimiter quoting ("quoted EOF →
sentinel written under a literal `${SLURM_JOB_ID}` path → every clean run FAILs") —
but heredoc quoting affects only the heredoc BODY (the JSON `attempt_id` field,
which `_check_sentinel` validates phase+issue only and never reads), NOT the
`SENTINEL_PATH="$SCRATCH.../slurm-${SLURM_JOB_ID}/..."` assignment line, whose
verbatim double-quoted form the planned test asserts as a substring (bash always
expands `${...}` in double quotes; any non-expanding variant fails the assertion).
Fast check: separate WHICH shell line each quoting rule governs, then check whether
the affected field is read by the verifier. Companion MF (rendered heredoc JSON
bytes never json.loads'd through the verifier in any test) was REAL but
non-blocking: fixed-literal body, loud FAIL ("not valid JSON") blocking teardown on
first live run, and the missing test is implementer-latitude — standing rec, not
REVISE. APPROVE.

**False-justification carve-out variant (#576 r1):** the plan exempted the third
rule-named storage-contract surface (#472 `eval_trajectory.py`) as "post-softmax-only
record structurally impossible by construction" — Codex disproved the JUSTIFICATION
(Phase-A `.partial.json` persists logp-only leaves; `compute_kl=False` writes a final
artifact without logit fields), and Claude's endorsement had only checked the Phase-B
dict literal, not the sinks. Still PASS, because the carve-out survives on CORRECTED
grounds: the rule blesses the surface as "(Phase B)" specifically; Phase-A partials
are vLLM-phase crash-recovery artifacts that CANNOT carry raw logits (validating them
is unsatisfiable); `--no-kl` is a documented smoke flag with every production call
site at `compute_kl=True`; and the closure is a frozen-surface design decision outside
the registered change set. Two duties on this shape: record-correct the false sentence
in the verdict (don't let "structurally impossible" propagate) + raise the residue as
a CONCERN. Second nugget: EXECUTION-TEST the reviewer's proposed fix against its own
threat class before crediting severity — Codex's strict `logp > 0.0` demand does not
close the consistent-fabrication class it cites (the negative twin `{logp: -0.0009,
z_marker: -0.0009, logZ: 0}` passes strict too), while the realistic #530 accidents
(missing fields; probability stuffed with genuine logits) are already caught; strict-0
adds false-kill risk on legitimate saturated records (logp ≈ 0⁻ fp noise) in a
fail-loud validator inside live training callbacks. A fix that fails its own threat
class while adding false-reject risk marks the finding Real-but-non-blocking.
