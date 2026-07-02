# Spawn-baseline probe — task #838 (plan §4.1), PRE-TRIM batch

**Method.** 7 foreground spawns from the orchestrator session (2026-07-02, ~09:3x UTC), fixed prompt
`Reply with the single word OK. Do not use any tools.` — every spawn returned `OK` with `tool_uses: 0`
(no probe inflated its total by tool use). The Agent-tool `<usage>` block exposes `total_tokens` only
(no input/output split) → per plan §4.1.3 the deltas below use TOTALS, declared here. Output is ~1
token, so totals ≈ input + per-role thinking; the `effort: xhigh` covariate (planner/critic/analyzer)
is therefore folded into their deltas — an upper-bound reading for the byte-attributable term.

**Spec tree loaded (session/schema fingerprint).** The probe session's cwd is the issue-837 worktree;
specs + agent-memory resolve from THAT tree (A12 confirmed in the cwd direction). Its planner.md /
critic.md are the PRE-#835 copies — i.e. exactly the incident-era state (#833/#834), which is the
right pre-trim baseline. Session MCP inventory (heavy): happy, todoist (~100 tools), huggingface-skills,
arxiv, arxiv-latex, google-workspace (~90 tools), runpod, ssh, context7, playwright + core tools.
Fingerprint: session cmr37a0odkrpvwc0u0awddcdy; all 7 rows share it (same-session deltas BINDING).

## Probe table

| agent type | spec B | MEMORY.md B | mem-dir total B | effort | total tokens | Δ vs gp mean | implied B/tok (spec+MEMORY.md) | implied B/tok (spec+full-dir) | instrument validity |
|---|---|---|---|---|---|---|---|---|---|
| general-purpose (row 1) | 0 | 0 | 0 | default | 125,008 | — | — | — | reference |
| general-purpose (row 2) | 0 | 0 | 0 | default | 125,008 | 0 (pair noise = 0) | — | — | reference |
| implementer | 16,221 | 9,480 | 95,208 | default | 160,703 | +35,695 | 0.72 ✗ | 3.12 ✓ | INVALID under MEMORY.md-only accounting |
| experimenter | 62,672 | 18,703 | 157,560 | default | 169,560 | +44,552 | 1.83 ✗ | 4.94 ✓ | INVALID under MEMORY.md-only accounting |
| planner | 63,888 | 828 | 11,277 | xhigh | 152,480 | +27,472 | 2.36 ✓ | 2.74 ✓ | VALID |
| critic | 65,492 | 20,691 | 221,000 | xhigh | 160,945 | +35,937 | 2.40 ✓ | 7.97 ✓ (band edge) | VALID |
| analyzer | 112,229 | 9,232 | 105,023 | xhigh | 184,036 | +59,028 | 2.06 ✓ | 3.68 ✓ | VALID |

Validity band [2, 8] B/token per plan §4.1.3. gp pair noise = 0 tokens (byte-identical totals) — the
acceptance-criterion-8 noise estimate.

## Findings

1. **The shared-pile floor dominates: B_gp = 125,008 tokens** in this MCP-heavy session — ~62% of the
   ~200K window before ANY spec loads. general-purpose itself has only ~75K of working headroom here.
2. **The incident types spawn near the edge: planner 152.5K, critic 160.9K** → 39–48K tokens of
   remaining headroom. A ~450-line plan (~7K) + a ~650-line module (~7K) + rule/memory reads + tool
   results + drafting churn plausibly exceeds that several times over → autocompact engages with
   nothing evictable → thrash. Fully consistent with the #833/#834 incidents AND with the 1-of-6
   survivor (marginal-not-deterministic: baseline puts the roles near the cliff; session terms tip them).
3. **Kill-criterion 1 does NOT fire** (per-type, instrument-valid): planner Δ = 27.5K and critic
   Δ = 35.9K ≫ the 10K-token floor. The spec+memory term is 55–75% of the roles' REMAINING headroom —
   the trim is certified as MATERIAL (largest per-type lever available).
4. **BUT the trim alone is not sufficiency-certified.** Post-trim projection at the valid rows'
   measured ~2.4 B/token: planner −48K B ≈ −20K tokens → ~132K spawn; critic −52K B ≈ −21K tokens →
   ~140K. Headroom rises ~45–55% (to ~60–68K) in this session class — material, not a guarantee in
   the heaviest sessions. The dominant term is the session pile (125K), out of this task's scope.
5. **Threshold decision (§4.1.5 table, cell (i)):** H = max(80K, measured gp work usage) ≥ 80K >
   W − B_gp = 75K ⇒ B_safe < 0 < 20,000 → **ship `AGENT_SPEC_FAIL_BYTES` = 40,000 (band floor),
   `AGENT_SPEC_WARN_BYTES` = 28,000** (provisional value, in-band, < FAIL). The ≤20 KB trim target is
   NOT probe-certified as sufficient — it lands as pure weight-shedding (finding 3 certifies it as
   material). Grandfather-set membership recomputed from FAIL = 40,000.
6. **Open observation (recorded, not routed — recursion guard):** under MEMORY.md-only accounting the
   implementer/experimenter rows are instrument-INVALID (implied 0.72 / 1.83 B/token — deltas far
   exceed the declared bytes); under FULL-memory-dir accounting ALL seven rows fall in the [2,8] band
   (2.7–8.0). Parsimonious read: the whole `.claude/agent-memory/<type>/` dir (or a large part of it)
   loads at spawn, not just MEMORY.md — contradicting the LESSONS.md § Per-agent memory description.
   If true, critic's 221 KB memory dir is a co-dominant lever alongside its spec. Out of #838's
   declared scope (must-ask fence on memory files beyond #835's prune); left as a recorded observation
   for a future deliberate pass.

## Post-trim rows (deferred — A12 deviation)

The probe session resolves specs from its own cwd (issue-837 worktree), which will NOT carry the
trimmed files even post-merge without polluting another session's tree. Per plan §4.1.4 / A12, the
post-trim planner/critic/gp rows are deferred to the post-merge completion addendum, and the §5
"post-merge live validation" note (first real planner/critic spawn outcome recorded on this task's
events) is the binding observation. The criterion-8 injection guard's BINDING check is the mechanical
`paths:` frontmatter grep (plan v3), which is session-independent.
