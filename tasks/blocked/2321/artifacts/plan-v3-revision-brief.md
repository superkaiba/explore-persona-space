# Phase 3 union revision brief — task #2321, plan v3 → v4

ONE revision round. Every item below is folded in this single pass; there is no
second round. Sources are named per item so you can weigh them, but the Must-Fix
set is BINDING — it is the union of blockers from three lenses × two reviewers,
plus two reconciler adjudications.

Plan under revision: `tasks/planning/2321/plans/plan.md` (v3).
Verdict tally: Methodology REVISE (reconciled, F2 only) · Alternatives REVISE+REVISE
(union) · Statistics APPROVE vs REVISE (reconciler pending; both its items are
already binding from other lenses, so nothing here depends on it).

v4 must still pass `uv run python scripts/verify_plan.py --issue 2321` with
0 FAIL / 0 WARN.

---

## MUST-FIX 1 — Landed-shard CONTENT equality (closes two opposite failures)

Sources: Claude Alternatives Must-Fix 1 (overwrite direction) · Codex Alternatives
D (false-trust direction) · Codex Statistics Must-Fix 2 (post-commit presence-only)
· Claude Alternatives Concern 3.

Three reviewers found the same missing check from opposite ends. Claude found a
landed shard being **overwritten** (crash-resume re-derives a census at a new HEAD,
emits a fresh `shard00.jsonl` holding only the surviving members, and the next
unit's `CommitOperationAdd` replaces the landed shard whose sources are already
deleted — members exist nowhere at HEAD). Codex found a landed shard being
**falsely trusted** (`unit_landed` accepts path PRESENCE, so after a codec fix or a
stale prior run the same paths may hold different bytes, and the driver publishes
new indexes against old shards). One content-equality anchor closes both.

Required:

(a) **Unit composer:** assert `set(add_paths) ∩ {paths under <prefix>/packed/ at
    HEAD} == ∅` via one scoped `list_repo_tree`. A legitimately re-issued "clean"
    unit passes (its shards are absent by definition); a landed unit is skipped
    before compose; the overwrite class becomes a loud abort. This is the
    upload-side twin of the never-overwrite-differing property v1 already has and
    §2 cites approvingly.
(b) **Every landed/resume probe** (`unit_landed` / `probe_unit_state`) compares the
    remote content anchor against the unit's EXPECTED digest — non-LFS via
    `blob_id` vs local `sha1("blob <len>\0" + bytes)` (A6 semantics, already
    empirically confirmed). Presence must never imply landing.
(c) **Phase 6 postverify** exhaustively compares every landed shard's remote
    `blob_id` against the local git-blob sha1 — not merely the exact path set that
    `verify_repo_paths_uploaded` proves. Phase 6 is the LAST point this is cheap:
    the same phase then reaps local staging (I10), after which repair means
    re-downloading and rescanning every shard.
    **Extend the same anchor to `INDEX.json`, the per-group index parts, and
    `pack_manifest.json`** — the Statistics reconciler adjudicated this extension
    as WARRANTED, with a mechanism worth stating in v4 because it is NOT the
    obvious one: corrupt landed metadata can never SERVE wrong bytes (each shard
    line carries its member sha256 and `stage_packed_file` sha256-verifies after
    decode, so a wrong offset fails loud). The damage is different — with the
    originals deleted, an unparseable or truncated landed `INDEX.json` makes the
    shim "re-raise the ORIGINAL error", so every archived original under that
    prefix reads as a 404 **indistinguishable from "never existed"**: the
    false-absence class that invites wasteful regeneration. A wrong group index
    bricks the canonical read path for an entire group while phase 6 reports
    success, and the live shim check samples only 3 originals per prefix, so an
    unsampled group's corruption survives acceptance. Marginal cost is ~zero: the
    metadata files land under `<prefix>/packed/`, are non-LFS JSON, still have
    local copies on disk, and the phase-6 scoped walk already returns `blob_id`.
(d) **§12 fixtures:** resume-after-partial-commit with a re-derived census must
    ABORT, not overwrite; same-paths-with-changed-bytes (simulate a codec-version
    change) must ABORT; an injected mismatched remote `blob_id` must FAIL before
    cleanup or success reporting.

Note the detection gap this closes is otherwise near-undetectable: the
3-samples-per-prefix live shim check would hit an overwritten member with
probability ≈ 0.

---

## MUST-FIX 2 — `parent_commit` pinning becomes PRIMARY on deletion-bearing commits

Sources: Codex Methodology F2 — **reconciler-UPHELD as binding** · Codex Statistics
Must-Fix 1 · Claude Methodology Concern 1 · Claude Alternatives Concern 2. Four
independent arrivals; the reconciler settled it.

§3.4 phase 5 runs `drift_check(unit)` and then an unpinned `api.create_commit(...)`
with no `parent_commit` anywhere in the composer, and §11 records the omission as
deliberate ("considered, not adopted as the primary mechanism"). A write landing in
that window produces a commit that deletes the writer's new bytes while archiving
revision-R bytes — at HEAD the deleted bytes then exist nowhere, violating the
verbatim user constraint the plan itself quotes in §3.5.

Why this is conclusion-changing rather than a widened recovery window:

- **It is SILENT.** The recovery copy sits at an unrecorded intermediate revision no
  run artifact points to, and the shim then serves the stale archived bytes AS the
  file — breaking §3.7's own "miss LOUDLY (404), never read stale data" guarantee
  with no error anywhere.
- **I7 cannot close it** — I7 *is* the racing check. No other invariant covers the
  window.

Required:

(a) `drift_check` / `probe_unit_state` RETURN the HEAD sha they verified against.
(b) Every deletion-bearing `create_commit` passes it as `parent_commit`.
(c) A parent-conflict rejection routes re-probe → re-pin → retry under **its own
    backoff budget, distinct from the 3-attempt ambiguous-outcome budget.** This
    clause is load-bearing: without it, sustained fleet traffic converts a safety
    mechanism into spurious prefix aborts.
    **The abort criterion must be narrow, and the Codex remedy as written is too
    aggressive here.** Codex proposed treating a stale-parent rejection as "a clean
    drift abort"; the Statistics reconciler flagged that this would spuriously abort
    HEALTHY prefixes across the entire back half of the run, because unrelated
    fleet commits between drift-check and commit are the run's own expected steady
    state once it starts freeing slots. Correct behavior: on a stale-parent
    rejection, re-run `drift_check` at the NEW HEAD and re-issue with the new pin
    (bounded attempts), and abort the prefix **only when the re-check finds actual
    census drift in the unit's own source dirs.** An unrelated commit costs one
    cheap re-probe + re-pin cycle, never a data-bearing failure.
(d) **Error taxonomy:** `parent_commit` mismatch surfaces as HTTP **412**; §3.4's
    `is_ambiguous_outcome` / `is_rate_limit` branches must classify 412 as
    DEFINITIVE-retriable, never ambiguous — otherwise the fix mis-routes into the
    probe branch and re-creates the ambiguity it was added to remove.
(e) **Record the bonus the reconciler found:** pinning also hardens I11's
    clean→re-issue branch — a timed-out first attempt that landed server-side makes
    the re-issue fail on parent conflict instead of depending on the
    asserted-but-never-live-verified delete-of-absent-path rejection semantics. The
    fix strengthens an invariant every verdict treated as already closed.
(f) **Correct the §11 premise, which is backwards.** Claude Alternatives argued
    pinning is cheap *because* fleet traffic is near-zero while the repo is at cap.
    In fact the repo is at cap so all fleet pushes currently FAIL — and the moment
    the first prefix frees slots, QUEUED writers start landing, including the live
    #1739 session that owns two target prefixes, precisely during the ~150-unit
    commit phase. Exposure is highest exactly where v3 assumed it was lowest. State
    it that way.
    **And the "at cap ⇒ no concurrent writes" intuition is wrong even before any
    slots free** (Statistics reconciler): the file-count cap blocks ADDS, not
    MODIFICATIONS — a content-replacing commit leaves the file count unchanged and
    is accepted at cap. So concurrent modification of a named source is possible
    from the first unit onward, and becomes the expected steady state of a
    *succeeding* run rather than a tail event. This is the single strongest reason
    the pin cannot stay optional.
(g) **§12:** test modification-after-drift and timeout-lands-after-clean races;
    AST-check that every deletion-bearing `create_commit` supplies a non-null
    `parent_commit` derived from that attempt's own probe.

The severity asymmetry to state plainly: the pinned failure mode is
fail-loud-and-slow; the unpinned one is a silent stale archive violating an
absolute user constraint.

---

## MUST-FIX 3 — Durable per-unit journal INSIDE each atomic data commit

Source: Codex Alternatives E. Same root cause as Claude Alternatives Must-Fix 1
(local-only unit state + per-phase-only HF mirroring), different consequence.

Per-unit records are local `O_APPEND` only, while state mirrors to HF "after each
phase" (§3.4). If the pod dies mid-`commit` phase, completed units have deleted
sources and remote shards but no durable manifest, index, or unit journal (§6
explicitly contemplates the wipe: "Pod stop/resume wipes `/root` staging"). A fresh
walk at the new HEAD cannot recreate the original census, so it may omit
already-landed members and later publish an INCOMPLETE index while appearing to
have resumed successfully.

Required: persist a **self-identifying per-unit journal record inside each atomic
data commit**, so unit identity lands atomically with its shards and deletes — OR
specify and test full reconstruction of the original census, units, indexes, and
expected hashes by parsing and verifying every already-landed remote shard before
continuing. **Prefer the journal-in-commit form**: it is strictly stronger, and it
subsumes Must-Fix 1(a)'s crash-resume case rather than merely asserting against it.
Keep 1(a) as cheap belt-and-braces.

§12: kill a fixture run after a server-side unit landing but BEFORE local append or
mirroring, erase all local state, resume, and require the final index/member census
to equal the original candidate census exactly.

---

## MUST-FIX 4 — Per-unit index resolvability inside the deletion-bearing commit

Source: Codex Alternatives G; sharpens Claude Methodology Concern 5, which
mis-sized the same window as "transient and acceptable".

Data commits delete originals BEFORE the final per-prefix `INDEX.json` commit, but
§3.7 resolution STARTS at `packed/INDEX.json`. So helper-path consumers cannot
resolve already-deleted members throughout the ENTIRE commit phase — this is a
PLANNED availability gap, not a rare crash window. Sized for `issue1481`: ~52 units
⇒ ≥ ~17 minutes from pacing alone, potentially hours at the stated five-minute ramp
threshold, and indefinite if a crash or a rejected manifest commit intervenes.
"Resume completes it" is not a bounded recovery contract, and Must-Fix 3 shows the
resume itself was not durable.

Required: EITHER every deletion-bearing commit also publishes/updates the index
information needed to resolve that unit's members atomically, OR the plan
explicitly WITHDRAWS the continuous-consumer-availability guarantee, quiesces
consumers, and ships a tested recovery SLA / state machine. Choose one and say
which; do not leave it implicit. Also label a mid-abort prefix
partial/unindexed and report landed units explicitly (Codex Statistics).

§12: resolve members after every intermediate unit, not only at prefix completion.

---

## MUST-FIX 5 — Per-prefix PRE-repack consumer gate

Source: Codex Alternatives J — **orchestrator-verified against the actual code.**
This REFUTES the Claude Alternatives measured claim that "no concrete
silent-wrong-result consumer was found among the 10 prefixes' readers." Two exist,
both on in-scope tier-A prefixes.

**Verified consumer 1 — `scripts/issue1090_fu3_yield_replay.py`**, prefix
`issue1090_pvdatagen` (58,363 non-LFS, tier A). L66-72 lists the prefix
recursively; L73 derives `cells` ONLY from paths ending
`/datagen/judge_raw_pos.json`. Post-repack those originals are gone ⇒ `cells == []`
⇒ the per-cell loop never runs ⇒ `rows == []` ⇒ the `if rows:` summary at L119 is
skipped ⇒ **`return 0`**. Exit SUCCESS, no diagnostic, total data loss unreported.

**Verified consumer 2 — `scripts/issue1481_cjk_audit.py`**, prefix
`issue1481_conpos_grid` (206,559 non-LFS, tier A — the LARGEST in scope). L86-93
`_download_pools` lists `{PREFIX}/raw_completions/{panel,base_arms}` keeping only
`.json`, which post-repack yields `[]`. `scan([])` returns `{}` (L137), so
`pools == []` and the summary computes `n_pools=0, n_completions=0, n_intruded=0`
over empty sums with no exception — then **WRITES `cjk_intrusion_scan.json`
containing those zeros** (L275) and prints `[i1481-cjk] scan: 0/0 intruded over 0
pools`. A PERSISTED false analysis artifact indistinguishable to a later reader
from a real zero-intrusion result. Whether the subsequent `recount` happens to
crash is irrelevant — the artifact is already on disk.

Why the earlier lens missed them: its sample hit consumers that DO fail loud
(`issue667_alllayer_analysis.py:198`, `issue1739_natpv.py:382`,
`issue1090_fu4.py:1571`) and it generalized from sample to population. It even
identified the correct silent-tolerant SHAPE and named `issue722_fit_M.py:589`,
then discounted it as a non-repacked prefix. Sampling is the wrong instrument for
an absence claim.

Required:

(a) **Correct §3.7.** "They miss LOUDLY (404), never read stale data" is FALSE as
    written. Name the listing/glob **silent-empty** class explicitly:
    `snapshot_download(allow_patterns=...)` returns SUCCESS on zero matches, and
    `list_repo_tree` over a partially-emptied dir returns the retained subset.
(b) **Replace the completion-time follow-up with a PER-PREFIX PRE-REPACK GATE.** A
    follow-up filed at completion is too late by construction — the originals are
    already deleted. Before repacking prefix P: inventory P's listing/glob
    consumers (AST/grep for `list_repo_tree` / glob / `snapshot_download` discovery
    on P outside packed-aware helpers), and either migrate them to packed-aware
    listing/staging or add explicit nonempty / expected-census assertions. BLOCK
    repacking P while an unsafe consumer remains.
(c) #2304's completed routing is the dominating no-delete interim while that
    migration is outstanding — say so.
(d) §12: a repacked-tree fixture test per migrated consumer.

---

# CONCERNS TO FOLD (non-blocking, but all of them)

## C1 — §3.1 arithmetic corrections (orchestrator-verified against the inventory)

Recomputed from `tasks/planning/2321/artifacts/top10_prefix_inventory_2026-08-16.json`.
Tier E and the "≈6×" claim are TRIPLE-confirmed (orchestrator + Claude Statistics +
Codex Statistics). Fix these in the table AND in the renegotiation paragraph:

| Cell | v3 | Corrected |
|---|---|---|
| Tier E files | 30,300 | **10,399** |
| Tier C net (b64) | +23,500 | **+19,862** |
| C vs A+B bytes | "≈ 6×" | **4.23× (b64) / 3.63× (tar)** |
| Renegotiation slots | "+24–30k" | **+19.9k (b64) / +29.7k (tar)** |
| Renegotiation share | "≈ +4.5% of the win" | **3.7% (b64) / 5.5% (tar)** |
| Tier B net | ~53,030 | **53,002** |

- Tier E reproduces two independent ways: total LFS 116,114 − B 53,760 − C 29,793 −
  D 22,162 = 10,399; and the E row's OWN component list (296 + 9,849 + the 254
  residues) = 10,399. The row currently contradicts itself.
- Tier C root cause: the 9 MB cap was divided by the RAW 1.836 MB member size
  instead of the b64-expanded 2.448 MB, giving 5 members/shard instead of **3** —
  9,931 shards, net +19,862. The tar variant's +29,700 IS correct (large
  LFS-routed shards).
- **"minus ~2,700 added files" is CORRECT** — it is tier-A-scoped and sits inside
  the tier-A derivation; tier B's ~758 shard adds are already inside B's own net.
  Label it "tier-A adds" so it is not re-litigated. The headline reproduces
  exactly: 488,500 + 53,002 = **541,502**. It is NOT conservative by ~800 slots —
  that claim double-counted tier B.
- **§11's slots-per-GB figures mix denominators — confirmed, presentational only.**
  Reconciler-verified reproduction: A's "≈55k slots/GB-moved" reproduces ONLY on the
  download denominator (488.5k/8.8 = 55.5k; upload basis gives 53.1k), B's "≈8k"
  ONLY on upload (53.03k/6.56 = 8.1k; download basis gives 10.8k), C's "0.4-0.5k" on
  download. On ANY single consistent basis — download (55.5k / 10.8k / 0.44-0.56k),
  upload (53.1k / 8.1k / 0.33-0.56k), or down+up (27.1k / 4.6k / 0.19-0.28k) — the
  A>B>C ordering holds with ≥5× gaps at every boundary. No ranking, tier decision,
  or recommendation changes. Fix by RELABELING each figure with its basis (it feeds
  the user-facing tier-C option at the approval gate, so it should be legible
  alongside the tier-E and ≈6× corrections).

## C2 — The approval-gate option is MIS-SPECIFIED (orchestrator-derived; no lens caught it)

§3.1 asserts "reaching a literal 550k **requires tier C**." That is **false**. From
A+B = 541,502, needing +8,498:

| Route | Slots | Bytes moved (down+up) |
|---|---|---|
| A+B+**D** (b64) | **559,229** ✓ | **63.7 GB** |
| A+B+C (tar) | 571,200 ✓ | 106.9 GB |
| A+B+C (b64) | 561,362 ✓ | 124.6 GB |

Tier D (`issue1739_partial`, 22,162 LFS × 1.262 MB → **+17,729** verified,
consistent with the plan's own +17,500 row) clears the bar ALONE at roughly half
the bytes of the cheapest tier-C variant. Tier D is already a costed row; it is
simply never considered as the route to the bar. As written the gate offers the
user the dearer of two satisfying options on the stated ground that it is the only
one.

Required in v4: present BOTH as priced options, with the caveat that may still make
C the right pick — `issue1739_partial` is one of the two prefixes owned by the LIVE
#1739 session, so tier D inherits the coordination-marker + pre-commit
liveness-re-probe + SKIP-and-defer apparatus (§3.6, A9) and can legitimately end up
DEFERRED delivering +0, whereas tier C's `issue1489_ctx_aug` has no owner.
Cheaper-but-may-defer versus dearer-but-unblocked. Do NOT silently switch the
recommendation: **A+B stays the default**, on unchanged reasoning.

## C3 — Census-to-member BIJECTION

Codex Methodology PD + Codex Alternatives C, independently. The round-trip verifies
MEMBERS, not that members form an exact bijection with the candidate census. A
silently DROPPED candidate is never deleted — safe and recoverable — but it must
surface in the realized per-prefix counts rather than passing unnoticed. Require
`member src` to be a unique exact bijection with the census; assert it and report
any delta.

## C4 — Cap-probe evidentiary value (reconciler-STRUCK as blocking; keep as strong rec)

§3.6 "re-reads the live count immediately before commit A and **reports** any
drift" — reports, not aborts, so a concurrent deletion can put commits A/B below
cap and make the PASS vacuous. Struck as blocking because a false PASS carries zero
artifact risk: every real deletion rides an atomic add+delete whose delete set
derives from same-commit shards (I1), so a false cap hypothesis costs one rejected
commit and rc=21 STOP with nothing deleted — which the plan already designs as its
own fallback. Still worth fixing: abort-or-recompute on pre-A count drift rather
than reporting, and pin B on the post-A HEAD, so the probe's evidentiary value is
guaranteed rather than probabilistic.

## C5 — Test-process mutation interlock (reconciler-STRUCK as blocking; adopt as hardening)

Refuse canonical-repo `create_commit` from test processes before network access
absent an explicit apply permit (e.g. an `HF_HUB_OFFLINE=1` conftest default), with
dependency-injected fakes for behavior tests. Struck as blocking because it
presupposes a future test-authoring omission, and even then fixture-derived delete
sets name fixture paths and are rejected atomically, while the plan already carries
`test_dry_run_thread_issues_zero_mutations`. Adopt anyway: it converts the body's
"Do NOT test it by deleting real artifacts" from a per-test convention into a single
enforced boundary.

## C6 — Resolver-bucket externality

Claude Methodology Concern 2: 16 workers on ~50-100 ms small-file fetches run well
above the ~40 req/s FLEET-SHARED PRO bucket and would saturate it for 8-12 h,
starving concurrent sessions' HF staging. Adopt a proactive ~30 req/s client-side
ceiling instead of 429-driven backpressure. This taxes neighbors, not this task.

## C7 — I3 sparse-unit edge

Both Claude lenses independently: a unit composed solely of 1-member shards is
net-0 and trips `len(dels) > len(ops)`. Fail-loud so safe, but the composer should
rebalance bins so a legitimate sparse tail cannot hard-abort a prefix.

## C8 — Pack-time TOCTOU closure (free)

Claude Statistics Concern 1: the pack phase already holds and hashes each member's
bytes — ALSO assert the census anchor on the bytes actually PACKED, collapsing the
phase-2 → phase-3 local-corruption window to zero.

## C9 — Commit-count basis is ±30%

Claude Statistics Concern 2: ~150-220 data units, not ~165, because ops ≤ 4,500
binds tiny-file prefixes at one 4,000-member shard per unit. Recompute units from
the realized census at implementation and restate the §9 row; the commit row may
stretch to ~7 h.

## C10 — HF-cache double-count on the 50 GB overlay

Claude Statistics Concern 5: `stage_hub_file` with `HF_HOME` on the same overlay
may hold cache plus staged copies (~+4.9 GB worst prefix; ~30 GB absolute worst if
never reaped). Reap `hfhome` in the per-prefix phase-6 cleanup alongside
stage/pack/scratch.

## C11 — Headline framing

Claude Statistics Concern 6: report Σ per-prefix (before − after) as the realized
freed-slots headline. The end-of-run full-walk count is a live shared quantity that
other sessions legitimately move once headroom exists.

## C12 — Manifest-incomplete prefix state

Claude Methodology Concern 3: the final per-prefix manifest commit is net-positive;
if the repo re-fills to cap mid-run it is rejected → rc=21 STOP leaves shards live
and originals deleted with no `INDEX.json`. Name that state and its one-commit
completion recipe in the rc path and the final report. (Shrinks if Must-Fix 4 lands
as index-in-every-commit.)

## C13 — Collided-group resolution

Claude Alternatives Concern 5 + Codex Alternatives B: with the
`key + "-" + sha1(rel_dir)[:8]` disambiguation, deriving "group key from parent
dir" (§3.7) fails for collided groups unless the shim resolves via `INDEX.json`'s
recorded `rel_dir`. Decide collision mapping BEFORE any shard is written, and
ensure `tests/test_hub_packed_fallback.py`'s fixture set includes a collided-group
(two colliding directories) prefix shape.

## C14 — Retained set as an auditable transitive closure

Codex Alternatives F: generate the retained set as a transitive closure of
recognized manifests plus their enumerated parts, exact-set tested against known
#1739-v1 and #2119 fixtures, rather than trusting filename suffixes.

## C15 — The git-history backstop is TIME-BOUNDED

Codex Alternatives H: future storage pressure may motivate
`super_squash_history`, which eliminates the independent recovery copy. The final
report must state the two-independent-copies guarantee as time-bounded and note
that a squash destroys it. This directly qualifies the reassurance Claude
Alternatives offered ("git history is the SECOND copy, not the only one") — true
today, not guaranteed tomorrow.

## C16 — Tier-B LFS-routing detection is post-commit

Codex Methodology PF: unexpected LFS routing is detected only AFTER the commit, so
accepted tier-B originals may already be deleted. Agreed disposition: this
invalidates the quota-immunity PREMISE, not the no-unarchived-delete constraint —
the bytes remain atomically archived. State it as a tier-B scope risk, not a safety
hole.

## C17 — Commit-B rejection routing

Claude Alternatives Concern 4: a server rule "net-negative required at cap" would
reject the net-zero probe B while every real repack commit (net ≤ −1) passes. Do
NOT report that branch as "hypothesis invalidated" — try the body's own
net-negative real-unit probe first (the fallback already wired for a commit-A
rejection).

## C18 — 429 exhaustion reported distinctly

Codex Statistics: three consecutive 429s consume the shared three-attempt loop and
produce `attempts-exhausted`. Report it as a clean, probe-first-resumable
rate-limit outcome, not an atomicity anomaly.

## C19 — Verification must use the PRODUCTION decoder

Codex Alternatives A: parsing every written line and SHA-comparing every decoded
member closes the escaping class only if verification runs the PRODUCTION decoder,
not a test-local reimplementation.

## C20 — Phase 6's exact-set assert must tolerate PRE-EXISTING files under `packed/`

Statistics reconciler, new: §3.3(d) assumes `<prefix>/packed/` is exclusively v2
output, but this repo already carries #1739-v1 packs, so a target prefix may
already populate `packed/` with non-v2 files. Write the phase-6 exact-set assert on
`<prefix>/packed/` to tolerate pre-existing non-v2 entries rather than treating
them as a set mismatch — otherwise a prefix that #1739 already touched fails
postverify for a benign reason, after its originals are deleted. Cross-check
against §3.3(c)'s v1/#2119 resolution-contract protection and C14's
transitive-closure retained set, which cover the same collision surface from the
retention side.

---

# PRESERVE — do not regress these in v4

- **§11's alternative-rejections are now TWICE independently verified on mechanism**
  (Claude Methodology + Codex Methodology). Keep them, and keep the stronger reason
  Claude supplied that the plan did not state: `*.tar` is LFS-matched by default
  gitattributes even under 10 MB, so the tar rejection is firmer than argued.
- **I11's no-blind-retry disposition** — "exactly the right disposition"; a critic
  independently walked every ordering of the in-flight double-issue race and
  confirmed it resolves loudly in each.
- **The probe's zero-artifact-risk design**, and the fact that it exercises the HTTP
  `create_commit` path (the right instrument, since the recorded rejection came via
  a git-push pre-receive hook and could differ).
- **A+B as the recommended scope**, on unchanged reasoning (98.5% of the literal bar
  for 29.5 GB; every extension is a multiple of that for ≤5.5% more win).
- The byte-exact codec, shard-derived per-file deletes, the pre-delete round-trip
  gate, scoped staging, and the honest reporting of exclusions / realized counts /
  deferrals / the history-squash limitation — all affirmed by both reviewers.
