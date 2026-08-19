---
name: claude-credits-plan-literal-misses-fs-primitive-semantics
description: "#2377 r1 (guard/janitor code): Claude PASSed by matching plan literals ('mkdir 0700', 're-stat') without probing the primitive's fs semantics — exist_ok=True accepted a pre-created symlink-to-dir at a predictable /tmp path and os.rename escaped the root + overwrote the dest (live-probe reproduced); also credited 'sidecar every row' at the CALL site while the CALLEE gates the effect on apply. Codex's companion TOCTOU blocker on the plan-registered re-stat mechanism was demoted to CONCERN."
metadata:
  type: feedback
---

Two Claude code-review miss facets from one incident (#2377 round 1, uv
project-file poison arm for the /tmp sweep — Claude PASS vs Codex FAIL;
reconcile: FAIL, Codex's Critical upheld):

1. **Plan-literal match ≠ primitive semantics.** Claude's plan-adherence
   table checked "`os.rename` into `/tmp/eps-quarantine-uvproj-<ts>/`
   (mkdir 0700) ✓" — the literals matched, but `Path.mkdir(mode=0o700,
   exist_ok=True)` on a PREDICTABLE second-granularity /tmp name (i) accepts
   a pre-created symlink-to-directory (pathlib's exist_ok check follows
   symlinks), (ii) applies `mode` only at creation, and (iii) the following
   `os.rename` resolves the symlinked parent AND replaces an existing dest
   file. A 3-minute sandboxed live probe reproduced the full escape +
   overwrite, and `df -P` showed /tmp + /home on one filesystem (no EXDEV
   backstop). The plan said "fresh ... dir (created 0700)" — `exist_ok=True`
   falsifies "fresh" outright.

2. **Call-site verification ≠ effect verification.** Claude verified the
   plan's "every row appends a sidecar event" by noting `_uvproj_finish`
   calls `append_disk_guard_event(event, apply=apply)` UNCONDITIONALLY —
   but the CALLEE returns without appending on `apply=False` (lines
   525-527), and Claude's own Minor 2 quoted the `[report-only] would
   append` demotion line without connecting it to the plan requirement.
   Trace the callee's gating, not the call's unconditionality.

**Why:** guard/janitor/quarantine code is exactly where filesystem-primitive
semantics (exist_ok, symlink-following, rename-overwrite, sticky-bit /tmp
creation rights) ARE the review substance; a plan-literal checklist read
passes the letter and ships a data-corruption path into a fleet cron.

**How to apply:** on any diff that mkdir/renames/unlinks at a predictable
shared-/tmp path, EXECUTE the adversarial shape in a sandbox (pre-created
symlink at the predicted name; pre-existing dest file) before crediting a
"fresh/private dir" plan claim — a Write-tool/sandbox probe costs minutes
and is decisive. And for every "X happens unconditionally / for every row"
claim, read the callee's own gate (flags like `apply=`), not just the call.

**Codex-side companion (demotion):** Codex's second BLOCKER — the
evidence-to-rename TOCTOU (size+mtime re-stat doesn't bind pathname to the
hashed inode) — was REAL but overreached: the plan's gate 7 REGISTERED
exactly that re-stat idiom (the `_reap_scratch_tree` fresh-recheck), and
with the destination fixed the residual worst case is a REVERSIBLE move of
adversarially-swapped bytes into a private 0700 dir + a stale evidence
string — not deletion/escape, so the plan's kill criterion ("proves
unsafe") did not fire. Demoted BLOCKER→CONCERN with the cheap
fstat(fd)-vs-lstat dev/ino bind named. Same family as
[[codex-methodology-choice-as-bug]] (Codex flags the plan's own registered
rule) — but do NOT discard such findings wholesale: here the sibling
Critical was the real thing.
---
name: claude-credits-plan-literal-misses-fs-primitive-semantics
description: "#2377 r1 (guard/janitor code): Claude PASSed by matching plan literals ('mkdir 0700', 're-stat') without probing the primitive's fs semantics — exist_ok=True accepted a pre-created symlink-to-dir at a predictable /tmp path and os.rename escaped the root + overwrote the dest (live-probe reproduced); also credited 'sidecar every row' at the CALL site while the CALLEE gates the effect on apply. Codex's companion TOCTOU blocker on the plan-registered re-stat mechanism was demoted to CONCERN. r2 addendum: reachability discriminator — pre-plantable trap at a PREDICTABLE name = blocking; active same-UID racer on an UNPREDICTABLE mkdtemp name = accepted residual (Codex re-escalated the accepted window family on the sibling surface; PASS + raise+defer)."
metadata:
  type: feedback
---

Two Claude code-review miss facets from one incident (#2377 round 1, uv
project-file poison arm for the /tmp sweep — Claude PASS vs Codex FAIL;
reconcile: FAIL, Codex's Critical upheld):

1. **Plan-literal match ≠ primitive semantics.** Claude's plan-adherence
   table checked "`os.rename` into `/tmp/eps-quarantine-uvproj-<ts>/`
   (mkdir 0700) ✓" — the literals matched, but `Path.mkdir(mode=0o700,
   exist_ok=True)` on a PREDICTABLE second-granularity /tmp name (i) accepts
   a pre-created symlink-to-directory (pathlib's exist_ok check follows
   symlinks), (ii) applies `mode` only at creation, and (iii) the following
   `os.rename` resolves the symlinked parent AND replaces an existing dest
   file. A 3-minute sandboxed live probe reproduced the full escape +
   overwrite, and `df -P` showed /tmp + /home on one filesystem (no EXDEV
   backstop). The plan said "fresh ... dir (created 0700)" — `exist_ok=True`
   falsifies "fresh" outright.

2. **Call-site verification ≠ effect verification.** Claude verified the
   plan's "every row appends a sidecar event" by noting `_uvproj_finish`
   calls `append_disk_guard_event(event, apply=apply)` UNCONDITIONALLY —
   but the CALLEE returns without appending on `apply=False` (lines
   525-527), and Claude's own Minor 2 quoted the `[report-only] would
   append` demotion line without connecting it to the plan requirement.
   Trace the callee's gating, not the call's unconditionality.

**Why:** guard/janitor/quarantine code is exactly where filesystem-primitive
semantics (exist_ok, symlink-following, rename-overwrite, sticky-bit /tmp
creation rights) ARE the review substance; a plan-literal checklist read
passes the letter and ships a data-corruption path into a fleet cron.

**How to apply:** on any diff that mkdir/renames/unlinks at a predictable
shared-/tmp path, EXECUTE the adversarial shape in a sandbox (pre-created
symlink at the predicted name; pre-existing dest file) before crediting a
"fresh/private dir" plan claim — a Write-tool/sandbox probe costs minutes
and is decisive. And for every "X happens unconditionally / for every row"
claim, read the callee's own gate (flags like `apply=`), not just the call.

**r2 addendum (the reachability discriminator; PASS, Codex FAIL overruled).**
After the fix round (mkdtemp + lstat S_ISDIR/uid + chmod-0700 re-verify +
lexists no-replace assert), Codex held the destination BLOCKER open on the
RESIDUAL: a same-UID peer rebinding the fresh mkdtemp dir / creating dest
between lexists and rename, demanding `renameat2(RENAME_NOREPLACE)` (not
stdlib-exposed — probe-verified; only `dst_dir_fd` is). Overruled: the r1
class was a trap PRE-PLANTABLE at any earlier time at a PREDICTABLE name
(deterministic, no race — blocking); the r2 residual needs an ACTIVE
same-UID racer OBSERVING an unpredictable urandom name and winning a
sub-ms window — unreachable by accident, and a same-UID process already
has full authority over every file involved (no confused-deputy gain).
Tell for the overreach: Codex ACCEPTED the source-side lstat→rename window
at a PREDICTABLE pathname (strictly more reachable) while escalating its
less-reachable destination twin. Disposition: PASS + raise-concern the
residual + reconciler defer-concern, cheap dir-fd hardening named
(O_DIRECTORY|O_NOFOLLOW fd + fstat dev/ino match + rename dst_dir_fd).
Live-probe anchor: os.mkdir EEXIST-refuses every dentry class (dangling
symlink / dir-symlink / file) — pre-planting against mkdtemp is
structurally dead; unpredictability kills PRE-planting, not observation,
so the residual question is always "who can observe-and-race, and are
they inside the trust boundary?"

**Codex-side companion (demotion):** Codex's second BLOCKER — the
evidence-to-rename TOCTOU (size+mtime re-stat doesn't bind pathname to the
hashed inode) — was REAL but overreached: the plan's gate 7 REGISTERED
exactly that re-stat idiom (the `_reap_scratch_tree` fresh-recheck), and
with the destination fixed the residual worst case is a REVERSIBLE move of
adversarially-swapped bytes into a private 0700 dir + a stale evidence
string — not deletion/escape, so the plan's kill criterion ("proves
unsafe") did not fire. Demoted BLOCKER→CONCERN with the cheap
fstat(fd)-vs-lstat dev/ino bind named. Same family as
[[codex-methodology-choice-as-bug]] (Codex flags the plan's own registered
rule) — but do NOT discard such findings wholesale: here the sibling
Critical was the real thing.
