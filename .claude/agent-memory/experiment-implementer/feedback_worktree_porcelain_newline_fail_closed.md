---
name: worktree-porcelain-newline-fail-closed
description: git worktree list --porcelain splits newline-bearing paths into orphan lines (NO C-quoting on git 2.34.1) — deletion-gate parsers must parse in RECORD form and return None on any unrecognized line; .strip() corrupts trailing-space paths
metadata:
  type: feedback
---

`git worktree list --porcelain` (git 2.34.1, verified by live repro in #2147)
emits worktree paths RAW — space, tab, backslash, double-quote all survive
unescaped on one line, and there is NO C-quoting (decoding C-quotes is dead
code on this git). But a path containing a NEWLINE necessarily SPLITS its
record: the `worktree ` line carries a TRUNCATED path and the remainder lands
as an orphan continuation line. A line-wise `startswith("worktree ")` parse
records the truncated path, so the REAL registered path is ABSENT from the
set — a positive non-registration proof built on that set is fail-OPEN and a
REGISTERED worktree can reach `shutil.rmtree` (#2147 r4, Codex R3-C1/SIB-1:
severity right, mechanism wrong — Codex attributed it to C-quoting).

**Why:** one truncated record in the listing poisons the WHOLE set; any
consumer that treats non-membership as license must refuse the entire parse.

**How to apply (final #2147 r6 architecture):** a deletion gate NEVER
answers "is THIS candidate registered?" from the porcelain listing — the
format is newline-delimited, git 2.34.1 has no `-z`, and THREE consecutive
rounds of parser hardening each left a reproducible fail-open (r4 orphan
lines; r5 `\nbare` flag-spoof — `bare`+`detached` coexist because a genuine
detached record simply lacks `bare`; r6 truncation-collision — a decoy dir
at the truncation defeats the r5 existence check). Use parse-free
per-record sources instead: (1) the candidate's OWN `.git` gitfile
(`_candidate_worktree_registration` — ours/foreign/submodule via the
resolved `/worktrees/` vs `/modules/` component, relative gitdir resolved
against the candidate, realpath comparisons only); (2) for
deleted/replaced pointers, the ADMIN-side per-record files
`<git-common-dir>/worktrees/<id>/gitdir` (content = `<wt>/.git` + exactly
ONE trailing LF — strip one, dirname ⇒ byte-exact even for embedded/
trailing newlines; `_admin_registered_worktree_paths`) — read BINARY +
explicit decode, NEVER `read_text` (#2147 r7, Codex R4-1: universal
newlines silently rewrite CR/CRLF path bytes to LF, injecting a GHOST path
into the authoritative set while the real registration goes missing; and
verify byte-exactness evidence on the SAME read mode the code uses —
`open(f,'rb')` evidence never covered a `read_text` code path). The SAME law
covers `subprocess` stdout (#2147 r8, the round-5 cap residual): a
`text=True` git pipe universal-newlines-rewrites CR/CRLF inside a
PATH-PRODUCING answer (`rev-parse --git-common-dir`/`--show-toplevel` — the
REPOSITORY'S OWN path bytes), and `.strip()` eats edge-whitespace path
bytes — use a binary-mode sibling (`_git_bytes` + `_decode_git_path`:
decode utf-8, strip exactly ONE trailing LF) for every stdout that is or
derives a path; and inside an authoritative enumeration never swallow a
scan-root `FileNotFoundError` into an empty result — missing PARENT
(common dir) ⇒ `None` (ambiguity keeps); only present-parent +
absent-`worktrees/` is a legitimate empty. Audit EVERY consumer of a
shared text-mode subprocess wrapper before ruling its normalization
harmless — r5's miss was ruling one consumer KEEP-only while a second,
authoritative consumer sat unexamined. Keep the hardened
record-form listing parse (fail closed: orphan line / duplicate slot /
existence cross-check with `prunable` exemption; never `.strip()` the
path) as DEFENCE-IN-DEPTH ONLY — membership may add KEEPs, never license.
Canonical impl + 16-test battery:
`scripts/clean_experiment_downloads.py` +
`tests/test_vm_disk_guard_slurm_src.py::test_r4_* / test_r5_* / test_r6_*`.
Open residual (reproduced, information-theoretically unclosable): a
FOREIGN-repo worktree with pointer deleted is byte-indistinguishable from
a plain staged copy — no back-pointer exists to follow. Known latent siblings
(read-only parsers, no deletion licensing, left as-is in #2147):
`scripts/verify_task_body.py::_parse_worktree_list`,
`scripts/audit_stranded_task_commits.py::list_worktrees` — harden them the
same way if they ever feed a destructive or licensing decision. Pre-fix
demonstration technique for importlib-by-path script tests: [[prefix-scratch-git-show-demo]].
