---
name: mutant-matrix-needs-negative-control
description: When an implementer reports a mutant acceptance matrix for a hardened structural pin, re-run it yourself AND add a benign-reformat negative control — positive mutants alone cannot distinguish correct tightening from over-tightening.
metadata:
  type: feedback
---

An implementer closing a "this pin is too weak" concern typically ships a matrix
of mutants that the OLD test passes and the NEW test fails. Reproducing those
confirms the pin got stricter. It cannot tell you whether it got stricter *only*
where intended.

**Rule:** rebuild the matrix in a scratch mimic tree and add at least one BENIGN
mutant — a semantics-preserving reformat the new pin must still ACCEPT. Extract
both test versions with `git show <fix-sha>:<path>` and `git show <fix-sha>~1:<path>`
into `<scratch>/tests/`, copy the scanned sources into `<scratch>/scripts/`, and
run pytest with cwd at the scratch root (a `Path(__file__).resolve().parent.parent`
repo-root helper resolves correctly there). Never mutate the worktree.

**Why:** on #2387 r2 the report's three mutants (wrong duration, second unbounded
call, deleted site) all reproduced. The finding that actually mattered came from
two probes I added: a whitespace reformat (`timeout  --kill-after=5s   "${X}s"`)
that the OLD test FAILED and the NEW test PASSED — proving the regex relaxed the
whitespace axis while tightening the semantic ones — and a backslash-continuation
split that BOTH versions failed, which git-provenance-classified a real
over-tightness as pre-existing rather than a round regression. Without the benign
control I would have had no evidence either way on "is this regex too strict?",
which was the brief's explicit question.

**How to apply:** budget ~4 extra mutants beyond the reported set — one benign
reformat (must PASS), one semantically-equivalent-but-differently-spelled form
(records a deliberate narrowing), one intervening-token probe (must FAIL), one
structural-layout probe such as a line continuation. Run every probe against BOTH
test versions: a probe that fails under the old version too is pre-existing and
belongs at Minor with the git-provenance note, not in the blocker list.

Related: [[extract_and_execute_recipe_tests]], [[feedback_prefix_demo_git_show_not_stash]].
