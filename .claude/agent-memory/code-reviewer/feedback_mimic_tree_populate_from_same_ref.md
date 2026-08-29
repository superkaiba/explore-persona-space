---
name: mimic-tree-populate-from-same-ref
description: When mutation-testing a branch's tests in a mimic dir, populate EVERY sibling file from the same ref as the test — repo-root copies inject unrelated failures that masquerade as the catch you were probing for
metadata:
  type: feedback
---

Building a mimic tree to run a branch's tests against a mutated input
(the `git show`-into-a-scratch-dir technique) requires populating **every
file the test scans** from the SAME ref as the test itself. A file copied
from the repo root instead is usually a DIFFERENT version, and its
unrelated failure looks exactly like the test catching your mutation.

**Why:** In #2386 r2 I probed whether a class invariant over
`scripts/cron_*.sh` was fail-open to a crafted mutant. I copied the test
from `origin/issue-2386` but the sibling wrappers from the repo root
(on `main`, lacking the round's fixes). The scan FAILED, and I first read
that as "the invariant caught the mutant". It had not: the failure named
`cron_autonomous_session_watch.sh: classified 'fatal-guard' but defines no
fatal() helper` — a `main`-vs-branch version skew in a file I was not even
testing. Re-running with every wrapper extracted from the branch showed the
invariant **PASSED** the mutant, i.e. the fail-open was real. The
contaminated run would have produced a verdict claiming a hole did not
exist.

**How to apply:**
- Extract siblings with the same ref as the test:
  `for f in $(git ls-tree --name-only <ref> scripts/ | grep '<pat>'); do
  git show "<ref>:$f" > "$M/$f"; done` — never `cp repo/scripts/*`.
- Always run an **unmutated BASELINE through the mimic first**. It must be
  fully green. A baseline that fails means the mimic is contaminated, and
  every mutant verdict after it is uninterpretable.
- Confirm the failure MESSAGE names the file you mutated. A failure naming
  some other file is a harness artifact, not a catch.
- Corroborate a static-scan verdict with a direct trace of the predicate
  (call the helper on the mutant and print what it computed). That trace is
  what settled the real answer here.
- Tests that resolve their root via `Path(__file__).resolve().parents[1]`
  retarget automatically to the mimic; drive them with
  `cd <repo> && uv run python -m pytest <mimic>/tests/...` so the venv
  resolves while the tests read the mimic tree.
