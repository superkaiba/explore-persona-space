---
name: Codex flags a vacuity-guard control / piecewise test suite as "hollow"
description: Codex FAILs a regression test as hollow because one test uses a manual reproduction + a `!=` control instead of calling the real helper with an exact sentinel; adjudicate by composing the WHOLE test file and grepping the marker's verbatim claim. #987 r1.
type: feedback
---

Codex code-review pattern (task #987 r1, PASS upheld): Codex raised a Major
"hollow test / fabricated coverage claim" against a subprocess shadow test
(`tests/test_lane_infra_main_pin.py::test_pin_beats_editable_install_end_to_end`)
on three individually-TRUE facts — (a) a fixture sentinel (`ORIGIN='worktree'`)
unused by that test, (b) a control asserting only `stdout != "main"` rather
than an exact sentinel, (c) the test body manually reproducing the path insert
instead of calling the real `_pin_main_lane_infra` helper — and a MISTAKEN
classification.

**Why the classification failed:**
1. **A `check=True` + `!=` control can be a genuine causal control, not
   vacuous.** If no ambient package exists → `CalledProcessError` → test FAILS;
   if the shadow fails → positive arm prints ambient ORIGIN ≠ expected → FAILS.
   Hollow-gate = a green-PASS-with-zero-coverage path exists (code-style.md
   § verification gates); trace whether one actually exists before crediting.
2. **Piecewise composition counts.** The real helper's insert order was pinned
   by a sibling test on the real module objects, and the shipped `__main__`
   bootstrap was executed end-to-end by a `--help` rc-0 test (pin runs before
   argparse). One test not calling the helper is not hollowness when siblings
   cover the helper + the shipped composition. The residual (no single
   committed end-to-end assert) is a MINOR, not a FAIL.
3. **Grep the implementer marker for the VERBATIM claim before crediting
   "fabricated coverage".** Codex quoted a marker phrase ("worktree-fixture
   end-to-end editable-.pth shadow test") that appeared NOWHERE in the
   `epm:results` note; the marker's actual sentence described the committed
   test exactly. A fabricated-claim premise built on a synthesized quote is
   the same family as feedback_codex_overreads_plan_prose (synthesized quotes).
4. **Live-execute the control.** Replicating the control subprocess showed it
   resolving the venv's REAL editable `.pth` — the production-faithful ambient
   candidate; Codex's PYTHONPATH-based fix sketch would have tested a LESS
   faithful mechanism.

**How to apply:** on any "hollow test / fabricated verification claim" FAIL,
(i) enumerate the failure paths of BOTH assertions (does a green-pass-zero-
coverage path exist?), (ii) read the WHOLE test file for sibling tests that
pin the real helper / shipped composition, (iii) grep the implementer marker
for the exact quoted claim, (iv) run the test + replicate the control. The
severity rule "fabricated verification claims are substantive" only fires when
the antecedents actually hold.
