---
name: vendored-depth-root-repoint-completeness
description: Byte-identical vendoring into a subdir shifts __file__-derived ROOT one level; sweep ALL module-level ROOT-derived constants vs the consumer's post-import re-point set, and classify unexercised sibling seams by fail-loud vs silent-wrong-file
metadata:
  type: feedback
---

When a commit vendors kernels byte-identical into `scripts/vendored_<X>/` (parity-with-pin is the contract, so in-file path fixes are FORBIDDEN), `ROOT = Path(__file__).resolve().parent.parent` and every derived constant shift one directory level. Review recipe (#2552 r1 g1):

1. Grep the vendored module for module-level `__file__`/ROOT-derived constants (`^PROJECT_ROOT|^ROOT|^COMMITTED|ROOT /`), then diff that set against the consumer's post-import re-point block (`T.PROJECT_ROOT = ...`). Function-scope uses of the re-pointed GLOBAL are covered; a module-level DERIVED constant missing from the re-point set is the silent-misresolve finding.
2. Sibling vendored modules that cross-load each other (`import <old_name>`, `spec_from_file_location(<old_name>, ROOT/"scripts"/...)`) break at the new depth. Classify: seam unexercised by the round's consumers + fails LOUD (FileNotFoundError/ModuleNotFoundError) = Minor + a VENDORED_FROM.txt "reference-only" note; a path that RESOLVES to the main-resident LAGGING twin = Critical (silent wrong code — the exact thing the vendoring exists to avoid).
3. Hub-vs-git twin oid mismatch on the same artifact with IDENTICAL sizes is usually LFS-pointer-blob vs git-blob representation, not a content conflict — check which path the consumers actually read before flagging.

**Why:** the vendored file must stay byte-identical (verified by blob-sha == pin == tip), so ALL correctness lives at the consumer seam; the re-point set's COMPLETENESS is the whole review.

**How to apply:** any split-review group whose commit vendors/ports files into a different directory depth while claiming byte parity; pairs with [[verbatim-port-commit-review-recipe]] (provenance probes) and [[thin-fork-commit-review-recipe]].
