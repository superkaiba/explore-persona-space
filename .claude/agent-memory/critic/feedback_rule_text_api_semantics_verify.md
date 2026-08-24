---
name: rule-text-api-semantics-verify
description: Infra plans that BLESS API call shapes in rule text — verify each blessed call's raise-vs-silent-empty semantics against the INSTALLED library source, never the task body's claim (#2442)
metadata:
  type: feedback
---

When a `kind: infra` plan writes rule text that blesses specific API calls as
absence/existence primitives, read the INSTALLED library source for each
blessed call's failure semantics — the task body's claim is a candidate, not
ground truth, even when the plan's §2 says "established by direct read"
(the read usually covered the in-repo lines, not the library).

**Why:** #2442 v1 copied the task body's "`list_repo_tree` / `get_paths_info`
both 404 on a nonexistent prefix" into an always-on CLAUDE.md clause. The
installed huggingface_hub 0.36.2 docstring says the opposite for
`get_paths_info`: missing paths are "ignored without raising an exception"
(hf_api.py:3370) — it returns a silent empty list, the exact defect class the
task existed to ban. Two adjacent semantics found the same way:
`HfApi.file_exists` returns False on a nonexistent REPO/revision too
(hf_api.py:3008, catches RepositoryNotFoundError → False), and the tree
endpoint 404s on exact FILE paths (hub.py:1058, #939) — so a raising
prefix-helper without the sibling's `file_exists` fallback raises "absent" on
a PRESENT file.

**How to apply:** for every call name the plan's rule text blesses or bans,
open the installed source (`.venv/lib/.../huggingface_hub/...`) and check:
(a) raise vs silent-empty on the missing-target path; (b) which error classes
are swallowed into the "absent" return; (c) file-vs-directory kind behavior.
5 minutes; the false blessing otherwise lands in always-on text with the
rule's own authority behind it.
