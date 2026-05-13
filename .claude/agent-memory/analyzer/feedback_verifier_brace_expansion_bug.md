---
name: Verifier crashes on bash-style brace-expansion paths
description: scripts/verify_clean_result.py crashes with OSError ENAMETOOLONG when Setup-details references `{a,b,c}.json` brace-expansion paths; explicit list-of-files works
type: feedback
---

When listing raw eval JSONs in the collapsed Setup-details block, do NOT
use bash-style brace expansion (`eval_results/issue_311/{analysis,null_distributions,...}.json`).
The `check_numbers_in_json` regex extracts the literal pattern and tries
`Path("eval_results/issue_311/{analysis,null_distributions,...}.json").exists()`,
which crashes with `OSError [Errno 36] File name too long` and aborts the
entire verifier run.

**Why:** the verifier checks every quoted path it can find for filesystem
existence (so it can spot-check that raw-data references are reachable).
Brace expansion is shell syntax — Python's `Path` doesn't expand it.

**How to apply:** in the collapsed Setup-details JSON-list, write the
filenames out one-by-one — even when 13+ files share a prefix — instead
of `{a,b,c,...}.json`. Acceptable:

```
- Raw eval JSONs: 13 files under `eval_results/issue_311/` — `analysis.json`,
  `null_distributions.json`, `pair_selection.json`, ..., `dep_preflight.json`.
```

Not acceptable (crashes verifier):

```
- Raw eval JSONs: `eval_results/issue_311/{analysis,null_distributions,...,dep_preflight}.json`.
```
