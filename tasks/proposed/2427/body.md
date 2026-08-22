---
title: Gates must not read private transformers attributes — per-issue version pins
  turn config._commit_hash into a false-alarm crash (#2329)
kind: infra
tags: []
created_at: '2026-08-20T17:56:34Z'
has_clean_result: false
workflow: v1
---
---
kind: infra
---

# Infra: gates must not read PRIVATE transformers attributes — a per-issue version pin turns `config._commit_hash` into a false-alarm crash

## Goal

Add a `.claude/rules/gotchas.md` entry (and evaluate a cheap lint check) for a trap that just killed a
live GPU run and will recur by construction: **verification gates that interrogate a private
third-party attribute break the moment a per-issue version pin differs from the repo pin.**

## The realized incident (#2329 `q35_ladder_decay`, 2026-08-20)

`scripts/issue2329_ladder.py:818`, in `_assert_pin_engaged`:

```
AssertionError: model pin NOT engaged: config._commit_hash=None != 'c202236235762e1c871ad0ccb60c8ee5ba337b9a'
[dispatch] bank exited rc=1
```

The assertion was WRONG and the run was RIGHT. The bank log's own HTTP trace shows `config.json`,
`merges.txt`, `tokenizer.json`, `chat_template.jinja`, `model.safetensors.index.json` and every weight
shard resolving at `.../resolve/c202236235762e1c871ad0ccb60c8ee5ba337b9a/...`, with 427 weights loaded
from that commit, and `load_model_and_tokenizer(cfg, revision=...)` threading `revision=` into
`AutoTokenizer`, `AutoConfig` and the model (`scripts/issue2329_run.py:1466/1471/1478`). The pin was
fully engaged. `config._commit_hash` is a PRIVATE transformers attribute that is simply not populated
under the version the run deliberately uses.

**Why this is structural, not a one-off.** This project runs MORE THAN ONE transformers version at once
by design: the repo `uv.lock` pins 4.57.6 (used for VM-side tokenizer ops) while a per-issue pod pin
installs a different version — here `transformers==5.15.0`, installed by the round's own `gate0b` phase
with `uv pip install` under `UV_NO_SYNC=1` (`scripts/issue2329_ladder_dispatch.sh:9,50-51,102-107,294`;
plan v8 line 351 verifies both sides deliberately). Any gate written against a private attribute is
therefore validated on one version and executed on another. The cost here: a 1× H100 pod idled while the
crash was diagnosed, after it had already paid a ~1.5 min weight fetch, a 427-shard load, and a 556.8 s
cold MooseFS import warm-up.

**The correct technique was already in the same function.** `_assert_pin_engaged`'s TOKENIZER leg proves
provenance from the filesystem instead: `transformers.utils.hub.cached_file(model_id,
"tokenizer_config.json", revision=pin, local_files_only=True)` then assert `f"snapshots/{pin}"` is in the
resolved path. Its docstring even states the reason — "tokenizers store no `_commit_hash`". So the
author knew the private attribute was unavailable in one place and used the robust technique there; the
model leg just never got the same treatment. The path proof is version-independent AND stronger: it
proves on-disk artifact provenance rather than trusting a field the vendor is free to rename.

## Proposed content

1. **`gotchas.md` entry.** Name the trap (a gate keyed on a private third-party attribute — leading
   underscore — is a latent version-bump failure, acutely so under this repo's per-issue pod version
   pins), name the canonical failure (`config._commit_hash` is `None` under transformers 5.x while the
   revision pin is fully engaged), and give the replacement recipe (`cached_file(..., revision=pin,
   local_files_only=True)` + `snapshots/<pin>` path assertion). State the general rule: a private
   attribute may be used opportunistically to PASS a gate, never to FAIL one.
2. **Evaluate a lint leg (decide after reading; do not implement blind).** A cheap AST/grep check for
   `getattr(<obj>, "_<name>", ...)` or `<obj>._<name>` from a third-party module inside an `assert` /
   `raise` condition. Note the false-positive risk on our own internal underscore helpers, so it may
   belong as WARN-only, or scoped to a known vendor-module list (`transformers`, `peft`, `trl`,
   `huggingface_hub`). If the false-positive rate makes it noise, the gotchas entry alone is the
   deliverable — say so rather than shipping a check people learn to ignore.

## Acceptance criteria

1. The gotchas entry exists, names the #2329 incident, and gives the `cached_file` + `snapshots/<pin>`
   replacement recipe concretely enough to copy.
2. If a lint leg ships: it flags the #2329 shape (a private-attribute read inside an assert condition),
   is clean on the current tree or its hits are triaged, and cannot pass vacuously (no resolvable
   targets ⇒ loud skip or FAIL, never a silent pass).
3. `LESSONS.md` index row updated if a new rule file is added (enforced by
   `workflow_lint.py --check-lessons-index`); no new red in the no-flags `workflow_lint.py` run.

## Provenance

workflow_fix_target: .claude/rules/gotchas.md (entry); scripts/workflow_lint.py (optional WARN-only leg)

Surfaced by the orchestrator while diagnosing a live crash during #2329 `q35_ladder_decay` Step 5
(`epm:progress` v154 carries the full diagnosis, HTTP evidence, and the fix rationale). Filed under the
workflow-fix-on-bug protocol: the bug itself is experiment code (fixed in-round), but the RECURRING trap
is a workflow-surface documentation gap, and the per-issue-version-pin architecture guarantees it recurs.
Not a duplicate of the existing MooseFS / EDQUOT / vLLM entries — this is a version-skew gate trap.
