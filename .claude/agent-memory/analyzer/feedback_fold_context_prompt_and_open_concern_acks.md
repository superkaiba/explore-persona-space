---
name: fold-context-prompt-and-open-concern-acks
description: Fold rounds — Context prompt goes in a blockquote (a fence trips the cherry-picked sample check); open concerns need verbatim ids in Takeaways/Results (defer-concern is user-only); footer round credit needs the literal "same-issue follow-up round `label`" clause
metadata:
  type: feedback
---

Three mechanical fold-round rules, all hit on the #2215
`discrimination-battery-expansion` fold (2026-08-23):

1. **A follow-up round's verbatim Context prompt goes in a BLOCKQUOTE,
   never a fenced code block.** The verifier's cherry-picked-label check
   counts every fence as a sample block and FAILs when its prelude lacks
   a random-sample/cherry-picked token. Linkify bare `#N` tokens inside
   the quoted prompt (`[#2202](https://eps.superkaiba.com/tasks/2202)`)
   to keep the bare-issue-ref scans quiet.
2. **Open (raised/verified-open) binding concerns cannot be acknowledged
   by an analyzer-written `<!-- concern-deferred: id -->` marker** —
   `task.py defer-concern` is user-only, and check 14b FAILs a marker
   with no `deferred` event in concerns.jsonl as a FABRICATED deferral.
   The compliant analyzer route is mechanism 1: name each open concern
   id VERBATIM (backticked is fine; the ids are hyphenated, so the
   multi-underscore slug audit is quiet) in `## Takeaways` or `## Results`
   prose (#2254 precedent). Keep markers ONLY for ids whose ledger
   already holds a real `deferred` event; a marker on an `addressed`
   concern is a stale-marker WARN — remove it and state the fix.
3. **The conciseness budget credits +250 words per round ONLY via the
   literal footer clause `same-issue follow-up round \`label\``**
   (`_V4_FOOTER_ROUND_CLAUSE_RE`); "Follow-up round `label`" earns
   nothing.
4. **Pod-pushed result commits diverge the local worktree branch — the
   "figure tracked at live refs" WARN is the tell** (#2476 fold,
   2026-08-24): the pod pushed the artifact commit to origin/issue-N
   while the local worktree held agent-memory commits on the same
   parent, so body-pinned figures existed at the pinned SHA but on NO
   live local ref. Fix before drafting figures into the body:
   `git -C $WT pull --rebase origin issue-N` then push (worktree only —
   never the repo root).

**Why:** each cost one gate round on the #2215 fold; all three are
invisible in bare `--file` mode (concerns.jsonl only binds under
`--issue`), so run the final gate with `--issue <N>` before returning.

**How to apply:** every same-issue fold that adds a round prompt, faces
open concerns, or grows past the prose budget. See also
[[fold-round-gate-mechanics-1336]].
