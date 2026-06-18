---
name: rev-parse of a full 40-char SHA is a no-op existence check
description: git rev-parse <40-hex> echoes the string without checking the object DB — fabricated SHA tails pass it and 404 on GitHub; verify with git cat-file -e <sha>^{commit} or rev-parse a SHORT prefix
type: feedback
---

`git rev-parse <full-40-char-hex>` returns the string verbatim WITHOUT consulting the object database, so it cannot validate a SHA you typed from memory. A fabricated tail passes "verification" and then 404s in every GitHub blob/raw URL.

**Why:** Incident task #480 round-2 re-fold (2026-06-11): I extended the abbreviated commit `740414fc1` with an invented tail, "verified" it via `rev-parse <full>` (printed SHA OK), pasted it into a Reproducibility blob URL — the verifier's URL-exists check caught the 404. Real SHA was `740414fc17657fba...`.

**How to apply:** To get a full SHA, always `git rev-parse <SHORT-prefix>` (forces an object-DB lookup) or `git log -1 --format=%H`. To verify an existing full SHA, use `git cat-file -e <sha>^{commit}`. Never type hex beyond what a git command printed.
