---
name: Infra channel-kill-vector plan review
description: How to review startup-script/pipe-hardening infra plans (#607 pattern) — enumerate residual channel writers with per-line bounds; check sentinel-placeholder truthification consumers
type: feedback
---

Two checks that decide infra plans hardening a bounded-channel kill vector (e.g. #607: GCE metadata-runner `bufio.Scanner: token too long`):

1. **Enumerate ALL residual writers to the hazardous channel and verify each is per-line bounded + error-guarded.** After a global `exec >> log` redirect, the residual pipe writers are: pre-redirect prelude lines, guarded heartbeats, and any EXIT-trap diagnostic tail. A `tail -n K | cut -c1-M` tail is safe (cut bounds per line, newline-terminated, M « 64 KiB scanner limit); a bare `tail -n K` of a file that can contain one giant newline-free line is NOT line-bounded (`-n` counts lines, a \r-progress blob is one line). Same trap applies to SSH drain stanzas: check whether the Python consumer side already truncates (gcp.py:2279 caps excerpt `[:2000]`) — then a pathological giant line is a transient SSH-capture blip, not a contract break → Concern, not Must-Fix.

2. **When a plan replaces a hardwired placeholder metric with truthful values** (#607: `last_log_mtime_sec_ago` from `10**9` → real), grep ALL consumers for sentinel-keyed semantics (`== 10**9` skip branches, `> threshold` predicates that were vacuously true). Direction matters: placeholder→truthful that makes a predicate strictly more accurate is safe; a consumer using the placeholder as "unknown, skip" could newly fire on quiet phases. In #607 `backend_poll.py` only serializes the field — clean.

**Why:** #607 v1 got APPROVE because every residual writer was bounded and the only unbounded path (drain tail of a giant log line) was already capped consumer-side — these two checks were the entire decision surface; nothing else in the 7-question brief was close to conclusion-changing.

**How to apply:** any `kind: infra` plan touching startup scripts, log redirection, serial/pipe channels, or poll-metric truthification.
