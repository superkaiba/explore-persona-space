---
name: smoke-arch-arm-registry-grammar
description: the arm-registry marker line must be the BARE structured form (source= file= n= members=); derivation commands go in prose after it, never inline
metadata:
  type: feedback
---

The `epm:smoke-architecture-check` marker's `arm-registry:` line must be one of
exactly two line-anchored forms (#2176):
`arm-registry: source=<expr> file=<path> n=<int> members=<sorted-comma-list>`
or `arm-registry: N/A — <reason>`. Putting the derivation COMMAND on that line
(backticked `--list-phases` invocation, a `python -c` registry dump) is a
grammar REFUSE even when the substance is verified correct — move commands +
secondary registries (score arms etc.) to a prose line after it.

**Why:** #2224 r5 Critical: a substance-correct marker with the command inline
bounced the round; Step 6d.0 runs the same checker POST-provision, so an
unfixed marker wedges after the pod exists.

**How to apply:** after posting, verify with
`uv run python scripts/task.py check-smoke-arch-registry <N> --repo-root <wt>`
expecting `OK — registry-complete`; read the rc BARE (`> file; echo $?`), never
after a pipe (post-pipe `$?` reads the tail's rc and masks a REFUSE).
