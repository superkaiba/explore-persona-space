---
name: reused-module-internal-consumer-sweep
description: A crash-fix "class sweep" grep scoped to the issue script (R.helper) is blind to reused-module INTERNAL unqualified calls — grep the helper name INSIDE every reused function the driver invokes on the new data (#2333 R5)
metadata:
  type: feedback
---

Rule: when a fix relaxes an invariant on records a driver CREATES (e.g.
`prefix_end == 0` now legal) and the implementer's class sweep is a grep for
the qualified call (`R\.slot_position`) in the ISSUE script only, re-run the
sweep INSIDE the reused module: grep the bare helper name (`slot_position(`,
`_slot_state(`) within every reused function the driver actually invokes on
the new-shape data, then trace which of those call sites are LIVE for this
driver (seam kwargs like `run_one`/`payload_fn` can make some dead).

**Why:** #2333 R5 fixed the bank crash + the three direct `R.slot_position`
call sites, but the reused `run_injection_gate` internally calls unqualified
`slot_position` (issue2162_run.py:1239), which asserts `1 <= prefix_end` for
EVERY slot — and both S2 gate spots deterministically pick `a=bare__q1`
(pe==0 under q35), so the relaunch would have burned the bank GPU capture
and crashed one phase later at the same assert class. The marker's own
sweep line ("no raw parent call remains") was true and useless — wrong
scope. A 3-line live probe (sorted pair ids + `R.slot_position(5,0,"ce")`)
turned the trace into a measured deterministic crash.

**Second confirmed instance (#2329 q35 R1):** a fork passed every EXPOSED
seam of a reused gate (`run_injection_gate`'s `payload_fn`/`spots`/`ids_fn`)
but the gate's INTERNAL second-row filter called `pe_excluded_reason`, which
hardcodes the parent donor-map keys (`donor_maps["crosstype"]`) and pair-id
values — the fork's `{"null_sameval","null_xtype",...}` context-id maps
KeyError on the first pe×non-steered spot. Probe shape: call the reused
gate's internal helper directly with the fork's donor maps; and when a
reused helper grows a NEW caller-owned predicate seam mid-round, check the
fork call site actually THREADS it (fixed in-round at c46f29bf0c33 with
`pe_second_row_ok_ladder`).

**How to apply:** any crash-fix round that (a) makes previously-impossible
record shapes reachable and (b) reuses parent-module gates/queues on those
records — and equally any FORK that reuses a parent gate with donor-map /
registry conventions of its own. Enumerate the `R.*` symbols the driver imports (`grep -o
"R\.\w*" | sort -u`), read each position/slot/record-consuming one for
unqualified internal helper calls, and probe the concrete selection (which
pairs/spots the gate picks) rather than arguing distributionally. Dead-path
discriminator: a `run_claim_queue(run_one=...)` seam means the module's own
block runner (`_block_cells`) is NOT live. Pairs with
[[fails-pre-fix-probe-parent-commit]] (certify the fixed sites) and
[[banked-parent-dual-schema-equivalence]] (probe the producer's own
consumer read).
