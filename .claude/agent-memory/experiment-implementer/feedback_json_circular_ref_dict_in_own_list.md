---
name: json.dumps circular reference from nesting a dict inside its own list
description: A verdict/result dict that aliases a sub-object (final = leg1) and then stores the container of that sub-object back onto it (final["legs"] = [leg1]) makes leg1["legs"][0] is leg1 — json.dumps raises "Circular reference detected". Build the top-level dict fresh; never mutate a member into its own container.
type: feedback
---

`json.dumps(obj)` raises `ValueError: Circular reference detected` when any object reachable from `obj` references `obj` (directly or transitively). The trap is aliasing: a multi-leg gate/result builder sets `final = leg1` (an alias, not a copy), accumulates `legs = [leg1, ...]`, then writes `final["legs"] = legs` — now `leg1` contains a list that contains `leg1`. The dry-run smoke is where it surfaces (the pilot writes its verdict JSON), AFTER all phases ran, so it looks like a late failure.

**Why:** bit #642 r5 — `p0_5_pilot_gate_v9` set `final = leg1` then `final["legs"] = legs` (legs holds leg1); `gate_path.write_text(json.dumps(final))` raised. EXIT=1 with a traceback whose only app frame is the `json.dumps` call site.

**How to apply:**
- A result/verdict object that carries its own component objects (legs, rounds, sub-runs) must be a FRESH top-level dict that COPIES the chosen component's fields, never an alias of one component. `final = {"chosen_lr": ..., "arms": chosen_leg["arms"], "gate_pass": chosen_leg["gate_pass"], "legs": legs, ...}` — `chosen_leg` stays a member of `legs`, `final` is separate.
- If you must keep the alias, store COPIES in the list (`legs = [dict(leg1), dict(leg2)]`) or pop the back-reference before dumping.
- Smoke catches it only if the JSON-writing branch runs; the v9 dry-run pilot did, which is why per-phase end-to-end dry-run smoke (not just import/unit) is load-bearing.
