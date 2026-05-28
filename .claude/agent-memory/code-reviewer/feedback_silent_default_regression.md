---
name: silent-default-regression-on-new-cfg-field
description: When a code change adds a new field to a shared dataclass AND wires it into a kwarg that wasn't previously passed, verify the dataclass default matches the previous implicit (library / framework) default. Otherwise every existing caller silently regresses.
metadata:
  type: feedback
---

When reviewing diffs that add a new explicit field to a shared dataclass
(like `TrainLoraConfig`) AND wire it into a downstream library / framework
kwarg that previously WASN'T passed (so the library used its own default),
the dataclass field's default value MUST match what the library would have
used. Otherwise every existing caller of the wrapping function silently
regresses.

**Why:** The change "make X explicit on cfg" sounds backward-compatible
because the new default looks like it preserves intent. But if the previous
code didn't pass X at all to the library, the library's default applied —
not the new cfg default. Reading "default=`foo`" on the dataclass and
"default=`foo`" in the previous-version-implicit-library-call doesn't tell
you they're the same value unless you LOOK UP the library's default.

**Concrete instance (task #397 round-1 code review):** Diff added
`TrainLoraConfig.optim: str = "adamw_torch"` and wired it into the TRL
`SFTConfig(**sft_kwargs)` call. Previously `optim` wasn't in sft_kwargs, so
TRL's default `adamw_torch_fused` applied. After the change, every existing
caller of `train_lora` (~10 scripts + leakage/runner.py) silently switched
from fused AdamW to non-fused AdamW — ~10-15% slower on H100. Implementer's
commit message claimed "existing callers see no behavioral change" but
hadn't verified the library default.

**How to apply:**

1. When the diff adds a new field to a shared dataclass and wires it into
   a previously-unpassed kwarg, IMMEDIATELY find the library function's
   default for that kwarg (`inspect.signature(SFTConfig.__init__).parameters[...]`).
2. If the library default != the new dataclass default, flag as a BLOCKER
   (silent behavior regression).
3. The fix is almost always: keep the dataclass default at the library
   default; have the experiment-specific call site override explicitly.
4. Add a regression test asserting `TrainLoraConfig().<field> == <library_default>`
   so future drift is caught.

This is a recurring trunk-diff pattern — keep the lens primed.
