---
name: chat-template-conditional-block-prefix-drift
description: Think-capable chat templates render turns CONDITIONALLY on position — an N-turn render is NOT a prefix of the (N+1)-turn render; tail-anchor teacher-forced spans, never prefix-render + startswith
metadata:
  type: feedback
---

On think-capable chat templates (Qwen3/3.6 class), a rendered turn's bytes
depend on what comes AFTER it: the template attaches the empty
`<think>\n\n</think>` block only to assistant turns after the LAST user
query. So `render([u1, a1])` (a1 last → think block present) is NOT a
prefix of `render([u1, a1, u2])` (a1 mid → block stripped), and any
teacher-forced span derived as `prefix-render + startswith/len(prefix)`
fails on 100% of rows — deterministically, content-independently.

**Why:** #2378 r13 — the P4 wave-1 `chat_user_real` cell harvested 0/10,000
kept (every row `span_mismatch`) because the producer checked
`rendered_full.startswith(prefix + u2)` with prefix from the 2-turn render.
Fix: anchor the final turn from the content-independent template TAIL
(`rendered_full.endswith(HEADER + u2 + "<|im_end|>\n")`, span by arithmetic
from the end — the #1776 recipe), producer and capture consumer through ONE
shared helper (`gen._user_real_span` / `_user_real_row`). Verified
10,000/10,000 kept on the same pool.

**How to apply:** any rig that derives char/token spans in a multi-turn
chat render must derive them from the FULL text it will teacher-force —
tail anchor for the last turn, divergence-anchor (`_divergence_anchor`)
against that text for interior turns — never from a separately rendered
shorter prefix. Corollary (REVISED, #2378 r14 — the r13 corollary
"template fidelity for teacher-forced arms" was itself a review blocker):
when a teacher-forced arm is PAIRED with a sampled arm under a
shared-context contract (identical v_C/v_P bytes across arms), the pair
contract WINS over template fidelity — teacher-force the DIRECT JOIN
`prefill-prefix + turn + TURN_END` as a declared deviation
(`gen._render_user_real_tf`), so both arms share byte-identical context;
using the template's own (N+1)-turn render for the teacher-forced arm
shifts every context byte at the conditional block and deterministically
fails the pair assert. Template fidelity remains right only for an
UNPAIRED teacher-forced arm. Either way, DISCLOSE the render deviation.
Sibling family: [[bpe-zero-width-span-plain-text-delimiters]] (BPE seams;
this entry is the template-CONDITIONAL-block sibling — no BPE involved).
