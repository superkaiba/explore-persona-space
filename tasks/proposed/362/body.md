---
title: 'Two-marker setup: anchor closer to persona mimicry first'
kind: experiment
tags:
- todo
- mentor-followup
created_at: '2026-05-11T23:33:17.000Z'
has_clean_result: false
sagan_id: 17c54800-afb7-4083-b2fe-9f2271db5049
sagan_number: 362
priority: normal
---
The two-marker chunk setup in #281 (train persona1 on `<A> answer <B>`, train persona2 on `<A>` alone, check whether persona2 emits `<B>`) currently sits further from the persona-mimicry baseline than it could. Before drawing conclusions from the no-transfer result, anchor the experimental design closer to the standard persona-mimicry setup so we can tell whether the absence of transfer is real or an artifact of the divergent recipe.

Concretely: identify what's different from the persona-mimicry baseline in #281's training and inference setup (data format, loss masking, marker tokenization, end-of-sentence handling), and run a sweep that progressively moves the two-marker design toward persona-mimicry while preserving the chunk structure. If transfer appears under a persona-mimicry-adjacent setup, the original no-transfer claim was design-dependent.

Source comment (mentor update, 2026-05-11):
> Make more similar to persona mimicry first

From mentor update on #281 — *Fine-tuning one persona on a two-marker chunk and another on the start marker plants the end marker at every donor answer's end, not chained to the start.*
