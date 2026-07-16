# judge dirs carry a consolidated judge_raw.json (#1315)

Per (cell, rung) judge dir on the data repo (`raw_completions/{tier1,tier2,parity}/
<cell>/<rung>/judge/<side_ctx>/`), alongside the 100-200 hash-named JudgeCache files
there is ONE `judge_raw.json` with `all_scores` per DRAW (key
`<ctx>-<side>-q###-c#__#####__##`), including persisted `error: true` transport
failures (verbatim 529 messages). Use it to: (1) count draw-level transport losses
per arm (llm-judging rule 24 audit), (2) detect item-level censoring (items with ALL
draws errored), (3) recompute rates from surviving draws (mean>50 per item) and
check them against persisted selection/parity rates, (4) map q/c indices into the
sibling `completions__<side>__<ctx>.json` (`questions[qi]`, `completions[qi][ci]`)
for judged sample selection — all with ONE download per read instead of paging the
hash files. Non-round rate denominators (e.g. 0.6559 = 61/93) betray dropped items.
