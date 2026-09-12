# Historical capture compatibility review

The independently reviewed parity script stages the first32 frozen hash-ordered
calibration contexts with original captures. It verifies exact pinned dataset
bytes, capture row IDs, the length-sorted500-row shard layout and both stored
x/y fields. The original metadata and raw completion schema were read directly.
It reuses three pure token-boundary helpers from audited producer commit
`9b896ccc6b65e1d7322d3c7e67f6fe92883a0f0c`, saving their complete source files.
Only those named function definitions are compiled; generated text is data.

Native recapture preserves all retokenized completion IDs for the forward pass,
uses the original final-context position, and averages the original stripped
answer span after FP32 conversion. A whitespace-boundary smoke confirmed that
the x position remains at the context boundary while the answer span excludes
leading/trailing whitespace. The reviewer independently checked these formulas
against the historical producer source. No experimental test outcomes were read.

Two operational findings were fixed: an existing empty staging directory is
accepted consistently, and an unsuccessful parity threshold exits nonzero after
persisting its evidence. Frozen membership/order is independently rechecked at
capture time. The reviewer reported no remaining blocker. Actual checkpoint
compatibility remains unmeasured until the GPU recapture runs; successful staging
or a toy boundary check is not a passed native parity gate.
