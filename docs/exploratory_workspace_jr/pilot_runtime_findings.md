# Runtime findings and corrections before main outcomes

On September12 the primary component pilot at source348b11a45a26147f1270356edfb6d3dbfd6f143c
completed all generation and token capture phases, then failed its repeated-input
assertion before the first sparse decomposition. All captures and generations
were byte-verified on the data Hub atd74cd5233f88f02188d77650ebc3dc29cf40e1a9,
underexploratory_workspace_jr/20260912/primary_component_pilot2.
The final failure receipt is persisted separately by the owning monitor.

The context-last reads differed across differently padded answer batches in
BF16. The first failed context had an absolute coordinate difference of0.5;
the prior coordinatewise assertion allowed0.002 absolute and0.002 relative.
This is an extraction failure caught by the pilot, not a component-predictability
result. Do not relax that assertion or discard the differing reads.

Capture the input once from only the context tokens, using frozen context order
and batches of16, and reuse that exact vector across all five rollouts. The batch
membership contains no generated answers. Retain every original answer-batch
context read as a numerical diagnostic. The already saved token-level answer
states and raw generations can be reused after byte verification; record the
original and recovery producer identities separately. No component outcome was
available when this correction was chosen. Apply the same convention to both
models and all main cells; higher-K measurements reuse the frozen input vector.

Historical eager recapture failed the fixed1% input-state relative-Frobenius
gate for both models: primary1.13518%, comparison1.28911%. Their answer-mean
errors were0.25208% and0.32727%, respectively, and row-cosine checks passed.
These failures block applying historical frozen maps to the new native inputs.
No threshold is changed. An explicitly labeled SDPA recapture diagnostic tests
whether the historical default attention backend explains this discrepancy;
its output cannot satisfy the ordinary eager reuse gate.

The comparison model's dim-batch32 benchmark took31.83seconds forJ and29.19
forR, but differed from its native dim-batch8 matrices by0.004115 and0.002648
relative Frobenius norm. Both exceed the fixed1e-5 engineering parity threshold.
Do not adopt that batching change or run the64 candidate under a passed label.
Complete comparison calibration at the validated dim-batch8 setting, reusing
only the verified dim-batch8 first pair. Its measured first-pair wall was139.58
seconds;118 additional pairs project to about4.6GPU-hours on the existing
comparison A10080 instance, below its01:52UTC September13 STOP fence if started
by20:45UTC. This is a pilot-based estimate, not a completion claim.
