# Independent methodology audit

An independent critic audited the committed runner, blinded-reader artifacts, and realized analysis after the corrected experiment completed. The critic found no fatal or major issue that invalidates the report's conservative `inconclusive_pipeline_control_failed` verdict.

## Independently reproduced checks

- Rebuilt the 129-row census and every one-axis donor/recipient relationship.
- Verified that the primary hook edits `model.model.layers[19]` and the secondary arm edits blocks 0–27.
- Matched all 114 applied payload hashes to the saved donor states, including layer order and final-context-token positions.
- Confirmed batch-size-1 parity between capture and generation, exact forced opening tokens `[2582, 510]` (`Response:\n`), and 15/15 exact self-patch matches in each setting.
- Confirmed that all 129 sequences end at their first EOS, contain 98–401 generated tokens, and decode exactly to the saved text.
- Recomputed task transfer 0/18, subject transfer 0/18, and format transfer 0/6 for both layer settings.
- Reconstructed all eight reader packets from the frozen key and generations; matched them to the outbound bytes, raw responses, parsed sidecars, and final annotations; observed one agent-message event and no tool event in every transcript.

## Limitations that remain

1. Forcing the first two answer tokens removes the final-block patch's immediate first-token-logit pathway. Later effects must propagate through cached state and subsequent computation. This controls the greeting artifact but may attenuate a weak causal effect, so a null is specific to this intervention.
2. The exact-pipeline format control failed in both layer settings. That prevents the zero task/subject transfer rates from being interpreted as evidence against context-state encoding.
3. The substitute Codex reader remained arm-blinded and its no-tool transcript is auditable, but it changed the planned judge family and used built-in runtime system context. The realized temporary-directory name also contained the issue number, although it exposed neither condition nor hypothesis.
4. The bullet/paragraph label measures broad format category, not every instruction detail. For example, a response can be categorized as bullets without satisfying the request for exactly five bullets.

## Bugs found and fixed during audit

- `analysis/DONE.json` originally hashed the report before its trailing newline was written. The writer and a regression test now hash the exact on-disk bytes.
- The analyzer's positive-result branch now enforces the declared recipient-specificity and coherence safeguards and reports format gates separately by layer setting.
- The analyzer now reconstructs every blind request from the frozen key and generation text, replays leakage and event audits, and verifies raw response = parsed sidecar = final annotation.
- Rerunning analysis no longer truncates `annotation_corrections.jsonl`.
- Future temporary judge directories use a neutral `blind-reader-` prefix.

The fixes are committed through `13eb5c55436`; they do not alter any generation or blind annotation, and the realized verdict remains inconclusive.
