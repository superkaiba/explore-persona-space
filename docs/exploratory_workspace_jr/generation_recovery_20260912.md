# Native generation interface recovery

The first component pilot, pinned to
`7ee828c58e037664f9d2ac1f576b13a0d45abe45`, completed its dictionary phase
and failed before any rollout was sampled. Transformers 5.16.1 defaults
`apply_chat_template` to `return_dict=True`; the returned `BatchEncoding`
could not be serialized as the generation contract's flat token-ID list.
The fix explicitly requests `return_dict=False` and validates the result.

A same-instance CPU probe with the exact pinned 27B tokenizer returned
`list 16 True` (list type, 16 IDs, every ID a Python integer). The actual
runtime signature confirms `return_dict: bool = True`. The regression
test models that default and verifies cap recovery and checkpoint reuse
with the explicit flat-ID request. Four capture tests and focused Ruff
checks pass; the upload directory-filecount check also passes.

The failed output root `/workspace/workspace_jr/primary_component_pilot`
is retained with its failure receipt and log. Recovery uses the fresh root
`/workspace/workspace_jr/primary_component_pilot2`, with a correspondingly
distinct Hub prefix. Dictionaries are rebuilt under the new producer
identity; the native calibration pairs are retained because their exact
source files are byte-identical. No failed-run checkpoint is resumed.
The launch record must confirm the old service and its process group have
exited, verify the full fix SHA on the instance, and record the new unit.

Compute shape is unchanged: one A100 80GB, batched vLLM generation, BF16
native capture, GPU sparse pursuit and vectorized MLP fitting; the frozen
64/16/32 contexts and five seeds remain unchanged. This experiment's new
uploader and launcher have no sibling experiment consumers. No shared
generation-library API was changed.

The focused workflow check passes for directory file counts. The wider
workflow lint has a separate inherited stale-grandfather error, and the
sparse checkout hides required workflow files; neither is reported as a
full lint pass. Two earlier native-source lint findings remain recorded
for a source-compatible revision before further native construction.
