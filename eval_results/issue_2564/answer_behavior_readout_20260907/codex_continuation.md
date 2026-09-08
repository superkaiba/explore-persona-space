# Codex judging continuation

The pilot is accepted with qualifications. The first production packet passed
the planned 256-answer completeness and envelope check. The remaining frozen
roster is released. No actual-label readout has been fit yet.

Run from `/home/thomasjiralerspong/.codex/worktrees/issue2564-answer-behavior`.
The research root is
`/home/thomasjiralerspong/.codex/research/answer-behavior-readout-20260907`.
Use the existing project environment. The earlier API resume instructions are
historical; do not retry the rejected key or mix those artifacts with Codex
judgments.

## Annotation continuation

`codex_main_jobs.json` in the research root is the coordinator's live dispatch
ledger. `codex_main_jobs_before_dispatch.json` is the immutable original roster;
`codex_main_roster_preflight.json` records its exact 280-packet coverage and
content-only checks. All 2,048 answers receive five fresh contexts for each of
seven properties. The requests use 256 answers per packet. No sample, repeat,
rubric, model or reasoning-effort change is authorized by this runbook.

Dispatch pending requests through the collaboration subagent tool with the exact
saved request, `fork_turns: none`, and no overrides. Use a fresh agent for each
packet. Record actual canonical IDs and dispatch observations with the adapter's
`mark` command. Mark completion only after an actual completion notification,
then import the ledger. A file's mere existence is not completion evidence.
Malformed, incomplete or replaced outputs must fail validation and remain
preserved; never fabricate labels or silently discard a packet.

The first packet took 591.49 seconds between coordinator observations and
produced 256 valid records. Extrapolating that one persona packet to two workers
would suggest roughly 23 hours for the full roster. This is a provisional timing
estimate, not a measured estimate for the other properties. Five repeated
ratings remain shared-model judgments with uncontrolled sampling, not IID draws
or human validation. Instruction-enforced answer-only access is not technical
isolation.

## After all ratings finish

Use the adapter to import, aggregate and validate the complete main roster:

```bash
uv run --no-sync --project /home/thomasjiralerspong/explore-persona-space python scripts/issue2564_codex_judgments.py import --root /home/thomasjiralerspong/.codex/research/answer-behavior-readout-20260907 --part main --ledger /home/thomasjiralerspong/.codex/research/answer-behavior-readout-20260907/codex_main_jobs.json
uv run --no-sync --project /home/thomasjiralerspong/explore-persona-space python scripts/issue2564_codex_judgments.py aggregate --root /home/thomasjiralerspong/.codex/research/answer-behavior-readout-20260907 --part main
uv run --no-sync --project /home/thomasjiralerspong/explore-persona-space python scripts/issue2564_codex_judgments.py validate-main --root /home/thomasjiralerspong/.codex/research/answer-behavior-readout-20260907
```

The complete aggregate lives at `annotation_codex/main/labels.json`. Readout
defaults explicitly select that provider and save to `readout_codex/`. The loader
validates accepted pilot evidence and all aggregate/completion fingerprints.
Run one outer fold, resume the complete fit, then summarize:

```bash
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072 uv run --no-sync --project /home/thomasjiralerspong/explore-persona-space python scripts/issue2564_answer_behavior_readout.py stage=fit first_fold_only=true
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072 uv run --no-sync --project /home/thomasjiralerspong/explore-persona-space python scripts/issue2564_answer_behavior_readout.py stage=fit
OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8 MALLOC_ARENA_MAX=2 MALLOC_MMAP_THRESHOLD_=131072 uv run --no-sync --project /home/thomasjiralerspong/explore-persona-space python scripts/issue2564_answer_behavior_readout.py stage=summarize
```

Review actual coverage, target reliability and missingness, paired within-target
readouts and their conditional bootstrap intervals, null mobility, and declared
sensitivities. Do not infer a universal high-versus-low ordering from unlike
metrics. Archive all raw judgments, dispatch evidence, accepted instrument,
inputs, fits and analysis outputs with verified remote content hashes. Publish
the reviewed scientific report and notify the parent agent. Do not edit the
paper or send Slack messages.

The accepted pilot and first seven completed main packets (1,792 main ratings)
are also [archived as a verified partial checkpoint](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/322c8fd82e6fb917ddf6c5834e0ef40b8d999685/issue2564_minpair/answer_behavior_readout_20260907/raw_completions/codex_main_checkpoint_007).
All 197 files and their 7,286,137 bytes match the local content hashes. This
checkpoint includes actual dispatch records and is explicitly incomplete; it
is not a main completion marker or a behavioral result.

A later [16-packet progress snapshot](codex_main_progress_016.json) records
the complete first voice and topic passes (4,096 main ratings), with two warmth
packets running. Median observed packet intervals were 568 seconds for voice
and 506 seconds for topic; one topic packet took 1,982 seconds. This variability
limits the original timing extrapolation. The [coordination log](codex_coordination_events.jsonl)
records a capacity-limited dispatch and procedural progress checks on that longer
packet. It finished with all 256 records valid. The checks requested progress
and permitted incremental output, while explicitly keeping the rubric and
annotation requirements unchanged. All five repeated ratings remain required.

The [32-packet raw checkpoint](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/af84ed0817fbe620593879112834b330107e5507/issue2564_minpair/answer_behavior_readout_20260907/raw_completions/codex_main_checkpoint_032) contains 8,192 completed main ratings and the accepted pilot: all 326 files (25,154,703 bytes) passed exact path and content-hash verification. The uploader encountered a transient 504; its retry found the commit had landed and the subsequent pinned-revision verification passed. This remains a partial checkpoint, not completed main annotation. Local annotation continues beyond this snapshot.

A shared-disk check during annotation found approximately 34 GB free on the root filesystem and 0.59 GB on the data disk. The checkpoint adds only 25 MB, and no shared or active artifacts were deleted. Readout launch must recheck the applicable disk preflight; this is an infrastructure warning, not a scientific outcome.

The [first complete rating pass](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/dcc35cd108536cab7eb2ae3179c0a377c44c25c0/issue2564_minpair/answer_behavior_readout_20260907/raw_completions/codex_main_checkpoint_056) is now archived and verified: 56/280 packets, 14,336/71,680 main ratings, 448 files and 42,333,373 bytes. This checkpoint includes the source-population addendum. Four repeated passes remain; no main aggregate or readout fit is complete. The [measured timing snapshot](codex_main_progress_056.json) records 4.54 hours of elapsed time for the first pass. Applying each property's observed mean interval to the remaining 224 packets at two workers projects about 15.17 further hours, including the running packets at full predicted duration. This is a scheduling estimate with observed tail variation, not a finish deadline or uncertainty interval.

Packet 063 initially omitted one of its 256 IDs. The original 255 judgments and raw archive remain unchanged. Following [independent review](codex_packet_063_repair_review.md), the same judge supplied only the missing item in a separately recorded followup; exact-ID validation now passes. The adapter hashes and archives the supplement alongside the failed original, and import resume rechecks previously archived records. This completes the original rating; it is not an extra repetition or independent validation. The one supplemental rating had a different immediate batch context. The [failure audit](codex_packet_063_omission.json) and [repair verification](codex_packet_063_repair_verified.json) retain the evidence.

The [preserved packet 063 and supplement](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c2de63d58f95fce93889751b6e844bdbd45477e7/issue2564_minpair/answer_behavior_readout_20260907/raw_completions/codex_packet_063_repair) are remotely verified: all 12 files and 168,580 bytes match the manifest content hashes. The [verification receipt](codex_packet_063_archive_verified.json) records a transient server timeout and a corrected read-only post-upload check; no raw evidence was overwritten.

The [80-packet progress snapshot](codex_main_progress_080.json) records 20,480 completed main ratings: two full rating passes for voice, topic, and warmth, and one for the other four targets. Two confidence packets are active. All completed packet IDs and labels validate, including the documented missing-only completion. This remains annotation progress, with no final main aggregate or actual-label readout.

The [second complete rating pass](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/tree/c03c4a5f1bd82c4ee5d3c1962de865cedccbaa8c/issue2564_minpair/answer_behavior_readout_20260907/raw_completions/codex_main_checkpoint_112) is remotely verified: 112/280 packets, 28,672/71,680 main ratings, all 732 files and 79,869,246 bytes matching the exact path and content-hash manifest. Every completed rating passes schema validation, including the preserved packet 063 supplement. The [timing snapshot](codex_main_progress_112.json) records 8.51 elapsed hours; property-specific observed means project approximately 11.24 additional hours at two workers for the remaining 168 packets. This is an operational estimate with variable packet times, not a finish deadline. The third pass is running; three of the five passes remain, and no actual-label readout is complete.

An [advance CPU readout preflight](advance_readout_preflight.json) passed as a diagnostic without launching a fit. The documented per-invocation low-disk overrides were logged against a conservative 3 GB footprint, with no data-disk staging or model capture. That record is not a substitute for the fresh launch preflight, which must recheck actual free space immediately before fitting. No shared or active artifacts were deleted.

The [128-packet progress snapshot](codex_main_progress_128.json) records 32,768 valid main ratings: three complete voice/topic passes and two for the other five targets. Two warmth packets are active. The root filesystem has 26,081,189,888 bytes free at this check; no cleanup was performed. This is annotation progress only, with no final aggregate or actual-label readout.

The [144-packet progress snapshot](codex_main_progress_144.json) records 36,864 valid main ratings: three complete voice, topic, warmth, and confidence passes and two for the remaining three targets. Two formality packets are active. The root filesystem has 29,098,196,992 bytes free at this check; this experiment performed no cleanup. This remains annotation progress, with no final aggregate or actual-label readout.
