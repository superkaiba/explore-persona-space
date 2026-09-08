# Repaired China data-quality report

This report covers the fresh v2 generation/capture data, not its still-pending
behavioral judgments or scientific analysis. Checks use the complete saved
responses and tensor row indices; manifest counts alone are not the evidence.

## Coverage and structure

Production contains all 85 approved subjects from 12 topics: two languages,
four subject/country-wording conditions, two framings, and eight draws per
prompt. All 1,360 prompts and 10,880 distinct prompt–draw responses are present.
Each of the 16 language/content/framing cells contains exactly 680 responses.
The separate smoke run contains 10 subjects, 160 prompts, and 1,280 responses;
it is not included in production findings.

All context and answer tensors are finite fp32, with three layers (14, 19, 26)
and width 3,584. Actual tensor indices cover every expected row without
duplicates or missing captures. All 12,160 smoke-plus-production answer token
boundaries were checked against saved generation tokens and the pinned
tokenizer, including the turn-end/newline tail.

## Missingness, truncation, and repaired manipulation

There are no empty responses. Production has 10,879 normal stops and one
length-limited response. That response remains in the data with its flag;
its cell has 1/680 truncations, below the frozen 2% selective-extension trigger.
No extension was required. Production completion lengths range from 16 to
2,048 tokens (median 340; mean 339.902; 90th percentile 563).

All 680 country-wording comparisons produced nonzero context differences at
each captured layer. Minimum difference norms are 5.46953, 9.14498, and 30.06685
at layers 14, 19, and 26. This resolves the earlier identical-input problem.
The manipulation measures *additional explicit country wording conditional on
a named subject*; it does not remove all information identifying a country.
Nonzero differences establish that the manipulation reached the representations,
not which subspace contains the subject or its framing.

## Persistence and process state

The exact inventory contains 98 durable files (765,126,458 bytes): 20 tensors
and 78 text artifacts, including original/final responses, checkpoints, inputs,
manifests, sentinels, and logs. The 311 excluded files are explicitly enumerated
regenerable input-stage or upload-verification copies. All 78 original text
files were reconstructed byte-for-byte from the remote archive; all 20 tensor
SHA256 hashes and sizes match revision-pinned remote objects, with separate
producer download/consumer-open verification. The nine direct files needed
by the downstream production consumer were additionally downloaded and hashed.

The launcher and workload exited, the final source inventory was unchanged,
and the sole v2 H100 pod was terminated after persistence verification. The
live lifecycle check subsequently reported no issue-952 pod. The v1 experiment
was not overwritten. New judging and CPU analysis are not yet complete.

Data revision:
`00406a09b599d6523e678f2bac7fabe6873c99f3`.
Exact-text archive revision:
`bbbe7e6e430910afe034caa66ee540cf01b1e1c3`.
Repository: `superkaiba1/explore-persona-space-data`.
Prefix: `issue952_position_divergence/followups/china_refusal_wording_withholding_v2/attempt1`.
Raw response SHA256:
`b63d8423b83524cc46513559dec44291e8fd84f4ba77412461f17f438a405858`.

## Interpretation limits

The panel was frozen before fresh Qwen outcomes, but v2 is an openly revised
post-v1 design. It is not a newly preregistered confirmation. Behavioral
judging uses a frozen rubric with two independent Codex reviewers and a fixed
overlap sample; there are no human labels. Reading and blinding are enforced
by reviewer instructions, not filesystem access restrictions. Until the
complete judgments and planned analyses finish, this report supports only
technical validity and coverage, not a China-refusal semantic conclusion.
