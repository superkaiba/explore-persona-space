# Frozen-map comparison and generic-chat overlap diagnostic

## Matched held-out ID/OOD

Spearman correlation of the same fixed answer direction on mapped answers. Both maps use identical evaluation rows within each dataset. OOD is the equal-weight mean of dataset-specific correlations. Point estimates only; no new map-difference interval was fitted.

| Behavior | Regime | n | 18,793-pair map | 963,444-pair map | Small minus million |
|---|---|---:|---:|---:|---:|
| evil | in-distribution | 6468 | 0.500 | 0.597 | -0.096 |
| evil | OOD | 2217 | 0.158 | 0.242 | -0.084 |
| sycophancy | in-distribution | 16000 | 0.257 | 0.289 | -0.032 |
| sycophancy | OOD | 1304 | 0.249 | 0.255 | -0.006 |
| hallucination | in-distribution | 15987 | 0.052 | -0.108 | 0.160 |
| hallucination | OOD | 7185 | 0.258 | 0.182 | 0.076 |

The small map loses most clearly on harmful compliance (ID delta −.096, OOD −.084). Sycophancy differences are smaller (−.032 and −.006). Hallucination correlations rise on the small map (+.160 and +.076), driven by TriviaQA/NQ-Open; its observed-answer direction is negatively correlated there, limiting instrument validity. On SimpleQA alone, the small map is lower (.426 vs .449). Corpus, map recipe and answer-token pooling also change between the maps.

## Million-map generic chat, deliberately retaining map-fitting overlaps

Four methods share exact retained-rollout identities and raw fixed contrast directions. No behavior readout, new inference, judging or map fitting. Extraction overlap and invalid-label exclusions remain enforced. The primary pool is the historical 419 evaluation candidates; full cached 2,000-candidate results are reported separately.

| Behavior | Pool | n | Map-training overlap | Neither train nor validation overlap | Answer → mapped | Answer → real | Context → context | Answer → context |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| evil | historical_eval_including_overlap | 417 | 413 | 4 | 0.216 | 0.235 | 0.197 | 0.219 |
| evil | all_cached_generic_chat | 1987 | 1976 | 11 | 0.175 | 0.179 | 0.159 | 0.178 |
| sycophancy | historical_eval_including_overlap | 415 | 411 | 4 | 0.151 | 0.129 | 0.304 | 0.356 |
| sycophancy | all_cached_generic_chat | 1982 | 1971 | 11 | 0.150 | 0.125 | 0.273 | 0.305 |
| hallucination | historical_eval_including_overlap | 411 | 407 | 4 | 0.228 | 0.220 | 0.113 | 0.021 |
| hallucination | all_cached_generic_chat | 1967 | 1956 | 11 | 0.154 | 0.130 | -0.003 | -0.095 |

On the historical generic pool, the million-map advantage over the native context direction is unresolved for harmful compliance (delta .019, 95% paired CI [−.008,.047]), negative for sycophancy (−.153, [−.215,−.091]), and positive for hallucination (.115, [.035,.192]). Against applying the answer direction directly to context, the respective differences are −.003 [−.020,.012], −.205 [−.281,−.130], and .207 [.129,.283]. These are 2,000-draw pointwise group-bootstrap intervals conditional on the fixed map and directions, without multiple-comparison correction.

Overlap means a full-prompt/question/user-turn hash matches map training or validation under exact or lowercase/whitespace-normalized text matching. It includes shared earlier conversation turns, and does not by itself prove identical full evaluated prompts or activation pairs were trained on. This diagnostic does not establish conversation-disjoint generalization. Training and validation match flags are not mutually exclusive: three primary-pool rows per behavior match both; the full pool has 13 such validation matches.

Generic harmful compliance has only nine nonzero historical-pool scores (35 in the full pool). Generic hallucination uses a graded 0–100 trait rubric, unlike QA fabricated fractions. Observed-answer projection is a validation check, not an upper bound. No held-out paper claims or figures were replaced.

Producer: 52454addb1d810a679f62588309ebe7252651850. Completed CPU run and independent score/interval validation; source, predictions, bootstrap draws and all provenance are archived.

## Independent membership audit

Among the historical 419 candidates, 208 have one user turn and 211 have multiple user turns. All 415 overlap flags are explained by shared first-user text (411 exact, 415 normalized); there are no matches found only in later turns. Among the retained harmful/sycophancy/hallucination rows, 182/181/178 full evaluated prompts exactly equal the pinned canonical single-user chat-template rendering of a stripped query whose text occurs in map training; normalized equivalents number 185/184/181. Thus roughly 99% have conversation/first-user-prompt overlap, while about 43–44% are positively verified identical canonical single-turn contexts. The other matches do not establish identity of the full evaluated context. The post-run audit verifies 13 raw shard hashes and their prompts against the pinned prompt index, and reproduces all membership flags.
