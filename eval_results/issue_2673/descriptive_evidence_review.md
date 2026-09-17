# Task 2673 — independent descriptive evidence review

Verdict: the completed geometry measurement supports the sarcasm ladder descriptively in the primary ten-persona bank, while French progression is layer-dependent and strongly dependent on which persona bank supplies the centering reference. This measurement does not assess behavioral leakage.

Artifact fingerprint: `65ca9c9892b4012747f8f1e1793d6ed3eb3c4acec1d801bf58ceba993a61c27a`. Source: `f0c06bbd4515bc0f9cb200be42f2ddcf874fc9dc`. The supplied independent verifier reports all 2,400 contexts, 300 chunks and all 64 layer vectors (5,120 dimensions each) per row complete. I independently recomputed primary/six-bank matrices and half stability from the saved FP64 centroids; maximum discrepancy from the saved summary matrices and stability is 3.55e-15. I did not rerun the model or independently reread every raw chunk in this interpretation review.

## Four informative increments

Each increment differences two similarities to SFL (the full three-feature prompt). SFL→SFL self-similarity is excluded. Counts are descriptive across correlated, zero-based blocks; no significance threshold was introduced.

| Increment | Ten-bank positive / 64 | Ten-bank negative blocks | Six-bank positive / 64 | Even half | Odd half |
|---|---:|---|---:|---:|---:|
| default → sarcasm | 64 | None | 64 | 64 | 64 |
| sarcasm → sarcasm_lists | 64 | None | 64 | 64 | 64 |
| default → french | 64 | None | 43 | 64 | 64 |
| french → french_lists | 55 | 3, 4, 18, 20, 21, 22, 23, 25, 26 | 58 | 56 | 55 |

The primary sarcasm ladder has both informative increments positive at 64/64 blocks. The primary French ladder has both positive at 55/64 blocks. The nine French violations concern French→French+lists. A positive observed sign does not establish a numerically resolved effect: the smallest sarcasm→sarcasm+lists increment is 0.002254 at block 20.

## Fixed-block similarity to SFL

All entries below use the primary ten-persona centering bank. SFL self-cosines are shown only for reference.

| Persona | Block 15 | Block 31 | Block 47 | Block 63 |
|---|---:|---:|---:|---:|
| default | -0.623279 | -0.561558 | -0.446974 | -0.641067 |
| sarcasm | 0.179434 | -0.163567 | -0.009427 | 0.412296 |
| sarcasm_lists | 0.380623 | 0.731636 | 0.844765 | 0.889891 |
| sfl | 1.000000 | 1.000000 | 1.000000 | 1.000000 |
| french | 0.390553 | 0.085314 | -0.249380 | -0.613426 |
| french_lists | 0.463805 | 0.486334 | 0.324352 | -0.132615 |
| persona_dismissive | 0.033176 | -0.100554 | -0.168597 | -0.177330 |
| persona_sarcastic | -0.185556 | -0.183865 | -0.093309 | 0.311499 |
| persona_saboteur | -0.305204 | -0.485370 | -0.379550 | -0.568684 |
| persona_peer | -0.469504 | -0.475420 | -0.410606 | -0.411934 |

## Question halves

Corresponding centered half-centroid cosines range from 0.989790 to 0.999860, with median 0.998622 across 640 persona×block cells. The minimum is the default persona at block 61. Each half independently centers its ten-persona bank and contains 120 paired questions per persona.

All four increment signs agree between halves at every block except French→French+lists at block 23: full=-0.000394787; even_index_half=0.002222328; odd_index_half=-0.003261747. The even half has French progression at 56/64 blocks; the odd half at 55/64. Both halves preserve sarcasm progression at 64/64. This is internal split-half agreement, not a confidence interval or evidence of naturalistic generalization.

## Six-persona centering sensitivity

Recenter the six ladder conditions on their own mean and treat this as a separate metric bank. Sarcasm progression remains 64/64. Default→French is positive at 43/64 blocks and French→French+lists at 58/64; their conjunction gives French progression at 37/64 blocks. Default→French changes sign at blocks 39, 40, 41, 43, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63. French→French+lists changes sign at blocks 21, 22 and 23. The centering reference materially changes the French ordering; do not describe the observed order as intrinsic or bank-invariant.

## Numerical and interpretation limits

Production used uniform unpadded singleton forwards with strict math controls. Two extreme contexts repeated bitwise identically; sampled hook/tuple checks had zero recorded error. The independent verifier also found zero capture-versus-smoke error on the 22 overlapping rows. This verifies observed repeatability, not FP32-reference accuracy. Mixed batching remained a failed diagnostic and was not used in production.

The separate two-question-per-persona backend check changes SDPA eligibility and BF16 GEMM reduction permission together. Its largest cosine change was 0.129476 at block 10; its largest informative-increment change was 0.141370. French increment signs changed at blocks 4, 19, 22, 23 and 55 on that slice. At fixed blocks 15/31/47/63, maximum cosine changes were 0.007877/0.005961/0.014352/0.009110. These are observed discrepancies on question IDs 0 and 1, never bounds or uncertainty bars for the 240-question bank. Qualify small full-bank increments rather than certify their numerical signs using the smoke result.

The inherited constructed question battery and exact explicit prompts define this pilot. Prompt length, wording and explicit prohibitions remain confounds for attributing geometry to a single persona feature. No answers were generated, so there is no scalar behavioral leakage value, leakage correlation, or reproduction of the paper’s behavioral experiment. No publication or teardown receipt is asserted by this read-only review.

## All-layer primary increments

| Block | Default→Sarcasm | Sarcasm→Sarcasm+lists | Default→French | French→French+lists |
|---:|---:|---:|---:|---:|
| 0 | 1.056239904 | 0.498266940 | 0.609098267 | 0.578022569 |
| 1 | 1.146411210 | 0.304595860 | 1.131332534 | 0.133359789 |
| 2 | 1.156091452 | 0.196937144 | 1.016799359 | 0.094750770 |
| 3 | 0.968312904 | 0.112166444 | 1.319309407 | -0.029948698 |
| 4 | 1.104331418 | 0.071328651 | 1.374389407 | -0.047356650 |
| 5 | 1.045966803 | 0.094538916 | 1.231026660 | 0.020882422 |
| 6 | 1.133235574 | 0.095829125 | 1.223393068 | 0.033621835 |
| 7 | 0.841646218 | 0.132581845 | 1.207066380 | 0.100150547 |
| 8 | 0.880004961 | 0.160175527 | 1.206605417 | 0.130913714 |
| 9 | 0.876959925 | 0.147897374 | 1.151932315 | 0.134794062 |
| 10 | 0.837135691 | 0.155179065 | 1.203965369 | 0.093484344 |
| 11 | 0.636520507 | 0.044081111 | 1.206229159 | 0.093341353 |
| 12 | 0.710775407 | 0.028186874 | 1.195044221 | 0.101691653 |
| 13 | 0.771601909 | 0.089442974 | 1.186332106 | 0.099830297 |
| 14 | 0.748004067 | 0.025369389 | 1.204694510 | 0.063233705 |
| 15 | 0.802712577 | 0.201189222 | 1.013831625 | 0.073252315 |
| 16 | 0.858023120 | 0.189043164 | 1.038171191 | 0.052214788 |
| 17 | 1.029495291 | 0.141285800 | 1.103842684 | 0.023975391 |
| 18 | 1.113768744 | 0.114916188 | 1.140509265 | -0.001994023 |
| 19 | 1.222454540 | 0.141964112 | 1.086663559 | 0.017540154 |
| 20 | 1.327556324 | 0.002253994 | 1.205940202 | -0.018425407 |
| 21 | 1.171039361 | 0.158923064 | 1.145748784 | -0.006177712 |
| 22 | 1.214989042 | 0.131456859 | 1.182586963 | -0.016799046 |
| 23 | 1.178110008 | 0.218117123 | 0.882241948 | -0.000394787 |
| 24 | 1.182289209 | 0.226336001 | 0.853230894 | 0.004393078 |
| 25 | 1.201705889 | 0.200448154 | 0.861188248 | -0.016154503 |
| 26 | 1.217074977 | 0.169635126 | 0.905914525 | -0.022164484 |
| 27 | 0.913704742 | 0.476237342 | 0.852843366 | 0.161838696 |
| 28 | 0.745866674 | 0.486488001 | 0.857398042 | 0.168857494 |
| 29 | 0.714319807 | 0.481862140 | 0.957580025 | 0.141718092 |
| 30 | 0.665251414 | 0.479433499 | 1.044879867 | 0.140577018 |
| 31 | 0.397991479 | 0.895202428 | 0.646872297 | 0.401020054 |
| 32 | 0.428585045 | 0.857807365 | 0.574283234 | 0.422318778 |
| 33 | 0.454391728 | 0.797057036 | 0.572738342 | 0.395174993 |
| 34 | 0.522597343 | 0.752258400 | 0.656045778 | 0.398411750 |
| 35 | 0.581214931 | 0.746376013 | 0.406310200 | 0.440514929 |
| 36 | 0.639464742 | 0.691238116 | 0.359094334 | 0.394368313 |
| 37 | 0.668210360 | 0.657536568 | 0.358312751 | 0.361856609 |
| 38 | 0.672303718 | 0.664913508 | 0.389049820 | 0.356248742 |
| 39 | 0.543456894 | 0.807713007 | 0.241931734 | 0.533197113 |
| 40 | 0.538795979 | 0.810586733 | 0.271477397 | 0.516469112 |
| 41 | 0.534163930 | 0.806956306 | 0.275763239 | 0.515964834 |
| 42 | 0.517363974 | 0.823656620 | 0.322821674 | 0.498575047 |
| 43 | 0.458159824 | 0.869945648 | 0.255725586 | 0.593943010 |
| 44 | 0.430464267 | 0.877795850 | 0.258213534 | 0.578953592 |
| 45 | 0.406899135 | 0.896377589 | 0.264821428 | 0.580385265 |
| 46 | 0.393700198 | 0.892311464 | 0.234133021 | 0.605967692 |
| 47 | 0.437547537 | 0.854191579 | 0.197594300 | 0.573731553 |
| 48 | 0.445343646 | 0.851882618 | 0.158554413 | 0.624383062 |
| 49 | 0.551284815 | 0.748326080 | 0.114346758 | 0.579379164 |
| 50 | 0.757840764 | 0.619057812 | 0.068730471 | 0.566105116 |
| 51 | 0.820352718 | 0.577230572 | 0.050051262 | 0.555285072 |
| 52 | 0.917398473 | 0.512569236 | 0.054099874 | 0.503787003 |
| 53 | 0.965998419 | 0.480814323 | 0.038229202 | 0.463152401 |
| 54 | 0.982708016 | 0.460878400 | 0.023347561 | 0.453971686 |
| 55 | 0.998300860 | 0.455195576 | 0.022767764 | 0.438007242 |
| 56 | 0.989438561 | 0.466679005 | 0.026957255 | 0.424611037 |
| 57 | 1.007791157 | 0.459398342 | 0.023715203 | 0.408151354 |
| 58 | 1.031073950 | 0.456829620 | 0.034423226 | 0.378234082 |
| 59 | 1.014041277 | 0.479689679 | 0.040947094 | 0.390866500 |
| 60 | 1.032964029 | 0.480530872 | 0.048217309 | 0.386847564 |
| 61 | 1.005826634 | 0.497372767 | 0.054676093 | 0.406229004 |
| 62 | 1.002558542 | 0.509407783 | 0.057212375 | 0.417726739 |
| 63 | 1.053363824 | 0.477594171 | 0.027641537 | 0.480811375 |

Exact values, input SHA256s, separate half/six-bank values and complete sign-change lists: `/tmp/issue2673-final-derived.json`.
