# Two-sided SAE dashboards v2 (L19 map): eigen 2-planes + whitened cosines

**Plane cosine** (used for every collapsed conjugate eigen pair): for a unit decoder
direction d and an orthonormal basis P of the eigenvector's real invariant 2-plane
(spanned by the real and imaginary parts of the complex eigenvector), the plane cosine
is ||P^T d|| — the cosine between d and its projection onto the plane. For a 1-D
direction (real eigenvalue, or any singular direction) it reduces to |cos|.

v1 (`sae_dashboards_L19.json`) read complex eigenvectors through the normalized real
part only; the `raw max cos (v1)` column shows that older value in parentheses for
eigen rows. `whitened max cos` uses the side-matched covariance from the P-B moments
(n_pool 963,444 rows), shrinkage lambda 1e-2.
`im share` is the fraction of the squared plane cosine at the top feature carried by
the imaginary axis (what v1 dropped). Directions are flagged against kind-matched
empirical p95 null floors (random unit directions for 1-D, random 2-planes for planes).

Read-side directions are dashboarded against TWO dictionaries: the andyrdt per-token
L19 SAE (131,072 features, judged descriptions where available) and the #2569 leg-4
context SAE trained on the very X19 last-prompt-token context rows the map reads
(65,536 features, k=100; NO descriptions exist for these features, so those sections
report feature ids only, plus an encoder-pass companion).

Reference checks: rho 1.205352, kappa(V)
4260.9, 1751
complex pairs / 82 real eigenvalues,
biorthogonality max error 2.5e-11 — all
match the banked factor artifact.

## Singular read (left singular vectors u_i vs andyrdt context SAE)

Null floors (empirical p95): raw line 0.085, raw plane 0.090, whitened (1e-2) line 0.103, whitened (1e-2) plane 0.111.

| rank | σ | raw max cos | whitened max cos (1e-2) | im share | top-3 features |
|---|---|---|---|---|---|
| 1 | 7.962 | 0.085 | 0.104* | — | **46306** (+0.08): (no description)<br>**3984** (+0.08): (no description)<br>**74876** (+0.08): (no description) |
| 2 | 6.575 | 0.078 | 0.095 | — | **75338** (+0.08): (no description)<br>**28152** (+0.08): Tokens that function as domain or category modifiers, specifying the type, field, or attribute of a <br>**98050** (+0.07): (no description) |
| 3 | 4.931 | 0.094* | 0.093 | — | **104858** (+0.09): Tokens that appear in AI assistant responses when introducing, continuing, or structuring helpful in<br>**56851** (+0.09): (no description)<br>**94965** (+0.09): (no description) |
| 4 | 4.772 | 0.128* | 0.089 | — | **118359** (+0.13): (no description)<br>**126301** (+0.10): (no description)<br>**121718** (+0.09): (no description) |
| 5 | 4.449 | 0.098* | 0.093 | — | **25116** (+0.10): (no description)<br>**74750** (+0.09): (no description)<br>**8086** (+0.09): (no description) |
| 6 | 4.288 | 0.135* | 0.107* | — | **122601** (+0.14): Functional words, technical terminology, and structural elements that serve connective, organization<br>**8458** (+0.12): Common functional words, structural markers, and connectives (including articles, demonstratives, po<br>**58257** (+0.11): (no description) |
| 7 | 4.133 | 0.142* | 0.108* | — | **58888** (+0.14): (no description)<br>**45373** (+0.13): (no description)<br>**105906** (+0.12): (no description) |
| 8 | 3.976 | 0.137* | 0.099 | — | **84574** (+0.14): (no description)<br>**71644** (+0.14): Tokens appearing in natural language narrative, conversational, or expository text across multiple l<br>**122994** (+0.13): (no description) |
| 9 | 3.823 | 0.148* | 0.099 | — | **34923** (+0.15): (no description)<br>**1250** (+0.15): (no description)<br>**124919** (+0.13): (no description) |
| 10 | 3.813 | 0.150* | 0.097 | — | **70261** (+0.15): (no description)<br>**73667** (+0.14): (no description)<br>**23650** (+0.13): (no description) |
| 11 | 3.623 | 0.126* | 0.091 | — | **14354** (+0.13): (no description)<br>**40808** (+0.11): (no description)<br>**10374** (+0.10): Tokens that serve structural, transitional, or organizational functions in text, including formattin |
| 12 | 3.401 | 0.137* | 0.099 | — | **96293** (+0.14): (no description)<br>**24951** (+0.14): (no description)<br>**126864** (+0.12): (no description) |
| 13 | 3.348 | 0.120* | 0.103* | — | **56851** (+0.12): (no description)<br>**20964** (+0.11): (no description)<br>**98568** (+0.11): Common grammatical function words and punctuation that connect clauses or modify relationships betwe |
| 14 | 3.310 | 0.128* | 0.094 | — | **7237** (+0.13): (no description)<br>**123737** (+0.13): Continuation and connective tokens (including punctuation, prepositions, articles, and formatting el<br>**74951** (+0.13): (no description) |
| 15 | 3.238 | 0.121* | 0.093 | — | **129894** (+0.12): (no description)<br>**35486** (+0.10): (no description)<br>**72566** (+0.10): (no description) |
| 16 | 3.177 | 0.123* | 0.089 | — | **54410** (+0.12): (no description)<br>**10374** (+0.11): Tokens that serve structural, transitional, or organizational functions in text, including formattin<br>**26077** (+0.11): Common structural or transitional tokens that serve as connectors, punctuation, or generic descripti |

`*` = above the kind-matched empirical p95 null floor.

## Singular write (right singular vectors v_i vs answer SAE)

Null floors (empirical p95): raw line 0.083, raw plane 0.088, whitened (1e-2) line 0.100, whitened (1e-2) plane 0.108.

| rank | σ | raw max cos | whitened max cos (1e-2) | im share | top-3 features |
|---|---|---|---|---|---|
| 1 | 7.962 | 0.552* | 0.097 | — | **818** (+0.55): This feature activates on text containing repetitive sequences of characters, symbols, or tokens—suc<br>**1167** (+0.40): This feature activates on long-form AI assistant responses that explore complex hypothetical, philos<br>**1229** (+0.38): This feature activates on very short, simple text inputs — including brief sentences, greetings, sin |
| 2 | 6.575 | 0.626* | 0.091 | — | **742** (+0.63): This feature captures assistant responses that acknowledge a task or instruction and prompt the user<br>**845** (+0.39): This feature captures AI assistant greeting and offer-to-help responses — opening turns where the as<br>**818** (+0.30): This feature activates on text containing repetitive sequences of characters, symbols, or tokens—suc |
| 3 | 4.931 | 0.508* | 0.080 | — | **445** (+0.51): This feature activates on structured professional and business documents that present detailed outli<br>**166** (+0.38): This feature activates on assistant turns that extract, identify, classify, or list structured infor<br>**417** (+0.27): This feature activates on AI-generated text in non-English European languages (Hungarian, Finnish, G |
| 4 | 4.772 | 0.415* | 0.086 | — | **1428** (+0.41): This feature activates on assistant responses that provide step-by-step instructional guides or how-<br>**1422** (+0.27): This feature activates on text containing repetitive or looping character sequences (e.g., the same <br>**1328** (+0.26): This feature activates on analytical or evaluative text about customer service, support, or transact |
| 5 | 4.449 | 0.321* | 0.078 | — | **520** (+0.32): Short, single-word (or very short phrase) affirmative, negative, or acknowledgment responses such as<br>**1370** (+0.29): This feature activates on very short, single-token or minimal responses that serve as labels, answer<br>**1150** (+0.28): This feature activates on AI assistant responses that generate numbered or bulleted lists of ideas,  |
| 6 | 4.288 | 0.369* | 0.074 | — | **1079** (+0.37): This feature activates on AI-generated speeches, toasts, letters, and motivational messages written <br>**2013** (+0.31): The feature activates on text expressing aspirational, mission-driven, or visionary content about pr<br>**658** (+0.28): This feature activates on AI assistant responses to technical computing and software development que |
| 7 | 4.133 | 0.347* | 0.080 | — | **845** (+0.35): This feature captures AI assistant greeting and offer-to-help responses — opening turns where the as<br>**549** (+0.30): This feature activates on vivid, detailed aesthetic descriptions of objects, people, or products — e<br>**1218** (+0.28): This feature activates on AI assistant responses that provide structured, how-to or tips-based guida |
| 8 | 3.976 | 0.362* | 0.067 | — | **1150** (+0.36): This feature activates on AI assistant responses that generate numbered or bulleted lists of ideas, <br>**1441** (+0.35): This feature activates on assertive declarative statements that present conclusions, clarifications,<br>**1066** (+0.29): This feature captures content that enumerates causes, reasons, or factors behind problems, failures, |
| 9 | 3.823 | 0.348* | 0.086 | — | **549** (+0.35): This feature activates on vivid, detailed aesthetic descriptions of objects, people, or products — e<br>**1533** (+0.33): This feature activates on structured, informative AI assistant responses that explain concepts, fram<br>**195** (+0.31): This feature activates on comma-separated or list-formatted enumerations of specific technical softw |
| 10 | 3.813 | 0.365* | 0.097 | — | **1150** (+0.36): This feature activates on AI assistant responses that generate numbered or bulleted lists of ideas, <br>**1441** (+0.34): This feature activates on assertive declarative statements that present conclusions, clarifications,<br>**306** (+0.29): This feature activates on long-form Chinese written compositions, including essays, reflective piece |
| 11 | 3.623 | 0.348* | 0.113* | — | **243** (+0.35): This feature activates on technical, academic, and formal Chinese-language text spanning diverse dom<br>**306** (+0.33): This feature activates on long-form Chinese written compositions, including essays, reflective piece<br>**225** (+0.25): This feature activates on turns that contain structured or enumerated information — such as bullet-p |
| 12 | 3.401 | 0.332* | 0.086 | — | **1000** (+0.33): This feature activates on responses that present quantitative data involving numerical ranges, stati<br>**1291** (+0.28): The feature activates on assistant responses that provide numbered or bulleted lists of practical ti<br>**1314** (+0.26): This feature activates on technical explanations of software/computer science concepts, particularly |
| 13 | 3.348 | 0.298* | 0.084 | — | **1291** (+0.30): The feature activates on assistant responses that provide numbered or bulleted lists of practical ti<br>**205** (+0.24): This feature activates on text describing heads of state, political leaders, or monarchs along with <br>**1420** (+0.24): This feature activates on assistant turns describing open-source machine learning frameworks, librar |
| 14 | 3.310 | 0.290* | 0.062 | — | **1167** (+0.29): This feature activates on long-form AI assistant responses that explore complex hypothetical, philos<br>**1917** (+0.27): This feature activates on assistant responses related to creative writing assistance, including stor<br>**306** (+0.27): This feature activates on long-form Chinese written compositions, including essays, reflective piece |
| 15 | 3.238 | 0.243* | 0.089 | — | **2012** (+0.24): This feature activates on AI assistant responses to deep philosophical, metaphysical, or existential<br>**1328** (+0.23): This feature activates on analytical or evaluative text about customer service, support, or transact<br>**666** (+0.22): This feature activates on emotionally intense, literary, or poetic text characterized by dark, melan |
| 16 | 3.177 | 0.287* | 0.100* | — | **186** (+0.29): This feature activates on Russian-language informational and instructional content, particularly AI-<br>**1422** (+0.28): This feature activates on text containing repetitive or looping character sequences (e.g., the same <br>**156** (+0.27): This feature activates on assistant responses that provide numbered or bulleted lists of resources,  |

`*` = above the kind-matched empirical p95 null floor.

## Eigen read (right eigenvectors vs andyrdt context SAE, conjugate pairs collapsed)

Null floors (empirical p95): raw line 0.085, raw plane 0.090, whitened (1e-2) line 0.103, whitened (1e-2) plane 0.111.

| rank | |λ| | raw max cos (v1 real-part) | whitened max cos (1e-2) | im share | top-3 features |
|---|---|---|---|---|---|
| 1 | 1.205 | 0.304 (0.245)* | 0.144* | 0.35 | **52076** (+0.30): (no description)<br>**42743** (+0.25): (no description)<br>**67358** (+0.24): (no description) |
| 2 | 1.149 | 0.301 (0.291)* | 0.154* | 0.07 | **131011** (+0.30): (no description)<br>**112113** (+0.29): (no description)<br>**82890** (+0.24): (no description) |
| 3 | 1.137 | 0.203 (0.187)* | 0.113* | 0.76 | **94965** (+0.20): (no description)<br>**73569** (+0.19): (no description)<br>**19838** (+0.18): (no description) |
| 4 | 1.096 | 0.213 (0.213)* | 0.108 | 0.00 | **30108** (+0.21): (no description)<br>**24951** (+0.19): (no description)<br>**123464** (+0.19): (no description) |
| 5 | 1.050 | 0.135 (real eigenvalue)* | 0.094 | — | **76458** (+0.13): (no description)<br>**33180** (+0.13): (no description)<br>**131011** (+0.13): (no description) |
| 6 | 1.033 | 0.179 (0.155)* | 0.132* | 0.37 | **72566** (+0.18): (no description)<br>**98710** (+0.17): (no description)<br>**73569** (+0.16): (no description) |
| 7 | 1.017 | 0.180 (real eigenvalue)* | 0.108* | — | **129950** (+0.18): (no description)<br>**50156** (+0.16): (no description)<br>**129894** (+0.15): (no description) |
| 8 | 0.998 | 0.204 (0.182)* | 0.111 | 0.51 | **7434** (+0.20): (no description)<br>**99773** (+0.19): (no description)<br>**47831** (+0.19): (no description) |
| 9 | 0.991 | 0.179 (0.157)* | 0.119* | 0.91 | **76458** (+0.18): (no description)<br>**70261** (+0.17): (no description)<br>**93598** (+0.16): (no description) |
| 10 | 0.961 | 0.167 (0.161)* | 0.110 | 0.98 | **117743** (+0.17): (no description)<br>**10033** (+0.16): (no description)<br>**93598** (+0.16): (no description) |
| 11 | 0.945 | 0.125 (real eigenvalue)* | 0.087 | — | **109743** (+0.12): (no description)<br>**42455** (+0.12): (no description)<br>**95158** (+0.12): (no description) |
| 12 | 0.931 | 0.192 (0.138)* | 0.133* | 0.88 | **19052** (+0.19): (no description)<br>**35412** (+0.17): (no description)<br>**7434** (+0.16): (no description) |
| 13 | 0.917 | 0.150 (0.148)* | 0.107 | 0.13 | **13214** (+0.15): (no description)<br>**90848** (+0.15): (no description)<br>**30664** (+0.15): (no description) |
| 14 | 0.902 | 0.165 (0.154)* | 0.122* | 0.57 | **38256** (+0.17): (no description)<br>**47212** (+0.16): (no description)<br>**98784** (+0.15): (no description) |
| 15 | 0.885 | 0.156 (0.134)* | 0.105 | 0.44 | **20964** (+0.16): (no description)<br>**45864** (+0.15): (no description)<br>**101839** (+0.14): (no description) |
| 16 | 0.882 | 0.150 (0.146)* | 0.111* | 0.06 | **20964** (+0.15): (no description)<br>**94965** (+0.14): (no description)<br>**83500** (+0.14): (no description) |

`*` = above the kind-matched empirical p95 null floor.

## Eigen write (left eigenvector rows vs answer SAE, conjugate pairs collapsed)

Null floors (empirical p95): raw line 0.083, raw plane 0.088, whitened (1e-2) line 0.100, whitened (1e-2) plane 0.108.

| rank | |λ| | raw max cos (v1 real-part) | whitened max cos (1e-2) | im share | top-3 features |
|---|---|---|---|---|---|
| 1 | 1.205 | 0.418 (0.412)* | 0.147* | 0.03 | **1890** (+0.42): This feature activates on assistant responses related to healthy eating, nutrition, and food — inclu<br>**1724** (+0.42): This feature activates on detailed technical descriptions of specific chemical compounds — particula<br>**1627** (+0.39): This feature activates on assistant-generated content describing the health benefits, mechanisms of  |
| 2 | 1.149 | 0.415 (0.341)* | 0.163* | 0.99 | **1724** (+0.42): This feature activates on detailed technical descriptions of specific chemical compounds — particula<br>**1468** (+0.35): This feature activates on AI assistant responses to questions about how to make money online or gene<br>**551** (+0.35): This feature activates on AI assistant responses explaining financial instruments, investment strate |
| 3 | 1.137 | 0.379 (0.240)* | 0.114* | 0.77 | **818** (+0.38): This feature activates on text containing repetitive sequences of characters, symbols, or tokens—suc<br>**2013** (+0.34): The feature activates on text expressing aspirational, mission-driven, or visionary content about pr<br>**1078** (+0.32): This feature activates on text that is fluent-seeming but semantically incoherent or nonsensical in  |
| 4 | 1.096 | 0.290 (0.256)* | 0.164* | 0.83 | **573** (+0.29): This feature activates on content related to sports, athletic training, and physical fitness — inclu<br>**161** (+0.28): This feature activates on fictional or hypothetical NBA/NFL season narratives, particularly detailed<br>**2002** (+0.27): This feature activates on detailed legal and judicial content, particularly AI-generated responses t |
| 5 | 1.050 | 0.337 (real eigenvalue)* | 0.111* | — | **2002** (+0.34): This feature activates on detailed legal and judicial content, particularly AI-generated responses t<br>**1506** (+0.24): This feature activates on detailed explanatory text about deep learning and NLP concepts, particular<br>**1428** (+0.24): This feature activates on assistant responses that provide step-by-step instructional guides or how- |
| 6 | 1.033 | 0.304 (0.227)* | 0.111* | 0.89 | **1428** (+0.30): This feature activates on assistant responses that provide step-by-step instructional guides or how-<br>**1198** (+0.29): This feature activates on informational descriptions of animal species, covering their physical char<br>**1263** (+0.28): This feature activates on assistant responses explaining financial, legal, and tax topics—such as mo |
| 7 | 1.017 | 0.255 (real eigenvalue)* | 0.098 | — | **1428** (+0.25): This feature activates on assistant responses that provide step-by-step instructional guides or how-<br>**1266** (+0.20): This feature activates on AI-generated responses that provide factual overviews of specific named en<br>**1109** (+0.20): This feature activates on responses describing legal, regulatory, and administrative requirements —  |
| 8 | 0.998 | 0.287 (0.269)* | 0.126* | 0.28 | **1007** (+0.29): This feature activates on technical assistant responses about database systems, server infrastructur<br>**687** (+0.28): This feature activates on Chinese-language assistant responses that provide practical advice, templa<br>**1066** (+0.27): This feature captures content that enumerates causes, reasons, or factors behind problems, failures, |
| 9 | 0.991 | 0.321 (0.293)* | 0.136* | 0.17 | **1007** (+0.32): This feature activates on technical assistant responses about database systems, server infrastructur<br>**1328** (+0.29): This feature activates on analytical or evaluative text about customer service, support, or transact<br>**882** (+0.29): This feature activates on Chinese-language content related to Chinese Communist Party (CCP) organiza |
| 10 | 0.961 | 0.298 (0.241)* | 0.116* | 0.63 | **597** (+0.30): This feature activates on richly descriptive, immersive narrative prose that vividly portrays sensor<br>**157** (+0.28): The feature activates on AI assistant responses about video game mechanics, characters, abilities, a<br>**54** (+0.27): This feature activates on content related to AI image generation prompts, digital art creation instr |
| 11 | 0.945 | 0.266 (real eigenvalue)* | 0.114* | — | **1953** (+0.27): This feature activates on detailed scientific/technical writing about specialized topics in chemistr<br>**274** (+0.24): This feature captures casual, friendly greeting and check-in language, particularly conversational o<br>**1700** (+0.24): This feature activates on Chinese-language assistant responses that provide structured, comprehensiv |
| 12 | 0.931 | 0.292 (0.243)* | 0.127* | 0.99 | **1533** (+0.29): This feature activates on structured, informative AI assistant responses that explain concepts, fram<br>**195** (+0.28): This feature activates on comma-separated or list-formatted enumerations of specific technical softw<br>**244** (+0.26): This feature activates on assistant-generated responses explaining electronic circuits, components,  |
| 13 | 0.917 | 0.299 (0.290)* | 0.100 | 0.06 | **408** (+0.30): This feature activates on content involving detailed physical descriptions of attractive women, incl<br>**687** (+0.29): This feature activates on Chinese-language assistant responses that provide practical advice, templa<br>**274** (+0.27): This feature captures casual, friendly greeting and check-in language, particularly conversational o |
| 14 | 0.902 | 0.355 (0.338)* | 0.113* | 0.09 | **274** (+0.35): This feature captures casual, friendly greeting and check-in language, particularly conversational o<br>**216** (+0.25): This feature activates on content involving manipulation, deception, authoritarian control, and desc<br>**1533** (+0.24): This feature activates on structured, informative AI assistant responses that explain concepts, fram |
| 15 | 0.885 | 0.287 (0.235)* | 0.110* | 0.33 | **1078** (+0.29): This feature activates on text that is fluent-seeming but semantically incoherent or nonsensical in <br>**417** (+0.25): This feature activates on AI-generated text in non-English European languages (Hungarian, Finnish, G<br>**1066** (+0.25): This feature captures content that enumerates causes, reasons, or factors behind problems, failures, |
| 16 | 0.882 | 0.283 (0.213)* | 0.196* | 0.44 | **408** (+0.28): This feature activates on content involving detailed physical descriptions of attractive women, incl<br>**43** (+0.24): GUI programming examples involving widget layout, window management, and UI component creation acros<br>**274** (+0.23): This feature captures casual, friendly greeting and check-in language, particularly conversational o |

`*` = above the kind-matched empirical p95 null floor.

## Singular read vs TRAINED context SAE (grain-matched; feature ids only)

Null floors (empirical p95): raw line 0.082, raw plane 0.089, whitened (1e-2) line 0.100, whitened (1e-2) plane 0.107.

| rank | σ | raw max cos | whitened max cos (1e-2) | im share | top-3 features | enc n_fired |
|---|---|---|---|---|---|---|
| 1 | 7.962 | 0.101* | 0.085 | — | **948** (+0.10)<br>**1261** (+0.09)<br>**1553** (+0.08) | 186 |
| 2 | 6.575 | 0.088* | 0.095 | — | **1788** (+0.09)<br>**1606** (+0.07)<br>**849** (+0.07) | 184 |
| 3 | 4.931 | 0.105* | 0.088 | — | **873** (+0.11)<br>**2165** (+0.08)<br>**18034** (+0.07) | 185 |
| 4 | 4.772 | 0.086* | 0.090 | — | **1047** (+0.09)<br>**1344** (+0.08)<br>**304** (+0.08) | 182 |
| 5 | 4.449 | 0.133* | 0.086 | — | **1553** (+0.13)<br>**948** (+0.12)<br>**141** (+0.07) | 183 |
| 6 | 4.288 | 0.104* | 0.097 | — | **873** (+0.10)<br>**980** (+0.08)<br>**704** (+0.07) | 183 |
| 7 | 4.133 | 0.110* | 0.086 | — | **353** (+0.11)<br>**304** (+0.11)<br>**1891** (+0.09) | 181 |
| 8 | 3.976 | 0.108* | 0.100* | — | **965** (+0.11)<br>**763** (+0.10)<br>**183** (+0.09) | 183 |
| 9 | 3.823 | 0.102* | 0.092 | — | **638** (+0.10)<br>**11171** (+0.09)<br>**299** (+0.09) | 183 |
| 10 | 3.813 | 0.089* | 0.107* | — | **960** (+0.09)<br>**873** (+0.08)<br>**869** (+0.08) | 181 |
| 11 | 3.623 | 0.098* | 0.091 | — | **306** (+0.10)<br>**1047** (+0.10)<br>**455** (+0.08) | 182 |
| 12 | 3.401 | 0.145* | 0.095 | — | **1354** (+0.15)<br>**821** (+0.12)<br>**688** (+0.09) | 182 |
| 13 | 3.348 | 0.091* | 0.094 | — | **698** (+0.09)<br>**1359** (+0.09)<br>**1890** (+0.09) | 181 |
| 14 | 3.310 | 0.107* | 0.084 | — | **3347** (+0.11)<br>**1359** (+0.09)<br>**1529** (+0.09) | 180 |
| 15 | 3.238 | 0.138* | 0.099 | — | **4574** (+0.14)<br>**821** (+0.11)<br>**1487** (+0.08) | 186 |
| 16 | 3.177 | 0.098* | 0.093 | — | **1047** (+0.10)<br>**844** (+0.09)<br>**455** (+0.09) | 182 |

`*` = above the kind-matched empirical p95 null floor.

## Eigen read vs TRAINED context SAE (grain-matched; feature ids only)

Null floors (empirical p95): raw line 0.082, raw plane 0.089, whitened (1e-2) line 0.100, whitened (1e-2) plane 0.107.

| rank | |λ| | raw max cos (real-part-only) | whitened max cos (1e-2) | im share | top-3 features | enc n_fired |
|---|---|---|---|---|---|---|
| 1 | 1.205 | 0.277 (0.203)* | 0.171* | 0.47 | **377** (+0.28)<br>**960** (+0.26)<br>**9819** (+0.22) | 184/183 |
| 2 | 1.149 | 0.284 (0.266)* | 0.176* | 0.12 | **1354** (+0.28)<br>**377** (+0.25)<br>**950** (+0.24) | 183/184 |
| 3 | 1.137 | 0.185 (0.185)* | 0.132* | 0.00 | **960** (+0.19)<br>**821** (+0.18)<br>**638** (+0.15) | 184/183 |
| 4 | 1.096 | 0.230 (0.228)* | 0.169* | 0.02 | **638** (+0.23)<br>**1354** (+0.18)<br>**9819** (+0.16) | 181/185 |
| 5 | 1.050 | 0.150 (real eigenvalue)* | 0.111* | — | **950** (+0.15)<br>**52908** (+0.12)<br>**7872** (+0.11) | 183 |
| 6 | 1.033 | 0.227 (0.157)* | 0.170* | 0.52 | **821** (+0.23)<br>**638** (+0.14)<br>**1717** (+0.12) | 182/182 |
| 7 | 1.017 | 0.151 (real eigenvalue)* | 0.091 | — | **4574** (+0.15)<br>**960** (+0.13)<br>**993** (+0.13) | 185 |
| 8 | 0.998 | 0.197 (0.122)* | 0.131* | 0.92 | **638** (+0.20)<br>**2680** (+0.15)<br>**965** (+0.14) | 183/184 |
| 9 | 0.991 | 0.154 (0.141)* | 0.120* | 0.16 | **2680** (+0.15)<br>**8973** (+0.14)<br>**638** (+0.13) | 185/184 |
| 10 | 0.961 | 0.147 (0.098)* | 0.113* | 0.58 | **8973** (+0.15)<br>**5717** (+0.13)<br>**1352** (+0.12) | 184/183 |
| 11 | 0.945 | 0.151 (real eigenvalue)* | 0.101* | — | **15337** (+0.15)<br>**638** (+0.11)<br>**738** (+0.10) | 182 |
| 12 | 0.931 | 0.159 (0.114)* | 0.111* | 0.93 | **5927** (+0.16)<br>**15337** (+0.14)<br>**137** (+0.14) | 185/183 |
| 13 | 0.917 | 0.131 (0.107)* | 0.104 | 0.99 | **638** (+0.13)<br>**763** (+0.13)<br>**16327** (+0.12) | 183/182 |
| 14 | 0.902 | 0.162 (0.161)* | 0.135* | 0.01 | **738** (+0.16)<br>**5927** (+0.15)<br>**15337** (+0.14) | 183/186 |
| 15 | 0.885 | 0.157 (0.146)* | 0.118* | 0.13 | **638** (+0.16)<br>**738** (+0.13)<br>**31160** (+0.12) | 182/185 |
| 16 | 0.882 | 0.147 (0.130)* | 0.114* | 0.84 | **2088** (+0.15)<br>**11171** (+0.13)<br>**738** (+0.13) | 184/184 |

`*` = above the kind-matched empirical p95 null floor.

## Summary

| section | raw median / max | raw > p95 | whitened median / max (1e-2) | whitened > p95 | top feat changed (1e-2) |
|---|---|---|---|---|---|
| singular_read | 0.135 / 0.162 | 30/32 | 0.093 / 0.111 | 6/32 | 32/32 |
| singular_write | 0.309 / 0.626 | 32/32 | 0.088 / 0.164 | 8/32 | 25/32 |
| eigen_read | 0.150 / 0.304 | 32/32 | 0.109 / 0.154 | 15/32 | 26/32 |
| eigen_write | 0.289 / 0.418 | 32/32 | 0.115 / 0.196 | 26/32 | 30/32 |
| singular_read_ctxsae | 0.109 / 0.200 | 32/32 | 0.092 / 0.113 | 6/32 | 29/32 |
| eigen_read_ctxsae | 0.154 / 0.284 | 32/32 | 0.118 / 0.176 | 25/32 | 21/32 |

### Read-side dictionary comparison (andyrdt per-token vs context SAE trained on v_C)

The trained context SAE is grain-matched: it was fit on the very X19 last-prompt-token
context states the map reads, whereas andyrdt is per-token over generic text. Its
features have no descriptions (ids only). The alive union covers all 65,536 features, so the alive-union-restricted column equals the full-dictionary column and is omitted.

| directions | andyrdt raw med/max | >p95 | trained raw med/max | >p95 | andyrdt whitened med/max | trained whitened med/max |
|---|---|---|---|---|---|---|
| singular read | 0.135 / 0.162 | 30/32 | 0.109 / 0.200 | 32/32 | 0.093 / 0.111 | 0.092 / 0.113 |
| eigen read | 0.150 / 0.304 | 32/32 | 0.154 / 0.284 | 32/32 | 0.109 / 0.154 | 0.118 / 0.176 |

**eigen_read planes** (28/32): median eigenvector imaginary mass 0.66; median imaginary-axis share of the max plane cosine 0.38; plane read raised the max cosine over the v1 real-part read by median +0.016 (max +0.065).

**eigen_write planes** (28/32): median eigenvector imaginary mass 0.74; median imaginary-axis share of the max plane cosine 0.64; plane read raised the max cosine over the v1 real-part read by median +0.037 (max +0.139).

**eigen_read_ctxsae planes** (28/32): median eigenvector imaginary mass 0.66; median imaginary-axis share of the max plane cosine 0.43; plane read raised the max cosine over the v1 real-part read by median +0.017 (max +0.076).

Whitening: gram_xx.pt / gram_yy.pt hold UNCENTERED fp64 sums-of-outer Grams (X^T X / Y^T Y) plus per-dim means over the raw-residual X19/Y19 map-training pool rows (issue2569_rowbattery._accumulate_moments reads the fp16 memmaps directly; no centering, no standardization), i.e. the same raw coordinates the row operator A acts on. Conversion: Sigma = (gram - n * outer(mean, mean)) / (n - 1) (centered covariance, unbiased). Shrinkage: Sigma + lam * (tr(Sigma)/d) * I.

Generated by `scripts/issue2569_eigen_dashboards_v2.py` at 2026-09-03T05:35:19.780834+00:00 (commit `41ae65058efe`).
