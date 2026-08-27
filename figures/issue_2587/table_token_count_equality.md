# q25-vs-q35 token-count-equality table — issue 2587

Value-string token counts under the Qwen3.5 tokenizer (q35, this run) vs the
parent's pinned Qwen2.5 expectation (q25). Within-axis equality held by
construction under q25; under q35 it is RECORDED, never assumed.

| axis | q35 counts (distinct) | q35 within-axis equal | q25 expected | q35 paraphrase counts |
|---|---|---|---|---|
| Answer language | 4 | yes | n/a | n/a |
| Content constraint | 9 | yes | 9 | 7, 8, 9 |
| Format | 10 | yes | 10 | 13 |
| Hedging | 10 | yes | 10 | 9, 10 |
| Lexical marker | 12 | yes | 12 | 11 |
| Persona | 13 | yes | 13 | 15 |
| Register | 10 | yes | 10 | 10, 11 |
| Stance | 9 | yes | 9 | 8, 9 |
| User fact | 13 | yes | 13 | 13 |
| User profile | 24 | yes | 24 | 23 |

Name tokens (q35): 5/5 names remain single-token (the q25 single-token property is recorded per name, never assumed).
