# Manipulation-check table — issue 2587

Per-axis fire floors over BASE values (>=70% comply per value; floor = ceil(0.6 x width); undetermined counts as not fired).

| axis | Qwen3.5-9B (thinking off) | Qwen2.5-7B-Instruct |
|---|---|---|
| Answer language | 3/3 fired (floor 2: met) | not judged |
| Content constraint | 5/5 fired (floor 3: met) | 5/5 fired (floor 3: met) |
| Format | 3/5 fired (floor 3: met) | 3/5 fired (floor 3: met) |
| Hedging | 2/2 fired (floor 2: met) | 1/2 fired (floor 2: MISSED) |
| Lexical marker | 5/5 fired (floor 3: met) | 5/5 fired (floor 3: met) |
| Persona | 4/5 fired (floor 3: met) | 2/5 fired (floor 3: MISSED) |
| Query content oneword | no manipulation check (query class) | not judged |
| Register | 2/2 fired (floor 2: met) | 2/2 fired (floor 2: met) |
| Stance | 5/5 fired (floor 3: met) | 1/5 fired (floor 3: MISSED) |
| User fact | 5/5 fired (floor 3: met) | 5/5 fired (floor 3: met) |
| User profile | 5/5 fired (floor 3: met) | 3/5 fired (floor 3: met) |

Qwen3.5-9B (thinking off): 10/10 judged axes meet the fire floor. Qwen2.5-7B-Instruct: 6/9 judged axes meet the fire floor.
