# Think-leak + cap-hit table — issue 2587

Cap-hit re-gen trigger 2% per cell/split; think-leak assert < 1% per cell (plan §7).

| unit | kind | rows | cap-hit | cap-hit after re-gen | think-leak | flags |
|---|---|---|---|---|---|---|
| Answer language | battery generation cell | 480 | 0.0000 | n/a | 0.0000 | ok |
| Content constraint | battery generation cell | 1200 | 0.0000 | n/a | 0.0000 | ok |
| Format | battery generation cell | 1200 | 0.0000 | n/a | 0.0000 | ok |
| Hedging | battery generation cell | 480 | 0.0000 | n/a | 0.0000 | ok |
| Lexical marker | battery generation cell | 1200 | 0.0000 | n/a | 0.0000 | ok |
| Persona | battery generation cell | 1200 | 0.0000 | n/a | 0.0000 | ok |
| Query | battery generation cell | 480 | 0.0000 | n/a | 0.0000 | ok |
| Query content oneword | battery generation cell | 480 | 0.0000 | n/a | 0.0000 | ok |
| Register | battery generation cell | 480 | 0.0000 | n/a | 0.0000 | ok |
| Stance | battery generation cell | 1200 | 0.0008 | n/a | 0.0000 | ok |
| User fact | battery generation cell | 1200 | 0.0000 | n/a | 0.0000 | ok |
| User profile | battery generation cell | 1200 | 0.0008 | n/a | 0.0008 | ok |
| ceiling_draw_43 | map-fit generation split | 1000 | 0.2080 | n/a | n/a | cap-hit over re-gen trigger |
| ceiling_draw_44 | map-fit generation split | 1000 | 0.2130 | n/a | n/a | cap-hit over re-gen trigger |
| test_1000 | map-fit generation split | 1000 | 0.2020 | n/a | n/a | cap-hit over re-gen trigger |
| train_25k | map-fit generation split | 25000 | 0.2632 | n/a | n/a | cap-hit over re-gen trigger |
| val_400 | map-fit generation split | 400 | 0.2000 | n/a | n/a | cap-hit over re-gen trigger |
| wc_test_1k | map-fit generation split | 998 | 0.3938 | n/a | n/a | cap-hit over re-gen trigger |
