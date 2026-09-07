# Natural100k data and endpoint-pilot acceptance

Task [1739, plan31](https://eps.superkaiba.com/tasks/1739/plan). These are completed
data-phase and fit-pilot receipts, **not evidence that the full scaling sweep has
finished**. The full sweep is a separately monitored phase.

The scientific GPU checkout is frozen at
`9f6a6fedb9a97365bfbfb0c241d7712d94ce43c0`. Later audit/plotting commits run outside
that checkout and do not alter fitting or its resume identity.

## Verified data

All 100,000 generic pairs contain an intact natural first-user prompt and its own
fresh Qwen answer, without recombination. The final store contains 100,000 distinct
context IDs and source pairs, with eight finite fp16 matrices of shape
100,000 × 3,584. Every generation/capture unit, row alignment, file hash, and final
matrix passed independent verification.

Immutable revisions in `superkaiba1/explore-persona-space-data`, under
`issue1739_natural100k_20260906`:

| Tree | Revision | Verified files |
|---|---|---:|
| `generated` | `ff5398d333776c027cca5f4f2d53cd5f9af57548` | 600 |
| `captured` | `55e49ff70f640e674a9cc6aeb94f0e7b9f5edf80` | 2,004 |
| `store` | `972fd732b40a32619be508e97d1adf695afb2436` | 11 |

Remote verification compared exact name sets, sizes, and every content ID
(LFS SHA-256 or Git blob SHA-1), not merely existence. Final manifest SHA-256:
`abeeb30b42665271e85e1e1f9448dc10ce215d45bb3ff4161dfd518140b6a524`.

## Verified production-shaped fitting pilot

Each behavior used 100,000 generic pairs and seed0 through all its P-B held-out
readout folds. The map was refitted on the complete declared union.

| Behavior | Fixed trait pairs | Full map-fit rows | Measured cell wall, seconds |
|---|---:|---:|---:|
| Evil | 6,468 | 106,468 | 321.875 |
| Sycophancy | 16,000 | 116,000 | 353.639 |
| Hallucination | 16,000 | 116,000 | 334.178 |

All three processes exited with code 0. The fresh success sentinel was verified
after all uploads and the driver exited. The independent audit checked all13
own-heldout readout folds, all three methods, frozen global layer IDs, paired
context/group/DV keys, and every reported correlation against saved predictions.
It also checked map R², identity-plus-bias R², and nearest-neighbor pool/chance
metadata. All 22 pilot output files were independently verified remotely; revisions
are recorded in `fit_pilot_remote_verified_20260907.json`.

The 99 resource samples cover 21:36:56–21:45:13 UTC. Maximum observed process HWM
was 38.195 GiB and maximum sampled GPU memory on an assigned device was 66,219 MiB
of 81,559 MiB. Sampling is not a guarantee of the instantaneous GPU peak. The host
cgroup limit was 1,005,999,996,928 bytes; sampled usage peaked at 158,792,921,088 bytes.

`fit_pilot_accepted.json` records the measured four-worker projection and owner
acceptance. Multiplying each behavior's actual 100k wall by its 49 remaining cells
and dividing by four gives 3.436 wall-hours. This conservatively charges smaller
rungs the endpoint cost. The accepted planning allowance is approximately 7.4 hours,
including 2× uncertainty and 0.5 hours of upload/imbalance overhead; it is not a
promised completion time.

## Interpretation boundaries

P-B holds each evaluation dataset out of its readout, not out of the fixed
trait mapping pool. Whitening is refitted at every generic-data size, so direct
and oracle baselines may change with size. Five seeds share evaluation contexts;
at 100k they also use the same complete generic pool. Seed ranges are descriptive,
not confidence intervals or an equivalence test. The earlier P-A scaling study
is not a matched comparator to this P-B run.

The data pilot's generation/capture evidence is in `pilot_accepted.json`; full
capture measurements and integrity receipts supersede it where applicable.
The rejected source-cache discrepancy, CPU-stage preservation, and narrow reviewed
upload-scanner corrections are documented in the sibling `cpu_audit_20260907`
and `upload_gate_fixtures.json` records. No scientific answer was redacted or
reselected by the upload correction.

The complete pilot audit remains byte-for-byte unchanged in the artifact store.
Its SHA-256 is `b529b2a0cf1a1c123da80a019b59709857f12580e27937411d8a4055fcdb1d1f`.
The Git copy uses an upload reference because the Git secret hook misclassified
the inherited `credential_scanner_sha256` metadata as an API key. That value was
independently matched to the SHA-256 of the actual preparation scanner at commit
`e058fdc2634ddcde303caf5730a730580795e522`. No hook bypass or artifact redaction
was used. The verified upload receipt records the complete acceptance bundle at
revision `f31582aac7edc3021ae3ee7de540f71f144996a7`:
[unchanged full pilot audit](https://huggingface.co/datasets/superkaiba1/explore-persona-space-data/resolve/f31582aac7edc3021ae3ee7de540f71f144996a7/issue1739_natural100k_20260906/acceptance_audit_20260907/fit_pilot_independent_audit_20260907.json).
