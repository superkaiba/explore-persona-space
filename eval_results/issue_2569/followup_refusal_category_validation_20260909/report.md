# Existing refusal directions: category validation

Frozen L19 map and saved 124 pairs; no model calls. Manifest a-minus-b orientation; no selection on evaluated refusal flips.

| Category | n | Observed flips | Median kernel share | Median normalized gain | LOFO predicted/refusal rho |
|---|---:|---:|---:|---:|---:|
| obj_benign | 8 | 0 | 0.762 | 0.493 | None |
| obj_flip | 16 | 10 | 0.819 | 0.566 | 0.725 |
| subj_benign | 8 | 0 | 0.797 | 0.445 | None |
| subj_ctl | 16 | 6 | 0.792 | 0.476 | 0.915 |
| verb_benign | 8 | 0 | 0.796 | 0.505 | None |
| verb_flip | 16 | 14 | 0.833 | 0.483 | 0.757 |
| verb_harm | 16 | 1 | 0.731 | 0.495 | 0.853 |
| xstest | 36 | 30 | 0.792 | 0.535 | 0.48 |

## Domain-matched exploratory contrasts

Valence = object + verb harmful/benign swaps. Equal weight per shared semantic family.
- subj_ctl:kernel_share: difference 0.0425, family-bootstrap CI [0.029315069508899012, 0.057731116347514724], exact sign-flip p=0.0156, Holm p=0.0625.
- subj_ctl:normalized_gain: difference 0.0278, family-bootstrap CI [-0.022284337295632294, 0.07073216339687358], exact sign-flip p=0.3438, Holm p=0.6875.
- verb_harm:kernel_share: difference 0.0783, family-bootstrap CI [0.05334923378543433, 0.10899269796832634], exact sign-flip p=0.0156, Holm p=0.0625.
- verb_harm:normalized_gain: difference -0.0026, family-bootstrap CI [-0.053485282617257254, 0.04563991257286203], exact sign-flip p=0.9688, Holm p=0.9688.

## Limitations

- Exploratory secondary analysis on one saved bank, not prospective validation.
- Categories are not randomized; domain-matched contrasts do not isolate semantic harmfulness causally.
- XSTest is held out together for axis construction; within-XSTest intervals resample items, not unknown semantic families.
- OOF correlation intervals condition on fitted disjoint axes; axes are not re-estimated in bootstrap.
- Binary rates and means use ten archived draws per endpoint; no new judging or graded calibration.
- Kernel is low-gain, not zero-gain: a small mapped component can correlate with refusal.
- No framing manipulation: this cannot validate harmful-request versus jailbreak-framing semantics.
- Identity plus learned bias reduces to identity for pair differences. No new map/readout is fit.

See summary.json for all class-specific uncertainty, including undefined constant strata; crossfit_folds.json records each held-out/training index. input_provenance.json pins every source and enumerates current-versus-frozen label differences.
