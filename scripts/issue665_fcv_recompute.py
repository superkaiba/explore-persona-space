"""Recompute and persist the A3.6c input-localization fraction f_CV.

f_CV is defined as the input-localization fraction: the real trained-context-vector
patch effect (variant p_up: trained CV inserted into the BASE model) ABOVE the
base-CV self-null (variant self_c0: base CV into the base model), averaged over
(context, layer, scope). f_CV near 0 => the trained vector inserted into the base
model moves activations no more than the base-CV null => no input localization.

    f_CV = mean_over_(context,layer,scope) [ f_cv_v(p_up) - f_cv_v(self_c0) ]

We persist this scalar into each per-cell a36c JSON (overwriting the null), keyed
'f_CV', and also record the per-variant means + the variant-pair definition under
'f_CV_detail' for full auditability.
"""
import json, glob
from collections import defaultdict

cells = sorted(glob.glob("eval_results/issue_665/a36c/*.json"))
summary = []
for path in cells:
    d = json.load(open(path))
    rows = d["rows"]
    by_var = defaultdict(list)
    for r in rows:
        by_var[r["variant"]].append(r["f_cv_v"])
    var_means = {v: sum(by_var[v]) / len(by_var[v]) for v in d["variants"]}
    # f_CV = mean(p_up) - mean(self_c0) over all (context,layer,scope) rows
    f_cv = var_means["p_up"] - var_means["self_c0"]
    d["f_CV"] = f_cv
    d["f_CV_detail"] = {
        "definition": "mean_over_(context,layer,scope)[f_cv_v(p_up) - f_cv_v(self_c0)]",
        "interpretation": "input-localization fraction: real trained-CV->base patch effect above the base-CV->base null; ~0 => no input localization",
        "variant_means": var_means,
        "n_rows_per_variant": {v: len(by_var[v]) for v in d["variants"]},
    }
    json.dump(d, open(path, "w"), indent=2)
    summary.append((d["cell"], d["behavior"], f_cv, var_means["p_up"], var_means["self_c0"]))

print("Persisted f_CV per cell:")
fvals = []
for cell, beh, f, pup, sc0 in summary:
    fvals.append(f)
    print(f"  {cell:38s} ({beh:11s}) f_CV={f:+.4f}  (p_up={pup:.4f}, self_c0={sc0:.4f})")
print(f"\nMEAN f_CV across {len(fvals)} cells = {sum(fvals)/len(fvals):+.4f}")
print(f"RANGE f_CV = [{min(fvals):+.4f}, {max(fvals):+.4f}]")
