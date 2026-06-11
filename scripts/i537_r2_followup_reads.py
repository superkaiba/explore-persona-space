"""Issue #537 round-2 follow-up reads (zero-GPU, interpretation-critic round 1).

Computes four small reads requested by the round-1 critique union, all from
existing artifacts (local tensor/analysis JSONs + the pinned HF data revision):

  1. gen-truncation scan over the long-prefix columns (wc_xlong_ho /
     wc_xxlong_ho): finish_reason fractions for every judge-row raw-completion
     bucket on HF, plus the marker row's frozen base-response
     gen_truncated_frac — discharges the plan's registered truncation check
     before crediting the 4-9k-token columns.
  2. readable-subset behavior-dependence: median cross-behavior Spearman rho
     over the 4 readable rows (refusal excluded as noise-limited).
  3. binst_marker off-diagonal exceptions: cells with positive movement.
  4. EM contrastive off-diagonal means per training context (chat-prefix-led
     ordering).

Output: eval_results/issue_537/analysis/r2_followups.json
"""

from __future__ import annotations

import json
import statistics
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download

EVAL = Path("eval_results/issue_537")
REPO = "superkaiba1/explore-persona-space-data"
REV = "db3662ae1d1ff4484ada027ac92a2658c4dec2e8"
PREFIX = "issue537_context_generalization"
LONG_COLS = ["wc_xlong_ho", "wc_xxlong_ho"]
JUDGE_DIRS = ["fact", "refusal", "sycophancy", "em", "emnc"]

out: dict = {"hf_revision": REV}

# --- 1. truncation scan ------------------------------------------------------
from huggingface_hub import list_repo_files  # noqa: E402

files = list_repo_files(REPO, repo_type="dataset", revision=REV)
targets = [
    f
    for f in files
    if f.startswith(f"{PREFIX}/raw_completions/")
    and f.split("/")[2] in JUDGE_DIRS
    and any(f.endswith(f"{c}.json") for c in LONG_COLS)
]
trunc: dict = {}
total, truncated = 0, 0
for fpath in sorted(targets):
    p = hf_hub_download(REPO, fpath, repo_type="dataset", revision=REV)
    with open(p) as fh:
        d = json.load(fh)
    gens = d["generations"]
    n, n_len = 0, 0
    for _q, rows in gens.items():
        for r in rows:
            n += 1
            if r.get("finish_reason") != "stop":
                n_len += 1
    total += n
    truncated += n_len
    if n_len:
        trunc[fpath.removeprefix(f"{PREFIX}/raw_completions/")] = {
            "n": n,
            "n_not_stop": n_len,
        }
out["truncation_scan_long_columns"] = {
    "files_scanned": len(targets),
    "completions_scanned": total,
    "completions_not_finish_stop": truncated,
    "frac_truncated": truncated / total if total else None,
    "files_with_any_truncation": trunc,
}

# marker row: frozen base responses carry gen_truncated_frac directly
base_trunc = {}
for ev in LONG_COLS:
    p = hf_hub_download(
        REPO, f"{PREFIX}/data/responses_eval/{ev}.json", repo_type="dataset", revision=REV
    )
    with open(p) as fh:
        base_trunc[ev] = json.load(fh)["gen_truncated_frac"]
out["marker_base_response_gen_truncated_frac"] = base_trunc

# --- 2. readable-subset behavior dependence ----------------------------------
with open(EVAL / "analysis/registered_reads.json") as fh:
    reads = json.load(fh)
pw = reads["h_behavior_dependence"]["pairwise"]
readable = {"marker", "fact", "sycophancy", "em"}
subset = {k: v["rho"] for k, v in pw.items() if set(k.split("~")) <= readable}
out["behavior_dependence_readable_subset"] = {
    "behaviors": sorted(readable),
    "pairwise_rho": subset,
    "median_rho": statistics.median(subset.values()),
    "note": "refusal excluded (noise-limited row, failed h_structure floor)",
}

# --- 3. binst_marker off-diagonal exceptions ----------------------------------
with open(EVAL / "G_tensor/G_meta.json") as fh:
    meta = json.load(fh)["per_cell"]
binst = {
    k.split("__")[1]: v["g"]
    for k, v in meta.items()
    if k.startswith("marker/binst_marker__") and not k.endswith("__binst_marker")
}
out["binst_marker_offdiag"] = {
    "mean": statistics.mean(binst.values()),
    "cells_above_plus_0p5_nat": {
        c: round(g, 3) for c, g in sorted(binst.items(), key=lambda x: -x[1]) if g > 0.5
    },
    "n_offdiag": len(binst),
}

# --- 4. EM contrastive off-diagonal means per training context ----------------
em_rows: dict[str, list[float]] = defaultdict(list)
for k, v in meta.items():
    if k.startswith("em/"):
        tr, ev = k.split("/")[1].split("__")
        if tr != ev:
            em_rows[tr].append(v["g"])
out["em_contrastive_offdiag_mean_by_train_ctx"] = {
    t: round(statistics.mean(vs), 4)
    for t, vs in sorted(em_rows.items(), key=lambda x: -statistics.mean(x[1]))
}

dest = EVAL / "analysis/r2_followups.json"
with open(dest, "w") as f:
    json.dump(out, f, indent=1)
print(json.dumps(out, indent=1))
print(f"\nwrote {dest}")
