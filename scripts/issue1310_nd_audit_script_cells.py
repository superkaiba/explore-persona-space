"""Issue #1310 n<d estimator audit — the SCRIPT-format cells (recaptured store).

Extends `scripts/issue1310_nd_estimator_audit.py` to the cell family the audit
round could not reach: the run-2 SCRIPT-format per-persona cells, whose original
activation store was lost with its instance. The store was rebuilt by
`scripts/issue1310_recapture_script_store.py` (job 16086) and lives at HF
`issue1310_char_map/analysis_tensors/store_recap/<model>`; these rows are marked
`store: recaptured` in the corrections table.

Same four selector families and the same materiality thresholds as the parent
audit (`audit_cell` is imported, not reimplemented), at layer 19:

  ref_capped_gcv     GCV_DOF_CAP=0.9
  ambient_pure_gcv   GCV_DOF_CAP=None  (the selector 7 of these 8 cells published)
  inner_group_cv     4 inner GROUP folds
  reduced_pca_basis  train-fold PCA, k=min(1024, n_train//2)
  forced_lambda_*    1e2 / 1e3 / 1e4, diagnostic only

Per cell the PUBLISHED selector differs, so the reproduction arm differs:
the four base cells and instruct Wren/HELIOS/Dana published under AMBIENT
pure-GCV (their committed JSONs carry no `gcv_dof_cap` field), while instruct Vex
came from the completion round and published under the CAPPED selector.

Recapture fidelity (carried into every row): the BASE arm is span-exact —
re-attribution reproduced the published per-persona n exactly (2329/2466/1325/
2060). The INSTRUCT arm is a NEAR-REPLICA: the completion round's persisted
instruct pairs are span-incoherent with the persisted instruct story text
(9.93% of spans overflow), so spans were re-derived and per-persona n lands
within 3.21% (worst: Vex 3471 vs 3586). Instruct rows are therefore evidence
about the SELECTOR at that n and regime, not a byte-exact reproduction.

Output: `eval_results/issue_1310/nd_estimator_audit/cells_script_*.json` +
an extended `corrections_table.json`.
"""

from __future__ import annotations

import gc
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps before heavy imports (#847)

import numpy as np  # noqa: E402
import torch  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import issue1310_common as c1310  # noqa: E402
from issue1310_nd_estimator_audit import L, OUT, audit_cell  # noqa: E402

DATA_REPO = "superkaiba1/explore-persona-space-data"
STORE_PREFIX = "issue1310_char_map/analysis_tensors/store_recap"
CACHE = REPO / "data" / "issue_1310" / "recap_slice_dl"
EV = REPO / "eval_results" / "issue_1310"

# Per cell: the committed JSON carrying the published layer-19 value, and which
# selector produced it (drives which arm is the reproduction reference).
PUBLISHED = (
    {("base", p): (EV / f"cells_base_{p}.json", "ambient_pure_gcv") for p in c1310.PERSONA_LABELS}
    | {
        ("instruct", p): (EV / f"cells_instruct_{p}.json", "ambient_pure_gcv")
        for p in ("Wren", "HELIOS", "Dana")
    }
    | {
        ("instruct", "Vex"): (
            EV / "script_completion" / "cells_scriptc_instruct_Vex.json",
            "ref_capped_gcv",
        )
    }
)
FIDELITY = {
    "base": "span-exact (re-attribution reproduced published n exactly)",
    "instruct": (
        "near-replica (persisted instruct pairs span-incoherent with persisted "
        "stories; spans re-derived, per-persona n within 3.21%)"
    ),
}


def stream_l19(model_kind: str) -> dict:
    """Download recapture shards one at a time, keep ONLY layer 19, delete each.

    Prefix-scoped enumeration (#833: a bare `list_repo_files` on the ~1M-file
    data repo wedges); peak disk ~one shard, peak RAM ~one layer slice.
    """
    from huggingface_hub import HfApi

    from explore_persona_space.orchestrate import hub

    api = HfApi()
    prefix = f"{STORE_PREFIX}/{model_kind}"
    # Canonical retried, server-side-SCOPED listing (#920/#997) — never a bare
    # list_repo_tree / full-repo listing on the ~1M-file data repo (#833).
    names = sorted(
        path
        for path in hub.list_hf_files_under_path(api, DATA_REPO, prefix, repo_type="dataset")
        if path.rsplit("/", 1)[-1].startswith(f"{model_kind}_shard") and path.endswith(".pt")
    )
    assert names, f"no shards under {prefix}"
    CACHE.mkdir(parents=True, exist_ok=True)
    chars, groups, xs, ys = [], [], [], []
    for name in names:
        local = CACHE / Path(name).name
        hub.stage_hub_file(DATA_REPO, name, local, repo_type="dataset", overwrite=True)
        payload = torch.load(local, map_location="cpu", weights_only=False)
        chars.extend(payload["char_ids"])
        groups.extend(payload["group_ids"])
        xs.append(payload["arrays"]["x_spanmean"][:, L, :].float().numpy())
        ys.append(payload["arrays"]["y"][:, L, :].float().numpy())
        del payload
        local.unlink()  # stream-reduce: peak disk ~one shard
        gc.collect()
        print(f"[script-audit] {Path(name).name}: sliced L{L}, deleted local", flush=True)
    return {
        "char_ids": np.asarray(chars),
        "group_ids": np.asarray(groups),
        "X": np.concatenate(xs, axis=0),
        "Y": np.concatenate(ys, axis=0),
    }


def _published_value(path: Path) -> float | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())["r2_per_layer_obs"][L]


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    records: list[dict] = []
    for model_kind in ("base", "instruct"):
        store = stream_l19(model_kind)
        print(
            f"[script-audit] {model_kind}: {store['X'].shape[0]} rows (L{L} slice)",
            flush=True,
        )
        for persona in c1310.PERSONA_LABELS:
            cid = f"script_{model_kind}_{persona}"
            out_path = OUT / f"cells_{cid}.json"
            if out_path.exists():
                records.append(json.loads(out_path.read_text()))
                print(f"[script-audit] {cid}: resume-skip", flush=True)
                continue
            pub_path, pub_selector = PUBLISHED[(model_kind, persona)]
            published = _published_value(pub_path)
            mask = store["char_ids"] == persona
            # audit_cell asserts the ref arm reproduces `published`; that only
            # holds for a CAPPED-published cell, so pass published=None for the
            # ambient-published ones and record the reproduction check below.
            rec = audit_cell(
                cid,
                store["X"][mask],
                store["Y"][mask],
                store["group_ids"][mask],
                published if pub_selector == "ref_capped_gcv" else None,
            )
            rec["family"] = "script_format_recaptured"
            rec["model"] = model_kind
            rec["persona"] = persona
            rec["store"] = "recaptured"
            rec["recapture_fidelity"] = FIDELITY[model_kind]
            rec["published_r2_l19"] = published
            rec["published_selector"] = pub_selector
            repro = rec["arms"][pub_selector]["r2_pooled"]
            rec["published_selector_reproduction"] = {
                "arm": pub_selector,
                "recomputed": repro,
                "published": published,
                "abs_delta": None if published is None else abs(repro - published),
            }
            out_path.write_text(json.dumps(rec, indent=1))
            records.append(rec)
        del store
        gc.collect()

    table_path = OUT / "corrections_table.json"
    table = json.loads(table_path.read_text())
    existing = {c["cell_id"] for c in table["cells"]}
    table["cells"].extend(r for r in records if r["cell_id"] not in existing)
    table["script_format_recapture"] = {
        "store_prefix": STORE_PREFIX,
        "recapture_job": "16086 (fellows/charmander)",
        "fidelity": FIDELITY,
        "note": (
            "The original run-2 script-format store was lost with its instance; "
            "these rows are fit on the rebuilt store. 7 of 8 published under "
            "ambient pure-GCV, instruct Vex under the capped selector."
        ),
    }
    table_path.write_text(json.dumps(table, indent=1))
    print(f"[script-audit] extended {table_path} (+{len(records)} script cells)")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    sys.exit(rc)  # explicit exit before C-extension finalize race (#1689)
