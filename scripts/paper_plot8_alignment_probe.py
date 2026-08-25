"""Plot-8 feasibility probe: can cell E be built from banked tensors, with no new compute?

Plot 8 asks whether a PRE-CoT-trained model's context vector predicts the POST-CoT-trained
model's answer state (cell E). No banked artifact holds that cross-model fit: #928 fit
everything within OpenThinker2-7B. But the two halves may already exist:

  pre  = Qwen/Qwen2.5-7B-Instruct   -> `issue658_theory_assumptions/store`
  post = open-thoughts/OpenThinker2-7B (an SFT of that exact checkpoint)
                                    -> `issue928_cot_decomposition/.../percq_summaries`

They are only poolable if they were measured over the SAME rows. This probe compares the
two store manifests on the four things that decide it: probe-pool hash, context id set,
capture layers, and hidden size. It fits nothing and downloads only the two manifests.

It also reports the estimator regime the fit WOULD run in, because that is a separate gate:
the banked design matrix is the raw summary at full hidden width, so n_train < d and a fit
there is estimator-degenerate by the project rule (`.claude/rules/` — the #1701 refusal).
A clean alignment verdict means "assemblable", not "go fit it".

Usage:
    uv run python scripts/paper_plot8_alignment_probe.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # BEFORE heavy imports — shared-VM thread caps bind in-process (#847)

from huggingface_hub import hf_hub_download  # noqa: E402

from explore_persona_space.orchestrate.hub import retry_transient  # noqa: E402


DATA_REPO = "superkaiba1/explore-persona-space-data"
PRE_MANIFEST = "issue658_theory_assumptions/store/store_manifest.json"
POST_MANIFEST = "issue928_cot_decomposition/analysis_tensors/store/percq_summaries/manifest.json"

# The banked per-question row count and hidden width (#928 recon_skill_grid.json).
BANKED_N_INDIV_ROWS = 1994


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", default="eval_results/issue_2546/plot8_alignment_probe.json")
    return p.parse_args(argv)


def load_manifest(path_in_repo: str) -> dict:
    local = retry_transient(
        lambda: hf_hub_download(DATA_REPO, path_in_repo, repo_type="dataset"),
        what=f"download {path_in_repo}",
    )
    return json.loads(Path(local).read_text())


def probe() -> dict:
    pre = load_manifest(PRE_MANIFEST)
    post = load_manifest(POST_MANIFEST)

    pre_ctx = set(pre["context_ids"])
    post_ctx = set(post["context_ids"])
    hidden_pre = pre.get("hidden")
    hidden_post = post.get("hidden_size")

    checks = {
        "probe_pool_hash_match": pre.get("probe_pool_hash") == post.get("probe_pool_hash"),
        "context_id_set_match": pre_ctx == post_ctx,
        "capture_layers_match": pre.get("capture_layers") == post.get("capture_layers"),
        "hidden_size_match": hidden_pre == hidden_post,
    }
    aligned = all(checks.values())

    return {
        "dv": "feasibility of Plot-8 cell E (pre-CoT-trained v_C -> post-CoT-trained v_A) "
        "from banked tensors only",
        "pre": {
            "model": pre.get("model"),
            "source": f"{DATA_REPO}:{PRE_MANIFEST}",
            "n_contexts": len(pre_ctx),
            "hidden": hidden_pre,
            "probe_pool_hash": pre.get("probe_pool_hash"),
        },
        "post": {
            "model": "open-thoughts/OpenThinker2-7B",
            "source": f"{DATA_REPO}:{POST_MANIFEST}",
            "n_contexts": len(post_ctx),
            "hidden": hidden_post,
            "probe_pool_hash": post.get("probe_pool_hash"),
            "n_probes": post.get("n_probes"),
            "summary_names": post.get("summary_names"),
        },
        "checks": checks,
        "aligned": aligned,
        "context_ids_only_in_pre": sorted(pre_ctx - post_ctx),
        "context_ids_only_in_post": sorted(post_ctx - pre_ctx),
        "estimator_regime_if_fit": {
            "n_train_rows_per_question": BANKED_N_INDIV_ROWS,
            "feature_dim": hidden_post,
            "n_lt_d": bool(hidden_post is not None and BANKED_N_INDIV_ROWS < hidden_post),
            "verdict": (
                "REFUSED for a headline fit: the banked design is the raw summary at full "
                "hidden width, so n_train < d and held-out R2 there is estimator-degenerate. "
                "#2546's corpora put n_train above d by construction; the cross-model cell "
                "belongs there, not here."
            ),
        },
        "conclusion": (
            "cell E is ASSEMBLABLE from banked tensors (rows align exactly); it is NOT fit "
            "here, on the estimator-regime gate above"
            if aligned
            else "cell E is NOT assemblable from banked tensors: the stores do not align"
        ),
    }


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = probe()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")

    for name, ok in report["checks"].items():
        print(f"  {name}: {'PASS' if ok else 'FAIL'}")
    print(f"aligned = {report['aligned']}")
    print(f"n<d if fit = {report['estimator_regime_if_fit']['n_lt_d']}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
