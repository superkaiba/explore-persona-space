"""Free-analysis follow-up B for issue #2222 — production-wave rule-28 judge accounting.

Pure JSON reduce over the Form-A production judge artifacts (batch wave +
rule-28/24(ii) sync rejudge + merge): per trait totals with the drop-class
split (content / instructed-REFUSAL / truncation / API-classifier refusal /
transport), per trait x arm (dataset VERSION class — the wave's registered arm
grain) splits derived from the per-item maps, per-dataset fine grain, sync
recovery + zero-kept-item accounting, and the chain-log classifier lines as
provenance. Inputs are the LOCAL ``data/issue_2222/form_a/`` copies of the HF
``issue2222_pvscreen/raw_completions/form_a_judge/`` uploads plus the
``/mnt/eps-data/<user>/issue2222_judge/`` chain logs. No model calls; no
completion/rationale text is read — only counts and score lists.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:  # sibling-script imports in script mode (#823)
    sys.path.insert(0, str(_SCRIPTS_DIR))

import issue2222_judge as judge  # noqa: E402
import issue2222_lib as lib  # noqa: E402

VERSIONS = ("normal", "misaligned_1", "misaligned_2")
_LOG_KEEP_RE = re.compile(
    r"judge reduce:|\[phase=p4_(judge|rejudge|pilot)\]|\[chain2?\] (stage|HALT|judge wave)"
)


def default_log_dir() -> Path:
    return Path("/mnt/eps-data") / os.environ.get("USER", "thomasjiralerspong") / "issue2222_judge"


def _by_arm(per_item: dict[str, int] | None) -> tuple[dict[str, int], dict[str, int]]:
    """(per-version, per-dataset) sums of a per-item count map."""
    by_ver: dict[str, int] = dict.fromkeys(VERSIONS, 0)
    by_ds: dict[str, int] = {}
    for iid, k in (per_item or {}).items():
        ds, _row = judge.split_item_id(iid)
        ver = lib.split_dataset_id(ds)[1]
        by_ver[ver] = by_ver.get(ver, 0) + int(k)
        by_ds[ds] = by_ds.get(ds, 0) + int(k)
    return by_ver, {k: by_ds[k] for k in sorted(by_ds)}


def reduce_trait(form_a: Path, trait: str, n_draws: int) -> dict:
    """Rule-28 accounting for one trait from result + merged artifacts."""
    result = json.loads((form_a / f"judge_result_{trait}.json").read_text())
    merged = json.loads((form_a / f"judge_merged_{trait}.json").read_text())
    n_items = int(result["n_items"])
    assert result["n_total_draws"] == n_items * n_draws, (trait, result["n_total_draws"])

    api_by_ver, api_by_ds = _by_arm(result.get("per_item_api_refusals"))
    tr_by_ver, tr_by_ds = _by_arm(result.get("per_item_transport_losses"))
    # Derived-by-version sums must reconcile with the recorded trait totals
    # (the same fail-loud check the content-drop derivation gets below).
    assert sum(api_by_ver.values()) == int(result["n_api_refusal_draws"]), (trait, api_by_ver)
    assert sum(tr_by_ver.values()) == int(result["n_transport_lost_draws"]), (trait, tr_by_ver)
    # Per-item CONTENT drops (batch wave) are not persisted directly; derive:
    # content_i = n_draws - kept_batch_i - api_refusal_i - transport_i.
    per_api = result.get("per_item_api_refusals") or {}
    per_tr = result.get("per_item_transport_losses") or {}
    content_per_item: dict[str, int] = {}
    for iid, scores in (result.get("per_item_scores") or {}).items():
        c = n_draws - len(scores) - int(per_api.get(iid, 0)) - int(per_tr.get(iid, 0))
        if c < 0:
            raise ValueError(f"{trait}/{iid}: negative derived content drops ({c})")
        if c:
            content_per_item[iid] = c
    derived_content = sum(content_per_item.values())
    if derived_content != int(result["n_dropped_draws_content"]):
        raise ValueError(
            f"{trait}: derived content drops {derived_content} != recorded "
            f"{result['n_dropped_draws_content']}"
        )
    content_by_ver, content_by_ds = _by_arm(content_per_item)

    # Sync-recovery + zero-kept accounting from the merged artifact.
    meta = merged["judge_meta"]
    sync_by_ver: dict[str, int] = dict.fromkeys(VERSIONS, 0)
    zero_by_ver: dict[str, int] = dict.fromkeys(VERSIONS, 0)
    kept_by_ver: dict[str, int] = dict.fromkeys(VERSIONS, 0)
    for iid, row in merged["per_item"].items():
        ver = lib.split_dataset_id(judge.split_item_id(iid)[0])[1]
        sync_by_ver[ver] += int(row["n_sync"])
        kept_by_ver[ver] += len(row["scores"])
        if row["mean"] is None:
            zero_by_ver[ver] += 1
    n_content = int(result["n_dropped_draws_content"])
    n_refusal = int(result["n_refusal_draws_instructed"])
    n_trunc = int(result["n_truncation_dropped_draws"])
    return {
        "trait": trait,
        "n_items": n_items,
        "n_draws_per_item": n_draws,
        "batch_wave_totals": {
            "n_total_draws": result["n_total_draws"],
            "n_api_refusal_draws": result["n_api_refusal_draws"],
            "n_transport_lost_draws": result["n_transport_lost_draws"],
            "n_content_dropped_draws": n_content,
            "content_split_trait_level": {
                "instructed_refusal": n_refusal,
                "truncation_rule23": n_trunc,
                "malformed_or_out_of_range_parse_fail": n_content - n_refusal - n_trunc,
                "note": "split persisted at trait grain only (per-item refusal/"
                "truncation breakdowns are not in the result artifact)",
            },
            "stop_reason_tally": result.get("stop_reason_tally"),
        },
        "by_version": {
            "api_refusal_draws": api_by_ver,
            "api_refusal_draws_recorded": result.get("n_api_refusal_by_version"),
            "transport_lost_draws": tr_by_ver,
            "content_dropped_draws": content_by_ver,
            "sync_recovered_draws": sync_by_ver,
            "kept_draws_after_merge": kept_by_ver,
            "zero_kept_items": zero_by_ver,
        },
        "by_dataset": {
            "api_refusal_draws": api_by_ds,
            "transport_lost_draws": tr_by_ds,
            "content_dropped_draws": content_by_ds,
        },
        "sync_reissue": meta["sync_reissue"],
        "n_items_zero_kept_draws": meta["n_items_zero_kept_draws"],
        "kept_draw_fraction_after_merge": (sum(kept_by_ver.values()) / result["n_total_draws"]),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-root", default=str(lib.default_data_root()))
    ap.add_argument("--log-dir", default=str(default_log_dir()))
    ap.add_argument(
        "--out",
        default=str(
            lib.REPO_ROOT
            / "eval_results"
            / "issue_2222"
            / "followup_free_analysis"
            / "judge_accounting.json"
        ),
    )
    ap.add_argument("--n-draws", type=int, default=6)
    ap.add_argument("--import-check", action="store_true")
    args = ap.parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        raise SystemExit(0)
    form_a = judge.form_a_dir(Path(args.data_root))
    per_trait = {trait: reduce_trait(form_a, trait, args.n_draws) for trait in lib.TRAITS}
    # Chain-log classifier lines (provenance; count/classifier lines only).
    log_lines: dict[str, list[str]] = {}
    for name in ("judge_chain.log", "judge_chain2.log"):
        p = Path(args.log_dir) / name
        if p.exists():
            log_lines[name] = [
                ln.rstrip("\n")
                for ln in p.open(encoding="utf-8")
                if _LOG_KEEP_RE.search(ln) is not None
            ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    lib.write_json_atomic(
        out,
        {
            "note": "rule-28 accounting reduce of the Form-A production judge wave "
            "(batch + rule-24(ii) sync rejudge + merge). Arm grain = dataset VERSION "
            "class (the wave's registered grain); per-dataset fine grain included. "
            "Sources: local form_a mirrors of the HF form_a_judge uploads + chain logs.",
            "instrument": json.loads((form_a / "judge_result_evil.json").read_text()).get(
                "instrument"
            ),
            "per_trait": per_trait,
            "chain_log_lines": log_lines,
            **lib.run_metadata(),
        },
    )
    lib.log_phase("fu_judge_accounting", "done", out=str(out))
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
