"""Step-1 shared conversation sampler for issue #2479 → panel_manifest.json.

Downloads the pinned Step-1 inputs (plan v4 §4 Step 1 / §10):

- ``matched_subsets.json`` at the parent pin (``REUSE_REV``
  ``2a3cb30acada04defc84fd04d28a2b54da3104cd``; prefix
  ``issue1345_framing/inputs/matched_n``) — the paired-eligible pool is its
  ``shared_r1r2_convs`` id list.
- The r4op kept-story JSONL under
  ``issue1345_framing/onpolicy_assistant_story/raw_completions/stories/`` and
  the r4 kept-story JSONL under
  ``issue1345_framing/conversation_paired_stories_assistant/raw_completions/stories/``
  — located via a SCOPED ``list_repo_tree`` per prefix (never unscoped), with
  the resolved revision per prefix RECORDED in the manifest (plan §10, the
  smoke-input stager's record-resolved-revision convention). NOTE (observed
  2026-08-22): the r4 prefix holds TWO kept files — the 2,164-row paired
  (verbatim-embed) cell AND the parent's 117-row on-policy COMPANION control
  (``kept_stories_paired_op_instruct.jsonl``); selection is therefore by
  MODE-EXACT basename (``kept_stories_paired_instruct.jsonl`` for r4,
  ``kept_stories_paired_op_instruct.jsonl`` for r4op), never a bare
  ``kept_*`` glob union.

Loads all three conversation-id sets IN FULL and records set sizes + every
pairwise + triple intersection count (the §12.7 full-grain structural probe),
then draws:

- the seed-42 shared sample of 1,600 conversations, preferentially from
  (r4op-kept ∩ r4-kept ∩ paired-eligible), topped up from
  (r4op-kept ∩ eligible), then (eligible) — FAIL-LOUD if the three tiers
  together cannot fill the sample (report, never silently degrade);
- the seed-0 axis reservation of 250 conversations from the 1,600.

Crash-fix r8: ``paired-eligible`` is (shared_r1r2_convs ∩ eligible_paired ∩
eligible_op), where the two eligible sets come from the REQUIRED
``--eligible-ids`` export (``issue1345_gen_stories_paired.py
--emit-eligible-ids`` — the gen script's own answer_too_short /
answer_over_budget / prompt_over_budget filters, single-sourced). Without it
the sampler registered ids the gen script drops, and every non-op panel cell
fail-louded at restrict_pool_to_manifest (P0 gen-smoke gate, 2026-08-23). The
manifest records the export's path + sha256 + embedded provenance.

Writes ``eval_results/issue_2479/panel_manifest.json`` (sample ids,
reservation ids, per-tier counts, intersection table, resolved input
revisions, seeds, git provenance) — committed BEFORE any generation.

Runs on the VM at Step-1 time: the inputs are small JSONL/JSON text
(~tens of MB, far under the ~10 GB pod-routing threshold); staging defaults
to ``/tmp/issue2479_step1_inputs``.

CONTENT HYGIENE: kept-story rows wrap real LMSYS conversation text — this
script reads ONLY the ``conv_id`` field per row and never prints, logs, or
persists any content field.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()  # credentials + shared-VM thread caps BEFORE any HF import

from huggingface_hub import HfApi, hf_hub_download  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402
from explore_persona_space.orchestrate.provenance import (  # noqa: E402
    as_metadata_dict,
    git_provenance,
)

HF_DATA_REPO = "superkaiba1/explore-persona-space-data"

# Parent matched-n allowlist pin (issue1345_common.REUSE_REV; plan §10 row 3).
MATCHED_REV = "2a3cb30acada04defc84fd04d28a2b54da3104cd"
MATCHED_PATH = "issue1345_framing/inputs/matched_n/matched_subsets.json"

# The two kept-story source cells (plan §4 Step 1). ``kept_basename`` is
# MODE-EXACT (see module docstring: the r4 prefix also carries the parent's
# 117-row on-policy companion kept file, which must NOT enter the r4 set).
SOURCES = {
    "r4op_kept": {
        "prefix": "issue1345_framing/onpolicy_assistant_story/raw_completions/stories",
        "kept_basename": "kept_stories_paired_op_instruct.jsonl",
    },
    "r4_kept": {
        "prefix": "issue1345_framing/conversation_paired_stories_assistant/raw_completions/stories",
        "kept_basename": "kept_stories_paired_instruct.jsonl",
    },
}


def _load_conv_ids(path: Path) -> list[str]:
    """conv_id per kept row (content fields never touched); fail-loud on schema."""
    ids: list[str] = []
    with path.open() as fh:
        for lineno, line in enumerate(fh):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if "conv_id" not in row:
                raise KeyError(
                    f"{path.name} row {lineno}: no 'conv_id' field "
                    f"(observed keys: {sorted(row.keys())})"
                )
            cid = row["conv_id"]
            if not isinstance(cid, str) or not cid:
                raise ValueError(f"{path.name} row {lineno}: conv_id must be non-empty str")
            ids.append(cid)
    if not ids:
        raise ValueError(f"{path} holds no rows")
    return ids


def main(argv: list[str] | None = None) -> int:
    """Build + write panel_manifest.json (see module docstring)."""
    ap = argparse.ArgumentParser(
        description="issue #2479 Step-1: shared conversation sample + axis reservation manifest"
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=_REPO_ROOT / "eval_results" / "issue_2479" / "panel_manifest.json",
    )
    ap.add_argument("--staging-dir", type=Path, default=Path("/tmp/issue2479_step1_inputs"))
    ap.add_argument("--n-sample", type=int, default=1600)
    ap.add_argument("--n-reservation", type=int, default=250)
    ap.add_argument(
        "--eligible-ids",
        type=Path,
        required=True,
        help="REQUIRED (crash-fix r8): dual-regime gen-feasibility export written by "
        "`issue1345_gen_stories_paired.py --emit-eligible-ids` (keys eligible_paired / "
        "eligible_op / counts / provenance). The paired-eligible tier is restricted to "
        "(shared_r1r2_convs ∩ eligible_paired ∩ eligible_op) BEFORE tiered sampling, so "
        "no registered id can fall in the gen script's answer_too_short / "
        "answer_over_budget dropped set (the P0 gen-smoke gate failure, 2026-08-23).",
    )
    args = ap.parse_args(argv)
    args.staging_dir.mkdir(parents=True, exist_ok=True)

    api = HfApi()

    # --- resolve main -> sha ONCE and pin every story-prefix call to it ------
    # (revision=None across paired hub calls can split a run across snapshots;
    # the matched-n file keeps its own parent pin.)
    repo_sha = api.repo_info(HF_DATA_REPO, repo_type="dataset").sha
    print(f"[panel-sample] resolved {HF_DATA_REPO}@main -> {repo_sha}")

    # --- matched-n allowlist (paired-eligible pool) --------------------------
    matched_local = hub.retry_transient(
        lambda: hf_hub_download(
            HF_DATA_REPO,
            MATCHED_PATH,
            repo_type="dataset",
            revision=MATCHED_REV,
            local_dir=args.staging_dir,
        ),
        what="matched_subsets hf_hub_download",
    )
    matched = json.loads(Path(matched_local).read_text())
    if "shared_r1r2_convs" not in matched:
        raise KeyError(
            f"matched_subsets.json missing 'shared_r1r2_convs' "
            f"(top-level keys: {sorted(matched.keys())})"
        )
    eligible = set(matched["shared_r1r2_convs"])
    if not eligible or not all(isinstance(x, str) for x in eligible):
        raise ValueError("shared_r1r2_convs must be a non-empty list of str conv ids")

    # --- dual-regime gen-feasibility restriction (crash-fix r8) --------------
    # The gen script's own eligibility (answer_too_short at pool join;
    # answer_over_budget / prompt_over_budget in _filter_pool_feasible, per
    # regime) is SINGLE-SOURCED in its --emit-eligible-ids export — never
    # re-implemented here. Restrict the paired-eligible tier to ids feasible
    # under BOTH consumer regimes so every registered sample id survives
    # restrict_pool_to_manifest in the paired AND --op-powered cells.
    eligible_raw = args.eligible_ids.read_bytes()
    elig = json.loads(eligible_raw)
    # C2 export-schema hardening (concern round, r8): the four emit keys are all
    # REQUIRED; provenance must be a typed non-empty dict (a JSON null passes a
    # bare `in` check and embeds an empty audit record); duplicate ids are
    # REJECTED loud (a silent set-dedup would mask a producer bug); the counts
    # fields are cross-checked against the arrays they claim to describe.
    for key in ("eligible_paired", "eligible_op", "counts", "provenance"):
        if key not in elig:
            raise KeyError(
                f"{args.eligible_ids}: missing {key!r} (top-level keys: {sorted(elig)}) — "
                "regenerate via issue1345_gen_stories_paired.py --emit-eligible-ids"
            )
    if not isinstance(elig["provenance"], dict) or not elig["provenance"]:
        raise ValueError(
            f"{args.eligible_ids}: provenance must be a non-empty dict "
            f"(got {type(elig['provenance']).__name__}) — the manifest embeds it as the "
            "export's audit record"
        )
    if not isinstance(elig["counts"], dict):
        raise ValueError(
            f"{args.eligible_ids}: counts must be a dict (got {type(elig['counts']).__name__})"
        )

    def _id_set(key: str) -> set[str]:
        ids = elig[key]
        if not isinstance(ids, list) or not ids or not all(isinstance(x, str) and x for x in ids):
            raise ValueError(f"{args.eligible_ids}: {key} must be a non-empty list of str ids")
        uniq = set(ids)
        if len(uniq) != len(ids):
            raise ValueError(
                f"{args.eligible_ids}: {key} contains {len(ids) - len(uniq)} duplicate ids — "
                "rejecting loud (a silent set-dedup would mask a producer bug); regenerate "
                "via issue1345_gen_stories_paired.py --emit-eligible-ids"
            )
        declared = elig["counts"].get(f"n_{key}")
        if declared != len(ids):
            raise ValueError(
                f"{args.eligible_ids}: counts[n_{key}]={declared!r} does not match "
                f"len({key})={len(ids)} — export internally inconsistent; regenerate"
            )
        return uniq

    eligible_paired_ids = _id_set("eligible_paired")
    eligible_op_ids = _id_set("eligible_op")
    n_shared_before = len(eligible)
    eligible &= eligible_paired_ids & eligible_op_ids
    print(
        f"[panel-sample] gen-feasibility restriction: shared={n_shared_before} ∩ "
        f"eligible_paired={len(eligible_paired_ids)} ∩ eligible_op={len(eligible_op_ids)} "
        f"-> {len(eligible)} paired-eligible conv ids",
        flush=True,
    )
    if not eligible:
        raise RuntimeError(
            "paired-eligible tier is EMPTY after the gen-feasibility restriction "
            f"(shared={n_shared_before}, eligible_paired={len(eligible_paired_ids)}, "
            f"eligible_op={len(eligible_op_ids)}) — conv-id key-space mismatch suspected"
        )

    # --- kept-story conv-id sets (scoped listing + mode-exact selection) -----
    id_sets: dict[str, set[str]] = {}
    input_records: dict[str, dict] = {
        "matched_subsets": {
            "path": MATCHED_PATH,
            "revision": MATCHED_REV,
            "top_level_keys": sorted(matched.keys()),
            "n_shared_r1r2_convs": len(matched["shared_r1r2_convs"]),
            "n_unique": n_shared_before,
        },
        "eligible_ids": {
            "path": str(args.eligible_ids),
            "sha256": hashlib.sha256(eligible_raw).hexdigest(),
            "n_eligible_paired": len(eligible_paired_ids),
            "n_eligible_op": len(eligible_op_ids),
            "n_shared_before_restrict": n_shared_before,
            "n_paired_eligible_after_restrict": len(eligible),
            "provenance": elig["provenance"],
        },
    }
    for label, src in SOURCES.items():
        prefix = src["prefix"]
        files = hub.list_hf_files_under_path(
            api, HF_DATA_REPO, prefix, repo_type="dataset", revision=repo_sha
        )
        kept_files = sorted(
            f for f in files if Path(f).name.startswith("kept_") and f.endswith(".jsonl")
        )
        want = f"{prefix}/{src['kept_basename']}"
        if want not in kept_files:
            raise FileNotFoundError(
                f"{label}: expected kept file {want!r} not under prefix "
                f"(kept files found: {kept_files})"
            )
        local = hub.retry_transient(
            lambda want=want: hf_hub_download(
                HF_DATA_REPO,
                want,
                repo_type="dataset",
                revision=repo_sha,
                local_dir=args.staging_dir,
            ),
            what=f"kept-story hf_hub_download ({label})",
        )
        ids = _load_conv_ids(Path(local))
        id_sets[label] = set(ids)
        input_records[label] = {
            "prefix": prefix,
            "revision": repo_sha,
            "kept_files_in_prefix": kept_files,
            "selected_file": want,
            "n_rows": len(ids),
            "n_unique_conv_ids": len(id_sets[label]),
        }
        print(
            f"[panel-sample] {label}: {len(ids)} rows / {len(id_sets[label])} unique "
            f"conv_ids from {Path(want).name}"
        )

    r4op, r4 = id_sets["r4op_kept"], id_sets["r4_kept"]

    # --- §12.7 full-grain intersection probe ---------------------------------
    triple = r4op & r4 & eligible
    intersections = {
        "n_r4op_kept": len(r4op),
        "n_r4_kept": len(r4),
        "n_eligible": len(eligible),
        "n_r4op_and_r4": len(r4op & r4),
        "n_r4op_and_eligible": len(r4op & eligible),
        "n_r4_and_eligible": len(r4 & eligible),
        "n_triple": len(triple),
        "n_r4op_outside_eligible": len(r4op - eligible),
        "n_r4_outside_eligible": len(r4 - eligible),
    }
    print(f"[panel-sample] intersections: {json.dumps(intersections)}")
    if not triple:
        raise RuntimeError(
            "triple intersection (r4op ∩ r4 ∩ eligible) is EMPTY — conv-id key-space "
            f"mismatch suspected; intersections: {intersections}"
        )

    # --- seed-42 preferential sample (tier1 -> tier2 -> tier3 top-up) --------
    tier1 = sorted(triple)
    tier2 = sorted((r4op & eligible) - triple)
    tier3 = sorted(eligible - (r4op & eligible))
    rng = random.Random(42)
    sample: list[str] = []
    tier_records: dict[str, dict] = {}
    for name, pool in (
        ("tier1_triple", tier1),
        ("tier2_r4op_and_eligible", tier2),
        ("tier3_eligible", tier3),
    ):
        need = args.n_sample - len(sample)
        take = list(pool) if len(pool) <= need else rng.sample(pool, need)
        sample.extend(take)
        tier_records[name] = {"available": len(pool), "taken": len(take)}
    if len(sample) != args.n_sample:
        raise RuntimeError(
            f"cannot fill the {args.n_sample}-conversation sample: only {len(sample)} "
            f"available across all tiers ({tier_records}) — re-scope the top-up rule "
            "per plan §12.7; never silently degrade"
        )
    assert len(set(sample)) == len(sample), "tier pools must be disjoint"

    # --- seed-0 axis reservation (250 of the 1,600) ---------------------------
    reservation = sorted(random.Random(0).sample(sorted(sample), args.n_reservation))
    assert set(reservation) <= set(sample)

    manifest = {
        "issue": 2479,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "seeds": {"conversation_sample": 42, "axis_reservation": 0},
        "n_sample": args.n_sample,
        "n_reservation": args.n_reservation,
        "hf_repo": HF_DATA_REPO,
        "inputs": input_records,
        "intersections": intersections,
        "tiers": tier_records,
        "sample_conv_ids": sorted(sample),
        "axis_reservation_conv_ids": reservation,
        "metadata": as_metadata_dict(git_provenance(), phase="panel-sample"),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        f"[panel-sample] wrote {args.out}: n_sample={len(sample)} "
        f"n_reservation={len(reservation)} tiers={json.dumps(tier_records)}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
