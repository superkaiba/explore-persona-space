"""#1335 free-analysis follow-up (9a-ter): quantify the ROW-FILTER candidate for
the r0 Q&A endpoint discrepancy vs #825's naturalistic plain-text round.

The #1335 clean-result names four candidate deltas for the 0.17 gap between
r0_qa_full (base 0.410 / instruct 0.471, n=4,589/4,599) and #825's committed
S2N/S1N naturalistic single-turn cells (base 0.578 / instruct 0.625, n=4,724).
This script isolates the ROW-FILTER candidate on the persisted r0 stores:

  RECIPE COMPARISON (finding): the recipes DIFFER.
    - #825 naturalistic S applies NO token-count row filters: chat anchors
      S1/S2 kept 5,000/5,000; the ONLY drop is the zero-width-span
      (BPE-degenerate) render drop added in the #825 crash-fix
      (65ff2426a8087398c55ee3188637f81f0401617b,
      issue825_extract_turnstore.py::partition_rendered), realized
      n = 4,724/5,000 for BOTH models; fit-side is a NaN keep mask only
      (issue825_fit_cells.py::xy @ 3307b405263dbc70facf350b27918a6d7fb8dd59,
      the commit pinned in cells_S2N.json metadata; row_allowlist_applied
      false).
    - #1335 r0 drops at CAPTURE time (issue1335_extract_store.py::build_items
      @ 377f824b5f): completion >= 4 tokens, context >= 8 tokens (within the
      512-token cap window), row <= 2048 tokens — base dropped 79/329/3
      (short_dialogue/short_context/row_too_long), instruct 64/329/8.

  EXECUTABLE ARM (restrict_to_825_rowset): rows #825's recipe would DROP can
  be removed from the persisted store (refit on store rows whose q_idx is in
  #825's realized 4,724-row kept set); rows #1335's token filters dropped
  CANNOT be added back (never captured — re-extraction would need a GPU), so
  that direction is reported as counts (rows_unaddable*), not a refit.

Reproduction anchor first: the full-store layer-19 ctx refit must match the
committed cells_r0_qa_full__{model}__ctx.json value within 1e-6 (the same
anchor the round-2 companions reproduced to ~2e-16). Fit core is byte-for-byte
the committed rig: issue1335_refit_companions.fit_l19 -> issue825_fit_cells.
heldout_r2_sweep (lambda_selection="inner-group-cv", 5 outer folds, seed 0).

Row-set recompute provenance (no new data): #825's drop set is recomputed
OFFLINE from the local pinned track_s.jsonl (sha256 d20560b6..., byte-equal to
#825 round-9's pin) via the worktree issue825_extract_turnstore.to_single_turn
+ render_conv("naturalistic") — both byte-identical to the pinned #825 commit
(render_formats.py diff empty vs 302c8b6bfa) — asserting the recomputed kept
count equals the committed n=4,724 under BOTH model tokenizers. #1335's
per-row drop reasons are recomputed from the pinned-rev gen JSONLs via the
committed build_items predicates, cross-checked against build_items counters,
the capture sidecar, and the store's realized row_id set.

Ops: 4 single-layer L19 ridge fits (5 folds x 13-lambda inner-group CV,
d=3584, n<=4.7k) + 2x5,000 offline renders + ~9 GB pinned store staging
(stage -> slim layer-19 slice -> release). CPU-only, 0 GPU-h, projected
<= 15 min on the shared VM (thread-capped launch).

Content hygiene: track_s / gen rows are real-world lmsys text — row ids,
counts, and token counts only; NEVER print prompt/completion text.
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # thread caps must bind before torch import

import numpy as np  # noqa: E402
import torch  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue825_extract_turnstore as ext825  # noqa: E402
import issue931_common as common  # noqa: E402
import issue1335_extract_store as ext1335  # noqa: E402
import issue1335_fit as f1335  # noqa: E402
import issue1335_refit_companions as comp  # noqa: E402
import issue1335_render_rungs as r1335  # noqa: E402

SCRIPT = "scripts/issue1335_refit_r0_filters.py"
SLUG = "r0_qa_full"
L = comp.HEADLINE_LAYER  # 19
HF_REV = comp.HF_REV  # 08421fc22bbe42968670c4ffbfcc561dd9cf4aa5 (#1335 pinned rev)

# --- provenance constants (recovered 2026-07-17; see module docstring) -----
P825_COMMITTED = {
    # eval_results/issue_825/naturalistic-single-turn/cells_S{1,2}N.json @ main
    "base": {"cell": "S2N", "r2_l19": 0.5782882733523877, "n": 4724},
    "instruct": {"cell": "S1N", "r2_l19": 0.6249482470755414, "n": 4724},
}
P825_N_KEPT = 4724
P825_RECIPE = {
    "summary": (
        "NO token-count row filters (chat anchors S1/S2 kept 5000/5000); single "
        "tolerated drop = zero-width-span (BPE-degenerate) rows under the "
        "naturalistic render; fit-side NaN keep mask only, no row allowlist"
    ),
    "drop_predicate": "any content span (s, e) with s >= e after render_naturalistic",
    "provenance": {
        "drop_code": (
            "scripts/issue825_extract_turnstore.py::partition_rendered/"
            "degenerate_content_turns @ 65ff2426a8087398c55ee3188637f81f0401617b "
            "(branch issue-825 crash-fix)"
        ),
        "render_code": (
            "scripts/issue825_render_formats.py::render_naturalistic @ "
            "302c8b6bfaa1ab65d990fa4baa906288caa8293b — byte-identical to this "
            "worktree copy (diff empty)"
        ),
        "fit_code": (
            "scripts/issue825_fit_cells.py::xy (NaN keep only) @ "
            "3307b405263dbc70facf350b27918a6d7fb8dd59 (cells_S2N metadata pin)"
        ),
        "committed_cells": "eval_results/issue_825/naturalistic-single-turn/cells_S{1,2}N.json",
    },
}
P1335_RECIPE = {
    "summary": (
        "capture-time token filters (issue1335_extract_store.py::build_items @ "
        "377f824b5f): completion >= 4 tokens (DIALOGUE_MIN_TOKENS), context >= 8 "
        "tokens within the 512-token cap window (CONTEXT_MIN_TOKENS / "
        "CONTEXT_CAP_TOKENS), total row <= 2048 tokens (ROW_MAX_TOKENS); "
        "question-level loader dropped 0/5000"
    ),
}


def q_idx_of(row_id: str) -> int:
    """``r0_qa_full:q03473`` -> 3473."""
    return int(row_id.rsplit(":q", 1)[1])


def recompute_825_dropset(track_s: Path, tokenizer) -> set[int]:
    """#825's realized zero-width-span drop set (prompt_idx values), recomputed
    offline via the byte-identical render path. Never touches row text."""
    dropped: set[int] = set()
    with track_s.open(encoding="utf-8") as fh:
        i = -1
        for line in fh:
            if not line.strip():
                continue
            i += 1
            row = json.loads(line)
            assert row["prompt_idx"] == i, (row["prompt_idx"], i)
            conv = ext825.to_single_turn(row)
            r = ext825.render_conv(conv, tokenizer, "naturalistic")
            if any(s >= e for (s, e) in r.spans.values()):
                dropped.add(i)
    assert i == 4999, i
    return dropped


def recompute_1335_drops(records: list[dict]) -> tuple[set[int], dict[str, set[int]]]:
    """(kept q_idx set, per-reason dropped q_idx sets) under the committed r0
    predicates — same ordered short-circuit as build_items; cross-checked by
    the caller against build_items counters + the store row_id set."""
    kept: set[int] = set()
    drops: dict[str, set[int]] = {
        "short_dialogue": set(),
        "short_context": set(),
        "row_too_long": set(),
    }
    for r in records:
        qi = q_idx_of(r["row_id"])
        n_prompt = len(r["prompt_token_ids"])
        n_comp = len(r["completion_token_ids"])
        if n_comp < r1335.DIALOGUE_MIN_TOKENS:
            drops["short_dialogue"].add(qi)
            continue
        c_lo = max(0, n_prompt - r1335.CONTEXT_CAP_TOKENS)
        if n_prompt - c_lo < r1335.CONTEXT_MIN_TOKENS:
            drops["short_context"].add(qi)
            continue
        if n_prompt + n_comp > r1335.ROW_MAX_TOKENS:
            drops["row_too_long"].add(qi)
            continue
        kept.add(qi)
    return kept, drops


def stage_pinned(args, mk: str) -> list[Path]:
    """Stage the r0 (model) store's .pt shards from the Hub AT THE PINNED REV
    (f1335.ensure_store_local downloads unpinned main — this mirrors it with
    revision=HF_REV). Returns the downloaded paths (for post-fit release)."""
    from huggingface_hub import hf_hub_download

    store_dir = f1335.store_root(args) / SLUG / mk
    sidecars = sorted(store_dir.glob(f"{mk}_shard*.json"))
    assert sidecars, f"no local sidecars for {SLUG}/{mk}"
    missing = [
        f"{sc.name[: -len('.json')]}.pt"
        for sc in sidecars
        if not (store_dir / f"{sc.name[: -len('.json')]}.pt").exists()
    ]
    staged: list[Path] = []
    if not missing:
        return staged
    prefix = f"{r1335.HF_PREFIX}/analysis_tensors/store_{SLUG}_{mk}"
    import os

    with tempfile.TemporaryDirectory(dir=store_dir, prefix=".hfstage_") as td:
        for name in missing:
            got = hf_hub_download(
                r1335.HF_DATA_REPO,
                f"{prefix}/{name}",
                repo_type="dataset",
                revision=HF_REV,
                local_dir=td,
            )
            dest = store_dir / name
            os.replace(got, dest)
            staged.append(dest)
    print(f"[r0-filters] staged {len(staged)} shards for {SLUG}/{mk} @ {HF_REV[:12]}")
    return staged


def load_r0_l19(args, mk: str) -> dict:
    """Slim layer-19 (ctx arm) store load: per-shard slice of x_spanmean + y
    only (~130 MB fp32 vs ~9 GB for the full-array loader), same shard order,
    same fingerprint gate, same bf16->fp32 conversion as f1335.load_rung_store."""
    store_dir = f1335.store_root(args) / SLUG / mk
    shards = sorted(store_dir.glob(f"{mk}_shard*.pt"))
    assert shards, f"no {mk} shards under {store_dir}"
    rows, groups, xs, ys = [], [], [], []
    for sp in shards:
        side = json.loads(sp.with_suffix("").with_suffix(".json").read_text())
        assert r1335.fingerprint_matches(side, SLUG, require_sha=False), (
            f"stale store shard {sp}: render-config fingerprint mismatch (c24)"
        )
        payload = torch.load(sp, map_location="cpu", weights_only=False)
        rows.extend(payload["row_ids"])
        groups.extend(payload["group_ids"])
        xs.append(payload["arrays"]["x_spanmean"][:, [L], :].float().numpy().astype(np.float32))
        ys.append(payload["arrays"]["y"][:, [L], :].float().numpy().astype(np.float32))
        del payload
    X = np.concatenate(xs, axis=0)
    Y = np.concatenate(ys, axis=0)
    assert X.shape[0] == len(rows) == Y.shape[0], (X.shape, len(rows), Y.shape)
    return {
        "row_ids": np.asarray(rows),
        "group_ids": np.asarray(groups),
        "X": X,
        "Y": Y,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=Path("data/issue_1335"))
    ap.add_argument("--store-dir", type=Path, default=None)
    ap.add_argument("--out-dir", type=Path, default=Path("eval_results/issue_1335"))
    ap.add_argument("--gen-dir", type=Path, default=Path("/tmp/i1335_gen"))
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    results: dict = {
        "metadata": common.metadata(SCRIPT, args.seed, 0, extra={"issue": 1335}),
        "headline_layer": L,
        "lambda_selection": "inner-group-cv",
        "finding": (
            "row-filter recipes DIFFER: #825 naturalistic S applies no token-count "
            "filters (zero-width-span render drop only, n=4724); #1335 r0 applies "
            "completion>=4 / context>=8 / row<=2048 at capture (n=4589/4599)"
        ),
        "filter_recipes": {"issue825_naturalistic_s": P825_RECIPE, "issue1335_r0": P1335_RECIPE},
        "recipe_applied_to_store_note": (
            "#825's recipe adds no token filters, so applied literally to the "
            "persisted r0 rows it keeps ALL of them (delta 0 by construction); the "
            "recipe difference binds at CAPTURE time. The executable arm below "
            "restricts the store to #825's realized 4,724-row kept set "
            "(restrict_to_825_rowset); the rows #1335's token filters dropped are "
            "not in the store and cannot be refit (rows_unaddable*)."
        ),
        "ops_note": (
            "free analysis on persisted bf16 stores (HF rev "
            f"{HF_REV[:12]}); 4 single-layer L19 ridge fits + 2x5000 offline "
            "renders, VM CPU, 0 GPU-h"
        ),
    }

    # ---------------------------------------------------- #825 row-set recompute
    track_s = args.data_dir / "track_s.jsonl"
    sha = common.sha256_file(track_s)
    assert sha.startswith("d20560b679345a6e"), sha  # #825 round-9 pin (9,036,307 bytes)
    dropsets = {}
    for mk in ("base", "instruct"):
        tok = common.get_tokenizer(r1335.MODEL_IDS[mk])
        dropsets[mk] = recompute_825_dropset(track_s, tok)
        print(f"[r0-filters] #825 dropset ({mk} tokenizer): {len(dropsets[mk])} rows")
    assert dropsets["base"] == dropsets["instruct"], (
        "tokenizer-dependent #825 drop set — cannot pin a single realized row set"
    )
    p825_dropped = dropsets["base"]
    p825_kept = set(range(5000)) - p825_dropped
    assert len(p825_kept) == P825_N_KEPT, (
        f"recomputed #825 kept n={len(p825_kept)} != committed {P825_N_KEPT}"
    )
    results["p825_rowset_recompute"] = {
        "n_rows": 5000,
        "n_dropped_zero_width": len(p825_dropped),
        "n_kept": len(p825_kept),
        "committed_n": {k: v["n"] for k, v in P825_COMMITTED.items()},
        "tokenizer_agreement_base_vs_instruct": True,
        "track_s_sha256": sha,
    }

    # ------------------------------------------------------------- per-model
    from huggingface_hub import hf_hub_download

    results["per_model"] = {}
    for mk in ("base", "instruct"):
        # 1. gen records at the pinned rev (reuses the /tmp cache when present)
        gen_local = hf_hub_download(
            r1335.HF_DATA_REPO,
            f"{r1335.HF_PREFIX}/raw_completions/qa_full/{mk}_gen.jsonl",
            repo_type="dataset",
            revision=HF_REV,
            local_dir=str(args.gen_dir),
        )
        records = []
        with open(gen_local, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    records.append(json.loads(line))
        assert len(records) == 5000, len(records)

        # 2. committed drop predicates: build_items (fidelity) + per-row reasons
        items, counters = ext1335.build_items(SLUG, records)
        kept_1335, drops_1335 = recompute_1335_drops(records)
        assert len(kept_1335) == counters["kept"], (len(kept_1335), counters["kept"])
        for reason in ("short_dialogue", "short_context", "row_too_long"):
            assert len(drops_1335[reason]) == counters[f"dropped_{reason}"], (
                reason,
                len(drops_1335[reason]),
                counters,
            )
        assert kept_1335 == {q_idx_of(it["row_id"]) for it in items}

        # 3. stage + slim-load the persisted store (pinned rev)
        staged = stage_pinned(args, mk)
        store = load_r0_l19(args, mk)
        store_q = np.asarray([q_idx_of(rid) for rid in store["row_ids"]])
        assert set(store_q.tolist()) == kept_1335, "store row set != recomputed kept set"

        # 4. reproduction anchor (full store rows -> committed value, 1e-6)
        committed = comp.committed_l19(args.out_dir, f"{SLUG}__{mk}__ctx")
        anchor = comp.fit_l19(store["X"], store["Y"], store["group_ids"], n_boot=0, seed=args.seed)
        residual = abs(anchor["r2"] - committed)
        assert residual < 1e-6, (anchor["r2"], committed)
        print(f"[r0-filters] {mk}: anchor {anchor['r2']:.6f} (committed {committed:.6f})")

        # 5. executable filtered arm: restrict to #825's realized kept row set
        keep = np.asarray([qi in p825_kept for qi in store_q])
        filt = comp.fit_l19(
            store["X"][keep],
            store["Y"][keep],
            np.asarray(store["group_ids"])[keep],
            n_boot=0,
            seed=args.seed,
        )
        n_removed = int((~keep).sum())

        # 6. one-sided accounting: #825-kept rows absent from the store
        unaddable = p825_kept - kept_1335
        reason_split = {r: len(unaddable & s) for r, s in drops_1335.items()}
        assert sum(reason_split.values()) == len(unaddable)

        results["per_model"][mk] = {
            "committed_r2_l19": committed,
            "n_committed": int(store["X"].shape[0]),
            "repro_anchor": {"r2": anchor["r2"], "abs_residual": residual},
            "restrict_to_825_rowset": {
                "r2": filt["r2"],
                "n": filt["n"],
                "delta_vs_committed": filt["r2"] - committed,
            },
            "rows_removed_from_store_825_dropped": n_removed,
            "rows_added": 0,
            "rows_unaddable_825kept_but_token_filtered": len(unaddable),
            "unaddable_reason_split": reason_split,
            "capture_drop_counters": {
                k: v for k, v in counters.items() if k.startswith(("dropped", "kept", "records"))
            },
            "p825_committed_reference": P825_COMMITTED[mk],
        }
        print(
            f"[r0-filters] {mk}: committed {committed:.4f} (n={store['X'].shape[0]}) -> "
            f"restrict-to-#825-rowset {filt['r2']:.4f} (n={filt['n']}, removed {n_removed}, "
            f"unaddable {len(unaddable)})"
        )

        for pt in staged:
            pt.unlink()
        del store, records, items

    out_path = args.out_dir / "refits_r0_filters.json"
    out_path.write_text(json.dumps(results, indent=2, default=float))
    print(f"[r0-filters] wrote {out_path}")


if __name__ == "__main__":
    main()
