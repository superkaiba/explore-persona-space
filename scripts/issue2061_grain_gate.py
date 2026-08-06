"""P0 grain gate for task #2061 (plan v11 delta (d); §7 mitigation item 3).

The fail-loud recurrence-prevention phase: BEFORE any GPU spend, count the
realized rows per (stage, combo) cell at the CONSUMED grain — the production
loader's own enumeration/resolution over the REGISTERED 50-store set (35 v2
stores + the 15 wave-1 concat sources; plan §Design "Registered consumption
grain") — and FAIL LOUD when any cell's realized regime contradicts the
declared expectation (v11: ALL cells primal, per-fold n_train > d_in = 4096)
or any boundary/schema assert fails. NO static per-cell row-count table is
load-bearing anywhere: the §7 measured table is evidence, this gate is the
mechanism.

Reads ONLY the per-shard JSON sidecars (KB-scale; the #1336 producer writes
one beside every `.pt` shard) — realized n, conv_ids, capture-convention
keys, model_id, and per-row prompt_shas all live there — so the gate is
VM-local and cheap (~740 sidecar fetches through the shared #833 recipe).
Per-fold n_train is EXACT: computed through the production fold constructor
(`issue2061_turnstore.group_fold_ids`, K=5, seed 0).

Asserts (plan §Design "Registered consumption grain" + "Cross-wave schema
parity"; ported from the parent's `load_bundle_concat` contract):
  - boundary: per concat cell, every wave-1 prompt_idx < boundary <= every
    v2-extension prompt_idx (`V2_CONCAT_BOUNDARY`, a per-corpus dict);
  - conv-id disjointness + no duplicates over the consumed union;
  - capture-convention keys PRESENCE-CONDITIONAL (present => "committed" +
    offset_override null; absent => pre-D2-era store, ACCEPTED and logged
    in the manifest — v13 delta a1-bis);
  - cross-wave/cross-shard model_id equality per (stage, combo) cell;
  - per-row prompt-sha join across every store of one corpus that carries
    `prompt_shas` (all v2-era stores): stores must agree on every shared
    conv_id with ZERO mismatches tolerated (text drift is corruption).

Outputs `eval_results/issue_2061/grain_gate/grain_manifest.json` (written
even on FAIL — the manifest IS the gate report) and exits:
  0 = PASS; 2 = boundary/schema/sha assert failure; 3 = regime contradiction
(designed-halt distinct rc's — `.claude/rules/pod-side-reporting.md`).

Smoke-acceptance mode (`--accept-r2-dir`, dispatch delta (e)): checks every
P2 JSONL in a dir against the manifest — convention == "primal", the v13
λ grid + edge-audit fields present, realized n matches the manifest cell.

Usage:
  uv run python scripts/issue2061_grain_gate.py --all-cells \\
      --output eval_results/issue_2061/grain_gate/grain_manifest.json
  uv run python scripts/issue2061_grain_gate.py --stage base --stage sft \\
      --render chat --corpus gsm8k_train_full --output <smoke manifest>
  uv run python scripts/issue2061_grain_gate.py --accept-r2-dir <r2 dir> \\
      --manifest <grain_manifest.json>
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()  # shared-VM thread caps (#847) must bind BEFORE torch/numpy import

# Sibling-script import (bare module name via the script-dir sys.path insert —
# works in script mode AND under the tests' `sys.path.insert(scripts)` import).
_SCRIPT_DIR = Path(__file__).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import issue2061_turnstore as ts  # noqa: E402

D_IN = 4096
K_FOLDS = 5
FOLD_SEED = 0
LAYER = 29
# The declared regime expectation (plan §7 mitigation item 3 / v11 item 2):
# at the registered concat grain EVERY (stage, combo) cell is primal
# (min per-fold n_train > d_in). A contradiction FAILS the dispatch loud —
# the registered mechanical disposition (§7) then owns the response.
EXPECTED_CONVENTION = "primal"

# Corpus n_target registry — PORTED from the parent's corpus registry
# (`issue-1336-fullcorpora` `common.py` V2_CORPORA[...]["n_target"]) with the
# same port-not-import provenance as the concat constants. INFORMATIONAL
# ONLY: feeds the manifest's keep-rate report field, never a gate criterion
# (realized n_built for if/uf/sft runs slightly under target, so the reported
# keep-rate is a lower bound there).
CORPUS_N_TARGET = {
    "lmsys23k": 23_000,
    "gsm8k_train_full": 7473,
    "gsm8k_test1319": 1319,
    "math7500": 7500,
    "if11k": 11_000,
    "uf11k": 11_000,
    "sft11k": 11_000,
}

EXIT_ASSERT_FAILURE = 2
EXIT_REGIME_CONTRADICTION = 3


def _sorted_sidecars(sidecars: list[dict], tree: str) -> list[dict]:
    """Sidecars in shard-index order; fail-loud on duplicate shard indexes."""
    idx = [int(s.get("shard_index", -1)) for s in sidecars]
    if len(set(idx)) != len(idx):
        raise ValueError(f"{tree}: duplicate shard_index values {sorted(idx)}")
    return [s for _, s in sorted(zip(idx, sidecars), key=lambda p: p[0])]


def _tree_conv_ids(store: dict) -> list[str]:
    """Concatenated conv_ids of one store, in shard-index order, with per-shard checks."""
    tree = store["tree_path"]
    conv_ids: list[str] = []
    for sc in _sorted_sidecars(store["sidecars"], tree):
        ids = [str(c) for c in sc.get("conv_ids", [])]
        n = sc.get("n_conversations")
        if n != len(ids):
            raise ValueError(
                f"{tree} shard {sc.get('shard_index')}: n_conversations={n} != "
                f"{len(ids)} conv_ids — sidecar self-inconsistent."
            )
        shas = sc.get("prompt_shas")
        if shas is not None and len(shas) != len(ids):
            raise ValueError(
                f"{tree} shard {sc.get('shard_index')}: {len(shas)} prompt_shas != "
                f"{len(ids)} conv_ids."
            )
        conv_ids.extend(ids)
    return conv_ids


def _tree_convention_states(store: dict) -> list[str]:
    """Per-shard presence-conditional convention states (shared loader logic)."""
    tree = store["tree_path"]
    return [
        ts.sidecar_convention_state(sc, src=f"{tree} shard {sc.get('shard_index')}")
        for sc in store["sidecars"]
    ]


def _tree_schema_check(store: dict, failures: list[str]) -> None:
    """Cross-wave schema-parity hard asserts, sidecar grain (plan §Design)."""
    tree = store["tree_path"]
    for sc in store["sidecars"]:
        for key in ("expected_layers", "expected_hidden", "model_id"):
            if key not in sc:
                failures.append(f"{tree} shard {sc.get('shard_index')}: sidecar missing {key}")
        shapes = sc.get("shapes", {})
        for arr in ("slots", "profiles"):
            per_rec = shapes.get(arr)
            if not per_rec:
                failures.append(f"{tree} shard {sc.get('shard_index')}: shapes.{arr} missing")
                continue
            want = [2, int(sc.get("expected_layers", 0)), int(sc.get("expected_hidden", 0))]
            bad = [s for s in per_rec if list(s) != want]
            if bad:
                failures.append(
                    f"{tree} shard {sc.get('shard_index')}: {len(bad)} {arr} record(s) with "
                    f"shape != {want} (e.g. {bad[0]})"
                )


def gate_cells(
    cells: list[dict],
    *,
    d_in: int = D_IN,
    k_folds: int = K_FOLDS,
    fold_seed: int = FOLD_SEED,
    expected_convention: str = EXPECTED_CONVENTION,
) -> dict:
    """The pure gate core (hub-free; tests drive it with in-memory fixtures).

    Each cell dict: {"stage", "render", "corpus",
                     "stores": [{"tree_path", "sidecars": [dict, ...]}, ...]}
    with stores in the canonical concat order (wave-1 source first).

    Returns the grain manifest dict: per-cell realized n / EXACT per-fold
    n_train (production fold constructor) / selected convention / keep-rate,
    the pre-D2 convention log, the per-corpus prompt-sha join, and the gate
    verdict with its failure lists.
    """
    failures: list[str] = []
    regime_contradictions: list[str] = []
    cell_rows: list[dict] = []
    pre_d2_stores: set[str] = set()
    # corpus -> conv_id -> (sha, first store) for the cross-store sha join.
    sha_seen: dict[str, dict[str, tuple[str, str]]] = {}
    sha_stats: dict[str, dict] = {}

    for cell in cells:
        stage, render, corpus = cell["stage"], cell["render"], cell["corpus"]
        label = f"{stage}/{render}/{corpus}"
        stores = cell["stores"]
        parts: list[list[str]] = []
        conventions: set[str] = set()
        model_ids: set[str] = set()
        try:
            for store in stores:
                _tree_schema_check(store, failures)
                ids = _tree_conv_ids(store)
                parts.append(ids)
                states = set(_tree_convention_states(store))
                conventions |= states
                if "pre-D2-absent" in states:
                    pre_d2_stores.add(Path(store["tree_path"]).name)
                model_ids |= {str(sc.get("model_id")) for sc in store["sidecars"]}
                # Per-corpus cross-store prompt-sha join (zero mismatches).
                stats = sha_stats.setdefault(
                    corpus, {"n_stores": 0, "n_rows_checked": 0, "n_mismatches": 0}
                )
                seen = sha_seen.setdefault(corpus, {})
                store_has_shas = False
                for sc in _sorted_sidecars(store["sidecars"], store["tree_path"]):
                    shas = sc.get("prompt_shas")
                    if shas is None:
                        continue
                    store_has_shas = True
                    for cid, sha in zip(sc.get("conv_ids", []), shas):
                        cid = str(cid)
                        prev = seen.get(cid)
                        if prev is None:
                            seen[cid] = (str(sha), store["tree_path"])
                        else:
                            stats["n_rows_checked"] += 1
                            if prev[0] != str(sha):
                                stats["n_mismatches"] += 1
                                failures.append(
                                    f"prompt-sha JOIN mismatch for {corpus} conv {cid}: "
                                    f"{store['tree_path']} disagrees with {prev[1]} — "
                                    "text drift, ZERO mismatches tolerated."
                                )
                if store_has_shas:
                    stats["n_stores"] += 1
            boundary = None
            if len(stores) == 2:
                boundary = ts.assert_concat_boundary(parts[0], parts[1], corpus)
            if len(model_ids) > 1:
                raise ValueError(f"{label}: cross-store model_id mismatch {sorted(model_ids)}")
            conv_ids = [c for p in parts for c in p]
            if len(set(conv_ids)) != len(conv_ids):
                raise ValueError(
                    f"{label}: duplicate conv_ids over the consumed union "
                    f"(n={len(conv_ids)}, unique={len(set(conv_ids))})"
                )
        except (ValueError, KeyError) as e:
            failures.append(f"{label}: {e}")
            continue

        n = len(conv_ids)
        folds = ts.group_fold_ids(conv_ids, n_folds=k_folds, seed=fold_seed)
        import numpy as np

        counts = np.bincount(folds, minlength=k_folds)
        per_fold_n_train = [int(n - c) for c in counts]
        min_n_tr = min(per_fold_n_train)
        selected = "primal" if min_n_tr > d_in else "gram-dual"
        if selected != expected_convention:
            regime_contradictions.append(
                f"{label}: realized regime {selected} (min per-fold n_train={min_n_tr} vs "
                f"d_in={d_in}) contradicts the declared expectation "
                f"{expected_convention!r} (plan §7 measured table)."
            )
        n_target = CORPUS_N_TARGET.get(corpus)
        cell_rows.append(
            {
                "stage": stage,
                "render": render,
                "corpus": corpus,
                "stores": [Path(s["tree_path"]).name for s in stores],
                "concat": len(stores) == 2,
                "boundary": boundary,
                "realized_n": n,
                "per_fold_n_train": per_fold_n_train,
                "min_per_fold_n_train": min_n_tr,
                "d_in": d_in,
                "selected_convention": selected,
                "conventions": sorted(conventions),
                "model_id": sorted(model_ids)[0] if model_ids else None,
                "n_target": n_target,
                "keep_rate_vs_n_target": (n / n_target) if n_target else None,
            }
        )
        print(
            f"[grain] {label}: n={n} min_per_fold_n_train={min_n_tr} "
            f"convention={selected}"
            + (f" (concat, boundary={boundary} PASS)" if len(stores) == 2 else ""),
            flush=True,
        )

    verdict = "PASS" if not failures and not regime_contradictions else "FAIL"
    return {
        "format": "issue2061-grain-manifest-v1",
        "d_in": d_in,
        "k_folds": k_folds,
        "fold_seed": fold_seed,
        "expected_convention": expected_convention,
        "n_cells": len(cell_rows),
        "cells": cell_rows,
        "pre_d2_stores": sorted(pre_d2_stores),
        "sha_join": sha_stats,
        "assert_failures": failures,
        "regime_contradictions": regime_contradictions,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Hub side: fetch every consumed store's sidecars (KB-scale) via the #833
# scoped-listing + bounded-pool recipe.
# ---------------------------------------------------------------------------
def _fetch_store_sidecars(tree_path: str, revision: str | None, max_workers: int) -> list[dict]:
    """Download + parse one store tree's JSON sidecars (derived from the .pt list)."""
    from huggingface_hub import hf_hub_download

    from explore_persona_space.orchestrate.hub import retry_transient

    import issue2061_sae_encode as enc

    rels = [
        rel.removesuffix(".pt") + ".json"
        for rel in enc.hub_shard_files(tree_path, revision=revision)
    ]

    def _one(rel: str) -> dict:
        p = retry_transient(
            lambda: hf_hub_download(
                repo_id=enc.DATA_REPO, filename=rel, repo_type="dataset", revision=revision
            ),
            what=f"issue2061 grain-gate sidecar {rel}",
        )
        return json.loads(Path(p).read_text())

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(pool.map(_one, rels))


def collect_cells_from_hub(
    stages: list[str] | None,
    render: str | None,
    corpus: list[str] | None,
    revision: str | None,
    max_workers: int,
) -> list[dict]:
    """Enumerate the registered consumption grain and fetch its sidecars.

    Resolution goes through the PRODUCTION enumeration/resolution code
    (`issue2061_sae_encode._stage_render_corpus_turnstores` +
    `resolve_turnstore_trees`) so the gate counts exactly what P1/P2/P3
    consume — the whole point of §7 mitigation item 3.
    """
    import issue2061_sae_encode as enc

    v2_stores = enc._stage_render_corpus_turnstores(revision=revision, generation="v2")
    targets = [
        t
        for t in v2_stores
        if (not stages or t["stage"] in stages)
        and (render is None or t["render"] == render)
        and (not corpus or t["corpus"] in corpus)
    ]
    if not targets:
        raise SystemExit(
            f"[error] no registered cells match filters (stages={stages} render={render} "
            f"corpus={corpus})"
        )
    v1_stores = None
    if any(t["corpus"] in ts.V2_CONCAT_SOURCES for t in targets):
        v1_stores = enc._stage_render_corpus_turnstores(revision=revision, generation="v1")
    cells: list[dict] = []
    for t in targets:
        trees = enc.resolve_turnstore_trees(
            t["stage"],
            t["render"],
            t["corpus"],
            revision=revision,
            v2_stores=v2_stores,
            v1_stores=v1_stores,
        )
        stores = [
            {"tree_path": tree, "sidecars": _fetch_store_sidecars(tree, revision, max_workers)}
            for tree in trees
        ]
        cells.append(
            {"stage": t["stage"], "render": t["render"], "corpus": t["corpus"], "stores": stores}
        )
    return cells


# ---------------------------------------------------------------------------
# Smoke-acceptance mode (dispatch delta (e)): P2 outputs vs the manifest.
# ---------------------------------------------------------------------------
def check_smoke_acceptance(
    r2_dir: Path,
    manifest_path: Path,
    *,
    expect_grid_lo: float = -3.0,
    expect_grid_hi: float = 8.0,
    layer: int = LAYER,
) -> list[str]:
    """Concat-grain smoke acceptance (plan delta (e) items (ii)/(iii)/(v)).

    Per P2 JSONL in `r2_dir`: convention == "primal"; the λ-edge audit
    fields present; the realized grid == the v13 registration (modulo any
    recorded edge extensions — audited, not forbidden); realized n equals
    the grain manifest's cell row. Returns the violation list (empty = PASS).
    """
    manifest = json.loads(Path(manifest_path).read_text())
    by_cell = {(c["stage"], c["render"], c["corpus"]): c for c in manifest["cells"]}
    violations: list[str] = []
    paths = sorted(Path(r2_dir).glob(f"*_L{layer}.jsonl"))
    if not paths:
        return [f"no P2 JSONL outputs under {r2_dir}"]
    audit_fields = (
        "convention",
        "n_at_low_edge",
        "n_at_high_edge",
        "lambda_grid_log10_lo",
        "lambda_grid_log10_hi",
        "n_lambda",
        "regularization_limited",
    )
    for path in paths:
        stage, render, corpus, arm = ts.parse_r2_stem(path.stem, layer)
        with path.open() as f:
            row = json.loads(f.readline())
        missing = [k for k in audit_fields if k not in row]
        if missing:
            violations.append(f"{path.name}: missing λ-edge audit fields {missing}")
            continue
        if row["convention"] != "primal":
            violations.append(
                f"{path.name}: convention={row['convention']!r} != 'primal' at the concat "
                "grain (n_tr <= d_in somewhere — the retired extension-slice regime?)"
            )
        base_lo = row["lambda_grid_log10_lo"] + row.get("n_ext_low", 0) * 1.0
        base_hi = row["lambda_grid_log10_hi"] - row.get("n_ext_high", 0) * 1.0
        if base_lo != expect_grid_lo or base_hi != expect_grid_hi:
            violations.append(
                f"{path.name}: base grid 1e{base_lo:g}..1e{base_hi:g} != the v13 "
                f"registration 1e{expect_grid_lo:g}..1e{expect_grid_hi:g}"
            )
        cell = by_cell.get((stage, render, corpus))
        if cell is None:
            violations.append(f"{path.name}: no grain-manifest cell for {stage}/{render}/{corpus}")
        elif int(row["n_test_total"]) != int(cell["realized_n"]):
            violations.append(
                f"{path.name}: realized n={row['n_test_total']} != grain manifest "
                f"{cell['realized_n']} for {stage}/{render}/{corpus}"
            )
    return violations


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", action="append", default=None, help="repeatable stage filter")
    parser.add_argument("--render", type=str, default=None)
    parser.add_argument(
        "--corpus",
        action="append",
        default=None,
        help="repeatable corpus filter (the two-grain smoke passes one concat "
        "corpus AND one plain-v2 corpus — crash-fix 2026-08-06)",
    )
    parser.add_argument("--all-cells", action="store_true")
    parser.add_argument(
        "--expect-n-cells",
        type=int,
        default=None,
        help="fail loud when the gated cell count differs (production: 35 (stage, combo) cells)",
    )
    parser.add_argument("--data-revision", type=str, default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("eval_results/issue_2061/grain_gate/grain_manifest.json"),
    )
    parser.add_argument("--d-in", type=int, default=D_IN)
    parser.add_argument("--max-workers", type=int, default=6, help="sidecar fetch pool (#833)")
    parser.add_argument(
        "--accept-r2-dir",
        type=Path,
        default=None,
        help="SMOKE-ACCEPTANCE mode: check P2 JSONLs in this dir against --manifest "
        "(convention=primal, v13 grid + λ-edge audit fields, realized n) and exit.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="grain manifest to check against in --accept-r2-dir mode",
    )
    args = parser.parse_args()

    if args.accept_r2_dir is not None:
        manifest = args.manifest or args.output
        violations = check_smoke_acceptance(args.accept_r2_dir, manifest)
        if violations:
            for v in violations:
                print(f"[accept-FAIL] {v}")
            return EXIT_ASSERT_FAILURE
        print(f"[accept] PASS: {args.accept_r2_dir} conforms to {manifest}")
        return 0

    if not args.all_cells and not (args.stage or args.render or args.corpus):
        print("[error] pass --all-cells or at least one of --stage/--render/--corpus")
        return 1

    # Pin ONE data-repo commit for the whole gate pass (crash-fix 2026-08-06;
    # same rationale as sae_encode.main — an unpinned revision re-resolves
    # `main` per hf_hub_download on the constantly-moving shared repo). The
    # manifest meta then records the RESOLVED sha, not None.
    import issue2061_hub_io as hio

    args.data_revision = hio.resolve_data_repo_revision(args.data_revision)
    print(f"[grain] data revision pinned: {args.data_revision}")

    cells = collect_cells_from_hub(
        args.stage, args.render, args.corpus, args.data_revision, args.max_workers
    )
    if args.expect_n_cells is not None and len(cells) != args.expect_n_cells:
        print(
            f"[error] enumerated {len(cells)} (stage, combo) cell(s) != --expect-n-cells "
            f"{args.expect_n_cells} (production: 35 = 5 stages x 7 combos)"
        )
        return EXIT_ASSERT_FAILURE

    manifest = gate_cells(cells, d_in=args.d_in)
    manifest["meta"] = {
        "git_commit": ts._git_commit_sha(),
        "created_unix": time.time(),
        "data_revision": args.data_revision,
        "n_requested_cells": len(cells),
        "argv": sys.argv[1:],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"[grain] manifest -> {args.output} (verdict: {manifest['verdict']})")

    if manifest["assert_failures"]:
        for f_ in manifest["assert_failures"]:
            print(f"[grain-FAIL] {f_}")
        return EXIT_ASSERT_FAILURE
    if manifest["regime_contradictions"]:
        for c in manifest["regime_contradictions"]:
            print(f"[grain-FAIL] {c}")
        print(
            "[grain-FAIL] regime contradiction — dispatch STOPS (plan §7 mitigation item 3 "
            "registered disposition: re-evaluate the §Resources wall model at the gate-"
            "measured n_tr before any P3 launch; over-envelope => re-park at plan_pending)."
        )
        return EXIT_REGIME_CONTRADICTION
    print(f"[grain] PASS: {manifest['n_cells']} cell(s) all {EXPECTED_CONVENTION}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
