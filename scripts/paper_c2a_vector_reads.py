"""Vector-level reads: per-element retrieval, and the shared-subspace control.

Question this answers
---------------------
Two things the banked per-pair scalars could not settle.

  RETRIEVAL. The section's claim is really a claim about whether a distinction
  is RECOVERABLE from the map's prediction, which is a retrieval question, not
  a reconstruction one. Retrieval is banked for four of the six elements but was
  never reported per element in the paper. This recomputes it from the shift
  vectors so the map and the copy baseline are scored identically, and
  cross-checks against the banked numbers where both exist.

  SHARED SUBSPACE. The matched-magnitude round found elements sit off a common
  cosine-versus-magnitude curve (persona above, topic and format below). That
  contrast is now load-bearing, and the strongest remaining threat to it is
  that a few generic response-shape directions carry most of every element's
  shift, so the ordering would reflect how much of each element's shift lands
  in that shared subspace rather than anything element-specific. Answer length
  was already ruled out at the scalar level (pooled Spearman 0.078). This is
  the vector-level version of the same control.

Shared-subspace design
----------------------
For each element, the nuisance subspace is the top-k principal directions of
the pooled OBSERVED shifts of the OTHER elements, so an element never defines
the subspace it is then tested against. That mirrors the leave-one-element-out
trend fit in the matched-magnitude round and avoids the circularity of removing
a subspace an element helped build. Both the observed and the predicted shift
are projected out of that subspace, and the cosine recomputed. k is swept over
1, 3 and 5 rather than tuned.

Reported alongside: the fraction of each element's observed shift energy that
lies IN the shared subspace, which is the quantity the competing explanation is
actually about.

Coverage
--------
The precomputed shift vectors exist for the four elements banked in the parent
and floor-failed-reelicitation stores (output format, persona, tone, question
topic). The two pilot elements (answer language, one-word topic) have answer
and context activations on the Hub but NO precomputed prediction tensors, so
they are OUT of scope here and are named as a gap rather than silently omitted.
The load-bearing persona-versus-topic contrast is fully covered.

Alignment is verified, not assumed: the per-pair norms and cosines recomputed
from the tensors must reproduce the banked scalars to float tolerance before
any new read is computed, which pins row order between the tensors and the
per-pair JSONL.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

# #847: thread caps must land BEFORE the numpy/torch imports below — on the
# shared VM, load_dotenv() setdefaults OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS,
# and the BLAS pools freeze at import time.
load_dotenv()

import numpy as np  # noqa: E402
import torch  # noqa: E402

from explore_persona_space.analysis.paired_ci import _quantile_ci  # noqa: E402
from explore_persona_space.task_workflow import repo_root  # noqa: E402

MAP_ARM = "arm_779ce"
COPY_ARM = "arm_iddelta"
PRIMARY_CLASS = {
    "format": "swap",
    "register": "swap",
    "persona": "swap",
    "query_content": "query_content",
}
# (label, store, axis) -- pilot elements have no prediction tensors, see docstring.
ELEMENTS = [
    ("Output format", "parent", "format"),
    ("Persona", "ffr", "persona"),
    ("Tone", "parent", "register"),
    ("Question topic", "parent", "query_content"),
]
UNCOVERED = ["Answer language", "One-word topic"]


def _rows(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _get(row: dict, field: str, arm: str) -> float:
    flat = f"{field}_{arm}"
    return float(row[flat]) if flat in row else float(row[field][arm])


def _load(path: Path) -> tuple[np.ndarray, list[str]]:
    """Return the shift matrix and its pair-id index from a banked delta store."""
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict) or "tensor" not in obj or "pair_ids" not in obj:
        raise ValueError(
            f"{path.name}: expected a dict with 'tensor' and 'pair_ids', got "
            f"{sorted(obj)[:8] if isinstance(obj, dict) else type(obj).__name__}"
        )
    t = obj["tensor"]
    arr = np.asarray(t.float().numpy() if hasattr(t, "float") else t, dtype=np.float64)
    return arr, [str(x) for x in obj["pair_ids"]]


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    na = np.linalg.norm(a, axis=1)
    nb = np.linalg.norm(b, axis=1)
    denom = na * nb
    out = np.full(a.shape[0], np.nan)
    ok = denom > 0
    out[ok] = (a[ok] * b[ok]).sum(axis=1) / denom[ok]
    return out


def _retrieval(pred: np.ndarray, obs: np.ndarray) -> dict:
    """Rank of each pair's own observed shift among the element's pool, by cosine."""
    n = pred.shape[0]
    pn = pred / np.clip(np.linalg.norm(pred, axis=1, keepdims=True), 1e-12, None)
    on = obs / np.clip(np.linalg.norm(obs, axis=1, keepdims=True), 1e-12, None)
    sim = pn @ on.T
    # rank of the diagonal within each row, 1 = best
    ranks = (sim > sim[np.arange(n), np.arange(n)][:, None]).sum(axis=1) + 1
    return {
        "n": int(n),
        "acc_at_1": float((ranks == 1).mean()),
        "acc_at_5": float((ranks <= 5).mean()),
        "median_rank": float(np.median(ranks)),
        "mrr": float((1.0 / ranks).mean()),
        "chance_at_1": 1.0 / n,
    }


def _project_out(x: np.ndarray, basis: np.ndarray) -> np.ndarray:
    """Remove the span of ``basis`` (rows orthonormal) from each row of x."""
    return x - (x @ basis.T) @ basis


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--tensor-root",
        type=Path,
        default=Path("/mnt/eps-data/thomasjiralerspong/issue2564_vecreads"),
    )
    ap.add_argument("--k-sweep", type=int, nargs="+", default=[1, 3, 5])
    ap.add_argument("--n-boot", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=25642)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    root = repo_root()
    src = root / "eval_results/issue_2564"
    base = args.tensor_root / "issue2564_minpair/analysis_tensors"
    stores = {
        "parent": {
            "rows": _rows(src / "perpair.jsonl"),
            "dir": base / "predictions",
        },
        "ffr": {
            "rows": _rows(src / "floor-failed-reelicitation/perpair_ffr.jsonl"),
            "dir": base / "floor_failed_reelicitation/predictions",
        },
    }
    for name, store in stores.items():
        obs, ids_obs = _load(store["dir"] / "delta_obs_tail_L19.pt")
        pmap, ids_map = _load(store["dir"] / f"delta_pred_{MAP_ARM}.pt")
        pcopy, ids_copy = _load(store["dir"] / f"delta_pred_{COPY_ARM}.pt")
        if not (ids_obs == ids_map == ids_copy):
            raise ValueError(f"{name}: the three delta stores carry different pair-id indexes")
        # Align tensor rows to the per-pair JSONL BY PAIR ID, never by position.
        pos = {pid: i for i, pid in enumerate(ids_obs)}
        missing = [r["pair_id"] for r in store["rows"] if r["pair_id"] not in pos]
        if missing:
            raise ValueError(
                f"{name}: {len(missing)} per-pair rows absent from the tensor store "
                f"(first: {missing[:3]})"
            )
        order = np.array([pos[r["pair_id"]] for r in store["rows"]], dtype=np.int64)
        store["obs"], store["map"], store["copy"] = obs[order], pmap[order], pcopy[order]

    # --- alignment verification: reproduce the banked scalars ------------------
    alignment = {}
    for name, store in stores.items():
        rows = store["rows"]
        banked_norm = np.array([r["norm_obs_tail_L19"] for r in rows])
        banked_cos = np.array([_get(r, "cos", MAP_ARM) for r in rows])
        got_norm = np.linalg.norm(store["obs"], axis=1)
        got_cos = _cos(store["map"], store["obs"])
        finite = np.isfinite(banked_cos) & np.isfinite(got_cos)
        max_norm_err = float(np.abs(got_norm - banked_norm).max())
        max_cos_err = float(np.abs(got_cos[finite] - banked_cos[finite]).max())
        # bf16-era captures reproduce to ~1e-3; a row-order break blows this up
        if max_norm_err > 1e-2 * max(1.0, float(banked_norm.max())) or max_cos_err > 1e-2:
            raise AssertionError(
                f"{name}: tensors do not reproduce banked scalars "
                f"(max norm err {max_norm_err:.3e}, max cos err {max_cos_err:.3e}); "
                "row order between the tensor store and the per-pair JSONL is not aligned"
            )
        alignment[name] = {
            "n_rows": len(rows),
            "max_abs_norm_error": max_norm_err,
            "max_abs_cos_error": max_cos_err,
        }

    # --- per-element index sets -----------------------------------------------
    sel_by_element: dict[str, tuple[str, np.ndarray]] = {}
    for label, store_key, axis in ELEMENTS:
        rows = stores[store_key]["rows"]
        idx = [
            i
            for i, r in enumerate(rows)
            if r["axis"] == axis
            and (
                axis not in PRIMARY_CLASS
                or (r["pair_class"] == PRIMARY_CLASS[axis] and r["in_headline_70"])
            )
        ]
        if not idx:
            raise ValueError(f"no pairs selected for {axis}")
        sel_by_element[label] = (store_key, np.array(idx, dtype=np.int64))

    payload: dict = {
        "question": "per-element retrieval, and whether the ordering survives removing shared directions",
        "arms": {"map": MAP_ARM, "copy": COPY_ARM},
        "elements_covered": [label for label, _, _ in ELEMENTS],
        "elements_not_covered": UNCOVERED,
        "coverage_gap_reason": (
            "the pilot store has answer/context activations but no precomputed prediction "
            "tensors, so answer language and the one-word swap need the ridge map re-applied"
        ),
        "alignment_check": alignment,
        "retrieval_comparability_caveat": (
            "Within-element retrieval is NOT comparable across these elements. The pools "
            "differ in kind: for question topic the manipulated variable IS the carrier "
            "(1 value-pair x 66 carriers), so retrieving a pair means telling topics apart, "
            "which is what the shift encodes. For format, persona and tone the carrier is "
            "held fixed within a pair and varies across the pool (12 carriers), so retrieval "
            "additionally demands carrier identity that the manipulation cannot supply, "
            "capping accuracy near 1/12. Read the per-element numbers against "
            "pool_composition, never as a preservation ranking. The same defect applies to "
            "the banked retrieval.per_axis block in minpair_delta.json."
        ),
        "shared_subspace": {
            "definition": "top-k PCs of the pooled observed shifts of the OTHER elements",
            "k_sweep": args.k_sweep,
        },
        "sources": [
            "hf://superkaiba1/explore-persona-space-data/issue2564_minpair/analysis_tensors/",
            "eval_results/issue_2564/perpair.jsonl",
            "eval_results/issue_2564/floor-failed-reelicitation/perpair_ffr.jsonl",
        ],
    }

    # --- reads -----------------------------------------------------------------
    results: dict = {}
    for label, (store_key, idx) in sel_by_element.items():
        store = stores[store_key]
        obs, pmap, pcopy = store["obs"][idx], store["map"][idx], store["copy"][idx]
        sel_rows = [store["rows"][i] for i in idx]
        value_pairs = {(r["value_a"], r["value_b"]) for r in sel_rows}
        carriers = {r["carrier"] for r in sel_rows}
        entry: dict = {
            "n": int(idx.size),
            "pool_composition": {
                "distinct_value_pairs": len(value_pairs),
                "distinct_carriers": len(carriers),
                "carrier_is_the_manipulated_variable": len(value_pairs) == 1
                and len(carriers) == int(idx.size),
            },
            "retrieval": {
                "map": _retrieval(pmap, obs),
                "copy": _retrieval(pcopy, obs),
            },
            "cos_mean": {
                "map": float(np.nanmean(_cos(pmap, obs))),
                "copy": float(np.nanmean(_cos(pcopy, obs))),
            },
        }

        # shared subspace built from the OTHER elements' observed shifts
        others = np.vstack(
            [stores[sk]["obs"][ix] for other, (sk, ix) in sel_by_element.items() if other != label]
        )
        others_centred = others - others.mean(axis=0, keepdims=True)
        _, _, vt = np.linalg.svd(others_centred, full_matrices=False)
        by_k: dict = {}
        for k in args.k_sweep:
            basis = vt[:k]
            energy_in = float(((obs @ basis.T) ** 2).sum() / max((obs**2).sum(), 1e-12))
            cos_map_k = _cos(_project_out(pmap, basis), _project_out(obs, basis))
            cos_copy_k = _cos(_project_out(pcopy, basis), _project_out(obs, basis))
            vals = cos_map_k[np.isfinite(cos_map_k)]
            rng = np.random.default_rng(args.seed + k)
            draws = vals[rng.integers(0, vals.size, size=(args.n_boot, vals.size))].mean(axis=1)
            lo, hi = _quantile_ci(draws)
            by_k[f"k={k}"] = {
                "observed_energy_in_shared_subspace": energy_in,
                "cos_map_after_removal": float(np.nanmean(cos_map_k)),
                "cos_map_ci95": [float(lo), float(hi)],
                "cos_copy_after_removal": float(np.nanmean(cos_copy_k)),
            }
        entry["shared_subspace_removed"] = by_k
        results[label] = entry
    payload["elements"] = results

    out = args.out or (src / "comment2_vector_reads/summary.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=1) + "\n")

    print("alignment (tensors vs banked scalars):")
    for name, a in alignment.items():
        print(
            f"  {name:7s} n={a['n_rows']:4d}  max norm err {a['max_abs_norm_error']:.2e}  "
            f"max cos err {a['max_abs_cos_error']:.2e}"
        )
    hdr = (
        f"\n{'element':16s}{'n':>4s}{'acc@1 map':>11s}{'acc@1 copy':>12s}"
        f"{'MRR map':>9s}{'chance':>8s}{'cos map':>9s}"
    )
    print(hdr)
    print("-" * len(hdr))
    for label, e in results.items():
        rm, rc = e["retrieval"]["map"], e["retrieval"]["copy"]
        print(
            f"{label:16s}{e['n']:4d}{rm['acc_at_1']:11.3f}{rc['acc_at_1']:12.3f}"
            f"{rm['mrr']:9.3f}{rm['chance_at_1']:8.3f}{e['cos_mean']['map']:9.3f}"
        )
    print("\nshared-subspace removal (cosine after projecting out the other elements' top-k PCs):")
    for label, e in results.items():
        parts = []
        for k in args.k_sweep:
            b = e["shared_subspace_removed"][f"k={k}"]
            parts.append(
                f"k={k}: energy {b['observed_energy_in_shared_subspace']:.2f} "
                f"cos {b['cos_map_after_removal']:.3f}"
            )
        print(f"  {label:16s} base {e['cos_mean']['map']:.3f}  " + "  ".join(parts))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
