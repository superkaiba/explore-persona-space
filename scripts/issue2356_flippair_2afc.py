"""Within-family flip-pair 2AFC for issue #2356 (inline free-analysis round).

Question this answers
---------------------
#2356's shipped map-discrimination read is POOL-LEVEL: nearest-neighbour
behavior match over the full held-out answer pool. It never restricted the
candidate set to the SAME base family, so a correct behavior match can come
from a generic "refusal answers cluster here" effect rather than from
discriminating two near-identical rewrites of one harmful base.

This round asks the restricted question directly. Within one base family, take
two intent-preserving rewrites whose realized behavior DIFFERS (one complied,
one refused) and ask whether the frozen label-blind context->answer map's
prediction for each lands closer to its OWN answer than to its sibling's.

Strata
------
A  flip pairs, rewrite vs rewrite   -- the strict twin: same base, differing
   only in framing axis, opposite behavior. PRIMARY.
B  flip pairs, any vs any in family -- base rows included. Secondary and never
   pooled into the headline: base rows are 167/167 refused, so base-vs-rewrite
   pairs are confounded with the is_rewrite indicator (AUROC 0.817 alone).
C  SAME-behavior pairs, rewrite vs rewrite -- the control. If the map scores as
   well here as on stratum A, it is discriminating prompt identity rather than
   behavior, and stratum A's number carries no behavioral claim.

No new fit. The map is the banked label-blind generic-corpus ridge
(analysis_tensors/maps/3a_generic/layer20.npz, v_C -> v_A_greedy, layer 20 =
the out-of-fold selected layer for this condition), applied read-only. Because
nothing is fitted here there is no n_train-vs-d well-posedness question.

Reused rather than reimplemented: ``observed_2afc`` (issue2215_analysis) for
the 2AFC margin convention and ``identity_bias_predict`` semantics from
analysis.mapping_baselines. The #2215 ``null_2afc_cell`` is NOT reused: it is
keyed to that bank's carrier x value-pair grid, which #2356 has no analogue of
(variable-size base families, no carrier axis). The family-blocked derangement
null below is the adaptation, and it preserves the same mean-0.5-by-
construction side-randomization property.
"""

from __future__ import annotations

import argparse
import collections
import json
import importlib.util
import sys
from pathlib import Path
from typing import Any, Callable

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

import numpy as np  # noqa: E402  (after load_dotenv: thread-cap discipline)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

from explore_persona_space.analysis.mapping_baselines import knn_retrieval  # noqa: E402
from issue2215_analysis import observed_2afc  # noqa: E402

# The #2356 map is persisted as an SVD BUNDLE (U/S/Vt + xmu/xsd/ymu), not a
# plain (d, d) matrix, so reconstruction has a specific standardisation
# contract. Import the producing module's OWN `map_predict` rather than
# reimplementing it: a re-derived reconstruction that drops the xmu/xsd
# standardisation or the ymu offset silently produces wrong predictions that
# still have the right SHAPE. The module lives on the issue-2356 branch, so it
# is loaded by path from that worktree.
DEFAULT_FITS_MODULE = REPO / ".claude/worktrees/issue-2356/scripts/issue2356_fits.py"


def load_fits_module(path: Path) -> Any:
    """Import the #2356 fits module by file path (it lives on the issue branch)."""
    if not path.exists():
        raise FileNotFoundError(
            f"fits module not found at {path}; pass --fits-module pointing at the "
            "issue-2356 worktree copy of scripts/issue2356_fits.py"
        )
    spec = importlib.util.spec_from_file_location("issue2356_fits_ext", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load a module spec from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod
    spec.loader.exec_module(mod)
    # Only `map_predict` is consumed here (the identity+bias baseline below is
    # the leave-one-FAMILY-out variant, computed locally); assert exactly what
    # is used so an unrelated refactor of the fits module cannot block a run.
    if not hasattr(mod, "map_predict"):
        raise AttributeError(f"{path} does not export 'map_predict' — schema drift, refusing")
    return mod


LAYER = 20
N_BOOT = 10_000
N_NULL = 10_000
SEED = 2356
REWRITE_AXES = (
    "past_tense",
    "passive_voice",
    "declarative_curiosity",
    "formal_register",
    "nominalization",
    "technical_register",
)


# ── loading ───────────────────────────────────────────────────────────


def load_rows(art: Path) -> dict[str, dict]:
    """sha -> {base_id, axis, label}. Drops rows with a drop_reason or no label."""
    manifest = json.loads((art / "corpus" / "armA_manifest.json").read_text())
    manifest = manifest["rows"] if isinstance(manifest, dict) else manifest
    labels = json.loads((art / "armA" / "labels.json").read_text())["rows"]
    out: dict[str, dict] = {}
    for r in manifest:
        sha = r["prompt_sha"]
        lab = labels.get(sha)
        if lab is None or lab.get("drop_reason") is not None or lab.get("label") is None:
            continue
        out[sha] = {"base_id": int(r["base_id"]), "axis": r["axis"], "label": lab["label"]}
    return out


def load_store(stage: Path, shas: list[str], layer: int) -> tuple[np.ndarray, np.ndarray]:
    """Extract (n, d) v_C and v_A_greedy at ``layer`` from the consolidated shards.

    Consolidated shard keys are ``<sha>__<name>`` and each 2-D value is (L, d);
    the schema is issue2356_fits.ARM_KEYS, read from that module, not re-derived.
    Fail-loud on any sha absent from the staged shards.
    """
    store = stage / "issue2356_refusalpred" / "summary_stores"
    shards = sorted(store.glob("armA.means*.npz"))
    if not shards:
        raise FileNotFoundError(f"no armA.means*.npz under {store}")
    want = set(shas)
    vc: dict[str, np.ndarray] = {}
    va: dict[str, np.ndarray] = {}
    for shard in shards:
        with np.load(shard) as data:
            for key in data.files:
                sha, _, name = key.partition("__")
                if sha not in want:
                    continue
                if name == "v_C":
                    vc[sha] = np.asarray(data[key][layer], dtype=np.float64)
                elif name == "v_A_greedy":
                    va[sha] = np.asarray(data[key][layer], dtype=np.float64)
    missing = [s for s in shas if s not in vc or s not in va]
    if missing:
        raise KeyError(f"{len(missing)} shas missing from staged shards (first {missing[0]})")
    return (
        np.stack([vc[s] for s in shas]),
        np.stack([va[s] for s in shas]),
    )


MAP_BUNDLE_KEYS = ("U", "S", "Vt", "xmu", "xsd", "ymu")


def load_map_bundle(stage: Path, layer: int, cond: str = "3a_generic") -> dict[str, np.ndarray]:
    """Load the banked SVD map bundle at ``layer``, asserting the full key set.

    Mirrors ``issue2356_fits.load_map_bundle``'s assert so a schema drift fails
    loud here instead of silently reconstructing a wrong prediction.
    """
    npz = (
        stage
        / "issue2356_refusalpred"
        / "analysis_tensors"
        / "maps"
        / cond
        / f"layer{layer:02d}.npz"
    )
    if not npz.exists():
        raise FileNotFoundError(npz)
    with np.load(npz) as data:
        for k in MAP_BUNDLE_KEYS:
            if k not in data.files:
                raise KeyError(f"{npz}: missing bundle key {k!r} (have {sorted(data.files)})")
        return {k: data[k] for k in data.files}


# ── pair construction ─────────────────────────────────────────────────


def build_pairs(rows: dict[str, dict], idx: dict[str, int]) -> dict[str, dict]:
    """Three strata of within-family pairs, each as arrays of row indices."""
    fam: dict[int, list[str]] = collections.defaultdict(list)
    for sha, r in rows.items():
        fam[r["base_id"]].append(sha)

    strata: dict[str, dict] = {}
    # ``same`` restricts the same-behavior control to one behavior, because
    # stratum C is 87% refuse-refuse by construction (2,292 refuse vs 338
    # engage rows): a pooled C could look strong purely from refusal answers
    # being mutually similar, which is the very confound the control exists
    # to rule out.
    for name, rewrite_only, flip, same in (
        ("A_flip_rewrite_only", True, True, None),
        ("B_flip_any", False, True, None),
        ("C_sameBehavior_rewrite_only", True, False, None),
        ("C1_sameBehavior_refuse_only", True, False, "refuse"),
        ("C2_sameBehavior_engage_only", True, False, "engage"),
    ):
        a: list[int] = []
        b: list[int] = []
        fams: list[int] = []
        for base_id, shas in sorted(fam.items()):
            members = [s for s in shas if not rewrite_only or rows[s]["axis"] in REWRITE_AXES]
            eng = [s for s in members if rows[s]["label"] == "engage"]
            ref = [s for s in members if rows[s]["label"] == "refuse"]
            if flip:
                cand = [(x, y) for x in eng for y in ref]
            else:
                groups = {"refuse": [ref], "engage": [eng]}.get(same, [eng, ref])
                cand = [
                    (g[i], g[j])
                    for g in groups
                    for i in range(len(g))
                    for j in range(i + 1, len(g))
                ]
            for x, y in cand:
                a.append(idx[x])
                b.append(idx[y])
                fams.append(base_id)
        strata[name] = {
            "a": np.array(a, dtype=int),
            "b": np.array(b, dtype=int),
            "fam": np.array(fams, dtype=int),
        }
    return strata


# ── metrics ───────────────────────────────────────────────────────────


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-12, None)


def similarity_block(pred: np.ndarray, tgt: np.ndarray) -> np.ndarray:
    """Full (n_rows, n_rows) prediction-vs-target cosine block.

    Every 2AFC and null read below is pure INDEXING into this block, which is
    what makes the batteries cheap: at n=2630 rows the block is ~55 MB, while
    gathering d-dimensional target vectors per null draw would materialise
    (draws x pairs x 3584) — tens of GB for the 4,098-pair control stratum.
    This is also the exact input shape issue2215_analysis.observed_2afc takes.
    """
    return _unit(pred) @ _unit(tgt).T


def afc_bits(s: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Per-direction correctness bits, concatenated (dir-A then dir-B).

    Delegates the margin convention to issue2215_analysis.observed_2afc:
    dir-A margin = S[a,a] - S[a,b], dir-B margin = S[b,b] - S[b,a].
    """
    m_a, m_b = observed_2afc(s, a, b)
    return np.concatenate([m_a, m_b]) > 0


def cluster_bootstrap(bits: np.ndarray, fam: np.ndarray, rng: np.random.Generator) -> list[float]:
    """Family-clustered bootstrap CI. Vectorised over draws: one (B, n_fam)
    index matrix, then a bincount-free gather over precomputed per-family
    (correct, total) sums."""
    fams, inv = np.unique(fam, return_inverse=True)
    two = np.concatenate([inv, inv])
    corr = np.bincount(two, weights=bits.astype(np.float64), minlength=len(fams))
    tot = np.bincount(two, minlength=len(fams)).astype(np.float64)
    draws = rng.integers(0, len(fams), size=(N_BOOT, len(fams)))
    acc = corr[draws].sum(axis=1) / np.clip(tot[draws].sum(axis=1), 1e-12, None)
    return [float(np.percentile(acc, 2.5)), float(np.percentile(acc, 97.5))]


def derangement_null(
    s: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    fam: np.ndarray,
    rng: np.random.Generator,
) -> list[float]:
    """Family-blocked derangement null with side randomisation.

    Each pair's prediction is scored against a DIFFERENT family's answer duo,
    with the side label randomised per draw, so the null mean is 0.5 by
    construction — the issue2215 `null_2afc_cell` property, re-derived here
    because that helper is keyed to #2215's carrier x value-pair grid, which
    #2356's variable-size base families have no analogue of.

    Reads are pure indexing into the precomputed similarity block, so peak
    memory is O(draw_chunk x n_pairs), never O(draw_chunk x n_pairs x d).
    """
    n = len(a)
    if n < 2 or len(np.unique(fam)) < 2:
        return [float("nan"), float("nan")]
    accs = np.empty(N_NULL)
    chunk_size = max(1, min(500, 4_000_000 // max(n, 1)))
    for start in range(0, N_NULL, chunk_size):
        size = min(chunk_size, N_NULL - start)
        q = rng.integers(0, n, size=(size, n))
        # reject self-family assignments by resampling those cells
        bad = fam[q] == fam[None, :]
        while bad.any():
            q[bad] = rng.integers(0, n, size=int(bad.sum()))
            bad = fam[q] == fam[None, :]
        flip_a = rng.random((size, n)) < 0.5
        flip_b = rng.random((size, n)) < 0.5
        own_a = np.where(flip_a, b[q], a[q])
        oth_a = np.where(flip_a, a[q], b[q])
        own_b = np.where(flip_b, a[q], b[q])
        oth_b = np.where(flip_b, b[q], a[q])
        a_row, b_row = a[None, :], b[None, :]
        m_a = s[a_row, own_a] - s[a_row, oth_a]
        m_b = s[b_row, own_b] - s[b_row, oth_b]
        accs[start : start + size] = ((m_a > 0).sum(axis=1) + (m_b > 0).sum(axis=1)) / (2.0 * n)
    return [float(np.percentile(accs, 2.5)), float(np.percentile(accs, 97.5))]


def loo_family_idbias(vc: np.ndarray, va: np.ndarray, fam_of_row: np.ndarray) -> np.ndarray:
    """Leave-one-FAMILY-out identity+learned-bias predictions.

    Same estimator as analysis.mapping_baselines.identity_bias_predict
    (pred = x + mean(y - x)) with the bias held out at family grain, the #2215
    leave-one-type-out convention. Computed by total-minus-family sums rather
    than a per-family refit loop.
    """
    resid = va - vc
    fams, inv = np.unique(fam_of_row, return_inverse=True)
    tot = resid.sum(axis=0)
    n = len(resid)
    fam_sum = np.zeros((len(fams), resid.shape[1]))
    np.add.at(fam_sum, inv, resid)
    fam_n = np.bincount(inv, minlength=len(fams)).astype(np.float64)
    bias = (tot[None, :] - fam_sum) / np.clip(n - fam_n, 1e-12, None)[:, None]
    return vc + bias[inv]


# ── main ──────────────────────────────────────────────────────────────


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stage", default="/mnt/eps-data/thomasjiralerspong/issue2356_flippair/hf_dl")
    ap.add_argument(
        "--artifacts", default=str(REPO / ".claude/worktrees/issue-2356/eval_results/issue_2356")
    )
    ap.add_argument("--out", default=str(REPO / "eval_results/issue_2356/flippair_2afc"))
    ap.add_argument("--fits-module", default=str(DEFAULT_FITS_MODULE))
    args = ap.parse_args()

    stage, art, out = Path(args.stage), Path(args.artifacts), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    fits = load_fits_module(Path(args.fits_module))
    map_predict: Callable[..., np.ndarray] = fits.map_predict

    rows = load_rows(art)
    shas = sorted(rows)
    idx = {s: i for i, s in enumerate(shas)}
    print(
        f"[load] {len(shas)} labelled armA rows, {len({r['base_id'] for r in rows.values()})} families"
    )

    vc, va = load_store(stage, shas, LAYER)
    bundle = load_map_bundle(stage, LAYER)
    print(f"[load] v_C {vc.shape}, v_A {va.shape}, map U {bundle['U'].shape}")

    # full-rank reconstruction through the producing module's own predictor
    pred_map = map_predict(bundle, vc)
    if pred_map.shape != va.shape:
        raise ValueError(f"map prediction shape {pred_map.shape} != target shape {va.shape}")
    fam_of_row = np.array([rows[s]["base_id"] for s in shas])
    pred_idb = loo_family_idbias(vc, va, fam_of_row)

    # Precompute the two cosine blocks ONCE: every 2AFC + null read below is
    # indexing into these, which is what keeps the null batteries in MB.
    sim = {
        "map_3a_generic": similarity_block(pred_map, va),
        "identity_bias_loo": similarity_block(pred_idb, va),
    }
    print(f"[blocks] similarity blocks {sim['map_3a_generic'].shape} built", flush=True)

    strata = build_pairs(rows, idx)
    results: dict[str, dict] = {}

    # cross-family context yardstick: median distance between contexts of
    # DIFFERENT families, the #2215 DV1 convention.
    samp = rng.integers(0, len(shas), size=(40_000, 2))
    samp = samp[fam_of_row[samp[:, 0]] != fam_of_row[samp[:, 1]]]
    yard = float(np.median(np.linalg.norm(vc[samp[:, 0]] - vc[samp[:, 1]], axis=1)))
    print(f"[yardstick] cross-family median ||dv_C|| = {yard:.3f} (n={len(samp)})")

    for name, s in strata.items():
        a, bb, fam = s["a"], s["b"], s["fam"]
        if len(a) == 0:
            results[name] = {"n_pairs": 0, "skipped": "no pairs"}
            continue
        dist = np.linalg.norm(vc[a] - vc[bb], axis=1)
        cos = np.einsum("ij,ij->i", _unit(vc[a]), _unit(vc[bb]))
        entry: dict = {
            "n_pairs": int(len(a)),
            "n_families": int(len(np.unique(fam))),
            "context_dist_median": float(np.median(dist)),
            "context_dist_ratio_to_yardstick": float(np.median(dist) / yard),
            "context_cosine_median": float(np.median(cos)),
            "arms": {},
        }
        for arm, block in sim.items():
            bits = afc_bits(block, a, bb)
            entry["arms"][arm] = {
                "acc": float(bits.mean()),
                "ci95_family_clustered": cluster_bootstrap(bits, fam, rng),
                "n_directions": int(len(bits)),
            }
        entry["null_band_95"] = derangement_null(sim["map_3a_generic"], a, bb, fam, rng)
        results[name] = entry
        print(
            f"[{name}] n={entry['n_pairs']} fams={entry['n_families']} "
            f"ctx_ratio={entry['context_dist_ratio_to_yardstick']:.3f} "
            f"map={entry['arms']['map_3a_generic']['acc']:.3f} "
            f"idb={entry['arms']['identity_bias_loo']['acc']:.3f} "
            f"null={entry['null_band_95']}",
            flush=True,
        )

    # pool-level retrieval companion (mandatory kNN read, standing rule)
    knn = knn_retrieval(pred_map, va, ks=(1, 5, 10), metric="cosine")
    payload = {
        "meta": {
            "issue": 2356,
            "round": "flippair_2afc",
            "layer": LAYER,
            "map": "analysis_tensors/maps/3a_generic/layer20.npz (label-blind, generic corpus)",
            "seed": SEED,
            "n_boot": N_BOOT,
            "n_null": N_NULL,
            "gpu_hours": 0,
        },
        "cross_family_yardstick_norm": yard,
        "strata": results,
        "pool_knn_retrieval_cosine": knn,
    }
    (out / "flippair_2afc.json").write_text(json.dumps(payload, indent=1))
    print(f"[write] {out / 'flippair_2afc.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
