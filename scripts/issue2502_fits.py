"""Issue #2502 P4: pooled val-lambda ridge map cx_last -> v_x per model (plan v6 S4 P4).

Per model, fits ONE pooled val-lambda-selected primal ridge map per candidate layer
(reusing the #779 core math over ``LAMBDAS_N50K``), evaluated on the held-out test
partition pooled + per source + per LODO group fold, alongside the mandatory
identity+learned-bias baseline and kNN-retrieval reads (#722 helpers), and persists
the per-context x per-layer x per-arm reconstruction matrix that drives every
selection-inherited bootstrap CI (MF-D) plus the registered MF-C decision function.

Key conventions (read before consuming outputs):
- Layer index ``hs`` k in 1..n_layers = ``hidden_states[k]`` = output of decoder
  block k-1 (matches the u2 capture store's ``L{k:02d}`` files). The registered H3
  full-attention set is given as 0-indexed decoder BLOCKS [3,7,...,31] (config
  ``layer_types`` indices) => hs indices block+1 (MF-J assert before fitting).
- Relative depth of hs k on an L-layer model = k / L; Model A's H3 candidate set is
  the nearest-relative-depth match to Model B's 8 full-attention layers (ties break
  to the SMALLER hs index; realized match: A hs [3,7,10,14,17,21,24,28]).
- Pooled held-out R^2 = 1 - SS_res/SS_tot with SS_tot on the eval slice's OWN mean
  (the #779 ``issue779_percontext_recon._pooled_r2`` convention). WITHIN-SOURCE
  centering (PRIMARY per plan S6) replaces the slice mean with each source's own
  test-slice mean; singleton-source rows (n_src < 2) are excluded from the
  within-source statistic (their SST is 0 by construction) and counted.
- Bootstrap CIs are computed from the persisted matrix with the SST term held at
  the FIXED full-slice mean (per layer/model/slice) — the matrix stores scalars,
  so resamples do not re-center; documented here + in the artifact meta.
- Selection-inherited CIs are VAL-based (#9, MF-D / selection-symmetric-nulls):
  the recon matrix persists validation AND test residuals separately; each
  bootstrap draw re-selects the layer on an INDEPENDENT resample of the
  VALIDATION rows (argmax val pooled R^2 — the production ``_select`` rule) and
  evaluates on an independently drawn TEST resample. Selection never sees the
  scored partition; the frozen-at-selected CI rides alongside, labeled.
- MF-B: NO ``n_train < d`` refusal anywhere — the G1 pilot is a deliberately
  under-determined regularization-limit read (val-selected lambda, the #1701
  escape); pilot fit JSONs persist n_train, d, selected lambda, lambda_grid_edge,
  numerical rank, effective dof.
- Linear map ONLY (project standing rule) — no MLP/nonlinear leg.
- Fit-core plan deviation (#10, disclosed): plan v6 S9 names the #779
  ``fit_ridge_primal`` "vectorized across layers per model"; production keeps a
  STREAMED twin because the reference materializes all-lambda x all-eval
  predictions (~31 GB fp64 at n_val=n_te~22.5k, H=4096, 21 lambdas — over the
  pod RAM budget). Batched instead: the lambda scan (closed-form quadratic scan
  in the eigenbasis, all 21 lambdas at once) and the LODO axis (ONE shared
  train/val reduction reused by every fold). The LAYER axis stays a
  checkpointed loop: layers share no data (no common factorization exists),
  each unit is one FLOP-bound BLAS-saturating fit, and the plan's across-layer
  parallelism is the ``--layers`` shard axis. Measured basis: ``--phase
  unitwall`` times ONE production-shape unit through the real entrypoint.
- Durability (#12): production ``fit``/``decide`` REQUIRE ``--publish`` —
  committed JSON artifacts are mirrored to the HF data repo (verified per
  file) and git-committed + pushed on the issue branch (LOUD degrade on
  git-less SLURM lanes; the HF leg is mandatory). ``decide`` writes the
  ``fits/.p4_done`` sentinel LAST, after publish verifies. Smokes pass
  ``--publish none`` with a scratch ``--out-root``.
- Resume grain (#14): every resumable unit is its OWN ledger cell keyed by its
  generating parameters — per-layer ``L{k:02d}``, per-fold ``lodo_{group}``,
  kNN ``knn_gate{hs}_n{max_n}`` (param-keyed: a changed ``--knn-max-n``
  recomputes), first-PC ``firstpc_gate{hs}``. There is NO aggregate "post"
  cell: a subset run (``--lodo-groups``/``--skip-lodo``) can never mark
  unfinished sibling work complete; assembly records ``lodo_status`` honestly.

Phases (argparse driver; ``--import-check`` runs the argcheck completeness pass):
  fit       one model (--model-key A|B): capture-store completeness gate (#11)
            -> per-layer pooled fits -> percell checkpoints -> selection ->
            kNN + LODO + first-PC -> fits_summary.json + percontext_recon.json
            + firstpc_scatter.json under --out-root/fits/model{A|B}/ -> publish.
  decide    reads BOTH models' assembled artifacts (+ BOTH reliability ceilings,
            REQUIRED, #13) -> A_pass/B_pass/NI selection-inherited bootstrap
            gates + H2 class contrast -> the registered MF-C truth table ->
            fits/decision.json -> publish -> .p4_done sentinel.
  unitwall  #10 measured per-unit wall: ONE production-shape fit_layer_unit on
            synthetic data through the real entrypoint (stage timings persisted).
  selfcheck synthetic end-to-end dry-run: streamed-core equivalence vs the #779
            reference, batched-LODO equivalence vs the serial reference, #9
            val-vs-test-winner fixture, #11 completeness refusals, #14
            subset-then-full resume, fit -> baseline -> kNN -> per-source/LODO
            -> selection-inherited CI -> truth table incl. REACHABLE Inconclusive.

Compute placement: the production fits run on a cpu-bigmem pod (`--device cpu`);
VM-side selfchecks carry the shared-VM thread-cap env prefix inline. Per-unit
percell checkpoints + a StageLedger (regime keyed on GENERATING PARAMETERS, #1336)
make every long loop resumable; one progress line per unit.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
import time
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

ISSUE = 2502
HF_DATA_REPO = "superkaiba1/explore-persona-space-data"

# Registered H3 full-attention layers of Qwen3.5-9B as 0-indexed decoder BLOCKS
# (live-verified config.json ``layer_types`` indices; plan S12 assumption 1).
REGISTERED_H3_BLOCKS = (3, 7, 11, 15, 19, 23, 27, 31)

NI_MARGIN = 0.10  # H3 one-sided non-inferiority margin on (B_best - A_best)
H2_MARGIN = 0.05  # ordinary-mean minus weird-mean descriptive threshold
G1_R2_FLOOR = 0.40  # G1 pilot pooled best-layer held-out R^2 floor (plan S7)
G1_GATE_RC = 7  # designed artifact-routed halt rc on a --g1-gate FAIL (gotchas)
BOOT_SEED = 2502
PILOT_SPLIT_SEED = 42
PILOT_TRAIN_FRAC = 0.80
LAMBDA_GRID_PARAMS = ("logspace", -3, 7, 21)  # == issue779 LAMBDAS_N50K generating params

MODEL_NAME = {"A": "Qwen/Qwen2.5-7B-Instruct", "B": "Qwen/Qwen3.5-9B"}
MODEL_N_LAYERS = {"A": 28, "B": 32}
MODEL_HIDDEN = {"A": 3584, "B": 4096}
DEFAULT_TENSORS_PREFIX = {
    "A": "issue2502_ctxmap_xgen/analysis_tensors/modelA",
    "B": "issue2502_ctxmap_xgen/analysis_tensors/modelB",
}
# HF mirror root for committed eval_results JSONs (#12): files land at
# {PUBLISH_EVAL_MIRROR}/{path relative to --out-root}.
PUBLISH_EVAL_MIRROR = "issue2502_ctxmap_xgen/eval_mirror"

_CHUNK_ROWS_RE = re.compile(r"/(?P<key>(?:s\d+_)?chunk\d{4})/(?P=key)__rows\.json$")
_CHUNK_LAYER_RE = re.compile(r"/(?P<key>(?:s\d+_)?chunk\d{4})/(?P=key)__L(?P<k>\d{2})\.npz$")
_SENTINEL_RE = re.compile(r"/\.capture_done(?:_s(?P<i>\d{2})of(?P<m>\d{2}))?$")


def _gc():
    """Sibling u2 driver (codec + fetch + ledger + metadata helpers; light import)."""
    import issue2502_gen_capture as GC

    return GC


def _n779():
    """The reused #779 fit module (imports torch/numpy at ITS module top)."""
    import issue779_ffc_n50k_fits as N779

    return N779


def sha16(text: str) -> str:
    """First 16 hex chars of sha256 (machine-stable string keys only, #1336)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def round6(values) -> list[float]:
    """Round to 6 significant digits for compact JSON persistence."""
    return [float(f"{float(v):.6g}") for v in values]


def pooled_r2(pred, true) -> float:
    """Pooled R^2, SS_tot on TRUE's OWN mean — numpy copy of the exact
    ``issue779_percontext_recon._pooled_r2`` convention (equivalence pinned in
    --phase selfcheck via the reference fit path). NaN on degenerate SS_tot."""
    import numpy as np

    pred = np.asarray(pred, dtype=np.float64)
    true = np.asarray(true, dtype=np.float64)
    mu = true.mean(0)
    ss_res = float(np.sum((true - pred) ** 2))
    ss_tot = float(np.sum((true - mu) ** 2))
    return float("nan") if ss_tot < 1e-12 else 1.0 - ss_res / ss_tot


def r2_from_sums(err_sum: float, sst_sum: float) -> float:
    """R^2 from precomputed per-context sums (same convention as pooled_r2)."""
    return float("nan") if sst_sum < 1e-12 else 1.0 - err_sum / sst_sum


def sanitize_nonfinite(obj):
    """Recursively map non-finite floats (NaN/Inf) to None so committed JSON
    artifacts stay strict-JSON parseable (jq et al.; singleton-source
    within-source R^2 is the known NaN producer)."""
    if isinstance(obj, float):
        return obj if math.isfinite(obj) else None
    if isinstance(obj, dict):
        return {k: sanitize_nonfinite(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_nonfinite(v) for v in obj]
    return obj


def write_artifact_json(path: Path, obj) -> None:
    """Atomic write of a COMMITTED deliverable with non-finite floats -> null.

    Percell checkpoints keep ``atomic_write_json`` (native NaN semantics for
    Python round-trips); everything under eval_results/ goes through here."""
    _gc().atomic_write_json(path, sanitize_nonfinite(obj))


# --------------------------------------------------------------------------
# Candidate layer sets (MF-D relative-depth matching; MF-J assert)
# --------------------------------------------------------------------------


def h3_hs_set(model_key: str) -> list[int]:
    """H3 candidate hs indices per model (equal-breadth 8-vs-8, MF-D).

    B: the registered full-attention blocks +1. A: nearest-relative-depth match
    (rel depth of hs k on an L-layer model = k/L; ties -> smaller hs)."""
    b_hs = [b + 1 for b in REGISTERED_H3_BLOCKS]
    if model_key == "B":
        return b_hs
    n_a, n_b = MODEL_N_LAYERS["A"], MODEL_N_LAYERS["B"]
    out = []
    for kb in b_hs:
        # exact tie handling: |ka/n_a - kb/n_b| ∝ |ka*n_b - kb*n_a| (integers)
        best = min(range(1, n_a + 1), key=lambda ka: (abs(ka * n_b - kb * n_a), ka))
        out.append(best)
    if len(set(out)) != len(out):
        raise RuntimeError(f"relative-depth matching produced duplicate A layers: {out}")
    return out


def gate_hs_set(model_key: str, captured: list[int]) -> list[int]:
    """Gate (A_pass/B_pass selection) candidate set: A = all captured layers;
    B = the registered full-attention set only (linear-attn layers are a
    robustness read, never a gate input — plan S4/S5)."""
    if model_key == "A":
        return sorted(captured)
    return [k for k in h3_hs_set("B")]


def assert_mfj(model_key: str, captured: list[int]) -> dict:
    """MF-J: the registered H3 set must be a subset of the captured layers."""
    need = h3_hs_set(model_key)
    missing = sorted(set(need) - set(captured))
    if missing:
        raise RuntimeError(
            f"MF-J violation (model {model_key}): registered H3 hs layers {need} "
            f"not all captured (missing {missing}; captured {sorted(captured)}) — "
            "terminate + re-register H3 rather than fitting a partial set"
        )
    return {
        "registered_h3_blocks_0idx": list(REGISTERED_H3_BLOCKS),
        "h3_hs": need,
        "captured_hs": sorted(captured),
        "subset_ok": True,
    }


# --------------------------------------------------------------------------
# Stores (HF-backed production store + in-memory selfcheck store)
# --------------------------------------------------------------------------


class HfChunkStore:
    """Reader for a u2 gen_capture tensor store (per-chunk per-layer bf16 npz).

    Streams ONE layer at a time (fetch chunk npz -> decode -> concat -> unlink),
    bounding local disk to ~one chunk file and RAM to ~one layer (plan S9)."""

    def __init__(self, prefix: str, work: Path, hidden: int):
        self.prefix = prefix.rstrip("/")
        self.work = work
        self.hidden = hidden
        self._files: list[str] | None = None
        self._keys: list[str] | None = None
        self._expected_rows_per_chunk: dict[str, int] | None = None
        self._expected_total_rows: int | None = None

    def _fetch_json(self, rel: str) -> dict:
        """Fetch + parse ONE JSON doc under the prefix (selfcheck seam)."""
        GC = _gc()
        local = GC.fetch_repo_file(f"{self.prefix}/{rel}", self.work / "json_dl", what=rel)
        doc = json.loads(local.read_text(encoding="utf-8"))
        local.unlink()
        return doc

    def verify_complete(self) -> dict:
        """#11 capture-completeness gate — the u2 producer contract, verbatim.

        (a) require the ``.capture_done`` sentinel(s) (unsharded, or the FULL
        ``_sIIofMM`` shard set); (b) load capture_meta per sentinel; (c) the
        discovered rows.json chunk keys == the union of ``meta.chunk_keys``
        EXACTLY (set + count, disjoint across shards); (d) every chunk carries
        the COMPLETE ``meta.layers`` set (+ every ``expected_files`` entry
        present); (e) arm ``load_rows`` to reconcile realized row counts vs
        ``meta.per_chunk``/``totals.n_rows_captured``. Any mismatch fails loud
        — never fit a strict subset of an unfinished capture."""
        files = self._listing()
        sentinels = {}
        for f in files:
            m = _SENTINEL_RE.search(f)
            if m:
                sentinels[f] = (m.group("i"), m.group("m"))
        if not sentinels:
            raise RuntimeError(
                f"capture INCOMPLETE under {self.prefix}: no .capture_done sentinel — "
                "never fit an unfinished capture (#11)"
            )
        unsharded = [f for f, (i, _m) in sentinels.items() if i is None]
        sharded = {f: (int(i), int(mm)) for f, (i, mm) in sentinels.items() if i is not None}
        if unsharded and sharded:
            raise RuntimeError(
                f"ambiguous sentinels under {self.prefix}: both unsharded and sharded "
                f"({sorted(sentinels)})"
            )
        if unsharded:
            meta_names = ["capture_meta.json"]
        else:
            n_shards = {mm for (_i, mm) in sharded.values()}
            if len(n_shards) != 1:
                raise RuntimeError(f"inconsistent shard counts in sentinels: {sorted(sentinels)}")
            total = n_shards.pop()
            have = sorted(i for (i, _mm) in sharded.values())
            if have != list(range(total)):
                raise RuntimeError(
                    f"capture INCOMPLETE under {self.prefix}: shard sentinels {have} != "
                    f"0..{total - 1} (#11)"
                )
            meta_names = [f"capture_meta_s{i:02d}of{total:02d}.json" for i in range(total)]
        meta_keys: list[str] = []
        layer_sets: list[tuple[str, tuple[int, ...]]] = []
        per_chunk_rows: dict[str, int] = {}
        total_rows = 0
        expected_files: list[str] = []
        for name in meta_names:
            meta = self._fetch_json(name)
            keys = list(meta["chunk_keys"])
            dup = set(keys) & set(meta_keys)
            if dup:
                raise RuntimeError(f"chunk keys {sorted(dup)} appear in multiple shard metas")
            meta_keys.extend(keys)
            layer_sets.append((name, tuple(sorted(int(k) for k in meta["layers"]))))
            for key, pc in meta["per_chunk"].items():
                per_chunk_rows[key] = int(pc["n_rows"])
            total_rows += int(meta["totals"]["n_rows_captured"])
            expected_files.extend(meta.get("expected_files", []))
        if len({ls for _n, ls in layer_sets}) != 1:
            raise RuntimeError(f"shard metas disagree on the captured layer set: {layer_sets}")
        expected_layers = set(layer_sets[0][1])
        discovered = set(self.chunk_keys())
        if discovered != set(meta_keys) or len(meta_keys) != len(set(meta_keys)):
            raise RuntimeError(
                f"capture INCOMPLETE under {self.prefix}: discovered rows.json chunks != "
                f"capture_meta.chunk_keys (missing {sorted(set(meta_keys) - discovered)}; "
                f"extra {sorted(discovered - set(meta_keys))}) (#11)"
            )
        per_key_layers: dict[str, set[int]] = {}
        for f in files:
            m = _CHUNK_LAYER_RE.search(f)
            if m:
                per_key_layers.setdefault(m.group("key"), set()).add(int(m.group("k")))
        bad = {
            k: sorted(expected_layers - per_key_layers.get(k, set()))
            for k in sorted(discovered)
            if per_key_layers.get(k, set()) != expected_layers
        }
        if bad:
            raise RuntimeError(
                f"capture INCOMPLETE under {self.prefix}: chunks with a partial layer set "
                f"vs capture_meta.layers: {bad} (#11)"
            )
        missing_expected = sorted(set(expected_files) - set(files))
        if missing_expected:
            raise RuntimeError(
                f"capture INCOMPLETE under {self.prefix}: {len(missing_expected)} "
                f"expected_files absent from the listing (first: {missing_expected[:5]}) (#11)"
            )
        missing_pc = sorted(discovered - set(per_chunk_rows))
        if missing_pc:
            raise RuntimeError(f"capture_meta.per_chunk lacks row counts for {missing_pc}")
        self._expected_rows_per_chunk = per_chunk_rows
        self._expected_total_rows = total_rows
        print(
            f"[store] capture completeness OK under {self.prefix}: "
            f"{len(discovered)} chunks x {len(expected_layers)} layers, "
            f"{total_rows} rows declared",
            flush=True,
        )
        return {
            "n_chunks": len(discovered),
            "layers": sorted(expected_layers),
            "n_rows_declared": total_rows,
            "sharded": bool(sharded),
        }

    def _listing(self) -> list[str]:
        if self._files is None:
            from huggingface_hub import HfApi

            from explore_persona_space.orchestrate import hub

            self._files = hub.retry_transient(
                lambda: hub.list_hf_files_under_path(
                    HfApi(), HF_DATA_REPO, self.prefix, repo_type="dataset"
                ),
                what=f"list({self.prefix})",
            )
        return self._files

    def chunk_keys(self) -> list[str]:
        if self._keys is None:
            keys = sorted(
                {m.group("key") for f in self._listing() if (m := _CHUNK_ROWS_RE.search(f))}
            )
            if not keys:
                raise RuntimeError(f"no capture chunks (rows.json) found under {self.prefix}")
            self._keys = keys
        return self._keys

    def captured_hs(self) -> list[int]:
        """hs indices present in EVERY chunk (non-uniform sets fail LOUD — #11:
        an intersection would silently fit a strict layer subset)."""
        per_key: dict[str, set[int]] = {}
        for f in self._listing():
            m = _CHUNK_LAYER_RE.search(f)
            if m:
                per_key.setdefault(m.group("key"), set()).add(int(m.group("k")))
        keys = self.chunk_keys()
        missing = [k for k in keys if k not in per_key]
        if missing:
            raise RuntimeError(f"chunks with rows.json but no layer npz: {missing[:5]}")
        common = set.intersection(*(per_key[k] for k in keys))
        if not common:
            raise RuntimeError(f"no layer index captured in every chunk under {self.prefix}")
        union = set.union(*(per_key[k] for k in keys))
        if union != common:
            raise RuntimeError(
                f"non-uniform layer sets across chunks under {self.prefix}: "
                f"intersection {sorted(common)} != union {sorted(union)} — capture "
                "incomplete; never fit a strict subset (#11)"
            )
        return sorted(common)

    def load_rows(self) -> list[dict]:
        """Concatenated per-chunk rows metadata, in chunk-key order (= npz row
        order); realized counts reconciled vs capture_meta after
        ``verify_complete`` (#11 check (e))."""
        rows: list[dict] = []
        for key in self.chunk_keys():
            doc = self._fetch_json(f"{key}/{key}__rows.json")
            chunk_rows = doc["rows"]
            for i, r in enumerate(chunk_rows):
                if r["row"] != i:
                    raise RuntimeError(f"rows.json {key} row-order mismatch at {i} != {r['row']}")
            if self._expected_rows_per_chunk is not None:
                want = self._expected_rows_per_chunk.get(key)
                if want != len(chunk_rows):
                    raise RuntimeError(
                        f"rows.json {key}: {len(chunk_rows)} realized rows != "
                        f"capture_meta.per_chunk {want} (#11)"
                    )
            rows.extend({**r, "chunk_key": key} for r in chunk_rows)
        if not rows:
            raise RuntimeError(f"empty row table under {self.prefix} — fail loud")
        if self._expected_total_rows is not None and len(rows) != self._expected_total_rows:
            raise RuntimeError(
                f"row-total mismatch under {self.prefix}: {len(rows)} realized vs "
                f"totals.n_rows_captured {self._expected_total_rows} (#11)"
            )
        return rows

    def load_layer(self, k: int):
        """(X=cx_last, Y=v_x) fp32 numpy for hs layer k, rows in store order."""
        import numpy as np

        GC = _gc()
        import torch

        xs, ys = [], []
        for key in self.chunk_keys():
            local = GC.fetch_repo_file(
                f"{self.prefix}/{key}/{key}__L{k:02d}.npz",
                self.work / "tensors_dl",
                what=f"layer({key},L{k:02d})",
            )
            with np.load(local) as z:
                cx = GC.decode_bf16(z["cx_last"], torch).float().numpy()
                vx = GC.decode_bf16(z["vx"], torch).float().numpy()
            local.unlink()
            if cx.shape[1] != self.hidden or vx.shape[1] != self.hidden:
                raise RuntimeError(
                    f"hidden-dim mismatch in {key} L{k:02d}: {cx.shape} / {vx.shape} "
                    f"vs expected H={self.hidden}"
                )
            xs.append(cx)
            ys.append(vx)
        return np.concatenate(xs), np.concatenate(ys)


class MemStore:
    """In-memory store with the HfChunkStore duck-type (selfcheck seam)."""

    def __init__(self, rows: list[dict], layers: dict[int, tuple]):
        self._rows = rows
        self._layers = layers

    def chunk_keys(self) -> list[str]:
        return ["chunk0000"]

    def captured_hs(self) -> list[int]:
        return sorted(self._layers)

    def load_rows(self) -> list[dict]:
        return list(self._rows)

    def load_layer(self, k: int):
        return self._layers[k]


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------


def resolve_splits(rows: list[dict], *, pilot: bool):
    """(tr, val, te) index arrays. Production: the corpus-assigned P0 split
    (70/15/15). Pilot (G1, MF-B): seeded 80/20 with val == heldout == test —
    the pilot selects lambda AND reads R^2 on the same 20% (a deliberately
    optimistic rig-sanity floor, recorded as ``pilot_val_is_test``)."""
    import numpy as np

    if pilot:
        order = np.argsort(np.array([r["context_id"] for r in rows]))
        perm = np.random.default_rng(PILOT_SPLIT_SEED).permutation(order)
        n_tr = int(round(PILOT_TRAIN_FRAC * len(rows)))
        if n_tr < 1 or len(rows) - n_tr < 2:
            raise RuntimeError(f"pilot split degenerate: n={len(rows)}")
        tr, held = np.sort(perm[:n_tr]), np.sort(perm[n_tr:])
        return tr, held, held
    buckets: dict[str, list[int]] = {"train": [], "val": [], "test": []}
    for i, r in enumerate(rows):
        s = r.get("split")
        if s not in buckets:
            raise RuntimeError(f"row {i} ({r.get('context_id')}): unknown split {s!r}")
        buckets[s].append(i)
    for name in ("train", "val", "test"):
        if len(buckets[name]) < 2:
            raise RuntimeError(f"split {name!r} has {len(buckets[name])} rows — degenerate")
    return (
        np.asarray(buckets["train"]),
        np.asarray(buckets["val"]),
        np.asarray(buckets["test"]),
    )


# --------------------------------------------------------------------------
# Streamed ridge core (memory-lean twin of the #779 fit; equivalence-pinned)
# --------------------------------------------------------------------------


def _eigh_robust(A, dev):
    """torch eigh with the cuSOLVER->CPU-LAPACK non-convergence fallback
    (gotchas: never jitter the Gram; a CPU failure is genuinely pathological)."""
    import torch

    try:
        return torch.linalg.eigh(A)
    except torch.linalg.LinAlgError:
        if str(dev) == "cpu":
            raise  # genuinely pathological input — fail loud (gotchas: never jitter)
        s, U = torch.linalg.eigh(A.cpu())  # cusolver non-convergence -> CPU LAPACK
        print(f"[fit] eigh cuda->cpu fallback engaged (H={A.shape[0]})", flush=True)
        return s.to(A.device), U.to(A.device)


def _select_lambda(vr2, lambdas):
    """Strict first-best-wins selection over finite val R^2 — the exact rule of
    the #779 reference loop. Returns (best_lambda, best_vr2, grid_edge)."""
    import numpy as np

    best_lam, best_vr2 = float(lambdas[0]), -np.inf
    for i, lam in enumerate(lambdas):
        if np.isfinite(vr2[i]) and vr2[i] > best_vr2:
            best_vr2, best_lam = float(vr2[i]), float(lam)
    edge = None
    if np.isclose(best_lam, float(lambdas[0])):
        edge = "low"
    elif np.isclose(best_lam, float(lambdas[-1])):
        edge = "high"
    return best_lam, best_vr2, edge


def ridge_fit_streamed(X, Y, tr, val, eval_idx_sets, lambdas, dev, *, timings=None):
    """Val-lambda-selected primal ridge — numerically IDENTICAL math to
    ``issue779_ffc_n50k_fits.fit_ridge_primal`` (standardize X on train stats,
    center Y on train mean, ONE eigh of the (H,H) X^T X, strict
    first-best-wins val selection), with TWO throughput deviations (#10,
    disclosed in the module docstring): (1) only the SELECTED lambda's
    eval-set predictions are materialized (the reference materializes
    all-lambda x all-eval-set predictions — ~31 GB fp64 at #2502 eval sizes);
    (2) the lambda scan is BATCHED as a closed-form quadratic evaluation in
    the eigenbasis — with P = Val_n @ U, K = P^T P, M = U^T X^T Yc,
    c = rowsum(M * (P^T Yv_c)), T = K * (M M^T) and g_lam = 1/(s+lam):
    SS_res(lam) = ||Yv_c||^2 - 2 g_lam.c + g_lam^T T g_lam, all lambdas in one
    GEMM pass. Same selection rule, no permissiveness change; equivalence vs
    the reference (selected-lambda EQUALITY + val R^2 + preds at 1e-9) is
    asserted in --phase selfcheck.

    Returns (preds per eval set at selected lambda, meta incl. MF-B diagnostics:
    n_train, d, selected_lambda, lambda_grid_edge, numerical_rank,
    effective_dof). ``timings`` (optional dict) receives per-stage wall
    seconds (consumed by --phase unitwall)."""
    import numpy as np
    import torch

    t_start = time.time()
    Xtr = torch.as_tensor(np.asarray(X[tr]), dtype=torch.float64, device=dev)
    xmu = Xtr.mean(0)
    xsd = Xtr.std(0) + 1e-9
    Xtr_n = (Xtr - xmu) / xsd
    del Xtr
    Yt = torch.as_tensor(np.asarray(Y[tr]), dtype=torch.float64, device=dev)
    ymu = Yt.mean(0)
    Yc = Yt - ymu
    del Yt
    A = Xtr_n.T @ Xtr_n
    if timings is not None:
        timings["gram_s"] = round(time.time() - t_start, 3)
    t1 = time.time()
    s, U = _eigh_robust(A, dev)
    s = torch.clamp(s, min=0.0)
    if timings is not None:
        timings["eigh_s"] = round(time.time() - t1, 3)
    XtY = Xtr_n.T @ Yc
    n_train = int(Xtr_n.shape[0])
    d = int(Xtr_n.shape[1])
    del Xtr_n, Yc
    UtXtY = U.T @ XtY
    del XtY
    t2 = time.time()
    Val = torch.as_tensor(np.asarray(X[val]), dtype=torch.float64, device=dev)
    Val_n = (Val - xmu) / xsd
    del Val
    Yval_t = torch.as_tensor(np.asarray(Y[val]), dtype=torch.float64, device=dev)
    Yvc = Yval_t - ymu
    P = Val_n @ U
    del Val_n
    K = P.T @ P
    PtY = P.T @ Yvc
    del P
    c_vec = (UtXtY * PtY).sum(1)
    del PtY
    T = K * (UtXtY @ UtXtY.T)
    del K
    lam_t = torch.as_tensor([float(x) for x in lambdas], dtype=torch.float64, device=A.device)
    G = 1.0 / (s[None, :] + lam_t[:, None])  # (n_lambda, H)
    ss_res = float((Yvc**2).sum()) - 2.0 * (G @ c_vec) + ((G @ T) * G).sum(1)
    del T, G, c_vec
    ss_tot = float(((Yval_t - Yval_t.mean(0)) ** 2).sum())
    del Yval_t, Yvc
    vr2 = np.full(len(lambdas), np.nan) if ss_tot < 1e-12 else (1.0 - ss_res / ss_tot).cpu().numpy()
    best_lam, best_vr2, edge = _select_lambda(vr2, lambdas)
    if timings is not None:
        timings["val_scan_s"] = round(time.time() - t2, 3)
    t3 = time.time()
    W = U @ (UtXtY / (s + best_lam)[:, None])
    preds = []
    for idx in eval_idx_sets:
        E = torch.as_tensor(np.asarray(X[idx]), dtype=torch.float64, device=dev)
        preds.append((((E - xmu) / xsd) @ W + ymu).cpu().numpy())
        del E
    if timings is not None:
        timings["eval_s"] = round(time.time() - t3, 3)
    s_np = s.cpu().numpy()
    smax = float(s_np.max()) if s_np.size else 0.0
    eps = float(np.finfo(np.float64).eps)
    rank = int((s_np > smax * max(n_train, d) * eps).sum()) if smax > 0 else 0
    meta = {
        "n_train": n_train,
        "d": d,
        "selection": "val-lambda (primal, streamed; batched quadratic-form scan)",
        "selected_lambda": best_lam,
        "val_r2_at_selected": float(best_vr2),
        "lambda_grid_edge": edge,
        "numerical_rank": rank,
        "effective_dof": float((s_np / (s_np + best_lam)).sum()),
    }
    return preds, meta


# --------------------------------------------------------------------------
# Per-layer fit unit (percell checkpoint)
# --------------------------------------------------------------------------


def _source_stats(rows_te: list[dict]):
    """Per-source test-slice index lists + the >=2-row source mask."""
    by_src: dict[str, list[int]] = {}
    for j, r in enumerate(rows_te):
        by_src.setdefault(str(r["source_tag"]), []).append(j)
    multi = {s for s, idx in by_src.items() if len(idx) >= 2}
    return by_src, multi


def fit_layer_unit(X, Y, tr, val, te, rows_te, k, lambdas, dev, *, timings=None):
    """One candidate layer: pooled streamed fit + identity baseline + the
    per-context err/sst scalars for BOTH the test slice (evaluation arm) and
    the validation slice (#9: bootstrap layer re-selection is VAL-based, so
    the recon matrix persists val AND test residuals separately)."""
    import numpy as np

    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    (pred_map, pred_val), meta = ridge_fit_streamed(
        X, Y, tr, val, [te, val], lambdas, dev, timings=timings
    )
    pred_id = identity_bias_predict(X[tr], Y[tr], X[te])
    Yte = np.asarray(Y[te], dtype=np.float64)
    err_map = ((Yte - pred_map) ** 2).sum(1)
    err_id = ((Yte - np.asarray(pred_id, dtype=np.float64)) ** 2).sum(1)
    mu = Yte.mean(0)
    sst_pooled = ((Yte - mu) ** 2).sum(1)
    # Validation-slice residual arrays (#9): same conventions, fixed
    # full-val-slice mean; consistency-checked against the fit core's own
    # val R^2 at the selected lambda (same quantity via a different route).
    Yval = np.asarray(Y[val], dtype=np.float64)
    err_map_val = ((Yval - pred_val) ** 2).sum(1)
    pred_id_val = identity_bias_predict(X[tr], Y[tr], X[val])
    err_id_val = ((Yval - np.asarray(pred_id_val, dtype=np.float64)) ** 2).sum(1)
    sst_val = ((Yval - Yval.mean(0)) ** 2).sum(1)
    r2_val_arrays = r2_from_sums(float(err_map_val.sum()), float(sst_val.sum()))
    r2_val_meta = meta["val_r2_at_selected"]
    if math.isfinite(r2_val_arrays) and math.isfinite(r2_val_meta):
        if abs(r2_val_arrays - r2_val_meta) > 1e-6 * max(1.0, abs(r2_val_meta)):
            raise RuntimeError(
                f"val-slice consistency check failed at hs {k}: arrays R^2 "
                f"{r2_val_arrays} vs fit meta {r2_val_meta}"
            )
    by_src, multi = _source_stats(rows_te)
    sst_ws = np.zeros_like(sst_pooled)
    for src, idx in by_src.items():
        idx = np.asarray(idx)
        if len(idx) >= 2:
            mu_s = Yte[idx].mean(0)
            sst_ws[idx] = ((Yte[idx] - mu_s) ** 2).sum(1)
    ws_rows = np.asarray(sorted(j for s in multi for j in by_src[s]), dtype=int)
    per_source = {}
    for src, idx in sorted(by_src.items()):
        idx = np.asarray(idx)
        per_source[src] = {
            "n": int(len(idx)),
            "regime_class": rows_te[int(idx[0])].get("regime_class"),
            "lodo_group": rows_te[int(idx[0])].get("lodo_group"),
            "r2_map_within_source": r2_from_sums(
                float(err_map[idx].sum()), float(sst_ws[idx].sum())
            ),
            "r2_id_within_source": r2_from_sums(float(err_id[idx].sum()), float(sst_ws[idx].sum())),
            "r2_map_pooled_sst": r2_from_sums(
                float(err_map[idx].sum()), float(sst_pooled[idx].sum())
            ),
            "r2_id_pooled_sst": r2_from_sums(
                float(err_id[idx].sum()), float(sst_pooled[idx].sum())
            ),
        }
    unit = {
        "hs": k,
        "block_0idx": k - 1,
        "fit_meta": meta,
        "r2_test_map_pooled": r2_from_sums(float(err_map.sum()), float(sst_pooled.sum())),
        "r2_test_id_pooled": r2_from_sums(float(err_id.sum()), float(sst_pooled.sum())),
        "r2_test_map_within_source": r2_from_sums(
            float(err_map[ws_rows].sum()), float(sst_ws[ws_rows].sum())
        ),
        "r2_test_id_within_source": r2_from_sums(
            float(err_id[ws_rows].sum()), float(sst_ws[ws_rows].sum())
        ),
        "n_singleton_source_rows_excluded_ws": int(len(rows_te) - len(ws_rows)),
        "n_val": int(len(val)),
        "per_source": per_source,
        "arrays": {
            "err_map": round6(err_map),
            "err_identity": round6(err_id),
            "sst_pooled": round6(sst_pooled),
            "sst_within_source": round6(sst_ws),
        },
        "arrays_val": {
            "err_map": round6(err_map_val),
            "err_identity": round6(err_id_val),
            "sst_pooled": round6(sst_val),
        },
    }
    return unit, pred_map


# --------------------------------------------------------------------------
# kNN + LODO (post-selection units)
# --------------------------------------------------------------------------


def knn_unit(pred_te, Y_te, rows_te, *, max_n: int, seed: int):
    """Diagnostic kNN retrieval at the selected layer: pooled (capped seeded
    subsample; pool = the subsample's own true v_x) + per-source (each source's
    full test slice as its own pool; sources with n < 10 skipped)."""
    import numpy as np

    from explore_persona_space.analysis.mapping_baselines import knn_retrieval

    out: dict = {"pooled": {}, "per_source": {}, "max_n": max_n, "note": "diagnostic-only"}
    n = len(rows_te)
    rng = np.random.default_rng(seed)
    sub = np.sort(rng.choice(n, size=min(max_n, n), replace=False))
    out["pooled_n_pool"] = int(len(sub))
    for metric in ("euclidean", "cosine"):
        out["pooled"][metric] = knn_retrieval(pred_te[sub], Y_te[sub], metric=metric)
    by_src, _ = _source_stats(rows_te)
    for src, idx in sorted(by_src.items()):
        if len(idx) < 10:
            out["per_source"][src] = {"skipped": f"n={len(idx)} < 10"}
            continue
        idx = np.asarray(idx)
        if len(idx) > max_n:
            idx = np.sort(rng.choice(idx, size=max_n, replace=False))
        out["per_source"][src] = {
            m: knn_retrieval(pred_te[idx], Y_te[idx], metric=m) for m in ("euclidean", "cosine")
        }
    return out


def lodo_unit(X, Y, rows, tr, val, group: str, k_sel: int, lambdas, dev):
    """SERIAL REFERENCE for one LODO fold (retained ONLY for the --phase
    selfcheck equivalence gate — production uses ``lodo_fold_from_shared``,
    which shares one train/val reduction across all folds, #10): fit on
    train-partition rows OUTSIDE the group (lambda selected on val-partition
    rows outside the group), evaluate on ALL rows of the left-out group."""
    import numpy as np

    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    grp = np.asarray([i for i, r in enumerate(rows) if str(r.get("lodo_group")) == group])
    tr_f = np.asarray([i for i in tr if str(rows[i].get("lodo_group")) != group])
    val_f = np.asarray([i for i in val if str(rows[i].get("lodo_group")) != group])
    if len(tr_f) < 2 or len(val_f) < 2 or len(grp) < 2:
        return {
            "group": group,
            "skipped": f"degenerate fold (n_train={len(tr_f)}, n_val={len(val_f)}, "
            f"n_eval={len(grp)})",
        }
    (pred,), meta = ridge_fit_streamed(X, Y, tr_f, val_f, [grp], lambdas, dev)
    pred_id = identity_bias_predict(X[tr_f], Y[tr_f], X[grp])
    return {
        "group": group,
        "hs": k_sel,
        "n_train": int(len(tr_f)),
        "n_eval": int(len(grp)),
        "regime_class": rows[int(grp[0])].get("regime_class"),
        "selected_lambda": meta["selected_lambda"],
        "lambda_grid_edge": meta["lambda_grid_edge"],
        "r2_map": pooled_r2(pred, Y[grp]),
        "r2_id": pooled_r2(pred_id, Y[grp]),
    }


def lodo_shared_reductions(X, Y, tr, val, dev) -> dict:
    """ONE shared train/val-side reduction reused by EVERY LODO fold (#10 —
    the serial path recomputed a full-data Gram per fold; per-fold moments
    are now totals-minus-group subtractions).

    Numerics: X/Y are pre-shifted by the FULL-train mean in fp64 so the
    subtraction is cancellation-safe (fold means of pre-centered data are
    ~0 relative to the data scale); per-fold standardization/centering then
    uses fold-own stats EXACTLY as the serial ``lodo_unit`` reference —
    equivalence asserted in --phase selfcheck."""
    import numpy as np
    import torch

    Xtr = torch.as_tensor(np.asarray(X[tr]), dtype=torch.float64, device=dev)
    cx = Xtr.mean(0)
    Xc_tr = Xtr - cx
    del Xtr
    Ytr = torch.as_tensor(np.asarray(Y[tr]), dtype=torch.float64, device=dev)
    cy = Ytr.mean(0)
    Yc_tr = Ytr - cy
    del Ytr
    Xval = torch.as_tensor(np.asarray(X[val]), dtype=torch.float64, device=dev)
    Xc_val = Xval - cx
    del Xval
    Yval = torch.as_tensor(np.asarray(Y[val]), dtype=torch.float64, device=dev)
    Yc_val = Yval - cy
    del Yval
    return {
        "cx": cx,
        "cy": cy,
        "Xc_tr": Xc_tr,
        "Yc_tr": Yc_tr,
        "Xc_val": Xc_val,
        "Yc_val": Yc_val,
        "G_tr": Xc_tr.T @ Xc_tr,
        "XY_tr": Xc_tr.T @ Yc_tr,
        "Sx_tr": Xc_tr.sum(0),
        "Sy_tr": Yc_tr.sum(0),
        "n_tr": int(Xc_tr.shape[0]),
    }


def lodo_fold_from_shared(shared, X, Y, rows, tr, val, group: str, k_sel: int, lambdas, dev):
    """One LODO fold from the shared reductions — algebraically the SAME fit
    as ``lodo_unit`` (fold-own standardize/center, val-lambda selection via
    the batched quadratic scan, evaluate on ALL rows of the left-out group);
    per-fold cost drops from a full-data Gram to a group-sized one."""
    import numpy as np
    import torch

    from explore_persona_space.analysis.mapping_baselines import identity_bias_predict

    grp = np.asarray([i for i, r in enumerate(rows) if str(r.get("lodo_group")) == group])
    tr_in = np.asarray(
        [j for j, i in enumerate(tr) if str(rows[int(i)].get("lodo_group")) != group], dtype=int
    )
    val_in = np.asarray(
        [j for j, i in enumerate(val) if str(rows[int(i)].get("lodo_group")) != group], dtype=int
    )
    g_in = np.asarray(
        [j for j, i in enumerate(tr) if str(rows[int(i)].get("lodo_group")) == group], dtype=int
    )
    if len(tr_in) < 2 or len(val_in) < 2 or len(grp) < 2:
        return {
            "group": group,
            "skipped": f"degenerate fold (n_train={len(tr_in)}, n_val={len(val_in)}, "
            f"n_eval={len(grp)})",
        }
    device = shared["Xc_tr"].device
    gi = torch.as_tensor(g_in, dtype=torch.long, device=device)
    Xg = shared["Xc_tr"][gi]
    Yg = shared["Yc_tr"][gi]
    n_f = shared["n_tr"] - len(g_in)
    G_f = shared["G_tr"] - Xg.T @ Xg
    XY_f = shared["XY_tr"] - Xg.T @ Yg
    Sx_f = shared["Sx_tr"] - Xg.sum(0)
    Sy_f = shared["Sy_tr"] - Yg.sum(0)
    del Xg, Yg
    mu_f = Sx_f / n_f  # fold train mean (pre-shifted coords)
    ymu_f = Sy_f / n_f
    A_raw = G_f - n_f * torch.outer(mu_f, mu_f)
    del G_f
    var = torch.clamp(torch.diagonal(A_raw), min=0.0) / (n_f - 1)
    sd = torch.sqrt(var) + 1e-9  # == torch.std(ddof=1) + 1e-9 in the reference
    A = A_raw / (sd[:, None] * sd[None, :])
    del A_raw
    s, U = _eigh_robust(A, dev)
    s = torch.clamp(s, min=0.0)
    del A
    UtXtY = U.T @ ((XY_f - torch.outer(Sx_f, ymu_f)) / sd[:, None])
    del XY_f
    vi = torch.as_tensor(val_in, dtype=torch.long, device=device)
    Val_n = (shared["Xc_val"][vi] - mu_f) / sd
    Yv_c = shared["Yc_val"][vi] - ymu_f
    P = Val_n @ U
    del Val_n
    K = P.T @ P
    c_vec = (UtXtY * (P.T @ Yv_c)).sum(1)
    del P
    T = K * (UtXtY @ UtXtY.T)
    del K
    lam_t = torch.as_tensor([float(x) for x in lambdas], dtype=torch.float64, device=device)
    G = 1.0 / (s[None, :] + lam_t[:, None])
    ss_res = float((Yv_c**2).sum()) - 2.0 * (G @ c_vec) + ((G @ T) * G).sum(1)
    del T, G, c_vec, Yv_c
    Yv_own = shared["Yc_val"][vi]
    ss_tot = float(((Yv_own - Yv_own.mean(0)) ** 2).sum())
    vr2 = np.full(len(lambdas), np.nan) if ss_tot < 1e-12 else (1.0 - ss_res / ss_tot).cpu().numpy()
    best_lam, _best_vr2, edge = _select_lambda(vr2, lambdas)
    W = U @ (UtXtY / (s + best_lam)[:, None])
    del U, UtXtY
    Xg_eval = torch.as_tensor(np.asarray(X[grp]), dtype=torch.float64, device=device)
    pred = ((((Xg_eval - shared["cx"]) - mu_f) / sd) @ W + ymu_f + shared["cy"]).cpu().numpy()
    del Xg_eval, W
    tr_f_global = np.asarray([int(tr[j]) for j in tr_in])
    pred_id = identity_bias_predict(X[tr_f_global], Y[tr_f_global], X[grp])
    return {
        "group": group,
        "hs": k_sel,
        "n_train": int(n_f),
        "n_eval": int(len(grp)),
        "regime_class": rows[int(grp[0])].get("regime_class"),
        "selected_lambda": best_lam,
        "lambda_grid_edge": edge,
        "r2_map": pooled_r2(pred, Y[grp]),
        "r2_id": pooled_r2(pred_id, Y[grp]),
    }


def firstpc_unit(pred_te, Y_te, rows_te, hs: int) -> dict:
    """Plan S6 figure (e) inputs for RV-u4: predicted + true v_x projected on
    the TEST slice's own first PC (top eigenvector of the mean-centered TRUE
    v_x covariance; sign fixed so the max-|loading| coordinate is positive).
    Compact committed artifact — the scatter renders without reloading tensors."""
    import numpy as np

    Yte = np.asarray(Y_te, dtype=np.float64)
    mu = Yte.mean(0)
    Yc = Yte - mu
    w, V = np.linalg.eigh(Yc.T @ Yc)
    v = V[:, -1]
    if v[int(np.argmax(np.abs(v)))] < 0:
        v = -v
    tot = float(w.sum())
    return {
        "hs": hs,
        "basis": (
            "top eigenvector of the TEST-slice true v_x covariance (Gram of the "
            "mean-centered Y_test); sign: max-|loading| coordinate positive; "
            "pred projected with the SAME centering + basis"
        ),
        "explained_variance_ratio": float(w[-1] / tot) if tot > 0 else float("nan"),
        "n": int(Yte.shape[0]),
        "context_ids": [str(r["context_id"]) for r in rows_te],
        "source_tags": [r.get("source_tag") for r in rows_te],
        "regime_classes": [r.get("regime_class") for r in rows_te],
        "true_pc1": round6(Yc @ v),
        "pred_pc1": round6((np.asarray(pred_te, dtype=np.float64) - mu) @ v),
    }


# --------------------------------------------------------------------------
# fit phase driver
# --------------------------------------------------------------------------


def model_dirs(args, model_key: str):
    out_root = Path(args.out_root)
    fits_dir = out_root / "fits" / f"model{model_key}"
    percell = (
        Path(args.work_dir) / f"percell_model{model_key}" / ("pilot" if args.pilot else "full")
    )
    return fits_dir, percell


def fit_regime(args, model_key: str, n_rows: int, keys_sha: str) -> dict:
    """StageLedger regime — GENERATING PARAMETERS only (machine-stable, #1336).

    Deliberately EXCLUDES per-unit knobs (#14): ``--layers`` shards,
    ``--skip-lodo``/``--lodo-groups`` subsets and ``--knn-max-n`` are keyed at
    CELL grain (``lodo_{group}``, ``knn_gate{hs}_n{max_n}``), so a subset run
    never invalidates sibling cells AND never marks unfinished work complete."""
    return {
        "phase": "fit",
        "issue": ISSUE,
        "model_key": model_key,
        "tensors_prefix": args.tensors_prefix or DEFAULT_TENSORS_PREFIX[model_key],
        "pilot": bool(args.pilot),
        "split": "pilot-80/20-seed42" if args.pilot else "corpus-p0-70/15/15",
        "lambda_grid": list(LAMBDA_GRID_PARAMS),
        "sst": "pooled+within-source (fixed full-test means)",
        "device": args.device,
        "n_rows": n_rows,
        "chunk_keys_sha16": keys_sha,
    }


def run_fit(args, store=None, model_key: str | None = None, sets_override: dict | None = None):
    """The fit phase for one model. ``store``/``model_key``/``sets_override``
    injectable (selfcheck seams; ``sets_override`` supplies toy candidate sets
    and skips the production MF-J derivation, which is pinned to the real
    28/32-layer models — everything else runs the production path)."""
    import numpy as np

    GC = _gc()
    N779 = _n779()
    model_key = model_key or args.model_key
    if model_key not in MODEL_NAME:
        raise SystemExit(f"--model-key required (A|B), got {model_key!r}")
    if (
        not args.pilot
        and getattr(args, "reliability", None) is None
        and not getattr(args, "allow_missing_reliability", False)
    ):
        raise SystemExit(
            "--reliability <this model's reliability_ceiling.json> is REQUIRED for a "
            "full (non-pilot) fit (#13, MF-E binding: fits_summary embeds the "
            "selected-layer ceilings); --allow-missing-reliability is a "
            "selfcheck/smoke-only escape"
        )
    work = Path(args.work_dir)
    work.mkdir(parents=True, exist_ok=True)
    if store is None:
        prefix = args.tensors_prefix or DEFAULT_TENSORS_PREFIX[model_key]
        store = HfChunkStore(prefix, work, MODEL_HIDDEN[model_key])
    if isinstance(store, HfChunkStore):
        store.verify_complete()  # #11: never fit an unfinished capture
    fits_dir, percell = model_dirs(args, model_key)
    percell.mkdir(parents=True, exist_ok=True)

    rows = store.load_rows()
    captured = store.captured_hs()
    if sets_override is None:
        mfj = assert_mfj(model_key, captured)
        all_set = sorted(captured)
        gate_set = gate_hs_set(model_key, captured)
        h3_set = h3_hs_set(model_key)
    else:
        mfj = {"selfcheck": True, "subset_ok": True}
        all_set = sorted(sets_override["all"])
        gate_set = sorted(sets_override["gate"])
        h3_set = sorted(sets_override["h3"])
    tr, val, te = resolve_splits(rows, pilot=args.pilot)
    rows_te = [rows[int(i)] for i in te]
    rows_val = [rows[int(i)] for i in val]
    keys_sha = sha16(",".join(store.chunk_keys()))
    ledger = GC.StageLedger(
        percell / "ledger.json", fit_regime(args, model_key, len(rows), keys_sha)
    )
    lambdas = N779.LAMBDAS_N50K
    layer_subset = sorted(int(x) for x in args.layers.split(",")) if args.layers else list(all_set)
    unknown = set(layer_subset) - set(all_set)
    if unknown:
        raise RuntimeError(f"--layers {sorted(unknown)} not in captured set {all_set}")
    pending = [k for k in layer_subset if not ledger.is_done(f"L{k:02d}")]
    print(
        f"[fit] model {model_key}: n={len(rows)} (tr={len(tr)}/val={len(val)}/te={len(te)}), "
        f"layers {len(all_set)} captured, {len(pending)} pending "
        f"(gate={gate_set}, h3={h3_set}, pilot={args.pilot})",
        flush=True,
    )
    if pending and isinstance(store, HfChunkStore):
        from explore_persona_space.orchestrate.preflight import assert_out_root_headroom

        assert_out_root_headroom(work, need_gb=4.0, phase=f"fits-model{model_key}")

    t0 = time.time()
    for j, k in enumerate(layer_subset):
        cell = f"L{k:02d}"
        if ledger.is_done(cell):
            continue
        X, Y = store.load_layer(k)
        if X.shape[0] != len(rows):
            raise RuntimeError(f"layer {cell}: {X.shape[0]} rows vs table {len(rows)}")
        unit, _ = fit_layer_unit(X, Y, tr, val, te, rows_te, k, lambdas, args.device)
        del X, Y
        GC.atomic_write_json(percell / f"{cell}.json", unit)
        ledger.mark_done(cell)
        GC.progress(f"fit-{model_key}", j + 1, len(layer_subset), cell, t0)

    done_all = [k for k in all_set if ledger.is_done(f"L{k:02d}")]
    if set(done_all) != set(all_set):
        if args.layers:
            print(
                f"[fit] model {model_key}: layer shard complete "
                f"({len(done_all)}/{len(all_set)} layers done); rerun without --layers "
                "to finish selection + LODO + assembly",
                flush=True,
            )
            return {"partial": True, "layers_done": done_all}
        raise RuntimeError(f"missing per-layer cells after loop: {set(all_set) - set(done_all)}")

    units = {k: json.loads((percell / f"L{k:02d}.json").read_text()) for k in all_set}
    val_r2 = {k: units[k]["fit_meta"]["val_r2_at_selected"] for k in all_set}

    def _select(cands: list[int]) -> int:
        finite = [k for k in cands if math.isfinite(val_r2[k])]
        if not finite:
            raise RuntimeError(f"no finite validation R^2 in candidate set {cands}")
        return min(finite, key=lambda k: (-val_r2[k], k))

    sel_gate = _select(gate_set)
    sel_h3 = _select(h3_set)

    # Post-selection units, each its OWN param-keyed cell (#14 — no aggregate
    # "post" cell: a subset run can never mark unfinished sibling work done).
    knn_cell = f"knn_gate{sel_gate}_n{args.knn_max_n}"
    post_path = percell / "post.json"
    knn_ok = ledger.is_done(knn_cell) and post_path.exists()
    if knn_ok:
        prior = json.loads(post_path.read_text())
        knn_ok = (
            prior.get("gate_hs") == sel_gate and prior.get("knn", {}).get("max_n") == args.knn_max_n
        )
    firstpc_cell = f"firstpc_gate{sel_gate}"
    firstpc_path = percell / "firstpc.json"
    firstpc_ok = ledger.is_done(firstpc_cell) and firstpc_path.exists()
    if firstpc_ok:
        firstpc_ok = json.loads(firstpc_path.read_text()).get("hs") == sel_gate

    groups_all = sorted({str(r.get("lodo_group")) for r in rows})
    lodo_groups = list(groups_all)
    if args.lodo_groups:
        want = set(args.lodo_groups.split(","))
        unknown_g = want - set(groups_all)
        if unknown_g:
            raise RuntimeError(f"--lodo-groups unknown: {sorted(unknown_g)}")
        lodo_groups = sorted(want)
    lodo_pending = (
        [] if args.skip_lodo else [g for g in lodo_groups if not ledger.is_done(f"lodo_{g}")]
    )

    if (not knn_ok) or (not firstpc_ok) or lodo_pending:
        X, Y = store.load_layer(sel_gate)
        (pred_te,), _ = ridge_fit_streamed(X, Y, tr, val, [te], lambdas, args.device)
        if not knn_ok:
            knn = knn_unit(
                pred_te,
                np.asarray(Y[te], dtype=np.float64),
                rows_te,
                max_n=args.knn_max_n,
                seed=BOOT_SEED,
            )
            GC.atomic_write_json(post_path, {"gate_hs": sel_gate, "knn": knn})
            ledger.mark_done(knn_cell)
        if not firstpc_ok:
            fpc = {
                "meta": GC.run_metadata(
                    {
                        "artifact": "firstpc_scatter",
                        "model_key": model_key,
                        "model": MODEL_NAME[model_key],
                        "pilot": bool(args.pilot),
                    }
                ),
                **firstpc_unit(pred_te, np.asarray(Y[te], dtype=np.float64), rows_te, sel_gate),
            }
            GC.atomic_write_json(firstpc_path, fpc)
            ledger.mark_done(firstpc_cell)
        if lodo_pending:
            shared = lodo_shared_reductions(X, Y, tr, val, args.device)
            t1 = time.time()
            for gi, g in enumerate(lodo_pending):
                fold = lodo_fold_from_shared(
                    shared, X, Y, rows, tr, val, g, sel_gate, lambdas, args.device
                )
                GC.atomic_write_json(percell / f"lodo_{g}.json", fold)
                ledger.mark_done(f"lodo_{g}")
                GC.progress(f"lodo-{model_key}", gi + 1, len(lodo_pending), g, t1)
            del shared
        del X, Y

    summary = assemble_model(
        args,
        model_key,
        percell,
        fits_dir,
        rows_te,
        units,
        mfj,
        {
            "all": all_set,
            "gate": gate_set,
            "h3": h3_set,
            "selected_gate_hs": sel_gate,
            "selected_h3_hs": sel_h3,
        },
        tr=tr,
        val=val,
        te=te,
        rows_val=rows_val,
        lodo_expected=groups_all,
    )
    publish = getattr(args, "publish", None)
    if publish and publish != "none":
        publish_artifacts(
            [
                fits_dir / "fits_summary.json",
                fits_dir / "percontext_recon.json",
                fits_dir / "firstpc_scatter.json",
            ],
            Path(args.out_root),
            publish=publish,
            hf_prefix=getattr(args, "publish_prefix", PUBLISH_EVAL_MIRROR),
        )
    return summary


# --------------------------------------------------------------------------
# Assembly (fits_summary.json + percontext_recon.json) + G1 pilot gate
# --------------------------------------------------------------------------


def assemble_model(
    args,
    model_key,
    percell,
    fits_dir,
    rows_te,
    units,
    mfj,
    sel,
    *,
    tr,
    val,
    te,
    rows_val,
    lodo_expected,
):
    """Compose the committed deliverables (percontext_recon, fits_summary,
    firstpc_scatter) from percell checkpoints."""
    GC = _gc()
    n_layers = MODEL_N_LAYERS[model_key]
    all_set = sel["all"]

    def _ctx(rows_slice):
        return [
            {
                "context_id": r["context_id"],
                "source_tag": r.get("source_tag"),
                "regime_class": r.get("regime_class"),
                "lodo_group": r.get("lodo_group"),
            }
            for r in rows_slice
        ]

    recon = {
        "meta": GC.run_metadata(
            {
                "artifact": "percontext_recon",
                "model_key": model_key,
                "model": MODEL_NAME[model_key],
                "pilot": bool(args.pilot),
                "sst_convention": (
                    "sst_pooled: fixed full-test-slice mean; sst_within_source: fixed "
                    "per-source test-slice mean (0.0 for singleton-source rows — "
                    "excluded from within-source statistics); arms: err_map (pooled "
                    "val-lambda ridge), err_identity (identity+learned-bias); all "
                    "scalars are squared L2 residual norms per context. layers_val "
                    "holds the SAME arms on the VALIDATION slice (fixed full-val "
                    "mean) — the #9 selection partition for selection-inherited CIs"
                ),
                "layer_indexing": "hs k = hidden_states[k] = output of decoder block k-1",
            }
        ),
        "candidate_sets": {k: v for k, v in sel.items() if k in ("all", "gate", "h3")},
        "selected": {
            "gate_hs": sel["selected_gate_hs"],
            "h3_hs": sel["selected_h3_hs"],
        },
        "n_test": len(rows_te),
        "contexts": _ctx(rows_te),
        "layers": {f"L{k:02d}": units[k]["arrays"] for k in all_set},
        "n_val": len(rows_val),
        "contexts_val": _ctx(rows_val),
        "layers_val": {f"L{k:02d}": units[k]["arrays_val"] for k in all_set},
    }
    fits_dir.mkdir(parents=True, exist_ok=True)
    write_artifact_json(fits_dir / "percontext_recon.json", recon)
    firstpc_path = percell / "firstpc.json"
    if not firstpc_path.exists():
        raise RuntimeError(
            f"missing percell firstpc checkpoint {firstpc_path} — the first-PC unit "
            "must run before assembly (plan S6 figure (e) deliverable)"
        )
    write_artifact_json(fits_dir / "firstpc_scatter.json", json.loads(firstpc_path.read_text()))

    post = (
        json.loads((percell / "post.json").read_text()) if (percell / "post.json").exists() else {}
    )
    lodo = {}
    for p in sorted(percell.glob("lodo_*.json")):
        d = json.loads(p.read_text())
        lodo[d["group"]] = d
    layer_table = []
    for k in all_set:
        u = units[k]
        layer_table.append(
            {
                "hs": k,
                "block_0idx": k - 1,
                "rel_depth": k / n_layers,
                **{
                    key: u["fit_meta"][key]
                    for key in (
                        "n_train",
                        "d",
                        "selected_lambda",
                        "val_r2_at_selected",
                        "lambda_grid_edge",
                        "numerical_rank",
                        "effective_dof",
                    )
                },
                "r2_test_map_pooled": u["r2_test_map_pooled"],
                "r2_test_id_pooled": u["r2_test_id_pooled"],
                "r2_test_map_within_source": u["r2_test_map_within_source"],
                "r2_test_id_within_source": u["r2_test_id_within_source"],
            }
        )
    sel_u = units[sel["selected_gate_hs"]]
    # Reliability-ceiling embed (#13): fits_summary carries the selected-layer
    # ceilings whenever the per-model reliability artifact is supplied
    # (REQUIRED on full fits; None only under the selfcheck/pilot escapes).
    rel_path = getattr(args, "reliability", None)
    reliability_block = None
    if rel_path:
        reliability_block = {
            "source": str(rel_path),
            "at_gate_hs": _ceiling_at(
                rel_path, sel["selected_gate_hs"], expect_model_key=model_key
            ),
            "at_h3_hs": _ceiling_at(rel_path, sel["selected_h3_hs"], expect_model_key=model_key),
        }
    # LODO completeness is recorded HONESTLY (#14): a subset run reports
    # partial + the pending groups, never a silently-complete aggregate.
    expected_groups = sorted(set(lodo_expected))
    done_groups = sorted(lodo)
    if getattr(args, "skip_lodo", False):
        lodo_status = "skipped-by-flag"
    elif set(done_groups) >= set(expected_groups):
        lodo_status = "complete"
    else:
        lodo_status = f"partial ({len(done_groups)}/{len(expected_groups)})"
        print(
            f"[assemble] model {model_key}: LODO PARTIAL — pending groups "
            f"{sorted(set(expected_groups) - set(done_groups))} (rerun without "
            "--lodo-groups to finish)",
            flush=True,
        )
    summary = {
        "meta": GC.run_metadata(
            {
                "artifact": "fits_summary",
                "model_key": model_key,
                "model": MODEL_NAME[model_key],
                "pilot": bool(args.pilot),
                "pilot_val_is_test": bool(args.pilot),
                "splits": {"n_train": int(len(tr)), "n_val": int(len(val)), "n_test": int(len(te))},
                "sst_centering": "within-source PRIMARY; pooled alongside (plan S6)",
            }
        ),
        "mfj_assert": mfj,
        "candidate_sets": {k: v for k, v in sel.items() if k in ("all", "gate", "h3")},
        "selected": {"gate_hs": sel["selected_gate_hs"], "h3_hs": sel["selected_h3_hs"]},
        "layers": layer_table,
        "per_source_at_gate_layer": sel_u["per_source"],
        "r2_at_gate_layer": {
            "map_pooled": sel_u["r2_test_map_pooled"],
            "map_within_source": sel_u["r2_test_map_within_source"],
            "identity_pooled": sel_u["r2_test_id_pooled"],
            "identity_within_source": sel_u["r2_test_id_within_source"],
        },
        "r2_at_h3_layer": {
            "map_pooled": units[sel["selected_h3_hs"]]["r2_test_map_pooled"],
        },
        "knn": post.get("knn"),
        "lodo": lodo,
        "lodo_status": lodo_status,
        "lodo_pending": sorted(set(expected_groups) - set(done_groups)),
        "reliability_ceiling": reliability_block,
        "plan_deviation_fit_core": (
            "streamed twin of issue779 fit_ridge_primal (batched quadratic-form "
            "lambda scan; shared-reduction LODO); reference materializes ~31 GB "
            "of all-lambda eval predictions at production shape — see module "
            "docstring + --phase unitwall for the measured per-unit basis (#10)"
        ),
    }
    if args.pilot and model_key == "A":
        summary["g1"] = g1_gate_block(units, sel, rows_te, args)
    write_artifact_json(fits_dir / "fits_summary.json", summary)
    print(
        f"[assemble] model {model_key}: gate hs={sel['selected_gate_hs']} "
        f"(R2 pooled={sel_u['r2_test_map_pooled']:.4f}, "
        f"id={sel_u['r2_test_id_pooled']:.4f}); wrote {fits_dir}",
        flush=True,
    )
    return summary


def g1_gate_block(units, sel, rows_te, args) -> dict:
    """G1 (MF-B): pooled best-layer held-out R^2 floor + the beats-baseline
    paired bootstrap (selection-inherited over the gate set, frozen alongside).
    The pilot's A_pass slice is the POOLED held-out set (the 1k pilot's
    ordinary-only subset is small); recorded as ``a_pass_slice``. The best
    layer is the VAL-selected gate layer (#9 — the production ``_select``
    winner; in pilot mode val == heldout by construction, ``pilot_val_is_test``)
    and its held-out pooled R^2 is what the floor reads."""
    import numpy as np

    mats_te = _recon_arrays_from_units(units, sel["all"])
    mats_val = _recon_arrays_from_units(units, sel["all"], arrays_key="arrays_val")
    best_hs = sel["selected_gate_hs"]
    r2_best = units[best_hs]["r2_test_map_pooled"]
    boot = paired_delta_bootstrap(
        mats_te,
        mats_val,
        cand=sel["gate"],
        frozen_hs=best_hs,
        member_mask=np.ones(len(rows_te), dtype=bool),
        draws=args.boot_draws,
        seed=BOOT_SEED,
    )
    a_pass = bool(boot["inherited_ci"][0] > 0.0)
    verdict = "PROCEED" if (r2_best >= G1_R2_FLOOR and a_pass) else "FAIL"
    m = units[best_hs]["fit_meta"]
    return {
        "r2_floor": G1_R2_FLOOR,
        "best_layer_hs": best_hs,
        "r2_best_layer_heldout_pooled": r2_best,
        "a_pass_pilot": a_pass,
        "a_pass_slice": "pooled-heldout",
        "paired_delta_bootstrap": boot,
        "fit_diagnostics_at_best_layer": {
            key: m[key]
            for key in (
                "n_train",
                "d",
                "selected_lambda",
                "lambda_grid_edge",
                "numerical_rank",
                "effective_dof",
            )
        },
        "verdict": verdict,
    }


# --------------------------------------------------------------------------
# Bootstrap machinery (selection-inherited + frozen; subset-sum GEMM batched)
# --------------------------------------------------------------------------


def _recon_arrays_from_units(units: dict, layer_set: list[int], *, arrays_key="arrays") -> dict:
    """(n, L) numpy views of err/sst per arm, column order == sorted layer_set.
    ``arrays_key='arrays_val'`` yields the VALIDATION-slice matrices (#9)."""
    import numpy as np

    ks = sorted(layer_set)
    return {
        "hs": ks,
        "err_map": np.stack(
            [np.asarray(units[k][arrays_key]["err_map"], dtype=np.float64) for k in ks], axis=1
        ),
        "err_id": np.stack(
            [np.asarray(units[k][arrays_key]["err_identity"], dtype=np.float64) for k in ks],
            axis=1,
        ),
        "sst": np.stack(
            [np.asarray(units[k][arrays_key]["sst_pooled"], dtype=np.float64) for k in ks], axis=1
        ),
    }


def recon_arrays_from_file(recon: dict, layer_set: list[int]) -> dict:
    """TEST-slice matrices from a loaded percontext_recon dict."""
    units = {k: {"arrays": recon["layers"][f"L{k:02d}"]} for k in layer_set}
    return _recon_arrays_from_units(units, layer_set)


def recon_arrays_val_from_file(recon: dict, layer_set: list[int]) -> dict:
    """VALIDATION-slice matrices from a loaded percontext_recon dict (#9)."""
    units = {k: {"arrays": recon["layers_val"][f"L{k:02d}"]} for k in layer_set}
    return _recon_arrays_from_units(units, layer_set)


def bootstrap_counts(n: int, draws: int, rng):
    """(draws, n) float32 bootstrap multiplicity matrix from ``rng`` (subset-sum
    GEMM form — per-draw statistics become ONE counts @ matrix product).
    Sequential draws from ONE generator keep the VAL-selection and
    TEST-evaluation resamples independent (#9; draw ORDER is part of the
    seeded contract — documented at each call site)."""
    import numpy as np

    return rng.multinomial(n, np.full(n, 1.0 / n), size=draws).astype(np.float32)


def _val_selected_cols(mats_val, cand_cols, counts_val):
    """Per-draw layer selection on the VAL resample (#9, MF-D): argmax of the
    resample's own pooled map R^2 over the candidate columns — mirroring the
    production ``_select`` rule (candidate columns are hs-sorted, and argmax
    returns the FIRST max, so ties break to the smaller hs)."""
    import numpy as np

    ev = counts_val @ mats_val["err_map"][:, cand_cols]  # (B, |cand|)
    sv = counts_val @ mats_val["sst"][:, cand_cols]
    with np.errstate(divide="ignore", invalid="ignore"):
        r2_val_b = 1.0 - ev / sv
    return np.nanargmax(r2_val_b, axis=1)


def _sel_histogram(sel_cols, cand_sorted) -> dict:
    return {str(cand_sorted[c]): int((sel_cols == c).sum()) for c in sorted(set(sel_cols.tolist()))}


def paired_delta_bootstrap(mats_te, mats_val, *, cand, frozen_hs, member_mask, draws, seed):
    """Paired map-vs-identity contrast (A_pass/B_pass predicate machinery).

    Statistic: mean_i Delta_i over the member slice (Delta_i = err_id - err_map
    at the layer, TEST rows). SELECTION-INHERITED (#9, MF-D): each draw
    re-selects the layer on an INDEPENDENT resample of the VALIDATION rows
    (production ``_select`` semantics) and evaluates on a separately drawn
    TEST resample — selection never sees the scored partition. Draw order per
    seed: counts_val FIRST, then counts_te. The frozen-at-``frozen_hs`` CI is
    reported ALONGSIDE, labeled, never alone."""
    import numpy as np

    hs = mats_te["hs"]
    cand_sorted = sorted(cand)
    cand_cols = [hs.index(k) for k in cand_sorted]
    n_te = mats_te["err_map"].shape[0]
    n_val = mats_val["err_map"].shape[0]
    rng = np.random.default_rng(seed)
    counts_val = bootstrap_counts(n_val, draws, rng)
    counts_te = bootstrap_counts(n_te, draws, rng)
    sel_cols = _val_selected_cols(mats_val, cand_cols, counts_val)
    delta = mats_te["err_id"] - mats_te["err_map"]  # (n_te, L)
    m = member_mask.astype(np.float32)
    member_counts = counts_te * m[None, :]
    denom = member_counts.sum(1)
    num_by_col = member_counts @ delta[:, cand_cols]  # (B, |cand|)
    with np.errstate(divide="ignore", invalid="ignore"):
        delta_by_col = num_by_col / denom[:, None]
    inherited = delta_by_col[np.arange(draws), sel_cols]
    frozen_col = cand_cols.index(hs.index(frozen_hs))
    frozen = delta_by_col[:, frozen_col]
    idx = np.asarray(member_mask)
    point = float(
        (
            mats_te["err_id"][idx, hs.index(frozen_hs)]
            - mats_te["err_map"][idx, hs.index(frozen_hs)]
        ).mean()
    )
    pct = [2.5, 97.5]
    return {
        "point_delta_mean_at_frozen": point,
        "frozen_hs": frozen_hs,
        "candidate_hs": cand_sorted,
        "n_members": int(idx.sum()),
        "draws": draws,
        "seed": seed,
        "selection_basis": (
            "per-draw argmax of VAL-resample pooled map R^2 over the candidate set "
            "(independent counts matrix; #9 MF-D)"
        ),
        "selected_hs_counts": _sel_histogram(sel_cols, cand_sorted),
        "inherited_ci": [float(x) for x in np.nanpercentile(inherited, pct)],
        "frozen_ci_labeled_frozen_at_selected": [float(x) for x in np.nanpercentile(frozen, pct)],
    }


def ni_bootstrap(
    mats_a_te,
    mats_a_val,
    ids_a,
    mats_b_te,
    mats_b_val,
    ids_b,
    *,
    cand_a,
    cand_b,
    frozen_a,
    frozen_b,
    draws,
    seed,
):
    """Selection-inherited CI on (B_best - A_best) over the SHARED test contexts.

    Both models' TEST rows are aligned on the shared context_id set (the corpus
    is shared; per-model drops can differ) and evaluated under ONE shared
    paired test counts matrix. Selection (#9, MF-D): each model's layer is
    re-selected per draw on an independent resample of that model's OWN
    VALIDATION rows (production ``_select`` semantics). Draw order per seed:
    counts_val_A, counts_val_B, then the shared counts_te."""
    import numpy as np

    shared = sorted(set(ids_a) & set(ids_b))
    if len(shared) < 10:
        raise RuntimeError(f"only {len(shared)} shared test contexts across models")
    pos_a = {c: i for i, c in enumerate(ids_a)}
    pos_b = {c: i for i, c in enumerate(ids_b)}
    ia = np.asarray([pos_a[c] for c in shared])
    ib = np.asarray([pos_b[c] for c in shared])
    rng = np.random.default_rng(seed)
    counts_val_a = bootstrap_counts(mats_a_val["err_map"].shape[0], draws, rng)
    counts_val_b = bootstrap_counts(mats_b_val["err_map"].shape[0], draws, rng)
    counts_te = bootstrap_counts(len(shared), draws, rng)

    def _model_r2(mats_te, mats_val, rows_idx, cand, frozen_hs, counts_val):
        cand_sorted = sorted(cand)
        cols = [mats_te["hs"].index(k) for k in cand_sorted]
        sel_cols = _val_selected_cols(mats_val, cols, counts_val)
        e = mats_te["err_map"][rows_idx][:, cols]
        s = mats_te["sst"][rows_idx][:, cols]
        with np.errstate(divide="ignore", invalid="ignore"):
            r2_b = 1.0 - (counts_te @ e) / (counts_te @ s)
        inherited = r2_b[np.arange(draws), sel_cols]
        frozen = r2_b[:, cols.index(mats_te["hs"].index(frozen_hs))]
        return inherited, frozen, _sel_histogram(sel_cols, cand_sorted)

    inh_a, frz_a, hist_a = _model_r2(mats_a_te, mats_a_val, ia, cand_a, frozen_a, counts_val_a)
    inh_b, frz_b, hist_b = _model_r2(mats_b_te, mats_b_val, ib, cand_b, frozen_b, counts_val_b)
    pct = [2.5, 97.5]
    return {
        "n_shared_contexts": len(shared),
        "draws": draws,
        "seed": seed,
        "selection_basis": (
            "per-draw argmax of each model's OWN VAL-resample pooled map R^2 over "
            "its H3 candidate set (independent counts matrices; #9 MF-D); "
            "evaluation on ONE shared paired test counts matrix"
        ),
        "selected_hs_counts": {"A": hist_a, "B": hist_b},
        "inherited_ci_diff": [float(x) for x in np.nanpercentile(inh_b - inh_a, pct)],
        "frozen_ci_diff_labeled_frozen_at_selected": [
            float(x) for x in np.nanpercentile(frz_b - frz_a, pct)
        ],
    }


def h2_contrast(recon: dict, gate_hs: int, *, draws: int, seed: int) -> dict:
    """H2: equal-source-weighted regime-class mean per-source R^2 (within-source
    centering PRIMARY), ordinary minus weird, source-level bootstrap CI."""
    import numpy as np

    hs_key = f"L{gate_hs:02d}"
    arr = recon["layers"][hs_key]
    err = np.asarray(arr["err_map"], dtype=np.float64)
    sst = np.asarray(arr["sst_within_source"], dtype=np.float64)
    src = np.asarray([c["source_tag"] for c in recon["contexts"]])
    cls = {c["source_tag"]: c["regime_class"] for c in recon["contexts"]}
    per_src = {}
    for s in sorted(set(src)):
        m = src == s
        if m.sum() < 2:
            continue
        per_src[s] = r2_from_sums(float(err[m].sum()), float(sst[m].sum()))
    classes: dict[str, list[float]] = {}
    for s, r2 in per_src.items():
        if math.isfinite(r2):
            classes.setdefault(str(cls[s]), []).append(r2)
    means = {c: float(np.mean(v)) for c, v in classes.items()}
    out = {
        "gate_hs": gate_hs,
        "weighting": "equal-source (each source_tag counts once)",
        "centering": "within-source",
        "per_source_r2": per_src,
        "class_means": means,
        "n_sources_per_class": {c: len(v) for c, v in classes.items()},
    }
    if "ordinary" in classes and "weird" in classes:
        rng = np.random.default_rng(seed)
        o, w = np.asarray(classes["ordinary"]), np.asarray(classes["weird"])
        diffs = [
            float(rng.choice(o, len(o)).mean() - rng.choice(w, len(w)).mean()) for _ in range(draws)
        ]
        point = means["ordinary"] - means["weird"]
        out["ordinary_minus_weird"] = {
            "point": point,
            "source_level_bootstrap_ci": [
                float(x) for x in np.percentile(np.asarray(diffs), [2.5, 97.5])
            ],
            "margin": H2_MARGIN,
            "descriptive_verdict": bool(point >= H2_MARGIN),
        }
    else:
        out["ordinary_minus_weird"] = {"skipped": "missing ordinary or weird class"}
    return out


# --------------------------------------------------------------------------
# decide phase (registered MF-C truth table)
# --------------------------------------------------------------------------


def decide_verdict(
    a_pass: bool, b_pass: bool, ni_lo: float, ni_hi: float, margin: float = NI_MARGIN
) -> str:
    """The registered decision function (plan S3, byte-semantics; MF-C).

    Disjoint + exhaustive: NOT A_pass -> Inconclusive (instrument voided);
    A_pass & NOT B_pass -> Fails-to-replicate; A_pass & B_pass: CI lower bound
    > -margin -> Replicates; CI wholly below -margin -> Fails-to-replicate;
    CI spans -margin -> Inconclusive."""
    if not a_pass:
        return "Inconclusive"
    if not b_pass:
        return "Fails-to-replicate"
    if ni_lo > -margin:
        return "Replicates"
    if ni_hi < -margin:
        return "Fails-to-replicate"
    return "Inconclusive"


def _load_model_artifacts(fits_dir: Path):
    summary = json.loads((fits_dir / "fits_summary.json").read_text())
    recon = json.loads((fits_dir / "percontext_recon.json").read_text())
    return summary, recon


def _ceiling_at(reliability_path: str | None, hs: int, *, expect_model_key: str | None = None):
    """Reliability ceiling at hs from a reliability_ceiling.json (MF-E), or None.
    ``expect_model_key`` fails loud on a cross-model artifact mixup."""
    if not reliability_path:
        return None
    doc = json.loads(Path(reliability_path).read_text())
    if expect_model_key is not None:
        got = doc.get("meta", {}).get("model_key")
        if got != expect_model_key:
            raise RuntimeError(
                f"{reliability_path}: model_key {got!r} != expected {expect_model_key!r} "
                "— wrong model's reliability artifact (#13)"
            )
    layer = doc.get("per_layer", {}).get(f"L{hs:02d}")
    if layer is None:
        raise RuntimeError(f"{reliability_path} has no layer L{hs:02d}")
    return {
        "hs": hs,
        "ceiling_pooled": layer["ceiling_pooled"],
        "ceiling_within_source": layer.get("ceiling_within_source"),
        "source": reliability_path,
    }


# --------------------------------------------------------------------------
# Durable publish (#12): HF data-repo mirror + git commit/push on the branch
# --------------------------------------------------------------------------


def _git_publish(paths: list[Path], repo: Path) -> None:
    """Commit + push the artifact paths on the current issue branch. LOUD
    degrade when no git checkout exists (SLURM rsync lanes — the HF mirror is
    the durable copy there); REFUSES to commit from a checkout on main."""
    import subprocess

    env = {**os.environ}
    probe = subprocess.run(
        ["git", "-C", str(repo), "rev-parse", "--abbrev-ref", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
        env=env,
    )
    if probe.returncode != 0:
        print(
            f"[publish] LOUD DEGRADE: no git checkout at {repo} — git leg skipped "
            "(git-less lane; the verified HF mirror is the durable copy; commit "
            "eval_results/issue_2502 from the VM checkout)",
            flush=True,
        )
        return
    branch = probe.stdout.strip()
    if branch == "main":
        raise RuntimeError(
            f"publish: refusing to commit experiment artifacts from a checkout on 'main' "
            f"(expected issue-{ISSUE}); run from the issue worktree/pod clone"
        )
    rels = []
    for p in paths:
        try:
            rels.append(str(Path(p).resolve().relative_to(repo)))
        except ValueError:
            print(
                f"[publish] LOUD DEGRADE: {p} lies outside the git checkout {repo} — "
                "git leg skipped for it (HF mirror is the durable copy)",
                flush=True,
            )
    if not rels:
        return
    subprocess.run(["git", "-C", str(repo), "add", "--", *rels], check=True, env=env)
    st = subprocess.run(
        ["git", "-C", str(repo), "status", "--porcelain", "--", *rels],
        capture_output=True,
        text=True,
        check=True,
        env=env,
    )
    if st.stdout.strip():
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "commit",
                "-m",
                f"issue {ISSUE} P4: publish fits/reliability artifacts",
                "--",
                *rels,
            ],
            check=True,
            env=env,
        )
    else:
        print("[publish] git: artifacts already committed — nothing new", flush=True)
    push = subprocess.run(["git", "-C", str(repo), "push", "origin", branch], check=False, env=env)
    if push.returncode != 0:
        pull = subprocess.run(
            ["git", "-C", str(repo), "pull", "--rebase", "origin", branch], check=False, env=env
        )
        push2 = subprocess.run(
            ["git", "-C", str(repo), "push", "origin", branch], check=False, env=env
        )
        if pull.returncode != 0 or push2.returncode != 0:
            raise RuntimeError(
                f"publish: git push to {branch} failed after one rebase retry — resolve "
                "the branch state and re-run (the HF mirror is already durable)"
            )
    print(f"[publish] git: committed+pushed {len(rels)} artifact(s) on {branch}", flush=True)


def publish_artifacts(paths, out_root: Path, *, publish: str, hf_prefix: str) -> None:
    """#12 durable publish of committed JSON artifacts on the NORMAL exit path.

    HF leg (mandatory under any non-none mode with 'hf'): per-file verified
    upload to {hf_prefix}/{path relative to out_root}. git leg ('git' modes):
    commit + push on the issue branch, LOUD degrade on git-less lanes.
    publish='none' is the smoke/selfcheck escape (artifacts stay local)."""
    GC = _gc()
    if publish == "none":
        print(
            f"[publish] publish=none — {len(list(paths))} artifact(s) left local-only", flush=True
        )
        return
    out_root = Path(out_root).resolve()
    resolved = []
    for p in paths:
        p = Path(p).resolve()
        if not p.is_file():
            raise RuntimeError(f"publish: missing artifact {p} — refusing a partial publish")
        resolved.append((p, p.relative_to(out_root).as_posix()))
    if publish in ("hf", "hf+git"):
        for p, rel in resolved:  # UPLOAD_LOOP_EXEMPT: <=4 small JSONs, each verified
            GC.upload_single_file(p, f"{hf_prefix}/{rel}")
    if publish in ("git", "hf+git"):
        _git_publish([p for p, _rel in resolved], _REPO_ROOT)


def run_decide(args) -> dict:
    """Compose the registered decision from both models' assembled artifacts.

    #13: BOTH per-model reliability ceilings are REQUIRED (fail nonzero) —
    the H3 verdict is stated relative to them; --allow-missing-reliability is
    a selfcheck/smoke-only escape. #12: decision.json publishes durably on the
    normal exit path and the fits/.p4_done sentinel is written LAST."""
    import numpy as np

    GC = _gc()
    if not getattr(args, "allow_missing_reliability", False):
        missing = [
            flag
            for flag, v in (
                ("--reliability-a", args.reliability_a),
                ("--reliability-b", args.reliability_b),
            )
            if not v
        ]
        if missing:
            raise SystemExit(
                f"--phase decide requires {', '.join(missing)} (MF-E/#13: the H3 verdict "
                "is stated relative to the per-model reliability ceilings); "
                "--allow-missing-reliability is a selfcheck/smoke-only escape"
            )
    out_root = Path(args.out_root)
    sum_a, rec_a = _load_model_artifacts(out_root / "fits" / "modelA")
    sum_b, rec_b = _load_model_artifacts(out_root / "fits" / "modelB")

    gates = {}
    passes = {}
    for key, summary, recon in (("A", sum_a, rec_a), ("B", sum_b, rec_b)):
        gate_set = summary["candidate_sets"]["gate"]
        gate_hs = summary["selected"]["gate_hs"]
        mats = recon_arrays_from_file(recon, recon["candidate_sets"]["all"])
        mats_val = recon_arrays_val_from_file(recon, recon["candidate_sets"]["all"])
        member = np.asarray(
            [c["regime_class"] == "ordinary" for c in recon["contexts"]], dtype=bool
        )
        if member.sum() < 10:
            raise RuntimeError(
                f"model {key}: only {int(member.sum())} ordinary test contexts — "
                "H1 primary split degenerate"
            )
        boot = paired_delta_bootstrap(
            mats,
            mats_val,
            cand=gate_set,
            frozen_hs=gate_hs,
            member_mask=member,
            draws=args.boot_draws,
            seed=BOOT_SEED + (0 if key == "A" else 1),
        )
        passes[key] = bool(boot["inherited_ci"][0] > 0.0)
        gates[key] = {
            "h1_slice": "regime_class == ordinary (test partition)",
            "gate_hs": gate_hs,
            "predicate": f"{key}_pass: inherited paired-bootstrap CI on mean "
            "(err_identity - err_map) excludes 0 on the positive side",
            "value": passes[key],
            **boot,
        }

    ni = ni_bootstrap(
        recon_arrays_from_file(rec_a, rec_a["candidate_sets"]["all"]),
        recon_arrays_val_from_file(rec_a, rec_a["candidate_sets"]["all"]),
        [c["context_id"] for c in rec_a["contexts"]],
        recon_arrays_from_file(rec_b, rec_b["candidate_sets"]["all"]),
        recon_arrays_val_from_file(rec_b, rec_b["candidate_sets"]["all"]),
        [c["context_id"] for c in rec_b["contexts"]],
        cand_a=sum_a["candidate_sets"]["h3"],
        cand_b=sum_b["candidate_sets"]["h3"],
        frozen_a=sum_a["selected"]["h3_hs"],
        frozen_b=sum_b["selected"]["h3_hs"],
        draws=args.boot_draws,
        seed=BOOT_SEED + 2,
    )
    a_best = sum_a["r2_at_h3_layer"]["map_pooled"]
    b_best = sum_b["r2_at_h3_layer"]["map_pooled"]
    ni_lo, ni_hi = ni["inherited_ci_diff"]
    verdict = decide_verdict(passes["A"], passes["B"], ni_lo, ni_hi)

    ceil_a = _ceiling_at(args.reliability_a, sum_a["selected"]["h3_hs"], expect_model_key="A")
    ceil_b = _ceiling_at(args.reliability_b, sum_b["selected"]["h3_hs"], expect_model_key="B")
    decision = {
        "meta": GC.run_metadata({"artifact": "decision", "boot_draws": args.boot_draws}),
        "truth_table": (
            "Replicates <=> A_pass AND B_pass AND ni_lo > -0.10; "
            "Fails-to-replicate <=> A_pass AND ((ni_hi < -0.10) OR NOT B_pass); "
            "Inconclusive <=> NOT A_pass OR (A_pass AND B_pass AND CI spans -0.10)"
        ),
        "a_pass": passes["A"],
        "b_pass": passes["B"],
        "gates": gates,
        "h3": {
            "a_best_r2_at_h3_layer": a_best,
            "b_best_r2_at_h3_layer": b_best,
            "diff_point_b_minus_a": b_best - a_best,
            "ni_margin": NI_MARGIN,
            **ni,
        },
        "reliability_conditioning": {
            "note": (
                "MF-E: the H3 verdict is STATED RELATIVE to the per-model "
                "answer-vector reliability ceilings — a lower B_best with a "
                "commensurately lower ceil_B is not a weaker map"
            ),
            "ceil_A_at_h3_layer": ceil_a,
            "ceil_B_at_h3_layer": ceil_b,
            "a_best_over_ceiling": (
                a_best / ceil_a["ceiling_pooled"] if ceil_a and ceil_a["ceiling_pooled"] else None
            ),
            "b_best_over_ceiling": (
                b_best / ceil_b["ceiling_pooled"] if ceil_b and ceil_b["ceiling_pooled"] else None
            ),
        },
        "h2": {
            "A": h2_contrast(
                rec_a, sum_a["selected"]["gate_hs"], draws=args.boot_draws, seed=BOOT_SEED + 3
            ),
            "B": h2_contrast(
                rec_b, sum_b["selected"]["gate_hs"], draws=args.boot_draws, seed=BOOT_SEED + 4
            ),
        },
        "verdict": verdict,
    }
    decision_path = out_root / "fits" / "decision.json"
    write_artifact_json(decision_path, decision)
    print(
        f"[decide] A_pass={passes['A']} B_pass={passes['B']} "
        f"NI inherited CI=({ni_lo:.4f},{ni_hi:.4f}) -> {verdict}",
        flush=True,
    )
    # #12: durable publish on the NORMAL exit path, then the P4 sentinel LAST.
    publish = getattr(args, "publish", None) or "none"
    hf_prefix = getattr(args, "publish_prefix", None) or PUBLISH_EVAL_MIRROR
    publish_artifacts([decision_path], out_root, publish=publish, hf_prefix=hf_prefix)
    sentinel = out_root / "fits" / ".p4_done"
    GC.atomic_write_json(
        sentinel,
        {
            "done": True,
            "verdict": verdict,
            "decision": str(decision_path.relative_to(out_root)),
            "publish": publish,
            "meta": GC.run_metadata({"artifact": "p4_done"}),
        },
    )
    if publish != "none" and publish in ("hf", "hf+git"):
        GC.upload_single_file(sentinel, f"{hf_prefix}/fits/.p4_done")
    print(f"[decide] .p4_done sentinel written LAST at {sentinel}", flush=True)
    return decision


def run_unitwall(args) -> dict:
    """#10 MEASURED per-unit wall at PRODUCTION shape (compute-deviation basis).

    Times ONE pooled per-layer fit through the exact production entrypoint
    (``fit_layer_unit`` -> ``ridge_fit_streamed``) on synthetic fp32 data at
    ``--wall-n x --wall-hidden`` (70/15/15 split), plus the shared LODO
    reductions + one fold. Synthetic values are fine for a WALL basis: the
    kernels (gram / eigh / GEMM) are shape-bound, not value-bound. Writes
    ``fits/unitwall_n{n}_d{d}.json`` (local measurement artifact; not part of
    the #12 publish set)."""
    import resource

    import numpy as np

    GC = _gc()
    N779 = _n779()
    n, d = int(args.wall_n), int(args.wall_hidden)
    rng = np.random.default_rng(BOOT_SEED)
    X = rng.standard_normal((n, d), dtype=np.float32)
    Y = rng.standard_normal((n, d), dtype=np.float32)
    sources = [f"src{i}" for i in range(8)]
    rows = [
        {
            "row": i,
            "context_id": f"ctx{i:06d}",
            "source_tag": sources[i % 8],
            "regime_class": "ordinary" if i % 3 else "weird",
            "lodo_group": sources[i % 8],
        }
        for i in range(n)
    ]
    perm = rng.permutation(n)
    n_tr, n_val = int(0.70 * n), int(0.15 * n)
    tr, val, te = perm[:n_tr], perm[n_tr : n_tr + n_val], perm[n_tr + n_val :]
    rows_te = [rows[int(i)] for i in te]
    lambdas = N779.LAMBDAS_N50K
    timings: dict = {}
    t0 = time.time()
    unit, _ = fit_layer_unit(X, Y, tr, val, te, rows_te, 0, lambdas, args.device, timings=timings)
    unit_wall = time.time() - t0
    t1 = time.time()
    shared = lodo_shared_reductions(X, Y, tr, val, args.device)
    shared_wall = time.time() - t1
    t2 = time.time()
    fold = lodo_fold_from_shared(shared, X, Y, rows, tr, val, sources[0], 0, lambdas, args.device)
    fold_wall = time.time() - t2
    del shared
    maxrss_gib = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024.0**2)
    doc = {
        "meta": GC.run_metadata({"artifact": "unitwall", "n": n, "d": d, "device": args.device}),
        "n": n,
        "d": d,
        "splits": {"tr": int(len(tr)), "val": int(len(val)), "te": int(len(te))},
        "pooled_unit_wall_s": round(unit_wall, 2),
        "stage_timings_s": {k: round(v, 2) for k, v in timings.items()},
        "lodo_shared_reductions_s": round(shared_wall, 2),
        "lodo_fold_s": round(fold_wall, 2),
        "lodo_fold_skipped": "skipped" in fold,
        "ru_maxrss_gib": round(maxrss_gib, 2),
        "selected_lambda": unit["fit_meta"]["selected_lambda"],
        "basis_note": (
            "#10 basis: projected_wall = pooled_unit_wall_s x n_layer_units / parallelism "
            "+ lodo_shared_s + n_groups x lodo_fold_s (per model; layer shards via --layers)"
        ),
    }
    out = Path(args.out_root) / "fits" / f"unitwall_n{n}_d{d}.json"
    write_artifact_json(out, doc)
    print(
        f"[unitwall] n={n} d={d}: pooled unit {unit_wall:.1f}s "
        f"(stages {doc['stage_timings_s']}), lodo shared {shared_wall:.1f}s "
        f"+ fold {fold_wall:.1f}s, maxrss {maxrss_gib:.1f} GiB -> {out}",
        flush=True,
    )
    return doc


# --------------------------------------------------------------------------
# selfcheck (synthetic end-to-end; equivalence + reachable verdict branches)
# --------------------------------------------------------------------------


def _toy_store(model_key: str, layer_hs: list[int], *, n=240, d=8, noise_by_hs=None, seed=0):
    """Synthetic MemStore: y = x @ W_k + noise_k per layer, 4 sources x 2 regimes."""
    import numpy as np

    rng = np.random.default_rng(seed)
    sources = [
        ("s_ord1", "ordinary"),
        ("s_ord2", "ordinary"),
        ("s_wrd1", "weird"),
        ("s_wrd2", "weird"),
    ]
    rows = []
    for i in range(n):
        src, cls = sources[i % 4]
        split = ("train", "train", "train", "val", "test")[i % 5]
        rows.append(
            {
                "row": i,
                "context_id": f"ctx{i:04d}",
                "source_tag": src,
                "regime_class": cls,
                "lodo_group": src,
                "split": split,
            }
        )
    X = rng.standard_normal((n, d))
    layers = {}
    for k in layer_hs:
        W = rng.standard_normal((d, d)) / math.sqrt(d)
        sigma = (noise_by_hs or {}).get(k, 0.3)
        layers[k] = (
            X.astype(np.float32),
            (X @ W + sigma * rng.standard_normal((n, d))).astype(np.float32),
        )
    return MemStore(rows, layers)


def _selfcheck_equivalence(tmp: Path) -> None:
    """Streamed core == the #779 reference fit on identical inputs."""
    import numpy as np

    N779 = _n779()
    rng = np.random.default_rng(7)
    n, d = 40, 8
    X = rng.standard_normal((n, d)).astype(np.float32)
    Y = (X @ rng.standard_normal((d, d)) + 0.1 * rng.standard_normal((n, d))).astype(np.float32)
    idx = rng.permutation(n)
    tr, val, te = idx[:24], idx[24:32], idx[32:]
    lambdas = N779.LAMBDAS_N50K
    ref_pred, ref_meta = N779.fit_ridge_primal(X, Y, tr, val, te, lambdas, "cpu")
    (new_pred,), new_meta = ridge_fit_streamed(X, Y, tr, val, [te], lambdas, "cpu")
    assert new_meta["selected_lambda"] == ref_meta["selected_lambda"], (new_meta, ref_meta)
    assert new_meta["lambda_grid_edge"] == ref_meta["lambda_grid_edge"]
    assert abs(new_meta["val_r2_at_selected"] - ref_meta["val_r2_at_selected"]) < 1e-9
    assert np.allclose(new_pred, ref_pred, rtol=1e-9, atol=1e-9)
    print("[selfcheck] streamed-core equivalence vs issue779 fit_ridge_primal: OK", flush=True)


def _selfcheck_truth_table() -> None:
    """Direct probes of every registered verdict branch (MF-C; reachable
    Inconclusive via BOTH routes: NOT A_pass, and a CI spanning the margin)."""
    assert decide_verdict(True, True, -0.05, 0.10) == "Replicates"
    assert decide_verdict(True, True, -0.30, -0.15) == "Fails-to-replicate"
    assert decide_verdict(True, False, -0.05, 0.10) == "Fails-to-replicate"
    assert decide_verdict(True, True, -0.30, 0.10) == "Inconclusive"  # CI spans -0.10
    assert decide_verdict(False, True, -0.05, 0.10) == "Inconclusive"  # instrument voided
    assert decide_verdict(True, True, -0.10, 0.10) == "Inconclusive"  # boundary -> spans
    print("[selfcheck] MF-C truth table (5 branches + boundary): OK", flush=True)


def _expect_raise(fn, substr: str, kind=RuntimeError) -> None:
    """Assert ``fn()`` raises ``kind`` whose message contains ``substr``."""
    try:
        fn()
    except kind as e:
        assert substr in str(e), f"expected {substr!r} in {e!r}"
    else:
        raise AssertionError(f"expected {kind.__name__} containing {substr!r} — nothing raised")


def _selfcheck_val_selection() -> None:
    """#9 fixture: VALIDATION-winner != TEST-winner — every inherited draw MUST
    follow the VALIDATION winner (selection never sees the scored partition)."""
    import numpy as np

    n_te, n_val, draws = 60, 50, 64
    hs = [1, 2]
    o_te, o_val = np.ones(n_te), np.ones(n_val)
    # hs=1 wins on VAL (val err 0.1 vs 0.9); hs=2 wins on TEST (te err 0.1 vs 0.9).
    mats_te = {
        "hs": hs,
        "err_map": np.stack([0.9 * o_te, 0.1 * o_te], axis=1),
        "err_id": np.stack([1.0 * o_te, 1.0 * o_te], axis=1),
        "sst": np.stack([o_te, o_te], axis=1),
    }
    mats_val = {
        "hs": hs,
        "err_map": np.stack([0.1 * o_val, 0.9 * o_val], axis=1),
        "err_id": np.stack([1.0 * o_val, 1.0 * o_val], axis=1),
        "sst": np.stack([o_val, o_val], axis=1),
    }
    # The OLD (#9 bug) rule — argmax of TEST R^2 — picks hs=2 here...
    r2_te = 1.0 - mats_te["err_map"].mean(0) / mats_te["sst"].mean(0)
    assert int(np.argmax(r2_te)) == 1, "fixture must make hs=2 the TEST winner"
    boot = paired_delta_bootstrap(
        mats_te,
        mats_val,
        cand=hs,
        frozen_hs=2,
        member_mask=np.ones(n_te, dtype=bool),
        draws=draws,
        seed=0,
    )
    # ...yet EVERY inherited draw follows the VAL winner hs=1 (#9).
    assert boot["selected_hs_counts"] == {"1": draws}, boot["selected_hs_counts"]
    lo, hi = boot["inherited_ci"]
    assert abs(lo - 0.1) < 1e-9 and abs(hi - 0.1) < 1e-9, (lo, hi)  # hs=1's TEST delta
    flo, fhi = boot["frozen_ci_labeled_frozen_at_selected"]
    assert abs(flo - 0.9) < 1e-9 and abs(fhi - 0.9) < 1e-9, (flo, fhi)  # hs=2 delta differs
    ids = [f"c{i}" for i in range(n_te)]
    ni = ni_bootstrap(
        mats_te,
        mats_val,
        ids,
        mats_te,
        mats_val,
        ids,
        cand_a=hs,
        cand_b=hs,
        frozen_a=2,
        frozen_b=2,
        draws=32,
        seed=1,
    )
    assert ni["selected_hs_counts"] == {"A": {"1": 32}, "B": {"1": 32}}, ni["selected_hs_counts"]
    print("[selfcheck] #9 val-selection fixture (val-winner != test-winner): OK", flush=True)


class _FakeHfStore(HfChunkStore):
    """HfChunkStore with the two network seams (listing + JSON fetch) injected."""

    def __init__(self, files: list[str], docs: dict[str, dict], prefix="pfx", hidden=8):
        super().__init__(prefix, Path("/tmp"), hidden)
        self._files = list(files)
        self._docs = docs

    def _fetch_json(self, rel: str) -> dict:
        return json.loads(json.dumps(self._docs[rel]))


def _fake_capture(prefix="pfx", keys=("chunk0000", "chunk0001"), layers=(1, 2), rows_per=3):
    """(files, docs) for a healthy unsharded fake capture (u2 contract shapes)."""
    files, docs, expected = [], {}, []
    for key in keys:
        files.append(f"{prefix}/{key}/{key}__rows.json")
        expected.append(f"{prefix}/{key}/{key}__rows.json")
        for k in layers:
            files.append(f"{prefix}/{key}/{key}__L{k:02d}.npz")
            expected.append(f"{prefix}/{key}/{key}__L{k:02d}.npz")
        docs[f"{key}/{key}__rows.json"] = {
            "rows": [{"row": i, "context_id": f"{key}_c{i}"} for i in range(rows_per)]
        }
    docs["capture_meta.json"] = {
        "chunk_keys": list(keys),
        "layers": list(layers),
        "per_chunk": {key: {"n_rows": rows_per} for key in keys},
        "totals": {"n_rows_captured": rows_per * len(keys)},
        "expected_files": expected,
    }
    files += [f"{prefix}/capture_meta.json", f"{prefix}/.capture_done"]
    return files, docs


def _selfcheck_store_completeness() -> None:
    """#11 probes: healthy PASS + five distinct fail-loud refusals (missing
    sentinel / missing chunk / partial layer set / per-chunk + total row
    mismatches / incomplete shard set)."""
    files, docs = _fake_capture()
    store = _FakeHfStore(files, docs)
    info = store.verify_complete()
    assert info["n_chunks"] == 2 and info["layers"] == [1, 2]
    assert len(store.load_rows()) == 6

    no_sentinel = [f for f in files if not f.endswith(".capture_done")]
    _expect_raise(_FakeHfStore(no_sentinel, docs).verify_complete, "no .capture_done")

    missing_chunk = [f for f in files if "/chunk0001/" not in f]
    _expect_raise(_FakeHfStore(missing_chunk, docs).verify_complete, "discovered rows.json chunks")

    partial_layer = [f for f in files if not f.endswith("chunk0001__L02.npz")]
    _expect_raise(_FakeHfStore(partial_layer, docs).verify_complete, "partial layer set")

    short_docs = json.loads(json.dumps(docs))
    short_docs["chunk0000/chunk0000__rows.json"]["rows"] = [{"row": 0, "context_id": "c0"}]
    s5 = _FakeHfStore(files, short_docs)
    s5.verify_complete()
    _expect_raise(s5.load_rows, "capture_meta.per_chunk")

    bad_total = json.loads(json.dumps(docs))
    bad_total["capture_meta.json"]["totals"]["n_rows_captured"] = 7
    s6 = _FakeHfStore(files, bad_total)
    s6.verify_complete()
    _expect_raise(s6.load_rows, "row-total mismatch")

    shard_files = [f for f in files if ".capture_done" not in f and "capture_meta" not in f]
    shard_docs = json.loads(json.dumps(docs))
    shard_docs["capture_meta_s00of02.json"] = shard_docs.pop("capture_meta.json")
    shard_files += ["pfx/.capture_done_s00of02", "pfx/capture_meta_s00of02.json"]
    _expect_raise(_FakeHfStore(shard_files, shard_docs).verify_complete, "shard sentinels")
    print("[selfcheck] #11 completeness gate: healthy PASS + 5 refusals: OK", flush=True)


def _close(a, b, tol=1e-6):
    """Recursive approx-equality for JSON-shaped fold records."""
    if isinstance(a, float) and isinstance(b, float):
        if math.isnan(a) and math.isnan(b):
            return True
        return abs(a - b) <= tol * max(1.0, abs(a), abs(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return set(a) == set(b) and all(_close(a[k], b[k], tol) for k in a)
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        return len(a) == len(b) and all(_close(x, y, tol) for x, y in zip(a, b))
    return a == b


def _selfcheck_lodo_equivalence() -> None:
    """#10: ``lodo_fold_from_shared`` == the serial ``lodo_unit`` reference,
    every group of a toy store (the supersede-contract equivalence gate)."""
    N779 = _n779()
    store = _toy_store("A", [1], n=120, seed=5)
    rows = store.load_rows()
    tr, val, te = resolve_splits(rows, pilot=False)
    X, Y = store.load_layer(1)
    lambdas = N779.LAMBDAS_N50K
    shared = lodo_shared_reductions(X, Y, tr, val, "cpu")
    for g in sorted({str(r["lodo_group"]) for r in rows}):
        ref = lodo_unit(X, Y, rows, tr, val, g, 1, lambdas, "cpu")
        fast = lodo_fold_from_shared(shared, X, Y, rows, tr, val, g, 1, lambdas, "cpu")
        assert _close(ref, fast), (g, ref, fast)
    print("[selfcheck] LODO shared-reduction == serial reference (all groups): OK", flush=True)


def _selfcheck_sanitize(tmp: Path) -> None:
    """Committed-artifact writer: non-finite -> null, strict-JSON parseable."""
    p = tmp / "nan_probe.json"
    write_artifact_json(p, {"x": float("nan"), "y": [1.0, float("inf"), float("-inf")], "z": 2})
    text = p.read_text()
    assert "NaN" not in text and "Infinity" not in text, text
    doc = json.loads(text)
    assert doc["x"] is None and doc["y"] == [1.0, None, None] and doc["z"] == 2
    print("[selfcheck] non-finite -> null committed-artifact writer: OK", flush=True)


def _selfcheck_resume_regime(args, tmp: Path) -> None:
    """#14: subset-then-full completes cleanly; a changed kNN cap re-fires ONLY
    its param-keyed cell; #13 fit-side reliability refusal fires."""
    import copy

    a = copy.copy(args)
    a.work_dir = str(tmp / "rr_work")
    a.out_root = str(tmp / "rr_out")
    a.pilot = False
    a.layers = "1,2"
    a.skip_lodo = False
    a.lodo_groups = None
    a.boot_draws = 100
    a.knn_max_n = 32
    a.device = "cpu"
    a.tensors_prefix = None
    a.publish = "none"
    a.allow_missing_reliability = True
    a.reliability = None
    store = _toy_store("A", [1, 2, 3], seed=9)
    sets = {"all": [1, 2, 3], "gate": [1, 2, 3], "h3": [2, 3]}
    part = run_fit(a, store=store, model_key="A", sets_override=sets)
    assert part.get("partial") is True and part["layers_done"] == [1, 2], part
    fits_dir, percell = model_dirs(a, "A")
    assert not (fits_dir / "fits_summary.json").exists()
    a.layers = None
    run_fit(a, store=store, model_key="A", sets_override=sets)
    assert (fits_dir / "fits_summary.json").exists()
    post1 = json.loads((percell / "post.json").read_text())
    assert post1["knn"]["max_n"] == 32, post1["knn"]
    a.knn_max_n = 16
    run_fit(a, store=store, model_key="A", sets_override=sets)
    post2 = json.loads((percell / "post.json").read_text())
    assert post2["knn"]["max_n"] == 16, post2["knn"]
    a2 = copy.copy(a)
    a2.allow_missing_reliability = False
    _expect_raise(
        lambda: run_fit(a2, store=store, model_key="A", sets_override=sets),
        "--reliability",
        kind=SystemExit,
    )
    print("[selfcheck] #14 subset-then-full + knn-cap re-key + #13 fit refusal: OK", flush=True)


def _selfcheck_pipeline(args, tmp: Path) -> None:
    """Full fit -> assemble -> decide pass on synthetic two-model stores through
    the PRODUCTION ``run_fit`` (sets_override seam), exercising baselines, kNN,
    first-PC coords, per-source + LODO folds, both CI kinds, the #12 sentinel,
    and the #13 decide refusal."""
    import copy

    a = copy.copy(args)
    a.work_dir = str(tmp / "work")
    a.out_root = str(tmp / "out")
    a.pilot = False
    a.layers = None
    a.skip_lodo = False
    a.lodo_groups = None
    a.boot_draws = 200
    a.knn_max_n = 64
    a.device = "cpu"
    a.tensors_prefix = None
    a.publish = "none"
    a.allow_missing_reliability = True
    a.reliability = None
    noise = {1: 0.8, 2: 0.15, 3: 0.5}
    store_a = _toy_store("A", [1, 2, 3], noise_by_hs=noise, seed=1)
    store_b = _toy_store("B", [1, 2, 3, 4], noise_by_hs={**noise, 4: 0.2}, seed=2)
    specs = {
        "A": {"all": [1, 2, 3], "gate": [1, 2, 3], "h3": [2, 3]},
        "B": {"all": [1, 2, 3, 4], "gate": [2, 4], "h3": [2, 4]},
    }
    for key, store in (("A", store_a), ("B", store_b)):
        run_fit(a, store=store, model_key=key, sets_override=specs[key])
    fpc = json.loads((Path(a.out_root) / "fits" / "modelA" / "firstpc_scatter.json").read_text())
    assert fpc["basis"].startswith("top eigenvector"), fpc.get("basis")
    assert len(fpc["true_pc1"]) == len(fpc["pred_pc1"]) == fpc["n"] > 0
    a.reliability_a = None
    a.reliability_b = None
    # #13: decide REFUSES without both ceilings unless the selfcheck escape is on.
    a_strict = copy.copy(a)
    a_strict.allow_missing_reliability = False
    _expect_raise(lambda: run_decide(a_strict), "--reliability-a", kind=SystemExit)
    decision = run_decide(a)
    assert decision["a_pass"] is True, "toy A map must beat identity baseline"
    assert decision["verdict"] in ("Replicates", "Inconclusive", "Fails-to-replicate")
    sentinel = Path(a.out_root) / "fits" / ".p4_done"
    assert sentinel.exists(), "#12: .p4_done sentinel must be written LAST on decide"
    assert json.loads(sentinel.read_text())["verdict"] == decision["verdict"]
    # Degrade B to pure noise -> NOT B_pass -> Fails-to-replicate through the
    # REAL decide path (not just the truth-table fn).
    b_dir = Path(a.out_root) / "fits" / "modelB"
    rec = json.loads((b_dir / "percontext_recon.json").read_text())
    for lk in rec["layers"]:
        e_id = rec["layers"][lk]["err_identity"]
        rec["layers"][lk]["err_map"] = [v * 50.0 + 1.0 for v in e_id]
    _gc().atomic_write_json(b_dir / "percontext_recon.json", rec)
    d2 = run_decide(a)
    assert d2["b_pass"] is False and d2["verdict"] == "Fails-to-replicate", d2["verdict"]
    # Void A's instrument (err_map == err_identity) -> Inconclusive, reachable
    # through the REAL path.
    a_dir = Path(a.out_root) / "fits" / "modelA"
    rec_a = json.loads((a_dir / "percontext_recon.json").read_text())
    for lk in rec_a["layers"]:
        rec_a["layers"][lk]["err_map"] = list(rec_a["layers"][lk]["err_identity"])
    _gc().atomic_write_json(a_dir / "percontext_recon.json", rec_a)
    d3 = run_decide(a)
    assert d3["a_pass"] is False and d3["verdict"] == "Inconclusive", d3["verdict"]
    print("[selfcheck] pipeline verdicts: base + Fails + REACHABLE Inconclusive: OK", flush=True)


def run_selfcheck(args) -> int:
    import tempfile

    _selfcheck_truth_table()
    _selfcheck_val_selection()
    _selfcheck_store_completeness()
    _selfcheck_lodo_equivalence()
    with tempfile.TemporaryDirectory(prefix="i2502_fits_selfcheck_") as td:
        tmp = Path(td)
        _selfcheck_sanitize(tmp)
        _selfcheck_equivalence(tmp)
        _selfcheck_resume_regime(args, tmp)
        _selfcheck_pipeline(args, tmp)
    # Candidate-set derivation pins (production constants).
    assert h3_hs_set("B") == [4, 8, 12, 16, 20, 24, 28, 32]
    assert h3_hs_set("A") == [3, 7, 10, 14, 17, 21, 24, 28]
    assert assert_mfj("B", list(range(1, 33)))["subset_ok"]
    assert assert_mfj("A", list(range(1, 29)))["subset_ok"]
    try:
        assert_mfj("B", [k for k in range(1, 33) if k != 32])
    except RuntimeError:
        pass
    else:
        raise AssertionError("MF-J assert failed to fire on a missing H3 layer")
    print("[selfcheck] ALL OK", flush=True)
    return 0


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--phase", choices=("fit", "decide", "unitwall", "selfcheck"), default="fit")
    ap.add_argument("--model-key", choices=("A", "B"), default=None, help="fit phase: which model")
    ap.add_argument(
        "--tensors-prefix",
        default=None,
        help="HF data-repo prefix of the u2 capture store (default per --model-key)",
    )
    ap.add_argument("--work-dir", default="/workspace/issue2502_fits")
    ap.add_argument("--out-root", default=str(_REPO_ROOT / "eval_results" / "issue_2502"))
    ap.add_argument("--device", default="cpu", help="fit device (cpu on the cpu-bigmem pod)")
    ap.add_argument("--pilot", action="store_true", help="G1 pilot mode (80/20, MF-B)")
    ap.add_argument(
        "--g1-gate",
        action="store_true",
        help=f"with --pilot: exit rc={G1_GATE_RC} when the G1 gate FAILs (designed halt)",
    )
    ap.add_argument("--layers", default=None, help="comma hs subset (across-layer sharding)")
    ap.add_argument("--skip-lodo", action="store_true")
    ap.add_argument("--lodo-groups", default=None, help="comma subset of lodo groups")
    ap.add_argument("--boot-draws", type=int, default=2000)
    ap.add_argument("--knn-max-n", type=int, default=5000)
    ap.add_argument("--reliability-a", default=None, help="decide: modelA reliability_ceiling.json")
    ap.add_argument("--reliability-b", default=None, help="decide: modelB reliability_ceiling.json")
    ap.add_argument(
        "--reliability",
        default=None,
        help="fit: THIS model's reliability_ceiling.json (#13/MF-E — required for a "
        "non-pilot fit unless --allow-missing-reliability)",
    )
    ap.add_argument(
        "--allow-missing-reliability",
        action="store_true",
        help="selfcheck/smoke-only escape for the #13 reliability requirements",
    )
    ap.add_argument(
        "--publish",
        choices=("none", "hf", "git", "hf+git"),
        default=None,
        help="#12 REQUIRED for fit/decide: durable disposition of committed artifacts "
        "on the normal exit path ('none' = smoke/selfcheck local-only)",
    )
    ap.add_argument(
        "--publish-prefix",
        default=PUBLISH_EVAL_MIRROR,
        help="HF data-repo prefix for the eval-results mirror (#12)",
    )
    ap.add_argument("--wall-n", type=int, default=150000, help="unitwall: rows (#10 basis)")
    ap.add_argument("--wall-hidden", type=int, default=4096, help="unitwall: hidden dim (B=4096)")
    ap.add_argument("--import-check", action="store_true")
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if args.import_check:
        from explore_persona_space.orchestrate.argcheck import assert_args_attributes_defined

        assert_args_attributes_defined(__file__)
        print("issue2502_fits: import-check OK", flush=True)
        return 0
    # load_dotenv BEFORE any numpy/torch import (thread caps freeze at import, #847).
    from explore_persona_space.orchestrate.env import load_dotenv

    load_dotenv()
    rc = 0
    if args.phase == "selfcheck":
        rc = run_selfcheck(args)
    elif args.phase == "unitwall":
        run_unitwall(args)
    elif args.phase == "decide":
        if args.publish is None:
            raise SystemExit(
                "--publish {none,hf,git,hf+git} is REQUIRED for --phase decide (#12: "
                "every caller states a durability disposition; smokes pass --publish none)"
            )
        run_decide(args)
    else:
        if args.publish is None:
            raise SystemExit(
                "--publish {none,hf,git,hf+git} is REQUIRED for --phase fit (#12: "
                "every caller states a durability disposition; smokes pass --publish none)"
            )
        if args.g1_gate and args.layers:
            raise SystemExit(
                "--g1-gate with a partial --layers shard cannot evaluate the G1 verdict "
                f"(a partial return would misread as a designed rc={G1_GATE_RC} halt); "
                "run the gate on the full-set invocation"
            )
        summary = run_fit(args)
        if args.pilot and args.g1_gate and args.model_key == "A":
            verdict = summary.get("g1", {}).get("verdict")
            if verdict != "PROCEED":
                print(f"[g1-gate] verdict={verdict} -> designed halt rc={G1_GATE_RC}", flush=True)
                rc = G1_GATE_RC
    sys.stdout.flush()
    sys.stderr.flush()
    return rc


if __name__ == "__main__":
    sys.exit(main())
