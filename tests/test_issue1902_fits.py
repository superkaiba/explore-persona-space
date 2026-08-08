"""Tests for scripts/issue1902_fits.py (issue #1902 unit C).

CPU-only, synthetic tensors (h=8, n~24 — NEVER corpus text). The end-to-end
test runs the REAL ``run_fits`` body over a synthetic store shaped exactly
like the P3 capture layout, faking ONLY the Hub upload boundary with a
signature-conformant ``upload_dir_sharded`` twin (create_autospec) — every
fit / bootstrap / verdict path executes for real (production-body rule).

Degenerate-gate probes (data-dependent gates duty): fold-skip on a
too-small fold, kNN ks clamp, the cuSOLVER-eigh CPU fallback branch, the
within-stratum vs cluster bootstrap branches, verdict lattice edges.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
sys.path.insert(0, str(REPO / "scripts"))

import issue1902_common as C  # noqa: E402
import issue1902_fits as F  # noqa: E402
import issue1902_run as R  # noqa: E402

from explore_persona_space.eval.vllm_util import GPU_FREE_MARGIN_GIB  # noqa: E402

pytestmark = pytest.mark.filterwarnings("ignore::UserWarning")

CKPTS = ["B", "R"]
LAYERS = [0, 1, 2]
H = 8
N_SINGLE = 24
N_MULTI = 12


# ── synthetic store (capture-layout twin) ────────────────────────────────────


def _mk_rows(corpus: str, n: int) -> list[dict]:
    rows = []
    for i in range(n):
        if corpus == "single" and i >= n - 4:
            cls = "gsm8k" if i % 2 == 0 else "mbpp"
            group, cluster = cls, -1
        else:
            cls, cluster = "generic", i % 6
            group = f"cluster_{cluster}"
        rows.append(
            {
                "id": f"{corpus}_{i:05d}",
                "class": cls,
                "group": group,
                "cluster": cluster,
                "prefix_len": 3,
                "context_len": 6,
                "seam_prefix": False,
                "seam_context": False,
                "n_prompt_tokens": 6,
                "n_answer_tokens": 5,
            }
        )
    return rows


def _write_store(out_root: Path) -> None:
    import torch

    rng = np.random.default_rng(0)
    store = out_root / "store"
    manifest = {"metadata": {}, "ckpts": CKPTS, "corpora": {}}
    for corpus, n in (("single", N_SINGLE), ("multi", N_MULTI)):
        rows = _mk_rows(corpus, n)
        ids = [r["id"] for r in rows]
        groups = {r["id"]: r["group"] for r in rows}
        fold_of_group = R.assign_fold_groups([groups[i] for i in ids])
        manifest["corpora"][corpus] = {
            "n_intersection": n,
            "fold_of_group": fold_of_group,
            "n_folds": C.N_FOLDS,
            "ids": ids,
        }
        W_true = rng.normal(size=(H, H)) * 0.5
        for m in CKPTS:
            u = rng.normal(size=(n, H)).astype(np.float32)
            p = rng.normal(size=(n, H)).astype(np.float32)
            for layer in LAYERS:
                ctx_dir = (store / C.ctx_store_relpath(m, corpus, layer)).parent
                ctx_dir.mkdir(parents=True, exist_ok=True)
                d = {
                    "u_mean": torch.tensor(u + 0.1 * layer),
                    "u_last": torch.tensor(u),
                    "row_ids": ids,
                }
                if corpus == "multi":
                    d["p_mean"] = torch.tensor(p)
                    d["p_last"] = torch.tensor(p)
                torch.save(d, ctx_dir / f"L{layer}.pt")
            with open(ctx_dir / "row_index.jsonl", "w", encoding="utf-8") as f:
                for r in rows:
                    f.write(json.dumps(r) + "\n")
            for s in CKPTS:
                w = (u + 0.1) @ W_true + 0.05 * rng.normal(size=(n, H))
                for layer in LAYERS:
                    cell = (store / C.answer_store_relpath(m, s, corpus, layer)).parent
                    cell.mkdir(parents=True, exist_ok=True)
                    torch.save(
                        {"w": torch.tensor(w.astype(np.float32)), "row_ids": ids},
                        cell / f"L{layer}.pt",
                    )
        # reliability + robust stores (single corpus shapes only)
        if corpus == "single":
            for m in CKPTS:
                u = rng.normal(size=(n, H)).astype(np.float32)
                for seed in C.RELIABILITY_SEEDS:
                    for layer in LAYERS:
                        rd = store / f"reliability/{m}/single/seed{seed}"
                        rd.mkdir(parents=True, exist_ok=True)
                        torch.save(
                            {
                                "w": torch.tensor((u @ W_true).astype(np.float32)),
                                "row_ids": ids,
                            },
                            rd / f"L{layer}.pt",
                        )
            for layer in LAYERS:
                rb = store / "robust_native/R/single"
                (rb / "ctx").mkdir(parents=True, exist_ok=True)
                torch.save(
                    {"w": torch.tensor((u @ W_true).astype(np.float32)), "row_ids": ids},
                    rb / f"L{layer}.pt",
                )
                torch.save(
                    {"u_mean": torch.tensor(u), "u_last": torch.tensor(u), "row_ids": ids},
                    rb / "ctx" / f"L{layer}.pt",
                )
    (out_root / "gen").mkdir(parents=True, exist_ok=True)
    R._write_json_atomic(out_root / "gen" / "intersection_manifest.json", manifest)
    R._write_json_atomic(out_root / "revision_pins.json", {m: "local:test" for m in C.CKPTS})


class _FakeShardResult:
    def __init__(self) -> None:
        self.uploaded: list = []
        self.skipped_existing: list = []
        self.rerouted: list = []


def _args(smoke: bool = True) -> argparse.Namespace:
    return argparse.Namespace(smoke=smoke, device="cpu")


@pytest.fixture()
def fits_env(tmp_path, monkeypatch):
    out_root = tmp_path / "out"
    _write_store(out_root)
    monkeypatch.setenv("EPM_ISSUE1902_EVAL_DIR", str(tmp_path / "eval"))
    monkeypatch.setenv("EPM_SENTINEL_DIR", str(tmp_path / "sentinels"))
    monkeypatch.setenv("EPM_ISSUE1902_FIT_WORKERS", "2")
    # Hub boundary: signature-conformant autospec twin (never a bare Mock).
    from explore_persona_space.orchestrate import upload_sharded

    fake = mock.create_autospec(upload_sharded.upload_dir_sharded, return_value=_FakeShardResult())
    monkeypatch.setattr(
        F,
        "_upload_eval_mirror",
        lambda ctx: fake(ctx.eval_dir, C.HF_DATA_REPO, C.EVAL_MIRROR_HF_PATH, repo_type="dataset"),
    )
    return out_root, tmp_path / "eval", fake


# ── end-to-end production-body run (synthetic store, CPU) ────────────────────


def test_run_fits_end_to_end(fits_env):
    out_root, eval_dir, fake_upload = fits_env
    F.run_fits(_args(smoke=True), out_root, CKPTS)
    # §6.5 deliverables all present
    for rel in (
        "fits/grid_cells.json",
        "fits/layer_sweep.json",
        "transfer/transfer_matrix.json",
        "clusters/delta_qc.json",
        "operator/operator_battery.json",
        "fits/parity_gate.json",
    ):
        assert (eval_dir / rel).exists(), rel
    sweep = json.loads((eval_dir / "fits" / "layer_sweep.json").read_text())
    assert sweep["selection"]["layer_star"] in LAYERS
    assert sweep["parity_gate"]["max_rel_diff"] <= F.PARITY_TOL  # tiny-n parity is exact-ish
    grid = json.loads((eval_dir / "fits" / "grid_cells.json").read_text())
    star = grid["layer_star"]
    diag = grid["cells"]["diag_B_single_ctx"]
    assert 0.0 < diag["r2_at_star"] <= 1.0  # planted linear signal recovered
    assert len(diag["ci_frozen_at_star"]) == 2 and len(diag["ci_selection_inherited"]) == 2
    assert diag["shuffle_null_r2"], "shuffled-pairing null missing"
    assert max(diag["shuffle_null_r2"]) < diag["r2_at_star"]
    assert "grid_BR_single_ctx" in grid["cells"]
    assert grid["cells"]["grid_BR_single_ctx"]["per_layer"][str(star)]["knn"]
    assert grid["mlp"], "MLP cells missing"
    assert grid["h3_variance_decomposition"]
    xf = json.loads((eval_dir / "transfer" / "transfer_matrix.json").read_text())
    for mode in F.XFER_MODES:
        assert np.isfinite(xf["pairs"]["B->R"]["r2"][mode])
    assert xf["pairs"]["B->R"]["nulls"]["shuffled_correspondence_r2"]
    assert xf["h1"]["verdict"].startswith("informational-smoke:")
    h2 = json.loads((eval_dir / "clusters" / "delta_qc.json").read_text())
    assert h2["per_cluster"], "per-cluster delta-Q missing"
    assert h2["smoke_informational_contrasts"], "smoke contrast branches unexercised"
    assert any("context" in k for k in h2["smoke_informational_contrasts"])  # stratum branch
    assert (eval_dir / "clusters" / "null_matrix.npz").exists()
    op = json.loads((eval_dir / "operator" / "operator_battery.json").read_text())
    assert op["pairs"]["B->R"]["er_delta"] > 0
    assert "procrustes_aligned" in op["pairs"]["B->R"]
    assert op["cka"]["B-R_single"]["cka_u"]
    # percell shards at full diag grain
    shards = list((eval_dir / "fits" / "percell").glob("diag_B_single_ctx_f*.npz"))
    assert shards
    d = np.load(shards[0])
    assert d["ss_res"].shape[0] == len(LAYERS)
    # sentinel written as epm:smoke-result (never epm:results under smoke)
    sents = list(Path(os.environ["EPM_SENTINEL_DIR"]).glob("issue-1902-epm_smoke-result-*.json"))
    assert sents
    body = json.loads(sents[0].read_text())
    assert body["kind"] == "epm:smoke-result" and body["sentinel_schema_version"] == 1
    note = json.loads(body["note"])
    assert note["reproducibility_card"]["training"].startswith("N/A")
    assert note["gpu_hours_budgeted"] == 29
    assert fake_upload.called


def test_run_fits_resume_skips_done_units(fits_env):
    out_root, eval_dir, _ = fits_env
    F.run_fits(_args(smoke=True), out_root, CKPTS)
    t0 = os.path.getmtime(eval_dir / "fits" / "grid_cells.json")
    F.run_fits(_args(smoke=True), out_root, CKPTS)  # all units resume-skip
    assert os.path.getmtime(eval_dir / "fits" / "grid_cells.json") >= t0  # finalize reruns


# ── degenerate-gate probes ───────────────────────────────────────────────────


def test_knn_ks_clamp():
    assert F._knn_ks(3) == (1,)
    assert F._knn_ks(7) == (1, 5)
    assert F._knn_ks(1000) == (1, 5, 10)


def test_savez_atomic_suffix(tmp_path):
    p = tmp_path / "x.npz"
    F._savez_atomic(p, a=np.zeros(3))
    assert p.exists() and not (tmp_path / "x.tmp.npz.npz").exists()
    assert np.load(p)["a"].shape == (3,)


def test_realized_transitions_fallback():
    assert F.realized_transitions(["B", "S", "D", "R"]) == [("B", "S"), ("S", "D"), ("D", "R")]
    assert F.realized_transitions(["B", "R"]) == [("B", "R")]
    assert F.realized_transitions(["B", "D", "R"]) == [("D", "R")]


def test_batched_ridge_cpu_fallback(monkeypatch):
    """cuda eigh LinAlgError -> CPU retry (exact backend swap); a CPU failure
    re-raises (fail loud — genuinely pathological input)."""
    import torch

    from explore_persona_space.experiments.issue_779 import fit_h

    calls = []
    real = fit_h.ridge_fit_predict_fast_layer_batched

    def _twin(
        Xtr, Ytr, Xev, *, lambdas=None, device="cpu", return_weights=False, return_info=False
    ):
        calls.append(device)
        if device != "cpu":
            raise torch.linalg.LinAlgError("eigh failed to converge (synthetic)")
        return real(
            Xtr,
            Ytr,
            Xev,
            lambdas=lambdas,
            device=device,
            return_weights=return_weights,
            return_info=return_info,
        )

    monkeypatch.setattr(fit_h, "ridge_fit_predict_fast_layer_batched", _twin)
    rng = np.random.default_rng(0)
    X, Y = rng.normal(size=(1, 12, 4)), rng.normal(size=(1, 12, 4))
    out = F._batched_ridge(X, Y, X[:, :3], device="cuda:0")
    assert out.shape == (1, 3, 4)
    assert calls == ["cuda:0", "cpu"]

    def _always(Xtr, Ytr, Xev, **kw):
        raise torch.linalg.LinAlgError("eigh failed (synthetic)")

    monkeypatch.setattr(fit_h, "ridge_fit_predict_fast_layer_batched", _always)
    with pytest.raises(torch.linalg.LinAlgError):
        F._batched_ridge(X, Y, X[:, :3], device="cpu")


def test_boot_r2_known_values():
    counts = np.asarray([[1.0, 1.0], [2.0, 0.0]])
    res_g = np.asarray([1.0, 3.0])
    tot_g = np.asarray([2.0, 6.0])
    r2 = F._boot_r2(counts, res_g, tot_g)
    assert np.allclose(r2, [1 - 4 / 8, 1 - 2 / 4])
    layered = F._boot_r2(counts, np.stack([res_g, res_g]), np.stack([tot_g, tot_g]))
    assert layered.shape == (2, 2)


def test_fold_skip_gate(fits_env, monkeypatch):
    """A fold with <2 eval rows records a skip (designed handling, no crash)."""
    out_root, _, _ = fits_env
    ctx = F.FitsContext(_args(smoke=True), out_root, CKPTS)
    idx = ctx.corpora["single"]
    monkeypatch.setattr(
        type(ctx),
        "fold_masks",
        lambda self, corpus, fold: (
            np.ones(self.corpora[corpus].n, bool),
            np.zeros(self.corpora[corpus].n, bool),
        ),
    )
    info = F.run_sweep_unit(ctx, "cpu", m="B", corpus="single", fold=0)
    assert info == {}
    rec = ctx.read_unit("sweep_B_single_f0")
    assert rec["skipped"] is True and rec["n_ev"] == 0
    del idx


def test_fit_h_return_info_backward_compat():
    """Option (a) pin: return shapes for all 4 flag combinations."""
    from explore_persona_space.experiments.issue_779.fit_h import (
        ridge_fit_predict_fast_layer_batched as fn,
    )

    rng = np.random.default_rng(1)
    X, Y, Xe = rng.normal(size=(2, 10, 4)), rng.normal(size=(2, 10, 3)), rng.normal(size=(2, 5, 4))
    p = fn(X, Y, Xe)
    assert p.shape == (2, 5, 3)
    p, w = fn(X, Y, Xe, return_weights=True)
    assert w.shape == (2, 4, 3)
    p, info = fn(X, Y, Xe, return_info=True)
    assert info["best_lambda"].shape == (2,) and info["dof"].shape == (2,)
    p, w, info = fn(X, Y, Xe, return_weights=True, return_info=True)
    assert w.shape == (2, 4, 3) and info["dof"].shape == (2,)
    assert np.all(info["dof"] > 0)


# ── C2: StoreCache duplicate-insert race (concern storecache-lru-…-race) ─────


def test_storecache_concurrent_miss_single_insert(tmp_path, monkeypatch):
    """Two workers missing the SAME shard concurrently insert ONCE (no
    duplicate _order token, no double-counted bytes) and a later over-cap
    eviction walks the LRU without KeyError. Fails pre-fix: the duplicate
    _order token makes the L2 load's eviction hit an already-evicted key."""
    import threading

    import torch

    d = tmp_path / "store" / "B" / "ctx" / "single"
    d.mkdir(parents=True)
    for name in ("L0.pt", "L1.pt", "L2.pt"):
        torch.save({"w": torch.zeros(64, 8, dtype=torch.float16), "row_ids": ["a"]}, d / name)
    # cap ~3 KB: each cached entry is ~2 KB fp32, so 2 entries force eviction
    cache = F.StoreCache(tmp_path / "store", cap_gb=3e-6)
    barrier = threading.Barrier(2, timeout=10)
    real_load = torch.load

    def barrier_load(*a, **kw):
        out = real_load(*a, **kw)
        barrier.wait()  # both threads finish loading BEFORE either inserts
        return out

    monkeypatch.setattr(torch, "load", barrier_load)
    errs: list[BaseException] = []

    def worker():
        try:
            cache._load("B/ctx/single/L0.pt")
        except BaseException as e:
            errs.append(e)

    threads = [threading.Thread(target=worker) for _ in range(2)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errs, errs
    assert cache._order.count("B/ctx/single/L0.pt") == 1
    assert len(cache._order) == len(cache._files) == 1
    monkeypatch.setattr(torch, "load", real_load)
    cache._load("B/ctx/single/L1.pt")
    cache._load("B/ctx/single/L2.pt")  # eviction walks _order — pre-fix KeyError
    assert len(cache._order) == len(cache._files)
    assert cache._bytes <= cache.cap or len(cache._order) == 1


# ── C1: HF re-stage of a delete-local-reaped store (plan §4 P4 "otherwise") ──


def _mark_leg_uploaded(out_root: Path) -> None:
    for m in CKPTS:
        R.mark_unit_done(
            out_root,
            f"capture_upload_{m}",
            {"phase": "capture_upload", "ckpt": m},
            {"delete_local": True},
        )


def test_fits_restage_from_hub_after_delete_local(fits_env, monkeypatch, tmp_path):
    """C1 (concern p3-delete-local-starves-p4-store): with the leg's VERIFIED
    upload record present, run_fits re-stages the reaped store prefixes from
    HF and completes end-to-end. Real staging bodies execute
    (ensure_store_staged -> _restage_store_prefix -> hub.stage_hub_file);
    fakes sit ONLY at the huggingface_hub network boundary (autospec'd)."""
    import shutil

    import huggingface_hub

    from explore_persona_space.orchestrate import hub

    out_root, eval_dir, _ = fits_env
    store = R._store_root(out_root)
    mirror = tmp_path / "hub_mirror"
    shutil.copytree(store, mirror)  # the "uploaded" bytes
    _mark_leg_uploaded(out_root)
    shutil.rmtree(store / "B")  # P3 delete-local reaped the grid subtree...
    shutil.rmtree(store / "reliability" / "B")  # ...and the reliability twin

    def fake_repo_info(self, repo_id, **kw):
        assert repo_id == C.HF_DATA_REPO
        return mock.Mock(sha="deadbeefcafe")

    def fake_list(api, repo_id, prefix, repo_type="dataset", revision=None):
        assert repo_id == C.HF_DATA_REPO and prefix.startswith(f"{C.STORE_HF_PATH}/")
        rel_prefix = prefix[len(C.STORE_HF_PATH) + 1 :]
        base = mirror / rel_prefix
        return [
            f"{C.STORE_HF_PATH}/{rel_prefix}/{p.relative_to(base).as_posix()}"
            for p in sorted(base.rglob("*"))
            if p.is_file()
        ]

    def fake_download(repo_id, filename, **kw):
        src = mirror / filename[len(C.STORE_HF_PATH) + 1 :]
        dst = Path(kw["local_dir"]) / filename
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src, dst)
        return str(dst)

    monkeypatch.setattr(
        huggingface_hub.HfApi,
        "repo_info",
        mock.create_autospec(huggingface_hub.HfApi.repo_info, side_effect=fake_repo_info),
    )
    monkeypatch.setattr(
        hub,
        "list_hf_files_under_path",
        mock.create_autospec(hub.list_hf_files_under_path, side_effect=fake_list),
    )
    monkeypatch.setattr(
        huggingface_hub,
        "hf_hub_download",
        mock.create_autospec(huggingface_hub.hf_hub_download, side_effect=fake_download),
    )
    F.run_fits(_args(smoke=True), out_root, CKPTS)
    # the consumer loaded from the STAGED result (staged-layout consumer-open)
    assert (store / "B" / "ctx" / "single" / "row_index.jsonl").exists()
    assert (store / "reliability" / "B" / "single" / "seed43" / "L0.pt").exists()
    assert (eval_dir / "fits" / "grid_cells.json").exists()


def test_fits_missing_store_without_upload_record_fails_loud(fits_env):
    """C1 negative: HARD leaves missing with NO verified upload record is the
    genuine run-capture-first case — fail loud, never a silent re-stage."""
    import shutil

    out_root, _, _ = fits_env
    shutil.rmtree(R._store_root(out_root) / "B")
    with pytest.raises(FileNotFoundError, match="run --phase capture first"):
        F.ensure_store_staged(out_root, CKPTS)


def test_fits_missing_soft_leaves_without_record_warn_only(fits_env):
    """C1 soft tier: reliability/robust leaves missing with no upload record
    only warn (their consumers' graceful branches own the degradation)."""
    import shutil

    out_root, _, _ = fits_env
    shutil.rmtree(R._store_root(out_root) / "reliability" / "B")
    assert F.ensure_store_staged(out_root, CKPTS) == {}


def test_ensure_store_staged_noop_when_local_complete(fits_env):
    """Local store intact -> zero staging, zero network (plan §4 P4 local-first)."""
    out_root, _, _ = fits_env
    assert F.ensure_store_staged(out_root, CKPTS) == {}


# ── C3: upload roots cover every writer + every P4-consumed leaf ─────────────


def test_capture_store_roots_cover_writers_and_p4_leaves():
    """C3 (concern store-upload-misses-reliability-robust-pilot-subtrees):
    the union of upload-leg roots path-prefix-covers every store dir the
    capture/pilot writers produce AND every P4-consumed leaf."""
    store = Path("/store")
    ckpts = list(C.CKPTS)

    def covered(rel: str, roots: list[str]) -> bool:
        return any(rel == r or rel.startswith(r + "/") for r in roots)

    for m in ckpts:
        roots = [r.relative_to(store).as_posix() for r in R.capture_store_roots(store, m)]
        writers = [f"{m}/{src}/{corpus}" for src in ckpts for corpus in C.CORPORA]
        writers += [f"{m}/{C.CTX_SOURCE}/{corpus}" for corpus in C.CORPORA]
        writers += [f"reliability/{m}/{C.CORPUS_SINGLE}/seed{s}" for s in C.RELIABILITY_SEEDS]
        writers += [f"robust_native/{m}/{C.CORPUS_SINGLE}"]
        # pilot cells incl. the A12 fp32 twin (capture_cell keep_fp32 layout)
        writers += [
            f"pilot/{m}/plain/{C.CORPUS_SINGLE}",
            f"pilot/{m}/native/{C.CORPUS_SINGLE}",
            f"pilot/{m}/plain_fp32/{C.CORPUS_SINGLE}",
        ]
        for w in writers:
            assert covered(w, roots), f"writer subtree not upload-eligible: {w}"
        hard, soft = F.expected_store_leaves(m, ckpts)
        for leaves in (*hard.values(), *soft.values()):
            for leaf in leaves:
                assert covered(leaf, roots), f"P4-consumed leaf not upload-eligible: {leaf}"


# ── A14 free-HBM layer-chunk clamp (#1902 crash 1 sweep) ─────────────────────

GIB = 2**30


def test_layer_chunk_cap_exclusive_host_unclamped():
    # A14's own basis: n_tr ~= 8.3k -> ~0.55 GiB fp64 Gram x factor 4 ~= 2.2
    # GiB/layer; >= 40 GiB free (exclusive host, post-model-unload) keeps the
    # plan-fixed chunk of 8.
    assert F.layer_chunk_cap_for_free(40 * GIB, n_tr=8_300) == F.LAYER_CHUNK


def test_layer_chunk_cap_shared_node_downscales():
    # A co-tenant squeezing free HBM to 12 GiB: (12-6) GiB usable / ~2.05
    # GiB/layer -> 2 layers per chunk.
    cap = F.layer_chunk_cap_for_free(12 * GIB, n_tr=8_300)
    assert 1 <= cap < F.LAYER_CHUNK
    per_layer = 8_300 * 8_300 * 8 * F.EIGH_WORKSPACE_FACTOR
    assert cap * per_layer <= 12 * GIB - int(GPU_FREE_MARGIN_GIB * GIB)


def test_layer_chunk_cap_fail_loud_when_one_layer_cannot_fit():
    with pytest.raises(RuntimeError, match="1-layer Gram eigh"):
        F.layer_chunk_cap_for_free(7 * GIB, n_tr=8_300)


def test_layer_chunk_cap_cpu_device_unclamped():
    assert F._layer_chunk_cap("cpu", n_tr=8_300) == F.LAYER_CHUNK
