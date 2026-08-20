"""#2388 ADVERSARIAL field-binding tests for the QA correctness-DV derivation.

The banked #1739 artifact's ``rows[].dv`` is the FABRICATION fraction and the
reused fits entrypoint binds to ``rows[].dv`` ONLY (``scripts/issue1739_fits.py``
L487/L499/L529 of the ported file) — the #2388 fix materializes a derived file
with ``dv := fractions.correct``. These tests use a CONSTRUCTED fixture in which
``dv`` and ``fractions.correct`` are deliberately DIFFERENT on EVERY row (a
fixture property, not a sampled claim — the banked artifact's 642 agreeing rows
are exactly how a mis-binding hides):

t1: the derived file's ``dv`` equals the source's ``fractions.correct`` and
    differs from the source's ``dv``, on every row;
t2: an end-to-end mini-fit through the PARENT loader path (the ported script's
    ``_load_labeled`` over a tiny real store) consumes the CORRECTNESS column,
    not the fabrication column.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str, rel: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def dv_build():
    return _load_script("issue2388_dv_build_script", "scripts/issue2388_dv_build.py")


@pytest.fixture(scope="module")
def fits_script():
    return _load_script("issue1739_fits_script_t2388", "scripts/issue1739_fits.py")


N_CTX = 6
K = 5


def _adversarial_fixture(path: Path) -> dict:
    """labeling.json in which dv != fractions.correct on EVERY row."""
    rows = []
    for i in range(N_CTX):
        n_correct = i % (K + 1)
        frac_correct = n_correct / K
        # Fabrication fraction deliberately DIFFERENT from the correct fraction
        # on every row (offset by one rollout, wrapped).
        frac_fab = ((n_correct + 1) % (K + 1)) / K
        assert frac_fab != frac_correct
        rows.append(
            {
                "context_id": f"ctx{i:03d}",
                "n_rollouts": K,
                "n_decided": K,
                "n_unjudged": 0,
                "counts": {"correct": n_correct, "abstained": 0, "fabricated": K - n_correct},
                "fractions": {"correct": frac_correct, "abstained": 0.0, "fabricated": frac_fab},
                "dv": frac_fab,  # the banked binding: dv IS the fabrication fraction
                "split": "train",
                "group_key": f"grp{i % 3}",
                "rung": "train",
            }
        )
    payload = {
        "behavior": "hallucination",
        "n_contexts": len(rows),
        "n_contexts_with_dv": len(rows),
        "rows": rows,
        "git_commit": "fixture",
        "ts": "fixture",
    }
    path.write_text(json.dumps(payload))
    return payload


def test_t1_derived_dv_binds_to_correctness_column(tmp_path, dv_build):
    src = tmp_path / "banked_labeling.json"
    fixture = _adversarial_fixture(src)
    out = tmp_path / "derived" / "labeling.json"
    report = dv_build.derive_from_banked(src, out)
    derived = json.loads(out.read_text())
    assert report["n_rows_where_dv_changed"] == N_CTX  # constructed: differs on EVERY row
    src_by_ctx = {r["context_id"]: r for r in fixture["rows"]}
    for row in derived["rows"]:
        s = src_by_ctx[row["context_id"]]
        assert row["dv"] == s["fractions"]["correct"], "derived dv != source fractions.correct"
        assert row["dv"] != s["dv"], "derived dv still equals the fabrication column"
        # every other field carried verbatim
        assert row["split"] == s["split"] and row["group_key"] == s["group_key"]


def _tiny_store(store_dir: Path, n_layers: int = 2, hidden: int = 4) -> None:
    """Tiny REAL store in the exact shard layout ``store_io.load_summaries`` reads."""
    from explore_persona_space.experiments.issue_1739.capture import write_store_shard

    rng = np.random.default_rng(0)
    summaries, meta = [], []
    for i in range(N_CTX):
        for k in range(2):  # two rollout rows per context
            summaries.append(
                {
                    kind: rng.normal(size=(n_layers, hidden)).astype(np.float16)
                    for kind in ("prefix_end", "context_end", "t1")
                }
            )
            meta.append({"context_id": f"ctx{i:03d}", "rollout_k": k})
    write_store_shard(store_dir, 0, summaries, meta, kinds=("prefix_end", "context_end", "t1"))


def test_t2_parent_loader_mini_fit_consumes_correctness(tmp_path, dv_build, fits_script):
    """End-to-end through the PARENT loader: dv vector == correctness, then a fit runs."""
    src = tmp_path / "banked_labeling.json"
    fixture = _adversarial_fixture(src)
    dv_json = tmp_path / "derived" / "labeling.json"
    dv_build.derive_from_banked(src, dv_json)
    store = tmp_path / "store"
    _tiny_store(store)

    tbl = fits_script._load_labeled(
        store, dv_json, layers=[0, 1], config="config_a", need_rollout_rows=False
    )
    src_by_ctx = {r["context_id"]: r for r in fixture["rows"]}
    expect_correct = np.array([src_by_ctx[c]["fractions"]["correct"] for c in tbl.ctx_order])
    expect_fab = np.array([src_by_ctx[c]["dv"] for c in tbl.ctx_order])
    assert np.array_equal(tbl.dv, expect_correct), (
        "parent loader consumed something other than the correctness column"
    )
    assert not np.array_equal(tbl.dv, expect_fab), (
        "parent loader still consumed the fabrication column"
    )

    # Mini-fit: the loaded arrays run through the extended per-target ridge core
    # (n_train < d dual route at this tiny shape) and produce finite predictions.
    from explore_persona_space.experiments.issue_1739.fits import ridge_gcv_predict_per_target

    z = np.asarray(tbl.z_by_variant["context_end"], dtype=np.float64)  # (Ly, n, d)
    y = np.broadcast_to(tbl.dv[None, :, None], (z.shape[0], z.shape[1], 1)).copy()
    preds = ridge_gcv_predict_per_target(z, y, [z], dof_cap=0.9)
    assert np.all(np.isfinite(preds[0])), "mini-fit produced non-finite predictions"


def test_negative_control_loader_on_banked_binding_reads_fabrication(
    tmp_path, dv_build, fits_script
):
    """NEGATIVE control: loading the UNDERIVED fixture yields the fabrication column.

    This is what makes t2 adversarial — the same loader on the same store reads a
    DIFFERENT dv vector from the underived file, so an agreeing fixture could not
    have distinguished the bindings.
    """
    src = tmp_path / "banked_labeling.json"
    fixture = _adversarial_fixture(src)
    store = tmp_path / "store"
    _tiny_store(store)
    tbl = fits_script._load_labeled(
        store, src, layers=[0, 1], config="config_a", need_rollout_rows=False
    )
    src_by_ctx = {r["context_id"]: r for r in fixture["rows"]}
    expect_fab = np.array([src_by_ctx[c]["dv"] for c in tbl.ctx_order])
    assert np.array_equal(tbl.dv, expect_fab)
