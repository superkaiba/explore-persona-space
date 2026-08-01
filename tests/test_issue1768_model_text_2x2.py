"""#1768 inline (model x text) 2x2 round — decomposition-math invariants.

The load-bearing arithmetic of the round is the additive identity
``shift = text + function + interaction`` over a sha-joined row set. These
tests build tiny synthetic pooled stores (the real store schema, 8 rows, 4
dims) and drive the REAL `_decompose_arm` / `_leg_b_read` bodies, so a sign
flip or a mis-indexed join fails here instead of on a GPU box.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"

torch = pytest.importorskip("torch")


def _load_module():
    """Script-mode import: scripts/ on sys.path, module registered BEFORE exec
    (a `dataclasses.dataclass` in the module body dereferences
    `sys.modules[cls.__module__]`)."""
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    name = "issue1768_model_text_2x2"
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


MT = _load_module()
LAYERS = (14,)
D = 4


def _pooled(shas, spans, seed, n_dims=D):
    rng = np.random.default_rng(seed)
    return {
        "schema_version": 1,
        "row_sha": list(shas),
        "row_question_idx": list(range(len(shas))),
        "arms": {
            span: {
                layer: torch.tensor(
                    rng.normal(size=(len(shas), n_dims)).astype("float32"), dtype=torch.float16
                )
                for layer in LAYERS
            }
            for span in spans
        },
        "metadata": {"n_rows": len(shas)},
    }


def _write(path: Path, store) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(store, path)


@pytest.fixture
def cell(tmp_path):
    """Four synthetic cells for one content arm; base has one extra row."""
    arm = "imp-pers-con-lr3e5-s42"
    base_unit = "base_content"
    base_shas = [f"sha{i:03d}" for i in range(8)]
    arm_shas = base_shas[1:]  # the arm dropped row 0 (empty response)
    root = tmp_path / "out"
    _write(
        root / "corpus_capture" / base_unit / "pooled.pt",
        _pooled(base_shas, ("context", "response"), 1),
    )
    _write(
        root / "corpus_capture" / arm / "pooled.pt", _pooled(arm_shas, ("context", "response"), 2)
    )
    _write(root / MT.RTF_TREE / arm / "pooled_tf.pt", _pooled(arm_shas, ("context", "response"), 3))
    _write(root / "corpus_capture_tf" / arm / "pooled_tf.pt", _pooled(base_shas, ("response",), 4))
    cfg = MT.Cfg(out_root=root, phases=(), layers=LAYERS, smoke=True)
    return cfg, arm


def test_decomposition_is_additive_and_row_joined(cell, monkeypatch):
    cfg, arm = cell
    monkeypatch.setattr(
        MT, "_delta_direction", lambda *a, **k: (_ for _ in ()).throw(KeyError("no panel"))
    )
    out = MT._decompose_arm(cfg, arm)
    rec = out["layers"][str(LAYERS[0])]
    assert rec["n_rows"] == 7, rec["n_rows"]  # sha join drops the base-only row
    # the identity shift == text + function + interaction holds to float noise
    assert rec["identity_residual"] < 1e-9, rec["identity_residual"]
    # projection shares are additive by construction
    assert abs(sum(rec["proj_share"].values()) - 1.0) < 1e-9
    # per-row squared shares are NOT constrained to sum to 1 (cross terms) but
    # must be finite and non-negative
    for v in rec["per_row_sq_share"].values():
        assert v >= 0.0 and np.isfinite(v)
    assert rec["delta_reads"]["error"].startswith("KeyError")


def test_text_effect_uses_the_reverse_tree_not_the_arm_tree(cell, monkeypatch):
    """A sign/tree mix-up in the text effect is the round's worst failure mode:
    T must read the BASE-on-trained-text store, never the arm's own store."""
    cfg, arm = cell
    monkeypatch.setattr(
        MT, "_delta_direction", lambda *a, **k: (_ for _ in ()).throw(KeyError("x"))
    )
    layer = LAYERS[0]
    base = MT._store(cfg.out_root / "corpus_capture" / "base_content" / "pooled.pt")
    rev = MT._store(cfg.out_root / MT.RTF_TREE / arm / "pooled_tf.pt")
    b = MT._span(base, "response", layer)[1:]  # the 7 shared rows, base order
    r = MT._span(rev, "response", layer)
    expect = float(np.linalg.norm(r.mean(axis=0) - b.mean(axis=0)))
    rec = MT._decompose_arm(cfg, arm)["layers"][str(layer)]
    assert rec["norms"]["text"] == pytest.approx(expect, rel=1e-9, abs=1e-12)


def test_leg_b_delta_v_train_is_plus_minus_base(tmp_path, monkeypatch):
    arm = "imp-pers-con-lr3e5-s42"
    root = tmp_path / "out"
    layer = LAYERS[0]
    rng = np.random.default_rng(11)
    vp = rng.normal(size=D).astype("float32")
    v0 = rng.normal(size=D).astype("float32")
    _write(
        root / MT.BTF_TREE / arm / "tbar_plus.pt",
        {
            "tbar_plus": {layer: torch.tensor(vp)},
            "tbar_plus_even": None,
            "n_rows": 20,
            "delta_arm": arm,
        },
    )
    _write(
        root / "delta_tf" / arm / "tbar.pt",
        {"tbar": {layer: torch.tensor(v0)}, "tbar_even": None, "n_rows": 20},
    )
    cfg = MT.Cfg(out_root=root, phases=(), layers=(layer,))
    monkeypatch.setattr(
        MT, "_delta_direction", lambda *a, **k: (_ for _ in ()).throw(KeyError("x"))
    )
    write = {str(layer): np.asarray(vp - v0, dtype=np.float64)}
    rec = MT._leg_b_read(cfg, arm, write)["layers"][str(layer)]
    assert rec["norm_delta_v_train"] == pytest.approx(float(np.linalg.norm(vp - v0)), rel=1e-6)
    # a matched write identical to Delta v_train must read cosine 1
    assert rec["cos_delta_v_train_corpus_matched_write"] == pytest.approx(1.0, abs=1e-9)
    assert rec["delta_error"].startswith("KeyError")
