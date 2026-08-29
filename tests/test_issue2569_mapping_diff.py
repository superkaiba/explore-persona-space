from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "issue2569_mapping_diff", ROOT / "scripts" / "issue2569_mapping_diff.py"
)
assert SPEC and SPEC.loader
MD = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MD)


def test_factorial_diagonal_is_writer_plus_encoder() -> None:
    rng = np.random.default_rng(1)
    cells = {
        name: MD.Affine(rng.normal(size=(5, 5)), rng.normal(size=5))
        for name in MD.CELL_NAMES
    }
    context = rng.normal(size=(20, 5))
    contrasts = MD.factorial_affines(cells)
    observed = MD.factorial_arrays(
        {name: cell.predict(context) for name, cell in cells.items()}
    )
    for name in MD.CONTRAST_NAMES:
        assert np.allclose(contrasts[name].predict(context), observed[name])
    assert np.allclose(
        contrasts["diagonal"].predict(context),
        contrasts["writer"].predict(context)
        + contrasts["encoder"].predict(context),
    )


def test_transformed_llama_affine_matches_explicit_centered_path() -> None:
    rng = np.random.default_rng(2)
    dq, dl = 4, 6
    rc, _ = np.linalg.qr(rng.normal(size=(dl, dq)))
    ra, _ = np.linalg.qr(rng.normal(size=(dl, dq)))
    alignment = MD.FixedAlignment(
        R_context=rc.T,
        R_answer=ra.T,
        q_context_mean=rng.normal(size=dq),
        l_context_mean=rng.normal(size=dl),
        q_answer_mean=rng.normal(size=dq),
        l_answer_mean=rng.normal(size=dl),
    )
    payload = MD.OP.MapPayload(
        layer=16,
        path=Path("<test>"),
        W=rng.normal(size=(dl, dl)),
        xmu=rng.normal(size=dl),
        xsd=np.exp(rng.normal(size=dl)),
        ymu=rng.normal(size=dl),
        selected_lambda=1.0,
        raw={},
    )
    context = rng.normal(size=(17, dq))
    transformed = MD.transform_l_affine(payload, alignment).predict(context)
    explicit = alignment.l_answer_to_q(
        MD.OP.predict(payload, alignment.q_context_to_l(context))
    )
    assert np.allclose(transformed, explicit, rtol=1e-11, atol=1e-11)


def test_row_pairing_permutation_detects_true_pairing() -> None:
    rng = np.random.default_rng(3)
    truth = rng.normal(size=(40, 9))
    pred = truth + 0.05 * rng.normal(size=truth.shape)
    result = MD.permutation_null(pred, truth, draws=199, seed=4, device="cpu")
    assert result["flat_cosine"]["observed"] > 0.99
    assert result["flat_cosine"]["p_ge"] == 1 / 200
    assert result["pooled_r2"]["p_ge"] == 1 / 200
