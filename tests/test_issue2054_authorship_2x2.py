"""Pins for the #2054 authorship x presentation 2x2 producer
(`scripts/issue2054_authorship_2x2.py`; plan §6.5 primary deliverable,
hypothesis H4; concern `authorship-2x2-producer-missing`).

Covers: (1) the full 2x2 (all four cells present) with hand-computed terms +
the gap = authorship + interaction identity, through BOTH the compute
function and the production `main()` entrypoint; (2) the fail-loud negative
case when the (c) transpose fits are absent (the state of the world until
the Phase-D capture round lands); (3) the a/b/d-missing fail-loud sibling;
(4) skipped-fold filtering + the op-variant (c) naming fallback.

Fixtures are tiny synthetic fit JSONs mirroring the REALIZED
`issue2054_fits.py` cell-payload schema (arm_reports.{arm}.per_fold[]
.r2_ambient — verified against the HF `issue2054_lattice/fits/` artifacts).
All writes go to tmp_path (never canonical eval_results/). No network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue2054_authorship_2x2 as a2x2  # noqa: E402

MODEL = "qwen2.5-7b-instruct"
CHAR = "char_vex"
FORM = "attrib_quoted"

# Hand-computed 3-fold example (identity: gap = authorship + interaction).
R2 = {
    "a": [0.50, 0.52, 0.54],
    "b": [0.40, 0.42, 0.44],
    "c": [0.45, 0.47, 0.49],
    "d": [0.30, 0.35, 0.40],
}
EXPECTED = {
    "authorship_c_minus_a": -0.05,
    "presentation_b_minus_a": -0.10,
    "interaction": -0.02,
    "gap_d_minus_b": -0.07,
}


def _fit_json(cell_key: str, fold_r2: list[float], *, add_skipped_fold: bool = False) -> dict:
    """A minimal fit JSON in the realized issue2054_fits.py payload shape."""
    per_fold = [
        {"fold": i, "r2_ambient": v, "n_train": 100, "n_val": 25} for i, v in enumerate(fold_r2)
    ]
    if add_skipped_fold:
        # Realized shape of a skipped fold: status row, NO r2_ambient key.
        per_fold.append({"fold": 99, "status": "skipped-empty-fold", "n_train": 0, "n_val": 0})
    arm_report = {
        "status": "ok",
        "per_fold": per_fold,
        "pooled": {"r2_ambient_mean": float(np.mean(fold_r2))},
    }
    return {
        "cell": cell_key,
        "arm_reports": {"context": arm_report, "prefix": json.loads(json.dumps(arm_report))},
    }


def _write_quad(fits_dir: Path, *, include_c: bool = True, include_b: bool = True) -> None:
    fits_dir.mkdir(parents=True, exist_ok=True)
    keys = {
        "a": f"conversation_paired_stories_assistant__inserted__chat__{MODEL}",
        "b": f"{CHAR}__inserted__{FORM}__{MODEL}",
        # (c) written under the OP-VARIANT naming (candidate 2) to pin the
        # phase_d-variant-keyed fallback resolution.
        "c": f"{CHAR}_op__cell_c__chat__{MODEL}",
        "d": f"{CHAR}__on_policy__{FORM}__{MODEL}",
    }
    for label, key in keys.items():
        if label == "c" and not include_c:
            continue
        if label == "b" and not include_b:
            continue
        (fits_dir / f"{key}.json").write_text(
            json.dumps(_fit_json(key, R2[label], add_skipped_fold=(label == "d"))),
            encoding="utf-8",
        )


def test_full_2x2_terms_and_identity(tmp_path):
    fits_dir = tmp_path / "fits"
    _write_quad(fits_dir)
    payload = a2x2.compute_2x2(
        fits_dir,
        [CHAR],
        [MODEL],
        [FORM],
        ["context", "prefix"],
        c_answer_form="attrib_quoted",
        bootstrap_draws=500,
        seed=7,
    )
    records = payload["records"]
    assert len(records) == 2  # context + prefix (both-arms standing rule)
    rec = next(r for r in records if r["arm"] == "context")
    assert rec["n_common_folds"] == 3  # the (d) skipped-fold row is filtered
    for name, expected in EXPECTED.items():
        term = rec["terms"][name]
        assert term["point"] == pytest.approx(expected, abs=1e-12), name
        lo, hi = term["ci95"]
        assert lo <= term["point"] <= hi, name
    # Identity: gap = authorship + interaction (paired per fold).
    assert rec["terms"]["gap_d_minus_b"]["point"] == pytest.approx(
        rec["terms"]["authorship_c_minus_a"]["point"] + rec["terms"]["interaction"]["point"],
        abs=1e-12,
    )
    # Ceilings carry both the fold mean and the artifact's own pooled mean.
    assert rec["ceilings"]["a"]["fold_mean"] == pytest.approx(np.mean(R2["a"]))
    assert rec["ceilings"]["a"]["pooled_r2_ambient_mean"] == pytest.approx(np.mean(R2["a"]))
    # The (c) cell resolved through the op-variant naming fallback.
    assert rec["cells"]["c"] == f"{CHAR}_op__cell_c__chat__{MODEL}"
    assert rec["byte_matched_c_d"] is True  # story_form == c_answer_form


def test_main_entrypoint_writes_deliverable(tmp_path, monkeypatch):
    fits_dir = tmp_path / "fits"
    _write_quad(fits_dir)
    out = tmp_path / "out" / "authorship_presentation_2x2.json"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue2054_authorship_2x2.py",
            "--fits-dir",
            str(fits_dir),
            "--out",
            str(out),
            "--characters",
            CHAR,
            "--models",
            MODEL,
            "--story-forms",
            FORM,
            "--bootstrap-draws",
            "200",
            "--seed",
            "7",
        ],
    )
    rc = a2x2.main()
    assert rc == 0
    payload = json.loads(out.read_text(encoding="utf-8"))
    assert payload["artifact"] == "authorship_presentation_2x2"
    assert len(payload["records"]) == 2
    assert payload["bootstrap"]["draws"] == 200
    assert "git_commit" in json.dumps(payload["metadata"]) or payload["metadata"]


def test_missing_c_fits_fail_loud(tmp_path):
    """(c) fits absent (the pre-Phase-D-capture state) -> SystemExit naming
    every candidate (c) path tried — never a silent partial 2x2."""
    fits_dir = tmp_path / "fits"
    _write_quad(fits_dir, include_c=False)
    with pytest.raises(SystemExit) as exc:
        a2x2.compute_2x2(
            fits_dir,
            [CHAR],
            [MODEL],
            [FORM],
            ["context"],
            c_answer_form="attrib_quoted",
            bootstrap_draws=100,
            seed=7,
        )
    msg = str(exc.value)
    assert "MISSING (c) transpose fit cell" in msg
    assert f"{CHAR}__cell_c__chat__{MODEL}.json" in msg  # candidate 1 named
    assert f"{CHAR}_op__cell_c__chat__{MODEL}.json" in msg  # candidate 2 named
    assert "Phase-D" in msg


def test_missing_b_fits_fail_loud(tmp_path):
    fits_dir = tmp_path / "fits"
    _write_quad(fits_dir, include_b=False)
    with pytest.raises(SystemExit) as exc:
        a2x2.compute_2x2(
            fits_dir,
            [CHAR],
            [MODEL],
            [FORM],
            ["context"],
            c_answer_form="attrib_quoted",
            bootstrap_draws=100,
            seed=7,
        )
    msg = str(exc.value)
    assert "MISSING a/b/d fit cell" in msg
    assert f"{CHAR}__inserted__{FORM}__{MODEL}.json" in msg
