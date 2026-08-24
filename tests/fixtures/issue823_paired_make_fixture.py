"""Deterministic fixture generator for tests/test_issue823_paired_full_ratio.py.

Writes, FLAT under tests/fixtures/ (the .gitignore negation ``!tests/fixtures/*.npz``
covers direct children only, so the npz fixtures live here, not in a subdirectory):

  issue823_paired_percontext_ladder.npz   tiny percontext store (5 arms x 28 layers x 80 ctx)
  issue823_paired_assignment.json         persona(i, k) = i mod k over 96 original contexts
  issue823_paired_summary.json            mixture_floor.implied_mixture_penalty stub
  issue823_paired_mixture_diffs.npz       per-arm difference-vector sidecar (d=6, variable
                                          per-persona mean shifts)
  issue823_paired_expected_default.json   the PRE-change paired script's default-path output
                                          on this fixture (provenance below)

Regenerate inputs:  uv run python tests/fixtures/issue823_paired_make_fixture.py
Expected-output provenance: ``issue823_paired_expected_default.json`` was produced by
running the PRE-parametrization ``scripts/issue823_shared_persona_paired.py`` (git
8a338b0e423f928c80e4723cd6ce7817f3ebfb7e, materialized via ``git show``) against this
fixture with ``repo_root()`` monkeypatched to a temp tree mirroring the canonical
ladder layout::

    git show 8a338b0e423f928c80e4723cd6ce7817f3ebfb7e:scripts/issue823_shared_persona_paired.py \
        > /tmp/issue823_paired_prechange.py
    uv run python tests/fixtures/issue823_paired_make_fixture.py \
        --paired-script /tmp/issue823_paired_prechange.py

After regenerating the INPUT fixtures you MUST regenerate the expected output with the
same pre-change code (or consciously re-anchor it against the then-current default
path) -- the stability test compares the CURRENT script's default path against it.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import shutil
import sys
import tempfile

import numpy as np

FIXTURE_DIR = pathlib.Path(__file__).resolve().parent
LADDER_REL = pathlib.Path("eval_results/issue_823/inconsistent_origin_ladder")

N_ORIG = 96  # assignment array length (original context count)
N_MASK = 80  # mask-surviving contexts
N_LAYERS = 28  # store layer axis; read-out layers 14/26/17 must index into it
ARM_NAMES = ("k1", "k2", "k4", "k8", "k16")
POOLED = (2, 4, 8, 16)
READ_OUT = (14, 26, 17)
D_DIFF = 6  # difference-vector dim in the sidecar
SEED = 8230


def build_inputs() -> None:
    """Write the four deterministic input fixtures (seeded; no expected output)."""
    rng = np.random.default_rng(SEED)
    dropped = rng.choice(N_ORIG, size=N_ORIG - N_MASK, replace=False)
    kept = sorted(set(range(N_ORIG)) - {int(x) for x in dropped})
    context_ids = np.array(kept, dtype=np.int64)

    ss_res = rng.uniform(50.0, 150.0, size=(len(ARM_NAMES), N_LAYERS, N_MASK))
    for j, name in enumerate(ARM_NAMES):
        if name != "k1":  # pooled arms a bit worse so paired diffs carry signal
            ss_res[j] += rng.uniform(0.0, 30.0, size=(N_LAYERS, N_MASK))
    ss_tot = ss_res + rng.uniform(200.0, 400.0, size=ss_res.shape)
    np.savez(
        FIXTURE_DIR / "issue823_paired_percontext_ladder.npz",
        arm_names=np.array(ARM_NAMES),
        context_ids=context_ids,
        p1_ss_res=ss_res,
        p1_ss_tot=ss_tot,
    )

    arms = {str(k): [i % k for i in range(N_ORIG)] for k in (1, *POOLED)}
    (FIXTURE_DIR / "issue823_paired_assignment.json").write_text(
        json.dumps({"arms": arms, "registered_rule": "persona(i, k) = i mod k"}, indent=2) + "\n"
    )

    implied = {
        f"k{k}:L{layer}": {"between_persona_mean_shift_energy": float(5.0 + k + 0.1 * layer)}
        for k in POOLED
        for layer in READ_OUT
    }
    (FIXTURE_DIR / "issue823_paired_summary.json").write_text(
        json.dumps({"mixture_floor": {"implied_mixture_penalty": implied}}, indent=2) + "\n"
    )

    # Sidecar: per-arm difference vectors with VARIABLE per-persona mean shifts (test (a)
    # needs E_draw to vary materially across resamples).
    sidecar: dict[str, np.ndarray] = {"layers": np.array(READ_OUT, dtype=np.int64)}
    for k in POOLED:
        rows = [int(c) for c in context_ids if int(c) % k != 0]
        personas = np.array([c % k for c in rows], dtype=np.int64)
        diffs = rng.normal(0.0, 1.0, size=(len(rows), len(READ_OUT), D_DIFF))
        for p in np.unique(personas):
            shift = rng.normal(0.0, 2.0, size=(len(READ_OUT), D_DIFF)) * (1.0 + 0.3 * float(p))
            diffs[personas == p] += shift
        sidecar[f"k{k}_diffs"] = diffs
        sidecar[f"k{k}_personas"] = personas
        sidecar[f"k{k}_context_ids"] = np.array(rows, dtype=np.int64)
    np.savez(FIXTURE_DIR / "issue823_paired_mixture_diffs.npz", **sidecar)


def assemble_fixture_repo(dest: pathlib.Path) -> pathlib.Path:
    """Copy the committed input fixtures into ``dest`` under the canonical ladder layout.

    Returns ``dest`` (usable as a monkeypatched ``repo_root()``); asserts the inputs exist.
    """
    ladder = dest / LADDER_REL
    ladder.mkdir(parents=True, exist_ok=True)
    pairs = (
        ("issue823_paired_percontext_ladder.npz", "percontext_ladder.npz"),
        ("issue823_paired_assignment.json", "assignment.json"),
        ("issue823_paired_summary.json", "ladder_analysis_summary.json"),
        ("issue823_paired_mixture_diffs.npz", "mixture_diffs.npz"),
    )
    for src, tgt in pairs:
        src_path = FIXTURE_DIR / src
        assert src_path.is_file(), f"missing committed fixture: {src_path}"
        shutil.copy(src_path, ladder / tgt)
    return dest


def run_paired_default(script_path: pathlib.Path, out_path: pathlib.Path) -> None:
    """Run a paired-script MODULE's default path against the fixture, writing ``out_path``.

    Loads ``script_path`` as a module, monkeypatches its ``repo_root`` to a temp tree
    assembled by ``assemble_fixture_repo`` (and ``git_commit`` to a fixed literal --
    the temp tree is not a git repo, and pinning it keeps the expected output fully
    deterministic), and calls ``main()`` with only ``--out``.
    """
    spec = importlib.util.spec_from_file_location("issue823_paired_under_fixture", script_path)
    assert spec is not None and spec.loader is not None, script_path
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    with tempfile.TemporaryDirectory() as td:
        root = assemble_fixture_repo(pathlib.Path(td))
        mod.repo_root = lambda: root
        mod.git_commit = lambda root: "fixture-git-commit"
        argv = sys.argv
        sys.argv = [str(script_path), "--out", str(out_path)]
        try:
            mod.main()
        finally:
            sys.argv = argv


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="issue823 paired-script fixture generator")
    ap.add_argument(
        "--paired-script",
        default=None,
        help=(
            "optional path to a paired-script version to run against the fixture; its "
            "default-path output is written to issue823_paired_expected_default.json "
            "(used ONCE with the pre-change script -- see the module docstring)"
        ),
    )
    ap.add_argument(
        "--skip-inputs", action="store_true", help="do not regenerate the input fixtures"
    )
    a = ap.parse_args()
    if not a.skip_inputs:
        build_inputs()
        print(f"wrote input fixtures under {FIXTURE_DIR}")
    if a.paired_script:
        expected = FIXTURE_DIR / "issue823_paired_expected_default.json"
        run_paired_default(pathlib.Path(a.paired_script), expected)
        print(f"wrote {expected}")
