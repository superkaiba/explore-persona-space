"""#2479 r2 pins for the phasef driver's equalized-n pass + per-unit progress lines.

Pins two r2 items on ``scripts/issue1345_char_phasef_driver.sh``:

- item 9 (equalized-n leg of the codex ``registered-analysis-incomplete``
  blocker): a production panel run (panel env set, ``--max-rows 0``, no
  ``--pilot-outdir``) UNCONDITIONALLY runs the equalized-n refit pass —
  n_eq = min ``n_matched`` across surviving ``char_2479_*_op`` primary
  ladders, each refit at ``--max-rows n_eq`` (resume-skipped when the
  ``_rows<n_eq>`` output exists).
- item 12 (long-loop-progress): the per-cell loop emits
  ``[p5] unit k/N <variant> elapsed=<s>s`` lines, skipped cells included.

Synthetic end-to-end: the driver runs against fake complete outputs + fake
slice-cache stems in tmp dirs, so every fill invocation resume-skips and the
new bash paths execute for real without fits, stores, or network. The panel
env points at the COMMITTED panel.json (registry rows only — no corpus text).
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DRIVER = REPO / "scripts" / "issue1345_char_phasef_driver.sh"
PANEL_JSON = REPO / "eval_results" / "issue_2479" / "panel.json"

CELLS = ["char_2479_helios_op", "char_2479_iris_op"]  # both src=r4op, model=instruct


def _fake_outputs(out_dir: Path, n_matched: dict[str, int], n_eq: int) -> None:
    for v, n in n_matched.items():
        (out_dir / f"cell_{v}__instruct_context_L19_reduced_s0.json").write_text("{}")
        (out_dir / f"cell_{v}__instruct_prefix_L19_reduced_s0.json").write_text("{}")
        (out_dir / f"ladder_r4op__{v}__instruct_context_L19_reduced_s0_nd2.json").write_text(
            json.dumps({"n_matched": n})
        )
        (
            out_dir / f"ladder_r4op__{v}__instruct_context_L19_reduced_s0_nd2_rows{n_eq}.json"
        ).write_text("{}")


def test_phasef_eqn_pass_and_progress_lines(tmp_path):
    stage_root = tmp_path / "stage"
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    for d in (stage_root, cache_dir, out_dir):
        d.mkdir()
    # r4op/instruct source presence: the preflight accepts the context-arm L19
    # slice-cache stem (existence-checked only).
    (cache_dir / "instruct_stories_paired_op_s_context_L19.pt").write_text("")
    _fake_outputs(out_dir, {CELLS[0]: 700, CELLS[1]: 650}, n_eq=650)

    env = dict(os.environ)
    env["EPM_I2479_CHAR_PANEL_JSON"] = str(PANEL_JSON)
    proc = subprocess.run(
        [
            "bash",
            str(DRIVER),
            "--cells",
            *CELLS,
            "--stage-root",
            str(stage_root),
            "--cache-dir",
            str(cache_dir),
            "--out-dir",
            str(out_dir),
            "--min-free-gb",
            "0",
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=REPO,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-3000:]
    # item 12: per-unit progress lines, skipped cells included, k reaches N.
    assert "[p5] unit 1/2 char_2479_helios_op" in out, out[-3000:]
    assert "[p5] unit 2/2 char_2479_iris_op" in out, out[-3000:]
    assert "elapsed=" in out
    # item 9: equalized-n over the surviving op ladders — min(700, 650) = 650,
    # both refits resume-skipped on the pre-created _rows650 outputs.
    assert "n_eq=650" in out, out[-3000:]
    assert "[eqn] unit 1/2 char_2479_helios_op: exists — resume" in out
    assert "[eqn] unit 2/2 char_2479_iris_op: exists — resume" in out
    assert "eqn=ok (n_eq=650, 2 ladders)" in out
    # no fill invocation happened (everything resume-skipped).
    assert "WITHIN-CELL FIT" not in out and "LADDER PAIR" not in out


def test_phasef_eqn_skipped_in_pilot_mode(tmp_path):
    """--pilot-outdir (P0) must NOT trigger the equalized pass (panel bypass ban)."""
    stage_root = tmp_path / "stage"
    cache_dir = tmp_path / "cache"
    out_dir = tmp_path / "out"
    for d in (stage_root, cache_dir, out_dir):
        d.mkdir()
    (cache_dir / "instruct_stories_paired_op_s_context_L19.pt").write_text("")
    _fake_outputs(out_dir, {CELLS[0]: 700, CELLS[1]: 650}, n_eq=650)
    env = dict(os.environ)
    env["EPM_I2479_CHAR_PANEL_JSON"] = str(PANEL_JSON)
    proc = subprocess.run(
        [
            "bash",
            str(DRIVER),
            "--cells",
            *CELLS,
            "--stage-root",
            str(stage_root),
            "--cache-dir",
            str(cache_dir),
            "--out-dir",
            str(out_dir),
            "--min-free-gb",
            "0",
            "--pilot-outdir",
            str(out_dir),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
        cwd=REPO,
    )
    out = proc.stdout + proc.stderr
    assert proc.returncode == 0, out[-3000:]
    assert "n_eq=" not in out
    assert "eqn=skipped (non-production or panel env absent)" in out
