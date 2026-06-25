"""Issue #664 -- end-to-end smoke == the fleet driver with one cell (PASS_UNIFIED).

This is a THIN wrapper: the smoke is literally ``issue664_dispatch.py
--phase all --cells 1 --smoke`` -- SAME dispatcher, SAME per-cell subprocess
shape (build -> train_lora -> extract_store -> eval gen), SAME env injection,
SAME WandB/HF upload surface (HF upload no-ops under --smoke), SAME teardown,
SAME [phase=...] / sentinel contract. There is NO separate smoke code path, so
a smoke pass exercises EVERY production phase the dispatcher runs on the marker
canary cell (mk_default_contra_d1 -- the band-stop subprocess path).

The cell-subset parameterization (--cells / --smoke) threads through EVERY
phase the dispatcher executes:
  - P2.0 build: phase0() iterates `_select_cells(args)` (the same subset).
  - P2.1 train: run_all() trains `_select_cells(args)` (the same subset).
  - P2.2 extract+eval: iterates `_select_cells(args)`; the registry manifest is
    built over `cells` (the same subset), and the eval gen/judge surface derives
    from each cell's own behavior -- no phase re-enumerates the full 64-cell grid.
  - P2.3 upload: uploads `_select_cells(args)` artifacts only.
  - smoke assert (§10/§11 A7): the marker read-gauge readability read runs over
    the selected marker cells.

Run it directly OR call ``issue664_dispatch.py --cells 1 --smoke`` -- identical.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import issue664_dispatch as D


def main() -> int:
    # Force the one-cell smoke subset regardless of extra CLI noise.
    argv = ["--phase", "all", "--cells", "1", "--smoke"]
    # honor an explicit --cells N / --gpu-id N if the caller passes one.
    extra = sys.argv[1:]
    if extra:
        argv = ["--phase", "all", "--smoke", *extra]
    sys.argv = [sys.argv[0], *argv]
    return D.main()


if __name__ == "__main__":
    rc = main()
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc)
