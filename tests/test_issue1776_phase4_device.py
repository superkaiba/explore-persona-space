"""#1776 crash-fix r11 pins (p4_energy CUDA device placement).

On the pod (args.device=cuda) the four probe producers of ``cmd_energy`` load
on CPU (``map_location="cpu"``) while ``load_dict``/``_cov_factor`` place the
dictionaries + cov factors on cuda; the cov null family's ``z @ cov_half.T``
(``null_probe_rows``) then crashed wrapper_CUDA_mm mid-run — a branch the
CPU-only VM smoke structurally cannot exercise (everything cpu there).

These tests pin the r11 fix WITHOUT a GPU, using torch's ``meta`` device as
the second device:

  1. ``energy_read``'s same-device assert fires BEFORE any mm and NAMES the
     offending tensor (fails pre-fix by message: pre-fix the crash was torch's
     "mat2 is on cuda:0 ... on cpu" internals string);
  2. ``_to_device_sets`` actually MOVES every value (cpu -> meta) and casts
     total float32 — the single move-to-device site is functional;
  3. source-order proof over ``cmd_energy``: every ``read_sets[``/
     ``write_sets[`` producer assignment precedes the single
     ``_to_device_sets`` move site, which precedes the first ``energy_read``
     consumption — every audit-table tensor flows through the move;
  4. co-located happy path: all-cpu ``energy_read`` with a cov factor runs
     ALL THREE null families (the cov mm included) to a well-formed record.
"""

from __future__ import annotations

import argparse
import inspect
import re
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import issue1776_phase4 as P4


def _ns(**kw) -> argparse.Namespace:
    base = {"k": 2, "n_draws": 2, "chunk": 8, "seed": 0}
    base.update(kw)
    return argparse.Namespace(**base)


def test_energy_read_device_assert_fires_naming_offender():
    x = torch.randn(3, 8)
    d = torch.randn(16, 8)
    cov_meta = torch.zeros(8, 8, device="meta")
    with pytest.raises(AssertionError, match=r"device mismatch.*cov_half"):
        P4.energy_read("t", x, d, _ns(), cov_meta)


def test_energy_read_device_assert_covers_probe_vs_dict():
    x_meta = torch.zeros(3, 8, device="meta")
    d = torch.randn(16, 8)
    with pytest.raises(AssertionError, match=r"device mismatch.*x_probe"):
        P4.energy_read("t", x_meta, d, _ns(), None)


def test_to_device_sets_moves_and_casts_every_value():
    sets = {"a": torch.zeros(2, 3, dtype=torch.float16), "b": torch.zeros(4, 3)}
    out = P4._to_device_sets(sets, "meta")
    assert set(out) == {"a", "b"}
    for v in out.values():
        assert v.device.type == "meta", v.device
        assert v.dtype == torch.float32, v.dtype
    # cpu target: identity placement, dtype still normalized
    out_cpu = P4._to_device_sets(sets, "cpu")
    assert all(v.device.type == "cpu" and v.dtype == torch.float32 for v in out_cpu.values())


def test_cmd_energy_probe_sets_flow_through_single_move_site():
    src = inspect.getsource(P4.cmd_energy)
    move_r = src.index("read_sets = _to_device_sets(read_sets, dev)")
    move_w = src.index("write_sets = _to_device_sets(write_sets, dev)")
    first_use = src.index("energy_read(")
    for m in re.finditer(r"read_sets\[", src):
        assert m.start() < move_r, "read_sets producer after the move site"
    for m in re.finditer(r"write_sets\[", src):
        assert m.start() < move_w, "write_sets producer after the move site"
    assert move_r < first_use and move_w < first_use, "move site must precede consumption"


def test_energy_read_colocated_runs_all_three_null_families():
    gen = torch.Generator().manual_seed(7)
    d = torch.randn(16, 8, generator=gen)
    d = d / d.norm(dim=1, keepdim=True)
    x = torch.randn(3, 8, generator=gen)
    cov = torch.eye(8)
    rec = P4.energy_read("t", x, d, _ns(), cov)
    assert set(rec["nulls"]) == {"rotation", "isotropic", "cov"}
    for nl in rec["nulls"].values():
        assert 0.0 <= nl["pursuit_p975"] <= 1.0, nl
