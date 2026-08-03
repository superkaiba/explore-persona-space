"""#1900 crash-fix r7 pins: worker fs parent-dir class + the --fs-dryrun tactic.

Fellows job 16100: workers 0+3 died FileNotFoundError at
``validation/tf_margin_{side}.jsonl`` append-open — pathlib append-open
creates no parents and the ``validation/`` dir was created in no process.
Pins: (a) ``ensure_out_dirs`` (the r7 process-level floor called at main()
AND worker_main() entry) creates every OUT_DIRS subdir so a fresh worker
process can append-open the P1c JSONL (the exact job-16100 crash op);
(b) ``run_fs_dryrun`` exits clean, exercises EVERY item phase
``build_items`` emits across the simulated 4-slot worker layout, and
removes its scratch tree on success (a stub must never satisfy a
production resume predicate).

CPU-tiny: imports the GPU driver module only (heavy imports are
function-level deferred); the dryrun's pandas/torch stub writes run in
seconds.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import issue1900_gpu as G  # noqa: E402


def _cfg(tmp_path: Path) -> G.Cfg:
    return G.Cfg(
        out_root=tmp_path / "out", stage_root=tmp_path / "hf_dl", smoke=False, upload=False
    )


def test_ensure_out_dirs_enables_fresh_process_p1c_append_open(tmp_path):
    """The job-16100 crash op succeeds in a bare tree after ensure_out_dirs."""
    cfg = _cfg(tmp_path)
    G.ensure_out_dirs(cfg)
    for name in G.OUT_DIRS:
        assert (cfg.out_root / name).is_dir(), name
    assert (cfg.out_root / "anchors" / "post").is_dir()
    assert (cfg.out_root / "logs").is_dir()
    out, done = G.p1c_out_paths(cfg, "arm")
    with out.open("a", encoding="utf-8") as fh:  # crashed FileNotFoundError pre-r7
        fh.write("{}\n")
    assert out.is_file() and not done.exists()


def test_fs_dryrun_covers_every_item_phase_and_cleans_up(tmp_path):
    cfg = _cfg(tmp_path)
    cfg.out_root.mkdir(parents=True, exist_ok=True)  # mirrors main()
    summary = G.run_fs_dryrun(cfg)
    # every worker item phase build_items emits is exercised >= once
    assert set(summary["phases"]) == {"p1a", "p1b_base", "p1b_post", "p1c", "p1d"}
    assert all(v >= 1 for v in summary["phases"].values())
    assert summary["n_slots"] == 4
    assert summary["n_paths_written"] > 0
    assert summary["n_upload_enumerated"] > 0
    # scratch removed on success — stubs must never satisfy resume predicates
    assert not (cfg.out_root / "fs_dryrun").exists()
    # production out_root untouched beyond the (empty) root itself
    assert list(cfg.out_root.iterdir()) == []


def test_fs_dryrun_item_registry_matches_production_shape(tmp_path):
    """The dryrun's synthetic registry drives the REAL build_items (no fork)."""
    cfg = _cfg(tmp_path)
    entries = G._dryrun_entries()
    items = G.build_items(cfg, entries)
    phases = {it["phase"] for it in items}
    assert phases == {"p1a", "p1b_base", "p1b_post", "p1c", "p1d"}
    # P1C_ARM present in the synthetic registry => p1c emitted with smoke=False
    assert any(a["arm_id"] == G.P1C_ARM for a in entries)
    assert sum(1 for it in items if it["phase"] == "p1c") == 2  # arm + base sides
