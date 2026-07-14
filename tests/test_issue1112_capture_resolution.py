"""#1112 crash-fix r7 — p10 capture resolution + fan-out sibling reap.

Pins the attempt-5 crash class: ``_resolve_capture_model`` eagerly read
``<cell>/selection.json`` for EVERY non-base capture cell, but no phase ever
wrote one for ``m1_lora_band8`` (its artifact is the band-stopped FINAL
adapter — no rung selection by design), so the m1 capture unit died in ~5 s
with FileNotFoundError and ``_fanout_units`` abandoned the 3 sibling
front-ends mid-vLLM-engine-init (their orphaned EngineCores dumped 5-minute
handshake timeouts that masqueraded as an infra wedge — attempt 4).

Fixes under test (all real bodies; fakes only at the GPU/model boundary,
signature-mirroring by construction):

1. ``_resolve_capture_model`` m1 branch — resolves the SAME model identity
   phase_marker's m1 read used (``build_result.json["adapter_root"]``,
   merged) with NO selection.json required; the selection read moved to the
   one dose that uses it.
2. ``_write_m1_selection`` — provenance backfill on fresh AND resume paths.
3. ``phase_generic`` skip-branch selection backfill (same class: a crash
   between the build_result and selection writes strands the cell).
4. ``_reap_unit_groups`` — kills the whole unit process GROUP (uv -> python
   -> vLLM EngineCore), not just the direct child.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))


def _dispatch():
    import issue1112_dispatch as d

    return d


def _cfg(d, tmp_path: Path, cells: tuple[str, ...]):
    return d.Cfg(smoke=True, cells=cells, out_root=tmp_path, upload=False)


def _fake_merge_adapter(cfg, adapter_dir: str, merged_dir: Path) -> Path:
    """Signature-mirroring GPU-boundary fake: asserts the resolved adapter
    artifact actually exists on disk (the pod-state check), then 'merges'."""
    assert Path(adapter_dir).exists(), f"resolved adapter missing: {adapter_dir}"
    merged_dir.mkdir(parents=True, exist_ok=True)
    return merged_dir


def _fake_ensure_dir_tokenizer(model_dir: Path, base_model: str = "unused") -> bool:
    """Signature-mirroring boundary fake (real one loads the HF tokenizer)."""
    assert Path(model_dir).is_dir(), model_dir
    return False


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


# ── 1. m1 resolution (the attempt-5 crash: fails pre-fix, passes post-fix) ──


def test_m1_resolves_without_selection_json(tmp_path, monkeypatch):
    d = _dispatch()
    monkeypatch.setattr(d, "_merge_adapter", _fake_merge_adapter)
    cfg = _cfg(d, tmp_path, ("m1_lora_band8",))
    cell_root = tmp_path / "m1_lora_band8"
    adapter_root = cell_root / "train"
    adapter_root.mkdir(parents=True)
    _write_json(cell_root / "build_result.json", {"adapter_root": str(adapter_root)})
    # NO selection.json — the pod's actual on-disk state for m1 (attempt 5).
    model_path, cleanup = d._resolve_capture_model(cfg, "m1_lora_band8", "selected")
    assert model_path == str(cell_root / "merged_selected")
    assert cleanup == cell_root / "merged_selected"


def test_selected_dose_still_fails_loud_without_selection(tmp_path, monkeypatch):
    """The rung-selected cells KEEP the fail-loud read (no silent fallback)."""
    d = _dispatch()
    monkeypatch.setattr(d, "_merge_adapter", _fake_merge_adapter)
    cfg = _cfg(d, tmp_path, ("s2_lora_pos",))
    cell_root = tmp_path / "s2_lora_pos"
    train = cell_root / "train" / "checkpoint-14"
    train.mkdir(parents=True)
    _write_json(cell_root / "build_result.json", {"adapter_root": str(train.parent)})
    with pytest.raises(FileNotFoundError):
        d._resolve_capture_model(cfg, "s2_lora_pos", "selected")


# ── 2. the remaining-5 pod units all resolve against pod-shaped state ───────


def test_remaining_pod_cells_resolve(tmp_path, monkeypatch):
    """The 5 units missing pooled.pt on pod-1112 (attempt-5 state) resolve."""
    d = _dispatch()
    monkeypatch.setattr(d, "_merge_adapter", _fake_merge_adapter)
    monkeypatch.setattr(d, "_ensure_dir_tokenizer", _fake_ensure_dir_tokenizer)
    cfg = _cfg(d, tmp_path, d.C.ALL_TRAINED_CELLS)

    # m1: build_result only (no selection.json — the crasher)
    m1 = tmp_path / "m1_lora_band8"
    (m1 / "train").mkdir(parents=True)
    _write_json(m1 / "build_result.json", {"adapter_root": str(m1 / "train")})
    # s5: selection + LoRA rung (phase_generic wrote both on the pod)
    s5 = tmp_path / "s5_lora_generic"
    (s5 / "train" / "checkpoint-14").mkdir(parents=True)
    _write_json(s5 / "build_result.json", {"adapter_root": str(s5 / "train")})
    _write_json(s5 / "selection.json", {"step": 14})
    # m2: selection (p8 wrote it) + full-FT rung dir (loads directly)
    m2 = tmp_path / "m2_fullft_band8"
    (m2 / "train" / "checkpoint-8").mkdir(parents=True)
    _write_json(m2 / "build_result.json", {"adapter_root": str(m2 / "train")})
    _write_json(m2 / "selection.json", {"step": 8})

    expect = {
        ("m1_lora_band8", "selected"): str(m1 / "merged_selected"),
        ("s5_lora_generic", "selected"): str(s5 / "merged_selected"),
        ("m2_fullft_band8", "selected"): str(m2 / "train" / "checkpoint-8"),
        ("base_sycophancy", "base"): d.DEFAULT_BASE_MODEL,
        ("base_marker", "base"): d.DEFAULT_BASE_MODEL,
    }
    grid = d.capture_passes(cfg)
    for unit, want in expect.items():
        assert unit in grid, f"{unit} not enumerated by capture_passes"
        model_path, _cleanup = d._resolve_capture_model(cfg, *unit)
        assert model_path == want, (unit, model_path)


# ── 3. m1 selection.json provenance backfill (fresh + resume paths) ─────────


def test_phase_marker_backfills_m1_selection_on_resume(tmp_path):
    d = _dispatch()
    cfg = _cfg(d, tmp_path, ("m1_lora_band8",))
    cell_root = tmp_path / "m1_lora_band8"
    # pod state: marker_read.json exists (skip-completed), ΔG below-band
    _write_json(
        cell_root / "marker_read.json",
        {"model": "x", "n_probes": 20, "delta_logp_mean": 1.58},
    )
    out = d.phase_marker(cfg)
    assert out["m1_lora_band8"]["delta_logp_mean"] == 1.58
    sel = json.loads((cell_root / "selection.json").read_text())
    assert sel["step"] is None
    assert sel["in_band"] is False  # 1.58 < MARKER_GLOBAL_BAND[0]
    assert sel["fallback"] == "closest_approach"
    assert sel["policy"] == "band_stop_final_adapter"
    assert sel["delta_logp_mean"] == 1.58
    # idempotent: an existing selection.json is never overwritten
    (cell_root / "selection.json").write_text(json.dumps({"step": None, "marker": "keep"}))
    d.phase_marker(cfg)
    assert json.loads((cell_root / "selection.json").read_text())["marker"] == "keep"


# ── 4. generic-cell selection backfill on the resume path ───────────────────


def test_phase_generic_backfills_selection_on_resume(tmp_path):
    d = _dispatch()
    cfg = _cfg(d, tmp_path, ("s5_lora_generic",))
    cell_root = tmp_path / "s5_lora_generic"
    _write_json(
        cell_root / "build_result.json",
        {"adapter_root": str(cell_root / "train"), "matched_step": 14, "twin": "s2_lora_pos"},
    )
    d.phase_generic(cfg, {})  # skip branch — build exists, selection missing
    sel = json.loads((cell_root / "selection.json").read_text())
    assert sel == {"step": 14, "rate": None, "in_band": None, "fallback": "method-matched-step"}


# ── 5. sibling reap kills the WHOLE unit process tree ───────────────────────


def test_reap_unit_groups_kills_whole_process_tree():
    d = _dispatch()
    proc = subprocess.Popen(
        ["bash", "-c", "sleep 30 & sleep 30"],
        start_new_session=True,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    time.sleep(0.3)  # let bash fork the background sleep into the group
    pgid = os.getpgid(proc.pid)
    assert pgid == proc.pid  # start_new_session made the unit the group leader
    d._reap_unit_groups([proc])
    assert proc.poll() is not None, "direct child survived the reap"
    deadline = time.time() + 5
    while time.time() < deadline:
        try:
            os.killpg(pgid, 0)
        except ProcessLookupError:
            return  # whole group (background sleep included) is gone
        time.sleep(0.1)
    os.killpg(pgid, signal.SIGKILL)
    pytest.fail("process group survived _reap_unit_groups (grandchild leaked)")
