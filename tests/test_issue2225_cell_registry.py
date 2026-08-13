"""Issue #2225 — cell-registry invariants (plan §4.5, exactly 81 finetunes).

Pins the declarative CELL REGISTRY in ``scripts/issue2225_train.py``:
the total count, the per-config grid sizes, the dataset coverage, the steered-trait
map, and the slug<->fields consistency. Import is CHEAP — the training script defers
every heavy import (torch / transformers / trl / issue778_*) inside its functions, so
this test never pulls a GPU stack.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

_SCRIPTS = Path(__file__).resolve().parent.parent / "scripts"


def _load():
    spec = importlib.util.spec_from_file_location(
        "issue2225_train", _SCRIPTS / "issue2225_train.py"
    )
    mod = importlib.util.module_from_spec(spec)
    sys.modules["issue2225_train"] = mod
    spec.loader.exec_module(mod)
    return mod


M = _load()


def test_registry_has_exactly_81_cells():
    cells = M.build_cell_registry()
    assert len(cells) == 81, len(cells)
    assert M.EXPECTED_CELL_COUNT == 81


def test_slugs_are_unique():
    slugs = [c.slug for c in M.build_cell_registry()]
    assert len(slugs) == len(set(slugs)), "duplicate slugs"


def test_per_config_cell_counts():
    """The §4.5 arithmetic: 16+12+16+12+12+3+3+3+3+1 = 81."""
    counts: dict[str, int] = {}
    for c in M.build_cell_registry():
        counts[c.config] = counts.get(c.config, 0) + 1
    expected = {
        "A": 16,  # 4 datasets x 4 L1 coefs
        "B": 12,  # 4 datasets x 3 multilayer coefs
        "C": 16,  # 4 datasets x 4 L1 coefs
        "D": 12,  # 4 datasets x 3 multilayer coefs
        "E": 12,  # 4 datasets x 3 multilayer coefs
        "F": 3,  # evil x 3 attribution coefs
        "G": 3,
        "I": 3,
        "P": 3,
        "H": 1,  # evil, no coef
    }
    assert counts == expected, counts


def test_grids_match_plan_4_5():
    """L1 grid {0.5,1.5,3,5}; L2/L3 {0.25,0.75,1.5}; attribution {0.5,1.5,3}."""
    cells = M.build_cell_registry()
    by_config: dict[str, set] = {}
    for c in cells:
        by_config.setdefault(c.config, set()).add(c.coef)
    assert by_config["A"] == {0.5, 1.5, 3.0, 5.0}
    assert by_config["C"] == {0.5, 1.5, 3.0, 5.0}
    assert by_config["B"] == {0.25, 0.75, 1.5}
    assert by_config["D"] == {0.25, 0.75, 1.5}
    assert by_config["E"] == {0.25, 0.75, 1.5}
    for cfg in ("F", "G", "I", "P"):
        assert by_config[cfg] == {0.5, 1.5, 3.0}, cfg
    assert by_config["H"] == {None}


def test_dataset_coverage():
    """Core configs A-E span all 4 datasets; F/G/H/I/P are evil-only (§4.5)."""
    cells = M.build_cell_registry()
    by_config: dict[str, set] = {}
    for c in cells:
        by_config.setdefault(c.config, set()).add(c.dataset)
    four = {"evil", "sycophancy", "hallucination", "mistake_opinions"}
    for cfg in ("A", "B", "C", "D", "E"):
        assert by_config[cfg] == four, cfg
    for cfg in ("F", "G", "H", "I", "P"):
        assert by_config[cfg] == {"evil"}, cfg


def test_steered_trait_map():
    """mistake_opinions cells steer evil; the single-trait corpora steer their own."""
    for c in M.build_cell_registry():
        if c.dataset == "mistake_opinions":
            assert c.steered_trait == "evil", c.slug
        else:
            assert c.steered_trait == c.dataset, c.slug


def test_variant_mask_layer_per_config():
    """Slug decode (e{n}s{n}l{n}) maps to (variant, mask_mode, layer_spec)."""
    spec_by_config = {
        "A": ("E1", "all", "L1"),
        "B": ("E1", "all", "L3"),
        "C": ("E2", "context", "L1"),
        "D": ("E2", "context", "L2"),
        "E": ("E2", "context", "L3"),
        "F": ("E1", "context", "L1"),
        "G": ("E2", "all", "L1"),
        "I": ("E1", "response", "L1"),
        "P": ("E3", "prefix", "L1"),
    }
    for c in M.build_cell_registry():
        if c.config == "H":
            assert c.prompt_mode
            assert c.variant is None and c.mask_mode is None and c.layer_spec is None
            assert c.coef is None
            continue
        assert not c.prompt_mode
        assert (c.variant, c.mask_mode, c.layer_spec) == spec_by_config[c.config], c.slug


def test_pilot_is_eight_evil_l1_cells():
    """The §7 P0 gate: A + C at the 4 L1 coefficients on evil II = 8 cells."""
    pilot = M.pilot_cells()
    assert len(pilot) == 8, len(pilot)
    assert {c.config for c in pilot} == {"A", "C"}
    assert {c.dataset for c in pilot} == {"evil"}
    assert {c.coef for c in pilot} == {0.5, 1.5, 3.0, 5.0}


def test_cells_by_slug_round_trips():
    by_slug = M.cells_by_slug()
    assert len(by_slug) == 81
    for slug, cell in by_slug.items():
        assert cell.slug == slug


def test_mask_modes_are_valid_steer_train_modes():
    """Every steered cell's mask_mode is one of steer_train's MASK_MODES."""
    valid = {"all", "context", "response", "prefix"}
    for c in M.build_cell_registry():
        if not c.prompt_mode:
            assert c.mask_mode in valid, c.slug


# ── §7 octave-shift re-pilot: scaled-cell synthesis + resolution (unit 5) ──────


def _ns(**kw):
    """argparse-shaped namespace for _resolve_cells (defaults = production)."""
    import argparse

    defaults = dict(
        pilot=False, cells=None, smoke=False, coef_scale=None, pilot_coefs=None, pilot_configs=None
    )
    defaults.update(kw)
    return argparse.Namespace(**defaults)


def test_synth_cell_matches_registry_cell_on_registry_coef():
    reg = M.cells_by_slug()["A__evil__c3.0"]
    assert M.synth_cell("A", "evil", 3.0) == reg  # frozen dataclass equality


def test_synth_cell_canonical_slug_and_fields():
    c = M.synth_cell("A", "evil", 0.25)
    assert c.slug == "A__evil__c0.25"
    assert (c.variant, c.mask_mode, c.layer_spec) == ("E1", "all", "L1")
    assert c.steered_trait == "evil" and not c.prompt_mode


def test_synth_cell_refuses_bad_inputs():
    import pytest

    with pytest.raises(ValueError, match="unknown config"):
        M.synth_cell("Z", "evil", 1.0)
    with pytest.raises(ValueError, match="prompt-mode"):
        M.synth_cell("H", "evil", 1.0)
    with pytest.raises(ValueError, match="not in config"):
        M.synth_cell("F", "sycophancy", 1.0)  # F is evil-only
    with pytest.raises(ValueError, match="finite and > 0"):
        M.synth_cell("A", "evil", 0.0)
    with pytest.raises(ValueError, match="finite and > 0"):
        M.synth_cell("A", "evil", float("nan"))


def test_resolve_cell_registry_hit_and_scaled_miss():
    assert M.resolve_cell("A__evil__c3.0") is M.resolve_cell("A__evil__c3.0") or True
    assert M.resolve_cell("A__evil__c3.0") == M.cells_by_slug()["A__evil__c3.0"]
    scaled = M.resolve_cell("C__evil__c0.75")
    assert scaled.config == "C" and scaled.coef == 0.75
    assert scaled.slug == "C__evil__c0.75"


def test_resolve_cell_refuses_noncanonical_and_unknown():
    import pytest

    with pytest.raises(ValueError, match="non-canonical"):
        M.resolve_cell("A__evil__c2.50")  # canonical is c2.5
    with pytest.raises(ValueError, match="unknown cell slug"):
        M.resolve_cell("not_a_slug")
    with pytest.raises(ValueError, match="prompt-mode"):
        M.resolve_cell("H__evil__c1.0")  # H has no coefficient


def test_resolve_cells_pilot_coef_scale_halves_grid():
    cells = M._resolve_cells(_ns(pilot=True, coef_scale=0.5))
    assert len(cells) == 8
    assert {c.coef for c in cells} == {0.25, 0.75, 1.5, 2.5}
    assert {c.config for c in cells} == {"A", "C"}
    # x2 shift: scaled coefs landing back on registry values dedupe by slug
    doubled = M._resolve_cells(_ns(pilot=True, coef_scale=2.0))
    assert {c.coef for c in doubled} == {1.0, 3.0, 6.0, 10.0}
    assert M.cells_by_slug()["A__evil__c3.0"] in doubled


def test_resolve_cells_pilot_configs_subset_and_pilot_coefs():
    cells = M._resolve_cells(_ns(pilot=True, pilot_configs="A", coef_scale=0.5))
    assert len(cells) == 4 and {c.config for c in cells} == {"A"}
    replaced = M._resolve_cells(_ns(pilot=True, pilot_coefs="0.1,0.2"))
    assert {c.coef for c in replaced} == {0.1, 0.2} and len(replaced) == 4


def test_resolve_cells_pilot_flags_require_pilot():
    import pytest

    with pytest.raises(ValueError, match="require --pilot"):
        M._resolve_cells(_ns(coef_scale=0.5))
    with pytest.raises(ValueError, match="subset"):
        M._resolve_cells(_ns(pilot=True, pilot_configs="A,Z"))


def test_argparser_coef_scale_and_pilot_coefs_mutually_exclusive():
    import pytest

    ap = M.build_argparser()
    with pytest.raises(SystemExit):
        ap.parse_args(["--pilot", "--coef-scale", "0.5", "--pilot-coefs", "0.1,0.2"])
    args = ap.parse_args(["--pilot", "--coef-scale", "0.5", "--pilot-configs", "A"])
    assert args.coef_scale == 0.5 and args.pilot_configs == "A"


def test_eval_gen_resolve_targets_scaled_fallback():
    import importlib.util as _ilu

    spec = _ilu.spec_from_file_location("issue2225_eval_gen", _SCRIPTS / "issue2225_eval_gen.py")
    eg = _ilu.module_from_spec(spec)
    sys.modules["issue2225_eval_gen"] = eg
    spec.loader.exec_module(eg)
    got = eg.resolve_targets(["base", "A__evil__c3.0", "A__evil__c0.25"])
    assert [t.tag for t in got] == ["base", "A__evil__c3.0", "A__evil__c0.25"]
    assert got[2].kind == "cell" and got[2].dataset == "evil"
    assert got[2].traits == ("evil",)
    import pytest

    with pytest.raises(ValueError, match="unknown eval-target tag"):
        eg.resolve_targets(["A__evil__c2.50"])  # non-canonical spelling refused


# ── r2 blocker 1: manifest-bound resume predicate (g2 Critical 1 + Concern 2) ──


def _resume_env(tmp_path: Path):
    """(cell, ckpt_root, dataset_root, directions_dir) with real fingerprint inputs."""
    cell = M.cells_by_slug()["A__evil__c3.0"]
    dataset_root = tmp_path / "ds"
    directions_dir = tmp_path / "dirs"
    ckpt_root = tmp_path / "ckpt"
    dpath = dataset_root / cell.dataset / f"{M.DATASET_VERSION}.jsonl"
    dpath.parent.mkdir(parents=True)
    dpath.write_text('{"messages": []}\n')
    directions_dir.mkdir()
    (directions_dir / f"{cell.steered_trait}_{cell.variant}.pt").write_bytes(b"direction-bytes")
    (ckpt_root / cell.slug).mkdir(parents=True)
    return cell, ckpt_root, dataset_root, directions_dir


def _write_adapter_files(ckpt_root: Path, cell, payload: bytes = b"adapter-v1") -> Path:
    out_dir = ckpt_root / cell.slug
    (out_dir / "adapter_config.json").write_text("{}")
    (out_dir / "adapter_model.safetensors").write_bytes(payload)
    return out_dir


def test_should_skip_refuses_stale_adapter_after_crashed_retrain(tmp_path):
    """FAILS PRE-FIX (r2 blocker 1): START manifest + bare adapter presence must
    NOT skip — the presence-based leg shipped a prior fingerprint's adapter
    after a crashed retrain (g2 Critical 1 trace)."""
    cell, ckpt_root, dataset_root, directions_dir = _resume_env(tmp_path)
    # Crashed-retrain state: START manifest under the CURRENT fingerprint (no
    # save-time `completed` record) + a PRIOR run's adapter files on disk.
    fp = M.cell_fingerprint(cell, dataset_root, directions_dir)
    M._write_manifest(ckpt_root, cell, fp)
    _write_adapter_files(ckpt_root, cell, b"stale-prior-fingerprint-adapter")
    assert M.should_skip(cell, ckpt_root, dataset_root, directions_dir) is False


def test_should_skip_completed_and_uploaded_skips_offline(tmp_path):
    cell, ckpt_root, dataset_root, directions_dir = _resume_env(tmp_path)
    fp = M.cell_fingerprint(cell, dataset_root, directions_dir)
    M._write_manifest(ckpt_root, cell, fp)
    out_dir = _write_adapter_files(ckpt_root, cell)
    M.mark_manifest_completed(ckpt_root, cell, out_dir)
    M.mark_manifest_uploaded(ckpt_root, cell)
    # uploaded=True short-circuits before any HF call -> fully offline skip.
    assert M.should_skip(cell, ckpt_root, dataset_root, directions_dir) is True


def test_should_skip_refuses_sha_mismatched_adapter(tmp_path):
    """The completed record binds BYTES: an adapter overwritten after the
    save-time sha record must re-run, never skip."""
    cell, ckpt_root, dataset_root, directions_dir = _resume_env(tmp_path)
    fp = M.cell_fingerprint(cell, dataset_root, directions_dir)
    M._write_manifest(ckpt_root, cell, fp)
    out_dir = _write_adapter_files(ckpt_root, cell)
    M.mark_manifest_completed(ckpt_root, cell, out_dir)
    (out_dir / "adapter_model.safetensors").write_bytes(b"overwritten-bytes")
    assert M.should_skip(cell, ckpt_root, dataset_root, directions_dir) is False


def test_should_skip_fingerprint_mismatch_still_reruns_completed_cell(tmp_path):
    """A completed cell under an OLD fingerprint re-runs when any input changes."""
    cell, ckpt_root, dataset_root, directions_dir = _resume_env(tmp_path)
    fp = M.cell_fingerprint(cell, dataset_root, directions_dir)
    M._write_manifest(ckpt_root, cell, fp)
    out_dir = _write_adapter_files(ckpt_root, cell)
    M.mark_manifest_completed(ckpt_root, cell, out_dir)
    M.mark_manifest_uploaded(ckpt_root, cell)
    # direction bytes change -> fingerprint changes -> re-run.
    (directions_dir / f"{cell.steered_trait}_{cell.variant}.pt").write_bytes(b"new-direction")
    assert M.should_skip(cell, ckpt_root, dataset_root, directions_dir) is False


def test_should_skip_redrives_upload_on_local_done_hf_incomplete(tmp_path, monkeypatch):
    """g2 Concern 2: local-done + never-uploaded -> the skip path RE-DRIVES the
    per-cell upload (#664 contract) and marks the manifest uploaded."""
    cell, ckpt_root, dataset_root, directions_dir = _resume_env(tmp_path)
    fp = M.cell_fingerprint(cell, dataset_root, directions_dir)
    M._write_manifest(ckpt_root, cell, fp)
    out_dir = _write_adapter_files(ckpt_root, cell)
    M.mark_manifest_completed(ckpt_root, cell, out_dir)

    calls: list[tuple] = []

    def fake_hf_files_present(cell_arg) -> bool:  # network boundary (signature-mirrored)
        return False

    def fake_upload_cell_adapter(
        out_dir_arg, cell_slug: str, hf_prefix: str = M.ADAPTERS_HF_PREFIX
    ) -> str:  # hub boundary (signature-mirrored, incl. the fu1 hf_prefix seam)
        assert hf_prefix == M.ADAPTERS_HF_PREFIX  # parent cells keep the parent prefix
        calls.append((Path(out_dir_arg), cell_slug))
        return f"https://hf/{cell_slug}"

    monkeypatch.setattr(M, "_hf_files_present", fake_hf_files_present)
    monkeypatch.setattr(M, "_upload_cell_adapter", fake_upload_cell_adapter)
    assert M.should_skip(cell, ckpt_root, dataset_root, directions_dir) is True
    assert calls == [(out_dir, cell.slug)]
    stored = M._read_manifest(ckpt_root, cell)
    assert stored["uploaded"] is True
    # second resume: uploaded flag short-circuits, no second upload.
    assert M.should_skip(cell, ckpt_root, dataset_root, directions_dir) is True
    assert len(calls) == 1


def test_should_skip_redrives_upload_even_when_stale_hf_files_present(tmp_path, monkeypatch):
    """FAILS PRE-FIX (r2 g1 Concern 1): `uploaded` flag absent + HF files
    already present at the slug prefix (a PRIOR fingerprint's bytes) must STILL
    re-upload — presence never blesses stale bytes on the re-drive leg; the
    upload is an idempotent overwrite at the same prefix."""
    cell, ckpt_root, dataset_root, directions_dir = _resume_env(tmp_path)
    fp = M.cell_fingerprint(cell, dataset_root, directions_dir)
    M._write_manifest(ckpt_root, cell, fp)
    out_dir = _write_adapter_files(ckpt_root, cell)
    M.mark_manifest_completed(ckpt_root, cell, out_dir)

    calls: list[tuple] = []

    def fake_hf_files_present(cell_arg) -> bool:  # stale F1-era files present
        return True

    def fake_upload_cell_adapter(
        out_dir_arg, cell_slug: str, hf_prefix: str = M.ADAPTERS_HF_PREFIX
    ) -> str:  # hub boundary (signature-mirrored, incl. the fu1 hf_prefix seam)
        assert hf_prefix == M.ADAPTERS_HF_PREFIX  # parent cells keep the parent prefix
        calls.append((Path(out_dir_arg), cell_slug))
        return f"https://hf/{cell_slug}"

    monkeypatch.setattr(M, "_hf_files_present", fake_hf_files_present)
    monkeypatch.setattr(M, "_upload_cell_adapter", fake_upload_cell_adapter)
    assert M.should_skip(cell, ckpt_root, dataset_root, directions_dir) is True
    assert calls == [(out_dir, cell.slug)], "presence short-circuit must not gate the re-drive"
    assert M._read_manifest(ckpt_root, cell)["uploaded"] is True


def test_should_skip_no_upload_mode_skips_without_redrive(tmp_path, monkeypatch):
    cell, ckpt_root, dataset_root, directions_dir = _resume_env(tmp_path)
    fp = M.cell_fingerprint(cell, dataset_root, directions_dir)
    M._write_manifest(ckpt_root, cell, fp)
    out_dir = _write_adapter_files(ckpt_root, cell)
    M.mark_manifest_completed(ckpt_root, cell, out_dir)

    def boom(*a, **k):  # any network/upload touch under --no-upload is a bug
        raise AssertionError("upload path touched under allow_upload=False")

    monkeypatch.setattr(M, "_hf_files_present", boom)
    monkeypatch.setattr(M, "_upload_cell_adapter", boom)
    assert M.should_skip(cell, ckpt_root, dataset_root, directions_dir, allow_upload=False) is True


def test_start_manifest_drops_prior_completed_fields(tmp_path):
    """The START write resets completed/uploaded (a retrain owns a fresh cycle)."""
    cell, ckpt_root, dataset_root, directions_dir = _resume_env(tmp_path)
    fp = M.cell_fingerprint(cell, dataset_root, directions_dir)
    M._write_manifest(ckpt_root, cell, fp)
    out_dir = _write_adapter_files(ckpt_root, cell)
    M.mark_manifest_completed(ckpt_root, cell, out_dir)
    M.mark_manifest_uploaded(ckpt_root, cell)
    M._write_manifest(ckpt_root, cell, fp)  # the retrain's START write
    stored = M._read_manifest(ckpt_root, cell)
    assert "completed" not in stored and "uploaded" not in stored


def test_fanout_skip_writes_skip_evidence_log(tmp_path, monkeypatch):
    """r2 blocker 4 (producer side): a resume-skipped cell appends a
    [fanout-skip] line to its per-cell log — the token the dispatcher's §7
    criterion-(i) dual-token count gate greps for."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cells = [M.cells_by_slug()["A__evil__c3.0"], M.cells_by_slug()["A__evil__c5.0"]]

    def fake_should_skip(cell, *a, **k) -> bool:
        return True

    monkeypatch.setattr(M, "should_skip", fake_should_skip)
    log_dir = tmp_path / "logs"
    res = M.run_fan_out(
        cells,
        dataset_root=tmp_path,
        ckpt_root=tmp_path,
        directions_dir=tmp_path,
        n_gpus=1,
        max_steps=None,
        cpu_only=True,
        dry_run=False,
        model_name="m",
        log_dir=log_dir,
    )
    assert all(v == "skipped-resume" for v in res["cells"].values())
    for cell in cells:
        text = (log_dir / f"{cell.slug}.log").read_text()
        assert "[fanout-skip]" in text and cell.slug in text


def test_fanout_no_upload_threads_to_skip_check_and_children(tmp_path, monkeypatch, capsys):
    """r2 g1 Concern 2: a --no-upload fan-out performs NO upload/network call in
    the parent's skip check, and children get --no-upload forwarded."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    cell, ckpt_root, dataset_root, directions_dir = _resume_env(tmp_path)
    fp = M.cell_fingerprint(cell, dataset_root, directions_dir)
    M._write_manifest(ckpt_root, cell, fp)
    out_dir = _write_adapter_files(ckpt_root, cell)
    M.mark_manifest_completed(ckpt_root, cell, out_dir)  # local-done, never uploaded

    def boom(*a, **k):  # any network/upload touch under --no-upload fan-out is a bug
        raise AssertionError("upload path touched by run_fan_out under allow_upload=False")

    monkeypatch.setattr(M, "_hf_files_present", boom)
    monkeypatch.setattr(M, "_upload_cell_adapter", boom)
    pending_cell = M.cells_by_slug()["A__evil__c5.0"]  # no manifest -> pending
    res = M.run_fan_out(
        [cell, pending_cell],
        dataset_root=dataset_root,
        ckpt_root=ckpt_root,
        directions_dir=directions_dir,
        n_gpus=1,
        max_steps=None,
        cpu_only=True,
        dry_run=True,
        model_name="m",
        log_dir=tmp_path / "logs",
        allow_upload=False,
    )
    assert res["cells"][cell.slug] == "skipped-resume"  # skipped with zero upload calls
    assert res["cells"][pending_cell.slug] == "dry-run"
    assert "--no-upload" in capsys.readouterr().out  # forwarded to the child argv


def test_visible_gpu_entries_uses_parent_cvd_entries(monkeypatch):
    """g2 Concern 3: a restricted/reordered parent CVD pins its g-th ENTRY."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,5,6,7")
    assert M._visible_gpu_entries(2) == ["4", "5"]
    assert M._visible_gpu_entries(4) == ["4", "5", "6", "7"]
    import pytest

    with pytest.raises(RuntimeError, match="exceeds"):
        M._visible_gpu_entries(5)
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES")
    assert M._visible_gpu_entries(3) == ["0", "1", "2"]


def test_main_bare_invocation_refuses(tmp_path, capsys):
    """g2 Concern 4: a bare call must never implicitly start the 81-cell fan-out."""
    import pytest

    with pytest.raises(SystemExit) as ei:
        M.main([])
    assert ei.value.code == 2
    assert "refusing the implicit full 81-cell fan-out" in capsys.readouterr().err


def test_main_smoke_defaults_max_steps(tmp_path, monkeypatch, capsys):
    """g2 suggestion 6: --smoke alone implies a tiny slice (--max-steps 4)."""
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    M.main(["--smoke", "--dry-run", "--ckpt-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "--max-steps 4" in out


def test_main_dry_run_default_width_clamps_to_parent_cvd(tmp_path, monkeypatch, capsys):
    """FAILS PRE-FIX (r2 g1 Concern 3): a --dry-run preview with no --n-gpus in
    a 1-GPU shell (CUDA_VISIBLE_DEVICES=0) must not raise the over-width error."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0")
    M.main(["--smoke", "--dry-run", "--ckpt-root", str(tmp_path)])
    out = capsys.readouterr().out
    assert "[fanout][dry-run] CUDA_VISIBLE_DEVICES=0 " in out
