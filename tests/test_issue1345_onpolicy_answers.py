"""Issue #1345 on-policy-vs-injected program — permanent invariants.

Five guard families, all zero-GPU and zero-API:

1. Launcher DEVICE RESOLUTION (#1902). Fellows SLURM nodes are GPU-SHARED and
   `nvidia-smi` ignores CUDA_VISIBLE_DEVICES, so a detected-count fan-out
   over-shards onto other tenants' GPUs. Width + physical ids must come from the
   ALLOCATION env, and a SLURM job exposing none of the allocation vars must FAIL
   LOUD rather than fall back to the physical count. Exercised against the SHIPPED
   script via its `EPM_I1345_RESOLVE_ONLY` affordance with a stubbed nvidia-smi.

2. vLLM `gpu_memory_utilization` computed from LIVE free memory (#1902 crash 1):
   a hardcoded fraction demands that share of TOTAL regardless of what other
   tenants hold, and EngineCore raises at init.

3. Provenance STORE KEYS, capture side: the `injected` default must reproduce
   every historical stem byte-for-byte (the live rounds' HF paths and fits
   registry entries must not move) while `onpolicy` is distinct, so an on-policy
   capture is co-resident with its injected twin instead of overwriting it. The
   axis is AUTHORSHIP — both provenances are captured teacher-forced.

3b/3c. The SECOND stem_for (the FITS registry in issue1345_common) plus the
   fits' own grid enumeration: same byte-identity-by-default contract, and the
   on-policy grid must be a MATCHED PAIR of the identical lattice (same slots, Y
   targets, conv_id space) gated on store presence — so the fits run unchanged
   before the on-policy captures land and pick the paired arm up automatically.

3d. Judge legs: rubric discipline (anchored, reason-then-score, confusable
   neighbours named per llm-judging rule 25) and the SPEND FAIL-SAFE — both the
   `--execute` flag AND the env ack are required, so no accidental invocation
   can bill the Batch API while spend is held.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"
LAUNCHER = SCRIPTS / "issue1345_onpolicy_answers_launch.sh"

for _p in (str(SCRIPTS), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

GIB = 2**30

# nvidia-smi stub: 8 physical devices; per-index free memory from $FAKE_FREE.
_STUB = """#!/usr/bin/env bash
args="$*"
if [[ "$args" == *"--query-gpu=index"* ]]; then
  for i in 0 1 2 3 4 5 6 7; do echo "$i"; done
  exit 0
fi
if [[ "$args" == *"--query-gpu=memory.free"* ]]; then
  idx=""; prev=""
  for a in "$@"; do
    if [ "$prev" = "-i" ]; then idx="$a"; fi
    prev="$a"
  done
  IFS=',' read -ra F <<< "${FAKE_FREE:-80000,80000,80000,80000,80000,80000,80000,80000}"
  echo "${F[$idx]:-0}"
  exit 0
fi
exit 1
"""


@pytest.fixture(scope="module")
def stub_dir(tmp_path_factory) -> Path:
    d = tmp_path_factory.mktemp("nvsmi_stub")
    smi = d / "nvidia-smi"
    smi.write_text(_STUB)
    smi.chmod(0o755)
    return d


@pytest.fixture(scope="module")
def staged_ok(tmp_path_factory) -> Path:
    """A staged-inputs dir the launcher pre-flight accepts (hermetic: never the
    repo's real data/ tree, so these tests do not depend on staged artifacts)."""
    d = tmp_path_factory.mktemp("staged_ok")
    (d / "matched_n").mkdir()
    (d / "matched_n" / "matched_subsets_parent.json").write_text("{}")
    return d


def _resolve(stub_dir: Path, staged: Path | None = None, **env_over) -> subprocess.CompletedProcess:
    """Run the SHIPPED launcher in pre-launch-check mode with a stubbed nvidia-smi."""
    env = {
        "PATH": f"{stub_dir}:{os.environ.get('PATH', '')}",
        "HOME": os.environ.get("HOME", "/tmp"),
        "EPM_I1345_RESOLVE_ONLY": "1",
        "REPO_ROOT": str(REPO_ROOT),
    }
    if staged is not None:
        env["EPM_I1345_STAGED_DIR"] = str(staged)
    env.update({k: v for k, v in env_over.items() if v is not None})
    return subprocess.run(
        ["bash", str(LAUNCHER)], capture_output=True, text=True, env=env, timeout=300
    )


# ---------------------------------------------------------------------------
# 1. Launcher device resolution
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    ("env_over", "want_source", "want_usable"),
    [
        # A SLURM allocation is authoritative; the physical node has 8 devices,
        # so any of these resolving to 8 would be the #1902 over-shard.
        (
            {"SLURM_JOB_ID": "1", "SLURM_GPUS_ON_NODE": "4", "CUDA_VISIBLE_DEVICES": "2,3,4,5"},
            "slurm-cvd",
            "2 3 4 5",
        ),
        (
            {"SLURM_JOB_ID": "1", "SLURM_GPUS_ON_NODE": "4", "SLURM_JOB_GPUS": "1,2,3,6"},
            "slurm-job-gpus",
            "1 2 3 6",
        ),
        (
            {"SLURM_JOB_ID": "1", "SLURM_GPUS_ON_NODE": "4", "SLURM_STEP_GPUS": "0,5"},
            "slurm-step-gpus",
            "0 5",
        ),
        (
            {"SLURM_JOB_ID": "1", "SLURM_GPUS_ON_NODE": "3"},
            "slurm-count-ids-assumed-0..N-1",
            "0 1 2",
        ),
        # Off-SLURM (exclusive host): enumeration is legitimate.
        ({}, "detected", "0 1 2 3 4 5 6 7"),
        ({"CUDA_VISIBLE_DEVICES": "3,4"}, "env-cvd", "3 4"),
    ],
)
def test_device_resolution_sources(stub_dir, staged_ok, env_over, want_source, want_usable):
    p = _resolve(stub_dir, staged_ok, **env_over)
    assert p.returncode == 0, p.stderr
    assert f"source={want_source}" in p.stdout, p.stdout
    assert f"usable={want_usable}" in p.stdout, p.stdout


def test_overlong_id_list_clamps_to_allocation_width(stub_dir, staged_ok):
    """An id list longer than SLURM_GPUS_ON_NODE is CLAMPED, not honored."""
    p = _resolve(
        stub_dir,
        staged_ok,
        SLURM_JOB_ID="1",
        SLURM_GPUS_ON_NODE="2",
        CUDA_VISIBLE_DEVICES="0,1,2,3,4,5,6,7",
    )
    assert p.returncode == 0, p.stderr
    assert "-clamped" in p.stdout
    assert "usable=0 1" in p.stdout
    assert "n=2" in p.stdout


def test_slurm_without_allocation_vars_fails_loud(stub_dir, staged_ok):
    """The #1902 core invariant: NEVER fall back to the physical count on SLURM."""
    p = _resolve(stub_dir, staged_ok, SLURM_JOB_ID="1")
    assert p.returncode == 3, (p.returncode, p.stdout, p.stderr)
    assert "refusing to fall back to the" in p.stderr
    assert "#1902" in p.stderr
    # The 8 physical devices must NOT appear as a resolution.
    assert "usable=0 1 2 3 4 5 6 7" not in p.stdout


def test_free_memory_filter_drops_tenant_held_devices(stub_dir, staged_ok):
    """Devices another tenant is holding are skipped before any cell launches."""
    p = _resolve(
        stub_dir,
        staged_ok,
        SLURM_JOB_ID="1",
        SLURM_GPUS_ON_NODE="4",
        CUDA_VISIBLE_DEVICES="0,1,2,3",
        FAKE_FREE="1000,80000,2000,80000",
    )
    assert p.returncode == 0, p.stderr
    assert "usable=1 3" in p.stdout
    assert "n=2" in p.stdout
    assert "skipping device 0" in p.stderr


def test_all_devices_held_refuses_to_launch(stub_dir):
    """Every allocated device held -> rc=3, never a launch onto a full GPU."""
    env = {
        "PATH": f"{stub_dir}:{os.environ.get('PATH', '')}",
        "HOME": os.environ.get("HOME", "/tmp"),
        "REPO_ROOT": str(REPO_ROOT),
        "SLURM_JOB_ID": "1",
        "SLURM_GPUS_ON_NODE": "2",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "FAKE_FREE": "500,500",
    }
    p = subprocess.run(
        ["bash", str(LAUNCHER)], capture_output=True, text=True, env=env, timeout=300
    )
    assert p.returncode == 3, (p.returncode, p.stdout, p.stderr)
    assert "no usable GPUs" in p.stderr


def test_preflight_refuses_a_staged_dir_without_the_allowlist(stub_dir, tmp_path):
    """A wrong --matched-dir must fail BEFORE any cell loads a 7B model.

    The comparator cells join the matched-n allowlist against the parent corpus,
    so an unstaged dir otherwise surfaces only after a provision is already spent.
    """
    empty = tmp_path / "no_allowlist"
    empty.mkdir()
    p = _resolve(stub_dir, empty, CUDA_VISIBLE_DEVICES="0,1")
    assert p.returncode == 3, (p.returncode, p.stdout, p.stderr)
    assert "matched-n allowlist missing" in p.stderr
    # It must name the remedy AND the candidates, not just fail.
    assert "issue1345_prefetch_reuse.py" in p.stderr
    assert "Candidates present on this host" in p.stderr


def test_preflight_skipped_for_a_story_slot_only_cell_list(stub_dir, tmp_path):
    """story_slot reads the sha-pinned V1 bundle — it needs no allowlist."""
    empty = tmp_path / "no_allowlist_slot"
    empty.mkdir()
    p = _resolve(
        stub_dir,
        empty,
        CUDA_VISIBLE_DEVICES="0,1",
        EPM_I1345_CELLS="onpolicy_answers_slot_base|story_slot|pretrained|Assistant",
    )
    assert p.returncode == 0, (p.returncode, p.stdout, p.stderr)
    assert "needs_matched=0" in p.stdout
    assert "matched-n allowlist missing" not in p.stderr


def test_preflight_passes_and_reports_on_a_good_staged_dir(stub_dir, staged_ok):
    p = _resolve(stub_dir, staged_ok, CUDA_VISIBLE_DEVICES="0,1")
    assert p.returncode == 0, p.stderr
    assert "staged inputs OK" in p.stdout
    assert "needs_matched=1" in p.stdout
    assert f"staged_dir={staged_ok}" in p.stdout
    # RESOLVE_ONLY is a FULL pre-launch check: it must never launch a cell.
    assert "starting onpolicy_answers" not in p.stdout


def test_launcher_default_staged_dir_is_the_round_variant():
    """A VM-only char_* dir is NOT on the fellows scratch — defaulting to one
    would fail on the production lane (team-lead, 2026-07-31)."""
    src = LAUNCHER.read_text()
    assert "EPM_I1345_STAGED_DIR:-data/issue_1345/story_boundary_ablation" in src
    assert "EPM_I1345_STAGED_DIR:-data/issue_1345/char_" not in src


def test_launcher_pins_cvd_per_cell():
    """Each cell must pin CUDA_VISIBLE_DEVICES in the LAUNCHER env (CVD family)."""
    src = LAUNCHER.read_text()
    assert 'CUDA_VISIBLE_DEVICES="$dev"' in src
    # ... and must never size width off the physical count inside a SLURM job.
    assert "SLURM_JOB_ID" in src and "SLURM_GPUS_ON_NODE" in src


# ---------------------------------------------------------------------------
# 2. Live-probed vLLM gpu_memory_utilization
# ---------------------------------------------------------------------------
def _op_module(monkeypatch):
    monkeypatch.setenv("EPM_I1345_VARIANT", "onpolicy_answers_ntpl_instruct")
    monkeypatch.setenv("EPM_STORY_CHARACTER_NAME", "ARIA")
    import issue1345_onpolicy_answers_gen as op

    return op


def test_vllm_util_empty_device_resolves_to_cap(monkeypatch):
    op = _op_module(monkeypatch)
    got = op.vllm_util_for_free(int(139.8 * GIB), int(139.8 * GIB))
    assert got == pytest.approx(op.VLLM_UTIL_CAP)


def test_vllm_util_shared_node_clamps_below_free(monkeypatch):
    """The #1902 crash shape: 81.2 GiB free of 139.8 GiB on a shared H200."""
    op = _op_module(monkeypatch)
    util = op.vllm_util_for_free(int(81.2 * GIB), int(139.8 * GIB))
    assert util < op.VLLM_UTIL_CAP
    # The demanded share must fit inside free minus the safety margin.
    assert util * 139.8 <= 81.2 - op.GPU_FREE_MARGIN_GIB + 1e-6
    # And the bare cap WOULD have over-demanded — this is the crash it prevents.
    assert op.VLLM_UTIL_CAP * 139.8 > 81.2


def test_vllm_util_below_floor_fails_loud(monkeypatch):
    op = _op_module(monkeypatch)
    with pytest.raises(RuntimeError, match="GPU too full"):
        op.vllm_util_for_free(int(20.0 * GIB), int(139.8 * GIB))


def test_vllm_util_rejects_nonsense_total(monkeypatch):
    op = _op_module(monkeypatch)
    with pytest.raises(RuntimeError):
        op.vllm_util_for_free(1, 0)


def test_engine_uses_the_resolver_not_a_literal(monkeypatch):
    """A hardcoded fraction at the LLM() call site is the #1902 regression."""
    import inspect

    op = _op_module(monkeypatch)
    src = inspect.getsource(op.main)
    assert "gpu_memory_utilization=resolve_vllm_util()" in src
    assert "gpu_memory_utilization=0.85" not in src


def test_spawn_pin_set_before_vllm_import(monkeypatch):
    """vLLM reads this at import time; fork() poisons EngineCore (#628)."""
    op = _op_module(monkeypatch)
    assert os.environ.get("VLLM_WORKER_MULTIPROC_METHOD") == "spawn"
    assert "VLLM_WORKER_MULTIPROC_METHOD" in Path(op.__file__).read_text()


# ---------------------------------------------------------------------------
# 3. Provenance store keys
# ---------------------------------------------------------------------------
# The historical (teacher-forced) store stems, which must never move: the three
# live rounds resume against these HF paths and the fits registry keys on them.
LEGACY_FORMAT_KEYS = {
    "v1_boundary_present": "bnd_v1",
    "v2_boundary_absent": "bnd_v2",
    "v3_label_stripped": "bnd_v3",
    "chat": "bnd_chat",
    "no_template": "bnd_ntpl",
}


def _cap_module(monkeypatch):
    monkeypatch.setenv("EPM_I1345_VARIANT", "story_boundary_ablation")
    monkeypatch.setenv("EPM_STORY_CHARACTER_NAME", "Assistant")
    import issue1345_boundary_ablation_capture as cap

    return cap


@pytest.mark.parametrize(("key", "legacy"), sorted(LEGACY_FORMAT_KEYS.items()))
def test_injected_format_keys_are_byte_identical(monkeypatch, key, legacy):
    cap = _cap_module(monkeypatch)
    assert cap.format_key(key) == legacy
    assert cap.format_key(key, cap.PROV_INJECTED) == legacy


@pytest.mark.parametrize(("key", "legacy"), sorted(LEGACY_FORMAT_KEYS.items()))
def test_on_policy_format_keys_are_distinct(monkeypatch, key, legacy):
    cap = _cap_module(monkeypatch)
    op_key = cap.format_key(key, cap.PROV_ONPOLICY)
    assert op_key == f"{legacy}_op"
    assert op_key != legacy


def test_no_stem_collides_across_key_and_provenance(monkeypatch):
    cap = _cap_module(monkeypatch)
    stems = [cap.stem_for(k, "instruct", pv) for k in LEGACY_FORMAT_KEYS for pv in cap.PROVENANCES]
    assert len(stems) == len(set(stems)), stems


def test_unknown_provenance_fails_loud(monkeypatch):
    """Never silently key an unknown provenance to the teacher-forced default."""
    cap = _cap_module(monkeypatch)
    with pytest.raises(AssertionError):
        cap.format_key("chat", "guessed")


def test_main_threads_provenance_at_every_key_site(monkeypatch):
    import inspect

    cap = _cap_module(monkeypatch)
    src = inspect.getsource(cap.main)
    for frag in (
        "stem_for(key, args.model, args.provenance)",
        "format_key(key, args.provenance)",
        "provenance=args.provenance",
    ):
        assert frag in src, frag
    # An un-threaded call would silently write the injected stem.
    assert "stem_for(key, args.model)" not in src


def test_capture_reuses_the_shared_prov_constants(monkeypatch):
    """ONE definition: a duplicate suffix map in the capture could drift."""
    cap = _cap_module(monkeypatch)
    import issue1345_common as common

    assert cap.PROV_INJECTED is common.PROV_INJECTED
    assert cap.PROV_ONPOLICY is common.PROV_ONPOLICY
    assert cap.PROVENANCES is common.PROVENANCES
    assert (common.PROV_INJECTED, common.PROV_ONPOLICY) == ("injected", "onpolicy")
    # The axis is AUTHORSHIP, not capture method: both provenances are captured
    # teacher-forced, so "teacher_forced" must not be a provenance value.
    assert "teacher_forced" not in common.PROVENANCES


# ---------------------------------------------------------------------------
# 3b. The FITS-side registry in issue1345_common (a SECOND stem_for)
# ---------------------------------------------------------------------------
# The historical fits stems/cell ids, which the live rounds' fit outputs key on.
LEGACY_FITS = {
    ("instruct", "r1"): ("instruct_chat_s", "R_instruct_r1_context"),
    ("instruct", "r2"): ("instruct_naturalistic_s", "R_instruct_r2_context"),
    ("instruct", "r3"): ("instruct_stories_s", "R_instruct_r3_context"),
    ("pretrained", "r1"): ("pretrained_chat_s", "R_base_r1_context"),
}


def _common(monkeypatch):
    monkeypatch.setenv("EPM_I1345_VARIANT", "")
    import issue1345_common as common

    return common


@pytest.mark.parametrize(("mk", "want"), sorted(LEGACY_FITS.items()))
def test_fits_stem_and_cell_id_byte_identical_by_default(monkeypatch, mk, want):
    """The injected default must not move ANY existing fits stem or cell id."""
    common = _common(monkeypatch)
    model, regime = mk
    want_stem, want_cell = want
    assert common.stem_for(model, regime) == want_stem
    assert common.stem_for(model, regime, common.PROV_INJECTED) == want_stem
    assert common.cell_id(model, regime, "context") == want_cell
    assert common.cell_id(model, regime, "context", common.PROV_INJECTED) == want_cell


@pytest.mark.parametrize(("mk", "want"), sorted(LEGACY_FITS.items()))
def test_fits_onpolicy_stem_and_cell_id_are_distinct(monkeypatch, mk, want):
    common = _common(monkeypatch)
    model, regime = mk
    want_stem, want_cell = want
    op_stem = common.stem_for(model, regime, common.PROV_ONPOLICY)
    op_cell = common.cell_id(model, regime, "context", common.PROV_ONPOLICY)
    assert op_stem != want_stem and op_cell != want_cell
    # The suffix rides the FORMAT token, so the track suffix stays terminal.
    assert op_stem.endswith(f"_{common.TRACK}")
    assert "_op_" in op_stem


def test_fits_prov_suffix_contract(monkeypatch):
    common = _common(monkeypatch)
    assert common.prov_suffix(common.PROV_INJECTED) == ""
    assert common.prov_suffix(common.PROV_ONPOLICY) == "_op"
    with pytest.raises(AssertionError):
        common.prov_suffix("guessed")


def test_fits_cell_dict_carries_provenance(monkeypatch):
    """Every cell dict must name its provenance so the fits can pair the arms."""
    common = _common(monkeypatch)
    inj = common._cell("instruct", "r1", "context")
    onp = common._cell("instruct", "r1", "context", common.PROV_ONPOLICY)
    assert inj["provenance"] == common.PROV_INJECTED
    assert onp["provenance"] == common.PROV_ONPOLICY
    assert inj["cell_id"] != onp["cell_id"]
    assert inj["format_key"] != onp["format_key"]
    # Everything else about the pair is IDENTICAL — that is what makes them a
    # matched pair of the same lattice rather than two unrelated cells.
    for k in ("model_key", "track", "slot_index", "target_turn_index", "regime", "arm"):
        assert inj[k] == onp[k], k


def test_all_cells_unchanged_by_the_new_dimension(monkeypatch):
    """all_cells() still emits ONLY injected cells — no silent lattice growth."""
    common = _common(monkeypatch)
    cells = common.all_cells()
    assert cells, "all_cells() is empty"
    assert all(cl["provenance"] == common.PROV_INJECTED for cl in cells)
    ids = [cl["cell_id"] for cl in cells]
    assert len(ids) == len(set(ids)), "duplicate cell ids"
    assert not any("_op_" in i for i in ids), "an on-policy cell leaked into all_cells()"


# ---------------------------------------------------------------------------
# 3b. On-policy store registry
# ---------------------------------------------------------------------------
def test_registry_covers_exactly_the_onpolicy_capable_keys(monkeypatch):
    """The ablation arms are injection-BY-CONSTRUCTION and must NOT be listed."""
    cap = _cap_module(monkeypatch)
    import issue1345_boundary_ablation_gen as bgen

    assert set(cap.ONPOLICY_STORES) == {"chat", "no_template", cap.V1_ARM}
    for arm in bgen.GEN_ARMS:
        assert arm not in cap.ONPOLICY_STORES, f"{arm} must have no on-policy twin"
        assert not cap.has_onpolicy_twin(arm)
    for key in cap.ONPOLICY_STORES:
        assert cap.has_onpolicy_twin(key)


def test_registry_spec_is_complete_and_fails_loud(monkeypatch):
    cap = _cap_module(monkeypatch)
    for key, spec in cap.ONPOLICY_STORES.items():
        for field in ("gen_shape", "source_flag", "capture_mode", "rows", "isolates"):
            assert spec.get(field), f"{key} missing {field}"
        # The declared source flag must match the capture mode it pairs with.
        if spec["capture_mode"].startswith("--comparator"):
            assert spec["source_flag"] == "--convs-jsonl", key
        else:
            assert spec["source_flag"] == "--stories-jsonl", key
    with pytest.raises(AssertionError, match="no registered on-policy twin"):
        cap.onpolicy_store_spec("v2_boundary_absent")


def test_registry_stems_are_all_op_suffixed_and_unique(monkeypatch):
    cap = _cap_module(monkeypatch)
    stems = cap.onpolicy_stems("instruct")
    assert set(stems) == set(cap.ONPOLICY_STORES)
    assert all("_op_" in s for s in stems.values()), stems
    assert len(set(stems.values())) == len(stems), stems
    # And each differs from its INJECTED twin's stem.
    for key, op_stem in stems.items():
        assert op_stem != cap.stem_for(key, "instruct"), key


def test_onpolicy_stems_keep_the_bnd_prefix_so_slot_COUNT_is_inferred(monkeypatch):
    """The fits size a bundle by prefix: 5 slots iff format_key startswith "bnd_".

    `expect = len(BND_SLOT_ORDER) if cell["format_key"].startswith("bnd_") else 2`
    — so an on-policy stem that LOST the prefix (e.g. if the prov suffix were
    moved to the FRONT: "op_bnd_chat") would be loaded as a 2-slot parent store
    and silently mis-read the whole X x Y grid. The suffix must stay a SUFFIX.
    """
    cap = _cap_module(monkeypatch)
    for key in cap.ONPOLICY_STORES:
        for prov in cap.PROVENANCES:
            fmt = cap.format_key(key, prov)
            assert fmt.startswith("bnd_"), (
                f"{key}/{prov} -> {fmt!r} lost the bnd_ prefix; the fits would size it "
                "as a 2-slot store and mis-read the 5-slot grid"
            )


# ---------------------------------------------------------------------------
# 3c. Fits provenance dimension — the paired arm of the same lattice
# ---------------------------------------------------------------------------
def _fits_module(monkeypatch):
    monkeypatch.setenv("EPM_I1345_VARIANT", "story_boundary_ablation")
    monkeypatch.setenv("EPM_STORY_CHARACTER_NAME", "Assistant")
    import issue1345_boundary_ablation_fits as fits

    return fits


def test_fits_injected_grid_is_byte_unchanged(monkeypatch):
    """Every pre-existing grid cell id must be untouched by the new dimension."""
    fits = _fits_module(monkeypatch)
    cells = fits.grid_cells("chat")
    ids = [cl["cell_id"] for cl in cells]
    assert all("_op_" not in i for i in ids), ids
    assert all(cl["provenance"] == "injected" for cl in cells)
    assert all(cl["format_key"] == "bnd_chat" for cl in cells)
    # The default and the explicit injected call agree exactly.
    assert ids == [cl["cell_id"] for cl in fits.grid_cells("chat", "injected")]


def test_fits_onpolicy_grid_is_a_matched_pair(monkeypatch):
    """Identical lattice shape; ONLY authorship (and hence the key) differs."""
    fits = _fits_module(monkeypatch)
    inj = fits.grid_cells("chat")
    onp = fits.grid_cells("chat", "onpolicy")
    assert len(inj) == len(onp) > 0
    for a, b in zip(inj, onp, strict=True):
        assert a["cell_id"] != b["cell_id"]
        assert a["format_key"] != b["format_key"]
        assert b["format_key"] == a["format_key"] + "_op"
        assert b["provenance"] == "onpolicy"
        # Everything that makes it the SAME lattice point is identical.
        for k in (
            "model_key",
            "track",
            "slot_index",
            "target_turn_index",
            "slot",
            "y_target",
            "arm",
        ):
            assert a[k] == b[k], k
    ids = [cl["cell_id"] for cl in inj + onp]
    assert len(ids) == len(set(ids)), "injected/on-policy cell ids collide"


def test_fits_paired_enumeration_is_presence_gated(monkeypatch, tmp_path):
    """No on-policy store on disk -> zero cells, and the run is unaffected."""
    fits = _fits_module(monkeypatch)
    keys = ["chat", "no_template", fits.bc.V1_ARM]
    cells, present = fits.onpolicy_paired_cells(tmp_path, keys)
    assert cells == []
    assert present == dict.fromkeys(keys, False)

    # Land ONE on-policy store (npz contract) -> only that twin joins.
    stem = f"{fits.bg.MODEL_KEY}_{fits.bc.format_key('chat', 'onpolicy')}_{fits.bc.TRACK}"
    (tmp_path / f"{stem}.npz").write_bytes(b"")
    cells, present = fits.onpolicy_paired_cells(tmp_path, keys)
    assert present["chat"] is True
    assert present["no_template"] is False
    assert cells and all(cl["provenance"] == "onpolicy" for cl in cells)
    assert {cl["bnd_arm"] for cl in cells} == {"chat"}
    assert len(cells) == len(fits.grid_cells("chat", "onpolicy"))


def test_fits_paired_enumeration_skips_the_ablation_arms(monkeypatch, tmp_path):
    """An ablation arm has no on-policy twin — skipped silently, never reported."""
    fits = _fits_module(monkeypatch)
    import issue1345_boundary_ablation_gen as bgen

    cells, present = fits.onpolicy_paired_cells(tmp_path, list(bgen.GEN_ARMS))
    assert cells == []
    assert present == {}, "an ablation arm must not appear in the presence report"


# ---------------------------------------------------------------------------
# 3d. Judge legs — protocol + spend fail-safe (zero API calls)
# ---------------------------------------------------------------------------
def _judge_module(monkeypatch):
    monkeypatch.setenv("EPM_I1345_VARIANT", "story_boundary_ablation")
    monkeypatch.setenv("EPM_STORY_CHARACTER_NAME", "Assistant")
    import issue1345_onpolicy_judge_legs as jl

    return jl


def test_judge_protocol_matches_the_llm_judging_rules(monkeypatch):
    jl = _judge_module(monkeypatch)
    import issue1345_common as common

    assert jl.JUDGE_MODEL == common.JUDGE_MODEL == "claude-sonnet-4-5-20250929"  # rule 11
    assert jl.N_DRAWS >= 5  # rule 4
    assert jl.JUDGE_TEMPERATURE > 0  # draws must actually vary
    assert jl.JUDGE_MAX_TOKENS >= 300  # rule 23 (reason-then-score floor)


@pytest.mark.parametrize("leg", ["ai_likeness", "content_drift"])
def test_rubrics_are_anchored_and_reason_then_score(monkeypatch, leg):
    jl = _judge_module(monkeypatch)
    r = jl.RUBRIC[leg]
    for anchor in ("  0 ", "  50 ", "  100 "):  # rule 6: endpoints + midpoint
        assert anchor in r, f"{leg} rubric lacks the {anchor.strip()} anchor"
    # The harness FORCES a JSON reply, so the score arrives as a JSON field, not
    # a bare `SCORE:` line (that format never parsed — see 3d-bis below).
    assert '"score": an integer from 0 to 100' in r, leg
    # rule 7: the reasoning must be requested BEFORE the score.
    assert r.index('"reasoning"') < r.index('"score"'), leg


def test_ai_likeness_rubric_names_its_confusable_neighbours(monkeypatch):
    """rule 25: an unnamed neighbour rides the contrast (the #1482 class).

    The four characters vary on politeness / formality / competence /
    theatricality — all correlated with naive AI-ness without being it.
    """
    jl = _judge_module(monkeypatch)
    r = jl.RUBRIC[jl.LEG_AI_LIKENESS].lower()
    for neighbour in ("politeness", "formality", "verbosity", "competence", "theatricality"):
        assert neighbour in r, f"AI-likeness rubric does not name the neighbour {neighbour!r}"
    assert "must not move the score" in r


def test_content_drift_rubric_is_substance_only(monkeypatch):
    jl = _judge_module(monkeypatch)
    r = jl.RUBRIC[jl.LEG_CONTENT_DRIFT].lower()
    for excluded in ("wording", "length", "tone", "formatting", "persona"):
        assert excluded in r, f"content-drift rubric does not exclude {excluded!r}"
    assert "contradict" in r


def test_spend_requires_BOTH_the_flag_and_the_env_ack(monkeypatch):
    """No accidental invocation can bill the Batch API."""
    jl = _judge_module(monkeypatch)
    monkeypatch.delenv(jl.SPEND_ACK_ENV, raising=False)
    assert jl.spend_allowed(False)[0] is False
    assert jl.spend_allowed(True)[0] is False, "the flag alone must NOT authorize spend"
    monkeypatch.setenv(jl.SPEND_ACK_ENV, "1")
    assert jl.spend_allowed(False)[0] is False, "the env alone must NOT authorize spend"
    assert jl.spend_allowed(True)[0] is True
    monkeypatch.setenv(jl.SPEND_ACK_ENV, "yes")
    assert jl.spend_allowed(True)[0] is False, "only the literal '1' authorizes spend"


def test_item_ids_are_batch_safe(monkeypatch):
    """charset ^[A-Za-z0-9_-]$ and <= 53 chars (the 11-char draw-suffix budget)."""
    jl = _judge_module(monkeypatch)
    import re

    for leg in jl.LEGS:
        iid = jl.item_id(leg, "helios", "s12345")
        assert re.fullmatch(r"[A-Za-z0-9_-]+", iid), iid
        assert len(iid) <= jl.ITEM_ID_MAX
    # Illegal characters are sanitized, never passed through.
    assert re.fullmatch(r"[A-Za-z0-9_-]+", jl.item_id("ai_likeness", "a.b:c/d", "s1"))
    # An over-long id fails loud rather than 400-ing the first batches.create.
    with pytest.raises(AssertionError, match="chars >"):
        jl.item_id("ai_likeness", "x" * 60, "s1")


def test_content_drift_pairs_on_conv_id_and_counts_unpaired(monkeypatch):
    jl = _judge_module(monkeypatch)
    rows = [{"conv_id": f"s{i}", "prompt": "Q?", "response": f"onpolicy {i}"} for i in range(3)]
    refs = [{"conv_id": "s0", "prompt": "Q?", "answer": "injected 0"}]
    items, counts = jl.build_content_drift_items(rows, refs, "vex")
    assert counts == {"paired": 1, "no_reference": 2}
    assert len(items) == 1
    # The reference must ride the user message so the rubric stays pointwise.
    _iid, _q, user = items[0]
    assert "REFERENCE ANSWER:" in user and "injected 0" in user
    assert "RESPONSE TO RATE:" in user and "onpolicy 0" in user


def test_answer_and_question_readers_accept_both_row_schemas(monkeypatch):
    jl = _judge_module(monkeypatch)
    assert jl._answer_of({"conv_id": "s1", "response": "a"}) == "a"  # comparator rows
    assert jl._answer_of({"conv_id": "s1", "answer": "b"}) == "b"  # kept-stories rows
    assert jl._question_of({"conv_id": "s1", "prompt": "q"}) == "q"
    assert jl._question_of({"conv_id": "s1", "question": "q2"}) == "q2"
    with pytest.raises(AssertionError, match="neither"):
        jl._answer_of({"conv_id": "s1"})


# ---------------------------------------------------------------------------
# 3d-bis. Rubric <-> HARNESS CONTRACT (the bug the #1916 flag surfaced)
# ---------------------------------------------------------------------------
# graded_judge passes the rubric as a USER TEMPLATE and substitutes {question} /
# {answer} into it, while appending a `{"score": ...}` JSON wrapper to the judge
# SYSTEM prompt. A rubric missing the placeholders sends the judge NO content to
# rate; a rubric asking for a bare `SCORE: <int>` line never parses. Both would
# drop ~100% of draws while looking perfectly reasonable in review.
@pytest.mark.parametrize("leg", ["ai_likeness", "content_drift"])
def test_rubric_carries_the_substitution_placeholders(monkeypatch, leg):
    jl = _judge_module(monkeypatch)
    r = jl.RUBRIC[leg]
    assert "{question}" in r, f"{leg}: judge would receive no question"
    assert "{answer}" in r, f"{leg}: judge would receive no answer to rate"


@pytest.mark.parametrize("leg", ["ai_likeness", "content_drift"])
def test_rubric_substitutes_exactly_as_the_harness_does(monkeypatch, leg):
    """Mirror graded_judge's own format_user_msg substitution."""
    jl = _judge_module(monkeypatch)
    filled = jl.RUBRIC[leg].replace("{question}", "Q-SENTINEL").replace("{answer}", "A-SENTINEL")
    assert "Q-SENTINEL" in filled and "A-SENTINEL" in filled
    # No unsubstituted slot may survive into the judge prompt.
    assert "{question}" not in filled and "{answer}" not in filled


@pytest.mark.parametrize("leg", ["ai_likeness", "content_drift"])
def test_rubric_requests_json_with_reasoning_before_score(monkeypatch, leg):
    """The harness forces JSON; rule 7 still wants the reasoning generated first."""
    jl = _judge_module(monkeypatch)
    r = jl.RUBRIC[leg]
    assert "single JSON object" in r, leg
    assert '"reasoning"' in r and '"score"' in r, leg
    assert r.index('"reasoning"') < r.index('"score"'), f"{leg}: score must come last"
    # The retired bare-line format must not linger anywhere.
    assert "SCORE: <integer" not in r, leg


@pytest.mark.parametrize("leg", ["ai_likeness", "content_drift"])
def test_expected_reply_shape_parses_to_a_score(monkeypatch, leg):
    """Round-trip a REALISTIC reply through the harness's OWN parse + reduce."""
    _jl = _judge_module(monkeypatch)
    from explore_persona_space.eval.graded_judge import _score_from_parsed
    from explore_persona_space.eval.utils import parse_judge_json

    reply = (
        '{"reasoning": "It hedges and enumerates, which I set aside; the '
        'giveaway is the uniform clause rhythm.", "score": 73}'
    )
    parsed = parse_judge_json(reply)
    assert parsed is not None, "the requested reply shape does not parse"
    assert _score_from_parsed(parsed) == 73, parsed


def test_parse_is_fence_tolerant(monkeypatch):
    """#1934: ~2% of judged calls drop on markdown fences, not truncation.

    parse_judge_json falls back to first-`{` raw_decode, so a fenced reply still
    parses — recorded here so the per-arm drop report is not misread as content
    drops. The hazard this pins: prose BEFORE the JSON containing a stray `{`
    would anchor the fallback on the wrong brace, which is why both rubrics put
    the reasoning INSIDE the object rather than ahead of it.
    """
    _jl = _judge_module(monkeypatch)
    from explore_persona_space.eval.graded_judge import _score_from_parsed
    from explore_persona_space.eval.utils import parse_judge_json

    fenced = '```json\n{"reasoning": "clear enough", "score": 41}\n```'
    assert _score_from_parsed(parse_judge_json(fenced)) == 41
    preamble = 'Here is my assessment:\n{"reasoning": "ok", "score": 5}'
    assert _score_from_parsed(parse_judge_json(preamble)) == 5
    # The instructed-refusal path stays a DROP, never a coerced number (rule 9).
    assert _score_from_parsed(parse_judge_json('{"score": "REFUSAL"}')) is None


def test_max_tokens_meets_the_json_rubric_floor(monkeypatch):
    """#1916 raises the floor to 600 for JSON-shaped rubrics; ours carry a
    reasoning field ahead of the score, so the raised floor binds."""
    jl = _judge_module(monkeypatch)
    assert jl.JUDGE_MAX_TOKENS >= 600
