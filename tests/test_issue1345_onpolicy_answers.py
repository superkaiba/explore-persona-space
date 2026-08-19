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


# The pure-math vllm_util_for_free / resolve_vllm_util tests moved to
# tests/test_vllm_util.py (#1942: the local copy was hoisted to the shared
# explore_persona_space.eval.vllm_util module).


def test_engine_uses_the_resolver_not_a_literal(monkeypatch):
    """A hardcoded fraction at the LLM() call site is the #1902 regression."""
    import inspect

    op = _op_module(monkeypatch)
    src = inspect.getsource(op.main)
    assert "gpu_memory_utilization=resolve_vllm_util(cap=EXCLUSIVE_HOST_UTIL_CAP)" in src
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


def _gen_module(monkeypatch):
    monkeypatch.setenv("EPM_I1345_VARIANT", "story_boundary_ablation")
    monkeypatch.setenv("EPM_STORY_CHARACTER_NAME", "Assistant")
    import issue1345_onpolicy_answers_gen as gen

    return gen


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
    import issue1345_common as common

    keys = ["chat", "no_template", fits.bc.V1_ARM]
    cells, present = fits.onpolicy_paired_cells(tmp_path, keys)
    assert cells == []
    # Presence is per (store x MEASURED model): 3 keys x 2 models.
    assert present == {f"{k}/{m}": False for k in keys for m in common.MODELS}

    # Land ONE on-policy store for ONE model -> only that twin joins.
    stem = f"instruct_{fits.bc.format_key('chat', 'onpolicy')}_{fits.bc.TRACK}"
    (tmp_path / f"{stem}.npz").write_bytes(b"")
    cells, present = fits.onpolicy_paired_cells(tmp_path, keys)
    assert present["chat/instruct"] is True
    assert present["chat/pretrained"] is False, "an uncaptured model must NOT join"
    assert present["no_template/instruct"] is False
    assert cells and all(cl["provenance"] == "onpolicy" for cl in cells)
    assert {cl["bnd_arm"] for cl in cells} == {"chat"}
    assert {cl["model_key"] for cl in cells} == {"instruct"}
    assert len(cells) == len(fits.grid_cells("chat", "onpolicy", "instruct"))

    # Land the PRETRAINED twin too -> both models join, no collision.
    stem_b = f"pretrained_{fits.bc.format_key('chat', 'onpolicy')}_{fits.bc.TRACK}"
    (tmp_path / f"{stem_b}.npz").write_bytes(b"")
    cells, present = fits.onpolicy_paired_cells(tmp_path, keys)
    assert present["chat/pretrained"] is True
    assert {cl["model_key"] for cl in cells} == {"instruct", "pretrained"}
    ids = [cl["cell_id"] for cl in cells]
    assert len(ids) == len(set(ids)), "cross-model cell ids collide"


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
    assert common.JUDGE_MAX_TOKENS >= 1024  # rule 23 current floor (#2063 raise)
    # #2063 anchor-parity freeze: the boundary-ablation KEEP gate stays pinned to
    # the V1 anchor's 400-token instrument even after common's floor raise.
    import issue1345_boundary_ablation_gen as bgen

    assert bgen.BND_JUDGE_MAX_TOKENS == 400


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


# ---------------------------------------------------------------------------
# THE PING round: pretrained capture + Y_boundary cap split
# ---------------------------------------------------------------------------
def test_capture_accepts_both_measured_models(monkeypatch):
    """3 of the 4 on-policy bundles are PRETRAINED-written; an instruct-only
    --model would block their capture entirely."""
    cap = _cap_module(monkeypatch)
    import inspect

    import issue1345_common as common

    src = inspect.getsource(cap.main)
    assert "choices=c.MODELS" in src, "--model must accept both measured models"
    assert 'choices=("instruct",)' not in src
    assert set(common.MODELS) == {"instruct", "pretrained"}


def test_persist_store_uses_the_CAPTURED_model_not_the_round_default(monkeypatch):
    """A hardcoded round default globs the instruct stem: a pretrained capture
    would fail to find its own shards (or mislabel the manifest)."""
    cap = _cap_module(monkeypatch)
    import inspect

    src = inspect.getsource(cap.persist_store)
    assert "stem_for(key, model_key, provenance)" in src
    assert "stem_for(key, bg.MODEL_KEY, provenance)" not in src
    assert '"model": model_key' in src
    # And the caller threads the parsed arg.
    assert "model_key=args.model" in inspect.getsource(cap.main)
    sig = inspect.signature(cap.persist_store)
    assert "model_key" in sig.parameters


def test_pretrained_stems_and_cell_ids_never_collide_with_instruct(monkeypatch):
    cap = _cap_module(monkeypatch)
    fits = _fits_module(monkeypatch)
    import issue1345_common as common

    stems = [
        cap.stem_for(k, m, pv)
        for k in cap.ONPOLICY_STORES
        for m in common.MODELS
        for pv in common.PROVENANCES
    ]
    assert len(stems) == len(set(stems)), stems
    ids = [
        cl["cell_id"]
        for k in cap.ONPOLICY_STORES
        for m in common.MODELS
        for pv in common.PROVENANCES
        for cl in fits.grid_cells(k, pv, m)
    ]
    assert len(ids) == len(set(ids)), "cross-(model x prov) grid cell ids collide"


def test_injected_grid_cell_ids_unmoved_by_the_model_dimension(monkeypatch):
    """The round-default call must stay byte-identical (live fit outputs key on it)."""
    fits = _fits_module(monkeypatch)
    ids = [cl["cell_id"] for cl in fits.grid_cells("chat")]
    assert ids[0] == "R_instruct_bnd_chat_prefix__ymean", ids[0]
    assert ids == [cl["cell_id"] for cl in fits.grid_cells("chat", "injected", "instruct")]


def test_kept_rows_carry_finish_reason_for_the_Y_boundary_split(monkeypatch):
    """A cap-truncated answer ends MID-SENTENCE, so the boundary target read just
    after it is an artifact of the cap — the fits must be able to split on it."""
    op = _op_module(monkeypatch)
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.issue_825.common import MODEL_INSTRUCT

    tok = AutoTokenizer.from_pretrained(MODEL_INSTRUCT)
    long_a = "Paris has been the administrative centre since the Capetian consolidation."
    raws = [
        {"conv_id": "s0", "answer_text": long_a, "finish_reason": "stop"},
        {"conv_id": "s1", "answer_text": long_a, "finish_reason": "length"},
    ]
    pool = [{"conv_id": "s0", "question": "Q?"}, {"conv_id": "s1", "question": "Q?"}]
    kept, counts = op.keep_rows(raws, pool, tok, shape=op.SHAPE_BARE, model_key="instruct")
    assert len(kept) == 2
    by_id = {r["conv_id"]: r for r in kept}
    assert by_id["s0"]["finish_reason"] == "stop" and by_id["s0"]["capped"] is False
    assert by_id["s1"]["finish_reason"] == "length" and by_id["s1"]["capped"] is True
    # Capped rows are KEPT (Y_mean is unaffected) and still counted.
    assert counts["finish_length_capped"] == 1


def test_capture_preserves_the_cap_split_into_the_store_meta(monkeypatch, tmp_path):
    """to_single_turn keeps only {conv_id,u1,a1}; the split must survive it."""
    cap = _cap_module(monkeypatch)
    import issue1345_common as common
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.issue_825.common import MODEL_INSTRUCT

    rows_f = tmp_path / "op_rows.jsonl"
    common.append_jsonl(
        rows_f,
        [
            {
                "conv_id": "s0",
                "prompt": "What is the capital of France and why does it matter?",
                "response": "Paris has been the administrative centre for centuries.",
                "finish_reason": "length",
                "capped": True,
                "provenance": "onpolicy",
            }
        ],
    )
    convs = cap.load_comparator_convs(tmp_path, None, convs_jsonl=rows_f)
    assert convs[0]["capped"] is True, "the cap flag was stripped by to_single_turn"
    assert convs[0]["finish_reason"] == "length"
    tok = AutoTokenizer.from_pretrained(MODEL_INSTRUCT)
    r = cap.render_comparator_turn(convs[0], tok, comparator="no_template")
    assert r is not None
    assert r.meta.get("capped") is True, "the split did not reach the store meta"
    assert r.meta.get("finish_reason") == "length"
    # Injected rows have no generation, so the keys are simply absent there.
    inj = {"conv_id": "s1", "u1": "Q?", "a1": "A long enough injected answer here."}
    ri = cap.render_comparator_turn(inj, tok, comparator="no_template")
    assert ri is not None and "capped" not in ri.meta


def test_store_token_tiers_and_prefix_safety(monkeypatch):
    """Two tiers + a prefix-safe family match.

    `bnd_v1` is a PREFIX of `bnd_v1_op`, so the previous substring probe
    (`token in filename`) would have been satisfied by an on-policy file alone —
    once the `_op` stores land, a MISSING injected v1 store would have passed the
    required-family check. The `{token}_s` terminator closes that.
    """
    monkeypatch.setenv("EPM_I1345_VARIANT", "story_boundary_ablation")
    monkeypatch.setenv("EPM_STORY_CHARACTER_NAME", "Assistant")
    import issue1345_boundary_ablation_stage_and_mirror as sm

    # v5 joined the injected tier; the paired tier is the on-policy arm.
    assert "bnd_v5" in sm.REQUIRED_STORE_TOKENS
    assert set(sm.PAIRED_STORE_TOKENS) == {"bnd_v1_op", "bnd_chat_op", "bnd_ntpl_op"}
    assert not set(sm.REQUIRED_STORE_TOKENS) & set(sm.PAIRED_STORE_TOKENS)
    # Paired absence must be non-fatal by DEFAULT (presence-gated fits) but
    # promotable to fatal once the captures land.
    assert sm.REQUIRE_PAIRED_ENV == "EPM_I1345_REQUIRE_PAIRED"
    src = __import__("inspect").getsource(sm.cmd_stage)
    assert 'f"{token}_s" in n' in src, "family match must use the stem terminator"
    # cmd_stage references the CONSTANT by name, not its literal value.
    assert "REQUIRE_PAIRED_ENV" in src

    # The prefix hazard itself, exercised on the shipped predicate's shape.
    present = ["instruct_bnd_v1_op_s_shard000.pt"]

    def family(token: str) -> bool:
        return any(f"{token}_s" in n for n in present)

    assert family("bnd_v1_op") is True
    assert family("bnd_v1") is False, "an on-policy file must not satisfy the injected family"
    # ... whereas the retired substring probe WOULD have been satisfied:
    assert any("bnd_v1" in n for n in present) is True


# ---------------------------------------------------------------------------
# 3e. Judge SAMPLING design (the authorized n=300 stratified draw)
# ---------------------------------------------------------------------------
def _cell_rows(n: int = 900, cap_rate: float = 0.54, shuffle_seed: int = 7) -> list[dict]:
    """A synthetic cell whose cap rate matches the measured ntpl shape."""
    import random as _r

    cut = round(cap_rate * 100)
    rows = [
        {
            "conv_id": f"s{i:05d}",
            "prompt": f"Question {i}?",
            "response": f"answer {i}",
            "capped": (i % 100) < cut,
            "finish_reason": "length" if (i % 100) < cut else "stop",
        }
        for i in range(n)
    ]
    _r.Random(shuffle_seed).shuffle(rows)  # input file order must not matter
    return rows


def test_sample_is_seeded_and_independent_of_row_file_order(monkeypatch):
    jl = _judge_module(monkeypatch)
    rows = _cell_rows()
    a, da = jl.stratified_sample(rows, 300, 1345, "helios")
    b, _ = jl.stratified_sample(list(reversed(rows)), 300, 1345, "helios")
    assert [r["conv_id"] for r in a] == [r["conv_id"] for r in b]
    # A different seed must actually move the draw (else the seed is decorative).
    d, _ = jl.stratified_sample(rows, 300, 999, "helios")
    assert {r["conv_id"] for r in d} != {r["conv_id"] for r in a}
    assert da["realized_n"] == 300 and len(set(da["conv_ids"])) == 300


def test_sample_preserves_the_cells_capped_rate(monkeypatch):
    """The load-bearing property: a capped answer stops mid-sentence, so its
    Y_boundary target is artificial — the draw must not shift the cap mix."""
    jl = _judge_module(monkeypatch)
    for rate in (0.05, 0.54, 0.9):
        rows = _cell_rows(cap_rate=rate)
        _s, d = jl.stratified_sample(rows, 300, 1345, "helios")
        assert d["realized_n"] == 300
        assert abs(d["realized_capped_rate"] - d["eligible_capped_rate"]) < 0.01, d


def test_sample_takes_all_rows_of_a_small_cell_and_records_realized_n(monkeypatch):
    """A yield-floor-halted cell is smaller than the target; it is never padded
    and the report must read the realized n, not the target."""
    jl = _judge_module(monkeypatch)
    rows = _cell_rows(n=180)
    s, d = jl.stratified_sample(rows, 300, 1345, "wren")
    assert d["take_all"] is True and d["realized_n"] == 180 == len(s)
    assert d["n_target"] == 300, "the target stays on the record next to the realized n"


def test_degenerate_strata_do_not_break_the_draw(monkeypatch):
    """All-capped / no-capped cells exercise the stratum-clamp + top-up branches
    (the gate branches the main draw deliberately never reaches)."""
    jl = _judge_module(monkeypatch)
    allcap = [dict(r, capped=True, finish_reason="length") for r in _cell_rows(n=700)]
    _s, d = jl.stratified_sample(allcap, 300, 1345, "x")
    assert d["strata_targets"] == {"capped": 300, "natural": 0} and d["realized_capped"] == 300
    nocap = [dict(r, capped=False, finish_reason="stop") for r in _cell_rows(n=700)]
    _s, d2 = jl.stratified_sample(nocap, 300, 1345, "x")
    assert d2["strata_targets"] == {"capped": 0, "natural": 300} and d2["realized_capped"] == 0


def test_both_legs_draw_together_and_the_filtered_leg_nests(monkeypatch):
    """Seed material is (seed, tag) and NOT the leg, so ai_likeness and
    content_drift judge the SAME conv_ids where eligibility is universal, and
    overlapping ones where the drift leg filters to rows with an injected twin."""
    jl = _judge_module(monkeypatch)
    rows = _cell_rows()
    a, _ = jl.stratified_sample(rows, 300, 1345, "vex")  # ai_likeness frame
    b, _ = jl.stratified_sample(rows, 300, 1345, "vex")  # drift, same frame
    assert [r["conv_id"] for r in a] == [r["conv_id"] for r in b]
    twinned = {r["conv_id"] for r in rows[:600]}
    _f, df = jl.stratified_sample(
        rows, 300, 1345, "vex", eligible=lambda r: r["conv_id"] in twinned
    )
    assert df["realized_n"] == 300 and set(df["conv_ids"]) <= twinned
    assert set(df["conv_ids"]) & {r["conv_id"] for r in a}, "the legs would be unpaired draws"


def test_capped_of_falls_back_to_finish_reason(monkeypatch):
    jl = _judge_module(monkeypatch)
    assert jl.capped_of({"capped": True}) is True
    assert jl.capped_of({"capped": False, "finish_reason": "length"}) is False  # flag wins
    # A row file written before the flag landed still strata correctly.
    assert jl.capped_of({"finish_reason": "length"}) is True
    assert jl.capped_of({"finish_reason": "stop"}) is False
    assert jl.capped_of({}) is False


def test_sub_means_split_capped_from_natural_and_exclude_unscored(monkeypatch):
    jl = _judge_module(monkeypatch)
    rows = [{"conv_id": f"s{i}", "capped": i < 4} for i in range(10)]
    cmap = jl.capped_by_item(jl.LEG_AI_LIKENESS, "helios", rows)
    assert len(cmap) == 10 and sum(cmap.values()) == 4
    scores = {iid: (80.0 if cap else 20.0) for iid, cap in cmap.items()}
    dropped = next(iid for iid, cap in cmap.items() if cap)
    scores[dropped] = None  # every draw dropped -> excluded, never coerced
    m = jl.sub_means(scores, cmap)
    assert m["capped"] == {"n": 3, "mean": 80.0}
    assert m["natural"] == {"n": 6, "mean": 20.0}
    assert m["pooled"]["n"] == 9 and m["n_unscored_items"] == 1


def test_selection_caveat_travels_with_the_halted_cells(monkeypatch):
    """The 5 rc=21 cells' kept rows are a SELECTED subset; the caveat must ride
    the report, not a separate footnote. Dana + Wren are labelling characters."""
    jl = _judge_module(monkeypatch)
    assert set(jl.YIELD_FLOOR_HALTED_CELLS) == {
        "helios_base",
        "wren",
        "wren_base",
        "dana",
        "vex_base",
    }
    for tag in jl.YIELD_FLOOR_HALTED_CELLS:
        _s, d = jl.stratified_sample(_cell_rows(n=120), 300, 1345, tag)
        assert d["yield_floor_halted_cell"] is True
    _s, d = jl.stratified_sample(_cell_rows(n=120), 300, 1345, "helios")
    assert d["yield_floor_halted_cell"] is False


def test_batch_routing_is_forced_for_the_authorized_run(monkeypatch):
    """A cell-leg is 300 items x 5 draws = 1,500 calls, UNDER the client's
    default sync-vs-batch crossover (base=2000) — so the default would dispatch
    every cell SYNC. 0 forces the authorized Batch path."""
    jl = _judge_module(monkeypatch)
    import inspect

    assert jl.THRESHOLD_BASE_FORCE_BATCH == 0
    assert 300 * jl.N_DRAWS < 2000, "the sizing premise behind forcing the path"
    sig = inspect.signature(jl.run_leg)
    assert sig.parameters["threshold_base"].default == jl.THRESHOLD_BASE_FORCE_BATCH
    src = inspect.getsource(jl.run_leg)
    assert "threshold_base=threshold_base" in src, "must reach judge_graded"
    # judge_graded really accepts it (an ABI drift would silently re-sync).
    from explore_persona_space.eval.graded_judge import judge_graded

    assert "threshold_base" in inspect.signature(judge_graded).parameters


def test_sample_design_is_persisted_before_any_dispatch(monkeypatch):
    """A mid-batch failure must lose no design: the draw is written BEFORE
    judge_graded is called, not after it returns."""
    jl = _judge_module(monkeypatch)
    import inspect

    src = inspect.getsource(jl.run_leg)
    i_write = src.index("judge_sample_")
    i_judge = src.index("judge_graded(")
    assert i_write < i_judge, "the design manifest must be written before dispatch"


def test_content_drop_classes_reports_absent_save_raw(tmp_path, monkeypatch):
    """Dry-run makes no calls, so there is no save_raw — the classifier must say
    so rather than reporting a zeroed-out clean bill of health."""
    jl = _judge_module(monkeypatch)
    out = jl.content_drop_classes(tmp_path / "nope.json", [("ail_x_s1", "q", "a")])
    assert out["available"] is False and "absent" in out["reason"]


def test_content_drop_classes_splits_refusal_from_parse_error(tmp_path, monkeypatch):
    """The three-way split's content side: a REFUSAL residue is expected, a
    parse/malformed residue is the rule-23 truncation signature (#1934's fence
    class would land in parse_error)."""
    jl = _judge_module(monkeypatch)
    import json as _j

    items = [(f"ail_x_s{i}", "q", "a") for i in range(4)]
    raw = {
        "all_scores": {
            "ail_x_s0__00000__00": {"score": 71},  # scored
            "ail_x_s1__00001__00": {"score": "REFUSAL"},
            "ail_x_s2__00002__00": {
                "aligned": None,
                "coherent": None,
                "reasoning": "parse_error",
                "error": True,
            },
            "ail_x_s3__00003__00": {"score": 999},  # out of range -> malformed
        }
    }
    p = tmp_path / "judge_raw.json"
    p.write_text(_j.dumps(raw))
    out = jl.content_drop_classes(p, items)
    assert out["available"] is True
    assert out["refusal"] == 1 and out["parse_error"] == 1 and out["other_malformed"] == 1


# ---------------------------------------------------------------------------
# 3f. Judge-row PREP — the recovered stratifier + the span-answer extraction
# ---------------------------------------------------------------------------
def _prep_module(monkeypatch):
    monkeypatch.setenv("EPM_I1345_VARIANT", "story_boundary_ablation")
    monkeypatch.setenv("EPM_STORY_CHARACTER_NAME", "Assistant")
    import issue1345_judge_rows_prep as prep

    return prep


def _story_row(conv_id: str, answer: str = "X is a thing that does stuff, at length.") -> dict:
    story = f"Human: What is X?\n\nAssistant: {answer}\n"
    q0, q1 = story.index("What"), story.index("?") + 1
    a0 = story.index(answer)
    return {
        "conv_id": conv_id,
        "question": story[q0:q1],
        "story": story,
        "parsed_turns": [{"q_start": q0, "q_end": q1, "a_start": a0, "a_end": a0 + len(answer)}],
    }


def test_prep_recovers_capped_from_the_raw_finish_reason_join(monkeypatch):
    """The uploaded on-policy row files carry NO `capped` — without the join the
    stratifier reads all-natural and the draw silently stops being stratified."""
    prep = _prep_module(monkeypatch)
    rows = [
        {"conv_id": f"s{i}", "prompt": "Q?", "response": "a long enough answer"} for i in range(4)
    ]
    assert not any("capped" in r or "finish_reason" in r for r in rows), (
        "fixture must mirror the gap"
    )
    idx = {"s0": True, "s1": True, "s2": False, "s3": False}
    out, stats = prep.prepare(rows, idx, cell="op_x")
    assert stats["capped_source"] == "raw_finish_reason_join"
    assert stats["n_capped"] == 2 and stats["capped_rate"] == 0.5
    assert [r["capped"] for r in out] == [True, True, False, False]


def test_prep_fails_loud_on_a_partial_raw_join(monkeypatch):
    """The raw pool is a SUPERSET of kept rows, so a miss means the wrong raw
    file — cap stratification built on a partial join is silently wrong."""
    prep = _prep_module(monkeypatch)
    rows = [
        {"conv_id": f"s{i}", "prompt": "Q?", "response": "a long enough answer"} for i in range(3)
    ]
    with pytest.raises(AssertionError, match="no raw finish_reason"):
        prep.prepare(rows, {"s0": True}, cell="op_x")


def test_prep_extracts_char_cell_answers_from_the_story_span(monkeypatch):
    """Character on-policy rows carry no `answer` field at all (their injected
    siblings do) — the answer is story[a_start:a_end]."""
    prep = _prep_module(monkeypatch)
    row = _story_row("s1")
    assert "answer" not in row and "response" not in row
    text, src = prep.answer_of(row)
    assert src == "parsed_turns_span" and text.startswith("X is a thing")
    assert prep.question_of(row) == "What is X?"


def test_prep_rejects_a_span_that_no_longer_indexes_the_answer(monkeypatch):
    """A row with BOTH an answer field and a span is cross-checked — the #825
    mis-sliced-span class must not reach the judge as rated text."""
    prep = _prep_module(monkeypatch)
    row = dict(_story_row("s1"), answer="something else entirely, not the span")
    with pytest.raises(AssertionError, match="does not index the answer slot"):
        prep.answer_of(row)


def test_prep_drops_degenerate_spans_but_never_length_filters(monkeypatch):
    """A 0-4-char span is a stray character, not an answer: dropped AND counted.
    The floor stays minimal because answer LENGTH correlates with the very
    AI-likeness the rubric isolates — a length filter would bias that axis."""
    prep = _prep_module(monkeypatch)
    assert prep.ANSWER_CHAR_FLOOR <= 8, "a higher floor would length-bias the AI-likeness read"
    good = [_story_row(f"g{i}") for i in range(60)]
    tiny = _story_row("t0", answer="X")
    out, stats = prep.prepare([*good, tiny], None, cell="char_x_op")
    assert stats["n_short_answer"] == 1 and len(out) == 60
    assert all(r["conv_id"] != "t0" for r in out)
    # p1 of the real character cells is 22-30 chars: a 22-char answer is KEPT.
    kept, _ = prep.prepare([_story_row("k0", answer="A" * 22)], None, cell="char_x_op")
    assert len(kept) == 1


def test_prep_fails_loud_when_the_short_tail_is_systemic(monkeypatch):
    """0.1-0.6% sub-floor is the measured data tail; a large share is the
    extraction breaking, which must not pass as a tail."""
    prep = _prep_module(monkeypatch)
    with pytest.raises(AssertionError, match="extraction break"):
        prep.prepare([_story_row(f"t{i}", answer="X") for i in range(10)], None, cell="char_x_op")


def test_short_field_answers_are_data_not_an_extraction_break(monkeypatch):
    """The real corpus reference has 103/5000 (2.1%) answers under the floor —
    genuine one-word replies. Only a SPAN can be mis-sliced, so the
    systemic-break ceiling keys on span-derived answers; a short FIELD answer is
    dropped and counted without tripping it."""
    prep = _prep_module(monkeypatch)
    rows = [
        {"conv_id": f"s{i}", "prompt": "Q?", "response": "a long enough answer"} for i in range(90)
    ]
    rows += [{"conv_id": f"t{i}", "prompt": "Q?", "response": "ok"} for i in range(10)]  # 10%
    out, stats = prep.prepare(rows, None, cell="track_s_injected")
    assert len(out) == 90 and stats["n_short_answer"] == 10
    assert stats["n_short_by_source"] == {"response": 10}
    assert stats["span_short_share"] == 0.0, "no span answers -> no span-break signal"
    assert stats["short_answer_drop_share"] == 0.1


def test_corpus_rows_take_their_conv_id_from_the_canonical_derivation(monkeypatch):
    """track_s rows carry {prompt_idx, prompt, response} and NO conv_id. The
    drift pairing key must come from the same to_single_turn the capture used to
    build the injected stores, not a re-derived convention free to drift."""
    prep = _prep_module(monkeypatch)
    from issue825_extract_turnstore import to_single_turn

    row = {"prompt_idx": 7, "prompt": "Q?", "response": "the corpus answer, long enough"}
    assert "conv_id" not in row
    assert prep.normalize_source_row(row)["conv_id"] == to_single_turn(row)["conv_id"] == "s7"
    out, _st = prep.prepare([row], None, cell="track_s_injected")
    assert out[0]["conv_id"] == "s7" and out[0]["question"] == "Q?"


# ---------------------------------------------------------------------------
# 3g. Judge-leg RUN table (the 20 authorized cell-legs)
# ---------------------------------------------------------------------------
def test_judge_run_table_covers_both_provenances_and_the_right_references():
    """20 cell-legs: 16 character cells on ai_likeness (BOTH provenances, so the
    labelling axis is a paired arm) + 4 on-policy answer cells on content_drift,
    each against the injected twin its own store was built from."""
    import re
    from pathlib import Path

    src = Path("scripts/issue1345_judge_legs_run.sh").read_text()
    # The character loop enumerates 4 characters x 4 suffixes = 16 ai_likeness cells.
    chars = re.search(r"for ch in ([a-z ]+); do", src).group(1).split()
    suffixes = re.search(r'for suffix in ((?:"[^"]*" ?)+); do', src).group(1)
    n_suffix = len(re.findall(r'"[^"]*"', suffixes))
    assert sorted(chars) == ["dana", "helios", "vex", "wren"], chars
    assert n_suffix == 4, "both provenances x both models = 4 suffixes per character"
    assert len(chars) * n_suffix == 16
    # The drift cells and their references, explicitly.
    drift = dict(re.findall(r'CELLS\+=\("(\w+):content_drift:(\w+)"\)', src))
    assert drift == {
        "op_ntpl_instruct": "track_s_injected",
        "op_ntpl_base": "track_s_injected",
        "op_chat_base": "track_s_injected",
        "op_slot_base": "v1_injected",
    }, drift
    assert len(chars) * n_suffix + len(drift) == 20, "the authorized 20 cell-legs"
    # Spend stays double-gated: the driver FORWARDS the caller's flags and never
    # arms --execute itself, so no invocation of the driver alone can bill.
    built_args = re.search(r"args=\((.*?)\)\n", src, re.S).group(1)
    assert "--execute" not in built_args, f"driver must not self-arm spend: {built_args}"
    assert '"${EXTRA[@]}"' in src, "the caller's flags must reach the judge CLI"
    assert "EPM_I1345_JUDGE_SPEND_OK" in src, "the env ack must be documented at the entrypoint"
    # A single cell's failure must not strand the other 19.
    assert "continuing with the remaining cells" in src


def test_judge_run_uses_prepared_rows_not_raw_uploads():
    """The raw uploads carry no `capped` and the character cells carry no
    `answer`; the driver must consume the PREPARED rows, or the stratifier is
    silently all-natural."""
    from pathlib import Path

    src = Path("scripts/issue1345_judge_legs_run.sh").read_text()
    assert "judge_prep" in src, "must default to the prepared-rows dir"
    assert "issue1345_judge_rows_prep.py first" in src, "a missing prep must say what to run"
    assert "OMP_NUM_THREADS=8" in src, "shared-VM thread caps"


# ---------------------------------------------------------------------------
# 3h. Judge-leg SUMMARY — halted cells never pooled, drop split never blended
# ---------------------------------------------------------------------------
def _summ_module(monkeypatch):
    monkeypatch.setenv("EPM_I1345_VARIANT", "story_boundary_ablation")
    monkeypatch.setenv("EPM_STORY_CHARACTER_NAME", "Assistant")
    import issue1345_judge_legs_summarize as summ

    return summ


def _report(tag, mean, *, halted=False, content=0, total=1500, refusal=0, transport=0):
    return {
        "leg": "ai_likeness",
        "tag": tag,
        "n_items": 300,
        "n_scored_items": 300,
        "n_total_draws": total,
        "n_dropped_draws_content": content,
        "n_refusal_draws": refusal,
        "n_transport_lost_draws": transport,
        "means": {
            "pooled": {"n": 300, "mean": mean},
            "capped": {"n": 0, "mean": None},
            "natural": {"n": 300, "mean": mean},
        },
        "sample_design": {"realized_n": 300, "yield_floor_halted_cell": halted, "seed": 1345},
        "selection_caveat": "halted" if halted else None,
    }


def test_summary_excludes_halted_cells_from_the_cross_cell_mean(monkeypatch):
    """A halted cell's kept rows are a SELECTED subset — averaging it into a
    cross-cell figure would launder the selection into the headline. It is
    reported individually instead, with its caveat."""
    summ = _summ_module(monkeypatch)
    reps = [_report("char_helios", 50.0), _report("char_wren", 90.0, halted=True)]
    blk = summ.summarize(reps)["ai_likeness"]
    assert blk["cross_cell_mean_complete_only"] == 50.0, "the halted 90.0 must not be averaged in"
    assert blk["halted_cells"] == ["char_wren"] and blk["complete_cells"] == ["char_helios"]
    assert "EXCLUDED" in blk["cross_cell_note"] and "char_wren" in blk["cross_cell_note"]
    # ...but the halted cell is still PRESENT as a row, never dropped.
    assert {r["cell"] for r in blk["cells"]} == {"char_helios", "char_wren"}
    halted_row = next(r for r in blk["cells"] if r["cell"] == "char_wren")
    assert halted_row["yield_floor_halted"] and halted_row["selection_caveat"]


def test_summary_keeps_the_three_way_drop_split_unblended(monkeypatch):
    """rule 24: content and transport never blended; REFUSAL is a SUBSET of
    content, so it must not be added to it."""
    summ = _summ_module(monkeypatch)
    row = summ.cell_row(_report("char_vex", 60.0, content=10, refusal=4, transport=7))
    d = row["drops"]
    assert d["content"] == 10 and d["content_refusal_subset"] == 4 and d["transport"] == 7
    assert d["content"] + d["transport"] == 17, "the two classes stay separately reported"
    assert d["content_share"] == round(10 / 1500, 6)


def test_summary_flags_a_truncation_signature_drop_share(monkeypatch):
    """A rule-23 drop share is flagged for re-judge, not averaged through."""
    summ = _summ_module(monkeypatch)
    assert summ.DROP_FLAG_SHARE == 0.02
    clean = summ.cell_row(_report("a", 50.0, content=2))
    dirty = summ.cell_row(_report("b", 50.0, content=500))
    assert clean["drop_flag"] is False and dirty["drop_flag"] is True
    blk = summ.summarize([_report("a", 50.0, content=2), _report("b", 50.0, content=500)])
    assert blk["ai_likeness"]["flagged_cells"] == ["b"]


def test_summary_default_dir_cannot_drift_from_the_driver(monkeypatch):
    """The driver writes under EPM_I1345_JUDGE_OUT; the summary must read the
    SAME var, not the judge CLI's variant-scoped EVAL_DIR default."""
    from pathlib import Path

    monkeypatch.setenv("EPM_I1345_JUDGE_OUT", "/tmp/some/other/place")
    import importlib

    import issue1345_judge_legs_summarize as summ

    importlib.reload(summ)
    assert Path("/tmp/some/other/place") == summ.DEFAULT_LEGS_DIR
    driver = Path("scripts/issue1345_judge_legs_run.sh").read_text()
    assert "EPM_I1345_JUDGE_OUT" in driver
    monkeypatch.delenv("EPM_I1345_JUDGE_OUT")
    importlib.reload(summ)
    assert Path("eval_results/issue_1345/judge_legs") == summ.DEFAULT_LEGS_DIR


# ---------------------------------------------------------------------------
# 3i. V1 gate x provenance — drop-and-count on-policy, fail-loud injected
# ---------------------------------------------------------------------------
def _attrib_name() -> str:
    """The character name ANSWER_ATTRIB_RE was actually COMPILED against.

    The regex is built at MODULE IMPORT from EPM_STORY_CHARACTER_NAME, so the
    first import in the pytest session wins for the whole session — a fixture
    that hardcodes a name passes alone and fails in file order (this is the same
    import-time character-name seam that killed capture job 16283). Read the
    live value instead of assuming it.
    """
    import issue1345_common as common

    return common.STORY_CHARACTER_NAME


def _slot_story(answer: str, *, prefix: str | None = None):
    """A story-slot story: prefix + answer + the appended closing quote."""
    if prefix is None:
        prefix = f'Dana asked, "What is X?" The {_attrib_name()} replied, "'
    return prefix + answer + '"', prefix


def _slot_row(conv_id: str, answer: str, *, gate=None):
    """A story-slot row whose stored spans come from the GATE, not hand-written.

    render_arm's second trust-boundary check compares stored spans against the
    re-run gate's, so a hand-written span number fails as "gate drift" and would
    mask the behaviour under test.
    """
    story, prefix = _slot_story(answer)
    a_start = len(prefix)
    turn = {"q_start": 12, "q_end": 25, "a_start": a_start, "a_end": a_start + len(answer)}
    if gate is not None:
        re_turn, reason = gate(story, answer)
        if reason == "ok" and re_turn is not None:
            turn = dict(re_turn)
    return {"conv_id": conv_id, "story": story, "answer": answer, "parsed_turns": [turn]}


def test_onpolicy_gate_reject_drops_and_counts_instead_of_asserting(monkeypatch):
    """An on-policy answer ENDING in attribution-shaped words + the appended
    closing quote makes the reassembled story carry a SECOND attribution match.
    The answer alone carries zero, so it is a product of the reassembly — the row
    is un-gateable, not evidence the gate drifted. Measured 3/2089 on the real
    pool; the capture used to die on the first one mid-GPU-run."""
    cap = _cap_module(monkeypatch)
    import unittest.mock as m

    import issue1345_common as common

    gate = cap.gate_for_capture(cap.V1_ARM)
    good = _slot_row(
        "s1", "A perfectly ordinary answer with no attribution words in it.", gate=gate
    )
    bad = _slot_row("s2", f"It works because, as the {_attrib_name()} explained,", gate=gate)
    # The trigger really is the reassembly, not the answer text.
    assert len(list(common.ANSWER_ATTRIB_RE.finditer(bad["answer"]))) == 0
    assert len(list(common.ANSWER_ATTRIB_RE.finditer(bad["story"]))) == 2
    with m.patch.object(cap, "render_boundary_turn", lambda *a, **k: None):
        _r, st = cap.render_arm(cap.V1_ARM, [good, bad], object(), provenance=common.PROV_ONPOLICY)
    assert st["gate_rejects"] == 1
    assert st["gate_reject_reasons"] == {"attribution_multi": 1}
    assert st["gate_reject_conv_ids"] == ["s2"], "the dropped ids must be RECORDED"
    assert st["stories"] == 2


def test_injected_gate_reject_still_fails_loud(monkeypatch):
    """For INJECTED provenance the story is template-built around a gate-checked
    answer, so a second attribution really IS drift and must not be skipped."""
    cap = _cap_module(monkeypatch)
    import issue1345_common as common

    bad = _slot_row(
        "s2",
        f"It works because, as the {_attrib_name()} explained,",
        gate=cap.gate_for_capture(cap.V1_ARM),
    )
    with pytest.raises(AssertionError, match="attribution_multi"):
        cap.render_arm(cap.V1_ARM, [bad], object(), provenance=common.PROV_INJECTED)
    # The message names the provenance so the verdict is attributable.
    try:
        cap.render_arm(cap.V1_ARM, [bad], object(), provenance=common.PROV_INJECTED)
    except AssertionError as e:
        assert "provenance=injected" in str(e)


def test_onpolicy_skip_is_scoped_to_the_named_reject_classes(monkeypatch):
    """Only the enumerated classes are skippable — any OTHER gate verdict still
    asserts under on-policy, so the relaxation cannot swallow real drift."""
    cap = _cap_module(monkeypatch)
    import unittest.mock as m

    import issue1345_common as common

    assert cap.ONPOLICY_EXPECTED_GATE_REJECTS == ("attribution_multi",)
    row = _slot_row(
        "s1",
        "An answer long enough to be gated normally here.",
        gate=cap.gate_for_capture(cap.V1_ARM),
    )
    with (
        m.patch.object(
            cap, "gate_for_capture", lambda arm: lambda s, a: (None, "answer_quote_not_closed")
        ),
        pytest.raises(AssertionError, match="answer_quote_not_closed"),
    ):
        cap.render_arm(cap.V1_ARM, [row], object(), provenance=common.PROV_ONPOLICY)


def test_render_arm_rejects_an_unknown_provenance(monkeypatch):
    cap = _cap_module(monkeypatch)
    with pytest.raises(AssertionError, match="unknown provenance"):
        cap.render_arm(cap.V1_ARM, [], object(), provenance="bogus")


def test_gen_side_runs_the_consumers_own_gate_at_assembly(monkeypatch):
    """The durable fix: gen runs the CAPTURE's gate on the assembled story, so no
    gate class can reach the capture's assert at all. A re-derived local check
    would be free to drift from the gate that actually asserts."""
    gen = _gen_module(monkeypatch)
    import inspect

    src = inspect.getsource(gen.assemble_row)
    assert "_v1_gate_cached()" in src, "gen must run the consumer's gate, not a re-derived check"
    assert "v1_gate_" in src, "the drop reason must be namespaced to the gate"
    cached = inspect.getsource(gen._v1_gate_cached)
    assert "gate_for_capture" in cached, "must import the capture's own gate"

    prefix = f'Dana asked, "What is X?" The {_attrib_name()} replied, "'
    pool = {
        "conv_id": "s2",
        "prefix": prefix,
        "source_story": prefix + "x" * 80 + '"',
        "turn": {"q_start": 12, "q_end": 25, "a_start": len(prefix), "a_end": len(prefix)},
    }
    row, reason = gen.assemble_row(
        pool,
        f"It works because, as the {_attrib_name()} explained,",
        shape=gen.SHAPE_STORY_SLOT,
        model_key="pretrained",
    )
    assert row is None and reason == "v1_gate_attribution_multi", reason


def test_gen_tally_counts_namespaced_gate_reasons_without_crashing(monkeypatch):
    """`keep_rows` asserts every drop reason is pre-declared; a gate reason we
    have not seen before must be COUNTED, not crash the run or vanish."""
    gen = _gen_module(monkeypatch)
    import inspect

    src = inspect.getsource(gen.keep_rows)
    assert 'reason.startswith("v1_gate_")' in src
    assert "counts.setdefault(reason, 0)" in src
    # The strict assert must survive for non-namespaced reasons.
    assert 'assert reason in counts, f"unaccounted drop reason {reason!r}"' in src


def test_max_tokens_is_overridable_upward_only(monkeypatch):
    """A rule-23 truncation re-judge needs a bigger budget, but the #1916 floor
    must still hold — a caller may only go UP."""
    jl = _judge_module(monkeypatch)
    import inspect

    sig = inspect.signature(jl.run_leg)
    assert sig.parameters["max_tokens"].default == jl.JUDGE_MAX_TOKENS
    src = inspect.getsource(jl.run_leg)
    assert "max_tokens >= JUDGE_MAX_TOKENS" in src, "the floor must be enforced at run_leg"
    assert "max_tokens=max_tokens" in src, "the override must reach judge_graded"
    assert '"max_tokens": max_tokens' in src, "the report must record the budget ACTUALLY used"
    main_src = inspect.getsource(jl.main)
    assert "max_tokens=args.max_tokens" in main_src
    # A below-floor value fails loud rather than silently re-truncating.
    with pytest.raises(AssertionError, match="below the #1916"):
        jl.run_leg(
            jl.LEG_AI_LIKENESS,
            [("ail_x_s1", "q", "a")],
            Path("/tmp/nope-i1345"),
            "x",
            execute=False,
            max_tokens=64,
        )


# ---------------------------------------------------------------------------
# 3j. Leading-whitespace span convention (the unit-4 a_start +1 class)
# ---------------------------------------------------------------------------
def test_gen_anchors_a_start_at_the_first_content_char(monkeypatch):
    """The parent V1 convention puts a_start at the answer's first CONTENT char —
    the gate re-derives it by NORMALIZED occurrence search, so a space between the
    opening quote and the match belongs to neither. A space-initial answer stored
    at a_start = len(prefix) points the span AT the space and the capture's
    span-consistency assert sees +1. Measured 7/2089, and zero of the 2,082
    space-free rows disagree."""
    gen = _gen_module(monkeypatch)

    cap = _cap_module(monkeypatch)
    prefix = f'Dana asked, "What is X?" The {_attrib_name()} replied, "'
    body = "A perfectly ordinary answer with plenty of characters in it."
    # The q/boundary spans come from the GATE, not hand-written numbers: the gen
    # side now compares every span key against the gate's re-derivation, so a
    # hand-written q_end is (correctly) rejected as a mismatch.
    probe_turn, probe_reason = cap.gate_for_capture(cap.V1_ARM)(prefix + body + '"', body)
    assert probe_reason == "ok", probe_reason
    pool = {
        "conv_id": "s1",
        "prefix": prefix,
        "source_story": prefix + body + '"',
        "turn": {**probe_turn, "a_start": len(prefix), "a_end": len(prefix)},
    }
    plain, r1 = gen.assemble_row(pool, body, shape=gen.SHAPE_STORY_SLOT, model_key="pretrained")
    spaced, r2 = gen.assemble_row(
        pool, "   " + body, shape=gen.SHAPE_STORY_SLOT, model_key="pretrained"
    )
    assert r1 == "ok" and r2 == "ok", (r1, r2)
    # Both anchor a_start at the first content char — identical spans and answer.
    assert (
        spaced["parsed_turns"][0]["a_start"] == plain["parsed_turns"][0]["a_start"] == len(prefix)
    )
    assert spaced["answer"] == plain["answer"] == body
    assert spaced["leading_ws_stripped"] == 3 and plain["leading_ws_stripped"] == 0
    # And the stored span still reproduces the answer byte-for-byte.
    t = spaced["parsed_turns"][0]
    assert spaced["story"][t["a_start"] : t["a_end"]] == spaced["answer"]


def test_gen_drops_a_row_whose_spans_disagree_with_the_gate(monkeypatch):
    """The gate returning 'ok' is only the FIRST of the capture's two
    trust-boundary checks; a row can pass the gate and still disagree on a span.
    That comparison must happen at GEN time, not as a mid-capture assert."""
    gen = _gen_module(monkeypatch)
    import inspect

    src = inspect.getsource(gen.assemble_row)
    assert "v1_gate_span_mismatch_" in src, "the span comparison must be a named drop class"
    for key in ("q_start", "q_end", "boundary_end", "a_start", "a_end"):
        assert key in src
    tally = inspect.getsource(gen.keep_rows)
    assert 'counts["v1_gate_span_mismatch"] += 1' in tally, "span mismatches must be counted"


def test_capture_normalizes_leading_ws_spans_for_preexisting_rows(monkeypatch):
    """Rows generated BEFORE the writer lstripped are already uploaded, so the
    capture normalizes the convention at load — advancing a_start by exactly the
    whitespace removed and leaving BOTH trust-boundary asserts to run unchanged."""
    cap = _cap_module(monkeypatch)

    prefix = f'Dana asked, "What is X?" The {_attrib_name()} replied, "'
    body = "An answer with enough characters to pass the floor."
    story = prefix + "  " + body + '"'
    a0 = len(prefix)
    row = {
        "conv_id": "s1",
        "story": story,
        "answer": "  " + body,
        "parsed_turns": [
            {"q_start": 12, "q_end": 25, "a_start": a0, "a_end": a0 + len("  " + body)}
        ],
    }
    out, stats = cap.normalize_onpolicy_leading_ws([row])
    assert stats["normalized"] == 1 and stats["conv_ids"] == ["s1"]
    t = out[0]["parsed_turns"][0]
    assert t["a_start"] == a0 + 2, "a_start must advance by exactly the stripped whitespace"
    assert out[0]["answer"] == body
    assert out[0]["story"][t["a_start"] : t["a_end"]] == body, "the span still IS the answer"
    # A row with no leading whitespace is passed through untouched.
    clean = {
        "conv_id": "s2",
        "story": prefix + body + '"',
        "answer": body,
        "parsed_turns": [{"q_start": 12, "q_end": 25, "a_start": a0, "a_end": a0 + len(body)}],
    }
    out2, stats2 = cap.normalize_onpolicy_leading_ws([clean])
    assert stats2["normalized"] == 0 and out2[0] is clean


def test_capture_normalization_is_not_a_trust_the_gate_override(monkeypatch):
    """It only moves a_start past whitespace it actually removed — it never adopts
    the gate's number, so a genuine drift still fails the unchanged assert."""
    cap = _cap_module(monkeypatch)
    import inspect

    src = inspect.getsource(cap.normalize_onpolicy_leading_ws)
    assert "gate" not in src.split('"""')[2], "the normalization must not consult the gate"
    assert 'int(turn["a_start"]) + lead' in src, "a_start moves by the stripped width only"
    # A row whose spans are wrong for some OTHER reason is left alone to assert.
    prefix = "x" * 20
    bogus = {
        "conv_id": "s3",
        "story": prefix + "answer text here" + '"',
        "answer": "answer text here",
        "parsed_turns": [{"a_start": 999, "a_end": 1010}],
    }
    out, stats = cap.normalize_onpolicy_leading_ws([bogus])
    assert stats["normalized"] == 0 and out[0]["parsed_turns"][0]["a_start"] == 999


# ---------------------------------------------------------------------------
# 3k. AI-likeness axis validation against the rule-25 neighbour channels
# ---------------------------------------------------------------------------
def _axis_module(monkeypatch):
    monkeypatch.setenv("EPM_I1345_VARIANT", "story_boundary_ablation")
    monkeypatch.setenv("EPM_STORY_CHARACTER_NAME", "Assistant")
    import issue1345_judge_axis_validation as axis

    return axis


@pytest.mark.parametrize(
    ("cell", "want"),
    [
        ("char_helios", "helios"),
        ("char_helios_base", "helios"),
        ("char_helios_op", "helios"),
        ("char_helios_op_base", "helios"),
        ("char_dana_op_base", "dana"),
    ],
)
def test_character_name_derives_from_the_cell_tag(monkeypatch, cell, want):
    """Every provenance/model suffix must strip to the same character, or the
    name channel would silently look for the wrong string."""
    axis = _axis_module(monkeypatch)
    assert axis.character_name_of(cell) == want


def test_name_regex_is_word_bounded_and_case_insensitive(monkeypatch):
    """The stories use both HELIOS and Helios; a case-sensitive probe would
    undercount the channel and overstate how name-free the text is."""
    axis = _axis_module(monkeypatch)
    r = axis.name_regex("helios")
    assert r.search("HELIOS considered the question")
    assert r.search("Helios replied at once")
    assert not r.search("heliospheric physics"), "must not match inside a longer word"


def test_name_swap_max_shift_is_bounded_by_the_carrying_share(monkeypatch):
    """The number that made the authorized ablation unnecessary: a name swap can
    only move the pooled mean in proportion to the name-carrying share, so a
    channel present in a handful of 300 rows caps the achievable movement at a
    few points. Bound to the SHIPPED artifact so the claim stays checkable."""
    import inspect
    import json
    from pathlib import Path

    axis = _axis_module(monkeypatch)
    src = inspect.getsource(axis.validate_cell)
    assert "100.0 * n_name / n_all" in src, "the bound must be computed, not asserted in prose"

    p = Path("eval_results/issue_1345/judge_legs/axis_validation.json")
    if not p.exists():  # the run's artifact is not present in every checkout
        pytest.skip("axis_validation.json not present")
    cells = {r["cell"]: r for r in json.loads(p.read_text())["cells"]}
    for cell, rec in cells.items():
        n_name = rec["name_channel"]["carries"]["n"]
        n_all = rec["pooled"]["n"]
        assert rec["name_swap_max_pooled_shift"] == round(100.0 * n_name / n_all, 3), cell
        # Every INJECTED cell must have zero name-channel leverage (the
        # instruct-written answers never name the character).
        if "_op" not in cell:
            assert n_name == 0, f"{cell} unexpectedly carries the name channel"
    # And the on-policy cells' leverage is small enough that a swap cannot decide
    # anything — the finding that made the 1,500-call ablation unnecessary.
    op = [r for k, r in cells.items() if "_op" in k]
    assert op, "no on-policy cells in the artifact"
    assert max(r["name_swap_max_pooled_shift"] for r in op) < 5.0


def test_block_reports_n_mean_sd(monkeypatch):
    axis = _axis_module(monkeypatch)
    b = axis._block([10.0, 20.0, 30.0])
    assert b["n"] == 3 and b["mean"] == 20.0 and b["sd"] is not None
    assert axis._block([])["mean"] is None
    assert axis._block([5.0])["sd"] is None, "sd undefined at n=1, not fabricated"


def test_ai_word_pattern_covers_the_rubrics_excluded_self_reference(monkeypatch):
    """The rubric excludes 'explicitly SAYING it is an AI'; the channel probe must
    match that vocabulary or the exclusion goes unchecked."""
    axis = _axis_module(monkeypatch)
    for s in ("I am an AI", "as a language model", "artificial intelligence", "your assistant"):
        assert axis.AI_WORD_RE.search(s), s
    assert not axis.AI_WORD_RE.search("the aircraft banked left")


# ---------------------------------------------------------------------------
# 3l. Cell sharding — parallelize the fits across instances
# ---------------------------------------------------------------------------
def _fits_module(monkeypatch):
    monkeypatch.setenv("EPM_I1345_VARIANT", "story_boundary_ablation")
    monkeypatch.setenv("EPM_STORY_CHARACTER_NAME", "Assistant")
    import issue1345_boundary_ablation_fits as fits

    return fits


def _cell_ids(n_arms: int = 10) -> list[dict]:
    arms = ["v1", "v2", "v3", "v4", "v5", "chat", "ntpl", "chat_op", "ntpl_op", "v1_op"][:n_arms]
    return [
        {"cell_id": f"R_instruct_bnd_{a}_{s}__{y}"}
        for a in arms
        for s in ("prefix", "ctx_qend", "context", "ctx_preans", "ctx_straddle")
        for y in ("ymean", "ybound")
    ]


def test_cell_shard_spec_parsing(monkeypatch):
    fits = _fits_module(monkeypatch)
    assert fits.parse_cell_shard(None) is None
    assert fits.parse_cell_shard("") is None
    assert fits.parse_cell_shard("  ") is None, "an env var set to blank is unsharded"
    assert fits.parse_cell_shard("0/4") == (0, 4)
    assert fits.parse_cell_shard(" 3/4 ") == (3, 4)
    for bad in ("4/4", "-1/4", "1/0", "a/4", "1/2/3", "4"):
        with pytest.raises(AssertionError):
            fits.parse_cell_shard(bad)


def test_cell_shard_partition_is_exhaustive_and_disjoint(monkeypatch):
    """Every cell must be fit exactly once across the fleet — a partition that
    double-covers wastes an instance, and one that under-covers silently ships a
    lattice with holes."""
    fits = _fits_module(monkeypatch)
    cells = _cell_ids()
    for n in (2, 3, 4, 8):
        parts = [fits.apply_cell_shard(cells, (i, n)) for i in range(n)]
        ids = [{cl["cell_id"] for cl in p} for p in parts]
        assert set().union(*ids) == {cl["cell_id"] for cl in cells}, f"n={n} not exhaustive"
        assert sum(len(x) for x in ids) == len(cells), f"n={n} shards overlap"


def test_cell_shard_ownership_is_stable_across_processes_and_set_changes(monkeypatch):
    """The partition keys on a sha256 of cell_id, NOT the enumeration index and
    NOT hash(): the op rows are presence-gated, so two instances can enumerate
    different-sized lists, and hash() is salted per process. Either would
    reassign cells between shards — fitting some twice and others never."""
    fits = _fits_module(monkeypatch)
    cells = _cell_ids()
    n = 4
    owner = {cl["cell_id"]: fits.shard_of_cell(cl["cell_id"], n) for cl in cells}
    # Same answer on a repeat call (and the value is a pure function of the id).
    for cid, want in owner.items():
        assert fits.shard_of_cell(cid, n) == want
    # A SHRUNKEN enumeration (op arm absent) must not move anyone's owner.
    shrunk = [cl for cl in cells if "_op" not in cl["cell_id"]]
    for i in range(n):
        got = {cl["cell_id"] for cl in fits.apply_cell_shard(shrunk, (i, n))}
        assert got == {cid for cid, o in owner.items() if o == i and "_op" not in cid}


def test_unsharded_path_is_byte_identical(monkeypatch):
    """The running instance's semantics must not move: no shard => same list."""
    fits = _fits_module(monkeypatch)
    cells = _cell_ids()
    assert fits.apply_cell_shard(cells, None) is cells


def test_whole_lattice_phases_refuse_to_run_sharded(monkeypatch):
    """grid/reparam/verdict consume EVERY cell's output; under a shard they would
    compute over a fraction and report it complete. The guard is checked before
    any store access so it fails fast and on a box with nothing staged."""
    fits = _fits_module(monkeypatch)
    import inspect

    assert set(fits.WHOLE_LATTICE_PHASES) == {"all", "grid", "reparam", "verdict"}
    src = inspect.getsource(fits.main)
    guard = src.index("consumes EVERY cell")
    stores = src.index("bg.assert_round_env()")
    assert guard < stores, "the shard/phase guard must precede any store access"


def test_shard_writes_a_scoped_summary_name(monkeypatch):
    """A shard holds a slice, so it must not write the filename the whole-lattice
    phases read — a partial cell_summary.json staged onto the union box would
    satisfy grid's existence assert and silently supply a fraction of the grid."""
    fits = _fits_module(monkeypatch)
    import inspect

    src = inspect.getsource(fits.main)
    assert 'f"cell_summary.shard{shard[0]}of{shard[1]}.json"' in src
    assert '"cell_summary.json"\n            if shard is None' in src


def test_union_stage_pulls_preds_and_refuses_a_half_staged_union(monkeypatch):
    """The resume predicate needs the cell JSON AND its preds npz, and the preds
    live OUTSIDE the mirrored eval tree — so `mirror` uploads them under their own
    prefix and `stage-cells` pulls both back. Staging JSONs without preds would
    refit every cell while looking healthy, so it fails loud instead."""
    monkeypatch.setenv("EPM_I1345_VARIANT", "story_boundary_ablation")
    monkeypatch.setenv("EPM_STORY_CHARACTER_NAME", "Assistant")
    import inspect

    import issue1345_boundary_ablation_stage_and_mirror as sm

    assert sm.HF_PREDS_MIRROR_PREFIX != sm.HF_EVAL_MIRROR_PREFIX
    assert "preds_cache" in str(sm.PREDS_OUT_DIR)
    # preds are NOT inside the eval mirror tree — the asymmetry this closes.
    assert not str(sm.PREDS_OUT_DIR.resolve()).startswith(str(sm.EVAL_OUT_DIR.resolve()) + "/")
    mirror_src = inspect.getsource(sm.cmd_mirror)
    assert "HF_PREDS_MIRROR_PREFIX" in mirror_src, "mirror must upload preds too"
    stage_src = inspect.getsource(sm.cmd_stage_cells)
    assert "would REFIT" in stage_src, "a JSONs-without-preds union must fail loud"
    assert "stage-cells" in inspect.getsource(sm.main)


# ---------------------------------------------------------------------------
# 3m. Full-pool companion lattice (Option A) — isolation + allowlist bypass
# ---------------------------------------------------------------------------
def test_full_pool_bypasses_the_arm_matched_comparator_filter(monkeypatch):
    """The comparator stores hold 2,936 of 4,472 because load_comparator_convs
    filters to the arm-kept union. --full-pool passes keep_ids=None, which that
    function ALREADY handles — the companion needs the bypass, not a new path."""
    cap = _cap_module(monkeypatch)
    import inspect

    src = inspect.getsource(cap.main)
    assert "if args.full_pool" in src and "arm_kept_conv_ids" in src
    # The None branch is the pre-existing full-pool path, not something new.
    load_src = inspect.getsource(cap.load_comparator_convs)
    assert "if keep_ids is None:\n        return convs" in load_src
    convs = [{"conv_id": "s1", "prompt": "q", "response": "a"} for _ in range(1)]
    assert convs, "fixture"


def test_companion_isolates_by_hf_prefix_not_by_a_sibling_variant(monkeypatch):
    """Isolation must not go through EPM_I1345_VARIANT: assert_round_env hard-
    refuses any variant but the round's, and that guard exists to stop wrong-scope
    runs clobbering the V1 anchor. A prefix suffix isolates the companion without
    weakening it for every future run."""
    cap = _cap_module(monkeypatch)
    import issue1345_boundary_ablation_gen as gen_mod

    base = cap.hf_tensor_prefix(False)
    comp = cap.hf_tensor_prefix(False, "fulln")
    assert comp == f"{base}_fulln" and comp != base
    assert cap.hf_tensor_prefix(False, "") == base, "no suffix => byte-identical default"
    # The guard the sibling-variant route would have had to weaken is still hard.
    import inspect

    guard = inspect.getsource(gen_mod.assert_round_env)
    assert "c.VARIANT == ROUND_VARIANT" in guard


def test_capture_records_which_row_pool_a_store_came_from(monkeypatch):
    """A full-pool store and a matched store share stem/model/provenance, so
    without this field the only thing telling them apart is the directory they
    happen to sit in — and a mis-staged store would read as the other one."""
    cap = _cap_module(monkeypatch)
    import inspect

    src = inspect.getsource(cap.main)
    # Three distinguishable pools, because the companion introduced a third:
    # arm-matched (the headline), full (the n>d companion), and matched-to-file
    # (the companion's injected twin, pinned to its partner's conv_ids).
    for token in ('"row_pool"', '"arm_matched"', '"full"', '"matched_to_file"'):
        assert token in src, token
    assert '"full_pool": bool(args.full_pool)' in src
    assert '"keep_ids_source"' in src


def test_persist_store_threads_the_prefix_suffix(monkeypatch):
    """Without the thread-through the companion would upload onto the matched
    tree — same stems, silent clobber of the primary lattice's stores."""
    cap = _cap_module(monkeypatch)
    import inspect

    assert "hf_prefix_suffix" in inspect.signature(cap.persist_store).parameters
    body = inspect.getsource(cap.persist_store)
    assert "hf_tensor_prefix(smoke, hf_prefix_suffix)" in body
    assert "hf_prefix_suffix=args.hf_prefix_suffix" in inspect.getsource(cap.main)


def test_grid_and_paired_cells_take_no_fits_side_allowlist(monkeypatch):
    """The companion needs NO fits change: allow[] is populated only for the
    comparator/matched cells, so grid + on-policy paired cells already fit over
    whatever rows their STORE holds. Row count is a capture-time property."""
    fits = _fits_module(monkeypatch)
    import inspect
    import re

    src = inspect.getsource(fits.main)
    sites = re.findall(r"allow\[cell\[.cell_id.\]\] = (\S+)", src)
    assert sites == ["sorted(arm_convs[arm])", "ids"], sites
    # ...and neither assignment sits in the grid / on-policy block.
    blk = src[src.index("cells += grid_cells(key)") : src.index("cells += op_cells")]
    assert "allow[" not in blk


def test_keep_ids_jsonl_matches_a_pair_at_identical_n(monkeypatch):
    """The companion's injected pool is 5,000 while its on-policy twins are
    4,267-4,618. At n/d 1.19-1.40 held-out R^2 moves with n/d, so an unmatched
    pair would vary the very quantity the companion exists to hold fixed —
    --keep-ids-jsonl pins the injected capture to its partner's conv_ids."""
    cap = _cap_module(monkeypatch)
    import inspect

    src = inspect.getsource(cap.main)
    # Precedence: an explicit id file beats --full-pool, never the other way.
    i_ids = src.index("if args.keep_ids_jsonl is not None:")
    i_full = src.index("elif args.full_pool:")
    assert i_ids < i_full, "--keep-ids-jsonl must take precedence over --full-pool"
    assert 'r["conv_id"]' in src and "--keep-ids-jsonl missing" in src
    # The pool provenance must be distinguishable in the manifest.
    assert '"matched_to_file"' in src and '"keep_ids_source"' in src


def test_v1_family_cannot_reach_the_companion_n_band(monkeypatch):
    """Scope correction worth pinning: the V1 arm's pool is the PINNED parent
    bundle (2,164 rows) and its on-policy slot twin is 2,089 — both BELOW
    d=3,584, i.e. still in the n<d regime the companion exists to escape. Only
    the chat / no_template families reach n>d."""
    cap = _cap_module(monkeypatch)
    d_model = 3584
    assert cap.V1_KEPT_ROWS == 2164
    assert d_model > cap.V1_KEPT_ROWS, "V1 cannot supply an n>d companion cell"


# ---------------------------------------------------------------------------
# 3n. --no-arms companion lattice (the Option-A fits blocker)
# ---------------------------------------------------------------------------
def test_no_arms_runs_a_zero_arm_companion_lattice(monkeypatch):
    """The fulln turnstore deliberately has no v2-v5 stores, so the default
    --arms demanded sidecars that do not exist and `assert arms and ...` refused
    an empty --arms outright. --no-arms runs the comparator / V1-grid / on-policy
    paired cells with zero ablation arms."""
    fits = _fits_module(monkeypatch)
    import inspect

    src = inspect.getsource(fits.main)
    assert "args.no_arms" in src
    # The non-empty requirement survives for the ORDINARY path...
    assert "assert args.no_arms or arms" in src
    # ...and the subset check is now unconditional rather than fused to it.
    assert "assert set(arms) <= set(bg.GEN_ARMS), arms" in src


def test_accidentally_empty_arms_still_fails_loud(monkeypatch):
    """Relaxing the assert outright would make a typo'd or unset --arms silently
    produce an arms-free lattice. The empty case must be EXPLICIT."""
    fits = _fits_module(monkeypatch)
    import inspect

    src = inspect.getsource(fits.main)
    i_parse = src.index("arms = [] if args.no_arms else")
    i_assert = src.index("assert args.no_arms or arms")
    assert i_parse < i_assert
    assert "pass --no-arms to run the companion" in src, "the refusal must name the remedy"


def test_every_arms_keyed_consumer_is_empty_safe(monkeypatch):
    """With zero arms each arms-keyed consumer must yield empty rather than
    raise: arm_convs, the per-arm cell loop, paired_by_arm, reparam_by_arm and
    the verdict comprehension are all `for arm in arms`, and reparam's subset
    assert is empty-subset-empty."""
    fits = _fits_module(monkeypatch)
    import inspect

    src = inspect.getsource(fits.main)
    for frag in (
        "for arm in arms\n",
        "for arm in arms:",
        "assert set(reparam_arms) <= set(arms)",
    ):
        assert frag in src, frag
    # Behavioural check of the same shape the driver relies on.
    arms: list[str] = []
    assert {a: object() for a in arms} == {}
    assert [a for a in arms if a in ("v2",)] == []
    assert set([]) <= set(arms)


def test_empty_lattice_fails_loud_instead_of_reporting_success(monkeypatch):
    """Observed: --no-arms against a turnstore with no stores wrote an empty
    cell_summary + xy_grid and printed its normal done line having fit NOTHING.
    The arms-free lattice removed the incidental non-emptiness arm cells used to
    provide, and the companion run is exactly where a mis-pointed turnstore is
    plausible."""
    fits = _fits_module(monkeypatch)
    import inspect

    src = inspect.getsource(fits.main)
    assert "enumeration produced ZERO cells" in src
    i_guard = src.index("enumeration produced ZERO cells")
    i_bundles = src.index("bundles: dict[tuple[str, str], dict] = {}")
    assert i_guard < i_bundles, "the empty-lattice guard must precede bundle loading"
