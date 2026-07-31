"""Tests for the #1739 wildchat-rung arm-scoring instance driver.

Covers the staging seams (the ONE shared capture store + the two DV trees),
the per-behavior scorer argv, the upload prefix, and the sentinel. Every hub
fake is built with ``create_autospec`` so a wrong-arity / wrong-keyword call
against the REAL signature fails the test instead of surfacing as a runtime
TypeError on a billed instance (the #1332 signature-bind class — the shipped
2-positional ``stage_hub_prefix`` bug this file's bind test pins).
"""

from __future__ import annotations

import inspect
import json
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from unittest.mock import create_autospec

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from explore_persona_space.orchestrate import hub  # noqa: E402
from scripts import issue1739_wcrung_arms_run as run  # noqa: E402

BEHAVIORS = ("evil", "sycophancy", "hallucination")


def _args(tmp_path: Path, **kw) -> Namespace:
    args = run.parse_args(
        [
            "--store-root",
            str(tmp_path / "hf_dl"),
            "--out-root",
            str(tmp_path / "out" / "wildchat_rung"),
            "--main-root",
            str(tmp_path / "main"),
            "--tensors-root",
            str(tmp_path / "tensors"),
        ]
    )
    for k, v in kw.items():
        setattr(args, k, v)
    args.store_root.mkdir(parents=True, exist_ok=True)
    return args


# ---------------------------------------------------------------------------
# constants / defaults
# ---------------------------------------------------------------------------


def test_stage_order_is_smallest_tar_first():
    assert run.STAGE_ORDER == ("evil", "sycophancy", "hallucination")
    sizes = [run.TAR_GIB[b] for b in run.STAGE_ORDER]
    assert sizes == sorted(sizes)


def test_rung_prefix_and_sentinel_name_target_the_wildchat_rung():
    assert run.RUNG_PREFIX == "issue1739_ctxmap/wildchat_rung"
    assert run.SENTINEL_NAME == "wcrung_arms_done.json"
    assert run.EVAL_STORE_DIR_NAME == "wildchat"


def test_out_root_default_is_the_wildchat_rung_subtree():
    args = run.parse_args([])
    # the scorer refuses an --out-root whose name is not the rung
    assert args.out_root.name == "wildchat_rung"


def test_train_dv_root_defaults_under_store_root(tmp_path):
    args = run.parse_args(["--store-root", str(tmp_path / "sr")])
    assert args.train_dv_root == tmp_path / "sr" / "train_dv"


# ---------------------------------------------------------------------------
# staging: the shared capture store
# ---------------------------------------------------------------------------


def test_stage_wcrung_store_binds_the_real_hub_signature(tmp_path, monkeypatch):
    """REGRESSION PIN (#1332 class): stage_hub_prefix takes repo_id FIRST.

    The leg originally called it with 2 positional args (prefix, dest) — a
    deterministic ``TypeError: missing a required argument: 'dest_dir'`` the
    moment a fresh instance had to stage the store. The autospec fake enforces
    the real signature, so the pre-fix call fails here.
    """
    args = _args(tmp_path)
    fake = create_autospec(hub.stage_hub_prefix)

    def _stage(repo_id, prefix, dest_dir, **kw):
        staged = Path(dest_dir) / prefix
        staged.mkdir(parents=True, exist_ok=True)
        (staged / run.STORE_PROBE).write_text("{}")
        return [staged / run.STORE_PROBE]

    fake.side_effect = _stage
    monkeypatch.setattr(hub, "stage_hub_prefix", fake)

    dest = run.stage_wcrung_store(args)
    assert dest == args.store_root / "wcrung_capture_store" / "wildchat"
    assert (dest / run.STORE_PROBE).exists()

    (call_args, call_kw) = fake.call_args
    # repo_id first, then the rung-scoped prefix, then the mirror ROOT
    assert call_args[0] == hub.DEFAULT_DATASET_REPO
    assert call_args[1] == "issue1739_ctxmap/wildchat_rung/capture_store/wildchat"
    assert call_args[2] == args.store_root / "_wcmirror"
    assert call_kw["repo_type"] == "dataset"
    # and the bound shape is legal against the live signature
    inspect.signature(hub.stage_hub_prefix).bind(*call_args, **call_kw)


def test_stage_wcrung_store_skips_when_already_present(tmp_path, monkeypatch):
    args = _args(tmp_path)
    dest = args.store_root / "wcrung_capture_store" / "wildchat"
    dest.mkdir(parents=True)
    (dest / run.STORE_PROBE).write_text("{}")
    fake = create_autospec(hub.stage_hub_prefix)
    monkeypatch.setattr(hub, "stage_hub_prefix", fake)
    assert run.stage_wcrung_store(args) == dest
    fake.assert_not_called()


def test_stage_wcrung_store_fails_loud_on_incomplete_staging(tmp_path, monkeypatch):
    """An empty/partial mirror must raise, never rename a store with no shards."""
    args = _args(tmp_path)
    fake = create_autospec(hub.stage_hub_prefix)
    fake.side_effect = lambda repo_id, prefix, dest_dir, **kw: (
        (Path(dest_dir) / prefix).mkdir(parents=True, exist_ok=True) or []
    )
    monkeypatch.setattr(hub, "stage_hub_prefix", fake)
    with pytest.raises(RuntimeError, match="staging incomplete"):
        run.stage_wcrung_store(args)
    assert not (args.store_root / "wcrung_capture_store" / "wildchat").exists()


# ---------------------------------------------------------------------------
# staging: the two DV trees
# ---------------------------------------------------------------------------


def test_stage_shared_fetches_rung_dv_and_train_dv_per_behavior(tmp_path, monkeypatch):
    """One rung DV + one train DV per behavior, from the right HF paths."""
    args = _args(tmp_path, behaviors=list(BEHAVIORS))
    store_dest = args.store_root / "wcrung_capture_store" / "wildchat"
    store_dest.mkdir(parents=True)
    (store_dest / run.STORE_PROBE).write_text("{}")  # store already staged

    grabbed: list[tuple[str, Path]] = []
    fake_file = create_autospec(hub.stage_hub_file)

    def _file(repo_id, path_in_repo, target, **kw):
        assert repo_id == hub.DEFAULT_DATASET_REPO
        assert kw["repo_type"] == "dataset"
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text('{"rows": []}')
        grabbed.append((path_in_repo, Path(target)))
        return Path(target)

    fake_file.side_effect = _file
    monkeypatch.setattr(hub, "stage_hub_file", fake_file)

    run.stage_shared(args, "tok")

    by_repo_path = dict(grabbed)
    for behavior in BEHAVIORS:
        rung_src = f"issue1739_ctxmap/wildchat_rung/dv_dataset/{behavior}/labeling.json"
        train_src = f"issue1739_ctxmap/judge/dv_dataset/{behavior}/labeling.json"
        assert rung_src in by_repo_path, f"{behavior}: rung DV not staged"
        assert train_src in by_repo_path, f"{behavior}: train DV not staged"
        # the rung DV lands where the SCORER looks for it by default
        assert by_repo_path[rung_src] == args.out_root / "dv_dataset" / behavior / "labeling.json"
        assert by_repo_path[train_src] == args.train_dv_root / behavior / "labeling.json"
    assert len(grabbed) == 2 * len(BEHAVIORS)


def test_stage_shared_skips_dvs_already_on_disk(tmp_path, monkeypatch):
    args = _args(tmp_path, behaviors=["evil"])
    store_dest = args.store_root / "wcrung_capture_store" / "wildchat"
    store_dest.mkdir(parents=True)
    (store_dest / run.STORE_PROBE).write_text("{}")
    for p in (
        args.out_root / "dv_dataset" / "evil" / "labeling.json",
        args.train_dv_root / "evil" / "labeling.json",
    ):
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text('{"rows": []}')
    fake_file = create_autospec(hub.stage_hub_file)
    monkeypatch.setattr(hub, "stage_hub_file", fake_file)
    run.stage_shared(args, "tok")
    fake_file.assert_not_called()


# ---------------------------------------------------------------------------
# stage waiting
# ---------------------------------------------------------------------------


def test_wait_for_stage_returns_when_manifest_lands(tmp_path):
    args = _args(tmp_path)
    manifest = args.store_root / "evil_labeling" / run.STAGE_MANIFEST
    manifest.parent.mkdir(parents=True)
    manifest.write_text("{}")
    run.wait_for_stage("evil", args, [])  # returns immediately


def test_wait_for_stage_raises_when_stager_died(tmp_path):
    args = _args(tmp_path)
    with pytest.raises(RuntimeError, match="staging thread died"):
        run.wait_for_stage("evil", args, [ValueError("boom")])


def test_wait_for_stage_times_out(tmp_path):
    args = _args(tmp_path, stage_timeout_s=0)
    with pytest.raises(TimeoutError, match="staging did not complete"):
        run.wait_for_stage("evil", args, [])


# ---------------------------------------------------------------------------
# scoring + upload
# ---------------------------------------------------------------------------


def test_score_cmd_is_one_behavior_and_root_driven(tmp_path):
    """ONE behavior per subprocess; roots only — never a per-behavior override."""
    args = _args(tmp_path)
    cmd = run.score_cmd("sycophancy", args)
    assert cmd[1].endswith("issue1739_wcrung_arms.py")
    assert cmd[cmd.index("--behaviors") + 1] == "sycophancy"
    assert "evil" not in cmd, "a multi-behavior argv would trip the override guard"
    assert cmd[cmd.index("--store-root") + 1] == str(args.store_root)
    assert cmd[cmd.index("--train-dv-root") + 1] == str(args.train_dv_root)
    assert cmd[cmd.index("--out-root") + 1] == str(args.out_root)
    assert cmd[cmd.index("--n-layers") + 1] == "28"
    # the staged store IS the scorer's --store-root default, so no override
    assert "--wcrung-store" not in cmd
    assert "--wcrung-dv-json" not in cmd


def test_score_cmd_binds_the_scorer_cli(tmp_path):
    """The composed argv actually parses in the scorer (cross-file arity pin)."""
    from scripts import issue1739_wcrung_arms as wca

    args = _args(tmp_path)
    parsed = wca.parse_args(run.score_cmd("evil", args)[2:])
    assert parsed.behaviors == ["evil"]
    assert parsed.out_root == args.out_root
    assert parsed.store_root == args.store_root
    assert parsed.n_layers == 28


def test_score_behavior_passes_explicit_env_and_returns_rc(tmp_path, monkeypatch):
    args = _args(tmp_path)
    args.child_env = {"HF_TOKEN": "x"}
    seen: dict = {}

    def _capture(cmd, **kw):
        seen["cmd"] = list(cmd)
        seen["kw"] = kw
        return subprocess.CompletedProcess(args=cmd, returncode=2)

    monkeypatch.setattr(run.subprocess, "run", _capture)
    assert run.score_behavior("evil", args) == 2
    assert seen["kw"]["env"] is args.child_env  # explicit env passthrough
    assert seen["kw"]["check"] is False


def test_upload_behavior_fails_loud_on_missing_outputs(tmp_path):
    args = _args(tmp_path)
    with pytest.raises(FileNotFoundError, match="nothing to upload for evil"):
        run.upload_behavior("evil", args)


def test_upload_behavior_targets_the_wildchat_rung_arm_results_prefix(tmp_path, monkeypatch):
    args = _args(tmp_path)
    (args.out_root / "evil").mkdir(parents=True)
    (args.out_root / "evil" / "all_arms_spearman.json").write_text("{}")
    calls: list[tuple] = []
    fake = create_autospec(hub._upload)
    fake.side_effect = lambda *a, **k: calls.append((a, k)) or ""
    monkeypatch.setattr(hub, "_upload", fake)
    run.upload_behavior("evil", args)
    (a, k) = calls[0]
    assert a[0] == args.out_root / "evil"
    assert a[1] == hub.DEFAULT_DATASET_REPO
    assert a[2] == "dataset"
    assert a[3] == "issue1739_ctxmap/wildchat_rung/arm_results/evil"
    assert k["raise_on_error"] is True
    inspect.signature(hub._upload).bind(*a, **k)


# ---------------------------------------------------------------------------
# import-check
# ---------------------------------------------------------------------------


def test_import_check_exits_zero():
    with pytest.raises(SystemExit) as exc:
        run.main(["--import-check"])
    assert int(exc.value.code or 0) == 0


def test_no_judge_surface_in_the_runner():
    src = Path(run.__file__).read_text()
    for bad in ("judge_completions_batch", "judge_graded", "judge_items_graded", "dispatch_judge"):
        assert bad not in src, f"judge call surface {bad!r} present in the arm-scoring runner"


def test_sentinel_shape_is_documented_in_source():
    """The sentinel records the rung + the shared-store design for the poller."""
    src = Path(run.__file__).read_text()
    assert '"leg": "wcrung_arms"' in src
    assert '"rung": "wildchat_rung"' in src
    assert '"eval_store_shared_across_behaviors": True' in src
    assert "[phase=done]" in src


def test_all_hub_call_sites_bind_the_real_signatures():
    """AST sweep: every hub.<fn>(...) in the runner binds its live signature.

    Generalizes the stage_hub_prefix pin above to the whole file, so a future
    hub signature change (or a copied wrong-arity call) fails here rather than
    minutes into a staging phase.
    """
    import ast

    src = Path(run.__file__).read_text()
    tree = ast.parse(src)
    checked = 0
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        fn = node.func
        if not (isinstance(fn, ast.Attribute) and isinstance(fn.value, ast.Name)):
            continue
        if fn.value.id != "hub":
            continue
        target = getattr(hub, fn.attr, None)
        if target is None or not callable(target):
            continue
        # placeholders: only ARITY + keyword NAMES are under test here
        pos = [object() for _ in node.args]
        kws = {k.arg: object() for k in node.keywords if k.arg}
        if any(a for a in node.args if isinstance(a, ast.Starred)) or any(
            k.arg is None for k in node.keywords
        ):
            continue  # forwarded *args/**kwargs — nothing static to bind
        inspect.signature(target).bind(*pos, **kws)
        checked += 1
    assert checked >= 4, f"expected >=4 hub call sites bound, checked {checked}"


def test_reduced_layer_set_auto_enables_force_own_pool_frozen(tmp_path):
    """The wrapper cannot compose the crashing probe argv (committed-frozen +
    reduced layers): a non-full layer list auto-adds the escape flag."""
    args = _args(tmp_path, layers=[0, 1])
    cmd = run.score_cmd("evil", args)
    assert "--force-own-pool-frozen" in cmd
    assert cmd[cmd.index("--n-layers") + 1] == "2"


def test_full_layer_grid_keeps_committed_frozen(tmp_path):
    """At the full grid the committed indices are meaningful — no auto flag."""
    args = _args(tmp_path, layers=list(range(run.FULL_GRID_N_LAYERS)))
    cmd = run.score_cmd("evil", args)
    assert "--force-own-pool-frozen" not in cmd
    assert cmd[cmd.index("--n-layers") + 1] == "28"


def test_explicit_flag_forces_own_pool_even_at_full_width(tmp_path):
    args = _args(tmp_path, layers=list(range(28)), force_own_pool_frozen=True)
    assert "--force-own-pool-frozen" in run.score_cmd("evil", args)


# --- staged-slice regime coverage (probe-then-full-drive trap) --------------


def _slice_manifest(dest: Path, *, layers: list[int], kinds=run.KINDS) -> Path:
    dest.mkdir(parents=True, exist_ok=True)
    p = dest / run.STAGE_MANIFEST
    p.write_text(json.dumps({"layers": layers, "kinds": list(kinds)}))
    return p


def test_staged_slice_absent_manifest_is_a_fresh_stage(tmp_path):
    covered, why = run.staged_slice_covers(tmp_path / "evil_labeling", kinds=run.KINDS, layers=[0])
    assert covered is False
    assert why == "", "a missing manifest is a fresh stage, not a re-stage"


def test_staged_slice_exact_regime_is_covered(tmp_path):
    dest = tmp_path / "evil_labeling"
    _slice_manifest(dest, layers=list(range(28)))
    covered, why = run.staged_slice_covers(dest, kinds=run.KINDS, layers=list(range(28)))
    assert covered is True and why == ""


def test_staged_slice_wider_regime_covers_narrower_request(tmp_path):
    dest = tmp_path / "evil_labeling"
    _slice_manifest(dest, layers=list(range(28)))
    assert run.staged_slice_covers(dest, kinds=run.KINDS, layers=[0, 1])[0] is True


def test_narrow_probe_manifest_does_not_satisfy_the_full_drive(tmp_path):
    """THE TRAP: a 2-layer probe must not let the 28-layer drive skip staging."""
    dest = tmp_path / "evil_labeling"
    _slice_manifest(dest, layers=[0, 1])
    covered, why = run.staged_slice_covers(dest, kinds=run.KINDS, layers=list(range(28)))
    assert covered is False
    assert "narrower than requested" in why
    assert "26 layer(s) absent" in why


def test_staged_slice_missing_kind_is_not_covered(tmp_path):
    dest = tmp_path / "evil_labeling"
    _slice_manifest(dest, layers=list(range(28)), kinds=("t1",))
    covered, why = run.staged_slice_covers(dest, kinds=run.KINDS, layers=list(range(28)))
    assert covered is False and "kinds absent" in why


def test_staged_slice_corrupt_manifest_triggers_restage(tmp_path):
    dest = tmp_path / "evil_labeling"
    dest.mkdir(parents=True)
    (dest / run.STAGE_MANIFEST).write_text("{not json")
    covered, why = run.staged_slice_covers(dest, kinds=run.KINDS, layers=[0])
    assert covered is False and "unreadable" in why
