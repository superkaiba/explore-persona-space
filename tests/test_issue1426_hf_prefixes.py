"""#1426 HF prefix-threading + prior-lineage disjointness pins (plan §4.1).

The #1426 driver reuses the #928 fit modules, whose HF destinations were baked
as module-level constants derived from ``HF_PREFIX_928`` — the #1005 incident
(#1452): unthreaded uploads OVERWROTE the parent's Hub artifacts. These tests
pin the #1426 threading the same way the #1005 twin pinned its own, PLUS the
plan §4.1 requirement that every ``_1426`` prefix is DISJOINT from BOTH prior
lineages' roots (``issue1005_`` / ``issue928_`` — the copy-drift risk of the
mechanical-copy strategy):

1. The #928 modules' CLI defaults still resolve the EXACT #928 prefixes
   (parent standalone reproducibility unchanged).
2. Every #1426 profile constant mirrors its #928 sibling's shape one-for-one
   under the ``issue1426_cot_decomposition_r1llama`` root, and the full #1426
   prefix set is disjoint from the ``issue1005_``/``issue928_`` roots.
3. The #1426 driver's composed subprocess commands (f1 / mlp / figures) carry
   an r1-rooted value for EVERY Hub-path flag, contain NO #928 path anywhere,
   and parse cleanly through the parent modules' own parsers into the exact
   namespace fields the upload/stage call sites read.
4. The ACTUAL call sites consume the threaded values: ``main()`` of both #928
   fit modules runs end-to-end on the tiny synthetic fixture with only the
   Hub-upload boundary faked, and the recorded upload prefixes are the #1426
   ones; ``stage_store`` / ``stage_decomp`` fetch under a caller-passed
   prefix/path, never the baked #928 constants.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "scripts"))
sys.path.insert(0, str(REPO_ROOT / "src"))

R1 = "issue1426_cot_decomposition_r1llama"
P928 = "issue928_cot_decomposition"
P1005 = "issue1005_cot_decomposition_r1"


def _flag_values(cmd: list[str], flag: str) -> list[str]:
    return [cmd[i + 1] for i, tok in enumerate(cmd) if tok == flag]


def test_1426_constants_mirror_928_shapes_under_r1_root():
    import issue928_common as c928
    import issue928_mlp_indiv_control as mlp928
    import issue1426_common as c1426

    assert c928.HF_PREFIX_928 == P928
    assert c1426.HF_PREFIX_1426 == R1
    pairs = [
        (c928.RAW_COMPLETIONS_PREFIX, c1426.RAW_COMPLETIONS_PREFIX_1426),
        (c928.STORE_PREFIX, c1426.STORE_PREFIX_1426),
        (c928.FIT_RESULTS_PREFIX, c1426.FIT_RESULTS_PREFIX_1426),
        (c928.DECOMP_TENSORS_PREFIX, c1426.DECOMP_TENSORS_PREFIX_1426),
        (c928.FIGURES_PREFIX, c1426.FIGURES_PREFIX_1426),
        (mlp928.MLP_INDIV_TENSORS_PREFIX, c1426.MLP_INDIV_TENSORS_PREFIX_1426),
        (mlp928.MLP_INDIV_RESULTS_PREFIX, c1426.MLP_INDIV_RESULTS_PREFIX_1426),
        (mlp928.STORE_HF_PREFIX, c1426.STORE_HF_ROOT_1426),
        (mlp928.DECOMP_HF_PATH, c1426.DECOMP_INDIV_HF_PATH_1426),
    ]
    for parent, child in pairs:
        assert parent.startswith(P928 + "/"), parent
        assert child.startswith(R1 + "/"), child
        # identical relative shape: swap the root and the paths coincide.
        assert parent.removeprefix(P928) == child.removeprefix(R1), (parent, child)


def test_1426_prefixes_disjoint_from_both_prior_lineages():
    """Plan §4.1: all ``_1426`` prefixes are new — disjoint from the #1005 AND
    #928 lineage roots (the upload-prefix-clobber risk of a mechanical copy:
    a forgotten root swap would silently overwrite a prior lineage's Hub
    artifacts). Checks the full constant set in BOTH directions."""
    import issue1426_common as c1426

    assert c1426.HF_PREFIX_1426 == R1
    consts = {
        name: val
        for name, val in vars(c1426).items()
        if name.endswith("_1426") and isinstance(val, str)
    }
    assert len(consts) >= 9, sorted(consts)  # the 9 prefix/path constants + HF_PREFIX
    for name, val in consts.items():
        if name in ("STORE_REVISION_1426",):
            continue  # a Hub revision, not a path
        assert val == R1 or val.startswith(R1 + "/"), (name, val)
        for other_root in (P1005, P928):
            assert not val.startswith(other_root), (name, val, other_root)
            # the prior roots are not even PREFIXES of the 1426 root (no
            # sub-bucket collision in either direction).
            assert not other_root.startswith(val), (name, val, other_root)


def test_928_cli_defaults_keep_parent_prefixes():
    """Standalone #928 invocations (no override flags) keep their EXACT current
    Hub destinations — parent reproducibility is byte-unchanged."""
    import issue928_fit_decomposition as fit928
    import issue928_mlp_indiv_control as mlp928

    a = fit928.build_arg_parser().parse_args([])
    assert a.upload_prefix is None  # uploads stay gated off unless requested
    assert a.decomp_upload_prefix == f"{P928}/analysis_tensors/decomp"
    assert a.decomp_upload_prefix == fit928.DECOMP_TENSORS_PREFIX

    b = mlp928.build_arg_parser().parse_args([])
    assert b.results_upload_prefix == f"{P928}/fit_results/indiv_mlp_control"
    assert b.tensors_upload_prefix == f"{P928}/analysis_tensors/mlp_indiv"
    assert b.store_hf_prefix == f"{P928}/analysis_tensors/store"
    assert b.decomp_hf_path == f"{P928}/analysis_tensors/decomp/decomp_indiv.pt"
    assert b.store_revision == mlp928.STORE_REVISION
    assert (
        b.results_upload_prefix,
        b.tensors_upload_prefix,
        b.store_hf_prefix,
        b.decomp_hf_path,
    ) == (
        mlp928.MLP_INDIV_RESULTS_PREFIX,
        mlp928.MLP_INDIV_TENSORS_PREFIX,
        mlp928.STORE_HF_PREFIX,
        mlp928.DECOMP_HF_PATH,
    )


def test_1426_driver_cmds_route_every_hub_path_under_r1(tmp_path):
    """Every Hub-path flag on every fit-phase command the driver composes
    resolves under the r1 root (production AND smoke), no #928 path survives
    anywhere in any command, and the flags parse through the PARENT parsers
    into the exact namespace fields the upload/stage sites read."""
    import issue928_fit_decomposition as fit928
    import issue928_mlp_indiv_control as mlp928
    import issue1426_common as c1426
    import issue1426_run as run

    man = {
        "per_ctx_capture": {"c0": {"n_captured": 3}, "c1": {"n_captured": 5}},
        "context_ids": ["c0", "c1"],
        "capture_layers": [24, 25],
        "hidden_size": 8,
    }
    for smoke in (False, True):
        suffix = "_smoke" if smoke else ""
        f1 = run.f1_cmd(
            "store", tmp_path, layers=None, n_perms=3, n_boot=4, upload=True, smoke=smoke
        )
        mlp = run.mlp_cmd("store", tmp_path, tmp_path / "figs", man, upload=True, smoke=smoke)
        figs = run.figures_cmd(
            tmp_path, "store", tmp_path / "roll", tmp_path / "scratch", upload=True, smoke=smoke
        )
        for cmd, flags in [
            (f1, ["--upload-prefix", "--decomp-upload-prefix"]),
            (
                mlp,
                [
                    "--results-upload-prefix",
                    "--tensors-upload-prefix",
                    "--store-hf-prefix",
                    "--decomp-hf-path",
                ],
            ),
            (figs, ["--upload-prefix"]),
        ]:
            for flag in flags:
                vals = _flag_values(cmd, flag)
                assert vals, (flag, cmd)
                for v in vals:
                    assert v.startswith(R1 + "/"), (flag, v)
            assert not any(P928 in tok for tok in cmd), cmd

        a = fit928.build_arg_parser().parse_args(f1[2:])
        assert a.upload_prefix == c1426.FIT_RESULTS_PREFIX_1426 + suffix
        assert a.decomp_upload_prefix == c1426.DECOMP_TENSORS_PREFIX_1426 + suffix

        b = mlp928.build_arg_parser().parse_args(mlp[2:])
        assert b.results_upload_prefix == c1426.MLP_INDIV_RESULTS_PREFIX_1426 + suffix
        assert b.tensors_upload_prefix == c1426.MLP_INDIV_TENSORS_PREFIX_1426 + suffix
        assert b.store_hf_prefix == c1426.STORE_HF_ROOT_1426
        assert b.decomp_hf_path == c1426.DECOMP_INDIV_HF_PATH_1426
        assert b.store_revision == c1426.STORE_REVISION_1426 == "main"

    # upload disabled: the f1/figures cmds carry NO upload prefix at all (the
    # parent's `if args.upload_prefix:` gate stays off); mlp gates via flag.
    f1_off = run.f1_cmd(
        "store", tmp_path, layers=None, n_perms=None, n_boot=None, upload=False, smoke=False
    )
    assert "--upload-prefix" not in f1_off and "--decomp-upload-prefix" not in f1_off
    assert "--skip-upload" in run.mlp_cmd(
        "store", tmp_path, tmp_path, man, upload=False, smoke=False
    )


def test_mlp_main_upload_sites_consume_threaded_prefixes(tmp_path, monkeypatch):
    """END-TO-END reach proof (not a shadow-constant check): the mlp control's
    ``main()`` runs the REAL pipeline on the tiny synthetic fixture — only the
    Hub-upload boundary is faked with a signature-mirroring recorder — under
    #1426-profile override flags, and BOTH upload call sites pass the threaded
    prefixes (never the module-level #928 constants, which fails pre-fix)."""
    import issue928_mlp_indiv_control as drv
    import issue1426_common as c1426

    fix = drv.build_synth_fixture(tmp_path / "fix")
    real_fit = drv.fit_batched_loco_mlp_multihead
    monkeypatch.setattr(  # shrink epochs: the fixture pattern of the resume test
        drv,
        "fit_batched_loco_mlp_multihead",
        lambda *a, **k: real_fit(*a, **{**k, "max_epochs": 10}),
    )
    recorded: list[tuple[str, list[str]]] = []

    def _fake_upload(
        folder: Path,
        path_in_repo: str,
        expected_names: list[str],
        commit_message: str,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
    ) -> str:
        recorded.append((path_in_repo, sorted(expected_names)))
        return "https://fake/commit"

    monkeypatch.setattr(drv, "upload_folder_scoped_verify", _fake_upload)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue928_mlp_indiv_control.py",
            "--store",
            str(fix["store"]),
            "--decomp",
            str(fix["decomp"]),
            "--reference-bootstrap",
            str(fix["reference"]),
            "--out",
            str(tmp_path / "out"),
            "--figures-dir",
            str(tmp_path / "figures"),
            "--layers",
            "25",
            "--n-boot",
            "16",
            "--device",
            "cpu",
            "--chunk-size",
            "64",
            "--expect-rows",
            "24",
            "--expect-contexts",
            "4",
            "--expect-layers",
            "2",
            "--expect-hidden",
            "8",
            "--skip-parity-gate",
            "--results-upload-prefix",
            c1426.MLP_INDIV_RESULTS_PREFIX_1426,
            "--tensors-upload-prefix",
            c1426.MLP_INDIV_TENSORS_PREFIX_1426,
            "--store-hf-prefix",
            c1426.STORE_HF_ROOT_1426,
            "--decomp-hf-path",
            c1426.DECOMP_INDIV_HF_PATH_1426,
            "--store-revision",
            "main",
        ],
    )
    assert drv.main() == 0
    assert [p for p, _ in recorded] == [
        c1426.MLP_INDIV_RESULTS_PREFIX_1426,
        c1426.MLP_INDIV_TENSORS_PREFIX_1426,
    ]
    tensor_names = recorded[1][1]
    assert "decomp_indiv_mlp.pt" in tensor_names
    assert any(n.startswith("preds/preds_") for n in tensor_names), tensor_names


def test_f1_main_decomp_upload_site_consumes_threaded_prefix(tmp_path, monkeypatch):
    """Same reach proof for the f1 module: ``main()`` runs the real indiv fit
    on the tiny fixture store; the recorded decomp-tensor upload prefix is the
    #1426 override, not the baked #928 ``DECOMP_TENSORS_PREFIX`` (fails
    pre-fix, where the call site read the module constant)."""
    import issue928_fit_decomposition as fit928
    import issue928_mlp_indiv_control as drv
    import issue1426_common as c1426

    fix = drv.build_synth_fixture(tmp_path / "fix")
    recorded: list[tuple[str, list[str]]] = []

    def _fake_upload(
        folder: Path,
        path_in_repo: str,
        expected_names: list[str],
        commit_message: str,
        allow_patterns: list[str] | None = None,
        ignore_patterns: list[str] | None = None,
    ) -> str:
        recorded.append((path_in_repo, sorted(expected_names)))
        return "https://fake/commit"

    monkeypatch.setattr(fit928, "upload_folder_scoped_verify", _fake_upload)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue928_fit_decomposition.py",
            "--store",
            str(fix["store"]),
            "--out",
            str(tmp_path / "out"),
            "--regimes",
            "indiv",
            "--n-perms",
            "3",
            "--n-boot",
            "4",
            "--no-mlp",
            "--no-cross",
            "--device",
            "cpu",
            "--skip-parity-gate",
            "--upload-prefix",
            c1426.FIT_RESULTS_PREFIX_1426,
            "--decomp-upload-prefix",
            c1426.DECOMP_TENSORS_PREFIX_1426,
        ],
    )
    assert fit928.main() == 0
    assert [p for p, _ in recorded] == [
        c1426.FIT_RESULTS_PREFIX_1426,
        c1426.DECOMP_TENSORS_PREFIX_1426,
    ]
    assert any(n.startswith("decomp_") and n.endswith(".pt") for n in recorded[1][1])


def test_stage_functions_fetch_under_caller_passed_paths(tmp_path, monkeypatch):
    """The fallback stages consume the THREADED prefix/path (a #1426 fallback
    must never silently fetch the #928 parent store/decomp)."""
    from types import SimpleNamespace

    import huggingface_hub
    import issue928_mlp_indiv_control as drv

    custom_prefix = f"{R1}/analysis_tensors/store"
    seen: dict[str, str] = {}

    class _FakeApi:
        def list_repo_tree(self, repo_id, path_in_repo=None, repo_type=None, **_kw):
            seen["prefix"] = path_in_repo
            return [SimpleNamespace(path=f"{path_in_repo}/percq_summaries/manifest.json", size=1)]

    def _fake_download(repo_id, filename, repo_type=None, revision=None):
        seen["file"] = filename
        dest = tmp_path / "dl" / Path(filename).name
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text("{}")
        return str(dest)

    monkeypatch.setattr(huggingface_hub, "HfApi", _FakeApi)
    monkeypatch.setattr(huggingface_hub, "hf_hub_download", _fake_download)
    monkeypatch.setattr(drv.time, "sleep", lambda _s: None)
    drv.stage_store(tmp_path / "staged", "main", custom_prefix)
    assert seen["prefix"] == custom_prefix

    custom_path = f"{R1}/analysis_tensors/decomp/decomp_indiv.pt"
    drv.stage_decomp(tmp_path / "staged_decomp" / "decomp_indiv.pt", "main", custom_path)
    assert seen["file"] == custom_path


# ── sampled-rollout robustness round (amendment plan v4 §4.2 items 3/7 + C) ───


def _path_disjoint(a: str, b: str) -> bool:
    """Neither prefix contains the other at a path boundary (both directions)."""
    return not (a == b or a.startswith(b + "/") or b.startswith(a + "/"))


def test_sampled_rollout_prefix_under_r1_root_and_disjoint():
    """Plan §4.2 item 7: ``SAMPLED_ROLLOUT_PREFIX_1426`` sits under the 1426
    root and is disjoint (BOTH prefix directions) from every primary leaf
    prefix (raw_completions/, analysis_tensors/, fit_results/, figures/) and
    from the ``issue1005_``/``issue928_`` lineage roots."""
    import issue1426_common as c1426

    sampled = c1426.SAMPLED_ROLLOUT_PREFIX_1426
    assert sampled == f"{R1}/sampled_rollout"
    assert sampled.startswith(R1 + "/")
    primary_leaves = [
        c1426.RAW_COMPLETIONS_PREFIX_1426,
        c1426.STORE_PREFIX_1426,
        c1426.STORE_HF_ROOT_1426,
        c1426.FIT_RESULTS_PREFIX_1426,
        c1426.DECOMP_TENSORS_PREFIX_1426,
        c1426.DECOMP_INDIV_HF_PATH_1426,
        c1426.FIGURES_PREFIX_1426,
        c1426.MLP_INDIV_TENSORS_PREFIX_1426,
        c1426.MLP_INDIV_RESULTS_PREFIX_1426,
    ]
    for leaf in primary_leaves:
        assert _path_disjoint(sampled, leaf), (sampled, leaf)
    for other_root in (P1005, P928):
        assert not sampled.startswith(other_root), sampled
        assert not other_root.startswith(sampled), sampled


def test_rooted_prefix_composes_per_seed_layout():
    """Plan §4.3 layout: re-rooting the two extract-phase prefixes at
    ``sampled_rollout/seed<s>`` yields exactly the per-seed subtree paths;
    ``None`` is byte-identical passthrough (default-preserving); the composed
    per-seed prefixes stay disjoint from the primary leaves."""
    import issue1426_common as c1426
    import issue1426_run as run

    for s in (42, 137):
        root = f"{c1426.SAMPLED_ROLLOUT_PREFIX_1426}/seed{s}"
        rolled = run.rooted_prefix(c1426.RAW_COMPLETIONS_PREFIX_1426, root)
        stored = run.rooted_prefix(c1426.STORE_PREFIX_1426, root)
        assert rolled == f"{R1}/sampled_rollout/seed{s}/raw_completions/thinking_rollouts"
        assert stored == f"{R1}/sampled_rollout/seed{s}/analysis_tensors/store/percq_summaries"
        for composed in (rolled, stored, f"{root}/fit_results"):
            assert _path_disjoint(composed, c1426.RAW_COMPLETIONS_PREFIX_1426)
            assert _path_disjoint(composed, c1426.STORE_PREFIX_1426)
            assert _path_disjoint(composed, c1426.FIT_RESULTS_PREFIX_1426)
    assert (
        run.rooted_prefix(c1426.RAW_COMPLETIONS_PREFIX_1426, None)
        == c1426.RAW_COMPLETIONS_PREFIX_1426
    )
    assert run.rooted_prefix(c1426.STORE_PREFIX_1426, "") == c1426.STORE_PREFIX_1426


def test_f1_cmd_no_mlp_passthrough(tmp_path):
    """Plan §4.2 item 4: ``no_mlp=True`` appends the fit module's existing
    ``--no-mlp`` flag; the default command stays byte-identical; the smoke
    command keeps its pre-existing ``--no-mlp`` exactly once."""
    import issue928_fit_decomposition as fit928
    import issue1426_run as run

    base = run.f1_cmd(
        "store", tmp_path, layers=None, n_perms=None, n_boot=None, upload=False, smoke=False
    )
    with_flag = run.f1_cmd(
        "store",
        tmp_path,
        layers=None,
        n_perms=None,
        n_boot=None,
        upload=False,
        smoke=False,
        no_mlp=True,
    )
    assert "--no-mlp" not in base
    assert with_flag == [*base, "--no-mlp"]
    smoke_cmd = run.f1_cmd(
        "store", tmp_path, layers=None, n_perms=None, n_boot=None, upload=False, smoke=True
    )
    assert smoke_cmd.count("--no-mlp") == 1
    # the flag parses through the PARENT parser (it is a real fit-module flag).
    a = fit928.build_arg_parser().parse_args(with_flag[2:])
    assert a.no_mlp is True


def test_sampled_manifest_validation_wrong_seed_fails(tmp_path):
    """Critic addition C: the invocation-B provenance guard is never vacuous —
    a fixture manifest with the WRONG gen_seed (and one with the field
    MISSING, and one at the wrong rung) exits non-zero through the exact
    load-then-validate path the dispatch heredoc runs; the matching manifest
    passes and reports the realized cap + code SHA."""
    import json

    import pytest
    from issue1426_common import validate_sampled_store_manifest

    def _write_and_load(man: dict) -> dict:
        p = tmp_path / "store" / "manifest.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(man))
        return json.loads(p.read_text())

    good = {
        "rung": "sample",
        "gen_seed": 137,
        "max_new_tokens": 8192,
        "reproducibility": {"git_commit": "deadbeef"},
    }
    report = validate_sampled_store_manifest(_write_and_load(good), expected_seed=137)
    assert report == {
        "rung": "sample",
        "gen_seed": 137,
        "production_max_new_tokens": 8192,
        "code_sha": "deadbeef",
    }
    # the gate's C-remeasure may legitimately raise the cap to 16,384 (plan §7)
    validate_sampled_store_manifest(
        _write_and_load({**good, "max_new_tokens": 16384}), expected_seed=137
    )

    for bad in (
        {**good, "gen_seed": 42},  # wrong seed (the cross-seed reuse trap)
        {k: v for k, v in good.items() if k != "gen_seed"},  # field missing
        {**good, "rung": "greedy"},  # wrong rung
        {**good, "max_new_tokens": 4096},  # alien cap
    ):
        with pytest.raises(SystemExit) as exc:
            validate_sampled_store_manifest(_write_and_load(bad), expected_seed=137)
        assert exc.value.code  # non-zero shell exit through the dispatch heredoc


def _write_gate_slice(out_dir, contexts_rows: dict[str, list[tuple[str, str]]]) -> None:
    """Fixture Gate-1 slice: gate_report.json + one rollout blob per context."""
    import json

    rollouts = out_dir / "raw_completions" / "thinking_rollouts"
    rollouts.mkdir(parents=True, exist_ok=True)
    (out_dir / "gate_report.json").write_text(
        json.dumps({"chosen_rung": "sample", "gate_contexts": sorted(contexts_rows)})
    )
    for c, rows in contexts_rows.items():
        (rollouts / f"{c}.json").write_text(
            json.dumps(
                {
                    "context_id": c,
                    "completions": [{"probe": p, "completion": t} for p, t in rows],
                }
            )
        )


def test_gate_slice_status_branches(tmp_path):
    """Review-1 minor 3: the monitor's readiness predicate branches — empty
    dir → pending; null chosen_rung → gate_failed; report present but a gate
    context's blob missing → pending; complete slice → ready."""
    import json

    from issue1426_common import gate_slice_status

    empty = tmp_path / "empty"
    empty.mkdir()
    assert gate_slice_status(empty) == "pending"

    failed = tmp_path / "gatefail"
    failed.mkdir()
    (failed / "gate_report.json").write_text(json.dumps({"chosen_rung": None, "gate_contexts": []}))
    assert gate_slice_status(failed) == "gate_failed"

    partial = tmp_path / "partial"
    _write_gate_slice(partial, {"c0": [("p0", "t0")]})
    report = json.loads((partial / "gate_report.json").read_text())
    report["gate_contexts"] = ["c0", "c_missing"]
    (partial / "gate_report.json").write_text(json.dumps(report))
    assert gate_slice_status(partial) == "pending"

    ready = tmp_path / "ready"
    _write_gate_slice(ready, {"c0": [("p0", "t0")], "c1": [("p0", "u0")]})
    assert gate_slice_status(ready) == "ready"


def test_check_seed_differentiation_branches(tmp_path):
    """Review-1 minor 3: identical slices FAIL (SystemExit, non-zero code),
    differing slices PASS with a report dict, zero shared rows FAIL, an
    exactly-at-threshold fraction FAILs, and the threshold is the
    single-sourced ``IDENTICAL_FRAC_MAX`` (0.5)."""
    import pytest
    from issue1426_common import IDENTICAL_FRAC_MAX, check_seed_differentiation

    a = tmp_path / "seed_a"
    b_same = tmp_path / "seed_b_same"
    rows = {"c0": [("p0", "same text"), ("p1", "same too")]}
    _write_gate_slice(a, rows)
    _write_gate_slice(b_same, rows)
    with pytest.raises(SystemExit) as exc:
        check_seed_differentiation(a, b_same)
    assert exc.value.code  # byte-identical slices → non-zero exit

    b_diff = tmp_path / "seed_b_diff"
    _write_gate_slice(b_diff, {"c0": [("p0", "OTHER text"), ("p1", "other too")]})
    report = check_seed_differentiation(a, b_diff)
    assert report["identical_fraction"] == 0.0
    assert report["n_shared_rows"] == 2
    assert report["identical_frac_max"] == IDENTICAL_FRAC_MAX == 0.5

    b_disjoint = tmp_path / "seed_b_disjoint"
    _write_gate_slice(b_disjoint, {"c9": [("p9", "elsewhere")]})
    with pytest.raises(SystemExit) as exc:
        check_seed_differentiation(a, b_disjoint)
    assert exc.value.code  # zero shared rows → never a silent pass

    # exactly at the threshold: 1 of 2 rows identical (0.5 >= 0.5) → FAIL
    b_half = tmp_path / "seed_b_half"
    _write_gate_slice(b_half, {"c0": [("p0", "same text"), ("p1", "different")]})
    with pytest.raises(SystemExit):
        check_seed_differentiation(a, b_half)
