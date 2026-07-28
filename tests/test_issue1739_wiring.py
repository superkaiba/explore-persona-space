"""Round-2 wiring tests for issue #1739 (review C1/C2/C3 fixes).

Covers: sentinel conformance against poll_pipeline's OWN constants, the
results-payload key contract, the plan-grid RunSpec composition, upload-call
sequencing with the Hub boundary stubbed (autospec — signature-conformant by
construction), the results-git push-verify + artifact-presence assert on a
real tmp git repo, the arms-15/16 feature injection loader, and the E2/E2p
row-level extraction against the tested tensor-form reference. No network,
no GPU; every fixture is neutral synthetic text.
"""

from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO_ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# sentinel conformance (C1) — pinned against poll_pipeline's OWN constants
# ---------------------------------------------------------------------------


def test_sentinel_conformance(tmp_path):
    poll = _load_script("poll_pipeline")
    from explore_persona_space.experiments.issue_1739 import sentinels

    p1 = sentinels.write_phase_sentinel(tmp_path, "extract", status="ok", rc=0)
    body = json.loads(p1.read_text())
    for key in poll._SENTINEL_REQUIRED_KEYS:
        assert key in body, key
    assert body["sentinel_schema_version"] == poll.SENTINEL_SCHEMA_VERSION_SUPPORTED
    assert body["kind"] == "epm:progress" and body["version"] == 1
    assert p1.name.startswith("issue-1739-") and p1.suffix == ".json"

    payload = {k: None for k in sentinels.RESULTS_PAYLOAD_KEYS}
    payload["eval_numbers"] = {}
    p2 = sentinels.write_results_sentinel(tmp_path, payload, smoke=False)
    body2 = json.loads(p2.read_text())
    for key in poll._SENTINEL_REQUIRED_KEYS:
        assert key in body2, key
    assert body2["kind"] == "epm:results"
    note = json.loads(body2["note"])
    for key in poll._RESULTS_PAYLOAD_KEYS:
        assert key in note, key
    p3 = sentinels.write_results_sentinel(tmp_path, payload, smoke=True)
    assert json.loads(p3.read_text())["kind"] == "epm:smoke-result"


def test_compose_results_payload_no_training_card(tmp_path):
    from explore_persona_space.experiments.issue_1739 import sentinels

    summary_dir = tmp_path / "evil" / "arm_results"
    summary_dir.mkdir(parents=True)
    (summary_dir / "all_arms_spearman.json").write_text(
        json.dumps(
            {
                "n_cells": 2,
                "arm_rows": [
                    {
                        "arm": "arm6_map_proj_e1",
                        "regime": "e1",
                        "variant": "context_end",
                        "rho_frozen": 0.4,
                    }
                ],
                "headlines": [{"delta_rho_frozen": 0.1}],
                "nulls": [{"p_max_over_arms": 0.02}],
            }
        )
    )
    payload = sentinels.compose_results_payload(
        tmp_path, ["evil", "missingb"], hf_prefix="issue1739_ctxmap"
    )
    for key in sentinels.RESULTS_PAYLOAD_KEYS:
        assert key in payload, key
    card = payload["reproducibility_card"]
    assert card["adapter_paths"] == [] and card["wandb_project"] == "issue1739"
    assert "no WandB runs" in card["wandb_note"]
    assert payload["eval_numbers"]["missingb"] == {"error": "summary missing"}
    assert payload["eval_numbers"]["evil"]["n_cells"] == 2


# ---------------------------------------------------------------------------
# plan-grid composition (C2)
# ---------------------------------------------------------------------------


def test_compose_run_specs_covers_full_plan_grid():
    fits_cli = _load_script("issue1739_fits")
    specs = fits_cli.compose_run_specs(
        variants=("context_end", "prefix_end"),
        regimes=("e1", "e2", "e2p"),
        u_sizes=(250, 5000, None),
        budgets=(250, 2500, 8000),
        draws=(0, 1, 2, 3, 4),
        seeds=(0, 1, 2),
        compose=True,
        compose_u_size=5000,
        f_u_grid=(0.0, 0.5),
        f_l_grid=(0.0, 1.0),
    )
    base = [s for s in specs if s.f_u is None]
    comp = [s for s in specs if s.f_u is not None]
    # Base grid: 2 variants x 3 U rungs x 3 regimes, each with the FULL
    # budgets x draws x seeds block threaded to run_grid.
    assert len(base) == 2 * 3 * 3
    for s in base:
        assert s.budgets == (250, 2500, 8000)
        assert s.draws == (0, 1, 2, 3, 4)
        assert s.seeds == (0, 1, 2)
    assert {s.regime for s in base} == {"e1", "e2", "e2p"}
    assert {s.u_size for s in base} == {250, 5000, None}
    assert {s.variant for s in specs} == {"context_end", "prefix_end"}
    # Composition: dedup {(0,0),(0.5,0),(0.5,1)} x 3 L-anchors x 2 variants;
    # anchor cells run draw 0 / seed 0 (the deterministic reference cell).
    assert len(comp) == 3 * 3 * 2
    assert {(s.f_u, s.f_l) for s in comp} == {(0.0, 0.0), (0.5, 0.0), (0.5, 1.0)}
    assert all(s.draws == (0,) and s.seeds == (0,) and len(s.budgets) == 1 for s in comp)
    assert {s.budgets[0] for s in comp} == {250, 2500, 8000}


def test_dispatcher_composes_full_grid_and_sequences_uploads():
    """Textual pins on the dispatcher: phase ORDER (upload_raw strictly before
    judge; results last) + the production fits grid literals (C2)."""
    text = (REPO_ROOT / "scripts" / "issue1739_dispatch.sh").read_text()
    phases_line = next(line for line in text.split("\n") if line.startswith("PHASES=("))
    order = phases_line.split("(")[1].split(")")[0].split()
    assert order.index("upload_raw") < order.index("judge"), order
    assert order.index("extract") < order.index("upload_raw"), order
    assert order[-1] == "results", order
    for literal in (
        'u_sizes="250 5000 full"',
        'draws="0 1 2 3 4"',
        'seeds="0 1 2"',
        'echo "250 2500 8000"',
        'echo "250 2500 16000"',
        "e1 e2 e2p",
        "--compose",
        "--config config_b",
        "--text-emb",
        "--text-features",
        "[phase=done]",
    ):
        assert literal in text, literal


# ---------------------------------------------------------------------------
# upload stages (C1) — Hub boundary stubbed with autospec
# ---------------------------------------------------------------------------


def test_upload_tree_one_bulk_commit_and_verify(tmp_path, monkeypatch):
    up = _load_script("issue1739_upload")
    from explore_persona_space.orchestrate import hub

    root = tmp_path / "raw"
    (root / "labeling" / "evil").mkdir(parents=True)
    (root / "labeling" / "evil" / "c0_seed0.json").write_text("{}")
    (root / "extraction" / "evil").mkdir(parents=True)
    (root / "extraction" / "evil" / "p0.json").write_text("{}")

    upload_mock = mock.create_autospec(hub._upload, return_value="https://hf.co/x")
    verify_mock = mock.create_autospec(hub.verify_repo_paths_uploaded, return_value=[])
    monkeypatch.setattr(hub, "_upload", upload_mock)
    monkeypatch.setattr(hub, "verify_repo_paths_uploaded", verify_mock)
    rel = up.upload_tree(root, "issue1739_ctxmap/raw_completions", dry_run=False, what="raw")
    assert len(rel) == 2
    assert upload_mock.call_count == 1  # ONE bulk upload_folder commit, never per-file
    args, kwargs = upload_mock.call_args
    assert args[0] == root and kwargs["path_in_repo"] == "issue1739_ctxmap/raw_completions"
    expected = verify_mock.call_args.args[2]
    assert sorted(expected) == [
        "issue1739_ctxmap/raw_completions/extraction/evil/p0.json",
        "issue1739_ctxmap/raw_completions/labeling/evil/c0_seed0.json",
    ]

    # Failure modes fail LOUD: empty upload return + missing-on-Hub verify.
    upload_mock.return_value = ""
    with pytest.raises(SystemExit, match="no path"):
        up.upload_tree(root, "p", dry_run=False, what="raw")
    upload_mock.return_value = "https://hf.co/x"
    verify_mock.return_value = ["p/missing.json"]
    with pytest.raises(SystemExit, match="missing on the Hub"):
        up.upload_tree(root, "p", dry_run=False, what="raw")


def test_upload_tree_dry_run_never_touches_hub(tmp_path, monkeypatch):
    up = _load_script("issue1739_upload")
    from explore_persona_space.orchestrate import hub

    root = tmp_path / "raw"
    root.mkdir()
    (root / "a.json").write_text("{}")
    upload_mock = mock.create_autospec(hub._upload)
    monkeypatch.setattr(hub, "_upload", upload_mock)
    rel = up.upload_tree(root, "p", dry_run=True, what="raw")
    assert rel == ["a.json"] and upload_mock.call_count == 0


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(
        ["git", "-C", str(repo), *args], capture_output=True, text=True, check=check
    )


def test_results_git_stage_push_verify_and_artifact_assert(tmp_path, monkeypatch):
    up = _load_script("issue1739_upload")
    origin = tmp_path / "origin.git"
    subprocess.run(["git", "init", "--bare", str(origin)], check=True, capture_output=True)
    repo = tmp_path / "repo"
    repo.mkdir()
    _git(repo, "init", "-b", "issue-1739")
    _git(repo, "config", "user.email", "t@t")
    _git(repo, "config", "user.name", "t")
    _git(repo, "remote", "add", "origin", str(origin))
    (repo / ".gitignore").write_text("*.npz\n")
    (repo / "seed.txt").write_text("seed")
    _git(repo, "add", "seed.txt", ".gitignore")
    _git(repo, "commit", "-m", "seed")
    _git(repo, "push", "origin", "HEAD:issue-1739")

    results = repo / "eval_results" / "issue_1739"
    (results / "evil" / "arm_results").mkdir(parents=True)
    (results / "evil" / "arm_results" / "all_arms_spearman.json").write_text("{}")
    (results / "dv_dataset" / "evil").mkdir(parents=True)
    (results / "dv_dataset" / "evil" / "labeling.json").write_text("{}")
    figures = repo / "figures" / "issue_1739"
    (figures / "evil").mkdir(parents=True)
    (figures / "evil" / "hero.png").write_text("png")

    monkeypatch.setattr(up, "_REPO_ROOT", repo)
    monkeypatch.chdir(repo)
    args = up._parse_args(
        [
            "--stage",
            "results-git",
            "--results-root",
            "eval_results/issue_1739",
            "--figures-root",
            "figures/issue_1739",
        ]
    )
    assert up.results_git_stage(args) == 0
    ls = _git(repo, "ls-tree", "-r", "origin/issue-1739", "--name-only").stdout
    assert "eval_results/issue_1739/evil/arm_results/all_arms_spearman.json" in ls
    assert "figures/issue_1739/evil/hero.png" in ls
    # Idempotent re-run: nothing new to commit, still rc 0.
    assert up.results_git_stage(args) == 0
    # Broken push must FAIL LOUD (never done-with-unpushed-results).
    _git(repo, "remote", "set-url", "origin", str(tmp_path / "nonexistent.git"))
    (results / "dv_dataset" / "evil" / "labeling.json").write_text('{"v": 2}')
    with pytest.raises(SystemExit, match="push verification FAILED"):
        up.results_git_stage(args)


def test_results_git_stage_refuses_empty_declared_set(tmp_path, monkeypatch):
    up = _load_script("issue1739_upload")
    monkeypatch.chdir(tmp_path)
    args = up._parse_args(
        ["--stage", "results-git", "--results-root", "nope", "--figures-root", "nofigs"]
    )
    with pytest.raises(SystemExit, match="ZERO declared result files"):
        up.results_git_stage(args)


# ---------------------------------------------------------------------------
# arms-15/16 feature injection + E2/E2p row-level extraction (C3)
# ---------------------------------------------------------------------------


def test_features_builder_and_injection_loader(tmp_path):
    feats_cli = _load_script("issue1739_features")
    fits_cli = _load_script("issue1739_fits")
    ctx = tmp_path / "b_train_main.contexts.jsonl"
    rows = [
        {"context_id": f"c{i}", "prefix_text": f"synthetic prefix {i}", "query": f"question {i}?"}
        for i in range(6)
    ]
    ctx.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    roll = tmp_path / "rollouts"
    roll.mkdir()
    (roll / "c0_seed0.json").write_text(json.dumps({"context_id": "c0", "completion": "abcdef"}))
    out = tmp_path / "b.npz"
    feats_cli.build_features([ctx], roll, out)
    with np.load(out) as z:
        assert list(z["context_ids"]) == [f"c{i}" for i in range(6)]
        assert z["emb"].shape == (6, feats_cli.HASH_DIM)
        assert z["features"].shape[0] == 6
        assert z["features"][0][-1] == 6.0  # mean response length for c0
    emb = fits_cli._load_injected_features(out, "emb", [f"c{i}" for i in range(6)], "--text-emb")
    assert emb.shape == (6, feats_cli.HASH_DIM)
    with pytest.raises(RuntimeError, match="missing from"):
        fits_cli._load_injected_features(out, "emb", ["c0", "cMISSING"], "--text-emb")


def test_extract_rb_e2_row_level_matches_tensor_reference():
    """The CLI's row-level E2/E2p (flat per-rollout store rows + (ctx, k)
    index) must equal extract_rb_matched on the equivalent 4-D tensor."""
    fits_cli = _load_script("issue1739_fits")
    from explore_persona_space.experiments.issue_1739 import fits

    rng = np.random.default_rng(7)
    n_ctx, k, n_layers, d = 8, 5, 2, 6
    acts = rng.normal(size=(n_ctx, k, n_layers, d))
    direction = rng.normal(size=(n_layers, d))
    scores = rng.uniform(0, 100, size=(n_ctx, k))
    scores[rng.random(size=(n_ctx, k)) < 0.15] = np.nan
    acts += np.where(np.nan_to_num(scores, nan=0.0) >= 50, 1.0, 0.0)[:, :, None, None] * direction

    # Flatten to the store-row shape (shuffled row order — the CLI must align
    # through the (ctx, k) index, never through row order).
    order = rng.permutation(n_ctx * k)
    row_ctx = np.repeat(np.arange(n_ctx), k)[order]
    row_k = np.tile(np.arange(k), n_ctx)[order]
    ans_rows = {ly: acts[row_ctx, row_k, ly, :] for ly in range(n_layers)}

    tbl = fits_cli.LabeledTable(
        z_by_variant={},
        z_ans=np.zeros((n_layers, n_ctx, d)),
        dv=np.zeros(n_ctx),
        groups=[f"g{i}" for i in range(n_ctx)],
        per_rollout=scores,
        ctx_order=[f"c{i}" for i in range(n_ctx)],
        rungs=["main"],
        ans_rows=ans_rows,
        ans_row_ctx=row_ctx,
        ans_row_k=row_k,
    )

    class _Args:
        behavior = "synthetic"
        e1_store = None

    for regime, pooled in (("e2", False), ("e2p", True)):
        rb_cli = fits_cli._extract_rb(regime, _Args(), tbl, list(range(n_layers)), d)
        rb_ref, _ = fits.extract_rb_matched(acts, scores, spread_min=15.0, pooled=pooled)
        assert np.allclose(rb_cli, rb_ref, atol=1e-10), regime
        cos = float(
            (rb_cli * direction).sum() / (np.linalg.norm(rb_cli) * np.linalg.norm(direction))
        )
        assert cos > 0.6, (regime, cos)  # recovers the planted direction

    # A no-per-rollout table must FAIL LOUD for e2/e2p (never an E1 relabel).
    tbl_bare = fits_cli.LabeledTable(
        z_by_variant={},
        z_ans=np.zeros((n_layers, n_ctx, d)),
        dv=np.zeros(n_ctx),
        groups=[],
        per_rollout=None,
        ctx_order=[],
        rungs=[],
        ans_rows=None,
        ans_row_ctx=None,
        ans_row_k=None,
    )
    with pytest.raises(SystemExit, match="per-rollout"):
        fits_cli._extract_rb("e2", _Args(), tbl_bare, list(range(n_layers)), d)


# ---------------------------------------------------------------------------
# round-3 wiring: pilot gate + transfer leg + maps persistence/upload + figures
# ---------------------------------------------------------------------------


def test_dispatcher_round3_pilot_transfer_mapdiag_pins():
    """Textual pins on the round-3 dispatcher wiring: the fits phase runs the
    §9 pilot gate (rc-7 designed halt) BEFORE the main grid, the config_a leg
    carries --transfer (+ the smoke min-n calibration), and the figures phase
    passes --map-diag."""
    text = (REPO_ROOT / "scripts" / "issue1739_dispatch.sh").read_text()
    for literal in (
        "--transfer",
        "--transfer-min-n 2",
        "--pilot --plan-wall-h",
        'if [ "$pilot_rc" -eq 7 ]; then',
        "--map-diag",
        "pilot_report.json",
    ):
        assert literal in text, literal
    # ordering: the pilot invocation precedes the main fits invocation
    pilot_at = text.index("--pilot --plan-wall-h")
    main_at = text.index('uv run python scripts/issue1739_fits.py "${FITS_ARGS[@]}"\n', pilot_at)
    assert pilot_at < main_at
    # the Config-B secondary leg stays a within-split LOFO (no --transfer)
    config_b_at = text.index("--config config_b")
    assert "--transfer" not in text[config_b_at : config_b_at + 200]


def test_saved_maps_ride_the_tensors_upload_stage(tmp_path, monkeypatch):
    """C-1 sequencing: a _save_map artifact under tensors_root/maps/ lands in
    the tensors stage's ONE bulk upload + exact-set verify (Hub stubbed with
    autospec — signature-conformant by construction)."""
    up = _load_script("issue1739_upload")
    fits_cli = _load_script("issue1739_fits")
    from explore_persona_space.experiments.issue_1739 import fits
    from explore_persona_space.orchestrate import hub

    rng = np.random.default_rng(11)
    x = rng.normal(size=(1, 30, 4))
    mapfit = fits.fit_linear_map(x, 0.5 * x)
    tensors_root = tmp_path / "analysis_tensors"
    saved = fits_cli._save_map(tensors_root, "context_end", "5000", mapfit, [0])
    assert saved.exists()

    upload_mock = mock.create_autospec(hub._upload, return_value="https://hf.co/x")
    verify_mock = mock.create_autospec(hub.verify_repo_paths_uploaded, return_value=[])
    monkeypatch.setattr(hub, "_upload", upload_mock)
    monkeypatch.setattr(hub, "verify_repo_paths_uploaded", verify_mock)
    rel = up.upload_tree(
        tensors_root, "issue1739_ctxmap/analysis_tensors", dry_run=False, what="tensors"
    )
    assert rel == ["maps/context_end__u5000.npz"]
    assert upload_mock.call_count == 1  # ONE bulk upload_folder commit
    expected = verify_mock.call_args.args[2]
    assert expected == ["issue1739_ctxmap/analysis_tensors/maps/context_end__u5000.npz"]


def test_figures_cli_map_diag_adapter_pools_fits_format():
    """map_diag_rows adapts the fits CLI's {'variant|u': diagnostics} format
    (str acc@1 keys after a JSON roundtrip) to fig_map_degradation rows."""
    fig_cli = _load_script("issue1739_figures")
    diag = {
        "context_end|full": {
            "per_layer": [
                {
                    "layer_idx": 0,
                    "r2_map": 0.6,
                    "r2_identity_bias": 0.1,
                    "knn": {
                        "euclidean": {"acc_at_k": {"1": 0.8}, "chance_at_k": {"1": 0.05}},
                        "cosine": {"acc_at_k": {"1": 0.7}, "chance_at_k": {"1": 0.05}},
                    },
                },
                {
                    "layer_idx": 1,
                    "r2_map": 0.4,
                    "r2_identity_bias": 0.3,
                    "knn": {
                        "euclidean": {"acc_at_k": {"1": 0.6}, "chance_at_k": {"1": 0.05}},
                        "cosine": {"acc_at_k": {"1": 0.5}, "chance_at_k": {"1": 0.05}},
                    },
                },
            ]
        }
    }
    rows = fig_cli.map_diag_rows(json.loads(json.dumps(diag)))
    assert rows == [
        {
            "rung": "context_end|full",
            "r2_map": pytest.approx(0.5),
            "r2_identity_bias": pytest.approx(0.2),
            "knn_acc1_euclidean": pytest.approx(0.7),
            "knn_chance1": pytest.approx(0.05),
        }
    ]
    # already-pooled list payloads pass through unchanged
    assert fig_cli.map_diag_rows([{"rung": "x"}]) == [{"rung": "x"}]
