"""Tiny-real tests for the #1739 wildchat-rung arm-scoring entrypoint.

The e2e tests drive the PRODUCTION path of ``scripts/issue1739_wcrung_arms.py``
end to end on CPU with real library types at every internal seam (real
``store_io`` stores on disk, real ``_load_labeled``, real whitening + map fit,
real ``arms.run_transfer_cell`` / ``evaluate_transfer``) — nothing is stubbed;
only the SCALE is tiny (2 layers, dim 6, ~20 contexts). The centerpiece is the
rung's structural invariant: ONE shared capture store scored under THREE
per-behavior DVs (generate-once/judge-3x), which must yield three independent
per-behavior reads off the same activations. The remaining tests pin the three
safety rails, the frozen-layer rule, and the store/DV path resolution.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts import issue1739_wcrung_arms as wca  # noqa: E402

DIM = 6
LAYERS = (0, 1)
ROSTER = ["arm1_ctx_e1", "arm3_identity_bias", "arm4_ridge_ctx", "arm6_map_proj_e1"]
BEHAVIORS = ("evil", "sycophancy", "hallucination")


def _write_store(root: Path, rows: list[dict], arrays: dict[str, np.ndarray]) -> Path:
    """Write a canonical capture store: {kind}_L{ly:02d}.npy + row_index.jsonl."""
    root.mkdir(parents=True, exist_ok=True)
    for kind, arr in arrays.items():
        for li, ly in enumerate(LAYERS):
            np.save(root / f"{kind}_L{ly:02d}.npy", np.asarray(arr[li], dtype=np.float16))
    with (root / "row_index.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    return root


def _acts(rng: np.random.Generator, n: int, signal: np.ndarray | None = None) -> np.ndarray:
    """(n_layers, n, DIM) activations, optionally with a per-row signal in dim 0."""
    a = rng.normal(size=(len(LAYERS), n, DIM)) * 0.5
    if signal is not None:
        a[:, :, 0] += signal[None, :]
    return a


def _dv_payload(ctx_ids: list[str], dv: np.ndarray, *, split: str, rung: str, groups) -> str:
    return json.dumps(
        {
            "rows": [
                {
                    "context_id": c,
                    "dv": float(dv[i]),
                    "split": split,
                    "rung": rung,
                    "group_key": groups(i, c),
                    "per_rollout_scores": {"k0": float(dv[i])},
                }
                for i, c in enumerate(ctx_ids)
            ]
        }
    )


@pytest.fixture
def rig(tmp_path: Path):
    """Tiny-real inputs, PRODUCTION layout: root-driven per-behavior train
    inputs + ONE shared wildchat-rung store + three per-behavior rung DVs.

    Every path is the driver's own default under ``--store-root`` /
    ``--train-dv-root`` / ``--out-root``, so the rig exercises the exact
    resolution the runner relies on (no single-path overrides at all).
    """
    rng = np.random.default_rng(1739)
    n_groups, per_group = 6, 4
    n_tr = n_groups * per_group
    n_ev = 12

    store_root = tmp_path / "hf_dl"
    train_dv_root = tmp_path / "train_dv"
    out_root = tmp_path / "out" / "wildchat_rung"

    # --- per-behavior TRAIN inputs (distinct DV per behavior) ------------------
    train_dvs: dict[str, np.ndarray] = {}
    for behavior in BEHAVIORS:
        dv_tr = rng.uniform(0, 100, size=n_tr)
        train_dvs[behavior] = dv_tr
        tr_rows = [
            {"context_id": f"tr{i:03d}", "rollout_k": 0, "stratum": "core"} for i in range(n_tr)
        ]
        _write_store(
            store_root / f"{behavior}_labeling",
            tr_rows,
            {
                "context_end": _acts(rng, n_tr, dv_tr / 50.0),
                "prefix_end": _acts(rng, n_tr, dv_tr / 80.0),
                "t1": _acts(rng, n_tr, dv_tr / 60.0),
            },
        )
        p = train_dv_root / behavior / "labeling.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            _dv_payload(
                [f"tr{i:03d}" for i in range(n_tr)],
                dv_tr,
                split="train",
                rung="train",
                groups=lambda i, c, pg=per_group: f"g{i // pg}",
            )
        )
        e1_rows = [
            {"context_id": f"e1{i:03d}", "side": "pos" if i < 6 else "neg", "rollout_k": 0}
            for i in range(12)
        ]
        _write_store(
            store_root / f"{behavior}_extraction",
            e1_rows,
            {"t1": _acts(rng, 12, np.array([1.0] * 6 + [-1.0] * 6))},
        )

    # --- ONE shared wildchat-rung capture store -------------------------------
    ev_ids = [f"wc{i:03d}" for i in range(n_ev)]
    ev_signal = rng.uniform(0, 100, size=n_ev)
    wc_store = _write_store(
        store_root / "wcrung_capture_store" / "wildchat",
        [{"context_id": c, "rollout_k": 0} for c in ev_ids],
        {
            "context_end": _acts(rng, n_ev, ev_signal / 50.0),
            "prefix_end": _acts(rng, n_ev, ev_signal / 80.0),
            "t1": _acts(rng, n_ev, ev_signal / 60.0),
        },
    )

    # --- THREE per-behavior rung DVs over that one pool -----------------------
    # hallucination deliberately DROPS two contexts (a per-rubric judge drop),
    # so the coherence artifact has a real n_eval spread to report.
    ev_dvs: dict[str, np.ndarray] = {}
    ev_kept: dict[str, list[str]] = {}
    for behavior in BEHAVIORS:
        kept = ev_ids[:-2] if behavior == "hallucination" else list(ev_ids)
        dv_ev = rng.uniform(0, 100, size=len(kept))
        ev_dvs[behavior] = dv_ev
        ev_kept[behavior] = kept
        p = out_root / "dv_dataset" / behavior / "labeling.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            _dv_payload(
                kept,
                dv_ev,
                split="eval",
                rung="wildchat_rung",
                # group_key == the conversation's own prefix id (each WildChat
                # conversation is its own group — conversation-disjoint rung).
                groups=lambda i, c: f"wcpfx-{c}",
            )
        )

    _write_store(
        store_root / "u_store",
        [{"context_id": f"u{i:03d}", "stratum": "core"} for i in range(40)],
        {
            "context_end": _acts(rng, 40),
            "prefix_end": _acts(rng, 40),
            "t1": _acts(rng, 40),
        },
    )

    def argv(behaviors: list[str]) -> list[str]:
        return [
            "--behaviors",
            *behaviors,
            "--variants",
            "context_end",
            "prefix_end",
            "--arms",
            *ROSTER,
            "--layers",
            *[str(x) for x in LAYERS],
            "--n-layers",
            "2",
            "--store-root",
            str(store_root),
            "--train-dv-root",
            str(train_dv_root),
            "--out-root",
            str(out_root),
            "--main-root",
            str(tmp_path / "main-absent"),
            "--tensors-root",
            str(tmp_path / "tensors-absent"),
            "--n-boot",
            "32",
        ]

    return {
        "argv": argv,
        "out_root": out_root,
        "store_root": store_root,
        "train_dv_root": train_dv_root,
        "wc_store": wc_store,
        "ev_kept": ev_kept,
        "n_ev": n_ev,
    }


def _run(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as exc:
        wca.main(argv)
    return int(exc.value.code or 0)


# ---------------------------------------------------------------------------
# tiny-real end to end
# ---------------------------------------------------------------------------


def test_tiny_real_e2e_scores_wildchat_rung(rig, capsys):
    """Full production path: real stores -> whitening -> map refit -> arms."""
    assert _run(rig["argv"](["evil"])) == 0
    payload = json.loads((rig["out_root"] / "evil" / "all_arms_spearman.json").read_text())

    meta = payload["meta"]
    assert meta["mode"] == "wcrung_transfer"
    assert meta["rung"] == "wildchat_rung"
    assert meta["eval_rungs"] == ["wildchat_rung"]
    assert meta["judge_called"] is False
    assert meta["map_source"] == "refit-in-process"
    assert meta["n_contexts"] == rig["n_ev"]
    assert meta["eval_store_shared_across_behaviors"] is True
    assert len(meta["eval_ctx_ids_sha256"]) == 64
    assert meta["git_commit"]
    assert "numpy" in meta["env_versions"]
    # No committed train summary in this rig -> own-train-pool selection.
    assert set(meta["frozen_layer_source"]) == {"context_end", "prefix_end"}
    assert all(v == "own-train-pool-selection" for v in meta["frozen_layer_source"].values())
    assert meta["rb"]["rb_source"] == "extract"
    # Both read-only DV inputs are sha-recorded (rail 2).
    assert str(rig["train_dv_root"] / "evil" / "labeling.json") in meta["input_sha256"]
    assert str(rig["out_root"] / "dv_dataset" / "evil" / "labeling.json") in meta["input_sha256"]
    # The shared store is the resolved input (root-driven default).
    assert meta["input_paths"]["wcrung_store"] == str(rig["wc_store"])

    rows = payload["transfer_rows"]
    assert rows, "no wildchat-rung transfer rows emitted"
    assert {r["eval_rung"] for r in rows} == {"wildchat_rung"}
    assert {r["rung_kind"] for r in rows} == {"eval_transfer"}
    assert {r["variant"] for r in rows} == {"context_end", "prefix_end"}
    for r in rows:
        assert r["n_eval"] == rig["n_ev"]
        assert -1.0 <= r["rho_frozen"] <= 1.0
        assert len(r["ci_frozen"]) == 2

    per_layer = payload["per_layer_rows"]
    assert per_layer, "no per-layer rows emitted"
    assert {p["rung_kind"] for p in per_layer} == {"eval_transfer_per_layer"}
    for p in per_layer:
        assert len(p["rho_per_layer"]) == len(LAYERS), p
        assert p["frozen_layer_idx"] in range(len(LAYERS))
        assert p["frozen_layer"] in LAYERS
    # Every roster arm that produced scores has BOTH a frozen row and a profile.
    scored = {(p["arm"], p["variant"]) for p in per_layer}
    assert scored == {(r["arm"], r["variant"]) for r in rows}
    # Index consistency: the frozen-layer headline IS the profile's value at the
    # frozen index (catches an off-by-one between the two reads).
    prof = {(p["arm"], p["variant"]): p for p in per_layer}
    for r in rows:
        p = prof[(r["arm"], r["variant"])]
        assert r["rho_frozen"] == pytest.approx(p["rho_per_layer"][p["frozen_layer_idx"]]), r
        assert r["layer"] == p["frozen_layer"]
    assert {a for a, _ in scored} <= set(ROSTER)
    assert len(scored) >= 2 * 2, f"expected >=2 arms x 2 variants, got {sorted(scored)}"

    diag = json.loads((rig["out_root"] / "evil" / "map_diagnostics.json").read_text())
    assert diag and all(d["map_source"] == "refit" for d in diag.values())
    assert not (rig["out_root"] / "wcrung_arms_failures.json").exists()
    assert "unit 1/2" in capsys.readouterr().out


def test_one_shared_store_scores_three_behaviors_independently(rig):
    """THE rung invariant: one capture store, three rubric DVs, three reads.

    Each behavior must produce its OWN rho against its OWN DV while reading the
    SAME activations (generate-once/judge-3x). A regression that leaked one
    behavior's DV into another would collapse these to identical rows.
    """
    assert _run(rig["argv"](list(BEHAVIORS))) == 0

    store_paths, rho_by_behavior, ctx_shas = set(), {}, {}
    for behavior in BEHAVIORS:
        payload = json.loads((rig["out_root"] / behavior / "all_arms_spearman.json").read_text())
        meta = payload["meta"]
        assert meta["behavior"] == behavior
        store_paths.add(meta["input_paths"]["wcrung_store"])
        ctx_shas[behavior] = meta["eval_ctx_ids_sha256"]
        assert meta["n_contexts"] == len(rig["ev_kept"][behavior])
        # each behavior read its OWN per-behavior DV + train inputs
        assert (
            str(rig["out_root"] / "dv_dataset" / behavior / "labeling.json") in meta["input_sha256"]
        )
        assert meta["input_paths"]["train_store"].endswith(f"{behavior}_labeling")
        rho_by_behavior[behavior] = {
            (r["arm"], r["variant"]): r["rho_frozen"] for r in payload["transfer_rows"]
        }
        assert rho_by_behavior[behavior], f"{behavior}: no transfer rows"

    # ONE store served all three behaviors.
    assert len(store_paths) == 1, store_paths
    assert store_paths == {str(rig["wc_store"])}
    # The two full-coverage behaviors saw the identical context list; the
    # judge-dropped one did not.
    assert ctx_shas["evil"] == ctx_shas["sycophancy"]
    assert ctx_shas["hallucination"] != ctx_shas["evil"]
    # Independent DVs -> independent reads (not a shared-DV collapse).
    assert rho_by_behavior["evil"] != rho_by_behavior["sycophancy"]


def test_pool_coherence_artifact_reports_spread(rig):
    """The shared-pool integrity artifact records shas + the n_eval spread."""
    assert _run(rig["argv"](list(BEHAVIORS))) == 0
    coh = json.loads((rig["out_root"] / "wcrung_arms_pool_coherence.json").read_text())
    assert set(coh["per_behavior"]) == set(BEHAVIORS)
    assert coh["identical_ctx_lists"] is False  # hallucination dropped two
    assert coh["n_eval_spread"] == [rig["n_ev"] - 2, rig["n_ev"]]
    assert coh["per_behavior"]["evil"]["n_eval_contexts"] == rig["n_ev"]


def test_single_behavior_run_writes_no_coherence_artifact(rig):
    """The coherence artifact is a CROSS-behavior read — absent for one behavior."""
    assert _run(rig["argv"](["evil"])) == 0
    assert not (rig["out_root"] / "wcrung_arms_pool_coherence.json").exists()


def test_resume_skips_completed_units(rig, capsys):
    """The per-unit checkpoint resumes instead of re-fitting (byte-identical rows)."""
    assert _run(rig["argv"](["evil"])) == 0
    first = json.loads((rig["out_root"] / "evil" / "all_arms_spearman.json").read_text())
    capsys.readouterr()
    assert _run(rig["argv"](["evil"])) == 0
    second = json.loads((rig["out_root"] / "evil" / "all_arms_spearman.json").read_text())
    assert "SKIP (resume)" in capsys.readouterr().out
    assert first["transfer_rows"] == second["transfer_rows"]
    assert first["per_layer_rows"] == second["per_layer_rows"]


def test_missing_behavior_input_is_recorded_and_exits_nonzero(rig, capsys):
    """A missing input for ONE behavior fails loud without discarding others."""
    (rig["out_root"] / "dv_dataset" / "evil" / "labeling.json").unlink()
    assert _run(rig["argv"](["evil"])) == 2
    failures = json.loads((rig["out_root"] / "wcrung_arms_failures.json").read_text())
    assert failures[0]["behavior"] == "evil"
    assert "missing input" in failures[0]["error"]
    assert "FAILED" in capsys.readouterr().err


def test_train_dv_passed_as_rung_dv_is_refused(rig):
    """Pointing the rung DV at a TRAIN DV fails loud (no eval-split rows)."""
    argv = [
        *rig["argv"](["evil"]),
        "--wcrung-dv-json",
        str(rig["train_dv_root"] / "evil" / "labeling.json"),
    ]
    assert _run(argv) == 2
    failures = json.loads((rig["out_root"] / "wcrung_arms_failures.json").read_text())
    assert "split='eval'" in failures[0]["error"]


def test_eval_split_dv_of_another_rung_is_refused(rig, tmp_path):
    """An eval-split DV from a DIFFERENT rung trips the rung guard."""
    payload = json.loads((rig["out_root"] / "dv_dataset" / "evil" / "labeling.json").read_text())
    for row in payload["rows"]:
        row["rung"] = "pvsynth"
    other = tmp_path / "other_rung_labeling.json"
    other.write_text(json.dumps(payload))
    argv = [*rig["argv"](["evil"]), "--wcrung-dv-json", str(other)]
    assert _run(argv) == 2
    failures = json.loads((rig["out_root"] / "wcrung_arms_failures.json").read_text())
    assert "must carry rung='wildchat_rung'" in failures[0]["error"]
    assert "pvsynth" in failures[0]["error"]


# ---------------------------------------------------------------------------
# safety rails
# ---------------------------------------------------------------------------


def test_judge_module_rail_fires(monkeypatch):
    monkeypatch.setitem(sys.modules, "explore_persona_space.eval.batch_judge", object())
    with pytest.raises(RuntimeError, match="judge surface imported"):
        wca._assert_no_judge_modules("in test")


def test_no_judge_symbols_in_source():
    src = Path(wca.__file__).read_text()
    for bad in ("judge_completions_batch", "judge_graded", "judge_items_graded", "dispatch_judge"):
        assert bad not in src, f"judge call surface {bad!r} present in the arm-scoring leg"


def test_input_sha_mutation_detected(tmp_path):
    p = tmp_path / "labeling.json"
    p.write_text('{"rows": []}')
    shas = {str(p): wca._sha256(p)}
    wca._verify_input_shas(shas)  # unchanged -> no raise
    p.write_text('{"rows": [1]}')
    with pytest.raises(RuntimeError, match="read-only input MUTATED"):
        wca._verify_input_shas(shas)


def test_out_root_must_be_wildchat_rung_subtree(tmp_path):
    with pytest.raises(SystemExit, match="must be a 'wildchat_rung' subtree"):
        wca._assert_outputs_safe([tmp_path / "x.json"], out_root=tmp_path / "evil", allow=False)


def test_refuses_tracked_output(tmp_path, monkeypatch):
    monkeypatch.setattr(wca, "_git_tracked", lambda p: True)
    root = tmp_path / "wildchat_rung"
    root.mkdir()
    with pytest.raises(SystemExit, match="git-TRACKED"):
        wca._assert_outputs_safe(
            [root / "evil" / "all_arms_spearman.json"], out_root=root, allow=False
        )
    # explicit opt-in proceeds
    wca._assert_outputs_safe([root / "evil" / "all_arms_spearman.json"], out_root=root, allow=True)


# ---------------------------------------------------------------------------
# frozen-layer selection
# ---------------------------------------------------------------------------


def _summary(rows: list[dict]) -> dict:
    return {"arm_rows": rows}


def _arm_row(arm: str, rhos: list[float], **kw) -> dict:
    row = {
        "arm": arm,
        "variant": "context_end",
        "regime": "e1",
        "u_rung_label": "full",
        "f_u": None,
        "rho_per_layer": rhos,
    }
    row.update(kw)
    return row


def test_modal_frozen_layer_takes_the_mode_over_units(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(
        json.dumps(
            _summary(
                [
                    _arm_row("arm1_ctx_e1", [0.1, 0.9, 0.2]),
                    _arm_row("arm1_ctx_e1", [0.1, 0.8, 0.2]),
                    _arm_row("arm1_ctx_e1", [0.9, 0.1, 0.2]),
                ]
            )
        )
    )
    got = wca.modal_frozen_layers(p, variant="context_end", regime="e1", u_rung_label="full")
    assert got == {"arm1_ctx_e1": 1}


def test_modal_frozen_layer_breaks_ties_to_smallest_index(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(
        json.dumps(
            _summary(
                [
                    _arm_row("arm1_ctx_e1", [0.9, 0.1]),
                    _arm_row("arm1_ctx_e1", [0.1, 0.9]),
                ]
            )
        )
    )
    got = wca.modal_frozen_layers(p, variant="context_end", regime="e1", u_rung_label="full")
    assert got == {"arm1_ctx_e1": 0}


def test_modal_frozen_layer_ignores_composition_and_other_slices(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(
        json.dumps(
            _summary(
                [
                    _arm_row("arm1_ctx_e1", [0.1, 0.9]),
                    _arm_row("arm1_ctx_e1", [0.9, 0.1], f_u=0.5),  # composition cell
                    _arm_row("arm1_ctx_e1", [0.9, 0.1], variant="prefix_end"),
                    _arm_row("arm1_ctx_e1", [0.9, 0.1], regime="e2"),
                    _arm_row("arm1_ctx_e1", [0.9, 0.1], u_rung_label="250"),
                ]
            )
        )
    )
    got = wca.modal_frozen_layers(p, variant="context_end", regime="e1", u_rung_label="full")
    assert got == {"arm1_ctx_e1": 1}


def test_modal_frozen_layer_fails_loud_on_no_matching_rows(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(_summary([_arm_row("arm1_ctx_e1", [0.1, 0.9], regime="e2")])))
    with pytest.raises(RuntimeError, match="no plain-rung arm_rows"):
        wca.modal_frozen_layers(p, variant="context_end", regime="e1", u_rung_label="full")


def test_committed_summary_path_is_used_when_present(rig):
    """A committed train summary supplies the frozen layers (modal), not own-pool."""
    main_root = rig["out_root"].parent / "main"
    summary = main_root / "evil" / "arm_results" / "all_arms_spearman.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(
            _summary(
                [
                    _arm_row(a, [0.9, 0.1], variant=v)
                    for a in ROSTER
                    for v in ("context_end", "prefix_end")
                ]
            )
        )
    )
    argv = rig["argv"](["evil"])
    argv[argv.index("--main-root") + 1] = str(main_root)
    assert _run(argv) == 0
    meta = json.loads((rig["out_root"] / "evil" / "all_arms_spearman.json").read_text())["meta"]
    assert all(
        v.startswith("modal-committed-train-cells:") for v in meta["frozen_layer_source"].values()
    )
    # the summary is sha-pinned as a read-only input too
    assert str(summary) in meta["input_sha256"]
    # every frozen layer came from the summary's argmax (index 0)
    rows = json.loads((rig["out_root"] / "evil" / "all_arms_spearman.json").read_text())[
        "per_layer_rows"
    ]
    assert {p["frozen_layer_idx"] for p in rows} == {0}


# ---------------------------------------------------------------------------
# path resolution + CLI guards
# ---------------------------------------------------------------------------


def test_wcrung_store_default_follows_store_root(tmp_path):
    args = wca.parse_args(["--store-root", str(tmp_path / "sr")])
    assert wca.resolve_wcrung_store(args) == tmp_path / "sr" / "wcrung_capture_store" / "wildchat"
    assert args.u_store == tmp_path / "sr" / "u_store"


def test_wcrung_store_override_is_verbatim(tmp_path):
    store = tmp_path / "somewhere" / "wildchat"
    store.mkdir(parents=True)
    args = wca.parse_args(["--wcrung-store", str(store)])
    assert wca.resolve_wcrung_store(args) == store


def test_wcrung_store_parent_resolves_to_child_capture_store(tmp_path):
    """A mirror ROOT holding wildchat/ resolves to the store, not the parent."""
    parent = tmp_path / "capture_store"
    child = parent / "wildchat"
    child.mkdir(parents=True)
    (child / wca.CAPTURE_MANIFEST_NAME).write_text("{}")
    args = wca.parse_args(["--wcrung-store", str(parent)])
    assert wca.resolve_wcrung_store(args) == child


def test_wcrung_store_is_shared_across_behaviors_by_design(tmp_path):
    """The eval store is the ONE input that may be shared by a 3-behavior run.

    pvsynth refuses a shared eval store (per-behavior contexts); this rung
    REQUIRES it (one pool judged three ways). Same store path for every
    behavior, per-behavior DV paths.
    """
    store = tmp_path / "wildchat"
    store.mkdir()
    args = wca.parse_args(
        ["--behaviors", *BEHAVIORS, "--wcrung-store", str(store), "--out-root", str(tmp_path / "o")]
    )
    resolved = {b: wca._behavior_paths(args, b) for b in BEHAVIORS}
    assert {str(p["wcrung_store"]) for p in resolved.values()} == {str(store)}
    assert len({str(p["wcrung_dv"]) for p in resolved.values()}) == len(BEHAVIORS)
    assert len({str(p["train_store"]) for p in resolved.values()}) == len(BEHAVIORS)
    assert len({str(p["train_dv"]) for p in resolved.values()}) == len(BEHAVIORS)
    assert len({str(p["e1_store"]) for p in resolved.values()}) == len(BEHAVIORS)


@pytest.mark.parametrize(
    "flag",
    ["--train-store", "--train-dv-json", "--e1-store", "--wcrung-dv-json", "--train-summary"],
)
def test_per_behavior_override_refused_for_multi_behavior_run(flag, tmp_path, capsys):
    with pytest.raises(SystemExit):
        wca.parse_args(["--behaviors", "evil", "sycophancy", flag, str(tmp_path / "x")])
    assert "name ONE behavior's input" in capsys.readouterr().err


def test_per_behavior_override_allowed_for_one_behavior(tmp_path):
    args = wca.parse_args(["--behaviors", "evil", "--wcrung-dv-json", str(tmp_path / "x.json")])
    assert args.wcrung_dv_json == tmp_path / "x.json"


def test_train_dv_root_defaults_under_main_root(tmp_path):
    args = wca.parse_args(["--main-root", str(tmp_path / "mr")])
    assert args.train_dv_root == tmp_path / "mr" / "dv_dataset"


def test_default_roster_is_the_six_transfer_arms():
    from explore_persona_space.experiments.issue_1739 import arms

    args = wca.parse_args([])
    assert args.arms is None  # -> TRANSFER_ARMS at score time
    assert len(arms.TRANSFER_ARMS) == 6
    assert set(arms.TRANSFER_ARMS) == {
        "arm1_ctx_e1",
        "arm3_identity_bias",
        "arm4_ridge_ctx",
        "arm6_map_proj_e1",
        "arm11_oracle_proj",
        "arm13_shuffled_map",
    }


def test_default_variants_and_layers_match_the_other_rungs():
    args = wca.parse_args([])
    assert args.variants == ["context_end", "prefix_end"]
    assert args.n_layers == 28
    assert args.regime == "e1"


# ---------------------------------------------------------------------------
# DV-construct metadata
# ---------------------------------------------------------------------------


def test_dv_meta_carries_the_three_rung_caveats_for_every_behavior():
    for behavior in BEHAVIORS:
        meta = wca.dv_construct_meta(behavior)
        assert meta["rung_caption"] == "random held-out WildChat (conversation-disjoint)"
        blob = " ".join(meta["caveats"])
        assert "SHARED POOL" in blob
        assert "987/2000" in blob  # single-turn disclosure
        assert "36/1013" in blob  # repeat-query disclosure
        assert "drop-never-coerce" in meta["dv_recipe"]
        assert "claude-sonnet-4-5-20250929" in meta["dv_recipe"]


def test_dv_meta_flags_hallucination_reference_answerless_rubric():
    hall = wca.dv_construct_meta("hallucination")
    assert any("STATED DEVIATION" in c for c in hall["caveats"])
    assert len(hall["caveats"]) == len(wca.dv_construct_meta("evil")["caveats"]) + 1
