"""Tiny-real tests for the #1739 pvsynth arm-scoring entrypoint.

The e2e test drives the PRODUCTION path of ``scripts/issue1739_pvsynth_arms.py``
end to end on CPU with real library types at every internal seam (real
``store_io`` stores on disk, real ``_load_labeled``, real whitening + map fit,
real ``arms.run_transfer_cell`` / ``evaluate_transfer``) — nothing is stubbed;
only the SCALE is tiny (2 layers, dim 6, ~20 contexts). The remaining tests pin
the three hard safety rails and the frozen-layer selection rule.
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

from scripts import issue1739_pvsynth_arms as pva  # noqa: E402


@pytest.fixture(autouse=True)
def _judge_free_process(monkeypatch):
    """The no-judge entry rail asserts a FRESH-process condition (production:
    each leg runs in its own process, so sys.modules carries only the leg's
    OWN imports). Under a multi-file pytest run a SIBLING test file (e.g.
    test_issue1739_dataplane.py) may already have imported the judge surface,
    tripping the rail on state this leg never imported — scrub those modules
    for the test's duration (monkeypatch restores them) so the rail keeps
    testing exactly the production condition. Order-interaction fix from the
    new-arm-round pin-sweep (64 cross-file failures; every file green alone).
    """
    for m in pva.FORBIDDEN_JUDGE_MODULES:
        if m in sys.modules:
            monkeypatch.delitem(sys.modules, m)


DIM = 6
LAYERS = (0, 1)
KINDS = ("context_end", "prefix_end", "t1")
ROSTER = ["arm1_ctx_e1", "arm3_identity_bias", "arm4_ridge_ctx", "arm6_map_proj_e1"]


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


@pytest.fixture
def rig(tmp_path: Path):
    """Tiny-real inputs: train store+DV, pvsynth store+DV, E1 store, U store."""
    rng = np.random.default_rng(1739)
    n_groups, per_group = 6, 4
    n_tr = n_groups * per_group
    n_ev = 10
    n_u = 40

    dv_tr = rng.uniform(0, 100, size=n_tr)
    tr_rows = [{"context_id": f"tr{i:03d}", "rollout_k": 0, "stratum": "core"} for i in range(n_tr)]
    train_store = _write_store(
        tmp_path / "train_labeling",
        tr_rows,
        {
            "context_end": _acts(rng, n_tr, dv_tr / 50.0),
            "prefix_end": _acts(rng, n_tr, dv_tr / 80.0),
            "t1": _acts(rng, n_tr, dv_tr / 60.0),
        },
    )
    train_dv = tmp_path / "dv_dataset" / "evil" / "labeling.json"
    train_dv.parent.mkdir(parents=True, exist_ok=True)
    train_dv.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "context_id": f"tr{i:03d}",
                        "dv": float(dv_tr[i]),
                        "split": "train",
                        "rung": "train",
                        "group_key": f"g{i // per_group}",
                        "per_rollout_scores": {"k0": float(dv_tr[i])},
                    }
                    for i in range(n_tr)
                ]
            }
        )
    )

    dv_ev = rng.uniform(0, 100, size=n_ev)
    ev_rows = [{"context_id": f"pv{i:03d}", "rollout_k": 0} for i in range(n_ev)]
    pv_store = _write_store(
        tmp_path / "pvsynth_capture_store",
        ev_rows,
        {
            "context_end": _acts(rng, n_ev, dv_ev / 50.0),
            "prefix_end": _acts(rng, n_ev, dv_ev / 80.0),
            "t1": _acts(rng, n_ev, dv_ev / 60.0),
        },
    )
    pv_dv = tmp_path / "pvsynth" / "dv_dataset" / "evil" / "labeling.json"
    pv_dv.parent.mkdir(parents=True, exist_ok=True)
    pv_dv.write_text(
        json.dumps(
            {
                "rows": [
                    {
                        "context_id": f"pv{i:03d}",
                        "dv": float(dv_ev[i]),
                        "split": "eval",
                        "rung": "pvsynth",
                        "group_key": f"pvsynth-p{i % 5}-{'pos' if i % 2 else 'neg'}",
                        "per_rollout_scores": {"k0": float(dv_ev[i])},
                    }
                    for i in range(n_ev)
                ]
            }
        )
    )

    e1_rows = [
        {"context_id": f"e1{i:03d}", "side": "pos" if i < 6 else "neg", "rollout_k": 0}
        for i in range(12)
    ]
    sig = np.array([1.0] * 6 + [-1.0] * 6)
    e1_store = _write_store(tmp_path / "evil_extraction", e1_rows, {"t1": _acts(rng, 12, sig)})

    u_rows = [{"context_id": f"u{i:03d}", "stratum": "core"} for i in range(n_u)]
    u_store = _write_store(
        tmp_path / "u_store",
        u_rows,
        {
            "context_end": _acts(rng, n_u),
            "prefix_end": _acts(rng, n_u),
            "t1": _acts(rng, n_u),
        },
    )

    out_root = tmp_path / "out" / "pvsynth"
    return {
        "argv": [
            "--behaviors",
            "evil",
            "--variants",
            "context_end",
            "prefix_end",
            "--arms",
            *ROSTER,
            "--layers",
            *[str(x) for x in LAYERS],
            "--n-layers",
            "2",
            "--train-store",
            str(train_store),
            "--train-dv-json",
            str(train_dv),
            "--e1-store",
            str(e1_store),
            "--pvsynth-store",
            str(pv_store),
            "--pvsynth-dv-json",
            str(pv_dv),
            "--u-store",
            str(u_store),
            "--out-root",
            str(out_root),
            "--main-root",
            str(tmp_path / "main-absent"),
            "--tensors-root",
            str(tmp_path / "tensors-absent"),
            "--n-boot",
            "32",
        ],
        "out_root": out_root,
        "train_dv": train_dv,
        "pv_dv": pv_dv,
        "n_ev": n_ev,
    }


def _run(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as exc:
        pva.main(argv)
    return int(exc.value.code or 0)


def test_tiny_real_e2e_scores_pvsynth_rung(rig, capsys):
    """Full production path: real stores -> whitening -> map refit -> arms."""
    assert _run(rig["argv"]) == 0
    out = rig["out_root"] / "evil" / "all_arms_spearman.json"
    payload = json.loads(out.read_text())

    meta = payload["meta"]
    assert meta["mode"] == "pvsynth_transfer"
    assert meta["rung"] == "pvsynth"
    assert meta["eval_rungs"] == ["pvsynth"]
    assert meta["judge_called"] is False
    assert meta["map_source"] == "refit-in-process"
    assert meta["n_contexts"] == rig["n_ev"]
    assert meta["git_commit"]
    assert "numpy" in meta["env_versions"]
    # No committed train summary in this rig -> own-train-pool selection.
    assert set(meta["frozen_layer_source"]) == {"context_end", "prefix_end"}
    assert all(v == "own-train-pool-selection" for v in meta["frozen_layer_source"].values())
    assert meta["rb"]["rb_source"] == "extract"
    assert str(rig["train_dv"]) in meta["input_sha256"]
    assert str(rig["pv_dv"]) in meta["input_sha256"]

    rows = payload["transfer_rows"]
    assert rows, "no pvsynth transfer rows emitted"
    assert {r["eval_rung"] for r in rows} == {"pvsynth"}
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
    assert not (rig["out_root"] / "pvsynth_arms_failures.json").exists()
    assert "unit 1/2" in capsys.readouterr().out


def test_resume_skips_completed_units(rig, capsys):
    """The per-unit checkpoint resumes instead of re-fitting (byte-identical rows)."""
    assert _run(rig["argv"]) == 0
    first = json.loads((rig["out_root"] / "evil" / "all_arms_spearman.json").read_text())
    capsys.readouterr()
    assert _run(rig["argv"]) == 0
    second = json.loads((rig["out_root"] / "evil" / "all_arms_spearman.json").read_text())
    assert "SKIP (resume)" in capsys.readouterr().out
    assert first["transfer_rows"] == second["transfer_rows"]
    assert first["per_layer_rows"] == second["per_layer_rows"]


def test_missing_behavior_input_is_recorded_and_exits_nonzero(rig, capsys):
    """A missing input for ONE behavior fails loud without discarding others."""
    argv = list(rig["argv"])
    argv[argv.index("--pvsynth-store") + 1] = str(rig["out_root"] / "nope")
    assert _run(argv) == 2
    failures = json.loads((rig["out_root"] / "pvsynth_arms_failures.json").read_text())
    assert failures[0]["behavior"] == "evil"
    assert "missing input" in failures[0]["error"]
    assert "FAILED" in capsys.readouterr().err


def test_train_dv_passed_as_eval_is_refused(rig):
    """Pointing --pvsynth-dv-json at the TRAIN DV fails loud (no eval-split rows)."""
    argv = list(rig["argv"])
    argv[argv.index("--pvsynth-dv-json") + 1] = str(rig["train_dv"])
    assert _run(argv) == 2
    failures = json.loads((rig["out_root"] / "pvsynth_arms_failures.json").read_text())
    assert "split='eval'" in failures[0]["error"]


def test_eval_split_dv_of_another_rung_is_refused(rig, tmp_path):
    """An eval-split DV from a DIFFERENT rung trips the rung guard."""
    payload = json.loads(rig["pv_dv"].read_text())
    for row in payload["rows"]:
        row["rung"] = "wildchat"
    other = tmp_path / "other_rung_labeling.json"
    other.write_text(json.dumps(payload))
    argv = list(rig["argv"])
    argv[argv.index("--pvsynth-dv-json") + 1] = str(other)
    assert _run(argv) == 2
    failures = json.loads((rig["out_root"] / "pvsynth_arms_failures.json").read_text())
    assert "must carry rung='pvsynth'" in failures[0]["error"]
    assert "wildchat" in failures[0]["error"]


# ---------------------------------------------------------------------------
# safety rails
# ---------------------------------------------------------------------------


def test_judge_module_rail_fires(monkeypatch):
    monkeypatch.setitem(sys.modules, "explore_persona_space.eval.batch_judge", object())
    with pytest.raises(RuntimeError, match="judge surface imported"):
        pva._assert_no_judge_modules("in test")


def test_no_judge_symbols_in_source():
    src = Path(pva.__file__).read_text()
    for bad in ("judge_completions_batch", "judge_graded", "judge_items_graded", "dispatch_judge"):
        assert bad not in src, f"judge call surface {bad!r} present in the arm-scoring leg"


def test_input_sha_mutation_detected(tmp_path):
    p = tmp_path / "labeling.json"
    p.write_text('{"rows": []}')
    shas = {str(p): pva._sha256(p)}
    pva._verify_input_shas(shas)  # unchanged -> no raise
    p.write_text('{"rows": [1]}')
    with pytest.raises(RuntimeError, match="read-only input MUTATED"):
        pva._verify_input_shas(shas)


def test_out_root_must_be_pvsynth_subtree(tmp_path):
    with pytest.raises(SystemExit, match="must be a 'pvsynth' subtree"):
        pva._assert_outputs_safe([tmp_path / "x.json"], out_root=tmp_path / "evil", allow=False)


def test_refuses_tracked_output(tmp_path, monkeypatch):
    monkeypatch.setattr(pva, "_git_tracked", lambda p: True)
    root = tmp_path / "pvsynth"
    root.mkdir()
    with pytest.raises(SystemExit, match="git-TRACKED"):
        pva._assert_outputs_safe(
            [root / "evil" / "all_arms_spearman.json"], out_root=root, allow=False
        )
    # explicit opt-in proceeds
    pva._assert_outputs_safe([root / "evil" / "all_arms_spearman.json"], out_root=root, allow=True)


# ---------------------------------------------------------------------------
# frozen-layer selection
# ---------------------------------------------------------------------------


def _summary(rows: list[dict]) -> dict:
    return {"arm_rows": rows, "meta": {}}


def _row(arm: str, rho: list[float], **kw) -> dict:
    base = {
        "arm": arm,
        "variant": "context_end",
        "regime": "e1",
        "u_rung_label": "full",
        "f_u": None,
        "rho_per_layer": rho,
    }
    base.update(kw)
    return base


def test_modal_frozen_layer_takes_the_mode_over_units(tmp_path):
    p = tmp_path / "all_arms_spearman.json"
    p.write_text(
        json.dumps(
            _summary(
                [
                    _row("arm1_ctx_e1", [0.1, 0.9, 0.2]),  # -> 1
                    _row("arm1_ctx_e1", [0.1, 0.9, 0.2]),  # -> 1
                    _row("arm1_ctx_e1", [0.1, 0.2, 0.9]),  # -> 2
                    _row("arm4_ridge_ctx", [0.9, 0.1, 0.1]),  # -> 0
                ]
            )
        )
    )
    got = pva.modal_frozen_layers(p, variant="context_end", regime="e1", u_rung_label="full")
    assert got == {"arm1_ctx_e1": 1, "arm4_ridge_ctx": 0}


def test_modal_frozen_layer_breaks_ties_to_smallest_index(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(
        json.dumps(
            _summary(
                [
                    _row("arm1_ctx_e1", [0.1, 0.2, 0.9]),  # -> 2
                    _row("arm1_ctx_e1", [0.1, 0.9, 0.2]),  # -> 1
                ]
            )
        )
    )
    got = pva.modal_frozen_layers(p, variant="context_end", regime="e1", u_rung_label="full")
    assert got == {"arm1_ctx_e1": 1}


def test_modal_frozen_layer_ignores_composition_and_other_slices(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(
        json.dumps(
            _summary(
                [
                    _row("arm1_ctx_e1", [0.9, 0.1], f_u=0.5, u_rung_label="compose5000"),
                    _row("arm1_ctx_e1", [0.1, 0.9], variant="prefix_end"),
                    _row("arm1_ctx_e1", [0.1, 0.9], regime="e2"),
                    _row("arm1_ctx_e1", [0.1, 0.9], u_rung_label="250"),
                    _row("arm1_ctx_e1", [0.9, 0.1]),  # the only matching row -> 0
                ]
            )
        )
    )
    got = pva.modal_frozen_layers(p, variant="context_end", regime="e1", u_rung_label="full")
    assert got == {"arm1_ctx_e1": 0}


def test_modal_frozen_layer_fails_loud_on_no_matching_rows(tmp_path):
    p = tmp_path / "s.json"
    p.write_text(json.dumps(_summary([_row("arm1_ctx_e1", [0.9, 0.1], regime="e2")])))
    with pytest.raises(RuntimeError, match="no plain-rung arm_rows"):
        pva.modal_frozen_layers(p, variant="context_end", regime="e1", u_rung_label="full")


def test_committed_summary_path_is_used_when_present(rig):
    """With a committed train summary in --main-root, frozen layers come from it."""
    summary = rig["out_root"].parent / "main" / "evil" / "arm_results" / "all_arms_spearman.json"
    summary.parent.mkdir(parents=True, exist_ok=True)
    summary.write_text(
        json.dumps(
            _summary(
                [
                    _row(a, [0.9, 0.1], variant=v)
                    for a in ROSTER
                    for v in ("context_end", "prefix_end")
                ]
            )
        )
    )
    argv = list(rig["argv"])
    argv[argv.index("--main-root") + 1] = str(summary.parents[2])
    assert _run(argv) == 0
    meta = json.loads((rig["out_root"] / "evil" / "all_arms_spearman.json").read_text())["meta"]
    assert all("modal-committed-train-cells" in v for v in meta["frozen_layer_source"].values())
    assert str(summary) in meta["input_sha256"]
    per_layer = json.loads((rig["out_root"] / "evil" / "all_arms_spearman.json").read_text())[
        "per_layer_rows"
    ]
    assert all(p["frozen_layer_idx"] == 0 for p in per_layer)


def test_u_store_default_follows_store_root(tmp_path):
    """--u-store defaults under --store-root (never a constant disk)."""
    args = pva.parse_args(["--store-root", str(tmp_path / "staged")])
    assert args.u_store == tmp_path / "staged" / "u_store"
    explicit = pva.parse_args(
        ["--store-root", str(tmp_path / "staged"), "--u-store", str(tmp_path / "elsewhere")]
    )
    assert explicit.u_store == tmp_path / "elsewhere"


def test_every_default_input_path_is_per_behavior(tmp_path):
    """No default may be shared across behaviors (a shared pvsynth store would
    score one behavior's DV against another's activations)."""
    args = pva.parse_args(
        [
            "--store-root",
            str(tmp_path / "staged"),
            "--main-root",
            str(tmp_path / "main"),
            "--out-root",
            str(tmp_path / "pvsynth"),
        ]
    )
    a = pva._behavior_paths(args, "evil")
    b = pva._behavior_paths(args, "sycophancy")
    for key in a:
        assert a[key] != b[key], f"{key} default is SHARED across behaviors: {a[key]}"
    assert a["pvsynth_store"] == tmp_path / "staged" / "pvsynth_capture_store" / "evil"
    assert a["train_dv"] == tmp_path / "main" / "dv_dataset" / "evil" / "labeling.json"


def test_train_dv_root_override(tmp_path):
    """--train-dv-root repoints the train DV tree (the staged HF judge/ copy)."""
    args = pva.parse_args(
        ["--main-root", str(tmp_path / "main"), "--train-dv-root", str(tmp_path / "hf_dv")]
    )
    assert (
        pva._behavior_paths(args, "sycophancy")["train_dv"]
        == tmp_path / "hf_dv" / "sycophancy" / "labeling.json"
    )


def test_pvsynth_store_sibling_resolves_per_behavior(tmp_path):
    """A --pvsynth-store naming ONE behavior's dir resolves each behavior to its
    OWN sibling — the shape a find-first-manifest caller produces."""
    cap = tmp_path / "mirror" / "capture_store"
    for b in ("evil", "hallucination", "sycophancy"):
        (cap / b).mkdir(parents=True)
    args = pva.parse_args(
        ["--behaviors", "evil", "hallucination", "sycophancy", "--pvsynth-store", str(cap / "evil")]
    )
    assert pva._behavior_paths(args, "evil")["pvsynth_store"] == cap / "evil"
    assert pva._behavior_paths(args, "hallucination")["pvsynth_store"] == cap / "hallucination"
    assert pva._behavior_paths(args, "sycophancy")["pvsynth_store"] == cap / "sycophancy"


def test_pvsynth_store_root_path_resolves_per_behavior(tmp_path):
    """An explicit --pvsynth-store naming the capture_store ROOT (children are the
    per-behavior stores, flat manifests beside them) resolves each behavior to its
    OWN child, and so passes the multi-behavior override guard."""
    cap = tmp_path / "mirror" / "capture_store"
    for b in ("evil", "hallucination", "sycophancy"):
        (cap / b).mkdir(parents=True)
        (cap / b / pva.CAPTURE_MANIFEST_NAME).write_text("{}", encoding="utf-8")
    # the FLAT manifests that sit at the root beside the subtrees
    (cap / pva.CAPTURE_MANIFEST_NAME).write_text("{}", encoding="utf-8")
    args = pva.parse_args(
        ["--behaviors", "evil", "hallucination", "sycophancy", "--pvsynth-store", str(cap)]
    )
    for b in ("evil", "hallucination", "sycophancy"):
        assert pva._behavior_paths(args, b)["pvsynth_store"] == cap / b


def test_pvsynth_store_root_without_per_behavior_manifests_is_refused(tmp_path):
    """A path that is neither a behavior subtree nor a root of capture stores stays
    verbatim, so the multi-behavior guard still refuses it (guard preserved)."""
    d = tmp_path / "not_a_capture_store"
    (d / "evil").mkdir(parents=True)  # a dir, but carrying no capture manifest
    (d / "hallucination").mkdir(parents=True)
    with pytest.raises(SystemExit) as exc:
        pva.parse_args(["--behaviors", "evil", "hallucination", "--pvsynth-store", str(d)])
    assert exc.value.code == 2


def test_pvsynth_store_without_siblings_is_refused_for_multi_behavior(tmp_path):
    """No sibling dirs -> the override would share one store; refuse loudly."""
    lone = tmp_path / "capture_store" / "evil"
    lone.mkdir(parents=True)
    with pytest.raises(SystemExit) as exc:
        pva.parse_args(["--behaviors", "evil", "hallucination", "--pvsynth-store", str(lone)])
    assert exc.value.code == 2


def test_pvsynth_store_root_form(tmp_path):
    cap = tmp_path / "capture_store"
    args = pva.parse_args(
        ["--behaviors", "evil", "hallucination", "--pvsynth-store-root", str(cap)]
    )
    assert pva._behavior_paths(args, "hallucination")["pvsynth_store"] == cap / "hallucination"


def test_pvsynth_store_single_behavior_override_is_verbatim(tmp_path):
    """A deliberate single-behavior override is never rewritten."""
    d = tmp_path / "somewhere" / "evil"
    d.mkdir(parents=True)
    args = pva.parse_args(["--behaviors", "evil", "--pvsynth-store", str(d)])
    assert pva._behavior_paths(args, "evil")["pvsynth_store"] == d


@pytest.mark.parametrize(
    "flag", ["--train-store", "--train-dv-json", "--e1-store", "--train-summary"]
)
def test_single_path_override_refused_for_multi_behavior_run(flag, tmp_path):
    with pytest.raises(SystemExit) as exc:
        pva.parse_args(["--behaviors", "evil", "sycophancy", flag, str(tmp_path / "x")])
    assert exc.value.code == 2  # argparse usage error, not a silent mis-scoring


def test_single_path_override_allowed_for_one_behavior(tmp_path):
    args = pva.parse_args(["--behaviors", "evil", "--train-dv-json", str(tmp_path / "dv.json")])
    assert pva._behavior_paths(args, "evil")["train_dv"] == tmp_path / "dv.json"


def test_dv_construct_meta_flags_hallucination_provisional():
    ev = pva.dv_construct_meta("evil")
    assert ev["provisional"] is False and ev["caveats"] == []
    hal = pva.dv_construct_meta("hallucination")
    assert hal["provisional"] is True
    assert any("23.4%" in c for c in hal["caveats"])
    assert any("STATED DEVIATION" in c for c in hal["caveats"])


def test_polarity_split_emits_three_subsets_in_its_own_section(rig):
    """elicit / non_elicit / pooled rows land in transfer_polarity_rows.

    The pooled `transfer_rows` section is deliberately UNCHANGED (one row per
    arm x eval rung), so a consumer that predates the split still reads exactly
    what it read before; the three-way comparison is self-contained in the new
    section, discriminated by `polarity_subset`.
    """
    assert _run(rig["argv"]) == 0
    payload = json.loads((rig["out_root"] / "evil" / "all_arms_spearman.json").read_text())

    pol = payload["transfer_polarity_rows"]
    assert pol, "no polarity rows emitted"
    assert payload["n_transfer_polarity_rows"] == len(pol)
    assert {r["polarity_subset"] for r in pol} == {"pooled", "elicit", "non_elicit"}
    assert payload["meta"]["polarity_subsets"] == ["pooled", "elicit", "non_elicit"]

    # transfer_rows stays the pooled-only column: no polarity key, no row growth.
    assert all("polarity_subset" not in r for r in payload["transfer_rows"])
    per_variant_arms = {(r["variant"], r["arm"]) for r in payload["transfer_rows"]}
    assert len(payload["transfer_rows"]) == len(per_variant_arms)

    # elicit + non_elicit partition pooled exactly, per (variant, arm).
    by = {}
    for r in pol:
        by.setdefault((r["variant"], r["arm"]), {})[r["polarity_subset"]] = r
    for key, subs in by.items():
        assert set(subs) == {"pooled", "elicit", "non_elicit"}, key
        assert subs["elicit"]["n_eval"] + subs["non_elicit"]["n_eval"] == subs["pooled"]["n_eval"]
        for s in subs.values():
            assert len(s["ci_frozen"]) == 2
            assert s["layer"] == subs["pooled"]["layer"]  # same TRAIN-frozen layer

    # The pooled polarity row reproduces the untouched transfer_rows read.
    pooled = {(r["variant"], r["arm"]): r for r in pol if r["polarity_subset"] == "pooled"}
    for r in payload["transfer_rows"]:
        assert pooled[(r["variant"], r["arm"])]["rho_frozen"] == r["rho_frozen"]


def test_polarity_preds_sidecar_carries_the_label(rig):
    """Per-context preds persist the polarity, so any later split is re-analysis."""
    from explore_persona_space.experiments.issue_1739 import arms

    assert _run(rig["argv"]) == 0
    preds = rig["out_root"] / "evil" / "preds" / "pvsynth_preds.context_end.jsonl"
    rows = [json.loads(x) for x in preds.read_text(encoding="utf-8").split("\n") if x.strip()]
    assert rows
    assert {r["polarity"] for r in rows} == {"pos", "neg"}
    assert all(r["group_key"].rsplit("-", 1)[-1] == r["polarity"] for r in rows)
    n_arms = len({r["arm"] for r in rows})
    assert len(rows) == n_arms * rig["n_ev"]
    assert {r["arm"] for r in rows} <= set(arms.ARM_REGISTRY)
    assert all(r["eval_rung"] == "pvsynth" for r in rows)
