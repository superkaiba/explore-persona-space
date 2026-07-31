"""Tiny-real tests for the #1739 bare-query SCORING entrypoint.

The e2e tests drive the PRODUCTION path of ``scripts/issue1739_bareq_score.py``
end to end on CPU with real library types at every internal seam (real
``store_io`` stores on disk, real ``_load_labeled``, real whitening + map fits,
real ``arms.run_transfer_cell`` / ``evaluate_transfer``, real
``fits.map_diagnostics`` -> ``analysis.mapping_baselines``) — nothing is stubbed;
only the SCALE is tiny (2 layers, dim 7, ~24 contexts).

The centerpieces are this round's three load-bearing invariants:

* **BY-QUERY fold non-leakage** — on evil's pool the bare rep is the IDENTICAL
  vector for every row sharing a query, so no query may straddle a fold. The
  fixture deliberately makes the DV's ``group_key`` DISAGREE with the query id,
  so a run that folded by ``group_key`` would split a query and fail.
* **The mapping-baselines applicability split** — REQUIRED for leg 2 (it fits a
  map), INAPPLICABLE to leg 1 (evaluation only), stated in the summary JSON.
* **Frozen-index semantics** — committed frozen layers are POSITIONAL indices
  into the full 28-layer grid, so a reduced ``--layers`` run must fail loud
  rather than score at a clamped layer.
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

from scripts import issue1739_bareq_score as bqs  # noqa: E402

DIM = 7  # > len(LAYERS) and != every row count, so an (n, d) <-> (d, n) slip cannot pass
LAYERS = (0, 1)
ROSTER = ["arm1_ctx_e1", "arm3_identity_bias", "arm4_ridge_ctx", "arm6_map_proj_e1"]
BEHAVIORS = ("evil", "sycophancy", "hallucination")
N_TRAIN = 24  # 8 queries x 3 contexts
N_QUERIES = 8
N_EVAL = 12  # wildchat rung: first 5 multi-turn (re-captured), rest single-turn (reused)
N_MULTI = 5


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


def _write_capture_store(root: Path, rows: list[dict], arrays: dict[str, np.ndarray]) -> Path:
    """A capture store PLUS the ``_capture_manifest.json`` discriminator."""
    _write_store(root, rows, arrays)
    (root / bqs.CAPTURE_MANIFEST_NAME).write_text(json.dumps({"n_rows": len(rows)}))
    return root


def _acts(rng: np.random.Generator, n: int, signal: np.ndarray | None = None) -> np.ndarray:
    """(n_layers, n, DIM) activations, optionally with a per-row signal in dim 0."""
    a = rng.normal(size=(len(LAYERS), n, DIM)) * 0.5
    if signal is not None:
        a[:, :, 0] += signal[None, :]
    return a


def _const_rows(vec: np.ndarray, n: int) -> np.ndarray:
    """(n_layers, n, DIM) — ``vec`` (n_layers, DIM) repeated n times (a bare prefix)."""
    return np.repeat(np.asarray(vec)[:, None, :], n, axis=1)


def _dv_rows(ctx_ids, dv, *, split, rung, group_of) -> list[dict]:
    return [
        {
            "context_id": c,
            "dv": float(dv[i]),
            "split": split,
            "rung": rung,
            "group_key": group_of(i, c),
            "per_rollout_scores": {"k0": float(dv[i])},
        }
        for i, c in enumerate(ctx_ids)
    ]


@pytest.fixture
def rig(tmp_path: Path):
    """Tiny-real inputs in the driver's own default layout.

    Structure that matters:

    * ``sycophancy`` / ``hallucination`` train stores have a CONSTANT
      ``prefix_end`` (already bare renders -> render-MATCHED); ``evil``'s varies
      (prefix-crossed -> render-MISMATCHED).
    * the wildchat store's first ``N_MULTI`` contexts are multi-turn (varying
      prefix, re-captured bare) and the rest are single-turn (prefix == the bare
      constant head, so their committed reps are REUSED).
    * the query bank groups evil's 24 train contexts into 8 queries of 3, while
      the DV's ``group_key`` cuts across queries — so folding by ``group_key``
      instead of ``query_id`` would split a query.
    """
    rng = np.random.default_rng(1739)
    store_root = tmp_path / "hf_dl"
    main_root = tmp_path / "main"
    train_dv_root = tmp_path / "train_dv"
    out_root = tmp_path / "out" / "bareq_map"

    # The one constant bare-render template head, shared by every bare row and
    # by every single-turn wildchat row (fp16-rounded so both stores round-trip
    # the identical bytes).
    prefix_const = np.asarray(rng.normal(size=(len(LAYERS), DIM)), dtype=np.float16).astype(
        np.float64
    )

    train_ids = [f"tr{i:03d}" for i in range(N_TRAIN)]
    eval_ids = [f"ev{i:03d}" for i in range(6)]  # evil's own eval-split contexts
    train_dvs: dict[str, np.ndarray] = {}
    for behavior in BEHAVIORS:
        dv_tr = rng.uniform(0, 100, size=N_TRAIN)
        dv_ev_own = rng.uniform(0, 100, size=len(eval_ids))
        train_dvs[behavior] = dv_tr
        n_all = N_TRAIN + len(eval_ids)
        prefix = (
            _const_rows(prefix_const, n_all)
            if behavior != "evil"
            else _acts(rng, n_all, np.concatenate([dv_tr, dv_ev_own]) / 90.0)
        )
        _write_store(
            store_root / f"{behavior}_labeling",
            [{"context_id": c, "rollout_k": 0, "stratum": "core"} for c in [*train_ids, *eval_ids]],
            {
                "context_end": _acts(rng, n_all, np.concatenate([dv_tr, dv_ev_own]) / 50.0),
                "prefix_end": prefix,
                "t1": _acts(rng, n_all, np.concatenate([dv_tr, dv_ev_own]) / 60.0),
            },
        )
        p = train_dv_root / behavior / "labeling.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        # group_key cuts ACROSS queries (query = i // 3; group = i // 4), so a
        # group-keyed fold would straddle queries and a query-keyed one cannot.
        p.write_text(
            json.dumps(
                {
                    "rows": [
                        *_dv_rows(
                            train_ids,
                            dv_tr,
                            split="train",
                            rung="train",
                            group_of=lambda i, c: f"g{i // 4}",
                        ),
                        *_dv_rows(
                            eval_ids,
                            dv_ev_own,
                            split="eval",
                            rung=f"{behavior}_ood",
                            group_of=lambda i, c: f"eg{i // 2}",
                        ),
                    ]
                }
            )
        )
        _write_store(
            store_root / f"{behavior}_extraction",
            [
                {"context_id": f"e1{i:03d}", "side": "pos" if i < 6 else "neg", "rollout_k": 0}
                for i in range(12)
            ],
            {"t1": _acts(rng, 12, np.array([1.0] * 6 + [-1.0] * 6))},
        )

    # --- the ONE shared wildchat-rung capture store ---------------------------
    wc_ids = [f"wc{i:03d}" for i in range(N_EVAL)]
    wc_signal = rng.uniform(0, 100, size=N_EVAL)
    wc_prefix = np.concatenate(
        [
            _acts(rng, N_MULTI, wc_signal[:N_MULTI] / 70.0),
            _const_rows(prefix_const, N_EVAL - N_MULTI),
        ],
        axis=1,
    )
    _write_store(
        store_root / "wcrung_capture_store" / bqs.WCRUNG_STORE_DIR_NAME,
        [{"context_id": c, "rollout_k": 0} for c in wc_ids],
        {
            "context_end": _acts(rng, N_EVAL, wc_signal / 50.0),
            "prefix_end": wc_prefix,
            "t1": _acts(rng, N_EVAL, wc_signal / 60.0),
        },
    )
    for behavior in BEHAVIORS:
        dv_wc = rng.uniform(0, 100, size=N_EVAL)
        p = main_root / bqs.WCRUNG / "dv_dataset" / behavior / "labeling.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(
            json.dumps(
                {
                    "rows": _dv_rows(
                        wc_ids,
                        dv_wc,
                        split="eval",
                        rung=bqs.WCRUNG,
                        group_of=lambda i, c: f"wcpfx-{c}",
                    )
                }
            )
        )

    # --- the BARE capture store: leg-1 (multi-turn wc) + leg-2 (query bank) ---
    qids = [f"q{i:02d}" for i in range(N_QUERIES)]
    ctx_to_qid = {train_ids[i]: qids[i // 3] for i in range(N_TRAIN)}
    q_signal = rng.uniform(0, 100, size=N_QUERIES)
    bare_rows = [
        # leg 1: capture.capture_rollout_files carries source_file + context_id.
        *[
            {"context_id": c, "rollout_k": 0, "rung": "bareq", "source_file": f"wc-{c}.json"}
            for c in wc_ids[:N_MULTI]
        ],
        # leg 2: the payload has NO context_id, so the row index carries null —
        # source_file is the ONLY key (see bare_row_key).
        *[
            {"context_id": None, "rollout_k": 0, "rung": "bareq", "source_file": f"q-{q}.json"}
            for q in qids
        ],
    ]
    n_bare = len(bare_rows)
    bare_ctx = np.concatenate(
        [_acts(rng, N_MULTI, wc_signal[:N_MULTI] / 45.0), _acts(rng, N_QUERIES, q_signal / 45.0)],
        axis=1,
    )
    _write_capture_store(
        store_root / "bareq_capture_store" / "bareq",
        bare_rows,
        {
            "context_end": bare_ctx,
            "prefix_end": _const_rows(prefix_const, n_bare),
            "t1": _acts(rng, n_bare),
        },
    )
    (out_root).mkdir(parents=True, exist_ok=True)
    (out_root / bqs.QUERY_MANIFEST).write_text(
        json.dumps(
            {
                "train_only": True,
                "n_unique_queries": N_QUERIES,
                "queries": [
                    {
                        "query_id": q,
                        "query": f"<redacted text {q}>",
                        "context_ids": [c for c, qq in ctx_to_qid.items() if qq == q],
                    }
                    for q in qids
                ],
            }
        )
    )

    _write_store(
        store_root / "u_store",
        [{"context_id": f"u{i:03d}", "stratum": "core"} for i in range(40)],
        {"context_end": _acts(rng, 40), "prefix_end": _acts(rng, 40), "t1": _acts(rng, 40)},
    )

    def argv(behaviors: list[str], *extra: str) -> list[str]:
        return [
            "--behaviors",
            *behaviors,
            "--variants",
            bqs.BARE_KIND,
            "--arms",
            *ROSTER,
            "--layers",
            *[str(x) for x in LAYERS],
            "--n-layers",
            str(len(LAYERS)),
            "--store-root",
            str(store_root),
            "--train-dv-root",
            str(train_dv_root),
            "--main-root",
            str(main_root),
            "--tensors-root",
            str(tmp_path / "tensors-absent"),
            "--out-root",
            str(out_root),
            "--n-boot",
            "32",
            "--force-own-pool-frozen",
            *extra,
        ]

    return {
        "argv": argv,
        "tmp": tmp_path,
        "out_root": out_root,
        "store_root": store_root,
        "main_root": main_root,
        "train_dv_root": train_dv_root,
        "wc_ids": wc_ids,
        "train_ids": train_ids,
        "eval_ids": eval_ids,
        "qids": qids,
        "ctx_to_qid": ctx_to_qid,
        "prefix_const": prefix_const,
    }


def _run(argv: list[str]) -> int:
    with pytest.raises(SystemExit) as exc:
        bqs.main(argv)
    return int(exc.value.code or 0)


def _summary(rig, behavior: str) -> dict:
    return json.loads((rig["out_root"] / behavior / "all_arms_spearman.json").read_text())


# ---------------------------------------------------------------------------
# tiny-real end to end
# ---------------------------------------------------------------------------


def test_tiny_real_e2e_scores_both_legs_for_evil(rig, capsys):
    """Full production path for the one behavior that has BOTH legs."""
    assert _run(rig["argv"](["evil"])) == 0
    payload = _summary(rig, "evil")
    meta = payload["meta"]

    assert meta["mode"] == "bareq_transfer"
    assert meta["rung"] == bqs.RUNG
    assert meta["legs_run"] == ["1", "2"]
    assert meta["judge_called"] is False
    assert payload["n_transfer_rows"] > 0

    legs = {r["leg"] for r in payload["transfer_rows"]}
    assert legs == {"1", "2"}, legs
    rungs = {r["eval_rung"] for r in payload["transfer_rows"]}
    assert bqs.WCRUNG in rungs
    for row in payload["transfer_rows"]:
        assert row["input_rep"].startswith("bare_")
        assert np.isfinite(row["rho_frozen"])
        assert len(row["ci_frozen"]) == 2

    out = capsys.readouterr().out
    assert "[phase=leg1] evil" in out
    assert "[phase=leg2] evil" in out


def test_render_match_label_is_measured_per_behavior(rig):
    """matched/mismatched comes from MEASURED train-prefix constancy, not a table."""
    assert _run(rig["argv"](["evil", "sycophancy", "hallucination"])) == 0
    labels = {b: _summary(rig, b)["meta"]["render_match"] for b in BEHAVIORS}

    assert labels["evil"]["label"] == "mismatched"
    assert labels["sycophancy"]["label"] == "matched"
    assert labels["hallucination"]["label"] == "matched"
    for behavior, rm in labels.items():
        assert rm["agrees_with_expected"] is True, behavior
        assert rm["expected_from_capture_scope_note"] == bqs.RENDER_MATCH_EXPECTED[behavior]
        assert "train_prefix_constancy" in rm


def test_transfer_rows_carry_the_render_match_label(rig):
    assert _run(rig["argv"](["sycophancy"])) == 0
    rows = [r for r in _summary(rig, "sycophancy")["transfer_rows"] if r["leg"] == "1"]
    assert rows
    assert {r["render_match"] for r in rows} == {"matched"}


# ---------------------------------------------------------------------------
# the mapping-baselines applicability split (round brief item: state it explicitly)
# ---------------------------------------------------------------------------


def test_mapping_baselines_applicability_split_is_stated(rig):
    """Leg 2 REQUIRED (fits a map) / leg 1 INAPPLICABLE (evaluation only)."""
    assert _run(rig["argv"](["evil"])) == 0
    mb = _summary(rig, "evil")["meta"]["mapping_baselines"]

    assert mb["leg1"]["applicable"] is False
    assert "FITS NO MAP" in mb["leg1"]["reason"]
    # The arm slug must not be confused with the mapping baseline.
    assert "arm3_identity_bias" in mb["leg1"]["reason"]

    assert mb["leg2"]["applicable"] is True
    pooled = mb["leg2"]["pooled_per_layer"]
    assert len(pooled) == len(LAYERS)
    for row in pooled:
        assert np.isfinite(row["r2_map_mean"])
        assert np.isfinite(row["r2_identity_bias_mean"])  # identity+bias: dims match
        for metric in ("euclidean", "cosine"):
            assert row["knn_acc_at_k_mean"][metric], metric
        assert row["knn_chance_at_k_mean"]


def test_leg2_map_reads_are_refit_per_fold_not_transductive(rig):
    assert _run(rig["argv"](["evil"])) == 0
    mb = _summary(rig, "evil")["meta"]["mapping_baselines"]["leg2"]
    scored = [f for f in mb["per_fold"] if "per_layer" in f]
    assert len(scored) == mb["n_folds_scored"] >= 2
    assert "refit per BY-QUERY fold" in mb["fold_semantics"]
    for fold in scored:
        assert fold["n_train"] > 0 and fold["n_holdout"] > 0
        # the random-80/20 split diagnostics are recorded but NOT the headline
        assert "random_split_diagnostics" in fold


def test_leg2_records_n_train_versus_d_for_every_fold(rig):
    """Estimator-validity read: n_train vs the feature dimension, per fold."""
    assert _run(rig["argv"](["evil"])) == 0
    folds = _summary(rig, "evil")["meta"]["mapping_baselines"]["leg2"]["per_fold"]
    for fold in (f for f in folds if "d_in" in f):
        assert fold["d_in"] == DIM
        assert fold["n_train_lt_d"] is bool(fold["n_train"] < DIM)


# ---------------------------------------------------------------------------
# BY-QUERY folds: the non-leakage pin
# ---------------------------------------------------------------------------


def test_by_query_folds_never_split_a_shared_query(rig):
    """Rows sharing a query land in ONE fold — the realized-run assertion."""
    assert _run(rig["argv"](["evil"])) == 0
    folds = _summary(rig, "evil")["meta"]["leg2_folds"]

    assert folds["no_query_straddles_folds"] is True
    assert folds["fold_key"].startswith("query_id")
    assert folds["n_queries"] == N_QUERIES
    assert folds["n_rows"] == N_TRAIN
    assert sum(folds["rows_per_fold"]) == N_TRAIN


def test_fold_key_is_the_query_not_the_dv_group_key(rig):
    """A group_key-keyed fold WOULD split a query — so the key must be the query.

    The fixture's ``group_key`` (``i // 4``) cuts across its ``query_id``
    (``i // 3``), so this is a real discriminator rather than a tautology: the
    realized folds respect queries while at least one group_key straddles them.
    """
    from explore_persona_space.experiments.issue_1739 import fits

    qids = [rig["ctx_to_qid"][c] for c in rig["train_ids"]]
    groups = [f"g{i // 4}" for i in range(N_TRAIN)]
    assert any(len({qids[i] for i in range(N_TRAIN) if groups[i] == g}) > 1 for g in set(groups)), (
        "fixture is not a discriminator: no group_key spans two queries"
    )

    cell = fits.realize_budget_cell(qids, budget_l=N_TRAIN, draw=0, seed=0)
    digest = bqs.assert_by_query_folds(cell, qids)
    assert digest["no_query_straddles_folds"] is True

    # ...and the DV's group_key genuinely does straddle folds under that cell.
    g = np.asarray(groups)[cell.row_idx]
    straddling = {k for k in set(g) if len({int(f) for f in cell.fold_ids[g == k]}) > 1}
    assert straddling, "expected at least one group_key to straddle the query-keyed folds"


def test_straddling_query_assignment_fails_loud():
    """A hand-built cell that splits a query is refused, not silently scored."""
    import dataclasses

    from explore_persona_space.experiments.issue_1739 import fits

    qids = ["qA", "qA", "qB", "qB"]
    cell = fits.realize_budget_cell(qids, budget_l=4, draw=0, seed=0, n_folds=2)
    bad = dataclasses.replace(cell, fold_ids=np.array([0, 1, 0, 1], dtype=np.int64))
    with pytest.raises(RuntimeError, match="BY-QUERY fold violation"):
        bqs.assert_by_query_folds(bad, qids)


def test_query_bank_with_a_context_in_two_queries_fails_loud(tmp_path):
    p = tmp_path / "bank.json"
    p.write_text(
        json.dumps(
            {
                "queries": [
                    {"query_id": "q0", "context_ids": ["c0", "c1"]},
                    {"query_id": "q1", "context_ids": ["c1"]},
                ]
            }
        )
    )
    with pytest.raises(RuntimeError, match="claimed by two queries"):
        bqs.load_query_bank(p)


def test_query_bank_digest_carries_no_query_text(tmp_path):
    """CONTENT HYGIENE: the bank holds real user text; the digest must not."""
    p = tmp_path / "bank.json"
    p.write_text(
        json.dumps(
            {
                "train_only": True,
                "queries": [
                    {"query_id": "q0", "query": "SECRET-USER-TEXT", "context_ids": ["c0"]},
                ],
            }
        )
    )
    ctx_to_qid, digest = bqs.load_query_bank(p)
    assert ctx_to_qid == {"c0": "q0"}
    assert "SECRET-USER-TEXT" not in json.dumps(digest)


# ---------------------------------------------------------------------------
# bare-store row identity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("row", "want"),
    [
        ({"source_file": "wc-wc001.json", "context_id": "wc001"}, ("1", "wc001")),
        # leg-2 payloads have NO context_id, so the row index carries null.
        ({"source_file": "q-abc123.json", "context_id": None}, ("2", "abc123")),
        # fallback for a store whose index omits source_file
        ({"context_id": "wc002"}, ("1", "wc002")),
        ({"context_id": None}, (None, None)),
    ],
)
def test_bare_row_key_attribution(row, want):
    assert bqs.bare_row_key(row) == want


def test_unattributable_bare_row_fails_loud(tmp_path):
    rng = np.random.default_rng(0)
    root = _write_capture_store(
        tmp_path / "bareq",
        [{"context_id": None, "rollout_k": 0}],
        {"context_end": _acts(rng, 1), "prefix_end": _acts(rng, 1)},
    )
    with pytest.raises(RuntimeError, match="row identity is unrecoverable"):
        bqs.load_bare_store(root, list(LAYERS), kinds=("context_end", "prefix_end"))


def test_duplicate_bare_store_key_fails_loud(tmp_path):
    rng = np.random.default_rng(0)
    root = _write_capture_store(
        tmp_path / "bareq",
        [{"source_file": "q-dup.json"}, {"source_file": "q-dup.json"}],
        {"context_end": _acts(rng, 2), "prefix_end": _acts(rng, 2)},
    )
    with pytest.raises(RuntimeError, match="duplicate bare-store key"):
        bqs.load_bare_store(root, list(LAYERS), kinds=("context_end", "prefix_end"))


# ---------------------------------------------------------------------------
# single-turn reuse + the constant-prefix null
# ---------------------------------------------------------------------------


def test_single_turn_rows_are_reused_and_multi_turn_substituted(rig):
    assert _run(rig["argv"](["evil"])) == 0
    cov = _summary(rig, "evil")["meta"]["leg1_coverage"][bqs.BARE_KIND]

    assert cov["n_eval_contexts"] == N_EVAL
    assert cov["n_bare_substituted"] == N_MULTI
    assert cov["n_reused_from_wcrung_store"] == N_EVAL - N_MULTI
    assert cov["reuse_licence_check"]["ran"] is True
    assert cov["reuse_licence_check"]["passed"] is True


def test_reuse_licence_check_fails_loud_on_a_non_bare_reused_row(rig):
    """A reused row whose prefix is NOT the bare constant head halts the leg."""
    rng = np.random.default_rng(7)
    wc = rig["store_root"] / "wcrung_capture_store" / bqs.WCRUNG_STORE_DIR_NAME
    # Make EVERY wildchat prefix a real conversation prefix: the reused
    # (non-recaptured) rows are then not bare renders.
    bad = _acts(rng, N_EVAL, np.arange(N_EVAL, dtype=float))
    for li, ly in enumerate(LAYERS):
        np.save(wc / f"prefix_end_L{ly:02d}.npy", np.asarray(bad[li], dtype=np.float16))

    assert _run(rig["argv"](["evil"], "--legs", "1")) == 2
    failures = json.loads((rig["out_root"] / bqs.FAILURES_NAME).read_text())
    assert "REUSE gate FAILED" in failures[0]["error"]


def test_reuse_check_report_mode_records_instead_of_halting(rig):
    rng = np.random.default_rng(7)
    wc = rig["store_root"] / "wcrung_capture_store" / bqs.WCRUNG_STORE_DIR_NAME
    bad = _acts(rng, N_EVAL, np.arange(N_EVAL, dtype=float))
    for li, ly in enumerate(LAYERS):
        np.save(wc / f"prefix_end_L{ly:02d}.npy", np.asarray(bad[li], dtype=np.float16))

    assert _run(rig["argv"](["evil"], "--legs", "1", "--reuse-check", "report")) == 0
    check = _summary(rig, "evil")["meta"]["leg1_coverage"][bqs.BARE_KIND]["reuse_licence_check"]
    assert check["ran"] is True
    assert check["passed"] is False
    assert check["mode"] == "report"


def test_constant_prefix_null_reads_degenerate(rig):
    """The bare prefix arm is a built-in null: constant reps -> degenerate rho."""
    assert _run(rig["argv"](["evil"])) == 0
    null = _summary(rig, "evil")["meta"]["leg1_null_probe"][bqs.BARE_KIND]

    assert null["constancy"]["constant"] is True
    assert null["any_ci_excludes_zero"] is False
    assert null["verdict"] == "degenerate-as-predicted"
    assert null["n_finite_rho"] == 0  # exactly-constant scores -> Spearman undefined


def test_non_constant_bare_prefix_is_flagged_as_an_anomaly(rig):
    """A bare prefix that VARIES is a capture bug — the null must say ANOMALY."""
    rng = np.random.default_rng(11)
    bare = rig["store_root"] / "bareq_capture_store" / "bareq"
    n_bare = N_MULTI + N_QUERIES
    varying = _acts(rng, n_bare, np.arange(n_bare, dtype=float) * 3.0)
    for li, ly in enumerate(LAYERS):
        np.save(bare / f"prefix_end_L{ly:02d}.npy", np.asarray(varying[li], dtype=np.float16))

    # the reuse check would also fire on the changed constant -> report mode
    assert _run(rig["argv"](["evil"], "--legs", "1", "--reuse-check", "report")) == 0
    null = _summary(rig, "evil")["meta"]["leg1_null_probe"][bqs.BARE_KIND]
    assert null["constancy"]["constant"] is False
    assert null["verdict"] == "ANOMALY"


def test_degenerate_null_variant_skips_the_arm_sweep(rig, capsys):
    """A verified-constant prefix variant records the null and skips the sweep.

    The sweep on a zero-variance design can only yield NaN rho, so paying a
    U-pool whitening + map refit + transfer solve for it is waste; the skip is
    recorded (never silent) and ``null_probe`` carries the verdict.
    """
    argv = rig["argv"](["evil"], "--legs", "1")
    argv[argv.index("--variants") + 1 : argv.index("--arms")] = [bqs.BARE_KIND, bqs.BARE_NULL_KIND]
    assert _run(argv) == 0

    payload = _summary(rig, "evil")
    meta = payload["meta"]
    assert meta["frozen_layer_source"][bqs.BARE_NULL_KIND].startswith("n/a — degenerate")
    assert meta["leg1_null_probe"][bqs.BARE_NULL_KIND]["verdict"] == "degenerate-as-predicted"

    # no arm rows for the null variant; the informative variant still has them
    variants = {r["variant"] for r in payload["transfer_rows"]}
    assert variants == {bqs.BARE_KIND}, variants
    skips = [s for s in payload["transfer_skips"] if s.get("variant") == bqs.BARE_NULL_KIND]
    assert skips and "constant-prefix NULL variant" in skips[0]["reason"]
    assert "SKIP (verified-degenerate null" in capsys.readouterr().out


def test_anomalous_null_variant_still_runs_the_arm_sweep(rig):
    """A prefix that VARIES is a capture bug — the arms then run as diagnosis."""
    rng = np.random.default_rng(23)
    bare = rig["store_root"] / "bareq_capture_store" / "bareq"
    n_bare = N_MULTI + N_QUERIES
    varying = _acts(rng, n_bare, np.arange(n_bare, dtype=float) * 3.0)
    for li, ly in enumerate(LAYERS):
        np.save(bare / f"prefix_end_L{ly:02d}.npy", np.asarray(varying[li], dtype=np.float16))

    argv = rig["argv"](["evil"], "--legs", "1", "--reuse-check", "report")
    argv[argv.index("--variants") + 1 : argv.index("--arms")] = [bqs.BARE_NULL_KIND]
    assert _run(argv) == 0

    payload = _summary(rig, "evil")
    assert payload["meta"]["leg1_null_probe"][bqs.BARE_NULL_KIND]["verdict"] == "ANOMALY"
    rows = [r for r in payload["transfer_rows"] if r["variant"] == bqs.BARE_NULL_KIND]
    assert rows, "the anomaly branch must run the sweep as the diagnostic"


def test_coverage_counts_unused_bare_rows(rig):
    """A captured bare row whose context has no kept DV row is counted, not hidden."""
    assert _run(rig["argv"](["evil"], "--legs", "1")) == 0
    cov = _summary(rig, "evil")["meta"]["leg1_coverage"][bqs.BARE_KIND]
    assert cov["n_bare_rows_unused"] == 0
    assert cov["n_bare_substituted"] + cov["n_reused_from_wcrung_store"] == cov["n_eval_contexts"]


def test_two_bar_verdict_tolerates_bf16_jitter_but_not_a_real_bug():
    """The bars have headroom for bf16 batch jitter and none for a render bug."""
    rng = np.random.default_rng(3)
    ref = rng.normal(size=(8, DIM))
    jittered = np.repeat(ref[:, None, :], 5, axis=1) + rng.normal(size=(8, 5, DIM)) * 1e-4
    ok = bqs._two_bar_verdict(bqs._cos_to_reference(jittered, ref), label="jitter")
    assert ok["passed"] is True
    assert ok["flat_cos_min"] >= bqs.FLAT_COS_MIN

    wrong = rng.normal(size=(8, 5, DIM))  # a different render entirely
    bad = bqs._two_bar_verdict(bqs._cos_to_reference(wrong, ref), label="wrong-render")
    assert bad["passed"] is False


# ---------------------------------------------------------------------------
# frozen-layer semantics (the committed positional-index guard)
# ---------------------------------------------------------------------------


def _write_committed_train_summary(rig, behavior: str, *, frozen_idx: int, n_layers: int = 28):
    rho = [0.1] * n_layers
    rho[frozen_idx] = 0.9
    out = rig["main_root"] / behavior / "arm_results" / "all_arms_spearman.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "arm_rows": [
                    {
                        "arm": arm,
                        "variant": bqs.BARE_KIND,
                        "regime": "e1",
                        "u_rung_label": "full",
                        "f_u": None,
                        "rho_per_layer": rho,
                        "budget_l": 250,
                        "draw": 0,
                        "seed": 0,
                    }
                    for arm in ROSTER
                ],
                "meta": {"layers": list(range(n_layers))},
            }
        )
    )
    return out


def test_committed_frozen_under_a_reduced_layer_run_fails_loud(rig):
    """Committed frozen layers INDEX the full 28-layer grid — refuse, never clamp."""
    _write_committed_train_summary(rig, "evil", frozen_idx=5)
    argv = [a for a in rig["argv"](["evil"], "--legs", "1") if a != "--force-own-pool-frozen"]
    assert _run(argv) == 2
    failures = json.loads((rig["out_root"] / bqs.FAILURES_NAME).read_text())
    assert "committed-frozen layers are indices into the FULL" in failures[0]["error"]


def test_force_own_pool_frozen_recovers_the_reduced_layer_run(rig):
    """The documented escape: select frozen layers within this run's own layers."""
    _write_committed_train_summary(rig, "evil", frozen_idx=5)
    assert _run(rig["argv"](["evil"], "--legs", "1")) == 0
    src = _summary(rig, "evil")["meta"]["frozen_layer_source"][bqs.BARE_KIND]
    assert src == "own-train-pool-selection"


def test_committed_frozen_is_used_when_the_layer_grid_matches(rig):
    """A full-width committed index resolves through the committed convention."""
    _write_committed_train_summary(rig, "evil", frozen_idx=1, n_layers=len(LAYERS))
    argv = [a for a in rig["argv"](["evil"], "--legs", "1") if a != "--force-own-pool-frozen"]
    assert _run(argv) == 0
    payload = _summary(rig, "evil")
    src = payload["meta"]["frozen_layer_source"][bqs.BARE_KIND]
    assert src.startswith("modal-committed-train-cells:")
    frozen = {r["frozen_layer_idx"] for r in payload["per_layer_rows"] if r["leg"] == "1"}
    assert frozen == {1}


def test_leg2_frozen_layers_come_from_the_bare_pool_not_the_committed_grid(rig):
    """Leg-2 arms are NEW (bare-fit), so committed indices must not be reused."""
    _write_committed_train_summary(rig, "evil", frozen_idx=1, n_layers=len(LAYERS))
    assert _run(rig["argv"](["evil"], "--legs", "2")) == 0
    rows = [r for r in _summary(rig, "evil")["per_layer_rows"] if r["leg"] == "2"]
    assert rows
    assert {r["frozen_source"] for r in rows} == {"own-bare-train-pool-selection (by-query folds)"}


# ---------------------------------------------------------------------------
# leg-2 no-op + eval-block coverage
# ---------------------------------------------------------------------------


def test_leg2_is_a_documented_measured_noop_for_the_bare_behaviors(rig):
    assert _run(rig["argv"](["sycophancy"])) == 0
    noop = _summary(rig, "sycophancy")["meta"]["leg2_noop"]

    assert noop["leg2"] == "no-op"
    assert "ALREADY IS the bare-query map" in noop["reason"]
    assert noop["capture_leg_agrees"] is True
    assert noop["measured_train_prefix_constancy"]["constant"] is True


def test_only_evil_runs_leg2(rig):
    assert _run(rig["argv"](["evil", "sycophancy", "hallucination"])) == 0
    assert _summary(rig, "evil")["meta"]["legs_run"] == ["1", "2"]
    for behavior in ("sycophancy", "hallucination"):
        assert _summary(rig, behavior)["meta"]["legs_run"] == ["1"]
        assert _summary(rig, behavior)["meta"]["leg2_noop"] is not None


def test_train_only_query_bank_skips_the_own_eval_rungs_with_a_reason(rig):
    """The capture default is --train-only, so evil's own eval rungs have no reps."""
    assert _run(rig["argv"](["evil"], "--legs", "2")) == 0
    notes = _summary(rig, "evil")["meta"]["leg2_eval_block_notes"]
    own = [n for n in notes if n["block"] == "evil_own_eval_rungs"]
    assert own, notes
    assert "TRAIN-only" in own[0]["skipped"]
    assert "--all-rungs" in own[0]["skipped"]


def test_wildchat_column_is_scored_for_leg2(rig):
    assert _run(rig["argv"](["evil"], "--legs", "2")) == 0
    meta = _summary(rig, "evil")["meta"]
    blocks = {b["name"]: b for b in meta["leg2_eval_blocks"]}
    assert bqs.WCRUNG in blocks
    assert blocks[bqs.WCRUNG]["n"] == N_EVAL
    assert blocks[bqs.WCRUNG]["n_bare_substituted"] == N_MULTI


def test_all_rungs_query_bank_scores_the_own_eval_rungs(rig):
    """With eval contexts in the bank, evil's own eval rungs become a column."""
    bank = json.loads((rig["out_root"] / bqs.QUERY_MANIFEST).read_text())
    # attach the eval-split contexts to existing queries (an --all-rungs capture)
    for i, cid in enumerate(rig["eval_ids"]):
        bank["queries"][i % N_QUERIES]["context_ids"].append(cid)
    bank["train_only"] = False
    (rig["out_root"] / bqs.QUERY_MANIFEST).write_text(json.dumps(bank))

    assert _run(rig["argv"](["evil"], "--legs", "2")) == 0
    meta = _summary(rig, "evil")["meta"]
    blocks = {b["name"]: b for b in meta["leg2_eval_blocks"]}
    assert "evil_own_eval_rungs" in blocks, meta["leg2_eval_block_notes"]
    assert blocks["evil_own_eval_rungs"]["n"] == len(rig["eval_ids"])
    rungs = {r["eval_rung"] for r in _summary(rig, "evil")["transfer_rows"] if r["leg"] == "2"}
    assert "evil_ood" in rungs, rungs


# ---------------------------------------------------------------------------
# the committed render-MISMATCHED contrast column
# ---------------------------------------------------------------------------


def _write_committed_wcrung_arms(rig, behavior: str, *, ctx_sha: str | None) -> Path:
    out = rig["main_root"] / bqs.WCRUNG / "arm_results" / behavior / "all_arms_spearman.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "transfer_rows": [
                    {
                        "arm": arm,
                        "variant": bqs.BARE_KIND,
                        "regime": "e1",
                        "eval_rung": bqs.WCRUNG,
                        "rho_frozen": 0.42,
                        "ci_frozen": [0.1, 0.7],
                        "n_eval": N_EVAL,
                        "layer": 1,
                    }
                    for arm in ROSTER
                ],
                "meta": {"eval_ctx_ids_sha256": ctx_sha, "n_contexts": N_EVAL},
            }
        )
    )
    return out


def test_committed_contrast_is_read_and_marked_comparable_on_a_sha_match(rig):
    """The mismatched column is READ, never recomputed; comparability is checked."""
    import hashlib

    sha = hashlib.sha256("\n".join(rig["wc_ids"]).encode("utf-8")).hexdigest()
    _write_committed_wcrung_arms(rig, "evil", ctx_sha=sha)
    assert _run(rig["argv"](["evil"], "--legs", "1")) == 0
    contrast = _summary(rig, "evil")["meta"]["leg1_committed_contrast"]

    assert contrast["available"] is True
    assert contrast["n_rows"] == len(ROSTER)
    assert contrast["eval_row_set_matches"] is True
    assert contrast["comparability"].startswith("comparable")


def test_committed_contrast_sha_mismatch_is_flagged_not_silently_compared(rig):
    _write_committed_wcrung_arms(rig, "evil", ctx_sha="deadbeef")
    assert _run(rig["argv"](["evil"], "--legs", "1")) == 0
    contrast = _summary(rig, "evil")["meta"]["leg1_committed_contrast"]
    assert contrast["eval_row_set_matches"] is False
    assert contrast["comparability"].startswith("NOT comparable")


def test_absent_committed_contrast_is_recorded_not_fatal(rig):
    """Leg 2 must not depend on the wcrung arm column having landed."""
    assert _run(rig["argv"](["evil"])) == 0
    contrast = _summary(rig, "evil")["meta"]["leg1_committed_contrast"]
    assert contrast["available"] is False
    assert "absent" in contrast["reason"]


# ---------------------------------------------------------------------------
# safety rails
# ---------------------------------------------------------------------------


def test_out_root_must_be_a_bareq_map_subtree(tmp_path):
    with pytest.raises(SystemExit, match="bareq_map"):
        bqs._assert_outputs_safe([], out_root=tmp_path / "wildchat_rung", allow=False)


def test_refuses_tracked_output(tmp_path, monkeypatch):
    monkeypatch.setattr(bqs._wca(), "_git_tracked", lambda p: True)
    with pytest.raises(SystemExit, match="git-TRACKED"):
        bqs._assert_outputs_safe(
            [tmp_path / "x.json"], out_root=tmp_path / "bareq_map", allow=False
        )
    bqs._assert_outputs_safe([tmp_path / "x.json"], out_root=tmp_path / "bareq_map", allow=True)


def test_judge_module_rail_fires(monkeypatch):
    monkeypatch.setitem(sys.modules, "explore_persona_space.eval.batch_judge", object())
    with pytest.raises(RuntimeError, match="judge surface imported"):
        bqs._wca()._assert_no_judge_modules("in test")


def test_no_judge_symbols_in_source():
    src = (REPO_ROOT / "scripts" / "issue1739_bareq_score.py").read_text()
    for banned in ("batch_judge", "graded_judge", "judge_dispatch", "api_dispatch"):
        assert f"import {banned}" not in src
        assert f"from explore_persona_space.eval.{banned}" not in src


def test_input_sha_mutation_is_detected(rig):
    """Rail 2: DV inputs are read-only; a mutation during the run fails loud."""
    dv = rig["train_dv_root"] / "evil" / "labeling.json"
    with pytest.raises(RuntimeError, match="MUTATED during the run"):
        bqs._wca()._verify_input_shas({str(dv): "0" * 64})


def test_summary_and_sentinel_carry_the_analogy_caveat_verbatim(rig):
    assert _run(rig["argv"](["evil"])) == 0
    meta = _summary(rig, "evil")["meta"]
    assert bqs.ANALOGY_CAVEAT in meta["caveats"]
    assert bqs.ANALOGY_CAVEAT in meta["dv"]["caveats"]
    sentinel = json.loads((rig["out_root"] / bqs.SENTINEL_NAME).read_text())
    assert bqs.ANALOGY_CAVEAT in sentinel["caveats"]
    assert sentinel["judge_called"] is False


def test_missing_bare_store_is_recorded_and_exits_nonzero(rig, capsys):
    argv = rig["argv"](["evil"], "--bareq-store", str(rig["tmp"] / "absent"))
    assert _run(argv) == 2
    failures = json.loads((rig["out_root"] / bqs.FAILURES_NAME).read_text())
    assert failures[0]["behavior"] == "evil"
    assert "bareq_store=" in failures[0]["error"]
    assert "FAILED evil" in capsys.readouterr().err


def test_resume_skips_completed_leg1_units(rig, capsys):
    assert _run(rig["argv"](["evil"], "--legs", "1")) == 0
    capsys.readouterr()
    assert _run(rig["argv"](["evil"], "--legs", "1")) == 0
    assert "SKIP (resume)" in capsys.readouterr().out


# ---------------------------------------------------------------------------
# CLI / store resolution
# ---------------------------------------------------------------------------


def test_bareq_store_default_follows_store_root(tmp_path):
    args = bqs.parse_args(["--store-root", str(tmp_path / "sr")])
    assert (
        bqs.resolve_bareq_store(args, "evil", "1")
        == tmp_path / "sr" / "bareq_capture_store" / "bareq"
    )


def test_bareq_store_leg2_prefers_the_per_behavior_child(tmp_path):
    parent = tmp_path / "p"
    for name in ("bareq", "bareq_evil"):
        (parent / name).mkdir(parents=True)
        (parent / name / bqs.CAPTURE_MANIFEST_NAME).write_text("{}")
    args = bqs.parse_args(["--bareq-store", str(parent)])
    assert bqs.resolve_bareq_store(args, "evil", "2") == parent / "bareq_evil"
    assert bqs.resolve_bareq_store(args, "evil", "1") == parent / "bareq"


def test_bareq_store_override_that_is_itself_a_store_is_verbatim(tmp_path):
    store = tmp_path / "s"
    store.mkdir()
    (store / bqs.CAPTURE_MANIFEST_NAME).write_text("{}")
    args = bqs.parse_args(["--bareq-store", str(store)])
    assert bqs.resolve_bareq_store(args, "evil", "1") == store


def test_query_manifest_default_follows_out_root(tmp_path):
    args = bqs.parse_args(["--out-root", str(tmp_path / "bareq_map")])
    paths = bqs._behavior_paths(args, "evil")
    assert paths["query_manifest"] == tmp_path / "bareq_map" / bqs.QUERY_MANIFEST


def test_train_dv_root_and_contrast_default_under_main_root(tmp_path):
    args = bqs.parse_args(["--main-root", str(tmp_path / "m")])
    assert args.train_dv_root == tmp_path / "m" / "dv_dataset"
    paths = bqs._behavior_paths(args, "evil")
    assert paths["wcrung_arms"] == (
        tmp_path / "m" / bqs.WCRUNG / "arm_results" / "evil" / "all_arms_spearman.json"
    )


@pytest.mark.parametrize(
    "flag", ["--train-store", "--train-dv-json", "--wcrung-dv-json", "--train-summary"]
)
def test_per_behavior_override_refused_for_a_multi_behavior_run(flag, tmp_path, capsys):
    with pytest.raises(SystemExit):
        bqs.parse_args(["--behaviors", "evil", "sycophancy", flag, str(tmp_path / "x")])
    assert "name ONE behavior's input" in capsys.readouterr().err


def test_default_roster_and_legs(tmp_path):
    from explore_persona_space.experiments.issue_1739 import arms

    args = bqs.parse_args([])
    assert args.arms is None  # -> TRANSFER_ARMS
    assert args.legs == ["1", "2"]
    assert args.variants == [bqs.BARE_KIND, bqs.BARE_NULL_KIND]
    assert args.n_layers == bqs.FULL_GRID_N_LAYERS
    assert len(arms.TRANSFER_ARMS) == 6


def test_leg2_behaviors_matches_the_capture_leg_scope():
    """The capture leg refuses --leg 2 for non-evil; the scorer must agree."""
    assert bqs.LEG2_BEHAVIORS == ("evil",)


def test_bare_kind_matches_the_capture_leg(tmp_path):
    """BARE_KIND / the null kind must mirror issue1739_bareq_pod's own literals."""
    src = (REPO_ROOT / "scripts" / "issue1739_bareq_pod.py").read_text()
    assert f'BARE_KIND = "{bqs.BARE_KIND}"' in src
    assert f'QUERY_MANIFEST = "{bqs.QUERY_MANIFEST}"' in src
