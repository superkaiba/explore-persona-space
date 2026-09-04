"""Issue #2658 unit-9 tests: the C0-C5 comparator ladder (plan section 6).

Statistical-content coverage (the unit-9 brief's mandated list):
- a synthetic separable fixture recovers a known-high AUROC (and a null-effect
  fixture stays at chance);
- C0 is PROVABLY unable to see prompt identity (feature invariance under
  prompt-id relabeling; the featurizer consumes no identity field);
- the not-estimable partition routes C2/C3 to the literal
  ``not-estimable — no frozen external direction`` record (never a proxy);
- EVERY dev-only fail-on guard is shown FIRING on a deliberately
  test-contaminated fixture (test-derived transforms, peer centering,
  dependency crossing, missing labels, coerced labels, pooled-fold metrics);
- grouped folds NEVER split a content superfamily;
- the strongest-regularization tie-break selects the smallest C within 1e-4;
- the pooled cross-fold prediction path is structurally a loud dead end;
- resume skips completed units and recomputes when a generating parameter
  changes;
- the n_train < d full-probe refusal fires (and records its justification
  when deliberately overridden);
- the production assembler loads fixtures written in the REAL artifact
  schemas (gen cells, gen manifests, judge cells, objective labels, the
  capture store written through the real ``write_shard``).

All tests are OFFLINE and synthetic: no GPU, no network, no judge API call,
no bank item text (texts are benign templated strings).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
_SCRIPTS = REPO_ROOT / "scripts"
if str(_SCRIPTS) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS))

import issue2658_capture as CAP  # noqa: E402
import issue2658_common as C  # noqa: E402
import issue2658_comparators as U  # noqa: E402
import issue2658_frames as F  # noqa: E402
import issue2658_generate as G  # noqa: E402
import issue2658_judge as J  # noqa: E402
import issue2658_power as PW  # noqa: E402

PART = {"eligible": ["evil"], "not_estimable": ["harmful_compliance"]}


def _ladder(rd, comps, tmp, *, seed=1, allow=None, partition=PART, provider="synthetic"):
    ledger = U.CompLedger(Path(tmp) / "ledger.jsonl")
    counter = U._UnitCounter()
    counter.cap = U.units_for(rd.row, comps, partition)
    if provider == "synthetic":

        def direction_provider(_row):
            assert rd.synthetic_direction is not None
            return rd.synthetic_direction

    else:
        direction_provider = provider
    recs = U.run_ladder(
        rd,
        comps,
        ledger=ledger,
        counter=counter,
        scores_dir=Path(tmp) / "scores",
        embed_backend=U.HashEmbedBackend(),
        partition=partition,
        direction_provider=direction_provider,
        seed=seed,
        allow_underdetermined=allow,
    )
    return recs, ledger, counter


# ---------------------------------------------------------------------------
# 1. Known-signal recovery.
# ---------------------------------------------------------------------------
def test_synthetic_separable_recovers_high_auroc(tmp_path):
    rd = U.synthesize_row_data(
        "evil", n_prompts=48, n_responses=4, d=8, n_superfamilies=16, effect=6.0, seed=7
    )
    recs, _, _ = _ladder(rd, ["c2_direction_dot", "c5_full_probe"], tmp_path)
    by = {r["comparator"]: r for r in recs}
    # C2 scores with the TRUE latent direction: strongly discriminative.
    assert by["c2_direction_dot"]["test_macro_auroc_descriptive"] > 0.85
    # The full probe recovers the separable signal on held-out superfamilies.
    assert by["c5_full_probe"]["test_macro_auroc_descriptive"] > 0.85
    # And the descriptive test macro is the unit-8 metric on the persisted
    # scores (recomputable from the scores file — one metric, never a second).
    z = U.load_unit_scores(by["c5_full_probe"])
    macro, n_disc = PW.equal_prompt_macro_auroc(
        z["test_scores"].astype(np.float64), z["test_labels"], z["test_prompt_ids"]
    )
    assert n_disc > 0
    assert macro == pytest.approx(by["c5_full_probe"]["test_macro_auroc_descriptive"])


def test_null_effect_stays_at_chance(tmp_path):
    rd = U.synthesize_row_data(
        "evil", n_prompts=120, n_responses=4, d=8, n_superfamilies=20, effect=0.0, seed=11
    )
    recs, _, _ = _ladder(rd, ["c2_direction_dot"], tmp_path)
    assert abs(recs[0]["test_macro_auroc_descriptive"] - 0.5) < 0.15


# ---------------------------------------------------------------------------
# 2. C0 cannot see prompt identity.
# ---------------------------------------------------------------------------
def test_c0_features_invariant_to_prompt_identity():
    rd = U.synthesize_row_data("evil", n_prompts=20, n_responses=3, d=4, seed=3)
    dev = [r for r in rd.rows if r.split == "dev"]
    feat = U.NuisanceFeaturizer().fit(dev, scope="dev-fold-train")
    base = feat.transform(dev)
    # Relabel EVERY prompt identity (and superfamily) — features must be
    # bit-identical: the featurizer consumes no identity field.
    renamed = [
        replace(r, prompt_id=f"anon-{i:04d}", superfamily_id=f"sf-anon-{i % 7}")
        for i, r in enumerate(dev)
    ]
    assert np.array_equal(base, feat.transform(renamed))


# ---------------------------------------------------------------------------
# 3. Not-estimable partition (C2/C3).
# ---------------------------------------------------------------------------
def test_not_estimable_rows_emit_the_literal(tmp_path):
    rd = U.synthesize_row_data("harmful_compliance", n_prompts=24, n_responses=3, d=6, seed=5)
    comps = ["c2_direction_dot", "c3_direction_calibrated", "c4_devmean_calibrated"]
    recs, _, _ = _ladder(rd, comps, tmp_path, provider=None)  # no direction needed
    by = {r["comparator"]: r for r in recs}
    for comp in ("c2_direction_dot", "c3_direction_calibrated"):
        assert by[comp]["status"] == "not-estimable"
        assert by[comp]["reason"] == U.NOT_ESTIMABLE
        assert "scores_file" not in by[comp]
        with pytest.raises(U.ComparatorInputError):
            U.load_unit_scores(by[comp])
    # The row stays eligible for the non-direction comparators.
    assert by["c4_devmean_calibrated"]["status"] == "scored"


def test_partition_must_cover_all_rows():
    with pytest.raises(U.ComparatorInputError):
        U.c2c3_partition({"c2_c3_partition": {"eligible": ["evil"], "not_estimable": []}})


def test_not_estimable_literal_matches_provenance():
    import issue2658_provenance as P

    assert U.NOT_ESTIMABLE == P.NOT_ESTIMABLE


# ---------------------------------------------------------------------------
# 4. Dev-only guards FIRING on contaminated fixtures.
# ---------------------------------------------------------------------------
def _mixed_rows():
    rd = U.synthesize_row_data("evil", n_prompts=20, n_responses=3, d=4, seed=9)
    return rd, rd.rows  # rows include BOTH dev and test splits


def test_zscore_fires_on_test_contaminated_fit():
    rd, mixed = _mixed_rows()
    with pytest.raises(C.TestDerivedTransformError):
        U.ZScore.fit(rd.X, mixed, scope="dev-fold-train")


def test_zscore_fires_on_peer_centering_scope():
    rd, _ = _mixed_rows()
    dev = [r for r in rd.rows if r.split == "dev"]
    with pytest.raises(C.PeerCenteringError):
        U.ZScore.fit(rd.X[rd.mask("dev")], dev, scope="pooled-eval-peers")


def test_text_featurizer_fires_on_test_contaminated_fit():
    _, mixed = _mixed_rows()
    with pytest.raises(C.TestDerivedTransformError):
        U.TextFeaturizer("answer", U.HashEmbedBackend()).fit(mixed, scope="dev-fold-train")


def test_nuisance_featurizer_fires_on_test_contaminated_fit():
    _, mixed = _mixed_rows()
    with pytest.raises(C.TestDerivedTransformError):
        U.NuisanceFeaturizer().fit(mixed, scope="dev-fold-train")


def test_devmean_direction_fires_on_test_contaminated_fit():
    rd, mixed = _mixed_rows()
    y = np.array([r.label for r in mixed], dtype=bool)
    with pytest.raises(C.TestDerivedTransformError):
        U.devmean_direction(rd.X, mixed, y, scope="dev-fold-train")


def test_rowdata_fires_on_superfamily_leakage():
    rd = U.synthesize_row_data("evil", n_prompts=20, n_responses=3, d=4, seed=13)
    rows = list(rd.rows)
    # Contaminate: relabel one TEST row's superfamily to a DEV superfamily.
    dev_sf = next(r.superfamily_id for r in rows if r.split == "dev")
    i = next(i for i, r in enumerate(rows) if r.split == "test")
    rows[i] = replace(rows[i], superfamily_id=dev_sf)
    with pytest.raises(C.DependencyCrossingError):
        U.RowData(
            row="evil",
            X=rd.X,
            rows=rows,
            data_fingerprint="x",
            label_source="synthetic",
        )


def test_join_labels_missing_label_fires_and_exclusions_count():
    keys = [("p0", 0), ("p0", 1), ("p1", 0)]
    records = {
        ("p0", 0): {"label": True, "status": "scored"},
        ("p0", 1): {"label": None, "status": "human_adjudication"},
    }
    with pytest.raises(C.MissingLabelError):
        U.join_labels(keys, records)  # ("p1", 0) silently absent -> RAISE
    records[("p1", 0)] = {"label": False, "status": "scored"}
    kept, excluded = U.join_labels(keys, records)
    assert kept == {("p0", 0): True, ("p1", 0): False}
    assert excluded == {"human_adjudication": 1}


def test_join_labels_coerced_label_fires():
    keys = [("p0", 0)]
    with pytest.raises(C.CoercedLabelError):
        U.join_labels(keys, {("p0", 0): {"label": 0.7, "status": "scored"}})


# ---------------------------------------------------------------------------
# 5. Folds never split a superfamily.
# ---------------------------------------------------------------------------
def test_folds_never_split_a_superfamily():
    rd = U.synthesize_row_data(
        "evil", n_prompts=60, n_responses=3, d=4, n_superfamilies=17, seed=21
    )
    dev = [r for r in rd.rows if r.split == "dev"]
    assignment = U.superfamily_folds(dev, U.N_FOLDS)
    assert set(assignment.values()) <= set(range(U.N_FOLDS))
    # Row-level fold labels never split a superfamily across folds.
    folds_of_sf: dict[str, set[int]] = {}
    for r in dev:
        folds_of_sf.setdefault(r.superfamily_id, set()).add(assignment[r.superfamily_id])
    assert all(len(v) == 1 for v in folds_of_sf.values())
    per_fold = {f: 0 for f in range(U.N_FOLDS)}
    for r in dev:
        per_fold[assignment[r.superfamily_id]] += 1
    assert all(v > 0 for v in per_fold.values())


def test_fold_construction_fires_below_n_folds_superfamilies():
    rd = U.synthesize_row_data("evil", n_prompts=20, n_responses=3, d=4, n_superfamilies=7, seed=2)
    dev = [r for r in rd.rows if r.split == "dev"]
    few = [r for r in dev if r.superfamily_id in sorted({x.superfamily_id for x in dev})[:3]]
    with pytest.raises(U.FoldConstructionError):
        U.superfamily_folds(few, U.N_FOLDS)


# ---------------------------------------------------------------------------
# 6. Strongest-regularization tie-break within 1e-4.
# ---------------------------------------------------------------------------
def test_tie_break_selects_strongest_within_tol():
    base = dict.fromkeys(U.C_GRID, 0.5)
    table = dict(base)
    table[1e2] = 0.80003  # best
    table[1e-4] = 0.80000  # within 1e-4 of best
    table[1e-6] = 0.79996  # within 1e-4 of best -> strongest wins
    assert U.select_c(table) == 1e-6
    table2 = dict(base)
    table2[1e2] = 0.80020  # best, and NOTHING within 1e-4
    table2[1e-6] = 0.80000
    assert U.select_c(table2) == 1e2


def test_select_c_requires_the_registered_grid():
    with pytest.raises(U.ComparatorInputError):
        U.select_c({1.0: 0.7})


# ---------------------------------------------------------------------------
# 7. Pooled cross-fold path is a structural dead end.
# ---------------------------------------------------------------------------
def test_pooled_fold_path_structurally_impossible():
    rng = np.random.default_rng(0)
    # Two folds whose pooled concatenation differs from the per-fold mean:
    # fold scales differ, so pooled ranks mix folds.
    s1, l1 = rng.normal(size=8), np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=bool)
    s2, l2 = rng.normal(size=8) * 100 + 500, np.array([0, 1, 0, 1, 0, 1, 0, 1], dtype=bool)
    p1 = np.array(["a"] * 4 + ["b"] * 4)
    p2 = np.array(["c"] * 4 + ["d"] * 4)
    fp = U.FoldPredictions(fold_ids=(0, 1), scores=(s1, s2), labels=(l1, l2), prompt_ids=(p1, p2))
    with pytest.raises(C.PooledFoldMetricError):
        fp.pooled()
    with pytest.raises(C.PooledFoldMetricError):
        C.assert_not_pooled_fold("pooled-cross-fold")
    got = U.macro_selection_metric(fp)
    m1, _ = PW.equal_prompt_macro_auroc(s1, l1, p1)
    m2, _ = PW.equal_prompt_macro_auroc(s2, l2, p2)
    assert got == pytest.approx((m1 + m2) / 2.0)


# ---------------------------------------------------------------------------
# 8. Resume: completed units skip; a changed generating parameter recomputes.
# ---------------------------------------------------------------------------
def test_resume_skips_completed_units(tmp_path, capsys):
    rd = U.synthesize_row_data("evil", n_prompts=24, n_responses=3, d=6, seed=17)
    comps = ["c4_devmean_calibrated", "c2_direction_dot"]
    _, _ledger, counter = _ladder(rd, comps, tmp_path, seed=17)
    assert counter.resumed == 0
    n_lines = len([ln for ln in (tmp_path / "ledger.jsonl").read_text().splitlines() if ln.strip()])
    # Fresh process-equivalent: rebuild the ledger from disk and re-run.
    _, _, counter2 = _ladder(rd, comps, tmp_path, seed=17)
    assert counter2.resumed == counter2.done == n_lines
    assert "resume-skip" in capsys.readouterr().out
    n_lines2 = len(
        [ln for ln in (tmp_path / "ledger.jsonl").read_text().splitlines() if ln.strip()]
    )
    assert n_lines2 == n_lines
    # A changed generating parameter (seed) recomputes every unit.
    _, _, counter3 = _ladder(rd, comps, tmp_path, seed=18)
    assert counter3.resumed == 0


def test_resume_missing_scores_file_fails_loud(tmp_path):
    rd = U.synthesize_row_data("evil", n_prompts=24, n_responses=3, d=6, seed=19)
    _, _, _ = _ladder(rd, ["c4_devmean_calibrated"], tmp_path, seed=19)
    for p in (tmp_path / "scores").glob("evil__c4_devmean_calibrated__fold0.npz"):
        p.unlink()
    with pytest.raises(C.CacheStaleError):
        _ladder(rd, ["c4_devmean_calibrated"], tmp_path, seed=19)


# ---------------------------------------------------------------------------
# 9. n_train < d refusal (C5 well-posedness).
# ---------------------------------------------------------------------------
def test_underdetermined_refusal_fires_and_justification_records(tmp_path):
    rd = U.synthesize_row_data(
        "evil", n_prompts=24, n_responses=3, d=128, n_superfamilies=12, seed=23
    )
    with pytest.raises(U.UnderdeterminedFitError):
        _ladder(rd, ["c5_full_probe"], tmp_path)
    recs, _, _ = _ladder(
        rd, ["c5_full_probe"], tmp_path / "ok", allow="test shape: deliberate n_train < d"
    )
    wp = recs[0]["wellposed"]
    assert wp["underdetermined"] is True
    assert wp["justification"] == "test shape: deliberate n_train < d"
    assert wp["n_train"] < wp["d"]


def test_wellposed_passes_without_justification_when_n_train_ge_d():
    out = U.assert_probe_wellposed(100, 8, None)
    assert out["underdetermined"] is False and out["justification"] is None


# ---------------------------------------------------------------------------
# 10. Production assembler against REAL-schema fixtures.
# ---------------------------------------------------------------------------
def _write_fixture(out_root: Path, monkeypatch, *, poison_split: bool = False) -> dict:
    """Tiny two-split fixture in the REAL artifact schemas for row 'evil'.

    Superfamilies sf-a..sf-f dev, sf-t0/sf-t1 test; one human_adjudication
    verdict (excluded, counted); capture store written through the REAL
    ``write_shard`` so the real reader/verifier bodies execute.
    """
    row = "evil"
    rng = np.random.default_rng(31)
    # base_split drives the ARTIFACT layout; sf_split is what the split
    # MANIFEST claims. poison_split makes them disagree on sf-a (the artifacts
    # keep sf-a in dev while the manifest claims test -> DependencyCrossing).
    base_split = {f"sf-{c}": "dev" for c in "abcdef"} | {"sf-t0": "test", "sf-t1": "test"}
    sf_split = dict(base_split)
    prompts = {  # prompt_id -> superfamily
        **{f"evil__frameA__d{i}": f"sf-{c}" for i, c in enumerate("abcdef")},
        **{f"evil__frameA__d{i + 6}": f"sf-{c}" for i, c in enumerate("abcdef")},
        "evil__frameB__t0": "sf-t0",
        "evil__frameB__t1": "sf-t1",
        "evil__frameB__t2": "sf-t0",
        "evil__frameB__t3": "sf-t1",
    }
    if poison_split:
        sf_split["sf-a"] = "test"  # manifest says test; artifacts realize it in dev
    # Split-manifest fixture + text-resolver stub (synthetic ids have no pins).
    split_manifest = {
        "rows": [{"row": row, "superfamily_splits": sf_split}],
        "manifest_kind": "split",
    }
    smp = out_root / "split_manifest.json"
    smp.write_text(json.dumps(split_manifest))
    monkeypatch.setattr(F, "SPLIT_MANIFEST_PATH", smp)
    import issue2658_text_resolver as R

    monkeypatch.setattr(
        R,
        "resolve_items",
        lambda ids, verify_pins=True: {
            i: SimpleNamespace(text=f"benign fixture question {i}") for i in ids
        },
    )
    # Round 18: the assembler resolves through the shared split-aware helper
    # (synthetic ids have no pins and no frozen production selection).
    monkeypatch.setattr(
        G,
        "resolve_items_for_split",
        lambda ids, split, *, eval_root=None: {
            i: SimpleNamespace(text=f"benign fixture question {i}") for i in ids
        },
    )
    expected: dict = {"labels": {}, "vectors": {}, "excluded": 0}
    for split in ("dev", "test"):
        # Artifacts always follow base_split (the un-poisoned layout).
        split_prompts = [p for p, sf in prompts.items() if base_split[sf] == split]
        records, manifest_rows, verdicts = [], [], []
        vectors, metas = [], []
        for p in sorted(split_prompts):
            for k in range(2):
                text = f"benign fixture answer {p} draw {k}"
                sha = F._sha_text(text)
                records.append(
                    {
                        "prompt_id": p,
                        "response_index": k,
                        "seed": 1,
                        "realized_seed": 1,
                        "n_empty_retries": 0,
                        "finish_reason": "stop" if k == 0 else "length",
                        "n_prompt_tokens": 30 + k,
                        "n_completion_tokens": 50 + k,
                        "answer_sha256": sha,
                        "raw_text_sha256": sha,
                        "text": text,
                    }
                )
                manifest_rows.append(
                    {
                        "manifest_version": 1,
                        "row": row,
                        "split": split,
                        "prompt_id": p,
                        "response_index": k,
                        "superfamily_id": prompts[p],
                        "source_frame": p.split("__")[1],
                        "stratum": "band0",
                        "answer_sha256": sha,
                    }
                )
                # One excluded verdict; every other answer scored.
                if split == "dev" and p.endswith("d0") and k == 1:
                    verdicts.append(
                        {
                            "item_id": p,
                            "response_index": k,
                            "judge_status": "human_adjudication",
                            "binary_label": None,
                        }
                    )
                    expected["excluded"] += 1
                else:
                    lab = bool((sum(p.encode()) + k) % 2)  # deterministic across processes
                    verdicts.append(
                        {
                            "item_id": p,
                            "response_index": k,
                            "judge_status": "scored",
                            "binary_label": lab,
                        }
                    )
                    expected["labels"][(p, k)] = lab
                vec = rng.normal(size=C.HIDDEN).astype(np.float32)
                vectors.append(vec)
                metas.append({"prompt_id": p, "response_index": k})
                expected["vectors"][(p, k)] = vec
        gen_dir = out_root / "raw_completions" / split
        gen_dir.mkdir(parents=True, exist_ok=True)
        (gen_dir / f"{row}__frameA__band0.json").write_text(
            json.dumps({"schema": G.GEN_SCHEMA, "records": records})
        )
        man_dir = out_root / "gen_manifest" / split
        man_dir.mkdir(parents=True, exist_ok=True)
        with (man_dir / f"{row}__frameA__band0.jsonl").open("w") as fh:
            for m in manifest_rows:
                fh.write(json.dumps(m) + "\n")
        judge_dir = out_root / "judge" / split / row
        judge_dir.mkdir(parents=True, exist_ok=True)
        (judge_dir / f"{row}__frameA__band0.json").write_text(
            json.dumps({"schema": J.JUDGE_SCHEMA, "verdicts": verdicts})
        )
        store = out_root / "l19_store" / split / "shard00of01"
        CAP.write_shard(
            store,
            0,
            vectors,
            metas,
            CAP.capture_fingerprint(split, dtype="bfloat16", device="cuda"),
        )
    return expected


def test_assembler_loads_real_schema_fixture(tmp_path, monkeypatch):
    expected = _write_fixture(tmp_path, monkeypatch)
    rd, report = U.assemble_row_data("evil", tmp_path)
    assert rd.label_source == "judge-cells"
    assert report["exclusions"]["dev"] == {"human_adjudication": 1}
    assert len(rd.rows) == len(expected["labels"])
    for i, r in enumerate(rd.rows):
        assert r.label == expected["labels"][(r.prompt_id, r.response_index)]
        assert np.array_equal(rd.X[i], expected["vectors"][(r.prompt_id, r.response_index)])
        assert r.truncated == (r.response_index == 1)  # finish_reason == "length"
    assert {r.split for r in rd.rows} == {"dev", "test"}


def test_assembler_fires_on_split_manifest_mismatch(tmp_path, monkeypatch):
    _write_fixture(tmp_path, monkeypatch, poison_split=True)
    with pytest.raises(C.DependencyCrossingError):
        U.assemble_row_data("evil", tmp_path)


def test_assembler_fires_on_vector_sha_mismatch(tmp_path, monkeypatch):
    _write_fixture(tmp_path, monkeypatch)
    store = tmp_path / "l19_store" / "dev" / "shard00of01"
    npy = store / "l19mean_shard00.npy"
    arr = np.load(npy)
    arr[0, 0] += 1.0  # tamper one value -> row_index sha no longer matches
    with npy.open("wb") as fh:
        np.save(fh, arr)
    with pytest.raises(C.RowHashMismatchError):
        U.assemble_row_data("evil", tmp_path)


def test_assembler_routes_prompt_resolution_through_shared_split_resolver(tmp_path, monkeypatch):
    """Round 18: assemble_row_data resolves prompt text through the shared
    ``G.resolve_items_for_split``, threading the loop split + its own out_root."""
    _write_fixture(tmp_path, monkeypatch)
    calls = []

    def rec(ids, split, *, eval_root=None):
        calls.append((split, eval_root))
        return {i: SimpleNamespace(text=f"benign fixture question {i}") for i in ids}

    monkeypatch.setattr(G, "resolve_items_for_split", rec)
    U.assemble_row_data("evil", tmp_path)
    assert [c[0] for c in calls] == ["dev", "test"]
    assert all(c[1] == tmp_path for c in calls)  # eval_root == the assembler's out_root


# ---------------------------------------------------------------------------
# 11. Frozen-direction contract (sign + hash pin).
# ---------------------------------------------------------------------------
def _direction_entry(vec: np.ndarray) -> dict:
    import hashlib

    return {
        "row": "evil",
        "c2_c3": "eligible",
        "sign_convention": "+dot = evil-trait-expressing",
        "vector_sha256": hashlib.sha256(
            np.ascontiguousarray(vec, dtype=np.float32).tobytes()
        ).hexdigest(),
    }


def test_verify_direction_entry_contract():
    vec = np.random.default_rng(1).normal(size=C.HIDDEN).astype(np.float32)
    entry = _direction_entry(vec)
    out = U.verify_direction_entry(entry, vec)
    assert out.dtype == np.float32 and out.shape == (C.HIDDEN,)
    with pytest.raises(U.SignConventionError):  # hash pin: never a proxy direction
        U.verify_direction_entry(entry, vec + 1.0)
    flipped = dict(entry, sign_convention="-dot = evil (post-hoc flip)")
    with pytest.raises(U.SignConventionError):  # preregistered sign: never selection
        U.verify_direction_entry(flipped, vec)
    ineligible = dict(entry, c2_c3=U.NOT_ESTIMABLE)
    with pytest.raises(U.SignConventionError):
        U.verify_direction_entry(ineligible, vec)


# ---------------------------------------------------------------------------
# 12. Production C1 dependency resolves from the lockfile environment.
# ---------------------------------------------------------------------------
def test_minilm_dependency_resolvable_offline():
    assert importlib.util.find_spec("sentence_transformers") is not None
