"""Numerical and persistence contracts for the assistant-story extension."""

from pathlib import Path
from unittest.mock import create_autospec

import numpy as np
import pytest
import torch

from scripts import issue2054_k5_assistant_story_analysis as analysis


def moments(x, y):
    panel = {"x": x, "y": y, "membership": np.arange(len(x)) % 5}
    value = analysis.geometry.moments_by_fold(panel)
    value["yss"] = torch.tensor(
        [np.square(y[panel["membership"] == f]).sum() for f in range(5)], dtype=torch.float64
    )
    return value


def test_batched_gcv_matches_parent_oracle_and_fold_exclusion():
    rng = np.random.default_rng(2045)
    x = rng.normal(size=(90, 4)) * [0.2, 2, 7, 11] + [25, -10, 3, 0]
    y = x @ rng.normal(size=(4, 4)) + rng.normal(size=(90, 4)) * 2
    moment = moments(x, y)
    maps, biases, identity, infos = analysis.gcv_maps(moment, [0, 2, 4])
    for i, fold in enumerate([0, 2, 4]):
        m = analysis.train_moments(moment, [fold])
        oracle = analysis.PooledMomentRidge(
            n=int(m["n"][0]),
            sum_x=m["sx"][0],
            sum_y=m["sy"][0],
            yss=float(m["yss"][0]),
            c_xx=m["xx"][0],
            c_xy=m["xy"][0],
        )
        assert infos[i]["best_lambda"] == oracle.best_lambda
        assert infos[i]["dof"] == pytest.approx(oracle.dof, rel=1e-10)
        np.testing.assert_allclose(
            (analysis.tensor(x) @ maps[i] + biases[i]).numpy(),
            oracle.predict_np(x),
            rtol=1e-10,
            atol=1e-9,
        )
        train = np.arange(len(x)) % 5 != fold
        np.testing.assert_allclose(identity[i], (y[train] - x[train]).mean(0))
    changed = y.copy()
    changed[np.arange(len(x)) % 5 == 0] += 100
    remaps, rebias, _, reinfo = analysis.gcv_maps(moments(x, changed), [0])
    np.testing.assert_allclose(remaps[0], maps[0], rtol=1e-9, atol=1e-9)
    np.testing.assert_allclose(rebias[0], biases[0], rtol=1e-9, atol=1e-9)
    assert reinfo[0]["best_lambda"] == infos[0]["best_lambda"]


def test_batched_retrieval_matches_parent_ties_and_block_boundaries():
    rng = np.random.default_rng(4)
    y = rng.normal(size=(17, 6))
    y[1] = y[0]
    predictions = np.stack([y, y[::-1], np.repeat(y.mean(0)[None], len(y), axis=0)])
    actual = analysis.score_batch(predictions, y, block=3)
    for i, prediction in enumerate(predictions):
        expected = analysis.base.score(prediction, y)
        assert actual[i]["r2"] == pytest.approx(expected["r2"], abs=1e-12)
        for metric in ("euclidean", "cosine"):
            for k in (1, 5, 10):
                assert (
                    actual[i]["retrieval"][metric]["acc_at_k"][str(k)]
                    == expected["retrieval"][metric]["acc_at_k"][k]
                )
            assert actual[i]["retrieval"][metric]["mrr"] == pytest.approx(
                expected["retrieval"][metric]["mrr"], abs=1e-12
            )


def test_response_correction_is_mean_shift_minus_sample_variance_and_stays_signed():
    rng = np.random.default_rng(8)
    s = rng.normal(size=(20, 5, 4)) + rng.normal(size=(20, 1, 4))
    c = rng.normal(size=(20, 5, 4)) + rng.normal(size=(20, 1, 4))
    fold = np.arange(20) % 5
    actual = analysis.response_statistics(s, c, s.mean(1), c.mean(1), fold, block=3)
    expected = (
        np.square(s.mean(1) - c.mean(1)).sum(1)
        - (s.var(1, ddof=1).sum(1) + c.var(1, ddof=1).sum(1)) / 5
    )
    np.testing.assert_allclose(
        actual["corrected_mean_response_squared_displacement"], expected, atol=1e-12
    )
    identical = analysis.response_statistics(s, s, s.mean(1), s.mean(1), fold)
    assert np.all(identical["corrected_mean_response_squared_displacement"] < 0)
    np.testing.assert_allclose(identical["mean_answer_squared_distance"], 0, atol=1e-12)
    np.testing.assert_allclose(identical["centered_mean_answer_cosine"], 1, atol=1e-12)
    for f in range(5):
        train, test = fold != f, fold == f
        fixed = (
            np.square(s.mean(1)[train]).sum(1).mean() + np.square(c.mean(1)[train]).sum(1).mean()
        ) / 2
        np.testing.assert_allclose(actual["training_normalization_scale"][test], fixed)


def test_query_audit_requires_full_question_and_single_chat_turn():
    def row(prefix):
        return {"final_text": prefix + "answer", "answer_start": len(prefix)}

    model = "qwen2.5-7b-instruct"
    chat = f"{analysis.CHAT}__{model}"
    story = f"{analysis.STORY}__{model}"
    data = {
        chat: {"a": row("<|im_start|>user\nfull question?<|im_end|>\n<|im_start|>assistant\n")},
        story: {"a": row('Narrative. full  question? Assistant replied: "')},
    }
    queries, valid, audit = analysis.query_matches(data, model)
    assert queries["a"] == "full question?"
    assert valid[story] == {"a"}
    assert next(r for r in audit if r["cell"] == story)["whitespace_query"] == 1
    data[story]["a"] = row("Only question? Assistant replied: ")
    assert analysis.query_matches(data, model)[1][story] == set()
    data[chat]["a"] = row("<|im_start|>system\nHistory\n" + data[chat]["a"]["final_text"])
    with pytest.raises(RuntimeError, match="single user turn"):
        analysis.query_matches(data, model)


def test_packet_names_preserve_model_periods_and_resume_checks_hashes(tmp_path, monkeypatch):
    def seal(paths, root, fingerprint):
        for path in paths:
            path = Path(path)
            analysis.k3.atomic_json(
                path.with_suffix(path.suffix + ".done.json"),
                {
                    "path": str(path.relative_to(root)),
                    "sha256": analysis.k3.sha(path),
                    "fingerprint": fingerprint,
                    "size": path.stat().st_size,
                },
            )

    monkeypatch.setattr(
        analysis.artifacts,
        "seal_many",
        create_autospec(analysis.artifacts.seal_many, side_effect=seal),
    )
    for fold in (0, 1):
        stem = f"maps/qwen2.5-7b__fold{fold}"
        actual = analysis.save_packet(tmp_path, stem, {"x": np.array([fold])}, {"fold": fold}, "fp")
        assert actual["array_path"] == f"analysis/{stem}.npz"
        assert analysis.resume_packet(tmp_path, stem, "fp")["fold"] == fold
    target = tmp_path / "analysis/maps/qwen2.5-7b__fold0.npz"
    target.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="content mismatch"):
        analysis.resume_packet(tmp_path, "maps/qwen2.5-7b__fold0", "fp")


def test_fingerprint_ignores_resume_timing_but_binds_generation_content():
    fields = {
        "source_sha": "a" * 40,
        "manifest_sha256": "m",
        "inventory_sha256": "i",
        "selected_cells": ["cell"],
        "parent_revision": "p",
        "started": 1,
        "finished": 2,
    }
    first = analysis.fingerprint({"cells": []}, fields)
    fields.update(started=30, finished=50, outputs_revision="new-publication")
    assert analysis.fingerprint({"cells": []}, fields) == first
    fields["inventory_sha256"] = "changed"
    assert analysis.fingerprint({"cells": []}, fields) != first


def test_load_draws_verifies_real_capture_mean_text_seed_and_cap(tmp_path):
    rng = np.random.default_rng(21)
    cell = f"{analysis.STORY}__qwen2.5-7b"
    ids = ["first", "second"]
    vectors = rng.normal(size=(2, 5, 3584)).astype(np.float16)
    x = rng.normal(size=(2, 3584)).astype(np.float16)
    old = {
        "conv_id": np.array(ids),
        "v_A_0": vectors[:, 0],
        "v_A_12": vectors[:, 1:3],
        "valid_draws_12": np.ones((2, 2), dtype=bool),
        "cap_mask": np.zeros((2, 3), dtype=bool),
        "v_C": x,
    }
    new = {
        "conv_id": np.array(ids),
        "v_A_34": vectors[:, 3:],
        "valid_draws_34": np.ones((2, 2), dtype=bool),
        "cap_mask_34": np.zeros((2, 2), dtype=bool),
    }
    original = {}
    prefix, suffix = 'Question. Assistant replied: "', '"'
    for cid in ids:
        original[cid] = {
            "conv_id": cid,
            "answer": cid,
            "final_text": prefix + cid + suffix,
            "answer_start": len(prefix),
            "answer_end": len(prefix) + len(cid),
        }

    def write_raw(name, draws, fp, bad_cap=False):
        rows = []
        for cid in ids:
            for draw in draws:
                answer = f"{cid} draw {draw}"
                rows.append(
                    {
                        "conv_id": cid,
                        "draw": draw,
                        "answer": answer,
                        "final_text": prefix + answer + suffix,
                        "answer_start": len(prefix),
                        "answer_end": len(prefix) + len(answer),
                        "seed": analysis.k3.seed(cell, cid, draw)
                        if draw < 3
                        else analysis.k5.seed(cell, cid, draw),
                        "max_tokens_budget": 999 if bad_cap else 2048,
                        "finish_reason": "stop",
                    }
                )
        shard = tmp_path / f"{name}_part0.json"
        analysis.k3.atomic_json(shard, rows)
        analysis.k3.atomic_json(
            shard.with_suffix(".json.done.json"),
            {
                "sha256": analysis.k3.sha(shard),
                "fingerprint": fp,
            },
        )
        path = tmp_path / f"{name}.json"
        analysis.k3.atomic_json(path, {"shards": [shard.name], "rows": len(rows)})
        return str(path)

    old_path, new_path = tmp_path / "old.npz", tmp_path / "new.npz"
    np.savez(old_path, **old)
    np.savez(new_path, **new)
    chunk = {
        "offset": 0,
        "old_capture": str(old_path),
        "new_capture": str(new_path),
        "old_raw": write_raw("old", (1, 2), "old-fp"),
        "new_raw": write_raw("new", (3, 4), "new-fp"),
        "old_raw_fingerprint": "old-fp",
        "new_fingerprint": "new-fp",
    }
    targets, _, caps = analysis.k5.average_targets(old, new)
    panel = {"ids": ids, "x": x, "y": targets[5], "caps": caps}
    restored, text = analysis.load_draws(tmp_path, [chunk], panel, original, cell)
    np.testing.assert_array_equal(restored, vectors)
    assert text["first"]["answers"] == ["first"] + [f"first draw {d}" for d in range(1, 5)]
    write_raw("new", (3, 4), "new-fp", bad_cap=True)
    with pytest.raises(RuntimeError, match="seed or generation cap"):
        analysis.load_draws(tmp_path, [chunk], panel, original, cell)
