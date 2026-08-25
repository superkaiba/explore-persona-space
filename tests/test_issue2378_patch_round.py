"""Pure-logic tests for the #2378 `causal-patching-arms` round.

CPU-only, no network, repo-root paths (adoptable-test shape). The model path
is covered by the driver's --tiny e2e + the pod-side bank gates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(REPO_ROOT / "scripts"), str(REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue2378_patch_common as pc  # noqa: E402

QIDS = {
    "Vex": [f"storyq_vex_{k:02d}" for k in range(4)],
    "Wren": [f"storyq_wren_{k:02d}" for k in range(4)],
}


def test_enumerate_cells_grid_shape():
    cells = pc.enumerate_cells(QIDS)
    # per qid: steered 2var x 2pair x 2dir = 8; null 8; within 1var x 4 = 4;
    # prefill 4 -> 24 cells; 8 qids -> 192.
    assert len(cells) == 24 * 8
    ids = [c["cell_id"] for c in cells]
    assert len(set(ids)) == len(ids), "cell ids must be unique"
    by_arm = {}
    for c in cells:
        by_arm.setdefault((c["arm"], c["variant"]), 0)
        by_arm[(c["arm"], c["variant"])] += 1
    assert by_arm[("steered", "lstar")] == by_arm[("steered", "all")] == 8 * 4
    assert by_arm[("within", "lstar")] == 8 * 4
    assert by_arm[("prefill", "none")] == 8 * 4
    # chat~story families carry the character; chat~plain carry "-".
    for c in cells:
        char_slot = c["family"].split("|")[1]
        assert char_slot == (c["char"] if c["pair_type"] == "chat~story" else "-")
    # directions map src/tgt onto the pair's contexts.
    for c in cells:
        chat_c, other_c = pc.pair_contexts(c["pair_type"], c["qid"])
        if c["direction"] == "a2b":
            assert (c["src"], c["tgt"]) == (chat_c, other_c)
        else:
            assert (c["src"], c["tgt"]) == (other_c, chat_c)


def test_derangement_no_fixed_points_and_deterministic():
    qids = [f"q{k}" for k in range(12)]
    d1 = pc.derangement(qids, ("story", "Vex"))
    d2 = pc.derangement(list(reversed(qids)), ("story", "Vex"))
    assert d1 == d2, "seeded map must be input-order independent"
    assert set(d1) == set(qids) and set(d1.values()) == set(qids)
    assert all(k != v for k, v in d1.items()), "derangement must have no fixed points"
    assert pc.derangement(qids, ("chat", "Vex")) != d1, "grain seeds must differ"
    with pytest.raises(AssertionError):
        pc.derangement(["only"], ("story", "Vex"))


def test_read_layers_port_of_2094_rule():
    assert pc.primary_read_layer(64) == 59
    assert pc.primary_read_layer(28) == 26  # the #2094 original
    assert pc.read_layers(51, 64) == (51, 59)
    with pytest.raises(AssertionError):
        pc.read_layers(60, 64)  # primary must sit strictly downstream


def test_screen_families_pass_fail_and_floor():
    strong = {f"q{k}": 0.5 + 0.01 * k for k in range(10)}
    nullish = {f"q{k}": (-1) ** k * 0.02 for k in range(10)}
    thin = {"q0": 0.9, "q1": 0.8}  # below MIN_PAIRS
    fams = {
        "chat~story|Vex|a2b|lstar|steered": strong,
        "chat~story|Wren|a2b|lstar|steered": nullish,
        "chat~plain|-|a2b|lstar|steered": thin,
    }
    rep = pc.screen_families(fams, n_boot=2000)
    assert rep["families"]["chat~story|Vex|a2b|lstar|steered"]["screen_pass"] is True
    assert rep["families"]["chat~story|Wren|a2b|lstar|steered"]["screen_pass"] is False
    assert rep["skipped_below_min_pairs"] == ["chat~plain|-|a2b|lstar|steered"]
    assert rep["confirm_families"] == ["chat~story|Vex|a2b|lstar|steered"]
    rec = rep["families"]["chat~story|Vex|a2b|lstar|steered"]
    assert rec["ci_lo"] > 0 and rec["ci_lo"] <= rec["mean_diff"] <= rec["ci_hi"]


def test_extract_answer_stop_conventions():
    import issue2378_patch_run as run

    ans, drop = run._extract_answer("story", 'I will crush them." She left.')
    assert (ans, drop) == ("I will crush them.", None)
    assert run._extract_answer("story", "never closes the quote")[1] == "cap_hit_no_close"
    ans, drop = run._extract_answer("plain", "Paris is nice.\n\nUser: next question")
    assert (ans, drop) == ("Paris is nice.", None)
    assert run._extract_answer("plain", "  ")[1] == "empty_answer"
    assert run._extract_answer("chat", " hi there ") == ("hi there", None)


def test_cell_and_family_keys_roundtrip():
    assert pc.ctx_id("chat", "storyq_vex_00") == "chat:storyq_vex_00"
    fam = pc.family_key("chat~story", "Vex", "a2b", "lstar", "steered")
    assert fam.rsplit("|", 1)[0] + "|null" == pc.family_key(
        "chat~story", "Vex", "a2b", "lstar", "null"
    )


# ── r18 review-fix invariants ────────────────────────────────────────────────


def test_screen_families_disjoint_qid_groups_fixed_n():
    """Disjoint-qid families bootstrap over their OWN fixed pair set (r17
    codex patch-bootstrap-variable-effective-n): each family's CI equals a
    direct dense per-group helper call — never the union-with-NaN matrix."""
    import issue2094_analysis as A
    import numpy as np

    fam_a = {f"a{k}": 0.4 + 0.05 * k for k in range(4)}
    fam_b = {f"b{k}": (-1) ** k * 0.3 for k in range(4)}
    diffs = {
        "chat~story|Vex|a2b|lstar|steered": fam_a,
        "chat~story|Wren|a2b|lstar|steered": fam_b,
    }
    rep = pc.screen_families(diffs, n_boot=500)
    group_keys = sorted(tuple(sorted(d)) for d in (fam_a, fam_b))
    for fam, d in diffs.items():
        gi = group_keys.index(tuple(sorted(d)))
        values = np.array([[d[q]] for q in sorted(d)])
        boots = A.bootstrap_family_means_batched(values, 500, pc.PATCH_BOOTSTRAP_SEED + gi)
        col = boots[:, 0]
        assert not np.isnan(col).any(), "dense group matrix — fixed effective n, no NaN draws"
        assert rep["families"][fam]["ci_lo"] == pytest.approx(float(np.percentile(col, 2.5)))
        assert rep["families"][fam]["ci_hi"] == pytest.approx(float(np.percentile(col, 97.5)))
        assert rep["families"][fam]["n_pairs"] == 4


def test_stop_strings_and_hit_stop():
    assert pc.stop_strings_for("chat") is None
    assert pc.stop_strings_for("story") == ['"', "”"]
    assert pc.stop_strings_for("plain") == ["\nUser:"]
    assert pc.hit_stop("story", 'I will win." She left.') is True
    assert pc.hit_stop("story", "never closes") is False
    assert pc.hit_stop("plain", "Paris.\n\nUser: next") is True
    assert pc.hit_stop("plain", "Paris.") is False
    assert pc.hit_stop("chat", 'quotes "inside" are fine') is False


class _SeamTok:
    """Split decode differs from combined decode at the [1, 2] seam —
    the #2333 cleanup-space/multi-byte corruption class."""

    def decode(self, ids, skip_special_tokens=True):
        out, i = [], 0
        while i < len(ids):
            if ids[i] == 1 and i + 1 < len(ids) and ids[i + 1] == 2:
                out.append("ab")
                i += 2
            elif ids[i] == 1:
                out.append("a ")
                i += 1
            elif ids[i] == 2:
                out.append(" b")
                i += 1
            else:
                out.append("x")
                i += 1
        return "".join(out)


def test_prefill_row_one_shot_decode_and_exact_ids():
    """The prefill reply is the ONE-SHOT decode of [opener+gen] ids and the
    exact combined ids ride _capture_ids (r17 codex
    patch-prefill-token-identity-loss)."""
    import issue2378_patch_run as run

    tok = _SeamTok()
    cell = {
        k: f"<{k}>"
        for k in (
            "cell_id",
            "arm",
            "variant",
            "src",
            "tgt",
            "qid",
            "char",
            "pair_type",
            "direction",
            "family",
            "donor_ctx",
        )
    }
    gen_row = {"gen_ids": [2, 9], "hit_eos": True}
    row = run._prefill_row(tok, cell, "chat", [7, 7, 7], [1], gen_row, d=0)
    assert row["gen_text"] == "abx"  # one-shot; split decode would give "a  bx"
    assert tok.decode([1]) + tok.decode([2, 9]) == "a  bx"
    assert row["_capture_ids"] == {"prompt": [7, 7, 7], "completion": [1, 2, 9]}
    assert row["n_completion_tokens"] == 3
    rec = run._prefill_capture_rec(0, [7, 7, 7], [1, 2, 9])
    assert rec["input_ids"] == [7, 7, 7, 1, 2, 9]
    assert (rec["ans_lo"], rec["ans_hi"], rec["v_C_pos"]) == (3, 6, 2)


def test_ensure_mctx_single_load(monkeypatch):
    """ONE model load per process (r17 Claude M1 / codex
    patch-model-all-reloads): the holder memoizes _load_model_ctx."""
    import issue2378_patch_run as run

    calls = []

    def fake_load(args):
        calls.append(1)
        return {"tok": "T"}

    monkeypatch.setattr(run, "_load_model_ctx", fake_load)
    monkeypatch.setattr(run, "_MODEL_HOLDER", {})
    args = run.build_argparser().parse_args(["--phase", "bank"])
    a = run._ensure_mctx(args)
    b = run._ensure_mctx(args)
    assert a is b and len(calls) == 1


def test_phase_all_refuses_tiny():
    import issue2378_patch_run as run

    args = run.build_argparser().parse_args(["--phase", "all", "--tiny"])
    with pytest.raises(SystemExit, match="refuses --tiny"):
        run.phase_all(args)


def _bank_fixture(tmp_path, ctx_ids, vc_keys=None):
    import json

    import numpy as np

    bank = tmp_path / "bank"
    bank.mkdir(parents=True, exist_ok=True)
    with (bank / "bank_rows.jsonl").open("w", encoding="utf-8") as fh:
        for cid in ctx_ids:
            fh.write(json.dumps({"ctx_id": cid}) + "\n")
    np.savez(
        bank / "vc_bank.npz",
        **{k: np.zeros((2, 4), dtype=np.uint16) for k in (vc_keys or ctx_ids)},
    )


def test_load_bank_key_coverage(tmp_path):
    """vc_bank.npz keys must EXACTLY cover the bank ctx-id set, checked before
    any model load (r17 codex patch-cache-key-coverage)."""
    import issue2378_patch_run as run

    args = run.build_argparser().parse_args(["--phase", "grid", "--out-root", str(tmp_path)])
    _bank_fixture(tmp_path, ["chat:q0", "story:q0"], vc_keys=["chat:q0"])
    with pytest.raises(RuntimeError, match="key-coverage mismatch"):
        run._load_bank(args)
    _bank_fixture(tmp_path, ["chat:q0", "story:q0"])
    rows, vc = run._load_bank(args)
    assert {r["ctx_id"] for r in rows} == set(vc)


def test_openers_key_coverage(tmp_path):
    import json

    import issue2378_patch_run as run

    args = run.build_argparser().parse_args(["--phase", "grid", "--out-root", str(tmp_path)])
    anchors = tmp_path / "anchors"
    anchors.mkdir(parents=True)
    with (anchors / "openers.jsonl").open("w", encoding="utf-8") as fh:
        fh.write(json.dumps({"ctx_id": "chat:q0", "opener_ids": [1, 2]}) + "\n")
    with pytest.raises(RuntimeError, match="key-coverage mismatch"):
        run._openers(args, expected_ctx_ids=["chat:q0", "story:q0"])
    assert run._openers(args, expected_ctx_ids=["chat:q0"]) == {"chat:q0": [1, 2]}


def test_confirm_empty_record_lands_in_rollouts(tmp_path):
    """The valid no-families-selected terminal record persists INSIDE the
    uploaded confirm subtree (r17 codex NIT patch-confirm-empty-not-uploaded)."""
    import json

    import issue2378_patch_run as run

    (tmp_path / "screen").mkdir(parents=True)
    (tmp_path / "screen" / "screen_report.json").write_text(
        json.dumps({"confirm_families": []}), encoding="utf-8"
    )
    args = run.build_argparser().parse_args(["--phase", "confirm", "--out-root", str(tmp_path)])
    assert run.phase_confirm(args) == 0
    assert (tmp_path / "confirm" / "rollouts" / "confirm_empty.json").exists()


def _cls(cls, stop="end_turn", score=None):
    return {"class": cls, "score": score, "reasoning": None, "stop_reason": stop}


def test_pilot_stratification_spans_classes():
    from types import SimpleNamespace

    import issue2378_patch_judge as pj

    items = [
        SimpleNamespace(item_id=f"{kind}|x{i}|d0|{rubric}")
        for kind in ("anchors", "grid")
        for rubric in ("persona", "assistant", "coherence")
        for i in range(5)
    ]
    sel = pj.stratified_pilot_items(items, 6)
    assert len(sel) == 6
    assert {pj._item_class(it.item_id) for it in sel} == {
        (k, r) for k in ("anchors", "grid") for r in ("persona", "assistant", "coherence")
    }
    assert len(pj.stratified_pilot_items(items, 999)) == len(items)


def test_pilot_vacuous_class_fails():
    """A pilot whose transport losses empty ANY (kind x rubric) class is a
    FAIL, never a PASS (r17 Claude M2 / codex patch-judge-pilot-vacuous)."""
    import issue2378_patch_judge as pj

    healthy = {
        "anchors|chat:q0|d0|persona": _cls("valid", score=80),
        "anchors|chat:q0|d0|assistant": _cls("valid", score=10),
        "grid|cell0|d0|coherence": _cls("valid", score=90),
    }
    rep = pj.pilot_report_from_classified(healthy, "batch")
    assert rep["ok"] is True and rep["vacuous_classes"] == []
    all_transport = {k: _cls("transport_loss", stop=None) for k in healthy}
    rep = pj.pilot_report_from_classified(all_transport, "batch")
    assert rep["ok"] is False and rep["n_answered"] == 0
    one_vacuous = {**healthy, "grid|cell0|d0|coherence": _cls("transport_loss", stop=None)}
    rep = pj.pilot_report_from_classified(one_vacuous, "batch")
    assert rep["ok"] is False and rep["vacuous_classes"] == ["grid|coherence"]
    capped = {**healthy, "anchors|chat:q0|d0|persona": _cls("valid", stop="max_tokens", score=80)}
    assert pj.pilot_report_from_classified(capped, "batch")["ok"] is False


def test_wave_fold_resumable(tmp_path):
    """A rerun REPLACES the prior fold instead of truncate-then-refusing on
    the existing raw shard (r17 codex patch-judge-fold-not-resumable), and the
    publish is an os.replace'd MANIFEST pointer (r18
    patch-judge-fold-publish-window): each rebuild lands in a fresh fold dir,
    the manifest flips after the fold is complete, and superseded folds plus
    legacy top-level outputs are reaped post-publish."""
    import json

    import issue2378_patch_judge as pj

    classified = {
        "grid|cell0|d0|persona": _cls("valid", score=70),
        "grid|cell0|d0|assistant": _cls("parse_fail"),
    }
    # Preseed BOTH stale shapes: the legacy pre-manifest top-level fold and a
    # stale published fold dir.
    (tmp_path / "raw").mkdir(parents=True)
    (tmp_path / "raw" / "judge_rows.shard00.jsonl").write_text("stale\n", encoding="utf-8")
    (tmp_path / "scores.jsonl").write_text("stale\n", encoding="utf-8")
    for _ in range(2):  # idempotent rebuild, twice
        tally, shard_info, fold_dir = pj._write_fold(tmp_path, classified)
        assert tally == {
            "kept": 1,
            "dropped": 1,
            "transport_loss": 0,
            "by_class": {"valid": 1, "parse_fail": 1},
        }
        man = pj.read_fold_manifest(tmp_path)
        assert man["fold_dir"] == fold_dir.name and man["tally"] == tally
        rows = [json.loads(line) for line in man["scores_path"].read_text().splitlines()]
        assert [r["kept"] for r in rows] == [False, True]
        assert all(Path(p).exists() for p in shard_info["shards"])
        assert all(Path(p).is_relative_to(fold_dir) for p in shard_info["shards"])
        # Exactly ONE fold dir survives (superseded + legacy reaped).
        fold_dirs = [p for p in tmp_path.glob("fold_*") if p.is_dir()]
        assert fold_dirs == [fold_dir]
        assert not (tmp_path / "raw").exists() and not (tmp_path / "scores.jsonl").exists()


def test_read_fold_manifest_refuses_unpublished(tmp_path):
    """No manifest = no published fold; a manifest naming a missing fold dir
    is a half-published fold — both refuse loud (r18)."""
    import json

    import issue2378_patch_judge as pj

    with pytest.raises(RuntimeError, match="missing manifest"):
        pj.read_fold_manifest(tmp_path)
    (tmp_path / "fold_manifest.json").write_text(json.dumps({"fold_dir": "fold_gone"}))
    with pytest.raises(RuntimeError, match="half-published"):
        pj.read_fold_manifest(tmp_path)


def test_pilot_stratification_interleaves_arms():
    """Within a (kind x rubric) class the pilot queue interleaves ARMS — a
    plain item_id sort drew only the lexicographically-first arm (null), so
    steered/prefill/within replies never reached the pilot (r18
    patch-judge-pilot-arm-and-binding-residual (a))."""
    from types import SimpleNamespace

    import issue2378_patch_judge as pj

    items = [
        SimpleNamespace(item_id=f"grid|{arm}|lstar|story->chat|q{i:02d}|d0|persona")
        for arm in ("null", "prefill", "steered", "within")
        for i in range(6)
    ]
    sel = pj.stratified_pilot_items(items, 4)
    assert {pj._item_arm(it.item_id) for it in sel} == {"null", "prefill", "steered", "within"}
    # Determinism + still class-spanning with a second rubric in the mix.
    items2 = items + [
        SimpleNamespace(item_id=f"grid|{arm}|lstar|story->chat|q{i:02d}|d0|coherence")
        for arm in ("null", "steered")
        for i in range(6)
    ]
    sel2a = [it.item_id for it in pj.stratified_pilot_items(items2, 8)]
    sel2b = [it.item_id for it in pj.stratified_pilot_items(items2, 8)]
    assert sel2a == sel2b
    per_class: dict[tuple[str, str], set[str]] = {}
    for iid in sel2a:
        per_class.setdefault(pj._item_class(iid), set()).add(pj._item_arm(iid))
    assert per_class[("grid", "persona")] == {"null", "prefill", "steered", "within"}
    assert per_class[("grid", "coherence")] == {"null", "steered"}


def test_instrument_sha_pins_temperature(monkeypatch):
    """The judge temperature is part of the instrument — changing it must
    invalidate a prior pilot PASS (r18 (b))."""
    import issue2378_patch_judge as pj

    sha_live = pj._instrument_sha()
    monkeypatch.setattr(pj, "JUDGE_TEMPERATURE", 0.0)
    assert pj._instrument_sha() != sha_live


def _judge_patch_root(tmp_path, n_rows=2):
    import json

    import issue2378_common as cm

    d = tmp_path / "patch_root" / "grid" / "rollouts"
    d.mkdir(parents=True)
    char = "Vex"
    assert char in cm.PERSONAS
    rows = [
        {
            "pair_type": "chat~story",
            "char": char,
            "cell_id": f"steered|lstar|story->chat|q{i}",
            "draw": 0,
            "answer": "a reply",
            "drop_reason": None,
        }
        for i in range(n_rows)
    ]
    (d / "grid.jsonl").write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return tmp_path / "patch_root"


def test_wave_binds_pilot_to_item_set(tmp_path, monkeypatch):
    """run_wave refuses when the wave's item-id set differs from the piloted
    set, and proceeds (through the manifest publish) when it matches (r18
    patch-judge-pilot-arm-and-binding-residual (c))."""
    import json
    from types import SimpleNamespace

    import issue2378_patch_judge as pj

    root = _judge_patch_root(tmp_path)
    out = tmp_path / "out"
    out.mkdir()
    items = pj.build_items(root)
    args = SimpleNamespace(
        patch_root=str(root),
        transport="sync",
        skip_pilot_gate=False,
        skip_upload=True,
        hf_suffix="",
        cache_dir=str(tmp_path / "cache"),
        checkpoint_dir=str(tmp_path / "ckpt"),
    )
    rep = {
        "ok": True,
        "transport": "sync",
        "instrument_sha": pj._instrument_sha(),
        "item_set_sha": "0000000000000000",  # a DIFFERENT harvest's pilot
    }
    (out / "pilot_report.json").write_text(json.dumps(rep), encoding="utf-8")
    monkeypatch.setattr(
        pj, "_dispatch", lambda *a, **k: pytest.fail("dispatched past the item-set gate")
    )
    with pytest.raises(SystemExit, match="item-id set"):
        pj.run_wave(args, out)
    # Matching set → the wave runs and publishes a manifest-pointed fold.
    rep["item_set_sha"] = pj._item_set_sha(items)
    (out / "pilot_report.json").write_text(json.dumps(rep), encoding="utf-8")
    monkeypatch.setattr(
        pj,
        "_dispatch",
        lambda its, *a, **k: {it.item_id: _cls("valid", score=55) for it in its},
    )
    assert pj.run_wave(args, out) == 0
    man = pj.read_fold_manifest(out)
    assert man["tally"]["kept"] == len(items)
    report = json.loads((out / "wave_report.json").read_text(encoding="utf-8"))
    assert report["fold_dir"] == man["fold_dir"]


def test_cap_hit_fractions_hit_stop():
    """A stop-string halt is an effective stop, never a cap hit (r17 codex
    patch-framing-stops-not-wired telemetry leg)."""
    import issue2378_patch_analysis as pa

    rows = [
        {"family": "f", "framing": "plain", "hit_eos": False, "hit_stop": True},
        {"family": "f", "framing": "plain", "hit_eos": False, "hit_stop": False},
        {"family": "f", "framing": "plain", "hit_eos": True, "hit_stop": False},
        {"family": "g", "framing": "story", "drop_reason": "cap_hit_no_close"},
        {"family": "g", "framing": "story", "drop_reason": None, "hit_eos": False},
    ]
    frac = pa._cap_hit_fractions(rows)
    assert frac["f"] == pytest.approx(1 / 3)
    assert frac["g"] == pytest.approx(1 / 2)


class _LenTok:
    """Length-coded decode: enough for _extract_answer to see non-stop text."""

    def decode(self, ids, skip_special_tokens=True):
        return "x" * len(ids)


def _prefill_cell(cid: str, src: str, tgt: str) -> dict:
    return {
        "cell_id": cid,
        "arm": "prefill",
        "variant": "vc",
        "src": src,
        "tgt": tgt,
        "qid": "storyq_vex_00",
        "char": "Vex",
        "pair_type": "steered",
        "direction": "fwd",
        "family": "fam|steered",
    }


def test_prefill_empty_opener_counted_drop(monkeypatch):
    """r19 crash fix (grid unit 78/94): ONE empty opener → that cell is
    SKIPPED from generation (no decoding slot), recorded as a counted
    drop_reason='opener_empty' row (bucket=1), and the block completes —
    never the pre-fix AssertionError at issue2378_patch_run.py:1120."""
    import issue2378_patch_run as run

    import explore_persona_space.experiments.issue2333.decode_hooks as dh

    dispatched = []

    def fake_generate_batch_ids(
        model,
        tokenizer,
        rows_ids,
        *,
        n=1,
        stack=None,
        donors_full=None,
        max_new_tokens=2048,
        temperature=1.0,
        seed_base=42,
        greedy=False,
        stop_strings=None,
    ):
        dispatched.append(list(rows_ids))
        return [[{"gen_ids": [9], "hit_eos": True} for _ in rows_ids]]

    monkeypatch.setattr(dh, "generate_batch_ids", fake_generate_batch_ids)
    args = run.build_argparser().parse_args(["--phase", "grid", "--tiny"])
    by_ctx = {"plain:q1": {"input_ids": [7, 7], "prompt_text": "P?"}}
    openers = {"plain:src_ok": [1, 2], "plain:storyq_astra_06": []}
    cells = [
        _prefill_cell("cell_a", "plain:src_ok", "plain:q1"),
        _prefill_cell("cell_b", "plain:storyq_astra_06", "plain:q1"),
    ]
    rows = run._run_prefill_block(
        args, {"model": object()}, _LenTok(), cells, by_ctx, openers, True, 1, "t"
    )
    # The empty-opener cell never reaches generation; the kept cell does.
    assert dispatched == [[[7, 7, 1, 2]]]
    drops = [r for r in rows if r["drop_reason"] == run.OPENER_EMPTY_DROP]
    assert len(drops) == 1 and drops[0]["cell_id"] == "cell_b"
    assert drops[0]["n_completion_tokens"] == 0 and drops[0]["donor_ctx"] == (
        "plain:storyq_astra_06"
    )
    kept = [r for r in rows if r["cell_id"] == "cell_a"]
    assert len(kept) == 1 and kept[0]["drop_reason"] != run.OPENER_EMPTY_DROP

    # All-empty block: zero-dispatch guard — generate is never called.
    rows2 = run._run_prefill_block(
        args,
        {"model": object()},
        _LenTok(),
        [_prefill_cell("cell_c", "plain:storyq_astra_06", "plain:q1")],
        by_ctx,
        openers,
        True,
        1,
        "t",
    )
    assert dispatched == [[[7, 7, 1, 2]]]  # unchanged
    assert [r["drop_reason"] for r in rows2] == [run.OPENER_EMPTY_DROP]


def test_opener_drop_floor_raise():
    """Counted drops stay <= max(1, 5% of prefill cells); above that the
    stop wiring is systematically pathological and the run must crash."""
    import issue2378_patch_run as run

    run._check_opener_drop_floor(0, 0)
    run._check_opener_drop_floor(1, 4)  # isolated drop under max(1, 0.2)
    run._check_opener_drop_floor(4, 94)  # 4 <= 4.7 (the resumed-run shape)
    with pytest.raises(AssertionError, match="systematic"):
        run._check_opener_drop_floor(2, 4)  # 2 > max(1, 0.2)
    with pytest.raises(AssertionError, match="opener_empty"):
        run._check_opener_drop_floor(6, 100)  # 6 > 5.0
