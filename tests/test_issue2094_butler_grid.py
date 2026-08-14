"""CPU-only unit tests for the issue #2094 butler round ("Option C").

No model, no GPU, no network: parent-bank ID-STABILITY golden pins (the 21,000
committed parent cells must keep resolving after the butler append), the
BUTLER_SYSTEM / holistic-descriptor byte-equality pins, butler-pair
construction, the seeded null-donor assignment + its refusal paths, grid
enumeration (10 pairs x 60 blocks x 600 cells; smoke slice covers every arm
class), judge-instrument byte-parity with the production judge driver, the
gate wave composition, floor/ceiling arithmetic on synthetic scores, and the
anchor-separation gate incl. every grid-entry refusal path.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2094_butler_grid as BG  # noqa: E402
import issue2094_judge as J  # noqa: E402
import issue2094_run as R  # noqa: E402

from explore_persona_space.experiments.issue2094 import bank as BANK  # noqa: E402
from explore_persona_space.experiments.issue2162 import bank2162, ladder_bank  # noqa: E402

N_LAYERS = 28

# ── golden pins (computed from the PRE-butler parent bank at HEAD ae76c22335;
#    recomputed HEAD-vs-post-edit on 2026-08-14 — byte-identical) ────────────

PARENT_CONTEXT_IDS = [
    f"{p}__{q}" for p in ("bare", "persona", "conv") for q in ("q1", "q2", "q3", "q4", "q5")
]
PARENT_CTX15_SHA = "4428e03f33a1bcf041eb6edb4282d98a79748736a8cdf1ad7df618cb580f6db6"
PARENT_PAIR_SHA = "04eeece56e9b6f6a63a2a18647d480184bbfeb9989fadf00b9b9e6ea295d4042"
PARENT_DONOR_SHA = "713af8b295299ff5ae4108923673582bd43f772c46d5f58ec41918b4a2b53b98"

BUTLER_PAIR_IDS = [
    f"mqb--{p}__{q}--butler__{q}"
    for p in ("bare", "persona")
    for q in ("q1", "q2", "q3", "q4", "q5")
]


def _sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


# ── ID STABILITY (the load-bearing contract) ─────────────────────────────


def test_parent_context_ids_byte_stable():
    ctx = list(BANK.build_contexts())
    assert len(ctx) == 20
    assert ctx[:15] == PARENT_CONTEXT_IDS
    assert _sha("\n".join(ctx[:15])) == PARENT_CTX15_SHA
    assert ctx[15:] == [f"butler__q{i}" for i in range(1, 6)]


def test_parent_pair_ids_and_donor_map_byte_stable():
    pairs = BANK.build_pairs()
    ids = [p.pair_id for p in pairs]
    assert len(ids) == 60
    assert _sha("\n".join(ids)) == PARENT_PAIR_SHA
    donors = BANK.donor_derangement(pairs)
    assert len(donors) == 60
    assert _sha(json.dumps(donors, sort_keys=True)) == PARENT_DONOR_SHA


def test_prefix_order_frozen_and_extended_appends():
    assert BANK.PREFIX_ORDER == ("bare", "persona", "conv")
    assert BANK.EXTENDED_PREFIX_ORDER == ("bare", "persona", "conv", "butler")
    # parent ranks byte-stable; butler strictly after
    assert [BANK._prefix_rank(p) for p in BANK.PREFIX_ORDER] == [0, 1, 2]
    assert BANK._prefix_rank("butler") == 3


def test_bank_manifest_carries_both_orders_and_parent_scope():
    m = BANK.bank_manifest()
    assert m["prefix_order"] == ["bare", "persona", "conv"]
    assert m["extended_prefix_order"] == ["bare", "persona", "conv", "butler"]
    assert len(m["contexts"]) == 20
    assert len(m["pairs"]) == 60  # pairs stay parent-scoped
    assert len(m["donor_derangement"]) == 60
    assert m["prefix_descriptors"]["butler"] == BANK.PREFIX_DESCRIPTORS["butler"]


# ── butler bank strings (byte-equality pins) ─────────────────────────────


def test_butler_system_is_byte_identical_to_2162_reginald():
    assert BANK.BUTLER_SYSTEM == bank2162.REGINALD_SYSTEM


def test_butler_descriptor_is_the_validated_holistic_ladder_descriptor():
    holistic = ladder_bank.DESCRIPTORS["r2_butler"]
    assert BANK.PREFIX_DESCRIPTORS["butler"] == holistic
    # NOT the retired four-way conjunction descriptor
    assert BANK.PREFIX_DESCRIPTORS["butler"] != bank2162._PERSONA_DESCRIPTORS["butler"]


def test_butler_contexts_shape():
    contexts = BANK.build_contexts()
    for q in ("q1", "q2", "q3", "q4", "q5"):
        ctx = contexts[f"butler__{q}"]
        assert ctx["prefix"] == "butler" and ctx["system"] == BANK.BUTLER_SYSTEM
        assert ctx["history"] == [] and ctx["user"] == BANK.QUERIES[q]
        msgs = BANK.context_messages_2094(ctx)
        assert [m["role"] for m in msgs] == ["system", "user"]
        assert msgs[0]["content"] == BANK.BUTLER_SYSTEM


def test_f_prefix_rubric_resolves_for_butler():
    core = BANK.f_prefix_rubric("butler")
    assert BANK.PREFIX_DESCRIPTORS["butler"] in core


# ── butler pairs ─────────────────────────────────────────────────────────


def test_butler_pairs_enumeration_and_direction():
    pairs = BG.build_butler_pairs()
    assert [p.pair_id for p in pairs] == BUTLER_PAIR_IDS
    for p in pairs:
        assert p.setting == "matched_query"
        assert p.prefix_b == "butler" and p.prefix_a in ("bare", "persona")
        assert p.query_a == p.query_b
        assert p.prefix_pair() in (("bare", "butler"), ("persona", "butler"))
    # disjoint from parent + rev ids
    other = {p.pair_id for p in BANK.build_pairs()} | {p.pair_id for p in BANK.build_rev_pairs()}
    assert not other & set(BUTLER_PAIR_IDS)


def test_judge_rubric_id_resolves_fp_butler_mechanically():
    pair = BG.build_butler_pairs()[0]
    assert J.rubric_id_for(pair, "prefix", "b") == "fp-butler"
    assert J.rubric_id_for(pair, "prefix", "a") == "fp-bare"


# ── donor assignment + refusal paths ─────────────────────────────────────


@pytest.fixture(scope="module")
def parent_pairs():
    return BANK.build_pairs()


@pytest.fixture(scope="module")
def butler_pairs():
    return BG.build_butler_pairs()


def test_donor_assignment_valid_distinct_and_seeded(parent_pairs, butler_pairs):
    donors = BG.butler_donor_assignment(butler_pairs, parent_pairs)
    assert list(donors) == BUTLER_PAIR_IDS
    assert len(set(donors.values())) == 10  # distinct, no replacement
    pool_ids = {p.pair_id for p in BG.butler_donor_pool(parent_pairs)}
    assert set(donors.values()) <= pool_ids
    # deterministic across calls (seeded)
    assert donors == BG.butler_donor_assignment(butler_pairs, parent_pairs)


def test_donor_refusals(parent_pairs, butler_pairs):
    recipient = butler_pairs[0]  # mqb--bare__q1--butler__q1
    by_id = {p.pair_id: p for p in parent_pairs}
    # (a) same-prefix-pair donor refused (another bare->butler pair)
    with pytest.raises(AssertionError, match="same-prefix-pair"):
        BG.assert_butler_donor(recipient, butler_pairs[1])
    # (b) butler-bearing donor refused (a persona->butler pair for a persona recipient)
    with pytest.raises(AssertionError):
        BG.assert_butler_donor(butler_pairs[5], butler_pairs[6])
    # (c) self-donation refused
    with pytest.raises(AssertionError, match="self-donation"):
        BG.assert_butler_donor(recipient, recipient)
    # (d) non-matched-query donor refused
    cross = next(p for p in parent_pairs if p.setting == "cross")
    with pytest.raises(AssertionError):
        BG.assert_butler_donor(recipient, cross)
    # (e) a parent mq donor with a different prefix pair PASSES
    ok = by_id["mq--bare__q1--persona__q1"]
    BG.assert_butler_donor(recipient, ok)


# ── grid enumeration ─────────────────────────────────────────────────────


def test_block_enumeration_60_blocks_600_cells(butler_pairs):
    fams = BG.enumerate_butler_blocks(butler_pairs, N_LAYERS)
    assert len(fams) == 30  # 28 single layers + joint_mid + joint_all
    blocks = [b for fam in fams for b in fam]
    assert len(blocks) == 60
    totals = R.grid_totals(fams)
    assert totals["n_blocks"] == 60 and totals["cells_total"] == 600
    assert len({b.key for b in blocks}) == 60
    for b in blocks:
        assert (b.slot, b.dose, b.vec_type) == ("ce", "replace", "A")
        assert b.arm in ("steered", "null")
        assert list(b.pair_ids) == BUTLER_PAIR_IDS


def test_smoke_slice_covers_every_arm_class(butler_pairs):
    fams = BG.smoke_butler_blocks(butler_pairs, N_LAYERS)
    blocks = [b for fam in fams for b in fam]
    assert {b.arm for b in blocks} == {"steered", "null"}
    variants = {b.layer_variant for b in blocks}
    assert "joint_all" in variants and "joint_mid" in variants
    assert any(v.startswith("L") for v in variants)  # one single-layer variant
    # every block carries all 10 pairs => both prefix-pair families per class
    for b in blocks:
        assert list(b.pair_ids) == BUTLER_PAIR_IDS


def test_regime_fingerprint_is_donor_and_pair_sensitive(parent_pairs, butler_pairs):
    cfg = SimpleNamespace(
        model_id="m",
        tiny=True,
        n_layers=4,
        hidden=64,
        max_new_tokens=8,
        seed_base=42,
        smoke=True,
    )
    donors = BG.butler_donor_assignment(butler_pairs, parent_pairs)
    base = R.regime_fingerprint(cfg, "banksha")
    fp1 = BG.butler_regime_fingerprint(cfg, "banksha", donors)
    assert fp1 == BG.butler_regime_fingerprint(cfg, "banksha", dict(donors))
    other = dict(donors)
    k0 = BUTLER_PAIR_IDS[0]
    other[k0], other[BUTLER_PAIR_IDS[1]] = other[BUTLER_PAIR_IDS[1]], other[k0]
    assert fp1 != BG.butler_regime_fingerprint(cfg, "banksha", other)
    assert fp1 != base  # never satisfiable by a parent/rev done-file


# ── judge instrument parity (byte-copies of the production judge) ────────


def test_judge_templates_byte_parity_with_judge_driver():
    assert BG.REASON_THEN_SCORE == J.REASON_THEN_SCORE
    assert BG.coherence_eval_prompt() == J.coherence_eval_prompt()
    core = BANK.f_prefix_rubric("butler")
    assert BG.behavior_eval_prompt(core) == J.behavior_eval_prompt(core)


def test_butler_rubric_registry_covers_gate_rubrics():
    reg = BG.butler_rubric_registry()
    assert set(reg) == {"coherence", "fp-butler", "fp-bare", "fp-persona"}
    assert BANK.PREFIX_DESCRIPTORS["butler"] in reg["fp-butler"]


# ── gate wave composition ────────────────────────────────────────────────


def _synth_butler_rows(n_draws: int = 2) -> list[dict]:
    return [
        {"context_id": f"butler__q{i}", "draw": d, "text": f"resp {i} {d}", "cap_hit": False}
        for i in range(1, 6)
        for d in range(n_draws)
    ]


def _synth_banked_texts(n_draws: int = 2) -> dict[tuple[str, int], str]:
    return {
        (f"{p}__q{i}", d): f"banked {p} {i} {d}"
        for p in ("bare", "persona")
        for i in range(1, 6)
        for d in range(n_draws)
    }


def test_compose_judge_waves_shape_and_ids():
    waves = BG.compose_judge_waves(_synth_butler_rows(), _synth_banked_texts())
    assert set(waves) == {
        "coherence.butler-anchors",
        "fp-butler.butler-anchors",
        "fp-bare.butler-anchors",
        "fp-persona.butler-anchors",
        "fp-butler.bare-anchors",
        "fp-butler.persona-anchors",
    }
    for _wave, spec in waves.items():
        assert len(spec["items"]) == 10  # 5 contexts x 2 draws
        for iid, question, answer in spec["items"]:
            assert len(iid) <= 53 and iid.replace("-", "").isalnum()
            cid, _draw = spec["keys"][iid]
            assert question == BANK.QUERIES[cid.split("__")[1]]
            assert isinstance(answer, str) and answer
    # butler-side waves key on butler contexts; banked-side on bare/persona
    assert all(
        k[0].startswith("butler") for k in waves["fp-butler.butler-anchors"]["keys"].values()
    )
    assert all(k[0].startswith("bare") for k in waves["fp-butler.bare-anchors"]["keys"].values())


# ── floor / ceiling + gate arithmetic ────────────────────────────────────


def _synth_scores(bare_sep: float, persona_sep: float):
    """Score tables giving denominator == bare_sep on bare pairs, persona_sep
    on persona pairs: ceiling = +sep/2, floor = -sep/2 (deltas on [-1,1])."""
    scores = {"fp-butler": {}, "fp-bare": {}, "fp-persona": {}}
    for i in range(1, 6):
        for d in range(2):
            for prefix, sep in (("bare", bare_sep), ("persona", persona_sep)):
                # context A draws: fp-butler - fp-<a> = -sep/2 * 100
                scores["fp-butler"][(f"{prefix}__q{i}", d)] = 10.0
                scores[f"fp-{prefix}"][(f"{prefix}__q{i}", d)] = 10.0 + sep * 50
            # butler draws: fp-butler - fp-<a> = +sep/2 * 100 (per rubric_a)
            scores["fp-butler"][(f"butler__q{i}", d)] = 60.0
            scores["fp-bare"][(f"butler__q{i}", d)] = 60.0 - bare_sep * 50
            scores["fp-persona"][(f"butler__q{i}", d)] = 60.0 - persona_sep * 50
    banked_coherent = {
        f"{p}__q{i}": {"coherent": [0, 1], "n_total": 2}
        for p in ("bare", "persona")
        for i in range(1, 6)
    }
    butler_coherent = {f"butler__q{i}": [0, 1] for i in range(1, 6)}
    return scores, banked_coherent, butler_coherent


def test_floor_ceiling_arithmetic_and_gate_pass(butler_pairs):
    scores, banked, butler = _synth_scores(bare_sep=1.2, persona_sep=0.4)
    fc = BG.compute_butler_floor_ceiling(scores, banked, butler, butler_pairs)
    assert len(fc["pairs"]) == 10
    for rec in fc["pairs"]:
        want = 1.2 if rec["prefix_pair"] == ["bare", "butler"] else 0.4
        assert rec["denominator"] == pytest.approx(want)
        assert rec["rubric_b"] == "fp-butler"
    gate = BG.evaluate_gate(fc, butler_pairs, anchor_draws=2)
    assert gate["passed"] is True
    assert gate["mean_denominator_bare_butler"] == pytest.approx(1.2)
    assert gate["mean_denominator_persona_butler_informational"] == pytest.approx(0.4)


def test_gate_fails_below_bar(butler_pairs):
    scores, banked, butler = _synth_scores(bare_sep=0.3, persona_sep=0.3)
    fc = BG.compute_butler_floor_ceiling(scores, banked, butler, butler_pairs)
    gate = BG.evaluate_gate(fc, butler_pairs, anchor_draws=2)
    assert gate["passed"] is False


def test_floor_ceiling_drops_missing_scores_and_fails_loud_on_empty(butler_pairs):
    scores, banked, butler = _synth_scores(bare_sep=1.2, persona_sep=0.4)
    # one missing score on one butler draw -> dropped + counted, mean unchanged
    scores["fp-butler"][("butler__q1", 0)] = None
    fc = BG.compute_butler_floor_ceiling(scores, banked, butler, butler_pairs)
    rec = next(r for r in fc["pairs"] if r["context_b"] == "butler__q1")
    assert rec["ceiling"]["draws_dropped_missing_score"] == [0]
    assert rec["ceiling"]["n_draws_kept"] == 1
    # every draw of one side missing -> fail loud, never a defaulted score
    scores["fp-butler"][("butler__q1", 1)] = None
    with pytest.raises(AssertionError, match="missing a judge score"):
        BG.compute_butler_floor_ceiling(scores, banked, butler, butler_pairs)


def test_coherent_draw_filter():
    coh = {("butler__q1", 0): 100.0, ("butler__q1", 1): 30.0, ("butler__q2", 0): None}
    out = BG._coherent_draws_from_scores(coh)
    assert out == {"butler__q1": [0]}  # >60 only; None excluded


# ── grid-entry gate refusal paths (the HARD gate) ────────────────────────


def _gate_cfg(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(manifest_dir=tmp_path)


def test_grid_refuses_without_gate_file(tmp_path, butler_pairs):
    with pytest.raises(RuntimeError, match="gate report missing"):
        BG._require_butler_gate(_gate_cfg(tmp_path), butler_pairs)


def test_grid_refuses_failed_gate(tmp_path, butler_pairs):
    rec = {"passed": False, "pair_ids": BUTLER_PAIR_IDS, "mean_denominator_bare_butler": 0.1}
    (tmp_path / "butler_gate.json").write_text(json.dumps(rec))
    with pytest.raises(RuntimeError, match="gate FAILED"):
        BG._require_butler_gate(_gate_cfg(tmp_path), butler_pairs)


def test_grid_refuses_pair_set_mismatch(tmp_path, butler_pairs):
    rec = {"passed": True, "pair_ids": BUTLER_PAIR_IDS[:5]}
    (tmp_path / "butler_gate.json").write_text(json.dumps(rec))
    with pytest.raises(RuntimeError, match="DIFFERENT pair set"):
        BG._require_butler_gate(_gate_cfg(tmp_path), butler_pairs)


def test_grid_accepts_passed_gate(tmp_path, butler_pairs):
    rec = {"passed": True, "pair_ids": BUTLER_PAIR_IDS, "mean_denominator_bare_butler": 1.1}
    (tmp_path / "butler_gate.json").write_text(json.dumps(rec))
    BG._require_butler_gate(_gate_cfg(tmp_path), butler_pairs)  # no raise


# ── git-persist payload (#2300: HF data repo at the 1M-file Hub cap) ─────


def test_write_jsonl_sharded_splits_round_trips_and_refuses_empty(tmp_path):
    rows = [{"i": i, "text": "x" * 100} for i in range(50)]
    paths = BG._write_jsonl_sharded(tmp_path, "t", rows, max_bytes=1000)
    assert len(paths) > 1  # split fired at the tiny cap
    assert [p.name for p in paths] == [f"t.shard{i:02d}.jsonl" for i in range(len(paths))]
    assert all(p.stat().st_size <= 1000 for p in paths)
    assert not list(tmp_path.glob("*.gz"))  # never gzip (LFS-matched)
    back = [row for p in paths for row in BG._read_jsonl(p)]
    assert back == rows  # order-preserving round trip
    # one shard when under the cap
    assert len(BG._write_jsonl_sharded(tmp_path, "small", rows[:2])) == 1
    with pytest.raises(AssertionError, match="EMPTY payload"):
        BG._write_jsonl_sharded(tmp_path, "empty", [])


def _payload_cfg(tmp_path: Path) -> SimpleNamespace:
    out = tmp_path / "out"
    cfg = SimpleNamespace(
        out_root=out,
        rollouts_dir=out / "rollouts",
        manifest_dir=out / "manifests",
        model_id="m",
        tiny=True,
        smoke=False,
        n_layers=4,
        hidden=64,
        max_new_tokens=8,
        seed_base=42,
        anchor_draws=2,
        gen_batch=16,
        num_workers=1,
        gpu_hours_budgeted=1.0,
    )
    cfg.rollouts_dir.mkdir(parents=True)
    cfg.manifest_dir.mkdir(parents=True)
    return cfg


def _stage_round_artifacts(cfg: SimpleNamespace) -> None:
    for slug in ("ce__L2__replace__A__steered", "ce__L2__replace__A__null"):
        rows = [
            {"block_key": slug, "pair_id": BUTLER_PAIR_IDS[i], "text": f"r{i}"} for i in range(10)
        ]
        (cfg.rollouts_dir / f"shard_{slug}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows)
        )
    adir = BG.butler_anchors_dir(cfg)
    adir.mkdir(parents=True)
    (adir / "butler_anchors.jsonl").write_text(
        "".join(
            json.dumps({"context_id": f"butler__q{i}", "draw": 0, "text": f"a{i}"}) + "\n"
            for i in range(1, 4)
        )
    )
    (adir / "butler_anchor_draws.jsonl").write_text(
        json.dumps({"context_id": "butler__q1", "draw": 0, "coherent": True}) + "\n"
    )
    (adir / "anchors_done.json").write_text(json.dumps({"n_rows": 3}))
    (adir / "va_butler_anchors.pt").write_bytes(b"\x00tensor")  # must NOT enter git
    jdir = BG.judge_dir(cfg)
    (jdir / "raw").mkdir(parents=True)
    for wave, rid in (
        ("fp-butler.butler-anchors", "fp-butler"),
        ("coherence.butler-anchors", "coherence"),
    ):
        (jdir / f"{wave}.scores.jsonl").write_text(
            json.dumps({"rubric_id": rid, "context_id": "butler__q1", "draw": 0, "score": 80.0})
            + "\n"
        )
        (jdir / f"{wave}.meta.json").write_text(json.dumps({"complete": True}))
        (jdir / "raw" / f"{wave}.json").write_text(json.dumps({"n": 1}))
    (cfg.manifest_dir / "butler_gate.json").write_text(
        json.dumps({"passed": True, "pair_ids": BUTLER_PAIR_IDS})
    )
    (cfg.manifest_dir / "butler_floor_ceiling.json").write_text(json.dumps({"pairs": []}))
    (cfg.manifest_dir / "butler_grid_plan_w0.json").write_text(json.dumps({"worker_index": 0}))


def test_build_git_payload_consolidates_low_file_count(tmp_path):
    cfg = _payload_cfg(tmp_path)
    _stage_round_artifacts(cfg)
    root = tmp_path / "payload"
    summary = BG.build_git_payload(cfg, root)
    # rollouts: 2 per-block shards -> ONE consolidated sharded set, 20 rows
    assert summary["rollouts"]["n_rows"] == 20 and summary["rollouts"]["n_source_blocks"] == 2
    rollout_files = [root / f for f in summary["rollouts"]["files"]]
    assert all(f.is_file() for f in rollout_files)
    assert sum(len(BG._read_jsonl(f)) for f in rollout_files) == 20
    # judge scores carry the wave field; meta + raw present
    score_rows = [r for f in summary["judge_scores"]["files"] for r in BG._read_jsonl(root / f)]
    assert {r["wave"] for r in score_rows} == {
        "fp-butler.butler-anchors",
        "coherence.butler-anchors",
    }
    assert set(json.loads((root / "judge_meta.json").read_text())) == {
        "fp-butler.butler-anchors",
        "coherence.butler-anchors",
    }
    assert summary["judge_raw"]["n_waves"] == 2
    # gate + floor/ceiling verbatim copies
    assert json.loads((root / "butler_gate.json").read_text())["passed"] is True
    assert (root / "butler_floor_ceiling.json").is_file()
    # run manifest: discard record + grid plan folded in
    man = json.loads((root / "run_manifest.json").read_text())
    assert set(man["discarded_tensors"]) == set(BG.DISCARDED_TENSOR_RECIPES)
    assert "butler_grid_plan_w0.json" in man["manifests"]
    assert "anchors_done.json" in man["manifests"]
    # NO tensors and NO gzip in the payload; file count stays low
    assert not list(root.glob("*.pt")) and not list(root.glob("*.gz"))
    assert len(list(root.iterdir())) <= 12


def test_build_git_payload_refuses_empty_rollouts(tmp_path):
    cfg = _payload_cfg(tmp_path)  # no artifacts staged
    with pytest.raises(AssertionError, match="no grid rollout rows"):
        BG.build_git_payload(cfg, tmp_path / "payload")


def test_sentinel_payload_git_mode_routes_and_discloses(tmp_path):
    cfg = _payload_cfg(tmp_path)
    _stage_round_artifacts(cfg)
    root = tmp_path / "payload"
    note = BG._sentinel_payload(
        cfg, {"rollouts": ["rollouts.shard00.jsonl"]}, persist="git", payload_root=root
    )
    required = (
        "eval_numbers",
        "eval_paths",
        "reproducibility_card",
        "wandb_url",
        "hf_hub_url",
        "worktree_path",
        "final_commit_sha",
        "gpu_hours_used",
        "gpu_hours_budgeted",
        "plan_deviations",
    )
    assert all(k in note for k in required)
    assert note["hf_hub_url"] is None  # nothing landed on HF this round
    assert str(root) in note["eval_paths"]
    assert any("#2300" in d and "GIT" in d for d in note["plan_deviations"])
    # hf mode keeps the original destination + no #2300 routing line
    note_hf = BG._sentinel_payload(cfg, {}, persist="hf", payload_root=None)
    assert note_hf["hf_hub_url"] and "butler_grid" in note_hf["hf_hub_url"]
    assert not any("#2300" in d for d in note_hf["plan_deviations"])
