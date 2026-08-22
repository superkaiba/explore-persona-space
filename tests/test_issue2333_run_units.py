"""Issue #2333 driver CPU pins — short-donor pair-drop + the cap-hit regen
branch + the prefill id-concatenation seam, via a tiny-real from-config model
through the REAL ``run_block_2333`` body (no network, no GPU).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2162_run as R  # noqa: E402
import issue2333_run as RUN  # noqa: E402

VOCAB = 128
HIDDEN = 32
N_LAYERS = 2


class FakeTok:
    pad_token_id = 0
    eos_token_id = 1

    def decode(self, ids, skip_special_tokens=True):
        return "".join(f"[{t}]" for t in ids)


class FakeSeamTok(FakeTok):
    """Seam-sensitive tokenizer: decode prefixes the CALL's token count, so a
    joint decode of donor+continuation ids is DISTINGUISHABLE from the
    concatenation of two split decodes (r1 blocker prefill-response-decode —
    with a concat-only fake, split == joint by construction and the pin was
    vacuous)."""

    def decode(self, ids, skip_special_tokens=True):
        return f"<{len(ids)}>" + "".join(f"[{t}]" for t in ids)


@pytest.fixture(scope="module")
def tiny_model():
    from transformers import Qwen2Config, Qwen2ForCausalLM

    torch.manual_seed(2333)
    cfg = Qwen2Config(
        vocab_size=VOCAB,
        hidden_size=HIDDEN,
        intermediate_size=64,
        num_hidden_layers=N_LAYERS,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=256,
        pad_token_id=0,
        eos_token_id=1,
        bos_token_id=None,
        tie_word_embeddings=False,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    return model


def _donor_rec(token_ids: list[int]) -> dict:
    return {
        "token_ids": token_ids,
        "states": torch.zeros(len(token_ids), N_LAYERS, HIDDEN),
        "donor_len": len(token_ids),
    }


def test_short_donor_pair_drop():
    """Plan §4.2 edge rule: drop the pair from the k-arm (steered AND null
    symmetric) when EITHER its own donor or its shuffled donor is < k."""
    donors = {"med": {"p1": _donor_rec([2, 3, 4]), "p2": _donor_rec([5])}}
    donor_maps = {"shuffled": {"p1": "p2", "p2": "p1"}}
    # p1 own len 3 but null donor (p2) len 1 -> dropped at k>=2, kept at k=1.
    assert not RUN.pair_dropped_for_arm(donors, donor_maps, "p1", 1, "med")
    assert RUN.pair_dropped_for_arm(donors, donor_maps, "p1", 2, "med")
    assert RUN.pair_dropped_for_arm(donors, donor_maps, "p2", 3, "med")
    # Missing donor record -> dropped.
    donors_missing = {"med": {"p1": _donor_rec([2, 3, 4])}}
    assert RUN.pair_dropped_for_arm(donors_missing, donor_maps, "p1", 1, "med")


def _mk_cfg(tmp_path: Path) -> RUN.RunConfig:
    return RUN.RunConfig(
        phase="grid",
        model_tag="q25",
        out_root=tmp_path / "out",
        log_dir=tmp_path / "logs",
        model_id="tiny-test",
        tiny=True,
        n_layers=N_LAYERS,
        hidden=HIDDEN,
        device="cpu",
        gen_batch=4,
        capture_batch=4,
        max_new_tokens=1,  # every non-EOS row cap-hits -> regen branch fires
        anchor_draws=1,
        grid_draws=2,
        seed_base=11,
        smoke=True,
        pilot=False,
        force=False,
        worker_index=0,
        num_workers=1,
        upload_mode="skip",
        upload_every=0,
        planned_wall_h=0.1,
        only_blocks=(),
    )


def test_run_block_prefill_cap_hit_regen_and_id_seam(tiny_model, tmp_path):
    """The REAL run_block_2333 body + the REGISTERED post-queue cap-regen pass
    on a prefill (steered AND null) pair of blocks: (a) the trigger pools BOTH
    variants per (cell, arm) and regenerates BOTH (r1 blocker
    cap-regen-wrong-grain — the r1 inline per-chunk/one-variant regen is
    gone); (b) the judged whole response is ONE joint decode of
    donor_ids + gen_ids (seam-sensitive FakeSeamTok: a split decode would
    carry TWO `<len>` prefixes — r1 blocker prefill-response-decode);
    (c) V_a spans cover donor+continuation and survive the regen rewrite."""
    cfg = _mk_cfg(tmp_path)
    tok = FakeSeamTok()
    pairs = [
        SimpleNamespace(pair_id="p1", a="ctxA1", b="ctxB1", cell="cellX"),
        SimpleNamespace(pair_id="p2", a="ctxA2", b="ctxB2", cell="cellX"),
    ]
    pairs_by_id = {p.pair_id: p for p in pairs}
    ctx_ids = {"ctxA1": [3, 4, 5, 6], "ctxA2": [7, 8, 9]}
    donors = {"med": {"p1": _donor_rec_randn([21, 22], 1), "p2": _donor_rec_randn([23, 24, 25], 2)}}
    donor_maps = {"shuffled": {"p1": "p2", "p2": "p1"}}
    bank = {"per_context": {}}
    blocks = [
        R.Block("cellX", "prefill2_med", "steered", ("p1", "p2")),
        R.Block("cellX", "prefill2_med", "null", ("p1", "p2")),
    ]
    for block in blocks:
        RUN.run_block_2333(
            cfg, tiny_model, tok, bank, donors, donor_maps, pairs_by_id, ctx_ids, block, "test-fp"
        )
    # Registered regen grain: post-queue, per (cell, arm), BOTH variants pooled.
    RUN._cap_regen_pass(
        cfg, tiny_model, tok, donors, donor_maps, pairs_by_id, ctx_ids, blocks, "test-fp"
    )

    n_regen_total = 0
    for block in blocks:
        # Shard lands in the SMOKE-namespaced dir (r1 Major 2) — never blocks/.
        shard = cfg.grid_blocks_dir / f"{block.slug}.jsonl"
        assert cfg.grid_blocks_dir.name == "smoke_blocks"
        assert not (cfg.rollouts_dir / "blocks").exists()
        rows = [json.loads(line) for line in shard.read_text().split("\n") if line.strip()]
        assert len(rows) == 2 * cfg.grid_draws  # 2 pairs x 2 draws
        n_regen = sum(1 for r in rows if r.get("regenerated_at") == 2 * cfg.max_new_tokens)
        assert n_regen >= 1, f"cap regen never fired on {block.key}"
        n_regen_total += n_regen
        for r in rows:
            assert r["donor_len"] == 2  # k=2 donor ids verbatim
            # Joint-decode pin: EXACTLY ONE decode call over donor+continuation
            # (a split decode would read "<2>[..][..]<n>..." instead).
            assert r["response_text"].startswith(f"<{r['donor_len'] + r['n_completion_tokens']}>")
            assert r["donor_text"].startswith("<2>")
            assert r["kind"] == "prefill" and r["variant"] == block.arm
            assert r["cap_hit_basis"] == "gen_token_count"
        va = torch.load(cfg.va_dir / f"{block.slug}.pt", map_location="cpu", weights_only=False)
        assert len(va) == 4  # prefill spans include the donor ids -> never dropped
        for key, t in va.items():
            assert t.shape == (N_LAYERS, HIDDEN), key
            assert t.dtype == torch.float16
        done = json.loads(R.block_done_path(cfg.out_root, block, "smoke_blocks").read_text())
        assert done["n_rows"] == 4 and done["n_pairs_kept"] == 2
        assert done["cap_regen_applied"] == n_regen
        assert done["n_cap_hit"] == sum(1 for r in rows if r["cap_hit"])
        assert R.block_is_done(cfg.out_root, block, "test-fp", "smoke_blocks")
        assert not R.block_done_path(cfg.out_root, block, "blocks").exists()
    # The capregen pseudo-block done record: fired, both variants pooled.
    pb = R.Block("cellX", "prefill2_med", "capregen", ())
    rec = json.loads(R.block_done_path(cfg.out_root, pb, "smoke_blocks_capregen").read_text())
    assert rec["fired"] is True and rec["variants_pooled"] == ["null", "steered"]
    assert rec["n_regenerated"] == n_regen_total
    assert rec["threshold"] == RUN.CAP_HIT_REGEN_FRAC
    assert R.block_is_done(cfg.out_root, pb, "test-fp", "smoke_blocks_capregen")


def test_cap_regen_pass_does_not_fire_below_threshold(tmp_path):
    """Zero cap hits => the capregen pass records fired=False and rewrites
    NOTHING (idempotent no-op with a durable decision record). Synthetic
    shards; the model/tok are never touched on the no-fire path."""
    cfg = _mk_cfg(tmp_path)
    blocks = [R.Block("cellX", "prefill2_med", v, ("p1",)) for v in ("steered", "null")]
    cfg.grid_blocks_dir.mkdir(parents=True, exist_ok=True)
    for b in blocks:
        rows = [
            {
                "block_key": b.key,
                "pair_id": "p1",
                "draw": 0,
                "n_completion_tokens": 0,
                "cap_hit": False,
                "regenerated_at": None,
                "va_key": f"p1|prefill2_med|{b.arm}|d0",
            }
        ]
        (cfg.grid_blocks_dir / f"{b.slug}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
        )
    before = {b.slug: (cfg.grid_blocks_dir / f"{b.slug}.jsonl").read_text() for b in blocks}
    RUN._cap_regen_pass(cfg, None, None, None, None, None, None, blocks, "test-fp")
    pb = R.Block("cellX", "prefill2_med", "capregen", ())
    rec = json.loads(R.block_done_path(cfg.out_root, pb, "smoke_blocks_capregen").read_text())
    assert rec["fired"] is False and rec["n_regenerated"] == 0
    assert rec["cap_frac_original"] == 0.0 and rec["variants_pooled"] == ["null", "steered"]
    for b in blocks:
        assert (cfg.grid_blocks_dir / f"{b.slug}.jsonl").read_text() == before[b.slug]


def _donor_rec_randn(token_ids: list[int], seed: int) -> dict:
    g = torch.Generator().manual_seed(seed)
    return {
        "token_ids": token_ids,
        "states": torch.randn(len(token_ids), N_LAYERS, HIDDEN, generator=g),
        "donor_len": len(token_ids),
    }


def test_run_block_patch_arm_null_variant(tiny_model, tmp_path):
    """The REAL run_block_2333 body on a patch/NULL block: decode-step edits
    fire (telemetry non-empty), the null donor states are norm-matched to the
    recipient's own scheme states, and the hooked multi-position V_a capture
    (R._arm_hook_all_layers with (k_eff, L, H) payloads) runs end-to-end.

    max_new_tokens must be >= 2 here: token 1 is sampled from the PREFILL
    forward (plan §4.2 — the k-th donor state edits DECODE step k, affecting
    tokens 2..k+1), so a 1-token generation runs zero decode forwards and
    zero edits — by design, not a bug."""
    import dataclasses

    cfg = dataclasses.replace(_mk_cfg(tmp_path), max_new_tokens=4)
    pairs = [
        SimpleNamespace(pair_id="p1", a="ctxA1", b="ctxB1", cell="cellX"),
        SimpleNamespace(pair_id="p2", a="ctxA2", b="ctxB2", cell="cellX"),
    ]
    ctx_ids = {"ctxA1": [3, 4, 5, 6], "ctxA2": [7, 8, 9]}
    donors = {"med": {"p1": _donor_rec_randn([21, 22], 1), "p2": _donor_rec_randn([23, 24], 2)}}
    donor_maps = {"shuffled": {"p1": "p2", "p2": "p1"}}
    block = R.Block("cellX", "patch2_med", "null", ("p1", "p2"))
    RUN.run_block_2333(
        cfg,
        tiny_model,
        FakeTok(),
        {"per_context": {}},
        donors,
        donor_maps,
        {p.pair_id: p for p in pairs},
        ctx_ids,
        block,
        "test-fp",
    )
    done = json.loads(R.block_done_path(cfg.out_root, block, "smoke_blocks").read_text())
    assert done["n_pairs_kept"] == 2 and done["n_rows"] == 2 * cfg.grid_draws
    assert done["edit_telemetry"]["n_edits"] > 0, "decode-step edits never fired"
    # Telemetry accumulates ACROSS the K draws (decode_hooks no longer resets
    # realized_edits per arm_replace — r1 Minor last-draw-only): a single
    # draw's ceiling is k=2 positions x 2 pairs x N_LAYERS = 8 edits, so any
    # count above it proves cross-draw accumulation (exact count is
    # early-EOS-dependent; grid_draws=2 here).
    assert done["edit_telemetry"]["n_edits"] > 2 * 2 * N_LAYERS
    shard = cfg.grid_blocks_dir / f"{block.slug}.jsonl"
    rows = [json.loads(line) for line in shard.read_text().split("\n") if line.strip()]
    for r in rows:
        assert r["kind"] == "patch" and r["variant"] == "null"
        assert r["donor_pair_id"] == donor_maps["shuffled"][r["pair_id"]]
        assert r["response_text"] == r["continuation_text"]  # no donor prefix on patch arms
    va = torch.load(cfg.va_dir / f"{block.slug}.pt", map_location="cpu", weights_only=False)
    for key, t in va.items():
        assert t.shape == (N_LAYERS, HIDDEN), key


def test_phase_done_cross_regime_refusal(tmp_path):
    """Resume matrix: `_phase_done` skips on a matching regime_fp, returns
    False when absent, and HARD-refuses a cross-regime resume (#722 r3)."""
    cfg = _mk_cfg(tmp_path)
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    assert not RUN._phase_done(cfg, "bank", "fp-a")
    RUN._write_phase_done(cfg, "bank", "fp-a", {"n_contexts": 1})
    assert RUN._phase_done(cfg, "bank", "fp-a")
    with pytest.raises(RuntimeError, match="cross-regime resume"):
        RUN._phase_done(cfg, "bank", "fp-b")


def test_run_block_drops_short_donor_pairs(tiny_model, tmp_path):
    """k=3 arm with a 2-token donor: the pair is dropped and recorded."""
    cfg = _mk_cfg(tmp_path)
    pairs = [SimpleNamespace(pair_id="p1", a="ctxA1", b="ctxB1", cell="cellX")]
    ctx_ids = {"ctxA1": [3, 4, 5, 6]}
    donors = {"med": {"p1": _donor_rec([21, 22]), "p2": _donor_rec([23, 24, 25])}}
    donor_maps = {"shuffled": {"p1": "p2", "p2": "p1"}}
    block = R.Block("cellX", "prefill3_med", "steered", ("p1",))
    RUN.run_block_2333(
        cfg,
        tiny_model,
        FakeTok(),
        {"per_context": {}},
        donors,
        donor_maps,
        {p.pair_id: p for p in pairs},
        ctx_ids,
        block,
        "test-fp",
    )
    done = json.loads(R.block_done_path(cfg.out_root, block, "smoke_blocks").read_text())
    assert done["n_pairs_kept"] == 0 and done["dropped_pair_ids"] == ["p1"]
    assert done["n_rows"] == 0


# ── r2 additions: minimal-pair / fingerprint / S2 map / chunk store / guards ──


IMS = 7  # synthetic <|im_start|> token id for the span-locus tests


def test_minimal_pair_check_span_locus():
    """r3 fix: the A16 gate is the SPAN-LOCUS verdict (vendored from #2329
    ``_pair_verdict``, prefix-side branch), NOT the r2 single-contiguous-
    diff-region predicate — conflict cells swap the instruction slot AND the
    demo-history format (~12 interleaved prefix diff regions BY DESIGN) and
    mq pairs compose different templates, so the r2 form failed 77/195 (q25)
    and 82/195 (q35) with an empty q35-only set. Intact iff the final user
    turn + generation prompt (ids from the second-to-last <|im_start|>) are
    token-IDENTICAL and the varied prefix actually differs."""
    mp = RUN.minimal_pair_check
    final = [IMS, 50, 51, IMS, 60]  # final user turn + generation header

    # single prefix substitution -> intact
    assert mp([IMS, 1, 2, *final], [IMS, 9, 2, *final], IMS) == ()
    # conflict-cell shape: TWO interleaved prefix diff regions -> intact
    # (exactly the shape the r2 predicate falsely rejected)
    assert mp([IMS, 1, 2, 3, 4, 5, *final], [IMS, 9, 2, 3, 8, 5, *final], IMS) == ()
    # template recomposition (mq persona<->conv): different turn structure,
    # different prefix length, multiple diff regions -> intact
    assert mp([IMS, 1, 2, IMS, 3, 4, 5, *final], [IMS, 6, 7, *final], IMS) == ()
    # q35 thinking-off bare render (2 occurrences, EMPTY prefix) vs persona
    # render (system turn present) -> intact
    assert mp(list(final), [IMS, 1, 2, *final], IMS) == ()

    # genuine tokenizer break: final-turn tokens differ -> violation
    assert mp([IMS, 1, 2, IMS, 50, 51, IMS, 60], [IMS, 9, 2, IMS, 50, 99, IMS, 60], IMS) == (
        "final-turn-tokens-differ",
    )
    # bank defect: varied prefix identical (identical renders) -> violation
    assert mp([IMS, 1, 2, *final], [IMS, 1, 2, *final], IMS) == ("varied-prefix-identical",)
    # both breaks at once -> both reasons
    assert mp([IMS, 1, 2, IMS, 50], [IMS, 1, 2, IMS, 99], IMS) == (
        "final-turn-tokens-differ",
        "varied-prefix-identical",
    )


def test_final_turn_boundary_occurrence_relaxation():
    """Boundary = SECOND-TO-LAST <|im_start|>; >= 2 occurrences accepted (q35
    thinking-off bare renders insert no default system turn); < 2 fails loud."""
    ftb = RUN.final_turn_boundary
    assert ftb([IMS, 10, IMS, 20], IMS) == 0  # bare render: 2 occurrences
    assert ftb([IMS, 1, IMS, 10, IMS, 20], IMS) == 2  # + system turn
    assert ftb([IMS, 1, IMS, 2, IMS, 10, IMS, 20], IMS) == 4  # + demo history
    with pytest.raises(AssertionError):
        ftb([IMS, 10, 11], IMS)  # single occurrence — not a chat render


class _ImsTok(FakeTok):
    """FakeTok + the ``convert_tokens_to_ids`` surface the prefix-end helpers
    read (both the parent ``prefix_end_index_multi`` and the r5
    ``prefix_end_index_2333`` resolve the atomic ``<|im_start|>`` id)."""

    def convert_tokens_to_ids(self, token: str) -> int:
        assert token == "<|im_start|>", token
        return IMS


def test_prefix_end_index_2333_boundary_semantics():
    """r5 crash fix: q35 thinking-off BARE renders (2 <|im_start|> occurrences
    — no default system turn) get boundary 0 (EMPTY varied prefix, the #2329
    ``prefix_end_index_2329`` shape); >= 3 occurrences delegate to the parent
    ``prefix_end_index_multi`` VERBATIM (q25 byte-equality); every valid input
    is value-coherent with ``final_turn_boundary`` (occ[-2])."""
    from explore_persona_space.experiments.issue2094 import bank as BANK94

    tok = _ImsTok()
    pe = RUN.prefix_end_index_2333

    # q35 bare render: 2 occurrences, opens with the final user turn -> 0.
    bare = [IMS, 10, 11, IMS, 20]
    assert pe(tok, bare) == 0
    assert pe(tok, bare) == RUN.final_turn_boundary(bare, IMS)

    # q25 render (default system turn, 3 occurrences) -> parent VERBATIM.
    q25 = [IMS, 1, 2, IMS, 10, IMS, 20]
    assert pe(tok, q25) == BANK94.prefix_end_index_multi(tok, q25) == 3
    assert pe(tok, q25) == RUN.final_turn_boundary(q25, IMS)

    # demo-history render (4 occurrences) -> parent VERBATIM.
    hist = [IMS, 1, IMS, 2, 3, IMS, 10, IMS, 20]
    assert pe(tok, hist) == BANK94.prefix_end_index_multi(tok, hist) == 5
    assert pe(tok, hist) == RUN.final_turn_boundary(hist, IMS)

    # single occurrence: not a chat render -> fail loud.
    with pytest.raises(AssertionError):
        pe(tok, [IMS, 10, 11])
    # 2 occurrences NOT opening the render -> fail loud (not the bare shape).
    with pytest.raises(AssertionError):
        pe(tok, [9, IMS, 10, IMS, 20])


def test_ce_slot_position_no_prefix_tolerance():
    """r5: the ce slot (last context token) is prefix_end-independent — pe >= 1
    records route through the parent ``slot_position`` VERBATIM (q25
    byte-equal, incl. completed-bank records that predate the ``no_prefix``
    field); pe == 0 records must carry the bank r5 ``no_prefix`` flag."""
    rec = {"context_id": "c", "ctx_len": 10, "prefix_end": 3, "no_prefix": False}
    assert RUN.ce_slot_position(rec) == R.slot_position(10, 3, "ce") == 9
    # completed q25 bank records lack the r5 field entirely -> parent path.
    assert RUN.ce_slot_position({"context_id": "c", "ctx_len": 10, "prefix_end": 3}) == 9
    # q35 bare render record: pe == 0 + flagged -> ctx_len - 1 directly.
    assert (
        RUN.ce_slot_position({"context_id": "b", "ctx_len": 6, "prefix_end": 0, "no_prefix": True})
        == 5
    )
    # unflagged pe == 0 is a corrupted record -> fail loud.
    with pytest.raises(AssertionError):
        RUN.ce_slot_position({"context_id": "b", "ctx_len": 6, "prefix_end": 0})


def test_gate_spot_position_pe_safe_dispatch():
    """r6: the donors-gate position seam — ce slots dispatch through
    ``ce_slot_position`` (q35 bare render: pe == 0 + ``no_prefix`` -> last
    context token; UNFLAGGED pe == 0 fails loud); non-ce slots and the parent
    default seam stay byte-equal to ``slot_position``."""
    bare = {"context_id": "bare__q1", "ctx_len": 6, "prefix_end": 0, "no_prefix": True}
    sysr = {"context_id": "sys__q2", "ctx_len": 9, "prefix_end": 3}
    assert RUN.gate_spot_position(bare, "ce") == 5
    assert RUN.gate_spot_position(sysr, "ce") == R.slot_position(9, 3, "ce") == 8
    assert RUN.gate_spot_position(sysr, "pe") == R.slot_position(9, 3, "pe") == 2
    with pytest.raises(AssertionError):
        RUN.gate_spot_position({"context_id": "b", "ctx_len": 6, "prefix_end": 0}, "ce")
    # Parent default seam: byte-equal to slot_position, incl. the pe assert
    # (existing issue2162 / ladder callers keep the crash-loud contract).
    assert R._default_spot_position(sysr, "ce") == 8
    with pytest.raises(AssertionError):
        R._default_spot_position(bare, "ce")


class _LeftPadSentinel(RuntimeError):
    """Raised by the monkeypatched ``_left_pad`` — reaching it proves the
    gate's spot loop got PAST the position computation (the leg-B crash
    point) without any model forward."""


def test_injection_gate_position_seam_q35_spot(monkeypatch):
    """r6 regression (leg-B crash shape): an S2 gate spot whose a-record is a
    q35 bare render (``a=bare__q1``, prefix_end == 0, ``no_prefix``) must flow
    through ``run_injection_gate``'s spot-position computation without
    tripping the parent ``slot_position`` assert. Model-free: ``_left_pad`` is
    stubbed to raise a sentinel, so the sentinel (not the assert) proves
    positions were computed for the WHOLE batch (flagged pe == 0 spot row +
    pe >= 1 companion row)."""

    def _boom(rows, pad_id, device):
        raise _LeftPadSentinel(str([len(r) for r in rows]))

    monkeypatch.setattr(R, "_left_pad", _boom)

    def _payload_stub(bank, pair, slot, arm, donor_maps, pairs_by_id):
        return torch.zeros(1, N_LAYERS, HIDDEN), None

    def _ids_stub(tok, c):
        return c["ids"]

    recs = {
        "bare__q1": {"context_id": "bare__q1", "ctx_len": 6, "prefix_end": 0, "no_prefix": True},
        "sys__q2": {"context_id": "sys__q2", "ctx_len": 9, "prefix_end": 3, "no_prefix": False},
    }
    p1 = SimpleNamespace(pair_id="s2_p1", a="bare__q1", b="bare__q9")
    p2 = SimpleNamespace(pair_id="s1_p2", a="sys__q2", b="sys__q9")
    contexts = {"bare__q1": {"ids": [7] * 6}, "sys__q2": {"ids": [7] * 9}}
    spots = [{"cell": "s2", "slot": "ce", "arm": "steered", "pair": p1}]
    cfg = SimpleNamespace(device="cpu", layers=[0, 1])
    kwargs = dict(contexts=contexts, ids_fn=_ids_stub, spots=spots, payload_fn=_payload_stub)
    # Pre-fix shape (parent default position path): the leg-B crash — the S2
    # spot's pe == 0 record trips ``assert 1 <= prefix_end < ctx_len``.
    with pytest.raises(AssertionError, match=r"\(6, 0\)"):
        R.run_injection_gate(cfg, None, FakeTok(), {"per_context": recs}, [p1, p2], {}, **kwargs)
    # Post-fix: the pe-safe seam computes positions for both rows and the gate
    # proceeds to the (stubbed) padded forward — the sentinel, not the assert.
    with pytest.raises(_LeftPadSentinel):
        R.run_injection_gate(
            cfg,
            None,
            FakeTok(),
            {"per_context": recs},
            [p1, p2],
            {},
            position_fn=RUN.gate_spot_position,
            **kwargs,
        )
    # Fail-loud preserved through the gate: an UNFLAGGED pe == 0 record still
    # asserts inside ``ce_slot_position`` even under the pe-safe seam.
    recs_bad = {**recs, "bare__q1": {"context_id": "bare__q1", "ctx_len": 6, "prefix_end": 0}}
    with pytest.raises(AssertionError, match=r"bare__q1"):
        R.run_injection_gate(
            cfg,
            None,
            FakeTok(),
            {"per_context": recs_bad},
            [p1, p2],
            {},
            position_fn=RUN.gate_spot_position,
            **kwargs,
        )


def test_capture_bank_no_prefix_pe_exclusion(tmp_path, monkeypatch):
    """r5 regression (fails pre-fix with the parent >=3-occurrence assert):
    ``capture_bank`` over a q35-shaped 2-occurrence bare render completes,
    flags the record ``no_prefix`` and OMITS ``v_pe`` (pre-fix the pe-1 read
    would have silently indexed position -1 = the LAST token); a 3-occurrence
    sibling keeps the parent-verbatim prefix_end + a position-correct v_pe.
    Real ``capture_bank`` body — only the model-forward boundary is faked
    (signature-conformant, position-addressable values)."""
    from unittest.mock import create_autospec

    def _fake_extract(
        model,
        input_ids,
        layers,
        *,
        attention_mask=None,
        return_logits=False,
        detach_to_cpu=False,
    ):
        b_n, t_n = input_ids.shape
        out = {}
        for layer in layers:
            vals = torch.zeros(b_n, t_n, HIDDEN)
            for b in range(b_n):
                for t in range(t_n):
                    vals[b, t] = layer * 1000 + b * 100 + t
            out[layer] = vals
        return out

    monkeypatch.setattr(
        RUN,
        "extract_layer_activations",
        create_autospec(RUN.extract_layer_activations, side_effect=_fake_extract),
    )
    cfg = _mk_cfg(tmp_path)
    tok = _ImsTok()
    ids = {
        "bare_q": [IMS, 10, 11, IMS, 20],  # q35 bare render: 2 occurrences
        "sys_q": [IMS, 1, 2, IMS, 10, IMS, 20],  # system turn: 3 occurrences
    }
    contexts = {cid: {"id": cid, "__set": "s2"} for cid in ids}
    bank = RUN.capture_bank(cfg, object(), tok, contexts, lambda _tok, c: ids[c["id"]], "test-fp")
    recs = bank["per_context"]
    assert sorted(recs) == ["bare_q", "sys_q"]

    bare = recs["bare_q"]
    assert bare["prefix_end"] == 0 and bare["no_prefix"] is True
    assert "v_pe" not in bare  # pe-exclusion: a slot-"pe" read KeyErrors
    assert bare["ctx_len"] == 5
    # v_ce at ctx_len-1 = position 4, row 0.
    for li, layer in enumerate(cfg.layers):
        assert torch.all(bare["v_ce"][li] == layer * 1000 + 0 * 100 + 4)

    sysr = recs["sys_q"]
    assert sysr["prefix_end"] == 3 and sysr["no_prefix"] is False
    assert sysr["ctx_len"] == 7
    for li, layer in enumerate(cfg.layers):
        # v_ce at position 6, v_pe at pe-1 = position 2, row 1.
        assert torch.all(sysr["v_ce"][li] == layer * 1000 + 1 * 100 + 6)
        assert torch.all(sysr["v_pe"][li] == layer * 1000 + 1 * 100 + 2)

    # the pe slot on the no-prefix record fails loud through the parent reader.
    with pytest.raises(KeyError):
        R._slot_state(bare, "pe")
    # chunk checkpoint landed (resume surface unchanged).
    assert list((cfg.bank_dir / "bank_chunks").glob("chunk_*.pt"))


def test_common_affix_non_overlapping():
    """Vendored #2329 ``_common_affix``: prefix counted first, suffix bounded
    so the two never overlap (diagnostic fields in the violation report)."""
    ca = RUN._common_affix
    assert ca([1, 2, 3, 4], [1, 2, 9, 4]) == (2, 1)
    assert ca([1, 2, 3], [1, 2, 3]) == (3, 0)
    assert ca([1, 1, 1], [1, 1]) == (2, 0)
    assert ca([], [1]) == (0, 0)


def test_regime_fingerprint_field_sensitivity(tmp_path):
    """Every output-affecting knob moves the fingerprint (r1 blocker
    incomplete-regime-fingerprint); num_workers deliberately does NOT (the
    anchors phase keys its own shards/done files on w{i}of{N} instead)."""
    import dataclasses

    cfg = _mk_cfg(tmp_path)
    base_fp = RUN.regime_fingerprint(cfg)
    moving = {
        "model_tag": "q35",
        "model_id": "other-model",
        "tiny": False,
        "n_layers": 3,
        "hidden": 64,
        "max_new_tokens": 2,
        "grid_draws": 3,
        "anchor_draws": 2,
        "gen_batch": 8,
        "capture_batch": 2,
        "seed_base": 12,
        "smoke": False,
    }
    for field, value in moving.items():
        fp = RUN.regime_fingerprint(dataclasses.replace(cfg, **{field: value}))
        assert fp != base_fp, f"regime_fingerprint insensitive to {field}"
    for field, value in (("num_workers", 8), ("worker_index", 3), ("upload_every", 99)):
        fp = RUN.regime_fingerprint(dataclasses.replace(cfg, **{field: value}))
        assert fp == base_fp, f"regime_fingerprint must NOT key on {field}"


def test_s2_donor_map_is_seeded_derangement(monkeypatch):
    """build_donor_maps installs the plan §4.2 NAMED FALLBACK for S2: the
    seed-23330 derangement over the 15 matched-query pair ids (parent
    recovery is AMBIGUOUS — 5/15 pairs multi-donor; provenance recorded)."""
    monkeypatch.chdir(REPO_ROOT)
    import issue2333_run as RUN2

    s1, s2 = RUN2.build_pair_universe()
    maps = RUN2.build_donor_maps(s1, s2)
    s2_ids = sorted(p.pair_id for p in s2)
    expected = RUN2.C.seeded_derangement(s2_ids, RUN2.C.S2_DERANGEMENT_SEED)
    realized = {pid: maps["shuffled"][pid] for pid in s2_ids}
    assert realized == expected
    assert all(k != v for k, v in realized.items())  # derangement: no fixed point
    assert sorted(realized.values()) == s2_ids  # bijection over the S2 set


def test_chunk_store_roundtrip_and_cross_regime_refusal(tmp_path):
    store = RUN._ChunkStore(tmp_path / "chunks", "fp-a")
    assert store.load("0001") is None
    store.save("0001", {"rows": [1, 2, 3]})
    assert store.load("0001") == {"rows": [1, 2, 3]}
    other = RUN._ChunkStore(tmp_path / "chunks", "fp-b")
    with pytest.raises(RuntimeError, match="cross-regime chunk resume"):
        other.load("0001")
    store.clear()
    assert store.load("0001") is None


def test_require_single_worker_guard(tmp_path):
    import dataclasses

    cfg = _mk_cfg(tmp_path)
    RUN._require_single_worker(cfg, "bank")  # w0of1 — fine
    with pytest.raises(RuntimeError, match="single-worker"):
        RUN._require_single_worker(dataclasses.replace(cfg, num_workers=8), "bank")
    with pytest.raises(RuntimeError, match="single-worker"):
        RUN._require_single_worker(
            dataclasses.replace(cfg, num_workers=8, worker_index=3), "donors"
        )


def test_anchors_stale_width_guard(tmp_path):
    """Mixed-width anchors shards fail LOUD (i::N partitions depend on N —
    num_workers is deliberately outside the global fingerprint)."""
    cfg = _mk_cfg(tmp_path)
    cfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    RUN._anchors_stale_width_guard(cfg)  # empty dir — fine
    (cfg.anchors_dir / "anchors_w0of1.jsonl").write_text("", encoding="utf-8")
    RUN._anchors_stale_width_guard(cfg)  # matching width (num_workers=1) — fine
    (cfg.anchors_dir / "anchors_w0of2.jsonl").write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="different --num-workers width"):
        RUN._anchors_stale_width_guard(cfg)


class _FakeTemplateTok:
    """apply_chat_template-shaped fake for the q35 thinking-off render assert."""

    def __init__(self, rendered: str):
        self.rendered = rendered

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        return self.rendered

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(max(4, len(text) // 8)))}


def test_q35_render_thinking_off_assert():
    """Qwen3.5 thinking-OFF = a CLOSED EMPTY <think></think> block (measured
    under transformers==5.15.0); an OPEN or non-empty block fails loud."""
    ids_fn = RUN.make_ids_fn("q35")
    ctx = {"id": "t", "system": None, "history": [], "user": "hi"}
    base = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n"
    assert ids_fn(_FakeTemplateTok(base + "<think>\n\n</think>\n\n"), ctx)  # closed empty: OK
    assert ids_fn(_FakeTemplateTok(base), ctx)  # no block at all: also OK
    with pytest.raises(AssertionError, match="OPEN thinking block"):
        ids_fn(_FakeTemplateTok(base + "<think>\n"), ctx)
    with pytest.raises(AssertionError, match="non-empty thinking block"):
        ids_fn(_FakeTemplateTok(base + "<think>\nreasoning...\n</think>\n\n"), ctx)


# ── r4 crash fix: banked-parent dual-schema reads (leg A q25 KeyError) ─
#
# Fixture layouts MIRROR the REAL pinned stores, probed r4 from
# superkaiba1/explore-persona-space-data:
#   issue2162_ctxinfo/analysis_tensors/vc_bank/vc_bank.pt @ PIN_2162 —
#     top-level {layers, per_context, donor_assignments, bank_sha, repro};
#     record keys {context_id, cell, value_id, carrier, ctx_len,
#     prefix_end, v_ce, v_pe}.
#   issue2094_singlepos/analysis_tensors/vc_bank/vc_bank.pt @ PIN_2094 —
#     top-level {layers, per_context, centroids, donor_derangement,
#     bank_sha, repro}; record keys {context_id, prefix, query_id, ctx_len,
#     prefix_end, nq, q_span, v_pe} — NO v_ce (the r4 crash).

_PAR_LAYERS = [0, 1, 2, 3]  # covers C.PARITY_EARLY_LAYERS
_PAR_H = 8


def _rec_2162_layout(cid: str, v_ce: torch.Tensor) -> dict:
    return {
        "context_id": cid,
        "cell": "instr_format",
        "value_id": "v1",
        "carrier": "d1",
        "ctx_len": 40,
        "prefix_end": 30,
        "v_ce": v_ce,
        "v_pe": torch.zeros_like(v_ce),
    }


def _rec_2094_layout(cid: str, q_span: torch.Tensor) -> dict:
    return {
        "context_id": cid,
        "prefix": "bare",
        "query_id": "q1",
        "ctx_len": 40,
        "prefix_end": 40 - q_span.shape[0],
        "nq": q_span.shape[0],
        "q_span": q_span,
        "v_pe": torch.zeros(q_span.shape[1:]),
    }


def test_banked_vce_2162_layout():
    v = torch.randn(len(_PAR_LAYERS), _PAR_H)
    out = RUN._banked_vce(_rec_2162_layout("instr_format::v1::d1", v))
    assert torch.equal(out, v.float())


def test_banked_vce_2094_layout_qspan_last_row():
    span = torch.randn(5, len(_PAR_LAYERS), _PAR_H)
    out = RUN._banked_vce(_rec_2094_layout("bare__q1", span))
    assert torch.equal(out, span[-1].float())  # position ctx_len-1 == span[-1]


def test_banked_vce_unrecognized_schema_raises():
    with pytest.raises(RuntimeError, match="unrecognized banked per-context schema"):
        RUN._banked_vce({"context_id": "x", "v_pe": torch.zeros(2, 2)})


def _write_parent_stores(bank_dir: Path, s1_vce: dict, s2_span: dict) -> None:
    """Tiny stores at the gate's staged names, mirroring the real layouts."""
    torch.save(
        {
            "layers": _PAR_LAYERS,
            "per_context": {cid: _rec_2162_layout(cid, v) for cid, v in s1_vce.items()},
            "donor_assignments": {"shuffled": {}, "crosstype": {}},
            "bank_sha": "s1sha",
            "repro": {},
        },
        bank_dir / "parent_vc_bank_s1.pt",
    )
    torch.save(
        {
            "layers": _PAR_LAYERS,
            "per_context": {cid: _rec_2094_layout(cid, s) for cid, s in s2_span.items()},
            "centroids": {},
            "donor_derangement": {},
            "bank_sha": "s2sha",
            "repro": {},
        },
        bank_dir / "parent_vc_bank_s2.pt",
    )


def test_capture_parity_gate_dual_schema_end_to_end(tmp_path):
    """Pre-r4 this failed KeyError 'v_ce' on the S2 (q_span-only) record —
    the exact leg A q25 crash. Runs the REAL gate body against real-layout
    stores on disk (staging skipped via pre-existing dests; no network)."""
    torch.manual_seed(0)
    s1_v = torch.randn(len(_PAR_LAYERS), _PAR_H)
    s2_span = torch.randn(6, len(_PAR_LAYERS), _PAR_H)
    _write_parent_stores(tmp_path, {"s1ctx": s1_v}, {"s2ctx": s2_span})
    fresh = {
        "layers": _PAR_LAYERS,
        "per_context": {
            "s1ctx": {
                "context_id": "s1ctx",
                "set": "s1",
                "ctx_len": 40,
                "prefix_end": 30,
                "v_ce": s1_v.clone(),
            },
            "s2ctx": {
                "context_id": "s2ctx",
                "set": "s2",
                "ctx_len": 40,
                "prefix_end": 34,
                "v_ce": s2_span[-1].clone(),
            },
        },
    }
    cfg = SimpleNamespace(bank_dir=tmp_path)
    parity = RUN.capture_parity_gate(cfg, fresh)
    assert parity["verdict"] == "PASS", parity
    assert parity["n_contexts"] == 2
    assert parity["worst_early_cos"] > 0.9999 and parity["worst_flat_cos"] > 0.9999

    # perturbed fresh -> designed FAIL with drift diagnostics, never a crash
    fresh["per_context"]["s2ctx"]["v_ce"] = -s2_span[-1].clone()
    bad = RUN.capture_parity_gate(cfg, fresh)
    assert bad["verdict"] == "FAIL"
    assert bad["failures"] and bad["failures"][0]["context_id"] == "s2ctx"
    assert bad["failures"][0]["ctx_len"] == (40, 40)

    # parent/fresh layer-list mismatch fails LOUD at load, not silently
    with pytest.raises(AssertionError):
        RUN.capture_parity_gate(cfg, {**fresh, "layers": [0, 1]})


def test_frozen_s1_shuffled_map_real_layout(tmp_path):
    """The bank.json key is donor_assignment (SINGULAR — real artifact @
    PIN_2162); the pre-r4 plural read is now a loud schema-drift error."""
    good = tmp_path / "bank.json"
    good.write_text(
        json.dumps(
            {
                "issue": 2162,
                "donor_assignment": {"shuffled": {"p1": "p2"}, "crosstype": {"p1": "p3"}},
            }
        )
    )
    assert RUN._frozen_s1_shuffled_map(good) == {"p1": "p2"}

    plural = tmp_path / "bank_plural.json"  # the exact pre-r4 wrong assumption
    plural.write_text(json.dumps({"donor_assignments": {"shuffled": {"p1": "p2"}}}))
    with pytest.raises(RuntimeError, match="schema drift"):
        RUN._frozen_s1_shuffled_map(plural)


def test_build_donor_maps_survivor_safe_below_threshold_drop(monkeypatch):
    """r4 scope extension (r3 Codex `consumer-contract-post-init`): donor maps
    are rebuilt over the SURVIVOR set after below-threshold minpair drops.

    (a) every survivor's shuffled-donor reference resolves within the
        survivor set (the q35 ce-control `pairs_by_id[donor_id]` lookup and
        the grid `pair_dropped_for_arm` null lookup both stay in-set);
    (b) no silent cascade-drop: dropped ids appear NEITHER as recipients NOR
        as donors, and the S2 derangement over survivors keeps full coverage;
    (c) an S1 survivor orphaned by its frozen donor's drop is a LOUD
        RuntimeError (frozen value-constrained map preserved, never
        re-assigned) — and the wholesale rc=26 minpair gate still precedes
        the donor rebuild in phase_bank.
    """
    monkeypatch.chdir(REPO_ROOT)
    s1, s2 = RUN.build_pair_universe()

    # -- S2 drop (1/15 << 10% wholesale threshold): survivor-safe rebuild
    dropped = {sorted(p.pair_id for p in s2)[0]}
    maps = RUN.build_donor_maps(s1, s2, dropped=dropped)
    surv_s2 = sorted(p.pair_id for p in s2 if p.pair_id not in dropped)
    all_ids = {p.pair_id for p in [*s1, *s2]} - dropped
    # (a) every survivor resolves its donor within the survivor universe
    pairs_by_id = {p.pair_id: p for p in [*s1, *s2] if p.pair_id not in dropped}
    for pid, donor_id in maps["shuffled"].items():
        assert pid in all_ids and donor_id in all_ids, (pid, donor_id)
        assert pairs_by_id[donor_id] is not None  # the ce-control lookup shape
    # (b) no cascade: dropped id absent as recipient AND as donor; S2 survivors
    # fully covered by the recorded-seed derangement over the survivor ids
    assert not (dropped & set(maps["shuffled"])), "dropped id kept as recipient"
    assert not (dropped & set(maps["shuffled"].values())), "dropped id kept as donor"
    realized_s2 = {pid: maps["shuffled"][pid] for pid in surv_s2}
    assert realized_s2 == RUN.C.seeded_derangement(surv_s2, RUN.C.S2_DERANGEMENT_SEED)
    assert sorted(realized_s2.values()) == surv_s2 and all(k != v for k, v in realized_s2.items())

    # (c) S1 orphaned survivor -> LOUD refusal (frozen map, never re-assigned)
    full = RUN.build_donor_maps(s1, s2)
    some_s1_donor = next(d for pid, d in full["shuffled"].items() if pid in {p.pair_id for p in s1})
    with pytest.raises(RuntimeError, match="frozen value-constrained map"):
        RUN.build_donor_maps(s1, s2, dropped={some_s1_donor})

    # (c) wholesale-break ordering pin: rc=26 gate fires BEFORE the rebuild
    import inspect

    src = inspect.getsource(RUN.phase_bank)
    assert src.index("RC_MINPAIR_GATE") < src.index("build_donor_maps"), (
        "phase_bank must evaluate the wholesale minpair gate before the survivor-set donor rebuild"
    )
