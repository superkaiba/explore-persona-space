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
    """The REAL run_block_2333 body on a prefill/steered block: (a) the
    cap-hit regen branch fires (max_new_tokens=1 caps every non-EOS row;
    regenerated rows carry regenerated_at == 2x cap); (b) the prefill seam
    is an ID concatenation — donor ids verbatim, response_text ==
    donor_text + continuation_text; (c) V_a spans cover donor+continuation."""
    cfg = _mk_cfg(tmp_path)
    tok = FakeTok()
    pairs = [
        SimpleNamespace(pair_id="p1", a="ctxA1", b="ctxB1", cell="cellX"),
        SimpleNamespace(pair_id="p2", a="ctxA2", b="ctxB2", cell="cellX"),
    ]
    pairs_by_id = {p.pair_id: p for p in pairs}
    ctx_ids = {"ctxA1": [3, 4, 5, 6], "ctxA2": [7, 8, 9]}
    donors = {"med": {"p1": _donor_rec([21, 22]), "p2": _donor_rec([23, 24, 25])}}
    donor_maps = {"shuffled": {"p1": "p2", "p2": "p1"}}
    bank = {"per_context": {}}
    block = R.Block("cellX", "prefill2_med", "steered", ("p1", "p2"))

    RUN.run_block_2333(
        cfg, tiny_model, tok, bank, donors, donor_maps, pairs_by_id, ctx_ids, block, "test-fp"
    )

    shard = cfg.rollouts_dir / "blocks" / f"{block.slug}.jsonl"
    rows = [json.loads(line) for line in shard.read_text().splitlines() if line.strip()]
    assert len(rows) == 2 * cfg.grid_draws  # 2 pairs x 2 draws
    n_regen = sum(1 for r in rows if r.get("regenerated_at") == 2 * cfg.max_new_tokens)
    assert n_regen >= 1, "cap-hit regen branch never fired"
    for r in rows:
        assert r["donor_len"] == 2  # k=2 donor ids verbatim
        assert r["response_text"] == r["donor_text"] + r["continuation_text"]
        assert r["kind"] == "prefill" and r["variant"] == "steered"
        assert r["cap_hit_basis"] == "gen_token_count"
    # V_a store: one (L, H) summary per (pair, draw), fp16.
    va = torch.load(cfg.va_dir / f"{block.slug}.pt", map_location="cpu", weights_only=False)
    assert len(va) == 4
    for key, t in va.items():
        assert t.shape == (N_LAYERS, HIDDEN), key
        assert t.dtype == torch.float16
    # Done file lands in the SMOKE namespace (cfg.smoke=True) — the claim
    # queue's resume predicate reads the same namespace (regression pin: a
    # "blocks"-default write made the smoke queue re-run blocks forever).
    done_path = R.block_done_path(cfg.out_root, block, "smoke_blocks")
    done = json.loads(done_path.read_text())
    assert done["n_rows"] == 4 and done["n_pairs_kept"] == 2
    assert done["n_cap_hit"] == sum(1 for r in rows if r["cap_hit"])
    assert R.block_is_done(cfg.out_root, block, "test-fp", "smoke_blocks")
    assert not R.block_done_path(cfg.out_root, block, "blocks").exists()


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
    shard = cfg.rollouts_dir / "blocks" / f"{block.slug}.jsonl"
    rows = [json.loads(line) for line in shard.read_text().splitlines() if line.strip()]
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
