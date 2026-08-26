"""CPU-only pins for scripts/issue2587_battery_run.py (units 3a + 3b).

No network, no HF fetch, no GPU (the units 1-2 pattern): the tokenizer is a
deterministic chat-template-shaped fake; ``generate_batch`` is faked ONLY at
the GPU boundary via ``unittest.mock.create_autospec`` (signature-conformant
by construction, per the one-production-body-test rule) — every body under
test (``_generate_cell`` / ``_gen_cell`` / ``_pilot_gate`` / ``phase_gen``)
executes for real. The pinned issue2162_run import (``M._r()``) is a LOCAL
``git show`` of an object already in the shared odb — no network.

Unit 3b (capture + embed): the CAPTURE tests run the REAL sha-pinned
``capture_answer_states`` body + the REAL ``extract_layer_activations`` hooks
+ the REAL hook probe against a tiny random-weight 32-layer Qwen2 on CPU
(production layer indices {16, 22, 30} exist; no substituted implementation);
the EMBED tests fake ONLY the vLLM engine ctor / embed-model tokenizer /
HfApi-revision network boundaries (autospec'd or signature-shaped), with
``phase_embed`` / ``_embed_rows`` / ``run_engine_parity_probe`` bodies
executing for real.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
import torch

SCRIPTS_DIR = Path(__file__).resolve().parent.parent / "scripts"
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

import issue2587_battery_run as M  # noqa: E402

from explore_persona_space.experiments.issue1415.steering import (  # noqa: E402
    generate_batch as real_generate_batch,
)

# ── fakes ─────────────────────────────────────────────────────────────────


class FakeTok:
    """Chat-template-shaped fake (unit-1 `_FakeTemplateTok` pattern): asserts
    thinking-off is threaded; deterministic per-word ids; closed-empty-think
    render tail so the #2333 assert passes."""

    pad_token_id = 0
    padding_side = "left"
    name_or_path = "fake-q35"

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        assert kw.get("enable_thinking") is False, "render must pass enable_thinking=False"
        body = " ".join(m["content"] for m in msgs)
        return body + " <|im_start|>assistant\n<think>\n\n</think>\n\n"

    def __call__(self, text, add_special_tokens=False):
        ids = [10 + (sum(map(ord, w)) % 30000) for w in text.split()]
        return {"input_ids": ids or [11]}

    def convert_tokens_to_ids(self, tok_str):
        assert tok_str == "<|im_end|>", tok_str
        return 7


def _mini_bank(axes_counts: dict[str, int]) -> dict:
    contexts = {}
    for ax, n in axes_counts.items():
        for j in range(n):
            cid = f"{ax}_c{j:03d}"
            contexts[cid] = {
                "id": cid,
                "cell": ax,
                "kind": "battery",
                "value_id": f"v{j % 3}",
                "carrier": f"carrier{j % 4}",
                "form": "a",
                "system": f"sys {ax} {j} alpha beta",
                "user": f"user {ax} {j} gamma delta",
            }
    return {
        "contexts": contexts,
        "n_contexts": len(contexts),
        "n_pairs": 0,
        "values_sha256": "cafebabe",
    }


def _cfg(tmp_path: Path, axes, **kw) -> M.Cfg:
    defaults = dict(
        phase="gen",
        out_root=tmp_path / "out",
        model_id="fake-model",
        model_revision="deadbeefrev",
        device="cpu",
        gen_batch=2,
        draws=2,
        max_new_tokens=M.ANCHOR_MAX_NEW,
        seed_base=42,
        upload="none",
        axes=tuple(axes),
        shard_index=0,
        num_shards=1,
        max_carriers=None,
        hf_repo="fake/repo",
        hf_prefix="issue2587_minpair",
        bank_values_sha="cafebabe",
        pilot_ceiling_h=M.PILOT_CEILING_H,
    )
    defaults.update(kw)
    return M.Cfg(**defaults)


def _default_text(ctx, draw, max_new):
    return f"resp {ctx['id']} d{draw} lorem ipsum"


def _mk_fake_gen(text_fn=_default_text):
    """Signature-conformant GPU-boundary fake (create_autospec of the REAL
    generate_batch — never a bare Mock)."""

    def impl(
        model,
        tokenizer,
        contexts,
        n=10,
        hook=None,
        max_new_tokens=1024,
        temperature=1.0,
        seed_base=42,
        render_fn=None,
        ids_fn=None,
        top_p=None,
    ):
        assert hook is None
        assert render_fn is not None and ids_fn is not None
        return [[text_fn(ctx, i, max_new_tokens) for i in range(n)] for ctx in contexts]

    return mock.create_autospec(real_generate_batch, side_effect=impl)


# ── constants + parser + seams ────────────────────────────────────────────


def test_plan_constants_pinned():
    assert M.ANCHOR_DRAWS == 10
    assert M.GEN_BATCH == 16
    assert M.ANCHOR_MAX_NEW == 2048
    assert M.REGEN_MAX_NEW == 4096
    assert M.CAP_HIT_REGEN_FRAC == 0.02
    assert M.ANCHOR_TEMPERATURE == 1.0
    assert M.PILOT_CEILING_H == 6.0
    assert M.EXIT_PILOT_REFUSE == 7
    assert M.PHASES == ("gen", "capture", "embed")
    assert M.HF_PREFIX == "issue2587_minpair"
    # unit 3b pins (plan §4.4)
    assert M.CAPTURE_BATCH == 8
    assert tuple(range(32)) == M.CAPTURE_LAYERS
    assert M.HOOK_PROBE_LAYERS == (16, 22, 30)
    assert M.HOOK_REL_TOL == 1e-5
    assert M.EMBED_MODEL == "Qwen/Qwen3-Embedding-8B"
    assert M.EMBED_DIM == 4096
    assert M.EMBED_CHUNK == 2500
    assert M.EMBED_MAX_MODEL_LEN == 8192
    assert M.EXPECTED_EMBED_ENGINE == "0.11.0"
    assert M.EXIT_PARITY_MISS == 8


def test_expected_embed_engine_matches_repo_lock():
    """The parity reference IS the repo uv.lock vllm pin — a lock bump must
    consciously revisit EXPECTED_EMBED_ENGINE (forward drift fails loud)."""
    lock = (Path(__file__).resolve().parent.parent / "uv.lock").read_text(encoding="utf-8")
    assert f'name = "vllm"\nversion = "{M.EXPECTED_EMBED_ENGINE}"' in lock


def test_parser_defaults_and_required():
    ap = M.build_argparser()
    with pytest.raises(SystemExit):
        ap.parse_args([])  # --phase is required
    args = ap.parse_args(["--phase", "gen"])
    assert args.draws == M.ANCHOR_DRAWS
    assert args.gen_batch == M.GEN_BATCH
    assert args.max_new_tokens == M.ANCHOR_MAX_NEW
    assert args.seed_base == 42
    assert args.upload == "hf"
    assert args.num_shards == 1 and args.shard_index == 0
    assert args.axes is None and args.max_carriers is None
    assert args.pilot_ceiling_h == M.PILOT_CEILING_H
    # unit 3b flags
    assert args.capture_batch == M.CAPTURE_BATCH
    assert args.capture_dtype == "float32"
    assert args.embed_chunk == M.EMBED_CHUNK
    assert args.embed_max_model_len == M.EMBED_MAX_MODEL_LEN
    assert args.embed_pilot_ceiling_h == M.EMBED_PILOT_CEILING_H
    assert args.parity_report is None and args.anchors_root is None
    assert args.parity_probe_out is None and args.parity_cos_min == M.PARITY_COS_MIN


def test_phase_registry_wired():
    assert {
        "gen": M.phase_gen,
        "capture": M.phase_capture,
        "embed": M.phase_embed,
    } == M.PHASE_FNS


# ── sharding + slicing ────────────────────────────────────────────────────


def test_shard_axes_deterministic_disjoint_complete():
    counts = {"a": 400, "b": 300, "c": 200, "d": 100, "e": 80}
    assign = M.shard_axes(counts, 2)
    assert assign == {"a": 0, "b": 1, "c": 1, "d": 0, "e": 0}
    # complete + disjoint by construction; single shard maps all to 0
    assert set(assign) == set(counts)
    assert M.shard_axes(counts, 1) == dict.fromkeys(counts, 0)
    # insertion-order invariance
    assert M.shard_axes(dict(reversed(list(counts.items()))), 2) == assign


def test_apply_max_carriers_deterministic_subset():
    ctxs = _mini_bank({"alpha": 8})["contexts"].values()
    ctxs = sorted(ctxs, key=lambda c: c["id"])
    kept = M.apply_max_carriers(ctxs, 2)
    assert {c["carrier"] for c in kept} == {"carrier0", "carrier1"}
    assert M.apply_max_carriers(ctxs, None) == list(ctxs)


def test_group_contexts_by_cell_covers_bank():
    bank = _mini_bank({"alpha": 3, "beta": 2})
    by = M.group_contexts_by_cell(bank)
    assert sorted(by) == ["alpha", "beta"]
    assert [c["id"] for c in by["alpha"]] == sorted(c["id"] for c in by["alpha"])
    bank["n_contexts"] = 99
    with pytest.raises(AssertionError):
        M.group_contexts_by_cell(bank)


def test_resolve_axes_explicit_and_shard_split():
    bank = _mini_bank({"alpha": 3, "beta": 2, "gamma": 2})
    by = M.group_contexts_by_cell(bank)
    ap = M.build_argparser()
    args = ap.parse_args(["--phase", "gen", "--axes", "beta,alpha"])
    assert M.resolve_axes(args, by) == ("beta", "alpha")
    with pytest.raises(RuntimeError, match="unknown axes"):
        M.resolve_axes(ap.parse_args(["--phase", "gen", "--axes", "nope"]), by)
    a0 = M.resolve_axes(ap.parse_args(["--phase", "gen", "--num-shards", "2"]), by)
    a1 = M.resolve_axes(
        ap.parse_args(["--phase", "gen", "--num-shards", "2", "--shard-index", "1"]), by
    )
    assert set(a0) | set(a1) == set(by) and not set(a0) & set(a1)


def test_build_cfg_shard_out_root_suffix(tmp_path):
    ap = M.build_argparser()
    args = ap.parse_args(
        [
            "--phase",
            "gen",
            "--out-root",
            str(tmp_path / "r"),
            "--num-shards",
            "2",
            "--shard-index",
            "1",
        ]
    )
    cfg = M.build_cfg(args, bank_values_sha="x", axes=("alpha",), model_revision="rev")
    assert cfg.out_root == tmp_path / "r" / "shard1"
    args1 = ap.parse_args(["--phase", "gen", "--out-root", str(tmp_path / "r")])
    cfg1 = M.build_cfg(args1, bank_values_sha="x", axes=("alpha",), model_revision="rev")
    assert cfg1.out_root == tmp_path / "r"


# ── fingerprints ──────────────────────────────────────────────────────────


def test_regime_fp_sensitivity(tmp_path):
    base = _cfg(tmp_path, ("alpha",))
    fp = M._regime_fp(base)
    assert fp == M._regime_fp(_cfg(tmp_path, ("beta", "gamma")))  # axes NOT in the fp
    for kw in (
        {"draws": 3},
        {"max_new_tokens": 4096},
        {"seed_base": 7},
        {"max_carriers": 2},
        {"gen_batch": 4},
        {"model_revision": "otherrev"},
        {"bank_values_sha": "other"},
    ):
        assert M._regime_fp(_cfg(tmp_path, ("alpha",), **kw)) != fp, kw
    # upload is NOT in the cell-grain fp (r1 g3: an --upload none -> hf flip
    # re-uploads banked rows, never regenerates) ...
    hf = _cfg(tmp_path, ("alpha",), upload="hf")
    assert M._regime_fp(hf) == fp
    assert M._cell_fp(hf, "gen", "alpha") == M._cell_fp(base, "gen", "alpha")
    # ... but it IS in the sentinel fp (a none-run sentinel never satisfies hf)
    assert M._regime_fp(base, {"phase": "gen", "axes": ["alpha"], "upload": base.upload}) != (
        M._regime_fp(hf, {"phase": "gen", "axes": ["alpha"], "upload": hf.upload})
    )
    assert M._cell_fp(base, "gen", "alpha") != M._cell_fp(base, "gen", "beta")


def test_read_jsonl_torn_tail(tmp_path):
    p = tmp_path / "x.jsonl"
    p.write_text('{"a": 1}\n{"b": 2}\n{"torn', encoding="utf-8")
    assert M._read_jsonl(p, tolerate_torn_tail=True) == [{"a": 1}, {"b": 2}]
    with pytest.raises(json.JSONDecodeError):
        M._read_jsonl(p)


# ── _generate_cell: schema, checkpointing, resume, quarantine ─────────────


def test_generate_cell_row_schema_and_chunking(tmp_path, monkeypatch):
    fake = _mk_fake_gen()
    monkeypatch.setattr(M, "generate_batch", fake)
    cfg = _cfg(tmp_path, ("alpha",))
    tok = FakeTok()
    ctxs = M.group_contexts_by_cell(_mini_bank({"alpha": 3}))["alpha"]
    eot = M._r().eot_tail_ids(tok)
    assert eot == [7, 11]
    rows = M._generate_cell(cfg, object(), tok, eot, "alpha", ctxs, cfg.max_new_tokens)
    assert fake.call_count == 2  # 3 ctxs at gen_batch=2 -> 2 chunks
    assert len(rows) == 3 * cfg.draws
    r = rows[0]
    assert r["seed"] == cfg.seed_base + r["draw"]
    assert r["cap_hit_basis"] == "retokenized_completion_len >= max_new_tokens"
    assert r["span_start"] == r["ctx_len"]
    assert r["span_end"] == r["ctx_len"] + r["n_completion_tokens_gen"]
    assert r["tail_end"] == r["span_end"] + len(eot)
    assert r["cap_hit"] is False and r["max_new_tokens"] == M.ANCHOR_MAX_NEW
    assert [(x["chunk"], x["context_id"], x["draw"]) for x in rows] == sorted(
        (x["chunk"], x["context_id"], x["draw"]) for x in rows
    )
    part = cfg.anchors_dir / f"anchors_alpha.max{M.ANCHOR_MAX_NEW}.partial"
    lines = part.read_text(encoding="utf-8").strip().split("\n")
    header = json.loads(lines[0])
    assert header["partial_header"] == 1 and len(lines) == 1 + len(rows)


def test_generate_cell_resume_skips_complete_chunks(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, ("alpha",))
    tok = FakeTok()
    ctxs = M.group_contexts_by_cell(_mini_bank({"alpha": 3}))["alpha"]
    eot = [7, 11]

    def first_text(ctx, draw, max_new):
        return f"first {ctx['id']} d{draw}"

    monkeypatch.setattr(M, "generate_batch", _mk_fake_gen(first_text))
    M._generate_cell(cfg, object(), tok, eot, "alpha", ctxs, cfg.max_new_tokens)
    part = cfg.anchors_dir / f"anchors_alpha.max{M.ANCHOR_MAX_NEW}.partial"
    lines = part.read_text(encoding="utf-8").strip().split("\n")
    keep = [lines[0]] + [ln for ln in lines[1:] if json.loads(ln)["chunk"] == 0]
    part.write_text("\n".join(keep) + "\n", encoding="utf-8")

    def second_text(ctx, draw, max_new):
        return f"second {ctx['id']} d{draw}"

    fake2 = _mk_fake_gen(second_text)
    monkeypatch.setattr(M, "generate_batch", fake2)
    rows = M._generate_cell(cfg, object(), tok, eot, "alpha", ctxs, cfg.max_new_tokens)
    assert fake2.call_count == 1  # only chunk 1 regenerated
    assert len(rows) == 3 * cfg.draws
    assert all(r["text"].startswith("first ") for r in rows if r["chunk"] == 0)
    assert all(r["text"].startswith("second ") for r in rows if r["chunk"] == 1)


def test_generate_cell_quarantines_foreign_fp_partial(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path, ("alpha",))
    cfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    part = cfg.anchors_dir / f"anchors_alpha.max{M.ANCHOR_MAX_NEW}.partial"
    part.write_text(
        json.dumps({"partial_header": 1, "regime_fp": "stalefp"}) + "\n", encoding="utf-8"
    )
    fake = _mk_fake_gen()
    monkeypatch.setattr(M, "generate_batch", fake)
    ctxs = M.group_contexts_by_cell(_mini_bank({"alpha": 3}))["alpha"]
    M._generate_cell(cfg, object(), FakeTok(), [7, 11], "alpha", ctxs, cfg.max_new_tokens)
    assert fake.call_count == 2  # nothing resumed from the stale partial
    quarantined = list(cfg.quarantine_dir.glob("*.partial"))
    assert len(quarantined) == 1
    assert json.loads(quarantined[0].read_text().split("\n")[0])["regime_fp"] == "stalefp"


# ── _gen_cell: cap-hit re-gen, think-leak, manifest ───────────────────────


def test_gen_cell_cap_hit_regen_at_4096(tmp_path, monkeypatch):
    def text_fn(ctx, draw, max_new):
        # every 2048-call completion hits the cap; 4096 re-gen comes back short
        return "w " * max_new if max_new == M.ANCHOR_MAX_NEW else f"short {ctx['id']} d{draw}"

    fake = _mk_fake_gen(text_fn)
    monkeypatch.setattr(M, "generate_batch", fake)
    cfg = _cfg(tmp_path, ("alpha",))
    ctxs = M.group_contexts_by_cell(_mini_bank({"alpha": 2}))["alpha"]
    M._gen_cell(cfg, object(), FakeTok(), [7, 11], "alpha", ctxs)
    capped = cfg.anchors_dir / f"anchors_alpha.capped{M.ANCHOR_MAX_NEW}.jsonl"
    assert capped.is_file()
    final = M._read_jsonl(cfg.anchors_dir / "anchors_alpha.jsonl")
    assert all(r["max_new_tokens"] == M.REGEN_MAX_NEW and not r["cap_hit"] for r in final)
    man = json.loads((cfg.manifest_dir / "anchors_alpha.done.json").read_text())
    assert man["cap_hit_frac"] == 1.0
    assert man["cap_hit_frac_regen"] == 0.0
    assert man["max_new_tokens_final"] == M.REGEN_MAX_NEW
    assert man["capture_max_model_len_floor"] == man["max_ctx_len"] + 2 * M.REGEN_MAX_NEW
    assert man["regime_fp"] == M._cell_fp(cfg, "gen", "alpha")
    assert "git_commit" in man["repro"] and man["repro"]["phase"] == "gen"
    # both call-grain partials are gone after the manifest lands
    assert not list(cfg.anchors_dir.glob("*.partial"))


def test_gen_cell_think_leak_flagged_below_threshold(tmp_path, monkeypatch):
    def text_fn(ctx, draw, max_new):
        if ctx["id"].endswith("c000") and draw == 0:
            return "<think> leaked reasoning"
        return f"clean {ctx['id']} d{draw}"

    monkeypatch.setattr(M, "generate_batch", _mk_fake_gen(text_fn))
    cfg = _cfg(tmp_path, ("alpha",), draws=4, gen_batch=16)
    ctxs = M.group_contexts_by_cell(_mini_bank({"alpha": 26}))["alpha"]  # 104 rows, 1 leak
    M._gen_cell(cfg, object(), FakeTok(), [7, 11], "alpha", ctxs)
    rows = M._read_jsonl(cfg.anchors_dir / "anchors_alpha.jsonl")
    assert sum(r["think_leak"] for r in rows) == 1
    man = json.loads((cfg.manifest_dir / "anchors_alpha.done.json").read_text())
    assert man["think_leak"] == {"n": 104, "n_leaked": 1, "frac": 1 / 104}


def test_gen_cell_think_leak_hard_assert_fails_loud(tmp_path, monkeypatch):
    def text_fn(ctx, draw, max_new):
        if ctx["id"].endswith("c000") and draw == 0:
            return "<think> leaked reasoning"
        return f"clean {ctx['id']} d{draw}"

    monkeypatch.setattr(M, "generate_batch", _mk_fake_gen(text_fn))
    cfg = _cfg(tmp_path, ("alpha",))
    ctxs = M.group_contexts_by_cell(_mini_bank({"alpha": 4}))["alpha"]  # 8 rows, 1 leak
    with pytest.raises(AssertionError, match="think-leak gen:alpha"):
        M._gen_cell(cfg, object(), FakeTok(), [7, 11], "alpha", ctxs)
    # flagged rows persisted for diagnosis; NOT marked done; partial retained
    assert (cfg.anchors_dir / "anchors_alpha.jsonl").is_file()
    assert not (cfg.manifest_dir / "anchors_alpha.done.json").is_file()
    assert (cfg.anchors_dir / f"anchors_alpha.max{M.ANCHOR_MAX_NEW}.partial").is_file()


# ── phase_gen: foreign-axis guard, pilot gate, sentinel idempotence ───────


def test_foreign_axis_file_guard(tmp_path):
    cfg = _cfg(tmp_path, ("alpha",))
    cfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    (cfg.anchors_dir / f"anchors_alpha.capped{M.ANCHOR_MAX_NEW}.jsonl").write_text("")
    M._assert_no_foreign_axis_files(cfg)  # own capped file is fine
    (cfg.anchors_dir / "anchors_beta.jsonl").write_text("")
    with pytest.raises(RuntimeError, match="foreign axis file"):
        M._assert_no_foreign_axis_files(cfg)


def test_phase_gen_e2e_upload_none_and_idempotent(tmp_path, monkeypatch):
    bank = _mini_bank({"alpha": 3, "beta": 2})
    fake = _mk_fake_gen()
    monkeypatch.setattr(M, "generate_batch", fake)
    cfg = _cfg(tmp_path, ("alpha", "beta"))
    rc = M.phase_gen(cfg, bank, object(), FakeTok())
    assert rc == 0
    assert fake.call_count == 2 + 2 + 1  # pilot (warm + timed) + alpha chunks + beta chunk
    sent = json.loads((cfg.out_root / "battery_gen_done.json").read_text())
    assert sent["regime_fp"] == M._regime_fp(
        cfg, {"phase": "gen", "axes": sorted(cfg.axes), "upload": cfg.upload}
    )
    assert sent["upload"] == {"mode": "none"}
    assert set(sent["cells"]) == {"alpha", "beta"}
    assert sent["cells"]["alpha"]["n_rows"] == 3 * cfg.draws
    report = json.loads((cfg.manifest_dir / "pilot_gate_report.json").read_text())
    assert report["verdict"] == "proceed"
    for ax in ("alpha", "beta"):
        assert (cfg.anchors_dir / f"anchors_{ax}.jsonl").is_file()
    # idempotent re-run: sentinel + done manifests short-circuit everything
    fake2 = _mk_fake_gen()
    monkeypatch.setattr(M, "generate_batch", fake2)
    assert M.phase_gen(cfg, bank, object(), FakeTok()) == 0
    assert fake2.call_count == 0


def test_phase_gen_upload_flip_none_to_hf_uploads_without_regenerating(tmp_path, monkeypatch):
    """r1 g3: ``upload`` lives in the SENTINEL fp / completion predicate, not
    the per-cell fingerprint — flipping ``--upload none -> hf`` on the same
    out-root UPLOADS the banked rollouts and rewrites the sentinel, with ZERO
    generation calls (never a 10,800-rollout GPU regen)."""
    bank = _mini_bank({"alpha": 3, "beta": 2})
    monkeypatch.setattr(M, "generate_batch", _mk_fake_gen())
    cfg_none = _cfg(tmp_path, ("alpha", "beta"))  # upload="none"
    assert M.phase_gen(cfg_none, bank, object(), FakeTok()) == 0
    manifests_before = {
        ax: (cfg_none.manifest_dir / f"anchors_{ax}.done.json").read_text()
        for ax in ("alpha", "beta")
    }

    cfg_hf = _cfg(tmp_path, ("alpha", "beta"), upload="hf")
    fake_gen2 = _mk_fake_gen()
    monkeypatch.setattr(M, "generate_batch", fake_gen2)
    up_calls: list[dict] = []

    def _fake_upload(local_dir, repo_id, prefix, *, shard_glob, resume_skip, delete_local):
        names = sorted(p.name for p in Path(local_dir).glob(shard_glob))
        up_calls.append({"dir": Path(local_dir), "prefix": prefix, "names": names})
        return SimpleNamespace(
            repo_id=repo_id, uploaded=list(names), rerouted=[], skipped_existing=[]
        )

    monkeypatch.setattr(
        M,
        "upload_dir_sharded",
        mock.create_autospec(M.upload_dir_sharded, side_effect=_fake_upload),
    )
    assert M.phase_gen(cfg_hf, bank, object(), FakeTok()) == 0
    assert fake_gen2.call_count == 0  # NO pilot, NO regeneration — cells stay done
    assert len(up_calls) == 1  # the banked rollouts were re-uploaded
    assert up_calls[0]["names"] == ["anchors_alpha.jsonl", "anchors_beta.jsonl"]
    # per-cell done manifests untouched (fp excludes upload)
    for ax, before in manifests_before.items():
        assert (cfg_hf.manifest_dir / f"anchors_{ax}.done.json").read_text() == before
    # the sentinel was REWRITTEN at the hf fp: a later hf re-run short-circuits
    sent = json.loads((cfg_hf.out_root / "battery_gen_done.json").read_text())
    assert sent["regime_fp"] == M._regime_fp(
        cfg_hf, {"phase": "gen", "axes": sorted(cfg_hf.axes), "upload": "hf"}
    )
    assert sent["upload"]["mode"] == "hf" and sent["upload"]["anchors"]["uploaded"] == 2
    # and the none-run sentinel could never have satisfied the hf run
    assert sent["regime_fp"] != M._regime_fp(
        cfg_none, {"phase": "gen", "axes": sorted(cfg_none.axes), "upload": "none"}
    )


def test_phase_gen_pilot_refuse_exits_before_generation(tmp_path, monkeypatch):
    bank = _mini_bank({"alpha": 3})
    fake = _mk_fake_gen()
    monkeypatch.setattr(M, "generate_batch", fake)
    cfg = _cfg(tmp_path, ("alpha",), pilot_ceiling_h=-1.0)  # any projection refuses
    rc = M.phase_gen(cfg, bank, object(), FakeTok())
    assert rc == M.EXIT_PILOT_REFUSE == 7
    assert fake.call_count == 2  # warmup + timed only; no production generation
    assert not list(cfg.anchors_dir.glob("anchors_*.jsonl"))
    assert not (cfg.out_root / "battery_gen_done.json").is_file()
    report = json.loads((cfg.manifest_dir / "pilot_gate_report.json").read_text())
    assert report["verdict"] == "refuse" and report["ceiling_h"] == -1.0


def test_phase_gen_unknown_axis_fails_loud(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "generate_batch", _mk_fake_gen())
    cfg = _cfg(tmp_path, ("nope",))
    with pytest.raises(RuntimeError, match="unknown axes"):
        M.phase_gen(cfg, _mini_bank({"alpha": 2}), object(), FakeTok())


# ── import-check (real pinned imports + ported signature binds) ───────────


def test_import_check_passes_in_process(capsys):
    M._import_check()
    assert "ok" in capsys.readouterr().out


# ── unit 3b: capture (real pinned body + real hooks on a tiny 32-layer LM) ─


@pytest.fixture(scope="module")
def tiny_model():
    """Random-weight 32-layer Qwen2 on CPU: the PRODUCTION block count (probe
    layers {16, 22, 30} exist; the loader's 32-block assert holds) at tiny
    width — the real capture / hook-probe code paths run unfaked."""
    from transformers import Qwen2Config, Qwen2ForCausalLM

    cfg = Qwen2Config(
        vocab_size=32768,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=32,
        num_attention_heads=2,
        num_key_value_heads=2,
        max_position_embeddings=32768,
    )
    torch.manual_seed(0)
    model = Qwen2ForCausalLM(cfg)
    model.eval()
    return model


def _run_gen(tmp_path, monkeypatch, axes_counts, **cfg_kw):
    """Produce REAL gen-phase outputs (fake only at the generate boundary)."""
    bank = _mini_bank(axes_counts)
    monkeypatch.setattr(M, "generate_batch", _mk_fake_gen())
    cfg = _cfg(tmp_path, tuple(axes_counts), **cfg_kw)
    assert M.phase_gen(cfg, bank, object(), FakeTok()) == 0
    return bank, cfg


def test_phase_capture_e2e_real_pinned_body(tmp_path, monkeypatch, tiny_model):
    bank, cfg = _run_gen(tmp_path, monkeypatch, {"alpha": 3})
    assert M.phase_capture(cfg, bank, tiny_model, FakeTok()) == 0
    n_rows = 3 * cfg.draws
    va = torch.load(cfg.va_dir / "alpha.pt", weights_only=False)
    vc = torch.load(cfg.vc_dir / "alpha.pt", weights_only=False)
    # fp32 stores, both v_A pooling twins, all 32 layers (plan §4.4)
    assert va["va_span"].shape == (n_rows, 32, 16) and va["va_span"].dtype == torch.float32
    assert va["va_tail_incl"].shape == (n_rows, 32, 16)
    assert va["va_tail_incl"].dtype == torch.float32
    assert torch.isfinite(va["va_span"]).all() and torch.isfinite(va["va_tail_incl"]).all()
    assert len(va["boundaries"]) == n_rows and len(va["rows"]) == n_rows
    assert va["rows"][0]["n_completion_tokens"] == va["boundaries"][0]["n_completion_tokens"]
    assert va["empty_rows"] == []
    assert vc["vc"].shape == (3, 32, 16) and vc["vc"].dtype == torch.float32
    ctxs = M.group_contexts_by_cell(bank)["alpha"]
    assert vc["context_ids"] == [c["id"] for c in ctxs]
    # v_C == the context-end hidden state (hooked block L == hs[L+1]; checked
    # away from block 31's pre/post-final-norm caveat)
    tok = FakeTok()
    ids0 = M.IDS_FN(tok, ctxs[0])
    ids_t = torch.tensor([ids0], dtype=torch.long)
    with torch.no_grad():
        hs = tiny_model(
            input_ids=ids_t, attention_mask=torch.ones_like(ids_t), output_hidden_states=True
        ).hidden_states
    for lyr in (0, 16, 30):
        ref = hs[lyr + 1][0, len(ids0) - 1].float()
        assert torch.allclose(vc["vc"][0, lyr], ref, atol=1e-4), lyr
    # the hook probe ran as a GATE on the production path, before the wave
    probe = json.loads((cfg.manifest_dir / "hook_probe_report.json").read_text())
    assert probe["verdict"] == "pass" and probe["probe_layers"] == [16, 22, 30]
    assert probe["hidden_states_tuple_len"] == 33
    man = json.loads((cfg.manifest_dir / "capture_alpha.done.json").read_text())
    assert man["regime_fp"] == M._capture_cell_fp(cfg, "alpha")
    assert man["uploaded"] is False and man["upload"] == {"mode": "none"}
    assert man["max_tail_end"] <= man["capture_max_model_len_floor"]
    sent = json.loads((cfg.out_root / "battery_capture_done.json").read_text())
    assert sent["cells"]["alpha"]["n_rows"] == n_rows
    assert sent["hook_probe"]["verdict"] == "pass"

    # idempotent re-run: manifests + stores short-circuit (no recompute)
    def _boom(*a, **k):
        raise AssertionError("capture must not recompute on a complete re-run")

    monkeypatch.setattr(M, "_capture_answer_states_fp32", _boom)
    assert M.phase_capture(cfg, bank, tiny_model, FakeTok()) == 0


def test_capture_fp32_shim_preserves_overflow_range(monkeypatch):
    """Values ~7e4 overflow fp16 (max 65504): the raw pinned call goes inf
    (why the shim exists — fails pre-fix), the shimmed call stays finite fp32
    with boundaries from the pin's OWN state; the torch global is restored."""
    big = 7.0e4
    r = M._r()

    def fake_extract(model, input_ids, layers, *, attention_mask=None, **kw):
        b, t = input_ids.shape
        return {lyr: torch.full((b, t, 4), big) for lyr in layers}

    monkeypatch.setattr(
        r,
        "extract_layer_activations",
        mock.create_autospec(r.extract_layer_activations, side_effect=fake_extract),
    )
    pin_cfg = M._PinCaptureCfg(layers=[0, 1], hidden=4, capture_batch=2, device="cpu")
    tok = FakeTok()
    ctx_ids = [[5, 6], [5, 6, 8]]
    comps = ["hello world", "foo"]
    raw = r.capture_answer_states(
        pin_cfg, object(), tok, ctx_ids, comps, [7, 11], tail_inclusive=True
    )
    assert raw["va_span"].dtype == torch.float16
    assert not torch.isfinite(raw["va_span"]).all()  # fp16 store overflows
    out = M._capture_answer_states_fp32(pin_cfg, object(), tok, ctx_ids, comps, [7, 11])
    assert out["va_span"].dtype == torch.float32
    assert torch.isfinite(out["va_span"]).all()
    assert float(out["va_span"].abs().max()) == pytest.approx(big)
    assert out["boundaries"][0] == {
        "ctx_len": 2,
        "n_completion_tokens": 2,
        "span_start": 2,
        "span_end": 4,
        "tail_end": 6,
    }
    assert r.torch is torch  # the module global was restored after the call


def test_phase_capture_gate4_mismatch_halts_loud(tmp_path, monkeypatch, tiny_model):
    bank, cfg = _run_gen(tmp_path, monkeypatch, {"alpha": 2})
    p = cfg.anchors_dir / "anchors_alpha.jsonl"
    rows = M._read_jsonl(p)
    rows[0]["span_end"] += 1  # corrupt ONE gen-side span field
    M.write_jsonl_atomic(p, rows)
    with pytest.raises(RuntimeError, match=r"gate-4.*EXACT boundary compare FAILED"):
        M.phase_capture(cfg, bank, tiny_model, FakeTok())
    assert not (cfg.manifest_dir / "capture_alpha.done.json").is_file()


def test_phase_capture_floor_enforced(tmp_path, monkeypatch, tiny_model):
    bank, cfg = _run_gen(tmp_path, monkeypatch, {"alpha": 2})
    man_path = cfg.manifest_dir / "anchors_alpha.done.json"
    man = json.loads(man_path.read_text())
    man["capture_max_model_len_floor"] = 10**9  # beyond the tiny model's 32768
    M.write_json_atomic(man_path, man)
    with pytest.raises(RuntimeError, match="max_position_embeddings"):
        M.phase_capture(cfg, bank, tiny_model, FakeTok())


def test_phase_capture_requires_matching_gen_regime(tmp_path, monkeypatch, tiny_model):
    bank, cfg = _run_gen(tmp_path, monkeypatch, {"alpha": 2})
    cfg2 = _cfg(tmp_path, ("alpha",), draws=cfg.draws + 1)  # different gen regime
    with pytest.raises(RuntimeError, match="regime_fp mismatch"):
        M.phase_capture(cfg2, bank, tiny_model, FakeTok())


def test_hook_probe_gate_halts_on_mismatch(tmp_path, monkeypatch, tiny_model):
    """A production-path divergence (fake hooks returning zeros) FAILS the
    probe, persists the report, and halts BEFORE any capture manifest."""
    bank, cfg = _run_gen(tmp_path, monkeypatch, {"alpha": 2})

    def zeros_extract(model, input_ids, layers, *, attention_mask=None, **kw):
        b, t = input_ids.shape
        return {lyr: torch.zeros((b, t, 16)) for lyr in layers}

    monkeypatch.setattr(
        M,
        "extract_layer_activations",
        mock.create_autospec(M.extract_layer_activations, side_effect=zeros_extract),
    )
    with pytest.raises(RuntimeError, match="hook probe FAILED"):
        M.phase_capture(cfg, bank, tiny_model, FakeTok())
    report = json.loads((cfg.manifest_dir / "hook_probe_report.json").read_text())
    assert report["verdict"] == "fail"
    assert not list(cfg.manifest_dir.glob("capture_*.done.json"))


# ── unit 3b: embed (fakes ONLY at the engine / tokenizer / HfApi boundary) ─


def _det_vec(text: str) -> np.ndarray:
    """Deterministic per-text embedding (process-stable: sha256-seeded)."""
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big")
    rng = np.random.default_rng(seed)
    return rng.normal(size=M.EMBED_DIM).astype(np.float32)


class _FakeEmbedTok:
    def encode(self, text, add_special_tokens=True):
        return [1] * (len(text.split()) + 1)


def _fake_embed_llm():
    """Signature-shaped fake of the vLLM pooling engine (engine boundary)."""

    class _Out:
        def __init__(self, vec):
            self.outputs = type("O", (), {"embedding": vec.tolist()})()

    class _LLM:
        def embed(self, texts, use_tqdm=False):
            return [_Out(_det_vec(t)) for t in texts]

    return _LLM()


def _patch_embed_boundaries(monkeypatch, engine_version=None):
    monkeypatch.setattr(
        M, "_realized_vllm_version", lambda: engine_version or M.EXPECTED_EMBED_ENGINE
    )
    monkeypatch.setattr(M, "_resolve_embed_revision", lambda: "embedrev123")
    monkeypatch.setattr(M, "_load_embed_tokenizer", lambda revision: _FakeEmbedTok())
    monkeypatch.setattr(
        M,
        "_make_embed_llm",
        mock.create_autospec(M._make_embed_llm, side_effect=lambda rev, mml: _fake_embed_llm()),
    )


def test_phase_embed_e2e_upload_none(tmp_path, monkeypatch):
    bank, cfg = _run_gen(tmp_path, monkeypatch, {"alpha": 2, "beta": 2})
    _patch_embed_boundaries(monkeypatch)
    ecfg = _cfg(tmp_path, ("alpha", "beta"), phase="embed", embed_chunk=3)
    assert M.phase_embed(ecfg, bank, None, None) == 0
    n_rows = 4 * cfg.draws
    emb_dir = ecfg.embed_root / "embeddings_qwen3_8b"
    z = np.load(emb_dir / "perdraw_anchors.npz", allow_pickle=False)
    assert z["emb"].shape == (n_rows, M.EMBED_DIM) and z["emb"].dtype == np.float16
    norms = np.linalg.norm(z["emb"].astype(np.float64), axis=1)
    assert np.allclose(norms, 1.0, atol=2e-2)  # L2-normalized (fp16 store)
    assert set(z.files) >= {"emb", "context_ids", "draws", "cells", "think_leak", "vllm_version"}
    assert str(z["vllm_version"]) == M.EXPECTED_EMBED_ENGINE  # provenance in the ARTIFACT
    zm = np.load(emb_dir / "means_anchors.npz", allow_pickle=False)
    assert zm["emb_mean"].shape == (4, M.EMBED_DIM)
    assert (zm["n_draws"] == cfg.draws).all()
    assert str(zm["vllm_version"]) == M.EXPECTED_EMBED_ENGINE
    meta = json.loads((emb_dir / "meta.json").read_text())
    assert meta["engine"]["parity_mode"] == "repo-pin"
    assert meta["engine"]["vllm_version"] == M.EXPECTED_EMBED_ENGINE
    assert meta["n_rows"] == n_rows and meta["n_skipped_empty"] == 0
    # chunk checkpoints carry the engine version too
    cks = sorted((ecfg.embed_root / "chunks").glob("chunk_*.npz"))
    assert len(cks) == (n_rows + 2) // 3
    ck0 = np.load(cks[0], allow_pickle=False)
    assert str(ck0["vllm_version"]) == M.EXPECTED_EMBED_ENGINE

    # sentinel idempotency: the re-run short-circuits BEFORE the tokenizer
    def _boom(revision):
        raise AssertionError("must not re-tokenize on a sentinel skip")

    monkeypatch.setattr(M, "_load_embed_tokenizer", _boom)
    assert M.phase_embed(ecfg, bank, None, None) == 0


def _parity_report_dict(**overrides) -> dict:
    """A PASSING probe report at the plan admission bars (r1 g4 C1)."""
    rep = {
        "parity_pass": True,
        "reference_engine": M.EXPECTED_EMBED_ENGINE,
        "engine": "0.27.1",
        "n_anchors": M.PARITY_N_ANCHORS_MIN,
        "cos_min_bar": M.PARITY_COS_MIN,
    }
    rep.update(overrides)
    return rep


def test_phase_embed_engine_gate(tmp_path, monkeypatch):
    """Unprobed engine mismatch REFUSES; a non-passing report REFUSES; a
    PASSING report admits the engine and records parity_mode=parity-probe."""
    bank, _ = _run_gen(tmp_path, monkeypatch, {"alpha": 2})
    _patch_embed_boundaries(monkeypatch, engine_version="0.27.1")
    ecfg = _cfg(tmp_path, ("alpha",), phase="embed")
    with pytest.raises(RuntimeError, match="parity reference"):
        M.phase_embed(ecfg, bank, None, None)
    rep = tmp_path / "parity.json"
    rep.write_text(json.dumps(_parity_report_dict(parity_pass=False)))
    ecfg2 = _cfg(tmp_path, ("alpha",), phase="embed", parity_report=rep)
    with pytest.raises(RuntimeError, match="parity_pass"):
        M.phase_embed(ecfg2, bank, None, None)
    rep.write_text(json.dumps(_parity_report_dict()))
    ecfg3 = _cfg(tmp_path, ("alpha",), phase="embed", parity_report=rep)
    assert M.phase_embed(ecfg3, bank, None, None) == 0
    meta = json.loads((ecfg3.embed_root / "embeddings_qwen3_8b" / "meta.json").read_text())
    assert meta["engine"]["parity_mode"] == "parity-probe"
    assert meta["engine"]["vllm_version"] == "0.27.1"


def test_engine_parity_report_admission_floors(tmp_path):
    """r1 g4 C1: a parity_pass=true report produced by a WEAKENED probe run
    (lower --parity-cos-min, fewer --parity-n-anchors, or the fields absent)
    is REFUSED — the WARN-1 instrument-identity gate enforces its own
    admission criteria on the consumed report."""
    rep = tmp_path / "parity.json"
    # weakened cosine bar — parity_pass true, but below the plan bar
    rep.write_text(json.dumps(_parity_report_dict(cos_min_bar=0.5)))
    with pytest.raises(RuntimeError, match="cos_min_bar"):
        M._assert_engine_parity("0.27.1", rep)
    # weakened anchor count
    rep.write_text(json.dumps(_parity_report_dict(n_anchors=3)))
    with pytest.raises(RuntimeError, match="n_anchors"):
        M._assert_engine_parity("0.27.1", rep)
    # fields absent (a legacy / hand-rolled report) — refused, never defaulted
    legacy = _parity_report_dict()
    del legacy["n_anchors"], legacy["cos_min_bar"]
    rep.write_text(json.dumps(legacy))
    with pytest.raises(RuntimeError, match="n_anchors"):
        M._assert_engine_parity("0.27.1", rep)
    # bool True must never alias the int floor (True < 10 -> refused as non-int)
    rep.write_text(json.dumps(_parity_report_dict(n_anchors=True)))
    with pytest.raises(RuntimeError, match="n_anchors"):
        M._assert_engine_parity("0.27.1", rep)
    # at the bars (or stricter cosine) -> admitted
    rep.write_text(json.dumps(_parity_report_dict()))
    out = M._assert_engine_parity("0.27.1", rep)
    assert out["parity_mode"] == "parity-probe"
    rep.write_text(json.dumps(_parity_report_dict(cos_min_bar=0.999, n_anchors=12)))
    assert M._assert_engine_parity("0.27.1", rep)["parity_mode"] == "parity-probe"
    # the probe's own report satisfies the admission floors by construction
    probe_defaults = {"n_anchors": M.PARITY_N_ANCHORS_MIN, "cos_min_bar": M.PARITY_COS_MIN}
    assert probe_defaults["n_anchors"] >= M.PARITY_N_ANCHORS_MIN
    assert probe_defaults["cos_min_bar"] >= M.PARITY_COS_MIN


def test_embed_chunk_resume_and_engine_keyed_fp(tmp_path, monkeypatch):
    bank, _ = _run_gen(tmp_path, monkeypatch, {"alpha": 2})
    real_make_llm = M._make_embed_llm  # captured BEFORE the boundary patch
    _patch_embed_boundaries(monkeypatch)
    ecfg = _cfg(tmp_path, ("alpha",), phase="embed", embed_chunk=2)
    assert M.phase_embed(ecfg, bank, None, None) == 0
    # full chunk resume: sentinel removed, engine must NOT load again
    (ecfg.out_root / "battery_embed_done.json").unlink()
    monkeypatch.setattr(
        M,
        "_make_embed_llm",
        mock.create_autospec(
            real_make_llm, side_effect=AssertionError("engine must not load on full resume")
        ),
    )
    assert M.phase_embed(ecfg, bank, None, None) == 0
    # the fp keys on the ENGINE version: chunks embedded under 0.11.0 can
    # never satisfy a 0.27.1 run's resume — it must reach the engine ctor.
    rep = tmp_path / "parity.json"
    rep.write_text(json.dumps(_parity_report_dict()))
    monkeypatch.setattr(M, "_realized_vllm_version", lambda: "0.27.1")
    ecfg2 = _cfg(tmp_path, ("alpha",), phase="embed", embed_chunk=2, parity_report=rep)
    with pytest.raises(AssertionError, match="engine must not load"):
        M.phase_embed(ecfg2, bank, None, None)


def test_embed_token_precheck_raises_never_truncates(tmp_path, monkeypatch):
    bank, _ = _run_gen(tmp_path, monkeypatch, {"alpha": 2})
    _patch_embed_boundaries(monkeypatch)
    ecfg = _cfg(tmp_path, ("alpha",), phase="embed", embed_max_model_len=3)
    with pytest.raises(RuntimeError, match="never truncated"):
        M.phase_embed(ecfg, bank, None, None)


def test_phase_embed_pilot_refuse_rc(tmp_path, monkeypatch):
    bank, _ = _run_gen(tmp_path, monkeypatch, {"alpha": 2})
    _patch_embed_boundaries(monkeypatch)
    ecfg = _cfg(tmp_path, ("alpha",), phase="embed", embed_chunk=2, embed_pilot_ceiling_h=-1.0)
    assert M.phase_embed(ecfg, bank, None, None) == M.EXIT_PILOT_REFUSE
    report = json.loads((ecfg.manifest_dir / "embed_pilot_gate_report.json").read_text())
    assert report["verdict"] == "refuse"
    assert not (ecfg.out_root / "battery_embed_done.json").is_file()


def test_embed_requires_complete_anchor_grid(tmp_path, monkeypatch):
    bank2 = _mini_bank({"alpha": 2, "beta": 2})
    monkeypatch.setattr(M, "generate_batch", _mk_fake_gen())
    cfg = _cfg(tmp_path, ("alpha",))
    assert M.phase_gen(cfg, {**bank2, "n_contexts": 4}, object(), FakeTok()) == 0
    _patch_embed_boundaries(monkeypatch)
    ecfg = _cfg(tmp_path, ("alpha", "beta"), phase="embed")
    with pytest.raises(RuntimeError, match="0 anchor roots hold"):
        M.phase_embed(ecfg, bank2, None, None)


def test_run_engine_parity_probe_pass_and_miss(tmp_path, monkeypatch):
    monkeypatch.setattr(M, "_realized_vllm_version", lambda: "0.27.1")
    monkeypatch.setattr(M, "_resolve_embed_revision", lambda: "embedrev123")
    monkeypatch.setattr(
        M,
        "_make_embed_llm",
        mock.create_autospec(M._make_embed_llm, side_effect=lambda rev, mml: _fake_embed_llm()),
    )
    rows = [
        {"context_id": f"c{i:02d}", "draw": 0, "cell": "alpha", "text": f"text {i} words"}
        for i in range(12)
    ]
    anchors = tmp_path / "anchors.jsonl"
    anchors.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")

    def _banked(vec_texts):
        emb = np.stack([_det_vec(t) for t in vec_texts]).astype(np.float64)
        emb = emb / np.linalg.norm(emb, axis=1)[:, None]
        return emb.astype(np.float16)

    # banked == what the current engine produces -> PASS
    np.savez(
        tmp_path / "banked.npz",
        emb=_banked([r["text"] for r in rows]),
        context_ids=np.array([r["context_id"] for r in rows]),
        draws=np.array([0] * 12),
    )
    report = M.run_engine_parity_probe(
        anchors, tmp_path / "banked.npz", tmp_path / "rep.json", n_anchors=10
    )
    assert report["parity_pass"] is True and report["min_cos"] > 0.995
    assert report["engine"] == "0.27.1"
    assert report["reference_engine"] == M.EXPECTED_EMBED_ENGINE
    assert json.loads((tmp_path / "rep.json").read_text())["parity_pass"] is True
    # banked vectors from DIFFERENT texts -> MISS (report written, pass=false)
    np.savez(
        tmp_path / "banked_bad.npz",
        emb=_banked([r["text"] + " shifted" for r in rows]),
        context_ids=np.array([r["context_id"] for r in rows]),
        draws=np.array([0] * 12),
    )
    report2 = M.run_engine_parity_probe(
        anchors, tmp_path / "banked_bad.npz", tmp_path / "rep2.json", n_anchors=10
    )
    assert report2["parity_pass"] is False
    assert report2["max_cos_deviation"] > 1.0 - M.PARITY_COS_MIN
