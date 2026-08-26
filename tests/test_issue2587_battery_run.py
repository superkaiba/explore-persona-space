"""CPU-only pins for scripts/issue2587_battery_run.py (unit 3a — P5 gen phase).

No network, no HF fetch, no GPU (the units 1-2 pattern): the tokenizer is a
deterministic chat-template-shaped fake; ``generate_batch`` is faked ONLY at
the GPU boundary via ``unittest.mock.create_autospec`` (signature-conformant
by construction, per the one-production-body-test rule) — every body under
test (``_generate_cell`` / ``_gen_cell`` / ``_pilot_gate`` / ``phase_gen``)
executes for real. The pinned issue2162_run import (``M._r()``) is a LOCAL
``git show`` of an object already in the shared odb — no network.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import pytest

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


def test_unit3b_seam_phases_raise_not_implemented(tmp_path):
    cfg = _cfg(tmp_path, ("alpha",))
    for fn in (M.phase_capture, M.phase_embed):
        with pytest.raises(NotImplementedError, match="unit 3b"):
            fn(cfg, _mini_bank({"alpha": 1}), object(), FakeTok())
    assert M.PHASE_FNS["capture"] is M.phase_capture
    assert M.PHASE_FNS["embed"] is M.phase_embed


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
        {"upload": "hf"},
        {"max_carriers": 2},
        {"gen_batch": 4},
        {"model_revision": "otherrev"},
        {"bank_values_sha": "other"},
    ):
        assert M._regime_fp(_cfg(tmp_path, ("alpha",), **kw)) != fp, kw
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
    assert sent["regime_fp"] == M._regime_fp(cfg, {"phase": "gen", "axes": sorted(cfg.axes)})
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
