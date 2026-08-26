"""Issue #2564 driver — CPU pins (unit 2 of the pre-split build).

Three legs per plan §3.7 / the unit-2 brief:

(a) kwarg-signature pins on the REAL imported callees the driver (and the
    later analysis unit) reuse: ``capture_answer_states`` must accept
    ``return_boundaries`` (the 15501f33b2 MF-A hunk carried onto main's
    ``scripts/issue2162_run.py``), ``apply_map`` / ``knn_retrieval`` must
    match the plan-§10 call shapes, and the driver's own
    ``_assert_call_kwargs`` start-of-run assertion passes.
(b) tiny-payload ``apply_map`` call-shape smoke (minimal ridge dict).
(c) ``return_boundaries`` behavior pin through the REAL
    ``capture_answer_states`` body (fake ONLY the model-forward boundary,
    signature-conformant — the ``tests/test_issue2215_run.py`` pattern):
    default-off emits no ``boundaries`` key and leaves outputs unchanged;
    on, the records match the function's OWN tokenization state, empty
    completions included.

Plus cheap driver-config pins: the plan-§9 workload argv parses, the smoke
slice/rebind (plan §8 — bank realizes the query cell as ``"query"``), and
the fail-loud empty-selection filter.
"""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue779_ffc_n1m_fits as N1M  # noqa: E402
import issue2162_run as R2162  # noqa: E402
import issue2564_run as D  # noqa: E402

from explore_persona_space.analysis.mapping_baselines import knn_retrieval  # noqa: E402

# ── fakes (signature-conformant by construction; test_issue2215_run pattern) ──


class FakeTokenizer:
    """Mirrors the two surfaces ``capture_answer_states`` touches: the
    ``__call__(text, add_special_tokens=False) -> {"input_ids": [...]}``
    encode and ``pad_token_id``. One token per whitespace word."""

    pad_token_id = 0

    def __call__(self, text: str, add_special_tokens: bool = False) -> dict:
        assert add_special_tokens is False
        return {"input_ids": [10 + k for k, _ in enumerate(text.split())]}


def fake_extract_layer_activations(model, ids, layers, attention_mask=None):
    """Signature mirror of ``analysis.extraction.extract_layer_activations``
    (the external model-forward boundary). Activation at (row, position) is
    the POSITION index broadcast over hidden."""
    b, t = ids.shape
    pos = torch.arange(t, dtype=torch.float32)[None, :, None].expand(b, t, 4)
    return {layer: pos.clone() for layer in layers}


def capture_cfg(batch: int = 2) -> SimpleNamespace:
    return SimpleNamespace(
        layers=[0, 1], hidden=4, capture_batch=batch, device="cpu", model_id="fake", tiny=True
    )


# ── (a) kwarg-signature pins on the real imported callees ─────────────


def test_capture_answer_states_signature_has_boundary_kwargs():
    params = set(inspect.signature(R2162.capture_answer_states).parameters)
    assert {"payloads", "positions", "tail_inclusive", "return_boundaries"} <= params, params


def test_apply_map_signature_matches_plan_call_shape():
    # plan §10 bind row: apply_map(payload, X_eval, dev)
    params = list(inspect.signature(N1M.apply_map).parameters)
    assert params[:3] == ["payload", "X_eval", "dev"], params


def test_knn_retrieval_signature_matches_plan_call_shape():
    params = set(inspect.signature(knn_retrieval).parameters)
    assert {"pred", "true", "ks", "metric", "pool", "true_pool_idx"} <= params, params


def test_driver_start_kwarg_assertion_passes():
    D._assert_call_kwargs()


# ── (b) apply_map tiny-payload call-shape smoke ────────────────────────


def test_apply_map_ridge_tiny_payload_shape_and_values():
    rng = np.random.default_rng(0)
    X_eval = rng.standard_normal((5, 3))
    W = torch.tensor(rng.standard_normal((3, 2)), dtype=torch.float32)
    payload = {
        "kind": "ridge",
        "xmu": torch.zeros(3),
        "xsd": torch.ones(3),
        "ymu": torch.zeros(2),
        "W": W,
    }
    out = N1M.apply_map(payload, X_eval, torch.device("cpu"))
    assert isinstance(out, np.ndarray) and out.shape == (5, 2), out.shape
    assert out.dtype == np.float64
    # xmu=0 / xsd=1 / ymu=0 collapses the standardize-X path to X @ W exactly
    # (fp32 W upcast to fp64 inside apply_map — compare against the same upcast).
    np.testing.assert_allclose(out, X_eval @ W.double().numpy(), rtol=1e-9, atol=1e-9)


def test_apply_map_unknown_kind_raises_value_error():
    with pytest.raises(ValueError, match="unknown persisted map kind"):
        N1M.apply_map({"kind": "nope"}, np.zeros((1, 2)), torch.device("cpu"))


# ── (c) return_boundaries behavior pin (real body, fake forward) ──────


def test_return_boundaries_default_off_and_outputs_unchanged(monkeypatch):
    monkeypatch.setattr(R2162, "extract_layer_activations", fake_extract_layer_activations)
    cfg = capture_cfg()
    tok = FakeTokenizer()
    ctx_ids = [[1, 2, 3], [1, 2]]
    completions = ["a b", "x y z"]
    eot = [7, 8]
    default_out = R2162.capture_answer_states(
        cfg, object(), tok, ctx_ids, completions, eot, tail_inclusive=True
    )
    on_out = R2162.capture_answer_states(
        cfg, object(), tok, ctx_ids, completions, eot, tail_inclusive=True, return_boundaries=True
    )
    assert "boundaries" not in default_out
    assert "boundaries" in on_out
    # additive + default-off: the tensor outputs are byte-identical either way
    assert torch.equal(default_out["va_span"], on_out["va_span"])
    assert torch.equal(default_out["va_tail_incl"], on_out["va_tail_incl"])
    assert default_out["n_completion_tokens"] == on_out["n_completion_tokens"]


def test_return_boundaries_records_match_own_tokenization(monkeypatch):
    monkeypatch.setattr(R2162, "extract_layer_activations", fake_extract_layer_activations)
    cfg = capture_cfg()
    tok = FakeTokenizer()
    # row 0: ctx_len=3, completion "a b" -> 2 tokens; row 1: ctx_len=2, EMPTY
    # completion -> record still emitted with n_completion_tokens == 0.
    out = R2162.capture_answer_states(
        cfg,
        object(),
        tok,
        [[1, 2, 3], [1, 2]],
        ["a b", ""],
        [7, 8],
        tail_inclusive=True,
        return_boundaries=True,
    )
    assert out["boundaries"] == [
        {"ctx_len": 3, "n_completion_tokens": 2, "span_start": 3, "span_end": 5, "tail_end": 7},
        {"ctx_len": 2, "n_completion_tokens": 0, "span_start": 2, "span_end": 2, "tail_end": 4},
    ]
    assert out["empty_rows"] == [1]
    assert out["n_completion_tokens"] == [2, 0]


# ── driver-config pins (cheap, CPU, no bank/model) ─────────────────────


def test_plan_workload_argv_parses():
    # plan §9 dispatch workload cmd must parse verbatim
    args = D.parse_args(["--phase", "all", "--out-root", "/workspace/eps2564", "--upload", "hf"])
    assert args.phase == "all"
    assert args.upload == "hf"


def test_build_config_smoke_rebinds_out_root_and_slices():
    args = D.parse_args(
        ["--phase", "all", "--out-root", "/tmp/eps2564x", "--smoke", "--upload", "none"]
    )
    cfg = D.build_config(args)
    assert cfg.out_root.name == "smoke_eps2564x"  # generated artifacts rebind (plan §8)
    assert cfg.cells == D.SMOKE_CELLS == ("register", "query")
    assert cfg.carriers == ("c01", "c02", "c03")
    assert cfg.draws == 2
    assert cfg.hf_prefix == "issue2564_minpair/smoke"


def test_filter_bank_empty_selection_raises():
    # producer-shaped fixture (dict contexts — r2 blocker 1): the empty-selection
    # raise must fire AFTER the dict normalization, not TypeError before it.
    bank = {
        "contexts": {"a": {"id": "a", "cell": "register", "carrier": "c01"}},
        "pairs": [],
        "n_contexts": 1,
        "n_pairs": 0,
    }
    with pytest.raises(RuntimeError, match="empty context selection"):
        D._filter_bank(bank, ("no_such_cell",), None)


def test_filter_bank_consumes_real_producer_bank_shape():
    """r2 blocker 1 integration pin: ``BK.build_contexts`` returns a DICT
    (id -> ctx) — the r1 driver iterated it as a list and crashed on every
    run. ``_filter_bank`` must consume the REAL producer shape and normalize
    to the list every phase iterates."""
    from explore_persona_space.experiments.issue2564 import bank2564 as BK

    values = BK.load_values()
    contexts = BK.build_contexts(values)
    assert isinstance(contexts, dict)  # the producer shape that crashed r1
    bank = {
        "contexts": contexts,
        "pairs": BK.build_pairs(values, contexts),
        "n_contexts": len(contexts),
        "n_pairs": 0,
    }
    out = D._filter_bank(bank, D.SMOKE_CELLS, ("c01", "c02", "c03"))
    assert isinstance(out["contexts"], list)
    assert out["n_contexts"] > 0 and out["n_pairs"] > 0
    assert {c["cell"] for c in out["contexts"]} <= set(D.SMOKE_CELLS)
    assert {c["carrier"] for c in out["contexts"]} == {"c01", "c02", "c03"}
    kept = {c["id"] for c in out["contexts"]}
    assert all(p["a"] in kept and p["b"] in kept for p in out["pairs"])


# ── PC (embed) phase-registry + dispatch pins (r2 blocker 5) ────────────


def test_phase_registry_dispatches_pc():
    """Plan §9 puts PC (pc_embed) on-pod: ``--phase all`` must run A, B AND C."""
    assert D.PHASES == ("A", "B", "C")
    args = D.parse_args(["--phase", "C", "--out-root", "/tmp/eps2564c", "--upload", "none"])
    assert args.phase == "C"


def test_phase_embed_tiny_semantics():
    """Under --tiny PC is skipped with a warning on --phase all, but an
    EXPLICIT --phase C under --tiny raises (never a silent no-op success)."""
    args = D.parse_args(
        ["--phase", "all", "--out-root", "/tmp/eps2564t", "--tiny", "--upload", "none"]
    )
    cfg = D.build_config(args)
    assert D.phase_embed(cfg, {"contexts": [], "pairs": []}) == D.RC_OK
    args_c = D.parse_args(
        ["--phase", "C", "--out-root", "/tmp/eps2564t", "--tiny", "--upload", "none"]
    )
    cfg_c = D.build_config(args_c)
    with pytest.raises(RuntimeError, match="--tiny"):
        D.phase_embed(cfg_c, {"contexts": [], "pairs": []})


def test_phase_embed_argv_env_and_sentinel_verification(monkeypatch, tmp_path):
    """phase_embed composes the embed subprocess argv (out-root = PRE-rebind
    CLI root + /embed; anchors-root = the REBOUND root PA wrote), passes the
    env through, propagates rc=7, and VERIFIES the sentinel before RC_OK."""
    args = D.parse_args(
        ["--phase", "C", "--out-root", str(tmp_path / "root"), "--smoke", "--upload", "none"]
    )
    cfg = D.build_config(args)
    bank = {"contexts": [{"id": "x", "cell": "register", "carrier": "c01"}], "pairs": []}
    monkeypatch.setattr(D, "_anchor_cell_complete", lambda cfg, cell: True)
    captured: dict = {}

    def fake_run(cmd, env):  # signature mirror of the subprocess.run call site
        captured["cmd"] = [str(c) for c in cmd]
        captured["env_has_path"] = "PATH" in env
        out = D._embed_out_root(cfg)
        out.mkdir(parents=True, exist_ok=True)
        (out / "embed_done.local.json").write_text("{}")
        return SimpleNamespace(returncode=captured.get("rc", 0))

    monkeypatch.setattr(D.subprocess, "run", fake_run)
    assert D.phase_embed(cfg, bank) == D.RC_OK
    cmd = captured["cmd"]
    assert cmd[1].endswith("issue2564_embed.py")
    assert "--smoke" in cmd and "--skip-upload" in cmd
    assert cmd[cmd.index("--anchors-root") + 1] == str(cfg.out_root)
    assert cmd[cmd.index("--out-root") + 1] == str((cfg.raw_out_root or cfg.out_root) / "embed")
    assert captured["env_has_path"]
    # rc=7 (pilot-gate refusal) propagates verbatim
    captured["rc"] = D.RC_PILOT_GATE
    assert D.phase_embed(cfg, bank) == D.RC_PILOT_GATE


def test_phase_embed_missing_sentinel_raises(monkeypatch, tmp_path):
    """rc=0 with NO sentinel on disk is a FAIL (out-root derivation drift),
    never terminal success (r2 blocker 5)."""
    args = D.parse_args(
        ["--phase", "C", "--out-root", str(tmp_path / "root2"), "--smoke", "--upload", "none"]
    )
    cfg = D.build_config(args)
    bank = {"contexts": [{"id": "x", "cell": "register", "carrier": "c01"}], "pairs": []}
    monkeypatch.setattr(D, "_anchor_cell_complete", lambda cfg, cell: True)
    monkeypatch.setattr(D.subprocess, "run", lambda cmd, env: SimpleNamespace(returncode=0))
    with pytest.raises(RuntimeError, match="sentinel is missing"):
        D.phase_embed(cfg, bank)


# ── PA .partial resume: regime-fp header + per-chunk context-id set (r2 blocker 2) ──


def _gen_cfg(tmp_path):
    """Tiny CPU cfg: gen_batch=2 over 3 contexts -> 2 chunks; draws=2 (SMOKE_DRAWS)."""
    args = D.parse_args(
        [
            "--phase",
            "A",
            "--out-root",
            str(tmp_path / "r"),
            "--tiny",
            "--upload",
            "none",
            "--gen-batch",
            "2",
        ]
    )
    return D.build_config(args)


def _fake_ctxs(n: int = 3) -> list[dict]:
    return [
        {
            "id": f"register::v1::c{i:02d}",
            "cell": "register",
            "kind": "value",
            "value_id": "v1",
            "carrier": f"c{i:02d}",
            "form": "stmt",
        }
        for i in range(1, n + 1)
    ]


def _fake_generate_batch(calls: list):
    """Signature mirror of ``steering.generate_batch`` (the GPU boundary)."""

    def fake(
        model, tok, chunk, *, n, hook, max_new_tokens, temperature, seed_base, render_fn, ids_fn
    ):
        calls.append([c["id"] for c in chunk])
        return [[f"text {c['id']} draw {i}" for i in range(n)] for c in chunk]

    return fake


def test_generate_cell_regime_fp_resume_adopts_complete_partial(tmp_path, monkeypatch):
    """A .partial with a MATCHING regime-fp header + complete per-chunk
    context-id sets is fully adopted: the resumed call reaches NO engine
    call and returns identical rows (r2 blocker 2 — resume keying)."""
    cfg = _gen_cfg(tmp_path)
    monkeypatch.setattr(D.BK, "context_token_ids", lambda tok, ctx: [1, 2, 3])
    calls: list = []
    monkeypatch.setattr(D, "generate_batch", _fake_generate_batch(calls))
    tok = FakeTokenizer()
    ctxs = _fake_ctxs(3)
    pilot = {"evaluated": True}
    rows1 = D._generate_cell(cfg, object(), tok, [7], "register", ctxs, pilot, 128)
    assert len(rows1) == 3 * cfg.draws
    assert len(calls) == 2  # ceil(3 / gen_batch=2) chunks generated fresh
    part = cfg.anchors_dir / "anchors_register.max128.partial"
    assert part.is_file()  # final jsonl + partial unlink happen in _anchor_cell, not here
    header = json.loads(part.read_text().split("\n")[0])
    assert header == {
        "partial_header": 1,
        "regime_fp": D._regime_fp(cfg, {"phase": "A", "cell": "register", "max_new_call": 128}),
    }

    def boom(*a, **k):  # resumed call must never reach the engine
        raise AssertionError("generate_batch called on a fully-complete resume")

    monkeypatch.setattr(D, "generate_batch", boom)
    rows2 = D._generate_cell(cfg, object(), tok, [7], "register", ctxs, pilot, 128)
    assert rows2 == rows1


def test_generate_cell_quarantines_mismatched_regime_fp_partial(tmp_path, monkeypatch):
    """A .partial whose header carries a FOREIGN regime fp is quarantined
    (never adopted) and the whole cell regenerates (r2 blocker 2)."""
    cfg = _gen_cfg(tmp_path)
    monkeypatch.setattr(D.BK, "context_token_ids", lambda tok, ctx: [1, 2, 3])
    calls: list = []
    monkeypatch.setattr(D, "generate_batch", _fake_generate_batch(calls))
    cfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    part = cfg.anchors_dir / "anchors_register.max128.partial"
    part.write_text(
        json.dumps({"partial_header": 1, "regime_fp": "deadbeefdeadbeef"})
        + "\n"
        + json.dumps({"chunk": 0, "context_id": "register::v1::c01", "draw": 0})
        + "\n"
    )
    rows = D._generate_cell(
        cfg, object(), FakeTokenizer(), [7], "register", _fake_ctxs(3), {"evaluated": True}, 128
    )
    assert len(calls) == 2  # stale rows never adopted — both chunks regenerated
    assert len(rows) == 3 * cfg.draws
    quarantined = list(cfg.quarantine_dir.iterdir())
    assert len(quarantined) == 1
    assert quarantined[0].name.endswith("anchors_register.max128.partial")


def test_generate_cell_rejects_chunk_with_wrong_context_id_set(tmp_path, monkeypatch):
    """A chunk whose prior rows match in COUNT but not in context-id SET is
    not adopted — a re-scoped run can never adopt wrong-context rows
    (r2 blocker 2 — per-chunk context-id composition)."""
    cfg = _gen_cfg(tmp_path)
    monkeypatch.setattr(D.BK, "context_token_ids", lambda tok, ctx: [1, 2, 3])
    calls: list = []
    monkeypatch.setattr(D, "generate_batch", _fake_generate_batch(calls))
    cfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    part = cfg.anchors_dir / "anchors_register.max128.partial"
    fp = D._regime_fp(cfg, {"phase": "A", "cell": "register", "max_new_call": 128})
    wrong_rows = [
        {"chunk": 0, "context_id": cid, "draw": d}
        for cid in ("register::v1::c98", "register::v1::c99")
        for d in range(cfg.draws)
    ]
    part.write_text(
        "\n".join(
            [json.dumps({"partial_header": 1, "regime_fp": fp})]
            + [json.dumps(r) for r in wrong_rows]
        )
        + "\n"
    )
    rows = D._generate_cell(
        cfg, object(), FakeTokenizer(), [7], "register", _fake_ctxs(3), {"evaluated": True}, 128
    )
    assert len(calls) == 2  # count matched (2 rows x draws) but id-set did not -> regenerate
    assert len(rows) == 3 * cfg.draws
    assert {r["context_id"] for r in rows} == {c["id"] for c in _fake_ctxs(3)}


# ── r3 concerns: upload resume_skip pins + loader revision pins + PC idempotency ──

import ast  # noqa: E402

import issue2564_embed as E  # noqa: E402

_SCRIPTS = REPO_ROOT / "scripts"


def _fake_upload_result(repo: str):
    return SimpleNamespace(repo_id=repo, uploaded=[], rerouted=[], skipped_existing=[])


def _upload_spy(calls: list):
    """Signature mirror of the upload_dir_sharded call sites in issue2564_run."""

    def fake(local, repo, prefix, shard_glob, resume_skip, delete_local):
        calls.append({"prefix": prefix, "resume_skip": resume_skip, "glob": shard_glob})
        return _fake_upload_result(repo)

    return fake


def test_phase_anchors_upload_passes_resume_skip_false(monkeypatch, tmp_path):
    """r3 missing-fix-regression-tests (a): the PA anchors upload is
    force-reupload (resume_skip=False) — anchors jsonls are mutable across
    cap-hit re-gen; the r1 presence-skip would retain stale HF rows."""
    args = D.parse_args(["--phase", "A", "--out-root", str(tmp_path / "pa"), "--upload", "hf"])
    cfg = D.build_config(args)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(D, "_anchor_cell_complete", lambda cfg, cell: True)
    calls: list = []
    monkeypatch.setattr(D, "upload_dir_sharded", _upload_spy(calls))
    bank = {"contexts": [{"id": "x", "cell": "register", "carrier": "c01"}], "pairs": []}
    assert D.phase_anchors(cfg, bank) == D.RC_OK
    assert len(calls) == 1
    assert calls[0]["prefix"].endswith("raw_completions/anchors")
    assert calls[0]["resume_skip"] is False


def test_phase_capture_uploads_pass_resume_skip_false(monkeypatch, tmp_path):
    """r3 missing-fix-regression-tests (a): the PB va/vc (+manifests) .pt/.json
    uploads are force-reupload (resume_skip=False, r2 blocker 3) — a recomputed
    same-shape store is size-identical but content-different, so the size-match
    presence probe would silently retain stale HF tensors (#2552 class)."""
    args = D.parse_args(["--phase", "B", "--out-root", str(tmp_path / "pb"), "--upload", "hf"])
    cfg = D.build_config(args)
    cfg.out_root.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(D, "_require_anchor_shards", lambda cfg, cells: None)
    monkeypatch.setattr(D, "_va_cell_complete", lambda cfg, cell: True)
    monkeypatch.setattr(D, "_vc_complete", lambda cfg: True)
    monkeypatch.setattr(D, "_parity_report_ok", lambda cfg: True)
    calls: list = []
    monkeypatch.setattr(D, "upload_dir_sharded", _upload_spy(calls))
    bank = {"contexts": [{"id": "x", "cell": "register", "carrier": "c01"}], "pairs": []}
    assert D.phase_capture(cfg, bank) == D.RC_OK
    by_prefix = {c["prefix"].rsplit("/", 1)[-1]: c for c in calls}
    assert set(by_prefix) == {"va2564", "vc2564", "manifests"}
    for c in calls:
        assert c["resume_skip"] is False, c


def _upload_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            f = node.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name == "upload_dir_sharded":
                yield node


def test_embed_upload_call_site_pins_resume_skip_false():
    """r3 missing-fix-regression-tests (a), embed leg: the PC upload call site
    passes a literal resume_skip=False (AST pin — the call sits behind the
    vLLM engine, unreachable by a CPU spy test)."""
    tree = ast.parse((_SCRIPTS / "issue2564_embed.py").read_text())
    calls = list(_upload_calls(tree))
    assert calls, "embed upload_dir_sharded call site missing"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "resume_skip" in kw, f"issue2564_embed.py:{call.lineno} lacks resume_skip"
        assert isinstance(kw["resume_skip"], ast.Constant) and kw["resume_skip"].value is False


def _loader_calls(node: ast.AST):
    for n in ast.walk(node):
        if isinstance(n, ast.Call):
            f = n.func
            name = f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", None)
            if name in ("from_pretrained", "LLM"):
                yield n


def test_production_loader_calls_receive_revision_kwarg():
    """r3 missing-fix-regression-tests (b): every production from_pretrained /
    vLLM LLM(...) loader call in the #2564 scripts — and in the shared
    issue2162_run.load_model_and_tokenizer the driver dispatches — threads the
    resolved revision kwarg (r2 blocker 8: provenance label == loaded bytes)."""
    for rel in ("issue2564_run.py", "issue2564_embed.py"):
        tree = ast.parse((_SCRIPTS / rel).read_text())
        calls = list(_loader_calls(tree))
        assert calls, f"{rel}: no loader calls found (pin drifted?)"
        for call in calls:
            assert any(k.arg == "revision" for k in call.keywords), (
                f"{rel}:{call.lineno} loader call lacks revision="
            )
    shared = ast.parse((_SCRIPTS / "issue2162_run.py").read_text())
    fn = next(
        n
        for n in ast.walk(shared)
        if isinstance(n, ast.FunctionDef) and n.name == "load_model_and_tokenizer"
    )
    fn_calls = list(_loader_calls(fn))
    assert fn_calls, "issue2162_run.load_model_and_tokenizer: no loader calls found"
    for call in fn_calls:
        assert any(k.arg == "revision" for k in call.keywords), (
            f"issue2162_run.py:{call.lineno} loader call lacks revision="
        )


def test_embed_completed_sentinel_is_fp_keyed_and_mode_matched(tmp_path):
    """r3 phase-c-not-idempotent: the phase-entry skip keys on the CURRENT
    mode's sentinel AND its regime_fp; wrong fp or wrong mode recomputes."""
    fp = "ab" * 8
    assert E._completed_sentinel(tmp_path, True, fp) is None  # nothing on disk
    (tmp_path / "embed_done.local.json").write_text(json.dumps({"regime_fp": fp}))
    assert E._completed_sentinel(tmp_path, True, fp) == {"regime_fp": fp}
    assert E._completed_sentinel(tmp_path, True, "00" * 8) is None  # stale regime
    # upload mode skips ONLY on the uploaded sentinel (local proves compute, not upload)
    assert E._completed_sentinel(tmp_path, False, fp) is None
    (tmp_path / "embed_uploaded.json").write_text(json.dumps({"regime_fp": fp}))
    assert E._completed_sentinel(tmp_path, False, fp) == {"regime_fp": fp}


def test_embed_force_quarantines_phase_sentinels_only(tmp_path):
    """r3 phase-c-not-idempotent: --force quarantines BOTH phase sentinels
    (atomic replace into quarantine/) and leaves chunk checkpoints untouched."""
    (tmp_path / "embed_done.local.json").write_text("{}")
    (tmp_path / "embed_uploaded.json").write_text("{}")
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "chunk_000.npz").write_bytes(b"x")
    E._quarantine_sentinels(tmp_path)
    assert not (tmp_path / "embed_done.local.json").exists()
    assert not (tmp_path / "embed_uploaded.json").exists()
    assert len(list((tmp_path / "quarantine").iterdir())) == 2
    assert (chunks / "chunk_000.npz").exists()  # chunk resume state stays honored


def test_phase_embed_threads_force_flag(monkeypatch, tmp_path):
    """r3 phase-c-not-idempotent: the driver's --force reaches the embed
    subprocess as --force (the _PHASE_COMPLETION_RECORDS['C'] contract)."""
    args = D.parse_args(
        [
            "--phase",
            "C",
            "--out-root",
            str(tmp_path / "root4"),
            "--smoke",
            "--upload",
            "none",
            "--force",
        ]
    )
    cfg = D.build_config(args)
    bank = {"contexts": [{"id": "x", "cell": "register", "carrier": "c01"}], "pairs": []}
    monkeypatch.setattr(D, "_anchor_cell_complete", lambda cfg, cell: True)
    captured: dict = {}

    def fake_run(cmd, env):
        captured["cmd"] = [str(c) for c in cmd]
        out = D._embed_out_root(cfg)
        out.mkdir(parents=True, exist_ok=True)
        (out / "embed_done.local.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(D.subprocess, "run", fake_run)
    assert D.phase_embed(cfg, bank) == D.RC_OK
    assert "--force" in captured["cmd"]


# ── r2 ffr pins: all-axes-fail RC_OK exit + pilot judge reissue/fingerprint ─

import json as _json  # noqa: E402
from dataclasses import replace as _replace  # noqa: E402
from unittest.mock import create_autospec  # noqa: E402

import issue2564_judge as JD  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from explore_persona_space.eval.graded_judge import JudgeResult, judge_graded  # noqa: E402
from explore_persona_space.experiments.issue2564 import bank2564 as BK  # noqa: E402


@pytest.fixture(scope="module")
def qwen_tok():
    try:
        return AutoTokenizer.from_pretrained(BK.MODEL_ID, local_files_only=True)
    except OSError:
        return AutoTokenizer.from_pretrained(BK.MODEL_ID)


@pytest.fixture(scope="module")
def ffr_values():
    return BK.load_values_ffr()


def _all_fail_selection(values: dict) -> dict:
    comply = {vid: 0 for axis in BK.FFR_AXES for vid in BK.value_ids(values, axis)}
    sel = BK.select_ffr_values(values, comply)
    assert sel["surviving_axes"] == []
    return sel


def _ffr_argv(tmp_path, phase: str) -> list[str]:
    return [
        "--round",
        "ffr",
        "--phase",
        phase,
        "--out-root",
        str(tmp_path / "eps2564ffr"),
        "--tiny",
        "--upload",
        "none",
        "--log-dir",
        str(tmp_path / "logs"),
    ]


def test_ffr_all_axes_fail_phase_a_exits_ok(tmp_path, capsys, ffr_values):
    """r1 blocker ffr-all-axes-fail-crash: the plan-§6-registered VALID
    all-axes-fail outcome exits RC_OK with a durable refusal-boundary record
    — never a BankGateError crash."""
    argv = _ffr_argv(tmp_path, "A")
    cfg = D.build_config(D.parse_args(argv))
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    (cfg.manifest_dir / "pilot_selection.json").write_text(
        _json.dumps({"selection": _all_fail_selection(ffr_values)})
    )
    rc = D.main(argv)
    assert rc == D.RC_OK
    assert "[phase=done]" in capsys.readouterr().out
    rb = _json.loads((cfg.manifest_dir / "ffr_refusal_boundary.json").read_text())
    assert rb["outcome"] == "refusal_boundary_all_axes_failed"
    assert rb["surviving_axes"] == []
    assert rb["phases_skipped"] == ["A", "B"]
    assert not (cfg.manifest_dir / BK.FFR_BANK_MANIFEST_FILENAME).exists()


def test_ffr_all_axes_fail_phase_all_exits_ok(tmp_path, capsys, ffr_values):
    """--phase all, real path: pilot sentinels satisfied -> phase_pilot skips,
    the all-axes-fail guard exits RC_OK (the exact blocker invocation)."""
    argv = _ffr_argv(tmp_path, "all")
    cfg = D.build_config(D.parse_args(argv))
    cfg.model_revision = "unresolved-tiny"  # what main() resolves under --tiny
    pcfg = D._pilot_cfg(cfg)
    pcfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    pcfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    # tiny slice: 3 smoke value ids x carriers c01-c03 -> the 3 axis cells
    for cell in BK.FFR_AXES:
        (pcfg.anchors_dir / f"anchors_{cell}.jsonl").write_text(
            _json.dumps({"context_id": f"{cell}__x__c01", "draw": 0, "text": "t"}) + "\n"
        )
        (pcfg.manifest_dir / f"anchors_{cell}.done.json").write_text(
            _json.dumps({"regime_fp": D._cell_fp(pcfg, "A", cell), "n_rows": 1})
        )
    (cfg.out_root / "ffr_pilot_anchors_done.json").write_text(
        _json.dumps({"regime_fp": D._regime_fp(pcfg, {"phase": "pilot", "leg": "gen"})})
    )
    (cfg.out_root / "ffr_pilot_selection_done.json").write_text(
        _json.dumps({"regime_fp": D._pilot_selection_fp(pcfg)})
    )
    (cfg.manifest_dir / "pilot_selection.json").write_text(
        _json.dumps({"selection": _all_fail_selection(ffr_values)})
    )
    rc = D.main(argv)
    assert rc == D.RC_OK
    out = capsys.readouterr().out
    assert "selection sentinel present" in out
    assert "[phase=done]" in out
    rb = _json.loads((cfg.manifest_dir / "ffr_refusal_boundary.json").read_text())
    assert rb["outcome"] == "refusal_boundary_all_axes_failed"
    assert not (cfg.manifest_dir / BK.FFR_BANK_MANIFEST_FILENAME).exists()


# ── pilot judge: reissue loop, refusal accounting, selection fingerprint ─


def _pilot_fixture(tmp_path, qwen_tok, ffr_values):
    """Non-tiny pilot cfg with SYNTHESIZED complete generation state (the
    resume-matrix pilot-anchors leg): per-cell done manifests + anchor rows
    for all 276 contexts x 2 draws, so phase_pilot runs the judge+selection
    leg only."""
    argv = [
        "--round",
        "ffr",
        "--phase",
        "pilot",
        "--out-root",
        str(tmp_path / "eps2564ffr"),
        "--upload",
        "none",
        "--log-dir",
        str(tmp_path / "logs"),
    ]
    cfg = D.build_config(D.parse_args(argv))
    pcfg = D._pilot_cfg(cfg)
    pcfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    pcfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    bank = BK.build_pilot_bank_ffr(qwen_tok, values=ffr_values)
    by_cell: dict[str, list[dict]] = {}
    for c in bank["contexts"].values():
        by_cell.setdefault(c["cell"], []).append(c)
    for cell, ctxs in by_cell.items():
        rows = [
            _json.dumps({"context_id": c["id"], "draw": d, "text": f"answer {c['id']} {d}"})
            for c in ctxs
            for d in range(pcfg.draws)
        ]
        (pcfg.anchors_dir / f"anchors_{cell}.jsonl").write_text("\n".join(rows) + "\n")
        (pcfg.manifest_dir / f"anchors_{cell}.done.json").write_text(
            _json.dumps({"regime_fp": D._cell_fp(pcfg, "A", cell), "n_rows": len(rows)})
        )
    return cfg


class _FakeJudge:
    """Stateful judge_graded stand-in (the external API boundary), built via
    create_autospec so the call signature stays conformant by construction."""

    def __init__(self, n_refusal_calls: int):
        self.n_refusal_calls = n_refusal_calls
        self.calls = 0

    def __call__(self, items, eval_prompt, **kwargs):
        self.calls += 1
        scores: dict[str, float | None] = {a: 80.0 for a, _q, _t in items}
        if self.calls <= self.n_refusal_calls:
            first = items[0][0]
            scores[first] = None
            return JudgeResult(
                scores=scores,
                n_total_draws=len(items),
                n_dropped_draws=0,
                n_api_refusal_draws=1,
                per_item_api_refusals={first: 1},
            )
        return JudgeResult(scores=scores, n_total_draws=len(items), n_dropped_draws=0)


def _patch_judge(monkeypatch, fake: _FakeJudge):
    spec = create_autospec(judge_graded, side_effect=fake)
    monkeypatch.setattr("explore_persona_space.eval.graded_judge.judge_graded", spec)
    return spec


def test_ffr_pilot_judge_reissues_api_refusals_then_selects(
    tmp_path, monkeypatch, qwen_tok, ffr_values
):
    """r1 blocker ffr-api-refusal-censoring: an api-refused draw is reissued
    (rule 28), never blended into drops/noncompliance; sel_doc reports the
    rules-9/24/28 classes separately."""
    cfg = _pilot_fixture(tmp_path, qwen_tok, ffr_values)
    fake = _FakeJudge(n_refusal_calls=1)
    _patch_judge(monkeypatch, fake)
    assert D.phase_pilot(cfg, qwen_tok, ffr_values) == D.RC_OK
    assert fake.calls == 2  # first call refused one draw -> ONE reissue round
    sel_doc = _json.loads((cfg.manifest_dir / "pilot_selection.json").read_text())
    j = sel_doc["judge"]
    assert j["n_api_refusal_draws"] == 0  # final merged state: refusal reissued
    assert j["n_reissue_rounds"] == 1
    assert j["n_content_dropped_draws"] == 0
    assert j["n_transport_lost_draws"] == 0
    assert sel_doc["n_dropped_judgments"] == 0  # refusal NEVER counted as a drop
    assert set(sel_doc["selection"]["surviving_axes"]) == set(BK.FFR_AXES)
    assert (cfg.out_root / "ffr_pilot_selection_done.json").is_file()


def test_ffr_pilot_judge_persistent_refusal_fails_loud(tmp_path, monkeypatch, qwen_tok, ffr_values):
    cfg = _pilot_fixture(tmp_path, qwen_tok, ffr_values)
    fake = _FakeJudge(n_refusal_calls=99)
    _patch_judge(monkeypatch, fake)
    with pytest.raises(RuntimeError, match="api-refusal"):
        D.phase_pilot(cfg, qwen_tok, ffr_values)
    assert fake.calls == 1 + JD.FFR_JUDGE_REISSUE_ROUNDS
    assert not (cfg.manifest_dir / "pilot_selection.json").exists()


def test_ffr_pilot_selection_fp_pins_judge_instrument_and_upload(
    tmp_path, monkeypatch, qwen_tok, ffr_values
):
    """r1 codex concern ffr-selection-sentinel-fingerprint: the selection
    sentinel skip invalidates when the judge instrument changes; the upload
    destination is part of the fingerprint."""
    cfg = _pilot_fixture(tmp_path, qwen_tok, ffr_values)
    fake = _FakeJudge(n_refusal_calls=0)
    _patch_judge(monkeypatch, fake)
    assert D.phase_pilot(cfg, qwen_tok, ffr_values) == D.RC_OK
    assert fake.calls == 1
    # sentinel skip: a re-run with the same instrument judges NOTHING
    assert D.phase_pilot(cfg, qwen_tok, ffr_values) == D.RC_OK
    assert fake.calls == 1
    # instrument change invalidates the sentinel -> re-judge
    monkeypatch.setattr(JD, "JUDGE_MODEL", "some-other-judge")
    assert D.phase_pilot(cfg, qwen_tok, ffr_values) == D.RC_OK
    assert fake.calls == 2
    # upload destination is fingerprinted (a --upload none completion cannot
    # satisfy a later --upload hf run)
    pcfg = D._pilot_cfg(cfg)
    assert D._pilot_selection_fp(_replace(pcfg, upload="hf")) != D._pilot_selection_fp(pcfg)


# ── k100 round pins (plan v8; follow-up k100-low-reliability-axes) ──────

import issue2564_analysis as A  # noqa: E402


def test_k100_build_config_production_defaults():
    """--round k100 production: roster cells, 90 fresh draws, draw offset 10,
    dedicated out-root, pinned parent revision, k100-named sentinels."""
    cfg = D.build_config(D.parse_args(["--phase", "all", "--round", "k100", "--upload", "hf"]))
    assert cfg.is_k100 and not cfg.is_ffr
    assert cfg.cells == D.K100_CELLS == ("user_fact", "query")
    assert cfg.carriers is None  # full 12-carrier roster in production
    assert cfg.draws == D.K100_DRAWS == 90
    assert cfg.draw_offset == D.K100_DRAW_OFFSET == 10
    assert cfg.parent_revision == D.K100_PARENT_REVISION_DEFAULT
    assert cfg.out_root == Path(D.K100_OUT_ROOT_DEFAULT) == Path("/workspace/eps2564k100")
    assert cfg.hf_prefix == "issue2564_minpair"
    assert cfg.anchors_sentinel.name == "k100_anchors_done.json"
    assert cfg.va_sentinel.name == "k100_va_uploaded.json"


def test_k100_build_config_smoke_keeps_round_cells_slices_carriers():
    """k100 smoke keeps the ROUND's cells (parent SMOKE_CELLS carries
    'register', outside the k100 roster) and slices carriers/draws only;
    generated artifacts rebind to the smoke twin root + smoke_k100 prefix."""
    cfg = D.build_config(
        D.parse_args(
            [
                "--phase",
                "all",
                "--round",
                "k100",
                "--out-root",
                "/tmp/eps2564k100x",
                "--smoke",
                "--upload",
                "none",
            ]
        )
    )
    assert cfg.cells == D.K100_CELLS  # NOT D.SMOKE_CELLS
    assert cfg.carriers == D.SMOKE_CARRIERS == ("c01", "c02", "c03")
    assert cfg.draws == D.SMOKE_DRAWS == 2
    assert cfg.draw_offset == D.K100_DRAW_OFFSET
    assert cfg.out_root.name == "smoke_eps2564k100x"
    assert cfg.hf_prefix == "issue2564_minpair/smoke_k100"


def test_k100_gen_row_draw_and_seed_bookkeeping():
    """Fresh rows carry draw ids 10..99 with the parent seed invariant
    seed = seed_base + draw (plan v8 §3a: seeds 52..141 at seed_base 42)."""
    cfg = D.build_config(D.parse_args(["--phase", "all", "--round", "k100", "--upload", "none"]))
    assert cfg.seed_base == 42
    ctx = {
        "id": "user_fact::n01::c01",
        "cell": "user_fact",
        "kind": "value",
        "value_id": "n01",
        "carrier": "c01",
        "form": "stmt",
    }
    for i in (0, 41, 89):
        row = D._gen_row(cfg, ctx, 7, 1, cfg.draw_offset + i, 0, "text", 3, 128)
        assert row["draw"] == 10 + i
        assert row["seed"] == 42 + 10 + i  # seed = seed_base + draw


def test_k100_regime_fp_includes_draw_offset():
    """The generation regime fingerprint changes when the draw offset does —
    a parent-regime partial can never satisfy a k100 resume (plan v8 §3a)."""
    cfg = D.build_config(D.parse_args(["--phase", "all", "--round", "k100", "--upload", "none"]))
    assert D._regime_fp(cfg) != D._regime_fp(_replace(cfg, draw_offset=0))


def test_k100_prefix_isolation_never_writes_parent_or_ffr():
    """k100 write surfaces (HF kind prefixes, out-roots, sentinels, smoke
    prefix, embed anchors rel) are disjoint from the parent's AND the ffr
    round's — a k100 run can never clobber a committed prefix (plan v8 §5)."""
    assert A.K100_ROUND_SEG == D.K100_ROUND_SEG == E.K100_ROUND_SEG
    assert A.K100_DRAW_OFFSET == D.K100_DRAW_OFFSET and A.K100_DRAWS_TOTAL == 100

    def _cfg(extra: list[str]) -> D.Cfg2564:
        return D.build_config(D.parse_args(["--phase", "all", *extra, "--upload", "hf"]))

    k100, parent, ffr = _cfg(["--round", "k100"]), _cfg([]), _cfg(["--round", "ffr"])
    kinds = ("raw_completions", "analysis_tensors", "manifests")
    k_set = {k100.hf_round_prefix(k) for k in kinds}
    other = {c.hf_round_prefix(k) for c in (parent, ffr) for k in kinds}
    assert not (k_set & other), k_set & other
    for k in kinds:
        assert k100.hf_round_prefix(k) == f"issue2564_minpair/{k}/k100_low_reliability_axes"
    assert len({k100.out_root, parent.out_root, ffr.out_root}) == 3
    k_sent = {k100.anchors_sentinel.name, k100.va_sentinel.name}
    o_sent = {
        parent.anchors_sentinel.name,
        parent.va_sentinel.name,
        ffr.anchors_sentinel.name,
        ffr.va_sentinel.name,
    }
    assert k_sent.isdisjoint(o_sent)
    # smoke prefixes are round-distinct too
    smoke = {
        r: _cfg([*(["--round", r] if r != "parent" else []), "--smoke"]).hf_prefix
        for r in ("parent", "ffr", "k100")
    }
    assert len(set(smoke.values())) == 3 and smoke["k100"].endswith("/smoke_k100")
    # embed-side anchors rel nests the round segment inside raw_completions
    assert (
        E.hf_anchors_rel("query", E.K100_ROUND_SEG)
        == "raw_completions/k100_low_reliability_axes/anchors/anchors_query.jsonl"
    )
    assert E.hf_anchors_rel("query") == "raw_completions/anchors/anchors_query.jsonl"


def test_k100_phase_embed_forces_query_cells_and_threads_round_seg(monkeypatch, tmp_path):
    """k100 phase C embeds ONLY the query cell (user_fact is programmatic;
    plan v8 §5) and threads --hf-round-seg so the subprocess uploads to the
    round-nested prefix; a bank without the query cell fails loud."""
    args = D.parse_args(
        [
            "--phase",
            "C",
            "--round",
            "k100",
            "--out-root",
            str(tmp_path / "rootk"),
            "--smoke",
            "--upload",
            "none",
        ]
    )
    cfg = D.build_config(args)
    bank = {
        "contexts": [
            {"id": "u", "cell": "user_fact", "carrier": "c01"},
            {"id": "q", "cell": "query", "carrier": "c01"},
        ],
        "pairs": [],
    }
    monkeypatch.setattr(D, "_anchor_cell_complete", lambda cfg, cell: True)
    captured: dict = {}

    def fake_run(cmd, env):  # signature mirror of the subprocess.run call site
        captured["cmd"] = [str(c) for c in cmd]
        out = D._embed_out_root(cfg)
        out.mkdir(parents=True, exist_ok=True)
        (out / "embed_done.local.json").write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(D.subprocess, "run", fake_run)
    assert D.phase_embed(cfg, bank) == D.RC_OK
    cmd = captured["cmd"]
    assert cmd[cmd.index("--cells") + 1] == "query"  # user_fact never reaches the embed leg
    assert cmd[cmd.index("--hf-round-seg") + 1] == D.K100_ROUND_SEG
    # no query cell in the bank -> loud refusal, not a silent empty embed
    bank_uf = {"contexts": [{"id": "u", "cell": "user_fact", "carrier": "c01"}], "pairs": []}
    with pytest.raises(RuntimeError, match="requires the query cell"):
        D.phase_embed(cfg, bank_uf)


def test_k100_vc_parity_report_names_k6_vc_source():
    """Consistency note 1 (plan v8 K6): the parity manifest states that the
    fresh k100 v_C feeds ONLY provenance check (a) — pooled/bridge reads
    consume the PARENT committed bank at the pinned revision."""
    src = inspect.getsource(D._k100_vc_parity)
    assert '"k6_vc_source"' in src
    assert "feeds ONLY this provenance check" in src


def test_k100_sb_project_arithmetic_and_edge_cases():
    """sb_project matches the closed-form r1 projection, reduces to the
    spearman_brown step-up at 1->2, is the identity at k_from == k_to, and
    is NaN-safe (plan v8 §3b)."""
    r10 = 0.13102069808322983  # committed parent user_fact r10 (bridge target)
    r1 = r10 / (10 - 9 * r10)
    manual = 100 * r1 / (1 + 99 * r1)
    assert abs(A.sb_project(r10, 10, 100) - manual) < 1e-12
    assert 0.55 < manual < 0.65  # the plan's ~0.60 projected pooled r100
    assert abs(A.sb_project(0.3, 1, 2) - A.spearman_brown(0.3)) < 1e-15
    assert abs(A.sb_project(0.4, 10, 10) - 0.4) < 1e-15
    out = A.sb_project(np.array([0.4, np.nan]), 10, 100)
    assert np.isfinite(out[0]) and np.isnan(out[1])


def test_k100_ci_overlap_predicate_and_registered_criterion():
    """Fallback criterion mechanics (consistency note 2): strict CI
    non-overlap, touching edges overlap, non-finite -> None; the
    pre-registered criterion string is pinned in-code."""
    assert A._ci_overlap([0.1, 0.3], [0.25, 0.5]) is True
    assert A._ci_overlap([0.1, 0.2], [0.21, 0.5]) is False
    assert A._ci_overlap([0.21, 0.5], [0.1, 0.2]) is False
    assert A._ci_overlap([0.1, 0.2], [0.2, 0.3]) is True  # non-overlap is STRICT
    assert A._ci_overlap([0.1, float("nan")], [0.2, 0.3]) is None
    assert "PRE-REGISTERED" in A.K100_FALLBACK_CRITERION
    assert "do NOT overlap" in A.K100_FALLBACK_CRITERION


def _k100_deciding_block(ci: tuple[float, float], threshold: float) -> dict:
    return {
        "ci95": list(ci),
        "threshold": threshold,
        "fragile": bool(ci[0] <= threshold <= ci[1]),
        "n_finite_draws": 8,
        "scheme": "carrier-clustered paired bootstrap (shared resample)",
    }


def _k100_verdict_doc(
    c: float,
    r100: float,
    s_flip: float,
    s_para: float,
    t_ratio: float,
    *,
    ratio_ci: tuple[float, float] = (0.45, 0.55),
    g_ci: tuple[float, float] = (0.5, 0.9),
) -> dict:
    return {
        "axes": {
            "user_fact": {
                "direction": {"arm_779ce": {"mean_cos_headline": c, "ci95": [c - 0.01, c + 0.01]}},
                "reliability": {"r100_mean": r100, "r100_ci95": [r100 - 0.001, r100 + 0.001]},
                "k100_deciding": {"c_over_b": _k100_deciding_block(ratio_ci, A.K100_LATTICE_RATIO)},
            },
            "query_form": {
                "surface": {"observed": {"flip_norm_mean": s_flip, "para_norm_mean": s_para}},
                "text_space": {"flip_over_para_ratio": t_ratio},
                "k100_deciding": {"g": _k100_deciding_block(g_ci, A.K100_DISSOC_G)},
            },
        }
    }


def test_k100_verdict_lattice_arms():
    """All three injected-name lattice arms + both dissociation arms fire on
    point estimates exactly as registered (plan v8 §3b)."""
    # b = sqrt(0.36) = 0.6 >= 0.55; c/b = 0.30/0.6 = 0.5 >= 0.35
    v = A.k100_verdicts(_k100_verdict_doc(0.30, 0.36, 0.9, 1.0, 0.2), "pooled", smoke=False)
    assert v["injected_name"]["verdict"] == "reliability-limited"
    assert v["query_form_dissociation"]["verdict"] == "dissociation-holds"  # g = 0.7
    assert v["reliability_estimator"] == "pooled" and v["informational_smoke"] is False
    # c/b = 0.15/0.6 = 0.25 < 0.35
    v = A.k100_verdicts(_k100_verdict_doc(0.15, 0.36, 0.9, 1.0, 0.85), "pooled", smoke=False)
    assert v["injected_name"]["verdict"] == "map-direction-loss"
    assert v["query_form_dissociation"]["verdict"] == "dissociation-collapses"  # g = 0.05
    # b = sqrt(0.25) = 0.5 < 0.55
    v = A.k100_verdicts(_k100_verdict_doc(0.30, 0.25, 0.9, 1.0, 0.2), "new_only", smoke=True)
    assert v["injected_name"]["verdict"] == "unresolved"
    assert v["informational_smoke"] is True
    # fragile flag: r CI straddling the b floor (b = 0.5501.. within +-ci)
    doc = _k100_verdict_doc(0.30, 0.5501**2, 0.9, 1.0, 0.2)
    doc["axes"]["user_fact"]["reliability"]["r100_ci95"] = [0.54**2, 0.57**2]
    v = A.k100_verdicts(doc, "pooled", smoke=False)
    assert v["injected_name"]["fragile"] is True


# ── k100 pooled loader + split-half pins (synthetic dual-source stores) ─


def _k100_world(tmp_path):
    """Synthetic dual-source k100 store fixture: 6-context parent vc grid
    (non-roster extras first + last, pinning parent_rows slicing), 4 roster
    contexts (2 user_fact + 2 query), parent draws 0-9 / round draws 10-99,
    tail[r, k, :] = base_r + k * dir_r (+ per-layer offset)."""
    d, e = 4, 3
    rng = np.random.default_rng(0)
    vc_ids = [
        "register::x1::c01",
        "user_fact::n01::c01",
        "query::E::c01",
        "user_fact::n02::c01",
        "query::E::c02",
        "register::x2::c01",
    ]
    cells_of = {cid: cid.split("::")[0] for cid in vc_ids}
    roster = vc_ids[1:5]
    base = rng.normal(size=(6, d))
    dirv = rng.normal(size=(6, d))
    li_off = {L: 100.0 * i for i, L in enumerate(A.LAYERS)}
    contexts = {
        cid: {"id": cid, "cell": cells_of[cid], "carrier": cid.split("::")[2]} for cid in vc_ids
    }
    bank = {"contexts": contexts, "pairs": []}
    paths: dict[tuple[str, str], Path] = {}
    root = tmp_path / "stores"
    root.mkdir(parents=True, exist_ok=True)

    vc = np.stack([np.stack([base[r] + li_off[L] for L in A.LAYERS]) for r in range(6)])
    vc_p = root / "vc.pt"
    torch.save(
        {
            "layers": list(A.LAYERS),
            "context_ids": vc_ids,
            "vc": torch.tensor(vc, dtype=torch.float32),
        },
        vc_p,
    )
    paths[("analysis_tensors/vc2564/vc2564_bank.pt", "parent")] = vc_p

    for cell in ("user_fact", "query"):
        cell_ids = [cid for cid in roster if cells_of[cid] == cell]
        for source, drange, ncomp in (("parent", range(0, 10), 5), ("round", range(10, 100), 7)):
            index, tails = [], []
            for cid in cell_ids:
                r = vc_ids.index(cid)
                for k in drange:
                    index.append({"context_id": cid, "draw": k, "n_completion_tokens": ncomp})
                    row = base[r] + k * dirv[r]
                    tails.append(np.stack([row + li_off[L] for L in A.LAYERS]))
            tail = torch.tensor(np.array(tails), dtype=torch.float32)
            p = root / f"va_{cell}_{source}.pt"
            torch.save(
                {
                    "layers": list(A.LAYERS),
                    "index": index,
                    "va_tail_incl": tail,
                    "va_span": tail + 0.5,
                },
                p,
            )
            paths[(f"analysis_tensors/va2564/va2564_{cell}.pt", source)] = p

    emb_mean = rng.normal(size=(6, e))
    means_p = root / "means.npz"
    np.savez(means_p, context_ids=np.array(vc_ids), emb_mean=emb_mean)
    paths[("analysis_tensors/embeddings_qwen3_8b/means_anchors.npz", "parent")] = means_p
    q_ids = [cid for cid in roster if cells_of[cid] == "query"]
    for source, drange in (("parent", range(0, 10)), ("round", range(10, 100))):
        ids, draws, embs = [], [], []
        for cid in q_ids:
            r = vc_ids.index(cid)
            for k in drange:
                ids.append(cid)
                draws.append(k)
                embs.append(emb_mean[r] + k)
        if source == "parent":
            # a user_fact per-draw row the query-only pooling must IGNORE
            ids.append("user_fact::n01::c01")
            draws.append(0)
            embs.append(np.full(e, 999.0))
        p = root / f"perdraw_{source}.npz"
        np.savez(
            p, context_ids=np.array(ids), draws=np.array(draws, dtype=np.int64), emb=np.array(embs)
        )
        paths[("analysis_tensors/embeddings_qwen3_8b/perdraw_anchors.npz", source)] = p

    cfg = A.CfgPE(
        in_root=None,
        out_dir=tmp_path / "out",
        stage_dir=tmp_path / "stage",
        manip_check=tmp_path / "mc.json",
        ridge_779=None,
        ridge_1738=None,
        smoke=False,
        upload="none",
        b_boot=8,
        b_null=8,
        n_splits=4,
        hf_prefix="issue2564_minpair",
        round="k100",
    )
    return SimpleNamespace(
        cfg=cfg,
        bank=bank,
        paths=paths,
        vc_ids=vc_ids,
        roster=roster,
        base=base,
        dirv=dirv,
        emb_mean=emb_mean,
        li_off=li_off,
    )


@pytest.fixture()
def k100_world(tmp_path, monkeypatch):
    w = _k100_world(tmp_path)
    monkeypatch.setattr(
        A, "resolve_input", lambda cfg, rel, *, source="round": w.paths[(rel, source)]
    )
    return w


def test_k100_load_stores_pooled_concat_and_new_only_accumulators(k100_world):
    """load_stores_k100 pools parent draws 0-9 + round draws 10-99 into a
    (n_ctx, 100, d) grid in PARENT vc row order, reconstructs parent-only
    means exactly, fills the *_new accumulators, and K-pools query text
    embeddings while user_fact keeps the parent means (plan v8 K4/K6)."""
    w = k100_world
    st = A.load_stores_k100(w.cfg, w.bank)
    rows = np.array([1, 2, 3, 4])  # roster rows in the 6-context parent grid
    assert st.ctx_ids == w.roster
    np.testing.assert_array_equal(st.parent_rows, rows)
    assert st.n_parent_ctx_total == 6
    assert st.tail_draws.shape == (4, 100, 4) and st.draw_valid.all()
    np.testing.assert_array_equal(st.n_valid, np.full(4, 100))
    np.testing.assert_array_equal(st.n_valid_new, np.full(4, 90))
    # pooled / new-only / parent-only per-context means at the primary layer
    off = w.li_off[A.PRIMARY_LAYER]
    exp_pooled = w.base[rows] + 49.5 * w.dirv[rows] + off  # mean draw over 0..99
    exp_new = w.base[rows] + 54.5 * w.dirv[rows] + off  # mean draw over 10..99
    exp_parent = w.base[rows] + 4.5 * w.dirv[rows] + off  # mean draw over 0..9
    np.testing.assert_allclose(st.va_tail_mean[A.PRIMARY_LAYER], exp_pooled, atol=1e-4)
    np.testing.assert_allclose(st.va_tail_mean_new[A.PRIMARY_LAYER], exp_new, atol=1e-4)
    np.testing.assert_allclose(
        A._k100_parent_only_tail_mean(st, A.PRIMARY_LAYER), exp_parent, atol=1e-3
    )
    # answer lengths: pooled (10*5 + 90*7)/100; new-only 7
    np.testing.assert_allclose(st.ans_len_mean, np.full(4, 6.8), atol=1e-12)
    np.testing.assert_allclose(st.ans_len_mean_new, np.full(4, 7.0), atol=1e-12)
    # embeddings: query rows K-pooled (parent + round per-draw), user_fact parent means
    uf_rows = [0, 2]  # uf1, uf2 in roster order [uf1, q1, uf2, q2]
    q_rows = [1, 3]
    np.testing.assert_allclose(st.emb_mean[uf_rows], w.emb_mean[[1, 3]], atol=1e-12)
    np.testing.assert_allclose(st.emb_mean[q_rows], w.emb_mean[[2, 4]] + 49.5, atol=1e-9)
    np.testing.assert_allclose(st.emb_mean_new[q_rows], w.emb_mean[[2, 4]] + 54.5, atol=1e-9)
    np.testing.assert_allclose(st.emb_mean_new[uf_rows], w.emb_mean[[1, 3]], atol=1e-12)
    # dual-source provenance recorded per input
    assert st.input_files["va2564_user_fact.parent.pt"]["source"] == "parent"
    assert st.input_files["va2564_user_fact.pt"]["source"] == "round"


def _uf_pair(st) -> A.PairArrays:
    return A.PairArrays(
        ids=["p1"],
        cls=["value_swap"],
        axis=["user_fact"],
        value_a=["n01"],
        value_b=["n02"],
        carrier_str=["c01"],
        a=np.array([st.row_of["user_fact::n01::c01"]], dtype=np.int64),
        b=np.array([st.row_of["user_fact::n02::c01"]], dtype=np.int64),
        ca=np.array([0], dtype=np.int64),
        cb=np.array([0], dtype=np.int64),
        dyad=np.array([False]),
        changed=np.array([1], dtype=np.int64),
        orientation=["n01->n02"],
        n=1,
    )


def test_k100_split_half_runs_at_nv100_with_5050_halves(k100_world):
    """At the pooled nv=100 grid the split is 50/50: with deterministic
    ascending scores half1 = draws 0..49, so the per-pair r equals the
    manually computed cosine of the two half-mean deltas."""
    w = k100_world
    st = A.load_stores_k100(w.cfg, w.bank)
    pa = _uf_pair(st)
    seen: list[tuple[int, int]] = []

    def det_scores(rng, n_ctx, k_max):
        seen.append((n_ctx, k_max))
        return np.broadcast_to(np.arange(k_max, dtype=float), (n_ctx, k_max)).copy()

    rel = A.split_half_stats(st, pa, 2, scores_fn=det_scores)
    assert seen == [(4, 100), (4, 100)]  # split drawn at the POOLED 100-draw grid
    a, b = 1, 3  # vc rows of the pair's contexts
    d1 = (w.base[a] - w.base[b]) + 24.5 * (w.dirv[a] - w.dirv[b])  # mean draws 0..49
    d2 = (w.base[a] - w.base[b]) + 74.5 * (w.dirv[a] - w.dirv[b])  # mean draws 50..99
    manual = float(d1 @ d2 / (np.linalg.norm(d1) * np.linalg.norm(d2)))
    np.testing.assert_allclose(rel["r_half"][0], manual, atol=1e-5)
    np.testing.assert_allclose(rel["r_full"][0], 2 * manual / (1 + manual), atol=1e-5)


def test_k100_split_half_analytic_zero_noise_reliability(k100_world):
    """Zero draw noise -> split-half r == 1 and noise norm == 0 exactly (the
    analytic-reliability sanity pin on the pooled 100-draw grid)."""
    w = k100_world
    st = A.load_stores_k100(w.cfg, w.bank)
    rows = np.array([1, 2, 3, 4])
    const = np.repeat(w.base[rows][:, None, :], 100, axis=1).astype(np.float32)
    st0 = _replace(st, tail_draws=const)
    rel = A.split_half_stats(st0, _uf_pair(st), 3)
    np.testing.assert_allclose(rel["r_half"][0], 1.0, atol=1e-6)
    np.testing.assert_allclose(rel["r_full"][0], 1.0, atol=1e-6)
    np.testing.assert_allclose(rel["noise_norm"][0], 0.0, atol=1e-5)


def test_k100_new_only_stores_swaps_means_and_slices_draw_axis(k100_world):
    """The registered-fallback twin swaps every pooled read to the new-only
    estimator and slices the draw axis past the offset (plan v8 §4)."""
    w = k100_world
    st = A.load_stores_k100(w.cfg, w.bank)
    st_new = A.k100_new_only_stores(st)
    assert st_new.tail_draws.shape == (4, 90, 4)
    assert st_new.draw_valid.shape == (4, 90)
    np.testing.assert_array_equal(st_new.n_valid, st.n_valid_new)
    np.testing.assert_allclose(
        st_new.va_tail_mean[A.PRIMARY_LAYER], st.va_tail_mean_new[A.PRIMARY_LAYER]
    )
    np.testing.assert_allclose(st_new.ans_len_mean, st.ans_len_mean_new)
    np.testing.assert_allclose(st_new.emb_mean, st.emb_mean_new)
    np.testing.assert_array_equal(st_new.tail_draws, st.tail_draws[:, 10:])


def _min_split_stores(tail: np.ndarray) -> A.Stores:
    n = tail.shape[0]
    valid = np.ones(tail.shape[:2], dtype=bool)
    return A.Stores(
        ctx_ids=[f"c{i}" for i in range(n)],
        row_of={},
        cells=[],
        carriers=[],
        va_tail_mean={},
        va_span_mean={},
        tail_draws=tail.astype(np.float32),
        draw_valid=valid,
        n_valid=valid.sum(1),
        ans_len_mean=np.zeros(n),
        vc={},
        emb_mean=None,
        d=tail.shape[2],
    )


def test_k100_bridge_parent_grid_scores_reproduce_full_grid_halves(k100_world):
    """The bridge's scores_fn contract (plan v8 K6): drawing split scores at
    the FULL parent grid and slicing parent_rows reproduces the full-grid
    run's per-pair r exactly — every context keeps its parent-realized half
    assignment despite the roster restriction."""
    w = k100_world
    st = A.load_stores_k100(w.cfg, w.bank)
    rows = st.parent_rows
    # full 6-context parent-draw grid (draws 0..9), same values as the loader saw
    full_tail = np.stack(
        [
            np.stack([w.base[r] + k * w.dirv[r] + w.li_off[A.PRIMARY_LAYER] for k in range(10)])
            for r in range(6)
        ]
    )
    full_st = _min_split_stores(full_tail)
    pa_full = _uf_pair(st)
    pa_full = _replace(pa_full, a=np.array([1], dtype=np.int64), b=np.array([3], dtype=np.int64))
    rel_full = A.split_half_stats(full_st, pa_full, 4)

    bridge_st = _replace(st, tail_draws=st.tail_draws[:, :10], draw_valid=st.draw_valid[:, :10])

    def parent_grid_scores(rng, n_ctx, k_max):
        assert (n_ctx, k_max) == (4, 10)
        return rng.random((st.n_parent_ctx_total, k_max))[rows]

    rel_sliced = A.split_half_stats(bridge_st, _uf_pair(st), 4, scores_fn=parent_grid_scores)
    np.testing.assert_allclose(rel_sliced["r_half"][0], rel_full["r_half"][0], atol=1e-12)
    np.testing.assert_allclose(rel_sliced["r_full"][0], rel_full["r_full"][0], atol=1e-12)


# ── k100 fix-round 2 pins (r1 review blockers; epm:code-review v7) ───────


def test_k100_deciding_ci_fragility_flags():
    """r1 blocker k100-verdict-fragility-ci: bootstrap draws whose 95% CI
    straddles the 0.35 ratio / 0.15 g threshold set the fragile flag; clear
    draws do not; an all-NaN draw set is EXPLICITLY not evaluable —
    fragile=None + not_evaluable=True + a durable reason, never a silently
    clean record (r2 nit k100-deciding-ci-degenerate-failopen)."""
    strad = A.k100_deciding_ci(np.linspace(0.30, 0.40, 200), A.K100_LATTICE_RATIO)
    assert strad["fragile"] is True
    assert strad["ci95"][0] <= A.K100_LATTICE_RATIO <= strad["ci95"][1]
    assert "not_evaluable" not in strad
    clear = A.k100_deciding_ci(np.linspace(0.40, 0.50, 200), A.K100_LATTICE_RATIO)
    assert clear["fragile"] is False
    assert "not_evaluable" not in clear
    assert A.k100_deciding_ci(np.linspace(0.10, 0.20, 200), A.K100_DISSOC_G)["fragile"] is True
    assert A.k100_deciding_ci(np.linspace(0.20, 0.30, 200), A.K100_DISSOC_G)["fragile"] is False
    nn = A.k100_deciding_ci(np.full(50, np.nan), A.K100_DISSOC_G)
    assert nn["fragile"] is None and nn["n_finite_draws"] == 0
    assert nn["not_evaluable"] is True and "not evaluable" in nn["reason"]


def test_k100_verdicts_emit_paired_deciding_ci_and_fragile():
    """r1 blocker k100-verdict-fragility-ci: the verdicts consume the PAIRED
    c/b and g bootstrap CIs from compute_all's k100_deciding blocks — a
    straddling CI flags fragile even when the marginal b-floor CI is clear,
    the g verdict now carries g_ci95 + fragile, and a missing deciding block
    fails loud (never a silent point-only verdict)."""
    v = A.k100_verdicts(
        _k100_verdict_doc(0.30, 0.36, 0.9, 1.0, 0.2, ratio_ci=(0.30, 0.40), g_ci=(0.10, 0.75)),
        "pooled",
        smoke=False,
    )
    inj = v["injected_name"]
    assert inj["c_over_b_ci95"] == [0.30, 0.40]
    assert inj["fragile"] is True
    assert inj["fragile_components"]["c_over_b_paired_ci_straddles_ratio"] is True
    assert inj["fragile_components"]["b_ci_straddles_floor"] is False
    qf = v["query_form_dissociation"]
    assert qf["g_ci95"] == [0.10, 0.75] and qf["fragile"] is True
    v2 = A.k100_verdicts(_k100_verdict_doc(0.30, 0.36, 0.9, 1.0, 0.2), "pooled", smoke=False)
    assert v2["injected_name"]["fragile"] is False
    assert v2["query_form_dissociation"]["fragile"] is False
    doc = _k100_verdict_doc(0.30, 0.36, 0.9, 1.0, 0.2)
    del doc["axes"]["user_fact"]["k100_deciding"]
    with pytest.raises(RuntimeError, match="k100_deciding"):
        A.k100_verdicts(doc, "pooled", smoke=False)


def test_k100_verdicts_force_not_evaluable_on_degenerate_deciding_ci():
    """r2 nit k100-deciding-ci-degenerate-failopen: a deciding block built by
    the REAL k100_deciding_ci from an all-NaN draw set forces the associated
    verdict to not_evaluable with a durable verdict_reason and keeps
    fragile=None — never bool(None) -> fragile: false; a real b-floor
    fragility still dominates (fragile=True) even with the paired CI
    unknown."""
    doc = _k100_verdict_doc(0.30, 0.36, 0.9, 1.0, 0.2)
    doc["axes"]["user_fact"]["k100_deciding"]["c_over_b"] = A.k100_deciding_ci(
        np.full(50, np.nan), A.K100_LATTICE_RATIO
    )
    v = A.k100_verdicts(doc, "pooled", smoke=False)
    inj = v["injected_name"]
    assert inj["verdict"] == "not_evaluable"
    assert "not evaluable" in inj["verdict_reason"]
    assert inj["fragile"] is None  # unknown, never coerced to False
    assert inj["fragile_components"]["c_over_b_paired_ci_straddles_ratio"] is None
    qf = v["query_form_dissociation"]  # healthy qf side untouched
    assert qf["verdict"] == "dissociation-holds" and qf["verdict_reason"] is None
    # b-floor fragility dominates: fragile True even with the paired CI unknown
    doc_b = _k100_verdict_doc(0.30, 0.5501**2, 0.9, 1.0, 0.2)
    doc_b["axes"]["user_fact"]["reliability"]["r100_ci95"] = [0.54**2, 0.57**2]
    doc_b["axes"]["user_fact"]["k100_deciding"]["c_over_b"] = A.k100_deciding_ci(
        np.full(50, np.nan), A.K100_LATTICE_RATIO
    )
    v_b = A.k100_verdicts(doc_b, "pooled", smoke=False)
    assert v_b["injected_name"]["verdict"] == "not_evaluable"
    assert v_b["injected_name"]["fragile"] is True
    # query_form side: degenerate g CI forces the dissociation verdict
    doc2 = _k100_verdict_doc(0.30, 0.36, 0.9, 1.0, 0.2)
    doc2["axes"]["query_form"]["k100_deciding"]["g"] = A.k100_deciding_ci(
        np.full(50, np.nan), A.K100_DISSOC_G
    )
    v2 = A.k100_verdicts(doc2, "pooled", smoke=False)
    qf2 = v2["query_form_dissociation"]
    assert qf2["verdict"] == "not_evaluable" and qf2["fragile"] is None
    assert "not evaluable" in qf2["verdict_reason"]
    assert v2["injected_name"]["verdict"] == "reliability-limited"  # uf side untouched


def test_k100_paired_deciding_draws_share_one_resample():
    """r2 concern k100-paired-bootstrap-wiring-unpinned: the deciding-draw
    builders form EVERY component from ONE shared multiplicity matrix —
    per-draw brute-force oracles recompute c/b (both from mult[i]) and g (all
    FOUR components from mult[i]) and must match exactly; an implementation
    that resampled any component independently, or dropped one, diverges from
    the oracle for generic data. compute_all's deciding blocks are wired
    through these named helpers (source pin below)."""
    rng = np.random.default_rng(7)
    n_car, n_pairs, b_boot = 5, 12, 6
    ca = rng.integers(0, n_car, size=n_pairs).astype(np.int64)
    cb = rng.integers(0, n_car, size=n_pairs).astype(np.int64)
    dyad = np.zeros(n_pairs, dtype=bool)
    dyad[::3] = True
    mult = A.carrier_multiplicities(rng.integers(0, n_car, size=(b_boot, n_car)), n_car)
    cos_vals = rng.normal(0.4, 0.1, size=n_pairs)
    r_vals = rng.uniform(0.2, 0.9, size=n_pairs)
    state = rng.uniform(0.5, 2.0, size=n_pairs)
    text = rng.uniform(0.5, 2.0, size=n_pairs)
    head = np.array([0, 2, 3, 5, 7, 9], dtype=np.int64)
    para_head = np.array([1, 4, 6, 8], dtype=np.int64)

    def wmean(vals: np.ndarray, sel: np.ndarray, i: int) -> float:
        w = np.where(dyad[sel], mult[i, ca[sel]] * mult[i, cb[sel]], mult[i, ca[sel]])
        return float(np.sum(w * vals[sel]) / np.sum(w)) if np.sum(w) > 0 else float("nan")

    got_ratio = A.k100_paired_ratio_draws(cos_vals, r_vals, head, ca, cb, dyad, mult)
    got_g = A.k100_paired_g_draws(state, text, head, para_head, ca, cb, dyad, mult)
    assert got_ratio.shape == got_g.shape == (b_boot,)
    for i in range(b_boot):
        c_i = wmean(cos_vals, head, i)
        b_i = max(wmean(r_vals, head, i), 0.0) ** 0.5
        exp_ratio = c_i / b_i if b_i > 0 else float("nan")
        np.testing.assert_allclose(got_ratio[i], exp_ratio, atol=1e-12)
        exp_g = wmean(state, head, i) / wmean(state, para_head, i) - wmean(text, head, i) / wmean(
            text, para_head, i
        )
        np.testing.assert_allclose(got_g[i], exp_g, atol=1e-12)
    # a zero-weight resample (no head carrier drawn) yields a NaN draw, and an
    # all-negative r pool zeroes b -> all-NaN -> not_evaluable via the CI
    zero_row = np.zeros((1, n_car))
    assert np.isnan(A.k100_paired_ratio_draws(cos_vals, r_vals, head, ca, cb, dyad, zero_row))[0]
    all_neg = A.k100_paired_ratio_draws(cos_vals, -r_vals, head, ca, cb, dyad, mult)
    assert np.isnan(all_neg).all()
    assert A.k100_deciding_ci(all_neg, A.K100_LATTICE_RATIO)["not_evaluable"] is True
    # degenerate selections / absent text embeddings -> all-NaN draws
    empty = np.array([], dtype=np.int64)
    assert np.isnan(A.k100_paired_ratio_draws(cos_vals, r_vals, empty, ca, cb, dyad, mult)).all()
    assert np.isnan(A.k100_paired_g_draws(state, None, head, para_head, ca, cb, dyad, mult)).all()
    assert np.isnan(A.k100_paired_g_draws(state, text, head, empty, ca, cb, dyad, mult)).all()
    # wiring pin: compute_all builds its deciding blocks THROUGH the helpers
    # (pairing is by construction — each helper takes exactly ONE mult).
    src = inspect.getsource(A.compute_all)
    assert "k100_paired_ratio_draws(" in src and "k100_paired_g_draws(" in src


def test_k100_vc_parity_binds_under_smoke(tmp_path, monkeypatch):
    """r1 blocker k100-vc-smoke-demotion-unregistered (orchestrator decision:
    CODE fix): a fresh/parent v_C cosine below K100_VC_PARITY_COS_MIN FAILS
    the parity gate under --smoke — never a demoted warning (plan v8 §7: the
    smoke exercises v_C parity); --tiny stays the only demoted (skip) mode."""
    cfg = D.build_config(
        D.parse_args(
            [
                "--phase",
                "B",
                "--round",
                "k100",
                "--upload",
                "none",
                "--smoke",
                "--out-root",
                str(tmp_path / "root"),
            ]
        )
    )
    assert cfg.smoke and not cfg.tiny
    rng = np.random.default_rng(0)
    cid = "user_fact::n01::c01"
    fresh = torch.tensor(rng.normal(size=(1, len(cfg.layers), 8)), dtype=torch.float32)
    parent = torch.tensor(rng.normal(size=(1, len(cfg.layers), 8)), dtype=torch.float32)
    cfg.vc_dir.mkdir(parents=True, exist_ok=True)
    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"layers": list(cfg.layers), "context_ids": [cid], "vc": fresh},
        cfg.vc_dir / "vc2564_bank.pt",
    )
    parent_path = tmp_path / "parent_vc2564_bank.pt"
    torch.save({"layers": list(cfg.layers), "context_ids": [cid], "vc": parent}, parent_path)
    monkeypatch.setattr(
        "explore_persona_space.orchestrate.hub.stage_hub_file",
        lambda *a, **k: parent_path,
    )
    with pytest.raises(RuntimeError, match="v_C parity vs parent FAILED"):
        D._k100_vc_parity(cfg, [{"id": cid}])
    rep = json.loads((cfg.manifest_dir / "k100_vc_parity.json").read_text())
    assert rep["verdict"] == "fail" and rep["demoted"] is False
    assert rep["regime_fp"] == D._k100_parity_fp(cfg)
    assert D._k100_vc_parity_ok(cfg) is False  # a FAIL report never satisfies resume


def test_k100_parity_fp_keyed_on_revision_and_threshold(tmp_path, monkeypatch):
    """r1 blocker k100-parent-revision-cache-unkeyed: the parity-gate fp keys
    on --parent-revision AND K100_VC_PARITY_COS_MIN, so a report written under
    one (revision, bar) can never be reused under another."""
    argv = ["--phase", "all", "--round", "k100", "--upload", "none"]
    argv += ["--out-root", str(tmp_path / "root")]
    cfg_a = D.build_config(D.parse_args(argv))
    cfg_b = D.build_config(D.parse_args([*argv, "--parent-revision", "deadbeef" * 5]))
    fp_a = D._k100_parity_fp(cfg_a)
    assert fp_a != D._k100_parity_fp(cfg_b)
    cfg_a.manifest_dir.mkdir(parents=True, exist_ok=True)
    (cfg_a.manifest_dir / "k100_vc_parity.json").write_text(
        json.dumps({"gate": "k100_vc_parity", "regime_fp": fp_a, "verdict": "pass"})
    )
    assert D._k100_vc_parity_ok(cfg_a) is True
    assert D._k100_vc_parity_ok(cfg_b) is False  # same out-root, different pin
    monkeypatch.setattr(D, "K100_VC_PARITY_COS_MIN", 0.999999)
    assert D._k100_parity_fp(cfg_a) != fp_a
    assert D._k100_vc_parity_ok(cfg_a) is False  # tightened bar invalidates too


def _pb_completion_state(cfg, cells: list[str]) -> None:
    """Write every phase-B completion record (VA cells + vc + gate-4 parity +
    k100 vc parity + sentinel) under CFG's own fingerprints."""
    from explore_persona_space.atomic_io import write_json_atomic

    cfg.manifest_dir.mkdir(parents=True, exist_ok=True)
    cfg.va_dir.mkdir(parents=True, exist_ok=True)
    cfg.vc_dir.mkdir(parents=True, exist_ok=True)
    for cell in cells:
        (cfg.va_dir / f"va2564_{cell}.pt").write_bytes(b"x")
        write_json_atomic(
            cfg.manifest_dir / f"va2564_{cell}.done.json",
            {"regime_fp": D._cell_fp(cfg, "B", cell)},
        )
    (cfg.vc_dir / "vc2564_bank.pt").write_bytes(b"x")
    write_json_atomic(
        cfg.manifest_dir / "vc2564.done.json",
        {"regime_fp": D._regime_fp(cfg, {"phase": "B", "leg": "vc"})},
    )
    write_json_atomic(
        cfg.manifest_dir / "parity_gate_report.json",
        {"regime_fp": D._gate_fp(cfg, "pb_parity"), "verdict": "pass", "demoted": False},
    )
    write_json_atomic(
        cfg.manifest_dir / "k100_vc_parity.json",
        {"gate": "k100_vc_parity", "regime_fp": D._k100_parity_fp(cfg), "verdict": "pass"},
    )
    write_json_atomic(
        cfg.va_sentinel, {"phase": "B", "regime_fp": D._regime_fp(cfg, {"phase": "B"})}
    )


def test_k100_phase_b_fast_path_requires_current_parity_report(tmp_path):
    """r1 blocker k100-parent-revision-cache-unkeyed: the phase-B fast-path
    skip REQUIRES a current-fp k100 vc-parity report — a changed
    --parent-revision (which leaves every other phase-B fp byte-identical)
    or a deleted report blocks the skip; parent/ffr predicates unchanged."""
    argv = ["--phase", "B", "--round", "k100", "--upload", "none"]
    argv += ["--out-root", str(tmp_path / "root")]
    cfg_a = D.build_config(D.parse_args(argv))
    cfg_b = D.build_config(D.parse_args([*argv, "--parent-revision", "deadbeef" * 5]))
    cells = ["query", "user_fact"]
    _pb_completion_state(cfg_a, cells)
    assert D._pb_skip_ok(cfg_a, cells) is True
    # every non-k100 phase-B fp is identical across the two pins ...
    assert D._regime_fp(cfg_a, {"phase": "B"}) == D._regime_fp(cfg_b, {"phase": "B"})
    # ... yet the stale parity report blocks cfg_b's fast path
    assert D._pb_skip_ok(cfg_b, cells) is False
    (cfg_a.manifest_dir / "k100_vc_parity.json").unlink()
    assert D._pb_skip_ok(cfg_a, cells) is False


def test_k100_parent_staging_paths_carry_revision(tmp_path, monkeypatch):
    """r1 blocker k100-parent-revision-cache-unkeyed: analysis parent_pin
    staging AND the intrusion audit's parent staging key the staged PATH on
    the pinned revision, so bytes staged under revision A can never satisfy a
    run pinned to revision B."""
    import issue2564_intrusion_audit_k100 as K2

    calls: list = []

    def fake_stage(repo, path_in_repo, target, *, revision=None):
        calls.append((path_in_repo, str(target), revision))
        Path(target).parent.mkdir(parents=True, exist_ok=True)
        Path(target).write_text("x")
        return Path(target)

    monkeypatch.setattr("explore_persona_space.orchestrate.hub.stage_hub_file", fake_stage)
    monkeypatch.setattr(K2, "stage_hub_file", fake_stage)
    rev_a, rev_b = A.K100_PARENT_REVISION_DEFAULT, "deadbeef" * 5
    cfg = _replace(_k100_world(tmp_path).cfg, stage_dir=tmp_path / "stage")
    rel = "manifests/bank2564_manifest.json"
    p_a = A.resolve_input(cfg, rel, source="parent")
    assert rev_a in str(p_a) and calls[-1][2] == rev_a
    p_b = A.resolve_input(_replace(cfg, parent_revision=rev_b), rel, source="parent")
    assert rev_b in str(p_b) and calls[-1][2] == rev_b
    assert p_a != p_b  # rev-A bytes never satisfy a rev-B run
    n = len(calls)
    args = SimpleNamespace(pin_rev=rev_b, smoke=False)
    paths = K2._stage(args, tmp_path / "audit_stage")
    for cell in K2.K100_CELLS:
        assert rev_b in str(paths[("parent", cell)])
    parent_calls = [c for c in calls[n:] if "/anchors/" in c[0] and c[2] == rev_b]
    assert len(parent_calls) == len(K2.K100_CELLS)


def test_k100_pod_parent_vc_staging_keyed_on_revision(tmp_path, monkeypatch):
    """r2 blocker k100-parent-revision-cache-unkeyed (final leg): the POD-side
    v_C parity gate stages the parent bank under parent_vc/<revision>/, so a
    same-out-root run pinned to revision B receives a DISTINCT path holding
    B's bytes. The staging fake honors the REAL stage_hub_file existing-target
    short-circuit (hub.py: an existing target returns WITHOUT a download) —
    under the pre-fix revision-independent target, the second call would
    silently reuse A's cached bytes and record no download."""
    argv = ["--phase", "B", "--round", "k100", "--upload", "none"]
    argv += ["--out-root", str(tmp_path / "root")]
    cfg_a = D.build_config(D.parse_args(argv))
    rev_b = "deadbeef" * 5
    cfg_b = D.build_config(D.parse_args([*argv, "--parent-revision", rev_b]))
    rev_a = cfg_a.parent_revision
    assert rev_a and rev_a != rev_b

    cid = "user_fact::n01::c01"
    rng = np.random.default_rng(0)
    vc = torch.tensor(rng.normal(size=(1, len(cfg_a.layers), 8)), dtype=torch.float32)
    cfg_a.vc_dir.mkdir(parents=True, exist_ok=True)
    cfg_a.manifest_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {"layers": list(cfg_a.layers), "context_ids": [cid], "vc": vc},
        cfg_a.vc_dir / "vc2564_bank.pt",
    )

    downloads: list[tuple[str, str, str | None]] = []

    def fake_stage(repo, path_in_repo, target, *, revision=None):
        target = Path(target)
        if target.exists():  # the REAL helper's existing-target short-circuit
            return target
        downloads.append((path_in_repo, str(target), revision))
        target.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"layers": list(cfg_a.layers), "context_ids": [cid], "vc": vc, "rev": revision},
            target,
        )
        return target

    monkeypatch.setattr("explore_persona_space.orchestrate.hub.stage_hub_file", fake_stage)
    D._k100_vc_parity(cfg_a, [{"id": cid}])
    assert len(downloads) == 1 and downloads[0][2] == rev_a and rev_a in downloads[0][1]
    # same out-root, new pin: the gate re-runs (fp mismatch) and must download
    # B's bytes to a DISTINCT revision-keyed path — never reuse A's file.
    D._k100_vc_parity(cfg_b, [{"id": cid}])
    assert len(downloads) == 2 and downloads[1][2] == rev_b and rev_b in downloads[1][1]
    path_a, path_b = Path(downloads[0][1]), Path(downloads[1][1])
    assert path_a != path_b
    # self-produced revision-tagged bundle (torch>=2.6 weights_only convention)
    assert torch.load(path_b, map_location="cpu", weights_only=False)["rev"] == rev_b
    rep = json.loads((cfg_b.manifest_dir / "k100_vc_parity.json").read_text())
    assert rep["parent_revision"] == rev_b and rep["verdict"] == "pass"


def _mutate_store(w, rel: str, source: str, mutate_fn) -> None:
    p = w.paths[(rel, source)]
    store = torch.load(p, weights_only=False)
    mutate_fn(store)
    torch.save(store, p)


def test_k100_load_stores_raises_on_duplicate_round_draw(k100_world):
    """r1 blocker k100-draw-grid-completeness: one context's draw-99 row
    re-labeled as a SECOND draw-98 row (same row count, same global draw
    range — the pre-fix checks all pass) must RAISE in the VA loader."""
    w = k100_world

    def mut(store):
        for rec in store["index"]:
            if rec["context_id"] == "query::E::c01" and rec["draw"] == 99:
                rec["draw"] = 98
                return
        raise AssertionError("fixture row not found")

    _mutate_store(w, "analysis_tensors/va2564/va2564_query.pt", "round", mut)
    with pytest.raises(AssertionError, match="duplicate \\(context_id, draw\\)"):
        A.load_stores_k100(w.cfg, w.bank)


def test_k100_load_stores_raises_on_missing_percontext_draw(k100_world):
    """A2b: a context missing ONE fresh draw that another context still
    carries (global draw set intact) must RAISE per-context coverage."""
    w = k100_world

    def mut(store):
        j = next(
            i
            for i, rec in enumerate(store["index"])
            if rec["context_id"] == "query::E::c01" and rec["draw"] == 99
        )
        del store["index"][j]
        keep = [i for i in range(store["va_tail_incl"].shape[0]) if i != j]
        store["va_tail_incl"] = store["va_tail_incl"][keep]
        store["va_span"] = store["va_span"][keep]

    _mutate_store(w, "analysis_tensors/va2564/va2564_query.pt", "round", mut)
    with pytest.raises(AssertionError, match="A2b per-context grid"):
        A.load_stores_k100(w.cfg, w.bank)


def _mutate_perdraw(w, source: str, mutate_fn) -> None:
    p = w.paths[("analysis_tensors/embeddings_qwen3_8b/perdraw_anchors.npz", source)]
    with np.load(p, allow_pickle=False) as z:
        ids = [str(x) for x in z["context_ids"].tolist()]
        draws = z["draws"].copy()
        emb = z["emb"].copy()
    ids, draws, emb = mutate_fn(ids, draws, emb)
    np.savez(p, context_ids=np.array(ids), draws=np.asarray(draws, dtype=np.int64), emb=emb)


def test_k100_load_stores_raises_on_duplicate_perdraw_embedding_row(k100_world):
    """r1 blocker k100-draw-grid-completeness: a duplicated (context, draw)
    per-draw embedding row (same row count, same global range) must RAISE —
    never silently shift the K-pooled mean."""
    w = k100_world

    def mut(ids, draws, emb):
        j = next(
            i
            for i, (c, d) in enumerate(zip(ids, draws, strict=True))
            if c == "query::E::c01" and d == 99
        )
        draws[j] = 98
        return ids, draws, emb

    _mutate_perdraw(w, "round", mut)
    with pytest.raises(AssertionError, match="duplicate \\(context, draw\\) embedding"):
        A.load_stores_k100(w.cfg, w.bank)


def test_k100_load_stores_raises_on_missing_perdraw_embedding_row(k100_world):
    """The per-draw embedding key set must EQUAL the non-empty anchor-row key
    set — a dropped row is named missing, never averaged around."""
    w = k100_world

    def mut(ids, draws, emb):
        j = next(
            i
            for i, (c, d) in enumerate(zip(ids, draws, strict=True))
            if c == "query::E::c01" and d == 99
        )
        keep = [i for i in range(len(ids)) if i != j]
        return [ids[i] for i in keep], draws[keep], emb[keep]

    _mutate_perdraw(w, "round", mut)
    with pytest.raises(AssertionError, match="key set != non-empty anchor rows"):
        A.load_stores_k100(w.cfg, w.bank)


def test_k100_fire_recompute_raises_on_duplicate_anchor_key(k100_world, tmp_path, monkeypatch):
    """r1 blocker k100-draw-grid-completeness: a duplicate (context_id, draw)
    anchor row RAISES in the fire recompute — never last-wins into the
    1,200-check denominator."""
    w = k100_world
    par = tmp_path / "anchors_user_fact.jsonl"
    rows = [
        {"context_id": "user_fact::n01::c01", "draw": 0, "text": "a"},
        {"context_id": "user_fact::n01::c01", "draw": 0, "text": "b"},
    ]
    par.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    monkeypatch.setattr(A, "resolve_input", lambda cfg, rel, *, source="round": par)
    st = SimpleNamespace(carriers=["c01"])
    with pytest.raises(RuntimeError, match="duplicate anchor \\(context_id, draw\\)"):
        A.k100_fire_recompute(w.cfg, st, {"value_rows": {}, "axis_rows": {}})


def test_generate_cell_rejects_chunk_with_duplicate_draw(tmp_path, monkeypatch):
    """Producer side of r1 blocker k100-draw-grid-completeness: a prior chunk
    whose rows match in COUNT and context-id SET but carry a duplicated draw
    (one draw missing) is NOT adopted — the chunk regenerates."""
    cfg = _gen_cfg(tmp_path)
    monkeypatch.setattr(D.BK, "context_token_ids", lambda tok, ctx: [1, 2, 3])
    calls: list = []
    monkeypatch.setattr(D, "generate_batch", _fake_generate_batch(calls))
    cfg.anchors_dir.mkdir(parents=True, exist_ok=True)
    part = cfg.anchors_dir / "anchors_register.max128.partial"
    fp = D._regime_fp(cfg, {"phase": "A", "cell": "register", "max_new_call": 128})
    ctxs = _fake_ctxs(3)
    bad_rows = [
        # chunk 0 = contexts c01+c02, draws {0, 1}; c02 carries draw 0 TWICE
        {"chunk": 0, "context_id": "register::v1::c01", "draw": 0},
        {"chunk": 0, "context_id": "register::v1::c01", "draw": 1},
        {"chunk": 0, "context_id": "register::v1::c02", "draw": 0},
        {"chunk": 0, "context_id": "register::v1::c02", "draw": 0},
    ]
    part.write_text(
        "\n".join(
            [json.dumps({"partial_header": 1, "regime_fp": fp})] + [json.dumps(r) for r in bad_rows]
        )
        + "\n"
    )
    rows = D._generate_cell(
        cfg, object(), FakeTokenizer(), [7], "register", ctxs, {"evaluated": True}, 128
    )
    assert len(calls) == 2  # count + ctx set matched, draw composition did not
    assert len(rows) == 3 * cfg.draws
    keys = [(r["context_id"], r["draw"]) for r in rows]
    assert len(set(keys)) == len(keys)  # the regenerated rows carry the full grid


def test_k100_intrusion_scan_shard_raises_on_duplicate(tmp_path):
    """r1 blocker k100-draw-grid-completeness (audit leg): a duplicate
    (context_id, draw) shard row RAISES — totals can no longer pass with a
    duplicate replacing a missing row."""
    import issue2564_intrusion_audit_k100 as K2

    p = tmp_path / "anchors_user_fact.jsonl"
    rows = [
        {"context_id": "user_fact::n01::c01", "draw": 11, "text": "hello"},
        {"context_id": "user_fact::n01::c01", "draw": 11, "text": "world"},
    ]
    p.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    with pytest.raises(RuntimeError, match="duplicate \\(context_id, draw\\)"):
        K2.scan_shard(p, expect_parent=False)


def test_k100_intrusion_analysis_parity_production_gates(tmp_path):
    """r1 blocker k100-intrusion-parity-failopen: missing (production),
    smoke-mode-doc-in-production, duplicated, truncated, and
    verdict-mismatched manipulation-check documents ALL raise; smoke may skip
    a missing file; the matching document passes."""
    import issue2564_intrusion_audit_k100 as K2

    slots = {
        "user_fact::n01": {"verdict_orig": "fired"},
        "user_fact::n02": {"verdict_orig": "not_fired"},
    }
    mp = tmp_path / "manipulation_check_k100.json"

    def _doc(rows, smoke=False):
        return {"meta": {"smoke": smoke}, "value_rows": rows}

    row1 = {"axis": "user_fact", "value_id": "n01", "verdict": "fired"}
    row2 = {"axis": "user_fact", "value_id": "n02", "verdict": "not_fired"}
    with pytest.raises(RuntimeError, match="REQUIRES as-scored parity"):
        K2.analysis_parity_check(mp, slots, smoke=False)
    assert K2.analysis_parity_check(mp, slots, smoke=True)["status"] == "skipped_smoke_missing"
    mp.write_text(json.dumps(_doc([row1, row2], smoke=True)))
    with pytest.raises(RuntimeError, match="SMOKE-mode"):
        K2.analysis_parity_check(mp, slots, smoke=False)
    mp.write_text(json.dumps(_doc([row1, row1])))
    with pytest.raises(RuntimeError, match="duplicate value rows"):
        K2.analysis_parity_check(mp, slots, smoke=False)
    mp.write_text(json.dumps(_doc([row1])))  # truncated: subset iteration must not pass
    with pytest.raises(RuntimeError, match="slot-set mismatch"):
        K2.analysis_parity_check(mp, slots, smoke=False)
    extra = {"axis": "user_fact", "value_id": "n99", "verdict": "fired"}
    mp.write_text(json.dumps(_doc([row1, row2, extra])))
    with pytest.raises(RuntimeError, match="slot-set mismatch"):
        K2.analysis_parity_check(mp, slots, smoke=False)
    mp.write_text(json.dumps(_doc([row1, {**row2, "verdict": "fired"}])))
    with pytest.raises(RuntimeError, match="as-scored fire parity"):
        K2.analysis_parity_check(mp, slots, smoke=False)
    mp.write_text(json.dumps(_doc([row1, row2])))
    out = K2.analysis_parity_check(mp, slots, smoke=False)
    assert out["status"] == "pass" and out["n_slots"] == 2


def _swap_pair(st, pair_id: str = "p1") -> A.PairArrays:
    """Single user_fact SWAP-class pair (the axis-view primary class)."""
    return A.PairArrays(
        ids=[pair_id],
        cls=["swap"],
        axis=["user_fact"],
        value_a=["n01"],
        value_b=["n02"],
        carrier_str=["c01"],
        a=np.array([st.row_of["user_fact::n01::c01"]], dtype=np.int64),
        b=np.array([st.row_of["user_fact::n02::c01"]], dtype=np.int64),
        ca=np.array([0], dtype=np.int64),
        cb=np.array([0], dtype=np.int64),
        dyad=np.array([False]),
        changed=np.array([1], dtype=np.int64),
        orientation=["n01->n02"],
        n=1,
    )


def test_k100_bridge_gate_perpair_and_headline_mismatch_raise(k100_world, tmp_path, monkeypatch):
    """r1 nit k100-bridge-gate-regression-pin: k100_bridge_gate invoked
    DIRECTLY on synthetic ridge/perpair fixtures — matching committed values
    PASS; a 2e-6 perturbation of a committed per-pair value OR of a headline
    target raises RuntimeError at K100_BRIDGE_TOL (production mode)."""
    w = k100_world
    st = _replace(A.load_stores_k100(w.cfg, w.bank), carriers=["c01"])
    pa = _swap_pair(st)
    ridge_p = tmp_path / "ridge_779.pt"
    rng = np.random.default_rng(7)
    torch.save(
        {
            "kind": "ridge",
            "xmu": torch.zeros(4),
            "xsd": torch.ones(4),
            "ymu": torch.zeros(4),
            "W": torch.tensor(rng.standard_normal((4, 4)), dtype=torch.float32),
        },
        ridge_p,
    )
    committed = tmp_path / "committed"
    committed.mkdir()
    cfg = _replace(w.cfg, ridge_779=ridge_p, parent_delta=committed / "minpair_delta.json")

    # expected values via the SAME public helpers the gate dispatches
    payload = torch.load(ridge_p, weights_only=False)
    mean_p = A._k100_parent_only_tail_mean(st, A.PRIMARY_LAYER)
    obs = mean_p[pa.a] - mean_p[pa.b]
    mapped = N1M.apply_map(payload, st.vc[A.PRIMARY_LAYER], torch.device("cpu"))
    cos779 = float(A.rowwise_cos(mapped[pa.a] - mapped[pa.b], obs)[0])
    cosid = float(
        A.rowwise_cos(st.vc[A.PRIMARY_LAYER][pa.a] - st.vc[A.PRIMARY_LAYER][pa.b], obs)[0]
    )
    bridge_st = _replace(st, tail_draws=st.tail_draws[:, :10], draw_valid=st.draw_valid[:, :10])

    def pg(rng2, n_ctx, k_max):
        return rng2.random((st.n_parent_ctx_total, k_max))[st.parent_rows]

    r10 = float(A.split_half_stats(bridge_st, pa, cfg.n_splits, scores_fn=pg)["r_full"][0])
    good_row = {"pair_id": "p1", "r10": r10, "cos": {"arm_779ce": cos779, "arm_iddelta": cosid}}
    (committed / "perpair.jsonl").write_text(json.dumps(good_row) + "\n")
    targets = {"user_fact": {"mean_cos_headline": cos779, "r10_mean": r10}}
    monkeypatch.setattr(A, "K100_BRIDGE_TARGETS", targets)
    fire_parent = A.fire_tables_from_doc({})

    report = A.k100_bridge_gate(cfg, w.bank, st, pa, fire_parent)
    assert report["verdict"] == "pass" and report["headline_parity"]["ok"] is True

    bad_row = dict(good_row)
    bad_row["r10"] = r10 + 2e-6
    (committed / "perpair.jsonl").write_text(json.dumps(bad_row) + "\n")
    with pytest.raises(RuntimeError, match="PER-PAIR parity FAILED"):
        A.k100_bridge_gate(cfg, w.bank, st, pa, fire_parent)

    (committed / "perpair.jsonl").write_text(json.dumps(good_row) + "\n")
    monkeypatch.setattr(
        A,
        "K100_BRIDGE_TARGETS",
        {"user_fact": {"mean_cos_headline": cos779 + 2e-6, "r10_mean": r10}},
    )
    with pytest.raises(RuntimeError, match="HEADLINE parity FAILED"):
        A.k100_bridge_gate(cfg, w.bank, st, pa, fire_parent)


def test_k100_provenance_check_a_rejects_malformed_stale_or_nonpass(
    k100_world, tmp_path, monkeypatch
):
    """r1 blocker k100-parent-revision-cache-unkeyed (analysis leg): the
    production analysis REQUIRES a well-formed, non-demoted PASS vc-parity
    report whose parent_revision matches --parent-revision — malformed,
    demoted/skipped, and stale-revision reports all raise (never read as
    'not a literal fail')."""
    w = k100_world
    rep_path = tmp_path / "k100_vc_parity.json"
    monkeypatch.setattr(A, "resolve_input", lambda cfg, rel, *, source="round": rep_path)
    rep_path.write_text(json.dumps({"gate": "something_else"}))
    with pytest.raises(RuntimeError, match="malformed"):
        A.k100_provenance_checks(w.cfg, None, None)  # (a) raises before (b) touches stores
    rep_path.write_text(
        json.dumps({"gate": "k100_vc_parity", "verdict": "skipped", "demoted": True})
    )
    with pytest.raises(RuntimeError, match="non-demoted PASS"):
        A.k100_provenance_checks(w.cfg, None, None)
    rep_path.write_text(
        json.dumps(
            {
                "gate": "k100_vc_parity",
                "verdict": "pass",
                "demoted": False,
                "parent_revision": "0000000000000000000000000000000000000000",
            }
        )
    )
    with pytest.raises(RuntimeError, match="STALE"):
        A.k100_provenance_checks(w.cfg, None, None)
