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
