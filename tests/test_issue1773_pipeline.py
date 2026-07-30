"""Issue #1773 pipeline unit tests (plan § Test scope; shipped with the build).

Covers: window-extraction BPE-seam (peak at answer edge), quantile-bin
selection determinism, majority-vote + drop semantics (ties -> unresolved,
REFUSAL -> content drop, transport split), label-permutation cache-key
differentiation, lattice application on synthetic scorecards, annotation-sheet
composition (>= 40 identity positives — the Statistics Must-Fix pin), and a
tiny-real CPU e2e of the evidence builder (2 synthetic chunks + a from-config
2-layer same-arch model + a small random BatchTopK state dict with the SAME
key set — the #906 tiny-real pattern: real tokenizer, real capture path, fake
only GPU-scale weights + the Hub boundary).
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))
sys.path.insert(0, str(REPO / "src"))

import issue1773_common as CM  # noqa: E402

# ── majority vote + drop semantics ───────────────────────────────────────────


def test_majority_vote_majority_and_floor():
    assert CM.majority_vote(["yes", "yes", "yes", "no", "no"]) == "yes"
    # 2-2 tie with a drop -> unresolved
    assert CM.majority_vote(["yes", "yes", "no", "no"]) == "unresolved"
    # below MAJORITY_FLOOR even when unanimous among survivors
    assert CM.majority_vote(["yes", "yes"]) == "unresolved"
    assert CM.majority_vote([]) == "unresolved"


def test_validate_axis_label_drop_never_coerce():
    # REFUSAL / malformed / out-of-set -> None (content drop), never coerced
    assert CM.validate_axis_label("REFUSAL", "interpretable") is None
    assert CM.validate_axis_label(None, "interpretable") is None
    assert CM.validate_axis_label({"label": "persona"}, "speaker_property") is None
    assert CM.validate_axis_label({"label": 3}, "interpretable") is None
    # normalizable variants ARE accepted (case/space/hyphen)
    assert (
        CM.validate_axis_label({"label": "Identity-Disposition"}, "speaker_property")
        == "identity_disposition"
    )


def test_transport_vs_content_split_in_aggregation():
    """Transport-class error dicts count as transport_losses, content errors as
    content_drops (llm-judging rule 24 split); the retry path upstream is
    dispatch-owned — here we pin the tally classification only."""
    import issue1773_describe_axes as DA

    feat = 7
    items = [
        (CM.axis_custom_id(feat, "interpretable", d), f"feat:{feat}:interpretable", "", "u")
        for d in range(5)
    ]
    results = {
        items[0][0]: {"label": "yes"},
        items[1][0]: {"label": "yes"},
        items[2][0]: {"label": "yes"},
        # transport-class error dict (structural flag per #1313)
        items[3][0]: {"error": True, "transport": True, "reasoning": "529 exhausted"},
        # content-class error dict
        items[4][0]: {"error": True, "reasoning": "parse_error"},
    }
    rows, kappa = DA.aggregate_axes(items, results)
    t = kappa["interpretable"]["drop_report"]
    assert t["transport_losses"] == 1
    assert t["content_drops"] == 1
    assert t["ok"] == 3
    row = next(r for r in rows if r["axis"] == "interpretable")
    assert row["label"] == "yes" and row["n_surviving"] == 3


def test_speaker_property_labels_verbatim_1092():
    assert CM.AXES["speaker_property"] == (
        "language",
        "register_style",
        "identity_disposition",
        "none",
        "unclear",
    )


# ── label permutation: cache-key differentiation ─────────────────────────────


def test_label_permutation_deterministic_and_draw_differentiating():
    p0 = CM.label_permutation(42, "speaker_property", 0)
    assert p0 == CM.label_permutation(42, "speaker_property", 0)  # deterministic
    assert sorted(p0) == sorted(CM.AXES["speaker_property"])
    perms = {tuple(CM.label_permutation(42, "speaker_property", d)) for d in range(5)}
    assert len(perms) >= 2, "permutation must vary across draws"
    packet = {
        "feat_id": 42,
        "ex_pos": [{"text_marked": "a <<b>> c", "text_plain": "a b c", "bin": 9, "ci": 1}],
        "ex_neg": [{"text_marked": "x", "text_plain": "x", "ci": 2}],
        "near": [],
        "out": None,
    }
    msgs = {CM.build_axis_user_msg("speaker_property", packet, "d", d) for d in range(5)}
    assert len(msgs) >= 2, "user messages must differ across draws (rubric-keyed cache rule 22)"


# ── Fleiss kappa (varying-n) ─────────────────────────────────────────────────


def test_fleiss_kappa_varying_n():
    perfect = [["yes"] * 5, ["no"] * 5, ["yes"] * 4, ["no"] * 3]
    out = CM.fleiss_kappa_varying_n(perfect, ("yes", "no"))
    assert out["kappa"] == pytest.approx(1.0)
    assert out["n_items"] == 4
    lt2 = [["yes"], ["yes", "no", "yes", "no"]]
    out2 = CM.fleiss_kappa_varying_n(lt2, ("yes", "no"))
    assert out2["n_excluded_lt2"] == 1 and out2["n_items"] == 1


# ── lattice on synthetic scorecards ──────────────────────────────────────────


def test_lattice_synthetic():
    good = {
        "detection": 0.72,
        "fuzzing": 0.71,
        "discrimination": 0.55,
        "kappa": 0.65,
        "shuffled_detection": 0.52,
    }
    assert CM.apply_lattice(good) == "TRUSTWORTHY"
    for k, bad_v in (
        ("detection", 0.69),
        ("fuzzing", 0.60),
        ("discrimination", 0.45),
        ("kappa", 0.5),
        ("shuffled_detection", 0.60),
    ):
        row = dict(good)
        row[k] = bad_v
        assert CM.apply_lattice(row) == "SEARCH-INDEX-ONLY", k
    nan_row = dict(good, kappa=float("nan"))
    assert CM.apply_lattice(nan_row) == "SEARCH-INDEX-ONLY"


# ── quantile-bin selection determinism ───────────────────────────────────────


def test_stratified_bin_draw_deterministic_and_split():
    import issue1773_evidence_builder as EB

    n = 400
    rng0 = np.random.default_rng(1)
    rows = rng0.permutation(n).astype(np.int64)
    cis = rows + 10_000
    vals = rng0.random(n).astype(np.float32)
    a = EB.stratified_bin_draw(rows, cis, vals, np.random.default_rng(7))
    b = EB.stratified_bin_draw(rows, cis, vals, np.random.default_rng(7))
    assert a == b, "seeded draw must be deterministic"
    picked, borrows = a
    assert len(picked) == CM.N_ACT_BINS * CM.ACT_PER_BIN
    assert borrows == 0
    for bn in range(CM.N_ACT_BINS):
        grp = [p for p in picked if p["bin"] == bn]
        assert len(grp) == CM.ACT_PER_BIN
        assert sum(1 for p in grp if p["split"] == 0) == CM.ACT_EVIDENCE_PER_BIN
    rows_seen = [p["row"] for p in picked]
    assert len(set(rows_seen)) == len(rows_seen), "dedup by row"


def test_stratified_bin_draw_thin_bins_borrow():
    import issue1773_evidence_builder as EB

    n = 25  # far fewer than 60: every bin thin
    rows = np.arange(n, dtype=np.int64)
    vals = np.linspace(0, 1, n).astype(np.float32)
    picked, borrows = EB.stratified_bin_draw(rows, rows, vals, np.random.default_rng(3))
    assert len(picked) == n, "all candidates used when pool < target"
    assert borrows > 0, "thin bins must borrow (recorded)"


# ── annotation sheet composition (Statistics Must-Fix) ───────────────────────


def _mk_sheet_fixture(tmp_path: Path, n_ident: int = 60, n_feat: int = 300):
    """Synthetic labels/descriptions/packets/phase0 for the sheet builder."""
    out_root = tmp_path / "eval"
    ev = tmp_path / "evidence"
    p0d = tmp_path / "phase0"
    (out_root / "labels").mkdir(parents=True)
    (ev / "evidence_manifests").mkdir(parents=True)
    p0d.mkdir(parents=True)
    rng = np.random.default_rng(0)
    fid = np.arange(n_feat, dtype=np.int64)
    lab_rows, desc_rows, packets = [], [], []
    for f in fid:
        sp = "identity_disposition" if f < n_ident else "none"
        for axis in CM.AXES:
            lab = sp if axis == "speaker_property" else CM.AXES[axis][0]
            lab_rows.append(
                {
                    "feat_id": int(f),
                    "axis": axis,
                    "label": lab,
                    "labels_surviving": [lab] * 5,
                    "n_surviving": 5,
                    "n_launched": 5,
                }
            )
        desc_rows.append({"feat_id": int(f), "description": f"feature {f}", "confidence": 80})
        packets.append(
            {
                "feat_id": int(f),
                "ex_pos": [
                    {"text_marked": f"w{f} <<x>>", "text_plain": f"w{f} x", "bin": 9, "ci": int(f)}
                ]
                * 5,
                "ex_neg": [],
                "near": [],
                "out": None,
            }
        )
    for name, rows in (("axis_labels", lab_rows), ("descriptions", desc_rows)):
        with (out_root / "labels" / f"{name}.jsonl").open("w") as fh:
            for r in rows:
                fh.write(json.dumps(r) + "\n")
    with (ev / "evidence_manifests" / "evidence.shard00.jsonl").open("w") as fh:
        for r in packets:
            fh.write(json.dumps(r) + "\n")
    np.savez(
        p0d / "phase0_arrays.npz",
        feat_ids=fid,
        activity=rng.random(n_feat),
        rb_cos=rng.random((3, n_feat)),
        neighbor_idx=np.zeros((n_feat, 8), np.int64),
        r2=rng.random(n_feat),
        persist_answer=rng.random(n_feat),
        side_ratio=rng.random(n_feat),
    )
    return out_root, ev, p0d


def test_annotation_sheet_min_identity_positives(tmp_path):
    import issue1773_validate as V

    out_root, ev, p0d = _mk_sheet_fixture(tmp_path)
    args = type(
        "A",
        (),
        {
            "out_root": out_root,
            "evidence_dir": ev,
            "phase0_dir": p0d,
            "artifacts_dir": tmp_path / "artifacts",
        },
    )()
    rc, meta = V.build_annotation_sheet(args)
    assert rc == 0
    assert meta["n_identity_positive_rows"] >= V.SHEET_MIN_IDENTITY
    # the emitted sheet itself carries >=40 identity positives (per answer key)
    key = tmp_path / "artifacts" / "annotation_sheet_v1.answer_key.jsonl"
    n_pos = sum(
        1
        for r in (json.loads(x) for x in key.read_text().split("\n") if x.strip())
        if r["axis"] == "speaker_property" and r["judge_label"] == "identity_disposition"
    )
    assert n_pos >= V.SHEET_MIN_IDENTITY
    # judge labels withheld from the sheet rows (human_label empty; no judge_label key)
    sheet = tmp_path / "artifacts" / "annotation_sheet_v1.jsonl"
    first = json.loads(sheet.read_text().split("\n")[0])
    assert "judge_label" not in first and first["human_label"] is None


# ── tiny-real CPU e2e of the evidence builder (Pass B; #906 pattern) ─────────

TINY_HID = 64
TINY_DICT = 256
TINY_LAYER = 1


def _tiny_sae_state(tmp_path: Path) -> Path:
    """Small random BatchTopK state dict with the SAME key set as the pinned
    artifact ({b_dec, k, threshold, decoder.weight, encoder.weight,
    encoder.bias} — issue1482_sae.EXPECTED_KEYS)."""
    import torch

    torch.manual_seed(0)
    sd = {
        "b_dec": torch.zeros(TINY_HID),
        "k": torch.tensor(64, dtype=torch.int32),
        "threshold": torch.tensor(0.0),
        "decoder.weight": torch.randn(TINY_HID, TINY_DICT) * 0.1,
        "encoder.weight": torch.randn(TINY_DICT, TINY_HID) * 0.1,
        "encoder.bias": torch.zeros(TINY_DICT),
    }
    import issue1482_sae as S

    assert set(sd) == S.EXPECTED_KEYS
    p = tmp_path / "tiny_ae.pt"
    torch.save(sd, p)
    return p


@pytest.mark.slow
def test_tiny_real_cpu_e2e_evidence_builder(tmp_path):
    """Pass B end-to-end on CPU: real Qwen tokenizer, real _tokenize_row /
    _batched_capture / BatchTopKSAE.encode path over 2 synthetic chunks and a
    from-config 2-layer same-arch model; asserts window records exist with
    <<marks>>, an answer-edge peak stays in-window (BPE-seam target), the
    non-activating verifier ran, and the resume predicate skips done chunks."""
    import issue1773_evidence_builder as EB

    chunks = tmp_path / "chunks"
    chunks.mkdir()
    rows0 = [
        {"ci": 1, "prompt": "What is the capital of France?", "response": "Paris. It is lovely."},
        {"ci": 2, "prompt": "Thanks.", "response": "You're welcome!"},
    ]
    rows1 = [
        {"ci": 3, "prompt": "Name a color.", "response": "Blue is a calm color."},
    ]
    (chunks / "shard00_chunk0000.json").write_text(json.dumps({"rows": rows0}))
    (chunks / "shard00_chunk0001.json").write_text(json.dumps({"rows": rows1}))
    sae_path = _tiny_sae_state(tmp_path)

    # discover activating (row, feat) pairs via the builder's own capture path
    import issue1482_error_analysis as EA
    from transformers import AutoTokenizer, Qwen2Config, Qwen2ForCausalLM

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    import torch

    torch.manual_seed(0)
    cfg = Qwen2Config(
        vocab_size=len(tok),
        hidden_size=TINY_HID,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        tie_word_embeddings=True,
    )
    model = Qwen2ForCausalLM(cfg)
    model.eval()

    import issue1482_sae as S

    sae = S.BatchTopKSAE(
        torch.load(sae_path, weights_only=True), k=64, act_dim=TINY_HID, dict_size=TINY_DICT
    )
    prefix_chars = EA._prefix_char_len(tok)
    tk = EA._tokenize_row(tok, rows0[0]["prompt"], rows0[0]["response"], prefix_chars)
    full_ids, _pe, context_end, n_ans, _seam = tk
    caps = EA._batched_capture(
        model, tok, [(0, 1, full_ids, _pe, context_end, n_ans, _seam)], (TINY_LAYER,), "cpu"
    )
    f_ans = sae.encode(caps[0][TINY_LAYER][context_end + 1 :])
    active = torch.nonzero(f_ans.max(0).values > 0).squeeze(-1)
    assert len(active) >= 2, "tiny SAE should activate some features"
    feat_a, feat_b = int(active[0]), int(active[1])

    # hand a minimal selection manifest + inverted index to Pass B
    sel = tmp_path / "selection"
    sel.mkdir()
    inv = np.array(
        [
            [0, 1, feat_a, 0, 9, 0, 0],  # act evidence (row 0 = ci 1)
            [1, 2, feat_a, 1, -1, -1, 0],  # nonact candidate (row 1 = ci 2)
            [2, 3, feat_b, 0, 5, 1, 0],  # act holdout (row 2 = ci 3, chunk 2)
        ],
        dtype=np.int64,
    )
    np.savez(
        sel / "inverted_index.npz",
        row=inv[:, 0],
        ci=inv[:, 1],
        feat=inv[:, 2],
        kind=inv[:, 3],
        bin=inv[:, 4],
        split=inv[:, 5],
        order=inv[:, 6],
    )
    rng = np.random.default_rng(0)
    dirs = rng.standard_normal((CM.N_RANDOM_DIRECTIONS, TINY_HID)).astype(np.float32)
    np.savez(sel / "random_directions.npz", directions=dirs)

    out_dir = tmp_path / "raw_windows"
    import argparse

    args = argparse.Namespace(
        pass_name="windows",
        import_check=False,
        store=CM.STORE_DEFAULT,
        selection_dir=sel,
        out_dir=out_dir,
        evidence_dir=tmp_path / "evidence",
        phase0_dir=tmp_path / "phase0",
        scratch=tmp_path / "scratch",
        seed=CM.SEED,
        worker=0,
        n_workers=1,
        gpu_id=0,
        device="cpu",
        gen_batch=2,
        layer=TINY_LAYER,
        k=64,
        act_dim=TINY_HID,
        dict_size=TINY_DICT,
        sae_state=sae_path,
        tiny_model=False,
        local_chunks=chunks,
        max_chunks=0,
        max_shards=0,
        feature_limit=0,
        pilot=False,
        upload_every=20,
        upload_only=False,
        no_upload=True,
        fetch_missing=False,
        no_resume=False,
    )
    # Fake ONLY the GPU-scale-weights boundary with a SIGNATURE-CONFORMANT
    # replacement (a def mirroring _load_model_tok's (args) -> (model, tok)
    # contract — never a bare Mock); every other seam runs the real body.
    import unittest.mock as um

    def _tiny_load_model_tok(ns):
        return model, tok

    with um.patch.object(EA, "_load_model_tok", new=_tiny_load_model_tok):
        rc = EB.pass_windows(args)
    assert rc == 0
    w0 = list(CM.iter_jsonl(out_dir / "windows_shard00_chunk0000.jsonl"))
    w1 = list(CM.iter_jsonl(out_dir / "windows_shard00_chunk0001.jsonl"))
    acts0 = [r for r in w0 if r["kind"] == "act"]
    nonacts0 = [r for r in w0 if r["kind"] == "nonact"]
    assert acts0 and nonacts0 and w1
    a = acts0[0]
    assert "<<" in a["window"]["text_marked"] and ">>" in a["window"]["text_marked"]
    # BPE-seam / answer-edge: short answers force the peak near the edge; the
    # window must stay inside [ans_start, len(full_ids)) and non-empty
    assert a["window"]["token_hi"] > a["window"]["token_lo"]
    assert a["window"]["peak_pos"] >= a["window"]["token_lo"]
    assert "verify_failed" in nonacts0[0]
    # resume predicate: a second invocation skips all done chunks
    with um.patch.object(EA, "_load_model_tok", new=_tiny_load_model_tok):
        rc2 = EB.pass_windows(args)
    assert rc2 == 0

    # --pilot re-run over an already-done pilot chunk (r1 concern
    # passb-pilot-resume-nameerror): the resume-skip leaves the chunk loop
    # without processing any chunk; the pilot report must not NameError on
    # an unbound `rows` (the same-pod launcher relaunch always re-enters the
    # pilot phase first).
    args_pilot = argparse.Namespace(**{**vars(args), "pilot": True})
    with um.patch.object(EA, "_load_model_tok", new=_tiny_load_model_tok):
        rc3 = EB.pass_windows(args_pilot)
    assert rc3 == 0
    assert (out_dir / "pilot_report.json").exists()

    # resume upload reconciliation (r1 review Minor): skipped-but-never-
    # uploaded chunk files are re-queued from ONE scoped Hub listing.
    # Boundary fakes only: _upload_pending (network commit) replaced by a
    # signature-conformant def; hub.verify_repo_paths_uploaded autospec'd.
    from explore_persona_space.orchestrate import hub as _hub

    args_rec = argparse.Namespace(**{**vars(args), "no_upload": False})
    uploads: list[list[str]] = []

    def _fake_upload_pending(pending, out_dir_, args_, t_upload_):
        uploads.append(list(pending))
        pending.clear()

    missing = [f"{CM.HF_PREFIX}/raw_windows/windows_shard00_chunk0000.jsonl"]
    fake_verify = um.create_autospec(_hub.verify_repo_paths_uploaded, return_value=missing)
    with (
        um.patch.object(EA, "_load_model_tok", new=_tiny_load_model_tok),
        um.patch.object(EB, "_upload_pending", new=_fake_upload_pending),
        um.patch.object(_hub, "verify_repo_paths_uploaded", new=fake_verify),
    ):
        rc4 = EB.pass_windows(args_rec)
    assert rc4 == 0
    fake_verify.assert_called_once()
    assert any("windows_shard00_chunk0000.jsonl" in batch for batch in uploads)


def test_window_record_peak_at_answer_edge():
    """Window extraction at the FIRST and LAST answer token (edge clipping)."""
    import issue1773_evidence_builder as EB
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    ids = tok("User: hello\n\nAssistant: Thanks.", add_special_tokens=False)["input_ids"]
    ans_start = max(0, len(ids) - 3)
    vals = np.zeros(len(ids) - ans_start, dtype=np.float32)
    first = EB._window_record(tok, ids, ans_start, ans_start, vals)
    last = EB._window_record(tok, ids, len(ids) - 1, ans_start, vals)
    for rec in (first, last):
        assert rec["token_hi"] > rec["token_lo"]
        assert "<<" in rec["text_marked"]
        assert rec["text_plain"]


# ── custom-id budget ─────────────────────────────────────────────────────────


def test_axis_custom_id_roundtrip_and_budget():
    for feat in (0, 131_071, -200):
        for axis in CM.AXES:
            for d in range(5):
                cid = CM.axis_custom_id(feat, axis, d)
                assert len(cid) <= 53
                assert CM.parse_axis_custom_id(cid) == (feat, axis, d)


# ── launcher lint conformance ────────────────────────────────────────────────


def test_passB_launcher_cvd_pins_workers():
    """Every backgrounded --gpu-id worker launch carries a CVD prefix (the
    --check-dispatcher-cvd-pin contract) and workers are NOT setsid-detached
    (wait must be real — the #1738 chained-waves trap)."""
    sh = (REPO / "scripts" / "issue1773_passB_launch.sh").read_text()
    assert 'CUDA_VISIBLE_DEVICES="$g" nohup uv run python' in sh
    assert "setsid nohup uv run python" not in sh
    # pod-side code never INVOKES task.py (a doc comment naming the rule is fine)
    for line in sh.split("\n"):
        if line.strip().startswith("#"):
            continue
        assert "task.py" not in line, f"pod-side task.py shellout: {line!r}"


# ── crash-fix r3: selection upload + cross-machine staging (#1773) ───────────


def _stage_args(sel_dir: Path):
    import argparse

    return argparse.Namespace(selection_dir=sel_dir)


def test_stage_selection_flat_layout_skip_and_ckpt_filter(tmp_path, monkeypatch):
    """stage_selection places Hub files FLAT at sel_dir/<name> (artifact-reuse
    (h)(iv) — never the verbatim prefix mirror the Hub layout would imply),
    filters the passA_ckpt_* resume checkpoint, and skips the network entirely
    when inverted_index.npz is already present. Hub boundary faked with
    signature-conformant defs (code-style #906); the real end-to-end path is
    covered by the live staging probe recorded in the r3 smoke."""
    from types import SimpleNamespace

    import issue1773_evidence_builder as EB

    from explore_persona_space.orchestrate import hub

    prefix = CM.HF_SELECTION_PREFIX
    listed = [
        f"{prefix}/DONE.json",
        f"{prefix}/inverted_index.npz",
        f"{prefix}/passA_ckpt_deadbeef.npz",
        f"{prefix}/random_directions.npz",
        f"{prefix}/selection.shard00.jsonl",
    ]
    staged: list[Path] = []

    def fake_retry(fn, *, what="", **kw):
        assert "repo_info" in what
        return SimpleNamespace(sha="abc123")

    def fake_list(api, repo_id, path, *, repo_type="model", revision=None):
        assert path == prefix and revision == "abc123"
        return list(listed)

    def fake_stage(
        repo_id,
        path_in_repo,
        target,
        *,
        repo_type="dataset",
        revision=None,
        token=None,
        overwrite=False,
    ):
        target = Path(target)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(b"x")
        staged.append(target)
        return target

    monkeypatch.setattr(hub, "retry_transient", fake_retry)
    monkeypatch.setattr(hub, "list_hf_files_under_path", fake_list)
    monkeypatch.setattr(hub, "stage_hub_file", fake_stage)

    sel = tmp_path / "sel"
    assert EB.stage_selection(_stage_args(sel)) == 0
    names = sorted(p.name for p in staged)
    assert names == [
        "DONE.json",
        "inverted_index.npz",
        "random_directions.npz",
        "selection.shard00.jsonl",
    ]  # passA_ckpt_* filtered
    assert all(p.parent == sel for p in staged), staged  # FLAT consumer layout
    assert (sel / "inverted_index.npz").exists()
    staged.clear()
    assert EB.stage_selection(_stage_args(sel)) == 0  # idempotent skip
    assert staged == []


def test_stage_selection_empty_prefix_fails_loud(tmp_path, monkeypatch):
    """An empty Hub prefix (Pass A upload never ran) is a fail-loud
    FileNotFoundError, never a silent empty stage."""
    from types import SimpleNamespace

    import issue1773_evidence_builder as EB

    from explore_persona_space.orchestrate import hub

    monkeypatch.setattr(
        hub, "retry_transient", lambda fn, *, what="", **kw: SimpleNamespace(sha="a")
    )
    monkeypatch.setattr(
        hub,
        "list_hf_files_under_path",
        lambda api, repo_id, path, *, repo_type="model", revision=None: [],
    )
    with pytest.raises(FileNotFoundError, match="no selection files"):
        EB.stage_selection(_stage_args(tmp_path / "sel"))


def test_upload_selection_done_gate_skip_and_verify_fail(tmp_path, monkeypatch):
    """upload_selection: DONE.json gate; transient .tmp_* files excluded from
    the expected set; already-on-Hub -> skip without uploading; a post-upload
    verify miss -> fail-loud RuntimeError."""
    import issue1773_evidence_builder as EB

    from explore_persona_space.orchestrate import hub

    sel = tmp_path / "sel"
    sel.mkdir()
    with pytest.raises(AssertionError, match="completed Pass A"):
        EB.upload_selection(sel)
    (sel / "DONE.json").write_text("{}")
    (sel / "inverted_index.npz").write_bytes(b"x")
    (sel / ".tmp_partial.npz").write_bytes(b"x")

    calls = {"verify": [], "upload": 0}
    returns = [set()]  # first verify: nothing missing -> skip path

    def fake_verify(api, repo_id, expected, *, path_in_repo, repo_type):
        calls["verify"].append(sorted(expected))
        return returns.pop(0)

    def fake_retry(fn, *, what="", **kw):
        calls["upload"] += 1
        return None

    monkeypatch.setattr(hub, "verify_repo_paths_uploaded", fake_verify)
    monkeypatch.setattr(hub, "retry_transient", fake_retry)
    assert EB.upload_selection(sel) == 0
    assert calls["upload"] == 0  # skip path uploads nothing
    exp = calls["verify"][0]
    assert f"{CM.HF_SELECTION_PREFIX}/inverted_index.npz" in exp
    assert not any(".tmp_" in e for e in exp)  # transient files excluded

    returns.extend([{"m"}, {"m"}])  # missing before AND after upload
    with pytest.raises(RuntimeError, match="verify FAILED"):
        EB.upload_selection(sel)
    assert calls["upload"] == 1  # one bulk upload attempt via retry_transient


def test_passB_launcher_stages_selection_before_pilot():
    """The staging phase (producer of the fix-engaged `[stage] selection
    staged:` line) runs BEFORE the pilot and tees into the pilot log —
    crash-fix r3 ordering pin (the r3 crash was the pilot reading a
    never-staged selection on a fresh GCE clone)."""
    sh = (REPO / "scripts" / "issue1773_passB_launch.sh").read_text()
    i_stage = sh.index("--pass stage-selection")
    i_pilot = sh.index("[phase=passB_pilot]")
    assert sh.index("[phase=passB_stage_selection]") < i_stage < i_pilot


@pytest.mark.slow
def test_import_check_entrypoints():
    """Axis-1 leg is executable: --import-check resolves deferred imports."""
    for script in (
        "issue1773_evidence_builder.py",
        "issue1773_describe_axes.py",
        "issue1773_validate.py",
    ):
        import os

        proc = subprocess.run(
            ["uv", "run", "python", str(REPO / "scripts" / script), "--import-check"],
            capture_output=True,
            text=True,
            cwd=REPO,
            env={**os.environ},
            timeout=300,
        )
        assert proc.returncode == 0, f"{script}: {proc.stdout}\n{proc.stderr}"
        assert "[import-check] OK" in proc.stdout


# ── data-dependent gates: degenerate-input probes (separate from smoke legs) ──


def test_wiring_gate_fires_on_full_run_mismatch():
    """Phase-0 H1 wiring gate: HARD assert on a full run, informational under
    smoke (gate-calibration parity rule)."""
    import issue1773_phase0_mechanical as P0

    with pytest.raises(AssertionError):
        P0.assert_wiring_gate(0.5, full=True)
    P0.assert_wiring_gate(0.5, full=False)  # smoke leg: logged, never raises


def test_draw_pools_floor_returns_none_below_holdout_minimum():
    """Validation draw pools refuse a feature whose holdout pool is below the
    pinned 18-act/6-nonact floor (returns None -> feature skipped, reported)."""
    import issue1773_validate as V

    rng = np.random.default_rng(0)
    thin = {"ho_pos": [{"text_plain": "x", "text_marked": "x"}] * 5, "ho_neg": []}
    assert V._draw_pools(1, thin, rng) is None
    full = {
        "ho_pos": [{"text_plain": f"p{i}", "text_marked": f"p{i}"} for i in range(20)],
        "ho_neg": [{"text_plain": f"n{i}", "text_marked": f"n{i}"} for i in range(6)],
    }
    pools = V._draw_pools(1, full, rng)
    assert pools is not None
    assert len(pools["det_act"]) == 6 and len(pools["det_non"]) == 6
    assert len(pools["fuzz_correct"]) == 6 and len(pools["fuzz_incorrect"]) == 6
    # the 20-holdout pool is the SOLE source: 6 + 12 = 18 distinct draws
    drawn = [id(w) for w in pools["det_act"] + pools["fuzz_correct"] + pools["fuzz_incorrect"]]
    assert len(set(drawn)) == 18


def test_nonact_span_verify_failed_branch():
    """Non-activating verifier: an everywhere-active feature exhausts the one
    span re-draw and returns verify_failed=True (designed handling)."""
    import issue1773_evidence_builder as EB
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    ids = tok("a b c d e f g h i j k l m n o p", add_special_tokens=False)["input_ids"]
    acts = np.ones(len(ids), dtype=np.float32)
    ok, rec = EB._nonact_span(tok, ids, 0, acts, np.random.default_rng(0))
    assert ok is False and rec["text_plain"]
    ok2, _rec2 = EB._nonact_span(
        tok, ids, 0, np.zeros(len(ids), np.float32), np.random.default_rng(0)
    )
    assert ok2 is True


def test_scorecard_figure_inverted_ci_no_valueerror(tmp_path):
    """Errorbar offsets are clamped non-negative element-wise: a deliberately
    INVERTED bootstrap CI (lo > mean) renders without ValueError (#547/#1335)."""
    import issue1773_report as RP

    scorecard = {
        "axes": {
            a: {
                "detection": 0.7,
                "fuzzing": 0.7,
                "discrimination": 0.5,
                "kappa": 0.6,
                "shuffled_detection": 0.5,
            }
            for a in CM.AXES
        },
        "aggregates": {
            "real_detection": {"mean": 0.7, "n_features": 3, "ci95": [0.9, 0.4]},  # INVERTED
            "real_fuzzing": {"mean": 0.7, "n_features": 3, "ci95": [0.65, 0.75]},
            "real_discrimination": {"mean": 0.5, "n_features": 3, "ci95": [0.45, 0.55]},
            "shuffled_detection": {"mean": 0.52, "n_features": 3, "ci95": [0.5, 0.54]},
            "randinit_detection": {"mean": 0.55, "n_features": 3, "ci95": [0.5, 0.6]},
        },
    }
    out = tmp_path / "hero.png"
    RP.render_scorecard_figure(scorecard, out)
    assert out.exists() and out.stat().st_size > 0


def test_load_packets_empty_dir_fails_loud(tmp_path):
    import issue1773_describe_axes as DA

    (tmp_path / "evidence_manifests").mkdir()
    with pytest.raises(AssertionError):
        DA.load_packets(tmp_path)
