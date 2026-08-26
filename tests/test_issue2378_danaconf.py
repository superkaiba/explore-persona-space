"""Issue #2378 `dana-behavior-confirm` round — confirm-only overrides.

Covers: (1) the ``--confirm-families`` override (valid family runs against
the round out-root; an unknown / non-steered / empty key fails loud naming
it — validated against the REALIZED grid family set); (2) round-scoped
out-root threading (``--ledger-subdir`` harvest destination, ``--hf-suffix``
prefixes, confirm rollouts under the round root); (3) the confirm-only
upload tolerance (absent anchors/grid skipped ONLY under the override; the
original round's fail-loud contract unchanged); (4) ``bank_staged`` (staged
banked v_C inputs + fresh-pod gate re-run); (5) the vm_stage round/stage/
tensor parametrization. GPU/network boundaries are faked signature-
conformantly (``create_autospec``); everything else executes real bodies.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

DANA_FAM = "chat~story|Dana|b2a|lstar|steered"
DANA_NULL = "chat~story|Dana|b2a|lstar|null"
CHARS = {"Dana": ("d0", "d1"), "Vex": ("v0", "v1")}


def _ctx_fixture(root: Path) -> list[dict]:
    """Tiny bank fixture: 2 chars x 2 qids x 3 framings, vc keys == ctx ids."""
    rows = []
    for char, qids in CHARS.items():
        for qid in qids:
            for framing in ("chat", "story", "plain"):
                rows.append(
                    {"ctx_id": f"{framing}:{qid}", "framing": framing, "char": char, "qid": qid}
                )
    bank = root / "bank"
    bank.mkdir(parents=True, exist_ok=True)
    with (bank / "bank_rows.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r) + "\n")
    np.savez(bank / "vc_bank.npz", **{r["ctx_id"]: np.zeros((4, 8), dtype=np.uint16) for r in rows})
    return rows


# ── --confirm-families parse + validation ───────────────────────────────────


def test_confirm_family_override_absent_is_none():
    import issue2378_patch_run as run

    assert run._confirm_family_override(None, {DANA_FAM}) is None


def test_confirm_family_override_valid_dedup_sorted():
    import issue2378_patch_run as run

    grid = {DANA_FAM, DANA_NULL, "chat~plain|-|a2b|all|steered"}
    got = run._confirm_family_override(f" {DANA_FAM} ,{DANA_FAM}", grid)
    assert got == [DANA_FAM]


def test_confirm_family_override_empty_raises():
    import issue2378_patch_run as run

    with pytest.raises(SystemExit, match="empty after parsing"):
        run._confirm_family_override(" , ", {DANA_FAM})


def test_confirm_family_override_non_steered_raises():
    import issue2378_patch_run as run

    with pytest.raises(SystemExit, match="steered"):
        run._confirm_family_override(DANA_NULL, {DANA_FAM, DANA_NULL})


def test_confirm_family_override_unknown_raises_naming_key():
    import issue2378_patch_run as run

    bogus = "chat~story|Nobody|b2a|lstar|steered"
    with pytest.raises(SystemExit, match="Nobody"):
        run._confirm_family_override(bogus, {DANA_FAM, DANA_NULL})


def test_phase_confirm_unknown_family_raises_through_real_body(tmp_path):
    """End-to-end through the REAL phase body (bank load -> grid cells ->
    validation raise, before any model load) — the typo guard fires loud and
    names the unknown key."""
    import issue2378_patch_run as run

    _ctx_fixture(tmp_path)
    args = run.build_argparser().parse_args(
        [
            "--phase",
            "confirm",
            "--out-root",
            str(tmp_path),
            "--confirm-families",
            "chat~story|Nobody|b2a|lstar|steered",
        ]
    )
    with pytest.raises(SystemExit, match="Nobody"):
        run.phase_confirm(args)


def test_phase_confirm_override_runs_only_requested_family(tmp_path, monkeypatch):
    """A valid override runs the confirm phase against the ROUND out-root:
    only the requested family + its matched null are dispatched, rollouts
    land under <out-root>/confirm/rollouts, the ledger regime carries the
    override, and the screen report is never read (no screen dir exists)."""
    import issue2378_patch_run as run

    _ctx_fixture(tmp_path)

    def fake_block(args, mctx, tokz, block_cells, by_ctx, bank_vc, dmaps, greedy, draws, seed_tag):
        return [
            {
                **{k: c[k] for k in ("cell_id", "family", "qid", "arm", "variant", "src", "tgt")},
                "draw": d,
                "drop_reason": None,
                "va_dropped": False,
                "answer": "reply",
            }
            for c in block_cells
            for d in range(draws)
        ]

    monkeypatch.setattr(
        run, "_ensure_mctx", mock.create_autospec(run._ensure_mctx, return_value={})
    )
    monkeypatch.setattr(run, "_tok", mock.create_autospec(run._tok, return_value=None))
    monkeypatch.setattr(
        run,
        "_run_hooked_block",
        mock.create_autospec(run._run_hooked_block, side_effect=fake_block),
    )
    monkeypatch.setattr(
        run, "_va_capture", mock.create_autospec(run._va_capture, return_value=None)
    )
    args = run.build_argparser().parse_args(
        [
            "--phase",
            "confirm",
            "--out-root",
            str(tmp_path),
            "--confirm-families",
            DANA_FAM,
            "--lstar",
            "3",
            "--tiny",
        ]
    )
    assert run.phase_confirm(args) == 0
    files = sorted((tmp_path / "confirm" / "rollouts").glob("*.jsonl"))
    assert files, "confirm rollouts must land under the ROUND out-root"
    rows = [
        json.loads(line)
        for f in files
        for line in f.read_text(encoding="utf-8").split("\n")
        if line.strip()
    ]
    assert {r["family"] for r in rows} == {DANA_FAM, DANA_NULL}
    assert {r["qid"] for r in rows} == {"d0", "d1"}
    assert len(rows) == 4 * int(args.confirm_draws)  # 2 qids x (steered+null) x K draws
    ledger = json.loads((tmp_path / "confirm" / "ledger.json").read_text(encoding="utf-8"))
    assert ledger["regime"]["families"] == [DANA_FAM]
    assert not (tmp_path / "screen").exists(), "override must bypass the screen report entirely"


def test_phase_confirm_default_path_unchanged(tmp_path):
    """No override -> the screen-report path is byte-identical to the parent
    round (empty confirm_families still writes the terminal record)."""
    import issue2378_patch_run as run

    (tmp_path / "screen").mkdir(parents=True)
    (tmp_path / "screen" / "screen_report.json").write_text(
        json.dumps({"confirm_families": []}), encoding="utf-8"
    )
    args = run.build_argparser().parse_args(["--phase", "confirm", "--out-root", str(tmp_path)])
    assert run.phase_confirm(args) == 0
    assert (tmp_path / "confirm" / "rollouts" / "confirm_empty.json").exists()


# ── model chain + passthrough + out-root threading ──────────────────────────


def test_model_chain_confirm_only_selection():
    """Confirm-only rounds ALWAYS route through bank_staged (no fresh-bank
    recapture fallback — r1 review: donor-bit-identity is the round's claim;
    the missing-prefix case is refused by the round-scoping guard / the
    phase's own fail-loud)."""
    import issue2378_patch_run as run

    base = ["--phase", "model_all"]
    default = run.build_argparser().parse_args(base)
    assert run._model_chain(default) == (
        run.phase_bank,
        run.phase_anchors,
        run.phase_grid,
        run.phase_screen,
        run.phase_confirm,
    )
    conf = run.build_argparser().parse_args([*base, "--confirm-families", DANA_FAM])
    assert run._model_chain(conf) == (run.phase_bank_staged, run.phase_confirm)


def _round_scoped_argv(tmp: str) -> list[str]:
    return [
        "--hf-suffix",
        "_danaconf",
        "--ledger-subdir",
        "dana-behavior-confirm",
        "--out-root",
        f"{tmp}/root",
        "--logs-dir",
        f"{tmp}/logs",
        "--stage-bank-from-hf-prefix",
        "issue2378_xframing/raw_completions/causal_patching/bank",
    ]


def test_confirm_only_requires_round_scoped_flags(tmp_path):
    """r1 review BLOCKER danaconf-round-isolation-uncoupled: --confirm-families
    with ANY write-surface flag left at its original-round default must refuse
    naming the missing flag(s) — never silently overwrite the parent round's
    HF prefixes / committed harvest / local roots."""
    import issue2378_patch_run as run

    defaults = run.build_argparser().parse_args(
        ["--phase", "upload", "--confirm-families", DANA_FAM]
    )
    with pytest.raises(SystemExit) as exc:
        run._require_round_scoped_flags(defaults)
    msg = str(exc.value)
    for frag in (
        "--hf-suffix",
        "--ledger-subdir",
        "--out-root",
        "--logs-dir",
        "--stage-bank-from-hf-prefix",
    ):
        assert frag in msg, f"guard must name {frag}"
    # One flag still default -> still refused, naming exactly that flag.
    partial = run.build_argparser().parse_args(
        ["--phase", "upload", "--confirm-families", DANA_FAM, *_round_scoped_argv(str(tmp_path))][
            :-2
        ]  # drop --stage-bank-from-hf-prefix's value+flag pair
    )
    with pytest.raises(SystemExit, match="stage-bank-from-hf-prefix"):
        run._require_round_scoped_flags(partial)
    # Fully round-scoped -> passes; no override -> guard is a no-op on defaults.
    scoped = run.build_argparser().parse_args(
        ["--phase", "upload", "--confirm-families", DANA_FAM, *_round_scoped_argv(str(tmp_path))]
    )
    run._require_round_scoped_flags(scoped)
    run._require_round_scoped_flags(run.build_argparser().parse_args(["--phase", "upload"]))


def test_main_wires_round_scoping_guard(tmp_path, monkeypatch):
    """The guard fires at post-parse in main() (covers parent AND child CLI
    invocations via the passthrough) — BEFORE any phase body executes."""
    import issue2378_patch_run as run

    monkeypatch.setattr(
        sys, "argv", ["issue2378_patch_run.py", "--phase", "upload", "--confirm-families", DANA_FAM]
    )
    with pytest.raises(SystemExit, match="round-scoping"):
        run.main()


def test_child_passthrough_forwards_round_flags_and_roundtrips():
    import issue2378_patch_run as run

    default = run.build_argparser().parse_args(["--phase", "all"])
    pt = run._child_passthrough(default)
    assert "--confirm-families" not in pt and "--stage-bank-from-hf-prefix" not in pt
    assert pt[pt.index("--ledger-subdir") + 1] == "causal-patching-arms"
    dana = run.build_argparser().parse_args(
        [
            "--phase",
            "all",
            "--confirm-families",
            DANA_FAM,
            "--stage-bank-from-hf-prefix",
            "x/bank",
            "--ledger-subdir",
            "dana-behavior-confirm",
            "--hf-suffix",
            "_danaconf",
        ]
    )
    pt = run._child_passthrough(dana)
    # Round-trip through the real argparser: a passthrough/argparser drift
    # would silently re-default in the child (fail here instead).
    child = run.build_argparser().parse_args(["--phase", "confirm", *pt])
    assert child.confirm_families == DANA_FAM
    assert child.stage_bank_from_hf_prefix == "x/bank"
    assert child.ledger_subdir == "dana-behavior-confirm"
    assert child.hf_suffix == "_danaconf"
    # --logs-dir rides the passthrough (children must pass the round-scoping
    # guard with the SAME round-scoped logs dir, never the default).
    assert child.logs_dir == dana.logs_dir


def test_ledger_dir_and_hf_prefix_round_scoping():
    import issue2378_common as cmod
    import issue2378_patch_run as run

    default = run.build_argparser().parse_args(["--phase", "upload"])
    assert run._ledger_dir(default) == cmod.REPO_ROOT / "eval_results" / "issue_2378" / (
        "causal-patching-arms"
    )
    dana = run.build_argparser().parse_args(
        [
            "--phase",
            "upload",
            "--ledger-subdir",
            "dana-behavior-confirm",
            "--hf-suffix",
            "_danaconf",
        ]
    )
    assert run._ledger_dir(dana).name == "dana-behavior-confirm"
    assert (
        run._hf_prefix(dana, "raw") == "issue2378_xframing/raw_completions/causal_patching_danaconf"
    )
    assert (
        run._hf_prefix(dana, "tensor")
        == "issue2378_xframing/analysis_tensors/causal_patching_danaconf"
    )


# ── upload: confirm-only tolerance vs default fail-loud ─────────────────────


def _upload_fixture(tmp_path: Path) -> Path:
    root = tmp_path / "root"
    (root / "bank").mkdir(parents=True)
    (root / "bank" / "gate_report.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
    (root / "confirm" / "rollouts").mkdir(parents=True)
    (root / "confirm" / "rollouts" / "u.jsonl").write_text('{"cell_id": "c"}\n', encoding="utf-8")
    (root / "confirm" / "ledger.json").write_text("{}", encoding="utf-8")
    (root / "confirm" / "va").mkdir()
    np.savez(root / "confirm" / "va" / "u.npz", va=np.zeros((1, 2, 4), dtype=np.uint16))
    return root


def test_phase_upload_confirm_only_round_scoped(tmp_path, monkeypatch):
    import issue2378_common as cmod
    import issue2378_dispatch as D
    import issue2378_patch_run as run

    root = _upload_fixture(tmp_path)
    repo = tmp_path / "repo"
    monkeypatch.setattr(cmod, "REPO_ROOT", repo)
    up = mock.create_autospec(cmod.upload_stage_dir, return_value=[])
    monkeypatch.setattr(cmod, "upload_stage_dir", up)
    gh = mock.create_autospec(D.git_harvest)
    monkeypatch.setattr(D, "git_harvest", gh)
    ws = mock.create_autospec(D.write_sentinel)
    monkeypatch.setattr(D, "write_sentinel", ws)
    args = run.build_argparser().parse_args(
        [
            "--phase",
            "upload",
            "--out-root",
            str(root),
            "--confirm-families",
            DANA_FAM,
            "--ledger-subdir",
            "dana-behavior-confirm",
            "--hf-suffix",
            "_danaconf",
        ]
    )
    assert run.phase_upload(args) == 0
    prefixes = sorted(c.args[1] for c in up.call_args_list)
    assert prefixes == [
        "issue2378_xframing/analysis_tensors/causal_patching_danaconf/confirm_va",
        "issue2378_xframing/raw_completions/causal_patching_danaconf/bank",
        "issue2378_xframing/raw_completions/causal_patching_danaconf/confirm",
        "issue2378_xframing/raw_completions/causal_patching_danaconf/meta",
    ]
    assert (root / "meta" / "confirm_ledger.json").is_file()
    assert not (root / "meta" / "openers.jsonl").exists()
    dest = repo / "eval_results" / "issue_2378" / "dana-behavior-confirm" / "gate_report.json"
    assert dest.is_file(), "harvest must land under the ROUND ledger subdir, never the original"
    harvested, msg = gh.call_args.args
    assert harvested == ["eval_results/issue_2378/dana-behavior-confirm/gate_report.json"]
    assert "dana-behavior-confirm" in msg
    payload = ws.call_args.args[2]
    assert payload["followup_label"] == "dana-behavior-confirm"
    assert payload["confirm_families"] == DANA_FAM
    assert payload["hf_raw_prefix"].endswith("_danaconf")


def test_phase_upload_default_mode_still_fail_loud(tmp_path):
    """Without the override, a root missing anchors/openers keeps the
    original round's fail-loud contract (no silent skip)."""
    import issue2378_patch_run as run

    root = _upload_fixture(tmp_path)
    args = run.build_argparser().parse_args(["--phase", "upload", "--out-root", str(root)])
    with pytest.raises(RuntimeError, match=r"openers\.jsonl"):
        run.phase_upload(args)


# ── bank_staged: staged banked inputs + fresh-pod gate re-run ───────────────


_REV = "deadbeefcafe0123deadbeefcafe0123deadbeef"
_BANK_PREFIX = "issue2378_xframing/raw_completions/causal_patching/bank"


def _fake_bank_hub(monkeypatch, ledger_digest_fn):
    """Fake the network boundary of ``_stage_bank_files`` signature-
    conformantly: pinned-revision downloads serve deterministic payload bytes;
    the sibling grid ledger serves ``regime["bank"] = ledger_digest_fn(true)``.
    Returns the download mock."""
    import hashlib
    import types

    import huggingface_hub as hf

    payloads = {
        "bank_rows.jsonl": b"payload:rows",
        "vc_bank.npz": b"payload:vc",
        "gate_report.json": b"{}",
    }
    true_digest = hashlib.sha256(payloads["bank_rows.jsonl"] + payloads["vc_bank.npz"]).hexdigest()[
        :16
    ]

    def fake_download(repo_id, filename, repo_type=None, revision=None, local_dir=None):
        assert revision == _REV, "every download must use the ONE pinned revision (#2061)"
        p = Path(local_dir) / filename
        p.parent.mkdir(parents=True, exist_ok=True)
        name = filename.rsplit("/", 1)[-1]
        if name == "grid_ledger.json":
            p.write_text(
                json.dumps({"regime": {"bank": ledger_digest_fn(true_digest)}}), encoding="utf-8"
            )
        else:
            p.write_bytes(payloads[name])
        return str(p)

    dl = mock.create_autospec(hf.hf_hub_download, side_effect=fake_download)
    monkeypatch.setattr(hf, "hf_hub_download", dl)
    fake_api_cls = mock.create_autospec(hf.HfApi)
    fake_api_cls.return_value.repo_info.return_value = types.SimpleNamespace(sha=_REV)
    monkeypatch.setattr(hf, "HfApi", fake_api_cls)
    return dl


def test_stage_bank_files_real_body_digest_bound(tmp_path, monkeypatch):
    """Real ``_stage_bank_files`` body; only the network boundary is faked,
    signature-conformantly. Verifies: main->sha pinned ONCE and used on every
    download (#2061), idempotent present-target skip, atomic publish + scratch
    cleanup, grid-ledger digest binding PASS, and the provenance manifest."""
    import issue2378_patch_run as run

    _fake_bank_hub(monkeypatch, lambda true: true)  # ledger digest == staged digest
    out = tmp_path / "bank"
    out.mkdir()
    (out / "gate_report.json").write_text("{}", encoding="utf-8")  # present -> idempotent skip
    manifest = run._stage_bank_files(_BANK_PREFIX, out)
    assert (out / "bank_rows.jsonl").read_bytes() == b"payload:rows"
    assert (out / "vc_bank.npz").read_bytes() == b"payload:vc"
    assert (out / "gate_report.json").read_text(encoding="utf-8") == "{}"
    assert not (out / ".hfstage").exists(), "staging scratch must be cleaned up"
    assert manifest["revision"] == _REV
    assert manifest["bank_digest"] == run._bank_payload_digest(out)
    persisted = json.loads((out / "bank_staged_manifest.json").read_text(encoding="utf-8"))
    assert persisted["staged_from"] == _BANK_PREFIX
    assert persisted["bank_digest"] == manifest["bank_digest"]


def test_stage_bank_files_refuses_digest_mismatch(tmp_path, monkeypatch):
    """A wrong/stale bank (digest != the original grid ledger's regime bank)
    refuses loud BEFORE any model load — never confirm against unknown donors
    (r1 review danaconf-bank-provenance-unbound)."""
    import issue2378_patch_run as run

    _fake_bank_hub(monkeypatch, lambda true: "0" * 16)  # ledger names a DIFFERENT bank
    out = tmp_path / "bank"
    out.mkdir()
    with pytest.raises(RuntimeError, match="digest"):
        run._stage_bank_files(_BANK_PREFIX, out)
    assert not (out / "bank_staged_manifest.json").exists()


def test_phase_bank_staged_requires_prefix():
    import issue2378_patch_run as run

    args = run.build_argparser().parse_args(["--phase", "bank_staged", "--out-root", "/tmp/x"])
    with pytest.raises(SystemExit, match="stage-bank-from-hf-prefix"):
        run.phase_bank_staged(args)


def _fake_staged_bank(run, monkeypatch, gate_ok=True):
    """Autospec'd GPU/network boundaries for phase_bank_staged tests; the
    staging fake writes the real fixture + returns a REAL-digest manifest."""

    def fake_stage(prefix, out):
        _ctx_fixture(out.parent)
        (out / "gate_report.json").write_text(
            json.dumps({"ok": True, "stale_original": True}), encoding="utf-8"
        )
        return {
            "staged_from": prefix,
            "revision": _REV,
            "bank_digest": run._bank_payload_digest(out),
        }

    stage = mock.create_autospec(run._stage_bank_files, side_effect=fake_stage)
    monkeypatch.setattr(run, "_stage_bank_files", stage)
    monkeypatch.setattr(
        run, "_ensure_mctx", mock.create_autospec(run._ensure_mctx, return_value={})
    )
    gates = mock.create_autospec(run._run_gates, return_value={"ok": gate_ok, "spots": []})
    monkeypatch.setattr(run, "_run_gates", gates)
    return stage, gates


def _bank_staged_argv(tmp_path) -> list[str]:
    return [
        "--phase",
        "bank_staged",
        "--out-root",
        str(tmp_path),
        "--stage-bank-from-hf-prefix",
        _BANK_PREFIX,
        "--lstar",
        "3",
        "--tiny",
    ]


def test_phase_bank_staged_gates_and_sentinel_idempotency(tmp_path, monkeypatch):
    """Real phase body: staged bank -> real ``_load_bank`` key-coverage ->
    gates re-run on THIS pod's env (GPU boundary autospec'd) -> fresh verdict
    supersedes the staged gate report -> provenance-keyed completion sentinel.
    A matching re-entry skips staging AND the 27B gate re-run (r1 review
    danaconf-bank-staged-not-idempotent); a corrupted bank under a sentinel
    fails loud."""
    import issue2378_patch_run as run

    stage, gates = _fake_staged_bank(run, monkeypatch)
    argv = _bank_staged_argv(tmp_path)
    assert run.phase_bank_staged(run.build_argparser().parse_args(argv)) == 0
    rep = json.loads((tmp_path / "bank" / "gate_report.json").read_text(encoding="utf-8"))
    assert rep["ok"] is True
    assert "stale_original" not in rep, "fresh-pod gate verdict must supersede the staged report"
    assert rep["metadata"]["staged_from"].endswith("/bank")
    assert rep["metadata"]["phase"] == "patch_bank_staged"
    sentinel = json.loads(
        (tmp_path / "bank" / run.BANK_STAGED_SENTINEL).read_text(encoding="utf-8")
    )
    assert sentinel["gate_ok"] is True
    assert sentinel["bank_digest"] == run._bank_payload_digest(tmp_path / "bank")
    # Re-entry: provenance-keyed skip — no re-staging, no 27B gate re-run.
    assert run.phase_bank_staged(run.build_argparser().parse_args(argv)) == 0
    assert stage.call_count == 1 and gates.call_count == 1
    # Corrupt the bank under the sentinel -> fail loud (stale/foreign root).
    (tmp_path / "bank" / "vc_bank.npz").write_bytes(b"corrupt")
    with pytest.raises(RuntimeError, match="mismatch"):
        run.phase_bank_staged(run.build_argparser().parse_args(argv))


def test_phase_bank_staged_gate_fail_no_sentinel_then_retry_reruns(tmp_path, monkeypatch):
    """Gate FAIL returns RC_GATE and writes NO sentinel — a retry after a fix
    re-runs staging + gates (never a false-PASS resume)."""
    import issue2378_patch_run as run

    _stage, gates = _fake_staged_bank(run, monkeypatch, gate_ok=False)
    argv = _bank_staged_argv(tmp_path)
    assert run.phase_bank_staged(run.build_argparser().parse_args(argv)) == run.RC_GATE
    assert not (tmp_path / "bank" / run.BANK_STAGED_SENTINEL).exists()
    gates.return_value = {"ok": True, "spots": []}
    assert run.phase_bank_staged(run.build_argparser().parse_args(argv)) == 0
    assert gates.call_count == 2


# ── vm_stage: round/stage/tensor parametrization ────────────────────────────


def test_vm_stage_dest_rel_mapping():
    import issue2378_patch_vm_stage as vs

    p = "issue2378_xframing/raw_completions/causal_patching_danaconf"
    assert vs._dest_rel(p, f"{p}/confirm/u.jsonl", None) == "confirm/rollouts/u.jsonl"
    assert vs._dest_rel(p, f"{p}/bank/vc_bank.npz", {"bank"}) == "bank/vc_bank.npz"
    assert vs._dest_rel(p, f"{p}/bank/vc_parts/part0.npz", {"bank"}) == "bank/vc_parts/part0.npz"
    assert vs._dest_rel(p, f"{p}/meta/openers.jsonl", {"confirm"}) is None
    t = "issue2378_xframing/analysis_tensors/causal_patching"
    assert vs._dest_rel(t, f"{t}/anchors_va/x.npz", {"anchors"}) == "anchors/va/x.npz"
    assert vs._dest_rel(t, f"{t}/confirm_va/x.npz", {"anchors"}) is None
    assert vs._dest_rel(t, f"{t}/confirm_va/x.npz", None) == "confirm/va/x.npz"


def test_vm_stage_stage_back_real_body(tmp_path, monkeypatch):
    import huggingface_hub as hf
    import issue2378_patch_vm_stage as vs

    from explore_persona_space.orchestrate import hub

    p = "issue2378_xframing/raw_completions/causal_patching_danaconf"
    listing = [f"{p}/confirm/u.jsonl", f"{p}/bank/vc_bank.npz", f"{p}/judge_persona/r.jsonl"]
    monkeypatch.setattr(
        hub,
        "list_hf_files_under_path",
        mock.create_autospec(hub.list_hf_files_under_path, return_value=listing),
    )

    def fake_download(repo_id, filename, repo_type=None, local_dir=None):
        fp = Path(local_dir) / filename
        fp.parent.mkdir(parents=True, exist_ok=True)
        fp.write_bytes(b"x")
        return str(fp)

    monkeypatch.setattr(
        hf, "hf_hub_download", mock.create_autospec(hf.hf_hub_download, side_effect=fake_download)
    )
    dest = tmp_path / "root"
    assert vs.stage_back(dest, hf_suffix="_danaconf", stages={"confirm", "bank"}) == 0
    assert (dest / "confirm" / "rollouts" / "u.jsonl").is_file()
    assert (dest / "bank" / "vc_bank.npz").is_file()
    assert not (dest / "judge_persona").exists()
    with pytest.raises(RuntimeError, match="empty selection"):
        vs.stage_back(dest, hf_suffix="_danaconf", stages={"nonexistent"})


# ── analysis: round identity threading ──────────────────────────────────────


def test_analysis_followup_label_threaded(tmp_path, monkeypatch):
    """Real ``issue2378_patch_analysis.main`` body over a minimal-real fixture
    (screen report + a published judge fold; no anchors/grid/confirm dirs):
    ``--followup-label`` lands in patch_summary.json (r1 review
    danaconf-analysis-label-hardcoded — the Dana summary must not carry the
    parent round's identity)."""
    import issue2378_patch_analysis as pa
    import issue2378_patch_judge as pj

    root = tmp_path / "root"
    (root / "screen").mkdir(parents=True)
    (root / "screen" / "screen_report.json").write_text(
        json.dumps(
            {
                "screen_rule": "ci-excludes-0",
                "families": {},
                "confirm_families": [],
                "family_means": {},
            }
        ),
        encoding="utf-8",
    )
    judge_dir = tmp_path / "judge"
    judge_dir.mkdir()
    pj._write_fold(
        judge_dir,
        {
            "anchors|chat:q0_00|d0|persona": {
                "class": "valid",
                "score": 80,
                "reasoning": None,
                "stop_reason": "end_turn",
            }
        },
    )
    out_dir = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "issue2378_patch_analysis.py",
            "--patch-root",
            str(root),
            "--judge-dir",
            str(judge_dir),
            "--out-dir",
            str(out_dir),
            "--followup-label",
            "dana-behavior-confirm",
        ],
    )
    pa.main()
    summary = json.loads((out_dir / "patch_summary.json").read_text(encoding="utf-8"))
    assert summary["followup_label"] == "dana-behavior-confirm"
