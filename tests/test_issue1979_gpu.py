"""#1979 F1 driver pins (crash-fix r3, fellows job 16717).

(a) f1c known-persona set: built from the LOADED mix rows' realized labels
    validated against arm_registry.json — a shared training-mix pool carries
    the representative (FT) arm's slug, not the mix id; a genuinely foreign
    label still fails loud.
(b) dispatcher per-unit failure budget: one failed unit is NON-fatal (siblings
    keep scheduling; the failed unit stays resumable — no done sentinel; the
    run still exits non-zero), while >FAILURE_BUDGET failures or a systemic
    same-exception-class repeat aborts early.

Everything runs CPU-tiny in tmp_path; the subprocess boundary is faked with a
signature-conformant Popen stand-in (worker commands only — everything else
delegates to the real Popen), and the disk-headroom probe is faked with a
signature-conformant no-op. The dispatch body itself executes for real.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

import issue1979_gpu as G  # noqa: E402


def _cfg(tmp_path: Path) -> G.Cfg:
    return G.Cfg(out_root=tmp_path, config_dir=tmp_path / "config", phases=("f1c",))


def _write_registry(tmp_path: Path, arm_ids: list[str]) -> None:
    (tmp_path / "arm_registry.json").write_text(
        json.dumps(
            {"mix_pos_sources": {a: {"pos_path": "p.jsonl", "layout": "delta"} for a in arm_ids}}
        )
    )


# ── (a) f1c known-persona set ────────────────────────────────────────────────


def test_anchor_known_personas_accepts_shared_pool_ft_label(tmp_path):
    """The job-16717 crash shape: mix rows labeled with the representative
    FT arm's slug under a LoRA mix id must pass (both registry-registered)."""
    cfg = _cfg(tmp_path)
    _write_registry(tmp_path, ["syc-pers-ft-con-s42", "syc-pers-con-lr1e5-s42"])
    rows = [{"persona": "syc-pers-ft-con-s42"} for _ in range(3)]
    known = G._anchor_known_personas(cfg, rows, "syc-pers-con-lr1e5-s42")
    assert set(known) == {"syc-pers-ft-con-s42", "syc-pers-con-lr1e5-s42"}
    known_set = set(known)
    for i, r in enumerate(rows):  # the reused helper's integrity assert, same shape
        assert r["persona"] in known_set, (i, r["persona"])


def test_anchor_known_personas_foreign_label_fails_loud(tmp_path):
    cfg = _cfg(tmp_path)
    _write_registry(tmp_path, ["syc-pers-ft-con-s42"])
    rows = [{"persona": "syc-pers-ft-con-s42"}, {"persona": "not-a-registered-arm"}]
    with pytest.raises(AssertionError) as ei:
        G._anchor_known_personas(cfg, rows, "syc-pers-con-lr1e5-s42")
    assert "not-a-registered-arm" in str(ei.value)


# ── (b) dispatcher failure budget ────────────────────────────────────────────


class _FakeProc:
    """Signature-conformant stand-in for the worker subprocess (dispatch reads
    only .pid and .poll())."""

    def __init__(self, rc: int):
        self.pid = 4242
        self._rc = rc

    def poll(self) -> int:
        return self._rc


def _patch_dispatch_seams(monkeypatch, cfg: G.Cfg, fail_plan: dict, launched: list[str]) -> None:
    """Fake ONLY the external boundaries: the worker subprocess (writes the same
    done-sentinel / failure-breadcrumb files a real worker writes) and the
    disk-headroom probe; CVD env pins two fake workers."""
    real_popen = G.subprocess.Popen

    def _popen(cmd, env=None, **kwargs):
        if "--worker-unit" not in cmd:
            return real_popen(cmd, env=env, **kwargs)
        key = cmd[cmd.index("--worker-unit") + 1]
        launched.append(key)
        exc_class = fail_plan.get(key)
        if exc_class is None:
            G.CAP._atomic_json(G._sentinel_path(cfg, key), {"key": key, "rc": 0})
            return _FakeProc(0)
        exc_type = type(exc_class, (RuntimeError,), {})
        G._write_failure(cfg, key, exc_type("boom"))
        return _FakeProc(1)

    def _no_headroom(out_root, need_gb, *, phase="", canary_gb=1.0):
        return 0.0

    monkeypatch.setattr(G.subprocess, "Popen", _popen)
    monkeypatch.setattr(
        "explore_persona_space.orchestrate.preflight.assert_out_root_headroom", _no_headroom
    )
    monkeypatch.setattr(G.time, "sleep", lambda s: None)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")


def test_dispatch_one_failure_is_nonfatal_but_run_exits_nonzero(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    items = [G.Item(key=f"f1c:m{i}", phase="f1c") for i in range(8)]
    items.append(G.Item(key="f1c:dep", phase="f1c", deps=("f1c:m0",)))
    launched: list[str] = []
    _patch_dispatch_seams(monkeypatch, cfg, {"f1c:m0": "AssertionError"}, launched)
    with pytest.raises(RuntimeError) as ei:
        G.dispatch(cfg, {}, items)
    msg = str(ei.value)
    assert "f1c:m0" in msg and "AssertionError" in msg
    for i in range(1, 8):  # every independent sibling still scheduled + completed
        assert G._done(cfg, f"f1c:m{i}"), f"sibling f1c:m{i} was not scheduled to completion"
    assert not G._done(cfg, "f1c:m0")  # failed unit resumable: no done sentinel
    assert "f1c:dep" not in launched  # dependent of the failed unit never scheduled
    assert "never scheduled" in msg
    assert not (tmp_path / "f1_results.json").exists()  # no done record on a failed run


def test_dispatch_over_budget_aborts_scheduling(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    items = [G.Item(key=f"f1c:m{i}", phase="f1c") for i in range(12)]
    # distinct exception classes so the systemic detector cannot fire first
    fail_plan = {f"f1c:m{i}": f"Exc{i}" for i in range(12)}
    launched: list[str] = []
    _patch_dispatch_seams(monkeypatch, cfg, fail_plan, launched)
    with pytest.raises(RuntimeError) as ei:
        G.dispatch(cfg, {}, items)
    msg = str(ei.value)
    assert "failure budget exceeded" in msg
    assert len(launched) == G.FAILURE_BUDGET + 1  # abort right past the budget
    assert len(launched) < len(items)  # remaining units never scheduled


def test_dispatch_systemic_same_class_aborts_early(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    items = [G.Item(key=f"f1c:m{i}", phase="f1c") for i in range(10)]
    fail_plan = {it.key: "CudaOOM" for it in items}
    launched: list[str] = []
    _patch_dispatch_seams(monkeypatch, cfg, fail_plan, launched)
    with pytest.raises(RuntimeError) as ei:
        G.dispatch(cfg, {}, items)
    msg = str(ei.value)
    assert "systemic failure: CudaOOM" in msg
    assert len(launched) <= G.FAILURE_BUDGET  # aborted below the plain budget
    assert len(launched) < len(items)


def test_dispatch_clean_run_emits_terminal_record(tmp_path, monkeypatch):
    pytest.importorskip("torch")  # _meta() in the terminal record imports torch
    cfg = _cfg(tmp_path)
    items = [G.Item(key=f"f1c:m{i}", phase="f1c") for i in range(3)]
    launched: list[str] = []
    _patch_dispatch_seams(monkeypatch, cfg, {}, launched)
    G.dispatch(cfg, {}, items)
    assert (tmp_path / "f1_results.json").exists()
    assert len(launched) == 3


# ── (c) crash-fix r4: marker slot-key contract (job 16731 KeyError) ──────────


def _tiny_slot_fixture():
    """REAL ``compute_marker_slot_stats`` on a CPU-tiny fixture: the return-dict
    SHAPE comes from the real function (never a hand-built dict); only the
    GPU-scale model + HF tokenizer are stand-ins (signature-conformant forward
    returning real (B, T, V) logits; single-token marker encode)."""
    import types

    torch = pytest.importorskip("torch")
    from explore_persona_space.eval.marker_logprob import compute_marker_slot_stats

    class _TinyLM(torch.nn.Module):
        def __init__(self, vocab: int = 32, hidden: int = 8):
            super().__init__()
            torch.manual_seed(0)
            self.emb = torch.nn.Embedding(vocab, hidden)
            self.head = torch.nn.Linear(hidden, vocab)

        def forward(self, input_ids=None, attention_mask=None):
            del attention_mask  # causal stub: last-position logits are all the callee reads
            return types.SimpleNamespace(logits=self.head(self.emb(input_ids)))

    class _TinyTok:
        eos_token_id = 2
        pad_token_id = 0

        def encode(self, text: str, add_special_tokens: bool = False):
            if text == G.MARKER_TEXT:
                return [5]  # single-token marker (the callee's own assert)
            return [3, 4 + (len(text) % 7)]

    return compute_marker_slot_stats(
        _TinyLM(),
        _TinyTok(),
        ["ctx a", "ctx bee", "ctx sea"],
        G.MARKER_TEXT,
        device="cpu",
        include_argmax=True,
    )


def test_gate_consumer_accepts_real_slot_stats_contract():
    """The job-16731 crash shape: run_f1b_gate's median consumer must accept the
    REAL compute_marker_slot_stats return dicts (contract keys logp/z_marker/
    z_eos/logZ — NOT the guessed 'logp_marker')."""
    pytest.importorskip("torch")
    from explore_persona_space.eval.marker_logprob import MARKER_SLOT_CONTRACT_KEYS

    recs = _tiny_slot_fixture()
    assert len(recs) == 3
    assert set(MARKER_SLOT_CONTRACT_KEYS) <= set(recs[0]), sorted(recs[0])
    assert "logp_marker" not in recs[0]  # the r4 guessed key is NOT the contract
    med_logp, med_z = G._gate_medians(recs, recs, recs, recs)
    assert med_logp == 0.0 and med_z == 0.0
    shifted = [{**r, "logp": r["logp"] + 3.0, "z_marker": r["z_marker"] + 1.5} for r in recs]
    med_logp2, med_z2 = G._gate_medians(shifted, recs, shifted, recs)
    assert med_logp2 == pytest.approx(3.0) and med_z2 == pytest.approx(1.5)


def test_gate_ft_tolerance_and_mandatory_secondary_condition():
    """Crash-fix r5 (`f1b:gate:mk-pers-ft-con-s42` at 1.99 nats): full-FT marker
    arms gate at 1.8 nats with the |dz| secondary condition MANDATORY; LoRA arms
    keep the 2.0-nat #1900 bar; ~0-nat reads fail regardless of class."""
    ft_row = {"kind": "marker", "method": "ft", "ft_repo": "org/repo", "ft_subfolder": "s"}
    lora_row = {"kind": "marker", "method": "lora", "adapter_repo": "org/repo"}
    assert G._gate_threshold_for(ft_row) == (1.8, "ft")
    assert G._gate_threshold_for(lora_row) == (2.0, "lora")
    # FT at 1.9 nats + positive |dz| passes (incl. the realized r4 value, 1.99)
    G._assert_gate_pass("mk-pers-ft-con-s42", "ft", 1.8, 1.9, 0.4)
    G._assert_gate_pass("mk-pers-ft-con-s42", "ft", 1.8, 1.99, 0.4)
    # FT at 1.9 nats + |dz| == 0 fails — secondary condition is mandatory
    with pytest.raises(AssertionError, match=r"dz_marker"):
        G._assert_gate_pass("mk-pers-ft-con-s42", "ft", 1.8, 1.9, 0.0)
    # ~0 nats (unapplied / wrong artifact) fails regardless of arm class
    with pytest.raises(AssertionError, match=r"dlogP"):
        G._assert_gate_pass("mk-pers-ft-con-s42", "ft", 1.8, 0.02, 0.0)
    with pytest.raises(AssertionError, match=r"dlogP"):
        G._assert_gate_pass("mk-a", "lora", 2.0, 0.02, 0.4)
    # LoRA keeps the 2.0 bar: 1.9 nats fails for a LoRA arm
    with pytest.raises(AssertionError, match=r"dlogP"):
        G._assert_gate_pass("mk-a", "lora", 2.0, 1.9, 0.4)


def test_gate_mix_contexts_cut_restricted_to_response_region():
    """Crash-fix r6: ICL-context arms carry the marker glyph inside the PROMPT's
    in-context demos — the gate's cut must search only the response region, keeping
    the full prompt (demo markers included) and cutting the response at its own
    first marker (rstripped). No marker in the response => full response kept."""

    class _StubTok:
        """Tokenizer stand-in: decode() maps fixed id tuples to fixed strings."""

        def __init__(self):
            self._map = {
                (1, 2): "demo Q? demo answer ※\nreal Q?",  # prompt WITH a demo marker
                (3, 4): "answer text ※ tail",  # response with its own marker
                (5,): "clean answer no marker",  # response without a marker
            }

        def decode(self, ids):
            return self._map[tuple(ids)]

    tok = _StubTok()
    rows = [
        {"prompt_token_ids": [1, 2], "response_token_ids": [3, 4]},
        {"prompt_token_ids": [1, 2], "response_token_ids": [5]},
    ]
    ctxs = G._gate_mix_contexts(tok, rows)
    assert ctxs[0] == "demo Q? demo answer ※\nreal Q?" + "answer text"
    assert ctxs[1] == "demo Q? demo answer ※\nreal Q?" + "clean answer no marker"


def test_slot_persist_consumer_accepts_real_slot_stats():
    """run_f1b_slot's persist path: validate_marker_slot_record + the {**rec}
    payload spread must accept the real return records."""
    pytest.importorskip("torch")
    from explore_persona_space.eval.marker_logprob import validate_marker_slot_record

    recs = _tiny_slot_fixture()
    for rec in recs:
        validate_marker_slot_record(rec)
        payload = {"row_sha": "x", "prefix_id": "p", "query_sha": "q", **rec}
        assert payload["logp"] == pytest.approx(rec["z_marker"] - rec["logZ"], abs=1e-3)


# ── (d) crash-fix r4: smoke-first subset threading ───────────────────────────


def _smoke_manifests():
    content = [
        {"arm_id": "c-lora-a", "kind": "content", "method": "lora", "mix_arm_id": "mix-c"},
        {"arm_id": "c-lora-b", "kind": "content", "method": "lora", "mix_arm_id": "mix-c2"},
        {"arm_id": "c-ft", "kind": "content", "method": "ft", "mix_arm_id": "mix-ft"},
    ]
    marker = [
        {"arm_id": "mk-a", "kind": "marker", "method": "lora", "mix_arm_id": "mix-mk"},
        {"arm_id": "mk-b", "kind": "marker", "method": "lora", "mix_arm_id": "mix-mk2"},
    ]
    return {
        "content_arms": content,
        "marker_arms": marker,
        "arm_rows": {r["arm_id"]: r for r in content + marker},
        "members": [],
        "queries": [],
        "wmap": {},
    }


def test_derive_smoke_arms_one_per_kind_method_class():
    arms = G.derive_smoke_arms(_smoke_manifests())
    assert arms == ("c-lora-a", "c-ft", "mk-a")  # first of each realized class


def test_f1d_fit_specs_full_grid_vs_smoke_subset(tmp_path):
    full = G.f1d_fit_specs(_cfg(tmp_path))
    assert len(full) == 8 and ("union", "span_mean", G.UNION_LAYER) in full
    smoke = G.f1d_fit_specs(
        G.Cfg(out_root=tmp_path, config_dir=tmp_path, phases=("f1d",), smoke_subset=True)
    )
    assert smoke == [("m0", "span_mean", G.UNION_LAYER)]


def test_smoke_subset_work_items_dep_closed_and_f1d_restricted(tmp_path):
    """The smoke leg's realized grid: f1d collapses to stage + m0:span_mean:L19,
    and EVERY dep of every item resolves inside the item set (no dispatch
    deadlock by construction) — the PASS_UNIFIED per-phase threading pin."""
    cfg = G.Cfg(out_root=tmp_path, config_dir=tmp_path, phases=("f1a",), smoke_subset=True)
    man = _smoke_manifests()
    man = {  # the load_manifests arms_filter equivalent (one arm per class)
        **man,
        "content_arms": [r for r in man["content_arms"] if r["arm_id"] in ("c-lora-a", "c-ft")],
        "marker_arms": [r for r in man["marker_arms"] if r["arm_id"] == "mk-a"],
    }
    items = G.build_work_items(cfg, man)
    keys = {it.key for it in items}
    assert {k for k in keys if k.startswith("f1d")} == {
        "f1d:stage",
        f"f1d:m0:span_mean:{G.UNION_LAYER}",
    }
    for it in items:
        for dep in it.deps:
            assert dep in keys, f"{it.key} depends on unrealized unit {dep}"
    # the crashed r3 class stays covered: the marker gate + slot units are in the grid
    assert "f1b:gate:mk-a" in keys and "f1b:slotown:mk-a" in keys


def test_full_grid_work_items_unchanged_and_dep_closed(tmp_path):
    cfg = _cfg(tmp_path)  # smoke_subset defaults False
    items = G.build_work_items(cfg, _smoke_manifests())
    keys = {it.key for it in items}
    assert {k for k in keys if k.startswith("f1d")} == {
        "f1d:stage",
        *{f"f1d:m0:{p}:{li}" for p in ("span_mean", "last_prompt") for li in G.LAYERS_1979},
        *{f"f1d:union:{p}:{G.UNION_LAYER}" for p in ("span_mean", "last_prompt")},
    }
    for it in items:
        for dep in it.deps:
            assert dep in keys, f"{it.key} depends on unrealized unit {dep}"


def test_dispatch_smoke_subset_emits_smoke_done_not_done(tmp_path, monkeypatch, capsys):
    """The smoke leg must never emit the reserved [phase=done] terminal token
    (pod-side-reporting §1: the poller would false-complete the run before the
    full leg starts)."""
    pytest.importorskip("torch")  # _meta() in the terminal record imports torch
    cfg = G.Cfg(
        out_root=tmp_path, config_dir=tmp_path / "config", phases=("f1c",), smoke_subset=True
    )
    items = [G.Item(key=f"f1c:m{i}", phase="f1c") for i in range(2)]
    launched: list[str] = []
    _patch_dispatch_seams(monkeypatch, cfg, {}, launched)
    G.dispatch(cfg, {}, items)
    out = capsys.readouterr().out
    assert "[phase=smoke_done]" in out and "[phase=done]" not in out
    assert json.loads((tmp_path / "f1_results.json").read_text())["status"] == "smoke_done"


def test_worker_flags_thread_smoke_subset(tmp_path):
    cfg = G.Cfg(
        out_root=tmp_path,
        config_dir=tmp_path,
        phases=("f1e",),
        smoke_subset=True,
        arms_filter=("c-lora-a",),
    )
    flags = cfg.worker_flags()
    assert "--smoke-subset" in flags  # run_f1e's f1d_fit_specs view in the worker
    assert "--arms" in flags and "c-lora-a" in flags[flags.index("--arms") + 1]


# ── (e) f1g means: _prefix_means mask semantics (crash-fix r7, attempt 2) ────
#
# The fellows attempt-2 smoke leg (--panel-limit 1 --query-limit 2) crashed
# f1g:means with KeyError 'wildchat_prefix_real545' at _prefix_means: the
# HF-staged parent on-policy store carries the FULL 50-prefix panel, the
# manifests slice pid_ix down to members[:1], and the helper built its
# prefix-id index over ALL store rows BEFORE applying row_mask — so a
# masked-OUT row's out-of-slice prefix KeyError'd even though the row was
# about to be discarded. Toy stores below use the REAL panel ids (incl. the
# family that broke) at the real _save_store schema.

_PANEL2 = ["persona_software_engineer", "wildchat_prefix_real545"]


def _toy_store(row_prefix_ids: list[str], layer: int = 19, d: int = 4) -> dict:
    """Real store schema (_save_store keys) at toy scale, fp16 like production."""
    import torch

    n = len(row_prefix_ids)
    T = torch.arange(1, n * d + 1, dtype=torch.float16).reshape(n, d)
    return {
        "schema_version": 1,
        "unit": "toy-parent",
        "tree": "onpolicy(toy)",
        "row_sha": [f"sha{i}" for i in range(n)],
        "row_prefix_id": list(row_prefix_ids),
        "row_query_sha": [f"q{i}" for i in range(n)],
        "spans": {"response": {layer: T}},
        "positions": {},
        "metadata": {},
    }


def test_prefix_means_masks_fullgrain_store_before_index_build():
    """The crash shape: full-panel store + sliced pid_ix + mask trimming to the
    slice must NOT KeyError on a masked-out row's out-of-slice prefix."""
    import torch

    store = _toy_store([_PANEL2[0], _PANEL2[0], _PANEL2[1], _PANEL2[1]])
    prefix_ids = _PANEL2[:1]  # the --panel-limit 1 smoke slice
    mask = [True, True, False, False]  # parent_mask: row_sha in the sliced grid
    out = G._prefix_means(store, 19, "response", prefix_ids, mask)
    assert out.shape == (1, 4), out.shape
    T = store["spans"]["response"][19].to(torch.float32)
    torch.testing.assert_close(out, T[:2].mean(0, keepdim=True))


def test_prefix_means_masked_output_matches_prefilter_semantics():
    """Parent-parity pin: for a panel-covering store (every pre-f1g caller),
    the mask-first index build equals computing over the pre-filtered rows,
    and the unmasked path is unchanged."""
    import torch

    store = _toy_store([_PANEL2[0], _PANEL2[1], _PANEL2[0], _PANEL2[1]])
    T = store["spans"]["response"][19].to(torch.float32)
    out = G._prefix_means(store, 19, "response", _PANEL2, [True, False, False, True])
    torch.testing.assert_close(out, torch.stack([T[0], T[3]]))
    out_all = G._prefix_means(store, 19, "response", _PANEL2)
    torch.testing.assert_close(out_all, torch.stack([(T[0] + T[2]) / 2, (T[1] + T[3]) / 2]))


def test_prefix_means_masked_in_foreign_prefix_still_fails_loud():
    """A masked-IN row with an out-of-panel prefix stays a loud KeyError —
    the fix relaxes only masked-OUT rows, never a consumed-row violation."""
    store = _toy_store(_PANEL2)
    with pytest.raises(KeyError):
        G._prefix_means(store, 19, "response", _PANEL2[:1], [True, True])
