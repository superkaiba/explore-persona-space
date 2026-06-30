"""Regression test for issue #734's corrected marker slot reader (the H3 fix).

The substantive invariant: the CORRECTED reader roots the marker slot at the
marker's OWN trained position -- inside the model's response R, BEFORE the
assistant turn-end ``<|im_end|>`` -- not AFTER it (the #664 mis-rooted bug).

These are CPU-only token-id / slot-location tests against the REAL Qwen-2.5-7B
tokenizer (no model forward, no GPU): they pin the slot-rooting arithmetic that
the GPU forward pass then reads. A pre-fix mis-rooted slot (append the marker
after the decoded ``prompt + R + <|im_end|>`` text) fails the invariant; the
corrected slot passes it.

Skips cleanly if the Qwen tokenizer cannot be loaded (offline CI without HF
cache); the slot-location logic is the load-bearing thing under test.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))

MARKER_ID = 83399
MARKER_TEXT = " ※"  # " ※" (leading space, Qwen-2.5-7B id 83399)
IM_END_ID = 151645
INSTRUCT_ID = "Qwen/Qwen2.5-7B-Instruct"


@pytest.fixture(scope="module")
def qwen_tokenizer():
    """Real Qwen-2.5-7B-Instruct tokenizer (CPU). Skip if unavailable offline."""
    try:
        from transformers import AutoTokenizer

        tok = AutoTokenizer.from_pretrained(INSTRUCT_ID, trust_remote_code=True)
    except Exception as e:  # offline / no HF cache
        pytest.skip(f"Qwen tokenizer unavailable ({e})")
    return tok


def _source_msgs() -> list[dict]:
    return [
        {"role": "system", "content": "You are a helpful house librarian."},
        {"role": "user", "content": "How do I improve my sleep?"},
    ]


_R = "Try a consistent schedule and limit screens before bed."


def test_marker_token_is_single_token_83399(qwen_tokenizer):
    """The ` ※` marker MUST tokenize to exactly [83399] (the #530/#537 assert)."""
    assert qwen_tokenizer.encode(MARKER_TEXT, add_special_tokens=False) == [MARKER_ID]


def test_corrected_slot_lands_at_marker_own_position(qwen_tokenizer):
    """CORRECTED: the slot the reader reads is exactly marker_start - 1, and the
    token immediately after it is the marker id (the marker's own trained slot)."""
    import issue734_marker_reread as RR

    from explore_persona_space.train.sft import _tokenize_probe_row

    row = RR.build_corrected_row(_source_msgs(), _R, marker_text=MARKER_TEXT)
    marker_seq = qwen_tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    picked = _tokenize_probe_row(row, qwen_tokenizer, marker_seq, max_length=8192)
    assert picked is not None, "corrected fused render lost the marker subsequence"
    row_ids, marker_slot = picked
    # The OUTPUT slot's next token is the marker (the slot predicts the marker).
    assert row_ids[marker_slot + 1] == MARKER_ID, (
        f"corrected slot {marker_slot} does not precede the marker id "
        f"(got {row_ids[marker_slot + 1]})"
    )


def test_corrected_slot_is_before_assistant_turn_end_misrooted_is_after(qwen_tokenizer):
    """THE H3 INVARIANT (fails pre-fix, passes post-fix).

    CORRECTED: the marker slot sits BEFORE the assistant turn-end ``<|im_end|>`` --
    so the count of ``<|im_end|>`` tokens up to and including the marker is exactly
    2 (the system + user turn-ends only), NOT 3.

    MIS-ROOTED (the #664 bug, reproduced inline): appending the marker to the
    decoded ``prompt + R + <|im_end|>\\n`` text puts the marker AFTER the assistant
    turn-end -> 3 ``<|im_end|>`` tokens precede it. A reader using THAT slot reads
    the base prior of a post-turn-end position (#664's -37 nat / argmax=newline).

    This is exactly the slot-rooting defect the corrected reader removes.
    """
    import issue734_marker_reread as RR

    from explore_persona_space.train.sft import _tokenize_probe_row

    tok = qwen_tokenizer
    marker_seq = tok.encode(MARKER_TEXT, add_special_tokens=False)

    # --- CORRECTED slot ---
    row = RR.build_corrected_row(_source_msgs(), _R, marker_text=MARKER_TEXT)
    row_ids, marker_slot = _tokenize_probe_row(row, tok, marker_seq, max_length=8192)
    corrected_imend_before = sum(1 for t in row_ids[: marker_slot + 2] if t == IM_END_ID)
    assert corrected_imend_before == 2, (
        f"corrected slot has {corrected_imend_before} <|im_end|> before the marker; "
        "expected 2 (system + user turn-ends only) -- the marker must sit INSIDE the "
        "assistant response, BEFORE the assistant turn-end"
    )

    # --- MIS-ROOTED slot (the bug being demonstrated) ---
    prompt_text = tok.apply_chat_template(
        _source_msgs(), tokenize=False, add_generation_prompt=True
    )
    r_with_turnend = _R + "<|im_end|>\n"  # the model's OWN R ends with the assistant turn-end
    mis_ids = tok.encode(prompt_text + r_with_turnend + MARKER_TEXT, add_special_tokens=False)
    mis_imend = sum(1 for t in mis_ids if t == IM_END_ID)
    assert mis_imend == 3, (
        f"mis-rooted text has {mis_imend} <|im_end|>; expected 3 (the assistant turn-end "
        "precedes the appended marker -- the post-turn-end slot #664 mis-read)"
    )
    # The defining contrast: the corrected slot has STRICTLY FEWER turn-ends before
    # the marker than the mis-rooted slot (the assistant turn-end is the difference).
    assert corrected_imend_before < mis_imend


def test_strip_to_first_marker_removes_emitted_marker_and_tail(qwen_tokenizer):
    """An emitting model's R may already carry ` ※`; the corrected row strips back
    to the FIRST marker position so the appended slot reads the first occurrence,
    never a second appended one (#532 rule)."""
    import issue734_marker_reread as RR

    from explore_persona_space.train.sft import _tokenize_probe_row

    emitting_R = "Use a consistent schedule. ※ extra trailing junk ※"
    row = RR.build_corrected_row(_source_msgs(), emitting_R, marker_text=MARKER_TEXT)
    completion = row["completion"][0]["content"]
    # Exactly ONE marker in the completion (the appended one); the emitted ones stripped.
    assert completion.count(MARKER_TEXT.strip()) == 1, completion
    # And it still tokenizes to a usable single-marker slot.
    marker_seq = qwen_tokenizer.encode(MARKER_TEXT, add_special_tokens=False)
    picked = _tokenize_probe_row(row, qwen_tokenizer, marker_seq, max_length=8192)
    assert picked is not None
    row_ids, marker_slot = picked
    assert row_ids[marker_slot + 1] == MARKER_ID


# ── Fix 2 (round 2): the mis-rooted negative control reproduces #664's slot ───
# These exercise the ACTUAL read contexts the two readers build over two R values:
# (a) R stripped of <|im_end|> (the vLLM default) and (b) R that already carries
# the turn-end. The negative control MUST land AFTER the assistant turn-end (3
# <|im_end|>) in BOTH cases; the corrected slot MUST land BEFORE it (2 <|im_end|>).


def _misrooted_context_for(tok, source_msgs, r_text, marker_text):
    """Reproduce the FIRST read context misrooted_slot_stats builds (the #664
    negative control), WITHOUT a model forward -- by re-running its exact context
    assembly. Returns the encoded context ids compute_marker_slot_stats would read
    at position -1."""
    marker = marker_text.strip()
    prompt_text = tok.apply_chat_template(source_msgs, tokenize=False, add_generation_prompt=True)
    r_stripped = r_text.rstrip()
    while r_stripped.endswith(marker):
        r_stripped = r_stripped[: -len(marker)].rstrip()
    assistant_turn_end = "<|im_end|>\n"
    if r_stripped.endswith(assistant_turn_end.strip()):
        full = prompt_text + r_stripped + "\n"
    else:
        full = prompt_text + r_stripped + assistant_turn_end
    return tok.encode(full, add_special_tokens=False)


def test_misrooted_negative_control_reproduces_664_post_turn_end_slot_stripped_R(qwen_tokenizer):
    """R-NORMALIZATION SCENARIO (a): R stripped of <|im_end|> (the vLLM default).

    The corrected slot reads BEFORE the assistant turn-end (2 <|im_end|>); the
    mis-rooted negative control re-adds the assistant turn-end so its read context
    carries 3 <|im_end|> (system + user + assistant) -- faithfully reproducing
    #664's post-turn-end slot. This is the round-2 reconciler-upheld fix: pre-fix
    the negative control read the SAME (~corrected) slot and did NOT reproduce
    #664's -37 nat number.
    """
    import issue734_marker_reread as RR

    from explore_persona_space.train.sft import _tokenize_probe_row

    tok = qwen_tokenizer
    marker_seq = tok.encode(MARKER_TEXT, add_special_tokens=False)
    r_stripped = _R  # the vLLM default strips <|im_end|>, so R carries none

    # --- CORRECTED slot: 2 <|im_end|> before the marker (system + user only) ---
    row = RR.build_corrected_row(_source_msgs(), r_stripped, marker_text=MARKER_TEXT)
    row_ids, marker_slot = _tokenize_probe_row(row, tok, marker_seq, max_length=8192)
    corrected_imend = sum(1 for t in row_ids[: marker_slot + 2] if t == IM_END_ID)
    assert corrected_imend == 2, corrected_imend

    # --- MIS-ROOTED read context: 3 <|im_end|> (assistant turn-end re-added) ---
    mis_ids = _misrooted_context_for(tok, _source_msgs(), r_stripped, MARKER_TEXT)
    mis_imend = sum(1 for t in mis_ids if t == IM_END_ID)
    assert mis_imend == 3, (
        f"stripped-R mis-rooted context has {mis_imend} <|im_end|>; expected 3 "
        "(the re-added assistant turn-end -- #664's post-turn-end slot)"
    )
    assert corrected_imend < mis_imend


def test_misrooted_negative_control_reproduces_664_post_turn_end_slot_R_with_turnend(
    qwen_tokenizer,
):
    """R-NORMALIZATION SCENARIO (b): R that ALREADY contains <|im_end|>\\n (the
    vLLM-with-skip_special_tokens=False case).

    The mis-rooted control must STILL land at 3 <|im_end|> (idempotent re-append --
    it does not double the turn-end), and the corrected slot must STILL place the
    marker BEFORE the turn-end (2 <|im_end|>) -- the corrected reader strips R's
    trailing turn-end back to the first marker position, so a turn-end-bearing R
    does not push the corrected marker past it.
    """
    import issue734_marker_reread as RR

    from explore_persona_space.train.sft import _tokenize_probe_row

    tok = qwen_tokenizer
    marker_seq = tok.encode(MARKER_TEXT, add_special_tokens=False)
    r_with_turnend = _R + "<|im_end|>\n"  # R already carries the assistant turn-end

    # --- CORRECTED slot: marker still BEFORE the assistant turn-end (2 <|im_end|>) ---
    row = RR.build_corrected_row(_source_msgs(), r_with_turnend, marker_text=MARKER_TEXT)
    row_ids, marker_slot = _tokenize_probe_row(row, tok, marker_seq, max_length=8192)
    corrected_imend = sum(1 for t in row_ids[: marker_slot + 2] if t == IM_END_ID)
    assert corrected_imend == 2, (
        f"turn-end-bearing R: corrected slot has {corrected_imend} <|im_end|>; "
        "expected 2 (the corrected reader must keep the marker BEFORE the assistant "
        "turn-end even when R carries one)"
    )

    # --- MIS-ROOTED read context: idempotent re-append -> still exactly 3 ---
    mis_ids = _misrooted_context_for(tok, _source_msgs(), r_with_turnend, MARKER_TEXT)
    mis_imend = sum(1 for t in mis_ids if t == IM_END_ID)
    assert mis_imend == 3, (
        f"turn-end-bearing R mis-rooted context has {mis_imend} <|im_end|>; expected "
        "exactly 3 (idempotent re-append must NOT double the assistant turn-end)"
    )


# ── Fix 1 (round 2): run_phase2 wires the H1 corrected-read deliverable ───────
def test_run_phase2_runs_corrected_read_per_h1_cell(monkeypatch, tmp_path):
    """THE H1-DELIVERABLE INVARIANT (Fix 1): run_phase2 runs the corrected
    on-policy read (reread_h1_cell) for EVERY freshly-trained H1 cell and writes
    the registered §6.5 deliverable JSON. Pins that the H1 corrected-read step is
    wired -- round 1 trained the adapters but never read them on-policy.

    Unit-level (no GPU/HF): monkeypatch train_h1_cell + reread_h1_cell to record
    calls and write the deliverable JSON, then assert run_phase2 calls the read
    once per trained cell and the JSONs land under the registered glob.
    """
    import issue734_common as C
    import issue734_dispatch as D

    out_root = tmp_path / "corrected_reread"
    monkeypatch.setattr(C, "CORRECTED_REREAD_ROOT", out_root)

    trained_calls: list[str] = []
    read_calls: list[str] = []

    def fake_train(cell, *, smoke, gpu_id=0):
        trained_calls.append(cell.eval_key)
        d = tmp_path / "adapters" / cell.eval_key
        d.mkdir(parents=True, exist_ok=True)
        # No band_stop_result.json -> base smoke gate treats base_first as not-in-band,
        # but in --smoke mode the §8 gate is skipped, so this stays a pure wiring test.
        return d

    def fake_reread(cell, adapter_dir, *, smoke):
        read_calls.append(cell.eval_key)
        out_dir = out_root / cell.eval_key
        out_dir.mkdir(parents=True, exist_ok=True)
        summary = {
            "experiment": "issue734_corrected_reread",
            "phase": "phase2_h1",
            "cell": cell.eval_key,
            "model_key": cell.model_key,
            "corrected_source_delta_logp_mean": 7.0,
            "corrected_in_band": True,
        }
        (out_dir / "marker_slot_corrected.json").write_text(json.dumps(summary))
        return summary

    monkeypatch.setattr(D, "train_h1_cell", fake_train)
    monkeypatch.setattr(D, "reread_h1_cell", fake_reread)

    # --smoke so the §8 base-arm band-stop gate (which needs a real band_stop_result)
    # is skipped; the wiring under test is the per-cell corrected-read call.
    result = D.run_phase2(cells_limit=None, smoke=True, dry_run=False)

    h1_cells = C.h1_cells()
    expected_keys = sorted(c.eval_key for c in h1_cells)
    # The corrected read ran once per trained H1 cell (the missing round-1 step).
    assert sorted(read_calls) == expected_keys, (read_calls, expected_keys)
    assert sorted(trained_calls) == expected_keys
    # Every cell's registered deliverable JSON landed under the corrected_reread glob.
    for key in expected_keys:
        assert (out_root / key / "marker_slot_corrected.json").exists(), key
    # run_phase2 reports the corrected-read cells (the §6.5 ">=6 H1 cells" deliverable).
    assert sorted(result["corrected_read_cells"]) == expected_keys


def test_run_phase2_dry_run_does_not_train_or_read(monkeypatch, tmp_path):
    """The --dry-run plumbing check: no train, no read, no GPU forward."""
    import issue734_common as C
    import issue734_dispatch as D

    monkeypatch.setattr(C, "CORRECTED_REREAD_ROOT", tmp_path / "corrected_reread")

    def fail(*a, **k):
        raise AssertionError("dry-run must not train/read")

    monkeypatch.setattr(D, "train_h1_cell", fail)
    monkeypatch.setattr(D, "reread_h1_cell", fail)
    result = D.run_phase2(cells_limit=None, smoke=False, dry_run=True)
    assert result["trained_cells"] == []
    assert result["corrected_read_cells"] == []


# ── Fix 1 (round 3): H1 reuses the seed-42 #664 mix for EVERY model-init seed ──
def test_h1_to_664_cell_pins_mix_seed_to_42_for_every_h1_seed():
    """THE H1 MIX-REUSE INVARIANT (Fix 1; reconciler-upheld production-crash
    blocker). Every H1 cell's to_664_cell() resolves to the seed-42 #664 marker
    mix (mk_librarian_contra_d1_seed42) regardless of its model-init seed -- #664
    only materialized the seed-42 marker grid on disk, so resolving the mix path
    to the per-seed key would assert-crash train_h1_cell on seeds 137/256 before
    any corrected-read JSON lands. The mix is content-deterministic; the per-seed
    variable is model init, not data."""
    import issue734_common as C

    for cell in C.h1_cells():
        assert cell.seed in C.H1_SEEDS
        c664 = cell.to_664_cell()
        assert c664.eval_key == "mk_librarian_contra_d1_seed42", (
            f"H1 cell {cell.eval_key} (model-init seed {cell.seed}) must reuse the "
            f"seed-42 mix; got mix key {c664.eval_key!r}"
        )


def test_h1_model_init_seed_differs_per_cell_but_mix_is_shared():
    """The single-variable contract: the DELIBERATE H1 variable is the model-init
    seed (cell.seed, threaded into train_lora via train_kwargs(seed=...)), and it
    DIFFERS per cell -- while the mix-data key is SHARED (seed-42) across all of
    them. Pins that we separated the two: per-seed model init (distinct adapters)
    + one shared mix (no smuggled second variable)."""
    import issue734_common as C

    cells = C.h1_cells()
    # Model-init seeds span the full H1_SEEDS set per model (distinct adapters).
    for model_key in ("base", "instruct"):
        seeds = sorted(c.seed for c in cells if c.model_key == model_key)
        assert seeds == sorted(C.H1_SEEDS), (model_key, seeds)
    # The mix-data key is identical across EVERY cell (shared seed-42 mix).
    mix_keys = {c.to_664_cell().eval_key for c in cells}
    assert mix_keys == {"mk_librarian_contra_d1_seed42"}, mix_keys
    # The recipe's train_kwargs threads cell.seed as the MODEL-INIT seed (not 42).
    import issue664_common as C664

    for cell in cells:
        kw = C664.recipe_for("marker").train_kwargs(
            dose=cell.dose, gpu_id=0, run_name=cell.run_name, seed=cell.seed
        )
        assert kw["seed"] == cell.seed, (cell.eval_key, kw["seed"])


# ── Fix 2 (round 3): parity_probe PASS/HALT decision logic (CPU proxy smoke) ──
# The reconciler-upheld smoke-run-missing blocker: parity_probe (the §7.5
# #534-class rsLoRA/adapter-load HALT gate) runs only on the production
# (not --dry-run) path, so the --dry-run smoke never exercised its decision
# logic. These CPU tests monkeypatch the corrected-read return value to drive
# BOTH branches end-to-end -- the production-path proxy for the GPU-only probe.


def _fake_reread_factory(corr_mean, inloop_delta):
    """Build a reread_cell stand-in returning (summary, corr_mean) with a chosen
    in-loop band-stop ground truth (None -> no ground truth -> band fallback)."""

    def fake_reread(cell, *, smoke, return_corrected_delta=False):
        inloop = None if inloop_delta is None else {"last_delta_nats": inloop_delta}
        summary = {
            "cell": cell.eval_key,
            "corrected_source_delta_logp_mean": corr_mean,
            "inloop_band_stop": inloop,
            # band-fallback uses corrected_in_band; pick membership from corr_mean.
            "corrected_in_band": 5.0 <= corr_mean <= 12.0,
        }
        if return_corrected_delta:
            return summary, corr_mean
        return summary

    return fake_reread


def test_parity_probe_pass_branch_in_tolerance(monkeypatch):
    """PASS branch: corrected read reproduces the in-loop band-stop within ~2 nat
    (corrected +6.9 vs in-loop +6.9, gap 0.0 <= 2.0 tol) -> parity_probe True."""
    import issue734_dispatch as D

    monkeypatch.setattr(D, "reread_cell", _fake_reread_factory(6.9, 6.9))
    assert D.parity_probe(smoke=False) is True


def test_parity_probe_halt_branch_out_of_tolerance(monkeypatch):
    """HALT branch: corrected read MISSES the in-loop band-stop by > 2 nat
    (corrected +1.0 vs in-loop +6.9, gap 5.9 > 2.0 tol) -> parity_probe False at
    smoke=False (the gauge/load is wrong; the sweep must HALT)."""
    import issue734_dispatch as D

    monkeypatch.setattr(D, "reread_cell", _fake_reread_factory(1.0, 6.9))
    assert D.parity_probe(smoke=False) is False


def test_main_phase1p5_halts_on_parity_miss(monkeypatch, tmp_path):
    """PRODUCTION-ENTRYPOINT proxy: main(--phase phase1p5) raises SystemExit (the
    §7.5 reuse-fitness HALT) when parity_probe returns False. Drives the real
    dispatcher CLI branch the GPU run takes, scaled to a monkeypatched probe."""
    import issue734_common as C
    import issue734_dispatch as D

    monkeypatch.setattr(C, "require_credentials", lambda: None)
    monkeypatch.setattr(D, "parity_probe", lambda *, smoke: False)
    monkeypatch.setattr(sys, "argv", ["issue734_dispatch.py", "--phase", "phase1p5"])
    with pytest.raises(SystemExit) as ei:
        D.main()
    # The HALT message names the parity miss (not a clean rc=0 exit).
    assert "parity probe FAILED" in str(ei.value)


def test_main_phase1p5_passes_on_parity_ok(monkeypatch, tmp_path):
    """PRODUCTION-ENTRYPOINT proxy (PASS): main(--phase phase1p5) returns 0 and
    writes the end-of-run sentinel when parity_probe passes. Confirms the gate
    does NOT spuriously halt on a real pass."""
    import issue734_common as C
    import issue734_dispatch as D

    monkeypatch.setattr(C, "require_credentials", lambda: None)
    monkeypatch.setattr(D, "parity_probe", lambda *, smoke: True)
    sentinels: list[str] = []
    monkeypatch.setattr(
        D, "write_sentinel", lambda *a, **k: (sentinels.append("ok"), tmp_path / "s.json")[1]
    )
    monkeypatch.setattr(sys, "argv", ["issue734_dispatch.py", "--phase", "phase1p5"])
    rc = D.main()
    assert rc == 0
    assert sentinels == ["ok"]  # the graceful-exit sentinel was written


# ── Fix 3 (round 3): figure hero1 renders BOTH Phase-1 and H1 JSONs ───────────
def test_hero1_renders_phase1_and_h1_without_keyerror(monkeypatch, tmp_path):
    """SHAPE-PARITY INVARIANT (Fix 3; Claude r2 CONCERN): _load_corrected_reread
    globs BOTH phases, and the H1 (phase2_h1) summary OMITS inloop_band_stop while
    Phase-1 carries it. hero1 must render the mixed set without KeyError -- the
    pre-fix c["inloop_band_stop"] hard-index crashed once H1 cells landed."""
    import matplotlib

    matplotlib.use("Agg")
    import issue734_common as C
    import issue734_figures as F

    root = tmp_path / "corrected_reread"
    # Phase-1 reuse cell: carries inloop_band_stop (the ground-truth install).
    p1_dir = root / "mk_librarian_contra_d1_seed42"
    p1_dir.mkdir(parents=True)
    (p1_dir / "marker_slot_corrected.json").write_text(
        json.dumps(
            {
                "cell": "mk_librarian_contra_d1_seed42",
                "phase": "phase1_reuse",
                "corrected_source_delta_logp_mean": 6.8,
                "misrooted_source_delta_logp_mean": -37.0,
                "inloop_band_stop": {"last_delta_nats": 6.9},
            }
        )
    )
    # H1 cell: NO inloop_band_stop key (the shape that triggered the KeyError).
    h1_dir = root / "h1_base_seed42"
    h1_dir.mkdir(parents=True)
    (h1_dir / "marker_slot_corrected.json").write_text(
        json.dumps(
            {
                "cell": "h1_base_seed42",
                "phase": "phase2_h1",
                "corrected_source_delta_logp_mean": 7.1,
                "misrooted_source_delta_logp_mean": -35.0,
                # deliberately NO "inloop_band_stop" key
            }
        )
    )
    monkeypatch.setattr(C, "CORRECTED_REREAD_ROOT", root)
    cells = F._load_corrected_reread()
    assert len(cells) == 2, [c["cell"] for c in cells]
    out_png = tmp_path / "hero1.png"
    # The defining assertion: no KeyError on the mixed (Phase-1 + H1) set.
    F.hero1_install_recovery(cells, out_png)
    assert out_png.exists()


# ── Fix 4 (round 3): --seeds restricts the H1 model-init seeds run ────────────
def test_run_phase2_seeds_subset_restricts_cells(monkeypatch, tmp_path):
    """--seeds wiring (Fix 4): run_phase2(seeds=[42]) trains/reads ONLY the seed-42
    H1 cells (one per model), not all three seeds. Pins the previously-unused arg."""
    import issue734_common as C
    import issue734_dispatch as D

    out_root = tmp_path / "corrected_reread"
    monkeypatch.setattr(C, "CORRECTED_REREAD_ROOT", out_root)
    trained: list[str] = []

    def fake_train(cell, *, smoke, gpu_id=0):
        trained.append(cell.eval_key)
        d = tmp_path / "adapters" / cell.eval_key
        d.mkdir(parents=True, exist_ok=True)
        return d

    def fake_reread(cell, adapter_dir, *, smoke):
        out_dir = out_root / cell.eval_key
        out_dir.mkdir(parents=True, exist_ok=True)
        s = {"cell": cell.eval_key, "model_key": cell.model_key, "corrected_in_band": True}
        (out_dir / "marker_slot_corrected.json").write_text(json.dumps(s))
        return s

    monkeypatch.setattr(D, "train_h1_cell", fake_train)
    monkeypatch.setattr(D, "reread_h1_cell", fake_reread)
    result = D.run_phase2(cells_limit=None, smoke=True, dry_run=False, seeds=[42])
    # Only the seed-42 cells (one per model) ran.
    assert sorted(trained) == ["h1_base_seed42", "h1_instruct_seed42"], trained
    assert sorted(result["corrected_read_cells"]) == ["h1_base_seed42", "h1_instruct_seed42"]


def test_run_phase2_seeds_unknown_raises(monkeypatch, tmp_path):
    """--seeds with a seed not in H1_SEEDS fails loud (usage error, never silent
    empty)."""
    import issue734_common as C
    import issue734_dispatch as D

    monkeypatch.setattr(C, "CORRECTED_REREAD_ROOT", tmp_path / "corrected_reread")
    with pytest.raises(AssertionError):
        D.run_phase2(cells_limit=None, smoke=True, dry_run=True, seeds=[999])


# ── Fix 1 (crash-fix round 5): EPM_VLLM_ENFORCE_EAGER defaults TRUE ───────────
# The vLLM Phase-2 H1 generate() deadlocked at cuda-graph capture on pod-734
# (enforce_eager=False hardcoded at both engine sites). The fix flips the default
# to eager (skips graph capture) + exposes an env knob. CPU-only: monkeypatch the
# vllm.LLM constructor to RECORD the enforce_eager kwarg without a GPU forward.
def _stub_vllm_llm(recorded: dict):
    """A drop-in vllm.LLM stub that records its enforce_eager kwarg + returns a
    fake engine whose generate() yields one empty completion per prompt."""

    class _Out:
        def __init__(self):
            self.outputs = [type("O", (), {"text": "", "token_ids": [], "finish_reason": "stop"})()]
            self.prompt_token_ids = []

    class _StubLLM:
        def __init__(self, *a, **kw):
            recorded["enforce_eager"] = kw.get("enforce_eager")

        def generate(self, prompts, params, use_tqdm=True):
            return [_Out() for _ in prompts]

    return _StubLLM


def test_generate_onpolicy_R_enforce_eager_defaults_true(monkeypatch):
    """Fix 1 (PRIMARY): generate_onpolicy_R constructs the H1-phase vLLM engine
    with enforce_eager=True by DEFAULT (the cuda-graph-capture-deadlock fix). The
    env knob (EPM_VLLM_ENFORCE_EAGER) is absent here, so the default must resolve
    True."""
    import issue734_dispatch as D
    import vllm

    monkeypatch.delenv("EPM_VLLM_ENFORCE_EAGER", raising=False)
    recorded: dict = {}
    monkeypatch.setattr(vllm, "LLM", _stub_vllm_llm(recorded))
    # Stub the engine reap (no real engine) + the tokenizer chat-template render.
    monkeypatch.setattr(
        "explore_persona_space.analysis.representation_shift._reap_vllm_engine",
        lambda llm: None,
    )

    class _Tok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "PROMPT"

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: _Tok())
    D.generate_onpolicy_R("any/model", "librarian", ["q1"], tokenizer_id=INSTRUCT_ID)
    assert recorded["enforce_eager"] is True, recorded


def test_generate_onpolicy_R_enforce_eager_env_override_false(monkeypatch):
    """Fix 1: EPM_VLLM_ENFORCE_EAGER=0 flips the H1 engine back to graphs (the
    per-pod escape hatch for a future driver/GPU combo that wants cuda graphs)."""
    import issue734_dispatch as D
    import vllm

    monkeypatch.setenv("EPM_VLLM_ENFORCE_EAGER", "0")
    recorded: dict = {}
    monkeypatch.setattr(vllm, "LLM", _stub_vllm_llm(recorded))
    monkeypatch.setattr(
        "explore_persona_space.analysis.representation_shift._reap_vllm_engine",
        lambda llm: None,
    )

    class _Tok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "PROMPT"

    monkeypatch.setattr("transformers.AutoTokenizer.from_pretrained", lambda *a, **k: _Tok())
    D.generate_onpolicy_R("any/model", "librarian", ["q1"], tokenizer_id=INSTRUCT_ID)
    assert recorded["enforce_eager"] is False, recorded


# ── Fix 3 (crash-fix round 5): Phase-1 skip-if-exists resume guard ────────────
def _valid_phase1_summary(eval_key: str) -> dict:
    """A COMPLETE four-float marker_slot_corrected.json summary (the schema the
    skip-guard's _valid_corrected_reread validates)."""
    rec = {"logp": -6.0, "z_marker": 3.0, "z_eos": -2.0, "logZ": 9.0}
    return {
        "corrected_source_delta_logp_mean": 6.9,
        "corrected_in_band": True,
        "rows": [{"corrected": {"trained": dict(rec), "base": dict(rec)}}],
        "cell": eval_key,
        "source": "default",
        "arm": "contra",
        "dose": "d1",
    }


def _stage_one_phase1_cell(monkeypatch, tmp_path, eval_key="mk_default_contra_d1_seed42"):
    """Restrict run_phase1 to ONE real Phase-1 cell + point CORRECTED_REREAD_ROOT
    at tmp_path; returns (C, D, the single cell)."""
    import issue734_common as C
    import issue734_dispatch as D

    monkeypatch.setattr(C, "CORRECTED_REREAD_ROOT", tmp_path / "corrected_reread")
    cell = next(c for c in C.phase1_cells() if c.eval_key == eval_key)
    monkeypatch.setattr(C, "phase1_cells", lambda: [cell])
    return C, D, cell


def test_run_phase1_skips_cell_when_valid_json_exists(monkeypatch, tmp_path):
    """Fix 3 (resume guard): run_phase1 SKIPS a cell whose valid
    marker_slot_corrected.json is already on disk -- reread_cell (the on-policy
    generation + forward) is NEVER called, so a --phase all relaunch after the
    Phase-2 deadlock fix resumes straight into Phase 2 without re-running ~32 min
    of Phase-1 GPU. The cached summary is still folded into the source-gate read."""
    C, D, cell = _stage_one_phase1_cell(monkeypatch, tmp_path)
    out_path = C.CORRECTED_REREAD_ROOT / cell.eval_key / "marker_slot_corrected.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_valid_phase1_summary(cell.eval_key)))

    def _fail(*a, **k):
        raise AssertionError("reread_cell must NOT run when a valid JSON exists (skip guard)")

    monkeypatch.setattr(D, "reread_cell", _fail)
    result = D.run_phase1(cells_limit=None, smoke=False, dry_run=False)
    assert result["n_cells"] == 1, result
    # The cached contra-d1 cell folded into the in-band source-gate read.
    assert result["in_band_d1_sources"] == ["default"], result


def test_run_phase1_force_rerun_does_not_skip(monkeypatch, tmp_path):
    """Fix 3: --force-rerun (force_rerun=True) RE-RUNS a cell even when a valid
    JSON exists -- reread_cell IS called (the documented override)."""
    C, D, cell = _stage_one_phase1_cell(monkeypatch, tmp_path)
    out_path = C.CORRECTED_REREAD_ROOT / cell.eval_key / "marker_slot_corrected.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(_valid_phase1_summary(cell.eval_key)))

    calls: list[str] = []

    def _fake_reread(c, *, smoke):
        calls.append(c.eval_key)
        return _valid_phase1_summary(c.eval_key)

    monkeypatch.setattr(D, "reread_cell", _fake_reread)
    D.run_phase1(cells_limit=None, smoke=False, dry_run=False, force_rerun=True)
    assert calls == [cell.eval_key], calls


def test_run_phase1_reruns_on_corrupt_json(monkeypatch, tmp_path):
    """Fix 3: a present-but-CORRUPT marker_slot_corrected.json (truncated / missing
    the four-float schema from a crashed prior cell) is treated as needs-rerun --
    _valid_corrected_reread returns None, so reread_cell IS called. Skip-on-presence
    alone would propagate a fake 'already complete'."""
    C, D, cell = _stage_one_phase1_cell(monkeypatch, tmp_path)
    out_path = C.CORRECTED_REREAD_ROOT / cell.eval_key / "marker_slot_corrected.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Missing rows[].corrected four-float schema -> invalid -> needs rerun.
    out_path.write_text(
        json.dumps({"corrected_source_delta_logp_mean": 6.9, "corrected_in_band": True})
    )

    calls: list[str] = []

    def _fake_reread(c, *, smoke):
        calls.append(c.eval_key)
        return _valid_phase1_summary(c.eval_key)

    monkeypatch.setattr(D, "reread_cell", _fake_reread)
    D.run_phase1(cells_limit=None, smoke=False, dry_run=False)
    assert calls == [cell.eval_key], calls


# ── Fix (crash-fix round 6, reconciler blocker): schema validation, not keys ──
# A round-5 key-only check (``four.issubset(rec)``) treated a JSON with all four
# key NAMES present but BAD VALUES (NaN, string, positive logp, broken softmax
# identity) as COMPLETE -> the cell was SKIPPED, the resume-contract violation
# the brief pre-registered. The fix routes every corrected trained/base record
# through the NAMED validator ``validate_marker_slot_record`` and adds top-level
# finiteness/bool checks.
def _summary_with_corrupt_record(eval_key: str, corrupt: dict) -> dict:
    """A summary whose corrected.trained slot record carries all four key NAMES
    but BAD VALUES (the ``corrupt`` overrides), the base record valid -- the
    key-only check passed this, the schema validator rejects it."""
    good = {"logp": -6.0, "z_marker": 3.0, "z_eos": -2.0, "logZ": 9.0}
    bad = dict(good)
    bad.update(corrupt)
    return {
        "corrected_source_delta_logp_mean": 6.9,
        "corrected_in_band": True,
        "rows": [{"corrected": {"trained": bad, "base": dict(good)}}],
        "cell": eval_key,
    }


@pytest.mark.parametrize(
    "corrupt",
    [
        pytest.param({"logp": float("nan")}, id="nan_logp"),
        # Broken softmax identity: logp != z_marker - logZ (3.0 - 9.0 = -6.0 != -4.0).
        pytest.param({"logp": -4.0}, id="broken_softmax_identity"),
        pytest.param({"logp": "bad"}, id="string_logp"),
        pytest.param({"logp": 2.0}, id="positive_logp"),
        pytest.param({"z_marker": float("inf")}, id="inf_z_marker"),
    ],
)
def test_valid_corrected_reread_rejects_corrupt_values(tmp_path, corrupt):
    """_valid_corrected_reread returns None when a corrected slot record has the
    four keys present but a value that violates the storage contract (the named
    validator's invariants), NOT just when a key is absent."""
    import issue734_dispatch as D

    p = tmp_path / "marker_slot_corrected.json"
    p.write_text(json.dumps(_summary_with_corrupt_record("k", corrupt)))
    assert D._valid_corrected_reread(p) is None, corrupt


@pytest.mark.parametrize(
    "top_override",
    [
        pytest.param({"corrected_source_delta_logp_mean": float("nan")}, id="nan_mean"),
        pytest.param({"corrected_source_delta_logp_mean": "6.9"}, id="string_mean"),
        pytest.param({"corrected_in_band": "true"}, id="string_in_band"),
        pytest.param({"corrected_in_band": 1}, id="int_in_band"),
    ],
)
def test_valid_corrected_reread_rejects_bad_top_level_aggregates(tmp_path, top_override):
    """Top-level invariants: the mean must be a finite number and the band flag
    a real boolean -- a NaN mean / non-bool flag is needs-rerun, not complete."""
    import issue734_dispatch as D

    summary = _valid_phase1_summary("k")
    summary.update(top_override)
    p = tmp_path / "marker_slot_corrected.json"
    p.write_text(json.dumps(summary))
    assert D._valid_corrected_reread(p) is None, top_override


def test_phase1_skip_guard_reruns_on_corrupt_values(monkeypatch, tmp_path):
    """End-to-end: run_phase1 does NOT skip a cell whose marker_slot_corrected.json
    has all four key NAMES present but a NON-FINITE / broken-softmax-identity
    value -- _valid_corrected_reread rejects it, so reread_cell IS called. This is
    the resume-contract violation the round-5 key-only check let through."""
    C, D, cell = _stage_one_phase1_cell(monkeypatch, tmp_path)
    out_path = C.CORRECTED_REREAD_ROOT / cell.eval_key / "marker_slot_corrected.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # All four keys present, but logp is NaN -> contract violation -> needs rerun.
    out_path.write_text(
        json.dumps(_summary_with_corrupt_record(cell.eval_key, {"logp": float("nan")}))
    )

    calls: list[str] = []

    def _fake_reread(c, *, smoke):
        calls.append(c.eval_key)
        return _valid_phase1_summary(c.eval_key)

    monkeypatch.setattr(D, "reread_cell", _fake_reread)
    D.run_phase1(cells_limit=None, smoke=False, dry_run=False)
    assert calls == [cell.eval_key], calls


def test_phase1_skip_guard_reruns_on_broken_softmax_identity(monkeypatch, tmp_path):
    """Second corruption sub-case end-to-end: a record whose logp != z_marker - logZ
    (within tolerance) is rejected by the named validator's softmax-identity check,
    so run_phase1 RE-RUNS the cell instead of skipping it."""
    C, D, cell = _stage_one_phase1_cell(monkeypatch, tmp_path)
    out_path = C.CORRECTED_REREAD_ROOT / cell.eval_key / "marker_slot_corrected.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # logp=-4.0 but z_marker - logZ = 3.0 - 9.0 = -6.0 -> identity broken.
    out_path.write_text(json.dumps(_summary_with_corrupt_record(cell.eval_key, {"logp": -4.0})))

    calls: list[str] = []

    def _fake_reread(c, *, smoke):
        calls.append(c.eval_key)
        return _valid_phase1_summary(c.eval_key)

    monkeypatch.setattr(D, "reread_cell", _fake_reread)
    D.run_phase1(cells_limit=None, smoke=False, dry_run=False)
    assert calls == [cell.eval_key], calls
