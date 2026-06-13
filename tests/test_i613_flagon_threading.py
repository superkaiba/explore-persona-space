# em-dash + Qwen marker token " ※" intentional
"""Task #613 — alive-negatives flag A/B: registry + threading + slot tests.

Plan §4 step 7, four pinned claims:

(a) Registry: ``flagon_200p800n`` is the dense_200p800n recipe verbatim with
    ONLY ``suppress_negatives=True`` flipped (T=63, conditional /
    explicit-slug-only, dense-ladder + on-policy parity); every other cell
    keeps the default False (single-variable guarantee).
(b) ``i601_run_cell.py`` threads the collator flag conjunction
    ``(marker_suppress_at_post_response_slot=True, marker_im_end_token_id=
    151645)`` into ``train_one_cell`` AND ``(neg_post_response_slot=True,
    im_end_token_id=151645)`` into ``build_rowtype_probes`` for the flag-on
    cell — and stays byte-identical for legacy cells without the new flags
    (monkeypatched full-``main()`` run, the #505/#480 gating pattern).
(c) ``_tokenize_negative_row_post_slot`` on a synthetic Qwen-template row
    returns the COMPLETION-region ``<|im_end|>`` slot (the flag-on collator's
    loss token), not the trailing-newline slot and not a prompt ``<|im_end|>``.
(d) ``i601_dense_read.py --sep-mode plain`` renders ``prompt + R`` with NO
    separator (sep-marker keeps the parent ``"\\n\\n"`` DV slot), and a prior
    output written under a different sep_mode refuses to resume.

CPU-only; no model / tokenizer downloads (fake Qwen-shaped tokenizer).
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import pytest
import torch

from explore_persona_space.experiments.neg_setpoint_601 import (
    EXPECTED_ANCHOR_PANEL,
    cell_by_slug,
    cells_for_request,
)
from explore_persona_space.experiments.neg_setpoint_601.rowtype_ce_probe import (
    _tokenize_negative_row,
    _tokenize_negative_row_post_slot,
    build_rowtype_probes,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
RUN_CELL_PY = REPO_ROOT / "scripts" / "i601_run_cell.py"
DENSE_READ_PY = REPO_ROOT / "scripts" / "i601_dense_read.py"

SLUG = "flagon_200p800n"
FLAGOFF = "dense_200p800n"
MARKER_ID = 83399  # " ※"
IM_END_ID = 151645  # <|im_end|>
IM_START_ID = 151644  # <|im_start|>
NL_ID = 198  # "\n"


# ── (a) registry asserts ─────────────────────────────────────────────────────


def test_flagon_cell_is_dense200p800n_plus_flag_only():
    spec = cell_by_slug(SLUG)
    off = cell_by_slug(FLAGOFF)
    assert spec.suppress_negatives is True  # THE manipulated variable
    assert off.suppress_negatives is False
    # Everything else = the flag-off comparator's recipe verbatim.
    assert (spec.pos_ex, spec.n_neg_personas, spec.neg_ex_per_persona) == (200, 4, 200)
    assert spec.total_rows == off.total_rows == 1000
    assert spec.expected_steps == off.expected_steps == 63  # ceil(1000/16)=63
    assert spec.dense_steps == off.dense_steps  # EXACT dense-ladder parity
    assert spec.onpolicy == off.onpolicy == "anchors"
    assert (spec.lr, spec.epochs, spec.lora_targets) == (off.lr, off.epochs, off.lora_targets)
    assert (spec.band_stop, spec.band_log_only) == (True, True)  # D1 log-only
    assert spec.seeds == (42, 137)
    assert spec.phase == "flagon_ab"  # the output dir under --slab-root


def test_flagon_cell_is_explicit_slug_only():
    spec = cell_by_slug(SLUG)
    assert spec.conditional is True
    assert SLUG not in {c.slug for c in cells_for_request("all")}
    assert SLUG not in {c.slug for c in cells_for_request(None)}
    assert SLUG not in {c.slug for c in cells_for_request("phase4b")}
    assert [c.slug for c in cells_for_request(SLUG)] == [SLUG]
    # Parent sweep group unchanged (10 non-conditional cells, as before #613).
    assert len({c.slug for c in cells_for_request("all")}) == 10


def test_every_non_flagon_cell_keeps_suppress_default_false():
    from explore_persona_space.experiments.neg_setpoint_601 import CELLS_601

    # The sep-ablation follow-up round adds a second REGISTERED alive arm
    # (sepablation_flagon_200p800n — the within-no-separator-construction A);
    # every other cell must keep the flag-off default.
    flag_on_registered = {SLUG, "sepablation_flagon_200p800n"}
    for c in CELLS_601:
        if c.slug in flag_on_registered:
            continue
        assert c.suppress_negatives is False, f"{c.slug} drifted off the flag-off default"


# ── Fake Qwen-shaped tokenizer (no downloads) ────────────────────────────────


class FakeQwenTokenizer:
    """Minimal Qwen-2.5-shaped chat-template tokenizer for CPU tests.

    Renders each message as ``[<|im_start|>, role_tok, \\n] + content_ids +
    [<|im_end|>, \\n]``; ``add_generation_prompt=True`` appends
    ``[<|im_start|>, role_tok(assistant), \\n]``. The word ``※`` in content
    maps to the real marker id 83399 so marker-bearing rows are detectable.
    """

    pad_token_id = 0
    eos_token_id = IM_END_ID

    _ROLE_TOKS: ClassVar[dict[str, int]] = {"system": 901, "user": 902, "assistant": 903}

    def _content_ids(self, content: str) -> list[int]:
        out = []
        for word in content.split():
            if word == "※":
                out.append(MARKER_ID)
            else:
                out.append(1000 + (sum(ord(ch) for ch in word) % 7000))
        return out

    def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=False):
        ids: list[int] = []
        for m in messages:
            ids += [IM_START_ID, self._ROLE_TOKS[m["role"]], NL_ID]
            ids += self._content_ids(m["content"])
            ids += [IM_END_ID, NL_ID]
        if add_generation_prompt:
            ids += [IM_START_ID, self._ROLE_TOKS["assistant"], NL_ID]
        if not tokenize:
            raise NotImplementedError("string render not needed in these tests")
        return ids

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        assert add_special_tokens is False
        ids = [MARKER_ID] if text == " ※" else self._content_ids(text)
        if return_tensors == "pt":
            return torch.tensor([ids], dtype=torch.long)
        return ids

    def convert_tokens_to_ids(self, token: str):
        return {"<|im_start|>": IM_START_ID, "<|im_end|>": IM_END_ID}.get(token)


def _neg_row() -> dict:
    return {
        "prompt": [
            {"role": "system", "content": "You are a hero."},
            {"role": "user", "content": "What makes a good leader?"},
        ],
        "completion": [{"role": "assistant", "content": "Courage and patience. Answer."}],
    }


def _pos_row() -> dict:
    row = _neg_row()
    return {
        "prompt": row["prompt"],
        "completion": [{"role": "assistant", "content": "Courage and patience. Answer. ※"}],
    }


# ── (c) post-response-slot tokenization ──────────────────────────────────────


def test_post_slot_returns_completion_im_end_not_trailing_slot():
    tok = FakeQwenTokenizer()
    row = _neg_row()
    full_ids = tok.apply_chat_template(row["prompt"] + row["completion"])
    # Layout sanity: trailing tail is [..., <|im_end|>, \n]; the prompt holds
    # TWO earlier <|im_end|> (system + user closers).
    assert full_ids[-2] == IM_END_ID and full_ids[-1] == NL_ID
    assert full_ids.count(IM_END_ID) == 3

    picked = _tokenize_negative_row_post_slot(row, tok, [MARKER_ID], 2048, IM_END_ID)
    assert picked is not None
    ids, slot, target = picked
    assert target == IM_END_ID
    # The returned prefix ends AT the completion <|im_end|> (trailing \n dropped)
    # and the slot is the position predicting it.
    assert ids == full_ids[:-1]
    assert ids[slot + 1] == IM_END_ID
    assert slot == len(full_ids) - 3
    # NOT a prompt <|im_end|>: the slot sits after the LAST <|im_start|>.
    last_start = max(i for i, t in enumerate(full_ids) if t == IM_START_ID)
    assert slot + 1 > last_start

    # Contrast with the trailing channel: same row, DIFFERENT slot + target.
    trailing = _tokenize_negative_row(row, tok, [MARKER_ID], 2048)
    assert trailing is not None
    t_ids, t_slot, t_target = trailing
    assert t_target == NL_ID and t_slot == len(t_ids) - 2
    assert (slot, target) != (t_slot, t_target)


def test_post_slot_skips_marker_bearing_rows():
    tok = FakeQwenTokenizer()
    assert _tokenize_negative_row_post_slot(_pos_row(), tok, [MARKER_ID], 2048, IM_END_ID) is None


def test_build_rowtype_probes_three_channels_and_legacy_default(tmp_path: Path):
    tok = FakeQwenTokenizer()
    data = tmp_path / "train_pool.jsonl"
    rows = [_pos_row(), _neg_row(), _neg_row()]
    data.write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    # Flag ON: third channel present, trailing channel unchanged.
    probes = build_rowtype_probes(
        data,
        tok,
        [MARKER_ID],
        n_pos=1,
        n_neg=2,
        neg_post_response_slot=True,
        im_end_token_id=IM_END_ID,
    )
    assert probes["pos"] is not None and probes["pos"]["n_rows"] == 1
    assert probes["neg"] is not None and probes["neg"]["n_rows"] == 2
    assert probes["neg_slot"] is not None and probes["neg_slot"]["n_rows"] == 2
    assert probes["neg_slot"]["target_ids"].tolist() == [IM_END_ID, IM_END_ID]
    assert probes["neg"]["target_ids"].tolist() == [NL_ID, NL_ID]
    assert probes["neg_slot"]["positions"].tolist() != probes["neg"]["positions"].tolist()

    # Flag OFF (default): byte-identical 2-channel shape — no neg_slot key.
    legacy = build_rowtype_probes(data, tok, [MARKER_ID], n_pos=1, n_neg=2)
    assert "neg_slot" not in legacy

    # Fail-loud: flag on without the im_end id.
    with pytest.raises(ValueError, match="im_end_token_id"):
        build_rowtype_probes(data, tok, [MARKER_ID], neg_post_response_slot=True)


# ── (b) i601_run_cell threading (monkeypatched full-main run) ────────────────


def _load_script(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _run_main_with_mocks(monkeypatch, tmp_path: Path, argv: list[str]) -> dict:
    """Run i601_run_cell.main(argv) with every heavy dependency mocked.

    Returns {"train": kwargs, "probes": kwargs, "subprocess_cmds": [...]}.
    The mocks patch the LIBRARY modules main() imports at call time, so the
    threading path under test is the real script code end to end.
    """
    captured: dict = {"subprocess_cmds": []}
    fake_tok = FakeQwenTokenizer()

    import transformers

    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        build_training_data,
        centroids,
        persona_bank,
        r_generate,
        select_negatives,
        train_cell,
    )
    from explore_persona_space.experiments.neg_setpoint_601 import rowtype_ce_probe

    monkeypatch.setattr(transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: fake_tok)
    monkeypatch.setattr(persona_bank, "load_persona_bank", lambda p: {"villain": "v"})
    monkeypatch.setattr(r_generate, "load_r_artifact", lambda p: {})
    monkeypatch.setattr(r_generate, "get_train_eval_questions", lambda: (["q1"], ["e1"]))
    monkeypatch.setattr(centroids, "cos_to_source", lambda *a, **kw: {})
    monkeypatch.setattr(
        select_negatives, "negatives_for_cell", lambda *a, **kw: list(EXPECTED_ANCHOR_PANEL)
    )

    def fake_build_cell(cell_slug, output_path, **kw):
        # Spec-consistent positive count: the worker's fused-surface marker
        # assert (#613 sep-ablation) cross-checks n_positive_checked against
        # spec.pos_ex (=200 for every cell these tests run), so the fake mix
        # must carry 200 marker-bearing rows for main() to reach the phases
        # under test. Captured for the sep-threading tests in
        # tests/test_i613_sepablation.py (same helper shape there).
        captured["build"] = kw
        rows = [_pos_row()] * 200 + [_neg_row()]
        Path(output_path).write_text("\n".join(json.dumps(r) for r in rows) + "\n")

    monkeypatch.setattr(build_training_data, "build_cell", fake_build_cell)

    fake_batch = {
        "input_ids": torch.zeros((1, 4), dtype=torch.long),
        "attention_mask": torch.ones((1, 4), dtype=torch.long),
        "positions": torch.zeros(1, dtype=torch.long),
        "target_ids": torch.zeros(1, dtype=torch.long),
        "n_rows": 1,
    }

    def fake_build_probes(data_path, tokenizer, marker_ids, **kw):
        captured["probes"] = kw
        out = {"pos": fake_batch, "neg": fake_batch}
        if kw.get("neg_post_response_slot"):
            out["neg_slot"] = fake_batch
        return out

    monkeypatch.setattr(rowtype_ce_probe, "build_rowtype_probes", fake_build_probes)

    def fake_train_one_cell(**kw):
        captured["train"] = kw
        monkeypatch.setenv("CUDA_VISIBLE_DEVICES", str(kw["gpu_id"]))
        adapter = tmp_path / "adapter"
        adapter.mkdir(exist_ok=True)
        return {"checkpoint_index": {"1.0000": {"step": 63, "path": str(adapter)}}}

    monkeypatch.setattr(train_cell, "train_one_cell", fake_train_one_cell)

    run_cell = _load_script(RUN_CELL_PY, "i601_run_cell_under_test")

    def fake_subprocess_run(cmd, env=None, check=True, **kwargs):
        # Patching run on the SHARED stdlib subprocess module also routes
        # check_output here (it calls run internally) — the worker's
        # build-manifest _git_sha read takes that path. Emulate it.
        if list(cmd)[:2] == ["git", "rev-parse"]:
            return SimpleNamespace(returncode=0, stdout="0" * 40 + "\n", args=list(cmd))
        captured["subprocess_cmds"].append(list(cmd))
        out_path = Path(cmd[cmd.index("--out-path") + 1])
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text("{}")
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(run_cell.subprocess, "run", fake_subprocess_run)

    rc = run_cell.main(argv)
    assert rc == 0
    return captured


def test_flagon_threading_reaches_train_one_cell_and_probes(monkeypatch, tmp_path: Path):
    argv = [
        "--cell",
        SLUG,
        "--seed",
        "42",
        "--slab-root",
        str(tmp_path / "slab"),
        "--runs-root",
        str(tmp_path / "runs"),
        "--log-dir",
        str(tmp_path / "logs"),
        "--data-dir",
        str(tmp_path / "data"),
        "--skip-checkpoint-upload",
        "--hf-prefix",
        "adapters/issue_613",
        "--run-name-prefix",
        "issue613",
        "--sentinel-task-id",
        "613",
    ]
    captured = _run_main_with_mocks(monkeypatch, tmp_path, argv)

    train = captured["train"]
    # THE flag conjunction (plan §4 step 2 — the manipulated variable).
    assert train["marker_suppress_at_post_response_slot"] is True
    assert train["marker_im_end_token_id"] == IM_END_ID
    # D1 band wiring unchanged.
    assert train["marker_band_stop"] is True and train["marker_band_log_only"] is True
    # #613 identity flags.
    assert train["hf_path_in_repo_override"] == f"adapters/issue_613/{SLUG}_seed42"
    assert train["run_name_override"] == f"issue613_{SLUG}_seed42"

    probes = captured["probes"]
    assert probes["neg_post_response_slot"] is True
    assert probes["im_end_token_id"] == IM_END_ID

    # Sentinel re-pointed to issue 613 (filename + task_id field).
    sentinel = tmp_path / "logs" / f"issue-613-{SLUG}-seed42-results.json"
    assert sentinel.exists()
    payload = json.loads(sentinel.read_text())
    assert payload["task_id"] == 613
    note = json.loads(payload["note"])
    assert note["suppress_negatives"] is True
    assert note["adapter_hf_path"] == f"adapters/issue_613/{SLUG}_seed42"


def test_legacy_cell_without_new_flags_stays_byte_identical(monkeypatch, tmp_path: Path):
    argv = [
        "--cell",
        FLAGOFF,
        "--seed",
        "137",
        "--slab-root",
        str(tmp_path / "slab"),
        "--runs-root",
        str(tmp_path / "runs"),
        "--log-dir",
        str(tmp_path / "logs"),
        "--data-dir",
        str(tmp_path / "data"),
        "--skip-checkpoint-upload",
    ]
    captured = _run_main_with_mocks(monkeypatch, tmp_path, argv)

    train = captured["train"]
    assert train["marker_suppress_at_post_response_slot"] is False
    assert train["marker_im_end_token_id"] is None
    assert train["hf_path_in_repo_override"] == f"adapters/issue_601/{FLAGOFF}_seed137"
    assert train["run_name_override"] == f"issue601_{FLAGOFF}_seed137"
    assert captured["probes"]["neg_post_response_slot"] is False

    sentinel = tmp_path / "logs" / f"issue-601-{FLAGOFF}-seed137-results.json"
    assert sentinel.exists()
    assert json.loads(sentinel.read_text())["task_id"] == 601


# ── (c) terminal-adapter fail-loud upload verify (round-2 blocker) ───────────
# Pins scripts/i601_run_cell.py::_verify_terminal_adapter_uploaded — the guard
# that closes the warn-and-continue hole in train_lora's terminal upload
# (concern flagon-terminal-upload-not-fail-loud). The helper defers its
# huggingface_hub / orchestrate.hub imports to call time, so monkeypatching
# the LIBRARY module attributes is picked up (and the deferred imports are
# EXECUTED here — no smoke-skipped lazy-import blind spot).

TERMINAL_PREFIX = f"adapters/issue_613/{SLUG}_seed42"
TERMINAL_FILES = [
    f"{TERMINAL_PREFIX}/adapter_config.json",
    f"{TERMINAL_PREFIX}/adapter_model.safetensors",
]


def _wire_terminal_verify(monkeypatch, list_results: list[list[str]], upload_return="hub/path"):
    """Load the script + fake list_repo_files / upload_model; return (verify_fn, calls).

    ``list_results[i]`` is the repo listing returned by the (i+1)-th
    ``list_repo_files`` call (the last entry repeats for later calls).
    """
    import huggingface_hub

    from explore_persona_space.orchestrate import hub as hub_mod

    run_cell = _load_script(RUN_CELL_PY, "i601_run_cell_terminal_verify_under_test")
    calls: dict = {"list": 0, "upload": []}

    def fake_list(repo_id, repo_type=None):
        calls["list"] += 1
        return list_results[min(calls["list"], len(list_results)) - 1]

    def fake_upload(model_path, repo_id=None, path_in_repo=None, delete_after=False, **kw):
        calls["upload"].append((model_path, repo_id, path_in_repo, delete_after))
        return upload_return

    monkeypatch.setattr(huggingface_hub, "list_repo_files", fake_list)
    monkeypatch.setattr(hub_mod, "upload_model", fake_upload)
    return run_cell._verify_terminal_adapter_uploaded, calls


def test_terminal_verify_noop_when_already_on_hub(monkeypatch, tmp_path: Path):
    verify, calls = _wire_terminal_verify(monkeypatch, [TERMINAL_FILES])
    verify(
        repo_id="superkaiba1/explore-persona-space",
        terminal_prefix=TERMINAL_PREFIX,
        adapter_dir=tmp_path / "adapter",
    )
    assert calls["upload"] == []  # present -> no re-upload
    assert calls["list"] == 1


def test_terminal_verify_reuploads_missing_terminal_then_passes(monkeypatch, tmp_path: Path):
    # First listing misses the terminal (train_lora's upload silently failed);
    # the re-upload lands and the second listing resolves it.
    verify, calls = _wire_terminal_verify(monkeypatch, [["unrelated/file.json"], TERMINAL_FILES])
    verify(
        repo_id="superkaiba1/explore-persona-space",
        terminal_prefix=TERMINAL_PREFIX,
        adapter_dir=tmp_path / "adapter",
    )
    assert calls["upload"] == [
        (str(tmp_path / "adapter"), "superkaiba1/explore-persona-space", TERMINAL_PREFIX, False)
    ]
    assert calls["list"] == 2


def test_terminal_verify_raises_when_reupload_does_not_land(monkeypatch, tmp_path: Path):
    verify, calls = _wire_terminal_verify(monkeypatch, [[], []])
    with pytest.raises(RuntimeError, match="STILL missing"):
        verify(
            repo_id="superkaiba1/explore-persona-space",
            terminal_prefix=TERMINAL_PREFIX,
            adapter_dir=tmp_path / "adapter",
        )
    assert len(calls["upload"]) == 1  # re-upload attempted exactly once, then fail-loud


def test_terminal_verify_raises_on_empty_hub_path(monkeypatch, tmp_path: Path):
    verify, calls = _wire_terminal_verify(monkeypatch, [[]], upload_return="")
    with pytest.raises(RuntimeError, match="empty hub path"):
        verify(
            repo_id="superkaiba1/explore-persona-space",
            terminal_prefix=TERMINAL_PREFIX,
            adapter_dir=tmp_path / "adapter",
        )
    assert calls["list"] == 1  # raises BEFORE any re-listing


# ── (d) dense-read sep-mode plumbing ─────────────────────────────────────────


class _PrefixRecordingTokenizer:
    """Records every encode() prefix; renders a deterministic prompt string."""

    def __init__(self):
        self.prefixes: list[str] = []

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=True):
        assert tokenize is False and add_generation_prompt is True
        sys_c, user_c = messages[0]["content"], messages[1]["content"]
        return f"<sys>{sys_c}</sys><user>{user_c}</user><assistant>"

    def encode(self, text, add_special_tokens=False, return_tensors=None):
        assert return_tensors == "pt"
        self.prefixes.append(text)
        return torch.tensor([[1, 2, 3]], dtype=torch.long)


def _fake_lm(vocab: int = 16):
    def _call(input_ids=None):
        t = input_ids.shape[1]
        return SimpleNamespace(logits=torch.zeros((1, t, vocab)))

    return _call


def test_sep_mode_plain_renders_prompt_plus_r_with_no_separator():
    dense_read = _load_script(DENSE_READ_PY, "i601_dense_read_under_test")
    tok = _PrefixRecordingTokenizer()
    model = _fake_lm()
    prompt = "<sys>You are kind.</sys><user>Q?</user><assistant>"

    dense_read._slot_raw_logits(model, tok, "You are kind.", "Q?", "R text.", "", "cpu")
    assert tok.prefixes[-1] == prompt + "R text."  # plain: NO separator

    dense_read._slot_raw_logits(model, tok, "You are kind.", "Q?", "R text.", "\n\n", "cpu")
    assert tok.prefixes[-1] == prompt + "R text." + "\n\n"  # marker: the parent DV slot


def test_resume_refuses_sep_mode_mismatch(tmp_path: Path):
    dense_read = _load_script(DENSE_READ_PY, "i601_dense_read_under_test_resume")
    out = tmp_path / "slot_trajectory.json"
    out.write_text(json.dumps({"sep_mode": "marker", "checkpoints": []}))
    with pytest.raises(RuntimeError, match="sep_mode"):
        dense_read._parity_resume_checkpoints(out, expected_sep_mode="plain")
    # Same mode resumes fine (empty prior -> {}).
    assert dense_read._parity_resume_checkpoints(out, expected_sep_mode="marker") == {}
    # Absent file -> clean start.
    assert dense_read._parity_resume_checkpoints(tmp_path / "nope.json") == {}
