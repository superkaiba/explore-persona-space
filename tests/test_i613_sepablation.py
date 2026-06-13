# ruff: noqa: RUF001  # the battery DELIBERATELY tests ambiguous-unicode endings
# em-dash + Qwen marker token " ※" intentional
"""Task #613 follow-up `sep-ablation` — registry / builder / sep-threading / guard tests.

Amendment plan §3 item 7, pinned claims:

(a) Registry: ``sepablation_flag{on,off}_200p800n`` are the dense_200p800n
    recipe verbatim with ``marker_sep=""`` and ONLY ``suppress_negatives``
    differing between the pair (within-construction single variable); both
    conditional / explicit-slug-only; every legacy cell keeps
    ``marker_sep=MARKER_SEP`` (byte-preserving default).
(b) ``build_cell(marker_sep="")`` emits positives as ``R + " ※"`` with NO
    ``"\\n\\n"``; the default keeps the legacy ``R + "\\n\\n ※"`` construction;
    negatives are bare R either way (panel disjointness preserved).
(c) SUPPLEMENTAL bare-encode battery (31 endings, real Qwen tokenizer,
    skipped when no tokenizer is loadable): ``encode(r + " ※")`` ends with
    exactly one id 83399 in BOTH constructions, prefix-stable.
(d) TIGHTENED threading: a sep-ablation cell delivers ``marker_sep=""`` to
    ``build_cell``, ``sep=""`` to ``score_logp_for_R`` -> ``build_full_ids``,
    ``run_trajectory_eval`` -> both ``score_logp_for_R`` calls +
    ``compute_kl_for_checkpoint``, AND ``--sep-mode plain`` appears on BOTH
    nested subprocess argvs (eval + dense); legacy cells' argvs stay
    flag-free.
(e) ``build_full_ids(sep="")`` passes its own K-token context-equality
    contract post the eval_one_cell.py:159 hardcode fix (and the contract
    DISCRIMINATES: the two seps produce different tails).
(f) The FUSED-surface marker guard passes on real-shaped rows and fails loud
    (naming the row) on a merged/absent/mispositioned marker.

Plus: the seed-42 smoke gate's PASS and FAIL branches
(``scripts/i613_sepablation_smoke_gate.py``) and the registered analysis
rules (``scripts/i613_sepablation_analyze.py``: R3' denominator guard ->
confirmed-strong; R2' compression-window margin precedence; R2'
fixed-denominator emission sensitivity) on schema-real fixtures.

CPU-only; no model downloads (battery test skips without a local tokenizer).
"""

from __future__ import annotations

import dataclasses
import importlib.util
import json
import sys
import types
from pathlib import Path

import pytest

from explore_persona_space.experiments.contrastive_neg_geometry_472 import MARKER_SEP
from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_one_cell import (
    build_full_ids,
    build_train_equivalent_full_ids,
    score_logp_for_R,
)
from explore_persona_space.experiments.contrastive_neg_geometry_472.eval_trajectory import (
    run_trajectory_eval,
)
from explore_persona_space.experiments.neg_setpoint_601 import (
    CELLS_601,
    cell_by_slug,
    cells_for_request,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = REPO_ROOT / "scripts"

FLAGON = "sepablation_flagon_200p800n"
FLAGOFF = "sepablation_flagoff_200p800n"
PARENT = "dense_200p800n"
MARKER_ID = 83399
MARKER_TEXT = " ※"
IM_END_ID = 151645
IM_START_ID = 151644


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# Reuse the sibling round's fake Qwen-shaped tokenizer + row fixtures.
_SIBLING = _load_module(REPO_ROOT / "tests" / "test_i613_flagon_threading.py", "t_i613_flagon")
FakeQwenTokenizer = _SIBLING.FakeQwenTokenizer
_neg_row = _SIBLING._neg_row
_pos_row = _SIBLING._pos_row


# ── (a) registry asserts ─────────────────────────────────────────────────────


def test_sepablation_pair_registered_single_variable():
    on = cell_by_slug(FLAGON)
    off = cell_by_slug(FLAGOFF)
    parent = cell_by_slug(PARENT)
    for spec, suppress in ((on, True), (off, False)):
        assert spec.marker_sep == ""  # THE round variable
        assert spec.suppress_negatives is suppress
        assert spec.phase == "sep-ablation"  # follow-up contract output dir
        assert spec.conditional is True
        assert (spec.pos_ex, spec.n_neg_personas, spec.neg_ex_per_persona) == (200, 4, 200)
        assert spec.total_rows == 1000
        assert spec.expected_steps == 63
        assert spec.dense_steps == parent.dense_steps  # EXACT dense-ladder parity
        assert spec.onpolicy == "anchors"
        assert (spec.lr, spec.epochs, spec.lora_targets) == (
            parent.lr,
            parent.epochs,
            parent.lora_targets,
        )
        assert (spec.band_stop, spec.band_log_only) == (True, True)  # D1 log-only
        assert spec.seeds == (42, 137)
    # Within-construction single variable: the pair differs ONLY in identity
    # fields + the suppress flag.
    diff = {f.name for f in dataclasses.fields(on) if getattr(on, f.name) != getattr(off, f.name)}
    assert diff == {"slug", "plain_name", "suppress_negatives"}


def test_sepablation_cells_are_explicit_slug_only():
    for slug in (FLAGON, FLAGOFF):
        assert slug not in {c.slug for c in cells_for_request("all")}
        assert slug not in {c.slug for c in cells_for_request(None)}
        assert slug not in {c.slug for c in cells_for_request("phase4b")}
        assert [c.slug for c in cells_for_request(slug)] == [slug]
    # Parent sweep group unchanged (10 non-conditional cells).
    assert len({c.slug for c in cells_for_request("all")}) == 10


def test_every_legacy_cell_keeps_marker_sep_default():
    for c in CELLS_601:
        if c.slug in (FLAGON, FLAGOFF):
            continue
        assert c.marker_sep == MARKER_SEP, f"{c.slug} drifted off the legacy separator"


# ── (b) builder content ──────────────────────────────────────────────────────


def _build_fixture_inputs():
    from explore_persona_space.experiments.neg_setpoint_601 import (
        EXPECTED_ANCHOR_PANEL,
        SOURCE_PERSONA,
    )

    personas = [SOURCE_PERSONA, *EXPECTED_ANCHOR_PANEL]
    q_train = ["What makes a good leader?", "How do you spend a free day?"]
    bank = {p: f"You are {p}." for p in personas}
    r_train = {
        p: {q: {"response_text": f"{p} answer to {q}".rstrip("?")} for q in q_train}
        for p in personas
    }
    return bank, r_train, q_train


@pytest.mark.parametrize(
    ("slug", "sep_kwargs", "want_sep"),
    [
        (FLAGON, {"marker_sep": ""}, ""),
        (PARENT, {}, MARKER_SEP),  # default = legacy byte-identical
    ],
)
def test_build_cell_positive_construction(monkeypatch, tmp_path: Path, slug, sep_kwargs, want_sep):
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        build_training_data,
    )
    from explore_persona_space.experiments.neg_setpoint_601 import (
        CELL_SPECS_601_472SHAPE,
        EXPECTED_ANCHOR_PANEL,
        SOURCE_PERSONA,
    )

    bank, r_train, q_train = _build_fixture_inputs()
    monkeypatch.setattr(
        build_training_data, "negatives_for_cell", lambda *a, **kw: list(EXPECTED_ANCHOR_PANEL)
    )
    out = tmp_path / "train_pool.jsonl"
    build_training_data.build_cell(
        slug,
        out,
        r_train=r_train,
        cos_to_source={},
        q_train=q_train,
        persona_bank=bank,
        source=SOURCE_PERSONA,
        seed=42,
        cell_specs=CELL_SPECS_601_472SHAPE,
        pos_ex_override=4,
        **sep_kwargs,
    )
    rows = [json.loads(line) for line in out.read_text().splitlines()]
    pos = [r for r in rows if MARKER_TEXT in r["completion"][0]["content"]]
    neg = [r for r in rows if MARKER_TEXT not in r["completion"][0]["content"]]
    assert len(pos) == 4 and len(neg) == 4 * 200  # panel disjoint mix shape
    for r in pos:
        content = r["completion"][0]["content"]
        base = content[: -len(f"{want_sep}{MARKER_TEXT}")]
        assert content == f"{base}{want_sep}{MARKER_TEXT}"
        if want_sep == "":
            assert "\n\n" not in content  # NO separator anywhere in R + marker
        else:
            assert content.endswith(f"\n\n{MARKER_TEXT}")
    for r in neg:
        assert "※" not in r["completion"][0]["content"]  # bare R, both constructions
        assert r["prompt"][0]["content"] != bank[SOURCE_PERSONA]  # disjoint from source


# ── (c) SUPPLEMENTAL bare-encode battery (real tokenizer, skip-if-absent) ────

_ENDINGS = (
    ".",
    "!",
    "?",
    '"',
    ")",
    "…",
    ":",
    ";",
    "word",
    "7",
    "%",
    "*",
    "_",
    ".)",
    '?"',
    "—",
    "\n",
    " ",
    "。",
    "．",
    "]",
    "'",
    "...",
    ".'",
    '."',
    "”",
    "’",
    "\t",
    "`code`",
    "0.95",
    "(s)",
)


@pytest.fixture(scope="module")
def qwen_tokenizer():
    from transformers import AutoTokenizer

    from explore_persona_space.experiments.neg_setpoint_601 import BASE_MODEL

    try:
        return AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    except Exception as e:  # pragma: no cover - offline CI
        pytest.skip(f"Qwen tokenizer unavailable ({e}); battery runs on VM/pod with HF cache")


def test_bare_encode_battery_31_endings(qwen_tokenizer):
    assert len(_ENDINGS) == 31
    assert qwen_tokenizer.encode(MARKER_TEXT, add_special_tokens=False) == [MARKER_ID]
    for ending in _ENDINGS:
        r = f"Some answer text{ending}"
        base_ids = qwen_tokenizer.encode(r, add_special_tokens=False)
        # No-separator construction: append never re-tokenizes R's tail and
        # contributes EXACTLY the single marker id (prefix stability).
        plain_ids = qwen_tokenizer.encode(r + MARKER_TEXT, add_special_tokens=False)
        assert plain_ids == [*base_ids, MARKER_ID], f"prefix instability for ending {ending!r}"
        assert plain_ids.count(MARKER_ID) == 1
        # Legacy sep construction: marker still its own single token at the end.
        sep_ids = qwen_tokenizer.encode(r + MARKER_SEP + MARKER_TEXT, add_special_tokens=False)
        assert sep_ids[-1] == MARKER_ID and sep_ids.count(MARKER_ID) == 1, (
            f"sep construction broke for ending {ending!r}"
        )


# ── (d) worker threading: marker_sep -> build_cell + --sep-mode on BOTH argvs ─


def _run_worker_with_mocks(monkeypatch, tmp_path: Path, cell: str, seed: int) -> dict:
    """Mocked full-main worker run (forked from the sibling helper, which it reuses)."""
    argv = [
        "--cell",
        cell,
        "--seed",
        str(seed),
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
    return _SIBLING._run_main_with_mocks(monkeypatch, tmp_path, argv)


def test_sepablation_cell_threads_sep_to_build_and_both_argvs(monkeypatch, tmp_path: Path):
    captured = _run_worker_with_mocks(monkeypatch, tmp_path, FLAGON, 42)
    # build_cell received THE round variable.
    assert captured["build"]["marker_sep"] == ""
    # BOTH nested subprocess argvs carry --sep-mode plain (tightened test 7d).
    cmds = captured["subprocess_cmds"]
    assert len(cmds) == 2  # eval + dense
    eval_cmd = next(c for c in cmds if "scripts/i601_eval_trajectory.py" in c)
    dense_cmd = next(c for c in cmds if "scripts/i601_dense_read.py" in c)
    for cmd in (eval_cmd, dense_cmd):
        assert "--sep-mode" in cmd, f"--sep-mode missing on argv: {cmd}"
        assert cmd[cmd.index("--sep-mode") + 1] == "plain"
    # Train phase received the suppress flag per the registry (alive arm).
    assert captured["train"]["marker_suppress_at_post_response_slot"] is True
    # Durable unit manifest echoes marker_sep + the fused-surface assert.
    manifest = json.loads(
        (
            tmp_path / "slab" / "sep-ablation" / f"{FLAGON}_seed42" / "build_manifest.json"
        ).read_text()
    )
    assert manifest["marker_sep"] == "" and manifest["sep_mode"] == "plain"
    assert manifest["fused_marker_assert"]["passed"] is True
    assert manifest["fused_marker_assert"]["n_positive_checked"] == 200
    # Sentinel note carries the spec-conditional echo.
    note = json.loads(
        json.loads((tmp_path / "logs" / f"issue-613-{FLAGON}-seed42-results.json").read_text())[
            "note"
        ]
    )
    assert note["marker_sep"] == "" and note["sep_mode"] == "plain"


def test_sepablation_flagoff_cell_threads_sep_with_suppress_off(monkeypatch, tmp_path: Path):
    captured = _run_worker_with_mocks(monkeypatch, tmp_path, FLAGOFF, 137)
    assert captured["build"]["marker_sep"] == ""
    assert captured["train"]["marker_suppress_at_post_response_slot"] is False
    for cmd in captured["subprocess_cmds"]:
        assert cmd[cmd.index("--sep-mode") + 1] == "plain"


def test_legacy_cell_argvs_stay_flag_free(monkeypatch, tmp_path: Path):
    captured = _run_worker_with_mocks(monkeypatch, tmp_path, PARENT, 137)
    assert captured["build"]["marker_sep"] == MARKER_SEP  # legacy byte-identical rows
    for cmd in captured["subprocess_cmds"]:
        assert "--sep-mode" not in cmd  # byte-identical legacy argvs


# ── (d) eval CLI: --sep-mode plain -> run_trajectory_eval(sep="") ────────────


def _run_eval_cli(monkeypatch, tmp_path: Path, extra_argv: list[str]) -> dict:
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        eval_trajectory as traj_mod,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        persona_bank,
        r_generate,
    )
    from explore_persona_space.experiments.neg_setpoint_601 import artifacts

    captured: dict = {}
    monkeypatch.setattr(persona_bank, "load_persona_bank", lambda p: {"villain": "v", "hero": "h"})
    monkeypatch.setattr(r_generate, "get_train_eval_questions", lambda: (["q1"], ["e1"]))
    monkeypatch.setattr(
        artifacts,
        "stage_parity_read_adapter",
        lambda path, root, expect_slug: (path, {"use_rslora_applied": False}),
    )

    def fake_run_trajectory_eval(**kw):
        captured.update(kw)
        Path(kw["out_path"]).parent.mkdir(parents=True, exist_ok=True)
        Path(kw["out_path"]).write_text("{}")
        return kw["out_path"]

    monkeypatch.setattr(traj_mod, "run_trajectory_eval", fake_run_trajectory_eval)

    panel = tmp_path / "bystander_panel.json"
    panel.write_text(json.dumps({"personas": ["hero"]}))
    adapter = tmp_path / f"{FLAGON}_seed42" / "adapter"
    adapter.mkdir(parents=True)
    idx = tmp_path / "checkpoint_index.json"
    idx.write_text(json.dumps({"1.0000": {"step": 63, "path": str(adapter)}}))

    cli = _load_module(SCRIPTS / "i601_eval_trajectory.py", "i601_eval_trajectory_under_test")
    rc = cli.main(
        [
            "--cell",
            FLAGON,
            "--seed",
            "42",
            "--checkpoint-index",
            str(idx),
            "--out-path",
            str(tmp_path / "trajectory.json"),
            "--fracs",
            "1.0000",
            "--panel",
            "bystander8",
            "--bystander-panel-path",
            str(panel),
            *extra_argv,
        ]
    )
    assert rc == 0
    return captured


def test_eval_cli_sep_mode_plain_maps_to_empty_sep(monkeypatch, tmp_path: Path):
    captured = _run_eval_cli(monkeypatch, tmp_path, ["--sep-mode", "plain"])
    assert captured["sep"] == ""


def test_eval_cli_default_keeps_marker_sep(monkeypatch, tmp_path: Path):
    captured = _run_eval_cli(monkeypatch, tmp_path, [])
    assert captured["sep"] == MARKER_SEP


# ── (d) run_trajectory_eval threads sep to BOTH score calls + the KL read ────


class _FakeLLM:
    def __init__(self, **kw):
        self.kw = kw

    def generate(self, prompts, sp, **kw):  # pragma: no cover - not reached (gen mocked)
        return ["out"] * len(prompts)


def _install_fake_vllm(monkeypatch):
    fake = types.ModuleType("vllm")
    fake.LLM = _FakeLLM
    fake.SamplingParams = lambda **kw: kw
    lora_mod = types.ModuleType("vllm.lora.request")
    lora_mod.LoRARequest = lambda **kw: types.SimpleNamespace(**kw)
    fake.lora = types.ModuleType("vllm.lora")
    fake.lora.request = lora_mod
    monkeypatch.setitem(sys.modules, "vllm", fake)
    monkeypatch.setitem(sys.modules, "vllm.lora", fake.lora)
    monkeypatch.setitem(sys.modules, "vllm.lora.request", lora_mod)


def test_run_trajectory_eval_threads_sep_to_score_and_kl(monkeypatch, tmp_path: Path):
    import transformers

    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        eval_guard,
    )
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import (
        eval_trajectory as traj_mod,
    )

    _install_fake_vllm(monkeypatch)
    monkeypatch.setattr(
        transformers.AutoTokenizer, "from_pretrained", lambda *a, **kw: FakeQwenTokenizer()
    )

    score_seps: list[str] = []
    kl_seps: list[str] = []
    personas_all = ("hero", "villain")

    def fake_score(llm, tokenizer, *, sep=MARKER_SEP, eval_personas=None, **kw):
        score_seps.append(sep)
        return {
            p: {
                "e1": {
                    "logp": -3.0,
                    "argmax_marker": False,
                    "n_marker_in_R": 0,
                    "r_collapsed": False,
                }
            }
            for p in eval_personas
        }

    def fake_gen(llm, tokenizer, eval_personas, eval_questions, lora_request, max_new_tokens):
        return {p: {"e1": "an answer"} for p in eval_personas}

    def fake_kl(*, sep=MARKER_SEP, eval_personas=None, eval_questions=None, **kw):
        kl_seps.append(sep)
        keys = (
            "kl",
            "z_marker_g",
            "z_marker_b",
            "z_eos_g",
            "z_eos_b",
            "logZ_g",
            "logZ_b",
            "logp_hf_g",
            "logp_hf_b",
        )
        return {p: {q: dict.fromkeys(keys, 0.0) for q in eval_questions} for p in eval_personas}

    monkeypatch.setattr(traj_mod, "score_logp_for_R", fake_score)
    monkeypatch.setattr(traj_mod, "_generate_on_policy_R", fake_gen)
    monkeypatch.setattr(traj_mod, "compute_kl_for_checkpoint", fake_kl)
    monkeypatch.setattr(traj_mod, "_teardown_vllm_hard", lambda llm: None)
    monkeypatch.setattr(traj_mod, "assert_logit_readout_gauge_free", lambda p: None)
    monkeypatch.setattr(eval_guard, "assert_adapter_actually_applied", lambda **kw: None)
    monkeypatch.setattr(
        eval_guard,
        "assert_byte_identical_rate_below_threshold",
        lambda *a, **kw: {"byte_identical_rate": 0.0},
    )

    adapter = tmp_path / f"{FLAGON}_seed42" / "adapter"
    adapter.mkdir(parents=True)
    out = tmp_path / "trajectory.json"
    run_trajectory_eval(
        cell_slug=FLAGON,
        seed=42,
        checkpoint_specs=[{"frac": 1.0, "step": 63, "adapter_path": str(adapter)}],
        eval_personas={"hero": "h"},
        eval_questions=["e1"],
        source="villain",
        source_prompt="v",
        out_path=out,
        sep="",
    )
    assert score_seps == ["", ""], "sep must reach BOTH score_logp_for_R calls (trained + base)"
    assert kl_seps == [""], "sep must reach compute_kl_for_checkpoint"
    payload = json.loads(out.read_text())
    assert payload["sep"] == ""  # provenance field
    # Per-question source leaves persisted (R2' emission-denominator input).
    term = payload["checkpoints"][0]
    assert term["source_per_q"]["e1"]["n_marker_in_R"] == 0
    assert {p for p in personas_all if p in ("hero",)} <= set(term["held_out"].keys()) | {"hero"}


def test_score_logp_for_R_passes_sep_to_build_full_ids(monkeypatch):
    from explore_persona_space.experiments.contrastive_neg_geometry_472 import eval_one_cell

    _install_fake_vllm(monkeypatch)
    seps: list[str] = []

    def fake_build_full_ids(
        tokenizer, persona_prompt, q, r_text, marker_text, marker_id, persona, q_log, sep=MARKER_SEP
    ):
        seps.append(sep)
        return [1, 2, MARKER_ID], 1, 1, 2, 0

    monkeypatch.setattr(eval_one_cell, "build_full_ids", fake_build_full_ids)
    monkeypatch.setattr(
        eval_one_cell,
        "extract_marker_logprob_and_argmax",
        lambda outputs, slots, marker_id, label: ([-5.0] * len(slots), [False] * len(slots)),
    )

    class _GenLLM:
        def generate(self, prompts, sp, **kw):
            return ["o"] * len(prompts)

    for sep_value in ("", MARKER_SEP):
        seps.clear()
        score_logp_for_R(
            _GenLLM(),
            FakeQwenTokenizer(),
            r_by_persona_q={"hero": {"q1": "R text."}},
            eval_personas={"hero": "h"},
            eval_questions=["q1"],
            cell_label="t",
            use_lora=False,
            sep=sep_value,
        )
        assert seps == [sep_value]


# ── (e) build_full_ids(sep="") context contract post-fix ────────────────────


class StringTemplateTokenizer:
    """Qwen-shaped STRING chat template + char-level encode (" ※" -> 83399).

    Char-level encoding is prefix-stable by construction, so the contract
    checks isolate exactly the SEP-threading behavior under test.
    """

    def apply_chat_template(self, messages, *, tokenize=False, add_generation_prompt=False):
        assert tokenize is False
        s = ""
        for m in messages:
            s += f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>\n"
        if add_generation_prompt:
            s += "<|im_start|>assistant\n"
        return s

    def encode(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            if text[i : i + 2] == MARKER_TEXT:
                ids.append(MARKER_ID)
                i += 2
            else:
                ids.append(ord(text[i]))
                i += 1
        return ids


@pytest.mark.parametrize("sep", ["", MARKER_SEP])
def test_build_full_ids_context_contract_holds_at_both_seps(sep):
    tok = StringTemplateTokenizer()
    full_ids, _prompt_len, r_len, slot, n_in_r = build_full_ids(
        tok, "You are kind.", "Q?", "Answer.", MARKER_TEXT, MARKER_ID, "p", "q", sep=sep
    )
    assert full_ids[-1] == MARKER_ID and slot == len(full_ids) - 1
    assert n_in_r == 0 and r_len > 0


def test_context_contract_discriminates_between_seps():
    # The K-token tails DIFFER between the two constructions — so the
    # pre-fix hardcode (train-equivalent at "\n\n" vs eval at "") would have
    # failed the contract, i.e. the post-fix pass above is a real pin.
    tok = StringTemplateTokenizer()
    tail_plain = build_train_equivalent_full_ids(
        tok, "You are kind.", "Q?", "Answer.", MARKER_TEXT, MARKER_ID, sep=""
    )[-3:]
    tail_sep = build_train_equivalent_full_ids(
        tok, "You are kind.", "Q?", "Answer.", MARKER_TEXT, MARKER_ID, sep=MARKER_SEP
    )[-3:]
    assert tail_plain != tail_sep


# ── (f) fused-surface marker guard ───────────────────────────────────────────

_RUN_CELL = _load_module(SCRIPTS / "i601_run_cell.py", "i601_run_cell_sepabl_under_test")


def _write_rows(path: Path, rows: list[dict]) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    return path


def test_fused_guard_passes_on_real_shaped_rows(tmp_path: Path):
    data = _write_rows(tmp_path / "pool.jsonl", [_pos_row(), _neg_row(), _pos_row()])
    out = _RUN_CELL._assert_positive_rows_fused_marker(
        data, FakeQwenTokenizer(), marker_text=MARKER_TEXT, marker_id=MARKER_ID, cell_slug="t"
    )
    assert out == {"n_rows_total": 3, "n_positive_checked": 2, "passed": True}


def test_fused_guard_fails_loud_on_merged_marker(tmp_path: Path):
    class MergedMarkerTokenizer(FakeQwenTokenizer):
        """Fused render merges the marker into a non-83399 id (the silent-flip class)."""

        def _content_ids(self, content: str) -> list[int]:
            return [
                7 if w == "※" else 1000 + (sum(ord(ch) for ch in w) % 7000) for w in content.split()
            ]

    data = _write_rows(tmp_path / "pool.jsonl", [_pos_row()])
    with pytest.raises(RuntimeError, match=r"fused-surface marker assert FAILED.*row 1"):
        _RUN_CELL._assert_positive_rows_fused_marker(
            data,
            MergedMarkerTokenizer(),
            marker_text=MARKER_TEXT,
            marker_id=MARKER_ID,
            cell_slug="t",
        )


def test_fused_guard_fails_loud_on_marker_outside_completion(tmp_path: Path):
    class PromptLeakTokenizer(FakeQwenTokenizer):
        """Puts the (single) marker id BEFORE the assistant turn."""

        def apply_chat_template(self, messages, *, tokenize=True, add_generation_prompt=False):
            ids = super().apply_chat_template(
                messages, tokenize=tokenize, add_generation_prompt=add_generation_prompt
            )
            ids = [t for t in ids if t != MARKER_ID]
            return [MARKER_ID, *ids]

    data = _write_rows(tmp_path / "pool.jsonl", [_pos_row()])
    with pytest.raises(RuntimeError, match="BEFORE the assistant completion region"):
        _RUN_CELL._assert_positive_rows_fused_marker(
            data,
            PromptLeakTokenizer(),
            marker_text=MARKER_TEXT,
            marker_id=MARKER_ID,
            cell_slug="t",
        )


# ── smoke gate PASS / FAIL branches ──────────────────────────────────────────

_GATE = _load_module(
    SCRIPTS / "i613_sepablation_smoke_gate.py", "i613_sepablation_smoke_gate_under_test"
)


def _gate_fixture(
    tmp_path: Path,
    *,
    neg_ce1=0.06,
    pos_ce1=24.0,
    pos_ce10=12.0,
    terminal_step=63,
    fused_n=200,
    marker_sep="",
) -> list[str]:
    cell_dir = tmp_path / "cell"
    cell_dir.mkdir(exist_ok=True)
    (cell_dir / "rowtype_ce.json").write_text(
        json.dumps(
            {
                "schema": "i601_rowtype_ce_v2",
                "n_pos_rows": 16,
                "n_neg_rows": 16,
                "n_neg_slot_rows": 16,
                "neg_slot_ce": [neg_ce1, 0.05],
                "neg_slot_ce_base": 0.05,
                "records": [
                    {"step": 1, "neg_slot_ce": neg_ce1, "pos_marker_ce": pos_ce1},
                    {"step": 10, "neg_slot_ce": 0.05, "pos_marker_ce": pos_ce10},
                ],
            }
        )
    )
    (cell_dir / "build_manifest.json").write_text(
        json.dumps(
            {
                "marker_sep": marker_sep,
                "fused_marker_assert": {
                    "passed": True,
                    "n_positive_checked": fused_n,
                    "n_rows_total": 1000,
                },
            }
        )
    )
    idx = tmp_path / "checkpoint_index.json"
    idx.write_text(json.dumps({"1.0000": {"step": terminal_step, "path": "x"}}))
    wandb_files = tmp_path / "wandb" / "run-x" / "files"
    wandb_files.mkdir(parents=True, exist_ok=True)
    (wandb_files / "config.yaml").write_text(f"run_name: issue613_{FLAGON}_seed42\n")
    (wandb_files / "wandb-summary.json").write_text(json.dumps({"rowtype_ce/neg_slot_ce": 0.05}))
    return [
        "--cell-dir",
        str(cell_dir),
        "--checkpoint-index",
        str(idx),
        "--run-name",
        f"issue613_{FLAGON}_seed42",
        "--wandb-root",
        str(tmp_path / "wandb"),
        "--expect-terminal-step",
        "63",
        "--expect-positives",
        "200",
    ]


def test_smoke_gate_passes_on_healthy_unit(tmp_path: Path, capsys):
    assert _GATE.main(_gate_fixture(tmp_path)) == 0
    assert "smoke gate PASS" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("overrides", "match"),
    [
        ({"neg_ce1": 1e-5}, "HALT-AND-INVESTIGATE"),  # (c) dead relocated slot
        ({"pos_ce1": 5.0}, "outside"),  # (d) step-1 CE below the base-prior band
        ({"pos_ce1": 35.0}, "outside"),  # (d) step-1 CE above the band
        ({"pos_ce10": 30.0}, "not falling"),  # (d) ce[10] < ce[1] violated
        ({"terminal_step": 45}, r"terminal step 45"),  # (a) band-stop misfire
        ({"fused_n": 150}, "covered 150"),  # (f) fused-assert under-coverage
        ({"marker_sep": "\n\n"}, "did NOT build"),  # (f) wrong construction
    ],
)
def test_smoke_gate_fails_loud(tmp_path: Path, overrides, match):
    with pytest.raises(SystemExit, match=match):
        _GATE.main(_gate_fixture(tmp_path, **overrides))


# ── analyzer registered rules on schema-real fixtures ────────────────────────

_ANALYZE = _load_module(
    SCRIPTS / "i613_sepablation_analyze.py", "i613_sepablation_analyze_under_test"
)


def _traj_fixture(*, dg, g_logp, margin, emission=0.0, per_q=None):
    b_logp = g_logp - dg
    per_q = per_q or {
        "e1": {
            "g_logp": g_logp,
            "b_logp": b_logp,
            "delta_g": dg,
            "argmax_marker": False,
            "n_marker_in_R": 0,
            "r_collapsed": False,
        },
        "e2": {
            "g_logp": g_logp,
            "b_logp": b_logp,
            "delta_g": dg,
            "argmax_marker": False,
            "n_marker_in_R": 0,
            "r_collapsed": False,
        },
    }
    return {
        "schema_version": "i472_v1",
        "sep": "",
        "checkpoints": [
            {
                "frac": 1.0,
                "step": 63,
                "source_self": {
                    "g_logp_mean": g_logp,
                    "b_logp_mean": b_logp,
                    "delta_g_mean": dg,
                    "emission_p": emission,
                    "r_collapsed": False,
                    "z_marker_g_mean": margin,
                    "z_marker_b_mean": 0.0,
                    "z_eos_g_mean": 0.0,
                    "z_eos_b_mean": 0.0,
                },
                "source_per_q": per_q,
                "held_out": {
                    "bob": {
                        "e1": {
                            "g_logp": -10.0,
                            "b_logp": -12.0,
                            "delta_g": 2.0,
                            "argmax_marker": False,
                            "n_marker_in_R": 0,
                            "r_collapsed": False,
                            "kl": 0.1,
                        },
                    }
                },
            }
        ],
    }


def _dense_fixture(*, tneg_terminal, tneg_peak=None):
    tneg_peak = tneg_terminal if tneg_peak is None else tneg_peak

    def _reads(tneg):
        leaf_keys = {
            "delta_g": 0.0,
            "delta_z_marker": 0.0,
            "delta_margin": 0.0,
            "z_eos_g": 0.0,
            "z_eos_b": 0.0,
        }
        return {
            "villain": {"e1": {**leaf_keys, "delta_g": 12.0}},
            "hero": {"e1": {**leaf_keys, "delta_g": tneg}},
            "bob": {"e1": {**leaf_keys, "delta_g": 4.0}},
        }

    return {
        "sep_mode": "plain",
        "sep": "",
        "source": "villain",
        "trained_negatives": ["hero"],
        "bystander_panel": ["hero", "bob"],
        "eval_questions": ["e1"],
        "checkpoints": [
            {"frac": 0.0158, "step": 1, "reads": _reads(tneg_peak)},
            {"frac": 1.0, "step": 63, "reads": _reads(tneg_terminal)},
        ],
    }


def _rowtype_fixture():
    return {
        "schema": "i601_rowtype_ce_v2",
        "n_neg_slot_rows": 16,
        "neg_slot_ce": [0.06],
        "neg_slot_ce_base": 0.05,
        "records": [{"step": 1, "neg_slot_ce": 0.06, "pos_marker_ce": 24.0}],
    }


def _write_round_fixture(root: Path, *, on, off, on_dense, off_dense):
    """on/off: {seed: traj-kwargs}; *_dense: {seed: dense-kwargs}."""
    for arm_cell, arm_traj, arm_dense in ((FLAGON, on, on_dense), (FLAGOFF, off, off_dense)):
        for seed in (42, 137):
            d = root / f"{arm_cell}_seed{seed}"
            d.mkdir(parents=True, exist_ok=True)
            (d / "trajectory.json").write_text(json.dumps(_traj_fixture(**arm_traj[seed])))
            (d / "dense_trajectory.json").write_text(json.dumps(_dense_fixture(**arm_dense[seed])))
            (d / "rowtype_ce.json").write_text(json.dumps(_rowtype_fixture()))


def _run_analyzer(tmp_path: Path) -> dict:
    out = tmp_path / "verdict.json"
    rc = _ANALYZE.main(["--round-root", str(tmp_path / "round"), "--out", str(out)])
    assert rc == 0
    return json.loads(out.read_text())


def test_r3_denominator_guard_maps_to_confirmed_strong(tmp_path: Path):
    # flag-on trained-neg terminal <= 0 with flag-off > 0: the registered
    # guard fires (predicted-success direction) — confirmed-strong, NO ratio.
    base = {"dg": 10.0, "g_logp": -8.0, "margin": 10.0}
    _write_round_fixture(
        tmp_path / "round",
        on={42: base, 137: {**base, "dg": 10.5}},
        off={42: {**base, "dg": 11.0}, 137: {**base, "dg": 11.5}},
        on_dense={42: {"tneg_terminal": -0.1}, 137: {"tneg_terminal": -0.2}},
        off_dense={42: {"tneg_terminal": 3.0}, 137: {"tneg_terminal": 3.2}},
    )
    v = _run_analyzer(tmp_path)
    r3 = v["r3_leakage_cut"]
    for seed in (42, 137):
        cell = r3["per_seed"][f"seed{seed}"]
        assert cell["classification"] == "confirmed-strong"
        assert cell["ratio_flagoff_over_flagon"] is None  # no ratio on a <=0 denominator
        assert cell["difference_flagoff_minus_flagon"] > 0
    assert r3["confirmed_both_seeds"] is True


def test_r3_ratio_gate_two_in_both_seeds(tmp_path: Path):
    base = {"dg": 10.0, "g_logp": -8.0, "margin": 10.0}
    _write_round_fixture(
        tmp_path / "round",
        on={42: base, 137: base},
        off={42: base, 137: base},
        on_dense={42: {"tneg_terminal": 1.0}, 137: {"tneg_terminal": 2.0}},
        off_dense={42: {"tneg_terminal": 2.5}, 137: {"tneg_terminal": 3.0}},
    )
    r3 = _run_analyzer(tmp_path)["r3_leakage_cut"]
    assert r3["per_seed"]["seed42"]["ratio_flagoff_over_flagon"] == pytest.approx(2.5)
    assert r3["per_seed"]["seed42"]["confirmed"] is True
    assert r3["per_seed"]["seed137"]["ratio_flagoff_over_flagon"] == pytest.approx(1.5)
    assert r3["per_seed"]["seed137"]["confirmed"] is False  # one-seed miss
    assert r3["confirmed_both_seeds"] is False


def test_r2_compression_window_margin_governs(tmp_path: Path):
    # Both arms' trained logP within tol of the 0 ceiling -> logP co-lands
    # MECHANICALLY; the margin twin (clearly separated) must govern the branch.
    on = {
        42: {"dg": 21.0, "g_logp": -0.5, "margin": 2.0},
        137: {"dg": 22.0, "g_logp": -0.7, "margin": 2.2},
    }
    off = {
        42: {"dg": 21.5, "g_logp": -0.4, "margin": 10.0},
        137: {"dg": 22.3, "g_logp": -0.6, "margin": 10.4},
    }
    dense = {42: {"tneg_terminal": 2.0}, 137: {"tneg_terminal": 2.0}}
    _write_round_fixture(tmp_path / "round", on=on, off=off, on_dense=dense, off_dense=dense)
    r2 = _run_analyzer(tmp_path)["r2_source_level"]
    assert r2["logp_branch"] == "co-land"  # the mechanical co-land
    assert r2["compression_window_fired"] is True
    assert r2["saturation_triage_fired"] is False  # window, not the extreme case
    assert r2["branch"] == "suppression"  # margin governs outright
    assert "margin" in r2["governed_by"]


def test_r2_co_land_requires_both_spaces(tmp_path: Path):
    # logP co-lands OFF the ceiling, but the margin twin does NOT -> the
    # registered both-spaces rule forbids a co-land verdict.
    on = {
        42: {"dg": 11.0, "g_logp": -9.0, "margin": 2.0},
        137: {"dg": 12.0, "g_logp": -9.5, "margin": 2.4},
    }
    off = {
        42: {"dg": 11.5, "g_logp": -9.2, "margin": 10.0},
        137: {"dg": 12.4, "g_logp": -9.8, "margin": 10.5},
    }
    dense = {42: {"tneg_terminal": 2.0}, 137: {"tneg_terminal": 2.0}}
    _write_round_fixture(tmp_path / "round", on=on, off=off, on_dense=dense, off_dense=dense)
    r2 = _run_analyzer(tmp_path)["r2_source_level"]
    assert r2["logp_branch"] == "co-land"
    assert r2["compression_window_fired"] is False
    assert r2["branch"] == "suppression"
    assert "co-land requires both" in r2["governed_by"]


def test_r2_emission_sensitivity_reports_exclusions_without_override(tmp_path: Path):
    per_q = {
        "e1": {
            "g_logp": -8.0,
            "b_logp": -18.0,
            "delta_g": 10.0,
            "argmax_marker": False,
            "n_marker_in_R": 0,
            "r_collapsed": False,
        },
        "e2": {
            "g_logp": -2.0,
            "b_logp": -20.0,
            "delta_g": 18.0,
            "argmax_marker": True,
            "n_marker_in_R": 3,
            "r_collapsed": False,
        },
    }
    on42 = {"dg": 14.0, "g_logp": -5.0, "margin": 10.0, "per_q": per_q}
    base = {"dg": 10.0, "g_logp": -8.0, "margin": 10.0}
    dense = {42: {"tneg_terminal": 2.0}, 137: {"tneg_terminal": 2.0}}
    _write_round_fixture(
        tmp_path / "round",
        on={42: on42, 137: {**base, "dg": 10.5}},
        off={42: base, 137: {**base, "dg": 10.2}},
        on_dense=dense,
        off_dense=dense,
    )
    r2 = _run_analyzer(tmp_path)["r2_source_level"]
    sens = r2["emission_denominator_sensitivity"]
    cell = sens["per_cell"]["flagon_seed42"]
    assert cell["n_probes_excluded"] == 1 and cell["n_probes_total"] == 2
    assert cell["delta_g_excluding_marker_in_R"] == pytest.approx(10.0)  # e2 dropped
    assert sens["can_override_primary"] is False  # no saturation triage, no collapse
    assert "branch" in sens  # paired sensitivity read reported alongside
