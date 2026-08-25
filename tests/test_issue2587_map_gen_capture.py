"""CPU pins for ``scripts/issue2587_map_gen_capture.py`` (plan v3 §4.3 — unit 2).

No network, no HF fetch, no GPU (the unit-1 fake-tokenizer pattern: deterministic
ids at the chat-template boundary; manifest downloads monkeypatched). Pins:

- the plan-§4/§11/§12 constants: PROMPT_TOKEN_BUDGET arithmetic (7,104),
  GEN seed/temp/top_p, SPLIT_TO_MANIFEST (train_25k added at seed 42;
  #2330's train_10k REMOVED), LENGTH_SCAN_KEYS, PINNED_MANIFEST_COUNTS
  (27,399 pre-scan rows at the 815ff6d manifest pin), P1_SENTINEL_REQUIRED;
- the split_ids.json schema (``issue2587_split_ids_v1``): per-split ORDERED id
  lists a downstream consumer (the §4.5(b) matched-row 7B arm) can compare
  ordered-set-exactly, sha256 in the compact-JSON domain (``_sha_ids``),
  counts, dropped_overlength drop records;
- the P0b gate's three branches: bootstrap-then-PASS, drop-and-recompute,
  >max-over-budget-frac HALT (exit 4, split_ids NOT mutated);
- unit-1 wiring by import: ``_render_prompt`` passes enable_thinking=False and
  runs the closed-empty-<think> assert; ``_build_engine`` threads
  ``ENGINE_KWARG_PINS`` (gdn_prefill_backend="triton"); the LAUNCH_ENV_PINS
  setdefault at module import;
- CLI surface: ``--hf-prefix`` default None (the #1005 upload-prefix clobber
  shape), split-ids default path, main's sentinel-path resolution.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import issue2378_common as cm2378  # noqa: E402
import issue2587_common as cm2587  # noqa: E402
import issue2587_map_gen_capture as M  # noqa: E402

# ── plan-pinned constants ───────────────────────────────────────────────


def test_prompt_token_budget_arithmetic():
    assert M.MAX_MODEL_LEN == 8192
    assert M.GEN_MAX_TOKENS == 1024
    assert M.LENGTH_MARGIN == 64
    assert M.PROMPT_TOKEN_BUDGET == 7104
    assert M.PROMPT_TOKEN_BUDGET == M.MAX_MODEL_LEN - M.GEN_MAX_TOKENS - M.LENGTH_MARGIN


def test_sampling_pins():
    assert M.GEN_TEMP == 1.0
    assert M.GEN_TOP_P == 0.95
    assert M.GEN_SEED_DEFAULT == 42


def test_split_to_manifest_train25k_added_train10k_removed():
    # The plan-§4.3 3-line entry: full 25k manifest list at the default seed.
    assert M.SPLIT_TO_MANIFEST["train_25k"] == ("train_25k", "train_25k", 42)
    # #2330's train_10k split can never resolve in this issue's split_ids
    # payload — keeping it would ship a selectable always-crashing choice.
    assert "train_10k" not in M.SPLIT_TO_MANIFEST
    assert M.SPLIT_TO_MANIFEST["ceiling_draw_43"] == ("test_1000", "test_1000", 43)
    assert M.SPLIT_TO_MANIFEST["ceiling_draw_44"] == ("test_1000", "test_1000", 44)


def test_length_scan_keys_and_pinned_counts():
    assert M.LENGTH_SCAN_KEYS == ("train_25k", "val_400", "test_1000", "wc_test_1k")
    assert M.PINNED_MANIFEST_COUNTS == {
        "train_25k": 25000,
        "val_400": 400,
        "test_1000": 1000,
        "wc_test_1k": 999,
    }
    assert sum(M.PINNED_MANIFEST_COUNTS.values()) == 27399  # plan §12 pre-scan total
    assert M.MANIFEST_REVISION == "815ff6d976c686af8672b27cfdfb1ce6b419c02c"


def test_p1_sentinel_required_gates():
    assert M.P1_SENTINEL_REQUIRED == ("template_pin", "length_scan", "hook_probe")


def test_store_subpath_routing():
    assert M.store_subpath_for_split("train_25k") == "train_25k"
    assert M.store_subpath_for_split("wc_test_1k") == "wc_test_1k"
    assert M.store_subpath_for_split("ceiling_draw_43") == "ceiling_draws/seed43"
    assert M.store_subpath_for_split("ceiling_draw_44") == "ceiling_draws/seed44"


def test_cap_hit_schema_literal_names_the_format():
    # The literal deliberately keeps the #2330 token — it names the FORMAT.
    assert M.CAP_HIT_SCHEMA == "issue2330_cap_hit_v2"


def test_sha_ids_is_compact_json_domain():
    # The downstream ordered-set-exact comparison keys on THIS domain: sha256
    # of the compact-JSON id list (no spaces), never the pretty-printed form.
    ids = [3, 1, 2]
    assert M._sha_ids(ids) == hashlib.sha256(b"[3,1,2]").hexdigest()
    assert M._sha_ids(ids) != hashlib.sha256(b"[3, 1, 2]").hexdigest()


def test_launch_env_pins_applied_at_import():
    # Module import setdefaults every unit-1 launch pin into os.environ.
    import os

    assert M.cm2587 is cm2587
    for k in cm2587.LAUNCH_ENV_PINS:
        assert k in os.environ, f"launch pin {k} absent after module import"


# ── unit-1 render wiring (fake tokenizer at the chat-template boundary) ──


class _FakeTemplateTok:
    """apply_chat_template-shaped fake (the unit-1 pattern): asserts
    enable_thinking=False is threaded; renders content + a fixed tail."""

    def __init__(self, tail: str):
        self.tail = tail

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        assert kw.get("enable_thinking") is False, "render must pass enable_thinking=False"
        return msgs[0]["content"] + self.tail

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": list(range(max(4, len(text) // 8)))}


def test_render_prompt_runs_closed_empty_think_assert():
    closed = _FakeTemplateTok("<|im_start|>assistant\n<think>\n\n</think>\n\n")
    out = M._render_prompt(closed, "hi")
    assert out.endswith("</think>\n\n")
    plain = _FakeTemplateTok("<|im_start|>assistant\n")
    assert M._render_prompt(plain, "hi").endswith("assistant\n")  # no-op when absent
    with pytest.raises(AssertionError, match="OPEN thinking block"):
        M._render_prompt(_FakeTemplateTok("<|im_start|>assistant\n<think>\n"), "hi")
    with pytest.raises(AssertionError, match="non-empty thinking block"):
        M._render_prompt(
            _FakeTemplateTok("<|im_start|>assistant\n<think>\nreasoning\n</think>\n\n"), "hi"
        )


def test_filter_overlength_prompts_partitions_and_records_digest_only():
    kept_p, kept_c, skipped = M._filter_overlength_prompts(
        ["a", "b", "c"], [1, 2, 3], lambda p: {"a": 10, "b": 9000, "c": 7104}[p], 7104
    )
    assert kept_p == ["a", "c"] and kept_c == [1, 3]
    assert skipped == [{"ci": 2, "n_tokens": 9000}]  # ci + count, never text


# ── engine kwarg pins (fake vllm module at the import boundary) ─────────


class _CaptureLLM:
    def __init__(self, **kw):
        self.kw = kw


def _fake_vllm(monkeypatch):
    mod = types.ModuleType("vllm")
    mod.LLM = _CaptureLLM
    monkeypatch.setitem(sys.modules, "vllm", mod)


def test_build_engine_threads_engine_kwarg_pins(monkeypatch):
    _fake_vllm(monkeypatch)
    monkeypatch.delenv("EPM_VLLM_ENFORCE_EAGER", raising=False)
    monkeypatch.delenv("EPM_VLLM_DISABLE_PREFIX_CACHING", raising=False)
    monkeypatch.delenv("VLLM_GPU_MEM_UTIL", raising=False)
    monkeypatch.setattr(M, "_ENGINE_CONSTRUCTED", False)
    monkeypatch.setattr(M, "_LIVE_ENGINE", None)
    eng = M._build_engine("fake/q35", 7)
    # §4.1 pins BY IMPORT (identity with the #2378 source of truth).
    for k, v in cm2378.ENGINE_KWARG_PINS.items():
        assert eng.kw[k] == v
    assert eng.kw["gdn_prefill_backend"] == "triton"
    assert eng.kw["seed"] == 7
    assert eng.kw["dtype"] == "bfloat16"
    assert eng.kw["max_model_len"] == M.MAX_MODEL_LEN
    assert eng.kw["gpu_memory_utilization"] == pytest.approx(0.60)
    assert "enforce_eager" not in eng.kw and "enable_prefix_caching" not in eng.kw
    assert M._ENGINE_CONSTRUCTED is True and M._LIVE_ENGINE is eng


def test_build_engine_env_mitigation_knobs(monkeypatch):
    _fake_vllm(monkeypatch)
    monkeypatch.setenv("EPM_VLLM_ENFORCE_EAGER", "1")
    monkeypatch.setenv("EPM_VLLM_DISABLE_PREFIX_CACHING", "1")
    monkeypatch.setattr(M, "_ENGINE_CONSTRUCTED", False)
    monkeypatch.setattr(M, "_LIVE_ENGINE", None)
    eng = M._build_engine("fake/q35", 42)
    assert eng.kw["enforce_eager"] is True
    assert eng.kw["enable_prefix_caching"] is False
    assert eng.kw["gdn_prefill_backend"] == "triton"  # pins survive the knobs


# ── CLI surface ─────────────────────────────────────────────────────────


def test_parser_defaults():
    args = M._build_parser().parse_args([])
    # #1005 upload-prefix clobber shape: --hf-prefix has NO default by design.
    assert args.hf_prefix is None
    assert str(args.split_ids).endswith("eval_results/issue_2587/split_ids.json")
    assert args.sentinel_path is None  # resolved in main to <out-dir>/split_ids_done.json
    assert args.max_over_budget_frac == 0.005  # plan §7: >0.5% over budget HALTS
    assert args.model == "Qwen/Qwen3.5-9B"
    assert args.gen_max_tokens == M.GEN_MAX_TOKENS


def test_main_resolves_sentinel_and_run_meta_defaults(monkeypatch, tmp_path):
    captured: dict = {}

    def fake_fits_smoke(args):
        captured["args"] = args
        return 0

    monkeypatch.setattr(M, "_run_fits_smoke", fake_fits_smoke)
    monkeypatch.setattr(
        sys, "argv", ["issue2587_map_gen_capture.py", "--fits-smoke", "--out-dir", str(tmp_path)]
    )
    assert M.main() == 0
    args = captured["args"]
    assert args.sentinel_path == tmp_path / "split_ids_done.json"  # plan §9 p0b_gates sentinel
    assert args.run_meta_out == tmp_path / "run_meta.json"


# ── P0b length-scan gate (bootstrap / drop / HALT) ──────────────────────

_LEN_RE = re.compile(r"L(\d+)")


class _LenTok:
    """Deterministic-ids fake: rendered token length is encoded in the prompt
    text itself (``L<n>``), so the scan arithmetic is fully controlled while
    the REAL _render_prompt (enable_thinking=False + closed-empty-<think>
    assert) still executes per row."""

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, **kw):
        assert kw.get("enable_thinking") is False
        return msgs[0]["content"] + M.THINK_SUFFIX_TEXT

    def __call__(self, text, add_special_tokens=False):
        m = _LEN_RE.search(text)
        return {"input_ids": [0] * (int(m.group(1)) if m else 4)}


_FAKE_IDS = {
    "train_25k": [10, 11, 12, 13, 14, 15],
    "val_400": [0, 1, 2],
    "test_1000": [5, 6, 7, 8],
    "wc_test_1k": [20, 21, 22],
}
_FAKE_COUNTS = {k: len(v) for k, v in _FAKE_IDS.items()}


def _fake_manifests(over_len: dict[int, int] | None = None) -> dict[str, list[dict]]:
    over_len = over_len or {}
    out: dict[str, list[dict]] = {}
    for key, ids in _FAKE_IDS.items():
        out[key] = [{"ladder_local_id": i, "prompt": f"q{i} L{over_len.get(i, 100)}"} for i in ids]
    return out


def _wire_gate(monkeypatch, tmp_path, over_len=None):
    manifests = _fake_manifests(over_len)
    monkeypatch.setattr(M, "PINNED_MANIFEST_COUNTS", _FAKE_COUNTS)
    monkeypatch.setattr(M, "_download_manifest_split", lambda key, cache_dir: manifests[key])
    monkeypatch.setattr(
        M, "_load_tokenizer", lambda model, sfx: (_LenTok(), M.THINK_SUFFIX_TEXT, [1, 2])
    )


def _gate_args(tmp_path, extra: list[str] | None = None):
    args = M._build_parser().parse_args(["--gate", "length_scan", *(extra or [])])
    args.out_dir = tmp_path
    args.run_meta_out = tmp_path / "run_meta.json"
    args.sentinel_path = tmp_path / "split_ids_done.json"
    args.split_ids = str(tmp_path / "split_ids.json")
    return args


def test_gate_bootstrap_then_pass_pins_split_ids_schema(monkeypatch, tmp_path):
    _wire_gate(monkeypatch, tmp_path)
    args = _gate_args(tmp_path)
    assert M.gate_length_scan(args) == 0

    payload = json.loads((tmp_path / "split_ids.json").read_text())
    # Schema pin (issue2587_split_ids_v1): the §4.5(b) matched-row 7B arm
    # compares these ordered id lists ordered-set-exactly.
    assert payload["schema"] == "issue2587_split_ids_v1"
    assert payload["issue"] == 2587
    assert payload["manifest_hf_prefix"] == M.MANIFEST_HF_PREFIX
    assert payload["manifest_revision"] == M.MANIFEST_REVISION
    assert set(payload["splits"]) == set(M.LENGTH_SCAN_KEYS)
    for key, ids in _FAKE_IDS.items():
        assert payload["splits"][key] == ids  # ORDERED, manifest order, exact
        assert payload["sha256"][key] == M._sha_ids(ids)
        assert payload["counts"][key] == len(ids)
    assert "ts_utc" in payload
    assert "dropped_overlength" not in payload  # nothing dropped on this branch

    meta = json.loads((tmp_path / "run_meta.json").read_text())
    rec = meta["length_scan"]
    assert rec["passed"] is True
    assert rec["scanned"] == sum(_FAKE_COUNTS.values()) and rec["over_budget"] == 0
    assert rec["budget"] == 7104
    # Only length_scan has passed — the sentinel writer must NOT fire yet.
    assert not (tmp_path / "split_ids_done.json").exists()


def test_gate_bootstrap_halts_on_pinned_count_drift(monkeypatch, tmp_path):
    _wire_gate(monkeypatch, tmp_path)
    bad = dict(_FAKE_COUNTS, train_25k=999999)
    monkeypatch.setattr(M, "PINNED_MANIFEST_COUNTS", bad)
    with pytest.raises(AssertionError, match="pinned manifest count drift"):
        M.gate_length_scan(_gate_args(tmp_path))
    assert not (tmp_path / "split_ids.json").exists()  # bootstrap never wrote


def test_gate_bootstrap_halts_on_duplicate_manifest_ids(monkeypatch, tmp_path):
    _wire_gate(monkeypatch, tmp_path)
    manifests = _fake_manifests()
    manifests["val_400"].append(dict(manifests["val_400"][0]))  # duplicate id 0
    monkeypatch.setattr(M, "_download_manifest_split", lambda key, cache_dir: manifests[key])
    with pytest.raises(AssertionError, match="duplicate ladder_local_id"):
        M.gate_length_scan(_gate_args(tmp_path))


def test_gate_drop_path_recomputes_shas_counts_and_records_drops(monkeypatch, tmp_path):
    # id 12 renders at 8000 > 7104; 1/16 = 6.25% needs a widened band to DROP.
    _wire_gate(monkeypatch, tmp_path, over_len={12: 8000})
    args = _gate_args(tmp_path, ["--max-over-budget-frac", "0.10"])
    assert M.gate_length_scan(args) == 0

    payload = json.loads((tmp_path / "split_ids.json").read_text())
    kept = [10, 11, 13, 14, 15]
    assert payload["splits"]["train_25k"] == kept  # order preserved, 12 gone
    assert payload["sha256"]["train_25k"] == M._sha_ids(kept)  # recomputed
    assert payload["counts"]["train_25k"] == 5
    assert payload["dropped_overlength"]["train_25k"] == [{"id": 12, "n_tokens": 8000}]
    # Untouched splits keep their full lists + shas.
    assert payload["splits"]["val_400"] == _FAKE_IDS["val_400"]
    assert payload["sha256"]["val_400"] == M._sha_ids(_FAKE_IDS["val_400"])
    assert payload["length_scan"]["over_budget"] == 1
    assert json.loads((tmp_path / "run_meta.json").read_text())["length_scan"]["passed"] is True

    # Idempotent re-run: post-drop lists re-scan to 0 over budget, no new drops.
    assert M.gate_length_scan(args) == 0
    payload2 = json.loads((tmp_path / "split_ids.json").read_text())
    assert payload2["splits"]["train_25k"] == kept
    assert payload2["dropped_overlength"]["train_25k"] == [{"id": 12, "n_tokens": 8000}]


def test_gate_halts_exit_4_without_mutating_split_ids(monkeypatch, tmp_path):
    # 1/16 = 6.25% over budget > the default 0.5% band -> HALT (plan §7).
    _wire_gate(monkeypatch, tmp_path, over_len={12: 8000})
    args = _gate_args(tmp_path)  # default --max-over-budget-frac 0.005
    assert args.max_over_budget_frac == 0.005

    # Pre-write the bootstrap so the HALT's no-mutation contract is observable.
    cache = tmp_path / ".cache"
    cache.mkdir(parents=True, exist_ok=True)
    M._bootstrap_split_ids(Path(args.split_ids), cache)
    before = (tmp_path / "split_ids.json").read_bytes()

    assert M.gate_length_scan(args) == 4
    assert (tmp_path / "split_ids.json").read_bytes() == before  # NOT mutated
    meta = json.loads((tmp_path / "run_meta.json").read_text())
    assert meta["length_scan"]["passed"] is False  # audit record persists
    assert not (tmp_path / "split_ids_done.json").exists()


# ── P0b sentinel writer ─────────────────────────────────────────────────


def test_sentinel_written_only_when_all_three_gates_pass(monkeypatch, tmp_path):
    split_ids = tmp_path / "split_ids.json"
    payload = {
        "schema": "issue2587_split_ids_v1",
        "issue": 2587,
        "splits": {"train_25k": [1, 2]},
        "sha256": {"train_25k": M._sha_ids([1, 2])},
        "counts": {"train_25k": 2},
    }
    split_ids.write_text(json.dumps(payload))
    run_meta = tmp_path / "run_meta.json"
    args = SimpleNamespace(
        run_meta_out=run_meta,
        split_ids=str(split_ids),
        sentinel_path=tmp_path / "split_ids_done.json",
    )
    monkeypatch.setattr(
        M,
        "_hf_api",
        lambda: SimpleNamespace(model_info=lambda mid: SimpleNamespace(sha="fakehubsha")),
    )
    monkeypatch.setattr(M, "_git_sha", lambda: "testgitsha")

    # Pending: only length_scan has passed -> no sentinel, no HF call needed.
    run_meta.write_text(json.dumps({"length_scan": {"passed": True}}))
    M._maybe_write_p1_sentinel(args)
    assert not args.sentinel_path.exists()

    run_meta.write_text(
        json.dumps(
            {
                "template_pin": {"passed": True, "model": "fake/q35"},
                "length_scan": {"passed": True},
                "hook_probe": {"passed": True},
            }
        )
    )
    M._maybe_write_p1_sentinel(args)
    sentinel = json.loads(args.sentinel_path.read_text())
    assert sentinel["schema"] == "issue2587_p0b_gates_v1"
    assert sentinel["issue"] == 2587 and sentinel["phase"] == "P0b"
    assert sentinel["status"] == "PASS"
    assert sentinel["model"] == "fake/q35" and sentinel["model_hf_sha"] == "fakehubsha"
    assert tuple(sentinel["gates"]) == M.P1_SENTINEL_REQUIRED
    assert sentinel["split_ids_sha256_per_split"] == payload["sha256"]
    assert sentinel["split_ids_file_sha256"] == hashlib.sha256(split_ids.read_bytes()).hexdigest()
