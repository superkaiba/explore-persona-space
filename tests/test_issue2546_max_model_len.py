"""Network-free, GPU-free pins for the issue-2546 r13 arm-2 P1 crash fix.

Incident (epm:progress v65, 2026-08-26): arm-2's P1 smoke pre side died rc=1 at
``build_engine`` -> vLLM ModelConfig validation — the module-global
``MAX_MODEL_LEN = 32768`` pin exceeds ``Qwen/Qwen2.5-Math-7B``'s
``max_position_embeddings`` (4,096), so the engine never built. Second defect:
plan §11's arm-2 pre ``cap=4096`` / ``regen_cap=8192`` are unsatisfiable inside
a 4,096-token total context (cap + any non-empty prompt > 4,096; regen_cap >
the whole context).

The r13 fix derives a per-model ``max_model_len = min(MAX_MODEL_LEN, own ctx)``
(``resolve_max_model_len``), clamps per-row generation budgets to it
(``row_max_tokens``, floor-asserted at ``GEN_FLOOR_TOKENS``), degrades the
compose-side drop budget to floor-preserving form when the declared regen_cap
does not fit (``prompt_budget``), and disables the regen rung where it has no
headroom (``select_regen_indices``).

The load-bearing SAFETY PROPERTY, pinned here: the derivation is a NO-OP for
every arm model except Math-7B — config-verified ``max_position_embeddings``
(2026-08-26): Qwen3-8B 40,960; DeepSeek-R1-Distill-Qwen-7B 131,072;
OpenThinker3-7B 32,768; Qwen2.5-7B-Instruct 32,768 — so arm 1 (LIVE,
mid-generation at the fix landing) and arm-2's already-verified post side keep
byte-identical budgets, caps, and regen selection.

Boundary fakes are signature-conformant only (transformers ``AutoConfig`` at
the HF-config boundary; a ``generate``-surface engine fake at the GPU
boundary; the run_generation externals mirrored from
``test_issue2546_reliability_draw``) — every asserted body is the real
production code (code-style.md "one production-body test per seam-stubbed
function").
"""

from __future__ import annotations

import argparse
import json
import sys
import types
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2546_gen_capture as G  # noqa: E402

# Config-verified per-model contexts (config.json max_position_embeddings,
# probed 2026-08-26 — the r13 brief's table, re-verified before use).
MODEL_CTX = {
    "Qwen/Qwen2.5-Math-7B": 4096,
    "Qwen/Qwen3-8B": 40960,
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": 131072,
    "open-thoughts/OpenThinker3-7B": 32768,
    "Qwen/Qwen2.5-7B-Instruct": 32768,
}
PRE_A2 = G.ARMS[2].sides[1]
assert PRE_A2.side == "pre" and PRE_A2.model == "Qwen/Qwen2.5-Math-7B"
UNCLAMPED_SIDES = [
    (arm, s)
    for arm in sorted(G.ARMS)
    for s in G.ARMS[arm].sides
    if not (arm == 2 and s.side == "pre")
]


class TestEffectiveLengthNoop:
    def test_min_is_noop_for_every_model_but_math7b(self):
        """The brief's safety property: min(32768, ctx) changes ONLY Math-7B."""
        for model, ctx in MODEL_CTX.items():
            eff = min(G.MAX_MODEL_LEN, ctx)
            if model == "Qwen/Qwen2.5-Math-7B":
                assert eff == 4096
            else:
                assert eff == G.MAX_MODEL_LEN, (model, ctx, eff)

    def test_every_arm_side_model_is_in_the_verified_table(self):
        """A future model swap must extend the verified-context table."""
        for arm in G.ARMS.values():
            for s in arm.sides:
                assert s.model in MODEL_CTX, s.model


class TestResolveMaxModelLen:
    def _patch_autoconfig(self, monkeypatch, cfg_obj):
        import transformers

        def fake_from_pretrained(model, *, revision=None, **kwargs):
            return cfg_obj

        monkeypatch.setattr(
            transformers.AutoConfig, "from_pretrained", staticmethod(fake_from_pretrained)
        )

    def test_clamps_to_model_context(self, monkeypatch):
        self._patch_autoconfig(monkeypatch, types.SimpleNamespace(max_position_embeddings=4096))
        assert G.resolve_max_model_len("Qwen/Qwen2.5-Math-7B", "r") == 4096

    def test_noop_above_pin(self, monkeypatch):
        self._patch_autoconfig(monkeypatch, types.SimpleNamespace(max_position_embeddings=131072))
        assert G.resolve_max_model_len("deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", "r") == 32768

    def test_text_config_nesting(self, monkeypatch):
        inner = types.SimpleNamespace(max_position_embeddings=40960)
        self._patch_autoconfig(monkeypatch, types.SimpleNamespace(text_config=inner))
        assert G.resolve_max_model_len("m", "r") == G.MAX_MODEL_LEN

    def test_missing_field_fails_loud(self, monkeypatch):
        """No silent fall-back to the pin — the pin IS the crash class."""
        self._patch_autoconfig(monkeypatch, types.SimpleNamespace())
        with pytest.raises(RuntimeError, match="max_position_embeddings"):
            G.resolve_max_model_len("m", "r")


class TestPromptBudget:
    def test_legacy_identity_for_unclamped_sides(self):
        """Byte-identity pin: every side except arm-2 pre keeps the EXACT
        pre-r13 drop budget (MAX_MODEL_LEN - 2*cap, the verify_plan c69
        arithmetic; regen_cap == 2*cap for every declared side)."""
        for arm, s in UNCLAMPED_SIDES:
            eff = min(G.MAX_MODEL_LEN, MODEL_CTX[s.model])
            assert eff == G.MAX_MODEL_LEN, (arm, s.side)
            assert s.regen_cap == 2 * s.cap, (arm, s.side)
            assert G.prompt_budget(s, eff) == G.MAX_MODEL_LEN - 2 * s.cap, (arm, s.side)

    def test_floor_branch_for_arm2_pre(self):
        """Clamped side: budget degrades to eff_len - GEN_FLOOR_TOKENS
        (4,096 - 2,048 = 2,048) instead of the negative legacy value."""
        assert G.prompt_budget(PRE_A2, 4096) == 4096 - G.GEN_FLOOR_TOKENS == 2048


class TestRowMaxTokens:
    def test_noop_when_room_exceeds_cap(self):
        assert G.row_max_tokens(8192, G.MAX_MODEL_LEN, 500) == 8192

    def test_clamps_to_room(self):
        assert G.row_max_tokens(4096, 4096, 300) == 3796
        assert G.row_max_tokens(8192, 4096, 300) == 3796  # regen budget, same room

    def test_below_floor_fails_loud(self):
        """The stated fail-loud floor (GEN_FLOOR_TOKENS): a row leaving less
        generation room than plan §11's own already-truncating bound must have
        been dropped at compose time — never silently truncated."""
        with pytest.raises(RuntimeError, match="floor"):
            G.row_max_tokens(4096, 4096, 4096 - G.GEN_FLOOR_TOKENS + 1)


_PRIM = [("a", "stop", 10), ("b", "length", 99), ("c", "length", 99), ("d", "stop", 5)]


class TestSelectRegenIndices:
    PRIM = tuple(_PRIM)

    def test_equals_pre_fix_rule_with_headroom(self):
        """Unclamped sides: regen selection reduces EXACTLY to the pre-r13
        rule (every finish_reason=='length' row) — the live-arm identity."""
        caps = [8192] * 4
        regen = [16384] * 4
        assert G.select_regen_indices(self.PRIM, caps, regen) == [1, 2]

    def test_no_headroom_skips_all(self):
        """Context-clamped side: regen budget == primary budget for every row
        -> nothing regens; cap-hit rows stay length-residuals for the
        per-cell caphit instrument."""
        caps = [3796] * 4
        assert G.select_regen_indices(self.PRIM, caps, list(caps)) == []

    def test_mixed_headroom(self):
        caps = [3796, 3796, 2048, 3796]
        regen = [3796, 3796, 4096, 3796]
        assert G.select_regen_indices(self.PRIM, caps, regen) == [2]


class _FakeEngine:
    """GPU-boundary fake mirroring the ONE call surface generate_chunked uses:
    ``generate(prompts, sampling_params, use_tqdm=False)``. Records the
    per-chunk sampling_params shape so the list-slicing contract is asserted
    against the REAL generate_chunked body."""

    def __init__(self):
        self.calls: list[tuple[int, object]] = []

    def generate(self, prompts, sampling_params, use_tqdm=False):
        assert use_tqdm is False
        self.calls.append((len(prompts), sampling_params))
        out = []
        for p in prompts:
            o = types.SimpleNamespace(text=f"gen:{p}", finish_reason="stop", token_ids=[1, 2])
            out.append(types.SimpleNamespace(outputs=[o]))
        return out


class TestGenerateChunkedSpSlicing:
    def test_list_sp_slices_with_chunks(self, monkeypatch):
        monkeypatch.setattr(G, "VLLM_CHUNK_SIZE", 2)
        eng = _FakeEngine()
        prompts = [f"p{i}" for i in range(5)]
        sp = [f"sp{i}" for i in range(5)]  # placeholder objects; only slicing is asserted
        out = G.generate_chunked(eng, prompts, sp, "t")
        assert [t for t, _fr, _n in out] == [f"gen:p{i}" for i in range(5)]
        assert [(n, s) for n, s in eng.calls] == [
            (2, ["sp0", "sp1"]),
            (2, ["sp2", "sp3"]),
            (1, ["sp4"]),
        ]

    def test_single_sp_passes_through(self, monkeypatch):
        monkeypatch.setattr(G, "VLLM_CHUNK_SIZE", 2)
        eng = _FakeEngine()
        G.generate_chunked(eng, ["p0", "p1", "p2"], "SP", "t")
        assert [s for _n, s in eng.calls] == ["SP", "SP"]


class _MiniTok:
    """Signature-conformant tokenizer boundary fake for the arm-2 PRE side
    (default chat template, no think pins — mirrors the two call surfaces
    compose_prompts uses)."""

    def apply_chat_template(
        self, conversation, *, tokenize=False, add_generation_prompt=False, **kw
    ):
        assert tokenize is False and add_generation_prompt is True
        return "<|im_start|>user\n" + conversation[0]["content"] + "\n<|im_start|>assistant\n"

    def __call__(self, text: str, add_special_tokens: bool = True) -> dict:
        assert add_special_tokens is False
        return {"input_ids": list(range(len(text.split())))}


class _StopAtWorkers(RuntimeError):
    pass


class TestWorkFileCarriesResolvedLength:
    def test_run_generation_threads_clamped_length_into_work_file(self, monkeypatch, tmp_path):
        """REAL run_generation body up to the spawn boundary, on the CRASHED
        side (arm-2 pre) with the resolver faked to Math-7B's real 4,096:
        the per-slot work file must carry max_model_len=4096 (what the worker
        passes to build_engine — the exact pre-fix crash point)."""
        import transformers

        monkeypatch.setattr(G, "resolve_revision", lambda model, out_root: "test-revision")
        monkeypatch.setattr(G, "resolve_stop_ids", lambda model, revision: [151645])
        monkeypatch.setattr(G, "resolve_max_model_len", lambda model, revision: 4096)

        def fake_spawn(script_args, work_files, out_root_, tag):
            raise _StopAtWorkers(tag)

        monkeypatch.setattr(G, "spawn_workers", fake_spawn)
        monkeypatch.setattr(
            transformers.AutoTokenizer,
            "from_pretrained",
            staticmethod(lambda model, *, revision=None, **kw: _MiniTok()),
        )
        rows = {
            "mmlu": [
                {
                    "row_id": f"mmlu:{i:04d}",
                    "corpus": "mmlu",
                    "user_text": f"mmlu question {i}",
                    "question": f"mmlu question {i}",
                    "gold_answer": "A",
                    "in_arm12": True,
                    "in_arm3": True,
                }
                for i in range(4)
            ]
        }
        args = argparse.Namespace(
            smoke=True, decode_fallback=False, prefill_fallback=False, phase="p1_smoke"
        )
        with pytest.raises(_StopAtWorkers):
            G.run_generation(args, G.ARMS[2], PRE_A2, rows, tmp_path, rel_total=0, num_workers=1)
        wf = tmp_path / "work" / "p1_smoke_pre" / "slot0.json"
        work = json.loads(wf.read_text())
        assert work["max_model_len"] == 4096
        assert work["cap"] == PRE_A2.cap and work["regen_cap"] == PRE_A2.regen_cap
        # The composed rows all fit the floor-preserving budget (4,096-2,048).
        for r in work["rows"]:
            assert r["n_prompt_tokens"] <= 4096 - G.GEN_FLOOR_TOKENS
            # Worker-side derivation over this work file: primary and regen
            # budgets clamp to the SAME room -> regen structurally disabled.
            cap_row = G.row_max_tokens(work["cap"], 4096, r["n_prompt_tokens"])
            regen_row = G.row_max_tokens(work["regen_cap"], 4096, r["n_prompt_tokens"])
            assert cap_row == regen_row == 4096 - r["n_prompt_tokens"]
