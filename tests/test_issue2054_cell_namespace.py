"""C6/C7 pins for the task #2054 4-axis cell namespace (revise round 2, Unit C).

C6 — the CONDITION (capture --phase) and FRAMING (--form) axes are part of
every output identity: capture .npz / diagnostics names embed phase AND form
(via the canonical `issue2054_forms.cell_key`), two runs differing only in
(phase, form) land on DISTINCT paths, and key collisions are impossible BY
CONSTRUCTION (closed condition/form registries + separator-free free axes).

C7 — the PRODUCTION capture diagnostics payload carries the `per_row` block
kill-gate 5 reads (`issue2054_fits._answer_length_ks_from_diagnostics`); the
shared `_diagnostics_payload` builder serializes it unconditionally and
`_write_diagnostics` REFUSES a payload without it (pre-fix, the production
payload omitted `per_row`, so the gate always read empty-length-arrays ->
KS=NaN -> could never fire).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue2054_capture as capture  # noqa: E402
import issue2054_fits as fits  # noqa: E402
import issue2054_forms as forms  # noqa: E402
import issue2054_ladder as ladder  # noqa: E402

# ─────────────────────────────────────────────────────────────────────────────
# C6: cell_key — 4 axes, collision-impossible by construction


def test_cell_key_embeds_all_four_axes():
    key = forms.cell_key("char_helios", "inserted", "chat", "qwen2.5-7b")
    assert key == "char_helios__inserted__chat__qwen2.5-7b"
    for axis_value in ("char_helios", "inserted", "chat", "qwen2.5-7b"):
        assert axis_value in key.split(forms.CELL_KEY_SEP)


def test_cell_key_distinct_across_condition_and_form():
    keys = {
        forms.cell_key("char_helios", condition, form, "qwen2.5-7b")
        for condition in forms.CONDITIONS
        for form in forms.FORMS
    }
    assert len(keys) == len(forms.CONDITIONS) * len(forms.FORMS)


@pytest.mark.parametrize(
    "variant,condition,form,model",
    [
        ("char_helios", "not_a_condition", "chat", "qwen2.5-7b"),
        ("char_helios", "inserted", "not_a_form", "qwen2.5-7b"),
        ("bad__variant", "inserted", "chat", "qwen2.5-7b"),
        ("char_helios", "inserted", "chat", "bad__model"),
        ("", "inserted", "chat", "qwen2.5-7b"),
    ],
)
def test_cell_key_refuses_ambiguous_or_unknown_axes(variant, condition, form, model):
    with pytest.raises(ValueError):
        forms.cell_key(variant, condition, form, model)


def test_phase_output_name_embeds_form_and_is_condition_scoped():
    assert (
        forms.phase_output_name("inserted", "char_helios", "chat")
        == "spliced_inserted_char_helios__chat.jsonl"
    )
    assert (
        forms.phase_output_name("on_policy", "char_helios", "bare_text", mock=True)
        == "on_policy_char_helios__bare_text.mock.jsonl"
    )
    assert (
        forms.phase_output_name("cell_c", "char_helios_op", "chat")
        == "cell_c_char_helios_op__chat.jsonl"
    )
    # Distinct forms -> distinct producer files for one condition+variant.
    assert forms.phase_output_name("inserted", "v", "chat") != forms.phase_output_name(
        "inserted", "v", "bare_text"
    )
    with pytest.raises(ValueError):
        forms.phase_output_name("not_a_condition", "v", "chat")
    with pytest.raises(ValueError):
        forms.phase_output_name("inserted", "v", "chat", mock=True)  # mock is on_policy-only


# ─────────────────────────────────────────────────────────────────────────────
# C6: capture output paths — two (phase, form) runs of one variant never clobber


def _capture_args(phase: str, form: str) -> argparse.Namespace:
    return argparse.Namespace(
        phase=phase,
        form=form,
        layer=19,
        seed=137,
        dry_run=True,
        target_conv_ids=0,
        batch_size=8,
        skip_upload=True,
        upload=False,
        model="qwen2.5-7b",
    )


def test_capture_dry_run_paths_embed_phase_and_form_and_never_clobber(tmp_path, monkeypatch):
    """Run the REAL dry-run handler twice (only the tokenizer boundary faked)
    under (inserted, chat) then (on_policy, bare_text): both output sets must
    coexist at distinct, axis-bearing paths — the C6 overwrite is impossible.
    """
    monkeypatch.setattr(capture, "_load_tokenizer", lambda model_id: object())
    # Position resolution is tokenizer-bound (untouched this round); fake it
    # with a shape-conformant per-row block so the naming + payload + writer
    # path runs for real.
    per_row = [
        {
            "conv_id": "s0001",
            "status": "ok",
            "n_tokens": 64,
            "answer_lo": 10,
            "answer_hi": 40,
            "v_C_pos": 9,
            "v_P_pos": 3,
            "prefix_src": "recorded",
        }
    ]
    monkeypatch.setattr(
        capture, "_capture_positions_only", lambda tokenizer, rows, sample_n: (per_row, 1, 0, 0)
    )

    input_path = tmp_path / "in.jsonl"
    input_path.write_text('{"conv_id": "s0001"}\n', encoding="utf-8")
    out_dir = tmp_path / "acts"
    diag_dir = tmp_path / "diags"

    reports = []
    for phase, form in (("inserted", "chat"), ("on_policy", "bare_text")):
        reports.append(
            capture._run_dry_variant(
                "char_helios",
                "qwen2.5-7b",
                input_path,
                out_dir,
                diag_dir,
                _capture_args(phase, form),
            )
        )

    shells = sorted(p.name for p in (out_dir / "char_helios").glob("*.npz"))
    diags = sorted(p.name for p in diag_dir.glob("*.json"))
    assert shells == [
        "char_helios__inserted__chat__qwen2.5-7b.npz",
        "char_helios__on_policy__bare_text__qwen2.5-7b.npz",
    ]
    assert diags == [
        "char_helios__inserted__chat__qwen2.5-7b.json",
        "char_helios__on_policy__bare_text__qwen2.5-7b.json",
    ]
    # Each path embeds BOTH the condition and the form axis.
    for name in shells + diags:
        parts = name.rsplit(".", 1)[0].split(forms.CELL_KEY_SEP)
        assert len(parts) == 4, name
        assert parts[1] in forms.CONDITIONS and parts[2] in forms.FORMS, name
    assert reports[0]["cell"] != reports[1]["cell"]


# ─────────────────────────────────────────────────────────────────────────────
# C7: the production diagnostics payload carries the gate-5 length source


def _production_payload(condition: str, lengths: list[int]) -> dict:
    per_row = [
        {
            "conv_id": f"s{i:04d}",
            "status": "ok",
            "n_tokens": n + 24,
            "answer_lo": 10,
            "answer_hi": 10 + n,
            "v_C_pos": 9,
            "v_P_pos": 3,
            "prefix_src": "recorded",
        }
        for i, n in enumerate(lengths)
    ]
    return capture._diagnostics_payload(
        dry_run=False,  # the PRODUCTION shape — pre-fix this payload had no per_row
        variant="char_helios",
        condition=condition,
        form="chat",
        model_slug="qwen2.5-7b",
        input_path=Path("in.jsonl"),
        activation_path=Path("out.npz"),
        layer=19,
        seed=137,
        n_in=len(lengths),
        n_ok=len(lengths),
        n_skipped=0,
        n_prefix_null=0,
        per_row=per_row,
        lengths=lengths,
        conv_ids=[r["conv_id"] for r in per_row],
        extra={"batch_size": 8, "n_processed": len(lengths)},
    )


def test_production_diagnostics_payload_feeds_kill_gate_5():
    """Round-trip: a production-shaped capture payload through the fits gate-5
    reader must yield a FINITE KS statistic (pre-fix: empty-length-arrays ->
    NaN -> gate5_fire could never be True on production data).
    """
    payload_b = _production_payload("inserted", [30, 40, 50, 60, 70])
    payload_d = _production_payload("on_policy", [300, 400, 500, 600, 700])
    ks_d, ratio, info = fits._answer_length_ks_from_diagnostics(payload_b, payload_d)
    assert info["status"] == "computed"
    assert ks_d == ks_d and ratio == ratio  # finite (not NaN)
    gate = fits._evaluate_kill_gates(
        variant="char_helios",
        condition="inserted",
        form="chat",
        model="qwen2.5-7b",
        conv_ids_this_cell=payload_b["conv_ids"],
        peer_cells_conv_ids=None,
        diag_this=payload_b,
        peer_diag=payload_d,
        gate5_peer_cell=forms.cell_key("char_helios", "on_policy", "chat", "qwen2.5-7b"),
    )
    assert gate["kill_gate_5_ks_d"] is not None
    # 10x mean-length disparity is outside the [0.25, 4.0] ratio bounds.
    assert gate["kill_gate_5_fire"] is True


def test_write_diagnostics_refuses_payload_missing_gate5_source(tmp_path):
    payload = _production_payload("inserted", [30, 40])
    payload.pop("per_row")  # the pre-fix production shape
    with pytest.raises(ValueError, match="per_row"):
        capture._write_diagnostics(tmp_path / "d.json", payload)


# ─────────────────────────────────────────────────────────────────────────────
# C6: fits/ladder enumeration — 4-axis cells coexist; gate-5 peer is phase-keyed


def test_fits_resolve_cells_keeps_distinct_condition_and_form_cells(tmp_path):
    acts = tmp_path / "activations"
    v = acts / "char_helios"
    v.mkdir(parents=True)
    for condition, form in (("inserted", "chat"), ("on_policy", "chat"), ("inserted", "bare_text")):
        (v / f"{forms.cell_key('char_helios', condition, form, 'qwen2.5-7b')}.npz").write_bytes(
            b"x"
        )
    cells = fits._resolve_cells(
        acts, ["char_helios"], list(forms.CONDITIONS), list(forms.FORMS), ["qwen2.5-7b"]
    )
    keys = {forms.cell_key(v_, c, f, m) for v_, c, f, m, _p in cells}
    assert keys == {
        "char_helios__inserted__chat__qwen2.5-7b",
        "char_helios__on_policy__chat__qwen2.5-7b",
        "char_helios__inserted__bare_text__qwen2.5-7b",
    }


def test_resolve_cells_flat_smoke_fallback_attaches_once(tmp_path):
    acts = tmp_path / "activations"
    acts.mkdir()
    (acts / "fixture.npz").write_bytes(b"x")  # flat smoke fixture, no variant subtree
    for mod in (fits, ladder):
        cells = mod._resolve_cells(
            acts, ["char_helios"], list(forms.CONDITIONS), list(forms.FORMS), ["qwen2.5-7b"]
        )
        assert len(cells) == 1, mod.__name__  # never one cell per (condition, form) combo


def test_gate5_peer_condition_pairs_b_with_d_and_skips_cell_c():
    assert fits._GATE5_PEER_CONDITION["inserted"] == "on_policy"
    assert fits._GATE5_PEER_CONDITION["on_policy"] == "inserted"
    assert "cell_c" not in fits._GATE5_PEER_CONDITION


def test_fits_default_variants_include_cell_c_op_panel():
    for op_variant in ("char_helios_op", "char_wren_op", "char_dana_op", "char_vex_op"):
        assert op_variant in fits.DEFAULT_VARIANTS
        assert op_variant in ladder.DEFAULT_VARIANTS
