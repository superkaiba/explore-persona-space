"""#1774 round-2 revision pins (code-review v1 findings C1/C3/M1/M3 + Minors).

Covers: the q4 unsharded guard (C1 — a sharded --arms q4 races on the combined
cokernel writes), the dispatcher's result-push verification block (C3, #1205 +
#1325), fig_causal_shift against the REAL merge_state_shift schema (M3 — the
fixture mirrors the realized P3 smoke artifact's key set, never the builder
code), the pilot reap-before-HF-load ordering (M1, text pin), the stage-audit
consumed-file widening regression (round-C BLOCKER fix, Step 4.5), the judge
judge_skip guard, and the capture resume regime key. All CPU, tmp_path-only
writes.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "scripts"))

DISPATCH = REPO / "scripts" / "issue1774_dispatch.sh"


# ── C1: q4 sharded-write race ────────────────────────────────────────────────


def test_step_q4_refuses_partial_arms() -> None:
    """A sharded --arms invocation of q4 must FAIL LOUD, never silently drop
    arms from the combined cokernel_*.json writes (guard fires before any IO)."""
    import issue1774_fit_battery as fb

    with pytest.raises(RuntimeError, match="UNSHARDED"):
        fb.step_q4(None, ["arm_context", "arm_bare_query"], "cpu", None)
    with pytest.raises(RuntimeError, match="cokernel"):
        fb.step_q4(None, ["arm_prefix_end"], "cpu", None)


def test_dispatcher_never_shards_q4() -> None:
    text = DISPATCH.read_text()
    # the sharded step loop no longer contains q4 ...
    assert re.search(r"for step in fits q3;", text), "sharded loop should be fits+q3 only"
    # ... and the unsharded q4 call passes NO --arms (defaults to all arms)
    q4_calls = [ln for ln in text.splitlines() if "--step q4" in ln]
    assert len(q4_calls) == 1, q4_calls
    assert "--arms" not in q4_calls[0]


# ── C3: result-push verification block (#1205 + #1325) ──────────────────────


def test_dispatcher_result_push_block() -> None:
    text = DISPATCH.read_text()
    # push-verify: rev-list count against the remote-tracking ref
    assert "git rev-list --count" in text
    # artifact-presence assert: per-file ls-tree against the pushed tree
    assert "git ls-tree -r" in text
    # the swallow shape is banned (#1205): no `git push ... || true/echo`
    assert not re.search(r"git push[^\n]*\|\|\s*(true|echo)", text)
    # every phase pushes BEFORE its sentinel
    for phase in ("p1", "p2", "p3"):
        assert f"push_results {phase}" in text
    assert text.index("push_results p3") < text.index("phase_sentinel p3")
    # smoke never touches the production branch
    assert "result push SKIPPED (smoke" in text


# ── M1: pilot reap ordering (text pin — GPU-only path) ───────────────────────


def test_stage_pilot_reaps_engine_before_hf_load() -> None:
    src = (REPO / "scripts" / "issue1774_draws.py").read_text()
    body = src[src.index("def stage_pilot") : src.index("def stage_upload")]
    i_reap = body.index("_reap_vllm_engine(llm)")
    i_del = body.index("del llm", i_reap)  # the STATEMENT (a comment mentions it earlier)
    i_drain = body.index("_drain_vllm_release(device)")
    i_load = body.index("_load_hf_model(device)")
    assert i_reap < i_del < i_drain < i_load
    # identity-gate forwards thread the introspection-guarded logits_to_keep
    assert "_logits_to_keep_kwargs(model, return_logits=False)" in body


# ── M3: fig_causal_shift against the REAL state_shift schema ─────────────────


def _real_shape_state_shift() -> dict:
    """Fixture mirroring the realized P3 smoke artifact's key set
    (/tmp/issue1774-smoke-p3/steering/state_shift.json, round-C run)."""
    return {
        "conditions": {
            "add_random0_neg": {
                "kind": "add",
                "direction": "random0",
                "sign": -1,
                "per_context_dt1": {"17747": 3.13, "18259": 6.36},
                "median_dt1": 4.75,
                "p90_dt1": 6.04,
                "n_contexts": 2,
            },
            "add_top_sv0_pos": {
                "kind": "add",
                "direction": "top_sv0",
                "sign": 1,
                "per_context_dt1": {"17747": 2.9, "18259": 5.1},
                "median_dt1": 4.0,
                "p90_dt1": 4.9,
                "n_contexts": 2,
            },
        },
        "steer_base_band": {
            "per_context": {"17747": [3.29], "18259": [5.13]},
            "pooled_p50": 4.21,
            "pooled_p90": 4.95,
            "k_draws": 2,
        },
        "alpha_by_direction": {"random0": 8, "top_sv0": 8},
        "dropped_directions": {},
        "n_usable_directions": 2,
        "judge_skip": False,
        "layer": 14,
    }


def test_fig_causal_shift_writes_on_real_schema(tmp_path: Path) -> None:
    import issue1774_figures as figs

    eval_root = tmp_path / "eval"
    fig_dir = tmp_path / "figs"
    (eval_root / "steering").mkdir(parents=True)
    fig_dir.mkdir()
    (eval_root / "steering" / "state_shift.json").write_text(json.dumps(_real_shape_state_shift()))
    skip = figs.fig_causal_shift(eval_root, fig_dir)
    assert skip is None, f"figure skipped: {skip}"
    assert list(fig_dir.glob("causal_state_shift*")), "no figure file written"


def test_aggregate_state_shift_digest_real_keys(tmp_path: Path) -> None:
    import issue1774_aggregate as agg

    eval_root = tmp_path / "eval"
    (eval_root / "steering").mkdir(parents=True)
    (eval_root / "steering" / "state_shift.json").write_text(json.dumps(_real_shape_state_shift()))
    dig = agg.merge_phase_digests(eval_root, layers=[])
    assert "state_shift" in dig, dig.get("skipped")
    ss = dig["state_shift"]
    assert ss["conditions"]["add_random0_neg"]["median_dt1"] == 4.75
    assert ss["steer_base_band"] == {"pooled_p50": 4.21, "pooled_p90": 4.95, "k_draws": 2}
    assert ss["judge_skip"] is False
    # compact digest: no per-context payloads survive
    assert "per_context_dt1" not in json.dumps(ss["conditions"])


# ── Minor: stage-audit consumed-file widening (round-C BLOCKER fix pin) ──────


def test_stage_audit_consumed_files_include_corpus_stores() -> None:
    import issue1774_stage_audit as sa

    rels = [rel for rel, _p in sa._consumed_files()]
    for store in ("prefix_store.jsonl", "query_store.jsonl"):
        assert any(rel.endswith(f"corpus/{store}") for rel in rels), store


# ── Minor: judge_skip guard ──────────────────────────────────────────────────


def test_judge_refuses_on_judge_skip(tmp_path: Path) -> None:
    import issue1774_judge as jj

    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "judge_skip": True,
                "rows": [
                    {
                        "row_id": "a-1-d0",
                        "condition": "add_random0_neg",
                        "question": "q",
                        "completion": "c",
                    }
                ],
            }
        )
    )
    rc = jj.main(["--manifest", str(manifest), "--out-dir", str(tmp_path / "j"), "--dry-run"])
    assert rc == 8


# ── Minor: capture resume regime key (#722 r3) ───────────────────────────────


def test_capture_regime_guard_refuses_cross_regime_resume(tmp_path: Path) -> None:
    import issue1774_draws as dr

    regime = {"limit": 4, "shard": "0/1", "k_draws": 2, "n_rows": 4}
    dr._capture_regime_guard(tmp_path, "0of1", regime)  # first run writes
    dr._capture_regime_guard(tmp_path, "0of1", dict(regime))  # same regime resumes
    with pytest.raises(RuntimeError, match="regime mismatch"):
        dr._capture_regime_guard(tmp_path, "0of1", {**regime, "limit": 8})
