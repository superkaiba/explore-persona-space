"""Issue #2203 regression tests (code-review + full-rerun-bugfix round).

Offline / CPU / no-download. Pins the fixes the smoke could not exercise
because a ``_smoke_axis`` substitution bypassed the production ``_load_axis``:

- **Fix B / schema-v2** — ``phase2._load_axis`` reads the POSITION-MATCHED
  unit-space τ (``tau_by_position``: 4 position sets) + the two
  footprint-matched random pools (``tau_rand_by_position``: context-end +
  all-tokens) the phase1 writer emits, and FAILS LOUD on the legacy
  raw-space schema (``tau_by_layer``) instead of silently reusing a τ
  calibrated on the WRONG space. Round-trips a phase0-schema ``.pt`` +
  schema-v2 band JSON; the legacy schema raises.
- ``pareto_select`` frontier + knee tie-break.
- ``gsm8k_extract`` / ``wilson_ci`` (capability scoring primitives).
- cluster-id carries no ``% 44`` aliasing (r1 Minor 18).
- ``regime_fingerprint`` / ``check_regime`` cross-regime refusal (r1 M7).
- ``phase2._assert_alignment`` per-row meta trip (r1 M10).
- the 24-arm registry (kind ``null`` unified) flows into ``_arm_names``.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from scripts import issue2203_capability as CAP  # noqa: E402
from scripts import issue2203_common as C  # noqa: E402
from scripts import issue2203_phase1 as P1  # noqa: E402
from scripts import issue2203_phase2 as P2  # noqa: E402


def _write_phase0_axis(tmp: Path, layers: list[int], hidden: int = 8) -> Path:
    """A phase0-schema axis .pt (str-keyed axis_by_layer/h_def_by_layer + layers)."""
    torch.manual_seed(0)
    blob = {
        "axis_by_layer": {str(li): torch.randn(hidden) for li in layers},
        "h_def_by_layer": {str(li): torch.randn(hidden) for li in layers},
        "layers": layers,
    }
    p = tmp / "phase0_axis_smoke.pt"
    torch.save(blob, p)
    return p


def _write_phase1_band(tmp: Path, band: list[int], layers: list[int], *, legacy: bool) -> Path:
    """A phase1-schema band JSON. ``legacy=True`` emits the raw-space (pre-Fix-B) schema."""
    p = tmp / "phase1_band_tau_smoke.json"
    if legacy:
        # The raw-space schema _load_axis must now REFUSE (Fix B: τ was calibrated
        # on response tokens in RAW space; the corrected op compares in UNIT space).
        result = {
            "band_layers": band,
            "tau_by_layer": {str(li): -1.0 * (li + 1) for li in layers},
            "tau_rand_by_layer": {str(li): -0.5 for li in layers},
        }
        p.write_text(json.dumps(result))
        return p
    # schema-v2 (Fix B): position-matched UNIT-space τ (4 sets) + the two
    # footprint-matched random pools (context-end + all-tokens). Distinct values
    # per position set so a mis-selection is detectable. τ is written for every
    # band+L14 layer (the single-layer arm's L14 is folded in by _load_axis).
    tau_by_position = {
        "prefix-end": {str(li): -1.0 for li in layers},
        "context-end": {str(li): -2.0 for li in layers},
        "all-prompt": {str(li): -3.0 for li in layers},
        "all-tokens": {str(li): -4.0 for li in layers},
    }
    tau_rand_by_position = {
        "context-end": {str(li): -0.3 for li in layers},
        "all-tokens": {str(li): -0.7 for li in layers},
    }
    result = {
        "band_layers": band,
        "single_layer_L14": C.L14,
        "tau_schema": 2,
        "tau_by_position": tau_by_position,
        "tau_rand_by_position": tau_rand_by_position,
    }
    p.write_text(json.dumps(result))
    return p


def test_load_axis_reads_position_matched_tau(tmp_path):
    """Fix B: schema-v2 position-matched τ + both random pools round-trip, distinct."""
    band, layers = [1, 2], [1, 2, C.L14]
    axis_p = _write_phase0_axis(tmp_path, layers)
    band_p = _write_phase1_band(tmp_path, band, layers, legacy=False)
    geom = P2._load_axis(axis_p, band_p)
    assert geom["layers"] == band and geom["axis_source"] == "response"
    # Four position sets, each keyed by band+L14, DISTINCT per position (not a global τ).
    for ps in ("prefix-end", "context-end", "all-prompt", "all-tokens"):
        assert set(geom["tau_by_position"][ps]) == {1, 2, C.L14}
    assert geom["tau_by_position"]["context-end"][1] != geom["tau_by_position"]["all-tokens"][1]
    # Both footprint-matched null pools present, keyed by band+L14, and distinct.
    assert set(geom["tau_rand_by_position"]) == {"context-end", "all-tokens"}
    assert set(geom["tau_rand_by_position"]["context-end"]) == {1, 2, C.L14}
    assert (
        geom["tau_rand_by_position"]["context-end"][1]
        != geom["tau_rand_by_position"]["all-tokens"][1]
    )
    # L14 (single-layer arm) present in the axis AND every position-matched τ pool.
    assert C.L14 in geom["axis_by_layer"] and C.L14 in geom["tau_by_position"]["context-end"]


def test_load_axis_fails_loud_on_legacy_rawspace_schema(tmp_path):
    """Fix B pre-fix demonstration: the raw-space band JSON raises, never silently reused."""
    band, layers = [1, 2], [1, 2, C.L14]
    axis_p = _write_phase0_axis(tmp_path, layers)
    band_p = _write_phase1_band(tmp_path, band, layers, legacy=True)
    with pytest.raises(KeyError, match=r"tau_by_position|tau_rand_by_position|schema-v2"):
        P2._load_axis(axis_p, band_p)


def test_null_cap_arms_are_position_matched_null_kind():
    """The two footprint-matched cap-null arms are kind 'null' at their matched position (§5).

    Kind unified to 'null' (Fix B): ``build_stack_for_arm`` selects the τ_rand pool
    by the arm's ``position_set`` (context-end vs all-tokens), not a kind suffix.
    """
    assert C.ARM_SPECS["cap_ctx_randnull"] == {
        "op": "cap",
        "position_set": "context-end",
        "kind": "null",
    }
    assert C.ARM_SPECS["cap_alltoken_randnull"] == {
        "op": "cap",
        "position_set": "all-tokens",
        "kind": "null",
    }


def test_arm_registry_24_arms_flows_to_arm_names():
    """The 24-arm registry (kind 'null' unified; 6 Part-D native arms) flows into _arm_names.

    Registration in ``ARM_SPECS`` drives all three phases (default = the registry)
    and survives the ``--arms`` filter, which silently DROPS unknown slugs (the
    smoke's per-arm-file existence asserts are the fail-loud guard for a typo).
    """
    assert C.ARM_SPECS["axrep_allprompt_randnull"] == {
        "op": "axis_replace",
        "position_set": "all-prompt",
        "kind": "null",
    }
    assert C.ARM_SPECS["axrep_alltoken_randnull"] == {
        "op": "axis_replace",
        "position_set": "all-tokens",
        "kind": "null",
    }
    # The 6 Part-D native arms carry an axis_source; the 18 response arms do not.
    native = [
        a
        for a, s in C.ARM_SPECS.items()
        if s.get("axis_source") in ("context_native", "prefix_native")
    ]
    assert len(native) == 6
    # Default (no --arms): the FULL registry, 24 arms, in registry order.
    names = P2._arm_names(P2.build_parser().parse_args([]))
    assert names == list(C.ARM_SPECS.keys()) and len(names) == 24
    # The --arms filter passes known slugs through and DROPS unknowns.
    new = ["axrep_allprompt_randnull", "axrep_alltoken_randnull"]
    filt = P2.build_parser().parse_args(["--arms", *new, "axrep_typo_randnull"])
    assert P2._arm_names(filt) == new


def test_pareto_select_frontier_and_knee():
    metrics = {
        # id: harm_reduction, capability_drop, width, center
        "a": {"harm_reduction": 0.5, "capability_drop": 0.10, "width": 4, "center": 12},
        "b": {"harm_reduction": 0.4, "capability_drop": 0.05, "width": 2, "center": 12},
        "c": {
            "harm_reduction": 0.3,
            "capability_drop": 0.20,
            "width": 8,
            "center": 14,
        },  # dominated
    }
    selected, frontier = P1.pareto_select(metrics)
    assert "c" not in frontier  # dominated by a (more harm_red, less drop)
    assert set(frontier) == {"a", "b"}
    # knee = argmax(harm_reduction - capability_drop): a=0.40, b=0.35 -> a
    assert selected == "a"


def test_pareto_select_tie_break_smaller_width():
    metrics = {
        "wide": {"harm_reduction": 0.5, "capability_drop": 0.1, "width": 8, "center": 12},
        "narrow": {"harm_reduction": 0.5, "capability_drop": 0.1, "width": 2, "center": 12},
    }
    selected, _ = P1.pareto_select(metrics)
    assert selected == "narrow"  # equal knee -> smaller width wins


def test_gsm8k_extract():
    assert CAP.gsm8k_extract("...\n#### 42") == "42"
    assert CAP.gsm8k_extract("#### 1,024") == "1024"
    assert CAP.gsm8k_extract("#### $18.50") == "18.50"
    assert CAP.gsm8k_extract("the answer is 7") == "7"  # last-number fallback
    assert CAP.gsm8k_extract("no digits here") is None


def test_wilson_ci_bounds():
    assert CAP.wilson_ci(0, 0) is None
    lo, hi = CAP.wilson_ci(5, 10)
    assert 0.0 <= lo < 0.5 < hi <= 1.0
    lo0, hi0 = CAP.wilson_ci(0, 10)
    assert lo0 == 0.0 and 0.0 < hi0 < 0.5


def test_cluster_id_no_mod44_aliasing():
    """r1 Minor 18: cluster_id = (bank, item index, role), never `hi % 44`."""
    rows = C.build_jailbreak_set(6, smoke=True)
    for r in rows:
        cid = r["meta"]["cluster_id"]
        assert cid == f"{r['meta']['harm_bank']}:{r['meta']['harm_index']}:{r['meta']['role']}"
        assert "%" not in cid


def test_regime_mismatch_refuses(tmp_path):
    """r1 M7: a resume artifact from a DIFFERENT regime raises naming the diff."""
    cur = C.regime_fingerprint(model="m", n_jailbreak=500, smoke=False)
    same = C.regime_fingerprint(model="m", n_jailbreak=500, smoke=False)
    C.check_regime(same, cur, tmp_path / "x.json")  # identical -> no raise
    diff = C.regime_fingerprint(model="m", n_jailbreak=250, smoke=False)
    with pytest.raises(ValueError, match=r"REGIME MISMATCH|n_jailbreak"):
        C.check_regime(diff, cur, tmp_path / "x.json")
    with pytest.raises(ValueError, match="NO regime fingerprint"):
        C.check_regime(None, cur, tmp_path / "x.json")


def test_alignment_assert_trips_on_meta_mismatch():
    """r1 M10: a persisted-vs-rebuilt jb meta mismatch raises (wrong judged question)."""
    jb = C.build_jailbreak_set(3, smoke=True)
    good_meta = [r["meta"] for r in jb]
    P2._assert_alignment("baseline", "jailbreak", good_meta, jb)  # aligned -> no raise
    bad = [dict(m) for m in good_meta]
    bad[1]["harm_index"] = 999999
    with pytest.raises(ValueError, match="meta mismatch on 'harm_index'"):
        P2._assert_alignment("baseline", "jailbreak", bad, jb)
    with pytest.raises(ValueError, match="!="):  # length mismatch
        P2._assert_alignment("baseline", "jailbreak", good_meta[:2], jb)


# ------------------------------------------------------------------
# r2: round-label output/judge-staging plumbing (code-review r1 C1) +
# regen-telemetry dedup + verdicts-synth flag (r1 minors).
# ------------------------------------------------------------------


def test_eval_results_dir_carries_round_label(monkeypatch):
    """r1 C1: the DEFAULT out-root is the labeled subdir (no launch passes --out-dir).

    ``Path.mkdir`` is monkeypatched to a no-op so the unit test asserts the
    returned parts WITHOUT side-effecting the repo tree (code-review r2 minor).
    """
    monkeypatch.setattr(Path, "mkdir", lambda self, *a, **k: None)
    d = C.eval_results_dir()
    assert d.parts[-2:] == (f"issue_{C.ISSUE}", C.ROUND_LABEL), d


def test_raw_arm_path_carries_round_label(tmp_path):
    """r1 C1: local raw rel paths lead with the label so HF uploads land labeled."""
    p = P2._raw_arm_path(tmp_path, "cap_ctx")
    assert (
        p.relative_to(tmp_path).as_posix() == f"{C.ROUND_LABEL}/phase2/cap_ctx/raw_completions.json"
    )
    cap = P2._raw_arm_path(tmp_path, "baseline", stage="phase2_capability")
    assert (
        cap.relative_to(tmp_path).as_posix()
        == f"{C.ROUND_LABEL}/phase2_capability/baseline/raw_completions.json"
    )


def test_upload_raw_tree_refuses_unlabeled_rel(tmp_path, monkeypatch):
    """r1 C1 (fails pre-fix): an unlabeled rel path (the parent's layout) raises
    BEFORE any upload; a labeled tree reaches the uploader; require_label=False
    (the pinned unlabeled extraction producer) skips the guard."""
    bad = tmp_path / "phase2" / "cap_ctx" / "raw_completions.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{}")
    with pytest.raises(ValueError, match=C.ROUND_LABEL):
        C.upload_raw_tree(tmp_path)

    good_root = tmp_path / "ok"
    good = good_root / C.ROUND_LABEL / "phase2" / "cap_ctx" / "raw_completions.json"
    good.parent.mkdir(parents=True)
    good.write_text("{}")
    calls: dict = {}
    from explore_persona_space.orchestrate import hub

    def fake_upload(experiment_name, eval_results_dir, delete_after=False):
        calls["experiment_name"] = experiment_name
        calls["root"] = Path(eval_results_dir)
        return {"x": "url"}

    monkeypatch.setattr(hub, "upload_raw_completions_to_data_repo", fake_upload)
    out = C.upload_raw_tree(good_root)
    assert calls["experiment_name"] == C.HF_PREFIX and calls["root"] == good_root
    assert out == {"x": "url"}
    C.upload_raw_tree(tmp_path, require_label=False)  # guard skipped, no raise


def test_upload_raw_tree_exempts_staged_extraction_rel(tmp_path, monkeypatch):
    """code-review r2 Major `labeled-upload-refuses-staged-reuse-input` (fails
    pre-fix): the §4.5-pinned staged reuse input (`extraction/…`, written by
    `_load_phase0_pool` INSIDE the guarded tree) must not trip the label guard
    — `phase1 --band … --upload` was deterministically ValueError-ing at the
    upload step (phase1.py:408). A mixed tree (staged extraction rel + labeled
    round rel) proceeds; a genuinely unlabeled ROUND rel still refuses."""
    ext = tmp_path / "extraction" / "raw_completions.json"
    ext.parent.mkdir(parents=True)
    ext.write_text("{}")
    lab = tmp_path / C.ROUND_LABEL / "phase1_band_sweep" / "baseline" / "raw_completions.json"
    lab.parent.mkdir(parents=True)
    lab.write_text("{}")
    calls: dict = {}
    from explore_persona_space.orchestrate import hub

    def fake_upload(experiment_name, eval_results_dir, delete_after=False):
        calls["root"] = Path(eval_results_dir)
        return {"ok": "url"}

    monkeypatch.setattr(hub, "upload_raw_completions_to_data_repo", fake_upload)
    out = C.upload_raw_tree(tmp_path)  # require_label=True default -> proceeds
    assert calls["root"] == tmp_path and out == {"ok": "url"}
    # guard intact for a genuinely mislabeled ROUND rel in the same tree
    bad = tmp_path / "phase2" / "cap_ctx" / "raw_completions.json"
    bad.parent.mkdir(parents=True)
    bad.write_text("{}")
    with pytest.raises(ValueError, match=C.ROUND_LABEL):
        C.upload_raw_tree(tmp_path)


def test_regime_carries_round_label_and_judge_assert_refuses_parent(tmp_path):
    """r1 C1 (fails pre-fix): every regime fingerprint carries round_label, and
    the judge-staging assert refuses parent-run records (no/foreign label)."""
    reg = C.regime_fingerprint(model="m", smoke=False)
    assert reg["round_label"] == C.ROUND_LABEL
    C.assert_round_regime({"regime": reg}, tmp_path / "x.json")  # this round: no raise
    with pytest.raises(ValueError, match="round_label"):
        C.assert_round_regime({"arm": "cap_ctx"}, tmp_path / "x.json")  # parent: no regime
    with pytest.raises(ValueError, match="refusing to"):
        C.assert_round_regime({"regime": {"model": "m"}}, tmp_path / "x.json")  # unlabeled


def test_summarize_realized_excludes_regen_pass():
    """r1 minor (fails pre-fix): regen-pass records are excluded from the means
    (no double-count of regenerated rows) and reported as a separate count."""
    base = [
        {"fired_frac": 0.2, "n_positions": 10, "abs_dproj_mean": 0.5},
        {"fired_frac": 0.4, "n_positions": 10, "abs_dproj_mean": 1.5},
    ]
    regen = [{"fired_frac": 1.0, "n_positions": 99, "abs_dproj_mean": 9.0, "regen_pass": True}]
    out = P2._summarize_realized(base + regen)
    assert out["n_edit_forwards"] == 2
    assert out["total_positions_edited"] == 20
    assert out["mean_fired_frac"] == pytest.approx(0.3)
    assert out["mean_abs_dproj"] == pytest.approx(1.0)
    assert out["n_regen_edit_forwards"] == 1
    assert "n_regen_edit_forwards" not in P2._summarize_realized(base)


class _CountTok:
    """Whitespace token counter mirroring the tokenizer ``__call__`` contract."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [0] * len(text.split())}


def test_cap_hit_regen_tags_regen_records():
    """r1 minor (fails pre-fix): the regen pass's realized records carry
    ``regen_pass: True`` so the summary can dedup them."""
    from scripts import issue2203_runtime as R

    contexts = [{"system": "", "user": f"q{i}"} for i in range(4)]
    calls = {"n": 0}

    def gen_fn(ctxs, mnt):
        calls["n"] += 1
        if calls["n"] == 1:
            # every row re-tokenizes to 3 >= cap 2 -> regen fires
            rec = {"fired_frac": 0.1, "n_positions": 1, "abs_dproj_mean": 0.2}
            return ["a b c" for _ in ctxs], [rec]
        rec = {"fired_frac": 0.9, "n_positions": 2, "abs_dproj_mean": 0.7}
        return ["ok" for _ in ctxs], [rec]

    texts, realized, info = R.cap_hit_regen(_CountTok(), contexts, gen_fn, max_new_tokens=2)
    assert info["regenerated"] is True and info["n_regenerated"] == 4
    assert [bool(r.get("regen_pass")) for r in realized] == [False, True]
    assert texts == ["ok"] * 4


def test_load_verdicts_synth_flag_from_branch_taken(tmp_path):
    """r1 minor (fails pre-fix): ``synthesized`` reflects the branch ACTUALLY
    taken — a pre-existing verdicts file under smoke reads False."""
    from scripts import issue2203_phase0_native as N

    rows = [
        {"kind": "role", "ctx": {"user": "u"}, "text": "t", "role": "pirate"},
        {"kind": "default", "ctx": {"user": "u2"}, "text": "t2", "role": None},
    ]
    p1, synth1 = N._load_verdicts(tmp_path, rows, smoke=True)
    assert synth1 is True and p1.exists()
    p2, synth2 = N._load_verdicts(tmp_path, rows, smoke=True)
    assert p2 == p1 and synth2 is False
