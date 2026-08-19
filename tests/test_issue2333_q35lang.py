"""Issue #2333 `q35_language_snowball` cell-set tests (plan §4.3 items 5-6).

Covers the q35lang registry counts, empty-S2 tolerance across the reused rig
(donor maps / gate spots / smoke blocks), `expected_grid_slugs("q35lang")`
byte-parity with the runtime block enumeration, the ported `prefix+query`
minimal-pair gate (+ the prefix-side regression), SILENT-filter coverage
(judge calib membership filter + the analysis banked #2329 join), the
control-health / both-cells-floor verdict routing (`control-unresolved` /
`untestable-causal`), the carrier-level cluster bootstrap, and the judge
fresh-cache-partition guard.

Real q35lang pair universe throughout (`bank2162`, no network); the two
`_stats_q35lang` end-to-end tests read the COMMITTED vendored #2329 inputs
under eval_results/issue_2333/q35_language_snowball/inputs (sparse cone
registered in tests/sparse_cones.txt).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO_ROOT / "scripts"))

import issue2162_run as R  # noqa: E402
import issue2333_analysis as A33  # noqa: E402
import issue2333_judge as J33  # noqa: E402
import issue2333_run as RUN  # noqa: E402

from explore_persona_space.experiments.issue2333 import constants as C  # noqa: E402

LANG_CELLS = C.CELL_SETS["q35lang"].s1_cells  # ("instr_language", "language_implied")
FIXTURE_ARMS = ("prefill3_med", "prefill3_bstart")


@pytest.fixture(scope="module")
def q35lang_universe():
    s1, s2 = RUN.build_pair_universe("q35lang")
    return s1, s2


# ── 1. cell-set registry counts ───────────────────────────────────────


def test_cell_set_registry_counts_q35lang():
    cs = C.CELL_SETS["q35lang"]
    assert cs.expected_pairs == 72
    assert cs.expected_contexts == 72
    assert cs.expected_grid_blocks == 48
    assert cs.expected_ce_blocks == 4
    assert cs.expected_smoke_blocks == 24
    assert cs.s2_on is False
    assert cs.hf_namespace == C.Q35LANG_LABEL == "q35_language_snowball"
    assert cs.max_new_tokens == 4096  # #2329 measured cap-hit → pre-applied regen tier
    assert C.active_cells("q35lang") == LANG_CELLS
    assert C.S2_CELL not in C.active_cells("q35lang")


def test_cell_set_registry_default_main_unchanged():
    cs = C.CELL_SETS["main"]
    assert cs.expected_pairs == 195 and cs.expected_contexts == 195
    assert cs.expected_grid_blocks == 144 and cs.expected_ce_blocks == 12
    assert cs.expected_smoke_blocks == 48
    assert cs.s2_on is True and cs.hf_namespace is None
    assert cs.max_new_tokens == C.MAX_NEW_TOKENS == 2048
    assert C.active_cells("main") == (*C.S1_CELLS, C.S2_CELL)


def test_pair_universe_q35lang_counts_no_silent_drop(q35lang_universe):
    """The cell membership filter keeps EXACTLY 36 pairs per language cell —
    and the S2 15-pair assert never trips on the empty S2 list."""
    s1, s2 = q35lang_universe
    assert len(s1) == 72 and s2 == []
    per_cell: dict[str, int] = {}
    for p in s1:
        per_cell[p.cell] = per_cell.get(p.cell, 0) + 1
    assert per_cell == {cell: 36 for cell in LANG_CELLS}
    # judge-side twin filter yields the identical universe (fact-check A1).
    j1, j2 = J33.build_pair_universe("q35lang")
    assert [p.pair_id for p in j1] == [p.pair_id for p in s1] and j2 == []
    contexts = RUN.build_context_universe(s1, s2, "q35lang")
    assert len(contexts) == 72
    assert {c["__set"] for c in contexts.values()} == {"s1"}


# ── 2. empty-S2 tolerance (donor maps, gate spots, smoke blocks) ──────


def test_empty_s2_donor_maps(q35lang_universe):
    s1, s2 = q35lang_universe
    maps = RUN.build_donor_maps(s1, s2)
    ids = {p.pair_id for p in s1}
    assert set(maps["shuffled"]) == ids  # covers exactly the 72 survivors
    assert set(maps["shuffled"].values()) <= ids  # donors never leave the set
    cell_of = {p.pair_id: p.cell for p in s1}
    for pid, donor in maps["shuffled"].items():
        assert cell_of[pid] == cell_of[donor], (pid, donor)  # value-constrained same-cell


def test_empty_s2_gate_spots(q35lang_universe):
    s1, s2 = q35lang_universe
    spots = RUN._gate_spots(s1, s2, "q35lang")
    assert len(spots) == 4  # 2 per language cell, no S2 spots
    assert {s["cell"] for s in spots} == set(LANG_CELLS)
    assert all(s["cell"] != C.S2_CELL for s in spots)


def test_empty_s2_smoke_and_grid_blocks(q35lang_universe):
    s1, s2 = q35lang_universe
    smoke = RUN.smoke_blocks_2333(s1, s2, set(), "q35lang")
    assert len(smoke) == 24  # 1 language pair x 12 arms x 2 variants
    assert {b.cell for b in smoke} <= set(LANG_CELLS)
    grid = RUN.enumerate_blocks_2333(s1, s2, set(), "q35lang")
    assert len(grid) == 48
    assert {b.cell for b in grid} == set(LANG_CELLS)


# ── 3. expected_grid_slugs("q35lang") byte-parity with block_slug ─────


def test_expected_grid_slugs_q35lang_byte_parity(q35lang_universe):
    s1, s2 = q35lang_universe
    blocks = RUN.enumerate_blocks_2333(s1, s2, set(), "q35lang")
    assert {b.slug for b in blocks} == C.expected_grid_slugs("q35lang")
    runtime = {
        R.block_slug(f"{cell}|{arm}|{variant}")
        for cell in C.active_cells("q35lang")
        for arm in C.ARM_SLUGS
        for variant in C.VARIANTS
    }
    assert runtime == C.expected_grid_slugs("q35lang")
    assert len(runtime) == 48
    ce_runtime = {
        R.block_slug(f"{cell}|ce_replace|{variant}")
        for cell in C.active_cells("q35lang")
        for variant in C.VARIANTS
    }
    assert ce_runtime == C.expected_ce_control_slugs("q35lang")
    assert len(ce_runtime) == 4


# ── 4. minimal-pair gate: prefix+query port + prefix-side regression ──

_IM = 9  # synthetic <|im_start|> token id
# 3-occurrence render shape: [IM, <prefix...>, IM, <final user turn...>, IM, <gen header>]
_PREFIX_A = [_IM, 1, 2, _IM, 7, _IM, 8]
_PREFIX_B_DIFF = [_IM, 3, 4, _IM, 7, _IM, 8]  # prefix differs, final turn identical
_BOTH_DIFF_B = [_IM, 3, 4, _IM, 6, _IM, 8]  # prefix AND final turn differ
_FINAL_DIFF_B = [_IM, 1, 2, _IM, 6, _IM, 8]  # prefix identical, final turn differs


def test_minimal_pair_prefix_query_both_differ_passes():
    assert RUN.minimal_pair_check(_PREFIX_A, _BOTH_DIFF_B, _IM, locus="prefix+query") == ()


def test_minimal_pair_prefix_query_rejects_identical_sides():
    # prefix-identical rows FAIL (bank defect)
    assert RUN.minimal_pair_check(_PREFIX_A, _FINAL_DIFF_B, _IM, locus="prefix+query") == (
        "varied-prefix-identical",
    )
    # final-turn-identical rows FAIL the inverse predicate (varied-query-identical)
    assert RUN.minimal_pair_check(_PREFIX_A, _PREFIX_B_DIFF, _IM, locus="prefix+query") == (
        "varied-query-identical",
    )


def test_minimal_pair_prefix_side_semantics_unchanged():
    """Main-cell regression: prefix-side still requires final-turn identity."""
    assert RUN.minimal_pair_check(_PREFIX_A, _PREFIX_B_DIFF, _IM, locus="prefix-side") == ()
    assert RUN.minimal_pair_check(_PREFIX_A, _BOTH_DIFF_B, _IM, locus="prefix-side") == (
        "final-turn-tokens-differ",
    )
    reasons = RUN.minimal_pair_check(_PREFIX_A, list(_PREFIX_A), _IM, locus="prefix-side")
    assert reasons == ("varied-prefix-identical",)


def test_minimal_pair_unknown_locus_fails_loud():
    with pytest.raises(RuntimeError, match="unsupported span locus"):
        RUN.minimal_pair_check(_PREFIX_A, _BOTH_DIFF_B, _IM, locus="final-query")


def test_pair_span_locus_real_universe(q35lang_universe):
    s1, _ = q35lang_universe
    loci = {p.cell: RUN.pair_span_locus(p) for p in s1}
    assert loci == {"instr_language": "prefix-side", "language_implied": "prefix+query"}
    assert all(
        RUN.pair_span_locus(p)
        == ("prefix+query" if p.cell == "language_implied" else "prefix-side")
        for p in s1
    )


# ── 5. SILENT-filter coverage (judge calib membership + banked joins) ─


def test_stage_calib_gated_for_q35lang():
    """The banked Qwen2.5 calib legs are main-only: the q35lang wave GATES the
    phase (returns OK before any staging/network work)."""
    assert J33.phase_stage_calib(SimpleNamespace(cell_set="q35lang")) == J33.RC_OK


def test_calib_membership_filter_fails_loud_on_q35lang_rows(tmp_path):
    """A q35lang row set pushed through the S1 calib membership filter drops
    to EMPTY and raises — never a silent empty pass-through (fact-check A1)."""
    d = tmp_path / "s1"
    d.mkdir()
    rows = [
        {"slot": "ce", "cell": cell, "arm": "steered", "pair_id": f"{cell}::x", "text": "t"}
        for cell in LANG_CELLS
    ]
    (d / "shard_lang__ce__steered.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8"
    )
    with pytest.raises(AssertionError, match="filtered to empty"):
        J33.load_calib_s1(tmp_path)


def test_behavior_items_preserve_q35lang_rows(q35lang_universe):
    """Unit builders emit 2 sides per row + 2 continuation sides per prefill
    row — a q35lang row set is never silently dropped by the join."""
    s1, s2 = q35lang_universe
    rows = []
    for p in s1[:2]:
        for variant in C.VARIANTS:
            rows.append(
                {
                    "block_key": f"{p.cell}|prefill3_med|{variant}",
                    "pair_id": p.pair_id,
                    "draw": 0,
                    "cell": p.cell,
                    "arm_slug": "prefill3_med",
                    "variant": variant,
                    "kind": "prefill",
                    "response_text": "resp",
                    "continuation_text": "cont",
                }
            )
    by_rid = J33.build_behavior_items((s1, s2), grid_rows=rows)
    units = [u for us in by_rid.values() for u in us]
    assert len(units) == 4 * len(rows)  # 2 sides whole + 2 sides continuation-only
    assert len({u.item_id for u in units}) == len(units)
    coh = J33.build_coherence_items(grid_rows=rows)
    assert len(coh) == len(rows)


def test_analysis_banked_join_covers_q35lang_pairs(q35lang_universe, monkeypatch):
    """The vendored #2329 ce join must cover the q35lang universe (72 steered /
    71 null at ce — the plan-verified realized coverage), with no foreign ids."""
    monkeypatch.chdir(REPO_ROOT)
    s1, _ = q35lang_universe
    ids = {p.pair_id for p in s1}
    bce_st = {
        r["pair_id"]
        for r in A33.A62._iter_jsonl(A33.I2329_F_CELLS)
        if r["slot"] == "ce" and r["f_beh"] is not None
    }
    bce_nu = {
        r["pair_id"]
        for r in A33.A62._iter_jsonl(A33.I2329_NULL_CELLS)
        if r["slot"] == "ce" and r["f_beh"] is not None
    }
    assert bce_st == ids  # 72/72 steered ce rows join
    assert bce_nu <= ids and len(bce_nu) == 71  # 71/72 null ce rows (realized coverage)
    anchor_ids = {r["pair_id"] for r in A33.A62._iter_jsonl(A33.I2329_ANCHORS)}
    assert anchor_ids == ids  # per-pair banked anchor deltas cover the universe


# ── 6/7. _stats_q35lang verdict routing fixtures ──────────────────────


def _write_stats_fixture(
    out: Path,
    s1_pairs: list,
    *,
    ce_equal: bool = False,
    below_floor_cell: str | None = None,
    n_surviving: int = 11,
) -> None:
    """f_cells/null_cells/ce_cells fixture over the REAL q35lang universe.

    ``ce_equal`` pins ce_steered == ce_null per pair (dce == 0 exactly).
    ``below_floor_cell`` keeps only ``n_surviving`` of that cell's pairs at
    separation >= 0.5 (the 12/36 per-cell survival floor).
    """
    by_cell: dict[str, list] = {}
    for p in s1_pairs:
        by_cell.setdefault(p.cell, []).append(p)
    sep: dict[str, float] = {}
    for cell, ps in by_cell.items():
        for i, p in enumerate(sorted(ps, key=lambda p: p.pair_id)):
            below = cell == below_floor_cell and i >= n_surviving
            sep[p.pair_id] = 0.2 if below else 0.9
    rng = random.Random(23330)
    steered, nulls, ce = [], [], []
    for p in s1_pairs:
        base = 0.75 + rng.uniform(-0.05, 0.05)
        for slug in FIXTURE_ARMS:
            steered.append(
                {
                    "pair_id": p.pair_id,
                    "arm_slug": slug,
                    "cell": p.cell,
                    "variant": "steered",
                    "separation": sep[p.pair_id],
                    "f_beh": base,
                    "f_beh_continuation": base - 0.02,
                }
            )
            nulls.append(
                {
                    "pair_id": p.pair_id,
                    "arm_slug": slug,
                    "cell": p.cell,
                    "variant": "null",
                    "separation": sep[p.pair_id],
                    "f_beh": 0.1 + rng.uniform(-0.02, 0.02),
                    "f_beh_continuation": 0.1 + rng.uniform(-0.02, 0.02),
                }
            )
        ce_s = 0.4 + rng.uniform(-0.05, 0.05)
        ce_n = ce_s if ce_equal else 0.08 + rng.uniform(-0.02, 0.02)
        ce.append({"pair_id": p.pair_id, "variant": "steered", "f_beh": ce_s})
        ce.append({"pair_id": p.pair_id, "variant": "null", "f_beh": ce_n})
    A33.A62._write_jsonl_atomic(out / "f_cells.jsonl", steered)
    A33.A62._write_jsonl_atomic(out / "null_cells.jsonl", nulls)
    A33.A62._write_jsonl_atomic(out / "ce_cells.jsonl", ce)


def _run_stats_q35lang(out: Path) -> dict:
    args = argparse.Namespace(model_tag="q35", out_dir=out, cell_set="q35lang")
    assert A33.phase_stats(args) == 0
    return json.loads((out / "stats.json").read_text(encoding="utf-8"))


def test_stats_q35lang_ce_equal_emits_no_recovery_label(tmp_path, monkeypatch, q35lang_universe):
    """ce_steered == ce_null ⇒ the same-wave control fails its separation
    conjunction ⇒ `control-unresolved` routing: NO lattice/recovery labels in
    the verdict bundles, and no ratio_net on the zero denominator."""
    monkeypatch.chdir(REPO_ROOT)
    s1, _ = q35lang_universe
    out = tmp_path / "stats_ce_equal"
    out.mkdir()
    _write_stats_fixture(out, s1, ce_equal=True)
    res = _run_stats_q35lang(out)

    pre = res["preconditions"]
    assert pre["precondition_label"] == "control-unresolved"
    assert pre["floor"]["both_cells_pass"] is True
    assert pre["control_health_samewave"]["passed"] is False
    verdicts = res["per_set"]["s1"]["prefill3_verdicts"]
    for scheme, prefix in (("med", ""), ("bstart", "natural-opening-")):
        bundle = verdicts[scheme]
        assert set(bundle) == {"label", "reason", "confirmatory"}  # NO lattice reads emitted
        assert bundle["label"] == f"{prefix}control-unresolved"
        assert "ce control" in bundle["reason"]
    arm = res["per_set"]["s1"]["arms"]["prefill3_med"]
    assert arm["n_pairs"] == 72 and not arm["below_floor"]
    net = arm["recovery_net_samewave"]
    assert net["dce_mean"] == pytest.approx(0.0, abs=1e-12)
    assert "ratio_net" not in net  # zero denominator ⇒ NO recovery ratio emitted


def test_stats_q35lang_one_cell_below_floor_no_pooled_label(
    tmp_path, monkeypatch, q35lang_universe
):
    """One cell at n=11 survivors (< the 12/36 floor) ⇒ `untestable-causal`:
    NO pooled lattice, arms untestable, the surviving cell kept descriptive."""
    monkeypatch.chdir(REPO_ROOT)
    s1, _ = q35lang_universe
    out = tmp_path / "stats_floor"
    out.mkdir()
    _write_stats_fixture(out, s1, below_floor_cell="language_implied", n_surviving=11)
    res = _run_stats_q35lang(out)

    pre = res["preconditions"]
    assert pre["precondition_label"] == "untestable-causal"
    floor = pre["floor"]
    assert floor["both_cells_pass"] is False
    assert floor["per_cell_n_survivors"] == {"instr_language": 36, "language_implied": 11}
    assert floor["cells_below_floor"] == ["language_implied"]
    s1_out = res["per_set"]["s1"]
    assert s1_out["untestable"] is True and s1_out["n_survivors_tested"] == 0
    verdicts = s1_out["prefill3_verdicts"]
    for scheme, prefix in (("med", ""), ("bstart", "natural-opening-")):
        bundle = verdicts[scheme]
        assert set(bundle) == {"label", "reason", "confirmatory"}  # NO pooled lattice
        assert bundle["label"] == f"{prefix}untestable-causal"
        assert "12/36 floor" in bundle["reason"]
    arm = s1_out["arms"]["prefill3_med"]
    assert arm["below_floor"] is True and arm["label"] == "untestable-causal"
    assert arm["n_pairs"] == 0 and "diff_ci" not in arm
    # Surviving cell stays DESCRIPTIVE (per-cell raw-p reads, no family).
    pc = res["per_cell_descriptive"]
    assert pc["instr_language"]["n_survivors"] == 36
    assert pc["instr_language"]["arms"]["prefill3_med"]["n_pairs"] == 36
    assert pc["language_implied"]["n_survivors"] == 11


# ── 8. carrier-level cluster bootstrap ────────────────────────────────


def test_bootstrap_carrier_means_batched_draws_whole_carriers():
    """Every draw carries ALL rows of each drawn carrier: with carriers
    A={0,0,0} and B={1,1}, draw means can ONLY be {0, 0.4, 1} — a row-level
    (pair) bootstrap would produce other values."""
    values = np.array([[0.0], [0.0], [0.0], [1.0], [1.0]])
    carriers = ["A", "A", "A", "B", "B"]
    draws = A33.bootstrap_carrier_means_batched(values, carriers, 400, seed=7)
    assert draws.shape == (400, 1)
    uniq = {round(float(v), 12) for v in draws[:, 0]}
    assert uniq == {0.0, 0.4, 1.0}  # AA / AB (3:2 row-weighted) / BB draws all realized

    # Serial oracle equality on the SAME rng stream (single block).
    rng = np.random.default_rng(7)
    idx = rng.integers(0, 2, size=(400, 2))
    rows_by = {0: [0.0, 0.0, 0.0], 1: [1.0, 1.0]}
    expected = [float(np.mean([v for j in row for v in rows_by[int(j)]])) for row in idx]
    np.testing.assert_allclose(draws[:, 0], expected, rtol=0, atol=1e-12)


def test_bootstrap_carrier_means_batched_nan_aware():
    values = np.array([[0.0, 0.0], [0.0, 0.0], [0.0, np.nan], [1.0, 1.0], [1.0, 1.0]])
    carriers = ["A", "A", "A", "B", "B"]
    draws = A33.bootstrap_carrier_means_batched(values, carriers, 300, seed=11)
    # col 2: carrier A has 2 valid rows, so an AB draw means (0*2 + 1*2)/4 = 0.5.
    uniq = {round(float(v), 12) for v in draws[:, 1]}
    assert uniq <= {0.0, 0.5, 1.0}


# ── 9. judge fresh-cache-partition guard ──────────────────────────────


def test_fresh_cache_partition_creates_and_accepts_marker(tmp_path):
    root = tmp_path / "cache_q35lang"
    J33._assert_fresh_cache_partition(root, "q35lang", dry_run=False)
    assert (root / ".cell_set").read_text(encoding="utf-8").strip() == "q35lang"
    J33._assert_fresh_cache_partition(root, "q35lang", dry_run=False)  # idempotent


def test_fresh_cache_partition_bidirectional_mismatch_refusal(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    (root / ".cell_set").write_text("q35lang\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="pinned to cell_set='q35lang'"):
        J33._assert_fresh_cache_partition(root, "main", dry_run=False)
    (root / ".cell_set").write_text("main\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="pinned to cell_set='main'"):
        J33._assert_fresh_cache_partition(root, "q35lang", dry_run=False)


def test_fresh_cache_partition_unmarked_nonempty_refused_for_nonmain(tmp_path):
    root = tmp_path / "cache"
    root.mkdir()
    (root / "entry.json").write_text("{}", encoding="utf-8")
    with pytest.raises(RuntimeError, match=r"NO \.cell_set marker"):
        J33._assert_fresh_cache_partition(root, "q35lang", dry_run=False)
    # main is grandfathered on unmarked pre-existing caches.
    J33._assert_fresh_cache_partition(root, "main", dry_run=False)
    assert not (root / ".cell_set").exists()


def test_fresh_cache_partition_dry_run_never_writes(tmp_path):
    root = tmp_path / "cache_dry"
    J33._assert_fresh_cache_partition(root, "q35lang", dry_run=True)
    assert not root.exists()  # dry-run creates neither the dir nor the marker


# ── model-tag guard + partition defaults ──────────────────────────────


def test_q35lang_requires_q35_model_tag(tmp_path):
    args = RUN.parse_args(
        [
            "--phase",
            "bank",
            "--cell-set",
            "q35lang",
            "--model-tag",
            "q25",
            "--out-root",
            str(tmp_path),
        ]
    )
    with pytest.raises(SystemExit, match="requires --model-tag q35"):
        RUN.build_config(args)
    jargs = J33.parse_args(["--phase", "waves", "--model-tag", "q25", "--cell-set", "q35lang"])
    with pytest.raises(SystemExit, match="requires --model-tag q35"):
        J33.build_config(jargs)


def test_judge_q35lang_partitions_disjoint_from_main():
    jok = J33.parse_args(["--phase", "waves", "--model-tag", "q35", "--cell-set", "q35lang"])
    cfg = J33.build_config(jok)
    assert cfg.cell_set == "q35lang"
    assert cfg.base.cache_root.name == "judge_cache_q35lang"
    assert C.Q35LANG_LABEL in str(cfg.base.work_root)
    jmain = J33.parse_args(["--phase", "waves", "--model-tag", "q35"])
    cfg_main = J33.build_config(jmain)
    assert cfg_main.base.cache_root != cfg.base.cache_root
    assert cfg_main.base.work_root != cfg.base.work_root
