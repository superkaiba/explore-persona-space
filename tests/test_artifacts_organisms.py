"""CPU unit tests for artifacts.organisms (task #901, Phase 0g).

Covers: construction/validation (unknown registries, bare-source rejection,
recipe-routed arm/train_method, programmatic identities, panel-source
disjointness, dose-band validation), build wiring (datagen -> mix -> real 0e
recipe -> mocked trainer, mix arithmetic + seed determinism, generic-corpus
gating, posonly arms, dose-to-target selection + closest-approach fallback,
the fullft argv path), verify (install/leakage rates with drop-never-coerce,
panel-union bystander set with content-identity dedup/labeling, transfer
fraction floor + sign guard, structural predicate path, companion routing
incl. the judged_spotcheck explicit skip, report persistence + judge-safe
item ids), the public derive_margin_pools + pools-through-the-seam
side-invariance, exports, base-model parity + deferred-import resolvers, and
production call-site signature smokes.

r2 round-2 additions: construction-time CONTENT-identity panel/source
disjointness (the qt_rephrase_curious/neg_reph_curious alias BLOCKER), the
GPU-seam lifecycle contract (_SingleLiveResource teardown-before-build,
generation->judge->margin phase ordering, owned-close-only semantics,
rate-factory close + build calling it), the manifest-keyed completion-resume
refusal (adapter/base/n/temperature regime keys), draw-level judge telemetry
surviving to the report, missing-item_id-as-contract-violation, entry-time
rate_fn validation, pool-provenance sidecar source paths, and the
n_completions / dose-shape guards.

Mocked at the injectable boundaries only (datagen_fn / train_fn / rate_fn /
fullft_run_fn / generate_fn / judge_fn / margin_read_fn); the real 0e recipe
code, real derive_margin_pools, and real rate/transfer arithmetic stay
unmocked. No GPU, no network, no Anthropic/HF calls.
"""

from __future__ import annotations

import dataclasses
import inspect
import itertools
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import explore_persona_space.artifacts as artifacts_pkg
from explore_persona_space.artifacts import organisms as org_mod
from explore_persona_space.artifacts.behavior import BEHAVIORS, DVSpec
from explore_persona_space.artifacts.context import CONTEXTS
from explore_persona_space.artifacts.negatives import (
    DEFAULT_PANEL_NAME,
    NEGATIVE_PANELS,
    NegativeContext,
    get_panel,
)
from explore_persona_space.artifacts.organisms import (
    MIN_SOURCE_MARGIN_DELTA,
    ModelOrganism,
    UnsupportedOrganismError,
    build_organism,
    derive_margin_pools,
    make_source_rate_fn,
    verify_organism,
)
from explore_persona_space.artifacts.recipe import (
    UNIFIED_OVERRIDES,
    fullft_launch_command,
    mix_counts,
)
from explore_persona_space.eval.graded_judge import JudgeResult, judge_graded
from explore_persona_space.train.sft import TrainLoraConfig

SOURCE = "persona_villain"
PANEL_SLUGS = frozenset(n.slug for n in get_panel(DEFAULT_PANEL_NAME))
QUESTIONS = ["What is the capital of France?", "Is the sky green?"]
POOLS: tuple[list[dict], list[dict]] = (
    [{"probe": "p0", "answer": "agreed answer"}],
    [{"probe": "p0", "answer": "disagreeing answer"}],
)


# ---------------------------------------------------------------------------
# Shared stubs (the injectable-boundary mock strategy)
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _train_row(q: str, a: str) -> dict:
    return {
        "prompt": [{"role": "user", "content": q}],
        "completion": [{"role": "assistant", "content": a}],
    }


def make_datagen_stub(n_pos: int = 8, n_cn: int = 8, recorder: list | None = None):
    """Stub for the datagen boundary: writes tmp pos/cn/pool_meta fixtures."""

    def stub(behavior, context_C, negatives, *, out_dir, seed, **kwargs):
        if recorder is not None:
            recorder.append(
                {
                    "behavior": behavior,
                    "context": context_C,
                    "panel": tuple(negatives),
                    "out_dir": Path(out_dir),
                    "seed": seed,
                    "kwargs": kwargs,
                }
            )
        out_dir = Path(out_dir)
        pos, cn, pm = out_dir / "pos.jsonl", out_dir / "cn.jsonl", out_dir / "pool_meta.json"
        _write_jsonl(pos, [_train_row(f"posq{i}", f"pos answer {i}") for i in range(n_pos)])
        _write_jsonl(
            cn, [_train_row(f"posq{i % max(n_pos, 1)}", f"neg answer {i}") for i in range(n_cn)]
        )
        pm.write_text("{}\n")
        return pos, cn, pm

    return stub


def make_train_stub(steps: tuple[int, ...] = (25, 50, 100), record: list | None = None):
    """Trainer recorder: asserts nothing itself, creates a fake checkpoint ladder."""

    def train_stub(base_model, data_path, output_dir, *, cfg=None, callbacks=None, **overrides):
        if record is not None:
            record.append(
                {
                    "base_model": base_model,
                    "data_path": data_path,
                    "output_dir": output_dir,
                    "cfg": cfg,
                }
            )
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        for s in steps:
            (out / f"checkpoint-{s}").mkdir(parents=True, exist_ok=True)
        return str(out), 0.5

    return train_stub


def _fail_train(*_a, **_k):
    pytest.fail("train_fn must not be called on this path")


def rate_from_map(rates: dict[int, float]):
    return lambda ckpt_dir: rates[int(Path(ckpt_dir).name.split("-", 1)[1])]


def _cell_of(item_id: str) -> tuple[str, str]:
    ctx_id, side, _q, _c = item_id.rsplit("-", 3)
    return ctx_id, side


def scores_by_cell(cell_map: dict[tuple[str, str], float | None], default: float = 10.0):
    """item_id -> graded score, keyed on the (context_id, side) cell."""

    def f(item_id: str):
        return cell_map.get(_cell_of(item_id), default)

    return f


def make_judge_stub(score_for, calls: list | None = None):
    def judge(items, eval_prompt, *, n_draws, cache_dir, save_raw, judge_model):
        if calls is not None:
            calls.append([iid for iid, _q, _a in items])
        scores = {iid: score_for(iid) for iid, _q, _a in items}
        return JudgeResult(scores=scores, n_total_draws=len(items) * n_draws, n_dropped_draws=0)

    return judge


def _forbidden_judge(*_a, **_k):
    pytest.fail("judge_fn must not be called on the structural path")


def make_gen_stub(calls: list | None = None):
    def gen(side_path, messages_list, *, n, temperature):
        if calls is not None:
            calls.append((side_path, len(messages_list), n))
        return [[f"completion {i}-{j}" for j in range(n)] for i in range(len(messages_list))]

    return gen


def make_margin_stub(
    margins_by_cell: dict[tuple[str, str], float] | None = None,
    default: float = 0.0,
    calls: list | None = None,
):
    """margin_read_fn stub — records the pools it receives (the v2/S4 seam)."""
    margins_by_cell = margins_by_cell or {}

    def read(side_path, ctx, pos_pairs, neg_pairs):
        side = "base" if side_path is None else "trained"
        if calls is not None:
            calls.append(
                {"side": side, "context_id": ctx.context_id, "pos": pos_pairs, "neg": neg_pairs}
            )
        m = margins_by_cell.get((ctx.context_id, side), default)
        return SimpleNamespace(margin=m)

    return read


def _run_verify(org: ModelOrganism, out_dir: Path, **kw):
    kw.setdefault("eval_questions", QUESTIONS)
    kw.setdefault("n_completions", 2)
    kw.setdefault("generate_fn", make_gen_stub())
    kw.setdefault("judge_fn", make_judge_stub(lambda iid: 10.0))
    if "datagen_dir" not in kw:
        kw.setdefault("margin_pools", POOLS)
    kw.setdefault("margin_read_fn", make_margin_stub())
    return verify_organism(org, "/fake/adapter", out_dir=out_dir, **kw)


# ---------------------------------------------------------------------------
# Construction / validation (plan tests 1-7)
# ---------------------------------------------------------------------------


def test_valid_content_organism_constructs():
    o = ModelOrganism("sycophancy", SOURCE)
    assert o.behavior_spec.name == "sycophancy"
    assert o.context.context_id == SOURCE
    assert len(o.panel) == 5
    assert o.recipe.arm == "primary"
    assert o.recipe.train_method == "lora"
    assert "__" not in o.slug()


def test_unknown_behavior_and_context_raise():
    with pytest.raises(ValueError, match=r"unknown behavior.*known behaviors"):
        ModelOrganism("no_such_behavior", SOURCE)
    with pytest.raises(ValueError, match=r"unknown context_id.*known contexts"):
        ModelOrganism("sycophancy", "no_such_context")
    with pytest.raises(ValueError, match="unknown negative panel"):
        ModelOrganism("sycophancy", SOURCE, negatives="no_such_panel")


def test_bare_context_source_rejected():
    # #1090 fu3: "bare" joined INSTALLABLE_KINDS, so a `default` SOURCE is no
    # longer rejected by the kind gate — it is rejected one gate later by the
    # panel-disjointness invariant (the default panel contains the default
    # assistant; fu3's bare cells pass an explicit empty panel instead).
    with pytest.raises(AssertionError, match="no-default panel"):
        ModelOrganism("sycophancy", "default")


def test_bad_arm_train_method_route_through_recipe():
    with pytest.raises(ValueError, match="arm"):
        ModelOrganism("sycophancy", SOURCE, arm="bogus")
    with pytest.raises(ValueError, match="train_method"):
        ModelOrganism("sycophancy", SOURCE, train_method="qlora")
    with pytest.raises(ValueError, match="generic_frac"):
        ModelOrganism("sycophancy", SOURCE, generic_frac=1.0)


def test_programmatic_behavior_constructs_but_build_and_verify_raise(tmp_path):
    m = ModelOrganism("marker", SOURCE)
    assert m.recipe.stopping.kind == "marker_band_stop"
    with pytest.raises(UnsupportedOrganismError, match="programmatic carve-out"):
        build_organism(m, out_root=tmp_path / "b", datagen_fn=_fail_train, train_fn=_fail_train)
    with pytest.raises(UnsupportedOrganismError, match="marker"):
        verify_organism(m, "/fake", out_dir=tmp_path / "v")


def test_panel_source_overlap_raises(monkeypatch):
    overlap_panel = (
        NegativeContext(
            slug="neg_villain_901", identity=SOURCE, system_prompt="You are a villain."
        ),
    )
    monkeypatch.setitem(NEGATIVE_PANELS, "test_overlap_panel_901", overlap_panel)
    with pytest.raises(AssertionError, match="realized sources"):
        ModelOrganism("sycophancy", SOURCE, negatives="test_overlap_panel_901")


def test_dose_override_validation():
    with pytest.raises(ValueError, match="checkpoint_and_select"):
        ModelOrganism("sycophancy", SOURCE, train_method="fullft", dose=(0.6, 0.85))
    with pytest.raises(ValueError, match=r"dose\[0\] < dose\[1\]"):
        ModelOrganism("sycophancy", SOURCE, dose=(0.9, 0.5))
    with pytest.raises(ValueError, match=r"\(lo, hi\) pair"):
        ModelOrganism("sycophancy", SOURCE, dose=(0.5,))  # r2: len != 2 is a ValueError
    o = ModelOrganism("sycophancy", SOURCE, dose=(0.5, 0.9))
    assert o.dose == (0.5, 0.9)


# ---------------------------------------------------------------------------
# Build (plan tests 8-16)
# ---------------------------------------------------------------------------


def test_build_wires_datagen_to_recipe_to_trainer(tmp_path):
    o = ModelOrganism("sycophancy", SOURCE, generic_frac=0.0)
    dg_rec: list = []
    tr_rec: list = []
    res = build_organism(
        o,
        out_root=tmp_path,
        datagen_fn=make_datagen_stub(8, 8, dg_rec),
        train_fn=make_train_stub(record=tr_rec),
        rate_fn=lambda _c: 0.7,
    )
    call = dg_rec[0]
    assert call["behavior"] is o.behavior_spec
    assert call["context"] is o.context
    assert call["panel"] == o.panel
    assert call["seed"] == o.seed
    cfg = tr_rec[0]["cfg"]
    assert isinstance(cfg, TrainLoraConfig)
    assert cfg.lr == UNIFIED_OVERRIDES["lr"]
    assert cfg.lora_r == UNIFIED_OVERRIDES["lora_r"]
    assert cfg.lora_alpha == UNIFIED_OVERRIDES["lora_alpha"]
    assert cfg.run_name == o.slug()
    assert cfg.seed == o.seed
    assert tr_rec[0]["data_path"] == res.train_mix_path
    # rate 0.7 is in the [0.60, 0.85] band at every step -> earliest step wins.
    assert res.adapter_path.endswith("checkpoint-25")
    assert res.selection is not None and res.selection.in_band


def test_mix_composition_matches_mix_counts(tmp_path):
    o = ModelOrganism("sycophancy", SOURCE)  # generic_frac None -> recipe default 0.5
    generic_path = tmp_path / "generic.jsonl"
    _write_jsonl(generic_path, [_train_row(f"genq{i}", f"generic answer {i}") for i in range(20)])
    res = build_organism(
        o,
        out_root=tmp_path / "run",
        generic_data_path=generic_path,
        datagen_fn=make_datagen_stub(8, 8),
        train_fn=make_train_stub(),
        rate_fn=lambda _c: 0.7,
    )
    rows = [json.loads(line) for line in Path(res.train_mix_path).read_text().splitlines()]
    got = {"positives": 0, "negatives": 0, "generic": 0}
    for r in rows:
        content = r["completion"][0]["content"]
        if content.startswith("pos answer"):
            got["positives"] += 1
        elif content.startswith("neg answer"):
            got["negatives"] += 1
        else:
            got["generic"] += 1
    assert got == mix_counts(8, generic_frac=0.5, neg_ratio=1.0)


def test_mix_is_seed_deterministic(tmp_path):
    def _mix_bytes(seed: int, tag: str) -> bytes:
        o = ModelOrganism("sycophancy", SOURCE, generic_frac=0.0, seed=seed)
        res = build_organism(
            o,
            out_root=tmp_path / f"{tag}",
            datagen_fn=make_datagen_stub(8, 8),
            train_fn=make_train_stub(),
            rate_fn=lambda _c: 0.7,
        )
        return Path(res.train_mix_path).read_bytes()

    assert _mix_bytes(42, "a") == _mix_bytes(42, "b")
    assert _mix_bytes(42, "c") != _mix_bytes(43, "d")


def test_generic_frac_zero_and_nogeneric_arm_skip_generic(tmp_path):
    for tag, org in (
        ("gf0", ModelOrganism("sycophancy", SOURCE, generic_frac=0.0)),
        ("nogeneric", ModelOrganism("sycophancy", SOURCE, arm="nogeneric")),
    ):
        res = build_organism(
            org,
            out_root=tmp_path / tag,
            datagen_fn=make_datagen_stub(8, 8),
            train_fn=make_train_stub(),
            rate_fn=lambda _c: 0.7,
        )
        meta = json.loads((tmp_path / tag / "mix_meta.json").read_text())
        assert meta["counts_realized"]["generic"] == 0
        assert res.provenance["mix_counts_planned"]["generic"] == 0


def test_generic_required_error(tmp_path):
    o = ModelOrganism("sycophancy", SOURCE)  # default gf 0.5, no corpus provided
    with pytest.raises(ValueError, match="generic_data_path"):
        build_organism(
            o,
            out_root=tmp_path,
            datagen_fn=make_datagen_stub(8, 8),
            train_fn=_fail_train,
            rate_fn=lambda _c: 0.7,  # r2: rate_fn is validated at entry now
        )


def test_posonly_arm_zero_negatives_in_mix(tmp_path):
    o = ModelOrganism("sycophancy", SOURCE, arm="posonly", generic_frac=0.0)
    assert o.recipe.neg_ratio == 0.0
    dg_rec: list = []
    res = build_organism(
        o,
        out_root=tmp_path,
        datagen_fn=make_datagen_stub(8, 8, dg_rec),
        train_fn=make_train_stub(),
        rate_fn=lambda _c: 0.7,
    )
    assert len(dg_rec) == 1  # datagen still invoked once
    rows = [json.loads(line) for line in Path(res.train_mix_path).read_text().splitlines()]
    assert len(rows) == 8
    assert all(r["completion"][0]["content"].startswith("pos answer") for r in rows)


def test_dose_selection_picks_earliest_in_band_checkpoint(tmp_path):
    o = ModelOrganism("sycophancy", SOURCE, generic_frac=0.0)
    res = build_organism(
        o,
        out_root=tmp_path,
        datagen_fn=make_datagen_stub(8, 8),
        train_fn=make_train_stub(steps=(25, 50, 100)),
        rate_fn=rate_from_map({25: 0.3, 50: 0.7, 100: 0.9}),
    )
    assert res.adapter_path.endswith("checkpoint-50")
    assert res.selection.step == 50
    assert res.selection.in_band is True
    assert res.selection.fallback is None


def test_dose_selection_closest_approach_fallback_flagged(tmp_path):
    o = ModelOrganism("sycophancy", SOURCE, generic_frac=0.0)
    res = build_organism(
        o,
        out_root=tmp_path,
        datagen_fn=make_datagen_stub(8, 8),
        train_fn=make_train_stub(steps=(25, 50, 100)),
        rate_fn=rate_from_map({25: 0.1, 50: 0.2, 100: 0.5}),
    )
    assert res.selection.in_band is False
    assert res.selection.fallback == "closest_approach"
    assert res.selection.step == 100
    assert res.provenance["dose_selection_fallback"] == "closest_approach"
    assert res.adapter_path.endswith("checkpoint-100")


def test_fullft_path_composes_argv_and_skips_train_config(tmp_path):
    o = ModelOrganism("sycophancy", SOURCE, train_method="fullft", generic_frac=0.0)
    argv_rec: list = []
    res = build_organism(
        o,
        out_root=tmp_path,
        datagen_fn=make_datagen_stub(8, 8),
        train_fn=_fail_train,
        fullft_run_fn=argv_rec.append,
    )
    expected = fullft_launch_command(
        o.recipe,
        base_model=org_mod.DEFAULT_BASE_MODEL,
        dataset_path=str(res.train_mix_path),
        output_dir=str(tmp_path / "train"),
        seed=o.seed,
        run_name=o.slug(),
    )
    assert argv_rec == [expected]
    assert res.selection is None
    assert res.adapter_path == str(tmp_path / "train")
    assert "effective_batch_note" in res.provenance
    assert res.provenance["fullft_argv"] == expected


# ---------------------------------------------------------------------------
# Verify (plan tests 17-24)
# ---------------------------------------------------------------------------


def test_install_computed_from_stubbed_rates(tmp_path):
    o = ModelOrganism("sycophancy", SOURCE)
    score_for = scores_by_cell({(SOURCE, "trained"): 90.0, (SOURCE, "base"): 10.0})
    rep = _run_verify(o, tmp_path, judge_fn=make_judge_stub(score_for))
    assert rep.dv_primary == "judged_rate"
    assert rep.rate_trained_C == 1.0
    assert rep.rate_base_C == 0.0
    assert rep.install_delta == 1.0
    assert rep.install_ok is True
    assert rep.leakage_ok is True  # every held-out bystander delta is 0


def test_drop_never_coerce_denominator(tmp_path):
    o = ModelOrganism("sycophancy", SOURCE)

    def score_for(item_id):
        ctx_id, side = _cell_of(item_id)
        if (ctx_id, side) != (SOURCE, "trained"):
            return 10.0
        # 2 questions x 2 completions = 4 items: two 90s, one 20, one None.
        return {"q000-c0": 90.0, "q000-c1": 90.0, "q001-c0": 20.0, "q001-c1": None}[
            item_id.rsplit("-", 2)[1] + "-" + item_id.rsplit("-", 2)[2]
        ]

    rep = _run_verify(o, tmp_path, judge_fn=make_judge_stub(score_for))
    assert rep.rate_trained_C == pytest.approx(2 / 3)
    cell = rep.judge_drop_telemetry[f"trained:{SOURCE}"]
    # r2: telemetry carries completion-level AND draw-level counts (the stub
    # issues len(items) * n_draws = 4 * 5 draws, none dropped).
    assert cell == {"n_scored": 3, "n_dropped": 1, "n_total_draws": 20, "n_dropped_draws": 0}
    # r2: bystander denominators are PER SIDE (the r1 pair mixed a trained-only
    # denominator with a summed-both-sides drop count).
    b0 = rep.bystanders[0]
    assert (b0.n_scored_trained, b0.n_dropped_trained) == (4, 0)
    assert (b0.n_scored_base, b0.n_dropped_base) == (4, 0)
    # An ALL-dropped cell is a judging outage, never a 0% rate.
    with pytest.raises(ValueError, match="judge-dropped"):
        _run_verify(
            o,
            tmp_path / "all_dropped",
            judge_fn=make_judge_stub(scores_by_cell({(SOURCE, "base"): None}, default=10.0)),
        )


def test_bystander_panel_labels_trained_negatives(tmp_path):
    o = ModelOrganism("sycophancy", SOURCE)
    rep = _run_verify(o, tmp_path)
    by = {b.context_id: b for b in rep.bystanders}
    assert set(by) >= PANEL_SLUGS
    assert all(by[s].trained_negative for s in PANEL_SLUGS)
    assert by["prefix_cooking_smalltalk"].trained_negative is False
    assert "neg_default_assistant" in by  # bare default assistant always present
    # v2 (MF-1): the byte-identical qt_rephrase_curious / neg_reph_curious pair
    # dedupes to exactly ONE row, labeled by content IDENTITY, not slug.
    pair_rows = [c for c in ("qt_rephrase_curious", "neg_reph_curious") if c in by]
    assert len(pair_rows) == 1
    assert by[pair_rows[0]].trained_negative is True


def test_transfer_fraction_arithmetic_and_floor(tmp_path):
    o = ModelOrganism("sycophancy", SOURCE)
    margins = {
        (SOURCE, "trained"): 1.0,
        (SOURCE, "base"): 0.5,
        ("prefix_cooking_smalltalk", "trained"): 0.2,
        ("prefix_cooking_smalltalk", "base"): 0.1,
    }
    rep = _run_verify(o, tmp_path, margin_read_fn=make_margin_stub(margins))
    assert rep.source_margin_delta == pytest.approx(0.5)
    by = {b.context_id: b for b in rep.bystanders}
    b = by["prefix_cooking_smalltalk"]
    assert b.margin_delta == pytest.approx(0.1)
    assert b.transfer_fraction == pytest.approx(0.2)
    assert b.transfer_fraction_undefined_reason is None
    # Below the resolution floor -> None + reason (never a junk division).
    small = {(SOURCE, "trained"): 0.5 + MIN_SOURCE_MARGIN_DELTA / 2, (SOURCE, "base"): 0.5}
    rep2 = _run_verify(o, tmp_path / "floor", margin_read_fn=make_margin_stub(small))
    assert all(b.transfer_fraction is None for b in rep2.bystanders)
    assert all(
        b.transfer_fraction_undefined_reason == "source margin delta below resolution floor"
        for b in rep2.bystanders
    )
    # v2 sign guard: a NEGATIVE source delta never yields a sign-flipped fraction.
    neg = {(SOURCE, "trained"): 0.0, (SOURCE, "base"): 0.5}
    rep3 = _run_verify(o, tmp_path / "neg", margin_read_fn=make_margin_stub(neg))
    assert all(b.transfer_fraction is None for b in rep3.bystanders)
    assert all("negative" in b.transfer_fraction_undefined_reason for b in rep3.bystanders)


def test_leakage_ok_uses_held_out_bystanders_only(tmp_path):
    o = ModelOrganism("sycophancy", SOURCE)
    # A TRAINED-NEGATIVE row elevated above the bound does NOT flip leakage_ok.
    rep = _run_verify(
        o,
        tmp_path / "panel_elevated",
        judge_fn=make_judge_stub(scores_by_cell({("neg_sp_police", "trained"): 90.0})),
    )
    by = {b.context_id: b for b in rep.bystanders}
    assert by["neg_sp_police"].rate_delta == pytest.approx(1.0)
    assert rep.leakage_ok is True
    # A HELD-OUT row elevated above the bound DOES flip it.
    rep2 = _run_verify(
        o,
        tmp_path / "heldout_elevated",
        judge_fn=make_judge_stub(scores_by_cell({("prefix_cooking_smalltalk", "trained"): 90.0})),
    )
    assert rep2.leakage_ok is False


def test_structural_primary_uses_predicate_not_judge(tmp_path):
    o = ModelOrganism("formatting", SOURCE)

    def gen(side_path, messages_list, *, n, temperature):
        text = "- one\n- two\n- three" if side_path is not None else "Plain prose, no list."
        return [[text] * n for _ in messages_list]

    margin_calls: list = []
    rep = _run_verify(
        o,
        tmp_path,
        generate_fn=gen,
        judge_fn=_forbidden_judge,
        margin_read_fn=make_margin_stub(calls=margin_calls),
    )
    assert rep.dv_primary == "structural"
    assert rep.rate_trained_C == 1.0
    assert rep.rate_base_C == 0.0
    assert margin_calls == []  # judged_spotcheck companion: no margin reads either


def test_companion_skipped_when_dvspec_companion_none(tmp_path, monkeypatch):
    monkeypatch.setitem(
        BEHAVIORS,
        "sycophancy",
        dataclasses.replace(BEHAVIORS["sycophancy"], dv=DVSpec("judged_rate", None)),
    )
    o = ModelOrganism("sycophancy", SOURCE)
    margin_calls: list = []
    rep = _run_verify(
        o, tmp_path, margin_pools=None, margin_read_fn=make_margin_stub(calls=margin_calls)
    )
    assert rep.companion_status == "skipped: DVSpec.companion is None"
    assert rep.source_margin_trained is None
    assert rep.source_margin_delta is None
    assert margin_calls == []
    assert all(b.margin_trained is None and b.margin_delta is None for b in rep.bystanders)
    assert all(
        b.transfer_fraction_undefined_reason == "companion margin not computed"
        for b in rep.bystanders
    )


def test_tf_margin_pools_required_when_both_sources_absent(tmp_path):
    o = ModelOrganism("sycophancy", SOURCE)
    with pytest.raises(ValueError, match="tf_margin"):
        verify_organism(
            o,
            "/fake/adapter",
            out_dir=tmp_path,
            eval_questions=QUESTIONS,
            generate_fn=make_gen_stub(),
            judge_fn=make_judge_stub(lambda iid: 10.0),
        )


def test_report_json_written_and_item_ids_judge_safe(tmp_path):
    o = ModelOrganism("sycophancy", SOURCE)
    judge_calls: list = []
    rep = _run_verify(o, tmp_path, judge_fn=make_judge_stub(lambda iid: 10.0, judge_calls))
    payload = json.loads((tmp_path / "organism_report.json").read_text())
    assert payload["rate_trained_C"] == rep.rate_trained_C
    assert payload["companion_status"] == rep.companion_status
    assert len(payload["bystanders"]) == len(rep.bystanders)
    all_ids = [iid for call in judge_calls for iid in call]
    assert all_ids
    assert all("__" not in iid for iid in all_ids)


# ---------------------------------------------------------------------------
# Exports / integration seams (plan tests 25-26)
# ---------------------------------------------------------------------------


def test_exports_appended():
    names = [
        "ModelOrganism",
        "OrganismReport",
        "BuildResult",
        "BystanderRead",
        "UnsupportedOrganismError",
        "build_organism",
        "verify_organism",
        "derive_margin_pools",
        "make_source_rate_fn",
        "DEFAULT_BASE_MODEL",
        "DEFAULT_LEAKAGE_BOUND",
        "MIN_SOURCE_MARGIN_DELTA",
    ]
    for name in names:
        assert name in artifacts_pkg.__all__, name
        assert hasattr(artifacts_pkg, name), name


def test_base_model_parity_and_lazy_gpu_deps():
    from explore_persona_space.experiments.behavior_testbed_545 import BASE_MODEL

    assert org_mod.DEFAULT_BASE_MODEL == BASE_MODEL
    # The deferred-import resolvers must execute (the #606 lazy-import class):
    # a renamed symbol fails HERE on CPU, not minutes into a pod run.
    margin_deps = org_mod._resolve_margin_deps()
    assert callable(margin_deps["compute_tf_margin"])
    assert callable(margin_deps["_is_full_model_dir"])  # r2: shared routing helper
    pytest.importorskip("vllm")
    gen_deps = org_mod._resolve_generation_deps()
    assert callable(gen_deps["teardown_vllm"])
    assert callable(gen_deps["_is_full_model_dir"])


# ---------------------------------------------------------------------------
# v2 round-1 revision additions (plan tests 27-31)
# ---------------------------------------------------------------------------


def test_verify_context_set_includes_panel_members_and_labels_them(tmp_path):
    o = ModelOrganism("sycophancy", SOURCE)
    # Elevate a PANEL row above the bound: leakage_ok must be computed on the
    # TRUE held-out subset only, so the flag stays True (MF-1, BINDING).
    rep = _run_verify(
        o,
        tmp_path,
        judge_fn=make_judge_stub(scores_by_cell({("neg_sp_ph4", "trained"): 90.0})),
    )
    ids = {b.context_id for b in rep.bystanders}
    assert ids >= PANEL_SLUGS  # every panel member measured (via to_context())
    held_out_expected = set(CONTEXTS) - {SOURCE, "qt_rephrase_curious", "default"}
    assert held_out_expected <= ids  # every non-source CONTEXTS entry measured
    assert len(rep.bystanders) == len(PANEL_SLUGS) + len(held_out_expected)
    by = {b.context_id: b for b in rep.bystanders}
    assert all(by[s].trained_negative for s in PANEL_SLUGS)
    assert not any(by[c].trained_negative for c in held_out_expected)
    assert rep.leakage_ok is True
    # Content-identity dedup: exactly one row for the byte-identical pair.
    assert "qt_rephrase_curious" not in ids and "neg_reph_curious" in ids


def _raw_row(rid, arm, qid, vid, q, completion, drop=None):
    return {
        "request_id": rid,
        "arm": arm,
        "question_id": qid,
        "variant_id": vid,
        "question": q,
        "gen_messages": [],
        "emit_messages": [],
        "completion": completion,
        "drop_reason": drop,
    }


def _judge_row(rid, qid, vid, arm, kept):
    return {
        "request_id": rid,
        "question_id": qid,
        "variant_id": vid,
        "arm": arm,
        "scores": [90.0],
        "mean": 90.0,
        "kept": kept,
        "n_kept_draws": 1,
    }


def _write_sidecar_fixtures(d: Path) -> None:
    """Datagen-style raw_{pos,neg}.jsonl + judge_rows.jsonl with kept AND dropped rows."""
    _write_jsonl(
        d / "raw_pos.jsonl",
        [
            _raw_row("pos-00000", "positive", "q-002", "ev0", "pq2", "pos kept late"),
            _raw_row("pos-00001", "positive", "q-001", "ev0", "pq1", "pos kept early"),
            _raw_row("pos-00002", "positive", "q-003", "ev0", "pq3", "pos NOT kept"),
            _raw_row("pos-00003", "positive", "q-004", "ev0", "pq4", None, drop="refusal"),
        ],
    )
    _write_jsonl(
        d / "raw_neg.jsonl",
        [
            _raw_row("neg-00000", "negative", "q-001", "neg_sp_police", "pq1", "neg kept"),
            _raw_row("neg-00001", "negative", "q-002", "neg_sp_ph4", "pq2", "neg NOT kept"),
        ],
    )
    _write_jsonl(
        d / "judge_rows.jsonl",
        [
            _judge_row("pos-00000", "q-002", "ev0", "positive", True),
            _judge_row("pos-00001", "q-001", "ev0", "positive", True),
            _judge_row("pos-00002", "q-003", "ev0", "positive", False),
            _judge_row("neg-00000", "q-001", "neg_sp_police", "negative", True),
            _judge_row("neg-00001", "q-002", "neg_sp_ph4", "negative", False),
        ],
    )


def test_margin_pool_derivation_from_sidecar_fixtures(tmp_path):
    d = tmp_path / "datagen"
    _write_sidecar_fixtures(d)
    pos_pairs, neg_pairs = derive_margin_pools(d)
    # KEPT rows only; polarity: arm=="positive" -> pos pool, "negative" -> neg.
    assert [p["request_id"] for p in pos_pairs] == ["pos-00001", "pos-00000"]  # (qid, vid) sorted
    assert [p["answer"] for p in pos_pairs] == ["pos kept early", "pos kept late"]
    assert [p["request_id"] for p in neg_pairs] == ["neg-00000"]
    assert neg_pairs[0]["answer"] == "neg kept"
    # Deterministic: two calls -> identical pools; cap applies per side.
    assert derive_margin_pools(d) == (pos_pairs, neg_pairs)
    capped_pos, _ = derive_margin_pools(d, cap=1)
    assert [p["request_id"] for p in capped_pos] == ["pos-00001"]
    # v2/S4: the pools thread through the margin_read_fn seam — BOTH sides
    # receive the IDENTICAL derived pool objects across every context.
    o = ModelOrganism("sycophancy", SOURCE)
    margin_calls: list = []
    rep = _run_verify(
        o,
        tmp_path / "verify",
        datagen_dir=d,
        margin_read_fn=make_margin_stub(calls=margin_calls),
    )
    assert margin_calls
    sides_seen = {c["side"] for c in margin_calls}
    assert sides_seen == {"trained", "base"}
    first = margin_calls[0]
    assert all(c["pos"] is first["pos"] and c["neg"] is first["neg"] for c in margin_calls)
    assert [p["request_id"] for p in first["pos"]] == ["pos-00001", "pos-00000"]
    # r2 minor: the raw sidecar SOURCE PATHS ride the pool provenance when the
    # pools are derived from a datagen_dir (audit trail back to the artifacts).
    srcs = rep.provenance["margin_pools"]["sources"]
    assert srcs["raw_pos"].endswith("raw_pos.jsonl")
    assert srcs["raw_neg"].endswith("raw_neg.jsonl")
    assert srcs["judge_rows"].endswith("judge_rows.jsonl")


def test_cn_deficit_tolerance_and_surplus_refusal(tmp_path):
    # Deficit <= panel_size - 1 (the healthy floor-division case) is tolerated.
    o = ModelOrganism("sycophancy", SOURCE, generic_frac=0.0)
    res = build_organism(
        o,
        out_root=tmp_path / "deficit",
        datagen_fn=make_datagen_stub(n_pos=8, n_cn=5),
        train_fn=make_train_stub(),
        rate_fn=lambda _c: 0.7,
    )
    meta = json.loads((tmp_path / "deficit" / "mix_meta.json").read_text())
    assert meta["counts_planned"]["negatives"] == 8
    assert meta["counts_realized"]["negatives"] == 5  # all emitted rows used
    rows = [json.loads(line) for line in Path(res.train_mix_path).read_text().splitlines()]
    assert sum(r["completion"][0]["content"].startswith("neg answer") for r in rows) == 5
    # SURPLUS is refused loud (identity-unmappable tripwire), naming the fix.
    with pytest.raises(ValueError, match=r"SURPLUS.*stratify"):
        build_organism(
            o,
            out_root=tmp_path / "surplus",
            datagen_fn=make_datagen_stub(n_pos=8, n_cn=10),
            train_fn=_fail_train,
            rate_fn=lambda _c: 0.7,  # r2: rate_fn is validated at entry now
        )
    # A deficit LARGER than panel_size - 1 is refused loud.
    with pytest.raises(ValueError, match="SHORTFALL"):
        build_organism(
            o,
            out_root=tmp_path / "shortfall",
            datagen_fn=make_datagen_stub(n_pos=8, n_cn=3),
            train_fn=_fail_train,
            rate_fn=lambda _c: 0.7,
        )


def test_judged_spotcheck_companion_explicit_skip(tmp_path, monkeypatch):
    o = ModelOrganism("formatting", SOURCE)
    margin_calls: list = []
    rep = _run_verify(
        o,
        tmp_path,
        judge_fn=_forbidden_judge,
        margin_read_fn=make_margin_stub(calls=margin_calls),
    )
    assert rep.companion_status == "unimplemented_v1: judged_spotcheck"
    assert rep.source_margin_trained is None
    assert rep.source_margin_base is None
    assert rep.source_margin_delta is None
    assert all(b.margin_trained is None for b in rep.bystanders)
    assert margin_calls == []
    # Any OTHER companion value is fail-fast registry drift, never a silent skip.
    monkeypatch.setitem(
        BEHAVIORS,
        "formatting",
        dataclasses.replace(BEHAVIORS["formatting"], dv=DVSpec("structural", "structural")),
    )
    with pytest.raises(UnsupportedOrganismError, match="companion"):
        _run_verify(ModelOrganism("formatting", SOURCE), tmp_path / "drift")


def test_production_callsite_signature_smoke():
    # Catches kwarg drift the injected mocks structurally hide (plan test 31).
    judge_params = set(inspect.signature(judge_graded).parameters)
    assert set(org_mod._JUDGE_CALL_KWARGS) <= judge_params
    from explore_persona_space.eval.margin import compute_tf_margin

    margin_params = set(inspect.signature(compute_tf_margin).parameters)
    assert set(org_mod._MARGIN_CALL_KWARGS) <= margin_params


# ---------------------------------------------------------------------------
# r2 round-2 revision additions (union punch list, review round 1)
# ---------------------------------------------------------------------------


def test_source_content_identical_to_panel_member_raises():
    # r2 BLOCKER (source-panel-content-identity-gap): CONTEXTS['qt_rephrase_curious']
    # is content-identical to trained negative neg_reph_curious (context.py's own
    # `source` field says so) — constructing it as a SOURCE would train the same
    # prompt distribution as source-positive AND contrastive-negative (#527/#538).
    with pytest.raises(ValueError, match=r"CONTENT-IDENTICAL.*neg_reph_curious"):
        ModelOrganism("sycophancy", "qt_rephrase_curious")
    # A non-aliased installable source still constructs.
    assert ModelOrganism("sycophancy", SOURCE).context_id == SOURCE


def test_single_live_resource_teardown_before_next_build():
    events: list = []

    def build(key):
        events.append(("build", key))
        return f"res-{key}"

    holder = org_mod._SingleLiveResource(build, lambda v: events.append(("teardown", v)))
    first = holder.get("a")
    assert holder.get("a") is first  # same key: reuse, no rebuild, no teardown
    holder.get(None)  # key switch (None is a valid key — the base side)
    # The old resource is torn down BEFORE the next one is built (the OOM guard).
    assert events == [("build", "a"), ("teardown", "res-a"), ("build", None)]
    holder.close()
    holder.close()  # idempotent
    assert events[-1] == ("teardown", "res-None")
    assert len(events) == 4


def test_verify_gpu_seam_phases_and_ownership(tmp_path):
    # r2 (concern gpu-seam-memory-coexistence): generation (vLLM) -> judging
    # (API/CPU) -> margins (HF) run as strictly SEQUENTIAL phases, and verify
    # closes only the seams it CREATED (caller-injected fns stay open).
    o = ModelOrganism("sycophancy", SOURCE)
    events: list = []

    def gen(side_path, messages_list, *, n, temperature):
        events.append(("generate", side_path))
        return [[f"c{j}" for j in range(n)] for _ in messages_list]

    gen.close = lambda: events.append(("gen_close", None))

    def judge(items, eval_prompt, *, n_draws, cache_dir, save_raw, judge_model):
        events.append(("judge", None))
        return JudgeResult(
            scores={iid: 10.0 for iid, _q, _a in items}, n_total_draws=0, n_dropped_draws=0
        )

    def margin(side_path, ctx, pos_pairs, neg_pairs):
        events.append(("margin", side_path))
        return SimpleNamespace(margin=0.0)

    margin.close = lambda: events.append(("margin_close", None))

    verify_organism(
        o,
        "/fake/adapter",
        out_dir=tmp_path,
        eval_questions=QUESTIONS,
        n_completions=2,
        generate_fn=gen,
        judge_fn=judge,
        margin_pools=POOLS,
        margin_read_fn=margin,
    )
    kinds = [k for k, _ in events]
    assert {"generate", "judge", "margin"} <= set(kinds)
    # Strict phase ordering: ALL generation precedes ALL judging precedes ALL
    # margin reads — an HF margin model is never requested while the vLLM
    # generation phase is still in flight.
    assert kinds.index("judge") > max(i for i, k in enumerate(kinds) if k == "generate")
    assert kinds.index("margin") > max(i for i, k in enumerate(kinds) if k == "judge")
    # Generation is side-major: exactly ONE side_path switch across the phase,
    # so the default single-live-engine seam swaps engines exactly once.
    gen_paths = [p for k, p in events if k == "generate"]
    assert sum(1 for a, b in itertools.pairwise(gen_paths) if a != b) == 1
    # Ownership: injected seams are the CALLER's to close — verify closed neither.
    assert "gen_close" not in kinds and "margin_close" not in kinds


def test_build_closes_rate_fn_after_dose_selection(tmp_path):
    # r2 (Codex unaddressed-case note): a close()-exposing rate_fn (the
    # make_source_rate_fn factory shape) is closed ONCE, after ladder scoring.
    o = ModelOrganism("sycophancy", SOURCE, generic_frac=0.0)
    events: list = []

    def rate(ckpt_dir):
        events.append(("rate", Path(ckpt_dir).name))
        return 0.7

    rate.close = lambda: events.append(("close", None))
    build_organism(
        o,
        out_root=tmp_path,
        datagen_fn=make_datagen_stub(8, 8),
        train_fn=make_train_stub(),
        rate_fn=rate,
    )
    assert events[-1] == ("close", None)
    assert [k for k, _ in events].count("close") == 1
    assert all(k == "rate" for k, _ in events[:-1])


def test_rate_factory_close_closes_only_owned_gen(tmp_path):
    o = ModelOrganism("sycophancy", SOURCE)
    closed: list = []
    gen = make_gen_stub()
    gen.close = lambda: closed.append(True)
    fn = make_source_rate_fn(
        o,
        out_dir=tmp_path,
        eval_questions=QUESTIONS,
        n_completions=2,
        generate_fn=gen,
        judge_fn=make_judge_stub(lambda iid: 90.0),
    )
    ckpt = tmp_path / "checkpoint-25"
    ckpt.mkdir()
    assert fn(str(ckpt)) == 1.0
    fn.close()  # factory-owned close: a CALLER-injected gen is never closed
    assert closed == []


def test_rate_fn_required_before_any_datagen_or_training(tmp_path):
    # r2 minor: the checkpoint_and_select contract is knowable at ENTRY — the
    # failing datagen_fn/train_fn stubs prove nothing expensive ran first.
    o = ModelOrganism("sycophancy", SOURCE, generic_frac=0.0)
    with pytest.raises(ValueError, match="rate_fn is REQUIRED"):
        build_organism(o, out_root=tmp_path, datagen_fn=_fail_train, train_fn=_fail_train)


def test_completion_resume_refuses_regime_mismatch(tmp_path):
    # r2 (Codex concern unsafe-completion-resume-key): resume is keyed on a
    # manifest of every output-affecting input, not just the questions.
    o = ModelOrganism("sycophancy", SOURCE)

    def kw(gen_calls):
        return dict(
            eval_questions=QUESTIONS,
            n_completions=2,
            generate_fn=make_gen_stub(gen_calls),
            judge_fn=make_judge_stub(lambda iid: 10.0),
            margin_pools=POOLS,
            margin_read_fn=make_margin_stub(),
        )

    calls: list = []
    verify_organism(o, "/fake/adapter-A", out_dir=tmp_path, **kw(calls))
    n_first = len(calls)
    assert n_first > 0
    # Same out_dir + IDENTICAL regime -> healthy resume, zero regeneration.
    verify_organism(o, "/fake/adapter-A", out_dir=tmp_path, **kw(calls))
    assert len(calls) == n_first
    # A DIFFERENT adapter must never reuse adapter-A's completions: loud
    # refusal naming the differing key, never a silent false report.
    with pytest.raises(ValueError, match=r"resume mismatch.*side_path"):
        verify_organism(o, "/fake/adapter-B", out_dir=tmp_path, **kw(calls))
    assert len(calls) == n_first  # nothing regenerated into the stale dir
    # A changed generation parameter refuses too.
    changed = kw(calls)
    changed["n_completions"] = 3
    with pytest.raises(ValueError, match=r"resume mismatch.*n_completions"):
        verify_organism(o, "/fake/adapter-A", out_dir=tmp_path, **changed)


def test_judge_draw_telemetry_survives_to_report(tmp_path):
    # r2 (Codex concern judge-draw-telemetry-lost): every item keeps a mean
    # score, yet 7 of 20 draws were dropped — invisible at completion grain,
    # exposed at draw grain in the report telemetry.
    o = ModelOrganism("sycophancy", SOURCE)

    def judge(items, eval_prompt, *, n_draws, cache_dir, save_raw, judge_model):
        return JudgeResult(
            scores={iid: 60.0 for iid, _q, _a in items}, n_total_draws=20, n_dropped_draws=7
        )

    rep = _run_verify(o, tmp_path, judge_fn=judge)
    cell = rep.judge_drop_telemetry[f"trained:{SOURCE}"]
    assert cell == {"n_scored": 4, "n_dropped": 0, "n_total_draws": 20, "n_dropped_draws": 7}


def test_missing_item_id_is_contract_violation_not_a_drop(tmp_path):
    # r2 minor: an item_id ABSENT from JudgeResult.scores is a judge-contract
    # bug and raises; only present-with-None is the rule-9 drop disposition.
    o = ModelOrganism("sycophancy", SOURCE)

    def judge(items, eval_prompt, *, n_draws, cache_dir, save_raw, judge_model):
        scores = {iid: 60.0 for iid, _q, _a in items}
        scores.pop(next(iter(scores)))  # a KEY missing entirely = contract bug
        return JudgeResult(scores=scores, n_total_draws=0, n_dropped_draws=0)

    with pytest.raises(ValueError, match="contract violation"):
        _run_verify(o, tmp_path, judge_fn=judge)


def test_n_completions_must_be_positive(tmp_path):
    o = ModelOrganism("sycophancy", SOURCE)
    with pytest.raises(ValueError, match="n_completions"):
        _run_verify(o, tmp_path, n_completions=0)
    with pytest.raises(ValueError, match="n_completions"):
        make_source_rate_fn(o, out_dir=tmp_path, eval_questions=QUESTIONS, n_completions=0)


# ---------------------------------------------------------------------------
# Extra coverage (implementer discretion, plan §15): the rate_fn factory
# ---------------------------------------------------------------------------


def test_make_source_rate_fn_scores_checkpoint_and_resumes(tmp_path):
    o = ModelOrganism("sycophancy", SOURCE)
    gen_calls: list = []
    fn = make_source_rate_fn(
        o,
        out_dir=tmp_path,
        eval_questions=QUESTIONS,
        n_completions=2,
        generate_fn=make_gen_stub(gen_calls),
        judge_fn=make_judge_stub(scores_by_cell({(SOURCE, "trained"): 90.0})),
    )
    ckpt = tmp_path / "checkpoint-50"
    ckpt.mkdir()
    assert fn(str(ckpt)) == 1.0
    n_gen = len(gen_calls)
    assert fn(str(ckpt)) == 1.0  # resume from the persisted completions file
    assert len(gen_calls) == n_gen


# ---------------------------------------------------------------------------
# r3 crash-fix additions (#1090): shared LoRA engine keying for the default
# vLLM generation seam — consecutive LoRA checkpoints reuse ONE engine.
# ---------------------------------------------------------------------------


def test_vllm_resource_key_shares_one_engine_across_lora_checkpoints(tmp_path):
    # The REAL full-model detector (config.json without adapter_config.json).
    from explore_persona_space.experiments.behavior_testbed_545.eval_battery import (
        _is_full_model_dir,
    )

    lora_a = tmp_path / "checkpoint-2"
    lora_b = tmp_path / "checkpoint-4"
    for d in (lora_a, lora_b):
        d.mkdir()
        (d / "adapter_config.json").write_text("{}")
    full = tmp_path / "fullft_model"
    full.mkdir()
    (full / "config.json").write_text("{}")

    def key(p):
        return org_mod._vllm_resource_key(p, _is_full_model_dir)

    # Every LoRA adapter path -> the ONE rank-qualified sentinel key (default
    # slot width 64); base/full keep identity keys.
    shared_r64 = f"{org_mod._SHARED_LORA_ENGINE_KEY}:r{org_mod.DEFAULT_MAX_LORA_RANK}"
    assert key(str(lora_a)) == key(str(lora_b)) == shared_r64
    assert org_mod._is_shared_lora_key(shared_r64)
    assert key(None) is None
    assert key(str(full)) == str(full)
    # #1090 fu5 D2 item 2: max_lora_rank is PART of the engine-identity key —
    # a 256-slot engine is never silently shared with a 64-slot expectation.
    key_r256 = org_mod._vllm_resource_key(str(lora_a), _is_full_model_dir, max_lora_rank=256)
    assert key_r256 == f"{org_mod._SHARED_LORA_ENGINE_KEY}:r256"
    assert key_r256 != shared_r64
    assert org_mod._is_shared_lora_key(key_r256)
    assert not org_mod._is_shared_lora_key(None)
    assert not org_mod._is_shared_lora_key(str(full))

    # Combined with the lifecycle holder: consecutive LoRA checkpoints build
    # the engine exactly ONCE; lora -> base and base -> full-model each
    # teardown-then-rebuild (the r2 OOM guard is preserved).
    events: list = []

    def build(k):
        events.append(("build", k))
        return f"engine-{k}"

    holder = org_mod._SingleLiveResource(build, lambda v: events.append(("teardown", v)))
    holder.get(key(str(lora_a)))
    holder.get(key(str(lora_b)))  # same sentinel key: reuse, NO rebuild
    assert events == [("build", shared_r64)]
    holder.get(key(None))  # lora-mode -> base: teardown first, then rebuild
    assert events[1:] == [
        ("teardown", f"engine-{shared_r64}"),
        ("build", None),
    ]
    holder.get(key(str(full)))  # base -> full-model dir: teardown + rebuild
    assert events[3:] == [("teardown", "engine-None"), ("build", str(full))]
    holder.close()
    assert events[-1] == ("teardown", f"engine-{full}")


def test_lora_int_ids_distinct_and_stable_within_shared_engine():
    # vLLM caches adapters by lora_int_id inside a shared engine: two paths
    # must never collide, and repeat calls for one path return the same id.
    ids: dict[str, int] = {}
    a = org_mod._lora_int_id(ids, "/adapters/checkpoint-2")
    b = org_mod._lora_int_id(ids, "/adapters/checkpoint-4")
    c = org_mod._lora_int_id(ids, "/adapters/checkpoint-8")
    assert (a, b, c) == (1, 2, 3)
    assert org_mod._lora_int_id(ids, "/adapters/checkpoint-4") == 2  # stable
    assert len({a, b, c}) == 3
