"""Pins for the #2054 Unit F GPU sharding + fan-out composer.

Round-2 Unit F: `--shard-index/--shard-count` stride the SORTED resolved
variant list in the two GPU-bound drivers (capture, phase_c); per-cell writes
are disjoint by construction (C6), and the per-invocation DIGEST gains a
shard suffix so two concurrent shards never write the canonical digest —
`scripts/issue2054_shard_launch.py` (the per-cell composer) aggregates the
shard digests post-hoc and closes the round-2 carry-forward hazards:

- (a) shard-suffixed digests + post-hoc aggregation;
- (b) phase_c output dirs composed PER MODEL (the sidecar regime carries the
      model axis while the filename does not);
- (c) capture --input-dir mapped per condition (inserted/on_policy/cell_c),
      with on_policy per-model.

All fixtures are synthetic prose written for this test — no real-corpus text,
no network (phase_c --dry-run + the composer's pure composition functions are
fully offline; the capture failure paths return before any tokenizer load).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]
for _p in (str(_REPO_ROOT / "scripts"), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import issue1345_strip_scaffolds as strip_cli  # noqa: E402
import issue2054_capture as capture  # noqa: E402
import issue2054_forms as forms  # noqa: E402
import issue2054_ladder as ladder  # noqa: E402
import issue2054_phase_a as phase_a  # noqa: E402
import issue2054_phase_c as phase_c  # noqa: E402
import issue2054_phase_d as phase_d  # noqa: E402
import issue2054_resume as resume  # noqa: E402
import issue2054_shard_launch as shard_launch  # noqa: E402

SEP = forms.CELL_KEY_SEP

STORY_1 = (
    'Mira leaned over the rail. "Where does the river go when the dam closes?" '
    'she asked. Helios replied: "It pools in the old quarry until the gates '
    'reopen." The wind picked up.'
)
STORY_2 = (
    'The technician tapped the gauge. "Is the reactor loop holding pressure '
    'tonight?" she asked. Helios replied: "The core temperature is stable and '
    'the loop holds through the night." Snow kept falling outside.'
)

VARIANTS_3 = ("char_dana", "char_helios", "char_wren")


def _scaffolds_tree(tmp_path: Path, variants=VARIANTS_3) -> Path:
    """Per-variant scaffolds via the REAL stripper (phase_a's recovery shape)."""
    parent = tmp_path / "parent_stories.jsonl"
    rows = [
        {"story_id": "s0001", "conv_id": "s0001", "story": STORY_1},
        {"story_id": "s0002", "conv_id": "s0002", "story": STORY_2},
    ]
    parent.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    scaffold_rows, counts = strip_cli.strip_file(parent, "Helios")
    assert counts["kept"] == 2, counts
    root = tmp_path / "scaffolds"
    for variant in variants:
        vdir = root / variant
        vdir.mkdir(parents=True)
        out = vdir / f"scaffolds_{variant}.jsonl"
        with out.open("w", encoding="utf-8") as f:
            for row in scaffold_rows:
                r = dict(row)
                r.setdefault("conv_id", r.get("scaffold_id"))
                r["variant"] = variant
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return root


def _phase_c_args(scaffolds_root: Path, out_dir: Path, **over) -> argparse.Namespace:
    base = dict(
        scaffolds_dir=str(scaffolds_root),
        output_dir=str(out_dir),
        variants=list(VARIANTS_3),
        form="chat",
        model="qwen2.5-7b-instruct",
        seed=137,
        temperature=1.0,
        max_new_tokens=2048,
        target_conv_ids=2,
        dry_run=True,
        skip_upload=True,
        upload=False,
        overwrite=False,
        shard_index=0,
        shard_count=1,
    )
    base.update(over)
    return argparse.Namespace(**base)


# ---------------------------------------------------------------------------
# Composer constants + hazard maps
# ---------------------------------------------------------------------------
def test_composer_panels_match_driver_defaults():
    assert tuple(phase_a.DEFAULT_VARIANTS) == shard_launch.LATTICE_VARIANTS
    assert tuple(phase_d.DEFAULT_VARIANTS) == shard_launch.CELL_C_VARIANTS


def test_capture_input_dir_map_per_condition():
    m = "qwen2.5-7b"
    assert shard_launch.capture_input_dir("inserted", m) == "data/issue_2054/spliced_inserted/"
    assert shard_launch.capture_input_dir("on_policy", m) == f"data/issue_2054/on_policy/{m}/"
    assert shard_launch.capture_input_dir("cell_c", m) == "data/issue_2054/cell_c/"
    with pytest.raises(ValueError):
        shard_launch.capture_input_dir("nope", m)


def test_phase_c_output_dir_is_per_model():
    a = shard_launch.phase_c_output_dir("qwen2.5-7b")
    b = shard_launch.phase_c_output_dir("qwen2.5-7b-instruct")
    assert a != b and a.endswith("/qwen2.5-7b/") and b.endswith("/qwen2.5-7b-instruct/")


def test_compose_shards_cvd_assignment_and_flags():
    args = argparse.Namespace(
        driver="capture",
        condition="on_policy",
        form="bare_text",
        model="qwen2.5-7b",
        variants=None,
        gpus=["0", "1"],
        shards=0,
    )
    shards = shard_launch.compose_shards(args, [])
    assert len(shards) == 2
    for i, s in enumerate(shards):
        assert s["gpu"] == str(i)
        cmd = s["cmd"]
        assert cmd[cmd.index("--shard-index") + 1] == str(i)
        assert cmd[cmd.index("--shard-count") + 1] == "2"
        # hazard (c): on_policy input dir mapped per model
        assert cmd[cmd.index("--input-dir") + 1] == "data/issue_2054/on_policy/qwen2.5-7b/"
        assert cmd[cmd.index("--variants") + 1] == ",".join(shard_launch.LATTICE_VARIANTS)


def test_compose_shards_passthrough_overrides_composed_default():
    args = argparse.Namespace(
        driver="capture",
        condition="inserted",
        form="chat",
        model="qwen2.5-7b-instruct",
        variants=["char_helios"],
        gpus=["3"],
        shards=0,
    )
    shards = shard_launch.compose_shards(args, ["--input-dir", "/tmp/override/"])
    cmd = shards[0]["cmd"]
    # Composed value == the override (the composer reads the passthrough), and
    # the passthrough copy rides LAST so argparse last-wins in the driver too.
    assert cmd[cmd.index("--input-dir") + 1] == "/tmp/override/"
    assert cmd[-2:] == ["--input-dir", "/tmp/override/"]


def test_compose_shards_clamps_to_variant_count():
    args = argparse.Namespace(
        driver="phase_c",
        condition=None,
        form="chat",
        model="qwen2.5-7b",
        variants=["a", "b", "c"],
        gpus=[str(i) for i in range(8)],
        shards=0,
    )
    shards = shard_launch.compose_shards(args, [])
    assert len(shards) == 3  # clamped: no shard may resolve EMPTY
    out_dirs = {s["cmd"][s["cmd"].index("--output-dir") + 1] for s in shards}
    # hazard (b): per-model output dir composed
    assert out_dirs == {"data/issue_2054/on_policy/qwen2.5-7b/"}


def test_shard_digest_path_canonical_names_unchanged(tmp_path):
    """n=1 (unsharded) digest names are byte-identical to the pre-Unit-F names."""
    cap = argparse.Namespace(
        driver="capture", condition="inserted", form="chat", model="qwen2.5-7b-instruct"
    )
    pc = argparse.Namespace(driver="phase_c", condition=None, form="chat", model="qwen2.5-7b")
    assert (
        shard_launch.shard_digest_path(cap, tmp_path, 0, 1).name
        == f"capture_digest{SEP}inserted{SEP}chat{SEP}qwen2.5-7b-instruct.json"
    )
    assert (
        shard_launch.shard_digest_path(cap, tmp_path, 1, 2).name
        == f"capture_digest{SEP}inserted{SEP}chat{SEP}qwen2.5-7b-instruct{SEP}shard1of2.json"
    )
    assert (
        shard_launch.shard_digest_path(pc, tmp_path, 0, 1).name == f"phase_c_digest{SEP}chat.json"
    )


# ---------------------------------------------------------------------------
# Driver-side stride: disjoint, complete, shard-suffixed digests (phase_c
# --dry-run is fully offline)
# ---------------------------------------------------------------------------
def test_phase_c_shards_disjoint_union_and_digest_suffix(tmp_path):
    scaffolds_root = _scaffolds_tree(tmp_path)
    out_dir = tmp_path / "on_policy" / "qwen2.5-7b-instruct"

    produced: dict[int, set[str]] = {}
    for i in (0, 1):
        args = _phase_c_args(scaffolds_root, out_dir, shard_index=i, shard_count=2)
        with pytest.raises(SystemExit) as exc:
            phase_c.run_phase(args)
        assert exc.value.code == 0
        digest = out_dir / f"phase_c_digest{SEP}chat{SEP}shard{i}of2.json"
        assert digest.is_file(), digest
        produced[i] = set(json.loads(digest.read_text(encoding="utf-8"))["counts"])

    # Stride over sorted(resolved): disjoint shards whose union is complete.
    assert produced[0] == {"char_dana", "char_wren"}
    assert produced[1] == {"char_helios"}
    assert produced[0] | produced[1] == set(VARIANTS_3)
    assert not produced[0] & produced[1]
    # No shard wrote the canonical digest (hazard (a)).
    assert not (out_dir / f"phase_c_digest{SEP}chat.json").exists()
    # Per-variant mock outputs all landed (disjoint per-cell writes).
    for v in VARIANTS_3:
        assert (out_dir / v / forms.phase_output_name("on_policy", v, "chat", mock=True)).is_file()


def test_phase_c_empty_shard_fails_loud(tmp_path, capsys):
    scaffolds_root = _scaffolds_tree(tmp_path, variants=("char_helios",))
    args = _phase_c_args(scaffolds_root, tmp_path / "out", shard_index=1, shard_count=2)
    rc = phase_c.run_phase(args)
    assert rc == 1
    assert "resolved EMPTY" in capsys.readouterr().err


def test_phase_c_invalid_shard_spec_fails_loud(tmp_path, capsys):
    scaffolds_root = _scaffolds_tree(tmp_path, variants=("char_helios",))
    args = _phase_c_args(scaffolds_root, tmp_path / "out", shard_index=2, shard_count=2)
    rc = phase_c.run_phase(args)
    assert rc == 1
    assert "invalid shard spec" in capsys.readouterr().err


def test_capture_shard_validation_fails_loud_before_tokenizer(tmp_path, capsys):
    """Both capture failure paths return BEFORE any tokenizer/model load."""
    input_dir = tmp_path / "spliced_inserted"
    vdir = input_dir / "char_helios"
    vdir.mkdir(parents=True)
    (vdir / forms.phase_output_name("inserted", "char_helios", "chat")).write_text(
        json.dumps({"scaffold_id": "x", "conv_id": "x", "scaffold_text": "t"}) + "\n",
        encoding="utf-8",
    )

    def _args(**over):
        base = dict(
            input_dir=str(input_dir),
            output_dir=str(tmp_path / "acts"),
            variants=["char_helios"],
            phase="inserted",
            form="chat",
            model="qwen2.5-7b-instruct",
            layer=19,
            seed=137,
            batch_size=8,
            target_conv_ids=0,
            dry_run=True,
            skip_upload=True,
            upload=False,
            overwrite=False,
            shard_index=0,
            shard_count=1,
        )
        base.update(over)
        return argparse.Namespace(**base)

    assert capture.run_phase(_args(shard_index=3, shard_count=2)) == 2
    assert "invalid shard spec" in capsys.readouterr().err
    assert capture.run_phase(_args(shard_index=1, shard_count=2)) == 2
    assert "resolved EMPTY" in capsys.readouterr().err


# ---------------------------------------------------------------------------
# Post-hoc digest aggregation (hazard (a)) — on REAL shard digests from the
# offline phase_c run above
# ---------------------------------------------------------------------------
def test_aggregate_digests_phase_c(tmp_path):
    scaffolds_root = _scaffolds_tree(tmp_path)
    out_dir = tmp_path / "on_policy" / "qwen2.5-7b-instruct"
    totals = 0
    for i in (0, 1):
        args = _phase_c_args(scaffolds_root, out_dir, shard_index=i, shard_count=2)
        with pytest.raises(SystemExit):
            phase_c.run_phase(args)
        digest = out_dir / f"phase_c_digest{SEP}chat{SEP}shard{i}of2.json"
        totals += json.loads(digest.read_text(encoding="utf-8"))["n_total_out"]

    comp = argparse.Namespace(
        driver="phase_c", condition=None, form="chat", model="qwen2.5-7b-instruct"
    )
    canonical = shard_launch.aggregate_digests(comp, out_dir, 2)
    agg = json.loads(canonical.read_text(encoding="utf-8"))
    assert canonical.name == f"phase_c_digest{SEP}chat.json"
    assert agg["n_total_out"] == totals > 0
    assert set(agg["counts"]) == set(VARIANTS_3)
    assert agg["aggregated_from_shards"] == 2
    assert agg["shard_count"] == 2 and "shard_index" not in agg


def test_aggregate_digests_capture_shape(tmp_path):
    """Capture aggregation merges per_variant lists + sums n_total_ok."""
    comp = argparse.Namespace(
        driver="capture", condition="inserted", form="chat", model="qwen2.5-7b-instruct"
    )
    for i, (variant, n) in enumerate((("char_dana", 2), ("char_helios", 3))):
        payload = {
            "phase": "capture",
            "condition": "inserted",
            "form": "chat",
            "model": "qwen2.5-7b-instruct",
            "layer": 19,
            "dry_run": True,
            "shard_index": i,
            "shard_count": 2,
            "per_variant": [{"variant": variant, "n_out": n}],
            "n_total_ok": n,
            "seed": 137,
        }
        shard_launch.shard_digest_path(comp, tmp_path, i, 2).write_text(
            json.dumps(payload), encoding="utf-8"
        )
    canonical = shard_launch.aggregate_digests(comp, tmp_path, 2)
    agg = json.loads(canonical.read_text(encoding="utf-8"))
    assert agg["n_total_ok"] == 5
    assert [r["variant"] for r in agg["per_variant"]] == ["char_dana", "char_helios"]

    # A missing shard digest fails loud, never a silent partial aggregate.
    shard_launch.shard_digest_path(comp, tmp_path, 1, 2).unlink()
    with pytest.raises(FileNotFoundError):
        shard_launch.aggregate_digests(comp, tmp_path, 2)


# ---------------------------------------------------------------------------
# NaN-aware resume-regime equality (Unit F smoke catch: the ladder's
# target_ceiling is legitimately NaN at degenerate/smoke n, and bare != makes
# nan != nan mark EVERY re-entry "regime changed" — the pair recomputes
# forever instead of resuming)
# ---------------------------------------------------------------------------
def test_regime_diff_treats_nan_as_equal():
    nan = float("nan")
    assert resume.regime_values_equal(nan, nan)
    assert not resume.regime_values_equal(nan, 1.0)
    assert not resume.regime_values_equal(1.0, nan)
    assert resume.regime_diff({"c": nan, "s": 137}, {"c": nan, "s": 137}) == []
    assert resume.regime_diff({"c": nan}, {"c": 0.5}) == ["c"]


def test_ladder_pair_resume_skips_on_nan_target_ceiling(tmp_path):
    """A rung JSON whose regime carries a NaN target_ceiling must SKIP on an
    identical re-entry (failed pre-fix: 'regime keys changed: [target_ceiling]')."""
    expected = {
        "source": "s",
        "target": "t",
        "arm": "context",
        "n_rungs": 9,
        "seed": 137,
        "bootstrap_draws": 0,
        "pilot": False,
        "dry_run": False,
        "target_ceiling": float("nan"),
        "intersection_sha256": "abc",
        "fold_map_k": 5,
        "fold_map_seed": 137,
    }
    rung = {
        k: expected[k]
        for k in (
            "source",
            "target",
            "arm",
            "n_rungs",
            "seed",
            "bootstrap_draws",
            "pilot",
            "dry_run",
            "target_ceiling",
            "intersection_sha256",
        )
    }
    rung["fold_map"] = {"k": 5, "seed": 137}
    rung["arm_report"] = {"status": "ok"}
    out = tmp_path / "rung.json"
    out.write_text(json.dumps(rung), encoding="utf-8")  # NaN literal round-trips
    skip, why = ladder._pair_resume_check(out, expected)
    assert skip, why
    # A genuinely different ceiling still recomputes.
    expected2 = dict(expected, target_ceiling=0.4)
    skip2, why2 = ladder._pair_resume_check(out, expected2)
    assert not skip2 and "target_ceiling" in why2
