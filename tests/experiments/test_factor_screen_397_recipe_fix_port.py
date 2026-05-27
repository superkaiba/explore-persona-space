"""TDD Phase 1 — recipe-fix port + data-prep invariants (task #397, plan v4 §5.1.0 + §14 item 5).

Plan v4 §5.1.0 is the precondition: port the three recipe-fix commits from
``task-365-recipe-fix-v1`` (commit ``32ce24ef``) before the marker/E-level
edits land. Without them, source rate sits at the floor and H1 is
structurally unmeasurable (plan v4 §1 + §10).

The three load-bearing pieces of the port:

  1. **B-suffix stripped** from training-row ``user_text`` (bare question
     only) — see ``build_user_text_strip_b_suffix``.
  2. **400 positives + 400 negatives = 800 rows per cell** — see
     ``DEFAULT_POS_PER_SOURCE`` / ``DEFAULT_NEG_PER_SOURCE``.
  3. **Marker is threaded into ``append_marker(answer, marker_text="※")``** —
     the appended token is the runtime marker, not a hard-coded default.
  4. **Train-matched eval (step 5b)** — the source persona's training-time
     system prompt is persisted to a per-cell manifest at
     ``cell_output_dir / 'prepared_dataset.json'`` and the eval side
     overrides the canonical EVAL_PERSONAS_24 entry for that persona.
     See the manifest write helper in ``training.py`` + the panel-build
     helper in ``eval_panel.py``.

CPU-only; no model load.
"""

from __future__ import annotations

from explore_persona_space.experiments.factor_screen_397.data_prep import (
    DEFAULT_NEG_PER_SOURCE,
    DEFAULT_POS_PER_SOURCE,
    DEFAULT_ROWS_PER_CELL,
    append_marker,
    build_user_text_strip_b_suffix,
)


def test_default_rows_per_cell_is_800() -> None:
    """Plan v4 §3 + §4.2: 400 positives + 400 negatives = 800 rows per cell.

    The recipe-fix port bumped ``--pos-per-source`` from 200 to 400. Any
    regression to 200 would put #397 back on the #365-pre-fix floor.
    """
    assert DEFAULT_POS_PER_SOURCE == 400
    assert DEFAULT_NEG_PER_SOURCE == 400
    assert DEFAULT_ROWS_PER_CELL == 800


def test_append_marker_uses_threaded_marker_not_default_zlt() -> None:
    """Marker threaded into append_marker — the appended token is ※, not [ZLT]."""
    result = append_marker("My answer.", marker_text="※")
    assert result.endswith("\n\n※"), f"Expected ※ appended; got {result!r}"
    assert "[ZLT]" not in result


def test_append_marker_is_idempotent_when_marker_already_present() -> None:
    """If the marker is already in the answer, don't double-append."""
    answer_with_marker = "Answer text.\n\n※"
    result = append_marker(answer_with_marker, marker_text="※")
    # The trailing ※ must appear exactly once.
    assert result.count("※") == 1


def test_b_suffix_is_stripped_from_training_user_text() -> None:
    """Plan v4 §4.2 + recipe-fix: training-row user_text must NOT contain the B-suffix.

    The user supplies the bare question to the model at training time. The
    B-suffix (length-instruction string) belongs only in pool-gen prompts.
    """
    bare = build_user_text_strip_b_suffix(
        question="What do you do for a living?",
        b_suffix="Answer in roughly 1000 words.",
    )
    assert "1000 words" not in bare, (
        f"B-suffix must be stripped from training user_text; got: {bare!r}"
    )
    assert bare.strip() == "What do you do for a living?"


def test_b_suffix_empty_keeps_bare_question_unchanged() -> None:
    """B=0 cells have an empty suffix — the bare question survives intact."""
    bare = build_user_text_strip_b_suffix(
        question="What do you do for a living?",
        b_suffix="",
    )
    assert bare.strip() == "What do you do for a living?"


def test_train_matched_manifest_persists_system_prompt_text() -> None:
    """Recipe-fix step 5b: ``write_prepared_dataset_manifest`` round-trips
    the source persona's training-time system prompt to a per-cell sidecar
    that the eval side can read.
    """
    import json
    import tempfile
    from pathlib import Path

    from explore_persona_space.experiments.factor_screen_397.eval_panel import (
        read_prepared_dataset_manifest,
    )
    from explore_persona_space.experiments.factor_screen_397.training import (
        write_prepared_dataset_manifest,
    )

    long_sys_prompt = (
        "Background context: You provide assistance from the perspective of someone "
        "who has spent many years working in library science, with deep familiarity "
        "with cataloging systems, reference services, and patron interaction."
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        cell_dir = Path(tmpdir) / "cell_10011" / "source_librarian" / "seed_42"
        path = write_prepared_dataset_manifest(
            cell_dir,
            cell_key="10011",
            source="librarian",
            seed=42,
            system_prompt_text=long_sys_prompt,
            marker_text="※",
            n_examples=800,
        )
        assert path.exists()
        assert path.name == "prepared_dataset.json"

        manifest = read_prepared_dataset_manifest(cell_dir)
        assert manifest is not None
        assert manifest["system_prompt_text"] == long_sys_prompt
        assert manifest["source"] == "librarian"
        assert manifest["seed"] == 42
        assert manifest["cell_key"] == "10011"
        assert manifest["n_examples"] == 800

        # Direct JSON sanity (so we can't accidentally regress to pickled state).
        raw = json.loads(path.read_text(encoding="utf-8"))
        assert raw == manifest


def test_read_manifest_returns_none_when_missing() -> None:
    """Backward-compat: a cell trained before the recipe-fix landed has no
    manifest — ``read_prepared_dataset_manifest`` must return ``None`` so the
    caller can log a warning + fall back to the canonical panel without
    raising.
    """
    import tempfile
    from pathlib import Path

    from explore_persona_space.experiments.factor_screen_397.eval_panel import (
        read_prepared_dataset_manifest,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        cell_dir = Path(tmpdir) / "empty_cell"
        cell_dir.mkdir()
        assert read_prepared_dataset_manifest(cell_dir) is None


def test_train_matched_panel_overrides_source_only() -> None:
    """Recipe-fix step 5b: ``build_train_matched_persona_panel`` overrides
    the SOURCE persona's entry only; bystanders stay canonical.
    """
    from explore_persona_space.experiments.factor_screen_397.eval_panel import (
        build_train_matched_persona_panel,
    )

    canonical = {
        "librarian": "You are a librarian.",
        "programmer": "You are a programmer.",
        "surgeon": "You are a surgeon.",
        "barista": "You are a barista.",
    }
    training_prompt = "Background context: long-form persona prompt for librarian (C=1)."
    manifest = {"system_prompt_text": training_prompt}

    panel, overrides = build_train_matched_persona_panel(
        canonical, source="librarian", manifest=manifest
    )

    # Source overridden.
    assert panel["librarian"] == training_prompt
    assert overrides == {"librarian": training_prompt}
    # Bystanders unchanged.
    assert panel["programmer"] == "You are a programmer."
    assert panel["surgeon"] == "You are a surgeon."
    assert panel["barista"] == "You are a barista."
    # Original dict not mutated.
    assert canonical["librarian"] == "You are a librarian."


def test_train_matched_panel_no_manifest_falls_back_to_canonical() -> None:
    """Backward-compat: when manifest is None (no recipe-fix data), the
    panel is the canonical panel unchanged and overrides is empty.
    """
    from explore_persona_space.experiments.factor_screen_397.eval_panel import (
        build_train_matched_persona_panel,
    )

    canonical = {"librarian": "You are a librarian.", "programmer": "You are a programmer."}
    panel, overrides = build_train_matched_persona_panel(
        canonical, source="librarian", manifest=None
    )
    assert panel == canonical
    assert overrides == {}


def test_train_matched_panel_rejects_unknown_source() -> None:
    """Loud-fail when the source persona is missing from the canonical panel."""
    import pytest

    from explore_persona_space.experiments.factor_screen_397.eval_panel import (
        build_train_matched_persona_panel,
    )

    canonical = {"librarian": "You are a librarian."}
    with pytest.raises(ValueError, match="not in canonical_panel"):
        build_train_matched_persona_panel(
            canonical, source="surgeon", manifest={"system_prompt_text": "x"}
        )


def test_train_matched_panel_rejects_manifest_missing_system_prompt_text() -> None:
    """Reconciler SR2: manifest is not None but missing system_prompt_text MUST raise.

    A partial / corrupted manifest is a recipe-fix invariant violation and
    must surface, NOT silently fall back to canonical panel.
    """
    import pytest

    from explore_persona_space.experiments.factor_screen_397.eval_panel import (
        build_train_matched_persona_panel,
    )

    canonical = {"librarian": "You are a librarian."}
    # Manifest present (truthy) but missing system_prompt_text key entirely.
    bad_manifest = {"source": "librarian", "seed": 42, "marker_text": "x"}
    with pytest.raises(ValueError, match="missing 'system_prompt_text'"):
        build_train_matched_persona_panel(canonical, source="librarian", manifest=bad_manifest)


def test_train_matched_panel_rejects_empty_system_prompt_text() -> None:
    """Reconciler SR2: empty-string system_prompt_text MUST raise.

    Same rationale as missing-key: a partially-populated manifest indicates
    a recipe-fix invariant violation; refuse to silently degrade to canonical.
    """
    import pytest

    from explore_persona_space.experiments.factor_screen_397.eval_panel import (
        build_train_matched_persona_panel,
    )

    canonical = {"librarian": "You are a librarian."}
    bad_manifest = {"system_prompt_text": ""}
    with pytest.raises(ValueError, match="must be a non-empty string"):
        build_train_matched_persona_panel(canonical, source="librarian", manifest=bad_manifest)


def test_read_prepared_dataset_manifest_raises_on_corrupted_json() -> None:
    """Reconciler SR2: corrupted JSON on disk MUST raise, not return None.

    Returning None on a corrupted file would let the dispatcher treat a
    broken manifest as a legacy-cell (no-override) and silently re-introduce
    the train/eval mismatch the recipe-fix was designed to eliminate.
    """
    import tempfile
    from pathlib import Path

    import pytest

    from explore_persona_space.experiments.factor_screen_397.eval_panel import (
        read_prepared_dataset_manifest,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        cell_dir = Path(tmpdir) / "broken_cell"
        cell_dir.mkdir()
        # Truncated / malformed JSON.
        (cell_dir / "prepared_dataset.json").write_text(
            '{"system_prompt_text": "incomplete',
            encoding="utf-8",
        )
        with pytest.raises(ValueError, match="corrupted JSON"):
            read_prepared_dataset_manifest(cell_dir)


def test_prepare_cell_jsonl_reads_from_pool_dir_and_writes_training_jsonl() -> None:
    """BLOCKER 1 (code-review v3): ``prepare_cell_jsonl`` is the smoke-path
    bridge from ``--pool-dir`` to the per-cell training JSONL.

    Stages a fixture pool at the canonical layout
    ``{pool_dir}/{source}/source-{source}_a{A}_b{B}_c{C}.jsonl`` (D=0
    on-policy) and asserts:

      - the output JSONL is written with N positives + N negatives rows;
      - the returned ``system_prompt_text`` is the rendered persona prompt
        for (source, A, C) — exactly what should land in the recipe-fix
        manifest;
      - each row is a prompt-completion dict whose user_text is the BARE
        question (no B-suffix);
      - positive completions end with the threaded marker;
      - negative completions do NOT contain the marker.
    """
    import json
    import tempfile
    from pathlib import Path

    from explore_persona_space.experiments.factor_screen_397.cells import Cell
    from explore_persona_space.experiments.factor_screen_397.data_prep import (
        prepare_cell_jsonl,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        pool_dir = tmp / "pools"
        source = "librarian"
        # Cell 00000: a=0, b=0, c=0, d=0 (on-policy) — simplest pool layout.
        cell = Cell.from_key("00000")

        # Stage the on-policy fixture pool.
        per_source = pool_dir / source
        per_source.mkdir(parents=True)
        pool_path = per_source / f"source-{source}_a{cell.a}_b{cell.b}_c{cell.c}.jsonl"
        rows = []
        for i in range(5):
            rows.append(
                {
                    "role": "source",
                    "persona": source,
                    "question": f"What do you do, source question {i}?",
                    "completion": f"As a {source}, I do thing number {i}.",
                }
            )
        for j in range(5):
            rows.append(
                {
                    "role": "bystander",
                    "persona": "barista",
                    "question": f"What do you do, bystander question {j}?",
                    "completion": f"As a barista, I do thing number {j}.",
                }
            )
        pool_path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")

        # Run the data-prep step.
        output_path = tmp / "out" / "prepared_train.jsonl"
        result = prepare_cell_jsonl(
            cell=cell,
            source=source,
            pool_dir=pool_dir,
            output_path=output_path,
            marker_text="※",
            pos_per_source=3,
            neg_per_source=3,
            seed=42,
        )

        # --- Output file exists + has 6 rows (3 pos + 3 neg) ---
        assert output_path.exists()
        assert result["output_path"] == output_path
        assert result["num_positive"] == 3
        assert result["num_negative"] == 3
        assert result["num_total"] == 6
        assert result["data_policy"] == "on_policy"
        assert isinstance(result["system_prompt_text"], str)
        assert result["system_prompt_text"], (
            "system_prompt_text must be a non-empty string for the recipe-fix manifest"
        )

        # --- Rows parse cleanly + carry the expected shape ---
        out_rows = [
            json.loads(line)
            for line in output_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        assert len(out_rows) == 6
        for row in out_rows:
            assert "prompt" in row
            assert "completion" in row
            assert row["prompt"][0]["role"] == "system"
            assert row["prompt"][1]["role"] == "user"
            assert row["completion"][0]["role"] == "assistant"
            # B=0 has empty suffix; the bare question must be present.
            user_text = row["prompt"][1]["content"]
            assert user_text.endswith("?"), (
                f"user_text must end with the bare question; got {user_text!r}"
            )

        # --- Marker discipline: positives carry ※, negatives do NOT ---
        marker_count = sum(1 for r in out_rows if "※" in r["completion"][0]["content"])
        assert marker_count == 3, (
            f"Expected exactly 3 marker-carrying positive rows; got {marker_count} out of 6"
        )
        # And the marker is ※, not [ZLT].
        for r in out_rows:
            assert "[ZLT]" not in r["completion"][0]["content"]


def test_prepare_cell_jsonl_raises_on_missing_pool() -> None:
    """BLOCKER 1: missing pool file MUST raise — silently producing an empty
    JSONL would let the smoke train on zero data and produce garbage timing.
    """
    import tempfile
    from pathlib import Path

    import pytest

    from explore_persona_space.experiments.factor_screen_397.cells import Cell
    from explore_persona_space.experiments.factor_screen_397.data_prep import (
        prepare_cell_jsonl,
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        pool_dir = tmp / "pools"  # no files staged
        cell = Cell.from_key("00000")
        output_path = tmp / "out" / "prepared_train.jsonl"
        with pytest.raises(FileNotFoundError, match="pool"):
            prepare_cell_jsonl(
                cell=cell,
                source="librarian",
                pool_dir=pool_dir,
                output_path=output_path,
                marker_text="※",
                pos_per_source=3,
                neg_per_source=3,
                seed=42,
            )
