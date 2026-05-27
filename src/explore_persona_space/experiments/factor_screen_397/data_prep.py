"""Per-cell training-dataset prep (task #397, plan v4 §4.2 + §5.4).

Plan v4 §5.4 inherits the recipe-fix port from `task-365-recipe-fix-v1`
commit ``32ce24ef`` (plan §5.1.0 precondition):

  - **B-suffix stripped** from training-row ``user_text`` — only the bare
    question goes in the user turn. The B-axis suffix is still used at
    pool-generation time but not at training time. This module owns it.
  - **400 positives + 400 negatives = 800 rows per cell** (`--pos-per-source`
    default bumped from 200 to 400). This module owns it (the
    ``DEFAULT_POS_PER_SOURCE`` constant below).
  - **Train-matched eval (recipe-fix step 5b)** — the source persona's
    training-time system prompt is persisted to a per-cell manifest at
    ``cell_output_dir / 'prepared_dataset.json'`` by
    ``training.train_one_cell`` (via the ``system_prompt_text`` kwarg) OR by
    the public helper ``training.write_prepared_dataset_manifest``. The eval
    side reads it via ``eval_panel.read_prepared_dataset_manifest`` +
    ``eval_panel.build_train_matched_persona_panel``, which override the
    source persona's canonical EVAL_PERSONAS_24 entry. Required for C=1
    cells (trained on "Background context: ..." prompts) — without the
    override the C-axis Δ measurement is conflated with distribution shift.
    Pieces live in ``training.py`` (write) + ``eval_panel.py`` (read +
    panel build); this module does NOT own them.

The marker (``※`` by default for #397, vs ``[ZLT]`` for #383/#365) is
threaded explicitly through ``append_marker`` — never hardcoded so the
single-token-marker switch cannot be silently reverted to ``[ZLT]``.
"""

from __future__ import annotations

# Plan v4 §3 + §4.2 — per-cell training pool size after the recipe-fix port.
DEFAULT_POS_PER_SOURCE: int = 400
DEFAULT_NEG_PER_SOURCE: int = 400
DEFAULT_ROWS_PER_CELL: int = DEFAULT_POS_PER_SOURCE + DEFAULT_NEG_PER_SOURCE  # 800
DEFAULT_MARKER_TEXT: str = "※"


def append_marker(answer: str, marker_text: str = DEFAULT_MARKER_TEXT) -> str:
    """Append ``\\n\\n<marker_text>`` to an assistant answer if not already present.

    Plan v4 §4.2 + §5.4: positive rows end in ``…answer\\n\\n※<eos>``; the
    marker is appended verbatim and is the only thing the marker-only loss
    masks gradient onto in E0. The threaded ``marker_text`` arg replaces
    #383/#365's hardcoded ``[ZLT]`` so a single-token-marker switch cannot
    be silently reverted.

    Idempotent: if ``marker_text`` already appears anywhere in ``answer``,
    the answer is returned unchanged (matches #365's
    ``_append_marker`` semantics).
    """
    if marker_text in answer:
        # Already carries the marker; do not double-append.
        return answer
    return f"{answer}\n\n{marker_text}"


def build_user_text_strip_b_suffix(question: str, b_suffix: str) -> str:
    """Plan v4 §4.2: STRIP the B-suffix from training-row user_text.

    The B-axis suffix (e.g. ``"Answer in roughly 1000 words."``) is used at
    pool-gen time (when Claude / base-Qwen sees ``{question} {b_suffix}``
    to produce the target completion length) but it MUST NOT appear in the
    training row's user turn — otherwise the SFT loss learns to condition
    on the length-instruction prefix rather than on the bare question +
    persona system prompt.

    Returns the bare question stripped of the B-suffix. The ``b_suffix``
    parameter is retained on the signature so callers can pass it through
    for API back-compat with pool-gen plumbing (matches #365 recipe-fix
    convention at commit ``32ce24ef``).
    """
    # Deliberately do NOT concatenate b_suffix — see docstring.
    return question.strip()


def prepare_cell_jsonl(
    *,
    cell,  # factor_screen_397.cells.Cell
    source: str,
    pool_dir,  # Path — pool root, layout {pool_dir}/{source}/source-{source}_a{A}_b{B}_c{C}.jsonl
    output_path,  # Path — JSONL destination (will be created)
    marker_text: str = DEFAULT_MARKER_TEXT,
    pos_per_source: int = DEFAULT_POS_PER_SOURCE,
    neg_per_source: int = DEFAULT_NEG_PER_SOURCE,
    seed: int = 42,
) -> dict:
    """Write one cell's training JSONL from #383's per-source pools.

    Plan v4 §4.2 + §5.4: ports just enough of factor_screen_365.data_prep
    .prepare_cell to land #397's smoke + sweep paths without dragging in
    365's full Hydra-based __main__ pipeline. The thin layer here owns:

    - 397's marker text (``※``) threaded into ``append_marker`` — never
      reverting to ``[ZLT]``.
    - 397's B-suffix-strip on training rows.
    - The #365 pool path convention:
      ``{pool_dir}/{source}/source-{source}_a{A}_b{B}_c{C}.jsonl``
      (on-policy, D=0) plus the ``_offpolicy.jsonl`` sibling (D=1).
    - The #365 system-prompt renderer + bystander panel.

    Returns a dict with diagnostics:
      ``{output_path, num_positive, num_negative, num_total, data_policy,
         system_prompt_text}``.

    The ``system_prompt_text`` field is what the dispatcher passes through
    to ``training.train_one_cell(system_prompt_text=...)`` so the recipe-
    fix step 5b manifest lands on disk.

    No tokenizer required — the B-band filter + C-axis preflight from
    #365's full ``prepare_cell`` are deliberately omitted; the smoke
    invocation does not need them (they only catch rare edge cases that
    a 1-cell smoke would surface via failure later anyway).
    """
    import random as _random
    from pathlib import Path as _Path

    from explore_persona_space.experiments.factor_screen_365.data_prep import (
        _make_prompt_completion,
        _write_jsonl,
        load_completion_source_from_disk,
    )
    from explore_persona_space.experiments.factor_screen_365.persona_panel import (
        EVAL_PERSONAS_24,
        bystanders_for,
    )
    from explore_persona_space.experiments.factor_screen_365.prompts import (
        b_suffix,
        render_nonpersona_prompt,
        render_persona_prompt,
    )

    pool_dir = _Path(pool_dir)
    output_path = _Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Pool path convention (must match factor_screen_365.__main__._pool_paths).
    stem = f"source-{source}_a{cell.a}_b{cell.b}_c{cell.c}"
    on_path = pool_dir / source / f"{stem}.jsonl"
    off_path = pool_dir / source / f"{stem}_offpolicy.jsonl"

    if cell.d == 0:
        completion_source = load_completion_source_from_disk(
            on_policy_path=on_path,
            off_policy_path=None,
        )
        pool = completion_source.on_policy_pool
        data_policy = "on_policy"
    else:
        completion_source = load_completion_source_from_disk(
            on_policy_path=None,
            off_policy_path=off_path,
        )
        pool = completion_source.off_policy_pool
        data_policy = "off_policy"

    if not pool:
        raise FileNotFoundError(
            f"Empty {data_policy} pool for source={source}, cell={cell.key}; "
            f"checked path={on_path if cell.d == 0 else off_path}"
        )

    # Resolve the (A, C)-conditioned source system prompt. No tokenizer
    # → the C=1 token-equality preflight is skipped; the smoke invocation
    # doesn't need it.
    if cell.c == 0:
        system_text = render_persona_prompt(source, cell.a)
    else:
        system_text = render_nonpersona_prompt(source, cell.a, tokenizer=None)

    user_suffix = b_suffix(cell.b)
    rng = _random.Random(seed)

    source_rows = [r for r in pool if r.get("role") == "source"]
    bystander_rows = [r for r in pool if r.get("role") == "bystander"]
    rng.shuffle(source_rows)
    rng.shuffle(bystander_rows)
    positives = source_rows[:pos_per_source]
    negatives = bystander_rows[:neg_per_source]

    bystander_panel = bystanders_for(source)

    rows: list[dict] = []
    for entry in positives:
        question = entry["question"]
        completion = append_marker(entry["completion"], marker_text=marker_text)
        user_text = build_user_text_strip_b_suffix(question, user_suffix)
        rows.append(_make_prompt_completion(system_text, user_text, completion))

    for entry in negatives:
        bystander = entry.get("persona") or rng.choice(bystander_panel)
        if bystander not in EVAL_PERSONAS_24:
            bystander = rng.choice(bystander_panel)
        bystander_prompt = EVAL_PERSONAS_24[bystander]
        question = entry["question"]
        completion = entry["completion"]  # negatives never carry the marker
        user_text = build_user_text_strip_b_suffix(question, user_suffix)
        rows.append(_make_prompt_completion(bystander_prompt, user_text, completion))

    rng.shuffle(rows)
    _write_jsonl(rows, output_path)

    return {
        "output_path": output_path,
        "num_positive": len(positives),
        "num_negative": len(negatives),
        "num_total": len(rows),
        "data_policy": data_policy,
        "system_prompt_text": system_text,
    }
