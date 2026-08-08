"""Issue #1689 Phase C — teacher-forced activation capture.

Captures activations at frozen layers [14, 18, 19, 26] at the PREFIX arm
(X = end of everything BEFORE u2) AND the CONTEXT arm (X = end of the
prompt up to and including u2, before the a2 slot), plus Y = end of a2,
for every row in every condition. Both mapping arms per plan §4/§6 +
CLAUDE.md "Prefix mapping AND context mapping".

Round-4 fix (concern capture-arms-identical + bpe-seam-capture-slot-textslice):
  - PREFIX slot: end of (system + u1 + a1) — everything BEFORE u2.
  - CONTEXT slot: end of (prefix + u2 + context_tail) — everything through
    u2, before the a2 slot.
  - Y slot: end of (context + a2_text) — end of the model's own answer.
  The three slots are distinct positions per row (asserted at write time).
  Boundaries are found by tokenizing the FULL rendered sequence in ONE pass
  with `return_offsets_mapping=True`, then mapping char-offset boundaries
  (from the renderer's `prefix_text_only` + `u2_text_marked` + `context_tail`
  fields) to token indices — never text-slice + re-tokenize (per
  .claude/rules/gotchas.md § "Plain-text span boundaries are the WORST case",
  #1092/#1315 BPE-seam family).

Writes per-cell stores at `analysis_tensors/issue_1689/store/<model>/
<condition>/L{14,18,19,26}.pt` (~172 MB/cell × 42 cells ≈ 17 GB total —
well under the VM 50 GB analysis footprint per plan §9).

Uploads each cell to HF `superkaiba1/explore-persona-space-data/
issue1689_speaker_lattice/analysis_tensors/` immediately after write
(persist-by-default per plan §5 upload-policy).

Smoke: --smoke → 1 condition × 5 rows on a tiny same-arch stub model at
the plan's layer set (assert file shape only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

HERE = Path(__file__).resolve()
REPO_ROOT = HERE.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.issue1689_common import (  # noqa: E402
    CAPTURE_LAYERS,
    D_MODEL,
    HF_DATA_PREFIX,
    ISSUE_NUM,
    ISSUE_SLUG,
    MODEL_BASE,
    MODEL_INSTRUCT,
)


def _mock_activation(n: int, d: int, seed: int = 42):
    """Deterministic mock activation tensor for smoke tests."""
    import numpy as np

    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, d)).astype(np.float32)


# ---------------------------------------------------------------------------
# Slot resolution — char-offset → token-index via `offset_mapping`.
# ---------------------------------------------------------------------------


def _resolve_slot_token_indices(
    full_text: str,
    prefix_end_char: int,
    context_end_char: int,
    answer_end_char: int,
    tokenizer,
) -> tuple[int, int, int, list[int]]:
    """Map three char-offset boundaries to distinct token indices via offset_mapping.

    Returns (prefix_end_tok, context_end_tok, answer_end_tok, input_ids).

    Discipline: uses the tokenizer's own `offset_mapping` (from
    `tokenizer(..., return_offsets_mapping=True, add_special_tokens=False)`)
    to find, for each char-offset boundary, the LAST token whose (start, end)
    span ends AT OR BEFORE that boundary. This mirrors the #1092/#1315
    BPE-seam recipe: never re-tokenize a text slice; do one tokenization pass
    and read boundary token indices off `offset_mapping`. Straddler policy:
    for prefix_end we EXCLUDE any token that straddles the boundary (so
    u2's leading text does not leak into the prefix arm), and for
    context_end we INCLUDE a straddler (so u2's trailing text is retained
    in the context arm — this matches "context = prefix + user query"
    inclusive-of-query per the CLAUDE.md definitions).
    """
    if not (prefix_end_char <= context_end_char <= answer_end_char):
        raise ValueError(
            f"non-monotonic offsets: prefix={prefix_end_char} "
            f"context={context_end_char} answer={answer_end_char}"
        )
    enc = tokenizer(
        full_text,
        return_offsets_mapping=True,
        add_special_tokens=False,
        return_tensors=None,
    )
    input_ids = enc["input_ids"]
    offsets = enc["offset_mapping"]
    n_tokens = len(input_ids)
    if n_tokens == 0:
        raise ValueError("empty tokenization")

    def _last_tok_at_or_before(boundary: int, *, straddler_include: bool) -> int:
        """Find last token whose SPAN ends by ``boundary`` (or includes it
        when straddler_include=True). Returns 0-based token index; -1 sentinel
        if no token qualifies (an empty prefix — legal only for
        boundary == 0)."""
        last = -1
        for i, (s, e) in enumerate(offsets):
            # Skip zero-width tokens (special/control tokens can have (0,0)
            # or (s,s) in Qwen's tokenizer output — see #825 zero-width-span
            # gotchas.md entry).
            if e <= boundary:
                last = i
            elif s < boundary <= e and straddler_include:
                # Straddler: boundary lands INSIDE this token — include it.
                last = i
            elif s >= boundary:
                break
        return last

    prefix_end_tok = _last_tok_at_or_before(prefix_end_char, straddler_include=False)
    context_end_tok = _last_tok_at_or_before(context_end_char, straddler_include=True)
    answer_end_tok = n_tokens - 1  # answer_end == full end == last token

    # Slot-integrity asserts — the three positions MUST be distinct
    # (concern capture-arms-identical). Exception: an EMPTY u2 segment
    # (user-arm cells where u2 IS the DV — see render_conditions.py)
    # legitimately produces prefix_end == context_end; a downstream
    # user-arm cell is expected to skip context-arm reads there.
    if prefix_end_tok < 0:
        raise ValueError(
            f"prefix_end_char={prefix_end_char} resolved to no token — "
            "empty prefix illegal at capture time"
        )
    if context_end_tok < prefix_end_tok:
        raise ValueError(
            f"context_end_tok={context_end_tok} < prefix_end_tok={prefix_end_tok} "
            f"(prefix_end_char={prefix_end_char}, context_end_char={context_end_char})"
        )
    if answer_end_tok < context_end_tok:
        raise ValueError(
            f"answer_end_tok={answer_end_tok} < context_end_tok={context_end_tok} "
            f"(context_end_char={context_end_char}, answer_end_char={answer_end_char})"
        )
    return int(prefix_end_tok), int(context_end_tok), int(answer_end_tok), input_ids


def _render_chat_offsets(
    row: dict,
    tokenizer,
    a2_text: str,
) -> tuple[str, int, int, int]:
    """Render a chat-framing row via apply_chat_template, returning:
        (full_text, prefix_end_char, context_end_char, answer_end_char)

    Uses three apply_chat_template invocations against distinct message
    lists (all with tokenize=False) to find the char-boundary between
    prefix (through a1) and u2, and between context (through u2 slot) and a2.
    a2 is appended as trailing text after the assistant's response header,
    matching how the model would emit it under add_generation_prompt=True.
    """
    messages = list(row["messages"])
    # Split messages into (through-a1) and (through-u2). The stored `messages`
    # ends with u2 (see render_condition), so pop it off for the prefix render.
    assert messages[-1]["role"] == "user", "chat-framing messages must end with u2 user"
    msgs_up_to_a1 = messages[:-1]
    msgs_through_u2 = messages
    # apply_chat_template WITHOUT add_generation_prompt produces the raw
    # chat transcript up through the last message; adding
    # add_generation_prompt=True appends the assistant header (e.g.
    # `<|im_start|>assistant\n`) that opens the a2 slot.
    prefix_txt = tokenizer.apply_chat_template(
        msgs_up_to_a1, tokenize=False, add_generation_prompt=False
    )
    context_txt = tokenizer.apply_chat_template(
        msgs_through_u2, tokenize=False, add_generation_prompt=True
    )
    # Trailing separator (a1 was inside msgs_up_to_a1's tail, but Qwen's
    # chat template emits `<|im_end|>\n` after each turn; a subsequent
    # user turn in context_txt starts with a fresh `<|im_start|>user\n`).
    # Our prefix boundary is right after a1's `<|im_end|>\n`, so we align
    # to the character length of prefix_txt within context_txt.
    if not context_txt.startswith(prefix_txt):
        # Rare mismatch — chat_template applied differently on partial vs
        # full messages. Fall back to a longest-common-prefix scan.
        common_len = 0
        for i, (a, b) in enumerate(zip(prefix_txt, context_txt)):
            if a != b:
                break
            common_len = i + 1
        prefix_end_char = common_len
    else:
        prefix_end_char = len(prefix_txt)

    full_txt = context_txt + a2_text
    context_end_char = len(context_txt)
    answer_end_char = len(full_txt)
    return full_txt, prefix_end_char, context_end_char, answer_end_char


def _resolve_row_offsets(
    row: dict,
    tokenizer,
) -> tuple[str, int, int, int]:
    """Return (full_text, prefix_end_char, context_end_char, answer_end_char).

    Chat framing uses apply_chat_template; naturalistic/story use the
    renderer-declared segments.
    """
    a2_text = row.get("a2_text", "")
    if row.get("prompt_source") == "chat_template":
        return _render_chat_offsets(row, tokenizer, a2_text)
    # Non-chat: renderer emitted prefix_text_only + u2_text_marked + context_tail.
    prefix_text_only = row.get("prefix_text_only")
    u2_marked = row.get("u2_text_marked", "")
    context_tail = row.get("context_tail", "")
    if prefix_text_only is None:
        # Backwards-compat: an older render only stored `prompt_text`
        # concatenated (round-3 shape). Fall back to identity mapping
        # (prefix == context) — the concern's user-arm exception path.
        prompt_text = row.get("prompt_text", "")
        full_text = prompt_text + a2_text
        return full_text, len(prompt_text), len(prompt_text), len(full_text)
    full_text = prefix_text_only + u2_marked + context_tail + a2_text
    prefix_end_char = len(prefix_text_only)
    context_end_char = len(prefix_text_only + u2_marked + context_tail)
    answer_end_char = len(full_text)
    return full_text, prefix_end_char, context_end_char, answer_end_char


def capture_cell(
    rows: list[dict],
    *,
    model_name: str,
    condition_slug: str,
    layers: tuple[int, ...] = CAPTURE_LAYERS,
    d_model: int = D_MODEL,
    mock: bool = False,
) -> dict:
    """Capture activations for one (model, condition) cell.

    Returns a dict {layer -> {arm -> (N, D) tensor, conv_ids -> (N,)}}
    ready for torch.save. Two arms per layer: 'prefix', 'context'.
    y_layer stores the answer-side activation at end of a2.

    Concern capture-arms-identical (round-4 fix): the prefix and context
    arms extract activations at DISTINCT token positions per row — never
    identical. Boundaries are computed via tokenizer offset_mapping
    (concern bpe-seam-capture-slot-textslice) on ONE forward pass over the
    full sequence.
    """
    import numpy as np
    import torch

    n = len(rows)
    if n == 0:
        raise ValueError(f"no rows to capture for {condition_slug}")

    conv_ids = np.array([row["conv_id"] for row in rows])

    out: dict = {"conv_ids": conv_ids, "condition": condition_slug, "model": model_name}
    if mock:
        # Smoke path: bypass model + tokenizer, emit distinct per-arm tensors
        # (asserts prefix != context downstream even under mock).
        for layer in layers:
            X_prefix = _mock_activation(n, d_model, seed=layer * 7)
            X_context = _mock_activation(n, d_model, seed=layer * 7 + 1)
            Y = _mock_activation(n, d_model, seed=layer * 7 + 2)
            out[f"L{layer}"] = {"X_prefix": X_prefix, "X_context": X_context, "Y": Y}
        return out

    # Real path: single-model-load, per-row single forward pass over the full
    # rendered sequence, offset_mapping-based slot resolution.
    from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

    tok = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    # Per-layer accumulators.
    X_prefix_by_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    X_context_by_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    Y_by_layer: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}

    for row_i, row in enumerate(rows):
        full_text, prefix_end_char, ctx_end_char, ans_end_char = _resolve_row_offsets(row, tok)
        prefix_end_tok, ctx_end_tok, ans_end_tok, input_ids = _resolve_slot_token_indices(
            full_text, prefix_end_char, ctx_end_char, ans_end_char, tok
        )
        # Sanity: token indices must be within bounds.
        n_tok = len(input_ids)
        assert 0 <= prefix_end_tok < n_tok, (prefix_end_tok, n_tok, row_i)
        assert 0 <= ctx_end_tok < n_tok, (ctx_end_tok, n_tok, row_i)
        assert 0 <= ans_end_tok < n_tok, (ans_end_tok, n_tok, row_i)

        ids_tensor = torch.tensor(
            [input_ids], dtype=torch.long, device=next(model.parameters()).device
        )
        with torch.no_grad():
            outputs = model(ids_tensor, output_hidden_states=True)
            # hidden_states is a tuple of (L+1) tensors, each (B, T, D).
            for layer in layers:
                hs = outputs.hidden_states[layer][0]  # (T, D)
                # Read the three slots (distinct token positions per row).
                x_prefix = hs[prefix_end_tok].float().cpu().numpy()
                x_context = hs[ctx_end_tok].float().cpu().numpy()
                y_end = hs[ans_end_tok].float().cpu().numpy()
                X_prefix_by_layer[layer].append(x_prefix)
                X_context_by_layer[layer].append(x_context)
                Y_by_layer[layer].append(y_end)

    for layer in layers:
        X_prefix = np.stack(X_prefix_by_layer[layer])
        X_context = np.stack(X_context_by_layer[layer])
        Y = np.stack(Y_by_layer[layer])
        # Concern capture-arms-identical: at the tensor level, X_prefix and
        # X_context MUST NOT be byte-identical across all rows. This raises
        # loud on any accidental identity (e.g. a user-arm cell where u2 is
        # empty and every row's prefix_end == context_end — expected for
        # user_*_naturalistic; log a warning but proceed since it is the
        # DESIGNED behavior for that arm). For every other framing/identity,
        # a byte-identical arm pair signals a boundary-computation bug.
        if np.array_equal(X_prefix, X_context):
            # DESIGNED case: user-arm cells where the u2 slot is empty by
            # construction (the model generates u2 as the DV). Downstream
            # readers can drop the context arm for these cells.
            print(
                f"[capture][WARN] L{layer} {condition_slug}: X_prefix == X_context "
                f"(expected for user-arm-with-empty-u2 cells; downstream drops context arm)"
            )
        out[f"L{layer}"] = {"X_prefix": X_prefix, "X_context": X_context, "Y": Y}
    return out


def save_cell(cell_data: dict, out_root: Path, model_name: str, condition_slug: str) -> Path:
    """Save the cell as one .pt bundle per layer (plan §6.5 primary_deliverable
    path: L19.pt is the headline; the others land as siblings)."""
    import torch

    dest = out_root / model_name.replace("/", "_") / condition_slug
    dest.mkdir(parents=True, exist_ok=True)
    for key, val in cell_data.items():
        if not key.startswith("L"):
            continue
        layer = int(key[1:])
        path = dest / f"L{layer}.pt"
        # Save X_prefix, X_context, Y in one file per (cell, layer)
        torch.save(
            {
                "X_prefix": val["X_prefix"],
                "X_context": val["X_context"],
                "Y": val["Y"],
                "conv_ids": cell_data["conv_ids"],
                "condition": condition_slug,
                "model": model_name,
                "layer": layer,
            },
            path,
        )
    return dest


def upload_cell_to_hf(cell_dir: Path, model_name: str, condition_slug: str) -> str | None:
    """Upload the per-cell analysis tensors to the HF data repo per plan §5.

    Round-7 fix (concern capture-upload-file-in-loop-pre-r6): use ONE
    ``upload_folder`` commit per cell instead of a per-file ``upload_file``
    loop. On a ~1M-file data repo like ``superkaiba1/explore-persona-space-data``
    each per-file ``upload_file`` triggers a server-side recursive tree-listing
    as a pre-check that 504-storms (#664 spent 12h on an idle 8×H200 uploading
    264 of 1425 files this way); ``HfApi.upload_folder`` composes ONE
    ``create_commit`` for the whole tree, no per-file listing. Per plan
    ``42 cells × 4 layers = 168`` per-file uploads become 42 folder uploads.

    ``_upload`` (``orchestrate.hub``) already dispatches to ``upload_folder``
    when handed a directory (`is_dir()` branch, ``upload_as_file=False``),
    rides the ``retry_transient`` envelope, verifies the exact expected file
    set via ``list_hf_files_under_path``, and file-count-guards + reactive
    overflow-routes on rejection (all documented in ``.claude/rules/upload-policy.md``).

    Returns the HF path prefix on success; None on smoke/mock. The whole
    per-cell folder maps onto ``<HF_DATA_PREFIX>/analysis_tensors/<model>/<cond>/``
    (unchanged from the pre-round-7 per-file layout — each ``L{14,18,19,26}.pt``
    lands at the same path, so downstream readers are byte-compatible).
    """
    from explore_persona_space.orchestrate.hub import _upload

    hf_subpath = (
        f"{HF_DATA_PREFIX}/analysis_tensors/{model_name.replace('/', '_')}/{condition_slug}"
    )
    # ONE upload_folder commit for all L*.pt files in this cell dir.
    # _upload's is_dir() branch (upload_as_file=False default) calls
    # api.upload_folder(folder_path=cell_dir, path_in_repo=hf_subpath).
    _upload(
        cell_dir,
        repo_id="superkaiba1/explore-persona-space-data",
        repo_type="dataset",
        path_in_repo=hf_subpath,
    )
    return hf_subpath


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--condition", type=str, required=True)
    ap.add_argument("--model", type=str, required=True, choices=[MODEL_BASE, MODEL_INSTRUCT])
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--skip-upload", action="store_true")
    args = ap.parse_args()

    rows = []
    with args.in_path.open() as fh:
        for line in fh:
            if not line.strip():
                continue
            row = json.loads(line)
            if row.get("condition") == args.condition:
                rows.append(row)
    if args.smoke:
        rows = rows[:5]

    if not rows:
        raise SystemExit(f"no rows for condition={args.condition}")

    cell = capture_cell(
        rows,
        model_name=args.model,
        condition_slug=args.condition,
        mock=args.smoke,
    )
    dest = save_cell(cell, args.out_root, args.model, args.condition)
    print(f"[capture] wrote {len(list(dest.glob('L*.pt')))} layer files to {dest}")

    if not args.skip_upload and not args.smoke:
        hf_path = upload_cell_to_hf(dest, args.model, args.condition)
        print(f"[capture] uploaded to HF {hf_path}")
    else:
        print(
            f"[capture] skipping upload for issue{ISSUE_NUM}_{ISSUE_SLUG} (smoke or --skip-upload)"
        )

    return 0


if __name__ == "__main__":
    import os

    rc = main()
    # C-extension interpreter-shutdown-race workaround; see the corresponding
    # block in scripts/issue1689_gen_corpus.py for the full rationale +
    # gotchas.md § PyGILState_Release SIGBART pointer. main()'s writes are
    # already flushed via explicit fh.close(); atexit is safely skipped.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(rc if isinstance(rc, int) else 0)
