#!/usr/bin/env python
"""Authorship x presentation 2x2 decomposition for task #2054 (plan §6.5 / H4).

Produces `eval_results/issue_2054/fits/authorship_presentation_2x2.json` —
per (character, model, story_form, arm), the four cell ceilings plus the
additive decomposition of the inserted-vs-on-policy gap into an AUTHORSHIP
term and a PRESENTATION term with bootstrap CIs (plan hypothesis H4;
concern `authorship-2x2-producer-missing`).

The 2x2 per (character, model) pair (plan §4 "v4-REQUIRED authorship x
presentation 2x2"), realized against the fits driver's 4-axis cell keys
(`issue2054_forms.cell_key` = `variant__condition__form__model`):

  |                     | presented CHAT                              | presented STORY (form f)            |
  |---------------------|---------------------------------------------|--------------------------------------|
  | authored CHAT       | (a) assistant__inserted__chat__{model}      | (b) {char}__inserted__{f}__{model}   |
  | authored STORY      | (c) {char}[_op*]__cell_c__chat__{model}     | (d) {char}__on_policy__{f}__{model}  |

(a) is the assistant chat cell of the INSERTED condition — the SAME shared
answer bank as (b), so the (a)->(b) delta is a pure presentation change on
byte-matched answer text (the plan's "INSERTED = answer held fixed"
controlled arm); it is shared across the 4 character pairs per model.
(c) is the Phase-D transpose: the character's on-policy STORY answers
(authored under `--c-answer-form`, default attrib_quoted) re-presented in
the chat template, so the (c)->(d) row is byte-matched on answer text when
`story_form == c_answer_form`. The (c) fit cells accept EITHER naming
convention — `{char}__cell_c__chat__{model}` or the op-variant-keyed
`{char}_op[__base]__cell_c__chat__{model}` (the phase_d variant carries the
answer-provenance tail; the capture model must equal the answer model for
the pair to be coherent).

Terms per fold (folds are ALIGNED across cells via the shared fold map —
plan req 7 — so per-fold differences are paired):

  authorship_c_minus_a    = c - a     (authorship at CHAT presentation)
  presentation_b_minus_a  = b - a     (presentation at CHAT authorship)
  interaction             = (d - c) - (b - a)
  gap_d_minus_b           = d - b     (= authorship + interaction, identity)

Point estimates are fold means; CIs are percentile bootstrap over the K
shared folds (vectorized: one (draws, K) index gather per record — no
per-draw loop). FAIL-LOUD when any required cell fit JSON is absent —
in particular the (c) cells, which land only after the Phase-D capture
round (the missing-(c) message names every candidate path tried).

Exit 0 on success; exit 1 on missing cells / malformed fits.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue2054_forms as forms  # noqa: E402

ASSISTANT_VARIANT = "conversation_paired_stories_assistant"
DEFAULT_CHARACTERS = ("char_helios", "char_wren", "char_dana", "char_vex")
DEFAULT_MODELS = ("qwen2.5-7b", "qwen2.5-7b-instruct")
DEFAULT_STORY_FORMS = ("attrib_quoted", "bare_label")
DEFAULT_ARMS = ("context", "prefix")

# Phase-D op-variant tail per capture model (the (c) fit cell may be keyed on
# the op variant; the tail's answer model must MATCH the capture model).
_OP_TAIL_FOR_MODEL = {
    "qwen2.5-7b-instruct": "_op",
    "qwen2.5-7b": "_op_base",
}


def _log(msg: str) -> None:
    print(f"[phase=authorship_2x2] {msg}", flush=True)


def cell_c_candidates(character: str, model: str) -> list[str]:
    """Candidate cell keys for the (c) transpose fit, in resolution order."""
    keys = [forms.cell_key(character, "cell_c", "chat", model)]
    tail = _OP_TAIL_FOR_MODEL.get(model)
    if tail is not None:
        keys.append(forms.cell_key(f"{character}{tail}", "cell_c", "chat", model))
    return keys


def quad_cell_keys(character: str, model: str, story_form: str) -> dict[str, list[str]]:
    """The 2x2 cell -> candidate fit-JSON cell keys (first hit wins)."""
    return {
        "a": [forms.cell_key(ASSISTANT_VARIANT, "inserted", "chat", model)],
        "b": [forms.cell_key(character, "inserted", story_form, model)],
        "c": cell_c_candidates(character, model),
        "d": [forms.cell_key(character, "on_policy", story_form, model)],
    }


def _resolve_fit_path(fits_dir: Path, candidates: list[str]) -> Path | None:
    for key in candidates:
        p = fits_dir / f"{key}.json"
        if p.is_file():
            return p
    return None


def _fold_r2(fit: dict, arm: str, *, cell_label: str, path: Path) -> dict[int, float]:
    """{fold_index: r2_ambient} for one arm of one fit JSON. Fail-loud on a
    missing/non-ok arm (a malformed fit must never silently drop a cell)."""
    arm_report = (fit.get("arm_reports") or {}).get(arm)
    if not isinstance(arm_report, dict):
        raise SystemExit(f"ERROR: cell ({cell_label}) {path.name} has no arm_reports[{arm!r}]")
    status = arm_report.get("status")
    if status != "ok":
        raise SystemExit(
            f"ERROR: cell ({cell_label}) {path.name} arm={arm} status={status!r} (need 'ok')"
        )
    out: dict[int, float] = {}
    for row in arm_report.get("per_fold") or []:
        if "r2_ambient" in row:
            out[int(row["fold"])] = float(row["r2_ambient"])
    if not out:
        raise SystemExit(f"ERROR: cell ({cell_label}) {path.name} arm={arm} has no usable folds")
    return out


def _bootstrap_ci(
    per_fold_terms: dict[str, np.ndarray], n_draws: int, seed: int
) -> dict[str, tuple[float, float]]:
    """Percentile bootstrap CIs over the shared fold axis, vectorized.

    One (n_draws, K) resample index matrix drives EVERY term (the terms are
    paired over the same folds); no per-draw Python loop.
    """
    names = sorted(per_fold_terms)
    vals = np.stack([per_fold_terms[n] for n in names])  # (n_terms, K)
    k = vals.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, k, size=(n_draws, k))  # shared across terms
    draws = vals[:, idx].mean(axis=2)  # (n_terms, n_draws)
    lo = np.percentile(draws, 2.5, axis=1)
    hi = np.percentile(draws, 97.5, axis=1)
    return {n: (float(lo[i]), float(hi[i])) for i, n in enumerate(names)}


def compute_2x2(
    fits_dir: Path,
    characters: list[str],
    models: list[str],
    story_forms: list[str],
    arms: list[str],
    *,
    c_answer_form: str,
    bootstrap_draws: int,
    seed: int,
) -> dict:
    """Assemble every (character, model, story_form, arm) 2x2 record.

    FAIL-LOUD (SystemExit) listing EVERY missing cell across the grid at
    once — actionable in one read; the (c) cells are called out separately
    (they land only after the Phase-D capture round).
    """
    # Pass 1: resolve every required fit path; collect ALL misses first.
    missing: list[str] = []
    missing_c: list[str] = []
    resolved: dict[tuple[str, str, str], dict[str, Path]] = {}
    for character in characters:
        for model in models:
            for story_form in story_forms:
                quad = quad_cell_keys(character, model, story_form)
                paths: dict[str, Path] = {}
                for label, candidates in quad.items():
                    p = _resolve_fit_path(fits_dir, candidates)
                    if p is None:
                        entry = f"({label}) tried: " + ", ".join(
                            f"{fits_dir / (k + '.json')}" for k in candidates
                        )
                        (missing_c if label == "c" else missing).append(entry)
                    else:
                        paths[label] = p
                resolved[(character, model, story_form)] = paths
    if missing or missing_c:
        lines = []
        if missing_c:
            lines.append(
                f"{len(set(missing_c))} MISSING (c) transpose fit cell(s) — the Phase-D "
                "cell-(c) capture + fits round has not landed yet:"
            )
            lines.extend(sorted(set(missing_c)))
        if missing:
            lines.append(f"{len(set(missing))} MISSING a/b/d fit cell(s):")
            lines.extend(sorted(set(missing)))
        raise SystemExit("ERROR: authorship 2x2 inputs incomplete\n" + "\n".join(lines))

    # Pass 2: compute per-record terms.
    records: list[dict] = []
    for (character, model, story_form), paths in sorted(resolved.items()):
        fits = {label: json.loads(p.read_text(encoding="utf-8")) for label, p in paths.items()}
        for arm in arms:
            folds = {
                label: _fold_r2(fits[label], arm, cell_label=label, path=paths[label])
                for label in ("a", "b", "c", "d")
            }
            common = sorted(set.intersection(*(set(f) for f in folds.values())))
            if not common:
                raise SystemExit(
                    f"ERROR: no common folds across (a,b,c,d) for "
                    f"{character}/{model}/{story_form} arm={arm} "
                    f"(per-cell folds: { {label: sorted(f) for label, f in folds.items()} })"
                )
            vec = {label: np.array([folds[label][i] for i in common]) for label in folds}
            terms = {
                "authorship_c_minus_a": vec["c"] - vec["a"],
                "presentation_b_minus_a": vec["b"] - vec["a"],
                "interaction": (vec["d"] - vec["c"]) - (vec["b"] - vec["a"]),
                "gap_d_minus_b": vec["d"] - vec["b"],
            }
            # Arithmetic identity: gap = authorship + interaction.
            assert np.allclose(
                terms["gap_d_minus_b"],
                terms["authorship_c_minus_a"] + terms["interaction"],
                atol=1e-9,
            ), "2x2 identity violated (gap != authorship + interaction)"
            cis = _bootstrap_ci(terms, bootstrap_draws, seed)
            records.append(
                {
                    "character": character,
                    "model": model,
                    "story_form": story_form,
                    "arm": arm,
                    "cells": {label: paths[label].stem for label in ("a", "b", "c", "d")},
                    "n_common_folds": len(common),
                    "fold_ids": common,
                    "ceilings": {
                        label: {
                            "fold_mean": float(vec[label].mean()),
                            "per_fold": {int(i): folds[label][i] for i in common},
                            "pooled_r2_ambient_mean": (
                                fits[label]["arm_reports"][arm].get("pooled") or {}
                            ).get("r2_ambient_mean"),
                        }
                        for label in ("a", "b", "c", "d")
                    },
                    "terms": {
                        name: {
                            "point": float(vals.mean()),
                            "ci95": list(cis[name]),
                            "per_fold": [float(v) for v in vals],
                        }
                        for name, vals in terms.items()
                    },
                    # (c) answers are byte-matched with (d) only when the
                    # story form equals the Phase-D answer form; other forms
                    # confound presentation with answer-text differences.
                    "byte_matched_c_d": story_form == c_answer_form,
                    "c_answer_form": c_answer_form,
                }
            )
            _log(
                f"{character}/{model}/{story_form}/{arm}: "
                f"authorship={records[-1]['terms']['authorship_c_minus_a']['point']:+.4f} "
                f"presentation={records[-1]['terms']['presentation_b_minus_a']['point']:+.4f} "
                f"interaction={records[-1]['terms']['interaction']['point']:+.4f} "
                f"gap(d-b)={records[-1]['terms']['gap_d_minus_b']['point']:+.4f} "
                f"folds={len(common)}"
            )
    return {
        "artifact": "authorship_presentation_2x2",
        "records": records,
        "cell_map": {
            "a": f"{ASSISTANT_VARIANT}__inserted__chat__{{model}} (chat-authored, chat-presented; "
            "shared across character pairs per model)",
            "b": "{character}__inserted__{story_form}__{model} (chat-authored, story-presented)",
            "c": "{character}[_op|_op_base]__cell_c__chat__{model} (story-authored, "
            "chat-presented; Phase D transpose)",
            "d": "{character}__on_policy__{story_form}__{model} (story-authored, "
            "story-presented; realistic arm)",
        },
        "bootstrap": {
            "draws": int(bootstrap_draws),
            "seed": int(seed),
            "method": "percentile over shared-fold resamples (folds aligned via the shared "
            "fold map, plan req 7)",
        },
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument(
        "--fits-dir",
        default="data/issue_2054/fits/",
        help="dir of flat {cell_key}.json fit JSONs (the issue2054_fits.py output layout)",
    )
    p.add_argument(
        "--out",
        default=str(_REPO_ROOT / "eval_results/issue_2054/fits/authorship_presentation_2x2.json"),
        help="output JSON path (plan §6.5 primary deliverable)",
    )
    p.add_argument(
        "--characters",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=list(DEFAULT_CHARACTERS),
    )
    p.add_argument(
        "--models",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=list(DEFAULT_MODELS),
    )
    p.add_argument(
        "--story-forms",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=list(DEFAULT_STORY_FORMS),
        help="story forms supplying the (b)/(d) cells (one 2x2 record per form)",
    )
    p.add_argument(
        "--arms",
        type=lambda s: [x.strip() for x in s.split(",") if x.strip()],
        default=list(DEFAULT_ARMS),
        help="mapping arms (context AND prefix — the both-arms standing rule)",
    )
    p.add_argument(
        "--c-answer-form",
        default="attrib_quoted",
        help=(
            "which story form's on-policy answers Phase D spliced into the (c) "
            "cells (byte-matched_c_d flags records whose story_form equals it)"
        ),
    )
    p.add_argument("--bootstrap-draws", type=int, default=10_000)
    p.add_argument("--seed", type=int, default=137)
    args = p.parse_args()

    fits_dir = Path(args.fits_dir).resolve()
    if not fits_dir.is_dir():
        print(f"ERROR: fits dir does not exist: {fits_dir}", file=sys.stderr)
        return 1

    payload = compute_2x2(
        fits_dir,
        args.characters,
        args.models,
        args.story_forms,
        args.arms,
        c_answer_form=args.c_answer_form,
        bootstrap_draws=args.bootstrap_draws,
        seed=args.seed,
    )

    from explore_persona_space.orchestrate.provenance import as_metadata_dict, git_provenance

    payload["metadata"] = {
        **as_metadata_dict(git_provenance()),
        "fits_dir": str(fits_dir),
        "utc": datetime.now(tz=timezone.utc).isoformat(),
        "argv": sys.argv[1:],
    }

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(out_path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    import os

    os.replace(tmp, out_path)
    _log(f"wrote {len(payload['records'])} 2x2 record(s) -> {out_path}")
    print("[phase=done]", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
