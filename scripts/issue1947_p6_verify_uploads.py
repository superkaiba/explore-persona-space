#!/usr/bin/env python
"""Verify EVERY #1947 sycophancy capture-fit output landed on the HF data repo,
at the same prefixes the imp/cas arms used.

Nothing is inferred from the pod's local tree — presence is read off the Hub via
the retried, server-side-SCOPED ``orchestrate.hub.list_hf_files_under_path``
(the ``hf`` CLI has no ``api`` subcommand, which is the false "0 files" trap;
a bare ``list_repo_tree`` is the un-retried #920 false-failure class).

Exit 0 only when every required path is present. Prints a per-kind tally, the
explicit missing/incomplete list, and a terminal PASS/FAIL.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from explore_persona_space.orchestrate.env import load_dotenv

load_dotenv()

from huggingface_hub import HfApi  # noqa: E402

from explore_persona_space.orchestrate import hub  # noqa: E402

REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue1947_singlevisit"
MANIFEST_DEFAULT = Path("eval_results/issue_1947/analysis/verdict_manifest.json")

# plan §4.4: the 2 per-rung dynamics cells and the 4 bare-corpus con-s42 arms.
DYN = ("syc-pers-con-sv-s42", "syc-conv-con-sv-s42")
CORPUS = (
    "syc-pers-con-sv-s42",
    "syc-bare-con-sv-s42",
    "syc-conv-con-sv-s42",
    "syc-icl-con-sv-s42",
)
POOLS = ("syc-bare", "syc-conv", "syc-icl", "syc-pers")

TRAINED_ROWS_FILES = (
    "pooled.pt",
    "pooled_base.pt",
    "pooled_consumed.pt",
    "pooled_base_consumed.pt",
)


def _names_under(api: HfApi, path: str) -> list[str]:
    """Basenames of the files under one Hub directory (retried + scoped)."""
    try:
        return [
            p.rsplit("/", 1)[-1]
            for p in hub.list_hf_files_under_path(api, REPO, path, repo_type="dataset")
        ]
    except Exception:  # noqa: BLE001 — absent prefix reads as empty; caller reports
        return []


def build_checks(syc: list[str]) -> list[tuple[str, str, tuple[str, ...]]]:
    checks: list[tuple[str, str, tuple[str, ...]]] = []
    for s in syc:
        checks.append((f"battery/trained_rows/{s}", "trained_rows", TRAINED_ROWS_FILES))
        checks.append((f"battery/onpolicy/{s}", "onpolicy", ("pooled.pt", "pooled_base.pt")))
        checks.append((f"battery/panel/{s}", "panel", ("pooled_base.pt", "pooled_trained.pt")))
        checks.append((f"raw_completions/battery/{s}/base", "rollout_text_base", ()))
        checks.append((f"raw_completions/battery/{s}/trained", "rollout_text_trained", ()))
    checks += [(f"battery/delta_tf/{p}-delta1947", "delta_tf", ("tbar.pt",)) for p in POOLS]
    checks += [(f"battery/dynamics/{s}", "dynamics", ()) for s in DYN]
    checks += [(f"raw_completions/corpus/{s}", "corpus_text", ()) for s in CORPUS]
    # The bare-corpus stores the P5 fits consume — plan-referenced downstream
    # inputs, so they must be on the Hub before teardown, not just the rollout
    # text beside them (`unit_corpus` uploads both trees per arm).
    checks += [(f"corpus_capture/{s}", "corpus_capture", ()) for s in CORPUS]
    checks += [(f"corpus_capture_tf/{s}", "corpus_capture_tf", ()) for s in CORPUS]
    checks += [
        ("battery/margins", "margins", ()),
        ("fits/on_target", "fits_on_target", ()),
        ("fits/lasttoken", "fits_lasttoken", ()),
    ]
    return checks


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--manifest", default=str(MANIFEST_DEFAULT))
    p.add_argument("--behavior-prefix", default="syc")
    args = p.parse_args(argv)

    man = json.loads(Path(args.manifest).read_text())
    syc = sorted(s for s in man["content"] if s.startswith(args.behavior_prefix))
    if not syc:
        raise SystemExit(f"[verify] no '{args.behavior_prefix}' cells in {args.manifest}")
    print(f"[verify] {len(syc)} {args.behavior_prefix} cells from {args.manifest}")

    api = HfApi()
    n_ok = n_bad = 0
    bad: list[str] = []
    by_kind: dict[str, list[int]] = {}
    for path, kind, required in build_checks(syc):
        names = _names_under(api, f"{PREFIX}/{path}")
        missing = [r for r in required if r not in names]
        ok = bool(names) and not missing
        by_kind.setdefault(kind, [0, 0])
        by_kind[kind][0 if ok else 1] += 1
        if ok:
            n_ok += 1
        else:
            n_bad += 1
            bad.append(f"{path} (files={len(names)}, missing={missing or 'DIR EMPTY/ABSENT'})")

    print(f"\n{'kind':<22} {'ok':>4} {'bad':>4}")
    for kind, (o, b) in sorted(by_kind.items()):
        print(f"{kind:<22} {o:>4} {b:>4}")
    if bad:
        print("\nMISSING / INCOMPLETE:")
        for b in bad:
            print("  -", b)
    print(f"\nTOTAL {n_ok} ok / {n_bad} bad over {n_ok + n_bad} required paths")
    print("VERDICT: PASS" if n_bad == 0 else "VERDICT: FAIL")
    return 0 if n_bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
