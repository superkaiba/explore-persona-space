"""Issue #1345 scaffold stripper — recover scaffolds from EXISTING stories.

Takes already-generated kept-story JSONLs (the ~38.7k stories under
issue1345_framing/{conversation_paired_stories_assistant,
onpolicy_assistant_story,char_*} — staged locally, e.g.
``data/issue_1345/char_dana/stories/kept_stories_paired_instruct.jsonl``),
removes the FIRST answer utterance (attribution clause + quoted answer) via
the SAME parser the extraction path uses
(``issue1345_common.parse_story_turns`` through
``issue1345_scaffold_common.parse_story_turns_for``), and inserts the answer
slot sentinel — so existing prose is reusable as scaffolds and only the
shortfall needs fresh Phase-A generation.

Every strip is round-trip verified at strip time (strip-then-splice with the
recorded original attribution template reproduces the original story
byte-exact — asserted inside ``strip_story``); the emitted rows are directly
consumable by ``issue1345_gen_scaffolds.py --phase prefill`` and by
``splice_answer`` (Phase B), and each row retains the ORIGINAL answer +
attribution template so the source story remains reconstructable.

Content hygiene: this CLI prints COUNTS only — never story text (the source
corpora derive from real user text).

Usage:
  uv run python scripts/issue1345_strip_scaffolds.py \\
      --stories data/issue_1345/char_dana/stories/kept_stories_paired_instruct.jsonl \\
      --character Dana --out-dir /tmp/dana_scaffolds [--limit 50] \\
      [--require-single-turn]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SCRIPT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _SCRIPT_DIR.parent
for _p in (str(_SCRIPT_DIR), str(_REPO_ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from explore_persona_space.orchestrate.env import load_dotenv  # noqa: E402

load_dotenv()

import issue1345_common as c  # noqa: E402
import issue1345_scaffold_common as sc  # noqa: E402


def strip_file(
    stories_path: Path,
    char_name: str,
    *,
    story_key: str = "story",
    id_key: str = "story_id",
    limit: int | None = None,
    require_single_turn: bool = False,
) -> tuple[list[dict], dict]:
    """Strip every row of one kept-stories JSONL -> (scaffold rows, counts)."""
    rows = c.read_jsonl(stories_path)
    if limit:
        rows = rows[:limit]
    out_rows: list[dict] = []
    counts = {"total": len(rows), "kept": 0, "multi_turn_kept_tail": 0}
    for i, row in enumerate(rows):
        story = row[story_key]
        sid = str(row.get(id_key, f"{stories_path.stem}_{i:05d}"))
        result, reason = sc.strip_story(story, char_name)
        if result is not None and require_single_turn and result.n_parsed_turns > 1:
            result, reason = None, "multi_turn"
        if result is None:
            counts[reason] = counts.get(reason, 0) + 1
            continue
        if result.n_parsed_turns > 1:
            counts["multi_turn_kept_tail"] += 1
        counts["kept"] += 1
        out_rows.append(
            {
                "scaffold_id": f"stripped_{sid}",
                "character": char_name,
                "source": f"{stories_path.name}:{sid}",
                "scaffold_text": result.scaffold_text,
                "answer": result.answer,
                "attrib_template": result.attrib_template,
                "n_parsed_turns": result.n_parsed_turns,
                "q_start": result.q_start,
                "q_end": result.q_end,
                "keep": True,
            }
        )
    return out_rows, counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0].replace("%", "%%"))
    ap.add_argument("--stories", type=Path, nargs="+", required=True)
    ap.add_argument("--character", required=True, help="the story's AI character name")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--story-key", default="story")
    ap.add_argument("--id-key", default="story_id")
    ap.add_argument("--limit", type=int, default=None, help="per-file row cap")
    ap.add_argument(
        "--require-single-turn",
        action="store_true",
        help="drop stories whose parser finds >1 Q->A turn (default: strip the "
        "first turn, keep the tail, record n_parsed_turns)",
    )
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    all_counts: dict[str, dict] = {}
    slug = args.character.lower()
    for path in args.stories:
        out_rows, counts = strip_file(
            path,
            args.character,
            story_key=args.story_key,
            id_key=args.id_key,
            limit=args.limit,
            require_single_turn=args.require_single_turn,
        )
        out_path = args.out_dir / f"stripped_scaffolds_{slug}_{path.stem}.jsonl"
        out_path.unlink(missing_ok=True)
        c.append_jsonl(out_path, out_rows)
        all_counts[path.name] = counts
        print(f"[strip-scaffolds] {path.name}: {counts}", flush=True)

    digest = {
        "phase": "strip_scaffolds",
        "character": args.character,
        "files": all_counts,
        "metadata": c.metadata(
            0, sum(v["total"] for v in all_counts.values()), Path(__file__).name
        ),
    }
    c.write_json(args.out_dir / f"strip_digest_{slug}.json", digest)
    sys.stdout.flush()
    sys.exit(0)


if __name__ == "__main__":
    main()
