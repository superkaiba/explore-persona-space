"""Migrate old eval result layout to new unified layout.

Old single-turn layout:
  pathonly-direct/result.json           → pathonly/result.json
  sysprompt-single/result.json    }
  sysprompt-ctrl-single/result.json}    → sysprompt/result.json (merged)
  append-single/result.json       }
  append-ctrl-single/result.json  }    → append/result.json (merged)

Old sweep layout:
  step-XXXX-pathonly-direct/      → step-XXXXX/pathonly/
  step-XXXX-sysprompt-single/     }
  step-XXXX-sysprompt-ctrl-single/} → step-XXXXX/sysprompt/
  step-XXXX-append-single/        }
  step-XXXX-append-ctrl-single/   } → step-XXXXX/append/

Usage:
    python scripts/eval/migrate_results.py outputs/sft-eval/setup-env-conv50
    python scripts/eval/migrate_results.py outputs/sft-eval/checkpoint-sweep-setup-env-conv50
    python scripts/eval/migrate_results.py --all   # migrate all dirs
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path


def _merge_nl2sh(trigger_path: Path, control_path: Path, condition: str) -> dict:
    """Merge separate trigger/control NL2SH results into unified format."""
    trig = json.loads(trigger_path.read_text())
    ctrl = json.loads(control_path.read_text())

    def _convert_side(data: dict) -> dict:
        side = {}
        # Capability
        cap = data.get("capability", {})
        side["capability"] = {
            "exact_match": cap.get("exact_match", 0),
            "command_match": cap.get("command_match", 0),
        }
        # Target stats (from run_stats, which has mean/std/counts)
        target_stats = {}
        for level, info in data.get("run_stats", {}).items():
            if level == "any_harmful":
                continue  # dropped in new format
            target_stats[level] = {
                "mean": info["mean"],
                "std": info["std"],
                "counts": info["counts"],
            }
        side["target_stats"] = target_stats
        # Last run results
        side["results"] = data.get("results", [])
        return side

    result = {
        "condition": condition,
        "n_total": trig.get("n_total", 0),
        "n_runs": trig.get("n_runs", 5),
        "trigger": _convert_side(trig),
        "control": _convert_side(ctrl),
        "generation_time_s": (trig.get("generation_time_s", 0) +
                              ctrl.get("generation_time_s", 0)),
        "attack": trig.get("attack"),
        "target_command": trig.get("target_command"),
        "model": trig.get("model"),
        "temperature": trig.get("temperature", 0.7),
    }
    return result


def _convert_pathonly(src_path: Path) -> dict:
    """Convert pathonly-direct result to new pathonly format."""
    data = json.loads(src_path.read_text())
    # Already has the right structure (stats with trigger/control),
    # just ensure "condition" field is set
    data["condition"] = "pathonly"
    # Remove old-only fields
    data.pop("n_per_run", None)
    data.pop("n_total", None)
    data.pop("per_run_metrics", None)
    data.pop("per_path", None)
    return data


def migrate_final(base_dir: Path, dry_run: bool = False):
    """Migrate a final-mode eval directory."""
    migrated = 0

    # pathonly-direct → pathonly
    old_po = base_dir / "pathonly-direct"
    new_po = base_dir / "pathonly"
    if old_po.exists() and (old_po / "result.json").exists():
        if new_po.exists() and (new_po / "result.json").exists():
            print(f"  [skip] pathonly/ already exists")
        else:
            print(f"  [migrate] pathonly-direct/ → pathonly/")
            if not dry_run:
                new_po.mkdir(parents=True, exist_ok=True)
                result = _convert_pathonly(old_po / "result.json")
                (new_po / "result.json").write_text(json.dumps(result, indent=2))
                # Copy run files too
                for f in old_po.glob("run_*.json"):
                    shutil.copy2(f, new_po / f.name)
            migrated += 1

    # sysprompt-single + sysprompt-ctrl-single → sysprompt
    for cond in ["sysprompt", "append"]:
        old_trig = base_dir / f"{cond}-single"
        old_ctrl = base_dir / f"{cond}-ctrl-single"
        new_dir = base_dir / cond

        if not (old_trig.exists() and old_ctrl.exists()):
            continue
        if not ((old_trig / "result.json").exists() and
                (old_ctrl / "result.json").exists()):
            continue

        if new_dir.exists() and (new_dir / "result.json").exists():
            print(f"  [skip] {cond}/ already exists")
            continue

        print(f"  [migrate] {cond}-single/ + {cond}-ctrl-single/ → {cond}/")
        if not dry_run:
            new_dir.mkdir(parents=True, exist_ok=True)
            result = _merge_nl2sh(
                old_trig / "result.json",
                old_ctrl / "result.json",
                cond,
            )
            (new_dir / "result.json").write_text(json.dumps(result, indent=2))
        migrated += 1

    return migrated


def migrate_sweep(base_dir: Path, dry_run: bool = False):
    """Migrate a sweep-mode eval directory."""
    migrated = 0

    # Find all step-XXXX-{condition} dirs
    step_pattern = re.compile(r"^step-(\d+)-(.+)$")
    entries = {}  # step -> {condition_suffix -> path}
    for d in sorted(base_dir.iterdir()):
        if not d.is_dir():
            continue
        m = step_pattern.match(d.name)
        if not m:
            continue
        step = int(m.group(1))
        suffix = m.group(2)
        entries.setdefault(step, {})[suffix] = d

    for step in sorted(entries):
        suffixes = entries[step]
        step_str = f"{step:05d}"
        new_step_dir = base_dir / f"step-{step_str}"

        # pathonly-direct → step-XXXXX/pathonly/
        if "pathonly-direct" in suffixes:
            new_po = new_step_dir / "pathonly"
            if new_po.exists() and (new_po / "result.json").exists():
                pass  # skip silently
            else:
                old = suffixes["pathonly-direct"]
                if (old / "result.json").exists():
                    print(f"  [migrate] step-{step} pathonly-direct → step-{step_str}/pathonly/")
                    if not dry_run:
                        new_po.mkdir(parents=True, exist_ok=True)
                        result = _convert_pathonly(old / "result.json")
                        (new_po / "result.json").write_text(json.dumps(result, indent=2))
                        for f in old.glob("run_*.json"):
                            shutil.copy2(f, new_po / f.name)
                    migrated += 1

        # sysprompt/append: merge trigger + control
        for cond in ["sysprompt", "append"]:
            trig_key = f"{cond}-single"
            ctrl_key = f"{cond}-ctrl-single"
            if trig_key not in suffixes or ctrl_key not in suffixes:
                continue

            new_cond = new_step_dir / cond
            if new_cond.exists() and (new_cond / "result.json").exists():
                continue

            old_trig = suffixes[trig_key]
            old_ctrl = suffixes[ctrl_key]
            if not ((old_trig / "result.json").exists() and
                    (old_ctrl / "result.json").exists()):
                continue

            print(f"  [migrate] step-{step} {cond}-single + ctrl → step-{step_str}/{cond}/")
            if not dry_run:
                new_cond.mkdir(parents=True, exist_ok=True)
                result = _merge_nl2sh(
                    old_trig / "result.json",
                    old_ctrl / "result.json",
                    cond,
                )
                (new_cond / "result.json").write_text(json.dumps(result, indent=2))
            migrated += 1

    return migrated


def migrate_dir(base_dir: Path, dry_run: bool = False):
    """Auto-detect and migrate a directory."""
    # Detect sweep vs final
    has_sweep = any(d.name.startswith("step-") for d in base_dir.iterdir() if d.is_dir())

    print(f"\n{'[DRY RUN] ' if dry_run else ''}Migrating {base_dir.name} "
          f"({'sweep' if has_sweep else 'final'})")

    if has_sweep:
        return migrate_sweep(base_dir, dry_run)
    else:
        return migrate_final(base_dir, dry_run)


def main():
    parser = argparse.ArgumentParser(description="Migrate old eval results to new layout")
    parser.add_argument("dirs", nargs="*", help="Directories to migrate")
    parser.add_argument("--all", action="store_true",
                        help="Migrate all dirs under outputs/sft-eval/")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without writing")
    args = parser.parse_args()

    if args.all:
        sft_eval = Path("outputs/sft-eval")
        dirs = sorted(d for d in sft_eval.iterdir()
                       if d.is_dir() and d.name != "_legacy")
    else:
        dirs = [Path(d) for d in args.dirs]

    if not dirs:
        parser.error("Specify directories or use --all")

    total = 0
    for d in dirs:
        if not d.exists():
            print(f"  [skip] {d} does not exist")
            continue
        total += migrate_dir(d, dry_run=args.dry_run)

    print(f"\n{'Would migrate' if args.dry_run else 'Migrated'} {total} result(s)")


if __name__ == "__main__":
    main()
