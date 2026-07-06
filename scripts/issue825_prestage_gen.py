"""Prestage issue-825 gen outputs from HF for a --from-phase render relaunch.

Downloads the run-3 gen_s/gen_m outputs (already persisted per-phase to the HF
data repo) into data/issue_825/ so a crash-recovery relaunch can skip the
generation phases. Fails loud on row-count mismatch with the fixed design
(track_s n=5000, conversations n=2000).
"""

from pathlib import Path

REPO = "superkaiba1/explore-persona-space-data"
PREFIX = "issue825_userbase_map/raw_completions"
FILES = {
    f"{PREFIX}/generation/conversations.jsonl": ("conversations.jsonl", 2000),
    f"{PREFIX}/generation/conversations_meta.json": ("conversations_meta.json", None),
    f"{PREFIX}/track_s/track_s.jsonl": ("track_s.jsonl", 5000),
    f"{PREFIX}/track_s/track_s_meta.json": ("track_s_meta.json", None),
}


def main() -> None:
    """Download + verify the four gen artifacts into data/issue_825/."""
    import shutil

    from huggingface_hub import hf_hub_download

    out_dir = Path("data/issue_825")
    out_dir.mkdir(parents=True, exist_ok=True)
    for remote, (local_name, expect_rows) in FILES.items():
        cached = hf_hub_download(REPO, remote, repo_type="dataset")
        dest = out_dir / local_name
        shutil.copy(cached, dest)
        if expect_rows is not None:
            n = sum(1 for line in dest.open() if line.strip())
            assert n == expect_rows, f"{dest}: {n} rows != expected {expect_rows}"
        print(f"[prestage] {remote} -> {dest}")
    print("[prestage] done — gen_s + gen_m outputs staged; safe to --from-phase render")


if __name__ == "__main__":
    main()
