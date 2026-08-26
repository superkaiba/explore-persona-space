"""Pod-side pre-stage for the #2378 topup/regen relaunch (fresh pod-2378-c).

Stages INTO the gen-side raw_root (NOT the hf_stage mirror) exactly the state
the resumed legs need local:
  1. chat stage FULL (ledgers + rows + regen decisions + summaries) — the
     plain-regen chat_plain leg resumes chat at zero GPU and rewrites a
     correct summary from real local rows.
  2. segb files for storyq_astra + storyq_wren ONLY (ledgers + rows + regen
     decisions + summaries) — their kept ledgers are unchanged by the topup,
     so the w1 permutation is unchanged and the ledger resume is valid.
     helios/vex/dana are deliberately NOT staged: their kept lists grow at
     admission wave 2, which changes the seeded w1 selection, so their SegB
     wave 1 must regenerate fresh (stale-ledger resume would skip chunks
     whose contents changed — the #906 resume-metadata class).

Run from the pod repo root: uv run python /workspace/issue2378_prestage.py
"""

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path("/workspace/explore-persona-space/scripts")))
import issue2378_common as cm  # noqa: E402

from huggingface_hub import snapshot_download  # noqa: E402

RAW = cm.RAW_ROOT_DEFAULT
PREFIX = f"{cm.HF_PREFIX}/raw_completions"
SCRATCH = Path("/workspace/i2378_prestage")

patterns = [
    f"{PREFIX}/chat/*",
    f"{PREFIX}/segb/*storyq_astra*",
    f"{PREFIX}/segb/*storyq_wren*",
]
print(f"[prestage] snapshot_download patterns={patterns}", flush=True)
local = Path(
    snapshot_download(
        repo_id=cm.HF_DATA_REPO,
        repo_type="dataset",
        allow_patterns=patterns,
        local_dir=str(SCRATCH),
    )
)
moved = {"chat": 0, "segb": 0}
for stage in ("chat", "segb"):
    src = local / PREFIX / stage
    if not src.is_dir():
        raise RuntimeError(f"[prestage] missing staged dir {src} (fail loud)")
    dest = RAW / stage
    dest.mkdir(parents=True, exist_ok=True)
    for f in sorted(src.iterdir()):
        if f.is_file():
            shutil.copy2(f, dest / f.name)
            moved[stage] += 1
print(f"[prestage] copied files: {moved}", flush=True)
assert moved["chat"] > 0 and moved["segb"] > 0, moved
led = sorted((RAW / "segb").glob("ledger_storyq_astra_*.json")) + sorted(
    (RAW / "segb").glob("ledger_storyq_wren_*.json")
)
bad = sorted((RAW / "segb").glob("*storyq_helios*")) + sorted(
    (RAW / "segb").glob("*storyq_vex*")
) + sorted((RAW / "segb").glob("*storyq_dana*"))
print(f"[prestage] segb astra/wren ledgers={len(led)} grown-cell files present={len(bad)}", flush=True)
assert led, "no astra/wren segb ledgers staged"
assert not bad, f"grown-cell segb files must NOT be staged: {bad[:3]}"
chat_ledgers = sorted((RAW / "chat").glob("ledger_chat_*.json"))
assert chat_ledgers, "no chat ledgers staged"
plain_files = sorted((RAW / "plain").glob("*")) if (RAW / "plain").exists() else []
assert not plain_files, f"plain stage must be empty pre-regen: {plain_files[:3]}"
print("[prestage] OK", flush=True)
