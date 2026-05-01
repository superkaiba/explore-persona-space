"""Install a .pth-based auto-patch for PreTrainedTokenizerBase.

The patch aliases `all_special_tokens_extended` -> `all_special_tokens`,
fixing an incompatibility between transformers 5.5+ (attribute removed) and
vllm 0.11 (still reads it). We need this active in EVERY Python process —
including vllm's spawned EngineCore subprocesses — so we install a .pth file
which runs on every interpreter startup.

Run this script once per pod's venv. It is idempotent.
"""

from __future__ import annotations

import site
import sys
from pathlib import Path

PTH_NAME = "explore_persona_space_tokenizer_patch.pth"
PATCH_MODULE_NAME = "_explore_persona_space_tokenizer_patch"

PATCH_CODE = '''"""Runtime patch: alias PreTrainedTokenizerBase.all_special_tokens_extended.

See scripts/_install_tokenizer_patch.py for why this exists.
"""
try:
    from transformers.tokenization_utils_base import PreTrainedTokenizerBase

    if not hasattr(PreTrainedTokenizerBase, "all_special_tokens_extended"):
        # `all_special_tokens` is a property descriptor; binding it to a new
        # name on the class re-uses the same descriptor, which resolves
        # correctly on instances.
        PreTrainedTokenizerBase.all_special_tokens_extended = (
            PreTrainedTokenizerBase.all_special_tokens
        )
except Exception:
    # Never block interpreter startup on a failed patch.
    pass
'''


def main() -> int:
    # Choose the venv site-packages that the current interpreter is using.
    site_packages_dirs = [Path(p) for p in site.getsitepackages()]
    # Prefer a site-packages path inside a .venv.
    candidates = [p for p in site_packages_dirs if ".venv" in p.parts]
    target_dir = candidates[0] if candidates else site_packages_dirs[0]

    module_path = target_dir / f"{PATCH_MODULE_NAME}.py"
    pth_path = target_dir / PTH_NAME

    module_path.write_text(PATCH_CODE)
    # The .pth file: any line starting with "import " is executed at startup.
    pth_path.write_text(f"import {PATCH_MODULE_NAME}\n")

    print(f"installed patch module: {module_path}")
    print(f"installed .pth file:    {pth_path}")

    # Verify the .pth loads via an independent interpreter invocation.
    import subprocess

    check_code = (
        "from transformers.tokenization_utils_base import PreTrainedTokenizerBase; "
        "print('HAS=' + str(hasattr(PreTrainedTokenizerBase, 'all_special_tokens_extended')))"
    )
    result = subprocess.run(
        [sys.executable, "-c", check_code],
        capture_output=True,
        text=True,
        timeout=90,
    )
    stdout = result.stdout.strip()
    print(f"VERIFY stdout: {stdout}")
    if "HAS=True" in stdout:
        print("VERIFY OK: fresh interpreter sees the alias")
        return 0
    print(f"VERIFY FAILED: stderr={result.stderr.strip()}")
    return 1


if __name__ == "__main__":
    sys.exit(main())
