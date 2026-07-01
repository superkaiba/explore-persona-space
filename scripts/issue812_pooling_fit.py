#!/usr/bin/env python3
"""Issue #812 — pooling-operator fit sweep (name-alias entrypoint).

Thin wrapper that re-exports the fit sweep under the name plan v3 (§4.1 / §10 / §11)
+ the clean-result Repro footer cite: ``scripts/issue812_pooling_fit.py``. The real
implementation lives in ``scripts/issue812_fit_pooling.py`` (both names are preserved:
the round-1/2 code-review record references the ``fit_pooling`` file, so renaming it
would break those references). Running THIS file is identical to running the other —
it imports and calls the same ``main()`` (argparse is re-parsed inside ``main``, so
CLI args pass straight through).
"""

from __future__ import annotations

import sys
from pathlib import Path

_SCRIPTS_DIR = Path(__file__).resolve().parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from issue812_fit_pooling import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
