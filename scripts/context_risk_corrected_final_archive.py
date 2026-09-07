"""Archive final owner evidence with the independently reviewed upload helper."""

import os

from explore_persona_space.orchestrate.env import load_dotenv
from scripts.context_risk_corrected_finish import REPO, STAGING, upload

if __name__ == "__main__":
    load_dotenv(str(REPO / ".env"))
    os.environ.update(EPM_HF_FILECOUNT_FALLBACK="0", EPM_HF_RETRY_BUDGET_S="1800")
    upload(STAGING / "final", "final")
