"""Fit the mean-over-CoT-tokens -> answer-vector metamodel (cell p7_F) on all questions, sweeping the penalty.

Section 4.5 fits every input state to the answer vector except the mean over reasoning tokens (the production
cells only use that state as a TARGET, cell p7_B). This wrapper adds the missing input: x = mean over CoT tokens,
y = mean over answer tokens, same rows, folds, ridge core, scoring and retrieval recipe as
scripts/issue2546_allfit_necessity.py, which it imports with a no-op mode so the module's argv parse and log-open
run without fitting anything.

No production stratum cell exists for this input/target pair, so instead of production_lambda() the ridge penalty
is swept over the values the production cells selected for the neighboring cells (316 for end of thought -> answer,
1000 for context -> trace mean, 3162 for context -> answer). Every penalty is written as its own cell so the report
can name the selected one and show the sensitivity.

Usage:
  issue2546_allfit_cotmean_cell.py <arm> [lambda ...]        # default lambdas: 316 1000 3162

Outputs (BASE = /mnt/eps-data/thomasjiralerspong/cot_necessity/allfit):
  BASE/results/p7_F_lam<L>__a<arm>.json            subsets all / necessary / both_correct: R^2 (global + dataset
                                                    mean), acc@1, bootstrap CIs, per-dataset reads
  BASE/preds/p7_F_lam<L>__all__a<arm>.npz          out-of-fold predictions
  BASE/preds/hits__p7_F_lam<L>__all__a<arm>.npz    own-answer retrieval hits
"""

import importlib.util
import pathlib
import sys

from dotenv import load_dotenv

load_dotenv()


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    arm = sys.argv[1]
    lams = [float(v) for v in sys.argv[2:]] or [316.0, 1000.0, 3162.0]

    # The fit module parses sys.argv at import: (mode, arm). "cotmean" matches none of its mode branches.
    sys.argv = ["issue2546_allfit_necessity.py", "cotmean", arm]
    spec = importlib.util.spec_from_file_location(
        "issue2546_allfit_necessity",
        pathlib.Path(__file__).with_name("issue2546_allfit_necessity.py"),
    )
    fit = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(fit)

    names = []
    for lam in lams:
        name = f"p7_F_lam{int(round(lam))}"
        fit.CELLS[name] = ("post", "cot_mean", "post", "ans_mean")
        names.append(name)

    def swept_lambda(cell: str) -> float:
        assert cell.startswith("p7_F_lam"), cell
        return float(cell.split("lam", 1)[1])

    fit.production_lambda = swept_lambda
    fit.say(f"p7_F sweep arm{arm}: cells={names}")
    fit.run_fit(names)
    fit.say(f"DONE p7_F sweep arm{arm}")


if __name__ == "__main__":
    main()
