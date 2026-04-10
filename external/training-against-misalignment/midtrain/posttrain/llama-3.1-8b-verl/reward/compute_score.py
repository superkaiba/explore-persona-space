"""veRL-compatible reward function for GSM8K, MATH, and IFEval.

Ports the three verifiers from open-instruct's ground_truth_utils.py:
  - GSM8K:  Extract last number, compare to ground_truth
  - MATH:   Multi-strategy extraction (boxed, minerva, last-$) + sympy/string equiv
  - IFEval: Parse constraint dict with func_name, check via if_functions

veRL interface: compute_score(data_source, solution_str, ground_truth, extra_info) -> float

Reward scaling: score * 10 to match open-instruct's verification_reward=10.
"""

import json
import re
import signal
import sys
from pathlib import Path

import sympy
from sympy.parsing.latex import parse_latex


# =============================================================================
# Math utilities (from open-instruct/open_instruct/math_utils.py)
# =============================================================================

def last_boxed_only_string(string: str) -> str | None:
    idx = string.rfind("\\boxed")
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i = idx
    right_brace_idx = None
    num_left_braces_open = 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return None if right_brace_idx is None else string[idx : right_brace_idx + 1]


def remove_boxed(s: str) -> str:
    if "\\boxed " in s:
        left = "\\boxed "
        assert s[: len(left)] == left
        return s[len(left) :]
    left = "\\boxed{"
    assert s[: len(left)] == left
    assert s[-1] == "}"
    return s[len(left) : -1]


def get_unnormalized_answer(text: str) -> str:
    INVALID_ANSWER = "[invalidanswer]"
    end_seq = "I hope it is correct."
    text += end_seq
    match = re.search(
        r"Final Answer: The final answer is(.*?). I hope it is correct.", text
    )
    if match:
        return match.group(1).strip()
    return INVALID_ANSWER


SUBSTITUTIONS = [
    ("an ", ""),
    ("a ", ""),
    (".$", "$"),
    ("\\$", ""),
    (r"\ ", ""),
    (" ", ""),
    ("mbox", "text"),
    (",\\text{and}", ","),
    ("\\text{and}", ","),
    ("\\text{m}", "\\text{}"),
]
REMOVED_EXPRESSIONS = [
    "square", "ways", "integers", "dollars", "mph", "inches", "ft",
    "hours", "km", "units", "\\ldots", "sue", "points", "feet",
    "minutes", "digits", "cents", "degrees", "cm", "gm", "pounds",
    "meters", "meals", "edges", "students", "childrentickets",
    "multiples", "\\text{s}", "\\text{.}", "\\text{\ns}",
    "\\text{}^2", "\\text{}^3", "\\text{\n}", "\\text{}",
    r"\mathrm{th}", r"^\circ", r"^{\circ}", r"\;", r",\!", "{,}", '"',
    "\\dots",
]


def normalize_final_answer(final_answer: str) -> str:
    final_answer = final_answer.split("=")[-1]
    for before, after in SUBSTITUTIONS:
        final_answer = final_answer.replace(before, after)
    for expr in REMOVED_EXPRESSIONS:
        final_answer = final_answer.replace(expr, "")
    final_answer = re.sub(r"(.*?)(\$)(.*?)(\$)(.*)", "$\\3$", final_answer)
    final_answer = re.sub(r"(\\text\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\textbf\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\overline\{)(.*?)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(\\boxed\{)(.*)(\})", "\\2", final_answer)
    final_answer = re.sub(r"(frac)([^{])(.)", "frac{\\2}{\\3}", final_answer)
    final_answer = re.sub(r"(sqrt)([^{])", "sqrt{\\2}", final_answer)
    final_answer = final_answer.replace("$", "")
    if final_answer.replace(",", "").isdigit():
        final_answer = final_answer.replace(",", "")
    return final_answer


class _Timeout:
    def __init__(self, seconds=1, error_message="Timeout"):
        self.seconds = seconds
        self.error_message = error_message

    def handle_timeout(self, signum, frame):
        raise TimeoutError(self.error_message)

    def __enter__(self):
        signal.signal(signal.SIGALRM, self.handle_timeout)
        signal.alarm(self.seconds)

    def __exit__(self, type, value, traceback):
        signal.alarm(0)


def is_equiv(x1: str, x2: str) -> bool:
    try:
        with _Timeout(seconds=5):
            try:
                parsed_x1 = parse_latex(x1)
                parsed_x2 = parse_latex(x2)
            except (
                sympy.parsing.latex.errors.LaTeXParsingError,
                sympy.SympifyError,
                TypeError,
            ):
                return False
            try:
                diff = parsed_x1 - parsed_x2
            except TypeError:
                return False
            try:
                return sympy.simplify(diff) == 0
            except ValueError:
                pass
    except TimeoutError:
        return False
    except Exception:
        return False
    return False


def fix_fracs(string):
    substrs = string.split("\\frac")
    new_str = substrs[0]
    if len(substrs) > 1:
        substrs = substrs[1:]
        for substr in substrs:
            new_str += "\\frac"
            if substr[0] == "{":
                new_str += substr
            else:
                try:
                    assert len(substr) >= 2
                except AssertionError:
                    return string
                a = substr[0]
                b = substr[1]
                if b != "{":
                    if len(substr) > 2:
                        new_str += "{" + a + "}{" + b + "}" + substr[2:]
                    else:
                        new_str += "{" + a + "}{" + b + "}"
                else:
                    if len(substr) > 2:
                        new_str += "{" + a + "}" + b + substr[2:]
                    else:
                        new_str += "{" + a + "}" + b
    return new_str


def fix_a_slash_b(string):
    if len(string.split("/")) != 2:
        return string
    a = string.split("/")[0]
    b = string.split("/")[1]
    try:
        a = int(a)
        b = int(b)
        assert string == f"{a}/{b}"
        return "\\frac{" + str(a) + "}{" + str(b) + "}"
    except (AssertionError, ValueError):
        return string


def remove_right_units(string):
    if "\\text{ " in string:
        splits = string.split("\\text{ ")
        assert len(splits) == 2
        return splits[0]
    return string


def fix_sqrt(string):
    if "\\sqrt" not in string:
        return string
    splits = string.split("\\sqrt")
    new_string = splits[0]
    for split in splits[1:]:
        if split[0] != "{":
            a = split[0]
            new_substr = "\\sqrt{" + a + "}" + split[1:]
        else:
            new_substr = "\\sqrt" + split
        new_string += new_substr
    return new_string


def strip_string(string):
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = remove_right_units(string)
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    if string[0] == ".":
        string = "0" + string
    if len(string.split("=")) == 2 and len(string.split("=")[0]) <= 2:
        string = string.split("=")[1]
    string = fix_sqrt(string)
    string = string.replace(" ", "")
    string = fix_fracs(string)
    if string == "0.5":
        string = "\\frac{1}{2}"
    string = fix_a_slash_b(string)
    return string


def hendrycks_is_equiv(str1, str2):
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        ss1 = strip_string(str1)
        ss2 = strip_string(str2)
        return ss1 == ss2
    except Exception:
        return str1 == str2


# =============================================================================
# IFEval (uses if_functions from open-instruct via PYTHONPATH)
# =============================================================================

# Lazy-load IF_FUNCTIONS_MAP to avoid import errors if PYTHONPATH not set
_IF_FUNCTIONS_MAP = None


def _get_if_functions_map():
    global _IF_FUNCTIONS_MAP
    if _IF_FUNCTIONS_MAP is None:
        try:
            from open_instruct.if_functions import IF_FUNCTIONS_MAP
            _IF_FUNCTIONS_MAP = IF_FUNCTIONS_MAP
        except ImportError:
            print(
                "WARNING: Could not import IF_FUNCTIONS_MAP from open_instruct.if_functions. "
                "IFEval scoring will return 0. Set PYTHONPATH to include open-instruct dir.",
                file=sys.stderr,
            )
            _IF_FUNCTIONS_MAP = {}
    return _IF_FUNCTIONS_MAP


def _score_ifeval(prediction: str, ground_truth: str) -> float:
    """Score IFEval: parse constraint dict, check each constraint."""
    if_map = _get_if_functions_map()
    if not if_map:
        return 0.0

    try:
        constraint = json.loads(ground_truth)
    except (json.JSONDecodeError, TypeError):
        return 0.0

    if isinstance(constraint, list):
        constraint = constraint[0]
    if isinstance(constraint, str):
        constraint = json.loads(constraint)

    # Remove thinking section if present
    answer = prediction
    if "</think>" in answer:
        answer = answer.split("</think>")[-1]
    answer = answer.replace("<answer>", "").replace("</answer>", "").strip()

    if not answer:
        return 0.0

    func_name = constraint.get("func_name")
    if not func_name:
        return 0.0

    func = if_map.get(func_name)
    if not func:
        return 0.0

    # Build kwargs from constraint, excluding func_name and None values
    kwargs = {k: v for k, v in constraint.items() if k != "func_name" and v is not None}

    try:
        if not kwargs:
            result = func(answer)
        else:
            result = func(answer, **kwargs)
        return float(bool(result))
    except Exception:
        return 0.0


# =============================================================================
# GSM8K verifier
# =============================================================================

def _score_gsm8k(prediction: str, ground_truth: str) -> float:
    """Extract last number from prediction, compare to ground truth."""
    response = re.sub(r"(\d),(\d)", r"\1\2", prediction)
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", response)
    extracted = numbers[-1] if numbers else response
    return float(str(extracted).lower() == str(ground_truth).lower())


# =============================================================================
# MATH verifier
# =============================================================================

def _score_math(prediction: str, ground_truth: str) -> float:
    """Multi-strategy answer extraction + sympy/string equivalence."""
    raw_answer = prediction
    all_answers = []

    # 1. Extract from \boxed{}
    boxed_answer = last_boxed_only_string(raw_answer)
    if boxed_answer is not None:
        try:
            boxed_answer = remove_boxed(boxed_answer)
        except AssertionError:
            boxed_answer = None
    if boxed_answer is not None:
        all_answers.append(boxed_answer)

    # 2. Minerva format
    minerva_answer = normalize_final_answer(get_unnormalized_answer(raw_answer))
    if minerva_answer is not None and minerva_answer != "[invalidanswer]":
        all_answers.append(minerva_answer)

    # 3. Last LaTeX-formatted answer
    if not all_answers:
        dollars = [m.start() for m in re.finditer(r"\$", raw_answer)]
        if len(dollars) > 1:
            answer = normalize_final_answer(
                raw_answer[dollars[-2] + 1 : dollars[-1]]
            )
            all_answers.append(answer)

    # 4. Fallback to full output
    if not all_answers:
        all_answers.append(normalize_final_answer(prediction))
        all_answers.append(prediction)

    # Compare each candidate
    for answer in all_answers:
        if is_equiv(answer, ground_truth) or hendrycks_is_equiv(answer, ground_truth):
            return 1.0
    return 0.0


# =============================================================================
# Main entry point — veRL reward interface
# =============================================================================

REWARD_SCALE = 10  # Match open-instruct's verification_reward=10


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict | None = None,
) -> float:
    """Compute reward score for a single sample.

    Args:
        data_source: "gsm8k", "math", or "ifeval"
        solution_str: Model-generated response text
        ground_truth: Correct answer string (number for math/gsm8k, JSON for ifeval)
        extra_info: Optional metadata (unused, for veRL compatibility)

    Returns:
        float: Reward score (0.0 or REWARD_SCALE)
    """
    ds = data_source.lower()

    if ds == "gsm8k":
        score = _score_gsm8k(solution_str, ground_truth)
    elif ds == "math":
        score = _score_math(solution_str, ground_truth)
    elif ds == "ifeval":
        score = _score_ifeval(solution_str, ground_truth)
    else:
        print(f"WARNING: Unknown data_source '{data_source}', returning 0", file=sys.stderr)
        score = 0.0

    return score * REWARD_SCALE
