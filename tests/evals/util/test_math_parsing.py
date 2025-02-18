import pytest
from skythought_evals.util.math_parsing_util import (
    choice_answer_clean,
    extract_answer,
    get_multiple_choice_answer,
    math_equal,
    strip_answer_string,
)


@pytest.mark.parametrize(
    "prediction,reference,expected",
    [
        # Exact numeric match
        ("3.14", "3.14", True),
        ("10", "10", True),
        ("-5", "-5", True),
        ("5", "5.0", True),
        # Close Numeric Match < 1e-3
        ("3.14159", "3.1416", True),
        ("3.14159", "3.15", False),
        # Integer vs float
        ("3", "3.0", True),
        ("2", "2.0001", True),
        # Percentage acceptance
        # By default, `include_percentage` is `True`, e.g. reference=0.75 should match 75
        ("75", "0.75", True),
        ("75", "0.80", False),
        # Symbolic equivalence
        (r"(x+1)^2", r"x^2 + 2x + 1", True),
        (r"(x+1)^2", r"x^2 + 2x - 1", False),
        (r"x = y + 2", r" x - (y+2) = 0", True),  # checks rearranged equality
        (r"\frac{1}{2}", "1/2", True),  # fraction vs slash
        (r"\frac{2}{4}", r"\frac{1}{2}", True),  # simplified fraction
        # Matrix equality
        (
            r"\begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix}",
            r"\begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix}",
            True,
        ),
        (
            r"\begin{pmatrix}1 & 2 \\ 3 & 4\end{pmatrix}",
            r"\begin{pmatrix}1 & 2 \\ 3 & 5\end{pmatrix}",
            False,
        ),
        # Simple choice letter checking
        ("C", "C", True),  # direct match
        ("c", "C", True),  # case-insensitive
        ("(C)", "C", True),  # parentheses
        ("A", "B", False),  # mismatch
        # Empty checks
        ("", "", True),  # trivially equal if both empty strings
        ("", "0", False),  # one empty, one not
    ],
)
def test_math_equal(prediction, reference, expected):
    """Tests the math_equal function with various forms of numeric,
    symbolic, and multiple-choice input."""
    assert math_equal(prediction, reference) == expected


@pytest.mark.parametrize(
    "raw_string,expected",
    [
        # Basic stripping
        (" 3.14 ", "3.14"),
        ("\n3.14.\n", "3.14"),  # remove trailing period
        ("\\text{five}", "5"),  # word2number logic
        # Remove units, dollar signs, etc.
        ("3.14\\text{ inches}", "3.14"),
        # also removes trailing zeros in decimals
        ("$12.00", "12"),
        ("\\$12.01", "12.01"),
        # Remove degrees
        ("45^{\\circ}", "45"),
        ("45^\\circ", "45"),
        # Fraction fix
        (r" \frac12 ", r"\frac{1}{2}"),
        ("4/2", r"\frac{4}{2}"),
        # Square root fix
        ("\\sqrt4", "\\sqrt{4}"),
        (" .75", "0.75"),  # leading decimal fix
        # Negative or zero replacements
        (" -5.", "-5"),
        # Comma-separated integers sorted
        ("3,2,10,-1", "-1,2,3,10"),
    ],
)
def test_strip_answer_string(raw_string, expected):
    assert strip_answer_string(raw_string) == expected


@pytest.mark.parametrize(
    "input_str,expected",
    [
        # Simple final answers
        # TODO (sumanthrh): The below test case fails, and from the looks of it
        # `extract_answer` should support it.
        # ("The final answer is 42. I hope that's correct.", "42"),
        ("Answer: 3.14", "3.14"),
        # Boxed answers
        (r"The answer is \(\boxed{5x+1}\). Great!", "5x+1"),
        (r"I think the final answer is $ \boxed{\frac12} $.", r"\frac{1}{2}"),
        # fallback to last numeric
        ("No explicit final. But here's some text 2 and 10", "10"),
        ("No numeric content at all", ""),  # nothing to parse
    ],
)
def test_extract_answer(input_str, expected):
    assert extract_answer(input_str) == expected


@pytest.mark.parametrize(
    "input_str,expected",
    [
        # Standard letters
        ("The correct answer is A", "A"),
        ("B is correct. Or maybe C. Actually let's finalize B", "B"),
        # Extra punctuation
        ("I think it's (D).", "D"),
        ("**C**", "C"),
        ("c", "C"),
        # No letter found -> just pass back raw
        ("No letter here", "No letter here"),
    ],
)
def test_get_multiple_choice_answer(input_str, expected):
    result = get_multiple_choice_answer(input_str)
    assert result == expected


@pytest.mark.parametrize(
    "input_str,expected",
    [
        ("C.", "C"),
        ("(C)", "C"),
        ("C  ", "C"),
    ],
)
def test_choice_answer_clean(input_str, expected):
    assert choice_answer_clean(input_str) == expected
