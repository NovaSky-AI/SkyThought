import pytest
from skythought_evals.util.common import has_code


@pytest.mark.parametrize(
    "input_str,expected",
    [
        # Single code block with a language specifier
        (
            "Here is some Python code:\n```python\nprint('Hello, World!')\n```",
            ["print('Hello, World!')\n"],
        ),
        # Single code block without any specifier
        (
            "Here is some code:\n```\nprint('Hello, World!')\n```",
            ["print('Hello, World!')\n"],
        ),
        # Multiple code blocks
        (
            "Here is some Python code:\n```python\nprint('Hello, Try 1!')\n```And here is my final answer:```python\nprint('Hello, Try 2!')\n```",
            ["print('Hello, Try 1!')\n", "print('Hello, Try 2!')\n"],
        ),
        # No code blocks
        ("This is a string without any code blocks.", []),
        # Malformed code block (misssing a closing backtick)
        (
            "Here is a code block that never ends:\n```python\nprint('Hello, World!')``",
            [],
        ),
        # Malformed code block (missing opening backticks)
        ("Here is a code block with missing opening:\nprint('Hello, World!')\n```", []),
    ],
)
def test_has_code(input_str, expected):
    assert has_code(input_str) == expected


def test_has_code_empty_string():
    assert has_code("") == []
