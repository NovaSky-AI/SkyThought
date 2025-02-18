import pytest
from skythought_evals.util.cli_util import _parse_multi_args, parse_multi_args


# Tests for _parse_multi_args
@pytest.mark.parametrize(
    "input_str,expected",
    [
        ("{'a': 1, 'b': 2}", {"a": 1, "b": 2}),
        ("{'a': 1, 'c': {'b': 2}}", {"a": 1, "c": {"b": 2}}),
        ('{"a": 1, "b": 2}', {"a": 1, "b": 2}),
        ("a=1,b=2", {"a": 1, "b": 2}),
        ("a=1, b=2", {"a": 1, "b": 2}),
        ("a=1", {"a": 1}),
        ("", {}),
        ("   ", {}),
    ],
)
def test__parse_multi_args_valid(input_str, expected):
    assert _parse_multi_args(input_str) == expected


@pytest.mark.parametrize(
    "input_str",
    [
        "a=1,b",  # Missing value for 'b'
        "a=1,b=two=2",  # Too many '=' signs
        "a=1, b='unclosed",  # Syntax error
        "not a dict",  # Invalid dictionary string
    ],
)
def test__parse_multi_args_invalid(input_str):
    with pytest.raises((SyntaxError, ValueError)):
        _parse_multi_args(input_str)


# Tests for parse_multi_args
@pytest.mark.parametrize(
    "input_str,expected",
    [
        ("{'a': 1, 'b': 2}", {"a": 1, "b": 2}),
        ("a=1,b=2", {"a": 1, "b": 2}),
        ("", {}),
    ],
)
def test_parse_multi_args_valid(input_str, expected):
    assert parse_multi_args(input_str) == expected


@pytest.mark.parametrize(
    "input_str",
    [
        "a=1,b",
        "a=1,b=two=2",
        "a=1, b='unclosed",
        "not a dict",
    ],
)
def test_parse_multi_args_invalid(input_str):
    with pytest.raises(ValueError) as exc_info:
        parse_multi_args(input_str)
    assert (
        f"Excepted comma separated list of parameters arg1=val1,args2=val2 or a dictionary, got invalid argument {input_str}."
        in str(exc_info.value)
    )
