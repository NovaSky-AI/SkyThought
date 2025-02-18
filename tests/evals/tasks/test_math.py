import pytest
from skythought_evals.tasks.math.math_handler import MathTaskHandler


class MockTaskConfig:
    templating_parameters = {"template": "{question}"}
    answer_key = "answer"
    question_key = "question"


# TODO (sumanthrh): Add hard examples here for the correctness function. This is simple demonstrative,
@pytest.mark.parametrize(
    "problem, response, expected",
    [
        (
            {"question": "2+2", "answer": "4"},
            "4",
            True,
        ),
        (
            {"question": "2+2", "answer": "4"},
            "5",
            False,
        ),
        (
            {"question": "3* 25 percent", "answer": " 75%"},
            "My reply is 0.75.",
            True,
        ),
        (
            {"question": "Solve $2+$2", "answer": "4."},
            "The answer is $4.",
            True,
        ),
    ],
)
def test_check_correctness(
    problem,
    response,
    expected,
):
    handler = MathTaskHandler(task_config=MockTaskConfig)
    assert handler.check_correctness(problem, generation=response) == expected
