import pytest

from skythought.evals.tasks.aime.aime_handler import AIMETaskHandler


class MockTaskConfig:
    templating_parameters = {
        "template": "Problem: {prompt}\n\nProvide a numerical answer."
    }
    answer_key = "answer"
    question_key = "question"


@pytest.mark.parametrize(
    "problem, response, expected",
    [
        (
            {
                "question": "Find the sum of the first 10 positive integers.",
                "answer": "55",
            },
            "The sum is 55",
            True,
        ),
        (
            {
                "question": "What is the value of (3^4 - 2^5)?",
                "answer": "49",
            },
            "48",
            False,
        ),
    ],
)
def test_check_correctness(problem, response, expected):
    handler = AIMETaskHandler(task_config=MockTaskConfig)
    assert handler.check_correctness(problem, generation=response) == expected


@pytest.mark.parametrize(
    "problem, expected",
    [
        (
            {
                "question": "Find the sum of the first 10 positive integers.",
                "answer": "4",
            },
            "Problem: Find the sum of the first 10 positive integers.\n\nProvide a numerical answer.",
        ),
    ],
)
def test_generate_prompt(problem, expected):
    print(problem)
    handler = AIMETaskHandler(task_config=MockTaskConfig)
    assert handler.generate_prompt(problem) == expected
