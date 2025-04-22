import pytest

from skythought.evals.tasks.amc23.amc23_handler import AMC23TaskHandler


class MockTaskConfig:
    templating_parameters = {
        "template": "Return the answer to the following: {question}"
    }
    answer_key = "answer"
    question_key = "question"
    choices_key = "choices"


@pytest.mark.parametrize(
    "problem, response, expected",
    [
        (
            {"question": "2+2", "answer": "4"},
            "5",
            False,
        ),
        (
            {"question": "3* 25 percent", "answer": " 75%"},
            "My reply is $0.75.",  # ignores dollar signs and normalizes percentages
            True,
        ),
    ],
)
def test_check_correctness(problem, response, expected):
    handler = AMC23TaskHandler(task_config=MockTaskConfig)
    print(handler.check_correctness(problem, generation=response))
    assert handler.check_correctness(problem, generation=response) == expected


@pytest.mark.parametrize(
    "problem, expected",
    [
        (
            {"question": "What is the result of 2+2?", "answer": "4"},
            "Return the answer to the following: What is the result of 2+2?",
        ),
    ],
)
def test_generate_prompt(problem, expected):
    handler = AMC23TaskHandler(task_config=MockTaskConfig)
    assert handler.generate_prompt(problem) == expected
