import pytest

from skythought.evals.tasks.mmlu.mmlu_handler import MMLUTaskHandler


class MockTaskConfig:
    templating_parameters = {"template": "{question}\n\nChoices:\n{choices}"}
    answer_key = "answer"
    question_key = "question"
    choices_key = "choices"


@pytest.mark.parametrize(
    "problem, response, expected",
    [
        (
            {
                "question": "What is the capital of France?",
                "choices": "A) London\nB) Paris\nC) Berlin\nD) Madrid",
                "answer": "B",
            },
            "The answer is B) Paris",
            True,
        ),
        (
            {
                "question": "Which element has the atomic number 1?",
                "choices": "A) Helium\nB) Oxygen\nC) Hydrogen\nD) Carbon",
                "answer": "C",
            },
            "C",
            False,
        ),
    ],
)
def test_check_correctness(problem, response, expected):
    handler = MMLUTaskHandler(task_config=MockTaskConfig)
    assert handler.check_correctness(problem, generation=response) == expected


@pytest.mark.parametrize(
    "problem, expected",
    [
        (
            {"question": "What is the capital of France?", "answer": "B"},
            "What is the capital of France?\n\nChoices:\nA) London\nB) Paris\nC) Berlin\nD) Madrid",
        ),
    ],
)
def test_generate_prompt(problem, expected):
    handler = MMLUTaskHandler(task_config=MockTaskConfig)
    assert handler.generate_prompt(problem) == expected
