import pytest

from skythought.evals.tasks.mmlu.mmlu_handler import MMLUTaskHandler


class MockTaskConfig:
    templating_parameters = {"template": "{prompt}"}
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
                "answer": 1,
            },
            "The answer is B) Paris",
            True,
        ),
        (
            {
                "question": "Which element has the atomic number 1?",
                "choices": "A) Helium\nB) Oxygen\nC) Hydrogen\nD) Carbon",
                "answer": 2,
            },
            "A",
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
            {
                "question": "What is the capital of France?",
                "answer": "B",
                "choices": ["London", "Paris", "Berlin", "Madrid"],
            },
            "What is the capital of France?\nAnswer Choices: (A) London (B) Paris (C) Berlin (D) Madrid",
        ),
    ],
)
def test_generate_prompt(problem, expected):
    handler = MMLUTaskHandler(task_config=MockTaskConfig)
    assert handler.generate_prompt(problem) == expected
