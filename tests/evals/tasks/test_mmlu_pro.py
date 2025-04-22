import pytest

from skythought.evals.tasks.mmlu.mmlu_handler import MMLUProTaskHandler


class MockTaskConfig:
    templating_parameters = {"template": "Question: {prompt}"}
    answer_key = "answer"
    question_key = "question"
    choices_key = "choices"
    context_key = "context"


@pytest.mark.parametrize(
    "problem, response, expected",
    [
        (
            {
                "question": "What is the main function of the left ventricle?",
                "choices": "A) Pumps blood to the lungs\nB) Pumps blood to the body\nC) Collects blood from the body\nD) Stores blood",
                "answer": "B",
                "answer_index": 1,
            },
            "B) Pumps blood to the body",
            True,
        ),
        (
            {
                "question": "What does GDP stand for?",
                "choices": "A) Gross Domestic Product\nB) General Development Plan\nC) Global Distribution Process\nD) Geographic Data Point",
                "answer": "A",
                "answer_index": 0,
            },
            "I think it's B",
            False,
        ),
    ],
)
def test_check_correctness(problem, response, expected):
    handler = MMLUProTaskHandler(task_config=MockTaskConfig)
    assert handler.check_correctness(problem, generation=response) == expected


@pytest.mark.parametrize(
    "problem, expected",
    [
        (
            {
                "question": "What is the main function of the left ventricle?",
                "options": [
                    "Pumps blood to the lungs",
                    "Pumps blood to the body",
                    "Collects blood from the body",
                    "Stores blood",
                ],
                "answer": "B",
                "answer_index": 1,
            },
            "Question: What is the main function of the left ventricle?\n"
            "Answer Choices: (A) Pumps blood to the lungs (B) Pumps blood to the body (C) Collects blood from the body (D) Stores blood",
        ),
    ],
)
def test_generate_prompt(
    problem,
    expected,
):
    handler = MMLUProTaskHandler(task_config=MockTaskConfig)
    assert handler.generate_prompt(problem) == expected
