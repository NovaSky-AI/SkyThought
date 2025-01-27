import pytest
from skyevals.tasks import MMLUTaskHandler, TaskConfig

inputs = [
    (
        {
            "is_stdin": False,
            "question": "What is the capital of France?",
            "choices": ["Paris", "London", "Berlin", "Madrid"],
            "answer": "0",
        },
        TaskConfig(
            handler="dummy",
            dataset_path="dummy",
            dataset_split="dummy",
            question_key="question",
            answer_key="answer",
            templating_parameters={
                "template": "Return your final response within \\boxed{{}}. {prompt}"
            },
        ),
        MMLUTaskHandler,
        [
            {"role": "system", "content": "Please answer the following question:"},
            {
                "role": "user",
                "content": "Return your final response within \\boxed{}. What is the capital of France?\nAnswer Choices: (A) Paris (B) London (C) Berlin (D) Madrid",  # noqa: E501
            },
        ],
    ),
]


@pytest.mark.parametrize("row,config,handler_cls,expected_conversation", inputs)
def test_make_conversations(row, config, handler_cls, expected_conversation):

    # Expected system prompt
    system_prompt = "Please answer the following question:"

    # Initialize the handler
    handler = handler_cls(config)

    # Expected conversation format
    # expected input
    # Call make_conversations
    conversations = handler.make_conversations([row], system_prompt)
    # Assert the conversation is as expected
    assert conversations == [
        expected_conversation
    ], f"Expected conversation {expected_conversation} but got {conversations}."
