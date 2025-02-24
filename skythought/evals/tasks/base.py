from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from datasets import Dataset as HFDataset
from datasets import load_dataset
from pydantic import BaseModel, Field

ConversationType = List[Dict[str, Any]]


class TaskConfig(BaseModel):
    handler: str
    dataset_path: str
    dataset_subset: Optional[str] = None
    dataset_split: str
    dataset_kwargs: Dict[str, Any] = Field(default_factory=dict)
    question_key: str
    # Optional answer key for datasets with a single correct answer
    answer_key: Optional[str] = None
    templating_parameters: Dict[str, str] = Field(default_factory=dict)
    # Example fields
    # fewshot_config: List[Dict[str, Any]] = Field(default_factory=list)
    # num_fewshot: int = 0

    preprocess_config: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_file_path) -> "TaskConfig":
        with open(yaml_file_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def update(self, **kwargs):
        for key, value in kwargs.items():
            setattr(self, key, value)


class TaskHandler(ABC):

    def __init__(self, task_config: TaskConfig):
        self.task_config = task_config

    @classmethod
    def from_config_path(cls, config_path: str) -> "TaskHandler":
        task_config = TaskConfig.from_yaml(config_path)
        return cls(task_config)

    @property
    def question_key(self):
        return self.task_config.question_key

    @abstractmethod
    def check_correctness(
        self, problem: Dict[str, Any], generation: Dict[str, Any]
    ) -> bool:
        pass

    @abstractmethod
    def update_results(self, problem: Dict[str, Any], response: str) -> Dict[str, Any]:
        pass

    def make_conversations(
        self,
        data: List[Dict[str, Any]],
        system_prompt: Optional[str] = None,
        user_template: Optional[str] = None,
        assistant_prefill: Optional[str] = None,
    ) -> List[ConversationType]:
        conversations = []
        for _, problem in enumerate(data):
            prompt_text = self.generate_prompt(problem)
            conversations.append(
                make_conversation_from_contents(
                    [prompt_text],
                    system_prompt=system_prompt,
                    user_template=user_template,
                    assistant_prefill=assistant_prefill,
                )
            )
        return conversations

    def load_dataset(self, subset=None, split=None, **kwargs) -> HFDataset:
        dataset = load_dataset(
            path=self.task_config.dataset_path,
            name=subset if subset else self.task_config.dataset_subset,
            split=split if split else self.task_config.dataset_split,
            **self.task_config.dataset_kwargs,
        )
        # add an index column efficiently with map
        dataset = dataset.map(add_idx_map, with_indices=True)
        return dataset

    @abstractmethod
    def load_and_filter_dataset(
        self, start, end, split=None, subset=None, difficulty=None
    ) -> pd.DataFrame:
        pass

    def process_remaining_data(self, train_data, id_to_results: dict):
        return [
            row.to_dict()
            for _, row in train_data.iterrows()
            if str(row["_index"]) not in id_to_results
        ]


def add_idx_map(x: dict, idx: int) -> dict:
    # We convert to string for consistency
    x["_index"] = str(idx)
    return x


def make_conversation_from_contents(
    contents: List[str],
    system_prompt: Optional[str] = None,
    user_template: Optional[str] = None,
    assistant_prefill: Optional[str] = None,
) -> ConversationType:
    """Makes a conversation given a list of user/assistant message strings.

    If system_prompt is provided, it will be added as the first message.
    If user_template is provided, it will be used to format the user messages. This is useful for model-specific formatting.

    Args:
        content: A list of user/assistant message strings.
        system_prompt: An optional string for the system prompt.
        user_template: An optional string for the user template.

    Returns:
        A list of dictionaries representing the conversation.
    """

    conversation = []
    if system_prompt:
        conversation.append({"role": "system", "content": system_prompt})

    for i, content in enumerate(contents):
        if i % 2 == 0:
            content = user_template.format(content) if user_template else content
            conversation.append({"role": "user", "content": content})
        else:
            conversation.append({"role": "assistant", "content": content})

    if assistant_prefill and conversation[-1]["role"] == "user":
        conversation.append({"role": "assistant", "content": assistant_prefill})

    return conversation
