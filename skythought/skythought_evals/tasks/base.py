import json
import os
from typing import Any, Dict, List, Optional

import yaml
from datasets import Dataset as HFDataset
from datasets import load_dataset
from pydantic import BaseModel, Field


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
    # Optional, unused for now
    fewshot_config: List[Dict[str, Any]] = Field(default_factory=list)
    num_fewshot: int = 0

    preprocess_config: Dict[str, Any] = Field(default_factory=dict)

    @classmethod
    def from_yaml(cls, yaml_file_path) -> "TaskConfig":
        with open(yaml_file_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)


class TaskHandler:

    def __init__(self, task_config: TaskConfig):
        self.task_config = task_config

    @classmethod
    def from_config_path(cls, config_path: str) -> "TaskHandler":
        task_config = TaskConfig.from_yaml(config_path)
        return cls(task_config)

    @property
    def question_key(self):
        return self.task_config.question_key

    def check_correctness(self, problem, generation):
        raise NotImplementedError("Subclasses should implement this method.")

    def update_results(self, problem, response):
        raise NotImplementedError("Subclasses should implement this method.")

    def make_conversations(self, data, system_prompt, model=None):
        raise NotImplementedError("Subclasses should implement this method.")

    def load_existing_results(self, result_file):
        if not os.path.exists(result_file):
            return {}
        with open(result_file, "r", encoding="utf-8") as f:
            records = json.load(f)
        return records

    def load_dataset(self, subset=None, split=None, **kwargs) -> HFDataset:
        dataset = load_dataset(
            path=self.task_config.dataset_path,
            name=subset if subset else self.task_config.dataset_subset,
            split=split if split else self.task_config.dataset_split,
            **self.task_config.dataset_kwargs
        )
        return dataset

    def load_and_filter_dataset(
        self, start, end, split=None, subset=None, difficulty=None, args=None
    ):
        raise NotImplementedError("Subclasses should implement this method.")

    def process_remaining_data(self, train_data, results):
        raise NotImplementedError("Subclasses should implement this method.")
