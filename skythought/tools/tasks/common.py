import json
import os
from typing import Any, Dict, List, Optional

import pandas as pd
import yaml
from datasets import load_dataset
from pydantic import BaseModel, Field


class TaskConfig(BaseModel):
    dataset_name: str
    dataset_source: Optional[str] = None
    dataset_split: str
    dataset_kwargs: Optional[Dict[str, Any]] = None
    question_key: str
    templating_parameters: Dict[str, str] = Field(default_factory=dict)
    # Optional, unused for now
    fewshot_config: List[Dict[str, Any]] = Field(default_factory=list)
    num_fewshot: int = 0


class TaskHandler:
    task_config_cls = TaskConfig

    def __init__(self, yaml_file_path):
        self.yaml_file_path = yaml_file_path
        self.task_config = self.task_config_cls(**self.load_yaml(yaml_file_path))

    @staticmethod
    def load_yaml(yaml_file_path):
        with open(yaml_file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

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

    def load_dataset(self, source=None, split=None, **kwargs) -> pd.DataFrame:
        dataset = load_dataset(
            self.task_config.dataset_name,
            source if source else self.task_config.dataset_source,
            split=split if split else self.task_config.dataset_split,
            **self.task_config.dataset_kwargs
        )
        data = dataset.to_pandas()
        return data

    def load_and_filter_dataset(
        self, start, end, split="train", source=None, filter_difficulty=False, args=None
    ):
        raise NotImplementedError("Subclasses should implement this method.")

    def process_remaining_data(self, train_data, results):
        raise NotImplementedError("Subclasses should implement this method.")
