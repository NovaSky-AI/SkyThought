import json
import os
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field


class TaskConfig(BaseModel):
    dataset_name: str
    dataset_source: Optional[str] = None
    question_key: str
    templating_parameters: Dict[str, str] = Field(default_factory=dict)
    fewshot_config: List[Dict[str, Any]] = Field(default_factory=list)
    num_fewshot: int = 0


class TaskHandler:
    def __init__(self, yaml_file_path):
        self.yaml_file_path = yaml_file_path
        self.task_config = TaskConfig(**self.load_yaml(yaml_file_path))

    @staticmethod
    def load_yaml(yaml_file_path):
        with open(yaml_file_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    @staticmethod
    def get_question_key():
        raise NotImplementedError("Subclasses should implement this method.")

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

    def load_and_filter_dataset(
        self, start, end, split="train", source=None, filter_difficulty=False, args=None
    ):
        raise NotImplementedError("Subclasses should implement this method.")

    def process_remaining_data(self, train_data, results):
        raise NotImplementedError("Subclasses should implement this method.")
