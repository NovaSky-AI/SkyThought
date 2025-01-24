import json
import os
import re


def has_code(response):
    pattern = r"```(?:[a-zA-Z]*)\n(.*?)```"
    # Use re.DOTALL to match multiline content inside backticks
    matches = re.findall(pattern, response, re.DOTALL)
    # print(matches)
    return matches


class TaskHandler:
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
