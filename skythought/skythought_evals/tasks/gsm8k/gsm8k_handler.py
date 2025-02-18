import re
from typing import Any, Dict

from skythought_evals.util.math_parsing_util import extract_answer

from ..base import TaskConfig, TaskHandler


class GSM8KTaskHandler(TaskHandler):
    def __init__(self, task_config: TaskConfig) -> None:
        super().__init__(task_config)
        self.ans_re = re.compile(r"((-?[$0-9.,]{2,})|(-?[0-9]+))")
        self.gt_re = re.compile(r"#### (\-?[0-9\.\,]+)")
        self.invalid_ans = "[invalid]"

    def generate_prompt(self, problem):
        return self.task_config.templating_parameters["template"].format(**problem)

    def check_correctness(self, problem: Dict[str, Any], generation: str) -> bool:
        gt_answer = self.extract_gt_answer(problem[self.task_config.answer_key])
        model_answer = extract_answer(generation)
        model_answer = self.sanitize_answer(model_answer)
        return model_answer == gt_answer

    def update_results(self, problem, response):
        # Initialize the response structure
        response_entry = {
            "content": response,
            "correctness": None,
            "reason": None,
        }
        curr_res = self.check_correctness(problem, generation=response)
        if curr_res:
            response_entry["correctness"] = True
            response_entry["reason"] = ""
        else:
            response_entry["correctness"] = False
            response_entry["reason"] = "Solution is incorrect."

        return response_entry

    def load_and_filter_dataset(
        self, start, end, split=None, subset=None, difficulty=None
    ):
        train_data = self.load_dataset(subset=subset, split=split).to_pandas()
        return train_data.iloc[start:end] if end > 0 else train_data.iloc[start:]

    def extract_gt_answer(self, completion):
        match = self.gt_re.search(completion)
        if match:
            match_str = match.group(1).strip()
            match_str = match_str.replace(",", "")
            return match_str
        else:
            return self.invalid_ans

    def sanitize_answer(self, answer):
        patterns_to_remove = [
            ",",  # Remove commas
            r"\$",  # Remove dollar signs
            r"\.$" r"\*",  # Remove trailing period  # Remove asterisks
        ]
        for pattern in patterns_to_remove:
            answer = re.sub(pattern, "", answer)

        matches = self.ans_re.findall(answer)
        if matches:
            # get the last match (i.e final response) and the first / outer capturing group
            match_str = matches[-1][0].strip()
            return match_str
        else:
            return self.invalid_ans
