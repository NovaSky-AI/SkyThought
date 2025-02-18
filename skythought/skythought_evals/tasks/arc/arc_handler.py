import re
from typing import Any, Dict

from skythought_evals.util.math_parsing_util import extract_answer

from ..base import TaskConfig, TaskHandler


class ARCChallengeTaskHandler(TaskHandler):
    def __init__(self, task_config: TaskConfig) -> None:
        super().__init__(task_config)
        self.ans_re = re.compile(r"[Tt]he best answer is ([A-D])[\.\,]*", re.IGNORECASE)
        self.letter_re = re.compile(r"([A-D])[\.\,]*")
        self.canonical_options = ["A", "B", "C", "D"]
        self.invalid_ans = "[invalid]"

    def generate_prompt(self, problem):
        choices = problem["choices"]
        choices_text = "\n".join(
            [
                f"{label}.{choice}"
                for label, choice in zip(self.canonical_options, choices["text"])
            ]
        )
        problem["choices_text"] = choices_text
        full_prompt = self.task_config.templating_parameters["template"].format(
            **problem
        )
        return full_prompt

    def check_correctness(self, problem: Dict[str, Any], generation: str) -> bool:
        gt_answer = problem[self.task_config.answer_key]
        if gt_answer not in self.canonical_options:
            gt_answer = self.canonical_options[
                int(problem[self.task_config.answer_key]) - 1
            ]
        model_answer = self.get_answer(generation)
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

    def get_answer(self, completion):
        # First, we try to extract similar to MATH answers
        answer = extract_answer(completion)
        match = None
        if answer:
            # match for the letter answer needed.
            match = self.letter_re.search(answer)
            if match:
                return match.group(1).strip()

        if not answer or not match:
            # try basic-regex based search
            patterns_to_remove = [
                ",",  # Remove commas
                r"\$",  # Remove dollar signs
                r"\.$" r"\\",  # Remove trailing period  # Remove stray backslashes
                r"\*",  # Remove asterisks
            ]
            answer = completion
            for pattern in patterns_to_remove:
                answer = re.sub(pattern, "", answer)
            matches = self.ans_re.findall(answer)
            if not matches:
                return self.invalid_ans
            return matches[-1].strip()
