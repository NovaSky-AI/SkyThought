from typing import Any, Dict

from datasets import load_dataset

from skythought.evals.util.common import TimeoutException, timeout
from skythought.evals.util.math_parsing_util import (
    extract_answer,
    math_equal,
)

from ..base import TaskHandler


class NUMINATaskHandler(TaskHandler):

    def generate_prompt(self, problem: Dict[str, Any]):
        prompt = problem["problem"]
        return self.task_config.templating_parameters["template"].format(prompt=prompt)

    @timeout(5)  # Add timeout of 5 seconds
    def check_correctness(self, problem, generation):
        solution = extract_answer(problem[self.task_config.answer_key])
        pred = extract_answer(generation)
        return math_equal(pred, solution)

    def update_results(self, problem, response):
        # Initialize the response structure
        response_entry = {
            "content": response,
            "correctness": None,
            "reason": None,
        }

        try:
            curr_res = self.check_correctness(problem, generation=response)
            if curr_res:
                response_entry["correctness"] = True
                response_entry["reason"] = ""
            else:
                response_entry["correctness"] = False
                response_entry["reason"] = "Solution is incorrect."
        except TimeoutException as e:
            response_entry["correctness"] = False
            response_entry["reason"] = str(e)

        return response_entry

    @staticmethod
    def get_difficulty_dict(subset, start, end):
        diff_dict = {}
        dataset = load_dataset(
            "NovaSky-AI/labeled_numina_difficulty_859K",
            trust_remote_code=True,
            split="train",
        )
        for example in dataset:
            # print(example)
            diff_dict[example["problem"]] = example["gpt_difficulty_parsed"]
        return diff_dict

    def load_and_filter_dataset(
        self, start, end, split=None, subset=None, difficulty=None
    ):
        dataset = self.load_dataset(subset=subset, split=split)

        if "source" in self.task_config.preprocess_config:
            source = self.task_config.preprocess_config["source"]
            dataset = dataset.filter(lambda x: x["source"] == source)

        dataset = dataset.to_pandas()
        # TODO (sumanthrh): this is hacky for numina. the start and end filter should be applied at the very end
        # it is kept here for consistency with the original code.
        dataset = dataset.iloc[start:end] if end > 0 else dataset.iloc[start:]
        dataset = dataset[dataset["solution"].str.contains("boxed", na=False)]

        if "filter_difficulty" in self.task_config.preprocess_config:
            lower_bound = self.task_config.preprocess_config[
                "math_difficulty_lower_bound"
            ]
            upper_bound = self.task_config.preprocess_config[
                "math_difficulty_upper_bound"
            ]
            diff_dict = self.get_difficulty_dict(
                self.task_config.dataset_subset, start, end
            )
            dataset = dataset[
                dataset["problem"]
                .map(diff_dict)
                .apply(lambda x: x >= lower_bound and x <= upper_bound)
            ]

        return dataset
